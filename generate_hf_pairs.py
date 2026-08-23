"""Generate balanced image-caption relationship pairs from AstroCLIMB metadata.

The generator reads only metadata columns from the Hugging Face Parquet dataset;
the 72 GB image column is never loaded.  Outputs are compact manifests containing
row indices and stable UUIDs that a later embedding job can resolve back to the
source records.

Example (Kaggle):
  python generate_hf_pairs.py \
      --metadata /kaggle/input/astroclimb/AstroCLIMB.parquet \
      --output-dir /kaggle/working/hf_pairs \
      --train-pairs-per-class 50000 \
      --validation-pairs-per-class 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


LABELS = (
    "same_figure",
    "same_paper",
    "related_papers",
    "unrelated_papers",
)

METADATA_COLUMNS = (
    "UUID",
    "Image ID",
    "Paper DOI",
    "Image Caption",
    "References DOIs",
    "Citing DOIs",
)

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+_.-]{2,}")
STOPWORDS = frozenset(
    {
        "about", "after", "also", "are", "between", "bottom", "caption",
        "color", "data", "different", "figure", "figures", "from", "left",
        "lines", "model", "observed", "our", "panel", "panels", "paper",
        "results", "right", "same", "shown", "shows", "the", "their", "this",
        "top", "using", "values", "where", "which", "with",
    }
)


@dataclass(frozen=True)
class Record:
    row_index: int
    uuid: str
    image_id: str
    doi: str
    caption: str
    references: frozenset[str]
    citations: frozenset[str]


@dataclass(frozen=True)
class Pair:
    image_row: int
    caption_row: int
    relationship: str
    sampling: str
    shared_token: str = ""


def normalize_doi(value: Any) -> str:
    if value is None:
        return ""
    doi = str(value).strip().casefold()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.rstrip(".,; ")


def doi_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return frozenset()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                value = [stripped]
        else:
            value = [stripped]
    try:
        values: Iterable[Any] = list(value)
    except TypeError:
        values = [value]
    return frozenset(filter(None, (normalize_doi(item) for item in values)))


def caption_tokens(caption: str) -> frozenset[str]:
    text = unicodedata.normalize("NFKC", caption).casefold()
    return frozenset(
        token for token in TOKEN_PATTERN.findall(text)
        if token not in STOPWORDS and not token.isdigit()
    )


def _record_from_mapping(row_index: int, row: dict[str, Any]) -> Record:
    return Record(
        row_index=row_index,
        uuid=str(row.get("UUID") or ""),
        image_id=str(row.get("Image ID") or ""),
        doi=normalize_doi(row.get("Paper DOI")),
        caption=str(row.get("Image Caption") or ""),
        references=doi_set(row.get("References DOIs")),
        citations=doi_set(row.get("Citing DOIs")),
    )


def iter_parquet_records(path: Path, batch_size: int) -> Iterator[Record]:
    try:
        import pyarrow.dataset as pyarrow_dataset
    except ImportError as exc:
        raise SystemExit(
            "Parquet input requires pyarrow. Install it with: pip install pyarrow"
        ) from exc

    dataset = pyarrow_dataset.dataset(str(path), format="parquet")
    required = {"UUID", "Image ID", "Paper DOI", "Image Caption"}
    missing = required.difference(dataset.schema.names)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    columns = [name for name in METADATA_COLUMNS if name in dataset.schema.names]
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    row_index = 0
    for batch in scanner.to_batches():
        values = batch.to_pylist()
        for row in values:
            yield _record_from_mapping(row_index, row)
            row_index += 1


def iter_csv_records(path: Path, batch_size: int) -> Iterator[Record]:
    """CSV support is mainly useful for tests and compact metadata exports."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("CSV input requires pandas. Install it with: pip install pandas") from exc

    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"UUID", "Image ID", "Paper DOI", "Image Caption"}
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    columns = [name for name in METADATA_COLUMNS if name in header]
    row_index = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=batch_size, keep_default_na=False):
        for row in chunk.to_dict(orient="records"):
            yield _record_from_mapping(row_index, row)
            row_index += 1


def load_records(path: Path, batch_size: int = 4096) -> list[Record]:
    iterator = (
        iter_csv_records(path, batch_size)
        if path.suffix.casefold() in {".csv", ".tsv"}
        else iter_parquet_records(path, batch_size)
    )
    records = [record for record in iterator if record.doi]
    if not records:
        raise ValueError("No records with a non-empty Paper DOI were found")
    return records


def paper_rows(records: Sequence[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.doi].append(record)
    return dict(grouped)


def citation_adjacency(records: Sequence[Record]) -> dict[str, set[str]]:
    known_dois = {record.doi for record in records}
    adjacency: dict[str, set[str]] = {doi: set() for doi in known_dois}
    for record in records:
        for neighbor in record.references | record.citations:
            if neighbor in known_dois and neighbor != record.doi:
                adjacency[record.doi].add(neighbor)
                adjacency[neighbor].add(record.doi)
    return adjacency


def split_records_by_doi(
    records: Sequence[Record], validation_fraction: float, seed: int
) -> tuple[list[Record], list[Record], dict[str, str]]:
    dois = sorted({record.doi for record in records})
    if len(dois) < 2:
        raise ValueError("At least two distinct paper DOIs are required")
    rng = random.Random(seed)
    rng.shuffle(dois)
    validation_count = min(
        len(dois) - 1,
        max(1, round(len(dois) * validation_fraction)),
    )
    validation_dois = set(dois[:validation_count])
    assignment = {
        doi: "validation" if doi in validation_dois else "train" for doi in dois
    }
    train = [record for record in records if assignment[record.doi] == "train"]
    validation = [record for record in records if assignment[record.doi] == "validation"]
    return train, validation, assignment


def _sample_unique(
    count: int,
    draw: Any,
    capacity: int,
    label: str,
) -> list[Pair]:
    if count > capacity:
        raise ValueError(
            f"Requested {count:,} {label} pairs, but the theoretical unique-pair "
            f"capacity is only {capacity:,}. Lower the per-class count."
        )
    selected: dict[tuple[int, int], Pair] = {}
    max_attempts = max(10_000, count * 100)
    attempts = 0
    while len(selected) < count and attempts < max_attempts:
        pair = draw()
        attempts += 1
        if pair is not None:
            selected.setdefault((pair.image_row, pair.caption_row), pair)
    if len(selected) != count:
        raise RuntimeError(
            f"Could sample only {len(selected):,}/{count:,} unique {label} pairs after "
            f"{attempts:,} attempts. Lower the requested count or change the split seed."
        )
    return list(selected.values())


def sample_same_figure(records: Sequence[Record], count: int, rng: random.Random) -> list[Pair]:
    chosen = rng.sample(list(records), count)
    return [Pair(row.row_index, row.row_index, "same_figure", "exact_alignment") for row in chosen]


def sample_same_paper(records: Sequence[Record], count: int, rng: random.Random) -> list[Pair]:
    grouped = {doi: rows for doi, rows in paper_rows(records).items() if len(rows) >= 2}
    dois = list(grouped)
    capacity = sum(len(rows) * (len(rows) - 1) for rows in grouped.values())
    if not dois:
        raise ValueError("This split has no papers containing at least two figures")

    def draw() -> Pair:
        rows = grouped[rng.choice(dois)]
        image, caption = rng.sample(rows, 2)
        return Pair(image.row_index, caption.row_index, "same_paper", "same_doi")

    return _sample_unique(count, draw, capacity, "same_paper")


def sample_related(records: Sequence[Record], count: int, rng: random.Random) -> list[Pair]:
    grouped = paper_rows(records)
    adjacency = citation_adjacency(records)
    edges = sorted(
        (left, right)
        for left, neighbors in adjacency.items()
        for right in neighbors
        if left < right
    )
    capacity = sum(2 * len(grouped[left]) * len(grouped[right]) for left, right in edges)
    if not edges:
        raise ValueError(
            "This DOI split contains no internal citation edges. Try another split seed or "
            "reduce the validation fraction."
        )

    def draw() -> Pair:
        left, right = rng.choice(edges)
        if rng.random() < 0.5:
            left, right = right, left
        image = rng.choice(grouped[left])
        caption = rng.choice(grouped[right])
        return Pair(image.row_index, caption.row_index, "related_papers", "citation_edge")

    return _sample_unique(count, draw, capacity, "related_papers")


def sample_unrelated(
    records: Sequence[Record],
    count: int,
    rng: random.Random,
    hard_fraction: float,
) -> list[Pair]:
    rows = list(records)
    adjacency = citation_adjacency(records)
    tokens_by_row = {row.row_index: caption_tokens(row.caption) for row in rows}
    token_rows: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        for token in tokens_by_row[row.row_index]:
            token_rows[token].append(row)
    # Very common terms make weak negatives and large candidate lists.
    useful_tokens = {
        token for token, candidates in token_rows.items()
        if 2 <= len(candidates) <= max(50, len(rows) // 50)
    }

    def is_unrelated(left: Record, right: Record) -> bool:
        return left.doi != right.doi and right.doi not in adjacency[left.doi]

    selected: dict[tuple[int, int], Pair] = {}
    hard_target = round(count * hard_fraction)
    hard_attempts = 0
    max_hard_attempts = max(20_000, hard_target * 200)
    while len(selected) < hard_target and hard_attempts < max_hard_attempts:
        image = rng.choice(rows)
        tokens = sorted(tokens_by_row[image.row_index] & useful_tokens)
        hard_attempts += 1
        if not tokens:
            continue
        token = rng.choice(tokens)
        caption = rng.choice(token_rows[token])
        if is_unrelated(image, caption):
            pair = Pair(
                image.row_index,
                caption.row_index,
                "unrelated_papers",
                "shared_caption_token",
                token,
            )
            selected.setdefault((pair.image_row, pair.caption_row), pair)

    random_attempts = 0
    max_random_attempts = max(20_000, count * 100)
    while len(selected) < count and random_attempts < max_random_attempts:
        image, caption = rng.choice(rows), rng.choice(rows)
        random_attempts += 1
        if is_unrelated(image, caption):
            pair = Pair(
                image.row_index,
                caption.row_index,
                "unrelated_papers",
                "random_nonedge",
            )
            selected.setdefault((pair.image_row, pair.caption_row), pair)
    if len(selected) != count:
        raise RuntimeError(
            f"Could sample only {len(selected):,}/{count:,} unrelated pairs. "
            "Lower the per-class count."
        )
    return list(selected.values())


def generate_balanced_pairs(
    records: Sequence[Record],
    pairs_per_class: int,
    seed: int,
    hard_negative_fraction: float,
) -> list[Pair]:
    if pairs_per_class <= 0:
        raise ValueError("pairs_per_class must be positive")
    if pairs_per_class > len(records):
        raise ValueError(
            f"Requested {pairs_per_class:,} same_figure pairs from only "
            f"{len(records):,} eligible records"
        )
    rng = random.Random(seed)
    pairs = []
    pairs.extend(sample_same_figure(records, pairs_per_class, rng))
    pairs.extend(sample_same_paper(records, pairs_per_class, rng))
    pairs.extend(sample_related(records, pairs_per_class, rng))
    pairs.extend(
        sample_unrelated(records, pairs_per_class, rng, hard_negative_fraction)
    )
    rng.shuffle(pairs)
    return pairs


def validate_pairs(pairs: Sequence[Pair], records: Sequence[Record]) -> None:
    by_row = {record.row_index: record for record in records}
    adjacency = citation_adjacency(records)
    counts = Counter(pair.relationship for pair in pairs)
    if len(set(counts.values())) != 1 or set(counts) != set(LABELS):
        raise AssertionError(f"Pair classes are not exactly balanced: {dict(counts)}")
    if len({(pair.image_row, pair.caption_row) for pair in pairs}) != len(pairs):
        raise AssertionError("Duplicate image-caption pairs were generated")
    for pair in pairs:
        left, right = by_row[pair.image_row], by_row[pair.caption_row]
        if pair.relationship == "same_figure" and left.row_index != right.row_index:
            raise AssertionError("Invalid same_figure pair")
        if pair.relationship == "same_paper" and (
            left.doi != right.doi or left.row_index == right.row_index
        ):
            raise AssertionError("Invalid same_paper pair")
        if pair.relationship == "related_papers" and right.doi not in adjacency[left.doi]:
            raise AssertionError("Invalid related_papers pair")
        if pair.relationship == "unrelated_papers" and (
            left.doi == right.doi or right.doi in adjacency[left.doi]
        ):
            raise AssertionError("Invalid unrelated_papers pair")


def write_pairs(
    path: Path,
    split: str,
    pairs: Sequence[Pair],
    all_records: Sequence[Record],
) -> None:
    by_row = {record.row_index: record for record in all_records}
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "pair_id", "split", "image_row", "caption_row", "image_uuid",
        "caption_uuid", "image_id", "caption_image_id", "image_doi",
        "caption_doi", "relationship", "sampling", "shared_token",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, pair in enumerate(pairs):
            image, caption = by_row[pair.image_row], by_row[pair.caption_row]
            writer.writerow(
                {
                    "pair_id": f"{split}_{index:08d}",
                    "split": split,
                    "image_row": pair.image_row,
                    "caption_row": pair.caption_row,
                    "image_uuid": image.uuid,
                    "caption_uuid": caption.uuid,
                    "image_id": image.image_id,
                    "caption_image_id": caption.image_id,
                    "image_doi": image.doi,
                    "caption_doi": caption.doi,
                    "relationship": pair.relationship,
                    "sampling": pair.sampling,
                    "shared_token": pair.shared_token,
                }
            )


def write_doi_split(path: Path, assignment: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("paper_doi", "split"))
        writer.writerows(sorted(assignment.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Parquet file/directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pairs-per-class", type=int, default=50_000)
    parser.add_argument("--validation-pairs-per-class", type=int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 <= args.hard_negative_fraction <= 1:
        raise ValueError("hard_negative_fraction must be between 0 and 1")

    print("Reading metadata columns only; the image column will not be loaded")
    records = load_records(args.metadata, args.batch_size)
    print(f"Loaded {len(records):,} records from {len({r.doi for r in records}):,} papers")
    train_records, validation_records, assignment = split_records_by_doi(
        records, args.validation_fraction, args.seed
    )
    print(
        f"DOI-disjoint split: {len(train_records):,} train records, "
        f"{len(validation_records):,} validation records"
    )

    train_pairs = generate_balanced_pairs(
        train_records,
        args.train_pairs_per_class,
        args.seed + 1,
        args.hard_negative_fraction,
    )
    validation_pairs = generate_balanced_pairs(
        validation_records,
        args.validation_pairs_per_class,
        args.seed + 2,
        args.hard_negative_fraction,
    )
    validate_pairs(train_pairs, train_records)
    validate_pairs(validation_pairs, validation_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs(args.output_dir / "hf_train_pairs.csv", "train", train_pairs, records)
    write_pairs(
        args.output_dir / "hf_validation_pairs.csv",
        "validation",
        validation_pairs,
        records,
    )
    write_doi_split(args.output_dir / "hf_doi_split.csv", assignment)
    summary = {
        "metadata": str(args.metadata),
        "seed": args.seed,
        "eligible_records": len(records),
        "paper_dois": len(assignment),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "train_class_counts": dict(Counter(pair.relationship for pair in train_pairs)),
        "validation_class_counts": dict(Counter(pair.relationship for pair in validation_pairs)),
        "train_sampling_counts": dict(Counter(pair.sampling for pair in train_pairs)),
        "validation_sampling_counts": dict(Counter(pair.sampling for pair in validation_pairs)),
    }
    (args.output_dir / "hf_pair_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
