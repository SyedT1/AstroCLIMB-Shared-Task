"""Match AstroCLIMB Kaggle objects to Hugging Face metadata.

The matcher builds compact caption and image-hash indexes from an AstroCLIMB
Parquet file, then streams a Kaggle pair CSV in small chunks.  It never writes
the Base64 objects to its output.  Exact metadata matches are used to infer the
four competition relationships from figure identity, DOI equality, and the
citation graph.

Example:
  python metadata_matching.py \
      --pairs data/train_1000.csv \
      --hf-dataset adsabs/AstroCLIMB \
      --metadata-only \
      --output results/train_metadata_matches.csv
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import pickle
import re
import struct
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image, ImageOps


LABELS = (
  "same_figure",
  "same_paper",
  "related_papers",
  "unrelated_papers",
)


def normalize_caption(value: Any) -> str:
  """Normalize formatting without removing scientifically meaningful text."""
  if value is None:
      return ""
  text = unicodedata.normalize("NFKC", str(value))
  return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_doi(value: Any) -> str:
  if value is None:
      return ""
  doi = str(value).strip().casefold()
  doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
  return doi.rstrip(".,; ")


def doi_set(value: Any) -> frozenset[str]:
  """Convert Parquet list, numpy array, or JSON-like DOI value to a set."""
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


def sha256_bytes(raw: bytes) -> str:
  return hashlib.sha256(raw).hexdigest()


def caption_key(value: Any) -> str:
  """Use a fixed-size key instead of retaining full captions in the index."""
  return sha256_bytes(normalize_caption(value).encode("utf-8"))


def pixel_sha256(raw: bytes) -> str:
  """Hash decoded pixels so equivalent PNG encodings can still match."""
  with Image.open(io.BytesIO(raw)) as image:
      image = ImageOps.exif_transpose(image).convert("RGB")
      size = struct.pack(">II", image.width, image.height)
      return sha256_bytes(size + image.tobytes())


def is_base64_image(value: Any) -> bool:
  if not isinstance(value, str):
      return False
  stripped = value.lstrip()
  return stripped.startswith("iVBOR") or stripped.startswith("data:image/")


def decode_base64_image(value: str) -> bytes:
  payload = value.strip()
  if payload.startswith("data:image/"):
      payload = payload.split(",", 1)[1]
  return base64.b64decode(payload, validate=False)


def parquet_image_bytes(value: Any, base_dir: Path) -> bytes | None:
  """Read common Hugging Face Image representations returned by Parquet."""
  if value is None:
      return None
  if isinstance(value, bytes):
      return value
  if isinstance(value, (bytearray, memoryview)):
      return bytes(value)
  if isinstance(value, Image.Image):
      buffer = io.BytesIO()
      value.save(buffer, format="PNG")
      return buffer.getvalue()
  if isinstance(value, dict):
      raw = value.get("bytes")
      if raw is not None:
          return bytes(raw)
      path = value.get("path")
      if path:
          image_path = Path(path)
          if not image_path.is_absolute():
              image_path = base_dir / image_path
          return image_path.read_bytes()
  return None


def collect_pair_image_hashes(
  pairs_path: Path,
  chunk_size: int,
  limit: int | None,
  include_pixel_hashes: bool,
) -> tuple[set[str], set[str]]:
  """Collect compact hashes for only the images that occur in the pair CSV."""
  raw_hashes: set[str] = set()
  pixel_hashes: set[str] = set()
  processed = 0
  for chunk in pd.read_csv(
      pairs_path,
      usecols=["obj_1", "obj_2"],
      chunksize=chunk_size,
      keep_default_na=False,
  ):
      if limit is not None:
          remaining = limit - processed
          if remaining <= 0:
              break
          chunk = chunk.iloc[:remaining]
      for column in ("obj_1", "obj_2"):
          for value in chunk[column]:
              if not is_base64_image(value):
                  continue
              try:
                  raw = decode_base64_image(value)
              except Exception:
                  continue
              raw_hashes.add(sha256_bytes(raw))
              if include_pixel_hashes:
                  try:
                      pixel_hashes.add(pixel_sha256(raw))
                  except Exception:
                      pass
      processed += len(chunk)
  print(
      f"Pair image targets: {len(raw_hashes)} raw hashes"
      + (f", {len(pixel_hashes)} pixel hashes" if include_pixel_hashes else "")
  )
  return raw_hashes, pixel_hashes


def append_index(index: dict[str, list[int]], key: str, row_id: int) -> None:
  if key and row_id not in index[key]:
      index[key].append(row_id)


@dataclass(frozen=True)
class Match:
  method: str
  candidates: tuple[int, ...]

  @property
  def unique_row(self) -> int | None:
      return self.candidates[0] if len(self.candidates) == 1 else None


class MetadataIndex:
  def __init__(
      self,
      metadata_path: Path,
      include_images: bool = True,
      include_pixel_hashes: bool = False,
      parquet_batch_size: int = 256,
  ) -> None:
      try:
          import pyarrow.dataset as pyarrow_dataset
      except ImportError as exc:
          raise SystemExit(
              "Reading Parquet requires pyarrow. Install it with: pip install pyarrow"
          ) from exc

      dataset = pyarrow_dataset.dataset(str(metadata_path), format="parquet")
      required = {"image", "Image Caption", "Paper DOI", "Image ID"}
      missing = required.difference(dataset.schema.names)
      if missing:
          raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

      self.records: list[dict[str, Any]] = []
      self.caption_index: dict[str, list[int]] = defaultdict(list)
      self.raw_image_index: dict[str, list[int]] = defaultdict(list)
      self.pixel_image_index: dict[str, list[int]] = defaultdict(list)
      self.includes_images = include_images
      self.includes_pixel_hashes = include_images and include_pixel_hashes

      metadata_columns = [
          column
          for column in (
              "UUID",
              "Image ID",
              "Paper DOI",
              "Image Caption",
              "References DOIs",
              "Citing DOIs",
          )
          if column in dataset.schema.names
      ]
      scanner = dataset.scanner(
          columns=metadata_columns,
          batch_size=max(parquet_batch_size, 256),
          use_threads=False,
      )
      uuid_to_internal: dict[str, int] = {}
      for batch in scanner.to_batches():
          values = batch.to_pydict()
          for offset in range(batch.num_rows):
              def get(column: str, default: Any = "") -> Any:
                  return values[column][offset] if column in values else default

              caption = str(get("Image Caption") or "")
              record = {
                  "meta_row": len(self.records),
                  "uuid": str(get("UUID") or ""),
                  "image_id": str(get("Image ID") or ""),
                  "doi": normalize_doi(get("Paper DOI")),
                  "references": doi_set(get("References DOIs", [])),
                  "citations": doi_set(get("Citing DOIs", [])),
              }
              internal_id = len(self.records)
              self.records.append(record)
              if record["uuid"]:
                  uuid_to_internal[record["uuid"]] = internal_id
              append_index(
                  self.caption_index,
                  caption_key(caption),
                  internal_id,
              )
          if len(self.records) % 10_000 < batch.num_rows:
              print(f"Indexed {len(self.records)} metadata rows")

      if include_images:
          print(
              "Building the image hash index. This streams the image column once; "
              "for the full dataset it reads approximately 72 GB."
          )
          base_dir = metadata_path if metadata_path.is_dir() else metadata_path.parent
          image_columns = ["image"]
          if "UUID" in dataset.schema.names:
              image_columns.insert(0, "UUID")
          image_scanner = dataset.scanner(
              columns=image_columns,
              batch_size=min(max(parquet_batch_size, 1), 32),
              use_threads=False,
          )
          sequential_row = 0
          for batch in image_scanner.to_batches():
              values = batch.to_pydict()
              for offset in range(batch.num_rows):
                  uuid = str(values.get("UUID", [""] * batch.num_rows)[offset] or "")
                  internal_id = uuid_to_internal.get(uuid, sequential_row)
                  sequential_row += 1
                  raw = parquet_image_bytes(values["image"][offset], base_dir)
                  if not raw:
                      continue
                  append_index(self.raw_image_index, sha256_bytes(raw), internal_id)
                  if include_pixel_hashes:
                      try:
                          append_index(self.pixel_image_index, pixel_sha256(raw), internal_id)
                      except Exception as exc:
                          print(f"Warning: metadata image {internal_id} could not be decoded: {exc}")
              if sequential_row % 1_000 < batch.num_rows:
                  print(f"Hashed {sequential_row} metadata images")

      print(
          "Metadata index:",
          {
              "rows": len(self.records),
              "captions": len(self.caption_index),
              "raw_images": len(self.raw_image_index),
              "pixel_images": len(self.pixel_image_index),
          },
      )

  def lookup(self, value: Any) -> Match:
      if is_base64_image(value):
          try:
              raw = decode_base64_image(value)
          except Exception:
              return Match("invalid_base64_image", ())

          candidates = tuple(self.raw_image_index.get(sha256_bytes(raw), ()))
          if candidates:
              method = "image_raw_exact" if len(candidates) == 1 else "image_raw_ambiguous"
              return Match(method, candidates)

          if not self.pixel_image_index:
              return Match("image_unmatched", ())
          try:
              candidates = tuple(self.pixel_image_index.get(pixel_sha256(raw), ()))
          except Exception:
              return Match("invalid_decoded_image", ())
          method = "image_pixel_exact" if len(candidates) == 1 else (
              "image_pixel_ambiguous" if candidates else "image_unmatched"
          )
          return Match(method, candidates)

      candidates = tuple(self.caption_index.get(caption_key(value), ()))
      method = "caption_exact" if len(candidates) == 1 else (
          "caption_ambiguous" if candidates else "caption_unmatched"
      )
      return Match(method, candidates)

  def record(self, match: Match) -> dict[str, Any] | None:
      row_id = match.unique_row
      return self.records[row_id] if row_id is not None else None


class HuggingFaceMetadataIndex(MetadataIndex):
  """Stream a Hugging Face dataset without downloading it in full first."""

  def __init__(
      self,
      dataset_name: str,
      split: str = "train",
      include_images: bool = False,
      include_pixel_hashes: bool = False,
  ) -> None:
      try:
          from datasets import Image as HuggingFaceImage
          from datasets import load_dataset
      except ImportError as exc:
          raise SystemExit(
              "Hugging Face streaming requires datasets. Install it with: "
              "pip install datasets"
          ) from exc

      self.records: list[dict[str, Any]] = []
      self.caption_index: dict[str, list[int]] = defaultdict(list)
      self.raw_image_index: dict[str, list[int]] = defaultdict(list)
      self.pixel_image_index: dict[str, list[int]] = defaultdict(list)
      self.includes_images = include_images
      self.includes_pixel_hashes = include_images and include_pixel_hashes

      print(f"Streaming metadata from {dataset_name!r}, split {split!r}")
      stream = load_dataset(dataset_name, split=split, streaming=True)
      required = {"image", "Image Caption", "Paper DOI", "Image ID"}
      missing = required.difference(stream.column_names)
      if missing:
          raise ValueError(f"Hugging Face dataset is missing columns: {sorted(missing)}")

      metadata_columns = [
          column
          for column in (
              "UUID",
              "Image ID",
              "Paper DOI",
              "Image Caption",
              "References DOIs",
              "Citing DOIs",
          )
          if column in stream.column_names
      ]
      uuid_to_internal: dict[str, int] = {}
      for row in stream.select_columns(metadata_columns):
          caption = str(row.get("Image Caption") or "")
          record = {
              "meta_row": len(self.records),
              "uuid": str(row.get("UUID") or ""),
              "image_id": str(row.get("Image ID") or ""),
              "doi": normalize_doi(row.get("Paper DOI")),
              "references": doi_set(row.get("References DOIs", [])),
              "citations": doi_set(row.get("Citing DOIs", [])),
          }
          internal_id = len(self.records)
          self.records.append(record)
          if record["uuid"]:
              uuid_to_internal[record["uuid"]] = internal_id
          append_index(self.caption_index, caption_key(caption), internal_id)
          if len(self.records) % 10_000 == 0:
              print(f"Indexed {len(self.records)} Hugging Face metadata rows")

      if include_images:
          print(
              "Streaming the full Hugging Face image column once. "
              "This transfers approximately 72 GB."
          )
          image_stream = load_dataset(dataset_name, split=split, streaming=True)
          image_stream = image_stream.cast_column(
              "image", HuggingFaceImage(decode=False)
          )
          image_columns = ["image"]
          if "UUID" in image_stream.column_names:
              image_columns.insert(0, "UUID")
          sequential_row = 0
          for row in image_stream.select_columns(image_columns):
              uuid = str(row.get("UUID") or "")
              internal_id = uuid_to_internal.get(uuid, sequential_row)
              sequential_row += 1
              raw = parquet_image_bytes(row.get("image"), Path("."))
              if not raw:
                  continue
              append_index(self.raw_image_index, sha256_bytes(raw), internal_id)
              if include_pixel_hashes:
                  try:
                      append_index(
                          self.pixel_image_index,
                          pixel_sha256(raw),
                          internal_id,
                      )
                  except Exception as exc:
                      print(
                          f"Warning: Hugging Face image {internal_id} "
                          f"could not be decoded: {exc}"
                      )
              if sequential_row % 1_000 == 0:
                  print(f"Hashed {sequential_row} Hugging Face images")

      print(
          "Metadata index:",
          {
              "rows": len(self.records),
              "captions": len(self.caption_index),
              "raw_images": len(self.raw_image_index),
              "pixel_images": len(self.pixel_image_index),
          },
      )


def load_or_build_metadata_index(
  metadata_path: Path | None,
  hf_dataset: str | None,
  hf_split: str,
  cache_path: Path | None,
  include_images: bool,
  include_pixel_hashes: bool,
  parquet_batch_size: int,
  rebuild: bool,
) -> MetadataIndex:
  if cache_path is not None and cache_path.exists() and not rebuild:
      print(f"Loading cached metadata index from {cache_path}")
      with cache_path.open("rb") as handle:
          cached = pickle.load(handle)
      cache_is_sufficient = not include_images or (
          getattr(cached, "includes_images", False)
          and (
              not include_pixel_hashes
              or getattr(cached, "includes_pixel_hashes", False)
          )
      )
      if cache_is_sufficient:
          return cached
      print("Cached index lacks requested image hashes; rebuilding it")

  if metadata_path is not None:
      index = MetadataIndex(
          metadata_path,
          include_images=include_images,
          include_pixel_hashes=include_pixel_hashes,
          parquet_batch_size=parquet_batch_size,
      )
  elif hf_dataset:
      index = HuggingFaceMetadataIndex(
          dataset_name=hf_dataset,
          split=hf_split,
          include_images=include_images,
          include_pixel_hashes=include_pixel_hashes,
      )
  else:
      raise ValueError("Provide either --metadata or --hf-dataset")
  if cache_path is not None:
      cache_path.parent.mkdir(parents=True, exist_ok=True)
      temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
      with temporary.open("wb") as handle:
          pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
      os.replace(temporary, cache_path)
      print(f"Saved reusable metadata index to {cache_path}")
  return index


def add_target_image_hashes(
  index: MetadataIndex,
  metadata_path: Path | None,
  hf_dataset: str | None,
  hf_split: str,
  target_raw_hashes: set[str],
  target_pixel_hashes: set[str],
  image_batch_size: int,
) -> None:
  """Stream metadata images in batches, retaining only requested pair hashes."""
  completed_raw = set(getattr(index, "targeted_raw_hashes_scanned", set()))
  completed_pixels = set(getattr(index, "targeted_pixel_hashes_scanned", set()))
  requested_raw = target_raw_hashes.difference(completed_raw)
  requested_pixels = target_pixel_hashes.difference(completed_pixels)
  missing_raw = set(requested_raw)
  missing_pixels = set(requested_pixels)
  if not requested_raw and not requested_pixels:
      print("All requested pair-image hashes are already cached")
      return

  uuid_to_internal = {
      record["uuid"]: row_id
      for row_id, record in enumerate(index.records)
      if record["uuid"]
  }
  scanned = 0
  matched_raw = 0
  matched_pixels = 0

  def consume(values: dict[str, list[Any]], base_dir: Path) -> bool:
      nonlocal scanned, matched_raw, matched_pixels
      images = values.get("image", [])
      uuids = values.get("UUID", [""] * len(images))
      for offset, value in enumerate(images):
          internal_id = uuid_to_internal.get(str(uuids[offset] or ""), scanned)
          scanned += 1
          raw = parquet_image_bytes(value, base_dir)
          if not raw:
              continue
          raw_hash = sha256_bytes(raw)
          if raw_hash in requested_raw:
              append_index(index.raw_image_index, raw_hash, internal_id)
              if raw_hash in missing_raw:
                  missing_raw.discard(raw_hash)
                  matched_raw += 1
          if requested_pixels:
              try:
                  pixel_hash = pixel_sha256(raw)
              except Exception:
                  pixel_hash = ""
              if pixel_hash in requested_pixels:
                  append_index(index.pixel_image_index, pixel_hash, internal_id)
                  if pixel_hash in missing_pixels:
                      missing_pixels.discard(pixel_hash)
                      matched_pixels += 1
      if scanned % 1_000 < len(images):
          print(
              f"Scanned {scanned} metadata images; matched "
              f"{matched_raw} raw and {matched_pixels} pixel targets"
          )
      # Scan the entire metadata split so duplicate hashes remain ambiguous
      # instead of being incorrectly reported as unique matches.
      return False

  batch_size = max(1, image_batch_size)
  if metadata_path is not None:
      try:
          import pyarrow.dataset as pyarrow_dataset
      except ImportError as exc:
          raise SystemExit(
              "Reading Parquet requires pyarrow. Install it with: pip install pyarrow"
          ) from exc
      dataset = pyarrow_dataset.dataset(str(metadata_path), format="parquet")
      columns = ["image"]
      if "UUID" in dataset.schema.names:
          columns.insert(0, "UUID")
      scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=False)
      base_dir = metadata_path if metadata_path.is_dir() else metadata_path.parent
      for batch in scanner.to_batches():
          if consume(batch.to_pydict(), base_dir):
              break
  elif hf_dataset:
      try:
          from datasets import Image as HuggingFaceImage
          from datasets import load_dataset
      except ImportError as exc:
          raise SystemExit(
              "Hugging Face streaming requires datasets. Install it with: "
              "pip install datasets"
          ) from exc
      print(
          f"Batch-matching pair images from {hf_dataset!r}, split {hf_split!r}; "
          "only matching hashes will be retained"
      )
      stream = load_dataset(hf_dataset, split=hf_split, streaming=True)
      stream = stream.cast_column("image", HuggingFaceImage(decode=False))
      columns = ["image"]
      if "UUID" in stream.column_names:
          columns.insert(0, "UUID")
      for batch in stream.select_columns(columns).iter(batch_size=batch_size):
          if consume(batch, Path(".")):
              break
  else:
      raise ValueError("Provide either --metadata or --hf-dataset")

  index.targeted_raw_hashes_scanned = completed_raw.union(requested_raw)
  index.targeted_pixel_hashes_scanned = completed_pixels.union(requested_pixels)

  print(
      "Targeted image pass:",
      {
          "metadata_images_scanned": scanned,
          "raw_targets_matched": matched_raw,
          "pixel_targets_matched": matched_pixels,
          "raw_targets_unmatched": len(missing_raw),
          "pixel_targets_unmatched": len(missing_pixels),
      },
  )


def infer_relationship(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
  if left is None or right is None:
      return "unmatched"
  if left["meta_row"] == right["meta_row"]:
      return "same_figure"
  left_doi, right_doi = left["doi"], right["doi"]
  if left_doi and left_doi == right_doi:
      return "same_paper"
  if left_doi and right_doi and (
      right_doi in left["references"]
      or right_doi in left["citations"]
      or left_doi in right["references"]
      or left_doi in right["citations"]
  ):
      return "related_papers"
  return "unrelated_papers"


def true_relationship(row: pd.Series) -> str:
  active = [label for label in LABELS if str(row.get(label, "0")) in {"1", "1.0"}]
  return active[0] if len(active) == 1 else "unknown"


def match_output_fields(prefix: str, match: Match, record: dict[str, Any] | None) -> dict[str, Any]:
  return {
      f"{prefix}_match_method": match.method,
      f"{prefix}_candidate_count": len(match.candidates),
      f"{prefix}_meta_row": "" if record is None else record["meta_row"],
      f"{prefix}_uuid": "" if record is None else record["uuid"],
      f"{prefix}_image_id": "" if record is None else record["image_id"],
      f"{prefix}_doi": "" if record is None else record["doi"],
  }


def bootstrap_same_figure_image(
  index: MetadataIndex,
  image_value: Any,
  image_match: Match,
  caption_record: dict[str, Any] | None,
) -> Match:
  """Learn an image-to-record mapping from a labeled same-figure train pair."""
  if image_match.unique_row is not None or caption_record is None:
      return image_match
  if not is_base64_image(image_value):
      return image_match
  try:
      raw_hash = sha256_bytes(decode_base64_image(image_value))
  except Exception:
      return image_match
  append_index(index.raw_image_index, raw_hash, caption_record["meta_row"])
  return index.lookup(image_value)


def match_pairs(
  metadata_path: Path | None,
  hf_dataset: str | None,
  hf_split: str,
  pairs_path: Path,
  output_path: Path,
  chunk_size: int,
  limit: int | None,
  index_cache: Path | None,
  include_images: bool,
  include_pixel_hashes: bool,
  targeted_image_matching: bool,
  image_batch_size: int,
  parquet_batch_size: int,
  rebuild_index: bool,
) -> None:
  index = load_or_build_metadata_index(
      metadata_path=metadata_path,
      hf_dataset=hf_dataset,
      hf_split=hf_split,
      cache_path=index_cache,
      include_images=include_images,
      include_pixel_hashes=include_pixel_hashes,
      parquet_batch_size=parquet_batch_size,
      rebuild=rebuild_index,
  )
  header = pd.read_csv(pairs_path, nrows=0).columns.tolist()
  required = {"id", "obj_1", "obj_2"}
  missing = required.difference(header)
  if missing:
      raise ValueError(f"Pair CSV is missing columns: {sorted(missing)}")

  if targeted_image_matching:
      target_raw_hashes, target_pixel_hashes = collect_pair_image_hashes(
          pairs_path=pairs_path,
          chunk_size=chunk_size,
          limit=limit,
          include_pixel_hashes=include_pixel_hashes,
      )
      add_target_image_hashes(
          index=index,
          metadata_path=metadata_path,
          hf_dataset=hf_dataset,
          hf_split=hf_split,
          target_raw_hashes=target_raw_hashes,
          target_pixel_hashes=target_pixel_hashes,
          image_batch_size=image_batch_size,
      )

  usecols = ["id", "obj_1", "obj_2"] + [label for label in LABELS if label in header]
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(output_path.suffix + ".tmp")
  if temporary.exists():
      temporary.unlink()

  processed = 0
  first_output = True
  for chunk in pd.read_csv(
      pairs_path,
      usecols=usecols,
      chunksize=chunk_size,
      keep_default_na=False,
  ):
      if limit is not None:
          remaining = limit - processed
          if remaining <= 0:
              break
          chunk = chunk.iloc[:remaining]

      output_rows = []
      for _, row in chunk.iterrows():
          left_match = index.lookup(row["obj_1"])
          right_match = index.lookup(row["obj_2"])
          left_record = index.record(left_match)
          right_record = index.record(right_match)
          truth = true_relationship(row)
          if truth == "same_figure":
              if is_base64_image(row["obj_1"]) and not is_base64_image(row["obj_2"]):
                  left_match = bootstrap_same_figure_image(
                      index, row["obj_1"], left_match, right_record
                  )
                  left_record = index.record(left_match)
              elif is_base64_image(row["obj_2"]) and not is_base64_image(row["obj_1"]):
                  right_match = bootstrap_same_figure_image(
                      index, row["obj_2"], right_match, left_record
                  )
                  right_record = index.record(right_match)
          inferred = infer_relationship(left_record, right_record)
          result = {
              "id": row["id"],
              "obj_1_modality": "image" if is_base64_image(row["obj_1"]) else "text",
              "obj_2_modality": "image" if is_base64_image(row["obj_2"]) else "text",
              "inferred_relationship": inferred,
              "true_relationship": truth,
              "metadata_correct": "" if truth == "unknown" or inferred == "unmatched" else inferred == truth,
          }
          result.update(match_output_fields("obj_1", left_match, left_record))
          result.update(match_output_fields("obj_2", right_match, right_record))
          output_rows.append(result)

      pd.DataFrame(output_rows).to_csv(
          temporary,
          mode="w" if first_output else "a",
          header=first_output,
          index=False,
      )
      first_output = False
      processed += len(chunk)
      print(f"Processed {processed} pairs")

  os.replace(temporary, output_path)
  print(f"Saved compact matches to {output_path}")

  if index_cache is not None:
      index_cache.parent.mkdir(parents=True, exist_ok=True)
      cache_temporary = index_cache.with_suffix(index_cache.suffix + ".tmp")
      with cache_temporary.open("wb") as handle:
          pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
      os.replace(cache_temporary, index_cache)
      print(f"Updated metadata index cache at {index_cache}")

  audit = pd.read_csv(output_path, keep_default_na=False)
  both_matched = (
      audit["obj_1_candidate_count"].eq(1)
      & audit["obj_2_candidate_count"].eq(1)
  )
  inferred = audit["inferred_relationship"].ne("unmatched")
  known_truth = audit["true_relationship"].ne("unknown")
  summary = {
      "pairs": len(audit),
      "both_objects_uniquely_matched": int(both_matched.sum()),
      "both_objects_match_rate": round(float(both_matched.mean()), 4),
      "relationship_inference_coverage": round(float(inferred.mean()), 4),
  }
  scored = inferred & known_truth
  if scored.any():
      summary["accuracy_when_inferred"] = round(
          float((audit.loc[scored, "inferred_relationship"] == audit.loc[scored, "true_relationship"]).mean()),
          4,
      )
  print("Summary:", summary)
  methods = Counter(audit["obj_1_match_method"]) + Counter(audit["obj_2_match_method"])
  print("Object match methods:", dict(methods))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--metadata",
      type=Path,
      default=None,
      help="Optional local Parquet file/directory; overrides --hf-dataset",
  )
  parser.add_argument(
      "--hf-dataset",
      default="adsabs/AstroCLIMB",
      help="Hugging Face dataset streamed when --metadata is not provided",
  )
  parser.add_argument("--hf-split", default="train")
  parser.add_argument(
      "--pairs",
      type=Path,
      default=None,
      help="Explicit pair CSV path; otherwise locate --competition-file under /kaggle/input",
  )
  parser.add_argument(
      "--competition-file",
      choices=("train.csv", "test.csv"),
      default="train.csv",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=None,
  )
  parser.add_argument("--chunk-size", type=int, default=32)
  parser.add_argument("--parquet-batch-size", type=int, default=256)
  parser.add_argument(
      "--image-batch-size",
      type=int,
      default=32,
      help="Number of Hugging Face/Parquet images handled at a time",
  )
  parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit")
  parser.add_argument(
      "--index-cache",
      type=Path,
      default=None,
      help="Persistent metadata/hash index; reused by later train and test runs",
  )
  parser.add_argument(
      "--metadata-only",
      action="store_true",
      help="Skip the default targeted, batch-by-batch image matching pass",
  )
  parser.add_argument(
      "--full-image-index",
      action="store_true",
      help="Retain hashes for every metadata image instead of only pair-image targets",
  )
  parser.add_argument(
      "--pixel-hashes",
      action="store_true",
      help="Also decode all 94k metadata images for format-independent pixel hashes",
  )
  parser.add_argument(
      "--rebuild-index",
      action="store_true",
      help="Ignore and replace an existing index cache",
  )
  return parser.parse_args()


def find_competition_file(filename: str) -> Path:
  search_roots = [Path("/kaggle/input"), Path("data")]
  matches: list[Path] = []
  for root in search_roots:
      if root.exists():
          matches.extend(path for path in root.rglob(filename) if path.is_file())
  if not matches:
      raise FileNotFoundError(
          f"Could not find {filename!r}. Attach the AstroCLIMB competition "
          "data to the Kaggle notebook or pass --pairs explicitly."
      )
  matches.sort(key=lambda path: ("competitions" not in path.parts, len(path.parts), str(path)))
  print(f"Using competition file: {matches[0]}")
  if len(matches) > 1:
      print("Other matches:", [str(path) for path in matches[1:]])
  return matches[0]


if __name__ == "__main__":
  arguments = parse_args()
  pairs_path = arguments.pairs or find_competition_file(arguments.competition_file)
  working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results")
  output_path = arguments.output or working / f"{pairs_path.stem}_metadata_matches.csv"
  index_cache = arguments.index_cache or working / "astroclimb_metadata_index.pkl"
  include_images = arguments.full_image_index and not arguments.metadata_only
  targeted_image_matching = not arguments.metadata_only and not include_images
  match_pairs(
      metadata_path=arguments.metadata,
      hf_dataset=arguments.hf_dataset,
      hf_split=arguments.hf_split,
      pairs_path=pairs_path,
      output_path=output_path,
      chunk_size=arguments.chunk_size,
      limit=arguments.limit,
      index_cache=index_cache,
      include_images=include_images,
      include_pixel_hashes=arguments.pixel_hashes,
      targeted_image_matching=targeted_image_matching,
      image_batch_size=arguments.image_batch_size,
      parquet_batch_size=arguments.parquet_batch_size,
      rebuild_index=arguments.rebuild_index,
  )
