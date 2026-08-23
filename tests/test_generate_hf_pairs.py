import unittest

from generate_hf_pairs import (
    LABELS,
    Record,
    generate_balanced_pairs,
    split_records_by_doi,
    validate_pairs,
)


def record(row, doi, references=(), caption="galaxy spectrum wavelength survey"):
    return Record(
        row_index=row,
        uuid=f"uuid-{row}",
        image_id=f"image-{row}",
        doi=doi,
        caption=f"{caption} object-{row % 3}",
        references=frozenset(references),
        citations=frozenset(),
    )


class GeneratePairsTest(unittest.TestCase):
    def setUp(self):
        self.records = []
        edges = {"a": ("b",), "c": ("d",), "e": ("f",)}
        for paper_index, doi in enumerate("abcdef"):
            for offset in range(3):
                self.records.append(
                    record(
                        paper_index * 3 + offset,
                        doi,
                        edges.get(doi, ()),
                        caption=f"shared-topic-{paper_index // 2} telescope spectrum",
                    )
                )

    def test_balanced_pairs_obey_relationships(self):
        pairs = generate_balanced_pairs(
            self.records,
            pairs_per_class=4,
            seed=7,
            hard_negative_fraction=0.5,
        )
        validate_pairs(pairs, self.records)
        counts = {label: 0 for label in LABELS}
        for pair in pairs:
            counts[pair.relationship] += 1
        self.assertEqual(counts, {label: 4 for label in LABELS})

    def test_doi_split_is_disjoint_and_deterministic(self):
        train_a, validation_a, assignment_a = split_records_by_doi(
            self.records, validation_fraction=1 / 3, seed=11
        )
        train_b, validation_b, assignment_b = split_records_by_doi(
            self.records, validation_fraction=1 / 3, seed=11
        )
        self.assertEqual(assignment_a, assignment_b)
        self.assertEqual(
            [row.row_index for row in train_a], [row.row_index for row in train_b]
        )
        self.assertEqual(
            [row.row_index for row in validation_a],
            [row.row_index for row in validation_b],
        )
        self.assertTrue({row.doi for row in train_a}.isdisjoint({row.doi for row in validation_a}))


if __name__ == "__main__":
    unittest.main()
