from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from tempfile import TemporaryDirectory
import zipfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.training.corpus import DEFAULT_LIMITS, _load_archive, load_ontology_labels

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_ner_hardcases_1_17.py"
spec = importlib.util.spec_from_file_location("build_ner_hardcases_1_17", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load the NER 1.17 corpus builder")
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


class Hardcases117BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels = load_ontology_labels(ROOT / "data" / "corpora" / "ru-pii" / "v1" / "ontology.json")
        cls.base_sources, cls.base_records = builder._read_base_sources(
            builder.BASE_TRAIN, builder.BASE_VALIDATION, cls.labels
        )

    def test_builds_deterministic_importable_archive_with_all_17_labels(self) -> None:
        cases = list(builder.build_cases(self.labels))
        first, manifest, _, _, _ = builder.archive_bytes(
            cases, self.labels, self.base_sources["train"], self.base_sources["validation"]
        )
        second, _, _, _, _ = builder.archive_bytes(
            cases, self.labels, self.base_sources["train"], self.base_sources["validation"]
        )
        self.assertEqual(first, second)
        self.assertTrue(builder.ACTIVE_LABELS.issubset(set(manifest["labels"])))
        release, records = _load_archive(first, self.labels, DEFAULT_LIMITS)
        self.assertEqual(release.corpus_version, "1.17.0")
        self.assertEqual(set(records), {"train", "validation", "test"})
        self.assertEqual(release.record_counts, manifest["record_counts"])
        self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())

    def test_combined_splits_preserve_approved_source_bytes_and_keep_authored_rows_separate(self) -> None:
        cases = list(builder.build_cases(self.labels))
        _, _, combined, authored, authored_cases = builder.archive_bytes(
            cases, self.labels, self.base_sources["train"], self.base_sources["validation"]
        )
        self.assertTrue(combined["train"].startswith(self.base_sources["train"]))
        self.assertTrue(combined["validation"].startswith(self.base_sources["validation"]))
        self.assertEqual(combined["train"][len(self.base_sources["train"]):].lstrip(b"\n"), authored["train"])
        self.assertEqual(combined["validation"][len(self.base_sources["validation"]):].lstrip(b"\n"), authored["validation"])
        metadata = builder.source_metadata(authored_cases, {
            split: hashlib.sha256(raw).hexdigest() for split, raw in self.base_sources.items()
        })
        self.assertEqual(len(metadata["authored_records"]["train"]), len(authored_cases["train"]))
        self.assertFalse(set(metadata["authored_group_assignments"]["train"]) & set(metadata["authored_group_assignments"]["test"]))
        self.assertEqual(len(self.base_records["train"]), 8_384)
        self.assertEqual(len(self.base_records["validation"]), 1_141)

    def test_rejects_modified_or_mislabeled_base_source_copy(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "1.16.0"
            root.mkdir()
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_bytes(self.base_sources["train"] + b" ")
            validation.write_bytes(self.base_sources["validation"])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                builder._read_base_sources(train, validation, self.labels)
            wrong = root / "test.jsonl"
            wrong.write_bytes(self.base_sources["train"])
            with self.assertRaisesRegex(ValueError, "Only copied 1.16.0 train"):
                builder._read_base_sources(wrong, validation, self.labels)

    def test_split_sources_and_templates_are_disjoint_and_negative_rows_are_empty(self) -> None:
        cases = builder.build_cases(self.labels)
        self.assertEqual(len({case.template for case in cases}), len(cases))
        self.assertEqual(len({case.text for case in cases}), len(cases))
        for split in builder.SPLITS:
            selected = [case for case in cases if case.split == split]
            self.assertTrue(any(case.label is None and not case.entities for case in selected))
            self.assertEqual(
                {entity["label"] for case in selected for entity in case.entities},
                builder.ACTIVE_LABELS,
            )
        for left in builder.SPLITS:
            for right in builder.SPLITS:
                if left >= right:
                    continue
                self.assertFalse(
                    {case.group for case in cases if case.split == left}
                    & {case.group for case in cases if case.split == right}
                )
                self.assertFalse(
                    {case.template for case in cases if case.split == left}
                    & {case.template for case in cases if case.split == right}
                )

    def test_authored_scenarios_preserve_multi_spans_and_test_narratives(self) -> None:
        source = builder.authored_scenarios.build_records()
        cases = builder.build_cases(self.labels)
        source_by_template = {
            row["template"]: row
            for rows in source.values()
            for row in rows
        }
        authored = [case for case in cases if case.template in source_by_template]
        by_template = {case.template: case for case in authored}
        self.assertEqual(set(by_template), set(source_by_template))
        self.assertTrue(any(len(case.entities) > 1 for case in authored))
        for template, row in source_by_template.items():
            case = by_template[template]
            self.assertEqual(case.group, row["group"])
            self.assertEqual(case.text, row["text"])
            self.assertEqual(
                [(entity["start"], entity["end"], entity["label"]) for entity in case.entities],
                [(entity["start"], entity["end"], entity["label"]) for entity in row["entities"]],
            )
        self.assertEqual(
            {case.text for case in authored if case.split == "test"},
            {row["text"] for row in source["test"]},
        )

    def test_challenge_stays_separate_and_sparse_support_is_not_evaluation_ready(self) -> None:
        records = builder.authored_scenarios.build_records()
        cases = list(builder.build_cases(self.labels))
        archive, manifest, split_bytes, authored_by_split, _ = builder.archive_bytes(
            cases, self.labels, self.base_sources["train"], self.base_sources["validation"]
        )
        challenge_raw, challenge_cases = builder._authored_bytes(cases, "challenge")
        challenge_meta = builder.challenge_metadata(cases, challenge_raw, records)
        report = builder.quality_report(
            cases, archive, manifest,
            split_bytes, authored_by_split,
            {split: hashlib.sha256(raw).hexdigest() for split, raw in self.base_sources.items()},
            self.labels,
        )
        with zipfile.ZipFile(io.BytesIO(archive)) as packaged:
            self.assertEqual(set(packaged.namelist()), {"manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"})
        self.assertEqual(len(challenge_cases), len(records["challenge"]))
        self.assertEqual(challenge_meta["record_count"], len(records["challenge"]))
        self.assertFalse(challenge_meta["evaluation_ready"])
        self.assertFalse(report["evaluation_ready"])
        self.assertEqual(set(report["target_labels"]), builder.authored_scenarios.PRIORITY_LABELS)
        self.assertTrue(all(count < 30 for count in challenge_meta["target_support"].values() if count))

    def test_authored_build_is_reproducible_and_rejects_shifted_offsets(self) -> None:
        first = list(builder.build_cases(self.labels))
        second = list(builder.build_cases(self.labels))
        self.assertEqual(first, second)
        first_archive = builder.archive_bytes(
            first, self.labels, self.base_sources["train"], self.base_sources["validation"]
        )[0]
        second_archive = builder.archive_bytes(
            second, self.labels, self.base_sources["train"], self.base_sources["validation"]
        )[0]
        self.assertEqual(first_archive, second_archive)
        index = next(i for i, case in enumerate(first) if case.split == "test" and len(case.entities) > 1)
        case = first[index]
        shifted = tuple({**entity, "start": entity["start"] + 1} if i == 0 else entity for i, entity in enumerate(case.entities))
        first[index] = replace(case, entities=shifted)
        with self.assertRaisesRegex(ValueError, "Entity span does not match its expected surface"):
            builder.validate_cases(first, self.labels)

    def test_rejects_group_or_template_crossing_splits(self) -> None:
        cases = list(builder.build_cases(self.labels))
        cases[-1] = replace(cases[-1], group=cases[0].group)
        with self.assertRaisesRegex(ValueError, "source group crosses"):
            builder.validate_cases(cases, self.labels)
        cases = list(builder.build_cases(self.labels))
        cases[-1] = replace(cases[-1], template=cases[0].template)
        with self.assertRaisesRegex(ValueError, "template crosses"):
            builder.validate_cases(cases, self.labels)

    def test_rejects_invalid_identifier_checksum_and_misaligned_span(self) -> None:
        cases = list(builder.build_cases(self.labels))
        index = next(i for i, case in enumerate(cases) if case.split == "train" and case.label == "SNILS")
        case = cases[index]
        entity = case.entities[0]
        changed = case.text[:entity["end"] - 1] + ("0" if case.text[entity["end"] - 1] != "0" else "1") + case.text[entity["end"]:]
        cases[index] = replace(case, text=changed, expected_surfaces=(changed[entity["start"]:entity["end"]],))
        with self.assertRaisesRegex(ValueError, "Invalid SNILS checksum"):
            builder.validate_cases(cases, self.labels)

        cases = list(builder.build_cases(self.labels))
        case = next(row for row in cases if row.label == "PER")
        corrupted_span = ({**case.entities[0], "end": case.entities[0]["end"] - 1},)
        cases[cases.index(case)] = replace(case, entities=corrupted_span)
        with self.assertRaisesRegex(ValueError, "checksum|span|surface"):
            builder.validate_cases(cases, self.labels)

    def test_negative_record_cannot_hide_a_known_entity_surface(self) -> None:
        cases = list(builder.build_cases(self.labels))
        negative_index = next(i for i, case in enumerate(cases) if case.split == "train" and case.label is None)
        negative = cases[negative_index]
        disease = next(case.expected_surfaces[0] for case in cases if case.split == "train" and case.label == "DISEASE")
        self.assertIsNotNone(disease)
        cases[negative_index] = replace(negative, text=f"{negative.text}: {disease}")
        with self.assertRaisesRegex(ValueError, "Negative record contains"):
            builder.validate_cases(cases, self.labels)

    def test_ocr_edit_shifts_unicode_offsets_and_refuses_entity_overlap(self) -> None:
        text = "Ёлка: Мария Соколова; отделение № 2"
        start = text.index("Мария")
        entity = ({"start": start, "end": start + len("Мария Соколова"), "label": "PER"},)
        edited, shifted = builder.ocr_substitute(text, entity, 0, len("Ёлка"), "Елка́")
        expected_start = start + len("Елка́") - len("Ёлка")
        self.assertEqual(edited[shifted[0]["start"]:shifted[0]["end"]], "Мария Соколова")
        self.assertEqual(shifted[0]["start"], expected_start)
        with self.assertRaisesRegex(ValueError, "overlaps an annotated span"):
            builder.ocr_substitute(text, entity, start, start + 2, "XX")
        after_span, unchanged = builder.ocr_substitute(text, entity, entity[0]["end"], entity[0]["end"], "X")
        self.assertEqual(after_span[unchanged[0]["start"]:unchanged[0]["end"]], "Мария Соколова")

    def test_importer_limit_is_enforced_before_archive_creation(self) -> None:
        cases = list(builder.build_cases(self.labels))
        case = next(row for row in cases if row.label is None)
        oversized = replace(case, text="а" * (builder.MAX_TEXT_CHARS + 1))
        cases[cases.index(case)] = oversized
        with self.assertRaisesRegex(ValueError, "exceeds the importer limit"):
            builder.validate_cases(cases, self.labels)

    def test_private_output_writer_is_repeatable_but_refuses_replacement(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            builder._write_identical_or_new(path, b"first")
            builder._write_identical_or_new(path, b"first")
            self.assertEqual(path.read_bytes(), b"first")
            with self.assertRaises(FileExistsError):
                builder._write_identical_or_new(path, b"changed")


if __name__ == "__main__":
    unittest.main()
