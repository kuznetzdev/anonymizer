from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ner_1_17" / "authored_scenarios.py"
DISEASE_SURFACES = (
    "болезнь Паркинсона", "болезнь Бехтерева", "болезни Бехтерева", "синдром Гийена—Барре",
    "синдром Шегрена", "синдром Марфана", "болезнь Гентингтона", "синдрома Лайелла",
    "синдрома Рейно", "болезни Крона",
)
TRAIN_FORMAT_MARKERS = (
    "Приёмное отделение, 12.03.2026", "Продольная запись наблюдения",
    "Домашний патронаж, маршрут для медсестры", "Координация выписки, административная часть",
    "Сверка регистрационной формы", "Журнал обслуживания оборудования",
)
SPEC = importlib.util.spec_from_file_location("ner_1_17_authored_scenarios", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load locally authored NER 1.17 scenarios")
scenarios = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scenarios
SPEC.loader.exec_module(scenarios)


class AuthoredScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = scenarios.build_records()

    def test_each_split_covers_all_seventeen_classes_with_exact_unicode_offsets(self) -> None:
        for split, rows in self.records.items():
            labels = {entity["label"] for row in rows for entity in row["entities"]}
            self.assertEqual(labels, scenarios.ACTIVE_LABELS, split)
            for row in rows:
                for entity in row["entities"]:
                    self.assertTrue(row["text"][entity["start"]:entity["end"]].strip())
                    self.assertLessEqual(entity["end"], len(row["text"]))

    def test_multiline_notes_keep_multispan_annotations(self) -> None:
        train_long = next(row for row in self.records["train"] if "longitudinal" in row["group"])
        challenge_long = next(row for row in self.records["challenge"] if "review-board" in row["group"])
        self.assertGreater(len(train_long["text"]), 1_000)
        self.assertGreater(len(challenge_long["text"]), 1_500)
        self.assertIn("\n", train_long["text"])
        self.assertIn("\n", challenge_long["text"])
        self.assertGreater(len(train_long["entities"]), 10)
        self.assertGreater(len(challenge_long["entities"]), 10)

    def test_address_chain_and_institution_org_distinctions_are_explicit(self) -> None:
        row = next(row for row in self.records["test"] if "field-ledger" in row["group"])
        by_label = {entity["label"]: row["text"][entity["start"]:entity["end"]] for entity in row["entities"]}
        self.assertEqual(
            {"COUNTRY", "REGION", "LOCALITY", "STREET", "HOUSE", "APART"} & set(by_label),
            {"COUNTRY", "REGION", "LOCALITY", "STREET", "HOUSE", "APART"},
        )
        self.assertIn("ООО", by_label["ORG"])
        self.assertIn("терапии", by_label["INSTITUTION"])
        self.assertEqual(by_label["STREET"], "улица Зелёная Роща")

    def test_eponym_mentions_and_person_spans_are_distinct(self) -> None:
        row = next(row for row in self.records["challenge"] if "discrepancy-board" in row["template"])
        person = next(entity for entity in row["entities"] if entity["label"] == "PER")
        diseases = [entity for entity in row["entities"] if entity["label"] == "DISEASE"]
        self.assertEqual(row["text"][person["start"]:person["end"]], "Софьи Крыловой")
        self.assertEqual(len(diseases), 2)
        self.assertEqual([row["text"][entity["start"]:entity["end"]] for entity in diseases], ["синдром Марфана"] * 2)

    def test_every_occurrence_of_an_annotated_disease_surface_is_labeled(self) -> None:
        for rows in self.records.values():
            for row in rows:
                diseases = [entity for entity in row["entities"] if entity["label"] == "DISEASE"]
                for disease in diseases:
                    surface = row["text"][disease["start"]:disease["end"]]
                    occurrences: list[int] = []
                    offset = 0
                    while (offset := row["text"].find(surface, offset)) >= 0:
                        occurrences.append(offset)
                        offset += len(surface)
                    labeled = [entity["start"] for entity in diseases if row["text"][entity["start"]:entity["end"]] == surface]
                    self.assertEqual(occurrences, labeled, (row["template"], surface))

    def test_negative_records_contain_no_disease_name(self) -> None:
        for split, rows in self.records.items():
            for row in rows:
                if row["entities"]:
                    continue
                for disease in DISEASE_SURFACES:
                    self.assertNotIn(disease, row["text"], (split, row["template"], disease))

    def test_test_geography_and_heldout_formats_are_coherent_and_distinct_from_train(self) -> None:
        transcript = next(row for row in self.records["test"] if "call-transcript" in row["group"])
        address = {entity["label"]: transcript["text"][entity["start"]:entity["end"]] for entity in transcript["entities"]}
        self.assertEqual((address["COUNTRY"], address["REGION"], address["LOCALITY"]), ("Беларусь", "Минская область", "Минск"))
        self.assertIn("Устная передача по защищённому каналу", transcript["text"])
        challenge = next(row for row in self.records["challenge"] if "review-board" in row["group"])
        self.assertIn("Протокол комиссии по расхождениям", challenge["text"])
        for row in [*self.records["test"], *self.records["challenge"]]:
            for marker in TRAIN_FORMAT_MARKERS:
                self.assertNotIn(marker, row["text"], (row["template"], marker))

    def test_negative_examples_are_empty_and_have_contextual_numeric_hard_negatives(self) -> None:
        for split in scenarios.SPLITS:
            negatives = [row for row in self.records[split] if not row["entities"]]
            self.assertTrue(negatives, split)
            self.assertTrue(all(
                any(token in row["text"] for token in ("0000", "1111", "123456789012", "000 000"))
                for row in negatives
            ))

    def test_identifier_checksums_validate_for_every_authored_occurrence(self) -> None:
        for rows in self.records.values():
            for row in rows:
                for entity in row["entities"]:
                    if entity["label"] in {"SNILS", "INN", "OGRN"}:
                        surface = row["text"][entity["start"]:entity["end"]]
                        self.assertTrue(scenarios._valid_checksum(entity["label"], surface), surface)

    def test_heldout_groups_and_templates_are_split_independent_and_support_is_blocked(self) -> None:
        all_rows = [row for rows in self.records.values() for row in rows]
        self.assertEqual(len({row["template"] for row in all_rows}), len(all_rows))
        self.assertEqual(len({row["group"] for row in all_rows}), len(all_rows))
        for split in ("test", "challenge"):
            report = scenarios.support_report(self.records)[split]
            self.assertFalse(report["evaluation_ready"])
            self.assertEqual(set(report["support"]), scenarios.ACTIVE_LABELS)
            self.assertEqual(set(report["blocked_labels"]), scenarios.PRIORITY_LABELS)

    def test_additional_medical_scenarios_are_train_only_and_keep_split_sources_disjoint(self) -> None:
        self.assertEqual(len(self.records["train"]), 13)
        self.assertEqual(len(self.records["validation"]), 2)
        self.assertEqual(len(self.records["test"]), 3)
        self.assertEqual(len(self.records["challenge"]), 2)
        added = [row for row in self.records["train"] if row["group"].endswith("-v2")]
        self.assertEqual(len(added), 10)
        self.assertTrue(all(row["group"].startswith("train-") and row["template"].startswith("train-") for row in added))
        train_groups = {row["group"] for row in self.records["train"]}
        train_templates = {row["template"] for row in self.records["train"]}
        for split in ("validation", "test", "challenge"):
            self.assertFalse(train_groups & {row["group"] for row in self.records[split]})
            self.assertFalse(train_templates & {row["template"] for row in self.records[split]})


if __name__ == "__main__":
    unittest.main()
