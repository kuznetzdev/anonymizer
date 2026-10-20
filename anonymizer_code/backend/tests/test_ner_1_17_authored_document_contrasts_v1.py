from __future__ import annotations

import unittest

from scripts.ner_1_17 import authored_document_contrasts_v1 as corpus


class AuthoredDocumentContrastCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = corpus.build_records()

    def test_offsets_are_codepoint_exact_and_non_overlapping(self) -> None:
        for split_records in self.records.values():
            for record in split_records:
                previous_end = 0
                for entity in record["entities"]:
                    self.assertGreaterEqual(entity["start"], previous_end)
                    self.assertLess(entity["start"], entity["end"])
                    self.assertTrue(record["text"][entity["start"]:entity["end"]])
                    previous_end = entity["end"]
                self.assertNotIn("[[", record["text"])

    def test_offsets_count_unicode_codepoints(self) -> None:
        record = corpus._record("unit", "unicode-offset", "Начало ё [[PER|Ёжик]] и конец.")
        entity = record["entities"][0]
        self.assertEqual(record["text"][entity["start"]:entity["end"]], "Ёжик")
        self.assertEqual(entity["start"], record["text"].index("Ёжик"))

    def test_document_parent_maps_concrete_leaf_labels_only(self) -> None:
        labels = {
            entity["label"]
            for records in self.records.values()
            for record in records
            for entity in record["entities"]
        }
        self.assertEqual(set(corpus.PARENT_BY_LABEL), corpus.DOCUMENT_LEAF_LABELS)
        self.assertEqual(set(corpus.PARENT_BY_LABEL.values()), {"DOCUMENT"})
        self.assertEqual(corpus.DOCUMENT_LEAF_LABELS - labels, set())
        self.assertNotIn("DOCUMENT", labels)

    def test_every_split_has_all_document_leaf_labels(self) -> None:
        for split, records in self.records.items():
            labels = {entity["label"] for record in records for entity in record["entities"]}
            with self.subTest(split=split):
                self.assertTrue(corpus.DOCUMENT_LEAF_LABELS <= labels)

    def test_all_authored_snils_pass_the_project_checksum(self) -> None:
        for records in self.records.values():
            for record in records:
                for entity in record["entities"]:
                    if entity["label"] != "SNILS":
                        continue
                    value = record["text"][entity["start"]:entity["end"]]
                    digits = "".join(char for char in value if char.isdigit())
                    weighted_sum = sum(
                        int(digit) * weight
                        for digit, weight in zip(digits[:9], range(9, 0, -1))
                    )
                    checksum = weighted_sum if weighted_sum < 100 else weighted_sum % 101
                    if checksum in (100, 101):
                        checksum = 0
                    with self.subTest(value=value):
                        self.assertEqual(len(digits), 11)
                        self.assertEqual(int(digits[9:]), checksum)

    def test_record_texts_are_exactly_unique(self) -> None:
        texts = [record["text"] for records in self.records.values() for record in records]
        self.assertEqual(len(texts), len(set(texts)))

    def test_eponym_disease_street_and_person_are_separate_entities(self) -> None:
        entities = {
            (entity["label"], record["text"][entity["start"]:entity["end"]])
            for records in self.records.values()
            for record in records
            for entity in record["entities"]
        }
        self.assertIn(("DISEASE", "болезни Боткина"), entities)
        self.assertIn(("STREET", "улица Боткина"), entities)
        self.assertIn(("PER", "Сергей Боткин"), entities)

    def test_source_groups_and_templates_are_unique_across_all_splits(self) -> None:
        groups: dict[str, str] = {}
        templates: dict[str, str] = {}
        for split, records in self.records.items():
            for record in records:
                self.assertNotIn(record["group"], groups)
                self.assertNotIn(record["template"], templates)
                groups[record["group"]] = split
                templates[record["template"]] = split
        self.assertTrue(all(groups[group] == "test" for group in groups if group.startswith("doc-test-")))
        self.assertTrue(all(groups[group] == "challenge" for group in groups if group.startswith("doc-challenge-")))

    def test_negative_number_cues_have_no_entity_overlap(self) -> None:
        negatives = (
            "Кабинет 214",
            "давление 118/76",
            "доза 5 мг",
            "код пробирки 483921",
            "комната 5902",
            "Температура 37,1",
            "Номер койки 22",
            "Номер кабинета 4419",
        )
        texts = "\n".join(record["text"] for records in self.records.values() for record in records)
        for cue in negatives:
            with self.subTest(cue=cue):
                self.assertIn(cue, texts)
                for records in self.records.values():
                    for record in records:
                        offset = record["text"].find(cue)
                        if offset < 0:
                            continue
                        end = offset + len(cue)
                        self.assertFalse(
                            any(entity["start"] < end and offset < entity["end"] for entity in record["entities"])
                        )


if __name__ == "__main__":
    unittest.main()
