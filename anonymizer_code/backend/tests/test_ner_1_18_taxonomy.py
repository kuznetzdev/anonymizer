from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path
import sys

from app.category_taxonomy import (
    DOCUMENT_LABEL_DESCRIPTIONS,
    DOCUMENT_LABELS,
    category_parent,
)
from app.training.corpus import load_ontology_descriptions, load_ontology_labels


ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "corpora"
    / "ru-pii"
    / "v1"
    / "ontology.json"
)
CODE_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = CODE_ROOT / "scripts" / "build_ner_hardcases_1_17.py"
builder_spec = importlib.util.spec_from_file_location("build_ner_hardcases_1_17", BUILDER_PATH)
if builder_spec is None or builder_spec.loader is None:
    raise RuntimeError("Cannot load the 1.17 corpus builder")
legacy_builder = importlib.util.module_from_spec(builder_spec)
sys.modules[builder_spec.name] = legacy_builder
builder_spec.loader.exec_module(legacy_builder)
ACTIVE_LABELS_1_18 = {
    "APART",
    "COUNTRY",
    "DISEASE",
    "DRIVER_LICENSE_ID",
    "EMAIL",
    "HOUSE",
    "INN",
    "INSTITUTION",
    "KPP",
    "LOCALITY",
    "MEDICAL_RECORD_ID",
    "OGRN",
    "OMS_POLICY_ID",
    "ORG",
    "OTHER_DOCUMENT_ID",
    "PASSPORT",
    "PER",
    "PHONE",
    "REFERRAL_ID",
    "REGION",
    "SNILS",
    "STREET",
}


class Ner118TaxonomyTests(unittest.TestCase):
    def test_ontology_imports_exactly_22_active_leaf_labels(self) -> None:
        ontology_labels = load_ontology_labels(ONTOLOGY_PATH)

        self.assertEqual(ontology_labels - {"ADDR"}, ACTIVE_LABELS_1_18)
        self.assertIn("ADDR", ontology_labels)
        self.assertEqual(ACTIVE_LABELS_1_18 & DOCUMENT_LABELS, {
            "DRIVER_LICENSE_ID",
            "MEDICAL_RECORD_ID",
            "OMS_POLICY_ID",
            "OTHER_DOCUMENT_ID",
            "PASSPORT",
            "REFERRAL_ID",
        })

    def test_document_parent_covers_new_leaves_and_legacy_category(self) -> None:
        self.assertTrue(all(category_parent(label) == "DOCUMENT" for label in DOCUMENT_LABELS))
        self.assertEqual(category_parent("PATIENT_CARD_ID"), "DOCUMENT")
        self.assertIsNone(category_parent("DISEASE"))
        self.assertEqual(set(DOCUMENT_LABEL_DESCRIPTIONS), DOCUMENT_LABELS | {"PATIENT_CARD_ID"})
        self.assertIn("MEDICAL_RECORD_ID", DOCUMENT_LABEL_DESCRIPTIONS["PATIENT_CARD_ID"])
        ontology_descriptions = load_ontology_descriptions(ONTOLOGY_PATH)
        for label in DOCUMENT_LABELS - {"PASSPORT"}:
            with self.subTest(label=label):
                self.assertEqual(DOCUMENT_LABEL_DESCRIPTIONS[label], ontology_descriptions[label])

    def test_legacy_patient_card_label_is_not_a_new_ontology_leaf(self) -> None:
        self.assertNotIn("PATIENT_CARD_ID", load_ontology_labels(ONTOLOGY_PATH))

    def test_legacy_builder_accepts_superset_but_emits_only_17_registered_labels(self) -> None:
        cases = legacy_builder.build_cases(load_ontology_labels(ONTOLOGY_PATH))
        emitted_labels = {entity["label"] for case in cases for entity in case.entities}

        self.assertEqual(len(legacy_builder.ACTIVE_LABELS), 17)
        self.assertEqual(emitted_labels, legacy_builder.ACTIVE_LABELS)

    def test_legacy_builder_rejects_missing_registered_label(self) -> None:
        labels = load_ontology_labels(ONTOLOGY_PATH) - {"SNILS"}

        with self.assertRaisesRegex(ValueError, "missing registered 1.17 labels"):
            legacy_builder.build_cases(labels)
        with self.assertRaisesRegex(ValueError, "missing registered 1.17 labels"):
            legacy_builder.validate_cases([], labels)


if __name__ == "__main__":
    unittest.main()
