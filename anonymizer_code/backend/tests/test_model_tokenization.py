from __future__ import annotations

import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher

from app.model import FEATURES, SNILS_RE, decode_artifact, labels_for_tokens, predict, tokenize, train_model


def artifact(
    version: int,
    classes: list[str] | None = None,
    intercept: list[float] | None = None,
    weights: csr_matrix | None = None,
) -> bytes:
    buffer = BytesIO()
    classes = classes or ["B-ID"]
    intercept_values = np.array(intercept or [0.0] * len(classes), dtype=np.float64)
    matrix = weights if weights is not None else csr_matrix((len(classes), FEATURES), dtype=np.float64)
    np.savez_compressed(
        buffer,
        version=np.array([version], dtype=np.int16),
        features=np.array([FEATURES], dtype=np.int32),
        classes=np.array(classes),
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.array(matrix.shape, dtype=np.int32),
        intercept=intercept_values,
    )
    return buffer.getvalue()


def _snils_sequence_artifact() -> bytes:
    hasher = FeatureHasher(n_features=FEATURES, input_type="dict", alternate_sign=False)
    cue = int(hasher.transform([{"w-1=снилс": 1.0}]).indices[0])
    weights = csr_matrix(
        (np.array([10.0]), (np.array([0]), np.array([cue]))),
        shape=(3, FEATURES),
    )
    return artifact(2, ["B-SNILS", "I-SNILS", "O"], [0.0, 1.0, -1.0], weights)


class TokenizationTests(unittest.TestCase):
    def test_splits_unicode_letters_from_digits_and_preserves_offsets(self) -> None:
        text = "ИНН5964761404 Ёж_12"
        self.assertEqual(
            tokenize(text),
            [("ИНН", 0, 3), ("5964761404", 3, 13), ("Ёж", 14, 16), ("_", 16, 17), ("12", 17, 19)],
        )

    def test_numeric_entity_has_exact_token_boundary(self) -> None:
        text = "ИНН5964761404"
        tokens = tokenize(text)
        record = {"text": text, "entities": [{"start": 3, "end": 13, "label": "ID"}]}
        self.assertEqual(labels_for_tokens(record, tokens, {"ID"}), ["O", "B-ID"])

    def test_partial_token_boundary_fails_with_offsets(self) -> None:
        text = "код123"
        record = {"text": text, "entities": [{"start": 1, "end": 6, "label": "ID"}]}
        with self.assertRaisesRegex(ValueError, r"offsets 1:6, matched 2 tokens"):
            labels_for_tokens(record, tokenize(text), {"ID"}, record_index=7)

    def test_legacy_artifact_keeps_legacy_token_boundaries(self) -> None:
        text = "ИНН5964761404"
        self.assertEqual(predict(text, artifact(1)), [{"start": 0, "end": len(text), "label": "ID", "confidence": 1.0}])
        self.assertEqual(
            predict(text, artifact(2)),
            [
                {"start": 0, "end": 3, "label": "ID", "confidence": 1.0},
                {"start": 3, "end": len(text), "label": "ID", "confidence": 1.0},
            ],
        )

    def test_model_confidence_is_bounded_and_deterministic(self) -> None:
        text = "Рита"
        model = artifact(2, ["B-ID", "O"], [0.0, 0.0])
        first = predict(text, model)
        second = predict(text, model)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["confidence"], 0.5)
        self.assertGreaterEqual(first[0]["confidence"], 0.0)
        self.assertLessEqual(first[0]["confidence"], 1.0)

    def test_multi_token_confidence_is_minimum_chosen_tag_probability(self) -> None:
        text = "Alice Smith"
        hasher = FeatureHasher(n_features=FEATURES, input_type="dict", alternate_sign=False)
        first_word = int(hasher.transform([{"word=alice": 1.0}]).indices[0])
        second_word = int(hasher.transform([{"word=smith": 1.0}]).indices[0])
        weights = csr_matrix(
            (
                np.array([2.0, 1.0]),
                (np.array([0, 1]), np.array([first_word, second_word])),
            ),
            shape=(3, FEATURES),
        )
        result = predict(text, artifact(2, ["B-ID", "I-ID", "O"], weights=weights))
        expected_minimum = float(np.exp(1.0) / (np.exp(1.0) + 2.0))
        self.assertEqual((result[0]["start"], result[0]["end"], result[0]["label"]), (0, len(text), "ID"))
        self.assertAlmostEqual(result[0]["confidence"], expected_minimum)

    def test_phone_rule_override_has_no_confidence(self) -> None:
        text = "+7 (999) 123-45-67"
        spans = predict(text, artifact(2))
        self.assertEqual(spans, [{"start": 0, "end": len(text), "label": "PHONE", "source": "rule"}])

    def test_snils_requires_a_supported_full_value_format(self) -> None:
        model = artifact(2, ["B-SNILS", "O"], [10.0, 0.0])
        self.assertEqual(predict("190", model), [])

        for value in ("12345678901", "123-456-789 01", "123-456-789-01", "123 456 789 01"):
            with self.subTest(value=value):
                self.assertIsNotNone(SNILS_RE.fullmatch(value))
                start = len("СНИЛС ")
                predicted = predict("СНИЛС " + value, _snils_sequence_artifact())
                self.assertEqual(
                    [(span["start"], span["end"], span["label"]) for span in predicted],
                    [(start, start + len(value), "SNILS")],
                )

        for value in ("190", "123-456-789-012", "123456789012"):
            with self.subTest(value=value):
                self.assertIsNone(SNILS_RE.fullmatch(value))

    def test_finalized_v17_reference_entities_align(self) -> None:
        release = Path(__file__).parents[2] / "data" / "corpora" / "ru-pii" / "releases" / "cc98f17c8b1742dea862322ae6da8e1fb43094c4c8b356a5884552ecbf256a37"
        if not release.is_dir():
            self.skipTest("finalized v1.7 corpus release is not present in this checkout")
        for split in ("train", "validation", "test"):
            with (release / f"{split}.jsonl").open(encoding="utf-8") as source:
                for index, line in enumerate(source):
                    record = json.loads(line)
                    labels_for_tokens(record, tokenize(record["text"]), {
                        entity["label"] for entity in record["entities"]
                    }, index)

    def test_reviewed_examples_matching_test_holdout_are_excluded(self) -> None:
        class Classifier:
            def __init__(self, **kwargs: object) -> None:
                self.classes_: np.ndarray = np.array([], dtype=str)

            def partial_fit(self, matrix: object, labels: list[str], classes: np.ndarray | None = None) -> Classifier:
                if classes is not None:
                    self.classes_ = classes
                return self

            @property
            def coef_(self) -> np.ndarray:
                return np.zeros((len(self.classes_), FEATURES))

            @property
            def intercept_(self) -> np.ndarray:
                return np.zeros(len(self.classes_))

            def predict(self, matrix: object) -> np.ndarray:
                return np.array(["O"] * matrix.shape[0])

        train = {"text": "код 123", "entities": [{"start": 4, "end": 7, "label": "ID"}]}
        validation = {"text": "номер 456", "entities": [{"start": 6, "end": 9, "label": "ID"}]}
        heldout = {"text": "тест 789", "entities": [{"start": 5, "end": 8, "label": "ID"}]}
        with patch("app.model.SGDClassifier", Classifier):
            _, metrics = train_model(
                {"ID"}, [train], [validation], [heldout], heldout_texts={heldout["text"]},
            )
        self.assertEqual(metrics["reviewed_examples"], 0)
        self.assertEqual(metrics["reviewed_test_duplicates_removed"], 1)

    def test_corpus_records_cannot_overlap_test_holdout(self) -> None:
        record = {"text": "код 123", "entities": [{"start": 4, "end": 7, "label": "ID"}]}
        validation = {"text": "номер 456", "entities": [{"start": 6, "end": 9, "label": "ID"}]}
        with self.assertRaisesRegex(ValueError, "overlap test holdout"):
            train_model({"ID"}, [record], [validation], heldout_texts={record["text"]})

    def test_training_rejects_labels_outside_active_set(self) -> None:
        train = {"text": "код 123", "entities": [{"start": 4, "end": 7, "label": "UNKNOWN"}]}
        validation = {"text": "номер 456", "entities": [{"start": 6, "end": 9, "label": "ID"}]}
        with self.assertRaisesRegex(ValueError, "Unknown active label in training corpus"):
            train_model({"ID"}, [train], [validation])

    def test_warm_start_resumes_v2_weights_and_adds_new_bio_classes(self) -> None:
        initial = artifact(2, ["B-ID", "I-ID", "O"], [10.0, 0.0, -10.0])
        train = [
            {"text": "код 123", "entities": [{"start": 4, "end": 7, "label": "ID"}]},
            {"text": "дата 01.02.2026", "entities": [{"start": 5, "end": 15, "label": "DATE"}]},
        ]
        validation = [{"text": "номер 456", "entities": [{"start": 6, "end": 9, "label": "ID"}]}]

        warmed, _ = train_model({"ID", "DATE"}, train, validation, initial_artifact=initial)

        _, classes, _, _, version = decode_artifact(warmed)
        self.assertEqual(version, 2)
        self.assertTrue({"B-ID", "I-ID", "O", "B-DATE", "I-DATE"}.issubset(classes))
        self.assertTrue(any(span["label"] == "ID" for span in predict("код 789", warmed)))
        with np.load(BytesIO(warmed), allow_pickle=False) as saved:
            self.assertEqual(
                set(saved.files),
                {"version", "features", "classes", "data", "indices", "indptr", "shape", "intercept"},
            )

    def test_warm_start_rejects_v1_legacy_tokenizer_artifact(self) -> None:
        train = [{"text": "код 123", "entities": [{"start": 4, "end": 7, "label": "ID"}]}]
        validation = [{"text": "номер 456", "entities": [{"start": 6, "end": 9, "label": "ID"}]}]

        with self.assertRaisesRegex(ValueError, "Warm-start requires a v2 artifact"):
            train_model({"ID"}, train, validation, initial_artifact=artifact(1))

    def test_alignment_preflight_fails_before_classifier_construction(self) -> None:
        train = {"text": "код 123", "entities": [{"start": 4, "end": 7, "label": "ID"}]}
        invalid_train = {"text": "код 123", "entities": [{"start": 5, "end": 7, "label": "ID"}]}
        valid_validation = {"text": "номер 456", "entities": [{"start": 6, "end": 9, "label": "ID"}]}
        invalid_validation = {"text": "номер 456", "entities": [{"start": 7, "end": 9, "label": "ID"}]}
        for invalid_split, training, valid in (
            ("training", [invalid_train], [valid_validation]),
            ("validation", [train], [invalid_validation]),
        ):
            with self.subTest(invalid_split=invalid_split):
                with patch("app.model.SGDClassifier") as classifier:
                    with self.assertRaisesRegex(ValueError, "Entity token alignment failed"):
                        train_model({"ID"}, training, valid)
                classifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
