from __future__ import annotations

import unittest

from app.model import normalize_legacy_prediction_labels


class LegacyPredictionLabelTests(unittest.TestCase):
    def test_retired_address_label_is_normalized_without_changing_other_spans(self) -> None:
        spans = [
            {"start": 0, "end": 5, "label": "ADDR", "confidence": 0.9},
            {"start": 6, "end": 10, "label": "PER", "confidence": 0.8},
        ]

        self.assertEqual(
            normalize_legacy_prediction_labels(spans),
            [
                {"start": 0, "end": 5, "label": "LOCALITY", "confidence": 0.9},
                spans[1],
            ],
        )


if __name__ == "__main__":
    unittest.main()
