from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ner_1_17"))

from train_candidate import PRIORITY_LABELS, independent_support_is_ready


class CandidateReadinessTests(unittest.TestCase):
    def test_independent_readiness_requires_review_and_both_splits_support(self) -> None:
        support = {label: 30 for label in PRIORITY_LABELS}
        ready_report = {
            "evaluation_ready": True,
            "editorial_review_complete": True,
            "target_support_by_split": {"test": support, "challenge": support},
        }
        self.assertTrue(independent_support_is_ready(ready_report))

        for incomplete in (
            {**ready_report, "evaluation_ready": False},
            {**ready_report, "editorial_review_complete": False},
            {
                **ready_report,
                "target_support_by_split": {"test": support, "challenge": {**support, "SNILS": 29}},
            },
            {**ready_report, "target_support_by_split": {"challenge": support}},
        ):
            with self.subTest(incomplete=incomplete):
                self.assertFalse(independent_support_is_ready(incomplete))


if __name__ == "__main__":
    unittest.main()
