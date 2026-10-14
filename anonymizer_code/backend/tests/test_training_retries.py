from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from app import worker
from app.store import (
    AUTO_TRAIN_MAX_ATTEMPTS,
    AUTO_TRAIN_RETRY_DELAY,
    automatic_training_retry_blocked,
    recover_interrupted_training_jobs,
)


class AutomaticTrainingRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
        self.corpus_snapshot = [{"corpus_id": "base", "corpus_version": "1.0.0", "release_id": "a" * 64}]
        self.category_snapshot = {"PER"}

    def blocked(self, jobs: list[dict[str, object]]) -> bool:
        connection = MagicMock()
        execute = connection.__enter__.return_value.execute
        execute.return_value.fetchall.return_value = jobs
        with patch("app.store.connect", return_value=connection):
            result = automatic_training_retry_blocked(42, self.corpus_snapshot, self.category_snapshot)
        query, params = execute.call_args.args
        self.assertIn("trigger='automatic'", query)
        self.assertIn("annotation_watermark=%s", query)
        self.assertIn("corpus_snapshot=%s", query)
        self.assertIn("category_snapshot=%s", query)
        self.assertEqual(params[0], 42)
        self.assertEqual(params[1].obj, self.corpus_snapshot)
        self.assertEqual(params[2].obj, ["PER"])
        return result

    def job(
        self,
        status: str,
        *,
        finished_at: datetime | None = None,
        model_id: int | None = None,
    ) -> dict[str, object]:
        return {
            "status": status,
            "finished_at": finished_at,
            "model_id": model_id,
            "database_now": self.now,
        }

    def test_no_prior_job_allows_first_attempt(self) -> None:
        self.assertFalse(self.blocked([]))

    def test_queued_running_and_succeeded_jobs_block_duplicates(self) -> None:
        for status in ("queued", "running", "succeeded"):
            with self.subTest(status=status):
                self.assertTrue(self.blocked([self.job(status)]))

    def test_saved_model_blocks_retry_even_if_job_status_is_failed(self) -> None:
        old = self.now - AUTO_TRAIN_RETRY_DELAY - timedelta(seconds=1)
        self.assertTrue(self.blocked([self.job("failed", finished_at=old, model_id=7)]))

    def test_recent_failure_waits_and_old_failure_can_retry(self) -> None:
        recent = self.now - AUTO_TRAIN_RETRY_DELAY + timedelta(seconds=1)
        old = self.now - AUTO_TRAIN_RETRY_DELAY - timedelta(seconds=1)
        self.assertTrue(self.blocked([self.job("failed", finished_at=recent)]))
        self.assertFalse(self.blocked([self.job("failed", finished_at=old)]))

    def test_unknown_failure_time_waits_safely(self) -> None:
        self.assertTrue(self.blocked([self.job("failed")]))

    def test_attempt_limit_stops_retries_for_same_watermark(self) -> None:
        old = self.now - AUTO_TRAIN_RETRY_DELAY - timedelta(seconds=1)
        failures = [self.job("failed", finished_at=old) for _ in range(AUTO_TRAIN_MAX_ATTEMPTS)]
        self.assertTrue(self.blocked(failures))

    def test_changed_corpus_or_category_snapshot_has_a_separate_retry_window(self) -> None:
        recent = self.now - timedelta(minutes=1)
        self.assertTrue(self.blocked([self.job("failed", finished_at=recent)]))

        changed_corpus = [{**self.corpus_snapshot[0], "release_id": "b" * 64}]
        changed_categories = {"PHONE"}
        blocked = MagicMock()
        blocked.__enter__.return_value.execute.return_value.fetchall.return_value = []
        with patch("app.store.connect", return_value=blocked):
            self.assertFalse(automatic_training_retry_blocked(42, changed_corpus, self.category_snapshot))
        query, params = blocked.__enter__.return_value.execute.call_args.args
        self.assertIn("corpus_snapshot=%s", query)
        self.assertEqual(params[1].obj, changed_corpus)

        with patch("app.store.connect", return_value=blocked):
            self.assertFalse(automatic_training_retry_blocked(42, self.corpus_snapshot, changed_categories))
        query, params = blocked.__enter__.return_value.execute.call_args.args
        self.assertIn("category_snapshot=%s", query)
        self.assertEqual(params[2].obj, ["PHONE"])

    def test_interrupted_running_jobs_are_closed_and_use_retry_cooldown(self) -> None:
        db = MagicMock()
        recover_interrupted_training_jobs(db)
        lock_query = db.execute.call_args_list[0].args[0]
        self.assertIn("pg_advisory_xact_lock(84327121)", lock_query)
        failed_query = db.execute.call_args_list[1].args[0]
        self.assertIn("status='failed'", failed_query)
        self.assertIn("WHERE status='running' AND model_id IS NULL", failed_query)
        self.assertIn("finished_at=clock_timestamp()", failed_query)
        recovered_query = db.execute.call_args_list[2].args[0]
        self.assertIn("status='succeeded'", recovered_query)
        self.assertIn("job.model_id=model.id", recovered_query)
        self.assertIn("finished_at=clock_timestamp()", recovered_query)

        recent = self.now - timedelta(minutes=1)
        self.assertTrue(self.blocked([self.job("failed", finished_at=recent)]))

    def test_worker_uses_current_corpus_and_categories_for_retry_and_queue(self) -> None:
        release_data = {"corpus_id": "base", "corpus_version": "1.0.0", "release_id": "a" * 64}
        release = MagicMock()
        release.release_id = release_data["release_id"]
        release.corpus_id = release_data["corpus_id"]
        release.corpus_version = release_data["corpus_version"]
        release.sha256 = "a" * 64
        release.as_dict.return_value = release_data
        with (
            patch("app.worker.annotation_watermark", return_value=42),
            patch(
                "app.worker.new_annotations_since_last_version",
                return_value=worker.config.auto_train_min_new_examples,
            ),
            patch("app.worker.training_corpus_release_ids", return_value={release.release_id}),
            patch("app.worker.list_corpus_releases", return_value=[release]),
            patch("app.worker.training_labels", return_value={"PER"}),
            patch("app.worker.automatic_training_retry_blocked", return_value=False) as retry_check,
            patch("app.worker.queue_training_job") as queue_job,
        ):
            worker.maybe_queue_automatic_training()
        retry_check.assert_called_once_with(42, [release_data], {"PER"})
        queue_job.assert_called_once_with("automatic", [release_data], {"PER"}, 42)

    def test_success_records_wall_clock_completion_time(self) -> None:
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = {
            "corpus_snapshot": [{"release_id": "release-1"}],
            "category_snapshot": ["PER"],
            "annotation_watermark": 42,
            "model_id": None,
        }
        with (
            patch("app.worker.load_corpus_records", return_value=([], [])),
            patch("app.worker.train_snapshot_at", return_value=({"PER"}, [])),
            patch("app.worker.load_corpus_holdout_texts", return_value=set()),
            patch("app.worker.active_model", return_value=None),
            patch("app.worker.train_model", return_value=(b"artifact", {"entity_micro_f1": 1.0})),
            patch("app.worker.save_model_version"),
        ):
            worker.run_job(db, 7)

        completion_query = db.execute.call_args_list[-1].args[0]
        self.assertIn("status='succeeded'", completion_query)
        self.assertIn("finished_at=clock_timestamp()", completion_query)

    def test_worker_warm_starts_from_active_v2_and_records_its_parent(self) -> None:
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = {
            "corpus_snapshot": [{"release_id": "release-1"}],
            "category_snapshot": ["PER"],
            "annotation_watermark": 42,
            "model_id": None,
        }
        with (
            patch("app.worker.load_corpus_records", return_value=([], [])),
            patch("app.worker.train_snapshot_at", return_value=({"PER"}, [])),
            patch("app.worker.load_corpus_holdout_texts", return_value=set()),
            patch("app.worker.active_model", return_value={"id": 9, "model_series": "ru-pii"}),
            patch("app.worker.model_artifact", return_value=b"active-v2"),
            patch("app.worker.decode_artifact", return_value=(None, [], None, None, 2)),
            patch("app.worker.train_model", return_value=(b"candidate", {"entity_micro_f1": 1.0})) as train,
            patch("app.worker.save_model_version") as save,
        ):
            worker.run_job(db, 7)

        self.assertEqual(train.call_args.kwargs["initial_artifact"], b"active-v2")
        self.assertEqual(save.call_args.kwargs["parent_model_id"], 9)

    def test_failure_records_wall_clock_completion_time(self) -> None:
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = {
            "corpus_snapshot": [{"release_id": "release-1"}],
            "category_snapshot": ["PER"],
            "annotation_watermark": 42,
            "model_id": None,
        }
        with (
            patch("app.worker.load_corpus_records", side_effect=RuntimeError("expected test failure")),
            patch("app.worker.logger.error"),
        ):
            worker.run_job(db, 7)

        failure_query = db.execute.call_args_list[-1].args[0]
        self.assertIn("status='failed'", failure_query)
        self.assertIn("finished_at=clock_timestamp()", failure_query)


if __name__ == "__main__":
    unittest.main()
