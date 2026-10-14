from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from .config import config
from .model import decode_artifact, train_model
from .store import (
    active_model, automatic_training_retry_blocked, connect, initialize, model_artifact,
    new_annotations_since_last_version, open_pool, queue_training_job,
    recover_interrupted_training_jobs, save_model_version,
    train_snapshot_at, training_labels,
    annotation_watermark, training_corpus_release_ids,
)
from .training.corpus import list_corpus_releases, load_corpus_holdout_texts, load_corpus_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("anonymizer.worker")
LOCK_ID = 84327119
CORPUS_ROOT = Path(config.data_dir) / "corpora" / "ru-pii"


def claim_job(db: Any) -> dict[str, Any] | None:
    return db.execute(
        """WITH candidate AS (
            SELECT id FROM training_jobs WHERE status IN ('queued','running')
            ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at
            LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        UPDATE training_jobs SET status='running', started_at=COALESCE(started_at,now())
        WHERE id=(SELECT id FROM candidate)
        RETURNING id""",
    ).fetchone()


def maybe_queue_automatic_training() -> None:
    watermark = annotation_watermark()
    if new_annotations_since_last_version(watermark) < config.auto_train_min_new_examples:
        return
    selected_ids = training_corpus_release_ids()
    catalog = list_corpus_releases(CORPUS_ROOT)
    if selected_ids - {release.release_id for release in catalog}:
        raise RuntimeError("A selected corpus release is missing from data storage")
    releases = [release for release in catalog if release.release_id in selected_ids]
    if not releases:
        return
    snapshot = [release.as_dict() for release in sorted(
        releases, key=lambda release: (release.corpus_id, release.corpus_version, release.sha256),
    )]
    categories = training_labels()
    if automatic_training_retry_blocked(watermark, snapshot, categories):
        return
    queue_training_job("automatic", snapshot, categories, watermark)


def run_job(db: Any, job_id: Any) -> None:
    try:
        job = db.execute(
            "SELECT corpus_snapshot,category_snapshot,annotation_watermark,model_id FROM training_jobs WHERE id=%s",
            (job_id,),
        ).fetchone()
        if job is None:
            raise RuntimeError("Training job disappeared")
        if job["model_id"] is not None:
            saved = db.execute("SELECT metrics FROM models WHERE id=%s", (job["model_id"],)).fetchone()
            if saved is None:
                raise RuntimeError("Training job points to a missing model")
            db.execute(
                "UPDATE training_jobs SET status='succeeded',metrics=%s,finished_at=clock_timestamp() WHERE id=%s",
                (Jsonb(saved["metrics"]), job_id),
            )
            return
        corpus_snapshot = job["corpus_snapshot"]
        release_ids = [release["release_id"] for release in corpus_snapshot]
        corpus_train, corpus_validation = load_corpus_records(CORPUS_ROOT, release_ids)
        labels, examples = train_snapshot_at(
            int(job["annotation_watermark"]), set(job["category_snapshot"]),
        )
        if labels != set(job["category_snapshot"]):
            raise RuntimeError("Training label snapshot is invalid")
        current = active_model()
        initial_artifact = None
        parent_model_id = None
        if current is not None and current["model_series"] == "ru-pii":
            candidate = model_artifact(int(current["id"]))
            if decode_artifact(candidate)[-1] == 2:
                initial_artifact = candidate
                parent_model_id = int(current["id"])
        artifact, metrics = train_model(
            labels, corpus_train, corpus_validation, examples,
            heldout_texts=load_corpus_holdout_texts(CORPUS_ROOT, release_ids),
            initial_artifact=initial_artifact,
        )
        metrics.update({
            "corpus_releases": len(corpus_snapshot),
            "annotation_watermark": int(job["annotation_watermark"]),
            "category_labels": sorted(labels),
        })
        save_model_version(
            artifact, metrics, corpus_snapshot, sorted(labels),
            int(job["annotation_watermark"]), job_id,
            parent_model_id=parent_model_id,
        )
        db.execute(
            "UPDATE training_jobs SET status='succeeded',metrics=%s,finished_at=clock_timestamp() WHERE id=%s",
            (Jsonb(metrics), job_id),
        )
    except Exception as exc:
        # Never log source text, annotations, or exception payloads.
        logger.error("Training job failed (job_id=%s, error_type=%s)", job_id, type(exc).__name__)
        db.execute(
            "UPDATE training_jobs SET status='failed',error='Training failed; inspect worker logs',finished_at=clock_timestamp() WHERE id=%s",
            (job_id,),
        )


def main() -> None:
    config.validate(auth_required=False)
    open_pool()
    initialize()
    logger.info("Training worker started")
    while True:
        with connect() as db:
            locked = db.execute("SELECT pg_try_advisory_lock(%s) AS locked", (LOCK_ID,)).fetchone()["locked"]
            job = None
            try:
                if locked:
                    recover_interrupted_training_jobs(db)
                    # Retry checks use separate pooled connections; publish recovery first.
                    db.commit()
                    maybe_queue_automatic_training()
                    job = claim_job(db)
                    if job:
                        db.commit()
                        run_job(db, job["id"])
            finally:
                if locked:
                    db.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
        if not locked or not job:
            time.sleep(1.5)


if __name__ == "__main__":
    main()
