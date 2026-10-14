from __future__ import annotations

import hashlib
import time
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ContextManager

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import config

if TYPE_CHECKING:
    from .seed import SeedModel

pool = ConnectionPool(
    conninfo=config.database_conninfo,
    min_size=2,
    max_size=config.db_pool_max,
    timeout=10,
    kwargs={"row_factory": dict_row, "connect_timeout": 10},
    open=False,
)

AUTO_TRAIN_MAX_ATTEMPTS = 3
AUTO_TRAIN_RETRY_DELAY = timedelta(minutes=5)


def open_pool() -> None:
    pool.open(wait=True)


def close_pool() -> None:
    pool.close()


def connect() -> ContextManager[psycopg.Connection]:
    return pool.connection()


def initialize() -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS categories(label TEXT PRIMARY KEY, active BOOLEAN NOT NULL DEFAULT TRUE, built_in BOOLEAN NOT NULL DEFAULT FALSE, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
    CREATE TABLE IF NOT EXISTS examples(id BIGSERIAL PRIMARY KEY, text TEXT NOT NULL, text_hash CHAR(64) NOT NULL UNIQUE, entities JSONB NOT NULL, empty_review_confirmed BOOLEAN NOT NULL DEFAULT FALSE, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
    ALTER TABLE examples ADD COLUMN IF NOT EXISTS empty_review_confirmed BOOLEAN NOT NULL DEFAULT FALSE;
    ALTER TABLE examples ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
    CREATE TABLE IF NOT EXISTS example_revisions(id BIGSERIAL PRIMARY KEY, example_id BIGINT NOT NULL REFERENCES examples(id) ON DELETE CASCADE, text TEXT NOT NULL, entities JSONB NOT NULL, empty_review_confirmed BOOLEAN NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
    CREATE TABLE IF NOT EXISTS feedback(id BIGSERIAL PRIMARY KEY, example_id BIGINT NOT NULL REFERENCES examples(id) ON DELETE CASCADE, span_start INTEGER NOT NULL, span_end INTEGER NOT NULL, label TEXT NOT NULL, reason TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', created_at TIMESTAMPTZ NOT NULL DEFAULT now());
    CREATE TABLE IF NOT EXISTS models(id BIGSERIAL PRIMARY KEY, artifact BYTEA NOT NULL, artifact_version SMALLINT NOT NULL, artifact_sha256 CHAR(64) NOT NULL, metrics JSONB NOT NULL, trained_at TIMESTAMPTZ NOT NULL DEFAULT now(), active BOOLEAN NOT NULL DEFAULT FALSE, model_series TEXT NOT NULL DEFAULT 'legacy', version_no INTEGER, parent_model_id BIGINT REFERENCES models(id), corpus_snapshot JSONB, category_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb, annotation_watermark BIGINT NOT NULL DEFAULT 0);
    ALTER TABLE models ADD COLUMN IF NOT EXISTS model_series TEXT NOT NULL DEFAULT 'legacy';
    ALTER TABLE models ADD COLUMN IF NOT EXISTS version_no INTEGER;
    ALTER TABLE models ADD COLUMN IF NOT EXISTS parent_model_id BIGINT REFERENCES models(id);
    ALTER TABLE models ADD COLUMN IF NOT EXISTS corpus_snapshot JSONB;
    ALTER TABLE models ADD COLUMN IF NOT EXISTS category_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb;
    ALTER TABLE models ADD COLUMN IF NOT EXISTS annotation_watermark BIGINT NOT NULL DEFAULT 0;
    CREATE TABLE IF NOT EXISTS training_jobs(id UUID PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')), error TEXT, metrics JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, trigger TEXT NOT NULL DEFAULT 'manual', corpus_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb, category_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb, annotation_watermark BIGINT NOT NULL DEFAULT 0, model_series TEXT NOT NULL DEFAULT 'ru-pii', model_id BIGINT REFERENCES models(id), model_version INTEGER);
    CREATE TABLE IF NOT EXISTS corpus_releases(release_id CHAR(64) PRIMARY KEY, corpus_id TEXT NOT NULL, corpus_version TEXT NOT NULL, installed_at TIMESTAMPTZ NOT NULL DEFAULT now());
    ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS trigger TEXT NOT NULL DEFAULT 'manual';
    ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS corpus_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb;
    ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS category_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb;
    ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS annotation_watermark BIGINT NOT NULL DEFAULT 0;
    ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS model_series TEXT NOT NULL DEFAULT 'ru-pii';
    ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS model_id BIGINT REFERENCES models(id);
    ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS model_version INTEGER;
    CREATE TABLE IF NOT EXISTS auth_sessions(token_hash CHAR(64) PRIMARY KEY, expires_at TIMESTAMPTZ NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
    INSERT INTO example_revisions(example_id,text,entities,empty_review_confirmed)
        SELECT e.id,e.text,e.entities,e.empty_review_confirmed FROM examples e
        WHERE NOT EXISTS (SELECT 1 FROM example_revisions r WHERE r.example_id=e.id);
    WITH ranked AS (
        SELECT id, row_number() OVER (PARTITION BY model_series ORDER BY id) AS version_no
        FROM models WHERE version_no IS NULL
    )
    UPDATE models m SET version_no=ranked.version_no FROM ranked WHERE ranked.id=m.id;
    CREATE INDEX IF NOT EXISTS feedback_example_idx ON feedback(example_id);
    CREATE INDEX IF NOT EXISTS revisions_example_idx ON example_revisions(example_id,id DESC);
    CREATE INDEX IF NOT EXISTS examples_entities_gin_idx ON examples USING GIN (entities jsonb_path_ops);
    CREATE INDEX IF NOT EXISTS training_created_idx ON training_jobs(created_at DESC);
    CREATE INDEX IF NOT EXISTS corpus_releases_latest_idx ON corpus_releases(corpus_id,installed_at DESC,release_id DESC);
    CREATE INDEX IF NOT EXISTS auth_sessions_expiry_idx ON auth_sessions(expires_at);
    CREATE UNIQUE INDEX IF NOT EXISTS model_series_version_idx ON models(model_series,version_no);
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_model_idx ON models(active) WHERE active;
    """
    for attempt in range(30):
        try:
            with connect() as db:
                db.execute("SELECT pg_advisory_xact_lock(84327118)")
                for statement in schema.split(";"):
                    if statement.strip():
                        db.execute(statement)
                return
        except psycopg.OperationalError:
            if attempt == 29:
                raise
            time.sleep(2)


def active_categories() -> list[dict[str, Any]]:
    with connect() as db:
        return list(db.execute("SELECT label, active, built_in FROM categories ORDER BY label").fetchall())


def example_page(
    category: str | None, before_id: int | None, limit: int,
) -> tuple[list[dict[str, Any]], int | None] | None:
    with connect() as db:
        if category is not None and db.execute(
            "SELECT 1 FROM categories WHERE label=%s", (category,),
        ).fetchone() is None:
            return None
        conditions: list[str] = []
        params: list[Any] = []
        if before_id is not None:
            conditions.append("e.id < %s")
            params.append(before_id)
        if category is not None:
            conditions.append("e.entities @> %s")
            params.append(Jsonb([{"label": category}]))
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = db.execute(
            f"""SELECT e.id, e.text, e.entities, e.created_at,
                       CASE WHEN jsonb_array_length(e.entities)=0
                            THEN e.empty_review_confirmed
                            ELSE NOT EXISTS (
                                SELECT 1 FROM jsonb_array_elements(e.entities) entity
                                LEFT JOIN categories c ON c.label=entity->>'label'
                                WHERE c.label IS NULL
                            )
                       END AS training_eligible
                FROM examples e{where}
                ORDER BY e.id DESC LIMIT %s""",
            (*params, limit + 1),
        ).fetchall()
        has_more = len(rows) > limit
        page = rows[:limit]
        feedback_by_id: dict[int, list[dict[str, Any]]] = {row["id"]: [] for row in page}
        if page:
            feedback = db.execute(
                """SELECT example_id, span_start, span_end, label, reason, note
                   FROM feedback WHERE example_id = ANY(%s) ORDER BY example_id, id""",
                ([row["id"] for row in page],),
            ).fetchall()
            for item in feedback:
                feedback_by_id[item["example_id"]].append({
                    "start": item["span_start"], "end": item["span_end"],
                    "label": item["label"], "reason": item["reason"], "note": item["note"],
                })
        items = [
            {**row, "feedback": feedback_by_id[row["id"]]}
            for row in page
        ]
        next_cursor = page[-1]["id"] if has_more and page else None
        return items, next_cursor


def active_model() -> dict[str, Any] | None:
    with connect() as db:
        return db.execute("SELECT id, metrics, trained_at, model_series, version_no, corpus_snapshot, category_snapshot, annotation_watermark FROM models WHERE active=TRUE ORDER BY id DESC LIMIT 1").fetchone()


def model_versions() -> list[dict[str, Any]]:
    with connect() as db:
        return list(db.execute(
            "SELECT id, model_series, version_no, parent_model_id, metrics, trained_at, active, corpus_snapshot, category_snapshot, annotation_watermark FROM models ORDER BY model_series, version_no DESC"
        ).fetchall())


def register_corpus_release(corpus_id: str, corpus_version: str, release_id: str) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO corpus_releases(release_id,corpus_id,corpus_version) VALUES (%s,%s,%s) ON CONFLICT(release_id) DO NOTHING",
            (release_id, corpus_id, corpus_version),
        )


def register_corpus_releases(releases: list[dict[str, Any]]) -> None:
    with connect() as db:
        with db.cursor() as cursor:
            for release in sorted(
                releases, key=lambda item: (item["corpus_id"], item["corpus_version"], item["release_id"]),
            ):
                cursor.execute(
                    "INSERT INTO corpus_releases(release_id,corpus_id,corpus_version) VALUES (%s,%s,%s) ON CONFLICT(release_id) DO NOTHING",
                    (release["release_id"], release["corpus_id"], release["corpus_version"]),
                )


def training_corpus_release_ids() -> set[str]:
    with connect() as db:
        rows = db.execute(
            """SELECT DISTINCT ON (corpus_id) release_id FROM corpus_releases
               ORDER BY corpus_id,
                 split_part(corpus_version,'.',1)::bigint DESC,
                 split_part(corpus_version,'.',2)::bigint DESC,
                 split_part(corpus_version,'.',3)::bigint DESC,
                 installed_at DESC,release_id DESC"""
        ).fetchall()
    return {row["release_id"].strip() for row in rows}


def model_artifact(model_id: int) -> bytes:
    with connect() as db:
        row = db.execute("SELECT artifact, artifact_version, artifact_sha256 FROM models WHERE id=%s", (model_id,)).fetchone()
    if row is None:
        raise RuntimeError("Active model disappeared")
    artifact = bytes(row["artifact"])
    if row["artifact_version"] != 1 or hashlib.sha256(artifact).hexdigest() != row["artifact_sha256"].strip():
        raise RuntimeError("Stored model artifact failed integrity validation")
    return artifact


def install_seed_model(seed: SeedModel) -> bool:
    """Install the bundled base model once, without replacing user models."""
    if hashlib.sha256(seed.artifact).hexdigest() != seed.sha256:
        raise ValueError("Seed model artifact failed integrity validation")
    metrics = dict(seed.metrics)
    metrics["seed_model"] = {
        "version": seed.version,
        **({"training": seed.training} if seed.training is not None else {}),
    }
    with connect() as db:
        # Serialize with training saves and model activation to avoid replacing a
        # concurrently-created user model or violating the single-active index.
        db.execute("SELECT pg_advisory_xact_lock(84327122)")
        if db.execute(
            "SELECT 1 FROM models WHERE model_series='ru-pii' LIMIT 1",
        ).fetchone() is not None:
            return False
        if db.execute("SELECT 1 FROM models WHERE active=TRUE LIMIT 1").fetchone() is not None:
            return False
        db.execute(
            """INSERT INTO models(artifact,artifact_version,artifact_sha256,metrics,active,
                       model_series,version_no,parent_model_id,corpus_snapshot,category_snapshot,annotation_watermark)
               VALUES (%s,1,%s,%s,TRUE,'ru-pii',1,NULL,'[]'::jsonb,%s,0)""",
            (seed.artifact, seed.sha256, Jsonb(metrics), Jsonb(list(seed.categories))),
        )
    return True


def ensure_categories(labels: set[str]) -> None:
    with connect() as db:
        with db.cursor() as cursor:
            cursor.executemany("INSERT INTO categories(label,built_in) VALUES (%s,TRUE) ON CONFLICT(label) DO UPDATE SET built_in=TRUE", [(label,) for label in sorted(labels)])


def archive_categories(labels: set[str]) -> None:
    if not labels:
        return
    with connect() as db:
        db.execute(
            "UPDATE categories SET active=FALSE, updated_at=now() WHERE label = ANY(%s)",
            (sorted(labels),),
        )


def ensure_corpus_categories(labels: set[str]) -> None:
    with connect() as db:
        with db.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO categories(label,built_in) VALUES (%s,FALSE) ON CONFLICT(label) DO NOTHING",
                [(label,) for label in sorted(labels)],
            )


def training_labels() -> set[str]:
    with connect() as db:
        return {row["label"] for row in db.execute("SELECT label FROM categories ORDER BY label").fetchall()}


def annotation_watermark() -> int:
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(84327123)")
        return int(db.execute("SELECT COALESCE(max(id),0) AS watermark FROM example_revisions").fetchone()["watermark"])


def train_snapshot_at(watermark: int, labels: set[str] | None = None) -> tuple[set[str], list[dict[str, Any]]]:
    with connect() as db:
        if labels is None:
            labels = {row["label"] for row in db.execute("SELECT label FROM categories ORDER BY label").fetchall()}
        examples = db.execute(
            """SELECT DISTINCT ON (example_id) example_id, id, text, entities, empty_review_confirmed
               FROM example_revisions WHERE id <= %s ORDER BY example_id, id DESC""",
            (watermark,),
        ).fetchall()
    eligible = [
        {"text": item["text"], "entities": item["entities"]}
        for item in examples
        if (item["entities"] and all(entity["label"] in labels for entity in item["entities"]))
        or (not item["entities"] and item["empty_review_confirmed"])
    ]
    return labels, eligible


def new_annotations_since_last_version(up_to_watermark: int) -> int:
    with connect() as db:
        watermark = db.execute(
            "SELECT COALESCE(max(annotation_watermark),0) AS watermark FROM models WHERE model_series='ru-pii'"
        ).fetchone()["watermark"]
        return int(db.execute(
            "SELECT count(DISTINCT example_id) AS n FROM example_revisions WHERE id > %s AND id <= %s",
            (watermark, up_to_watermark),
        ).fetchone()["n"])


def recover_interrupted_training_jobs(db: Any) -> None:
    db.execute("SELECT pg_advisory_xact_lock(84327121)")
    db.execute(
        """UPDATE training_jobs
           SET status='failed', error='Training interrupted by worker restart', finished_at=clock_timestamp()
           WHERE status='running' AND model_id IS NULL""",
    )
    db.execute(
        """UPDATE training_jobs AS job
           SET status='succeeded', error=NULL, metrics=model.metrics, finished_at=clock_timestamp()
           FROM models AS model
           WHERE job.status='running' AND job.model_id=model.id""",
    )


def automatic_training_retry_blocked(
    watermark: int,
    corpus_snapshot: list[dict[str, Any]],
    category_snapshot: set[str],
) -> bool:
    with connect() as db:
        jobs = db.execute(
            """SELECT status, finished_at, model_id, now() AS database_now
               FROM training_jobs
               WHERE trigger='automatic' AND annotation_watermark=%s
                 AND corpus_snapshot=%s AND category_snapshot=%s
               ORDER BY created_at""",
            (watermark, Jsonb(corpus_snapshot), Jsonb(sorted(category_snapshot))),
        ).fetchall()
    if any(
        job["status"] in {"queued", "running", "succeeded"} or job["model_id"] is not None
        for job in jobs
    ):
        return True
    failed = [job for job in jobs if job["status"] == "failed"]
    if len(failed) >= AUTO_TRAIN_MAX_ATTEMPTS:
        return True
    return any(
        job["finished_at"] is None
        or job["database_now"] < job["finished_at"] + AUTO_TRAIN_RETRY_DELAY
        for job in failed
    )


def queue_training_job(
    trigger: str,
    corpus_snapshot: list[dict[str, Any]],
    category_snapshot: set[str],
    watermark: int,
) -> dict[str, Any]:
    if trigger not in {"manual", "automatic", "initial"}:
        raise ValueError("Unsupported training trigger")
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(84327121)")
        existing = db.execute(
            "SELECT id,status FROM training_jobs WHERE status IN ('queued','running') ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if existing:
            return {"id": existing["id"], "status": existing["status"]}
        job_id = db.execute(
            """INSERT INTO training_jobs(id,status,trigger,corpus_snapshot,category_snapshot,annotation_watermark)
               VALUES (%s,'queued',%s,%s,%s,%s) RETURNING id,status""",
            (uuid.uuid4(), trigger, Jsonb(corpus_snapshot), Jsonb(sorted(category_snapshot)), watermark),
        ).fetchone()
        return job_id


def save_model_version(
    artifact: bytes,
    metrics: dict[str, Any],
    corpus_snapshot: list[dict[str, Any]],
    category_snapshot: list[str],
    annotation_revision_watermark: int,
    job_id: Any,
    *,
    parent_model_id: int | None = None,
) -> tuple[int, int, bool]:
    digest = hashlib.sha256(artifact).hexdigest()
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(84327122)")
        existing_job = db.execute(
            "SELECT model_id,model_version FROM training_jobs WHERE id=%s",
            (job_id,),
        ).fetchone()
        if existing_job is None:
            raise RuntimeError("Training job disappeared before model save")
        if existing_job["model_id"] is not None:
            row = db.execute(
                "SELECT active FROM models WHERE id=%s",
                (existing_job["model_id"],),
            ).fetchone()
            if row is None:
                raise RuntimeError("Training job references a missing model")
            return int(existing_job["model_id"]), int(existing_job["model_version"]), bool(row["active"])
        previous = db.execute(
            "SELECT id,version_no FROM models WHERE model_series='ru-pii' ORDER BY version_no DESC LIMIT 1"
        ).fetchone()
        version_no = int(previous["version_no"]) + 1 if previous else 1
        activate = previous is None
        parent_id = previous["id"] if previous else None
        if parent_model_id is not None:
            parent = db.execute(
                "SELECT model_series FROM models WHERE id=%s",
                (parent_model_id,),
            ).fetchone()
            if parent is None or parent["model_series"] != "ru-pii":
                raise RuntimeError("Warm-start parent model is missing or incompatible")
            parent_id = parent_model_id
        if activate:
            db.execute("UPDATE models SET active=FALSE WHERE active=TRUE")
        inserted = db.execute(
            """INSERT INTO models(artifact,artifact_version,artifact_sha256,metrics,active,
                       model_series,version_no,parent_model_id,corpus_snapshot,category_snapshot,annotation_watermark)
               VALUES (%s,1,%s,%s,%s,'ru-pii',%s,%s,%s,%s,%s) RETURNING id""",
            (artifact, digest, Jsonb(metrics), activate, version_no, parent_id,
             Jsonb(corpus_snapshot), Jsonb(category_snapshot), annotation_revision_watermark),
        ).fetchone()
        model_id = int(inserted["id"])
        db.execute(
            "UPDATE training_jobs SET model_id=%s,model_version=%s WHERE id=%s",
            (model_id, version_no, job_id),
        )
        return model_id, version_no, activate


def activate_model(model_id: int) -> bool:
    with connect() as db:
        row = db.execute("SELECT id FROM models WHERE id=%s", (model_id,)).fetchone()
        if row is None:
            return False
        db.execute("SELECT pg_advisory_xact_lock(84327122)")
        db.execute("UPDATE models SET active=FALSE WHERE active=TRUE")
        db.execute("UPDATE models SET active=TRUE WHERE id=%s", (model_id,))
    return True
