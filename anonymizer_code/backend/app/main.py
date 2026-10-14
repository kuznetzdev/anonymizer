from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from psycopg.types.json import Jsonb

from .config import config
from .model import normalize_legacy_prediction_labels, predict
from .seed import load_seed_model
from .training.corpus import CorpusImportError, install_corpus_archive, list_corpus_releases, load_ontology_descriptions
from .store import (
    activate_model, active_categories, active_model, annotation_watermark, archive_categories,
    close_pool, connect, ensure_categories, ensure_corpus_categories, example_page,
    initialize, install_seed_model, model_artifact, model_versions, new_annotations_since_last_version,
    open_pool, queue_training_job, register_corpus_release, register_corpus_releases,
    training_corpus_release_ids, training_labels,
)

LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
REASONS = {"correct", "false_positive", "wrong_label", "wrong_boundary", "missed_span", "other"}
LEGACY_CATEGORY_LABELS = {"ADDR"}
MAX_TEXT = 20000
COOKIE = "anonymizer_session"
SESSION_SECONDS = 12 * 60 * 60
app = FastAPI(title="Local Redaction API", docs_url=None, redoc_url=None)
_model_cache: tuple[int, bytes] | None = None
CORPUS_ROOT = Path(config.data_dir) / "corpora" / "ru-pii"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_category_descriptions: dict[str, str] = {}


def current_model() -> tuple[dict[str, Any], bytes] | None:
    global _model_cache
    metadata = active_model()
    if metadata is None:
        return None
    if _model_cache is None or _model_cache[0] != metadata["id"]:
        _model_cache = (metadata["id"], model_artifact(metadata["id"]))
    return metadata, _model_cache[1]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Credentials(InputModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class EntityInput(InputModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    label: str = Field(min_length=1, max_length=32)


class FeedbackInput(EntityInput):
    reason: str = Field(min_length=1, max_length=40)
    note: str = Field(default="", max_length=1000)


class TextInput(InputModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)


class ExampleInput(TextInput):
    entities: list[EntityInput] = Field(max_length=500)
    feedback: list[FeedbackInput] = Field(default_factory=list, max_length=500)
    empty_review_confirmed: bool = Field(default=False, strict=True)


class CategoryInput(InputModel):
    label: str = Field(min_length=1, max_length=32)


class CategoryState(InputModel):
    active: bool


def require_origin(request: Request) -> None:
    origin = request.headers.get("origin", "")
    expected = config.app_origin.rstrip("/")
    try:
        same_host = urlsplit(origin).netloc == urlsplit(expected).netloc
    except ValueError:
        same_host = False
    if origin.rstrip("/") != expected or not same_host:
        raise HTTPException(status_code=403, detail="Invalid request origin")


def require_auth(request: Request) -> None:
    token = request.cookies.get(COOKIE)
    if not token or len(token) > 128:
        raise HTTPException(status_code=401, detail="Authentication required")
    token_hash = hmac.new(config.session_secret.encode(), token.encode("ascii", errors="ignore"), hashlib.sha256).hexdigest()
    if not session_exists(token_hash):
        raise HTTPException(status_code=401, detail="Authentication required")


def session_exists(token_hash: str) -> bool:
    with connect() as db:
        return db.execute(
            "SELECT 1 FROM auth_sessions WHERE token_hash=%s AND expires_at > now()",
            (token_hash,),
        ).fetchone() is not None


def corpus_catalog() -> list[dict[str, object]]:
    releases = list_corpus_releases(CORPUS_ROOT)
    return [release.as_dict() for release in sorted(
        releases, key=lambda release: (release.corpus_id, release.corpus_version, release.sha256),
    )]


def corpus_snapshot() -> list[dict[str, object]]:
    catalog = corpus_catalog()
    selected_ids = training_corpus_release_ids()
    available_ids = {release["release_id"] for release in catalog}
    if selected_ids - available_ids:
        raise CorpusImportError("A selected corpus release is missing from data storage")
    return [release for release in catalog if release["release_id"] in selected_ids]


@app.on_event("startup")
def startup() -> None:
    global _category_descriptions
    config.validate()
    open_pool()
    initialize()
    install_seed_model(load_seed_model())
    _category_descriptions = load_ontology_descriptions(CORPUS_ROOT / "v1" / "ontology.json")
    ensure_categories(set(_category_descriptions))
    archive_categories(LEGACY_CATEGORY_LABELS)
    catalog = corpus_catalog()
    register_corpus_releases(catalog)
    releases = corpus_snapshot()
    if releases:
        labels = {label for release in catalog for label in release["labels"]}
        ensure_corpus_categories(labels)
    with connect() as db:
        has_series_model = db.execute(
            "SELECT 1 FROM models WHERE model_series='ru-pii' LIMIT 1",
        ).fetchone() is not None
    if releases and not has_series_model:
        queue_training_job("initial", releases, training_labels(), annotation_watermark())


@app.on_event("shutdown")
def shutdown() -> None:
    close_pool()


@app.get("/api/health")
def health() -> dict[str, str]:
    with connect() as db:
        db.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/api/auth/me")
def auth_me(request: Request) -> dict[str, bool]:
    token = request.cookies.get(COOKIE)
    authenticated = bool(token and len(token) <= 128 and session_exists(hmac.new(config.session_secret.encode(), token.encode("ascii", errors="ignore"), hashlib.sha256).hexdigest()))
    return {"authenticated": authenticated}


@app.post("/api/auth/login")
def login(body: Credentials, request: Request, response: Response) -> dict[str, bool]:
    require_origin(request)
    valid_user = hmac.compare_digest(body.username.encode(), config.username.encode())
    valid_password = hmac.compare_digest(body.password.encode(), config.password.encode())
    if not (valid_user and valid_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_urlsafe(32)
    token_hash = hmac.new(config.session_secret.encode(), token.encode("ascii"), hashlib.sha256).hexdigest()
    with connect() as db:
        db.execute("DELETE FROM auth_sessions WHERE expires_at <= now()")
        db.execute("INSERT INTO auth_sessions(token_hash, expires_at) VALUES (%s, now() + interval '12 hours')", (token_hash,))
    response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, secure=config.cookie_secure, samesite="strict", path="/")
    return {"authenticated": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    require_origin(request)
    token = request.cookies.get(COOKIE)
    if token and len(token) <= 128:
        token_hash = hmac.new(config.session_secret.encode(), token.encode("ascii", errors="ignore"), hashlib.sha256).hexdigest()
        with connect() as db:
            db.execute("DELETE FROM auth_sessions WHERE token_hash=%s", (token_hash,))
    response.delete_cookie(COOKIE, path="/", httponly=True, secure=config.cookie_secure, samesite="strict")
    return {"authenticated": False}


@app.get("/api/categories")
def categories(request: Request) -> dict[str, Any]:
    require_auth(request)
    return {"categories": [
        {
            **category,
            "description": _category_descriptions.get(category["label"]),
            "deprecated": category["label"] in LEGACY_CATEGORY_LABELS,
        }
        for category in active_categories()
    ]}


@app.post("/api/categories", status_code=201)
def add_category(body: CategoryInput, request: Request) -> dict[str, Any]:
    require_origin(request)
    require_auth(request)
    label = body.label.upper()
    if not LABEL_RE.fullmatch(label):
        raise HTTPException(status_code=422, detail="Label must use A-Z, 0-9, and underscore")
    if label in LEGACY_CATEGORY_LABELS:
        raise HTTPException(status_code=409, detail="This category is reserved for legacy corpus compatibility")
    with connect() as db:
        row = db.execute("INSERT INTO categories(label,active) VALUES (%s,TRUE) ON CONFLICT(label) DO UPDATE SET active=TRUE, updated_at=now() RETURNING label, active, built_in", (label,)).fetchone()
    return {"category": row}


@app.patch("/api/categories/{label}")
def set_category(label: str, body: CategoryState, request: Request) -> dict[str, Any]:
    require_origin(request)
    require_auth(request)
    if label in LEGACY_CATEGORY_LABELS and body.active:
        raise HTTPException(status_code=409, detail="This legacy category cannot be restored for new annotations")
    with connect() as db:
        row = db.execute("UPDATE categories SET active=%s, updated_at=now() WHERE label=%s RETURNING label, active, built_in", (body.active, label)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"category": row}


@app.get("/api/status")
def status(request: Request) -> dict[str, Any]:
    require_auth(request)
    with connect() as db:
        saved = db.execute(
            """SELECT count(*) AS n FROM examples e
               WHERE (jsonb_array_length(e.entities)>0 OR e.empty_review_confirmed)
                 AND NOT EXISTS (
                   SELECT 1 FROM jsonb_array_elements(e.entities) entity
                   LEFT JOIN categories c ON c.label=entity->>'label'
                   WHERE c.label IS NULL
               )"""
        ).fetchone()["n"]
        job = db.execute("SELECT id, status, error, metrics, trigger, model_version, created_at FROM training_jobs ORDER BY created_at DESC LIMIT 1").fetchone()
    model = active_model()
    metrics = model["metrics"] if model else None
    releases = corpus_snapshot()
    watermark = annotation_watermark()
    corpus_counts = {
        "releases": len(releases),
        "train": sum(item["record_counts"]["train"] for item in releases),
        "validation": sum(item["record_counts"]["validation"] for item in releases),
        "test": sum(item["record_counts"]["test"] for item in releases),
    }
    return {
        "categories": active_categories(),
        "corpus": corpus_counts,
        "saved_examples": saved,
        "new_examples_since_training": new_annotations_since_last_version(watermark),
        "auto_train_min_new_examples": config.auto_train_min_new_examples,
        "model": {
            "ready": model is not None, "metrics": metrics,
            "trained_at": model["trained_at"].isoformat() if model else None,
            "series": model["model_series"] if model else None,
            "version": model["version_no"] if model else None,
            "id": model["id"] if model else None,
        },
        "training_job": {**job, "id": str(job["id"]), "created_at": job["created_at"].isoformat()} if job else None,
    }


@app.get("/api/corpora")
def get_corpora(request: Request) -> dict[str, Any]:
    require_auth(request)
    return {"releases": corpus_catalog(), "training_releases": corpus_snapshot()}


@app.post("/api/corpora", status_code=201)
async def upload_corpus(request: Request) -> dict[str, Any]:
    require_origin(request)
    require_auth(request)
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/zip":
        raise HTTPException(status_code=415, detail="Upload a ZIP archive with Content-Type application/zip")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_ARCHIVE_BYTES:
                raise HTTPException(status_code=413, detail="Corpus archive exceeds 64 MiB")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="corpus-", suffix=".zip", delete=False) as archive:
            temp_path = Path(archive.name)
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise HTTPException(status_code=413, detail="Corpus archive exceeds 64 MiB")
                archive.write(chunk)
        known_labels = {row["label"] for row in active_categories()}
        release = install_corpus_archive(temp_path, CORPUS_ROOT, known_labels)
    except CorpusImportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="This corpus release is already installed") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    ensure_corpus_categories(set(release.labels))
    register_corpus_release(release.corpus_id, release.corpus_version, release.release_id)
    with connect() as db:
        has_series_model = db.execute(
            "SELECT 1 FROM models WHERE model_series='ru-pii' LIMIT 1",
        ).fetchone() is not None
    initial_job = None
    if not has_series_model:
        initial_job = queue_training_job(
            "initial", corpus_snapshot(), training_labels(), annotation_watermark(),
        )
    return {
        "release": release.as_dict(),
        "training_job": {"id": str(initial_job["id"]), "status": initial_job["status"]} if initial_job else None,
    }


@app.get("/api/models")
def get_models(request: Request) -> dict[str, Any]:
    require_auth(request)
    return {
        "versions": [
            {**version, "trained_at": version["trained_at"].isoformat()}
            for version in model_versions()
        ],
    }


@app.post("/api/models/{model_id}/activate")
def set_active_model(model_id: int, request: Request) -> dict[str, bool]:
    require_origin(request)
    require_auth(request)
    if not activate_model(model_id):
        raise HTTPException(status_code=404, detail="Model version not found")
    return {"activated": True}


@app.get("/api/examples")
def list_examples(
    request: Request,
    category: str | None = Query(default=None, min_length=1, max_length=32),
    before_id: int | None = Query(default=None, gt=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    require_auth(request)
    result = example_page(category, before_id, limit)
    if result is None:
        raise HTTPException(status_code=404, detail="Category not found")
    items, next_cursor = result
    return {
        "items": [
            {**item, "created_at": item["created_at"].isoformat()}
            for item in items
        ],
        "next_cursor": next_cursor,
    }


@app.post("/api/analyze")
def analyze(body: TextInput, request: Request) -> dict[str, Any]:
    require_auth(request)
    model = current_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    active = {row["label"] for row in active_categories() if row["active"]}
    spans = [
        span for span in normalize_legacy_prediction_labels(predict(body.text, model[1]))
        if span["label"] in active
    ]
    spans.sort(key=lambda span: (span["start"], span["end"]))
    entities = [{"id": str(uuid.uuid4()), **span, "text": body.text[span["start"]:span["end"]]} for span in spans]
    masked = body.text
    for span in reversed(spans):
        masked = masked[:span["start"]] + f"<{span['label']}>" + masked[span["end"]:]
    return {"masked_text": masked, "entities": entities}


def validate_spans(text: str, entities: list[EntityInput], feedback: list[FeedbackInput] = ()) -> None:
    known = {row["label"] for row in active_categories()}
    ordered = sorted(entities, key=lambda span: (span.start, span.end))
    previous_end = -1
    for span in ordered:
        if span.end > len(text) or span.start >= span.end:
            raise HTTPException(status_code=422, detail="Entity offsets are outside the text")
        if span.start < previous_end:
            raise HTTPException(status_code=422, detail="Entity spans must not overlap")
        if span.label not in known:
            raise HTTPException(status_code=422, detail="Entity label does not exist")
        previous_end = span.end
    for item in feedback:
        if item.end > len(text) or item.start >= item.end or item.reason not in REASONS:
            raise HTTPException(status_code=422, detail="Invalid feedback span or reason")
        if item.label not in known:
            raise HTTPException(status_code=422, detail="Feedback label does not exist")


@app.post("/api/examples", status_code=201)
def save_example(body: ExampleInput, request: Request, response: Response) -> dict[str, Any]:
    require_origin(request)
    require_auth(request)
    if not body.entities and not body.empty_review_confirmed:
        raise HTTPException(status_code=422, detail="Confirm that the text contains no entities")
    if body.entities and body.empty_review_confirmed:
        raise HTTPException(status_code=422, detail="Empty review confirmation requires no entities")
    validate_spans(body.text, body.entities, body.feedback)
    normalized = sorted(body.entities, key=lambda span: (span.start, span.end))
    entities = [{"start": item.start, "end": item.end, "label": item.label} for item in normalized]
    digest = hashlib.sha256(body.text.encode("utf-8")).hexdigest()
    feedback = sorted((issue.start, issue.end, issue.label, issue.reason, issue.note) for issue in body.feedback)
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(84327123)")
        inserted = db.execute(
            "INSERT INTO examples(text, text_hash, entities, empty_review_confirmed) VALUES (%s,%s,%s,%s) ON CONFLICT(text_hash) DO NOTHING RETURNING id",
            (body.text, digest, Jsonb(entities), body.empty_review_confirmed),
        ).fetchone()
        if inserted:
            example_id = inserted["id"]
            result = "created"
        else:
            existing = db.execute(
                "SELECT id, text, entities, empty_review_confirmed FROM examples WHERE text_hash=%s FOR UPDATE",
                (digest,),
            ).fetchone()
            if existing is None or existing["text"] != body.text:
                raise HTTPException(status_code=409, detail="Text hash collision")
            example_id = existing["id"]
            old_entities = sorted(
                (item["start"], item["end"], item["label"]) for item in existing["entities"]
            )
            old_feedback = sorted((
                row["span_start"], row["span_end"], row["label"], row["reason"], row["note"]
            ) for row in db.execute(
                "SELECT span_start, span_end, label, reason, note FROM feedback WHERE example_id=%s",
                (example_id,),
            ).fetchall())
            new_entities = sorted((item["start"], item["end"], item["label"]) for item in entities)
            if (old_entities == new_entities and old_feedback == feedback
                    and existing["empty_review_confirmed"] == body.empty_review_confirmed):
                response.status_code = 200
                return {"id": example_id, "saved": True, "status": "unchanged"}
            db.execute(
                "UPDATE examples SET entities=%s, empty_review_confirmed=%s, updated_at=now() WHERE id=%s",
                (Jsonb(entities), body.empty_review_confirmed, example_id),
            )
            db.execute("DELETE FROM feedback WHERE example_id=%s", (example_id,))
            result = "updated"
        db.execute(
            "INSERT INTO example_revisions(example_id,text,entities,empty_review_confirmed) VALUES (%s,%s,%s,%s)",
            (example_id, body.text, Jsonb(entities), body.empty_review_confirmed),
        )
        if feedback:
            with db.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO feedback(example_id,span_start,span_end,label,reason,note) VALUES (%s,%s,%s,%s,%s,%s)",
                    [(example_id, *item) for item in feedback],
                )
        response.status_code = 201 if result == "created" else 200
    return {"id": example_id, "saved": True, "status": result}


@app.post("/api/train", status_code=202)
def queue_training(request: Request) -> dict[str, str]:
    require_origin(request)
    require_auth(request)
    releases = corpus_snapshot()
    if not releases:
        raise HTTPException(status_code=409, detail="Upload a base corpus before training")
    if not sum(item["record_counts"]["train"] for item in releases):
        raise HTTPException(status_code=409, detail="The selected corpus has no training records")
    if not sum(item["record_counts"]["validation"] for item in releases):
        raise HTTPException(status_code=409, detail="The selected corpus has no validation records")
    with connect() as db:
        category_count = db.execute("SELECT count(*) AS n FROM categories").fetchone()["n"]
        if category_count == 0:
            raise HTTPException(status_code=409, detail="At least one category must exist before training")
    job = queue_training_job("manual", releases, training_labels(), annotation_watermark())
    return {"job_id": str(job["id"]), "status": job["status"]}


@app.get("/api/training/{job_id}")
def get_training(job_id: uuid.UUID, request: Request) -> dict[str, Any]:
    require_auth(request)
    with connect() as db:
        row = db.execute("SELECT id,status,error,metrics,trigger,model_id,model_version,created_at,started_at,finished_at FROM training_jobs WHERE id=%s", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Training job not found")
    return {**row, "id": str(row["id"]), **{key: value.isoformat() if isinstance(value, datetime) else value for key, value in row.items() if key.endswith("_at")}}
