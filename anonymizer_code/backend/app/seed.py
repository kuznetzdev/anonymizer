"""Load and validate the public, pre-trained model seed."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from .model import FEATURES, LABEL_RE, decode_artifact

SEED_MANIFEST = Path(__file__).with_name("artifacts") / "ru-pii-base-0.1.0.json"
SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
CORPUS_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
METRIC_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
BIO_CLASS_RE = re.compile(r"^([BI])-([A-Z][A-Z0-9_]{0,31})$")


class SeedArtifactError(ValueError):
    """The bundled seed manifest or model artifact is invalid."""


@dataclass(frozen=True)
class SeedModel:
    artifact: bytes
    sha256: str
    version: str
    classes: tuple[str, ...]
    categories: tuple[str, ...]
    metrics: dict[str, object]
    training: dict[str, object] | None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SeedArtifactError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SeedArtifactError(f"Unsupported JSON value in seed manifest: {value}")


def _json_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SeedArtifactError(f"Seed manifest field '{field}' must be a JSON object")
    return cast(dict[str, object], value)


def _metrics_object(value: object, field: str) -> dict[str, object]:
    metrics = _json_object(value, field)

    def valid_metric(item: object) -> bool:
        if isinstance(item, bool):
            return False
        if isinstance(item, int):
            return True
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, dict):
            return all(
                isinstance(key, str) and METRIC_KEY_RE.fullmatch(key) and valid_metric(nested)
                for key, nested in item.items()
            )
        return False

    if not all(METRIC_KEY_RE.fullmatch(key) and valid_metric(item) for key, item in metrics.items()):
        raise SeedArtifactError(f"Seed manifest field '{field}' must contain only numeric metrics")
    return metrics


def _training_metadata(value: object) -> dict[str, object]:
    training = _json_object(value, "training")
    if set(training) != {"corpus_id", "corpus_version", "split_counts", "metrics"}:
        raise SeedArtifactError(
            "Seed manifest 'training' must contain only corpus_id, corpus_version, split_counts, and metrics"
        )
    corpus_id = training["corpus_id"]
    corpus_version = training["corpus_version"]
    if not isinstance(corpus_id, str) or not CORPUS_ID_RE.fullmatch(corpus_id):
        raise SeedArtifactError("Seed training corpus_id must be a safe identifier")
    if not isinstance(corpus_version, str) or not SEMVER_RE.fullmatch(corpus_version):
        raise SeedArtifactError("Seed training corpus_version must use semantic versioning")
    split_counts = _json_object(training["split_counts"], "training.split_counts")
    if set(split_counts) != {"train", "validation", "test"} or any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in split_counts.values()
    ):
        raise SeedArtifactError("Seed training split_counts must contain positive train/validation/test counts")
    training_metrics = _metrics_object(training["metrics"], "training.metrics")
    return {
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "split_counts": split_counts,
        "metrics": training_metrics,
    }


def _classes_and_categories(value: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise SeedArtifactError("Seed manifest 'classes' must be a non-empty string array")
    classes = tuple(value)
    if len(set(classes)) != len(classes) or "O" not in classes:
        raise SeedArtifactError("Seed artifact classes must be unique and include 'O'")
    categories: set[str] = set()
    beginnings: set[str] = set()
    for label in classes:
        if label == "O":
            continue
        match = BIO_CLASS_RE.fullmatch(label)
        if match is None or not LABEL_RE.fullmatch(match.group(2)):
            raise SeedArtifactError(f"Invalid BIO class in seed artifact: {label!r}")
        prefix, category = match.groups()
        categories.add(category)
        if prefix == "B":
            beginnings.add(category)
    if not categories or categories != beginnings:
        raise SeedArtifactError("Each seed category must have a B- class and BIO classes must be valid")
    return classes, tuple(sorted(categories))


def load_seed_model(manifest_path: Path = SEED_MANIFEST) -> SeedModel:
    """Load the paired NPZ and JSON files without reading any training corpus."""
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(
            raw_manifest.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SeedArtifactError) as exc:
        raise SeedArtifactError(f"Cannot read seed manifest {manifest_path.name}: {exc}") from exc
    data = _json_object(manifest, "root")
    required = {
        "model_series", "version", "artifact_version", "tokenizer_version",
        "sha256", "classes", "categories", "metrics",
    }
    if not required.issubset(data) or set(data) - (required | {"training"}):
        missing = sorted(required - set(data))
        unexpected = sorted(set(data) - (required | {"training"}))
        raise SeedArtifactError(f"Invalid seed manifest fields (missing={missing}, unexpected={unexpected})")
    if data["model_series"] != "ru-pii":
        raise SeedArtifactError("Seed model_series must be 'ru-pii'")
    version = data["version"]
    if not isinstance(version, str) or not SEMVER_RE.fullmatch(version):
        raise SeedArtifactError("Seed version must use semantic versioning")
    if data["artifact_version"] != 2 or isinstance(data["artifact_version"], bool):
        raise SeedArtifactError("Seed artifact_version must be 2")
    if data["tokenizer_version"] != 2 or isinstance(data["tokenizer_version"], bool):
        raise SeedArtifactError("Seed tokenizer_version must be 2")
    digest = data["sha256"]
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise SeedArtifactError("Seed manifest sha256 must be a lowercase SHA-256 digest")
    declared_classes, declared_categories = _classes_and_categories(data["classes"])
    categories = data["categories"]
    if (not isinstance(categories, list) or any(not isinstance(label, str) for label in categories)
            or len(set(categories)) != len(categories) or set(categories) != set(declared_categories)):
        raise SeedArtifactError("Seed manifest categories must exactly match the categories in BIO classes")
    metrics = _metrics_object(data["metrics"], "metrics")
    if "seed_model" in metrics:
        raise SeedArtifactError("Seed metrics cannot use the reserved 'seed_model' key")
    training = _training_metadata(data["training"]) if "training" in data else None

    artifact_path = manifest_path.with_suffix(".npz")
    try:
        artifact = artifact_path.read_bytes()
    except OSError as exc:
        raise SeedArtifactError(f"Cannot read seed artifact {artifact_path.name}: {exc}") from exc
    actual_digest = hashlib.sha256(artifact).hexdigest()
    if actual_digest != digest:
        raise SeedArtifactError("Seed artifact SHA-256 does not match its manifest")
    try:
        hasher, artifact_classes, weights, intercept, artifact_version = decode_artifact(artifact)
    except (ValueError, KeyError, OSError, EOFError, TypeError, IndexError, OverflowError) as exc:
        raise SeedArtifactError(f"Cannot decode seed NPZ artifact: {exc}") from exc
    if artifact_version != 2:
        raise SeedArtifactError("Seed NPZ must use artifact version 2")
    if tuple(artifact_classes) != declared_classes:
        raise SeedArtifactError("Seed manifest classes do not match the NPZ class order")
    if (hasher.n_features != FEATURES
            or weights.ndim != 2 or weights.shape != (len(artifact_classes), FEATURES)
            or intercept.shape != (len(artifact_classes),)):
        raise SeedArtifactError("Seed NPZ has incompatible coefficient dimensions")
    if (len(artifact_classes) < 2 or weights.dtype.kind not in "iuf" or intercept.dtype.kind not in "iuf"
            or not np.isfinite(weights.data).all() or not np.isfinite(intercept).all()):
        raise SeedArtifactError("Seed NPZ has invalid classes or non-finite coefficients")
    return SeedModel(
        artifact=artifact,
        sha256=digest,
        version=version,
        classes=declared_classes,
        categories=declared_categories,
        metrics=metrics,
        training=training,
    )
