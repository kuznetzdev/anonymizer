from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

CODE = Path(__file__).resolve().parents[2]
ROOT = CODE.parent
BACKEND = CODE / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(CODE / "scripts" / "ner_1_17"))

import numpy
import scipy
import sklearn
from app.model import decode_artifact, train_model
from run_dev import read_jsonl, score

REFERENCES = ROOT / "references"
RELEASE = REFERENCES / "corpora" / "anonymizer-ner-hardcases" / "1.18.1"
OUTPUT = REFERENCES / "evaluations" / "medical-1.18.2-taxonomy-ablation-corrected"
BASELINE_16 = REFERENCES / "evaluations" / "medical-1.16-dev" / "candidate_1_16.npz"
MODEL_1_18_1 = REFERENCES / "evaluations" / "medical-1.18.1" / "candidate_1_1_18_1.npz"
LEGACY_LABELS = frozenset({
    "APART", "COUNTRY", "DISEASE", "EMAIL", "HOUSE", "INN", "INSTITUTION", "KPP",
    "LOCALITY", "OGRN", "ORG", "PASSPORT", "PER", "PHONE", "REGION", "SNILS", "STREET",
})
MERGED_DOCUMENT_LABELS = frozenset({
    "OMS_POLICY_ID", "MEDICAL_RECORD_ID", "REFERRAL_ID", "DRIVER_LICENSE_ID", "OTHER_DOCUMENT_ID",
})
FROZEN_LABELS = LEGACY_LABELS | MERGED_DOCUMENT_LABELS
ABLATION_LABELS = LEGACY_LABELS | {"DOCUMENT_ID"}
LABEL_MAP = {label: "DOCUMENT_ID" for label in MERGED_DOCUMENT_LABELS}
LEGACY_PREDICTION_MAP: Mapping[str, str] = LABEL_MAP


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_paths(root: Path) -> dict[str, Path]:
    """Explicitly enumerate permitted development inputs; no other split is opened."""
    references = root / "references"
    release = references / "corpora" / "anonymizer-ner-hardcases" / "1.18.1"
    return {
        "release_train": release / "train.jsonl",
        "release_validation": release / "validation.jsonl",
        "common_validation": references / "expanded_dataset" / "val.jsonl",
        "medical_1_16_validation": references / "corpora" / "anonymizer-ner-hardcases" / "1.16.0" / "validation.jsonl",
        "baseline_1_16": references / "evaluations" / "medical-1.16-dev" / "candidate_1_16.npz",
        "candidate_1_18_1": references / "evaluations" / "medical-1.18.1" / "candidate_1_1_18_1.npz",
    }


def remap_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Copy records and merge only the five sparse document subtype labels."""
    remapped: list[dict[str, object]] = []
    for row in rows:
        entities = row["entities"]
        if not isinstance(entities, list):
            raise ValueError("Record entities must be a list")
        copied_entities: list[dict[str, object]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("Record entity must be an object")
            label = entity.get("label")
            if label not in FROZEN_LABELS:
                raise ValueError(f"Unsupported source label: {label}")
            copied_entities.append({
                **entity,
                "label": LABEL_MAP.get(str(label), str(label)),
            })
        remapped.append({"text": row["text"], "entities": copied_entities})
    return remapped


def masking_aware_score(
    rows: list[dict[str, object]],
    artifact: bytes,
    labels: set[str],
    prediction_map: Mapping[str, str] | None = None,
) -> dict[str, object]:
    metrics = score(rows, artifact, labels, prediction_map)
    pii_rows = [
        {"text": row["text"], "entities": [entity for entity in row["entities"] if entity["label"] != "DISEASE"]}
        for row in rows
    ]
    pii = score(pii_rows, artifact, labels - {"DISEASE"}, prediction_map)
    metrics["preserve_diseases_masking"] = {
        "preserve_diseases": True,
        "DISEASE_spans_remain_in_exact_span_metrics": True,
        "pii_document_leak_rate_excludes_disease_only_documents": pii["mask_coverage"]["document_leak_rate"],
        "pii_documents_with_gold": pii["mask_coverage"]["documents_with_gold"],
    }
    return metrics


def _read(path: Path) -> list[dict[str, object]]:
    return remap_records(read_jsonl(path, set(FROZEN_LABELS)))


def run() -> dict[str, object]:
    paths = development_paths(ROOT)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot taxonomy ablation: {OUTPUT}")
    source_files = {
        **paths,
        "model_source": BACKEND / "app" / "model.py",
        "scorer_source": CODE / "scripts" / "ner_1_17" / "run_dev.py",
        "ablation_runner": Path(__file__).resolve(),
    }
    before = {name: sha256(path) for name, path in source_files.items()}
    baseline = paths["baseline_1_16"].read_bytes()
    current = paths["candidate_1_18_1"].read_bytes()
    if decode_artifact(baseline)[4] != 3 or decode_artifact(current)[4] != 3:
        raise ValueError("The ablation requires the frozen feature-version-3 models")
    train = _read(paths["release_train"])
    corrected_validation = _read(paths["release_validation"])
    common_validation = _read(paths["common_validation"])
    medical_validation = _read(paths["medical_1_16_validation"])
    if before != {name: sha256(path) for name, path in source_files.items()}:
        raise ValueError("A registered development input changed during preflight")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    snapshot = OUTPUT / "snapshot"
    snapshot.mkdir()
    for name, source in source_files.items():
        shutil.copyfile(source, snapshot / f"{name}{source.suffix}")
    snapshot_hashes = {path.stem: sha256(path) for path in snapshot.iterdir()}
    if snapshot_hashes != before:
        raise ValueError("Frozen input snapshot does not match preflight hashes")
    (OUTPUT / "input_manifest.json").write_text(json.dumps({
        "sha256": before,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {
            "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "snapshots": {path.name: sha256(path) for path in snapshot.iterdir()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    try:
        candidate, training_metrics = train_model(set(ABLATION_LABELS), train, corrected_validation, feature_version=3)
    except Exception as exc:
        (OUTPUT / "run_status.json").write_text(json.dumps({
            "status": "FAILED", "stage": "training", "error_type": type(exc).__name__, "error": str(exc),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    elapsed = time.monotonic() - started

    splits = {
        "common_validation": common_validation,
        "medical_1_16_validation": medical_validation,
        "corrected_1_18_1_validation": corrected_validation,
    }
    artifacts = {
        "baseline_1_16": (baseline, None),
        "candidate_1_18_1_collapsed": (current, LEGACY_PREDICTION_MAP),
        "candidate_document_id": (candidate, None),
    }
    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for split_name, rows in splits.items():
        comparisons[split_name] = {}
        for model_name, (artifact, prediction_map) in artifacts.items():
            comparisons[split_name][model_name] = {
                "legacy_17_labels": masking_aware_score(rows, artifact, set(LEGACY_LABELS), prediction_map),
                "reduced_18_labels": masking_aware_score(rows, artifact, set(ABLATION_LABELS), prediction_map),
            }

    after = {name: sha256(path) for name, path in source_files.items()}
    if before != after:
        (OUTPUT / "run_status.json").write_text(json.dumps({
            "status": "FAILED", "stage": "input_integrity", "error": "A frozen input changed during the run",
        }, indent=2) + "\n", encoding="utf-8")
        raise ValueError("A frozen development input changed during training or scoring")
    artifact_path = OUTPUT / "candidate_document_id.npz"
    artifact_path.write_bytes(candidate)
    report = {
        "status": "HOLD",
        "purpose": "one-shot development taxonomy ablation; no test/challenge data or metadata opened",
        "taxonomy": {
            "retained_legacy_labels": sorted(LEGACY_LABELS),
            "merged_source_labels": sorted(MERGED_DOCUMENT_LABELS),
            "merged_label": "DOCUMENT_ID",
            "retained_separate_document_labels": ["PASSPORT", "SNILS"],
        },
        "candidate_artifact_sha256": sha256(artifact_path),
        "input_sha256": before,
        "environment": {
            "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "training": {
            "entrypoint": "app.model.train_model", "feature_version": 3, "epochs": 3,
            "random_state": 17, "loss": "log_loss", "alpha": 1e-5, "average": True,
            "batch_size": 4096, "warm_start": False, "record_count_used": len(train),
            "validation_records": len(corrected_validation), "elapsed_seconds": elapsed,
            "reported_metrics": training_metrics,
        },
        "comparisons": comparisons,
        "independent_evaluation": {
            "ready": False,
            "reason": "No fresh independent holdout validates the reduced taxonomy.",
        },
        "decision": "HOLD; development ablation does not change the official 1.18.1 release or activate a model.",
    }
    (OUTPUT / "candidate_document_id_dev.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"],
        "candidate_artifact_sha256": result["candidate_artifact_sha256"],
    }, indent=2))
