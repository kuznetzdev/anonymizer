from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

CODE = Path(__file__).resolve().parents[2]
ROOT = CODE.parent
BACKEND = CODE / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(CODE / "scripts" / "ner_1_17"))

import numpy
import scipy
import sklearn
from app.model import decode_artifact, normalize_legacy_prediction_labels, predict, train_model
from app.training.corpus import load_ontology_labels
from app.category_taxonomy import DOCUMENT_LABELS
from run_dev import read_jsonl, score

REFERENCES = ROOT / "references"
RELEASE = REFERENCES / "corpora" / "anonymizer-ner-hardcases" / "1.18.0"
OUTPUT = REFERENCES / "evaluations" / "medical-1.18"
COMMON_TRAIN = REFERENCES / "expanded_dataset" / "train.jsonl"
COMMON_VALIDATION = REFERENCES / "expanded_dataset" / "val.jsonl"
MEDICAL_VALIDATION = REFERENCES / "corpora" / "anonymizer-ner-hardcases" / "1.16.0" / "validation.jsonl"
BASELINE_16 = REFERENCES / "evaluations" / "medical-1.16-dev" / "candidate_1_16.npz"
BASELINE_17 = REFERENCES / "evaluations" / "medical-1.17" / "candidate_1_17.npz"
PROTOCOL = CODE / "docs" / "ner-1.18" / "EVALUATION_PROTOCOL.md"
MODEL_SOURCE = BACKEND / "app" / "model.py"
SCORER_SOURCE = CODE / "scripts" / "ner_1_17" / "run_dev.py"
TRAIN_RUNNER = Path(__file__).resolve()
BUILD_RUNNER = CODE / "scripts" / "ner_1_18" / "build_release.py"
LEGACY_LABELS = frozenset({
    "APART", "COUNTRY", "DISEASE", "EMAIL", "HOUSE", "INN", "INSTITUTION", "KPP",
    "LOCALITY", "OGRN", "ORG", "PASSPORT", "PER", "PHONE", "REGION", "SNILS", "STREET",
})
PRIORITY_LABELS = frozenset({
    "APART", "COUNTRY", "DISEASE", "HOUSE", "INSTITUTION", "LOCALITY", "ORG",
    "PASSPORT", "PER", "REGION", "SNILS", "STREET",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(baseline: dict[str, object], candidate: dict[str, object], *, medical: bool) -> dict[str, object]:
    base_f1 = cast(float, baseline["exact_span"]["micro_f1"])
    new_f1 = cast(float, candidate["exact_span"]["micro_f1"])
    base_leak = cast(float, baseline["mask_coverage"]["document_leak_rate"])
    new_leak = cast(float, candidate["mask_coverage"]["document_leak_rate"])
    base_cov = cast(dict[str, dict[str, float]], baseline["mask_coverage"]["per_label_coverage_any_class"])
    new_cov = cast(dict[str, dict[str, float]], candidate["mask_coverage"]["per_label_coverage_any_class"])
    coverage_deltas = {
        label: new_cov.get(label, {}).get("recall", 0.0) - base_cov.get(label, {}).get("recall", 0.0)
        for label in sorted(PRIORITY_LABELS)
    }
    checks = {
        "exact_f1_delta_at_least_minus_0_01": new_f1 - base_f1 >= -0.01,
        "leak_delta_at_most_plus_0_01": new_leak - base_leak <= 0.01,
        "priority_coverage_deltas_at_least_minus_0_02": all(delta >= -0.02 for delta in coverage_deltas.values()),
    }
    if medical:
        checks["medical_leak_improves_at_least_0_01"] = new_leak <= base_leak - 0.01
        old_snils = base_cov.get("SNILS", {}).get("recall", 0.0)
        new_snils = new_cov.get("SNILS", {}).get("recall", 0.0)
        checks["snils_improves_at_least_0_02_and_reaches_45_of_49"] = (
            new_snils >= old_snils + 0.02 and new_snils >= 45 / 49
            and new_cov.get("SNILS", {}).get("support") == 49
        )
    return {"checks": checks, "passed": all(checks.values()), "exact_f1_delta": new_f1 - base_f1,
            "document_leak_delta": new_leak - base_leak, "priority_coverage_deltas": coverage_deltas}


def _masking_aware_score(rows: list[dict[str, object]], artifact: bytes, labels: set[str]) -> dict[str, object]:
    metrics = score(rows, artifact, labels)
    pii_rows = [{"text": row["text"], "entities": [entity for entity in row["entities"] if entity["label"] != "DISEASE"]} for row in rows]
    pii = score(pii_rows, artifact, labels - {"DISEASE"})
    metrics["preserve_diseases_masking"] = {
        "preserve_diseases": True,
        "DISEASE_spans_remain_in_exact_span_metrics": True,
        "pii_document_leak_rate_excludes_disease_only_documents": pii["mask_coverage"]["document_leak_rate"],
        "pii_documents_with_gold": pii["mask_coverage"]["documents_with_gold"],
    }
    return metrics


def _records(path: Path, labels: set[str]) -> list[dict[str, object]]:
    return read_jsonl(path, labels)


def _per_label_exact(rows: list[dict[str, object]], artifact: bytes, labels: set[str]) -> dict[str, dict[str, float | int]]:
    counts = {label: {"true_positive": 0, "false_positive": 0, "false_negative": 0} for label in labels}
    for row in rows:
        text = str(row["text"])
        gold = {(int(entity["start"]), int(entity["end"]), str(entity["label"])) for entity in row["entities"]}
        predicted = {(int(span["start"]), int(span["end"]), str(span["label"]))
                     for span in normalize_legacy_prediction_labels(predict(text, artifact))
                     if span["label"] in labels}
        for label in labels:
            gold_for_label = {item for item in gold if item[2] == label}
            predicted_for_label = {item for item in predicted if item[2] == label}
            counts[label]["true_positive"] += len(gold_for_label & predicted_for_label)
            counts[label]["false_positive"] += len(predicted_for_label - gold_for_label)
            counts[label]["false_negative"] += len(gold_for_label - predicted_for_label)
    result: dict[str, dict[str, float | int]] = {}
    for label, values in counts.items():
        true_positive = values["true_positive"]
        false_positive = values["false_positive"]
        false_negative = values["false_negative"]
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        result[label] = {**values, "precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return dict(sorted(result.items()))


def _parent_document_exact(rows: list[dict[str, object]], artifact: bytes) -> dict[str, int | float]:
    gold: set[tuple[int, int, int]] = set()
    predicted: set[tuple[int, int, int]] = set()
    for row_index, row in enumerate(rows):
        gold.update((row_index, int(entity["start"]), int(entity["end"]))
                    for entity in row["entities"] if entity["label"] in DOCUMENT_LABELS)
        predicted.update((row_index, int(span["start"]), int(span["end"]))
                         for span in normalize_legacy_prediction_labels(predict(str(row["text"]), artifact))
                         if span["label"] in DOCUMENT_LABELS)
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return {
        "gold_spans": len(gold), "predicted_spans": len(predicted),
        "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def run() -> dict[str, object]:
    artifact_path = OUTPUT / "candidate_1_18.npz"
    report_path = OUTPUT / "candidate_1_18_dev.json"
    if OUTPUT.exists() or artifact_path.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot 1.18 evaluation directory: {OUTPUT}")
    inputs = {
        "common_train": COMMON_TRAIN,
        "common_validation": COMMON_VALIDATION,
        "medical_1_16_validation": MEDICAL_VALIDATION,
        "release_train": RELEASE / "train.jsonl",
        "release_validation": RELEASE / "validation.jsonl",
        "authored_train": RELEASE / "authored_train.jsonl",
        "authored_validation": RELEASE / "authored_validation.jsonl",
        "authored_case_index": RELEASE / "authored_cases.json",
        "quality_report": RELEASE / "quality_report.json",
        "corrections": RELEASE / "corrections.json",
        "authored_review": RELEASE / "authored_review.md",
        "ontology": CODE / "data" / "corpora" / "ru-pii" / "v1" / "ontology.json",
        "baseline_1_16": BASELINE_16,
        "baseline_1_17": BASELINE_17,
        "model": MODEL_SOURCE,
        "scorer": SCORER_SOURCE,
        "runner": TRAIN_RUNNER,
        "builder": BUILD_RUNNER,
        "corpus_importer": BACKEND / "app" / "training" / "corpus.py",
        "mixed_cases": Path(__file__).with_name("mixed_cases.py"),
        "numeric_cases": Path(__file__).with_name("numeric_cases.py"),
        "taxonomy": BACKEND / "app" / "category_taxonomy.py",
        "protocol": PROTOCOL,
        "release_manifest": RELEASE / "manifest.json",
        "release_zip": RELEASE / "anonymizer-ner-hardcases-1.18.0_import.zip",
    }
    before = {name: sha256(path) for name, path in inputs.items()}
    ontology_labels = load_ontology_labels(inputs["ontology"])
    labels = set(ontology_labels - {"ADDR"})
    if len(labels) != 22 or "PATIENT_CARD_ID" in labels:
        raise ValueError("Expected the frozen 22-label ontology")
    manifest = json.loads(inputs["release_manifest"].read_text(encoding="utf-8"))
    if manifest["corpus_id"] != "anonymizer-ner-hardcases" or manifest["corpus_version"] != "1.18.0" or set(manifest["labels"]) != labels:
        raise ValueError("1.18 release manifest identity or labels changed")
    for split in ("train", "validation"):
        if manifest["split_sha256"][split] != sha256(RELEASE / f"{split}.jsonl"):
            raise ValueError(f"1.18 manifest hash mismatch for {split}")
    train = _records(inputs["common_train"], labels) + _records(inputs["release_train"], labels)
    common_validation = _records(inputs["common_validation"], labels)
    medical_validation = _records(inputs["medical_1_16_validation"], labels)
    authored_validation = _records(inputs["authored_validation"], labels)
    release_validation = _records(inputs["release_validation"], labels)
    case_index = json.loads(inputs["authored_case_index"].read_text(encoding="utf-8"))
    numeric_negative_ids = {
        row["id"] for row in case_index
        if row["source"] == "numeric_cases.py" and row["split"] == "validation" and not row["entities"]
    }
    numeric_negative_rows = [
        row for item in case_index
        if item["id"] in numeric_negative_ids
        for row in [{"text": item["text"], "entities": item["entities"]}]
    ]
    baseline_16 = inputs["baseline_1_16"].read_bytes()
    baseline_17 = inputs["baseline_1_17"].read_bytes()
    for name, content in (("1.16", baseline_16), ("1.17", baseline_17)):
        version = decode_artifact(content)[4]
        if version != 3:
            raise ValueError(f"Baseline {name} artifact has unsupported feature version {version}")
    if before != {name: sha256(path) for name, path in inputs.items()}:
        raise ValueError("A preregistered input changed during preflight")

    # Freeze all validated inputs before the one-shot training call.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    snapshot = OUTPUT / "snapshot"
    snapshot.mkdir()
    for name, source in inputs.items():
        shutil.copyfile(source, snapshot / f"{name}{source.suffix}")
    snapshot_hashes = {path.stem: sha256(path) for path in snapshot.iterdir()}
    if snapshot_hashes != before:
        raise ValueError("Pre-training source snapshot does not match pinned input hashes")
    (OUTPUT / "input_manifest.json").write_text(json.dumps({
        "sha256": before,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "protocol_sha256": before["protocol"],
        "snapshots": {path.name: sha256(path) for path in snapshot.iterdir()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        candidate, training_metrics = train_model(labels, train, common_validation + release_validation, feature_version=3)
    except Exception as exc:
        (OUTPUT / "run_status.json").write_text(json.dumps({"status": "FAILED", "stage": "training", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    elapsed = time.monotonic() - started

    artifacts = {"1.16": baseline_16, "1.17": baseline_17, "1.18": candidate}
    try:
        splits = {"common_validation": common_validation, "medical_1_16_validation": medical_validation}
        per_model: dict[str, dict[str, object]] = {}
        for model_name, artifact in artifacts.items():
            per_model[model_name] = {
                split_name: {
                    "legacy_17_labels": _masking_aware_score(rows, artifact, set(LEGACY_LABELS)),
                    "all_22_labels_raw_prediction": _masking_aware_score(rows, artifact, labels),
                    "parent_DOCUMENT_exact_span": _parent_document_exact(rows, artifact),
                }
                for split_name, rows in splits.items()
            }
        authored_scores = {
            model_name: {
                **_masking_aware_score(authored_validation, artifact, labels),
                "per_label_exact_span": _per_label_exact(authored_validation, artifact, labels),
                "parent_DOCUMENT_exact_span": _parent_document_exact(authored_validation, artifact),
                "numeric_negative_validation": _masking_aware_score(numeric_negative_rows, artifact, labels),
            }
            for model_name, artifact in artifacts.items()
        }
        release_validation_scores = {
            model_name: _masking_aware_score(release_validation, artifact, labels)
            for model_name, artifact in artifacts.items()
        }
        gates = {
            split_name: gate(per_model["1.16"][split_name]["legacy_17_labels"], per_model["1.18"][split_name]["legacy_17_labels"], medical=split_name == "medical_1_16_validation")
            for split_name in splits
        }
        development_passed = all(result["passed"] for result in gates.values())
        status = "HOLD"  # No fresh independent test/challenge set exists for all 22 labels.
    except Exception as exc:
        (OUTPUT / "run_status.json").write_text(json.dumps({"status": "FAILED", "stage": "scoring", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise

    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        (OUTPUT / "run_status.json").write_text(json.dumps({"status": "FAILED", "stage": "input_integrity", "error": "A frozen training or evaluation input changed during the run"}, indent=2) + "\n", encoding="utf-8")
        raise ValueError("A frozen training or evaluation input changed during the run")
    artifact_path.write_bytes(candidate)
    report = {
        "status": status,
        "candidate": "1.18.0 development candidate; never activated automatically",
        "purpose": "one-shot development run; no test or challenge split opened",
        "candidate_artifact_sha256": sha256(artifact_path),
        "input_sha256": before,
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "training": {"entrypoint": "app.model.train_model", "feature_version": 3, "epochs": 3, "random_state": 17, "loss": "log_loss", "alpha": 1e-5, "average": True, "batch_size": 4096, "warm_start": False, "record_count_before_deduplication": len(train), "elapsed_seconds": elapsed, "reported_metrics": training_metrics},
        "baseline_1_16_and_1_17_vs_candidate": per_model,
        "curated_22_label_validation": {"support_and_metrics": authored_scores},
        "full_1_18_validation": release_validation_scores,
        "development_gate_against_1_16": gates,
        "development_gate_passed": development_passed,
        "independent_evaluation": {"ready": False, "reason": "No fresh independent test/challenge corpus covers the expanded 22-label ontology."},
        "decision": "HOLD until development and independent evaluation gates pass.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the single frozen 1.18 development evaluation.")
    parser.add_argument("--source-reviewed", action="store_true", help="Required acknowledgement that authored source cases are reviewed and frozen.")
    args = parser.parse_args()
    if not args.source_reviewed:
        raise SystemExit("Refusing to train before the parent reviewer confirms authored cases are frozen; pass --source-reviewed after review.")
    result = run()
    print(json.dumps({"status": result["status"], "candidate_artifact_sha256": result["candidate_artifact_sha256"], "development_gate_passed": result["development_gate_passed"]}, indent=2))
