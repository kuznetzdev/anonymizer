from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "anonymizer_code" / "backend"))

import numpy
import scipy
import sklearn
from app.model import train_model
from run_dev import EXPECTED_BASELINE_SHA256, EXPECTED_DATA_SHA256, read_jsonl, score, sha256

REFERENCES = ROOT / "references"
EXPANDED = REFERENCES / "expanded_dataset"
RELEASE = REFERENCES / "corpora" / "anonymizer-ner-hardcases"
BASE_RELEASE = RELEASE / "1.16.0"
CANDIDATE_RELEASE = RELEASE / "1.17.0-integrated" / "scenario-build-v1"
ARTIFACT_PATH = REFERENCES / "evaluations" / "medical-1.17" / "candidate_1_17.npz"
REPORT_PATH = REFERENCES / "evaluations" / "medical-1.17" / "candidate_1_17_dev.json"
BASELINE_PATH = REFERENCES / "evaluations" / "medical-1.16-dev" / "candidate_1_16.npz"
ONTOLOGY_PATH = ROOT / "anonymizer_code" / "data" / "corpora" / "ru-pii" / "v1" / "ontology.json"
BASELINE_REPORT_PATH = REFERENCES / "evaluations" / "medical-1.17" / "baseline_1_16_validation.json"
EXPECTED_BASELINE_REPORT_SHA256 = "7c3bd72d5954203f95e5be0c6bd799ec78bcec6e0e5257af004a832268116e62"
EXPECTED_BASELINE_REVISION = "fc7ac7ba71fd2ce57a946fc379c20af7bf701d6f"
EXPECTED_CANDIDATE_DATA_SHA256 = {
    "candidate_train": "869bb4ff6fb5c66ff366bfe4674938176dce9a213b283be070304ab121e0ef03",
    "candidate_validation": "7fa86f01040730e50234826bcc6ce16b2dabffa25e146fcf0bd60dd4151f8062",
    "archive": "f215a3282043539b5c8fbac011ecf42deeb117e3d018ad4b6b3da13f4735f123",
    "protocol": "087eeb439b231c0066bd02287c3ecf8340076d6668b38a6ecd3302e45f8fb122",
}
EXPECTED_BUILD_REPORT_SHA256 = "6cfc6c1c57b34de08f7cc17646d5a83f13404e288b6a0fc32a8ec88d76017399"
PRIORITY_LABELS = {
    "APART", "COUNTRY", "DISEASE", "HOUSE", "INSTITUTION", "LOCALITY",
    "ORG", "PASSPORT", "PER", "REGION", "SNILS", "STREET",
}


def gate(
    baseline: dict[str, object], candidate: dict[str, object], *, split: str,
) -> dict[str, object]:
    base_f1 = cast(float, baseline["exact_span"]["micro_f1"])
    candidate_f1 = cast(float, candidate["exact_span"]["micro_f1"])
    base_leak = cast(float, baseline["mask_coverage"]["document_leak_rate"])
    candidate_leak = cast(float, candidate["mask_coverage"]["document_leak_rate"])
    base_coverage = cast(dict[str, dict[str, float]], baseline["mask_coverage"]["per_label_coverage_any_class"])
    candidate_coverage = cast(dict[str, dict[str, float]], candidate["mask_coverage"]["per_label_coverage_any_class"])
    coverage_deltas = {
        label: candidate_coverage[label]["recall"] - base_coverage[label]["recall"]
        for label in sorted(PRIORITY_LABELS)
    }
    checks = {
        "exact_f1_delta_at_least_minus_0_01": candidate_f1 - base_f1 >= -0.01,
        "leak_delta_at_most_plus_0_01": candidate_leak - base_leak <= 0.01,
        "all_priority_coverage_deltas_at_least_minus_0_02": all(value >= -0.02 for value in coverage_deltas.values()),
    }
    if split == "medical_validation":
        checks["medical_leak_improves_at_least_0_01"] = candidate_leak <= base_leak - 0.01
        base_snils = base_coverage["SNILS"]["recall"]
        candidate_snils = candidate_coverage["SNILS"]["recall"]
        checks["snils_improves_at_least_0_02_and_reaches_45_of_49"] = (
            candidate_snils >= base_snils + 0.02 and candidate_snils >= 45 / 49
            and candidate_coverage["SNILS"]["support"] == 49
        )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "exact_f1_delta": candidate_f1 - base_f1,
        "document_leak_delta": candidate_leak - base_leak,
        "priority_coverage_deltas": coverage_deltas,
    }


def independent_support_is_ready(build_report: dict[str, object]) -> bool:
    if build_report.get("evaluation_ready") is not True or build_report.get("editorial_review_complete") is not True:
        return False
    support_by_split = build_report.get("target_support_by_split")
    if not isinstance(support_by_split, dict):
        return False
    return all(
        isinstance(support_by_split.get(split), dict)
        and all(support_by_split[split].get(label, 0) >= 30 for label in PRIORITY_LABELS)
        for split in ("test", "challenge")
    )


def main() -> None:
    if ARTIFACT_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("Candidate output already exists; this registered run is one-shot")
    corpus_paths = {
        "common_train": EXPANDED / "train.jsonl",
        "common_validation": EXPANDED / "val.jsonl",
        "medical_train": BASE_RELEASE / "train.jsonl",
        "medical_validation": BASE_RELEASE / "validation.jsonl",
        "candidate_train": CANDIDATE_RELEASE / "train.jsonl",
        "candidate_validation": CANDIDATE_RELEASE / "validation.jsonl",
        "ontology": ONTOLOGY_PATH,
        "baseline_artifact": BASELINE_PATH,
        "model": ROOT / "anonymizer_code" / "backend" / "app" / "model.py",
        "training_runner": Path(__file__).resolve(),
        "scoring_runner": Path(__file__).with_name("run_dev.py"),
        "archive": CANDIDATE_RELEASE / "anonymizer-ner-hardcases-1.17.0_import.zip",
        "protocol": ROOT / "anonymizer_code" / "docs" / "ner-1.17" / "EVALUATION_PROTOCOL.md",
    }
    before = {name: sha256(path) for name, path in corpus_paths.items()}
    if before["baseline_artifact"] != EXPECTED_BASELINE_SHA256:
        raise ValueError("Frozen 1.16.0 artifact hash changed")
    for name, expected in EXPECTED_DATA_SHA256.items():
        if before[name] != expected:
            raise ValueError(f"Registered baseline input changed: {name}")
    for name, expected in EXPECTED_CANDIDATE_DATA_SHA256.items():
        if before[name] != expected:
            raise ValueError(f"Registered 1.17 candidate input changed: {name}")
    if sha256(BASELINE_REPORT_PATH) != EXPECTED_BASELINE_REPORT_SHA256:
        raise ValueError("Registered frozen baseline report hash changed")
    if sha256(CANDIDATE_RELEASE / "quality_report.json") != EXPECTED_BUILD_REPORT_SHA256:
        raise ValueError("Registered corpus quality report hash changed")
    build_report = json.loads((CANDIDATE_RELEASE / "quality_report.json").read_text(encoding="utf-8"))
    if (not isinstance(build_report.get("evaluation_ready"), bool)
            or not isinstance(build_report.get("editorial_review_complete"), bool)):
        raise ValueError("Corpus readiness metadata must contain boolean flags")
    baseline_path_metrics = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    if (baseline_path_metrics.get("status") != "complete"
            or baseline_path_metrics.get("git_revision") != EXPECTED_BASELINE_REVISION
            or baseline_path_metrics.get("input_sha256", {}).get("baseline_artifact") != EXPECTED_BASELINE_SHA256
            or any(baseline_path_metrics.get("input_sha256", {}).get(name) != value
                   for name, value in EXPECTED_DATA_SHA256.items())):
        raise ValueError("Frozen baseline report provenance does not match the registered run")

    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8-sig"))
    labels = {item["label"] for item in ontology["categories"] if item["label"] != "ADDR"}
    common_train = read_jsonl(corpus_paths["common_train"], labels)
    medical_train = read_jsonl(corpus_paths["candidate_train"], labels)
    common_validation = read_jsonl(corpus_paths["common_validation"], labels)
    medical_validation = read_jsonl(corpus_paths["medical_validation"], labels)
    candidate_validation = read_jsonl(corpus_paths["candidate_validation"], labels)
    validation = common_validation + medical_validation
    train = common_train + medical_train
    started = time.monotonic()
    artifact, training_metrics = train_model(
        labels, train, validation, feature_version=3,
    )
    elapsed = time.monotonic() - started

    baseline_artifact = BASELINE_PATH.read_bytes()
    candidate_scores = {
        "common_validation": score(common_validation, artifact, labels),
        "medical_1_16_validation": score(medical_validation, artifact, labels),
        "medical_1_17_validation": score(candidate_validation, artifact, labels),
    }
    baseline_scores = baseline_path_metrics["splits"]
    development_gate = {
        "common_validation": gate(baseline_scores["common_validation"], candidate_scores["common_validation"], split="common_validation"),
        "medical_validation": gate(baseline_scores["medical_validation"], candidate_scores["medical_1_16_validation"], split="medical_validation"),
    }
    development_passed = all(item["passed"] for item in development_gate.values())
    challenge_support = build_report["target_support_by_split"]["challenge"]
    test_support = build_report["target_support_by_split"]["test"]
    independent_support_ready = independent_support_is_ready(build_report)
    status = "READY_FOR_FROZEN_INDEPENDENT_EVALUATION" if development_passed and independent_support_ready else "HOLD"

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_bytes(artifact)
    after = {name: sha256(path) for name, path in corpus_paths.items()}
    if before != after:
        ARTIFACT_PATH.unlink()
        raise ValueError("Training or scoring modified a registered input")
    report = {
        "status": status,
        "candidate": "1.17.0 development candidate; never auto-activated",
        "purpose": "single reproducible development run; no test/challenge records opened or scored",
        "artifact_sha256": sha256(ARTIFACT_PATH),
        "inputs_sha256": before,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {
            "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "training": {
            "entrypoint": "app.model.train_model", "feature_version": 3,
            "feature_count": 16384, "epochs": 3, "random_state": 17,
            "loss": "log_loss", "alpha": 1e-5, "average": True,
            "batch_size": 4096, "warm_start": False,
            "train_records_before_deduplication": len(train),
            "elapsed_seconds": elapsed, "reported_metrics": training_metrics,
        },
        "baseline_artifact_sha256": EXPECTED_BASELINE_SHA256,
        "baseline_development_scores": baseline_scores,
        "candidate_development_scores": candidate_scores,
        "development_gate": development_gate,
        "independent_support_gate": {
            "minimum_per_priority_label": 30,
            "test_support": test_support,
            "challenge_support": challenge_support,
            "passed": independent_support_ready,
            "evaluation_ready": build_report["evaluation_ready"],
        },
        "decision": "Candidate remains HOLD unless development and independent gates pass; no automatic activation.",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "artifact_sha256": report["artifact_sha256"], "development_gate": development_gate}, indent=2))


if __name__ == "__main__":
    main()
