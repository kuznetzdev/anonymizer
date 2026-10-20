from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_BASELINE_SHA256 = "e39ca86b2daa1a3f3e505ca644222e0e148fe4b4a2224b5a6b692188346700ba"
EXPECTED_DATA_SHA256 = {
    "common_train": "921716a5524295c1f8d59bd23904a4104e1905620e8d1888b88b7e1a3767d6bd",
    "common_validation": "7f0a3eb27a64fe5a217c296f59a2930291fdf6faa5a71e5c85aaf334529df59b",
    "medical_train": "43a7f35e0d479ab0edf4abfa23acdd4832b96ad557c8a03a5882b69a3bae6d28",
    "medical_validation": "dca4098b494da6b7e62d70c93126ddbd71e8df324b92568fc60c03ff3ad0949f",
    "ontology": "618768a500e52b8d5505ed99aad79584069818a3588a9d3dfff10bc7320d58e7",
}
sys.path.insert(0, str(ROOT / "anonymizer_code" / "backend"))

import numpy
import scipy
import sklearn
from app.model import decode_artifact, normalize_legacy_prediction_labels, predict


class Entity(TypedDict):
    start: int
    end: int
    label: str


class Record(TypedDict):
    text: str
    entities: list[Entity]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, labels: set[str]) -> list[Record]:
    rows: list[Record] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != {"text", "entities"}:
                raise ValueError(f"Invalid record structure at {path.name}:{line_number}")
            text = value["text"]
            entities = value["entities"]
            if not isinstance(text, str) or not isinstance(entities, list):
                raise ValueError(f"Invalid record values at {path.name}:{line_number}")
            normalized: list[Entity] = []
            previous_end = -1
            for entity in entities:
                if not isinstance(entity, dict) or set(entity) != {"start", "end", "label"}:
                    raise ValueError(f"Invalid entity structure at {path.name}:{line_number}")
                start, end, label = entity["start"], entity["end"], entity["label"]
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not isinstance(label, str)
                    or label not in labels
                    or start < previous_end
                    or start < 0
                    or end <= start
                    or end > len(text)
                ):
                    raise ValueError(f"Invalid entity span at {path.name}:{line_number}")
                normalized.append({"start": start, "end": end, "label": label})
                previous_end = end
            rows.append({"text": text, "entities": normalized})
    return rows


def score(rows: list[Record], artifact: bytes, labels: set[str]) -> dict[str, object]:
    gold: set[tuple[int, int, int, str]] = set()
    predicted: set[tuple[int, int, int, str]] = set()
    gold_by_label: Counter[str] = Counter()
    covered_by_label: Counter[str] = Counter()
    documents_with_gold = 0
    documents_with_leak = 0
    false_positive_spans = 0
    overmasked_characters = 0
    predicted_masked_characters = 0

    for row_index, row in enumerate(rows):
        entities = row["entities"]
        spans = [
            span
            for span in normalize_legacy_prediction_labels(predict(row["text"], artifact))
            if span["label"] in labels
        ]
        gold.update((row_index, item["start"], item["end"], item["label"]) for item in entities)
        predicted.update((row_index, item["start"], item["end"], item["label"]) for item in spans)
        documents_with_gold += int(bool(entities))
        if entities and any(
            not any(span["start"] <= entity["start"] and span["end"] >= entity["end"] for span in spans)
            for entity in entities
        ):
            documents_with_leak += 1

        for entity in entities:
            gold_by_label[entity["label"]] += 1
            if any(span["start"] <= entity["start"] and span["end"] >= entity["end"] for span in spans):
                covered_by_label[entity["label"]] += 1
        for span in spans:
            overlaps = [
                entity for entity in entities
                if span["start"] < entity["end"] and span["end"] > entity["start"]
            ]
            overlap_length = sum(
                min(span["end"], entity["end"]) - max(span["start"], entity["start"])
                for entity in overlaps
            )
            false_positive_spans += int(not overlaps)
            predicted_masked_characters += span["end"] - span["start"]
            overmasked_characters += span["end"] - span["start"] - overlap_length

    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return {
        "records": len(rows),
        "exact_span": {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "gold_spans": len(gold),
            "predicted_spans": len(predicted),
        },
        "mask_coverage": {
            "documents_with_gold": documents_with_gold,
            "documents_with_any_unmasked_gold_span": documents_with_leak,
            "document_leak_rate": documents_with_leak / documents_with_gold if documents_with_gold else None,
            "per_label_coverage_any_class": {
                label: {
                    "covered": covered_by_label[label],
                    "support": gold_by_label[label],
                    "recall": covered_by_label[label] / gold_by_label[label],
                }
                for label in sorted(gold_by_label)
            },
            "false_positive_spans_with_zero_gold_overlap": false_positive_spans,
            "overmasked_characters": overmasked_characters,
            "predicted_masked_characters": predicted_masked_characters,
            "overmask_fraction": overmasked_characters / predicted_masked_characters
            if predicted_masked_characters else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frozen NER 1.16.0 on development validation only.")
    parser.add_argument("--references-root", type=Path, default=ROOT / "references")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    references = args.references_root.resolve()
    expanded = references / "expanded_dataset"
    release = references / "corpora" / "anonymizer-ner-hardcases" / "1.16.0"
    artifact_path = references / "evaluations" / "medical-1.16-dev" / "candidate_1_16.npz"
    ontology_path = ROOT / "anonymizer_code" / "data" / "corpora" / "ru-pii" / "v1" / "ontology.json"
    model_path = ROOT / "anonymizer_code" / "backend" / "app" / "model.py"
    input_paths = {
        "common_train": expanded / "train.jsonl",
        "common_validation": expanded / "val.jsonl",
        "medical_train": release / "train.jsonl",
        "medical_validation": release / "validation.jsonl",
        "baseline_artifact": artifact_path,
        "ontology": ontology_path,
        "model": model_path,
        "evaluator": Path(__file__).resolve(),
    }
    before = {name: sha256(path) for name, path in input_paths.items()}
    if before["baseline_artifact"] != EXPECTED_BASELINE_SHA256:
        raise ValueError("Frozen 1.16.0 candidate hash does not match the registered baseline")
    for name, expected in EXPECTED_DATA_SHA256.items():
        if before[name] != expected:
            raise ValueError(f"Frozen development input hash does not match the registered {name} input")

    ontology = json.loads(ontology_path.read_text(encoding="utf-8-sig"))
    labels = {item["label"] for item in ontology["categories"] if item["label"] != "ADDR"}
    artifact = artifact_path.read_bytes()
    hasher, classes, weights, intercept, version = decode_artifact(artifact)
    if version != 3:
        raise ValueError(f"Expected baseline artifact v3, got v{version}")
    dimensions_match = (
        hasher.n_features == 16_384
        and len(classes) == 35
        and weights.shape[0] == len(classes)
        and len(intercept) == len(classes)
    )
    if not dimensions_match:
        raise ValueError("Baseline artifact dimensions do not match the registered v3 contract")
    common_validation = read_jsonl(input_paths["common_validation"], labels)
    medical_validation = read_jsonl(input_paths["medical_validation"], labels)
    report = {
        "status": "complete",
        "purpose": "baseline-only development validation; no test or challenge data read",
        "baseline": "candidate 1.16.0",
        "artifact_version": version,
        "artifact_features": hasher.n_features,
        "artifact_classes": len(classes),
        "labels": sorted(labels),
        "masking_policy": {
            "active_labels": sorted(labels),
            "all_builtin_labels_enabled": True,
            "disease_contributes_to_document_leak": True,
        },
        "input_sha256": before,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "baseline_training": {
            "entrypoint": "app.model.train_model",
            "epochs": 3,
            "random_state": 17,
            "loss": "log_loss",
            "alpha": 1e-5,
            "average": True,
            "batch_size": 4096,
            "note": "Frozen 1.16.0 artifact is evaluated as-is; this runner does not retrain it.",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "splits": {
            "common_validation": score(common_validation, artifact, labels),
            "medical_validation": score(medical_validation, artifact, labels),
        },
    }
    after = {name: sha256(path) for name, path in input_paths.items()}
    if after != before:
        raise ValueError("A frozen baseline or validation input changed during scoring")
    output = args.output or references / "evaluations" / "medical-1.17" / "baseline_1_16_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate validation report: {output}")


if __name__ == "__main__":
    main()
