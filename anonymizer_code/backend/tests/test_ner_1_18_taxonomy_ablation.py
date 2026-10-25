from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
SCRIPT = CODE / "scripts" / "ner_1_18" / "taxonomy_ablation.py"
SPEC = importlib.util.spec_from_file_location("ner_1_18_taxonomy_ablation_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ablation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ablation
SPEC.loader.exec_module(ablation)


def test_remap_merges_only_sparse_document_subtypes_and_preserves_offsets() -> None:
    labels = sorted(ablation.MERGED_DOCUMENT_LABELS)
    source = {
        "text": "0 1 2 3 4 5 6",
        "entities": [
            {"start": index * 2, "end": index * 2 + 1, "label": label}
            for index, label in enumerate(labels)
        ] + [
            {"start": 12, "end": 13, "label": "PASSPORT"},
            {"start": 14, "end": 15, "label": "SNILS"},
        ],
    }

    result = ablation.remap_records([source])

    assert [entity["label"] for entity in result[0]["entities"]] == [
        "DOCUMENT_ID" for _ in labels
    ] + ["PASSPORT", "SNILS"]
    assert [(entity["start"], entity["end"]) for entity in result[0]["entities"]] == [
        (entity["start"], entity["end"]) for entity in source["entities"]
    ]
    assert source["entities"][0]["label"] == labels[0]
    assert result[0]["text"] == source["text"]


def test_development_input_allowlist_contains_only_named_dev_files(tmp_path: Path) -> None:
    paths = ablation.development_paths(tmp_path)

    assert set(paths) == {
        "release_train", "release_validation", "common_validation",
        "medical_1_16_validation", "baseline_1_16", "candidate_1_18_1",
    }
    assert paths["release_train"].name == "train.jsonl"
    assert paths["release_validation"].name == "validation.jsonl"
    assert all("test" not in path.name.casefold() and "challenge" not in path.name.casefold()
               for path in paths.values())


def test_collapsed_predictions_are_scored_with_the_same_document_id_mapping(monkeypatch) -> None:
    run_dev = sys.modules[ablation.score.__module__]
    monkeypatch.setattr(
        run_dev, "predict", lambda text, artifact: [{"start": 0, "end": 3, "label": "MEDICAL_RECORD_ID"}],
    )
    rows = [{"text": "123", "entities": [{"start": 0, "end": 3, "label": "DOCUMENT_ID"}]}]

    metrics = ablation.masking_aware_score(
        rows, b"unused", set(ablation.ABLATION_LABELS), ablation.LEGACY_PREDICTION_MAP,
    )

    assert metrics["exact_span"]["micro_f1"] == 1.0
    assert metrics["mask_coverage"]["document_leak_rate"] == 0.0


def test_legacy_score_excludes_document_id_gold_and_leak(monkeypatch) -> None:
    run_dev = sys.modules[ablation.score.__module__]
    monkeypatch.setattr(
        run_dev, "predict", lambda text, artifact: [{"start": 0, "end": 1, "label": "PER"}],
    )
    rows = [{"text": "x y", "entities": [
        {"start": 0, "end": 1, "label": "PER"},
        {"start": 2, "end": 3, "label": "DOCUMENT_ID"},
    ]}]

    metrics = ablation.masking_aware_score(rows, b"unused", set(ablation.LEGACY_LABELS))

    assert metrics["exact_span"]["gold_spans"] == 1
    assert metrics["mask_coverage"]["documents_with_gold"] == 1
    assert metrics["mask_coverage"]["document_leak_rate"] == 0.0
