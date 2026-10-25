from __future__ import annotations

import json
import io
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "scripts" / "ner_1_18"))

import build_release
TRAIN_CANDIDATE_SPEC = importlib.util.spec_from_file_location(
    "ner_1_18_train_candidate", CODE / "scripts" / "ner_1_18" / "train_candidate.py"
)
if TRAIN_CANDIDATE_SPEC is None or TRAIN_CANDIDATE_SPEC.loader is None:
    raise RuntimeError("Could not load the 1.18 evaluator module")
train_candidate = importlib.util.module_from_spec(TRAIN_CANDIDATE_SPEC)
sys.modules[TRAIN_CANDIDATE_SPEC.name] = train_candidate
TRAIN_CANDIDATE_SPEC.loader.exec_module(train_candidate)
import run_dev
from app.training.corpus import DEFAULT_LIMITS, _canonical_records_bytes, _load_archive, _make_release, _manifest_for_release, _parse_jsonl


def test_marked_case_offsets_keep_unicode_codepoints() -> None:
    row, _ = build_release.parse_marked_case("u1", "train", "family", "Ёж [[PER|Аня]] живёт в [[LOCALITY|Йошкар-Оле]].")
    assert row == {
        "text": "Ёж Аня живёт в Йошкар-Оле.",
        "entities": [
            {"start": 3, "end": 6, "label": "PER"},
            {"start": 15, "end": 25, "label": "LOCALITY"},
        ],
    }


@pytest.mark.parametrize("marked", [
    "x [[UNKNOWN|y]]",
    "x [[PER|nested [[PER|y]]]]",
    "x [[PER|y]",
    "x [[PER|]]",
])
def test_marked_case_rejects_malformed_and_unknown_markers(marked: str) -> None:
    with pytest.raises(ValueError):
        build_release.parse_marked_case("bad", "train", "family", marked)


def test_checksum_exceptions_match_numeric_authoring() -> None:
    rows = build_release._cases()
    ids = {row[0] for row in rows}
    assert build_release.numeric_cases.INVALID_SNILS_CASE_IDS <= ids
    with pytest.raises(ValueError, match="checksum exception"):
        original = build_release.numeric_cases.INVALID_SNILS_CASE_IDS
        build_release.numeric_cases.INVALID_SNILS_CASE_IDS = frozenset()
        try:
            build_release._cases()
        finally:
            build_release.numeric_cases.INVALID_SNILS_CASE_IDS = original


def test_street_prefix_normalization_changes_only_offsets() -> None:
    source = [{"text": "Адрес: ул. Чапаева, д. 3", "entities": [{"start": 7, "end": 18, "label": "STREET"}]}]
    normalized, log = build_release.normalize_street_prefixes(source)
    assert normalized[0]["text"] == source[0]["text"]
    assert normalized[0]["entities"] == [{"start": 11, "end": 18, "label": "STREET"}]
    assert log == [{"row": 1, "label": "STREET", "removed_prefix": "ул. ", "surface": "Чапаева"}]
    assert source[0]["entities"][0]["start"] == 7
    outside = [{"text": "Адрес: ул. Чапаева", "entities": [{"start": 11, "end": 18, "label": "STREET"}]}]
    outside_result, outside_log = build_release.normalize_street_prefixes(outside)
    assert outside_result == outside
    assert outside_log == []


def test_quoted_disease_patch_is_narrow_and_preserves_existing_spans() -> None:
    row = {"text": build_release.BASE_DISEASE_TEXT, "entities": [
        {"start": 50, "end": 59, "label": "DISEASE"},
        {"start": 86, "end": 102, "label": "DISEASE"},
    ]}
    original_text = row["text"]
    rows, change = build_release.patch_quoted_diseases([row])
    assert rows[0]["text"] == original_text
    assert rows[0]["entities"] == [
        {"start": 25, "end": 34, "label": "DISEASE"},
        {"start": 50, "end": 59, "label": "DISEASE"},
        {"start": 64, "end": 80, "label": "DISEASE"},
        {"start": 86, "end": 102, "label": "DISEASE"},
    ]
    assert [original_text[item["start"]:item["end"]] for item in rows[0]["entities"]] == [
        "гепатит C", "гепатит C", "болезнь Вильсона", "болезнь Вильсона",
    ]
    assert len(row["entities"]) == 2
    assert len(change["added"]) == 2
    with pytest.raises(ValueError, match="exactly once"):
        build_release.patch_quoted_diseases([row, row])


def test_legacy_document_corrections_are_guarded_and_remove_only_reviewed_clones() -> None:
    _, _, source_train, source_validation = build_release._base_records()
    train = [{"text": row["text"], "entities": [dict(entity) for entity in row["entities"]]} for row in source_train]
    validation = [{"text": row["text"], "entities": [dict(entity) for entity in row["entities"]]} for row in source_validation]
    train.append({"text": "Договор № 12345678901 — отдельный пример договора.", "entities": []})
    train, validation, audit = build_release.correct_legacy_documents(train, validation)

    assert len(train) == len(source_train) - 6 + 1
    assert len(validation) == len(source_validation) - 2
    expected = {
        2097: ("12345678901", "OTHER_DOCUMENT_ID"),
        8306: ("DOC-00000051", "OTHER_DOCUMENT_ID"),
        8315: ("MED-000060", "OTHER_DOCUMENT_ID"),
        8307: ("00000052", "MEDICAL_RECORD_ID"),
        8311: ("00000056", "MEDICAL_RECORD_ID"),
    }
    for index, (surface, label) in expected.items():
        target = source_train[index]
        corrected_index = index - sum(1 for split, removed_index in build_release.OBSOLETE_NEGATIVE_ROWS if split == "train" and removed_index < index)
        match = train[corrected_index]["entities"]
        assert any(target["text"][entity["start"]:entity["end"]] == surface and entity["label"] == label for entity in match)
    assert train[-1]["text"] == "Договор № 12345678901 — отдельный пример договора."
    assert len(audit["annotations"]) == 5
    assert len(audit["removed_reviewed_near_clones"]) == 8
    assert all(item["entities"] == [] and item["source_sha256"] for item in audit["removed_reviewed_near_clones"])
    bad_train = list(source_train)
    bad_train[8306] = {"text": "different", "entities": []}
    with pytest.raises(ValueError, match="source changed"):
        build_release.correct_legacy_documents(bad_train, source_validation)


def test_audited_common_document_cues_preserve_old_spans_and_reject_sum_context() -> None:
    _, source = build_release._common_records()
    corrected, audit = build_release.annotate_common_documents(source)
    assert audit["actual_additions"] == {
        "train": {"MEDICAL_RECORD_ID": 40, "OTHER_DOCUMENT_ID": 90, "_CONTRACT_DOCUMENT_ID": 41},
        "validation": {"MEDICAL_RECORD_ID": 0, "OTHER_DOCUMENT_ID": 8, "_CONTRACT_DOCUMENT_ID": 0},
    }
    for split, rows in source.items():
        for index, row in enumerate(rows):
            updated = corrected[split][index]
            assert all(entity in updated["entities"] for entity in row["entities"])
    contract_pattern = build_release.COMMON_CUES["_CONTRACT_DOCUMENT_ID"]
    assert contract_pattern.search("Сумма по договору: 905130 рублей") is None
    assert contract_pattern.search("Номер договора № 905130") is not None
    assert contract_pattern.search("Номер договора № 12345678901234567890123456") is None
    medical_pattern = build_release.COMMON_CUES["MEDICAL_RECORD_ID"]
    assert medical_pattern.search("медкарта № 000123") is not None
    assert medical_pattern.search("номер карты учреждения 000123") is None


def test_release_snapshot_composes_sources_without_multiplication_or_cross_split_reuse() -> None:
    _, common = build_release._common_records()
    common, _ = build_release.annotate_common_documents(common)
    _, _, medical_train, medical_validation = build_release._base_records()
    medical_train, medical_validation, _ = build_release.correct_legacy_documents(medical_train, medical_validation)
    medical_train, _ = build_release.normalize_street_prefixes(medical_train)
    medical_validation, _ = build_release.normalize_street_prefixes(medical_validation)
    medical_train, _ = build_release.patch_quoted_diseases(medical_train)
    authored, _ = build_release._validate_case_set(build_release._cases(), {
        "train": common["train"] + medical_train,
        "validation": common["validation"] + medical_validation,
    })
    train_sources = [("common", common["train"]), ("medical", medical_train), ("authored", authored["train"])]
    validation_sources = [("common", common["validation"]), ("medical", medical_validation), ("authored", authored["validation"])]
    train_texts = {row["text"] for _, rows in train_sources for row in rows}
    validation_texts = {row["text"] for _, rows in validation_sources for row in rows}
    assert train_texts.isdisjoint(validation_texts)
    train, train_audit = build_release._combine_split("train", train_sources, validation_texts)
    validation, validation_audit = build_release._combine_split("validation", validation_sources, train_texts)
    assert train_audit["records_before_deduplication"] == 13731
    assert validation_audit["records_before_deduplication"] == 2218
    assert len(train) == 13731
    assert len(validation) == 2218
    assert train_audit["exact_duplicate_count"] == validation_audit["exact_duplicate_count"] == 0


def test_runner_preflight_rejects_mismatched_sources_and_baselines(tmp_path: Path) -> None:
    source_paths = [tmp_path / "common-train.jsonl", tmp_path / "common-val.jsonl", tmp_path / "medical-val.jsonl"]
    for path in source_paths:
        path.write_bytes(b"frozen")
    quality = {"base": {
        "common_train_sha256": train_candidate.sha256(source_paths[0]),
        "common_validation_sha256": train_candidate.sha256(source_paths[1]),
        "medical_validation_sha256": train_candidate.sha256(source_paths[2]),
    }}
    changed_quality = {"base": {**quality["base"], "common_train_sha256": "changed"}}
    with pytest.raises(ValueError, match="common_train_sha256"):
        train_candidate._validate_release_provenance(*source_paths, changed_quality, b"baseline16", b"baseline17")

    with pytest.raises(ValueError, match="Baseline 1.16 artifact hash changed"):
        train_candidate._validate_release_provenance(*source_paths, quality, b"baseline16", b"baseline17")


def test_duplicate_normalized_text_and_masked_shape_are_rejected() -> None:
    inherited = {"train": [{"text": "Пациентка Анна пришла.", "entities": [{"start": 10, "end": 14, "label": "PER"}]}], "validation": []}
    same_text = [("case-1", "train", "family-a", "Пациентка [[PER|Анна]] пришла.", "fixture")]
    with pytest.raises(ValueError, match="Duplicate case text"):
        build_release._validate_case_set(same_text, inherited)
    clone = [("case-2", "validation", "family-b", "Пациентка [[PER|Мария]] пришла.", "fixture")]
    with pytest.raises(ValueError, match="Masked-shape clone"):
        build_release._validate_case_set(clone, inherited)


def test_full_snapshot_dedupes_identical_annotations_and_rejects_conflicts_or_cross_split_text() -> None:
    common = [{"text": "Договор № 12345", "entities": [{"start": 10, "end": 15, "label": "OTHER_DOCUMENT_ID"}]}]
    medical = [{"text": "Договор № 12345", "entities": [{"start": 10, "end": 15, "label": "OTHER_DOCUMENT_ID"}]}]
    combined, audit = build_release._combine_split("train", [("common", common), ("medical", medical)], set())
    assert combined == common
    assert audit["source_record_counts"] == {"common": 1, "medical": 1}
    assert audit["records_before_deduplication"] == 2
    assert audit["records_after_deduplication"] == 1
    assert audit["exact_duplicate_count"] == 1

    conflicting = [{"text": "Договор № 12345", "entities": [{"start": 10, "end": 15, "label": "PASSPORT"}]}]
    with pytest.raises(ValueError, match="Conflicting annotations"):
        build_release._combine_split("train", [("common", common), ("medical", conflicting)], set())
    with pytest.raises(ValueError, match="across release splits"):
        build_release._combine_split("validation", [("common", common)], {"Договор № 12345"})


def test_family_cannot_cross_splits() -> None:
    cases = [
        ("a", "train", "same-family", "[[PER|Анна]] дома.", "numeric_cases.py"),
        ("b", "validation", "same-family", "Врач видел [[PER|Бориса]].", "numeric_cases.py"),
    ]
    with pytest.raises(ValueError, match="crosses splits"):
        build_release._validate_case_set(cases, {"train": [], "validation": []})


def test_builder_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "release"
    target.mkdir()
    (target / "keep").write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_release.build(target)
    assert (target / "keep").read_text(encoding="utf-8") == "unchanged"


def test_synthetic_import_zip_roundtrips_and_copies_test_member_exactly() -> None:
    labels = build_release.ACTIVE_LABELS
    train_bytes = b'{"text":"training row","entities":[]}\n'
    validation_bytes = b'{"text":"validation row","entities":[]}\n'
    opaque_test_bytes = b'{"text":"holdout sentinel","entities":[]}\n'
    records = {
        "train": _parse_jsonl(train_bytes, "train", labels, DEFAULT_LIMITS),
        "validation": _parse_jsonl(validation_bytes, "validation", labels, DEFAULT_LIMITS),
        "test": _parse_jsonl(opaque_test_bytes, "test", labels, DEFAULT_LIMITS),
    }
    release = _make_release(build_release.CORPUS_ID, build_release.VERSION, records)
    archive_bytes = build_release._zip_bytes({
        "manifest.json": _manifest_for_release(release),
        "train.jsonl": _canonical_records_bytes(records["train"]),
        "validation.jsonl": _canonical_records_bytes(records["validation"]),
        "test.jsonl": opaque_test_bytes,
    })
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {"manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"}
        assert archive.read("test.jsonl") == opaque_test_bytes
    checked_release, _ = _load_archive(archive_bytes, labels, DEFAULT_LIMITS)
    assert checked_release.sha256 == release.sha256


def test_preserved_disease_does_not_cover_pii_for_masking_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked = lambda text, artifact: [{"start": 0, "end": 20, "label": "DISEASE"}]
    monkeypatch.setattr(train_candidate, "predict", mocked)
    monkeypatch.setattr(run_dev, "predict", mocked)
    rows = [{"text": "Алексей в примере.", "entities": [{"start": 0, "end": 6, "label": "PER"}]}]
    result = train_candidate._masking_aware_score(rows, b"unused", set(build_release.ACTIVE_LABELS))
    assert result["mask_coverage"]["document_leak_rate"] == 0
    assert result["preserve_diseases_masking"]["pii_document_leak_rate_excludes_disease_only_documents"] == 1


def test_new_document_leaf_false_positive_remains_visible_in_22_label_score(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked = lambda text, artifact: [{"start": 0, "end": 5, "label": "MEDICAL_RECORD_ID"}]
    monkeypatch.setattr(train_candidate, "predict", mocked)
    monkeypatch.setattr(run_dev, "predict", mocked)
    rows = [{"text": "строка", "entities": []}]
    legacy = train_candidate._masking_aware_score(rows, b"unused", set(train_candidate.LEGACY_LABELS))
    expanded = train_candidate._masking_aware_score(rows, b"unused", set(build_release.ACTIVE_LABELS))
    assert legacy["mask_coverage"]["false_positive_spans_with_zero_gold_overlap"] == 0
    assert expanded["mask_coverage"]["false_positive_spans_with_zero_gold_overlap"] == 1


def test_parent_document_score_accepts_exact_span_with_wrong_child(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked = lambda text, artifact: [{"start": 2, "end": 8, "label": "MEDICAL_RECORD_ID"}]
    monkeypatch.setattr(train_candidate, "predict", mocked)
    rows = [{"text": "ID 123456", "entities": [{"start": 2, "end": 8, "label": "PASSPORT"}]}]
    leaf = train_candidate._per_label_exact(rows, b"unused", {"PASSPORT", "MEDICAL_RECORD_ID"})
    parent = train_candidate._parent_document_exact(rows, b"unused")
    assert leaf["PASSPORT"]["false_negative"] == 1
    assert leaf["PASSPORT"]["support"] == 1
    assert parent["true_positive"] == 1
    assert parent["f1"] == 1
