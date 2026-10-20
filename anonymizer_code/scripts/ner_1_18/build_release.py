from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, TypedDict

CODE = Path(__file__).resolve().parents[2]
ROOT = CODE.parent
BACKEND = CODE / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.training.corpus import (  # noqa: E402
    DEFAULT_LIMITS, _canonical_records_bytes, _load_archive, _make_release,
    _manifest_for_release, _parse_jsonl, load_ontology_labels,
)
import mixed_cases  # noqa: E402
import numeric_cases  # noqa: E402

VERSION = "1.18.0"
CORPUS_ID = "anonymizer-ner-hardcases"
BASE = ROOT / "references" / "corpora" / CORPUS_ID / "1.16.0"
OUTPUT = ROOT / "references" / "corpora" / CORPUS_ID / VERSION
BASE_HASHES = {
    "train": "43a7f35e0d479ab0edf4abfa23acdd4832b96ad557c8a03a5882b69a3bae6d28",
    "validation": "dca4098b494da6b7e62d70c93126ddbd71e8df324b92568fc60c03ff3ad0949f",
}
BASE_TEST_SHA256 = "e89f2a5f07b6cc5405836a1cf8c28849aad8748da5c2e29756308e27c326191f"
ACTIVE_LABELS = frozenset(load_ontology_labels(CODE / "data/corpora/ru-pii/v1/ontology.json") - {"ADDR"})
LEGACY_LABELS = frozenset({
    "APART", "COUNTRY", "DISEASE", "EMAIL", "HOUSE", "INN", "INSTITUTION", "KPP",
    "LOCALITY", "OGRN", "ORG", "PASSPORT", "PER", "PHONE", "REGION", "SNILS", "STREET",
})
if len(ACTIVE_LABELS) != 22 or not LEGACY_LABELS < ACTIVE_LABELS or "PATIENT_CARD_ID" in ACTIVE_LABELS:
    raise ValueError("1.18 build requires the 22 active leaf labels and no PATIENT_CARD_ID")
MARKER = re.compile(r"\[\[([^\[\]|]+)\|([^\[\]]*)\]\]")
BASE_DISEASE_TEXT = "В сравнительной заметке «гепатит C» обозначен как гепатит C, а «болезнь Вильсона» как болезнь Вильсона; оба названия остаются такими же при подозрении."


class Entity(TypedDict):
    start: int
    end: int
    label: str


class Record(TypedDict):
    text: str
    entities: list[Entity]


class AuditCase(TypedDict):
    id: str
    split: str
    family: str
    source: str
    text: str
    marked_text: str
    entities: list[Entity]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def parse_marked_case(case_id: str, split: str, family: str, marked: str) -> tuple[Record, str]:
    if split not in {"train", "validation"} or not case_id or not family:
        raise ValueError(f"Invalid case metadata: {case_id}")
    chunks: list[str] = []
    entities: list[Entity] = []
    cursor = 0
    plain_cursor = 0
    while (start := marked.find("[[", cursor)) >= 0:
        if "[[" in marked[cursor + 2:start] or "]]" in marked[cursor:start]:
            raise ValueError(f"Malformed marker in {case_id}")
        chunks.append(marked[cursor:start])
        plain_cursor += start - cursor
        match = MARKER.match(marked, start)
        if match is None:
            raise ValueError(f"Malformed marker in {case_id}")
        label, surface = match.groups()
        if label not in ACTIVE_LABELS or not surface:
            raise ValueError(f"Unknown label or empty surface in {case_id}: {label}")
        end = plain_cursor + len(surface)
        entities.append({"start": plain_cursor, "end": end, "label": label})
        chunks.append(surface)
        plain_cursor = end
        cursor = match.end()
    chunks.append(marked[cursor:])
    text = "".join(chunks)
    if "[[" in text or "]]" in text:
        raise ValueError(f"Unmatched marker in {case_id}")
    row = {"text": text, "entities": entities}
    # Reuse the importer's record validation for bounds, overlap, and Unicode offsets.
    _parse_jsonl((json.dumps(row, ensure_ascii=False) + "\n").encode(), case_id, ACTIVE_LABELS, DEFAULT_LIMITS)
    return row, marked


def _cases() -> list[tuple[str, str, str, str, str]]:
    records = list(mixed_cases.CASES) + list(numeric_cases.CASES)
    invalid_ids = set(numeric_cases.INVALID_SNILS_CASE_IDS)
    all_ids = {row[0] for row in records}
    if len(all_ids) != len(records) or not invalid_ids <= all_ids:
        raise ValueError("Case IDs must be unique and checksum exceptions must reference existing cases")
    if any(not row[0].startswith("num-") for row in records if row[0] in invalid_ids):
        raise ValueError("Only numeric authored cases may declare checksum exceptions")
    declared_invalid: set[str] = set()
    for case_id, split, family, marked in numeric_cases.CASES:
        row, _ = parse_marked_case(case_id, split, family, marked)
        for entity in row["entities"]:
            if entity["label"] != "SNILS":
                continue
            value = str(row["text"])[int(entity["start"]):int(entity["end"])]
            digits = "".join(char for char in value if char.isdigit())
            total = sum(int(digit) * weight for digit, weight in zip(digits[:9], range(9, 0, -1)))
            remainder = total % 101
            checksum = "00" if total in (100, 101) or remainder == 100 else f"{remainder:02d}"
            if checksum != digits[9:]:
                declared_invalid.add(case_id)
    if declared_invalid != invalid_ids:
        raise ValueError("Numeric SNILS checksum exception IDs do not match authored invalid values")
    return [(case_id, split, family, marked, source)
            for source, source_rows in (("mixed_cases.py", mixed_cases.CASES), ("numeric_cases.py", numeric_cases.CASES))
            for case_id, split, family, marked in source_rows]


def _base_records() -> tuple[bytes, bytes, list[Record], list[Record]]:
    raw: dict[str, bytes] = {}
    parsed: dict[str, list[Record]] = {}
    for split in ("train", "validation"):
        raw[split] = (BASE / f"{split}.jsonl").read_bytes()
        if sha256(raw[split]) != BASE_HASHES[split]:
            raise ValueError(f"Frozen 1.16 {split} checksum changed")
        records = _parse_jsonl(raw[split], f"1.16/{split}", ACTIVE_LABELS, DEFAULT_LIMITS)
        parsed[split] = [{"text": row.text, "entities": [{"start": entity.start, "end": entity.end, "label": entity.label} for entity in row.entities]} for row in records]
    return raw["train"], raw["validation"], parsed["train"], parsed["validation"]


def normalize_street_prefixes(rows: list[Record]) -> tuple[list[Record], list[dict[str, object]]]:
    normalized: list[Record] = []
    log: list[dict[str, object]] = []
    for row_index, original in enumerate(rows):
        row: Record = {"text": original["text"], "entities": [dict(item) for item in original["entities"]]}
        text = row["text"]
        for entity in row["entities"]:
            if entity["label"] != "STREET":
                continue
            start = entity["start"]
            prefix = re.match(r"^(?:ул\.?|улица|проспект|переулок)\s+", text[start:entity["end"]], re.IGNORECASE)
            if prefix:
                entity["start"] = start + prefix.end()
                log.append({"row": row_index + 1, "label": "STREET", "removed_prefix": prefix.group(), "surface": text[start + prefix.end():entity["end"]]})
        normalized.append(row)
    return normalized, log


def patch_quoted_diseases(rows: list[Record]) -> tuple[list[Record], dict[str, object]]:
    corrected = [{"text": row["text"], "entities": [dict(entity) for entity in row["entities"]]} for row in rows]
    matches = [row for row in corrected if row["text"] == BASE_DISEASE_TEXT]
    if len(matches) != 1:
        raise ValueError("Guarded disease correction target must occur exactly once")
    row = matches[0]
    original = [dict(entity) for entity in row["entities"]]
    expected = [{"start": 50, "end": 59, "label": "DISEASE"}, {"start": 86, "end": 102, "label": "DISEASE"}]
    if (original != expected or BASE_DISEASE_TEXT[25:34] != "гепатит C"
            or BASE_DISEASE_TEXT[50:59] != "гепатит C"
            or BASE_DISEASE_TEXT[64:80] != "болезнь Вильсона"
            or BASE_DISEASE_TEXT[86:102] != "болезнь Вильсона"):
        raise ValueError("Guarded disease correction source changed")
    row["entities"] = sorted(original + [
        {"start": 25, "end": 34, "label": "DISEASE"},
        {"start": 64, "end": 80, "label": "DISEASE"},
    ], key=lambda item: item["start"])
    return corrected, {"source": "1.16 train row guarded by exact text and entity offsets", "added": [
        {"surface": "гепатит C", "start": 25, "end": 34},
        {"surface": "болезнь Вильсона", "start": 64, "end": 80},
    ]}


def _shape(row: Record) -> str:
    text = row["text"]
    parts: list[str] = []
    cursor = 0
    for entity in row["entities"]:
        start, end = entity["start"], entity["end"]
        parts.extend((text[cursor:start], f"<{entity['label']}>") )
        cursor = end
    parts.append(text[cursor:])
    return _normalize(re.sub(r"\s+", " ", "".join(parts)).strip())


def _validate_case_set(cases: list[tuple[str, str, str, str, str]], inherited: dict[str, list[Record]]) -> tuple[dict[str, list[Record]], list[AuditCase]]:
    rows = {"train": [], "validation": []}
    audit: list[AuditCase] = []
    ids: set[str] = set()
    numeric_family_split: dict[str, str] = {}
    seen_text = {_normalize(str(row["text"])): split for split, records in inherited.items() for row in records}
    seen_shape = {_shape(row): split for split, records in inherited.items() for row in records}
    for case_id, split, family, marked, source in cases:
        if case_id in ids:
            raise ValueError(f"Repeated case id: {case_id}")
        ids.add(case_id)
        if source == "numeric_cases.py" and family in numeric_family_split and numeric_family_split[family] != split:
            raise ValueError(f"Numeric contrast family crosses splits: {family}")
        if source == "numeric_cases.py":
            numeric_family_split[family] = split
        row, _ = parse_marked_case(case_id, split, family, marked)
        normalized = _normalize(row["text"])
        shape = _shape(row)
        if normalized in seen_text:
            raise ValueError(f"Duplicate case text after NFC/casefold: {case_id}")
        if shape in seen_shape:
            raise ValueError(f"Masked-shape clone: {case_id}")
        seen_text[normalized] = split
        seen_shape[shape] = split
        rows[split].append(row)
        audit.append({"id": case_id, "split": split, "family": family, "source": source, "text": row["text"], "marked_text": marked, "entities": row["entities"]})
    return rows, audit


def _support(rows: Iterable[Record]) -> dict[str, object]:
    records = list(rows)
    documents: dict[str, set[str]] = {label: set() for label in ACTIVE_LABELS}
    spans: Counter[str] = Counter()
    lengths: list[int] = []
    count = 0
    for row in records:
        count += 1
        text = row["text"]
        lengths.append(len(text))
        for entity in row["entities"]:
            label = entity["label"]
            documents[label].add(hashlib.sha256(_normalize(text).encode()).hexdigest())
            spans[label] += 1
    return {"records": count, "labels": {label: {"distinct_documents": len(documents[label]), "spans": spans[label]} for label in sorted(ACTIVE_LABELS)}, "long_cases_over_1000_chars": sum(length > 1000 for length in lengths), "negative_records": sum(not row["entities"] for row in records)}


def _review_markdown(audit: list[AuditCase]) -> str:
    lines = ["# 1.18 authored case review", "", "Generated from mixed_cases.py and numeric_cases.py. Each row retains its authored ID, split, family, source, marked text, and parsed spans.", ""]
    for item in audit:
        lines.extend([f"## {item['id']} · {item['split']} · {item['family']}", "", f"Source: `{item['source']}`", "", str(item["marked_text"]), ""])
        for entity in item["entities"]:
            lines.append(f"- {entity['label']}: `{item['text'][entity['start']:entity['end']]}` [{entity['start']}:{entity['end']}]")
        lines.append("")
    return "\n".join(lines)


def build(output: Path = OUTPUT) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable 1.18 output: {output}")
    base_train_raw, base_val_raw, base_train, base_val = _base_records()
    old_train, street_train_log = normalize_street_prefixes(base_train)
    old_val, street_val_log = normalize_street_prefixes(base_val)
    old_train, disease_log = patch_quoted_diseases(old_train)
    authored, audit = _validate_case_set(_cases(), {"train": old_train, "validation": old_val})
    train_rows = old_train + authored["train"]
    validation_rows = old_val + authored["validation"]
    labels = load_ontology_labels(CODE / "data/corpora/ru-pii/v1/ontology.json")
    train = _parse_jsonl(_canonical_rows(train_rows), "1.18/train", labels, DEFAULT_LIMITS)
    validation = _parse_jsonl(_canonical_rows(validation_rows), "1.18/validation", labels, DEFAULT_LIMITS)
    test_bytes = (BASE / "test.jsonl").read_bytes()
    if sha256(test_bytes) != BASE_TEST_SHA256:
        raise ValueError("Frozen 1.16 test bytes changed")
    full = {"train": train, "validation": validation, "test": _parse_jsonl(test_bytes, "1.16/test", labels, DEFAULT_LIMITS)}
    release = _make_release(CORPUS_ID, VERSION, full)
    manifest = _manifest_for_release(release)
    archive = _zip_bytes({"manifest.json": manifest, "train.jsonl": _canonical_records_bytes(train), "validation.jsonl": _canonical_records_bytes(validation), "test.jsonl": test_bytes})
    checked, _ = _load_archive(archive, labels, DEFAULT_LIMITS)
    if checked.sha256 != release.sha256:
        raise ValueError("Generated import ZIP did not round-trip")
    new_data = {"train.jsonl": _canonical_rows(authored["train"]), "validation.jsonl": _canonical_rows(authored["validation"])}
    quality = {
        "corpus": CORPUS_ID, "version": VERSION, "release_sha256": release.sha256,
        "base": {"version": "1.16.0", "train_sha256": sha256(base_train_raw), "validation_sha256": sha256(base_val_raw), "test_sha256": sha256(test_bytes), "test_sha256_expected": BASE_TEST_SHA256, "test_copied_byte_exact": True},
        "records": {"train": len(train), "validation": len(validation), "authored_train": len(authored["train"]), "authored_validation": len(authored["validation"])},
        "authored_support": {"train": _support(authored["train"]), "validation": _support(authored["validation"])},
        "development_support": {"train": _support(train_rows), "validation": _support(validation_rows)},
        "authored_group_contrasts": dict(sorted(Counter(str(item["family"]) for item in audit).items())),
        "declared_invalid_snils_case_ids": sorted(numeric_cases.INVALID_SNILS_CASE_IDS),
        "authoring_source_sha256": {
            "mixed_cases.py": sha256(Path(__file__).with_name("mixed_cases.py").read_bytes()),
            "numeric_cases.py": sha256(Path(__file__).with_name("numeric_cases.py").read_bytes()),
        },
        "new_labels": sorted(ACTIVE_LABELS - LEGACY_LABELS),
        "normalization": {"street_prefix_corrections": street_train_log + street_val_log, "quoted_disease_correction": disease_log},
        "training_policy": "Uses 1.16.0 train/validation base; replaces the flawed 1.17 additive draft with freshly curated authored examples.",
        "evaluation_ready": False,
        "evaluation_readiness_reason": "No fresh independent test or challenge set exists for the expanded 22-label ontology.",
        "inherited_test_support_by_new_label": "not evaluated or claimed",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "anonymizer-ner-hardcases-1.18.0_import.zip").write_bytes(archive)
    (output / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "corrections.json").write_text(json.dumps(quality["normalization"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "authored_review.md").write_text(_review_markdown(audit), encoding="utf-8")
    (output / "authored_cases.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "authored_train.jsonl").write_bytes(new_data["train.jsonl"])
    (output / "authored_validation.jsonl").write_bytes(new_data["validation.jsonl"])
    (output / "train.jsonl").write_bytes(_canonical_records_bytes(train))
    (output / "validation.jsonl").write_bytes(_canonical_records_bytes(validation))
    (output / "manifest.json").write_bytes(manifest)
    return quality


def _canonical_rows(rows: list[dict[str, object]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    import io
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    return output.getvalue()


if __name__ == "__main__":
    result = build()
    print(json.dumps({"release_sha256": result["release_sha256"], "evaluation_ready": result["evaluation_ready"]}, indent=2))
