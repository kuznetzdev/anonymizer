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

VERSION = "1.18.1"
CORPUS_ID = "anonymizer-ner-hardcases"
BASE = ROOT / "references" / "corpora" / CORPUS_ID / "1.16.0"
OUTPUT = ROOT / "references" / "corpora" / CORPUS_ID / VERSION
BASE_HASHES = {
    "train": "43a7f35e0d479ab0edf4abfa23acdd4832b96ad557c8a03a5882b69a3bae6d28",
    "validation": "dca4098b494da6b7e62d70c93126ddbd71e8df324b92568fc60c03ff3ad0949f",
}
COMMON_BASE = ROOT / "references" / "expanded_dataset"
COMMON_HASHES = {
    "train": "921716a5524295c1f8d59bd23904a4104e1905620e8d1888b88b7e1a3767d6bd",
    "validation": "7f0a3eb27a64fe5a217c296f59a2930291fdf6faa5a71e5c85aaf334529df59b",
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
LEGACY_DOCUMENT_ANNOTATIONS = {
    ("train", 8306): (
        "В реестре для пациентки Марии Никитиной реквизит паспорта записан как AB-00000051; номер договора — DOC-00000051.",
        [{"start": 24, "end": 39, "label": "PER"}, {"start": 70, "end": 81, "label": "PASSPORT"}],
        {"start": 100, "end": 112, "label": "OTHER_DOCUMENT_ID"},
    ),
    ("train", 8315): (
        "Для выдачи заключения проверили паспорт 00 00 000060; договор на услуги имеет номер MED-000060.",
        [{"start": 40, "end": 52, "label": "PASSPORT"}],
        {"start": 84, "end": 94, "label": "OTHER_DOCUMENT_ID"},
    ),
    ("train", 8307): (
        "К направлению приложена копия паспорта: серия и номер 00 00 000052; номер медкарты 00000052.",
        [{"start": 54, "end": 66, "label": "PASSPORT"}],
        {"start": 83, "end": 91, "label": "MEDICAL_RECORD_ID"},
    ),
    ("train", 8311): (
        "OCR-строка анкеты: ПАСПОРТ № AB 00000056; № медкарты 00000056.",
        [{"start": 29, "end": 40, "label": "PASSPORT"}],
        {"start": 53, "end": 61, "label": "MEDICAL_RECORD_ID"},
    ),
}
OBSOLETE_NEGATIVE_ROWS = {
    ("train", 2114): "Договор № 12345678901 подготовили для архива, это не СНИЛС. Доступ к этой записи есть только у сотрудников. После проверки электронной формы лист вложений перенесли в отдельную папку.",
    ("train", 2134): "Договор № 12345678901 подготовили для архива, это не СНИЛС. Бухгалтеру не потребовались дополнительные документы. После совещания отдела участники согласовали новую редакцию.",
    ("train", 2154): "Договор № 12345678901 подготовили для архива, это не СНИЛС. Администратор согласовал исправление описания. При сортировке входящих документов в системе исправили порядок отображения строк.",
    ("train", 2169): "Договор № 12345678901 подготовили для архива, это не СНИЛС. Длинный номер сам по себе не определяет его тип. Когда архив получил копии, подписанный лист передали на хранение.",
    ("train", 2197): "Договор № 12345678901 подготовили для архива, это не СНИЛС. В отчёте это поле специально оставили без категории. После уточнения очередности работ папку вернули в общий архив.",
    ("train", 2207): "Договор № 12345678901 подготовили для архива, это не СНИЛС. Не переносить это число в карточку гражданина. Когда все замечания собрали, редактор внёс небольшую правку в заголовок.",
    ("validation", 386): "Договор № 12345678901 подготовили для архива, это не СНИЛС. Ошибку заметили на втором этапе сверки. После телефонного разговора с редактором распечатку приложили к учебным материалам.",
    ("validation", 408): "Договор № 12345678901 подготовили для архива, это не СНИЛС. Проверку проводили без связи с реальными субъектами. Перед переносом черновика в архив ответственному передали распечатку без личных приложений.",
}
LEGACY_CONTRACT_NEGATIVE = (
    "Договор № 12345678901 подготовили для архива, это не СНИЛС. Проверили вечером, результат оставили в черновике. Перед внесением правок в журнал корректор уточнил написание заголовка.",
    {"start": 10, "end": 21, "label": "OTHER_DOCUMENT_ID"},
)
COMMON_CUES = {
    "MEDICAL_RECORD_ID": re.compile(
        r"(?:мед\.?\s*карт\w*|медицинск\w+\s+карт\w*|амбулаторн\w+\s+карт\w*|истори\w+\s+болезн\w*|карт\w+\s+(?:пациента|больного))\s*(?:№|номер(?:\s+карт\w*)?|:|#)\s*([0-9][A-ZА-ЯЁa-zа-яё0-9/-]{0,24})(?![A-ZА-ЯЁa-zа-яё0-9/-])",
        re.IGNORECASE,
    ),
    "OTHER_DOCUMENT_ID": re.compile(
        r"(?:номер\s+документ\w*|код\s+документ\w*)\s*(?:№|#|:)?\s*([0-9][A-ZА-ЯЁa-zа-яё0-9/-]{0,24})(?![A-ZА-ЯЁa-zа-яё0-9/-])",
        re.IGNORECASE,
    ),
    "_CONTRACT_DOCUMENT_ID": re.compile(
        r"(?:номер\s+)?(?:договор\w*|контракт\w*)\s*(?:№|номер|#)\s*([0-9][A-ZА-ЯЁa-zа-яё0-9/-]{0,24})(?![A-ZА-ЯЁa-zа-яё0-9/-])",
        re.IGNORECASE,
    ),
}


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


def _common_records() -> tuple[dict[str, bytes], dict[str, list[Record]]]:
    raw = {split: (COMMON_BASE / f"{split if split == 'train' else 'val'}.jsonl").read_bytes() for split in ("train", "validation")}
    parsed: dict[str, list[Record]] = {}
    for split, source in (("train", "train.jsonl"), ("validation", "val.jsonl")):
        if sha256(raw[split]) != COMMON_HASHES[split]:
            raise ValueError(f"Frozen common {split} checksum changed")
        parsed[split] = [
            {"text": row.text, "entities": [{"start": entity.start, "end": entity.end, "label": entity.label} for entity in row.entities]}
            for row in _parse_jsonl(raw[split], f"common/{source}", ACTIVE_LABELS, DEFAULT_LIMITS)
        ]
    return raw, parsed


def annotate_common_documents(rows_by_split: dict[str, list[Record]]) -> tuple[dict[str, list[Record]], dict[str, object]]:
    corrected = {
        split: [{"text": row["text"], "entities": [dict(entity) for entity in row["entities"]]} for row in rows]
        for split, rows in rows_by_split.items()
    }
    expected_counts = {
        "train": {"MEDICAL_RECORD_ID": 40, "OTHER_DOCUMENT_ID": 90, "_CONTRACT_DOCUMENT_ID": 41},
        "validation": {"MEDICAL_RECORD_ID": 0, "OTHER_DOCUMENT_ID": 8, "_CONTRACT_DOCUMENT_ID": 0},
    }
    counts = {split: {label: 0 for label in expected_counts[split]} for split in expected_counts}
    audit: list[dict[str, object]] = []
    covered_matches = {split: 0 for split in expected_counts}
    for split, rows in corrected.items():
        if split not in expected_counts:
            raise ValueError(f"Unexpected common corpus split: {split}")
        for index, row in enumerate(rows):
            original_entities = list(row["entities"])
            additions: list[Entity] = []
            seen_spans: set[tuple[int, int]] = set()
            for cue, pattern in COMMON_CUES.items():
                label = "OTHER_DOCUMENT_ID" if cue == "_CONTRACT_DOCUMENT_ID" else cue
                for match in pattern.finditer(row["text"]):
                    start, end = match.span(1)
                    if any(entity["start"] < end and start < entity["end"] for entity in original_entities):
                        covered_matches[split] += 1
                        continue
                    if (start, end) in seen_spans:
                        continue
                    seen_spans.add((start, end))
                    entity_text = row["text"][start:end]
                    if not entity_text[0].isdigit() or len(entity_text) > 25:
                        raise ValueError(f"Invalid audited document value at common/{split}:{index}")
                    additions.append({"start": start, "end": end, "label": label})
                    counts[split][cue] += 1
            if additions:
                original_record = {"text": row["text"], "entities": original_entities}
                source_hash = sha256(json.dumps(original_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
                row["entities"] = sorted(original_entities + additions, key=lambda entity: entity["start"])
                audit.append({
                    "source_id": f"common/{split}:{index}",
                    "source_sha256": source_hash,
                    "added": [{**entity, "surface": row["text"][entity["start"]:entity["end"]]} for entity in additions],
                })
    if counts != expected_counts:
        raise ValueError(f"Audited common document-cue counts changed: {counts}")
    return corrected, {"expected_additions": expected_counts, "actual_additions": counts, "covered_existing_entity_matches": covered_matches, "annotated_records": audit}


def _combine_split(
    split: str,
    sources: list[tuple[str, list[Record]]],
    other_split_texts: set[str],
) -> tuple[list[Record], dict[str, object]]:
    combined: list[Record] = []
    seen: dict[str, tuple[list[Entity], str]] = {}
    duplicates: list[dict[str, str]] = []
    source_counts = {source: len(rows) for source, rows in sources}
    for source, rows in sources:
        for index, row in enumerate(rows):
            text = row["text"]
            entities = sorted((dict(entity) for entity in row["entities"]), key=lambda entity: entity["start"])
            if text in other_split_texts:
                raise ValueError(f"Exact text occurs across release splits: {source}/{split}:{index}")
            existing = seen.get(text)
            if existing is None:
                seen[text] = (entities, f"{source}/{split}:{index}")
                combined.append({"text": text, "entities": entities})
            elif existing[0] != entities:
                raise ValueError(f"Conflicting annotations for duplicate release text: {source}/{split}:{index}")
            else:
                duplicates.append({"kept_source_id": existing[1], "duplicate_source_id": f"{source}/{split}:{index}", "text_sha256": sha256(text.encode("utf-8"))})
    return combined, {"source_record_counts": source_counts, "records_before_deduplication": sum(source_counts.values()), "records_after_deduplication": len(combined), "exact_duplicate_count": len(duplicates), "exact_duplicates": duplicates}


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


def correct_legacy_documents(train: list[Record], validation: list[Record]) -> tuple[list[Record], list[Record], dict[str, object]]:
    splits = {"train": train, "validation": validation}
    changes: list[dict[str, object]] = []
    removals: list[dict[str, object]] = []
    for (split, index), (expected_text, expected_entities, addition) in LEGACY_DOCUMENT_ANNOTATIONS.items():
        rows = splits[split]
        if index >= len(rows):
            raise ValueError(f"Legacy annotation source row is missing: {split}:{index}")
        row = rows[index]
        if row["text"] != expected_text or row["entities"] != expected_entities:
            raise ValueError(f"Guarded legacy annotation source changed: {split}:{index}")
        entity_text = row["text"][addition["start"]:addition["end"]]
        expected_surface = {"OTHER_DOCUMENT_ID": ("DOC-00000051", "MED-000060"), "MEDICAL_RECORD_ID": ("00000052", "00000056")}[addition["label"]]
        if entity_text not in expected_surface or addition["label"] not in ACTIVE_LABELS:
            raise ValueError(f"Invalid guarded legacy annotation: {split}:{index}")
        row["entities"] = sorted(row["entities"] + [addition], key=lambda entity: entity["start"])
        changes.append({"source_id": f"1.16.0/{split}:{index}", "reason": "explicit medical-chart or contract identifier was unlabeled", "source_sha256": sha256(json.dumps({"text": expected_text, "entities": expected_entities}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()), "added": {**addition, "surface": entity_text}})

    contract_text, contract_entity = LEGACY_CONTRACT_NEGATIVE
    contract_row = splits["train"][2097]
    if contract_row["text"] != contract_text or contract_row["entities"]:
        raise ValueError("Guarded legacy contract-versus-SNILS source changed: train:2097")
    contract_surface = contract_text[contract_entity["start"]:contract_entity["end"]]
    if contract_surface != "12345678901":
        raise ValueError("Guarded legacy contract span changed: train:2097")
    contract_hash = sha256(json.dumps(contract_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    contract_row["entities"] = [contract_entity]
    changes.append({
        "source_id": "1.16.0/train:2097",
        "reason": "explicit contract number is a document ID; 11 digits alone are not a SNILS",
        "source_sha256": contract_hash,
        "added": {**contract_entity, "surface": contract_surface},
    })

    for (split, index), expected_text in OBSOLETE_NEGATIVE_ROWS.items():
        rows = splits[split]
        if index >= len(rows):
            raise ValueError(f"Legacy negative source row is missing: {split}:{index}")
        row = rows[index]
        if row["text"] != expected_text or row["entities"]:
            raise ValueError(f"Guarded obsolete negative source changed: {split}:{index}")
        source_bytes = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        removals.append({"source_id": f"1.16.0/{split}:{index}", "reason": "reviewed near-clone with the same fixed contract number appears across development splits", "source_sha256": sha256(source_bytes), "text": row["text"], "entities": []})
    for split, rows in splits.items():
        removed_indexes = {index for row_split, index in OBSOLETE_NEGATIVE_ROWS if row_split == split}
        rows[:] = [row for index, row in enumerate(rows) if index not in removed_indexes]
    return splits["train"], splits["validation"], {"annotations": changes, "removed_reviewed_near_clones": removals}


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
    common_raw, common = _common_records()
    common, common_corrections = annotate_common_documents(common)
    old_train, legacy_val, legacy_corrections = correct_legacy_documents(base_train, base_val)
    old_train, street_train_log = normalize_street_prefixes(old_train)
    old_val, street_val_log = normalize_street_prefixes(legacy_val)
    old_train, disease_log = patch_quoted_diseases(old_train)
    inherited = {
        "train": common["train"] + old_train,
        "validation": common["validation"] + old_val,
    }
    authored, audit = _validate_case_set(_cases(), inherited)
    train_sources = [("common", common["train"]), ("medical", old_train), ("authored", authored["train"])]
    validation_sources = [("common", common["validation"]), ("medical", old_val), ("authored", authored["validation"])]
    all_train_texts = {row["text"] for _, rows in train_sources for row in rows}
    all_validation_texts = {row["text"] for _, rows in validation_sources for row in rows}
    if overlap := all_train_texts & all_validation_texts:
        raise ValueError(f"Exact text occurs across release splits: {sha256(next(iter(overlap)).encode('utf-8'))}")
    train_rows, train_merge = _combine_split("train", train_sources, all_validation_texts)
    validation_rows, validation_merge = _combine_split("validation", validation_sources, all_train_texts)
    labels = load_ontology_labels(CODE / "data/corpora/ru-pii/v1/ontology.json")
    train = _parse_jsonl(_canonical_rows(train_rows), f"{VERSION}/train", labels, DEFAULT_LIMITS)
    validation = _parse_jsonl(_canonical_rows(validation_rows), f"{VERSION}/validation", labels, DEFAULT_LIMITS)
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
        "base": {"medical_version": "1.16.0", "medical_train_sha256": sha256(base_train_raw), "medical_validation_sha256": sha256(base_val_raw), "common_train_sha256": sha256(common_raw["train"]), "common_validation_sha256": sha256(common_raw["validation"]), "test_sha256": sha256(test_bytes), "test_sha256_expected": BASE_TEST_SHA256, "test_copied_byte_exact": True},
        "records": {"train": len(train), "validation": len(validation), "authored_train": len(authored["train"]), "authored_validation": len(authored["validation"]), "before_deduplication": {"train": train_merge["records_before_deduplication"], "validation": validation_merge["records_before_deduplication"]}, "deduplication": {"train": train_merge, "validation": validation_merge}},
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
        "common_quality_corrections": common_corrections,
        "legacy_quality_corrections": legacy_corrections,
        "training_policy": "Standalone 22-label snapshot of corrected common and medical train/validation plus the 74 reviewed authored cases; deduplicated exact duplicate text with matching annotations.",
        "evaluation_ready": False,
        "evaluation_readiness_reason": "No fresh independent test or challenge set exists for the expanded 22-label ontology.",
        "inherited_test_support_by_new_label": "not evaluated or claimed",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / f"{CORPUS_ID}-{VERSION}_import.zip").write_bytes(archive)
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
