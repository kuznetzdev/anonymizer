from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = CODE_ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "ner_1_17"))

import authored_scenarios  # noqa: E402
from app.training.corpus import (  # noqa: E402
    DEFAULT_LIMITS,
    _canonical_records_bytes,
    _load_archive,
    _make_release,
    _manifest_for_release,
    _parse_jsonl,
    load_ontology_labels,
)

CORPUS_ID = "anonymizer-ner-hardcases"
CORPUS_VERSION = "1.17.0"
SPLITS = ("train", "validation", "test")
AUTHORED_SPLITS = (*SPLITS, "challenge")
ACTIVE_LABELS = frozenset({
    "APART", "COUNTRY", "DISEASE", "EMAIL", "HOUSE", "INN", "INSTITUTION", "KPP",
    "LOCALITY", "OGRN", "ORG", "PASSPORT", "PER", "PHONE", "REGION", "SNILS", "STREET",
})
MAX_TEXT_CHARS = 20_000
OUTPUT_ROOT = REPO_ROOT / "references" / "corpora" / CORPUS_ID / "1.17.0-integrated" / "scenario-build-v1"
CHALLENGE_ROOT = REPO_ROOT / "references" / "evaluations" / "medical-1.17" / "challenge_sealed_v1"
BASE_TRAIN = REPO_ROOT / "references" / "corpora" / CORPUS_ID / "1.16.0" / "train.jsonl"
BASE_VALIDATION = REPO_ROOT / "references" / "corpora" / CORPUS_ID / "1.16.0" / "validation.jsonl"
BASE_SHA256 = {
    "train": "43a7f35e0d479ab0edf4abfa23acdd4832b96ad557c8a03a5882b69a3bae6d28",
    "validation": "dca4098b494da6b7e62d70c93126ddbd71e8df324b92568fc60c03ff3ad0949f",
}


class Entity(TypedDict):
    start: int
    end: int
    label: str


class Record(TypedDict):
    text: str
    entities: list[Entity]


@dataclass(frozen=True)
class Case:
    split: str
    group: str
    template: str
    label: str | None
    text: str
    entities: tuple[Entity, ...]
    expected_surfaces: tuple[str, ...]


def _checksum(value: str, weights: tuple[int, ...], modulus: int = 11) -> str:
    return str((sum(int(digit) * weight for digit, weight in zip(value, weights)) % modulus) % 10)


def _snils(base: str) -> str:
    total = sum(int(digit) * weight for digit, weight in zip(base, range(9, 0, -1)))
    check = "00" if total in (100, 101) else f"{total % 101:02d}"
    if int(check) == 100:
        check = "00"
    return f"{base}{check}"


def _inn(base: str) -> str:
    if len(base) == 9:
        weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        return base + _checksum(base, weights)
    if len(base) != 10:
        raise ValueError("INN checksum requires 9 or 10 payload digits")
    first = base + _checksum(base, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8))
    return first + _checksum(first, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8))


def _ogrn(base: str) -> str:
    check = str((int(base) % 11) % 10)
    return base + check


def _identifier(split: str, label: str) -> str:
    seed = hashlib.sha256(f"ner-1.17:{split}:{label}".encode()).hexdigest()

    def numeric(width: int) -> str:
        return f"{int(seed, 16) % (10 ** width):0{width}d}"

    if label == "SNILS":
        return _snils(numeric(9))
    if label == "INN":
        return _inn("5" + numeric(8))
    if label == "OGRN":
        return _ogrn("1" + numeric(11))
    if label == "KPP":
        return "77" + numeric(7)
    if label == "PASSPORT":
        digits = numeric(10)
        return f"{digits[:4]} {digits[4:]}"
    raise ValueError(f"Unsupported generated identifier label: {label}")


def _checksums_valid(label: str, value: str) -> bool:
    digits = "".join(char for char in value if char.isdigit())
    if label == "SNILS":
        return len(digits) == 11 and _snils(digits[:9]) == digits
    if label == "INN":
        return len(digits) in (10, 12) and _inn(digits[:-1] if len(digits) == 10 else digits[:9]) == digits
    if label == "OGRN":
        return len(digits) == 13 and _ogrn(digits[:12]) == digits
    return True


def _contains_surface(text: str, surface: str) -> bool:
    start = 0
    while (index := text.find(surface, start)) >= 0:
        end = index + len(surface)
        left_ok = index == 0 or not (surface[0].isalnum() and text[index - 1].isalnum())
        right_ok = end == len(text) or not (surface[-1].isalnum() and text[end].isalnum())
        if left_ok and right_ok:
            return True
        start = index + 1
    return False


def _tagged_text(split: str, label: str | None, value: str | None, template: str) -> tuple[str, tuple[Entity, ...]]:
    wrappers = {
        "train": ("В карте первичного приёма указано: ", ". План лечения согласован.", "; наблюдение продолжается."),
        "validation": ("В записи консультации отмечено: ", ". Данные сверены с пациентом.", "; контроль назначен."),
        "test": ("В выписке профильного отделения зафиксировано: ", ". Сведения внесены лечащим врачом.", "; эпизод закрыт."),
    }
    prefix, suffix, negative_suffix = wrappers[split]
    if label is None:
        return prefix + template + negative_suffix, ()
    assert value is not None
    start = len(prefix) + template.index("{value}")
    text = prefix + template.replace("{value}", value) + suffix
    return text, ({"start": start, "end": start + len(value), "label": label},)


_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "train": {
        "PER": ("Пациентка {value} поступила на плановое обследование", "clinical"),
        "DISEASE": ("В анамнезе отмечена {value}", "clinical"),
        "INSTITUTION": ("Направление выдано в {value}", "facility"),
        "ORG": ("Поставщик лабораторных реагентов — {value}", "organization"),
        "STREET": ("Корпус лаборатории расположен на улице {value}", "address"),
        "HOUSE": ("Номер дома в адресе пациента: {value}", "address"),
        "APART": ("Номер квартиры в карточке: {value}", "address"),
        "LOCALITY": ("Место рождения пациента — город {value}", "address"),
        "REGION": ("Регистрация указана в {value}", "address"),
        "COUNTRY": ("Гражданство пациента — {value}", "address"),
        "SNILS": ("Страховой номер пациента {value}", "identifier"),
        "PASSPORT": ("Паспортные данные пациента {value}", "identifier"),
        "INN": ("ИНН медицинского поставщика {value}", "identifier"),
        "KPP": ("КПП медицинского поставщика {value}", "identifier"),
        "OGRN": ("ОГРН медицинского поставщика {value}", "identifier"),
        "PHONE": ("Для связи с регистратурой оставлен номер {value}", "contact"),
        "EMAIL": ("Результат направить на адрес {value}", "contact"),
    },
    "validation": {
        "PER": ("Врач внесла в историю болезни имя пациента: {value}", "clinical"),
        "DISEASE": ("В дифференциальном диагнозе рассмотрена {value}", "clinical"),
        "INSTITUTION": ("Пациента принял дежурный специалист {value}", "facility"),
        "ORG": ("Расходный материал поставила компания {value}", "organization"),
        "STREET": ("В адресе филиала указана улица {value}", "address"),
        "HOUSE": ("Здание отделения отмечено номером {value}", "address"),
        "APART": ("Квартира пациента указана как {value}", "address"),
        "LOCALITY": ("Пациент приехал из населённого пункта {value}", "address"),
        "REGION": ("Медицинская карта открыта в регионе {value}", "address"),
        "COUNTRY": ("Страна рождения в анкете — {value}", "address"),
        "SNILS": ("Для сверки предъявлен СНИЛС {value}", "identifier"),
        "PASSPORT": ("Серия и номер предъявленного паспорта {value}", "identifier"),
        "INN": ("Реквизит клиники ИНН {value}", "identifier"),
        "KPP": ("Реквизит клиники КПП {value}", "identifier"),
        "OGRN": ("В регистрационной выписке указан ОГРН {value}", "identifier"),
        "PHONE": ("Регистратура перезвонит по номеру {value}", "contact"),
        "EMAIL": ("Электронную копию отправить на {value}", "contact"),
    },
    "test": {
        "PER": ("В выписном эпикризе пациент назван: {value}", "clinical"),
        "DISEASE": ("В качестве сопутствующего диагноза указана {value}", "clinical"),
        "INSTITUTION": ("Перевод оформлен в {value}", "facility"),
        "ORG": ("Оборудование передала организация {value}", "organization"),
        "STREET": ("Пункт забора находится на улице {value}", "address"),
        "HOUSE": ("В строке адреса указан дом {value}", "address"),
        "APART": ("Для доставки справки указана квартира {value}", "address"),
        "LOCALITY": ("Пациент проживает в городе {value}", "address"),
        "REGION": ("Территориальная принадлежность: {value}", "address"),
        "COUNTRY": ("Гражданство, указанное при поступлении: {value}", "address"),
        "SNILS": ("Номер индивидуального лицевого счёта {value}", "identifier"),
        "PASSPORT": ("Документ пациента: паспорт {value}", "identifier"),
        "INN": ("ИНН подрядчика по питанию {value}", "identifier"),
        "KPP": ("Код постановки подрядчика на учёт, КПП {value}", "identifier"),
        "OGRN": ("В договоре на обслуживание указан ОГРН {value}", "identifier"),
        "PHONE": ("Связаться с сопровождающим по телефону {value}", "contact"),
        "EMAIL": ("Копию выписки направить по адресу {value}", "contact"),
    },
}

_VALUES: dict[str, dict[str, str]] = {
    "train": {
        "PER": "Мария Соколова", "DISEASE": "болезнь Паркинсона", "INSTITUTION": "Городская клиническая больница № 8",
        "ORG": "ООО «МедСнаб Урал»", "STREET": "Липовая", "HOUSE": "17к2", "APART": "48",
        "LOCALITY": "Каменск-Уральский", "REGION": "Свердловская область", "COUNTRY": "Казахстан",
        "PHONE": "+7 912 345-67-89", "EMAIL": "m.sokolova@example.invalid",
    },
    "validation": {
        "PER": "Антонова Елена Игоревна", "DISEASE": "болезнь Бехтерева", "INSTITUTION": "Детская городская поликлиника № 4",
        "ORG": "АО «ЛабТех-Сервис»", "STREET": "Сосновый проезд", "HOUSE": "6Б", "APART": "113",
        "LOCALITY": "Великий Новгород", "REGION": "Республика Татарстан", "COUNTRY": "Армения",
        "PHONE": "+7 (921) 234-56-78", "EMAIL": "registry@example.invalid",
    },
    "test": {
        "PER": "Илья Воронцов", "DISEASE": "синдром Гийена—Барре", "INSTITUTION": "Областной онкологический диспансер",
        "ORG": "ООО «СеверМедПоставка»", "STREET": "Берёзовая набережная", "HOUSE": "102/3", "APART": "7",
        "LOCALITY": "Петрозаводск", "REGION": "Краснодарский край", "COUNTRY": "Беларусь",
        "PHONE": "8 (800) 123-45-67", "EMAIL": "care@example.invalid",
    },
}

_NEGATIVES = {
    "train": "Показатели контрольного образца в пределах референсного интервала; повторный забор не требуется",
    "validation": "Контроль качества серии завершён; отклонений от лабораторного протокола не выявлено",
    "test": "Плановое исследование выполнено; технический контроль пройден без замечаний",
}


def build_cases(
    ontology_labels: set[str],
    scenarios: dict[str, list[authored_scenarios.AuthoredRecord]] | None = None,
) -> tuple[Case, ...]:
    if not ACTIVE_LABELS <= ontology_labels - {"ADDR"}:
        raise ValueError("Ontology is missing registered 1.17 labels")
    cases: list[Case] = []
    for split in SPLITS:
        for label in sorted(ACTIVE_LABELS):
            if split == "test":
                continue
            template, domain = _TEMPLATES[split][label]
            value = _identifier(split, label) if label in {"INN", "KPP", "OGRN", "PASSPORT", "SNILS"} else _VALUES[split][label]
            text, entities = _tagged_text(split, label, value, template)
            cases.append(Case(split, f"{split}-{domain}-v1", f"{split}-{label.lower()}-v1", label, text, entities, (value,)))
        if split != "test":
            text, entities = _tagged_text(split, None, _NEGATIVES[split], _NEGATIVES[split])
            cases.append(Case(split, f"{split}-negative-v1", f"{split}-negative-v1", None, text, entities, ()))
    source_records = scenarios if scenarios is not None else authored_scenarios.build_records()
    authored_scenarios.validate_records(source_records)
    for split in authored_scenarios.SPLITS:
        for row in source_records[split]:
            entities = tuple(dict(entity) for entity in row["entities"])
            surfaces = tuple(row["text"][entity["start"]:entity["end"]] for entity in entities)
            label = entities[0]["label"] if len(entities) == 1 else None
            cases.append(Case(split, row["group"], row["template"], label, row["text"], entities, surfaces))
    validate_cases(cases, ontology_labels)
    return tuple(cases)


def validate_cases(cases: list[Case] | tuple[Case, ...], ontology_labels: set[str]) -> None:
    if not ACTIVE_LABELS <= ontology_labels - {"ADDR"}:
        raise ValueError("Ontology is missing registered 1.17 labels")
    seen_templates: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    supported: set[str] = set()
    known_surfaces = {surface for case in cases for surface in case.expected_surfaces if surface}
    for case in cases:
        if case.split not in AUTHORED_SPLITS or not case.group or not case.template:
            raise ValueError("Invalid split or source metadata")
        for registry, identifier, kind in (
            (seen_templates, case.template, "template"), (seen_groups, case.group, "source group"),
        ):
            previous = registry.setdefault(identifier, case.split)
            if previous != case.split:
                raise ValueError(f"{kind} crosses corpus splits: {identifier}")
        if case.text in seen_text:
            if seen_text[case.text] != case.split:
                raise ValueError("Duplicate text crosses corpus splits")
            raise ValueError("Duplicate authored text appears more than once")
        seen_text[case.text] = case.split
        if not case.text.strip() or len(case.text) > MAX_TEXT_CHARS:
            raise ValueError("Text is empty or exceeds the importer limit")
        if not case.entities and case.label is not None:
            raise ValueError("Negative record cannot have a positive label")
        if not case.entities:
            leaked_surface = next((surface for surface in known_surfaces if _contains_surface(case.text, surface)), None)
            if leaked_surface is not None:
                raise ValueError(
                    f"Negative record contains a known annotated entity surface in {case.group}: {leaked_surface!r}"
                )
        if case.label is not None and (case.label not in ACTIVE_LABELS or len(case.entities) != 1):
            raise ValueError("Single-label record metadata is inconsistent")
        if len(case.expected_surfaces) != len(case.entities):
            raise ValueError("Every annotation must retain its expected surface")
        previous_end = -1
        for index, entity in enumerate(case.entities):
            start, end, label = entity["start"], entity["end"], entity["label"]
            if start < previous_end or start < 0 or end <= start or end > len(case.text):
                raise ValueError("Entity span is invalid or overlapping")
            surface = case.text[start:end]
            if surface != case.expected_surfaces[index]:
                raise ValueError("Entity span does not match its expected surface")
            if label not in ACTIVE_LABELS:
                raise ValueError(f"Unsupported or deprecated label: {label}")
            if case.label and label != case.label:
                raise ValueError("Entity label differs from the case label")
            if label in {"SNILS", "INN", "OGRN"} and not _checksums_valid(label, surface):
                raise ValueError(f"Invalid {label} checksum: {surface}")
            supported.add(label)
            previous_end = end
    if supported != ACTIVE_LABELS:
        raise ValueError(f"Generated corpus does not cover all active labels: {sorted(ACTIVE_LABELS - supported)}")
    for split in SPLITS:
        split_cases = [case for case in cases if case.split == split]
        if not split_cases or not any(case.label is None and not case.entities for case in split_cases):
            if split != "test" or not split_cases:
                raise ValueError(f"Split {split} must include records; training and validation also need a true negative")


def ocr_substitute(text: str, entities: tuple[Entity, ...], start: int, end: int, replacement: str) -> tuple[str, tuple[Entity, ...]]:
    if start < 0 or end < start or end > len(text) or not replacement:
        raise ValueError("Invalid OCR edit range or replacement")
    if any(
        (start < entity["end"] and end > entity["start"])
        or (start == end and entity["start"] < start < entity["end"])
        for entity in entities
    ):
        raise ValueError("OCR edit overlaps an annotated span")
    delta = len(replacement) - (end - start)
    shifted = tuple(
        {**entity, "start": entity["start"] + (delta if entity["start"] >= end else 0),
         "end": entity["end"] + (delta if entity["end"] > end else 0)}
        for entity in entities
    )
    return text[:start] + replacement + text[end:], shifted


def _read_base_sources(train_path: Path, validation_path: Path, ontology_labels: set[str]) -> tuple[dict[str, bytes], dict[str, tuple[object, ...]]]:
    paths = {"train": train_path, "validation": validation_path}
    raw_by_split: dict[str, bytes] = {}
    parsed_by_split: dict[str, tuple[object, ...]] = {}
    for split, path in paths.items():
        if path.name != f"{split}.jsonl" or path.parent.name != "1.16.0":
            raise ValueError(f"Only copied 1.16.0 {split}.jsonl is an allowed source")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Cannot read approved 1.16.0 {split} source: {path}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != BASE_SHA256[split]:
            raise ValueError(f"Approved 1.16.0 {split} source hash mismatch")
        records = _parse_jsonl(raw, f"1.16.0/{split}.jsonl", ontology_labels, DEFAULT_LIMITS)
        raw_by_split[split] = raw
        parsed_by_split[split] = records
    return raw_by_split, parsed_by_split


def _authored_bytes(cases: list[Case], split: str) -> tuple[bytes, list[Case]]:
    rows = sorted((case for case in cases if case.split == split), key=lambda case: case.template)
    raw = b"".join(
        (json.dumps({"text": case.text, "entities": list(case.entities)}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for case in rows
    )
    return raw, rows


def _join_jsonl(base: bytes, authored: bytes) -> bytes:
    if not base:
        return authored
    return base + (b"" if base.endswith(b"\n") else b"\n") + authored


def _archive(split_bytes: dict[str, bytes], ontology_labels: set[str]) -> tuple[bytes, dict[str, object]]:
    parsed = {split: _parse_jsonl(raw, f"{split}.jsonl", ontology_labels, DEFAULT_LIMITS) for split, raw in split_bytes.items()}
    release = _make_release(CORPUS_ID, CORPUS_VERSION, parsed)
    manifest = _manifest_for_release(release)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        files = [("manifest.json", manifest)] + [(f"{split}.jsonl", split_bytes[split]) for split in SPLITS]
        for name, payload in files:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    payload = buffer.getvalue()
    _load_archive(payload, ontology_labels, DEFAULT_LIMITS)
    return payload, release.as_dict()


def archive_bytes(
    cases: list[Case], ontology_labels: set[str], base_train: bytes, base_validation: bytes,
) -> tuple[bytes, dict[str, object], dict[str, bytes], dict[str, bytes], dict[str, list[Case]]]:
    validate_cases(cases, ontology_labels)
    authored_by_split: dict[str, bytes] = {}
    authored_cases: dict[str, list[Case]] = {}
    split_bytes: dict[str, bytes] = {}
    for split in SPLITS:
        authored_by_split[split], authored_cases[split] = _authored_bytes(cases, split)
        base = base_train if split == "train" else base_validation if split == "validation" else b""
        split_bytes[split] = _join_jsonl(base, authored_by_split[split])
    archive, manifest = _archive(split_bytes, ontology_labels)
    return archive, manifest, split_bytes, authored_by_split, authored_cases


def source_metadata(authored_cases: dict[str, list[Case]], source_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "base_sources": {
            split: {"corpus_version": "1.16.0", "path_name": f"{split}.jsonl", "sha256": source_hashes[split], "copied_byte_for_byte": True}
            for split in ("train", "validation")
        },
        "authored_records": {
            split: [
                {"record_index": index, "source_group": case.group, "template_id": case.template, "domain": case.group.removeprefix(f"{split}-").removesuffix("-v1")}
                for index, case in enumerate(authored_cases[split])
            ]
            for split in SPLITS
        },
        "authored_group_assignments": {split: sorted({case.group for case in authored_cases[split]}) for split in SPLITS},
        "authored_template_assignments": {split: sorted(case.template for case in authored_cases[split]) for split in SPLITS},
    }


def challenge_metadata(cases: list[Case], raw: bytes, records: dict[str, list[authored_scenarios.AuthoredRecord]]) -> dict[str, object]:
    rows = sorted((case for case in cases if case.split == "challenge"), key=lambda case: case.template)
    module_support = authored_scenarios.support_report(records, minimum=30)["challenge"]
    return {
        "schema_version": "1.0.0",
        "split": "challenge",
        "record_count": len(rows),
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "source_groups": sorted({case.group for case in rows}),
        "template_ids": sorted(case.template for case in rows),
        "record_provenance": [
            {"record_index": index, "source_group": case.group, "template_id": case.template}
            for index, case in enumerate(rows)
        ],
        "target_support": module_support["support"],
        "blocked_target_labels_below_30": module_support["blocked_labels"],
        "support_gate_passed": module_support["evaluation_ready"],
        "review_complete": False,
        "evaluation_ready": False,
    }


def quality_report(
    cases: list[Case], archive: bytes, manifest: dict[str, object], split_bytes: dict[str, bytes],
    authored_by_split: dict[str, bytes], source_hashes: dict[str, str], ontology_labels: set[str],
) -> dict[str, object]:
    authored_counts = Counter(entity["label"] for case in cases for entity in case.entities)
    combined_records = {split: _parse_jsonl(raw, f"{split}.jsonl", ontology_labels, DEFAULT_LIMITS) for split, raw in split_bytes.items()}
    base_records = {"train": _parse_jsonl(split_bytes["train"][:len(split_bytes["train"]) - len(authored_by_split["train"])], "base/train.jsonl", ontology_labels, DEFAULT_LIMITS)}
    base_records["validation"] = _parse_jsonl(split_bytes["validation"][:len(split_bytes["validation"]) - len(authored_by_split["validation"])], "base/validation.jsonl", ontology_labels, DEFAULT_LIMITS)
    support: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        support[split] = dict(sorted(Counter(entity.label for record in combined_records[split] for entity in record.entities).items()))
    priority_labels = sorted(authored_scenarios.PRIORITY_LABELS)
    support_report = {
        split: {
            label: sum(entity["label"] == label for case in cases if case.split == split for entity in case.entities)
            for label in priority_labels
        }
        for split in AUTHORED_SPLITS
    }
    test_and_challenge_supported = all(
        support_report[split][label] >= 30 for split in ("test", "challenge") for label in priority_labels
    )
    editorial_review_complete = False
    return {
        "status": "structural_only_insufficient_for_independent_evaluation",
        "evaluation_ready": test_and_challenge_supported and editorial_review_complete,
        "editorial_review_complete": editorial_review_complete,
        "source": "immutable 1.16.0 train/validation copies plus locally authored synthetic additions; no 1.16.0 test/challenge or external data read",
        "source_sha256": source_hashes,
        "record_counts": dict(manifest["record_counts"]),
        "base_record_counts": {split: len(rows) for split, rows in base_records.items()},
        "authored_record_counts": {split: sum(case.split == split for case in cases) for split in SPLITS},
        "authored_label_support": dict(sorted(authored_counts.items())),
        "target_labels": priority_labels,
        "target_support_by_split": support_report,
        "target_support_gate": {"minimum_per_label": 30, "test_and_challenge_passed": test_and_challenge_supported},
        "combined_label_support": support,
        "negative_records": sum(case.label is None and not case.entities for case in cases),
        "source_groups": {split: sorted({case.group for case in cases if case.split == split}) for split in AUTHORED_SPLITS},
        "template_ids": {split: sorted(case.template for case in cases if case.split == split) for split in AUTHORED_SPLITS},
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "release_sha256": manifest["release_id"],
        "checksums_validated": ["SNILS", "INN", "OGRN"],
        "limitations": ["Authored support is sparse and does not meet the minimum 30 examples per target label in test and challenge.", "All held-out authored records require blind editorial review.", "Structural/import checks do not establish model-quality improvement."],
    }


def _write_identical_or_new(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace a different private corpus artifact: {path}")
        return
    with path.open("xb") as output:
        output.write(content)


def write_release(output_dir: Path, base_train_path: Path, base_validation_path: Path) -> None:
    target = output_dir.resolve()
    references = (REPO_ROOT / "references").resolve()
    try:
        target.relative_to(references)
    except ValueError as exc:
        raise ValueError("Corpus output must remain under the ignored references/ directory") from exc
    ontology_path = CODE_ROOT / "data" / "corpora" / "ru-pii" / "v1" / "ontology.json"
    labels = load_ontology_labels(ontology_path)
    base_sources, base_records = _read_base_sources(base_train_path, base_validation_path, labels)
    scenario_records = authored_scenarios.build_records()
    cases = list(build_cases(labels, scenario_records))
    archive, manifest, split_bytes, authored_by_split, authored_cases = archive_bytes(
        cases, labels, base_sources["train"], base_sources["validation"]
    )
    challenge_raw, challenge_cases = _authored_bytes(cases, "challenge")
    source_hashes = {split: hashlib.sha256(raw).hexdigest() for split, raw in base_sources.items()}
    # Fail before publishing if source copies or authored rows overlap across the combined corpus.
    base_texts = {record.text for records in base_records.values() for record in records}
    authored_texts = {case.text for case in cases}
    if base_texts & authored_texts:
        raise ValueError("Authored text duplicates an immutable 1.16.0 source record")
    if not challenge_cases:
        raise ValueError("Authored challenge split is empty")
    main_texts = {case.text for case in cases if case.split != "challenge"}
    if main_texts & {case.text for case in challenge_cases}:
        raise ValueError("Challenge text overlaps a train, validation, or test record")
    target.mkdir(parents=True, exist_ok=True)
    (target / "authored").mkdir(parents=True, exist_ok=True)
    challenge_dir = CHALLENGE_ROOT
    challenge_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target / f"{CORPUS_ID}-{CORPUS_VERSION}_import.zip"
    report = (json.dumps(quality_report(cases, archive, manifest, split_bytes, authored_by_split, source_hashes, labels), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    report = report.replace(b"\n", os.linesep.encode("ascii"))
    metadata = (json.dumps(source_metadata(authored_cases, source_hashes), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    metadata = metadata.replace(b"\n", os.linesep.encode("ascii"))
    heldout_metadata = (json.dumps(challenge_metadata(cases, challenge_raw, scenario_records), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    heldout_metadata = heldout_metadata.replace(b"\n", os.linesep.encode("ascii"))
    _write_identical_or_new(archive_path, archive)
    _write_identical_or_new(target / "quality_report.json", report)
    _write_identical_or_new(target / "source_groups.json", metadata)
    for split in SPLITS:
        _write_identical_or_new(target / f"{split}.jsonl", split_bytes[split])
        _write_identical_or_new(target / "authored" / f"{split}.jsonl", authored_by_split[split])
    _write_identical_or_new(challenge_dir / "records.jsonl", challenge_raw)
    _write_identical_or_new(challenge_dir / "metadata.json", heldout_metadata)
    print(f"Wrote private synthetic corpus archive: {archive_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the private synthetic 1.17.0 hardcases corpus.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--base-train", type=Path, default=BASE_TRAIN)
    parser.add_argument("--base-validation", type=Path, default=BASE_VALIDATION)
    args = parser.parse_args()
    write_release(args.output_dir, args.base_train, args.base_validation)


if __name__ == "__main__":
    main()
