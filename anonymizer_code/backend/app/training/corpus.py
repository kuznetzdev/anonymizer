"""Validate, install, and read versioned PII corpora without loading test data."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import MappingProxyType

SCHEMA_VERSION = "1.0.0"
SPLITS = ("train", "validation", "test")
RELEASE_FILES = {"manifest.json", *(f"{split}.jsonl" for split in SPLITS)}
LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
CORPUS_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
VERSION_RE = re.compile(r"^[0-9]{1,9}\.[0-9]{1,9}\.[0-9]{1,9}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class CorpusLimits:
    """Hard import limits to bound decompression and parsed corpus size."""

    max_archive_bytes: int = 64 * 1024 * 1024
    max_member_bytes: int = 64 * 1024 * 1024
    max_total_uncompressed_bytes: int = 64 * 1024 * 1024
    max_records_total: int = 100_000
    max_text_chars: int = 20_000
    max_entities_per_record: int = 1_000


DEFAULT_LIMITS = CorpusLimits()


class CorpusImportError(ValueError):
    """Raised when an archive or release violates the corpus contract."""


@dataclass(frozen=True)
class CorpusRelease:
    corpus_id: str
    corpus_version: str
    schema_version: str
    sha256: str
    record_counts: Mapping[str, int]
    labels: tuple[str, ...]
    split_sha256: Mapping[str, str]

    @property
    def release_id(self) -> str:
        return self.sha256

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "sha256": self.sha256,
            "record_counts": dict(self.record_counts),
            "labels": list(self.labels),
            "split_sha256": dict(self.split_sha256),
        }


@dataclass(frozen=True)
class _Entity:
    start: int
    end: int
    label: str

    def as_dict(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "label": self.label}


@dataclass(frozen=True)
class _Record:
    text: str
    entities: tuple[_Entity, ...]

    def as_dict(self) -> dict[str, object]:
        return {"text": self.text, "entities": [entity.as_dict() for entity in self.entities]}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusImportError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CorpusImportError(f"Unsupported JSON value: {value}")


def _decode_json(raw: bytes, source: str) -> object:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusImportError(f"Invalid UTF-8 or JSON in {source}") from exc


def _decode_json_line(line: str, source: str, line_number: int) -> object:
    try:
        return json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise CorpusImportError(f"Invalid JSON in {source} at line {line_number}") from exc


def load_ontology_descriptions(path: Path) -> dict[str, str]:
    """Load label descriptions from a tracked ontology file and reject malformed entries."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CorpusImportError(f"Cannot read ontology: {path}") from exc
    value = _decode_json(raw, "ontology")
    if not isinstance(value, dict) or set(value) != {"schema_version", "categories"}:
        raise CorpusImportError("Ontology must contain schema_version and categories")
    if value["schema_version"] != SCHEMA_VERSION:
        raise CorpusImportError("Unsupported ontology schema_version")
    categories = value["categories"]
    if not isinstance(categories, list) or not categories:
        raise CorpusImportError("Ontology categories must be a non-empty array")
    descriptions: dict[str, str] = {}
    for category in categories:
        if not isinstance(category, dict) or set(category) != {"label", "description"}:
            raise CorpusImportError("Each ontology category needs label and description")
        label, description = category["label"], category["description"]
        if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
            raise CorpusImportError("Ontology labels must be uppercase identifiers")
        if not isinstance(description, str) or not description.strip():
            raise CorpusImportError(f"Ontology category {label} needs a description")
        if label in descriptions:
            raise CorpusImportError(f"Duplicate ontology label: {label}")
        descriptions[label] = description.strip()
    return descriptions


def load_ontology_labels(path: Path) -> set[str]:
    return set(load_ontology_descriptions(path))


def _parse_manifest(raw: bytes) -> tuple[str, str, dict[str, int]]:
    manifest = _decode_json(raw, "manifest.json")
    if not isinstance(manifest, dict):
        raise CorpusImportError("manifest.json must contain an object")
    required = {"corpus_id", "corpus_version", "schema_version", "record_counts"}
    if not required.issubset(manifest):
        raise CorpusImportError("manifest.json is missing required fields")
    corpus_id, corpus_version, schema_version = (
        manifest["corpus_id"], manifest["corpus_version"], manifest["schema_version"]
    )
    if not isinstance(corpus_id, str) or not CORPUS_ID_RE.fullmatch(corpus_id):
        raise CorpusImportError("Invalid corpus_id")
    if not isinstance(corpus_version, str) or not VERSION_RE.fullmatch(corpus_version):
        raise CorpusImportError("Invalid corpus_version")
    if schema_version != SCHEMA_VERSION:
        raise CorpusImportError(f"schema_version must be {SCHEMA_VERSION}")
    raw_counts = manifest["record_counts"]
    if not isinstance(raw_counts, dict) or set(raw_counts) != set(SPLITS):
        raise CorpusImportError("record_counts must name train, validation, and test")
    counts: dict[str, int] = {}
    for split in SPLITS:
        count = raw_counts[split]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CorpusImportError(f"Invalid {split} record count")
        counts[split] = count
    return corpus_id, corpus_version, counts


def _canonical_records_bytes(records: tuple[_Record, ...]) -> bytes:
    return b"".join(
        (json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def _make_release(
    corpus_id: str,
    corpus_version: str,
    records_by_split: Mapping[str, tuple[_Record, ...]],
) -> CorpusRelease:
    record_counts = {split: len(records_by_split[split]) for split in SPLITS}
    split_hashes = {
        split: hashlib.sha256(_canonical_records_bytes(records_by_split[split])).hexdigest()
        for split in SPLITS
    }
    labels = tuple(sorted({entity.label for records in records_by_split.values() for record in records for entity in record.entities}))
    digest_document = {
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "schema_version": SCHEMA_VERSION,
        "record_counts": record_counts,
        "labels": list(labels),
        "split_sha256": split_hashes,
    }
    digest = hashlib.sha256(
        json.dumps(digest_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CorpusRelease(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        schema_version=SCHEMA_VERSION,
        sha256=digest,
        record_counts=MappingProxyType(record_counts),
        labels=labels,
        split_sha256=MappingProxyType(split_hashes),
    )


def _validate_record(value: object, source: str, line_number: int, labels: Collection[str], limits: CorpusLimits) -> _Record:
    if not isinstance(value, dict) or set(value) != {"text", "entities"}:
        raise CorpusImportError(f"Invalid record fields in {source} at line {line_number}")
    text, raw_entities = value["text"], value["entities"]
    if not isinstance(text, str) or not text.strip() or len(text) > limits.max_text_chars:
        raise CorpusImportError(f"Invalid text in {source} at line {line_number}")
    if not isinstance(raw_entities, list) or len(raw_entities) > limits.max_entities_per_record:
        raise CorpusImportError(f"Invalid entities in {source} at line {line_number}")
    entities: list[_Entity] = []
    for raw_entity in raw_entities:
        if not isinstance(raw_entity, dict):
            raise CorpusImportError(f"Invalid entity in {source} at line {line_number}")
        if set(raw_entity) not in ({"start", "end", "label"}, {"start", "end", "label", "surface"}):
            raise CorpusImportError(f"Invalid entity fields in {source} at line {line_number}")
        start, end, label = raw_entity["start"], raw_entity["end"], raw_entity["label"]
        if (isinstance(start, bool) or not isinstance(start, int) or
                isinstance(end, bool) or not isinstance(end, int) or
                not isinstance(label, str) or label not in labels or
                start < 0 or start >= end or end > len(text)):
            raise CorpusImportError(f"Invalid entity span or label in {source} at line {line_number}")
        if "surface" in raw_entity and raw_entity["surface"] != text[start:end]:
            raise CorpusImportError(f"Entity surface mismatch in {source} at line {line_number}")
        entities.append(_Entity(start, end, label))
    entities.sort(key=lambda entity: (entity.start, entity.end, entity.label))
    for index in range(1, len(entities)):
        if entities[index].start < entities[index - 1].end:
            raise CorpusImportError(f"Overlapping entities in {source} at line {line_number}")
    return _Record(text=text, entities=tuple(entities))


def _parse_jsonl(
    raw: bytes,
    source: str,
    labels: Collection[str],
    limits: CorpusLimits,
    max_records: int | None = None,
) -> tuple[_Record, ...]:
    if not raw:
        return ()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CorpusImportError(f"Invalid UTF-8 in {source}") from exc
    records: list[_Record] = []
    lines = text.split("\n")
    for line_number, line in enumerate(lines, start=1):
        if line.endswith("\r"):
            line = line[:-1]
        if not line and line_number == len(lines) and text.endswith("\n"):
            continue
        if not line.strip():
            raise CorpusImportError(f"Blank JSONL line in {source} at line {line_number}")
        records.append(_validate_record(_decode_json_line(line, source, line_number), source, line_number, labels, limits))
        if len(records) > min(limits.max_records_total, max_records if max_records is not None else limits.max_records_total):
            raise CorpusImportError("Corpus exceeds record limit")
    return tuple(records)


def _validate_archive_members(archive: zipfile.ZipFile, archive_size: int, limits: CorpusLimits) -> dict[str, zipfile.ZipInfo]:
    if archive_size > limits.max_archive_bytes:
        raise CorpusImportError("Archive exceeds compressed size limit")
    infos = archive.infolist()
    if len(infos) != len(RELEASE_FILES):
        raise CorpusImportError("Archive must contain exactly four root files")
    by_name: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for info in infos:
        name = info.filename
        if name not in RELEASE_FILES or "/" in name or "\\" in name or name in by_name:
            raise CorpusImportError("Archive has duplicate, nested, or unsafe members")
        if info.is_dir() or info.flag_bits & 0x1:
            raise CorpusImportError("Archive directories and encrypted members are forbidden")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise CorpusImportError("Archive compression method is unsupported")
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type not in (0, stat.S_IFREG):
            raise CorpusImportError("Archive special files and symlinks are forbidden")
        if info.file_size < 0 or info.file_size > limits.max_member_bytes:
            raise CorpusImportError("Archive member exceeds uncompressed size limit")
        total_size += info.file_size
        by_name[name] = info
    if set(by_name) != RELEASE_FILES:
        raise CorpusImportError("Archive members do not match the required file set")
    if total_size > limits.max_total_uncompressed_bytes:
        raise CorpusImportError("Archive exceeds total uncompressed size limit")
    return by_name


def _read_archive_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    try:
        with archive.open(info, "r") as source:
            content = source.read(limit + 1)
            if len(content) > limit or source.read(1):
                raise CorpusImportError("Archive member exceeds uncompressed size limit")
            return content
    except (zipfile.BadZipFile, EOFError, RuntimeError, OSError) as exc:
        raise CorpusImportError("Invalid or unreadable ZIP member") from exc


def _load_archive(
    archive_bytes: bytes,
    known_labels: Collection[str],
    limits: CorpusLimits,
) -> tuple[CorpusRelease, dict[str, tuple[_Record, ...]]]:
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            members = _validate_archive_members(archive, len(archive_bytes), limits)
            manifest_bytes = _read_archive_member(archive, members["manifest.json"], limits.max_member_bytes)
            actual_total_size = len(manifest_bytes)
            corpus_id, corpus_version, expected_counts = _parse_manifest(manifest_bytes)
            parsed: dict[str, tuple[_Record, ...]] = {}
            text_split: dict[str, str] = {}
            total_records = 0
            for split in SPLITS:
                split_bytes = _read_archive_member(archive, members[f"{split}.jsonl"], limits.max_member_bytes)
                actual_total_size += len(split_bytes)
                if actual_total_size > limits.max_total_uncompressed_bytes:
                    raise CorpusImportError("Archive exceeds total uncompressed size limit")
                records = _parse_jsonl(
                    split_bytes, f"{split}.jsonl", known_labels, limits,
                    max_records=limits.max_records_total - total_records,
                )
                if len(records) != expected_counts[split]:
                    raise CorpusImportError(f"{split} record count does not match manifest")
                total_records += len(records)
                for record in records:
                    previous_split = text_split.setdefault(record.text, split)
                    if previous_split != split:
                        raise CorpusImportError("Duplicate text appears across corpus splits")
                parsed[split] = records
            if total_records > limits.max_records_total:
                raise CorpusImportError("Corpus exceeds record limit")
            release = _make_release(corpus_id, corpus_version, parsed)
            return release, parsed
    except (zipfile.BadZipFile, EOFError, RuntimeError, OSError) as exc:
        raise CorpusImportError("Invalid or unreadable ZIP archive") from exc


def _manifest_for_release(release: CorpusRelease) -> bytes:
    payload = {
        "corpus_id": release.corpus_id,
        "corpus_version": release.corpus_version,
        "schema_version": release.schema_version,
        "record_counts": dict(release.record_counts),
        "labels": list(release.labels),
        "split_sha256": dict(release.split_sha256),
        "sha256": release.sha256,
    }
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as destination:
        destination.write(content)
        destination.flush()
        os.fsync(destination.fileno())


def _publish(release: CorpusRelease, records: Mapping[str, tuple[_Record, ...]], corpus_root: Path) -> Path:
    root = corpus_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    releases_dir = root / "releases"
    releases_dir.mkdir(exist_ok=True)
    if releases_dir.is_symlink() or not releases_dir.is_dir():
        raise CorpusImportError("Release directory must be a regular directory")
    target = releases_dir / release.sha256
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Corpus release already exists: {release.sha256}")
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=releases_dir))
    try:
        _write_file(staging / "manifest.json", _manifest_for_release(release))
        for split in SPLITS:
            _write_file(staging / f"{split}.jsonl", _canonical_records_bytes(records[split]))
        try:
            os.rename(staging, target)
        except FileExistsError as exc:
            raise FileExistsError(f"Corpus release already exists: {release.sha256}") from exc
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def install_corpus_archive(
    archive_path: Path,
    corpus_root: Path,
    additional_labels: Collection[str] = (),
) -> CorpusRelease:
    """Validate and atomically install an archive under releases/<sha256>."""
    labels = load_ontology_labels(corpus_root / "v1" / "ontology.json") | set(additional_labels)
    try:
        with archive_path.open("rb") as source:
            archive_bytes = source.read(DEFAULT_LIMITS.max_archive_bytes + 1)
    except OSError as exc:
        raise CorpusImportError(f"Cannot read corpus archive: {archive_path}") from exc
    if len(archive_bytes) > DEFAULT_LIMITS.max_archive_bytes:
        raise CorpusImportError("Archive exceeds compressed size limit")
    release, records = _load_archive(archive_bytes, labels, DEFAULT_LIMITS)
    _publish(release, records, corpus_root)
    return release


def _read_release_metadata(release_dir: Path, expected_id: str) -> CorpusRelease:
    if release_dir.is_symlink() or not release_dir.is_dir():
        raise CorpusImportError("Release must be a regular directory")
    manifest_path = release_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CorpusImportError("Release manifest is missing or unsafe")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise CorpusImportError("Release manifest exceeds size limit")
    manifest = _decode_json(manifest_path.read_bytes(), "release manifest")
    if not isinstance(manifest, dict) or set(manifest) != {
        "corpus_id", "corpus_version", "schema_version", "record_counts", "labels", "split_sha256", "sha256"
    }:
        raise CorpusImportError("Invalid release manifest fields")
    corpus_id, corpus_version, counts = _parse_manifest(
        json.dumps(
            {key: manifest[key] for key in ("corpus_id", "corpus_version", "schema_version", "record_counts")},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    labels = manifest["labels"]
    hashes = manifest["split_sha256"]
    digest = manifest["sha256"]
    if (not isinstance(labels, list) or any(not isinstance(label, str) or not LABEL_RE.fullmatch(label) for label in labels)
            or labels != sorted(set(labels))):
        raise CorpusImportError("Invalid release labels")
    if not isinstance(hashes, dict) or set(hashes) != set(SPLITS) or any(
        not isinstance(value, str) or not SHA256_RE.fullmatch(value) for value in hashes.values()
    ):
        raise CorpusImportError("Invalid release split hashes")
    if not isinstance(digest, str) or digest != expected_id or not SHA256_RE.fullmatch(digest):
        raise CorpusImportError("Release id does not match its manifest")
    release = CorpusRelease(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        schema_version=SCHEMA_VERSION,
        sha256=digest,
        record_counts=MappingProxyType(counts),
        labels=tuple(labels),
        split_sha256=MappingProxyType(dict(hashes)),
    )
    expected_digest = _make_digest(release)
    if expected_digest != digest:
        raise CorpusImportError("Release metadata digest is invalid")
    return release


def _make_digest(release: CorpusRelease) -> str:
    payload = {
        "corpus_id": release.corpus_id,
        "corpus_version": release.corpus_version,
        "schema_version": release.schema_version,
        "record_counts": dict(release.record_counts),
        "labels": list(release.labels),
        "split_sha256": dict(release.split_sha256),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def list_corpus_releases(corpus_root: Path) -> list[CorpusRelease]:
    """List validated immutable release metadata without loading dataset rows."""
    releases_dir = corpus_root.resolve() / "releases"
    if releases_dir.is_symlink():
        raise CorpusImportError("Release directory must be a regular directory")
    if not releases_dir.exists():
        return []
    if releases_dir.is_symlink() or not releases_dir.is_dir():
        raise CorpusImportError("Release directory must be a regular directory")
    releases: list[CorpusRelease] = []
    for child in sorted(releases_dir.iterdir(), key=lambda path: path.name):
        if child.name.startswith(".staging-"):
            continue
        if not SHA256_RE.fullmatch(child.name):
            raise CorpusImportError("Unexpected entry in release directory")
        releases.append(_read_release_metadata(child, child.name))
    return releases


def _release_records(release_dir: Path, split: str, release: CorpusRelease) -> tuple[_Record, ...]:
    path = release_dir / f"{split}.jsonl"
    if path.is_symlink() or not path.is_file():
        raise CorpusImportError(f"Release {split} file is missing or unsafe")
    if path.stat().st_size > DEFAULT_LIMITS.max_member_bytes:
        raise CorpusImportError(f"Release {split} file exceeds size limit")
    records = _parse_jsonl(path.read_bytes(), f"release {split}", release.labels, DEFAULT_LIMITS)
    if len(records) != release.record_counts[split]:
        raise CorpusImportError(f"Release {split} count does not match manifest")
    if hashlib.sha256(_canonical_records_bytes(records)).hexdigest() != release.split_sha256[split]:
        raise CorpusImportError(f"Release {split} digest does not match manifest")
    return records


def load_corpus_records(corpus_root: Path, release_ids: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load selected train and validation rows; the test file is never opened."""
    if len(release_ids) != len(set(release_ids)):
        raise CorpusImportError("Duplicate release ids are not allowed")
    root = corpus_root.resolve()
    releases_dir = root / "releases"
    if releases_dir.is_symlink():
        raise CorpusImportError("Release directory must be a regular directory")
    if release_ids and not releases_dir.is_dir():
        raise CorpusImportError("Release directory is missing")
    train: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    seen: dict[str, tuple[str, tuple[tuple[int, int, str], ...]]] = {}
    for release_id in release_ids:
        if not SHA256_RE.fullmatch(release_id):
            raise CorpusImportError("Release ids must be lowercase SHA-256 digests")
        release_dir = releases_dir / release_id
        release = _read_release_metadata(release_dir, release_id)
        if len(train) + len(validation) + release.record_counts["train"] + release.record_counts["validation"] > DEFAULT_LIMITS.max_records_total:
            raise CorpusImportError("Selected releases exceed record limit")
        train_records = _release_records(release_dir, "train", release)
        validation_records = _release_records(release_dir, "validation", release)
        for split, records, destination in (
            ("train", train_records, train),
            ("validation", validation_records, validation),
        ):
            for record in records:
                signature = tuple((entity.start, entity.end, entity.label) for entity in record.entities)
                previous = seen.get(record.text)
                if previous is not None:
                    if previous[0] != split:
                        raise CorpusImportError("Duplicate text appears across selected training splits")
                    if previous[1] != signature:
                        raise CorpusImportError("Conflicting annotations for duplicate text in selected releases")
                    continue
                seen[record.text] = (split, signature)
                destination.append(record.as_dict())
        if len(train) + len(validation) > DEFAULT_LIMITS.max_records_total:
            raise CorpusImportError("Selected releases exceed record limit")
    return train, validation


def load_corpus_holdout_texts(corpus_root: Path, release_ids: list[str]) -> set[str]:
    """Reserve test texts against training leakage; never return test annotations."""
    catalog = {release.release_id: release for release in list_corpus_releases(corpus_root)}
    if len(release_ids) != len(set(release_ids)) or any(release_id not in catalog for release_id in release_ids):
        raise CorpusImportError("Invalid or missing held-out corpus release")
    if sum(catalog[release_id].record_counts["test"] for release_id in release_ids) > DEFAULT_LIMITS.max_records_total:
        raise CorpusImportError("Selected test splits exceed record limit")
    texts: set[str] = set()
    for release_id in release_ids:
        release_dir = corpus_root.resolve() / "releases" / release_id
        texts.update(record.text for record in _release_records(release_dir, "test", catalog[release_id]))
    return texts
