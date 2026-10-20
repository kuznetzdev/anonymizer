"""CPU-only token classifier with sparse, pickle-free NPZ artifacts."""
from __future__ import annotations

import json
import re
from io import BytesIO
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

LEGACY_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TOKEN_RE = re.compile(r"\d+|[^\W\d_]+|_|[^\w\s]", re.UNICODE)
PHONE_RE = re.compile(r"(?<!\w)\+7[\s().-]*\d(?:[\s().-]*\d){8,9}(?!\d)")
# The all-space grouping is retained for compatibility with immutable v1.7 artifacts.
SNILS_RE = re.compile(
    r"(?:[0-9]{11}|[0-9]{3}-[0-9]{3}-[0-9]{3}(?: [0-9]{2}|-[0-9]{2})|"
    r"[0-9]{3} [0-9]{3} [0-9]{3} [0-9]{2})\Z"
)
LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
MAX_TEXT = 20000
FEATURES = 1 << 14
BATCH_SIZE = 4096
STREET_CONTEXT = frozenset({
    "аллея", "бульвар", "наб", "набережная", "пер", "переулок", "пл", "площадь",
    "просп", "проспект", "проезд", "ш", "шоссе", "тупик", "ул", "улица", "улице",
    "улицы", "улицу", "улицей",
})
PERSON_CONTEXT = frozenset({
    "врач", "врача", "врачи", "врачом", "доктор", "доктора", "доктором", "пациент",
    "пациента", "пациентка", "пациенту", "гражданин", "гражданка", "господин", "госпожа",
    "сотрудник", "сотрудница", "коллега", "медсестра", "фельдшер",
})


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _tokenize_legacy(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in LEGACY_TOKEN_RE.finditer(text)]


def shape(value: str) -> str:
    value = re.sub(r"\d", "0", value)
    value = re.sub(r"[A-ZА-ЯЁ]", "A", value)
    return re.sub(r"[a-zа-яё]", "a", value)[:16]


def features(tokens: Sequence[tuple[str, int, int]], index: int) -> dict[str, float]:
    token = tokens[index][0]
    result = {"bias": 1.0, "word=" + token.lower(): 1.0, "shape=" + shape(token): 1.0}
    for offset in (-2, -1, 1, 2):
        pos = index + offset
        if 0 <= pos < len(tokens):
            result[f"w{offset}=" + tokens[pos][0].lower()] = 1.0
    if token[:1].isupper():
        result["capitalized"] = 1.0
    if token.isdigit():
        result["digits"] = 1.0
        result["len=" + str(min(len(token), 16))] = 1.0
    for n in (2, 3, 4):
        if len(token) >= n:
            result["pre=" + token[:n].lower()] = 1.0
            result["suf=" + token[-n:].lower()] = 1.0
    return result


def role_features(tokens: Sequence[tuple[str, int, int]], index: int) -> dict[str, float]:
    """Version 3 feature contract: bind tokens to nearby role cues."""
    result = features(tokens, index)
    token = tokens[index][0].lower()
    for offset in (-4, -3, -2, -1, 1, 2, 3, 4):
        pos = index + offset
        if not 0 <= pos < len(tokens):
            continue
        cue = tokens[pos][0].lower()
        if cue in STREET_CONTEXT:
            result[f"role=street|offset={offset}|word={token}"] = 1.0
        if cue in PERSON_CONTEXT:
            result[f"role=person|offset={offset}|word={token}"] = 1.0
    return result


def labels_for_tokens(
    record: dict[str, Any], tokens: Sequence[tuple[str, int, int]], active: set[str],
    record_index: int | None = None,
) -> list[str]:
    labels = ["O"] * len(tokens)
    for entity_index, entity in enumerate(record["entities"]):
        label = entity["label"]
        if label not in active:
            continue
        matched = [i for i, (_, start, end) in enumerate(tokens) if start < entity["end"] and end > entity["start"]]
        if (not matched or tokens[matched[0]][1] != entity["start"] or
                tokens[matched[-1]][2] != entity["end"]):
            where = f"record {record_index}" if record_index is not None else "record"
            raise ValueError(
                f"Entity token alignment failed at {where}, entity {entity_index}: "
                f"offsets {entity['start']}:{entity['end']}, matched {len(matched)} tokens"
            )
        for n, i in enumerate(matched):
            labels[i] = ("B-" if n == 0 else "I-") + label
    return labels


def _validate_active_labels(records: list[dict[str, Any]], active: set[str], source: str) -> None:
    for record_index, record in enumerate(records):
        for entity_index, entity in enumerate(record["entities"]):
            if entity["label"] not in active:
                raise ValueError(
                    f"Unknown active label in {source} at record {record_index}, "
                    f"entity {entity_index}: {entity['label']}"
                )
        labels_for_tokens(record, tokenize(record["text"]), active, record_index)


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, list):
        raise ValueError(f"Expected dataset array: {path}")
    return _validated_unique_records(data, path)


def _validated_unique_records(records: list[Any], source: str) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    annotations: dict[str, tuple[tuple[int, int, str], ...]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"text", "entities"}:
            raise ValueError(f"Invalid record schema in {source} at index {index}")
        text, entities = record["text"], record["entities"]
        if not isinstance(text, str) or not text or len(text) > MAX_TEXT or not isinstance(entities, list):
            raise ValueError(f"Invalid record values in {source} at index {index}")
        normalized: list[tuple[int, int, str]] = []
        for entity in entities:
            if not isinstance(entity, dict) or set(entity) != {"start", "end", "label"}:
                raise ValueError(f"Invalid entity schema in {source} at index {index}")
            start, end, label = entity["start"], entity["end"], entity["label"]
            if (isinstance(start, bool) or not isinstance(start, int) or
                    isinstance(end, bool) or not isinstance(end, int) or
                    not isinstance(label, str) or not LABEL_RE.fullmatch(label) or
                    start < 0 or start >= end or end > len(text)):
                raise ValueError(f"Invalid entity in {source} at index {index}")
            normalized.append((start, end, label))
        normalized.sort()
        if any(normalized[i][0] < normalized[i - 1][1] for i in range(1, len(normalized))):
            raise ValueError(f"Overlapping entities in {source} at index {index}")
        signature = tuple(normalized)
        if text in unique:
            if annotations[text] != signature:
                raise ValueError(f"Conflicting annotations for duplicate text in {source}")
            continue
        unique[text] = {"text": text, "entities": [
            {"start": start, "end": end, "label": label}
            for start, end, label in normalized
        ]}
        annotations[text] = signature
    return list(unique.values())


def _merge_reviewed_examples(
    corpus: list[dict[str, Any]], examples: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], int]:
    if not examples:
        return corpus, 0
    reviewed = _validated_unique_records(examples, "reviewed examples")
    by_text = {record["text"]: record for record in corpus}
    by_text.update({record["text"]: record for record in reviewed})
    return list(by_text.values()), len(reviewed)


def train_model(
    active_labels: set[str],
    corpus_train: list[dict[str, Any]],
    corpus_validation: list[dict[str, Any]],
    examples: list[dict[str, Any]] | None = None,
    *,
    heldout_texts: set[str] | None = None,
    initial_artifact: bytes | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if not active_labels:
        raise ValueError("At least one active label is required")
    train = _validated_unique_records(corpus_train, "training corpus")
    val = _validated_unique_records(corpus_validation, "validation corpus")
    if not train or not val:
        raise ValueError("Training and validation splits must both contain records")
    heldout = heldout_texts or set()
    if any(record["text"] in heldout for record in (*train, *val)):
        raise ValueError("Corpus training or validation records overlap test holdout")
    validation_texts = {record["text"] for record in val}
    corpus_validation_duplicates = sum(record["text"] in validation_texts for record in train)
    train = [record for record in train if record["text"] not in validation_texts]
    reviewed_records = _validated_unique_records(examples or [], "reviewed examples")
    reviewed_test_duplicates = sum(record["text"] in heldout for record in reviewed_records)
    reviewed_records = [record for record in reviewed_records if record["text"] not in heldout]
    reviewed_validation_duplicates = sum(record["text"] in validation_texts for record in reviewed_records)
    reviewed_records = [record for record in reviewed_records if record["text"] not in validation_texts]
    train, reviewed_count = _merge_reviewed_examples(train, reviewed_records)
    if not train:
        raise ValueError("Training split is empty after removing validation duplicates")
    _validate_active_labels(train, active_labels, "training corpus")
    _validate_active_labels(val, active_labels, "validation corpus")
    _validate_active_labels(reviewed_records, active_labels, "reviewed examples")
    hasher = FeatureHasher(n_features=FEATURES, input_type="dict", alternate_sign=False)
    known = np.array(sorted({"O"} | {f"{prefix}-{label}" for label in active_labels for prefix in ("B", "I")}))
    if initial_artifact is None:
        classifier = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=17, average=True)
        first = True
    else:
        _, initial_classes, initial_weights, initial_intercept, artifact_version = decode_artifact(initial_artifact)
        if artifact_version != 2:
            raise ValueError("Warm-start requires a v2 artifact because v1 uses legacy tokenization")
        if initial_weights.shape != (len(initial_classes), FEATURES):
            raise ValueError("Warm-start artifact has incompatible coefficient dimensions")
        if initial_intercept.shape != (len(initial_classes),):
            raise ValueError("Warm-start artifact has incompatible intercept dimensions")
        if (len(initial_classes) < 2 or len(set(initial_classes)) != len(initial_classes) or
                "O" not in initial_classes or any(
                    label != "O" and not re.fullmatch(r"[BI]-[A-Z][A-Z0-9_]{0,31}", label)
                    for label in initial_classes
                )):
            raise ValueError("Warm-start artifact has invalid class labels")
        if not np.isfinite(initial_weights.data).all() or not np.isfinite(initial_intercept).all():
            raise ValueError("Warm-start artifact contains non-finite coefficients")

        known = np.array(sorted(set(known.tolist()) | set(initial_classes)))
        # Artifacts contain only finalized coefficients, so resume with a non-averaging
        # SGD configuration and a fixed small step size. Bootstrap through public
        # partial_fit, then load saved values; no sklearn-private state is serialized.
        classifier = SGDClassifier(
            loss="log_loss", alpha=1e-5, random_state=17, average=False,
            learning_rate="constant", eta0=0.001, shuffle=False,
        )
        classifier.partial_fit(csr_matrix((1, FEATURES), dtype=np.float64), [known[0]], classes=known)
        resumed_weights = np.zeros((len(known), FEATURES), dtype=np.float64)
        resumed_intercept = np.zeros(len(known), dtype=np.float64)
        for row, label in enumerate(initial_classes):
            target_row = int(np.flatnonzero(known == label)[0])
            resumed_weights[target_row] = initial_weights.getrow(row).toarray()[0]
            resumed_intercept[target_row] = initial_intercept[row]
        classifier.coef_ = resumed_weights
        classifier.intercept_ = resumed_intercept
        first = False
    batch_x: list[dict[str, float]] = []
    batch_y: list[str] = []
    for _epoch in range(3):
        for record_index, record in enumerate(train):
            tokens = tokenize(record["text"])
            batch_x.extend(features(tokens, i) for i in range(len(tokens)))
            batch_y.extend(labels_for_tokens(record, tokens, active_labels, record_index))
            if len(batch_x) >= BATCH_SIZE:
                classifier.partial_fit(hasher.transform(batch_x), batch_y, classes=known if first else None)
                first = False
                batch_x, batch_y = [], []
    if batch_x:
        classifier.partial_fit(hasher.transform(batch_x), batch_y, classes=known if first else None)

    gold: list[str] = []
    pred: list[str] = []
    for record_index, record in enumerate(val):
        tokens = tokenize(record["text"])
        if not tokens:
            continue
        gold.extend(labels_for_tokens(record, tokens, active_labels, record_index))
        pred.extend(classifier.predict(hasher.transform([features(tokens, i) for i in range(len(tokens))])).tolist())
    score = float(f1_score(gold, pred, average="micro", labels=[label for label in known if label != "O"], zero_division=0))
    matrix = csr_matrix(classifier.coef_)
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        version=np.array([2], dtype=np.int16),
        features=np.array([FEATURES], dtype=np.int32),
        classes=classifier.classes_,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.array(matrix.shape, dtype=np.int32),
        intercept=classifier.intercept_,
    )
    artifact = buffer.getvalue()
    gold_entities: set[tuple[int, int, str, int]] = set()
    predicted_entities: set[tuple[int, int, str, int]] = set()
    for record_id, record in enumerate(val):
        gold_entities.update(
            (record_id, entity["start"], entity["end"], entity["label"])
            for entity in record["entities"]
            if entity["label"] in active_labels
        )
        predicted_entities.update(
            (record_id, span["start"], span["end"], span["label"])
            for span in predict(record["text"], artifact)
        )
    exact_metrics: dict[str, Any] = {}
    class_f1: list[float] = []
    for label in sorted(active_labels):
        gold_for_label = {entity for entity in gold_entities if entity[3] == label}
        predicted_for_label = {entity for entity in predicted_entities if entity[3] == label}
        true_positive = len(gold_for_label & predicted_for_label)
        precision = true_positive / len(predicted_for_label) if predicted_for_label else 0.0
        recall = true_positive / len(gold_for_label) if gold_for_label else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if gold_for_label:
            class_f1.append(f1)
        exact_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(gold_for_label),
        }
    exact_tp = len(gold_entities & predicted_entities)
    exact_precision = exact_tp / len(predicted_entities) if predicted_entities else 0.0
    exact_recall = exact_tp / len(gold_entities) if gold_entities else 0.0
    exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if exact_precision + exact_recall else 0.0
    return artifact, {
        "entity_micro_precision": exact_precision,
        "entity_micro_recall": exact_recall,
        "entity_micro_f1": exact_f1,
        "entity_macro_f1": sum(class_f1) / len(class_f1) if class_f1 else 0.0,
        "per_label": exact_metrics,
        "token_micro_f1": score,
        "alignment_errors": 0,
        "corpus_train_records": len(corpus_train),
        "train_records": len(train),
        "validation_records": len(val),
        "reviewed_examples": reviewed_count,
        "corpus_train_validation_duplicates_removed": corpus_validation_duplicates,
        "reviewed_validation_duplicates_removed": reviewed_validation_duplicates,
        "reviewed_test_duplicates_removed": reviewed_test_duplicates,
    }


@lru_cache(maxsize=2)
def decode_artifact(artifact: bytes) -> tuple[FeatureHasher, list[str], csr_matrix, np.ndarray, int]:
    with np.load(BytesIO(artifact), allow_pickle=False) as saved:
        artifact_version = int(saved["version"][0])
        if artifact_version not in (1, 2, 3):
            raise ValueError("Unsupported model artifact version")
        feature_count = int(saved["features"][0])
        classes = saved["classes"].astype(str).tolist()
        shape_value = tuple(int(v) for v in saved["shape"])
        weights = csr_matrix((saved["data"], saved["indices"], saved["indptr"]), shape=shape_value)
        intercept = saved["intercept"].copy()
    return FeatureHasher(n_features=feature_count, input_type="dict", alternate_sign=False), classes, weights, intercept, artifact_version


def predict(text: str, artifact: bytes) -> list[dict[str, Any]]:
    hasher, classes, weights, intercept, artifact_version = decode_artifact(artifact)
    tokens = _tokenize_legacy(text) if artifact_version == 1 else tokenize(text)
    if not tokens:
        return []
    feature_fn = role_features if artifact_version == 3 else features
    matrix = hasher.transform([feature_fn(tokens, i) for i in range(len(tokens))]).tocsr()
    decision = (matrix @ weights.T).toarray() + intercept
    shifted = decision - decision.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    candidates = [classes[int(index)] for index in np.argmax(decision, axis=1)]
    confidences = probabilities[np.arange(len(tokens)), np.argmax(decision, axis=1)]
    spans: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for (token, start, end), tag, confidence in zip(tokens, candidates, confidences):
        if tag.startswith(("B-", "I-")):
            label = tag[2:]
            if active and active["label"] == label and tag.startswith("I-"):
                active["end"] = end
                active["confidence"] = min(active["confidence"], float(confidence))
            else:
                if active:
                    spans.append(active)
                active = {"start": start, "end": end, "label": label, "confidence": float(confidence)}
        elif active:
            spans.append(active)
            active = None
    if active:
        spans.append(active)
    spans = [
        span for span in spans
        if span["label"] != "SNILS" or SNILS_RE.fullmatch(text[span["start"]:span["end"]])
    ]
    # A Russian +7 prefix is a strong phone signal and numeric-only token
    # classifiers often split it from the following digit run.
    for match in PHONE_RE.finditer(text):
        spans = [span for span in spans if span["end"] <= match.start() or span["start"] >= match.end()]
        spans.append({"start": match.start(), "end": match.end(), "label": "PHONE", "source": "rule"})
    spans.sort(key=lambda span: (span["start"], span["end"]))
    return spans


def normalize_legacy_prediction_labels(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map retired labels to their current equivalent before active-category filtering."""
    return [
        {**span, "label": "LOCALITY"} if span["label"] == "ADDR" else span
        for span in spans
    ]
