"""Read-only structural checks for the editorial NER 1.18 research batch.

This script never reads or writes test/challenge, CASES or release artefacts.
Run: python anonymizer_code/scripts/ner_1_18/validate_research_candidates.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "ner-1.18"
ACTIVE = frozenset({
    "APART", "COUNTRY", "DISEASE", "EMAIL", "HOUSE", "INN",
    "INSTITUTION", "KPP", "LOCALITY", "MEDICAL_RECORD_ID",
    "OGRN", "ORG", "PASSPORT", "PER", "PHONE", "REGION",
    "REFERRAL_ID", "SNILS", "STREET", "DRIVER_LICENSE_ID",
    "OMS_POLICY_ID", "OTHER_DOCUMENT_ID",
})
SPAN = re.compile(r"\[\[([A-Z_]+)\|([^\]\n]+)\]\]")
CATEGORIES = (
    (1, 16, "documents"),
    (17, 26, "linkage"),
    (27, 34, "addresses"),
    (35, 42, "diseases"),
    (43, 50, "neutral"),
)


def read_jsonl(name: str) -> list[dict]:
    path = DOCS / name
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name}:{line_no}: invalid JSON") from exc
    return records


def validate() -> None:
    rows = read_jsonl("research_candidates.jsonl")
    gaps = read_jsonl("coverage_gaps.jsonl")
    assert len(rows) == 50, f"Expected 50 records, got {len(rows)}"
    assert len(gaps) <= 6, "No more than six ontology gaps allowed"
    ids = [r["id"] for r in rows]
    assert ids == [f"cand-{i:03d}" for i in range(1, 51)], "Candidate IDs must be sequential"
    seen_texts: set[str] = set()
    labels_seen: Counter[str] = Counter()

    for item in rows:
        ctx = item["id"]
        for key in (
            "source_type", "family", "marked_text", "labels",
            "linked_features", "unmapped_fields", "identity_risk",
            "health_risk", "decision_by_group", "provenance",
        ):
            assert key in item, f"{ctx}: missing {key}"
        assert item["source_type"].strip() and item["family"].strip()
        assert item["identity_risk"] in {
            "none", "text_internally_linkable", "requires_external_information"
        }
        assert item["health_risk"] in {"none", "sensitive_health_present"}
        text = item["marked_text"]
        spans = list(SPAN.finditer(text))
        unmarked = SPAN.sub(lambda m: m.group(2), text)
        assert "[[" not in unmarked and "]]" not in unmarked, f"{ctx}: damaged markup"
        norm = " ".join(unmarked.casefold().split())
        assert norm not in seen_texts, f"{ctx}: repeated text"
        seen_texts.add(norm)
        label_list = list(dict.fromkeys(m.group(1) for m in spans))
        assert label_list == item["labels"], f"{ctx}: labels disagree with spans"
        assert set(label_list) <= ACTIVE, f"{ctx}: unknown label"
        labels_seen.update(m.group(1) for m in spans)
        for m in spans:
            value, label = m.group(2), m.group(1)
            assert value and value == value.strip(), f"{ctx}: whitespace in span"
            if label in {
                "MEDICAL_RECORD_ID", "OMS_POLICY_ID", "REFERRAL_ID", "SNILS",
                "DRIVER_LICENSE_ID", "OTHER_DOCUMENT_ID", "PASSPORT",
            }:
                assert not re.match(
                    r"^(?:№|номер\s|паспорт\s|полис\s|СНИЛС\s|удостоверение\s)",
                    value, flags=re.I
                ), f"{ctx}: document cue included in span"
                assert "X" in value or "ТЕСТ" in value or "TEST" in value, (
                    f"{ctx}: identifier needs an explicit test-only sentinel"
                )
            if label == "EMAIL":
                assert re.fullmatch(
                    r"[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.invalid", value
                ), f"{ctx}: email not reserved .invalid"
            if label == "PHONE":
                assert "X" in value, f"{ctx}: phone must be a masked test pattern"

        decisions = item["decision_by_group"]
        assert decisions, f"{ctx}: no masking decision"
        covered = set()
        for d in decisions:
            assert d["decision"] in {"mask", "generalize", "leave", "manual_review"}
            assert isinstance(d["reason"], str) and d["reason"].strip()
            assert set(d["fields"]) <= set(label_list), f"{ctx}: extraneous decision label"
            covered.update(d["fields"])
        assert covered == set(label_list), f"{ctx}: not all labels have decisions"
        if "DISEASE" in label_list:
            assert any(
                "DISEASE" in d["fields"] and d["decision"] == "manual_review"
                for d in decisions
            ), f"{ctx}: disease policy must remain for human review"
        assert "train" not in item and "validation" not in item
        p = item["provenance"]
        assert p["mode"] == "synthetic" and p["authoring"].startswith("local_original")
        for ref in p["references"]:
            assert ref["url"].startswith("https://")
            assert ref["publisher"] and ref["title"]
            assert ref["license"] and ref["accessed"] == "2026-10-27"
            assert ref["used_for"]

    family_counts = Counter(r["family"] for r in rows)
    for start, stop, group in CATEGORIES:
        print(f"{group}: {stop - start + 1}")
    assert sum(1 for n in family_counts.values() if n == 2) == 8, (
        "Expected four address and four disease contrast pairs"
    )
    assert max(family_counts.values()) == 2
    for card in gaps:
        assert card["status"] == "coverage_gap_only"
        assert card["training_allowed"] is False
        assert card["unmapped_fields"], "Gap cards need explicit out-of-ontology fields"
        assert card["decision"] == "manual_review"
        assert not (set(card["unmapped_fields"]) & ACTIVE)
    print(f"PASS: {len(rows)} candidates, {len(family_counts)} families, "
          f"{len(gaps)} coverage gaps, {sum(labels_seen.values())} labeled spans")
    print("Labels seen:", dict(sorted(labels_seen.items())))


if __name__ == "__main__":
    validate()
