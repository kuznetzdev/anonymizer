# NER 1.18 annotation policy

This policy defines the 22 active leaf labels in `data/corpora/ru-pii/v1/ontology.json`. `DOCUMENT` is routing metadata derived from a leaf label; it is never an annotated span or model output. `ADDR` remains available only for reading legacy data and is deprecated.

## Span and class rules

- Offsets are zero-based Python Unicode code-point offsets with half-open `[start, end)` bounds. Annotate the complete entity surface and exclude punctuation and generic cues.
- For document identifiers, annotate only the identifier value. Exclude cues such as `паспорт`, `полис ОМС`, `карта`, `история болезни`, `направление`, `водительское удостоверение`, `номер` and `№`.
- Use `MEDICAL_RECORD_ID` for a patient's card, ambulatory or medical chart, and case history. These are one annotation class even when a record uses different local names or abbreviations. `PATIENT_CARD_ID` is retained only for compatibility with existing category rows and API responses; do not use it in new annotations or remap old values automatically.
- Use `REFERRAL_ID` only for an issued patient referral or its referral slip. A queue ticket, appointment number, sample number, equipment code, or other nearby number is not a referral by itself.
- Use `OTHER_DOCUMENT_ID` when the text explicitly identifies a numbered document or credential that has no dedicated leaf label. This includes an equipment passport or service document; use `OTHER_DOCUMENT_ID`, not `PASSPORT`, for equipment records. A bare number is insufficient.
- Annotate every disease mention as `DISEASE`, including negated, suspected, quoted, historical, or family-history mentions. Annotate a person's name as `PER` when the text identifies an actual person, including quoted or historical mentions.
- A checksum does not decide whether an explicitly identified sensitive identifier is annotated. An invalid-checksum value presented as a person's identifier remains annotated. Dates, measurements, doses, room and cabinet numbers, sample identifiers, and other bare numeric values remain negative unless the text explicitly identifies them as a supported entity.
- Keep the existing address-component policy: annotate explicit components separately; `STREET` excludes generic cues such as `ул.`, `проспект`, and `д.` unless the cue is part of the official name.

## Compatibility and review

- New corpus annotations use concrete leaf labels only. Do not add `DOCUMENT` as a span or trainable class.
- `PASSPORT` keeps its existing meaning. The five additional leaf labels are `DRIVER_LICENSE_ID`, `MEDICAL_RECORD_ID`, `OMS_POLICY_ID`, `OTHER_DOCUMENT_ID`, and `REFERRAL_ID`.
- Keep `PATIENT_CARD_ID` in the routing taxonomy for compatibility, but use `MEDICAL_RECORD_ID` for new annotations covering patient, ambulatory, or medical charts and histories.
- Resolve uncertain class boundaries before adding records. Review class contrasts and their negative numeric counterparts; do not multiply near-duplicate examples to meet counts. Insufficient independent support or unfinished review means `HOLD` for the affected evaluation gate.
