# ADR: parent group for document identifiers

- Status: accepted for the 1.17 follow-up corpus and API design
- Date: 2026-10-21
- Scope: document-number classification, routing metadata, and UI grouping

## Decision

Keep one concrete leaf label on every annotated span. Add the parent group `DOCUMENT` as taxonomy metadata, derived from that leaf. The classifier continues to predict the concrete label; the application may route on `parent="DOCUMENT"` and retain the leaf label for type-specific handling. The parent is not a second overlapping entity and is not a classifier output.

`PASSPORT` keeps its current meaning. New child labels are proposed for the next immutable corpus release: `OMS_POLICY_ID`, `PATIENT_CARD_ID`, `MEDICAL_RECORD_ID`, `REFERRAL_ID`, `DRIVER_LICENSE_ID`, and `OTHER_DOCUMENT_ID`. The `OTHER_DOCUMENT_ID` label requires an explicit document-number cue. A bare digit sequence, lab measurement, date, room number, or sample counter remains negative unless the text identifies it as a document identifier.

`SNILS`, `PHONE`, tax identifiers, and address labels keep their current leaf semantics and do not become document-number children. They can receive other parent groups in a later decision if routing requires it.

## Contract and compatibility

- Dataset annotations and model BIO classes contain leaf labels only.
- `/api/categories` exposes `parent_class` on each child category. Analysis entities may add the same `parent_class` field while preserving the existing leaf `label`; masking remains `<LEAF_LABEL>`.
- Unknown/custom labels have no parent unless explicitly added to the application's taxonomy mapping.
- No parent row is stored in the category table and no `DOCUMENT` label is added to training examples.
- The 1.16.0 corpus/model and the immutable 1.17.0 candidate stay unchanged. New examples and any expanded taxonomy belong in a new corpus version.
- `authored_document_contrasts_v1.py` is a public draft scenario source. Its `test` and `challenge` groups are not sealed or eligible as an independent model gate; a future release needs separately held, independently authored and reviewed evaluation data.
- The candidate remains `HOLD` until independent evaluation confirms the preregistered leak, SNILS, and per-class safety gates. It must not activate automatically.

## Rationale

The existing classifier is a single flat BIO classifier. Predicting both a generic parent span and a child span would create overlapping annotations; replacing every subtype with a generic document label would discard useful distinctions. A full coarse-to-fine model cascade would change the training and inference architecture and introduce a new failure point. A derived parent preserves the current model path while giving downstream code one stable route for all document types.

## Data and evaluation rule

External datasets, vocabularies, and benchmark examples are research references only. New Russian sentences, identifiers, entities, and offsets must be authored locally; do not copy external strings or rows. Each document child needs positive examples plus same-shape negative numbers and contrasting document cues in train, validation, and fresh independent evaluation splits. Holdout thresholds are frozen before scoring; insufficient support or unfinished editorial review means `HOLD`.
