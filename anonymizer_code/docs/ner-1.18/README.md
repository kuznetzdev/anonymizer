# Anonymizer NER 1.18 — current patch 1.18.1

## Scope and decisions

This release continues the 1.17 experiment with a curated replacement for its
open draft additions. New Russian examples are authored locally. External
sentences, corpus rows, and patient records are not imported. No model is
activated by building or evaluating this release.

- Keep a flat classifier with 22 concrete labels. `DOCUMENT` is derived routing
  metadata, never an overlapping annotation or an additional BIO class.
- Add `OMS_POLICY_ID`, `MEDICAL_RECORD_ID`, `REFERRAL_ID`,
  `DRIVER_LICENSE_ID`, and `OTHER_DOCUMENT_ID` to the import ontology.
- Patient cards, ambulatory charts, medical charts, and case histories all use
  `MEDICAL_RECORD_ID`. The earlier proposed `PATIENT_CARD_ID` remains readable
  through existing routing metadata, but is absent from new training examples.
- Build a standalone snapshot from the immutable common and medical 1.16
  train/validation sources plus the 74 reviewed 1.18 cases. Migrate explicit
  document numbers to the expanded ontology before combining the sources.
  Train on the released split once; do not mix the old common corpus into it
  again. The 1.17 draft additions are excluded; its candidate and results remain
  unchanged.
- Correct unequivocal generic STREET prefixes and the reviewed missing quoted
  disease mentions in the new copy. Keep source releases unchanged and record
  every correction in the build report.
- Retain one reviewed 11-digit contract-number contrast with its correct
  `OTHER_DOCUMENT_ID` label, and remove eight near-clones of that old group.
  Keep known medcard and contract numbers out of the negative class.
- Keep the inherited test bytes unchanged. They do not establish independent
  evaluation readiness for the expanded ontology. Do not manufacture extra
  test examples to meet a support threshold. Do not score sealed challenge data.

## Quality contract

The new cases cover repeated entities, person/street/disease role contrasts,
same-shape and identical numeric values under different semantic cues, mixed
documents, multiline layouts, and clearly negative measurements and technical
numbers. Invalid checksums do not turn explicitly presented personal identifiers
into negative examples. See [annotation policy](ANNOTATION_POLICY.md).

Source cases and executable release checks live in
`scripts/ner_1_18/` and `backend/tests/test_ner_1_18_*.py`.
Corpus payloads and model artifacts remain under ignored `references/`.

## Examples of the boundaries

These are extracts of the authored cases; the generated `authored_review.md`
contains complete texts and exact offsets.

| Text and role | Annotation |
| --- | --- |
| Врач Илья Боткин | `PER`: Илья Боткин |
| ул. Боткина | `STREET`: Боткина |
| болезнь Боткина не выявлена | `DISEASE`: болезнь Боткина |
| СНИЛС 791-612-345 11 | `SNILS`: 791-612-345 11 |
| телефон +7 916 123-45-11 | `PHONE`: +7 916 123-45-11 |
| технический счётчик обработки: 79161234511 единиц за смену | No entity |
| в паспорте лабораторного анализатора номер документа СВ-71-003604 | `OTHER_DOCUMENT_ID`: СВ-71-003604 |
| серийный номер оборудования | No document annotation without a document identifier cue |

The disease remains annotated under negation. Its preservation in masked output
is a separate option. `DOCUMENT` groups document leaves; it does not replace the
concrete labels in these examples.

## Evaluation boundary

Freeze source, data, protocol, environment, and baseline artifact hashes before
one development training run. Compare the candidate on the same permitted
validation data; report the new label diagnostics separately from the legacy
17-label comparison. Include masking with disease preservation so predicting a
disease cannot conceal a missed personal identifier.

Passing structural checks or editorial review does not establish model quality.
The candidate stays `HOLD` until development and independent evaluation gates
are satisfied. See [evaluation protocol](EVALUATION_PROTOCOL.md).
The completed run, measured gate failures, corpus counts, and artifact hashes are
recorded in [RESULTS.md](RESULTS.md).

The original 1.18.0 release and input snapshots are retained. Its training
process was interrupted and produced no candidate artifact or completed
metrics. The 1.18.1 correction follows a source-annotation audit, before any
1.18 model scores were available. It is not a hyperparameter search.

## Terminology references

References check terminology only; their sentences are not training material:
[WHO: Parkinson disease](https://www.who.int/ru/news-room/fact-sheets/detail/parkinson-disease),
[Orphanet: adult Refsum disease](https://www.orpha.net/en/disease/detail/773),
[NIH GARD: Landau-Kleffner syndrome](https://rarediseases.info.nih.gov/diseases/6855/landau-kleffner-syndrome/).
The corpus is synthetic NER material; clinical statements are not treatment guidance.
