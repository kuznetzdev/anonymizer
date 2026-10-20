# Anonymizer NER 1.18.0

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
- Start with immutable 1.16 train/validation. Replace the 1.17 draft additions
  with reviewed 1.18 examples instead of accumulating their defects. The 1.17
  candidate and its original results remain unchanged.
- Correct unequivocal generic STREET prefixes and the reviewed missing quoted
  disease mentions in the new copy. Keep source releases unchanged and record
  every correction in the build report.
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

## Evaluation boundary

Freeze source, data, protocol, environment, and baseline artifact hashes before
one development training run. Compare the candidate on the same permitted
validation data; report the new label diagnostics separately from the legacy
17-label comparison. Include masking with disease preservation so predicting a
disease cannot conceal a missed personal identifier.

Passing structural checks or editorial review does not establish model quality.
The candidate stays `HOLD` until development and independent evaluation gates
are satisfied. See [evaluation protocol](EVALUATION_PROTOCOL.md).

## Terminology references

References check terminology only; their sentences are not training material:
[WHO: Parkinson disease](https://www.who.int/ru/news-room/fact-sheets/detail/parkinson-disease),
[Orphanet: adult Refsum disease](https://www.orpha.net/en/disease/detail/773),
[NIH GARD: Landau-Kleffner syndrome](https://rarediseases.info.nih.gov/diseases/6855/landau-kleffner-syndrome/).
The corpus is synthetic NER material; clinical statements are not treatment guidance.
