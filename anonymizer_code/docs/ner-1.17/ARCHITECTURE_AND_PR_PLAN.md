# NER 1.17.0 architecture and PR plan

## Scope and boundaries

The release reuses the existing CPU token-classification system in `backend/app/model.py`, the 17-label ontology, and the corpus importer in `backend/app/training/corpus.py`. Training, corpus construction, scoring, and product inference stay separate responsibilities. Version 1.16.0 artifacts and source splits are immutable inputs; experimental corpora and model artifacts remain under ignored `references/` paths. No model is activated by the experiment.

## Data and model flow

1. Authored or licensed records are validated for source-group split isolation, annotation offsets, checksums, and importer limits.
2. The corpus builder creates train/validation/test; challenge data is kept outside the import ZIP.
3. The existing `train_model` entry point trains a v3 artifact with `combined_features`; its default remains v2 and warm-start remains v2-only.
4. A validation-only runner scores the frozen baseline and candidate using one fixed 17-label masking policy and the preregistered gates.
5. Independent test/challenge scoring is permitted only after development passes, editorial review is complete, and both held-out splits meet minimum target-label support. A freeze manifest and one-shot sentinel are required before that evaluation.
6. Product integration is a separate conditional change. It is omitted unless an independently reproduced inference/API defect requires it and a candidate passes the quality gates.

The candidate remains `HOLD` on any failed or under-supported gate. Test/challenge data never feeds training or tuning. Ontology additions require a separate architecture decision.

## PR sequence and acceptance

| PR | Scope | Acceptance |
|---|---|---|
| 1 | Baseline, label policy, compatibility and reproducibility | Frozen inputs/hash checks, v3 inference compatibility, backend regression suite, baseline report; candidate still gated |
| 2 | Authored hard cases, annotation policy, deterministic corpus integration | Span/source/checksum/split checks, blind review, archive/challenge separation; sparse splits remain explicitly `HOLD` |
| 3 | One reproducible v3 training run and validation comparison | Pinned source hashes, baseline and candidate aggregates, all registered development gates; failure stops independent evaluation |
| 4 | Conditional product inference/integration fix | Only if a reproducible product-path defect exists; tests and rollback evidence; otherwise no PR is created |

## Current disposition

- PR1: accepted for baseline compatibility and reproducible development scoring.
- PR2: structural integration passed; independent split support and editorial readiness remain `HOLD`.
- PR3: candidate 1.17.0 failed development gates and remains `HOLD`; no independent test/challenge score was produced.
- PR4: not created. The boundary-offset regression passes at the maximum supported text length, and the failed candidate is not eligible for product integration.

The code, corpus report, and evaluation protocol are the sources of truth for exact hashes and metrics.
