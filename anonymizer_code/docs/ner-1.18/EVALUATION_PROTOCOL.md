# NER 1.18 corpus and development evaluation protocol

## Corrective release 1.18.1

The source correction continues as immutable release `1.18.1`; the `1.18.0` payload and source revision remain unchanged. The earlier 1.18.0 evaluation was interrupted before any candidate artifact or metrics were available. This correction was made before inspecting candidate metrics and retains the registered training parameters and development gates.

The 1.18.1 builder applies four guarded annotations to medical rows: two contract values become `OTHER_DOCUMENT_ID`, and two chart values become `MEDICAL_RECORD_ID`. It labels one explicit 11-digit contract number in train as `OTHER_DOCUMENT_ID` while preserving the negative contrast with SNILS. It also adds 179 audited common-corpus spans: 40 chart IDs and 131 other document IDs in train, plus eight other document IDs in validation. Existing spans, including covered passport IDs, stay unchanged; contract cues require an explicit number marker so a sum after a colon is not relabeled. Each correction records its source row and content hash. Eight exact empty-entity near-clones with the repeated fixed contract-number sentence are removed; the retained contract row and the other contract examples remain. The existing 281 `STREET` prefix and two quoted-disease corrections stay in place, as do all 74 authored cases.

The versioned output is `references/corpora/anonymizer-ner-hardcases/1.18.1/`, and the one-shot evaluation output is `references/evaluations/medical-1.18.1/`. The release ZIP, candidate, and report names include `1.18.1`. Expected source totals before exact-text deduplication are 13,731 train, 2,218 validation, and 160 test records. Current source composition has no exact duplicates after correction; the builder deduplicates identical text only when annotations match and fails on conflicts or cross-split reuse. The 160 test records remain copied byte-for-byte from 1.16.0; they are not scored. Model status remains `HOLD` until an independent evaluation covers all 22 labels.

## Corpus build

The source cases are `scripts/ner_1_18/mixed_cases.py` and `numeric_cases.py`. Each row has a stable case ID, `train` or `validation` split, family, and `[[LABEL|surface]]` annotations. `build_release.py` checks markers and Unicode offsets, ontology labels, duplicate text and masked-shape clones, family split integrity, and the explicitly declared invalid-SNILS examples.

The build starts from immutable common `train.jsonl` and `val.jsonl` with SHA-256 values `921716a5524295c1f8d59bd23904a4104e1905620e8d1888b88b7e1a3767d6bd` and `7f0a3eb27a64fe5a217c296f59a2930291fdf6faa5a71e5c85aaf334529df59b`, plus 1.16.0 medical `train.jsonl` and `validation.jsonl` with SHA-256 values `43a7f35e0d479ab0edf4abfa23acdd4832b96ad557c8a03a5882b69a3bae6d28` and `dca4098b494da6b7e62d70c93126ddbd71e8df324b92568fc60c03ff3ad0949f`. The opaque inherited test file is pinned to SHA-256 `e89f2a5f07b6cc5405836a1cf8c28849aad8748da5c2e29756308e27c326191f`. The build writes a new ignored `references/corpora/anonymizer-ner-hardcases/1.18.1/` directory and refuses to overwrite it. The 1.16.0 test bytes are copied unchanged into the four-file import ZIP; the importer may parse them for structural validation, but the build report does not inspect, display, score, or claim support for that holdout. Only the inherited test hash is recorded.

The 74 authored rows remain available as `authored_train.jsonl` and `authored_validation.jsonl`, with `authored_review.md` and `authored_cases.json`. The full release is a standalone snapshot combining corrected common train/validation, corrected 1.16.0 medical train/validation, and the authored rows. It records per-label distinct-document support for development splits, cases longer than 1,000 characters, negative-record counts, family grouping, and correction details. No new label is assigned to `PATIENT_CARD_ID`.

The new copy preserves two earlier narrow corrections: unequivocal street-type prefixes are removed from `STREET` offsets in old development rows while source text remains unchanged; and the exact audited 1.16.0 row about quoted hepatitis C and Wilson disease receives the two missing quoted spans after exact text and original-offset checks. The 1.18.1 guarded document corrections are listed above. No corpus source file is edited.

The 1.18.1 release combines the corrected common and medical sources with the 74 curated cases. It does not accumulate the flawed 1.17 draft.

## One-shot development run

After editorial review confirms both author files are frozen, build the release and then run:

```powershell
python scripts/ner_1_18/build_release.py
python scripts/ner_1_18/train_candidate.py --source-reviewed
```

The explicit switch is an operator acknowledgement of that review. The runner refuses any existing `references/evaluations/medical-1.18.1/` output. Before training starts it hashes every dataset, baseline artifact, ontology, source module, protocol, and candidate ZIP; writes those hashes and snapshots the exact training runner, builder, model, scorer, ontology, and protocol. A post-run hash check fails if any frozen input changed.

The candidate reuses `app.model.train_model`, feature version 3, and its registered fixed training parameters. It trains only on the standalone 1.18.1 train split and uses only the corrected 1.18.1 validation split for training-time metrics. It scores frozen 1.16 and 1.17 artifacts and the new candidate in the same runtime on the original common and 1.16.0 medical validation splits as legacy benchmarks. The 17-label metrics are comparable; the 22-label metrics on these old annotations are overprediction diagnostics only because the gold labels omit the new document classes. On the corrected full 1.18.1 validation snapshot, the report includes per-label exact-span counts, support, metrics, and collapsed parent `DOCUMENT` scores. Authored validation is reported separately for all 22 labels with strict per-label span counts and metrics, per-label support, numeric negative-case false positives, and `preserve_diseases=True` masking-aware PII leakage; disease spans remain in exact-span metrics but do not count as masked PII.

The development gate compares the candidate to 1.16.0 on the original 17 labels, using the existing 1.17 thresholds: common exact-span micro-F1 may fall by no more than 0.01, document leak rate may rise by no more than 0.01, and each priority-label coverage recall may fall by no more than 0.02. On 1.16.0 medical validation, leak rate must improve by at least 0.01 and SNILS recall must improve by at least 0.02, reach at least 45/49, and retain support 49. No threshold tuning or repeat run is allowed after seeing results.

Development passage cannot release the candidate. The result stays `HOLD`: no fresh independent test or challenge set exists for the expanded 22-label ontology. No test/challenge file is an input to the runner, and the candidate artifact remains a local development NPZ without activation or Docker changes.
