# NER 1.18 corpus and development evaluation protocol

## Corpus build

The source cases are `scripts/ner_1_18/mixed_cases.py` and `numeric_cases.py`. Each row has a stable case ID, `train` or `validation` split, family, and `[[LABEL|surface]]` annotations. `build_release.py` checks markers and Unicode offsets, ontology labels, duplicate text and masked-shape clones, family split integrity, and the explicitly declared invalid-SNILS examples.

The build starts from immutable 1.16.0 `train.jsonl` and `validation.jsonl`, whose registered SHA-256 values are `43a7f35e0d479ab0edf4abfa23acdd4832b96ad557c8a03a5882b69a3bae6d28` and `dca4098b494da6b7e62d70c93126ddbd71e8df324b92568fc60c03ff3ad0949f`. The opaque inherited test file is pinned to SHA-256 `e89f2a5f07b6cc5405836a1cf8c28849aad8748da5c2e29756308e27c326191f`. The build writes a new ignored `references/corpora/anonymizer-ner-hardcases/1.18.0/` directory and refuses to overwrite it. The 1.16.0 test bytes are copied unchanged into the four-file import ZIP; the importer may parse them for structural validation, but the build report does not inspect, display, score, or claim support for that holdout. Only the inherited test hash is recorded.

The source rows are retained as the authored-only `authored_train.jsonl` and `authored_validation.jsonl`, with a generated `authored_review.md` and `authored_cases.json` audit index that expose each ID, split, family, source, marked text, and parsed span. The full release includes base plus authored train and validation rows. It records per-label distinct-document support for development splits, cases longer than 1,000 characters, negative-record counts, family grouping, and correction details. No new label is assigned to `PATIENT_CARD_ID`.

Two narrowly guarded corrections apply to the new copy: unequivocal street-type prefixes are removed from `STREET` offsets in the old development rows while source text remains unchanged; and the exact audited 1.16.0 row about quoted hepatitis C and Wilson disease receives the two missing quoted spans after exact text and original-offset checks. No corpus source file is edited.

The 1.18 release replaces the flawed 1.17 additive draft with freshly curated cases on the 1.16.0 development base. This prevents the 1.17 draft from being accumulated as inherited training data.

## One-shot development run

After editorial review confirms both author files are frozen, build the release and then run:

```powershell
python scripts/ner_1_18/build_release.py
python scripts/ner_1_18/train_candidate.py --source-reviewed
```

The explicit switch is an operator acknowledgement of that review. The runner refuses any existing `references/evaluations/medical-1.18/` output. Before training starts it hashes every dataset, baseline artifact, ontology, source module, protocol, and candidate ZIP; writes those hashes and snapshots the exact training runner, builder, model, scorer, ontology, and protocol. A post-run hash check fails if any frozen input changed.

The candidate reuses `app.model.train_model`, feature version 3, and its registered fixed training parameters. It trains on common train plus the complete corrected 1.18 train split, so both the guarded corrections and authored examples reach the model exactly once. Training metrics use common validation plus the complete corrected 1.18 validation split. It scores frozen 1.16 and 1.17 artifacts and the new candidate in the same runtime on common validation and 1.16.0 medical validation, both with the original 17 labels and all 22 labels. The all-label score keeps predictions from new categories visible as false positives on old-label data; derived `DOCUMENT` strict-span scores collapse its document leaf categories for comparison. Authored validation is reported separately for all 22 labels with strict per-label span counts and metrics, per-label support, numeric negative-case false positives, and `preserve_diseases=True` masking-aware PII leakage; disease spans remain in exact-span metrics but do not count as masked PII.

The development gate compares the candidate to 1.16.0 on the original 17 labels, using the existing 1.17 thresholds: common exact-span micro-F1 may fall by no more than 0.01, document leak rate may rise by no more than 0.01, and each priority-label coverage recall may fall by no more than 0.02. On 1.16.0 medical validation, leak rate must improve by at least 0.01 and SNILS recall must improve by at least 0.02, reach at least 45/49, and retain support 49. No threshold tuning or repeat run is allowed after seeing results.

Development passage cannot release the candidate. The result stays `HOLD`: no fresh independent test or challenge set exists for the expanded 22-label ontology. No test/challenge file is an input to the runner, and the candidate artifact remains a local development NPZ without activation or Docker changes.
