# Anonymizer NER 1.17.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release and evaluate an immutable NER corpus 1.17.0 with a candidate that reduces sensitive document leaks without regressing `SNILS` or other target classes.

**Architecture:** Reuse the existing hardcases corpus release builder, `FeatureHasher`/`SGDClassifier` model, and current evaluation contracts. Keep source corpora/evaluation records in the private ignored `references/` tree; tracked code, specification and plan live in Git. Candidate integration remains conditional on frozen evaluation.

**Tech Stack:** Python, JSONL/ZIP, NumPy/SciPy, scikit-learn, FastAPI/Pydantic, PostgreSQL driver already declared by backend requirements, Git.

**Spec:** `docs/superpowers/specs/2026-10-20-anonymizer-ner-1-17-design.md`

## Global Constraints

- Preserve `references/corpora/anonymizer-ner-hardcases/1.16.0/` byte-for-byte.
- Preserve exactly 17 labels and Unicode half-open `[start,end)` offsets.
- Never use sealed test/challenge outcomes to tune a candidate; evaluate each frozen 1.17.0 holdout once.
- Do not activate a candidate automatically; a failed or unmeasurable safety gate is `HOLD`.
- Do not publish raw corpora or private evaluation records from ignored `references/`.
- Do not add runtime dependencies or a parallel training system.
- Keep existing dirty checkout unchanged; use only explicitly selected NER files in this worktree.

## Review Focus

- A repeated patient or near-duplicate document crosses splits — builder tests assert source-group and masked-template isolation.
- OCR mutation shifts entity coordinates — tests revalidate `text[start:end]` and entity labels after each transformation.
- A long document's final or repeated ID is lost — API/model tests assert exact predicted offsets near the 20,000-character limit.
- Candidate v3 artifact is scored with incompatible code — loader test checks version, dimensions and recorded hash before evaluation.
- Small `SNILS`/`PASSPORT` support hides a regression — gate report includes support and returns HOLD when key-class support is below 30 spans.

---

### Task 1: PR 1 — Freeze baseline and repair experiment reproducibility

**Files:**
- Create: `anonymizer_code/scripts/ner_1_17/run_dev.py`
- Create: `anonymizer_code/docs/ner-1.17/BASELINE.md`
- Create: `anonymizer_code/docs/ner-1.17/EVALUATION_PROTOCOL.md`
- Create: `anonymizer_code/docs/ner-1.17/PR1_REPORT.md`
- Modify: `anonymizer_code/backend/app/model.py` (selectively port only the NER model changes required to decode frozen artifact v3)
- Modify: `anonymizer_code/backend/tests/test_model_tokenization.py` (artifact compatibility and long-input offsets)

**Interfaces:**
- Consumes: the frozen `candidate_1_16.npz` and validation-only inputs from 1.16.0 plus the existing main training corpus; do not copy or read 1.16 test/challenge records.
- Produces: baseline SHA/version, resolved dependency/test command, and development evaluator output with no test/challenge access.

- [ ] **Step 1: Add failing compatibility and long-input tests.** Add `test_v3_artifact_uses_combined_features_and_preserves_unicode_offsets` for v3 decode/predict and Unicode exact offsets, plus `test_long_input_keeps_entity_offsets_at_the_end` for a sensitive span at the end of a 19,999–20,000-character input.
- [ ] **Step 2: Run the focused tests.** Run from `anonymizer_code/backend`: `python -m unittest tests.test_model_tokenization`; confirm the baseline artifact-v3 failure before porting the minimum model changes.
- [ ] **Step 3: Selectively port NER model code.** Apply only `model.py` changes required to load artifact v3 and preserve the v3 feature contract; do not port seed installation, warm-start worker/store changes, UI or unrelated docs from the dirty checkout.
- [ ] **Step 4: Isolate the declared test environment.** Create a temporary venv outside the worktree and install `anonymizer_code/backend/requirements.txt`; verify `import psycopg` and run `python -m unittest tests.test_model_tokenization tests.test_training_retries`. Document exact command and versions. Use the repository's `unittest` tests without adding an undeclared `pytest` dependency.
- [ ] **Step 5: Record baseline hashes and development gates.** Record candidate 1.16 artifact SHA, model code SHA, main-corpus and validation input hashes, metric definitions and the baseline validation metrics required by the preregistered gates. Recompute only allowed validation metrics; do not inspect test/challenge predictions or examples.
- [ ] **Step 6: Review and commit PR 1.** Write `PR1_REPORT.md` with changes, executed checks, remaining limitations and PASS/HOLD criteria. Commit only task-owned source, tests and reports that are not private held-out content.

### Task 2: PR 2 — Build and validate immutable corpus 1.17.0

**Files:**
- Create: `anonymizer_code/scripts/build_ner_hardcases_1_17.py`
- Create: `anonymizer_code/backend/tests/test_ner_hardcases_1_17.py`
- Create: `anonymizer_code/docs/ner-1.17/PR2_REPORT.md`
- Create locally in ignored `references/corpora/anonymizer-ner-hardcases/1.17.0/`: `manifest.json`, `ANNOTATION_POLICY.md`, `DATASET_CARD.md`, `SOURCE_REGISTER.md`, `CHANGELOG.md`, `quality_report.json`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, `anonymizer-ner-hardcases-1.17.0_import.zip`
- Create locally in ignored `references/evaluations/medical-1.17/challenge_sealed_v1/`: `records.jsonl`, `CHALLENGE_CARD.md`, `editorial_review.json`
- Modify: `references/corpora/anonymizer-ner-hardcases/README.md`

**Interfaces:**
- Consumes: approved 1.16 train/validation only and the existing four-file importer contract.
- Produces: validated 1.17 release, new sealed challenge, quality report, and a ZIP that `_load_archive(..., DEFAULT_LIMITS)` accepts without publication.

- [ ] **Step 1: Add failing builder checks.** Add tests `test_rejects_misaligned_transformed_span`, `test_rejects_invalid_identifier_checksum`, `test_negative_record_contains_no_supported_label` and `test_source_groups_and_templates_are_split_disjoint`; cover all 17 labels, UTF-8 JSONL, Unicode offsets, non-overlap and record limits as well.
- [ ] **Step 2: Run `python -m unittest anonymizer_code.backend.tests.test_ner_hardcases_1_17` against intentionally invalid fixtures.** Confirm rejection for a shifted OCR span, an invalid checksum, a negative example containing `DISEASE`, and a group/template crossing splits. Use the repository's `unittest` stack; do not add an undeclared `pytest` dependency.
- [ ] **Step 3: Research primary sources and license constraints.** Record official source, purpose, commercial/ML rights, redistribution restrictions, provenance and real-PII risk for Synthea, NHS synthetic notes, ASQ-PHI, Microsoft clinical visit notes, Human Disease Ontology, Wikidata, Mondo and Faker. Inspect each primary repository's license and relevant dataset terms. Import none unless both rights and a concrete coverage benefit are verified.
- [ ] **Step 4: Author quality-first additions.** Build varied long multi-line medical records, hard negative contexts, eponym/PER contrasts, address chains, organization/institution contrasts, and passport/SNILS/other identifier positives and near-miss negatives.
- [ ] **Step 5: Build release deterministically.** Generate from 1.16 train/validation without reading 1.16 test/challenge; add 1.17 test and challenge from independent authored source groups. Do not optimize for a record-count quota; add only reviewed examples with a clear coverage contribution. Ensure the independent challenge and test each have at least 30 spans for every preregistered priority target label, or mark the affected safety gate `HOLD`. Record why each held-out group is independent of training templates and source groups.
- [ ] **Step 6: Validate and finalize the release.** Run `python -m unittest anonymizer_code.backend.tests.test_ner_hardcases_1_17`, then `python anonymizer_code/scripts/build_ner_hardcases_1_17.py` to write the quality report and `python anonymizer_code/scripts/build_ner_hardcases_1_17.py --finalize` to write manifest/ZIP. Assert all output hashes and call the app archive parser in memory without storage writes or model activation.
- [ ] **Step 7: Complete blind editorial review.** Independently review all held-out annotations and at least 50 train records across all scenario families; only label/structure corrections are allowed, never model predictions.
- [ ] **Step 8: Review and commit PR 2.** Report counts, class support, length distribution, duplicates, source/license decisions, reviewer scope, hashes and importer result. Private source text remains in ignored `references/`.

### Task 3: PR 3 — Reproducible candidate training and one-shot independent evaluation

**Files:**
- Create: `anonymizer_code/scripts/ner_1_17/run_experiment.py`
- Create: `anonymizer_code/scripts/ner_1_17/run_frozen.py`
- Create: `anonymizer_code/backend/tests/test_ner_1_17_evaluation.py`
- Create: `anonymizer_code/docs/ner-1.17/FINAL_RESULTS.md`
- Create: `anonymizer_code/docs/ner-1.17/PR3_REPORT.md`
- Create locally in ignored `references/evaluations/medical-1.17/`: `freeze_manifest.json`, frozen model artifacts, per-record predictions and raw metrics.

**Interfaces:**
- Consumes: PR 1 baseline and PR 2 frozen 1.17 release/challenge/test; baseline and candidate use the same model code, trainer, seed and hyperparameters. Train candidate through the existing `train_model` with an explicit v3 `combined_features` option; preserve the current v2 default for existing callers and runtime behavior.
- Produces: reproducible candidate artifact, dev and independent metrics, and a machine-checkable PASS/HOLD decision.

- [ ] **Step 1: Test gate and freeze behavior.** Add `test_frozen_metrics_enforce_preregistered_gates` and `test_frozen_runner_verifies_hashes_and_refuses_second_run`; run `python -m unittest anonymizer_code.backend.tests.test_ner_1_17_evaluation`. Cover spec thresholds, minimum support, split-overlap rejection and the one-shot sentinel.
- [ ] **Step 2: Train candidate from a fixed snapshot.** Use only the main corpus + combined 1.17 train split and permitted validation; record code/dependency versions, seeds, hashes, run time and resource usage. Do not tune on either closed 1.16 or 1.17 test/challenge.
- [ ] **Step 3: Apply development gates.** Select at most one candidate using allowed validation; if it fails gates or `SNILS` recovery target, record HOLD and do not open any independent labels.
- [ ] **Step 4: Freeze before evaluation.** Hash both models, evaluator, train/validation inputs, 1.17 challenge/test, protocol and release ZIP; validate split isolation and set the one-shot sentinel atomically only after all hashes and gates are ready. The frozen runner accepts a single run identifier and refuses to overwrite a completed result.
- [ ] **Step 5: Run the frozen evaluator once.** Score baseline and candidate on the new 1.17 challenge and test in one run; produce aggregate metrics only. Do not rerun or tune after seeing held-out results.
- [ ] **Step 6: Review and commit PR 3.** Include paired metrics, support, uncertainty/limitations, false-positive/overmask costs, and explicit PASS/HOLD decision; never claim real-world safety from synthetic data.

### Task 4: PR 4 — Conditional product inference correction

**Files:**
- Modify only the confirmed affected inference/API module under `anonymizer_code/backend/app/`.
- Test: `anonymizer_code/backend/tests/test_model_tokenization.py` and the relevant API test.
- Create: `anonymizer_code/docs/ner-1.17/PR4_REPORT.md` only if a product change is required.

**Interfaces:**
- Consumes: a reproducible long-document, offset, chunk-boundary or final-mask failure from PR 1/3.
- Produces: minimal backward-compatible fix, regression coverage, runtime change report and rollback note.

- [ ] **Step 1: Confirm a product defect.** If all documents within the 20,000-character API cap preserve predicted spans and final masks, mark PR 4 `NOT REQUIRED` with evidence and make no product change.
- [ ] **Step 2: Add the failing regression test.** Pin the specific input length, Unicode offsets, boundary position and expected mask before changing code.
- [ ] **Step 3: Fix only the demonstrated failure.** Preserve all 17 labels, output schema, privacy policy and prior model artifact compatibility.
- [ ] **Step 4: Run focused API/model tests and full backend suite in the isolated environment.** Report actual outcomes, including any external PostgreSQL/runtime checks not performed.
- [ ] **Step 5: Review and commit PR 4 only if needed.** Candidate activation remains an explicit, separate operation; keep 1.16 artifact available for rollback.
