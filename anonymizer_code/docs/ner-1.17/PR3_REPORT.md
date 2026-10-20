# PR 3 report — one-shot candidate training and development evaluation

## Result

Candidate `1.17.0` is **HOLD**. It did not pass preregistered development gates. Independent test/challenge evaluation was not opened or run.

## Changes

- Added explicit `feature_version=3` training to the existing `train_model` function. v3 uses the existing `combined_features` inference contract for train and internal validation. The default remains v2; v2-only warm-start behavior is preserved.
- Added training regression tests for v2 default, v3 artifact output, v3 prediction, and rejection of v3 warm-start.
- Added a one-shot candidate runner that pins baseline, 1.17 train/validation, archive, corpus report, evaluation protocol, and baseline report hashes. It checks baseline report provenance and requires review readiness and at least 30 examples per priority label in both test and challenge before independent evaluation can proceed.
- No ontology, product inference behavior, or 1.16.0 artifact was changed.

## Run and development results

- One candidate was trained with Python 3.12.7, NumPy 2.2.5, SciPy 1.18.1, and scikit-learn 1.8.0; seed 17, 3 epochs, `log_loss`, `alpha=1e-5`, averaged SGD, batch size 4096, 16,384 features, cold start. The run used 13,712 train records before deduplication and took about 100 seconds.
- Candidate artifact SHA-256: `63518bccabcf6539091b1240e1f904714d0e6ca53df3dcdf7a5078accc4c3567`.
- The one-shot run recorded training-runner SHA-256 `0751f30b7d1a3210405a9066b47534868a235bd7818603b2ad64ffe69c99163e` and model-source SHA-256 `4ac725f6009381928b8ad1a0c26250a3243c334ec1cc332976bb7414ff2b178e`; exact data, archive, protocol, evaluator, and baseline hashes are in the ignored aggregate report.
- Common validation exact F1 delta: `+0.000063`; document leak delta: `+0.002950` (worse). `DISEASE` any-class coverage fell by `0.020305`, just beyond the allowed loss, so the common gate failed.
- Medical validation exact F1 delta: `-0.001297`; document leak delta: `+0.007021` (worse), against a required improvement of at least `0.01`. `SNILS` coverage remained `44/49` (`0.897959`), below the required `45/49` and without the required `0.02` gain. The medical gate failed.
- The candidate was also scored on the small authored 1.17 validation slice for diagnostics; it was not used for tuning or to satisfy a gate.
- No sealed 1.16.0 data and no 1.17 test/challenge records were read or scored. Candidate and aggregate reports remain under ignored `references/evaluations/medical-1.17/`.

## Acceptance and disposition

The training compatibility change and its tests are accepted. Candidate quality is **HOLD**. The model has not demonstrated lower document leak, restored SNILS coverage, or safety against the registered gates. Do not activate this candidate or run independent evaluation against the under-supported held-outs. PR4 was not created: the existing 20,000-character Unicode boundary regression passes, and no product-path defect was reproduced that would justify integration changes.

The one-shot run report captures hashes of all files used, including the exact training-runner source hash. Subsequent reviewer fixes hardened the runner's provenance and readiness guards; the candidate was not retrained or rescored after those safeguards were added. The report's Git revision is the pre-PR3 commit, while the model and runner were dirty files whose exact hashes are recorded in the report; the original one-shot runner source is not preserved as a separate Git revision. This limits exact rerun reconstruction from Git alone, so PR3 is not claimed as a reproducibility pass.

## Review and verification

- Independent code review found and corrected an inverted readiness assertion and tightened readiness to require both holdout splits, editorial completion, and per-label support. No held-out data was accessed by the reviewer.
- Full backend suite after final changes: **69 passed, 1 pre-existing skip**. Python syntax compilation and `git diff --check` passed.
- Re-running the frozen baseline scorer after the training-code change reproduced common and medical validation exact F1, document leak, and medical `SNILS` coverage exactly; no 1.16.0 test/challenge inputs were read.
