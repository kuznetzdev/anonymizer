# PR 1 report — baseline and reproducibility

## Changes

- Added artifact v3 inference support using the exact `combined_features` contract. Artifact v1/v2 logic remains on its prior feature contract; default training output is unchanged.
- Added regression coverage for v3 paired features, Unicode span offsets, and a sensitive span at character offset `[19999, 20000)` in a 20,000-character input.
- Added `scripts/ner_1_17/run_dev.py`, which verifies the frozen candidate SHA, registered train/validation/ontology hashes, artifact format/dimensions, and unchanged inputs at end of scoring. It reads only common and medical validation for scoring; its output is aggregate metrics and hashes. The fixed masking policy enables all 17 non-deprecated built-in labels, including `DISEASE`.
- Added the frozen baseline and preregistered evaluation protocol under `docs/ner-1.17/`.
- Isolated testing in a Python 3.12.7 virtual environment outside the worktree. All backend requirements installed, including `psycopg 3.2.12`; no dependency or requirement file changed.

## Verification

- `python -m unittest discover -s tests -p 'test_*.py' -v` from `anonymizer_code/backend`: **42 tests passed, 1 existing corpus-dependent test skipped**.
- The v3 baseline runner completed against the copied frozen artifact and permitted validation files. Its common and medical exact micro F1, document leak rate, and `SNILS` support/coverage match the existing 1.16 development report exactly.
- Artifact SHA-256: `e39ca86b2daa1a3f3e505ca644222e0e148fe4b4a2224b5a6b692188346700ba`.
- The initial runner attempt exposed a feature-contract mismatch (`role_features` instead of v3 `combined_features`). That result was rejected; the decoder was corrected, and only the fresh corrected run is recorded in `BASELINE.md` and the private aggregate report.
- `git diff --check` and Python syntax compilation of the new runner pass.

## Acceptance

PR1 is **PASS** for v3 baseline compatibility, validation-only reproducibility, and the backend unit-test gate. Candidate quality remains **HOLD** until PR3 satisfies every preregistered development and independent gate. The existing 1.16 candidate itself remains `HOLD` against 1.15: its medical leak reduction was 0.60 percentage points and its `SNILS` coverage was 44/49.

## Scope and limitations

The isolated copy contains only the frozen candidate artifact, main train/validation, and 1.16 train/validation; no 1.16 test or challenge files were copied or scored. One broad metric search earlier surfaced a few lines from closed evaluation files in tool output; those lines were not used for corpus authoring, tuning, or scoring. The 1.17 independent corpus and candidate have not yet been built or evaluated.
