# PR 2 report — authored hard cases and corpus integration

## Result

Corpus construction and structural checks: **PASS**. Independent evaluation readiness: **HOLD**.

## Changes

- Added locally authored, Russian medical scenarios with long notes, OCR noise outside spans, multi-entity address chains, eponym/person contrasts, contextual numeric negatives, and valid identifier checksums.
- Kept all 17 existing ontology labels. The 1.16.0 train and validation files are copied only after fixed SHA-256 checks; their source bytes and record counts remain unchanged.
- Built a deterministic importer ZIP with train, validation, and test. The challenge is stored separately under ignored `references/` data and is absent from the ZIP.
- Added validation for offsets, labels, checksums, source-group/template split leakage, duplicate text, known annotated surfaces in negative rows, importer length limits, and refusal to overwrite existing output.
- No external corpus data was imported. Source/license review is recorded in `SOURCE_REGISTER.md`.

## Verification

- Full backend suite: **66 passed, 1 pre-existing skip** (`test_finalized_v17_reference_entities_align`; finalized release corpus is absent).
- Independent editorial review matched all spans in authored train/validation/test/challenge, found no unannotated disease mentions in negative records, and confirmed coherent address geography and distinct held-out formats.
- Independent build review confirmed the archive split membership, challenge separation, source equality, and no overlap between validation, test, and challenge texts.
- Archive SHA-256: `f215a3282043539b5c8fbac011ecf42deeb117e3d018ad4b6b3da13f4735f123`.
- Sealed challenge records SHA-256: `ec86ff1e959932e68494c1ad4d35d6eab4502537dd9c5f785bf6f816d68e86f8`.
- `git diff --check` passes. Private corpus contents and challenge data remain ignored and untracked.

## Acceptance and limits

Structural integration is accepted. Quality evaluation remains **HOLD**: the authored test has 2 examples per target label and challenge has 1 per target label (2 for `DISEASE`), below the preregistered minimum of 30. Metadata correctly records `evaluation_ready=false`; these small splits cannot support a quality claim. They must not be treated as independent release evidence or used to claim improvement. Candidate quality and activation remain gated on the preregistered development and independent thresholds.

## Review corrections

Initial blind review found missed repeated disease mentions, disease names in negative rows, inconsistent test geography, and overly similar held-out formats. The source scenarios were corrected; a fresh blind editorial review and final integration review passed. No sealed 1.16.0 test or challenge examples were opened, copied, or scored.
