# NER 1.17.0 evaluation protocol

Freeze this protocol before selecting or scoring the 1.17.0 candidate. The scorer uses a fixed evaluation policy with all 17 non-deprecated built-in labels active, including `DISEASE`, for both baseline and candidate. This matches the backend's default category state (`active TRUE`); runtime-specific category overrides are outside this offline comparison.

The safety target set is all 17 non-deprecated built-in labels: `APART`, `COUNTRY`, `DISEASE`, `EMAIL`, `HOUSE`, `INN`, `INSTITUTION`, `KPP`, `LOCALITY`, `OGRN`, `ORG`, `PASSPORT`, `PER`, `PHONE`, `REGION`, `SNILS`, and `STREET`. Legacy `ADDR` is excluded because the ontology marks it deprecated. A target label missing from a required split has insufficient support and cannot be reported as a pass.

## Development data

Development uses the common validation and medical validation splits only. It may guide the single planned candidate run. The 1.16.0 test and challenge are sealed and excluded from tuning and scoring. The 1.17.0 test and challenge are created from independent authored source groups, frozen with hashes, and each scored once after the candidate is frozen.

## Metrics

- Exact span/type micro precision, recall and F1, plus per-label support and scores.
- Document leak rate: fraction of documents with at least one supported gold entity not fully covered by a predicted mask span under the same active-label policy used for the comparison.
- Per-label any-class coverage: a gold span counts as covered if any predicted mask span fully contains it, regardless of predicted type.
- False-positive spans and overmasked characters, with support and denominators.
- `DISEASE` exact classification and support are reported separately; it contributes to document leak only when that label is active in the evaluated masking policy.
- Results are reported separately for common, medical, long-document and OCR slices, with support for each slice.

## Preregistered gates

### Development gate

- Exact micro F1 on common and medical validation is no worse than baseline by more than `0.01` absolute.
- Document leak rate on both splits is no worse than baseline by more than `0.01` absolute.
- Coverage for each target label is no worse than baseline by more than `0.02` absolute.
- Medical validation document leak improves by at least `0.01` absolute.
- `SNILS` coverage improves by at least `0.02` absolute from candidate 1.16 and reaches at least `45/49` (`0.9184`), matching the recorded 1.15 development score.

Failure of any required condition is `HOLD`; a candidate on `HOLD` does not proceed to independent evaluation.

### Independent 1.17 challenge

- Document leak rate improves by at least `0.05` absolute from baseline.
- Coverage of every target label is no worse by more than `0.05` absolute.
- Overmasked characters increase by no more than `20%` relative to baseline.
- Each target label has at least 30 gold spans. Lower support yields `HOLD` for that safety gate.

### Frozen 1.17 test

- Exact micro F1 is no worse by more than `0.01` absolute.
- Document leak rate is no worse by more than `0.01` absolute.
- Coverage of every target label is no worse by more than `0.02` absolute.
- Overmasked characters increase by no more than `20%` relative to baseline.
- Report support per label; low support cannot establish safety.

Any failed or under-supported independent gate means `HOLD`. After a frozen test or challenge is opened, do not alter the candidate or rerun on those same records. Create fresh held-out splits for another experiment.

## One-shot freeze

Before independent evaluation, hash both model artifacts, model and evaluator code, protocol, train and validation data, release ZIP, challenge and test. Validate group/template isolation. Store the freeze manifest and one-shot sentinel under ignored `references/evaluations/medical-1.17/`; the runner must refuse to overwrite a completed result. Reports tracked in Git contain aggregate metrics and hashes only.
