# Baseline 1.16.0

## Frozen candidate

- Artifact: `references/evaluations/medical-1.16-dev/candidate_1_16.npz`
- SHA-256: `e39ca86b2daa1a3f3e505ca644222e0e148fe4b4a2224b5a6b692188346700ba`
- Format: artifact v3, 16,384 sparse features, 35 BIO classes.
- Inference: `FeatureHasher` plus `SGDClassifier`; v3 rows use the `combined_features` contract: base token features plus nearby token and token-shape pairs at offsets ±1 and ±2.
- The artifact is evaluated as-is. PR1 does not retrain or update it.

## Allowed development inputs

Only these data files are copied into the isolated worktree:

| Input | SHA-256 |
|---|---|
| Common training `expanded_dataset/train.jsonl` | `921716a5524295c1f8d59bd23904a4104e1905620e8d1888b88b7e1a3767d6bd` |
| Common validation `expanded_dataset/val.jsonl` | `7f0a3eb27a64fe5a217c296f59a2930291fdf6faa5a71e5c85aaf334529df59b` |
| Hardcases 1.16 train | `43a7f35e0d479ab0edf4abfa23acdd4832b96ad557c8a03a5882b69a3bae6d28` |
| Hardcases 1.16 validation | `dca4098b494da6b7e62d70c93126ddbd71e8df324b92568fc60c03ff3ad0949f` |

The validation scorer reads common and medical validation only. Closed test and challenge records are not inputs to PR1. The artifact, validation files and aggregate output remain under ignored `references/`.

The runner requires the hashes above for all four corpus files and also pins the ontology SHA-256 to `618768a500e52b8d5505ed99aad79584069818a3588a9d3dfff10bc7320d58e7`. It records code hashes and Git revision as run provenance. Scoring uses all 17 non-deprecated labels as active, including `DISEASE`, for consistency with the backend default.

## Prior development decision

The existing 1.16.0 development report compared candidate 1.16.0 with 1.15.0 on common and medical validation. It returned `HOLD`: medical document leak improved by `0.006018` (0.60 percentage points), below the preregistered `0.01` minimum, and medical `SNILS` coverage fell by `0.020408` (2.04 percentage points). The 1.16.0 report is historical context; the new runner recomputes aggregate metrics for the frozen 1.16 artifact and records hashes.

The isolated PR1 runner reproduced the historical candidate 1.16.0 aggregates exactly:

| Split | Records | Exact micro F1 | Document leak rate |
|---|---:|---:|---:|
| Common validation | 1,061 | 0.705311 | 0.511308 |
| Medical validation | 1,141 | 0.871295 | 0.225677 |

Medical `SNILS` any-class coverage is `44/49` (`0.897959`). This is below the preregistered recovery threshold `45/49` and is one of the targets for 1.17. The private aggregate report also contains per-label support and hashes; it contains no text or per-record predictions.

## Reproduction

From the repository root, with the backend requirements installed in the isolated environment:

```powershell
python anonymizer_code/scripts/ner_1_17/run_dev.py
```

The command writes only aggregate metrics and hashes to `references/evaluations/medical-1.17/baseline_1_16_validation.json`. It does not serialize text, per-record predictions or error examples.
