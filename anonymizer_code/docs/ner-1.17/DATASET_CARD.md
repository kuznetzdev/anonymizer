# Anonymizer NER hardcases 1.17.0 dataset card

## Status

Draft. The release is private, stored under the ignored `references/` directory, and is not approved for model-quality claims until all splits are frozen, reviewed, and meet the preregistered support gates.

## Purpose and scope

Russian-language synthetic clinical and administrative notes for testing exact span recognition and masking coverage across the 17 existing built-in labels. Priority scenarios include patient names, disease/eponym mentions, street and address components, institutions versus suppliers, passport details, and SNILS.

## Data origin and privacy

New examples are authored locally from fictional names, organizations, places, contact details, and checksummed identifiers. No patient records or third-party corpus text are imported. Existing 1.16.0 train and validation records are copied unchanged into matching splits. 1.16.0 test and challenge are excluded.

## Intended use

Development and offline evaluation of the existing NER model under the fixed 17-label mask policy. Synthetic results do not establish safety on real clinical records.

## Known limitations

- Synthetic writing patterns may not represent real hospitals, OCR systems, or regional variation.
- Checksum validity does not establish that a synthetic identifier is allocated to a real person.
- Class support, document lengths, scenario composition, and editor review counts are recorded in the generated private `quality_report.json` and release reports; no unverified values are asserted here.
- The release must remain `HOLD` for an evaluation gate when a target label lacks required support or editorial review.
- The current authored evaluation splits provide 2 examples per priority label in test and 1 in challenge (2 for `DISEASE`), below the minimum of 30. Their metadata records `evaluation_ready=false` and `editorial_review_complete=false`.
- The 1.17.0 candidate failed the preregistered development gates; no independent test/challenge metrics support a release claim.

## Licensing

Local synthetic examples have no third-party data license dependency. External-source review and non-import decisions are recorded in `SOURCE_REGISTER.md`.
