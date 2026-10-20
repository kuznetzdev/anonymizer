# NER 1.17.0 changelog

## Draft

- Preserve 1.16.0 train/validation records as immutable source records in their matching splits.
- Add locally authored Russian synthetic cases for the existing 17 labels, including negative and long-document scenarios.
- Keep all new test and challenge groups independent from training and validation; evaluate only after candidate freeze.
- Do not import third-party data in this release; review terms and potential coverage value in `SOURCE_REGISTER.md`.

Release acceptance remains pending corpus validation, blind editorial review, and independent model evaluation.
