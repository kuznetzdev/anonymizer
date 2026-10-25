# NER 1.17.0 annotation policy

This policy applies to locally authored synthetic examples. The 1.16.0 release is an immutable source; its train and validation records are copied byte-for-byte into matching 1.17.0 splits. The old test and challenge are excluded.

## Span rules

- Offsets are zero-based Python Unicode code-point offsets with half-open `[start, end)` bounds. For every entity, `text[start:end]` must equal the full annotated surface.
- Spans must be ordered and non-overlapping. Include only the entity itself; exclude surrounding punctuation and generic cues such as `ул.`, `г.`, `дом`, `паспорт`, or `ИНН` unless the token is part of the name/value.
- Address components use their existing separate labels (`STREET`, `HOUSE`, `APART`, `LOCALITY`, `REGION`, `COUNTRY`). Annotate each explicit component, not the entire address under one label.
- Use the current ontology descriptions for all 17 supported labels. Do not emit deprecated `ADDR` or invent a new class.
- Annotate a disease mention as `DISEASE` regardless of negation, uncertainty, or whose history is discussed, following the current ontology. Eponymous disease names remain `DISEASE`; a person named in surrounding prose remains `PER` only when the text identifies an actual person.
- `INSTITUTION` is a named public or service institution by role; `ORG` is a company or commercial supplier. Annotate the organization name, not a generic role noun.
- Valid synthetic identifiers use checksum-valid forms where a checksum applies. If a record presents an invalid-checksum value as a person's identifier, annotate it as sensitive anyway; a checksum does not decide whether to mask. Unlabeled numeric hard negatives must have a clearly non-identifier role in context.
- A negative record contains no span of any of the 17 supported labels. Do not create a negative by merely omitting an uncertain annotation.

## Split and review rules

- Every source group and exact template belongs to one split only. Held-out challenge/test must be authored independently from training examples and have distinct source-group and template identifiers.
- OCR mutations are applied only outside entity spans unless an editor re-annotates the full changed record. Revalidate every shifted offset against its surface.
- Review all held-out records independently and at least 50 training records across every scenario family. Record the review scope and corrected-item counts without copying private text into tracked reports.
- Uncertain or disputed records are excluded until resolved; model predictions are never shown to annotators during this review.
