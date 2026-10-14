import assert from 'node:assert/strict';
import test from 'node:test';
import { buildFeedbackPayload, countFeedbackReason, examplePayloadSignature } from '../src/feedback_payload.js';

test('ignores feedback entries cleared by the error toggle', () => {
  assert.deepEqual(buildFeedbackPayload({ entity_1: undefined }, []), []);
});

test('counts feedback reasons safely when cleared entries remain', () => {
  assert.equal(countFeedbackReason({ cleared: undefined, falsePositive: { reason: 'false_positive' } }, 'false_positive'), 1);
});

test('treats an unchanged annotation as the same save even if array order differs', () => {
  const first = {
    text: 'name 123',
    entities: [{ start: 5, end: 8, label: 'ID' }, { start: 0, end: 4, label: 'PER' }],
    empty_review_confirmed: false,
    feedback: [{ start: 0, end: 4, label: 'PER', reason: 'correct', note: '' }],
  };
  const second = {
    ...first,
    entities: [...first.entities].reverse(),
    feedback: [...first.feedback].reverse(),
  };

  assert.equal(examplePayloadSignature(first), examplePayloadSignature(second));
  assert.notEqual(examplePayloadSignature(first), examplePayloadSignature({ ...first, text: 'other' }));
});

test('uses current entity values when building feedback payload', () => {
  const feedback = { entity_1: { start: 1, end: 2, label: 'OLD', reason: 'wrong_label' } };
  const entities = [{ id: 'entity_1', start: 4, end: 8, label: 'PER' }];

  assert.deepEqual(buildFeedbackPayload(feedback, entities), [
    { start: 4, end: 8, label: 'PER', reason: 'wrong_label', note: '' },
  ]);
});
