import test from 'node:test';
import assert from 'node:assert/strict';

import { groupCategories } from '../src/category_groups.js';

test('groups document leaves while keeping unclassified categories available', () => {
  const groups = groupCategories([
    { label: 'PHONE', parent_class: null },
    { label: 'PASSPORT', parent_class: 'DOCUMENT' },
    { label: 'OMS_POLICY_ID', parent_class: 'DOCUMENT' },
  ]);

  assert.deepEqual(groups, [
    { parentClass: '', items: [{ label: 'PHONE', parent_class: null }] },
    { parentClass: 'DOCUMENT', items: [
      { label: 'PASSPORT', parent_class: 'DOCUMENT' },
      { label: 'OMS_POLICY_ID', parent_class: 'DOCUMENT' },
    ] },
  ]);
});

test('supports older category payloads without parent_class', () => {
  assert.deepEqual(groupCategories([{ label: 'PER' }]), [
    { parentClass: '', items: [{ label: 'PER' }] },
  ]);
});
