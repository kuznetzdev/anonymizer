import assert from 'node:assert/strict';
import test from 'node:test';
import { isSnapshotPending, requestLogout } from '../src/session.js';

test('locks snapshot-changing interactions only while analyzing or saving', () => {
  assert.equal(isSnapshotPending('analyze'), true);
  assert.equal(isSnapshotPending('save'), true);
  assert.equal(isSnapshotPending('train'), false);
  assert.equal(isSnapshotPending(''), false);
});

test('logout treats success and an expired session as signed out', async () => {
  await assert.doesNotReject(() => requestLogout(async () => {}));
  await assert.doesNotReject(() => requestLogout(async () => {
    throw Object.assign(new Error('expired'), { status: 401 });
  }));
});

test('logout propagates other failures', async () => {
  const failure = new Error('network unavailable');
  await assert.rejects(() => requestLogout(async () => { throw failure; }), failure);
});
