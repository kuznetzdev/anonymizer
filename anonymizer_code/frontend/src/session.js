export function isSnapshotPending(busy) {
  return busy === 'analyze' || busy === 'save';
}

export async function requestLogout(apiRequest) {
  try {
    await apiRequest('/api/auth/logout', { method: 'POST' });
  } catch (reason) {
    if (reason.status !== 401) throw reason;
  }
}
