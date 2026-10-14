export function countFeedbackReason(feedback, reason) {
  return Object.values(feedback).filter((item) => item?.reason === reason).length;
}

export function examplePayloadSignature(payload) {
  const entities = [...payload.entities].sort((left, right) =>
    left.start - right.start || left.end - right.end || left.label.localeCompare(right.label));
  const feedback = [...payload.feedback].sort((left, right) =>
    left.start - right.start || left.end - right.end || left.label.localeCompare(right.label)
    || left.reason.localeCompare(right.reason) || left.note.localeCompare(right.note));
  return JSON.stringify({ ...payload, entities, feedback });
}

export function buildFeedbackPayload(feedback, entities) {
  return Object.entries(feedback).filter(([, item]) => item).map(([id, item]) => {
    const entity = entities.find((candidate) => candidate.id === id);
    return {
      start: entity?.start ?? item.start,
      end: entity?.end ?? item.end,
      label: entity?.label ?? item.label,
      reason: item.reason,
      note: item.note || '',
    };
  }).filter((item) => Number.isInteger(item.start) && Number.isInteger(item.end) && item.label);
}
