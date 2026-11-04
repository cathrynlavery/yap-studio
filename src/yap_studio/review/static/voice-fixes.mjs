/** Describe the proposed words at their time in the reviewed cut. */
export function voiceSentence(fix) {
  const start = fix.time ? `At ${fix.time} you said` : 'You said';
  return `${start} '${fix.said}'. Fixed: '${fix.say}'.`;
}

/** Distinguish a recorded charge from an estimate. */
export function voiceCost(fix) {
  if (fix.kind === 'own') return 'Cost: 0 credits.';
  return `Cost: ${fix.credits} ElevenLabs credits (${fix.credits_source}).`;
}

/** Show where the replacement audio came from. */
export function voiceSource(fix) {
  return fix.kind === 'own' ? `From your own recording at ${fix.from_time}`
    : 'Re-said by your ElevenLabs voice';
}

/** Keep the creator's saved choice visible. */
export function voiceState(fix) {
  return ({approved: 'Use the fix.', rejected: 'Keep my original.'})[fix.status]
    ?? 'Waiting for your choice.';
}

/** Offer the other choice after a decision. */
export function voiceActions(fix) {
  const approve = {action: 'approve', label: 'Use the fix'};
  const reject = {action: 'reject', label: 'Keep my original'};
  if (fix.status === 'approved') return [reject];
  if (fix.status === 'rejected') return [approve];
  return fix.status === 'ready' ? [approve, reject] : [];
}

export function voiceDecisionBody(fix, action) {
  return {fix: fix.id, action, generation: fix.generation};
}

/** Keep relative paths so a review with several videos uses its own token. */
export function voiceClipUrl(path, token) {
  if (!token) return path;
  const [route, query] = path.split('?');
  const params = new URLSearchParams(query);
  params.set('t', token);
  return `${route}?${params}`;
}
