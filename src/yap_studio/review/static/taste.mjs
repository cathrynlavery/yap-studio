import {frameTimeText, parseFps} from './timeline.mjs';

/** Name the part of a future video a rule affects. */
export function scopeName(scope) {
  return ({cut: 'Cut', captions: 'Captions', cards: 'Cards', pacing: 'Pacing',
    music: 'Music'})[scope] ?? 'Rule';
}

/** Describe a machine form in words the creator can read. */
export function machineText(machine) {
  if (machine?.kind === 'style_field') {
    const [section, key] = machine.field.split('.');
    const sectionWord = ({captions: 'caption', hook: 'title', graphics: 'card',
      music: 'music'})[section] ?? section;
    const unit = key.match(/_(px|s|ms)$/u)?.[1];
    const field = (unit ? key.slice(0, -unit.length - 1) : key).replaceAll('_', ' ');
    const value = typeof machine.value === 'boolean' ? (machine.value ? 'on' : 'off') : machine.value;
    return `Sets ${sectionWord} ${field} to ${value}${unit ? ` ${unit}` : ''}`;
  }
  if (machine?.kind === 'filler_word') {
    const position = machine.position === 'leading' ? 'at the start' : 'anywhere';
    return `Cuts the filler word "${machine.word}" ${position}`;
  }
  if (machine?.kind === 'placement') return `Keeps ${machine.layer} clear of the face`;
  return '';
}

/** Join the scope, condition and machine form without exposing their keys. */
export function metaText(item) {
  return [scopeName(item.scope), item.condition?.text ? `Only when: ${item.condition.text}` : '',
    machineText(item.machine)].filter(Boolean).join('. ');
}

/** Keep approved wording visible after an edited rule has ended. */
export function sentenceText(item) {
  return item.edited && item.rule ? item.rule.sentence : item.sentence;
}

/** Describe the saved decision and the revision at which it was approved. */
export function stateText(item) {
  if (item.decided_here === false) {
    if (item.state === 'rejected') return 'Rejected earlier.';
    if (item.state === 'approved') {
      return `Approved earlier (taste revision ${item.rule.approved_revision}).`;
    }
  }
  if (item.state === 'approved') {
    return item.edited
      ? `Approved with your wording (taste revision ${item.rule.approved_revision}).`
      : `Approved. Applies to future videos (taste revision ${item.rule.approved_revision}).`;
  }
  return ({rejected: 'Rejected.', revoked: 'Stopped. No longer applied.',
    superseded: 'Replaced by a newer rule.'})[item.state] ?? '';
}

/** List the controls allowed by the proposal's current state. */
export function tasteActions(item) {
  if (item.state === 'pending') return [
    {action: 'approve', label: 'Approve'}, {action: 'reject', label: 'Reject'},
    {action: 'edit', label: 'Edit wording'},
  ];
  if (item.state === 'rejected') return [{action: 'approve', label: 'Approve instead'}];
  if (item.state === 'approved') return [{action: 'revoke', label: 'Stop using'}];
  return [];
}

/** Split proposals without changing inbox order within either group. */
export function splitProposals(items) {
  const pending = [];
  const decided = [];
  for (const item of items) (item.state === 'pending' ? pending : decided).push(item);
  return {pending, decided};
}

/** Put pending proposals first, preserving inbox order within both groups. */
export function orderedProposals(items) {
  const {pending, decided} = splitProposals(items);
  return [...pending, ...decided];
}

/** Count proposals still waiting for a decision. */
export function pendingCount(items) {
  return items.filter((item) => item.state === 'pending').length;
}

/** Use each note's own version rate to describe its time and seek target. */
export function sourceRows(item, versions) {
  return item.sources.map((source) => {
    if (source.type === 'answer') return {label: 'From your setup answers',
      location: '', quote: source.answer, version: null, frame: null,
      canGo: false, time_s: null};
    const version = versions.find((entry) => entry.number === source.version);
    const fps = parseFps(version?.fps);
    const frame = Number.isInteger(source.start_frame) ? source.start_frame : null;
    const time = frame !== null && fps ? frameTimeText(frame, fps) : null;
    const label = source.label === 'keep' ? 'Keep' : 'Fix';
    const location = source.version === null ? 'Version unavailable' : `v${source.version}`;
    return {label, location: time === null ? location : `${location} at ${time}`,
      quote: source.quote, version: source.version, frame,
      canGo: Boolean(version) && frame !== null,
      time_s: frame !== null && fps ? frame * fps.den / fps.num : null};
  });
}

/** Send optional wording only when it changes the proposed sentence. */
export function decisionBody(item, action, sentence) {
  const body = {proposal: item.id, action};
  if (action === 'approve' && typeof sentence === 'string'
    && sentence.trim() !== item.sentence.trim()) body.sentence = sentence.trim();
  return body;
}
