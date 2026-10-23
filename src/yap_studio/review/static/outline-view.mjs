/** Label an outline beat; indices are zero based and a single beat is Opening. */
export function roleLabel(beat, index, count) {
  if (beat?.role === 'opening' || index === 0) return 'Opening';
  if (beat?.role === 'ending' || index === count - 1) return 'Ending';
  return `Beat ${index + 1}`;
}

/** Describe the render gate. formatTime receives a UTC deadline string. */
export function statusText(status, formatTime) {
  switch (status?.state) {
    case 'missing': return 'No story outline yet.';
    case 'stale': return 'The story changed. The agent is updating the outline.';
    case 'approved': return 'Approved. Rendering can start.';
    case 'notes_sent': return 'Notes sent. The agent is revising the story.';
    case 'timed_out': return 'The wait ran out, so rendering can start.';
    default:
      return status?.deadline
        ? `Rendering starts at ${formatTime(status.deadline)} unless you send notes or approve.`
        : 'Waiting for your OK on the story.';
  }
}

/** Display nonnegative finite seconds rounded to one decimal, without trailing zeroes. */
export function durationText(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return 'Length unavailable';
  const value = Math.round(seconds * 10) / 10;
  return `${value} ${value === 1 ? 'second' : 'seconds'}`;
}

/** Label the send action using only draft notes (batch is null). */
export function noteCountText(notes) {
  const count = notes.filter(note => note.batch == null).length;
  return `Send ${count} ${count === 1 ? 'note' : 'notes'}`;
}

/** Return fresh objects for the five supported wait choices, in display order. */
export function waitChoices() {
  return [
    {label: 'Not set', minutes: null},
    {label: '15 minutes', minutes: 15},
    {label: '1 hour', minutes: 60},
    {label: '4 hours', minutes: 240},
    {label: '1 day', minutes: 1440},
  ];
}

/** Pick the server's wait, falling back to its local deadline for other minutes. */
export function waitOption(status, formatTime) {
  if (!status?.deadline) return {value: 'none', label: 'Not set', minutes: null};
  const choice = waitChoices().find(item => item.minutes !== null
    && item.minutes === status.wait_minutes);
  if (choice) {
    return {...choice, value: String(choice.minutes)};
  }
  return {value: 'deadline', label: `Until ${formatTime(status.deadline)}`, minutes: null};
}

/** Keep text unchanged within limit; otherwise keep limit words and append three dots. */
export function clampWords(text, limit) {
  if (!Number.isInteger(limit) || limit < 1) return '';
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length <= limit ? text : `${words.slice(0, limit).join(' ')}...`;
}
