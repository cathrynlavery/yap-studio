/** Display names for the optional scope of a keep or fix comment. */
export const TARGET_NAMES = Object.freeze({
  story: 'Story', speech: 'Speech', caption: 'Caption', element: 'Graphic', video: 'Whole video',
});

/** Place a map item on the same bounded time axis as the scrubber. */
export function mapBlockPosition(item, duration) {
  if (!(duration > 0) || !Number.isFinite(duration)) return {left: 0, width: 0};
  const percent = (time) => Math.max(0, Math.min(100, time / duration * 100));
  const left = percent(item.start);
  return {left, width: Math.max(0, percent(item.end) - left)};
}

/** Return the picture within an object-fit contain video box. */
export function pictureRect(box, videoWidth, videoHeight) {
  const {left, top, width, height} = box;
  if (!(videoWidth > 0 && videoHeight > 0 && width > 0 && height > 0)) {
    return {left, top, width, height};
  }
  const scale = Math.min(width / videoWidth, height / videoHeight);
  const drawnWidth = videoWidth * scale;
  const drawnHeight = videoHeight * scale;
  return {left: left + (width - drawnWidth) / 2, top: top + (height - drawnHeight) / 2,
    width: drawnWidth, height: drawnHeight};
}

/** Return rounded picture fractions, ignoring points on black bars. */
export function pinAt(rect, clientX, clientY) {
  const x = (clientX - rect.left) / rect.width;
  const y = (clientY - rect.top) / rect.height;
  if (![x, y].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) return null;
  return {x: Math.round(x * 10000) / 10000, y: Math.round(y * 10000) / 10000};
}

/** Return the marker centre in the rectangle's coordinate system. */
export function pinPoint(rect, pin) {
  return {left: rect.left + pin.x * rect.width, top: rect.top + pin.y * rect.height};
}

/** Describe the spot without relying on colour. */
export function pinText(pin) {
  return `Pinned ${Math.round(pin.x * 100)}% across, ${Math.round(pin.y * 100)}% down`;
}

/** Attach the displayed frame or full player time to a draft's coordinates. */
export function pinBody(pin, fps, frame, time) {
  return pin ? {x: pin.x, y: pin.y, ...(fps ? {frame} : {time_s: time})} : null;
}

/** Check a candidate pin against the note's point or range. */
export function pinInSpan(comment, fps, frame, time) {
  if (fps) return comment.end_frame == null ? frame === comment.start_frame
    : comment.start_frame <= frame && frame < comment.end_frame;
  return comment.end_s == null ? Math.abs(time - comment.start_s) <= 0.0005
    : comment.start_s <= time && time <= comment.end_s;
}

/** Show a saved pin only on its recorded frame. Playback is checked by the caller. */
export function pinShown(comment, fps, frame, time) {
  if (!comment?.pin) return false;
  return fps ? frame === comment.pin.frame : Math.abs(time - comment.pin.time_s) <= 1 / 60;
}

/** Return the target's display name, or No target when it is not stated. */
export function targetName(target) {
  return TARGET_NAMES[target] ?? 'No target';
}

/** Offer a spelling request only for a Fix about Caption. */
export function spellingAvailable(kind, target) {
  return kind === 'fix' && target === 'caption';
}

/** Build a trimmed request, or null when both spelling fields are empty. */
export function spellingRequest(kind, target, wrong, right) {
  if (!spellingAvailable(kind, target)) return null;
  const from = wrong.trim();
  const to = right.trim();
  if (!from && !to) return null;
  for (const word of [wrong, right]) {
    const length = [...word.trim()].length;
    if (length < 1 || length > 80 || /[\r\n\v\f\u0085\u2028\u2029]/u.test(word)) {
      throw new Error('Shown as and Should be both need 1 to 80 characters, no newlines.');
    }
  }
  if (from === to) throw new Error('Shown as and Should be must differ.');
  return {type: 'spelling', from, to};
}

/** Describe the attached request using plain text for safe row rendering. */
export function spellingText(request) {
  return request?.type === 'spelling' ? `Spelling: ${request.from} -> ${request.to}` : '';
}

function cutQuote(words) {
  const text = [...words.map((word) => word.text).join(' ')];
  return `"${text.length > 120 ? `${text.slice(0, 117).join('')}...` : text.join('')}"`;
}

function wordCountText(words) {
  return `${words.length} ${words.length === 1 ? 'word' : 'words'}`;
}

/** Find a visible caption using the contained picture's coordinate system. */
export function captionAt(captions, time, rect, clientX, clientY, frame = null) {
  if (!captions?.available || !(captions.width > 0 && captions.height > 0)) return null;
  const point = pinAt(rect, clientX, clientY);
  if (!point) return null;
  const x = point.x * captions.width;
  const y = point.y * captions.height;
  return captions.phrases.find((phrase) => phrase.start <= time && time < phrase.end
    && phrase.boxes.some((box) => box.x <= x && x <= box.x + box.w
      && box.y <= y && y <= box.y + box.h
      && ((box.start_frame == null && box.end_frame == null)
        || (Number.isInteger(frame) && box.start_frame <= frame && frame < box.end_frame)))) ?? null;
}

/** Describe a typed request as plain text. */
export function requestText(request) {
  if (request?.type === 'cut') {
    return `Cut: ${cutQuote(request.words)} (${wordCountText(request.words)})`;
  }
  if (request?.type === 'restore') {
    if (request.text === null) return 'Restore: removed audio';
    const words = request.text.trim().split(/\s+/u).filter(Boolean).length;
    return `Restore: removed part (${words} ${words === 1 ? 'word' : 'words'})`;
  }
  return spellingText(request);
}

/** Keep story requests attached when their draft text is edited. */
export function editedRequest(comment, wrong, right) {
  if (['restore', 'cut', 'caption_fix'].includes(comment.request?.type)) return {...comment.request};
  return spellingRequest(comment.kind, comment.target, wrong, right);
}

function timeFraction(value) {
  if (typeof value === 'object') return [BigInt(value.num), BigInt(value.den)];
  const [coefficient, exponent = '0'] = String(value).split('e');
  const [whole, decimal = ''] = coefficient.split('.');
  const scale = decimal.length - Number(exponent);
  const numerator = BigInt(whole + decimal);
  return scale >= 0 ? [numerator, 10n ** BigInt(scale)]
    : [numerator * 10n ** BigInt(-scale), 1n];
}

/** Select output occurrences by exact midpoint, including start and excluding end. */
export function wordsInRange(words, start, end) {
  const [s, sd] = timeFraction(start);
  const [e, ed] = timeFraction(end);
  return words.filter((word) => {
    const [a, ad] = timeFraction(word.start);
    const [b, bd] = timeFraction(word.end);
    const mid = a * bd + b * ad;
    const den = 2n * ad * bd;
    return s * den <= mid * sd && mid * ed < e * den;
  });
}

/** Quote the selected words with the same truncation as the text export. */
export function cutWordsText(words) {
  return `${cutQuote(words)} · ${wordCountText(words)}`;
}

/** Offer Cut this only for a known list with 1 to 200 covered words. */
export function cutState(wordsState, covered) {
  const show = wordsState?.available === true;
  const reason = !show ? '' : covered.length === 0 ? 'No spoken words in this range.'
    : covered.length > 200 ? 'Select at most 200 words to cut in one note.' : '';
  return {show, enabled: show && !reason, reason};
}

/** Send only the selected edges and a nonempty optional note. */
export function cutBody(selection, fps, note = '') {
  const body = fps ? {start_frame: selection.start_frame, end_frame: selection.end_frame}
    : {start_s: selection.start_s, end_s: selection.end_s};
  if (note.trim()) body.text = note.trim();
  return body;
}

/** Show the range duration rounded to one decimal second. */
export function rangeLengthText(start, end) {
  return `${(end - start).toFixed(1)} s`;
}

/** Name a removal using creator-facing words. */
export function removedKindName(kind) {
  return ({filler: 'Filler', silence: 'Pause', stutter: 'Stutter', restart: 'Restart',
    manual: 'Manual cut', story: 'Cut from the story', trim: 'Trim'})[kind] ?? 'Cut';
}

/** Hide pauses unless the creator chooses to show them, preserving output order. */
export function visibleRemovedRanges(ranges, showPauses = false) {
  return ranges.filter((range) => showPauses || range.kind !== 'silence');
}

/** Find an attached restore request among this version's comments. */
export function rangeRequested(range, comments) {
  return comments.some(({request}) => request?.type === 'restore'
    && request.range === range.id && request.take === range.take
    && request.start === range.start && request.end === range.end);
}

/** Name a reviewer finding using plain, creator-facing words. */
export function findingKindLabel(kind) {
  return ({slow: 'Slow stretch', jarring_cut: 'Jarring cut', face_covered: 'Face covered',
    caption_issue: 'Caption problem', audio: 'Sound', other: 'Other'})[kind] ?? 'Other';
}

/** Distinguish agent notes from measurements made by an automatic check. */
export function findingSourceLabel(finding) {
  return finding.source === 'machine' ? 'Automatic check' : 'Agent';
}

/** Show finite evidence numbers in supplied key order, with at most two decimals. */
export function findingEvidenceText(evidence) {
  return Object.entries(evidence ?? {}).filter(([, value]) => Number.isFinite(value))
    .map(([key, value]) => `${key} ${Number(value.toFixed(2))}`).join(' · ');
}

/** Build a Fix draft with CFR frames or variable rate seconds clamped to duration. */
export function findingNoteBody(finding, cfr, duration = null) {
  const text = `Flagged: ${finding.reason} Suggested fix: ${finding.fix}`
    .replace(/\r\n/g, '\n').trim();
  const body = {kind: 'fix', target: finding.target ?? null,
    text: [...text].slice(0, 5000).join('').trimEnd()};
  if (cfr) return {...body, start_frame: finding.start_frame, end_frame: finding.end_frame ?? null};
  const start = duration == null ? finding.start_s : Math.min(finding.start_s, duration);
  const end = finding.end_s == null ? null
    : duration == null ? finding.end_s : Math.min(finding.end_s, duration);
  return {...body, start_s: start, end_s: end !== null && end > start ? end : null};
}

/** Match note text and timing, including clamped seconds and one-frame point collapse. */
export function findingAdded(finding, comments, duration = null) {
  const body = findingNoteBody(finding, false, duration);
  const endFrame = finding.end_frame === finding.start_frame + 1 ? null
    : finding.end_frame ?? null;
  return comments.some((comment) => comment.text === body.text
    && (comment.start_frame != null
      ? comment.start_frame === finding.start_frame
        && (comment.end_frame === comment.start_frame + 1 ? null : comment.end_frame ?? null) === endFrame
      : comment.start_s === body.start_s && (comment.end_s ?? null) === body.end_s));
}

/** Label the latest disposition of an earlier comment. */
export function threadStateLabel(state) {
  return ({open: 'Open', needs_clarification: 'Needs your answer', changed: 'Changed',
    kept: 'Kept', declined: 'Declined'})[state] ?? 'Open';
}

function evidenceFields(value, required, optional = []) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && required.every((key) => Object.hasOwn(value, key))
    && Object.keys(value).every((key) => required.includes(key) || optional.includes(key));
}

/** Return only complete, typed reply-evidence@1 records; older evidence stays readable. */
export function replyEvidence(reply) {
  const evidence = reply?.evidence;
  if (!evidenceFields(evidence, ['schema', 'batch', 'comment_version', 'version',
    'media_sha256', 'disposition', 'operations', 'moment', 'unmapped', 'change', 'keep', 'question'])
    || evidence.schema !== 'yap-studio/reply-evidence@1') return null;
  const strings = (value, keys) => keys.every((key) => typeof value[key] === 'string');
  const nullableString = (value) => value === null || typeof value === 'string';
  const seconds = (value) => Number.isFinite(value) && value >= 0;
  const frame = (value) => value === null || (Number.isSafeInteger(value) && value >= 0);
  if (!strings(evidence, ['batch', 'comment_version', 'version', 'media_sha256', 'disposition'])
    || !nullableString(evidence.unmapped)
    || !Array.isArray(evidence.operations)
    || !evidence.operations.every((item) => evidenceFields(item, ['id'], ['type'])
      && strings(item, ['id']) && (!Object.hasOwn(item, 'type') || strings(item, ['type'])))) return null;
  const {moment, change, keep, question} = evidence;
  if (moment !== null && (!evidenceFields(moment,
    ['status', 'start_s', 'end_s', 'start_frame', 'end_frame'])
    || !['mapped', 'moved', 'ambiguous', 'partial', 'removed', 'time'].includes(moment.status)
    || !seconds(moment.start_s) || !seconds(moment.end_s) || moment.end_s < moment.start_s
    || !frame(moment.start_frame) || !frame(moment.end_frame)
    || (moment.start_frame !== null && moment.end_frame !== null
      && moment.end_frame < moment.start_frame))) return null;
  const words = (value) => Array.isArray(value) && value.every((word) =>
    evidenceFields(word, ['id', 'segment', 'text']) && strings(word, ['id', 'segment', 'text']));
  if (change !== null && (!evidenceFields(change, ['before', 'after', 'differences'])
    || !words(change.before) || !words(change.after) || !Array.isArray(change.differences)
    || !change.differences.every((item) =>
      ['removed', 'words', 'order', 'timing', 'pause', 'layers'].includes(item)))) return null;
  if (keep !== null && (!evidenceFields(keep, ['status', 'detail'])
    || !['kept', 'broken', 'unmeasured'].includes(keep.status) || !strings(keep, ['detail']))) return null;
  if (question !== null && (!evidenceFields(question, ['question', 'answer', 'option'])
    || !strings(question, ['question']) || !nullableString(question.answer)
    || !nullableString(question.option))) return null;
  return evidence;
}

/** Name the reply's disposition and its series version, including legacy replies. */
export function replyHeading(reply) {
  const label = {changed: 'Changed', kept: 'Kept', declined: 'Declined',
    needs_clarification: 'Question'}[reply.status];
  return `${label} in v${reply.version}`;
}

/** Describe the measured destination using the reply version's own frame rate. */
export function replyMomentText(reply, versionFacts) {
  const moment = replyEvidence(reply)?.moment;
  if (!moment) return null;
  const rate = versionFacts?.cfr ? parseFps(versionFacts.fps) : null;
  const time = rate && Number.isInteger(moment.start_frame)
    ? frameTimeText(moment.start_frame, rate) : secondsText(moment.start_s);
  const version = `v${reply.version}`;
  return ({mapped: `See it in ${version} at ${time}`, moved: `See it in ${version} at ${time}`,
    removed: `Removed in ${version}, see the join at ${time}`,
    partial: `Partly removed in ${version}, see it at ${time}`,
    ambiguous: `See it in ${version} near ${time}`, time: `See ${version} at ${time}`})[moment.status];
}

/** Report measured keeps and the reasons for broken or unmeasured checks. */
export function keepCheckText(reply) {
  const keep = replyEvidence(reply)?.keep;
  if (!keep) return null;
  const result = keep.status === 'kept' ? 'kept'
    : `${keep.status === 'unmeasured' ? 'not measured' : 'broken'} (${keep.detail})`;
  return `Checked in v${reply.version}: ${result}`;
}

/** Summarize differences in recorded order, with removal covering changed words. */
export function changeText(reply) {
  const differences = replyEvidence(reply)?.change?.differences;
  if (!differences?.length) return null;
  const labels = {removed: 'Words removed', words: 'Words changed', order: 'Order changed',
    timing: 'Timing changed', pause: 'Pause changed', layers: 'Captions or graphics changed'};
  return differences.filter((item) => item !== 'words' || !differences.includes('removed'))
    .map((item) => labels[item]).join(', ');
}

/** Label a note's send state, retaining batch identity and failure details. */
export function sendStateLabel(send) {
  if (send?.state === 'sending') return 'Sending';
  if (send?.state === 'sent') return 'Sent';
  if (send?.state === 'failed') return `Failed: ${send.error?.message ?? 'Could not send this note.'}`;
  return 'Draft';
}

/** Describe the number of saved drafts ready for a new round. */
export function draftNotesText(comments, questions = []) {
  const count = comments.filter((item) => (item.send?.state ?? 'draft') === 'draft').length;
  const answers = questions.filter((item) => item.state === 'answered').length;
  return `${count} draft ${count === 1 ? 'note' : 'notes'}${answers ? `, ${answers} ${answers === 1 ? 'answer' : 'answers'}` : ''}`;
}

/** Select cards by version and id so repeated comment ids remain separate. */
export function questionsForComment(questions = [], version, id) {
  return questions.filter((item) => item.comment.version === version && item.comment.id === id);
}

/** Build the answer request using the saved project version rather than its series number. */
export function questionAnswerBody(question, answer, option = null) {
  const body = {batch: question.batch, comment_version: question.comment_version,
    comment_id: question.comment.id};
  if (option !== null) body.option = option;
  else body.answer = answer.trim();
  return body;
}

/** Explain a round's current progress or result. */
export function roundStatusText(round) {
  if (round.status === 'sending') return round.step || 'Sending notes';
  if (round.status === 'failed') return round.error?.message ?? 'Could not send these notes.';
  const count = round.comments.length;
  const answers = round.answers?.length ?? 0;
  return `Sent ${count} ${count === 1 ? 'note' : 'notes'}${answers ? ` and ${answers} ${answers === 1 ? 'answer' : 'answers'}` : ''}`;
}

/** Poll faster while any version has a round in progress. */
export function reviewPollInterval(rounds = [], offline = false) {
  return !offline && rounds.some((item) => item.status === 'sending') ? 1000 : 5000;
}

/** Keep Latest, Current and Approved separate when the current version is rolled back. */
export function versionBadges(number, series) {
  return [number === series.newest ? 'Latest' : null,
    number === series.current_version ? 'Current' : null,
    number === series.approved?.version ? 'Approved' : null].filter(Boolean);
}

/** Label a version's recorded check status without suggesting missing checks passed. */
export function reportStatusLabel(status) {
  return ({pass: 'Checks passed', warn: 'Checks with warnings', fail: 'Checks failed'})[status?.toLowerCase()] ?? 'Not checked';
}

/** Only a network failure goes offline; any HTTP answer proves the server is reachable. */
export function offlineState(offline, outcome) {
  if (outcome === 'network_error') return true;
  if (outcome === 'response') return false;
  return offline;
}

/** Explain the occurrence mapping without hiding ambiguity or deletion. */
export function mapStatusMessage(status, version) {
  return ({mapped: 'Same words', moved: 'Same words, from another part of the story',
    ambiguous: `Said more than once in v${version}`,
    removed: `Removed in v${version}; shown where it was cut`,
    time: 'No word map; matched by time'})[status] ?? 'Could not match this moment';
}

/** Set a version and optional token while preserving other query fields and fragments. */
export function versionUrl(path, version, token = null) {
  const hash = path.indexOf('#');
  const fragment = hash < 0 ? '' : path.slice(hash);
  const address = hash < 0 ? path : path.slice(0, hash);
  const query = address.indexOf('?');
  const base = query < 0 ? address : address.slice(0, query);
  const params = new URLSearchParams(query < 0 ? '' : address.slice(query + 1));
  if (version === null || version === undefined) params.delete('version');
  else params.set('version', String(version));
  if (token !== null) params.set('t', token);
  const suffix = params.toString();
  return `${base}${suffix ? `?${suffix}` : ''}${fragment}`;
}

/** Prefer the preceding version, or another version when the first is selected. */
export function defaultCompareVersion(current, versions) {
  const index = versions.findIndex((item) => item.number === current);
  return versions[index > 0 ? index - 1 : 1]?.number ?? null;
}

/** Parse a positive rational frame rate into integer numerator and denominator. */
export function parseFps(text) {
  if (typeof text !== 'string') return null;
  const match = /^(\d+)\/(\d+)$/.exec(text);
  if (!match) return null;
  const num = Number(match[1]);
  const den = Number(match[2]);
  if (!Number.isSafeInteger(num) || !Number.isSafeInteger(den) || num <= 0 || den <= 0) return null;
  return {num, den};
}

/** Format a frame rate the same way as the plain-text export. */
export function fpsLabel(fps) {
  return `${(fps.num / fps.den).toFixed(2).replace(/\.?0+$/, '')} fps`;
}

/** Map a player time to its containing frame, allowing a tiny boundary error. */
export function frameAt(t, fps) {
  return Math.max(0, Math.floor(t * fps.num / fps.den + 0.001));
}

/** Return a frame's exact start in seconds. */
export function frameStart(frame, fps) {
  return frame * fps.den / fps.num;
}

/** Return a stable seek position in the middle of a frame. */
export function frameMid(frame, fps) {
  return (frame + 0.5) * fps.den / fps.num;
}

/** Round a media duration to the count of complete frames. */
export function frameCount(durationS, fps) {
  return Math.max(1, Math.round(durationS * fps.num / fps.den));
}

/** Format an integer number of tenths as m:ss.s. */
export function tenthsText(tenths) {
  const minutes = Math.floor(tenths / 600);
  const seconds = Math.floor((tenths % 600) / 10);
  return `${minutes}:${String(seconds).padStart(2, '0')}.${tenths % 10}`;
}

/** Format the exact boundary of a frame, flooring to a tenth. */
export function frameTimeText(frame, fps) {
  return tenthsText(Math.floor(frame * fps.den * 10 / fps.num));
}

/** Format a player time, flooring to a tenth. */
export function secondsText(t) {
  return tenthsText(Math.floor(t * 10));
}

/** Format a point or range using frames when available. */
export function spanText(comment, fps) {
  const useFrames = comment.start_frame !== null && fps !== null;
  const start = useFrames
    ? frameTimeText(comment.start_frame, fps) : secondsText(comment.start_s);
  const end = useFrames
    ? (comment.end_frame === null ? null : frameTimeText(comment.end_frame, fps))
    : (comment.end_s === null ? null : secondsText(comment.end_s));
  return end === null ? start : `${start}-${end}`;
}

/** Normalize two inclusive frame marks to a half-open range, or a point. */
export function normalizeSelection(inFrame, outFrame) {
  const start_frame = Math.min(inFrame, outFrame);
  const last = Math.max(inFrame, outFrame);
  return {start_frame, end_frame: last === start_frame ? null : last + 1};
}

/** Normalize two player-time marks to a range, or a point. */
export function normalizeTimeSelection(inTime, outTime) {
  const start_s = Math.min(inTime, outTime);
  const end = Math.max(inTime, outTime);
  return {start_s, end_s: end === start_s ? null : end};
}

/** Assign each half-open range the lowest available nonoverlapping lane. */
export function assignLanes(ranges) {
  const laneEnds = [];
  return [...ranges].sort((a, b) => a.start - b.start).map((range) => {
    let lane = laneEnds.findIndex((end) => end <= range.start);
    if (lane < 0) lane = laneEnds.length;
    laneEnds[lane] = range.end;
    return {...range, lane};
  });
}

/** Return comments ordered by start, end, then numeric creation id. */
export function sortComments(comments) {
  function key(comment) {
    const start = comment.start_frame ?? comment.start_s;
    const end = comment.end_frame ?? comment.end_s ?? start;
    return [start, end, Number(comment.id.match(/\d+$/)[0])];
  }
  return [...comments].sort((a, b) => {
    const left = key(a);
    const right = key(b);
    return left[0] - right[0] || left[1] - right[1] || left[2] - right[2];
  });
}

/** A project's name from a saved version's file name, such as my-video-v3.mp4. */
export function projectTitle(name) {
  return String(name ?? '').replace(/\.[A-Za-z0-9]+$/, '').replace(/-v[0-9]+$/, '');
}

/** The Yap Studio home link that works from where this page was opened. */
export function homeUrl(home, hostname) {
  if (!home) return null;
  const local = ['127.0.0.1', 'localhost', '[::1]'].includes(hostname);
  return local ? (home.url ?? null) : (home.phone_url ?? home.url ?? null);
}

export function phoneLinkLabel(via) {
  return via === 'tailscale' ? 'Encrypted (Tailscale)' : via === 'wifi' ? 'Not encrypted (same Wi-Fi)' : '';
}

/** What happens after notes are sent, when Yap Studio home knows whether Claude is listening. */
export function sendNextText(series, home) {
  if (!home || !series?.rounds?.length) return null;
  const sent = series.rounds.filter((round) => round.status === 'sent');
  const latest = sent.at(-1);
  if (!latest || latest.version < series.newest) return null;
  if ((home.delivered ?? []).includes(latest.batch)) {
    return {state: 'listening', text: 'Claude has your notes and is making the next version. It shows up in Yap Studio when it is ready.'};
  }
  return home.listening
    ? {state: 'listening', text: 'Sent to Claude. It picks your notes up in a moment.'}
    : {state: 'waiting', text: "Your notes are saved. Claude Code isn't listening right now: open this folder in Claude Code and type continue."};
}

/** What the bottom of the player offers: approve a final, or ask for the final after a story cut. */
export function storyCutStep(series, selected, home) {
  const version = series?.versions?.find((item) => item.number === selected);
  const approveHint = 'Happy with it? Approving unlocks the download.';
  if (!series?.project || !home || version?.final !== false) {
    return {hideApprove: false, offerFinal: false, requested: false, hint: approveHint};
  }
  const newest = selected === series.newest;
  if (home.final_requested) {
    return {hideApprove: true, offerFinal: newest, requested: true,
      hint: 'Claude is adding captions, cards and music, then rendering and checking the final. It shows up in Yap Studio when it is ready.'};
  }
  return {hideApprove: true, offerFinal: newest, requested: false,
    hint: newest
      ? 'This is the story cut. Happy with the order? Claude adds captions, cards and music, then renders the final.'
      : 'This is an earlier story cut. Pick the newest version to go on.'};
}
