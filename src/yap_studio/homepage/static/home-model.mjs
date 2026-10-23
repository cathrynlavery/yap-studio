// Pure helpers for Yap Studio home: labels, progress and routing. No DOM here.

export const OFFERED_TYPES = ['talking_head', 'voiceover'];

export function connectionUpdate(connection, success, now = Date.now()) {
  return success ? {failures: 0, lastUpdated: now}
    : {...connection, failures: (connection.failures ?? 0) + 1};
}

export function connectionMessage(connection) {
  if (connection.failures < 2 || !connection.lastUpdated) return '';
  const when = new Date(connection.lastUpdated).toLocaleString([], {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit',
  });
  return `Lost contact with your Mac. Last updated ${when}. Trying again.`;
}

export function textHookProblem(text) {
  return text.trim().length > 80 ? 'Please use shorter text for the Text hook.' : '';
}

export const TYPE_COPY = {
  talking_head: {label: 'Talking head', description: 'You on camera, from one or more takes.',
    picture: 'presenter'},
  voiceover: {label: 'Voiceover', description: 'Your voice over your photos and clips.',
    picture: 'voice'},
};

export const NEW_STEPS = [
  {id: 'what', label: "What you're making"},
  {id: 'footage', label: 'Your footage'},
  {id: 'look', label: 'Your look'},
  {id: 'make', label: 'Make it'},
  {id: 'review', label: 'Watch and comment'},
];

const SOURCE_TEXT = {
  folder: 'your inbox or a folder', album: 'a Photos album', stock: 'stock clips you approve',
};

export function typeLabel(type) {
  return TYPE_COPY[type]?.label ?? 'Video';
}

export function brollText(sources) {
  if (!Array.isArray(sources)) return 'not chosen yet';
  const named = sources.filter((source) => SOURCE_TEXT[source]).map((source) => SOURCE_TEXT[source]);
  if (!named.length) return 'none';
  if (named.length === 1) return named[0];
  return `${named.slice(0, -1).join(', ')} and ${named.at(-1)}`;
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = String(total % 60).padStart(2, '0');
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${rest}` : `${minutes}:${rest}`;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes < 1000 * 1000) return `${Math.max(1, Math.round(bytes / 1000))} KB`;
  if (bytes < 1000 * 1000 * 1000) return `${Math.round(bytes / (1000 * 1000))} MB`;
  return `${(bytes / (1000 * 1000 * 1000)).toFixed(1)} GB`;
}

const VIDEO_SUFFIXES = ['.mp4', '.mov', '.m4v', '.mkv', '.webm'];

// A voiceover's inbox holds the voice memo; its clips come from a folder or album.
export function isVideoFile(name, type = '') {
  const lower = String(name ?? '').toLowerCase();
  return String(type).startsWith('video/') || VIDEO_SUFFIXES.some((suffix) => lower.endsWith(suffix));
}

export function formatWhen(iso, now = new Date()) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  const time = date.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  const day = (value) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const difference = Math.round((day(now) - day(date)) / 86400000);
  if (difference === 0) return `today ${time}`;
  if (difference === 1) return `yesterday ${time}`;
  return date.toLocaleDateString([], {month: 'short', day: 'numeric'});
}

export function plural(count, one, many = `${one}s`) {
  return `${count} ${count === 1 ? one : many}`;
}

export function takesText(takes) {
  if (!takes?.length) return 'none yet';
  return takes.length === 1 ? takes[0] : plural(takes.length, 'take');
}

const PHASES = {
  starting: {label: 'Starting', tone: 'working', title: 'Claude is setting up your video.'},
  transcribing: {label: 'Listening to your take', tone: 'working',
    title: 'Claude is listening to your take.'},
  story: {label: 'Choosing the story', tone: 'working', title: 'Claude is choosing the story.'},
  cutting: {label: 'Cutting the story', tone: 'working', title: 'Claude is cutting the story.'},
  cut_ready: {label: 'Story cut ready', tone: 'turn', title: 'Your story cut is ready.'},
  revising: {label: 'Working on your notes', tone: 'working',
    title: 'Claude is working on your notes.'},
  building: {label: 'Building the video', tone: 'working', title: 'Claude is building your video.'},
  final_ready: {label: 'Ready to watch', tone: 'turn', title: 'Your video is ready to watch.'},
  approved: {label: 'Done', tone: 'done', title: 'Your video is ready.'},
  stopped: {label: 'Stopped', tone: 'stopped', title: 'Claude stopped.'},
};

export function phaseInfo(phase) {
  return PHASES[phase] ?? PHASES.starting;
}

export function hasFinal(project) {
  return (project?.versions ?? []).some((version) => version.final);
}

const ROWS = [
  {key: 'copied', label: 'Copied your footage into a project', who: 'Claude'},
  {key: 'listen', label: 'Listening to your take', done: 'Listened to your take', who: 'Claude'},
  {key: 'story', label: 'Choosing the story', done: 'Chose the story', who: 'Claude'},
  {key: 'cut', label: 'Story cut ready to watch', done: 'You watched the story cut', who: 'You'},
  {key: 'build', label: 'Building your final video', done: 'Built your final video',
    who: 'Claude'},
  {key: 'final', label: 'Final video ready to watch', done: 'You watched the final', who: 'You'},
  {key: 'approve', label: 'Approve it', done: 'Approved', who: 'You'},
];

/** The rows of the progress list, with the current row taken from the project phase. */
export function progressRows(project) {
  const final = hasFinal(project);
  const phase = project?.phase ?? 'starting';
  let current = 0;
  let status = 'now';
  let sub = null;
  switch (phase) {
    case 'starting': current = 0; break;
    case 'transcribing': current = 1; break;
    case 'story': current = 2; break;
    case 'cutting': current = 2; sub = 'Cutting the story'; break;
    case 'cut_ready': current = 3; status = 'turn'; sub = 'Watch it and leave notes'; break;
    case 'revising': current = final ? 4 : 2; sub = 'Working on your notes'; break;
    case 'building': current = 4; sub = project?.step || null; break;
    case 'final_ready': current = 5; status = 'turn'; sub = 'Watch it, leave notes or approve'; break;
    case 'approved': current = ROWS.length; break;
    case 'stopped':
      current = final || (project?.stages ?? []).some((stage) =>
        ['captions', 'graphics', 'assemble'].includes(stage.name) && stage.state !== 'missing') ? 4 : 2;
      status = 'stopped';
      sub = project?.message || 'Open Claude Code to see what it needs.';
      break;
    default: current = 0;
  }
  return ROWS.map((row, index) => {
    if (row.key === 'build' && project?.shape === 'horizontal') {
      row = {...row, label: 'Building the horizontal final', done: 'Built the horizontal final'};
    }
    if (index < current) return {...row, label: row.done ?? row.label, status: 'done', sub: null};
    if (index === current) return {...row, status, sub};
    return {...row, status: 'next', sub: null};
  });
}

/** What to say about a request that has not become a project yet. */
export function requestInfo(request, listening) {
  const message = request?.ack?.message || null;
  if (request?.stalled) return {tone: 'stopped', title: 'Claude Code stopped before finishing.',
    lede: 'Send it again to pick up where it stopped.'};
  switch (request?.state) {
    case 'pending':
      return listening
        ? {tone: 'working', title: 'Sending this to Claude.', lede: 'Claude picks it up in a moment.'}
        : {tone: 'turn', title: 'Waiting for Claude Code.',
          lede: 'Your choices are saved. Claude starts as soon as Claude Code is open in this folder.'};
    case 'delivered':
      return {tone: 'working', title: 'Claude has it.', lede: 'It is copying your footage into a new project.'};
    case 'claimed':
      return {tone: 'working', title: 'Claude is starting your video.', lede: message ?? 'This takes a moment.'};
    case 'failed':
      return {tone: 'stopped', title: "Claude couldn't start this video.",
        lede: message ?? 'Open Claude Code to see what it needs.'};
    case 'cancelled':
      return {tone: 'stopped', title: 'Cancelled.', lede: 'This request was cancelled.'};
    case 'done':
      return {tone: 'done', title: 'Started.', lede: message ?? 'Opening your video.'};
    default:
      return {tone: 'working', title: 'Sending this to Claude.', lede: ''};
  }
}

/** A draft is ready for a step when every earlier step has what it needs. */
export function draftProblems(draft, state) {
  const problems = {};
  const type = draft?.video_type;
  if (!OFFERED_TYPES.includes(type)) problems.what = 'Choose what you are making.';
  const sources = Array.isArray(draft?.broll_sources) ? draft.broll_sources : [];
  if (draft?.shape === 'horizontal' && type !== 'talking_head') {
    problems.what = 'Horizontal voiceover is coming soon. Choose talking head or vertical.';
  }
  if (type === 'talking_head' && draft?.shape !== 'horizontal') {
    if (typeof draft?.has_broll !== 'boolean') problems.what ??= 'Say whether you have anything to show on top.';
    else if (draft.has_broll && !sources.length) problems.what ??= 'Pick where those clips and photos are.';
  }
  if (type === 'voiceover' && !sources.some((source) => source === 'folder' || source === 'album')) {
    problems.what ??= 'Pick where your photos and clips are: a folder or a Photos album.';
  }
  const names = new Set((state?.inbox?.takes ?? []).map((take) => take.name));
  const takes = (draft?.takes ?? []).filter((name) => names.has(name));
  const kinds = new Map((state?.inbox?.takes ?? []).map((take) => [take.name, take.kind]));
  if (type === 'talking_head' && !takes.some((name) => kinds.get(name) === 'take')) {
    problems.footage = 'Pick at least one take.';
  }
  const mismatch = shapeProblem(draft, state);
  if (mismatch) problems.footage ??= mismatch;
  return problems;
}

export function takeShape(take) {
  if (!(take?.width > 0 && take?.height > 0) || take.width === take.height) return null;
  return take.width > take.height ? 'horizontal' : 'vertical';
}

export function suggestedShape(draft, state) {
  const selected = (state?.inbox?.takes ?? []).filter((take) => draft.takes?.includes(take.name) && take.kind === 'take');
  const shapes = selected.map(takeShape);
  return shapes.length && shapes[0] && shapes.every((shape) => shape === shapes[0]) ? shapes[0] : null;
}

export function shapeProblem(draft, state) {
  const shape = draft?.shape ?? 'vertical';
  for (const take of state?.inbox?.takes ?? []) {
    if (!draft?.takes?.includes(take.name) || take.kind !== 'take') continue;
    const actual = takeShape(take);
    if (!actual && shape === 'horizontal') return `Cannot read a horizontal shape for ${take.name}. Replace it with a horizontal take.`;
    if (actual && actual !== shape) return `${take.name} is ${actual}, but this video is ${shape}. Choose ${actual} or replace it with a ${shape} take.`;
  }
  return '';
}

export function firstOpenStep(problems) {
  if (problems.what) return 'what';
  if (problems.footage) return 'footage';
  return 'look';
}

export function findProject(projects, name) {
  return projects.find((project) => project.name === name)
    ?? projects.find((project) => project.renamed_from?.includes(name));
}

/** Where a visit without a route should land. */
export function defaultRoute(state, welcomed) {
  const projects = state?.projects ?? [];
  const waiting = (state?.requests ?? []).find((request) =>
    request.stalled || ['pending', 'delivered', 'claimed'].includes(request.state) && !request.ack?.project
      && !(request.kind === 'final' && projects.some((project) => project.dir === request.payload?.project)));
  if (waiting) return `#request/${waiting.id}`;
  const active = projects.find((project) => !['approved'].includes(project.phase));
  if (active) return `#video/${encodeURIComponent(active.name)}`;
  if (projects.length) return '#videos';
  if (state?.first_run && !welcomed) return '#welcome';
  return '#new/what';
}

export function parseRoute(hash) {
  const value = (hash || '').replace(/^#/, '');
  const [head, ...rest] = value.split('/');
  const tail = rest.join('/');
  if (head === 'new' && (['shape', 'sources'].includes(tail) || NEW_STEPS.some((step) => step.id === tail)) && tail !== 'review') {
    return {view: 'new', step: tail};
  }
  if (head === 'video' && tail) return {view: 'video', name: decodeURIComponent(tail)};
  if (head === 'request' && tail) return {view: 'request', id: tail};
  if (head === 'videos') return {view: 'videos'};
  if (head === 'welcome') return {view: 'welcome'};
  return {view: 'default'};
}

/** Pick the link that works from where this page was opened. */
export function linkFor(urls, hostname) {
  if (!urls) return null;
  const local = ['127.0.0.1', 'localhost', '[::1]'].includes(hostname);
  return local ? (urls.url ?? null) : (urls.phone_url ?? urls.url ?? null);
}

export function phoneLinkLabel(via) {
  return via === 'tailscale' ? 'Encrypted (Tailscale)' : via === 'wifi' ? 'Not encrypted (same Wi-Fi)' : '';
}

/** Turn the creator's picks into the answers object the engine validates. */
export function answersPayload(picks) {
  const answers = {};
  if (picks.platforms) answers.platforms = [...picks.platforms];
  if (picks.cuts) answers.cuts = picks.cuts;
  if (picks.music) answers.music = picks.music;
  if (picks.never) {
    answers.never = {
      cover_face: picks.never.includes('cover_face'), emoji: picks.never.includes('emoji'),
      all_caps: picks.never.includes('all_caps'),
      other: (picks.other ?? '').split('\n').map((line) => line.trim()).filter(Boolean).slice(0, 10),
    };
  }
  if (typeof picks.about === 'string') answers.about = picks.about.replace(/\s+/g, ' ').trim().slice(0, 200);
  if (typeof picks.website === 'string') {
    const website = picks.website.trim();
    answers.brand = website ? {website, pick_in_gallery: false} : {website: null, pick_in_gallery: true};
  }
  if (typeof picks.fixes === 'string') {
    const fixes = {};
    for (const line of picks.fixes.split('\n')) {
      const [heard, correct] = line.split(/\s*(?:->|→|=)\s*/);
      if (heard?.trim() && correct?.trim()) fixes[heard.trim()] = correct.trim();
    }
    answers.word_fixes = fixes;
  }
  answers.finished_videos = {answer: picks.learn ? 'yes' : 'not_yet'};
  return answers;
}

/** One line that sums up saved answers. */
export function answersSummary(questions) {
  const byKey = Object.fromEntries((questions ?? []).map((question) => [question.answer_key, question]));
  const label = (key, value) => byKey[key]?.choices?.find((choice) => choice.value === value)?.label;
  const parts = [];
  const platforms = byKey.platforms?.current;
  if (Array.isArray(platforms) && platforms.length) parts.push(platforms.map((value) => label('platforms', value) ?? value).join(', '));
  if (byKey.cuts?.current) parts.push(`${label('cuts', byKey.cuts.current) ?? byKey.cuts.current} cuts`);
  if (byKey.music?.current) {
    parts.push(byKey.music.current === 'none' ? 'no music' : `${(label('music', byKey.music.current) ?? '').toLowerCase()} music`);
  }
  return parts.join(' · ');
}

/** The B-roll sources to send: none at all when a talking head has nothing on top. */
export function sourcesToSend(draft) {
  if (draft?.shape === 'horizontal') return [];
  if (draft?.video_type === 'talking_head' && draft.has_broll === false) return [];
  return (Array.isArray(draft?.broll_sources) ? draft.broll_sources : []).filter((source) => source !== 'none');
}

/** Start a draft from the saved default, so a returning creator sees their last answers. */
export function draftFromDefault(preset) {
  const type = OFFERED_TYPES.includes(preset?.video_type) ? preset.video_type : null;
  const sources = Array.isArray(preset?.broll_sources) ? preset.broll_sources.filter((source) => source !== 'none') : null;
  return {
    video_type: type,
    has_broll: type === 'talking_head' && sources ? sources.length > 0 : null,
    broll_sources: sources ?? [],
    takes: null, album: '',
  };
}
