import {
  NEW_STEPS, OFFERED_TYPES, TYPE_COPY, answersPayload, answersSummary, brollText, defaultRoute,
  draftFromDefault, draftProblems, findProject, firstOpenStep, formatBytes, formatDuration, formatWhen, hasFinal, isVideoFile, linkFor,
  parseRoute, phaseInfo, phoneLinkLabel, plural, progressRows, requestInfo, sourcesToSend, takesText, textHookProblem, typeLabel,
} from './home-model.mjs';

const token = new URLSearchParams(location.search).get('t') ?? '';
const $ = (id) => document.getElementById(id);
let state = null;
let pollTimer = null;
let busy = false;
const uploads = new Map();
let uploadSeq = 0;

// ---------- DOM helpers ----------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'text') el.textContent = value;
    else if (key === 'style') el.style.cssText = value;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const ICONS = {
  check: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.4l3 3 6-6.6" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  bigCheck: '<svg viewBox="0 0 26 26" aria-hidden="true"><path d="M6 13.6l4.6 4.6L20.4 8" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  mic: '<svg viewBox="0 0 18 18" aria-hidden="true"><rect x="6.2" y="2" width="5.6" height="9" rx="2.8" fill="#fff"/><path d="M3.8 8.6a5.2 5.2 0 0 0 10.4 0M9 13.8v2.4" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/></svg>',
  upload: '<svg viewBox="0 0 40 40" aria-hidden="true"><rect x="7" y="5" width="26" height="30" rx="5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M20 26V13.5M14.8 18.4 20 13.2l5.2 5.2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  wave: '<svg viewBox="0 0 26 26" aria-hidden="true"><path d="M4 13h1.5M8 9v8M11.5 6v14M15 10v6M18.5 7.5v11M22 12v2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
};

function icon(name) {
  const template = document.createElement('template');
  template.innerHTML = ICONS[name];
  return template.content.firstElementChild;
}

function toast(message, error = false) {
  const el = $('toast');
  el.textContent = message;
  el.classList.toggle('error', error);
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, error ? 7000 : 3200);
}

function copyButton(text, label = 'Copy') {
  const button = h('button', {type: 'button', class: 'btn', text: label});
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
    } catch {
      button.textContent = 'Select and copy it';
    }
    setTimeout(() => { button.textContent = label; }, 2400);
  });
  return button;
}

// ---------- API ----------

async function api(path, body, method = body === undefined ? 'GET' : 'POST') {
  const options = {method, headers: {}, credentials: 'same-origin'};
  if (method !== 'GET') {
    options.headers['X-Home-Token'] = token;
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body ?? {});
  }
  let response;
  try {
    response = await fetch(withToken(path), options);
  } catch {
    throw new Error('Yap Studio is not reachable. Check that it is still running on your Mac.');
  }
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) throw new Error(data?.error?.message ?? `Something went wrong (${response.status}).`);
  return data;
}

function withToken(path) {
  return token ? `${path}${path.includes('?') ? '&' : '?'}t=${encodeURIComponent(token)}` : path;
}

let signature = '';
let deferred = false;

function typing() {
  const active = document.activeElement;
  return active && $('main').contains(active) && ['INPUT', 'TEXTAREA'].includes(active.tagName)
    && !['checkbox', 'radio', 'file'].includes(active.type);
}

async function refresh() {
  try {
    const next = await api('/api/state');
    const nextSignature = JSON.stringify(next);
    state = next;
    $('loading')?.remove();
    if (nextSignature === signature && !uploads.size) return;
    signature = nextSignature;
    if (typing()) { deferred = true; return; }
    render();
  } catch (error) {
    if (!state) {
      $('main').replaceChildren(h('div', {class: 'banner error'}, h('strong', {text: 'Yap Studio could not load.'}),
        h('span', {text: error.message})));
    }
  }
}

function schedule() {
  clearTimeout(pollTimer);
  const fast = uploads.size || (state?.requests ?? []).some((item) => ['pending', 'delivered', 'claimed'].includes(item.state));
  pollTimer = setTimeout(async () => { await refresh(); schedule(); }, document.hidden ? 10000 : fast ? 1500 : 3000);
}

// ---------- Draft ----------

function draftKey() { return `yap-home-draft:${state?.workspace ?? ''}`; }

function loadDraft() {
  try {
    const saved = JSON.parse(localStorage.getItem(draftKey()) ?? 'null');
    if (saved && typeof saved === 'object') return saved;
  } catch { /* ignore */ }
  return draftFromDefault(state?.video_type?.default);
}

function saveDraft(draft) { localStorage.setItem(draftKey(), JSON.stringify(draft)); }
function clearDraft() { localStorage.removeItem(draftKey()); }

function welcomed() { return localStorage.getItem(`yap-home-welcomed:${state?.workspace ?? ''}`) === '1'; }
function markWelcomed() { localStorage.setItem(`yap-home-welcomed:${state?.workspace ?? ''}`, '1'); }

function unusedTakes() { return state?.inbox?.takes ?? []; }

function draftTakes(draft) {
  // Only files of the right kind count: video takes for a talking head, memos for a voiceover.
  const kind = draft.video_type === 'voiceover' ? 'memo' : 'take';
  const names = new Set(unusedTakes().filter((take) => take.kind === kind).map((take) => take.name));
  if (Array.isArray(draft.takes)) return draft.takes.filter((name) => names.has(name));
  const newest = unusedTakes().find((take) => take.kind === kind);
  return newest && draft.video_type !== 'voiceover' ? [newest.name] : [];
}

// ---------- Render ----------

let lastView = '';

function render() {
  if (!state) return;
  $('workspace-name').textContent = state.workspace ?? '';
  renderListener();
  let route = parseRoute(location.hash);
  if (route.view === 'default') {
    let target = defaultRoute(state, welcomed());
    if (target === '#new/what') {
      // Resume a video in progress at its first unfinished step.
      const draft = loadDraft();
      const problems = draftProblems({...draft, takes: draftTakes(draft)}, state);
      if (!problems.what && !problems.footage) target = state.style?.saved ? '#new/make' : '#new/look';
      else if (!problems.what && draft.takes) target = '#new/footage';
    }
    history.replaceState(null, '', `${location.pathname}${location.search}${target}`);
    route = parseRoute(location.hash);
  }
  const key = JSON.stringify(route);
  const main = $('main');
  const focusId = document.activeElement?.id;
  const scroll = window.scrollY;
  const view = {
    welcome: renderWelcome, new: renderNew, request: renderRequest, video: renderVideo, videos: renderVideos,
  }[route.view](route);
  $('shell').classList.toggle('solo', route.view === 'welcome');
  renderRail(route);
  if (key !== lastView) {
    view.classList.add('step');
    main.replaceChildren(view);
    main.focus({preventScroll: true});
    window.scrollTo(0, 0);
    lastView = key;
  } else {
    main.replaceChildren(view);
    if (focusId) document.getElementById(focusId)?.focus({preventScroll: true});
    window.scrollTo(0, scroll);
  }
}

function claudeWorking() {
  // A request Claude has picked up but not finished means it is busy, not gone.
  return (state?.requests ?? []).some((request) => !request.stalled && ['delivered', 'claimed'].includes(request.state));
}

function renderListener() {
  const el = $('listener');
  const mode = state.listening ? 'on' : claudeWorking() ? 'working' : 'off';
  el.replaceChildren(h('span', {class: 'dot', 'aria-hidden': 'true'}),
    {on: 'Claude is listening', working: 'Claude is working', off: "Claude Code isn't listening"}[mode]);
  el.className = `listener ${mode}`;
  const phoneLink = $('phone-link');
  phoneLink.hidden = !state.phone;
  if (state.phone) {
    phoneLink.href = state.phone.url;
    phoneLink.textContent = `Open on your phone: ${phoneLinkLabel(state.phone.via)}`;
  }
}

function notListening() {
  return h('div', {class: 'banner turn'},
    h('strong', {text: "Claude Code isn't listening right now."}),
    h('span', {text: 'Your choices are saved. Open this folder in Claude Code and type this, so Claude picks up where you are:'}),
    h('div', {class: 'row'}, h('code', {text: 'continue'}), copyButton('continue')));
}

function navigate(hash) {
  if (location.hash === hash) render();
  else location.hash = hash;
}

const openFolds = new Set();

function fold(id, summary, body) {
  const el = h('details', {class: 'fold', id, open: openFolds.has(id)}, summary, body);
  el.addEventListener('toggle', () => { if (el.open) openFolds.add(id); else openFolds.delete(id); });
  return el;
}

document.addEventListener('focusout', () => {
  setTimeout(() => { if (deferred && !typing()) { deferred = false; render(); } }, 0);
});

// ---------- Rail ----------

function renderRail(route) {
  const rail = $('rail');
  const blocks = [];
  if (route.view === 'new') {
    const draft = loadDraft();
    const problems = draftProblems({...draft, takes: draftTakes(draft)}, state);
    const values = {
      what: draft.video_type ? typeLabel(draft.video_type) : 'Not chosen',
      footage: footageValue(draft),
      look: state.style?.saved ? state.style.name : 'Not chosen',
      make: 'Not started', review: '',
    };
    const reachable = {what: true, footage: !problems.what, look: !problems.what && !problems.footage,
      make: !problems.what && !problems.footage, review: false};
    const order = NEW_STEPS.map((step) => step.id);
    const index = order.indexOf(route.step);
    blocks.push(h('div', {}, h('h2', {text: 'New video'}), h('ol', {class: 'rail-steps'},
      NEW_STEPS.map((step, position) => {
        const done = position < index && !problems[step.id];
        const status = position === index ? 'now' : done ? 'done' : '';
        return h('li', {}, railStep(position + 1, step.label, values[step.id], status,
          reachable[step.id] && step.id !== 'review' ? `#new/${step.id}` : null, position === index));
      }))));
  } else if (route.view === 'video') {
    const project = shown(state.projects.find((item) => item.name === route.name));
    if (project) {
      const rows = progressRows(project);
      const turn = rows.find((row) => row.status === 'turn');
      const approved = project.phase === 'approved';
      const making = !approved && !turn;
      blocks.push(h('div', {}, h('h2', {text: 'This video'}), h('ol', {class: 'rail-steps'}, [
        railStep(1, NEW_STEPS[0].label, typeLabel(project.video?.type), 'done'),
        railStep(2, NEW_STEPS[1].label, takesText(project.takes), 'done'),
        railStep(3, NEW_STEPS[2].label, state.style?.saved ? state.style.name : 'Default look', 'done'),
        railStep(4, NEW_STEPS[3].label, making ? phaseInfo(project.phase).label : 'Done',
          making ? 'now' : 'done', null, making),
        railStep(5, NEW_STEPS[4].label, approved ? 'Approved' : turn ? 'Your turn' : 'Next',
          approved ? 'done' : turn ? 'turn' : '', null, Boolean(turn)),
      ].map((step) => h('li', {}, step)))));
    }
  }
  if (state.projects.length) {
    blocks.push(h('div', {class: 'rail-block-videos'}, h('h2', {text: 'Your videos'}),
      h('div', {class: 'rail-videos'}, state.projects.slice(0, 6).map(shown).map((project) => {
        const info = phaseInfo(project.phase);
        return h('a', {class: 'rail-video', href: `#video/${encodeURIComponent(project.name)}`,
          'aria-current': route.view === 'video' && route.name === project.name ? 'page' : null},
        h('span', {class: 'name', text: project.title || project.name}), h('span', {class: `phase ${info.tone}`, text: info.label}));
      })),
      h('a', {class: 'rail-new', href: '#new/what', onclick: startNew, text: '+ New video'})));
  }
  rail.replaceChildren(...blocks);
  rail.hidden = !blocks.length;
}

function railStep(number, label, value, status, href = null, current = false) {
  const children = [h('span', {class: 'num', text: status === 'done' ? '' : String(number)},
    status === 'done' ? icon('check') : null),
  h('span', {}, h('span', {class: 'label', text: label}), value ? h('span', {class: 'value', text: value}) : null)];
  const attrs = {class: `rail-step ${status}`, 'aria-current': current ? 'step' : null};
  return href ? h('a', {...attrs, href}, children) : h('div', {...attrs, 'aria-disabled': 'true'}, children);
}

function footageValue(draft) {
  if (!draft.video_type) return 'Not chosen';
  const takes = draftTakes(draft);
  if (draft.video_type === 'voiceover') {
    if (takes.length) return takesText(takes);
    return draft.album ? `Album: ${draft.album}` : 'Photos and clips';
  }
  return takes.length ? takesText(takes) : 'Not chosen';
}

function startNew(event) {
  event?.preventDefault();
  clearDraft();
  navigate('#new/what');
}

// ---------- Welcome ----------

function renderWelcome() {
  const steps = NEW_STEPS.map((step, index) => h('li', {}, h('b', {text: String(index + 1)}), step.label,
    step.id === 'make' ? h('small', {text: 'Claude edits'}) : h('span')));
  return h('section', {class: 'welcome'},
    h('h1', {class: 'title wide', text: "You're set up. Let's make your first video."}),
    h('p', {class: 'lede', text: 'You pick what to make, your footage and your look. Claude picks the story from your own words, cuts it and adds captions, cards and music. You watch it, leave comments, and the next version follows.'}),
    h('ol', {class: 'welcome-steps'}, steps),
    h('p', {class: 'note', text: 'Your video files stay on your computer. Still frames from pinned notes and contact sheets of your clips go to Claude (Anthropic), along with the transcript, your notes and the project files Claude reads, as part of your Claude Code session. ElevenLabs, WaveSpeed and Pexels get only what they need, and only if you turn them on.'}),
    h('div', {class: 'actions'},
      h('button', {type: 'button', class: 'btn primary big', onclick: () => { markWelcomed(); clearDraft(); navigate('#new/what'); },
        text: 'Start my first video'})));
}

// ---------- New video steps ----------

function stepHeader(stepId, title, lede) {
  const index = NEW_STEPS.findIndex((step) => step.id === stepId);
  return [h('p', {class: 'step-count', text: `Step ${index + 1} of ${NEW_STEPS.length}`}),
    h('h1', {class: 'title', text: title}), lede ? h('p', {class: 'lede', text: lede}) : null];
}

function renderNew(route) {
  const draft = loadDraft();
  const problems = draftProblems({...draft, takes: draftTakes(draft)}, state);
  const order = ['what', 'footage', 'look', 'make'];
  const earliest = firstOpenStep(problems);
  if (order.indexOf(route.step) > order.indexOf(earliest) && earliest !== 'look') {
    queueMicrotask(() => navigate(`#new/${earliest}`));
  }
  return {what: stepWhat, footage: stepFootage, look: stepLook, make: stepMake}[route.step](draft, problems);
}

function stepWhat(draft, problems) {
  const question = state.video_type.questions.find((item) => item.id === 'video_type');
  const broll = state.video_type.questions.find((item) => item.id === 'broll_sources');
  const tiles = h('div', {class: 'tiles', role: 'radiogroup', 'aria-label': "What you're making"},
    question.choices.filter((choice) => OFFERED_TYPES.includes(choice.value)).map((choice) => {
      const copy = TYPE_COPY[choice.value];
      const selected = draft.video_type === choice.value;
      return h('button', {type: 'button', class: 'tile', role: 'radio', 'aria-checked': String(selected),
        onclick: () => {
          if (draft.video_type === choice.value) return;
          saveDraft({...draft, video_type: choice.value, has_broll: null, broll_sources: [], takes: null});
          render();
        }},
      h('div', {class: `pic ${copy.picture}`}, copy.picture === 'voice' ? h('span', {class: 'badge'}, icon('mic')) : null),
      h('span', {class: 'check'}, icon('check')),
      h('div', {class: 't'}, h('strong', {text: copy.label}), h('span', {text: copy.description})));
    }));
  const content = [...stepHeader('what', 'What are you making?', 'Each video can be a different kind. Your answer is remembered for next time.'),
    h('div', {class: 'section'}, tiles)];
  if (draft.video_type) {
    const sourceChoices = broll.choices.filter((choice) => choice.value !== 'none'
      && (!choice.only_for || choice.only_for.includes(draft.video_type)));
    const sourceRows = () => h('div', {class: 'checks'}, sourceChoices.map((choice) => {
      const checked = (draft.broll_sources ?? []).includes(choice.value);
      return h('label', {class: 'check-row'},
        h('input', {type: 'checkbox', checked, id: `source-${choice.value}`, onchange: (event) => {
          const next = new Set(loadDraft().broll_sources ?? []);
          if (event.target.checked) next.add(choice.value); else next.delete(choice.value);
          saveDraft({...loadDraft(), broll_sources: [...next]}); render();
        }}),
        h('span', {}, h('strong', {text: choice.label}), h('span', {text: choice.description}),
          choice.value === 'stock' && checked ? h('span', {class: 'extra',
            text: 'Stock clips need a free Pexels key. Claude shows you a pick sheet and uses only the clips you choose. Nothing is downloaded before you pick.'}) : null,
          choice.value === 'album' && checked ? h('span', {class: 'extra',
            text: 'Claude exports the album with osxphotos. Your Photos library is only read, never changed.'}) : null));
    }));
    if (draft.video_type === 'talking_head') {
      const answer = (value) => h('label', {class: 'check-row'},
        h('input', {type: 'radio', name: 'has-broll', checked: draft.has_broll === value, id: `has-broll-${value}`, onchange: () => {
          saveDraft({...loadDraft(), has_broll: value, broll_sources: value ? (loadDraft().broll_sources ?? []) : []}); render();
        }}),
        h('span', {}, h('strong', {text: value ? 'Yes, I have some' : 'No, just me talking'}),
          h('span', {text: value ? 'Clips, photos or screenshots to show over you or between your shots.' : 'Claude edits your take with captions and cards, and nothing else on top.'})));
      content.push(h('div', {class: 'section'},
        h('h2', {text: 'Anything to show on top of your video?'}),
        h('p', {class: 'hint', text: 'Clips, photos or screenshots that appear over you or between your shots. Editors call this B-roll.'}),
        h('div', {class: 'checks', role: 'radiogroup', 'aria-label': 'Anything to show on top of your video?'}, answer(true), answer(false))));
      if (draft.has_broll) {
        content.push(h('div', {class: 'section'}, h('h2', {text: 'Where are they?'}),
          h('p', {class: 'hint', text: 'Pick everything Claude may use. It uses nothing else.'}), sourceRows()));
      }
    } else {
      content.push(h('div', {class: 'section'}, h('h2', {text: 'Where are your photos and clips?'}),
        h('p', {class: 'hint', text: 'Pick everything Claude may use. It uses nothing else.'}), sourceRows()));
    }
  }
  content.push(h('div', {class: 'actions'},
    h('span', {class: 'spacer'}),
    h('button', {type: 'button', class: 'btn primary big', disabled: Boolean(problems.what), onclick: async (event) => {
      event.target.disabled = true;
      try {
        await api('/api/video-type', {video_type: draft.video_type, broll_sources: sourcesToSend(draft)});
        navigate('#new/footage');
      } catch (error) { toast(error.message, true); event.target.disabled = false; }
    }, text: 'Next'})),
  problems.what && draft.video_type ? h('p', {class: 'note', text: problems.what}) : null);
  return h('section', {}, content);
}

function stepFootage(draft, problems) {
  const voice = draft.video_type === 'voiceover';
  const kind = voice ? 'memo' : 'take';
  const takes = unusedTakes().filter((take) => take.kind === kind);
  const chosen = new Set(draftTakes(draft));
  const content = [...stepHeader('footage', voice ? 'Your photos, clips and voice' : 'Your footage',
    voice ? 'Claude tells the story in your voice over your own photos and clips.' : 'Add the takes for this video. Yap Studio copies them into a new project. Your originals never change.')];
  if (voice) {
    if (sourcesToSend(draft).includes('album')) {
      content.push(h('div', {class: 'section'}, h('div', {class: 'field'},
        h('label', {for: 'album', text: 'Photos album name'}),
        h('input', {id: 'album', type: 'text', value: draft.album ?? '', maxlength: '200', autocomplete: 'off',
          placeholder: 'For example: Kitchen rebuild', oninput: (event) => saveDraft({...loadDraft(), album: event.target.value})}),
        h('span', {class: 'sub', text: 'Leave it empty and Claude asks you which album.'}))));
    }
    if (sourcesToSend(draft).includes('folder')) {
      content.push(h('p', {class: 'note', text: 'Clips and photos from a folder: Claude asks you which folder when it starts.'}));
    }
  }
  content.push(h('div', {class: 'section'},
    h('h2', {text: voice ? 'Your voice' : 'Add footage'}),
    h('p', {class: 'hint', text: voice
      ? 'Already recorded it? Add the memo here. If not, Claude writes the script from your photos first, then you record it with Voice Memos.'
      : 'Film upright (9:16). One take or several.'}),
    dropZone(voice), uploadList()));
  content.push(h('div', {class: 'section'},
    h('h2', {text: voice ? 'Memos in your inbox' : 'Takes in your inbox'}),
    takes.length ? h('div', {class: 'takes'}, takes.map((take) => takeRow(take, chosen.has(take.name), (checked) => {
      const next = new Set(draftTakes(loadDraft()));
      if (checked) next.add(take.name); else next.delete(take.name);
      saveDraft({...loadDraft(), takes: [...next]}); render();
    })))
      : h('div', {class: 'empty'}, h('strong', {text: voice ? 'No memos yet' : 'No takes yet'}),
        voice ? 'That is fine: Claude writes the script first.' : 'Add a take above. It shows up here as soon as it is copied.')));
  content.push(h('div', {class: 'actions'},
    h('a', {class: 'link', href: '#new/what', text: 'Back'}), h('span', {class: 'spacer'}),
    h('button', {type: 'button', class: 'btn primary big', disabled: Boolean(problems.footage), onclick: () => {
      saveDraft({...loadDraft(), takes: draftTakes(loadDraft())});
      navigate('#new/look');
    }, text: 'Next'})));
  return h('section', {}, content);
}

function takeRow(take, checked, onchange) {
  const id = `take-${take.name.replace(/[^a-z0-9]+/gi, '-')}`;
  const meta = [formatDuration(take.duration_s), formatWhen(take.modified), formatBytes(take.bytes)].filter(Boolean).join(' · ');
  const thumb = take.thumb
    ? h('span', {class: 'thumb'}, h('img', {src: withToken(take.thumb), alt: '', loading: 'lazy', width: '56', height: '100'}))
    : h('span', {class: 'thumb'}, icon('wave'));
  return h('label', {class: 'take', for: id},
    h('input', {type: 'checkbox', id, checked, onchange: (event) => onchange(event.target.checked)}),
    thumb, h('span', {}, h('strong', {text: take.name}), h('span', {class: 'meta', text: meta})));
}

function dropZone(voice) {
  const input = h('input', {type: 'file', id: 'file-input', multiple: true, class: 'sr-only', tabindex: '-1',
    accept: voice ? 'audio/*,.m4a,.wav,.mp3,.aac,.caf,.aif,.aiff,.flac' : 'video/*,.mov,.mp4,.m4v,.mkv,.webm',
    onchange: (event) => { [...event.target.files].forEach(upload); event.target.value = ''; }});
  // On a phone there is no Finder to drag from; the picker reads the phone's own videos.
  const touch = window.matchMedia?.('(pointer: coarse)').matches;
  const zone = h('div', {class: 'drop'},
    icon('upload'),
    h('strong', {text: touch ? (voice ? 'Add a voice memo from this phone' : 'Add videos from this phone') : 'Drag files here from Finder'}),
    h('p', {text: touch ? 'They are copied to the inbox on your Mac.' : voice ? 'Voice memos and audio files' : 'Video files from your phone or camera'}),
    h('button', {type: 'button', class: 'btn', id: 'add-footage', onclick: () => input.click(), text: touch ? (voice ? 'Choose a memo' : 'Choose videos') : 'Choose files…'}),
    input);
  zone.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (event) => {
    event.preventDefault(); zone.classList.remove('over');
    [...(event.dataTransfer?.files ?? [])].forEach(upload);
  });
  return zone;
}

function uploadList() {
  if (!uploads.size) return null;
  return h('div', {class: 'uploads', 'aria-live': 'polite'}, [...uploads.values()].map((item) => {
    const percent = item.total ? Math.floor((item.sent / item.total) * 100) : 0;
    return h('div', {class: `upload${item.error ? ' failed' : ''}`},
      h('span', {text: item.error ? `${item.name}: ${item.error}` : item.queued ? `Waiting to copy ${item.name}` : `Copying ${item.name}`}),
      item.error
        ? h('button', {type: 'button', class: 'link', onclick: () => { uploads.delete(item.id); render(); }, text: 'Dismiss'})
        : h('span', {class: 'pct'}, `${percent}% of ${formatBytes(item.total)} `,
          h('button', {type: 'button', class: 'link', onclick: () => {
            if (item.xhr) item.xhr.abort(); else { uploads.delete(item.id); render(); }
          }, text: 'Cancel'})),
      item.error ? null : h('span', {class: 'bar'}, h('i', {style: `width:${percent}%`})));
  }));
}

function upload(file) {
  const id = ++uploadSeq;
  if (loadDraft().video_type === 'voiceover' && isVideoFile(file.name, file.type)) {
    uploads.set(id, {id, name: file.name, total: file.size, sent: 0, xhr: null, file, queued: false,
      error: 'This is a video. Add your voice memo here; Claude asks which folder has your clips.'});
    render();
    return;
  }
  uploads.set(id, {id, name: file.name, total: file.size, sent: 0, xhr: null, error: null, file, queued: true});
  pumpUploads();
  render();
  schedule();
}

function pumpUploads() {
  // The server copies one file at a time, so send them in order.
  if ([...uploads.values()].some((item) => item.xhr && !item.error)) return;
  const item = [...uploads.values()].find((entry) => entry.queued);
  if (item) startUpload(item);
}

function startUpload(item) {
  const {id, file} = item;
  const xhr = new XMLHttpRequest();
  item.xhr = xhr;
  item.queued = false;
  xhr.open('POST', withToken('/api/inbox'));
  xhr.setRequestHeader('X-Home-Token', token);
  xhr.setRequestHeader('X-File-Name', encodeURIComponent(file.name));
  xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  let last = 0;
  xhr.upload.onprogress = (event) => {
    item.sent = event.loaded;
    if (Date.now() - last > 250) { last = Date.now(); render(); }
  };
  xhr.onload = async () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      uploads.delete(id);
      let added = null;
      try { added = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      const draft = loadDraft();
      const name = added?.name;
      if (name) saveDraft({...draft, takes: [...new Set([...draftTakes(draft), name])]});
      toast(`${name ?? file.name} is in your inbox.`);
    } else {
      let message = 'Could not copy this file.';
      try { message = JSON.parse(xhr.responseText)?.error?.message ?? message; } catch { /* ignore */ }
      item.error = message;
    }
    await refresh();
  };
  xhr.onerror = () => { item.error = 'The copy stopped. Check that Yap Studio is still running.'; render(); pumpUploads(); };
  xhr.onabort = () => { uploads.delete(id); render(); pumpUploads(); };
  xhr.onloadend = () => pumpUploads();
  xhr.send(file);
}

function stepLook(draft) {
  const content = [...stepHeader('look', 'Your look', 'Captions, titles, cards, colors and music, picked once and used on every video.')];
  if (!state.answers.saved) content.push(answersCard());
  else {
    content.push(h('div', {class: 'section'}, h('h2', {text: 'About your videos'}),
      h('p', {class: 'hint'}, answersSummary(state.answers.questions) || 'Saved.', ' ',
        h('button', {type: 'button', class: 'link', onclick: () => { editAnswers = true; render(); }, text: 'Change'})),
      editAnswers ? answersCard() : null));
  }
  const saved = state.style?.saved;
  content.push(h('div', {class: 'section'}, h('h2', {text: 'Look'}),
    h('p', {class: 'hint', text: 'Pick a look on a still from your own take. You can fine-tune every part, or save a look as it is.'}),
    h('div', {class: 'look'}, h('span', {class: 'frame', 'aria-hidden': 'true'}),
      h('span', {class: 'info'}, h('strong', {text: saved ? state.style.name : 'No look yet'}),
        h('span', {text: saved ? `Saved ${formatWhen(state.style.updated_at)}` : 'Claude uses the default look until you pick one.'})),
      h('button', {type: 'button', class: `btn${saved ? '' : ' primary'}`, id: 'open-style', onclick: openStylePage,
        text: saved ? 'Change look' : 'Choose a look'}))));
  content.push(h('div', {class: 'actions'},
    h('a', {class: 'link', href: '#new/footage', text: 'Back'}), h('span', {class: 'spacer'}),
    h('button', {type: 'button', class: `btn big${saved ? ' primary' : ''}`, onclick: async (event) => {
      event.target.disabled = true;
      try { await ensureAnswersSaved(); navigate('#new/make'); }
      catch (error) { toast(error.message, true); event.target.disabled = false; }
    }, text: saved ? 'Next' : 'Skip for now'})));
  return h('section', {}, content);
}

let editAnswers = false;
let answerPicks = null;

function answersCard() {
  const questions = Object.fromEntries(state.answers.questions.map((question) => [question.id, question]));
  if (!answerPicks) {
    const pick = (question) => question.current ?? (question.kind === 'multi' || question.kind === 'toggles'
      ? question.choices.filter((choice) => choice.recommended).map((choice) => choice.value)
      : question.choices.find((choice) => choice.recommended)?.value);
    const never = questions.never?.current;
    answerPicks = {
      platforms: pick(questions.platforms) ?? [], cuts: pick(questions.cuts), music: pick(questions.music),
      never: never && typeof never === 'object' ? ['cover_face', 'emoji', 'all_caps'].filter((key) => never[key]) : pick(questions.never) ?? [],
      other: Array.isArray(never?.other) ? never.other.join('\n') : '',
      about: typeof questions.about?.current === 'string' ? questions.about.current : '',
      website: questions.brand?.current?.website ?? '',
      fixes: questions.word_fixes?.current && typeof questions.word_fixes.current === 'object'
        ? Object.entries(questions.word_fixes.current).map(([heard, correct]) => `${heard} -> ${correct}`).join('\n') : '',
    };
  }
  const chips = (question, multi) => h('div', {class: 'question'},
    h('p', {text: question.prompt.replace(/ Type any additional rules\.$/, '')}),
    h('div', {class: 'chips', role: 'group', 'aria-label': question.prompt}, question.choices.map((choice) => {
      const key = question.id;
      const value = answerPicks[key];
      const on = multi ? (value ?? []).includes(choice.value) : value === choice.value;
      return h('button', {type: 'button', class: 'chip', 'aria-pressed': String(on), onclick: () => {
        if (multi) {
          const next = new Set(answerPicks[key] ?? []);
          if (next.has(choice.value)) next.delete(choice.value); else next.add(choice.value);
          answerPicks[key] = [...next];
        } else answerPicks[key] = choice.value;
        render();
      }, text: choice.label});
    })));
  const field = (id, label, value, attrs = {}, sub = null, area = false) => h('div', {class: 'field'},
    h('label', {for: id, text: label}),
    h(area ? 'textarea' : 'input', {id, ...(area ? {} : {type: 'text', value}), ...attrs,
      oninput: (event) => { answerPicks[attrs['data-key']] = event.target.value; }}, area ? value : null),
    sub ? h('span', {class: 'sub', text: sub}) : null);
  return h('div', {class: 'section'},
    state.answers.saved ? null : h('h2', {text: 'About your videos'}),
    state.answers.saved ? null : h('p', {class: 'hint', text: 'Four quick choices. The usual answers are already picked; change any of them.'}),
    h('div', {}, chips(questions.platforms, true), chips(questions.cuts, false), chips(questions.music, false),
      chips(questions.never, true)),
    fold('answers-more', h('summary', {}, 'More about you ', h('span', {text: 'Website colors, what your videos are about, words the transcript gets wrong'})),
      h('div', {class: 'fold-body'},
        field('answer-website', 'Your website', answerPicks.website, {'data-key': 'website', type: 'url', inputmode: 'url', placeholder: 'yourwebsite.com', autocomplete: 'url'},
          'Yap Studio pulls an accent color from it.'),
        field('answer-about', 'What your videos are about, and who they are for', answerPicks.about, {'data-key': 'about', maxlength: '200'},
          'Claude uses this for titles and what to keep.'),
        field('answer-fixes', 'Words the transcript gets wrong', answerPicks.fixes, {'data-key': 'fixes', placeholder: 'yap studio -> Yap Studio'},
          'One per line: as heard -> correct.', true),
        field('answer-other', 'Anything else Claude should never do', answerPicks.other, {'data-key': 'other'}, 'One rule per line.', true),
        h('div', {class: 'field'}, h('label', {text: 'Videos you have already made'}),
          h('span', {class: 'sub', text: 'Claude can learn your style from finished videos. It asks you which folder in Claude Code.'}),
          h('div', {}, h('button', {type: 'button', class: 'btn', onclick: requestLearn, text: 'Ask Claude to learn from them'}))))),
    state.answers.saved
      ? h('div', {class: 'actions inline'},
        h('button', {type: 'button', class: 'btn', id: 'save-answers', onclick: saveAnswers, text: 'Save these answers'}))
      : h('p', {class: 'note', text: 'Saved when you continue.'}));
}

async function ensureAnswersSaved() {
  // First run: the answers card has no Save button; continuing saves the picks shown.
  if (state.answers.saved || !answerPicks) return;
  await api('/api/answers', {answers: answersPayload(answerPicks)});
  answerPicks = null;
  await refresh();
}

async function saveAnswers(event) {
  event.target.disabled = true;
  try {
    await api('/api/answers', {answers: answersPayload(answerPicks)});
    editAnswers = false;
    answerPicks = null;
    toast('Saved. Each new video starts from these answers.');
    await refresh();
  } catch (error) { toast(error.message, true); event.target.disabled = false; }
}

async function requestLearn(event) {
  event.target.disabled = true;
  try {
    await api('/api/requests', {kind: 'learn'});
    toast(state.listening ? 'Claude will ask you for the folder in Claude Code.' : 'Saved. Claude asks for the folder when Claude Code is open.');
    await refresh();
  } catch (error) { toast(error.message, true); event.target.disabled = false; }
}

async function openStylePage(event) {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Opening the style page…';
  try {
    await ensureAnswersSaved();
    const page = await api('/api/style-page', {});
    const url = linkFor({url: page.url, phone_url: page.phone?.url}, location.hostname);
    location.href = url;
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
    button.textContent = state.style?.saved ? 'Change look' : 'Choose a look';
  }
}

function stepMake(draft, problems) {
  const takes = draftTakes(draft);
  const voice = draft.video_type === 'voiceover';
  const hasHook = state.style?.hook_style === 'text_hook';
  let hookField = null;
  if (hasHook) {
    hookField = $('text-hook-field').content.firstElementChild.cloneNode(true);
    const input = hookField.querySelector('textarea');
    input.value = draft.text_hook ?? '';
    input.addEventListener('input', () => {
      draft.text_hook = input.value;
      saveDraft(draft);
      const error = hookField.querySelector('[role="alert"]');
      error.textContent = textHookProblem(input.value);
      error.hidden = !error.textContent;
      input.setCustomValidity(error.textContent);
    });
  }
  const rows = [
    ["What", typeLabel(draft.video_type), '#new/what'],
    [voice ? 'Voice' : 'Footage', takes.length ? takes.join(', ') : voice ? 'Claude writes the script first' : 'Not chosen', '#new/footage'],
    voice ? ['Pictures', draft.album ? `Photos album: ${draft.album}` : brollText(sourcesToSend(draft)), '#new/what'] : ['On top', draft.has_broll ? brollText(sourcesToSend(draft)) : 'Nothing, just you talking', '#new/what'],
    ['Look', state.style?.saved ? state.style.name : 'Default look', '#new/look'],
  ];
  const next = voice
    ? ['Claude looks through your photos and clips and writes a short script for you to approve.',
      'You record it with Voice Memos and add the memo here.',
      'Claude cuts your voice and lays your shots over it. You watch the story first, then the final.']
    : ['Claude listens to your take, picks the story from your own words and cuts it.',
      'You watch the story cut first and leave notes.',
      'Claude adds captions, cards, B-roll and music, renders the final and checks it. You watch it again and approve.'];
  return h('section', {},
    ...stepHeader('make', 'Make it', null),
    h('dl', {class: 'summary'}, rows.map(([label, value, href]) => h('div', {}, h('dt', {text: label}), h('dd', {text: value}),
      h('a', {href, text: 'Change'})))),
    hookField,
    h('div', {class: 'section'}, h('h2', {text: 'What happens next'}), h('ol', {class: 'next-list'}, next.map((line) => h('li', {text: line})))),
    h('p', {class: 'note', text: 'Rendering the final can take a while for a long video. You can close this page; it picks up where things are when you come back. Generated music uses your own ElevenLabs credits, and Claude asks before it spends any.'}),
    state.listening || claudeWorking() ? null : notListening(),
    h('div', {class: 'actions'},
      h('a', {class: 'link', href: '#new/look', text: 'Back'}), h('span', {class: 'spacer'}),
      h('button', {type: 'button', class: 'btn primary big', id: 'make', disabled: Boolean(problems.what || problems.footage) || busy,
        onclick: async (event) => {
          if (hasHook && textHookProblem(draft.text_hook ?? '')) {
            toast(textHookProblem(draft.text_hook), true);
            return;
          }
          busy = true; event.target.disabled = true;
          try {
            const request = await api('/api/requests', {kind: 'edit', video_type: draft.video_type,
              broll_sources: sourcesToSend(draft), takes, album: voice && draft.album?.trim() ? draft.album.trim() : null,
              ...(hasHook ? {text_hook: draft.text_hook ?? ''} : {})});
            clearDraft();
            await refresh();
            navigate(`#request/${request.id}`);
          } catch (error) { toast(error.message, true); event.target.disabled = false; }
          finally { busy = false; }
        }, text: 'Make my video'})));
}

// ---------- Request and video ----------

function renderRequest(route) {
  const request = state.requests.find((item) => item.id === route.id);
  if (!request) return h('section', {}, h('h1', {class: 'title', text: 'This request is gone.'}),
    h('div', {class: 'actions'}, h('a', {class: 'btn primary', href: '#videos', text: 'Your videos'})));
  const project = request.ack?.project && state.projects.find((item) => item.dir === request.ack.project);
  if (project && !request.stalled) { queueMicrotask(() => location.replace(`#video/${encodeURIComponent(project.name)}`)); }
  const info = requestInfo(request, state.listening);
  const kind = {learn: 'Learn from your finished videos', final: 'Make the final video', message: 'Message to Claude'}[request.kind]
    ?? `${typeLabel(request.payload?.video_type)} · ${takesText(request.payload?.takes ?? [])}`;
  return h('section', {},
    h('p', {class: 'step-count', text: kind}),
    h('h1', {class: 'title', text: info.title}),
    info.lede ? h('p', {class: 'lede', text: info.lede}) : null,
    !request.stalled ? h('ol', {class: 'progress'},
      h('li', {class: request.state === 'pending' ? 'now' : 'done'}, h('span', {class: 'mark'}),
        h('strong', {text: request.state === 'pending' ? 'Sending to Claude' : 'Claude has it'}), h('span', {class: 'who', text: 'Claude'})),
      h('li', {class: request.state === 'claimed' ? 'now' : request.state === 'failed' ? 'stopped' : ''}, h('span', {class: 'mark'}),
        h('strong', {text: 'Copying your footage into a project'}), h('span', {class: 'who', text: 'Claude'}))) : null,
    request.state === 'failed' ? h('div', {class: 'banner error'}, h('strong', {text: 'Open Claude Code to see what it needs.'})) : null,
    !state.listening && request.state === 'pending' ? notListening() : null,
    h('div', {class: request.stalled ? 'actions inline' : 'actions'},
      request.stalled ? h('button', {type: 'button', class: 'btn primary', onclick: async () => {
        try { await api(`/api/requests/${request.id}/resume`, {}); toast('Sent again.'); await refresh(); }
        catch (error) { toast(error.message, true); }
      }, text: 'Send again'}) : null,
      request.state === 'pending' || request.stalled ? h('button', {type: 'button', class: 'btn', onclick: async () => {
        try { await api(`/api/requests/${request.id}/cancel`, {}); toast('Cancelled.'); await refresh(); }
        catch (error) { toast(error.message, true); }
      }, text: 'Cancel'}) : null,
      ['failed', 'cancelled'].includes(request.state) ? h('a', {class: 'btn primary', href: '#new/what', text: 'Start again'}) : null));
}

function finalRequested(project) {
  return ['pending', 'delivered', 'claimed'].includes(project?.final_request?.state);
}

async function makeFinal(project, button) {
  button.disabled = true;
  try {
    await api('/api/requests', {kind: 'final', project: project.dir});
    toast(state.listening ? 'Sent to Claude.' : 'Saved. Claude starts when Claude Code is open.');
    await refresh();
  } catch (error) { toast(error.message, true); button.disabled = false; }
}

function shown(project) {
  // Once the final is asked for, show it as under way before Claude's first receipt lands.
  return project && project.phase === 'cut_ready' && finalRequested(project)
    ? {...project, phase: 'building', step: 'Starting the final'} : project;
}

function renderVideo(route) {
  const project = shown(findProject(state.projects, route.name));
  if (!project) {
    const request = state.requests.find((item) => ['pending', 'delivered', 'claimed'].includes(item.state)
      && item.ack?.project === `projects/${route.name}`);
    if (request) return renderRequest({id: request.id});
    return h('section', {}, h('h1', {class: 'title', text: 'This video is not here.'}),
      h('p', {class: 'lede', text: 'It may have been moved out of the projects folder.'}),
      h('div', {class: 'actions'}, h('a', {class: 'btn primary', href: '#videos', text: 'Your videos'})));
  }
  if (project.name !== route.name) {
    queueMicrotask(() => location.replace(`#video/${encodeURIComponent(project.name)}`));
  }
  const stalledRequest = state.requests.find((request) => request.stalled
    && request.ack?.project === project.dir);
  if (stalledRequest) return renderRequest({id: stalledRequest.id});
  const info = phaseInfo(project.phase);
  const review = linkFor(project.review, location.hostname);
  const phoneLabel = review && review === project.review?.phone_url ? phoneLinkLabel(project.review.phone_via) : '';
  const reviewLabel = phoneLabel ? ` · ${phoneLabel}` : '';
  const versions = project.versions ?? [];
  const latest = versions.at(-1);
  const content = [h('p', {class: 'step-count', text: `${typeLabel(project.video?.type)} · ${project.title || project.name}`})];
  if (project.phase === 'approved') {
    content.push(h('div', {class: 'done-hero'}, h('span', {class: 'done-mark'}, icon('bigCheck')),
      h('h1', {class: 'title', text: info.title})));
    content.push(h('p', {class: 'lede', text: `You approved version ${project.approved?.version?.replace(/^v/, '') ?? ''}. It is saved in your workspace.`}));
    if (project.approved?.file) content.push(h('p', {class: 'file', text: project.approved.file}));
    const onMac = state.on_this_mac;
    const macAction = (route, label, primary) => h('button', {type: 'button', class: `btn big${primary ? ' primary' : ''}`, onclick: async () => {
      try { await api(route, {project: project.dir}); } catch (error) { toast(error.message, true); }
    }, text: label});
    content.push(h('div', {class: 'actions inline'},
      onMac ? macAction('/api/open', 'Open', true) : null,
      onMac ? macAction('/api/reveal', 'Show in Finder', false) : null,
      project.approved?.download ? h('a', {class: `btn big${onMac ? '' : ' primary'}`, href: withToken(project.approved.download), download: '', text: 'Download'}) : null,
      review ? h('a', {class: 'link', href: review, text: `Watch it again${reviewLabel}`}) : null,
      h('span', {class: 'spacer'}),
      h('a', {class: 'link', href: '#new/what', onclick: startNew, text: 'Make another video'})));
    content.push(versionNotes(project), messagesSection(project));
    return h('section', {}, content);
  }
  content.push(h('h1', {class: 'title', text: info.title}));
  const lede = {
    cut_ready: 'Watch the story cut and leave notes on any moment. When you send them, Claude makes the next version.',
    final_ready: 'Watch the whole video. Leave notes for another version, or approve it to download it.',
    revising: 'Your notes reached Claude. The next version shows up here when it is ready.',
    building: 'Captions, titles, cards, B-roll and music go on, then the video renders and every check runs.',
    story: 'Claude reads your words and picks the order that tells the story best. It keeps your words and your meaning.',
    transcribing: 'Your speech is turned into words on this Mac. Nothing is uploaded.',
    starting: 'Claude is copying your footage into a new project.',
    cutting: 'Claude cuts your take to the story and checks every join for clipped words.',
    stopped: project.message || 'Open Claude Code to see what it needs.',
  }[project.phase];
  if (lede) content.push(h('p', {class: 'lede', text: lede}));
  if (['cut_ready', 'final_ready'].includes(project.phase)) {
    content.push(h('div', {class: 'actions inline'},
      review ? h('a', {class: 'btn primary big', href: review, text: (project.phase === 'cut_ready' ? 'Watch the story cut' : 'Watch and approve') + reviewLabel})
        : h('p', {class: 'note', text: 'Claude is opening the review page. The button shows up here in a moment.'}),
      project.phase === 'cut_ready' ? h('button', {type: 'button', class: 'btn big', id: 'make-final',
        onclick: (event) => makeFinal(project, event.target), text: 'Make the final video'}) : null));
    if (project.phase === 'cut_ready') {
      content.push(h('p', {class: 'note', text: 'Happy with the order of the story? Make the final video: Claude adds captions, cards and music, renders it and checks it. Want changes first? Leave notes on the review page.'}));
    }
  }
  content.push(h('ol', {class: 'progress'}, progressRows(project).map((row) => h('li', {class: row.status},
    h('span', {class: 'mark', 'aria-hidden': 'true'}),
    h('span', {}, h('strong', {text: row.label}), row.sub ? h('small', {text: row.sub}) : null,
      h('span', {class: 'sr-only', text: {done: ', done', now: ', in progress', turn: ', your turn', stopped: ', stopped', next: ''}[row.status]})),
    h('span', {class: 'who', text: row.who})))));
  const notes = (state.events ?? []).find((event) => event.kind === 'notes' && event.project === project.dir);
  if (notes && project.phase === 'revising') {
    content.push(h('div', {class: 'banner done'}, notes.delivered_at ? 'Claude has your notes.' : 'Your notes are waiting for Claude.'));
  }
  if (project.phase === 'stopped') content.push(h('div', {class: 'banner error'}, h('strong', {text: 'Open Claude Code to see what it needs.'})));
  const claudesTurn = !['cut_ready', 'final_ready'].includes(project.phase);
  if (!state.listening && !claudeWorking() && claudesTurn) content.push(notListening());
  content.push(versionNotes(project), messagesSection(project));
  return h('section', {}, content);
}

function versionNotes(project) {
  // Claude's own account of each version: what changed and why.
  const versions = [...(project.versions ?? [])].reverse();
  if (!versions.length) return null;
  const [latest, ...earlier] = versions;
  const entry = (version) => h('li', {}, h('strong', {text: `Version ${version.number}${version.final ? ' · final' : ' · story cut'}`}),
    h('span', {class: 'when', text: formatWhen(version.created_at)}),
    h('p', {text: version.note || 'No notes for this version.'}));
  return h('div', {class: 'section'}, h('h2', {text: "Claude's notes"}),
    h('ol', {class: 'version-notes'}, entry(latest)),
    earlier.length ? fold(`notes-${project.name}`, h('summary', {}, `Earlier versions `, h('span', {text: plural(earlier.length, 'version')})),
      h('ol', {class: 'version-notes fold-body'}, earlier.map(entry))) : null);
}

function messagesSection(project) {
  const messages = project.messages ?? [];
  const id = `message-${project.name}`;
  const send = async (event) => {
    const box = document.getElementById(id);
    const text = box.value.trim();
    if (!text) { box.focus(); return; }
    event.target.disabled = true;
    try {
      await api('/api/requests', {kind: 'message', project: project.dir, text});
      box.value = '';
      draftMessages.delete(id);
      toast(state.listening ? 'Sent to Claude.' : 'Saved. Claude reads it when Claude Code is open.');
      await refresh();
    } catch (error) { toast(error.message, true); }
    finally { event.target.disabled = false; }
  };
  return h('div', {class: 'section'},
    h('h2', {text: 'Talk to Claude about this video'}),
    h('p', {class: 'hint', text: 'Ask a question or say what you want changed. For a note on one moment, the review page is quicker.'}),
    messages.length ? h('ol', {class: 'messages'}, messages.map((message) => h('li', {},
      h('p', {class: 'mine', text: message.text}),
      message.reply ? h('p', {class: 'reply'}, h('strong', {text: 'Claude: '}), message.reply)
        : h('p', {class: 'waiting', text: message.state === 'pending' ? (state.listening ? 'Sending to Claude…' : 'Waiting for Claude Code.') : 'Claude is on it.'})))) : null,
    h('div', {class: 'field message-field'},
      h('label', {class: 'sr-only', for: id, text: 'Message to Claude'}),
      h('textarea', {id, rows: '3', maxlength: '2000', placeholder: 'For example: Can the captions sit a little lower?',
        oninput: (event) => draftMessages.set(id, event.target.value)}, draftMessages.get(id) ?? ''),
      h('div', {}, h('button', {type: 'button', class: 'btn', onclick: send, text: 'Send to Claude'}))));
}

const draftMessages = new Map();

function renderVideos() {
  const waiting = state.requests.filter((request) => request.stalled || request.kind === 'edit'
    && ['pending', 'delivered', 'claimed'].includes(request.state) && !request.ack?.project);
  return h('section', {},
    h('h1', {class: 'title', text: 'Your videos'}),
    h('div', {class: 'actions inline'},
      h('a', {class: 'btn primary big', href: '#new/what', onclick: startNew, text: 'Make a new video'})),
    h('div', {class: 'videos'},
      waiting.map((request) => h('a', {class: 'video-row', href: `#request/${request.id}`},
        h('span', {}, h('strong', {text: `${typeLabel(request.payload?.video_type)} · ${takesText(request.payload?.takes ?? [])}`}),
          h('span', {text: `Asked ${formatWhen(request.created_at)}`})),
        h('span', {class: `state ${requestInfo(request, state.listening).tone}`, text: request.stalled ? 'Stopped' : requestInfo(request, state.listening).title.replace(/\.$/, '')}))),
      state.projects.map(shown).map((project) => {
        const info = phaseInfo(project.phase);
        return h('a', {class: 'video-row', href: `#video/${encodeURIComponent(project.name)}`},
          h('span', {}, h('strong', {text: project.title || project.name}),
            h('span', {text: `${typeLabel(project.video?.type)} · ${takesText(project.takes)} · started ${formatWhen(project.created_at)}`})),
          h('span', {class: `state ${info.tone}`, text: info.label}));
      }),
      !waiting.length && !state.projects.length ? h('div', {class: 'empty'}, h('strong', {text: 'No videos yet'}),
        'Make your first one: it takes a few choices.') : null));
}

// ---------- Start ----------

window.addEventListener('hashchange', render);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); schedule(); });
refresh().then(schedule);
