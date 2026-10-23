import {
  roleLabel, statusText, durationText, noteCountText, waitChoices, waitOption, clampWords,
} from './outline-view.mjs';

const $ = id => document.getElementById(id);
const token = new URLSearchParams(location.search).get('t') || '';
const choices = waitChoices();
const composers = new Map();
const expanded = new Set();
let story = null;
let offline = false;
let snapshotError = true;
let busy = false;
let polling = false;
let fingerprint = '';
let pollError = false;
const formatTime = value => new Date(value).toLocaleTimeString([], {
  hour: 'numeric', minute: '2-digit',
});

function node(tag, className = '', text = '') {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function button(text, action, writes = false, className = '') {
  const element = node('button', className, text);
  element.type = 'button';
  if (writes) element.dataset.write = '';
  element.addEventListener('click', action);
  return element;
}

function readDraft(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key));
    if (value && typeof value.text === 'string' && ['fix', 'keep'].includes(value.kind)) {
      return {text: value.text, kind: value.kind, open: Boolean(value.text)};
    }
  } catch { /* The page still works when local storage is unavailable. */ }
  return null;
}

function saveDraft(key, value) {
  try {
    if (value.text) localStorage.setItem(key, JSON.stringify(value));
    else localStorage.removeItem(key);
  } catch { /* Saved server notes remain durable even without local storage. */ }
}

function composerKey(target, note) {
  return `yap-outline:${story.outline.id}:${note ? note.id : target ?? 'whole'}`;
}

function composerState(key, note) {
  if (!composers.has(key)) {
    composers.set(key, readDraft(key) || {
      kind: note?.kind || 'fix', text: note?.text || '', open: false,
    });
  }
  return composers.get(key);
}

function updateControls() {
  const blocked = busy || offline || snapshotError || !story?.outline
    || story.status.state === 'stale';
  document.querySelectorAll('[data-write]').forEach(element => {
    element.disabled = blocked || element.dataset.empty === 'true';
  });
  const notes = story?.notes || [];
  $('outline-send').textContent = noteCountText(notes);
  $('outline-send').disabled = blocked || !notes.some(note => note.batch == null);
  $('outline-offline').hidden = !offline;
}

async function api(path, method = 'GET', body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  let response;
  let payload;
  try {
    response = await fetch(path, {
      method, signal: controller.signal,
      headers: {'X-Yap-Token': token, 'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    payload = response.status === 204 ? null : await response.json();
  } catch (cause) {
    if (cause instanceof SyntaxError) throw new Error('The server returned an unreadable response.');
    offline = true;
    updateControls();
    const error = new Error('The outline server is not reachable. Your notes are saved.');
    error.network = true;
    throw error;
  } finally {
    clearTimeout(timer);
  }
  offline = false;
  updateControls();
  if (!response.ok) throw new Error(payload.error?.message || `Request failed (${response.status}).`);
  return payload;
}

async function refresh() {
  const value = await api('/api/outline');
  story = value;
  snapshotError = false;
  $('project-name').textContent = value.project;
  document.title = `${value.project} | Story outline`;
  $('outline-status').textContent = statusText(value.status, formatTime);
  $('outline-summary').hidden = !value.outline?.summary;
  $('outline-summary').textContent = value.outline?.summary || '';
  $('older-notes').hidden = value.older_notes === 0;
  $('older-notes').textContent = `${value.older_notes} ${value.older_notes === 1 ? 'note belongs' : 'notes belong'} to an earlier story.`;
  const nextFingerprint = JSON.stringify([value.outline, value.notes]);
  if (nextFingerprint !== fingerprint) {
    fingerprint = nextFingerprint;
    render();
  }
  syncWait();
  updateControls();
}

async function mutate(path, method = 'POST', body = {}, onSuccess = () => {}) {
  if (busy || offline || snapshotError || story?.status.state === 'stale') return;
  busy = true;
  $('outline-error').textContent = '';
  pollError = false;
  updateControls();
  try {
    const result = await api(path, method, body);
    onSuccess(result);
    await refresh();
  } catch (error) {
    if (!error.network) $('outline-error').textContent = error.message;
  } finally {
    busy = false;
    updateControls();
  }
}

function textBlock(text, key) {
  const container = node('div', 'outline-words');
  const paragraph = node('p', 'outline-spoken');
  const clamped = clampWords(text, 48);
  const hasMore = clamped !== text;
  const toggle = button('', () => {
    if (expanded.has(key)) expanded.delete(key);
    else expanded.add(key);
    update();
  });
  paragraph.id = `words-${key}`;
  toggle.setAttribute('aria-controls', paragraph.id);
  function update() {
    const all = expanded.has(key);
    paragraph.textContent = all ? text : clamped;
    paragraph.classList.toggle('outline-clamped', hasMore && !all);
    toggle.textContent = all ? 'Show less' : 'Show all';
    toggle.setAttribute('aria-expanded', String(all));
  }
  update();
  container.append(paragraph);
  if (hasMore) container.append(toggle);
  return container;
}

function compose(target, label, fixOnly, note = null) {
  const key = composerKey(target, note);
  const state = composerState(key, note);
  if (fixOnly) state.kind = 'fix';
  const container = node('div', 'outline-compose');
  const opener = button(fixOnly ? 'Bring this in' : 'Add a note', () => {
    state.open = true;
    render();
    focusComposer(key);
  }, true);
  opener.setAttribute('aria-expanded', String(state.open));
  if (!note && !state.open) {
    container.append(opener);
    return container;
  }
  if (!state.open) return container;
  const prompt = node('p', 'target-label', note ? `Edit note for ${label}` : `Note for ${label}`);
  container.append(prompt);
  if (!fixOnly) {
    const segments = node('div', 'kind-choice');
    segments.setAttribute('role', 'group');
    segments.setAttribute('aria-label', `Note kind for ${label}`);
    for (const kind of ['fix', 'keep']) {
      const choice = button(kind === 'fix' ? 'Fix' : 'Keep', () => {
        state.kind = kind;
        saveDraft(key, state);
        for (const child of segments.children) {
          child.setAttribute('aria-pressed', String(child.dataset.kind === kind));
        }
      }, true, `kind-button ${kind}`);
      choice.dataset.kind = kind;
      choice.setAttribute('aria-pressed', String(state.kind === kind));
      segments.append(choice);
    }
    container.append(segments);
  }
  const textarea = node('textarea');
  textarea.value = state.text;
  textarea.maxLength = 5000;
  textarea.dataset.write = '';
  textarea.dataset.focusKey = key;
  textarea.setAttribute('aria-label', `Note for ${label}`);
  textarea.placeholder = fixOnly ? 'What should the agent bring into the story?' : 'What should stay or change?';
  const actions = node('div', 'form-actions');
  const save = button(note ? 'Save changes' : 'Save note', () => {
    const path = note ? `/api/outline/notes/${note.id}` : '/api/outline/notes';
    const body = {kind: state.kind, text: textarea.value};
    if (!note) body.beat = target;
    mutate(path, note ? 'PATCH' : 'POST', body, () => {
      state.text = '';
      state.open = false;
      saveDraft(key, state);
      fingerprint = '';
    });
  }, true, 'primary');
  save.dataset.empty = String(!state.text.trim());
  textarea.addEventListener('input', () => {
    state.text = textarea.value;
    saveDraft(key, state);
    save.dataset.empty = String(!state.text.trim());
    updateControls();
  });
  const cancel = button('Cancel', () => {
    state.open = false;
    if (note) composers.delete(key);
    render();
  });
  actions.append(cancel, save);
  container.append(textarea, actions);
  return container;
}

function focusComposer(key) {
  const textarea = [...document.querySelectorAll('textarea')].find(el => el.dataset.focusKey === key);
  textarea?.focus();
}

function notesFor(target, label, fixOnly = false) {
  const container = node('div', 'outline-notes');
  for (const note of story.notes.filter(item => item.beat === target)) {
    const item = node('article', 'outline-note');
    const heading = node('div', 'outline-note-heading');
    heading.append(node('span', `outline-note-tag ${note.kind}`, note.kind === 'keep' ? 'Keep' : 'Fix'),
      node('span', 'muted', note.batch ? 'Sent' : 'Draft'));
    item.append(heading, node('p', 'comment-text', note.text));
    if (!note.batch) {
      const actions = node('div', 'comment-actions');
      actions.append(button('Edit', () => {
        const key = composerKey(target, note);
        composerState(key, note).open = true;
        render();
        focusComposer(key);
      }, true), button('Delete', () => mutate(`/api/outline/notes/${note.id}`, 'DELETE'), true));
      item.append(actions, compose(target, label, fixOnly, note));
    }
    container.append(item);
  }
  container.append(compose(target, label, fixOnly));
  return container;
}

function render() {
  const focused = document.activeElement;
  const focusKey = focused?.dataset.focusKey;
  const selection = focusKey ? [focused.selectionStart, focused.selectionEnd] : null;
  $('whole-notes').replaceChildren();
  $('outline-beats').replaceChildren();
  $('left-out-list').replaceChildren();
  const outline = story?.outline;
  $('whole-story').hidden = !outline;
  $('left-out').hidden = !outline?.left_out.length;
  if (!outline) return;
  $('whole-notes').append(notesFor(null, 'Whole story'));
  outline.beats.forEach((beat, index) => {
    const label = roleLabel(beat, index, outline.beats.length);
    const card = node('article', 'outline-card');
    card.append(node('h2', '', label));
    if (beat.note) card.append(node('p', 'outline-agent-note', beat.note));
    card.append(textBlock(beat.text, beat.id), node('p', 'outline-duration', durationText(beat.end - beat.start)),
      notesFor(beat.id, label));
    $('outline-beats').append(card);
  });
  outline.left_out.forEach((entry, index) => {
    const card = node('article', 'outline-card');
    card.append(node('h3', '', `Left out ${index + 1}`), textBlock(entry.text, entry.id),
      notesFor(entry.id, `Left out ${index + 1}`, true));
    $('left-out-list').append(card);
  });
  $('left-out-truncated').hidden = !outline.left_out_truncated;
  updateControls();
  if (focusKey) {
    focusComposer(focusKey);
    if (document.activeElement?.dataset.focusKey === focusKey) {
      document.activeElement.setSelectionRange(...selection);
    }
  }
}

function syncWait() {
  const deadline = story.status.deadline;
  $('wait-deadline').hidden = !deadline;
  $('wait-deadline').textContent = deadline ? `Wait ends at ${formatTime(deadline)}.` : '';
  const select = $('outline-wait');
  select.querySelector('[value="deadline"]')?.remove();
  const selected = waitOption(story.status, formatTime);
  if (selected.value === 'deadline') {
    const option = node('option', '', selected.label);
    option.value = selected.value;
    select.append(option);
  }
  select.value = selected.value;
}

for (const choice of choices) {
  const option = node('option', '', choice.label);
  option.value = choice.minutes === null ? 'none' : String(choice.minutes);
  $('outline-wait').append(option);
}
$('outline-wait').addEventListener('change', () => {
  const choice = choices.find(item => (item.minutes === null ? 'none' : String(item.minutes)) === $('outline-wait').value);
  if (!choice) return;
  mutate('/api/outline/wait', 'POST', {minutes: choice.minutes});
});
$('outline-send').addEventListener('click', () => mutate('/api/outline/send'));
$('outline-approve').addEventListener('click', () => $('approve-dialog').showModal());
$('approve-cancel').addEventListener('click', () => $('approve-dialog').close());
$('approve-confirm').addEventListener('click', () => {
  $('approve-dialog').close();
  mutate('/api/outline/approve');
});

async function poll() {
  if (busy || polling) return;
  polling = true;
  try {
    await refresh();
    if (pollError) $('outline-error').textContent = '';
    pollError = false;
  } catch (error) {
    snapshotError = true;
    if (!error.network) {
      $('outline-error').textContent = error.message;
      pollError = true;
    }
    updateControls();
  } finally {
    polling = false;
  }
}
poll();
setInterval(poll, 5000);
