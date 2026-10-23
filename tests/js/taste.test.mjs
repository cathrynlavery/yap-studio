import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as timelineHelpers from '../../src/yap_studio/review/static/timeline.mjs';
import * as tasteHelpers from '../../src/yap_studio/review/static/taste.mjs';
import {
  decisionBody, machineText, metaText, orderedProposals, pendingCount, scopeName,
  sentenceText, sourceRows, splitProposals, stateText, tasteActions,
} from '../../src/yap_studio/review/static/taste.mjs';

const proposal = Object.freeze({id: 'r-0123456789ab', scope: 'captions',
  sentence: 'Keep captions short.', state: 'pending', condition: null, machine: null,
  edited: false, rule: null});

test('scope names cover every supported scope', () => {
  for (const [scope, name] of Object.entries({cut: 'Cut', captions: 'Captions', cards: 'Cards',
    pacing: 'Pacing', music: 'Music'})) assert.equal(scopeName(scope), name);
  assert.equal(scopeName(null), 'Rule');
});

test('machine text describes style fields and values', () => {
  assert.equal(machineText({kind: 'style_field', field: 'captions.max_words', value: 3}),
    'Sets caption max words to 3');
  assert.equal(machineText({kind: 'style_field', field: 'captions.max_words_per_phrase', value: 3}),
    'Sets caption max words per phrase to 3');
  assert.equal(machineText({kind: 'style_field', field: 'captions.text_color', value: '#FFF4DD'}),
    'Sets caption text color to #FFF4DD');
  assert.equal(machineText({kind: 'style_field', field: 'music.enabled', value: false}),
    'Sets music enabled to off');
  assert.equal(machineText({kind: 'style_field', field: 'music.enabled', value: true}),
    'Sets music enabled to on');
  assert.equal(machineText({kind: 'style_field', field: 'hook.font_family', value: 'Arial'}),
    'Sets title font family to Arial');
  assert.equal(machineText({kind: 'style_field', field: 'graphics.enabled', value: true}),
    'Sets card enabled to on');
});

test('machine text moves trailing units from field names to values', () => {
  assert.equal(machineText({kind: 'style_field', field: 'captions.size_px', value: 96}),
    'Sets caption size to 96 px');
  assert.equal(machineText({kind: 'style_field', field: 'hook.duration_s', value: 4}),
    'Sets title duration to 4 s');
  assert.equal(machineText({kind: 'style_field', field: 'music.fade_ms', value: 250}),
    'Sets music fade to 250 ms');
  assert.equal(machineText({kind: 'style_field', field: 'graphics.border_width_px', value: 2}),
    'Sets card border width to 2 px');
  assert.equal(machineText({kind: 'style_field', field: 'captions.px_offset', value: 5}),
    'Sets caption px offset to 5');
});

test('machine text describes both filler positions and each placement layer', () => {
  assert.equal(machineText({kind: 'filler_word', word: 'so', position: 'leading'}),
    'Cuts the filler word "so" at the start');
  assert.equal(machineText({kind: 'filler_word', word: 'um', position: 'anywhere'}),
    'Cuts the filler word "um" anywhere');
  for (const layer of ['captions', 'title', 'cards']) {
    assert.equal(machineText({kind: 'placement', layer, rule: 'clear_of_face'}),
      `Keeps ${layer} clear of the face`);
  }
  assert.equal(machineText(null), '');
  assert.equal(machineText(undefined), '');
  assert.equal(machineText({kind: 'unknown'}), '');
});

test('meta text includes optional condition and machine words', () => {
  assert.equal(metaText(proposal), 'Captions');
  assert.equal(metaText({...proposal, condition: {text: 'A close shot', tags: ['close']}}),
    'Captions. Only when: A close shot');
  assert.equal(metaText({...proposal, condition: {text: 'A close shot'},
    machine: {kind: 'placement', layer: 'captions', rule: 'clear_of_face'}}),
  'Captions. Only when: A close shot. Keeps captions clear of the face');
});

test('sentence uses approved wording and keeps it after the rule ends', () => {
  assert.equal(sentenceText(proposal), 'Keep captions short.');
  for (const state of ['approved', 'revoked', 'superseded']) {
    assert.equal(sentenceText({...proposal, state, edited: true,
      rule: {sentence: 'Use short phrases.'}}), 'Use short phrases.');
  }
  assert.equal(sentenceText({...proposal, edited: false, rule: {sentence: 'Other'}}),
    proposal.sentence);
});

test('state text covers pending, both approvals and every ended state', () => {
  assert.equal(stateText(proposal), '');
  const approved = {...proposal, state: 'approved', rule: {approved_revision: 7}};
  assert.equal(stateText(approved), 'Approved. Applies to future videos (taste revision 7).');
  assert.equal(stateText({...approved, edited: true}),
    'Approved with your wording (taste revision 7).');
  for (const [state, text] of Object.entries({rejected: 'Rejected.',
    revoked: 'Stopped. No longer applied.', superseded: 'Replaced by a newer rule.'})) {
    assert.equal(stateText({...proposal, state}), text);
  }
});

test('state text identifies creator decisions made in another video', () => {
  const approved = {...proposal, state: 'approved', decided_here: false,
    rule: {approved_revision: 7}};
  assert.equal(stateText(approved),
    'Approved earlier (taste revision 7).');
  assert.equal(stateText({...approved, edited: true}),
    'Approved earlier (taste revision 7).');
  assert.equal(stateText({...approved, decided_here: true}),
    'Approved. Applies to future videos (taste revision 7).');
  assert.equal(stateText({...approved, decided_here: true, edited: true}),
    'Approved with your wording (taste revision 7).');
  const rejected = {...proposal, state: 'rejected', decided_here: false};
  assert.equal(stateText(rejected), 'Rejected earlier.');
  assert.equal(stateText({...rejected, decided_here: true}), 'Rejected.');
  assert.deepEqual(tasteActions(rejected), [{action: 'approve', label: 'Approve instead'}]);
});

test('actions cover every proposal state', () => {
  assert.deepEqual(tasteActions(proposal), [
    {action: 'approve', label: 'Approve'}, {action: 'reject', label: 'Reject'},
    {action: 'edit', label: 'Edit wording'},
  ]);
  assert.deepEqual(tasteActions({...proposal, state: 'rejected'}),
    [{action: 'approve', label: 'Approve instead'}]);
  assert.deepEqual(tasteActions({...proposal, state: 'approved'}),
    [{action: 'revoke', label: 'Stop using'}]);
  for (const state of ['revoked', 'superseded']) {
    assert.deepEqual(tasteActions({...proposal, state}), []);
  }
});

test('ordering preserves inbox order in both groups without changing the input', () => {
  const states = ['approved', 'pending', 'rejected', 'pending', 'revoked', 'superseded'];
  const items = Object.freeze(states.map((state, id) => Object.freeze({id, state})));
  assert.deepEqual(orderedProposals(items).map((item) => item.id), [1, 3, 0, 2, 4, 5]);
  assert.deepEqual(items.map((item) => item.state), states);
  assert.equal(pendingCount(items), 2);
  assert.equal(pendingCount([]), 0);
  assert.deepEqual(orderedProposals([]), []);
});

test('splitting preserves inbox order for pending proposals and every decided state', () => {
  const states = ['approved', 'pending', 'rejected', 'pending', 'revoked', 'superseded'];
  const items = Object.freeze(states.map((state, id) => Object.freeze({id, state})));
  const {pending, decided} = splitProposals(items);
  assert.deepEqual(pending.map((item) => item.id), [1, 3]);
  assert.deepEqual(decided.map((item) => item.id), [0, 2, 4, 5]);
  assert.deepEqual(items.map((item) => item.state), states);
  assert.deepEqual(splitProposals([]), {pending: [], decided: []});
  assert.deepEqual(splitProposals(pending), {pending, decided: []});
  assert.deepEqual(splitProposals(decided), {pending: [], decided});
});

const versions = [{number: 1, fps: '25/1'}, {number: 2, fps: '30000/1001'},
  {number: 3, fps: null}];
const source = {version: 2, label: 'fix', start_frame: 372, end_frame: 400,
  quote: 'Fewer words please.'};

test('source rows use each version rate and carry its seek target', () => {
  const sources = Object.freeze([
    Object.freeze(source), Object.freeze({...source, label: 'keep', version: 1, start_frame: 372}),
  ]);
  const rows = sourceRows({sources}, versions);
  assert.deepEqual(rows[0], {label: 'Fix', location: 'v2 at 0:12.4', quote: source.quote,
    version: 2, frame: 372, canGo: true, time_s: 372 * 1001 / 30000});
  assert.equal(rows[1].label, 'Keep');
  assert.equal(rows[1].location, 'v1 at 0:14.8');
  assert.equal(rows[1].time_s, 372 / 25);
  assert.equal(sourceRows({sources: [{...source, version: 1, start_frame: 90000}]}, versions)
    [0].location, 'v1 at 60:00.0');
});

test('null frames never show a time or offer a seek button', () => {
  const row = sourceRows({sources: [{...source, start_frame: null, end_frame: 400}]}, versions)[0];
  assert.equal(row.location, 'v2');
  assert.equal(row.frame, null);
  assert.equal(row.canGo, false);
  assert.equal(row.time_s, null);
  assert.deepEqual(sourceRows({sources: []}, versions), []);
});

test('sources outside the series keep their quote without a seek button or invented time', () => {
  for (const version of [null, 9]) {
    const row = sourceRows({sources: [{...source, version}]}, versions)[0];
    assert.equal(row.location, version === null ? 'Version unavailable' : 'v9');
    assert.equal(row.quote, source.quote);
    assert.equal(row.canGo, false);
    assert.equal(row.time_s, null);
  }
});

test('setup answer sources show their answer without a seek button', () => {
  const row = sourceRows({sources: [{type: 'answer', question: 'never', answer: 'cover_face'}]}, []);
  assert.deepEqual(row, [{label: 'From your setup answers', location: '', quote: 'cover_face',
    version: null, frame: null, canGo: false, time_s: null}]);
});

test('missing or invalid rates omit time while frames in the series remain seekable', () => {
  for (const fps of [null, 'bad', '0/1']) {
    const row = sourceRows({sources: [{...source, version: 3}]}, [{number: 3, fps}])[0];
    assert.equal(row.location, 'v3');
    assert.equal(row.canGo, true);
    assert.equal(row.time_s, null);
  }
  assert.equal(sourceRows({sources: [{...source, start_frame: 0}]}, versions)[0].location,
    'v2 at 0:00.0');
});

test('decision body strips changed wording and omits unchanged or irrelevant wording', () => {
  for (const action of ['approve', 'reject', 'revoke']) {
    assert.deepEqual(decisionBody(proposal, action), {proposal: proposal.id, action});
  }
  assert.deepEqual(decisionBody(proposal, 'approve', '  Use short phrases.  '),
    {proposal: proposal.id, action: 'approve', sentence: 'Use short phrases.'});
  assert.deepEqual(decisionBody(proposal, 'approve', `  ${proposal.sentence}  `),
    {proposal: proposal.id, action: 'approve'});
  assert.deepEqual(decisionBody(proposal, 'approve', '   '),
    {proposal: proposal.id, action: 'approve', sentence: ''});
  for (const action of ['reject', 'revoke']) {
    assert.deepEqual(decisionBody(proposal, action, 'Changed'), {proposal: proposal.id, action});
  }
  assert.deepEqual(decisionBody(proposal, 'approve', null),
    {proposal: proposal.id, action: 'approve'});
  assert.deepEqual(decisionBody(proposal, 'approve', 'Use short phrases.\n'),
    {proposal: proposal.id, action: 'approve', sentence: 'Use short phrases.'});
  assert.deepEqual(decisionBody(proposal, 'approve', 'Use short\nphrases.\n'),
    {proposal: proposal.id, action: 'approve', sentence: 'Use short\nphrases.'});
});

async function pageFixture() {
  let document;
  class Node {
    constructor(tag = '') {
      Object.assign(this, {tag, children: [], attributes: {}, listeners: {}, dataset: {},
        value: '', textContent: '', hidden: false, disabled: false, currentTime: 0,
        style: {setProperty() {}}, classList: {toggle() {}, add() {}, remove() {}}});
    }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; delete this[key]; }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    append(...children) {
      for (const child of children) { child.parent = this; this.children.push(child); }
    }
    get childElementCount() { return this.children.length; }
    get tagName() { return this.tag.toUpperCase(); }
    matches(selector) {
      if (selector.includes(',')) return selector.split(',').some((part) => this.matches(part.trim()));
      if (selector.startsWith('.')) return this.className?.split(' ').includes(selector.slice(1));
      if (selector.startsWith('[data-proposal=')) {
        return this.dataset.proposal === selector.match(/"([^"]+)"/u)?.[1];
      }
      return this.tag === selector;
    }
    querySelectorAll(selector) {
      const selectors = selector.split(',').map((part) => part.trim());
      return this.children.flatMap((child) => [
        ...(selectors.some((part) => child.matches(part)) ? [child] : []),
        ...child.querySelectorAll(selector),
      ]);
    }
    querySelector(selector) {
      const [first, rest] = selector.split(' ');
      const found = this.querySelectorAll(first)[0] ?? null;
      return rest ? found?.querySelector(rest) ?? null : found;
    }
    closest(selector) {
      if (selector === 'main') return main;
      if (this.matches(selector)) return this;
      return this.parent?.closest(selector) ?? null;
    }
    focus() { document.activeElement = this; }
    setSelectionRange() {}
    pause() {}
  }
  const main = new Node('main');
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  const items = [
    {...proposal, sources: [source]},
    {...proposal, id: 'r-123456789abc', sentence: 'Keep the pause.', state: 'rejected', sources: []},
  ];
  const state = {taste: {available: true, revision: 0, proposals: items}, network: true,
    changed: false, failure: null, gate: null, reviewGate: null, comments: []};
  const requests = [];
  document = {getElementById: element, createElement: (tag) => new Node(tag),
    listeners: {}, addEventListener(name, handler) { this.listeners[name] = handler; },
    dispatchEvent(event) { this.listeners[event.type]?.(event); },
    visibilityState: 'visible', activeElement: null};
  const context = vm.createContext({
    ...timelineHelpers, ...tasteHelpers, URLSearchParams, Element: Node,
    location: new URL('http://review.test/v/2/?t=token&version=1'),
    history: {replaceState() {}}, document, window: {addEventListener() {}},
    setInterval() { return 1; }, clearInterval() {},
    setTimeout() {}, clearTimeout() {},
    fetch: async (path, options) => {
      requests.push({path, options});
      if (!state.network) throw new Error('Network unavailable');
      if (path === 'api/taste') {
        if (state.gate) await state.gate;
        if (state.failure) return {ok: false, status: 409,
          json: async () => ({error: {message: state.failure}})};
        const body = JSON.parse(options.body);
        const item = state.taste.proposals.find((entry) => entry.id === body.proposal);
        state.taste.revision += body.action === 'reject' ? 0 : 1;
        item.state = ({approve: 'approved', reject: 'rejected', revoke: 'revoked'})[body.action];
        if (body.action === 'approve') {
          item.edited = Boolean(body.sentence);
          item.rule = {sentence: body.sentence ?? item.sentence,
            approved_revision: state.taste.revision};
        }
        return {ok: true, status: 200, json: async () => ({taste: structuredClone(state.taste)})};
      }
      if (path.startsWith('api/removed')) {
        return {ok: true, status: 200, json: async () => ({available: true, ranges: []})};
      }
      const number = Number(new URLSearchParams(path.split('?')[1]).get('version') ?? 1);
      const fresh = {media: {name: `v${number}.mp4`, sha256: `hash${number}`, duration_s: 20,
        cfr: true, fps: '30/1'}, media_changed: state.changed,
      comments: structuredClone(state.comments), previous: null,
      text: `v${number}`, series: {project: true, current: number, newest: 2, current_version: 2,
        rounds: [], questions: [], threads: [], versions: [
          {number: 1, fps: '30/1'}, {number: 2, fps: '30000/1001'},
        ], taste: structuredClone(state.taste)}};
      if (state.reviewGate) await state.reviewGate;
      return {ok: true, status: 200, json: async () => fresh};
    },
  });
  const code = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url),
    'utf8').replace(/^import \{[\s\S]*?\} from '.\/[^']+';\s*/gmu, '')
    .replace(/start\(\);\s*$/u,
      'globalThis.page = {start, renderTaste, refreshReview, loadReview, saveTasteDecision, openEdit};');
  vm.runInContext(code, context);
  await context.page.start();
  const click = (button) => button.listeners.click({stopPropagation() {}});
  const card = () => element('taste-list').children[0] ?? element('taste-decided-list').children[0];
  const button = (row, label) => row.querySelectorAll('button').find((node) => node.textContent === label);
  return {page: context.page, element, document, state, requests, click, card, button};
}

test('Enter in rule wording leaves an open comment editor unsaved', async () => {
  const {page, state, element, document, requests, click, card, button} = await pageFixture();
  const comment = {id: 'c1', kind: 'fix', target: 'captions', text: 'A draft note.',
    start_frame: 12, end_frame: null, start_s: 0.4, end_s: null, request: null};
  state.comments = [comment];
  await page.loadReview();
  const row = element('comment-list').children[0];
  page.openEdit(comment, row);
  const commentTextarea = row.querySelector('textarea');
  commentTextarea.value = 'Changed comment text.';
  click(button(card(), 'Edit wording'));
  const textarea = card().querySelector('textarea');
  for (const code of ['Enter', 'NumpadEnter', 'Escape']) {
    let prevented = false;
    document.dispatchEvent({type: 'keydown', target: textarea, code,
      preventDefault() { prevented = true; }});
    assert.equal(prevented, false);
    assert.equal(requests.some(({path, options}) => path.startsWith('api/comments/')
      && options.method === 'PATCH'), false);
    assert.equal(row.querySelector('textarea'), commentTextarea);
  }
  document.dispatchEvent({type: 'keydown', target: commentTextarea, code: 'Enter',
    preventDefault() {}});
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(requests.some(({path, options}) => path.startsWith('api/comments/c1')
    && options.method === 'PATCH'));
});

test('wording save strips a trailing newline and displays server errors for internal newlines', async () => {
  const {page, state, requests, click, card, button} = await pageFixture();
  click(button(card(), 'Edit wording'));
  const textarea = card().querySelector('textarea');
  textarea.value = 'Use short\nphrases.\n';
  textarea.listeners.input();
  state.failure = 'sentence: must be stripped with no newlines';
  await click(button(card(), 'Approve this wording'));
  const post = requests.find(({path}) => path === 'api/taste');
  assert.equal(JSON.parse(post.options.body).sentence, 'Use short\nphrases.');
  assert.equal(card().querySelector('.form-error').textContent, state.failure);
  assert.equal(card().querySelector('textarea').value, textarea.value);
  state.failure = null;
  card().querySelector('textarea').value = 'Use short phrases.\n';
  await click(button(card(), 'Approve this wording'));
  assert.equal(JSON.parse(requests.filter(({path}) => path === 'api/taste').at(-1).options.body)
    .sentence, 'Use short phrases.');
  assert.equal(card().querySelector('.taste-sentence').textContent, 'Use short phrases.');
});

test('page hides an empty inbox and shows no waiting rules only when proposals exist', async () => {
  const {page, state, element} = await pageFixture();
  state.taste.proposals = [];
  await page.loadReview();
  assert.equal(element('taste-section').hidden, true);
  assert.equal(element('taste-empty').hidden, true);
  state.taste.proposals = [{...proposal, state: 'rejected', decided_here: false, sources: []}];
  await page.loadReview();
  assert.equal(element('taste-section').hidden, false);
  assert.equal(element('taste-empty').hidden, false);
  assert.equal(element('taste-count').textContent, '0');
});

test('page renders notes safely, navigates to their version, and saves edited wording', async () => {
  const {page, element, state, requests, click, card, button} = await pageFixture();
  assert.equal(element('taste-section').hidden, false);
  assert.equal(element('taste-count').textContent, '1');
  assert.equal(card().dataset.state, 'pending');
  assert.equal(card().querySelector('.taste-sentence').textContent, proposal.sentence);
  assert.equal(card().querySelector('.taste-source').children[0].textContent, 'Fix. v2 at 0:12.4');
  assert.equal(card().querySelector('.taste-notes-heading').textContent, 'From your notes');
  await click(button(card(), 'Go to'));
  assert.ok(requests.some((entry) => entry.path === 'api/review?version=2'));
  assert.equal(element('video').src, 'media?version=2&t=token');
  assert.equal(element('video').currentTime, (372 + 0.5) / 30);
  click(button(card(), 'Edit wording'));
  const textarea = card().querySelector('textarea');
  assert.equal(textarea.maxLength, 280);
  assert.equal(textarea.attributes['aria-label'], 'Rule wording');
  textarea.value = '   ';
  textarea.listeners.input();
  assert.equal(button(card(), 'Approve this wording').disabled, true);
  textarea.value = '  Use <short> phrases.  ';
  textarea.listeners.input();
  page.renderTaste();
  assert.equal(card().querySelector('textarea').value, '  Use <short> phrases.  ');
  await page.saveTasteDecision(state.taste.proposals[0], 'approve', textarea.value);
  const post = requests.find((entry) => entry.path === 'api/taste');
  assert.equal(post.options.headers['X-Yap-Token'], 'token');
  assert.deepEqual(JSON.parse(post.options.body), {proposal: proposal.id, action: 'approve',
    sentence: 'Use <short> phrases.'});
  assert.equal(card().querySelector('.taste-sentence').textContent, 'Use <short> phrases.');
  assert.equal(card().querySelector('.taste-wording').textContent,
    'Your wording. Proposed: Keep captions short.');
  assert.equal(button(card(), 'Stop using').disabled, false);
  assert.equal(element('taste-count').textContent, '0');
});

test('page disables a saving card, keeps its draft on error and retries without losing it', async () => {
  const {page, state, click, card, button} = await pageFixture();
  click(button(card(), 'Edit wording'));
  card().querySelector('textarea').value = 'Use short phrases.';
  card().querySelector('textarea').listeners.input();
  let finish;
  state.gate = new Promise((resolve) => { finish = resolve; });
  state.failure = 'The taste file is locked.';
  const saving = page.saveTasteDecision(state.taste.proposals[0], 'approve', 'Use short phrases.');
  assert.equal(card().attributes['aria-busy'], 'true');
  assert.ok(card().querySelectorAll('button, textarea').every((control) => control.disabled));
  page.renderTaste();
  assert.ok(card().querySelectorAll('button, textarea').every((control) => control.disabled));
  finish();
  await saving;
  assert.equal(card().querySelector('.form-error').attributes.role, 'alert');
  assert.equal(card().querySelector('.form-error').textContent, 'The taste file is locked.');
  assert.equal(card().querySelector('textarea').value, 'Use short phrases.');
  assert.equal(button(card(), 'Approve this wording').disabled, false);
  state.failure = null;
  await page.saveTasteDecision(state.taste.proposals[0], 'approve', 'Use short phrases.');
  assert.equal(card().querySelector('.form-error').textContent, '');
  assert.equal(card().querySelector('textarea'), null);
});

test('page skips refreshes during wording focus and saves, including a refresh already in flight', async () => {
  const {page, state, requests, document, click, card, button} = await pageFixture();
  click(button(card(), 'Edit wording'));
  const before = requests.length;
  await page.refreshReview();
  assert.equal(requests.length, before);
  document.activeElement = null;
  state.taste.revision += 1;
  let finishReview;
  state.reviewGate = new Promise((resolve) => { finishReview = resolve; });
  const refresh = page.refreshReview();
  card().querySelector('textarea').focus();
  const focusedCard = card();
  finishReview();
  await refresh;
  assert.equal(card(), focusedCard);
  state.reviewGate = null;
  document.activeElement = null;
  let finishPost;
  state.gate = new Promise((resolve) => { finishPost = resolve; });
  const saving = page.saveTasteDecision(state.taste.proposals[0], 'reject');
  const during = requests.length;
  await page.refreshReview();
  assert.equal(requests.length, during);
  finishPost();
  await saving;
  assert.ok(button(card(), 'Approve instead'));
});

test('page keeps polling with Go to focused and applies a refresh with button focus in flight', async () => {
  const {page, state, requests, click, card, button} = await pageFixture();
  await click(button(card(), 'Go to'));
  const go = button(card(), 'Go to');
  go.focus();
  await click(go);
  const before = requests.length;
  await page.refreshReview();
  assert.equal(requests.length, before + 1);
  state.taste.revision += 1;
  let finishReview;
  state.reviewGate = new Promise((resolve) => { finishReview = resolve; });
  const refresh = page.refreshReview();
  go.focus();
  const focusedCard = card();
  finishReview();
  await refresh;
  assert.notEqual(card(), focusedCard);
});

test('page collapses decided rules, keeps their open state and hides an empty decided group', async () => {
  const {page, state, element} = await pageFixture();
  const details = element('taste-decided');
  assert.equal(details.hidden, false);
  assert.equal(details.open, false);
  assert.equal(element('taste-list').children.length, 1);
  assert.equal(element('taste-decided-list').children.length, 1);
  assert.equal(element('taste-decided-count').textContent, '1');
  assert.equal(element('taste-empty').hidden, true);
  for (const [index, decidedState] of ['approved', 'revoked', 'superseded'].entries()) {
    state.taste.proposals.push({...proposal, id: `r-decided${index}`, state: decidedState,
      sources: [], rule: {approved_revision: 1}});
  }
  await page.loadReview();
  assert.deepEqual(element('taste-decided-list').children.map((item) => item.dataset.state),
    ['rejected', 'approved', 'revoked', 'superseded']);
  assert.equal(element('taste-decided-count').textContent, '4');
  details.open = true;
  details.listeners.toggle();
  page.renderTaste();
  assert.equal(details.open, true);
  await page.loadReview();
  assert.equal(details.open, true);
  details.open = false;
  details.listeners.toggle();
  page.renderTaste();
  assert.equal(details.open, false);
  await page.saveTasteDecision(state.taste.proposals[0], 'approve');
  assert.equal(element('taste-list').children.length, 0);
  assert.equal(element('taste-decided-list').children.length, 5);
  assert.equal(element('taste-decided-count').textContent, '5');
  assert.equal(element('taste-empty').hidden, false);
  assert.equal(details.open, false);
  state.taste.proposals = [{...proposal, sources: []}];
  await page.loadReview();
  assert.equal(details.hidden, true);
  assert.equal(element('taste-decided-count').textContent, '0');
  assert.equal(element('taste-empty').hidden, true);
});

test('page displays unavailable taste and blocks writes offline or after media changes', async () => {
  const {page, state, requests, element, card, button} = await pageFixture();
  state.changed = true;
  await page.loadReview();
  assert.equal(button(card(), 'Approve').disabled, true);
  assert.ok(element('taste-decided-list').querySelectorAll('button')
    .every((control) => control.disabled));
  const count = requests.length;
  await page.saveTasteDecision(state.taste.proposals[0], 'approve');
  assert.equal(requests.length, count);
  state.changed = false;
  await page.loadReview();
  state.network = false;
  await page.refreshReview();
  assert.equal(button(card(), 'Approve').disabled, true);
  state.network = true;
  await page.refreshReview();
  assert.equal(button(card(), 'Approve').disabled, false);
  state.taste = {available: false, error: {message: 'The taste file could not be read.'}, proposals: []};
  await page.loadReview();
  assert.equal(element('taste-error').hidden, false);
  assert.equal(element('taste-error').textContent, 'The taste file could not be read.');
  assert.equal(element('taste-list').children.length, 0);
  assert.equal(element('taste-decided-list').children.length, 0);
  assert.equal(element('taste-decided').hidden, true);
  assert.equal(element('taste-empty').hidden, true);
  assert.equal(element('taste-section').hidden, false);
  state.taste = {available: true, revision: 0, proposals: []};
  await page.loadReview();
  assert.equal(element('taste-section').hidden, true);
});
