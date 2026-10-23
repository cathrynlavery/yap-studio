import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as helpers from '../../src/yap_studio/review/static/timeline.mjs';

const words = [
  {id: 't1:0', segment: 's1', text: 'so', start: 0.8, end: 1.2},
  {id: 't1:1', segment: 's1', text: 'basically', start: 1.2, end: 1.8},
  {id: 't1:0', segment: 's2', text: 'yeah', start: 1.8, end: 2.2},
];

test('wordsInRange uses exact half-open midpoints and preserves occurrence order', () => {
  assert.deepEqual(helpers.wordsInRange(words, 1, 2), words.slice(0, 2));
  assert.deepEqual(helpers.wordsInRange(words, 2, 3), words.slice(2));
  const decimal = [{start: 0.1, end: 0.2}];
  assert.deepEqual(helpers.wordsInRange(decimal, 0.15, 0.2), decimal);
  assert.deepEqual(helpers.wordsInRange(decimal, 0, 0.15), []);
  const fractional = [{start: 0.0333, end: 0.0334}, {start: 0.03336, end: 0.03338},
    {start: 0.13347, end: 0.13349}];
  assert.deepEqual(helpers.wordsInRange(fractional, {num: 1001, den: 30000},
    {num: 4004, den: 30000}), fractional.slice(1, 2));
});

test('cut helpers format quotes, counts, limits, duration and exact request bodies', () => {
  assert.equal(helpers.cutWordsText(words), '"so basically yeah" · 3 words');
  assert.equal(helpers.cutWordsText(words.slice(0, 1)), '"so" · 1 word');
  assert.equal(helpers.cutWordsText([{text: 'x'.repeat(121)}]), `"${'x'.repeat(117)}..." · 1 word`);
  assert.deepEqual(helpers.cutState({available: false}, words), {show: false, enabled: false, reason: ''});
  assert.deepEqual(helpers.cutState({available: true}, words), {show: true, enabled: true, reason: ''});
  assert.equal(helpers.cutState({available: true}, []).reason, 'No spoken words in this range.');
  assert.equal(helpers.cutState({available: true}, Array(200).fill(words[0])).enabled, true);
  assert.deepEqual(helpers.cutState({available: true}, Array(201).fill(words[0])),
    {show: true, enabled: false, reason: 'Select at most 200 words to cut in one note.'});
  assert.equal(helpers.rangeLengthText(1, 3.34), '2.3 s');
  assert.deepEqual(helpers.cutBody({start_frame: 30, end_frame: 61}, {num: 30, den: 1}, '  '),
    {start_frame: 30, end_frame: 61});
  assert.deepEqual(helpers.cutBody({start_s: 1.01, end_s: 2.02}, null, '  Keep pace. \n'),
    {start_s: 1.01, end_s: 2.02, text: 'Keep pace.'});
  const request = {type: 'cut', start: 1, end: 2, words: words.map(({id, segment, text}) => ({id, segment, text}))};
  assert.equal(helpers.requestText(request), 'Cut: "so basically yeah" (3 words)');
  assert.deepEqual(helpers.editedRequest({kind: 'fix', target: 'story', request}, '', ''), request);
});

async function cutPage({cfr = true, project = true, available = true, textHook = false} = {}) {
  let document;
  class Node {
    constructor(tag = 'div') {
      Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, listeners: {},
        dataset: {}, value: '', textContent: '', hidden: false, checked: false, paused: true,
        duration: 10, currentTime: 0, readyState: 1,
        style: {setProperty(key, value) { this[key] = value; }}, classList: {toggle() {}, add() {}, remove() {}}});
    }
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
    async emit(type, event = {}) {
      for (const fn of this.listeners[type] ?? []) {
        await fn({target: this, preventDefault() {}, stopPropagation() {}, ...event});
      }
    }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    append(...nodes) { this.children.push(...nodes); nodes.forEach((node) => { node.parent = this; }); }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    insertBefore(node, before) { this.children.splice(this.children.indexOf(before), 0, node); node.parent = this; }
    matches(selector) {
      return selector.startsWith('.') ? this.className?.split(' ').includes(selector.slice(1))
        : selector.startsWith('[data-comment-id=') ? selector.includes(`"${this.dataset.commentId}"`)
          : this.tagName === selector.toUpperCase();
    }
    closest(selector) {
      if (selector.split(',').some((part) => this.matches(part.trim()))) return this;
      return this.parent?.closest(selector) ?? null;
    }
    querySelectorAll(selector) {
      return this.children.flatMap((child) => [
        ...(selector.split(',').some((part) => child.matches(part.trim())) ? [child] : []),
        ...child.querySelectorAll(selector),
      ]);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
    focus() { document.activeElement = this; }
    setSelectionRange() {}
    getBoundingClientRect() { return {left: 0, width: 300}; }
    setPointerCapture(id) { this.capture = id; }
    hasPointerCapture(id) { return this.capture === id; }
    releasePointerCapture() { this.capture = null; }
    pause() { if (!this.paused) { this.paused = true; void this.emit('pause'); } }
    play() { this.paused = false; return this.emit('play'); }
  }
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  document = {getElementById: element, createElement: (tag) => new Node(tag), activeElement: null,
    visibilityState: 'visible', listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }};
  element('players').parent = new Node('main');
  for (const edge of ['start', 'end']) element(`selection-${edge}`).className = `selection-handle ${edge}`;
  element('compose').hidden = true;
  const state = {comments: [], available, words, fail: false, failWords: false, gate: null,
    home: null, versions: [{number: 1}, {number: 2}], map: [{name: 'Cards', items: [
      {start: 2, end: 4, label: '<A saved card>', target: 'element'},
      {start: 3, end: 5, label: 'Second card', target: 'element'},
    ]}]};
  const requests = [];
  if (textHook) state.hook = {ideas: ['Make it clear', 'Start with the hard part', 'What makes this work?'], chosen: null, source: 'idea'};
  const context = vm.createContext({...helpers, URLSearchParams, Element: Node, document,
    renderTextHook: (_panel, _plan, save, disabled) => { state.hookSave = save; state.hookDisabled = disabled; },
    location: new URL('http://review.example.net/?t=token&version=2'), history: {replaceState() {}},
    window: {addEventListener() {}}, setInterval() {}, clearInterval() {}, setTimeout() {}, clearTimeout() {},
    fetch: async (path, options) => {
      requests.push({path, options});
      const url = new URL(path, 'http://review.example.net/');
      const version = Number(url.searchParams.get('version') || 2);
      const reply = (data, status = 200) => ({ok: status < 400, status, json: async () => structuredClone(data)});
      if (options.method) {
        if (state.gate) await state.gate;
        if (state.fail) return reply({error: {message: state.failMessage
          ?? 'No spoken words in this range'}}, 400);
        const body = JSON.parse(options.body);
        if (body.text_hook) {
          state.hook = {...state.hook, chosen: body.text_hook.text, source: body.text_hook.source};
          const comment = {id: 'c1', kind: 'fix', target: 'element', text: `Text hook: "${body.text_hook.text}"`,
            start_frame: 0, end_frame: null, start_s: 0, end_s: null};
          state.comments.push({...comment, send: {state: 'draft'}});
          return reply(comment, 201);
        }
        if (options.method === 'PATCH') {
          Object.assign(state.comments[0], body);
          return reply(state.comments[0]);
        }
        const comment = {id: 'c1', kind: 'fix', target: 'story', text: body.text || 'Cut this.',
          start_frame: cfr ? body.start_frame : null, end_frame: cfr ? body.end_frame : null,
          start_s: cfr ? body.start_frame / 30 : body.start_s,
          end_s: cfr ? body.end_frame / 30 : body.end_s, send: {state: 'draft'},
          request: {type: 'cut', start: cfr ? body.start_frame / 30 : body.start_s,
            end: cfr ? body.end_frame / 30 : body.end_s,
            words: words.map(({id, segment, text}) => ({id, segment, text}))}};
        state.comments.push(comment);
        return reply(comment, 201);
      }
      if (path.startsWith('api/words')) {
        if (state.failWords) return reply({}, 500);
        return reply({available: state.available, version, words: state.words});
      }
      if (path.startsWith('api/removed')) return reply({available: true, version, ranges: []});
      if (path.startsWith('api/video-map')) {
        if (state.mapGate) await state.mapGate;
        return state.failMap ? reply({}, 500) : reply({version, lanes: state.map});
      }
      return reply({media: {name: 'render.mp4', sha256: `hash${version}`, duration_s: 10, cfr, fps: '30/1'},
        comments: state.comments, previous: null, text: '', media_changed: false, text_hook: state.hook,
        series: project ? {project: true, current: version, newest: 2, threads: [], questions: [],
          rounds: [], versions: state.versions} : null, ...(state.home ? {home: state.home} : {})});
    }});
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.page = {start, loadReview, selectVersion, setOffline, openEdit, saveEdit};'), context);
  await context.page.start();
  const at = async (time) => { element('video').currentTime = time; await element('video').emit('timeupdate'); };
  const click = (id) => element(id).emit('click');
  const mark = async () => { await at(1.01); await click('mark-start'); await at(2.01); await click('mark-end'); };
  const evaluate = (code) => vm.runInContext(code, context);
  return {page: context.page, element, state, requests, at, click, mark, evaluate, document};
}

test('choosing a Text hook refreshes its note so Send notes is enabled', async () => {
  const {state, element, requests, evaluate} = await cutPage({textHook: true});
  assert.equal(state.hookDisabled, false);
  assert.equal(element('send-notes').disabled, true);
  await state.hookSave({text_hook: {text: 'Make it clear', source: 'idea'}});
  assert.equal(evaluate('review.text_hook.chosen'), 'Make it clear');
  assert.equal(element('send-notes').disabled, false);
  const write = requests.findIndex((item) => item.options.method === 'POST');
  assert.ok(requests.slice(write + 1).some((item) => item.path.startsWith('api/review')));
});

test('map maths clamps times and preserves the scrubber time axis', () => {
  assert.deepEqual(helpers.mapBlockPosition({start: 2, end: 4}, 10), {left: 20, width: 20});
  assert.deepEqual(helpers.mapBlockPosition({start: -2, end: 12}, 10), {left: 0, width: 100});
  assert.deepEqual(helpers.mapBlockPosition({start: 10, end: 10}, 10), {left: 100, width: 0});
  assert.deepEqual(helpers.mapBlockPosition({start: 2, end: 4}, null), {left: 0, width: 0});
});

test('map blocks seek and prefill a quoted target without writing or retaining a range', async () => {
  const {element, mark, evaluate, requests, document, at} = await cutPage();
  await mark();
  const block = element('video-map-lanes').querySelector('button');
  assert.ok(block);
  assert.match(block.attributes['aria-label'], /Cards.*A saved card/);
  assert.equal(block.style['--start'], '20%');
  assert.equal(block.style['--width'], '20%');
  await block.emit('click');
  assert.equal(element('video').currentTime, 2);
  assert.equal(element('compose').hidden, false);
  assert.equal(element('compose-text').value, '"<A saved card>" ');
  assert.equal(element('compose-target-element').attributes['aria-pressed'], 'true');
  assert.equal(evaluate('selection'), null);
  assert.equal(document.activeElement, element('compose-text'));
  assert.equal(requests.filter(({options}) => options.method).length, 0);
  assert.equal(block.listeners.pointerdown, undefined);
  const tracks = element('video-map-lanes').querySelectorAll('.video-map-track');
  assert.equal(tracks.length, 1);
  assert.notEqual(tracks[0].children[0].style['--row'], tracks[0].children[1].style['--row']);
  await at(5);
  assert.equal(element('video-map-playhead').style['--position'], '50%');
});

for (const cfr of [true, false]) {
  test(`map clicks seek without replacing an open draft, cfr=${cfr}`, async () => {
    const {element, mark, click, evaluate, requests} = await cutPage({cfr});
    await mark();
    await click('new-comment');
    await click('compose-keep');
    await click('compose-target-caption');
    element('compose-text').value = 'Keep this wording in my draft.';
    const selection = evaluate('JSON.stringify(selection)');
    await element('video-map-lanes').querySelector('button').emit('click');
    assert.equal(element('video').currentTime, 2);
    assert.equal(element('compose').hidden, false);
    assert.equal(element('compose-text').value, 'Keep this wording in my draft.');
    assert.equal(element('compose-target-caption').attributes['aria-pressed'], 'true');
    assert.equal(element('compose-keep').attributes['aria-pressed'], 'true');
    assert.equal(evaluate('JSON.stringify(selection)'), selection);
    assert.equal(requests.filter(({options}) => options.method).length, 0);
  });
}

test('phone map starts collapsed and its summary toggles accessible expansion', async () => {
  const {element, click} = await cutPage();
  assert.equal(element('video-map').attributes['data-expanded'], 'false');
  assert.equal(element('video-map-toggle').attributes['aria-expanded'], 'false');
  assert.match(element('video-map-toggle').textContent, /2 items/);
  await click('video-map-toggle');
  assert.equal(element('video-map').attributes['data-expanded'], 'true');
  assert.equal(element('video-map-toggle').attributes['aria-expanded'], 'true');
  await click('video-map-toggle');
  assert.equal(element('video-map').attributes['data-expanded'], 'false');
  const css = readFileSync(new URL('../../src/yap_studio/review/static/app.css', import.meta.url), 'utf8');
  assert.match(css, /@media \(max-width: 480px\)[\s\S]*?\.video-map\[data-expanded="false"\] \.video-map-lanes\s*\{\s*display: none/);
});

test('map failure and switching versions clear old blocks', async () => {
  const {state, page, element} = await cutPage();
  state.failMap = true;
  await page.selectVersion(1);
  assert.equal(element('video-map-lanes').children.length, 0);
  assert.match(element('video-map-toggle').textContent, /unavailable/i);
  state.failMap = false;
  state.map = [{name: 'Story', items: []}];
  await page.selectVersion(2);
  assert.equal(element('video-map-lanes').querySelectorAll('button').length, 0);
  assert.match(element('video-map-toggle').textContent, /0 items/);
});

test('Mark buttons, touch handles, Clear, X and I/O share the frame selection', async () => {
  const {element, at, click, mark, evaluate, document} = await cutPage();
  await element('video').play();
  await mark();
  assert.equal(element('video').paused, false);
  element('video').pause();
  assert.equal(JSON.stringify(evaluate('completeSelection()')), '{"start_frame":30,"end_frame":61}');
  assert.equal(element('range-panel').hidden, false);
  assert.equal(element('range-length').textContent, '1.0 s');
  assert.equal(element('range-words').textContent, '"so basically yeah" · 3 words');
  for (const [edge, x, start, end] of [['start', 45, 45, 61], ['start', 90, 60, null],
    ['end', 120, 60, 121]]) {
    const event = {pointerId: 1, pointerType: 'touch', button: 0, clientX: x,
      target: element(`selection-${edge}`)};
    await element('scrubber').emit('pointerdown', event);
    await element('scrubber').emit('pointermove', event);
    await element('scrubber').emit('pointerup', event);
    assert.equal(JSON.stringify(evaluate('completeSelection()')),
      JSON.stringify({start_frame: start, end_frame: end}));
  }
  await click('selection-clear');
  assert.equal(evaluate('selection'), null);
  assert.equal(element('range-panel').hidden, true);
  for (const [code, time] of [['KeyI', 1.01], ['KeyO', 2.01]]) {
    await at(time);
    document.listeners.keydown({code, target: element('video'), preventDefault() {}});
  }
  assert.equal(evaluate('completeSelection().end_frame'), 61);
  document.listeners.keydown({code: 'KeyX', target: element('video'), preventDefault() {}});
  assert.equal(evaluate('selection'), null);
  await mark();
  await click('range-play');
  await element('video').emit('seeked');
  assert.equal(element('video').paused, false);
  await at(2.05);
  assert.equal(element('video').paused, true);
  assert.equal(evaluate('currentFrame()'), 60);
});

for (const blocked of ['offline', 'media changed']) {
  test(`I and O still mark a range while ${blocked}`, async () => {
    const {page, element, at, evaluate, document} = await cutPage();
    if (blocked === 'offline') page.setOffline(true);
    else evaluate('review.media_changed = true; renderSelection()');
    for (const [code, time] of [['KeyI', 1.01], ['KeyO', 2.01]]) {
      await at(time);
      document.listeners.keydown({code, target: element('video'), preventDefault() {}});
    }
    assert.equal(JSON.stringify(evaluate('completeSelection()')), '{"start_frame":30,"end_frame":61}');
    assert.equal(element('mark-start').disabled, true);
    assert.equal(element('mark-end').disabled, true);
  });
}

test('range panel hides Cut this without words and reports empty or oversized ranges', async () => {
  for (const options of [{available: false}, {project: false}]) {
    const {mark, element} = await cutPage(options);
    await mark();
    assert.equal(element('range-cut').hidden, true);
    assert.equal(element('range-words').textContent, '');
  }
  const {mark, state, page, element} = await cutPage();
  for (const [covered, message] of [[[], 'No spoken words in this range.'],
    [Array(201).fill(words[0]), 'Select at most 200 words to cut in one note.']]) {
    state.words = covered;
    await page.loadReview();
    await mark();
    assert.equal(element('range-words').textContent, message);
    assert.equal(element('range-cut').disabled, true);
  }
  state.failWords = true;
  await page.loadReview();
  assert.equal(element('range-cut').hidden, true);
  assert.equal(element('new-comment').disabled, false);
});

for (const cfr of [true, false]) {
  test(`Cut this posts once with the optional note and selects the saved comment, cfr=${cfr}`, async () => {
    const {mark, click, element, state, requests, evaluate} = await cutPage({cfr});
    await mark();
    element('compose').hidden = false;
    element('compose-text').value = '  Tighten this.  ';
    let release;
    state.gate = new Promise((resolve) => { release = resolve; });
    const saving = click('range-cut');
    await click('range-cut');
    assert.equal(element('range-cut').disabled, true);
    release();
    await saving;
    const posts = requests.filter(({options}) => options.method === 'POST');
    assert.equal(posts.length, 1);
    assert.equal(posts[0].path, 'api/cut?version=2');
    assert.equal(posts[0].options.headers['X-Yap-Token'], 'token');
    assert.deepEqual(JSON.parse(posts[0].options.body), cfr
      ? {start_frame: 30, end_frame: 61, text: 'Tighten this.'}
      : {start_s: 1.01, end_s: 2.01, text: 'Tighten this.'});
    assert.equal(evaluate('selectedId'), 'c1');
    assert.equal(evaluate('selection'), null);
    assert.equal(element('compose').hidden, true);
    assert.equal(element('toast').textContent, 'Cut saved');
    const row = element('comment-list').children[0];
    assert.equal(row.querySelector('.cut-text').textContent, 'Cut: "so basically yeah" (3 words)');
    assert.ok(row.querySelectorAll('button').some((node) => node.textContent === 'Clear request'));
  });
}

test('Cut this omits empty text, keeps a failed selection and obeys offline, edit and media guards', async () => {
  const {page, mark, click, element, state, evaluate, requests} = await cutPage();
  await mark();
  for (const [set, reset] of [['setOffline(true)', 'setOffline(false)'],
    ['review.media_changed = true; renderReview(review)', 'review.media_changed = false; renderReview(review)'],
    ['editState = {id: "c1"}; renderSelection()', 'editState = null; renderSelection()']]) {
    evaluate(set);
    assert.equal(element('range-cut').disabled, true);
    await click('range-cut');
    evaluate(reset);
  }
  assert.equal(requests.filter(({options}) => options.method === 'POST').length, 0);
  state.fail = true;
  await click('range-cut');
  assert.equal(element('range-error').textContent, 'No spoken words in this range');
  assert.equal(element('status').textContent, 'No spoken words in this range');
  assert.notEqual(evaluate('selection'), null);
  state.fail = false;
  await click('range-cut');
  assert.deepEqual(JSON.parse(requests.find(({options}) => options.method === 'POST').options.body),
    {start_frame: 30, end_frame: 61});
  await page.loadReview();
});

test('Cut this shows plain validation errors only in the range panel', async () => {
  const {mark, click, element, state, evaluate} = await cutPage();
  await mark();
  state.fail = true;
  for (const [message, plain] of [
    ['Bad value for range: No spoken words in this range', 'No spoken words in this range'],
    ['Bad value for end_frame: Invalid edge: select again', 'Invalid edge: select again'],
    ['Could not save: Bad value for range: Try again',
      'Could not save: Bad value for range: Try again'],
  ]) {
    state.failMessage = message;
    await click('range-cut');
    assert.equal(element('range-error').textContent, plain);
    assert.equal(element('range-error').hidden, false);
    assert.equal(element('status').textContent, message);
    assert.notEqual(evaluate('selection'), null);
  }
});

for (const cfr of [true, false]) {
  test(`editing a cut note keeps its own range and request, cfr=${cfr}`, async () => {
    const {page, mark, click, at, element, state, evaluate, requests} = await cutPage({cfr});
    await mark();
    await click('range-cut');
    const comment = structuredClone(state.comments[0]);
    page.openEdit(comment, element('comment-list').children[0]);
    assert.equal(evaluate('selection'), null);
    await at(4.01);
    await click('mark-start');
    await at(5.01);
    await click('mark-end');
    evaluate('editState.textarea.value = "Shorter note"');
    await page.saveEdit();
    const patch = JSON.parse(requests.find(({options}) => options.method === 'PATCH').options.body);
    assert.deepEqual(patch, {text: 'Shorter note', request: comment.request, ...(cfr
      ? {start_frame: comment.start_frame, end_frame: comment.end_frame}
      : {start_s: comment.start_s, end_s: comment.end_s})});
  });
}

test('switching versions drops the previous word list and mark controls follow availability', async () => {
  const {page, mark, element, state, evaluate} = await cutPage();
  await mark();
  state.failWords = true;
  await page.selectVersion(1);
  await mark();
  assert.equal(element('range-cut').hidden, true);
  assert.equal(evaluate('selectedVersion'), 1);
  page.setOffline(true);
  assert.equal(element('mark-start').disabled, true);
  assert.equal(element('mark-end').disabled, true);
});

test('Play range still accepts a comment timed only in seconds on a frame timeline', async () => {
  const {element, evaluate} = await cutPage();
  evaluate('playRange({id: "c2", start_frame: null, end_frame: null, start_s: 1, end_s: 2})');
  await element('video').emit('seeked');
  assert.equal(element('video').currentTime, 1);
  assert.equal(element('video').paused, false);
  assert.equal(evaluate('selectedId'), 'c2');
});

test('variable-rate mouse handles and range playback use the selected seconds', async () => {
  const {element, mark, click, at, evaluate} = await cutPage({cfr: false});
  await mark();
  for (const [edge, x] of [['start', 45], ['end', 30], ['end', 90]]) {
    const event = {pointerId: 2, pointerType: 'mouse', button: 0, clientX: x,
      target: element(`selection-${edge}`)};
    await element('scrubber').emit('pointerdown', event);
    await element('scrubber').emit('pointermove', event);
    await element('scrubber').emit('pointerup', event);
  }
  assert.equal(JSON.stringify(evaluate('completeSelection()')), '{"start_s":1.5,"end_s":3}');
  await click('range-play');
  await element('video').emit('seeked');
  assert.equal(element('video').currentTime, 1.5);
  assert.equal(element('video').paused, false);
  await at(3.01);
  assert.equal(element('video').paused, true);
  assert.equal(element('video').currentTime, 3);
});

test('a marked part takes a note with Change this, Keep this, or a note on Cut this', async () => {
  const {element, requests, mark, click, evaluate} = await cutPage();
  await mark();
  await click('range-change');
  assert.match(element('range-error').textContent, /Say what should change/);
  assert.equal(requests.filter((item) => item.options.method === 'POST').length, 0, 'no empty change note');
  element('range-note').value = 'Slow this part down a little.';
  await click('range-change');
  const change = requests.filter((item) => item.options.method === 'POST').at(-1);
  assert.match(change.path, /^api\/comments/);
  assert.deepEqual(JSON.parse(change.options.body), {kind: 'fix', target: null,
    text: 'Slow this part down a little.', start_frame: 30, end_frame: 61});
  assert.equal(element('range-note').value, '');
  assert.equal(evaluate('selection'), null);
  await mark();
  await click('range-keep');
  const keep = JSON.parse(requests.filter((item) => item.options.method === 'POST').at(-1).options.body);
  assert.deepEqual([keep.kind, keep.text], ['keep', 'Keep this part as it is.']);
  await mark();
  element('range-note').value = 'This whole bit drags.';
  await click('range-cut');
  const cut = requests.filter((item) => item.options.method === 'POST').at(-1);
  assert.match(cut.path, /^api\/cut/);
  assert.equal(JSON.parse(cut.options.body).text, 'This whole bit drags.');
});

test('an unsent marked-part note stays with the version it was written on', async () => {
  const {page, element, mark} = await cutPage();
  await mark();
  element('range-note').value = 'Slow this part down.';
  await page.selectVersion(1);
  assert.equal(element('range-note').value, '');
});

test('a final requested from Yap Studio home updates an open story-cut review', async () => {
  const {page, element, state, evaluate} = await cutPage();
  state.versions = [{number: 1, final: false}, {number: 2, final: false}];
  state.home = {url: 'http://127.0.0.1:9/', phone_url: null, listening: true, delivered: [],
    final_requested: false};
  await page.loadReview(2);
  assert.equal(element('make-final').hidden, false);
  assert.equal(element('make-final').textContent, 'Make the final video');
  state.home = {...state.home, final_requested: true};
  await evaluate('refreshReview()');
  assert.equal(element('make-final').textContent, 'Claude is making the final');
  assert.equal(element('make-final').disabled, true);
});
