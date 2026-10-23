import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as helpers from '../../src/yap_studio/review/static/timeline.mjs';
const words = [];

test('picture geometry handles letterboxes, pillarboxes and missing video size', () => {
  const box = {left: 10, top: 100, width: 370, height: 540};
  assert.deepEqual(helpers.pictureRect(box, 1080, 1920),
    {left: 43.125, top: 100, width: 303.75, height: 540});
  assert.deepEqual(helpers.pictureRect({left: 0, top: 0, width: 400, height: 400}, 1600, 900),
    {left: 0, top: 87.5, width: 400, height: 225});
  assert.deepEqual(helpers.pictureRect(box, 0, 0), box);
  assert.equal(helpers.pinAt(box, 9, 200), null);
  assert.equal(helpers.pinAt(box, 200, 641), null);
  assert.deepEqual(helpers.pinAt(box, 165.555, 267.65), {x: 0.4204, y: 0.3105});
  assert.deepEqual(helpers.pinAt(box, 380, 640), {x: 1, y: 1});
  assert.deepEqual(helpers.pinPoint(box, {x: 0.5, y: 0.5}), {left: 195, top: 370});
});

test('pin text, bodies, half-open spans and saved frame visibility', () => {
  const fps = helpers.parseFps('30000/1001');
  const pin = {x: 0.4213, y: 0.3105, frame: 447, time_s: 14.9149};
  assert.equal(helpers.pinText(pin), 'Pinned 42% across, 31% down');
  assert.equal(helpers.pinText({x: 0.425, y: 0.125}), 'Pinned 43% across, 13% down');
  assert.deepEqual(helpers.pinBody(pin, fps, 448, 15), {x: pin.x, y: pin.y, frame: 448});
  assert.deepEqual(helpers.pinBody(pin, null, null, 1.23456789),
    {x: pin.x, y: pin.y, time_s: 1.23456789});
  const comment = {start_frame: 440, end_frame: 460, start_s: 14, end_s: 16, pin};
  for (const [frame, inside, shown] of [[439, false, false], [440, true, false],
    [447, true, true], [459, true, false], [460, false, false]]) {
    assert.equal(helpers.pinInSpan(comment, fps, frame, 0), inside);
    assert.equal(helpers.pinShown(comment, fps, frame, 0), shown);
  }
  assert.equal(helpers.pinInSpan({...comment, end_frame: null}, fps, 441, 0), false);
  const variable = {...comment, pin: {...pin, frame: null}};
  assert.equal(helpers.pinShown(variable, null, null, pin.time_s + 0.01), true);
  assert.equal(helpers.pinShown(variable, null, null, pin.time_s + 0.02), false);
  assert.equal(helpers.pinInSpan(variable, null, null, 16), true);
  assert.equal(helpers.pinInSpan(variable, null, null, 16.001), false);
  assert.equal(helpers.pinInSpan({...variable, end_s: null}, null, null, 14.00049), true);
  assert.equal(helpers.pinInSpan({...variable, end_s: null}, null, null, 14.00051), false);
  assert.equal(helpers.pinShown({}, fps, 447, 0), false);
});

async function pinPage({cfr = true, project = true, available = true, captions = null} = {}) {
  let document;
  class Node {
    constructor(tag = 'div') {
      Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, listeners: {},
        dataset: {}, value: '', textContent: '', hidden: false, checked: false, paused: true,
        duration: 10, currentTime: 0, readyState: 1,
        style: {values: {}, setProperty(key, value) { this.values[key] = value; }}, classList: {toggle() {}, add() {}, remove() {}}});
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
    scrollIntoView() {}
    setSelectionRange() {}
    getBoundingClientRect() { return this.rect ?? {left: 0, top: 0, width: 300, height: 40}; }
    setPointerCapture(id) { this.capture = id; }
    hasPointerCapture(id) { return this.capture === id; }
    releasePointerCapture() { this.capture = null; }
    pause() { if (!this.paused) { this.paused = true; void this.emit('pause'); } }
    play() { this.paused = false; return this.emit('play'); }
    load() { this.loads = (this.loads ?? 0) + 1; this.currentTime = 0; this.readyState = 0; }
  }
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  document = {getElementById: element, createElement: (tag) => new Node(tag), activeElement: null,
    visibilityState: 'visible', listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }};
  element('players').parent = new Node('main');
  const wrap = new Node();
  wrap.className = 'video-wrap';
  wrap.rect = {left: 10, top: 100, width: 370, height: 540};
  wrap.append(element('video'), element('pin-layer'));
  element('pin-layer').append(element('pin-marker'));
  element('pin-marker').className = 'pin-marker';
  Object.assign(element('video'), {videoWidth: 1080, videoHeight: 1920, rect: wrap.rect});
  element('compare-video').parent = wrap;
  element('pin-layer').hidden = true;
  element('compose-pin-remove').hidden = true;
  for (const edge of ['start', 'end']) element(`selection-${edge}`).className = `selection-handle ${edge}`;
  element('compose').hidden = true;
  element('caption-fix').hidden = true;
  element('offline-banner').hidden = true;
  element('approval-confirmation').hidden = true;
  const state = {comments: [], available, words, fail: false, failWords: false, gate: null};
  const requests = [];
  const timers = new Map();
  const intervals = new Map();
  let clock = 0;
  let timerId = 0;
  const context = vm.createContext({...helpers, URLSearchParams, AbortController, DOMException, Element: Node, document,
    performance: {now: () => 0},
    location: new URL('http://review.example.net/?t=token&version=2'), history: {replaceState() {}},
    window: {addEventListener() {}}, confirm: () => true,
    setInterval(fn) { intervals.set(++timerId, fn); return timerId; },
    clearInterval(id) { intervals.delete(id); },
    setTimeout(fn, delay) { timers.set(++timerId, {fn, at: clock + delay}); return timerId; },
    clearTimeout(id) { timers.delete(id); }, requestAnimationFrame(fn) { fn(); return 1; },
    fetch: async (path, options) => {
      requests.push({path, options});
      if (state.readGate && path.startsWith('api/review')) await state.readGate;
      if (state.unreachable) throw new TypeError('Network unavailable');
      const url = new URL(path, 'http://review.example.net/');
      const version = Number(url.searchParams.get('version') || 2);
      const reply = (data, status = 200) => ({ok: status < 400, status, json: async () => structuredClone(data)});
      if (options.method) {
        if (state.gate) await state.gate;
        if (state.fail) return reply({error: {message: state.failMessage
          ?? 'No spoken words in this range'}}, state.failStatus ?? 400);
        const body = JSON.parse(options.body);
        if (path === 'api/approve') {
          state.approved = {version: body.version};
          return reply({});
        }
        if (url.pathname === '/api/rounds') {
          state.rounds = [{id: 'r1', status: 'sending', version, comments: ['c1']}];
          return reply(state.rounds[0], 201);
        }
        if (options.method === 'PATCH') {
          Object.assign(state.comments[0], body);
          return reply(state.comments[0]);
        }
        const comment = {id: `c${state.comments.length + 1}`, pin: body.pin ? {...body.pin, frame: body.pin.frame ?? null, time_s: cfr ? body.pin.frame / 30 : body.pin.time_s} : undefined, unused: 'c1', kind: 'fix', target: body.target ?? null, text: body.text || 'Cut this.',
          start_frame: cfr ? body.start_frame : null, end_frame: cfr ? body.end_frame ?? null : null,
          start_s: cfr ? body.start_frame / 30 : body.start_s,
          end_s: cfr ? (body.end_frame == null ? null : body.end_frame / 30) : body.end_s ?? null, send: {state: 'draft'},
          request: body.request ?? null};
        state.comments.push(comment);
        return reply(comment, 201);
      }
      if (path.startsWith('api/words')) {
        if (state.failWords) return reply({}, 500);
        return reply({available: state.available, version, words: state.words});
      }
      if (path.startsWith('api/captions')) return reply(captions
        ? {...captions, version} : {available: false, version, phrases: []});
      if (path.startsWith('api/removed')) return reply({available: true, version, ranges: []});
      if (path.startsWith('api/map')) return reply({status: 'exact', frame: 60, time_s: 2, candidates: []});
      return reply({media: {name: 'render.mp4', sha256: `hash${version}`, duration_s: 10, cfr, fps: '30/1'},
        comments: state.comments, previous: null, text: '', media_changed: state.writing ?? false,
        series: project ? {project: true, current: version, newest: 2, threads: [], questions: [],
          approved: state.approved, rounds: state.rounds ?? [], versions: [{number: 1}, {number: 2}]} : null});
    }});
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.page = {start, loadReview, selectVersion, setOffline, openEdit, saveEdit, cancelEdit, seekComment, seekFrame, seekTime, openThread};'), context);
  await context.page.start();
  const at = async (time) => { element('video').currentTime = time; await element('video').emit('timeupdate'); };
  const click = (id) => element(id).emit('click');
  const mark = async () => { await at(1.01); await click('mark-start'); await at(2.01); await click('mark-end'); };
  const evaluate = (code) => vm.runInContext(code, context);
  const tick = async (ms) => {
    clock += ms;
    for (const [id, timer] of [...timers]) {
      if (timer.at <= clock && timers.delete(id)) await timer.fn();
    }
  };
  return {page: context.page, element, state, requests, at, click, mark, evaluate, document, tick, intervals};
}

test('metadata and first frame have separate deadlines and clear after a frame loads', async () => {
  const page = await pinPage();
  await page.tick(9999);
  assert.equal(page.element('codec-error').hidden, true);
  await page.tick(1);
  assert.match(page.element('player-message').textContent, /load stalled/u);
  assert.equal(page.element('codec-error').hidden, false);
  await page.click('player-retry');
  await page.element('video').emit('loadedmetadata');
  await page.tick(10000);
  assert.equal(page.element('codec-error').hidden, true);
  await page.tick(5000);
  assert.match(page.element('player-message').textContent, /first frame/u);
  await page.element('video').emit('loadeddata');
  assert.equal(page.element('codec-error').hidden, true);
  await page.tick(30000);
  assert.equal(page.element('codec-error').hidden, true);
});

test('Retry reloads the source and seeks to the same time when metadata returns', async () => {
  const page = await pinPage();
  const video = page.element('video');
  video.currentTime = 4.25;
  video.error = {code: 3};
  await video.emit('error');
  assert.match(page.element('player-message').textContent, /browser cannot play/u);
  const source = video.src;
  await page.click('player-retry');
  assert.equal(video.loads, 1);
  assert.equal(video.src, source);
  await video.emit('loadedmetadata');
  assert.equal(video.currentTime, 4.25);
});

test('network errors retry once and use one offline banner with Retry', async () => {
  const page = await pinPage();
  const video = page.element('video');
  page.state.unreachable = true;
  video.currentTime = 2.5;
  video.error = {code: 2};
  await video.emit('error');
  assert.equal(video.loads, 1);
  assert.equal(page.element('offline-player').hidden, false);
  assert.match(page.element('offline-player-message').textContent, /could not reach the Mac/u);
  await video.emit('error');
  assert.equal(video.loads, 1);
  assert.equal(page.element('offline-banner').hidden, false);
  assert.equal(page.element('codec-error').hidden, true);
  assert.match(page.element('offline-player-message').textContent, /could not reach the Mac/u);
  page.state.unreachable = false;
  await page.click('offline-player-retry');
  assert.equal(video.loads, 2);
  await video.emit('loadedmetadata');
  assert.equal(video.currentTime, 2.5);
  await video.emit('loadeddata');
  assert.equal(page.element('codec-error').hidden, true);
});

test('a reachability timeout still marks the page offline', async () => {
  const page = await pinPage();
  let release;
  page.state.readGate = new Promise((resolve) => { release = resolve; });
  const pending = page.tick(10000);
  await page.tick(3000);
  page.state.readGate = null;
  release();
  await pending;
  assert.equal(page.element('offline-banner').hidden, false);
  assert.match(page.element('offline-player-message').textContent, /could not reach the Mac/u);
});

test('changed media reports a file still being written and does not retry automatically', async () => {
  const page = await pinPage();
  page.state.writing = true;
  page.element('video').error = {code: 4};
  await page.element('video').emit('error');
  assert.match(page.element('player-message').textContent, /file is still being written/u);
  assert.equal(page.element('video').loads, undefined);
});

test('switching away from an approval hides its confirmation and resumes round polling', async () => {
  const page = await pinPage();
  await page.click('approve-version');
  assert.equal(page.element('approval-confirmation').hidden, false);
  assert.equal(page.intervals.size, 0);
  await page.page.selectVersion(1);
  assert.equal(page.element('approval-confirmation').hidden, true);
  assert.equal(page.intervals.size, 1);
  await page.click('new-comment');
  page.element('compose-text').value = 'Please change this.';
  await page.click('compose-submit');
  await page.click('send-notes');
  page.state.rounds[0].step = 'Checking the whole story';
  const before = page.requests.length;
  await [...page.intervals.values()][0]();
  assert.ok(page.requests.length > before);
  assert.equal(page.evaluate('review.series.rounds[0].step'), 'Checking the whole story');
  assert.equal(page.element('round-list').children[0].children[0].textContent, 'Checking the whole story');
});

for (const retry of [false, true]) {
  test(`approval cancels player deadlines (retry: ${retry})`, async () => {
    const page = await pinPage();
    if (retry) await page.click('player-retry');
    await page.click('approve-version');
    const before = page.requests.length;
    page.state.unreachable = true;
    await page.tick(30000);
    await page.element('video').emit('error');
    assert.equal(page.requests.length, before);
    assert.equal(page.element('offline-banner').hidden, true);
    assert.equal(page.element('download-video').href, 'api/download?version=2&t=token');
  });
}

for (const failure of [false, true]) {
  test(`approval ignores a pending recovery probe (failure: ${failure})`, async () => {
    const page = await pinPage();
    let release;
    page.state.readGate = new Promise((resolve) => { release = resolve; });
    const pending = page.tick(10000);
    await page.click('approve-version');
    page.state.unreachable = failure;
    page.state.writing = !failure;
    release();
    await pending;
    assert.equal(page.element('approval-confirmation').hidden, false);
    assert.equal(page.element('offline-banner').hidden, true);
    assert.equal(page.element('codec-error').hidden, true);
    assert.equal(page.element('download-video').href, 'api/download?version=2&t=token');
  });
}

test('version changes cancel old deadlines and ignore an old recovery response', async () => {
  const page = await pinPage();
  await page.tick(5000);
  await page.page.selectVersion(1);
  await page.tick(5000);
  assert.equal(page.element('codec-error').hidden, true);
  await page.element('video').emit('loadedmetadata');
  await page.element('video').emit('loadeddata');
  await page.tick(20000);
  assert.equal(page.element('codec-error').hidden, true);
});

test('late metadata keeps the first frame deadline and stale errors cannot cover a new version', async () => {
  const page = await pinPage();
  let release;
  page.state.readGate = new Promise((resolve) => { release = resolve; });
  const pending = page.tick(10000);
  await page.element('video').emit('loadedmetadata');
  release();
  await pending;
  assert.equal(page.element('codec-error').hidden, true);
  await page.tick(5000);
  assert.match(page.element('player-message').textContent, /first frame/u);
  page.state.readGate = new Promise((resolve) => { release = resolve; });
  const oldError = page.element('video').emit('error');
  page.state.readGate = null;
  await page.page.selectVersion(1);
  release();
  await oldError;
  assert.equal(page.element('codec-error').hidden, true);
});

const captionTargets = {available: true, width: 1080, height: 1920, phrases: [
  {start: 1, end: 2, text: 'blogging well', boxes: [{x: 100, y: 1200, w: 800, h: 200}],
    words: [{id: 't1:0', text: 'blogging', heard: 'blogging', start: 1, end: 1.3, repeats: 2},
      {id: 't1:1', text: 'well', heard: 'well', start: 1.4, end: 1.7, repeats: 1}]},
]};

test('caption hit testing uses picture pixels and half-open time windows', () => {
  const rect = {left: 20, top: 30, width: 270, height: 480};
  const hit = (x, y, time = 1) => helpers.captionAt(captionTargets, time, rect, x, y);
  assert.equal(hit(50, 340)?.text, 'blogging well');
  assert.equal(hit(50, 340, 2), null);
  assert.equal(hit(50, 340, 0.99), null);
  assert.equal(hit(10, 340), null);
  assert.equal(hit(50, 100), null);
});

test('caption hit testing uses only pose boxes active at the displayed frame', () => {
  const moving = {...captionTargets, phrases: [{...captionTargets.phrases[0], boxes: [
    {x: 100, y: 1200, w: 200, h: 200, start_frame: 30, end_frame: 36},
    {x: 700, y: 1200, w: 200, h: 200, start_frame: 36, end_frame: 60},
  ]}]};
  const rect = {left: 20, top: 30, width: 270, height: 480};
  const hit = (x, frame) => helpers.captionAt(moving, 1.1, rect, x, 340, frame);
  assert.ok(hit(50, 30));
  assert.ok(hit(50, 35));
  assert.equal(hit(50, 36), null);
  assert.equal(hit(210, 35), null);
  assert.ok(hit(210, 36));
  assert.ok(hit(210, 59));
  assert.equal(hit(210, 60), null);
  assert.equal(hit(50, null), null);
  assert.equal(hit(50, undefined), null);
  assert.ok(helpers.captionAt(captionTargets, 1.1, rect, 50, 340));
});

test('a tap on an inactive caption pose pins a note; an active pose opens the caption', async () => {
  const moving = {...captionTargets, phrases: [{...captionTargets.phrases[0], boxes: [
    {...captionTargets.phrases[0].boxes[0], start_frame: 36, end_frame: 60},
  ]}]};
  const {element, at, click} = await pinPage({captions: moving});
  await at(1.1);
  await element('video').emit('click', {clientX: 195, clientY: 465});
  assert.equal(element('compose').hidden, false);
  assert.equal(element('caption-fix').hidden, true);
  await click('compose-cancel');
  await at(1.2);
  await element('video').emit('click', {clientX: 195, clientY: 465});
  assert.equal(element('compose').hidden, true);
  assert.equal(element('caption-fix').hidden, false);
});

test('merged caption chips are disabled with a visible reason and keyboard focus skips them', async () => {
  const reason = 'This caption joins several words. Fix it with a note.';
  for (const allMerged of [false, true]) {
    const targets = structuredClone(captionTargets);
    Object.assign(targets.phrases[0].words[0], {text: 'NYC', editable: false, reason});
    if (allMerged) targets.phrases[0].words = targets.phrases[0].words.slice(0, 1);
    const {element, at, click, document, requests} = await pinPage({captions: targets});
    await at(1.1);
    await click('fix-caption');
    const children = element('caption-words').children;
    const merged = children.find((node) => node.tagName === 'BUTTON');
    assert.equal(merged.disabled, true);
    const explanation = children.find((node) => node.textContent === reason);
    assert.ok(explanation);
    assert.equal(merged.attributes['aria-describedby'], explanation.id);
    assert.equal(document.activeElement, allMerged ? element('caption-cancel')
      : children.find((node) => node.tagName === 'BUTTON' && !node.disabled));
    await merged.emit('click');
    assert.equal(element('caption-to').disabled, true);
    assert.equal(element('caption-save').disabled, true);
    assert.equal(requests.some((item) => item.path.startsWith('api/caption-fixes')), false);
    if (!allMerged) {
      await children.find((node) => node.tagName === 'BUTTON' && !node.disabled).emit('click');
      assert.equal(element('caption-to').disabled, false);
      assert.equal(explanation.attributes['aria-pressed'], undefined);
    }
  }
});

test('a paused caption tap opens chips, saves a word id, and keeps outside pin taps', async () => {
  const {element, at, requests, document} = await pinPage({captions: captionTargets});
  await at(1.1);
  await element('video').emit('click', {clientX: 195, clientY: 465});
  assert.equal(element('caption-fix').hidden, false);
  assert.equal(element('compose').hidden, true);
  const chips = element('caption-words').children;
  assert.equal(chips.length, 2);
  assert.equal(document.activeElement, chips[0]);
  await chips[0].emit('click');
  assert.equal(document.activeElement, element('caption-to'));
  assert.equal(element('caption-every-label').hidden, false);
  element('caption-to').value = 'logging';
  await element('caption-save').emit('click');
  const save = requests.find((item) => item.path.startsWith('api/caption-fixes'));
  assert.deepEqual(JSON.parse(save.options.body),
    {word: 't1:0', start: 1, to: 'logging', every_time: false});
  assert.match(save.path, /version=2/u);
  assert.equal(element('caption-fix').hidden, true);
  await element('video').emit('click', {clientX: 195, clientY: 208});
  assert.equal(element('compose').hidden, false);
});

test('a repeated caption sends the selected occurrence start with its shared word id', async () => {
  const repeated = structuredClone(captionTargets);
  repeated.phrases.push({...repeated.phrases[0], start: 4, end: 5,
    words: [{...repeated.phrases[0].words[0], start: 4, end: 4.5}]});
  const {element, at, click, requests} = await pinPage({captions: repeated});
  await at(4.1);
  await click('fix-caption');
  await element('caption-words').children[0].emit('click');
  element('caption-to').value = 'logging';
  await click('caption-save');
  const save = requests.find((item) => item.path.startsWith('api/caption-fixes'));
  assert.deepEqual(JSON.parse(save.options.body),
    {word: 't1:0', start: 4, to: 'logging', every_time: false});
});

test('caption keyboard access, cancel, Escape, and repeated heard word scope', async () => {
  const {element, at, click, document, requests} = await pinPage({captions: captionTargets});
  await at(1.1);
  assert.equal(element('fix-caption').hidden, false);
  await click('fix-caption');
  await element('caption-words').children[1].emit('click');
  assert.equal(element('caption-every-label').hidden, true);
  await click('caption-cancel');
  assert.equal(element('caption-fix').hidden, true);
  assert.equal(document.activeElement, element('fix-caption'));
  await click('fix-caption');
  await document.listeners.keydown({target: element('caption-to'), code: 'Escape', preventDefault() {}});
  assert.equal(element('caption-fix').hidden, true);
  assert.equal(requests.some((item) => item.path.startsWith('api/caption-fixes')), false);
  await click('fix-caption');
  await element('caption-words').children[0].emit('click');
  element('caption-to').value = 'logging';
  element('caption-every').checked = true;
  await click('caption-save');
  const save = requests.find((item) => item.path.startsWith('api/caption-fixes'));
  assert.equal(JSON.parse(save.options.body).every_time, true);
});

test('caption failed save retains text and blocks duplicate saves and version switches', async () => {
  const {element, at, click, state, page, requests} = await pinPage({captions: captionTargets});
  await at(1.1);
  await click('fix-caption');
  await element('caption-words').children[0].emit('click');
  element('caption-to').value = 'logging';
  let release;
  state.gate = new Promise((resolve) => { release = resolve; });
  state.fail = true;
  const saving = click('caption-save');
  await click('caption-save');
  await page.selectVersion(1);
  assert.equal(requests.filter((item) => item.path.startsWith('api/caption-fixes')).length, 1);
  assert.equal(element('caption-to').disabled, true);
  assert.equal(element('caption-every').disabled, true);
  release();
  await saving;
  assert.equal(element('caption-fix').hidden, false);
  assert.equal(element('caption-to').value, 'logging');
  assert.ok(element('caption-error').textContent);
  assert.equal(element('caption-to').disabled, false);
  page.setOffline(true);
  assert.equal(element('caption-fix').hidden, false);
  assert.equal(element('caption-to').value, 'logging');
});

test('caption every time conflict shows the one-off message and keeps the editor open', async () => {
  const {element, at, click, state} = await pinPage({captions: captionTargets});
  await at(1.1);
  await click('fix-caption');
  await element('caption-words').children[0].emit('click');
  element('caption-to').value = 'logging';
  element('caption-every').checked = true;
  state.fail = true;
  state.failStatus = 409;
  state.failMessage = 'This word has a one-off fix. Undo it first.';
  await click('caption-save');
  assert.equal(element('caption-error').textContent, state.failMessage);
  assert.equal(element('caption-fix').hidden, false);
  assert.equal(element('caption-to').value, 'logging');
  assert.equal(element('caption-every').checked, true);
  assert.equal(element('caption-save').disabled, false);
});

test('a playing picture tap pauses first; a paused tap opens a pinned note before typing', async () => {
  const {element, at, evaluate, document} = await pinPage();
  const spot = {clientX: 195, clientY: 370};
  await element('video').play();
  await element('video').emit('click', spot);
  assert.equal(element('video').paused, true);
  assert.equal(element('compose').hidden, true);
  assert.equal(element('pin-layer').hidden, true);
  await at(1.25);
  await element('video').emit('click', spot);
  assert.equal(element('video').paused, true);
  assert.equal(element('compose').hidden, false);
  assert.equal(element('pin-layer').hidden, false);
  assert.equal(element('compose-pin-text').textContent, 'Pinned 50% across, 50% down');
  assert.equal(evaluate('currentFrame()'), 37);
  assert.equal(document.activeElement, element('compose-text'));
  element('play-pause').focus();
  await element('video').emit('click', {clientX: 195, clientY: 208});
  assert.equal(element('compose-pin-text').textContent, 'Pinned 50% across, 20% down');
  assert.equal(document.activeElement, element('play-pause'));
});

test('Comment ignores extra click fields and opens a focused note without a pin', async () => {
  const {element, evaluate, document} = await pinPage();
  await element('new-comment').emit('click', {pin: {x: 0.5, y: 0.5}, focus: false});
  assert.equal(element('compose').hidden, false);
  assert.equal(evaluate('composePin'), null);
  assert.equal(element('pin-layer').hidden, true);
  assert.equal(element('compose-pin-remove').hidden, true);
  assert.equal(document.activeElement, element('compose-text'));
});

test('a paused black bar tap plays without opening a note or pinning', async () => {
  const {element} = await pinPage();
  await element('video').emit('click', {clientX: 20, clientY: 370});
  assert.equal(element('video').paused, false);
  assert.equal(element('compose').hidden, true);
  assert.equal(element('pin-layer').hidden, true);
});

for (const blocked of ['offline', 'media_changed', 'switchingVersion', 'composeBusy', 'roundBusy',
  'releaseBusy', 'answerBusy', 'cutBusy', 'activeWrites']) {
  test(`paused picture cannot start a note while ${blocked}`, async () => {
    const {page, element, evaluate} = await pinPage();
    if (blocked === 'offline') page.setOffline(true);
    else if (blocked === 'media_changed') evaluate('review.media_changed = true');
    else evaluate(`${blocked} = ${blocked === 'activeWrites' ? 1 : true}`);
    await element('video').emit('click', {clientX: 195, clientY: 370});
    assert.equal(element('compose').hidden, true);
    assert.equal(element('pin-layer').hidden, true);
    assert.equal(evaluate('composePin'), null);
  });
}

test('rendering an unchanged draft leaves pin text, visibility and position alone', async () => {
  const {element, click, evaluate} = await pinPage();
  await click('new-comment');
  await element('video').emit('click', {clientX: 195, clientY: 370});
  const writes = [];
  for (const [id, key] of [['compose-pin-text', 'textContent'], ['compose-pin-remove', 'hidden'],
    ['pin-layer', 'hidden']]) {
    const node = element(id);
    let value = node[key];
    Object.defineProperty(node, key, {get: () => value, set(next) {
      writes.push(`${id}.${key}`); value = next;
    }});
  }
  const style = element('pin-marker').style;
  const setProperty = style.setProperty.bind(style);
  style.setProperty = (key, value) => { writes.push(key); setProperty(key, value); };
  evaluate('renderPin(); setDisplayedTime(1.25); setDisplayedTime(2.25)');
  assert.deepEqual(writes, []);
  evaluate('composePin = {x: 0.6, y: 0.5}; renderPin()');
  assert.deepEqual(writes, ['compose-pin-text.textContent', '--pin-left']);
});

test('watching and scrubbing do not pin; compose places, moves, removes and saves the displayed frame', async () => {
  const {element, click, at, evaluate, requests} = await pinPage();
  await click('video');
  assert.equal(element('video').paused, false);
  assert.equal(element('pin-layer').hidden, true);
  const pointer = {pointerId: 1, button: 0, clientX: 120};
  await element('scrubber').emit('pointerdown', pointer);
  await element('scrubber').emit('pointerup', pointer);
  assert.equal(element('pin-layer').hidden, true);
  await click('new-comment');
  await at(1.25);
  await element('video').emit('click', {clientX: 195, clientY: 370});
  assert.equal(element('video').paused, true);
  assert.equal(element('pin-layer').hidden, false);
  assert.equal(element('compose-pin-text').textContent, 'Pinned 50% across, 50% down');
  assert.equal(element('compose-pin-remove').hidden, false);
  assert.equal(element('pin-marker').style.values['--pin-left'], '185px');
  await element('video').emit('click', {clientX: 20, clientY: 370});
  assert.equal(element('compose-pin-text').textContent, 'Pinned 50% across, 50% down');
  await element('video').emit('click', {clientX: 270, clientY: 208});
  assert.equal(element('compose-pin-text').textContent, 'Pinned 75% across, 20% down');
  await click('compose-pin-remove');
  assert.equal(element('pin-layer').hidden, true);
  assert.equal(element('compose-pin-remove').hidden, true);
  await element('video').emit('click', {clientX: 195, clientY: 370});
  await at(2.25);
  assert.equal(element('pin-layer').hidden, false);
  element('compose-text').value = 'Move this card';
  await click('compose-submit');
  const post = requests.find(({options}) => options.method === 'POST');
  assert.deepEqual(JSON.parse(post.options.body).pin, {x: 0.5, y: 0.5, frame: 67});
  assert.equal(evaluate('currentFrame()'), 67);
  assert.equal(element('comment-list').querySelector('.pin-text').textContent,
    'Pinned 50% across, 50% down');
});

test('draft range refuses a pin outside the span and keeps the box open', async () => {
  const {element, click, mark, at, requests} = await pinPage();
  await mark();
  await click('new-comment');
  await element('video').emit('click', {clientX: 195, clientY: 370});
  await at(4);
  element('compose-text').value = 'This range';
  await click('compose-submit');
  assert.equal(element('compose-error').textContent,
    'Move to a frame inside the range, or remove the pin.');
  assert.equal(element('compose').hidden, false);
  assert.equal(requests.some(({options}) => options.method === 'POST'), false);
});

test('saved range selection seeks the pin and hides it off frame, playing, offline and on media changes', async () => {
  const {page, element, state, at, evaluate} = await pinPage();
  const comment = {id: 'c1', kind: 'fix', target: 'element', text: 'Here',
    start_frame: 30, end_frame: 90, start_s: 1, end_s: 3,
    pin: {x: 0.5, y: 0.5, frame: 60, time_s: 2}};
  state.comments.push(comment);
  await page.loadReview();
  page.seekComment(comment);
  assert.equal(element('video').currentTime, helpers.frameMid(60, helpers.parseFps('30/1')));
  assert.equal(element('pin-layer').hidden, false);
  await at(2.05);
  assert.equal(element('pin-layer').hidden, true);
  await at(2.01);
  assert.equal(element('pin-layer').hidden, false);
  await element('video').play();
  assert.equal(element('pin-layer').hidden, true);
  element('video').pause();
  assert.equal(element('pin-layer').hidden, false);
  page.setOffline(true);
  assert.equal(element('pin-layer').hidden, true);
  page.setOffline(false);
  evaluate('review.media_changed = true; renderReview(review)');
  assert.equal(element('pin-layer').hidden, true);
});

for (const cfr of [true, false]) {
  test(`earlier range thread seeks its pin in compare with cfr=${cfr}`, async () => {
    const {page, element, evaluate, requests} = await pinPage({cfr});
    evaluate(`review.series.versions.forEach((version) => Object.assign(version,
      {cfr: ${cfr}, fps: '30/1', duration_s: 10}))`);
    const thread = {id: 'c1', version: 1, kind: 'fix', target: 'element', text: 'Here',
      start_frame: cfr ? 30 : null, end_frame: cfr ? 90 : null, start_s: 1, end_s: 3,
      pin: {x: 0.5, y: 0.5, frame: cfr ? 60 : null, time_s: 2.123456}};
    await page.openThread(thread);
    assert.equal(element('compare-video').currentTime,
      cfr ? helpers.frameMid(60, helpers.parseFps('30/1')) : thread.pin.time_s);
    const map = new URL(requests.find(({path}) => path.startsWith('api/map')).path,
      'http://review.example.net/');
    assert.equal(map.searchParams.get(cfr ? 'frame' : 'time'),
      String(cfr ? 60 : thread.pin.time_s));
    await page.openThread({...thread, pin: null});
    assert.equal(element('compare-video').currentTime,
      cfr ? helpers.frameMid(30, helpers.parseFps('30/1')) : thread.start_s);
  });
}

test('edit omits untouched pin, moves it on the displayed frame, and sends null to remove', async () => {
  const {page, element, state, click, at, evaluate, requests} = await pinPage();
  const comment = {id: 'c1', kind: 'fix', target: 'element', text: 'Here', request: null,
    start_frame: 30, end_frame: 90, start_s: 1, end_s: 3,
    pin: {x: 0.5, y: 0.5, frame: 60, time_s: 2}};
  state.comments.push(comment);
  await page.loadReview();
  const edit = () => page.openEdit(state.comments[0], element('comment-list').children[0]);
  edit();
  assert.equal(element('pin-layer').hidden, false);
  await page.saveEdit();
  assert.equal('pin' in JSON.parse(requests.find(({options}) => options.method === 'PATCH').options.body), false);
  edit();
  await element('video').emit('click', {clientX: 195, clientY: 208});
  await at(2.25);
  await page.saveEdit();
  const patches = () => requests.filter(({options}) => options.method === 'PATCH').map(({options}) => JSON.parse(options.body));
  assert.deepEqual(patches()[1].pin, {x: 0.5, y: 0.2, frame: 67});
  edit();
  await evaluate('editState.pinRemove.emit("click")');
  await page.saveEdit();
  assert.equal(patches()[2].pin, null);
  assert.equal(element('pin-layer').hidden, true);
  await click('new-comment');
});

test('marker drag clamps to picture, absorbs its click and arrow keys move without stepping', async () => {
  const {element, click, evaluate} = await pinPage();
  await click('new-comment');
  await element('video').emit('click', {clientX: 195, clientY: 370});
  const marker = element('pin-marker');
  await marker.emit('pointerdown', {pointerId: 7, button: 0, clientX: 195, clientY: 370});
  await marker.emit('pointermove', {pointerId: 7, clientX: 500, clientY: 0});
  await marker.emit('pointerup', {pointerId: 7, clientX: 500, clientY: 0});
  await marker.emit('click', {clientX: 500, clientY: 0});
  assert.equal(element('compose-pin-text').textContent, 'Pinned 100% across, 0% down');
  let stopped = false;
  await marker.emit('keydown', {code: 'ArrowLeft', shiftKey: true,
    stopPropagation() { stopped = true; }});
  assert.equal(stopped, true);
  assert.equal(element('compose-pin-text').textContent, 'Pinned 95% across, 0% down');
  assert.equal(evaluate('currentFrame()'), 0);
  await marker.emit('keydown', {code: 'ArrowDown'});
  assert.equal(element('compose-pin-text').textContent, 'Pinned 95% across, 1% down');
});

test('variable rate compose sends the full displayed time', async () => {
  const {element, click, at, requests} = await pinPage({cfr: false});
  await click('new-comment');
  await at(1.23456789);
  await element('video').emit('click', {clientX: 195, clientY: 370});
  element('compose-text').value = 'Move it';
  await click('compose-submit');
  assert.deepEqual(JSON.parse(requests.find(({options}) => options.method === 'POST').options.body).pin,
    {x: 0.5, y: 0.5, time_s: 1.23456789});
});

test('editing refuses a changed span that strands an untouched pin before sending', async () => {
  const {page, element, state, evaluate, requests} = await pinPage();
  const comment = {id: 'c1', kind: 'fix', target: 'element', text: 'Here', request: null,
    start_frame: 30, end_frame: 90, start_s: 1, end_s: 3,
    pin: {x: 0.5, y: 0.5, frame: 60, time_s: 2}};
  state.comments.push(comment);
  await page.loadReview();
  page.openEdit(comment, element('comment-list').children[0]);
  evaluate('selection = {inFrame: 30, outFrame: 40}; renderSelection()');
  await page.saveEdit();
  assert.equal(evaluate('editState.errorNode.textContent'),
    'Move to a frame inside the range, or remove the pin.');
  assert.equal(requests.some(({options}) => options.method === 'PATCH'), false);
});

test('metadata sets the video aspect and resize placement uses the current picture box', async () => {
  const {element, click, evaluate} = await pinPage();
  await element('video').emit('loadedmetadata');
  assert.equal(element('video').style.aspectRatio, '1080 / 1920');
  Object.assign(element('compare-video'), {videoWidth: 1920, videoHeight: 1080});
  await element('compare-video').emit('resize');
  assert.equal(element('compare-video').style.aspectRatio, '1920 / 1080');
  await click('new-comment');
  await element('video').emit('click', {clientX: 195, clientY: 370});
  element('video').rect = {left: 10, top: 100, width: 200, height: 200};
  evaluate('renderPin()');
  assert.equal(element('pin-marker').style.values['--pin-left'], '100px');
  assert.equal(element('pin-marker').style.values['--pin-top'], '100px');
});
