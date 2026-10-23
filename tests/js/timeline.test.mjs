import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {decisionBody} from '../../src/yap_studio/review/static/taste.mjs';
import * as timelineHelpers from '../../src/yap_studio/review/static/timeline.mjs';
import {
  parseFps, fpsLabel, frameAt, frameStart, frameMid, frameCount, tenthsText,
  frameTimeText, secondsText, spanText, sortComments,
  normalizeSelection, normalizeTimeSelection, assignLanes,
  TARGET_NAMES, targetName,
  defaultCompareVersion, mapStatusMessage, threadStateLabel, versionUrl,
  replyEvidence, replyHeading, replyMomentText, keepCheckText, changeText,
  draftNotesText, roundStatusText, sendStateLabel, reviewPollInterval,
  questionsForComment, questionAnswerBody,
  spellingAvailable, spellingRequest, spellingText,
  requestText, editedRequest, removedKindName, visibleRemovedRanges, rangeRequested,
  findingKindLabel, findingSourceLabel, findingEvidenceText, findingNoteBody, findingAdded,
} from '../../src/yap_studio/review/static/timeline.mjs';

const fps = {num: 30000, den: 1001};

function reviewerFinding(overrides = {}) {
  return {id: 'f1', source: 'agent', reviewer: 'Reviewer', kind: 'slow', severity: 'medium',
    start_frame: 30, end_frame: 90, start_s: 1.09, end_s: 3.09,
    reason: '<b>The opening holds too long.</b>', fix: 'Shorten the hold.',
    target: 'story', check: null, evidence: {still_s: 4, lu_jump: 8.567}, ...overrides};
}

test('finding labels distinguish all kinds and automatic checks from agent notes', () => {
  for (const [kind, label] of Object.entries({slow: 'Slow stretch', jarring_cut: 'Jarring cut',
    face_covered: 'Face covered', caption_issue: 'Caption problem', audio: 'Sound', other: 'Other'})) {
    assert.equal(findingKindLabel(kind), label);
  }
  assert.equal(findingKindLabel('unknown'), 'Other');
  assert.equal(findingSourceLabel(reviewerFinding()), 'Agent');
  assert.equal(findingSourceLabel(reviewerFinding({source: 'machine', reviewer: null})), 'Automatic check');
});

test('finding evidence preserves supplied order and rounds finite numbers to at most two decimals', () => {
  assert.equal(findingEvidenceText({still_s: 4, lu_jump: 8.567, zero: 0, signed: -0.123}),
    'still_s 4 · lu_jump 8.57 · zero 0 · signed -0.12');
  assert.equal(findingEvidenceText({z: 1.234, a: 1.2}), 'z 1.23 · a 1.2');
  for (const evidence of [null, undefined, {}, {missing: null, bad: NaN, infinite: Infinity,
    text: '4', bool: true}]) assert.equal(findingEvidenceText(evidence), '');
});

test('finding drafts use exact half-open frames or seconds, Fix, target and only note fields', () => {
  const finding = reviewerFinding();
  const common = {kind: 'fix', target: 'story',
    text: 'Flagged: <b>The opening holds too long.</b> Suggested fix: Shorten the hold.'};
  assert.deepEqual(findingNoteBody(finding, true), {...common, start_frame: 30, end_frame: 90});
  assert.deepEqual(findingNoteBody(finding, false), {...common, start_s: 1.09, end_s: 3.09});
  const point = reviewerFinding({end_frame: null, end_s: null, target: null});
  assert.deepEqual(findingNoteBody(point, true), {...common, target: null, start_frame: 30, end_frame: null});
  assert.deepEqual(findingNoteBody(point, false), {...common, target: null, start_s: 1.09, end_s: null});
  assert.equal(finding.start_frame, 30);
});

test('variable rate finding point at 3.15 seconds clamps to a 3 second duration', () => {
  for (const end_s of [null, 3.2]) {
    const finding = reviewerFinding({start_s: 3.15, end_s});
    const body = findingNoteBody(finding, false, 3);
    assert.equal(body.start_s, 3);
    assert.equal(body.end_s, null);
    assert.equal(finding.start_s, 3.15);
    assert.equal(finding.end_s, end_s);
  }
});

test('variable rate finding range from 2.9 to 3.15 seconds clamps to a 3 second duration', () => {
  const finding = reviewerFinding({start_s: 2.9, end_s: 3.15});
  const body = findingNoteBody(finding, false, 3);
  assert.equal(body.start_s, 2.9);
  assert.equal(body.end_s, 3);
  assert.deepEqual(findingNoteBody(finding, false, null), findingNoteBody(finding, false));
  assert.equal(findingNoteBody(finding, false, null).end_s, 3.15);
  assert.deepEqual(findingNoteBody(finding, true, 3), findingNoteBody(finding, true));
});

test('variable rate finding duplicate detection matches clamped point and range bodies', () => {
  for (const [start_s, end_s, clampedStart, clampedEnd] of [
    [3.15, null, 3, null], [3.15, 3.2, 3, null], [2.9, 3.15, 2.9, 3],
  ]) {
    const finding = reviewerFinding({start_s, end_s});
    const comment = {...findingNoteBody(finding, false), start_frame: null, end_frame: null,
      start_s: clampedStart, end_s: clampedEnd};
    assert.equal(findingAdded(finding, [comment], 3), true);
    for (const mismatch of [{...comment, start_s: 2.8}, {...comment, end_s: 2.95},
      {...comment, text: 'An edited note'}]) {
      assert.equal(findingAdded(finding, [mismatch], 3), false);
    }
    assert.equal(findingAdded(finding, [comment], null), false);
    assert.equal(findingAdded(finding, [findingNoteBody(finding, true)], 3), true);
  }
});

test('finding draft text respects the 5000 character limit without splitting Unicode characters', () => {
  const body = findingNoteBody(reviewerFinding({reason: '🙂'.repeat(6000)}), true);
  assert.equal([...body.text].length, 5000);
  assert.ok(body.text.endsWith('🙂'));
  assert.equal(findingNoteBody(reviewerFinding({reason: 'Line\r\nnext', fix: 'Trim.  '}), false).text,
    'Flagged: Line\nnext Suggested fix: Trim.');
});

test('finding duplicate detection matches exact text and point or range timing for each frame mode', () => {
  const finding = reviewerFinding();
  const framed = {...findingNoteBody(finding, true), id: 'c1', start_s: 1, end_s: 3};
  const timed = {...findingNoteBody(finding, false), id: 'c2', start_frame: null, end_frame: null};
  assert.equal(findingAdded(finding, [framed]), true);
  assert.equal(findingAdded(finding, [timed]), true);
  for (const mismatch of [{...framed, start_frame: 31}, {...framed, end_frame: 91},
    {...timed, start_s: 1.1}, {...timed, end_s: null}, {...timed, text: `${timed.text} `}]) {
    assert.equal(findingAdded(finding, [mismatch]), false);
  }
  assert.equal(findingAdded(finding, []), false);
  const point = reviewerFinding({end_frame: null, end_s: null});
  assert.equal(findingAdded(point, [findingNoteBody(point, true)]), true);
  assert.equal(findingAdded(point, [{...findingNoteBody(point, false), start_frame: null}]), true);
  const oneFrame = reviewerFinding({end_frame: 31});
  assert.equal(findingAdded(oneFrame, [findingNoteBody(oneFrame, true)]), true);
  assert.equal(findingAdded(oneFrame, [{...findingNoteBody(oneFrame, true), end_frame: null}]), true);
});

async function findingsPage({cfr = true, series = true, duration = 10} = {}) {
  let document;
  const main = {inert: false};
  class Node {
    constructor(tag = 'div') {
      Object.assign(this, {tag, children: [], attributes: {}, listeners: {}, dataset: {},
        value: '', textContent: '', checked: false, hidden: false, open: false,
        currentTime: 0, paused: true, duration, readyState: 1,
        style: {properties: {}, setProperty(key, value) { this.properties[key] = value; }},
        classList: {toggle() {}, add() {}, remove() {}}});
    }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    closest(selector) { return selector === 'main' ? main : null; }
    pause() { this.paused = true; }
    focus() { document.activeElement = this; }
    querySelector(selector) {
      for (const child of this.children) {
        if (child.className?.split(' ').includes(selector.slice(1))) return child;
        const found = child.querySelector(selector);
        if (found) return found;
      }
      return null;
    }
  }
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  document = {getElementById: element, createElement: (tag) => new Node(tag),
    addEventListener() {}, visibilityState: 'visible', activeElement: null};
  const items = [reviewerFinding(), reviewerFinding({id: 'f2', source: 'machine', kind: 'audio',
    start_frame: 120, end_frame: null, start_s: 4.09, end_s: null, evidence: {}, target: 'speech'}),
  reviewerFinding({id: 'f3', kind: 'caption_issue', start_frame: 60, end_frame: 120,
    start_s: 2.09, end_s: 4.09, target: 'caption'})];
  const normalizedComment = (body, id) => ({id, ...body,
    start_frame: cfr ? body.start_frame : null, end_frame: cfr ? body.end_frame : null,
    start_s: cfr ? body.start_frame / 30 : body.start_s,
    end_s: cfr ? (body.end_frame === null ? null : body.end_frame / 30) : body.end_s,
    send: {state: 'draft'}});
  const state = {findings: {available: true, file: 'private/path.findings.json', stale: [], items},
    comments: [normalizedComment(findingNoteBody(items[2], cfr), 'c1')],
    mediaChanged: false, offline: false, failWrite: false, failRefresh: false, gate: null};
  const requests = [];
  const context = vm.createContext({
    ...timelineHelpers, URLSearchParams, Element: Node, document,
    location: new URL(`http://review.example.net/v/2/?t=token${series ? '&version=1' : ''}`),
    history: {replaceState() {}}, window: {addEventListener() {}},
    setInterval() {}, clearInterval() {}, confirm: () => true,
    setTimeout() {}, clearTimeout() {},
    fetch: async (path, options) => {
      requests.push({path, options});
      if (state.offline) throw new TypeError('offline');
      const url = new URL(path, 'http://review.example.net/v/2/');
      const version = Number(url.searchParams.get('version') || 1);
      const reply = (data, status = 200) => ({ok: status < 400, status,
        json: async () => structuredClone(data)});
      if (options.method) {
        if (state.gate) await state.gate;
        if (state.failWrite) return reply({error: {message: 'Could not save note.'}}, 409);
        if (options.method === 'POST') {
          const comment = normalizedComment(JSON.parse(options.body), `c${state.comments.length + 1}`);
          state.comments.push(comment);
          return reply(comment, 201);
        }
        Object.assign(state.comments[0], JSON.parse(options.body));
        return reply(state.comments[0]);
      }
      if (state.failRefresh) throw new TypeError('offline');
      return reply({media: {name: 'render.mp4', sha256: `hash${version}`, duration_s: duration,
        cfr, fps: cfr ? '30/1' : null}, comments: version === 1 ? state.comments : [],
      findings: state.findings, previous: null, text: 'Creator feedback only.', media_changed: state.mediaChanged,
      series: series ? {project: false, current: version, newest: 2, questions: [], rounds: [], threads: [],
        versions: [{number: 1}, {number: 2}]} : null});
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.page = {start, loadReview, refreshReview, selectVersion, setOffline, changeComment};'), context);
  await context.page.start();
  const click = (node) => node.listeners.click({stopPropagation() {}});
  const actions = (index) => element('findings-list').children[index].querySelector('.finding-actions').children;
  return {page: context.page, context, element, items, state, requests, click, actions};
}

test('variable rate finding page posts clamped notes and recognizes them after reload', async () => {
  const {page, state, requests, click, actions} = await findingsPage({cfr: false, duration: 3});
  state.findings.items = [reviewerFinding({start_s: 3.15, end_s: null}),
    reviewerFinding({id: 'f2', start_s: 2.9, end_s: 3.15})];
  state.comments = [];
  await page.loadReview();
  for (const index of [0, 1]) await click(actions(index)[1]);
  const bodies = requests.filter(({options}) => options.method === 'POST')
    .map(({options}) => JSON.parse(options.body));
  assert.deepEqual(bodies.map(({start_s, end_s}) => ({start_s, end_s})),
    [{start_s: 3, end_s: null}, {start_s: 2.9, end_s: 3}]);

  const reloaded = await findingsPage({cfr: false, duration: 3});
  reloaded.state.findings = state.findings;
  reloaded.state.comments = state.comments;
  await reloaded.page.loadReview();
  for (const index of [0, 1]) {
    assert.equal(reloaded.actions(index)[1].textContent, 'Added');
    assert.equal(reloaded.actions(index)[1].disabled, true);
  }
});

for (const cfr of [true, false]) {
  test(`finding rows, point and overlapping range markers seek paused and add drafts at cfr=${cfr}`, async () => {
    const {page, element, items, state, requests, click, actions} = await findingsPage({cfr});
    const section = element('findings-section');
    section.open = true;
    assert.equal(section.hidden, false);
    assert.equal(element('findings-count').textContent, '3');
    assert.equal(element('comment-count').textContent, '1');
    assert.equal(element('findings-status').hidden, true);
    const row = element('findings-list').children[0];
    assert.deepEqual(row.querySelector('.finding-heading').children.map((node) => node.textContent),
      ['0:01.0-0:03.0', 'Slow stretch', 'Severity: medium', 'Agent']);
    assert.equal(row.querySelector('.finding-text').textContent, items[0].reason);
    assert.equal(row.querySelector('.finding-text').children.length, 0);
    assert.equal(row.querySelector('.finding-fix').textContent, 'Suggested fix: Shorten the hold.');
    assert.equal(row.querySelector('.finding-evidence').textContent, 'still_s 4 · lu_jump 8.57');
    assert.equal(element('findings-list').children[1].querySelector('.finding-evidence'), null);
    assert.equal(element('findings-list').children[1].querySelector('.finding-heading').children[3].textContent,
      'Automatic check');
    assert.equal(actions(2)[1].textContent, 'Added');
    assert.equal(actions(2)[1].disabled, true);
    await click(actions(2)[1]);
    assert.equal(requests.filter(({options}) => options.method === 'POST').length, 0);
    const markers = element('finding-markers').children;
    assert.equal(markers.length, 3);
    assert.equal(markers[0].className, 'marker finding-marker');
    assert.match(markers[0].attributes['aria-label'], /^Automatic check finding · 0:04\.0 · Sound/);
    const ranges = markers.filter((marker) => marker.className === 'range-bar finding-range');
    assert.deepEqual(ranges.map((marker) => marker.style.properties['--lane-top']), ['0px', '44px']);
    assert.ok(Math.abs(parseFloat(ranges[0].style.properties['--start']) - (cfr ? 10 : 10.9)) < 1e-8);
    assert.equal(element('scrubber').style.properties['--finding-height'], '132px');
    assert.equal(element('ranges').children.length, 1);
    element('video').paused = false;
    await click(markers[0]);
    assert.equal(element('video').currentTime, cfr ? 120.5 / 30 : 4.09);
    assert.equal(element('video').paused, true);
    await click(ranges[0]);
    assert.equal(element('video').currentTime, cfr ? 30.5 / 30 : 1.09);
    await click(actions(0)[0]);
    assert.equal(element('video').currentTime, cfr ? 30.5 / 30 : 1.09);
    await click(actions(0)[1]);
    const posted = requests.find(({options}) => options.method === 'POST');
    assert.equal(posted.path, 'api/comments?version=1');
    assert.equal(posted.options.headers['X-Yap-Token'], 'token');
    assert.deepEqual(JSON.parse(posted.options.body), findingNoteBody(items[0], cfr));
    assert.equal(actions(0)[1].textContent, 'Added');
    assert.equal(actions(0)[1].disabled, true);
    assert.equal(section.open, true);
    assert.equal(element('comment-count').textContent, '2');
    assert.equal(element('comment-list').children.find((node) => node.dataset.commentId === 'c2')
      .querySelector('.send-state').textContent, 'Draft');
    assert.equal(element('copy-feedback').disabled, false);
    state.comments[1].text = 'My edited note';
    await page.loadReview();
    assert.equal(actions(0)[1].textContent, 'Added');
    await page.selectVersion(2);
    assert.equal(actions(0)[1].textContent, 'Add as my note');
    assert.equal(element('comment-count').textContent, '0');
    await page.selectVersion(1);
    assert.equal(actions(0)[1].textContent, 'Added');
    state.comments.pop();
    await page.loadReview();
    assert.equal(actions(0)[1].textContent, 'Add as my note');
  });
}

test('finding polling hides absent notes and shows only generic stale or error text', async () => {
  const {page, element, state} = await findingsPage();
  state.findings.items[0].reason = 'A new agent note';
  await page.refreshReview();
  assert.equal(element('findings-list').children[0].querySelector('.finding-text').textContent, 'A new agent note');
  for (const [findings, message] of [
    [undefined, ''], [{available: false, items: [], stale: []}, ''],
    [{available: false, items: [], stale: ['private/path.findings.json']},
      "The agent's notes are for an earlier render of this version."],
    [{available: false, items: [], stale: ['private/earlier.json'], error: 'secret/path.json: malformed'},
      "The agent's notes could not be read."],
  ]) {
    state.findings = findings;
    await page.refreshReview();
    assert.equal(element('findings-section').hidden, true);
    assert.equal(element('finding-markers').children.length, 0);
    assert.equal(element('findings-list').children.length, 0);
    assert.equal(element('scrubber').style.properties['--finding-height'], '0px');
    assert.equal(element('findings-status').textContent, message);
    assert.equal(element('findings-status').hidden, !message);
  }
  state.findings = {available: true, items: [], stale: ['private/earlier.json']};
  await page.refreshReview();
  assert.equal(element('findings-count').textContent, '0');
  assert.equal(element('findings-section').hidden, false);
  assert.equal(element('findings-status').hidden, true);
});

test('finding controls guard offline, changed media, busy writes and duplicate taps', async () => {
  const {page, context, element, state, requests, click, actions} = await findingsPage();
  const controls = () => [...actions(0), ...element('finding-markers').children];
  const blockedClick = async () => {
    const writes = requests.filter(({options}) => options.method).length;
    const time = element('video').currentTime;
    assert.ok(controls().every((node) => node.disabled));
    await click(actions(0)[0]);
    await click(actions(0)[1]);
    await click(element('finding-markers').children[0]);
    assert.equal(requests.filter(({options}) => options.method).length, writes);
    assert.equal(element('video').currentTime, time);
  };
  page.setOffline(true);
  await blockedClick();
  page.setOffline(false);
  for (const flag of ['composeBusy', 'roundBusy', 'releaseBusy', 'answerBusy', 'switchingVersion']) {
    vm.runInContext(`${flag} = true; renderVersions();`, context);
    await blockedClick();
    vm.runInContext(`${flag} = false; renderVersions();`, context);
  }
  vm.runInContext('editState = {busy: true}; updateFindingControls();', context);
  await blockedClick();
  vm.runInContext('editState = null; updateFindingControls();', context);
  state.mediaChanged = true;
  await page.refreshReview();
  await blockedClick();
  state.mediaChanged = false;
  await page.refreshReview();
  let finish;
  state.gate = new Promise((resolve) => { finish = resolve; });
  const otherWrite = page.changeComment('c1', {target: 'caption'});
  await blockedClick();
  finish();
  await otherWrite;
  state.gate = new Promise((resolve) => { finish = resolve; });
  const add = click(actions(0)[1]);
  await blockedClick();
  finish();
  await add;
  assert.equal(requests.filter(({options}) => options.method === 'POST').length, 1);
  assert.equal(actions(0)[1].textContent, 'Added');
  assert.equal(actions(0)[0].disabled, false);
  assert.ok(element('finding-markers').children.every((node) => !node.disabled));
});

test('a failed finding write can retry, and a saved draft survives a failed refresh on a single video', async () => {
  const {page, element, state, requests, click, actions} = await findingsPage({series: false});
  state.failWrite = true;
  await click(actions(0)[1]);
  assert.equal(actions(0)[1].textContent, 'Add as my note');
  assert.equal(actions(0)[1].disabled, false);
  assert.equal(element('status').textContent, 'Could not save note.');
  state.failWrite = false;
  state.failRefresh = true;
  await click(actions(0)[1]);
  assert.equal(actions(0)[1].textContent, 'Added');
  assert.equal(element('comment-count').textContent, '2');
  assert.equal(element('offline-banner').hidden, false);
  assert.equal(requests.filter(({options}) => options.method === 'POST').at(-1).path, 'api/comments');
  state.failRefresh = false;
  await page.refreshReview();
  assert.equal(actions(0)[1].textContent, 'Added');
  assert.equal(actions(0)[0].disabled, false);
});

function evidenceReply(overrides = {}) {
  return {version: 2, status: 'changed', text: 'Removed the repeated words', evidence: {
    schema: 'yap-studio/reply-evidence@1', batch: 'b1', comment_version: 'v1', version: 'v2',
    media_sha256: 'a'.repeat(64), disposition: 'applied', operations: [{id: 'op1', type: 'cut'}],
    moment: {status: 'mapped', start_s: 12.39, end_s: 12.5, start_frame: 296, end_frame: 300},
    unmapped: null, change: {before: [{id: 't1:1', segment: 's1', text: 'really'}], after: [],
      differences: ['removed', 'words']}, keep: null, question: null, ...overrides,
  }};
}

test('reply evidence accepts the complete writer shape and nullable measurements', () => {
  const reply = evidenceReply();
  assert.equal(replyEvidence(reply), reply.evidence);
  const unlocated = evidenceReply({moment: null, unmapped: 'whole video', change: null,
    operations: [{id: 'op1'}], keep: {status: 'unmeasured', detail: 'No constraint'},
    question: {question: 'Which line?', answer: null, option: null}});
  assert.equal(replyEvidence(unlocated), unlocated.evidence);
  unlocated.evidence.question = {question: 'Which line?', answer: 'The opening', option: 'opening'};
  assert.equal(replyEvidence(unlocated), unlocated.evidence);
  const timed = evidenceReply({moment: {status: 'time', start_s: 0, end_s: 0,
    start_frame: null, end_frame: null}});
  assert.equal(replyEvidence(timed), timed.evidence);
});

test('reply evidence rejects old schemas, missing fields and wrong nested types', () => {
  for (const reply of [null, {}, {evidence: null}, {evidence: []},
    {evidence: {frames: [4, 8]}}, evidenceReply({schema: 'yap-studio/reply-evidence@2'})]) {
    assert.equal(replyEvidence(reply), null);
  }
  const invalid = [
    (e) => { delete e.batch; }, (e) => { e.extra = 'unknown'; },
    ...['batch', 'comment_version', 'version', 'media_sha256', 'disposition']
      .map((key) => (e) => { e[key] = 2; }),
    (e) => { e.unmapped = false; }, (e) => { e.operations = {}; },
    (e) => { e.operations = [null]; }, (e) => { e.operations[0].id = 2; },
    (e) => { e.operations[0].type = null; }, (e) => { e.moment = []; },
    (e) => { delete e.moment.end_frame; }, (e) => { e.moment.status = 'unknown'; },
    ...['start_s', 'end_s'].flatMap((key) => ['12.3', true, NaN, Infinity, -1]
      .map((value) => (e) => { e.moment[key] = value; })),
    (e) => { e.moment.end_s = 1; },
    ...['start_frame', 'end_frame'].flatMap((key) => [1.5, '296', true, -1, 2 ** 53]
      .map((value) => (e) => { e.moment[key] = value; })),
    (e) => { e.moment.end_frame = 1; }, (e) => { e.change = []; },
    (e) => { e.change.before = {}; }, (e) => { e.change.after = [null]; },
    ...['id', 'segment', 'text'].map((key) => (e) => { e.change.before[0][key] = 2; }),
    (e) => { e.change.differences = 'words'; }, (e) => { e.change.differences = ['unknown']; },
    (e) => { e.keep = {}; }, (e) => { e.keep = {status: 'unknown', detail: 'Why'}; },
    (e) => { e.keep = {status: 'kept', detail: 2}; }, (e) => { e.question = []; },
    ...['question', 'answer', 'option'].map((key) => (e) => {
      e.question = {question: 'Which line?', answer: null, option: null, [key]: 2};
    }),
  ];
  for (const mutate of invalid) {
    const reply = evidenceReply();
    mutate(reply.evidence);
    assert.equal(replyEvidence(reply), null, mutate.toString());
    assert.equal(replyMomentText(reply, {cfr: true, fps: '24/1'}), null);
    assert.equal(keepCheckText(reply), null);
    assert.equal(changeText(reply), null);
  }
});

test('reply headings name every disposition using the reply series number', () => {
  for (const [status, label] of [['changed', 'Changed'], ['kept', 'Kept'],
    ['declined', 'Declined'], ['needs_clarification', 'Question']]) {
    assert.equal(replyHeading({status, version: 3}), `${label} in v3`);
  }
});

test('range whose last word was cut shows wrong join: partial evidence keeps its mapped start', () => {
  const reply = evidenceReply({moment: {status: 'partial', start_s: 1.09, end_s: 2.0,
    start_frame: 30, end_frame: 60}});
  assert.equal(replyEvidence(reply), reply.evidence);
  assert.equal(replyMomentText(reply, {cfr: true, fps: '30/1'}),
    'Partly removed in v2, see it at 0:01.0');
  assert.equal(replyMomentText(reply, {cfr: false, fps: null}),
    'Partly removed in v2, see it at 0:01.0');
  assert.equal(changeText(reply), 'Words removed');
});

test('reply moment text describes each mapping status in its own constant rate timeline', () => {
  const facts = {number: 2, cfr: true, fps: '24/1'};
  for (const [status, expected] of [
    ['mapped', 'See it in v2 at 0:12.3'], ['moved', 'See it in v2 at 0:12.3'],
    ['removed', 'Removed in v2, see the join at 0:12.3'],
    ['ambiguous', 'See it in v2 near 0:12.3'], ['time', 'See v2 at 0:12.3'],
  ]) {
    const reply = evidenceReply();
    reply.evidence.moment.status = status;
    assert.equal(replyMomentText(reply, facts), expected);
  }
  const reply = evidenceReply();
  assert.equal(replyMomentText(reply, {cfr: true, fps: '25/1'}), 'See it in v2 at 0:11.8');
  assert.equal(replyMomentText(reply, {cfr: true, fps: '30000/1001'}), 'See it in v2 at 0:09.8');
  assert.equal(replyMomentText(evidenceReply({moment: null}), facts), null);
});

test('reply moment text falls back to seconds for variable rate and missing frame facts', () => {
  const reply = evidenceReply();
  assert.equal(replyMomentText(reply, {cfr: false, fps: '25/1'}), 'See it in v2 at 0:12.3');
  assert.equal(replyMomentText(reply, {cfr: true, fps: null}), 'See it in v2 at 0:12.3');
  assert.equal(replyMomentText(reply, null), 'See it in v2 at 0:12.3');
  reply.evidence.moment.start_frame = null;
  assert.equal(replyMomentText(reply, {cfr: true, fps: '25/1'}), 'See it in v2 at 0:12.3');
});

test('keep check text includes limitations and broken check details', () => {
  for (const [status, expected] of [['kept', 'kept'],
    ['unmeasured', 'not measured (No layer data)'], ['broken', 'broken (No layer data)']]) {
    assert.equal(keepCheckText(evidenceReply({keep: {status, detail: 'No layer data'}})),
      `Checked in v2: ${expected}`);
  }
  assert.equal(keepCheckText(evidenceReply()), null);
});

test('change text summarizes all measured differences and omits absent changes', () => {
  const reply = evidenceReply();
  reply.evidence.change.differences = ['removed', 'words'];
  assert.equal(changeText(reply), 'Words removed');
  reply.evidence.change.differences = ['words'];
  assert.equal(changeText(reply), 'Words changed');
  reply.evidence.change.differences = ['removed', 'words', 'order', 'timing', 'pause', 'layers'];
  assert.equal(changeText(reply), 'Words removed, Order changed, Timing changed, '
    + 'Pause changed, Captions or graphics changed');
  reply.evidence.change.differences = [];
  assert.equal(changeText(reply), null);
  assert.equal(changeText(evidenceReply({change: null})), null);
});

async function replyPage({initial = 3, variable = false, oldVariable = false,
  questions = [], comments = []} = {}) {
  const scrolls = [];
  class Node {
    constructor(tag = 'div') {
      this.tagName = tag.toUpperCase();
      this.children = []; this.attributes = {}; this.listeners = new Map(); this.dataset = {};
      this.value = ''; this.textContent = ''; this.currentTime = 0;
      this.readyState = 1; this.duration = 60; this.paused = true;
      this.style = {setProperty() {}};
      this.classList = {toggle() {}, add() {}, remove() {}};
    }
    addEventListener(type, fn) {
      this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
    }
    removeEventListener(type, fn) {
      this.listeners.set(type, (this.listeners.get(type) ?? []).filter((item) => item !== fn));
    }
    emit(type) { return Promise.all((this.listeners.get(type) ?? []).map((fn) => fn({stopPropagation() {}}))); }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    set src(value) { this.attributes.src = value; this.currentTime = 0; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
    querySelectorAll(selector) {
      const selectors = selector.split(',').map((part) => part.trim());
      return this.children.flatMap((child) => [
        ...(selectors.some((part) => part.startsWith('.')
          ? child.className?.split(' ').includes(part.slice(1))
          : child.tagName === part.toUpperCase()) ? [child] : []),
        ...child.querySelectorAll(selector),
      ]);
    }
    closest() { return main; }
    focus() { context.document.activeElement = this; }
    setSelectionRange() {}
    pause() { if (!this.paused) { this.paused = true; void this.emit('pause'); } }
    scrollIntoView(options) {
      scrolls.push({options: {...options}, mainTime: element('video').currentTime,
        compareTime: element('compare-video').currentTime});
    }
  }
  const main = {inert: false};
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  const thread = {version: 1, id: 'c1', kind: 'fix', target: 'caption', text: 'Earlier note',
    state: 'changed', start_frame: 60, end_frame: null, start_s: 9.5, end_s: null,
    reply: evidenceReply({keep: {status: 'unmeasured', detail: 'No layer data'}})};
  const versions = [{number: 1, cfr: !oldVariable, fps: '30/1', duration_s: 60},
    {number: 2, cfr: !variable, fps: '24/1', duration_s: 60},
    {number: 3, cfr: false, fps: null, duration_s: 60}];
  const requests = [];
  const timers = new Map();
  let timerId = 0;
  const context = vm.createContext({
    ...timelineHelpers, decisionBody, URLSearchParams, Element: Node, performance: {now: () => 0},
    location: new URL(`http://review.example.net/?t=token&version=${initial}`),
    history: {replaceState() {}}, window: {addEventListener() {}},
    document: {getElementById: element, createElement: (tag) => new Node(tag), addEventListener() {},
      visibilityState: 'visible'},
    setInterval() { return 1; }, clearInterval() {},
    setTimeout(fn, delay) {
      const id = ++timerId;
      // Only comparison sync timers belong to this harness's pending-match assertions.
      if (delay === 150) timers.set(id, {fn, delay});
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
    fetch: async (path) => {
      requests.push(path);
      assert.ok(path.startsWith('api/review?'), `unexpected mapping or mutation: ${path}`);
      const number = Number(new URLSearchParams(path.split('?')[1]).get('version'));
      const facts = versions.find((item) => item.number === number);
      return {ok: true, status: 200, json: async () => ({
        media: {...facts, name: `sample-v${number}.mp4`, sha256: `hash${number}`},
        comments, previous: null, text: '', series: {current: number, newest: 3,
          current_version: 3, versions, threads: [thread,
            {...thread, id: 'c2', reply: {...thread.reply, evidence: {frames: [4, 8]}}},
            {...thread, id: 'c3', state: 'open', reply: null}], rounds: [], questions},
      })};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/,
      'globalThis.page = {start, openReplyEvidence, renderThreads, setOffline, saveTasteDecision};'), context);
  await context.page.start();
  return {page: context.page, context, element, thread, requests, main, timers, scrolls};
}

test('reply blocks are siblings of comment buttons and evidence taps seek both own timelines', async () => {
  const {element, requests, thread, main, timers, scrolls} = await replyPage();
  const [row, legacyRow, openRow] = element('thread-list').children;
  assert.equal(row.tagName, 'LI');
  const [comment, block] = row.children;
  assert.equal(comment.tagName, 'BUTTON');
  assert.ok(comment.children.every((child) => child.tagName === 'SPAN'));
  assert.equal(block.className, 'thread-reply');
  assert.deepEqual(block.children.map((child) => child.textContent), [
    'Changed in v2', thread.reply.text, 'Words removed',
    'Checked in v2: not measured (No layer data)', 'See it in v2 at 0:12.3',
  ]);
  assert.equal(block.children.at(-1).tagName, 'BUTTON');
  assert.equal(legacyRow.children[1].children.length, 2);
  assert.equal(legacyRow.children[1].children[0].textContent, 'Changed in v2');
  assert.equal(openRow.children.length, 1);
  assert.equal(openRow.children[0].children.at(-1).textContent, 'Open');
  await block.children.at(-1).emit('click');
  assert.deepEqual(requests, ['api/review?version=3', 'api/review?version=2']);
  assert.equal(element('video').getAttribute('src'), 'media?version=2&t=token');
  assert.equal(element('compare-video').getAttribute('src'), 'media?version=1&t=token');
  assert.equal(element('compare-panel').hidden, false);
  assert.equal(element('compare-video').currentTime, frameMid(60, {num: 30, den: 1}));
  assert.equal(element('video').currentTime, frameMid(296, {num: 24, den: 1}));
  assert.equal(element('video').paused, true);
  assert.equal(element('compare-video').paused, true);
  assert.equal(main.inert, false);
  assert.deepEqual(scrolls, [{options: {block: 'start'},
    mainTime: frameMid(296, {num: 24, den: 1}), compareTime: frameMid(60, {num: 30, den: 1})}]);
  // A delayed seeked or pause event must preserve the measured comparison pair.
  await element('video').emit('seeked');
  await element('video').emit('pause');
  assert.equal(timers.size, 0);
  assert.match(element('time-readout').textContent, /^0:12.3/);
  element('video').currentTime = 15;
  await element('video').emit('seeked');
  assert.equal([...timers.values()][0].delay, 150);
});

test('reply evidence uses seconds for variable rate and avoids reloading its selected version', async () => {
  const {page, element, thread, requests} = await replyPage({initial: 2, variable: true, oldVariable: true});
  await page.openReplyEvidence(thread);
  assert.deepEqual(requests, ['api/review?version=2']);
  assert.equal(element('video').currentTime, 12.39);
  assert.equal(element('compare-video').currentTime, thread.start_s);
});

test('reply evidence is disabled during a taste decision POST and enabled after it resolves', async () => {
  for (const initial of [3, 2]) {
    const {page, context, element, thread, requests, scrolls} = await replyPage({initial});
    const evidenceButton = () => element('thread-list').children[0].children[1].children.at(-1);
    const proposal = {id: 'p1', sentence: 'Use short phrases.'};
    let finishPost;
    const post = new Promise((resolve) => { finishPost = resolve; });
    const fetch = context.fetch;
    context.fetch = async (path, options) => {
      if (path !== 'api/taste') return fetch(path, options);
      requests.push(path);
      assert.equal(options.method, 'POST');
      assert.deepEqual(JSON.parse(options.body), {proposal: proposal.id, action: 'approve'});
      await post;
      return {ok: true, status: 200, json: async () => ({taste: {available: true, proposals: []}})};
    };
    element('video').currentTime = 7;
    element('compare-video').currentTime = 4;
    assert.equal(evidenceButton().disabled, false);
    const saving = page.saveTasteDecision(proposal, 'approve');
    assert.deepEqual(requests, [`api/review?version=${initial}`, 'api/taste']);
    assert.equal(evidenceButton().disabled, true);
    await page.openReplyEvidence(thread);
    assert.equal(vm.runInContext('selectedVersion', context), initial);
    assert.equal(element('video').currentTime, 7);
    assert.equal(element('compare-video').currentTime, 4);
    assert.equal(scrolls.length, 0);
    assert.deepEqual(requests, [`api/review?version=${initial}`, 'api/taste']);
    finishPost();
    await saving;
    assert.equal(evidenceButton().disabled, false);
    await evidenceButton().emit('click');
    assert.equal(vm.runInContext('selectedVersion', context), 2);
    assert.equal(element('video').currentTime, frameMid(296, {num: 24, den: 1}));
    assert.equal(element('compare-video').currentTime, frameMid(60, {num: 30, den: 1}));
  }
});

test('a second edit stays busy when the first save finishes its review refresh', async () => {
  const comments = ['c1', 'c2'].map((id, index) => ({id, kind: 'fix', target: 'story',
    text: `Note ${index + 1}`, start_frame: null, end_frame: null,
    start_s: index + 1, end_s: null, send: {state: 'draft'}}));
  const {page, context, element, thread, requests, scrolls} = await replyPage({comments});
  let finishRefresh;
  const refresh = new Promise((resolve) => { finishRefresh = resolve; });
  let refreshStarted;
  const refreshing = new Promise((resolve) => { refreshStarted = resolve; });
  let finishPatch;
  const patch = new Promise((resolve) => { finishPatch = resolve; });
  const fetch = context.fetch;
  const patches = [];
  let reviews = 0;
  context.fetch = async (path, options) => {
    if (options.method === 'PATCH') {
      patches.push(path);
      if (patches.length === 2) await patch;
      return {ok: true, status: 200, json: async () => ({})};
    }
    if (++reviews === 1) {
      refreshStarted();
      await refresh;
    }
    return fetch(path, options);
  };
  const openEdit = async (index) => {
    const row = element('comment-list').children[index];
    const edit = row.querySelector('.comment-actions').children
      .find((button) => button.textContent === 'Edit');
    await edit.emit('click');
    row.querySelector('textarea').value = `Edited note ${index + 1}`;
    return row.querySelector('.form-actions').children
      .find((button) => button.textContent === 'Save');
  };
  const evidenceButton = () => element('thread-list').querySelector('.reply-evidence');
  const firstSave = (await openEdit(0)).emit('click');
  await refreshing;
  assert.equal(vm.runInContext('editState', context), null);
  const secondSaveButton = await openEdit(1);
  const secondSave = secondSaveButton.emit('click');
  const secondState = vm.runInContext('editState', context);
  assert.equal(secondState.busy, true);
  assert.equal(evidenceButton().disabled, true);
  finishRefresh();
  await firstSave;
  assert.equal(vm.runInContext('editState', context), secondState);
  assert.equal(secondState.busy, true);
  assert.equal(evidenceButton().disabled, true);
  await secondSaveButton.emit('click');
  assert.deepEqual(patches, ['api/comments/c1?version=3', 'api/comments/c2?version=3']);
  await page.openReplyEvidence(thread);
  assert.equal(vm.runInContext('selectedVersion', context), 3);
  assert.equal(scrolls.length, 0);
  assert.deepEqual(requests, ['api/review?version=3', 'api/review?version=3']);
  finishPatch();
  await secondSave;
  assert.equal(secondState.busy, false);
  assert.equal(evidenceButton().disabled, false);
  await evidenceButton().emit('click');
  assert.equal(vm.runInContext('selectedVersion', context), 2);
  assert.equal(scrolls.length, 1);
});

for (const failureTiming of ['before', 'during', 'after']) {
  test(`question answer errors survive taste saves when the answer fails ${failureTiming} the save`, async () => {
    const question = {batch: 'b1', comment_version: 'v1', comment: {version: 1, id: 'c1'},
      state: 'asked', question: 'Which words?', options: [{value: 'caption', label: 'The caption'}]};
    const {page, context, element} = await replyPage({questions: [question]});
    const cards = () => element('thread-list').querySelectorAll('.question-card');
    const card = cards()[0];
    const errorNode = card.querySelectorAll('.form-error')[0];
    const textarea = card.querySelectorAll('textarea')[0];
    textarea.value = 'The repeated words';
    await textarea.emit('input');
    const evidenceButtons = () => element('thread-list').querySelectorAll('.reply-evidence');
    const evidenceButton = evidenceButtons()[0];
    let finishAnswer;
    const answerPost = new Promise((resolve) => { finishAnswer = resolve; });
    let finishTaste;
    const tastePost = new Promise((resolve) => { finishTaste = resolve; });
    const fetch = context.fetch;
    context.fetch = async (path, options) => {
      if (path.startsWith('api/questions/answer')) {
        assert.equal(options.method, 'POST');
        await answerPost;
        return {ok: false, status: 400, json: async () => ({error: {message: 'Try again'}})};
      }
      if (path === 'api/taste') {
        assert.equal(options.method, 'POST');
        await tastePost;
        return {ok: true, status: 200, json: async () => ({taste: {available: true, proposals: []}})};
      }
      return fetch(path, options);
    };
    const answering = card.querySelectorAll('button')[0].emit('click');
    if (failureTiming === 'before') {
      finishAnswer();
      await answering;
      assert.equal(errorNode.textContent, 'Try again');
    }
    const saving = page.saveTasteDecision({id: 'p1', sentence: 'Use short phrases.'}, 'approve');
    assert.equal(cards()[0], card);
    assert.equal(evidenceButtons()[0], evidenceButton);
    assert.equal(evidenceButton.disabled, true);
    assert.equal(errorNode.textContent, failureTiming === 'before' ? 'Try again' : '');
    if (failureTiming === 'during') {
      finishAnswer();
      await answering;
      assert.equal(cards()[0].querySelectorAll('.form-error')[0].textContent, 'Try again');
      assert.equal(evidenceButton.disabled, true);
    }
    finishTaste();
    await saving;
    assert.equal(cards()[0], card);
    assert.equal(evidenceButtons()[0], evidenceButton);
    // A pending answer still blocks navigation after the taste save completes.
    assert.equal(evidenceButton.disabled, failureTiming === 'after');
    if (failureTiming === 'after') {
      finishAnswer();
      await answering;
    }
    assert.equal(evidenceButtons()[0], evidenceButton);
    assert.equal(evidenceButton.disabled, false);
    assert.equal(cards()[0], card);
    assert.equal(textarea.value, 'The repeated words');
    assert.equal(Boolean(textarea.disabled), false);
    assert.equal(cards()[0].querySelectorAll('.form-error')[0].textContent, 'Try again');
  });
}

test('reply evidence abandons a stale version during the compare seek', async () => {
  const {page, context, element, thread, main, scrolls} = await replyPage({initial: 2});
  element('compare-video').readyState = 0;
  const opened = page.openReplyEvidence(thread);
  vm.runInContext("selectedVersion = 3; el.video.src = 'media?version=3&t=token';", context);
  element('video').currentTime = 7;
  element('status').textContent = 'New version selected';
  element('map-status').textContent = 'New comparison';
  await element('compare-video').emit('loadedmetadata');
  await opened;
  assert.equal(element('video').currentTime, 7);
  assert.equal(element('status').textContent, 'New version selected');
  assert.equal(element('map-status').textContent, 'New comparison');
  assert.equal(scrolls.length, 0);
  assert.equal(main.inert, false);
  assert.equal(vm.runInContext('openingReplyEvidence', context), false);
});

test('reply evidence ignores version buttons while navigation is pending', async () => {
  const {page, context, element, thread, requests, main} = await replyPage({initial: 2});
  element('compare-video').readyState = 0;
  const opened = page.openReplyEvidence(thread);
  await element('version-buttons').children[2].emit('click');
  await element('compare-video').emit('loadedmetadata');
  await opened;
  assert.equal(vm.runInContext('selectedVersion', context), 2);
  assert.deepEqual(requests, ['api/review?version=2']);
  assert.equal(element('video').currentTime, frameMid(296, {num: 24, den: 1}));
  assert.equal(main.inert, false);
});

test('reply evidence cancels stale media waits when either navigation changes', async () => {
  for (const playerId of ['compare-video', 'video']) {
    for (const invalidate of ['selectedVersion = 3;', 'loadVersion += 1;', 'setCompareOpen(true, 3);']) {
      const {page, context, element, thread, main, scrolls} = await replyPage({initial: 2});
      const waiting = element(playerId);
      waiting.readyState = 0;
      const opened = page.openReplyEvidence(thread);
      await Promise.resolve();
      assert.equal(waiting.listeners.get('loadedmetadata').length, playerId === 'video' ? 2 : 1);
      vm.runInContext(invalidate, context);
      waiting.currentTime = 7;
      element('status').textContent = 'New navigation';
      element('map-status').textContent = 'New comparison';
      await waiting.emit('loadedmetadata');
      await opened;
      assert.equal(waiting.currentTime, 7, `${playerId}: ${invalidate}`);
      assert.equal(element('status').textContent, 'New navigation');
      assert.equal(element('map-status').textContent, 'New comparison');
      assert.equal(scrolls.length, 0);
      assert.equal(main.inert, false);
    }
  }
});

test('reply evidence preserves newer status when a stale compare seek fails', async () => {
  const {page, context, element, thread, main, scrolls} = await replyPage({initial: 2});
  element('compare-video').readyState = 0;
  const opened = page.openReplyEvidence(thread);
  vm.runInContext('selectedVersion = 3;', context);
  element('status').textContent = 'New navigation';
  await element('compare-video').emit('error');
  await opened;
  assert.equal(element('status').textContent, 'New navigation');
  assert.equal(element('video').currentTime, 0);
  assert.equal(scrolls.length, 0);
  assert.equal(main.inert, false);
});

test('reply evidence respects busy and offline guards and waits for version selection', async () => {
  const {page, context, element, thread, requests, main, scrolls} = await replyPage();
  for (const flag of ['switchingVersion', 'composeBusy', 'roundBusy', 'releaseBusy', 'answerBusy']) {
    vm.runInContext(`${flag} = true; renderThreads();`, context);
    assert.equal(element('thread-list').children[0].children[1].children.at(-1).disabled, true);
    await page.openReplyEvidence(thread);
    assert.equal(requests.length, 1);
    assert.equal(scrolls.length, 0);
    vm.runInContext(`${flag} = false;`, context);
  }
  vm.runInContext('editState = {busy: true};', context);
  await page.openReplyEvidence(thread);
  vm.runInContext('editState = null;', context);
  vm.runInContext('review.media_changed = true;', context);
  await page.openReplyEvidence(thread);
  vm.runInContext('review.media_changed = false;', context);
  page.setOffline(true);
  assert.equal(element('thread-list').children[0].children[1].children.at(-1).disabled, true);
  await page.openReplyEvidence(thread);
  assert.equal(requests.length, 1);
  assert.equal(scrolls.length, 0);
  page.setOffline(false);
  let resolveSwitch;
  const fetch = context.fetch;
  context.fetch = async (path) => {
    await new Promise((resolve) => { resolveSwitch = resolve; });
    return fetch(path);
  };
  const opened = page.openReplyEvidence(thread);
  assert.equal(main.inert, true);
  assert.equal(element('compare-panel').hidden, undefined);
  assert.equal(element('video').currentTime, 0);
  assert.equal(scrolls.length, 0);
  await page.openReplyEvidence(thread);
  resolveSwitch();
  await opened;
  assert.equal(element('video').currentTime, frameMid(296, {num: 24, den: 1}));
  assert.equal(requests.length, 2);
  assert.equal(scrolls.length, 1);
});

test('reply evidence preserves version load errors and reports comparison media errors', async () => {
  const {page, context, element, thread, main, scrolls} = await replyPage();
  context.fetch = async () => ({ok: false, status: 404,
    json: async () => ({error: {message: 'Version is unavailable'}})});
  await page.openReplyEvidence(thread);
  assert.equal(element('status').textContent, 'Version is unavailable');
  assert.equal(element('video').getAttribute('src'), 'media?version=3&t=token');
  assert.equal(main.inert, false);
  assert.equal(scrolls.length, 0);
  const changed = await replyPage();
  const fetch = changed.context.fetch;
  changed.context.fetch = async (path) => {
    const response = await fetch(path);
    const fresh = await response.json();
    return {...response, json: async () => ({...fresh, media_changed: true})};
  };
  await changed.page.openReplyEvidence(changed.thread);
  assert.match(changed.element('status').textContent, /video changed after the page opened/);
  assert.equal(changed.element('video').currentTime, 0);
  assert.equal(changed.main.inert, false);
  assert.equal(changed.scrolls.length, 0);
  const same = await replyPage({initial: 2});
  same.element('compare-video').readyState = 0;
  const opened = same.page.openReplyEvidence(same.thread);
  await same.element('compare-video').emit('error');
  await opened;
  assert.equal(same.element('status').textContent, 'Could not load this version.');
  assert.equal(same.main.inert, false);
  assert.equal(same.timers.size, 0);
  assert.equal(same.scrolls.length, 0);
});

test('version badges distinguish current, newest and approved after rollback', () => {
  const series = {newest: 2, current_version: 1, approved: {version: 1}};
  assert.deepEqual(timelineHelpers.versionBadges(1, series), ['Current', 'Approved']);
  assert.deepEqual(timelineHelpers.versionBadges(2, series), ['Latest']);
  assert.deepEqual(timelineHelpers.versionBadges(3, series), []);
  assert.deepEqual(timelineHelpers.versionBadges(2, {...series, current_version: 2,
    approved: {version: 2}}), ['Latest', 'Current', 'Approved']);
});

test('report status and offline helpers preserve missing checks and HTTP reachability', () => {
  for (const [value, label] of [['PASS', 'Checks passed'], ['warn', 'Checks with warnings'], ['fail', 'Checks failed'],
    [null, 'Not checked'], [undefined, 'Not checked'], ['unknown', 'Not checked']]) {
    assert.equal(timelineHelpers.reportStatusLabel(value), label);
  }
  assert.equal(timelineHelpers.offlineState(false, 'network_error'), true);
  assert.equal(timelineHelpers.offlineState(true, 'response'), false);
  assert.equal(timelineHelpers.offlineState(true, 'pending'), true);
  assert.equal(reviewPollInterval([{status: 'sending'}], true), 5000);
});

test('page confirms rollback and approval and reconnects without losing notes', async () => {
  class Node {
    constructor() {
      this.children = [];
      this.attributes = {};
      this.listeners = {};
      this.dataset = {};
      this.value = '';
      this.textContent = '';
      this.style = {setProperty() {}};
      this.classList = {toggle() {}, add() {}, remove() {}};
    }
    addEventListener(type, fn) { this.listeners[type] = fn; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; delete this[key]; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    closest() { return null; }
    focus() {}
    pause() {}
  }
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  const requests = [];
  const confirmations = [];
  const intervals = [];
  let allow = false;
  let network = true;
  let httpError = false;
  let current = 2;
  let approved = null;
  const context = vm.createContext({
    ...timelineHelpers, URLSearchParams, Element: Node,
    location: new URL('http://review.example.net/?t=token&version=1'),
    history: {replaceState() {}},
    document: {getElementById: element, createElement: () => new Node(), addEventListener() {},
      visibilityState: 'visible'},
    window: {addEventListener() {}},
    setInterval(callback, delay) { intervals.push({callback, delay}); return intervals.length; },
    setTimeout() {}, clearTimeout() {},
    clearInterval() {},
    confirm(message) { confirmations.push(message); return allow; },
    fetch: async (path, options) => {
      requests.push({path, options});
      if (!network) throw new Error('Network unavailable');
      if (httpError) return {ok: false, status: 409, json: async () =>
        ({error: {message: 'This version failed its checks.'}})};
      if (path === 'api/rollback') current = JSON.parse(options.body).version;
      if (path === 'api/approve') approved = {version: JSON.parse(options.body).version,
        report_status: 'pass', approved_at: '2026-10-05T12:00:00Z'};
      return {ok: true, status: 200, json: async () => {
        if (path === 'api/rollback') return {version: `v${current}`, not_restored: []};
        if (path === 'api/approve') return approved;
        if (path.startsWith('api/removed')) return {available: true, version: 1, ranges: []};
        return {media: {name: 'sample-v1.mp4', sha256: 'hash1', duration_s: 1, cfr: false},
          comments: [{id: 'c1', text: 'Saved note', kind: 'fix', target: 'caption',
            start_frame: null, end_frame: null, start_s: 0, end_s: null, send: {state: 'draft'}}],
          previous: null, text: 'Saved note', series: {id: 'series', project: true, current: 1,
            newest: 2, current_version: current, approved, threads: [], rounds: [], questions: [],
            versions: [{number: 1, report_status: 'pass'}, {number: 2, report_status: 'warn'}]}};
      }};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  const runnable = source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.page = {start, releaseVersion, refreshReview, api, openCompose, submitCompose};');
  vm.runInContext(runnable, context);
  const page = context.page;
  await page.start();
  assert.equal(element('rollback-version').hidden, false);
  assert.equal(element('download-video').hidden, true);
  assert.equal(element('version-buttons').children[0].textContent, 'Version 1 · Checks passed');
  await page.releaseVersion('rollback');
  assert.equal(confirmations.at(-1), 'Make v1 current? The next notes build from v1.');
  assert.ok(!requests.some(({options}) => options.method === 'POST'));
  allow = true;
  await page.releaseVersion('rollback');
  assert.equal(element('rollback-version').hidden, true);
  await page.releaseVersion('approve');
  assert.match(confirmations.at(-1), /^Approve v1\?/);
  assert.equal(element('approve-version').hidden, true);
  assert.equal(element('download-video').href, 'api/download?version=1&t=token');
  assert.equal(element('download-report').href, 'api/download/report?version=1&t=token');
  const posts = requests.filter(({options}) => options.method === 'POST');
  assert.deepEqual(posts.map(({path, options}) => [path, JSON.parse(options.body),
    options.headers['X-Yap-Token']]), [['api/rollback', {version: 1}, 'token'],
    ['api/approve', {version: 1}, 'token']]);
  page.openCompose();
  element('compose-text').value = 'Unsent text survives reconnect';
  network = false;
  await page.refreshReview();
  assert.equal(element('offline-banner').hidden, false);
  for (const id of ['new-comment', 'send-notes', 'rollback-version', 'approve-version', 'compose-submit']) {
    assert.equal(element(id).disabled, true, id);
  }
  assert.equal(element('download-video').getAttribute('aria-disabled'), 'true');
  assert.equal(element('download-video').href, undefined);
  assert.equal(intervals.at(-1).delay, 5000);
  const offlinePosts = requests.filter(({options}) => options.method === 'POST').length;
  await page.submitCompose();
  await page.releaseVersion('rollback');
  assert.equal(requests.filter(({options}) => options.method === 'POST').length, offlinePosts);
  let prevented = false;
  element('download-video').listeners.click({preventDefault() { prevented = true; }});
  assert.equal(prevented, true);
  context.document.visibilityState = 'hidden';
  network = true;
  await intervals.at(-1).callback();
  assert.equal(element('offline-banner').hidden, true);
  assert.equal(element('compose-text').value, 'Unsent text survives reconnect');
  assert.equal(element('new-comment').disabled, false);
  assert.equal(element('send-notes').disabled, false);
  assert.equal(element('download-video').href, 'api/download?version=1&t=token');
  assert.equal(element('compose-submit').disabled, false);
  assert.equal(element('comment-count').textContent, '1');
  httpError = true;
  await assert.rejects(page.api('api/approve'), /This version failed its checks/);
  assert.equal(element('offline-banner').hidden, true);
  network = false;
  await assert.rejects(page.api('api/approve'), /Network unavailable/);
  assert.equal(element('offline-banner').hidden, false);
});

test('page polls and loads the review after an offline initial launch', async () => {
  class Node {
    constructor() {
      this.children = []; this.attributes = {}; this.value = ''; this.textContent = '';
      this.style = {setProperty() {}}; this.classList = {toggle() {}};
    }
    addEventListener() {}
    setAttribute(key, value) { this.attributes[key] = value; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
  }
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  let network = false;
  let poll;
  const context = vm.createContext({
    ...timelineHelpers, URLSearchParams, Element: Node,
    location: new URL('http://review.example.net/?t=token'),
    document: {getElementById: element, createElement: () => new Node(), addEventListener() {},
      visibilityState: 'visible'}, window: {addEventListener() {}},
    setInterval(callback, delay) { poll = callback; assert.equal(delay, 5000); return 1; },
    setTimeout() {}, clearTimeout() {},
    clearInterval() {}, fetch: async () => {
      if (!network) throw new Error('Network unavailable');
      return {ok: true, status: 200, json: async () => ({media: {name: 'sample.mp4',
        sha256: 'hash', duration_s: 1, cfr: false}, comments: [], previous: null, series: null, text: ''})};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.page = {start};'), context);
  await context.page.start();
  assert.equal(element('offline-banner').hidden, false);
  network = true;
  await poll();
  assert.equal(element('offline-banner').hidden, true);
  assert.equal(element('media-name').textContent, 'sample.mp4');
});

test('removed helpers name kinds, hide pauses and identify current restore requests', () => {
  for (const [kind, name] of Object.entries({filler: 'Filler', silence: 'Pause', stutter: 'Stutter',
    restart: 'Restart', manual: 'Manual cut', story: 'Cut from the story', trim: 'Trim'})) {
    assert.equal(removedKindName(kind), name);
  }
  const ranges = [{id: 's1.0', kind: 'filler', take: 't1', start: 1, end: 2},
    {id: 's1.1', kind: 'silence'}, {id: 's2.2', kind: 'story'}];
  assert.deepEqual(visibleRemovedRanges(ranges), [ranges[0], ranges[2]]);
  assert.deepEqual(visibleRemovedRanges(ranges, true), ranges);
  assert.deepEqual(visibleRemovedRanges([]), []);
  assert.equal(ranges.length, 3);
  const request = {type: 'restore', range: 's1.0', take: 't1', start: 1, end: 2, text: 'um well'};
  assert.equal(rangeRequested(ranges[0], [{request}]), true);
  assert.equal(rangeRequested(ranges[0], [{request, send: {state: 'sent'}}]), true);
  for (const change of [{type: 'spelling'}, {range: 's2.2'}, {take: 't2'}, {start: 0}, {end: 3}]) {
    assert.equal(rangeRequested(ranges[0], [{request: {...request, ...change}}]), false);
  }
  assert.equal(rangeRequested(ranges[0], [{request: null}, {}]), false);
  assert.equal(rangeRequested(ranges[0], []), false);
});

test('request descriptions and editing preserve restore identity and source bounds', () => {
  const request = {type: 'restore', range: 's1.0', take: 't1', start: 1, end: 2, text: 'um well'};
  assert.equal(requestText(request), 'Restore: removed part (2 words)');
  assert.equal(requestText({...request, text: '  um\twell\nagain  '}), 'Restore: removed part (3 words)');
  assert.equal(requestText({...request, text: 'um'}), 'Restore: removed part (1 word)');
  assert.equal(requestText({...request, text: ' \t\n '}), 'Restore: removed part (0 words)');
  assert.equal(requestText({...request, text: null}), 'Restore: removed audio');
  assert.equal(requestText({type: 'spelling', from: 'bad', to: 'good'}), 'Spelling: bad -> good');
  assert.equal(requestText(null), '');
  const edited = editedRequest({kind: 'fix', target: 'story', request}, '', '');
  assert.deepEqual(edited, request);
  assert.notEqual(edited, request);
  assert.equal(editedRequest({kind: 'fix', target: 'story', request: null}, '', ''), null);
  assert.deepEqual(editedRequest({kind: 'fix', target: 'caption'}, 'bad', 'good'),
    {type: 'spelling', from: 'bad', to: 'good'});
});

for (const cfr of [true, false]) {
  test(`page shows removed ranges, seeks and creates editable restore drafts at cfr=${cfr}`, async () => {
    let document;
    class Node {
      constructor(tag = '') {
        Object.assign(this, {tag, children: [], attributes: {}, listeners: {}, dataset: {},
          value: '', textContent: '', checked: false, hidden: false, open: false,
          currentTime: 0, paused: true, style: {setProperty() {}},
          classList: {toggle() {}, add() {}, remove() {}}});
      }
      addEventListener(name, handler) { this.listeners[name] = handler; }
      setAttribute(key, value) { this.attributes[key] = value; }
      getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; delete this[key]; }
      replaceChildren(...children) { this.children = children; }
      append(...children) { this.children.push(...children); }
      insertBefore(child, before) { this.children.splice(this.children.indexOf(before), 0, child); }
      closest() { return null; }
      focus() { document.activeElement = this; }
      pause() { this.paused = true; }
      setSelectionRange() {}
      querySelector(selector) {
        const matches = selector.startsWith('.')
          ? (node) => node.className?.split(' ').includes(selector.slice(1))
          : (node) => node.dataset.commentId === selector.match(/"([^"]+)"/)?.[1];
        for (const child of this.children) {
          if (matches(child)) return child;
          const found = child.querySelector(selector);
          if (found) return found;
        }
        return null;
      }
    }
    const elements = new Map();
    const element = (id) => {
      if (!elements.has(id)) elements.set(id, new Node());
      return elements.get(id);
    };
    document = {getElementById: element, createElement: (tag) => new Node(tag),
      addEventListener() {}, visibilityState: 'visible', activeElement: null};
    const ranges = [
      {id: 's1.0', take: 't1', start: 1, end: 2, at_s: 1, at_frame: 30,
        kind: 'story', text: '<b>cut</b>'},
      {id: 's1.1', take: 't1', start: 3, end: 4, at_s: 2, at_frame: 60,
        kind: 'silence', text: null},
      {id: 's1.2', take: 't1', start: 5, end: 6, at_s: 3, at_frame: 90,
        kind: 'trim', text: null},
    ];
    const comments = [];
    const calls = [];
    let project = true;
    const context = vm.createContext({
      ...timelineHelpers, URLSearchParams, Element: Node, document,
      location: new URL('http://review.example.net/v/2/?t=token&version=1'),
      history: {replaceState() {}}, window: {addEventListener() {}},
      setInterval() {}, clearInterval() {},
      setTimeout() {}, clearTimeout() {},
      fetch: async (path, options) => {
        calls.push({path, options});
        const url = new URL(path, 'http://review.example.net/v/2/');
        const version = Number(url.searchParams.get('version') || 1);
        const reply = (data, status = 200) => ({ok: true, status,
          json: async () => structuredClone(data)});
        if (url.pathname.endsWith('/api/removed')) {
          return reply({available: true, version, ranges: version === 1 ? ranges : []});
        }
        if (url.pathname.endsWith('/api/restore')) {
          const range = ranges.find((item) => item.id === JSON.parse(options.body).range);
          const request = {type: 'restore', range: range.id, take: range.take,
            start: range.start, end: range.end, text: range.text};
          const comment = {id: 'c1', kind: 'fix', target: 'story', text: 'Bring back this line',
            start_frame: cfr ? range.at_frame : null, start_s: range.at_s,
            end_frame: null, end_s: null, request, send: {state: 'draft'}};
          comments.push(comment);
          return reply(comment, 201);
        }
        if (options.method === 'PATCH') {
          Object.assign(comments[0], JSON.parse(options.body));
          return reply(comments[0]);
        }
        return reply({media: {name: 'render.mp4', sha256: `hash${version}`, duration_s: 10,
          cfr, fps: cfr ? '30/1' : null}, comments: version === 1 ? comments : [],
        previous: null, text: '', media_changed: false,
        series: {project, current: version, newest: 2, questions: [], rounds: [], threads: [],
          versions: [{number: 1}, {number: 2}]}});
      },
    });
    const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
    vm.runInContext(source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
      .replace(/start\(\);\s*$/, `globalThis.page = {start, loadReview, openEdit, saveEdit};`), context);
    await context.page.start();
    const section = element('removed-section');
    assert.equal(section.hidden, false);
    assert.equal(section.open, false);
    assert.equal(element('removed-count').textContent, '3');
    const list = element('removed-list');
    assert.equal(list.children.length, 2);
    assert.equal(list.children[0].querySelector('.removed-text').textContent, '<b>cut</b>');
    assert.equal(list.children[1].querySelector('.removed-text').textContent, 'Removed audio');
    let actions = list.children[0].querySelector('.removed-actions').children;
    await actions[0].listeners.click({stopPropagation() {}});
    assert.equal(element('video').currentTime, cfr ? 30.5 / 30 : 1);
    await actions[1].listeners.click({stopPropagation() {}});
    const posted = calls.find((call) => call.options.method === 'POST');
    assert.equal(posted.path, 'api/restore?version=1');
    assert.deepEqual(JSON.parse(posted.options.body), {range: 's1.0'});
    assert.equal(posted.options.headers['X-Yap-Token'], 'token');
    actions = list.children[0].querySelector('.removed-actions').children;
    assert.equal(actions[1].textContent, 'Requested');
    assert.equal(actions[1].disabled, true);
    let row = element('comment-list').children[0];
    assert.equal(row.querySelector('.send-state').textContent, 'Draft');
    assert.equal(row.querySelector('.restore-text').textContent, 'Restore: removed part (1 word)');
    context.page.openEdit(comments[0], row);
    const textarea = row.querySelector('.edit-box').children.find((node) => node.tag === 'textarea');
    textarea.value = 'Keep the whole sentence';
    await context.page.saveEdit();
    assert.equal(comments[0].request.type, 'restore');
    assert.equal(comments[0].text, 'Keep the whole sentence');
    row = element('comment-list').children[0];
    const clear = row.querySelector('.comment-actions').children.find((node) => node.textContent === 'Clear request');
    await clear.listeners.click({stopPropagation() {}});
    assert.equal(comments[0].request, null);
    assert.equal(list.children[0].querySelector('.removed-actions').children[1].textContent, 'Ask for it back');
    element('show-pauses').checked = true;
    element('show-pauses').listeners.change();
    assert.equal(list.children.length, 3);
    await context.page.loadReview(2);
    assert.equal(element('removed-count').textContent, '0');
    assert.equal(list.children.length, 0);
    project = false;
    await context.page.loadReview(1);
    assert.equal(section.hidden, true);
  });
}

test('spelling requests need Fix and Caption, trim words and omit empty fields', () => {
  assert.equal(spellingAvailable('fix', 'caption'), true);
  for (const [kind, target] of [['keep', 'caption'], ['fix', null], ['fix', 'story'],
    ['fix', 'speech'], ['fix', 'element'], ['fix', 'video']]) {
    assert.equal(spellingAvailable(kind, target), false);
    assert.equal(spellingRequest(kind, target, 'wrong', 'right'), null);
  }
  assert.equal(spellingRequest('fix', 'caption', '', ''), null);
  assert.equal(spellingRequest('fix', 'caption', '  ', '  '), null);
  assert.deepEqual(spellingRequest('fix', 'caption', ' tail net ', ' tailnet '),
    {type: 'spelling', from: 'tail net', to: 'tailnet'});
  assert.equal(spellingText({type: 'spelling', from: 'tail net', to: 'tailnet'}),
    'Spelling: tail net -> tailnet');
  assert.equal(spellingText(null), '');
  assert.equal(spellingText({type: 'restore'}), '');
});

test('spelling fields reject partial, identical, overlong and multiline words', () => {
  for (const [wrong, right] of [['wrong', ''], ['', 'right'], ['same', ' same '],
    ['x'.repeat(81), 'right'], ['wrong', 'x'.repeat(81)], ['two\nlines', 'right'],
    ['wrong', 'two\rlines'], ['wrong\n', 'right'], ['wrong', 'right\u2028word']]) {
    assert.throws(() => spellingRequest('fix', 'caption', wrong, right));
  }
  assert.equal(spellingRequest('fix', 'caption', 'a', 'b'.repeat(80)).to.length, 80);
  assert.equal([...spellingRequest('fix', 'caption', 'a', '🙂'.repeat(80)).to].length, 80);
  assert.throws(() => spellingRequest('fix', 'caption', 'a', '🙂'.repeat(81)));
});

test('page attaches spelling, edits and clears drafts, and renders sent notes read only', async () => {
  let document;
  class Node {
    constructor(tag = '') {
      this.tag = tag;
      this.children = [];
      this.attributes = {};
      this.listeners = {};
      this.dataset = {};
      this.value = '';
      this.textContent = '';
      this.style = {setProperty() {}};
      this.classList = {toggle() {}, add() {}, remove() {}};
    }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; delete this[key]; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    insertBefore(child, before) { this.children.splice(this.children.indexOf(before), 0, child); }
    closest() { return main; }
    focus() { document.activeElement = this; }
    pause() {}
    setSelectionRange() {}
    querySelector(selector) {
      const matches = selector.startsWith('.')
        ? (node) => node.className?.split(' ').includes(selector.slice(1))
        : (node) => node.dataset.commentId === selector.match(/"([^"]+)"/)?.[1];
      for (const child of this.children) {
        if (matches(child)) return child;
        const found = child.querySelector(selector);
        if (found) return found;
      }
      return null;
    }
  }
  const main = new Node();
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  document = {getElementById: element, createElement: (tag) => new Node(tag),
    addEventListener() {}, visibilityState: 'visible', activeElement: null};
  const comments = [];
  const requests = [];
  const context = vm.createContext({
    ...timelineHelpers, URLSearchParams, Element: Node, document,
    location: new URL('http://review.example.net/?t=token&version=1'),
    history: {replaceState() {}}, window: {addEventListener() {}},
    setInterval() {}, clearInterval() {},
    setTimeout() {}, clearTimeout() {},
    fetch: async (path, options) => {
      requests.push({path, options});
      if (options.method === 'POST') {
        const body = JSON.parse(options.body);
        comments.push({...body, id: `c${comments.length + 1}`, start_frame: null, end_frame: null,
          end_s: null, request: body.request ?? null, send: {state: 'draft'}});
        return {ok: true, status: 201, json: async () => comments.at(-1)};
      }
      if (options.method === 'PATCH') {
        const body = JSON.parse(options.body);
        Object.assign(comments[0], body);
        return {ok: true, status: 200, json: async () => comments[0]};
      }
      return {ok: true, status: 200, json: async () => ({
        media: {name: 'render.mp4', sha256: 'hash', duration_s: 1, cfr: false},
        comments: structuredClone(comments), previous: null, text: '', media_changed: false,
        series: {project: true, current: 1, newest: 1, questions: [], rounds: [],
          versions: [{number: 1}], threads: []},
      })};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, `globalThis.page = {start, openCompose, setComposeKind,
      setComposeTarget, submitCompose, openEdit, saveEdit, loadReview};`), context);
  await context.page.start();
  context.page.openCompose();
  assert.equal(element('compose-spelling').hidden, true);
  context.page.setComposeTarget('caption');
  assert.equal(element('compose-spelling').hidden, false);
  element('compose-text').value = 'Correct spelling';
  element('compose-spelling-from').value = 'tail net';
  element('compose-spelling-from').listeners.input();
  assert.equal(element('compose-spelling-from').required, true);
  assert.equal(element('compose-spelling-to').required, true);
  await context.page.submitCompose();
  assert.match(element('compose-error').textContent, /both need/);
  assert.equal(requests.filter((item) => item.options.method === 'POST').length, 0);
  element('compose-spelling-to').value = ' tailnet ';
  await context.page.submitCompose();
  const request = {type: 'spelling', from: 'tail net', to: 'tailnet'};
  assert.deepEqual(JSON.parse(requests.find((item) => item.options.method === 'POST').options.body).request,
    request);
  let row = element('comment-list').children[0];
  assert.equal(row.querySelector('.spelling-text').textContent, 'Spelling: tail net -> tailnet');
  context.page.openEdit(comments[0], row);
  let fields = row.querySelector('.spelling-fields');
  assert.equal(fields.hidden, false);
  assert.equal(fields.children[2].value, 'tail net');
  assert.equal(fields.children[4].value, 'tailnet');
  fields.children[4].value = 'tail-net';
  await context.page.saveEdit();
  assert.equal(comments[0].request.to, 'tail-net');
  row = element('comment-list').children[0];
  context.page.openEdit(comments[0], row);
  fields = row.querySelector('.spelling-fields');
  fields.children[2].value = '';
  fields.children[4].value = '';
  await context.page.saveEdit();
  assert.equal(comments[0].request, null);
  assert.equal(element('comment-list').children[0].querySelector('.spelling-text'), null);
  comments[0].request = request;
  comments[0].send = {state: 'sent', batch: 'b1'};
  await context.page.loadReview();
  row = element('comment-list').children[0];
  assert.equal(row.querySelector('.spelling-text').textContent, 'Spelling: tail net -> tailnet');
  assert.equal(row.children[0].children[1].disabled, true);
  assert.equal(row.children[1].disabled, true);
  assert.equal(row.querySelector('.comment-actions').children.length, 0);
  assert.equal(row.querySelector('.spelling-fields'), null);
  context.page.openCompose();
  context.page.setComposeTarget('caption');
  context.page.setComposeKind('keep');
  assert.equal(element('compose-spelling').hidden, true);
  context.page.setComposeKind('fix');
  element('compose-text').value = 'Caption appearance';
  await context.page.submitCompose();
  assert.equal(Object.hasOwn(JSON.parse(requests.filter((item) => item.options.method === 'POST')
    .at(-1).options.body), 'request'), false);
});

test('note send labels retain batch and error details', () => {
  assert.equal(sendStateLabel(undefined), 'Draft');
  assert.equal(sendStateLabel({state: 'draft'}), 'Draft');
  assert.equal(sendStateLabel({state: 'sending'}), 'Sending');
  assert.equal(sendStateLabel({state: 'sent', batch: 'b3'}), 'Sent');
  assert.equal(sendStateLabel({state: 'failed', error: {message: 'Temporary cause.'}}),
    'Failed: Temporary cause.');
});

test('send bar counts drafts and describes progress, failure and success', () => {
  assert.equal(draftNotesText([]), '0 draft notes');
  assert.equal(draftNotesText([{send: {state: 'draft'}}]), '1 draft note');
  const comments = ['draft', 'sent', 'sending', 'failed', 'draft'].map((state) => ({send: {state}}));
  assert.equal(draftNotesText(comments), '2 draft notes');
  assert.equal(roundStatusText({status: 'sending', step: 'Locating notes'}), 'Locating notes');
  assert.equal(roundStatusText({status: 'sending', step: 'Writing the batch'}), 'Writing the batch');
  assert.equal(roundStatusText({status: 'sending', step: null}), 'Sending notes');
  assert.equal(roundStatusText({status: 'failed', error: {message: 'Temporary cause.'}}),
    'Temporary cause.');
  assert.equal(roundStatusText({status: 'sent', comments: [{}, {}, {}], batch: 'b3'}),
    'Sent 3 notes');
  assert.equal(roundStatusText({status: 'sent', comments: [{}], batch: 'b1'}),
    'Sent 1 note');
  assert.equal(reviewPollInterval(), 5000);
  assert.equal(reviewPollInterval([{status: 'failed'}, {status: 'sent'}]), 5000);
  assert.equal(reviewPollInterval([{status: 'sent'}, {status: 'sending', version: 1}]), 1000);
});

test('question helpers preserve version identity and count option answers', () => {
  const questions = [
    {batch: 'b1', comment_version: 'v7', comment: {version: 2, id: 'c1'}, state: 'answered'},
    {batch: 'b2', comment_version: 'v3', comment: {version: 1, id: 'c1'}, state: 'asked'},
  ];
  assert.deepEqual(questionsForComment(questions, 2, 'c1'), [questions[0]]);
  assert.deepEqual(questionsForComment(undefined, 1, 'c1'), []);
  assert.deepEqual(questionAnswerBody(questions[0], '  The last line  '),
    {batch: 'b1', comment_version: 'v7', comment_id: 'c1', answer: 'The last line'});
  assert.deepEqual(questionAnswerBody(questions[0], null, 'caption'),
    {batch: 'b1', comment_version: 'v7', comment_id: 'c1', option: 'caption'});
  assert.equal(draftNotesText([{}, {}], questions), '2 draft notes, 1 answer');
  assert.equal(draftNotesText([], [...questions, questions[0]]), '0 draft notes, 2 answers');
  assert.equal(roundStatusText({status: 'sent', comments: [], answers: [{}], batch: 'b3'}),
    'Sent 0 notes and 1 answer');
  assert.equal(roundStatusText({status: 'sent', comments: [{}], answers: [{}, {}], batch: 'b3'}),
    'Sent 1 note and 2 answers');
});

test('page saves questions under notes and earlier threads and sends answers without drafts', async () => {
  class Node {
    constructor(tag = '') {
      this.tag = tag;
      this.children = [];
      this.attributes = {};
      this.listeners = {};
      this.dataset = {};
      this.value = '';
      this.textContent = '';
      this.style = {setProperty() {}};
      this.classList = {toggle() {}, add() {}, remove() {}};
    }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; delete this[key]; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    closest() { return main; }
    focus() {}
    pause() {}
    querySelectorAll() {
      return this.children.flatMap((child) => [
        ...(['button', 'textarea'].includes(child.tag) ? [child] : []), ...child.querySelectorAll(),
      ]);
    }
  }
  const main = new Node();
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  const current = {batch: 'b1', comment_version: 'v4', comment: {version: 2, id: 'c1'},
    state: 'asked', question: 'Which thing?', options: [{value: 'caption', label: 'The caption'}]};
  const earlier = {batch: 'b1', comment_version: 'v3', comment: {version: 1, id: 'c1'},
    state: 'asked', question: 'Which earlier moment?', options: []};
  let questions = [current, earlier];
  let fail = false;
  let answerRound = null;
  const requests = [];
  const comment = {id: 'c1', text: 'Change this', kind: 'fix', target: null,
    start_frame: null, end_frame: null, start_s: 0, end_s: null,
    send: {state: 'sent', batch: 'b1'}};
  const context = vm.createContext({
    ...timelineHelpers, URLSearchParams, Element: Node,
    location: new URL('http://review.example.net/?t=token&version=2'),
    history: {replaceState() {}},
    document: {getElementById: element, createElement: (tag) => new Node(tag),
      addEventListener() {}, visibilityState: 'visible'},
    window: {addEventListener() {}}, setInterval() {}, clearInterval() {},
    setTimeout() {}, clearTimeout() {},
    fetch: async (path, options) => {
      requests.push({path, options});
      if (path.startsWith('api/questions/answer')) {
        if (fail) return {ok: false, status: 400, json: async () => ({error: {message: 'Try again'}})};
        const body = JSON.parse(options.body);
        const question = questions.find((item) => item.comment_version === body.comment_version);
        question.state = 'answered';
        question.answer = body.answer ?? null;
        question.option = body.option ?? null;
      } else if (options.method === 'POST') {
        answerRound = {id: 'n2', version: 2, status: 'sent', batch: 'b2', comments: [],
          answers: questions.filter((item) => item.state === 'answered')};
        questions = questions.filter((item) => item.state !== 'answered');
      }
      return {ok: true, status: 200, json: async () => ({
        media: {name: 'render.mp4', sha256: 'hash', duration_s: 1, cfr: false},
        comments: [comment], previous: null, text: '', media_changed: false,
        series: {project: true, current: 2, newest: 2, questions: structuredClone(questions),
          rounds: answerRound ? [answerRound] : [], versions: [{number: 1}, {number: 2}],
          threads: [{...comment, version: 1, state: 'open'}]},
      })};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  const runnable = source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.page = {start, saveAnswer, submitRound, refreshReview};');
  vm.runInContext(runnable, context);
  await context.page.start();
  const noteCard = element('comment-list').children[0].children[5];
  const threadCard = element('thread-list').children[0].children[1];
  assert.equal(noteCard.children[0].textContent, 'Question');
  assert.equal(noteCard.children[1].textContent, 'Which thing?');
  assert.equal(noteCard.children[2].children[0].textContent, 'The caption');
  assert.equal(threadCard.children[1].textContent, 'Which earlier moment?');
  assert.equal(element('send-notes').disabled, true);
  const errorNode = noteCard.children[4];
  fail = true;
  await context.page.saveAnswer(current, null, 'caption', noteCard, errorNode, 'b1:v4:c1');
  assert.equal(errorNode.textContent, 'Try again');
  assert.equal(noteCard.children[2].children[0].disabled, false);
  assert.equal(noteCard.children[5].children[0].disabled, true);
  fail = false;
  await context.page.saveAnswer(current, null, 'caption', noteCard, errorNode, 'b1:v4:c1');
  const answerPost = requests.filter((item) => item.path.startsWith('api/questions/answer')).at(-1);
  assert.equal(answerPost.path, 'api/questions/answer?version=2');
  assert.deepEqual(JSON.parse(answerPost.options.body),
    {batch: 'b1', comment_version: 'v4', comment_id: 'c1', option: 'caption'});
  assert.equal(answerPost.options.headers['X-Yap-Token'], 'token');
  const answeredCard = element('comment-list').children[0].children[5];
  assert.equal(answeredCard.children[2].textContent, 'Answered; this goes with your next notes.');
  assert.equal(element('draft-count').textContent, '0 draft notes, 1 answer');
  assert.equal(element('send-notes').disabled, false);
  assert.equal(element('thread-list').children[0].children[1].children[3].disabled, false);
  await context.page.submitRound();
  assert.equal(requests.filter((item) => item.path.startsWith('api/rounds')).at(-1).options.body, '{}');
  assert.equal(element('round-list').children[0].children[0].textContent,
    'Sent 0 notes and 1 answer');
  assert.equal(element('comment-list').children[0].children.length, 5);
  assert.equal(element('draft-count').textContent, '0 draft notes');
  const remainingCard = element('thread-list').children[0].children[1];
  const textarea = remainingCard.children[3];
  textarea.value = '  The ending  ';
  textarea.listeners.input();
  assert.equal(remainingCard.children[5].children[0].disabled, false);
  await context.page.saveAnswer(earlier, textarea.value, null, remainingCard,
    remainingCard.children[4], 'b1:v3:c1');
  const textPost = requests.filter((item) => item.path.startsWith('api/questions/answer')).at(-1);
  assert.deepEqual(JSON.parse(textPost.options.body),
    {batch: 'b1', comment_version: 'v3', comment_id: 'c1', answer: 'The ending'});
  assert.equal(element('thread-list').children[0].children[1].children[2].textContent,
    'Answered; this goes with your next notes.');
});

test('page sends and retries the round version, polls progress, and locks rows', async () => {
  class Node {
    constructor() {
      this.children = [];
      this.attributes = {};
      this.dataset = {};
      this.value = '';
      this.textContent = '';
      this.style = {setProperty() {}};
      this.classList = {toggle() {}, add() {}, remove() {}};
    }
    addEventListener() {}
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; delete this[key]; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    closest() { return main; }
    focus() {}
    pause() {}
  }
  const main = new Node();
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  const requests = [];
  const intervals = [];
  let state = 'draft';
  let project = true;
  let seriesMode = true;
  const round = () => ({id: 'n1', version: 2, status: state, batch: state === 'sent' ? 'b1' : null,
    step: state === 'sending' ? 'Writing the batch' : null,
    comments: [{id: 'c1'}], error: state === 'failed' ? {message: 'Temporary cause.'} : null});
  const context = vm.createContext({
    ...timelineHelpers, URLSearchParams, Element: Node,
    location: new URL('http://review.example.net/v/2/?t=token&version=2'),
    history: {replaceState() {}},
    document: {getElementById: element, createElement: () => new Node(), addEventListener() {},
      visibilityState: 'visible'},
    window: {addEventListener() {}},
    setInterval(_callback, delay) { intervals.push(delay); return intervals.length; },
    setTimeout() {}, clearTimeout() {},
    clearInterval() {},
    fetch: async (path, options) => {
      requests.push({path, options});
      if (options.method === 'POST') state = 'sending';
      return {ok: true, status: options.method === 'POST' ? 202 : 200,
        json: async () => options.method === 'POST' ? round() : {
          media: {name: 'render.mp4', sha256: 'hash', duration_s: 1, cfr: false},
          comments: [{id: 'c1', text: 'Move caption', kind: 'fix', target: 'caption',
            start_frame: null, end_frame: null, start_s: 0, end_s: null,
            send: {state, round: state === 'draft' ? null : 'n1', batch: round().batch,
              error: round().error}}],
          previous: null, text: '',
          series: seriesMode ? {id: 'series', project, current: 2, newest: 2, threads: [],
            rounds: state === 'draft' ? [] : [round()], versions: [{number: 1}, {number: 2}]} : null,
        }};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  const runnable = source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.reviewPage = {start, submitRound, refreshReview, loadReview};');
  vm.runInContext(runnable, context);
  const page = context.reviewPage;
  await page.start();
  assert.equal(element('send-bar').hidden, false);
  assert.equal(element('draft-count').textContent, '1 draft note');
  assert.equal(element('send-notes').disabled, false);
  assert.equal(intervals.at(-1), 5000);
  await page.submitRound();
  const post = requests.find((item) => item.options.method === 'POST');
  assert.equal(post.path, 'api/rounds?version=2');
  assert.equal(post.options.body, '{}');
  assert.equal(post.options.headers['X-Yap-Token'], 'token');
  assert.equal(element('send-notes').disabled, true);
  assert.equal(element('round-list').children[0].children[0].textContent, 'Writing the batch');
  assert.equal(intervals.at(-1), 1000);
  for (const current of ['sending', 'sent']) {
    state = current;
    await page.refreshReview();
    const row = element('comment-list').children[0];
    assert.equal(row.children[0].children[1].disabled, true);
    assert.equal(row.children[1].disabled, true);
    assert.ok(row.children[4].children.every((button) => !['Edit', 'Delete'].includes(button.textContent)));
  }
  assert.equal(intervals.at(-1), 5000);
  assert.equal(element('round-list').children[0].children[0].textContent, 'Sent 1 note');
  state = 'failed';
  await page.refreshReview();
  const failedRow = element('comment-list').children[0];
  assert.equal(failedRow.children[0].children[1].disabled, false);
  assert.equal(failedRow.children[1].disabled, false);
  assert.equal(failedRow.children[2].textContent, 'Failed: Temporary cause.');
  assert.deepEqual(failedRow.children[4].children.map((button) => button.textContent), ['Edit', 'Delete']);
  assert.equal(element('round-list').children[0].children[1].textContent, 'Try again');
  await page.submitRound({...round(), version: 1});
  assert.equal(requests.filter((item) => item.options.method === 'POST').at(-1).path,
    'api/rounds/n1/retry?version=1');
  assert.equal(element('status').textContent, '');
  project = false;
  await page.refreshReview();
  assert.equal(element('send-bar').hidden, true);
  seriesMode = false;
  await page.refreshReview();
  assert.equal(element('send-bar').hidden, true);
});

test('saving a series comment re-enables version switching and retains its write version', async () => {
  class Node {
    constructor() {
      this.children = [];
      this.attributes = {};
      this.value = '';
      this.textContent = '';
      this.style = {setProperty() {}};
      this.classList = {toggle() {}, add() {}, remove() {}};
    }
    addEventListener() {}
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; delete this[key]; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    closest() { return main; }
    focus() {}
    pause() {}
  }
  const main = new Node();
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Node());
    return elements.get(id);
  };
  const requests = [];
  const context = vm.createContext({
    ...timelineHelpers, URLSearchParams, Element: Node,
    location: new URL('http://review.example.net/?t=token&version=1'),
    history: {replaceState() {}},
    document: {getElementById: element, createElement: () => new Node(), addEventListener() {}},
    window: {addEventListener() {}}, setInterval() {}, clearInterval() {},
    setTimeout() {}, clearTimeout() {},
    fetch: async (path, options) => {
      requests.push({path, options});
      const version = Number(new URL(path, 'http://review.example.net/').searchParams.get('version'));
      return {ok: true, status: options.method === 'POST' ? 201 : 200,
        json: async () => options.method === 'POST' ? {id: 'c1'} : {
          media: {name: `v${version}.mp4`, sha256: `hash${version}`, duration_s: 1, cfr: false},
          comments: [], previous: null, text: '',
          series: {id: 'series', current: version, newest: 2, threads: [],
            versions: [{number: 1}, {number: 2}]},
        }};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  const runnable = source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, 'globalThis.reviewPage = {start, submitCompose, selectVersion};');
  vm.runInContext(runnable, context);
  await context.reviewPage.start();
  element('compose-text').value = 'Earlier comment';
  await context.reviewPage.submitCompose();
  assert.equal(requests.find((item) => item.options.method === 'POST').path, 'api/comments?version=1');
  assert.ok(element('version-buttons').children.every((button) => !button.disabled));
  await context.reviewPage.selectVersion(2);
  assert.equal(element('video').src, 'media?version=2&t=token');
  assert.equal(element('copy-feedback').disabled, true);
});

test('thread state labels preserve clarification as unresolved', () => {
  for (const [state, label] of Object.entries({open: 'Open', changed: 'Changed', kept: 'Kept',
    declined: 'Declined', needs_clarification: 'Needs your answer'})) {
    assert.equal(threadStateLabel(state), label);
  }
  assert.equal(threadStateLabel('unknown'), 'Open');
});

test('map status messages explain movement, ambiguity, deletion and time fallback', () => {
  assert.equal(mapStatusMessage('mapped', 2), 'Same words');
  assert.equal(mapStatusMessage('moved', 2), 'Same words, from another part of the story');
  assert.equal(mapStatusMessage('ambiguous', 2), 'Said more than once in v2');
  assert.equal(mapStatusMessage('removed', 3), 'Removed in v3; shown where it was cut');
  assert.equal(mapStatusMessage('time', 2), 'No word map; matched by time');
});

test('version URLs preserve the token, video prefix, other fields and fragment', () => {
  assert.equal(versionUrl('api/review', 2), 'api/review?version=2');
  assert.equal(versionUrl('media', 1, 'a+b &'), 'media?version=1&t=a%2Bb+%26');
  assert.equal(versionUrl('/v/3/?t=token&version=1&other=yes#comments', 2),
    '/v/3/?t=token&version=2&other=yes#comments');
  assert.equal(versionUrl('api/comments/c1?version=1', 2), 'api/comments/c1?version=2');
  assert.equal(versionUrl('api/review?version=2&t=token', null), 'api/review?t=token');
  assert.equal(versionUrl('api/export.txt', null, 'token'), 'api/export.txt?t=token');
  assert.equal(versionUrl('api/review', undefined), 'api/review');
});

test('compare chooses a prior version or the next when viewing v1', () => {
  const versions = [{number: 1}, {number: 2}, {number: 3}];
  assert.equal(defaultCompareVersion(3, versions), 2);
  assert.equal(defaultCompareVersion(2, versions), 1);
  assert.equal(defaultCompareVersion(1, versions), 2);
  assert.equal(defaultCompareVersion(1, [{number: 1}]), null);
});

test('target names preserve scope order and use reviewer-facing names', () => {
  assert.deepEqual(Object.keys(TARGET_NAMES), ['story', 'speech', 'caption', 'element', 'video']);
  assert.deepEqual(Object.keys(TARGET_NAMES).map(targetName),
    ['Story', 'Speech', 'Caption', 'Graphic', 'Whole video']);
  assert.equal(targetName(null), 'No target');
  assert.equal(targetName(undefined), 'No target');
  assert.equal(targetName('unknown'), 'No target');
  assert.ok(Object.isFrozen(TARGET_NAMES));
});

test('frame rate parsing accepts positive integer ratios only', () => {
  assert.deepEqual(parseFps('30000/1001'), fps);
  assert.deepEqual(parseFps('25/1'), {num: 25, den: 1});
  for (const value of [null, '', '0/1', '30/0', '29.97', '1/2/3', '-1/1', '1.5/1', '9007199254740992/1']) {
    assert.equal(parseFps(value), null);
  }
});

test('frame rate labels match the text export', () => {
  for (const [rate, expected] of [
    [{num: 30, den: 1}, '30 fps'],
    [{num: 30000, den: 1001}, '29.97 fps'],
    [{num: 24000, den: 1001}, '23.98 fps'],
    [{num: 25, den: 2}, '12.5 fps'],
    [{num: 25, den: 1}, '25 fps'],
  ]) {
    assert.equal(fpsLabel(rate), expected);
  }
});

test('every frame in sixty seconds maps back from start, middle and a near-boundary time', () => {
  for (let frame = 0; frame < 1800; frame += 1) {
    const start = frameStart(frame, fps);
    assert.equal(frameAt(start, fps), frame);
    assert.equal(frameAt(frameMid(frame, fps), fps), frame);
    assert.equal(frameAt(start - 1e-6, fps), frame);
  }
  assert.equal(frameAt(-10, fps), 0);
  assert.equal(frameCount(19.986633, fps), 599);
});

test('time formatting floors instead of rounding', () => {
  assert.equal(tenthsText(0), '0:00.0');
  assert.equal(secondsText(59.99), '0:59.9');
  assert.equal(secondsText(3665.2), '61:05.2');
  for (const [frame, expected] of [
    [0, '0:00.0'], [96, '0:03.2'], [447, '0:14.9'],
    [966, '0:32.2'], [1323, '0:44.1'], [1538, '0:51.3'], [3000, '1:40.1'],
  ]) {
    assert.equal(frameTimeText(frame, fps), expected);
  }
});

test('point and range spans use exact frame boundaries or full player times', () => {
  assert.equal(spanText({start_frame: 447, end_frame: null, start_s: 14.9149, end_s: null}, fps), '0:14.9');
  assert.equal(spanText({start_frame: 1323, end_frame: 1538, start_s: 44.1, end_s: 51.3}, fps), '0:44.1-0:51.3');
  assert.equal(spanText({start_frame: null, end_frame: null, start_s: 14.948266, end_s: 15.123456}, null), '0:14.9-0:15.1');
  assert.equal(spanText({start_frame: 447, end_frame: 453, start_s: 14.948266, end_s: 15.123456}, null), '0:14.9-0:15.1');
});

test('sorting uses start, end, then numeric creation order without mutating input', () => {
  const point = (id, start_frame, end_frame = null) => ({id, start_frame, end_frame, start_s: 0, end_s: null});
  const comments = [point('c10', 5), point('c2', 5), point('c5', 5, 9), point('c4', 5, 8), point('c3', 3)];
  assert.deepEqual(sortComments(comments).map((item) => item.id), ['c3', 'c2', 'c10', 'c4', 'c5']);
  assert.equal(comments[0].id, 'c10');
  const variable = [
    {id: 'c2', start_frame: null, end_frame: null, start_s: 2, end_s: null},
    {id: 'c1', start_frame: null, end_frame: null, start_s: 1, end_s: null},
  ];
  assert.deepEqual(sortComments(variable).map((item) => item.id), ['c1', 'c2']);
});

test('inclusive frame marks swap and store a half-open end, with one frame as a point', () => {
  const marked = normalizeSelection(1323, 1537);
  assert.deepEqual(marked, {start_frame: 1323, end_frame: 1538});
  assert.deepEqual(normalizeSelection(1537, 1323), marked);
  assert.deepEqual(normalizeSelection(8, 8), {start_frame: 8, end_frame: null});
  assert.equal(spanText(marked, fps), '0:44.1-0:51.3');
});

test('player-time marks swap and equal marks make a point', () => {
  assert.deepEqual(normalizeTimeSelection(51.3, 44.1), {start_s: 44.1, end_s: 51.3});
  assert.deepEqual(normalizeTimeSelection(14.9, 14.9), {start_s: 14.9, end_s: null});
});

test('range lanes reuse a lane at a half-open boundary and stack overlaps', () => {
  const ranges = [
    {id: 'late', start: 10, end: 20},
    {id: 'first', start: 0, end: 10},
    {id: 'overlap', start: 5, end: 15},
    {id: 'third', start: 7, end: 12},
    {id: 'apart', start: 21, end: 23},
  ];
  const assigned = assignLanes(ranges);
  assert.deepEqual(assigned.map(({id, lane}) => [id, lane]), [
    ['first', 0], ['overlap', 1], ['third', 2], ['late', 0], ['apart', 0],
  ]);
  assert.deepEqual(ranges.map(({id}) => id), ['late', 'first', 'overlap', 'third', 'apart']);
  assert.ok(ranges.every((range) => !('lane' in range)));
  assert.deepEqual(assignLanes([{start: 0, end: 10}, {start: 10, end: 20}]).map(({lane}) => lane), [0, 0]);
  assert.deepEqual(assignLanes([{start: 0, end: 2}, {start: 3, end: 5}]).map(({lane}) => lane), [0, 0]);
});

test('review names the project, links home and says what happens after sending', () => {
  assert.equal(timelineHelpers.projectTitle('2026-10-19-filming-daily-v12.mp4'), '2026-10-19-filming-daily');
  assert.equal(timelineHelpers.projectTitle('plain.mov'), 'plain');
  const home = {url: 'http://127.0.0.1:5000/?t=a', phone_url: 'http://192.168.1.4:5000/?t=a', listening: true};
  assert.equal(timelineHelpers.homeUrl(home, '127.0.0.1'), home.url);
  assert.equal(timelineHelpers.homeUrl(home, '192.168.1.4'), home.phone_url);
  assert.equal(timelineHelpers.homeUrl({url: 'http://127.0.0.1:1/', phone_url: null}, 'box.ts.net'), 'http://127.0.0.1:1/');
  assert.equal(timelineHelpers.homeUrl(null, '127.0.0.1'), null);
  const series = {newest: 2, rounds: [{version: 2, status: 'sent', batch: 'b4'}]};
  assert.equal(timelineHelpers.sendNextText(series, null), null);
  assert.equal(timelineHelpers.sendNextText(series, home).state, 'listening');
  assert.match(timelineHelpers.sendNextText(series, home).text, /in a moment/);
  assert.equal(timelineHelpers.sendNextText(series, {...home, listening: false}).state, 'waiting');
  const got = timelineHelpers.sendNextText(series, {...home, listening: false, delivered: ['b4']});
  assert.equal(got.state, 'listening', 'a delivered batch counts even while the listener restarts');
  assert.match(got.text, /Claude has your notes/);
  assert.equal(timelineHelpers.sendNextText({newest: 3, rounds: series.rounds}, home), null);
  assert.equal(timelineHelpers.sendNextText({newest: 2, rounds: [{version: 2, status: 'failed'}]}, home), null);
});

test('a story cut offers the final instead of approval when Yap Studio home is running', () => {
  const series = {project: true, newest: 2, versions: [{number: 1, final: false}, {number: 2, final: false}]};
  const home = {url: 'http://127.0.0.1:1/', listening: true, final_requested: false};
  const newest = timelineHelpers.storyCutStep(series, 2, home);
  assert.deepEqual([newest.hideApprove, newest.offerFinal, newest.requested], [true, true, false]);
  assert.match(newest.hint, /story cut/);
  assert.equal(timelineHelpers.storyCutStep(series, 1, home).offerFinal, false, 'only the newest cut goes on');
  const asked = timelineHelpers.storyCutStep(series, 2, {...home, final_requested: true});
  assert.deepEqual([asked.offerFinal, asked.requested], [true, true]);
  const final = {...series, versions: [...series.versions, {number: 3, final: true}], newest: 3};
  assert.deepEqual([timelineHelpers.storyCutStep(final, 3, home).hideApprove,
    timelineHelpers.storyCutStep(final, 3, home).offerFinal], [false, false]);
  assert.equal(timelineHelpers.storyCutStep(series, 2, null).hideApprove, false, 'without home, approval works as before');
  assert.equal(timelineHelpers.storyCutStep({...series, versions: [{number: 2}]}, 2, home).hideApprove, false,
    'an older server without the final flag keeps approval');
});
