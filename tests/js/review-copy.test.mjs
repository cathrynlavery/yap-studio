import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import * as helpers from '../../src/yap_studio/review/static/timeline.mjs';

const root = new URL('../../src/yap_studio/review/static/', import.meta.url);
const html = readFileSync(new URL('index.html', root), 'utf8');
const source = readFileSync(new URL('app.mjs', root), 'utf8');

class Element {
  constructor(attributes = '') {
    this.hidden = /\bhidden\b/u.test(attributes);
    this.disabled = /\bdisabled\b/u.test(attributes);
    this.checked = false;
    this.listeners = {};
    this.children = [];
    this.style = {setProperty() {}};
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute(name, value) { this[name] = value; }
  replaceChildren(...nodes) { this.children = nodes; }
  querySelectorAll() { return []; }
  focus() { this.focused = true; }
  select() { this.selected = true; }
  showModal() { this.open = true; }
  async click() {
    if (!this.disabled) await this.listeners.click?.();
  }
}

function copyPage({project = false, comments = [], threads = [], text = '0 comments\n',
  textEarlier = null, earlierCount = 0, clipboard = 'available'} = {}) {
  const elements = new Map([...html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/gu)]
    .map(([tag, id]) => [id, new Element(tag)]));
  const copied = [];
  const document = {getElementById: (id) => elements.get(id), addEventListener() {},
    querySelectorAll: () => [], createElement: () => new Element()};
  const context = vm.createContext({...helpers, document, URLSearchParams,
    location: new URL('http://localhost/?t=review-token'), window: {addEventListener() {}},
    setTimeout() {}, clearTimeout() {},
    navigator: clipboard === 'missing' ? {} : {clipboard: {writeText(value) {
      if (clipboard === 'rejected') return Promise.reject(new Error('Clipboard unavailable'));
      if (clipboard === 'throws') throw new Error('Clipboard unavailable');
      copied.push(value);
      return Promise.resolve();
    }}},
  });
  const runnable = source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/u, '')
    .replace(/start\(\);\s*$/u, `globalThis.render = (value) => {
      review = value;
      renderHeader();
      renderSendBar();
    };
    wireEvents();`);
  vm.runInContext(runnable, context);
  const review = {media: {name: 'sample.mp4', duration_s: 10}, comments, text,
    text_earlier: textEarlier, earlier_count: earlierCount,
    series: project || threads.length ? {project, threads, rounds: [], questions: [], versions: []} : null};
  context.render(review);
  return {get: (id) => elements.get(id), copied, review, render: context.render};
}

for (const project of [false, true]) {
  test(`one Copy notes button is visible in the header with project=${project}`, () => {
    const page = copyPage({project});
    assert.equal([...html.matchAll(/<button\b[^>]*>Copy notes<\/button>/gu)].length, 1);
    const header = html.match(/<header class="header">([\s\S]*?)<\/header>/u)[1];
    assert.match(header, /id="copy-feedback"/u);
    assert.doesNotMatch(header, /<details\b/u);
    assert.doesNotMatch(html, /id="(?:copy-notes|notes-bar|export-text|export-json|include-earlier)"/u);
    assert.doesNotMatch(html, /Share notes|Download as text|Download as JSON|Include earlier versions/u);
    assert.equal(page.get('copy-feedback').hidden, false);
    assert.equal(page.get('send-bar').hidden, !project);
  });
}

test('one draft enables copying in one click and clears the empty hint', async () => {
  const page = copyPage({project: true, comments: [{send: {state: 'draft'}}],
    text: '1 comment: 1 fix\n\n0:01.0 [Fix] Move the caption.\n'});
  assert.equal(page.get('copy-feedback').disabled, false);
  assert.equal(page.get('copy-notes-hint').hidden, true);
  await page.get('copy-feedback').click();
  assert.deepEqual(page.copied, [page.review.text]);
  assert.equal(page.get('toast').textContent, 'Copied 1 comment');
});

test('one click copies the complete text for draft and sent notes', async () => {
  const text = 'Feedback on sample.mp4\n2 comments: 1 fix, 1 keep\n\n'
    + '0:01.0 [Fix] Move the caption.\n0:02.0 [Keep] Keep this pause.\n';
  const page = copyPage({project: true, text,
    comments: [{send: {state: 'draft'}}, {send: {state: 'sent'}}]});
  await page.get('copy-feedback').click();
  assert.deepEqual(page.copied, [text]);
  assert.equal(page.get('toast').textContent, 'Copied 2 comments');
});

test('no notes disables copying with a short visible hint', async () => {
  const page = copyPage();
  assert.equal(page.get('copy-feedback').disabled, true);
  assert.equal(page.get('copy-notes-hint').hidden, false);
  assert.match(html, /id="copy-notes-hint"[^>]*>Add a note to copy\./u);
  await page.get('copy-feedback').click();
  assert.deepEqual(page.copied, []);
});

test('earlier threads alone do not enable copying the current version', async () => {
  const page = copyPage({threads: [{id: 'c1'}], earlierCount: 1,
    text: '0 comments\nEarlier versions: 1 comment\n',
    textEarlier: '0 comments\nEarlier versions: 1 comment\n0:01.0 [Fix] Earlier note.\n'});
  assert.equal(page.get('copy-feedback').disabled, true);
  assert.equal(page.get('copy-notes-hint').hidden, false);
  await page.get('copy-feedback').click();
  assert.deepEqual(page.copied, []);
});

test('sent notes alone enable copying and earlier note text is excluded', async () => {
  const page = copyPage({comments: [{send: {state: 'sent'}}], threads: [{id: 'c1'}],
    text: '0:01.0 [Fix] Current note.\n', textEarlier: 'Earlier note.\n', earlierCount: 1});
  assert.equal(page.get('copy-feedback').disabled, false);
  await page.get('copy-feedback').click();
  assert.deepEqual(page.copied, [page.review.text]);
  assert.equal(page.get('toast').textContent, 'Copied 1 comment');
});

test('the earlier-review banner keeps its separate copy action', async () => {
  const page = copyPage();
  page.review.previous = {text: 'Previous review note.\n', count: 1};
  await page.get('previous-copy').click();
  assert.deepEqual(page.copied, [page.review.previous.text]);
  assert.equal(page.get('copy-feedback').disabled, true);
});

for (const clipboard of ['missing', 'rejected', 'throws']) {
  test(`visible copy uses the fallback dialog when the clipboard is ${clipboard}`, async () => {
    const page = copyPage({comments: [{}], text: '0:01.0 [Fix] Copy me.\n', clipboard});
    await page.get('copy-feedback').click();
    assert.equal(page.get('copy-fallback').open, true);
    assert.equal(page.get('copy-fallback-text').value, page.review.text);
    assert.equal(page.get('copy-fallback-text').selected, true);
  });
}

test('copy availability updates when the last note is removed', () => {
  const page = copyPage({comments: [{}], text: '0:01.0 [Fix] Copy me.\n'});
  assert.equal(page.get('copy-feedback').disabled, false);
  page.review.comments = [];
  page.render(page.review);
  assert.equal(page.get('copy-feedback').disabled, true);
  assert.equal(page.get('copy-notes-hint').hidden, false);
});
