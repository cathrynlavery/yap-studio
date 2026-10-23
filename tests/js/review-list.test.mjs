import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import * as timeline from '../../src/yap_studio/review/static/timeline.mjs';

const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');

class Element {
  constructor() {
    this.value = '';
    this.textContent = '';
    this.style = {setProperty() {}};
    this.classList = {toggle() {}, add() {}, remove() {}};
    this.children = [];
  }
  addEventListener() {}
  setAttribute(name, value) { this[name] = value; }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  append(...nodes) {
    this.children.push(...nodes);
    for (const node of nodes) node.parent = this;
  }
  after(node) {
    const index = this.parent.children.indexOf(this);
    this.parent.children.splice(index + 1, 0, node);
    node.parent = this.parent;
  }
  remove() {
    this.parent.children = this.parent.children.filter((node) => node !== this);
  }
  focus() {}
}

function findingsPage() {
  const elements = new Map();
  const getElement = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const container = new Element();
  container.append(getElement('findings-section'));
  getElement('findings-section').append(getElement('findings-list'));
  const context = vm.createContext({
    ...timeline, URLSearchParams, location: new URL('http://review.example.net/?t=video-token'),
    document: {getElementById: getElement, createElement: () => new Element(), addEventListener() {}},
    window: {addEventListener() {}}, setInterval() {},
  });
  const runnable = source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
    .replace(/start\(\);\s*$/, `globalThis.render = (findings) => {
      review = {media: {cfr: true, duration_s: 3, sha256: 'sample'}, comments: [], findings};
      fps = parseFps('30/1');
      renderFindings();
      renderMarkers();
    };`);
  vm.runInContext(runnable, context);
  return {render: context.render, getElement, container};
}

test('caption contrast findings say Hard to read in the list and markers', () => {
  const {render, getElement} = findingsPage();
  render({available: true, items: [{id: 'l1', kind: 'caption_contrast', source: 'machine',
    severity: 'error', reason: 'Words nearly vanish', fix: 'Back this phrase',
    start_frame: 15, end_frame: 16, start_s: 0.5, end_s: 16 / 30, target: 'caption'}]});
  const row = getElement('findings-list').children[0];
  assert.equal(row.children[0].children[1].textContent, 'Hard to read');
  assert.match(getElement('finding-markers').children[0]['aria-label'], /Hard to read/);
});

test('layer check findings show the label, pass, severity and literal message', () => {
  const {render, getElement} = findingsPage();
  const finding = {id: 'l1', kind: 'layer_check', source: 'machine', severity: 'warning',
    pass: 'layout', check: 'layout', reason: '<b>Text overlaps</b>', fix: 'Review the layer.',
    start_frame: 15, end_frame: 16, start_s: 0.5, end_s: 16 / 30, target: 'caption'};
  render({available: true, items: [finding], layer_checks: []});
  assert.equal(getElement('findings-section').hidden, false);
  assert.equal(getElement('findings-count').textContent, '1');
  const row = getElement('findings-list').children[0];
  const heading = row.children[0].children;
  assert.equal(heading[1].textContent, 'Layer check');
  assert.equal(heading[2].textContent, 'Severity: warning');
  assert.equal(heading[3].textContent, 'Automatic check');
  assert.equal(heading[4].textContent, 'layout');
  assert.equal(row.children[1].textContent, '<b>Text overlaps</b>');
  assert.deepEqual(timeline.findingNoteBody(finding, true), {
    kind: 'fix', target: 'caption', start_frame: 15, end_frame: 16,
    text: 'Flagged: <b>Text overlaps</b> Suggested fix: Review the layer.',
  });
});

test('hidden cards show the label and literal text in the list and timeline markers', () => {
  const {render, getElement} = findingsPage();
  const finding = {id: 'l1', kind: 'behind_hidden', source: 'machine', severity: 'warning',
    reason: 'Card <b>agent-card</b> is mostly hidden behind the speaker for 40 percent of its time',
    fix: 'Move the card or set it in front of the speaker',
    start_frame: 45, end_frame: 75, start_s: 1.5, end_s: 2.5, target: 'element',
    evidence: {id: '<b>agent-card</b>', frames: 30, hidden_frames: 12,
      hidden_share: 0.4, flag: true}};
  render({available: true, items: [finding]});
  assert.equal(getElement('findings-section').hidden, false);
  assert.equal(getElement('findings-count').textContent, '1');
  const row = getElement('findings-list').children[0];
  assert.equal(row.children[0].children[1].textContent, 'Hidden behind speaker');
  assert.equal(row.children[0].children[2].textContent, 'Severity: warning');
  assert.equal(row.children[1].textContent, finding.reason);
  assert.equal(row.children[2].textContent, `Suggested fix: ${finding.fix}`);
  const [marker] = getElement('finding-markers').children;
  assert.equal(marker.className, 'range-bar finding-range');
  assert.ok(marker.title.includes('Hidden behind speaker'));
  assert.ok(marker.title.includes(finding.reason));
  assert.equal(marker['aria-label'], marker.title);
  assert.deepEqual(timeline.findingNoteBody(finding, true), {
    kind: 'fix', target: 'element', start_frame: 45, end_frame: 75,
    text: `Flagged: ${finding.reason} Suggested fix: ${finding.fix}`,
  });
});

for (const layerStatus of ['incomplete', 'findings', 'passed']) {
  test(`skipped layer checks with status ${layerStatus} show one quiet line inside the checks fold`, () => {
    const {render, getElement, container} = findingsPage();
    const findings = {available: true, items: [], layer_checks: [
      {path: 'captions/captions.mov', status: 'passed', skipped: []},
      {path: 'graphics/agent-title.mov', status: layerStatus, skipped: [
        {pass: 'motion', reason: 'motion check not enabled'},
        {pass: 'contrast', reason: 'no text on an opaque background in this layer; '
          + 'captions are measured on the final <b>yet</b>'},
      ]},
    ]};
    render(findings);
    const section = getElement('findings-section');
    assert.equal(section.children[0], getElement('findings-list'));
    assert.equal(container.children[0], section);
    assert.equal(container.children.length, 1);
    assert.equal(section.children.length, 2);
    const summary = section.children[1];
    assert.equal(summary.className, 'finding-evidence');
    assert.equal(summary.textContent, 'Some layer checks did not run: '
      + 'graphics/agent-title.mov: motion (motion check not enabled); '
      + 'graphics/agent-title.mov: contrast (no text on an opaque background in this layer; '
      + 'captions are measured on the final <b>yet</b>).');
    assert.equal(section.hidden, false);
    render(findings);
    assert.equal(section.children.length, 2);
    render({available: false, items: [], layer_checks: []});
    assert.equal(section.children.length, 1);
    assert.equal(section.hidden, true);
  });
}

test('unlisted errors render as a layer check over the whole layer', () => {
  const {render, getElement} = findingsPage();
  const finding = {id: 'l1', kind: 'layer_check', source: 'machine', severity: 'error',
    pass: 'layout', reason: '3 more layout errors not listed', fix: 'Review the layer.',
    start_frame: 6, end_frame: 39, start_s: 0.2, end_s: 1.3, target: 'element'};
  render({available: true, items: [finding]});
  const row = getElement('findings-list').children[0];
  assert.equal(row.children[0].children[1].textContent, 'Layer check');
  assert.equal(row.children[0].children[2].textContent, 'Severity: error');
  assert.equal(row.children[1].textContent, '3 more layout errors not listed');
  assert.equal(row.children[0].children[0].textContent,
    timeline.spanText(finding, timeline.parseFps('30/1')));
});

for (const available of [true, false]) {
  test(`stale layer receipts show one quiet line with available=${available}`, () => {
    const {render, getElement, container} = findingsPage();
    const findings = {available, items: available ? [{id: 'l1', kind: 'layer_check',
      source: 'machine', pass: 'layout', severity: 'warning', reason: 'Text overlaps',
      fix: 'Review the layer.', start_frame: 15, end_frame: 16, start_s: 0.5, end_s: 16 / 30}] : [],
      stale: ['private/earlier.findings.json'], layer_stale: ['captions/receipt.json',
      'graphics/receipt<b>.json']};
    render(findings);
    assert.equal(getElement('findings-section').hidden, !available);
    assert.equal(getElement('findings-count').textContent, available ? '1' : '0');
    assert.equal(getElement('findings-status').hidden, available);
    assert.equal(getElement('findings-status').textContent, available ? ''
      : "The agent's notes are for an earlier render of this version.");
    const section = getElement('findings-section');
    assert.equal(container.children.length, 1);
    assert.equal(section.children.length, 2);
    const summary = section.children[1];
    assert.equal(summary.className, 'finding-evidence');
    assert.equal(summary.textContent, 'Layer check receipts do not match this render: '
      + 'captions/receipt.json, graphics/receipt<b>.json.');
    render(findings);
    assert.equal(section.children.length, 2);
    render({...findings, layer_stale: []});
    assert.equal(section.children.length, 1);
  });
}

test('skipped checks and stale layer receipts keep independent quiet lines', () => {
  const {render, getElement} = findingsPage();
  const findings = {available: true, items: [], layer_stale: ['captions/receipt.json'],
    layer_checks: [{path: 'graphics/agent-card.mov', status: 'findings', skipped: [
      {pass: 'motion', reason: 'motion check not enabled'},
    ]}]};
  render(findings);
  const section = getElement('findings-section');
  assert.equal(section.children.length, 3);
  const lines = section.children.slice(1).map((node) => node.textContent);
  assert.ok(lines.some((text) => text.includes('motion (motion check not enabled)')));
  assert.ok(lines.some((text) => text.includes('captions/receipt.json')));
  render({...findings, available: false, layer_checks: []});
  assert.equal(section.children.length, 2);
  assert.ok(section.children[1].textContent.includes('captions/receipt.json'));
  render({...findings, layer_stale: []});
  assert.equal(section.children.length, 2);
  assert.ok(section.children[1].textContent.includes('motion (motion check not enabled)'));
  render({available: false, items: []});
  assert.equal(section.children.length, 1);
});

for (const prefix of ['/', '/v/2/']) {
  test(`review requests and links stay under ${prefix}`, async () => {
    const base = new URL(`http://review.example.net${prefix}?t=video-token`);
    const elements = new Map();
    const getElement = (id) => {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    };
    const calls = [];
    const context = vm.createContext({
      ...timeline, URLSearchParams, location: base,
      document: {getElementById: getElement, addEventListener() {}},
      window: {addEventListener() {}}, setInterval() {}, confirm: () => true,
      setTimeout() {}, clearTimeout() {},
      fetch: async (route, options) => {
        calls.push({url: new URL(route, base), options});
        return {ok: true, status: options.method === 'DELETE' ? 204 : 200,
          json: async () => options.method === 'POST' ? {id: 'c1'} : {
            media: {name: 'sample <b>.mp4', duration_s: 1, cfr: false},
            comments: [], previous: null, text: '',
          }};
      },
    });
    // Run the page functions with a small DOM so routing is checked without browser dependencies.
    const runnable = source.replace(/^import \{[\s\S]*?\} from '.\/timeline.mjs';/, '')
      .replace(/start\(\);\s*$/, 'globalThis.page = {start, submitCompose, changeComment, deleteComment};');
    vm.runInContext(runnable, context);
    await context.page.start();
    assert.equal(calls[0].url.pathname, `${prefix}api/review`);
    assert.equal(getElement('status').textContent, '');
    assert.equal(getElement('media-name').textContent, 'sample <b>.mp4');
    const url = new URL(getElement('video').src, base);
    assert.equal(url.pathname, `${prefix}media`);
    assert.equal(url.searchParams.get('t'), 'video-token');
    getElement('compose-text').value = 'Synthetic comment';
    await context.page.submitCompose();
    await context.page.changeComment('c1', {text: 'Updated'});
    await context.page.deleteComment('c1');
    for (const method of ['POST', 'PATCH', 'DELETE']) {
      const call = calls.find(({options}) => options.method === method);
      assert.ok(call, `${method} was sent`);
      assert.equal(call.url.pathname, `${prefix}api/comments${method === 'POST' ? '' : '/c1'}`);
    }
    for (const call of calls) assert.equal(call.options.headers['X-Yap-Token'], 'video-token');
  });
}
