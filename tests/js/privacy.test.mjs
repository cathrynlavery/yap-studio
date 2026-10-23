import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import * as homeModel from '../../src/yap_studio/homepage/static/home-model.mjs';
import * as timeline from '../../src/yap_studio/review/static/timeline.mjs';

const homeRoot = new URL('../../src/yap_studio/homepage/static/', import.meta.url);
const reviewRoot = new URL('../../src/yap_studio/review/static/', import.meta.url);
const disclosure = 'Your footage, renders and review files stay on your computer. '
  + 'The transcript, your notes and the project files Claude reads go to Claude (Anthropic) '
  + 'as part of your Claude Code session. ElevenLabs, WaveSpeed and Pexels get only what '
  + 'they need, and only if you turn them on.';

class Element {
  constructor() {
    this.children = [];
    this.textContent = '';
    this.classList = {toggle() {}};
  }
  addEventListener() {}
  setAttribute(name, value) { this[name] = value; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
}

function page(root, file, helpers, setup, hostname = 'localhost') {
  const elements = new Map();
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const context = vm.createContext({...helpers, Node: Element, URLSearchParams,
    location: {hostname, search: '?t=synthetic'}, window: {addEventListener() {}},
    document: {getElementById: get, addEventListener() {},
      createElement: () => new Element(), createTextNode: (text) => text},
  });
  const source = readFileSync(new URL(file, root), 'utf8')
    .replace(/^import \{[\s\S]*?\} from '[^']+';\s*/gmu, '')
    .replace(/(?:start\(\)|refresh\(\)\.then\(schedule\));\s*$/u, setup);
  vm.runInContext(source, context);
  return {context, get};
}

function text(node) {
  return typeof node === 'string' ? node : node.textContent + node.children.map(text).join(' ');
}

test('the first home screen displays the complete disclosure before starting a video', () => {
  const {context} = page(homeRoot, 'home.js', homeModel,
    'globalThis.welcome = renderWelcome();');
  assert.ok(text(context.welcome).includes(disclosure));
});

for (const [via, label] of [['tailscale', 'Encrypted (Tailscale)'],
  ['wifi', 'Not encrypted (same Wi-Fi)']]) {
  test(`home shows the ${via} phone label beside its unchanged link`, () => {
    const phone = {via, url: 'http://synthetic.test/?t=synthetic'};
    const {context, get} = page(homeRoot, 'home.js', homeModel,
      'globalThis.renderPhone = (phone) => {state = {phone, requests: []}; renderListener();};');
    context.renderPhone(phone);
    assert.equal(get('phone-link').hidden, false);
    assert.equal(get('phone-link').href, phone.url);
    assert.equal(get('phone-link').textContent, `Open on your phone: ${label}`);
    assert.match(readFileSync(new URL('home.html', homeRoot), 'utf8'), /id="phone-link"/u);
    context.renderPhone(null);
    assert.equal(get('phone-link').hidden, true);
  });

  test(`review shows the ${via} phone label from the server record`, () => {
    const phone = {via, url: 'http://synthetic.test/?t=synthetic'};
    const {context, get} = page(reviewRoot, 'app.mjs', timeline,
      'globalThis.renderPhone = (phone) => {review = {phone, media: {name: "sample.mp4", duration_s: 2}, comments: []}; renderHeader();};');
    context.renderPhone(phone);
    assert.equal(get('phone-link').hidden, false);
    assert.equal(get('phone-link').href, phone.url);
    assert.equal(get('phone-link').textContent, `Open on your phone: ${label}`);
    assert.match(readFileSync(new URL('index.html', reviewRoot), 'utf8'), /id="phone-link"/u);
    context.renderPhone(null);
    assert.equal(get('phone-link').hidden, true);
  });

  test(`home labels review navigation over ${via}`, () => {
    const {context} = page(homeRoot, 'home.js', homeModel,
      'globalThis.video = (project) => {state = {projects: [project], requests: []}; return renderVideo({name: project.name});};',
      'synthetic.test');
    const project = {name: 'synthetic', dir: 'projects/synthetic', phase: 'final_ready',
      review: {url: 'http://localhost/', phone_url: 'http://synthetic.test/', phone_via: via}};
    assert.ok(text(context.video(project)).includes(`Watch and approve · ${label}`));
  });

  test(`review labels the home navigation over ${via}`, () => {
    const {context, get} = page(reviewRoot, 'app.mjs', timeline,
      'globalThis.show = (home) => {review = {home, media: {name: "sample.mp4", duration_s: 2}, comments: []}; renderHeader();};',
      'synthetic.test');
    context.show({url: 'http://localhost/', phone_url: 'http://synthetic.test/', phone_via: via});
    assert.equal(get('home-link').textContent, `Back to Yap Studio · ${label}`);
  });

  test(`style labels both home links over ${via}`, () => {
    const source = readFileSync(new URL('../../src/yap_studio/gallery/static/pick.js', import.meta.url), 'utf8')
      .split('function showHomeLinks(home) {')[1].split('async function boot()')[0];
    const elements = {'home-link': new Element(), 'saved-home': new Element()};
    const context = vm.createContext({$: (id) => elements[id], location: {hostname: 'synthetic.test'}});
    vm.runInContext('function showHomeLinks(home) {' + source + '; globalThis.show = showHomeLinks;', context);
    context.show({url: 'http://localhost/', phone_url: 'http://synthetic.test/', phone_via: via});
    for (const link of Object.values(elements)) {
      assert.equal(link.href, 'http://synthetic.test/');
      assert.equal(link.textContent, `Back to Yap Studio · ${label}`);
    }
  });
}
