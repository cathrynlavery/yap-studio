// Run the shipped controller with a small DOM and fake engine transport: no browser or sockets.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import * as choices from '../../src/yap_studio/gallery/static/choices.mjs';

const root = new URL('../../', import.meta.url);
const html = readFileSync(new URL('src/yap_studio/gallery/static/pick.html', root), 'utf8');
const source = readFileSync(new URL('src/yap_studio/gallery/static/pick.js', root), 'utf8')
  .replace(/^import \{[\s\S]*?\} from "\.\/choices\.mjs";\s*/, '');
const bundledFonts = ['Instrument Serif', 'Gelasio', 'Inter', 'Geist', 'Alata', 'Archivo Black',
  'Anton SC', 'Titan One', 'Permanent Marker', 'Space Mono'];
const schema = JSON.parse(readFileSync(new URL('src/yap_studio/style/schema-v1.json', root)));
function defaults(rule) {
  if ('default' in rule && rule.type !== 'object') return structuredClone(rule.default);
  if ('const' in rule) return rule.const;
  if (rule.type === 'object') return Object.fromEntries(Object.entries(rule.properties)
    .map(([key, child]) => [key, defaults(child)]).filter(([, value]) => value !== undefined));
}
function initialState() {
  const optIn = new Set(['edit_timeline', 'storyboard', 'jiggle_delete', 'file_drop', 'video_card',
    'prompt_bar', 'comment_cta', 'comment_drop', 'icon_strike', 'big_number', 'gravity_drop', 'choices']);
  const style = { ...defaults(schema), schema_version: 1, name: 'Editorial' };
  style.hook.answered = true;
  style.hook.font = style.captions.font;
  return { schema, style, bleep_builtin: JSON.parse(readFileSync(new URL('src/yap_studio/sfx/swear-words.json', root))).words,
    from_library: false, library: [
    { id: 'learned', name: 'SparrowLight', style: { ...structuredClone(style), name: 'SparrowLight' } },
  ], presets: [{ id: 'base', name: 'Editorial', style: structuredClone(style) },
    { id: 'clean_minimal', name: 'Clean Minimal', style: { ...structuredClone(style), name: 'Clean Minimal',
      captions: { ...style.captions, font: 'Inter', highlight_style: 'none', size_px: 76 },
      hook: { ...style.hook, font: 'Gelasio' } } }],
  fonts: { bundled: [...bundledFonts], creator: [] },
  card_kinds: Object.fromEntries(Object.entries({
    Words: ['text_tag', 'checklist', 'steps', 'flow', 'terminal', 'quote', 'specs_list',
      'name_tag', 'success_check', 'gravity_drop', 'choices'],
    Numbers: ['stat', 'chart', 'progress_stat', 'big_number'],
    Media: ['image_card', 'image_compare', 'transfer', 'callout_circle', 'icon_strike'],
    Chat: ['prompt_bar', 'comment_cta', 'comment_drop'],
    Editor: ['edit_timeline', 'storyboard', 'jiggle_delete', 'file_drop'],
    Video: ['video_card'],
  }).flatMap(([category, kinds]) => kinds.map((kind) => [kind, { category, opt_in: optIn.has(kind), off: optIn.has(kind) }]))),
  preview: { still: '/still?g=0', generation: 0, source: 'take', sentence: 'Your real first sentence.',
    grade_color: { color: '#1E6BD6', chroma: 0.18, colorful: true }, grade_default_strength: 0.5 } };
}
function markerState() {
  const state = initialState();
  const preset = JSON.parse(readFileSync(new URL('src/yap_studio/style/presets/marker.json', root)));
  for (const [key, value] of Object.entries(preset)) {
    state.style[key] = typeof value === 'object' ? { ...state.style[key], ...value } : value;
  }
  state.presets.push({ id: 'marker', name: 'Marker', style: structuredClone(state.style) });
  state.fonts.bundled.push('Permanent Marker');
  return state;
}
class Element {
  constructor(tag, document) {
    this.tagName = tag;
    this.ownerDocument = document;
    this.children = [];
    this.dataset = {};
    this.attrs = {};
    this.events = {};
    this.style = { setProperty(name, value) { this[name] = value; } };
    this.hidden = false;
    this.open = false;
    this.value = '';
    this.className = '';
    this.clientWidth = 200;
    this.classList = {
      contains: (name) => this.className.split(' ').includes(name),
      toggle: (name, force) => {
        const present = this.classList.contains(name);
        const next = force ?? !present;
        this.className = next ? present ? this.className : `${this.className} ${name}`
          : this.className.split(' ').filter((v) => v !== name).join(' ');
        return next;
      },
    };
  }
  matches(selector) { return selector === ':hover' && Boolean(this.hovered); }
  getBoundingClientRect() { return { height: this.measuredHeight ?? (this.id === 'savebar' ? 140 : 0) }; }
  get isConnected() { return this === this.ownerDocument.root || Boolean(this.parent?.isConnected); }
  get parentElement() { return this.parent; }
  get textContent() { return this.text + this.children.map((child) => child.textContent).join(''); }
  set textContent(value) { this.text = String(value); this.replaceChildren(); }
  append(...nodes) {
    for (let node of nodes) {
      if (typeof node !== 'object') {
        const text = new Element('#text', this.ownerDocument);
        text.text = String(node);
        node = text;
      }
      node.parent = this;
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) { for (const child of this.children) child.parent = null; this.children = []; this.append(...nodes); }
  remove() { this.parent.children = this.parent.children.filter((child) => child !== this); this.parent = null; }
  setAttribute(key, value) {
    this.attrs[key] = String(value);
    if (key === 'id') this.id = value;
    if (key === 'class') this.className = value;
    if (key === 'hidden') this.hidden = true;
    if (key === 'value') this.value = value;
    if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = value;
  }
  getAttribute(key) { return this.attrs[key] ?? null; }
  removeAttribute(key) { delete this.attrs[key]; }
  addEventListener(name, handler) { (this.events[name] ??= []).push(handler); }
  async fire(name, data = {}) { for (const handler of this.events[name] ?? []) await handler({ target: this, ...data }); }
  click() { return this.fire('click'); }
  showModal() { this.open = true; }
  close() { this.open = false; }
  select() { this.selected = true; }
  focus() { this.ownerDocument.activeElement = this; }
  scrollIntoView(options) { this.scrolled = options; }
  querySelectorAll(selector) {
    const matches = (node) => selector.startsWith('.') ? node.classList.contains(selector.slice(1))
      : selector === '[id]' ? Boolean(node.id) : selector.startsWith('[') ? node.attrs[selector.slice(1, -1)] !== undefined : node.tagName === selector;
    const found = [];
    for (const child of this.children) {
      if (matches(child)) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }
}
Element.prototype.text = '';
function makeDocument() {
  const document = { createElement: (tag) => new Element(tag, document) };
  document.root = document.createElement('root');
  const stack = [document.root];
  const voidTags = new Set(['meta', 'link', 'input', 'img']);
  for (const match of html.matchAll(/<\/?([\w-]+)([^>]*)>|([^<]+)/g)) {
    if (!match[1]) { stack.at(-1).append(match[3]); continue; }
    if (match[0].startsWith('</')) { stack.pop(); continue; }
    const element = document.createElement(match[1]);
    for (const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) element.setAttribute(attr[1], attr[2] ?? '');
    stack.at(-1).append(element);
    if (!voidTags.has(match[1]) && !match[0].endsWith('/>')) stack.push(element);
  }
  document.querySelectorAll = (selector) => document.root.querySelectorAll(selector);
  document.documentElement = document.querySelectorAll('html')[0];
  document.body = document.querySelectorAll('body')[0];
  document.getElementById = (id) => document.querySelectorAll('[id]').find((element) => element.id === id);
  return document;
}
const flush = async () => { for (let i = 0; i < 4; i++) await new Promise(setImmediate); };
async function harness(state = initialState(), hash = '', options = {}) {
  const document = makeDocument(), observers = [], requests = [];
  const location = { hash, search: '?t=token', origin: 'http://gallery' };
  const windowEvents = {}, resizeObservers = [];
  const createElement = document.createElement;
  document.createElement = (tag) => {
    const element = createElement(tag);
    if (tag === 'img') {
      element.naturalWidth = 100; element.naturalHeight = 200;
      element.decode = async () => {};
    }
    if (tag === 'canvas') element.getContext = () => ({
      measureText(text) { return { width: text.length * parseFloat(this.font.match(/([\d.]+)px/)[1]) * .7 }; },
      drawImage(image, ...patch) { requests.push({ image: image.src, patch }); },
      getImageData() { return { data: new Uint8ClampedArray([30, 107, 214, 255]) }; },
    });
    if (tag === 'iframe') {
      const events = [];
      let time = 0, playing = false;
      element.contentWindow = {
        document: { fonts: { ready: Promise.resolve() } }, __playerReady: true,
        __player: {
          events,
          seek(value) { time = value; events.push(['seek', value]); return Promise.resolve(); },
          pause() { playing = false; events.push(['pause']); return Promise.resolve(); },
          play() { playing = true; events.push(['play']); return Promise.resolve(); },
          getTime() { return time; }, getDuration() { return 4; }, isPlaying() { return playing; },
        },
      };
    }
    return element;
  };
  const storage = options.storage ?? new Map();
  const context = vm.createContext({ ...choices, styleFontMode: choices.fontMode, document, location, structuredClone,
    sessionStorage: { getItem: (key) => { if (options.storageError) throw new Error('Storage unavailable'); return storage.get(key); }, setItem: (key, value) => { if (options.storageError) throw new Error('Storage unavailable'); storage.set(key, value); } },
    URLSearchParams, AbortController, console, setTimeout, clearTimeout, setInterval() {},
    requestAnimationFrame(callback) { if (callback.name !== 'animate') setImmediate(callback); },
    matchMedia: (query) => ({ matches: query === '(prefers-reduced-motion: reduce)' ? Boolean(options.reducedMotion)
      : ['(hover: none)', '(max-width: 899px)'].includes(query) ? Boolean(options.phone) : !options.phone }),
    history: { replaceState(_state, _title, value) { location.hash = value; } },
    window: { addEventListener(name, fn) { windowEvents[name] = fn; }, scrollTo() {} },
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; resizeObservers.push(this); }
      observe() { this.connected = true; } disconnect() { this.connected = false; }
    },
    IntersectionObserver: class {
      constructor(callback) { this.callback = callback; this.hosts = new Set(); observers.push(this); }
      observe(host) { this.hosts.add(host); }
      unobserve(host) { this.hosts.delete(host); }
      disconnect() { this.hosts.clear(); }
      intersect(ratio = 1) { this.callback([...this.hosts].map((target) => ({ target, isIntersecting: ratio > 0, intersectionRatio: ratio }))); }
    },
    fetch: async (url, requestOptions = {}) => {
      const data = requestOptions.body ? JSON.parse(requestOptions.body) : null;
      requests.push({ url, data, options: requestOptions });
      if (url.startsWith('/layer/')) return { status: 200 };
      let body;
      if (url.startsWith('/api/state')) body = state;
      else if (url.startsWith('/api/preview')) body = options.previewResponse
        ? await options.previewResponse() : options.preview ?? state.preview;
      else if (url.startsWith('/api/grade-color')) {
        const color = new URLSearchParams(url.split('?')[1]).get('color');
        body = { color, colorful: color !== '#808080', chroma: color === '#808080' ? 0 : 0.18,
          message: 'This colour is too grey. Pick a more colourful point or colour.' };
        if (options.gradeColor) body = await options.gradeColor(body);
      }
      else if (url === '/api/layers') {
        if (options.failTiles && data.tile.tab !== 'preview') throw new Error('Engine unavailable');
        let names = [];
        if (options.layers) {
          const { tab, step, section } = data.tile;
          names = tab === 'preview' ? ['captions', ...(['colors', 'more', 'cards'].includes(step) || (step === 'font' && section === 'cards') ? ['card'] : data.style.hook.style === 'none' ? [] : ['title'])]
            : [tab === 'cards' ? 'card' : tab === 'title' ? 'title' : tab === 'chapter' ? 'chapter' : 'captions'];
          if (data.style.captions.camera_3d) names = names.filter((name) => name !== 'captions');
        }
        body = { frame: { width: 1080, height: 1920 }, layers: names.map((name, index) => ({
          name, template: name === 'card' ? data.tile.card ?? 'text_tag' : name, src: `/layer/fake/${index}/`,
          x: 0, y: 0, width: 1080, height: 300,
          ...options.layerGeometry?.[name],
          duration_s: name === 'captions' ? options.captionDuration ?? 4 : 4, start_s: 0,
          poster_s: name === 'captions' ? options.captionPoster ?? .65 : 1.5,
        })) };
      }
      else if (url === '/api/save') {
        if (options.nameConflict && !data.replace) {
          return { ok: false, status: 409, json: async () => ({ code: 'style_name_taken', id: 'stale',
            error: 'You already have a style called “Sunday”. Update it, or choose another name.' }) };
        }
        if (options.failSave) {
          options.failSave = false;
          return { ok: false, json: async () => ({ error: 'Could not save. Try again.' }) };
        }
        state.style = { ...structuredClone(data.style), name: data.name };
        state.from_library = true;
        const entry = { id: data.replace || 'saved', name: data.name, style: structuredClone(state.style) };
        const index = state.library.findIndex((item) => item.id === entry.id);
        if (index >= 0) state.library[index] = entry; else state.library.unshift(entry);
        body = { name: data.name, id: entry.id };
      } else if (url.startsWith('/api/site-colors')) body = {
        site: 'example.com', suggestion: { caption: '#FFFFFF', accent: null, paper: '#FFFFFF', ink: '#111111' },
        colors: [{ hex: '#111111', name: '' }],
      };
      else throw new Error(`Unexpected request: ${url}`);
      return { ok: true, json: async () => structuredClone(body) };
    },
  });
  vm.runInContext(source, context, { filename: 'pick.js' });
  await flush();
  return { document, location, observers, resizeObservers, requests, state, windowEvents, storage,
    window: context.window,
    $: (id) => document.getElementById(id),
    async click(id) { await document.getElementById(id).fire('click'); await flush(); },
    async go(screen) {
      if (['font', 'captions', 'colors', 'top-text', 'more'].includes(screen)) await this.click(`step-${screen}`);
      else if (screen === 'looks') await this.click('back-to-looks');
      else { await this.click('step-more'); await this.click(`choice-${screen}`); }
    },
    async poll() { await context.pollPreview(); await flush(); },
    library() { return vm.runInContext('structuredClone(state.library)', context); },
    baseKey() { return vm.runInContext('baseKey', context); },
    preview() { return vm.runInContext('structuredClone(preview)', context); },
    async save() {
      await this.click('save');
      if (this.$('save-sheet').open) await this.click('confirm-save');
      if (this.$('save-sheet').open && this.$('confirm-save').textContent.startsWith('Update ')) await this.click('confirm-save');
    },
    async pick(id, label) {
      if (id === 'background-tiles') { id = 'caption-look-tiles'; if (label === 'None') label = 'Plain'; }
      const screen = { 'grade-tiles': 'grade', 'title-tiles': 'opening', 'card-style-tiles': 'cards' }[id];
      if (screen && document.body.dataset.screen !== screen) await this.go(screen);
      for (let parent = this.$(id).parentElement; parent; parent = parent.parentElement) {
        if (parent.tagName === 'details' && !parent.open) {
          parent.open = true; await parent.fire('toggle');
        }
      }
      const button = document.getElementById(id).children.find((node) => node.textContent.includes(label));
      assert.ok(button, `Missing ${label} in ${id}`);
      await button.fire('click'); await flush();
    },
  };
}

test('unanswered top text has no selected answer or main title and Save asks first', async () => {
  for (const phone of [false, true]) {
    const state = initialState(); delete state.style.hook.answered;
    const page = await harness(state, '#looks', {phone, layers: true});
    assert.equal(page.requests.filter(request => request.data?.tile.tab === 'preview')
      .every(request => request.data.style.hook.style === 'none'), true);
    await page.click('save');
    assert.equal(page.location.hash, '#top-text');
    assert.equal(page.$('top-text-tiles').children.length, 3);
    assert.equal(page.$('top-text-tiles').children.some(tile => tile.getAttribute('aria-pressed') === 'true'), false);
    assert.equal(page.$('save-sheet').open, false);
    await page.go('opening');
    await page.pick('title-tiles', 'Shimmer');
    assert.equal(page.requests.filter(request => request.data?.tile.tab === 'preview')
      .at(-1).data.style.hook.style, 'none');
    await page.click('save');
    assert.equal(page.location.hash, '#top-text');
    await page.click('top-text-whole');
    await page.click('top-text-none');
    await page.click('top-text-opening');
    await page.save();
    assert.equal(savePayload(page).hook.answered, true);
    assert.equal(savePayload(page).hook.style, 'shimmer');
  }
});

test('grade tiles, strength, candidate stills and Reset follow the chosen grade', async () => {
  const state = initialState();
  delete state.style.grade;
  const page = await harness(state, '#grade');
  const selected = () => page.$('grade-tiles').children
    .filter((tile) => tile.getAttribute('aria-pressed') === 'true')
    .map((tile) => tile.children.at(-1).text);
  assert.deepEqual(selected(), ['None']);
  assert.equal(page.$('grade-strength').hidden, true);
  assert.equal(page.$('grade-strength-label').textContent, 'Strength');
  assert.equal(page.$('grade-strength-label').hidden, true);
  assert.match(page.$('grade-tiles').children[0].querySelectorAll('img')[0].src, /look=none&strength=1/);
  assert.match(page.$('grade-tiles').children[1].querySelectorAll('img')[0].src, /look=film&strength=1/);
  page.observers[1].intersect();
  await flush();
  for (const [index, look] of ['none', 'film'].entries()) {
    const images = page.$('grade-tiles').children[index].querySelectorAll('img');
    assert.equal(images.length, 2);
    for (const image of images) assert.match(image.src, new RegExp(`look=${look}&strength=1`));
  }
  await page.pick('grade-tiles', 'Film look');
  assert.deepEqual(selected(), ['Film look']);
  assert.equal(page.$('grade-strength').hidden, false);
  assert.equal(page.$('grade-strength-label').hidden, false);
  assert.equal(page.$('grade-strength').children[1].getAttribute('aria-pressed'), 'true');
  assert.match(page.$('change-count').textContent, /Changed Grade/);
  assert.match(page.$('main-frame').querySelectorAll('img')[0].src, /look=film&strength=1/);
  const light = page.$('grade-strength').children[0];
  await page.$('grade-strength').fire('click', { target: light });
  await flush();
  assert.match(page.$('main-frame').querySelectorAll('img')[0].src, /look=film&strength=0.5/);
  const last = page.requests.filter((request) => request.url === '/api/layers').at(-1);
  assert.deepEqual(JSON.parse(JSON.stringify(last.data.style.grade)), { look: 'film', strength: 0.5 });
  await page.pick('grade-tiles', 'None');
  assert.equal(page.$('grade-strength').hidden, true);
  assert.equal(page.$('grade-strength-label').hidden, true);
  await page.click('reset');
  assert.deepEqual(selected(), ['None']);
  assert.equal(page.$('grade-strength').hidden, true);
  assert.doesNotMatch(page.$('main-frame').querySelectorAll('img')[0].src, /look=film/);
});

test('before/after control is optional, saves its choice and updates still previews', async () => {
  const page = await harness(initialState(), '#more');
  assert.equal(page.$('grade-slider').hidden, true);
  await page.pick('grade-tiles', 'Film look');
  assert.equal(page.$('grade-slider').hidden, false);
  assert.equal(page.$('grade-slider-label').textContent, 'Before and after');
  const on = page.$('grade-slider').children[1];
  await page.$('grade-slider').fire('click', { target: on }); await flush();
  assert.equal(on.getAttribute('aria-pressed'), 'true');
  const request = page.requests.filter((item) => item.url === '/api/layers').at(-1);
  assert.equal(request.data.style.grade.slider, true);
  assert.match(page.$('main-frame').querySelectorAll('img')[0].src, /slider=1/);
  await page.pick('grade-tiles', 'None');
  assert.equal(page.$('grade-slider').hidden, true);
  await page.click('reset');
  await page.pick('grade-tiles', 'Film look');
  assert.equal(page.$('grade-slider').children[0].getAttribute('aria-pressed'), 'true');
});

test('Eyedropper picks the scene, validates hex and samples the ungraded still on tap', async () => {
  const page = await harness(initialState(), '#more');
  await page.pick('grade-tiles', 'Eyedropper');
  assert.equal(page.$('grade-color-fields').hidden, false);
  assert.equal(page.$('grade-color').value, '#1E6BD6');
  assert.equal(page.$('grade-picker').getAttribute('aria-label'), 'Pick a grade color');
  assert.doesNotMatch(html, /id="grade-swatch"/);
  assert.match(page.$('grade-hint').textContent, /Tap your video to pick a color/);
  let request = page.requests.filter((item) => item.url === '/api/layers').at(-1);
  assert.deepEqual(JSON.parse(JSON.stringify(request.data.style.grade)),
    { look: 'eyedropper', strength: 0.5, color: '#1E6BD6', slider: false });
  page.$('grade-color').value = '#oops';
  await page.$('grade-color').fire('change');
  assert.equal(page.$('grade-color').getAttribute('aria-invalid'), 'true');
  page.$('grade-color').value = '#808080';
  await page.$('grade-color').fire('change'); await flush();
  assert.match(page.$('grade-error').textContent, /too grey/);
  assert.equal(page.$('grade-picker').value, '#1E6BD6');
  page.$('grade-color').value = '#2FAE5A';
  await page.$('grade-color').fire('change'); await flush();
  assert.equal(page.$('grade-picker').value, '#2FAE5A');
  page.$('main-frame').getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 200 });
  await page.$('main-frame').fire('click', { clientX: 50, clientY: 100 }); await flush();
  assert.match(page.requests.find((item) => item.image)?.image, /look=none/);
  assert.equal(page.$('grade-color').value, '#1E6BD6');
  request = page.requests.filter((item) => item.url === '/api/layers').at(-1);
  assert.equal(request.data.style.grade.color, '#1E6BD6');
  await page.pick('grade-tiles', 'Film look');
  assert.equal(page.$('grade-color-fields').hidden, true);
});

test('a grey scene asks for a point on the same raster used for tap sampling', async () => {
  const state = initialState();
  state.preview.grade_color = { color: '#808080', colorful: false,
    message: 'This colour is too grey. Pick a more colourful point or colour.' };
  const page = await harness(state, '#more');
  await page.pick('grade-tiles', 'Eyedropper');
  assert.equal(page.$('grade-color-fields').hidden, false);
  assert.match(page.$('grade-error').textContent, /too grey/);
  assert.match(page.$('main-frame').querySelectorAll('img')[0].src, /look=none/);
  page.$('main-frame').getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 200 });
  await page.$('main-frame').fire('click', { clientX: 50, clientY: 100 }); await flush();
  assert.equal(page.$('grade-color').value, '#1E6BD6');
  assert.match(page.$('main-frame').querySelectorAll('img')[0].src, /look=eyedropper&strength=0.5/);
});

test('3D camera Eyedropper taps sample the same still point as flat captions', async () => {
  const samples = [];
  const bounds = { left: 10, top: 20, width: 180, height: 240 };
  const point = choices.coverPoint(bounds, 100, 200, 137, 91);
  const patch = choices.samplePatch(100, 200, point.x, point.y);
  for (const camera_3d of [false, true]) {
    const state = initialState(); state.style.captions.camera_3d = camera_3d;
    const page = await harness(state, '#grade');
    await page.pick('grade-tiles', 'Eyedropper');
    const main = page.$('main-frame');
    assert.equal(main.querySelectorAll('.camera3d-scene').length, 0);
    assert.equal(main.querySelectorAll('img').length, 1);
    main.getBoundingClientRect = () => bounds;
    await main.fire('click', { clientX: 137, clientY: 91, detail: 1 }); await flush();
    const sampled = page.requests.find((request) => request.image);
    assert.deepEqual(sampled.patch, [patch.x, patch.y, patch.width, patch.height,
      0, 0, patch.width, patch.height]);
    assert.equal(page.$('grade-color').value, '#1E6BD6');
    samples.push(sampled);
  }
  assert.deepEqual(samples[1], samples[0]);
});

for (const choice of ['Film look', 'None', 'selectStyle', 'startFrom', 'Reset']) {
  test(`leaving a grey-scene Eyedropper pick through ${choice} permits saving and ignores late colours`, async () => {
    const state = initialState();
    state.preview.grade_color = { color: '#808080', colorful: false, message: 'This colour is too grey.' };
    let finish;
    const page = await harness(state, '#more', {
      gradeColor: (body) => new Promise((resolve) => { finish = () => resolve(body); }),
    });
    await page.pick('grade-tiles', 'Eyedropper');
    page.$('grade-color').value = '#2FAE5A';
    const pending = page.$('grade-color').fire('change');
    await flush();
    assert.equal(typeof finish, 'function');
    if (choice === 'startFrom') {
      await page.click('back-to-looks');
      await page.pick('start-options', 'Clean Minimal');
    } else if (choice === 'selectStyle') {
      await page.click('step-font');
      await page.pick('font-options', 'Inter');
    } else if (choice === 'Reset') {
      assert.equal(page.$('reset').disabled, false);
      await page.click('reset');
    }
    else await page.pick('grade-tiles', choice);
    await page.save();
    assert.equal(page.$('save-error').textContent, '');
    assert.equal(page.$('saved').hidden, false);
    const saved = page.requests.find((request) => request.url === '/api/save').data.style;
    assert.equal(saved.grade.look, choice === 'Film look' ? 'film' : 'none');
    finish(); await pending; await flush();
    assert.equal(page.$('grade-error').textContent, '');
    const latest = page.requests.filter((request) => request.url === '/api/layers').at(-1);
    assert.equal(latest.data.style.grade.look, saved.grade.look);
  });
}

test('starting from another look clears an unresolved grey-scene pick before saving', async () => {
  const state = initialState();
  state.preview.grade_color = { color: '#808080', colorful: false, message: 'This colour is too grey.' };
  const page = await harness(state, '#more');
  await page.pick('grade-tiles', 'Eyedropper');
  await page.click('back-to-looks');
  await page.pick('start-options', 'Clean Minimal');
  await page.save();
  assert.equal(page.$('saved').hidden, false);
});

test('the shipped controller navigates, restores its hash and loads only current-step tiles', async () => {
  const page = await harness(initialState(), '#captions');
  assert.equal(page.$('preview-note').textContent, 'Your newest take · your first sentence');
  assert.equal(page.$('panel-captions').hidden, false);
  assert.equal(page.$('highlight-tiles').children.length, schema.properties.captions.properties.highlight_style.enum.length - 1);
  assert.equal(page.$('font-options').children.length, 0);
  page.observers[1].intersect();
  await flush();
  assert.ok(page.requests.filter((r) => r.url === '/api/layers' && r.data.tile.tab === 'captions').length > 0);
  await page.click('back');
  assert.equal(page.location.hash, '#font');
  assert.equal(page.$('highlight-tiles').children.length, 0);
  assert.equal(page.$('font-options').children.length, bundledFonts.length);
  assert.equal(page.$('savebar').hidden, false);
  assert.equal(page.$('fine-tune').open, false);
  await page.click('next');
  assert.equal(page.location.hash, '#captions');
  await page.click('step-colors');
  const panels = page.document.querySelectorAll('.step-panel');
  assert.deepEqual(panels.filter((node) => !node.hidden).map((node) => node.id), ['panel-colors']);
  page.location.hash = '#captions';
  page.windowEvents.hashchange();
  await flush();
  assert.equal(page.$('panel-captions').hidden, false);
});

test('setup no music survives looks and an explicit Sound choice wins', async () => {
  const state = initialState();
  state.music_answer = 'none';
  state.style.music.level = .09;
  const page = await harness(state, '#sound');
  const selected = () => page.$('music-level-tiles').children
    .filter((button) => button.getAttribute('aria-pressed') === 'true')
    .map((button) => button.querySelectorAll('.tile-label')[0].textContent);
  assert.deepEqual(selected(), ['No music']);
  assert.equal(page.$('music-value').textContent, '0% · Off');
  await page.go('looks');
  await page.pick('start-options', 'Editorial');
  await page.click('customize');
  await page.go('sound');
  assert.deepEqual(selected(), ['No music']);
  await page.pick('music-level-tiles', 'Medium');
  assert.equal(page.$('music-value').textContent, '12%');
  await page.go('looks');
  await page.pick('start-options', 'Clean Minimal');
  await page.click('customize');
  await page.go('sound');
  assert.ok(selected()[0].startsWith('Medium'));
  await page.save();
  assert.equal(page.requests.find((request) => request.url === '/api/save').data.style.music.level, .12);
});

test('a saved Sound level wins over setup no music across looks and saving', async () => {
  const state = initialState();
  state.music_answer = 'none';
  state.from_library = true;
  state.style.music.level = .12;
  const page = await harness(state, '#sound');
  const selected = () => page.$('music-level-tiles').children
    .filter((button) => button.getAttribute('aria-pressed') === 'true')
    .map((button) => button.querySelectorAll('.tile-label')[0].textContent);
  assert.ok(selected()[0].startsWith('Medium'));
  await page.go('looks');
  await page.pick('start-options', 'Clean Minimal');
  await page.click('customize');
  await page.go('sound');
  assert.ok(selected()[0].startsWith('Medium'));
  await page.save();
  assert.equal(page.requests.find((request) => request.url === '/api/save').data.style.music.level, .12);
});

test('a saved style without a Sound level keeps setup no music', async () => {
  const state = initialState();
  state.music_answer = 'none';
  state.from_library = true;
  delete state.style.music.level;
  const page = await harness(state, '#sound');
  assert.equal(page.$('music-value').textContent, '0% · Off');
  await page.go('looks');
  await page.pick('start-options', 'Clean Minimal');
  await page.click('customize');
  await page.go('sound');
  assert.equal(page.$('music-value').textContent, '0% · Off');
  await page.save();
  assert.equal(page.requests.find((request) => request.url === '/api/save').data.style.music.level, 0);
});

test('without a setup music answer a look keeps its own level', async () => {
  const page = await harness(initialState(), '#sound');
  await page.pick('music-level-tiles', 'Medium');
  await page.go('looks');
  await page.pick('start-options', 'Editorial');
  await page.click('customize');
  await page.go('sound');
  assert.equal(page.$('music-value').textContent, '6%');
});

test('the shipped save path round-trips selections, colors, custom width and card switches', async () => {
  const page = await harness();
  assert.ok(page.$('start-options').children[0].children[1].textContent.startsWith('SparrowLight'));
  await page.pick('start-options', 'SparrowLight');
  assert.equal(page.$('style-name').value, 'SparrowLight');
  await page.click('customize');
  await page.click('step-captions');
  await page.pick('word-tiles', 'A few words');
  await page.click('step-captions');
  await page.pick('highlight-tiles', 'Pill');
  await page.click('step-font');
  await page.pick('font-options', 'Instrument Serif');
  await page.pick('width-tiles', 'Condensed');
  await page.click('step-colors');
  page.$('color-accent').value = '#2BF';
  await page.$('color-accent').fire('change');
  await flush();
  await page.click('step-more');
  await page.go('top-text');
  await page.pick('top-text-tiles', 'None');
  page.$('music').value = .13;
  await page.$('music').fire('input');
  page.$('fine-width_scale').value = .79;
  await page.$('fine-width_scale').fire('input');
  await flush();
  await page.click('step-more');
  await page.pick('card-style-tiles', 'Bright');
  const numbers = page.$('card-categories').children[1];
  numbers.open = true;
  await numbers.fire('toggle');
  const toggle = numbers.querySelectorAll('input')[1];
  toggle.checked = false;
  await toggle.fire('change');
  assert.ok(numbers.textContent.includes('Sample values'));
  await page.save();
  assert.equal(page.$('saved').hidden, false);
  assert.equal(page.$('saved-name').textContent, 'SparrowLight');
  const saved = page.requests.find((r) => r.url === '/api/save').data.style;
  assert.equal(saved.captions.max_words_per_phrase, 3);
  assert.equal(saved.captions.highlight_style, 'pill');
  assert.equal(saved.captions.font, 'Instrument Serif');
  assert.equal(saved.captions.width_scale, .79);
  assert.equal(saved.captions.highlight_color, '#22BBFF');
  assert.equal(saved.hook.style, 'none');
  assert.equal(saved.music.level, .13);
  assert.equal(saved.graphics.style, 'bright');
  assert.deepEqual(saved.graphics.card_kinds_off, ['stat']);
  const reloaded = await harness(page.state, page.location.hash);
  assert.equal(reloaded.$('panel-cards').hidden, false);
  assert.equal(reloaded.$('style-name').value, 'SparrowLight');
  await reloaded.click('step-font');
  assert.ok(reloaded.$('width-note').textContent.includes('79%'));
  assert.equal(reloaded.$('fine-width_scale').value, .79);
  await reloaded.click('step-colors');
  assert.equal(reloaded.$('color-accent').value, '#22BBFF');
  await reloaded.click('step-more');
  assert.equal(reloaded.$('music').value, .13);
});

for (const phone of [false, true]) {
  test(`chat cards start off and save opt-ins on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState();
    state.style.graphics.card_kinds_off = ['stat'];
    state.card_kinds.stat.off = true;
    const originalGraphics = structuredClone(state.style.graphics);
    const page = await harness(state, '#cards', { phone });
    const categories = page.$('card-categories').children;
    assert.deepEqual(categories.map((group) => group.querySelectorAll('.group-state')[0].textContent), [
      '9 on', '2 on', '4 on', 'off', 'off', 'off',
    ]);
    const chat = categories[3];
    chat.open = true;
    await chat.fire('toggle');
    assert.equal(chat.querySelectorAll('.kind-row').length, 4);
    assert.deepEqual(chat.querySelectorAll('input').slice(1).map((toggle) => toggle.getAttribute('aria-label')),
      ['Prompt bar on', 'Comment CTA on', 'Comment drop on']);
    assert.ok(chat.querySelectorAll('input').slice(1).every((toggle) => !toggle.checked));
    const kinds = ['prompt_bar', 'comment_cta', 'comment_drop'];
    const enabled = new Set();
    async function change(index, on) {
      if (page.$('picker').hidden) await page.click('update');
      const group = page.$('card-categories').children[3];
      group.open = true;
      await group.fire('toggle');
      const toggle = group.querySelectorAll('input')[index + 1];
      assert.equal(toggle.checked, !on);
      toggle.checked = on;
      await toggle.fire('change');
      if (on) enabled.add(kinds[index]);
      else enabled.delete(kinds[index]);
      assert.equal(page.$('card-categories').children[3].querySelectorAll('.group-state')[0].textContent,
        enabled.size === 0 ? 'off' : enabled.size === 3 ? 'all on' : `${enabled.size} on`);
      await page.save();
      const saved = page.requests.filter((request) => request.url === '/api/save').at(-1).data.style;
      assert.deepEqual(saved.graphics,
        { ...originalGraphics, card_kinds_on: [...enabled].sort() });
    }
    for (const index of [0, 2, 1]) await change(index, true);
    for (const index of [2, 0, 1]) await change(index, false);
  });
}

for (const phone of [false, true]) {
  test(`icon strike starts off in Media on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState();
    state.style.graphics.card_kinds_off = ['stat'];
    state.card_kinds.stat.off = true;
    const originalGraphics = structuredClone(state.style.graphics);
    const page = await harness(state, '#cards', { phone });
    assert.equal(page.$('card-categories').children[2].querySelectorAll('.group-state')[0].textContent, '4 on');
    for (const on of [true, false]) {
      if (page.$('picker').hidden) await page.click('update');
      const media = page.$('card-categories').children[2];
      media.open = true;
      await media.fire('toggle');
      const toggles = media.querySelectorAll('input');
      assert.equal(toggles.length, 6);
      const toggle = toggles[5];
      assert.equal(toggle.getAttribute('aria-label'), 'Icon strike on');
      assert.equal(toggle.checked, !on);
      toggle.checked = on;
      await toggle.fire('change');
      assert.equal(page.$('card-categories').children[2].querySelectorAll('.group-state')[0].textContent,
        on ? 'all on' : '4 on');
      await page.save();
      const saved = page.requests.filter((request) => request.url === '/api/save').at(-1).data.style;
      assert.deepEqual(saved.graphics,
        { ...originalGraphics, card_kinds_on: on ? ['icon_strike'] : [] });
    }
  });
}

for (const phone of [false, true]) {
  test(`registry cards use state labels and save only opt-ins on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState();
    state.card_kinds['hf:sample'] = { category: 'Words', label: 'Sample headline', registry: true,
      opt_in: true, off: true };
    const original = structuredClone(state.style);
    const page = await harness(state, '#cards', { phone, layers: true });
    await openFold(page, 'card-group-words');
    const words = page.$('card-group-words'), toggle = page.$('kind-hf:sample');
    assert.ok(toggle);
    const row = toggle.parentElement;
    assert.equal(row.parentElement, words);
    assert.equal(row.children[1].textContent, 'Sample headline');
    assert.equal(row.children[1].htmlFor, toggle.id);
    assert.equal(row.children[0].getAttribute('aria-label'), 'Preview Sample headline');
    assert.equal(toggle.getAttribute('aria-label'), 'Sample headline on');
    assert.equal(toggle.checked, false);
    assert.equal(words.querySelectorAll('.new-tag').length, 1);
    page.observers[1].intersect(); await flush();
    assert.ok(page.requests.some((r) => r.url === '/api/layers' && r.data.tile.card === 'hf:sample'));
    await row.children[0].fire('click'); await flush();
    assert.equal(page.requests.filter((r) => r.url === '/api/layers').at(-1).data.tile.card, 'hf:sample');
    toggle.checked = true; await toggle.fire('change'); await flush();
    await page.save();
    assert.deepEqual(savePayload(page), { ...original,
      graphics: { ...original.graphics, card_kinds_on: ['hf:sample'] } });
  });

  test(`Choices starts off in Words on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState();
    const page = await harness(state, '#cards', { phone });
    const words = page.$('card-categories').children[0];
    words.open = true;
    await words.fire('toggle');
    const toggle = words.querySelectorAll('input').at(-1);
    assert.equal(toggle.getAttribute('aria-label'), 'Choices on');
    assert.equal(toggle.checked, false);
    toggle.checked = true;
    await toggle.fire('change');
    await page.save();
    const saved = page.requests.filter((request) => request.url === '/api/save').at(-1).data.style;
    assert.deepEqual(saved.graphics.card_kinds_on, ['choices']);
  });
}

test('opt-in card switches start off and save only card_kinds_on', async () => {
  const state = initialState();
  state.card_kinds.stat = { category: 'Numbers', opt_in: true, off: true };
  delete state.style.graphics.card_kinds_on;
  const page = await harness(state, '#cards');
  const numbers = page.$('card-categories').children[1];
  assert.ok(numbers.querySelectorAll('.group-state')[0].textContent === '2 on');
  numbers.open = true;
  await numbers.fire('toggle');
  const toggle = numbers.querySelectorAll('input')[1];
  assert.equal(toggle.checked, false);
  toggle.checked = true;
  await toggle.fire('change');
  assert.equal(page.$('card-categories').children[1].querySelectorAll('.group-state')[0].textContent, '3 on');
  assert.equal(page.$('kind-big_number').checked, false);
  await page.save();
  const saved = page.requests.find((r) => r.url === '/api/save').data.style;
  assert.deepEqual(saved.graphics.card_kinds_on, ['stat']);
  assert.deepEqual(saved.graphics.card_kinds_off, []);
});

test('website colors carry the gallery token and keep an accent for monochrome sites', async () => {
  const page = await harness(initialState(), '#colors');
  assert.equal(page.$('site-fold-value').textContent, '');
  const accent = page.$('color-accent').value;
  page.$('site-url').value = 'example.com';
  await page.click('site-pull');
  assert.equal(page.$('site-fold-value').textContent, 'example.com');
  assert.equal(page.$('color-accent').value, accent);
  assert.ok(page.$('site-status').textContent.includes('kept your highlight color'));
  const request = page.requests.find((r) => r.url.startsWith('/api/site-colors'));
  assert.ok(request.url.includes('url=example.com&t=token'));
  assert.equal(page.$('found-colors').children.length, 1);
});

test('#oops then Newsprint then Save replaces the error with a valid saved highlight', async () => {
  const page = await harness(initialState(), '#colors');
  const input = page.$('color-accent');
  input.focus();
  input.value = '#oops';
  await input.fire('change');
  assert.equal(input.getAttribute('aria-invalid'), 'true');
  await page.save();
  assert.equal(page.requests.some((request) => request.url === '/api/save'), false);
  page.$('palette-tiles').children.find((button) => button.textContent.includes('Newsprint')).focus();
  await page.pick('palette-tiles', 'Newsprint');
  assert.equal(input.value, '#E9C46A');
  assert.equal(input.getAttribute('aria-invalid'), null);
  assert.equal(page.$('error-accent').textContent, '');
  await page.save();
  assert.equal(page.$('saved').hidden, false);
  assert.equal(page.$('save-error').textContent, '');
  assert.equal(page.requests.find((request) => request.url === '/api/save').data.style.captions.highlight_color, '#E9C46A');
});

test('every valid replacement clears all color errors, even a focused unfinished edit', async (t) => {
  const sources = {
    palette: (page) => page.pick('palette-tiles', 'Newsprint'),
    website: async (page) => { page.$('site-url').value = 'example.com'; await page.click('site-pull'); },
    preset: async (page) => { await page.click('back-to-looks'); await page.pick('start-options', 'Editorial'); },
    library: async (page) => { await page.click('back-to-looks'); await page.pick('start-options', 'SparrowLight'); },
    reset: (page) => page.click('reset'),
    picker: async (page) => { for (const [role] of choices.roles) { page.$(`picker-${role}`).value = '#ABCDEF'; await page.$(`picker-${role}`).fire('input'); } await flush(); },
    card: async (page) => { await page.click('step-more'); await page.pick('card-style-tiles', 'Dark'); },
  };
  for (const [name, replace] of Object.entries(sources)) await t.test(name, async () => {
    const page = await harness(initialState(), '#colors');
    for (const [role] of choices.roles) {
      page.$(`color-${role}`).value = '#oops';
      await page.$(`color-${role}`).fire('change');
      assert.equal(page.$(`color-${role}`).getAttribute('aria-invalid'), 'true');
    }
    // Full replacements must override focus as well as stale markers. The native picker
    // focuses itself, so its prior text edit is no longer the active unfinished input.
    page.$(name === 'picker' ? 'picker-accent' : 'color-accent').focus();
    assert.equal(page.$('reset').disabled, false);
    await replace(page);
    for (const [role] of choices.roles) {
      assert.ok(choices.normalizeHex(page.$(`color-${role}`).value));
      assert.equal(page.$(`color-${role}`).value, page.$(`picker-${role}`).value);
      assert.equal(page.$(`color-${role}`).getAttribute('aria-invalid'), null);
      assert.equal(page.$(`error-${role}`).textContent, '');
    }
    await page.save();
    assert.equal(page.$('saved').hidden, false);
    assert.equal(page.requests.filter((request) => request.url === '/api/save').length, 1);
  });
});

test('kind counts use singular for one in every card group and the highlight fold', async () => {
  for (const single of [false, true]) {
    const state = initialState();
    if (single) {
      const categories = new Set();
      state.card_kinds = Object.fromEntries(Object.entries(state.card_kinds).filter(([, info]) => {
        if (categories.has(info.category)) return false;
        categories.add(info.category); return true;
      }));
    }
    const page = await harness(state, '#cards');
    for (const group of page.$('card-categories').children) {
      const category = group.id.replace('card-group-', '');
      const count = Object.values(state.card_kinds).filter((info) => info.category.toLowerCase() === category).length;
      const title = category[0].toUpperCase() + category.slice(1) + (['chat', 'editor', 'video'].includes(category) ? ' cards' : '');
      assert.equal(group.children[0].children[0].textContent, `${title} · ${count} ${count === 1 ? 'kind' : 'kinds'}`);
    }
  }
  for (const single of [false, true]) {
    const state = initialState(); state.schema = structuredClone(state.schema);
    if (single) state.schema.properties.captions.properties.highlight_style.enum = [state.style.captions.highlight_style];
    const page = await harness(state, '#captions');
    assert.ok(page.$('highlight-fold-value').textContent.endsWith(single ? ' · 1 kind' : ' · 24 kinds'));
  }
});

test('Numbers lists four kinds with Big number behind you off and marked new', async () => {
  for (const phone of [false, true]) {
    const state = initialState(), original = structuredClone(state.style), page = await harness(state, '#cards', { phone });
    const numbers = page.$('card-group-numbers');
    assert.ok(numbers.textContent.includes('Numbers · 4 kinds'));
    assert.equal(numbers.querySelectorAll('.new-tag').length, 1);
    assert.equal(numbers.querySelectorAll('.group-state')[0].textContent, '3 on');
    await openFold(page, numbers.id);
    assert.deepEqual(numbers.querySelectorAll('input').slice(1).map((input) => input.id),
      ['kind-stat', 'kind-chart', 'kind-progress_stat', 'kind-big_number']);
    assert.deepEqual(numbers.querySelectorAll('input').slice(1).map((input) => input.checked), [true, true, true, false]);
    const row = page.$('kind-big_number').parentElement;
    assert.equal(row.children[0].getAttribute('aria-label'), 'Preview Big number behind you');
    assert.equal(row.children[0].classList.contains('compact'), true);
    assert.equal(row.querySelectorAll('.kind-note')[0].textContent, 'Counts up behind your head');
    await page.save();
    assert.deepEqual(savePayload(page), original);
  }
});

test('Big number opt-in writes only the sorted card_kinds_on array', async () => {
  for (const on of [true, false]) {
    const state = initialState();
    state.style.graphics.card_kinds_on = on ? ['prompt_bar'] : ['big_number', 'prompt_bar'];
    state.style.graphics.card_kinds_off = ['stat'];
    const original = structuredClone(state.style), page = await harness(state, '#cards');
    await openFold(page, 'card-group-numbers');
    const toggle = page.$('kind-big_number');
    assert.equal(toggle.checked, !on);
    toggle.checked = on; await toggle.fire('change'); await flush();
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'graphics.card_kinds_on': on ? ['big_number', 'prompt_bar'] : ['prompt_bar'] }));
  }
});

test('only Big number live thumbnails and tapped main previews layer the person over the card', async () => {
  const state = initialState(); state.preview.cutout = '/cutout?g=0';
  const original = structuredClone(state.style), page = await harness(state, '#cards', { layers: true });
  await openFold(page, 'card-group-numbers'); page.observers[1].intersect(); await flush();
  const host = page.$('kind-big_number').parentElement.children[0];
  const overlay = host.querySelectorAll('.card-cutout')[0];
  assert.equal(overlay.src, '/cutout?g=0&t=token');
  assert.equal(host._galleryStage.stage.children.at(-1), overlay);
  assert.equal(host._galleryStage.frames.length, 1);
  for (const kind of ['stat', 'chart', 'progress_stat']) {
    assert.equal(page.$(`kind-${kind}`).parentElement.querySelectorAll('.card-cutout').length, 0);
  }
  assert.equal(page.$('card-style-tiles').querySelectorAll('.card-cutout').length, 0);
  assert.equal(page.$('main-frame').querySelectorAll('.card-cutout').length, 0);
  await host.click(); await flush();
  assert.equal(page.$('main-frame').querySelectorAll('.card-cutout')[0].src, overlay.src);
  const numberMain = page.$('main-frame')._galleryStage;
  assert.equal(numberMain.stage.children.at(-1), numberMain.cutout);
  await page.$('kind-stat').parentElement.children[0].click(); await flush();
  assert.equal(numberMain.active, false);
  assert.equal(page.$('main-frame').querySelectorAll('.card-cutout').length, 0);
  await page.save();
  assert.deepEqual(savePayload(page), original);
});

test('late cutouts update Big number overlays without remounting live frames or changing the style', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const options = { layers: true, preview: { ...state.preview, cutout: '/cutout?g=0' } };
  const page = await harness(state, '#cards', options);
  await openFold(page, 'card-group-numbers'); page.observers[1].intersect(); await flush();
  const host = page.$('kind-big_number').parentElement.children[0];
  await host.click(); await flush();
  const stages = [host._galleryStage, page.$('main-frame')._galleryStage];
  const frames = stages.map((stage) => [...stage.frames]);
  for (const frame of frames.flat()) await frame.fire('load');
  await flush();
  assert.equal(host.querySelectorAll('.card-cutout').length, 0);
  const count = page.requests.filter((request) => request.url === '/api/layers').length;
  await page.poll();
  const overlays = stages.map((stage) => stage.cutout);
  for (const [index, stage] of stages.entries()) {
    assert.equal(stage.cutout.src, '/cutout?g=0&t=token');
    assert.equal(stage.frames.length, frames[index].length);
    stage.frames.forEach((frame, frameIndex) => assert.equal(frame, frames[index][frameIndex]));
    assert.ok(stage.frames.every((frame) => frame._galleryReady));
  }
  options.preview = { ...options.preview, cutout: '/cutout?g=0&new=1' }; await page.poll();
  for (const [index, stage] of stages.entries()) {
    assert.equal(stage.cutout, overlays[index]);
    assert.equal(stage.cutout.src, '/cutout?g=0&new=1&t=token');
  }
  await overlays[0].fire('error'); assert.equal(overlays[0].hidden, true);
  await overlays[0].fire('load'); assert.equal(overlays[0].hidden, false);
  options.preview = { ...options.preview, cutout: null }; await page.poll();
  assert.ok(stages.every((stage) => stage.cutout === null && stage.active));
  assert.ok(overlays.every((overlay) => !overlay.isConnected));
  options.preview = { ...options.preview, cutout: '/cutout?g=0&new=2' }; await page.poll();
  const detached = stages[0].cutout;
  const numbers = page.$('card-group-numbers'); numbers.open = false; await numbers.fire('toggle'); await flush();
  options.preview = { ...options.preview, cutout: '/cutout?g=0&new=3' }; await page.poll();
  assert.equal(detached.src, '/cutout?g=0&new=2&t=token');
  assert.equal(stages[1].cutout.src, '/cutout?g=0&new=3&t=token');
  assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, count);
  await page.save();
  assert.deepEqual(savePayload(page), original);
});

test('a visible focused invalid edit survives unrelated redraws and blocks Save until replaced', async () => {
  const page = await harness(initialState(), '#colors');
  const input = page.$('color-accent');
  input.focus();
  input.value = '#oops';
  await input.fire('change');
  page.$('music').value = .2;
  await page.$('music').fire('input');
  assert.equal(input.value, '#oops');
  assert.equal(input.getAttribute('aria-invalid'), 'true');
  assert.ok(page.$('error-accent').textContent);
  // Typing clears the blur error, but the still-invalid visible text must block saving.
  input.value = '#oo';
  await input.fire('input');
  assert.equal(input.getAttribute('aria-invalid'), null);
  await page.save();
  assert.equal(page.requests.some((request) => request.url === '/api/save'), false);
  assert.equal(page.location.hash, '#colors');
  await page.pick('palette-tiles', 'Newsprint');
  await page.save();
  assert.equal(page.$('saved').hidden, false);
});

test('an unfocused invalid field stays visible for validation until Reset replaces it', async () => {
  const page = await harness(initialState(), '#colors');
  page.$('color-accent').value = '#oops'; await page.$('color-accent').fire('change');
  page.$('music').focus(); page.$('music').value = .2; await page.$('music').fire('input');
  assert.equal(page.$('color-accent').value, '#oops');
  assert.equal(page.$('color-accent').getAttribute('aria-invalid'), 'true');
  await page.save();
  assert.equal(page.$('colors-more').open, true);
  assert.equal(Boolean(page.$('colors-fold')), false);
  assert.equal(page.requests.some((request) => request.url === '/api/save'), false);
  await page.click('reset');
  assert.ok(choices.normalizeHex(page.$('color-accent').value));
  assert.equal(page.$('color-accent').getAttribute('aria-invalid'), null);
  assert.equal(page.$('error-accent').textContent, '');
  await page.save(); assert.equal(page.$('saved').hidden, false);
});

test('opening only Colors More exposes editable hex fields before the website fold', async () => {
  const page = await harness(initialState(), '#colors');
  page.$('colors-more').open = true; await page.$('colors-more').fire('toggle');
  const field = page.$('color-accent'), folds = [];
  for (let parent = field.parentElement; parent; parent = parent.parentElement) {
    assert.equal(parent.hidden, false);
    if (parent.tagName === 'details') { folds.push(parent.id); assert.equal(parent.open, true); }
  }
  assert.deepEqual(folds, ['colors-more']);
  field.value = '#114488'; await field.fire('change');
  await page.save();
  assert.equal(page.requests.find((r) => r.url === '/api/save').data.style.hook.accent_color, '#114488');
});

test('Your colors opens More without input focus while color fold hashes focus the field', async () => {
  for (const hash of ['#colors-fold', '#colors-more']) {
    const page = await harness(initialState(), hash);
    assert.equal(page.document.body.dataset.screen, 'colors');
    assert.equal(page.$('colors-more').open, true);
    assert.equal(page.document.activeElement.id, 'color-accent');
  }
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#colors', { phone });
    page.$('step-colors').focus();
    await page.pick('palette-tiles', 'Your colors');
    assert.equal(page.$('colors-more').open, true);
    assert.equal(page.document.activeElement.id, 'step-colors');
    assert.equal(page.$('color-fields').scrolled?.block, 'center');
    await page.go('font'); page.location.hash = '#colors-more'; page.windowEvents.hashchange(); await flush();
    assert.equal(page.document.body.dataset.screen, 'colors');
    assert.equal(page.$('colors-more').open, true);
  }
});

test('Marker loads and saves distinct title and card paper without collapsing either', async () => {
  const page = await harness(markerState(), '#colors');
  assert.equal(page.$('color-paper').value, '#FFFDF4');
  assert.equal(page.$('picker-paper').value, '#FFFDF4');
  const preview = page.requests.find((request) => request.url === '/api/layers').data.style;
  assert.equal(preview.hook.background_color, '#FFF6DC');
  assert.equal(preview.graphics.paper_color, '#FFFDF4');
  await page.save();
  const unchanged = page.requests.find((request) => request.url === '/api/save').data.style;
  assert.equal(unchanged.hook.background_color, '#FFF6DC');
  assert.equal(unchanged.graphics.paper_color, '#FFFDF4');
  await page.click('update');
  // An unrelated caption edit must not copy card paper into the title background.
  page.$('color-caption').value = '#ABCDEF';
  await page.$('color-caption').fire('change');
  await page.save();
  const saved = page.requests.filter((request) => request.url === '/api/save').at(-1).data.style;
  assert.equal(saved.captions.text_color, '#ABCDEF');
  assert.equal(saved.hook.background_color, '#FFF6DC');
  assert.equal(saved.graphics.paper_color, '#FFFDF4');
  assert.equal(saved.hook.accent_color, '#FF2E88');
  assert.equal(saved.graphics.accent_color, '#FF2E88');
  const reloaded = await harness(page.state, '#colors');
  assert.equal(reloaded.$('color-paper').value, '#FFFDF4');
});

test('manual card colors edit graphics and only the matching title color', async () => {
  const state = markerState();
  state.style.hook.text_color = '#202020';
  const page = await harness(state, '#colors');
  assert.equal(page.$('color-ink').value, '#1D1D1D');
  page.$('color-paper').value = '#ABCDEF';
  await page.$('color-paper').fire('change');
  let preview = page.requests.filter((request) => request.url === '/api/layers').at(-1).data.style;
  assert.equal(preview.graphics.paper_color, '#ABCDEF');
  assert.equal(preview.hook.background_color, '#ABCDEF');
  assert.equal(preview.hook.text_color, '#202020');
  assert.equal(preview.graphics.ink_color, '#1D1D1D');
  page.$('color-ink').value = '#101010';
  await page.$('color-ink').fire('change');
  await page.save();
  const saved = page.requests.find((request) => request.url === '/api/save').data.style;
  assert.equal(saved.graphics.paper_color, '#ABCDEF');
  assert.equal(saved.graphics.ink_color, '#101010');
  assert.equal(saved.hook.background_color, '#ABCDEF');
  assert.equal(saved.hook.text_color, '#101010');
});

test('a saved brand accent survives every card choice, Save and reload', async (t) => {
  for (const look of ['Paper', 'Minimal', 'Dark', 'Bright']) await t.test(look, async () => {
    const state = markerState();
    state.from_library = true;
    state.style.brand.accent_color = '#FF0000';
    const original = structuredClone(state.style);
    const page = await harness(state, '#more');
    await page.pick('card-style-tiles', look);
    await page.save();
    const saved = page.requests.find((request) => request.url === '/api/save').data.style;
    assert.equal(saved.brand.accent_color, '#FF0000');
    assert.deepEqual(saved.brand, original.brand);
    assert.deepEqual(saved.captions, original.captions);
    assert.deepEqual(saved.hook, original.hook);
    assert.equal(saved.graphics.accent_color, original.graphics.accent_color);
    if (look === 'Dark') {
      assert.equal(saved.graphics.paper_color, original.graphics.ink_color);
      assert.equal(saved.graphics.ink_color, '#FFFDF4');
    }
    const reloaded = await harness(page.state, '#colors');
    assert.equal(reloaded.$('color-accent').value, '#FF0000');
    assert.equal(reloaded.$('color-paper').value, saved.graphics.paper_color);
  });
});

test('explicitly editing Highlight replaces the brand accent and survives Dark then Save', async () => {
  const state = markerState();
  state.style.brand.accent_color = '#FF0000';
  const page = await harness(state, '#colors');
  page.$('color-accent').value = '#22BBFF';
  await page.$('color-accent').fire('change');
  await page.click('step-more');
  await page.pick('card-style-tiles', 'Dark');
  await page.save();
  const saved = page.requests.find((request) => request.url === '/api/save').data.style;
  assert.equal(saved.brand.accent_color, undefined);
  assert.equal(saved.captions.highlight_color, '#22BBFF');
  assert.equal(saved.hook.accent_color, '#22BBFF');
  assert.equal(saved.graphics.accent_color, '#22BBFF');
  assert.ok(choices.luminance(saved.graphics.paper_color) < .3);
});

test('a monochrome website keeps the saved brand accent through a card choice and Save', async () => {
  const state = markerState();
  state.style.brand.accent_color = '#FF0000';
  const page = await harness(state, '#colors');
  page.$('site-url').value = 'example.com';
  await page.click('site-pull');
  await page.click('step-more');
  await page.pick('card-style-tiles', 'Paper');
  await page.save();
  const saved = page.requests.find((request) => request.url === '/api/save').data.style;
  assert.equal(saved.brand.accent_color, '#FF0000');
  assert.equal(saved.graphics.paper_color, '#FFFFFF');
  assert.equal(saved.graphics.ink_color, '#111111');
});

test('the footer reserves its measured height and leaves the name in a sheet', async () => {
  const page = await harness(initialState(), '#font');
  assert.equal(page.document.documentElement.style['--savebar-height'], '140px');
  assert.equal(page.$('style-name').parent, page.$('save-sheet'));
  assert.equal(page.$('step-navigation').parent.parent.parent.id, 'savebar');
  assert.deepEqual(page.$('step-navigation').children.map((node) => node.id), ['back', 'next']);
  assert.equal(page.$('reset').parent.parent.id, 'step-header');
  page.$('savebar').measuredHeight = 64.2;
  page.resizeObservers[0].callback();
  assert.equal(page.document.documentElement.style['--savebar-height'], '65px');
  const css = readFileSync(new URL('src/yap_studio/gallery/static/pick.css', root), 'utf8');
  const phone = css.split('@media (max-width: 899px) {')[1];
  assert.match(css, /padding: 28px 28px calc\(var\(--savebar-height\) \+ 24px\)/);
  assert.match(phone, /padding: 14px 16px calc\(var\(--savebar-height\) \+ 24px\)/);
  assert.match(phone, /\.sticky-preview \{ position: sticky; top: var\(--topbar-height\)/);
  assert.match(phone, /\.topbar \{ position: sticky; top: 0/);
  assert.match(css, /body:not\(\[data-screen=looks\]\) \.save-name \{ display: block/);
  assert.match(css, /\.save-actions,\.step-navigation \{ display: grid; grid-template-columns: repeat\(2,minmax\(0,1fr\)\)/);
});

test('desktop template tiles pause at caption and title posters, play on hover, and restore posters', async () => {
  const page = await harness(initialState(), '#templates', { layers: true });
  page.observers[1].intersect();
  await flush();
  const host = page.$('start-options').children[0].children[0];
  const frames = host.querySelectorAll('iframe');
  assert.equal(frames.length, 2);
  for (const frame of frames) await frame.fire('load');
  await flush();
  assert.deepEqual(frames.map((frame) => frame.contentWindow.__player.getTime()), [.65, 1.5]);
  assert.ok(frames.every((frame) => frame._galleryReady && !frame.contentWindow.__player.isPlaying()));
  await host.fire('mouseenter');
  assert.ok(frames.every((frame) => frame.contentWindow.__player.isPlaying()));
  await frames[0].contentWindow.__player.seek(3.9);
  await host.fire('mouseleave');
  assert.deepEqual(frames.map((frame) => frame.contentWindow.__player.getTime()), [.65, 1.5]);
  assert.ok(frames.every((frame) => !frame.contentWindow.__player.isPlaying()));
});

test('reopening live folds keeps one hover handler pair and controls only the current stage', async () => {
  const page = await harness(initialState(), '#captions', { layers: true });
  await openFold(page, 'words-fold');
  page.observers[1].intersect(); await flush();
  const host = page.$('word-tiles').children[0].children[0];
  const disposed = [];
  for (let cycle = 0; cycle < 3; cycle++) {
    const stage = host._galleryStage;
    assert.ok(stage?.active);
    assert.equal(host.events.mouseenter.length, 1);
    assert.equal(host.events.mouseleave.length, 1);
    for (const frame of stage.frames) await frame.fire('load');
    await flush();
    await host.fire('mouseenter');
    assert.equal(stage.hovered, true);
    assert.ok(stage.frames.every((frame) => frame.contentWindow.__player.isPlaying()));
    for (const old of disposed) assert.equal(old.hovered, false);
    await host.fire('mouseleave');
    assert.equal(stage.hovered, false);
    assert.ok(stage.frames.every((frame) => !frame.contentWindow.__player.isPlaying()
      && frame.contentWindow.__player.getTime() === stage.posterTimes.get(frame)));
    const fold = page.$('words-fold');
    fold.open = false; await fold.fire('toggle'); await flush();
    assert.equal(stage.active, false);
    assert.equal(host._galleryStage, null);
    disposed.push(stage);
    await host.fire('mouseenter');
    assert.equal(stage.hovered, false);
    await host.fire('mouseleave');
    await openFold(page, 'words-fold');
    page.observers[1].intersect(); await flush();
    assert.notEqual(host._galleryStage, stage);
  }
});

test('phone tiles have posters before visibility, then play in view and restore on leaving', async () => {
  const page = await harness(initialState(), '#captions', { layers: true, phone: true });
  for (const id of ['caption-animation', 'words-fold']) { page.$(id).open = true; await page.$(id).fire('toggle'); }
  page.observers[1].intersect();
  await flush();
  const host = page.$('word-tiles').children[0].children[0];
  const frame = host.querySelectorAll('iframe')[0];
  await frame.fire('load');
  await flush();
  assert.equal(frame.contentWindow.__player.getTime(), .65);
  assert.equal(frame.contentWindow.__player.isPlaying(), false);
  page.observers[0].callback([{ target: host, isIntersecting: true }]);
  assert.equal(frame.contentWindow.__player.isPlaying(), true);
  await frame.contentWindow.__player.seek(3.9);
  page.observers[0].callback([{ target: host, isIntersecting: false }]);
  assert.equal(frame.contentWindow.__player.isPlaying(), false);
  assert.equal(frame.contentWindow.__player.getTime(), .65);
});

test('Dark selects a dark engine paper/ink pair and that visible treatment is saved', async () => {
  const page = await harness(initialState(), '#cards');
  page.observers[1].intersect();
  await flush();
  const candidates = page.requests.filter((request) => request.data?.tile.tab === 'cards').map((request) => request.data.style);
  const dark = candidates.find((style) => style.graphics.style === 'dark');
  const paper = candidates.find((style) => style.graphics.style === 'paper');
  assert.ok(choices.luminance(dark.graphics.paper_color) < .3);
  assert.ok(choices.luminance(paper.graphics.paper_color) >= .3);
  await page.pick('card-style-tiles', 'Dark');
  await page.save();
  const saved = page.requests.find((request) => request.url === '/api/save').data.style;
  assert.equal(saved.graphics.style, 'dark');
  assert.equal(saved.graphics.paper_color, dark.graphics.paper_color);
  assert.equal(saved.graphics.ink_color, dark.graphics.ink_color);
  const reloaded = await harness(page.state, '#colors');
  assert.equal(reloaded.$('color-paper').value, dark.graphics.paper_color);
  assert.equal(reloaded.$('color-ink').value, dark.graphics.ink_color);
});

test('a card poster seeks again after the runtime binds its GSAP timeline', async () => {
  const page = await harness(initialState(), '#cards', { layers: true });
  page.observers[1].intersect();
  await flush();
  const host = page.$('card-style-tiles').children[0].children[0];
  const frame = host.querySelectorAll('iframe')[0];
  const player = frame.contentWindow.__player;
  const seek = player.seek;
  let bound = false;
  player.seek = (at) => {
    if (!bound) { bound = true; return Promise.resolve(); }
    return seek(at);
  };
  await frame.fire('load');
  await flush();
  assert.equal(player.getTime(), 1.5);
  assert.equal(frame._galleryReady, true);
  assert.equal(player.isPlaying(), false);
});

test('the page opens on library-first looks with same-screen Save and Customize', async () => {
  const page = await harness();
  assert.equal(page.document.body.dataset.screen, 'looks');
  assert.equal(page.$('panel-looks').hidden, false);
  assert.equal(page.$('panel-start'), undefined);
  assert.equal(page.$('choose-path'), undefined);
  assert.equal(page.$('steps').hidden, true);
  assert.equal(page.$('step-header').hidden, true);
  assert.equal(page.$('step-navigation').hidden, true);
  assert.equal(page.$('savebar').hidden, false);
  assert.equal(page.$('customize').hidden, false);
  assert.equal(page.$('customize').textContent, 'Customize');
  assert.deepEqual(page.$('start-options').children.map((tile) => tile.children[1].text),
    ['SparrowLight', 'Editorial', 'Clean Minimal']);
  await page.pick('start-options', 'SparrowLight');
  const preview = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data;
  assert.equal(preview.tile.step, 'looks');
  assert.equal(preview.style.name, 'SparrowLight');
  assert.equal(page.$('style-name').value, 'SparrowLight');
  await page.save();
  assert.equal(page.$('saved').hidden, false);
  assert.equal(page.requests.filter((r) => r.url === '/api/save').length, 1);
  const reload = await harness(page.state);
  assert.equal(reload.$('steps').hidden, true);
  assert.equal(reload.$('style-name').value, 'SparrowLight');
  assert.equal(reload.$('start-options').children[0].getAttribute('aria-pressed'), 'true');
});

test('old landing hashes restore looks; builder links return without discarding edits', async () => {
  for (const hash of ['', '#start', '#templates', '#looks', '#unknown']) {
    const page = await harness(initialState(), hash, { phone: true });
    assert.equal(page.document.body.dataset.screen, 'looks');
    assert.equal(page.$('panel-looks').hidden, false);
    await page.click('customize');
    await page.pick('width-tiles', 'Narrow');
    await page.click('back');
    assert.equal(page.location.hash, '#looks');
    await page.click('customize');
    assert.equal(page.$('width-tiles').children[2].getAttribute('aria-pressed'), 'true');
    for (const step of ['font', 'captions', 'colors', 'more']) {
      await page.click(`step-${step}`);
      assert.equal(page.$('step-header').hidden, false);
      await page.click('back-to-looks');
      assert.equal(page.$('panel-looks').hidden, false);
      assert.equal(page.$('steps').hidden, true);
      assert.equal(page.$('step-navigation').hidden, true);
      assert.equal(page.$('savebar').hidden, false);
      assert.equal(page.requests.some((request) => request.url === '/api/save'), false);
      await page.click('customize');
    }
  }
});

test('review guidance describes Looks, the core three and More choices', async () => {
  const guidance = readFileSync(new URL('docs/style-page.md', root), 'utf8');
  const flow = guidance.split('## How the page works\n')[1].split('\n## ')[0].replace(/\s+/g, ' ');
  assert.match(flow, /opens on looks under your video/);
  assert.match(flow, /Customize opens four screens/);
  assert.match(flow, /Back to looks returns to the grid/);
  assert.match(flow, /Grade, Opening, Cards, B-roll, Moves, Moments and Sound/);
  assert.match(flow, /A screen stays hidden/);
  assert.match(flow, /A tile previews the effect without turning it on/);
  assert.match(flow, /More starts closed/);
  const page = await harness();
  assert.deepEqual(page.document.querySelectorAll('[data-step]').map((tab) => tab.textContent.replace(' Optional', '')),
    ['Font', 'Captions', 'Colors', 'Text at the top', 'More choices']);
});

test('Customize preserves the complete look currently on screen', async () => {
  const state = markerState();
  const page = await harness(state);
  await page.click('customize');
  assert.equal(page.location.hash, '#font');
  assert.equal(page.$('steps').hidden, false);
  assert.equal(page.$('step-count').textContent, 'Step 1 of 4');
  let preview = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style;
  assert.deepEqual(preview, state.style);
  await page.click('back');
  assert.equal(page.location.hash, '#looks');
  for (const name of ['Clean Minimal', 'SparrowLight', 'Marker']) {
    await page.pick('start-options', name);
    const selected = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style;
    await page.click('customize');
    preview = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style;
    assert.deepEqual(preview, selected);
    await page.click('back-to-looks');
  }
});

test('every font has a compact name tile in its own font, including creator and saved system fonts', async () => {
  const state = initialState();
  state.fonts.creator.push('Creator Serif');
  state.style = choices.withFont(state.style, 'Arial');
  const page = await harness(state, '#font', { phone: true });
  const fonts = [...bundledFonts, 'Creator Serif', 'Arial'];
  const buttons = page.$('font-options').children;
  assert.deepEqual(buttons.map((button) => button.textContent), fonts);
  for (const [index, button] of buttons.entries()) {
    assert.equal(button.style.fontFamily, choices.fontFamily(fonts[index], state.fonts));
    assert.equal(button.querySelectorAll('.stage-host').length, 0);
    await page.pick('font-options', fonts[index]);
    const selected = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style;
    assert.equal(selected.captions.font, fonts[index]);
    assert.equal(selected.hook.font, fonts[index]);
    assert.equal(selected.graphics.font, fonts[index]);
  }
  assert.equal(page.$('font-options').children.filter((button) =>
    button.getAttribute('aria-pressed') === 'true').length, 1);
  const css = readFileSync(new URL('src/yap_studio/gallery/static/pick.css', root), 'utf8');
  assert.match(css, /\.font-options \{ display: grid; grid-template-columns: repeat\(3,minmax\(0,1fr\)\)/);
  assert.match(css, /\.font-options \{ grid-template-columns: repeat\(2,minmax\(0,1fr\)\)/);
  assert.doesNotMatch(css, /overflow-x:\s*(auto|scroll)|scroll-snap/);
  assert.doesNotMatch(html, /Swipe for more/);
});

const lastFontPreview = (page) => page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data;
const pressed = (page, id) => page.$(id).children.filter((button) => button.getAttribute('aria-pressed') === 'true');
for (const phone of [false, true]) {
  test(`start look graphics fonts remain available after a change on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState();
    state.style.captions.font = 'Inter';
    state.style.hook.font = state.style.graphics.font = 'Arial';
    const page = await harness(state, '#font', { phone });
    assert.equal(page.$('font-more').open, false);
    await page.pick('graphics-font-options', 'Geist');
    assert.ok(page.$('graphics-font-options').children.some((button) => button.textContent === 'Arial'));
    await page.pick('graphics-font-options', 'Arial');
    assert.equal(lastFontPreview(page).style.captions.font, 'Inter');
    assert.equal(lastFontPreview(page).style.hook.font, 'Arial');
    assert.equal(lastFontPreview(page).style.graphics.font, 'Arial');
  });
  test(`same font saves the existing complete payload on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState(), expected = choices.withFont(state.style, 'Gelasio');
    const page = await harness(state, '#font', { phone });
    assert.equal(page.$('font-same').getAttribute('aria-pressed'), 'true');
    assert.deepEqual(page.$('font-options').children.map((button) => button.textContent), bundledFonts);
    await page.pick('font-options', 'Gelasio');
    assert.deepEqual(lastFontPreview(page).style, expected);
    await page.save();
    const payload = page.requests.find((r) => r.url === '/api/save').data.style;
    assert.deepEqual(payload, expected);
    assert.equal(JSON.stringify(payload), JSON.stringify(expected));
  });
  test(`separate font roles preview and save independently on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState();
    state.style.hook.font = 'Gelasio';
    const original = structuredClone(state.style);
    const page = await harness(state, '#font', { phone });
    await page.pick('font-options', 'Instrument Serif');
    assert.deepEqual(lastFontPreview(page).style, choices.withCaptionsFont(original, 'Instrument Serif'));
    await page.pick('graphics-font-options', 'Geist');
    assert.deepEqual(lastFontPreview(page).tile, { tab: 'preview', step: 'font', section: 'cards' });
    assert.equal(pressed(page, 'graphics-font-options')[0].textContent, 'Geist');
    assert.equal(page.$('font-pair').getAttribute('aria-pressed'), 'true');
    await page.click('step-captions'); await page.click('step-font');
    assert.equal(page.$('graphics-font-fold').open, true);
    assert.equal(page.$('font-note').textContent, 'Captions: Instrument Serif · Title: Geist · Cards: Geist');
    await page.save();
    assert.deepEqual(page.requests.find((r) => r.url === '/api/save').data.style, choices.withFontPair(original));
  });
  test(`font pairing, same font and Reset on ${phone ? 'phone' : 'desktop'}`, async () => {
    const page = await harness(initialState(), '#font', { phone });
    await page.click('font-pair');
    assert.equal(page.$('font-pair').getAttribute('aria-pressed'), 'true');
    assert.deepEqual(lastFontPreview(page).style, choices.withFontPair(page.state.style));
    const samples = page.$('font-pair').querySelectorAll('span').filter((span) => span.style.fontFamily);
    assert.deepEqual(samples.map((span) => span.style.fontFamily),
      ['Instrument Serif', 'Geist'].map((font) => choices.fontFamily(font, page.state.fonts)));
    await page.click('font-same');
    assert.deepEqual(lastFontPreview(page).style, choices.withFont(page.state.style, 'Instrument Serif'));
    assert.deepEqual(lastFontPreview(page).tile, { tab: 'preview', step: 'font', section: 'cards' });
    await page.click('reset');
    assert.deepEqual(lastFontPreview(page).style, page.state.style);
    assert.deepEqual(lastFontPreview(page).tile, { tab: 'preview', step: 'font' });
    await page.go('looks'); await page.pick('start-options', 'Clean Minimal'); await page.click('customize');
    assert.equal(page.$('font-same').getAttribute('aria-pressed'), 'false');
    const split = lastFontPreview(page).style;
    await page.pick('graphics-font-options', 'Geist'); await page.click('reset');
    assert.deepEqual(lastFontPreview(page).style, split);
  });
  test(`saved graphics fonts and brand overrides stay in step on ${phone ? 'phone' : 'desktop'}`, async () => {
    const state = initialState();
    state.style.brand = { ...state.style.brand, captions_font: 'Creator Serif', titles_font: 'Arial', cards_font: 'Georgia' };
    const original = structuredClone(state.style);
    const page = await harness(state, '#font', { phone });
    for (const name of ['Creator Serif', 'Arial', 'Georgia']) assert.ok(page.$('font-options').children.some((button) => button.textContent === name));
    await page.pick('font-options', 'Instrument Serif'); await page.pick('graphics-font-options', 'Geist');
    await page.save();
    assert.deepEqual(page.requests.find((r) => r.url === '/api/save').data.style, choices.withFontPair(original));
  });
}
test('letter width stays folded after the font grid and preserves a custom value', async () => {
  const page = await harness(initialState(), '#font', { phone: true });
  assert.ok(html.indexOf('id="font-options"') < html.indexOf('id="width-tiles"'));
  assert.equal(page.$('font-more').open, false);
  assert.equal(page.$('width-fold').open, false);
  assert.deepEqual(page.$('width-tiles').children.map((el) => el.textContent), ['Normal', 'Condensed', 'Narrow']);
  await page.pick('width-tiles', 'Narrow');
  assert.equal(lastFontPreview(page).style.captions.width_scale, .7);
});

test('font selection previews the caption with its current color, shadow, placement and size', async () => {
  const state = markerState();
  Object.assign(state.style.captions, { text_color: '#ABCDEF', position: 'middle', top_px: 700,
    size_px: 118, shadow_opacity: .8, shadow_blur_px: 3, shadow_x_px: 0, shadow_y_px: 6,
    background_mode: 'box', background_color: '#FFFFFF', background_opacity: .63,
    background_padding_px: 27, background_radius_px: 23, placement: 'fixed' });
  const original = structuredClone(state.style.captions);
  const page = await harness(state, '#font');
  await page.pick('font-options', 'Instrument Serif');
  const preview = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style;
  assert.equal(preview.captions.font, 'Instrument Serif');
  for (const field of ['text_color', 'position', 'top_px', 'size_px', 'shadow_opacity', 'shadow_blur_px', 'shadow_x_px', 'shadow_y_px',
    'background_mode', 'background_color', 'background_opacity', 'background_padding_px', 'background_radius_px', 'placement']) {
    assert.equal(preview.captions[field], original[field]);
  }
});

test('Captions puts treatments first and keeps all existing controls inside named folds', async () => {
  const page = await harness(initialState(), '#captions');
  const ids = page.$('panel-captions').querySelectorAll('[id]').map((el) => el.id);
  assert.equal(page.$('caption-animation').open, false);
  assert.equal(page.$('caption-look-tiles').children.length, 8);
  const controls = ['caption-look-tiles', 'word-tiles', 'highlight-tiles', 'italics', 'avoid-face', 'placements', 'caption-position', 'sizes', 'caption-color', 'shadow-toggle', 'shadow-strength', 'background-fields', 'motion', 'fine-tune'];
  assert.deepEqual(controls.map((id) => ids.indexOf(id)), controls.map((id) => ids.indexOf(id)).sort((a, b) => a - b));
  assert.equal(page.$('caption-position').parent.parent.id, 'position-fields');
  assert.equal(page.$('fine-size_px').parent.parent.id, 'fine-fields');
  assert.equal(page.$('caption-look-tiles').children.length, 8);
  assert.equal(page.$('avoid-face').checked, true);
  assert.equal(page.$('highlight-tiles').children.length, schema.properties.captions.properties.highlight_style.enum.length - 1);
  const press = async (id, dataset) => {
    await page.$(id).fire('click', { target: { dataset } }); await flush();
  };
  page.$('caption-color').value = '#ACF';
  await page.$('caption-color').fire('change'); await flush();
  await press('shadow-strength', { shadow: 'strong' });
  await press('shadow-toggle', { shadowToggle: 'off' });
  assert.ok(page.$('shadow-strength').children.every((el) => el.disabled));
  await press('shadow-toggle', { shadowToggle: 'on' });
  await press('placements', { place: 'top' });
  await press('sizes', { size: 'large' });
  await page.pick('word-tiles', 'One word');
  await page.pick('highlight-tiles', 'No highlight');
  let preview = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style;
  assert.equal(preview.captions.text_color, '#AACCFF');
  assert.equal(page.$('color-caption').value, '#AACCFF');
  assert.equal(preview.captions.shadow_opacity, .8);
  assert.equal(preview.captions.shadow_blur_px, 3);
  assert.equal(preview.captions.shadow_x_px, 0);
  assert.equal(preview.captions.shadow_y_px, 6);
  assert.equal(preview.captions.position, 'middle');
  assert.equal(preview.captions.top_px, 700);
  assert.equal(preview.captions.size_px, 118);
  assert.equal(preview.captions.italic_stress, false);
  page.$('fine-top_px').value = 850;
  await page.$('fine-top_px').fire('input'); await flush();
  await page.save();
  preview = page.requests.find((r) => r.url === '/api/save').data.style;
  assert.equal(preview.captions.top_px, 850);
  assert.equal(preview.captions.size_px, 118);
  assert.equal(preview.captions.text_color, '#AACCFF');
});

test('Background tiles are lazy and keep color, padding, corners, shadow and placement on every take preview', async () => {
  const state = initialState();
  Object.assign(state.style.captions, { background_opacity: .63, background_padding_px: 27,
    background_radius_px: 23, background_color: '#AACCFF', placement: 'fixed', top_px: 850 });
  const page = await harness(state, '#captions', { phone: true });
  assert.equal(page.$('background-fields').hidden, true);
  assert.equal(page.$('avoid-face').checked, false);
  assert.equal(page.requests.filter((r) => r.data?.tile?.tab === 'captions').length, 0);
  const hosts = page.$('caption-look-tiles').children.slice(0, 3).flatMap((tile) => tile.querySelectorAll('.stage-host'));
  page.observers[1].callback(hosts.map((target) => ({ target, isIntersecting: true })));
  await flush();
  const requests = page.requests.filter((r) => r.data?.tile?.tab === 'captions');
  assert.deepEqual(requests.map((r) => r.data.style.captions.background_mode), ['none', 'box', 'band']);
  for (const { data } of requests) {
    for (const field of ['background_color', 'background_opacity', 'background_padding_px', 'background_radius_px',
      'placement', 'top_px', 'shadow_opacity']) assert.equal(data.style.captions[field], state.style.captions[field]);
  }
  await page.pick('background-tiles', 'Box');
  assert.equal(page.$('background-fields').hidden, false);
  assert.equal(page.$('background-corner-fields').hidden, false);
  assert.equal(page.$('fine-field-background_radius_px').hidden, false);
  await page.pick('background-swatches', 'White');
  assert.match(page.$('caption-result').textContent, /Text switches to dark on this background so it stays readable.*#111111/);
  await page.pick('background-tiles', 'Band');
  assert.equal(page.$('background-corner-fields').hidden, true);
  assert.equal(page.$('fine-field-background_radius_px').hidden, true);
  await page.pick('background-tiles', 'None');
  assert.equal(page.$('background-fields').hidden, true);
  assert.equal(page.$('caption-result').hidden, true);
  await page.pick('background-tiles', 'Box');
  assert.equal(page.$('background-color').value, '#FFFFFF');
  assert.equal(page.$('fine-background_opacity').value, .63);
  assert.equal(page.$('fine-background_radius_px').value, 23);
  await page.click('step-font');
  assert.equal(page.$('caption-look-tiles').children.length, 0);
});

test('background, exact fine-tuning and face switch round-trip through save, templates and reset', async () => {
  const state = initialState();
  const original = structuredClone(state.style.captions);
  const page = await harness(state, '#captions');
  const press = async (id, dataset) => {
    await page.$(id).fire('click', { target: { dataset } }); await flush();
  };
  await page.pick('background-tiles', 'Box');
  for (const [choice, value] of [['light', .5], ['medium', .75], ['solid', .9]]) {
    await press('background-opacity', { opacity: choice });
    assert.equal(page.$('fine-background_opacity').value, value);
  }
  for (const [choice, value] of [['square', 0], ['rounded', 12], ['pill', 40]]) {
    await press('background-corners', { corners: choice });
    assert.equal(page.$('fine-background_radius_px').value, value);
  }
  page.$('background-picker').value = '#FFFFFF';
  await page.$('background-picker').fire('input'); await flush();
  assert.match(page.$('caption-result').textContent, /dark/);
  page.$('background-color').value = '#ACF';
  await page.$('background-color').fire('change'); await flush();
  for (const [field, value] of [['background_opacity', .63], ['background_padding_px', 27], ['background_radius_px', 23]]) {
    page.$(`fine-${field}`).value = value;
    await page.$(`fine-${field}`).fire('input'); await flush();
  }
  page.$('avoid-face').checked = false;
  await page.$('avoid-face').fire('change'); await flush();
  await press('placements', { place: 'top' });
  let captions = page.requests.filter((r) => r.data?.tile?.tab === 'preview').at(-1).data.style.captions;
  const expected = { background_mode: 'box', background_color: '#AACCFF', background_opacity: .63,
    background_padding_px: 27, background_radius_px: 23, placement: 'fixed', position: 'middle', top_px: 700 };
  for (const [field, value] of Object.entries(expected)) assert.equal(captions[field], value);
  await page.save();
  captions = page.requests.find((r) => r.url === '/api/save').data.style.captions;
  for (const [field, value] of Object.entries(expected)) assert.equal(captions[field], value);
  assert.equal(captions.text_color, original.text_color);
  const reopened = await harness(state, '#captions');
  assert.equal(reopened.$('avoid-face').checked, false);
  assert.equal(reopened.$('background-color').value, '#AACCFF');
  assert.equal(reopened.$('fine-background_opacity').value, .63);
  assert.ok(reopened.$('background-opacity').children.every((el) => el.getAttribute('aria-pressed') === 'false'));
  await page.click('update');
  await page.pick('background-tiles', 'Band');
  await press('background-opacity', { opacity: 'solid' });
  await page.click('reset');
  captions = page.requests.filter((r) => r.data?.tile?.tab === 'preview').at(-1).data.style.captions;
  for (const [field, value] of Object.entries(expected)) assert.equal(captions[field], value);
  await page.click('back-to-looks');
  await page.pick('start-options', 'Clean Minimal');
  await page.click('customize');
  await page.click('step-captions');
  assert.equal(page.$('background-fields').hidden, true);
  assert.equal(page.$('avoid-face').checked, true);
  await page.pick('background-tiles', 'Band');
  await page.click('reset');
  captions = page.requests.filter((r) => r.data?.tile?.tab === 'preview').at(-1).data.style.captions;
  for (const field of Object.keys(expected).filter((key) => key.startsWith('background_') || key === 'placement')) {
    assert.equal(captions[field], original[field]);
  }
  await page.click('back-to-looks');
  await page.pick('start-options', 'My style');
  await page.click('customize');
  await page.click('step-captions');
  captions = page.requests.filter((r) => r.data?.tile?.tab === 'preview').at(-1).data.style.captions;
  for (const [field, value] of Object.entries(expected)) assert.equal(captions[field], value);
});

test('highlight swatch uses the caption highlight and background text follows contrast without changing caption color', async () => {
  const state = initialState();
  state.style.captions.highlight_color = '#123456';
  state.style.graphics.accent_color = '#FFFFFF';
  const page = await harness(state, '#captions');
  await page.pick('background-tiles', 'Box');
  await page.pick('background-swatches', 'Highlight');
  assert.equal(page.$('background-color').value, '#123456');
  assert.match(page.$('caption-result').textContent, /light.*#FFFFFF/);
  assert.equal(page.$('caption-color').value, state.style.captions.text_color);
  page.$('fine-background_opacity').value = 0;
  await page.$('fine-background_opacity').fire('input'); await flush();
  assert.equal(page.$('caption-result').hidden, true);
  page.$('avoid-face').checked = false;
  await page.$('avoid-face').fire('change'); await flush();
  page.$('avoid-face').checked = true;
  await page.$('avoid-face').fire('change'); await flush();
  assert.equal(page.requests.filter((r) => r.data?.tile?.tab === 'preview').at(-1).data.style.captions.placement, 'avoid_face');
  state.style.brand.accent_color = '#ABCDEF';
  const branded = await harness(state, '#captions');
  await branded.pick('background-tiles', 'Box');
  await branded.pick('background-swatches', 'Highlight');
  assert.equal(branded.$('background-color').value, '#ABCDEF');
});

test('invalid background hex blocks saving and reset restores the preset', async () => {
  const page = await harness(initialState(), '#captions');
  await page.pick('background-tiles', 'Box');
  page.$('background-color').focus();
  page.$('background-color').value = '#oops';
  await page.$('background-color').fire('change');
  await page.save();
  assert.equal(page.requests.some((r) => r.url === '/api/save'), false);
  assert.equal(page.$('save-error').textContent, 'Check your background color.');
  assert.equal(page.$('background-color').getAttribute('aria-invalid'), 'true');
  await page.click('reset');
  assert.equal(page.$('background-color').getAttribute('aria-invalid'), null);
  assert.equal(page.$('background-color').value, '#000000');
  await page.save();
  assert.equal(page.$('saved').hidden, false);
});

test('Box and Band restore Medium only from zero opacity and save only their fields', async () => {
  for (const look of ['Plain', 'Box', 'Band']) for (const opacity of [0, .37]) {
    const state = initialState(); state.style.captions.background_opacity = opacity;
    const original = structuredClone(state.style), page = await harness(state, '#captions');
    await page.pick('caption-look-tiles', look);
    const expected = { 'captions.background_mode': look === 'Plain' ? 'none' : look.toLowerCase() };
    if (look !== 'Plain' && opacity === 0) expected['captions.background_opacity'] = .75;
    const rendered = page.requests.filter((request) => request.url === '/api/layers' && request.data.tile.tab === 'preview').at(-1).data.style;
    assert.equal(rendered.captions.background_opacity, look === 'Plain' ? opacity : opacity || .75);
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, expected));
  }
});

test('Plain clears an invalid background draft and saves the last valid color', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const page = await harness(state, '#captions');
  await page.pick('caption-look-tiles', 'Box');
  await openFold(page, 'background-fields');
  const input = page.$('background-color');
  input.value = '#ABCDEF'; await input.fire('change'); await flush();
  input.focus(); input.value = '#oops'; await input.fire('change'); await flush();
  assert.equal(input.getAttribute('aria-invalid'), 'true');
  assert.ok(page.$('background-error').textContent);
  await page.pick('caption-look-tiles', 'Plain');
  assert.equal(page.$('background-fields').hidden, true);
  assert.equal(input.value, '#ABCDEF');
  assert.equal(input.getAttribute('aria-invalid'), null);
  assert.equal(page.$('background-error').textContent, '');
  await page.save();
  assert.equal(page.$('save-error').textContent, '');
  assert.deepEqual(savePayload(page), withFields(original, { 'captions.background_color': '#ABCDEF' }));
});

test('Save ignores a hidden background draft without changing the saved style', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const page = await harness(state, '#captions');
  assert.equal(page.$('background-fields').hidden, true);
  page.$('background-color').value = '#oops';
  page.$('background-color').setAttribute('aria-invalid', 'true');
  await page.save();
  assert.equal(page.$('save-error').textContent, '');
  assert.deepEqual(savePayload(page), original);
});

test('desktop and phone keep background, shadow and placement inside their folds', async () => {
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#captions', { phone });
    assert.equal(page.$('caption-color-fold').open, false);
    assert.equal(page.$('background-fields').open, false);
    assert.equal(page.$('position-fold').open, false);
    assert.equal(page.$('caption-color').parent.parent.id, 'caption-color-fold');
    assert.equal(page.$('background-color').parent.parent.id, 'background-fields');
    assert.equal(page.$('avoid-face').parent.parent.id, 'position-fold');
    const ids = page.$('position-fold').querySelectorAll('[id]').map((el) => el.id);
    assert.ok(ids.indexOf('avoid-face') < ids.indexOf('placements'));
  }
});

test('a custom shadow survives Off then On without snapping to Soft or Strong', async () => {
  const state = initialState();
  const fields = { shadow_opacity: .47, shadow_blur_px: 11, shadow_x_px: -2, shadow_y_px: 7 };
  Object.assign(state.style.captions, fields);
  const page = await harness(state, '#captions');
  for (const mode of ['off', 'on']) {
    await page.$('shadow-toggle').fire('click', { target: { dataset: { shadowToggle: mode } } });
    await flush();
  }
  const preview = page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style;
  for (const [key, value] of Object.entries(fields)) assert.equal(preview.captions[key], value);
});

test('invalid caption color blocks saving and a palette or reset replaces the error', async () => {
  const page = await harness(initialState(), '#captions');
  page.$('caption-color').focus();
  page.$('caption-color').value = '#oops';
  await page.$('caption-color').fire('change');
  await page.save();
  assert.equal(page.location.hash, '#captions');
  assert.equal(page.requests.some((r) => r.url === '/api/save'), false);
  await page.click('step-colors');
  await page.pick('palette-tiles', 'Newsprint');
  assert.equal(page.$('caption-color').getAttribute('aria-invalid'), null);
  await page.save();
  assert.equal(page.$('saved').hidden, false);
});

test('More lists optional choices and each screen owns its preview', async () => {
  const page = await harness(initialState(), '#more');
  assert.deepEqual(page.$('more-choices').children.map((row) => row.id),
    ['choice-grade', 'choice-opening', 'choice-cards', 'choice-broll', 'choice-moves', 'choice-moments', 'choice-sound']);
  assert.equal(page.$('more-choices').querySelectorAll('iframe').length, 0);
  await page.click('choice-opening');
  assert.equal(page.$('panel-opening').hidden, false);
  await page.pick('title-tiles', 'Typewriter');
  assert.equal(page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.tile.step, 'opening');
  await page.go('cards'); await page.pick('card-style-tiles', 'Dark');
  assert.equal(page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.tile.step, 'cards');
  assert.equal(page.$('step-count').textContent, 'Optional · 3 of 7');
});

test('every phone screen uses the same compact main preview class', async () => {
  const css = readFileSync(new URL('src/yap_studio/gallery/static/pick.css', root), 'utf8');
  const [desktop, phone] = css.split('@media (max-width: 899px) {');
  assert.doesNotMatch(desktop, /\.main-frame\.phone-compact/);
  assert.match(phone, /\.main-frame\.phone-compact \{ width: calc\(min\(180px,24svh\) \* \.5625\)/);
  assert.match(phone, /\.sticky-preview\[data-editing=true\] \.main-frame\.phone-compact \{ width: calc\(min\(210px,27svh\) \* \.5625\)/);
  assert.doesNotMatch(phone, /body\[data-screen=font\] \.main-frame/);
  for (const screen of ['looks', 'font', 'captions', 'colors', 'more']) {
    const page = await harness(initialState(), `#${screen}`, { phone: true });
    assert.equal(page.$('main-frame').classList.contains('phone-compact'), true, screen);
    assert.equal(page.$(`panel-${screen}`).hidden, false, screen);
  }
});

test('Looks has one Save control beside Customize and the selected name in the sticky bar', async () => {
  const page = await harness(initialState(), '#templates', { phone: true });
  assert.equal(page.document.querySelectorAll('button').filter((el) => el.textContent === 'Save style').length, 2);
  assert.equal(page.$('customize').hidden, false);
  assert.equal(page.$('step-navigation').hidden, true);
  assert.equal(page.$('step-header').hidden, true);
  assert.equal(page.$('look-scope').textContent, 'Sets every choice');
  assert.equal(page.$('save').parent.parent.parent, page.$('savebar'));
  assert.equal(page.$('customize').parent, page.$('save').parent);
  assert.equal(page.$('reset').hidden, true);
  assert.equal(page.$('look-selected').textContent, 'Editorial');
  await page.pick('start-options', 'SparrowLight');
  assert.equal(page.$('look-selected').textContent, 'SparrowLight');
  assert.equal(page.$('style-name').value, 'SparrowLight');
  await page.click('customize');
  assert.equal(page.$('look-selected').hidden, true);
  assert.equal(page.$('customize').hidden, true);
  assert.equal(page.$('reset').hidden, false);
  await page.click('back-to-looks');
  assert.equal(page.$('look-selected').hidden, false);
  await page.save();
  assert.equal(page.$('saved-name').textContent, 'SparrowLight');
});
test('every builder footer has Back, Next and primary Save with the same name sheet', async () => {
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#font', { phone });
    for (const step of ['font', 'captions', 'colors', 'more', 'grade', 'opening', 'cards', 'broll', 'moves', 'moments', 'sound']) {
      await page.go(step);
      assert.equal(page.$('step-navigation').hidden, false);
      assert.equal(page.$('customize').hidden, true);
      assert.equal(page.$('save').hidden, false);
      assert.equal(page.$('next').hidden, step === 'sound');
      assert.equal(page.$('save').classList.contains('primary'), true);
      assert.equal(page.$('next').classList.contains('primary'), false);
      assert.equal(page.$('back').classList.contains('primary'), false);
      assert.equal(page.$('reset').parent.parent, page.$('step-header'));
      assert.equal(page.$('save-sheet').open, false);
    }
    await page.click('save');
    assert.equal(page.$('save-sheet').open, true);
    assert.equal(page.$('style-name').value, 'My style');
    assert.equal(page.document.activeElement, page.$('style-name'));
    assert.equal(page.requests.some((r) => r.url === '/api/save'), false);
    page.$('style-name').value = 'My captions'; await page.click('cancel-save');
    assert.equal(page.$('save-sheet').open, false);
    await page.go('font'); await page.click('save');
    assert.equal(page.$('style-name').value, 'My captions');
    await page.click('confirm-save');
    assert.equal(page.$('saved-name').textContent, 'My captions');
  }
});

test('caption animation starts closed and remembers its state through edits, steps and reloads', async () => {
  const page = await harness(initialState(), '#captions');
  assert.equal(page.$('caption-animation').open, false);
  assert.equal(page.$('caption-animation').querySelectorAll('.tile').length, 28);
  assert.equal(page.$('next').parent.parent.parent.parent, page.$('savebar'));
  page.$('caption-animation').open = true;
  await page.$('caption-animation').fire('toggle');
  await page.pick('highlight-tiles', 'Neon glow');
  assert.equal(page.$('caption-animation').open, true);
  await page.click('step-colors');
  await page.click('step-captions');
  assert.equal(page.$('caption-animation').open, true);
  const reload = await harness(page.state, '#captions', { storage: page.storage });
  assert.equal(reload.$('caption-animation').open, true);
  reload.$('caption-animation').open = false;
  await reload.$('caption-animation').fire('toggle');
  const closed = await harness(page.state, '#captions', { storage: page.storage });
  assert.equal(closed.$('caption-animation').open, false);
});

test('a taken name asks before Update and keeps one library entry selected', async () => {
  const page = await harness(initialState(), '#font');
  await page.click('save');
  const taken = initialState().library[0].name;
  page.$('style-name').value = `  ${taken.toUpperCase()}  `;
  await page.click('confirm-save');
  assert.equal(page.requests.filter((r) => r.url === '/api/save').length, 0);
  assert.equal(page.$('save-sheet').open, true);
  assert.equal(page.$('save-sheet-status').textContent,
    `You already have a style called “${taken}”. Update it, or choose another name.`);
  assert.equal(page.$('confirm-save').textContent, `Update ${taken}`);
  assert.equal(page.$('save-sheet-error').textContent, '');
  assert.equal(page.$('save-sheet-status').getAttribute('role'), 'status');
  assert.equal(page.$('save-sheet-status').classList.contains('field-hint'), true);
  assert.ok(page.$('style-name').getAttribute('aria-describedby').includes('save-sheet-status'));
  await page.$('style-name').fire('keydown', { key: 'Enter', preventDefault() {} }); await flush();
  assert.equal(page.requests.find((r) => r.url === '/api/save').data.replace, 'learned');
  assert.equal(page.library().length, 1);
  assert.equal(page.library()[0].id, 'learned');
  assert.equal(page.baseKey(), 'library:learned');
  assert.equal(page.$('save-sheet').open, false);
});

test('editing a taken name clears Update and Cancel keeps the sheet usable', async () => {
  const page = await harness(initialState());
  const taken = initialState().library[0].name;
  await page.click('save'); page.$('style-name').value = taken; await page.click('confirm-save');
  assert.equal(page.$('confirm-save').textContent, `Update ${taken}`);
  page.$('style-name').value = 'Sunday'; await page.$('style-name').fire('input');
  assert.equal(page.$('confirm-save').textContent, 'Save style');
  assert.equal(page.$('save-sheet-error').textContent, '');
  assert.equal(page.$('save-sheet-status').textContent, '');
  await page.click('cancel-save'); await page.click('save'); await page.click('confirm-save');
  assert.equal(page.requests.find((r) => r.url === '/api/save').data.replace, undefined);
  assert.equal(page.$('saved-name').textContent, 'Sunday');
});

test('the Update label shortens a long style name while the question keeps it', async () => {
  const state = initialState(), name = 'Sunday captions '.repeat(6).trim();
  state.library[0].name = name;
  const page = await harness(state);
  await page.click('save'); page.$('style-name').value = name; await page.click('confirm-save');
  assert.ok(page.$('save-sheet-status').textContent.includes(`“${name}”`));
  assert.ok(page.$('confirm-save').textContent.length <= 39);
  assert.ok(page.$('confirm-save').textContent.endsWith('…'));
  assert.equal(page.requests.some((r) => r.url === '/api/save'), false);
});

test('a stale library 409 asks before replacing the server entry', async () => {
  const page = await harness(initialState(), '', { nameConflict: true });
  await page.click('save'); page.$('style-name').value = 'Sunday'; await page.click('confirm-save');
  assert.equal(page.$('save-sheet').open, true);
  assert.equal(page.$('confirm-save').textContent, 'Update Sunday');
  assert.equal(page.$('save-sheet-error').textContent, '');
  assert.match(page.$('save-sheet-status').textContent, /already have a style called/);
  await page.click('confirm-save');
  assert.equal(page.requests.filter((r) => r.url === '/api/save').at(-1).data.replace, 'stale');
  assert.equal(page.library().filter((item) => item.id === 'stale').length, 1);
});

test('a failed save stays in the name sheet and can retry with the chosen name', async () => {
  const page = await harness(initialState(), '#font', { failSave: true });
  await page.click('save');
  page.$('style-name').value = 'Sunday captions';
  await page.click('confirm-save');
  assert.equal(page.$('save-sheet').open, true);
  assert.equal(page.$('saved').hidden, true);
  assert.equal(page.$('confirm-save').disabled, false);
  assert.equal(page.$('save-sheet-error').textContent, 'Could not save. Try again.');
  assert.equal(page.$('save-sheet-status').textContent, '');
  assert.equal(page.$('style-name').value, 'Sunday captions');
  await page.click('confirm-save');
  assert.equal(page.$('saved-name').textContent, 'Sunday captions');
  assert.equal(page.$('save-sheet').open, false);
});

test('the compact phone preview enlarges on tap and changes hold a caption-bearing engine frame', async () => {
  const page = await harness(initialState(), '#font', { phone: true, layers: true });
  assert.equal(page.$('main-frame').parent.classList.contains('sticky-preview'), true);
  await page.click('main-frame');
  assert.equal(page.$('main-frame').getAttribute('aria-expanded'), 'true');
  assert.equal(page.$('main-frame').classList.contains('phone-expanded'), true);
  await page.click('main-frame');
  assert.equal(page.$('main-frame').getAttribute('aria-expanded'), 'false');
  await page.pick('font-options', 'Instrument Serif');
  const frames = page.$('main-frame').querySelectorAll('iframe');
  for (const frame of frames) await frame.fire('load');
  await flush();
  assert.ok(frames.length > 0);
  assert.ok(frames.every((frame) => frame.contentWindow.__player.getTime() === .65));
  assert.ok(frames.every((frame) => !frame.contentWindow.__player.isPlaying()));
  assert.equal(page.$('play').textContent, 'Play');
  await page.click('play');
  assert.ok(frames.every((frame) => frame.contentWindow.__player.isPlaying()));
});

test('late caption speech holds the opening title at a shared visible poster time', async () => {
  const state = initialState();
  state.preview.words = [{ text: 'Hello', start: 6, end: 6.5 }];
  const page = await harness(state, '#font', { layers: true, captionPoster: 6.25, captionDuration: 8 });
  await page.pick('font-options', 'Instrument Serif');
  const frames = page.$('main-frame').querySelectorAll('iframe');
  assert.equal(frames.length, 2);
  for (const frame of frames) await frame.fire('load');
  await flush();
  assert.ok(frames.every((frame) => frame.contentWindow.__player.getTime() === 1.5));
  assert.ok(frames.every((frame) => !frame.contentWindow.__player.isPlaying()));
});

test('music slider updates and saves volume without replacing tiles or preparing layers', async () => {
  const page = await harness(initialState(), '#sound', { layers: true });
  page.observers[1].intersect();
  await flush();
  const titles = [...page.$('title-tiles').children];
  const cards = [...page.$('card-style-tiles').children];
  const main = [...page.$('main-frame').children];
  const musicTiles = [...page.$('music-level-tiles').children];
  const count = page.requests.filter((request) => request.url === '/api/layers').length;
  const playText = page.$('play').textContent;
  for (const level of [0, .01, .06, .1, .14, .19, .25, .31, .35, .4]) {
    page.$('music').value = String(level);
    await page.$('music').fire('input');
  }
  await flush();
  assert.equal(page.$('music-value').textContent, '40%');
  assert.equal(page.$('change-count').textContent, 'Changed Sound from Editorial');
  assert.equal(page.$('reset').disabled, false);
  assert.deepEqual(page.$('title-tiles').children, titles);
  assert.deepEqual(page.$('card-style-tiles').children, cards);
  assert.deepEqual(page.$('main-frame').children, main);
  assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, count);
  assert.equal(page.$('play').textContent, playText);
  assert.deepEqual(page.$('music-level-tiles').children, musicTiles);
  await page.save();
  assert.equal(page.requests.find((request) => request.url === '/api/save').data.style.music.level, .4);
});

test('face preference explains missing geometry and the still-based preview limitation', async () => {
  const page = await harness(initialState(), '#captions');
  assert.match(page.$('placement-note').textContent, /unavailable for this preview/);
  const state = initialState();
  state.preview.face_boxes = [{ x: 300, y: 100, w: 360, h: 400 }];
  const measured = await harness(state, '#captions');
  assert.match(measured.$('placement-note').textContent, /faces measured in the still/);
  assert.match(measured.$('placement-note').textContent, /throughout each phrase/);
  measured.$('avoid-face').checked = false;
  await measured.$('avoid-face').fire('change');
  await flush();
  assert.equal(measured.$('placement-note').textContent, 'Captions stay at your chosen position.');
});

test('every tile has a caption poster before preparation, throughout loading and after an engine error', async () => {
  const page = await harness(initialState(), '#captions', { layers: true });
  const highlights = page.$('highlight-tiles').children.map((tile) => tile.children[0]);
  assert.equal(highlights.length, 24);
  for (const host of highlights) {
    const poster = host.querySelectorAll('.tile-poster')[0];
    assert.ok(poster);
    assert.equal(poster.hidden, false);
    assert.equal(poster.querySelectorAll('img')[0].src, '/still?g=0&t=token');
    assert.match(poster.textContent, /Your/);
    assert.equal(host.querySelectorAll('iframe').length, 0);
  }
  for (const id of ['caption-animation', 'highlight-fold']) { page.$(id).open = true; await page.$(id).fire('toggle'); }
  page.observers[1].intersect();
  await flush();
  const host = highlights[0], poster = host.querySelectorAll('.tile-poster')[0];
  assert.equal(poster.hidden, false);
  assert.equal(host.querySelectorAll('.stage')[0].style.visibility, 'hidden');
  const frame = host.querySelectorAll('iframe')[0];
  await frame.fire('load'); await flush();
  assert.equal(poster.hidden, true);
  assert.equal(frame._galleryReady, true);
  assert.equal(host.querySelectorAll('.stage')[0].style.visibility, 'visible');
  const failed = await harness(initialState(), '#captions', { failTiles: true });
  failed.observers[1].intersect(); await flush();
  const failedHost = failed.$('caption-look-tiles').children[0].children[0];
  assert.equal(failedHost.querySelectorAll('.tile-poster')[0].hidden, false);
  assert.match(failedHost.textContent, /Your/);
  assert.equal(failedHost.getAttribute('title'), 'Engine unavailable');
});

test('automatic contrast sits immediately under Caption color with its chosen value and no position pixels', async () => {
  const page = await harness(initialState(), '#captions');
  await page.pick('background-tiles', 'Box');
  await page.pick('background-swatches', 'White');
  assert.match(page.$('caption-result').textContent, /Text switches to dark.*#111111/);
  assert.equal(page.$('caption-result').hidden, false);
  const children = page.$('caption-color-fold').children;
  assert.equal(children.indexOf(page.$('caption-result')), children.indexOf(page.$('caption-color').parent) + 1);
  assert.equal(page.$('caption-result').querySelectorAll('.color-dot')[0].style.backgroundColor, '#111111');
  assert.equal(page.$('caption-position-field').querySelectorAll('output').length, 0);
  assert.equal(page.$('fine-top_px').parent.parent, page.$('fine-fields'));
  page.$('caption-position').value = 720;
  await page.$('caption-position').fire('input'); await flush();
  assert.equal(page.$('fine-top_px').value, 720);
  assert.equal(page.requests.filter((r) => r.data?.tile.tab === 'preview').at(-1).data.style.captions.top_px, 720);
  await page.pick('background-swatches', 'Black');
  assert.match(page.$('caption-result').textContent, /Text switches to light.*#FFFFFF/);
  await page.pick('background-tiles', 'None');
  assert.equal(page.$('caption-result').hidden, true);
});


async function openFold(page, id) {
  const folds = [];
  for (let parent = page.$(id); parent; parent = parent.parentElement) if (parent.tagName === 'details') folds.unshift(parent);
  for (const fold of folds) if (!fold.open) { fold.open = true; await fold.fire('toggle'); }
  await flush();
}

test('Cards Flow look tiles request flow and choose Tags', async () => {
  const page = await harness();
  await page.go('cards');
  await openFold(page, 'slot-fold-cards-flow-look');
  for (const id of ['boxes', 'tags']) {
    const tile = page.$(`slot-cards-${id}`);
    assert.ok(tile.querySelectorAll('.tile-frame').length);
    assert.equal(tile.getAttribute('aria-pressed'), String(id === 'boxes'));
  }
  for (const observer of page.observers) observer.intersect();
  await flush();
  const requests = page.requests.filter((r) => r.data?.tile?.card === 'flow');
  assert.ok(requests.some((r) => r.data.style.graphics.flow_look === 'boxes'));
  assert.ok(requests.some((r) => r.data.style.graphics.flow_look === 'tags'));
  assert.ok(requests.every((r) => r.data.tile.tab === 'cards'));
  await page.click('slot-cards-tags');
  assert.equal(page.$('slot-cards-tags').getAttribute('aria-pressed'), 'true');
  assert.ok(page.$('screen-summary').textContent.endsWith(' · flow tags'));
  await page.save();
  assert.equal(page.state.style.graphics.flow_look, 'tags');
});
async function pressControl(page, id, index) {
  await openFold(page, id);
  const button = page.$(id).children[index];
  if (button.events.click?.length) await button.click();
  else await page.$(id).fire('click', { target: button });
  await flush();
}
function withFields(source, fields) {
  const result = structuredClone(source);
  for (const [path, value] of Object.entries(fields)) {
    const parts = path.split('.'), key = parts.pop();
    const section = parts.reduce((node, name) => node[name] ??= {}, result);
    section[key] = value;
  }
  return result;
}
const savePayload = (page) => page.requests.filter((request) => request.url === '/api/save').at(-1)?.data.style;

test('Giant is a main caption look with a live candidate and is absent from Highlight', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const page = await harness(state, '#captions', { layers: true });
  assert.match(page.$('slot-captions-giant').textContent, /Giant.*Big words in a solid block/);
  assert.equal(page.$('highlight-tiles').textContent.includes('Giant'), false);
  page.observers[1].intersect(); await flush();
  assert.ok(page.requests.some((request) => request.data?.tile.tab === 'captions'
    && request.data.style.captions.highlight_style === 'giant'));
  await page.click('slot-captions-giant');
  assert.equal(page.$('slot-captions-giant').getAttribute('aria-pressed'), 'true');
  assert.equal(page.$('caption-look-tiles').children.filter((tile) => tile.getAttribute('aria-pressed') === 'true').length, 1);
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'captions.highlight_style': 'giant' }));
  await page.pick('caption-look-tiles', 'Plain'); await page.save();
  assert.equal(savePayload(page).captions.highlight_style, 'none');
});

test('Giant and 3D camera press one tile and switching back preserves Giant fields', async () => {
  const state = initialState();
  Object.assign(state.style.captions, { entrance: 'rise', size_px: 83,
    max_words_per_phrase: 5, max_characters_per_phrase: 31,
    background_mode: 'box', background_opacity: .21, background_color: '#123456' });
  const original = structuredClone(state.style);
  const page = await harness(state, '#captions', { layers: true });
  for (const look of ['giant', 'camera_3d', 'giant']) {
    await page.click(`slot-captions-${look}`); await page.save();
    const pressed = page.$('caption-look-tiles').children.filter((tile) => tile.getAttribute('aria-pressed') === 'true');
    assert.deepEqual(pressed.map((tile) => tile.id), [`slot-captions-${look}`]);
    assert.deepEqual(savePayload(page), withFields(original, {
      'captions.camera_3d': look === 'camera_3d',
      'captions.highlight_style': look === 'giant' ? 'giant' : 'none',
    }));
  }
});

test('Kinetic hook is a main opening tile and keeps the base title first without duplicates', async () => {
  for (const base of ['quiet_serif', 'kinetic']) {
    const state = initialState(); state.style.hook.style = base;
    const original = structuredClone(state.style), page = await harness(state, '#opening', { layers: true });
    const names = page.$('title-tiles').children.map((tile) => tile.children[1].text);
    assert.equal(names[0], choices.labels[base]);
    assert.equal(names.filter((name) => name === 'Kinetic hook').length, 1);
    assert.equal(page.$('more-title-tiles').textContent.includes('Kinetic hook'), false);
    const all = [...names, ...page.$('more-title-tiles').children.map((tile) => tile.children[1].text)];
    assert.equal(all.length, schema.properties.hook.properties.style.enum.length - 2);
    assert.equal(new Set(all).size, all.length);
    page.observers[1].intersect(); await flush();
    assert.ok(page.requests.some((request) => request.data?.tile.tab === 'title' && request.data.style.hook.style === 'kinetic'));
    await page.pick('title-tiles', 'Kinetic hook'); await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'hook.style': 'kinetic' }));
  }
});

test('Chapter cards shows a live candidate preview without turning the switch on', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const page = await harness(state, '#opening', { layers: true });
  const fold = 'slot-fold-opening-chapter';
  assert.equal(page.$(fold).querySelectorAll('.slot-switch-preview').length, 1);
  assert.equal(page.$('slot-switch-opening-chapter').checked, false);
  page.observers[1].intersect(); await flush();
  assert.equal(page.requests.some((request) => request.data?.tile.tab === 'chapter'), false);
  await openFold(page, fold);
  page.observers[1].intersect(); await flush();
  const request = page.requests.find((request) => request.data?.tile.tab === 'chapter');
  assert.ok(request);
  assert.deepEqual(request.data.style, withFields(original, { 'chapters.style': 'title_card' }));
  assert.equal(page.$(fold).querySelectorAll('iframe').length, 1);
  assert.equal(page.$('slot-switch-opening-chapter').checked, false);
  await page.save(); assert.deepEqual(savePayload(page), original);
});

test('Opening rebuilds keep the phone scroll position when Chapter cards changes', async () => {
  const page = await harness(initialState(), '#opening', { phone: true });
  await openFold(page, 'slot-fold-opening-chapter');
  const tiles = page.$('title-tiles'), replace = tiles.replaceChildren.bind(tiles);
  tiles.replaceChildren = (...children) => {
    replace(...children);
    page.window.scrollY = 0;
  };
  page.window.scrollTo = (_x, y) => { page.window.scrollY = y; };
  for (const on of [true, false]) {
    page.window.scrollY = 2200;
    const toggle = page.$('slot-switch-opening-chapter');
    toggle.checked = on;
    await toggle.fire('change'); await flush();
    assert.equal(page.window.scrollY, 2200);
    assert.equal(page.$('slot-switch-opening-chapter').checked, on);
  }
});

test('Chapter cards switches on and off, preserves other fields and saves the exact style', async () => {
  for (const chapters of [undefined, { style: 'none', hold_s: 2.7 }]) {
    const state = initialState(); state.from_library = true;
    if (chapters) state.style.chapters = chapters; else delete state.style.chapters;
    const original = structuredClone(state.style), page = await harness(state, '#opening');
    const id = 'slot-switch-opening-chapter', fold = 'slot-fold-opening-chapter';
    assert.equal(page.$(id).checked, false);
    assert.equal(page.$(fold + '-value').textContent, 'Off');
    assert.equal(page.$('screen-summary').textContent.includes('chapter cards on'), false);
    await page.save(); assert.deepEqual(savePayload(page), original);
    for (const on of [true, false]) {
      await openFold(page, id);
      page.$(id).checked = on; await page.$(id).fire('change'); await flush();
      assert.equal(page.$(fold + '-value').textContent, on ? 'On' : 'Off');
      assert.equal(page.$('screen-summary').textContent.includes('chapter cards on'), on);
      assert.match(page.$('change-count').textContent, /Changed Opening/);
      await page.save();
      assert.deepEqual(savePayload(page), withFields(original, { 'chapters.style': on ? 'title_card' : 'none' }));
    }
  }
});

test('Words includes Gravity drop off by default and marks the group new', async () => {
  const state = initialState(); state.from_library = true;
  const original = structuredClone(state.style), page = await harness(state, '#cards');
  const words = page.$('card-group-words');
  assert.equal(words.querySelectorAll('.new-tag').length, 1);
  await openFold(page, 'card-group-words');
  assert.match(words.textContent, /Gravity drop/);
  assert.equal(page.$('kind-gravity_drop').getAttribute('aria-label'), 'Gravity drop on');
  assert.equal(page.$('kind-gravity_drop').checked, false);
  await page.save(); assert.deepEqual(savePayload(page), original);
  page.$('kind-gravity_drop').checked = true; await page.$('kind-gravity_drop').fire('change'); await flush();
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'graphics.card_kinds_on': ['gravity_drop'] }));
});

test('every optional row opens exactly one screen with More selected and sequential headers', async () => {
  const page = await harness(initialState(), '#more');
  const ids = ['grade', 'opening', 'cards', 'broll', 'moves', 'moments', 'sound'];
  assert.equal(page.$('step-count').textContent, 'Optional · 7 choices');
  for (const [index, id] of ids.entries()) {
    await page.go(id);
    assert.deepEqual(page.document.querySelectorAll('.step-panel').filter((panel) => !panel.hidden).map((panel) => panel.id), [`panel-${id}`]);
    assert.equal(page.$('step-more').getAttribute('aria-selected'), 'true');
    assert.equal(page.$('step-count').textContent, `Optional · ${index + 1} of 7`);
    assert.match(page.$('screen-summary').textContent, new RegExp('^' + choices.screens.find((screen) => screen.id === id).name + ':'));
    await page.click(`${id}-all-choices`);
    assert.equal(page.$('panel-more').hidden, false);
  }
  assert.deepEqual(page.$('more-choices').children.map((row) => row.querySelectorAll('.choice-number')[0].textContent), ['5', '6', '7', '8', '9', '10', '11']);
});

test('screen visits and opened folds save the whole original style for the base and Marker', async () => {
  for (const state of [initialState(), markerState()]) {
    const original = structuredClone(state.style), page = await harness(state);
    for (const id of ['font', 'captions', 'colors', 'more', 'grade', 'opening', 'cards', 'broll', 'moves', 'moments', 'sound']) {
      await page.go(id);
      for (const fold of page.$(`panel-${id}`).querySelectorAll('details')) await openFold(page, fold.id);
    }
    await page.save();
    assert.deepEqual(savePayload(page), original);
    assert.equal(JSON.stringify(savePayload(page)), JSON.stringify(original));
  }
});

test('Back and Next skip hidden screens, and the four tabs use their own keyboard order', async () => {
  const page = await harness(initialState(), '#font');
  for (const id of ['font', 'captions', 'colors', 'top-text', 'more', 'grade', 'opening', 'cards', 'broll', 'moves', 'moments', 'sound']) {
    assert.equal(page.document.body.dataset.screen, id);
    assert.equal(page.$('next').hidden, id === 'sound');
    if (id !== 'sound') await page.click('next');
  }
  await page.click('back'); assert.equal(page.location.hash, '#moments');
  await page.$('step-more').fire('keydown', { key: 'ArrowLeft', preventDefault() {} }); await flush();
  assert.equal(page.location.hash, '#top-text');
  assert.equal(page.document.activeElement.id, 'step-top-text');
  await page.$('step-colors').fire('keydown', { key: 'End', preventDefault() {} }); await flush();
  assert.equal(page.location.hash, '#more');
  await page.$('step-more').fire('keydown', { key: 'Home', preventDefault() {} }); await flush();
  assert.equal(page.location.hash, '#font');
  await page.click('back'); assert.equal(page.location.hash, '#looks');
  for (const [hash, screen] of Object.entries({ start: 'looks', templates: 'looks', words: 'captions',
    highlight: 'captions', title: 'opening', cards: 'cards', moves: 'moves', moments: 'moments', unknown: 'looks', constructor: 'looks' })) {
    const reload = await harness(initialState(), '#' + hash);
    assert.equal(reload.document.body.dataset.screen, screen);
  }
});

test('older styles display optional defaults without adding them to the saved payload', async () => {
  const state = initialState();
  delete state.style.broll;
  delete state.style.voice;
  delete state.style.grade;
  delete state.style.music.sound_effects;
  delete state.style.music.sound_effects_fallback;
  delete state.style.music.beat_snap;
  const original = structuredClone(state.style), page = await harness(state);
  for (const screen of ['font', 'captions', 'colors', 'more', 'grade', 'opening', 'cards', 'broll', 'moves', 'moments', 'sound']) await page.go(screen);
  assert.equal(page.$('voice-preset').children[0].getAttribute('aria-pressed'), 'true');
  await page.save();
  assert.deepEqual(savePayload(page), original);
});

test('empty or unsupported Moves slots hide the row, route and next step', async () => {
  const entries = choices.movesSlots.splice(0);
  try {
    const page = await harness(initialState(), '#moves');
    assert.equal(page.document.body.dataset.screen, 'more');
    assert.equal(page.$('choice-moves'), undefined);
    await page.go('broll'); await page.click('next');
    assert.equal(page.location.hash, '#moments');
    choices.movesSlots.push({ ...entries[0], requires: ['moves.missing', 'none'] });
    const unsupported = await harness(initialState(), '#moves');
    assert.equal(unsupported.document.body.dataset.screen, 'more');
    assert.equal(unsupported.$('choice-moves'), undefined);
  } finally { choices.movesSlots.splice(0, choices.movesSlots.length, ...entries); }
  const restored = await harness(initialState(), '#moves');
  assert.equal(restored.document.body.dataset.screen, 'moves');
  await restored.go('more');
  assert.ok(restored.$('choice-moves'));
});


test('each Moves tile saves only its zoom mode and each strength saves only its scale', async () => {
  for (const entry of choices.movesSlots.filter((entry) => !entry.type)) {
    const state = initialState(); state.style.moves = { zoom_mode: 'punches_and_push', punch_scale: 2 };
    const original = structuredClone(state.style), page = await harness(state, '#moves');
    await page.click(`slot-moves-${entry.id}`); await flush();
    assert.equal(page.$('moves-more').hidden, false);
    assert.equal(Boolean(page.$('slot-fold-moves-zoom-strength')), entry.id !== 'none');
    assert.equal(page.$(`slot-moves-${entry.id}`).getAttribute('aria-pressed'), 'true');
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'moves.zoom_mode': entry.id }));
  }
  for (const mode of ['punches', 'punches_and_push']) for (const scale of [1.5, 2]) {
    const state = initialState(); state.style.moves = { zoom_mode: mode, punch_scale: scale === 2 ? 1.5 : 2 };
    const original = structuredClone(state.style), page = await harness(state, '#moves');
    assert.equal(page.$('moves-more').open, false);
    assert.equal(page.$('slot-fold-moves-zoom-strength').open, false);
    await openFold(page, 'slot-fold-moves-zoom-strength');
    await page.click(`slot-moves-punch-${scale}`); await flush();
    assert.equal(page.$('slot-fold-moves-zoom-strength').open, true);
    assert.equal(page.$(`slot-moves-punch-${scale}`).getAttribute('aria-pressed'), 'true');
    assert.ok(page.$('screen-summary').textContent.endsWith(`at ${scale}x · all transitions`));
    assert.match(page.$('change-count').textContent, /Changed Moves/);
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'moves.punch_scale': scale }));
  }
});

test('Jump cuts in More save only their choice and keep the zoom mode', async () => {
  const entries = [['hard', 'Hard cut'], ['punch', 'Punch'], ['smooth', 'Smooth (punch on big moves)'], ['step_in', 'Step in']];
  for (const zoom_mode of ['none', 'punches', 'punches_and_push']) for (const [mode, name] of entries) {
    const state = initialState(); state.style.moves.zoom_mode = zoom_mode;
    const original = structuredClone(state.style), page = await harness(state, '#moves');
    assert.equal(page.$('moves-more').open, false);
    assert.equal(page.$('moves-more-hint').textContent,
      (zoom_mode === 'none' ? '' : 'Zoom strength, ') + 'Transitions, Jump cuts, Push in on the first line');
    const fold = page.$('slot-fold-moves-jump-cuts');
    assert.ok(fold); assert.equal(fold.open, false);
    assert.equal(fold.querySelectorAll('.fold-value')[0].textContent, 'Hard cut');
    await openFold(page, fold.id);
    assert.equal(page.$('slot-moves-hard').getAttribute('aria-pressed'), 'true');
    assert.equal(fold.querySelectorAll('.field-hint')[0].textContent,
      'What happens where a pause or a retake was cut out.');
    await page.click(`slot-moves-${mode}`); await flush();
    assert.equal(page.$(fold.id).open, true);
    assert.equal(page.$(fold.id).querySelectorAll('.fold-value')[0].textContent, name);
    assert.equal(page.$(`slot-moves-${mode}`).getAttribute('aria-pressed'), 'true');
    const summary = choices.screenSummary('moves', withFields(original, { 'moves.jump_cuts': mode }));
    assert.equal(page.$('screen-summary').textContent, `Moves: ${summary}`);
    if (mode !== 'hard') assert.match(page.$('change-count').textContent, /Changed Moves/);
    await page.go('more'); assert.ok(page.$('choice-moves').textContent.includes(summary));
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'moves.jump_cuts': mode }));
  }
});

test('each transition switch writes only its array and restores schema order after off then on', async () => {
  const order = schema.properties.moves.properties.transitions.items.enum;
  for (const entry of choices.transitionLooks) for (const on of [false, true]) {
    const state = initialState();
    state.style.moves.transitions = on ? order.filter((id) => id !== entry.id) : [...order];
    const original = structuredClone(state.style), page = await harness(state, '#moves');
    assert.equal(page.$('moves-more').hidden, false);
    assert.equal(page.$('slot-fold-moves-zoom-strength'), undefined);
    assert.equal(page.$('slot-fold-moves-transitions').open, false);
    await openFold(page, 'slot-fold-moves-transitions');
    const toggle = page.$(`slot-switch-moves-${entry.id}`);
    assert.equal(toggle.checked, !on);
    toggle.checked = on; await toggle.fire('change'); await flush();
    assert.equal(page.$('slot-fold-moves-transitions').open, true);
    assert.match(page.$('change-count').textContent, /Changed Moves/);
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'moves.transitions': on ? order : order.filter((id) => id !== entry.id) }));
  }
  const state = initialState(), original = structuredClone(state.style), page = await harness(state, '#moves');
  await openFold(page, 'slot-fold-moves-transitions');
  for (const entry of choices.transitionLooks) {
    const toggle = page.$(`slot-switch-moves-${entry.id}`); toggle.checked = false;
    await toggle.fire('change'); await flush();
  }
  assert.equal(page.$('slot-fold-moves-transitions-value').textContent, 'Off');
  assert.equal(page.$('screen-summary').textContent, 'Moves: No zooms · no transitions');
  await page.go('more');
  assert.ok(page.$('choice-moves').textContent.includes('No zooms · no transitions'));
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'moves.transitions': [] }));
});

test('turning each transition off and back on restores the whole original payload', async () => {
  const state = initialState(), original = structuredClone(state.style), page = await harness(state, '#moves');
  await openFold(page, 'slot-fold-moves-transitions');
  for (const entry of choices.transitionLooks) {
    for (const on of [false, true]) {
      const toggle = page.$(`slot-switch-moves-${entry.id}`); toggle.checked = on;
      await toggle.fire('change'); await flush();
    }
    assert.equal(page.$('slot-fold-moves-transitions-value').textContent, 'All three');
  }
  await page.save();
  assert.deepEqual(savePayload(page), original);
});

test('transition visits preserve absent and explicit arrays while showing effective defaults', async () => {
  const order = schema.properties.moves.properties.transitions.items.enum;
  for (const transitions of [undefined, [], order, ['whip_pan', 'cinematic_zoom'], ['light_leak', 'whip_pan']]) {
    const state = initialState();
    if (transitions === undefined) delete state.style.moves.transitions; else state.style.moves.transitions = transitions;
    const original = structuredClone(state.style), page = await harness(state, '#moves');
    await openFold(page, 'slot-fold-moves-transitions');
    assert.deepEqual(choices.transitionLooks.filter((entry) => page.$(`slot-switch-moves-${entry.id}`).checked).map((entry) => entry.id),
      order.filter((id) => (transitions ?? order).includes(id)));
    const value = transitions === undefined || transitions.length === 3 ? 'All three' : !transitions.length ? 'Off'
      : transitions.includes('cinematic_zoom') ? 'Whip and Zoom' : 'Whip and Light leak';
    assert.equal(page.$('slot-fold-moves-transitions-value').textContent, value);
    await page.go('more'); await page.go('moves'); await page.save();
    assert.deepEqual(savePayload(page), original);
  }
});

test('transition previews use the still and stand-in shot and play only on hover or half visibility', async () => {
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#moves', { phone });
    const fold = page.$('slot-fold-moves-transitions');
    assert.equal(fold.querySelectorAll('iframe').length, 0);
    assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, 0);
    const hosts = fold.querySelectorAll('.stage-host');
    assert.equal(hosts.length, 3);
    for (const [index, host] of hosts.entries()) {
      assert.equal(host.querySelectorAll('.transition-treatment')[0].classList.contains(choices.transitionLooks[index].preview), true);
      assert.ok(host.querySelectorAll('.transition-from')[0].querySelectorAll('img')[0].src.startsWith('/still?'));
      assert.equal(host.querySelectorAll('.transition-to')[0].querySelectorAll('img')[0].src, '/static/shot.jpg');
      assert.equal(host._drawn.element.classList.contains('is-playing'), false);
    }
    await openFold(page, fold.id);
    const host = hosts[0];
    if (phone) { page.observers[2].intersect(.49); assert.equal(host._drawn.element.classList.contains('is-playing'), false); page.observers[2].intersect(.5); }
    else await host.fire('mouseenter');
    assert.equal(host._drawn.element.classList.contains('is-playing'), true);
    if (phone) page.observers[2].intersect(0); else await host.fire('mouseleave');
    assert.equal(host._drawn.element.classList.contains('is-playing'), false);
    await host.fire('mouseenter');
    fold.open = false; await fold.fire('toggle');
    assert.equal(host._drawn.element.classList.contains('is-playing'), false);
  }
  const reduced = await harness(initialState(), '#moves', { reducedMotion: true });
  await openFold(reduced, 'slot-fold-moves-transitions');
  const host = reduced.$('slot-fold-moves-transitions').querySelectorAll('.stage-host')[0];
  await host.fire('mouseenter');
  assert.equal(host._drawn.element.classList.contains('is-playing'), false);
});

test('Moves visits keep absent and explicit fields and No zooms preserves strength', async () => {
  for (const moves of [undefined, { zoom_mode: 'none', punch_scale: 2 }, { zoom_mode: 'punches', punch_scale: 1.5 },
    { zoom_mode: 'punches_and_push', punch_scale: 2 }]) {
    const state = initialState();
    if (moves) state.style.moves = moves; else delete state.style.moves;
    const original = structuredClone(state.style), page = await harness(state, '#moves');
    assert.equal(page.$('moves-more').hidden, false);
    await openFold(page, 'slot-fold-moves-transitions');
    await openFold(page, 'slot-fold-moves-jump-cuts');
    if (moves && moves.zoom_mode !== 'none') await openFold(page, 'slot-fold-moves-zoom-strength');
    await page.go('more'); await page.go('moves'); await page.save();
    assert.deepEqual(savePayload(page), original);
  }
  const state = initialState(); state.style.moves = { zoom_mode: 'punches', punch_scale: 2 };
  const original = structuredClone(state.style);
  const page = await harness(state, '#moves');
  await page.click('slot-moves-none'); await flush();
  assert.equal(page.$('slot-fold-moves-zoom-strength'), undefined);
  assert.equal(page.$('moves-more').hidden, false);
  await page.click('slot-moves-punches_and_push'); await flush();
  assert.ok(page.$('slot-fold-moves-zoom-strength').textContent.includes('2x'));
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'moves.zoom_mode': 'punches_and_push' }));
});

test('Moves draw only the picture toward the first face and leave captions outside the zoom', async () => {
  const state = initialState(); state.preview.face_boxes = [{ x: 216, y: 384, w: 216, h: 384 }];
  state.style.moves.punch_scale = 2;
  const page = await harness(state, '#moves');
  assert.equal(page.$('track-row').hidden, true);
  assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, 0);
  for (const entry of choices.movesSlots.filter((entry) => !entry.type)) {
    const tile = page.$(`slot-moves-${entry.id}`), zoom = tile.querySelectorAll('.zoom-picture');
    assert.equal(zoom.length, entry.id === 'none' ? 0 : 1);
    assert.equal(tile.querySelectorAll('.drawn-still').length, 1);
    assert.equal(tile.querySelectorAll('.poster-caption').length, 1);
    if (entry.id === 'none') continue;
    assert.equal(zoom[0].style.transformOrigin, '30% 30%');
    assert.equal(zoom[0].style['--punch-scale'], '2');
    assert.equal(zoom[0].querySelectorAll('.poster-caption').length, 0);
    assert.equal(tile.querySelectorAll('.zoom-tag')[0].textContent, '2x');
  }
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#moves', { phone });
    const host = page.$('slot-moves-punches').children[0], drawing = host.children[0];
    assert.equal(host.querySelectorAll('.zoom-picture')[0].style.transformOrigin, '50% 42%');
    assert.equal(drawing.classList.contains('is-playing'), false);
    if (phone) { page.observers[2].intersect(.49); assert.equal(drawing.classList.contains('is-playing'), false); page.observers[2].intersect(.5); }
    else await host.fire('mouseenter');
    assert.equal(drawing.classList.contains('is-playing'), true);
    if (phone) page.observers[2].intersect(0); else await host.fire('mouseleave');
    assert.equal(drawing.classList.contains('is-playing'), false);
    await page.click('slot-moves-punches'); await flush();
    assert.equal(page.$('main-frame').children[0].classList.contains('is-playing'), true);
  }
  const reduced = await harness(initialState(), '#moves', { reducedMotion: true });
  await reduced.click('slot-moves-punches'); await flush();
  assert.equal(reduced.$('main-frame').children[0].classList.contains('is-playing'), false);
});

test('B-roll reads the style motion and uses the schema default only for display', async () => {
  for (const motion of [undefined, 'none', 'drift']) {
    const state = initialState();
    if (motion === undefined) delete state.style.broll.overlay_motion; else state.style.broll.overlay_motion = motion;
    const original = structuredClone(state.style), page = await harness(state, '#broll');
    const box = page.$('broll-layout-tiles').children[2].querySelectorAll('.shot-box')[0];
    assert.equal(box.classList.contains('drift'), motion === 'drift');
    await page.save(); assert.deepEqual(savePayload(page), original);
  }
});

test('Clone flash is blocked on a keyed preview without changing the saved style', async () => {
  const state = initialState();
  state.preview = { ...state.preview, screen: 'green', screen_checked: true };
  const original = structuredClone(state.style), page = await harness(state, '#moments');
  const reason = 'Needs a take filmed without a green screen.';
  assert.equal(page.$('moment-tile-clone_flash').querySelectorAll('small')[0].textContent, reason);
  await page.click('moment-tile-clone_flash');
  assert.equal(page.$('moment-clone_flash').disabled, true);
  assert.equal(page.$('moment-reason').textContent, reason);
  await page.save();
  assert.deepEqual(savePayload(page), original);
});

test('Clone flash already on can be turned off on a keyed preview', async () => {
  const state = initialState();
  state.from_library = true;
  state.preview = { ...state.preview, screen: 'blue', screen_checked: true };
  state.style.moments.styles = ['clone_flash', 'spotlight'];
  const original = structuredClone(state.style), page = await harness(state, '#moments');
  await page.click('moment-tile-clone_flash');
  let toggle = page.$('moment-clone_flash');
  assert.equal(toggle.checked, true);
  assert.equal(Boolean(toggle.disabled), false);
  assert.equal(page.$('moment-reason').textContent, 'Needs a take filmed without a blue screen.');
  await page.save();
  assert.deepEqual(savePayload(page), original);
  await page.click('update');
  toggle = page.$('moment-clone_flash');
  toggle.checked = false; await toggle.fire('change'); await flush();
  assert.equal(page.$('moment-clone_flash').disabled, true);
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'moments.styles': ['spotlight'] }));
});

test('Clone flash allows a person cutout despite a keyed preview and saved type', async () => {
  const state = initialState();
  state.style.broll.cutout = 'person';
  state.preview = { ...state.preview, screen: 'green', screen_checked: true };
  state.video_type = 'green_screen';
  const page = await harness(state, '#moments');
  await page.click('moment-tile-clone_flash');
  assert.equal(Boolean(page.$('moment-clone_flash').disabled), false);
  assert.equal(page.$('moment-tile-clone_flash').querySelectorAll('small')[0].textContent, 'Copies of you pop in beside you');
  assert.ok(!page.$('moment-reason') || page.$('moment-reason').hidden);
  const toggle = page.$('moment-clone_flash');
  toggle.checked = true; await toggle.fire('change'); await flush();
  await page.save();
  assert.deepEqual(savePayload(page).moments.styles, ['clone_flash']);
});

test('Clone flash controls follow screen polling within the same generation', async () => {
  const state = initialState(), options = { preview: { ...state.preview, screen: 'green', screen_checked: true } };
  const page = await harness(state, '#moments', options);
  await page.click('moment-tile-clone_flash');
  assert.equal(Boolean(page.$('moment-clone_flash').disabled), false);
  await page.poll();
  assert.equal(page.preview().screen, 'green');
  assert.equal(page.preview().screen_checked, true);
  assert.equal(page.$('moment-clone_flash').disabled, true);
  assert.equal(page.$('moment-tile-clone_flash').querySelectorAll('small')[0].textContent,
    'Needs a take filmed without a green screen.');
  options.preview = { ...options.preview, screen: 'blue' };
  await page.poll();
  assert.equal(page.$('moment-reason').textContent, 'Needs a take filmed without a blue screen.');
  options.preview = { ...options.preview, screen: null };
  await page.poll();
  assert.equal(Boolean(page.$('moment-clone_flash').disabled), false);
  assert.equal(page.$('moment-tile-clone_flash').querySelectorAll('small')[0].textContent, 'Copies of you pop in beside you');
});

test('Clone flash uses the saved type until screen checking finishes', async () => {
  const state = initialState(); state.video_type = 'green_screen';
  const options = { preview: { ...state.preview, screen: null, screen_checked: true } };
  const page = await harness(state, '#captions', options);
  await page.poll();
  assert.equal(page.preview().screen_checked, true);
  await page.go('moments'); await page.click('moment-tile-clone_flash');
  assert.equal(Boolean(page.$('moment-clone_flash').disabled), false);
  options.preview = { ...options.preview, screen_checked: false };
  await page.poll();
  assert.equal(page.$('moment-clone_flash').disabled, true);
  assert.equal(page.$('moment-reason').textContent, 'Needs a take filmed without a green or blue screen.');
});

test('each Moments switch changes only its array and its tile focus never opts in', async () => {
  for (const entry of choices.momentsSlots) for (const on of [true, false]) {
    const state = initialState();
    state.style.moments.styles = on ? [] : [entry.id];
    const original = structuredClone(state.style), page = await harness(state, '#moments');
    assert.equal(page.$('moments-tiles').children.length, 6);
    await page.click(`moment-tile-${entry.id}`);
    assert.equal(page.$('moment-settings').querySelectorAll('input').length, 1);
    const toggle = page.$(`moment-${entry.id}`);
    assert.equal(toggle.checked, !on);
    assert.equal(page.$(`moment-tile-${entry.id}`).getAttribute('aria-pressed'), 'true');
    toggle.checked = on; await toggle.fire('change'); await flush();
    assert.equal(page.$(`moment-tile-${entry.id}`).textContent.includes(' · On'), on);
    assert.match(page.$('change-count').textContent, /Changed Moments/);
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'moments.styles': on ? [entry.id] : [] }));
  }
});

test('Moments choose the first enabled tile in screen order and preserve fields on focus and visits', async () => {
  for (const selected of [undefined, [], ['background_blur', 'color_pop', 'spotlight']]) {
    const state = initialState();
    if (selected === undefined) delete state.style.moments;
    else state.style.moments.styles = selected;
    const original = structuredClone(state.style), page = await harness(state, '#moments');
    assert.equal(page.$('moment-tile-color_pop').getAttribute('aria-pressed'), 'true');
    assert.equal(page.$('moment-settings').querySelectorAll('input').length, 1);
    assert.equal(page.$('main-frame')._drawn.momentId, 'color_pop');
    for (const entry of choices.momentsSlots) {
      await page.click(`moment-tile-${entry.id}`);
      assert.equal(page.$('main-frame')._drawn.momentId, entry.id);
      assert.equal(page.$(`moment-${entry.id}`).checked, Boolean(selected?.includes(entry.id)));
    }
    await page.go('more');
    assert.equal(page.$('choice-moments').textContent.includes(selected?.length ? 'Color pop, Background blur, Spotlight' : 'All off'), true);
    await page.go('moments'); await page.save();
    assert.deepEqual(savePayload(page), original);
  }
});

test('Moments switches keep existing choices sorted and remove only the chosen one', async () => {
  const state = initialState(); state.style.moments.styles = ['spotlight'];
  const original = structuredClone(state.style), page = await harness(state, '#moments');
  assert.equal(page.$('moment-tile-spotlight').getAttribute('aria-pressed'), 'true');
  await page.click('moment-tile-color_pop');
  page.$('moment-color_pop').checked = true; await page.$('moment-color_pop').fire('change'); await flush();
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'moments.styles': ['color_pop', 'spotlight'] }));
});

test('folds and card groups start closed, preserve session state, and remove closed card rows', async () => {
  const page = await harness(initialState(), '#cards', { layers: true });
  assert.ok(page.document.querySelectorAll('details').every((details) => !details.open));
  assert.ok(page.$('card-categories').children.every((group) => group.children.length === 1));
  const chat = page.$('card-group-chat');
  await openFold(page, chat.id);
  assert.equal(chat.querySelectorAll('.kind-row').length, 4);
  assert.equal(chat.querySelectorAll('.new-tag').length, 1);
  assert.ok(chat.querySelectorAll('input').every((toggle) => !toggle.checked));
  page.observers[1].intersect(); await flush();
  assert.equal(chat.querySelectorAll('iframe').length, 3);
  chat.open = false; await chat.fire('toggle');
  assert.equal(chat.querySelectorAll('.kind-row').length, 0);
  assert.equal(chat.querySelectorAll('iframe').length, 0);
  await openFold(page, 'card-group-words');
  await page.go('captions'); await openFold(page, 'highlight-fold');
  const reload = await harness(page.state, '#captions', { storage: page.storage });
  assert.equal(reload.$('caption-animation').open, true);
  assert.equal(reload.$('highlight-fold').open, true);
  await reload.go('cards');
  assert.equal(reload.$('card-group-words').open, true);
  const unavailable = await harness(initialState(), '#captions', { storageError: true });
  await openFold(unavailable, 'highlight-fold');
  await unavailable.go('colors'); await unavailable.go('captions');
  assert.equal(unavailable.$('highlight-fold').open, true);
});

test('closed caption folds do not load compositions until they open, and closing cancels their rows', async () => {
  const page = await harness(initialState(), '#captions', { layers: true });
  page.observers[1].intersect(); await flush();
  assert.equal(page.$('highlight-tiles').querySelectorAll('iframe').length, 0);
  await openFold(page, 'highlight-fold'); page.observers[1].intersect(); await flush();
  assert.equal(page.$('highlight-tiles').querySelectorAll('iframe').length, 24);
  page.$('caption-animation').open = false; await page.$('caption-animation').fire('toggle');
  assert.equal(page.$('highlight-tiles').querySelectorAll('iframe').length, 0);
});

test('preview polling skips overlapping requests and resumes after the response', async () => {
  const state = initialState();
  let complete, calls = 0;
  const response = new Promise((resolve) => { complete = resolve; });
  const options = { previewResponse: () => { calls++; return response; } };
  const page = await harness(state, '#captions', options);
  const first = page.poll(); await flush();
  const second = page.poll(); await flush();
  try { assert.equal(calls, 1); }
  finally {
    complete({ ...state.preview, generation: 2, still: '/still?g=2' });
    await Promise.all([first, second]);
  }
  assert.equal(page.preview().generation, 2);
  options.previewResponse = () => { calls++; return { ...state.preview, generation: 3, still: '/still?g=3' }; };
  await page.poll();
  assert.equal(calls, 2);
  assert.equal(page.preview().generation, 3);
});

test('preview polling ignores older generations and can recover after a stale reply or error', async () => {
  const state = initialState();
  const current = { ...state.preview, generation: 2, still: '/still?g=2', cutout: '/cutout?g=2' };
  const options = { preview: current };
  const page = await harness(state, '#moments', options);
  await page.poll();
  const main = page.$('main-frame').children[0], tiles = [...page.$('moments-tiles').children];
  options.preview = { ...state.preview, generation: 1, still: '/still?g=1', cutout: null };
  await page.poll();
  assert.deepEqual(page.preview(), current);
  assert.equal(page.$('main-frame').children[0], main);
  assert.deepEqual(page.$('moments-tiles').children, tiles);
  options.previewResponse = () => { throw new Error('Synthetic connection failure'); };
  await page.poll();
  options.previewResponse = () => ({ ...current, generation: 3, still: '/still?g=3' });
  await page.poll();
  assert.equal(page.preview().generation, 3);
});

test('cutout-only polling updates preview metadata without rebuilding any preview', async () => {
  const options = { preview: { ...initialState().preview, cutout: '/cutout?g=0' } };
  const page = await harness(initialState(), '#captions', options);
  const main = page.$('main-frame').children[0], tiles = [...page.$('caption-look-tiles').children];
  const count = page.requests.filter((request) => request.url === '/api/layers').length;
  await page.poll();
  assert.equal(page.preview().cutout, '/cutout?g=0');
  assert.equal(page.$('main-frame').children[0], main);
  assert.deepEqual(page.$('caption-look-tiles').children, tiles);
  assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, count);
  options.preview = { ...options.preview, generation: 1, still: '/still?g=1', cutout: null };
  await page.poll();
  assert.equal(page.preview().cutout, null);
  assert.match(page.$('main-frame').querySelectorAll('img')[0].src, /g=1/);
  assert.ok(page.requests.filter((request) => request.url === '/api/layers').length > count);
});

test('graded cutouts match the still grade in Moments and Big number previews, including late arrivals', async () => {
  for (const grade of [{ look: 'film', strength: .5, slider: true },
    { look: 'eyedropper', strength: .75, color: '#1E6BD6', slider: false }]) {
    const state = initialState(); state.style.grade = grade;
    const original = structuredClone(state.style), options = { layers: true, preview: { ...state.preview, cutout: '/cutout?g=0' } };
    const page = await harness(state, '#moments', options);
    await page.poll();
    const assertGrade = (image) => {
      const url = new URL(image.src, 'http://gallery');
      assert.equal(url.searchParams.get('look'), grade.look);
      assert.equal(Number(url.searchParams.get('strength')), grade.strength);
      assert.equal(url.searchParams.get('color'), grade.color ?? null);
      assert.equal(url.searchParams.has('slider'), false);
    };
    for (const image of page.$('moments-tiles').querySelectorAll('.moment-person')) assertGrade(image);
    assertGrade(page.$('main-frame').querySelectorAll('.moment-person')[0]);
    await page.go('cards'); await openFold(page, 'card-group-numbers');
    page.observers[1].intersect(); await flush();
    const host = page.$('kind-big_number').parentElement.children[0];
    await host.click(); await flush();
    assertGrade(host.querySelectorAll('.card-cutout')[0]);
    const main = page.$('main-frame')._galleryStage;
    assertGrade(main.cutout);
    options.preview = { ...options.preview, cutout: '/cutout?g=0&new=1' }; await page.poll();
    assert.equal(main, page.$('main-frame')._galleryStage);
    assertGrade(main.cutout);
    assert.equal(new URL(main.cutout.src, 'http://gallery').searchParams.get('new'), '1');
    await page.save(); assert.deepEqual(savePayload(page), original);
  }
});

test('tall framing seg saves only its field and previews each choice', async () => {
  for (const [framing, label] of [['auto', 'Automatic'], ['keep', 'Keep your framing'],
    ['shrink', 'Shrink you in front of the page']]) {
    const state = initialState(); state.preview.cutout = '/cutout?g=0';
    state.style = choices.withBrollLayout(state.style, 'box', schema);
    state.style.broll.overlay_box = 'tall_behind'; state.style.graphics.paper_color = '#F2D8AA';
    const original = structuredClone(state.style), page = await harness(state, '#broll');
    assert.equal(page.$('broll-framing-fold').hidden, false);
    await pressControl(page, 'broll-tall-framing', ['auto', 'keep', 'shrink'].indexOf(framing));
    const treatment = page.$('main-frame').querySelectorAll('.tall-treatment')[0];
    assert.ok(treatment.classList.contains(`tall-framing-${framing}`));
    assert.equal(treatment.querySelectorAll('.tall-fill').length, 0);
    assert.equal(treatment.querySelectorAll('.tall-room').length, framing === 'shrink' ? 0 : 1);
    assert.equal(treatment.querySelectorAll('.tall-person').length, 1);
    assert.equal(treatment.querySelectorAll('.tall-page').length, 1);
    if (framing === 'shrink') assert.equal(treatment.style.backgroundColor, '#F2D8AA');
    assert.equal(page.$('broll-framing-fold-value').textContent, label);
    await page.save(); original.broll.tall_framing = framing;
    assert.deepEqual(savePayload(page), original);
  }
});

test('tall framing fold is hidden for other box positions', async () => {
  for (const box of ['headroom', 'side_left', 'side_right', 'tall_behind']) {
    const state = initialState(); state.style.broll.overlay_box = box;
    const page = await harness(state, '#broll');
    assert.equal(page.$('broll-framing-fold').hidden, box !== 'tall_behind');
  }
});

test('tall B-roll draws the real room, synthetic page and cached cutout in order', async () => {
  const state = initialState(); state.preview.cutout = '/cutout?g=0';
  state.style.broll.tall_framing = 'keep';
  state.style.broll.overlay_box = 'tall_behind'; state.style.broll.overlay_motion = 'scroll';
  const page = await harness(state, '#broll');
  const frame = page.$('main-frame');
  assert.equal(frame.querySelectorAll('.shot-full').length, 1);
  assert.equal(frame.querySelectorAll('.photo-inset')[0].querySelectorAll('.tall-treatment').length, 1);
  assert.equal(frame._drawn.usesCutout, true);
  const treatment = frame.querySelectorAll('.tall-treatment')[0];
  assert.deepEqual(treatment.children.map((child) => child.className),
    ['tall-room', 'tall-box', 'tall-person']);
  assert.equal(treatment.querySelectorAll('.tall-person')[0].src, '/cutout?g=0&t=token');
  assert.ok(treatment.querySelectorAll('.tall-page')[0].classList.contains('page-scroll'));
  assert.equal(treatment.querySelectorAll('.page-row').length, 32);
  const tile = page.$('broll-overlay-motion').children[2];
  assert.equal(tile.querySelectorAll('.tall-person')[0].src, '/cutout?g=0&t=token');
  assert.ok(tile.querySelectorAll('.tall-page')[0].classList.contains('page-scroll'));
  const original = structuredClone(state.style);
  await page.save(); assert.deepEqual(savePayload(page), original);
});

test('Scroll tile shows an animated drawn page in an ordinary headroom box', async () => {
  const state = initialState(); state.style.broll.overlay_motion = 'scroll';
  const page = await harness(state, '#broll');
  const tile = page.$('broll-overlay-motion').children[2];
  assert.equal(tile.querySelectorAll('.page-box').length, 1);
  assert.ok(tile.querySelectorAll('.tall-page')[0].classList.contains('page-scroll'));
  assert.equal(tile.querySelectorAll('.tall-person').length, 0);
  assert.equal(page.$('main-frame').querySelectorAll('.shot-full').length, 1);
  assert.equal(page.$('main-frame').querySelectorAll('.page-box').length, 1);
  assert.ok(page.$('main-frame').querySelectorAll('.page-box')[0].classList.contains('photo-inset'));
});

test('tall and scroll options preserve Full screen and Split tiles and main previews', async () => {
  const state = initialState(); state.preview.cutout = '/cutout?g=0';
  Object.assign(state.style.broll, { overlay_box: 'tall_behind', overlay_motion: 'scroll' });
  const page = await harness(state, '#broll');
  for (const [index, name, drawing] of [[0, 'Full screen', '.shot-full'], [3, 'Split', '.split-shot']]) {
    const tile = page.$('broll-layout-tiles').children[index];
    assert.equal(tile.querySelectorAll('.tall-treatment').length, 0);
    assert.equal(tile.querySelectorAll('.page-box').length, 0);
    assert.equal(tile.querySelectorAll(drawing).length, 1);
    await page.pick('broll-layout-tiles', name);
    const main = page.$('main-frame');
    assert.equal(main.querySelectorAll('.tall-treatment').length, 0);
    assert.equal(main.querySelectorAll('.page-box').length, 0);
    assert.equal(main.querySelectorAll(drawing).length, 1);
  }
});

test('ordinary Custom split clips and overlay photos keep the original split drawing', async () => {
  for (const motion of ['none', 'drift']) {
    const state = initialState(); state.style.broll.clip_placement = 'split';
    state.style.broll.overlay_motion = motion;
    const page = await harness(state, '#broll'), main = page.$('main-frame');
    assert.equal(main.querySelectorAll('.split-shot').length, 1);
    assert.equal(main.querySelectorAll('.split-speaker').length, 1);
    assert.equal(main.querySelectorAll('.split-seam').length, 1);
    assert.equal(main.querySelectorAll('.shot-box').length, 0);
    assert.equal(main.querySelectorAll('.shot-full').length, 0);
    assert.equal(main.querySelectorAll('.broll-box-picture').length, 0);
  }
});

test('Custom only draws tall and scroll boxes for overlay placements', async () => {
  for (const overlay of [false, true]) {
    const state = initialState();
    state.style = choices.withBrollLayout(state.style, 'full', schema);
    Object.assign(state.style.broll, { overlay_box: 'tall_behind', overlay_motion: 'scroll',
      photo_placement: 'split', screen_recording_placement: overlay ? 'overlay' : 'full_frame' });
    const page = await harness(state, '#broll'), main = page.$('main-frame');
    assert.equal(main.querySelectorAll('.shot-full').length, 1);
    assert.equal(main.querySelectorAll('.tall-treatment').length, overlay ? 1 : 0);
    assert.equal(main.querySelectorAll('.page-box').length, 0);
    assert.equal(main._drawn.usesCutout, overlay);
  }
});

test('3D camera exposes a pause control and keeps tile playback independent', async () => {
  const state = initialState(); state.style.captions.camera_3d = true;
  const page = await harness(state, '#captions');
  const main = page.$('main-frame'), tile = page.$('slot-captions-camera_3d').children[0];
  assert.equal(page.$('track-row').hidden, false);
  assert.equal(main._drawn.element.classList.contains('is-playing'), true);
  await page.click('play');
  assert.equal(main._drawn.element.classList.contains('is-playing'), false);
  assert.equal(page.$('play').getAttribute('aria-label'), 'Play preview');
  await tile.fire('mouseenter');
  assert.equal(tile._drawn.element.classList.contains('is-playing'), true);
  await page.go('colors');
  assert.equal(main._drawn.element.classList.contains('is-playing'), false);
  await page.click('play');
  assert.equal(main._drawn.element.classList.contains('is-playing'), true);
  assert.equal(page.$('play').getAttribute('aria-label'), 'Pause preview');
});

test('3D camera protects expanded heads and the engine safe text top with or without a cutout', async () => {
  for (const cutout of [null, '/cutout?g=0']) for (const face of [
    { x: 351, y: 796, w: 427, h: 427 }, { x: 351, y: 180, w: 427, h: 427 },
  ]) {
    const state = initialState(); state.style.captions.camera_3d = true;
    state.style.graphics.face_clearance_px = 42;
    state.preview.face_boxes = [face]; state.preview.cutout = cutout;
    const page = await harness(state, '#captions');
    const margin = 42 + 8.1;
    const head = { x: face.x - .15 * face.w - margin, y: face.y - .6 * face.h - margin,
      w: face.w * 1.3 + 2 * margin, h: face.h * 1.6 + 2 * margin };
    const push = (box, front) => ({ x: 540 + (box.x - 540 - (front ? 26 : 14)) * (front ? 1.09 : 1.05),
      y: 806.4 + (box.y - 806.4) * (front ? 1.09 : 1.05), w: box.w * (front ? 1.09 : 1.05), h: box.h * (front ? 1.09 : 1.05) });
    for (const line of page.$('main-frame').querySelectorAll('.camera3d-words')) {
      const box = { x: parseFloat(line.style.left), y: parseFloat(line.style.top), w: parseFloat(line.style.width),
        h: parseFloat(line.style.height) || parseFloat(line.style.fontSize) * 2.4 };
      for (const [a, b] of [[box, head], [push(box, true), push(head, false)]]) {
        assert.ok(a.y >= state.style.internal.key_text_top_px - 1e-6);
        assert.ok(a.x + a.w <= b.x || a.x >= b.x + b.w || a.y + a.h <= b.y || a.y >= b.y + b.h,
          JSON.stringify({ a, b, cutout }));
      }
    }
    for (const word of page.$('main-frame').querySelectorAll('.camera3d-behind')) {
      const box = { x: 0, y: parseFloat(word.style.top), w: 0, h: parseFloat(word.style.height) };
      assert.ok(box.y >= 280 && push(box, false).y >= 280 - 1e-6);
    }
  }
});

test('3D camera colors retain the hero accent when Your colors is picked', async () => {
  const state = initialState(); state.style.captions.camera_3d = true;
  state.style.brand = {}; state.style.hook.accent_color = '#12ABCD';
  state.style.captions.highlight_color = '#F08A45'; state.preview.cutout = '/cutout?g=0';
  const page = await harness(state, '#colors');
  assert.equal(page.$('color-accent').value, '#12ABCD');
  await page.pick('palette-tiles', 'Your colors');
  assert.equal(page.$('main-frame').querySelectorAll('.camera3d-behind')[0].style.color, '#12ABCD');
  page.$('color-accent').value = '#909090'; await page.$('color-accent').fire('change'); await flush();
  assert.equal(page.$('main-frame').querySelectorAll('.camera3d-behind')[0].style.color, '#909090');
});

test('3D camera ignores an inactive invalid ground draft and preserves the saved flat ground', async () => {
  for (const mode of ['box', 'band']) {
    const state = initialState(); state.style.captions.background_mode = mode;
    const before = structuredClone(state.style.captions);
    const page = await harness(state, '#captions');
    page.$('background-color').value = '#oops'; await page.$('background-color').fire('change'); await flush();
    assert.equal(page.$('background-color').getAttribute('aria-invalid'), 'true');
    await page.pick('caption-look-tiles', '3D camera');
    assert.equal(page.$('background-fields').hidden, true);
    page.$('background-color').value = '#still-invalid';
    await page.save();
    const saved = page.requests.find((request) => request.url === '/api/save');
    assert.ok(saved, page.$('save-error').textContent);
    for (const field of ['background_mode', 'background_color', 'background_opacity', 'background_padding_px', 'background_radius_px']) {
      assert.equal(saved.data.style.captions[field], before[field]);
    }
  }
});

test('3D camera matches front weight and ignores saved tracking while grading only the hero and its halo', async () => {
  for (const [highlight, weight] of [['slam', '800'], ['clip_wipe', '800'], ['none', '400']]) {
    const state = initialState();
    Object.assign(state.style.captions, { camera_3d: true, text_color: '#909090', highlight_style: highlight, letter_spacing_px: 5 });
    state.style.brand.accent_color = '#909090'; state.style.grade = { look: 'film', strength: 1 };
    state.preview.cutout = '/cutout?g=0';
    const page = await harness(state, '#captions');
    const hero = page.$('main-frame').querySelectorAll('.camera3d-behind')[0];
    assert.equal(hero.style.color, '#969593');
    assert.ok(hero.style.textShadow.includes('#001219'));
    for (const line of page.$('main-frame').querySelectorAll('.camera3d-words')) {
      assert.equal(line.style.color, '#909090');
      assert.equal(line.style.fontWeight, weight);
      assert.ok(!line.style.letterSpacing || line.style.letterSpacing === '0px');
      assert.ok(line.style.textShadow.includes('#101010'));
    }
    await page.save();
    assert.equal(page.requests.find((request) => request.url === '/api/save').data.style.captions.letter_spacing_px, 5);
  }
});

test('3D camera uses the title accent from Candy for the behind word', async () => {
  const state = initialState();
  const preset = JSON.parse(readFileSync(new URL('src/yap_studio/style/presets/candy.json', root)));
  for (const [key, value] of Object.entries(preset)) {
    state.style[key] = typeof value === 'object' ? { ...state.style[key], ...value } : value;
  }
  state.style.captions.camera_3d = true;
  state.preview.cutout = '/cutout?g=0';
  assert.notEqual(state.style.hook.accent_color, state.style.graphics.accent_color);
  const page = await harness(state, '#captions');
  const word = page.$('main-frame').querySelectorAll('.camera3d-behind')[0];
  assert.equal(word.style.color, state.style.hook.accent_color);
});

test('3D camera draws front groups, redraws the cutout and preserves title and card engine tiles', async () => {
  const state = initialState();
  state.style.captions.camera_3d = true;
  state.style.captions.background_mode = 'band';
  state.style.brand.accent_color = '#12ABCD';
  state.style.brand.titles_font = 'Gelasio';
  state.preview.face_boxes = [{ x: 340, y: 400, w: 360, h: 420 }];
  const next = { ...state.preview, cutout: '/cutout?g=0' };
  const page = await harness(state, '#captions', { layers: true, previewResponse: async () => next });
  const tile = page.$('slot-captions-camera_3d'), host = tile.children[0], main = page.$('main-frame');
  assert.equal(tile.getAttribute('aria-pressed'), 'true');
  assert.equal(page.$('camera3d-hint').hidden, false);
  assert.equal(page.$('track-row').hidden, false);
  assert.equal(host._drawn.usesCutout, true);
  assert.equal(main._drawn.screen, 'camera_3d');
  assert.equal(main.querySelectorAll('iframe').length, 0);
  assert.equal(main.querySelectorAll('.poster-caption').length, 0);
  assert.equal(host.querySelectorAll('.camera3d-behind').length, 0);
  const groups = main.querySelectorAll('.camera3d-words');
  assert.deepEqual(groups.map((group) => group.textContent), ['Your real', 'first sentence.']);
  assert.ok(parseFloat(groups[0].style.fontSize) > 60 && parseFloat(groups[0].style.fontSize) <= 84);
  assert.equal(groups[1].style.fontSize, '60px');
  for (const group of groups) {
    assert.equal(group.style.color, state.style.captions.text_color);
    assert.match(group.style.textShadow, /rgba\(0,0,0,/);
    assert.match(group.style.fontFamily, /Inter/);
    const top = parseFloat(group.style.top), height = parseFloat(group.style.height);
    const x = parseFloat(group.style.left), width = parseFloat(group.style.width);
    assert.ok(top + height < 400 || top > 820 || x + width < 340 || x > 700);
  }
  page.observers.find((observer) => observer.hosts.has(host)).intersect();
  await host.fire('mouseenter');
  assert.equal(host._drawn.element.classList.contains('is-playing'), true);
  await host.fire('mouseleave');
  assert.equal(host._drawn.element.classList.contains('is-playing'), false);
  await page.poll();
  const behind = host.querySelectorAll('.camera3d-behind')[0], person = host.querySelectorAll('.camera3d-person')[0];
  assert.equal(behind.hidden, true);
  await person.fire('load');
  assert.equal(behind.hidden, false);
  assert.equal(behind.style.color, '#12ABCD');
  assert.match(behind.style.fontFamily, /Gelasio/);
  await person.fire('error');
  assert.equal(behind.hidden, true);
  await page.go('opening'); page.observers[1].intersect(); await flush();
  assert.ok(page.$('title-tiles').querySelectorAll('iframe').length > 0);
  assert.equal(main._drawn, null);
  assert.ok(main._galleryStage);
  assert.deepEqual(Array.from(main._galleryStage.frames, (frame) => frame.title), ['title live preview']);
  assert.equal(page.$('track-row').hidden, false);
  await page.go('cards'); page.observers[1].intersect(); await flush();
  assert.ok(page.$('card-style-tiles').querySelectorAll('iframe').length > 0);
  assert.equal(main._drawn, null);
  assert.ok(main._galleryStage);
  assert.deepEqual(Array.from(main._galleryStage.frames, (frame) => frame.title), ['card live preview']);
  assert.equal(page.$('track-row').hidden, false);
  for (const screen of ['broll', 'moves', 'moments', 'sound']) {
    await page.go(screen);
    assert.equal(main._drawn.screen, screen);
    assert.equal(page.$('track-row').hidden, true);
  }
  for (const screen of ['looks', 'font', 'captions', 'colors']) {
    await page.go(screen);
    assert.equal(main._drawn.screen, 'camera_3d');
    assert.equal(page.$('track-row').hidden, false);
  }
  for (const screen of ['grade', 'more']) {
    await page.go(screen);
    assert.equal(main._drawn, null);
    assert.ok(main._galleryStage);
    assert.equal(page.$('track-row').hidden, false);
  }
  await page.go('captions');
  await page.pick('caption-look-tiles', 'Plain');
  assert.equal(page.$('camera3d-hint').hidden, true);
  assert.equal(main._drawn, null);
  assert.equal(page.$('track-row').hidden, false);
});

test("3D camera stays through words, font and colors; a look and Reset changes take the look's own value", async () => {
  const state = initialState();
  const cameraLook = choices.withCaptionLook(state.style, 'camera_3d');
  state.library.push({ id: 'camera', name: 'Camera look', style: { ...cameraLook, name: 'Camera look' } });
  const page = await harness(state, '#captions');
  await page.pick('caption-look-tiles', '3D camera');
  assert.equal(page.$('track-row').hidden, false);
  await page.pick('word-tiles', 'One word');
  await page.go('font');
  await page.pick('font-options', 'Gelasio');
  await page.go('colors');
  await page.pick('palette-tiles', choices.palettes[1].name);
  await page.go('captions');
  assert.equal(page.$('slot-captions-camera_3d').getAttribute('aria-pressed'), 'true');
  await page.click('reset');
  assert.equal(page.$('slot-captions-camera_3d').getAttribute('aria-pressed'), 'false');
  await page.pick('caption-look-tiles', '3D camera');
  await page.go('looks');
  await page.pick('start-options', 'Clean Minimal');
  await page.go('captions');
  assert.equal(page.$('slot-captions-camera_3d').getAttribute('aria-pressed'), 'false');
  await page.pick('caption-look-tiles', '3D camera');
  await page.click('reset');
  assert.equal(page.$('slot-captions-camera_3d').getAttribute('aria-pressed'), 'false');
  await page.go('looks');
  await page.pick('start-options', 'Camera look');
  await page.go('captions');
  assert.equal(page.$('slot-captions-camera_3d').getAttribute('aria-pressed'), 'true');
  await page.pick('caption-look-tiles', 'Plain');
  await page.click('reset');
  assert.equal(page.$('slot-captions-camera_3d').getAttribute('aria-pressed'), 'true');
  await page.save();
  assert.equal(page.requests.filter((request) => request.url === '/api/save').at(-1).data.style.captions.camera_3d, true);
});

test('3D camera tiles fill a phone crop while the main preview keeps the whole frame', async () => {
  const state = initialState(); state.style.captions.camera_3d = true;
  state.preview.cutout = '/cutout?g=0';
  state.preview.face_boxes = [{ x: 351, y: 796, w: 427, h: 427 }];
  const page = await harness(state, '#captions', { phone: true });
  const tile = page.$('slot-captions-camera_3d').children[0], main = page.$('main-frame');
  tile.clientHeight = 220;
  tile._drawn.resize.callback();
  const picture = tile.querySelectorAll('.drawn-picture')[0];
  const scale = Number(picture.style.transform.match(/scale\(([^)]+)\)/)[1]);
  const left = parseFloat(picture.style.left), top = parseFloat(picture.style.top);
  assert.ok(left <= 0 && left + 1080 * scale >= tile.clientWidth);
  assert.ok(top <= 0 && top + 1920 * scale >= tile.clientHeight);
  const face = state.preview.face_boxes[0];
  assert.ok(left + face.x * scale >= 0 && left + (face.x + face.w) * scale <= tile.clientWidth);
  assert.ok(top + face.y * scale >= 0 && top + (face.y + face.h) * scale <= tile.clientHeight);
  const first = tile.querySelectorAll('.camera3d-words')[0];
  assert.ok(top + parseFloat(first.style.top) * scale >= 0);
  assert.ok(top + (parseFloat(first.style.top) + parseFloat(first.style.fontSize) * 1.12) * scale <= tile.clientHeight);
  const hero = tile.querySelectorAll('.camera3d-behind')[0];
  assert.ok(top + parseFloat(hero.style.top) * scale >= 0);
  assert.ok(top + (parseFloat(hero.style.top) + parseFloat(hero.style.height)) * scale <= tile.clientHeight);
  const whole = main.querySelectorAll('.drawn-picture')[0];
  assert.equal(whole.style.transform, `scale(${main.clientWidth / 1080})`);
  assert.equal(whole.style.left, '0px');
  assert.equal(whole.style.top, '0px');
});

test('3D camera keeps at least ninety percent of the behind word clear of the estimated head', async () => {
  for (const face of [{ x: 351, y: 796, w: 427, h: 427 },
    { x: 351, y: 650, w: 427, h: 280 }, { x: 351, y: 180, w: 427, h: 427 }]) {
    const state = initialState(); state.style.captions.camera_3d = true;
    state.preview.sentence = 'Tell your story.';
    state.preview.cutout = '/cutout?g=0'; state.preview.face_boxes = [face];
    const page = await harness(state, '#captions');
    const word = page.$('main-frame').querySelectorAll('.camera3d-behind')[0];
    const box = { x: parseFloat(word.style.left ?? 60), y: parseFloat(word.style.top),
      w: parseFloat(word.style.width ?? 960), h: parseFloat(word.style.height ?? word.style.fontSize) };
    const margin = state.style.graphics.face_clearance_px + 8.1;
    const head = { x: face.x - .15 * face.w - margin, y: face.y - .6 * face.h - margin,
      w: 1.3 * face.w + 2 * margin, h: 1.6 * face.h + 2 * margin };
    const overlap = Math.max(0, Math.min(box.x + box.w, head.x + head.w) - Math.max(box.x, head.x))
      * Math.max(0, Math.min(box.y + box.h, head.y + head.h) - Math.max(box.y, head.y));
    assert.ok(1 - overlap / (box.w * box.h) >= .9 - 1e-6, JSON.stringify({ face, box, head }));
    assert.ok(box.y >= 280);
    if (face.y === 180) assert.ok(box.x + box.w <= head.x || box.x >= head.x + head.w);
    else assert.ok(box.y + box.h <= head.y + .1 * box.h + 1e-6);
    if (face.y === 650) assert.ok(box.h < 240);
  }
});

test('3D camera puts the first group beside a high face and respects phrase limits', async () => {
  const state = initialState(); state.style.captions.camera_3d = true;
  state.style.captions.max_words_per_phrase = 1;
  state.style.captions.max_characters_per_phrase = 8;
  state.preview.face_boxes = [{ x: 340, y: 80, w: 360, h: 420 }];
  const page = await harness(state, '#captions');
  const groups = page.$('main-frame').querySelectorAll('.camera3d-words');
  assert.deepEqual(groups.map((group) => group.textContent), ['Your', 'real']);
  const first = groups[0], x = parseFloat(first.style.left), width = parseFloat(first.style.width);
  assert.ok(x + width < 340 || x > 700);
});

test('3D camera keeps whole words inside a narrow side group', async () => {
  const state = initialState(); state.style.captions.camera_3d = true;
  state.style.captions.highlight_style = 'slam';
  state.preview.sentence = 'Tell your story.';
  state.preview.face_boxes = [{ x: 351, y: 796, w: 427, h: 427 }];
  state.preview.cutout = '/cutout?g=0';
  const page = await harness(state, '#captions');
  for (const group of page.$('main-frame').querySelectorAll('.camera3d-words')) {
    assert.equal(group.hidden, false);
    const size = parseFloat(group.style.fontSize), width = parseFloat(group.style.width);
    assert.ok(size >= 60 && size <= 84);
    assert.ok(Math.max(...group.textContent.split(/\s+/).map((word) => word.length * size * .7)) + 4 <= width,
      JSON.stringify({ text: group.textContent, size, width }));
  }
});

test('3D camera picks the higher contrast halo before the chosen shadow', async () => {
  for (const [color, halo] of [['#FFFFFF', '#101010'], ['#A0A0A0', '#101010'],
    ['#9F9F9F', '#101010'], ['#909090', '#101010'], ['#101010', '#FFFFFF']]) {
    const state = initialState();
    state.preview.cutout = '/cutout?g=0';
    state.style.brand.accent_color = color === '#101010' ? '#FFFFFF' : '#101010';
    Object.assign(state.style.captions, { camera_3d: true, text_color: color,
      shadow_x_px: 3, shadow_y_px: 6, shadow_blur_px: 8, shadow_opacity: .35 });
    const page = await harness(state, '#captions');
    for (const group of page.$('main-frame').querySelectorAll('.camera3d-words')) {
      const shadows = group.style.textShadow.split(', ');
      assert.equal(shadows.length, 9);
      assert.equal(new Set(shadows.slice(0, 8)).size, 8);
      for (const shadow of shadows.slice(0, 8)) {
        assert.ok(shadow.endsWith(` 0 ${halo}`), shadow);
        const [x, y] = shadow.split(' ').map(parseFloat);
        assert.ok(Math.max(Math.abs(x), Math.abs(y)) >= 1 && Math.max(Math.abs(x), Math.abs(y)) <= 2);
      }
      assert.equal(shadows[8], '3px 6px 8px rgba(0,0,0,0.35)');
    }
    const hero = page.$('main-frame').querySelectorAll('.camera3d-behind')[0];
    const heroShadows = hero.style.textShadow.split(', ');
    assert.equal(heroShadows.length, 8);
    const heroHalo = state.style.brand.accent_color === '#FFFFFF' ? '#101010' : '#FFFFFF';
    assert.ok(heroShadows.every((shadow) => shadow.endsWith(` 0 ${heroHalo}`)));
  }
});

test('3D camera keeps white ink and a dark halo after an opaque light Box or Band', async () => {
  for (const mode of ['box', 'band']) {
    const state = initialState();
    Object.assign(state.style.captions, { text_color: '#FFFFFF', background_mode: mode,
      background_color: '#FFFFFF', background_opacity: 1 });
    const page = await harness(state, '#captions');
    await page.pick('caption-look-tiles', '3D camera');
    for (const host of [page.$('slot-captions-camera_3d').children[0], page.$('main-frame')]) {
      const groups = host.querySelectorAll('.camera3d-words');
      assert.ok(groups.length);
      for (const group of groups) {
        assert.equal(group.style.color, '#FFFFFF');
        assert.ok(group.style.textShadow.split(', ').slice(0, 8)
          .every((shadow) => shadow.endsWith(' 0 #101010')));
      }
    }
    assert.equal(page.$('caption-result').hidden, true);
    await page.save();
    const saved = page.requests.filter((request) => request.url === '/api/save').at(-1).data.style.captions;
    assert.equal(saved.text_color, '#FFFFFF');
    assert.equal(saved.background_mode, mode);
    assert.equal(saved.background_color, '#FFFFFF');
    assert.equal(saved.background_opacity, 1);
    await page.pick('caption-look-tiles', mode === 'box' ? 'Box' : 'Band');
    assert.match(page.$('caption-result').textContent, /dark.*#111111/);
  }
});

test('3D camera grade tiles request their explicit candidate still including None', async () => {
  const state = initialState(); state.style.captions.camera_3d = true;
  state.preview.still = '/still?g=0&look=film&strength=0.5';
  const page = await harness(state, '#grade');
  for (const [index, look] of ['none', 'film', 'eyedropper'].entries()) {
    const host = page.$('grade-tiles').children[index].children[0];
    assert.equal(host._drawn.screen, 'camera_3d');
    const src = host.querySelectorAll('.drawn-still')[0].src;
    const params = new URLSearchParams(src.split('?')[1]);
    assert.equal(params.get('look'), look);
    assert.equal(params.get('strength'), String(host._drawn.candidate.grade.strength));
    assert.equal(host.querySelectorAll('.camera3d-words').length, 2);
  }
  await page.pick('grade-tiles', 'Film look');
  assert.equal(page.$('main-frame').querySelectorAll('.camera3d-scene').length, 0);
  assert.match(page.$('main-frame').querySelectorAll('img')[0].src, /look=film&strength=1/);
});

test('3D camera main plays while tiles play on hover or on a phone, and reduced motion stops both', async () => {
  for (const phone of [false, true]) for (const reducedMotion of [false, true]) {
    const state = initialState(); state.style.captions.camera_3d = true;
    const page = await harness(state, '#captions', { phone, reducedMotion });
    const main = page.$('main-frame');
    assert.equal(main._drawn.element.classList.contains('is-playing'), !reducedMotion);
    const host = page.$('slot-captions-camera_3d').children[0];
    const observer = page.observers.find((observer) => observer.hosts.has(host));
    assert.equal(host._drawn.element.classList.contains('is-playing'), false);
    observer.intersect();
    assert.equal(host._drawn.element.classList.contains('is-playing'), phone && !reducedMotion);
    await host.fire('mouseenter');
    assert.equal(host._drawn.element.classList.contains('is-playing'), !reducedMotion);
    await host.fire('mouseleave');
    assert.equal(host._drawn.element.classList.contains('is-playing'), phone && !reducedMotion);
    observer.intersect(0);
    assert.equal(host._drawn.element.classList.contains('is-playing'), false);
    assert.equal(main._drawn.element.classList.contains('is-playing'), !reducedMotion);
  }
});

test('Moments use the cutout, face-centered light and highlight color without engine requests', async () => {
  const state = initialState(); state.preview.cutout = '/cutout?g=0';
  state.preview.face_boxes = [{ x: 216, y: 384, w: 216, h: 384 }];
  const page = await harness(state, '#moments');
  assert.equal(page.$('track-row').hidden, true);
  assert.equal(page.$('main-frame').querySelectorAll('iframe').length, 0);
  for (const entry of choices.momentsSlots) {
    const tile = page.$(`moment-tile-${entry.id}`), treatment = tile.querySelectorAll('.moment-treatment')[0];
    assert.ok(treatment.classList.contains(`moment-${entry.id}`));
    assert.equal(treatment.classList.contains('moment-fallback'), false);
    assert.equal(tile.querySelectorAll('.moment-person')[0].src, '/cutout?g=0&t=token');
    assert.equal(tile.querySelectorAll('.poster-caption').length, 1);
    assert.equal(treatment.style['--moment-accent'], state.style.brand.accent_color || state.style.graphics.accent_color);
  }
  const light = page.$('moment-tile-spotlight').querySelectorAll('.moment-treatment')[0];
  assert.equal(light.style['--moment-light-x'], '30%');
  assert.equal(light.style['--moment-light-y'], '30%');
  const person = light.querySelectorAll('.moment-person')[0];
  await person.fire('error');
  assert.equal(person.hidden, true); assert.equal(light.classList.contains('moment-fallback'), true);
  assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, 0);
});

test('cutout changes redraw only Moments drawings and retain focus, switches and other thumbnails', async () => {
  const options = { preview: { ...initialState().preview, cutout: '/cutout?g=0' } };
  const page = await harness(initialState(), '#moments', options);
  await page.click('moment-tile-spotlight');
  const tiles = [...page.$('moments-tiles').children], toggle = page.$('moment-spotlight');
  const original = tiles.map((tile) => tile.children[0].children[0]), main = page.$('main-frame').children[0];
  await page.poll();
  assert.deepEqual(page.$('moments-tiles').children, tiles);
  assert.equal(page.$('moment-spotlight'), toggle);
  assert.notEqual(page.$('main-frame').children[0], main);
  assert.equal(page.$('main-frame')._drawn.momentId, 'spotlight');
  for (const [index, tile] of tiles.entries()) {
    assert.notEqual(tile.children[0].children[0], original[index]);
    assert.equal(tile.querySelectorAll('.moment-person')[0].src, '/cutout?g=0&t=token');
  }
  const held = page.$('main-frame').children[0]; await page.poll();
  assert.equal(page.$('main-frame').children[0], held);
  options.preview = { ...options.preview, cutout: '/cutout?g=0&new=1' }; await page.poll();
  assert.match(page.$('main-frame').querySelectorAll('.moment-person')[0].src, /new=1&t=token/);
  options.preview = { ...options.preview, cutout: null }; await page.poll();
  assert.equal(page.$('main-frame').querySelectorAll('.moment-person').length, 0);
  assert.ok(page.$('main-frame').querySelectorAll('.moment-treatment')[0].classList.contains('moment-fallback'));
  await page.go('more');
  const thumbnails = page.$('more-choices').children.map((row) => [row.id, row.children[0].children[0]]);
  const mainLive = page.$('main-frame').children[0], requests = page.requests.filter((request) => request.url === '/api/layers').length;
  options.preview = { ...options.preview, cutout: '/cutout?g=0' }; await page.poll();
  for (const [id, before] of thumbnails) {
    const after = page.$(id).children[0].children[0];
    if (id === 'choice-moments') assert.notEqual(after, before);
    else assert.equal(after, before, id);
  }
  assert.equal(page.$('main-frame').children[0], mainLive);
  assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, requests);
  assert.equal(page.$('choice-moments').querySelectorAll('.poster-caption').length, 0);
  assert.ok([...page.observers[2].hosts].every((host) => host.isConnected));
});

test('Moments play only on hover or half visibility, with held reduced-motion and missing-cutout previews', async () => {
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#moments', { phone });
    const host = page.$('moments-tiles').children[0].children[0], drawing = host.children[0];
    assert.equal(drawing.classList.contains('is-playing'), false);
    assert.ok(drawing.querySelectorAll('.moment-treatment')[0].classList.contains('moment-fallback'));
    assert.equal(page.$('main-frame').children[0].classList.contains('is-playing'), true);
    if (phone) {
      page.observers[2].intersect(.49); assert.equal(drawing.classList.contains('is-playing'), false);
      page.observers[2].intersect(.5);
    } else await host.fire('mouseenter');
    assert.equal(drawing.classList.contains('is-playing'), true);
    if (phone) page.observers[2].intersect(0); else await host.fire('mouseleave');
    assert.equal(drawing.classList.contains('is-playing'), false);
  }
  const reduced = await harness(initialState(), '#moments', { reducedMotion: true });
  assert.equal(reduced.$('main-frame').children[0].classList.contains('is-playing'), false);
  assert.ok(reduced.$('main-frame').querySelectorAll('.moment-treatment').length);
});

test('an empty Moments option list hides its row, route and next step', async () => {
  const entries = choices.momentsSlots.splice(0);
  try {
    const page = await harness(initialState(), '#moments');
    assert.equal(page.document.body.dataset.screen, 'more');
    assert.equal(page.$('choice-moments'), undefined);
    await page.go('moves'); await page.click('next');
    assert.equal(page.location.hash, '#sound');
  } finally { choices.momentsSlots.push(...entries); }
});

test('drawn previews use no compositions, keep More thumbnails static, and gate motion by visibility', async () => {
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#broll', { phone });
    assert.equal(page.$('track-row').hidden, true);
    assert.equal(page.$('main-frame').querySelectorAll('iframe').length, 0);
    assert.equal(page.$('broll-layout-tiles').querySelectorAll('iframe').length, 0);
    assert.equal(page.requests.filter((request) => request.url === '/api/layers').length, 0);
    const frame = page.$('broll-layout-tiles').children[2].children[0], drawing = frame.children[0];
    assert.equal(drawing.classList.contains('is-playing'), false);
    if (phone) {
      page.observers[2].intersect(.49); assert.equal(drawing.classList.contains('is-playing'), false);
      page.observers[2].intersect(.5);
    } else await frame.fire('mouseenter');
    assert.equal(drawing.classList.contains('is-playing'), true);
    if (phone) page.observers[2].intersect(0);
    else await frame.fire('mouseleave');
    assert.equal(drawing.classList.contains('is-playing'), false);
    await page.go('more');
    assert.equal(page.$('more-choices').querySelectorAll('iframe').length, 0);
    assert.equal(page.$('more-choices').querySelectorAll('.poster-caption').length, 0);
    assert.ok(page.$('more-choices').querySelectorAll('.drawn-preview').every((preview) => !preview.classList.contains('is-playing')));
  }
  const reduced = await harness(initialState(), '#broll', { reducedMotion: true });
  const frame = reduced.$('broll-layout-tiles').children[0].children[0];
  await frame.fire('mouseenter');
  assert.equal(frame.children[0].classList.contains('is-playing'), false);
});

test('bleep masks the 3D front words and behind word only when enabled', async () => {
  for (const [sentence, raw, masked] of [
    ['Fucking stories today.', 'Fucking', 'F******'], ['Bull shit.', 'Bull', 'B***'],
  ]) for (const on of [false, true]) {
    const state = initialState();
    state.style.captions.camera_3d = true;
    state.style.music.bleep_swears = on;
    state.preview.sentence = sentence;
    state.preview.cutout = '/cutout?g=0';
    const page = await harness(state, '#captions');
    const main = page.$('main-frame');
    const front = main.querySelectorAll('.camera3d-words').map((word) => word.textContent).join(' ');
    assert.ok(front.includes(on ? masked : raw), front);
    const behind = main.querySelectorAll('.camera3d-behind')[0];
    assert.equal(behind.textContent, on ? masked : raw);
  }
});

test('bleep masks every client preview and restores spoken text when off', async () => {
  for (const screen of ['sound', 'broll', 'opening', 'cards', 'moments']) {
    const state = initialState(); state.preview.sentence = 'Shit hits the fan.';
    state.style.music.bleep_swears = true;
    const page = await harness(state, '#' + screen);
    const words = page.document.querySelectorAll('.poster-word').map((word) => word.textContent);
    assert.ok(words.includes('S***'), screen + ': ' + words.join(' '));
    for (const selector of ['.drawn-title', '.drawn-card', '.letters-word']) {
      for (const item of page.document.querySelectorAll(selector)) assert.ok(!item.textContent.includes('Shit'), selector);
    }
  }
  const state = initialState(); state.preview.sentence = 'Shit hits the fan.';
  const page = await harness(state, '#sound');
  assert.equal(page.$('main-frame').querySelectorAll('.poster-word')[0].textContent, 'Shit');
  const toggle = page.$('bleep-swears'); toggle.checked = true; await toggle.fire('change'); await flush();
  assert.equal(page.$('main-frame').querySelectorAll('.poster-word')[0].textContent, 'S***');
  toggle.checked = false; await toggle.fire('change'); await flush();
  assert.equal(page.$('main-frame').querySelectorAll('.poster-word')[0].textContent, 'Shit');
});

test('drawn caption posters use the frame scale, placement and template weight', async () => {
  const state = initialState(); state.style.captions.highlight_style = 'none';
  state.style.captions.position = 'middle'; state.style.captions.top_px = 700;
  const page = await harness(state, '#sound');
  const caption = page.$('music-level-tiles').querySelectorAll('.poster-caption')[0];
  assert.ok(Math.abs(parseFloat(caption.style.fontSize) - state.style.captions.size_px * 200 / 1080) < .00001);
  assert.equal(caption.style.fontWeight, '400');
  assert.equal(caption.style.top, `${700 / 1920 * 100}%`);
  assert.match(caption.style.transform, /translateY\(-50%\)/);
  const bottom = await harness(initialState(), '#broll');
  const poster = bottom.$('main-frame').querySelectorAll('.poster-caption')[0];
  assert.equal(poster.style.top, 'auto');
  assert.ok(parseFloat(poster.style.bottom) > 0);
});

test('phone main caption tiles crop lower while other tiles use the whole frame, then refit on desktop', async () => {
  for (const [screen, grid, name] of [['captions', 'caption-look-tiles', 'captions'], ['opening', 'title-tiles', 'title'],
    ['cards', 'card-style-tiles', 'card'], ['grade', 'grade-tiles', 'face']]) {
    const layerGeometry = { captions: { y: 1384, height: 251 }, title: { y: 120, height: 300 }, card: { y: 340, height: 500 } };
    const settings = { phone: true, layers: true, layerGeometry };
    const page = await harness(initialState(), '#' + screen, settings);
    const host = page.$(grid).children[0].children[0]; host.clientHeight = screen === 'captions' ? 220 : 200 * 16 / 9;
    if (screen === 'captions') assert.equal(host.querySelectorAll('.poster-caption')[0].style.top, '72%');
    page.observers[1].callback([{ target: host, isIntersecting: true }]); await flush();
    const focus = name === 'face' ? { x: 540, y: 960, width: 0, height: 0 }
      : { x: 0, width: 1080, ...layerGeometry[name] };
    if (screen === 'captions') {
      const expected = choices.tileWindow(focus, undefined, { width: 200, height: 220 }, .72);
      assert.ok(Math.abs(parseFloat(host._galleryStage.stage.style.top) + expected.y * 200 / 900) < .00001, screen);
    } else {
      assert.equal(host._galleryStage.stage.style.top, '0px', screen);
      assert.equal(host._galleryStage.stage.style.transform, `scale(${200 / 1080})`, screen);
    }
    settings.phone = false;
    host.clientHeight = 200 * 16 / 9; host._galleryStage.resize.callback();
    const resized = choices.tileWindow(focus);
    assert.ok(Math.abs(parseFloat(host._galleryStage.stage.style.top) + resized.y * 200 / 900) < .00001, screen);
  }
});

test('phone caption fold tiles keep full frames and compact highlight crops', async () => {
  const page = await harness(initialState(), '#captions', { phone: true, layers: true });
  for (const [id, compact] of [['word-tiles', false], ['highlight-tiles', true]]) {
    await openFold(page, id);
    const host = page.$(id).children[0].children[0];
    host.clientHeight = 200 * (compact ? 7 / 16 : 16 / 9);
    page.observers[1].callback([{ target: host, isIntersecting: true }]); await flush();
    assert.equal(host._galleryStage.stage.style.transform, `scale(${200 / (compact ? 900 : 1080)})`);
  }
});

test('phone drawn tiles keep the whole still in main choices and B-roll folds', async () => {
  for (const [screen, grids] of [['broll', ['broll-layout-tiles', 'broll-overlay-motion', 'broll-still-push']], ['sound', ['music-level-tiles']]]) {
    const page = await harness(initialState(), '#' + screen, { phone: true });
    for (const grid of grids) for (const picture of page.$(grid).querySelectorAll('.drawn-picture')) {
      assert.equal(picture.style.transform, `scale(${200 / 1080})`);
      assert.equal(parseFloat(picture.style.top), 0);
    }
  }
});

test('B-roll shot kind fold keeps only the layout while the More row keeps stock clips', async () => {
  const seed = initialState();
  for (const [layout, name] of [['full', 'Full screen'], ['mixed', 'Mixed'], ['box', 'In a box'], ['custom', 'Custom']]) {
    const state = initialState();
    state.style = layout === 'custom' ? choices.withBrollPlacement(seed.style, 'clip', 'overlay') : choices.withBrollLayout(seed.style, layout);
    state.style = choices.withBrollField(state.style, 'stock_clips', 'ask');
    const original = structuredClone(state.style), page = await harness(state, '#broll');
    assert.equal(page.$('broll-kinds-fold-value').textContent, name);
    await page.go('more');
    assert.match(page.$('choice-broll').textContent, new RegExp(name + ' · stock clips: ask first'));
    await page.save();
    assert.deepEqual(savePayload(page), original);
  }
});

test('B-roll motion tiles stay tall and photo motion follows photo placement', async () => {
  for (const placement of ['overlay', 'full_frame', 'split']) {
    const state = initialState(); state.style.broll.photo_placement = placement;
    const page = await harness(state, '#broll');
    const photos = page.$('broll-still-push'), motion = page.$('broll-overlay-motion');
    assert.ok([...photos.children, ...motion.children].every((tile) => !tile.children[0].classList.contains('compact')));
    assert.equal(photos.querySelectorAll('.shot-full').length, placement === 'full_frame' ? 3 : 0);
    assert.equal(photos.querySelectorAll('.shot-box').length, placement === 'overlay' ? 3 : 0);
    assert.equal(photos.querySelectorAll('.split-shot').length, placement === 'split' ? 3 : 0);
    assert.equal(motion.querySelectorAll('.shot-box').length, 3);
  }
});

test('replacing drawn hosts releases intersection and resize observers', async () => {
  const page = await harness(initialState(), '#broll');
  const oldHosts = [...page.observers[2].hosts];
  const oldResize = oldHosts.map((host) => host._drawn.resize);
  await page.pick('broll-layout-tiles', 'In a box');
  assert.ok(oldResize.every((resize) => !resize.connected));
  assert.ok(oldHosts.filter((host) => host.id !== 'main-frame').every((host) => !page.observers[2].hosts.has(host) && !host._drawn));
  await page.go('captions');
  assert.equal(page.observers[2].hosts.size, 1);
  assert.equal([...page.observers[2].hosts][0], page.$('slot-captions-camera_3d').children[0]);
  assert.equal(page.$('main-frame')._drawn, null);
  await page.go('more'); await page.go('sound');
  assert.ok([...page.observers[2].hosts].every((host) => host.isConnected));
});

test('validation opens the screen and all folds around each bad color field', async () => {
  for (const [screen, id, folds] of [['captions', 'caption-color', ['caption-animation', 'caption-color-fold']],
    ['captions', 'background-color', ['caption-animation', 'background-fields']],
    ['colors', 'color-paper', ['colors-more']], ['grade', 'grade-color', []]]) {
    const page = await harness(initialState(), '#' + screen);
    if (id === 'background-color') await page.pick('caption-look-tiles', 'Box');
    if (screen === 'grade') await page.pick('grade-tiles', 'Eyedropper');
    page.$(id).value = '#oops'; await page.$(id).fire('change'); await flush();
    await page.go('sound'); await page.save();
    assert.equal(page.document.body.dataset.screen, screen);
    assert.equal(page.document.activeElement.id, id);
    assert.ok(folds.every((fold) => page.$(fold).open));
    assert.equal(page.requests.some((request) => request.url === '/api/save'), false);
    assert.equal(page.$(id).value, '#oops');
  }
});

test('each main tile and fold control saves exactly its documented fields', async (t) => {
  const cases = [];
  const add = (name, screen, fields, action, seed = {}, expected) => cases.push({ name, screen, fields, action, seed, expected });
  const pick = (id, label) => (page) => page.pick(id, label);
  const press = (id, index) => (page) => pressControl(page, id, index);
  const input = (id, value, event = 'input') => async (page) => { await openFold(page, id); page.$(id).value = value; await page.$(id).fire(event); await flush(); };
  const toggle = (id, on) => async (page) => { await openFold(page, id); page.$(id).checked = on; await page.$(id).fire('change'); await flush(); };
  for (const font of bundledFonts) {
    add('caption font ' + font, 'font', null, pick('font-options', font), {}, (style) => choices.withFont(style, font));
    add('graphics font ' + font, 'font', null, pick('graphics-font-options', font), {}, (style) => choices.withGraphicsFont(style, font));
  }
  add('font pairing', 'font', null, (page) => page.click('font-pair'), {}, choices.withFontPair);
  add('same as captions', 'font', null, (page) => page.click('font-same'), { 'hook.font': 'Geist', 'graphics.font': 'Geist' },
    (style) => choices.withFont(style, choices.captionsFont(style)));
  for (const width of choices.widths) add('width ' + width.name, 'font', { 'captions.width_scale': width.value }, pick('width-tiles', width.name));
  for (const [index, mode] of ['soft', 'strong'].entries()) add('shadow ' + mode, 'captions',
    Object.fromEntries(Object.entries(choices.shadows[mode]).map(([field, value]) => ['captions.' + field, value])), press('shadow-strength', index));
  add('shadow off', 'captions', { 'captions.shadow_opacity': 0 }, press('shadow-toggle', 1));
  add('shadow on', 'captions', Object.fromEntries(Object.entries(choices.shadows.soft).map(([field, value]) => ['captions.' + field, value])),
    press('shadow-toggle', 0), { 'captions.shadow_opacity': 0 });
  add('caption color hex', 'captions', { 'captions.text_color': '#AABBCC' }, input('caption-color', '#abc', 'change'));
  add('caption color picker', 'captions', { 'captions.text_color': '#ABCDEF' }, input('caption-picker', '#ABCDEF'));
  add('caption position slider', 'captions', { 'captions.top_px': 820 }, input('caption-position', 820));
  add('background color picker', 'captions', { 'captions.background_color': '#ABCDEF' }, input('background-picker', '#ABCDEF'), { 'captions.background_mode': 'box' });
  for (const [index, value] of ['#000000', '#FFFFFF', '#365A58'].entries()) add('background swatch ' + value, 'captions',
    { 'captions.background_color': value }, async (page) => { await openFold(page, 'background-fields'); await page.$('background-swatches').children[index].click(); await flush(); }, { 'captions.background_mode': 'box' });
  for (const palette of choices.palettes) add('palette ' + palette.name, 'colors', null, pick('palette-tiles', palette.name), {}, (style) => choices.withColors(style, palette));
  for (const [role] of choices.roles) add('color role ' + role, 'colors', null, input('color-' + role, '#ABCDEF', 'change'), {},
    (style) => choices.withColors(style, { ...choices.colorsOf(style), [role]: '#ABCDEF' }, [role]));
  for (const [name, look] of [['None', 'none'], ['Film look', 'film'], ['Eyedropper', 'eyedropper']]) add('grade ' + name, 'grade', null, pick('grade-tiles', name), {},
    (style) => choices.withGrade(style, look, look === 'eyedropper' ? .5 : 1, look === 'eyedropper' ? '#1E6BD6' : style.grade.color));
  for (const [index, strength] of [.5, 1].entries()) add('grade strength ' + strength, 'grade', { 'grade.strength': strength }, press('grade-strength', index), { 'grade.look': 'film', 'grade.strength': .72 });
  for (const on of [false, true]) add('grade comparison ' + on, 'grade', { 'grade.slider': on }, press('grade-slider', on ? 1 : 0), { 'grade.look': 'film' });
  for (const [look, name] of [['paper', 'Paper'], ['minimal', 'Minimal'], ['dark', 'Dark'], ['bright', 'Bright']]) add('cards ' + name, 'cards', null,
    pick('card-style-tiles', name), {}, (style) => choices.withCardStyle(style, look));
  const kinds = initialState().card_kinds;
  for (const category of new Set(Object.values(kinds).map((info) => info.category))) for (const on of [false, true]) add('group ' + category + ' ' + on, 'cards', null,
    async (page) => { await openFold(page, 'card-group-' + category.toLowerCase()); await toggle('card-all-' + category.toLowerCase(), on)(page); }, {}, (style) => Object.entries(kinds).filter(([, info]) => info.category === category)
      .reduce((current, [kind, info]) => choices.withKind(current, kind, on, Boolean(info.opt_in)), style));
  for (const [kind, info] of Object.entries(kinds)) add('kind ' + kind, 'cards', null,
    async (page) => { await openFold(page, 'card-group-' + info.category.toLowerCase()); await toggle('kind-' + kind, Boolean(info.opt_in))(page); }, {},
    (style) => choices.withKind(style, kind, Boolean(info.opt_in), Boolean(info.opt_in)));
  add('caption 3D camera', 'captions', { 'captions.camera_3d': true }, pick('caption-look-tiles', '3D camera'));
  const captionSeed = { 'captions.camera_3d': true, 'captions.entrance': 'rise', 'captions.max_words_per_phrase': 1,
    'captions.max_characters_per_phrase': 31, 'captions.background_mode': 'band' };
  for (const [name, mode] of [['Plain', 'none'], ['Box', 'box'], ['Band', 'band']]) add('caption ' + name, 'captions',
    { 'captions.camera_3d': false, 'captions.background_mode': mode, 'captions.entrance': 'slide', 'captions.max_words_per_phrase': 3, 'captions.max_characters_per_phrase': 18 },
    pick('caption-look-tiles', name), captionSeed);
  add('caption One word', 'captions', { 'captions.camera_3d': false, 'captions.entrance': 'slide', 'captions.max_words_per_phrase': 1, 'captions.max_characters_per_phrase': 16 }, pick('caption-look-tiles', 'One word'), captionSeed);
  add('caption Rise', 'captions', { 'captions.camera_3d': false, 'captions.entrance': 'rise' }, pick('caption-look-tiles', 'Rise'), captionSeed);
  for (const [mode, name] of [['one', 'One word'], ['few', 'A few words'], ['phrase', 'Short phrase'], ['sentence', 'Full sentence']]) add('words ' + mode, 'captions',
    { 'captions.max_words_per_phrase': choices.wordModes[mode].words, 'captions.max_characters_per_phrase': choices.wordModes[mode].chars }, pick('word-tiles', name));
  add('caption Giant', 'captions', { 'captions.highlight_style': 'giant' }, pick('caption-look-tiles', 'Giant'));
  for (const [index, look] of schema.properties.captions.properties.highlight_style.enum.filter((look) => look !== 'giant').entries()) add('highlight ' + look, 'captions',
    { 'captions.highlight_style': look, ...(look === 'none' ? { 'captions.italic_stress': false } : {}) },
    async (page) => { await openFold(page, 'highlight-fold'); await page.$('highlight-tiles').children[index].click(); await flush(); });
  for (const [index, [position, top]] of [['bottom', 1500], ['middle', 1180], ['middle', 700]].entries()) add('position ' + index, 'captions', { 'captions.position': position, 'captions.top_px': top }, press('placements', index));
  for (const [index, size] of [68, 92, 118].entries()) add('size ' + size, 'captions', { 'captions.size_px': size }, press('sizes', index));
  for (const on of [false, true]) {
    add('face ' + on, 'captions', { 'captions.placement': on ? 'avoid_face' : 'fixed' }, toggle('avoid-face', on));
    add('italics ' + on, 'captions', { 'captions.italic_stress': on }, press('italics', on ? 0 : 1));
  }
  for (const motion of choices.motions) add('motion ' + motion.name, 'captions',
    { 'captions.entry_rise_px': motion.rise, 'captions.entry_ms': motion.ms, 'captions.entrance': 'slide' },
    press('motion', choices.motions.indexOf(motion)), { 'captions.entrance': 'rise' });
  for (const [index, value] of [.5, .75, .9].entries()) add('background strength ' + value, 'captions',
    { 'captions.background_opacity': value }, press('background-opacity', index), { 'captions.background_mode': 'box' });
  for (const [index, value] of [0, 12, 40].entries()) add('background corners ' + value, 'captions',
    { 'captions.background_radius_px': value }, press('background-corners', index), { 'captions.background_mode': 'box' });
  for (const [field, value] of Object.entries({ size_px: 103, width_scale: .79, max_words_per_phrase: 4, max_characters_per_phrase: 29,
    top_px: 820, background_opacity: .63, background_padding_px: 27, background_radius_px: 23 })) add('fine ' + field, 'captions',
    { ['captions.' + field]: value }, input('fine-' + field, value), { 'captions.background_mode': 'box' });
  add('background hex', 'captions', { 'captions.background_color': '#AABBCC' }, input('background-color', '#abc', 'change'), { 'captions.background_mode': 'box' });
  for (const [layout, label] of [['full', 'Full screen'], ['mixed', 'Mixed'], ['box', 'In a box'], ['split', 'Split']]) add('layout ' + layout, 'broll',
    Object.fromEntries(['screen_recording', 'clip', 'photo', 'screenshot'].map((kind, index) =>
      ['broll.' + kind + '_placement', layout === 'split' ? 'split' : layout === 'full' || (layout === 'mixed' && index < 2) ? 'full_frame' : 'overlay'])), pick('broll-layout-tiles', label));
  // Only placements with a button on the page; green_screen has none yet.
  for (const kind of ['screen_recording', 'clip', 'photo', 'screenshot']) for (const [index, value] of schema.properties.broll.properties[kind + '_placement'].enum.filter((value) => ['full_frame', 'overlay', 'split'].includes(value)).entries()) add(kind + ' ' + value, 'broll',
    { ['broll.' + kind + '_placement']: value }, press('broll-' + kind, index));
  for (const [field, id, values] of [['overlay_box', 'broll-overlay-box', ['headroom', 'side_left', 'side_right', 'tall_behind']],
    ['stock_clips', 'broll-stock-clips', ['off', 'ask']]]) for (const [index, value] of values.entries()) add(field + ' ' + value, 'broll',
      { ['broll.' + field]: value }, press(id, index));
  for (const [field, id, values] of [['overlay_motion', 'broll-overlay-motion', [['none', 'Still'], ['drift', 'Drift'], ['scroll', 'Scroll']]],
    ['still_push', 'broll-still-push', [['in', 'Push in'], ['out', 'Pull out'], ['none', 'Off']]]]) for (const [value, label] of values) add(field + ' ' + value, 'broll',
      { ['broll.' + field]: value }, pick(id, label));
  for (const level of choices.musicLevels) add('music ' + level.name, 'sound', { 'music.level': level.value }, pick('music-level-tiles', level.name));
  add('exact music', 'sound', { 'music.level': .137 }, input('music', .137));
  for (const [index, value] of ['made', 'custom-kit', 'none'].entries()) add('effects ' + value, 'sound', { 'music.sound_effects': value }, press('sound-effects', index), { 'music.sound_effects': 'custom-kit' });
  for (const on of [false, true]) {
    add('sound fallback ' + on, 'sound', { 'music.sound_effects_fallback': on ? 'made' : 'none' },
      toggle('sound-fallback', on));
    add('beat ' + on, 'sound', { 'music.beat_snap': on }, toggle('beat-snap', on));
    add('bleep ' + on, 'sound', { 'music.bleep_swears': on }, toggle('bleep-swears', on));
    for (const field of ['repair_rumble', 'repair_room', 'repair_harshness']) add(field + ' ' + on, 'sound', { ['voice.' + field]: on }, toggle(field.replaceAll('_', '-'), on));
  }
  for (const [index, value] of ['none', 'clean', 'broadcast', 'warm'].entries()) add('voice ' + value, 'sound', { 'voice.preset': value }, press('voice-preset', index));
  for (const [index, value] of ['off', 'gentle', 'strong'].entries()) add('bass ' + value, 'sound', { 'voice.repair_proximity': value }, press('repair-proximity', index));
  for (const look of schema.properties.hook.properties.style.enum.filter((look) => !['none', 'text_hook'].includes(look))) add('opening ' + look, 'opening', { 'hook.style': look },
    async (page) => { const main = [page.state.style.hook.style, 'calm_card', 'headline_slam', 'shimmer', 'typewriter', 'video_letters', 'kinetic'];
      const id = main.includes(look) ? 'title-tiles' : 'more-title-tiles';
      await openFold(page, id); const index = schema.properties.hook.properties.style.enum.filter((value) => main.includes(look) ? main.includes(value) : !main.includes(value)).indexOf(look);
      if (main.includes(look)) {
        const order = [...new Set(main)];
        await page.$(id).children[order.indexOf(look)].click();
      } else await page.$(id).children[index].click();
      await flush();
      if (look === 'text_hook') assert.match(page.$('title-note').textContent, /whole video/);
    });
  for (const row of cases) await t.test(row.name, async () => {
    const state = initialState(); state.style = withFields(state.style, row.seed);
    const original = structuredClone(state.style), page = await harness(state, '#' + row.screen);
    await row.action(page);
    assert.deepEqual(state.style, original);
    await page.save();
    assert.deepEqual(savePayload(page), row.expected ? row.expected(original) : withFields(original, row.fields));
  });
});

test('sound effects off and on restores the opening kit after other sound choices', async () => {
  const state = initialState(); state.style.music.sound_effects = 'custom-kit';
  const original = structuredClone(state.style), page = await harness(state, '#sound');
  await pressControl(page, 'sound-effects', 2);
  await pressControl(page, 'voice-preset', 3);
  await pressControl(page, 'sound-effects', 0);
  await pressControl(page, 'sound-effects', 1);
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'voice.preset': 'warm' }));
});

test('Video in letters is a main opening choice and saves only the title style', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const page = await harness(state, '#opening', { layers: true });
  const labels = page.$('title-tiles').children.map((tile) => tile.children[1].textContent);
  assert.ok(labels.findIndex((name) => name.startsWith('Typewriter')) < labels.findIndex((name) => name.startsWith('Video in letters')));
  assert.ok(labels.some((name) => name.startsWith('Video in letters')));
  assert.ok(!labels.some((name) => name.startsWith('No title')));
  assert.ok(labels.some((name) => name.includes('A big word filled with your footage')));
  assert.equal(page.$('more-title-tiles').children.some((tile) => tile.textContent.includes('Video in letters')), false);
  await page.pick('title-tiles', 'Video in letters');
  assert.equal(page.$('screen-summary').textContent, 'Opening: Video in letters');
  assert.match(page.$('change-count').textContent, /Changed Opening/);
  const request = page.requests.filter(({ url, data }) => url === '/api/layers' && data.style.hook.style === 'video_letters').at(-1);
  assert.equal(request.data.tile.step, 'opening');
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'hook.style': 'video_letters' }));
});

test('Split fills the top half with the shot and keeps the face in the lower half', async () => {
  for (const face of [null, { x: 370, y: 510, w: 340, h: 460 }, { x: 700, y: 400, w: 180, h: 220 }]) {
    const state = initialState(); state.preview.face_boxes = face ? [face] : [];
    const page = await harness(state, '#broll', { phone: true });
    assert.deepEqual(page.$('broll-layout-tiles').children.map((tile) => tile.querySelectorAll('.tile-label')[0].text),
      ['Full screen', 'Mixed', 'In a box', 'Split']);
    const tile = page.$('slot-broll-split');
    assert.equal(tile.querySelectorAll('iframe').length, 0);
    assert.equal(tile.querySelectorAll('.split-shot').length, 1);
    assert.equal(tile.querySelectorAll('.split-seam').length, 1);
    assert.equal(tile.querySelectorAll('.drawn-picture')[0].style.transform, `scale(${200 / 1080})`);
    const photo = tile.querySelectorAll('.split-speaker')[0].querySelectorAll('img')[0];
    const scale = Number(photo.style.transform.match(/scale\(([^)]+)\)/)[1]);
    if (face) {
      const top = face.y * scale + parseFloat(photo.style.top);
      const bottom = (face.y + face.h) * scale + parseFloat(photo.style.top);
      assert.ok(top >= 24 && bottom <= 960, JSON.stringify({ top, bottom, face }));
    } else assert.equal(parseFloat(photo.style.top), 0);
    await page.pick('broll-layout-tiles', 'Split');
    assert.equal(page.$('slot-broll-split').getAttribute('aria-pressed'), 'true');
    assert.equal(page.$('broll-kinds-fold-value').textContent, 'Split');
    assert.equal(page.$('main-frame').querySelectorAll('.split-shot').length, 1);
    assert.equal(page.$('main-frame').querySelectorAll('.poster-caption').length, 1);
    await page.go('more');
    assert.match(page.$('choice-broll').textContent, /Split/);
    assert.equal(page.$('choice-broll').querySelectorAll('.split-speaker').length, 1);
    assert.equal(page.$('choice-broll').querySelectorAll('.poster-caption').length, 0);
  }
  const desktop = await harness(initialState(), '#broll');
  assert.equal(desktop.$('slot-broll-split').querySelectorAll('.drawn-picture')[0].style.transform, `scale(${200 / 1080})`);
});

test('Split controls follow each placement enum and disappear from an older schema', async () => {
  const state = initialState(); state.schema = structuredClone(schema);
  state.schema.properties.broll.properties.clip_placement.enum = ['full_frame', 'overlay'];
  let page = await harness(state, '#broll');
  assert.equal(page.$('broll-photo').children.length, 3);
  assert.equal(page.$('broll-clip').children.length, 2);
  for (const kind of ['screen_recording', 'photo', 'screenshot']) {
    state.schema.properties.broll.properties[kind + '_placement'].enum = ['full_frame', 'overlay'];
  }
  page = await harness(state, '#broll');
  assert.equal(page.$('slot-broll-split'), undefined);
  for (const kind of ['screen_recording', 'clip', 'photo', 'screenshot']) assert.equal(page.$('broll-' + kind).children.length, 2);
});

test('Match color writes only its field and its preview follows the switch and the take color', async () => {
  for (const on of [false, true]) {
    const state = initialState(); state.style.broll.color_match = on ? 'off' : 'on';
    const original = structuredClone(state.style), page = await harness(state, '#broll');
    const fold = 'slot-fold-broll-color-match', id = 'slot-switch-broll-color-match';
    assert.equal(page.$(fold).open, false);
    assert.equal(page.$(fold + '-value').textContent, on ? 'Off' : 'On');
    assert.match(page.$(fold).textContent, /Your face is never touched. On by default./);
    await openFold(page, id);
    page.$(id).checked = on; await page.$(id).fire('change'); await flush();
    assert.equal(page.$(fold).open, true);
    assert.equal(page.$(fold + '-value').textContent, on ? 'On' : 'Off');
    const tints = page.$(fold).querySelectorAll('.shot-tint');
    assert.equal(tints.length, on ? 1 : 0);
    if (on) {
      assert.equal(tints[0].style.backgroundColor, '#1E6BD6');
      assert.equal(tints[0].parentElement.className, 'shot-full');
    }
    assert.equal(page.$(fold).querySelectorAll('iframe').length, 0);
    assert.match(page.$('change-count').textContent, /Changed B-roll/);
    await page.go('more'); assert.equal(page.$('choice-broll').textContent.includes('color match off'), !on);
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'broll.color_match': on ? 'on' : 'off' }));
  }
  for (const color of [undefined, '#oops']) {
    const state = initialState(); state.preview.grade_color = { color };
    const page = await harness(state, '#broll');
    assert.equal(page.$('slot-fold-broll-color-match').querySelectorAll('.shot-tint').length, 0);
  }
});

test('visiting Split and Match color preserves missing, on and off fields without choices', async () => {
  for (const color of [undefined, 'on', 'off']) {
    const state = initialState();
    if (color === undefined) delete state.style.broll.color_match; else state.style.broll.color_match = color;
    const original = structuredClone(state.style), page = await harness(state, '#broll');
    await openFold(page, 'slot-switch-broll-color-match');
    assert.equal(page.$('slot-switch-broll-color-match').checked, color !== 'off');
    await openFold(page, 'broll-clip');
    await page.go('more'); await page.save();
    assert.deepEqual(savePayload(page), original);
  }
});

test('the B-roll letters switch saves only its array and reports its state', async () => {
  for (const on of [true, false]) {
    const state = initialState(); state.style.broll.emphasis_on = on ? [] : ['video_letters'];
    const original = structuredClone(state.style), page = await harness(state, '#broll');
    const fold = 'slot-fold-broll-video-letters', id = 'slot-switch-broll-video-letters';
    assert.equal(page.$(fold).open, false);
    assert.equal(page.$(fold + '-value').textContent, on ? 'Off' : 'On');
    await openFold(page, id);
    assert.equal(page.$(fold).querySelectorAll('.letters-word')[0].textContent, 'Your');
    assert.equal(page.$(fold).querySelectorAll('iframe').length, 0);
    page.$(id).checked = on; await page.$(id).fire('change'); await flush();
    assert.equal(page.$(fold + '-value').textContent, on ? 'On' : 'Off');
    assert.equal(page.$(fold).open, true);
    assert.match(page.$('change-count').textContent, /Changed B-roll/);
    assert.equal(page.$('screen-summary').textContent.includes('video in letters on'), on);
    await page.go('more'); assert.equal(page.$('choice-broll').textContent.includes('video in letters on'), on);
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'broll.emphasis_on': on ? ['video_letters'] : [] }));
  }
});

test('visiting the letters fold preserves absent, empty and enabled emphasis fields', async () => {
  for (const emphasis of [undefined, [], ['video_letters']]) {
    const state = initialState();
    if (emphasis === undefined) delete state.style.broll.emphasis_on;
    else state.style.broll.emphasis_on = emphasis;
    const original = structuredClone(state.style), page = await harness(state, '#broll');
    await openFold(page, 'slot-switch-broll-video-letters');
    await page.go('opening'); await page.save();
    assert.deepEqual(savePayload(page), original);
  }
});

test('switch slot groups reveal later options only while enabled', async () => {
  choices.brollMoreSlots.push({ id: 'test-switch', name: 'Test switch', type: 'switch', note: 'Synthetic choice',
    apply: (style, on) => choices.withBeatSnap(style, on), selected: (style) => style.music.beat_snap,
    options: [{ id: 'test-clean', name: 'Clean', note: '', apply: (style) => choices.withVoicePreset(style, 'clean'),
      selected: (style) => style.voice.preset === 'clean' }] });
  try {
    const page = await harness(initialState(), '#broll'), id = 'slot-switch-broll-test-switch';
    assert.equal(page.$('slot-broll-test-clean'), undefined);
    await openFold(page, id); page.$(id).checked = true; await page.$(id).fire('change'); await flush();
    assert.ok(page.$('slot-broll-test-clean'));
    page.$(id).checked = false; await page.$(id).fire('change'); await flush();
    assert.equal(page.$('slot-broll-test-clean'), undefined);
  } finally { choices.brollMoreSlots.pop(); }
});

// Browser checks use the shipped page and synthetic media through a fake engine transport.
test('opening push in Moves More has a drawn preview and saves only its switch', async () => {
  for (const phone of [false, true]) {
    const state = initialState(), original = structuredClone(state.style);
    const page = await harness(state, '#moves', { phone });
    await openFold(page, 'moves-more');
    await openFold(page, 'slot-fold-moves-opening-push');
    const fold = page.$('slot-fold-moves-opening-push');
    assert.equal(fold.querySelectorAll('.zoom-opening').length, 1);
    const zoom = fold.querySelectorAll('.zoom-opening')[0];
    assert.equal(zoom.style['--punch-scale'], '1.3');
    assert.equal(fold.querySelectorAll('.zoom-tag').length, 0);
    const toggle = page.$('slot-switch-moves-opening-push');
    assert.equal(toggle.checked, false);
    toggle.checked = true; await toggle.fire('change'); await flush();
    assert.ok(page.$('screen-summary').textContent.endsWith(' · opening push'));
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, { 'moves.opening_push': true }));
  }
});

test('Chromium walks every screen at phone and desktop sizes, keeps tiles clear and saves the exact style',
  { skip: !process.env.YAP_STUDIO_PUPPETEER || !process.env.YAP_STUDIO_CHROME }, async () => {
    const { createRequire } = await import('node:module');
    const { mkdir } = await import('node:fs/promises');
    const { join } = await import('node:path');
    const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
    const browser = await puppeteer.launch({ executablePath: process.env.YAP_STUDIO_CHROME,
      pipe: true, headless: true, args: ['--no-sandbox'],
      ...(process.env.YAP_STUDIO_BROWSER_PROFILE ? { userDataDir: process.env.YAP_STUDIO_BROWSER_PROFILE } : {}),
    });
    const shots = process.env.YAP_STUDIO_FLOW_SHOTS;
    if (shots) await mkdir(shots, { recursive: true });
    try {
      for (const [width, height, mobile] of [[390, 844, true], [1440, 900, false]]) {
        const page = await browser.newPage(), errors = [], saved = [], replacements = [];
        // Motion is the default here; hosted runners ask for reduced motion at the OS level.
        const motion = [{ name: 'prefers-reduced-motion', value: 'no-preference' }];
        await page.emulateMediaFeatures(motion);
        page.on('pageerror', (error) => errors.push(error.message));
        await page.setViewport({ width, height, isMobile: mobile, hasTouch: mobile });
        const state = initialState(), original = structuredClone(state.style);
        state.preview.sentence = 'Shit hits the fan.';
        const expected = choices.withOpeningPush(choices.withKind(choices.withTransition(choices.withPunchScale(choices.withZoomMode(choices.withMoment(
          choices.withEmphasis(choices.withGrade(original, 'film'), 'video_letters', true), 'color_pop', true), 'punches_and_push'), 2), 'light_leak', false), 'big_number', true, true), true);
        expected.captions.highlight_style = 'giant';
        expected.hook.style = 'kinetic';
        expected.chapters.style = 'title_card';
        expected.graphics.card_kinds_on.push('gravity_drop');
        expected.moves.jump_cuts = 'step_in';
        Object.assign(expected.broll, choices.withBrollLayout(original, 'split', schema).broll, {
          color_match: 'off', emphasis_on: ['video_letters'],
        });
        const cutout = Buffer.from((await page.evaluate(() => {
          const canvas = document.createElement('canvas'); canvas.width = 54; canvas.height = 96;
          const ctx = canvas.getContext('2d'); ctx.fillStyle = '#58B899';
          ctx.beginPath(); ctx.ellipse(27, 29, 10, 14, 0, 0, Math.PI * 2); ctx.fill();
          ctx.beginPath(); ctx.moveTo(7, 96); ctx.lineTo(10, 60); ctx.quadraticCurveTo(12, 43, 27, 43);
          ctx.quadraticCurveTo(42, 43, 44, 60); ctx.lineTo(47, 96); ctx.closePath(); ctx.fill();
          return canvas.toDataURL('image/png').split(',')[1];
        })), 'base64');
        const files = ['instrument-serif-regular.woff2', 'gelasio-variable.woff2', 'inter-variable.woff2',
          'geist-variable.woff2', 'alata-regular.woff2', 'archivo-black-regular.woff2', 'anton-sc-regular.woff2',
          'titan-one-regular.woff2', 'permanent-marker-regular.woff2', 'space-mono-regular.woff2'];
        const fontCSS = bundledFonts.map((name, index) => `@font-face {font-family: "Yap Studio ${name}";src:url("/fonts/${files[index]}") format("woff2");font-style:normal;font-weight:400;font-display:block;}`).join('\n');
        await page.setRequestInterception(true);
        page.on('request', async (request) => {
          const pathname = new URL(request.url()).pathname;
          const assets = { '/': ['text/html', html],
            '/static/pick.css': ['text/css', readFileSync(new URL('src/yap_studio/gallery/static/pick.css', root), 'utf8')],
            '/static/fonts.css': ['text/css', fontCSS],
            '/static/pick.js': ['text/javascript', readFileSync(new URL('src/yap_studio/gallery/static/pick.js', root), 'utf8')],
            '/static/choices.mjs': ['text/javascript', readFileSync(new URL('src/yap_studio/gallery/static/choices.mjs', root), 'utf8')],
            '/static/shot.jpg': ['image/jpeg', readFileSync(new URL('src/yap_studio/gallery/static/shot.jpg', root))],
            '/cutout': ['image/png', cutout],
            '/still': ['image/svg+xml', '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1920"><rect width="1080" height="1920" fill="#aab8c2"/></svg>'] };
          if (pathname.startsWith('/fonts/') && files.includes(pathname.slice(7))) {
            await request.respond({ status: 200, contentType: 'font/woff2', body: readFileSync(new URL(`src/yap_studio/fonts/${pathname.slice(7)}`, root)) });
          } else if (assets[pathname]) {
            const [contentType, body] = assets[pathname];
            await request.respond({ status: 200, contentType, body,
              ...(pathname === '/' ? { headers: { 'Content-Security-Policy': "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; font-src 'self'" } } : {}) });
          } else {
            const submitted = pathname === '/api/save' ? JSON.parse(request.postData()) : null;
            if (submitted) { saved.push(submitted.style); replacements.push(submitted.replace); }
            const body = pathname === '/api/state' ? state : pathname === '/api/preview' ? state.preview
              : submitted ? { name: submitted.name, id: 'saved' } : { frame: { width: 1080, height: 1920 }, layers: [] };
            await request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
          }
        });
        await page.goto('http://gallery/?t=token');
        await page.waitForFunction(() => document.getElementById('style-name').value === 'My style');
        const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.click('#customize'); await page.click('#step-more'); await page.click('#choice-sound');
        await page.$eval('#sound-more', (fold) => fold.open = true);
        await page.$eval('#effects-fold', (fold) => fold.open = true);
        assert.deepEqual(await page.$$eval('#sound-extras input', (inputs) =>
          inputs.map((input) => input.checked)), [false, false, false]);
        assert.equal(await page.$eval('#sound-fallback', (input) => input.checked), true);
        await page.$eval('#sound-extra-typing', (input) => input.click());
        await page.$eval('#sound-extras', (group) => group.scrollIntoView({ block: 'center' }));
        await settle();
        if (shots) await page.screenshot({ path: join(shots, `${width}-sound-extras.png`) });
        await page.click('#save'); await page.click('#confirm-save');
        await page.waitForFunction(() => !document.getElementById('saved').hidden);
        assert.deepEqual(saved, [choices.withSoundExtra(original, 'typing', true)]);
        await page.evaluate(() => sessionStorage.clear());
        await page.goto('http://gallery/?t=token');
        await page.waitForFunction(() => document.getElementById('style-name').value === 'My style');
        saved.length = 0; replacements.length = 0;
        const visit = async (screen, grid, fold) => {
          await page.waitForFunction((screen) => document.body.dataset.screen === screen, {}, screen)
            .catch(async (error) => { throw new Error(`Expected ${screen}, saw ${await page.$eval('body', body => body.dataset.screen)}: ${error.message}`); });
          await page.evaluate(() => window.scrollTo(0, 0)); await settle();
          await page.evaluate(() => document.fonts.ready);
          await page.$eval(`#${grid} > :first-child`, (tile) => tile.scrollIntoView({ block: 'start' })); await settle();
          await page.evaluate((grid) => {
            const tile = document.getElementById(grid).children[0], viewer = document.querySelector('.sticky-preview');
            const r = tile.getBoundingClientRect(), v = viewer.getBoundingClientRect();
            const overlaps = r.left < v.right && r.right > v.left && getComputedStyle(viewer).position === 'sticky';
            const topbar = document.querySelector('.topbar').getBoundingClientRect();
            window.scrollBy(0, r.top - (overlaps ? v.bottom + 12 : Math.max(0, topbar.bottom) + 12));
          }, grid); await settle();
          const metrics = await page.evaluate((grid) => {
            const tile = document.getElementById(grid).children[0], r = tile.getBoundingClientRect();
            const frame = tile.querySelector('.tile-frame'), f = frame?.getBoundingClientRect();
            const label = (tile.querySelector('.tile-label') || tile).getBoundingClientRect();
            const viewer = document.querySelector('.sticky-preview'), v = viewer.getBoundingClientRect();
            const overlaps = r.left < v.right && r.right > v.left && getComputedStyle(viewer).position === 'sticky';
            return { page: document.documentElement.scrollWidth, width: innerWidth,
              bar: document.getElementById('savebar').getBoundingClientRect().top,
              previewBottom: overlaps ? v.bottom : 0,
              panels: Array.from(document.querySelectorAll('.step-panel')).filter((panel) => !panel.hidden).map((panel) => panel.id),
              tile: { top: r.top, left: r.left, right: r.right, width: r.width }, labelBottom: label.bottom,
              frame: f ? { width: f.width, height: f.height } : null };
          }, grid);
          assert.equal(metrics.page, metrics.width, JSON.stringify({ screen, width, metrics }));
          assert.deepEqual(metrics.panels, [`panel-${screen}`]);
          assert.ok(metrics.tile.top >= metrics.previewBottom && metrics.labelBottom <= metrics.bar + 1, JSON.stringify({ screen, width, metrics }));
          assert.ok(metrics.tile.left >= 0 && metrics.tile.right <= width, JSON.stringify({ screen, width, metrics }));
          if (metrics.frame) {
            assert.ok(Math.abs(metrics.frame.width - (metrics.tile.width - 2)) < 2, JSON.stringify({ screen, width, metrics }));
            const ratio = mobile && screen === 'captions' ? 10 / 11 : 9 / 16;
            assert.ok(Math.abs(metrics.frame.width / metrics.frame.height - ratio) < .01, JSON.stringify({ screen, width, metrics }));
          }
          if (shots) await page.screenshot({ path: join(shots, `${width}-${screen}.png`) });
          if (fold) {
            const ancestors = await page.$eval(`#${fold}`, (details) => {
              const ids = []; for (let p = details; p; p = p.parentElement) if (p.tagName === 'DETAILS') ids.unshift(p.id); return ids;
            });
            for (const id of ancestors) {
              if (!await page.$eval(`#${id}`, (el) => el.open)) {
                await page.$eval(`#${id} > summary`, (el) => el.scrollIntoView({ block: 'center' }));
                await settle();
                await page.click(`#${id} > summary`);
                await page.waitForFunction((id) => document.getElementById(id).open, {}, id);
              }
            }
            assert.equal(await page.$eval(`#${fold}`, (el) => el.open), true);
            if (fold.startsWith('card-group-')) await page.waitForFunction((id) => document.getElementById(id).children.length > 1, {}, fold);
            const frames = await page.$$eval(`#${fold} .tile-frame`, (frames) => frames.map((frame) => {
              const r = frame.getBoundingClientRect();
              return { ratio: r.width / r.height, compact: frame.classList.contains('compact') };
            }));
            for (const frame of frames) assert.ok(Math.abs(frame.ratio - (frame.compact ? 16 / 7 : 9 / 16)) < .01, JSON.stringify({ screen, frame }));
            await settle();
            await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight)); await settle();
            const clearance = await page.evaluate(() => ({
              panel: document.querySelector('.step-panel:not([hidden])').getBoundingClientRect().bottom,
              bar: document.getElementById('savebar').getBoundingClientRect().top,
            }));
            assert.ok(clearance.panel <= clearance.bar + 1, JSON.stringify({ screen, clearance }));
          }
        };
        await visit('looks', 'start-options');
        assert.equal(await page.$$eval('#more-choices iframe', (frames) => frames.length), 0);
        assert.equal(await page.$$eval('#more-choices .poster-caption', (captions) => captions.length), 0);
        await page.click('#customize'); await visit('font', 'font-options', 'graphics-font-fold');
        const fonts = await page.$$eval('#font-options button', (buttons) => buttons.map((button) =>
          ({ name: button.textContent, font: getComputedStyle(button).fontFamily })));
        assert.deepEqual(fonts.map(({ name }) => name), bundledFonts);
        assert.ok(fonts.every(({ name, font }) => font.includes(`Yap Studio ${name}`)));
        await page.click('#step-captions'); await visit('captions', 'caption-look-tiles', 'highlight-fold');
        await page.$eval('#slot-captions-camera_3d', (tile) => tile.click()); await settle();
        await page.waitForFunction(() => document.getElementById('main-frame')._drawn?.screen === 'camera_3d');
        assert.equal(await page.$eval('#slot-captions-camera_3d', (tile) => tile.getAttribute('aria-pressed')), 'true');
        assert.equal(await page.$eval('#track-row', (row) => row.hidden), false);
        assert.equal(await page.$eval('#main-frame', (host) => host._drawn.element.classList.contains('is-playing')), false);
        await page.focus('#play'); await page.keyboard.press('Enter');
        // Play and pause apply asynchronously; a slow machine needs the wait.
        await page.waitForFunction(() => document.getElementById('main-frame')._drawn.element.classList.contains('is-playing'));
        await page.keyboard.press('Space');
        await page.waitForFunction(() => !document.getElementById('main-frame')._drawn.element.classList.contains('is-playing'));
        assert.equal(await page.$eval('#play', (button) => button.getAttribute('aria-label')), 'Play preview');
        await page.keyboard.press('Space');
        for (const selector of ['#slot-captions-camera_3d .tile-frame', '#main-frame']) {
          await page.$eval(selector, (host) => host.scrollIntoView({ block: 'center' })); await settle();
          if (!mobile && selector !== '#main-frame') await page.hover(selector);
          await page.waitForFunction((selector) => {
            const host = document.querySelector(selector);
            return host._drawn.element.classList.contains('is-playing')
              && ['.camera3d-scene', '.camera3d-front'].every((selector) => host.querySelector(selector)?.getAnimations()
                .some((animation) => animation.animationName === 'camera3d-push'
                  && animation.playState === 'running' && !animation.pending));
          }, {}, selector);
          const drawing = await page.$eval(selector, (host) => ({
            words: host.querySelectorAll('.camera3d-words').length,
            behind: host.querySelectorAll('.camera3d-behind').length,
            flat: host.querySelectorAll('iframe,.poster-caption').length,
            animations: host.getAnimations({ subtree: true }).length,
          }));
          assert.deepEqual(drawing, { words: 2, behind: 0, flat: 0, animations: 2 });
          const depth = await page.$eval(selector, async (host) => {
            const planes = () => ['.camera3d-scene', '.camera3d-front'].map((selector) => host.querySelector(selector));
            const animationsOf = (targets) => targets.map((target) => target?.getAnimations()
              .find((animation) => animation.animationName === 'camera3d-push'));
            const frame = () => new Promise(requestAnimationFrame);
            const seek = async (time) => {
              let last;
              // Scroll callbacks and cutout redraws can resume or replace the sampled animations.
              for (let attempt = 0; attempt < 12; attempt++) {
                const targets = planes(), animations = animationsOf(targets);
                if (host._drawn.element.classList.contains('is-playing') && animations.every(Boolean)) {
                  for (const animation of animations) { animation.pause(); animation.currentTime = time; }
                  await frame();
                  const current = planes(), applied = animationsOf(current);
                  last = applied.map((animation) => animation && ({ state: animation.playState,
                    pending: animation.pending, time: animation.currentTime }));
                  if (host._drawn.element.classList.contains('is-playing')
                    && current.every((target, i) => target === targets[i] && applied[i] === animations[i])
                    && animations.every((animation) => !animation.pending && animation.playState === 'paused'
                      && Math.abs(animation.currentTime - time) < .001)) return animations;
                }
                await frame();
              }
              throw new Error(`3D drawing did not hold ${time} ms: ${JSON.stringify(last)}`);
            };
            if (!host._drawn.main) {
              for (const time of [0, 3500, 7000]) {
                await seek(time);
                const tile = host.getBoundingClientRect(), picture = host.querySelector('.drawn-still').getBoundingClientRect();
                if (picture.left > tile.left + 1 || picture.right < tile.right - 1
                  || picture.top > tile.top + 1 || picture.bottom < tile.bottom - 1) {
                  throw new Error(`3D tile has empty bands: ${JSON.stringify({ time, tile: tile.toJSON(), picture: picture.toJSON() })}`);
                }
              }
            }
            const animations = await seek(3500);
            const scene = host.querySelector('.camera3d-scene'), front = host.querySelector('.camera3d-front');
            const backScale = new DOMMatrix(getComputedStyle(scene).transform).a;
            const frontScale = new DOMMatrix(getComputedStyle(front).transform).a;
            const r = scene.getBoundingClientRect();
            const face = { left: r.left + r.width / 3, right: r.left + r.width * 2 / 3,
              top: r.top + r.height * 420 / 1920, bottom: r.top + r.height * 840 / 1920 };
            const clear = [...host.querySelectorAll('.camera3d-words')].every((line) => {
              const box = line.getBoundingClientRect();
              return box.bottom < face.top || box.top > face.bottom || box.right < face.left || box.left > face.right;
            });
            for (const animation of animations) animation.play();
            return { backScale, frontScale, clear };
          });
          assert.ok(depth.frontScale > depth.backScale && depth.backScale > 1, JSON.stringify(depth));
          assert.equal(depth.clear, true);
        }
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        if (shots) await page.screenshot({ path: join(shots, `${width}-camera3d-front.png`) });
        await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
        assert.equal(await page.$eval('#main-frame .camera3d-scene', (scene) => getComputedStyle(scene).animationName), 'none');
        await page.emulateMediaFeatures(motion);
        await page.$$eval('#caption-look-tiles button', (tiles) => tiles.find((tile) => tile.querySelector('.tile-label').textContent.startsWith('Plain')).click());
        assert.equal(await page.$$eval('#highlight-tiles .tile-label', (labels) => labels.some((label) => label.textContent.includes('Giant'))), false);
        await page.click('#step-colors'); await visit('colors', 'palette-tiles', 'colors-more');
        await page.$eval('#colors-more', (fold) => { fold.open = false; });
        const ownColors = (await page.$$('#palette-tiles button')).at(-1);
        await ownColors.scrollIntoView(); await ownColors.click(); await settle();
        assert.equal(await page.$eval('#colors-more', (fold) => fold.open), true);
        assert.equal(await page.evaluate(() => document.activeElement?.tagName === 'INPUT'), false);
        await page.click('#reset'); await settle();
        assert.equal(await page.$('#colors-fold'), null);
        assert.ok(await page.$eval('#color-accent', (input) => input.getBoundingClientRect().height > 0 && !input.disabled));
        const caption = await page.$eval('#color-caption', (input) => input.value);
        await page.$eval('#color-caption', (input) => { input.value = '#114488'; input.dispatchEvent(new Event('change', { bubbles: true })); });
        assert.equal(await page.$eval('#color-caption', (input) => input.value), '#114488');
        await page.$eval('#color-caption', (input, value) => { input.value = value; input.dispatchEvent(new Event('change', { bubbles: true })); }, caption);
        if (shots) await page.screenshot({ path: join(shots, `${width}-colors-more.png`) });
        await page.click('#step-captions');
        await page.$eval('#slot-captions-giant', (tile) => tile.click());
        await page.waitForFunction(() => document.getElementById('slot-captions-giant').getAttribute('aria-pressed') === 'true');
        for (const look of ['camera_3d', 'giant']) {
          await page.$eval(`#slot-captions-${look}`, (tile) => tile.click());
          await page.waitForFunction((look) => document.getElementById(`slot-captions-${look}`).getAttribute('aria-pressed') === 'true', {}, look);
          assert.deepEqual(await page.$$eval('#caption-look-tiles [aria-pressed="true"]',
            (tiles) => tiles.map((tile) => tile.id)), [`slot-captions-${look}`]);
        }
        await page.click('#step-more'); await visit('more', 'more-choices');
        assert.equal(await page.$$eval('#more-choices iframe', (frames) => frames.length), 0);
        assert.ok(await page.$('#choice-moves'));
        assert.ok(await page.$('#choice-moments'));
        for (const [screen, grid, fold] of [['grade', 'grade-tiles', null], ['opening', 'title-tiles', 'more-titles-fold'],
          ['cards', 'card-style-tiles', 'card-group-words'], ['broll', 'broll-layout-tiles', 'broll-stock-fold'],
          ['moves', 'moves-tiles', null], ['moments', 'moments-tiles', null], ['sound', 'music-level-tiles', 'voice-fold']]) {
          await page.evaluate(() => window.scrollTo(0, 0)); await settle();
          await page.click('#step-more');
          await page.$eval(`#choice-${screen}`, row => row.scrollIntoView({block: 'center'}));
          await settle();
          await page.click(`#choice-${screen}`);
          await visit(screen, grid, fold);
          if (screen === 'opening') {
            await page.$$eval('#title-tiles button', (tiles) => tiles.find((tile) => tile.querySelector('.tile-label').textContent.startsWith('Video in letters')).click());
            await page.waitForFunction(() => document.getElementById('screen-summary').textContent === 'Opening: Video in letters');
            assert.equal(await page.$$eval('#more-title-tiles .tile-label', (labels) => labels.some((label) => label.textContent.includes('Kinetic hook'))), false);
            await page.$eval('#slot-opening-kinetic', (tile) => tile.click());
            await page.$eval('#slot-fold-opening-chapter', (fold) => fold.open = true);
            assert.equal(await page.$eval('#slot-switch-opening-chapter', (input) => input.checked), false);
            for (const on of [true, false, true]) {
              await page.$eval('#slot-switch-opening-chapter', (input) => input.click());
              await page.waitForFunction((on) => document.getElementById('screen-summary').textContent
                === 'Opening: Kinetic hook' + (on ? ' · chapter cards on' : ''), {}, on);
            }
          }
          if (screen === 'cards') {
            assert.equal(await page.$eval('#kind-gravity_drop', (input) => input.checked), false);
            assert.equal(await page.$eval('#kind-gravity_drop', (input) => input.getAttribute('aria-label')), 'Gravity drop on');
            assert.equal(await page.$$eval('#card-group-words .new-tag', (tags) => tags.length), 1);
            await page.$eval('#kind-gravity_drop', (input) => input.click());
            assert.equal(await page.$eval('#card-group-video > summary > span:first-child', (span) => span.textContent), 'Video cards · 1 kind');
            await page.$eval('#card-group-numbers > summary', (summary) => summary.scrollIntoView({ block: 'center' })); await settle();
            await page.click('#card-group-numbers > summary');
            await page.waitForSelector('#kind-big_number');
            assert.equal(await page.$eval('#card-group-numbers > summary', (summary) => summary.textContent.includes('Numbers · 4 kinds')), true);
            assert.equal(await page.$$eval('#card-group-numbers .new-tag', (tags) => tags.length), 1);
            assert.equal(await page.$eval('#kind-big_number', (input) => input.checked), false);
            assert.equal(await page.$eval('#kind-big_number', (input) => input.previousElementSibling.querySelector('.kind-note').textContent), 'Counts up behind your head');
            await page.$eval('#kind-big_number', (input) => input.click());
            await page.waitForFunction(() => document.getElementById('kind-big_number').checked);
            await page.$eval('#slot-fold-cards-specs-rows > summary', (summary) => summary.scrollIntoView({ block: 'center' }));
            await page.click('#slot-fold-cards-specs-rows > summary');
            await page.waitForFunction(() => document.getElementById('slot-fold-cards-specs-rows').open); await settle();
            assert.deepEqual(await page.$$eval('#slot-fold-cards-specs-rows .tile-label', (labels) => labels.map((label) => label.firstChild.textContent)),
              ['Together', 'Rows on cue', 'Rows on cue, glowing']);
            await page.$eval('#slot-cards-on_cue_glow', (tile) => tile.click());
            await page.waitForFunction(() => document.getElementById('screen-summary').textContent.endsWith(' · rows on cue, glowing'));
            assert.equal(await page.$eval('#slot-fold-cards-specs-rows', (fold) => fold.open), true);
            await page.$eval('#slot-fold-cards-specs-rows', (fold) => fold.scrollIntoView({ block: 'center' })); await settle();
            if (shots) await page.screenshot({ path: join(shots, `${width}-cards-rows.png`) });
            await page.$eval('#slot-cards-together', (tile) => tile.click());
          }
          if (screen === 'moves') {
            assert.equal(await page.$eval('#moves-more', (fold) => fold.hidden), false);
            for (const id of ['moves-more', 'slot-fold-moves-jump-cuts']) {
              await page.$eval(`#${id} > summary`, (el) => el.scrollIntoView({ block: 'center' }));
              await settle();
              await page.click(`#${id} > summary`);
              await page.waitForFunction((id) => document.getElementById(id).open, { timeout: 5000 }, id);
            }
            assert.equal(await page.$$eval('#slot-fold-moves-jump-cuts button', (tiles) => tiles.length), 4);
            assert.equal(await page.$eval('#slot-fold-moves-jump-cuts .fold-value', (value) => value.textContent), 'Hard cut');
            await page.$eval('#slot-moves-step_in', (tile) => tile.click());
            await settle();
            for (const id of ['punch', 'step_in']) {
              const selector = `#slot-moves-${id} .stage-host`;
              await page.$eval(selector, (host) => host.scrollIntoView({ block: 'center' })); await settle();
              if (!mobile) await page.hover(selector);
              await page.waitForFunction((selector) => document.querySelector(selector)._drawn.element.classList.contains('is-playing'), {}, selector);
              const scales = await page.$eval(selector, async (host, id) => {
                const zoom = host.querySelector('.zoom-picture'), animation = zoom.getAnimations()[0];
                animation.pause();
                const result = [];
                for (const time of id === 'step_in' ? [0, 599, 601, 1199, 1201, 1799, 1801, 2399, 2401, 2999] : [0, 599, 601, 1199, 1201, 1799]) {
                  animation.currentTime = time; await new Promise(requestAnimationFrame);
                  result.push(new DOMMatrix(getComputedStyle(zoom).transform).a);
                }
                return result;
              }, id);
              assert.deepEqual(scales, id === 'step_in' ? [1, 1, 1.15, 1.15, 1.3, 1.3, 1.45, 1.45, 1, 1] : [1, 1, 1.5, 1.5, 1, 1]);
            }
            if (shots) {
              await page.evaluate(() => window.scrollTo(0, 0)); await settle();
              await page.screenshot({ path: join(shots, `${width}-moves-jump-cuts.png`), fullPage: true });
            }
            await page.$eval('#slot-fold-moves-jump-cuts', (fold) => fold.open = false);
            await page.$eval('#moves-more', (fold) => fold.open = false);
            assert.equal(await page.$('#slot-fold-moves-zoom-strength'), null);
            for (const id of ['moves-more', 'slot-fold-moves-transitions']) {
              await page.$eval(`#${id} > summary`, (summary) => summary.scrollIntoView({ block: 'center' })); await settle();
              await page.click(`#${id} > summary`);
              await page.waitForFunction((id) => document.getElementById(id).open, {}, id);
            }
            assert.deepEqual(await page.$$eval('#slot-fold-moves-transitions input', (inputs) => inputs.map((input) => input.checked)), [true, true, true]);
            assert.equal(await page.$$eval('#moves-tiles iframe', (frames) => frames.length), 0);
            assert.equal(await page.$eval('#slot-moves-none .tile-frame', (frame) => Boolean(frame.querySelector('.zoom-picture'))), false);
            for (const mode of ['punches', 'punches_and_push']) {
              await page.$eval(`#slot-moves-${mode}`, (tile) => tile.click());
              await page.waitForFunction((mode) => document.querySelector(`#main-frame .zoom-${mode}`), {}, mode);
              const samples = await page.evaluate(async (mode) => {
                const zoom = document.querySelector('#main-frame .zoom-picture'), animation = zoom.getAnimations()[0];
                animation.pause();
                const times = mode === 'punches' ? [0, 1000 / 30, 2000 / 30, 1500, 2900, 2999]
                  : [0, 2000 / 30, 1500, 3000, 3500, 5500, 5999];
                const scales = [];
                for (const time of times) {
                  animation.currentTime = time;
                  await new Promise(requestAnimationFrame);
                  scales.push(new DOMMatrix(getComputedStyle(zoom).transform).a);
                }
                return { scales, captionZoomed: zoom.contains(document.querySelector('#main-frame .poster-caption')) };
              }, mode);
              assert.equal(samples.captionZoomed, false);
              assert.ok(Math.abs(samples.scales[0] - 1) < .001, JSON.stringify(samples));
              if (mode === 'punches') {
                assert.ok(samples.scales[1] > 1.3 && samples.scales[1] < 1.5, JSON.stringify(samples));
                assert.ok(Math.abs(samples.scales[2] - 1.5) < .001 && Math.abs(samples.scales[3] - 1.5) < .001, JSON.stringify(samples));
                assert.ok(samples.scales[4] > 1 && samples.scales[4] < 1.5, JSON.stringify(samples));
              } else {
                assert.ok(Math.abs(samples.scales[1] - 1.5) < .001 && Math.abs(samples.scales[2] - 1.5) < .001, JSON.stringify(samples));
                assert.ok(Math.abs(samples.scales[3] - 1) < .001 && samples.scales[4] > 1 && samples.scales[5] > samples.scales[4]
                  && samples.scales[5] <= 1.06, JSON.stringify(samples));
              }
              assert.ok(Math.abs(samples.scales.at(-1) - 1) < .001, JSON.stringify(samples));
            }
            for (const id of ['moves-more', 'slot-fold-moves-zoom-strength']) {
              if (await page.$eval(`#${id}`, (fold) => fold.open)) continue;
              await page.$eval(`#${id} > summary`, (summary) => summary.scrollIntoView({ block: 'center' })); await settle();
              await page.click(`#${id} > summary`);
              await page.waitForFunction((id) => document.getElementById(id).open, {}, id);
            }
            const strengthHint = await page.$eval('#slot-fold-moves-zoom-strength > p', (hint) => {
              const style = getComputedStyle(hint), other = getComputedStyle(document.querySelector('#moves-more .fold-hint'));
              return { text: hint.textContent, font: style.fontSize, color: style.color,
                otherFont: other.fontSize, otherColor: other.color };
            });
            assert.equal(strengthHint.text, 'Moves in fast, holds, and moves out at the end of the phrase.');
            assert.equal(strengthHint.font, strengthHint.otherFont);
            assert.equal(strengthHint.color, strengthHint.otherColor);
            await page.$eval('#slot-moves-punch-2', (tile) => tile.click());
            await page.waitForFunction(() => document.getElementById('screen-summary').textContent === 'Moves: Punches and a slow push at 2x · all transitions · step in');
            await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
            const held = await page.$eval('#main-frame .zoom-picture', (zoom) => ({
              animation: getComputedStyle(zoom).animationName, scale: new DOMMatrix(getComputedStyle(zoom).transform).a,
            }));
            assert.deepEqual(held, { animation: 'none', scale: 2 });
            assert.equal(await page.$eval('#slot-moves-punches .zoom-tag', (tag) => tag.textContent), '2x');
            if (shots) await page.screenshot({ path: join(shots, `${width}-moves-strength.png`) });
            await page.emulateMediaFeatures(motion);
            for (const entry of choices.transitionLooks) {
              const selector = `.slot-switch-preview:has(.${entry.preview})`;
              await page.$eval(selector, (host) => host.scrollIntoView({ block: 'center' })); await settle();
              if (!mobile) await page.hover(selector);
              await page.waitForFunction((selector) => document.querySelector(selector)._drawn.element.classList.contains('is-playing'), {}, selector);
              const frames = await page.$eval(selector, async (host) => {
                const from = host.querySelector('.transition-from'), to = host.querySelector('.transition-to');
                const images = [...host.querySelectorAll('img')];
                const animations = host.getAnimations({ subtree: true });
                for (const animation of animations) animation.pause();
                const samples = [];
                for (const time of [0, 864, 1200]) {
                  for (const animation of animations) animation.currentTime = time;
                  await new Promise(requestAnimationFrame);
                  samples.push([Number(getComputedStyle(from).opacity), Number(getComputedStyle(to).opacity)]);
                }
                return { samples, images: images.length === 2 && images.every((image) => image.complete && image.naturalWidth > 0),
                  frames: host.querySelectorAll('iframe').length };
              });
              assert.equal(frames.images, true);
              assert.equal(frames.frames, 0);
              assert.deepEqual(frames.samples[0], [1, 0]);
              assert.ok(frames.samples[1].every((value) => value > 0 && value < 1), JSON.stringify(frames));
              assert.deepEqual(frames.samples[2], [0, 1]);
              await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
              const held = await page.$eval(selector, (host) => ({
                animation: getComputedStyle(host.querySelector('.transition-from')).animationName,
                from: Number(getComputedStyle(host.querySelector('.transition-from')).opacity),
                to: Number(getComputedStyle(host.querySelector('.transition-to')).opacity),
              }));
              assert.equal(held.animation, 'none');
              assert.ok(held.from > 0 && held.to > 0);
              if (shots) await page.screenshot({ path: join(shots, `${width}-transition-${entry.id}.png`) });
              await page.emulateMediaFeatures(motion);
            }
            await page.$eval('#slot-switch-moves-light_leak', (input) => input.click());
            await page.waitForFunction(() => document.getElementById('screen-summary').textContent.endsWith('Whip and Zoom transitions · step in'));
            assert.equal(await page.$eval('#slot-fold-moves-transitions-value', (value) => value.textContent), 'Whip and Zoom');
            assert.equal(await page.$eval('#moves-more-hint', (hint) => hint.textContent), 'Zoom strength, Transitions, Jump cuts, Push in on the first line');
            assert.equal(await page.$eval('#slot-fold-moves-jump-cuts .fold-value', (value) => value.textContent), 'Step in');
            await page.$eval('#slot-fold-moves-jump-cuts > summary', (summary) => summary.scrollIntoView({ block: 'center' })); await settle();
            await page.click('#slot-fold-moves-jump-cuts > summary');
            await page.$eval('#slot-moves-smooth', (tile) => tile.click());
            await page.waitForFunction(() => document.getElementById('screen-summary').textContent.endsWith(' · smooth jump cuts'));
            assert.equal(await page.$eval('#slot-moves-smooth .tile-label', (label) => label.firstChild.textContent), 'Smooth (punch on big moves)');
            assert.equal(await page.$eval('#slot-moves-punches_and_push', (tile) => tile.getAttribute('aria-pressed')), 'true');
            assert.equal(await page.$eval('#slot-fold-moves-jump-cuts .fold-value', (value) => value.textContent), 'Smooth (punch on big moves)');
            await page.$eval('#slot-moves-step_in', (tile) => tile.click());
            await page.waitForFunction(() => document.getElementById('screen-summary').textContent.endsWith(' · step in'));
            await page.$eval('#slot-fold-moves-opening-push', (fold) => { fold.open = true; });
            assert.equal(await page.$$eval('#slot-fold-moves-opening-push .zoom-opening', (nodes) => nodes.length), 1);
            const openingSelector = '#slot-fold-moves-opening-push .slot-switch-preview';
            await page.$eval(openingSelector, (host) => host.scrollIntoView({ block: 'center' }));
            await settle();
            if (!mobile) await page.hover(openingSelector);
            await page.waitForFunction((selector) => document.querySelector(`${selector} .is-playing`), {}, openingSelector);
            const openingSamples = await page.$eval(`${openingSelector} .zoom-opening`, async (zoom) => {
              const animation = zoom.getAnimations()[0]; animation.pause();
              const samples = [];
              for (const time of [0, 516, 1031, 1033, 2000]) {
                animation.currentTime = time; await new Promise(requestAnimationFrame);
                samples.push(new DOMMatrix(getComputedStyle(zoom).transform).a);
              }
              return samples;
            });
            for (const [index, scale] of [1, 1.15, 1.3, 1, 1].entries()) {
              assert.ok(Math.abs(openingSamples[index] - scale) < .001, JSON.stringify(openingSamples));
            }
            await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
            assert.equal(await page.$eval(`${openingSelector} .zoom-opening`, (zoom) => getComputedStyle(zoom).animationName), 'none');
            await page.emulateMediaFeatures(motion);
            await page.$eval('#slot-switch-moves-opening-push', (input) => input.click());
            await page.waitForFunction(() => document.getElementById('screen-summary').textContent.endsWith(' · step in · opening push'));
            await page.$eval('#slot-fold-moves-opening-push', (fold) => fold.scrollIntoView({ block: 'center' }));
            await settle();
            if (shots) await page.screenshot({ path: join(shots, `${width}-opening-push.png`) });
          }
          if (screen === 'moments') {
            assert.equal(await page.$eval('#moment-settings', (panel) => panel.querySelectorAll('input').length), 1);
            assert.equal(await page.$$eval('#moments-tiles .moment-person', (images) => images.length), 0);
            const firstTile = await page.$('#moment-tile-color_pop');
            state.preview.cutout = '/cutout?g=0';
            await page.waitForFunction((count) =>
              document.querySelectorAll('#moments-tiles .moment-person').length === count,
              {}, choices.momentsSlots.length);
            assert.equal(await page.evaluate((first) => first === document.getElementById('moment-tile-color_pop'), firstTile), true);
            for (const entry of choices.momentsSlots) {
              await page.$eval(`#moment-tile-${entry.id}`, (tile) => tile.click());
              await page.waitForFunction((id) => document.getElementById('main-frame')._drawn.momentId === id, {}, entry.id);
              assert.equal(await page.$eval(`#moment-${entry.id}`, (input) => input.checked), false);
              assert.equal(await page.$eval('#main-frame .moment-person', (image) => image.complete && image.naturalWidth > 0), true);
              assert.equal(await page.$eval('#main-frame .moment-person', (image) => new URL(image.src).searchParams.get('look')), 'film');
            }
            const filters = await page.$$eval('#moments-tiles .moment-backdrop', (images) => images.map((image) => getComputedStyle(image).filter));
            assert.match(filters[0], /grayscale\(1\)/); assert.match(filters[1], /blur\(27px\)/);
            assert.match(filters[2], /brightness\(0.3\)/);
            await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
            assert.equal(await page.$eval('#main-frame .moment-treatment', (effect) => getComputedStyle(effect).animationName), 'none');
            assert.equal(await page.$eval('#main-frame .moment-treatment', (effect) => getComputedStyle(effect).opacity), '1');
            if (shots) await page.screenshot({ path: join(shots, `${width}-moments-cutout.png`) });
            await page.$eval('#moment-tile-sticker_freeze', (tile) => tile.click());
            await page.$eval('#moment-sticker_freeze', (input) => input.click());
            assert.equal(await page.$eval('#moment-tile-sticker_freeze .tile-label', (label) => label.textContent.includes(' · On')), true);
            if (shots) await page.screenshot({ path: join(shots, `${width}-moments-sticker.png`) });
            await page.$eval('#moment-sticker_freeze', (input) => input.click());
            await page.$eval('#moment-tile-color_pop', (tile) => tile.click());
            await page.$eval('#moment-color_pop', (input) => input.click());
            await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight)); await settle();
            const clearance = await page.$eval('#moment-settings', (panel) => ({
              bottom: panel.getBoundingClientRect().bottom, bar: document.getElementById('savebar').getBoundingClientRect().top,
            }));
            assert.ok(clearance.bottom <= clearance.bar + 1, JSON.stringify(clearance));
            await page.emulateMediaFeatures(motion);
          }
          if (screen === 'sound') {
            await page.$eval('#sound-more', (fold) => fold.open = true);
            await page.$eval('#effects-fold', (fold) => fold.open = true);
            await settle();
            assert.deepEqual(await page.$$eval('#sound-effects button', (buttons) =>
              buttons.map((button) => button.textContent)), ['Made by Yap Studio', 'ElevenLabs', 'Off']);
            for (const choice of ['made', 'none', 'elevenlabs']) {
              await page.$eval(`#sound-effects [data-effects="${choice}"]`,
                (button) => button.scrollIntoView({ block: 'center' }));
              await settle();
              await page.click(`#sound-effects [data-effects="${choice}"]`);
              assert.equal(await page.$eval(`#sound-effects [data-effects="${choice}"]`,
                (button) => button.getAttribute('aria-pressed')), 'true');
              assert.equal(await page.$eval('#sound-fallback-row', (row) => row.hidden),
                choice !== 'elevenlabs');
            }
            assert.equal(await page.$eval('#sound-fallback', (input) => input.checked), true);
            await page.click('#sound-fallback');
            assert.equal(await page.$eval('#sound-fallback', (input) => input.checked), false);
            await page.click('#sound-fallback');
            assert.equal(await page.$eval('#sound-extras', (group) => group.hidden), false);
            assert.deepEqual(await page.$$eval('#sound-extras input', (inputs) =>
              inputs.map((input) => input.checked)), [false, false, false]);
            await page.$eval('#sound-extra-typing', (input) => input.click());
            assert.deepEqual(await page.$$eval('#sound-extras input', (inputs) =>
              inputs.map((input) => input.checked)), [true, false, false]);
            expected.music.sound_extras = ['typing'];
            await page.click('#sound-effects [data-effects="made"]');
            expected.music.sound_effects = 'made';
            expected.music.sound_effects_fallback = 'made';
            await page.$eval('#bleep-fold', (fold) => fold.open = true);
            await page.click('#bleep-swears');
            assert.equal(await page.$eval('#bleep-swears', (input) => input.checked), true);
            assert.equal(await page.$eval('#bleep-fold-value', (value) => value.textContent), 'On');
            expected.music.bleep_swears = true;
            await page.waitForFunction(() => document.querySelector('#main-frame .poster-word')?.textContent === 'S***');
            assert.equal(await page.$eval('#main-frame .poster-word', (word) => word.textContent), 'S***');
            if (shots) await page.screenshot({ path: join(shots, `${width}-sound-made.png`) });
            const meter = await page.$eval('#music-level-tiles .tile-frame', (frame) => {
              const r = frame.getBoundingClientRect(), m = frame.querySelector('.sound-meters').getBoundingClientRect();
              return { top: (m.top - r.top) / r.height, bottom: (m.bottom - r.top) / r.height,
                font: parseFloat(getComputedStyle(frame.querySelector('.poster-caption')).fontSize), width: frame.clientWidth };
            });
            assert.ok(Math.abs(meter.top - .08) < .01 && Math.abs(meter.bottom - .28) < .01, JSON.stringify(meter));
            assert.ok(Math.abs(meter.font - original.captions.size_px * meter.width / 1080) < .01, JSON.stringify(meter));
          }
          if (screen === 'broll') {
            assert.deepEqual(await page.$$eval('#broll-layout-tiles .tile-label', (labels) => labels.map((label) => label.firstChild.textContent)),
              ['Full screen', 'Mixed', 'In a box', 'Split']);
            await page.$eval('#slot-broll-split', (tile) => tile.click());
            await page.waitForFunction(() => document.getElementById('slot-broll-split').getAttribute('aria-pressed') === 'true');
            const split = await page.$eval('#slot-broll-split .tile-frame', (frame) => {
              const top = frame.querySelector('.split-shot').getBoundingClientRect(), bottom = frame.querySelector('.split-speaker').getBoundingClientRect();
              return { join: Math.abs(top.bottom - bottom.top), ratio: top.height / bottom.height,
                height: top.height / frame.getBoundingClientRect().height };
            });
            assert.ok(split.join < 1 && Math.abs(split.ratio - 1) < .01 && Math.abs(split.height - .5) < .01, JSON.stringify(split));
            for (const kind of ['screen_recording', 'clip', 'photo', 'screenshot']) {
              assert.deepEqual(await page.$$eval(`#broll-${kind} button`, (buttons) => buttons.map((button) => button.textContent)), ['Full screen', 'In a box', 'Split']);
            }
            for (const id of ['broll-motion-fold', 'broll-push-fold']) {
              await page.$eval(`#${id} > summary`, (el) => el.scrollIntoView({ block: 'center' })); await settle();
              await page.click(`#${id} > summary`);
              await page.waitForFunction((id) => document.getElementById(id).open, {}, id);
              const frames = await page.$$eval(`#${id} .tile-frame`, (frames) => frames.map((frame) => {
                const r = frame.getBoundingClientRect(); return { ratio: r.width / r.height, compact: frame.classList.contains('compact') };
              }));
              assert.ok(frames.every((frame) => !frame.compact && Math.abs(frame.ratio - 9 / 16) < .01));
              const columns = await page.$eval(`#${id} .tile-grid`, (grid) => getComputedStyle(grid).gridTemplateColumns.split(' ').length);
              assert.equal(columns, mobile ? 2 : 3);
            }
            await page.$eval('#slot-fold-broll-video-letters', (fold) => fold.open = true);
            await settle();
            assert.equal(await page.$eval('#slot-fold-broll-video-letters .letters-word', (word) => getComputedStyle(word).backgroundClip), 'text');
            assert.equal(await page.$eval('#slot-switch-broll-video-letters', (input) => input.checked), false);
            await page.$eval('#slot-switch-broll-video-letters', (input) => input.click());
            await page.waitForFunction(() => document.getElementById('slot-fold-broll-video-letters-value').textContent === 'On');
            await page.$eval('#slot-fold-broll-color-match', (fold) => fold.open = true);
            assert.equal(await page.$eval('#slot-switch-broll-color-match', (input) => input.checked), true);
            assert.equal(await page.$$eval('#slot-fold-broll-color-match .shot-tint', (tints) => tints.length), 1);
            await page.$eval('#slot-switch-broll-color-match', (input) => input.click());
            await page.waitForFunction(() => document.getElementById('slot-fold-broll-color-match-value').textContent === 'Off');
            assert.equal(await page.$$eval('#slot-fold-broll-color-match .shot-tint', (tints) => tints.length), 0);
          }
          if (screen === 'grade') {
            await page.click('#grade-tiles button:nth-child(2)');
            for (const id of ['grade-more', 'grade-strength-fold']) {
              await page.$eval(`#${id} > summary`, (el) => el.scrollIntoView({ block: 'center' }));
              await settle();
              await page.click(`#${id} > summary`);
              await page.waitForFunction((id) => document.getElementById(id).open, { timeout: 5000 }, id);
            }
          }
          assert.equal(await page.$eval('#step-more', (el) => el.getAttribute('aria-selected')), 'true');
        }
        assert.equal(await page.$eval('#next', (el) => el.hidden), true);
        state.preview.face_boxes = [{ x: 351, y: 796, w: 427, h: 427 }];
        state.preview.generation += 1;
        await page.click('#step-captions');
        await page.$eval('#slot-captions-camera_3d', (tile) => tile.click()); await settle();
        expected.captions.camera_3d = true;
        expected.captions.highlight_style = 'none';
        assert.deepEqual(await page.$$eval('#caption-look-tiles [aria-pressed="true"]',
          (tiles) => tiles.map((tile) => tile.id)), ['slot-captions-camera_3d']);
        await page.waitForFunction(() => {
          const person = document.querySelector('#main-frame .camera3d-person');
          const word = document.querySelector('#main-frame .camera3d-behind');
          return person?.complete && person.naturalWidth > 0 && word && !word.hidden && parseFloat(word.style.height) > 100;
        });
        assert.equal(await page.$eval('#main-frame .camera3d-behind', (word) => word.textContent), 'S***');
        assert.equal(await page.$eval('#main-frame .camera3d-behind', (word) => getComputedStyle(word).fontWeight), '400');
        const visibility = await page.$eval('#slot-captions-camera_3d .tile-frame', async (host, margin) => {
          const animations = host.getAnimations({ subtree: true });
          for (const animation of animations) { animation.pause(); animation.currentTime = 7000; }
          await new Promise(requestAnimationFrame);
          const tile = host.getBoundingClientRect(), scene = host.querySelector('.camera3d-scene').getBoundingClientRect();
          const picture = host.querySelector('.drawn-still').getBoundingClientRect();
          const word = host.querySelector('.camera3d-behind').getBoundingClientRect();
          const first = host.querySelector('.camera3d-words').getBoundingClientRect();
          const scale = scene.width / 1080;
          const head = { left: scene.left + (351 - .15 * 427 - margin) * scale,
            right: scene.left + (778 + .15 * 427 + margin) * scale,
            top: scene.top + (796 - .6 * 427 - margin) * scale, bottom: scene.top + (1223 + margin) * scale };
          const overlap = Math.max(0, Math.min(word.right, head.right) - Math.max(word.left, head.left))
            * Math.max(0, Math.min(word.bottom, head.bottom) - Math.max(word.top, head.top));
          const wholeWords = [...host.querySelectorAll('.camera3d-words')].every((line) => {
            const box = line.getBoundingClientRect();
            return [...line.textContent.matchAll(/\S+/g)].every((match) => {
              const range = document.createRange();
              range.setStart(line.firstChild, match.index); range.setEnd(line.firstChild, match.index + match[0].length);
              const word = range.getBoundingClientRect();
              return range.getClientRects().length === 1 && word.left >= box.left - 1 && word.right <= box.right + 1;
            });
          });
          for (const animation of animations) animation.currentTime = 0;
          return { clear: 1 - overlap / (word.width * word.height), wholeWords,
            first: first.left >= tile.left - 1 && first.right <= tile.right + 1
              && first.top >= tile.top - 1 && first.bottom <= tile.bottom + 1,
            covers: picture.left <= tile.left + 1 && picture.right >= tile.right - 1
              && picture.top <= tile.top + 1 && picture.bottom >= tile.bottom - 1 };
        }, expected.graphics.face_clearance_px + 8.1);
        assert.ok(visibility.clear >= .9 - 1e-6, JSON.stringify(visibility));
        assert.equal(visibility.covers, true);
        assert.equal(visibility.first, true);
        assert.equal(visibility.wholeWords, true);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        if (shots) {
          await page.$eval('#slot-captions-camera_3d', (tile) => tile.scrollIntoView({ block: 'center' })); await settle();
          await page.screenshot({ path: join(shots, `${width}-camera3d-cutout.png`) });
        }
        await page.click('#step-more'); await page.click('#choice-grade'); await settle();
        await page.waitForFunction(() => !document.querySelector('#main-frame .camera3d-scene')
          && document.getElementById('main-frame')._galleryStage);
        assert.equal(await page.$$eval('#main-frame img', (images) => images.length), 1);
        await page.click('#step-captions'); await settle();
        await page.click('#save'); await page.click('#confirm-save');
        await page.waitForFunction(() => !document.getElementById('saved').hidden);
        assert.deepEqual(saved, [expected]);
        await page.click('#update'); await page.click('#save'); await page.click('#confirm-save');
        assert.equal(saved.length, 1);
        assert.equal(await page.$eval('#confirm-save', (button) => button.textContent), 'Update My style');
        assert.match(await page.$eval('#save-sheet-status', (line) => line.textContent), /“My style”/);
        assert.equal(await page.$eval('#save-sheet-error', (line) => line.textContent), '');
        assert.equal(await page.$eval('#save-sheet-status', (line) => line.getAttribute('role')), 'status');
        assert.equal(await page.$eval('#save-sheet-status', (line) => line.classList.contains('field-hint')), true);
        if (shots) await page.screenshot({ path: join(shots, `${width}-save-question.png`) });
        await page.$eval('#style-name', (input) => {
          input.value = 'Sunday'; input.dispatchEvent(new Event('input', { bubbles: true }));
        });
        assert.equal(await page.$eval('#confirm-save', (button) => button.textContent), 'Save style');
        assert.equal(await page.$eval('#save-sheet-status', (line) => line.textContent), '');
        await page.click('#cancel-save');
        assert.equal(await page.$eval('#save-sheet', (sheet) => sheet.open), false);
        await page.click('#save');
        await page.$eval('#style-name', (input) => {
          input.value = 'My style'; input.dispatchEvent(new Event('input', { bubbles: true }));
        });
        await page.click('#confirm-save'); await page.focus('#style-name'); await page.keyboard.press('Enter');
        await page.waitForFunction(() => !document.getElementById('saved').hidden);
        assert.deepEqual(replacements, [undefined, 'saved']);
        assert.deepEqual(saved[1], { ...expected, name: 'My style' });
        await page.click('#update'); await page.click('#back-to-looks');
        assert.equal(await page.$$eval('#start-options .tile-label', (labels) =>
          labels.filter((label) => label.textContent.startsWith('My style')).length), 1);
        assert.deepEqual(errors, []);
        await page.close();
      }
    } finally {
      await browser.close();
      // The browser can leave its stderr pipe open after exiting.
      browser.process()?.stderr?.destroy();
    }
  });

test('a placement without a button on the page stays off the B-roll screen', async () => {
  const state = initialState(); state.schema = structuredClone(schema);
  state.schema.properties.broll.properties.photo_placement.enum = ['full_frame', 'overlay', 'split', 'green_screen'];
  const page = await harness(state, '#broll');
  const labels = page.$('broll-photo').children.map((button) => button.textContent);
  assert.deepEqual(labels, ['Full screen', 'In a box', 'Split']);
});

test('Cards row choices preview Specs list tiles and save the selected row look', async () => {
  const page = await harness(initialState(), '#cards');
  await openFold(page, 'slot-fold-cards-specs-rows');
  page.observers[1].intersect(); await flush();
  for (const [look, name] of [['together', 'Together'], ['on_cue', 'Rows on cue'], ['on_cue_glow', 'Rows on cue, glowing']]) {
    const button = page.$('slot-cards-' + look);
    assert.ok(button);
    assert.ok(button.textContent.includes(name));
    const request = page.requests.find((request) => request.data?.tile.card === 'specs_list'
      && request.data.style.graphics.specs_rows === look);
    assert.deepEqual(request.data.tile, { tab: 'cards', card: 'specs_list', step: 'cards' });
  }
  await page.$('slot-cards-on_cue_glow').fire('click'); await flush();
  await page.save();
  assert.equal(page.requests.find((request) => request.url === '/api/save').data.style.graphics.specs_rows, 'on_cue_glow');
});

test('Jump cuts keeps its fold open when a choice precedes the toggle event', async () => {
  const page = await harness(initialState(), '#moves');
  await openFold(page, 'moves-more');
  page.$('slot-fold-moves-jump-cuts').open = true;
  await page.click('slot-moves-step_in'); await flush();
  assert.equal(page.$('slot-fold-moves-jump-cuts').open, true);
  assert.equal(page.$('slot-moves-step_in').getAttribute('aria-pressed'), 'true');
});

test('Moves More shows four jump cut previews and saves Step in alone', async () => {
  const state = initialState(), original = structuredClone(state.style), page = await harness(state, '#moves');
  await openFold(page, 'slot-fold-moves-jump-cuts');
  const group = choices.movesMoreSlots.find((entry) => entry.id === 'jump-cuts');
  assert.equal(group.options.length, 4);
  assert.deepEqual(group.options.map((entry) => entry.name), ['Hard cut', 'Punch', 'Smooth (punch on big moves)', 'Step in']);
  for (const entry of group.options) {
    const tile = page.$(`slot-moves-${entry.id}`);
    assert.ok(tile);
    assert.equal(tile.children[0]._drawn.screen, entry.preview);
  }
  await page.click('slot-moves-step_in'); await flush();
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'moves.jump_cuts': 'step_in' }));
});

test('sound fallback switch saves only fallback and made never becomes the remembered kit', async () => {
  for (const kit of ['made', 'custom-kit']) {
    const state = initialState(); state.style.music.sound_effects = kit;
    const original = structuredClone(state.style), page = await harness(state, '#sound');
    assert.equal(page.$('sound-fallback-row').hidden, kit === 'made');
    await pressControl(page, 'sound-effects', 1);
    assert.equal(page.$('sound-fallback-row').hidden, false);
    page.$('sound-fallback').checked = true;
    await page.$('sound-fallback').fire('change'); await flush();
    await page.save();
    assert.deepEqual(savePayload(page), withFields(original, {
      'music.sound_effects': kit === 'made' ? 'soft' : kit,
      'music.sound_effects_fallback': 'made',
    }));
  }
});

test('Cards offers Motion streak below Specs list rows and saves it alone', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const page = await harness(state, '#cards');
  await openFold(page, 'slot-fold-cards-card-motion');
  const fold = page.$('slot-fold-cards-card-motion');
  assert.ok(fold.textContent.includes('Cards that fly in leave a short trail; number cards and full-screen cards enter as usual.'));
  assert.equal(page.$('slot-cards-none').getAttribute('aria-pressed'), 'true');
  await page.click('slot-cards-streak'); await flush();
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'graphics.card_motion': 'streak' }));
});

test('Motion streak previews stay paused after a style change until Play', async () => {
  for (const phone of [false, true]) {
    const state = initialState(); state.style.graphics.card_motion = 'streak';
    const page = await harness(state, '#colors', { layers: true, phone });
    await page.pick('palette-tiles', 'Newsprint');
    assert.equal(page.$('play').textContent, 'Play');
    const frames = page.$('main-frame').querySelectorAll('iframe');
    assert.equal(frames.length, 2);
    for (const frame of frames) await frame.fire('load');
    await flush();
    assert.ok(frames.every((frame) => frame._galleryReady && !frame.contentWindow.__player.isPlaying()));
    assert.equal(frames[1].contentWindow.__player.getTime(), 1.5);
    await page.go('cards');
    await openFold(page, 'slot-fold-cards-card-motion');
    page.observers[1].intersect(); await flush();
    const host = page.$('slot-cards-streak').querySelectorAll('.stage-host')[0];
    const frame = host.querySelectorAll('iframe')[0];
    page.observers[0].callback([{ target: host, isIntersecting: true }]);
    await frame.fire('load'); await flush();
    await host.fire('mouseenter');
    assert.equal(frame._galleryReady, true);
    assert.equal(frame.contentWindow.__player.getTime(), host._galleryStage.posterTimes.get(frame));
    assert.equal(frame.contentWindow.__player.isPlaying(), false);
    page.observers[0].callback([{ target: host, isIntersecting: false }]);
    await page.click('play');
    assert.equal(frame.contentWindow.__player.isPlaying(), false);
    page.observers[0].callback([{ target: host, isIntersecting: true }]);
    assert.equal(frame.contentWindow.__player.isPlaying(), true);
  }
});

test('Motion streak previews respect reduced motion on load, visibility and hover', async () => {
  for (const phone of [false, true]) {
    const page = await harness(initialState(), '#cards', { layers: true, phone, reducedMotion: true });
    await openFold(page, 'slot-fold-cards-card-motion');
    page.observers[1].intersect(); await flush();
    const host = page.$('slot-cards-streak').querySelectorAll('.stage-host')[0];
    const frame = host.querySelectorAll('iframe')[0];
    page.observers[0].callback([{ target: host, isIntersecting: true }]);
    await frame.fire('load'); await flush();
    assert.equal(frame._galleryReady, true);
    assert.equal(frame.contentWindow.__player.getTime(), host._galleryStage.posterTimes.get(frame));
    assert.equal(frame.contentWindow.__player.isPlaying(), false);
    page.observers[0].callback([{ target: host, isIntersecting: true }]);
    await host.fire('mouseenter');
    assert.equal(frame.contentWindow.__player.isPlaying(), false);
  }
});

test('Motion streak options start at zero only when visible playback can begin', async () => {
  for (const phone of [false, true]) for (const visible of [false, true]) {
    const page = await harness(initialState(), '#cards', { layers: true, phone });
    await openFold(page, 'slot-fold-cards-card-motion');
    page.observers[1].intersect(); await flush();
    const host = page.$('slot-cards-streak').querySelectorAll('.stage-host')[0];
    const frame = host.querySelectorAll('iframe')[0];
    page.observers[0].callback([{ target: host, isIntersecting: visible }]);
    await frame.fire('load'); await flush();
    assert.equal(frame._galleryReady, true);
    assert.equal(frame.contentWindow.__player.getTime(), visible ? 0 : host._galleryStage.posterTimes.get(frame));
    assert.equal(frame.contentWindow.__player.isPlaying(), visible);
  }
});

test('more sounds switches start off and save only the selected array', async () => {
  const state = initialState(), original = structuredClone(state.style);
  const page = await harness(state, '#sound');
  assert.equal(page.$('sound-fallback').checked, true);
  for (const name of ['typing', 'message-ping', 'camera-shutter']) {
    assert.equal(page.$(`sound-extra-${name}`).checked, false);
  }
  page.$('sound-extra-typing').checked = true;
  await page.$('sound-extra-typing').fire('change'); await flush();
  await page.save();
  assert.deepEqual(savePayload(page), withFields(original, { 'music.sound_extras': ['typing'] }));
  await pressControl(page, 'sound-effects', 0);
  assert.equal(page.$('sound-extras').hidden, true);
});
