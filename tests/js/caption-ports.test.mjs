import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';

const require = createRequire(import.meta.url);
const ports = require('../../docs/design/style-page/caption-ports.js');
const template = JSON.parse(readFileSync(new URL(
  '../../src/yap_studio/templates/captions/template.json', import.meta.url), 'utf8'));
const legacy = new Set(['moving_block', 'underline', 'color', 'pill', 'gradient', 'editorial',
  'slam', 'weight_shift', 'clip_wipe', 'giant', 'none']);
const styles = template.variables.find((v) => v.id === 'highlight_style').options
  .map((v) => v.value).filter((id) => !legacy.has(id));
const variables = Object.fromEntries(template.variables.map((v) => [v.id, v.default]));
const caseExamples = JSON.parse(readFileSync(new URL(
  '../fixtures/caption-case-cases.json', import.meta.url), 'utf8'));

test('engine and preview preserve semantic case while keeping ordinary words soft', () => {
  const looks = {pill_karaoke: 'lower', weight_shift: 'lower', clip_wipe: 'upper',
    camera_follow: 'upper', glitch_rgb: 'upper', highlight: 'upper', neon_glow: 'upper',
    particle_burst: 'upper'};
  for (const [mode, expectedCase] of Object.entries(looks)) {
    for (const word of caseExamples) {
      assert.equal(ports.wordCase(word, mode), word[expectedCase], `${mode}: ${word.text}`);
      const span = {style: {}};
      ports.applyWordCase(span, word, mode);
      if (word[expectedCase] !== (expectedCase === 'lower' ? word.text.toLowerCase()
        : word.text.toUpperCase())) assert.equal(span.style.textTransform, 'none');
    }
  }
  assert.equal(ports.wordCase({text: "I'll"}, 'slam'), "I'LL");
  assert.equal(ports.wordCase({text: 'Jordan', preserve_case: true}, 'slam'), 'JORDAN');
  const span = {style: {}};
  ports.applyWordCase(span, {text: "I'll", preserve_case: true}, 'slam');
  assert.equal(span.style.textTransform, undefined);
});

class Element {
  constructor() {
    this.children = [];
    this.style = {setProperty(name, value) { this[name] = value; }};
    this.animations = [];
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  animate(keys, options) {
    this.animations.push({keys, options, currentTime: null,
      pause() { this.paused = true; }, cancel() { this.cancelled = true; }});
  }
  getAnimations() {
    return [...this.animations, ...this.children.flatMap((child) => child.getAnimations())];
  }
}

test('all fourteen new caption looks use the engine effects in the style preview', () => {
  assert.equal(styles.length, 14);
  assert.ok(styles.every((id) => ports.supports(id)));
  assert.ok([...legacy, 'unknown'].every((id) => !ports.supports(id)));
});

for (const id of styles) {
  test(`${id} preview effects pause, seek backwards and cancel on a new choice`, () => {
    globalThis.document = {createElement: () => new Element()};
    const words = new Element();
    const spans = [new Element(), new Element()];
    words.append(...spans);
    const phrase = {start: 0.2, end: 1.8, stress: 1,
      words: [{text: 'Small', start: 0.2, end: 0.6}, {text: 'steps', start: 0.6, end: 1.3}]};
    const control = ports.build({...variables, highlight_style: id}, phrase, spans, words, 2);
    const animations = words.getAnimations();
    assert.ok(animations.length > 0);
    assert.ok(animations.every((animation) => animation.paused));
    assert.deepEqual(spans.map((span) => span.children[0].textContent), ['Small', 'steps']);
    control.seek(1.2);
    assert.ok(animations.every((animation) => animation.currentTime === 1200));
    control.seek(0.3);
    assert.ok(animations.every((animation) => animation.currentTime === 300));
    control.cancel();
    assert.ok(animations.every((animation) => animation.cancelled));
    delete globalThis.document;
  });
}

for (const id of ['neon_glow', 'neon_accent', 'gradient_fill', 'glitch_rgb', 'particle_burst', 'texture']) {
  test(`${id} preview keeps safe requested ink and substitutes unsafe ink`, () => {
    globalThis.document = {createElement: () => new Element()};
    try {
      assert.equal(ports.solidCore(id), true);
      for (const [text, accent, expected] of [
        ['#0000FF', '#D8FF2F', '#0000FF'], ['#333333', '#FFFFFF', '#333333'],
        ['#FFFFFF', '#000000', '#FFFFFF'], ['#FF0000', '#D8FF2F', '#111111'],
        ['#FFFFFF', '#D8FF2F', '#111111'], ['#777777', '#777777', '#000000'],
      ]) {
        const words = new Element();
        const spans = [new Element(), new Element()];
        words.append(...spans);
        const phrase = {start: 0.2, end: 1.8, stress: 1,
          words: [{text: 'Small', start: 0.2, end: 0.6}, {text: 'steps', start: 0.6, end: 1.3}]};
        const control = ports.build({...variables, highlight_style: id, text_color: text,
          highlight_color: accent}, phrase, spans, words, 2);
        for (const time of [0.2, 0.4, 0.8, 1.6, 0.2]) {
          control.seek(time);
          for (const span of spans) {
            const ink = span.children[0];
            assert.equal(ink.style.color, expected, `${text} on ${accent}`);
            assert.equal(ink.style.opacity, '1');
            assert.equal(ink.animations.length, 0);
            assert.notEqual(ink.style.webkitTextStrokeColor, expected);
          }
        }
        control.cancel();
      }
    } finally {
      delete globalThis.document;
    }
  });
}
