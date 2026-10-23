import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const pageFile = (name) => readFileSync(new URL(`../../docs/design/style-page/${name}`, import.meta.url), 'utf8');
const ports = require('../../docs/design/style-page/caption-ports.js');
const context = {window: {
  YapLooks: require('../../docs/design/style-page/looks.js'),
  YapPhrases: require('../../docs/design/style-page/phrases.js'),
  YapCaptionPorts: ports,
}, document: {querySelector: () => null}, matchMedia: () => ({matches: false})};
vm.runInNewContext(pageFile('style-data.js'), context);
const app = pageFile('app.js').replace(/\n  start\(\);\n\}\)\(\);\s*$/,
  '\n  window.frameTest = {Frame, WORDS};\n})();');
vm.runInNewContext(app, context);
const {Frame, WORDS} = context.window.frameTest;
const defaults = (rule) => rule.type === 'object'
  ? Object.fromEntries(Object.entries(rule.properties).map(([key, child]) => [key, defaults(child)]))
  : rule.default;

function element() {
  return {style: {setProperty(name, value) {this[name] = value;}},
    classList: {toggle() {}}, offsetWidth: 300, offsetHeight: 85.12};
}

function frameFor(captions) {
  const frame = Object.create(Frame.prototype);
  frame.stage = {style: element().style, dataset: {}};
  frame.els = Object.fromEntries(['ground', 'line', 'words', 'cap', 'hook', 'gfx', 'mark']
    .map((key) => [key, element()]));
  frame.invalidate = () => {};
  frame.setStyle({...defaults(context.window.YAP_STYLE_DATA.schema),
    captions: {...defaults(context.window.YAP_STYLE_DATA.schema.properties.captions), ...captions}});
  return frame;
}

for (const mode of ['box', 'band']) {
  for (const position of ['bottom', 'middle']) {
    for (const placement of ['fixed', 'avoid_face']) {
      test(`transparent ${mode} uses absent-ground geometry at ${position}, ${placement}`, () => {
        const captions = {background_opacity: 0, background_padding_px: 80,
          shadow_opacity: 0, highlight_style: 'none', position, placement, top_px: 1200};
        const absent = frameFor({...captions, background_mode: 'none'});
        const transparent = frameFor({...captions, background_mode: mode});
        absent.layoutGround();
        transparent.layoutGround();
        assert.equal(transparent.stage.style['--cap-wrap-width'], '940px');
        assert.equal(transparent.stage.style['--cap-wrap-width'], absent.stage.style['--cap-wrap-width']);
        assert.equal(transparent.stage.style['--cap-outline'], '0px transparent');
        assert.equal(transparent.els.ground.style.display, 'none');
        assert.equal(transparent.els.cap.style.top, absent.els.cap.style.top);
        const visible = frameFor({...captions, background_mode: mode, background_opacity: 0.01});
        visible.layoutGround();
        assert.equal(visible.stage.style['--cap-wrap-width'], '780px');
        assert.equal(visible.els.ground.style.display, 'block');
        assert.notEqual(visible.stage.style['--cap-outline'], '0px transparent');
        // Toggling an existing visible ground also returns its geometry to none.
        visible.setStyle({...visible.style, captions: {...visible.style.captions, background_opacity: 0}});
        visible.layoutGround();
        assert.equal(visible.els.ground.style.display, 'none');
        assert.equal(visible.els.cap.style.top, absent.els.cap.style.top);
      });
    }
  }
}

const solidLooks = ['neon_glow', 'neon_accent', 'gradient_fill', 'glitch_rgb', 'particle_burst', 'texture'];
for (const mode of [...solidLooks, 'camera_follow', 'none']) {
  test(`${mode} phrase entry opacity matches the engine while rise still moves`, () => {
    const frame = frameFor({highlight_style: mode, entry_ms: 100, entry_rise_px: 12});
    const start = WORDS[0].start;
    frame.tl = {hook: null, graphic: {start: 10, end: 11},
      phrases: [{start, end: start + 1, ids: [0]}]};
    frame.pi = 0;
    frame.renderPhrase = () => {};
    frame.portAnimations = {seek() {}};
    frame.spans = [element()];
    const samples = [start, start + 0.03, start + 0.1, start + 0.03].map((time) => {
      frame.render(time);
      return {...frame.els.line.style};
    });
    assert.deepEqual(samples[1], samples[3]);
    if (solidLooks.includes(mode)) assert.deepEqual(samples.map((css) => css.opacity), [1, 1, 1, 1]);
    else {
      assert.equal(samples[0].opacity, 0);
      assert.ok(samples[1].opacity > 0 && samples[1].opacity < 1);
      assert.equal(samples[2].opacity, 1);
    }
    assert.equal(samples[0].transform, 'translateY(12px)');
    assert.notEqual(samples[1].transform, samples[0].transform);
    assert.notEqual(samples[1].transform, samples[2].transform);
    assert.equal(samples[2].transform, 'translateY(0px)');
    frame.render(start + 1);
    assert.equal(frame.els.cap.style.visibility, 'hidden');
  });
}

test('preview ships registry span contrast guards from the engine stylesheet', () => {
  const css = pageFile('caption-ports.css');
  const guard = css.match(/\.cap-slam \.w,.*?\{([^}]+)\}/s);
  assert.ok(guard);
  assert.ok(guard[0].includes('.cap-weight-shift .w'));
  assert.ok(guard[0].includes('.cap-clip-wipe .w'));
  assert.ok(guard[1].includes('-webkit-text-stroke:'));
  assert.ok(guard[1].includes('var(--legibility_halo)'));
  for (const mode of ['slam', 'weight_shift', 'clip_wipe']) {
    for (const color of ['#FFFFFF', '#000000', '#777777', '#FFFF00']) {
      const frame = frameFor({highlight_style: mode, shadow_opacity: 0, background_mode: 'none',
        text_color: color, highlight_color: color});
      const css = frame.stage.style;
      const {contrast} = context.window.YapLooks;
      assert.ok(contrast(color, css['--legibility_ink']) >= 4.5);
      assert.ok(contrast(css['--legibility_ink'], css['--legibility_halo']) >= 18);
      assert.ok(contrast(color, css['--accent_ink']) >= 4.5);
      assert.ok(contrast(css['--accent_ink'], css['--accent_halo']) >= 18);
    }
  }
});
