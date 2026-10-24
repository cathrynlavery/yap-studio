import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { create404Retry, ensureFontOption, previewNote } from '../../src/yap_studio/gallery/static/choices.mjs';

test('a system font remains selected even when it is not listed', () => {
  const select = {
    options: [{ value: 'Instrument Serif' }],
    ownerDocument: { createElement: () => ({}) },
    append(option) { this.options.push(option); },
    value: '',
  };
  ensureFontOption(select, 'Arial');
  assert.equal(select.value, 'Arial');
  assert.deepEqual(select.options.at(-1), { value: 'Arial', textContent: 'Arial' });
  ensureFontOption(select, 'Arial');
  assert.equal(select.options.length, 2);
});

test('a layer 404 triggers one fresh layers request', async () => {
  const requested = [];
  let refreshes = 0;
  const fakeFetch = async (url, options) => {
    requested.push([url, options.method]);
    return { status: 404 };
  };
  const retry = create404Retry(fakeFetch, async () => { refreshes += 1; });
  assert.equal(await retry('/layer/old/0/'), true);
  assert.equal(await retry('/layer/old/0/image.jpg'), false);
  assert.deepEqual(requested, [['/layer/old/0/', 'HEAD']]);
  assert.equal(refreshes, 1);
});

test('an asset 404 triggers the same one-time refresh', async () => {
  let refreshes = 0;
  const retry = create404Retry(async () => {
    assert.fail('the iframe already confirmed the asset was missing');
  }, async () => { refreshes += 1; });
  assert.equal(await retry('/layer/old/0/image.jpg', true), true);
  assert.equal(await retry('/layer/old/0/image.jpg', true), false);
  assert.equal(refreshes, 1);
});

test('a failed preview message appears in the preview note', () => {
  const message = 'Could not read your newest take; showing the sample still.';
  assert.equal(previewNote({ source: 'sample', message }), message);
});

import { steps, stepFromHash, adjacentStep, availableSteps, screenAvailable, optionAvailable,
  screens, movesSlots, momentsSlots,
  showStep, startList, palettes, normalizeHex,
  colorsOf, withColors, withFont, nearestWidth, colorWarnings, mix, contrast,
  changeSummary } from '../../src/yap_studio/gallery/static/choices.mjs';

test('screen navigation is bounded and restores visible screens and legacy hashes', () => {
  assert.deepEqual(steps.map(([id]) => id),
    ['looks', 'font', 'captions', 'colors', 'top-text', 'more', 'grade', 'opening', 'cards', 'broll', 'moves', 'moments', 'sound']);
  assert.deepEqual(screens.map(({ id }) => id),
    ['font', 'captions', 'colors', 'top-text', 'grade', 'opening', 'cards', 'broll', 'moves', 'moments', 'sound']);
  for (const [index, [step]] of steps.entries()) {
    assert.equal(stepFromHash(`#${step}`), step);
    assert.equal(adjacentStep(step, 1), steps[Math.min(index + 1, steps.length - 1)][0]);
    assert.equal(adjacentStep(step, -1), steps[Math.max(index - 1, 0)][0]);
  }
  assert.equal(stepFromHash('#looks'), 'looks');
  assert.equal(stepFromHash('#templates'), 'looks');
  assert.equal(stepFromHash('#start'), 'looks');
  assert.equal(stepFromHash('#words'), 'captions');
  assert.equal(stepFromHash('#highlight'), 'captions');
  assert.equal(stepFromHash('#cards'), 'cards');
  assert.equal(stepFromHash('#title'), 'opening');
  assert.equal(stepFromHash('#moves'), 'moves');
  assert.equal(stepFromHash('#moments'), 'moments');
  assert.equal(stepFromHash('#unknown'), 'looks');
  assert.equal(stepFromHash('#constructor'), 'looks');
  assert.equal(stepFromHash('#__proto__'), 'looks');
  assert.equal(stepFromHash(''), 'looks');
});

test('schema requirements and injected slots reveal optional screens in order', () => {
  const schema = { type: 'object', properties: {
    captions: { type: 'object', properties: { entrance: { type: 'string', enum: ['slide', 'rise'] } } },
  } };
  const entry = { id: 'test', name: 'Test', note: '', apply: structuredClone,
    selected: () => false, requires: ['captions.entrance', 'rise'] };
  assert.equal(movesSlots.filter((entry) => !entry.type).length, 3);
  assert.equal(movesSlots.filter((entry) => entry.type === 'switch').length, 1);
  assert.equal(screenAvailable('moves', undefined, []), false);
  assert.equal(stepFromHash('#moves', undefined, { moves: [] }), 'more');
  assert.equal(momentsSlots.length, 6);
  assert.deepEqual(momentsSlots.slice(-1).map(({ id, name, note }) => [id, name, note]), [
    ['clone_flash', 'Clone flash', 'Copies of you pop in beside you'],
  ]);
  assert.equal(screenAvailable('moments', undefined, []), false);
  assert.equal(stepFromHash('#moments', undefined, { moments: [] }), 'more');
  assert.equal(optionAvailable(entry, schema), true);
  schema.properties.captions.properties.camera_3d = { type: 'boolean', default: false };
  assert.equal(optionAvailable({ requires: ['captions.camera_3d', true] }, schema), true);
  assert.equal(optionAvailable({ requires: ['captions.camera_3d', false] }, schema), true);
  assert.equal(optionAvailable({ requires: ['captions.camera_3d', 'true'] }, schema), false);
  assert.equal(optionAvailable({ ...entry, requires: ['missing.path', true] }, schema), false);
  assert.equal(optionAvailable({ ...entry, requires: ['captions.entrance', 'other'] }, schema), false);
  assert.equal(optionAvailable({ ...entry, requires: ['constructor', 'rise'] }, schema), false);
  assert.equal(screenAvailable('moves', schema, [entry]), true);
  assert.equal(screenAvailable('moments', schema, [entry]), true);
  assert.equal(screenAvailable('constructor', schema), false);
  assert.equal(screenAvailable('moves', schema, [{ ...entry, requires: ['missing.path', true] }]), false);
  const slots = { moves: [entry] };
  assert.deepEqual(availableSteps(schema, slots).map(([id]) => id),
    ['looks', 'font', 'captions', 'colors', 'top-text', 'more', 'grade', 'opening', 'cards', 'broll', 'moves', 'sound']);
  assert.equal(adjacentStep('broll', 1, schema), 'sound');
  assert.equal(adjacentStep('sound', -1, schema), 'broll');
  assert.equal(adjacentStep('broll', 1, schema, slots), 'moves');
  assert.equal(adjacentStep('sound', -1, schema, slots), 'moves');
  assert.equal(stepFromHash('#moves', schema, slots), 'moves');
  slots.moments = [entry];
  assert.equal(adjacentStep('moves', 1, schema, slots), 'moments');
});

test('workspace instructions name the optional step shown on the page', () => {
  const instructions = readFileSync(new URL('../../workspace/AGENTS.md', import.meta.url), 'utf8');
  assert.ok(instructions.includes('More choices'));
});

test('the shipped page opens on looks and explains Customize, optional extras and saving', () => {
  const html = readFileSync(new URL('../../src/yap_studio/gallery/static/pick.html', import.meta.url), 'utf8');
  for (const copy of ['Tap a look to try it on your video.', '>Customize</button>', '>Back to looks</button>',
    'Reset changes', 'Background strength', 'Italicize emphasized words',
    'More choices', '>Optional</span>']) assert.ok(html.includes(copy), copy);
  assert.doesNotMatch(html, /How do you want to start|Change how I start|Use a template|Build my own|Swipe for more|Reset to preset|Italic key words|>Opacity<|for="caption-color">Captions/);
  assert.ok(html.indexOf('id="font-options"') < html.indexOf('id="width-tiles"'));
});

test('jumping between steps exposes exactly one panel and one selected tab', () => {
  const tabs = ['font', 'captions', 'colors', 'top-text', 'more'].map((step) => ({ dataset: { step }, attrs: {},
    setAttribute(name, value) { this.attrs[name] = value; } }));
  const panels = steps.map(([step]) => ({ id: `panel-${step}`, hidden: false }));
  for (const [step] of steps.filter(([id]) => id !== 'looks')) {
    showStep(step, tabs, panels);
    assert.deepEqual(panels.filter((panel) => !panel.hidden).map((panel) => panel.id), [`panel-${step}`]);
    assert.deepEqual(tabs.filter((tab) => tab.attrs['aria-selected'] === 'true').map((tab) => tab.dataset.step), [screens.find(({ id }) => id === step)?.optional ? 'more' : step]);
    assert.equal(tabs.filter((tab) => tab.tabIndex === 0).length, 1);
  }
});

test('the start list puts saved library names before the engine presets', () => {
  const list = startList({ library: [{ id: 'learned', name: 'SparrowLight' }, { id: 'personal', name: 'Sunday' }],
    presets: [{ id: 'electric', name: 'Electric' }] });
  assert.deepEqual(list.map(({ name, fromLibrary }) => [name, fromLibrary]),
    [['SparrowLight', true], ['Sunday', true], ['Electric', false]]);
});

const base = {
  name: 'Editorial', brand: { accent_color: '#FF0000', captions_font: 'Inter' },
  captions: { font: 'Inter', text_color: '#FFFFFF', highlight_style: 'pill',
    highlight_color: '#FF0000', highlight_text_color: '#FFFFFF', max_words_per_phrase: 3, width_scale: .79 },
  hook: { font: 'Gelasio', background_color: '#FFFFFF', text_color: '#111111' },
  graphics: { font: 'Alata', card_kinds_off: [], accent_color: '#FF0000',
    paper_color: '#F6F1E7', ink_color: '#222222', titlebar_color: '#E5E0D7' }, music: { level: .1 },
};

const marker = JSON.parse(readFileSync(new URL('../../src/yap_studio/style/presets/marker.json', import.meta.url)));

test('Cards reads Marker graphics paper without changing its distinct title background', () => {
  const original = structuredClone(marker);
  assert.equal(colorsOf(marker).paper, '#FFFDF4');
  assert.equal(colorsOf(marker).ink, marker.graphics.ink_color);
  assert.equal(marker.hook.background_color, '#FFF6DC');
  assert.deepEqual(marker, original);
  const distinctInk = structuredClone(marker);
  distinctInk.hook.text_color = '#202020';
  assert.equal(colorsOf(distinctInk).ink, '#1D1D1D');
});

test('manual colors share only the edited role with titles and preserve other source colors', () => {
  const values = { caption: '#FFF4DD', accent: '#00AAFF', paper: '#ABCDEF', ink: '#101010' };
  const fields = {
    caption: [['captions', 'text_color']],
    accent: [['captions', 'highlight_color'], ['hook', 'accent_color'], ['graphics', 'accent_color']],
    paper: [['hook', 'background_color'], ['graphics', 'paper_color']],
    ink: [['hook', 'text_color'], ['graphics', 'ink_color']],
  };
  for (const role of Object.keys(values)) {
    const changed = withColors(marker, { ...colorsOf(marker), [role]: values[role] }, [role]);
    for (const [section, field] of fields[role]) assert.equal(changed[section][field], values[role]);
    if (role !== 'paper') assert.equal(changed.hook.background_color, '#FFF6DC');
    if (role !== 'ink') assert.equal(changed.hook.text_color, marker.hook.text_color);
    if (role !== 'accent') {
      assert.equal(changed.hook.accent_color, '#FF2E88');
      assert.equal(changed.graphics.accent_color, '#FF2E88');
    }
    if (['caption', 'accent'].includes(role)) {
      assert.equal(changed.graphics.paper_color, '#FFFDF4');
      assert.equal(changed.graphics.titlebar_color, marker.graphics.titlebar_color);
    }
  }
  const palette = withColors(marker, palettes[0]);
  assert.equal(palette.hook.background_color, palettes[0].paper);
  assert.equal(palette.graphics.paper_color, palettes[0].paper);
  assert.equal(palette.hook.text_color, palettes[0].ink);
  assert.equal(palette.graphics.ink_color, palettes[0].ink);
});

test('manual caption, paper and ink edits keep a saved brand accent', () => {
  for (const role of ['caption', 'paper', 'ink']) {
    const changed = withColors(base, { ...colorsOf(base), [role]: '#ABCDEF' }, [role]);
    assert.deepEqual(changed.brand, base.brand);
  }
  const changed = withColors(base, { ...colorsOf(base), accent: '#ABCDEF' }, ['accent']);
  assert.equal(changed.brand.accent_color, undefined);
  assert.equal(colorsOf(changed).accent, '#ABCDEF');
});

test('ten prototype palettes and three/six-digit hex input retain the four roles', () => {
  assert.deepEqual(palettes.map(({ name }) => name), ['Newsprint', 'Sage', 'Rust', 'Cobalt', 'Terminal', 'Lime', 'Yellow', 'Tomato', 'Bubblegum', 'Highlighter']);
  assert.equal(normalizeHex('#2BF'), '#22BBFF');
  assert.equal(normalizeHex(' 2f3bff '), '#2F3BFF');
  assert.equal(normalizeHex('#xyg'), null);
  const colors = { caption: '#FFF4DD', accent: '#C4C4C4', paper: '#F6F1E7', ink: '#000000' };
  const changed = withColors(base, colors);
  assert.deepEqual(colorsOf(changed), colors);
  assert.equal(changed.brand.accent_color, undefined);
  assert.equal(changed.captions.highlight_text_color, '#000000');
  assert.equal(changed.graphics.titlebar_color, mix(colors.paper, colors.ink, .08));
  assert.equal(changed.graphics.accent_color, colors.accent);
  assert.equal(changed.hook.accent_color, colors.accent);
  assert.throws(() => withColors(base, { ...colors, caption: 'invalid' }));
  assert.equal(base.brand.accent_color, '#FF0000');
});

test('a font choice sets every role while preserving a saved custom letter width', () => {
  const changed = withFont(base, 'Instrument Serif');
  assert.equal(changed.captions.font, 'Instrument Serif');
  assert.equal(changed.brand.captions_font, 'Instrument Serif');
  assert.equal(changed.hook.font, 'Instrument Serif');
  assert.equal(changed.graphics.font, 'Instrument Serif');
  assert.equal(changed.captions.letter_spacing_px, -.8);
  assert.equal(changed.captions.width_scale, .79);
  assert.equal(nearestWidth(.79).name, 'Condensed');
});

test('pill warns about muted text even when spoken text has enough contrast', () => {
  const style = withColors(base, { caption: '#FFFFFF', accent: '#808080', paper: '#FFFFFF', ink: '#111111' });
  assert.ok(contrast(style.captions.highlight_text_color, '#808080') >= 4.5);
  assert.ok(colorWarnings(style).some((message) => message.includes('upcoming pill words')));
});

test('the persistent summary names the changed choices and the selected base', () => {
  assert.equal(changeSummary(base, base), 'Starting from Editorial');
  const changed = structuredClone(base);
  changed.captions.width_scale = .7;
  changed.graphics.card_kinds_off = ['stat'];
  assert.equal(changeSummary(changed, base), 'Changed Font, Cards from Editorial');
});

test('background and face preference edits appear in the persistent caption summary', () => {
  for (const [field, value] of Object.entries({ background_mode: 'box', background_color: '#FFFFFF',
    background_opacity: .63, background_padding_px: 27, background_radius_px: 23, placement: 'fixed' })) {
    const changed = structuredClone(base);
    changed.captions[field] = value;
    assert.equal(changeSummary(changed, base), 'Changed Captions from Editorial');
  }
});

test('background contrast replaces the dark-on-video warning until the ground is transparent or off', () => {
  const style = structuredClone(base);
  Object.assign(style.captions, { text_color: '#111111', background_mode: 'box',
    background_color: '#FFFFFF', background_opacity: .5 });
  assert.ok(!colorWarnings(style).some((message) => message.startsWith('Dark captions')));
  style.captions.background_opacity = 0;
  assert.ok(colorWarnings(style).some((message) => message.startsWith('Dark captions')));
  style.captions.background_opacity = .5;
  style.captions.background_mode = 'none';
  assert.ok(colorWarnings(style).some((message) => message.startsWith('Dark captions')));
});
