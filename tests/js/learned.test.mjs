import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const {readLearned, switchPreset, undoPreset} = require('../../docs/design/style-page/learned.js');
const pageFile = (name) => readFileSync(new URL(`../../docs/design/style-page/${name}`, import.meta.url), 'utf8');
const context = {window: {}};
vm.runInNewContext(pageFile('style-data.js'), context);
const schema = JSON.parse(JSON.stringify(context.window.YAP_STYLE_DATA.schema));
const defaults = (rule) => rule.type === 'object'
  ? Object.fromEntries(Object.entries(rule.properties).map(([key, child]) => [key, defaults(child)]))
  : rule.default;

function measurement() {
  const style = defaults(schema);
  delete style.internal;
  Object.assign(style.broll, {
    cutout: 'screen',
    screen_recording_placement: 'overlay', clip_placement: 'overlay',
    photo_placement: 'full_frame', screenshot_placement: 'full_frame',
    overlay_box: 'side_right', overlay_motion: 'none', avoid_tags: ['stock', 'busy_background'],
  });
  style.schema = 'yap-studio/resolved-style@1';
  style.name = 'Learned style (proposed)';
  const sources = {};
  for (const section of ['captions', 'hook', 'chapters', 'graphics', 'music', 'broll', 'grade']) {
    for (const key of Object.keys(style[section])) {
      sources[`${section}.${key}`] = {source: 'base', status: null, reason: 'Not measured.'};
    }
  }
  sources['captions.text_color'] = {source: 'measured', status: 'proposed', reason: null};
  sources['hook.duration_s'] = {source: 'base', status: 'outside_style', reason: 'Above the style limit.'};
  sources['captions.size_px'] = {source: 'base', status: 'partial', reason: 'Incomplete coverage.'};
  return {
    schema: 'yap-studio/taste-measurements@1',
    proposed_style: {base: {kind: 'schema_defaults', name: 'Schema defaults'}, style, sources},
  };
}

function app() {
  const element = (tag = 'div') => ({
    tag, children: [], style: {}, setAttribute() {},
    append(...children) {this.children.push(...children);},
    replaceChildren(...children) {this.children = children;},
  });
  const sandbox = {
    window: {YAP_STYLE_DATA: context.window.YAP_STYLE_DATA,
      YapLooks: require('../../docs/design/style-page/looks.js'),
      YapPhrases: require('../../docs/design/style-page/phrases.js')},
    matchMedia: () => ({matches: true}),
    document: {querySelector: () => null, createElement: element},
  };
  vm.runInNewContext(pageFile('app.js').replace(/\n  start\(\);\n\}\)\(\);\s*$/,
    `
  window.stylePageTest = {state, applyStartingStyle, output, check, withFont, withPalette, rolesOf, effectiveStyle,
    showLearned(learned, host) {
      learnedEvidence = host;
      state.learned = learned;
      refreshLearnedEvidence();
    }};
})();`), sandbox);
  return {...sandbox.window.stylePageTest, element};
}

test('learned style preserves all values and separates measured and base rows', () => {
  const data = measurement();
  const before = JSON.stringify(data);
  const result = readLearned(data, schema);
  assert.equal(result.base.name, 'Schema defaults');
  assert.equal(result.rows.measured.length, 1);
  assert.equal(result.rows.measured[0].label, 'Captions');
  assert.equal(result.rows.base.find((row) => row.field === 'captions.font').label, 'Font (captions)');
  assert.equal(result.rows.base.find((row) => row.field === 'hook.duration_s').reason, 'Above the style limit.');
  assert.equal(result.rows.base.find((row) => row.field === 'captions.size_px').reason, 'Incomplete coverage.');
  assert.equal(result.rows.measured.length + result.rows.base.length, Object.keys(data.proposed_style.sources).length);
  assert.equal(JSON.stringify(data), before);
  result.style.captions.size_px += 1;
  assert.notEqual(result.style.captions.size_px, data.proposed_style.style.captions.size_px);
});

test('learned chapter choices and evidence are preserved', () => {
  const data = measurement();
  data.proposed_style.style.chapters = {style: 'title_card', hold_s: 2};
  const result = readLearned(data, schema);
  assert.deepEqual(result.style.chapters, {style: 'title_card', hold_s: 2});
  assert.deepEqual(result.rows.base.filter((row) => row.field.startsWith('chapters.'))
    .map((row) => row.label), ['Chapter title card', 'Chapter duration']);
});

test('older learned styles inherit chapter defaults without changing the input', () => {
  const data = measurement();
  delete data.proposed_style.style.chapters;
  delete data.proposed_style.sources['chapters.style'];
  delete data.proposed_style.sources['chapters.hold_s'];
  const before = JSON.stringify(data);
  const result = readLearned(data, schema);
  assert.deepEqual(result.style.chapters, {style: 'none', hold_s: 1.6});
  assert.ok(result.rows.base.filter((row) => row.field.startsWith('chapters.'))
    .every((row) => row.reason));
  assert.equal(JSON.stringify(data), before);
});

test('a current learned chapter section requires both source decisions', () => {
  const data = measurement();
  delete data.proposed_style.sources['chapters.hold_s'];
  assert.throws(() => readLearned(data, schema), /chapters.hold_s.*source/);
});

test('learned broll fields keep their values and readable labels', () => {
  const data = measurement();
  const result = readLearned(data, schema);
  const rows = result.rows.base.filter((row) => row.field.startsWith('broll.'));
  assert.deepEqual(rows.filter((row) => row.field !== 'broll.cutout').map((row) => row.label), [
    'Screen recordings', 'Clips', 'Photos', 'Screenshots',
    'Overlay position', 'Overlay motion', 'Photo push', 'Push scale', 'Tags to avoid',
    'broll.stock_clips', 'broll.generated_pictures', 'broll.emphasis_on', 'broll.color_match',
    'broll.green_screen_layout', 'Framing behind the tall box',
  ]);
  assert.equal(rows.find((row) => row.field === 'broll.cutout').value, 'screen');
  assert.deepEqual(result.style.broll, data.proposed_style.style.broll);
  assert.deepEqual(result.style.broll.emphasis_on, []);
  assert.deepEqual(rows.map((row) => row.value), Object.values(data.proposed_style.style.broll));
  result.style.broll.avoid_tags.push('new_tag');
  assert.deepEqual(data.proposed_style.style.broll.avoid_tags, ['stock', 'busy_background']);
});

test('grade labels preserve choices and older source records can inherit them', () => {
  const data = measurement();
  data.proposed_style.style.grade = {look: 'film', strength: 0.5};
  for (const key of Object.keys(schema.properties.grade.properties)) delete data.proposed_style.sources[`grade.${key}`];
  const result = readLearned(data, schema);
  assert.deepEqual(result.style.grade, {look: 'film', strength: 0.5, color: null, slider: false});
  assert.deepEqual(result.rows.base.filter((row) => row.field.startsWith('grade.'))
    .map((row) => row.label), ['Color grade', 'Grade strength', 'Picked color', 'Before/after slider']);
  delete data.proposed_style.style.grade;
  assert.equal(readLearned(data, schema).style.grade, undefined);
});

test('every schema caption look round trips through a learned style base', () => {
  for (const highlight of schema.properties.captions.properties.highlight_style.enum) {
    const data = measurement();
    data.proposed_style.style.captions.highlight_style = highlight;
    const result = readLearned(data, schema);
    assert.equal(result.style.captions.highlight_style, highlight);
    assert.equal(result.rows.base.find((row) => row.field === 'captions.highlight_style').value,
      highlight);
  }
});

test('older learned documents and presets can omit the optional broll section', () => {
  const data = measurement();
  delete data.proposed_style.style.broll;
  for (const field of Object.keys(data.proposed_style.sources)) {
    if (field.startsWith('broll.')) delete data.proposed_style.sources[field];
  }
  const learned = readLearned(data, schema);
  assert.equal(Object.hasOwn(learned.style, 'broll'), false);
  const page = app();
  page.applyStartingStyle(learned.style, learned.style.name);
  assert.equal(Object.hasOwn(page.output(), 'broll'), false);
  const preset = {id: 'old', name: 'Old preset', style: structuredClone(learned.style)};
  assert.equal(Object.hasOwn(switchPreset(page.state, preset).next.style, 'broll'), false);
});

test('older learned sources treat broll values as inherited from the base', () => {
  const data = measurement();
  for (const field of Object.keys(data.proposed_style.sources)) {
    if (field.startsWith('broll.')) delete data.proposed_style.sources[field];
  }
  const before = JSON.stringify(data);
  const result = readLearned(data, schema);
  const rows = result.rows.base.filter((row) => row.field.startsWith('broll.'));
  assert.equal(rows.length, Object.keys(schema.properties.broll.properties).length);
  const emphasis = rows.find((row) => row.field === 'broll.emphasis_on');
  assert.equal(emphasis.label, 'broll.emphasis_on');
  assert.deepEqual(emphasis.value, []);
  assert.ok(rows.every((row) => row.reason === 'This choice is not measured from finished videos.'));
  assert.deepEqual(result.style.broll, data.proposed_style.style.broll);
  assert.equal(JSON.stringify(data), before);
});

test('older learned broll fills new fields from schema defaults without changing input', () => {
  const data = measurement();
  for (const key of ['cutout', 'still_push', 'still_push_scale', 'green_screen_layout', 'generated_pictures']) {
    delete data.proposed_style.style.broll[key];
    delete data.proposed_style.sources[`broll.${key}`];
  }
  const before = JSON.stringify(data);
  const result = readLearned(data, schema);
  for (const key of ['cutout', 'still_push', 'still_push_scale', 'green_screen_layout', 'generated_pictures']) {
    const field = `broll.${key}`;
    const row = result.rows.base.find((item) => item.field === field);
    assert.equal(result.style.broll[key], schema.properties.broll.properties[key].default);
    assert.equal(row.value, schema.properties.broll.properties[key].default);
    assert.equal(row.reason, 'This choice is not measured from finished videos.');
  }
  assert.equal(JSON.stringify(data), before);
  const invalid = structuredClone(data);
  invalid.proposed_style.style.broll.still_push = 'sideways';
  assert.throws(() => readLearned(invalid, schema), /style\.broll\.still_push: invalid choice/);
  const incomplete = structuredClone(data);
  incomplete.proposed_style.style.broll.still_push = 'in';
  assert.throws(() => readLearned(incomplete, schema), /broll\.still_push: missing or invalid source/);
});

test('older learned broll still requires its original fields', () => {
  const data = measurement();
  delete data.proposed_style.style.broll.overlay_motion;
  delete data.proposed_style.sources['broll.overlay_motion'];
  assert.throws(() => readLearned(data, schema), /broll\.overlay_motion: missing value/);
});

test('buyer output preserves broll and validates arrays when saving', () => {
  const learned = readLearned(measurement(), schema);
  const page = app();
  page.applyStartingStyle(learned.style, learned.style.name);
  const saved = JSON.parse(JSON.stringify(page.output()));
  assert.deepEqual(saved.broll, learned.style.broll);
  const errors = [];
  page.check(saved, schema, 'style', errors);
  assert.deepEqual(errors, []);
  saved.broll.avoid_tags = [];
  page.check(saved, schema, 'style', errors);
  assert.deepEqual(errors, []);
  page.state.style.broll.avoid_tags.push('new_tag');
  assert.deepEqual(learned.style.broll.avoid_tags, ['stock', 'busy_background']);
  assert.deepEqual(Array.from(page.state.startStyle.broll.avoid_tags), ['stock', 'busy_background']);
  for (const invalid of ['stock', [123], ['bad tag'], ['tag\n'], ['a'.repeat(41)], Array(21).fill('tag')]) {
    saved.broll.avoid_tags = invalid;
    const errors = [];
    page.check(saved, schema, 'style', errors);
    assert.ok(errors.some((error) => error.startsWith('style.broll.avoid_tags')));
  }
});

test('style page preserves a brand block when opening and saving a style', () => {
  const page = app();
  const style = defaults(schema);
  style.brand = {captions_font: 'Caption Face', titles_font: 'Title Face',
    cards_font: 'Card Face', accent_color: '#123ABC', logo: 'logo.png'};
  page.applyStartingStyle(style, 'Branded style');
  const saved = JSON.parse(JSON.stringify(page.output()));
  assert.deepEqual(saved.brand, style.brand);
  const errors = [];
  page.check(saved, schema, 'style', errors);
  assert.deepEqual(errors, []);
});

test('branded font and accent choices update the effective brand values', () => {
  const page = app();
  const style = defaults(schema);
  const own = {
    fonts: [style.captions.font, style.hook.font, style.graphics.font],
    colors: [style.captions.highlight_color, style.hook.accent_color, style.graphics.accent_color],
  };
  style.brand = {captions_font: 'Gelasio', titles_font: 'Alata',
    cards_font: 'Inter', accent_color: '#123ABC', logo: ''};
  page.applyStartingStyle(style, 'Branded style');
  assert.equal(page.effectiveStyle(page.state.style).captions.font, 'Gelasio');
  assert.equal(page.rolesOf(page.state.style).accent, '#123ABC');
  page.state.style = page.withFont(page.state.style, 'Anton SC');
  page.state.style = page.withPalette(page.state.style, {
    caption: '#FFFFFF', accent: '#ABCDEF', paper: '#111111', ink: '#EEEEEE',
  });
  const saved = JSON.parse(JSON.stringify(page.output()));
  assert.deepEqual([saved.brand.captions_font, saved.brand.titles_font, saved.brand.cards_font],
    ['Anton SC', 'Anton SC', 'Anton SC']);
  assert.equal(saved.brand.accent_color, '#ABCDEF');
  assert.deepEqual([saved.captions.font, saved.hook.font, saved.graphics.font], own.fonts);
  assert.deepEqual([saved.captions.highlight_color, saved.hook.accent_color, saved.graphics.accent_color], own.colors);
  assert.deepEqual([page.effectiveStyle(saved).captions.font, page.effectiveStyle(saved).hook.font,
    page.effectiveStyle(saved).graphics.font], ['Anton SC', 'Anton SC', 'Anton SC']);
  assert.equal(page.rolesOf(saved).accent, '#ABCDEF');
  const plain = defaults(schema);
  plain.brand = {captions_font: '', titles_font: '', cards_font: '', accent_color: '', logo: ''};
  const changed = page.withPalette(page.withFont(plain, 'Anton SC'),
    {caption: '#FFFFFF', accent: '#ABCDEF', paper: '#111111', ink: '#EEEEEE'});
  assert.deepEqual([changed.captions.font, changed.hook.font, changed.graphics.font],
    ['Anton SC', 'Anton SC', 'Anton SC']);
  assert.equal(changed.captions.highlight_color, '#ABCDEF');
  assert.equal(changed.brand.accent_color, '');
  const mixed = defaults(schema);
  mixed.brand = {captions_font: 'Gelasio', titles_font: '', cards_font: '', accent_color: '', logo: ''};
  const mixedChanged = page.withFont(mixed, 'Anton SC');
  assert.equal(mixedChanged.brand.captions_font, 'Anton SC');
  assert.equal(mixedChanged.captions.font, mixed.captions.font);
  assert.equal(mixedChanged.hook.font, 'Anton SC');
  assert.equal(mixedChanged.graphics.font, 'Anton SC');
});

test('switching presets retains the opened brand block and undo restores it', () => {
  const page = app();
  const style = defaults(schema);
  style.brand = {captions_font: 'Gelasio', titles_font: 'Alata',
    cards_font: 'Inter', accent_color: '#123ABC', logo: 'logo.png'};
  page.applyStartingStyle(style, 'Branded style');
  const preset = {id: 'other', name: 'Other preset', style: defaults(schema)};
  const {previous, next} = switchPreset(page.state, preset);
  assert.deepEqual(next.style.brand, style.brand);
  assert.deepEqual(next.startStyle.brand, style.brand);
  next.style.brand.captions_font = 'Anton SC';
  assert.equal(next.startStyle.brand.captions_font, 'Gelasio');
  assert.equal(undoPreset(next, previous).style.brand.captions_font, 'Gelasio');
});

test('style page opens older styles without a brand block', () => {
  const page = app();
  const style = defaults(schema);
  delete style.brand;
  page.applyStartingStyle(style, 'Older style');
  assert.equal(Object.hasOwn(page.output(), 'brand'), false);
  const errors = [];
  page.check(page.output(), schema, 'style', errors);
  assert.deepEqual(errors, []);
});

test('learned evidence renders enum values and comma separated tags as text', () => {
  const page = app();
  const host = page.element();
  const learned = readLearned(measurement(), schema);
  page.showLearned(learned, host);
  const text = (node) => typeof node === 'string' ? node : node.children.map(text).join(' ');
  assert.match(text(host), /Overlay position: side_right/);
  assert.match(text(host), /Tags to avoid: stock, busy_background \(Not measured\.\)/);
  const noTags = measurement();
  noTags.proposed_style.style.broll.avoid_tags = [];
  page.showLearned(readLearned(noTags, schema), host);
  assert.match(text(host), /Tags to avoid:  \(Not measured\.\)/);
});

test('wrong schema and older measurements without proposed style are refused', () => {
  assert.throws(() => readLearned({schema: 'other'}, schema), /measurements.json/);
  assert.throws(() => readLearned({schema: 'yap-studio/taste-measurements@1'}, schema), /complete proposed style/);
});

test('learned word limit is measured and font class explanation stays verbatim', () => {
  const data = measurement();
  const reason = 'Measured font class: serif, regular weight. Naming the exact typeface needs text recognition (OCR).';
  data.proposed_style.style.captions.max_words_per_phrase = 5;
  data.proposed_style.sources['captions.max_words_per_phrase'] = {
    source: 'measured', status: 'proposed', reason: null,
  };
  data.proposed_style.sources['captions.font'].reason = reason;
  const result = readLearned(data, schema);
  assert.equal(result.rows.measured.find((row) => row.field === 'captions.max_words_per_phrase').value, 5);
  assert.equal(result.rows.base.find((row) => row.field === 'captions.font').reason, reason);
});

test('invalid style values and missing leaves cannot enter the page', () => {
  for (const edit of [
    (data) => {data.proposed_style.style.hook.duration_s = 20;},
    (data) => {data.proposed_style.style.captions.highlight_style = 'unknown';},
    (data) => {delete data.proposed_style.style.captions.font;},
    (data) => {data.proposed_style.style.captions.text_color = '#XYZXYZ';},
    (data) => {data.proposed_style.style.internal = defaults(schema.properties.internal);},
    (data) => {data.proposed_style.style.broll.clip_placement = 'unknown';},
    (data) => {data.proposed_style.style.broll.overlay_box = 'unknown';},
    (data) => {data.proposed_style.style.broll.overlay_motion = 'unknown';},
    (data) => {delete data.proposed_style.style.broll.avoid_tags;},
    (data) => {data.proposed_style.style.broll.avoid_tags = 'stock';},
    (data) => {data.proposed_style.style.broll.avoid_tags = [123];},
    (data) => {data.proposed_style.style.broll.avoid_tags = ['bad tag'];},
    (data) => {data.proposed_style.style.broll.avoid_tags = ['tag\n'];},
    (data) => {data.proposed_style.style.broll.avoid_tags = ['a'.repeat(41)];},
    (data) => {data.proposed_style.style.broll.avoid_tags = Array(21).fill('tag');},
  ]) {
    const data = measurement();
    edit(data);
    assert.throws(() => readLearned(data, schema));
  }
});

test('sources must cover every leaf and agree with measured decisions', () => {
  for (const edit of [
    (data) => {delete data.proposed_style.sources['captions.font'];},
    (data) => {delete data.proposed_style.sources['broll.avoid_tags'];},
    (data) => {data.proposed_style.sources['captions.font'].source = 'unknown';},
    (data) => {data.proposed_style.sources['captions.text_color'].status = 'partial';},
    (data) => {data.proposed_style.sources['captions.font'].status = 'proposed';},
    (data) => {data.proposed_style.sources['captions.font'].reason = null;},
    (data) => {data.proposed_style.sources['unknown.key'] = {source: 'base'};},
    (data) => {data.proposed_style.base.kind = 'unknown';},
  ]) {
    const data = measurement();
    edit(data);
    assert.throws(() => readLearned(data, schema));
  }
});

test('classic script exports the same pure validator without a browser', () => {
  const sandbox = {window: {}};
  vm.runInNewContext(pageFile('learned.js'), sandbox);
  const actual = sandbox.window.YapLearned.readLearned(measurement(), schema);
  assert.deepEqual(JSON.parse(JSON.stringify(actual)), readLearned(measurement(), schema));
});

test('page loads learned helper before app and imports from a local file', () => {
  const scripts = [...pageFile('index.html').matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/g)];
  assert.deepEqual(scripts.map((match) => match[1]), ['style-data.js', 'phrases.js', 'learned.js', 'looks.js', 'caption-ports.js', 'app.js']);
  const app = pageFile('app.js');
  assert.match(app, /Open a learned style/);
  assert.match(app, /file\.files\[0\]\.text\(\)/);
  assert.match(app, /applyStartingStyle\(learned\.style, learned\.style\.name\)/);
  assert.match(app, /YapLearned\.switchPreset\(state, byId\[id\]\)/);
  assert.match(app, /YapLearned\.undoPreset\(state, before\)/);
  assert.match(app, /state\.learned = learned/);
  assert.match(app, /refreshLearnedEvidence\(\);/);
  assert.doesNotMatch(pageFile('learned.js'), /innerHTML|fetch\(|eval\(/);
});

test('preset undo restores edited learned style, starting point and evidence', () => {
  const learned = readLearned(measurement(), schema);
  const opened = {
    presetId: null, style: structuredClone(learned.style),
    startStyle: structuredClone(learned.style), startName: learned.style.name,
    custom: {caption: '#FFFFFF', accent: '#123456', paper: '#ABCDEF', ink: '#000000'},
    learned, name: 'My style', useNext: true,
  };
  opened.style.captions.size_px += 6;
  const before = structuredClone(opened);
  const preset = {id: 'example', name: 'Example preset', style: defaults(schema)};
  preset.style.captions.size_px += 12;
  preset.style.broll.avoid_tags = ['preset_tag'];
  const {previous, next} = switchPreset(opened, preset);
  assert.equal(next.learned, null);
  assert.equal(next.presetId, 'example');
  assert.equal(next.startName, 'Example preset');
  assert.equal(next.style.captions.size_px, preset.style.captions.size_px);
  assert.equal(next.startStyle.captions.size_px, preset.style.captions.size_px);
  assert.deepEqual(next.style.broll, preset.style.broll);
  assert.deepEqual(next.startStyle.broll, preset.style.broll);
  assert.equal(Object.hasOwn(next.style, 'internal'), false);
  assert.deepEqual(opened, before);
  next.style.captions.size_px += 3;
  next.style.broll.avoid_tags.push('new_tag');
  assert.deepEqual(next.startStyle.broll.avoid_tags, ['preset_tag']);
  assert.deepEqual(preset.style.broll.avoid_tags, ['preset_tag']);
  assert.notEqual(next.style.captions.size_px, next.startStyle.captions.size_px);
  const restored = undoPreset(next, previous);
  assert.deepEqual(restored, before);
  assert.deepEqual(restored.learned.rows, learned.rows);
  restored.style.captions.size_px += 1;
  restored.learned.rows.base[0].reason = 'Edited after undo';
  assert.deepEqual(opened, before);
  assert.deepEqual(previous.learned.rows, learned.rows);
});

test('reset from a restored learned starting point keeps evidence', () => {
  const learned = readLearned(measurement(), schema);
  const state = {presetId: null, style: structuredClone(learned.style), custom: {},
    startStyle: structuredClone(learned.style), startName: learned.style.name, learned};
  state.style.captions.size_px += 8;
  const {previous, next} = switchPreset(state, {id: 'example', name: 'Example', style: defaults(schema)});
  const restored = undoPreset(next, previous);
  const reset = {...restored, style: structuredClone(restored.startStyle)};
  assert.deepEqual(reset.style, learned.style);
  assert.deepEqual(reset.learned, learned);
  const app = pageFile('app.js');
  const resetSource = app.match(/function reset\(\) \{.*?\n  \}/s)[0];
  assert.doesNotMatch(resetSource, /learned\s*=/);
});
