import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as screenChoices from '../../src/yap_studio/gallery/static/choices.mjs';
import { withWords, withBrand, withKind, kindOn, validHex, tileWindow, choosePreset,
  startingName, frameWindow, withCardStyle, withColors, colorsOf, luminance, contrast,
  mix, withShadow, shadowMode, withPlacement, placementMode, sizes, nearestSize,
  withBackground, withBackgroundOpacity, withBackgroundCorners, withFaceAvoidance,
  captionTextColor, captionContrastNote, posterWords, fontFamily, reserveSavebar,
  captionsFont, graphicsFonts, fontMode, withCaptionsFont, withGraphicsFont,
  withFont, fontPair, withFontPair,
  withGrade, stillUrl, gradeLooks, coverPoint, samplePatch, patchColor } from '../../src/yap_studio/gallery/static/choices.mjs';
const style = { captions: { size_px: 92, position: 'bottom', top_px: 1500 }, graphics: { card_kinds_off: [] }, brand: {} };
test('card labels use built-in names and registry state labels', () => {
  const state = { card_kinds: {
    stat: { label: 'Ignore this' },
    'hf:sample': { label: 'Sample headline', registry: true },
  } };
  assert.equal(screenChoices.kindLabel(state, 'stat'), 'Stat');
  assert.equal(screenChoices.kindLabel(state, 'hf:sample'), 'Sample headline');
});
for (const [name, source, preview, videoType, expected] of [
  ['person', { broll: { cutout: 'person' } }, { screen: 'green', screen_checked: true }, 'green_screen', null],
  ['screen', { broll: { cutout: 'screen' } }, { screen: null, screen_checked: true }, 'talking_head', 'green or blue'],
  ['detected green', {}, { screen: 'green', screen_checked: true }, 'talking_head', 'green'],
  ['detected blue', {}, { screen: 'blue', screen_checked: true }, null, 'blue'],
  ['checked and clear', {}, { screen: null, screen_checked: true }, 'green_screen', null],
  ['unchecked green default', {}, { screen: null, screen_checked: false }, 'green_screen', 'green or blue'],
  ['unchecked other default', {}, { screen: null, screen_checked: false }, 'talking_head', null],
  ['nothing', undefined, undefined, undefined, null],
]) test(`keyedScreen follows ${name}`, () => {
  assert.equal(screenChoices.keyedScreen(source, preview, videoType), expected);
});

test('momentBlock explains only Clone flash on a keyed take', () => {
  for (const entry of screenChoices.momentsSlots) {
    assert.equal(screenChoices.momentBlock(entry.id, {}, { screen: 'blue', screen_checked: true }, null),
      entry.id === 'clone_flash' ? 'Needs a take filmed without a blue screen.' : null);
    assert.equal(screenChoices.momentBlock(entry.id, {}, { screen: null, screen_checked: true }, 'green_screen'), null);
  }
  assert.equal(screenChoices.momentBlock('clone_flash', {}, {}, 'green_screen'),
    'Needs a take filmed without a green or blue screen.');
});

const fontStyle = {
  name: 'Synthetic look', captions: { font: 'Inter', letter_spacing_px: 2, width_scale: .85 },
  hook: { font: 'Gelasio', style: 'quiet_serif' }, graphics: { font: 'Inter', style: 'paper' },
  brand: { captions_font: '', titles_font: '', cards_font: '', accent_color: '#123456' },
};
const screenSchema = JSON.parse(readFileSync(new URL('../../src/yap_studio/style/schema-v1.json', import.meta.url)));
function schemaDefaults(rule) {
  if ('default' in rule && rule.type !== 'object') return structuredClone(rule.default);
  if ('const' in rule) return rule.const;
  if (rule.type === 'object') return Object.fromEntries(Object.entries(rule.properties)
    .map(([key, child]) => [key, schemaDefaults(child)]).filter(([, value]) => value !== undefined));
}
const screenStyle = { ...schemaDefaults(screenSchema), schema_version: 1, name: 'Clean Minimal' };

test('more sounds are independent switches and leave an empty summary unchanged', () => {
  const original = structuredClone(screenStyle);
  assert.deepEqual(original.music.sound_extras, []);
  assert.equal(original.music.sound_effects_fallback, 'made');
  for (const name of ['typing', 'message-ping', 'camera-shutter']) {
    const next = screenChoices.withSoundExtra(original, name, true);
    assert.deepEqual(next.music.sound_extras, [name]);
    assert.deepEqual(screenChoices.withSoundExtra(next, name, true), next);
    assert.deepEqual(screenChoices.withSoundExtra(next, name, false), original);
    assert.match(screenChoices.screenSummary('sound', next), /1 more sound/);
  }
  assert.throws(() => screenChoices.withSoundExtra(original, 'applause', true));
  assert.throws(() => screenChoices.withSoundExtra(original, 'typing', 'yes'));
  assert.equal(screenChoices.screenSummary('sound', original),
    screenChoices.screenSummary('sound', { ...original, music: { ...original.music, sound_extras: undefined } }));
});

test('no music setup survives looks until a Sound choice', () => {
  const preset = { ...structuredClone(screenStyle), music: { ...screenStyle.music, level: .09 } };
  const muted = screenChoices.withPreset(screenStyle, preset, 0);
  assert.equal(muted.music.level, 0);
  assert.equal(screenChoices.musicLevels.find((entry) => entry.value === muted.music.level).name, 'No music');
  assert.match(screenChoices.screenSummary('sound', muted), /^Music off/);
  const chosen = screenChoices.withMusicLevel(muted, .12);
  assert.equal(chosen.music.level, .12);
  assert.equal(screenChoices.withPreset(chosen, preset, chosen.music.level).music.level, .12);
  assert.equal(screenChoices.withPreset(screenStyle, preset).music.level, .09);
  assert.deepEqual(muted.captions, preset.captions);
});

test('every visible schema option has a plain label', () => {
  const walk = (rule, path = '') => {
    if (rule.enum && !path.startsWith('internal.')) for (const option of rule.enum) {
      if (typeof option !== 'string') continue;
      const label = screenChoices.labels?.[option];
      assert.ok(label, `${path}: ${option} needs a label`);
      assert.ok(!label.includes('_'), `${path}: ${option} has a raw label`);
    }
    for (const [key, child] of Object.entries(rule.properties ?? {})) walk(child, path ? `${path}.${key}` : key);
    if (rule.items) walk(rule.items, `${path}[]`);
  };
  walk(screenSchema);
  for (const kind of ['big_number', 'edit_timeline', 'storyboard', 'jiggle_delete', 'file_drop',
    'video_card', 'prompt_bar', 'comment_cta', 'comment_drop', 'icon_strike', 'choices']) assert.ok(screenChoices.labels?.[kind]);
});
function onlyFields(source, apply, fields) {
  const before = structuredClone(source), expected = structuredClone(source);
  for (const [path, value] of Object.entries(fields)) {
    const parts = path.split('.'), key = parts.pop();
    const section = parts.reduce((node, field) => node[field] ??= {}, expected);
    section[key] = value;
  }
  const changed = apply(source);
  assert.notEqual(changed, source);
  assert.deepEqual(changed, expected);
  assert.deepEqual(source, before);
  assert.deepEqual(screenChoices.check(changed, screenSchema), []);
  return changed;
}

test('Giant and Kinetic hook slots change only their own field and require engine support', () => {
  for (const [slots, id, name, note, path, value] of [
    ['captionLookSlots', 'giant', 'Giant', 'Big words in a solid block', 'captions.highlight_style', 'giant'],
    ['openingSlots', 'kinetic', 'Kinetic hook', 'Big words land as you say them', 'hook.style', 'kinetic'],
  ]) {
    const entry = screenChoices[slots].find((entry) => entry.id === id);
    assert.ok(entry);
    assert.equal(entry.name, name);
    assert.equal(entry.note, note);
    assert.deepEqual(entry.requires, [path, value]);
    assert.equal(entry.selected(screenStyle), false);
    assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
    const changed = onlyFields(screenStyle, entry.apply, { [path]: value });
    assert.equal(entry.selected(changed), true);
    const [section, field] = path.split('.'), older = structuredClone(screenSchema);
    older.properties[section].properties[field].enum = older.properties[section].properties[field].enum.filter((item) => item !== value);
    assert.equal(screenChoices.optionAvailable(entry, older), false);
    delete older.properties[section].properties[field];
    assert.equal(screenChoices.optionAvailable(entry, older), false);
  }
});

test('Chapter cards starts off and switches only its style, including on older styles', () => {
  const entry = screenChoices.openingMoreSlots.find((entry) => entry.id === 'chapter');
  assert.ok(entry);
  assert.equal(entry.name, 'Chapter cards');
  assert.equal(entry.type, 'switch');
  assert.equal(entry.note, 'A numbered title over you where each new part starts');
  assert.deepEqual(entry.requires, ['chapters.style', 'title_card']);
  assert.equal(entry.selected(screenStyle), false);
  assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
  const older = structuredClone(screenSchema); delete older.properties.chapters;
  assert.equal(screenChoices.optionAvailable(entry, older), false);
  const missing = structuredClone(screenStyle); delete missing.chapters;
  assert.equal(entry.selected(missing), false);
  for (const source of [missing, { ...screenStyle, chapters: { style: 'none', hold_s: 2.7 } }]) {
    const on = onlyFields(source, (style) => entry.apply(style, true), { 'chapters.style': 'title_card' });
    assert.equal(entry.selected(on), true);
    const summary = screenChoices.screenSummary('opening', source);
    assert.equal(screenChoices.screenSummary('opening', on), summary + ' · chapter cards on');
    assert.equal(screenChoices.changeSummary(on, source), 'Changed Opening from Clean Minimal');
    const off = onlyFields(on, (style) => entry.apply(style, false), { 'chapters.style': 'none' });
    assert.equal(entry.selected(off), false);
    assert.equal(screenChoices.screenSummary('opening', off), summary);
  }
  const noTitle = { ...screenStyle, hook: { ...screenStyle.hook, style: 'none' } };
  assert.equal(screenChoices.screenSummary('opening', entry.apply(noTitle, true)), 'No title · chapter cards on');
});

test('the four effect labels use the approved names', () => {
  for (const [value, name] of Object.entries({ giant: 'Giant captions', kinetic: 'Kinetic hook',
    gravity_drop: 'Gravity drop', title_card: 'Chapter cards' })) assert.equal(screenChoices.labels[value], name);
});

test('Giant takes priority over other caption looks and leaving it clears only its highlight', () => {
  for (const entrance of ['slide', 'rise']) for (const words of [1, 3]) for (const background of ['none', 'box', 'band']) {
    const source = structuredClone(screenStyle);
    Object.assign(source.captions, { entrance, max_words_per_phrase: words, background_mode: background });
    const giant = onlyFields(source, screenChoices.captionLookSlots.find((entry) => entry.id === 'giant').apply,
      { 'captions.highlight_style': 'giant' });
    assert.equal(screenChoices.captionLook(giant), 'giant');
    assert.equal(screenChoices.screenSummary('captions', giant), 'Giant');
    for (const look of ['plain', 'box', 'band', 'one', 'rise']) {
      const expected = screenChoices.withCaptionLook(source, look);
      expected.captions.highlight_style = 'none';
      const changed = screenChoices.withCaptionLook(giant, look);
      assert.deepEqual(changed, expected);
      assert.equal(screenChoices.captionLook(changed), look);
      assert.equal(giant.captions.highlight_style, 'giant');
    }
  }
  for (const look of ['plain', 'box', 'band', 'one', 'rise']) {
    const source = structuredClone(screenStyle); source.captions.highlight_style = 'pill';
    assert.equal(screenChoices.withCaptionLook(source, look).captions.highlight_style, 'pill');
  }
});

test('Flow look choices change only flow_look and summarize under Cards', () => {
  const group = screenChoices.cardGroupSlots.find((entry) => entry.id === 'flow-look');
  assert.equal(group.name, 'Flow look');
  assert.deepEqual(group.options.map((entry) => entry.name), ['Boxes', 'Tags']);
  assert.deepEqual(group.options.map((entry) => entry.note),
    ['Drawn boxes in a card', 'Stacked tags with an arrow']);
  for (const entry of group.options) {
    assert.equal(entry.card, 'flow');
    assert.deepEqual(entry.requires, ['graphics.flow_look', entry.id]);
    const chosen = onlyFields(screenStyle, entry.apply, { 'graphics.flow_look': entry.id });
    assert.equal(entry.selected(chosen), true);
    assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
    assert.equal(screenChoices.screenSummary('cards', chosen).endsWith(' · flow tags'), entry.id === 'tags');
    if (entry.id === 'tags') assert.equal(screenChoices.changeSummary(chosen, screenStyle), 'Changed Cards from Clean Minimal');
  }
  const old = structuredClone(screenStyle); delete old.graphics.flow_look;
  assert.equal(group.options[0].selected(old), true);
});
test('Jump cuts follow the schema and change only their named field', () => {
  const { movesMoreSlots, withJumpCuts, screenSummary, changeSummary } = screenChoices;
  const group = movesMoreSlots.find((entry) => entry.id === 'jump-cuts');
  assert.equal(movesMoreSlots.filter((entry) => entry.id === 'jump-cuts').length, 1);
  assert.equal(group.name, 'Jump cuts');
  assert.equal(group.note, 'What happens where a pause or a retake was cut out.');
  assert.deepEqual(group.options.map((entry) => entry.id), screenSchema.properties.moves.properties.jump_cuts.enum);
  assert.deepEqual(group.options.map((entry) => [entry.name, entry.note]), [
    ['Hard cut', 'The cut stays as it is'], ['Punch', 'A quick push in on every jump'],
    ['Smooth (punch on big moves)', 'Small jumps glide, big ones punch in'],
    ['Step in', 'A closer framing on each cut, then back wide'],
  ]);
  const missing = structuredClone(screenStyle); delete missing.moves;
  assert.equal(group.options[0].selected(missing), true);
  assert.equal(group.value(missing), 'Hard cut');
  for (const zoom_mode of ['none', 'punches', 'punches_and_push']) for (const entry of group.options) {
    const source = { ...screenStyle, moves: { ...screenStyle.moves, zoom_mode } };
    assert.deepEqual(entry.requires, ['moves.jump_cuts', entry.id]);
    assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
    const chosen = onlyFields(source, entry.apply, { 'moves.jump_cuts': entry.id });
    assert.equal(entry.selected(chosen), true);
    assert.equal(group.options.filter((option) => option.selected(chosen)).length, 1);
    assert.equal(group.value(chosen), entry.name);
    assert.equal(screenSummary('moves', chosen), screenSummary('moves', source)
      + ({ hard: '', punch: ' · punch on jump cuts', smooth: ' · smooth jump cuts', step_in: ' · step in' })[entry.id]);
    if (entry.id !== 'hard') assert.equal(changeSummary(chosen, source), 'Changed Moves from Clean Minimal');
  }
  onlyFields(missing, (style) => withJumpCuts(style, 'smooth'), { 'moves.jump_cuts': 'smooth' });
  for (const mode of ['other', null, 1]) assert.throws(() => withJumpCuts(screenStyle, mode));
  const older = structuredClone(screenSchema); delete older.properties.moves.properties.jump_cuts;
  assert.ok(group.options.every((entry) => !screenChoices.optionAvailable(entry, older)));
});
test('opening push switch changes only its field and follows schema support', () => {
  const { withOpeningPush, screenSummary, changeSummary, optionAvailable } = screenChoices;
  const entry = screenChoices.movesMoreSlots.find((entry) => entry.id === 'opening-push');
  assert.equal(entry.preview, 'opening_push');
  assert.equal(entry.selected(screenStyle), false);
  const chosen = onlyFields(screenStyle, (style) => entry.apply(style, true), { 'moves.opening_push': true });
  assert.equal(entry.selected(chosen), true);
  assert.deepEqual(entry.apply(chosen, false), screenStyle);
  assert.equal(optionAvailable(entry, screenSchema), true);
  const older = structuredClone(screenSchema); delete older.properties.moves.properties.opening_push;
  assert.equal(optionAvailable(entry, older), false);
  assert.equal(screenSummary('moves', chosen), screenSummary('moves', screenStyle) + ' · opening push');
  assert.equal(screenSummary('moves', screenChoices.withJumpCuts(chosen, 'smooth')),
    screenSummary('moves', screenStyle) + ' · smooth jump cuts · opening push');
  assert.equal(screenSummary('moves', screenChoices.withJumpCuts(chosen, 'step_in')),
    screenSummary('moves', screenStyle) + ' · step in · opening push');
  assert.equal(changeSummary(chosen, screenStyle), 'Changed Moves from Clean Minimal');
  for (const value of [null, 1, 'true', undefined]) assert.throws(() => withOpeningPush(screenStyle, value));
});
test('Moves follow the schema and each choice changes only its named field', () => {
  const { zoomStrengths, withZoomMode, withPunchScale, screenSummary, changeSummary } = screenChoices;
  const movesSlots = screenChoices.movesSlots.filter((entry) => !entry.type);
  assert.deepEqual(movesSlots.map((entry) => entry.id), screenSchema.properties.moves.properties.zoom_mode.enum);
  assert.deepEqual(zoomStrengths.map((entry) => entry.requires[1]), screenSchema.properties.moves.properties.punch_scale.enum);
  for (const entry of movesSlots) {
    assert.deepEqual(entry.requires, ['moves.zoom_mode', entry.id]);
    assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
    const chosen = onlyFields(screenStyle, entry.apply, { 'moves.zoom_mode': entry.id });
    assert.equal(entry.selected(chosen), true);
    assert.equal(movesSlots.filter((option) => option.selected(chosen)).length, 1);
    for (const strength of zoomStrengths) {
      const scale = strength.requires[1];
      const sized = onlyFields(chosen, strength.apply, { 'moves.punch_scale': scale });
      assert.equal(strength.selected(sized), true);
      assert.equal(screenSummary('moves', sized), (entry.id === 'none' ? 'No zooms'
        : `${entry.id === 'punches' ? 'Punches' : 'Punches and a slow push'} at ${scale}x`) + ' · all transitions');
    }
  }
  const missing = structuredClone(screenStyle); delete missing.moves;
  assert.equal(screenSummary('moves', missing), 'No zooms · all transitions');
  onlyFields(missing, (style) => withZoomMode(style, 'punches'), { 'moves.zoom_mode': 'punches' });
  onlyFields(missing, (style) => withPunchScale(style, 2), { 'moves.punch_scale': 2 });
  assert.equal(withZoomMode(withPunchScale(screenStyle, 2), 'none').moves.punch_scale, 2);
  for (const mode of ['other', null, 1]) assert.throws(() => withZoomMode(screenStyle, mode));
  for (const scale of [1, 1.6, 3, '2', null]) assert.throws(() => withPunchScale(screenStyle, scale));
  for (const apply of [(style) => withZoomMode(style, 'punches'), (style) => withPunchScale(style, 2)]) {
    assert.equal(changeSummary(apply(screenStyle), screenStyle), 'Changed Moves from Clean Minimal');
  }
  const combined = screenChoices.withMoment(withZoomMode(screenChoices.withBrollLayout(screenStyle, 'box'), 'punches'), 'outline', true);
  assert.equal(changeSummary(combined, screenStyle), 'Changed B-roll, Moves, Moments from Clean Minimal');
  const older = structuredClone(screenSchema); delete older.properties.moves;
  assert.ok([...movesSlots, ...zoomStrengths].every((entry) => !screenChoices.optionAvailable(entry, older)));
});
test('transition switches write only their array in schema order and keep explicit empty lists', () => {
  const { transitionLooks, withTransition, transitionsSummary, screenSummary, changeSummary } = screenChoices;
  const order = screenSchema.properties.moves.properties.transitions.items.enum;
  assert.deepEqual(transitionLooks.map((entry) => entry.id), order);
  const group = screenChoices.movesSlots.find((entry) => entry.id === 'transitions');
  assert.equal(group.type, 'switch');
  assert.equal(group.switches, transitionLooks);
  assert.equal(group.value(screenStyle), 'All three');
  for (const entry of transitionLooks) {
    assert.deepEqual(entry.requires, ['moves.transitions', [entry.id]]);
    assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
    assert.equal(entry.selected(screenStyle), true);
    const off = onlyFields(screenStyle, (style) => entry.apply(style, false), { 'moves.transitions': order.filter((id) => id !== entry.id) });
    assert.equal(entry.selected(off), false);
    assert.deepEqual(onlyFields(off, (style) => entry.apply(style, true), { 'moves.transitions': order }), screenStyle);
    assert.equal(changeSummary(off, screenStyle), 'Changed Moves from Clean Minimal');
  }
  const absent = structuredClone(screenStyle); delete absent.moves;
  onlyFields(absent, (style) => withTransition(style, 'cinematic_zoom', false), { 'moves.transitions': ['whip_pan', 'light_leak'] });
  const duplicates = { moves: { zoom_mode: 'punches', punch_scale: 2, transitions: ['light_leak', 'whip_pan', 'whip_pan'] } };
  assert.deepEqual(withTransition(duplicates, 'cinematic_zoom', true).moves.transitions, order);
  assert.deepEqual(duplicates.moves.transitions, ['light_leak', 'whip_pan', 'whip_pan']);
  let empty = screenStyle;
  for (const entry of transitionLooks) empty = withTransition(empty, entry.id, false);
  assert.deepEqual(empty.moves.transitions, []);
  assert.deepEqual(withTransition(empty, 'whip_pan', false).moves.transitions, []);
  assert.equal(transitionsSummary(empty), 'Off');
  assert.equal(screenSummary('moves', empty), 'No zooms · no transitions');
  assert.equal(screenSummary('moves', withTransition(screenStyle, 'light_leak', false)), 'No zooms · Whip and Zoom transitions');
  assert.equal(screenSummary('moves', withTransition(empty, 'light_leak', true)), 'No zooms · Light leak transitions');
  assert.equal(screenSummary('moves', screenChoices.withPunchScale(screenChoices.withZoomMode(empty, 'punches_and_push'), 2)),
    'Punches and a slow push at 2x · no transitions');
  const older = structuredClone(screenSchema); delete older.properties.moves.properties.transitions;
  assert.ok([group, ...transitionLooks].every((entry) => !screenChoices.optionAvailable(entry, older)));
  assert.throws(() => withTransition(screenStyle, 'missing', true));
  assert.throws(() => withTransition(screenStyle, 'whip_pan', 1));
});
test('Moments follow the schema order and each switch changes only its sorted array', () => {
  const entries = screenChoices.momentsSlots;
  assert.deepEqual(entries.map((entry) => entry.id),
    ['color_pop', 'background_blur', 'spotlight', 'sticker_freeze', 'outline', 'clone_flash']);
  assert.deepEqual(entries.map((entry) => entry.id), screenSchema.properties.moments.properties.styles.items.enum);
  for (const entry of entries) {
    assert.deepEqual(entry.requires, ['moments.styles', [entry.id]]);
    assert.equal(entry.selected(screenStyle), false);
    const on = onlyFields(screenStyle, (style) => entry.apply(style, true), { 'moments.styles': [entry.id] });
    assert.equal(entry.selected(on), true);
    onlyFields(on, (style) => entry.apply(style, false), { 'moments.styles': [] });
  }
  const missing = structuredClone(screenStyle); delete missing.moments;
  onlyFields(missing, (style) => screenChoices.withMoment(style, 'outline', false), { 'moments.styles': [] });
  const original = { moments: { styles: ['spotlight', 'color_pop', 'spotlight'] } };
  const before = structuredClone(original);
  assert.deepEqual(screenChoices.withMoment(original, 'outline', true).moments.styles, ['color_pop', 'outline', 'spotlight']);
  assert.deepEqual(screenChoices.withMoment(original, 'color_pop', false).moments.styles, ['spotlight']);
  assert.deepEqual(original, before);
  assert.throws(() => screenChoices.withMoment(screenStyle, 'missing', true));
  assert.throws(() => screenChoices.withMoment(screenStyle, 'outline', 1));
  const on = screenChoices.withMoment(screenChoices.withMoment(screenStyle, 'spotlight', true), 'color_pop', true);
  assert.equal(screenChoices.screenSummary('moments', screenStyle), 'All off');
  assert.equal(screenChoices.screenSummary('moments', on), 'Color pop, Spotlight');
  assert.equal(screenChoices.changeSummary(on, screenStyle), 'Changed Moments from Clean Minimal');
});
test('the B-roll letters switch requires its engine field and starts off', () => {
  const entry = screenChoices.brollMoreSlots.find((entry) => entry.id === 'video-letters');
  assert.equal(entry.name, 'Video in letters');
  assert.equal(entry.type, 'switch');
  assert.equal(entry.selected(screenStyle), false);
  assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
  const older = structuredClone(screenSchema); delete older.properties.broll.properties.emphasis_on;
  assert.equal(screenChoices.optionAvailable(entry, older), false);
  onlyFields(screenStyle, (style) => entry.apply(style, true), { 'broll.emphasis_on': ['video_letters'] });
});

test('emphasis changes only its sorted unique array and keeps empty arrays', () => {
  const on = onlyFields(screenStyle, (style) => screenChoices.withEmphasis(style, 'video_letters', true),
    { 'broll.emphasis_on': ['video_letters'] });
  onlyFields(on, (style) => screenChoices.withEmphasis(style, 'video_letters', false), { 'broll.emphasis_on': [] });
  const missing = structuredClone(screenStyle); delete missing.broll.emphasis_on;
  onlyFields(missing, (style) => screenChoices.withEmphasis(style, 'video_letters', false), { 'broll.emphasis_on': [] });
  const synthetic = { broll: { emphasis_on: ['test-z', 'video_letters', 'test-a', 'video_letters'] } };
  const original = structuredClone(synthetic);
  assert.deepEqual(screenChoices.withEmphasis(synthetic, 'video_letters', true).broll.emphasis_on, ['test-a', 'test-z', 'video_letters']);
  assert.deepEqual(screenChoices.withEmphasis(synthetic, 'video_letters', false).broll.emphasis_on, ['test-a', 'test-z']);
  assert.deepEqual(synthetic, original);
  assert.throws(() => screenChoices.withEmphasis(screenStyle, 'test-effect', true));
  assert.throws(() => screenChoices.withEmphasis(screenStyle, 'video_letters', 1));
});

test('caption look uses 3D camera, entrance, words, band, box and plain in that order', () => {
  for (const camera_3d of [false, true]) for (const entrance of ['slide', 'rise']) for (const words of [1, 3, 5, 8]) {
    for (const background of ['none', 'box', 'band']) {
      const candidate = structuredClone(screenStyle);
      Object.assign(candidate.captions, { camera_3d, entrance, max_words_per_phrase: words, background_mode: background });
      const expected = camera_3d ? 'camera_3d' : entrance === 'rise' ? 'rise' : words === 1 ? 'one'
        : background === 'none' ? 'plain' : background;
      const before = structuredClone(candidate);
      assert.equal(screenChoices.captionLook(candidate), expected);
      assert.deepEqual(candidate, before);
    }
  }
});

test('caption main choices write only their named fields and reset only competing choices', () => {
  for (const camera_3d of [false, true, undefined]) for (const entrance of ['slide', 'rise', undefined]) for (const words of [1, 3, 5, 8]) {
    for (const background of ['none', 'box', 'band']) for (const look of ['plain', 'box', 'band', 'one', 'rise', 'camera_3d']) {
      const candidate = structuredClone(screenStyle);
      Object.assign(candidate.captions, { max_words_per_phrase: words,
        max_characters_per_phrase: 31, background_mode: background });
      if (entrance === undefined) delete candidate.captions.entrance;
      else candidate.captions.entrance = entrance;
      if (camera_3d === undefined) delete candidate.captions.camera_3d;
      else candidate.captions.camera_3d = camera_3d;
      const fields = { 'captions.camera_3d': look === 'camera_3d' };
      if (look === 'rise') fields['captions.entrance'] = 'rise';
      else if (look !== 'camera_3d') {
        if (entrance === 'rise') fields['captions.entrance'] = 'slide';
        if (look !== 'one') fields['captions.background_mode'] = look === 'plain' ? 'none' : look;
        if (look === 'one' || words === 1) {
          fields['captions.max_words_per_phrase'] = look === 'one' ? 1 : 3;
          fields['captions.max_characters_per_phrase'] = look === 'one' ? 16 : 18;
        }
      }
      const changed = onlyFields(candidate, (style) => screenChoices.withCaptionLook(style, look), fields);
      assert.equal(screenChoices.captionLook(changed), look);
    }
  }
  assert.throws(() => screenChoices.withCaptionLook(screenStyle, 'constructor'));
});

test('3D camera slot requires a boolean schema field and summarizes under Captions', () => {
  const entry = screenChoices.captionLookSlots.find((entry) => entry.id === 'camera_3d');
  assert.equal(entry.id, 'camera_3d');
  assert.equal(entry.name, '3D camera');
  assert.equal(entry.note, 'Words in depth, camera moves');
  assert.equal(entry.preview, 'camera_3d');
  assert.deepEqual(entry.requires, ['captions.camera_3d', true]);
  const changed = onlyFields(screenStyle, entry.apply, { 'captions.camera_3d': true });
  assert.equal(entry.selected(changed), true);
  assert.equal(entry.selected(screenStyle), false);
  assert.equal(screenChoices.screenSummary('captions', changed), '3D camera');
  assert.equal(screenChoices.changeSummary(changed, screenStyle), 'Changed Captions from Clean Minimal');
  assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
  assert.equal(screenChoices.optionAvailable({ requires: ['captions.camera_3d', false] }, screenSchema), true);
  assert.equal(screenChoices.optionAvailable({ requires: ['captions.camera_3d', 'true'] }, screenSchema), false);
  const older = structuredClone(screenSchema); delete older.properties.captions.properties.camera_3d;
  assert.equal(screenChoices.optionAvailable(entry, older), false);
});

test('Giant and 3D camera are alternatives and preserve the other caption fields', () => {
  const giant = screenChoices.captionLookSlots.find((entry) => entry.id === 'giant');
  const camera = screenChoices.captionLookSlots.find((entry) => entry.id === 'camera_3d');
  const source = structuredClone(screenStyle);
  Object.assign(source.captions, { entrance: 'rise', max_words_per_phrase: 5,
    max_characters_per_phrase: 31, size_px: 83, background_mode: 'box', background_opacity: .21 });
  const big = onlyFields(source, giant.apply, { 'captions.highlight_style': 'giant' });
  const depth = onlyFields(big, camera.apply, {
    'captions.camera_3d': true, 'captions.highlight_style': 'none',
  });
  const back = onlyFields(depth, giant.apply, {
    'captions.camera_3d': false, 'captions.highlight_style': 'giant',
  });
  assert.deepEqual(back, big);
  for (const [style, id] of [[big, 'giant'], [depth, 'camera_3d'], [back, 'giant']]) {
    assert.deepEqual(screenChoices.captionLookSlots.filter((entry) => entry.selected(style)).map((entry) => entry.id), [id]);
  }
});

test('3D colors show and edit the hero accent instead of the retained caption highlight', () => {
  const source = structuredClone(screenStyle);
  Object.assign(source.captions, { camera_3d: true, highlight_color: '#F08A45', highlight_style: 'pill' });
  source.hook.accent_color = '#12ABCD';
  source.brand = {};
  assert.equal(colorsOf(source).accent, '#12ABCD');
  assert.equal(colorsOf(withColors(source, colorsOf(source))).accent, '#12ABCD');
  const edited = withColors(source, { ...colorsOf(source), accent: '#909090' }, ['accent']);
  assert.equal(edited.hook.accent_color, '#909090');
  assert.equal(source.captions.highlight_color, '#F08A45');
  source.brand.accent_color = '#22BBFF';
  assert.equal(colorsOf(source).accent, '#22BBFF');
  source.captions.camera_3d = false;
  delete source.brand.accent_color;
  assert.equal(colorsOf(source).accent, '#F08A45');
  const palette = screenChoices.palettes[1], matched = withColors(screenStyle, palette);
  Object.assign(matched.captions, { camera_3d: true, highlight_color: '#010203' });
  assert.equal(screenChoices.screenSummary('colors', matched), palette.name);
});

test('preview grading matches apply_rgb for Film and Eyedropper, including halo and skin colors', () => {
  // Reference values from grade.apply_rgb with GradeSpec and rounded 8 bit RGB.
  const colors = ['#909090', '#101010', '#FFFFFF', '#12ABCD', '#D7A071', '#F08A45'];
  for (const [grade, expected] of [
    [{ look: 'film', strength: 1 }, ['#969593', '#001219', '#FFF3E3', '#54B1CA', '#E2AC7D', '#FA9654']],
    [{ look: 'film', strength: .5 }, ['#939292', '#0A1114', '#FFF9F1', '#3FAECC', '#DDA677', '#F5904D']],
    [{ look: 'eyedropper', strength: 1, color: '#1E6BD6' }, ['#7E91B0', '#101010', '#F7FFFF', '#00A9EB', '#D6A074', '#EF8B4A']],
    [{ look: 'eyedropper', strength: .5, color: '#F08A45' }, ['#9D8C82', '#101010', '#FFFDF9', '#4EA9C0', '#D8A070', '#F18943']],
  ]) assert.deepEqual(colors.map((color) => screenChoices.gradeColor(color, grade)), expected);
  for (const grade of [undefined, { look: 'none' }, { look: 'film', strength: 0 }]) {
    assert.equal(screenChoices.gradeColor('#12ABCD', grade), '#12ABCD');
  }
});

test('word, motion, font, color and backing choices preserve 3D camera', () => {
  const source = screenChoices.withCaptionLook(screenStyle, 'camera_3d');
  for (const changed of [withWords(source, 'one'), screenChoices.withMotion(source, 'Cut in'),
    withFont(source, 'Gelasio'), withColors(source, colorsOf(source)), withBackground(source, 'box')]) {
    assert.equal(changed.captions.camera_3d, true);
  }
});

test('a preset or saved look replaces 3D camera with its own value', () => {
  const camera = screenChoices.withCaptionLook(screenStyle, 'camera_3d');
  for (const source of [screenStyle, camera]) for (const look of [screenStyle, camera]) {
    for (const changed of [screenChoices.withPreset(source, look), choosePreset(source, look, 'New').style]) {
      assert.deepEqual(changed, look);
      assert.notEqual(changed, look);
    }
  }
});

test('caption backgrounds revive zero opacity and preserve every nonzero strength', () => {
  for (const opacity of [0, .12, .75, 1]) for (const look of ['plain', 'box', 'band']) {
    const candidate = structuredClone(screenStyle); candidate.captions.background_opacity = opacity;
    const fields = { 'captions.background_mode': look === 'plain' ? 'none' : look };
    if (look !== 'plain' && opacity === 0) fields['captions.background_opacity'] = screenChoices.backgroundOpacities.medium;
    onlyFields(candidate, (style) => screenChoices.withCaptionLook(style, look), fields);
  }
});

test('B-roll defaults are Mixed and every layout and placement preserves other fields', () => {
  assert.equal(screenChoices.brollLayout(screenStyle), 'mixed');
  const kinds = ['screen_recording', 'clip', 'photo', 'screenshot'];
  for (const [layout, placements] of Object.entries({ full: ['full_frame', 'full_frame', 'full_frame', 'full_frame'],
    mixed: ['full_frame', 'full_frame', 'overlay', 'overlay'], box: ['overlay', 'overlay', 'overlay', 'overlay'],
    split: ['split', 'split', 'split', 'split'] })) {
    const changed = onlyFields(screenStyle, (style) => screenChoices.withBrollLayout(style, layout, screenSchema),
      Object.fromEntries(kinds.map((kind, index) => [`broll.${kind}_placement`, placements[index]])));
    assert.equal(screenChoices.brollLayout(changed, screenSchema), layout);
  }
  for (const kind of kinds) for (const value of screenSchema.properties.broll.properties[`${kind}_placement`].enum) {
    onlyFields(screenStyle, (style) => screenChoices.withBrollPlacement(style, kind, value, screenSchema),
      { [`broll.${kind}_placement`]: value });
  }
  assert.equal(screenChoices.brollLayout(screenChoices.withBrollPlacement(screenStyle, 'clip', 'overlay')), 'custom');
  assert.throws(() => screenChoices.withBrollLayout(screenStyle, 'custom'));
  assert.throws(() => screenChoices.withBrollLayout(screenStyle, '__proto__'));
  assert.throws(() => screenChoices.withBrollPlacement(screenStyle, 'constructor', 'overlay'));
  assert.throws(() => screenChoices.withBrollPlacement(screenStyle, 'photo', 'other'));
});

test('Split is a schema-gated layout slot and mixed Split placements stay custom', () => {
  const entry = screenChoices.brollLayoutSlots.find((entry) => entry.id === 'split');
  assert.equal(entry.name, 'Split');
  assert.equal(entry.note, 'The shot on top, you below');
  assert.deepEqual(entry.requires, ['broll.clip_placement', 'split']);
  assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
  const changed = entry.apply(screenStyle, true, screenSchema);
  assert.equal(entry.selected(changed, screenSchema), true);
  assert.equal(entry.selected(screenStyle, screenSchema), false);
  for (const kind of ['screen_recording', 'clip', 'photo', 'screenshot']) {
    const mixed = screenChoices.withBrollPlacement(changed, kind, 'overlay', screenSchema);
    assert.equal(screenChoices.brollLayout(mixed, screenSchema), 'custom');
    const older = structuredClone(screenSchema);
    older.properties.broll.properties[`${kind}_placement`].enum = ['full_frame', 'overlay'];
    assert.throws(() => screenChoices.withBrollPlacement(screenStyle, kind, 'split', older));
    assert.throws(() => screenChoices.withBrollLayout(screenStyle, 'split', older));
    assert.equal(screenChoices.brollLayout(changed, older), 'custom');
    if (kind === 'clip') assert.equal(screenChoices.optionAvailable(entry, older), false);
  }
  assert.equal(screenChoices.screenSummary('broll', changed, {}, screenSchema), 'Split');
  assert.equal(screenChoices.changeSummary(changed, screenStyle), 'Changed B-roll from Clean Minimal');
});

test('Match color switches only its field and defaults on without inserting a value', () => {
  const entry = screenChoices.brollMoreSlots.find((entry) => entry.id === 'color-match');
  assert.equal(entry.name, 'Match color to me');
  assert.equal(entry.note, "Shift each shot toward your take's color");
  assert.equal(entry.hint, 'Your face is never touched. On by default.');
  assert.equal(entry.type, 'switch');
  assert.equal(entry.previewCurrent, true);
  assert.deepEqual(entry.requires, ['broll.color_match', 'on']);
  assert.equal(screenChoices.optionAvailable(entry, screenSchema), true);
  const missing = structuredClone(screenStyle); delete missing.broll.color_match;
  const before = structuredClone(missing);
  assert.equal(entry.selected(missing), true);
  assert.deepEqual(missing, before);
  for (const source of [screenStyle, missing]) for (const on of [true, false]) {
    const changed = onlyFields(source, (style) => entry.apply(style, on), { 'broll.color_match': on ? 'on' : 'off' });
    assert.equal(entry.selected(changed), on);
    assert.equal(screenChoices.screenSummary('broll', changed), on ? 'Mixed' : 'Mixed · color match off');
  }
  const off = entry.apply(screenStyle, false);
  assert.deepEqual(entry.apply(off, true), screenStyle);
  assert.equal(screenChoices.changeSummary(off, screenStyle), 'Changed B-roll from Clean Minimal');
  const older = structuredClone(screenSchema); delete older.properties.broll.properties.color_match;
  assert.equal(screenChoices.optionAvailable(entry, older), false);
  const combined = screenChoices.withEmphasis(screenChoices.withBrollField(off, 'stock_clips', 'ask'), 'video_letters', true);
  assert.equal(screenChoices.screenSummary('broll', combined), 'Mixed · stock clips: ask first · video in letters on · color match off');
});

test('B-roll fine choices accept only current fields and values', () => {
  for (const [field, values] of Object.entries({ overlay_box: ['headroom', 'side_left', 'side_right', 'tall_behind'],
    overlay_motion: ['none', 'drift', 'scroll'], tall_framing: ['auto', 'keep', 'shrink'], still_push: ['in', 'out', 'none'], stock_clips: ['off', 'ask'], color_match: ['off', 'on'] })) {
    for (const value of values) onlyFields(screenStyle,
      (style) => screenChoices.withBrollField(style, field, value), { [`broll.${field}`]: value });
    assert.throws(() => screenChoices.withBrollField(screenStyle, field, 'invalid'));
  }
  for (const field of ['constructor', '__proto__', 'avoid_tags']) {
    assert.throws(() => screenChoices.withBrollField(screenStyle, field, 'off'));
  }
});

test('music levels and exact values change only volume', () => {
  assert.deepEqual(screenChoices.musicLevels.map(({ name, value }) => [name, value]),
    [['No music', 0], ['Quiet', .06], ['Medium', .12], ['Loud', .2]]);
  for (const level of [0, .06, .12, .2, .137, .4]) onlyFields(screenStyle,
    (style) => screenChoices.withMusicLevel(style, level), { 'music.level': level });
  for (const invalid of [-1, .41, NaN, Infinity, '.06']) {
    assert.throws(() => screenChoices.withMusicLevel(screenStyle, invalid));
  }
});

test('effects off then on restores the supplied kit and otherwise uses soft', () => {
  const original = structuredClone(screenStyle);
  original.music.sound_effects = 'custom-kit';
  assert.equal(screenChoices.soundEffectsOn(original), true);
  const off = onlyFields(original, (style) => screenChoices.withSoundEffects(style, false),
    { 'music.sound_effects': 'none' });
  assert.equal(screenChoices.soundEffectsOn(off), false);
  onlyFields(off, (style) => screenChoices.withSoundEffects(style, true, original.music.sound_effects),
    { 'music.sound_effects': 'custom-kit' });
  for (const kit of [undefined, '', 'none']) onlyFields(off,
    (style) => screenChoices.withSoundEffects(style, true, kit), { 'music.sound_effects': 'soft' });
  onlyFields(original, (style) => screenChoices.withSoundEffects(style, true),
    { 'music.sound_effects': 'custom-kit' });
  assert.throws(() => screenChoices.withSoundEffects(original, 'yes'));
  assert.throws(() => screenChoices.withSoundEffects(original, true, 'not a kit'));
});

test('voice choices and beat switch touch only their fields', () => {
  for (const preset of ['none', 'clean', 'broadcast', 'warm']) onlyFields(screenStyle,
    (style) => screenChoices.withVoicePreset(style, preset), { 'voice.preset': preset });
  for (const field of ['repair_rumble', 'repair_room', 'repair_harshness']) for (const on of [false, true]) {
    onlyFields(screenStyle, (style) => screenChoices.withVoiceRepair(style, field, on), { [`voice.${field}`]: on });
  }
  for (const value of ['off', 'gentle', 'strong']) onlyFields(screenStyle,
    (style) => screenChoices.withVoiceRepair(style, 'repair_proximity', value), { 'voice.repair_proximity': value });
  for (const on of [true, false]) onlyFields(screenStyle,
    (style) => screenChoices.withBeatSnap(style, on), { 'music.beat_snap': on });
  assert.throws(() => screenChoices.withVoicePreset(screenStyle, 'constructor'));
  assert.throws(() => screenChoices.withVoiceRepair(screenStyle, 'preset', 'clean'));
  assert.throws(() => screenChoices.withVoiceRepair(screenStyle, 'repair_rumble', 'yes'));
  assert.throws(() => screenChoices.withVoiceRepair(screenStyle, 'repair_proximity', true));
  assert.throws(() => screenChoices.withBeatSnap(screenStyle, 1));
});

test('motion choices reset Rise while retaining every other caption value', () => {
  for (const entrance of [undefined, 'slide', 'rise']) for (const motion of screenChoices.motions) {
    const candidate = structuredClone(screenStyle);
    if (entrance === undefined) delete candidate.captions.entrance;
    else candidate.captions.entrance = entrance;
    for (const choice of [motion, motion.name]) onlyFields(candidate,
      (style) => screenChoices.withMotion(style, choice), {
        'captions.entry_rise_px': motion.rise, 'captions.entry_ms': motion.ms,
        ...(entrance === 'rise' ? { 'captions.entrance': 'slide' } : {}),
      });
  }
  assert.throws(() => screenChoices.withMotion(screenStyle, 'other'));
  assert.throws(() => screenChoices.withMotion(screenStyle, { name: 'Cut in', rise: 80, ms: 1000 }));
});

test('screen summaries describe grades, titles, enabled groups, B-roll and sound', () => {
  const { screenSummary, withBrollLayout, withBrollField, withMusicLevel,
    withSoundEffects, withVoicePreset, withKind } = screenChoices;
  const before = structuredClone(screenStyle);
  assert.equal(screenSummary('font', screenStyle), 'Inter');
  assert.equal(screenSummary('captions', screenStyle), 'Plain');
  assert.equal(screenSummary('colors', screenStyle), 'Sage');
  assert.equal(screenSummary('colors', withColors(screenStyle,
    { ...colorsOf(screenStyle), caption: '#ABCDEF' })), 'Your colors');
  assert.equal(screenSummary('grade', screenStyle), 'None');
  assert.equal(screenSummary('grade', withGrade(screenStyle, 'film', .5)), 'Film look, light');
  assert.equal(screenSummary('grade', withGrade(screenStyle, 'film')), 'Film look, full');
  for (const strength of [.5, 1]) assert.equal(screenSummary('grade', withGrade(screenStyle, 'eyedropper', strength, '#123456')),
    `Eyedropper #123456, ${strength === .5 ? 'light' : 'full'}`);
  assert.equal(screenSummary('opening', screenStyle, { quiet_serif: 'Quiet title' }), 'Quiet title');
  assert.equal(screenSummary('opening', { ...screenStyle, hook: { style: 'none' } }), 'No title');
  assert.equal(screenSummary('opening', { ...screenStyle, hook: { style: 'video_letters' } }, { video_letters: 'Video in letters' }), 'Video in letters');
  const labels = { minimal: 'Minimal', card_kinds: {
    text_tag: { category: 'Words' }, quote: { category: 'Words' },
    stat: { category: 'Numbers' }, prompt_bar: { category: 'Chat', opt_in: true },
  } };
  assert.equal(screenSummary('cards', screenStyle, labels), 'Minimal cards · 2 of 3 groups on');
  assert.equal(screenSummary('cards', withKind(screenStyle, 'prompt_bar', true, true), labels), 'Minimal cards · 3 of 3 groups on');
  let kindsOff = withKind(withKind(screenStyle, 'text_tag', false), 'quote', false);
  assert.equal(screenSummary('cards', kindsOff, labels), 'Minimal cards · 1 of 3 groups on');
  assert.equal(screenSummary('broll', screenStyle), 'Mixed');
  for (const [layout, name] of [['full', 'Full screen'], ['box', 'In a box']]) {
    assert.equal(screenSummary('broll', withBrollLayout(screenStyle, layout)), name);
  }
  assert.equal(screenSummary('broll', screenChoices.withBrollPlacement(screenStyle, 'clip', 'overlay')), 'Custom');
  assert.equal(screenSummary('broll', withBrollField(screenStyle, 'stock_clips', 'ask')), 'Mixed · stock clips: ask first');
  assert.equal(screenSummary('broll', screenChoices.withEmphasis(screenStyle, 'video_letters', true)), 'Mixed · video in letters on');
  assert.equal(screenSummary('broll', screenChoices.withEmphasis(withBrollField(screenStyle, 'stock_clips', 'ask'), 'video_letters', true)), 'Mixed · stock clips: ask first · video in letters on');
  assert.equal(screenSummary('sound', screenStyle), 'Music 6% · effects ElevenLabs · voice clean-up off');
  assert.equal(screenSummary('sound', withMusicLevel(screenStyle, 0)), 'Music off · effects ElevenLabs · voice clean-up off');
  assert.equal(screenSummary('sound', withVoicePreset(withSoundEffects(screenStyle, false), 'warm')),
    'Music 6% · effects off · voice clean-up warm');
  assert.equal(screenSummary('sound', withMusicLevel(screenStyle, .137)), 'Music 13.7% · effects ElevenLabs · voice clean-up off');
  assert.deepEqual(screenStyle, before);
});

test('change summary names screen groups in order without changing the base', () => {
  const base = structuredClone(screenStyle), changed = structuredClone(base);
  assert.equal(screenChoices.changeSummary(base, base), 'Starting from Clean Minimal');
  assert.equal(screenChoices.changeSummary(screenChoices.withEmphasis(base, 'video_letters', true), base), 'Changed B-roll from Clean Minimal');
  changed.captions.entrance = 'rise';
  changed.broll.overlay_box = 'side_right';
  assert.equal(screenChoices.changeSummary(changed, base), 'Changed Captions, B-roll from Clean Minimal');
  changed.captions.width_scale = .7;
  changed.graphics.paper_color = '#123456';
  changed.grade = { look: 'film', strength: .5 };
  changed.hook.style = 'none';
  changed.graphics.card_kinds_on = ['prompt_bar'];
  changed.music.beat_snap = true;
  assert.equal(screenChoices.changeSummary(changed, base),
    'Changed Font, Captions, Colors, Grade, Opening, Cards, B-roll, Sound from Clean Minimal');
  for (const apply of [(style) => screenChoices.withVoicePreset(style, 'warm'),
    (style) => screenChoices.withVoiceRepair(style, 'repair_rumble', true),
    (style) => screenChoices.withSoundEffects(style, false)]) {
    assert.equal(screenChoices.changeSummary(apply(base), base), 'Changed Sound from Clean Minimal');
  }
  assert.deepEqual(base, screenStyle);
});
test('font roles use nonempty brand overrides and detect Clean Minimal split fonts', () => {
  assert.equal(captionsFont(fontStyle), 'Inter');
  assert.deepEqual(graphicsFonts(fontStyle), { title: 'Gelasio', cards: 'Inter' });
  assert.equal(fontMode(fontStyle), 'different');
  const unbranded = structuredClone(fontStyle);
  delete unbranded.brand;
  assert.equal(captionsFont(unbranded), 'Inter');
  assert.deepEqual(graphicsFonts(unbranded), { title: 'Gelasio', cards: 'Inter' });
  assert.equal(fontMode(unbranded), 'different');
  const branded = { ...fontStyle, brand: {
    captions_font: 'Creator Serif', titles_font: 'Creator Sans', cards_font: 'Creator Sans' } };
  assert.equal(captionsFont(branded), 'Creator Serif');
  assert.deepEqual(graphicsFonts(branded), { title: 'Creator Sans', cards: 'Creator Sans' });
  assert.equal(fontMode(branded), 'different');
  branded.brand.captions_font = 'Creator Sans';
  assert.equal(fontMode(branded), 'same');
  assert.equal(fontMode(withFont(unbranded, 'Inter')), 'same');
});
test('captions and graphics font edits touch only their fields and preserve empty brand fields', () => {
  for (const brand of [undefined, fontStyle.brand, {
    captions_font: 'Creator Serif', titles_font: 'Creator Sans', cards_font: '', accent_color: '#123456' }, {
    captions_font: '', titles_font: '', cards_font: 'Creator Sans' }]) {
    const original = structuredClone({ ...fontStyle, ...(brand ? { brand } : {}) });
    if (!brand) delete original.brand;
    const before = structuredClone(original);
    const captionExpected = structuredClone(original);
    captionExpected.captions.font = 'Instrument Serif';
    captionExpected.captions.letter_spacing_px = -.8;
    if (brand?.captions_font) captionExpected.brand.captions_font = 'Instrument Serif';
    assert.deepEqual(withCaptionsFont(original, 'Instrument Serif'), captionExpected);
    const graphicsExpected = structuredClone(original);
    graphicsExpected.hook.font = graphicsExpected.graphics.font = 'Geist';
    if (brand?.titles_font) graphicsExpected.brand.titles_font = 'Geist';
    if (brand?.cards_font) graphicsExpected.brand.cards_font = 'Geist';
    assert.deepEqual(withGraphicsFont(original, 'Geist'), graphicsExpected);
    assert.deepEqual(original, before);
  }
  assert.equal(withCaptionsFont(fontStyle, 'Geist').captions.letter_spacing_px, -.5);
  assert.equal(withCaptionsFont(fontStyle, 'Creator Serif').captions.letter_spacing_px, 0);
});
test('font pairing applies Instrument Serif captions and Geist graphics or a custom pair', () => {
  assert.deepEqual(fontPair, { captions: 'Instrument Serif', graphics: 'Geist' });
  assert.deepEqual(withFontPair(fontStyle), withGraphicsFont(withCaptionsFont(fontStyle, 'Instrument Serif'), 'Geist'));
  assert.deepEqual(withFontPair(fontStyle, { captions: 'Gelasio', graphics: 'Inter' }),
    withGraphicsFont(withCaptionsFont(fontStyle, 'Gelasio'), 'Inter'));
});
test('same font keeps the existing complete output for Gelasio', () => {
  assert.deepEqual(withFont(fontStyle, 'Gelasio'), {
    name: 'Synthetic look', captions: { font: 'Gelasio', letter_spacing_px: -.3, width_scale: .85 },
    hook: { font: 'Gelasio', style: 'quiet_serif' }, graphics: { font: 'Gelasio', style: 'paper' },
    brand: { captions_font: '', titles_font: '', cards_font: '', accent_color: '#123456' },
  });
  const branded = { ...fontStyle, brand: { captions_font: 'Inter', titles_font: '', cards_font: 'Inter' } };
  assert.deepEqual(withFont(branded, 'Gelasio').brand,
    { captions_font: 'Gelasio', titles_font: '', cards_font: 'Gelasio' });
});
test('grade slider follows the candidate style and survives look changes', () => {
  const style = { grade: { look: 'film', strength: 1, slider: true },
    captions: { text_color: '#F8E4A0' }, graphics: { ink_color: '#123456' } };
  assert.equal(withGrade(style, 'film', 0.5).grade.slider, true);
  const url = new URL(stillUrl({ still: '/still?g=3' }, style), 'http://localhost');
  assert.equal(url.searchParams.get('slider'), '1');
  assert.equal(url.searchParams.get('text_color'), '#F8E4A0');
  assert.equal(url.searchParams.get('ink_color'), '#123456');
  assert.equal(new URL(stillUrl({ still: '/still?g=3' }, withGrade(style, 'none'), true),
    'http://localhost').searchParams.has('slider'), false);
});

test('cutout URLs use the picture grade without compare parameters or input changes', () => {
  const preview = { still: '/still?g=7', cutout: '/cutout?g=7' };
  for (const grade of [undefined, { look: 'none', strength: 1 }, { look: 'film', strength: .5, slider: true },
    { look: 'film', strength: 0 }, { look: 'eyedropper', strength: .75, color: '#1E6BD6', slider: true }]) {
    const candidate = { ...style, ...(grade ? { grade } : {}) }, before = structuredClone(candidate);
    const url = screenChoices.cutoutUrl(preview, candidate);
    const expected = grade && grade.look !== 'none' ? '/cutout?g=7&look=' + grade.look + '&strength=' + grade.strength
      + (grade.look === 'eyedropper' ? '&color=%231E6BD6' : '') : '/cutout?g=7';
    assert.equal(url, expected);
    assert.deepEqual(candidate, before);
  }
  assert.equal(screenChoices.cutoutUrl({ ...preview, cutout: null }, style), null);
  assert.equal(screenChoices.cutoutUrl({ still: '/still?g=7' }, style), null);
  assert.deepEqual(preview, { still: '/still?g=7', cutout: '/cutout?g=7' });
});

test('grade choices default to full, retain strength and build candidate still URLs', () => {
  const preview = { still: '/still?g=7' };
  assert.equal(stillUrl(preview, style), '/still?g=7');
  assert.equal(stillUrl(preview, style, true), '/still?g=7&look=none&strength=1');
  assert.equal(stillUrl(preview, withGrade(style, 'none'), true), '/still?g=7&look=none&strength=1');
  const film = withGrade(style, 'film');
  assert.deepEqual(film.grade, { look: 'film', strength: 1 });
  assert.equal(style.grade, undefined);
  assert.equal(stillUrl(preview, film), '/still?g=7&look=film&strength=1');
  const light = withGrade(film, 'film', 0.5);
  assert.equal(stillUrl(preview, light), '/still?g=7&look=film&strength=0.5');
  assert.equal(withGrade(withGrade(light, 'none'), 'film').grade.strength, 0.5);
  assert.deepEqual(withGrade({ grade: { look: 'none', strength: 1, color: null } }, 'none').grade,
    { look: 'none', strength: 1, color: null });
  assert.throws(() => withGrade(style, 'other'));
  assert.throws(() => withGrade(style, 'film', NaN));
});
test('eyedropper choices keep colour and build same-origin still URLs', () => {
  assert.deepEqual(gradeLooks.at(-1), ['eyedropper', 'Eyedropper']);
  const picked = withGrade(style, 'eyedropper', 0.5, '#1E6BD6');
  assert.deepEqual(picked.grade, { look: 'eyedropper', strength: 0.5, color: '#1E6BD6' });
  assert.equal(stillUrl({ still: '/still?g=7' }, picked),
    '/still?g=7&look=eyedropper&strength=0.5&color=%231E6BD6');
  assert.equal(withGrade(picked, 'film').grade.color, '#1E6BD6');
  assert.throws(() => withGrade(style, 'eyedropper', 0.5));
});
test('tap sampling accounts for cover crop and clamps a seven-pixel patch', () => {
  assert.deepEqual(coverPoint({ left: 10, top: 20, width: 100, height: 200 },
    400, 200, 60, 120), { x: 0.5, y: 0.5 });
  assert.deepEqual(coverPoint({ left: 10, top: 20, width: 100, height: 200 },
    400, 200, 10, 20), { x: 0.375, y: 0 });
  assert.deepEqual(samplePatch(20, 30, 0.5, 0.5), { x: 7, y: 12, width: 7, height: 7 });
  assert.deepEqual(samplePatch(20, 30, 1, 1), { x: 16, y: 26, width: 4, height: 4 });
  assert.equal(patchColor(new Uint8ClampedArray([20, 40, 60, 255, 40, 60, 80, 255])), '#1E3246');
});
test('font names resolve to the same local families as the renderer', () => {
  const available = { bundled: ['Instrument Serif', 'Inter'], creator: ['Creator Serif'] };
  assert.equal(fontFamily('Instrument Serif', available), '"Yap Studio Instrument Serif", sans-serif');
  assert.equal(fontFamily('Creator Serif', available), '"Yap Studio User Creator Serif", sans-serif');
  assert.equal(fontFamily('Arial', available), '"Arial", sans-serif');
  assert.equal(fontFamily('A "quoted" font', available), '"A \\"quoted\\" font", sans-serif');
});
test('footer reservation rounds up and tracks changing bar heights', () => {
  const values = {};
  const root = { style: { setProperty(key, value) { values[key] = value; } } };
  for (const [height, expected] of [[96, 96], [153.25, 154], [64.2, 65], [0, 0]]) {
    assert.equal(reserveSavebar({ getBoundingClientRect: () => ({ height }) }, root), expected);
    assert.equal(values['--savebar-height'], `${expected}px`);
  }
});
test('words on screen changes phrase limits while keeping the chosen size and placement', () => {
  const one = withWords(style, 'one');
  assert.equal(one.captions.max_words_per_phrase, 1);
  assert.equal(one.captions.top_px, 1500);
  assert.equal(one.captions.position, 'bottom');
  assert.equal(one.captions.size_px, 92);
  assert.equal(withWords(style, 'few').captions.max_words_per_phrase, 3);
  assert.equal(withWords(style, 'sentence').captions.max_words_per_phrase, 8);
});
test('brand edits only the brand block and validates hex', () => {
  assert.equal(withBrand(style, 'captions_font', 'Instrument Serif').brand.captions_font, 'Instrument Serif');
  assert.equal(validHex('#A1b2C3'), true);
  assert.equal(validHex('#abc'), false);
  assert.throws(() => withBrand(style, 'accent_color', '#abc'));
});
test('card switches write card_kinds_off', () => {
  const disabled = withKind(style, 'stat', false);
  assert.deepEqual(disabled.graphics.card_kinds_off, ['stat']);
  assert.equal(kindOn(disabled, 'stat'), false);
  assert.deepEqual(withKind(disabled, 'stat', true).graphics.card_kinds_off, []);
});
test('Choices starts off and saves through card_kinds_on', () => {
  assert.equal(screenChoices.labels.choices, 'Choices');
  assert.equal(kindOn(screenStyle, 'choices', true), false);
  const enabled = withKind(screenStyle, 'choices', true, true);
  assert.deepEqual(enabled.graphics.card_kinds_on, ['choices']);
  assert.equal(kindOn(enabled, 'choices', true), true);
  assert.deepEqual(withKind(enabled, 'choices', false, true).graphics.card_kinds_on, []);
});

test('opt-in card switches use card_kinds_on and preserve base switches', () => {
  const original = { graphics: { card_kinds_off: ['stat'] } };
  assert.equal(kindOn(original, 'new_kind', true), false);
  assert.equal(kindOn(style, 'stat'), true);
  assert.equal(kindOn({ graphics: {} }, 'stat'), true);
  assert.equal(kindOn({ graphics: {} }, 'new_kind', true), false);
  const enabled = withKind(original, 'new_kind', true, true);
  assert.deepEqual(enabled.graphics.card_kinds_on, ['new_kind']);
  assert.deepEqual(enabled.graphics.card_kinds_off, ['stat']);
  assert.equal(kindOn(enabled, 'new_kind', true), true);
  assert.deepEqual(withKind(enabled, 'new_kind', true, true), enabled);
  const second = withKind(enabled, 'another_kind', true, true);
  assert.deepEqual(second.graphics.card_kinds_on, ['another_kind', 'new_kind']);
  const disabled = withKind(enabled, 'new_kind', false, true);
  assert.deepEqual(disabled.graphics.card_kinds_on, []);
  assert.deepEqual(disabled.graphics.card_kinds_off, ['stat']);
  assert.equal(kindOn(disabled, 'new_kind', true), false);
  assert.deepEqual(original, { graphics: { card_kinds_off: ['stat'] } });
});
test('tile window follows the layer and stays inside the frame', () => {
  assert.deepEqual(tileWindow({ x: 0, y: 1384, width: 1080, height: 251 }),
    { x: 90, y: 320, width: 900, height: 1600 });
  assert.deepEqual(tileWindow({ x: 200, y: 100, width: 680, height: 300 }),
    { x: 90, y: 0, width: 900, height: 1600 });
});
test('tile windows follow phone, compact and resized host proportions', () => {
  const caption = { x: 0, y: 1384, width: 1080, height: 251 };
  assert.deepEqual(tileWindow(caption, undefined, { width: 150, height: 165 }),
    { x: 90, y: 930, width: 900, height: 990 });
  assert.deepEqual(tileWindow({ x: 0, y: 100, width: 1080, height: 200 }, undefined, { width: 160, height: 70 }),
    { x: 90, y: 3.125, width: 900, height: 393.75 });
  assert.deepEqual(tileWindow(caption, undefined, { width: 100, height: 500 }),
    { x: 90, y: 0, width: 900, height: 1920 });
});
test('phone caption crops keep the band near seventy-two percent and clamp at the frame edges', () => {
  const caption = { x: 0, y: 1384, width: 1080, height: 251 }, host = { width: 150, height: 165 };
  const window = tileWindow(caption, undefined, host, .72);
  assert.equal(window.width, 900);
  assert.equal(window.height, 990);
  assert.ok(Math.abs((caption.y + caption.height / 2 - window.y) / window.height - .72) < .00001);
  assert.equal(tileWindow({ ...caption, y: 0 }, undefined, host, .72).y, 0);
  assert.equal(tileWindow({ ...caption, y: 1850 }, undefined, host, .72).y, 930);
});
test('style name starts with a personal label unless the source is the library', () => {
  const preset = { name: 'Clean Minimal', brand: {} };
  assert.equal(startingName(preset, false), 'My style');
  assert.equal(startingName({ name: 'Sunday Notes' }, true), 'Sunday Notes');
  const picked = choosePreset(preset, { name: 'Electric', brand: {} }, 'My travel look');
  assert.equal(picked.style.name, 'Electric');
  assert.equal(picked.name, 'My travel look');
});
test('removed or windowless frames cannot be used by pending tile work', async () => {
  const frame = { isConnected: true, contentWindow: { __timelines: {} } };
  assert.equal(frameWindow(frame), frame.contentWindow);
  assert.equal(frameWindow(frame, false), null);
  let finishSeek;
  const seek = new Promise((resolve) => { finishSeek = resolve; });
  const pending = (async () => {
    const window = frameWindow(frame);
    await seek;
    return frameWindow(frame) === window ? window.__timelines : null;
  })();
  frame.isConnected = false;
  finishSeek();
  assert.equal(await pending, null);
  assert.equal(frameWindow(frame), null);
  frame.isConnected = true;
  frame.contentWindow = null;
  assert.equal(frameWindow(frame), null);
});

test('a preset replaces the complete style, including brand overrides', () => {
  const source = { brand: { accent_color: '#FF0000', captions_font: 'Inter' } };
  const preset = { name: 'Warm Editorial', brand: {}, captions: { font: 'Gelasio' } };
  const picked = choosePreset(source, preset, 'My style');
  assert.deepEqual(picked.style, preset);
  assert.notEqual(picked.style, preset);
});

const branded = {
  brand: { accent_color: '#FF0000', captions_font: 'Inter' },
  captions: { text_color: '#FFFFFF', highlight_color: '#FFD23F', highlight_text_color: '#1D1D1D',
    highlight_style: 'underline' },
  hook: { background_color: '#FFF6DC', text_color: '#202020', accent_color: '#FF2E88' },
  graphics: { style: 'minimal', paper_color: '#FFFDF4', ink_color: '#1D1D1D',
    accent_color: '#FF2E88', titlebar_color: '#F1EAD6' },
};

test('every card look preserves brand, captions, title colors and graphics accent', () => {
  const original = structuredClone(branded);
  for (const look of ['paper', 'minimal', 'dark', 'bright']) {
    const changed = withCardStyle(branded, look);
    assert.equal(changed.graphics.style, look);
    assert.deepEqual(changed.brand, branded.brand);
    assert.deepEqual(changed.captions, branded.captions);
    assert.deepEqual(changed.hook, branded.hook);
    assert.equal(changed.graphics.accent_color, branded.graphics.accent_color);
    if (look !== 'dark') assert.equal(changed.graphics.titlebar_color, branded.graphics.titlebar_color);
  }
  assert.deepEqual(branded, original);
  assert.throws(() => withCardStyle(branded, 'unknown'));
});

test('Dark swaps actual graphics paper and ink, then Paper and Minimal restore a readable light pair', () => {
  const dark = withCardStyle(branded, 'dark');
  assert.equal(dark.graphics.paper_color, branded.graphics.ink_color);
  assert.equal(dark.graphics.ink_color, branded.graphics.paper_color);
  assert.ok(luminance(dark.graphics.paper_color) < .3);
  assert.ok(contrast(dark.graphics.paper_color, dark.graphics.ink_color) >= 4.5);
  assert.equal(dark.graphics.titlebar_color, mix(dark.graphics.paper_color, dark.graphics.ink_color, .08));
  for (const look of ['paper', 'minimal']) {
    const light = withCardStyle(dark, look);
    assert.equal(light.graphics.paper_color, branded.graphics.paper_color);
    assert.equal(light.graphics.ink_color, branded.graphics.ink_color);
    assert.equal(light.brand.accent_color, '#FF0000');
    assert.deepEqual(light.hook, branded.hook);
  }
});

test('an explicit highlight edit survives a subsequent dark card choice', () => {
  const edited = withColors(branded, { ...colorsOf(branded), accent: '#22BBFF' }, ['accent']);
  const dark = withCardStyle(edited, 'dark');
  assert.equal(dark.brand.accent_color, undefined);
  assert.equal(dark.captions.highlight_color, '#22BBFF');
  assert.equal(dark.hook.accent_color, '#22BBFF');
  assert.equal(dark.graphics.accent_color, '#22BBFF');
});


test('shadow options map opacity, blur and both offsets to the existing engine fields', () => {
  const original = structuredClone(style);
  const soft = withShadow(style, 'soft');
  assert.deepEqual([soft.captions.shadow_opacity, soft.captions.shadow_blur_px,
    soft.captions.shadow_x_px, soft.captions.shadow_y_px], [.35, 8, 0, 3]);
  const strong = withShadow(style, 'strong');
  assert.deepEqual([strong.captions.shadow_opacity, strong.captions.shadow_blur_px,
    strong.captions.shadow_x_px, strong.captions.shadow_y_px], [.8, 3, 0, 6]);
  const off = withShadow(strong, 'off');
  assert.equal(shadowMode(off), 'off');
  assert.equal(off.captions.shadow_blur_px, 3);
  assert.equal(shadowMode(soft), 'soft');
  assert.equal(shadowMode(strong), 'strong');
  assert.deepEqual(style, original);
  assert.throws(() => withShadow(style, 'unknown'));
});

test('Bottom, Middle and Top are valid engine placements with separate size choices', () => {
  for (const [mode, position, top] of [['bottom', 'bottom', 1500], ['middle', 'middle', 1180], ['top', 'middle', 700]]) {
    const changed = withPlacement(style, mode);
    assert.equal(changed.captions.position, position);
    assert.equal(changed.captions.top_px, top);
    assert.equal(placementMode(changed), mode);
    assert.equal(changed.captions.size_px, 92);
  }
  assert.deepEqual(sizes, { small: 68, medium: 92, large: 118 });
  assert.equal(nearestSize({ captions: { size_px: 76 } }), 'small');
  assert.equal(nearestSize({ captions: { size_px: 100 } }), 'medium');
  assert.throws(() => withPlacement(style, 'unknown'));
});

test('background and face options map independently and retain custom values', () => {
  const original = { captions: { ...style.captions, background_mode: 'box',
    background_color: '#ACF123', background_opacity: .63, background_padding_px: 27,
    background_radius_px: 23, placement: 'fixed' } };
  for (const mode of ['none', 'box', 'band']) {
    const changed = withBackground(original, mode);
    assert.deepEqual(changed.captions, { ...original.captions, background_mode: mode });
  }
  for (const [mode, value] of [['light', .5], ['medium', .75], ['solid', .9]]) {
    assert.deepEqual(withBackgroundOpacity(original, mode).captions,
      { ...original.captions, background_opacity: value });
  }
  for (const [mode, value] of [['square', 0], ['rounded', 12], ['pill', 40]]) {
    assert.deepEqual(withBackgroundCorners(original, mode).captions,
      { ...original.captions, background_radius_px: value });
  }
  for (const enabled of [true, false]) {
    const changed = withFaceAvoidance(original, enabled);
    assert.deepEqual(changed.captions, { ...original.captions, placement: enabled ? 'avoid_face' : 'fixed' });
    for (const position of ['bottom', 'middle', 'top']) {
      assert.equal(withPlacement(changed, position).captions.placement, changed.captions.placement);
    }
  }
  assert.equal(original.captions.background_opacity, .63);
  assert.equal(original.captions.background_radius_px, 23);
  assert.equal(original.captions.placement, 'fixed');
  assert.throws(() => withBackground(original, 'pill'));
  assert.throws(() => withBackgroundOpacity(original, 'constructor'));
  assert.throws(() => withBackgroundCorners(original, 'unknown'));
  assert.throws(() => withFaceAvoidance(original, 'yes'));
});

test('live background text matches engine contrast while keeping the original caption color', () => {
  for (const mode of ['box', 'band']) {
    for (const [color, expected] of [['#FFFFFF', '#111111'], ['#000000', '#FFFFFF'],
      ['#FFFF00', '#111111'], ['#123456', '#FFFFFF'], ['#787878', '#000000'], ['#7A7A7A', '#000000']]) {
      const original = { captions: { text_color: '#ABCDEF', background_mode: mode,
        background_color: color, background_opacity: .5 } };
      assert.equal(captionTextColor(original), expected);
      assert.equal(captionTextColor(withBackground(original, 'none')), '#ABCDEF');
      assert.equal(captionTextColor({ captions: { ...original.captions, background_opacity: 0 } }), '#ABCDEF');
      assert.equal(original.captions.text_color, '#ABCDEF');
    }
  }
});

test('faint grounds preserve chosen text and hide automatic contrast copy', () => {
  for (const mode of ['box', 'band']) {
    for (const opacity of [0, .01, .49]) {
      for (const color of ['#FFFFFF', '#111111', '#ABCDEF']) {
        const style = { captions: { text_color: color, background_mode: mode,
          background_color: '#FFFFFF', background_opacity: opacity } };
        assert.equal(captionTextColor(style), color);
        assert.equal(captionContrastNote(style), '');
        assert.equal(style.captions.text_color, color);
      }
    }
  }
});

test('3D camera preserves chosen ink and hides contrast copy for its unused ground', () => {
  for (const mode of ['box', 'band']) {
    for (const color of ['#FFFFFF', '#111111', '#ABCDEF']) {
      const style = { captions: { camera_3d: true, text_color: color, background_mode: mode,
        background_color: '#FFFFFF', background_opacity: 1 } };
      assert.equal(captionTextColor(style), color);
      assert.equal(captionContrastNote(style), '');
      assert.equal(style.captions.text_color, color);
      assert.equal(style.captions.background_mode, mode);
    }
  }
});

test('automatic contrast copy names the rendered value without changing the saved color', () => {
  for (const mode of ['box', 'band']) {
    for (const [background, name, color] of [['#FFFFFF', 'dark', '#111111'], ['#000000', 'light', '#FFFFFF']]) {
      const candidate = { captions: { background_mode: mode, background_color: background,
        background_opacity: .75, text_color: '#ABCDEF' } };
      assert.equal(captionContrastNote(candidate),
        `Text switches to ${name} on this background so it stays readable. Using ${color}.`);
      assert.equal(candidate.captions.text_color, '#ABCDEF');
      assert.equal(captionContrastNote(withBackground(candidate, 'none')), '');
      candidate.captions.background_opacity = 0;
      assert.equal(captionContrastNote(candidate), '');
    }
  }
});

test('static tile captions use the take sentence and the look phrase limits', () => {
  const candidate = { captions: { max_words_per_phrase: 3, max_characters_per_phrase: 18 } };
  assert.deepEqual(posterWords(candidate, 'Choosing your own caption style'), ['Choosing', 'your', 'own']);
  candidate.captions.max_words_per_phrase = 1;
  assert.deepEqual(posterWords(candidate, 'Choosing your own caption style'), ['Choosing']);
  assert.deepEqual(posterWords(candidate, '  <real>   words  '), ['<real>']);
  assert.deepEqual(posterWords(candidate, ''), []);
});

test('Specs list row choices write only specs_rows and describe the Cards screen', () => {
  const group = screenChoices.cardGroupSlots.find((entry) => entry.name === 'Specs list rows');
  assert.ok(group);
  assert.deepEqual(group.options.map((entry) => entry.name), ['Together', 'Rows on cue', 'Rows on cue, glowing']);
  for (const entry of group.options) {
    assert.equal(entry.card, 'specs_list');
    assert.deepEqual(entry.requires, ['graphics.specs_rows', entry.id]);
    const chosen = onlyFields(screenStyle, entry.apply, { 'graphics.specs_rows': entry.id });
    assert.equal(entry.selected(chosen), true);
    assert.equal(group.options.filter((option) => option.selected(chosen)).length, 1);
    const summary = screenChoices.screenSummary('cards', chosen);
    if (entry.id === 'together') assert.ok(!summary.includes('rows on cue'));
    else assert.ok(summary.endsWith(entry.id === 'on_cue' ? ' · rows on cue' : ' · rows on cue, glowing'));
    if (entry.id !== 'together') assert.equal(screenChoices.changeSummary(chosen, screenStyle), 'Changed Cards from Clean Minimal');
  }
});

test('Card motion is one choice and writes only card_motion', () => {
  const group = screenChoices.cardGroupSlots.find(entry => entry.id === 'card-motion');
  assert.ok(group);
  assert.equal(group.note, 'Cards that fly in leave a short trail; number cards and full-screen cards enter as usual.');
  assert.deepEqual(group.options.map(entry => entry.name), ['None', 'Motion streak']);
  assert.equal(group.options[0].selected(screenStyle), true);
  for (const entry of group.options) {
    const chosen = onlyFields(screenStyle, entry.apply, { 'graphics.card_motion': entry.id });
    assert.equal(group.options.filter(option => option.selected(chosen)).length, 1);
    assert.equal(entry.selected(chosen), true);
  }
});

test('Jump cuts follow the enum and write only their field', () => {
  const group = screenChoices.movesMoreSlots.find((entry) => entry.id === 'jump-cuts');
  assert.ok(group);
  const options = group.options;
  assert.deepEqual(options.map((entry) => entry.id), screenSchema.properties.moves.properties.jump_cuts.enum);
  assert.deepEqual(options.map((entry) => entry.id), ['hard', 'punch', 'smooth', 'step_in']);
  const suffix = { hard: '', punch: ' · punch on jump cuts', smooth: ' · smooth jump cuts', step_in: ' · step in' };
  for (const entry of options) {
    assert.deepEqual(entry.requires, ['moves.jump_cuts', entry.id]);
    assert.equal(entry.preview, `jump_${entry.id}`);
    const chosen = onlyFields(screenStyle, entry.apply, { 'moves.jump_cuts': entry.id });
    assert.equal(entry.selected(chosen), true);
    assert.equal(group.value(chosen), entry.name);
    assert.equal(screenChoices.screenSummary('moves', chosen), 'No zooms · all transitions' + suffix[entry.id]);
    if (entry.id !== 'hard') assert.equal(screenChoices.changeSummary(chosen, screenStyle), 'Changed Moves from Clean Minimal');
  }
  const older = structuredClone(screenSchema); delete older.properties.moves.properties.jump_cuts;
  assert.ok([group, ...options].every((entry) => !screenChoices.optionAvailable(entry, older)));
  const missing = structuredClone(screenStyle); delete missing.moves.jump_cuts;
  assert.equal(screenChoices.screenSummary('moves', missing), 'No zooms · all transitions');
  assert.throws(() => screenChoices.withJumpCuts(screenStyle, 'other'));
});

test('sound kit choices and fallback change only their own fields', () => {
  const original = structuredClone(screenStyle);
  original.music.sound_effects = 'custom-kit';
  assert.equal(screenChoices.soundKitChoice(original), 'elevenlabs');
  for (const choice of ['made', 'none', 'elevenlabs']) {
    const next = onlyFields(original, (style) => screenChoices.withSoundKit(style, choice, 'custom-kit'),
      { 'music.sound_effects': choice === 'elevenlabs' ? 'custom-kit' : choice });
    assert.equal(screenChoices.soundKitChoice(next), choice);
  }
  for (const kit of [undefined, 'made', 'none']) onlyFields(original,
    (style) => screenChoices.withSoundKit(style, 'elevenlabs', kit),
    { 'music.sound_effects': 'soft' });
  for (const on of [false, true]) onlyFields(original,
    (style) => screenChoices.withSoundFallback(style, on),
    { 'music.sound_effects_fallback': on ? 'made' : 'none' });
  assert.throws(() => screenChoices.withSoundKit(original, 'bad'));
  assert.throws(() => screenChoices.withSoundFallback(original, 'yes'));
  const made = screenChoices.withSoundKit(original, 'made');
  assert.match(screenChoices.screenSummary('sound', made), /effects made/);
  assert.match(screenChoices.screenSummary('sound', original), /effects ElevenLabs/);
});

test('bleep switch changes only its field and keeps the off summary', () => {
  const before = screenChoices.screenSummary('sound', screenStyle);
  for (const on of [false, true]) {
    const next = onlyFields(screenStyle, (style) => screenChoices.withBleepSwears(style, on),
      { 'music.bleep_swears': on });
    assert.equal(screenChoices.screenSummary('sound', next), before + (on ? ' · bleep on' : ''));
    assert.equal(screenChoices.changeSummary(next, screenStyle), on ? 'Changed Sound from Clean Minimal' : 'Starting from Clean Minimal');
  }
  assert.throws(() => screenChoices.withBleepSwears(screenStyle, 'true'));
});

test('preview masks use the engine list and matching rules', () => {
  const builtin = JSON.parse(readFileSync(new URL('../../src/yap_studio/sfx/swear-words.json', import.meta.url))).words;
  const on = screenChoices.withBleepSwears(screenStyle, true);
  for (const [text, expected] of [
    ['Fuck, shit!', 'F***, s***!'], ["fuckin’ bull shit", "f*****’ b*** s***"],
    ['god-damn-it', 'g**-d***-it'], ['pre-god-damn-post', 'pre-g**-d***-post'],
    ['shıt ＳＨＩＴ', 's*** Ｓ***'], ['shit-show', 's***-show'],
    ['shit hits the fan', 's*** hits the fan'], ['s*** show', 's*** show'],
    ['class assess Scunthorpe shitake', 'class assess Scunthorpe shitake'],
    ['Safe,\ttext.\n', 'Safe,\ttext.\n'],
  ]) {
    assert.equal(screenChoices.maskPreviewText(text, on, builtin), expected);
    assert.equal(screenChoices.maskPreviewText(expected, on, builtin), expected);
    assert.equal(screenChoices.maskPreviewText(text, screenStyle, builtin), text);
  }
  on.music.bleep_words_add = ['crap']; on.music.bleep_words_remove = ['shit'];
  assert.equal(screenChoices.maskPreviewText('crap shit', on, builtin), 'c*** shit');
});
