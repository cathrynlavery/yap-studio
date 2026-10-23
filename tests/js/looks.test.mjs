import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const {highlightLooks, widthChoices, nearestWidth, pillText, pillContrast, hasUpcomingWords, gradientTint, wordFill} =
  require('../../docs/design/style-page/looks.js');
const {buildPhrases} = require('../../docs/design/style-page/phrases.js');
const {readLearned, switchPreset, undoPreset} = require('../../docs/design/style-page/learned.js');
const pageFile = (name) => readFileSync(new URL(`../../docs/design/style-page/${name}`, import.meta.url), 'utf8');
const context = {window: {}};
vm.runInNewContext(pageFile('style-data.js'), context);
const schema = JSON.parse(JSON.stringify(context.window.YAP_STYLE_DATA.schema));
const order = [
  'moving_block',
  'underline',
  'color',
  'pill',
  'gradient',
  'editorial',
  'slam',
  'weight_shift',
  'clip_wipe',
  'blend_difference',
  'camera_follow',
  'editorial_emphasis',
  'emoji_pop',
  'glitch_rgb',
  'gradient_fill',
  'highlight',
  'matrix_decode',
  'neon_accent',
  'neon_glow',
  'parallax_layers',
  'particle_burst',
  'pill_karaoke',
  'texture',
  'giant',
  'none',
];
const copy = (value) => JSON.parse(JSON.stringify(value));
const defaults = (rule) => rule.type === 'object'
  ? Object.fromEntries(Object.entries(rule.properties).map(([key, child]) => [key, defaults(child)]))
  : rule.default;

test('every engine highlight appears in the fixed page order with copy', () => {
  const before = JSON.stringify(schema);
  const looks = highlightLooks(schema);
  assert.deepEqual(looks.map((look) => look.id), order);
  assert.deepEqual([...looks.map((look) => look.id)].sort(),
    [...schema.properties.captions.properties.highlight_style.enum].sort());
  assert.deepEqual(looks.map((look) => look.name), [
    'Sliding block',
    'Underline',
    'Colored word',
    'Pill',
    'Word fill',
    'Bold key word',
    'Kinetic slam',
    'Weight shift',
    'Clip wipe',
    'Difference blend',
    'Camera follow',
    'Large key word',
    'Emoji pop',
    'RGB glitch',
    'Gradient fill',
    'Word background',
    'Matrix decode',
    'Neon accent',
    'Neon glow',
    'Parallax layers',
    'Particle burst',
    'Rounded karaoke',
    'Text texture',
    'Giant captions',
    'No highlight',
  ]);
  assert.ok(looks.every((look) => look.note && !look.note.includes('\n')));
  assert.equal(looks.find((look) => look.id === 'pill').note, 'Words light up inside a soft pill');
  assert.equal(looks.find((look) => look.id === 'gradient').note, 'Each word fills with color as it is spoken');
  assert.equal(looks.find((look) => look.id === 'editorial').note, 'The key word turns bold and colored');
  assert.equal(looks.find((look) => look.id === 'slam').note, 'One big word at a time, punched in');
  assert.equal(looks.find((look) => look.id === 'weight_shift').note, 'The line being said turns bold');
  assert.equal(looks.find((look) => look.id === 'clip_wipe').note, 'Each word wipes in as it is said');
  assert.equal(JSON.stringify(schema), before);
});

test('schema membership controls the choices while page order stays fixed', () => {
  const changed = copy(schema);
  changed.properties.captions.properties.highlight_style.enum.reverse();
  assert.deepEqual(highlightLooks(changed).map((look) => look.id), order);
  changed.properties.captions.properties.highlight_style.enum = ['none', 'pill'];
  assert.deepEqual(highlightLooks(changed).map((look) => look.id), ['pill', 'none']);
  const looks = highlightLooks(schema);
  looks[0].name = 'Changed';
  assert.equal(highlightLooks(schema)[0].name, 'Sliding block');
});

test('a new engine look fails with its name rather than disappearing', () => {
  for (const id of ['new_look', '__proto__', 'constructor']) {
    const changed = copy(schema);
    changed.properties.captions.properties.highlight_style.enum.push(id);
    assert.throws(() => highlightLooks(changed), (error) => error instanceof Error && error.message.includes(id));
  }
});

test('letter width choices use the engine bounds without changing the schema', () => {
  assert.deepEqual(widthChoices(schema).map(({name, value}) => ({name, value})), [
    {name: 'Normal', value: 1}, {name: 'Condensed', value: 0.85}, {name: 'Narrow', value: 0.7},
  ]);
  for (const [minimum, maximum, expected] of [
    [0.8, 0.9, [0.9, 0.85, 0.8]], [0.9, 0.95, [0.95, 0.9, 0.9]], [0.6, 0.65, [0.65, 0.65, 0.65]],
  ]) {
    const changed = copy(schema);
    Object.assign(changed.properties.captions.properties.width_scale, {minimum, maximum});
    const before = JSON.stringify(changed);
    assert.deepEqual(widthChoices(changed).map((choice) => choice.value), expected);
    assert.equal(JSON.stringify(changed), before);
  }
});

test('a custom width selects the closest choice without rounding its value', () => {
  const choices = widthChoices(schema);
  for (const [value, id] of [[1, 'normal'], [0.98, 'normal'], [0.873, 'condensed'], [0.73, 'narrow'], [0.6, 'narrow']]) {
    assert.equal(nearestWidth(value, choices).id, id);
  }
  assert.equal(nearestWidth(0.75, [{id: 'first', value: 1}, {id: 'second', value: 0.5}]).id, 'first');
  assert.throws(() => nearestWidth(0.8, []), /empty/);
  assert.equal(choices[1].value, 0.85);
});

test('pill text mixes 45 percent text into the pill, and word fill tints toward text', () => {
  assert.equal(pillText('#FFFFFF', '#000000'), '#737373');
  assert.equal(pillText('#000000', '#FFFFFF'), '#8C8C8C');
  assert.equal(pillText('#FF0000', '#00FF00'), '#738C00');
  assert.equal(pillText('#ffffff', '#ffffff'), '#FFFFFF');
  assert.equal(gradientTint('#FFFFFF', '#000000'), '#595959');
  assert.equal(gradientTint('#FF0000', '#00FF00'), '#59A600');
  assert.equal(gradientTint('#123456', '#123456'), '#123456');
});

test('black text on a white pill has readable spoken and muted words without a warning', () => {
  const ratios = pillContrast('#000000', '#FFFFFF');
  assert.equal(pillText('#000000', '#FFFFFF'), '#8C8C8C');
  assert.equal(ratios.spoken, 21);
  assert.ok(Math.abs(ratios.muted - 3.36) < 0.01);
  assert.equal(preview().pillWarning('#000000', '#FFFFFF', defaults(schema).captions), null);
});

test('pill warns for muted words below 3 even when spoken words pass 4.5', () => {
  const ratios = pillContrast('#555555', '#FFFFFF');
  assert.ok(ratios.spoken >= 4.5);
  assert.ok(ratios.muted < 3);
  const {WORDS, pillWarning} = preview();
  const captions = {...defaults(schema).captions, max_words_per_phrase: 3, max_characters_per_phrase: 18};
  const phrases = buildPhrases(WORDS, captions);
  assert.ok(phrases.some((phrase) => phrase.ids.length === 3));
  const message = pillWarning('#555555', '#FFFFFF', captions);
  assert.match(message, /Spoken or upcoming pill words are hard to read/);
  assert.equal(message.split('.').filter(Boolean).length, 1);
});

test('One word pills do not warn about muted words that never appear', () => {
  const {WORDS, pillWarning} = preview();
  const captions = {...defaults(schema).captions, max_words_per_phrase: 1, max_characters_per_phrase: 16};
  const phrases = buildPhrases(WORDS, captions);
  assert.ok(phrases.length > 0 && phrases.every((phrase) => phrase.ids.length === 1));
  assert.equal(hasUpcomingWords(phrases), false);
  assert.equal(pillWarning('#555555', '#FFFFFF', captions), null);
});

test('custom word and character limits with only single-word phrases skip muted warnings', () => {
  const {WORDS, pillWarning} = preview();
  for (const [words, chars] of [[1, 40], [8, 3]]) {
    const captions = {...defaults(schema).captions, max_words_per_phrase: words, max_characters_per_phrase: chars};
    const phrases = buildPhrases(WORDS, captions);
    assert.ok(phrases.length > 0 && phrases.every((phrase) => phrase.ids.length === 1));
    assert.equal(hasUpcomingWords(phrases), false);
    assert.equal(pillWarning('#555555', '#FFFFFF', captions), null);
  }
});

test('upcoming words are detected across all phrases, including after a single-word phrase', () => {
  const words = ['First.', 'Two', 'words.'].map((text, i) => ({id: `w${i}`, text, start: i, end: i + 0.5}));
  const phrases = buildPhrases(words, {max_words_per_phrase: 3, max_characters_per_phrase: 40});
  assert.deepEqual(phrases.map((phrase) => phrase.ids.length), [1, 2]);
  assert.equal(hasUpcomingWords(phrases), true);
  assert.equal(hasUpcomingWords([]), false);
});

test('low spoken contrast warns for every phrase mode, including custom limits', () => {
  const {pillWarning} = preview();
  for (const [words, chars] of [[1, 16], [3, 18], [5, 24], [8, 40], [1, 40], [8, 3]]) {
    const captions = {...defaults(schema).captions, max_words_per_phrase: words, max_characters_per_phrase: chars};
    assert.equal(typeof pillWarning('#FFFFFF', '#EEEEEE', captions), 'string');
  }
});

test('pill still warns for low spoken contrast and reports both ratios in the disk helper', () => {
  const ratios = pillContrast('#FFFFFF', '#EEEEEE');
  assert.ok(ratios.spoken < 4.5);
  assert.ok(ratios.muted < 3);
  assert.equal(typeof preview().pillWarning('#FFFFFF', '#EEEEEE', defaults(schema).captions), 'string');
  const disk = {window: {}};
  vm.runInNewContext(pageFile('looks.js'), disk);
  assert.deepEqual(JSON.parse(JSON.stringify(disk.window.YapLooks.pillContrast('#000000', '#FFFFFF'))),
    pillContrast('#000000', '#FFFFFF'));
});

test('word fill follows word timing, stays filled, and handles a very short word', () => {
  assert.equal(wordFill(0.5, 1, 2), 0);
  assert.equal(wordFill(1, 1, 2), 0);
  assert.equal(wordFill(1.25, 1, 2), 0.25);
  assert.equal(wordFill(1.5, 1, 2), 0.5);
  assert.equal(wordFill(2, 1, 2), 1);
  assert.equal(wordFill(3, 1, 2), 1);
  assert.equal(wordFill(0, 0, 0), 0);
  assert.equal(wordFill(1 / 60, 0, 0), 0.5);
  assert.equal(wordFill(1 / 30, 0, 0), 1);
});

test('the classic script exports the same helpers when opened from disk', () => {
  const disk = {window: {}};
  vm.runInNewContext(pageFile('looks.js'), disk);
  assert.deepEqual(Array.from(disk.window.YapLooks.highlightLooks(schema), (look) => look.id), order);
  assert.equal(disk.window.YapLooks.pillText('#FFFFFF', '#000000'), '#737373');
  assert.equal(disk.window.YapLooks.hasUpcomingWords([{ids: [0]}]), false);
  assert.equal(disk.window.YapLooks.hasUpcomingWords([{ids: [0, 1]}]), true);
});

test('learned styles, preset switches and Undo preserve every new look and custom width', () => {
  for (const look of order) {
    const style = defaults(schema);
    delete style.internal;
    style.captions.highlight_style = look;
    style.captions.width_scale = 0.873;
    const sources = {};
    for (const section of ['captions', 'hook', 'graphics', 'music']) {
      for (const key of Object.keys(style[section])) {
        sources[`${section}.${key}`] = {source: 'base', status: null, reason: 'From the base.'};
      }
    }
    const learned = readLearned({
      schema: 'yap-studio/taste-measurements@1',
      proposed_style: {base: {kind: 'schema_defaults', name: 'Schema defaults'}, sources,
        style: {...style, schema: 'yap-studio/resolved-style@1'}},
    }, schema);
    assert.equal(learned.style.captions.highlight_style, look);
    assert.equal(learned.style.captions.width_scale, 0.873);
    const state = {style: learned.style, startStyle: copy(learned.style), startName: 'Learned style',
      presetId: null, learned, custom: {}, name: 'My style'};
    const preset = {id: 'test', name: 'Test', style: defaults(schema)};
    const {next, previous} = switchPreset(state, preset);
    assert.equal(next.style.captions.width_scale, 1);
    const restored = undoPreset({...state, ...next}, previous);
    assert.equal(restored.style.captions.highlight_style, look);
    assert.equal(restored.style.captions.width_scale, 0.873);
    assert.equal(JSON.parse(JSON.stringify(restored.style)).captions.width_scale, 0.873);
  }
});

// Run the actual preview methods with word boxes supplied by the test, without a browser.
function preview({reducedMotion = true, document = {querySelector: () => null}} = {}) {
  const sandbox = {window: {YAP_STYLE_DATA: context.window.YAP_STYLE_DATA,
    YapLooks: require('../../docs/design/style-page/looks.js'),
    YapCaptionPorts: require('../../docs/design/style-page/caption-ports.js'),
    YapPhrases: require('../../docs/design/style-page/phrases.js')},
  matchMedia: () => ({matches: reducedMotion}), document};
  vm.runInNewContext(pageFile('app.js').replace(/\n  start\(\);\n\}\)\(\);\s*$/,
    '\n  window.previewTest = {Frame, WORDS, state, output, pillWarning, buildGround, changedGroups, syncers, effectiveStyle, setRefresh: fn => { refresh = fn; }};\n})();'), sandbox);
  const {Frame, WORDS, state, output, pillWarning} = sandbox.window.previewTest;
  const frame = Object.create(Frame.prototype);
  const properties = {};
  frame.stage = {style: {setProperty: (key, value) => {properties[key] = value;},
    getPropertyValue: key => properties[key]}, dataset: {}};
  frame.els = Object.fromEntries(['cap', 'line', 'words', 'mark', 'hook', 'gfx', 'ground'].map((key) => [key, {style: {}}]));
  frame.els.line.offsetWidth = 300;
  frame.els.words.offsetHeight = 80;
  frame.style = defaults(schema);
  const phrase = {ids: [0, 1, 2], start: WORDS[0].start, end: WORDS[2].end + 0.5, stress: WORDS[2].id};
  frame.tl = {phrases: [phrase], hook: null, graphic: {start: 100, end: 101}};
  frame.pi = 0;
  frame.spans = phrase.ids.map((id) => {
    const classes = new Set(['w']);
    if (WORDS[id].id === phrase.stress) {classes.add('is-key'); classes.add('is-stress');}
    return {style: {}, classList: {contains: (name) => classes.has(name),
      toggle: (name, on) => on ? classes.add(name) : classes.delete(name)}};
  });
  frame.rects = [{x: 100, y: 0, w: 40, h: 80}, {x: 160, y: 0, w: 80, h: 80}, {x: 260, y: 0, w: 120, h: 80}];
  frame.lineLeft = 300;
  return {...sandbox.window.previewTest, frame, properties, WORDS, phrase, state, output, pillWarning, document: sandbox.document};
}

test('background and placement controls change the exported style and change tracking', () => {
  const nodes = [];
  const document = {activeElement: null, querySelector: () => ({append() {}}), createElement: tag => {
    const node = {tag, attributes: {}, children: [], events: {}, style: {setProperty() {}},
      setAttribute(key, value) {this.attributes[key] = value; if (key === 'value') this.value = value;},
      append(...children) {this.children.push(...children);},
      addEventListener(key, callback) {this.events[key] = callback;}};
    nodes.push(node);
    return node;
  }};
  const {state, output, buildGround, changedGroups, syncers, setRefresh, effectiveStyle} = preview({document});
  state.style = defaults(schema);
  state.startStyle = copy(state.style);
  state.name = 'Boxed captions';
  let refreshed = 0;
  setRefresh(() => {refreshed++;});
  buildGround();
  for (const choice of ['box', 'fixed']) {
    nodes.find(node => node.attributes.value === choice).events.change();
  }
  const color = nodes.find(node => node.attributes.type === 'color');
  color.value = '#abcdef';
  color.events.input();
  const ranges = nodes.filter(node => node.attributes.type === 'range');
  [0.01, 36, 24].forEach((value, i) => {ranges[i].value = String(value); ranges[i].events.input();});
  assert.equal(refreshed, 6);
  assert.deepEqual(copy(changedGroups()), ['ground']);
  const saved = copy(output());
  assert.equal(saved.captions.background_mode, 'box');
  assert.equal(saved.captions.placement, 'fixed');
  assert.equal(saved.captions.background_color, '#ABCDEF');
  assert.equal(saved.captions.background_opacity, 0.01);
  assert.equal(saved.captions.background_padding_px, 36);
  assert.equal(saved.captions.background_radius_px, 24);
  // Preview contrast never replaces the saved ordinary text color.
  state.style.captions.text_color = '#FF0000';
  state.style.captions.background_color = '#000000';
  // Below half strength the footage dominates, so the chosen color stays (engine rule).
  assert.equal(effectiveStyle(state.style).captions.text_color, '#FF0000');
  state.style.captions.background_opacity = 0.5;
  assert.equal(effectiveStyle(state.style).captions.text_color, '#FFFFFF');
  assert.equal(output().captions.text_color, '#FF0000');
  for (const sync of syncers) sync(state.style);
  assert.equal(nodes.find(node => node.attributes.value === 'box').checked, true);
  assert.equal(nodes.find(node => node.attributes.value === 'fixed').checked, true);
  Object.assign(state.style.captions, state.startStyle.captions);
  assert.deepEqual(copy(changedGroups()), []);
  for (const sync of syncers) sync(state.style);
  assert.equal(nodes.find(node => node.attributes.value === 'none').checked, true);
  assert.equal(nodes.find(node => node.attributes.value === 'avoid_face').checked, true);
  assert.match(pageFile('app.js'), /buildStyle\(\);\s*buildGround\(\);/);
});

test('preview draws fitted or full width grounds and resets face avoidance for fixed placement', () => {
  const {frame, properties} = preview();
  Object.assign(frame.style.captions, {background_mode: 'box', background_color: '#000000',
    background_opacity: 0.01, background_padding_px: 18, background_radius_px: 12,
    size_px: 48, position: 'middle', top_px: 820, placement: 'fixed', shadow_opacity: 0});
  frame.setStyle(frame.style);
  frame.els.line.offsetWidth = 300;
  frame.els.words.offsetHeight = 80;
  frame.layoutGround();
  const ground = frame.els.ground.style;
  assert.equal(ground.background, 'rgb(0 0 0 / 0.01)');
  assert.equal(ground.width, '336px');
  assert.equal(ground.height, '116px');
  assert.equal(ground.borderRadius, '12px');
  assert.equal(properties['--cap-color'], '#FFFFFF');
  assert.equal(properties['--cap-outline'], '2px #111111');
  const fixedTop = parseFloat(frame.els.cap.style.top);
  frame.style.captions.placement = 'avoid_face';
  frame.layoutGround();
  const avoidTop = parseFloat(frame.els.cap.style.top);
  assert.ok(avoidTop > fixedTop);
  assert.ok(avoidTop - 18 >= 1150 + frame.style.graphics.face_clearance_px);
  frame.style.captions.placement = 'fixed';
  frame.layoutGround();
  assert.equal(parseFloat(frame.els.cap.style.top), fixedTop);
  frame.style.captions.background_mode = 'band';
  frame.layoutGround();
  assert.equal(ground.width, '1080px');
  assert.equal(ground.borderRadius, '0px');
  frame.style.captions.background_mode = 'none';
  frame.layoutGround();
  assert.equal(ground.display, 'none');
});

for (const mode of ['box', 'band']) {
  test(`transparent ${mode} previews match disabled grounds after an opacity change`, () => {
    const {frame, properties} = preview();
    for (const highlight of order) {
      for (const scale of [0.6, 1]) {
        for (const position of ['middle', 'bottom']) {
          for (const placement of ['fixed', 'avoid_face']) {
            const style = defaults(schema);
            Object.assign(style.captions, {background_mode: 'none', background_padding_px: 80,
              highlight_style: highlight, width_scale: scale, size_px: 48, position, placement,
              top_px: position === 'middle' ? 820 : 1660});
            frame.setStyle(style);
            frame.layoutGround();
            const expected = {width: properties['--cap-wrap-width'], inset: properties['--cap-inset'],
              top: frame.els.cap.style.top};
            if (highlight === 'none' && scale === 1) assert.equal(expected.width, '940px');

            style.captions.background_mode = mode;
            style.captions.background_opacity = 0.01;
            frame.setStyle(style);
            frame.layoutGround();
            assert.equal(frame.els.ground.style.display, 'block');
            assert.ok(parseFloat(properties['--cap-wrap-width']) < parseFloat(expected.width));
            assert.match(properties['--cap-outline'], /^2px /);
            if (highlight === 'none' && scale === 1) {
              assert.equal(properties['--cap-wrap-width'], '780px');
            }

            style.captions.background_opacity = 0;
            frame.setStyle(style);
            frame.layoutGround();
            const label = `${highlight}, scale ${scale}, ${position}, ${placement}`;
            assert.equal(properties['--cap-wrap-width'], expected.width, label);
            assert.equal(properties['--cap-inset'], expected.inset, label);
            assert.equal(properties['--cap-outline'], '0px transparent', label);
            assert.equal(frame.els.ground.style.display, 'none', label);
            assert.equal(frame.els.cap.style.top, expected.top, label);
          }
        }
      }
    }
  });
}

for (const mode of ['none', 'box', 'band']) {
  test(`${mode} preview preserves saved text color below half opacity`, () => {
    const {frame, properties, state, output, effectiveStyle} = preview();
    state.style = defaults(schema);
    for (const [background, text, contrasting] of [
      ['#FFFFFF', '#FFFFFF', '#111111'], ['#000000', '#111111', '#FFFFFF'],
      ['#ABCDEF', '#FF0000', '#111111'],
    ]) {
      for (const opacity of [0, 0.01, 0.49, 0.4999, 0.5, 0.85, 1]) {
        Object.assign(state.style.captions, {background_mode: mode, background_color: background,
          background_opacity: opacity, text_color: text});
        const saved = JSON.stringify(output());
        const expected = mode !== 'none' && opacity >= 0.5 ? contrasting : text;
        const label = `${background}, opacity ${opacity}`;
        assert.equal(effectiveStyle(state.style).captions.text_color, expected, label);
        frame.setStyle(state.style);
        assert.equal(properties['--cap-color'], expected, label);
        assert.equal(JSON.stringify(output()), saved, label);
      }
    }
  });
}

test('Save style output keeps the new looks and exact letter width', () => {
  const {state, output} = preview();
  state.name = 'My custom style';
  for (const look of ['pill', 'gradient', 'editorial', 'slam', 'weight_shift', 'clip_wipe']) {
    state.style = defaults(schema);
    state.style.captions.highlight_style = look;
    state.style.captions.width_scale = 0.873;
    const saved = JSON.parse(JSON.stringify(output()));
    assert.equal(saved.captions.highlight_style, look);
    assert.equal(saved.captions.width_scale, 0.873);
    assert.equal(saved.name, 'My custom style');
    assert.equal(saved.schema_version, 1);
    assert.equal('internal' in saved, false);
  }
});

test('pill lights earlier words, keeps upcoming words muted, and resets on a backward seek', () => {
  const {frame, WORDS} = preview();
  frame.style.captions.highlight_style = 'pill';
  frame.render(WORDS[1].start + 0.01);
  assert.deepEqual(frame.spans.map((span) => span.classList.contains('is-spoken')), [true, true, false]);
  assert.match(frame.els.mark.style.cssText, /opacity:1;.*border-radius:/);
  frame.render(WORDS[0].start);
  assert.deepEqual(frame.spans.map((span) => span.classList.contains('is-spoken')), [true, false, false]);
  assert.equal(frame.spans[2].classList.contains('is-stress'), true);
});

test('word fill sweeps the current word, keeps previous words filled and upcoming words empty', () => {
  const {frame, WORDS} = preview();
  frame.style.captions.highlight_style = 'gradient';
  frame.render((WORDS[1].start + WORDS[1].end) / 2);
  assert.match(frame.spans[0].style.backgroundPosition, /calc\(0% - 0em\)/);
  assert.match(frame.spans[1].style.backgroundPosition, /calc\(50% - 0\.04em\)/);
  assert.match(frame.spans[2].style.backgroundPosition, /calc\(100% - 0\.08em\)/);
  assert.equal(frame.els.mark.style.opacity, '0');
  frame.render(WORDS[0].start);
  assert.match(frame.spans[1].style.backgroundPosition, /calc\(100% - 0\.08em\)/);
});

test('bold key word changes spoken state without remeasuring the phrase', () => {
  const {frame, WORDS} = preview();
  frame.style.captions.highlight_style = 'editorial';
  let measurements = 0;
  frame.measureWords = () => {measurements += 1;};
  frame.render(WORDS[1].start);
  assert.equal(measurements, 0);
  frame.render(WORDS[2].start);
  const key = frame.spans.filter((span) => span.classList.contains('is-key') && span.classList.contains('is-spoken'));
  assert.equal(key.length, 1);
  assert.equal(key[0], frame.spans[2]);
  assert.equal(measurements, 0);
  frame.render(WORDS[2].end);
  assert.equal(measurements, 0);
  frame.render(WORDS[1].start);
  assert.equal(measurements, 0);
  assert.equal(frame.spans[2].classList.contains('is-spoken'), false);
  assert.equal(frame.spans[2].classList.contains('is-stress'), true);
});

test('editorial reserves bold key word width before measuring and colors only spoken words', () => {
  const css = pageFile('styles.css');
  const keyRule = css.match(/\.stage\[data-hl="editorial"\] \.w\.is-key\s*\{([^}]+)\}/);
  assert.ok(keyRule);
  assert.match(keyRule[1], /font-weight:\s*700/);
  const spokenRule = css.match(/\.stage\[data-hl="editorial"\] \.w\.is-key\.is-spoken\s*\{([^}]+)\}/);
  assert.ok(spokenRule);
  assert.match(spokenRule[1], /color:\s*var\(--hl\)/);
  assert.doesNotMatch(spokenRule[1], /font-weight/);

  const {frame, document} = preview();
  frame.style.captions.highlight_style = 'editorial';
  document.createElement = () => {
    const span = {className: '', classList: {
      toggle: (name, on) => {if (on) span.className += ` ${name}`;},
      contains: (name) => span.className.split(' ').includes(name),
    }};
    return span;
  };
  frame.els.words = {style: {}, children: [], append: (node) => {
    if (typeof node !== 'string') frame.els.words.children.push(node);
  }};
  let measured = false;
  frame.measureWords = () => {
    measured = true;
    assert.deepEqual(Array.from(frame.spans, (span) => span.classList.contains('is-key')), [false, false, true]);
    assert.ok(frame.spans.every((span) => !span.classList.contains('is-spoken')));
  };
  frame.renderPhrase(0);
  assert.equal(measured, true);
});

// Supply only the DOM operations and widths used to construct the new caption looks.
function captionPreview(mode, options = {}) {
  const result = preview(options);
  const {frame, document, WORDS} = result;
  const widths = options.widths || [40, 80, 120];
  document.createElement = () => {
    const node = {style: {}, className: '', children: [], contents: [], attributes: {}, parent: null};
    node.classList = {
      contains: (name) => node.className.split(' ').includes(name),
      toggle: (name, on) => {
        const classes = new Set(node.className.split(' ').filter(Boolean));
        if (on) classes.add(name); else classes.delete(name);
        node.className = [...classes].join(' ');
      },
    };
    node.setAttribute = (name, value) => {node.attributes[name] = value;};
    node.remove = () => {
      if (!node.parent) return;
      node.parent.children = node.parent.children.filter((child) => child !== node);
      node.parent.contents = node.parent.contents.filter((child) => child !== node);
      node.parent = null;
    };
    node.append = (...children) => children.forEach((child) => {
      if (typeof child !== 'string') {
        child.remove();
        child.parent = node;
        node.children.push(child);
      }
      node.contents.push(child);
    });
    node.replaceChildren = (...children) => {
      [...node.children].forEach((child) => child.remove());
      node.contents = [];
      node.append(...children);
    };
    Object.defineProperty(node, 'textContent', {
      get: () => node.contents.map((child) => typeof child === 'string' ? child : child.textContent).join(''),
      set: (text) => {node.replaceChildren(String(text));},
    });
    Object.defineProperty(node, 'offsetWidth', {get: () => {
      if (node.classList.contains('w') && node.children.length) {
        return Math.max(...node.children.map((child) => child.offsetWidth));
      }
      if (node.textContent === ' ') return 20;
      const index = WORDS.findIndex((word) => word.text === node.textContent);
      return index >= 0 ? widths[index] : node.textContent.length * 20;
    }});
    return node;
  };
  frame.els.words = document.createElement('span');
  frame.style.captions.highlight_style = mode;
  frame.measureWords = () => {};
  frame.renderPhrase(0);
  return result;
}

test('kinetic slam shows one centered capital word and punches in over 120 ms', () => {
  const {frame, WORDS, properties} = captionPreview('slam', {reducedMotion: false});
  frame.render(WORDS[1].start);
  assert.deepEqual(Array.from(frame.spans, (span) => span.style.visibility), ['hidden', 'visible', 'hidden']);
  assert.equal(frame.spans[1].style.transform, 'scale(1.3)');
  assert.equal(frame.spans[1].style.opacity, 0);
  frame.render(WORDS[1].start + 0.06);
  assert.ok(frame.spans[1].style.opacity > 0 && frame.spans[1].style.opacity < 1);
  frame.render(WORDS[1].start + 0.121);
  assert.equal(frame.spans[1].style.transform, 'scale(1)');
  assert.equal(frame.spans[1].style.opacity, 1);
  frame.render(WORDS[2].start + 0.121);
  assert.deepEqual(Array.from(frame.spans, (span) => span.style.visibility), ['hidden', 'hidden', 'visible']);
  assert.equal(frame.spans[2].classList.contains('is-key'), true);
  frame.render(WORDS[0].start + 0.121);
  assert.deepEqual(Array.from(frame.spans, (span) => span.style.visibility), ['visible', 'hidden', 'hidden']);
  assert.equal(frame.els.mark.style.opacity, '0');
  frame.setStyle(frame.style);
  assert.equal(properties['--cap-width-scale'], frame.style.captions.width_scale * 0.8);
  const css = pageFile('styles.css');
  assert.match(css, /\[data-hl="slam"\] \.cap-words\s*\{[^}]*display: grid;[^}]*text-transform: uppercase;[^}]*font-weight: 800/);
  assert.match(css, /\[data-hl="slam"\] \.w\s*\{[^}]*grid-area: 1 \/ 1;[^}]*transform-origin: center/);
  assert.match(css, /\[data-hl="slam"\] \.w\.is-key[^}]*color: var\(--hl\)/);
});

test('kinetic slam respects reduced motion without losing the current word', () => {
  const {frame, WORDS} = captionPreview('slam');
  frame.render(WORDS[1].start);
  assert.equal(frame.spans[1].style.visibility, 'visible');
  assert.equal(frame.spans[1].style.opacity, 1);
  assert.equal(frame.spans[1].style.transform, 'scale(1)');
});

test('weight shift balances lowercase lines and bolds the line being spoken on seeks', () => {
  const {frame, WORDS} = captionPreview('weight_shift');
  assert.deepEqual(copy(frame.rows.map(({start, end}) => [start, end])), [[0, 2], [2, 3]]);
  const nodes = Array.from(frame.rows, ({node}) => node);
  assert.deepEqual(nodes.map((row) => row.textContent),
    [`${WORDS[0].text} ${WORDS[1].text}`, WORDS[2].text]);
  frame.render(WORDS[1].start);
  assert.deepEqual(nodes.map((row) => row.style.fontWeight), ['700', '400']);
  assert.deepEqual(nodes.map((row) => row.style.opacity), ['1', '0.75']);
  frame.render(WORDS[2].start);
  assert.deepEqual(nodes.map((row) => row.style.fontWeight), ['400', '700']);
  assert.deepEqual(nodes.map((row) => row.style.opacity), ['0.75', '1']);
  frame.render(WORDS[0].start);
  assert.deepEqual(nodes.map((row) => row.style.fontWeight), ['700', '400']);
  assert.equal(frame.els.mark.style.opacity, '0');
  assert.match(pageFile('styles.css'), /\[data-hl="weight_shift"\] \.cap-words\s*\{[^}]*text-transform: lowercase;[^}]*font-weight: 700/);
});

test('weight shift eases weight and opacity for 100 ms and restores intermediate state on seeks', () => {
  const {frame, WORDS} = captionPreview('weight_shift', {reducedMotion: false});
  const nodes = Array.from(frame.rows, ({node}) => node);
  const switchAt = WORDS[2].start;
  const weights = () => nodes.map((row) => Number(row.style.fontWeight));
  const opacities = () => nodes.map((row) => Number(row.style.opacity));
  frame.render(switchAt);
  assert.deepEqual(weights(), [700, 400]);
  assert.deepEqual(opacities(), [1, 0.75]);
  frame.render(switchAt + 0.05);
  // Reference values from cubic-bezier(.2,.7,.2,1) at half of the switch interval.
  const midwayWeights = weights();
  const midwayOpacities = opacities();
  assert.ok(Math.abs(midwayWeights[0] - 421.1263538) < 0.0001);
  assert.ok(Math.abs(midwayWeights[1] - 678.8736462) < 0.0001);
  assert.ok(Math.abs(midwayOpacities[0] - 0.7676052948) < 0.000001);
  assert.ok(Math.abs(midwayOpacities[1] - 0.9823947052) < 0.000001);
  frame.render(switchAt + 0.101);
  assert.deepEqual(weights(), [400, 700]);
  assert.deepEqual(opacities(), [0.75, 1]);
  frame.render(switchAt + 0.05);
  assert.deepEqual(weights(), midwayWeights);
  assert.deepEqual(opacities(), midwayOpacities);
  frame.render(switchAt - 0.01);
  assert.deepEqual(weights(), [700, 400]);
  assert.deepEqual(opacities(), [1, 0.75]);
});

test('weight shift clamps its transition to the phrase end', () => {
  const {frame, WORDS, phrase} = captionPreview('weight_shift', {reducedMotion: false});
  const switchAt = WORDS[2].start;
  phrase.end = switchAt + 0.04;
  frame.render(switchAt + 0.02);
  const nodes = Array.from(frame.rows, ({node}) => node);
  assert.ok(Math.abs(Number(nodes[0].style.fontWeight) - 421.1263538) < 0.0001);
  assert.ok(Math.abs(Number(nodes[1].style.fontWeight) - 678.8736462) < 0.0001);
  frame.render(phrase.end - 0.000001);
  assert.ok(Number(nodes[0].style.fontWeight) < 400.001);
  assert.ok(Number(nodes[1].style.fontWeight) > 699.999);
});

test('a single weight shift word stays bold and changing looks clears its row styles', () => {
  const {frame, WORDS} = captionPreview('weight_shift');
  frame.tl.phrases[0].ids = [0];
  frame.renderPhrase(0);
  frame.render(WORDS[0].start);
  assert.equal(frame.rows.length, 1);
  assert.equal(frame.els.words.style.fontWeight, '700');
  assert.equal(frame.els.words.style.opacity, '1');
  frame.style.captions.highlight_style = 'none';
  frame.renderPhrase(0);
  assert.equal(frame.els.words.style.fontWeight, '');
  assert.equal(frame.els.words.style.opacity, '');
  assert.equal(frame.els.line.style.fontSize, '');
});

test('clip wipe dims upcoming words and reveals each word from left to right over its duration', () => {
  const {frame, WORDS} = captionPreview('clip_wipe');
  assert.equal(frame.rows.length, 1);
  const dim = frame.spans.map((span) => span.children[0]);
  assert.ok(dim.every((node) => node.className === 'cap-dim' && node.attributes['aria-hidden'] === 'true'));
  assert.deepEqual(frame.reveals.map((node) => node.textContent), WORDS.slice(0, 3).map((word) => word.text));
  frame.render((WORDS[1].start + WORDS[1].end) / 2);
  assert.equal(frame.reveals[0].style.clipPath, 'inset(0 0% 0 0)');
  assert.match(frame.reveals[1].style.clipPath, /inset\(0 50(?:\.0+)?% 0 0\)/);
  assert.equal(frame.reveals[2].style.clipPath, 'inset(0 100% 0 0)');
  frame.render(WORDS[2].end);
  assert.ok(frame.reveals.every((node) => node.style.clipPath === 'inset(0 0% 0 0)'));
  frame.render(WORDS[0].start);
  assert.ok(frame.reveals.every((node) => node.style.clipPath === 'inset(0 100% 0 0)'));
  assert.equal(frame.spans[2].classList.contains('is-key'), true);
  assert.equal(frame.els.mark.style.opacity, '0');
  const css = pageFile('styles.css');
  assert.match(css, /\.cap-dim\s*\{\s*opacity: 0\.35/);
  assert.match(css, /\[data-hl="clip_wipe"\] \.cap-words\s*\{[^}]*text-transform: uppercase;[^}]*white-space: nowrap/);
  assert.match(css, /\[data-hl="clip_wipe"\] \.w\.is-key\s*\{[^}]*color: var\(--hl\)/);
});

test('a long clip wipe phrase uses two balanced rows and fits each row before playback', () => {
  const {frame, WORDS} = captionPreview('clip_wipe', {widths: [500, 400, 500]});
  assert.deepEqual(copy(frame.rows.map(({start, end}) => [start, end])), [[0, 1], [1, 3]]);
  const nodes = Array.from(frame.rows, ({node}) => node);
  assert.ok(nodes.every((node) => node.className === 'cap-row'));
  assert.equal(frame.spans[0].parent, nodes[0]);
  assert.equal(frame.spans[1].parent, nodes[1]);
  assert.equal(frame.spans[2].parent, nodes[1]);
  assert.equal(frame.els.line.style.fontSize, `${frame.style.captions.size_px}px`);
  frame.render((WORDS[1].start + WORDS[1].end) / 2);
  assert.equal(frame.reveals[0].style.clipPath, 'inset(0 0% 0 0)');
  assert.match(frame.reveals[1].style.clipPath, /inset\(0 50(?:\.0+)?% 0 0\)/);
  assert.equal(frame.reveals[2].style.clipPath, 'inset(0 100% 0 0)');
  frame.style.captions.width_scale = 0.6;
  frame.renderPhrase(0);
  assert.equal(frame.rows.length, 1);
  assert.equal(frame.els.line.style.fontSize, `${frame.style.captions.size_px}px`);
});

test('clip wipe keeps a phrase at the width limit on one row and splits only when it exceeds the limit', () => {
  for (const [widths, rows] of [[[300, 300, 300], 1], [[301, 300, 300], 2]]) {
    const {frame} = captionPreview('clip_wipe', {widths});
    assert.equal(frame.rows.length, rows);
  }
  const widths = [600, 800, 1200];
  const clip = captionPreview('clip_wipe', {widths}).frame;
  const weight = captionPreview('weight_shift', {widths}).frame;
  assert.equal(clip.els.line.style.fontSize, weight.els.line.style.fontSize);
  assert.deepEqual(copy(clip.rows.map(({start, end}) => [start, end])),
    copy(weight.rows.map(({start, end}) => [start, end])));
});

test('new caption looks fit their widest word or row before playback', () => {
  for (const mode of ['slam', 'weight_shift', 'clip_wipe']) {
    const {frame, WORDS} = captionPreview(mode, {widths: [600, 800, 1200]});
    const fitted = frame.els.line.style.fontSize;
    assert.ok(parseFloat(fitted) < frame.style.captions.size_px);
    frame.render(WORDS[0].start);
    frame.render(WORDS[2].start);
    assert.equal(frame.els.line.style.fontSize, fitted);
  }
});

test('slam and clip wipe share colored text warnings while weight shift has no highlight warning', () => {
  const warningModes = pageFile('app.js').match(/\[([^\]]+)\]\.includes\(s\.captions\.highlight_style\) && luminance/);
  assert.ok(warningModes);
  assert.match(warningModes[1], /'color'/);
  assert.match(warningModes[1], /'slam'/);
  assert.match(warningModes[1], /'clip_wipe'/);
  assert.doesNotMatch(warningModes[1], /'weight_shift'/);
});

test('layout cache includes letter width and the editorial key word', () => {
  const {frame, properties} = preview();
  frame.t = 0;
  let measurements = 0;
  frame.invalidate = () => {measurements += 1;};
  frame.render = () => {};
  frame.setStyle(defaults(schema));
  assert.equal(measurements, 1);
  const narrow = defaults(schema);
  narrow.captions.width_scale = 0.7;
  frame.setStyle(narrow);
  assert.equal(measurements, 2);
  assert.equal(properties['--cap-width-scale'], 0.7);
  assert.equal(properties['--cap-wrap-width'], `${940 / 0.7}px`);
  assert.equal(properties['--cap-inset'], `${(1080 - 940 / 0.7) / 2}px`);
  frame.setStyle(copy(narrow));
  assert.equal(measurements, 2);
  narrow.captions.highlight_color = '#123456';
  frame.setStyle(narrow);
  assert.equal(measurements, 2);
  narrow.captions.highlight_style = 'editorial';
  frame.setStyle(narrow);
  assert.equal(measurements, 3);
  narrow.captions.highlight_color = '#654321';
  frame.setStyle(narrow);
  assert.equal(measurements, 3);
});

test('condensed words and their sliding marks use stage pixels rather than host pixels', () => {
  const {frame} = preview();
  frame.style.captions.width_scale = 0.7;
  frame.host = {clientWidth: 270};
  frame.stage.getBoundingClientRect = () => ({left: 10});
  frame.els.line.getBoundingClientRect = () => ({left: 90, top: 400});
  frame.spans = [{getBoundingClientRect: () => ({left: 130, top: 410, width: 35, height: 20})}];
  frame.measureWords();
  assert.equal(frame.lineLeft, 320);
  assert.deepEqual(JSON.parse(JSON.stringify(frame.rects)), [{x: 160, y: 40, w: 140, h: 80}]);
  const box = frame.markBox(frame.rects[0], 'moving_block');
  assert.equal(box.x + box.w / 2, 230);
  assert.equal(box.w, 140 + frame.style.captions.size_px * 0.16);
});
