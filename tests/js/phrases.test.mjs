import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const {buildPhrases} = require('../../docs/design/style-page/phrases.js');
const cases = JSON.parse(readFileSync(new URL('../fixtures/phrase-cases.json', import.meta.url), 'utf8'));
const pageFile = (name) => readFileSync(new URL(`../../docs/design/style-page/${name}`, import.meta.url), 'utf8');

for (const item of cases) {
  test(`shared phrase case: ${item.name}`, () => {
    const before = JSON.stringify(item);
    assert.deepEqual(buildPhrases(item.words, item.captions, item.options), item.phrases);
    assert.deepEqual(buildPhrases(item.words, item.captions, item.options), item.phrases);
    assert.equal(JSON.stringify(item), before);
  });
}

for (const [name, first, start, breaks, segment, merged] of [
  ['short gap', 'Zindle', 0.5, [], 's1', true],
  ['pause threshold', 'Zindle', 1.4, [], 's1', true],
  ['long gap', 'Zindle', 8, [], 's1', false],
  ['over pause threshold', 'Zindle', 1.401, [], 's1', false],
  ['sentence end', 'Zindle.', 0.5, [], 's1', false],
  ['forced break', 'Zindle', 0.5, ['b'], 's1', true],
  ['cut', 'Zindle', 0.5, [], 's2', false],
]) {
  test(`multiword fix respects speech: ${name}`, () => {
    const words = [
      {id: 'a', text: first, start: 0, end: 0.4, segment: 's1'},
      {id: 'b', text: 'Vay', start, end: start + 0.4, segment},
    ];
    const phrases = buildPhrases(words,
      {max_words_per_phrase: 4, max_characters_per_phrase: 80},
      {word_fixes: {'zindle vay': 'ZindleVay'}, forced_breaks: breaks});
    assert.equal(phrases.map((phrase) => phrase.text).join(' '),
      merged ? 'ZindleVay' : `${first} Vay`);
    assert.deepEqual(phrases.flatMap((phrase) => phrase.ids), merged ? ['a'] : ['a', 'b']);
  });
}

test('classic browser script exports the same engine without dependencies', () => {
  const context = {window: {}};
  vm.runInNewContext(pageFile('phrases.js'), context);
  for (const item of cases) {
    const actual = context.window.YapPhrases.buildPhrases(item.words, item.captions, item.options);
    assert.deepEqual(JSON.parse(JSON.stringify(actual)), item.phrases, item.name);
  }
});

test('page loads the classic phrase script before the app', () => {
  const scripts = [...pageFile('index.html').matchAll(/<script\s+src="([^"]+)"\s+defer><\/script>/g)];
  assert.deepEqual(scripts.map((match) => match[1]), ['style-data.js', 'phrases.js', 'looks.js', 'caption-ports.js', 'app.js']);
});

test('actual page sample and timeline match all four shared type cases', () => {
  const app = pageFile('app.js');
  const script = app.match(/const SCRIPT = \[.*?\n  \];/s)[0];
  const words = app.match(/const WORDS = \(\(\) => \{.*?\n  \}\)\(\);/s)[0];
  const types = app.match(/const TYPES = \[.*?\n  \];/s)[0];
  const timeline = app.match(/  function timeline\(style\) \{.*?\n  \}/s)[0];
  const context = {window: {YapPhrases: {buildPhrases}}};
  vm.runInNewContext(`${script}\n${words}\n${types}\n${timeline}\nthis.sample = WORDS; this.types = TYPES;`, context);
  for (const type of context.types) {
    const item = cases.find((entry) => entry.name === `page sample at ${type.words} words`);
    assert.ok(item);
    assert.equal(Object.hasOwn(item.options, 'stress_ids'), false);
    assert.deepEqual(JSON.parse(JSON.stringify(context.sample.map(({id, text, start, end}) => ({id, text, start, end})))), item.words);
    assert.deepEqual(item.captions, {max_words_per_phrase: type.words, max_characters_per_phrase: type.chars});
    const actual = context.timeline({
      captions: item.captions, hook: {style: 'none', duration_s: 0}, graphics: {hold_s: 1},
    }).phrases.map((phrase) => ({...phrase, ids: phrase.ids.map((index) => context.sample[index].id)}));
    assert.deepEqual(JSON.parse(JSON.stringify(actual)), item.phrases);
    assert.equal(context.sample.find((word) => word.cue).id, 'w11');
  }
});

test('caption limits must be positive integers', () => {
  for (const field of ['max_words_per_phrase', 'max_characters_per_phrase']) {
    for (const value of [0, -1, true, 1.5, '3']) {
      const captions = {max_words_per_phrase: 3, max_characters_per_phrase: 21, [field]: value};
      assert.throws(() => buildPhrases([], captions), /positive integers/);
    }
  }
});
