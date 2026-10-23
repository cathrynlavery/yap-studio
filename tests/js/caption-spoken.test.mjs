import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { captionLook, withCaptionLook, withMotion, screenSummary, labels } from '../../src/yap_studio/gallery/static/choices.mjs';

const style = { captions: { camera_3d: false, entrance: 'slide', highlight_style: 'none',
  max_words_per_phrase: 3, max_characters_per_phrase: 18, background_mode: 'box', background_opacity: 0.85 } };

test('Word by word selects spoken entrance and keeps the other caption choices', () => {
  assert.equal(labels.spoken, 'Word by word');
  const changed = withCaptionLook(style, 'spoken');
  assert.deepEqual(changed, { captions: { ...style.captions, entrance: 'spoken' } });
  assert.equal(captionLook(changed), 'spoken');
  assert.equal(screenSummary('captions', changed), 'Word by word');
  for (const look of ['plain', 'box', 'band', 'one', 'rise']) {
    assert.equal(captionLook(withCaptionLook(changed, look)), look);
  }
  assert.equal(withMotion(changed, 'Cut in').captions.entrance, 'slide');
  assert.equal(style.captions.entrance, 'slide');
});

test('the Captions screen offers Word by word next to Rise', () => {
  const source = readFileSync(new URL('../../src/yap_studio/gallery/static/pick.js', import.meta.url), 'utf8');
  assert.match(source, /\["rise", "Rise",[^\n]+\n?\s*\["spoken", "Word by word",/);
});
