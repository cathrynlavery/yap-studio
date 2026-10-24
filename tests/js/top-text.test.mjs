import assert from 'node:assert/strict';
import test from 'node:test';
import {topTextChoice, withTopText} from '../../src/yap_studio/gallery/static/choices.mjs';
import {topTextChoice as homeChoice, topTextSummary} from '../../src/yap_studio/homepage/static/home-model.mjs';

test('top text is unanswered until a deliberate answer, preserving each title look', () => {
  for (const look of ['shimmer', 'none', 'text_hook']) {
    const style = {hook: {style: look}};
    assert.equal(topTextChoice(style), null);
    for (const [answer, expected] of [['none', 'none'], ['opening', look === 'shimmer' ? look : 'quiet_serif'], ['whole', 'text_hook']]) {
      const next = withTopText(style, answer);
      assert.deepEqual(next.hook, {style: expected, answered: true});
      assert.equal(topTextChoice(next), answer);
      const home = {hook_answered: true, hook_style: expected};
      assert.equal(homeChoice(home), answer);
      assert.notEqual(topTextSummary(home), 'Not answered');
    }
    assert.deepEqual(style, {hook: {style: look}});
  }
  assert.equal(homeChoice({hook_style: 'text_hook'}), '');
  assert.equal(topTextSummary(null), 'Not answered');
  assert.throws(() => withTopText({hook: {}}, 'invented'));
});

test('horizontal offers a chosen opening and refuses the whole-video label', () => {
  for (const hook_style of ['none', 'quiet_serif', 'text_hook']) {
    const style = {hook_answered: true, hook_style};
    assert.equal(homeChoice(style, {shape: 'horizontal'}), '');
    for (const top_text of ['none', 'opening', 'whole']) {
      const draft = {shape: 'horizontal', top_text, text_hook: 'My chosen title'};
      assert.equal(homeChoice(style, draft), top_text === 'whole' ? '' : top_text);
      assert.equal(homeChoice(style, {...draft, shape: 'vertical'}), top_text);
      assert.equal(draft.top_text, top_text);
    }
  }
});
