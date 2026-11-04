import assert from 'node:assert/strict';
import test from 'node:test';
import {topTextAnswers, topTextChoice, withTopText} from '../../src/yap_studio/gallery/static/choices.mjs';
import {topTextAnswers as homeAnswers, topTextChoice as homeChoice, topTextSummary} from '../../src/yap_studio/homepage/static/home-model.mjs';

test('top text names what appears at the opening and keeps the stored answer ids', () => {
  const answers = [['none', 'None'], ['opening', 'Opening title'], ['whole', 'Short label']];
  assert.deepEqual(topTextAnswers, [
    [...answers[0], 'Nothing at the top'],
    [...answers[1], 'A title for the first few seconds, in your title look'],
    [...answers[2], 'A label like "Security rant" for the first few seconds'],
  ]);
  assert.deepEqual(homeAnswers, answers);
  for (const [hook_style, expected] of [['none', 'None'], ['quiet_serif', 'Opening title'], ['text_hook', 'Short label']]) {
    assert.equal(topTextSummary({hook_answered: true, hook_style}), expected);
  }
});

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

test('horizontal offers a chosen opening and refuses the short label', () => {
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
