import test from 'node:test';
import assert from 'node:assert/strict';
import * as outlineView from '../../src/yap_studio/review/static/outline-view.mjs';
import {
  roleLabel, statusText, durationText, noteCountText, waitChoices, clampWords,
} from '../../src/yap_studio/review/static/outline-view.mjs';

test('roleLabel follows ordered roles including a single opening', () => {
  assert.equal(roleLabel({role: 'opening'}, 0, 1), 'Opening');
  assert.equal(roleLabel({role: 'opening'}, 0, 4), 'Opening');
  assert.equal(roleLabel({role: 'beat'}, 1, 4), 'Beat 2');
  assert.equal(roleLabel({role: 'beat'}, 2, 4), 'Beat 3');
  assert.equal(roleLabel({role: 'ending'}, 3, 4), 'Ending');
  assert.equal(roleLabel({}, 1, 3), 'Beat 2');
  assert.equal(roleLabel({}, 2, 3), 'Ending');
});

test('statusText describes every gate and formats only a waiting deadline', () => {
  const format = value => {
    assert.equal(value, '2026-10-06T20:40:00Z');
    return '3:40 PM';
  };
  assert.equal(statusText({state: 'waiting'}, format), 'Waiting for your OK on the story.');
  assert.equal(statusText({state: 'waiting', deadline: '2026-10-06T20:40:00Z'}, format),
    'Rendering starts at 3:40 PM unless you send notes or approve.');
  const messages = {
    missing: 'No story outline yet.',
    notes_sent: 'Notes sent. The agent is revising the story.',
    approved: 'Approved. Rendering can start.',
    timed_out: 'The wait ran out, so rendering can start.',
    stale: 'The story changed. The agent is updating the outline.',
  };
  for (const [state, message] of Object.entries(messages)) {
    assert.equal(statusText({state, deadline: 'unused'}, () => assert.fail('unexpected format')), message);
  }
});

test('durationText shows seconds without trailing zeroes', () => {
  assert.equal(durationText(0), '0 seconds');
  assert.equal(durationText(1), '1 second');
  assert.equal(durationText(1.47), '1.5 seconds');
  assert.equal(durationText(120), '120 seconds');
  for (const value of [null, undefined, NaN, Infinity, -1, '5']) {
    assert.equal(durationText(value), 'Length unavailable');
  }
});

test('noteCountText counts drafts only and handles singular and empty lists', () => {
  assert.equal(noteCountText([]), 'Send 0 notes');
  assert.equal(noteCountText([{batch: null}, {batch: 'b1'}]), 'Send 1 note');
  assert.equal(noteCountText([{batch: null}, {batch: null}, {batch: 'b1'}]), 'Send 2 notes');
});

test('waitChoices has the five labels and minute values without shared mutations', () => {
  assert.deepEqual(waitChoices(), [
    {label: 'Not set', minutes: null},
    {label: '15 minutes', minutes: 15},
    {label: '1 hour', minutes: 60},
    {label: '4 hours', minutes: 240},
    {label: '1 day', minutes: 1440},
  ]);
  const choices = waitChoices();
  choices[0].minutes = 5;
  assert.equal(waitChoices()[0].minutes, null);
});

test('waitOption selects server minutes or a local deadline while a wait runs', () => {
  const deadline = '2026-10-06T20:40:00Z';
  const format = value => {
    assert.equal(value, deadline);
    return '3:40 PM';
  };
  assert.deepEqual(outlineView.waitOption?.({deadline, wait_minutes: 60}, format),
    {value: '60', label: '1 hour', minutes: 60});
  for (const choice of waitChoices().filter(item => item.minutes !== null)) {
    assert.deepEqual(outlineView.waitOption({deadline, wait_minutes: choice.minutes}, format),
      {...choice, value: String(choice.minutes)});
  }
  for (const minutes of [5, null, undefined, '60']) {
    assert.deepEqual(outlineView.waitOption({deadline, wait_minutes: minutes}, format),
      {value: 'deadline', label: 'Until 3:40 PM', minutes: null});
  }
  assert.deepEqual(outlineView.waitOption({deadline: null, wait_minutes: 60}, format),
    {value: 'none', label: 'Not set', minutes: null});
  assert.deepEqual(outlineView.waitOption({}, format),
    {value: 'none', label: 'Not set', minutes: null});
});

test('clampWords preserves short text and safely truncates at word boundaries', () => {
  assert.equal(clampWords('  one\n two  ', 2), '  one\n two  ');
  assert.equal(clampWords('one two three four', 2), 'one two...');
  assert.equal(clampWords('one\n two\tthree', 2), 'one two...');
  assert.equal(clampWords('', 5), '');
  assert.equal(clampWords('<script> text & more', 2), '<script> text...');
  for (const limit of [0, -1, 1.5, NaN]) assert.equal(clampWords('one', limit), '');
});
