import assert from 'node:assert/strict';
import test from 'node:test';
import {fileDecisionText, listTokenFrom, nextUndecided} from '../../src/yap_studio/review/static/timeline.mjs';

const video = (number, state) => ({number, status: {state}});

test('next undecided video comes after the current one and wraps round', () => {
  const videos = [video(1, 'open'), video(2, 'approved'), video(3, 'open'), video(4, 'sent')];
  assert.equal(nextUndecided(videos, 1).number, 3);
  assert.equal(nextUndecided(videos, 3).number, 1);
  assert.equal(nextUndecided([video(1, 'approved'), video(2, 'open')], 2), null);
  assert.equal(nextUndecided(null, 1), null);
});

test('file decision text names the approval, sent fixes and an earlier approval', () => {
  assert.equal(fileDecisionText(null), '');
  assert.equal(fileDecisionText({approved: {reviewer: 'Sample Reviewer'}}), 'Approved by Sample Reviewer.');
  assert.equal(fileDecisionText({approved: {reviewer: null}}), 'Approved.');
  assert.equal(fileDecisionText({approved: null, notes_sent: {fixes: 2}}), 'Notes sent (2 fixes).');
  assert.equal(fileDecisionText({approved: null, notes_sent: {fixes: 1}}), 'Notes sent (1 fix).');
  assert.match(fileDecisionText({approved: null, notes_sent: null, previous_approved: true}),
    /earlier version of this file was approved/);
  for (const text of [fileDecisionText({approved: null, notes_sent: {fixes: 1}, previous_approved: true})]) {
    assert.doesNotMatch(text, /[\u2013\u2014]/u);
  }
});

test('the list token comes only from a well-formed fragment', () => {
  assert.equal(listTokenFrom('#list=abcdefghijklmnopqrstuvwxyz_-12'), 'abcdefghijklmnopqrstuvwxyz_-12');
  assert.equal(listTokenFrom('#list=short'), null);
  assert.equal(listTokenFrom('#list=<script>abcdefghijklmnopqrst'), null);
  assert.equal(listTokenFrom(''), null);
});
