import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as helpers from '../../src/yap_studio/review/static/voice-fixes.mjs';
import {versionUrl} from '../../src/yap_studio/review/static/timeline.mjs';

const fix = Object.freeze({id: 'f1', time: '3:12', said: '<coal flunge>', say: 'cold plunge',
  generation: 'generation1',
  before_url: 'voice-fix-audio?fix=f1&clip=before&generation=generation1',
  after_url: 'voice-fix-audio?fix=f1&clip=after&generation=generation1',
  checks: ['Join: clean.', 'Heard as: cold plunge.'], credits: 11,
  credits_source: 'charged', cost_note: '', status: 'ready', decision: null});

test('plain voice text includes output time, cost and saved decision', () => {
  assert.equal(helpers.voiceSentence(fix),
    "At 3:12 you said '<coal flunge>'. Fixed: 'cold plunge'.");
  assert.equal(helpers.voiceSentence({...fix, time: null}),
    "You said '<coal flunge>'. Fixed: 'cold plunge'.");
  assert.equal(helpers.voiceCost(fix), 'Cost: 11 ElevenLabs credits (charged).');
  assert.equal(helpers.voiceCost({...fix, credits: 5.5, credits_source: 'estimated'}),
    'Cost: 5.5 ElevenLabs credits (estimated).');
  assert.equal(helpers.voiceState(fix), 'Waiting for your choice.');
  assert.equal(helpers.voiceState({...fix, status: 'approved'}), 'Use the fix.');
  assert.equal(helpers.voiceState({...fix, status: 'rejected'}), 'Keep my original.');
});

test('ready fixes offer both choices and decisions offer the other choice', () => {
  const approve = {action: 'approve', label: 'Use the fix'};
  const reject = {action: 'reject', label: 'Keep my original'};
  assert.deepEqual(helpers.voiceActions(fix), [approve, reject]);
  assert.deepEqual(helpers.voiceActions({...fix, status: 'approved'}), [reject]);
  assert.deepEqual(helpers.voiceActions({...fix, status: 'rejected'}), [approve]);
  assert.deepEqual(helpers.voiceDecisionBody(fix, 'approve'),
    {fix: 'f1', action: 'approve', generation: 'generation1'});
});

test('clip URLs keep the review path and add its token', () => {
  assert.equal(helpers.voiceClipUrl(fix.before_url, 'a b'),
    'voice-fix-audio?fix=f1&clip=before&generation=generation1&t=a+b');
  assert.equal(helpers.voiceClipUrl(fix.after_url, ''), fix.after_url);
});

test('recorded and clone sources have plain labels and costs', () => {
  const own = {...fix, kind: 'own', from_time: '3:28', credits: 0};
  assert.equal(helpers.voiceSource(own), 'From your own recording at 3:28');
  assert.equal(helpers.voiceSource({...fix, kind: 'clone'}), 'Re-said by your ElevenLabs voice');
  assert.equal(helpers.voiceCost(own), 'Cost: 0 credits.');
  const {page, review, card} = pageFixture();
  review.series.voice_fixes = [own];
  page.renderVoiceFixes();
  assert.ok(card().querySelectorAll('p').some((node) =>
    node.textContent === 'From your own recording at 3:28'));
});

function pageFixture() {
  let active;
  class Node {
    constructor(tag = '') {
      Object.assign(this, {tag, children: [], dataset: {}, attributes: {}, listeners: {},
        textContent: '', hidden: false});
    }
    append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener(key, value) { this.listeners[key] = value; }
    querySelectorAll(selector) {
      return this.children.flatMap((node) => [
        ...(selector.split(',').some((part) => part.trim().startsWith('.')
          ? node.className?.split(' ').includes(part.trim().slice(1))
          : node.tag === part.trim()) ? [node] : []), ...node.querySelectorAll(selector),
      ]);
    }
    focus() { active = this; }
    contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
  }
  const el = Object.fromEntries(['voiceFixes', 'voiceFixList', 'voiceFixError']
    .map((name) => [name, new Node()]));
  const review = {media_changed: false, series: {project: true, voice_fixes: [structuredClone(fix)]}};
  const state = {failure: null, gate: null, requests: []};
  const context = vm.createContext({...helpers, versionUrl, selectedVersion: 2,
    el, review, token: 'token', offline: false,
    switchingVersion: false, loadVersion: 0, voiceFixSaving: new Set(), voiceFixErrors: new Map(),
    tasteSaving: new Set(), releaseBusy: false, roundBusy: false,
    voiceFixRendered: null,
    document: {createElement: (tag) => new Node(tag), get activeElement() { return active; }},
    renderVersions() {}, updateReplyEvidenceButtons() {}, renderSendBar() {},
    makeButton(text, className, handler) {
      const button = new Node('button');
      Object.assign(button, {textContent: text, className, click: handler});
      return button;
    },
    errorMessage: (error) => error.message,
    api: async (path, options) => {
      state.requests.push({path, options});
      if (state.gate) await state.gate;
      if (state.failure) throw new Error(state.failure);
      const saved = {...fix, status: options.body.action === 'approve' ? 'approved' : 'rejected',
        decision: {action: options.body.action, via: 'review'}};
      return {voice_fixes: [saved]};
    },
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  const code = source.slice(source.indexOf('function updateVoiceFixControls()'),
    source.indexOf('function renderRemoved()'));
  vm.runInContext(`${code}\nglobalThis.page = {renderVoiceFixes, saveVoiceFixDecision};`, context);
  context.page.renderVoiceFixes();
  return {context, page: context.page, el, review, state,
    card: () => el.voiceFixList.children[0], active: () => active};
}

test('page renders safe text, labelled lazy audio and preserves players during refresh', () => {
  const {page, el, card} = pageFixture();
  assert.equal(el.voiceFixes.hidden, false);
  assert.equal(card().querySelectorAll('p')[0].textContent, helpers.voiceSentence(fix));
  const players = card().querySelectorAll('audio');
  assert.equal(players.length, 2);
  for (const [i, player] of players.entries()) {
    assert.equal(player.controls, true);
    assert.equal(player.preload, 'none');
    assert.match(player.attributes['aria-label'], new RegExp(`${i ? 'After' : 'Before'}.*3:12`, 'u'));
    assert.match(player.src, /&t=token$/u);
  }
  page.renderVoiceFixes();
  assert.equal(card().querySelectorAll('audio')[0], players[0]);
});

test('page saves both choices, keeps keyboard focus and shows the saved choice', async () => {
  const {page, card, state, active} = pageFixture();
  card().querySelectorAll('button')[0].focus();
  await page.saveVoiceFixDecision(fix, 'approve');
  assert.equal(state.requests[0].path, 'api/voice-fixes?version=2');
  assert.deepEqual(JSON.parse(JSON.stringify(state.requests[0].options.body)),
    {fix: 'f1', action: 'approve', generation: 'generation1'});
  assert.equal(card().querySelectorAll('button').length, 1);
  assert.equal(active().textContent, 'Keep my original');
  assert.ok(card().querySelectorAll('p').some((node) => node.textContent === 'Use the fix.'));
  await page.saveVoiceFixDecision({...fix, status: 'approved'}, 'reject');
  assert.equal(active().textContent, 'Use the fix');
});

test('a decision stays on the older version playing in the page', async () => {
  const {context, page, state} = pageFixture();
  context.selectedVersion = 1;
  await page.saveVoiceFixDecision(fix, 'approve');
  assert.equal(state.requests[0].path, 'api/voice-fixes?version=1');
});

test('a new generation replaces buffered players and sends its own fingerprint', async () => {
  const {page, card, review, state} = pageFixture();
  const players = card().querySelectorAll('audio');
  const fresh = {...fix, generation: 'generation2',
    before_url: fix.before_url.replace('generation1', 'generation2'),
    after_url: fix.after_url.replace('generation1', 'generation2')};
  review.series.voice_fixes = [fresh];
  page.renderVoiceFixes();
  for (const [i, player] of card().querySelectorAll('audio').entries()) {
    assert.notEqual(player, players[i]);
    assert.match(player.src, /generation=generation2&/u);
  }
  await card().querySelectorAll('button')[0].click();
  assert.equal(state.requests[0].options.body.generation, 'generation2');
});

test('a choice from a replaced card cannot approve the new generation', async () => {
  const {page, card, review, state} = pageFixture();
  const oldButton = card().querySelectorAll('button')[0];
  review.series.voice_fixes = [{...fix, generation: 'generation2'}];
  page.renderVoiceFixes();
  await oldButton.click();
  assert.equal(state.requests.length, 0);
});

test('page blocks double saves, shows refusal and allows retry', async () => {
  const {page, card, state} = pageFixture();
  let finish;
  state.gate = new Promise((resolve) => { finish = resolve; });
  state.failure = 'Consent was withdrawn.';
  const pending = page.saveVoiceFixDecision(fix, 'approve');
  assert.equal(card().attributes['aria-busy'], 'true');
  assert.ok(card().querySelectorAll('button').every((node) => node.disabled));
  await page.saveVoiceFixDecision(fix, 'approve');
  assert.equal(state.requests.length, 1);
  finish();
  await pending;
  assert.ok(card().querySelectorAll('p').some((node) => node.attributes.role === 'alert'
    && node.textContent === state.failure));
  state.failure = null;
  await page.saveVoiceFixDecision(fix, 'approve');
  assert.equal(card().querySelectorAll('button')[0].disabled, false);
});

test('page hides empty projects and blocks offline or changed media', async () => {
  const {context, page, el, card, review, state} = pageFixture();
  for (const reason of ['offline', 'changed']) {
    context.offline = reason === 'offline';
    review.media_changed = reason === 'changed';
    page.renderVoiceFixes();
    assert.ok(card().querySelectorAll('button').every((node) => node.disabled));
    await page.saveVoiceFixDecision(fix, 'approve');
  }
  assert.equal(state.requests.length, 0);
  review.series.voice_fixes = [];
  page.renderVoiceFixes();
  assert.equal(el.voiceFixes.hidden, true);
  assert.equal(el.voiceFixList.children.length, 0);
});
