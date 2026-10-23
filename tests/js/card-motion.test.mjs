import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../../src/yap_studio/templates/card-motion-host.js', import.meta.url), 'utf8');

test('registry copies settle inside the item hold before its own fade', async () => {
  for (const [entrance, exit, expected] of [[1.4, 1.5, 1.45], [0.6, 1.6, 1.1], [1.1, 1.1, 1.1]]) {
    const seeks = [], order = [];
    const tween = (start, end, opacity) => ({vars: {opacity}, startTime: () => start, endTime: () => end});
    const inner = {getChildren: () => [tween(0, entrance, 1), tween(exit, 2, 0)],
      seek: time => { seeks.push(time); order.push('seek'); }};
    const timeline = {fromTo() {}, to() {}, getChildren: () => [], seek() {}};
    const window = {__cardMotion: {duration: 2, offset: 1100, enter: false, inner: 'sample', settle: 1.66},
      __cardMotionTimeline: timeline, __timelines: {sample: inner}};
    vm.runInNewContext(source, {window, document: {fonts: {ready: Promise.resolve()},
      getElementById: () => ({previousElementSibling: null, querySelectorAll: () => [],
        getBoundingClientRect() {}})},
      attachEchoSmear: () => order.push('attach'), setTimeout});
    await new Promise(setImmediate);
    assert.equal(seeks[0], expected);
    assert.equal(seeks[1], 0);
    assert.deepEqual(order, ['seek', 'attach', 'seek']);
    assert.equal(window.__timelines['hf-host'], timeline);
  }
});

test('attachment waits for fonts requested by the inner card layout', async () => {
  let loaded, laidOut = false, attached = false;
  const pending = new Promise(resolve => { loaded = resolve; });
  const inner = {getChildren: () => [], seek() {}};
  const timeline = {fromTo() {}, to() {}, getChildren: () => [], seek() {}};
  const window = {__cardMotion: {duration: 2, offset: 1100, enter: false, inner: 'sample', settle: 1.66},
    __cardMotionTimeline: timeline, __timelines: {sample: inner}};
  const document = {fonts: {get ready() { return laidOut ? pending : Promise.resolve(); }},
    getElementById: () => ({previousElementSibling: null, querySelectorAll: () => [],
      getBoundingClientRect() { laidOut = true; }})};
  vm.runInNewContext(source, {window, document, setTimeout,
    attachEchoSmear: () => { attached = true; }});
  await new Promise(setImmediate);
  assert.equal(attached, false);
  loaded();
  await new Promise(setImmediate);
  assert.equal(attached, true);
});

test('image contents are decoded before the trail copies are frozen', async () => {
  let decoded, attached = false;
  const pending = new Promise(resolve => { decoded = resolve; });
  const inner = {getChildren: () => [], seek() {}};
  const timeline = {fromTo() {}, to() {}, getChildren: () => [], seek() {}};
  const window = {__cardMotion: {duration: 2, offset: 1100, enter: true, inner: 'sample', settle: 1.66},
    __cardMotionTimeline: timeline, __timelines: {sample: inner}};
  const document = {fonts: {ready: Promise.resolve()}, getElementById: () => ({
    previousElementSibling: null, getBoundingClientRect() {},
    querySelectorAll: () => [{decode: () => pending}],
  })};
  vm.runInNewContext(source, {window, document, setTimeout,
    attachEchoSmear: () => { attached = true; }});
  await new Promise(setImmediate);
  assert.equal(attached, false);
  decoded();
  await new Promise(setImmediate);
  assert.equal(attached, true);
});

test('quiet snapshot seeks paint counters and restore their entrance text', async () => {
  let position = 0, text = 0, copied;
  const counter = {vars: {onUpdate() { text = Math.round(position * 72 / 1.4); }},
    startTime: () => 0, endTime: () => 1.4};
  const inner = {getChildren: () => [counter], seek(time, quiet) {
    assert.equal(quiet, undefined); position = time;
  }};
  const timeline = {fromTo() {}, to() {}, getChildren: () => [], seek() {}};
  const window = {__cardMotion: {duration: 2, offset: 1100, enter: false, inner: 'sample', settle: 1.4},
    __cardMotionTimeline: timeline, __timelines: {sample: inner}};
  const document = {fonts: {ready: Promise.resolve()}, getElementById: () => ({
    previousElementSibling: null, getBoundingClientRect() {}, querySelectorAll: () => [],
  })};
  vm.runInNewContext(source, {window, document, setTimeout,
    attachEchoSmear: () => { copied = text; }});
  await new Promise(setImmediate);
  assert.equal(copied, 72);
  assert.equal(text, 0);
});

const drawn = readFileSync(new URL('../../src/yap_studio/templates/card-motion-drawn.js', import.meta.url), 'utf8');

function animation(start, duration, vars = {}, children = null) {
  return {vars, completed: false, stopped: false,
    startTime(value) { if (value !== undefined) start = value; return start; },
    totalDuration: () => duration, endTime: () => start + duration,
    globalTime: time => start + time,
    timeScale: () => 1,
    paused(value) { this.stopped = value; return this; },
    totalProgress(value) { this.completed = value === 1; return this; },
    ...(children ? {getChildren: () => children} : {}),
  };
}

test('streak finishes all entrance reveals and retains tagged and nested card actions', () => {
  const early = animation(0, .3), staggered = animation(.54, .5);
  const earlyCue = animation(.1, .3, {data: 'cue'});
  const scaledEarly = animation(0, .3);
  scaledEarly.timeScale = () => 2;
  const nestedEarly = animation(0, .2), nestedLate = animation(.6, .3, {data: 'cue'});
  const nested = animation(.1, 1, {}, [nestedEarly, nestedLate]);
  const middleFade = animation(1, .2, {opacity: 0, data: 'cue'});
  const fade = animation(2.7, .3, {opacity: 0});
  const alphaFade = animation(2.8, .2, {autoAlpha: 0});
  let children = [early, scaledEarly, nested, staggered, earlyCue, middleFade, fade, alphaFade];
  const tl = {getChildren: () => children, seek() {},
    remove(child) { children = children.filter(item => item !== child); },
    clear() { children = []; }, to(target, vars) { children.push(animation(0, vars.duration, vars)); }};
  vm.runInNewContext(drawn + '\ncardMotionDrawn(root, tl, duration);',
    {root: {dataset: {cardMotion: 'streak'}}, tl, duration: 3});
  assert.ok(children.includes(early));
  assert.equal(early.startTime(), -.3);
  assert.equal(early.completed, true);
  assert.equal(scaledEarly.completed, true);
  assert.ok(scaledEarly.endTime() <= 0);
  assert.equal(nested.startTime(), .1);
  assert.equal(nestedEarly.completed, true);
  assert.ok(nestedEarly.endTime() <= -.1);
  assert.equal(nestedLate.startTime(), .6);
  assert.equal(nestedLate.completed, false);
  assert.equal(staggered.startTime(), -.5);
  assert.equal(staggered.completed, true);
  assert.equal(earlyCue.startTime(), .1);
  assert.equal(earlyCue.completed, false);
  assert.ok(children.includes(middleFade));
  assert.ok(!children.includes(fade));
  assert.ok(!children.includes(alphaFade));
});

test('the shared reveal script leaves an unmarked timeline untouched', () => {
  const tl = new Proxy({}, {get() { throw new Error('unmarked card timeline changed'); }});
  vm.runInNewContext(drawn + '\ncardMotionDrawn(root, tl, 3);', {root: {dataset: {}}, tl});
});

test('built-in entrance and exit copies use separate targets and card poses', async () => {
  const seeks = [], inner = {getChildren: () => [animation(.7, 1.1),
    animation(0, 2, {data: 'card-motion-window'})], seek: time => seeks.push(time),
    globalTime: () => 0};
  const timeline = {fromTo() {}, to() {}, getChildren: () => [], seek() {}};
  const window = {__cardMotion: {duration: 2, offset: 1100, enter: true, inner: 'sample', settle: 1.66},
    __cardMotionTimeline: timeline, __timelines: {sample: inner}};
  const mounts = Object.fromEntries(['hf-mount', 'card-motion-in'].map(id => [id,
    {id, previousElementSibling: null, querySelectorAll: () => [], getBoundingClientRect() {}}]));
  const targets = [], poses = [];
  vm.runInNewContext(source, {window, document: {fonts: {ready: Promise.resolve()},
    getElementById: id => mounts[id]}, attachEchoSmear(target) {
      targets.push(target.id); poses.push(seeks.at(-1));
    }, setTimeout});
  await new Promise(setImmediate);
  assert.deepEqual(targets, ['card-motion-in', 'hf-mount']);
  assert.equal(poses[0], 0);
  assert.ok(poses[1] > 1.8 && poses[1] < 2, JSON.stringify(poses));
  assert.equal(seeks.at(-1), 0);
});
