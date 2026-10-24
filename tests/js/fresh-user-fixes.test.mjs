import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import * as home from '../../src/yap_studio/homepage/static/home-model.mjs';

const source = readFileSync(new URL('../../src/yap_studio/homepage/static/home.js', import.meta.url), 'utf8');
const h = (tag, attrs, ...children) => ({tag, ...attrs, children: children.flat(Infinity).filter(Boolean)});
const nodes = (node) => [node, ...(node.children ?? []).flatMap(nodes)];

test('a fresh tab and a request route follow a claimed final to its project', () => {
  const request = {id: 'r2', kind: 'final', state: 'claimed', ack: {project: null},
    payload: {project: 'projects/demo'}};
  const state = {requests: [request], projects: [{name: 'demo', dir: 'projects/demo', phase: 'building'}]};
  assert.equal(home.defaultRoute(state, true), '#video/demo');
  const code = source.slice(source.indexOf('function renderRequest('), source.indexOf('function finalRequested('));
  const redirects = [];
  const render = new Function('agentName', 'state', 'h', 'requestInfo', 'typeLabel', 'takesText', 'queueMicrotask',
    `${code}; return renderRequest;`)(home.agentName, state, h, home.requestInfo, home.typeLabel, home.takesText,
      (fn) => redirects.push(fn));
  render({id: 'r2'});
  assert.equal(redirects.length, 1);
});

for (const status of ['failed', 'cancelled']) {
  for (const project of [null, 'projects/demo']) {
    test(`${status} final keeps its request details with ack project ${project}`, () => {
      const request = {id: 'r2', kind: 'final', state: status, ack: {project, message: 'Render failed'},
        payload: {project: 'projects/demo'}};
      const state = {requests: [request], projects: [{name: 'demo', dir: 'projects/demo'}]};
      const code = source.slice(source.indexOf('function renderRequest('), source.indexOf('function finalRequested('));
      const redirects = [];
      const render = new Function('agentName', 'state', 'h', 'requestInfo', 'typeLabel', 'takesText', 'queueMicrotask',
        `${code}; return renderRequest;`)(home.agentName, state, h, home.requestInfo, home.typeLabel, home.takesText,
          (fn) => redirects.push(fn));
      const tree = nodes(render({id: 'r2'}));
      assert.equal(redirects.length, 0);
      assert.ok(tree.some((node) => node.text === 'Start again' && node.href === '#new/what'));
      assert.ok(tree.some((node) => node.text === home.requestInfo(request, false).lede));
    });
  }
}

test('make and building copy promise only the chosen look', () => {
  const state = {style: {saved: true, name: 'Just you'}, listening: true};
  const code = source.slice(source.indexOf('function stepMake('), source.indexOf('// ---------- Request'));
  const make = new Function('agentName', 'state', 'h', 'typeLabel', 'brollText', 'sourcesToSend', 'draftTakes',
    'stepHeader', 'busy', 'shapeProblem', 'topTextChoice', 'topTextAnswers',
    `${code}; return stepMake;`)(home.agentName, state, h, home.typeLabel, home.brollText,
      home.sourcesToSend, () => ['take.mp4'], () => [], false, home.shapeProblem,
      home.topTextChoice, home.topTextAnswers);
  const text = nodes(make({video_type: 'talking_head', has_broll: false}, {}))
    .filter((node) => node.tag === 'li').map((node) => node.text).join(' ');
  assert.match(text, /chosen look/);
  assert.doesNotMatch(text, /adds captions|cards|B-roll|music/);
  for (const phase of ['building', 'approved']) {
    const build = home.progressRows({phase}).find((row) => row.key === 'build');
    assert.doesNotMatch(build.label, /captions|cards|music/i);
  }
  assert.doesNotMatch(source, /Captions, titles, cards, B-roll and music go on/);
});
