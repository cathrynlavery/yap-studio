import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

import {
  answersPayload, answersSummary, brollText, defaultRoute, draftFromDefault, draftProblems,
  findProject, firstOpenStep, formatBytes, formatDuration, formatWhen, isVideoFile, linkFor, parseRoute, phaseInfo,
  progressRows, requestInfo, sourcesToSend, takesText, typeLabel,
} from '../../src/yap_studio/homepage/static/home-model.mjs';

const inbox = {takes: [
  {name: 'IMG_0002.mp4', kind: 'take'}, {name: 'memo.m4a', kind: 'memo'},
]};

test('labels read as plain words', () => {
  assert.equal(typeLabel('talking_head'), 'Talking head');
  assert.equal(typeLabel('green_screen'), 'Video');
  assert.equal(brollText(['folder']), 'your inbox or a folder');
  assert.equal(brollText(['folder', 'album', 'stock']),
    'your inbox or a folder, a Photos album and stock clips you approve');
  assert.equal(brollText([]), 'none');
  assert.equal(brollText(null), 'not chosen yet');
  assert.equal(formatDuration(27.4), '0:27');
  assert.equal(formatDuration(3723), '1:02:03');
  assert.equal(formatDuration(Number.NaN), '');
  assert.equal(formatBytes(46_200_000), '46 MB');
  assert.equal(formatBytes(1_250_000_000), '1.3 GB');
  assert.equal(takesText([]), 'none yet');
  assert.equal(takesText(['a.mp4']), 'a.mp4');
  assert.equal(takesText(['a.mp4', 'b.mp4']), '2 takes');
  const now = new Date(2026, 9, 19, 12, 0);
  assert.match(formatWhen(new Date(2026, 9, 19, 9, 41).toISOString(), now), /^today /);
  assert.match(formatWhen(new Date(2026, 9, 18, 9, 41).toISOString(), now), /^yesterday /);
  assert.equal(formatWhen('not a date', now), '');
});

test('a talking head asks whether there is anything on top before asking where', () => {
  assert.equal(firstOpenStep(draftProblems({}, {inbox})), 'what');
  const unasked = {video_type: 'talking_head', has_broll: null, broll_sources: [], takes: ['IMG_0002.mp4']};
  assert.match(draftProblems(unasked, {inbox}).what, /anything to show on top/);
  const none = {...unasked, has_broll: false};
  assert.deepEqual(draftProblems(none, {inbox}), {});
  assert.deepEqual(sourcesToSend({...none, broll_sources: ['folder']}), [], 'no means no sources at all');
  const yes = {...unasked, has_broll: true};
  assert.match(draftProblems(yes, {inbox}).what, /where those clips/);
  assert.deepEqual(draftProblems({...yes, broll_sources: ['album']}, {inbox}), {});
  assert.deepEqual(sourcesToSend({...yes, broll_sources: ['album', 'none']}), ['album']);
});

test('a draft needs a take for a talking head and pictures for a voiceover', () => {
  const talking = {video_type: 'talking_head', has_broll: false, broll_sources: [], takes: []};
  assert.deepEqual(Object.keys(draftProblems(talking, {inbox})), ['footage']);
  assert.equal(firstOpenStep(draftProblems(talking, {inbox})), 'footage');
  assert.deepEqual(draftProblems({...talking, takes: ['IMG_0002.mp4']}, {inbox}), {});
  assert.ok(draftProblems({...talking, takes: ['memo.m4a']}, {inbox}).footage, 'a memo is not a take');
  assert.ok(draftProblems({...talking, takes: ['gone.mp4']}, {inbox}).footage, 'a used take no longer counts');
  const voice = {video_type: 'voiceover', broll_sources: ['stock'], takes: []};
  assert.match(draftProblems(voice, {inbox}).what, /photos and clips/);
  assert.deepEqual(draftProblems({...voice, broll_sources: ['album']}, {inbox}), {});
  assert.ok(draftProblems({video_type: 'green_screen', broll_sources: []}, {inbox}).what);
});

test('a returning creator starts from their saved answers, a new one from nothing', () => {
  assert.deepEqual(draftFromDefault(null), {video_type: null, has_broll: null, broll_sources: [], takes: null, album: ''});
  assert.equal(draftFromDefault({video_type: 'talking_head', broll_sources: []}).has_broll, false);
  assert.equal(draftFromDefault({video_type: 'talking_head', broll_sources: ['folder']}).has_broll, true);
  assert.equal(draftFromDefault({video_type: 'voiceover', broll_sources: ['album']}).has_broll, null);
  assert.equal(draftFromDefault({video_type: 'green_screen', broll_sources: []}).video_type, null);
});

test('progress follows the project phase and never claims more than it knows', () => {
  const statuses = (phase, extra = {}) => progressRows({phase, versions: [], stages: [], ...extra})
    .map((row) => row.status);
  assert.deepEqual(statuses('starting'), ['now', 'next', 'next', 'next', 'next', 'next', 'next']);
  assert.deepEqual(statuses('transcribing').slice(0, 3), ['done', 'now', 'next']);
  assert.deepEqual(statuses('story').slice(0, 4), ['done', 'done', 'now', 'next']);
  assert.deepEqual(statuses('cut_ready').slice(2, 5), ['done', 'turn', 'next']);
  assert.deepEqual(statuses('revising').slice(2, 4), ['now', 'next']);
  assert.deepEqual(statuses('revising', {versions: [{final: true}]}).slice(3, 6), ['done', 'now', 'next']);
  assert.deepEqual(statuses('final_ready').slice(4, 7), ['done', 'turn', 'next']);
  assert.ok(statuses('approved').every((status) => status === 'done'));
  const stopped = progressRows({phase: 'stopped', message: 'Captions cover the face.', versions: [],
    stages: [{name: 'captions', state: 'done'}]});
  assert.equal(stopped[4].status, 'stopped');
  assert.equal(stopped[4].sub, 'Captions cover the face.');
  assert.equal(progressRows({phase: 'building', step: 'Rendering cards', versions: [{final: false}], stages: []})[4].sub,
    'Rendering cards');
  assert.equal(phaseInfo('nonsense').label, 'Starting');
  assert.equal(phaseInfo('cut_ready').tone, 'turn');
});

test('request copy depends on whether Claude is listening', () => {
  assert.equal(requestInfo({state: 'pending'}, true).tone, 'working');
  assert.equal(requestInfo({state: 'pending'}, false).title, 'Waiting for Claude Code.');
  assert.equal(requestInfo({state: 'claimed', ack: {message: 'Copying IMG_0002.mp4'}}, true).lede,
    'Copying IMG_0002.mp4');
  assert.equal(requestInfo({state: 'failed', ack: null}, true).tone, 'stopped');
});

test('a stalled request keeps its recovery buttons even when its project exists', async () => {
  const request = {id: 'r1', kind: 'edit', state: 'claimed', stalled: true,
    ack: {project: 'projects/demo', message: 'Story saved.'}};
  const state = {listening: false, requests: [request], projects: [{dir: 'projects/demo', name: 'demo'}]};
  assert.equal(defaultRoute(state, true), '#request/r1');
  assert.equal(requestInfo(request, false).title, 'Claude Code stopped before finishing.');
  const source = readFileSync(new URL('../../src/yap_studio/homepage/static/home.js', import.meta.url), 'utf8');
  const renderSource = source.slice(source.indexOf('function renderRequest('), source.indexOf('function finalRequested('));
  const h = (tag, attrs, ...children) => ({tag, ...attrs, children: children.flat(Infinity).filter(Boolean)});
  const sent = [], redirects = [];
  let refreshes = 0;
  const render = new Function('state', 'h', 'requestInfo', 'typeLabel', 'takesText', 'api', 'toast',
    'refresh', 'queueMicrotask', `${renderSource}; return renderRequest;`)(state, h, requestInfo,
      typeLabel, takesText, async (url, data) => sent.push([url, data]), () => {},
      async () => { refreshes++; }, (fn) => redirects.push(fn));
  const card = render({id: request.id});
  const nodes = (node) => [node, ...(node.children ?? []).flatMap(nodes)];
  assert.equal(redirects.length, 0);
  assert.ok(nodes(card).some((node) => node.text === 'Claude Code stopped before finishing.'));
  assert.equal(nodes(card).some((node) => node.tag === 'ol'), false);
  const videoSource = source.slice(source.indexOf('function renderVideo('), source.indexOf('function versionNotes('));
  const renderVideo = new Function('state', 'findProject', 'shown', 'renderRequest', 'h',
    `${videoSource}; return renderVideo;`)(state, findProject, (project) => project, render, h);
  assert.deepEqual(nodes(renderVideo({name: 'demo'})).map((node) => node.text),
    nodes(card).map((node) => node.text));
  const buttons = nodes(card).filter((node) => node.tag === 'button');
  assert.deepEqual(buttons.map((node) => node.text), ['Send again', 'Cancel']);
  for (const button of buttons) await button.onclick();
  assert.deepEqual(sent, [['/api/requests/r1/resume', {}], ['/api/requests/r1/cancel', {}]]);
  assert.equal(refreshes, 2);
  request.stalled = false;
  render({id: request.id});
  assert.equal(redirects.length, 1);
});

test('a visit lands on what is happening now', () => {
  assert.equal(defaultRoute({first_run: true, projects: [], requests: []}, false), '#welcome');
  assert.equal(defaultRoute({first_run: true, projects: [], requests: []}, true), '#new/what');
  assert.equal(defaultRoute({projects: [], requests: [{id: 'r2', state: 'delivered', ack: null}]}, true),
    '#request/r2');
  assert.equal(defaultRoute({projects: [{name: 'a', phase: 'approved'}, {name: 'b', phase: 'story'}], requests: []}, true),
    '#video/b');
  assert.equal(defaultRoute({projects: [{name: 'a', phase: 'approved'}], requests: []}, true), '#videos');
  assert.deepEqual(parseRoute('#new/look'), {view: 'new', step: 'look'});
  assert.deepEqual(parseRoute('#new/review'), {view: 'default'});
  assert.deepEqual(parseRoute('#video/2026-10-19%20a'), {view: 'video', name: '2026-10-19 a'});
  assert.deepEqual(parseRoute(''), {view: 'default'});
});

test('a video can be found by its current or earlier name', () => {
  const renamed = {name: 'tokens', renamed_from: ['raw', 'draft']};
  const legacy = {name: 'older'};
  const projects = [renamed, legacy];
  assert.equal(findProject(projects, 'tokens'), renamed);
  assert.equal(findProject(projects, 'raw'), renamed);
  assert.equal(findProject(projects, 'draft'), renamed);
  assert.equal(findProject(projects, 'older'), legacy);
  assert.equal(findProject(projects, 'unknown'), undefined);
  assert.equal(findProject([], 'raw'), undefined);
  assert.equal(findProject([renamed, {name: 'raw'}], 'raw').name, 'raw');
});

test('the default video route encodes its project name', () => {
  assert.equal(defaultRoute({projects: [{name: 'a video', phase: 'story'}], requests: []}, true),
    '#video/a%20video');
});

test('links match the device the page was opened on', () => {
  const urls = {url: 'http://127.0.0.1:1/?t=a', phone_url: 'http://192.168.1.9:1/?t=a'};
  assert.equal(linkFor(urls, '127.0.0.1'), urls.url);
  assert.equal(linkFor(urls, 'localhost'), urls.url);
  assert.equal(linkFor(urls, '192.168.1.9'), urls.phone_url);
  assert.equal(linkFor({url: urls.url, phone_url: null}, '192.168.1.9'), urls.url);
  assert.equal(linkFor(null, '127.0.0.1'), null);
});

test('answers become the shape the engine validates', () => {
  const answers = answersPayload({platforms: ['tiktok', 'reels'], cuts: 'natural', music: 'none',
    never: ['cover_face', 'emoji'], other: 'No swearing\n\n', about: ' Short  videos\nabout  filming ',
    website: '', fixes: 'yap studio -> Yap Studio\nbad line\nk8s = Kubernetes'});
  assert.deepEqual(answers.platforms, ['tiktok', 'reels']);
  assert.deepEqual(answers.never, {cover_face: true, emoji: true, all_caps: false, other: ['No swearing']});
  assert.equal(answers.about, 'Short videos about filming');
  assert.deepEqual(answers.brand, {website: null, pick_in_gallery: true});
  assert.deepEqual(answers.word_fixes, {'yap studio': 'Yap Studio', k8s: 'Kubernetes'});
  assert.deepEqual(answers.finished_videos, {answer: 'not_yet'});
  assert.deepEqual(answersPayload({website: 'example.com'}).brand, {website: 'example.com', pick_in_gallery: false});
  const summary = answersSummary([
    {answer_key: 'platforms', current: ['tiktok'], choices: [{value: 'tiktok', label: 'TikTok'}]},
    {answer_key: 'cuts', current: 'tight', choices: [{value: 'tight', label: 'Tight'}]},
    {answer_key: 'music', current: 'none', choices: [{value: 'none', label: 'None'}]},
  ]);
  assert.equal(summary, 'TikTok · Tight cuts · no music');
});

test('a voiceover inbox takes memos, not videos', () => {
  assert.equal(isVideoFile('scene.MP4'), true);
  assert.equal(isVideoFile('clip', 'video/quicktime'), true);
  assert.equal(isVideoFile('memo.m4a', 'audio/x-m4a'), false);
  assert.equal(isVideoFile('voice.wav'), false);
});
