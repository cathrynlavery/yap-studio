import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import * as model from '../../src/yap_studio/review/static/timeline.mjs';

test('approval names the version and points to the approved file on home', () => {
  const message = model.approvalConfirmation({number: 3, name: 'sample-v3.mp4'},
    {url: 'http://localhost:1234/?t=home', phone_url: 'https://mac.example/?t=home'}, 'mac.example');
  assert.equal(message.title, 'Approved: sample-v3.mp4');
  assert.match(message.text, /approved file on Yap Studio home/);
  assert.equal(message.url, 'https://mac.example/?t=home');
});

test('approval never points a phone to localhost', () => {
  for (const host of ['localhost', '127.0.0.1', '[::1]']) {
    const home = {url: `http://${host}:1234/?t=home`};
    assert.equal(model.approvalConfirmation({number: 1}, home, 'mac.example').url, null);
    assert.equal(model.approvalConfirmation({number: 1}, home, host).url, home.url);
  }
  assert.equal(model.approvalConfirmation({number: 1}, null, 'mac.example').url, null);
});

test('successful approve stays visible without another server request', async () => {
  const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
  const release = source.slice(source.indexOf('async function releaseVersion('), source.indexOf('async function requestFinal('));
  const nodes = Object.fromEntries(['approval', 'approvalTitle', 'approvalText', 'approvalHome', 'video']
    .map((name) => [name, {hidden: true, pause() {}, scrollIntoView() {}}]));
  const review = {series: {project: true}, home: {url: 'http://localhost:1234/'}};
  const calls = [];
  const run = new Function('el', 'review', 'api', 'loadReview', 'approvalConfirmation', `
    let offline = false, releaseBusy = false, switchingVersion = false, composeBusy = false,
      roundBusy = false, answerBusy = false, editState = null, selectedVersion = 3,
      approvedVersion = null, refreshTimer = null, loadVersion = 0;
    const tasteSaving = new Set(), location = {hostname: 'localhost'};
    const confirm = () => true, renderVersions = () => {}, renderSendBar = () => {},
      showStatus = (text) => { throw new Error(text); }, errorMessage = (error) => error.message,
      seriesVersion = () => ({number: 3, name: 'sample-v3.mp4'}), clearInterval = () => {};
    ${release}
    return releaseVersion('approve');
  `);
  await run(nodes, review, async () => { calls.push('approve'); return {}; },
    async () => { calls.push('reload'); throw new Error('Server stopped'); }, model.approvalConfirmation);
  assert.deepEqual(calls, ['approve']);
  assert.equal(nodes.approval.hidden, false);
  assert.equal(nodes.approvalTitle.textContent, 'Approved: sample-v3.mp4');
  assert.equal(nodes.approvalHome.href, review.home.url);
});

for (const approvedOnLoad of [false, true]) {
  test(`approval controls agree on load and after click (approved on load: ${approvedOnLoad})`, async () => {
    const source = readFileSync(new URL('../../src/yap_studio/review/static/app.mjs', import.meta.url), 'utf8');
    const render = source.slice(source.indexOf('function renderVersions('), source.indexOf('async function releaseVersion('));
    const release = source.slice(source.indexOf('async function releaseVersion('), source.indexOf('async function requestFinal('));
    const node = () => ({hidden: true, pause() {}, scrollIntoView() {}, append() {},
      replaceChildren() {}, setAttribute() {}, removeAttribute(key) { delete this[key]; }});
    const nodes = Object.fromEntries(['versionBar', 'versionButtons', 'versionActions', 'rollback',
      'approve', 'makeFinal', 'approveHint', 'downloadVideo', 'downloadReport', 'compareToggle',
      'compareVersion', 'approval', 'approvalTitle', 'approvalText', 'approvalHome', 'video']
      .map((name) => [name, node()]));
    const review = {series: {project: true, current_version: 4, newest: 4,
      approved: approvedOnLoad ? {version: 3} : null,
      versions: [{number: 3, name: 'sample-v3.mp4'}, {number: 4, name: 'sample-v4.mp4'}]},
    home: {url: 'http://localhost:1234/'}};
    const calls = [];
    const run = new Function('el', 'review', 'api', 'model', 'node', `
      let offline = false, releaseBusy = false, switchingVersion = false, composeBusy = false,
        roundBusy = false, answerBusy = false, editState = null, selectedVersion = 3,
        approvedVersion = null, refreshTimer = null, loadVersion = 0,
        compareVersion = 4, compareOpen = false, mapGeneration = 0;
      const {reportStatusLabel, versionBadges, storyCutStep, versionUrl,
        defaultCompareVersion, approvalConfirmation} = model;
      const tasteSaving = new Set(), location = new URL('http://localhost/?t=token'), token = 'token';
      const document = {createElement: node}, history = {replaceState() {}};
      const confirm = () => true, renderSendBar = () => {}, updateFindingControls = () => {},
        makeButton = node, clearInterval = () => {},
        showStatus = (text) => { throw new Error(text); }, errorMessage = (error) => error.message,
        seriesVersion = (number) => review.series.versions.find((item) => item.number === number),
        loadReview = () => { throw new Error('Approval must not need another server request'); };
      ${render}
      ${release}
      renderVersions();
      return {approve: () => releaseVersion('approve'),
        select: (number) => { selectedVersion = number; renderVersions(); }};
    `);
    const page = run(nodes, review, async (path, options) => {
      calls.push([path, options.body.version]);
      return {};
    }, model, node);
    const assertApproved = (version) => {
      assert.equal(nodes.approveHint.textContent, 'This version is approved. Your download is ready.');
      assert.equal(nodes.approve.hidden, true);
      assert.equal(nodes.downloadVideo.hidden, false);
      assert.equal(nodes.downloadVideo.href, `api/download?version=${version}&t=token`);
      assert.equal(nodes.downloadReport.hidden, false);
    };
    if (!approvedOnLoad) {
      assert.equal(nodes.approve.hidden, false);
      assert.equal(nodes.approveHint.textContent, 'Happy with it? Approving unlocks the download.');
      await page.approve();
    }
    assertApproved(3);
    page.select(4);
    assert.equal(nodes.approve.hidden, false);
    assert.equal(nodes.approve.disabled, false);
    assert.equal(nodes.downloadVideo.hidden, true);
    assert.equal(nodes.approveHint.textContent, 'Happy with it? Approving unlocks the download.');
    await page.approve();
    assertApproved(4);
    page.select(3);
    assert.equal(nodes.approve.hidden, false);
    assert.deepEqual(calls, approvedOnLoad ? [['api/approve', 4]] : [['api/approve', 3], ['api/approve', 4]]);
  });
}
