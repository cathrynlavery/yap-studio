import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {createInterface} from 'node:readline';

// Download a two-video review, lose the server, review both videos and add a note offline,
// then reconnect and check the note syncs. The harness on stdin stops and restarts the server.
// YAP_STUDIO_PLAYWRIGHT (playwright-core) runs WebKit or another Playwright browser;
// otherwise YAP_STUDIO_PUPPETEER and YAP_STUDIO_CHROME run Chrome.
async function step(lines, name) {
  console.log(name);
  const reply = await lines.next();
  assert.equal(reply.value, 'ok', `harness refused ${name}`);
}

async function launch() {
  const load = createRequire(import.meta.url);
  if (process.env.YAP_STUDIO_PLAYWRIGHT) {
    const playwright = load(process.env.YAP_STUDIO_PLAYWRIGHT);
    const name = process.env.YAP_STUDIO_PLAYWRIGHT_BROWSER || 'webkit';
    const browser = await playwright[name].launch({headless: true});
    const context = await browser.newContext({viewport: {width: 390, height: 844},
      isMobile: name !== 'firefox', hasTouch: true});
    return {name, browser, page: await context.newPage(), offline: (value) => context.setOffline(value)};
  }
  const puppeteer = load(process.env.YAP_STUDIO_PUPPETEER);
  const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
    pipe: true, headless: true, args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required']});
  const page = await browser.newPage();
  await page.setViewport({width: 390, height: 844, isMobile: true, hasTouch: true});
  return {name: 'chrome', browser, page, offline: (value) => page.setOfflineMode(value)};
}

async function until(page, check, arg = null, timeout = 30000, what = 'condition') {
  const deadline = Date.now() + timeout;
  for (;;) {
    const value = await page.evaluate(check, arg).catch(() => null);
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

async function playFor(page, seconds) {
  await until(page, () => document.getElementById('video').readyState >= 1, null, 30000, 'video metadata');
  const start = await page.evaluate(async () => {
    const video = document.getElementById('video');
    video.muted = true;
    await video.play();
    return video.currentTime;
  });
  await new Promise((resolve) => setTimeout(resolve, seconds * 1000));
  const end = await page.evaluate(() => {
    const video = document.getElementById('video');
    video.pause();
    return {time: video.currentTime, error: video.error?.code ?? null};
  });
  return {start, end: end.time, error: end.error};
}

async function run(listUrl, resultFile) {
  const lines = createInterface({input: process.stdin})[Symbol.asyncIterator]();
  const {name, browser, page, offline} = await launch();
  const result = {browser: name, errors: [], videos: []};
  page.on('pageerror', (error) => result.errors.push(error.message));
  try {
    await page.goto(listUrl);
    await until(page, () => document.querySelector('.offline-save')?.offsetParent, null, 30000,
      'a visible Download button on a phone');
    result.button = await page.evaluate(() => document.querySelector('.offline-save').textContent);
    await page.click('.offline-save');
    result.ready = await until(page, () => {
      const text = document.querySelector('.offline-status')?.textContent ?? '';
      return /ready offline/.test(text) ? text : null;
    }, null, 120000, 'download');
    result.states = await page.evaluate(() => [...document.querySelectorAll('.offline-state')]
      .map((node) => node.textContent));
    result.storage = await page.evaluate(() => document.querySelector('.offline-storage')?.textContent);
    const links = await page.evaluate(() => [...document.querySelectorAll('.review-list a')]
      .map((link) => link.href));

    await step(lines, 'offline');
    // The harness has stopped the server. Playwright's WebKit offline switch also blocks
    // requests its service worker would answer, which a phone in airplane mode does not.
    if (name !== 'webkit') await offline(true);
    await page.reload();
    result.offlineList = await until(page, () => document.querySelectorAll('.review-list li').length,
      null, 30000, 'offline list');
    for (const [index, link] of links.entries()) {
      await page.goto(link);
      await until(page, () => !/Loading/.test(document.getElementById('media-name').textContent),
        null, 30000, `video ${index + 1} page`);
      const played = await playFor(page, 2);
      result.videos.push({name: await page.evaluate(() => document.getElementById('media-name').textContent),
        ...played, banner: await page.evaluate(() => !document.getElementById('offline-banner').hidden)});
    }
    await page.click('#new-comment');
    await until(page, () => !document.getElementById('compose').hidden, null, 10000, 'compose');
    await page.type('#compose-text', 'Offline note from the car');
    await page.click('#compose-submit');
    result.pending = await until(page, () => document.querySelector('.comment.pending .send-state')
      ?.textContent, null, 10000, 'pending note');
    await page.reload();
    result.pendingAfterReload = await until(page, () => document.querySelector('.comment.pending .comment-text')
      ?.textContent, null, 30000, 'pending note after reload');

    await step(lines, 'online');
    if (name !== 'webkit') await offline(false);
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await until(page, () => !document.querySelector('.comment.pending')
      && [...document.querySelectorAll('.comment .comment-text')]
        .some((node) => node.textContent === 'Offline note from the car'), null, 60000, 'synced note');
    // Extra sync triggers must not post the note again.
    await page.evaluate(() => { window.dispatchEvent(new Event('focus')); window.dispatchEvent(new Event('online')); });
    await new Promise((resolve) => setTimeout(resolve, 1500));
    result.synced = await page.evaluate(() => [...document.querySelectorAll('.comment .comment-text')]
      .map((node) => node.textContent));
  } finally {
    await writeFile(resultFile, JSON.stringify(result, null, 2));
    await browser.close();
  }
}

// The broad Node gate can load this file without starting a browser.
if (process.argv[2]) {
  await run(...process.argv.slice(2));
  process.exit(0);
}
