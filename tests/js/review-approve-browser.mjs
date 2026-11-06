import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';

// Three synthetic clips: approve, send a Fix, approve, and land back on the list.
async function run(listUrl, resultFile, shots) {
  const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
  const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
    pipe: true, headless: true, args: ['--no-sandbox'],
    ...(process.env.YAP_STUDIO_BROWSER_PROFILE ? {userDataDir: process.env.YAP_STUDIO_BROWSER_PROFILE} : {}),
  });
  const result = {};
  const shot = async (page, name) => {
    if (!shots) return;
    await mkdir(shots, {recursive: true});
    await page.screenshot({path: join(shots, `${name}.png`), fullPage: true});
  };
  const statuses = (page) => page.$$eval('.review-status', (nodes) => nodes.map((node) => node.textContent));
  const videoPage = async (page, number) => {
    await page.waitForFunction((n) => location.pathname === `/v/${n}/`, {timeout: 20000}, number);
    await page.waitForSelector('#file-decision:not([hidden])', {timeout: 20000});
    await page.waitForFunction(() => document.getElementById('video').readyState >= 1, {timeout: 20000});
  };
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.setViewport({width: 1440, height: 900});
    await page.goto(listUrl);
    assert.deepEqual(await statuses(page), ['Not reviewed', 'Not reviewed', 'Not reviewed']);
    await page.waitForFunction(() => {
      const thumbnails = [...document.querySelectorAll('.review-thumbnail')];
      return thumbnails.length === 3 && thumbnails.every((image) => image.naturalWidth > 0);
    });
    await shot(page, '0-list');
    await page.click('#start-reviewing');
    await videoPage(page, 1);
    assert.equal(await page.evaluate(() => location.hash), '', 'the list token leaves the address bar');
    assert.equal(await page.$eval('#video', (node) => node.playbackRate), 1);
    await page.select('#playback-speed', '1.5');
    assert.equal(await page.$eval('#video', (node) => node.playbackRate), 1.5);
    await page.select('#playback-speed', '2');
    assert.equal(await page.$eval('#video', (node) => node.playbackRate), 2);
    await page.click('#file-approve');
    await page.waitForSelector('#file-undo:not([hidden])');
    result.undoShown = true;
    await shot(page, '1-approved');
    await videoPage(page, 2);
    assert.equal(await page.$eval('#playback-speed', (node) => node.value), '2');
    assert.equal(await page.$eval('#video', (node) => node.playbackRate), 2);

    await page.click('#new-comment');
    await page.waitForSelector('#compose:not([hidden])');
    await page.type('#compose-text', 'Trim the opening');
    result.approveBlockedByDraft = await page.$eval('#file-approve', (node) => node.disabled);
    await page.click('#compose-submit');
    await page.waitForFunction(() => !document.getElementById('file-send').disabled, {timeout: 10000});
    await page.click('#file-send');
    await videoPage(page, 3);
    await page.click('#file-approve');
    await page.waitForFunction(() => location.pathname === '/', {timeout: 20000});
    await page.waitForSelector('.review-list');
    result.final = await statuses(page);
    result.allDecided = await page.$eval('#start-reviewing', (node) => node.textContent);
    await shot(page, '2-list-done');

    const phone = await browser.newPage();
    phone.on('pageerror', (error) => errors.push(error.message));
    await phone.evaluateOnNewDocument(() => {
      Object.defineProperty(window, 'sessionStorage', {get() { throw new Error('Storage blocked'); }});
    });
    await phone.setViewport({width: 390, height: 844, isMobile: true, hasTouch: true});
    await phone.goto(listUrl);
    await phone.waitForFunction(() => [...document.querySelectorAll('.review-thumbnail')]
      .every((image) => image.naturalWidth > 0));
    assert.equal(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await shot(phone, '5-phone-list');
    await phone.tap('.review-list li:first-child a');
    await videoPage(phone, 1);
    assert.equal(await phone.$eval('#video', (node) => node.playbackRate), 1);
    await phone.select('#playback-speed', '1.5');
    assert.equal(await phone.$eval('#video', (node) => node.playbackRate), 1.5);
    const box = (selector) => phone.$eval(selector, (node) => {
      const {width, left, right} = node.getBoundingClientRect();
      const panel = node.closest('.player-panel').getBoundingClientRect();
      return {width, left, right, panel: panel.width};
    });
    result.phoneUndo = await box('#file-undo');
    await shot(phone, '3-phone-approved');
    await phone.tap('#file-undo');
    await phone.waitForSelector('#file-approve:not([hidden])');
    result.phoneApprove = await box('#file-approve');
    await shot(phone, '4-phone-undone');
    result.errors = errors;
  } finally {
    await browser.close();
  }
  await writeFile(resultFile, JSON.stringify(result));
}

// The broad Node gate can load this file without starting a browser.
if (process.argv[2]) await run(...process.argv.slice(2));
