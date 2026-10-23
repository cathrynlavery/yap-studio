import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {mkdir} from 'node:fs/promises';
import path from 'node:path';

const [styleURL, homeURL, widthText] = process.argv.slice(2);
const width = Number(widthText);
const out = process.env.YAP_STUDIO_TOP_TEXT_SHOTS;
const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
  headless: true, pipe: true});
try {
  if (out) await mkdir(out, {recursive: true});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.setViewport({width, height: width === 390 ? 844 : 900,
    isMobile: width === 390, hasTouch: width === 390});
  await page.emulateMediaFeatures([{name: 'prefers-reduced-motion', value: 'reduce'}]);
  await page.goto(`${styleURL}#top-text`);
  await page.waitForSelector('#top-text-tiles button');
  assert.deepEqual(await page.$$eval('#top-text-tiles .tile-label', nodes =>
    nodes.map(node => node.firstChild.textContent)), ['None', 'Just the opening', 'The whole video']);
  assert.equal(await page.$$eval('#top-text-tiles [aria-pressed="true"]', nodes => nodes.length), 0);
  await page.waitForFunction(() => document.getElementById('main-frame')._galleryStage?.frames
    .every(frame => frame._galleryReady));
  assert.equal(await page.$eval('#main-frame', node => node._galleryStage.candidate.hook.style), 'none');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width);
  if (out) await page.screenshot({path: path.join(out, `${width}-top-text.png`), fullPage: true});
  for (const [id, look] of [['whole', 'text_hook'], ['none', 'none'], ['opening', 'quiet_serif']]) {
    await page.$eval(`#top-text-${id}`, button => button.click());
    await page.waitForFunction(look => {
      const stage = document.getElementById('main-frame')._galleryStage;
      return stage?.candidate.hook.style === look && stage.frames.every(frame => frame._galleryReady);
    }, {}, look);
    assert.equal(await page.$eval(`#top-text-${id}`, button => button.getAttribute('aria-pressed')), 'true');
  }
  if (out) await page.screenshot({path: path.join(out, `${width}-top-text-opening.png`), fullPage: true});
  await page.click('#save');
  await page.waitForSelector('#save-sheet[open]');
  await page.$eval('#style-name', input => input.value = 'My top text');
  await page.click('#confirm-save');
  await page.waitForSelector('#saved:not([hidden])');
  await page.goto(homeURL);
  await page.evaluate(async () => {
    const state = await (await fetch('/api/state')).json();
    localStorage.setItem(`yap-home-draft:${state.workspace}`, JSON.stringify({
      video_type: 'voiceover', has_broll: true, broll_sources: ['album'], takes: [], album: 'Sample',
    }));
    location.hash = '#new/look';
  });
  await page.waitForFunction(() => document.querySelector('#main').textContent.includes('Text at the top: Just the opening'));
  await page.evaluate(() => location.hash = '#new/make');
  await page.waitForSelector('#top-text-choice');
  assert.equal(await page.$eval('#top-text-choice', input => input.value), 'opening');
  for (const [answer, hasText] of [['none', false], ['whole', true], ['opening', true]]) {
    await page.select('#top-text-choice', answer);
    assert.equal(Boolean(await page.$('#text-hook')), hasText);
  }
  await page.type('#text-hook', 'My chosen title');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width);
  if (out) await page.screenshot({path: path.join(out, `${width}-home-make.png`), fullPage: true});
  const sent = page.waitForResponse(response => response.url().includes('/api/requests')
    && response.request().method() === 'POST');
  await page.click('#make');
  const response = await sent;
  assert.equal(response.status(), 201);
  const request = await response.json();
  assert.equal(request.payload.top_text, 'opening');
  assert.equal(request.payload.text_hook, 'My chosen title');
  assert.deepEqual(errors, []);
  console.log(`Verified style and home at ${width}px`);
} finally {
  await browser.close();
}
