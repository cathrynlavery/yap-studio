import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';

async function run(reviewUrl, homeUrl, shots, resultFile) {
  const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
  const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
    pipe: true, headless: true, args: ['--no-sandbox']});
  const results = [], failures = [];
  await mkdir(shots, {recursive: true});
  const check = (name, action) => {
    try { action(); results.push({name, passed: true}); }
    catch (error) { results.push({name, passed: false, error: error.message}); failures.push(name); }
  };
  try {
    for (const width of [1440, 390]) {
      const mobile = width === 390;
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.setViewport({width, height: mobile ? 844 : 900, isMobile: mobile, hasTouch: mobile});
      await page.goto(reviewUrl);
      await page.waitForFunction(() => document.querySelector('#video').readyState >= 2
        && document.querySelector('.video-map-block'));
      const rect = (selector) => page.$eval(selector, (node) => {
        const {top, bottom, width, height} = node.getBoundingClientRect();
        return {top, bottom, width, height};
      });
      if (!mobile) {
        for (const selector of ['#play-pause', '#new-comment', '#mark-start', '#video-map']) {
          const box = await rect(selector);
          check(`desktop controls visible: ${selector}`, () => assert.ok(box.top >= 0 && box.bottom <= 900,
            JSON.stringify(box)));
        }
      } else {
        const video = await rect('#video');
        check('phone player keeps its width', () => assert.ok(video.width >= 320));
      }
      await page.screenshot({path: join(shots, `6-review-controls-${width}.png`)});
      await page.click('#play-pause');
      await page.waitForFunction(() => !document.querySelector('#video').paused);
      await page.click('#play-pause');
      if (mobile) await page.click('#video-map-toggle');
      await page.$eval('#video-map', (node) => node.scrollIntoView({block: 'start'}));
      const labels = await page.$$eval('.video-map-row[data-lane="0"] .video-map-block', (blocks) =>
        blocks.map((node) => ({label: node.querySelector('.video-map-label').textContent,
          width: node.clientWidth, content: node.querySelector('.video-map-label').scrollWidth,
          available: node.querySelector('.video-map-label').clientWidth,
          visible: getComputedStyle(node.querySelector('.video-map-label')).visibility})));
      check(`map has plain labels ${width}`, () => assert.ok(labels.every((item) =>
        !['s1', 's2', 'thats-it', 'payoff', 'trick'].includes(item.label))));
      if (mobile) check('phone map labels fit', () => assert.ok(labels.every((item) =>
        item.width >= 300 && item.content <= item.available && item.visible === 'visible'), JSON.stringify(labels)));
      await page.screenshot({path: join(shots, `8-video-map-${width}.png`)});
      const label = labels[0].label;
      await page.click('.video-map-row[data-lane="0"] .video-map-block');
      const comment = await page.$eval('#compose-text', (node) => node.value);
      check(`map comment uses the words ${width}`, () => assert.equal(comment, `"${label}" `));
      await page.close();

      const home = await browser.newPage();
      home.on('pageerror', (error) => errors.push(error.message));
      await home.setViewport({width, height: mobile ? 844 : 900, isMobile: mobile, hasTouch: mobile});
      await home.goto(homeUrl);
      await home.waitForFunction(() => document.querySelector('#main .title')?.textContent === 'Claude is building your video.');
      check(`fresh home follows claimed final ${width}`, () => assert.equal(new URL(home.url()).hash, '#video/demo'));
      const progress = await home.$eval('#main', (node) => node.textContent);
      check(`building copy respects the choices ${width}`, () => {
        assert.match(progress, /chosen look/);
        assert.doesNotMatch(progress, /Captions, titles, cards|Captions, cards and music/);
      });
      await home.screenshot({path: join(shots, `7-home-building-${width}.png`), fullPage: true});
      const state = await home.evaluate(async () => (await fetch('/api/state')).json());
      await home.evaluate((name) => localStorage.setItem(`yap-home-draft:${name}`, JSON.stringify({
        video_type: 'talking_head', has_broll: false, broll_sources: [], takes: ['take.mp4'], album: '',
      })), state.workspace);
      await home.goto(`${homeUrl}#new/make`);
      await home.waitForSelector('#make');
      const make = await home.$eval('.next-list', (node) => node.textContent);
      check(`make copy respects the choices ${width}`, () => {
        assert.match(make, /chosen look/);
        assert.doesNotMatch(make, /cards|B-roll|music/);
      });
      await home.screenshot({path: join(shots, `7-home-make-${width}.png`), fullPage: true});
      check(`no script errors ${width}`, () => assert.deepEqual(errors, []));
      await home.close();
    }
  } finally {
    await browser.close();
    await writeFile(resultFile, JSON.stringify(results, null, 2));
  }
  assert.deepEqual(failures, []);
}

if (process.argv.length >= 6) await run(...process.argv.slice(2));
