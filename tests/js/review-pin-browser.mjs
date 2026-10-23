import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';

async function run(url, resultFile, shots) {
  const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
  const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
    pipe: true, headless: true, args: ['--no-sandbox'],
    ...(process.env.YAP_STUDIO_BROWSER_PROFILE ? {userDataDir: process.env.YAP_STUDIO_BROWSER_PROFILE} : {}),
  });
  const results = [];
  try {
    for (const mobile of [true, false]) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.setViewport({width: mobile ? 390 : 1440, height: mobile ? 844 : 900,
        isMobile: mobile, hasTouch: mobile});
      await page.setRequestInterception(true);
      let holdMedia = true;
      const held = new Set();
      page.on('request', (request) => {
        if (holdMedia && new URL(request.url()).pathname.endsWith('/media')) held.add(request);
        else void request.continue();
      });
      await page.goto(url);
      await page.waitForSelector('#codec-error:not([hidden])', {timeout: 20000});
      assert.match(await page.$eval('#player-message', (node) => node.textContent), /load stalled/u);
      assert.equal(await page.$eval('#player-retry', (node) => node.textContent), 'Retry');
      assert.equal(await page.$eval('#offline-banner', (node) => node.hidden), true);
      await page.$eval('#player-retry', (node) => node.scrollIntoView({block: 'center'}));
      const retryBox = await page.$eval('#player-retry', (node) => {
        const {left, right, width, height} = node.getBoundingClientRect();
        return {left, right, width, height};
      });
      assert.ok(retryBox.left >= 0 && retryBox.right <= (mobile ? 390 : 1440));
      assert.ok(retryBox.width >= 44 && retryBox.height >= 44);
      if (shots) {
        const directory = mobile ? shots : join(shots, 'desktop');
        await mkdir(directory, {recursive: true});
        await page.screenshot({path: join(directory, '0-stalled.png')});
      }
      holdMedia = false;
      for (const request of held) await request.abort();
      if (mobile) await page.tap('#player-retry');
      else await page.click('#player-retry');
      await page.waitForFunction(() => document.getElementById('video').videoWidth > 0);
      await page.waitForFunction(() => document.getElementById('video').readyState >= 2);
      assert.equal(await page.$eval('#codec-error', (node) => node.hidden), true);
      const settle = () => page.evaluate(() => new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const show = async (selector) => {
        await page.$eval(selector, (element) => element.scrollIntoView({block: 'center'}));
        await settle();
      };
      const box = (selector) => page.$eval(selector, (element) => {
        const {left, top, width, height} = element.getBoundingClientRect();
        return {left, top, width, height};
      });
      const tap = async (selector) => {
        await show(selector);
        if (mobile) await page.tap(selector);
        else await page.click(selector);
      };
      const picture = async () => {
        const rect = await box('#video');
        const width = Math.min(rect.width, rect.height * 1080 / 1920);
        return {...rect, left: rect.left + (rect.width - width) / 2, width};
      };
      await show('#video');
      const initial = await picture();
      assert.ok(initial.width >= 280, JSON.stringify(initial));
      await show('#scrubber');
      const scrubber = await box('#scrubber');
      const start = {x: scrubber.left + scrubber.width * 0.2, y: scrubber.top + 18};
      const end = {x: scrubber.left + scrubber.width * 0.6, y: start.y};
      if (mobile) {
        const session = await page.createCDPSession();
        await session.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [start]});
        await session.send('Input.dispatchTouchEvent', {type: 'touchMove', touchPoints: [end]});
        await session.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
        await session.detach();
      } else {
        await page.mouse.move(start.x, start.y);
        await page.mouse.down();
        await page.mouse.move(end.x, end.y, {steps: 5});
        await page.mouse.up();
      }
      await page.waitForFunction(() => !document.getElementById('video').seeking);
      assert.equal(await page.$eval('#pin-layer', (element) => element.hidden), true);
      if (mobile) {
        await tap('#play-pause');
        await show('#video');
        const playingPicture = await picture();
        await page.touchscreen.tap(playingPicture.left + playingPicture.width * 0.42,
          playingPicture.top + playingPicture.height * 0.31);
        await page.waitForFunction(() => document.getElementById('video').paused);
        assert.equal(await page.$eval('#compose', (element) => element.hidden), true);
        assert.equal(await page.$eval('#pin-layer', (element) => element.hidden), true);
      } else await tap('#new-comment');
      await show('#video');
      const rect = await picture();
      const spot = {x: rect.left + rect.width * 0.42, y: rect.top + rect.height * 0.31};
      if (mobile) await page.touchscreen.tap(spot.x, spot.y);
      else await page.mouse.click(spot.x, spot.y);
      assert.equal(await page.$eval('#compose', (element) => element.hidden), false);
      if (mobile) assert.equal(await page.$eval('#compose-text', (element) =>
        document.activeElement === element), true);
      await page.waitForFunction(() => !document.getElementById('pin-layer').hidden);
      await show('#video');
      const pinnedPicture = await picture();
      const marker = await box('#pin-marker');
      assert.ok(marker.width >= 44 && marker.height >= 44, JSON.stringify(marker));
      assert.ok(Math.abs(marker.left + marker.width / 2
        - (pinnedPicture.left + pinnedPicture.width * 0.42)) <= 2);
      assert.ok(Math.abs(marker.top + marker.height / 2
        - (pinnedPicture.top + pinnedPicture.height * 0.31)) <= 2);
      const shotDir = shots ? (mobile ? shots : join(shots, 'desktop')) : null;
      const shot = async (name) => {
        if (shotDir) {
          await mkdir(shotDir, {recursive: true});
          await page.screenshot({path: join(shotDir, name)});
        }
      };
      await shot('1-pin-dropped.png');
      await show('#compose-text');
      await page.type('#compose-text', mobile ? 'Move this card on phone.' : 'Move this card on desktop.');
      await shot('2-typed.png');
      const frame = await page.$eval('#time-readout', (element) =>
        Number(element.title.match(/Frame (\d+)/u)[1]));
      const saved = page.waitForResponse((response) => response.url().includes('/api/comments')
        && response.request().method() === 'POST');
      await tap('#compose-submit');
      const response = await saved;
      assert.equal(response.status(), 201);
      const comment = await response.json();
      await page.waitForSelector(`[data-comment-id="${comment.id}"]`);
      await shot('3-saved.png');
      // Move away first so selecting the row must seek back to the pin.
      await page.$eval('#video', (video) => { video.currentTime = 0; });
      await page.waitForFunction(() => !document.getElementById('video').seeking);
      await tap(`[data-comment-id="${comment.id}"] .comment-text`);
      await page.waitForFunction(() => !document.getElementById('pin-layer').hidden);
      await show('#video');
      const selectedPicture = await picture();
      const selectedMarker = await box('#pin-marker');
      assert.ok(Math.abs(selectedMarker.left + selectedMarker.width / 2
        - (selectedPicture.left + selectedPicture.width * 0.42)) <= 2);
      assert.ok(Math.abs(selectedMarker.top + selectedMarker.height / 2
        - (selectedPicture.top + selectedPicture.height * 0.31)) <= 2);
      await shot('4-selected.png');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), mobile ? 390 : 1440);
      assert.deepEqual(errors, []);
      results.push({mobile, stall: true, retry: true, comment: comment.id, x: 0.42, y: 0.31, frame,
        pictureWidth: initial.width, markerWidth: marker.width, markerHeight: marker.height});
      await page.close();
    }
    await writeFile(resultFile, JSON.stringify(results, null, 2));
  } finally {
    await browser.close();
  }
}

// The broad Node gate can load this file without starting a browser.
if (process.argv[2]) await run(...process.argv.slice(2));
