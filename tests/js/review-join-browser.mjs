import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';

async function run(url, resultFile, shots, agent = 'Claude Code', cpuThrottle = '1') {
  const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
  const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
    pipe: true, headless: true, args: ['--no-sandbox']});
  const results = [];
  try {
    await mkdir(shots, {recursive: true});
    for (const mobile of [true, false]) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.setViewport({width: mobile ? 390 : 1440, height: mobile ? 844 : 900,
        isMobile: mobile, hasTouch: mobile});
      if (Number(cpuThrottle) > 1) {
        const cdp = await page.createCDPSession();
        await cdp.send('Emulation.setCPUThrottlingRate', {rate: Number(cpuThrottle)});
        await page.evaluateOnNewDocument(() => {
          // Playback advances while a busy main thread delays the playing observer.
          document.addEventListener('playing', () => {
            const until = performance.now() + 120;
            while (performance.now() < until) {}
          }, true);
          // Let EOF beat the final frame callback, as a busy main thread can on a slow Mac.
          window.joinEndStalled = false;
          const requestFrame = HTMLVideoElement.prototype.requestVideoFrameCallback;
          HTMLVideoElement.prototype.requestVideoFrameCallback = function(callback) {
            return requestFrame.call(this, (now, metadata) => {
              if (!window.joinEndStalled && metadata.mediaTime >= this.duration - 0.1 && !this.paused) {
                window.joinEndStalled = true;
                const until = performance.now() + 180;
                while (performance.now() < until) {}
              }
              callback(now, metadata);
            });
          };
        });
      }
      await page.goto(url);
      await page.waitForFunction(() => document.querySelector('#findings-count').textContent === '2');
      assert.ok(await page.$eval('#range-note', (el, name) => el.placeholder.includes(name), agent));
      assert.ok(await page.$eval('#version-note', (el, name) => el.textContent.startsWith(name), agent));
      await page.$eval('#findings-section', (node) => { node.open = true; });
      const buttons = await page.$$('#findings-list .finding-actions button:first-child');
      assert.equal(buttons.length, 2);
      const writes = [];
      page.on('request', (request) => { if (request.method() !== 'GET') writes.push(request.method()); });
      const played = [];
      for (const [index, button] of buttons.entries()) {
        assert.equal(await button.evaluate((node) => node.textContent), 'Listen');
        await button.evaluate((node) => node.scrollIntoView({block: 'center'}));
        // The previous Listen ends with a seek; its late seeked event must not count as this start.
        await page.waitForFunction(() => {
          const video = document.querySelector('#video');
          return video.paused && !video.seeking;
        });
        await page.$eval('#video', (node) => {
          const sample = window.joinPlayback = {start: null, end: null, seeking: null};
          const play = node.play;
          // Observe the settled seek the app plays from, before the media clock advances.
          node.play = function(...args) {
            node.play = play;
            sample.start = node.currentTime;
            sample.seeking = node.seeking;
            return play.apply(this, args);
          };
          const recordEnd = () => {
            if (sample.start === null) return;
            sample.end = node.currentTime;
            node.removeEventListener('pause', recordEnd);
          };
          node.addEventListener('pause', recordEnd);
        });
        await button.click();
        await page.waitForFunction(() => window.joinPlayback.start !== null && window.joinPlayback.end !== null);
        const {start, end, seeking} = await page.evaluate(() => window.joinPlayback);
        assert.equal(seeking, false, `Listen ${index} played before its seek settled`);
        // Allow less than two frames at 30 fps.
        assert.ok(Math.abs(start - (index ? 35 / 30 : 0)) < 0.06, `Listen ${index} start: ${start}`);
        assert.ok(Math.abs(end - (index ? 119.5 / 30 : 84.5 / 30)) < 0.06, `Listen ${index} end: ${end}`);
        played.push({start, end});
      }
      if (Number(cpuThrottle) > 1) assert.equal(await page.evaluate(() => window.joinEndStalled), true);
      assert.deepEqual(writes, []);
      assert.deepEqual(errors, []);
      const screenshot = join(shots, mobile ? 'joins-phone.png' : 'joins-desktop.png');
      await page.screenshot({path: screenshot});
      results.push({mobile, played, screenshot});
      await page.close();
    }
    await writeFile(resultFile, JSON.stringify(results));
  } finally {
    await browser.close();
  }
}

if (process.argv[2]) await run(...process.argv.slice(2));
