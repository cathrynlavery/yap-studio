import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';

async function run(url, resultFile, shots) {
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
      await page.goto(url);
      await page.waitForFunction(() => document.querySelector('#findings-count').textContent === '2');
      await page.$eval('#findings-section', (node) => { node.open = true; });
      const buttons = await page.$$('#findings-list .finding-actions button:first-child');
      assert.equal(buttons.length, 2);
      const writes = [];
      page.on('request', (request) => { if (request.method() !== 'GET') writes.push(request.method()); });
      const played = [];
      for (const [index, button] of buttons.entries()) {
        assert.equal(await button.evaluate((node) => node.textContent), 'Listen');
        await button.evaluate((node) => node.scrollIntoView({block: 'center'}));
        await button.click();
        await page.waitForFunction(() => !document.querySelector('#video').paused);
        const start = await page.$eval('#video', (node) => node.currentTime);
        assert.ok(Math.abs(start - (index ? 35 / 30 : 0)) < 0.3);
        await page.waitForFunction(() => document.querySelector('#video').paused);
        const end = await page.$eval('#video', (node) => node.currentTime);
        // Hosted runners pause a few frames late; 0.15 s still catches playing on into the next words.
        assert.ok(Math.abs(end - (index ? 119.5 / 30 : 84.5 / 30)) < 0.15);
        played.push({start, end});
      }
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
