import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {join} from 'node:path';

const [home, review, expected, output, project = 'demo'] = process.argv.slice(2);
const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
  headless: true, pipe: true, args: ['--no-sandbox']});
try {
  for (const width of [390, 1440]) {
    for (const [surface, url, selector] of [
      ['home', `${home}#video/${encodeURIComponent(project)}`, '[data-youtube-chapters]'],
      ['review', review, '#youtube-chapters'],
    ]) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.setViewport({width, height: 900});
      await page.evaluateOnNewDocument(() => {
        window.copiedChapterText = null;
        Object.defineProperty(navigator, 'clipboard', {configurable: true,
          value: {writeText: async (text) => { window.copiedChapterText = text; }}});
      });
      await page.goto(url);
      await page.waitForSelector(`${selector}:not([hidden]) pre`);
      if (surface === 'review') {
        await page.waitForFunction(() => document.getElementById('video').readyState >= 2);
        await page.$eval('#video', (video) => { video.currentTime = Math.min(2, video.duration / 2); });
        await page.waitForFunction(() => !document.getElementById('video').seeking
          && document.getElementById('video').readyState >= 2);
      }
      assert.equal(await page.$eval(`${selector} pre`, (el) => el.textContent), expected);
      await page.click(`${selector} button`);
      await page.waitForFunction((text) => window.copiedChapterText === text, {}, expected);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      assert.deepEqual(errors, []);
      await page.screenshot({path: join(output, `chapters-${surface}-${width}.png`), fullPage: true});
      await page.close();
    }
  }
} finally {
  await browser.close();
}
