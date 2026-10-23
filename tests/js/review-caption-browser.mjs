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
  const widths = [];
  try {
    for (const width of [390, 1440]) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.setViewport({width, height: width === 390 ? 844 : 900,
        isMobile: width === 390, hasTouch: width === 390});
      await page.evaluateOnNewDocument(() => Object.defineProperty(navigator, 'clipboard', {
        value: {writeText: async (text) => { window.copiedNotes = text; }},
      }));
      await page.goto(url);
      await page.waitForFunction(() => document.getElementById('video').videoWidth > 0
        && !document.getElementById('fix-caption').hidden);
      const tap = async (selector) => {
        await page.$eval(selector, (element) => element.scrollIntoView({block: 'center'}));
        if (width === 390) await page.tap(selector);
        else await page.click(selector);
      };
      const pictureTap = async (x, y) => {
        await page.$eval('#video', (element) => element.scrollIntoView({block: 'center'}));
        const spot = await page.$eval('#video', (video, point) => {
          const box = video.getBoundingClientRect();
          const scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight);
          return {x: box.left + (box.width - video.videoWidth * scale) / 2
              + point.x / 1080 * video.videoWidth * scale,
            y: box.top + (box.height - video.videoHeight * scale) / 2
              + point.y / 1920 * video.videoHeight * scale};
        }, {x, y});
        if (width === 390) await page.touchscreen.tap(spot.x, spot.y);
        else await page.mouse.click(spot.x, spot.y);
      };
      const visible = (selector) => page.$eval(selector, (element) => !element.hidden);
      await pictureTap(400, 1250);
      assert.equal(await visible('#caption-fix'), true);
      assert.equal(await visible('#compose'), false);
      assert.deepEqual(await page.$eval('#caption-words', (group) => {
        const chip = [...group.querySelectorAll('button')].find((word) => word.textContent === 'NYC');
        return {disabled: chip.disabled,
          reason: document.getElementById(chip.getAttribute('aria-describedby')).textContent};
      }), {disabled: true, reason: 'This caption joins several words. Fix it with a note.'});
      await mkdir(shots, {recursive: true});
      await page.$eval('#caption-fix', (element) => element.scrollIntoView({block: 'center'}));
      await page.screenshot({path: join(shots, `caption-${width}-merged.png`)});
      await tap('#caption-words button');
      assert.equal(await page.$eval('#caption-to', (element) => document.activeElement === element), true);
      await page.type('#caption-to', 'logging');
      assert.equal(await visible('#caption-every-label'), true);
      if (width === 1440) await tap('#caption-every');
      await mkdir(shots, {recursive: true});
      await page.screenshot({path: join(shots, `caption-${width}-panel.png`)});
      const saved = page.waitForResponse((response) => response.url().includes('/api/caption-fixes')
        && response.request().method() === 'POST');
      await tap('#caption-save');
      const response = await saved;
      assert.equal(response.status(), 201);
      const note = await response.json();
      assert.equal(note.request.word, 't1:0');
      assert.equal(note.request.to, 'logging');
      assert.equal(Boolean(note.request.heard), width === 1440);
      await page.waitForSelector(`[data-comment-id="${note.id}"]`);
      await tap('#copy-feedback');
      await page.waitForFunction(() => window.copiedNotes?.includes('Caption: BLOGGING -> logging'));
      await page.screenshot({path: join(shots, `caption-${width}-saved.png`)});
      if (width === 390) {
        await tap('#fix-caption');
        await tap('#caption-words button');
        await page.type('#caption-to', 'writing');
        await tap('#caption-every');
        const refused = page.waitForResponse((item) => item.url().includes('/api/caption-fixes')
          && item.request().method() === 'POST');
        await tap('#caption-save');
        assert.equal((await refused).status(), 409);
        await page.waitForFunction(() => document.getElementById('caption-error').textContent
          === 'This word has a one-off fix. Undo it first.');
        assert.equal(await visible('#caption-fix'), true);
        assert.equal(await page.$eval('#caption-to', (element) => element.value), 'writing');
        await page.screenshot({path: join(shots, `caption-${width}-conflict.png`)});
        await tap('#caption-cancel');
      }
      const undone = page.waitForResponse((item) => item.url().includes('/api/caption-fixes')
        && item.request().method() === 'POST');
      await tap(`[data-comment-id="${note.id}"] .comment-actions button`);
      assert.equal((await undone).status(), 200);
      await page.waitForFunction((id) => !document.querySelector(`[data-comment-id="${id}"]`), {}, note.id);
      await tap('#fix-caption');
      await tap('#caption-cancel');
      assert.equal(await visible('#caption-fix'), false);
      await page.focus('#fix-caption');
      await page.keyboard.press('Enter');
      assert.equal(await visible('#caption-fix'), true);
      await page.keyboard.press('Escape');
      assert.equal(await visible('#caption-fix'), false);
      await pictureTap(400, 300);
      assert.equal(await visible('#compose'), true);
      await tap('#compose-cancel');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width);
      assert.deepEqual(errors, []);
      widths.push(width);
      await page.close();
    }
    await writeFile(resultFile, JSON.stringify(widths));
  } finally {
    await browser.close();
  }
}

if (process.argv[2]) await run(...process.argv.slice(2));
