import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';

async function run(url, resultFile, shots, agent = 'Claude Code') {
  const puppeteer = createRequire(import.meta.url)(process.env.YAP_STUDIO_PUPPETEER);
  const browser = await puppeteer.launch({executablePath: process.env.YAP_STUDIO_CHROME,
    pipe: true, headless: true, args: ['--no-sandbox']});
  const results = [];
  try {
    for (const width of [390, 1440]) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.setViewport({width, height: width === 390 ? 844 : 900,
        isMobile: width === 390, hasTouch: width === 390});
      await page.goto(url);
      await page.waitForSelector('.welcome');
      const disclosure = await page.$eval('.welcome .note', (el) => el.textContent);
      for (const phrase of ["Anthropic in Claude Code and the Claude app's Code tab",
        'OpenAI in Codex', 'Improve the model for everyone', 'Cursor (Anysphere)',
        'Privacy Mode stops training', 'command output']) assert.ok(disclosure.includes(phrase));
      await mkdir(shots, {recursive: true});
      await page.screenshot({path: join(shots, `welcome-${width}.png`), fullPage: true});
      await page.goto(`${url}#new/what`);
      await page.waitForSelector('.tile');
      assert.ok(await page.$eval('#listener', (el, name) => el.textContent.includes(name), agent));
      const settle = () => page.evaluate(async () => {
        await Promise.allSettled(document.getAnimations().filter((animation) =>
          animation.effect.getTiming().iterations !== Infinity).map((animation) => animation.finished));
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      });
      const click = async (selector) => {
        await page.$eval(selector, (el) => el.scrollIntoView({block: 'center'}));
        await settle();
        const point = await page.$eval(selector, (el) => {
          const rect = el.getBoundingClientRect();
          return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
        });
        if (width === 390) await page.touchscreen.tap(point.x, point.y);
        else await page.mouse.click(point.x, point.y);
        await settle();
      };
      const titles = {'#new/shape': 'Vertical or horizontal?', '#new/footage': 'Your footage',
        '#new/look': 'Your look', '#new/make': 'Make it'};
      const route = (hash) => page.waitForFunction((value, title) => location.hash === value
        && document.querySelector('h1')?.textContent === title, {}, hash, titles[hash]);
      const next = async (hash) => {
        await click('.actions button');
        try { await route(hash); }
        catch (error) {
          console.error(await page.evaluate(() => ({hash: location.hash,
            text: document.querySelector('main').textContent, toast: document.getElementById('toast').textContent})));
          throw error;
        }
        await settle();
      };
      await click('.tile');
      await next('#new/shape');
      assert.equal(await page.$eval('input[value=vertical]', (el) => el.checked), true);
      await click('input[value=horizontal]');
      assert.equal(await page.$('#has-broll-true'), null);
      await mkdir(shots, {recursive: true});
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({path: join(shots, `shape-${width}.png`), fullPage: true});
      await next('#new/footage');
      await page.waitForSelector('[role=alert]');
      assert.match(await page.$eval('[role=alert]', (el) => el.textContent), /phone.mp4 is vertical/);
      assert.equal(await page.$eval('.actions button', (el) => el.disabled), true);
      await click('#take-phone-mp4');
      await click('#take-computer-mp4');
      assert.equal(await page.$eval('input[value=horizontal]', (el) => el.checked), true);
      await next('#new/look');
      assert.equal(await page.$eval('[role=switch]', (el) => el.checked), false);
      assert.equal(await page.$('#open-style'), null);
      await click('[role=switch]');
      assert.equal(await page.$eval('[role=switch]', (el) => el.checked), true);
      if (width === 1440) await click('[role=switch]');
      await next('#new/make');
      assert.equal(await page.$('#text-hook'), null);
      assert.equal(await page.$('#top-text-choice'), null);
      const sent = page.waitForResponse((response) => response.url().includes('/api/requests')
        && response.request().method() === 'POST');
      await click('#make');
      const response = await sent;
      assert.equal(response.status(), 201);
      const request = await response.json();
      assert.equal(request.payload.shape, 'horizontal');
      assert.equal(request.payload.captions, width === 390);
      const payload = JSON.parse(response.request().postData());
      assert.equal('top_text' in payload, false);
      assert.equal('text_hook' in payload, false);
      assert.equal('top_text' in request.payload, false);
      assert.equal('text_hook' in request.payload, false);
      await page.evaluate(async (id) => {
        const token = new URLSearchParams(location.search).get('t');
        const response = await fetch(`/api/requests/${id}/cancel`, {method: 'POST',
          headers: {'X-Home-Token': token, 'Content-Type': 'application/json'}, body: '{}'});
        if (!response.ok) throw new Error('Could not reset browser request');
      }, request.id);

      // Changing takes suggests shape until the creator explicitly chooses one.
      await page.evaluate(() => {
        localStorage.clear();
        const key = `yap-home-draft:${document.getElementById('workspace-name').textContent}`;
        localStorage.setItem(key, JSON.stringify({video_type: 'talking_head', shape_seen: true,
          has_broll: false, broll_sources: [], takes: ['computer.mp4']}));
        location.hash = '#new/footage';
      });
      await route('#new/footage');
      await page.waitForSelector('input[value=horizontal]:checked');
      assert.match(await page.$eval('main', (el) => el.textContent), /wider than it is tall/);
      await click('#take-computer-mp4');
      await click('#take-phone-mp4');
      assert.equal(await page.$eval('input[value=vertical]', (el) => el.checked), true);
      assert.match(await page.$eval('main', (el) => el.textContent), /taller than it is wide/);
      await click('input[value=horizontal]');
      assert.match(await page.$eval('[role=alert]', (el) => el.textContent), /Choose vertical/);
      await click('input[value=vertical]');
      await next('#new/look');
      assert.ok(await page.$('#open-style'), 'vertical keeps the existing look picker');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width);
      assert.deepEqual(errors, []);
      results.push({width, request: request.id});
      await page.close();
    }
    await writeFile(resultFile, JSON.stringify(results));
  } finally {
    await browser.close();
  }
}

if (process.argv[2]) await run(...process.argv.slice(2));
