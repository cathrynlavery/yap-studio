// Verify chapter choices and their actual composition on desktop and phone.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';

const [modules, browserPath, url, out] = process.argv.slice(2);
const require = createRequire(path.join(modules, 'noop.js'));
const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({ executablePath: browserPath, headless: true });
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['phone', 390, 844]]) {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    await page.setViewport({ width, height, isMobile: name === 'phone', hasTouch: name === 'phone' });
    await page.goto(`${url}#opening`, { waitUntil: 'load' });
    await page.waitForSelector('#slot-switch-opening-chapter');
    assert.equal(await page.$eval('#slot-switch-opening-chapter', (input) => input.checked), false);
    for (const id of ['opening-more', 'slot-fold-opening-chapter']) {
      await page.$eval(`#${id}`, (fold) => { fold.open = true; });
    }
    await page.$eval('#slot-fold-opening-chapter .slot-switch-preview', (preview) =>
      preview.scrollIntoView({ block: 'center' }));
    await page.waitForFunction(() => [...document.querySelectorAll('#slot-fold-opening-chapter iframe')]
      .some((frame) => frame.contentWindow?.__hfVariables?.number === '02'
        && frame.contentWindow.__playerReady));
    const result = await page.evaluate(async () => {
      const frames = [...document.querySelectorAll('#slot-fold-opening-chapter iframe')];
      const frame = frames.find((frame) => frame.contentWindow?.__hfVariables?.number === '02');
      const win = frame.contentWindow;
      await win.document.fonts.ready;
      await win.__player.pause();
      await win.__player.seek(.8);
      const root = win.document.querySelector('[data-composition-id]');
      const title = root.querySelector('.title');
      const style = win.getComputedStyle(title);
      return { number: root.querySelector('.number').textContent,
        title: title.textContent, timecode: root.querySelector('.timecode').textContent,
        width: root.offsetWidth, height: root.offsetHeight,
        lines: title.offsetHeight / parseFloat(style.lineHeight),
        visible: Number(style.opacity),
        wash: win.getComputedStyle(root.querySelector('.surface')).backgroundImage,
        shadow: style.textShadow,
        layerOpacity: Number(win.getComputedStyle(root).opacity),
        horizontalOverflow: document.documentElement.scrollWidth > innerWidth,
        order: frames.map((frame) => frame.title) };
    });
    assert.equal(result.number, '02');
    assert.equal(result.title, 'Sample chapter');
    assert.equal(result.timecode, '01:24');
    assert.deepEqual([result.width, result.height], [1080, 1920]);
    assert.ok(result.lines <= 3.05);
    assert.equal(result.visible, 1);
    assert.equal(result.layerOpacity, 1);
    assert.ok(result.wash.includes('gradient'));
    assert.notEqual(result.shadow, 'none');
    assert.equal(result.horizontalOverflow, false);
    assert.deepEqual(result.order, ['chapter live preview']);
    await page.$eval('#slot-switch-opening-chapter', (input) => input.click());
    await page.waitForFunction(() => document.getElementById('screen-summary')
      .textContent.includes('chapter cards on'));
    await page.evaluate(() => {
      const heading = document.getElementById('slot-fold-opening-chapter').getBoundingClientRect();
      const top = innerWidth < 900
        ? document.querySelector('.sticky-preview').getBoundingClientRect().bottom + 12 : 24;
      window.scrollBy(0, heading.top - top);
    });
    assert.equal(await page.$$eval('#slot-fold-opening-chapter .compact', (nodes) => nodes.length), 0);
    await page.screenshot({ path: path.join(out, `${name}-chapters.png`) });
    await page.$eval('#slot-switch-opening-chapter', (input) => input.click());
    await page.waitForFunction(() => !document.getElementById('screen-summary')
      .textContent.includes('chapter cards on'));
    assert.equal(await page.$eval('#slot-switch-opening-chapter', (input) => input.checked), false);
    await page.waitForFunction(() => [...document.querySelectorAll('#slot-fold-opening-chapter iframe')]
      .some((frame) => frame.contentWindow?.__hfVariables?.number === '02'
        && frame.contentWindow.__playerReady));
    assert.deepEqual(errors, []);
    console.log(`${name}: chapter candidate preview, switch and layout passed`);
    await page.close();
  }
} finally {
  await browser.close();
}
