// Capture one live gallery layer at a time, with a transparent background.
// usage: node live_capture.mjs NODE_MODULES BROWSER URL WIDTH HEIGHT SECONDS OUT.png
import { createRequire } from 'node:module';
import path from 'node:path';

const [modules, browserPath, url, width, height, seconds, out] = process.argv.slice(2);
const require = createRequire(path.join(modules, 'noop.js'));
const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({ executablePath: browserPath, headless: true });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  page.on('response', (response) => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });
  await page.setViewport({ width: Number(width), height: Number(height) });
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction('window.__playerReady', { timeout: 20000 });
  await page.evaluate('document.fonts.ready');
  // GSAP cards register their timeline once their fonts are in; seeking earlier is lost.
  await page.waitForFunction(() => document.querySelector('[data-no-timeline]')
    || Object.keys(window.__timelines || {}).length > 0, { timeout: 20000 });
  // The first seek after load binds a GSAP timeline without drawing it; seek twice.
  for (let pass = 0; pass < 2; pass += 1) {
    await page.evaluate(async (time) => {
      window.__player.pause();
      await window.__player.seek(time);
    }, Number(seconds));
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  await page.screenshot({ path: out, omitBackground: true });
  if (errors.length) throw new Error(errors.join('\n'));
} finally {
  await browser.close();
}
