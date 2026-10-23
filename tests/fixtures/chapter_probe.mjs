// Measure the chapter's actual glyph boxes after seeking its animations.
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [modules, browserPath, html] = process.argv.slice(2);
const require = createRequire(path.join(modules, 'noop.js'));
const puppeteer = require('puppeteer-core');
const variables = JSON.parse(fs.readFileSync(0, 'utf8'));
const browser = await puppeteer.launch({ executablePath: browserPath, headless: true });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1080, height: 1920 });
  await page.evaluateOnNewDocument(v => {
    window.__hyperframes = { getVariables: () => v };
  }, variables);
  await page.goto(pathToFileURL(html).href);
  await page.evaluate(() => document.fonts.ready);
  const boxes = await page.evaluate(() => {
    for (const animation of document.getAnimations()) {
      animation.pause();
      animation.currentTime = 800;
    }
    return Object.fromEntries(['number', 'title', 'timecode'].map(name => {
      const range = document.createRange();
      range.selectNodeContents(document.querySelector(`.${name}`));
      const r = range.getBoundingClientRect();
      return [name, [r.x, r.y, r.width, r.height]];
    }));
  });
  process.stdout.write(JSON.stringify(boxes));
} finally {
  await browser.close();
}
