// Check real phrase geometry across word starts, including invisible words.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [modules, browserPath, composition, variablesFile, out] = process.argv.slice(2);
const require = createRequire(path.join(modules, 'noop.js'));
const puppeteer = require('puppeteer-core');
const variables = JSON.parse(readFileSync(variablesFile, 'utf8'));
const browser = await puppeteer.launch({ executablePath: browserPath, headless: true, pipe: true });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.setViewport({ width: 1080, height: 500 });
  await page.evaluateOnNewDocument(variables => {
    window.__hyperframes = { getVariables: () => variables };
  }, variables);
  await page.goto(pathToFileURL(path.join(composition, 'index.html')).href);
  await page.waitForFunction(() => document.querySelectorAll('.cap-spoken-word').length === 4);
  const report = await page.evaluate(variables => {
    const phrase = JSON.parse(variables.phrases)[0];
    const root = document.querySelector('[data-composition-id]');
    const animations = root.getAnimations({ subtree: true });
    const rect = node => {
      const { x, y, width, height } = node.getBoundingClientRect();
      return { x, y, width, height };
    };
    const snapshots = [];
    for (const word of phrase.words) {
      const first = Math.ceil(Math.max(phrase.start, word.start) * variables.fps - 1e-6);
      for (const frame of [first - 1, first, first + 2]) {
        for (const animation of animations) {
          animation.pause();
          animation.currentTime = frame / variables.fps * 1000;
        }
        snapshots.push({ words: [...root.querySelectorAll('.w')].map(rect),
          ground: rect(root.querySelector('.cap-ground')) });
      }
    }
    return snapshots;
  }, variables);
  for (const snapshot of report) assert.deepEqual(snapshot, report[0]);
  assert.ok(report[0].ground.width >= Math.max(...report[0].words.map(r => r.x + r.width))
    - Math.min(...report[0].words.map(r => r.x)));
  for (const [label, time] of [['first', 0.35], ['second', 0.65]]) {
    await page.evaluate(time => {
      for (const animation of document.getAnimations()) animation.currentTime = time * 1000;
    }, time);
    await page.screenshot({ path: path.join(out, `${label}.png`), omitBackground: true });
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ snapshots: report.length, words: report[0].words.length,
    first_word: report[0].words[0] }));
} finally {
  await browser.close();
}
