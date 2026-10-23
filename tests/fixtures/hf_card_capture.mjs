// Probe root scoping and capture an offline card through the pinned browser runtime.
import { createRequire } from 'node:module';
import path from 'node:path';

const [modules, browserPath, url, width, height, out] = process.argv.slice(2);
const require = createRequire(path.join(modules, 'noop.js'));
const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({
  executablePath: browserPath,
  headless: true,
  args: ['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage'],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('response', response => {
    if (response.status() >= 400 && !response.url().endsWith('/favicon.ico')) {
      errors.push(`${response.status()} ${response.url()}`);
    }
  });
  await page.setViewport({ width: Number(width), height: Number(height) });
  await page.goto(url, { waitUntil: 'load' });
  try {
    await page.waitForFunction(() => window.__playerReady && window.__timelines?.sample,
      { timeout: 10000 });
  } catch (error) {
    const state = await page.evaluate(() => ({
      ready: window.__playerReady,
      startupError: window.__hfStartupError,
      timelines: Object.keys(window.__timelines || {}),
      itemRoot: window.__itemRootProbe,
    }));
    throw new Error(`${error}\n${errors.join('\n')}\n${JSON.stringify(state)}`);
  }
  await page.evaluate(() => document.fonts.ready);
  for (let pass = 0; pass < 2; pass += 1) {
    await page.evaluate(async () => {
      window.__player.pause();
      await window.__player.seek(1);
    });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
  }
  const report = await page.evaluate(() => {
    const root = document.getElementById('root');
    const host = document.querySelector('[data-composition-id="hf-host"]');
    return {
      globalRoot: root ? { id: root.id, composition: root.dataset.compositionId } : null,
      itemRoot: window.__itemRootProbe,
      hostBackground: getComputedStyle(host).backgroundColor,
      rootAlias: window.__hfSvgSelectorAliases.rewrite('#root'),
      roots: [...document.querySelectorAll('[data-composition-id], [data-hf-inner-root]')]
        .map(element => Object.fromEntries([...element.attributes]
          .map(attribute => [attribute.name, attribute.value]))),
      css: [...document.styleSheets].flatMap(sheet =>
        [...sheet.cssRules].map(rule => rule.cssText)),
    };
  });
  await page.screenshot({ path: out, omitBackground: true });
  if (errors.length) throw new Error(errors.join('\n'));
  process.stdout.write(JSON.stringify(report));
} finally {
  await browser.close();
}
