// Capture the same moving card through forward and backward live seeks.
import { createRequire } from 'node:module';
import path from 'node:path';

const [modules, browserPath, url, width, height, output, seconds] = process.argv.slice(2);
const require = createRequire(path.join(modules, 'noop.js'));
const browser = await require('puppeteer-core').launch({
  executablePath: browserPath, headless: true,
  args: ['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage',
    '--font-render-hinting=none', '--force-color-profile=srgb', '--force-device-scale-factor=1'],
});
try {
  const page = await browser.newPage(), errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('response', response => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });
  await page.setViewport({ width: Number(width), height: Number(height) });
  await page.goto(url, { waitUntil: 'load' });
  try {
    await page.waitForFunction(() => Boolean(window.__playerReady &&
      (window.__timelines?.band || window.__timelines?.['hf-host'])), { timeout: 15000 });
  } catch (error) {
    const state = await page.evaluate(() => ({ready: window.__playerReady,
      timelines: Object.keys(window.__timelines || {}), startup: window.__hfStartupError}));
    throw new Error(`${error}\n${errors.join('\n')}\n${JSON.stringify(state)}`);
  }
  await page.evaluate(() => document.fonts.ready);
  const reports = [];
  for (const [index, before] of [0, Number(seconds) + 0.05].entries()) {
    await page.evaluate(async ({before, seconds}) => {
      await window.__player.pause();
      await window.__player.seek(before);
      await window.__player.seek(seconds);
    }, { before, seconds: Number(seconds) });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
    await page.screenshot({ path: `${output}-${index}.png`, omitBackground: true });
    const report = await page.evaluate(() => {
      const mount = document.getElementById('hf-mount');
      const quote = mount.querySelector('blockquote'), panel = mount.querySelector('.panel');
      const rows = element => [...element.querySelectorAll('[data-probe-row]')]
        .map(row => Number(getComputedStyle(row).opacity));
      return { time: window.__player.getTime(), text: mount.innerText,
        quoteOpacity: quote ? Number(getComputedStyle(quote).opacity) : null,
        panelOpacity: panel ? Number(getComputedStyle(panel).opacity) : null,
        rows: rows(mount),
        bars: [...mount.querySelectorAll('.bar-fill')]
          .map(bar => new DOMMatrixReadOnly(getComputedStyle(bar).transform).a),
        box: mount.getBoundingClientRect().toJSON(),
        nestedGroups: document.querySelectorAll('[data-hf-echo-group] [data-hf-echo-group]').length,
        groups: [...document.querySelectorAll('[data-hf-echo-group]')].length,
        copies: [...document.querySelectorAll('[data-hf-echo-group] > *')].reverse()
          .filter(copy => getComputedStyle(copy).visibility !== 'hidden')
          .map(copy => ({ opacity: Number(getComputedStyle(copy).opacity), text: copy.innerText,
            rows: rows(copy),
            filter: getComputedStyle(copy).filter, box: copy.getBoundingClientRect().toJSON() })) };
    });
    reports.push({ from: before, ...report });
  }
  if (errors.length) throw new Error(errors.join('\n'));
  process.stdout.write(JSON.stringify(reports));
} finally {
  await browser.close();
}
