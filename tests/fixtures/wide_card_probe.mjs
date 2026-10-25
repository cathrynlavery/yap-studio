import fs from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const require = createRequire(import.meta.url);
const puppeteer = require(input.puppeteer);
const dist = path.join(path.dirname(input.puppeteer), 'hyperframes/dist');
const compiler = fs.readdirSync(dist).filter(name => /^compiler-.*\.js$/.test(name));
if (compiler.length !== 1) throw new Error('Expected one pinned compiler module');
const {bundleToSingleHtml} = await import(pathToFileURL(path.join(dist, compiler[0])).href);
const globals = JSON.stringify(input.variables).replaceAll('<', '\\u003c');
const compiled = (await bundleToSingleHtml(input.composition)).replace('<head>',
  `<head><script>window.__hfVariables=${globals}</script>`);
const browser = await puppeteer.launch({executablePath: input.browser, headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']});
try {
  const page = await browser.newPage();
  await page.setRequestInterception(true);
  page.on('request', request => request.url() === input.url
    ? request.respond({status: 200, contentType: 'text/html', body: compiled})
    : request.continue());
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.setViewport({width: 1920, height: 1080});
  await page.goto(input.url);
  await page.waitForFunction(() => window.__playerReady, {timeout: 15000});
  if (input.kind.startsWith('hf:')) {
    await page.waitForFunction(() => document.querySelector('.ring-stage, .cse-headline'),
      {timeout: 15000});
  }
  await page.evaluate(() => document.fonts.ready);
  for (let pass = 0; pass < 2; pass++) {
    await page.evaluate(async sample => {
      window.__player.pause(); await window.__player.seek(sample);
    }, input.sample ?? 2.5);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
  }
  const report = await page.evaluate(() => {
    const rect = node => {
      const r = node.getBoundingClientRect();
      return {x: r.x, y: r.y, width: r.width, height: r.height};
    };
    const paper = document.getElementById('wide-paper');
    const panel = document.querySelector('.cse-headline') || document.querySelector(
      '#wide-content .panel, #wide-content .stack, #wide-content .tag-stack, #wide-content #hf-mount');
    const visible = node => {
      for (let parent = node; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent);
        if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) < .01) return false;
      }
      return true;
    };
    const text = [...document.querySelectorAll('#wide-content *')].filter(node =>
      [...node.childNodes].some(child => child.nodeType === Node.TEXT_NODE && child.textContent.trim())
      && visible(node) && node.getBoundingClientRect().height > 0)
      .map(node => ({text: node.textContent, className: String(node.className),
        size: parseFloat(getComputedStyle(node).fontSize), ...rect(node)}));
    const chip = document.querySelector('.traveler .chip');
    const ring = document.querySelector('.ring');
    return {paper: rect(paper), panel: rect(panel), text,
      chip: chip && {...rect(chip), scrollWidth: chip.scrollWidth},
      progress: ring && parseFloat(getComputedStyle(ring).getPropertyValue('--ring-progress')),
      progressValue: ring && Number(ring.getAttribute('aria-valuenow')),
      steps: [...document.querySelectorAll('.chip[role=listitem]')].map(node => ({
        state: node.dataset.state, color: getComputedStyle(node).color,
        background: getComputedStyle(node).backgroundColor, ...rect(node)}))};
  });
  if (errors.length) throw new Error(errors.join('\n'));
  process.stdout.write(JSON.stringify(report));
} finally {
  await browser.close();
}
