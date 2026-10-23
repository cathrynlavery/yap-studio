// Verify the shipped style page at desktop and 390x844 phone sizes.
// usage: node gallery_viewport.mjs NODE_MODULES BROWSER URL OUT_DIR
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';

const [modules, browserPath, url, out, mode] = process.argv.slice(2);
const require = createRequire(path.join(modules, 'noop.js'));
const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({ executablePath: browserPath, headless: true, pipe: true,
  ...(process.env.YAP_STUDIO_BROWSER_PROFILE ? { userDataDir: process.env.YAP_STUDIO_BROWSER_PROFILE } : {}),
});
try {
  for (const [name, width, height] of [['desktop', 1440, 900], ['phone', 390, 844]]) {
    const page = await browser.newPage();
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'no-preference' }]);
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    await page.setViewport({ width, height, isMobile: name === 'phone', hasTouch: name === 'phone' });
    if (mode === 'spoken') {
      await page.goto(`${url}#captions`, { waitUntil: 'load' });
      await page.waitForSelector('#caption-look-tiles button');
      await page.$$eval('#caption-look-tiles button', tiles => {
        tiles.find(tile => tile.querySelector('.tile-label').textContent.startsWith('Word by word')).click();
      });
      await page.waitForFunction(() => [...document.querySelectorAll('#main-frame iframe')]
        .some(frame => frame._galleryReady && frame.contentWindow.__hfVariables?.entrance === 'spoken'));
      const counts = await page.evaluate(async () => {
        const frame = [...document.querySelectorAll('#main-frame iframe')]
          .find(frame => frame.contentWindow.__hfVariables?.entrance === 'spoken');
        const win = frame.contentWindow;
        await win.__player.pause();
        const phrase = JSON.parse(win.__hfVariables.phrases).find(phrase => phrase.words.length > 1);
        const counts = [];
        for (const time of [phrase.start + 2 / 30, phrase.words[1].start + 2 / 30]) {
          await win.__player.seek(time);
          counts.push([...win.document.querySelectorAll('.cap-spoken-word')].filter(word => {
            let opacity = 1;
            for (let node = word; node; node = node.parentElement) opacity *= Number(win.getComputedStyle(node).opacity);
            return opacity > 0;
          }).length);
        }
        return counts;
      });
      assert.deepEqual(counts, [1, 2]);
      assert.equal(await page.$$eval('#caption-look-tiles [aria-pressed="true"] .tile-label',
        labels => labels.map(label => label.childNodes[0].textContent).join()), 'Word by word');
      await page.screenshot({ path: path.join(out, `${name}-spoken.png`) });
      assert.deepEqual(errors, []);
      await page.close();
      continue;
    }
    if (mode === 'registry') {
      const failed = [], loaded = [];
      await page.setRequestInterception(true);
      page.on('request', (request) => {
        if (request.url().startsWith(new URL(url).origin + '/') || request.url().startsWith('data:')) request.continue();
        else { failed.push(request.url()); request.abort(); }
      });
      page.on('requestfailed', (request) => {
        // Tile availability probes can be canceled. Every failed asset GET still counts.
        const canceledProbe = request.method() === 'HEAD' && request.failure()?.errorText === 'net::ERR_ABORTED';
        if (request.url().includes('/layer/') && !canceledProbe) failed.push({
          url: request.url(), error: request.failure()?.errorText, method: request.method(),
        });
      });
      page.on('response', (response) => {
        if (!response.url().includes('/layer/')) return;
        if (response.status() >= 400) failed.push(`${response.status()} ${response.url()}`);
        else loaded.push(response.url());
      });
      await page.goto(`${url}#cards`, { waitUntil: 'load' });
      for (const [category, item, label] of [
        ['words', 'char-slam-explode', 'Slam headline'],
        ['numbers', 'conic-progress-ring', 'Progress ring'],
      ]) {
        const id = `kind-hf:${item}`;
        await page.waitForSelector(`#card-group-${category}`);
        await page.$eval(`#card-group-${category}`, (fold) => fold.open = true);
        await page.waitForFunction((id) => Boolean(document.getElementById(id)), {}, id);
        const row = await page.evaluate((id) => {
          const toggle = document.getElementById(id), row = toggle.closest('.kind-row');
          return { label: row.querySelector('label').textContent, checked: toggle.checked,
            aria: toggle.getAttribute('aria-label'), tile: row.querySelector('button').getAttribute('aria-label') };
        }, id);
        assert.deepEqual(row, { label, checked: false, aria: `${label} on`, tile: `Preview ${label}` });
        await page.evaluate((id) => document.getElementById(id).closest('.kind-row').scrollIntoView({ block: 'center' }), id);
        await page.waitForFunction((id) => {
          const host = document.getElementById(id).closest('.kind-row').querySelector('button');
          const frames = host._galleryStage?.frames;
          return frames?.length > 0 && frames.every((frame) => frame._galleryReady
            && frame.contentWindow.document.getElementById('hf-mount')?.children.length > 0);
        }, {}, id);
        assert.ok(loaded.some((asset) => asset.includes(`/compositions/components/${item}.html`)), item);
        await page.screenshot({ path: path.join(out, `${name}-${item}.png`) });
      }
      assert.deepEqual(failed, []);
      assert.deepEqual(errors, []);
      await page.close();
      continue;
    }
    if (mode === 'motion') {
      await page.goto(`${url}#cards`, { waitUntil: 'load' });
      await page.waitForSelector('#slot-fold-cards-card-motion');
      await page.$eval('#slot-fold-cards-card-motion', fold => {
        fold.open = true; fold.scrollIntoView({ block: 'center' });
      });
      assert.equal(await page.$eval('#slot-cards-none', tile => tile.getAttribute('aria-pressed')), 'true');
      await page.waitForFunction(() => {
        const host = document.querySelector('#slot-cards-streak .stage-host');
        return host?._galleryStage?.frames[0]?._galleryReady;
      });
      const report = await page.$eval('#slot-cards-streak .stage-host', async host => {
        const win = host._galleryStage.frames[0].contentWindow;
        const player = win.__player, mount = win.document.getElementById('hf-mount');
        await player.pause();
        const positions = [], content = [];
        for (const time of [0.1, 1, player.getDuration() - 0.1]) {
          await player.seek(time);
          positions.push(mount.getBoundingClientRect().left);
          content.push(Number(win.getComputedStyle(mount.querySelector('blockquote')).opacity));
        }
        await player.seek(0); await player.play();
        return { positions, content, playing: player.isPlaying(), autoplay: host._galleryStage.autoplay,
          start: host._galleryStage.startTimes.get(host._galleryStage.frames[0]) };
      });
      assert.ok(report.positions[0] > report.positions[1] + 50, JSON.stringify(report));
      assert.ok(report.positions[2] > report.positions[1] + 50, JSON.stringify(report));
      assert.equal(report.playing, true);
      assert.equal(report.autoplay, true);
      assert.equal(report.content[0], 1);
      assert.equal(report.start, 0);
      await page.screenshot({ path: path.join(out, `${name}-motion-streak.png`) });
      assert.deepEqual(errors, []);
      await page.close();
      continue;
    }
    await page.goto(`${url}#captions`, { waitUntil: 'load' });
    await page.waitForFunction(() => document.querySelectorAll('#caption-look-tiles button').length === 8);
    await page.evaluate(() => document.fonts.ready);
    await page.$eval('#caption-look-tiles > :first-child', (tile) => tile.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => {
      const tile = document.querySelector('#caption-look-tiles > :first-child').getBoundingClientRect();
      const viewer = document.querySelector('.sticky-preview'), preview = viewer.getBoundingClientRect();
      const overlaps = tile.left < preview.right && tile.right > preview.left && getComputedStyle(viewer).position === 'sticky';
      const topbar = document.querySelector('.topbar').getBoundingClientRect();
      window.scrollBy(0, tile.top - (overlaps ? preview.bottom + 12 : Math.max(0, topbar.bottom) + 12));
    });
    const openFold = async (id) => {
      await page.$eval(`#${id}`, (details) => {
        for (let parent = details; parent; parent = parent.parentElement) if (parent.tagName === 'DETAILS') parent.open = true;
      });
      await page.waitForFunction((id) => document.getElementById(id).open, {}, id);
    };
    const layout = await page.evaluate(() => {
      const $ = (id) => document.getElementById(id);
      const tile = $('caption-look-tiles').children[0];
      const label = tile.querySelector('.tile-label').getBoundingClientRect();
      const footer = $('savebar').getBoundingClientRect();
      return { labelBottom: label.bottom,
        footerTop: footer.top, sameFold: $('caption-color-fold').parentElement === $('background-fields').parentElement,
        moreClosed: !$('caption-animation').open, colorClosed: !$('caption-color-fold').open,
        horizontalOverflow: document.documentElement.scrollWidth > innerWidth,
        faceFirst: Boolean($('avoid-face').compareDocumentPosition($('placements')) & Node.DOCUMENT_POSITION_FOLLOWING),
        preferredPositionNext: Boolean($('placements').compareDocumentPosition($('fine-top_px')) & Node.DOCUMENT_POSITION_FOLLOWING),
        faceOn: $('avoid-face').checked };
    });
    assert.equal(layout.sameFold, true);
    assert.equal(layout.moreClosed, true);
    assert.equal(layout.colorClosed, true);
    await page.waitForFunction(() => document.querySelector('#caption-look-tiles .tile-poster')?.hidden);
    const captionCrop = await page.$eval('#caption-look-tiles .tile-frame', (host) => {
      const stage = host._galleryStage, layer = stage.frames[0];
      const scale = host.clientWidth / 900, y = -parseFloat(stage.stage.style.top) / scale;
      const height = Math.min(900 * host.clientHeight / host.clientWidth, 1920);
      const center = parseFloat(layer.style.top) + parseFloat(layer.style.height) / 2;
      return { y, expected: Math.max(0, Math.min(1920 - height, center - height * (innerWidth < 900 ? .72 : .5))),
        ratio: host.clientWidth / host.clientHeight };
    });
    assert.ok(Math.abs(captionCrop.y - captionCrop.expected) < 1, JSON.stringify({ name, captionCrop }));
    assert.ok(Math.abs(captionCrop.ratio - (name === 'phone' ? 10 / 11 : 9 / 16)) < .01);
    await page.screenshot({ path: path.join(out, `${name}-captions.png`) });
    assert.ok(layout.labelBottom < layout.footerTop, `${name}: The first tile label is behind the save bar: ${JSON.stringify(layout)}`);
    assert.equal(layout.horizontalOverflow, false);
    assert.equal(layout.faceFirst, true);
    assert.equal(layout.preferredPositionNext, true);
    assert.equal(layout.faceOn, true);
    await page.evaluate(() => document.querySelectorAll('#caption-look-tiles button')[1].click());
    await page.waitForFunction(() => !document.getElementById('background-fields').hidden);
    await openFold('background-fields');
    await page.evaluate(() => {
      document.querySelectorAll('#background-swatches button')[1].click();
    });
    await page.waitForFunction(() => {
      const frames = document.querySelectorAll('#main-frame iframe');
      return [...frames].some((frame) => frame.contentWindow?.__hfVariables?.background_mode === 'box'
        && frame.contentWindow.__hfVariables.text_color === '#111111');
    });
    assert.equal(await page.$eval('#caption-color', (input) => input.value), '#FFFFFF');
    await page.evaluate(() => document.querySelectorAll('#caption-look-tiles button')[2].click());
    await page.waitForFunction(() => document.getElementById('background-corner-fields').hidden);
    assert.equal(await page.$eval('#background-fields', (fields) => fields.hidden), false);
    await page.evaluate(() => document.querySelectorAll('#caption-look-tiles button')[0].click());
    await page.waitForFunction(() => document.getElementById('background-fields').hidden);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.click('#step-more');
    await page.waitForFunction(() => document.body.dataset.screen === 'more');
    await page.click('#choice-grade');
    await page.waitForFunction(() => document.querySelectorAll('#grade-tiles button').length === 3);
    assert.equal(await page.$eval('#grade-strength', (node) => node.hidden), true);
    assert.equal(await page.$eval('#grade-strength-label', (node) => node.hidden), true);
    assert.equal(await page.$eval('#grade-slider', (node) => node.hidden), true);
    await page.waitForFunction(() => [...document.querySelectorAll('#grade-tiles .tile-poster img')]
      .every((image) => image.complete && image.naturalWidth > 0));
    const gradeImages = await page.$$eval('#grade-tiles .tile-poster img', (images) => images.map((image) => ({
      look: new URL(image.src).searchParams.get('look'), width: image.naturalWidth, height: image.naturalHeight,
    })));
    assert.deepEqual(gradeImages.slice(0, 2).map((image) => image.look), ['none', 'film']);
    assert.deepEqual([gradeImages[0].width, gradeImages[0].height], [gradeImages[1].width, gradeImages[1].height]);
    await page.evaluate(() => document.querySelectorAll('#grade-tiles button')[1].click());
    await page.waitForFunction(() => {
      const image = document.querySelector('#main-frame .stage img');
      return image?.complete && image.naturalWidth > 0 && image.src.includes('look=film');
    });
    await openFold('grade-strength-fold');
    await page.evaluate(() => document.getElementById('grade-strength').scrollIntoView({ block: 'center' }));
    assert.equal(await page.$eval('#grade-strength', (node) => node.hidden), false);
    assert.equal(await page.$eval('#grade-strength-label', (node) => node.hidden), false);
    assert.equal(await page.$eval('#grade-strength-label', (node) => node.textContent), 'Strength');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    const gradeLayout = await page.$eval('#grade-strength', (node) => ({
      top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom,
      footer: document.getElementById('savebar').getBoundingClientRect().top,
    }));
    assert.ok(gradeLayout.top >= 0 && gradeLayout.bottom < gradeLayout.footer, `${name}: grade strength is covered`);
    await page.screenshot({ path: path.join(out, `${name}-grade.png`) });
    await openFold('grade-slider-fold');
    await page.evaluate(() => document.querySelectorAll('#grade-slider button')[1].click());
    await page.waitForFunction(() => {
      const image = document.querySelector('#main-frame .stage img');
      return image?.complete && image.naturalWidth > 0 && image.src.includes('slider=1');
    });
    await page.$eval('#grade-slider', (node) => node.scrollIntoView({ block: 'center' }));
    const sliderLayout = await page.$eval('#grade-slider', (node) => ({
      top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom,
      footer: document.getElementById('savebar').getBoundingClientRect().top,
    }));
    assert.ok(sliderLayout.top >= 0 && sliderLayout.bottom < sliderLayout.footer);
    await page.screenshot({ path: path.join(out, `${name}-grade-slider.png`) });
    await page.evaluate(() => document.querySelector('#grade-slider button').click());
    await page.evaluate(() => document.querySelector('#grade-strength button').click());
    await page.waitForFunction(() => document.querySelector('#main-frame .stage img')?.src.includes('strength=0.5'));
    await page.evaluate(() => document.querySelectorAll('#grade-tiles button')[0].click());
    await page.waitForFunction(() => document.getElementById('grade-strength').hidden);
    assert.equal(await page.$eval('#grade-strength-label', (node) => node.hidden), true);
    await page.evaluate(() => document.querySelectorAll('#grade-tiles button')[2].click());
    assert.equal(await page.$eval('#grade-color-fields', (node) => node.hidden), false);
    assert.equal(await page.$('#grade-swatch'), null);
    assert.equal(await page.$eval('#grade-picker', (node) => node.getAttribute('aria-label')), 'Pick a grade color');
    await page.$eval('#grade-color', (input) => {
      input.value = '#2FAE5A'; input.dispatchEvent(new Event('change'));
    });
    await page.waitForFunction(() => {
      const image = document.querySelector('#main-frame .stage img');
      const params = image && new URL(image.src).searchParams;
      return params?.get('look') === 'eyedropper' && params.get('color') === '#2FAE5A'
        && document.getElementById('grade-color').value === '#2FAE5A'
        && document.getElementById('grade-picker').value.toUpperCase() === '#2FAE5A';
    });
    assert.equal(await page.$eval('#grade-strength button', (node) => node.getAttribute('aria-pressed')), 'true');
    await page.$eval('#grade-color', (input) => {
      input.value = '#808080'; input.dispatchEvent(new Event('change'));
    });
    await page.waitForFunction(() => document.getElementById('grade-error').textContent.includes('too grey'));
    assert.equal(await page.$eval('#grade-picker', (node) => node.value.toUpperCase()), '#2FAE5A');
    const tap = await page.$eval('#main-frame', (node) => {
      const box = node.getBoundingClientRect();
      return { x: box.left + box.width * .5, y: box.top + box.height * .4 };
    });
    await page.mouse.click(tap.x, tap.y);
    await page.waitForFunction(() => document.getElementById('grade-color').value !== '#808080'
      && document.getElementById('grade-color').value !== '#2FAE5A'
      && !document.getElementById('grade-error').textContent);
    await page.waitForFunction(() => {
      const main = document.querySelector('#main-frame .stage img');
      const images = [...document.querySelectorAll('#grade-tiles img'), main];
      return main && new URL(main.src).searchParams.get('color') === document.getElementById('grade-color').value
        && images.every((image) => image?.complete && image.naturalWidth > 0);
    });
    await page.$eval('#grade-color-fields', (node) => node.scrollIntoView({ block: 'center' }));
    await page.screenshot({ path: path.join(out, `${name}-eyedropper.png`) });
    for (const [screen, grid] of [['opening', 'title-tiles'], ['cards', 'card-style-tiles']]) {
      await page.evaluate(() => document.getElementById('step-more').click());
      await page.waitForFunction(() => document.body.dataset.screen === 'more');
      await page.click(`#choice-${screen}`);
      await page.waitForFunction((grid) => document.querySelector(`#${grid} .tile-poster`)?.hidden, {}, grid);
      const crop = await page.$eval(`#${grid} .tile-frame`, (host) => {
        const stage = host._galleryStage, layer = stage.frames[0];
        const boxHeight = Math.min(900 * host.clientHeight / host.clientWidth, 1920);
        const center = parseFloat(layer.style.top) + parseFloat(layer.style.height) / 2;
        const y = Math.max(0, Math.min(1920 - boxHeight, center - boxHeight / 2));
        return { top: parseFloat(stage.stage.style.top), expected: innerWidth < 900 ? 0 : -y * host.clientWidth / 900,
          scale: parseFloat(stage.stage.style.transform.slice(6)), expectedScale: host.clientWidth / (innerWidth < 900 ? 1080 : 900),
          ratio: host.clientWidth / host.clientHeight };
      });
      assert.ok(Math.abs(crop.top - crop.expected) < 1, JSON.stringify({ name, screen, crop }));
      assert.ok(Math.abs(crop.ratio - 9 / 16) < .01);
      assert.ok(Math.abs(crop.scale - crop.expectedScale) < .000001);
      await page.screenshot({ path: path.join(out, `${name}-${screen}.png`) });
      if (screen === 'opening') {
        await page.$$eval('#title-tiles button', (tiles) => tiles.find((tile) => tile.querySelector('.tile-label').textContent.startsWith('Video in letters')).click());
        await page.waitForFunction(() => [...document.querySelectorAll('#main-frame iframe')].some((frame) => {
          const w = frame.contentWindow;
          return w?.__hfVariables?.look === 'video_letters' && w.document.getElementById('root')?.dataset.ready === 'true';
        }));
        const letters = await page.$eval('#main-frame', (host) => {
          const stage = host._galleryStage, frame = stage.frames.find((frame) => frame.contentWindow?.__hfVariables?.look === 'video_letters');
          const w = frame.contentWindow, text = w.document.querySelector('#letter-ink text'), ink = text.getBBox();
          const still = host.querySelector('.stage > img');
          return { text: text.textContent, width: ink.width, height: ink.height,
            opacity: Number(w.getComputedStyle(w.document.getElementById('plate')).opacity),
            mask: w.document.getElementById('plate-color').getAttribute('mask'),
            image: still.complete && still.naturalWidth > 0,
            time: w.__player.getTime(), hold: w.__hfVariables.hold_frames / w.__hfVariables.fps };
        });
        assert.ok(letters.text.trim() && letters.width > 100 && letters.height > 100, JSON.stringify(letters));
        assert.equal(letters.mask, 'url(#letter-mask)');
        assert.equal(letters.opacity, 1);
        assert.equal(letters.image, true);
        assert.ok(letters.time < letters.hold);
        await page.waitForFunction(() => document.getElementById('title-note').textContent === 'Shows for the first 1.6 seconds.');
        await page.screenshot({ path: path.join(out, `${name}-video-letters.png`) });
        const tile = await page.$$('#title-tiles button');
        const index = await page.$$eval('#title-tiles .tile-label', (labels) => labels.findIndex((label) => label.textContent.startsWith('Video in letters')));
        await tile[index].evaluate((tile) => tile.scrollIntoView({ block: 'center' }));
        await page.waitForFunction((index) => {
          const host = document.querySelectorAll('#title-tiles .tile-frame')[index];
          return host.querySelector('.tile-poster')?.hidden && host.querySelector('iframe')?.contentWindow.document.getElementById('root')?.dataset.ready === 'true';
        }, {}, index);
        await page.screenshot({ path: path.join(out, `${name}-video-letters-tile.png`) });
      }
      if (screen === 'cards') {
        await openFold('card-group-numbers');
        const selector = '[aria-label="Preview Big number behind you"]';
        assert.equal(await page.$eval('#card-group-numbers > summary', (summary) => summary.textContent.includes('Numbers · 4 kinds')), true);
        assert.equal(await page.$eval('#kind-big_number', (input) => input.checked), false);
        assert.equal(await page.$eval('#kind-big_number', (input) => input.previousElementSibling.querySelector('.kind-note').textContent), 'Counts up behind your head');
        await page.$eval(selector, (host) => host.scrollIntoView({ block: 'center' }));
        await page.waitForFunction((selector) => {
          const host = document.querySelector(selector);
          return host?._galleryStage?.frames.every((frame) => frame._galleryReady)
            && host.querySelector('.card-cutout')?.complete && host.querySelector('.card-cutout').naturalWidth === 1080;
        }, {}, selector);
        await page.$eval(selector, (host) => host.click());
        await page.waitForFunction(() => document.getElementById('main-frame')._galleryStage?.usesCutout
          && document.getElementById('main-frame')._galleryStage.frames.every((frame) => frame._galleryReady)
          && document.querySelector('#main-frame .card-cutout')?.naturalWidth === 1080);
        for (const target of [selector, '#main-frame']) {
          const overlay = await page.$eval(target, async (host) => {
            const stage = host._galleryStage, frame = stage.frames[0], image = stage.cutout;
            await frame.contentWindow.__player.pause();
            await frame.contentWindow.__player.seek(1.5);
            await new Promise(requestAnimationFrame);
            const digits = frame.contentWindow.document.querySelector('.number-strip svg:last-child g');
            const ink = digits.getBBox(), style = getComputedStyle(image);
            const background = new URL(stage.stage.querySelector('img').src), cutout = new URL(image.src);
            return { size: [style.width, style.height], pointer: style.pointerEvents,
              sameGrade: ['look', 'strength', 'color'].every((key) => background.searchParams.get(key) === cutout.searchParams.get(key)),
              last: stage.stage.lastElementChild === image, text: digits.textContent,
              width: ink.width, height: ink.height, opacity: Number(frame.contentWindow.getComputedStyle(frame.contentWindow.document.getElementById('root')).opacity) };
          });
          assert.deepEqual(overlay.size, ['1080px', '1920px']);
          assert.equal(overlay.pointer, 'none'); assert.equal(overlay.last, true);
          assert.equal(overlay.sameGrade, true);
          assert.equal(overlay.text, '10,000');
          assert.ok(overlay.width > 100 && overlay.height > 100 && overlay.opacity === 1, JSON.stringify(overlay));
        }
        assert.equal(await page.$$eval('#card-style-tiles .card-cutout', (images) => images.length), 0);
        await page.screenshot({ path: path.join(out, `${name}-big-number.png`) });
        await page.$eval('[aria-label="Preview Stat"]', (host) => host.click());
        await page.waitForFunction(() => !document.getElementById('main-frame')._galleryStage?.usesCutout);
        assert.equal(await page.$$eval('#main-frame .card-cutout', (images) => images.length), 0);
      }
    }
    await page.evaluate(() => document.getElementById('step-more').click());
    await page.waitForFunction(() => document.body.dataset.screen === 'more');
    await page.click('#choice-broll');
    await page.waitForSelector('#slot-broll-split');
    await page.$eval('#slot-broll-split', (tile) => tile.click());
    await page.waitForFunction(() => document.getElementById('slot-broll-split').getAttribute('aria-pressed') === 'true');
    const split = await page.$eval('#main-frame', (frame) => {
      const shot = frame.querySelector('.split-shot').getBoundingClientRect(), speaker = frame.querySelector('.split-speaker');
      const lower = speaker.getBoundingClientRect();
      const picture = speaker.querySelector('img'), scale = Number(picture.style.transform.match(/scale\(([^)]+)\)/)[1]);
      return { seam: Math.abs(shot.bottom - lower.top), ratio: shot.height / lower.height,
        nativeHeight: parseFloat(getComputedStyle(speaker).height), imageReady: picture.complete && picture.naturalWidth > 0,
        faceTop: 510 * scale + parseFloat(picture.style.top), faceBottom: 970 * scale + parseFloat(picture.style.top),
        captions: frame.querySelectorAll('.poster-caption').length, iframes: frame.querySelectorAll('iframe').length };
    });
    assert.ok(split.seam < 1 && Math.abs(split.ratio - 1) < .01, JSON.stringify(split));
    assert.equal(split.nativeHeight, 960);
    assert.equal(split.imageReady, true);
    assert.ok(split.faceTop >= 24 && split.faceBottom <= 960, JSON.stringify(split));
    assert.equal(split.captions, 1); assert.equal(split.iframes, 0);
    await page.$eval('#slot-broll-split', (tile) => tile.scrollIntoView({ block: 'center' }));
    await page.screenshot({ path: path.join(out, `${name}-broll-split.png`) });
    await openFold('slot-fold-broll-color-match');
    assert.equal(await page.$eval('#slot-switch-broll-color-match', (input) => input.checked), true);
    await page.$eval('#slot-switch-broll-color-match', (input) => input.click());
    await page.waitForFunction(() => document.getElementById('slot-fold-broll-color-match-value').textContent === 'Off');
    assert.equal(await page.$$eval('#slot-fold-broll-color-match .shot-tint', (tints) => tints.length), 0);
    await page.$eval('#slot-fold-broll-color-match', (fold) => fold.scrollIntoView({ block: 'center' }));
    await page.screenshot({ path: path.join(out, `${name}-broll-color-match-off.png`) });
    await page.evaluate(() => document.getElementById('step-more').click());
    await page.waitForFunction(() => document.body.dataset.screen === 'more');
    await page.click('#choice-moves');
    await page.waitForSelector('#slot-moves-punches');
    assert.equal(await page.$eval('#moves-more', (fold) => fold.hidden), false);
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    for (const mode of ['none', 'punches', 'punches_and_push']) {
      await page.$eval(`#slot-moves-${mode}`, (tile) => tile.click());
      await page.waitForFunction((mode) => document.getElementById(`slot-moves-${mode}`).getAttribute('aria-pressed') === 'true', {}, mode);
      assert.equal(await page.$eval('#main-frame', (frame) => Boolean(frame.querySelector('.zoom-picture'))), mode !== 'none');
      if (mode !== 'none') {
        const zoom = await page.$eval('#main-frame .zoom-picture', (zoom) => ({
          scale: new DOMMatrix(getComputedStyle(zoom).transform).a,
          origin: getComputedStyle(zoom).transformOrigin, animation: getComputedStyle(zoom).animationName,
          captionZoomed: Boolean(zoom.querySelector('.poster-caption')),
          image: zoom.querySelector('img').complete && zoom.querySelector('img').naturalWidth,
        }));
        assert.deepEqual(zoom, { scale: 1.5, origin: '540px 740px', animation: 'none', captionZoomed: false, image: 1080 });
      }
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(out, `${name}-move-${mode}.png`) });
    }
    await page.$eval('#moves-more', (fold) => fold.open = true);
    await page.$eval('#slot-fold-moves-zoom-strength', (fold) => fold.open = true);
    await page.$eval('#slot-moves-punch-2', (tile) => tile.click());
    await page.waitForFunction(() => document.getElementById('screen-summary').textContent === 'Moves: Punches and a slow push at 2x · all transitions');
    assert.equal(await page.$eval('#main-frame .zoom-picture', (zoom) => new DOMMatrix(getComputedStyle(zoom).transform).a), 2);
    await page.$eval('#slot-fold-moves-transitions', (fold) => fold.open = true);
    for (const look of ['whip_pan', 'cinematic_zoom', 'light_leak']) {
      const selector = `.transition_${look}`;
      await page.$eval(selector, (preview) => preview.scrollIntoView({ block: 'center' }));
      await page.waitForFunction((selector) => [...document.querySelectorAll(`${selector} img`)].every((image) => image.complete && image.naturalWidth > 0), {}, selector);
      const images = await page.$$eval(`${selector} > .transition-from, ${selector} > .transition-to`, (frames) => frames.map((frame) => ({
        ready: frame.querySelector('img').complete && frame.querySelector('img').naturalWidth > 0,
        filter: getComputedStyle(frame.querySelector('.transition-blur') || frame).filter,
        opacity: Number(getComputedStyle(frame).opacity), animation: getComputedStyle(frame).animationName,
      })));
      assert.equal(images.length, 2);
      assert.ok(images.every((image) => image.ready && image.opacity > 0 && image.opacity < 1 && image.animation === 'none'));
      if (look !== 'light_leak') assert.ok(images.every((image) => image.filter !== 'none'));
      assert.equal(await page.$eval(`#slot-switch-moves-${look}`, (input) => input.checked), true);
      await page.screenshot({ path: path.join(out, `${name}-transition-${look}.png`) });
    }
    await page.evaluate(() => document.getElementById('step-more').click());
    await page.waitForFunction(() => document.body.dataset.screen === 'more');
    await page.click('#choice-moments');
    await page.waitForFunction(() => {
      const tiles = document.getElementById('moments-tiles');
      return tiles?.children.length > 0 && tiles.querySelectorAll('.moment-person').length === tiles.children.length;
    });
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    for (const id of ['color_pop', 'background_blur', 'spotlight', 'sticker_freeze', 'outline']) {
      await page.$eval(`#moment-tile-${id}`, (tile) => tile.click());
      await page.waitForFunction(() => {
        const image = document.querySelector('#main-frame .moment-person');
        return image?.complete && image.naturalWidth === 1080 && image.naturalHeight === 1920;
      });
      assert.equal(await page.$eval(`#moment-${id}`, (input) => input.checked), false);
      assert.equal(await page.$eval('#main-frame .moment-treatment', (effect) => {
        const background = new URL(effect.querySelector('.moment-backdrop').src), cutout = new URL(effect.querySelector('.moment-person').src);
        return ['look', 'strength', 'color'].every((key) => background.searchParams.get(key) === cutout.searchParams.get(key));
      }), true);
      assert.equal(await page.$eval('#main-frame .moment-treatment', (effect) => getComputedStyle(effect).animationName), 'none');
      await page.screenshot({ path: path.join(out, `${name}-moment-${id}.png`) });
    }
    await page.$eval('#moment-outline', (input) => input.click());
    assert.equal(await page.$eval('#moment-outline', (input) => input.checked), true);
    assert.equal(await page.$eval('#moment-tile-outline .tile-label', (label) => label.textContent.includes(' · On')), true);
    assert.deepEqual(errors, []);
    await page.close();
  }
} finally {
  await browser.close();
}
