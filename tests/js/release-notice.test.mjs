import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import * as model from '../../src/yap_studio/homepage/static/home-model.mjs';

const root = new URL('../../src/yap_studio/homepage/static/', import.meta.url);

class Element {
  constructor() { this.children = []; this.textContent = ''; }
  set innerHTML(_) { throw new Error('Notice must use textContent'); }
  setAttribute(name, value) { this[name] = value; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener() {}
}

test('Home renders and clears a quiet release notice with safe text and release notes', () => {
  const elements = new Map();
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const context = vm.createContext({...model, Node: Element, URLSearchParams,
    location: {search: ''}, window: {addEventListener() {}},
    document: {getElementById: get, addEventListener() {},
      createElement: () => new Element(), createTextNode: (text) => text},
  });
  const source = readFileSync(new URL('home.js', root), 'utf8')
    .replace(/^import \{[\s\S]*?\} from '[^']+';\s*/gmu, '')
    .replace(/refresh\(\)\.then\(schedule\);\s*$/u, '');
  vm.runInContext(source, context);
  const message = 'Yap Studio 2.3.4 is out. Say "update Yap Studio". <img src=x>';
  const url = 'https://github.com/cathrynlavery/yap-studio/releases/tag/v2.3.4';
  vm.runInContext(`state = {release_notice: ${JSON.stringify({text: message, url})}};
    renderReleaseNotice();`, context);
  const notice = get('release-notice');
  assert.equal(notice.hidden, false);
  assert.equal(notice.children[0].textContent, message);
  assert.equal(notice.children[2].textContent, 'Release notes');
  assert.equal(notice.children[2].href, url);
  const children = notice.children;
  const link = notice.children[2];
  vm.runInContext(`state = {release_notice: ${JSON.stringify({text: message, url})}};
    renderReleaseNotice();`, context);
  assert.equal(notice.children, children);
  assert.equal(notice.children[2], link);
  const updatedMessage = 'Yap Studio 3.0.0 is out. Say "update Yap Studio".';
  vm.runInContext(`state = {release_notice: ${JSON.stringify({text: updatedMessage, url})}};
    renderReleaseNotice();`, context);
  assert.equal(notice.children[0].textContent, updatedMessage);
  const updatedUrl = 'https://github.com/cathrynlavery/yap-studio/releases/tag/v3.0.0';
  vm.runInContext(`state = {release_notice: ${JSON.stringify({text: updatedMessage, url: updatedUrl})}};
    renderReleaseNotice();`, context);
  assert.equal(notice.children[2].href, updatedUrl);
  vm.runInContext('state = {release_notice: null}; renderReleaseNotice();', context);
  assert.equal(notice.hidden, true);
  assert.equal(notice.children.length, 0);
  assert.match(source.slice(source.indexOf('function render()'), source.indexOf('function claudeWorking')),
    /renderReleaseNotice\(\)/u);
  const html = readFileSync(new URL('home.html', root), 'utf8');
  assert.match(html, /id="release-notice"[^>]*hidden/u);
});
