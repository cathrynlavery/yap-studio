import test from 'node:test';
import assert from 'node:assert/strict';
import {textHookProblem} from '../../src/yap_studio/homepage/static/home-model.mjs';
import {renderTextHook} from '../../src/yap_studio/review/static/text-hook.mjs';
import {openingSlots, openingMoreSlots, changeSummary} from '../../src/yap_studio/gallery/static/choices.mjs';

test('Opening offers Text hook with three independent box colors', () => {
  const base = {name: 'Synthetic', hook: {style: 'quiet_serif'}};
  const hook = openingSlots.find((entry) => entry.id === 'text-hook');
  const selected = hook.apply(base);
  assert.equal(selected.hook.style, 'text_hook');
  assert.equal(base.hook.style, 'quiet_serif');
  const group = openingMoreSlots.find((entry) => entry.id === 'text-hook-color');
  assert.deepEqual(group.options.map((entry) => entry.name), ['White', 'Black', 'Accent']);
  for (const [index, color] of ['white', 'black', 'accent'].entries()) {
    const candidate = group.options[index].apply(selected);
    assert.equal(candidate.hook.text_hook_color, color);
    assert.equal(group.options[index].selected(candidate), true);
    assert.match(changeSummary(candidate, selected), /Opening/);
  }
});

test('home allows an empty hook and bounds typed text before server fitting', () => {
  assert.equal(textHookProblem(''), '');
  assert.equal(textHookProblem('Make it clear'), '');
  assert.match(textHookProblem('a'.repeat(81)), /shorter text/);
});

class Node {
  constructor(tag) { this.tagName = tag; this.children = []; this.hidden = false; this.value = ''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(type, fn) { this[type] = fn; }
}

test('review shows three ideas and saves the selected or typed choice as a note', async () => {
  globalThis.document = {createElement: (tag) => new Node(tag)};
  const panel = new Node('section');
  const ideas = ['Make it clear', 'Start with the hard part', 'What makes this work?'];
  const plan = {ideas, chosen: null, source: 'idea'};
  const saved = [];
  const save = async (body) => saved.push(body);
  renderTextHook(panel, plan, save);
  const find = (tag, root = panel) => root.children.flatMap((node) =>
    [ ...(node.tagName === tag ? [node] : []), ...find(tag, node) ]);
  const buttons = find('button');
  assert.deepEqual(buttons.slice(0, 3).map((node) => node.textContent), ideas);
  await buttons[1].click();
  assert.equal(find('textarea')[0].value, ideas[1]);
  await buttons[3].click();
  assert.deepEqual(saved[0], {text_hook: {text: ideas[1], source: 'idea'}});
  assert.ok(find('p').some((node) => node.textContent === ideas[1]));
  renderTextHook(panel, {...plan, chosen: null}, save);
  find('textarea')[0].value = 'Choose clearer words';
  await find('textarea')[0].input();
  await find('button').at(-1).click();
  assert.deepEqual(saved[1], {text_hook: {text: 'Choose clearer words', source: 'typed'}});
  renderTextHook(panel, null, save);
  assert.equal(panel.hidden, true);
});
