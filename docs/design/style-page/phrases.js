/* Deterministic caption splitting shared with the Python engine. Classic script for file://. */
(() => {
  'use strict';

  const PUNCTUATION = `.,!?;:"'()[]{}<>…“”‘’«»—–-`;
  const CLOSERS = `"')]}>'”’»`;
  const FORWARD_WORDS = new Set((
    "a an the to of in on at for with from by and or but so my your our their his her its " +
    "this that these those is are was were be i we you they it's i'm i've i'll we're " +
    "you're they're"
  ).split(' '));
  const CLAUSE_WORDS = new Set((
    'and but so because which that when while if then or who where although though unless ' +
    'until since'
  ).split(' '));
  const STOP_WORDS = new Set([
    ...FORWARD_WORDS, ...CLAUSE_WORDS,
    ...('just really very also not no yes have has had do does did can will would could should ' +
      'there here what about into out up all some any been being than them me us him').split(' '),
  ]);

  function trimEnd(text, punctuation) {
    let end = text.length;
    while (end > 0 && punctuation.includes(text[end - 1])) end -= 1;
    return text.slice(0, end);
  }

  function stripped(text) {
    const tail = trimEnd(text, PUNCTUATION);
    let start = 0;
    while (start < tail.length && PUNCTUATION.includes(tail[start])) start += 1;
    return tail.slice(start).toLowerCase();
  }

  const ending = (text) => trimEnd(text, CLOSERS);
  const endsWith = (text, marks) => marks.includes(ending(text).slice(-1)) && ending(text) !== '';
  const milliseconds = (seconds) => Math.floor(seconds * 1000 + 0.5) / 1000;

  function segmentBoundary(previous, word, forcedBreaks) {
    return endsWith(previous.text, '.!?') || word.start - previous.end > 1.0 ||
      (word.source_ids || [word.id]).some((id) => forcedBreaks.includes(id));
  }

  function displayWords(words, wordFixes) {
    const visible = words.map((word, order) => ({...word, order})).filter((word) => !word.hidden)
      .sort((a, b) => a.start - b.start || a.order - b.order);
    const fixes = Object.entries(wordFixes).map(([key, replacement]) => ({
      parts: key.trim() ? key.trim().split(/\s+/).map(stripped) : [], replacement,
    })).filter((fix) => fix.parts.length).sort((a, b) => b.parts.length - a.parts.length);
    const normalized = visible.map((word) => stripped(word.text));
    const display = [];
    let i = 0;
    while (i < visible.length) {
      let count = 1;
      let text = visible[i].text;
      for (const {parts, replacement} of fixes) {
        if (i + parts.length <= visible.length && parts.every((part, k) => part === normalized[i + k])) {
          if (visible.slice(i + 1, i + parts.length).some((word, k) =>
            visible[i + k].segment !== word.segment ||
            segmentBoundary(visible[i + k], word, []))) continue;
          count = parts.length;
          const lastText = visible[i + count - 1].text;
          text = replacement + lastText.slice(trimEnd(lastText, PUNCTUATION).length);
          break;
        }
      }
      const matched = visible.slice(i, i + count);
      display.push({
        id: matched[0].id, text, start: matched[0].start, end: matched[matched.length - 1].end,
        source_ids: matched.map((word) => word.id),
      });
      i += count;
    }
    return display;
  }

  function segments(words, forcedBreaks) {
    const result = [];
    for (const word of words) {
      const current = result[result.length - 1];
      const previous = current && current[current.length - 1];
      if (!previous || segmentBoundary(previous, word, forcedBreaks)) result.push([]);
      result[result.length - 1].push(word);
    }
    return result;
  }

  function cost(words, start, end, maxWords) {
    const count = end - start;
    let value = 10 + 2 * Math.max(0, Math.min(maxWords, 3) - count);
    if (count === 1 && words.length !== 1) value += 30;
    const last = words[end - 1];
    if (end !== words.length && FORWARD_WORDS.has(stripped(last.text))) value += 40;
    if (endsWith(last.text, ',;:-–—')) value -= 15;
    for (let i = start; i < end - 1; i += 1) {
      if (endsWith(words[i].text, ',;:')) value += 25;
      if (words[i + 1].start - words[i].end > 0.3) value += 20;
    }
    if (start !== 0 && CLAUSE_WORDS.has(stripped(words[start].text))) value -= 8;
    return value;
  }

  function split(words, maxWords, maxChars) {
    const n = words.length;
    const costs = Array(n + 1).fill(0);
    const breaks = Array(n).fill(n);
    for (let start = n - 1; start >= 0; start -= 1) {
      let bestCost = Infinity;
      let characters = 0;
      for (let end = start + 1; end <= Math.min(n, start + maxWords); end += 1) {
        characters += [...words[end - 1].text].length + (end > start + 1 ? 1 : 0);
        if (characters > maxChars && end > start + 1) break;
        const candidate = cost(words, start, end, maxWords) + costs[end];
        // Suffixes already win their ties, so prefer the longer first phrase here.
        if (candidate < bestCost || (candidate === bestCost && end > breaks[start])) {
          bestCost = candidate;
          breaks[start] = end;
        }
      }
      costs[start] = bestCost;
    }
    const result = [];
    for (let start = 0; start < n; start = breaks[start]) result.push(words.slice(start, breaks[start]));
    return result;
  }

  function stress(words, stressIds) {
    for (const word of words) {
      if (word.source_ids.some((id) => stressIds.includes(id))) return word.id;
    }
    let bestId = null;
    let bestScore = -Infinity;
    for (const word of words) {
      const letters = (word.text.match(/[A-Za-z0-9]/g) || []).length;
      if (letters <= 2 || STOP_WORDS.has(stripped(word.text))) continue;
      const score = (word.end - word.start) / letters;
      if (score > bestScore) { bestId = word.id; bestScore = score; }
    }
    return bestId;
  }

  /** Apply text fixes, integer split costs, stress picks and nonoverlapping phrase timing. */
  function buildPhrases(words, captions, options = {}) {
    const maxWords = captions.max_words_per_phrase;
    const maxChars = captions.max_characters_per_phrase;
    for (const limit of [maxWords, maxChars]) {
      if (!Number.isInteger(limit) || limit <= 0) throw new RangeError('caption phrase limits must be positive integers');
    }
    const display = displayWords(words, options.word_fixes || {});
    const groups = segments(display, options.forced_breaks || []).flatMap((segment) => split(segment, maxWords, maxChars));
    return groups.map((group, index) => {
      const start = group[0].start;
      const lastStart = group[group.length - 1].start;
      let end = Math.max(lastStart + 1.4, start + 0.35);
      if (index + 1 < groups.length) {
        const nextStart = groups[index + 1][0].start;
        end = Math.max(Math.min(nextStart - 0.06, lastStart + 1.4), start + 0.35);
        end = Math.min(end, nextStart);
      }
      return {
        ids: group.map((word) => word.id), text: group.map((word) => word.text).join(' '),
        start: milliseconds(start), end: milliseconds(end), stress: stress(group, options.stress_ids || []),
      };
    }).filter((phrase) => phrase.end > phrase.start);
  }

  const api = {buildPhrases};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else window.YapPhrases = api;
})();
