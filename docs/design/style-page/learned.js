/* Learned style validation and evidence rows. Classic script for opening from disk. */
(() => {
  'use strict';

  const SECTIONS = ['captions', 'hook', 'chapters', 'graphics', 'music', 'broll', 'grade'];
  const ORIGINAL_BROLL_FIELDS = new Set([
    'screen_recording_placement', 'clip_placement', 'photo_placement',
    'screenshot_placement', 'overlay_box', 'overlay_motion', 'avoid_tags',
  ]);
  const LABELS = {
    'captions.font': 'Font (captions)', 'hook.font': 'Font (title)', 'graphics.font': 'Font (graphics)',
    'captions.size_px': 'Size (captions)', 'captions.top_px': 'Caption offset',
    'captions.position': 'Position (captions)', 'captions.text_color': 'Captions',
    'captions.highlight_style': 'Highlight', 'captions.highlight_color': 'Highlight color',
    'captions.highlight_text_color': 'Sliding block text color',
    'captions.italic_stress': 'Italic on key words',
    'captions.max_words_per_phrase': 'Most words on screen',
    'captions.max_characters_per_phrase': 'Characters per phrase',
    'hook.style': 'Opening title', 'hook.position': 'Title position',
    'hook.duration_s': 'Title duration', 'hook.size_px': 'Title size',
    'chapters.style': 'Chapter title card', 'chapters.hold_s': 'Chapter duration',
    'hook.background_color': 'Cards (title)', 'hook.text_color': 'Card text (title)',
    'hook.accent_color': 'Title accent color', 'graphics.paper_color': 'Cards (graphics)',
    'graphics.ink_color': 'Card text (graphics)', 'graphics.accent_color': 'Card accent color',
    'graphics.hold_s': 'Card duration', 'graphics.card_width_px': 'Card width',
    'music.level': 'Music volume',
    'grade.look': 'Color grade', 'grade.strength': 'Grade strength', 'grade.color': 'Picked color',
    'grade.slider': 'Before/after slider',
    'broll.screen_recording_placement': 'Screen recordings',
    'broll.clip_placement': 'Clips', 'broll.photo_placement': 'Photos',
    'broll.screenshot_placement': 'Screenshots', 'broll.overlay_box': 'Overlay position',
    'broll.overlay_motion': 'Overlay motion', 'broll.still_push': 'Photo push',
    'broll.still_push_scale': 'Push scale', 'broll.avoid_tags': 'Tags to avoid',
    'broll.tall_framing': 'Framing behind the tall box',
  };
  const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);

  function check(value, rule, location) {
    if (Array.isArray(rule.type) && value === null && rule.type.includes('null')) return;
    if (Array.isArray(rule.type)) rule = {...rule, type: rule.type.find((kind) => kind !== 'null')};
    const valid = {
      object, array: Array.isArray,
      string: (v) => typeof v === 'string', boolean: (v) => typeof v === 'boolean',
      integer: Number.isInteger, number: (v) => typeof v === 'number' && Number.isFinite(v),
    }[rule.type];
    if (!valid || !valid(value)) throw new Error(`${location}: expected ${rule.type}`);
    if (own(rule, 'const') && value !== rule.const) throw new Error(`${location}: invalid value`);
    if (rule.enum && !rule.enum.includes(value)) throw new Error(`${location}: invalid choice`);
    if (typeof value === 'number' && (
      !Number.isFinite(value) || value < (rule.minimum ?? -Infinity) ||
      value > (rule.maximum ?? Infinity)
    )) throw new Error(`${location}: outside the allowed range`);
    if (typeof value === 'string' && (
      [...value].length < (rule.minLength ?? 0) || [...value].length > (rule.maxLength ?? Infinity) ||
      (rule.pattern && !new RegExp(`^(?:${rule.pattern})$`).test(value))
    )) throw new Error(`${location}: invalid text`);
    if (rule.type === 'array') {
      if (value.length > (rule.maxItems ?? Infinity)) {
        throw new Error(`${location}: must contain at most ${rule.maxItems} items`);
      }
      value.forEach((child, index) => check(child, rule.items, `${location}[${index}]`));
    }
    if (rule.type === 'object') {
      for (const key of rule.required || []) {
        if (!own(value, key)) throw new Error(`${location}.${key}: missing field`);
      }
      for (const [key, child] of Object.entries(value)) {
        if (!own(rule.properties, key)) throw new Error(`${location}.${key}: unknown field`);
        check(child, rule.properties[key], `${location}.${key}`);
      }
    }
  }

  function readLearned(value, schema) {
    if (!object(value) || value.schema !== 'yap-studio/taste-measurements@1') {
      throw new Error('Choose a measurements.json written by taste learn.');
    }
    const proposal = value.proposed_style;
    if (!object(proposal) || !object(proposal.base) || !object(proposal.sources)) {
      throw new Error('This measurements file has no complete proposed style.');
    }
    if (!['preset', 'file', 'schema_defaults'].includes(proposal.base.kind) ||
        typeof proposal.base.name !== 'string' || !proposal.base.name.trim()) {
      throw new Error('The style base is missing or invalid.');
    }
    if (!object(proposal.style) || proposal.style.schema !== 'yap-studio/resolved-style@1') {
      throw new Error('The proposed style has an invalid schema.');
    }
    const style = {...proposal.style};
    delete style.schema;
    check(style, schema, 'style');
    if (own(style, 'internal')) throw new Error('A learned style must omit internal settings.');
    for (const section of ['broll', 'grade']) {
      if (own(style, section)) style[section] = {...style[section]};
    }
    const rows = {measured: [], base: []};
    const leaves = new Set();
    // Older documents have no source decisions for the optional broll section.
    const legacyBroll = !Object.keys(proposal.sources).some((field) => field.startsWith('broll.'));
    const legacyGrade = !Object.keys(proposal.sources).some((field) => field.startsWith('grade.'));
    const legacyChapters = !Object.keys(proposal.sources).some((field) =>
      field.startsWith('chapters.'));
    if (legacyChapters) {
      style.chapters = {...Object.fromEntries(Object.entries(schema.properties.chapters.properties)
        .map(([key, rule]) => [key, clone(rule.default)])), ...style.chapters};
    }
    for (const section of SECTIONS) {
      if (['broll', 'grade'].includes(section) && !own(style, section)) continue;
      if (!object(style[section])) throw new Error(`style.${section}: missing section`);
      for (const key of Object.keys(schema.properties[section].properties)) {
        const field = `${section}.${key}`;
        leaves.add(field);
        const missing = !own(style[section], key);
        if (missing) {
          const rule = schema.properties[section].properties[key];
          if (!['broll', 'grade'].includes(section) || (section === 'broll' && ORIGINAL_BROLL_FIELDS.has(key)) ||
              !own(rule, 'default') || own(proposal.sources, field)) {
            throw new Error(`${field}: missing value`);
          }
          style[section][key] = clone(rule.default);
        }
        const source = (section === 'broll' && (legacyBroll || missing)) ||
          (section === 'grade' && (legacyGrade || missing)) ||
          (section === 'chapters' && legacyChapters)
          ? {source: 'base', status: null, reason: 'This choice is not measured from finished videos.'}
          : proposal.sources[field];
        if (!object(source) || !['measured', 'base'].includes(source.source) ||
            ![null, 'proposed', 'varies', 'partial', 'outside_style'].includes(source.status) ||
            !(source.reason === null || typeof source.reason === 'string') ||
            (source.source === 'measured' && (source.status !== 'proposed' || source.reason !== null)) ||
            (source.source === 'base' && (source.status === 'proposed' ||
              typeof source.reason !== 'string' || !source.reason.trim()))) {
          throw new Error(`${field}: missing or invalid source`);
        }
        rows[source.source].push({
          field, label: LABELS[field] || field, value: style[section][key], reason: source.reason,
        });
      }
    }
    if (Object.keys(proposal.sources).some((field) => !leaves.has(field))) {
      throw new Error('The style contains an unknown source field.');
    }
    return JSON.parse(JSON.stringify({style, base: proposal.base, rows}));
  }

  const clone = (value) => JSON.parse(JSON.stringify(value));
  const buyerStyle = (style) => Object.fromEntries(SECTIONS
    .filter((key) => !['broll', 'grade', 'chapters'].includes(key) || own(style, key))
    .map((key) => [key, clone(style[key])]));

  function switchPreset(state, preset) {
    const previous = clone({
      presetId: state.presetId, style: state.style, custom: state.custom,
      startStyle: state.startStyle, startName: state.startName, learned: state.learned,
    });
    const style = buyerStyle(preset.style);
    if (own(state.style, 'brand')) style.brand = clone(state.style.brand);
    const custom = {
      caption: style.captions.text_color,
      accent: style.brand?.accent_color || (style.captions.highlight_style === 'none'
        ? style.graphics.accent_color : style.captions.highlight_color),
      paper: style.hook.background_color, ink: style.hook.text_color,
    };
    return {
      previous,
      next: {...state, presetId: preset.id, style, custom,
        startStyle: clone(style), startName: preset.name, learned: null},
    };
  }

  function undoPreset(state, previous) {
    return {...state, ...clone(previous)};
  }

  const api = {readLearned, switchPreset, undoPreset};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else window.YapLearned = api;
})();
