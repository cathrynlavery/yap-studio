/* Caption looks and letter widths from the engine schema. Classic script for disk use. */
(() => {
  'use strict';

  // The copy table also sets the page order. Unknown engine looks must get page copy.
  const HIGHLIGHTS = {
    moving_block: { name: 'Sliding block', note: 'A color block follows each word' },
    underline: { name: 'Underline', note: 'A line slides under each word' },
    color: { name: 'Colored word', note: 'The word being said changes color' },
    pill: { name: 'Pill', note: 'Words light up inside a soft pill' },
    gradient: { name: 'Word fill', note: 'Each word fills with color as it is spoken' },
    editorial: { name: 'Bold key word', note: 'The key word turns bold and colored' },
    slam: { name: 'Kinetic slam', note: 'One big word at a time, punched in' },
    weight_shift: { name: 'Weight shift', note: 'The line being said turns bold' },
    clip_wipe: { name: 'Clip wipe', note: 'Each word wipes in as it is said' },
    blend_difference: { name: 'Difference blend', note: 'A contrasting block inverts the word color' },
    camera_follow: { name: 'Camera follow', note: 'The view gently pans and pulls back across the phrase' },
    editorial_emphasis: { name: 'Large key word', note: 'A larger key word slides into the phrase' },
    emoji_pop: { name: 'Emoji pop', note: 'A smiley badge pops beside the key word' },
    glitch_rgb: { name: 'RGB glitch', note: 'A brief color split flicks across each word' },
    gradient_fill: { name: 'Gradient fill', note: 'A moving gradient colors each spoken word' },
    highlight: { name: 'Word background', note: 'A rounded color block opens behind each spoken word' },
    matrix_decode: { name: 'Matrix decode', note: 'Scrambled glyphs flicker behind readable words' },
    neon_accent: { name: 'Neon accent', note: 'A larger key word glows with a slight tilt' },
    neon_glow: { name: 'Neon glow', note: 'Each spoken word lights up with a neon glow' },
    parallax_layers: { name: 'Parallax layers', note: 'A colored text echo moves behind each word' },
    particle_burst: { name: 'Particle burst', note: 'Small particles burst beside the key word' },
    pill_karaoke: { name: 'Rounded karaoke', note: 'Words gradually darken inside a rounded rectangle' },
    texture: { name: 'Text texture', note: 'A moving texture fills the words' },
    giant: { name: 'Giant captions', note: 'The line oversized in a solid block' },
    none: { name: 'No highlight', note: 'Every word looks the same' },
  };

  function highlightLooks(schema) {
    const values = schema.properties.captions.properties.highlight_style.enum;
    for (const id of values) {
      if (!Object.prototype.hasOwnProperty.call(HIGHLIGHTS, id)) {
        throw new Error(`Caption highlight has no page copy: ${id}`);
      }
    }
    return Object.entries(HIGHLIGHTS).filter(([id]) => values.includes(id))
      .map(([id, copy]) => ({ id, ...copy }));
  }

  function widthChoices(schema) {
    const { minimum, maximum } = schema.properties.captions.properties.width_scale;
    return [
      { id: 'normal', name: 'Normal', value: 1, note: 'Full width letters' },
      { id: 'condensed', name: 'Condensed', value: 0.85, note: 'Slightly narrower letters' },
      { id: 'narrow', name: 'Narrow', value: 0.7, note: 'Narrower letters' },
    ].map((choice) => ({ ...choice, value: Math.min(maximum, Math.max(minimum, choice.value)) }));
  }

  function nearestWidth(value, choices) {
    if (!choices.length) throw new Error('Letter width choices are empty.');
    return choices.reduce((best, choice) => (
      Math.abs(choice.value - value) < Math.abs(best.value - value) ? choice : best
    ));
  }

  function mix(a, b, amount) {
    const rgb = (hex) => [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16));
    const end = rgb(b);
    return `#${rgb(a).map((channel, i) => Math.round(channel + (end[i] - channel) * amount)
      .toString(16).padStart(2, '0')).join('')}`.toUpperCase();
  }

  const pillText = (textColor, highlightColor) => mix(highlightColor, textColor, 0.45);
  function luminance(hex) {
    const [r, g, b] = [1, 3, 5].map((at) => {
      const c = parseInt(hex.slice(at, at + 2), 16) / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  function contrast(a, b) {
    const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  }
  function pillContrast(textColor, highlightColor) {
    return {
      spoken: contrast(textColor, highlightColor),
      muted: contrast(pillText(textColor, highlightColor), highlightColor),
    };
  }
  const hasUpcomingWords = (phrases) => phrases.some((phrase) => phrase.ids.length > 1);
  const gradientTint = (textColor, highlightColor) => mix(textColor, highlightColor, 0.65);
  const wordFill = (time, start, end) => Math.min(1, Math.max(0,
    (time - start) / Math.max(1 / 30, end - start)));

  const api = { highlightLooks, widthChoices, nearestWidth, pillText, pillContrast, hasUpcomingWords,
    luminance, contrast, gradientTint, wordFill };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else window.YapLooks = api;
})();
