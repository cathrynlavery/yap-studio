// Style field changes shared by the page and its small Node tests.
export const labels = {
  made: "Made by Yap Studio",
  typing: "Typing", "message-ping": "Message ping", "camera-shutter": "Camera shutter",
  hard: "Hard cut",
  punch: "Punch",
  smooth: "Smooth (punch on big moves)",
  step_in: "Step in",
  moving_block: "Sliding block",
  underline: "Underline",
  color: "Colored word",
  pill: "Pill",
  gradient: "Word fill",
  editorial: "Bold key word",
  slam: "Kinetic slam",
  weight_shift: "Weight shift",
  clip_wipe: "Clip wipe",
  blend_difference: "Difference blend",
  camera_follow: "Camera follow",
  editorial_emphasis: "Large key word",
  emoji_pop: "Emoji pop",
  glitch_rgb: "RGB glitch",
  gradient_fill: "Gradient fill",
  highlight: "Word background",
  matrix_decode: "Matrix decode",
  neon_accent: "Neon accent",
  neon_glow: "Neon glow",
  parallax_layers: "Parallax layers",
  particle_burst: "Particle burst",
  pill_karaoke: "Rounded karaoke",
  texture: "Text texture",
  giant: "Giant captions",
  none: "No highlight",
  paper_title: "Paper card",
  quiet_serif: "Quiet title",
  bold_band: "Bold band",
  calm_card: "Calm card",
  lockup: "Lockup",
  write_on: "Write on",
  headline_slam: "Headline slam",
  shard_reveal: "Shard reveal",
  textured: "Textured",
  shimmer: "Shimmer",
  stagger: "Stagger",
  morph: "Morph",
  typewriter: "Typewriter",
  video_letters: "Video in letters",
  kinetic: "Kinetic hook",
  text_hook: "Text hook",
  white: "White",
  black: "Black",
  accent: "Accent",
  title_card: "Chapter cards",
  paper: "Paper",
  minimal: "Minimal",
  dark: "Dark",
  bright: "Bright",
  text_tag: "Text tag",
  checklist: "Checklist",
  steps: "Steps",
  flow: "Flow",
  terminal: "Terminal",
  stat: "Stat",
  image_card: "Image card",
  image_compare: "Image compare",
  transfer: "Transfer",
  chart: "Chart",
  quote: "Quote",
  gravity_drop: "Gravity drop",
  progress_stat: "Progress",
  big_number: "Big number behind you",
  specs_list: "Specs list",
  callout_circle: "Callout circle",
  name_tag: "Name tag",
  success_check: "Success check",
  edit_timeline: "Edit timeline",
  storyboard: "Storyboard",
  jiggle_delete: "Jiggle delete",
  file_drop: "File drop",
  video_card: "Video card",
  prompt_bar: "Prompt bar",
  comment_cta: "Comment CTA",
  comment_drop: "Comment drop",
  icon_strike: "Icon strike",
  playing_grid: "Playing grid",
  reel_feed: "Reel feed",
  choices: "Choices",
  film: "Film look", eyedropper: "Eyedropper", box: "Box", band: "Band",
  bottom: "Bottom", middle: "Middle", fixed: "Fixed position", avoid_face: "Avoid faces",
  slide: "Slide", rise: "Rise", spoken: "Word by word", headroom: "Above your head", center: "Center", top: "Top",
  full_frame: "Full screen", overlay: "In a box", split: "Split", green_screen: "Behind you",
  tall_behind: "Tall, behind your head", scroll: "Scroll",
  keep: "Keep your framing", shrink: "Shrink you in front of the page",
  side_left: "Left side", side_right: "Right side", drift: "Drift", in: "Push in", out: "Pull out",
  off: "Off", on: "On", ask: "Ask first", auto: "Automatic", screen: "Screen color", person: "Person",
  full: "Full screen", corner_left: "Bottom left", corner_right: "Bottom right",
  punches: "Punches", punches_and_push: "Punches and a slow push",
  whip_pan: "Whip", cinematic_zoom: "Zoom", light_leak: "Light leak",
  clean: "Clean", broadcast: "Broadcast", warm: "Warm", gentle: "Gentle", strong: "Strong",
  level: "Steady level", words: "Follow words", color_pop: "Color pop",
  background_blur: "Background blur", spotlight: "Spotlight", sticker_freeze: "Sticker freeze",
  outline: "Accent outline", clone_flash: "Clone flash",
  boxes: "Boxes", tags: "Tags",
  together: "Together", on_cue: "Rows on cue", on_cue_glow: "Rows on cue, glowing",
  streak: "Motion streak",
};
export const wordModes = {
  one: { name: "One word", words: 1, chars: 16 },
  few: { name: "A few words", words: 3, chars: 18 },
  phrase: { name: "Short phrase", words: 5, chars: 24 },
  sentence: { name: "Full sentence", words: 8, chars: 40 },
};
export const copy = (value) => structuredClone(value);
export const gradeLooks = [["none", "None"], ["film", "Film look"], ["eyedropper", "Eyedropper"]];
export function withGrade(style, look, strength = style.grade?.strength ?? 1, color = style.grade?.color) {
  if (!gradeLooks.some(([value]) => value === look)) throw new Error("Choose None, Film look or Eyedropper");
  if (!Number.isFinite(strength) || strength < 0 || strength > 1) throw new Error("Use strength from 0 to 1");
  if (look === "eyedropper" && !validHex(color)) throw new Error("Pick a color like #1E6BD6");
  const result = copy(style);
  result.grade = { ...(style.grade?.slider !== undefined ? { slider: style.grade.slider } : {}),
    look, strength, ...(color !== undefined ? { color: color?.toUpperCase() ?? null } : {}) };
  return result;
}
export function stillUrl(preview, style, explicitLook = false) {
  const grade = style.grade ?? { look: "none", strength: 1 };
  if (grade.look === "none" && !explicitLook) return preview.still;
  const [path, query = ""] = preview.still.split("?");
  const params = new URLSearchParams(query);
  params.set("look", grade.look);
  params.set("strength", grade.strength ?? 1);
  if (grade.look === "eyedropper") params.set("color", grade.color);
  if (grade.look !== "none" && grade.strength !== 0 && grade.slider) {
    params.set("slider", "1");
    params.set("text_color", style.captions?.text_color ?? "#FFFFFF");
    params.set("ink_color", style.graphics?.ink_color ?? "#222222");
  }
  return `${path}?${params}`;
}
export function cutoutUrl(preview, style) {
  if (!preview.cutout) return null;
  const grade = style.grade && { ...style.grade, slider: false };
  return stillUrl({ still: preview.cutout }, { grade });
}
export function coverPoint(bounds, width, height, clientX, clientY) {
  const scale = Math.max(bounds.width / width, bounds.height / height);
  const clamp = (value) => Math.max(0, Math.min(1, value));
  return {
    x: clamp((clientX - bounds.left + (width * scale - bounds.width) / 2) / scale / width),
    y: clamp((clientY - bounds.top + (height * scale - bounds.height) / 2) / scale / height),
  };
}
export function samplePatch(width, height, x, y) {
  const cx = Math.min(width - 1, Math.round(x * width)), cy = Math.min(height - 1, Math.round(y * height));
  const left = Math.max(0, cx - 3), top = Math.max(0, cy - 3);
  return { x: left, y: top, width: Math.min(width, cx + 4) - left, height: Math.min(height, cy + 4) - top };
}
export function patchColor(rgba) {
  const sum = [0, 0, 0];
  for (let i = 0; i < rgba.length; i += 4) for (let c = 0; c < 3; c++) sum[c] += rgba[i + c];
  return `#${sum.map((value) => Math.round(value / (rgba.length / 4)).toString(16).padStart(2, "0")).join("")}`.toUpperCase();
}
export const startingName = (style, fromLibrary) => fromLibrary ? style.name : "My style";
export function previewNote(preview) {
  return preview.message || (preview.source === "take"
    ? "Your newest take · your first sentence"
    : "Sample still · your take will appear here");
}
export function placementNote(style, preview) {
  if (style.captions.placement === "fixed") return "Captions stay at your chosen position.";
  if (!preview.face_boxes?.length) {
    return "Face placement is unavailable for this preview. Captions will avoid measured faces in your rendered video.";
  }
  return "Choose a preferred position. This preview uses faces measured in the still; your video checks faces throughout each phrase.";
}
export function mainPosterTime(layers) {
  const captions = layers.find((layer) => layer.name === "captions");
  const title = layers.find((layer) => layer.name === "title");
  const at = captions?.poster_s ?? layers[0]?.poster_s ?? layers[0]?.start_s ?? 0;
  if (title && (at < Math.min(0.4, title.duration_s / 2)
    || at >= title.duration_s - Math.min(0.25, title.duration_s / 2))) {
    return title.poster_s ?? title.duration_s / 2;
  }
  return at;
}
export function ensureFontOption(select, value) {
  if (value && ![...select.options].some((option) => option.value === value)) {
    const option = select.ownerDocument.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
  select.value = value;
}
export function create404Retry(fetchResource, refresh) {
  let retried = false;
  return async (url, known404 = false) => {
    if (retried) return false;
    if (!known404 && (await fetchResource(url, { method: "HEAD" })).status !== 404) return false;
    retried = true;
    await refresh();
    return true;
  };
}
export function frameWindow(frame, active = true) {
  return active && frame?.isConnected ? frame.contentWindow ?? null : null;
}
export function tileWindow(layer, frame = { width: 1080, height: 1920 }, host = { width: 9, height: 16 }, anchorY = 0.5) {
  const width = Math.min(900, frame.width);
  const height = Math.min(width * host.height / host.width, frame.height);
  const clamp = (value, maximum) => Math.max(0, Math.min(maximum, value));
  return {
    x: clamp(layer.x + layer.width / 2 - width / 2, frame.width - width),
    y: clamp(layer.y + layer.height / 2 - height * anchorY, frame.height - height),
    width,
    height,
  };
}
export function withWords(style, mode) {
  if (!wordModes[mode]) throw new Error("Unknown words option");
  const result = copy(style);
  const choice = wordModes[mode];
  Object.assign(result.captions, {
    max_words_per_phrase: choice.words,
    max_characters_per_phrase: choice.chars,
  });
  return result;
}
export function wordsMode(style) {
  const words = style.captions.max_words_per_phrase;
  return words <= 1 ? "one" : words <= 3 ? "few" : words <= 6 ? "phrase" : "sentence";
}
export function validHex(value) {
  return /^#[0-9A-Fa-f]{6}$/.test(value);
}
export function withBrand(style, field, value) {
  if (!["captions_font", "titles_font", "cards_font", "accent_color"].includes(field)) {
    throw new Error("Unknown brand field");
  }
  if (field === "accent_color" && !validHex(value)) throw new Error("Use #RRGGBB");
  const result = copy(style);
  result.brand ??= {};
  result.brand[field] = value;
  return result;
}
export function kindLabel(state, kind) {
  return state.card_kinds[kind]?.registry ? state.card_kinds[kind].label : labels[kind];
}
export function kindOn(style, kind, optIn = false) {
  const graphics = style.graphics ?? {};
  return optIn ? (graphics.card_kinds_on ?? []).includes(kind)
    : !(graphics.card_kinds_off ?? []).includes(kind);
}
export function withKind(style, kind, enabled, optIn = false) {
  const result = copy(style);
  result.graphics ??= {};
  if (optIn) {
    const on = new Set(result.graphics.card_kinds_on ?? []);
    if (enabled) on.add(kind);
    else on.delete(kind);
    result.graphics.card_kinds_on = [...on].sort();
    return result;
  }
  const off = new Set(result.graphics.card_kinds_off ?? []);
  if (enabled) off.delete(kind);
  else off.add(kind);
  result.graphics.card_kinds_off = [...off].sort();
  return result;
}
export function withPreset(style, preset, musicLevel) {
  const result = copy(preset);
  if (musicLevel !== undefined) result.music = { ...result.music, level: musicLevel };
  return result;
}
export const choosePreset = (style, preset, name) => ({
  style: withPreset(style, preset), name,
});
export function check(value, rule, location = "style", errors = []) {
  if (Array.isArray(rule.type) && value === null && rule.type.includes("null")) return errors;
  const type = Array.isArray(rule.type) ? rule.type.find((kind) => kind !== "null") : rule.type;
  const valid = {
    object: (v) => v && typeof v === "object" && !Array.isArray(v),
    array: Array.isArray,
    string: (v) => typeof v === "string",
    boolean: (v) => typeof v === "boolean",
    integer: Number.isInteger,
    number: (v) => typeof v === "number" && Number.isFinite(v),
  }[type](value);
  if (!valid) {
    errors.push(`${location}: expected ${type}`);
    return errors;
  }
  if ("const" in rule && value !== rule.const) errors.push(`${location}: must be ${rule.const}`);
  if (rule.enum && !rule.enum.includes(value)) errors.push(`${location}: unknown choice`);
  if (type === "string") {
    if (value.length < (rule.minLength ?? 0) || value.length > (rule.maxLength ?? Infinity)) {
      errors.push(`${location}: length`);
    }
    if (rule.pattern && !new RegExp(rule.pattern).test(value)) {
      errors.push(`${location}: invalid value`);
    }
  }
  if (type === "number" || type === "integer") {
    if (value < (rule.minimum ?? -Infinity) || value > (rule.maximum ?? Infinity)) {
      errors.push(`${location}: range`);
    }
  }
  if (type === "array") {
    if (value.length > (rule.maxItems ?? Infinity)) errors.push(`${location}: too many`);
    value.forEach((item, index) => check(item, rule.items, `${location}[${index}]`, errors));
    if (rule.uniqueItems && new Set(value).size !== value.length) {
      errors.push(`${location}: duplicate`);
    }
  }
  if (type === "object") {
    if (value.look === "eyedropper" && !validHex(value.color)) errors.push(`${location}.color: pick a color`);
    for (const key of rule.required ?? []) {
      if (!(key in value)) errors.push(`${location}.${key}: missing`);
    }
    for (const [key, item] of Object.entries(value)) {
      if (!(key in rule.properties)) errors.push(`${location}.${key}: unknown`);
      else check(item, rule.properties[key], `${location}.${key}`, errors);
    }
  }
  return errors;
}

export function fontFamily(name, available) {
  const prefix = available.bundled.includes(name) ? "Yap Studio "
    : available.creator.includes(name) ? "Yap Studio User " : "";
  return `${JSON.stringify(prefix + name)}, sans-serif`;
}
// A slot needs an id, name, note, apply(style) and selected(style).
// An optional requires pair names a schema path and a value it must accept.
// A switch passes its on or off value as apply's second argument.
// type: "switch" makes a closed group with note as the switch label.
// switches holds several switch rows in one group. value(style) names their state.
// preview names a drawing or holds a live tile request. previewCurrent uses its current state.
// apply receives the schema as its third argument; selected receives it as its second.
export const captionLookSlots = [{
  id: "giant", name: "Giant", note: "Big words in a solid block",
  requires: ["captions.highlight_style", "giant"],
  apply: (style) => {
    const result = copy(style); result.captions.highlight_style = "giant";
    result.captions.camera_3d = false;
    return result;
  },
  selected: (style) => captionLook(style) === "giant",
}, {
  id: "camera_3d", name: "3D camera", note: "Words in depth, camera moves",
  requires: ["captions.camera_3d", true], preview: "camera_3d",
  apply: (style) => withCaptionLook(style, "camera_3d"),
  selected: (style) => captionLook(style) === "camera_3d",
}];
export const openingSlots = [{
  id: "kinetic", name: "Kinetic hook", note: "Big words land as you say them",
  requires: ["hook.style", "kinetic"],
  apply: (style) => {
    const result = copy(style); result.hook.style = "kinetic";
    return result;
  },
  selected: (style) => style.hook.style === "kinetic",
}];
export const topTextAnswers = [
  ["none", "None", "Nothing at the top"],
  ["opening", "Just the opening", "A title for the first few seconds, in your title look"],
  ["whole", "The whole video", 'A short label like "Security rant"'],
];
export function topTextChoice(style) {
  if (!style.hook?.answered) return null;
  return style.hook.style === "none" ? "none" : style.hook.style === "text_hook" ? "whole" : "opening";
}
export function withTopText(style, answer, titleLook = "quiet_serif") {
  if (!topTextAnswers.some(([id]) => id === answer)) throw new Error("Choose when text appears at the top");
  const result = copy(style);
  result.hook.answered = true;
  result.hook.style = answer === "none" ? "none" : answer === "whole" ? "text_hook"
    : !["none", "text_hook"].includes(style.hook.style) ? style.hook.style
    : !["none", "text_hook"].includes(titleLook) ? titleLook : "quiet_serif";
  return result;
}
export const openingMoreSlots = [{
  id: "text-hook-color", name: "Label color",
  options: ["white", "black", "accent"].map((color) => ({
    id: color, name: color[0].toUpperCase() + color.slice(1),
    requires: ["hook.text_hook_color", color],
    apply: (style) => ({...style, hook: {...style.hook, style: "text_hook", text_hook_color: color}}),
    selected: (style) => (style.hook.text_hook_color ?? "white") === color,
  })),
}, {
  id: "chapter", name: "Chapter cards", type: "switch", preview: { tab: "chapter" },
  note: "A numbered title over you where each new part starts",
  requires: ["chapters.style", "title_card"],
  apply: (style, on) => {
    const result = copy(style);
    result.chapters = { ...result.chapters, style: on ? "title_card" : "none" };
    return result;
  },
  selected: (style) => style.chapters?.style === "title_card",
}];
export const cardGroupSlots = [{
  id: "flow-look", name: "Flow look",
  options: [
    ["boxes", "Boxes", "Drawn boxes in a card"],
    ["tags", "Tags", "Stacked tags with an arrow"],
  ].map(([id, name, note]) => ({
    id, name, note, card: "flow", requires: ["graphics.flow_look", id],
    apply: (style) => withFlowLook(style, id),
    selected: (style) => (style.graphics?.flow_look ?? "boxes") === id,
  })),
}, {
  id: "specs-rows", name: "Specs list rows",
  options: [
    ["together", "Together", "All rows at once"],
    ["on_cue", "Rows on cue", "Each row appears as you say it"],
    ["on_cue_glow", "Rows on cue, glowing", "Bare text that glows in your accent color"],
  ].map(([value, name, note]) => ({
    id: value, name, note, card: "specs_list", requires: ["graphics.specs_rows", value],
    apply: (style) => ({ ...style, graphics: { ...style.graphics, specs_rows: value } }),
    selected: (style) => (style.graphics?.specs_rows ?? "together") === value,
  })),
}, {
  id: "card-motion", name: "Card motion",
  note: "Cards that fly in leave a short trail; number cards and full-screen cards enter as usual.",
  options: [["none", "None"], ["streak", "Motion streak"]].map(([id, name]) => ({
    id, name, card: "quote", requires: ["graphics.card_motion", id],
    apply: (style) => ({ ...style, graphics: { ...style.graphics, card_motion: id } }),
    selected: (style) => (style.graphics?.card_motion ?? "none") === id,
  })),
}];
export function withFlowLook(style, look) {
  if (!["boxes", "tags"].includes(look)) throw new Error("Unknown Flow look");
  const result = copy(style);
  (result.graphics ??= {}).flow_look = look;
  return result;
}
export const brollLayoutSlots = [{
  id: "split", name: "Split", note: "The shot on top, you below",
  requires: ["broll.clip_placement", "split"],
  apply: (style, _on, schema) => withBrollLayout(style, "split", schema),
  selected: (style, schema) => brollLayout(style, schema) === "split",
}];
export const brollMoreSlots = [{
  id: "video-letters", name: "Video in letters", type: "switch", preview: "video_letters",
  note: "Fill a big word with your footage on a key line",
  requires: ["broll.emphasis_on", ["video_letters"]],
  apply: (style, on) => withEmphasis(style, "video_letters", on),
  selected: (style) => Boolean(style.broll?.emphasis_on?.includes("video_letters")),
}, {
  id: "color-match", name: "Match color to me", type: "switch", preview: "color_match", previewCurrent: true,
  note: "Shift each shot toward your take's color", hint: "Your face is never touched. On by default.",
  requires: ["broll.color_match", "on"],
  apply: (style, on) => withBrollField(style, "color_match", on ? "on" : "off"),
  selected: (style) => (style.broll?.color_match ?? "on") === "on",
}];
export const transitionLooks = [
  ["whip_pan", "Whip", "A fast sideways blur into the next shot"],
  ["cinematic_zoom", "Zoom", "Pushes through into the next shot"],
  ["light_leak", "Light leak", "A warm wash of light across the cut"],
].map(([id, name, note]) => ({
  id, name, note, preview: `transition_${id}`, requires: ["moves.transitions", [id]],
  apply: (style, on) => withTransition(style, id, on),
  selected: (style) => (style.moves?.transitions ?? transitionLooks.map((entry) => entry.id)).includes(id),
}));
export const movesSlots = [...[
  ["none", "No zooms", "The frame stays put"],
  ["punches", "Punches", "Quick push in on a key word, then out"],
  ["punches_and_push", "Punches + slow push", "Plus one slow push on a longer line"],
].map(([id, name, note]) => ({
  id, name, note, requires: ["moves.zoom_mode", id],
  apply: (style) => withZoomMode(style, id),
  selected: (style) => (style.moves?.zoom_mode ?? "none") === id,
})), {
  id: "transitions", name: "Transitions", type: "switch", switches: transitionLooks,
  requires: ["moves.transitions", []], value: transitionsSummary,
}];
export const zoomStrengths = [1.5, 2].map((value) => ({
  id: `punch-${value}`, name: `${value}x`, note: "", requires: ["moves.punch_scale", value],
  apply: (style) => withPunchScale(style, value),
  selected: (style) => (style.moves?.punch_scale ?? 1.5) === value,
}));
export const jumpCutOptions = [
  ["hard", "Hard cut", "The cut stays as it is"],
  ["punch", "Punch", "A quick push in on every jump"],
  ["smooth", "Smooth (punch on big moves)", "Small jumps glide, big ones punch in"],
  ["step_in", "Step in", "A closer framing on each cut, then back wide"],
].map(([id, name, note]) => ({
  id, name, note, requires: ["moves.jump_cuts", id], preview: `jump_${id}`,
  apply: (style) => withJumpCuts(style, id),
  selected: (style) => (style.moves?.jump_cuts ?? "hard") === id,
}));
export const movesMoreSlots = [{
  id: "jump-cuts", name: "Jump cuts", note: "What happens where a pause or a retake was cut out.",
  requires: ["moves.jump_cuts", "hard"], options: jumpCutOptions,
  value: (style) => jumpCutOptions.find((entry) => entry.selected(style))?.name,
}, {
  id: "opening-push", name: "Push in on the first line", type: "switch", preview: "opening_push",
  note: "Open on a steady push in to your face, then cut wide", requires: ["moves.opening_push", true],
  apply: (style, on) => withOpeningPush(style, on),
  selected: (style) => Boolean(style.moves?.opening_push),
}];
export const momentsSlots = [
  ["color_pop", "Color pop", "The world goes gray, you stay in color"],
  ["background_blur", "Background blur", "The room softens behind you"],
  ["spotlight", "Spotlight", "The room dims, light falls on you"],
  ["sticker_freeze", "Sticker freeze", "You freeze with a white outline and shadow"],
  ["outline", "Outline glow", "A glow in your highlight color traces you"],
  ["clone_flash", "Clone flash", "Copies of you pop in beside you"],
].map(([id, name, note]) => ({
  id, name, note, requires: ["moments.styles", [id]],
  apply: (style, on) => withMoment(style, id, on),
  selected: (style) => Boolean(style.moments?.styles?.includes(id)),
}));
export function keyedScreen(style, preview, videoType) {
  if (style?.broll?.cutout === "person") return null;
  if (style?.broll?.cutout === "screen") return "green or blue";
  if (preview?.screen_checked) return preview.screen ?? null;
  return videoType === "green_screen" ? "green or blue" : null;
}
export function momentBlock(entryId, style, preview, videoType) {
  if (entryId !== "clone_flash") return null;
  const screen = keyedScreen(style, preview, videoType);
  return screen ? `Needs a take filmed without a ${screen} screen.` : null;
}
export const screens = [
  { id: "font", name: "Font", optional: false },
  { id: "captions", name: "Captions", optional: false },
  { id: "colors", name: "Colors", optional: false },
  { id: "top-text", name: "Text at the top", optional: false },
  { id: "grade", name: "Grade", optional: true },
  { id: "opening", name: "Opening", optional: true },
  { id: "cards", name: "Cards", optional: true },
  { id: "broll", name: "B-roll", optional: true },
  { id: "moves", name: "Moves", optional: true },
  { id: "moments", name: "Moments", optional: true },
  { id: "sound", name: "Sound", optional: true },
];
export function optionAvailable(entry, schema) {
  if (!entry.requires) return true;
  const [path, value] = entry.requires;
  const rule = path.split(".").reduce((node, key) =>
    node?.properties && Object.hasOwn(node.properties, key) ? node.properties[key] : undefined, schema);
  return Boolean(rule && check(value, rule).length === 0);
}
export function screenAvailable(id, schema, slots) {
  if (!screens.some((screen) => screen.id === id)) return false;
  if (id !== "moves" && id !== "moments") return true;
  const entries = Array.isArray(slots) ? slots : slots?.[id] ?? (id === "moves" ? movesSlots : momentsSlots);
  return entries.some((entry) => !schema || optionAvailable(entry, schema));
}
export function availableSteps(schema, slots) {
  return [["looks", "Looks"], ...screens.filter((screen) => !screen.optional)
    .map(({ id, name }) => [id, name]), ["more", "More choices"],
  ...screens.filter((screen) => screen.optional && screenAvailable(screen.id, schema, slots))
    .map(({ id, name }) => [id, name])];
}
export const steps = availableSteps();
export function stepFromHash(hash, schema, slots) {
  const value = hash.replace(/^#/, "");
  const legacy = { start: "looks", templates: "looks", words: "captions", highlight: "captions", title: "opening" };
  const id = Object.hasOwn(legacy, value) ? legacy[value] : value;
  if (screens.some((screen) => screen.optional && screen.id === id) && !screenAvailable(id, schema, slots)) return "more";
  return availableSteps(schema, slots).some(([step]) => step === id) ? id : "looks";
}
export function adjacentStep(step, direction, schema, slots) {
  const visible = availableSteps(schema, slots);
  const index = visible.findIndex(([id]) => id === step);
  return visible[Math.max(0, Math.min(visible.length - 1, index + direction))][0];
}

export function captionLook(style) {
  if (style.captions.camera_3d) return "camera_3d";
  if (style.captions.highlight_style === "giant") return "giant";
  if (["rise", "spoken"].includes(style.captions.entrance)) return style.captions.entrance;
  if (wordsMode(style) === "one") return "one";
  if (style.captions.background_mode === "band") return "band";
  if (style.captions.background_mode === "box") return "box";
  return "plain";
}
export function withCaptionLook(style, look) {
  if (!["plain", "box", "band", "one", "rise", "spoken", "camera_3d"].includes(look)) throw new Error("Unknown caption look");
  let result = copy(style);
  result.captions.camera_3d = look === "camera_3d";
  if (result.captions.highlight_style === "giant") result.captions.highlight_style = "none";
  if (look === "camera_3d") return result;
  if (["rise", "spoken"].includes(look)) result.captions.entrance = look;
  else {
    if (["rise", "spoken"].includes(result.captions.entrance)) result.captions.entrance = "slide";
    if (look === "one") result = withWords(result, "one");
    else {
      result.captions.background_mode = look === "plain" ? "none" : look;
      if (look !== "plain" && result.captions.background_opacity === 0) {
        result.captions.background_opacity = backgroundOpacities.medium;
      }
      if (wordsMode(result) === "one") result = withWords(result, "few");
    }
  }
  return result;
}
const brollKinds = ["screen_recording", "clip", "photo", "screenshot"];
const brollLayouts = {
  full: ["full_frame", "full_frame", "full_frame", "full_frame"],
  mixed: ["full_frame", "full_frame", "overlay", "overlay"],
  box: ["overlay", "overlay", "overlay", "overlay"],
};
function placementValues(kind, schema) {
  return schema?.properties?.broll?.properties?.[`${kind}_placement`]?.enum
    ?? (schema ? [] : [...new Set(Object.values(brollLayouts).flat())]);
}
export function brollLayout(style, schema) {
  const known = Object.keys(brollLayouts).find((layout) => brollKinds.every((kind, index) =>
    style.broll?.[`${kind}_placement`] === brollLayouts[layout][index]));
  if (known) return known;
  const value = style.broll?.clip_placement;
  return brollKinds.every((kind) => placementValues(kind, schema).includes(value)
    && style.broll?.[`${kind}_placement`] === value) ? value : "custom";
}
export function withBrollLayout(style, layout, schema) {
  const placements = Object.hasOwn(brollLayouts, layout) ? brollLayouts[layout] : brollKinds.map(() => layout);
  if (!brollKinds.every((kind, index) => placementValues(kind, schema).includes(placements[index]))) {
    throw new Error("Unknown B-roll layout");
  }
  const result = copy(style);
  result.broll ??= {};
  brollKinds.forEach((kind, index) => { result.broll[`${kind}_placement`] = placements[index]; });
  return result;
}
export function withBrollPlacement(style, kind, value, schema) {
  if (!brollKinds.includes(kind) || !placementValues(kind, schema).includes(value)) {
    throw new Error("Unknown B-roll placement");
  }
  const result = copy(style);
  result.broll ??= {};
  result.broll[`${kind}_placement`] = value;
  return result;
}
const brollFields = {
  tall_framing: ["auto", "keep", "shrink"],
  overlay_box: ["headroom", "side_left", "side_right", "tall_behind"], overlay_motion: ["none", "drift", "scroll"],
  still_push: ["in", "out", "none"], stock_clips: ["off", "ask"], color_match: ["off", "on"],
};
export function withBrollField(style, field, value) {
  if (!Object.hasOwn(brollFields, field) || !brollFields[field].includes(value)) {
    throw new Error("Unknown B-roll option");
  }
  const result = copy(style);
  result.broll ??= {};
  result.broll[field] = value;
  return result;
}
export function withEmphasis(style, effect, on) {
  if (effect !== "video_letters" || typeof on !== "boolean") throw new Error("Unknown emphasis switch");
  const result = copy(style);
  const effects = (result.broll?.emphasis_on ?? []).filter((value) => value !== effect);
  result.broll = { ...result.broll, emphasis_on: [...new Set([...effects, ...(on ? [effect] : [])])].sort() };
  return result;
}
export function withMoment(style, id, on) {
  if (!momentsSlots.some((entry) => entry.id === id) || typeof on !== "boolean") throw new Error("Unknown moment switch");
  const result = copy(style);
  const selected = (result.moments?.styles ?? []).filter((value) => value !== id);
  result.moments = { ...result.moments, styles: [...new Set([...selected, ...(on ? [id] : [])])].sort() };
  return result;
}
export function withZoomMode(style, mode) {
  if (!["none", "punches", "punches_and_push"].includes(mode)) throw new Error("Unknown zoom mode");
  const result = copy(style);
  result.moves = { ...result.moves, zoom_mode: mode };
  return result;
}
export function withJumpCuts(style, mode) {
  if (!["hard", "punch", "smooth", "step_in"].includes(mode)) throw new Error("Unknown jump cuts mode");
  const result = copy(style);
  result.moves = { ...result.moves, jump_cuts: mode };
  return result;
}
export function withPunchScale(style, scale) {
  if (![1.5, 2].includes(scale)) throw new Error("Unknown punch scale");
  const result = copy(style);
  result.moves = { ...result.moves, punch_scale: scale };
  return result;
}
export function withOpeningPush(style, on) {
  if (typeof on !== "boolean") throw new Error("Unknown opening push switch");
  const result = copy(style);
  result.moves = { ...result.moves, opening_push: on };
  return result;
}
export function withTransition(style, look, on) {
  if (!transitionLooks.some((entry) => entry.id === look) || typeof on !== "boolean") throw new Error("Unknown transition switch");
  const result = copy(style);
  const selected = new Set(result.moves?.transitions ?? transitionLooks.map((entry) => entry.id));
  if (on) selected.add(look); else selected.delete(look);
  result.moves = { ...result.moves, transitions: transitionLooks.filter((entry) => selected.has(entry.id)).map((entry) => entry.id) };
  return result;
}
export function transitionsSummary(style) {
  const names = transitionLooks.filter((entry) => entry.selected(style)).map((entry) => entry.name);
  return names.length === transitionLooks.length ? "All three" : names.join(" and ") || "Off";
}
export const musicLevels = [
  { id: "none", name: "No music", note: "", value: 0 },
  { id: "quiet", name: "Quiet", note: "About 6%", value: 0.06 },
  { id: "medium", name: "Medium", note: "About 12%", value: 0.12 },
  { id: "loud", name: "Loud", note: "About 20%", value: 0.2 },
];
export function withMusicLevel(style, level) {
  if (!Number.isFinite(level) || level < 0 || level > 0.4) throw new Error("Use volume from 0 to 0.4");
  const result = copy(style);
  result.music = { ...result.music, level };
  return result;
}
export function soundKitChoice(style) {
  const kit = style.music?.sound_effects ?? "soft";
  return kit === "made" || kit === "none" ? kit : "elevenlabs";
}
export function withSoundKit(style, choice, kit = "soft") {
  if (!["made", "elevenlabs", "none"].includes(choice)) throw new Error("Unknown sound kit choice");
  const selected = !kit || ["made", "none"].includes(kit) ? "soft" : kit;
  if (!/^[a-z][a-z0-9-]{0,63}$/.test(selected)) throw new Error("Unknown sound effects kit");
  const result = copy(style);
  result.music = { ...result.music, sound_effects: choice === "elevenlabs" ? selected : choice };
  return result;
}
export function withSoundFallback(style, on) {
  if (typeof on !== "boolean") throw new Error("Expected a sound fallback switch");
  const result = copy(style);
  result.music = { ...result.music, sound_effects_fallback: on ? "made" : "none" };
  return result;
}
export function withBleepSwears(style, on) {
  if (typeof on !== "boolean") throw new Error("Expected a bleep switch");
  const result = copy(style);
  result.music = { ...result.music, bleep_swears: on };
  return result;
}
export function maskPreviewText(text, style, builtin = []) {
  if (style.music?.bleep_swears !== true) return text;
  const words = new Set([...builtin, ...(style.music.bleep_words_add || [])]);
  for (const word of style.music.bleep_words_remove || []) words.delete(word);
  const normal = (word) => word.normalize("NFKC").toLowerCase().replaceAll("ı", "i")
    .replaceAll("ß", "ss").replaceAll("ς", "σ").replace(/[‘’]/g, "'")
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
  const listed = (word) => !word.includes("*") &&
    (words.has(normal(word)) || words.has(normal(word).replaceAll("-", "")));
  const mask = (word) => {
    let first = true;
    return word.replace(/[\p{L}\p{N}]/gu, (letter) => {
      if (!first) return "*";
      first = false; return letter;
    });
  };
  const maskToken = (word) => {
    if (listed(word)) return mask(word);
    const parts = word.split("-"), indices = new Set();
    parts.forEach((part, index) => { if (listed(part)) indices.add(index); });
    for (let i = 0; i + 1 < parts.length; i++) {
      if (normal(parts[i]) && normal(parts[i + 1]) && !parts[i].includes("*") && !parts[i + 1].includes("*") &&
          listed(normal(parts[i]) + normal(parts[i + 1]))) { indices.add(i); indices.add(i + 1); }
    }
    return parts.map((part, index) => indices.has(index) ? mask(part) : part).join("-");
  };
  const tokens = [...text.matchAll(/(?:[\p{L}\p{N}]\*+|[\p{L}\p{N}]+)(?:['‘’\-](?:[\p{L}\p{N}]\*+|[\p{L}\p{N}]+|\*+))*['‘’]?/gu)];
  const masks = new Map();
  for (let i = 0; i < tokens.length; i++) {
    const word = tokens[i][0], next = tokens[i + 1]?.[0];
    if (next && !word.includes("*") && !next.includes("*") && listed(normal(word) + normal(next))) {
      masks.set(i, mask(word)); masks.set(i + 1, mask(next)); i++;
    } else masks.set(i, maskToken(word));
  }
  for (let i = tokens.length - 1; i >= 0; i--) {
    const token = tokens[i];
    text = text.slice(0, token.index) + masks.get(i) + text.slice(token.index + token[0].length);
  }
  return text;
}
export function withSoundExtra(style, name, on) {
  if (!["typing", "message-ping", "camera-shutter"].includes(name)) throw new Error("Unknown extra sound");
  if (typeof on !== "boolean") throw new Error("Expected a more sounds switch");
  const result = copy(style);
  const selected = new Set(style.music?.sound_extras ?? []);
  if (on) selected.add(name); else selected.delete(name);
  result.music = { ...result.music, sound_extras: [...selected] };
  return result;
}
export const soundEffectsOn = (style) => (style.music?.sound_effects ?? "soft") !== "none";
export function withSoundEffects(style, on, kit) {
  if (typeof on !== "boolean") throw new Error("Expected a sound effects switch");
  const selected = kit && kit !== "none" ? kit : soundEffectsOn(style) ? style.music?.sound_effects ?? "soft" : "soft";
  if (!/^[a-z][a-z0-9-]{0,63}$/.test(selected)) throw new Error("Unknown sound effects kit");
  const result = copy(style);
  result.music = { ...result.music, sound_effects: on ? selected : "none" };
  return result;
}
export function withVoicePreset(style, preset) {
  if (!["none", "clean", "broadcast", "warm"].includes(preset)) throw new Error("Unknown voice preset");
  const result = copy(style);
  result.voice = { ...result.voice, preset };
  return result;
}
export function withVoiceRepair(style, field, value) {
  if (field === "repair_proximity" ? !["off", "gentle", "strong"].includes(value)
    : !["repair_rumble", "repair_room", "repair_harshness"].includes(field) || typeof value !== "boolean") {
    throw new Error("Unknown voice repair");
  }
  const result = copy(style);
  result.voice = { ...result.voice, [field]: value };
  return result;
}
export function withBeatSnap(style, on) {
  if (typeof on !== "boolean") throw new Error("Expected a beat switch");
  const result = copy(style);
  result.music = { ...result.music, beat_snap: on };
  return result;
}
export function withMotion(style, motion) {
  const choice = motions.find((entry) => typeof motion === "string" ? entry.name === motion
    : entry.name === motion?.name && entry.rise === motion.rise && entry.ms === motion.ms);
  if (!choice) throw new Error("Unknown caption motion");
  const result = copy(style);
  Object.assign(result.captions, { entry_rise_px: choice.rise, entry_ms: choice.ms });
  if (["rise", "spoken"].includes(result.captions.entrance)) result.captions.entrance = "slide";
  return result;
}
const displayName = (value) => String(value ?? "").replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
// Labels map engine values to copy. card_kinds holds the state response's kind metadata.
export function screenSummary(screen, style, labelMap = {}, schema) {
  const label = (value) => labelMap[value] ?? labels[value] ?? displayName(value);
  switch (screen) {
    case "font": return captionsFont(style);
    case "captions": return captionLook(style) === "camera_3d" ? "3D camera"
      : captionLook(style) === "one" ? "One word"
      : captionLook(style) === "spoken" ? labels.spoken : displayName(captionLook(style));
    case "colors": return palettes.find((palette) => sameColors(colorsOf(style), palette))?.name ?? "Your colors";
    case "grade": {
      const grade = style.grade;
      if (!grade || grade.look === "none") return "None";
      return `${grade.look === "film" ? "Film look" : `Eyedropper ${grade.color}`}, ${(grade.strength ?? 1) <= 0.5 ? "light" : "full"}`;
    }
    case "top-text": return topTextAnswers.find(([id]) => id === topTextChoice(style))?.[1] ?? "Not answered";
    case "opening": return (style.hook.style === "none" ? "No title" : label(style.hook.style))
      + (style.chapters?.style === "title_card" ? " · chapter cards on" : "");
    case "cards": {
      const groups = new Map();
      for (const [kind, info] of Object.entries(labelMap.card_kinds ?? {})) {
        groups.set(info.category, (groups.get(info.category) ?? false) || kindOn(style, kind, info.opt_in));
      }
      return `${label(style.graphics.style)} cards · ${[...groups.values()].filter(Boolean).length} of ${groups.size} groups on`
        + (style.graphics.flow_look === "tags" ? " · flow tags" : "")
        + ({ on_cue: " · rows on cue", on_cue_glow: " · rows on cue, glowing" }[style.graphics.specs_rows] ?? "")
        + (style.graphics.card_motion === "streak" ? " · Motion streak" : "");
    }
    case "broll": return ({ full: "Full screen", mixed: "Mixed", box: "In a box", split: "Split", custom: "Custom" })[brollLayout(style, schema)]
      + (style.broll?.stock_clips === "ask" ? " · stock clips: ask first" : "")
      + (style.broll?.emphasis_on?.includes("video_letters") ? " · video in letters on" : "")
      + (style.broll?.color_match === "off" ? " · color match off" : "");
    case "moves": {
      const zoom = (style.moves?.zoom_mode ?? "none") === "none" ? "No zooms"
        : `${style.moves.zoom_mode === "punches" ? "Punches" : "Punches and a slow push"} at ${style.moves.punch_scale ?? 1.5}x`;
      const transitions = transitionsSummary(style);
      return `${zoom} · ${transitions === "All three" ? "all transitions" : transitions === "Off" ? "no transitions" : `${transitions} transitions`}`
        + (style.moves?.jump_cuts === "smooth" ? " · smooth jump cuts"
          : style.moves?.jump_cuts === "punch" ? " · punch on jump cuts"
            : style.moves?.jump_cuts === "step_in" ? " · step in" : "")
        + (style.moves?.opening_push ? " · opening push" : "");
    }
    case "moments": return momentsSlots.filter((entry) => entry.selected(style)).map((entry) => entry.name).join(", ") || "All off";
    case "sound": {
      const level = Number(((style.music?.level ?? 0.06) * 100).toFixed(2));
      const voice = style.voice?.preset ?? "none";
      const extras = style.music?.sound_extras?.length ?? 0;
      return `${level === 0 ? "Music off" : `Music ${level}%`} · effects ${{ made: "made", elevenlabs: "ElevenLabs", none: "off" }[soundKitChoice(style)]} · voice clean-up ${voice === "none" ? "off" : label(voice).toLowerCase()}`
        + (extras && soundKitChoice(style) === "elevenlabs" ? ` · ${extras} more sound${extras === 1 ? "" : "s"}` : "")
        + (style.music?.bleep_swears ? " · bleep on" : "");
    }
    default: return "";
  }
}
export function showStep(step, tabs, panels) {
  for (const tab of tabs) {
    const active = tab.dataset.step === (screens.some((screen) => screen.id === step && screen.optional) ? "more" : step);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  for (const panel of panels) panel.hidden = panel.id !== `panel-${step}`;
}
export const startList = (state) => [
  ...state.library.map((item) => ({ ...item, fromLibrary: true })),
  ...state.presets.map((item) => ({ ...item, fromLibrary: false })),
];
export const widths = [
  { name: "Normal", value: 1 }, { name: "Condensed", value: 0.85 }, { name: "Narrow", value: 0.7 },
];
export const nearestWidth = (value) => widths.reduce((best, choice) =>
  Math.abs(choice.value - value) < Math.abs(best.value - value) ? choice : best);
export const shadows = {
  soft: { shadow_x_px: 0, shadow_y_px: 3, shadow_blur_px: 8, shadow_opacity: 0.35 },
  strong: { shadow_x_px: 0, shadow_y_px: 6, shadow_blur_px: 3, shadow_opacity: 0.8 },
};
export const shadowMode = (style) => style.captions.shadow_opacity === 0 ? "off"
  : style.captions.shadow_opacity >= 0.6 ? "strong" : "soft";
export function withShadow(style, mode) {
  if (mode !== "off" && !shadows[mode]) throw new Error("Unknown shadow option");
  const result = copy(style);
  Object.assign(result.captions, mode === "off" ? { shadow_opacity: 0 } : shadows[mode]);
  return result;
}
export function withBackground(style, mode) {
  if (!["none", "box", "band"].includes(mode)) throw new Error("Unknown background option");
  const result = copy(style);
  result.captions.background_mode = mode;
  return result;
}
export const backgroundOpacities = { light: 0.5, medium: 0.75, solid: 0.9 };
export const backgroundCorners = { square: 0, rounded: 12, pill: 40 };
export function withBackgroundOpacity(style, choice) {
  if (!Object.hasOwn(backgroundOpacities, choice)) throw new Error("Unknown opacity option");
  const result = copy(style);
  result.captions.background_opacity = backgroundOpacities[choice];
  return result;
}
export function withBackgroundCorners(style, choice) {
  if (!Object.hasOwn(backgroundCorners, choice)) throw new Error("Unknown corner option");
  const result = copy(style);
  result.captions.background_radius_px = backgroundCorners[choice];
  return result;
}
export function withFaceAvoidance(style, enabled) {
  if (typeof enabled !== "boolean") throw new Error("Expected a face avoidance switch");
  const result = copy(style);
  result.captions.placement = enabled ? "avoid_face" : "fixed";
  return result;
}
// Top uses the existing middle-position engine mode with an explicit top offset.
export const placements = { bottom: { position: "bottom", top_px: 1500 },
  middle: { position: "middle", top_px: 1180 }, top: { position: "middle", top_px: 700 } };
export const placementMode = (style) => style.captions.top_px < 940 ? "top" : style.captions.position;
export function withPlacement(style, mode) {
  if (!placements[mode]) throw new Error("Unknown placement option");
  const result = copy(style);
  Object.assign(result.captions, placements[mode]);
  return result;
}
export const sizes = { small: 68, medium: 92, large: 118 };
export const nearestSize = (style) => Object.keys(sizes).reduce((best, name) =>
  Math.abs(sizes[name] - style.captions.size_px) < Math.abs(sizes[best] - style.captions.size_px) ? name : best);
export const motions = [
  { name: "Cut in", rise: 0, ms: 0 }, { name: "Slide up", rise: 12, ms: 100 },
  { name: "Jump up", rise: 40, ms: 160 },
];
// The prototype's ten color palettes, independent of the engine's preset list.
export const palettes = [
  ["Newsprint", "#FFFFFF", "#E9C46A", "#F6F1E7", "#1C1B19"],
  ["Sage", "#FFFFFF", "#365A58", "#F4F3EE", "#222222"],
  ["Rust", "#FFF4DD", "#9B462D", "#E9DDC8", "#222222"],
  ["Cobalt", "#FFFFFF", "#2F3BFF", "#FFFFFF", "#0E1230"],
  ["Terminal", "#F4F1EA", "#7CFFB2", "#111111", "#F4F1EA"],
  ["Lime", "#FFFFFF", "#D8FF2F", "#101522", "#FFFFFF"],
  ["Yellow", "#FFFFFF", "#FFD93D", "#111111", "#FFFFFF"],
  ["Tomato", "#FFFFFF", "#FFE14D", "#FF4A1C", "#FFFFFF"],
  ["Bubblegum", "#FFFFFF", "#FF8AC6", "#FFE3F1", "#2B0F2E"],
  ["Highlighter", "#FFFFFF", "#FFD23F", "#FFF6DC", "#1D1D1D"],
].map(([name, caption, accent, paper, ink]) => ({ name, caption, accent, paper, ink }));
export const roles = [
  ["caption", "Captions"], ["accent", "Highlight"], ["paper", "Cards"], ["ink", "Card text"],
];
export function normalizeHex(raw) {
  const value = String(raw).trim().replace(/^#/, "").toUpperCase();
  if (/^[0-9A-F]{3}$/.test(value)) return `#${[...value].map((c) => c + c).join("")}`;
  return /^[0-9A-F]{6}$/.test(value) ? `#${value}` : null;
}
export function mix(a, b, amount) {
  return "#" + [1, 3, 5].map((at) => {
    const start = parseInt(a.slice(at, at + 2), 16);
    return Math.round(start + (parseInt(b.slice(at, at + 2), 16) - start) * amount)
      .toString(16).padStart(2, "0");
  }).join("").toUpperCase();
}
export function luminance(hex) {
  const channels = [1, 3, 5].map((at) => {
    const value = parseInt(hex.slice(at, at + 2), 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return channels.reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
}
export function contrast(a, b) {
  const [low, high] = [luminance(a), luminance(b)].sort((x, y) => x - y);
  return (high + 0.05) / (low + 0.05);
}
// Match apply_brand's ground contrast without overwriting the saved caption color.
export function captionTextColor(style) {
  const captions = style.captions;
  if (captions.camera_3d || !captions.background_mode || captions.background_mode === "none"
    || captions.background_opacity < 0.5) return captions.text_color;
  const background = captions.background_color || "#000000";
  const color = contrast("#FFFFFF", background) > contrast("#111111", background) ? "#FFFFFF" : "#111111";
  return contrast(color, background) < 4.5 ? "#000000" : color;
}
export function captionContrastNote(style) {
  const captions = style.captions;
  if (captions.camera_3d || !captions.background_mode || captions.background_mode === "none"
    || captions.background_opacity < 0.5) return "";
  const color = captionTextColor(style);
  return `Text switches to ${color === "#FFFFFF" ? "light" : "dark"} on this background so it stays readable. Using ${color}.`;
}
export function posterWords(style, sentence) {
  const captions = style.captions;
  const words = String(sentence).trim().split(/\s+/).filter(Boolean);
  const result = [];
  for (const word of words.slice(0, captions.max_words_per_phrase)) {
    if (result.length && [...result, word].join(" ").length > captions.max_characters_per_phrase) break;
    result.push(word);
  }
  return result;
}
export function colorsOf(style) {
  return {
    caption: style.captions.text_color,
    accent: style.captions.camera_3d ? heroAccent(style) : style.brand?.accent_color || (style.captions.highlight_style === "none"
      ? style.graphics.accent_color : style.captions.highlight_color),
    paper: style.graphics.paper_color, ink: style.graphics.ink_color,
  };
}
export const sameColors = (a, b) => roles.every(([id]) => a[id].toUpperCase() === b[id].toUpperCase());
export const heroAccent = (style) => style.brand?.accent_color || style.hook?.accent_color || style.captions.highlight_color;
export const camera3dHalo = (color) => contrast(color, "#101010") >= contrast(color, "#FFFFFF") ? "#101010" : "#FFFFFF";

// Match grade.apply_rgb for the hero pixels baked into the moving picture.
export function gradeColor(color, grade) {
  if (!grade?.look || grade.look === "none" || grade.strength === 0) return color;
  const clamp = (v) => Math.max(0, Math.min(1, v));
  const smooth = (low, high, value) => { const t = clamp((value - low) / (high - low)); return t * t * (3 - 2 * t); };
  const product = (matrix, vector) => matrix.map((row) => row.reduce((sum, v, i) => sum + v * vector[i], 0));
  const rgbToLms = [[.4122214708, .5363325363, .0514459929], [.2119034982, .6806995451, .1073969566], [.0883024619, .2817188376, .6299787005]];
  const lmsToLab = [[.2104542553, .793617785, -.0040720468], [1.9779984951, -2.428592205, .4505937099], [.0259040371, .7827717662, -.808675766]];
  const labToLms = [[.9999999984505196, .3963377921737678, .21580375806075877], [1.0000000088817607, -.10556134232365633, -.0638541747717059], [1.0000000546724108, -.08948418209496574, -1.2914855378640917]];
  const lmsToRgb = [[4.076741661347994, -3.3077115904081933, .23096992872942793], [-1.2684380040921763, 2.6097574006633715, -.3413193963102196], [-.004196086541837074, -.7034186144594495, 1.7076147009309446]];
  const labOf = (hex) => product(lmsToLab, product(rgbToLms,
    [1, 3, 5].map((at) => (parseInt(hex.slice(at, at + 2), 16) / 255) ** 2.4)).map(Math.cbrt));
  const lab = labOf(color), [light, a, b] = lab, chroma = Math.hypot(a, b);
  const hue = Math.atan2(b, a) * 180 / Math.PI;
  const distance = Math.abs(((hue - 55 + 180) % 360 + 360) % 360 - 180);
  const skin = (1 - smooth(20, 35, distance)) * smooth(.015, .035, chroma)
    * (1 - smooth(.17, .22, chroma)) * smooth(.2, .35, light);
  const graded = [...lab];
  if (grade.look === "eyedropper") {
    if (!grade.color) return color;
    const target = labOf(grade.color), amount = Math.hypot(target[1], target[2]);
    if (!amount) return color;
    const pull = .45 * Math.min(amount, .12) * smooth(.08, .4, light)
      * (1 - .6 * smooth(.85, 1, light)) * (1 - .92 * skin);
    graded[1] += pull * target[1] / amount; graded[2] += pull * target[2] / amount;
  } else {
    const xs = [0, .25, .55, .8, 1], ys = [.04, .215, .55, .84, .97];
    const widths = xs.slice(1).map((x, i) => x - xs[i]);
    const slopes = widths.map((w, i) => (ys[i + 1] - ys[i]) / w);
    const tangents = [slopes[0], ...slopes.slice(1).map((v, i) => (v + slopes[i]) / 2), slopes.at(-1)];
    slopes.forEach((slope, i) => {
      const radius = Math.hypot(tangents[i] / slope, tangents[i + 1] / slope);
      if (radius > 3) { tangents[i] *= 3 / radius; tangents[i + 1] *= 3 / radius; }
    });
    const at = xs.findIndex((x) => x >= light);
    const i = at < 0 ? 3 : Math.max(0, at - 1), t = (light - xs[i]) / widths[i];
    graded[0] = (2*t**3 - 3*t**2 + 1) * ys[i] + (t**3 - 2*t**2 + t) * widths[i] * tangents[i]
      + (-2*t**3 + 3*t**2) * ys[i + 1] + (t**3 - t**2) * widths[i] * tangents[i + 1];
    const shadow = 1 - smooth(.15, .6, light), highlight = smooth(.55, .95, light);
    graded[1] = a * (1 - .15 * (1 - .8 * skin)) + (-.020 * shadow + .010 * highlight) * (1 - .92 * skin);
    graded[2] = b * (1 - .15 * (1 - .8 * skin)) + (-.022 * shadow + .026 * highlight) * (1 - .92 * skin);
  }
  const blended = lab.map((v, i) => v + (grade.strength ?? 1) * (graded[i] - v));
  const linear = product(lmsToRgb, product(labToLms, blended).map((v) => v ** 3));
  return "#" + linear.map((v) => Math.round(clamp(v) ** (1 / 2.4) * 255).toString(16).padStart(2, "0")).join("").toUpperCase();
}
export function withColors(style, colors, editedRoles = roles.map(([id]) => id)) {
  if (!roles.every(([id]) => validHex(colors[id]))) throw new Error("Use a hex code like #2F3BFF.");
  const result = copy(style);
  const edited = new Set(editedRoles);
  const { caption, accent, paper, ink } = colors;
  // Loading only reads graphics colors. An edit shares only the chosen role with titles;
  // a palette or website suggestion deliberately replaces all four roles.
  if (edited.has("caption")) result.captions.text_color = caption;
  if (edited.has("accent")) {
    // Explicit highlight choices replace the brand override so resolution uses these colors.
    if (result.brand) delete result.brand.accent_color;
    result.captions.highlight_color = accent;
    result.hook.accent_color = accent;
    result.graphics.accent_color = accent;
  }
  if (edited.has("accent") || edited.has("ink")) {
    const highlight = result.brand?.accent_color || result.captions.highlight_color;
    const candidates = [ink, "#111111", "#FFFFFF"];
    result.captions.highlight_text_color = candidates.reduce((best, color) =>
      contrast(color, highlight) > contrast(best, highlight) ? color : best);
  }
  if (edited.has("paper")) {
    result.hook.background_color = paper;
    result.graphics.paper_color = paper;
  }
  if (edited.has("ink")) {
    result.hook.text_color = ink;
    result.graphics.ink_color = ink;
  }
  if (edited.has("paper") || edited.has("ink")) {
    result.graphics.titlebar_color = mix(result.graphics.paper_color, result.graphics.ink_color, 0.08);
  }
  return result;
}
const fontSpacing = { "Instrument Serif": -0.8, Gelasio: -0.3, Inter: -0.5, Geist: -0.5, Alata: -0.5,
  "Archivo Black": -2, "Anton SC": 1, "Titan One": 0, "Permanent Marker": 0.5, "Space Mono": -1.5 };
export const captionsFont = (style) => style.brand?.captions_font || style.captions.font;
export const graphicsFonts = (style) => ({
  title: style.brand?.titles_font || style.hook.font,
  cards: style.brand?.cards_font || style.graphics.font,
});
export function fontMode(style) {
  const captions = captionsFont(style), graphics = graphicsFonts(style);
  return captions === graphics.title && captions === graphics.cards ? "same" : "different";
}
export function withCaptionsFont(style, font) {
  const result = copy(style);
  result.captions.font = font;
  if (result.brand?.captions_font) result.brand.captions_font = font;
  result.captions.letter_spacing_px = fontSpacing[font] ?? 0;
  return result;
}
export function withGraphicsFont(style, font) {
  const result = copy(style);
  result.hook.font = font;
  result.graphics.font = font;
  for (const field of ["titles_font", "cards_font"]) {
    if (result.brand?.[field]) result.brand[field] = font;
  }
  return result;
}
export const fontPair = { captions: "Instrument Serif", graphics: "Geist" };
export const withFontPair = (style, pair = fontPair) =>
  withGraphicsFont(withCaptionsFont(style, pair.captions), pair.graphics);
export function withFont(style, font) {
  const result = copy(style);
  for (const [field, section] of [["captions_font", "captions"], ["titles_font", "hook"], ["cards_font", "graphics"]]) {
    result[section].font = font;
    if (result.brand?.[field]) result.brand[field] = font;
  }
  result.captions.letter_spacing_px = fontSpacing[font] ?? 0;
  return result;
}
export function colorWarnings(style) {
  const colors = colorsOf(style), messages = [];
  if (contrast(colors.ink, colors.paper) < 4.5) messages.push("Card text is hard to read on this card color.");
  if ((!style.captions.background_mode || style.captions.background_mode === "none"
    || style.captions.background_opacity === 0) && luminance(colors.caption) < 0.3) {
    messages.push("Dark captions can disappear on darker video.");
  }
  if (["color", "gradient", "editorial", "slam", "clip_wipe"].includes(style.captions.highlight_style)
    && luminance(colors.accent) < 0.12) {
    messages.push("This highlight is dark, so a colored word is hard to see on video. A sliding block shows it better.");
  }
  if (style.captions.highlight_style === "pill") {
    const text = style.brand?.accent_color
      ? ["#111111", "#FFFFFF"].reduce((best, color) => contrast(color, colors.accent) > contrast(best, colors.accent) ? color : best)
      : style.captions.highlight_text_color;
    if (contrast(text, colors.accent) < 4.5
      || (style.captions.max_words_per_phrase > 1 && contrast(mix(colors.accent, text, 0.45), colors.accent) < 3)) {
      messages.push("Spoken or upcoming pill words are hard to read on this highlight color, so choose colors with more contrast.");
    }
  }
  return messages;
}
export function withCardStyle(style, look) {
  if (!["paper", "minimal", "dark", "bright"].includes(look)) throw new Error("Unknown card style");
  const colors = { paper: style.graphics.paper_color, ink: style.graphics.ink_color };
  if (look === "dark" && luminance(colors.paper) >= 0.3) {
    [colors.paper, colors.ink] = [luminance(colors.ink) < 0.3 ? colors.ink : "#111111", colors.paper];
  } else if (["paper", "minimal"].includes(look) && luminance(colors.paper) < 0.3) {
    [colors.paper, colors.ink] = [luminance(colors.ink) >= 0.3 ? colors.ink : "#FFFFFF", colors.paper];
  }
  if (contrast(colors.paper, colors.ink) < 4.5) {
    colors.ink = contrast(colors.paper, "#111111") > contrast(colors.paper, "#FFFFFF") ? "#111111" : "#FFFFFF";
  }
  const result = copy(style);
  Object.assign(result.graphics, { style: look, paper_color: colors.paper, ink_color: colors.ink });
  if (colors.paper !== style.graphics.paper_color || colors.ink !== style.graphics.ink_color) {
    result.graphics.titlebar_color = mix(colors.paper, colors.ink, 0.08);
  }
  return result;
}
export function reserveSavebar(bar, root) {
  const height = Math.ceil(bar.getBoundingClientRect().height);
  root.style.setProperty("--savebar-height", `${height}px`);
  return height;
}
const groupFields = {
  font: ["captions.font", "hook.font", "graphics.font", "brand.captions_font", "brand.titles_font", "brand.cards_font", "captions.letter_spacing_px", "captions.width_scale"],
  captions: ["captions.max_words_per_phrase", "captions.max_characters_per_phrase", "captions.size_px", "captions.position", "captions.top_px", "captions.highlight_style", "captions.italic_stress", "captions.entry_rise_px", "captions.entry_ms", "captions.shadow_opacity", "captions.shadow_blur_px", "captions.shadow_x_px", "captions.shadow_y_px", "captions.text_color",
    "captions.background_mode", "captions.background_color", "captions.background_opacity", "captions.background_padding_px", "captions.background_radius_px", "captions.placement", "captions.entrance", "captions.camera_3d"],
  colors: ["captions.text_color", "captions.highlight_color", "captions.highlight_text_color", "hook.text_color", "hook.background_color", "hook.accent_color", "graphics.paper_color", "graphics.titlebar_color", "graphics.ink_color", "graphics.accent_color", "brand.accent_color"],
  grade: ["grade.look", "grade.strength", "grade.color", "grade.slider"],
  "top-text": ["hook.answered"],
  opening: ["hook.style", "hook.text_hook_color", "chapters.style"],
  cards: ["graphics.style", "graphics.card_kinds_off", "graphics.card_kinds_on", "graphics.flow_look", "graphics.specs_rows", "graphics.card_motion"],
  broll: brollKinds.map((kind) => `broll.${kind}_placement`).concat(Object.keys(brollFields).map((field) => `broll.${field}`), "broll.emphasis_on"),
  moves: ["moves.zoom_mode", "moves.punch_scale", "moves.transitions", "moves.jump_cuts", "moves.opening_push"],
  moments: ["moments.styles"],
  sound: ["music.level", "music.sound_effects", "music.sound_effects_fallback", "music.sound_extras", "music.beat_snap",
    "music.bleep_swears", "music.bleep_words_add", "music.bleep_words_remove", "voice.preset",
    "voice.repair_rumble", "voice.repair_room", "voice.repair_harshness", "voice.repair_proximity"],
};
export function changeSummary(style, base) {
  return summarizeChanges(style, base, groupFields);
}
function summarizeChanges(style, base, groups) {
  const value = (object, path) => path.split(".").reduce((node, key) => node?.[key], object);
  const changed = Object.entries(groups).filter(([, fields]) =>
    fields.some((field) => JSON.stringify(value(style, field)) !== JSON.stringify(value(base, field))))
    .map(([name]) => screens.find((screen) => screen.id === name)?.name ?? displayName(name));
  return changed.length ? `Changed ${changed.join(", ")} from ${base.name}` : `Starting from ${base.name}`;
}
