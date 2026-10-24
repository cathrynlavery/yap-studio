import {
  check, copy, create404Retry, fontFamily, frameWindow, mainPosterTime, placementNote, previewNote, startingName, tileWindow,
  kindLabel, kindOn, withKind, withWords, wordsMode, stepFromHash, adjacentStep, showStep, startList,
  widths, nearestWidth, motions, palettes, roles, normalizeHex, colorsOf, sameColors,
  withColors, withFont, withCardStyle, reserveSavebar, colorWarnings, changeSummary,
  withShadow, shadowMode, withPlacement, placementMode, sizes, nearestSize,
  withBackgroundOpacity, withBackgroundCorners, withFaceAvoidance,
  backgroundOpacities, backgroundCorners, captionTextColor, captionContrastNote, posterWords,
  captionsFont, graphicsFonts, fontMode as styleFontMode, withCaptionsFont, withGraphicsFont, fontPair, withFontPair,
  gradeLooks, withGrade, stillUrl, cutoutUrl, coverPoint, samplePatch, patchColor,
  screens, screenAvailable, screenSummary, optionAvailable, momentBlock, captionLookSlots, openingSlots,
  openingMoreSlots, cardGroupSlots, brollLayoutSlots, brollMoreSlots, movesSlots, movesMoreSlots, zoomStrengths, momentsSlots,
  captionLook, withCaptionLook, withMotion, brollLayout, withBrollLayout, withBrollPlacement,
  withBrollField, musicLevels, withMusicLevel, soundKitChoice, withSoundKit, withSoundFallback, withSoundExtra,
  withVoicePreset, withVoiceRepair, withBeatSnap, withBleepSwears, maskPreviewText,
  labels, withPreset, heroAccent, camera3dHalo, gradeColor, topTextAnswers, topTextChoice, withTopText,
} from "./choices.mjs";

const $ = (id) => document.getElementById(id);
const token = new URLSearchParams(location.search).get("t") ?? "";
const getUrl = (path) => `${path}${path.includes("?") ? "&" : "?"}t=${encodeURIComponent(token)}`;
const kindCount = (count) => `${count} ${count === 1 ? "kind" : "kinds"}`;

let musicChoice;
let openingLook = "quiet_serif";
let schema, style, preview, state, baseStyle, baseKey, customColors, generation = -1;
let choosingGradeColor = false, gradeColorRequest = 0, gradeColorPending = false;
let activeStep = stepFromHash(location.hash), activeRole = "accent", mainPaused = false;
let mainStage = null;
let lastShadow = null;
let fontRole = "captions", soundKit = "soft";
let pulledSite = "";
let focusedMoment = null;
let previewPollPending = false;
const drawings = new Set();
const foldState = new Map();
const coreTabs = ["font", "captions", "colors", "top-text", "more"];
const optionalScreens = () => screens.filter(({ id, optional }) => optional && screenAvailable(id, schema));
const drawnScreen = (id) => ["broll", "sound", "moves", "moments"].includes(id);
const mainDrawing = (id, candidate = style) => drawnScreen(id) ? id
  : candidate.captions.camera_3d && ["looks", "font", "captions", "colors"].includes(id) ? "camera_3d" : null;
function previewControls(drawing) {
  $("track-row").hidden = Boolean(drawing && drawing !== "camera_3d");
  $("track-row").classList.toggle("is-drawn", drawing === "camera_3d");
}
const availableMoments = () => momentsSlots.filter((entry) => optionAvailable(entry, schema));
const currentMoment = (candidate = style, focus = true) => {
  const entries = availableMoments();
  return (focus && entries.find((entry) => entry.id === focusedMoment))
    || entries.find((entry) => entry.selected(candidate)) || entries[0];
};
const sectionValues = (candidate, section) => ({
  ...Object.fromEntries(Object.entries(schema.properties[section].properties).map(([name, rule]) => [name, rule.default])),
  ...candidate[section],
});
const displayLayout = (candidate) => brollLayout({ broll: sectionValues(candidate, "broll") }, schema);
const summaryFor = (id) => screenSummary(id, id === "broll" ? { ...style, broll: sectionValues(style, "broll") } : style,
  { ...labels, card_kinds: state.card_kinds }, schema);
const stages = new Set();
let tileController = new AbortController();
let mainController = null;
let siteController = null;
const startKey = (item) => `${item.fromLibrary ? "library" : "preset"}:${item.id}`;
function runPlayer(frame, detail, method, ...args) {
  if (method === "play" && detail?.candidate?.graphics?.card_motion === "streak"
    && (mainPaused || matchMedia("(prefers-reduced-motion: reduce)").matches)) return;
  const window = frameWindow(frame, detail?.active);
  const player = window?.__player;
  if (!player) return;
  try {
    Promise.resolve(player[method](...args)).catch((error) => {
      if (frameWindow(frame, detail.active) === window) console.error(error);
    });
  } catch (error) {
    if (frameWindow(frame, detail.active) === window) console.error(error);
  }
}
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const stage = entry.target._galleryStage;
    if (!stage) continue;
    stage.visible = entry.isIntersecting;
    for (const frame of stage.frames) {
      if (!frame._galleryReady) continue;
      if (!entry.isIntersecting) {
        runPlayer(frame, stage, "pause");
        if (!stage.isMain) runPlayer(frame, stage, "seek", stage.posterTimes.get(frame));
      }
      else if (stage.autoplay || (!stage.isMain || !mainPaused) && matchMedia("(hover: none)").matches) {
        runPlayer(frame, stage, "play");
      }
    }
  }
}, { rootMargin: "0px" });

async function api(path, payload, signal) {
  const response = await fetch(
    payload ? path : getUrl(path),
    payload
      ? {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Gallery-Token": token },
        body: JSON.stringify(payload),
        signal,
      }
      : { signal },
  );
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(typeof body.error === "string" ? body.error : "Could not load preview");
    error.code = body.code;
    error.id = body.id;
    throw error;
  }
  return body;
}
function mount(host, spec, isMain = false, refreshOn404 = null, candidate = style) {
  disposeDrawing(host);
  const poster = host.querySelectorAll(".tile-poster")[0];
  const stage = document.createElement("div");
  stage.className = "stage";
  stage.style.visibility = poster ? "hidden" : "visible";
  const image = document.createElement("img");
  image.alt = "";
  image.src = getUrl(stillUrl(preview, candidate,
    host.dataset.gradePreview === "true" || (isMain && choosingGradeColor)));
  stage.append(image);
  if (isMain) {
    const safe = document.createElement("div");
    safe.className = "guide";
    stage.append(safe);
    const caption = document.createElement("div");
    caption.className = "guide caption";
    stage.append(caption);
  }
  const frames = [];
  let detail;
  for (const layer of spec.layers) {
    const frame = document.createElement("iframe");
    frame.title = `${layer.name} live preview`;
    frame.src = getUrl(layer.src);
    frame._galleryReady = false;
    frame.addEventListener("load", () => {
      const whenReady = async () => {
        const window = frameWindow(frame, detail?.active);
        if (!window) return;
        const player = window.__player;
        if (!window.__playerReady || !player) {
          requestAnimationFrame(whenReady);
          return;
        }
        await window.document.fonts.ready;
        if (frameWindow(frame, detail.active) !== window) return;
        await player.pause();
        await new Promise(requestAnimationFrame);
        if (frameWindow(frame, detail.active) !== window) return;
        const motion = layer.name === "card" && candidate?.graphics?.card_motion === "streak";
        const playing = (!motion || (!mainPaused && !matchMedia("(prefers-reduced-motion: reduce)").matches))
          && ((isMain && !mainPaused) || detail.hovered
            || (!isMain && detail.visible && (detail.autoplay || matchMedia("(hover: none)").matches)));
        const posterTime = isMain && !motion ? mainPosterTime(spec.layers) : layer.poster_s ?? layer.start_s ?? 0;
        const at = motion && playing ? 0 : posterTime;
        await player.seek(at);
        if (frameWindow(frame, detail.active) !== window) return;
        if (layer.name === "card") {
          // The first seek can bind a newly registered GSAP timeline without drawing it.
          await new Promise(requestAnimationFrame);
          if (frameWindow(frame, detail.active) !== window) return;
          await player.seek(at);
        }
        if (frameWindow(frame, detail.active) !== window) return;
        window.__timelines?.[layer.template]?.seek(player.getTime());
        if (frameWindow(frame, detail.active) !== window) return;
        frame._galleryReady = true;
        if (poster && frames.every((item) => item._galleryReady)) {
          poster.hidden = true;
          stage.style.visibility = "visible";
        }
        if (playing) runPlayer(frame, detail, "play");
      };
      const check = detail?.retry404 ? detail.retry404(frame.src) : Promise.resolve(false);
      check.then((retried) => {
        if (!retried) return whenReady();
      }).catch((error) => {
        if (frameWindow(frame, detail?.active)) console.error(error);
      });
    });
    Object.assign(frame.style, {
      left: `${layer.x}px`,
      top: `${layer.y}px`,
      width: `${layer.width}px`,
      height: `${layer.height}px`,
    });
    stage.append(frame);
    frames.push(frame);
  }
  host.replaceChildren(stage, ...(poster ? [poster] : []));
  detail = {
    host,
    stage,
    frames,
    startTimes: new Map(spec.layers.map((layer, index) => [frames[index],
      layer.name === "card" && candidate.graphics?.card_motion === "streak" ? 0 : layer.start_s ?? 0])),
    posterTimes: new Map(spec.layers.map((layer, index) => [frames[index], layer.poster_s ?? layer.start_s ?? 0])),
    visible: isMain,
    hovered: !isMain && host.matches(":hover") && matchMedia("(hover: hover)").matches,
    active: true,
    isMain,
    candidate,
    autoplay: !isMain && candidate.graphics?.card_motion === "streak"
      && !matchMedia("(prefers-reduced-motion: reduce)").matches,
    usesCutout: spec.layers.some((layer) => layer.name === "card" && layer.template === "big_number"),
    retry404: refreshOn404 ? create404Retry(fetch, refreshOn404) : null,
    duration: Math.max(1, ...spec.layers.map((l) => l.duration_s)),
  };
  host._galleryStage = detail;
  syncStageCutout(detail);
  stages.add(detail);
  observer.observe(host);
  const fit = () => {
    const phone = matchMedia("(max-width: 899px)").matches;
    const compact = host.classList.contains("compact");
    const captionTile = !compact && host.parentElement?.parentElement?.id === "caption-look-tiles";
    const layerName = activeStep === "opening" ? "title" : activeStep === "cards" ? "card" : "captions";
    const focus = activeStep === "grade" ? { x: 540, y: 960, width: 0, height: 0 }
      : spec.layers.find((layer) => layer.name === layerName) || spec.layers[0];
    const crop = !isMain && activeStep !== "top-text" && focus && (!phone || captionTile || compact);
    const windowBox = crop
      ? tileWindow(focus, spec.frame, hostSize(host), phone && captionTile ? .72 : .5)
      : { x: 0, y: 0, width: 1080, height: 1920 };
    const scale = host.clientWidth / windowBox.width;
    stage.style.transform = `scale(${scale})`;
    stage.style.left = `${-windowBox.x * scale}px`;
    stage.style.top = `${-windowBox.y * scale}px`;
  };
  fit();
  detail.resize = new ResizeObserver(fit);
  detail.resize.observe(host);
  if (!isMain) wireGalleryHover(host);
  return detail;
}
function wireGalleryHover(host) {
  if (host._galleryHoverWired || !matchMedia("(hover: hover)").matches) return;
  host._galleryHoverWired = true;
  host.addEventListener("mouseenter", () => {
    const current = host._galleryStage;
    if (!current) return;
    current.hovered = true;
    for (const frame of current.frames) runPlayer(frame, current, "play");
  });
  host.addEventListener("mouseleave", () => {
    const current = host._galleryStage;
    if (!current) return;
    current.hovered = false;
    if (current.autoplay) return;
    for (const frame of current.frames) {
      runPlayer(frame, current, "pause");
      runPlayer(frame, current, "seek", current.posterTimes.get(frame));
    }
  });
}
function syncStageCutout(detail) {
  if (!detail.active || !detail.usesCutout) return;
  if (!preview.cutout) {
    detail.cutout?.remove();
    detail.cutout = null;
    return;
  }
  if (!detail.cutout) {
    detail.cutout = imageNode(getUrl(cutoutUrl(preview, detail.candidate)), "card-cutout");
    const image = detail.cutout;
    image.addEventListener("error", () => { image.hidden = true; });
    image.addEventListener("load", () => { image.hidden = false; });
    detail.stage.append(image);
  }
  detail.cutout.src = getUrl(cutoutUrl(preview, detail.candidate));
}
function disposeStage(detail) {
  detail.active = false;
  for (const frame of detail.frames) frame._galleryReady = false;
  observer.unobserve(detail.host);
  detail.resize.disconnect();
  stages.delete(detail);
  if (detail.host._galleryStage === detail) detail.host._galleryStage = null;
}
function clearStages(includeMain = false) {
  lazyObserver.disconnect();
  drawnObserver.disconnect();
  for (const host of [...drawings]) {
    if (host._drawn.main && !includeMain) continue;
    disposeDrawing(host);
  }
  tileQueue = [];
  tileController.abort();
  tileController = new AbortController();
  if (includeMain) {
    mainController?.abort();
    mainController = null;
  }
  for (const detail of [...stages]) {
    if (detail.isMain && !includeMain) continue;
    disposeStage(detail);
  }
  if (includeMain) mainStage = null;
}
function animate() {
  for (const detail of stages) {
    if (!detail.visible || (detail.isMain ? mainPaused
      : !detail.autoplay && !detail.hovered && !matchMedia("(hover: none)").matches)) continue;
    for (const frame of detail.frames) {
      const player = frameWindow(frame, detail.active)?.__player;
      if (!frame._galleryReady || !player) continue;
      if (player.getTime() >= player.getDuration() - 0.05) {
        runPlayer(frame, detail, "seek", detail.startTimes.get(frame));
        runPlayer(frame, detail, "play");
      }
      if ((detail.autoplay || detail.isMain || matchMedia("(hover: none)").matches) && !player.isPlaying()) {
        runPlayer(frame, detail, "play");
      }
    }
  }
  const player = frameWindow(mainStage?.frames[0], mainStage?.active)?.__player;
  if (player) {
    const time = player.getTime(), duration = player.getDuration() || 1;
    $("time").textContent = `0:${time.toFixed(2).padStart(5, "0")}`;
    $("playhead").style.left = `${Math.min(100, time / duration * 100)}%`;
  }
  requestAnimationFrame(animate);
}
async function renderMain(retried = false) {
  mainController?.abort();
  disposeDrawing($("main-frame"));
  const controller = new AbortController();
  mainController = controller;
  if (mainStage) {
    disposeStage(mainStage);
    mainStage = null;
    $("main-frame").replaceChildren();
  }
  const candidate = copy(style);
  if (!candidate.hook.answered) candidate.hook.style = "none";
  const drawing = mainDrawing(activeStep, candidate);
  previewControls(drawing);
  if (drawing) {
    drawPreview($("main-frame"), candidate, drawing, true);
    return;
  }
  const spec = await api("/api/layers", { style: candidate, tile: {
    tab: "preview", step: activeStep,
    ...(activeStep === "font" && fontRole === "graphics" ? { section: "cards" } : {}),
  } }, controller.signal);
  if (controller.signal.aborted || $("picker").hidden) return;
  if (candidate.hook.style === "video_letters") {
    const title = spec.layers.find((layer) => layer.name === "title");
    if (title) $("title-note").textContent = `Shows for the first ${Number(title.duration_s.toFixed(1))} seconds.`;
  }
  mainStage = mount($("main-frame"), spec, true,
    retried ? null : () => renderMain(true), candidate);
}
async function renderTile(host, tile, candidate, signal, retried = false) {
  const poster = host.querySelectorAll(".tile-poster")[0];
  if (poster) poster.hidden = false;
  try {
    const spec = await api("/api/layers", { style: candidate, tile }, signal);
    if (signal.aborted || !host.isConnected || $("picker").hidden || !foldsVisible(host)) return;
    if (host._galleryStage) disposeStage(host._galleryStage);
    mount(host, spec, false, retried ? null : () => renderTile(host, tile, candidate, signal, true), candidate);
  } catch (error) {
    if (!signal.aborted && error.name !== "AbortError" && host.isConnected && !$("picker").hidden) {
      host.setAttribute("title", error.message);
    }
  }
}
let tileQueue = [], runningTiles = 0;
function pumpTiles() {
  while (runningTiles < 4 && tileQueue.length) {
    const task = tileQueue.shift();
    if (task.signal.aborted || !task.host.isConnected) continue;
    runningTiles += 1;
    renderTile(task.host, task.tile, task.candidate, task.signal).finally(() => {
      runningTiles -= 1;
      pumpTiles();
    });
  }
}
const lazyObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    lazyObserver.unobserve(entry.target);
    tileQueue.push(entry.target._tileTask);
  }
  pumpTiles();
}, { rootMargin: "120px" });
function queueTile(host, tile, candidate = style) {
  if (candidate.captions.camera_3d && ["captions", "preview"].includes(tile.tab)) {
    drawPreview(host, candidate, "camera_3d");
    return;
  }
  staticPoster(host, candidate);
  host._tileTask = { host, tile: { ...tile, step: activeStep }, candidate: copy(candidate), signal: tileController.signal };
  if (foldsVisible(host)) lazyObserver.observe(host);
}
function captionPoster(host, candidate, fullFrame = false) {
  const caption = document.createElement("div"), line = document.createElement("span");
  const c = candidate.captions, accent = candidate.brand?.accent_color || c.highlight_color;
  caption.className = "poster-caption";
  line.className = "poster-line";
  Object.assign(caption.style, {
    color: captionTextColor(candidate), fontFamily: fontFamily(candidate.brand?.captions_font || c.font, state.fonts),
    fontSize: `${fullFrame ? c.size_px * (host.clientWidth || 200) / 1080 : Math.max(12, c.size_px * (host.clientWidth || 200) / 900)}px`,
    fontWeight: ["slam", "clip_wipe", "camera_follow", "glitch_rgb", "highlight", "neon_glow", "particle_burst"].includes(c.highlight_style) ? "800" : "400",
    letterSpacing: `${c.letter_spacing_px * (fullFrame ? (host.clientWidth || 200) / 1080 : 1 / 5)}px`,
    transform: `scaleX(${c.width_scale})`,
    textShadow: `${c.shadow_x_px / 5}px ${c.shadow_y_px / 5}px ${c.shadow_blur_px / 5}px rgba(0,0,0,${c.shadow_opacity})`,
  });
  if (fullFrame) fitCaption(caption, host, candidate);
  if (c.background_mode !== "none" && c.background_opacity > 0) {
    const ground = c.background_mode === "band" ? caption : line;
    ground.style.backgroundColor = `${c.background_color}${Math.round(c.background_opacity * 255).toString(16).padStart(2, "0")}`;
    ground.style.borderRadius = `${c.background_radius_px / 5}px`;
    if (c.background_mode === "band") caption.className += " poster-band";
  }
  const words = posterWords(candidate, maskPreviewText(preview.sentence, candidate, state.bleep_builtin));
  words.forEach((text, index) => {
    const word = document.createElement("span");
    word.className = "poster-word";
    word.textContent = text;
    if (index === 0 && c.highlight_style !== "none") {
      word.style.color = accent;
      if (["moving_block", "pill", "highlight", "pill_karaoke", "blend_difference"].includes(c.highlight_style)) {
        Object.assign(word.style, { backgroundColor: accent, color: c.highlight_text_color,
          borderRadius: c.highlight_style === "moving_block" ? "0" : ".2em", padding: "0 .12em" });
      }
      if (c.highlight_style === "underline") word.style.borderBottom = `2px solid ${accent}`;
      if (["slam", "editorial_emphasis", "neon_accent"].includes(c.highlight_style)) word.style.transform = "scale(1.12)";
      if (["neon_accent", "neon_glow"].includes(c.highlight_style)) word.style.textShadow = `0 0 6px ${accent}`;
      if (["glitch_rgb", "parallax_layers"].includes(c.highlight_style)) word.style.textShadow = `2px 0 ${accent}, -2px 0 #FFFFFF`;
      if (c.highlight_style === "emoji_pop") word.textContent += " ☺";
      if (c.italic_stress) word.style.fontStyle = "italic";
      if (c.highlight_style === "editorial") word.style.fontWeight = "700";
    }
    line.append(word, index < words.length - 1 ? " " : "");
  });
  caption.append(line);
  return caption;
}
function fitCaption(caption, host, candidate) {
  const c = candidate.captions, scale = (host.clientWidth || 200) / 1080;
  Object.assign(caption.style, { fontSize: `${c.size_px * scale}px`, letterSpacing: `${c.letter_spacing_px * scale}px`,
    textShadow: `${c.shadow_x_px * scale}px ${c.shadow_y_px * scale}px ${c.shadow_blur_px * scale}px rgba(0,0,0,${c.shadow_opacity})` });
  if (c.position === "middle") {
    caption.style.top = `${c.top_px / 1920 * 100}%`;
    caption.style.transform = `translateY(-50%) scaleX(${c.width_scale})`;
  } else {
    let padding = Math.ceil(c.size_px * .08 + c.shadow_blur_px * 3 + Math.max(Math.abs(c.shadow_x_px), Math.abs(c.shadow_y_px))) + 8;
    if (["blend_difference", "camera_follow", "editorial_emphasis", "emoji_pop", "glitch_rgb", "gradient_fill", "highlight", "matrix_decode", "neon_accent", "neon_glow", "parallax_layers", "particle_burst", "pill_karaoke", "texture"].includes(c.highlight_style)) padding += Math.ceil(c.size_px * .25);
    if (c.background_mode !== "none" && c.background_opacity > 0) padding += c.background_padding_px;
    const lineHeight = c.size_px * 1.12;
    const top = Math.min(c.top_px, (candidate.internal?.caption_safe_bottom_px ?? 1740) - lineHeight - c.entry_rise_px - padding);
    caption.style.top = "auto";
    caption.style.bottom = `${(1920 - top - lineHeight) / 1920 * 100}%`;
  }
}
function staticPoster(host, candidate) {
  disposeDrawing(host);
  const poster = document.createElement("div");
  poster.className = "tile-poster";
  poster.setAttribute("aria-hidden", "true");
  const image = document.createElement("img");
  image.src = getUrl(stillUrl(preview, candidate, host.dataset.gradePreview === "true"));
  image.alt = "";
  const caption = captionPoster(host, candidate);
  if (matchMedia("(max-width: 899px)").matches && !host.classList.contains("compact") && host.parentElement?.parentElement?.id === "caption-look-tiles") {
    caption.style.top = "72%";
    caption.style.transform = `translateY(-50%) scaleX(${candidate.captions.width_scale})`;
  }
  poster.append(image, caption);
  host.replaceChildren(poster);
}
function tile(parent, name, selected, request, candidate, onClick, compact = false, note = "", momentId = null) {
  if (request?.tab === "preview" && !candidate.hook.answered) {
    candidate = copy(candidate);
    candidate.hook.style = "none";
  }
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tile";
  button.setAttribute("aria-pressed", String(selected));
  const frame = document.createElement("div");
  frame.className = `tile-frame stage-host${compact ? " compact" : ""}`;
  frame.dataset.gradePreview = String(parent.id === "grade-tiles");
  const label = document.createElement("span");
  label.className = "tile-label";
  label.textContent = name;
  if (note) {
    const detail = document.createElement("small");
    detail.textContent = note;
    label.append(detail);
  }
  button.append(frame, label);
  button.addEventListener("click", onClick);
  parent.append(button);
  if (typeof request === "string") drawPreview(frame, candidate, request, false, false, momentId);
  else queueTile(frame, request, candidate);
  return button;
}
function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  element.className = className;
  if (text) element.textContent = text;
  return element;
}
function imageNode(src, className = "") {
  const image = node("img", className);
  image.alt = "";
  image.src = src;
  return image;
}
function hostSize(host) {
  return { width: host.clientWidth || 200,
    height: host.clientHeight || (host.clientWidth || 200) * (host.classList.contains("compact") ? 7 / 16 : 16 / 9) };
}
function disposeDrawing(host) {
  drawnObserver.unobserve(host);
  host._drawn?.element.classList.toggle("is-playing", false);
  host._drawn?.resize.disconnect();
  host._drawn = null;
  drawings.delete(host);
}
const drawnObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.target._drawn) continue;
    entry.target._drawn.visible = entry.isIntersecting && entry.intersectionRatio >= 0.5;
    playDrawing(entry.target);
  }
}, { threshold: 0.5 });
function playDrawing(host) {
  const drawing = host._drawn;
  if (!drawing) return;
  const playing = !matchMedia("(prefers-reduced-motion: reduce)").matches && foldsVisible(host)
    && (drawing.main ? drawing.screen !== "camera_3d" || !mainPaused
      : matchMedia("(hover: none)").matches ? drawing.visible : drawing.hovered);
  drawing.element.classList.toggle("is-playing", playing);
}
function drawPreview(host, candidate, screen, main = false, still = false, momentId = null) {
  const sentence = maskPreviewText(preview.sentence, candidate, state.bleep_builtin);
  const previous = host._drawn;
  disposeDrawing(host);
  const drawing = node("div", "drawn-preview");
  drawing.setAttribute("aria-hidden", "true");
  const picture = node("div", "drawn-picture");
  const photo = imageNode(getUrl(stillUrl(preview, candidate, host.dataset.gradePreview === "true")), "drawn-still");
  picture.append(photo);
  drawing.append(picture);
  const camera3d = screen === "camera_3d";
  const cameraBounds = camera3d ? drawCamera3d(picture, candidate) : null;
  if (screen.startsWith("transition_")) {
    const treatment = node("div", `transition-treatment ${screen}`);
    for (const [side, src] of [["from", photo.src], ["to", "/static/shot.jpg"]]) {
      const frame = node("div", `transition-${side}`), image = imageNode(src, "transition-image");
      if (screen === "transition_whip_pan") {
        const stretch = node("div", "transition-stretch"), blur = node("div", "transition-blur");
        blur.append(image); stretch.append(blur); frame.append(stretch);
      } else frame.append(image);
      treatment.append(frame);
    }
    if (screen === "transition_light_leak") treatment.append(node("div", "transition-wash"));
    picture.replaceChildren(treatment);
  }
  if (screen === "moves" || screen === "opening_push") {
    const moves = sectionValues(candidate, "moves");
    const opening = screen === "opening_push";
    if (opening || moves.zoom_mode !== "none") {
      const zoom = node("div", `zoom-picture zoom-${opening ? "opening" : moves.zoom_mode}`);
      const face = preview.face_boxes?.[0];
      const x = face ? face.x + face.w / 2 : 540, y = face ? face.y + face.h / 2 : 1920 * .42;
      zoom.style.transformOrigin = `${Math.max(0, Math.min(100, x / 1080 * 100))}% ${Math.max(0, Math.min(100, y / 1920 * 100))}%`;
      zoom.style.setProperty("--punch-scale", String(opening ? 1.3 : moves.punch_scale));
      picture.replaceChildren(zoom); zoom.append(photo);
      if (!opening && !main && !still) drawing.append(node("span", "zoom-tag", `${moves.punch_scale}x`));
    }
  }
  if (["jump_punch", "jump_step_in"].includes(screen)) {
    const zoom = node("div", `zoom-picture ${screen}`);
    const face = preview.face_boxes?.[0];
    const x = face ? face.x + face.w / 2 : 540, y = face ? face.y + face.h / 2 : 1920 * .42;
    zoom.style.transformOrigin = `${Math.max(0, Math.min(100, x / 1080 * 100))}% ${Math.max(0, Math.min(100, y / 1920 * 100))}%`;
    zoom.style.transform = "scale(1)";
    picture.replaceChildren(zoom); zoom.append(photo);
  }
  const moment = screen === "moments" ? availableMoments().find((entry) => entry.id === momentId)
    || currentMoment(candidate, !still) : null;
  if (moment) drawMoment(picture, candidate, moment.id);
  let lettersWord;
  let splitLayout = false;
  let tallLayout = false;
  let pageLayout = false;
  const boxPictures = [];
  if (screen === "broll") {
    const broll = sectionValues(candidate, "broll"), layout = displayLayout(candidate);
    drawing.style.setProperty("--push-scale", String(broll.still_push_scale ?? 1.08));
    if (broll.overlay_box !== "tall_behind" && broll.overlay_motion !== "scroll") {
      if (layout === "split" || (layout === "custom" && broll.clip_placement === "split")) {
        splitLayout = true;
        drawSplit(picture, photo, broll);
      } else {
        const full = layout !== "box" && (layout !== "custom" || broll.clip_placement === "full_frame");
        const shot = node("div", full ? "shot-full" : `shot-box box-${broll.overlay_box}`);
        shot.append(imageNode("/static/shot.jpg", `shot-image push-${broll.still_push}`));
        if (!full && broll.overlay_motion === "drift") shot.className += " drift";
        drawing.append(shot);
        if (layout === "mixed" || (layout === "custom" && broll.photo_placement === "overlay" && full)) {
          const inset = node("div", `shot-box photo-inset box-${broll.overlay_box}${broll.overlay_motion === "drift" ? " drift" : ""}`);
          inset.append(imageNode("/static/shot.jpg", `shot-image push-${broll.still_push}`));
          drawing.append(inset);
        }
      }
    } else {
      const hasBox = layout === "box" || layout === "mixed" || (layout === "custom" &&
        ["screen_recording", "clip", "photo", "screenshot"].some((kind) => broll[`${kind}_placement`] === "overlay"));
      const split = layout === "split" || (layout === "custom" && broll.clip_placement === "split");
      const full = layout === "full" || layout === "mixed" || (layout === "custom" && broll.clip_placement === "full_frame");
      if (split) {
        splitLayout = true;
        drawSplit(picture, photo, broll);
      } else if (full) {
        const shot = node("div", "shot-full");
        shot.append(imageNode("/static/shot.jpg", `shot-image push-${broll.still_push}`));
        drawing.append(shot);
      }
      if (hasBox) {
        const inset = full || split;
        const canvas = node("div", "drawn-picture broll-box-picture");
        boxPictures.push(canvas); drawing.append(canvas);
        if (broll.overlay_box === "tall_behind") {
          tallLayout = true;
          if (inset) {
            const box = node("div", "shot-box photo-inset tall-inset");
            const scene = node("div", "drawn-picture tall-inset-picture");
            drawTallBox(scene, candidate, broll); box.append(scene); canvas.append(box);
          } else drawTallBox(canvas, candidate, broll);
        } else {
          pageLayout = true;
          const box = node("div", `shot-box page-box box-${broll.overlay_box}${inset ? " photo-inset" : ""}`);
          box.append(drawnPage("scroll")); canvas.append(box);
        }
      }
    }
  } else if (screen === "color_match") {
    const shot = node("div", "shot-full");
    shot.append(imageNode("/static/shot.jpg", "shot-image"));
    const color = preview.grade_color?.color;
    if (sectionValues(candidate, "broll").color_match === "on" && /^#[0-9a-f]{6}$/i.test(color ?? "")) {
      const tint = node("div", "shot-tint"); tint.style.backgroundColor = color; shot.append(tint);
    }
    drawing.append(shot);
  } else if (screen === "sound") {
    const shade = node("div", "sound-shade"), meters = node("div", "sound-meters");
    for (const [name, level] of [["Voice", 0.86], ["Music", (candidate.music?.level ?? 0.06) / 0.4]]) {
      const row = node("div", "meter-row"), track = node("div", "meter-track"), fill = node("i", `${name.toLowerCase()}-meter-fill`);
      row.append(node("span", "meter-label", name));
      fill.style.width = `${Math.min(100, Math.max(0, level * 100))}%`;
      fill.style.backgroundColor = candidate.brand?.accent_color || candidate.graphics.accent_color;
      track.append(fill); row.append(track); meters.append(row);
    }
    drawing.append(shade, meters);
  } else if (screen === "video_letters" || (screen === "opening" && candidate.hook.style === "video_letters")) {
    const token = candidate.music?.bleep_swears ? /[\p{L}\p{N}][\p{L}\p{N}*'’\-]*/u : /[\p{L}\p{N}][\p{L}\p{N}'’]*/u;
    lettersWord = node("div", "letters-word", sentence.match(token)?.[0] || "");
    lettersWord.style.fontFamily = fontFamily(screen === "opening" ? graphicsFonts(candidate).title : graphicsFonts(candidate).cards, state.fonts);
    drawing.append(node("div", "letters-shade"), lettersWord);
  } else if (screen === "opening" && candidate.hook.style !== "none") {
    const title = node("div", "drawn-title", sentence.split(/\s+/).slice(0, 5).join(" "));
    Object.assign(title.style, { color: candidate.hook.text_color, backgroundColor: candidate.hook.background_color,
      fontFamily: fontFamily(graphicsFonts(candidate).title, state.fonts) });
    drawing.append(title);
  } else if (screen === "cards") {
    const card = node("div", "drawn-card", sentence.split(/\s+/).slice(0, 5).join(" "));
    Object.assign(card.style, { color: candidate.graphics.ink_color, backgroundColor: candidate.graphics.paper_color,
      fontFamily: fontFamily(graphicsFonts(candidate).cards, state.fonts) });
    drawing.append(card);
  }
  const caption = still || camera3d ? null : captionPoster(host, candidate, !host.classList.contains("compact"));
  if (caption) drawing.append(caption);
  host.replaceChildren(drawing);
  const fit = () => {
    const focus = { x: 540, y: screen === "broll" ? 800 : screen === "sound" ? 850 : 960, width: 0, height: 0 };
    const full = camera3d || main || still || splitLayout || tallLayout || pageLayout || (matchMedia("(max-width: 899px)").matches && !host.classList.contains("compact"));
    const box = camera3d && !main ? camera3dTileWindow(cameraBounds, hostSize(host))
      : full ? { x: 0, y: 0, width: 1080 } : tileWindow(focus, undefined, hostSize(host));
    const scale = camera3d && main ? Math.min((host.clientWidth || 200) / 1080, hostSize(host).height / 1920)
      : (host.clientWidth || 200) / box.width;
    for (const canvas of [picture, ...boxPictures]) {
      Object.assign(canvas.style, { transform: `scale(${scale})`, left: `${-box.x * scale}px`, top: `${-box.y * scale}px` });
      if (camera3d && main) canvas.style.left = `${((host.clientWidth || 200) - 1080 * scale) / 2}px`;
    }
    if (caption && !host.classList.contains("compact")) fitCaption(caption, host, candidate);
    if (lettersWord) lettersWord.style.fontSize = `${(host.clientWidth || 200) * .84 / Math.max(1, lettersWord.textContent.length * .7)}px`;
  };
  const resize = new ResizeObserver(fit);
  host._drawn = { element: drawing, main, visible: previous?.visible ?? false, hovered: previous?.hovered ?? false, resize,
    candidate, screen, still, momentId: moment?.id, usesCutout: camera3d || Boolean(moment) || tallLayout };
  drawings.add(host);
  fit(); resize.observe(host);
  if (still) return;
  if (!host._drawnWired) {
    host.addEventListener("mouseenter", () => { if (host._drawn) { host._drawn.hovered = true; playDrawing(host); } });
    host.addEventListener("mouseleave", () => { if (host._drawn) { host._drawn.hovered = false; playDrawing(host); } });
    host._drawnWired = true;
  }
  drawnObserver.observe(host);
  playDrawing(host);
}
function camera3dTileWindow(bounds, host) {
  const union = (boxes) => {
    const x = Math.min(...boxes.map((box) => box.x)), y = Math.min(...boxes.map((box) => box.y));
    return { x, y, width: Math.max(...boxes.map((box) => box.x + box.w)) - x,
      height: Math.max(...boxes.map((box) => box.y + box.h)) - y };
  };
  const all = union(bounds.all), aspect = host.width / host.height;
  let priority = union(bounds.priority);
  const box = tileWindow(all, undefined, host);
  box.width = Math.min(1080, Math.max(box.width, all.width + 64, (all.height + 64) * aspect));
  box.height = Math.min(1920, box.width / aspect);
  if (bounds.hero) {
    const withHero = union([...bounds.priority, ...bounds.hero]);
    if (withHero.width <= box.width && withHero.height <= box.height) priority = withHero;
  }
  const position = (start, length, first, needed, size, limit) => {
    const low = Math.max(0, first + needed - size), high = Math.min(limit - size, first);
    const preferred = start + length / 2 - size / 2;
    return Math.max(0, Math.min(limit - size, low <= high
      ? Math.max(low, Math.min(high, preferred)) : first + needed / 2 - size / 2));
  };
  box.x = position(all.x, all.width, priority.x, priority.width, box.width, 1080);
  box.y = position(all.y, all.height, priority.y, priority.height, box.height, 1920);
  return box;
}
function camera3dPush(box, front = false) {
  const scale = front ? 1.09 : 1.05;
  return { x: 540 + (box.x - 540 - (front ? 26 : 14)) * scale,
    y: 806.4 + (box.y - 806.4) * scale, w: box.w * scale, h: box.h * scale };
}
function drawCamera3d(picture, candidate) {
  const c = candidate.captions;
  const haloShadows = (color, grade) => {
    const halo = gradeColor(camera3dHalo(color), grade);
    return [[-2, 0], [2, 0], [0, -2], [0, 2], [-1, -1], [1, -1], [-1, 1], [1, 1]]
      .map(([x, y]) => `${x}px ${y}px 0 ${halo}`);
  };
  const shadows = haloShadows(c.text_color);
  shadows.push(`${c.shadow_x_px}px ${c.shadow_y_px}px ${c.shadow_blur_px}px rgba(0,0,0,${c.shadow_opacity})`);
  const faces = preview.face_boxes?.length ? preview.face_boxes : [{ x: 360, y: 420, w: 360, h: 420 }];
  // Match head_box and camera3d_clearance.head_regions, including the paint guard.
  const margin = (candidate.graphics.face_clearance_px ?? 24) + 8.1;
  const heads = faces.map((face) => ({ x: face.x - .15 * face.w - margin, y: face.y - .6 * face.h - margin,
    w: face.w * 1.3 + 2 * margin, h: face.h * 1.6 + 2 * margin }));
  const head = heads[0], safeTop = candidate.internal?.key_text_top_px ?? 280;
  const bottom = Math.min(1720, candidate.internal?.caption_safe_bottom_px ?? 1740);
  // Account for the furthest push and the two pixel halo at the safe top.
  const frontTop = 806.4 + (safeTop + 2 - 806.4) / 1.09;
  const heroTop = 806.4 + (safeTop + 2 - 806.4) / 1.05;
  const bounds = { all: [...heads, ...heads.map((box) => camera3dPush(box))], priority: [head, camera3dPush(head)] };
  let behindBox;
  const scene = node("div", "camera3d-scene"), front = node("div", "camera3d-front");
  scene.append(...picture.children); picture.replaceChildren(scene, front);
  const sentence = maskPreviewText(preview.sentence, candidate, state.bleep_builtin);
  const tokens = sentence.trim().split(/\s+/).filter(Boolean);
  const key = (sentence.match(/[\p{L}\p{N}][\p{L}\p{N}*'’]*/gu) ?? [])
    .sort((a, b) => b.length - a.length)[0] || "";
  if (preview.cutout) {
    const word = node("div", "camera3d-behind", key);
    const size = Math.min(240, 960 / Math.max(1, key.length * .65));
    const above = Math.min(size, (head.y - heroTop) / .9);
    if (above >= 60) {
      behindBox = { x: 60, y: head.y - .9 * above, w: 960, h: above };
    } else {
      const sides = [{ x: 60, w: head.x - 92 }, { x: head.x + head.w + 32, w: 988 - head.x - head.w }];
      const side = sides.sort((a, b) => b.w - a.w)[0];
      const width = Math.max(1, side.w), height = Math.min(size, width / Math.max(1, key.length * .65));
      behindBox = { x: side.x, y: Math.max(heroTop, Math.min(bottom - height, head.y + head.h / 2 - height / 2)),
        w: width, h: height };
    }
    bounds.hero = [behindBox, camera3dPush(behindBox)];
    bounds.all.push(...bounds.hero);
    const accent = heroAccent(candidate);
    Object.assign(word.style, { color: gradeColor(accent, candidate.grade),
      textShadow: haloShadows(accent, candidate.grade).join(", "),
      fontFamily: fontFamily(graphicsFonts(candidate).title, state.fonts),
      left: `${behindBox.x}px`, top: `${behindBox.y}px`, width: `${behindBox.w}px`,
      height: `${behindBox.h}px`, fontSize: `${behindBox.h}px` });
    const person = imageNode(getUrl(cutoutUrl(preview, candidate)), "camera3d-person");
    word.hidden = true;
    person.addEventListener("load", () => { word.hidden = false; });
    person.addEventListener("error", () => { person.hidden = true; word.hidden = true; });
    scene.append(word, person);
  }
  const count = Math.max(1, Math.min(4, c.max_words_per_phrase, Math.ceil(tokens.length / 2)));
  const groupStyle = { ...candidate, captions: { ...c, max_words_per_phrase: count } };
  const first = posterWords(groupStyle, tokens.join(" "));
  const groups = [first, posterWords(groupStyle, tokens.slice(first.length).join(" "))];
  const occupied = behindBox ? [{ ...behindBox, front: false }] : [];
  const separate = (box, item, front = false, gap = 0) => {
    const pairs = [[box, item], [camera3dPush(box, true), camera3dPush(item, front)]];
    // Keep the same separating edge for the entire push, not just its endpoints.
    return pairs.every(([a, b]) => a.x + a.w + gap <= b.x)
      || pairs.every(([a, b]) => a.x >= b.x + b.w + gap)
      || pairs.every(([a, b]) => a.y + a.h + gap <= b.y)
      || pairs.every(([a, b]) => a.y >= b.y + b.h + gap);
  };
  const clear = (box) => [box, camera3dPush(box, true)].every((target) =>
    target.y >= safeTop + 2 - 1e-6 && target.y + target.h <= bottom + 1e-6)
    && heads.every((head) => separate(box, head))
    && occupied.every((item) => separate(box, item, item.front, 24));
  const safeY = (preferred, height) => {
    const positions = [preferred, frontTop, ...heads.map((box) => box.y - height - 24), ...heads.map((box) => box.y + box.h + 24),
      ...occupied.map((box) => box.y - height - 24), ...occupied.map((box) => box.y + box.h + 24), 1560];
    return positions.find((y) => y >= frontTop && clear({ x: 80, y, w: 920, h: height }));
  };
  const font = fontFamily(captionsFont(candidate), state.fonts);
  const weight = ["slam", "clip_wipe"].includes(c.highlight_style) ? "800" : "400";
  const measure = document.createElement("canvas").getContext("2d");
  const textWidth = (text, size) => {
    measure.font = `${weight} ${size}px ${font}`;
    return measure.measureText(text).width;
  };
  const textHeight = (words, width, size) => {
    let lines = 1, line = "";
    for (const word of words) {
      if (line && textWidth(`${line} ${word}`, size) > width - 4) { lines++; line = word; }
      else line = line ? `${line} ${word}` : word;
    }
    return lines * size * 1.12 + 4;
  };
  groups.forEach((words, index) => {
    const line = node("div", "camera3d-words", words.join(" "));
    let size = index ? 60 : 84, width = 920, x = 80, height = textHeight(words, width, size);
    let y = safeY(index ? head.y + head.h + 48 : head.y - 220, height);
    if (y === undefined || (!index && y >= head.y)) {
      const sides = [{ x: 80, width: head.x - 96 }, { x: head.x + head.w + 16, width: 984 - head.x - head.w }];
      for (const side of sides.sort((a, b) => b.width - a.width).filter((box) => box.width >= 120)) {
        const fit = Math.min(size, size * (side.width - 4) / Math.max(1, ...words.map((word) => textWidth(word, size))));
        if (fit < 60) continue;
        const h = textHeight(words, side.width, fit);
        const top = Math.max(frontTop, head.y + head.h / 2 - h / 2);
        if (clear({ x: side.x, y: top, w: side.width, h })) {
          x = side.x; width = side.width; y = top; height = h; size = fit;
          break;
        }
      }
    }
    if (y === undefined) { line.hidden = true; front.append(line); return; }
    const box = { x, y, w: width, h: height };
    occupied.push({ ...box, front: true });
    // Include the front plane's furthest camera3d-push from pick.css in the crop.
    const pushed = camera3dPush(box, true);
    bounds.all.push(box, pushed);
    if (!index) bounds.priority.push(box, pushed);
    Object.assign(line.style, { left: `${x}px`, width: `${width}px`, top: `${y}px`, height: `${height}px`,
      fontSize: `${size}px`, fontFamily: font, color: c.text_color, fontWeight: weight,
      textShadow: shadows.join(", ") });
    front.append(line);
  });
  return bounds;
}
function drawnPage(motion) {
  const page = node("div", `tall-page${motion === "scroll" ? " page-scroll" : ""}`);
  for (let section = 0; section < 8; section++) {
    page.append(node("div", "page-heading"));
    for (let row = 0; row < 4; row++) page.append(node("div", "page-row"));
  }
  return page;
}
function drawTallBox(picture, candidate, broll) {
  const treatment = node("div", `tall-treatment tall-framing-${broll.tall_framing}`);
  const still = getUrl(stillUrl(preview, candidate));
  if (broll.tall_framing === "shrink") treatment.style.backgroundColor = candidate.graphics.paper_color;
  else treatment.append(imageNode(still, "tall-room"));
  const box = node("div", "tall-box");
  box.append(drawnPage(broll.overlay_motion)); treatment.append(box);
  if (preview.cutout) {
    const person = imageNode(getUrl(cutoutUrl(preview, candidate)), "tall-person");
    person.addEventListener("error", () => { person.hidden = true; });
    treatment.append(person);
  }
  picture.replaceChildren(treatment);
}
function drawSplit(picture, photo, broll) {
  const shot = node("div", "split-shot"), speaker = node("div", "split-speaker");
  shot.append(imageNode("/static/shot.jpg", `shot-image push-${broll.still_push}`));
  const face = preview.face_boxes?.[0];
  const scale = face ? Math.max(1, Math.min(1.25, .34 * 960 / face.h)) : 1;
  const width = Math.floor(1080 / scale / 2) * 2, height = Math.floor(960 / scale / 2) * 2;
  const x = face ? Math.max(0, Math.min(1080 - width, Math.round(face.x + face.w / 2 - width / 2))) : 0;
  let y = face ? Math.max(0, Math.min(1920 - height, Math.round(face.y + face.h / 2 - .4 * height))) : 0;
  if (face) y = Math.max(0, Math.min(y, Math.floor(face.y - .4 * face.h - 24 / scale)));
  Object.assign(photo.style, { width: "1080px", height: "1920px", left: `${-x * scale}px`, top: `${-y * scale}px`,
    transform: `scale(${scale})`, transformOrigin: "0 0" });
  picture.replaceChildren(shot, speaker, node("div", "split-seam")); speaker.append(photo);
}
function drawMoment(picture, candidate, id) {
  const treatment = node("div", `moment-treatment moment-${id}${preview.cutout ? "" : " moment-fallback"}`);
  const still = getUrl(stillUrl(preview, candidate));
  treatment.append(imageNode(still, "moment-backdrop"));
  if (id === "spotlight") treatment.append(imageNode(still, "moment-light"));
  const face = preview.face_boxes?.[0];
  const x = face ? face.x + face.w / 2 : 540, y = face ? face.y + face.h / 2 : 960;
  treatment.style.setProperty("--moment-light-x", `${Math.max(0, Math.min(100, x / 1080 * 100))}%`);
  treatment.style.setProperty("--moment-light-y", `${Math.max(0, Math.min(100, y / 1920 * 100))}%`);
  treatment.style.setProperty("--moment-accent", candidate.brand?.accent_color || candidate.graphics.accent_color);
  if (preview.cutout) {
    const person = imageNode(getUrl(cutoutUrl(preview, candidate)), "moment-person");
    person.addEventListener("error", () => { person.hidden = true; treatment.classList.toggle("moment-fallback", true); });
    treatment.append(person);
  }
  picture.append(treatment);
}
function redrawCutouts() {
  for (const detail of stages) {
    if (detail.host.isConnected) syncStageCutout(detail);
  }
  for (const host of [...drawings]) {
    const drawing = host._drawn;
    if (drawing?.usesCutout && host.isConnected) {
      drawPreview(host, drawing.candidate, drawing.screen, drawing.main, drawing.still, drawing.momentId);
    }
  }
}
function foldsVisible(host) {
  for (let parent = host.parentElement; parent; parent = parent.parentElement) {
    if (parent.tagName.toLowerCase() === "details" && !parent.open) return false;
  }
  return true;
}
function rememberFold(details) {
  foldState.set(details.id, details.open);
  try { sessionStorage.setItem(`gallery-fold-${details.id}`, details.open ? "open" : "closed"); } catch { /* Keep the state for this visit. */ }
}
function restoreFold(details) {
  if (!foldState.has(details.id)) {
    try { foldState.set(details.id, sessionStorage.getItem(`gallery-fold-${details.id}`) === "open"); }
    catch { foldState.set(details.id, false); }
  }
  details.open = foldState.get(details.id);
}
function syncFold(details) {
  for (const host of details.querySelectorAll(".stage-host")) {
    if (host._drawn) playDrawing(host);
    if (!host._tileTask) continue;
    if (foldsVisible(host)) {
      if (!host._galleryStage) lazyObserver.observe(host);
    } else {
      lazyObserver.unobserve(host);
      tileQueue = tileQueue.filter((task) => task.host !== host);
      if (host._galleryStage) {
        disposeStage(host._galleryStage);
        staticPoster(host, host._tileTask.candidate);
      }
    }
  }
}
function wireFolds() {
  for (const details of document.querySelectorAll("details")) {
    if (details.classList.contains("category") || details._foldWired) continue;
    restoreFold(details);
    details.addEventListener("toggle", () => { rememberFold(details); syncFold(details); });
    details._foldWired = true;
    syncFold(details);
  }
}
function openField(screen, id) {
  goToStep(screen);
  const field = $(id);
  for (let parent = field.parentElement; parent; parent = parent.parentElement) {
    if (parent.tagName.toLowerCase() === "details") {
      parent.open = true; rememberFold(parent); syncFold(parent);
    }
  }
  field.focus();
  field.scrollIntoView?.({ block: "center" });
}
function routeHash() {
  if (["#colors-fold", "#colors-more"].includes(location.hash)) openField("colors", "color-accent");
  else goToStep(stepFromHash(location.hash, schema), false);
}
function options(path, slots = []) {
  return path.split(".").reduce((node, key) => node.properties[key], schema).enum
    .filter((value) => !slots.some((entry) => entry.requires?.[0] === path && entry.requires[1] === value));
}
function clearGradeColor() {
  choosingGradeColor = false;
  gradeColorRequest += 1;
  gradeColorPending = false;
  $("grade-error").textContent = "";
  $("grade-color").removeAttribute("aria-invalid");
}
function selectStyle(candidate) {
  clearGradeColor();
  if (!["none", "text_hook"].includes(candidate.hook.style)) {
    openingLook = candidate.hook.style;
    if (activeStep === "opening" && ["none", "text_hook"].includes(style.hook.style)) {
      candidate = copy(candidate);
      candidate.hook.style = style.hook.style;
    }
  }
  style = candidate;
  changed();
}
function startFrom(item) {
  clearGradeColor();
  focusedMoment = null;
  style = withPreset(style, item.style, musicChoice);
  openingLook = !["none", "text_hook"].includes(style.hook.style) ? style.hook.style : "quiet_serif";
  fontRole = "captions";
  soundKit = soundKitChoice(style) === "elevenlabs" ? style.music?.sound_effects || "soft" : "soft";
  siteController?.abort("style changed");
  baseStyle = copy(style);
  baseKey = startKey(item);
  lastShadow = null;
  customColors = colorsOf(style);
  $("style-name").value = startingName(style, item.fromLibrary);
  $("site-status").textContent = "";
  $("found-colors").replaceChildren();
  changed({ replaceColors: true });
}
function fontButtons(parent, fonts, graphics = false) {
  const current = graphicsFonts(style);
  for (const font of fonts) {
    const button = node("button", "tile font-tile", font);
    button.type = "button";
    button.style.fontFamily = fontFamily(font, state.fonts);
    button.setAttribute("aria-pressed", String(graphics
      ? current.title === font && current.cards === font : captionsFont(style) === font));
    button.addEventListener("click", () => {
      fontRole = graphics ? "graphics" : "captions";
      selectStyle((graphics ? withGraphicsFont : styleFontMode(style) === "same" ? withFont : withCaptionsFont)(style, font));
    });
    parent.append(button);
  }
}
function renderSteps() {
  const scrollY = window.scrollY;
  clearStages();
  for (const id of ["start-options", "caption-look-tiles", "word-tiles", "highlight-tiles", "font-options",
    "graphics-font-options", "width-tiles", "palette-tiles", "title-tiles", "more-title-tiles", "top-text-tiles", "top-text-options",
    "card-style-tiles", "card-categories", "grade-tiles", "more-choices", "broll-layout-tiles",
    "broll-placements", "broll-overlay-motion", "broll-still-push", "music-level-tiles",
    "moves-tiles", "moves-options", "moments-tiles", "moment-settings", "opening-more-slots", "card-group-slots", "broll-more-slots"]) {
    for (const details of $(id).querySelectorAll("details")) rememberFold(details);
    $(id).replaceChildren();
  }
  if (activeStep === "looks") {
    for (const item of startList(state)) {
      tile($("start-options"), item.name, baseKey === startKey(item),
        { tab: "preview" }, item.style, () => startFrom(item),
        false, item.fromLibrary ? (item.videos ? `Saved · from ${item.videos} finished videos` : "Saved style") : "Preset");
    }
  } else if (activeStep === "font") {
    const graphics = graphicsFonts(style), baseGraphics = graphicsFonts(baseStyle);
    const fonts = [...new Set([...state.fonts.bundled, ...state.fonts.creator,
      captionsFont(baseStyle), baseGraphics.title, baseGraphics.cards,
      captionsFont(style), graphics.title, graphics.cards])];
    fontButtons($("font-options"), fonts);
    fontButtons($("graphics-font-options"), fonts, true);
    const closest = nearestWidth(style.captions.width_scale);
    for (const width of widths) {
      const button = node("button", "", width.name);
      button.type = "button";
      button.setAttribute("aria-pressed", String(closest === width));
      button.addEventListener("click", () => {
        const candidate = copy(style); candidate.captions.width_scale = width.value;
        selectStyle(candidate);
      });
      $("width-tiles").append(button);
    }
  } else if (activeStep === "captions") {
    for (const [look, name, note] of [["plain", "Plain", "Soft shadow, no box"], ["box", "Box", "A dark box behind the line"],
      ["band", "Band", "A band across the frame"], ["one", "One word", "One word at a time"], ["rise", "Rise", "Words rise out of a line"],
      ["spoken", "Word by word", "Words appear as you say them"]]) {
      const candidate = withCaptionLook(style, look);
      tile($("caption-look-tiles"), name, captionLook(style) === look, { tab: "captions" }, candidate,
        () => selectStyle(candidate), false, note);
    }
    renderSlots($("caption-look-tiles"), captionLookSlots, "captions");
    $("camera3d-hint").hidden = !style.captions.camera_3d;
    for (const [mode, choice] of Object.entries({ one: "One word", few: "A few words", phrase: "Short phrase", sentence: "Full sentence" })) {
      const candidate = withWords(style, mode);
      tile($("word-tiles"), choice, wordsMode(style) === mode, { tab: "captions" }, candidate,
        () => selectStyle(candidate));
    }
    for (const look of options("captions.highlight_style", captionLookSlots)) {
      const candidate = copy(style);
      candidate.captions.highlight_style = look;
      if (look === "none") candidate.captions.italic_stress = false;
      tile($("highlight-tiles"), labels[look], style.captions.highlight_style === look,
        { tab: "captions" }, candidate, () => selectStyle(candidate), true);
    }
  } else if (activeStep === "colors") {
    const current = colorsOf(style), match = palettes.find((palette) => sameColors(current, palette));
    for (const palette of [...palettes, { name: "Your colors", ...customColors }]) {
      const candidate = withColors(style, palette);
      const button = tile($("palette-tiles"), palette.name,
        match ? match.name === palette.name : palette.name === "Your colors",
        { tab: "captions" }, candidate, () => {
          style = candidate; customColors = colorsOf(style); changed({ replaceColors: true });
          if (palette.name === "Your colors") {
            const details = $("colors-more");
            details.open = true; rememberFold(details); syncFold(details);
            $("color-fields").scrollIntoView?.({ block: "center" });
          }
        });
      const chips = node("span", "palette-chips");
      for (const [role] of roles) chips.append(colorDot(palette[role]));
      button.append(chips);
    }
  } else if (activeStep === "more") {
    optionalScreens().forEach((screen, index) => {
      const row = node("button", "choice-row");
      row.type = "button"; row.id = `choice-${screen.id}`;
      const thumbnail = node("span", "choice-thumb");
      drawPreview(thumbnail, style, screen.id, false, true);
      const text = node("span", "choice-text");
      text.append(node("strong", "", screen.name), node("small", "", summaryFor(screen.id)));
      row.append(thumbnail, node("span", "choice-number", String(index + 5)), text, node("span", "choice-change", "Change"));
      row.addEventListener("click", () => goToStep(screen.id));
      $("more-choices").append(row);
    });
  } else if (activeStep === "grade") {
    for (const [look, name] of gradeLooks) {
      const color = look === "eyedropper" ? style.grade?.color || preview.grade_color?.color : style.grade?.color;
      const needsColor = look === "eyedropper" && !style.grade?.color && !preview.grade_color?.colorful;
      const strength = look === "eyedropper" && style.grade?.look !== "eyedropper"
        ? preview.grade_default_strength : style.grade?.strength ?? 1;
      const candidate = needsColor ? withGrade(style, "none") : withGrade(style, look, strength, color);
      tile($("grade-tiles"), name, (style.grade?.look ?? "none") === look,
        { tab: "captions" }, candidate, () => {
          if (needsColor) {
            clearGradeColor(); choosingGradeColor = true;
            $("grade-error").textContent = preview.grade_color?.message || "Pick a more colorful point or color.";
            renderControls(); updateMain();
          } else selectStyle(candidate);
        }, false, { none: "Your picture as shot", film: "Warmer, softer contrast", eyedropper: "Matched to a color you pick" }[look]);
    }
  } else if (activeStep === "top-text") {
    for (const [id, name, note] of topTextAnswers) {
      const candidate = withTopText(style, id, openingLook);
      const button = tile($("top-text-tiles"), name, topTextChoice(style) === id,
        { tab: "preview", step: "top-text" }, candidate, () => selectStyle(candidate), false, note);
      button.id = `top-text-${id}`;
    }
    if (style.hook.style === "text_hook") renderSlotGroups($("top-text-options"),
      openingMoreSlots.filter((entry) => entry.id === "text-hook-color"), "top-text");
  } else if (activeStep === "opening") {
    const main = [...new Set([baseStyle.hook.style, "calm_card", "headline_slam", "shimmer", "typewriter", "video_letters"])]
      .filter((look) => !["none", "text_hook"].includes(look));
    for (const look of options("hook.style", openingSlots)) {
      if (main.includes(look) || ["none", "text_hook"].includes(look)) continue;
      const candidate = copy(style); candidate.hook.style = look;
      tile($("more-title-tiles"), labels[look],
        openingLook === look, { tab: "title" }, candidate, () => selectStyle(candidate));
    }
    for (const look of main) {
      const candidate = copy(style); candidate.hook.style = look;
      tile($("title-tiles"), look === "none" ? "No title" : labels[look], openingLook === look,
        { tab: "title" }, candidate, () => selectStyle(candidate), false,
        look === baseStyle.hook.style ? `${baseStyle.name}'s title` : {
          calm_card: "A simple title card", headline_slam: "A bold opening", shimmer: "A soft sweep of light",
          typewriter: "Letters appear one at a time", video_letters: "A big word filled with your footage", none: "Start straight on you",
        }[look]);
    }
    renderSlots($("title-tiles"), openingSlots.filter((entry) => !main.includes(entry.requires?.[1])), "opening");
    renderSlotGroups($("opening-more-slots"), openingMoreSlots.filter((entry) =>
      entry.id !== "text-hook-color"), "opening");
  } else if (activeStep === "cards") {
    for (const look of options("graphics.style")) {
      const candidate = withCardStyle(style, look);
      tile($("card-style-tiles"), labels[look], style.graphics.style === look,
        { tab: "cards", card: "text_tag" }, candidate, () => {
          style = candidate; customColors = colorsOf(style); changed({ replaceColors: true });
        }, false, { paper: "A paper card", minimal: "Simple and clean", dark: "Dark paper, light text", bright: "Your accent color" }[look]);
    }
    cardCategories();
    renderSlotGroups($("card-group-slots"), cardGroupSlots, "cards");
  } else if (activeStep === "broll") {
    for (const [layout, name, note] of [["full", "Full screen", "The shot fills the frame"], ["mixed", "Mixed", "Clips fill the frame, photos sit in a box"], ["box", "In a box", "A box above your head"]]) {
      const candidate = withBrollLayout(style, layout, schema);
      tile($("broll-layout-tiles"), name, displayLayout(style) === layout, "broll", candidate,
        () => selectStyle(candidate), false, note);
    }
    renderSlots($("broll-layout-tiles"), brollLayoutSlots, "broll");
    for (const [kind, name] of [["screen_recording", "Screen recordings"], ["clip", "Clips"], ["photo", "Photos"], ["screenshot", "Screenshots"]]) {
      const row = node("div", "placement-row");
      row.append(node("span", "", name));
      const buttons = node("div", "seg"); buttons.id = `broll-${kind}`;
      for (const value of options(`broll.${kind}_placement`)) {
        const label = { full_frame: "Full screen", overlay: "In a box", split: "Split" }[value];
        // A placement the schema adds before this page has a button for it stays off the page.
        if (!label) continue;
        const button = node("button", "", label); button.type = "button";
        button.setAttribute("aria-pressed", String(sectionValues(style, "broll")[`${kind}_placement`] === value));
        button.addEventListener("click", () => selectStyle(withBrollPlacement(style, kind, value, schema)));
        buttons.append(button);
      }
      row.append(buttons); $("broll-placements").append(row);
    }
    for (const [field, choices] of [["overlay_motion", [["none", "Still"], ["drift", "Drift"], ["scroll", "Scroll"]]],
      ["still_push", [["in", "Push in"], ["out", "Pull out"], ["none", "Off"]]]]) {
      for (const [value, name] of choices) {
        const candidate = withBrollField(style, field, value);
        const placement = sectionValues(style, "broll").photo_placement;
        const visual = withBrollLayout(candidate, field === "still_push"
          ? { full_frame: "full", overlay: "box", split: "split" }[placement] : "box", schema);
        tile($(field === "overlay_motion" ? "broll-overlay-motion" : "broll-still-push"), name,
          sectionValues(style, "broll")[field] === value, "broll", visual, () => selectStyle(candidate));
      }
    }
    renderSlotGroups($("broll-more-slots"), brollMoreSlots, "broll");
  } else if (activeStep === "sound") {
    for (const level of musicLevels) {
      const candidate = withMusicLevel(style, level.value);
      const button = tile($("music-level-tiles"), level.name, (style.music?.level ?? 0.06) === level.value,
        "sound", candidate, () => {
          if (state.music_answer === "none") musicChoice = level.value;
          selectStyle(candidate);
        }, false, level.note);
      button.dataset.level = String(level.value);
    }
  } else if (activeStep === "moves") {
    renderSlots($("moves-tiles"), movesSlots.filter((entry) => !entry.type), "moves");
    const strength = {
      id: "zoom-strength", name: "Zoom strength", options: zoomStrengths,
      note: "Moves in fast, holds, and moves out at the end of the phrase.",
      value: `${sectionValues(style, "moves").punch_scale}x`,
    };
    const groups = [...(sectionValues(style, "moves").zoom_mode !== "none"
      && movesSlots.some((entry) => !entry.type && optionAvailable(entry, schema))
      && zoomStrengths.some((entry) => optionAvailable(entry, schema)) ? [strength] : []),
    ...movesSlots.filter((entry) => entry.type), ...movesMoreSlots];
    $("moves-more").hidden = !groups.some((entry) => optionAvailable(entry, schema));
    $("moves-more-hint").textContent = groups.filter((entry) => optionAvailable(entry, schema)).map((entry) => entry.name).join(", ");
    renderSlotGroups($("moves-options"), groups, "moves");
  } else if (activeStep === "moments") {
    for (const entry of availableMoments()) {
      const button = tile($("moments-tiles"), entry.name + (entry.selected(style) ? " · On" : ""), currentMoment()?.id === entry.id,
        "moments", style, () => {
          focusedMoment = entry.id;
          for (const tile of $("moments-tiles").children) tile.setAttribute("aria-pressed", String(tile.dataset.moment === entry.id));
          renderMomentSettings(); updateMain();
        }, false, momentBlock(entry.id, style, preview, state.video_type) ?? entry.note, entry.id);
      button.id = `moment-tile-${entry.id}`; button.dataset.moment = entry.id;
    }
    renderMomentSettings();
  }
  window.scrollTo(0, scrollY);
}
function renderMomentSettings() {
  const entry = currentMoment(), panel = $("moment-settings");
  panel.replaceChildren();
  if (!entry) return;
  panel.append(node("h3", "", entry.name));
  const toggle = node("input"); toggle.type = "checkbox"; toggle.role = "switch";
  toggle.id = `moment-${entry.id}`; toggle.checked = Boolean(entry.selected(style));
  const reason = momentBlock(entry.id, style, preview, state.video_type);
  toggle.disabled = Boolean(reason) && !toggle.checked;
  toggle.addEventListener("change", () => { focusedMoment = entry.id; selectStyle(entry.apply(copy(style), toggle.checked)); });
  const label = node("label", "switch-row"); label.htmlFor = toggle.id;
  label.append(node("span", "", "Use in my videos"), toggle); panel.append(label);
  if (reason) {
    const note = node("p", "field-hint", reason); note.id = "moment-reason";
    toggle.setAttribute("aria-describedby", note.id); panel.append(note);
  }
}
function renderSlots(parent, entries, screen) {
  for (const entry of entries.filter((item) => optionAvailable(item, schema))) {
    const candidate = entry.apply(copy(style), true, schema);
    const request = entry.preview ?? (drawnScreen(screen) ? screen : { tab: screen === "opening" ? "title" : screen === "cards" ? "cards" : "captions" });
    if (screen === "cards" && entry.card && !entry.preview) request.card = entry.card;
    const selected = screen === "opening" && entry.requires?.[0] === "hook.style"
      ? openingLook === entry.requires[1] : entry.selected(style, schema);
    const button = tile(parent, entry.name, selected, request, candidate,
      () => selectStyle(candidate), false, entry.note);
    button.id = `slot-${screen}-${entry.id}`;
  }
}
function renderSlotGroups(parent, entries, screen) {
  for (const entry of entries.filter((item) => optionAvailable(item, schema))) {
    if (!entry.options?.length && entry.type !== "switch") {
      if (screen !== "moves") renderSlots(parent, [entry], screen);
      continue;
    }
    const details = node("details", "sub-fold"); details.id = `slot-fold-${screen}-${entry.id}`;
    const summary = node("summary", "", entry.name);
    details.append(summary);
    parent.append(details);
    if (entry.type === "switch") {
      const value = node("span", "fold-value", entry.value ? entry.value(style) : entry.selected(style) ? "On" : "Off");
      value.id = `${details.id}-value`; summary.append(value);
      for (const choice of (entry.switches ?? [entry]).filter((item) => optionAvailable(item, schema))) {
        const row = node("div", "slot-switch-choice");
        const toggle = node("input"); toggle.type = "checkbox"; toggle.role = "switch";
        toggle.id = `slot-switch-${screen}-${choice.id}`; toggle.checked = Boolean(choice.selected(style));
        toggle.addEventListener("change", () => selectStyle(choice.apply(copy(style), toggle.checked, schema)));
        const label = node("label", "switch-row"); label.htmlFor = toggle.id;
        const text = node("span", "slot-switch-copy");
        if (entry.switches) text.append(node("strong", "", choice.name), node("small", "", choice.note));
        else text.textContent = choice.note;
        label.append(text, toggle);
        if (choice.preview) {
          const poster = node("div", "slot-switch-preview tile-frame stage-host");
          const candidate = choice.previewCurrent ? copy(style) : choice.apply(copy(style), true, schema);
          if (typeof choice.preview === "string") drawPreview(poster, candidate, choice.preview);
          else queueTile(poster, choice.preview, candidate);
          row.append(poster);
        }
        row.append(label); details.append(row);
        if (choice.hint) details.append(node("p", "field-hint", choice.hint));
        if (choice.selected(style)) renderSlots(details, choice.options ?? [], screen);
      }
    } else {
      if (entry.value) summary.append(node("span", "fold-value", typeof entry.value === "function" ? entry.value(style) : entry.value));
      if (entry.note) details.append(node("p", "field-hint", entry.note));
      const grid = node("div", "tile-grid"); details.append(grid);
      renderSlots(grid, entry.options, screen);
    }
  }
  wireFolds();
}
function cardCategories() {
  const groups = {};
  for (const [kind, { category }] of Object.entries(state.card_kinds)) (groups[category] ??= []).push(kind);
  for (const [category, names] of Object.entries(groups)) {
    const details = node("details", "category"); details.id = `card-group-${category.toLowerCase()}`;
    const summary = node("summary"), text = node("span", "", `${["Chat", "Editor", "Video"].includes(category) ? category + " cards" : category} · ${kindCount(names.length)}`);
    const status = node("span", "group-state");
    const updateCount = () => {
      const count = names.filter((kind) => kindOn(style, kind, Boolean(state.card_kinds[kind].opt_in))).length;
      status.textContent = count === 0 ? "off" : count === names.length ? "all on" : `${count} on`;
    };
    summary.append(text);
    if (names.some((kind) => state.card_kinds[kind].opt_in)) summary.append(node("small", "new-tag", "new"));
    summary.append(status); details.append(summary); updateCount();
    const removeRows = () => {
      for (const host of details.querySelectorAll(".stage-host")) {
        lazyObserver.unobserve(host);
        tileQueue = tileQueue.filter((task) => task.host !== host);
        if (host._galleryStage) disposeStage(host._galleryStage);
      }
      for (const row of [...details.querySelectorAll(".kind-row")]) row.remove();
    };
    const openRows = () => {
      if (!details.open) { removeRows(); return; }
      if (details.children.length > 1) return;
      const allRow = node("label", "kind-row all-kinds");
      const all = node("input"); all.type = "checkbox"; all.setAttribute("role", "switch");
      all.id = `card-all-${category.toLowerCase()}`;
      all.setAttribute("aria-label", `All ${category} kinds`);
      const enabled = names.filter((kind) => kindOn(style, kind, Boolean(state.card_kinds[kind].opt_in))).length;
      all.checked = enabled === names.length; all.indeterminate = enabled > 0 && enabled < names.length;
      all.addEventListener("change", () => {
        for (const kind of names) style = withKind(style, kind, all.checked, Boolean(state.card_kinds[kind].opt_in));
        changed();
      });
      allRow.append(node("span", "", "All"), all); details.append(allRow);
      for (const kind of names) {
        const row = node("div", "kind-row"), frame = node("button", "kind-preview stage-host");
        if (kind === "big_number") frame.className += " compact";
        frame.type = "button"; frame.setAttribute("aria-label", `Preview ${kindLabel(state, kind)}`);
        frame.addEventListener("click", () => {
          if (frame._galleryStage) for (const iframe of frame._galleryStage.frames) runPlayer(iframe, frame._galleryStage, "play");
          const candidate = copy(style);
          mainController?.abort();
          const controller = new AbortController(); mainController = controller;
          api("/api/layers", { style: candidate, tile: { tab: "cards", step: "cards", card: kind } }, controller.signal)
            .then((spec) => {
              if (controller.signal.aborted || activeStep !== "cards" || !details.open || !frame.isConnected) return;
              if (mainStage) disposeStage(mainStage);
              mainStage = mount($("main-frame"), spec, true, null, candidate);
            }).catch((error) => { if (error.name !== "AbortError") setPreviewNote(error.message, true); });
        });
        const label = node("label", "", `${kindLabel(state, kind)}${kind === "stat" ? " · Sample values" : ""}`);
        if (kind === "big_number") label.append(node("small", "kind-note", "Counts up behind your head"));
        const toggle = node("input"); toggle.type = "checkbox"; toggle.setAttribute("role", "switch");
        toggle.id = `kind-${kind}`; label.htmlFor = toggle.id;
        toggle.checked = kindOn(style, kind, Boolean(state.card_kinds[kind].opt_in));
        toggle.setAttribute("aria-label", `${kindLabel(state, kind)} on`);
        toggle.addEventListener("change", () => { style = withKind(style, kind, toggle.checked, Boolean(state.card_kinds[kind].opt_in)); changed(); });
        row.append(frame, label, toggle); details.append(row);
        queueTile(frame, { tab: "cards", card: kind });
      }
    };
    restoreFold(details);
    details.addEventListener("toggle", () => { rememberFold(details); openRows(); syncFold(details); });
    $("card-categories").append(details); openRows();
  }
}
function colorDot(color) {
  const dot = document.createElement("span");
  dot.className = "color-dot";
  dot.style.backgroundColor = color;
  return dot;
}
function buildColorFields() {
  for (const [id, name] of roles) {
    const row = document.createElement("div");
    row.className = "color-role";
    const label = document.createElement("label");
    label.htmlFor = `color-${id}`;
    label.textContent = name;
    const picker = document.createElement("input");
    picker.type = "color";
    picker.id = `picker-${id}`;
    picker.setAttribute("aria-label", `Pick a color for ${name.toLowerCase()}`);
    const input = document.createElement("input");
    input.type = "text";
    input.id = `color-${id}`;
    input.maxLength = 7;
    input.spellcheck = false;
    input.setAttribute("aria-describedby", `error-${id}`);
    const error = document.createElement("span");
    error.id = `error-${id}`;
    error.className = "error";
    error.setAttribute("role", "alert");
    for (const field of [picker, input]) field.addEventListener("focus", () => { activeRole = id; });
    picker.addEventListener("input", () => setColor(id, picker.value));
    input.addEventListener("input", () => {
      const value = normalizeHex(input.value);
      error.textContent = "";
      input.removeAttribute("aria-invalid");
      $("reset").disabled = false;
      if (value && input.value.replace(/^#/, "").length === 6) setColor(id, value);
    });
    input.addEventListener("change", () => setColor(id, input.value));
    row.append(label, picker, input, error);
    $("color-fields").append(row);
  }
}
function setColor(id, raw) {
  const value = normalizeHex(raw);
  if (!value) {
    $(`error-${id}`).textContent = "Use a hex code like #2F3BFF.";
    $(`color-${id}`).setAttribute("aria-invalid", "true");
    $("reset").disabled = false;
    return;
  }
  $(`error-${id}`).textContent = "";
  $(`color-${id}`).removeAttribute("aria-invalid");
  $(`color-${id}`).value = value;
  // A manual edit starts with the colors currently on screen, including a stock palette.
  customColors = { ...colorsOf(style), [id]: value };
  style = withColors(style, customColors, [id]);
  changed();
}
function buildFineFields() {
  for (const [field, label, step] of [["size_px", "Size", 1], ["width_scale", "Letter width", 0.01],
    ["max_words_per_phrase", "Most words on screen", 1], ["max_characters_per_phrase", "Most characters", 1],
    ["top_px", "Caption position", 1], ["background_opacity", "Background strength", 0.01],
    ["background_padding_px", "Background padding", 1], ["background_radius_px", "Box corner radius", 1]]) {
    const rule = schema.properties.captions.properties[field];
    const wrapper = document.createElement("label");
    wrapper.className = "field";
    wrapper.id = `fine-field-${field}`;
    wrapper.textContent = label;
    const value = document.createElement("output");
    value.id = `fine-value-${field}`;
    const input = document.createElement("input");
    input.type = "range";
    input.id = `fine-${field}`;
    Object.assign(input, { min: rule.minimum, max: rule.maximum, step });
    input.addEventListener("input", () => {
      style.captions[field] = Number(input.value);
      changed();
    });
    wrapper.append(value, input);
    if (field === "top_px") {
      const position = document.createElement("label"), slider = document.createElement("input");
      position.className = "field";
      position.id = "caption-position-field";
      position.textContent = label;
      slider.id = "caption-position";
      slider.type = "range";
      Object.assign(slider, { min: rule.minimum, max: rule.maximum, step });
      slider.addEventListener("input", () => {
        style.captions.top_px = Number(slider.value);
        changed();
      });
      position.append(slider);
      $("position-fields").append(position);
    }
    $("fine-fields").append(wrapper);
  }
}
function renderBackground({ replaceColors = false } = {}) {
  const captions = style.captions, enabled = !captions.camera_3d && captions.background_mode !== "none";
  $("background-fields").hidden = !enabled;
  $("background-corner-fields").hidden = captions.background_mode !== "box";
  const input = $("background-color");
  if (!enabled || replaceColors || (document.activeElement !== input && input.getAttribute("aria-invalid") !== "true")) {
    input.value = captions.background_color;
    input.removeAttribute("aria-invalid");
    $("background-error").textContent = "";
  }
  $("background-picker").value = captions.background_color;
  $("background-swatches").replaceChildren();
  for (const [name, color] of [["Black", "#000000"], ["White", "#FFFFFF"],
    ["Highlight", style.brand?.accent_color || captions.highlight_color]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", String(color.toUpperCase() === captions.background_color.toUpperCase()));
    button.append(colorDot(color), name);
    button.addEventListener("click", () => {
      $("background-color").value = color;
      setBackgroundColor();
    });
    $("background-swatches").append(button);
  }
  for (const button of $("background-opacity").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(backgroundOpacities[button.dataset.opacity] === captions.background_opacity));
  }
  for (const button of $("background-corners").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(backgroundCorners[button.dataset.corners] === captions.background_radius_px));
  }
  $("background-opacity-note").textContent = "Exact strength is under Fine-tune.";
  const contrastNote = captionContrastNote(style);
  $("caption-result").hidden = !contrastNote;
  $("caption-result").replaceChildren(colorDot(captionTextColor(style)), ` ${contrastNote}`);
  for (const field of ["background_opacity", "background_padding_px", "background_radius_px"]) {
    $(`fine-field-${field}`).hidden = !enabled || (field === "background_radius_px" && captions.background_mode !== "box");
  }
}
function setBackgroundColor() {
  const input = $("background-color"), value = normalizeHex(input.value);
  $("background-error").textContent = value ? "" : "Use a hex code like #2F3BFF.";
  if (!value) {
    input.setAttribute("aria-invalid", "true");
    $("reset").disabled = false;
    return;
  }
  input.value = value;
  input.removeAttribute("aria-invalid");
  style.captions.background_color = value;
  changed();
}
function renderControls({ replaceColors = false } = {}) {
  renderBackground({ replaceColors });
  const pickingColor = activeStep === "grade" && (style.grade?.look === "eyedropper" || choosingGradeColor);
  $("grade-color-fields").hidden = !pickingColor;
  $("grade-hint").hidden = !pickingColor;
  $("main-frame").setAttribute("aria-label", pickingColor ? "Tap your video to pick a color" : "Enlarge preview");
  if (replaceColors || (document.activeElement !== $("grade-color") && $("grade-color").getAttribute("aria-invalid") !== "true")) {
    $("grade-color").value = style.grade?.color || "";
    $("grade-color").removeAttribute("aria-invalid");
  }
  $("grade-picker").value = style.grade?.color || "#1E6BD6";
  $("grade-strength").hidden = (style.grade?.look ?? "none") === "none";
  $("grade-more").hidden = $("grade-strength").hidden;
  $("grade-strength-label").hidden = $("grade-strength").hidden;
  $("grade-slider").hidden = $("grade-strength").hidden;
  $("grade-slider-label").hidden = $("grade-slider").hidden;
  for (const button of $("grade-slider").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String((button.dataset.gradeSlider === "true") === Boolean(style.grade?.slider)));
  }
  for (const button of $("grade-strength").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.gradeStrength) === (style.grade?.strength ?? 1)));
  }
  $("avoid-face").checked = style.captions.placement !== "fixed";
  $("avoid-face").setAttribute("aria-checked", String($("avoid-face").checked));
  $("placement-note").textContent = placementNote(style, preview);
  for (const button of $("italics").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.italic === String(style.captions.italic_stress)));
  }
  const motion = motions.reduce((best, choice) =>
    Math.abs(choice.rise - style.captions.entry_rise_px) < Math.abs(best.rise - style.captions.entry_rise_px) ? choice : best);
  for (const button of $("motion").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.motion === motion.name));
  }
  for (const button of $("placements").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.place === placementMode(style)));
  }
  const shadow = shadowMode(style);
  for (const button of $("shadow-toggle").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.shadowToggle === (shadow === "off" ? "off" : "on")));
  }
  for (const button of $("shadow-strength").querySelectorAll("button")) {
    button.disabled = shadow === "off";
    button.setAttribute("aria-pressed", String(button.dataset.shadow === shadow));
  }
  for (const button of $("sizes").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.size === nearestSize(style)));
  }
  const colors = colorsOf(style);
  if (replaceColors || (document.activeElement !== $("caption-color") && $("caption-color").getAttribute("aria-invalid") !== "true")) {
    $("caption-color").value = colors.caption;
    $("caption-color").removeAttribute("aria-invalid");
    $("caption-error").textContent = "";
  }
  $("caption-picker").value = colors.caption;
  for (const [id] of roles) {
    const input = $(`color-${id}`);
    if (replaceColors || (document.activeElement !== input && input.getAttribute("aria-invalid") !== "true")) {
      input.value = colors[id];
      input.removeAttribute("aria-invalid");
      $(`error-${id}`).textContent = "";
    }
    $(`picker-${id}`).value = colors[id];
  }
  const warnings = colorWarnings(style);
  $("color-warning").hidden = !warnings.length;
  $("color-warning").textContent = warnings.join(" ");
  const graphics = graphicsFonts(style);
  $("font-note").textContent = `Captions: ${captionsFont(style)} · Title: ${graphics.title} · Cards: ${graphics.cards}`;
  $("font-same").setAttribute("aria-pressed", String(styleFontMode(style) === "same"));
  $("font-pair").setAttribute("aria-pressed", String(captionsFont(style) === fontPair.captions
    && graphics.title === fontPair.graphics && graphics.cards === fontPair.graphics));
  const width = style.captions.width_scale, closest = nearestWidth(width);
  $("width-note").textContent = width === closest.value ? "Exact width is under Fine-tune in Captions."
    : `Custom width: ${Number((width * 100).toFixed(2))}%. Closest choice: ${closest.name}. Exact width is under Fine-tune in Captions.`;
  $("size-note").textContent = "Exact size is under Fine-tune.";
  $("caption-position").value = style.captions.top_px;
  $("look-selected").textContent = baseStyle.name;
  $("title-note").textContent = style.hook.style === "none" ? "Your video starts straight on you, with no title."
    : style.hook.style === "text_hook" ? "Stays on screen for the whole video."
    : style.hook.style === "video_letters" ? "Shows for the first few seconds."
    : `Shows for the first ${Number(style.hook.duration_s.toFixed(1))} seconds.`;
  const music = sectionValues(style, "music");
  $("music").value = music.level;
  $("music-value").textContent = `${Number((music.level * 100).toFixed(2))}%${music.level === 0 ? " · Off" : ""}`;
  for (const field of ["size_px", "width_scale", "max_words_per_phrase", "max_characters_per_phrase", "top_px",
    "background_opacity", "background_padding_px", "background_radius_px"]) {
    $(`fine-${field}`).value = style.captions[field];
    $(`fine-value-${field}`).textContent = ["width_scale", "background_opacity"].includes(field) ? `${Number((style.captions[field] * 100).toFixed(2))}%` : style.captions[field];
  }
  $("change-count").textContent = changeSummary(style, baseStyle);
  renderValues();
  $("reset").disabled = !choosingGradeColor && !gradeColorPending
    && JSON.stringify(style) === JSON.stringify(baseStyle)
    && normalizeHex($("caption-color").value) === colors.caption.toUpperCase()
    && normalizeHex($("background-color").value) === style.captions.background_color.toUpperCase()
    && roles.every(([id]) => normalizeHex($(`color-${id}`).value) === colors[id].toUpperCase());
}
function renderValues() {
  const value = (id, text) => { const target = $(`${id}-value`); if (target) target.textContent = text; };
  const broll = sectionValues(style, "broll"), voice = sectionValues(style, "voice"), music = sectionValues(style, "music");
  const c = style.captions, g = graphicsFonts(style), percent = (number) => `${Number((number * 100).toFixed(2))}%`;
  value("graphics-font-fold", styleFontMode(style) === "same" ? "Same as captions"
    : g.title === g.cards ? g.title : `${g.title} · ${g.cards}`);
  const width = nearestWidth(c.width_scale);
  value("width-fold", width.value === c.width_scale ? width.name : percent(c.width_scale));
  value("words-fold", { one: "One word", few: "A few words", phrase: "Short phrase", sentence: "Full sentence" }[wordsMode(style)]);
  value("highlight-fold", `${labels[c.highlight_style]} · ${kindCount(options("captions.highlight_style", captionLookSlots).length)}`);
  value("position-fold", `${placementMode(style)[0].toUpperCase()}${placementMode(style).slice(1)}`);
  const size = nearestSize(style);
  value("size-fold", size[0].toUpperCase() + size.slice(1));
  value("caption-color-fold", `${c.text_color} · shadow ${shadowMode(style)}`);
  value("background-fields", c.background_color);
  value("motion-fold", ["rise", "spoken"].includes(c.entrance) ? labels[c.entrance] : motions.reduce((best, entry) => Math.abs(entry.rise - c.entry_rise_px) < Math.abs(best.rise - c.entry_rise_px) ? entry : best).name);
  value("fine-tune", "Exact values");
  value("site-fold", pulledSite);
  value("grade-strength-fold", (style.grade?.strength ?? 1) <= 0.5 ? "Light" : "Full");
  value("grade-slider-fold", style.grade?.slider ? "On" : "Off");
  value("more-titles-fold", summaryFor("opening"));
  value("broll-kinds-fold", { full: "Full screen", mixed: "Mixed", box: "In a box", split: "Split", custom: "Custom" }[displayLayout(style)]);
  value("broll-box-fold", { headroom: "Above your head", side_left: "Left side", side_right: "Right side", tall_behind: "Tall, behind your head" }[broll.overlay_box]);
  value("broll-motion-fold", ({ none: "Still", drift: "Drift", scroll: "Scroll" })[broll.overlay_motion] || "Still");
  value("broll-framing-fold", labels[broll.tall_framing]);
  $("broll-framing-fold").hidden = broll.overlay_box !== "tall_behind";
  value("broll-push-fold", { in: "Push in", out: "Pull out", none: "Off" }[broll.still_push]);
  value("broll-stock-fold", broll.stock_clips === "ask" ? "Ask first" : "Off");
  value("music-fold", percent(music.level));
  value("effects-fold", { made: "Made by Yap Studio", elevenlabs: "ElevenLabs", none: "Off" }[soundKitChoice(style)]);
  value("beat-fold", music.beat_snap ? "On" : "Off");
  value("voice-fold", voice.preset === "none" ? "Off" : labels[voice.preset]);
  for (const [id, field, attribute] of [["broll-overlay-box", "overlay_box", "overlayBox"], ["broll-stock-clips", "stock_clips", "stockClips"], ["broll-tall-framing", "tall_framing", "tallFraming"]]) {
    for (const button of $(id).querySelectorAll("button")) button.setAttribute("aria-pressed", String(broll[field] === button.dataset[attribute]));
  }
  for (const button of $("sound-effects").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.effects === soundKitChoice(style)));
  for (const button of $("voice-preset").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.voicePreset === voice.preset));
  for (const button of $("repair-proximity").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.repairProximity === voice.repair_proximity));
  $("sound-fallback-row").hidden = soundKitChoice(style) !== "elevenlabs";
  $("sound-fallback").checked = (music.sound_effects_fallback ?? "made") === "made";
  $("sound-extras").hidden = soundKitChoice(style) !== "elevenlabs";
  for (const name of ["typing", "message-ping", "camera-shutter"]) {
    $("sound-extra-" + name).checked = (music.sound_extras ?? []).includes(name);
  }
  $("beat-snap").checked = music.beat_snap;
  $("bleep-swears").checked = music.bleep_swears === true;
  value("bleep-fold", music.bleep_swears ? "On" : "Off");
  for (const field of ["repair_rumble", "repair_room", "repair_harshness"]) $(field.replaceAll("_", "-")).checked = voice[field];
  $("music-custom").hidden = musicLevels.some((level) => level.value === music.level);
  $("music-custom").textContent = `Your look uses ${percent(music.level)}.`;
  for (const button of $("music-level-tiles").children) button.setAttribute("aria-pressed", String(Number(button.dataset.level) === music.level));
  if (activeStep === "sound") for (const fill of $("main-frame").querySelectorAll(".music-meter-fill")) fill.style.width = `${Math.min(100, music.level / 0.4 * 100)}%`;
  $("more-note").textContent = `${baseStyle.name} already set these. Change any one; the rest stay as they are.`;
  const current = screens.find(({ id, optional }) => optional && id === activeStep);
  $("screen-summary").textContent = current ? `${current.name}: ${summaryFor(current.id)}` : "";
}
async function setGradeColor(raw) {
  const request = ++gradeColorRequest;
  gradeColorPending = false;
  const input = $("grade-color"), value = normalizeHex(raw);
  if (!value) {
    $("grade-error").textContent = "Use a hex code like #2F3BFF.";
    input.setAttribute("aria-invalid", "true");
    return;
  }
  const selection = style, seen = generation;
  gradeColorPending = true;
  try {
    const info = await api(`/api/grade-color?color=${encodeURIComponent(value)}`);
    if (request !== gradeColorRequest || style !== selection || generation !== seen) return;
    if (!info.colorful) throw new Error(info.message);
    style = withGrade(style, "eyedropper", style.grade?.look === "eyedropper"
      ? style.grade.strength : preview.grade_default_strength, info.color);
    choosingGradeColor = false;
    $("grade-error").textContent = "";
    input.value = info.color;
    input.removeAttribute("aria-invalid");
    changed();
  } catch (error) {
    if (request !== gradeColorRequest || style !== selection || generation !== seen) return;
    $("grade-error").textContent = error.message;
    $("grade-picker").value = style.grade?.color || "#1E6BD6";
    input.setAttribute("aria-invalid", "true");
  } finally {
    if (request === gradeColorRequest) gradeColorPending = false;
  }
}
async function sampleGrade(event) {
  if (event.detail === 0 || !Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return;
  const request = ++gradeColorRequest;
  gradeColorPending = true;
  const selection = style, seen = generation;
  const bounds = $("main-frame").getBoundingClientRect();
  try {
    const image = document.createElement("img");
    image.src = getUrl(stillUrl(preview, { grade: { look: "none" } }, true));
    await image.decode();
    if (request !== gradeColorRequest || style !== selection || generation !== seen) return;
    const point = coverPoint(bounds, image.naturalWidth, image.naturalHeight, event.clientX, event.clientY);
    const patch = samplePatch(image.naturalWidth, image.naturalHeight, point.x, point.y);
    const canvas = document.createElement("canvas");
    canvas.width = patch.width; canvas.height = patch.height;
    const context = canvas.getContext("2d");
    context.drawImage(image, patch.x, patch.y, patch.width, patch.height, 0, 0, patch.width, patch.height);
    await setGradeColor(patchColor(context.getImageData(0, 0, patch.width, patch.height).data));
  } catch (error) {
    if (request === gradeColorRequest && style === selection && generation === seen) {
      $("grade-error").textContent = "Could not sample this point. Pick a color with the hex field.";
    }
  } finally {
    if (request === gradeColorRequest) gradeColorPending = false;
  }
}
function changed(options) {
  mainPaused = true;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-label", "Play preview");
  renderControls(options);
  renderSteps();
  updateMain();
}
// A warning (take unreadable, sample still, preview error) stays visible on every screen.
function setPreviewNote(text, alert) {
  $("preview-note").textContent = text;
  $("preview-note").dataset.alert = alert ? "true" : "false";
}
async function updateMain() {
  try {
    await renderMain();
    setPreviewNote(previewNote(preview), Boolean(preview.message) || preview.source !== "take");
  } catch (error) {
    if (error.name !== "AbortError") setPreviewNote(error.message, true);
  }
}
function goToStep(step, updateHash = true) {
  activeStep = stepFromHash(`#${step}`, schema);
  document.body.dataset.screen = activeStep;
  if (updateHash && location.hash !== `#${activeStep}`) history.replaceState(null, "", `#${activeStep}`);
  showStep(activeStep, document.querySelectorAll("[data-step]"), document.querySelectorAll(".step-panel"));
  const building = activeStep !== "looks", optional = optionalScreens(), index = optional.findIndex(({ id }) => id === activeStep);
  $("main-frame").parentElement.setAttribute("data-editing", String(building));
  $("steps").hidden = !building;
  $("step-header").hidden = !building;
  $("step-navigation").hidden = !building;
  $("savebar").hidden = $("picker").hidden;
  $("look-selected").hidden = building;
  $("look-scope").hidden = building;
  $("customize").hidden = building;
  $("reset").hidden = !building;
  $("change-count").hidden = !building;
  $("screen-summary").hidden = index < 0;
  $("next").hidden = activeStep === optional.at(-1)?.id;
  previewControls(mainDrawing(activeStep));
  $("step-count").textContent = index >= 0 ? `Optional · ${index + 1} of ${optional.length}`
    : activeStep === "more" ? `Optional · ${optional.length} choices`
    : `Step ${coreTabs.indexOf(activeStep) + 1} of 4`;
  if (style && !$("picker").hidden) {
    renderControls();
    renderSteps();
    updateMain();
  }
  if (updateHash) window.scrollTo(0, 0);
}
async function pullColors() {
  const raw = $("site-url").value.trim();
  if (!raw) {
    $("site-status").textContent = "Enter your website address first.";
    $("site-url").focus();
    return;
  }
  $("site-pull").disabled = true;
  $("site-pull").textContent = "Getting colors…";
  $("site-status").textContent = `Reading ${raw}…`;
  const controller = new AbortController();
  siteController = controller;
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const result = await api(`/api/site-colors?url=${encodeURIComponent(raw)}`, null, controller.signal);
    if (controller.signal.aborted) return;
    customColors = { ...result.suggestion, accent: result.suggestion.accent || colorsOf(style).accent };
    const editedRoles = roles.map(([id]) => id).filter((id) => id !== "accent" || result.suggestion.accent);
    style = withColors(style, customColors, editedRoles);
    pulledSite = result.site;
    changed({ replaceColors: true });
    $("site-status").textContent = `Filled in from ${result.site}.${result.suggestion.accent ? "" : ` ${result.site} is mostly black and white, so we kept your highlight color.`}`;
    $("found-colors").replaceChildren();
    for (const color of result.colors) {
      if (!normalizeHex(color.hex)) continue;
      const button = document.createElement("button");
      button.className = "found-chip";
      button.type = "button";
      button.setAttribute("aria-label", `Use ${color.hex} for the selected spot`);
      button.append(colorDot(color.hex), color.name || color.hex);
      button.addEventListener("click", () => setColor(activeRole, color.hex));
      $("found-colors").append(button);
    }
  } catch (error) {
    if (controller.signal.reason === "style changed") return;
    $("site-status").textContent = error.name === "AbortError"
      ? "That site took too long. Try again, or type hex codes above."
      : `${error.message} You can type hex codes above instead.`;
  } finally {
    clearTimeout(timer);
    if (siteController === controller) siteController = null;
    $("site-pull").disabled = false;
    $("site-pull").textContent = "Get colors";
  }
}
function wire() {
  const savebarResize = new ResizeObserver(() => {
    reserveSavebar($("savebar"), document.documentElement);
  });
  savebarResize.observe($("savebar"));
  reserveSavebar($("savebar"), document.documentElement);
  $("main-frame").addEventListener("click", (event) => {
    if (activeStep === "grade" && (style.grade?.look === "eyedropper" || choosingGradeColor)) return sampleGrade(event);
    const expanded = $("main-frame").classList.toggle("phone-expanded");
    $("main-frame").setAttribute("aria-expanded", String(expanded));
    $("main-frame").setAttribute("aria-label", expanded ? "Shrink preview" : "Enlarge preview");
  });
  $("grade-picker").addEventListener("input", () => setGradeColor($("grade-picker").value));
  $("grade-color").addEventListener("change", () => setGradeColor($("grade-color").value));
  $("grade-color").addEventListener("input", () => {
    gradeColorRequest += 1;
    gradeColorPending = false;
    $("grade-error").textContent = "";
    $("grade-color").removeAttribute("aria-invalid");
    const value = normalizeHex($("grade-color").value);
    if (value) setGradeColor(value);
  });
  wireFolds();
  for (const { id, optional } of screens) if (optional) $(`${id}-all-choices`).addEventListener("click", () => goToStep("more"));
  $("back-to-looks").addEventListener("click", () => goToStep("looks"));
  $("customize").addEventListener("click", () => goToStep("font"));
  $("caption-picker").addEventListener("input", () => {
    $("caption-color").value = $("caption-picker").value;
    setCaptionColor();
  });
  function setCaptionColor() {
    const value = normalizeHex($("caption-color").value);
    $("caption-error").textContent = value ? "" : "Use a hex code like #2F3BFF.";
    if (!value) {
      $("caption-color").setAttribute("aria-invalid", "true");
      $("reset").disabled = false;
      return;
    }
    $("caption-color").value = value;
    $("caption-color").removeAttribute("aria-invalid");
    customColors = { ...colorsOf(style), caption: value };
    style = withColors(style, customColors, ["caption"]);
    changed();
  }
  $("caption-color").addEventListener("change", setCaptionColor);
  $("caption-color").addEventListener("input", () => {
    if (/^#?[0-9a-f]{6}$/i.test($("caption-color").value)) setCaptionColor();
  });
  $("background-picker").addEventListener("input", () => {
    $("background-color").value = $("background-picker").value;
    setBackgroundColor();
  });
  $("background-color").addEventListener("change", setBackgroundColor);
  $("background-color").addEventListener("input", () => {
    $("background-error").textContent = "";
    $("background-color").removeAttribute("aria-invalid");
    $("reset").disabled = false;
    if (/^#?[0-9a-f]{6}$/i.test($("background-color").value)) setBackgroundColor();
  });
  $("background-opacity").addEventListener("click", (event) => {
    if (!event.target.dataset.opacity) return;
    selectStyle(withBackgroundOpacity(style, event.target.dataset.opacity));
  });
  $("background-corners").addEventListener("click", (event) => {
    if (!event.target.dataset.corners) return;
    selectStyle(withBackgroundCorners(style, event.target.dataset.corners));
  });
  $("avoid-face").addEventListener("change", () => {
    selectStyle(withFaceAvoidance(style, $("avoid-face").checked));
  });
  $("shadow-toggle").addEventListener("click", (event) => {
    const toggle = event.target.dataset.shadowToggle;
    if (!toggle) return;
    if (toggle === "off") {
      if (style.captions.shadow_opacity > 0) {
        lastShadow = Object.fromEntries(["shadow_opacity", "shadow_blur_px", "shadow_x_px", "shadow_y_px"]
          .map((key) => [key, style.captions[key]]));
      }
      style = withShadow(style, "off");
    } else if (shadowMode(style) === "off") {
      if (lastShadow) Object.assign(style.captions, lastShadow);
      else style = withShadow(style, "soft");
    }
    changed();
  });
  $("shadow-strength").addEventListener("click", (event) => {
    if (!event.target.dataset.shadow) return;
    style = withShadow(style, event.target.dataset.shadow);
    changed();
  });
  $("sizes").addEventListener("click", (event) => {
    if (!event.target.dataset.size) return;
    style.captions.size_px = sizes[event.target.dataset.size];
    changed();
  });
  $("guides").addEventListener("click", () => {
    const shown = $("main-frame").classList.toggle("show-guides");
    $("guides").setAttribute("aria-pressed", String(shown));
  });
  $("play").addEventListener("click", () => {
    mainPaused = !mainPaused;
    $("play").textContent = mainPaused ? "Play" : "Pause";
    $("play").setAttribute("aria-label", mainPaused ? "Play preview" : "Pause preview");
    for (const frame of mainStage?.frames ?? []) runPlayer(frame, mainStage, mainPaused ? "pause" : "play");
    playDrawing($("main-frame"));
  });
  $("italics").addEventListener("click", (event) => {
    if (!event.target.dataset.italic) return;
    style.captions.italic_stress = event.target.dataset.italic === "true";
    changed();
  });
  for (const motion of motions) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.motion = motion.name;
    button.textContent = motion.name;
    button.addEventListener("click", () => {
      style = withMotion(style, motion);
      changed();
    });
    $("motion").append(button);
  }
  $("placements").addEventListener("click", (event) => {
    if (!event.target.dataset.place) return;
    style = withPlacement(style, event.target.dataset.place);
    changed();
  });
  $("music").addEventListener("input", () => {
    const level = Number($("music").value);
    if (state.music_answer === "none") musicChoice = level;
    style = withMusicLevel(style, level);
    renderControls();
  });
  $("grade-slider").addEventListener("click", (event) => {
    if (!event.target.dataset.gradeSlider) return;
    style.grade = { ...style.grade, slider: event.target.dataset.gradeSlider === "true" };
    changed();
  });
  $("grade-strength").addEventListener("click", (event) => {
    if (!event.target.dataset.gradeStrength) return;
    style = withGrade(style, style.grade?.look ?? "none", Number(event.target.dataset.gradeStrength));
    changed();
  });
  for (const button of document.querySelectorAll("[data-step]")) {
    button.addEventListener("click", () => goToStep(button.dataset.step));
    button.addEventListener("keydown", (event) => {
      const current = coreTabs.indexOf(screens.some(({ id, optional }) => id === activeStep && optional) ? "more" : activeStep);
      let step;
      if (["ArrowRight", "ArrowLeft"].includes(event.key)) step = coreTabs[Math.max(0, Math.min(coreTabs.length - 1, current + (event.key === "ArrowRight" ? 1 : -1)))];
      if (event.key === "Home") step = "font";
      if (event.key === "End") step = "more";
      if (!step) return;
      event.preventDefault(); goToStep(step); $(`step-${step}`).focus();
    });
  }
  window.addEventListener("hashchange", routeHash);
  $("back").addEventListener("click", () => goToStep(adjacentStep(activeStep, -1, schema)));
  $("font-same").addEventListener("click", () => {
    fontRole = "graphics";
    selectStyle(withFont(style, captionsFont(style)));
  });
  for (const [font, label] of [[fontPair.captions, "captions"], [fontPair.graphics, "titles and cards"]]) {
    const sample = document.createElement("span"), note = document.createElement("small");
    sample.textContent = font;
    sample.style.fontFamily = fontFamily(font, state.fonts);
    note.textContent = label;
    sample.append(note);
    $("font-pair").append(sample);
  }
  $("font-pair").addEventListener("click", () => {
    fontRole = "graphics";
    selectStyle(withFontPair(style));
  });
  $("next").addEventListener("click", () => goToStep(adjacentStep(activeStep, 1, schema)));
  for (const [id, field, attribute] of [["broll-overlay-box", "overlay_box", "overlayBox"], ["broll-stock-clips", "stock_clips", "stockClips"], ["broll-tall-framing", "tall_framing", "tallFraming"]]) {
    $(id).addEventListener("click", (event) => {
      const value = event.target.dataset[attribute];
      if (value) selectStyle(withBrollField(style, field, value));
    });
  }
  $("sound-fallback").addEventListener("change", () => selectStyle(withSoundFallback(style, $("sound-fallback").checked)));
  for (const name of ["typing", "message-ping", "camera-shutter"]) {
    $("sound-extra-" + name).addEventListener("change", () =>
      selectStyle(withSoundExtra(style, name, $("sound-extra-" + name).checked)));
  }
  $("sound-effects").addEventListener("click", (event) => {
    if (event.target.dataset.effects) {
      if (soundKitChoice(style) === "elevenlabs") soundKit = style.music?.sound_effects || "soft";
      selectStyle(withSoundKit(style, event.target.dataset.effects, soundKit));
    }
  });
  $("beat-snap").addEventListener("change", () => selectStyle(withBeatSnap(style, $("beat-snap").checked)));
  $("bleep-swears").addEventListener("change", () => selectStyle(withBleepSwears(style, $("bleep-swears").checked)));
  $("voice-preset").addEventListener("click", (event) => {
    if (event.target.dataset.voicePreset) selectStyle(withVoicePreset(style, event.target.dataset.voicePreset));
  });
  for (const field of ["repair_rumble", "repair_room", "repair_harshness"]) {
    const input = $(field.replaceAll("_", "-"));
    input.addEventListener("change", () => selectStyle(withVoiceRepair(style, field, input.checked)));
  }
  $("repair-proximity").addEventListener("click", (event) => {
    if (event.target.dataset.repairProximity) selectStyle(withVoiceRepair(style, "repair_proximity", event.target.dataset.repairProximity));
  });
  $("site-pull").addEventListener("click", pullColors);
  $("site-url").addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); pullColors(); }
  });
  $("reset").addEventListener("click", () => {
    clearGradeColor();
    focusedMoment = null;
    siteController?.abort("style changed");
    style = copy(baseStyle);
    fontRole = "captions";
    lastShadow = null;
    customColors = colorsOf(style);
    changed({ replaceColors: true });
  });
  function validateSave() {
    siteController?.abort("style changed");
    $("save-error").textContent = "";
    if (choosingGradeColor || gradeColorPending || (style.grade?.look === "eyedropper"
      && ($("grade-color").getAttribute("aria-invalid") === "true" || !normalizeHex($("grade-color").value)))) {
      $("save-error").textContent = "Check your picked color.";
      openField("grade", "grade-color");
      return false;
    }
    if (!normalizeHex($("caption-color").value)) {
      $("save-error").textContent = "Check your caption color.";
      openField("captions", "caption-color");
      return false;
    }
    if (!style.captions.camera_3d && style.captions.background_mode !== "none" && !normalizeHex($("background-color").value)) {
      $("save-error").textContent = "Check your background color.";
      openField("captions", "background-color");
      return false;
    }
    if (roles.some(([id]) => !normalizeHex($(`color-${id}`).value))) {
      $("save-error").textContent = "Check your hex codes in Colors.";
      openField("colors", `color-${roles.find(([id]) => !normalizeHex($(`color-${id}`).value))[0]}`);
      return false;
    }
    const errors = check(style, schema);
    if (errors.length) { $("save-error").textContent = errors[0]; return false; }
    return true;
  }
  let replaceStyle = null;
  function clearSaveQuestion() {
    replaceStyle = null;
    $("save-sheet-status").textContent = "";
    $("save-sheet-error").textContent = "";
    $("confirm-save").textContent = "Save style";
  }
  function askToUpdate(entry) {
    replaceStyle = entry;
    $("save-sheet-error").textContent = "";
    $("save-sheet-status").textContent = `You already have a style called “${entry.name}”. Update it, or choose another name.`;
    const label = entry.name.length > 32 ? `${entry.name.slice(0, 31)}…` : entry.name;
    $("confirm-save").textContent = `Update ${label}`;
  }
  function openSaveSheet() {
    if (!style.hook.answered) { goToStep("top-text"); return; }
    if (!validateSave()) return;
    clearSaveQuestion();
    $("save-sheet").showModal();
    $("style-name").focus();
    $("style-name").select();
  }
  $("save").addEventListener("click", openSaveSheet);
  $("cancel-save").addEventListener("click", () => { clearSaveQuestion(); $("save-sheet").close(); });
  $("style-name").addEventListener("input", clearSaveQuestion);
  $("style-name").addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); $("confirm-save").click(); }
  });
  $("confirm-save").addEventListener("click", async () => {
    if ($("confirm-save").disabled || !validateSave()) return;
    const name = $("style-name").value.trim() || "My style";
    const existing = state.library.find((entry) => entry.name.trim().toLowerCase() === name.toLowerCase());
    if (!replaceStyle && existing) { askToUpdate(existing); return; }
    $("save-sheet-status").textContent = "";
    $("save-sheet-error").textContent = "";
    $("confirm-save").disabled = true;
    try {
      const saved = await api("/api/save", { style, name, ...(replaceStyle ? { replace: replaceStyle.id } : {}) });
      style.name = saved.name;
      baseStyle = copy(style);
      baseKey = `library:${saved.id}`;
      $("style-name").value = saved.name;
      $("saved-name").textContent = saved.name;
      $("saved-copy").textContent = "Saved in your style library.";
      const entry = { id: saved.id, name: saved.name, style: copy(style) };
      const index = state.library.findIndex((item) => item.id === saved.id);
      if (index >= 0) state.library[index] = { ...state.library[index], ...entry };
      else state.library.unshift(entry);
      clearStages(true);
      $("picker").hidden = true;
      $("savebar").hidden = true;
      $("saved").hidden = false;
      $("save-sheet").close();
      window.scrollTo(0, 0);
    } catch (error) {
      if (error.code === "style_name_taken" && error.id) {
        askToUpdate({ id: error.id, name });
        $("save-sheet-status").textContent = error.message;
      } else $("save-sheet-error").textContent = error.message;
    } finally {
      $("confirm-save").disabled = false;
    }
  });
  $("update").addEventListener("click", () => {
    $("saved").hidden = true;
    $("picker").hidden = false;
    renderControls();
    goToStep(activeStep);
  });
}
window.addEventListener("message", (event) => {
  if (event.origin !== location.origin || event.data?.type !== "gallery-layer-404") return;
  for (const detail of stages) {
    if (detail.frames.some((frame) => frame.contentWindow === event.source)) {
      detail.retry404?.(event.data.url, true).catch(console.error);
      return;
    }
  }
});
async function pollPreview() {
  if (previewPollPending) return;
  previewPollPending = true;
  try {
    const next = await api("/api/preview");
    if (next.generation < generation) return;
    if (next.generation !== generation) {
      generation = next.generation;
      preview = next;
      if (!$("picker").hidden) changed();
    } else {
      const screenChanged = (preview.screen ?? null) !== (next.screen ?? null)
        || Boolean(preview.screen_checked) !== Boolean(next.screen_checked);
      preview.screen = next.screen ?? null;
      preview.screen_checked = Boolean(next.screen_checked);
      if (preview.cutout !== (next.cutout ?? null)) {
        preview.cutout = next.cutout ?? null;
        if (!$("picker").hidden) redrawCutouts();
      }
      if (screenChanged && activeStep === "moments" && !$("picker").hidden) renderSteps();
    }
  } catch { /* The next poll can recover after a short connection drop. */ }
  finally { previewPollPending = false; }
}
function showHomeLinks(home) {
  // Opened from Yap Studio home: offer the way back, on this computer or the phone link.
  const local = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
  const url = home ? (local ? home.url : home.phone_url ?? home.url) : null;
  for (const id of ["home-link", "saved-home"]) {
    const link = $(id);
    if (!link) continue;
    link.hidden = !url;
    if (url) link.href = url;
    const via = !local && home?.phone_url ? home.phone_via : null;
    const label = via === "tailscale" ? "Encrypted (Tailscale)" : via === "wifi" ? "Not encrypted (same Wi-Fi)" : "";
    link.textContent = "Back to Yap Studio" + (label ? ` · ${label}` : "");
  }
}
async function boot() {
  try {
    state = await api("/api/state");
    schema = state.schema;
    musicChoice = state.music_answer === "none"
      ? (state.from_library ? (state.style.music?.level ?? 0) : 0) : undefined;
    style = withPreset(state.style, state.style, musicChoice);
    openingLook = !["none", "text_hook"].includes(style.hook.style) ? style.hook.style : "quiet_serif";
    fontRole = "captions";
    soundKit = soundKitChoice(style) === "elevenlabs" ? style.music?.sound_effects || "soft" : "soft";
    baseStyle = copy(style);
    const source = startList(state).find((item) => item.fromLibrary === state.from_library
      && item.name === style.name);
    baseKey = source ? startKey(source) : null;
    customColors = colorsOf(style);
    preview = state.preview;
    generation = preview.generation;
    $("style-name").value = startingName(style, state.from_library);
    showHomeLinks(state.home);
    buildColorFields();
    buildFineFields();
    wire();
    renderControls();
    routeHash();
    animate();
    setInterval(pollPreview, 1500);
  } catch (error) {
    setPreviewNote(error.message, true);
  }
}
boot();
