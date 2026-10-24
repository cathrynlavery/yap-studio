"""Template contracts, pure box plans, and opt-in real renderer coverage."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from test_render import _composition
from test_render import fake_renderer as fake_renderer

from yap_studio import ffmpeg, paths
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.errors import ProcessingError, ValidationError, YapError
from yap_studio.render import fonts, hyperframes
from yap_studio.stages import captions as caption_stage
from yap_studio.stages.captions import (
    LayerAssets,
    LayerPlan,
    caption_layer_plan,
    caption_shrink,
    caption_spelling,
    render_layer_plan,
    title_layer_plan,
)
from yap_studio.style import SCHEMA_PATH, ResolvedStyle, resolve, resolve_style_data

_TEMPLATES = Path(__file__).parents[1] / "src" / "yap_studio" / "templates"
_HIGHLIGHTS = (
    "moving_block", "underline", "color", "pill", "gradient", "editorial",
    "slam", "weight_shift", "clip_wipe", "none",
)
_REGISTRY_LOOKS = ("slam", "weight_shift", "clip_wipe")
_SHRINK_PREFIX = "yap-studio caption-shrink "


@pytest.mark.render
def test_real_giant_has_solid_block_readable_glyphs_and_identical_renders(tmp_path: Path) -> None:
    style = _style(highlight="giant", background_color="#27A1E5", shadow_opacity=0,
                   italic_stress=False)
    phrase = build_phrases([CaptionWord("w0", "H", 0.2, 0.4),
                            CaptionWord("w1", "H", 0.4, 0.7)], style.data["captions"])
    plan = caption_layer_plan(phrase, style, frames=30)
    with paths.operation():
        first = render_layer_plan(plan, tmp_path / "first", tmp_path / "first.mov").path
        second = render_layer_plan(plan, tmp_path / "second", tmp_path / "second.mov").path
    a, b = (_rgba_frames(out, plan) for out in (first, second))
    np.testing.assert_array_equal(a, b)
    frame = a[12]
    rows, columns = np.nonzero(frame[:, :, 3] > 250)
    center = frame[(rows.min() + rows.max()) // 2, (columns.min() + columns.max()) // 2]
    block = np.array([39, 161, 229])
    assert center[3] == 255
    np.testing.assert_allclose(center[:3], block, atol=3)
    ink = np.array([int(plan.variables["text_color"][i:i + 2], 16)
                    for i in (1, 3, 5)])
    glyphs = (np.abs(frame[:, :, :3].astype(int) - ink).max(axis=2) <= 3)
    assert np.count_nonzero(glyphs & (frame[:, :, 3] == 255)) > 100


def test_preview_selects_and_renders_automatic_phrase_stress() -> None:
    app = (
        Path(__file__).parents[1] / "docs" / "design" / "style-page" / "app.js"
    ).read_text()
    assert "buildPhrases(WORDS, captions)" in app
    assert "stress_ids" not in app
    assert "WORDS[id].id === phrase.stress" in app


class _Declarations(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.variables: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "html":
            self.variables = json.loads(dict(attrs)["data-composition-variables"] or "[]")


def _validate_variables(source: str, variables: Mapping[str, object]) -> None:
    """Match the pinned renderer's strict enum declaration and value checks."""
    parser = _Declarations()
    parser.feed(source)
    for declaration in parser.variables:
        if declaration.get("type") != "enum":
            continue
        options = declaration.get("options")
        valid = isinstance(options, list) and all(
            isinstance(option, dict)
            and isinstance(option.get("value"), str)
            and isinstance(option.get("label"), str)
            for option in options
        )
        value = variables.get(declaration["id"], declaration.get("default"))
        if not valid or value not in [option["value"] for option in options]:
            raise ProcessingError(
                "hyperframes render", 1, "Variable validation failed", code="hyperframes_exit_1"
            )


def _style(
    *, highlight: str = "moving_block", look: str = "paper_title", **caption_tokens: object
) -> ResolvedStyle:
    data = resolve_style_data(
        {
            "schema_version": 1,
            "name": "Render sample",
            "captions": {
                "size_px": 48,
                "max_characters_per_phrase": 40,
                "highlight_style": highlight,
                "highlight_color": "#D8FF2F",
                "highlight_text_color": "#111111",
                **caption_tokens,
            },
            "hook": {"style": look, "duration_s": 2, "size_px": 48},
        }
    )
    return ResolvedStyle(data["name"], "test", data)


def _phrases(style: ResolvedStyle) -> list:
    phrases = build_phrases(
        [
            CaptionWord("w0", "First", 0.2, 0.4),
            CaptionWord("w1", "phrase.", 0.4, 0.7),
            CaptionWord("w2", "Second", 1.0, 1.2),
            CaptionWord("w3", "phrase.", 1.2, 1.5),
            CaptionWord("w4", "Third", 1.8, 2.0),
            CaptionWord("w5", "phrase.", 2.0, 2.3),
        ],
        style.data["captions"],
    )
    return [replace(phrase, end=end) for phrase, end in zip(phrases, (0.7, 1.5, 2.3), strict=True)]


def test_layer_plan_balances_wrapping_words_only() -> None:
    style = _style(highlight="none", font="Instrument Serif", size_px=83,
                   max_characters_per_phrase=48)
    source = [CaptionWord(str(i), word, i * 0.2, i * 0.2 + 0.1)
              for i, word in enumerate("These thoughtful little moments matter".split())]
    phrase = build_phrases(source, style.data["captions"])[0]
    plan = caption_layer_plan([phrase], style, frames=90)
    slot = json.loads(plan.variables["phrases"])[0]
    assert slot["breaks"]
    assert len(slot["words"]) - slot["breaks"][-1] >= 2
    short = build_phrases(source[:2], style.data["captions"])[0]
    assert "breaks" not in json.loads(caption_layer_plan(
        [short], style, frames=90).variables["phrases"])[0]
    for look in ("slam", "weight_shift", "clip_wipe"):
        registry = _style(highlight=look, font="Instrument Serif", size_px=83)
        assert "breaks" not in json.loads(caption_layer_plan(
            [phrase], registry, frames=90).variables["phrases"])[0]


def test_caption_spelling_reports_near_brand_words() -> None:
    style = _style()
    source = [CaptionWord("a", "Try", 0, 0.1),
              CaptionWord("b", "Sparrowlite", 0.2, 0.3)]
    phrases = build_phrases(source, style.data["captions"], text_rules=True,
                            vocabulary=["SparrowLight"])
    assert caption_spelling(phrases, ["SparrowLight"]) == {
        "vocabulary": 1,
        "near": [{"text": "Sparrowlite", "entry": "SparrowLight",
                  "start_s": 0.2, "phrase": 0}],
    }


class _ScriptSources(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and "src" in attributes:
            self.sources.append(attributes["src"])


def _template_violations(file: Path) -> list[str]:
    """Scan one file, allowing local GSAP only in an adopted template's HTML."""
    forbidden = (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "http://",
        "https://",
        "gsap",
        "requestAnimationFrame",
    )
    source = file.read_text(encoding="utf-8")
    adopted = False
    if file.name == "template.html":
        try:
            metadata = json.loads((file.parent / "template.json").read_text(encoding="utf-8"))
            derived = metadata.get("derived_from")
            adopted = isinstance(derived, list) and bool(derived)
        except (OSError, ValueError, AttributeError):
            pass
    violations = [
        token for token in forbidden
        if token.casefold() in source.casefold() and not (token == "gsap" and adopted)
    ]
    if adopted:
        parser = _ScriptSources()
        parser.feed(source)
        if any(src != "gsap.min.js" for src in parser.sources):
            violations.append("script src must be gsap.min.js")
    return violations


def test_templates_have_safe_dom_and_local_assets() -> None:
    files = [file for file in _TEMPLATES.rglob("*") if file.is_file()]
    assert files
    for file in files:
        assert not _template_violations(file), file


@pytest.mark.parametrize(
    ("filename", "adopted", "source", "violation"),
    [
        ("template.html", True, "<script>node.innerHTML = '';</script>", "innerHTML"),
        ("template.html", True,
         '<script src="https://cdn.invalid/gsap.min.js"></script>',
         "script src must be gsap.min.js"),
        ("template.html", False, "<script>gsap.timeline();</script>", "gsap"),
        ("template.json", True, '{"note": "gsap"}', "gsap"),
        ("preview.json", True, '{"note": "gsap"}', "gsap"),
        ("other.html", True, "gsap", "gsap"),
        ("template.html", True, '<script src="local.js"></script>',
         "script src must be gsap.min.js"),
    ],
)
def test_adopted_template_exception_is_narrow(
    tmp_path: Path, filename: str, adopted: bool, source: str, violation: str,
) -> None:
    metadata = {"derived_from": [{"item": "sample"}] if adopted else []}
    (tmp_path / "template.json").write_text(json.dumps(metadata))
    file = tmp_path / filename
    file.write_text(source)
    assert violation in _template_violations(file)


def test_adopted_template_allows_local_gsap(tmp_path: Path) -> None:
    (tmp_path / "template.json").write_text('{"derived_from": [{"item": "sample"}]}')
    file = tmp_path / "template.html"
    file.write_text('<script src="gsap.min.js"></script><script>gsap.timeline();</script>')
    assert _template_violations(file) == []


@pytest.mark.parametrize("template", ["captions", "title"])
def test_template_metadata_matches_declared_variables(template: str) -> None:
    metadata = json.loads((_TEMPLATES / template / "template.json").read_text())
    parser = _Declarations()
    parser.feed((_TEMPLATES / template / "template.html").read_text())
    assert parser.variables == metadata["variables"]
    assert metadata["id"] == template
    assert metadata["version"] == (4 if template == "captions" else 2)
    style = _style()
    plan = (
        caption_layer_plan(_phrases(style), style, frames=90)
        if template == "captions"
        else title_layer_plan("Opening", "words", style)
    )
    declared = {variable["id"] for variable in parser.variables}
    optional = {"band_top_px", "band_height_px"} if template == "captions" else set()
    assert set(plan.variables) == declared - optional
    if template == "captions":
        moved = caption_layer_plan(_phrases(style), style, frames=90, offsets=[0, -300, 0])
        assert set(moved.variables) == declared
        _validate_variables((_TEMPLATES / template / "template.html").read_text(), moved.variables)
    _validate_variables((_TEMPLATES / template / "template.html").read_text(), plan.variables)


def test_caption_highlight_options_equal_schema_enum() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    parser = _Declarations()
    parser.feed((_TEMPLATES / "captions" / "template.html").read_text())
    highlight = next(
        variable for variable in parser.variables if variable["id"] == "highlight_style"
    )
    values = [option["value"] for option in highlight["options"]]
    assert values == schema["properties"]["captions"]["properties"]["highlight_style"]["enum"]
    assert values == [
        'moving_block',
        'underline',
        'color',
        'pill',
        'gradient',
        'editorial',
        'slam',
        'weight_shift',
        'clip_wipe',
        'blend_difference',
        'camera_follow',
        'editorial_emphasis',
        'emoji_pop',
        'glitch_rgb',
        'gradient_fill',
        'highlight',
        'matrix_decode',
        'neon_accent',
        'neon_glow',
        'parallax_layers',
        'particle_burst',
        'pill_karaoke',
        'texture',
        'giant',
        'none',
    ]
    assert [option["label"] for option in highlight["options"]] == [
        'Sliding block',
        'Underline',
        'Colored word',
        'Pill karaoke',
        'Gradient sweep',
        'Editorial emphasis',
        'Kinetic slam',
        'Weight shift',
        'Clip wipe',
        'Difference blend',
        'Camera follow',
        'Large key word',
        'Emoji pop',
        'RGB glitch',
        'Gradient fill',
        'Word background',
        'Matrix decode',
        'Neon accent',
        'Neon glow',
        'Parallax layers',
        'Particle burst',
        'Rounded karaoke',
        'Text texture',
        'Giant captions',
        'No highlight',
    ]


@pytest.mark.parametrize(
    "highlight", ["pill", "gradient", "editorial", "slam", "weight_shift", "clip_wipe"]
)
@pytest.mark.parametrize("width_scale", [0.6, 0.8, 1])
def test_new_caption_highlight_plans_validate(highlight: str, width_scale: float) -> None:
    style = _style(highlight=highlight, width_scale=width_scale)
    plan = caption_layer_plan(_phrases(style), style, frames=90)
    assert plan.variables["highlight_style"] == highlight
    assert plan.variables["width_scale"] == width_scale
    assert len(json.loads(plan.variables["phrases"])) == 3
    _validate_variables((_TEMPLATES / "captions" / "template.html").read_text(), plan.variables)


def _caption_dom(
    plan: LayerPlan, rects: list[dict], *, seek_times: list[float] | None = None,
    source: str | None = None, glyph_width: float = 0.6,
    opacity_frames: dict[str, dict[int, float]] | None = None,
    clip_rects: list[dict] | None = None,
    line_width: float = 300,
) -> dict:
    """Execute the template with recorded layout boxes, without launching a renderer."""
    node = shutil.which("node")
    assert node is not None, "Template animation contracts need Node"
    if source is None:
        source = (_TEMPLATES / "captions" / "template.html").read_text()
    script = source.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = [];
let animationCount = 0;
let fontsReady = false;
class Element {
  constructor(tag) {
    this.tag = tag;
    this.className = '';
    this.textContent = '';
    this.children = [];
    this.animations = [];
    this.style = { setProperty(name, value) { this[name] = value; } };
    nodes.push(this);
  }
  append(...children) {
    for (const child of children) {
      Object.defineProperty(child, 'parentElement', { value: this, configurable: true });
    }
    this.children.push(...children);
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  replaceWith(child) {
    const parent = this.parentElement;
    parent.children.splice(parent.children.indexOf(this), 1, child);
    Object.defineProperty(child, 'parentElement', { value: parent, configurable: true });
  }
  get clientWidth() {
    if (!fontsReady) throw new Error('Measured before fonts were ready');
    return parseFloat(this.parentElement.style.maxWidth || '940px');
  }
  get scrollWidth() {
    this.widthChecks = (this.widthChecks || 0) + 1;
    const size = parseFloat(this.parentElement.style.fontSize) || input.variables.size_px;
    const pieces = (node) => node.className === 'w' ? node.textContent.split('-') :
      (node.children || []).flatMap(pieces);
    return Math.max(this.clientWidth, ...pieces(this).map((text) =>
      text.length * size * input.glyph_width + text.length * input.variables.letter_spacing_px));
  }
  get offsetLeft() { return this.getBoundingClientRect()?.left - input.line.left; }
  get offsetTop() { return this.getBoundingClientRect()?.top - input.line.top; }
  get offsetWidth() {
    if (this.className === 'w') return this.getBoundingClientRect()?.width;
    return Math.min(parseFloat(this.style.maxWidth || '940'), input.line_width || 300);
  }
  get offsetHeight() {
    if (this.className === 'w') return this.getBoundingClientRect()?.height;
    const size = parseFloat(this.parentElement?.style.fontSize) || input.variables.size_px;
    return size * 1.12 * Math.max(1, this.children.filter((n) => n.className === 'cap-row').length);
  }
  get scrollHeight() { return this.offsetHeight; }
  animate(keys, options) {
    let previous = 0;
    for (const { offset } of keys) {
      if (!Number.isFinite(offset) || offset < 0 || offset > 1 || offset < previous) {
        throw new TypeError('Keyframe offsets must be finite, ordered and within 0..1');
      }
      previous = offset;
    }
    const initialTime = [null, 123.456, -7.25][animationCount++ % 3];
    let time = initialTime;
    const animation = { keys, options, initialTime, timeWrites: [], pause() {},
      get currentTime() { return time; },
      set currentTime(value) { time = value; this.timeWrites.push(value); },
    };
    this.animations.push(animation);
    return animation;
  }
  getAnimations({ subtree = false } = {}) {
    return [...this.animations, ...(subtree ? this.children.flatMap((child) =>
      child.getAnimations ? child.getAnimations({ subtree: true }) : []) : [])];
  }
  getContext() {
    if (!fontsReady) throw new Error('Measured before fonts were ready');
    return {
      font: '',
      measureText(text) {
        const size = Number(this.font.match(/([\d.]+)px/)[1]);
        const weight = Number(this.font.match(/(?:italic )?(\d+) /)[1]);
        return { width: text.length * size * 0.6 * (1 + weight / 2000) };
      },
    };
  }
  getBoundingClientRect() {
    if (!fontsReady) throw new Error('Measured before fonts were ready');
    if (this.className === 'cap-rise-clip') {
      const index = nodes.filter((node) => node.className === 'cap-rise-clip').indexOf(this);
      return input.clip_rects[index] || input.rects[index] || input.line;
    }
    const index = nodes.filter((node) => node.className === 'w').indexOf(this);
    if (index < 0) return input.line;
    let cap = this.parentElement;
    while (!cap.className.split(' ').includes('cap')) cap = cap.parentElement;
    this.rectReads = this.rectReads || [];
    this.rectReads.push(cap.getAnimations({ subtree: true }).map((animation) =>
      animation.currentTime));
    const base = input.rects[index];
    const rect = base?.frames?.[currentFrame(this)] ?? base;
    if (!rect) return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 };
    const width = rect.width ?? rect.right - rect.left;
    const height = rect.height ?? rect.bottom - rect.top;
    return { left: rect.left, top: rect.top, width, height,
      right: rect.left + width, bottom: rect.top + height };
  }
}
const root = new Element('div');
root.dataset = { duration: input.duration, height: input.height, width: input.width };
root.warnings = [];
console.warn = (message) => root.warnings.push(message);
global.window = { __hyperframes: { getVariables: () => input.variables } };
global.document = {
  querySelector: () => root,
  createElement: (tag) => new Element(tag),
  createTextNode: (text) => ({ textContent: text }),
  fonts: Object.assign(input.faces, {
    ready: Promise.resolve().then(() => { fontsReady = true; }),
  }),
};
global.getComputedStyle = (node) => {
  const hidden = node.className.split(' ').includes('cap-line') ||
    (node.className === 'w' && node.parentElement?.parentElement?.className.includes('cap-slam'));
  const opacity = hidden ? '0' : '1';
  const display = ['cap-rise-clip', 'cap-rise-mover'].includes(node.className) ?
    'contents' : 'block';
  let size = input.variables.size_px;
  for (let el = node; el; el = el.parentElement) {
    if (el.style?.fontSize) { size = parseFloat(el.style.fontSize); break; }
  }
  const style = { opacity, display, lineHeight: `${size * 1.12}px`, ...animatedStyle(node) };
  if (node.className === 'cap-words') {
    const line = node.parentElement;
    style.filter = line.style['--gradient_shadow'] || line.style['--clip_shadow'] || 'none';
  }
  const override = input.opacity_frames[node.className]?.[currentFrame(node)];
  if (override !== undefined) style.opacity = String(override);
  return style;
};
eval(input.script);
// Evaluate each animation independently at the requested frame time, including hard cuts.
function interpolate(a, b, amount) {
  if (Number.isFinite(Number(a)) && Number.isFinite(Number(b))) {
    return Number(a) + (Number(b) - Number(a)) * amount;
  }
  const numbers = /-?[\d.]+/g;
  const from = String(a).match(numbers)?.map(Number);
  const to = String(b).match(numbers)?.map(Number);
  if (!from || !to || from.length !== to.length) return a;
  let index = 0;
  return String(a).replace(numbers, () => {
    const value = from[index] + (to[index] - from[index]) * amount;
    index++;
    return value;
  });
}
function currentFrame(node) {
  let cap = node;
  while (cap.parentElement && !cap.className.split(' ').includes('cap')) cap = cap.parentElement;
  const time = cap.getAnimations({ subtree: true })[0]?.currentTime ?? 0;
  return Math.round(time / 1000 * input.variables.fps);
}
function animatedStyle(node, time) {
  const style = { ...node.style };
  for (const { keys, options, currentTime } of node.animations || []) {
    const milliseconds = time === undefined ? currentTime ?? 0 : time * 1000;
    const local = milliseconds - (options.delay || 0);
    if (local < 0 && !['both', 'backwards'].includes(options.fill)) continue;
    if (local >= options.duration && !['both', 'forwards'].includes(options.fill)) continue;
    const progress = Math.max(0, Math.min(1, local / options.duration));
    const properties = new Set(keys.flatMap((key) => Object.keys(key)));
    properties.delete('offset'); properties.delete('easing');
    for (const property of properties) {
      const frames = keys.filter((key) => property in key);
      let left = frames[0];
      let right = frames[frames.length - 1];
      for (const frame of frames) {
        if (frame.offset <= progress) left = frame;
        else { right = frame; break; }
      }
      const amount = right.offset > left.offset ?
        Math.max(0, (progress - left.offset) / (right.offset - left.offset)) : 0;
      style[property] = interpolate(left[property], right[property], amount);
    }
  }
  return style;
}
function seek(node, time) {
  const style = animatedStyle(node, time);
  return { className: node.className, textContent: node.textContent, style,
    children: (node.children || []).map((child) => seek(child, time)) };
}
setImmediate(() => {
  root.seeks = input.seek_times.map((time) => seek(root, time));
  process.stdout.write(JSON.stringify(root));
});
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=json.dumps(
            {
                "script": script,
                "variables": plan.variables,
                "glyph_width": glyph_width,
                "line_width": line_width,
                "duration": plan.frames / plan.variables["fps"],
                "height": plan.height,
                "width": plan.width,
                "seek_times": seek_times or [],
                "faces": [
                    {"family": bundled.family, "weight": face.weight, "style": face.style}
                    for bundled in [fonts.BUNDLED.get(plan.variables["font_name"])]
                    if bundled is not None
                    for face in bundled.faces
                ],
                "line": {"left": 100, "top": 100},
                "rects": rects,
                "opacity_frames": opacity_frames or {},
                "clip_rects": clip_rects or [],
            }
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_caption_plan_zero_offsets_preserve_original_bytes() -> None:
    style = _style()
    phrases = _phrases(style)
    original = caption_layer_plan(phrases, style, frames=90)
    explicit = caption_layer_plan(phrases, style, frames=90, offsets=[0, 0, 0])
    assert explicit == original
    assert json.dumps(explicit.variables) == json.dumps(original.variables)
    assert "band_top_px" not in explicit.variables
    assert all("offset_px" not in slot for slot in json.loads(explicit.variables["phrases"]))


def test_caption_plan_grows_around_both_offset_directions() -> None:
    style = _style()
    phrases = _phrases(style)
    original = caption_layer_plan(phrases, style, frames=90)
    moved = caption_layer_plan(phrases, style, frames=90, offsets=[-300, 0, 50])
    assert moved.y == original.y - 300
    assert moved.height == original.height + 350
    assert moved.variables["band_top_px"] == 300
    assert moved.variables["band_height_px"] == original.height
    assert [slot["offset_px"] for slot in json.loads(moved.variables["phrases"])] == [-300, 0, 50]


@pytest.mark.parametrize("offsets", [[0], [0, 0, 0, 0], [0, 0, 1.2], [0, 0, True]])
def test_caption_plan_requires_one_integer_offset_per_slot(offsets: list) -> None:
    style = _style()
    with pytest.raises(ValueError, match="one integer per caption slot"):
        caption_layer_plan(_phrases(style), style, frames=90, offsets=offsets)


@pytest.mark.parametrize("highlight", ["none", "pill", "slam", "weight_shift"])
@pytest.mark.parametrize("position", ["bottom", "middle"])
@pytest.mark.parametrize("offsets", [[0, -300], [0, 80]])
def test_caption_dom_placement_preserves_unmoved_anchor_and_sizing(
    highlight: str, position: str, offsets: list[int],
) -> None:
    style = _style(highlight=highlight, position=position)
    phrases = _phrases(style)[:2]
    original = caption_layer_plan(phrases, style, frames=60)
    moved = caption_layer_plan(phrases, style, frames=60, offsets=offsets)
    rects = [{"left": 100 + i * 100, "top": 100, "width": 80, "height": 40}
             for i in range(4)]
    before = _caption_dom(original, rects)
    after = _caption_dom(moved, rects)

    def output_anchor(plan: LayerPlan, cap: dict) -> float:
        css = cap["style"]
        if css.get("bottom") and css["bottom"] != "auto":
            return plan.y + plan.height - float(css["bottom"].removesuffix("px"))
        return plan.y + float(css["top"].removesuffix("px"))

    for slot, (old, new) in enumerate(zip(before["children"], after["children"], strict=True)):
        assert output_anchor(original, old) == output_anchor(moved, new)
        if slot == 0:
            assert "translate" not in new["style"]
        else:
            assert new["style"]["translate"] == f"0 {offsets[slot]}px"
        assert new["children"][0]["style"] == old["children"][0]["style"]
        assert new["children"][0]["animations"] == old["children"][0]["animations"]
        if highlight == "pill":
            assert new["children"][0]["children"][0]["style"] == (
                old["children"][0]["children"][0]["style"]
            )


@pytest.mark.parametrize("highlight", ["pill", "gradient", "editorial"])
@pytest.mark.parametrize("width_scale", [0.6, 1])
def test_new_caption_word_timelines_keep_layout_fixed(highlight: str, width_scale: float) -> None:
    style = _style(highlight=highlight, width_scale=width_scale)
    phrase = replace(_phrases(style)[0], start=0.3, end=0.6, stress_id="w1")
    plan = caption_layer_plan([phrase], style, frames=30)
    rects = [
        {"left": 100, "right": 220, "top": 100, "bottom": 150},
        {"left": 240, "right": 400, "top": 100, "bottom": 150},
    ]
    dom = _caption_dom(plan, rects)
    line = dom["children"][0]["children"][0]
    words = line["children"][1]
    spans = [child for child in words["children"] if child.get("className") == "w"]
    assert [span["textContent"] for span in spans] == ["First", "phrase."]
    assert spans[1]["style"]["fontStyle"] == "italic"
    assert words["style"].get("scale") == ("0.6 1" if width_scale < 1 else None)
    if highlight == "pill":
        for span, start in zip(spans, (0.3, 0.4), strict=True):
            keys = span["animations"][0]["keys"]
            assert keys[0]["color"] == "color-mix(in srgb, #111111 45%, #D8FF2F)"
            assert [key["offset"] for key in keys] == [0, start, start, 1]
            assert keys[-1]["color"] == "#111111"
    elif highlight == "gradient":
        for span, start, end in zip(spans, (0.3, 0.4), (0.4, 0.6), strict=True):
            assert "display" not in span["style"] and "position" not in span["style"]
            assert not span["children"]
            keys = span["animations"][0]["keys"]
            assert all(set(key) == {"backgroundPosition", "offset"} for key in keys)
            assert [key["offset"] for key in keys] == [0, start, end, 1]
            assert keys[0]["backgroundPosition"] == "calc(100% - 0.08em) 0, 0 0"
            assert keys[-1]["backgroundPosition"] == "0% 0, 0 0"
    else:
        assert not spans[0]["animations"] and "fontWeight" not in spans[0]["style"]
        assert spans[1]["style"]["fontWeight"] == "700"
        keys = spans[1]["animations"][0]["keys"]
        assert [key["offset"] for key in keys] == [0, 0.4, 0.4, 1]
        assert keys[0]["color"] == "#FFFFFF" and keys[-1]["color"] == "#D8FF2F"
    for element in (line, *spans, *(child for span in spans for child in span["children"])):
        for animation in element["animations"]:
            assert animation["options"] == {"duration": 1000, "fill": "both", "easing": "linear"}
            assert not any(
                property_name in key
                for key in animation["keys"]
                for property_name in ("fontWeight", "fontStyle", "width", "height")
            )


@pytest.mark.parametrize("two_lines", [False, True])
def test_pill_uses_scaled_union_and_line_dependent_radius(two_lines: bool) -> None:
    style = _style(highlight="pill", width_scale=0.6)
    plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
    second_top = 154 if two_lines else 100
    dom = _caption_dom(
        plan,
        [
            {"left": 200, "right": 400, "top": 100, "bottom": 150},
            {"left": 250, "right": 450, "top": second_top, "bottom": second_top + 50},
        ],
    )
    mark = dom["children"][0]["children"][0]["children"][0]
    height = (104 if two_lines else 50) + 48 * 0.08
    assert float(mark["style"]["left"].removesuffix("px")) == pytest.approx(100 - 48 * 0.4)
    assert float(mark["style"]["width"].removesuffix("px")) == pytest.approx(250 + 48 * 0.8)
    assert float(mark["style"]["top"].removesuffix("px")) == pytest.approx(-48 * 0.04)
    assert float(mark["style"]["height"].removesuffix("px")) == pytest.approx(height)
    radius = float(mark["style"]["borderRadius"].removesuffix("px"))
    assert radius == pytest.approx(48 * 0.6 if two_lines else height / 2)
    assert not mark["animations"]


def test_pill_reserves_room_for_padding_and_shadow_at_largest_size() -> None:
    style = _style(
        highlight="pill",
        width_scale=0.6,
        size_px=180,
        shadow_blur_px=20,
        shadow_x_px=-20,
        entry_rise_px=80,
    )
    plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
    dom = _caption_dom(
        plan,
        [
            {"left": 152, "right": 800, "top": 100, "bottom": 280},
            {"left": 400, "right": 928, "top": 302, "bottom": 482},
        ],
    )
    cap = dom["children"][0]
    line = cap["children"][0]
    mark = line["children"][0]
    assert float(line["style"]["maxWidth"].removesuffix("px")) * 0.6 == pytest.approx(776)
    left = 100 + float(mark["style"]["left"].removesuffix("px"))
    right = left + float(mark["style"]["width"].removesuffix("px"))
    shadow_room = 20 * 3 + 20
    assert left - shadow_room >= 0 and right + shadow_room <= 1080
    line_top = plan.height - plan.variables["bottom_padding_px"] - 180 * 2 * 1.12
    top = line_top + float(mark["style"]["top"].removesuffix("px"))
    bottom = top + float(mark["style"]["height"].removesuffix("px"))
    assert top >= 0 and bottom + 80 <= plan.height


_ALL_HIGHLIGHTS = tuple(json.loads(SCHEMA_PATH.read_text())["properties"]["captions"][
    "properties"]["highlight_style"]["enum"])


@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
def test_word_starting_after_phrase_end_keeps_keyframes_ordered(highlight: str) -> None:
    style = _style(highlight=highlight)
    first, second = _phrases(style)[:2]
    last = replace(first.words[-1], start=first.end + 0.05, end=first.end + 0.1)
    first = replace(first, words=(*first.words[:-1], last), stress_id=last.id)
    plan = caption_layer_plan([first, second], style, frames=60)
    dom = _caption_dom(plan, [])
    assert len(dom["children"]) == 2
    assert [slot["slot"] for slot in _shrink_summary(dom)["layout"]] == [0, 1]


def test_gradient_zero_length_word_fills_over_one_frame() -> None:
    style = _style(highlight="gradient")
    phrase = _phrases(style)[0]
    phrase = replace(phrase, words=(replace(phrase.words[0], end=phrase.words[0].start),))
    plan = caption_layer_plan([phrase], style, frames=30)
    dom = _caption_dom(plan, [])
    span = dom["children"][0]["children"][0]["children"][1]["children"][0]
    assert "display" not in span["style"] and "position" not in span["style"]
    assert not span["children"]
    animation = span["animations"][0]
    assert animation["options"] == {"duration": 1000, "fill": "both", "easing": "linear"}
    keys = animation["keys"]
    assert all(set(key) == {"backgroundPosition", "offset"} for key in keys)
    assert [key["offset"] for key in keys] == pytest.approx([0, 0.2, 0.2 + 1 / 30, 1])
    assert keys[2]["offset"] - keys[1]["offset"] == pytest.approx(1 / 30)


def test_editorial_without_stress_keeps_all_words_ordinary() -> None:
    style = _style(highlight="editorial", italic_stress=False)
    phrase = replace(_phrases(style)[0], stress_id=None)
    dom = _caption_dom(caption_layer_plan([phrase], style, frames=30), [])
    words = dom["children"][0]["children"][0]["children"][1]["children"]
    spans = [word for word in words if word.get("className") == "w"]
    assert all(not span["animations"] and not span["style"] for span in spans)


@pytest.mark.parametrize(
    ("item", "digest"),
    [
        ("caption-kinetic-slam",
         "86b7a54e045f626d1375636a2f1cde19c2f3cbdad1df1d7d0204fe6b4452044a"),
        ("caption-weight-shift",
         "8a5294debec3eadeb7ca0a8688c31a6e8c517ebb1c55ff410aa81df3807daa55"),
        ("caption-clip-wipe",
         "8b9e06353df3ea3bc0113a9c7ef3e480293a6ed75bee46744b053bf5e02dc5f6"),
    ],
)
def test_caption_registry_provenance(item: str, digest: str) -> None:
    provenance = json.loads((_TEMPLATES / "PROVENANCE.json").read_text())
    entries = [entry for entry in provenance["files"] if entry["item"] == item]
    assert len(entries) == 1
    entry = entries[0]
    upstream_path = f"registry/components/{item}/{item}.html"
    assert entry["upstream_path"] == upstream_path
    assert entry["upstream_sha256"] == digest
    assert entry["yap_path"] == "src/yap_studio/templates/captions/template.html"
    assert entry["changes"].endswith(".")
    metadata = json.loads((_TEMPLATES / "captions/template.json").read_text())
    assert {"item": item, "path": upstream_path} in metadata["derived_from"]
    source = (_TEMPLATES / "captions/template.html").read_text()
    header = source.split("<!--", 1)[1].split("-->", 1)[0]
    assert (
        f'Derived from HyperFrames registry item "{item}"\n'
        f'  ({upstream_path} at commit {provenance["upstream"]["commit"]})'
    ) in header
    assert provenance["upstream"]["copyright"] in header
    assert "Modified for Yap Studio, 2026:" in header
    assert "gsap" not in source.casefold()
    assert "data-no-timeline" in source


def _pilot_plan(style: ResolvedStyle) -> LayerPlan:
    words = tuple(
        CaptionWord(f"w{i}", text, 0.2 + i * 0.3, 0.5 + i * 0.3)
        for i, text in enumerate(("Small", "steps", "bring", "change"))
    )
    phrase = replace(_phrases(style)[0], words=words, start=0.2, end=1.6, stress_id="w3")
    return caption_layer_plan([phrase], style, frames=60)


def _dom_nodes(node: dict, class_name: str) -> list[dict]:
    found = [node] if node.get("className") == class_name else []
    return found + [
        item for child in node.get("children", []) for item in _dom_nodes(child, class_name)
    ]


def _word_plan(style: ResolvedStyle, text: str) -> LayerPlan:
    words = tuple(
        CaptionWord(f"w{i}", word, 0.2 + i * 0.15, 0.35 + i * 0.15)
        for i, word in enumerate(text.split())
    )
    phrase = replace(_phrases(style)[0], words=words, start=0.2, end=0.9, stress_id=None)
    return caption_layer_plan([phrase], style, frames=30)


def _word_rects(plan: LayerPlan) -> list[dict]:
    count = sum(len(phrase["words"]) for phrase in json.loads(plan.variables["phrases"]))
    return [
        {"left": 200 + i * 50, "right": 400 + i * 50,
         "top": 100, "bottom": 150, "width": 200, "height": 50}
        for i in range(count)
    ]


def _shrink_summary(dom: dict) -> dict:
    assert len(dom["warnings"]) == 1, dom["warnings"]
    warning = dom["warnings"][0]
    assert warning.startswith(_SHRINK_PREFIX), warning
    summary = json.loads(warning.removeprefix(_SHRINK_PREFIX))
    assert set(summary) == {"floor_px", "phrases", "layout"}
    return summary


def _expected_layout(plan: LayerPlan, rects: list[dict]) -> list[dict]:
    """Map the harness's fixed rectangles to the expected slot boxes."""
    layout = []
    index = 0
    for slot, phrase in enumerate(json.loads(plan.variables["phrases"])):
        boxes = []
        for _ in phrase["words"]:
            if index < len(rects):
                rect = rects[index]
                w = rect.get("width", rect["right"] - rect["left"])
                h = rect.get("height", rect["bottom"] - rect["top"])
                if w and h:
                    boxes.append({"x": round(rect["left"], 2), "y": round(rect["top"], 2),
                                  "w": round(w, 2), "h": round(h, 2)})
            index += 1
        layout.append({"slot": slot, "boxes": boxes})
    return layout


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
def test_caption_layout_reports_each_slot_and_rounded_word_boxes(highlight: str) -> None:
    style = _style(highlight=highlight)
    plan = caption_layer_plan(_phrases(style)[:2], style, frames=60)
    rects = _word_rects(plan)
    rects[0].update(left=200.123, top=100.456, width=199.987, height=50.234)
    dom = _caption_dom(plan, rects)
    assert _shrink_summary(dom)["layout"] == [
        {"slot": 0, "boxes": [
            {"x": 200.12, "y": 100.46, "w": 199.99, "h": 50.23},
            {"x": 250, "y": 100, "w": 200, "h": 50},
        ]},
        {"slot": 1, "boxes": [
            {"x": 300, "y": 100, "w": 200, "h": 50},
            {"x": 350, "y": 100, "w": 200, "h": 50},
        ]},
    ]


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
def test_caption_layout_restores_every_animation_time(highlight: str) -> None:
    plan = _pilot_plan(_style(highlight=highlight))
    dom = _caption_dom(plan, _word_rects(plan))
    animations = [animation for node in (
        dom["children"][0]["children"][0],
        *_dom_nodes(dom, "w"), *_dom_nodes(dom, "cap-row"), *_dom_nodes(dom, "cap-reveal"),
    ) for animation in node["animations"]]
    assert animations
    assert {animation["initialTime"] for animation in animations} >= {None}
    for animation in animations:
        assert animation["currentTime"] == animation["initialTime"]
        writes = animation["timeWrites"]
        assert writes[:-1] == pytest.approx([frame / 30 * 1000 for frame in range(6, 49)])
        assert writes[-1] == animation["initialTime"]


def test_caption_layout_unions_word_rectangles_over_every_drawn_frame() -> None:
    style = _style(highlight="none")
    phrase = replace(_phrases(style)[0], start=0.3, end=0.6)
    plan = caption_layer_plan([phrase], style, frames=30)
    rects = _word_rects(plan)
    invisible = {"left": -10000, "top": -10000, "width": 20000, "height": 20000}
    rects[0]["frames"] = {
        9: invisible, 18: invisible,
        10: {"left": 190, "top": 108, "width": 220, "height": 60},
        13: {"left": 180, "top": 90, "width": 200, "height": 50},
        17: {"left": 195, "top": 98, "width": 210, "height": 52},
    }
    rects[1]["frames"] = {
        9: invisible, 18: invisible,
        11: {"left": 240, "top": 105, "width": 220, "height": 55},
    }
    dom = _caption_dom(plan, rects)
    assert _shrink_summary(dom)["layout"] == [{"slot": 0, "boxes": [
        {"x": 180, "y": 90, "w": 230, "h": 78},
        {"x": 240, "y": 100, "w": 220, "h": 60},
    ]}]
    for span in _dom_nodes(dom, "w"):
        assert span["rectReads"] == [[frame / 30 * 1000] for frame in range(10, 18)]


@pytest.mark.parametrize("ancestor", ["w", "cap-words", "cap-line", "cap cap-bottom"])
def test_caption_layout_ignores_zero_opacity_on_word_or_any_ancestor(ancestor: str) -> None:
    style = _style(highlight="none")
    plan = caption_layer_plan([replace(_phrases(style)[0], start=0.3, end=0.6)], style, frames=30)
    rects = _word_rects(plan)
    for rect in rects:
        rect["frames"] = {
            frame: {"left": -10000, "top": -10000, "width": 20000, "height": 20000}
            for frame in (10, 12)
        }
    dom = _caption_dom(plan, rects, opacity_frames={ancestor: {10: 0, 11: 0.01, 12: 0, 13: 0.5}})
    assert _shrink_summary(dom)["layout"] == _expected_layout(plan, rects)
    for span in _dom_nodes(dom, "w"):
        assert span["rectReads"] == [[frame / 30 * 1000] for frame in (11, 13, 14, 15, 16, 17)]


def test_caption_layout_slam_unions_only_frames_where_each_word_is_visible() -> None:
    style = _style(highlight="slam")
    phrase = replace(_phrases(style)[0], start=0.3, end=0.6, words=(
        CaptionWord("w0", "Go", 0.3, 0.4),
        CaptionWord("w1", "up", 0.4, 0.6),
        CaptionWord("w2", "later", 0.7, 0.8),
    ), stress_id=None)
    plan = caption_layer_plan([phrase], style, frames=30)
    rects = _word_rects(plan)
    outside = {"left": -10000, "top": -10000, "width": 20000, "height": 20000}
    rects[0]["frames"] = {
        9: outside,
        10: {"left": 170, "top": 110, "width": 260, "height": 65},
        **dict.fromkeys(range(12, 19), outside),
    }
    rects[1]["frames"] = {
        **dict.fromkeys(range(9, 13), outside), 18: outside,
        13: {"left": 220, "top": 95, "width": 260, "height": 65},
    }
    dom = _caption_dom(plan, rects)
    assert _shrink_summary(dom)["layout"] == [{"slot": 0, "boxes": [
        {"x": 170, "y": 100, "w": 260, "h": 75},
        {"x": 220, "y": 95, "w": 260, "h": 65},
    ]}]
    spans = _dom_nodes(dom, "w")
    for span, frames in zip(spans[:2], (range(10, 12), range(13, 18)), strict=True):
        assert span["rectReads"] == [[frame / 30 * 1000] * 3 for frame in frames]
    assert "rectReads" not in spans[2]


@pytest.mark.parametrize(("start", "end", "frames"), [
    (0.3, 0.305, range(9, 11)), (-0.2, 0.2, range(0, 7)),
])
def test_caption_layout_samples_frame_window_and_allows_no_drawn_frames(
    start: float, end: float, frames: range,
) -> None:
    style = _style(highlight="none")
    phrase = replace(_phrases(style)[0], start=start, end=end)
    plan = caption_layer_plan([phrase], style, frames=30)
    dom = _caption_dom(plan, _word_rects(plan))
    animation = dom["children"][0]["children"][0]["animations"][0]
    assert animation["timeWrites"][:-1] == pytest.approx([frame / 30 * 1000 for frame in frames])
    if start >= 0:
        assert _shrink_summary(dom)["layout"] == [{"slot": 0, "boxes": []}]


def test_caption_layout_restores_animation_times_when_measurement_raises() -> None:
    plan = _pilot_plan(_style(highlight="slam"))
    source = (_TEMPLATES / "captions" / "template.html").read_text()
    source = source.replace("const rect = span.getBoundingClientRect();",
                            "throw new Error('layout failed');", 1)
    source = source.replace(
        "  });\n})();", "  }).catch((error) => { root.error = error.message; });\n})();",
    )
    dom = _caption_dom(plan, _word_rects(plan), source=source)
    assert dom["error"] == "layout failed"
    for node in (dom["children"][0]["children"][0], *_dom_nodes(dom, "w")):
        for animation in node["animations"]:
            assert animation["currentTime"] == animation["initialTime"]
            assert animation["timeWrites"][-1] == animation["initialTime"]


@pytest.mark.render
@pytest.mark.parametrize("checks", [False, True])
def test_real_caption_layout_error_refuses_output(tmp_path: Path, checks: bool) -> None:
    plan = _pilot_plan(_style(highlight="slam"))
    comp = caption_stage.prepare_layer_plan(plan, tmp_path / "throwing")
    html = comp / "index.html"
    source = html.read_text()
    injected = source.replace("const rect = span.getBoundingClientRect();",
                              "throw new Error('injected caption layout failure');", 1)
    assert injected != source
    html.write_text(injected)
    out = tmp_path / "throwing.mov"
    # Exercise render itself even when the advisory preflight checks are disabled.
    with pytest.raises(YapError, match="injected caption layout failure") as caught:
        hyperframes.render_layer(
            comp, out, frames=plan.frames, variables=plan.variables, checks=checks,
            workers=hyperframes.CAPTION_WORKERS, browser_gpu=hyperframes.CAPTION_BROWSER_GPU,
        )
    assert caught.value.code == "renderer_script_error"
    assert not out.exists()
    assert not list(comp.glob("*.mov"))
    assert not (comp / "variables.json").exists()


def test_caption_layout_skips_only_zero_size_word_boxes() -> None:
    plan = _pilot_plan(_style(highlight="none"))
    rects = _word_rects(plan)
    rects[0]["width"] = 0
    rects[1]["height"] = 0
    rects[2].update(width=0.004, height=0.004)
    dom = _caption_dom(plan, rects)
    assert _shrink_summary(dom)["layout"] == [{"slot": 0, "boxes": [
        {"x": 300, "y": 100, "w": 0, "h": 0},
        {"x": 350, "y": 100, "w": 200, "h": 50},
    ]}]


@pytest.mark.parametrize(
    "highlight", [style for style in _HIGHLIGHTS if style not in _REGISTRY_LOOKS],
)
@pytest.mark.parametrize("width_scale", [0.6, 1])
def test_long_word_shrinks_only_its_phrase_and_reports_fit(
    highlight: str, width_scale: float,
) -> None:
    style = _style(highlight=highlight, size_px=180, width_scale=width_scale)
    # At 0.6, documentation fits the wider line; the last word forces a shrink there too.
    text = "Write documentation" if width_scale == 1 else "Write documentation comprehensiveness"
    plan = _word_plan(style, text)
    dom = _caption_dom(plan, _word_rects(plan))
    line = dom["children"][0]["children"][0]
    size = float(line["style"]["fontSize"].removesuffix("px"))
    available = float(line["style"].get("maxWidth", "940px").removesuffix("px"))
    assert 32 <= size < 180
    for word in text.split():
        measured = len(word) * (size * 0.6 + plan.variables["letter_spacing_px"])
        assert measured <= available
    assert _shrink_summary(dom) == {
        "floor_px": 32,
        "phrases": [{"slot": 0, "size_px": round(size, 2), "fits": True}],
        "layout": _expected_layout(plan, _word_rects(plan)),
    }


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
@pytest.mark.parametrize("size_px", [48, 180])
def test_fitting_words_preserve_dom_and_registry_sizes(highlight: str, size_px: int) -> None:
    style = _style(highlight=highlight, size_px=size_px)
    plan = _pilot_plan(style) if size_px == 48 else _word_plan(style, "Go do it now")
    source = (_TEMPLATES / "captions" / "template.html").read_text()
    disabled = source.replace("const MIN_SIZE_PX = 32;", "const MIN_SIZE_PX = Infinity;")
    assert disabled != source
    dom = _caption_dom(plan, _word_rects(plan))
    baseline = _caption_dom(plan, _word_rects(plan), source=disabled)
    line = dom["children"][0]["children"][0]
    if highlight in _REGISTRY_LOOKS:
        assert float(line["style"]["fontSize"].removesuffix("px")) == size_px
    else:
        assert "fontSize" not in line["style"]
        assert line["children"][1]["widthChecks"] == 1
    assert _shrink_summary(dom) == {
        "floor_px": 32, "phrases": [], "layout": _expected_layout(plan, _word_rects(plan)),
    }
    assert _shrink_summary(baseline) == {
        "floor_px": size_px, "phrases": [], "layout": _expected_layout(plan, _word_rects(plan)),
    }
    dom.pop("warnings")
    baseline.pop("warnings")
    assert dom == baseline


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
def test_empty_caption_slots_emit_one_empty_shrink_summary(highlight: str) -> None:
    plan = caption_layer_plan([], _style(highlight=highlight), frames=3)
    dom = _caption_dom(plan, [])
    assert not dom["children"]
    assert _shrink_summary(dom) == {"floor_px": 32, "phrases": [], "layout": []}


@pytest.mark.parametrize("highlight", ["none", "slam", "weight_shift", "clip_wipe"])
def test_unbreakable_word_stops_at_readable_floor_and_reports_overflow(highlight: str) -> None:
    plan = _word_plan(_style(highlight=highlight, size_px=180), "x" * 60)
    dom = _caption_dom(plan, _word_rects(plan))
    line = dom["children"][0]["children"][0]
    assert line["style"]["fontSize"] == "32px"
    assert _shrink_summary(dom) == {
        "floor_px": 32, "phrases": [{"slot": 0, "size_px": 32, "fits": False}],
        "layout": _expected_layout(plan, _word_rects(plan)),
    }


@pytest.mark.parametrize("highlight", ["none", *_REGISTRY_LOOKS])
def test_word_overflow_at_style_floor_is_reported(highlight: str) -> None:
    plan = _word_plan(_style(highlight=highlight, size_px=32), "x" * 60)
    dom = _caption_dom(plan, _word_rects(plan))
    line = dom["children"][0]["children"][0]
    assert float(line["style"].get("fontSize", "32px").removesuffix("px")) == 32
    assert _shrink_summary(dom) == {
        "floor_px": 32, "phrases": [{"slot": 0, "size_px": 32, "fits": False}],
        "layout": _expected_layout(plan, _word_rects(plan)),
    }


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
@pytest.mark.parametrize("slot", [0, 1])
def test_shrink_summary_keeps_phrase_slot_indices(highlight: str, slot: int) -> None:
    style = _style(highlight=highlight, size_px=180)
    phrases = [
        replace(phrase, words=tuple(
            replace(word, text=text) for word, text in zip(phrase.words, ("Go", "now"), strict=True)
        ))
        for phrase in _phrases(style)[:2]
    ]
    phrases[slot] = replace(
        phrases[slot], words=(replace(phrases[slot].words[0], text="documentation"),),
        stress_id=None,
    )
    plan = caption_layer_plan(phrases, style, frames=60)
    dom = _caption_dom(plan, _word_rects(plan))
    lines = [cap["children"][0] for cap in dom["children"]]
    size = float(lines[slot]["style"]["fontSize"].removesuffix("px"))
    assert 32 <= size < 180
    if highlight in _REGISTRY_LOOKS:
        assert float(lines[1 - slot]["style"]["fontSize"].removesuffix("px")) == 180
    else:
        assert "fontSize" not in lines[1 - slot]["style"]
    assert _shrink_summary(dom) == {
        "floor_px": 32, "phrases": [{"slot": slot, "size_px": round(size, 2), "fits": True}],
        "layout": _expected_layout(plan, _word_rects(plan)),
    }


@pytest.mark.parametrize("highlight", ["moving_block", "underline", "pill"])
@pytest.mark.parametrize("width_scale", [0.6, 1])
def test_shrunk_phrase_mark_geometry_uses_effective_size(
    highlight: str, width_scale: float,
) -> None:
    plan = _word_plan(
        _style(highlight=highlight, size_px=180, width_scale=width_scale),
        "Write documentation comprehensiveness",
    )
    rects = _word_rects(plan)
    # This separation crosses the shrunk pill's two-line threshold, but not the original one.
    rects[1].update(top=180, bottom=380, height=200)
    rects[2].update(top=180, bottom=380, height=200)
    dom = _caption_dom(plan, rects)
    line = dom["children"][0]["children"][0]
    mark = line["children"][0]
    size = float(line["style"]["fontSize"].removesuffix("px"))
    assert 32 <= size < 180
    if highlight == "pill":
        geometry = mark["style"]
        assert float(geometry["left"].removesuffix("px")) == pytest.approx(100 - size * 0.4)
        assert float(geometry["width"].removesuffix("px")) == pytest.approx(300 + size * 0.8)
        assert float(geometry["top"].removesuffix("px")) == pytest.approx(-size * 0.04)
        assert float(geometry["height"].removesuffix("px")) == pytest.approx(280 + size * 0.08)
        assert float(geometry["borderRadius"].removesuffix("px")) == pytest.approx(size * 0.6)
    else:
        keys = mark["animations"][0]["keys"]
        for key in keys:
            if highlight == "underline":
                assert float(key["height"].removesuffix("px")) == pytest.approx(
                    max(6, size * 0.075)
                )
            else:
                assert float(key["width"].removesuffix("px")) == pytest.approx(200 + size * 0.16)
        expected_left = 100 - size * 0.08 if highlight == "moving_block" else 100
        assert float(keys[0]["left"].removesuffix("px")) == pytest.approx(expected_left)


def test_shrink_remeasures_fixed_letter_spacing() -> None:
    plan = _word_plan(_style(highlight="none", size_px=180, letter_spacing_px=10), "documentation")
    dom = _caption_dom(plan, [])
    line = dom["children"][0]["children"][0]
    size = float(line["style"]["fontSize"].removesuffix("px"))
    assert 32 < size < 180
    assert len("documentation") * (size * 0.6 + 10) <= 940
    assert _shrink_summary(dom)["phrases"][0]["fits"] is True


def test_slow_shrink_converges_to_largest_fitting_size_without_floor_jump() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    spacing = schema["properties"]["captions"]["properties"]["letter_spacing_px"]["maximum"]
    text = "i" * 50
    glyph_width = 0.2
    plan = _word_plan(_style(highlight="none", size_px=180, letter_spacing_px=spacing), text)
    # Narrow glyphs leave fixed tracking dominant; the old eighth step jumped to 32 px.
    dom = _caption_dom(plan, [], glyph_width=glyph_width)
    line = dom["children"][0]["children"][0]
    size = float(line["style"]["fontSize"].removesuffix("px"))
    available = float(line["style"].get("maxWidth", "940px").removesuffix("px"))
    limit = (available / len(text) - spacing) / glyph_width
    assert size > 43
    assert 0 <= limit - size <= 0.01
    assert len(text) * (size * glyph_width + spacing) <= available
    assert len(text) * ((size + 0.05) * glyph_width + spacing) > available
    assert line["children"][1]["widthChecks"] <= 24
    assert _shrink_summary(dom) == {
        "floor_px": 32, "phrases": [{"slot": 0, "size_px": size, "fits": True}],
        "layout": _expected_layout(plan, []),
    }


def test_hyphen_breaks_and_wider_lines_avoid_unneeded_shrink() -> None:
    for text, width_scale in (("twenty-first-century", 1), ("documentation", 0.6)):
        plan = _word_plan(_style(highlight="none", size_px=180, width_scale=width_scale), text)
        dom = _caption_dom(plan, [])
        line = dom["children"][0]["children"][0]
        assert "fontSize" not in line["style"]
        assert _shrink_summary(dom) == {
            "floor_px": 32, "phrases": [], "layout": _expected_layout(plan, []),
        }


@pytest.mark.parametrize("highlight", ["none", "clip_wipe"])
def test_floor_never_enlarges_a_smaller_input_size(highlight: str) -> None:
    plan = _word_plan(_style(highlight=highlight), "x" * 100)
    plan = replace(plan, variables={**plan.variables, "size_px": 24})
    dom = _caption_dom(plan, [])
    line = dom["children"][0]["children"][0]
    if highlight == "clip_wipe":
        assert line["style"]["fontSize"] == "24px"
    else:
        assert "fontSize" not in line["style"]
    assert _shrink_summary(dom) == {
        "floor_px": 24, "phrases": [{"slot": 0, "size_px": 24, "fits": False}],
        "layout": _expected_layout(plan, []),
    }


@pytest.mark.parametrize("width_scale", [0.6, 1])
def test_slam_seeks_show_only_current_word_and_stress_color(width_scale: float) -> None:
    plan = _pilot_plan(_style(highlight="slam", width_scale=width_scale))
    times = [0.35, 0.65, 0.95, 1.25, 0.2, 0.32, 1.7, 0.35]
    dom = _caption_dom(plan, [], seek_times=times)
    spans = _dom_nodes(dom, "w")
    assert [span["textContent"] for span in spans] == ["Small", "steps", "bring", "change"]
    assert spans[3]["style"]["color"] == plan.variables["highlight_color"]
    assert all("color" not in span["style"] for span in spans[:3])
    assert spans[3]["style"]["fontStyle"] == "italic"
    for index, snapshot in enumerate(dom["seeks"][:4]):
        visible = [span["textContent"] for span in _dom_nodes(snapshot, "w")
                   if span["style"]["opacity"] > 0]
        assert visible == [spans[index]["textContent"]]
    assert _dom_nodes(dom["seeks"][4], "w")[0]["style"]["transform"] == "scale(1.3)"
    assert _dom_nodes(dom["seeks"][5], "w")[0]["style"]["transform"] == "scale(1)"
    assert all(span["style"]["opacity"] == 0 for span in _dom_nodes(dom["seeks"][6], "w"))
    assert dom["seeks"][0] == dom["seeks"][-1]
    cap = dom["children"][0]
    assert cap["style"]["top"] == f"{plan.height / 2:g}px"
    assert cap["style"]["bottom"] == "auto"
    words = cap["children"][0]["children"][1]
    assert words["style"]["scale"] == f"{width_scale * 0.8:g} 1"


@pytest.mark.parametrize(
    ("font", "bold", "light"), [("Inter", 800, 300), ("Gelasio", 700, 400), ("Alata", 700, 400)]
)
def test_weight_shift_seeks_change_only_the_active_line(
    font: str, bold: int, light: int,
) -> None:
    plan = _pilot_plan(_style(highlight="weight_shift", font=font))
    dom = _caption_dom(plan, [], seek_times=[0.35, 0.65, 0.95, 1.25, 0.85, 0.35])
    rows = _dom_nodes(dom, "cap-row")
    assert len(rows) == 2
    assert [[span["textContent"] for span in _dom_nodes(row, "w")] for row in rows] == [
        ["Small", "steps"], ["bring", "change"],
    ]
    for index, snapshot in enumerate(dom["seeks"][:4]):
        sought = _dom_nodes(snapshot, "cap-row")
        active = int(index >= 2)
        assert sought[active]["style"]["fontWeight"] == bold
        assert sought[1 - active]["style"]["fontWeight"] == light
        assert sought[active]["style"]["opacity"] == 1
        assert sought[1 - active]["style"]["opacity"] == 0.75
    transition = _dom_nodes(dom["seeks"][4], "cap-row")
    assert all(light < row["style"]["fontWeight"] < bold for row in transition)
    assert dom["seeks"][0] == dom["seeks"][-1]
    assert _dom_nodes(dom, "w")[3]["style"]["fontStyle"] == "italic"
    for row in rows:
        keys = row["animations"][0]["keys"]
        assert keys[2]["offset"] - keys[1]["offset"] == pytest.approx(0.1 / 2)


def test_weight_shift_single_word_keeps_one_bold_line() -> None:
    style = _style(highlight="weight_shift")
    phrase = _phrases(style)[0]
    plan = caption_layer_plan([replace(phrase, words=phrase.words[:1])], style, frames=30)
    dom = _caption_dom(plan, [])
    rows = _dom_nodes(dom, "cap-row")
    assert len(rows) == 1
    assert rows[0]["style"]["fontWeight"] == "800"
    assert rows[0]["style"]["opacity"] == "1"
    assert not rows[0]["animations"]


@pytest.mark.parametrize(("size", "row_count"), [(40, 1), (48, 2)])
def test_clip_wipe_seeks_dim_future_words_and_keep_completed_words_revealed(
    size: int, row_count: int,
) -> None:
    plan = _pilot_plan(_style(highlight="clip_wipe", size_px=size))
    dom = _caption_dom(plan, [], seek_times=[0.35, 0.55, 1.45, 0.35])
    first = dom["seeks"][0]
    assert all(span["style"]["opacity"] == "0.35" for span in _dom_nodes(first, "cap-dim"))
    reveals = _dom_nodes(first, "cap-reveal")
    right_inset = float(reveals[0]["style"]["clipPath"].split()[1].removesuffix("%"))
    assert right_inset == pytest.approx(50)
    assert all(span["style"]["clipPath"] == "inset(0 100% 0 0)" for span in reveals[1:])
    assert _dom_nodes(dom["seeks"][1], "cap-reveal")[0]["style"]["clipPath"] == (
        "inset(0 0% 0 0)"
    )
    assert all(span["style"]["clipPath"] == "inset(0 0% 0 0)"
               for span in _dom_nodes(dom["seeks"][2], "cap-reveal"))
    assert _dom_nodes(dom, "w")[3]["style"]["color"] == plan.variables["highlight_color"]
    assert dom["seeks"][0] == dom["seeks"][-1]
    assert len(_dom_nodes(dom, "cap-row")) == row_count
    line = dom["children"][0]["children"][0]
    assert float(line["style"]["fontSize"].removesuffix("px")) == size
    assert line["style"]["--clip_shadow"] == "drop-shadow(3px 4px 5px rgb(0 0 0 / 0.25))"


@pytest.mark.parametrize("word_duration", [0, 0.02, 0.2])
def test_clip_wipe_uses_word_duration_with_eighty_millisecond_minimum(word_duration: float) -> None:
    style = _style(highlight="clip_wipe")
    phrase = _phrases(style)[0]
    word = replace(phrase.words[0], end=phrase.words[0].start + word_duration)
    plan = caption_layer_plan([replace(phrase, words=(word,))], style, frames=30)
    dom = _caption_dom(plan, [])
    keys = _dom_nodes(dom, "cap-reveal")[0]["animations"][0]["keys"]
    assert keys[2]["offset"] - keys[1]["offset"] == pytest.approx(max(0.08, word_duration))


@pytest.mark.parametrize("highlight", ["slam", "weight_shift", "clip_wipe"])
@pytest.mark.parametrize("position", ["bottom", "middle"])
@pytest.mark.parametrize("width_scale", [0.6, 1])
def test_registry_caption_layout_fits_band_with_extreme_tokens(
    highlight: str, position: str, width_scale: float,
) -> None:
    plan = _pilot_plan(_style(
        highlight=highlight, position=position, width_scale=width_scale,
        size_px=180, letter_spacing_px=10, entry_rise_px=80, shadow_blur_px=20,
        shadow_x_px=-20, shadow_y_px=20,
    ))
    dom = _caption_dom(plan, [])
    cap = dom["children"][0]
    line = cap["children"][0]
    size = float(line["style"]["fontSize"].removesuffix("px"))
    available = float(line["style"]["maxWidth"].removesuffix("px"))
    scale = width_scale * (0.8 if highlight == "slam" else 1)
    assert available * scale <= plan.width - 2 * plan.variables["padding_px"]
    rows = _dom_nodes(dom, "cap-row")
    groups = [[span] for span in _dom_nodes(dom, "w")] if highlight == "slam" else [
        _dom_nodes(row, "w") for row in rows
    ]
    if highlight == "slam":
        assert not rows
    else:
        assert 1 <= len(rows) <= 2
    for group in groups:
        text = " ".join(span["textContent"] for span in group)
        measured = len(text) * (size * 0.6 * (1 + 800 / 2000) + 10)
        assert measured * (1.3 if highlight == "slam" else 1) <= available + 0.001
    height = size * 1.12 * (1.3 if highlight == "slam" else len(rows))
    rise_room = 2 * 80 if highlight == "slam" else 80
    assert height + 2 * plan.variables["padding_px"] + rise_room <= plan.height + 0.001
    assert plan.y + plan.height <= 1740


@pytest.mark.parametrize("highlight", ["slam", "weight_shift", "clip_wipe"])
def test_registry_caption_slots_remain_plain_text(highlight: str) -> None:
    style = _style(highlight=highlight)
    phrase = _phrases(style)[0]
    word = replace(phrase.words[0], text='<img src=x onerror="1">')
    plan = caption_layer_plan([replace(phrase, words=(word,), stress_id=None)], style, frames=30)
    dom = _caption_dom(plan, [])
    spans = _dom_nodes(dom, "w")
    assert spans[0]["textContent"] == word.text
    assert "fontStyle" not in spans[0]["style"] and "color" not in spans[0]["style"]
    if highlight == "clip_wipe":
        assert all(child["textContent"] == word.text for child in spans[0]["children"])


@pytest.mark.parametrize("preset", ["clean_minimal", "electric", "warm_editorial"])
def test_shipped_caption_band_and_style_tokens(preset: str) -> None:
    style = resolve(f"preset:{preset}")
    plan = caption_layer_plan(_phrases(style), style, frames=90)
    assert plan.width == 1080 and plan.x == 0
    assert plan.y >= 0 and plan.y + plan.height <= style.data["internal"]["caption_safe_bottom_px"]
    assert plan.height >= 2 * style.data["captions"]["size_px"] * 1.12
    assert plan.variables["highlight_style"] == style.data["captions"]["highlight_style"]
    assert plan.variables["entry_ms"] == style.data["captions"]["entry_ms"]
    assert style.data["captions"]["font"] in plan.variables["font_family"]


def test_caption_width_scale_is_passed_to_template() -> None:
    style = _style(width_scale=0.8)
    plan = caption_layer_plan(_phrases(style), style, frames=90)
    assert plan.variables["width_scale"] == style.data["captions"]["width_scale"] == 0.8


@pytest.mark.parametrize("position", ["bottom", "middle"])
def test_caption_width_scale_preserves_band(position: str) -> None:
    normal = _style(position=position, width_scale=1)
    condensed = _style(position=position, width_scale=0.6)
    normal_plan = caption_layer_plan(_phrases(normal), normal, frames=90)
    condensed_plan = caption_layer_plan(_phrases(condensed), condensed, frames=90)
    assert (condensed_plan.y, condensed_plan.height) == (normal_plan.y, normal_plan.height)


@pytest.mark.parametrize("position", ["bottom", "middle"])
def test_caption_band_extreme_tokens_stay_inside_safe_bottom(position: str) -> None:
    style = _style(
        position=position,
        top_px=1700,
        size_px=180,
        entry_rise_px=80,
        shadow_blur_px=20,
        shadow_y_px=20,
    )
    plan = caption_layer_plan([], style, frames=3)
    assert plan.y + plan.height == 1740
    assert plan.height >= 180 * 2 * 1.12 + 80


@pytest.mark.parametrize("size_px", [76, 92, 104, 112])
@pytest.mark.parametrize("top_px", [1503, 1600, 1650])
def test_bottom_caption_anchors_last_line_and_warns_only_when_it_cannot_fit(
    size_px: int, top_px: int,
) -> None:
    style = _style(
        position="bottom", top_px=top_px, size_px=size_px,
        shadow_blur_px=5, shadow_x_px=3, shadow_y_px=4, entry_rise_px=12,
    )
    plan = caption_layer_plan(_phrases(style), style, frames=90)
    padding = math.ceil(size_px * 0.08 + 5 * 3 + 4 + 8)
    line_height = size_px * 1.12
    safe_bottom = 1740
    expected_top = min(top_px, safe_bottom - line_height - 12 - padding)
    # Older plans put a one-line phrase at the band's top padding.
    line_top = (
        plan.y + plan.height - plan.variables["bottom_padding_px"] - line_height
        if "bottom_padding_px" in plan.variables else plan.y + plan.variables["padding_px"]
    )
    assert line_top == pytest.approx(expected_top, abs=1e-9)
    assert plan.y + plan.height == math.ceil(min(top_px + line_height + 12 + padding,
                                               safe_bottom))
    assert plan.y + plan.height <= safe_bottom
    assert plan.variables["position"] == "bottom"
    assert plan.variables["padding_px"] == padding
    assert plan.height >= 2 * line_height + 12 + 2 * padding
    assert bool(plan.warnings) is (expected_top < top_px)
    placement = plan.placement("captions/captions.mov")
    if expected_top < top_px:
        warning = " ".join(plan.warnings)
        assert "clamped" in warning and "one line" in warning
        assert str(top_px) in warning and str(safe_bottom) in warning
        assert placement["warnings"] == list(plan.warnings)
    else:
        assert "warnings" not in placement


@pytest.mark.parametrize("size_px", [76, 92, 104, 112])
@pytest.mark.parametrize("top_px", [1100, 1503, 1650])
def test_middle_caption_keeps_existing_center_and_safe_clamp(size_px: int, top_px: int) -> None:
    style = _style(
        position="middle", top_px=top_px, size_px=size_px,
        shadow_blur_px=5, shadow_x_px=3, shadow_y_px=4, entry_rise_px=12,
    )
    plan = caption_layer_plan([], style, frames=3)
    padding = math.ceil(size_px * 0.08 + 5 * 3 + 4 + 8)
    height = math.ceil(2 * size_px * 1.12 + 12 + 2 * padding)
    assert (plan.y, plan.height) == (max(0, min(round(top_px - height / 2), 1740 - height)),
                                    height)


@pytest.mark.parametrize("highlight", [
    "moving_block", "underline", "color", "pill", "gradient", "editorial",
    "slam", "weight_shift", "clip_wipe", "none",
])
@pytest.mark.parametrize("position", ["bottom", "middle"])
def test_caption_template_uses_position_anchor_for_every_highlight(
    highlight: str, position: str,
) -> None:
    style = _style(highlight=highlight, position=position, width_scale=0.6, size_px=104)
    plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
    dom = _caption_dom(plan, [
        {"left": 200, "right": 400, "top": 100, "bottom": 150, "width": 200, "height": 50},
        {"left": 250, "right": 450, "top": 217, "bottom": 267, "width": 200, "height": 50},
    ])
    cap = dom["children"][0]
    if highlight == "slam":
        assert cap["style"]["top"] == f"{plan.height / 2:g}px"
        assert cap["style"]["bottom"] == "auto"
    elif position == "bottom":
        assert "top" not in cap["style"]
        assert float(cap["style"]["bottom"].removesuffix("px")) == pytest.approx(
            plan.variables["bottom_padding_px"]
        )
    else:
        assert cap["style"]["top"] == f'{plan.variables["padding_px"]}px'
        assert "bottom" not in cap["style"]


def test_middle_band_is_centered_and_phrase_slots_preserve_stress() -> None:
    style = _style(position="middle", top_px=1100)
    phrases = _phrases(style)
    phrases[0] = replace(phrases[0], stress_id="w1")
    plan = caption_layer_plan(phrases, style, frames=90)
    assert abs(plan.y + plan.height / 2 - 1100) <= 0.5
    slots = json.loads(plan.variables["phrases"])
    assert slots[0]["stress"] == 1
    assert slots[0]["words"][1] == {"id": "w1", "source_ids": ["w1"],
                                  "text": "phrase.", "start": 0.4, "end": 0.7}


@pytest.mark.parametrize(
    ("look", "width"), [("paper_title", 1080), ("quiet_serif", 940), ("bold_band", 1160)]
)
def test_title_box_sizes_and_frame_snap(look: str, width: int) -> None:
    style = _style(look=look)
    style.data["hook"]["duration_s"] = 1.015
    plan = title_layer_plan("Opening", "words", style)
    assert plan.width == width and plan.x == (1080 - width) // 2
    assert 0 <= plan.y < plan.y + plan.height <= 1920
    assert plan.frames == 30
    assert plan.variables["plain"] == "Opening" and plan.variables["key"] == "words"
    assert plan.variables["look"] == look


def test_none_and_zero_duration_titles_refused() -> None:
    with pytest.raises(ValidationError, match="hook.style is none"):
        title_layer_plan("Opening", "words", _style(look="none"))
    style = _style()
    style.data["hook"]["duration_s"] = 0
    with pytest.raises(ValidationError, match="duration_s"):
        title_layer_plan("Opening", "words", style)


def test_prepare_composition_keeps_slot_text_out_of_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    checks = {"schema": "yap-studio/layer-checks@1", "status": "passed", "passes": {}}

    def inspect(
        composition_dir: Path, out: Path, *, frames: int, variables: dict, fps: int = 30
    ) -> hyperframes.LayerResult:
        source = (composition_dir / "index.html").read_text()
        _validate_variables(source, variables)
        assert variables["plain"] == "<em>Words</em>"
        assert "<em>Words</em>" not in source
        assert "@font-face" in source and "BUNDLED_FONT_FACES" not in source
        assert list((composition_dir / "fonts").glob("*.woff2"))
        assert hyperframes.preflight(composition_dir, frames, fps) == hyperframes.layer_duration_s(
            frames,
            fps,
        )
        assert 'data-width="940"' in source
        seen.append(variables)
        return hyperframes.LayerResult(out, frames, plan.width, plan.height, frames / fps, checks)

    monkeypatch.setattr(hyperframes, "render_layer", inspect)
    plan = title_layer_plan("<em>Words</em>", "", _style(look="quiet_serif"), fps=24)
    result = render_layer_plan(plan, tmp_path, tmp_path / "layer.mov")
    assert result.path == tmp_path / "layer.mov"
    assert result.checks is checks
    assert seen == [plan.variables]
    invalid = replace(plan, variables={**plan.variables, "look": "unknown"})
    with pytest.raises(ProcessingError, match="Variable validation failed") as caught:
        render_layer_plan(invalid, tmp_path / "invalid", tmp_path / "invalid.mov")
    assert caught.value.returncode == 1


@pytest.mark.parametrize("template,workers", [("captions", "auto"), ("title", "1")])
def test_layer_plan_passes_its_worker_count_and_drawing_mode_to_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_renderer: Callable[..., hyperframes.RendererInstall],
    template: str,
    workers: str,
) -> None:
    composition = _composition(tmp_path / template, frames=3)
    monkeypatch.setattr(caption_stage, "prepare_layer_plan",
                        lambda plan, work_dir, assets=None: composition)
    monkeypatch.setattr(hyperframes, "ensure_installed", fake_renderer)
    plan = LayerPlan(template, 0, 0, 320, 180, 0, 3, {"fps": 30})
    out = tmp_path / f"{template}.mov"
    result = render_layer_plan(plan, tmp_path, out)
    args = json.loads((composition / "invoked.json").read_text())["args"]
    assert args[args.index("--workers") + 1] == workers
    assert result.checks["renderer"]["workers"] == ("auto" if template == "captions" else 1)
    assert ("--no-browser-gpu" in args) is (template == "captions")
    if template == "captions":
        assert result.checks["renderer"]["browser_gpu"] is False
    else:
        assert "browser_gpu" not in result.checks["renderer"]


def _alpha_frames(path: Path, plan: LayerPlan) -> np.ndarray:
    data = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "alphaextract,scale=108:48",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout_s=120 + plan.frames,
        binary=True,
    ).stdout
    return np.frombuffer(data, dtype=np.uint8).reshape(plan.frames, 48, 108)


def _rgba_frames(path: Path, plan: LayerPlan) -> np.ndarray:
    """Decode every frame without scaling or changing the frame cadence."""
    data = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-pix_fmt",
            "rgba",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout_s=120 + plan.frames,
        binary=True,
    ).stdout
    assert len(data) == plan.frames * plan.height * plan.width * 4
    return np.frombuffer(data, dtype=np.uint8).reshape(plan.frames, plan.height, plan.width, 4)


@pytest.mark.render
def test_real_caption_shrink_is_measured_without_runtime_warning(tmp_path: Path) -> None:
    style = _style(highlight="none", size_px=180, width_scale=1)
    long_phrase = replace(_phrases(style)[0], words=(
        CaptionWord("w0", "Write", 0.2, 0.35),
        CaptionWord("w1", "documentation", 0.35, 0.7),
    ), stress_id=None)
    fitting_phrase = replace(_phrases(style)[1], words=(
        CaptionWord("w2", "OK", 1.0, 1.2),
    ), stress_id=None)
    plan = caption_layer_plan([long_phrase, fitting_phrase], style, frames=60)
    result = render_layer_plan(plan, tmp_path, tmp_path / "captions.mov")
    shrink = caption_shrink(result.checks, plan)
    assert shrink["measured"] is True
    assert shrink["floor_px"] == 32 and shrink["style_size_px"] == 180
    assert len(shrink["phrases"]) == 1
    phrase = shrink["phrases"][0]
    assert phrase["text"] == "Write documentation"
    assert (phrase["start_s"], phrase["end_s"]) == (0.2, 0.7)
    assert phrase["fits"] is True and 32 <= phrase["size_px"] < 180
    assert result.checks is not None
    findings = result.checks["passes"]["runtime"]["findings"]
    shrink_findings = [item for item in findings if item["message"].startswith(_SHRINK_PREFIX)]
    assert len(shrink_findings) == 1 and shrink_findings[0]["severity"] == "info"
    assert not any(item["severity"] == "warning" for item in shrink_findings)


@pytest.mark.render
@pytest.mark.parametrize("highlight", ["slam", "weight_shift", "clip_wipe"])
def test_real_registry_caption_layer_has_exact_frames_and_alpha_in_band(
    tmp_path: Path, highlight: str,
) -> None:
    style = _style(highlight=highlight)
    plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
    with paths.operation():
        out = render_layer_plan(plan, tmp_path, tmp_path / "captions.mov").path
    info = ffmpeg.probe(out)
    assert (info.width, info.height, info.video_frames) == (plan.width, plan.height, 30)
    assert info.pix_fmt and info.pix_fmt.startswith("yuva")
    rgba = _rgba_frames(out, plan)
    for frame in (11, 17):
        alpha = rgba[frame, ..., 3]
        rows, columns = np.nonzero(alpha)
        assert rows.size > 0
        assert rows.min() > 0 and rows.max() < plan.height - 1
        assert columns.min() > 0 and columns.max() < plan.width - 1
        assert plan.y + int(rows.max()) < 1740
    assert rgba[0, ..., 3].max() == 0 and rgba[-1, ..., 3].max() == 0


@pytest.mark.render
def test_real_bottom_caption_line_box_and_upward_wrap(tmp_path: Path) -> None:
    style = _style(
        highlight="none", position="bottom", size_px=104, top_px=1503,
        shadow_opacity=0.25, shadow_blur_px=5, shadow_x_px=3, shadow_y_px=4,
        entry_rise_px=12, italic_stress=False,
    )
    bottoms = []
    for line_count, text in ((1, "First phrase"), (2, "First phrase second phrase")):
        phrases = build_phrases([
            CaptionWord(f"w{i}", word, 0.1 + i * 0.1, 0.18 + i * 0.1)
            for i, word in enumerate(text.split())
        ], style.data["captions"])
        assert len(phrases) == 1
        plan = caption_layer_plan([replace(phrases[0], end=0.9)], style, frames=30)
        with paths.operation():
            out = render_layer_plan(plan, tmp_path / str(line_count),
                                    tmp_path / f"caption-{line_count}.mov").path
        frame = _rgba_frames(out, plan)[15]
        rows = np.flatnonzero((frame[..., 3] > 0).any(axis=1))
        assert rows.size > 0
        if line_count == 1:
            assert 1503 + 5 <= plan.y + int(rows[0]) <= 1503 + 30
        glyph_rows = np.flatnonzero((frame[..., 3] > 200).any(axis=1))
        groups = np.split(glyph_rows, np.flatnonzero(np.diff(glyph_rows) > 1) + 1)
        assert len(groups) == line_count
        if line_count == 2:
            assert groups[1][0] - groups[0][0] == pytest.approx(104 * 1.12, abs=1)
        assert plan.y + int(rows[-1]) < 1740
        assert rows[0] > 0 and rows[-1] < plan.height - 1
        bottoms.append(plan.y + int(glyph_rows[-1]))
    assert bottoms[1] == pytest.approx(bottoms[0], abs=1)


@pytest.mark.render
def test_real_caption_alpha_matches_three_phrase_windows(tmp_path: Path) -> None:
    style = _style()
    plan = caption_layer_plan(_phrases(style), style, frames=84)
    out = render_layer_plan(plan, tmp_path, tmp_path / "captions.mov").path
    alpha = _alpha_frames(out, plan)
    for time in (0.5, 1.3, 2.1):
        assert alpha[round(time * 30)].max() > 0
    for time in (0.0, 0.1, 0.8, 0.9, 1.6, 1.7, 2.4, 2.6):
        assert alpha[round(time * 30)].max() == 0


@pytest.mark.render
def test_real_highlights_render_distinct_frames(tmp_path: Path) -> None:
    hashes = set()
    for highlight in ("moving_block", "underline", "color", "none"):
        style = _style(highlight=highlight)
        plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
        out = render_layer_plan(plan, tmp_path / highlight, tmp_path / (highlight + ".mov")).path
        pixels = ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-i",
                str(out),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "select=eq(n\\,16),format=rgba",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            timeout_s=120,
            binary=True,
        ).stdout
        assert pixels
        hashes.add(hashlib.sha256(pixels).hexdigest())
    assert len(hashes) == 4


@pytest.mark.render
@pytest.mark.parametrize("look", ["paper_title", "quiet_serif", "bold_band"])
def test_real_title_looks_have_alpha(tmp_path: Path, look: str) -> None:
    plan = title_layer_plan("Opening", "words", _style(look=look))
    with paths.operation():
        out = render_layer_plan(plan, tmp_path, tmp_path / "title.mov").path
    alpha = _alpha_frames(out, plan)
    assert alpha[30].max() > 0
    assert alpha[0].max() == 0


@pytest.mark.render
def test_real_project_caption_and_title_layers(tmp_path: Path) -> None:
    """Verify project boxes, exact frames and freeze-ready alpha with the pinned renderer."""
    from test_stage_captions import make_project, set_elements

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.captions import caption_snapshot, captions
    from yap_studio.stages.graphics import graphics, graphics_snapshot

    project = make_project(tmp_path)
    set_elements(project, {"id": "agent-title", "kind": "title", "start": {"time": 0},
                           "lines": ["Small", "steps"], "accent": "steps"})
    cap_result = captions(project)
    title_result = graphics(project)
    cap_plan = caption_snapshot(ProjectContext(project)).plan
    title_plan = graphics_snapshot(ProjectContext(project)).titles[0].plan
    for relative, plan in ((cap_result["out"], cap_plan),
                           (title_result["elements"][0]["path"], title_plan)):
        info = ffmpeg.probe(project / relative)
        assert (info.width, info.height, info.video_frames) == (
            plan.width, plan.height, plan.frames,
        )
        assert (info.color_space, info.color_transfer, info.color_primaries) == ("bt709",) * 3
        assert info.pix_fmt.startswith("yuva444p")
        alpha = _alpha_frames(project / relative, plan)
        assert alpha[15].max() > 0
        if plan.template == "captions":
            assert alpha[-1].max() > 0


@pytest.fixture(scope="module")
def render_caption(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[ResolvedStyle, int], tuple[LayerPlan, Path]]:
    """Cache movies for reuse between checks, without retaining decoded frames."""
    cached: dict[tuple[str, int], tuple[LayerPlan, Path]] = {}

    def render(style: ResolvedStyle, frames: int) -> tuple[LayerPlan, Path]:
        key = (json.dumps(style.data, sort_keys=True), frames)
        if key not in cached:
            plan = caption_layer_plan(_phrases(style), style, frames=frames)
            work = tmp_path_factory.mktemp("caption-render")
            with paths.operation():
                out = render_layer_plan(plan, work, work / "captions.mov").path
            cached[key] = (plan, out)
        return cached[key]

    return render


@pytest.mark.render
@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
def test_real_long_word_shrinks_inside_layer_in_every_style(
    tmp_path: Path, highlight: str,
) -> None:
    style = _style(highlight=highlight, size_px=180, italic_stress=False)
    plan = _word_plan(style, "Write documentation")
    with paths.operation():
        out = render_layer_plan(plan, tmp_path, tmp_path / "captions.mov").path
    # Documentation starts at 0.35 s; the entry rise and slam punch have both ended by 0.8 s.
    frame = _rgba_frames(out, plan)[round(0.8 * plan.variables["fps"])]
    ys, xs = np.nonzero(frame[..., 3] > 16)
    assert xs.size > 0, highlight
    left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    assert left > 0 and right < plan.width - 1, (highlight, left, right)
    assert top > 0 and bottom < plan.height - 1, (highlight, top, bottom)


@pytest.mark.render
@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
def test_real_fitting_phrases_are_frame_identical_without_shrink(
    tmp_path: Path,
    render_caption: Callable[[ResolvedStyle, int], tuple[LayerPlan, Path]],
    highlight: str,
) -> None:
    source = (_TEMPLATES / "captions" / "template.html").read_text()
    disabled = source.replace("const MIN_SIZE_PX = 32;", "const MIN_SIZE_PX = Infinity;")
    assert source.count("const MIN_SIZE_PX = 32;") == 1
    for size_px in (48, 180):
        style = _style(highlight=highlight, size_px=size_px)
        work = tmp_path / str(size_px)
        if size_px == 48:
            # Reuse the three-phrase movies from the existing renderer checks.
            plan, actual = render_caption(style, 84)
        else:
            plan = _word_plan(style, "Go do it now")
            with paths.operation():
                actual = render_layer_plan(plan, work / "actual", work / "actual.mov").path
        font = plan.variables["font_name"]
        assets = LayerAssets(
            html=disabled,
            metadata=(_TEMPLATES / "captions" / "template.json").read_bytes(),
            font_css=fonts.font_face_css([font]),
            font_files=tuple(
                (face.file, (fonts.FONTS_DIR / face.file).read_bytes())
                for face in fonts.BUNDLED[font].faces
            ),
        )
        with paths.operation():
            baseline = render_layer_plan(
                plan, work / "baseline", work / "baseline.mov", assets=assets,
            ).path
        actual_hashes = [
            hashlib.sha256(frame.tobytes()).hexdigest() for frame in _rgba_frames(actual, plan)
        ]
        baseline_hashes = [
            hashlib.sha256(frame.tobytes()).hexdigest() for frame in _rgba_frames(baseline, plan)
        ]
        assert actual_hashes == baseline_hashes, (highlight, size_px)


@pytest.mark.render
@pytest.mark.parametrize(
    ("highlight", "width_scale"),
    [
        pytest.param("pill", 1.0, id="pill"),
        pytest.param("gradient", 1.0, id="gradient"),
        pytest.param("editorial", 1.0, id="editorial"),
        pytest.param("moving_block", 0.8, id="moving_block-width-0.8"),
        pytest.param("underline", 0.8, id="underline-width-0.8"),
        pytest.param("pill", 0.8, id="pill-width-0.8"),
    ],
)
def test_real_caption_looks_are_deterministic_at_band_size(
    tmp_path: Path,
    render_caption: Callable[[ResolvedStyle, int], tuple[LayerPlan, Path]],
    highlight: str,
    width_scale: float,
) -> None:
    style = _style(highlight=highlight, width_scale=width_scale)
    plan, first = render_caption(style, 84)
    with paths.operation():
        second = render_layer_plan(plan, tmp_path, tmp_path / "captions.mov").path

    frame_hashes = []
    fps = plan.variables["fps"]
    for out in (first, second):
        info = ffmpeg.probe(out)
        assert info.width == plan.width == 1080
        assert info.height == plan.height
        assert info.video_frames == plan.frames
        assert info.pix_fmt and info.pix_fmt.startswith("yuva")
        rgba = _rgba_frames(out, plan)
        frame_hashes.append([hashlib.sha256(frame.tobytes()).hexdigest() for frame in rgba])
        alpha = rgba[..., 3]
        for time in (0.5, 1.3, 2.1):
            assert alpha[round(time * fps)].max() > 0, (out, time)
        for time in (0.0, 0.85, 1.65, 2.6):
            assert alpha[round(time * fps)].max() == 0, (out, time)
    assert frame_hashes[0] == frame_hashes[1]


@pytest.mark.render
def test_real_new_caption_looks_render_distinct_frames(
    render_caption: Callable[[ResolvedStyle, int], tuple[LayerPlan, Path]],
) -> None:
    hashes = {}
    for highlight in ("pill", "gradient", "editorial", "color", "none"):
        # Reuse the first determinism renders; baseline styles need only one phrase.
        frames = 84 if highlight in ("pill", "gradient", "editorial") else 30
        plan, out = render_caption(_style(highlight=highlight, width_scale=1.0), frames)
        rgba = _rgba_frames(out, plan)
        # The second word starts at 0.4 s and the entry rise ends at 0.3 s.
        frame = rgba[round(0.6 * plan.variables["fps"])]
        assert frame[..., 3].max() > 0, highlight
        hashes[highlight] = hashlib.sha256(frame.tobytes()).hexdigest()
    assert len(set(hashes.values())) == 5, hashes


@pytest.mark.render
def test_real_caption_width_scale_narrows_centered_glyph_boxes(
    render_caption: Callable[[ResolvedStyle, int], tuple[LayerPlan, Path]],
) -> None:
    boxes = []
    for width_scale in (1.0, 0.8):
        style = _style(
            highlight="none", width_scale=width_scale, shadow_opacity=0, italic_stress=False
        )
        plan, out = render_caption(style, 30)
        rgba = _rgba_frames(out, plan)
        # Measure after the entry rise, without a highlight or shadow enlarging the box.
        alpha = rgba[round(0.5 * plan.variables["fps"]), ..., 3]
        ys, xs = np.nonzero(alpha > 16)
        assert xs.size > 0, width_scale
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
        boxes.append((right - left + 1, bottom - top + 1, (left + right) / 2))

    (normal_width, normal_height, normal_center), (narrow_width, narrow_height, narrow_center) = (
        boxes
    )
    assert 0.70 <= narrow_width / normal_width <= 0.90, boxes
    assert abs(narrow_height - normal_height) <= 2, boxes
    assert abs(narrow_center - normal_center) <= 3, boxes


@pytest.mark.render
def test_real_gradient_wraps_hyphenated_word_and_fills_both_lines(tmp_path: Path) -> None:
    size_px = 120
    boxes = {}
    samples = {}
    for highlight in ("none", "gradient"):
        style = _style(
            highlight=highlight, size_px=size_px, shadow_opacity=0, italic_stress=False
        )
        phrase = build_phrases(
            [CaptionWord("w0", "twenty-first-century", 0.1, 0.6)], style.data["captions"]
        )[0]
        plan = caption_layer_plan([replace(phrase, end=0.9)], style, frames=30)
        with paths.operation():
            out = render_layer_plan(plan, tmp_path / highlight,
                                    tmp_path / (highlight + ".mov")).path
        frame = _rgba_frames(out, plan)[round(0.8 * plan.variables["fps"])]
        samples[highlight] = frame
        ys, xs = np.nonzero(frame[..., 3] > 16)
        assert xs.size > 0, highlight
        boxes[highlight] = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))

    none_box = boxes["none"]
    assert none_box[3] - none_box[2] + 1 > 1.5 * size_px, boxes
    left, right, top, bottom = boxes["gradient"]
    assert left > 0 and right < plan.width - 1, boxes
    assert top > 0 and bottom < plan.height - 1, boxes
    assert all(abs(a - b) <= 3 for a, b in zip(boxes["gradient"], none_box, strict=True)), boxes

    gradient = samples["gradient"]
    colored = (gradient[..., 3] > 200) & (
        gradient[..., 1].astype(np.int16) - gradient[..., 2].astype(np.int16) > 60
    )
    middle = (top + bottom + 1) // 2
    assert colored[top:middle, left : right + 1].any()
    assert colored[middle : bottom + 1, left : right + 1].any()


def _ease(x: float) -> float:
    """Evaluate the template's cubic-bezier(.2,.7,.2,1) at progress x."""
    low, high = 0.0, 1.0
    for _ in range(60):
        t = (low + high) / 2
        if 3 * (1 - t) ** 2 * t * 0.2 + 3 * (1 - t) * t * t * 0.2 + t**3 < x:
            low = t
        else:
            high = t
    t = (low + high) / 2
    return 3 * (1 - t) ** 2 * t * 0.7 + 3 * (1 - t) * t * t + t**3


def _underline_center(frame: np.ndarray) -> float:
    """Return the alpha-weighted horizontal center of the red underline."""
    red = (frame[..., 0].astype(np.int16) - frame[..., 1].astype(np.int16) > 100) & (
        frame[..., 3] > 0
    )
    cover = np.where(red, frame[..., 3], 0).max(axis=0).astype(float)
    assert cover.max() > 128
    return float((cover * np.arange(frame.shape[1])).sum() / cover.sum())


@pytest.mark.render
def test_real_long_caption_band_is_frame_exact_and_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every in-motion frame sits at its own frame time at both worker counts.

    Frames 30, 60 ... 210 are where worker chunks start. HyperFrames 0.8.145 to
    0.8.150 drew such a frame late. 0.8.151 does not, so the band may render in parallel.
    """
    style = _style(highlight="underline", highlight_color="#FF2020", max_words_per_phrase=3,
                   italic_stress=False, shadow_opacity=0)
    words = []
    for second in range(1, 8):
        for index, (text, start) in enumerate(
            (("one", second - 0.4), ("two", second - 0.02), ("six", second + 0.25))
        ):
            words.append(CaptionWord(f"w{second}{index}", text, start, start + 0.2))
    phrases = build_phrases(words, style.data["captions"])
    assert [len(phrase.words) for phrase in phrases] == [3] * 7
    phrases = [replace(phrase, end=phrase.start + 0.85) for phrase in phrases]
    plan = caption_layer_plan(phrases, style, frames=240)
    fps = plan.variables["fps"]
    movies = []
    for name in ("a", "b"):
        if name == "b":
            monkeypatch.setattr(hyperframes, "CAPTION_WORKERS", 4)
        with paths.operation():
            movies.append(render_layer_plan(plan, tmp_path / name, tmp_path / f"{name}.mov").path)
    first, second = (_rgba_frames(movie, plan) for movie in movies)
    for index, (one, two) in enumerate(zip(first, second, strict=True)):
        assert np.array_equal(one, two), f"frame {index} differs between renders"
    checked = 0
    for phrase in phrases:
        for word in phrase.words[1:]:
            start, end = word.start, word.start + 0.16
            rest, moved = math.floor(start * fps), math.ceil(end * fps)
            before, after = _underline_center(first[rest]), _underline_center(first[moved])
            assert after - before > 20
            for frame in range(rest + 1, moved):
                expected = before + (after - before) * _ease((frame / fps - start) / 0.16)
                actual = _underline_center(first[frame])
                assert abs(actual - expected) <= 2, (frame, actual, expected)
                checked += 1
    assert checked >= 50
