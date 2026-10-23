"""Prepare the render's own compositions so a browser can play them live.

Each layer is planned and prepared by the same functions a render uses, so the
HTML, fonts and variables match the render byte for byte. The browser gets the
variables and the pinned HyperFrames runtime injected at serve time; nothing is
written into the composition folder, which stays ready to render.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from yap_studio import cards, paths
from yap_studio.brand_words import list_words
from yap_studio.captions import (
    CaptionWord,
    build_phrases,
    caption_layout_settings,
    kinetic_word_times,
)
from yap_studio.errors import DependencyError, ValidationError
from yap_studio.faces.model import Box
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.hf_registry import catalog as registry_catalog
from yap_studio.render.hyperframes import RendererInstall
from yap_studio.stages.captions import (
    LayerPlan,
    caption_band_geometry,
    caption_layer_plan,
    caption_offsets,
    chapter_layer_plan,
    prepare_layer_plan,
    title_layer_plan,
)
from yap_studio.style import ResolvedStyle

SAMPLE_SENTENCE = "Most people think this takes hours. It takes about five minutes."
FPS = 30
_TAIL_S = 1.0
# Match riseWindow in templates/captions/template.html.
_RISE_FRAMES = 8
_RISE_STAGGER_FRAMES = 2
_RISE_MIN_FRAMES = 1


@dataclass(frozen=True)
class LiveLayer:
    """A prepared composition, its box on the 1080x1920 frame and its variables."""

    name: str
    template: str
    composition: Path
    x: int
    y: int
    width: int
    height: int
    frames: int
    fps: int
    variables: dict


def sample_words(text: str, *, start_s: float = 0.3) -> list[CaptionWord]:
    """Time words at an even speaking pace until real word timings are available."""
    tokens = text.split()
    if not tokens:
        raise ValidationError("words", "use at least one word")
    words, at = [], start_s
    for index, token in enumerate(tokens):
        length = 0.18 + 0.03 * len(token)
        words.append(CaptionWord(f"w{index}", token, round(at, 3), round(at + length, 3)))
        at += length + (0.3 if token[-1] in ".!?" else 0.06)
    return words


def caption_plan(
    style: ResolvedStyle, words: Sequence[CaptionWord], *,
    face_boxes: Sequence[Box] | None = None,
) -> LayerPlan:
    """Plan captions for the words, ending one second after the last word."""
    frames = round((words[-1].end + _TAIL_S) * FPS)
    settings = caption_layout_settings(style.data["captions"], style.data["brand"])
    phrases = build_phrases(words, settings)
    plan = caption_layer_plan(phrases, style, frames=frames, fps=FPS, caption_settings=settings)
    if face_boxes is None or style.data["captions"]["placement"] == "fixed":
        return plan
    geometry = caption_band_geometry(
        settings, safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
    )
    placements = caption_offsets(json.loads(plan.variables["phrases"]), geometry, None, style,
                                 fps=FPS, still_faces=face_boxes, caption_settings=settings)
    return caption_layer_plan(phrases, style, frames=frames, fps=FPS,
                              offsets=[row["offset_px"] for row in placements],
                              caption_settings=settings)


def _title_text(words: Sequence[CaptionWord]) -> tuple[str, str]:
    first = []
    for word in words:
        first.append(word.text.rstrip(".,!?;:"))
        if word.text[-1] in ".!?" or len(first) == 5:
            break
    split = max(1, len(first) // 2)
    return " ".join(first[:split]), " ".join(first[split:])


def card_slots(kind: str) -> dict[str, object]:
    """Return the template's own preview slots, the sample a gallery tile shows."""
    if registry_cards.is_kind(kind):
        return registry_cards.preview_slots(kind)
    source = cards.template_dir(kind) / "preview.json"
    return json.loads(source.read_text(encoding="utf-8"))["slots"]


def leading_words(words: Sequence[CaptionWord], limit: int) -> list[str]:
    """Take whole preview words within the limit, through at most one sentence."""
    tokens: list[str] = []
    for word in words:
        if len(" ".join([*tokens, word.text])) > limit:
            break
        tokens.append(word.text)
        if word.text.endswith((".", "!", "?")):
            break
    return tokens


def registry_card_slots(kind: str, words: Sequence[CaptionWord]) -> dict[str, object]:
    """Use preview words for Words registry slots, keeping other catalog samples."""
    slots = card_slots(kind)
    if not registry_cards.is_kind(kind):
        return slots
    entry = registry_catalog.item(registry_cards.item_name(kind))
    if entry["kind"]["category"] == "Words":
        for name, rule in entry["slots"].items():
            if rule["type"] != "string":
                continue
            text = " ".join(leading_words(words, rule["max_length"]))
            if len(text) >= max(1, rule.get("min_length", 0)):
                slots[name] = text
    return slots


def sentence_card_text(words: Sequence[CaptionWord], style: ResolvedStyle) -> str | None:
    """Shorten at word boundaries until the engine fits this style's sign look."""
    limit = cards.load_template("text_tag")["slots"]["text"]["max_length"]
    tokens = leading_words(words, limit)
    while tokens:
        text = " ".join(tokens)
        try:
            cards.card_layer_plan("text_tag", {"text": text, "variant": "sign"}, style, fps=FPS)
        except ValidationError as exc:
            if exc.parameter != "slots.text" or "shorten" not in exc.detail:
                raise
            tokens.pop()
        else:
            return text
    return None


def poster_time(layer: LiveLayer) -> float:
    """Choose a held, visible frame instead of the transparent entrance at zero."""
    duration = layer.frames / layer.fps
    if layer.name == "captions":
        entry = layer.variables["entry_ms"] / 1000
        slam = layer.variables["highlight_style"] == "slam"
        rise = layer.variables.get("entrance") == "rise"
        spoken = layer.variables.get("entrance") == "spoken"
        phrases = json.loads(layer.variables["phrases"])
        for phrase in phrases:
            entered = min(phrase["end"], phrase["start"] + (0 if spoken else entry))
            for index, word in enumerate(phrase["words"]):
                start = max(entered, word["start"] + (0.12 if slam and not spoken else 0))
                if spoken:
                    start = max(phrase["start"], word["start"]) + 2 / layer.fps
                if rise:
                    start = max(start, _rise_end(phrase, index, fps=layer.fps, slam=slam))
                end = min(phrase["end"], word["end"])
                if end - start >= 1 / layer.fps:
                    return (start + end) / 2
        # Extremely short phrases can spend their entire duration entering.
        longest = max(phrases, key=lambda phrase: phrase["end"] - phrase["start"])
        return max(longest["start"], longest["end"] - 0.001)
    if layer.name in {"title", "chapter"}:
        return duration / 2
    return min(1.5, duration - min(0.3, duration * 0.15) - 1 / layer.fps)


def _rise_end(phrase: Mapping, index: int, *, fps: int, slam: bool) -> float:
    """Wait for the chosen word's move using the template's frame timing."""
    frames = max(_RISE_MIN_FRAMES, (phrase["end"] - phrase["start"]) * fps)
    length = min(_RISE_FRAMES, frames)
    stagger = min(_RISE_STAGGER_FRAMES, (frames - length) / max(1, len(phrase["words"]) - 1))
    start = phrase["start"] * fps + index * stagger
    if slam:
        word_start = min(phrase["end"], max(phrase["start"], phrase["words"][index]["start"]))
        next_start = (phrase["words"][index + 1]["start"]
                      if index + 1 < len(phrase["words"]) else phrase["end"])
        next_start = min(phrase["end"], max(word_start, next_start))
        start = word_start * fps
        length = max(_RISE_MIN_FRAMES, min(length, next_start * fps - start))
    return (start + length) / fps


def live_layers(
    style: ResolvedStyle, words: Sequence[CaptionWord], work_dir: Path, *,
    title: tuple[str, str] | None = None, card: str | None = None,
    slots: Mapping[str, object] | None = None,
    only: str | None = None, show_title: bool = True, show_chapter: bool = False,
    face_boxes: Sequence[Box] | None = None,
    sample_image: Path | None = None,
) -> list[LiveLayer]:
    """Prepare captions, the opening title (unless the style has none) and one card."""
    if not words:
        raise ValidationError("words", "use at least one word")
    plans: list[tuple[str, LayerPlan | cards.CardPlan]] = []
    chapter = (only == "chapter" or show_chapter) and style.data["chapters"]["style"] != "none"
    if chapter:
        plans.append(("chapter", chapter_layer_plan("Sample chapter", 2, style,
                                                   start_frame=84 * FPS)))
    if only in (None, "captions") or (only == "chapter" and not chapter):
        plans.append(("captions", caption_plan(style, words,
                                              face_boxes=None if chapter else face_boxes)))
    if show_title and only in (None, "title") and style.data["hook"]["style"] != "none":
        plain, key = title if title is not None else _title_text(words)
        hook_frames = None
        if style.data["hook"]["style"] == "text_hook":
            from yap_studio.text_hook import fit_text_hook

            tokens = [word.text for word in words[:8]]
            plain, key = "Make your point clear", ""
            for count in range(len(tokens), 1, -1):
                candidate = " ".join(tokens[:count])
                try:
                    fit_text_hook(candidate)
                except ValidationError:
                    continue
                plain = candidate
                break
            hook_frames = round((words[-1].end + _TAIL_S) * FPS)
        word_times = None
        if style.data["hook"]["style"] == "kinetic":
            try:
                word_times = kinetic_word_times(
                    [plain, key], words, 0, style.data["hook"]["duration_s"], {},
                    vocabulary=tuple(list_words()),
                )
            except ValidationError:
                # Short preview windows still show every word using even spacing.
                pass
        plans.append(("title", title_layer_plan(plain, key, style, fps=FPS,
                                                word_times=word_times, frames=hook_frames)))
    if only in (None, "card") and card is not None:
        values = dict(slots) if slots is not None else card_slots(card)
        if card == "file_drop" and "clip" not in values:
            from yap_studio.templates_command import placeholder_clip

            clip = work_dir / "sample.mp4"
            paths.validate_output_dir(work_dir)
            placeholder_clip(clip, sample_image)
            values["clip"] = clip
        frames = cards.default_preview_frames(card, values, style, fps=FPS)
        values = cards.preview_videos(card, values, work_dir, frames=frames, fps=FPS)
        plan = cards.card_layer_plan(card, values, style, frames=frames, fps=FPS)
        if plan.template == "file_drop":
            plan = replace(plan, variables={**plan.variables, "play_window": "{}"})
        if "window_fill" in plan.variables:
            plan = replace(plan, variables={**plan.variables, "window_fill": "preview"})
        if card == "reel_feed":
            plans.insert(0, ("card", plan))
        else:
            plans.append(("card", plan))
    layers = []
    for name, plan in plans:
        folder = work_dir / name
        if isinstance(plan, cards.CardPlan):
            if plan.template == "storyboard" and sample_image is not None:
                thumbnails = json.loads(plan.variables["thumbnails"])
                for index, thumbnail in enumerate(thumbnails):
                    thumbnail["src"] = f"shot-{index + 1:02}.jpg"
                plan = replace(plan, variables={**plan.variables,
                                                "thumbnails": json.dumps(thumbnails)},
                               assets=tuple((item["src"], sample_image) for item in thumbnails))
            composition, fps = cards.prepare_card(plan, folder), plan.fps
        else:
            composition, fps = prepare_layer_plan(plan, folder), int(plan.variables["fps"])
        box = (plan.layer_box() if isinstance(plan, cards.CardPlan) else
               {"x": plan.x, "y": plan.y, "width": plan.width, "height": plan.height})
        layers.append(LiveLayer(
            name, plan.template, composition, box["x"], box["y"], box["width"], box["height"],
            plan.frames, fps, {} if registry_cards.is_kind(plan.template)
            or isinstance(plan, cards.CardPlan) and plan.motion else dict(plan.variables),
        ))
    return layers


def live_html(source: str, variables: Mapping[str, object], runtime_url: str) -> str:
    """Put the variables and the runtime first in <head>, before any template script."""
    head = source.find("<head>")
    if head < 0:
        raise ValidationError("template", "the composition has no <head>")
    payload = json.dumps(dict(variables), ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace(
        "\u2029", "\\u2029")
    inject = (f"<script>window.__hfVariables={payload};</script>"
              f'<script src="{runtime_url}"></script>')
    at = head + len("<head>")
    return source[:at] + inject + source[at:]


def runtime_file(install: RendererInstall) -> Path:
    """Return the browser runtime shipped with the pinned renderer."""
    runtime = (install.prefix / "node_modules" / "hyperframes" / "dist"
               / "hyperframe.runtime.iife.js")
    if not runtime.is_file():
        raise DependencyError(
            f"The HyperFrames browser runtime is missing from {install.prefix}.",
            error_type="dependency_error",
            code="renderer_runtime_missing",
            suggested_action=(
                f"Delete the renderer cache folder {install.prefix} and run the command again "
                "so the pinned renderer reinstalls."
            ),
        )
    return runtime
