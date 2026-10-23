"""Validate typed card slots and prepare local, style-driven compositions."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg, paths, receipt
from yap_studio.card_project import CardProject, sample_project
from yap_studio.card_stills import CardPoster, cached_frame
from yap_studio.cards_video import (
    GRID_CHIP_HEIGHT,
    GRID_PAD_BOTTOM,
    GRID_PAD_TOP,
    GRID_RING,
    caption_ceiling,
    feed_window,
    grid_layout,
    tile_window,
    window_rect,
)
from yap_studio.errors import InputError, ProcessingError, ValidationError
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.planner.number_behind import MIN_DURATION_S, number_plan
from yap_studio.render import fonts, hyperframes
from yap_studio.render.hyperframes import LayerResult, RendererInstall
from yap_studio.render.text_metrics import text_width_em, widest_digit_em
from yap_studio.sfx import bleep
from yap_studio.style import ResolvedStyle

CARD_TEMPLATES: tuple[str, ...] = (
    "terminal", "stat", "checklist", "text_tag", "image_card", "transfer", "image_compare", "steps",
    "flow", "chart", "quote", "progress_stat", "specs_list", "callout_circle", "name_tag",
    "success_check",
    "edit_timeline",
    "storyboard",
    "jiggle_delete",
    "file_drop",
    "video_card",
    "prompt_bar",
    "comment_cta",
    "comment_drop",
    "icon_strike",
    "big_number",
    "playing_grid", "reel_feed",
    "gravity_drop",
    "choices",
)
# Kinds a creator must enable in graphics.card_kinds_on. Add every kind after the first
# sixteen here and in CARD_TEMPLATES so it stays off for styles saved before it existed.
OPT_IN_KINDS: tuple[str, ...] = (
    "edit_timeline",
    "storyboard",
    "jiggle_delete",
    "file_drop",
    "video_card",
    "prompt_bar", "comment_cta", "comment_drop", "icon_strike",
    "big_number",
    "playing_grid", "reel_feed",
    "gravity_drop",
    "choices",
)


def card_kind_off(style: ResolvedStyle | Mapping[str, Any], kind: str) -> bool:
    """Return whether the creator's style has this card kind off."""
    data = style.data if isinstance(style, ResolvedStyle) else style
    graphics = data.get("graphics", {})
    if kind in OPT_IN_KINDS or registry_cards.is_kind(kind):
        return kind not in graphics.get("card_kinds_on", [])
    return kind in graphics.get("card_kinds_off", [])


def rows_on_cue(style: ResolvedStyle | Mapping[str, Any], template: str,
                slots: Mapping[str, object]) -> bool:
    """Return whether this card uses the row timing and extended hold."""
    data = style.data if isinstance(style, ResolvedStyle) else style
    return (data.get("graphics", {}).get("specs_rows", "together") != "together"
            and (template == "specs_list" or template == "stat" and bool(slots.get("rows"))))


def card_kind_off_detail(style: ResolvedStyle, kind: str) -> str:
    """Explain a refused card and suggest the kinds the creator allows."""
    from yap_studio.style import cards_on

    available = ", ".join(cards_on(style))
    suggestion = f"use one of {available} or no card" if available else "use no card"
    action = ("has not turned on" if kind in OPT_IN_KINDS or registry_cards.is_kind(kind)
              else "turned off")
    return f"the creator {action} {kind} cards; {suggestion}"


MIN_TEXT_PX = 30
EDIT_TIMELINE_MIN_S = 1.5
STORYBOARD_MIN_S = 1.5
JIGGLE_DELETE_MIN_S = 2.0
FILE_DROP_MIN_S = 1.5
GRAVITY_MIN_S = 1.2
GRID_PREVIEW_MIN_S = 1.5
PAD_X = 40
PAD_TOP = 24
PAD_BOTTOM = 64
MONO_FONT = "Space Mono"
TERMINAL_MIN_S = 1.5
PROMPT_BAR_MIN_S = 2.0
COMMENT_CTA_MIN_S = 2.0
COMMENT_DROP_MIN_S = 1.6
ICON_STRIKE_MIN_S = 1.7
CHOICES_MIN_S = 2.0
CHOICES_LAND_S = 5 / 30
CHOICES_HOLD_S = 0.5
CHOICES_SETTLE_S = 0.2
CHOICES_REST_S = 0.3
CHOICES_OUT_S = 0.1
CHOICES_SCALE = 1.6
CHOICES_MIN_GAP_S = CHOICES_LAND_S + CHOICES_SETTLE_S
CHOICES_MIN_TEXT_PX = 56
CHOICES_MAX_TEXT_PX = 64
CHOICES_LINE_HEIGHT = 1.1
CHOICES_ROW_GAP_PX = 4
TERMINAL_PRINT_S = 0.12
TERMINAL_WORD_IN_S = 0.3
TERMINAL_WORD_TAIL_S = 0.5
TERMINAL_BODY_PAD_Y = 20
TERMINAL_MONO_ADVANCE_EM = 0.612
CHECKLIST_REVEAL_S = 0.3
STEPS_MIN_S = 2.0
STEPS_PANEL_IN_S = 0.3
STEPS_CHIP_IN_S = 0.24
STEPS_STAGGER_S = 0.06
STEPS_REST_S = 0.6
STEPS_MAX_TEXT_PX = 36
STEPS_MONO_ADVANCE_EM = 0.62
STEPS_TRACKING_EM = 0.14
STEPS_LINE_HEIGHT = 1.2
STEPS_CHIP_PAD_X = 18
STEPS_CHIP_PAD_Y = 14
STEPS_CHIP_BORDER = 1
STEPS_CHIP_GAP = 12
FLOW_MIN_S = 2.1
FLOW_PANEL_IN_S = 0.3
FLOW_ARROW_DRAW_S = 0.24
FLOW_BOX_DRAW_S = 0.45
FLOW_LABEL_IN_S = 0.2
FLOW_NODE_REVEAL_S = FLOW_BOX_DRAW_S + FLOW_LABEL_IN_S
FLOW_REST_S = 0.6
FLOW_EXPLICIT_REST_S = 0.3
FLOW_MAX_TEXT_PX = 40
FLOW_ARROW_GAP = 72
FLOW_BOX_HEIGHT = 120
FLOW_LABEL_PAD_X = 18
FLOW_LABEL_PAD_Y = 18
FLOW_LINE_HEIGHT = 1.2
FLOW_STROKE_INSET = 6
FLOW_TAGS_MIN_S = 1.5
FLOW_TAGS_POP_S = 5 / 30
FLOW_TAGS_ARROW_S = 0.2
FLOW_TAGS_OUT_S = 0.1
FLOW_TAGS_REST_S = 0.3
FLOW_TAGS_ARROW_PX = 36
FLOW_TAGS_MIN_TEXT_PX = 52
FLOW_TAGS_MAX_TEXT_PX = 60
FLOW_TAGS_PAD_X_EM = 0.42
FLOW_TAGS_PAD_Y_EM = 0.23
FLOW_TAGS_LINE_HEIGHT = 1
FLOW_TAGS_SCALE = 1.16
COMPARE_MIN_S = 2.2
COMPARE_WIPE_S = 1.05
COMPARE_REST_S = 0.3
COMPARE_MAT = 8
COMPARE_CHIP_INSET = 12
COMPARE_CHIP_PAD = 12
COMPARE_CHIP_BORDER = 2
COMPARE_DIVIDER_KNOB = 32
COMPARE_CHIP_GAP = 4
TRANSFER_MIN_S = 2.3
TRANSFER_TRAVEL_S = 1.0
TRANSFER_LAND_S = 0.3
TRANSFER_COLUMN_SHARE = 3 / 8
TRANSFER_GEOMETRY = {
    "panel_inner_pad_px": 24,
    "device_width_px": 150,
    "device_height_px": 110,
    "device_label_gap_px": 10,
    "screen_height_px": 76,
    "laptop_screen_width_px": 120,
    "laptop_screen_top_px": 14,
    "desktop_screen_width_px": 136,
    "desktop_screen_top_px": 7,
    "chip_icon_px": 30,
    "chip_gap_px": 8,
    "chip_pad_x_px": 8,
    "chip_pad_y_px": 12,
}

# Conservative glyph widths and CSS line heights, independent of machine fonts.
PROPORTIONAL_EM = 0.6
KICKER_EM = 0.74
KICKER_TRACKING_EM = 0.14
MONO_EM = 0.62
DIGIT_EM = 0.64
STAT_SLOT_SAFETY_EM = 0.02
CARET_EM = 0.63
FIELD_CARET_MIN_PX = 3
FIELD_CARET_EM = 0.06
TERMINAL_LINE_HEIGHT = 1.35
BAR_LINE_HEIGHT = 1.2
CHECKLIST_LINE_HEIGHT = 1.3
TAG_LINE_HEIGHT = 1.06
SUBTEXT_LINE_HEIGHT = 1.2
# Left, right, top, bottom padding, matching each lower third's CSS.
TAG_PADDING = {
    "bar": (24, 24, 24, 24),
    "card": (32, 38, 24, 26),
    "block": (32, 36, 20, 24),
    "pill": (40, 64, 20, 20),
    "underline": (24, 24, 24, 24),
    "rule": (26, 32, 24, 24),
    "title": (32, 32, 32, 32),
}
# The pill reserves an 18 px dot, 22 px gap, and 32 px for its scale overshoot.
TAG_WIDTH_RESERVE = {
    "bar": 12, "card": 0, "block": 0, "pill": 72, "underline": 0, "rule": 8, "title": 0,
}
TAG_GAP = {
    "bar": 7, "card": 12, "block": 8, "pill": 4, "underline": 14, "rule": 12, "title": 18,
}
TAG_RULE_HEIGHT = {
    "bar": 0, "card": 4, "block": 0, "pill": 0, "underline": 6, "rule": 0, "title": 0,
}
TAG_UPPERCASE_EM = 0.74
TAG_TRACKING_EM = {"block": 0.01, "underline": 0.005, "rule": 0.01}
TAG_SUB_TRACKING_EM = {"block": 0.05, "underline": 0.04, "rule": 0.04, "title": 0.18}
TAG_SIGN_LOOKS = {"paper": "bar", "minimal": "bar", "dark": "card", "bright": "block"}
CONTENT_PAD = 24
BAR_HEIGHT = math.ceil(MIN_TEXT_PX * BAR_LINE_HEIGHT + 24)
# Reserve the frame outline on both sides of the image and titlebar.
IMAGE_FRAME_BORDER = 2
COLLAGE_MIN_S = 2.2
COLLAGE_SPAN_S = 1.25
COLLAGE_EXIT_S = 0.45
COLLAGE_REST_S = 0.3
_TEMPLATES = Path(__file__).parent / "templates"
_LOOKS = {
    "paper": (30, "0 12px 22px rgb(18 12 6 / 0.28), 0 3px 8px rgb(18 12 6 / 0.16)"),
    "minimal": (
        16,
        "0 0 0 2px color-mix(in srgb, {ink} 12%, transparent), 0 8px 18px rgb(0 0 0 / 0.16)",
    ),
    "dark": (30, "0 12px 22px rgb(0 0 0 / 0.5), 0 0 0 2px rgb(255 255 255 / 0.06)"),
    "bright": (36, "0 12px 22px rgb(0 0 0 / 0.26)"),
}


def _minimum_duration_s(template: str, slots: Mapping[str, object],
                        graphics: Mapping[str, Any] | None = None) -> float | None:
    if registry_cards.is_kind(template):
        return registry_cards.minimum_duration_s(template)
    if template == "reel_feed" and slots.get("ending", "handoff") == "handoff":
        return 2.5
    if template == "image_card" and slots.get("reveal") == "collage":
        return COLLAGE_MIN_S
    if template == "flow" and (graphics or {}).get("flow_look", "boxes") == "tags":
        return FLOW_TAGS_MIN_S
    return {
        "terminal": TERMINAL_MIN_S,
        "transfer": TRANSFER_MIN_S,
        "image_compare": COMPARE_MIN_S,
        "steps": STEPS_MIN_S,
        "flow": FLOW_MIN_S,
        "edit_timeline": EDIT_TIMELINE_MIN_S,
        "storyboard": STORYBOARD_MIN_S,
        "jiggle_delete": JIGGLE_DELETE_MIN_S,
        "file_drop": FILE_DROP_MIN_S,
        "video_card": 1.5,
        "prompt_bar": PROMPT_BAR_MIN_S,
        "comment_cta": COMMENT_CTA_MIN_S,
        "comment_drop": COMMENT_DROP_MIN_S,
        "icon_strike": ICON_STRIKE_MIN_S,
        "big_number": MIN_DURATION_S,
        "reel_feed": 2.0,
        "gravity_drop": GRAVITY_MIN_S,
        "choices": CHOICES_MIN_S,
    }.get(template)


def default_preview_frames(template: str, slots: Mapping[str, object], style: ResolvedStyle,
                           *, fps: int = 30) -> int:
    """Use the style hold with enough time for the card's default preview."""
    minimum_s = _minimum_duration_s(template, slots, style.data["graphics"]) or 0
    if template == "playing_grid":
        minimum_s = GRID_PREVIEW_MIN_S
    return max(1, round(style.data["graphics"]["hold_s"] * fps), math.ceil(minimum_s * fps))


@dataclass(frozen=True)
class CardPlan:
    """A validated layer box, frame window, variables, and local assets."""

    template: str
    x: int
    y: int
    width: int
    height: int
    frames: int
    fps: int
    variables: dict[str, object]
    fonts: tuple[str, ...]
    assets: tuple[tuple[str, Path], ...] = ()
    poster: CardPoster | None = None
    windows: tuple[dict, ...] = ()
    registry: dict | None = None
    motion: dict | None = None

    def layer_box(self) -> dict[str, int]:
        """Keep the placement box separate from the moving card's band layer."""
        return {"x": 0 if self.motion else self.x, "y": self.y,
                "width": 1080 if self.motion else self.width, "height": self.height}


def template_dir(template: str) -> Path:
    """Return a known card's vendored template directory."""
    if registry_cards.is_kind(template):
        from yap_studio.hf_registry.install import item_files

        return item_files(registry_cards.item_name(template))
    if template not in CARD_TEMPLATES:
        raise ValidationError("template", "choose " + ", ".join(CARD_TEMPLATES))
    return _TEMPLATES / template


def template_html(template: str, *, read: Callable[[Path], bytes] = Path.read_bytes,
                  motion: bool = False) -> str:
    """Expand shared card code before identifying or preparing the composition."""
    source = read(template_dir(template) / "template.html").decode("utf-8")
    if "/* CARD_GLOW_SHADOW */" in source:
        source = source.replace("/* CARD_GLOW_SHADOW */",
                                read(_TEMPLATES / "glow-shadow.js").decode("utf-8"))
    if motion:
        from yap_studio.card_motion import drawn_template

        source = drawn_template(source, template)
    return source


def load_template(template: str) -> dict:
    """Read the card's variable declarations and authoritative slot limits."""
    if registry_cards.is_kind(template):
        return registry_cards.template_metadata(template)
    return json.loads((template_dir(template) / "template.json").read_text(encoding="utf-8"))


def video_slots(template: str) -> dict[str, dict]:
    """Return the template's video source declarations."""
    return {name: rule for name, rule in load_template(template)["slots"].items()
            if rule["type"] in {"video", "video_list"} and "sources" in rule}


def preview_clip(work: Path, *, frames: int = 120, fps: int = 30) -> dict:
    """Make a local synthetic source for geometry-only card previews."""
    source = work / "preview-clip.mp4"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color=c=gray:s=320x180:r={fps}", "-frames:v", str(frames),
                "-c:v", "libx264", "-preset", "ultrafast", str(source)], timeout_s=30)
    return {"path": str(source), "sha256": receipt.sha256_file(source)}


def preview_videos(template: str, slots: dict, work: Path, *, frames: int = 120,
                   fps: int = 30) -> dict:
    """Fill omitted media slots for local template previews, preserving self defaults."""
    values = dict(slots)
    clip = None
    for name, rule in video_slots(template).items():
        if name in values or rule.get("default") == "self":
            continue
        if clip is None:
            preview_frames = max(120, frames)
            preview_work = work / f"video-{fps}-{preview_frames}"
            preview_work.mkdir(parents=True, exist_ok=True)
            source = preview_work / "preview-clip.mp4"
            clip = ({"path": str(source), "sha256": receipt.sha256_file(source)}
                    if source.exists() else preview_clip(preview_work, frames=preview_frames,
                                                        fps=fps))
        count = min(rule.get("max_items", 6),
                    max(rule.get("min_items", 1), len(values.get("labels", []))))
        values[name] = [dict(clip) for _ in range(count)] if rule["type"] == "video_list" else clip
    return values


def video_info(source: Path) -> ffmpeg.MediaInfo:
    """Require a moving picture with known duration and positive display dimensions."""
    info = ffmpeg.probe(source, count_frames=False)
    still = source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"} or any(
        name in {"image2", "image2pipe"} or name.endswith("_pipe")
        for name in (info.format_name or "").split(",")
    )
    duration = info.video_duration_s
    if (still or not info.has_video or not info.width or not info.height
            or info.width < 1 or info.height < 1 or duration is None
            or not math.isfinite(duration) or duration <= 0):
        raise ValidationError("clip", "use a video clip with known duration and positive size; "
                              "still images cannot fill video windows")
    return info


def _display_aspect(source: Path, info: ffmpeg.MediaInfo) -> float:
    result = ffmpeg.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=display_aspect_ratio", "-of", "json", str(source)], timeout_s=10)
    raw = json.loads(result.stdout)["streams"][0].get("display_aspect_ratio", "")
    try:
        ratio = float(Fraction(raw.replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        return info.width / info.height
    if ratio <= 0:
        return info.width / info.height
    return 1 / ratio if info.rotation in {90, 270} else ratio


def clip_aspect(clip: dict, *, minimum: float = 3 / 4) -> float:
    """Clamp the cropped display aspect to the card's supported range."""
    source = Path(clip["path"])
    info = video_info(source)
    ratio = _display_aspect(source, info)
    if "crop" in clip:
        # Keep the displayed pixel aspect when cropping an anamorphic clip.
        ratio *= clip["crop"][2] / clip["crop"][3] / (info.width / info.height)
    return min(16 / 9, max(minimum, ratio))


def grid_clip_window(index: int, frames: int, slots: Mapping, *, fps: int = 30) -> tuple[int, int]:
    """Return a grid clip's staggered start and playback length."""
    rule = load_template("playing_grid")["slots"]["stagger_s"]
    stagger = _number("stagger_s", slots.get("stagger_s", rule["default"]), rule)
    start = round(index * stagger * fps)
    return start, frames - start


def _clip_bounds(name: str, clip: dict | str, frames: int, fps: int) -> None:
    if clip == "self":
        return
    available = video_info(Path(clip["path"])).video_duration_s
    if clip["in_s"] + frames / fps * clip["speed"] > available + 1e-9:
        latest = max(0, available - frames / fps * clip["speed"])
        raise ValidationError(f"slots.{name}", f"the clip ends at {available:g} s; "
                              f"use in_s at most {latest:g} or a shorter hold")


def _feed_lines(text: str, width: int, family: str) -> tuple[int, list[str]]:
    for size in range(36, MIN_TEXT_PX - 1, -1):
        lines = []
        line = ""
        for word in text.split():
            if text_width_em(word, family, 600) * size > width:
                break
            candidate = f"{line} {word}" if line else word
            if text_width_em(candidate, family, 600) * size <= width:
                line = candidate
            else:
                lines.append(line)
                line = word
        else:
            if line:
                lines.append(line)
            if len(lines) <= 2:
                return size, lines
    raise ValidationError("slots.caption", "shorten the caption to fit two lines at 30 px")


def video_value(name: str, value: object, rule: dict) -> dict | str:
    """Validate and normalize one declared video slot value."""
    if value == "self":
        if "self" not in rule.get("sources", []):
            raise ValidationError(f"slots.{name}", "self is not allowed; use a media clip object")
        return "self"
    if isinstance(value, dict) and value.get("kind") == "self" and "crop" in value:
        raise ValidationError(f"slots.{name}", "self takes no crop; use a media clip object "
                              "with path and sha256 to crop a picture")
    keys = {"path", "sha256", "in_s", "speed", "focus", "audio", "crop"}
    if not isinstance(value, dict) or "media" not in rule.get("sources", []):
        raise ValidationError(f"slots.{name}", "use a media clip object with path and sha256")
    if set(value) - keys:
        raise ValidationError(f"slots.{name}", "unknown keys; use " + ", ".join(sorted(keys)))
    source, digest = value.get("path"), value.get("sha256")
    if not isinstance(source, str) or not source or _has_controls(source):
        raise ValidationError(f"slots.{name}", "path must name a readable clip file")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValidationError(f"slots.{name}", "supply sha256 as 64 lowercase hex characters")
    result = dict(value)
    for key, low, high, default in (("in_s", 0, math.inf, 0), ("speed", 0.5, 2, 1)):
        result[key] = _number(f"{name}.{key}", value.get(key, default),
                              {"type": "number", "minimum": low, "maximum": high})
    focus = value.get("focus", [0.5, 0.5])
    if (not isinstance(focus, list) or len(focus) != 2
            or any(isinstance(n, bool) or not isinstance(n, (int, float))
                   or not math.isfinite(n) or not 0 <= n <= 1 for n in focus)):
        raise ValidationError(f"slots.{name}", "focus must be [x, y], each within 0..1")
    result["focus"] = list(focus)
    audio = value.get("audio", "mute")
    if audio != "mute":
        if not isinstance(audio, dict) or set(audio) != {"gain_db"}:
            raise ValidationError(f"slots.{name}", "audio must be mute or an object with gain_db")
        audio = {"gain_db": _number(f"{name}.audio.gain_db", audio["gain_db"],
                                     {"type": "number", "minimum": -40, "maximum": 6})}
    result["audio"] = audio
    try:
        result["path"] = str(paths.validate_input(source))
        info = video_info(Path(result["path"]))
    except (InputError, ProcessingError, ValidationError) as exc:
        raise ValidationError(f"slots.{name}", "use a readable video clip with known duration "
                              "and positive size; still images cannot fill video windows") from exc
    if "crop" in value:
        crop = value["crop"]
        if (not isinstance(crop, list) or len(crop) != 4
                or any(type(part) is not int for part in crop)
                or min(crop[:2]) < 0 or min(crop[2:]) < 32
                or crop[0] + crop[2] > info.width or crop[1] + crop[3] > info.height):
            raise ValidationError(f"slots.{name}", f"crop must sit inside the {info.width}x"
                                  f"{info.height} picture; use [x, y, width, height] in integer "
                                  "pixels with width and height at least 32")
        result["crop"] = list(crop)
    return result


def _has_controls(value: str) -> bool:
    return any(
        ord(char) < 32 or 127 <= ord(char) <= 159 or char in "\u2028\u2029" for char in value
    )


def _string(name: str, value: object, rule: dict) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"slots.{name}", "expected a string")
    if _has_controls(value):
        raise ValidationError(f"slots.{name}", "remove control characters")
    low, high = rule.get("min_length", 0), rule["max_length"]
    if not low <= len(value) <= high or (low > 0 and not value.strip()):
        raise ValidationError(f"slots.{name}", f"use {low}..{high} characters; shorten longer text")
    return value


def _number(name: str, value: object, rule: dict) -> int | float:
    integer = rule["type"] == "integer"
    if (
        isinstance(value, bool)
        or not isinstance(value, int if integer else (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise ValidationError(
            f"slots.{name}", "expected an integer" if integer else "expected a finite number"
        )
    if not rule.get("minimum", -math.inf) <= value <= rule.get("maximum", math.inf):
        raise ValidationError(
            f"slots.{name}", f"use a value within {rule['minimum']}..{rule['maximum']}"
        )
    return value


def _slots(slots: Mapping[str, object], rules: dict) -> tuple[
    dict, dict[str, tuple[int, int] | ffmpeg.MediaInfo],
]:
    for name in slots:
        if name not in rules:
            raise ValidationError(f"slots.{name}", "unknown slot; use a declared slot name")
    result = {}
    image_sizes = {}
    for name, rule in rules.items():
        if name not in slots and rule["required"]:
            raise ValidationError(f"slots.{name}", "required slot is missing")
        value = slots[name] if name in slots else rule["default"]
        kind = rule["type"]
        if kind == "string":
            value = _string(name, value, rule)
        elif kind == "boolean":
            if type(value) is not bool:
                raise ValidationError(f"slots.{name}", "use true or false")
        elif kind in {"integer", "number"}:
            value = _number(name, value, rule)
        elif kind == "video" and "sources" in rule:
            value = video_value(name, value, rule)
        elif kind.endswith("_list"):
            if not isinstance(value, list):
                raise ValidationError(f"slots.{name}", "expected a list")
            if not rule["min_items"] <= len(value) <= rule["max_items"]:
                raise ValidationError(
                    f"slots.{name}", f"use {rule['min_items']}..{rule['max_items']} items"
                )
            if kind == "image_list":
                images = []
                for index, entry in enumerate(value):
                    location = f"{name}[{index}]"
                    image, image_sizes[location] = _image(location, entry, rule)
                    images.append(image)
                value = images
            elif kind == "video_list":
                value = [video_value(f"{name}[{index}]", entry, rule)
                         for index, entry in enumerate(value)]
            elif kind == "drop_list":
                validated = []
                for index, entry in enumerate(value):
                    location = f"{name}[{index}]"
                    if not isinstance(entry, dict) or set(entry) != {"text", "at"}:
                        raise ValidationError(f"slots.{location}",
                                              "expected an object with text and at")
                    text = _string(f"{location}.text", entry["text"], rule["items"]["text"])
                    at = _number(f"{location}.at", entry["at"], {"type": "number"})
                    if at < 0:
                        raise ValidationError(f"slots.{location}.at",
                                              "move the landing to 0 or later")
                    if validated and at < validated[-1]["at"]:
                        raise ValidationError(f"slots.{location}.at",
                                              "use non-decreasing landing times in reading order")
                    validated.append({"text": text, "at": at})
                value = validated
            elif kind in {"bar_list", "spec_list"}:
                fields = ("label", "value") if kind == "bar_list" else ("name", "value")
                item_rules = rule["items"]
                validated = []
                for index, entry in enumerate(value):
                    location = f"{name}[{index}]"
                    if not isinstance(entry, dict) or set(entry) != set(fields):
                        raise ValidationError(
                            f"slots.{location}", f"expected an object with {fields[0]} and value"
                        )
                    label = _string(f"{location}.{fields[0]}", entry[fields[0]],
                                    item_rules[fields[0]])
                    item_value = (_number(f"{location}.value", entry["value"],
                                          item_rules["value"])
                                  if kind == "bar_list" else
                                  _string(f"{location}.value", entry["value"],
                                          item_rules["value"]))
                    validated.append({fields[0]: label, "value": item_value})
                value = validated
            else:
                value = [
                    _string(name, entry, rule)
                    if kind == "string_list"
                    else _number(name, entry, {"type": "number"})
                    for entry in value
                ]
        elif kind == "enum":
            if not isinstance(value, str) or value not in rule["options"]:
                raise ValidationError(f"slots.{name}", "choose " + ", ".join(rule["options"]))
        elif kind == "image":
            if value == "" and not rule["required"]:
                result[name] = ""
                continue
            value, image_sizes[name] = _image(name, value, rule)
        elif kind == "video":
            value, image_sizes[name] = _video(name, value, rule)
        result[name] = value
    return result, image_sizes


def _image(name: str, value: object, rule: dict) -> tuple[Path, tuple[int, int]]:
    """Validate each image slot and list entry with the same decoder checks."""
    if not isinstance(value, (str, Path)):
        raise ValidationError(f"slots.{name}", "expected an image path")
    if isinstance(value, str) and _has_controls(value):
        raise ValidationError(f"slots.{name}", "remove control characters from the path")
    if Path(value).suffix.lower() not in rule["options"]:
        raise ValidationError(f"slots.{name}", "use a .png, .jpg, .jpeg or .webp file")
    try:
        value = paths.validate_input(value)
    except InputError as exc:
        raise ValidationError(
            f"slots.{name}", "choose an existing, readable image file"
        ) from exc
    if value.suffix.lower() not in rule["options"]:
        raise ValidationError(f"slots.{name}", "use a .png, .jpg, .jpeg or .webp file")
    readable = "choose a readable image with a video stream and positive dimensions"
    try:
        # Explode fails on decoder errors from damaged scan data; metadata and
        # container warnings still decode successfully.
        ffmpeg.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-err_detect", "explode", "-xerror",
                "-i", str(value),
                "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-",
            ],
            timeout_s=10,
        )
        info = ffmpeg.probe(value, count_frames=False)
    except (InputError, ProcessingError) as exc:
        if isinstance(exc, ProcessingError) and exc.code.endswith("_exec_failed"):
            raise
        raise ValidationError(f"slots.{name}", readable) from exc
    if not info.has_video or not info.width or info.width < 1 or not info.height or info.height < 1:
        raise ValidationError(f"slots.{name}", readable)
    return value, (info.width, info.height)


def _video(name: str, value: object, rule: dict) -> tuple[Path, ffmpeg.MediaInfo]:
    """Probe a local clip without decoding its poster during planning."""
    if not isinstance(value, (str, Path)):
        raise ValidationError(f"slots.{name}", "expected a video path")
    if _has_controls(str(value)):
        raise ValidationError(f"slots.{name}", "remove control characters from the path")
    suffix_fix = "use a .mov, .mp4 or .m4v file"
    if Path(value).suffix.lower() not in rule["options"]:
        raise ValidationError(f"slots.{name}", suffix_fix)
    try:
        source = paths.validate_input(value)
    except InputError as exc:
        raise ValidationError(f"slots.{name}", "choose an existing, readable video file") from exc
    if source.suffix.lower() not in rule["options"]:
        raise ValidationError(f"slots.{name}", suffix_fix)
    try:
        info = ffmpeg.probe(source, count_frames=False)
    except (InputError, ProcessingError) as exc:
        if isinstance(exc, ProcessingError) and exc.code.endswith("_exec_failed"):
            raise
        raise ValidationError(f"slots.{name}",
                              "choose a readable file with a video stream") from exc
    if not info.has_video or not info.width or info.width < 1 or not info.height or info.height < 1:
        raise ValidationError(f"slots.{name}", "choose a file with a video stream")
    if (info.video_duration_s is None or not math.isfinite(info.video_duration_s)
            or info.video_duration_s <= 0):
        raise ValidationError(f"slots.{name}", "choose a clip with a positive video duration")
    return source, info


def _luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def _contrast(a: str, b: str) -> float:
    low, high = sorted((_luminance(a), _luminance(b)))
    return (high + 0.05) / (low + 0.05)


def glow_colors(accent: str) -> tuple[str, str]:
    """Return a pale text tint and the accent for layered text glow."""
    channels = [int(accent[i:i + 2], 16) for i in (1, 3, 5)]
    tint = "#" + "".join(f"{round(channel * 0.35 + 255 * 0.65):02X}" for channel in channels)
    return tint, accent.upper()


def _tokens(graphics: dict, panel_height: int) -> dict[str, object]:
    paper, ink, accent, bar = (
        graphics[k] for k in ("paper_color", "ink_color", "accent_color", "titlebar_color")
    )
    bar_ink = next(
        (c for c in (ink, paper) if _contrast(c, bar) >= 3),
        "#111111" if _luminance(bar) > 0.4 else "#FFFFFF",
    )
    accent_ink = next(
        (c for c in (paper, ink, "#111111", "#FFFFFF") if _contrast(c, accent) >= 4.5),
        max(("#111111", "#FFFFFF"), key=lambda c: _contrast(c, accent)),
    )
    radius, shadow = _LOOKS[graphics["style"]]
    return {
        "paper_color": paper,
        "titlebar_color": bar,
        "bar_ink_color": bar_ink,
        "ink_color": ink,
        "accent_color": accent,
        "accent_ink_color": accent_ink,
        "font_family": fonts.font_stack(graphics["font"]),
        "mono_family": fonts.font_stack(MONO_FONT),
        "card_radius_px": radius,
        "card_shadow": shadow.format(ink=ink),
        "panel_width_px": graphics["card_width_px"],
        "panel_max_height_px": panel_height,
        "pad_x_px": PAD_X,
        "pad_top_px": PAD_TOP,
        "pad_bottom_px": PAD_BOTTOM,
    }


def _fit(
    name: str, text: str, width: float, em: float, maximum: int, *, look: str | None = None
) -> int:
    capacity = max(0, math.floor(width / (em * MIN_TEXT_PX)))
    if len(text) > capacity:
        raise ValidationError(
            f"slots.{name}",
            f"shorten to at most {capacity} characters to fit at {MIN_TEXT_PX} px"
            + (f" in the {look} look" if look else ""),
        )
    return min(maximum, math.floor(width / (em * max(1, len(text)))))


def _fit_measured(
    name: str, text: str, width_px: float, family: str, weight: int, maximum: int, *,
    tracking_em: float = 0.0, look: str | None = None,
) -> int:
    width_em = text_width_em(text, family, weight, tracking_em=tracking_em)
    for size in range(maximum, MIN_TEXT_PX - 1, -1):
        if width_em * size <= width_px:
            return size
    widest = max(
        (text_width_em(char, family, weight, tracking_em=tracking_em) for char in text),
        default=1.0,
    )
    capacity = max(0, math.floor(width_px / (widest * MIN_TEXT_PX)))
    raise ValidationError(
        f"slots.{name}",
        f"shorten to at most {capacity} characters to fit at {MIN_TEXT_PX} px"
        + (f" in the {look} look" if look else ""),
    )


def _fit_quote(text: str, width_px: float, family: str) -> int:
    """Choose the largest readable size that wraps the quote into two lines."""
    words = text.split(" ")
    for size in range(46, MIN_TEXT_PX - 1, -1):
        lines = 1
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if text_width_em(candidate, family, 600) * size <= width_px:
                current = candidate
            else:
                lines += 1
                current = word
                if text_width_em(word, family, 600) * size > width_px:
                    lines = 3
                    break
        if lines <= 2:
            return size
    raise ValidationError("slots.quote", "shorten the quote to fit two lines at 30 px")


def gravity_drop_fonts(style: ResolvedStyle) -> tuple[str, ...]:
    """Return the creator's font cycle in role order."""
    roles = ("graphics", "captions", "hook")
    return tuple(dict.fromkeys(style.data[role]["font"] for role in roles))


def _gravity_layout(
    drops: list[dict], style: ResolvedStyle, width: int, height: int,
) -> tuple[int, list[dict]]:
    """Fit centred words into one or two rows with room for motion and shadows."""
    captions = style.data["captions"]
    families = gravity_drop_fonts(style)
    shadow = math.ceil(3 * captions["shadow_blur_px"]
                       + max(abs(captions["shadow_x_px"]), abs(captions["shadow_y_px"]))) + 4
    room = width - 2 * shadow
    maximum = math.floor(captions["size_px"] * 1.35)
    try:
        maximum = min(maximum, *(
            _fit_measured("drops", drop["text"], room / 1.2,
                          families[index % len(families)], 400, maximum)
            for index, drop in enumerate(drops)
        ))
    except ValidationError as exc:
        raise ValidationError("slots.drops", "shorten drop words to fit at 30 px") from exc
    rotations = (-4, 3, -2, 5)
    for size in range(maximum, MIN_TEXT_PX - 1, -1):
        bounds = []
        for index, drop in enumerate(drops):
            text_width = text_width_em(drop["text"], families[index % len(families)], 400) * size
            angle = math.radians(abs(rotations[index]))
            # Include italic bearings, squash width and the rotated line box.
            w, h = text_width * 1.15 + size * 0.3, size * 1.4
            bounds.append((math.ceil(w * 1.08 * math.cos(angle) + h * math.sin(angle)),
                           math.ceil(w * 1.08 * math.sin(angle) + h * math.cos(angle))))
        gap = 16
        candidates = [[list(range(len(drops)))]] + [
            [list(range(split)), list(range(split, len(drops)))]
            for split in range(1, len(drops))
        ]
        candidates.sort(key=lambda rows: (len(rows), max(
            sum(bounds[i][0] for i in row) + gap * (len(row) - 1) for row in rows)))
        for rows in candidates:
            row_widths = [sum(bounds[i][0] for i in row) + gap * (len(row) - 1)
                          for row in rows]
            row_heights = [max(bounds[i][1] for i in row) for row in rows]
            total_height = sum(row_heights) + gap * (len(rows) - 1)
            if max(row_widths) > room or total_height > height - 2 * shadow:
                continue
            planned = []
            y = (height - total_height) / 2
            for row, row_width, row_height in zip(rows, row_widths, row_heights, strict=True):
                x = (width - row_width) / 2
                for index in row:
                    w, h = bounds[index]
                    planned.append({**drops[index], "fall_s": min(0.26, drops[index]["at"]),
                                    "font_family": fonts.font_stack(
                                        families[index % len(families)]),
                                    "font_style": "italic" if index % 2 else "normal",
                                    "rotation": rotations[index], "x": x, "y": y,
                                    "width": w, "height": row_height})
                    x += w + gap
                y += row_height + gap
            return size, planned
    raise ValidationError("slots.drops", "shorten drop words to fit two rows at 30 px")


def _bar_fit(title: str, width: float, *, host: str = "", mono: bool = False) -> None:
    # Symmetric room keeps the title centered even when the right pill is wider.
    side = 48.0
    if host:
        _fit("host", host, (width - 48) / 2 - 50, MONO_EM, MIN_TEXT_PX)
        side = max(side, len(host) * MONO_EM * MIN_TEXT_PX + 38)
    _fit(
        "title",
        title,
        width - 48 - 2 * (side + 12) - (24 if mono else 0),
        MONO_EM if mono else PROPORTIONAL_EM,
        MIN_TEXT_PX,
    )


def _fit_comment(text: str, width_px: float, family: str) -> tuple[int, list[str]]:
    """Fit the comment into at most two measured lines."""
    for size in range(40, MIN_TEXT_PX - 1, -1):
        lines = [""]
        for word in text.split(" "):
            candidate = lines[-1] + " " + word if lines[-1] else word
            if text_width_em(candidate, family, 500) * size <= width_px:
                lines[-1] = candidate
            elif text_width_em(word, family, 500) * size <= width_px:
                lines.append(word)
            else:
                break
        else:
            if len(lines) <= 2:
                return size, lines
    raise ValidationError("slots.text", "shorten the comment to fit two lines at 30 px")


def _cue_window(
    slots: dict, duration: float, count: int, *, reserve: float, lower: float = 0,
    unit: str = "row", upper_tolerance: float = 0, out_s: float | None = None,
) -> None:
    cues = slots["cues"]
    out_start = duration - (min(0.3, duration * 0.15) if out_s is None else out_s)
    end = out_start - reserve
    if len(cues) > count:
        raise ValidationError("slots.cues", f"use at most {count} cues, one per {unit}")
    if any(not lower <= cue <= end + upper_tolerance for cue in cues):
        raise ValidationError(
            "slots.cues",
            f"use seconds within {lower:g}..{end:g}; leave {reserve:g} s before "
            f"the fade-out starts at {out_start:g} s",
        )
    if any(a > b for a, b in zip(cues, cues[1:], strict=False)):
        raise ValidationError("slots.cues", "use non-decreasing cue times")


def _flow_tags_cues(explicit: list[float], duration: float, count: int) -> list[float]:
    """Share tag cues between the picture plan and sound placement."""
    first = 0.1
    latest = min(duration - FLOW_TAGS_OUT_S - FLOW_TAGS_POP_S - FLOW_TAGS_REST_S,
                 max(first + (count - 1) * (FLOW_TAGS_POP_S + FLOW_TAGS_ARROW_S),
                     duration * 0.6 - FLOW_TAGS_POP_S))
    gap = (latest - first) / (count - 1)
    cues = []
    for index in range(count):
        cue = (explicit[index] if index < len(explicit)
               else max(cues[-1] if cues else 0, min(latest, first + index * gap)))
        cues.append(cue)
    return cues


def _terminal_segments(command: str) -> list[str]:
    """Split runs using only U+0020 as the word separator."""
    return re.findall(r" +|[^ ]+", command)


def _terminal_word_window(
    values: dict, duration: float, *, tail_s: float = TERMINAL_WORD_TAIL_S,
) -> None:
    cues = values["word_cues"]
    words = sum(not segment.startswith(" ") for segment in _terminal_segments(values["command"]))
    if len(cues) > words:
        raise ValidationError(
            "slots.word_cues", f"use 0..{words} cues, at most one per U+0020-separated word"
        )
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - tail_s
    window = f"{TERMINAL_WORD_IN_S:g}..{latest:g}"
    if any(not TERMINAL_WORD_IN_S <= cue <= latest for cue in cues):
        raise ValidationError(
            "slots.word_cues",
            f"use nonnegative seconds within {window}; first cue must follow the panel entrance "
            f"at {TERMINAL_WORD_IN_S:g} s and last cue must leave {tail_s:g} s "
            f"before the fade-out starts at {out_start:g} s",
        )
    if any(a > b for a, b in zip(cues, cues[1:], strict=False)):
        raise ValidationError(
            "slots.word_cues", f"use non-decreasing cue times within {window}"
        )


def _terminal_schedule(
    command: str,
    output: list[str],
    cadence: str,
    type_at: float,
    word_cues: list[float],
    cues: list[float],
    duration: float,
    *,
    reserve_after_s: float = 0.1,
) -> dict[str, object]:
    """Produce the planner's typing and printing schedule for this command and output."""
    state = 0x51EC0DED

    def rnd() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) % 2**32
        return state / 2**32

    def chunks_of(word: str) -> list[str]:
        if cadence == "uniform":
            return list(word)
        chunks = []
        index = 0
        while index < len(word):
            count = 1 + math.floor(rnd() * 3)
            chunks.append(word[index : index + count])
            index += count
        return chunks

    type_start = word_cues[0] if word_cues else type_at
    rows = [[0.0, 0]]
    time = anchor = type_start
    count = word_index = 0
    anchor_row = len(rows)
    for segment in _terminal_segments(command):
        if segment.startswith(" "):
            if word_cues and word_index == 0:
                count += len(segment)
                continue
            time += 0.11 if cadence == "uniform" else 0.1 + rnd() * 0.1
            count += len(segment)
            rows.append([time, count])
            continue
        cued = word_index < len(word_cues)
        if cued:
            cue = word_cues[word_index]
            if word_index > 0 and time > cue:
                factor = (cue - anchor) / (time - anchor)
                for row in rows[anchor_row:]:
                    row[0] = anchor + (row[0] - anchor) * factor
                time = cue
            time = max(time, cue)
            anchor = time
            anchor_row = len(rows)
        for index, chunk in enumerate(chunks_of(segment)):
            gap = len(chunk) * (0.05 if cadence == "uniform" else 0.04 + rnd() * 0.03)
            if not cued or index > 0:
                time += gap
            count += len(chunk)
            rows.append([time, count])
        word_index += 1

    type_end = max(time, type_start + 0.01)
    exec_at = type_end + 0.22
    output_at = [exec_at + 0.3 + index * 0.3 for index in range(len(output))]
    natural_end = (output_at[-1] if output_at else exec_at) + TERMINAL_PRINT_S
    out_start = duration - min(0.3, duration * 0.15)
    squeeze_start = word_cues[-1] if word_cues else type_start
    max_end = max(squeeze_start, out_start - reserve_after_s)
    if natural_end > max_end:
        factor = (max_end - squeeze_start) / (natural_end - squeeze_start)

        def squeeze(at: float) -> float:
            return at if at <= squeeze_start else squeeze_start + (at - squeeze_start) * factor

        for row in rows:
            row[0] = squeeze(row[0])
        type_end = squeeze(type_end)
        exec_at = squeeze(exec_at)
        output_at = [squeeze(at) for at in output_at]

    # Validate against execution after squeezing, before explicit cues replace defaults.
    exec_at = round(exec_at, 6)
    if any(cue < exec_at for cue in cues):
        raise ValidationError(
            "slots.cues", f"output cues must be at or after execution at {exec_at:.6f} s"
        )
    _cue_window(
        {"cues": cues}, duration, len(output), reserve=TERMINAL_PRINT_S, lower=exec_at
    )
    previous = 0.0
    for index, at in enumerate(output_at):
        proposed = (cues[index] if index < len(cues) else previous + 0.3) if cues else at
        previous = max(previous, min(out_start - TERMINAL_PRINT_S, proposed))
        output_at[index] = round(previous, 6)

    # Keep the final endpoint at coincident times, including rounded cue boundaries.
    endpoints = []
    for at, count in rows:
        row = [round(at, 6), count]
        if endpoints and endpoints[-1][0] == row[0]:
            endpoints[-1] = row
        else:
            endpoints.append(row)
    return {
        "command": command,
        "output": output,
        "rows": endpoints,
        "type_start": round(type_start, 6),
        "type_end": round(type_end, 6),
        "exec_at": exec_at,
        "output_at": output_at,
    }


def _typing_schedule(
    text: str,
    cadence: str,
    type_at: float,
    word_cues: list[float],
    cues: list[float],
    duration: float,
) -> dict[str, object]:
    """Reuse terminal typing with room for confirmation and readable rest."""
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - 1.2
    if not 0 <= type_at <= latest:
        raise ValidationError(
            "slots.type_at", f"use seconds within 0..{latest:g}; raise graphics.hold_s "
            "to leave room for typing and confirmation",
        )
    _terminal_word_window({"command": text, "word_cues": word_cues}, duration, tail_s=1.2)
    schedule = _terminal_schedule(
        text, [], cadence, type_at, word_cues, [], duration, reserve_after_s=1.0,
    )
    _cue_window({"cues": cues}, duration, 1, reserve=0.7, lower=schedule["type_end"])
    return {
        "text": text,
        "rows": schedule["rows"],
        "type_start": schedule["type_start"],
        "type_end": schedule["type_end"],
        "confirm_at": cues[0] if cues else round(schedule["type_end"] + 0.3, 6),
    }


def _choices_cues(given: list[float], count: int, duration: float, fps: int) -> list[float]:
    minimum_gap = math.ceil(Fraction(CHOICES_MIN_GAP_S).limit_denominator() * fps) / fps
    latest = duration - (CHOICES_LAND_S + CHOICES_HOLD_S + CHOICES_SETTLE_S
                         + CHOICES_REST_S + CHOICES_OUT_S)
    if len(given) > count:
        raise ValidationError("slots.cues", f"use at most {count} cues, one per option")
    if any(cue < 0 or cue > latest + 1e-6 for cue in given):
        raise ValidationError("slots.cues", f"use seconds from 0 to {latest:g}; move the last "
                              "option earlier or end the card later")
    if any(b - a < minimum_gap - 1e-6 for a, b in zip(given, given[1:], strict=False)):
        raise ValidationError("slots.cues", "space the cues in increasing order at least "
                              f"{minimum_gap:g} s apart")
    gap = latest / (count - 1)
    cues = list(given)
    for i in range(len(cues), count):
        cue = max(cues[-1] + minimum_gap if cues else 0, i * gap)
        if cue > latest + 1e-6:
            raise ValidationError("slots.cues", "end the card later or give fewer options "
                                  "to leave room for the cue spacing and final rest")
        cues.append(round(cue, 6))
    return cues


def _row_cues(values: dict, count: int, duration: float) -> list[float]:
    latest = duration - min(0.3, duration * 0.15) - 0.6
    if count and latest < 0:
        raise ValidationError("slots.cues", "end the card later to leave 0.3 s for each row "
                              "to enter and 0.3 s of rest before the fade-out")
    _cue_window(values, duration, count, reserve=0.6, upper_tolerance=1e-9)
    first = min(0.3, latest)
    cues = list(values["cues"])
    for i in range(len(cues), count):
        default = first + i * (latest - first) / max(1, count - 1)
        cues.append(round(max(cues[-1] if cues else 0, default), 6))
    return cues


def _row_padding(glow: bool, main_px: int, shadow: str) -> tuple[int, int]:
    """Reserve the halo or panel shadow, including the 8 px entrance."""
    if glow:
        pad = math.ceil(0.5 * main_px)
        return pad, pad + 8
    top = bottom = 0.0
    for match in re.finditer(r"(?:^|,\s*)((?:[+-]?\d+(?:\.\d+)?(?:px)?\s+){2,4})", shadow):
        lengths = [float(token.removesuffix("px")) for token in match[1].split()]
        _, offset_y, blur, spread = lengths + [0.0] * (4 - len(lengths))
        top = max(top, blur - offset_y + spread)
        bottom = max(bottom, blur + offset_y + spread)
    # Keep the soft tail and entrance inside the layer, independently on each side.
    return math.ceil(top) + 8, math.ceil(bottom) + 8


def _readable_row_size(labels: list[tuple[str, float]], width: float,
                       minimum: int, maximum: int) -> int:
    size = maximum
    for index, (label, units) in enumerate(labels):
        fitted = min(maximum, math.floor(width / max(units, 0.01)))
        if fitted < minimum:
            raise ValidationError("slots.rows", f"shorten row {index + 1} ({label!r}) to fit "
                                  f"one line at {minimum} px, or widen the card")
        size = min(size, fitted)
    return size


def card_layer_plan(
    template: str,
    slots: Mapping[str, object],
    style: ResolvedStyle,
    *,
    frames: int | None = None,
    fps: int = 30,
    tiles: Sequence[dict] | None = None,
    ceiling: int | None = None,
    project: CardProject | None = None,
    max_width_px: int | None = None,
    max_height_px: int | None = None,
) -> CardPlan:
    """Fit typed content before preparing or rendering assets.

    Choices, Flow Tags and rows on cue can refit within placement width and height limits.
    Terminal schedule is produced by this planner. Overrides of command, output
    or the timing slots must go through card_layer_plan to rebuild the schedule.
    """
    vocabulary = bleep.listed(style.data["music"]) if bleep.enabled(style.data) else None
    if vocabulary is not None:
        slots, _ = bleep.mask_slots(dict(slots), load_template(template)["slots"], vocabulary)
    if registry_cards.is_kind(template):
        from yap_studio.card_motion import preview_motion

        return preview_motion(registry_cards.card_plan(template, slots, style,
                                                       frames=frames, fps=fps), style)
    metadata = load_template(template)
    graphics, internal = style.data["graphics"], style.data["internal"]
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValidationError("fps", "use a positive integer")
    if frames is None:
        frames = max(1, round(graphics["hold_s"] * fps))
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValidationError("frames", "use a positive integer")
    minimum_s = _minimum_duration_s(template, slots, graphics)
    if minimum_s is not None:
        minimum = math.ceil(minimum_s * fps)
        if frames < minimum:
            raise ValidationError(
                "frames",
                f"a {template} card needs at least {minimum_s:g} s "
                f"({minimum} frames at {fps} fps); raise graphics.hold_s or pass more frames",
            )
    duration = hyperframes.layer_duration_s(frames, fps)
    row_look = graphics.get("specs_rows", "together")
    if row_look == "together" and template in {"specs_list", "stat"}:
        for name in (("cues",) if template == "specs_list" else ("rows", "cues")):
            if name in slots:
                raise ValidationError(f"slots.{name}", "turn on Rows on cue in the style, "
                                      f"or remove {name}")
    values, image_sizes = _slots(slots, metadata["slots"])
    if vocabulary is not None and template == "file_drop" and not values["name"]:
        values["name"] = bleep.mask_text(values["clip"].name, vocabulary)
    height = internal["key_text_bottom_px"] - internal["key_text_top_px"]
    panel_height = height - PAD_TOP - PAD_BOTTOM
    width = graphics["card_width_px"]
    body_width = width - 2 * CONTENT_PAD
    variables = _tokens(graphics, panel_height)
    row_card = rows_on_cue(style, template, values)
    if row_card:
        # Glow and the rows' 12 px entrance need room on both sides.
        pad_x = 48 if row_look == "on_cue_glow" else PAD_X
        width = min(width, max_width_px - 2 * pad_x if max_width_px is not None else width)
        body_width = width - (0 if row_look == "on_cue_glow" else 24)
        variables.update(panel_width_px=width, pad_x_px=pad_x)
    branded_font = bool(style.data.get("brand", {}).get("cards_font"))
    if branded_font and template != "terminal":
        variables["mono_family"] = fonts.font_stack(graphics["font"])
    if template == "big_number":
        return number_plan(values, style, frames=frames, fps=fps, tokens=variables)
    assets = ()
    poster = None
    windows = ()
    if template == "terminal":
        out_start = duration - min(0.3, duration * 0.15)
        if not 0 <= values["type_at"] <= out_start - 0.5:
            raise ValidationError("slots.type_at", "leave at least 0.5 s before the fade-out")
        _terminal_word_window(values, duration)
        schedule = _terminal_schedule(
            values["command"], values["output"], values["cadence"], values["type_at"],
            values["word_cues"], values["cues"], duration,
        )
        _bar_fit(values["title"], width, host=values["host"])
        # Reserve the caret after the widest command, in addition to the glyph.
        prefix_units = (len(values["prompt"]) + 1) * MONO_EM + CARET_EM
        _fit("command", values["command"], body_width - MIN_TEXT_PX * prefix_units, MONO_EM, 40)
        size = min(40, math.floor(body_width / (len(values["command"]) * MONO_EM + prefix_units)))
        for line in values["output"]:
            size = min(size, _fit("output", line, body_width, MONO_EM, 40))
        lines = 1 + len(values["output"])
        size = min(
            size, math.floor(
                (panel_height - BAR_HEIGHT - 2 * TERMINAL_BODY_PAD_Y)
                / (lines * TERMINAL_LINE_HEIGHT)
            )
        )
        if size < MIN_TEXT_PX:
            raise ValidationError("slots.output", "remove output lines to fit the panel height")
        variables.update(
            {name: values[name] for name in ("command", "title", "host", "prompt")},
            text_px=size, bar_px=MIN_TEXT_PX,
        )
        variables["output"] = json.dumps(values["output"], ensure_ascii=False)
        variables["schedule"] = json.dumps(schedule)
    elif template == "stat":
        rows = values.pop("rows")
        cues = values.pop("cues")
        glow = row_look == "on_cue_glow"
        if glow and rows and values["reveal"] in {"wheel", "slot"}:
            raise ValidationError("slots.reveal", "use reveal count with the glowing rows look")
        row_min, row_max = (48, 56) if glow else (44, 52)
        row_px = (_readable_row_size(
            [(row, text_width_em(row, graphics["font"], 600)) for row in rows],
            body_width, row_min, row_max) if row_card else 32)
        kicker = _fit(
            "kicker", values["kicker"].upper(), body_width, KICKER_EM + KICKER_TRACKING_EM, 36
        )
        figures = ("end",) if values["reveal"] in {"wheel", "slot"} else ("start", "end")
        figure_units = max(len(f"{values[k]:,}") for k in figures) * DIGIT_EM
        slot_digit_em = next(
            variable["default"] for variable in metadata["variables"]
            if variable["id"] == "slot_digit_em"
        )
        if values["reveal"] == "slot":
            slot_digit_em = widest_digit_em(graphics["font"], 600) + STAT_SLOT_SAFETY_EM
            # Reserve each measured reel window instead of the old digit estimate.
            digits = sum(char.isdigit() for char in f"{values['end']:,}")
            figure_units += digits * (slot_digit_em - DIGIT_EM)
        affixes = len(values["prefix"]) + len(values["suffix"])
        units = figure_units + affixes * 0.55 * PROPORTIONAL_EM
        units += sum(bool(values[k]) for k in ("prefix", "suffix")) * 0.55 * 0.14
        available_height = panel_height - 48 - (kicker * 1.2 + 12 if values["kicker"] else 0)
        if row_card:
            if body_width / (units * 1.07) < (72 if glow else 60):
                raise ValidationError("slots.rows", "shorten the affixes or widen the card "
                                      "to keep the figure at a readable size")
            fixed = (0 if glow else 24) + (kicker * 1.2 + 12 if values["kicker"] else 0)
            for fitted_row in range(row_px, row_min - 1, -1):
                rest = (fixed + 12 + len(rows) * fitted_row * (1.2 if glow else 1.15)
                        + (len(rows) - 1) * 4)
                maximum = math.floor(min(90 if glow else 160, body_width / (units * 1.07)))
                for size in range(maximum, (72 if glow else 60) - 1, -1):
                    top_pad, bottom_pad = _row_padding(glow, size, variables["card_shadow"])
                    panel_height = math.ceil(rest + size * 1.07)
                    height = math.ceil((panel_height + top_pad + bottom_pad) / 2) * 2
                    if max_height_px is None or height <= max_height_px:
                        break
                else:
                    continue
                row_px = fitted_row
                break
            else:
                raise ValidationError("slots.rows", "use fewer or shorter rows, or another "
                                      "moment with more room above the head, to keep the "
                                      "figure and rows at a readable size")
            variables.update(panel_max_height_px=panel_height, pad_top_px=top_pad,
                             pad_bottom_px=bottom_pad, glow_pad_px=top_pad)
        else:
            size = math.floor(min(160, body_width / (units * 1.07), available_height / 1.07))
        if size < 2 * MIN_TEXT_PX:
            if rows:
                raise ValidationError("slots.rows", "remove breakdown rows, shorten the "
                                      "affixes or widen the card to keep the figure at 60 px")
            raise ValidationError(
                "slots.end", "shorten the formatted value or affixes to fit at 60 px"
            )
        variables.update(values, value_px=size, kicker_px=kicker, slot_digit_em=slot_digit_em)
        if row_look != "together" and (rows or "cues" in slots):
            variables.update(rows=json.dumps(rows, ensure_ascii=False), row_px=row_px,
                             cues=json.dumps(_row_cues({"cues": cues}, len(rows), duration)))
    elif template == "checklist":
        size = 44
        for item in values["items"]:
            # The tick scales with type; reserve its 1.3 em width and the gap.
            size = min(
                size, _fit("items", item, body_width - 1.3 * MIN_TEXT_PX - 16, PROPORTIONAL_EM, 44)
            )
            size = min(size, math.floor((body_width - 16) / (len(item) * PROPORTIONAL_EM + 1.3)))
        count = len(values["items"])
        size = min(
            size,
            math.floor(
                (panel_height - 48 - (count - 1) * 10 - count * 7) / (count * CHECKLIST_LINE_HEIGHT)
            ),
        )
        if size < MIN_TEXT_PX:
            raise ValidationError("slots.items", "shorten labels or remove rows to fit at 30 px")
        _cue_window(values, duration, count, reserve=CHECKLIST_REVEAL_S)
        variables.update(
            items=json.dumps(values["items"], ensure_ascii=False),
            cues=json.dumps(values["cues"]),
            text_px=size,
        )
    elif template == "text_tag":
        variant = values.pop("variant")
        look = (
            TAG_SIGN_LOOKS[graphics["style"]] if variant == "sign"
            else {"tag": "pill", "plain": "underline"}.get(variant, variant)
        )
        left, right, top, bottom = TAG_PADDING[look]
        content_width = width - left - right - TAG_WIDTH_RESERVE[look]
        if look in {"rule", "title"}:
            text_size = _fit_measured(
                "text", values["text"].upper() if look == "rule" else values["text"],
                content_width, graphics["font"], 700 if look == "rule" else 600, 64,
                tracking_em=TAG_TRACKING_EM.get(look, 0), look=look,
            )
            sub_size = _fit_measured(
                "subtext", values["subtext"].upper() if look == "title" else values["subtext"],
                content_width, graphics["font"], 400 if look == "rule" else 600, 36,
                tracking_em=TAG_SUB_TRACKING_EM[look], look=look,
            )
        else:
            uppercase = look in TAG_TRACKING_EM
            text_em = TAG_UPPERCASE_EM + TAG_TRACKING_EM[look] if uppercase else PROPORTIONAL_EM
            sub_em = MONO_EM if look in {"block", "underline"} else PROPORTIONAL_EM
            sub_em += TAG_SUB_TRACKING_EM.get(look, 0)
            text_size = _fit(
                "text", values["text"].upper() if uppercase else values["text"],
                content_width, text_em, 64, look=look,
            )
            sub_size = _fit(
                "subtext", values["subtext"], content_width, sub_em, 36, look=look,
            )
        rule = TAG_RULE_HEIGHT[look]
        gaps = int(bool(values["subtext"])) + int(bool(rule))
        available = panel_height - top - bottom - rule - gaps * TAG_GAP[look]
        if values["subtext"]:
            available -= sub_size * SUBTEXT_LINE_HEIGHT
        line_height = 1.02 if look == "card" else TAG_LINE_HEIGHT
        text_size = min(text_size, math.floor(available / line_height))
        if text_size < MIN_TEXT_PX:
            raise ValidationError(
                "slots.text", f"shorten text or remove subtext to fit the {look} look"
            )
        variables.update(
            values, look=look, text_px=text_size, subtext_px=sub_size,
            plain_ink_color=max(
                (graphics["paper_color"], graphics["ink_color"]), key=_luminance
            ),
        )
    elif template == "transfer":
        out_start = duration - min(0.3, duration * 0.15)
        latest = out_start - TRANSFER_TRAVEL_S - TRANSFER_LAND_S
        if not 0 <= values["travel_at"] <= latest:
            raise ValidationError(
                "slots.travel_at",
                f"use seconds within 0..{latest:g}; leave "
                f"{TRANSFER_TRAVEL_S + TRANSFER_LAND_S:g} s before "
                f"the fade-out starts at {out_start:g} s",
            )
        inset = TRANSFER_GEOMETRY["panel_inner_pad_px"]
        column_width = math.floor((width - 2 * inset) * TRANSFER_COLUMN_SHARE)
        device_width = TRANSFER_GEOMETRY["device_width_px"]
        left_device_x = inset + (column_width - device_width) / 2
        right_device_x = width - inset - column_width + (column_width - device_width) / 2
        left_center = left_device_x + device_width / 2
        right_center = right_device_x + device_width / 2
        # Both authored screens are centered in the device viewBox.
        chip_width = math.floor(2 * min(left_center - inset, width - inset - right_center))
        label_size = min(
            _fit(name, values[name], column_width, PROPORTIONAL_EM, 36)
            for name in ("left_label", "right_label")
        )
        chrome_width = (
            TRANSFER_GEOMETRY["chip_icon_px"]
            + TRANSFER_GEOMETRY["chip_gap_px"]
            + 2 * TRANSFER_GEOMETRY["chip_pad_x_px"]
        )
        chip_size = _fit(
            "payload", values["payload"], chip_width - chrome_width, PROPORTIONAL_EM, 40
        )
        variables.update(
            values,
            **TRANSFER_GEOMETRY,
            label_px=label_size,
            chip_px=chip_size,
            device_column_width_px=column_width,
            chip_max_width_px=chip_width,
            travel_s=TRANSFER_TRAVEL_S,
            land_s=TRANSFER_LAND_S,
        )
    elif template == "image_compare":
        out_start = duration - min(0.3, duration * 0.15)
        latest = out_start - COMPARE_WIPE_S - COMPARE_REST_S
        if not 0 <= values["wipe_at"] <= latest:
            raise ValidationError(
                "slots.wipe_at",
                f"use seconds within 0..{latest:g}; leave "
                f"{COMPARE_WIPE_S + COMPARE_REST_S:g} s before "
                f"the fade-out starts at {out_start:g} s",
            )
        before, after = (values.pop(name) for name in ("before", "after"))
        image_width, image_height = width - 2 * COMPARE_MAT, panel_height - 2 * COMPARE_MAT
        before_width, before_height = image_sizes["before"]
        frame_min_width = math.floor(
            before_width * min(1, image_width / before_width, image_height / before_height)
        )
        chrome = 2 * (COMPARE_CHIP_PAD + COMPARE_CHIP_BORDER)
        divider_clearance = COMPARE_DIVIDER_KNOB / 2 + COMPARE_CHIP_GAP
        chip_size = 36
        chip_widths = {}
        for name, share in (("after", values["split"] / 100),
                            ("before", (100 - values["split"]) / 100)):
            chip_width = max(0, math.floor(
                share * frame_min_width - COMPARE_CHIP_INSET - divider_clearance
            ))
            chip_widths[f"{name}_chip_max_width_px"] = chip_width
            label = values[f"{name}_label"]
            if not label:
                continue
            label_width = chip_width - chrome
            capacity = max(0, math.floor(label_width / (PROPORTIONAL_EM * MIN_TEXT_PX)))
            if len(label) > capacity:
                raise ValidationError(
                    f"slots.{name}_label",
                    f"shorten to at most {capacity} characters to fit at {MIN_TEXT_PX} px "
                    f"at split {values['split']}%",
                )
            chip_size = min(chip_size, math.floor(label_width / (PROPORTIONAL_EM * len(label))))
        # Invert the wipe's sine easing so the whole chip is revealed before its fade starts.
        reveal_progress = 0
        if values["after_label"]:
            after_right = COMPARE_CHIP_INSET + chip_widths["after_chip_max_width_px"]
            revealed_share = after_right / (frame_min_width * values["split"] / 100)
            reveal_progress = math.acos(1 - 2 * revealed_share) / math.pi
        assets = (
            ("before" + before.suffix.lower(), before),
            ("after" + after.suffix.lower(), after),
        )
        variables.update(
            values,
            before_src=assets[0][0],
            after_src=assets[1][0],
            chip_px=chip_size,
            **chip_widths,
            after_chip_reveal_progress=reveal_progress,
            image_max_width_px=image_width,
            image_max_height_px=image_height,
        )
    elif template == "steps":
        labels = values["steps"]
        count = len(labels)
        em = STEPS_MONO_ADVANCE_EM + STEPS_TRACKING_EM
        characters = sum(len(label.upper()) for label in labels)
        chrome = count * 2 * (STEPS_CHIP_PAD_X + STEPS_CHIP_BORDER)
        available = body_width - chrome - (count - 1) * STEPS_CHIP_GAP
        capacity = max(0, math.floor(available / (em * MIN_TEXT_PX)))
        if characters > capacity:
            raise ValidationError(
                "slots.steps",
                f"shorten labels to at most {capacity} uppercase characters in total or remove "
                f"steps to fit one row at {MIN_TEXT_PX} px",
            )
        size = min(STEPS_MAX_TEXT_PX, math.floor(available / (characters * em)))
        size = min(size, math.floor(
            (panel_height - 2 * CONTENT_PAD - 2 * (STEPS_CHIP_PAD_Y + STEPS_CHIP_BORDER))
            / STEPS_LINE_HEIGHT
        ))
        if size < MIN_TEXT_PX:
            raise ValidationError(
                "slots.steps", "remove steps or raise panel height to fit at 30 px"
            )
        _cue_window(values, duration, count, reserve=STEPS_REST_S, unit="step")
        # Default chip i lights at entrance_end + i * gap, where entrance_end is
        # 0.3 + 0.24 + (count - 1) * 0.06 and gap is
        # (duration - min(0.3, duration * 0.15) - 0.6 - entrance_end) / (count - 1).
        # Late partial cues clamp following defaults to the previous cue, keeping
        # the resolved rhythm non-decreasing and the final chip lit for 0.6 s.
        entrance_end = STEPS_PANEL_IN_S + STEPS_CHIP_IN_S + (count - 1) * STEPS_STAGGER_S
        latest = duration - min(0.3, duration * 0.15) - STEPS_REST_S
        gap = (latest - entrance_end) / (count - 1)
        cues = []
        for index in range(count):
            cue = (values["cues"][index] if index < len(values["cues"])
                   else max(cues[-1] if cues else 0, min(latest, entrance_end + index * gap)))
            cues.append(cue)
        variables.update(
            steps=json.dumps(labels, ensure_ascii=False), cues=json.dumps(cues), text_px=size,
        )
    elif template == "flow" and graphics.get("flow_look", "boxes") == "tags":
        labels = values["nodes"]
        count = len(labels)
        if count not in (2, 3):
            raise ValidationError("slots.nodes", "use 2 or 3 nodes, or the boxes look")
        limit = math.floor(min(864, max_width_px if max_width_px is not None else 864) / 4) * 4
        units = [text_width_em(label, graphics["font"], 600) for label in labels]
        for size in range(FLOW_TAGS_MAX_TEXT_PX, FLOW_TAGS_MIN_TEXT_PX - 1, -1):
            # Chromium rounds text and padding separately. Leave two extra pixels.
            tag_widths = [math.ceil(em * size) + 2 * math.ceil(FLOW_TAGS_PAD_X_EM * size) + 2
                          for em in units]
            # back.out(2.2) peaks below 1.16. Include the small shadow on every edge.
            layer_width = math.ceil((max(tag_widths) * FLOW_TAGS_SCALE + 16) / 4) * 4
            tag_height = math.ceil(size * (FLOW_TAGS_LINE_HEIGHT + 2 * FLOW_TAGS_PAD_Y_EM))
            reserve = (FLOW_TAGS_SCALE - 1) * tag_height / 2
            pad_top = math.ceil(reserve + 6)
            pad_bottom = math.ceil(reserve + 8)
            stack_height = count * tag_height + (count - 1) * FLOW_TAGS_ARROW_PX
            height = math.ceil((stack_height + pad_top + pad_bottom) / 2) * 2
            if layer_width <= limit and (max_height_px is None or height <= max_height_px):
                break
        else:
            for label, tag_width in zip(labels, tag_widths, strict=True):
                if tag_width * FLOW_TAGS_SCALE + 16 > limit:
                    raise ValidationError("slots.nodes", f'shorten tag "{label}" to fit one line '
                                          'at 52 px')
            raise ValidationError("slots.nodes",
                                  "use shorter tags, or another moment with more room above "
                                  "the head, to fit at 52 px")
        width = layer_width - 2 * PAD_X
        tag_width = max(tag_widths)
        _cue_window(values, duration, count, reserve=FLOW_TAGS_POP_S + FLOW_TAGS_REST_S,
                    unit="tag", out_s=FLOW_TAGS_OUT_S)
        cues = _flow_tags_cues(values["cues"], duration, count)
        variables.update(flow_look="tags", nodes=json.dumps(labels, ensure_ascii=False),
                         cues=json.dumps(cues), text_px=size, tag_height_px=tag_height,
                         tag_max_width_px=tag_width, panel_max_height_px=stack_height,
                         panel_width_px=layer_width, arrow_gap_px=FLOW_TAGS_ARROW_PX,
                         pad_x_px=8, pad_top_px=pad_top,
                         pad_bottom_px=height - stack_height - pad_top)
    elif template == "flow":
        labels = values["nodes"]
        count = len(labels)
        # Floor pixel geometry and center the row to absorb the division remainder.
        box_width = math.floor((body_width - (count - 1) * FLOW_ARROW_GAP) / count)
        box_height = min(FLOW_BOX_HEIGHT, panel_height - 2 * CONTENT_PAD)
        label_width = box_width - 2 * FLOW_LABEL_PAD_X
        size = min(
            _fit_measured("nodes", label, label_width, graphics["font"], 600, FLOW_MAX_TEXT_PX)
            for label in labels
        )
        size = min(size, math.floor((box_height - 2 * FLOW_LABEL_PAD_Y) / FLOW_LINE_HEIGHT))
        if size < MIN_TEXT_PX:
            raise ValidationError("slots.nodes", "raise panel height to fit labels at 30 px")
        _cue_window(
            values, duration, count, reserve=FLOW_NODE_REVEAL_S + FLOW_EXPLICIT_REST_S,
            unit="node",
        )
        # Default node i starts at first + i * gap, where first is the panel
        # entrance (0.3 s) plus the arrow lead (0.24 s). latest is
        # out_start - 0.65 - 0.6, capped at duration * 0.6 - 0.65, and gap is
        # (latest - first) / (count - 1). Tighten cue spacing for the preview
        # while keeping each box trace and label fade at their readable durations.
        # Partial cues clamp following defaults to the previous cue. This keeps
        # them ordered within the explicit window, which reserves a 0.3 s rest.
        first = FLOW_PANEL_IN_S + FLOW_ARROW_DRAW_S
        out_start = duration - min(0.3, duration * 0.15)
        latest = out_start - FLOW_NODE_REVEAL_S - FLOW_REST_S
        latest = min(latest, duration * 0.6 - FLOW_NODE_REVEAL_S)
        gap = (latest - first) / (count - 1)
        cues = []
        for index in range(count):
            cue = (values["cues"][index] if index < len(values["cues"])
                   else max(cues[-1] if cues else 0, min(latest, first + index * gap)))
            cues.append(cue)
        variables.update(
            nodes=json.dumps(labels, ensure_ascii=False), cues=json.dumps(cues), text_px=size,
            box_width_px=box_width, box_height_px=box_height, arrow_gap_px=FLOW_ARROW_GAP,
        )
    elif template == "choices":
        options = values["options"]
        fitted = False
        for wrap in (False, True):
            for size in range(CHOICES_MAX_TEXT_PX, CHOICES_MIN_TEXT_PX - 1, -1):
                glow_pad = math.ceil(0.5 * CHOICES_SCALE * size)
                line_limit = min(864, max_width_px - 2 * glow_pad
                                 if max_width_px is not None else 864)

                def measure(text: str, size: int = size) -> float:
                    return text_width_em(text, graphics["font"], 600) * CHOICES_SCALE * size

                lines = []
                for index, text in enumerate(options):
                    prefix = f"{chr(65 + index)}) " if values["letters"] else ""
                    label = prefix + text
                    if measure(label) <= min(648, line_limit):
                        lines.append([label])
                        continue
                    if not wrap:
                        break
                    words = text.split()
                    splits = [[prefix + " ".join(words[:i]), " ".join(words[i:])]
                              for i in range(1, len(words))]
                    splits = [pair for pair in splits
                              if max(map(measure, pair)) <= line_limit]
                    if not splits:
                        break
                    lines.append(min(splits, key=lambda pair: abs(
                        measure(pair[0]) - measure(pair[1]))))
                if len(lines) != len(options):
                    continue
                row_heights = [round(len(row) * size * CHOICES_LINE_HEIGHT, 6) for row in lines]
                overshoot = (CHOICES_SCALE - 1) / 2
                # Either neighbour may land while the other is settled.
                row_gaps = [math.ceil(overshoot * max(first, second) + CHOICES_ROW_GAP_PX)
                            for first, second in zip(row_heights, row_heights[1:], strict=False)]
                stack_top = math.ceil(glow_pad + overshoot * row_heights[0])
                extent = round(stack_top + sum(row_heights) + sum(row_gaps)
                               + overshoot * row_heights[-1] + glow_pad, 6)
                stack_height = math.ceil(extent / 2) * 2
                if max_height_px is None or stack_height <= max_height_px:
                    fitted = True
                    break
            if fitted:
                break
        else:
            if len(lines) != len(options):
                raise ValidationError("slots.options", f"shorten option {chr(65 + len(lines))} "
                                      "to fit two lines at a readable size")
            raise ValidationError("slots.options", "use fewer or shorter options, or another "
                                  "moment with more room above the head, to fit at 56 px")
        width = math.ceil((max(measure(line) for row in lines for line in row)
                           + 2 * glow_pad) / 4) * 4 - 2 * PAD_X
        height = stack_height
        cues = _choices_cues(values["cues"], len(options), duration, fps)
        text_color, glow_color = glow_colors(graphics["accent_color"])
        variables.update(
            options=json.dumps(options, ensure_ascii=False), cues=json.dumps(cues),
            lines=json.dumps(lines, ensure_ascii=False), stack_top_px=stack_top,
            letters=values["letters"], text_px=size, scale=CHOICES_SCALE,
            glow_color=glow_color, text_color=text_color, glow_pad_px=glow_pad,
            row_heights=json.dumps(row_heights), row_gaps=json.dumps(row_gaps),
            line_height=CHOICES_LINE_HEIGHT, land_s=CHOICES_LAND_S, hold_s=CHOICES_HOLD_S,
            settle_s=CHOICES_SETTLE_S, rest_s=CHOICES_REST_S, out_s=CHOICES_OUT_S,
            fade_in_s=2 / 30,
        )
    elif template == "chart":
        bars = values["bars"]
        count = len(bars)
        label_budget = math.floor(body_width * 0.4)
        value_budget = math.floor(body_width * 0.25)
        label_px = min(
            _fit_measured("bars", bar["label"], label_budget, graphics["font"], 600, 36)
            for bar in bars
        )
        value_px = min(
            _fit_measured("bars", f'{bar["value"]}{values["unit"]}', value_budget,
                          graphics["font"], 700, 32)
            for bar in bars
        )
        if count == 5:
            label_px = min(label_px, 30)
            value_px = min(value_px, 30)
        label_width = math.ceil(max(
            text_width_em(bar["label"], graphics["font"], 600) * label_px for bar in bars
        )) + 2
        value_width = math.ceil(max(
            text_width_em(f'{bar["value"]}{values["unit"]}', graphics["font"], 700)
            * value_px for bar in bars
        )) + 2
        track_width = math.floor(body_width - label_width - value_width - 32)
        if track_width < 180:
            raise ValidationError("slots.bars", "shorten labels or unit to leave room for bars")
        peak = max(bar["value"] for bar in bars)
        planned_bars = [
            {**bar, "length_px": round(track_width * bar["value"] / peak) if peak else 0}
            for bar in bars
        ]
        title_px = _fit_measured("title", values["title"], body_width,
                                 graphics["font"], 700, 30 if count == 5 else 42)
        row_height = max(label_px, value_px) + 5
        if panel_height - 32 - title_px * 1.15 - 6 < count * row_height + (count - 1) * 3:
            raise ValidationError("slots.bars", "remove bars or shorten labels to fit the panel")
        variables.update(title=values["title"], unit=values["unit"],
                         bars=json.dumps(planned_bars, ensure_ascii=False), title_px=title_px,
                         label_px=label_px, value_px=value_px,
                         label_width_px=label_width, value_width_px=value_width,
                         track_width_px=track_width, row_height_px=row_height)
    elif template == "quote":
        quote_px = _fit_quote(values["quote"], body_width - 106, graphics["font"])
        name_px = _fit_measured("name", values["name"], body_width - 100,
                                graphics["font"], 700, 30)
        role_px = _fit_measured("role", values["role"], body_width - 100,
                                graphics["font"], 400, 30)
        variables.update(values, quote_px=quote_px, name_px=name_px, role_px=role_px)
    elif template == "progress_stat":
        label_px = _fit_measured("label", values["label"], body_width,
                                 graphics["font"], 600, 40)
        value_px = _fit_measured("unit", f'{values["value"]}{values["unit"]}',
                                 body_width, graphics["font"], 700, 108)
        variables.update(values, label_px=label_px, value_px=value_px)
    elif template == "specs_list":
        rows = values["rows"]
        count = len(rows)
        if row_card:
            glow = row_look == "on_cue_glow"
            row_min, row_max = (48, 56) if glow else (44, 52)
            labels = [(f"{row['name']}: {row['value']}",
                       text_width_em(row["name"], graphics["font"], 600)
                       + text_width_em(row["value"], graphics["font"], 700)) for row in rows]
            row_px = _readable_row_size(labels, body_width - 20, row_min, row_max)
            title_min, title_max = (72, 90) if glow else (52, 60)
            title_units = text_width_em(values["title"], graphics["font"], 700)
            title_max = min(title_max, math.floor(body_width / max(title_units, 0.01)))
            if title_max < title_min:
                raise ValidationError("slots.title", "shorten the title to fit one line "
                                      f"at {title_min} px, or widen the card")
            fitted = False
            for title_px in range(title_max, title_min - 1, -1):
                for fitted_row in range(row_px, row_min - 1, -1):
                    content = (title_px * (1.1 if glow else 1.05) + 4
                               + count * (fitted_row * (1.05 if glow else 1.15)
                                          + (0 if glow else 2))
                               + (count - 1) * (3 if glow else 4) + (0 if glow else 24))
                    panel_height = math.ceil(content)
                    top_pad, bottom_pad = _row_padding(glow, title_px, variables["card_shadow"])
                    height = math.ceil((panel_height + top_pad + bottom_pad) / 2) * 2
                    if max_height_px is None or height <= max_height_px:
                        row_px = fitted_row
                        fitted = True
                        break
                if fitted:
                    break
            if not fitted:
                raise ValidationError("slots.rows", "use fewer or shorter rows, or another "
                                      "moment with more room above the head, to fit "
                                      "at a readable size")
            variables.update(title=values["title"], rows=json.dumps(rows, ensure_ascii=False),
                             title_px=title_px, row_px=row_px, panel_max_height_px=panel_height,
                             pad_top_px=top_pad, pad_bottom_px=bottom_pad, glow_pad_px=top_pad,
                             cues=json.dumps(_row_cues(values, count, duration)))
        else:
            label_width = (body_width - 36) // 2
            row_px = min(
                _fit_measured("rows", entry[key], label_width, graphics["font"], 600, 36)
                for entry in rows for key in ("name", "value")
            )
            title_px = min(
                _fit_measured("title", values["title"], body_width,
                              graphics["font"], 700, 40),
                30 if count == 5 else 40,
            )
            row_px = min(row_px, math.floor(
                (panel_height - 32 - 4 - title_px * 1.1 - (count - 1) * 3 - count * 2) / count
            ))
            if row_px < MIN_TEXT_PX:
                raise ValidationError("slots.rows", "remove rows or shorten values to fit at 30 px")
            variables.update(title=values["title"], rows=json.dumps(rows, ensure_ascii=False),
                             title_px=title_px, row_px=row_px)
    elif template == "callout_circle":
        image = values.pop("image")
        x, y, radius = (values[key] for key in ("x", "y", "r"))
        image_width, image_height = image_sizes["image"]
        radius_px = radius / 100 * min(image_width, image_height)
        center_x = x / 100 * image_width
        center_y = y / 100 * image_height
        if (center_x - radius_px < 0 or center_x + radius_px > image_width
                or center_y - radius_px < 0 or center_y + radius_px > image_height):
            raise ValidationError("slots.r", "keep the circle inside the image")
        scale = min(width / image_width, panel_height / image_height)
        display_width = math.floor(image_width * scale)
        display_height = math.floor(image_height * scale)
        # The absolute label is limited by its 12 px image inset on each side.
        # Its border box also includes 10 px padding on each side and a 6 px border.
        label_px = _fit_measured("label", values["label"], display_width - 24 - 20 - 6,
                                 graphics["font"], 700, 36)
        assets = (("image" + image.suffix.lower(), image),)
        variables.update(values, image_src=assets[0][0], image_width_px=display_width,
                         image_height_px=display_height, label_px=label_px)
    elif template == "name_tag":
        name_px = _fit_measured("name", values["name"], body_width - 60,
                                graphics["font"], 700, 56)
        role_px = _fit_measured("role", values["role"], body_width - 60,
                                graphics["font"], 400, 34)
        variables.update(values, name_px=name_px, role_px=role_px)
    elif template == "success_check":
        label_px = _fit_measured("label", values["label"], body_width - 110,
                                 graphics["font"], 700, 52)
        variables.update(values, label_px=label_px)
    elif template == "edit_timeline":
        from yap_studio.editor_cards import timeline_variables

        variables.update(timeline_variables(
            values, project if project is not None else sample_project(template, frames, fps),
            frames=frames, fps=fps, width=width, height=panel_height, font=graphics["font"],
            mono=graphics["font"] if branded_font else MONO_FONT,
        ))
    elif template == "storyboard":
        from yap_studio.editor_cards import storyboard_variables

        variables.update(storyboard_variables(
            values, project if project is not None else sample_project(template, frames, fps),
            frames=frames, fps=fps, width=width, height=panel_height,
        ))
    elif template == "jiggle_delete":
        from yap_studio.jiggle_cards import jiggle_variables

        variables.update(jiggle_variables(
            values, frames=frames, fps=fps, width=width, height=panel_height,
            font=graphics["font"],
        ))
        assets = tuple((f"item-{index + 1}{image.suffix.lower()}", image)
                       for index, image in enumerate(values["images"]))
    elif template == "file_drop":
        from yap_studio.file_cards import FILE_DROP_SETTLE_S, file_variables

        info = image_sizes["clip"]
        variables.update(file_variables(values, info, frames=frames, fps=fps, width=width,
                                        height=panel_height, font=graphics["font"],
                                        mono=graphics["font"] if branded_font else MONO_FONT,
                                        shadow=variables["card_shadow"]))
        video_duration = Fraction(str(info.video_duration_s))
        video_fps = info.fps or Fraction(fps)
        last_frame = max(0, math.ceil(video_duration * video_fps) - 1)
        poster_time = min(video_duration * Fraction(str(values["poster_at"])) / 100,
                          Fraction(last_frame) / video_fps)
        poster = CardPoster(values["clip"], str(poster_time))
        if values["playback"] == "play":
            start = math.ceil((Fraction(str(values["drop_at"]))
                               + Fraction(str(FILE_DROP_SETTLE_S))) * fps)
            playing = frames - start
            if poster_time + Fraction(playing, fps) > video_duration:
                latest = (video_duration - Fraction(playing, fps)) / video_duration * 100
                limit = f"{math.floor(latest * 1_000_000) / 1_000_000:.6f}".rstrip("0").rstrip(".")
                fix = (f"use poster_at at most {limit} or a shorter card"
                       if latest >= 0 else "use a shorter card")
                raise ValidationError("slots.poster_at", fix)
            fade_out = max(1, min(9, frames * 15 // 100))
            window = {"slot": "clip", "index": 0,
                      "x": PAD_X + (width - variables["file_panel_width_px"]) // 2 + 24,
                      "y": PAD_TOP + (panel_height - variables["poster_height_px"]) // 2,
                      "width": variables["poster_width_px"],
                      "height": variables["poster_height_px"],
                      "radius": min(8, variables["poster_width_px"] / 2,
                                    variables["poster_height_px"] / 2),
                      "start": start, "frames": playing, "fade_in": 2, "fade_out": fade_out,
                      "source": {"kind": "media", "path": str(values["clip"]), "sha256": "",
                                 "in_s": float(poster_time), "speed": 1,
                                 "focus": [0.5, 0.5], "audio": "mute"}}
            windows = (window,)
            variables["play_window"] = json.dumps({
                "x": 24, "y": (variables["panel_height_px"] - window["height"]) // 2,
                "width": window["width"], "height": window["height"], "radius": window["radius"],
                "clear_at_s": (start + window["fade_in"] - 0.5) / fps,
                "fade_out_s": fade_out / fps,
                "panel_left_px": window["x"] - 24,
                "panel_top_px": PAD_TOP + (panel_height - variables["panel_height_px"]) // 2,
            })
    elif template == "prompt_bar":
        schedule = _typing_schedule(
            values["request"], values["cadence"], values["type_at"],
            values["word_cues"], values["cues"], duration,
        )
        text_width = width - 2 * 36 - 76 - 24
        size = min(
            _fit_measured(name, values[name], text_width - FIELD_CARET_MIN_PX,
                          graphics["font"], 500, 52)
            for name in ("request", "placeholder")
        )
        widest = max(text_width_em(values[name], graphics["font"], 500)
                     for name in ("request", "placeholder"))
        size = min(size, math.floor(text_width / (widest + FIELD_CARET_EM)))
        variables.update(request=values["request"], placeholder=values["placeholder"],
                         text_px=size, schedule=json.dumps(schedule))
    elif template == "comment_cta":
        schedule = _typing_schedule(
            values["keyword"], values["cadence"], values["type_at"],
            values["word_cues"], values["cues"], duration,
        )
        minimum_height = (
            2 * CONTENT_PAD + 96 + (16 + MIN_TEXT_PX * 1.2 if values["label"] else 0)
        )
        if panel_height < minimum_height:
            raise ValidationError(
                "slots.label" if values["label"] else "slots.keyword",
                f"raise panel height to at least {minimum_height:g} px"
                + (" or remove slots.label" if values["label"] else ""),
            )
        text_width = body_width - 2 * (28 + 2) - 64 - 16
        size = min(
            _fit_measured(name, values[name], text_width - FIELD_CARET_MIN_PX,
                          graphics["font"], 500, 44)
            for name in ("keyword", "placeholder")
        )
        widest = max(text_width_em(values[name], graphics["font"], 500)
                     for name in ("keyword", "placeholder"))
        size = min(size, math.floor(text_width / (widest + FIELD_CARET_EM)))
        label_px = _fit_measured("label", values["label"], body_width,
                                 graphics["font"], 700, 40)
        if values["label"]:
            label_px = min(label_px, math.floor(
                (panel_height - 2 * CONTENT_PAD - 96 - 16) / 1.2
            ))
        variables.update(keyword=values["keyword"], placeholder=values["placeholder"],
                         label=values["label"], text_px=size, label_px=label_px,
                         schedule=json.dumps(schedule))
    elif template == "comment_drop":
        text = values["text"]
        if text.startswith(" ") or text.endswith(" ") or "  " in text:
            raise ValidationError(
                "slots.text", "use single spaces between words, with none at the start or end",
            )
        out_start = duration - min(0.3, duration * 0.15)
        latest = out_start - 0.6 - 0.6
        if not 0 <= values["drop_at"] <= latest:
            raise ValidationError(
                "slots.drop_at", f"use seconds within 0..{latest:g}; leave 0.6 s to land "
                "and 0.6 s of rest before the fade-out",
            )
        avatar = values["avatar"]
        initials = values["initials"]
        if not avatar and not initials:
            logo = style.data.get("brand", {}).get("logo", "")
            if logo:
                if _has_controls(logo):
                    raise ValidationError(
                        "brand.logo", "remove control characters from the image path",
                    )
                try:
                    logo_path = Path(logo).expanduser()
                except RuntimeError as exc:
                    raise ValidationError(
                        "brand.logo", "set an absolute path to a readable image",
                    ) from exc
                if not logo_path.is_absolute():
                    raise ValidationError("brand.logo", "set an absolute path to a readable image")
                try:
                    validated, _ = _slots({"avatar": logo_path},
                                          {"avatar": metadata["slots"]["avatar"]})
                except ValidationError as exc:
                    raise ValidationError("brand.logo", exc.detail) from exc
                avatar = validated["avatar"]
            else:
                initials = "".join(word[0] for word in values["name"].split()[:2]).upper()
        if vocabulary is not None:
            initials = bleep.mask_text(initials, vocabulary)
        if not avatar and not initials:
            raise ValidationError(
                "slots.initials",
                "set slots.initials or slots.name, pass slots.avatar, or set brand.logo",
            )
        column_width = width - 2 * 28 - 72 - 20
        text_px, lines = _fit_comment(text, column_width, graphics["font"])
        _fit_measured("name", values["name"], column_width, graphics["font"], 700, 30)
        content_width = max(
            text_width_em(values["name"], graphics["font"], 700) * 30,
            *(text_width_em(line, graphics["font"], 500) * text_px for line in lines),
        )
        bubble_height = 48 + max(72, len(lines) * text_px * 1.2 +
                                 (36 if values["name"] else 0))
        if bubble_height + 16 > panel_height:
            raise ValidationError("slots.text", f"raise panel height to at least "
                                  f"{math.ceil(bubble_height + 16)} px or shorten the comment")
        if avatar:
            assets = (("avatar" + avatar.suffix.lower(), avatar),)
        variables.update(
            text=text, name=values["name"], initials=initials,
            avatar_src=assets[0][0] if assets else "", side=values["side"],
            drop_at=values["drop_at"], text_px=text_px, lines=json.dumps(lines),
            bubble_width_px=math.ceil(56 + 72 + 20 + content_width),
        )
    elif template == "icon_strike":
        out_start = duration - min(0.3, duration * 0.15)
        latest = out_start - 0.28 - 0.3
        if not 0.6 <= values["strike_at"] <= latest:
            raise ValidationError(
                "slots.strike_at", f"use seconds within 0.6..{latest:g}; leave 0.28 s "
                "for the strike and 0.3 s of rest before the fade-out, or raise graphics.hold_s",
            )
        label_px = (_fit_measured("label", values["label"], body_width - 160 - 28,
                                  graphics["font"], 700, 56) if values["label"] else 56)
        if width < 208:
            raise ValidationError("graphics.card_width_px",
                                  "raise graphics.card_width_px to at least 208 px")
        if panel_height < 208:
            raise ValidationError("slots.label", "raise panel height to at least 208 px")
        icon = values["icon"]
        if icon:
            assets = (("icon" + icon.suffix.lower(), icon),)
        accent = graphics["accent_color"]
        strike_color = (accent if _contrast(accent, graphics["titlebar_color"]) >= 3
                        else variables["bar_ink_color"])
        variables.update(
            glyph=values["glyph"], label=values["label"], strike_at=values["strike_at"],
            icon_src=assets[0][0] if assets else "", label_px=label_px, strike_color=strike_color,
            content_width_px=math.ceil(48 + 160 + (28 + text_width_em(
                values["label"], graphics["font"], 700,
            ) * label_px if values["label"] else 0)),
        )
    elif template == "gravity_drop":
        out_start = duration - min(0.3, duration * 0.15)
        for index, drop in enumerate(values["drops"]):
            if drop["at"] > out_start - 0.5:
                raise ValidationError(f"slots.drops[{index}].at",
                                      "move the landing earlier or extend the card; "
                                      "leave at least 0.5 s before the fade-out")
        size, drops = _gravity_layout(values["drops"], style, width + 2 * PAD_X, height)
        captions = style.data["captions"]
        variables.update(drops=json.dumps(drops, ensure_ascii=False), text_px=size,
                         ink_color=captions["text_color"],
                         text_shadow=f'{captions["shadow_x_px"]}px {captions["shadow_y_px"]}px '
                                     f'{captions["shadow_blur_px"]}px '
                                     f'rgb(0 0 0 / {captions["shadow_opacity"]})')
    elif template == "image_card":
        if values["reveal"] == "collage":
            out_start = duration - min(0.3, duration * 0.15)
            latest = out_start - COLLAGE_SPAN_S - COLLAGE_REST_S
            if not 0 <= values["reveal_at"] <= latest:
                raise ValidationError(
                    "slots.reveal_at",
                    f"use seconds within 0..{latest:g}; leave "
                    f"{COLLAGE_SPAN_S:g} s for the lift and {COLLAGE_REST_S:g} s of rest "
                    f"before the fade-out starts at {out_start:g} s",
                )
        image = values.pop("image")
        assets = (("image" + image.suffix.lower(), image),)
        _bar_fit(values["title"], width, mono=values["chrome"] == "browser")
        variables.update(
            values,
            image_src="image" + image.suffix.lower(),
            bar_px=MIN_TEXT_PX,
            image_max_width_px=width - 2 * IMAGE_FRAME_BORDER,
            image_max_height_px=panel_height - BAR_HEIGHT - 2 * IMAGE_FRAME_BORDER,
            collage_span_s=COLLAGE_SPAN_S,
            collage_exit_s=COLLAGE_EXIT_S,
        )
    if template == "video_card":
        clip = values["clip"]
        _clip_bounds("clip", clip, frames, fps)
        aspect = values["aspect"]
        ratio = clip_aspect(clip, minimum=9 / 16) if aspect == "source" else (
            float(aspect.split(":")[0]) / float(aspect.split(":")[1]))
        mat = 16
        window_width = (width - 2 * mat) // 2 * 2
        window_height = int(window_width / ratio) // 2 * 2
        label = values["label"]
        label_px = _fit_measured("label", label, window_width - 16, graphics["font"], 600, 40)
        label_height = math.ceil((label_px * 1.2 + 16) / 2) * 2 if label else 0
        panel_height = window_height + 2 * mat + label_height
        height = panel_height + PAD_TOP + PAD_BOTTOM
        window = {"slot": "clip", "index": 0, "x": PAD_X + mat, "y": PAD_TOP + mat,
                  "width": window_width, "height": window_height,
                  "radius": max(6, _LOOKS[graphics["style"]][0] - mat),
                  "start": 0, "frames": frames, "fade_in": max(1, min(9, frames // 4)),
                  "fade_out": max(1, min(9, frames * 15 // 100)),
                  "source": {"kind": "media", **clip}}
        windows = (window,)
        variables.update(panel_max_height_px=panel_height, label=label, label_px=label_px,
                         rule_color=style.data["brand"]["accent_color"] or "",
                         windows=json.dumps([{key: value for key, value in window.items()
                                              if key not in {"slot", "index", "source"}}]),
                         window_fill="clear", fps=fps)
    if template in {"specs_list", "stat"} and row_look != "together":
        variables["specs_rows"] = row_look
        if row_look == "on_cue_glow":
            variables["text_color"], variables["glow_color"] = glow_colors(graphics["accent_color"])
    if template == "playing_grid":
        clips, labels = values["clips"], values["labels"]
        if len(labels) > len(clips):
            raise ValidationError("slots.labels", "do not supply more labels than clips")
        aspects = [clip_aspect(clip) for clip in clips]
        if tiles is None:
            tiles, _ = grid_layout(aspects, None, margin=graphics["face_clearance_px"],
                                   ceiling=caption_ceiling(style) if ceiling is None else ceiling,
                                   top_bottom=internal["key_text_bottom_px"], labels=labels,
                                   id="preview")
        if (len(tiles) != len(clips) or any(
            set(tile) != {"x", "y", "width", "height"}
            or any(type(value) is not int or value % 2 for value in tile.values())
            or tile["x"] < 0 or tile["y"] < 0 or tile["width"] < 200
            or tile["x"] + tile["width"] > 1080 or tile["y"] + tile["height"] > 1920
            for tile in tiles
        )):
            raise ValidationError("tiles", "use one even reserved box per clip inside the frame")
        fade_out = max(1, min(9, frames * 15 // 100))
        windows = []
        for index, (clip, aspect, tile) in enumerate(zip(clips, aspects, tiles, strict=True)):
            start, window_frames = grid_clip_window(index, frames, values, fps=fps)
            if start + 8 + fade_out > frames:
                raise ValidationError("slots.stagger_s", "leave eight frames for each entrance "
                                      "and the final fade-out; use less stagger or a longer hold")
            rect = tile_window(tile, aspect)
            needed = (rect["height"] + 2 * GRID_RING + GRID_PAD_TOP + GRID_PAD_BOTTOM
                      + (GRID_CHIP_HEIGHT if index < len(labels) and labels[index] else 0))
            if tile["height"] < needed:
                raise ValidationError("tiles", "reserve the ring, chip and shadow inside each tile")
            if index < len(labels):
                _fit_measured(f"labels[{index}]", labels[index], rect["width"] - 16,
                              graphics["font"], 600, 30)
            _clip_bounds(f"clips[{index}]", clip, window_frames, fps)
            windows.append({"slot": "clips", "index": index, **rect,
                            "radius": max(6, _LOOKS[graphics["style"]][0] - 6),
                            "start": start, "frames": window_frames, "fade_in": 8,
                            "fade_out": fade_out, "source": {"kind": "media", **clip}})
        windows = tuple(windows)
        variables.update(tiles=json.dumps(list(tiles)), labels=json.dumps(labels), chip_px=30,
                         windows=json.dumps([{key: value for key, value in window.items()
                                              if key not in {"slot", "index", "source"}}
                                             for window in windows]), window_fill="clear", fps=fps)
    if template == "reel_feed":
        backdrop = values["backdrop"]
        caption_color = style.data["captions"]["text_color"]
        if backdrop == "auto":
            backdrop = max(("paper", "ink"), key=lambda name: _contrast(
                caption_color, graphics[f"{name}_color"],
            ))
            if _contrast(caption_color, graphics[f"{backdrop}_color"]) < 3:
                raise ValidationError("slots.backdrop", "captions would be unreadable on paper "
                                      "or ink; use backdrop dim or change the caption, paper "
                                      "or ink colour")
        elif backdrop == "paper" and _contrast(caption_color, graphics["paper_color"]) < 3:
            raise ValidationError("slots.backdrop", "captions would be unreadable on paper; "
                                  "use backdrop auto or ink")
        post = values["post"]
        ending = values["ending"]
        if ending == "handoff" and post != "self":
            raise ValidationError("slots.ending", "handoff continues this video; "
                                  "use post self or ending fade")
        _clip_bounds("post", post, frames, fps)
        rect = feed_window(caption_ceiling(style) if ceiling is None else ceiling)
        caption_px, lines = _feed_lines(values["caption"], rect["width"] - 48, graphics["font"])
        handle = values["handle"]
        initials = "".join(word[0] for word in handle.lstrip("@").split()[:2])
        if vocabulary is not None:
            initials = bleep.mask_text(initials, vocabulary)
        handle_px = _fit_measured("handle", handle, rect["width"] - 128,
                                  graphics["font"], 600, 34)
        views = f"{values['views']} views" if values["views"] else ""
        if vocabulary is not None:
            views = bleep.mask_text(views, vocabulary)
        views_px = _fit_measured("views", views, rect["width"] - 102, graphics["font"], 600, 30)
        window = {"slot": "post", "index": 0, **rect, "start": 0, "frames": frames,
                  "fade_in": max(1, min(9, frames // 4)),
                  "fade_out": max(1, min(9, frames * 15 // 100)),
                  "source": {"kind": "self"} if post == "self" else {"kind": "media", **post}}
        if ending == "handoff":
            window.update(fade_out=0, motion={"start": frames - 18, "frames": 18,
                                             "to": [0, 0, 1080, 1920],
                                             "ease": "cubic_in_out"})
        windows = (window,)
        template_window = {key: value for key, value in window.items()
                           if key not in {"slot", "index", "source"}}
        if ending == "handoff":
            template_window["rects"] = [window_rect(window, frame)
                                         for frame in range(frames - 18, frames)]
        variables.update(handle=handle, handle_px=handle_px, initials=initials,
                         caption_lines=json.dumps(lines), caption_px=caption_px,
                         views_text=views, views_px=views_px, backdrop=backdrop, ending=ending,
                         progress_color=(style.data["brand"]["accent_color"]
                                         or graphics["ink_color"]),
                         overlay_ink=max((graphics["paper_color"], graphics["ink_color"]),
                                         key=_luminance),
                         sound_on=post == "self" or post["audio"] != "mute",
                         windows=json.dumps([template_window]),
                         window_fill="clear", fps=fps)
    full_frame = template in {"playing_grid", "reel_feed"}
    layer_width = width + 2 * (variables["pad_x_px"] if row_card else PAD_X)
    from yap_studio.card_motion import preview_motion

    return preview_motion(CardPlan(
        template,
        0 if full_frame else (1080 - layer_width) // 2,
        0 if full_frame else (
            math.ceil((internal["key_text_top_px"] - variables["glow_pad_px"]) / 2) * 2
            if template == "choices" or row_card else internal["key_text_top_px"]),
        1080 if full_frame else layer_width,
        1920 if full_frame else height,
        frames,
        fps,
        variables,
        (gravity_drop_fonts(style) if template == "gravity_drop" else
         tuple(dict.fromkeys((graphics["font"],) if branded_font and template != "terminal" else
                             (graphics["font"], MONO_FONT)))),
        assets,
        poster=poster,
        windows=windows,
    ), style)


def grid_card_plan(
    slots: Mapping[str, object], style: ResolvedStyle, faces: Sequence[Sequence] | None, *,
    frames: int, fps: int = 30, ceiling: int | None = None, layer: str = "front",
    reason: str = "faces not measured", id: str = "preview",
    face_windows: tuple[tuple[int, int], ...] = (), start_frame: int = 0,
) -> tuple[CardPlan, dict]:
    """Fit the declared clips around measured heads before fitting their chips."""
    values, _ = _slots(slots, load_template("playing_grid")["slots"])
    if len(values["labels"]) > len(values["clips"]):
        raise ValidationError("slots.labels", "do not supply more labels than clips")
    ceiling = caption_ceiling(style) if ceiling is None else ceiling
    tiles, clearance = grid_layout(
        [clip_aspect(clip) for clip in values["clips"]], faces,
        margin=style.data["graphics"]["face_clearance_px"], ceiling=ceiling,
        top_bottom=style.data["internal"]["key_text_bottom_px"], labels=values["labels"],
        layer=layer, reason=reason, id=id, face_windows=face_windows, start_frame=start_frame,
    )
    return card_layer_plan("playing_grid", slots, style, frames=frames, fps=fps,
                           tiles=tiles, ceiling=ceiling), clearance


class _RootParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.root: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if dict(attrs).get("id") == "root":
            self.root = self.get_starttag_text()


def prepare_card(plan: CardPlan, work_dir: Path, *, install: RendererInstall | None = None) -> Path:
    """Write a composition with bundled fonts and atomically copied local assets."""
    if registry_cards.is_kind(plan.template):
        composition = registry_cards.prepare(plan, work_dir, install=install)
        if plan.motion:
            from yap_studio.card_motion import prepare_registry

            composition = prepare_registry(plan, composition, install=install)
        return composition
    if plan.template in {"prompt_bar", "comment_cta"}:
        name = "request" if plan.template == "prompt_bar" else "keyword"
        try:
            schedule = json.loads(plan.variables["schedule"])
            matches = schedule["text"] == plan.variables[name]
        except (KeyError, TypeError, json.JSONDecodeError):
            matches = False
        if not matches:
            raise ValidationError(
                "schedule", f"schedule must come from card_layer_plan for the same {name}",
            )
    if plan.template == "terminal":
        try:
            schedule = json.loads(plan.variables["schedule"])
            matches = (
                schedule["command"] == plan.variables["command"]
                and schedule["output"] == json.loads(plan.variables["output"])
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            matches = False
        if not matches:
            raise ValidationError(
                "schedule",
                "schedule must come from card_layer_plan for the same command and output",
            )
    with paths.operation():
        if plan.template == "icon_strike" and plan.variables["icon_src"] and not re.fullmatch(
            r"icon\.(png|jpe?g|webp)", str(plan.variables["icon_src"])
        ):
            raise ValidationError("slots.icon", "use a local icon filename")
        if plan.template == "comment_drop" and plan.variables["avatar_src"] and not re.fullmatch(
            r"avatar\.(png|jpe?g|webp)", str(plan.variables["avatar_src"])
        ):
            raise ValidationError("slots.avatar", "use a local avatar filename")
        if plan.template in {"image_card", "callout_circle"} and not re.fullmatch(
            r"image\.(png|jpe?g|webp)", str(plan.variables["image_src"])
        ):
            raise ValidationError("slots.image", "use a local image filename")
        sources = plan.assets
        if plan.poster is not None:
            poster = plan.poster
            source = cached_frame(poster.cache_root or work_dir, poster.source,
                                   poster.sha256,
                                   source_time=poster.time_s,
                                   width=plan.variables["poster_width_px"] * 2,
                                   height=plan.variables["poster_height_px"] * 2,
                                   source_kind="clip")
            sources = (*sources, ("poster.jpg", source))
        assets = []
        for filename, asset in sources:
            if not re.fullmatch(
                r"(image|before|after|avatar|icon|item-[1-6])\.(png|jpe?g|webp)"
                r"|shot-0[1-7]\.jpg|poster\.(jpg|png)",
                filename,
            ):
                raise ValidationError(
                    "assets",
                    "use a local image, before, after, avatar, icon, shot, item or poster "
                    "filename",
                )
            assets.append((filename, paths.validate_input(asset)))
        source = template_html(plan.template, motion=bool(plan.motion))
        composition = work_dir / plan.template
        paths.validate_output_dir(composition)
        parser = _RootParser()
        parser.feed(source)
        if parser.root is None:
            raise ValidationError("template", "composition root is missing")
        root = parser.root
        if plan.motion:
            # The live player consumes clip timing before the inner script runs.
            root = (root[:-1] + ' data-card-motion="streak" data-card-duration="'
                    + str(hyperframes.layer_duration_s(plan.frames, plan.fps)) + '">')
        for attribute, value in (
            ("width", plan.width),
            ("height", plan.height),
            ("duration", hyperframes.layer_duration_s(plan.frames, plan.fps)),
        ):
            root = re.sub(rf'data-{attribute}="[^"]*"', f'data-{attribute}="{value}"', root)
        source = source.replace(parser.root, root, 1)
        source = source.replace("/* BUNDLED_FONT_FACES */", fonts.font_face_css(plan.fonts))
        fonts.copy_fonts(plan.fonts, composition / "fonts")
        for filename, image in assets:
            with paths.atomic_output(composition / filename) as temporary:
                shutil.copyfile(image, temporary)
        paths.atomic_write_text(composition / "index.html", source)
        if plan.motion:
            from yap_studio.card_motion import prepare_builtin

            composition = prepare_builtin(plan, composition, install=install)
    return composition


def render_card(
    plan: CardPlan,
    work_dir: Path,
    out: Path,
    *,
    install: RendererInstall | None = None,
) -> LayerResult:
    """Prepare and render a validated card through the pinned layer renderer."""
    composition = prepare_card(plan, work_dir, install=install)
    return hyperframes.render_layer(
        composition,
        out,
        frames=plan.frames,
        variables={} if plan.motion or registry_cards.is_kind(plan.template) else plan.variables,
        fps=plan.fps,
        install=install,
        browser_gpu=hyperframes.CARD_BROWSER_GPU,
    )
