"""Conform vertical clips and composite style-driven titles and captions."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from html.parser import HTMLParser
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from yap_studio import ffmpeg, paths, receipt, transcript
from yap_studio.brand_words import list_words
from yap_studio.captions import (
    CaptionWord,
    Phrase,
    build_phrases,
    caption_layout_settings,
    caption_word_case,
    kinetic_display_times,
    kinetic_word_times,
    phrase_key,
    phrase_keys,
)
from yap_studio.errors import (
    DependencyError,
    InputError,
    OutputPathError,
    ValidationError,
    YapError,
)
from yap_studio.faces.clearance import (
    DEFAULT_HAIR,
    LAYOUT_SCHEMA,
    Layout,
    face_clearance_px,
    gap_px,
    head_box,
)
from yap_studio.faces.mapping import ZoomPiece, map_cut_faces
from yap_studio.faces.model import Box, FacesData, VideoIdentity
from yap_studio.faces.placement import Band, caption_band, suggest_offset
from yap_studio.faces.project import cut_faces, cut_faces_identity
from yap_studio.project.lock import project_lock, write_json_cas
from yap_studio.project.model import CutWords, parse_document
from yap_studio.project.receipts import StageRun, projection_sha256, read_receipt
from yap_studio.project.status import ProjectContext
from yap_studio.render import fonts, hyperframes
from yap_studio.render.layer_checks import CAPTION_SHRINK_PREFIX
from yap_studio.render.text_metrics import text_width_em
from yap_studio.sfx import bleep
from yap_studio.stages.broll_split import caption_split_band, split_layouts, split_windows
from yap_studio.stages.green_screen import (
    LAYOUT_RULE,
    caption_breaks,
    clip_layout_phrases,
    layout_windows,
)
from yap_studio.style import ResolvedStyle
from yap_studio.style.loader import _contrast
from yap_studio.text_hook import TEXT_HOOK_ID
from yap_studio.text_rules import apply_text_rules, balance_lines, near_vocabulary
from yap_studio.zoom_curve import ZoomCurve

if TYPE_CHECKING:
    from yap_studio.stages.captions_3d import CameraSnapshot
    from yap_studio.stages.timeline import ResolvedTimeline

_TEMPLATES = Path(__file__).parents[1] / "templates"
_HOOK_TOP = {"headroom": 300, "top": 150, "center": 760}
CAPTION_LINE_HEIGHT = 1.12
HALO_PATH = ".yap/caption-halo.json"
HALO_SCHEMA = "yap-studio/caption-halo@1"


def caption_halos(context: ProjectContext) -> dict[str, dict]:
    """Read the automatic halos retained by this project snapshot."""
    try:
        document = context.json(HALO_PATH)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise InputError(HALO_PATH, "cannot read caption halos") from exc
    if (not isinstance(document, dict) or document.get("schema") != HALO_SCHEMA
            or not isinstance(document.get("phrases"), dict)):
        raise InputError(HALO_PATH, "invalid caption halos")
    for key, phrase in document["phrases"].items():
        if (not key.startswith("caption:") or not isinstance(phrase, dict)
                or not isinstance(phrase.get("text"), str)
                or type(phrase.get("ratio")) not in (int, float)
                or not math.isfinite(phrase["ratio"]) or not 1 <= phrase["ratio"] <= 21
                or ("window" in phrase and (
                    not isinstance(phrase["window"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", phrase["window"]) is None))):
            raise InputError(HALO_PATH, "invalid caption halo phrase")
    return document["phrases"]


def add_caption_halos(project_dir: Path, weak: Sequence[dict],
                      windows: Mapping[str, str]) -> list[str]:
    """Refresh automatic halos for the measured picture without changing choices."""
    with project_lock(project_dir):
        context = ProjectContext(project_dir)
        previous = caption_halos(context)
        halos = {key: entry for key, entry in previous.items()
                 if key in windows and entry.get("window") == windows[key]}
        added = []
        for phrase in weak:
            key = phrase.get("key")
            if (key in windows and key not in context.project.caption_backing
                    and key not in halos):
                halos[key] = {"text": phrase["text"], "ratio": phrase["ratio"],
                              "window": windows[key]}
                added.append(key)
        if halos != previous:
            paths.atomic_write_text(project_dir / HALO_PATH, json.dumps({
                "schema": HALO_SCHEMA, "phrases": halos,
            }, indent=2) + "\n")
        return added


class CaptionRefusal(YapError):
    """Refuse to replace an existing output without verified ownership."""

    def __init__(self, path: Path, *, force: bool) -> None:
        action = "Choose a new --out"
        if not force:
            action += ", or pass --force when the file came from this command"
        super().__init__(
            f"Refusing to overwrite {path}.",
            error_type="refusal",
            code="caption_output_exists",
            suggested_action=action + ".",
        )


@dataclass(frozen=True)
class LayerPlan:
    """A composition box, placement, frame window, and declared variables."""

    template: str
    x: int
    y: int
    width: int
    height: int
    start_frame: int
    frames: int
    variables: dict
    warnings: tuple[str, ...] = ()

    def placement(self, path: str) -> dict[str, Any]:
        """Return the box and exact frame window assembly should overlay."""
        return {"path": path, "x": self.x, "y": self.y, "width": self.width,
                "height": self.height, "start_frame": self.start_frame, "frames": self.frames,
                **({"warnings": list(self.warnings)} if self.warnings else {})}


@dataclass(frozen=True)
class LayerAssets:
    """Retained template and font bytes used for both hashing and rendering."""

    html: str
    metadata: bytes
    font_css: str
    font_files: tuple[tuple[str, bytes], ...]

    def template_identity(self, *, workers: int | str,
                          browser_gpu: bool | None = None) -> dict[str, Any]:
        """Identify the pinned renderer and the template implementation it renders."""
        return {"renderer": hyperframes.HYPERFRAMES_VERSION,
                "workers": workers,
                "capture": hyperframes.RENDER_CAPTURE,
                **({"browser_gpu": browser_gpu} if browser_gpu is not None else {}),
                "version": json.loads(self.metadata)["version"],
                "html": hashlib.sha256(self.html.encode("utf-8")).hexdigest(),
                "metadata": hashlib.sha256(self.metadata).hexdigest()}

    def font_identity(self) -> dict[str, Any]:
        """Identify only the font faces and family rules used by this layer."""
        return {"css": self.font_css,
                "faces": {name: hashlib.sha256(data).hexdigest()
                          for name, data in self.font_files}}


def layer_assets(template: str, font_name: str, context: ProjectContext) -> LayerAssets:
    """Read render assets once, retaining their bytes in the project snapshot."""
    def read(target: Path) -> bytes:
        return context.read(str(paths.validate_input(target)))

    faces = tuple((name, read(source)) for name, source in fonts.font_files([font_name]))
    return LayerAssets(read(_TEMPLATES / template / "template.html").decode("utf-8"),
                       read(_TEMPLATES / template / "template.json"),
                       fonts.font_face_css([font_name]), faces)


@dataclass(frozen=True)
class CaptionBandGeometry:
    """Shared band geometry and conversion from the position's anchor line box."""

    position: str
    y: int
    height: int
    padding: int
    line_height: float
    bottom_padding: float
    warnings: tuple[str, ...]

    def top_px_from_line_box(self, line_top: float) -> float:
        """Use the last line for bottom and the first line for middle."""
        if self.position == "middle":
            return line_top + (self.height / 2 - self.padding)
        return line_top


def caption_band_geometry(
    captions: Mapping[str, Any], *, safe_bottom_px: int, backing: Sequence[str | None] = (),
) -> CaptionBandGeometry:
    """Reserve two lines, anchoring bottom text on its last line within the safe boundary."""
    padding = math.ceil(
        captions["size_px"] * 0.08
        + captions["shadow_blur_px"] * 3
        + max(abs(captions["shadow_x_px"]), abs(captions["shadow_y_px"]))
        + 8
    )
    if captions["highlight_style"] in {
        "blend_difference", "camera_follow", "editorial_emphasis", "emoji_pop", "glitch_rgb",
        "gradient_fill", "highlight", "matrix_decode", "neon_accent", "neon_glow",
        "parallax_layers", "particle_burst", "pill_karaoke", "texture",
    }:
        padding += math.ceil(captions["size_px"] * 0.25)
    if ((captions.get("background_mode", "none") != "none"
         and captions["background_opacity"] > 0) or "box" in backing):
        padding += captions["background_padding_px"]
    if "shadow" in backing:
        padding = max(padding, math.ceil(0.12 * captions["size_px"]) + 8)
    line_height = captions["size_px"] * CAPTION_LINE_HEIGHT
    rise = captions["entry_rise_px"]
    height = math.ceil(2 * line_height + rise + 2 * padding)
    safe_bottom = safe_bottom_px
    warnings = ()
    bottom_padding = padding + rise
    if captions["position"] == "middle":
        y = max(0, min(round(captions["top_px"] - height / 2), safe_bottom - height))
    else:
        line_top = min(captions["top_px"], safe_bottom - line_height - rise - padding)
        band_bottom = math.ceil(line_top + line_height + rise + padding)
        y = max(0, band_bottom - height)
        # Carry the fractional inset so rounding the band does not move the line box.
        bottom_padding = y + height - line_top - line_height
        if line_top < captions["top_px"]:
            warnings = (
                f"captions.top_px={captions['top_px']} clamped to {line_top:g}: "
                "one line plus entry rise and padding would exceed "
                f"internal.caption_safe_bottom_px={safe_bottom}.",
            )
    return CaptionBandGeometry(captions["position"], y, height, padding, line_height,
                               bottom_padding, warnings)


def caption_layer_plan(
    phrases: Sequence[Phrase], style: ResolvedStyle, *, frames: int, fps: int = 30,
    offsets: Sequence[int] | None = None,
    caption_settings: Mapping[str, Any] | None = None,
    backing: Sequence[str | None] | None = None,
) -> LayerPlan:
    """Anchor bottom captions on their last line, reserving room to wrap upward."""
    hyperframes.layer_duration_s(frames, fps)
    if bleep.enabled(style.data):
        masked_words = iter(bleep.mask_display_words(
            [word for phrase in phrases for word in phrase.words],
            bleep.listed(style.data["music"])))
        phrases = [replace(phrase, words=tuple(next(masked_words) for _ in phrase.words))
                   for phrase in phrases]
    captions = (caption_layout_settings(style.data["captions"], style.data["brand"])
                if caption_settings is None else caption_settings)
    duration = frames / fps
    slots = []
    for phrase in phrases:
        start, end = max(0.0, phrase.start), min(duration, phrase.end)
        if end <= start:
            continue
        slot = {
            "start": start,
            "end": end,
            "stress": next(
                (i for i, word in enumerate(phrase.words) if word.id == phrase.stress_id), None
            ),
            "words": [
                {"id": word.id, "text": word.text, "start": word.start, "end": word.end,
                 "source_ids": list(getattr(word, "source_ids", (word.id,))),
                 **({"preserve_case": True} if word.preserve_case else {})}
                for word in phrase.words
            ],
        }
        slots.append(slot)
    if offsets is not None and (
        len(offsets) != len(slots) or any(type(offset) is not int for offset in offsets)
    ):
        raise ValueError("offsets must contain one integer per caption slot")
    if backing is not None and (
        len(backing) != len(slots) or any(look not in (None, "shadow", "box") for look in backing)
    ):
        raise ValueError("backing must contain one shadow, box or None per caption slot")
    if backing is not None:
        for slot, look in zip(slots, backing, strict=True):
            if look is not None:
                slot["backing"] = look
    if (captions["highlight_style"] not in {"slam", "weight_shift", "clip_wipe"}
            and captions["max_words_per_phrase"] > 1):
        for slot in slots:
            breaks = _caption_breaks(slot, captions, boxed=slot.get("backing") == "box")
            if breaks:
                slot["breaks"] = list(breaks)
    geometry = caption_band_geometry(
        captions, safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
    )
    outer = caption_band_geometry(
        captions, safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
        backing=backing if backing is not None else (),
    )
    if backing is not None and any(backing):
        for slot, look in zip(slots, backing, strict=True):
            if look is None:
                continue
            # Keep plain slots on the default band while backed phrases reserve their own.
            own = caption_band_geometry(
                captions, safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
                backing=(look,),
            )
            slot["band"] = {"top_px": own.y - geometry.y, "height_px": own.height,
                            "padding_px": own.padding, "bottom_padding_px": own.bottom_padding}
    moved = offsets is not None and any(offsets)
    top, bottom = outer.y, outer.y + outer.height
    band_variables = {}
    if moved:
        assert offsets is not None
        top += min(0, min(offsets))
        bottom += max(0, max(offsets))
        for slot, offset in zip(slots, offsets, strict=True):
            slot["offset_px"] = offset
    if moved or outer != geometry:
        band_variables = {"band_top_px": geometry.y - top, "band_height_px": geometry.height}
    tokens = {
        key: captions[key]
        for key in (
            "size_px",
            "text_color",
            "background_mode",
            "background_color",
            "background_opacity",
            "background_padding_px",
            "background_radius_px",
            "highlight_color",
            "highlight_text_color",
            "highlight_style",
            "italic_stress",
            "shadow_opacity",
            "shadow_blur_px",
            "shadow_x_px",
            "shadow_y_px",
            "letter_spacing_px",
            "width_scale",
            "entry_rise_px",
            "entry_ms",
            "entrance",
        )
    }
    if captions["entrance"] == "spoken" and captions["max_words_per_phrase"] == 1:
        tokens["entrance"] = "slide"
    return LayerPlan(
        "captions",
        0,
        top,
        1080,
        bottom - top,
        0,
        frames,
        {
            **tokens,
            **band_variables,
            "font_family": fonts.font_stack(captions["font"]),
            "font_name": captions["font"],
            "position": captions["position"],
            "padding_px": geometry.padding,
            "bottom_padding_px": geometry.bottom_padding,
            "fps": fps,
            "phrases": json.dumps(slots, ensure_ascii=False, allow_nan=False),
        },
        warnings=outer.warnings,
    )


def _caption_breaks(slot: Mapping[str, Any], captions: Mapping[str, Any], *,
                    boxed: bool = False) -> tuple[int, ...]:
    """Mirror the template's text width and line width for stable rows."""
    mode = captions["highlight_style"]
    size = captions["size_px"]
    scale = captions["width_scale"]
    ground = boxed or (captions["background_mode"] != "none"
                       and captions["background_opacity"] > 0)
    ground_pad = captions["background_padding_px"] if ground else 0
    ported = mode in {
        "blend_difference", "camera_follow", "editorial_emphasis", "emoji_pop",
        "glitch_rgb", "gradient_fill", "highlight", "matrix_decode", "neon_accent",
        "neon_glow", "parallax_layers", "particle_burst", "pill_karaoke", "texture",
    }
    effect_pad = math.ceil(size * 0.25) if ported else 0
    available = 940 / scale if scale < 1 else 940
    if ported:
        available = (940 - 2 * (ground_pad + effect_pad)) / scale
    if mode == "pill":
        shadow = (captions["shadow_blur_px"] * 3 + abs(captions["shadow_x_px"])
                  if captions["shadow_opacity"] > 0 else 0)
        available = min(940, 1080 - size * 0.8 - shadow * 2) / scale
    if ground:
        highlight_pad = size * (0.4 if mode == "pill" else
                                0.08 if mode == "moving_block" else 0)
        available = min(available, (940 - 2 * (ground_pad + highlight_pad)) / scale)
    if mode == "pill_karaoke":
        available -= size * 0.6
    weight = (800 if mode in {"camera_follow", "glitch_rgb", "highlight", "neon_glow",
                              "particle_burst"} else 700 if mode == "pill_karaoke" else 400)
    tracking = captions["letter_spacing_px"]
    widths = []
    for index, word in enumerate(slot["words"]):
        display = caption_word_case(word["text"], mode,
                                    preserve_case=word.get("preserve_case", False))
        word_weight = 700 if mode == "editorial" and index == slot["stress"] else weight
        if mode in {"editorial_emphasis", "neon_accent"} and index == slot["stress"]:
            word_weight = 800
        factor = (1.45 if mode == "editorial_emphasis" else 1.2 if mode == "neon_accent"
                  else 1) if index == slot["stress"] else 1
        width = text_width_em(display, captions["font"], word_weight) * size * factor
        width += len(display) * tracking
        if captions["italic_stress"] and index == slot["stress"]:
            width += size * 0.16
        widths.append(width)
    space = text_width_em(" ", captions["font"], weight) * size + tracking
    return balance_lines(widths, space, available)


def _phrase_box_width(phrase: Mapping[str, Any], captions: Mapping[str, Any]) -> float:
    """Conservatively bound a fitted ground using bundled glyph advances.

    Include the heaviest highlight state, entry punch, italic overhang and shadow.
    Long or wrapped phrases retain the full safe width.
    """
    mode = captions["highlight_style"]
    texts = [caption_word_case(word["text"], mode, preserve_case=word.get("preserve_case", False))
             for word in phrase["words"]]
    slam = mode == "slam" and not (
        captions["entrance"] == "spoken" and captions["max_words_per_phrase"] > 1
    )
    weight = 800 if mode in ("slam", "clip_wipe", "weight_shift") else (
        700 if mode == "editorial" else 400
    )
    size = captions["size_px"]
    tracking = captions["letter_spacing_px"]
    widths = []
    for text in texts:
        width = text_width_em(text, captions["font"], weight) * size
        width += len(text) * tracking
        widths.append(width)
    space = text_width_em(" ", captions["font"], weight) * size + tracking
    width = (max(widths, default=0) if slam else
             sum(widths) + max(0, len(widths) - 1) * space)
    width *= captions["width_scale"] * (0.8 * 1.3 if slam else 1)
    # Advance tables omit kerning and glyph overhang, including synthesized italics.
    width = max(0, width) * 1.1 + size * 0.16
    highlight_pad = size * (0.4 if mode == "pill" else 0.08 if mode == "moving_block" else 0)
    shadow_pad = (captions["shadow_blur_px"] * 3 + abs(captions["shadow_x_px"])
                  if captions["shadow_opacity"] > 0 else 0)
    return min(940, math.ceil(width + 2 * (captions["background_padding_px"]
                                          + highlight_pad + shadow_pad)))


def caption_offsets(
    phrases_slots: Sequence[Mapping[str, Any]], geometry: CaptionBandGeometry,
    faces: FacesData | None, style: ResolvedStyle | Mapping[str, Any], *, fps: int = 30,
    still_faces: Sequence[Box] | None = None,
    caption_settings: Mapping[str, Any] | None = None,
    splits: Sequence[tuple[int, int, int]] = (),
    text_hook_bottom: int | None = None,
) -> list[dict]:
    """Keep clear phrases and move colliding bands to the nearest clear position."""
    data = style.data if isinstance(style, ResolvedStyle) else style
    captions = (caption_layout_settings(data["captions"], data["brand"])
                if caption_settings is None else caption_settings)
    margin = face_clearance_px(data)
    cb = caption_band(data)
    band = Band(min(cb.top, geometry.y), max(cb.bottom, geometry.y + geometry.height))
    plain = caption_band_geometry(captions,
                                  safe_bottom_px=data["internal"]["caption_safe_bottom_px"])
    ground_mode = captions["background_mode"] if captions["background_opacity"] > 0 else "none"
    placements = []
    for slot, phrase in enumerate(phrases_slots):
        phrase_ground = ground_mode
        if phrase.get("backing") == "box":
            phrase_ground = "band" if captions["background_mode"] == "band" else "box"
        x, width = (0, 1080) if phrase_ground == "band" else (70, 940)
        if phrase_ground == "box":
            width = _phrase_box_width(phrase, captions)
            x = (1080 - width) / 2
        phrase_band = phrase.get("band", {})
        top = plain.y + phrase_band.get("top_px", 0)
        height = phrase_band.get("height_px", plain.height)
        original_top, phrase_bounds = top, band
        if bounds := caption_split_band(phrase, height, top, splits,
                                        data["internal"]["caption_safe_bottom_px"], fps=fps):
            phrase_bounds, top = bounds
        element = Box(x, top, width, height)
        offset = top - original_top
        window = []
        if captions["placement"] == "fixed":
            reason = "kept: fixed caption placement"
        elif faces is None and still_faces is None:
            reason = "kept: no faces data"
        elif faces is not None and faces.status != "measured":
            reason = f"kept: faces not measured: {faces.reason}"
        else:
            first = round(phrase["start"] * fps, 6)
            entering = captions["entry_ms"] > 0 or captions["entrance"] == "rise"
            if captions["entrance"] == "spoken" and captions["max_words_per_phrase"] > 1:
                entering = False
            start = math.floor(first) + 1 if entering else math.ceil(first)
            end = math.ceil(round(phrase["end"] * fps, 6))
            if still_faces is not None:
                window = [tuple(still_faces)]
            else:
                assert faces is not None
                window = [tuple(face.box for face in frame)
                          for frame in faces.frames[max(0, start):max(0, end)]]
            if not any(window):
                reason = "kept: no face detected"
            elif all(gap_px(element, head_box(face, DEFAULT_HAIR)) >= margin
                     for frame in window for face in frame):
                reason = "kept: clear of the head"
            else:
                hint = suggest_offset(window, x=x, width=width, height=height,
                                      band=phrase_bounds, margin_px=margin, preferred_y=top)
                if hint.y is None:
                    reason = "kept: " + hint.reason
                else:
                    offset, reason = hint.y - original_top, hint.reason
        if text_hook_bottom is not None and original_top + offset < text_hook_bottom + margin:
            floor = max(cb.top, math.ceil(text_hook_bottom + margin))
            if floor + height > phrase_bounds.bottom:
                raise ValidationError("captions", "Captions would cover the Text hook. "
                                      "Please use a lower position or smaller caption text.")
            hint = suggest_offset(window, x=x, width=width, height=height,
                                  band=Band(floor, phrase_bounds.bottom), margin_px=margin,
                                  preferred_y=original_top + offset)
            if hint.y is None:
                raise ValidationError("captions", "Captions cannot clear the Text hook and face. "
                                      "Please use a lower position or smaller caption text.")
            offset, reason = hint.y - original_top, "placed below the Text hook; " + hint.reason
        placements.append({"slot": slot, "text": " ".join(w["text"] for w in phrase["words"]),
                           "offset_px": offset, "reason": reason})
    return placements


def caption_shrink(checks: Mapping[str, Any] | None, plan: LayerPlan) -> dict[str, Any]:
    """Map the runtime shrink summary to caption text and output times for receipts."""
    if checks is None:
        return {"measured": False, "reason": "Layer checks are unavailable."}
    passes = checks.get("passes")
    runtime = passes.get("runtime") if isinstance(passes, Mapping) else None
    runtime = runtime if isinstance(runtime, Mapping) else {}
    if runtime.get("status") == "skipped":
        return {"measured": False,
                "reason": runtime.get("reason") or "Runtime checks were skipped."}
    findings = runtime.get("findings")
    findings = findings if isinstance(findings, list) else []
    message = next((item["message"] for item in findings
                    if isinstance(item, Mapping) and isinstance(item.get("message"), str)
                    and item["message"].startswith(CAPTION_SHRINK_PREFIX)), None)
    if message is None:
        return {"measured": False, "reason": "Caption shrink summary is missing."}
    try:
        summary = json.loads(message[len(CAPTION_SHRINK_PREFIX):])
        floor_px = summary["floor_px"]
        slots = json.loads(plan.variables["phrases"])
        if (isinstance(floor_px, bool) or not isinstance(floor_px, (int, float))
                or not math.isfinite(floor_px) or not isinstance(summary["phrases"], list)
                or not isinstance(slots, list)):
            raise ValueError("invalid shrink summary")
        phrases = []
        for item in sorted(summary["phrases"], key=lambda item: item["slot"]):
            slot, size_px, fits = item["slot"], item["size_px"], item["fits"]
            if (isinstance(slot, bool) or not isinstance(slot, int)
                    or not 0 <= slot < len(slots) or isinstance(size_px, bool)
                    or not isinstance(size_px, (int, float)) or not math.isfinite(size_px)
                    or not isinstance(fits, bool)):
                raise ValueError("invalid shrink phrase")
            phrase = slots[slot]
            phrases.append({"start_s": phrase["start"], "end_s": phrase["end"],
                            "text": " ".join(word["text"] for word in phrase["words"]),
                            "size_px": size_px, "fits": fits})
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return {"measured": False, "reason": "Caption shrink summary is invalid."}
    return {"measured": True, "floor_px": floor_px,
            "style_size_px": plan.variables["size_px"], "phrases": phrases}


def caption_frame_window(start_s: float, end_s: float, *, fps: int,
                         layer_start: int) -> tuple[int, int]:
    """Map a phrase's open time edges to its half-open output frame window."""
    if not math.isfinite(start_s) or not math.isfinite(end_s) or fps <= 0:
        raise ValueError("invalid caption phrase time")
    start = layer_start + math.floor(round(start_s * fps, 6)) + 1
    end = layer_start + math.ceil(round(end_s * fps, 6))
    return start, end


def caption_layout(checks: Mapping[str, Any] | None, plan: LayerPlan) -> dict[str, Any]:
    """Map glyph boxes to output pixels and frame windows.

    Consumers must check measured before trusting an empty caption list.
    Boxes cover the area each word's glyphs occupy on any drawn frame,
    including the entry rise and slam punch.
    """
    document = {"schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
                "measured": False, "reason": None, "elements": [], "captions": []}

    def unmeasured(reason: str) -> dict[str, Any]:
        return {**document, "reason": reason}

    if checks is None:
        return unmeasured("Layer checks are unavailable.")
    passes = checks.get("passes")
    runtime = passes.get("runtime") if isinstance(passes, Mapping) else None
    runtime = runtime if isinstance(runtime, Mapping) else {}
    if runtime.get("status") == "skipped":
        return unmeasured(runtime.get("reason") or "Runtime checks were skipped.")
    findings = runtime.get("findings")
    findings = findings if isinstance(findings, list) else []
    message = next((item["message"] for item in findings
                    if isinstance(item, Mapping) and isinstance(item.get("message"), str)
                    and item["message"].startswith(CAPTION_SHRINK_PREFIX)), None)
    if message is None:
        return unmeasured("Caption shrink summary is missing.")

    def number(value: Any) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError("invalid layout number")
        return float(value)

    try:
        summary = json.loads(message[len(CAPTION_SHRINK_PREFIX):])
        if not isinstance(summary, dict):
            raise ValueError("invalid layout summary")
        if "layout" not in summary:
            return unmeasured("Caption layout is missing.")
        measured = summary["layout"]
        slots = json.loads(plan.variables["phrases"])
        fps = number(plan.variables["fps"])
        if (not isinstance(measured, list) or not isinstance(slots, list)
                or len(measured) != len(slots) or fps <= 0):
            raise ValueError("invalid layout slots")
        captions = []
        seen = set()
        for item in sorted(measured, key=lambda item: item["slot"]):
            slot, boxes = item["slot"], item["boxes"]
            if (isinstance(slot, bool) or not isinstance(slot, int)
                    or not 0 <= slot < len(slots) or slot in seen
                    or not isinstance(boxes, list)):
                raise ValueError("invalid layout slot")
            seen.add(slot)
            translated = []
            for box in boxes:
                x, y, w, h = (number(box[key]) for key in ("x", "y", "w", "h"))
                if w < 0 or h < 0:
                    raise ValueError("invalid layout box")
                translated.append({"x": round(x + plan.x, 2), "y": round(y + plan.y, 2),
                                   "w": round(w, 2), "h": round(h, 2)})
            phrase = slots[slot]
            start, end = caption_frame_window(number(phrase["start"]),
                                              number(phrase["end"]), fps=fps,
                                              layer_start=plan.start_frame)
            words = []
            for word in phrase["words"]:
                if not isinstance(word["text"], str):
                    raise ValueError("invalid layout word")
                words.append({"id": word["id"], "text": word["text"],
                              "source_ids": word["source_ids"],
                              "start_s": round(number(word["start"]) + plan.start_frame / fps, 3),
                              "end_s": round(number(word["end"]) + plan.start_frame / fps, 3)})
            if end <= start:
                continue
            if not boxes:
                raise ValueError("invalid layout boxes")
            captions.append({"id": f"caption-{slot}", "text": " ".join(w["text"] for w in words),
                             "start_frame": start, "end_frame": end,
                             "boxes": translated, "words": words})
        document = {**document, "measured": True, "captions": captions}
        Layout.from_dict(document)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError, ValidationError):
        return {**document, "measured": False, "reason": "Caption layout is invalid.",
                "captions": []}
    return document


def _kinetic_display_lines(lines: Sequence[str], font: str) -> list[dict[str, Any]]:
    """Balance short display lines and reserve their width-fitted sizes."""
    count = sum(len(line.split()) for line in lines)

    def refuse(limit: int) -> None:
        raise ValidationError("title.kinetic",
                              f"shorten the kinetic title to {max(1, limit)} words")

    if not count or count > 16:
        refuse(min(16, count))
    display: list[list[str]] = []
    for line in lines:
        words = line.split()
        if not words:
            continue
        if len(words) <= 4 and text_width_em(" ".join(words), font, 400) * 110 <= 940:
            display.append(words)
            continue
        for rows in range(math.ceil(len(words) / 4), min(4, len(words) // 2) + 1):
            candidates = []
            for lengths in product(range(2, 5), repeat=rows):
                if sum(lengths) != len(words):
                    continue
                at, pieces = 0, []
                for length in lengths:
                    pieces.append(words[at:at + length])
                    at += length
                if all(text_width_em(" ".join(piece), font, 400) * 110 <= 940
                       for piece in pieces):
                    candidates.append((lengths, pieces))
            if candidates:
                _, pieces = min(candidates, key=lambda item: (
                    sum(length * length for length in item[0]), tuple(-n for n in item[0]),
                ))
                display.extend(pieces)
                break
        else:
            refuse(sum(map(len, display)) or count - 1)
    if (len(display) > 1 and len(display[-1]) == 1 and len(display[-2]) <= 3
            and text_width_em(" ".join(display[-2] + display[-1]), font, 400) * 110 <= 940):
        display[-2].extend(display.pop())
    if len(display) > 4:
        refuse(sum(len(line) for line in display[:4]))
    return [{"text": " ".join(words), "size_px": min(260, max(110, math.floor(
        940 / text_width_em(" ".join(words), font, 400) * 100) / 100))} for words in display]


def _kinetic_padding(lines: Sequence[Mapping[str, Any]], shadow: Mapping[str, Any]) -> int:
    """Reserve the punch overshoot, blur and caption shadow inside the layer."""
    line_height = max(math.ceil(line["size_px"] * .98) for line in lines)
    shadow_room = (3 * shadow["shadow_blur_px"]
                   + max(abs(shadow["shadow_x_px"]), abs(shadow["shadow_y_px"])))
    return math.ceil(line_height * .25 + 12
                     + (shadow_room if shadow["shadow_opacity"] else 0))


def scale_kinetic_title(plan: LayerPlan, scale: float) -> LayerPlan:
    """Scale the reserved lines together, clamping each at the readable floor."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1:
        raise ValidationError("graphics.receipt", "invalid kinetic title scale; rerun graphics")
    if scale == 1:
        return plan
    lines = [{**line, "size_px": max(110, math.floor(line["size_px"] * scale * 100) / 100)}
             for line in json.loads(plan.variables["kinetic_lines"])]
    height = sum(math.ceil(line["size_px"] * .98) for line in lines)
    padding = _kinetic_padding(lines, plan.variables)
    return replace(plan, height=height + 2 * padding, variables={
        **plan.variables, "kinetic_lines": json.dumps(lines, ensure_ascii=False, allow_nan=False),
        "size_px": max(line["size_px"] for line in lines),
        "padding_px": padding, "anchor_px": padding,
    })


def chapter_number(number: int) -> str:
    """Format a one-based chapter number with at least two digits."""
    return f"{number:02d}"


def chapter_timecode(start_frame: int) -> str:
    """Floor the output frame clock to a chapter timecode at 30 fps."""
    minutes, seconds = divmod(start_frame // 30, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _chapter_rows(text: str, font: str, size: int) -> int:
    rows, used = 1, 0.0
    space = text_width_em(" ", font, 400) * size
    for word in text.split():
        width = text_width_em(word, font, 400) * size
        if width > 900:
            return 4
        if used and used + space + width > 900:
            rows, used = rows + 1, 0
        used += (space if used else 0) + width
    return rows


def chapter_text_box(plan: LayerPlan) -> Box:
    """Bound the chapter's text and rule, including their downward entrance."""
    values = plan.variables
    font = values["font_name"]
    rows = _chapter_rows(values["title"], font, values["size_px"])
    # Match the chapter template's line heights, rule margins and content padding.
    height = 300 + 2 * 56 + 8 + rows * values["size_px"] * 1.12 + 64 + 48 * 1.2
    top = math.floor(120 + (plan.height - 120 - 260 - height) / 2)
    width = max(900, text_width_em(values["number"], font, 400) * 300,
                text_width_em(values["timecode"], font, 400, tracking_em=.16) * 48)
    return Box(plan.x + (plan.width - width) / 2, plan.y + top,
               width, math.ceil(height) + 40)


def chapter_layer_plan(
    title: str, number: int, style: ResolvedStyle, *, start_frame: int = 0,
    frames: int | None = None, element_id: str = "preview",
) -> LayerPlan:
    """Plan chapter text over the live picture on the resolved output word clock.

    Timeline anchors follow re-cuts and supply overlap checks and receipts.
    """
    if bleep.enabled(style.data):
        title = bleep.mask_text(title, bleep.listed(style.data["music"]))
    chapter = style.data["chapters"]
    if chapter["style"] == "none":
        raise ValidationError(f"chapter.{element_id}", "the style has no chapter title card; "
                              "choose Title card under chapters on the style page, "
                              "or remove the chapter")
    font = style.data["hook"]["font"]

    def fits(text: str, size: int) -> bool:
        return _chapter_rows(text, font, size) <= 3

    if not title.strip() or len(title) > 48 or not fits(title, 64):
        limit = min(48, len(title) - 1)
        while limit > 1 and not fits(title[:limit], 64):
            limit -= 1
        raise ValidationError(f"chapter.{element_id}.title",
                              f"shorten the chapter title to {max(1, limit)} characters")
    size = next(size for size in range(96, 63, -1) if fits(title, size))
    graphics = style.data["graphics"]
    paper, ink = graphics["paper_color"], graphics["ink_color"]
    caption = style.data["captions"]["text_color"]
    # Check the text band over both extremes of the live picture.
    background, text = ((paper, ink) if _contrast(paper, caption) >= _contrast(ink, caption)
                        else (ink, paper))
    composites = [
        "#" + "".join(f"{round(.7 * int(background[i:i + 2], 16) + .3 * ground):02X}"
                       for i in (1, 3, 5))
        for ground in (0, 255)
    ]

    def minimum_contrast(color: str) -> float:
        return min(_contrast(composite, color) for composite in composites)

    if minimum_contrast(text) < 4.5:
        text = max((text, "#111111", "#FFFFFF"), key=minimum_contrast)
    if minimum_contrast(text) < 4.5:
        background, text = "#111111", "#FFFFFF"
    if frames is None:
        frames = math.floor(Fraction(str(chapter["hold_s"])) * 30 + Fraction(1, 2))
    hyperframes.layer_duration_s(frames, 30)
    return LayerPlan("chapter", 0, 0, 1080, 1920, start_frame, frames, {
        "number": chapter_number(number), "title": title,
        "timecode": chapter_timecode(start_frame), "font_name": font,
        "font_family": fonts.font_stack(font), "size_px": size,
        "background_color": background, "text_color": text,
        "accent_color": graphics["accent_color"], "fps": 30,
    })


def title_layer_plan(
    plain: str, key: str, style: ResolvedStyle, *, fps: int = 30,
    lines: Sequence[str] | None = None, frames: int | None = None, start_frame: int = 0,
    word_times: Sequence[float] | None = None,
) -> LayerPlan:
    """Plan an opening card using the preview's look, spacing, and position."""
    hyperframes.layer_duration_s(1, fps)
    if bleep.enabled(style.data):
        vocabulary = bleep.listed(style.data["music"])
        text = bleep.mask_text(plain + "\n" + key, vocabulary)
        plain, key = text[:len(plain)], text[len(plain) + 1:]
        if lines is not None:
            lines = bleep.mask_text("\n".join(lines), vocabulary).split("\n")
    hook = style.data["hook"]
    look = hook["style"]
    if look == "text_hook":
        from yap_studio.text_hook import LINE_HEIGHT, PADDING_X, PADDING_Y, fit_text_hook

        fitted = fit_text_hook(" ".join(lines) if lines is not None else plain + " " + key)
        color = hook.get("text_hook_color", "white")
        background = {"white": "#FFFFFF", "black": "#000000",
                      "accent": hook["accent_color"]}[color]
        ink = max(("#000000", "#FFFFFF"), key=lambda value: _contrast(value, background))
        width = math.ceil(max(fitted.widths) + 2 * PADDING_X) + 2
        height = len(fitted.lines) * (math.ceil(fitted.size_px * LINE_HEIGHT) + 2 * PADDING_Y)
        frames = frames if frames is not None else max(1, round(hook["duration_s"] * fps))
        hyperframes.layer_duration_s(frames, fps)
        return LayerPlan("title", (1080 - width) // 2, 250, width, height, start_frame, frames, {
            "look": look, "text_hook_color": color, "plain": " ".join(fitted.lines), "key": "",
            "lines": json.dumps(fitted.lines, ensure_ascii=False),
            "word_times": "[]", "kinetic_lines": "[]", "font_name": "Inter",
            "font_family": fonts.font_stack("Inter"), "size_px": fitted.size_px,
            "text_color": ink, "background_color": background, "accent_color": hook["accent_color"],
            "padding_px": 0, "fps": fps, "centered": False, "anchor_px": 0,
        })
    if look == "video_letters":
        from yap_studio.letters import title_plan

        return title_plan(plain, key, style, lines=lines, frames=frames, start_frame=start_frame)
    if look == "none":
        raise ValidationError("title", "the style's hook.style is none; choose a title look")
    if frames is None and hook["duration_s"] <= 0:
        raise ValidationError("title", "the style's hook.duration_s must be greater than zero")
    new_looks = {
        # width, available text width, size factor, vertical card padding
        "calm_card": (960, 776, 0.72, 108),
        "lockup": (940, 760, 1.1, 116),
        "write_on": (1000, 820, 1.1, 110),
        "headline_slam": (1080, 800, 1.04, 112),
        "shard_reveal": (1060, 872, 1.1, 108),
        "textured": (1000, 820, 1.1, 110),
        "shimmer": (1000, 820, 1.1, 110),
        "stagger": (1000, 820, 1.05, 110),
        "morph": (1000, 820, 1.1, 110),
        "typewriter": (1000, 820, 0.82, 100),
        "kinetic": (1080, 960, 1.5, 16),
    }
    if look in new_looks:
        width, content_width, size_factor, card_padding = new_looks[look]
        size = 260 if look == "kinetic" else hook["size_px"] * size_factor
    else:
        width = {"paper_title": 1080, "quiet_serif": 940, "bold_band": 1160}[look]
        content_width = {"paper_title": 776, "quiet_serif": 688, "bold_band": 940}[look]
        size = hook["size_px"] * (0.72 if look == "quiet_serif" else 1)
    # Reserve wrapping space without depending on machine fonts.
    def wrapped_lines(texts: Sequence[str], font_size: float,
                      glyph_factor: float = 1.2) -> int:
        count = 0
        space = (text_width_em(" ", hook["font"], 400) * font_size if look == "kinetic"
                 else font_size * glyph_factor)
        for text in texts:
            count += 1
            used = 0.0
            for word in text.split():
                word_width = (text_width_em(word, hook["font"], 400) * font_size
                              if look == "kinetic" else len(word) * font_size * glyph_factor)
                if used and used + space + word_width > content_width:
                    count, used = count + 1, 0.0
                if word_width > content_width:
                    count += math.ceil(word_width / content_width) - 1
                    word_width %= content_width
                used += word_width + (space if used else 0)
        return count

    source_lines = lines if lines is not None else [plain + " " + key]
    kinetic_lines = _kinetic_display_lines(source_lines, hook["font"]) if look == "kinetic" else []
    full_text = " ".join(source_lines)
    accent = key.strip()
    if look == "kinetic":
        if lines is None:
            start = len(plain.split())
            accented = set(range(start, start + len(key.split())))
        else:
            normalized = " ".join(full_text.split())
            key_text = " ".join(key.split())
            start = normalized.find(key_text) if key_text else -1
            accented = {index for index, word in enumerate(re.finditer(r"\S+", normalized))
                        if start >= 0 and word.start() < start + len(key_text)
                        and word.end() > start}
        at = 0
        for line in kinetic_lines:
            count = len(line["text"].split())
            line["accent"] = [index in accented for index in range(at, at + count)]
            at += count
    kicker = plain.strip()
    if not kicker and accent and accent in full_text:
        kicker = " ".join(full_text.replace(accent, "", 1).split())
    if look == "kinetic":
        card_height = sum(math.ceil(line["size_px"] * .98) for line in kinetic_lines)
    elif look == "lockup":
        # The rendered lockup has a smaller kicker and one accented headline.
        if kicker and accent:
            kicker_size = hook["size_px"] * 0.55
            card_height = math.ceil(
                wrapped_lines([kicker], kicker_size, 1.4) * kicker_size * 1.1
                + 13 + wrapped_lines([accent], size) * size * 1.04 + 54 + 23
            )
        else:
            card_height = math.ceil(wrapped_lines(source_lines, size) * size * 1.04 + 77)
    else:
        card_height = math.ceil(
            wrapped_lines(source_lines, size) * size * 1.04
            + (card_padding if look in new_looks else 66 if look == "quiet_serif" else 102)
        )
    padding = 60 if look in new_looks else 40 if look == "quiet_serif" else 80
    if look == "kinetic":
        padding = _kinetic_padding(kinetic_lines, style.data["captions"])
    rotate_room = math.ceil(width * math.sin(math.radians(2.5))) if look == "bold_band" else 0
    height = card_height + 2 * padding + (0 if look == "kinetic" else 30 + rotate_room)
    top = _HOOK_TOP[hook["position"]]
    y = (round(520 - height / 2) if look == "kinetic" else
         round(top - card_height / 2 - padding) if hook["position"] == "center" else top - padding)
    if height > 1920 or y + height > 1920:
        raise ValidationError("title", "the title is too tall for this look; shorten the title")
    frames = frames if frames is not None else max(1, math.floor(hook["duration_s"] * fps + 0.5))
    hyperframes.layer_duration_s(frames, fps)
    return LayerPlan(
        "title",
        (1080 - width) // 2,
        max(0, y),
        width,
        height,
        start_frame,
        frames,
        {
            "plain": plain,
            "key": key,
            "lines": json.dumps(list(lines or ()), ensure_ascii=False, allow_nan=False),
            "word_times": json.dumps(list(word_times or ()), allow_nan=False),
            "kinetic_lines": json.dumps(kinetic_lines, ensure_ascii=False, allow_nan=False),
            "look": look,
            "text_hook_color": hook.get("text_hook_color", "white"),
            "font_name": hook["font"],
            "font_family": fonts.font_stack(hook["font"]),
            "size_px": (max(line["size_px"] for line in kinetic_lines) if look == "kinetic"
                        else hook["size_px"]),
            "text_color": hook["text_color"],
            "background_color": hook["background_color"],
            "accent_color": hook["accent_color"],
            "ink_color": style.data["captions"]["text_color"],
            **{name: style.data["captions"][name] for name in (
                "shadow_opacity", "shadow_blur_px", "shadow_x_px", "shadow_y_px",
            )},
            "padding_px": padding,
            "fps": fps,
            "centered": look != "kinetic" and hook["position"] == "center",
            "anchor_px": padding if look == "kinetic" else top - max(0, y),
        },
    )


class _RootParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.root: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if "data-composition-id" in dict(attrs):
            self.root = self.get_starttag_text()


def prepare_layer_plan(
    plan: LayerPlan, work_dir: Path, *, assets: LayerAssets | None = None,
) -> Path:
    """Prepare the exact local HTML and bundled fonts without rendering."""
    if plan.template not in {"captions", "title", "letters", "chapter"}:
        raise ValidationError("template", "choose captions, title, letters or chapter")
    with paths.operation():
        composition = work_dir / plan.template
        composition.mkdir(parents=True, exist_ok=True)
        source = (assets.html if assets else
                  (_TEMPLATES / plan.template / "template.html").read_text(encoding="utf-8"))
        parser = _RootParser()
        parser.feed(source)
        if parser.root is None:
            raise ValidationError("template", "composition root is missing")
        root = parser.root
        for attribute, value in (
            ("width", plan.width),
            ("height", plan.height),
            ("duration", hyperframes.layer_duration_s(plan.frames, int(plan.variables["fps"]))),
        ):
            root = re.sub(rf'data-{attribute}="[^"]*"', f'data-{attribute}="{value}"', root)
        source = source.replace(parser.root, root, 1)
        names = [plan.variables["font_name"]]
        source = source.replace("/* BUNDLED_FONT_FACES */",
                                assets.font_css if assets else fonts.font_face_css(names))
        if assets:
            dest = composition / "fonts"
            dest.mkdir(parents=True, exist_ok=True)
            for name, data in assets.font_files:
                (dest / name).write_bytes(data)
        else:
            fonts.copy_fonts(names, composition / "fonts")
        paths.atomic_write_text(composition / "index.html", source)
    return composition


def render_layer_plan(
    plan: LayerPlan, work_dir: Path, out: Path, *, assets: LayerAssets | None = None,
) -> hyperframes.LayerResult:
    """Prepare local HTML and bundled fonts, then render a BT.709 alpha layer."""
    with paths.operation():
        composition = prepare_layer_plan(plan, work_dir, assets=assets)
        options = ({"workers": hyperframes.CAPTION_WORKERS,
                    "browser_gpu": hyperframes.CAPTION_BROWSER_GPU}
                   if plan.template == "captions" else {})
        return hyperframes.render_layer(
            composition,
            out,
            frames=plan.frames,
            variables=plan.variables,
            fps=int(plan.variables["fps"]),
            **options,
        )


@dataclass(frozen=True)
class CaptionSnapshot:
    """An unplaced caption plan and its narrow receipt projection."""

    plan: LayerPlan
    phrases: tuple[Phrase, ...]
    assets: LayerAssets
    inputs: dict[str, str]
    caption_settings: dict[str, Any]
    vocabulary: tuple[str, ...]
    backing: tuple[str | None, ...] = ()
    auto: tuple[bool, ...] = ()
    windows: tuple[str, ...] = ()


def caption_windows(context: ProjectContext, timeline: ResolvedTimeline,
                    phrases: Sequence[Phrase]) -> tuple[str, ...]:
    """Identify the picture and caption style measured during each phrase."""
    picture = {
        "version": 1,
        "style": {key: value for key, value in context.render_style().data.items()
                  if key not in {"voice", "music"}},
        "cut": context.identity.sha256(context.project_dir / "cut/cut.mp4"),
        "crops": {take.id: take.crop_x for take in context.project.takes},
    }
    picture["style"]["captions"] = {key: value for key, value in
                                     picture["style"]["captions"].items() if key != "camera_3d"}
    windows = []
    for phrase in phrases:
        start, end = caption_frame_window(phrase.start, phrase.end, fps=30, layer_start=0)
        windows.append(projection_sha256({
            **picture, "frames": [start, end],
            "elements": [item.to_dict() for item in timeline.elements
                         if item.start_frame < end and item.end_frame > start],
        }))
    return tuple(windows)


def caption_spelling(phrases: Sequence[Phrase], vocabulary: Sequence[str]) -> dict[str, Any]:
    """Report words that resemble a creator spelling after caption fixes."""
    near = []
    for phrase in phrases:
        texts = [word.text for word in phrase.words]
        for index, length, entry in near_vocabulary(texts, vocabulary):
            near.append({"text": " ".join(texts[index:index + length]), "entry": entry,
                         "start_s": phrase.words[index].start, "phrase": phrase.index})
    return {"vocabulary": len(vocabulary), "near": near}


def _empty_camera_snapshot(context: ProjectContext, retained: CameraSnapshot) -> CaptionSnapshot:
    style = context.render_style()
    settings = caption_layout_settings(style.data["captions"], style.data["brand"])
    assets = layer_assets("captions", style.data["captions"]["font"], context)
    inputs = {**retained.inputs, "empty_layer": projection_sha256({
        "fonts": assets.font_identity(),
        "renderer": assets.template_identity(workers=hyperframes.CAPTION_WORKERS,
                                              browser_gpu=hyperframes.CAPTION_BROWSER_GPU),
    })}
    plan = caption_layer_plan((), style, frames=retained.frames, caption_settings=settings,
                              backing=())
    return CaptionSnapshot(plan, (), assets, inputs, settings, retained.vocabulary)


def _text_hook_bottom(timeline: ResolvedTimeline, style: ResolvedStyle) -> int | None:
    if style.data["hook"]["style"] != "text_hook":
        return None
    title = next((item for item in timeline.elements
                  if item.kind == "title" and item.id == TEXT_HOOK_ID), None)
    if title is None:
        return None
    plan = title_layer_plan("", "", style, lines=title.payload["lines"], frames=1)
    # The hook can only move upward, so its default bottom reserves every placement.
    return plan.y + plan.height


def caption_snapshot(context: ProjectContext) -> CaptionSnapshot | CameraSnapshot:
    """Build captions from retained output words, fixes, gaps and caption style."""
    style = context.render_style()
    if style.data["captions"].get("camera_3d", False):
        from yap_studio.stages.captions_3d import snapshot

        retained = snapshot(context)
        return retained if retained.words else _empty_camera_snapshot(context, retained)
    settings = caption_layout_settings(style.data["captions"], style.data["brand"])
    timeline = context.resolved_timeline()
    cut_bytes = context.read("cut/cut-words.json")
    words = cast(CutWords, parse_document("cut-words", json.loads(cut_bytes)))
    vocabulary = tuple(list_words())
    green_layouts = layout_windows(timeline)
    phrases = build_phrases(
        [CaptionWord(word.id, word.text, word.start, word.end) for word in words.words],
        settings, word_fixes=context.project.word_fixes,
        word_text_fixes=context.project.word_text_fixes,
        text_rules=True, vocabulary=vocabulary,
        **({"forced_breaks": caption_breaks(words.words, green_layouts)} if green_layouts else {}),
    )
    if bleep.enabled(style.data):
        masked_words = iter(bleep.mask_display_words(
            [word for phrase in phrases for word in phrase.words],
            bleep.listed(style.data["music"])))
        phrases = [replace(phrase, words=tuple(next(masked_words) for _ in phrase.words))
                   for phrase in phrases]
    if green_layouts:
        phrases = clip_layout_phrases(phrases, green_layouts)
    duration = timeline.total_frames / 30
    phrases = [replace(phrase, start=max(0.0, phrase.start), end=min(duration, phrase.end))
               for phrase in phrases if phrase.start < duration and phrase.end > 0]
    gaps = [item for item in timeline.elements if item.kind == "caption_gap"]
    phrases = [phrase for phrase in phrases if not any(
        phrase.start < gap.end_s and phrase.end > gap.start_s for gap in gaps
    )]
    assets = layer_assets("captions", style.data["captions"]["font"], context)
    keys = phrase_keys([phrase.word_ids for phrase in phrases])
    windows = caption_windows(context, timeline, phrases)
    halos = caption_halos(context)
    choices = context.project.caption_backing
    auto = tuple(key not in choices and halos.get(key, {}).get("window") == window
                 for key, window in zip(keys, windows, strict=True))
    backing = tuple((choices[key] if choices[key] != "none" else None) if key in choices
                    else "shadow" if automatic else None
                    for key, automatic in zip(keys, auto, strict=True))
    plan = caption_layer_plan(phrases, style, frames=timeline.total_frames,
                              caption_settings=settings, backing=backing)
    inputs = {
        # Word times alone cannot identify trailing picture, so retain the cut's frame clock.
        "cut/cut-words.json": projection_sha256({
            "sha256": hashlib.sha256(cut_bytes).hexdigest(), "frames": timeline.total_frames,
            # Older receipts may own a layer with its last phrase extended through trailing picture.
            "phrase_tail": "natural",
        }),
        "word_fixes": projection_sha256(context.project.word_fixes),
        **({"word_text_fixes": projection_sha256(context.project.word_text_fixes)}
           if context.project.word_text_fixes else {}),
        "brand:words": projection_sha256(vocabulary),
        "caption_gaps": projection_sha256([item.to_dict() for item in gaps]),
        "faces": projection_sha256(cut_faces_identity(
            context.identity.sha256(context.project_dir / "cut/cut.mp4"),
        )),
        "picture:faces": projection_sha256([
            {"id": item.id, "kind": item.kind,
             "start_frame": item.start_frame, "end_frame": item.end_frame,
             # Refresh placements cached when chapter cards hid the source face.
             **({"face_visible": True} if item.kind == "chapter" else {}),
             **(({"curve": item.payload["curve"]} if "move" in item.payload else
                 {"peak_scale": item.payload["peak_scale"], "center": item.payload["center"]})
                if item.kind == "zoom" else {})}
            for item in timeline.elements
            if item.kind in {"zoom", "insert", "chapter", "transition"}
        ]),
        **({"picture:layouts": projection_sha256({"windows": green_layouts, "rule": LAYOUT_RULE})}
           if green_layouts else {}),
        "style:face_clearance": projection_sha256(face_clearance_px(style.data)),
        "style:captions": projection_sha256({key: value for key, value in
                                             settings.items() if key != "camera_3d"}),
        "fonts": projection_sha256(assets.font_identity()),
        "renderer": projection_sha256(assets.template_identity(
            workers=hyperframes.CAPTION_WORKERS, browser_gpu=hyperframes.CAPTION_BROWSER_GPU)),
        **context.taste_inputs("captions"),
    }
    if (bottom := _text_hook_bottom(timeline, style)) is not None:
        inputs["text_hook"] = projection_sha256(bottom)
    if any(backing):
        inputs["caption_backing"] = projection_sha256(backing)
    if bleep.enabled(style.data):
        inputs["bleep"] = projection_sha256({**bleep.projection(style.data["music"]),
                                              "phrases": [phrase.to_dict() for phrase in phrases]})
    if layouts := split_layouts(context.project_dir, timeline):
        inputs["picture:faces"] = projection_sha256([
            inputs["picture:faces"], {key: layout.to_dict() for key, layout in layouts.items()},
        ])
    return CaptionSnapshot(plan, tuple(phrases), assets, inputs, settings, vocabulary,
                           backing, auto, windows)


def back_weak_captions(project_dir: Path, look: str, *, force: bool = False) -> dict:
    """Back the current final's weak phrases and render their caption layer."""
    if look not in {"shadow", "box", "none"}:
        raise InputError("back", "choose shadow, box or none")
    root = Path(project_dir).resolve()
    if ProjectContext(root).render_style().data["captions"].get("camera_3d", False):
        from yap_studio.render.camera3d_look import Camera3DRefusal

        raise Camera3DRefusal(
            "3D camera captions always draw their own halo, so --back does not apply.",
            "Run yap-studio captions DIR without --back.")
    path = root / "project.json"
    backed, skipped = [], 0
    with project_lock(root):
        before = receipt.sha256_file(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise InputError(str(path), "cannot read valid project JSON") from exc
        parse_document("project", data)
        final = read_receipt(root / "final") or {}
        caption_receipt = read_receipt(root / "captions") or {}
        contrast = final.get("caption_contrast")
        digest = next((item.get("sha256") for item in caption_receipt.get("outputs", [])
                       if item.get("path") == "captions/captions.mov"), None)
        current = (isinstance(contrast, dict) and contrast.get("measured") is True
                   and isinstance(contrast.get("weak"), list) and digest
                   and final.get("inputs", {}).get("media:captions/captions.mov") == digest)
        if not current and look != "none":
            error = InputError("final/receipt.json", "caption contrast is missing or stale")
            error.suggested_action = (
                "Run yap-studio run DIR so the final measures caption contrast, then try again."
            )
            raise error
        choices = data.setdefault("caption_backing", {})
        if look == "none":
            off = dict(caption_halos(ProjectContext(root)))
            off.update({key: {"text": ""} for key, value in choices.items()
                        if value in {"shadow", "box"}})
            for key, phrase in off.items():
                choices[key] = "none"
                backed.append({"key": key, "text": phrase["text"], "look": look})
        if current:
            for index, phrase in enumerate(contrast["weak"]):
                key = phrase.get("key")
                if key is None:
                    ids = phrase.get("ids")
                    if not ids:
                        skipped += 1
                        continue
                    key = phrase_key(ids, index)
                choices[key] = look
                if not any(item["key"] == key for item in backed):
                    backed.append({"key": key, "text": phrase["text"], "look": look})
        parse_document("project", data)
        write_json_cas(path, data, base_sha256=before)
    return {"backed": backed, "skipped": skipped, "captions": captions(root, force=force)}


def captions(project_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Publish a project caption band, output-time phrases, SRT and owned receipt."""
    with paths.operation():
        context = ProjectContext(project_dir)
        snapshot = caption_snapshot(context)
        if not isinstance(snapshot, CaptionSnapshot):
            from yap_studio.stages.captions_3d import captions as camera_captions

            return camera_captions(context, force=force, retained=snapshot)
        root = context.project_dir
        run = StageRun(root, "captions")
        layer = snapshot.plan.placement("captions/captions.mov")
        result = {"status": "verified", "out": layer["path"], "frames": snapshot.plan.frames,
                  "phrases": len(snapshot.phrases), "layer": layer,
                  "srt": "captions/captions.srt", "layout": "captions/layout.json",
                  "receipt": "captions/receipt.json",
                  "caption_spelling": caption_spelling(snapshot.phrases, snapshot.vocabulary)}
        if not force and run.up_to_date(snapshot.inputs):
            previous = read_receipt(root / "captions")
            return {**result, "layer": previous.get("layer", layer) if previous else layer,
                    "noop": True}
        previous = read_receipt(root / "captions")
        work_root = paths.validate_output_dir(root / ".yap/work")
        with tempfile.TemporaryDirectory(prefix="captions-", dir=work_root) as folder, run:
            work = Path(folder)
            for name, digest in snapshot.inputs.items():
                run.record_inputs(name, digest)
            style = context.render_style()
            timeline = context.resolved_timeline()
            layouts = split_layouts(root, timeline)
            splits = split_windows(timeline, layouts)
            faces = None
            frame_sources = []
            if style.data["captions"]["placement"] == "avoid_face":
                cut = cut_faces(root, root / "cut/cut.mp4")
                faces, frame_sources = map_cut_faces(
                    cut, total_frames=timeline.total_frames,
                    end_hold_frames=timeline.end_hold_frames,
                    zooms=[ZoomPiece(item.start_frame, item.end_frame,
                                     ZoomCurve.from_payload(item.start_frame, item.end_frame,
                                                            item.payload))
                           for item in timeline.elements if item.kind == "zoom"],
                    inserts=[(item.start_frame, item.end_frame)
                             for item in timeline.elements
                             if item.kind == "insert" and item.id not in layouts],
                    splits=splits,
                    transitions=[(item.start_frame, item.end_frame)
                                 for item in timeline.elements if item.kind == "transition"],
                    layouts=layout_windows(timeline),
                    video=VideoIdentity("picture.mp4", cut.video.sha256,
                                        timeline.total_frames + timeline.end_hold_frames,
                                        Fraction(30), 1080, 1920),
                )
            geometry = caption_band_geometry(
                snapshot.caption_settings,
                safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
                backing=snapshot.backing,
            )
            plan = snapshot.plan
            placements = caption_offsets(json.loads(plan.variables["phrases"]),
                                         geometry, faces, style,
                                         caption_settings=snapshot.caption_settings,
                                         text_hook_bottom=_text_hook_bottom(timeline, style),
                                         splits=[(start, end, layout.seam_y)
                                                 for start, end, layout in splits])
            if any(item["offset_px"] for item in placements):
                plan = caption_layer_plan(snapshot.phrases, style, frames=timeline.total_frames,
                                          offsets=[item["offset_px"] for item in placements],
                                          caption_settings=snapshot.caption_settings,
                                          backing=snapshot.backing)
            kept_reasons = {"kept: no face detected", "kept: clear of the head",
                            "kept: fixed caption placement", "kept: no faces data",
                            f"kept: faces not measured: {faces.reason if faces else None}"}
            placement = {
                "faces": {"status": faces.status if faces else "unmeasured",
                          "reason": faces.reason if faces else "fixed caption placement",
                          "frame_sources": frame_sources},
                "phrases": [item for item in placements
                            if item["offset_px"] or item["reason"] not in kept_reasons],
                "kept_phrases": sum(item["offset_px"] == 0 for item in placements),
            }
            layer = plan.placement("captions/captions.mov")
            result["layer"] = layer
            run.progress("rendering captions", 0.1)
            rendered = render_layer_plan(plan, work, work / "captions.mov",
                                         assets=snapshot.assets)
            run.progress("checking captions", 0.9)
            subtitles = work / "captions.srt"
            subtitles.write_text(_srt_text(snapshot.phrases), encoding="utf-8")
            phrases_path = work / "phrases.json"
            phrases_path.write_text(json.dumps({
                "schema": "yap-studio/phrases@1", "time_base": "output", "fps": 30,
                "layer": layer, "phrases": [phrase.to_dict() for phrase in snapshot.phrases],
            }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            layout = caption_layout(rendered.checks, plan)
            layout_path = work / "layout.json"
            layout_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False, allow_nan=False)
                                   + "\n", encoding="utf-8")
            for name, staged in (("captions.mov", rendered.path), ("captions.srt", subtitles),
                                 ("phrases.json", phrases_path), ("layout.json", layout_path)):
                run.replace_output(root / "captions" / name, staged)
            if any((root / "captions" / name).exists()
                   for name in ("camera3d.mp4", "camera3d.json")):
                from yap_studio.stages.captions_3d import retire_picture

                retire_picture(root, run, previous)
            run.finish(extra={"layer": layer, "phrases": len(snapshot.phrases),
                              "caption_spelling": result["caption_spelling"],
                              "layer_checks": rendered.checks,
                              "caption_shrink": caption_shrink(rendered.checks, plan),
                              "caption_placement": placement,
                              "caption_layout": {"measured": layout["measured"],
                                                 "reason": layout["reason"],
                                                 "captions": len(layout["captions"])},
                              "taste": context.taste_application().pin()})
        return {**result, "noop": False}


def _output_target(target: Path, *, force: bool, owners: Sequence[Path]) -> None:
    if target.exists() and not force:
        raise CaptionRefusal(target, force=False)
    try:
        paths.validate_output(target, allowed_existing_suffixes=frozenset(), owner_receipts=owners)
    except OutputPathError as exc:
        if exc.code == "unsafe_path" and target.is_file() and not target.is_symlink():
            raise CaptionRefusal(target, force=force) from exc
        raise


@contextmanager
def _publish_lock(out: Path, timeout_s: float) -> Iterator[None]:
    """Serialize final output checks and publication, retaining the lock file."""
    lock_path = out.with_name(f".{out.name}.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
        else:
            import fcntl

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise YapError(
                        f"Another caption run holds the output lock for {out}.",
                        error_type="processing_error",
                        code="caption_output_locked",
                        suggested_action="Wait for that run to finish, or choose a new --out.",
                    ) from exc
                time.sleep(min(0.05, remaining))
        locked = True
        yield
    finally:
        try:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _load_words(clip: Path, words: Path | None, asr: str) -> transcript.Transcript:
    if words is not None:
        try:
            result = transcript.Transcript.load(words)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValidationError(
                "words", f"expected a yap-studio/transcript@1 file: {exc}"
            ) from exc
    else:
        backend = transcript.resolve_backend(asr)
        if backend is None:
            raise DependencyError.missing_engine("speech engine")
        result = backend.transcribe_many([clip])[clip]
    for word in result.words:
        if (
            not math.isfinite(word.start)
            or not math.isfinite(word.end)
            or word.start < 0
            or word.end <= word.start
        ):
            raise ValidationError("words", "word times must be finite with 0 <= start < end")
    return result


def _srt_text(phrases: Sequence[Phrase]) -> str:
    def clock(seconds: float) -> str:
        milliseconds = math.floor(seconds * 1000 + 0.5)
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    return "".join(
        f"{i}\n{clock(phrase.start)} --> {clock(phrase.end)}\n{phrase.text}\n\n"
        for i, phrase in enumerate(phrases, 1)
    )


def _conform(clip: Path, out: Path, timeout_s: float) -> None:
    with paths.atomic_output(out) as temporary:
        ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(clip),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "setpts=PTS-STARTPTS,fps=30:start_time=0:round=near,"
                "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos:"
                "out_color_matrix=bt709:out_range=tv,"
                "crop=1080:1920,setsar=1,format=yuv420p,"
                "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
                "-af",
                "aresample=48000",
                "-c:v",
                "libx264",
                "-crf",
                "10",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                str(temporary),
            ],
            timeout_s=timeout_s,
        )


def _composite(
    base: Path, layers: Sequence[tuple[LayerPlan, Path]], out: Path, frames: int, timeout_s: float,
    *, audio: Path | None = None,
) -> None:
    command = ["ffmpeg", "-nostdin", "-y", "-i", str(base)]
    filters, previous = [], "0:v:0"
    for index, (plan, layer) in enumerate(layers, 1):
        command.extend(["-i", str(layer)])
        label = f"v{index}"
        last = plan.start_frame + plan.frames - 1
        filters.append(
            f"[{previous}][{index}:v:0]overlay=x={plan.x}:y={plan.y}:eof_action=pass:"
            f"enable='between(n,{plan.start_frame},{last})'[{label}]"
        )
        previous = label
    if audio is not None:
        command.extend(["-f", "f32le", "-ar", str(bleep.RATE), "-ac", "2", "-i", str(audio)])
    filters.append(
        f"[{previous}]setparams=color_primaries=bt709:color_trc=bt709:"
        "colorspace=bt709:range=tv[encoded]"
    )
    command.extend(["-filter_complex", ";".join(filters)])
    command.extend(
        [
            "-map",
            "[encoded]",
            "-map",
            f"{len(layers) + 1}:a:0" if audio is not None else "0:a:0?",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-c:a",
            "aac" if audio is not None else "copy",
            *(["-ar", str(bleep.RATE), "-ac", "2", "-b:a", "192k"] if audio is not None else []),
            "-movie_timescale",
            "48000",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    ffmpeg.run(command, timeout_s=timeout_s)


def caption_clip(
    clip: Path,
    *,
    style: ResolvedStyle,
    out: Path,
    words: Path | None,
    title: str | None,
    srt: bool,
    force: bool,
    asr: str = "auto",
) -> dict:
    """Caption and optionally title a vertical SDR clip, recording owned outputs."""
    if style.data["captions"].get("camera_3d", False):
        from yap_studio.render.camera3d_look import Camera3DRefusal

        raise Camera3DRefusal("3D camera captions run inside a project.",
                              "Run yap-studio run DIR, or pass --style with another look.")
    started = datetime.now(UTC)
    with paths.operation():
        vocabulary = tuple(list_words())
        source = paths.validate_input(clip)
        word_source = paths.validate_input(words) if words is not None else None
        info = ffmpeg.probe(source)
        if info.orientation != "vertical" or info.color_transfer in {"arib-std-b67", "smpte2084"}:
            raise InputError(
                str(clip),
                "captions requires a vertical SDR clip; use the project flow "
                "(yap-studio init, coming in a later version) for horizontal and HDR takes",
            )
        if out.suffix.lower() != ".mp4":
            raise ValidationError("out", "choose an .mp4 file")
        title_lines = None
        if title is not None:
            plain, _, key = title.partition("|")
            if not (plain.strip() or key.strip()):
                raise ValidationError("title", "provide opening title text")
            title_lines = [plain.strip(), key.strip()]
        receipt_path = out.with_suffix(".receipt.json")
        srt_path = out.with_suffix(".srt") if srt else None
        owners = (receipt_path,) if force else ()
        _output_target(out, force=force, owners=owners)
        if srt_path is not None:
            _output_target(srt_path, force=force, owners=owners)
        paths.validate_output(receipt_path)
        timed = _load_words(source, word_source, asr)
        caption_words = [CaptionWord(f"w{i}", word.text, word.start, word.end)
                         for i, word in enumerate(timed.words)]
        bleep_words = bleep.listed(style.data["music"]) if bleep.enabled(style.data) else None
        bleep_receipt = bleep_assert = bleep_audio = None
        title_plan = None
        title_display = []
        warnings = []
        if title_lines is not None:
            word_times = None
            if style.data["hook"]["style"] == "kinetic":
                title_frames = max(1, math.floor(style.data["hook"]["duration_s"] * 30 + 0.5))
                try:
                    word_times = kinetic_display_times(title_lines, kinetic_word_times(
                        title_lines, caption_words, 0, title_frames / 30, {}, vocabulary=vocabulary,
                    ), vocabulary)
                except ValidationError as exc:
                    warnings.append(f"Kinetic hook uses even spacing: {exc.detail}")
            plain = apply_text_rules(title_lines[0], vocabulary)
            key = apply_text_rules(title_lines[1], vocabulary)
            if bleep_words is not None:
                text = bleep.mask_text(plain + "\n" + key, bleep_words)
                plain, key = text[:len(plain)], text[len(plain) + 1:]
            title_display = [plain, key]
            title_plan = title_layer_plan(plain, key, style,
                                          word_times=word_times)
        settings = caption_layout_settings(style.data["captions"], style.data["brand"])
        phrases = build_phrases(
            caption_words, settings,
            text_rules=True, vocabulary=vocabulary,
        )
        if bleep_words is not None:
            masked_words = iter(bleep.mask_display_words(
                [word for phrase in phrases for word in phrase.words], bleep_words))
            phrases = [replace(phrase, words=tuple(next(masked_words) for _ in phrase.words))
                       for phrase in phrases]
        out.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".yap-captions-", dir=out.parent))
        timeout_s = 120 + 20 * (info.video_duration_s or info.duration_s or 0)
        try:
            base = work / "conformed.mp4"
            _conform(source, base, timeout_s)
            frames = ffmpeg.probe(base).video_frames
            if frames is None or frames < 1:
                raise InputError(str(clip), "conforming produced no readable frames")
            duration = frames / 30
            phrases = [
                replace(phrase, start=max(0.0, phrase.start), end=min(duration, phrase.end))
                for phrase in phrases
                if phrase.start < duration and phrase.end > 0
            ]
            if bleep_words is not None:
                from yap_studio.stages.assemble_audio import decode

                audible = [word for word in caption_words
                           if word.start < duration and word.end >= 0]
                occurrences = bleep.find_display(audible, bleep_words)
                samples = round(duration * bleep.RATE)
                has_audio = ffmpeg.probe(base).has_audio
                original = (decode(base, [], channels=2) if occurrences and has_audio else
                            np.zeros((samples if occurrences else 0, 2), dtype=np.float32))
                if occurrences and len(original) < samples:
                    original = np.pad(original, ((0, samples - len(original)), (0, 0)))
                original = original[:samples]
                muted, tone, bleep_receipt = bleep.render(occurrences, original, duration, samples)
                bleep_assert = bleep.check(
                    audible, music=style.data["music"], voice=muted, original_voice=original,
                    tone_stem=tone, phrases={"phrases": [phrase.to_dict() for phrase in phrases]},
                    title_layers=([{"id": "clip-title", "kind": "title",
                                    "display_text": title_display}] if title_plan else []),
                    timeline=SimpleNamespace(elements=()),
                )
                bleep_assert["fix"] = ("Run this caption export again with Bleep swear words on "
                                       "and check the supplied word timings.")
                if not bleep_assert["pass"]:
                    raise YapError("Bleep check failed for the caption export.",
                                   error_type="verification_failed", code="bleep_check_failed",
                                   suggested_action=bleep_assert["fix"])
                if occurrences and has_audio:
                    (work / "bleep-voice.f32le").write_bytes(muted.astype("<f4").tobytes())
                    (work / "bleep.f32le").write_bytes(tone.astype("<f4").tobytes())
                    bleep_audio = work / "bleep-mix.f32le"
                    bleep_audio.write_bytes((muted + tone).astype("<f4").tobytes())
                warnings = [bleep.mask_text(warning, bleep_words) for warning in warnings]
            spelling = caption_spelling(phrases, vocabulary)
            layers = []
            if title_plan is not None:
                layers.append((title_plan,
                               render_layer_plan(title_plan, work, work / "title.mov").path))
            captions = caption_layer_plan(phrases, style, frames=frames, caption_settings=settings)
            placement = None
            if (style.caption_placement_explicit
                    and style.data["captions"]["placement"] == "avoid_face"):
                from yap_studio.faces.detect import detect_faces

                faces = detect_faces(base, every=2)
                geometry = caption_band_geometry(
                    settings,
                    safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
                )
                placements = caption_offsets(json.loads(captions.variables["phrases"]),
                                             geometry, faces, style, caption_settings=settings)
                captions = caption_layer_plan(phrases, style, frames=frames,
                                              offsets=[p["offset_px"] for p in placements],
                                              caption_settings=settings)
                placement = {
                    "faces": {"status": faces.status, "reason": faces.reason},
                    "phrases": placements,
                    "kept_phrases": sum(p["offset_px"] == 0 for p in placements),
                }
            warnings.extend(captions.warnings)
            rendered = render_layer_plan(captions, work, work / "captions.mov")
            layers.append((captions, rendered.path))
            composite = work / "composite.mp4"
            _composite(base, layers, composite, frames, timeout_s,
                       **({"audio": bleep_audio} if bleep_audio is not None else {}))
            if ffmpeg.probe(composite).video_frames != frames:
                raise ValidationError("out", "the composite frame count differs from the clip")
            staged_outputs = [(out, composite)]
            if srt_path is not None:
                subtitles = work / "captions.srt"
                subtitles.write_text(_srt_text(phrases), encoding="utf-8")
                staged_outputs.append((srt_path, subtitles))
            result = receipt.build_receipt(
                [clip] + ([words] if words is not None else []),
                [staged for _, staged in staged_outputs],
                started,
                datetime.now(UTC),
                extra={
                    "stage": "captions",
                    "style": {"name": style.name, "sha256": style.sha256()},
                    "renderer": {"name": "hyperframes", "version": hyperframes.HYPERFRAMES_VERSION,
                                 "workers": hyperframes.CAPTION_WORKERS,
                                 "capture": hyperframes.RENDER_CAPTURE,
                                 **({"browser_gpu": False}
                                    if hyperframes.CAPTION_BROWSER_GPU is False else {})},
                    "status": "verified",
                    "caption_spelling": spelling,
                    "caption_shrink": caption_shrink(rendered.checks, captions),
                    **({"bleep": bleep_receipt, "asserts": [bleep_assert]}
                       if bleep_receipt is not None else {}),
                    **({"caption_placement": placement} if placement is not None else {}),
                    **({"warnings": warnings} if warnings else {}),
                },
            )
            with _publish_lock(out, timeout_s):
                old_outputs = []
                for (target, _), record in zip(staged_outputs, result["outputs"], strict=True):
                    _output_target(target, force=force, owners=owners)
                    record["path"] = target.name
                    if target.exists():
                        old_outputs.append({
                            "path": target.name,
                            "sha256": receipt.sha256_file(target),
                            "bytes": target.stat().st_size,
                        })
                receipt.write_receipt(receipt_path, {
                    **result, "status": "publishing", "outputs": old_outputs + result["outputs"],
                })
                for target, staged in staged_outputs:
                    with paths.atomic_output(target, owner_receipts=[receipt_path]) as temporary:
                        shutil.copyfile(staged, temporary)
                receipt.write_receipt(receipt_path, result)
        finally:
            if os.environ.get("YAP_STUDIO_KEEP_WORK") != "1":
                shutil.rmtree(work)
    return {
        "out": str(out),
        "srt": str(srt_path) if srt_path else None,
        "receipt": str(receipt_path),
        "frames": frames,
        "phrases": len(phrases),
        "caption_spelling": spelling,
        "style": {"name": style.name, "sha256": style.sha256()},
        **({"warnings": warnings} if warnings else {}),
    }
