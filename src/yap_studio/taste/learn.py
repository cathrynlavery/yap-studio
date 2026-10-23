"""Measure finished videos and publish evidence-backed proposals to a named folder."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.captions import caption_layout_settings
from yap_studio.errors import InputError, OutputPathError
from yap_studio.stages.captions import CAPTION_LINE_HEIGHT, caption_band_geometry
from yap_studio.style import loader, resolve
from yap_studio.style.library import config_root
from yap_studio.style.loader import RESOLVED_SCHEMA, resolve_style_data
from yap_studio.taste import measure_pacing, measure_shapes, measure_words
from yap_studio.taste.frames import (
    FrameTiming,
    delta_e,
    hex_color,
    iter_rgb,
    presentation_timing,
    rgb,
)
from yap_studio.taste.measure_panels import (
    Panel,
    fade_end,
    find_panels,
    panel_edge_match,
    panel_present,
)
from yap_studio.taste.measure_text import CaptionDiscovery, caption_band, panel_text
from yap_studio.taste.model import (
    PROPOSALS_SCHEMA,
    TasteError,
    TasteRefusal,
    validate_body,
    validate_style_field,
)
from yap_studio.transcript import resolve_backend
from yap_studio.transcript.model import Transcript

MEASUREMENTS_SCHEMA = "yap-studio/taste-measurements@1"
HIGHLIGHT_MIN_SHARE = 0.25
HIGHLIGHT_MIN_SAMPLES = 3
COLOR_TEXT_TOLERANCE = 6
COLOR_HIGHLIGHT_TOLERANCE = 8
COLOR_TITLE_TOLERANCE = 10
CAPTION_TOP_TOLERANCE = 12
CAPTION_SIZE_TOLERANCE = 4
TITLE_SIZE_TOLERANCE = 6
PANEL_WIDTH_TOLERANCE = 8
CORNER_RADIUS_TOLERANCE = 3
DURATION_TOLERANCE = 0.1
TITLE_DURATION_TOLERANCE = 0.05
ZOOM_SCALE_TOLERANCE = 0.05
PAUSE_TOLERANCE = 0.04
LUFS_TOLERANCE = 0.2

OCR_FIELDS = (
    "captions.font",
    "captions.italic_stress",
    "hook.font",
    "graphics.font",
)
UNMEASURED = (
    tuple(
        {"field": field, "reason": "Needs text recognition (OCR), which requires a new dependency."}
        for field in OCR_FIELDS
    )
    + (
        {
            "field": "captions.shadow_settings",
            "reason": "Shadows blend with the underlying picture.",
        },
        {
            "field": "captions.letter_spacing_px",
            "reason": "Needs recognised glyphs and font metrics.",
        },
        {"field": "captions.entry_animation", "reason": "Entry motion is not measured."},
        {"field": "hook.shadow_settings", "reason": "Shadows blend with the underlying picture."},
        {"field": "hook.letter_spacing", "reason": "Needs recognised glyphs and font metrics."},
        {"field": "hook.entry_animation", "reason": "Entry motion is not measured."},
        {"field": "chapters.style", "reason": "Chapter cards are a creator choice."},
        {"field": "chapters.hold_s", "reason": "Chapter holds are not measured from pixels."},
        {
            "field": "graphics.style",
            "reason": "Panel appearance does not identify a graphics style.",
        },
        {"field": "music.settings", "reason": "Music separation is not available."},
        {"field": "pacing.smooth_zooms", "reason": "Only punch-ins at cuts are measured."},
    )
    + tuple(
        {"field": f"captions.{key}", "reason": reason}
        for key, reason in (
            ("shadow_opacity", "Shadows blend with the underlying picture."),
            ("shadow_blur_px", "Shadows blend with the underlying picture."),
            ("shadow_x_px", "Shadows blend with the underlying picture."),
            ("shadow_y_px", "Shadows blend with the underlying picture."),
            ("entry_rise_px", "Entry motion is not measured."),
            ("entry_ms", "Entry motion is not measured."),
        )
    )
    + tuple(
        {"field": f"broll.{key}", "reason": "Timeline choices are not measured from pixels."}
        for key in (
            "cutout",
            "screen_recording_placement",
            "clip_placement",
            "photo_placement",
            "screenshot_placement",
            "overlay_box",
            "overlay_motion",
            "tall_framing",
            "still_push",
            "still_push_scale",
            "avoid_tags",
            "stock_clips",
            "generated_pictures",
            "emphasis_on",
            "color_match",
            "green_screen_layout",
        )
    )
)

Observation = tuple[Any, list[int], dict[str, Any]]


@dataclass(frozen=True)
class LearnResult:
    """Three proposed artifacts and their portable measurement summary."""

    style_path: Path
    proposals_path: Path
    measurements_path: Path
    measurements: dict[str, Any]
    proposed_rules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the command's JSON summary without changing creator settings."""
        proposal = self.measurements["proposed_style"]
        return {
            "proposed_rules": list(self.proposed_rules),
            "fields": self.measurements["fields"],
            "measures": self.measurements["measures"],
            "not_measured": self.measurements["not_measured"],
            "videos": self.measurements["videos"],
            "proposed_style": proposal,
            "style_base": proposal["base"]["name"],
            "measured_fields": [
                k for k, v in proposal["sources"].items() if v["source"] == "measured"
            ],
            "base_fields": [k for k, v in proposal["sources"].items() if v["source"] == "base"],
            "paths": {
                "style": str(self.style_path),
                "proposals": str(self.proposals_path),
                "measurements": str(self.measurements_path),
            },
        }


def tolerance(field: str) -> float:
    """Return the agreement tolerance in pixels, seconds, CIE76 or exact values."""
    if field in measure_words.LIMIT_FIELDS:
        return 0
    if field.endswith("color"):
        if field == "captions.text_color" or field == "graphics.paper_color":
            return COLOR_TEXT_TOLERANCE
        if field.startswith("hook.") and field != "hook.background_color":
            return COLOR_TITLE_TOLERANCE
        return COLOR_HIGHLIGHT_TOLERANCE
    if field == "captions.top_px":
        return CAPTION_TOP_TOLERANCE
    if field == "captions.size_px":
        return CAPTION_SIZE_TOLERANCE
    if field == "hook.size_px":
        return TITLE_SIZE_TOLERANCE
    if field == "graphics.card_width_px":
        return PANEL_WIDTH_TOLERANCE
    if field == "graphics.corner_radius_px":
        return CORNER_RADIUS_TOLERANCE
    if field == "hook.duration_s":
        return TITLE_DURATION_TOLERANCE
    if field == "graphics.hold_s":
        return DURATION_TOLERANCE
    if field.startswith("loudness."):
        return LUFS_TOLERANCE
    if field == "pacing.median_shot_length_s":
        return 1 / 30
    return 0


def aggregate(
    field: str,
    rows: list[dict[str, Any]],
    method: str,
    *,
    showing: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    """Require pairwise agreement; retain every observation when a field varies."""
    values = [row["value"] for row in rows]
    limit = tolerance(field)
    colors = field.endswith("color")
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if colors:
        agrees = all(delta_e(a, b) <= limit for a, b in combinations(values, 2))
        value = hex_color(np.median([rgb(v) for v in values], axis=0))
    elif numeric:
        agrees = max(values) - min(values) <= limit
        value = float(np.median(values))
        if all(isinstance(v, int) for v in values):
            value = round(value)
    else:
        agrees = all(v == values[0] for v in values)
        value = values[0] if agrees else None
    measured = len(rows)
    showing = measured if showing is None else showing
    partial = measured < showing
    feature = {"hook": "a title", "graphics": "cards"}.get(field.split(".")[0], "captions")
    return {
        "status": "partial" if partial else "proposed" if agrees else "varies",
        "value": round_seconds(value, field) if agrees else None,
        "tolerance": round_seconds(limit, field),
        "method": method,
        "per_video": rows,
        "coverage": {"measured": measured, "showing": showing, "videos": total or showing},
        **(
            {"reason": f"measured in {measured} of {showing} videos that show {feature}"}
            if partial
            else {}
        ),
    }


def round_seconds(value: Any, name: str = "") -> Any:
    """Round second-valued scalars and nested timings at the output boundary."""
    if isinstance(value, dict):
        return {key: round_seconds(item, key) for key, item in value.items()}
    if isinstance(value, list):
        return [round_seconds(item, name) for item in value]
    if isinstance(value, float) and name.endswith("_s"):
        return round(value, 3)
    return value


def apply_dependencies(fields: dict[str, Any]) -> None:
    """Keep caption offsets tied to their measured position across all videos."""
    offset = fields.get("captions.top_px")
    if offset is None:
        return
    position = fields.get("captions.position", {})
    positions = {row["video"]: row["value"] for row in position.get("per_video", [])}
    if position.get("status") != "proposed" or any(
        positions.get(row["video"]) != position.get("value") for row in offset["per_video"]
    ):
        offset["detail"] = {
            "reason": "Caption top_px depends on a position that varies or was not measured."
        }
        if offset["status"] != "partial":
            offset["status"] = "varies"
            offset["value"] = None
            offset["reason"] = offset["detail"]["reason"]


def _median_samples(samples: list[tuple[int, Any]], *, color: bool = False) -> Observation:
    values = [v for _, v in samples]
    if color:
        value = hex_color(np.median([rgb(v) for v in values], axis=0))
    elif isinstance(values[0], (int, float)):
        value = float(np.median(values))
    else:
        value = max(values, key=values.count)
    return value, sorted({i for i, _ in samples}), {}


def _highlight_samples(samples: list[tuple[int, str]]) -> Observation | None:
    """Select a uniquely supported highlight and its evidence frames."""
    if samples and len(samples) < HIGHLIGHT_MIN_SAMPLES:
        if len({look for _, look in samples}) != 1:
            return None
        return samples[0][1], sorted({i for i, _ in samples}), {}
    # A "none" sample can lack a stress word, so it is not an opposing highlight vote.
    leaders = Counter(look for _, look in samples if look != "none").most_common(2)
    minimum = max(HIGHLIGHT_MIN_SAMPLES, math.ceil(HIGHLIGHT_MIN_SHARE * len(samples)))
    style = "none"
    if (
        leaders
        and leaders[0][1] >= minimum
        and (len(leaders) == 1 or leaders[0][1] > leaders[1][1])
    ):
        style = leaders[0][0]
    return style, sorted({i for i, look in samples if style == "none" or look == style}), {}


def _window(
    path: Path, info: ffmpeg.MediaInfo, panel: Panel, observations: list[int], step: int
) -> tuple[int, int, list[int]]:
    """Refine both disappearance boundaries on exact decoded original frames."""
    count = info.video_frames or round((info.duration_s or 0) * float(info.fps or 30))
    first, last = min(observations), max(observations)
    reference = None
    for _, image in iter_rgb(path, info, [observations[len(observations) // 2]], width=360):
        ratio = 360 / (info.width or 1080)
        x, y, w, h = (round(v * ratio) for v in (panel.x, panel.y, panel.width, panel.height))
        reference = image[y : y + h, x : x + w].copy()
    indices = sorted(
        set(range(max(0, first - step), first + 1)) | set(range(last, min(count, last + step + 1)))
    )
    present = [
        i
        for i, image in iter_rgb(path, info, indices, width=360)
        if panel_present(image, panel, reference_width=info.width or 1080, reference=reference)
    ]
    start, end = min([first, *present]), max([last, *present])
    # Sparse discovery can miss a translucent fill on a bright part of the
    # picture. Continue the boundary search until the panel actually vanishes.
    while end >= indices[-1] - 1 and end < count - 1:
        extension = list(range(end + 1, min(count, end + step + 1)))
        extra = [
            i
            for i, image in iter_rgb(path, info, extension, width=360)
            if panel_present(image, panel, reference_width=info.width or 1080, reference=reference)
        ]
        if not extra:
            break
        end = max(extra)
        if end < extension[-1]:
            break
    # Ink amplitude follows opacity after correlation falls below the presence
    # threshold. Decode a short tail to estimate its background baseline and
    # extrapolate the decline to zero instead of clipping a faint exit fade.
    fade_indices = list(
        range(
            max(start, end - round(float(info.fps or 30) * 0.5)),
            min(count, end + round(float(info.fps or 30) * 0.75) + 1),
        )
    )
    fade_evidence = set()
    if reference is not None and end < count - 1:
        amplitudes = [
            (
                i,
                panel_edge_match(
                    image, panel, reference_width=info.width or 1080, reference=reference
                )[0],
            )
            for i, image in iter_rgb(path, info, fade_indices, width=360)
        ]
        fade_evidence.update(i for i, _ in amplitudes)
        end = fade_end(amplitudes, end)
    return start, end, sorted(set(observations) | fade_evidence | {start, end})


def _visual(
    path: Path, info: ffmpeg.MediaInfo
) -> tuple[dict[str, Observation], dict[str, Observation], list[dict[str, str]]]:
    fps, width = float(info.fps or 30), info.width or 1080
    scale = 1080 / width
    count = info.video_frames or round((info.duration_s or 0) * fps)
    step = max(1, round(fps * 0.8))
    indices = list(range(max(1, round(fps * 0.5)), count, step)) or [0]
    discovery = CaptionDiscovery(round((info.height or 1920) * 360 / width), 360)
    panel_frames = set()
    panel_diagnostics: dict[str, Any] = {
        "discovery_frames_examined": len(indices),
        "frames_with_panel_candidate": 0,
        "panels_with_text": 0,
        "rejections": {},
    }
    for index, image in iter_rgb(path, info, indices, width=360):
        discovery.add(image)
        if find_panels(image, detail=panel_diagnostics, discovery=True):
            panel_frames.add(index)
    panel_diagnostics["frames_with_panel_candidate"] = len(panel_frames)
    band = discovery.band()
    captions = []
    panels: list[tuple[Panel, list[int], list[dict[str, Any]]]] = []
    for index, image in iter_rgb(path, info, indices):
        sample = caption_band(image, band) if band else None
        if sample:
            captions.append((index, sample))
        for panel in find_panels(image, detail=panel_diagnostics) if index in panel_frames else []:
            text = panel_text(
                image[panel.y : panel.y + panel.height, panel.x : panel.x + panel.width],
                np.asarray(panel.fill),
                rotated=panel.rotated,
                require_glyphs=index < fps,
            )
            if "text_color" not in text:
                rejected = panel_diagnostics["rejections"]
                rejected["text_colour"] = rejected.get("text_colour", 0) + 1
                continue
            panel_diagnostics["panels_with_text"] += 1
            match = next(
                (
                    p
                    for p in panels
                    if abs(p[0].y - panel.y) * scale < 16
                    and abs(p[0].width - panel.width) * scale < 20
                    and np.linalg.norm(np.asarray(p[0].fill) - panel.fill) < 20
                    and index - p[1][-1] <= step * 3
                ),
                None,
            )
            if match:
                match[1].append(index)
                match[2].append({**text, "fill": panel.fill})
            else:
                panels.append((panel, [index], [{**text, "fill": panel.fill}]))
    if captions:
        # Repeated foreground colours survive picture changes. A single frame
        # cannot decide whether a skin or clothing cluster belongs to an overlay.
        colors = [s.color for _, s in captions]
        hint = max(colors, key=lambda c: sum(delta_e(c, other) < 6 for other in colors))
        accents = [s.highlight_color for _, s in captions if s.highlight_color]
        accent = (
            max(accents, key=lambda c: sum(delta_e(c, other) < 8 for other in accents))
            if accents
            else None
        )
        looks = [s.highlight for _, s in captions if s.highlight != "none"]
        look_hint = max(looks, key=looks.count) if looks else "none"
        refined_captions = []
        for index, image in iter_rgb(path, info, indices):
            sample = caption_band(image, band, hint, accent, look_hint)
            if sample:
                refined_captions.append((index, sample))
        captions = refined_captions
    fields: dict[str, Observation] = {}
    measures: dict[str, Observation] = {}
    missing = []
    caption_diagnostics = {
        "samples_examined": len(indices),
        "caption_samples": len(captions),
        "samples_without_caption": len(indices) - len(captions),
        "samples_with_two_glyph_rows": sum(
            s.pitch_detail.get("band_count") == 2 for _, s in captions
        ),
        "samples_with_pitch": sum(s.pitch is not None for _, s in captions),
        "rejections": dict(
            Counter(
                s.pitch_detail.get("reason", "band_count") for _, s in captions if s.pitch is None
            )
        ),
        "band_counts": dict(Counter(str(s.pitch_detail.get("band_count", 0)) for _, s in captions)),
    }
    pitch_samples = [s.pitch_detail for _, s in captions if s.pitch is not None]
    if pitch_samples:
        caption_diagnostics["minimum_margins"] = {
            key: min(sample[key] for sample in pitch_samples)
            for key in ("height_margin", "height_ratio_margin", "pitch_search_margin")
        }
    if captions:
        fields["captions.text_color"] = _median_samples(
            [(i, s.color) for i, s in captions], color=True
        )
        highlight_style = _highlight_samples(
            [(i, s.highlight) for i, s in captions]
        )
        if highlight_style is None:
            missing.append({
                "field": "captions.highlight_style",
                "reason": "Too few caption samples to separate a highlight "
                "from unstressed phrases.",
            })
        else:
            fields["captions.highlight_style"] = highlight_style
        highlight = highlight_style[0] if highlight_style is not None else None
        for field, attribute in (
            ("captions.highlight_color", "highlight_color"),
            ("captions.highlight_text_color", "highlight_text_color"),
        ):
            if highlight is None:
                missing.append({
                    "field": field,
                    "reason": "Highlight style was not measured; cannot identify its colours.",
                })
                continue
            samples = [
                (i, getattr(s, attribute))
                for i, s in captions
                if s.highlight == highlight and getattr(s, attribute) is not None
            ]
            if samples and (attribute != "highlight_text_color" or highlight == "moving_block"):
                fields[field] = _median_samples(samples, color=True)
            else:
                missing.append(
                    {"field": field, "reason": f"No visible {attribute} for {highlight}."}
                )
        two_lines = [(i, s) for i, s in captions if s.pitch is not None]
        evidence = [i for i, _ in captions]
        position = (
            "middle" if float(np.median([
                (s.last_line_top if s.last_line_top is not None else s.line_top) * scale
                for _, s in captions
            ])) < 1300 else "bottom"
        )
        fields["captions.position"] = (position, evidence, {})
        band_top = round(float(np.percentile([s.top for _, s in captions], 2)) * scale)
        band_bottom = round(float(np.percentile([s.bottom for _, s in captions], 98)) * scale)
        measures["captions.band"] = (
            {"top_px": band_top, "bottom_px": band_bottom, "height_px": band_bottom - band_top},
            evidence,
            {"bounds": "2nd and 98th percentiles of glyph rows, including two-line phrases"},
        )
        defaults = resolve_style_data({"schema_version": 1, "name": "Geometry"})
        base = defaults["captions"]
        size = base["size_px"]
        if two_lines:
            size = round(float(np.median([
                s.pitch * scale / CAPTION_LINE_HEIGHT for _, s in two_lines
            ])))
            fields["captions.size_px"] = (
                size,
                [i for i, _ in two_lines],
                {"line_height": CAPTION_LINE_HEIGHT, **caption_diagnostics},
            )
        else:
            missing.append({
                "field": "captions.size_px", "reason": "No two-line caption was found.",
                "detail": caption_diagnostics,
            })
        if two_lines or position == "bottom":
            # Centre glyph ink in the line box. This uses measured pitch, never
            # infers size from glyph height. One-line-only bottom footage uses
            # the default size for an explicitly labelled position estimate.
            samples = two_lines or captions
            first_tops, last_tops = [], []
            for _, sample in samples:
                pitch = (sample.pitch * scale if sample.pitch is not None
                         else size * CAPTION_LINE_HEIGHT)
                first_tops.append(sample.line_top * scale -
                                  (pitch - sample.glyph_height * scale) / 2)
                last_ink_top = (sample.last_line_top if sample.last_line_top is not None
                                else sample.line_top + (sample.pitch or 0))
                last_height = (sample.last_glyph_height if sample.last_glyph_height is not None
                               else sample.glyph_height)
                last_tops.append(last_ink_top * scale - (pitch - last_height * scale) / 2)
            first_top, last_top = float(np.median(first_tops)), float(np.median(last_tops))
            position = "middle" if last_top < 1300 else "bottom"
            geometry = caption_band_geometry(
                caption_layout_settings({**base, "size_px": size, "position": position},
                                        defaults["brand"]),
                safe_bottom_px=defaults["internal"]["caption_safe_bottom_px"],
            )
            line_top = last_top if position == "bottom" else first_top
            top_px = round(geometry.top_px_from_line_box(line_top))
            evidence = [i for i, _ in samples]
            fields["captions.position"] = (position, evidence, {})
            fields["captions.top_px"] = (
                top_px,
                evidence,
                {
                    "line_box_top_px": line_top,
                    "first_line_box_top_px": first_top,
                    "last_line_box_top_px": last_top,
                    "padding_px": geometry.padding,
                    "engine_band_height_px": geometry.height,
                    "engine_band_line_capacity": 2,
                    "line_count": 2 if two_lines else 1,
                    "size_source": "measured" if two_lines else "default",
                    "size_used_px": size,
                    **({"size_reason": "No two-line caption was found; the engine default "
                        "size estimates the line box."} if not two_lines else {}),
                    "geometry": (
                        "bottom anchor uses the last line box; two-line phrases grow upward"
                        if position == "bottom" else "middle band stays centered on top_px"
                    ) + "; unmeasured settings use engine defaults",
                    **caption_diagnostics,
                },
            )
        else:
            missing.append({
                "field": "captions.top_px", "reason": "No two-line caption was found.",
                "detail": caption_diagnostics,
            })
    else:
        missing.append({"field": "captions", "reason": "No stable caption glyph band was found."})
    cards = []
    opening = [p for p in panels if p[1][0] < fps and len(p[1]) >= 2]
    title = max(opening, key=lambda p: len(p[1]), default=None)
    panel_diagnostics["title_frames"] = len(title[1]) if title else 0
    panel_diagnostics["title_frames_with_pitch"] = (
        sum("pitch" in text for text in title[2]) if title else 0
    )
    panel_diagnostics["title_pitch_rejections"] = dict(
        Counter(
            text.get("pitch_detail", {}).get("reason", "band_count")
            for text in (title[2] if title else [])
            if "pitch" not in text
        )
    )
    for panel, evidence, texts in panels:
        if len(evidence) < 2:
            continue
        if evidence[0] < fps and title is not None and panel is not title[0]:
            continue
        first, last, refined = _window(path, info, panel, evidence, step)
        is_title = first < fps
        if is_title and (title is None or panel is not title[0]):
            continue
        box = panel.box(scale)
        window = {"box": box, "start_frame": first, "end_frame": last}
        if is_title:
            look = (
                "bold_band"
                if panel.rotated
                else ("quiet_serif" if panel.width * scale < 820 else "paper_title")
            )
            anchor = (
                panel.y * scale + panel.width * scale * math.tan(math.radians(2.5)) / 2
                if panel.rotated
                else panel.y * scale
            )
            position = "center" if anchor > 500 else ("top" if anchor < 225 else "headroom")
            fields["hook.style"] = (
                look,
                evidence,
                {
                    "panel_margins": panel.margins,
                    "text_margins": texts[0].get("colour_margins", {}),
                    "pitch": texts[0].get("pitch_detail", {}),
                },
            )
            fields["hook.position"] = (position, evidence, {"anchor_px": anchor})
            fields["hook.duration_s"] = ((last + 1) / fps, refined, {"end_frame": last})
            fills = [np.asarray(t["fill"]) for t in texts]
            fill = np.median(fills, axis=0)
            if look == "quiet_serif":
                # Estimate the background locally outside the settled panel.
                for _, image in iter_rgb(path, info, [evidence[len(evidence) // 2]]):
                    margin = max(1, round(10 / scale))
                    adjacent = image[
                        panel.y : panel.y + panel.height, max(0, panel.x - margin) : panel.x
                    ]
                    if adjacent.size:
                        background = np.median(adjacent.reshape(-1, 3), axis=0)
                        fill = (fill - 0.06 * background) / 0.94
            fields["hook.background_color"] = (
                hex_color(fill),
                evidence,
                {"opacity": 0.94 if look == "quiet_serif" else 1},
            )
            for key in ("text_color", "accent_color"):
                values = [(i, t[key]) for i, t in zip(evidence, texts, strict=True) if key in t]
                if values:
                    fields[f"hook.{key}"] = _median_samples(values, color=True)
                else:
                    missing.append(
                        {
                            "field": f"hook.{key}",
                            "reason": "No stable colour core found.",
                            "detail": panel_diagnostics,
                        }
                    )
            pitches = [
                (i, t["pitch"] * scale)
                for i, t in zip(evidence, texts, strict=True)
                if "pitch" in t
            ]
            if pitches:
                line_height = 0.98 if look == "bold_band" else 1.04
                size = round(
                    float(np.median([v for _, v in pitches]))
                    / line_height
                    / (0.72 if look == "quiet_serif" else 1)
                )
                fields["hook.size_px"] = (
                    size,
                    [i for i, _ in pitches],
                    {
                        "line_height": line_height,
                        "quiet_serif_factor": 0.72 if look == "quiet_serif" else 1,
                    },
                )
            else:
                missing.append(
                    {
                        "field": "hook.size_px",
                        "reason": "No two-line title was found.",
                        "detail": panel_diagnostics,
                    }
                )
            measures["hook.box"] = (box, evidence, {})
            measures["hook.window"] = (window, refined, {})
        else:
            cards.append((window, evidence, texts, refined))
    if cards:
        measures["graphics.cards"] = (
            [c[0] for c in cards],
            sorted({i for c in cards for i in c[3]}),
            {},
        )
        for key, text_key in (
            ("paper_color", "fill"),
            ("ink_color", "text_color"),
            ("accent_color", "accent_color"),
        ):
            samples = [
                (i, hex_color(t[text_key]) if text_key == "fill" else t[text_key])
                for _, evidence, texts, _ in cards
                for i, t in zip(evidence, texts, strict=True)
                if text_key in t
            ]
            if samples:
                fields[f"graphics.{key}"] = _median_samples(samples, color=True)
        fields["graphics.card_width_px"] = (
            round(float(np.median([c[0]["box"]["width"] for c in cards]))),
            sorted({i for c in cards for i in c[1]}),
            {},
        )
        fields["graphics.hold_s"] = (
            float(np.median([(c[0]["end_frame"] - c[0]["start_frame"] + 1) / fps for c in cards])),
            sorted({i for c in cards for i in c[3]}),
            {},
        )
    expected = (
        "captions.text_color",
        "captions.highlight_style",
        "captions.highlight_color",
        "captions.highlight_text_color",
        "captions.position",
        "captions.top_px",
        "captions.size_px",
        "hook.style",
        "hook.position",
        "hook.duration_s",
        "hook.background_color",
        "hook.text_color",
        "hook.accent_color",
        "hook.size_px",
        "graphics.paper_color",
        "graphics.ink_color",
        "graphics.accent_color",
        "graphics.card_width_px",
        "graphics.hold_s",
    )
    recorded = {row["field"] for row in missing}
    for field in expected:
        if field not in fields and field not in recorded:
            missing.append(
                {
                    "field": field,
                    "reason": "No card was found."
                    if field.startswith("graphics.")
                    else "No stable panel with text was found in the opening second."
                    if field.startswith("hook.") and title is None
                    else "No stable visible evidence was found.",
                    **(
                        {"detail": panel_diagnostics}
                        if field.startswith("hook.")
                        else {"detail": caption_diagnostics}
                        if field in {"captions.size_px", "captions.top_px"}
                        else {}
                    ),
                }
            )
    return fields, measures, missing


def compact_evidence(frames: Sequence[int], detail: dict[str, Any]) -> list[int]:
    """Keep event indices, evenly sampling unusually large event lists."""
    result = sorted(set(frames))
    if len(result) > 10000:
        detail["evidence_sampling"] = {
            "method": "even sample",
            "event_frames": len(result),
            "retained_frames": 10000,
        }
        result = [result[i] for i in np.linspace(0, len(result) - 1, 10000, dtype=int)]
    return result


def _proposals(measures: dict[str, Any], videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name in ("pacing.median_shot_length_s", "pacing.zoom_punch_ins", "pacing.pause_s"):
        rows = measures.get(name, {}).get("per_video", [])
        if not rows:
            continue
        if name.endswith("zoom_punch_ins"):
            if not any(row["value"]["count"] for row in rows):
                continue
            value = float(np.median([row["value"]["per_minute"] for row in rows]))
            sentence = f"Consider about {value:.1f} centre punch-ins per minute."
        elif name.endswith("pause_s"):
            rows = [r for r in rows if r["value"]["median_s"] is not None]
            if not rows:
                continue
            value = float(np.median([r["value"]["median_s"] for r in rows]))
            sentence = f"Consider internal speech pauses around {value:.2f} seconds."
        else:
            value = float(np.median([r["value"] for r in rows]))
            sentence = f"Consider shots around {value:.2f} seconds long."
        sources = [
            {
                "type": "measurement",
                "media_sha256": videos[row["video"]]["sha256"],
                "media_name": videos[row["video"]]["name"],
                "measure": name,
                "frames": row["frames"],
            }
            for row in rows
        ]
        body = {
            "scope": "pacing",
            "condition": None,
            "sentence": sentence,
            "machine": None,
            "context": (
                f"Measured in {len(rows)} of {len(videos)} finished "
                f"{'video' if len(videos) == 1 else 'videos'}; a candidate, not a rule yet."
            ),
            "sources": sources,
        }
        output.append(validate_body(body))
    return output


def _timed_visual(
    fields: dict[str, Observation], measures: dict[str, Observation], timing: FrameTiming
) -> None:
    """Replace rate-derived overlay durations with presentation-time windows."""
    if "hook.window" in measures and "hook.duration_s" in fields:
        window = measures["hook.window"][0]
        _, frames, detail = fields["hook.duration_s"]
        fields["hook.duration_s"] = (timing.boundary(window["end_frame"] + 1), frames, detail)
    if "graphics.cards" in measures and "graphics.hold_s" in fields:
        holds = [
            timing.window_s(card["start_frame"], card["end_frame"])
            for card in measures["graphics.cards"][0]
        ]
        if holds:
            _, frames, detail = fields["graphics.hold_s"]
            fields["graphics.hold_s"] = (float(np.median(holds)), frames, detail)


def _caption_font_reason(
    value: dict[str, str] | None, missing: Sequence[dict[str, Any]]
) -> str:
    """Explain the measured font class while keeping exact typefaces unmeasured."""
    if value is not None:
        reason = f"Measured font class: {value['family']}, {value['weight']} weight."
    else:
        reasons = dict.fromkeys(
            row["reason"] for row in missing if row["field"] == "captions.font_class"
        )
        reason = " ".join(reasons) or "Measured font classes vary across videos."
    return reason + " Naming the exact typeface needs text recognition (OCR)."


def measure_style(path: Path, fields: Collection[str] | None = None) -> dict[str, dict]:
    """Measure one finished vertical picture without speech, sound or output files."""
    with paths.operation():
        source = paths.validate_input(Path(path).expanduser())
        info = ffmpeg.probe(source)
        if not info.has_video or not info.fps or not info.width or not info.height:
            raise InputError(str(source), "no usable video stream")
        if info.height <= info.width:
            raise InputError(
                str(source), "video must be vertical (displayed height greater than width)"
            )
        if not info.video_frames or not info.duration_s:
            raise InputError(str(source), "video must contain frames and a positive duration")
        requested = (
            list(dict.fromkeys(fields))
            if fields is not None
            else [
                f"{section}.{key}"
                for section in ("captions", "hook", "chapters", "graphics", "broll", "music")
                for key, rule in loader._schema()["properties"][section]["properties"].items()
                if rule["type"] != "object"
            ]
        )
        if not requested:
            return {}
        observed, measures, absent = _visual(source, info)
        _timed_visual(observed, measures, presentation_timing(source, info))
        reasons = {row["field"]: row["reason"] for row in (*UNMEASURED, *absent)}
        reasons.update(dict.fromkeys(measure_words.LIMIT_FIELDS, measure_words.NEEDS_WORDS))
        if "captions.font" in requested:
            shape_measures, shape_missing = measure_shapes.measure(source, info, observed, measures)
            font_class = shape_measures.get("captions.font_class")
            reasons["captions.font"] = _caption_font_reason(
                font_class[0] if font_class else None, shape_missing
            )
        result = {}
        for field in requested:
            if field in observed:
                value, frames, detail = observed[field]
                result[field] = {
                    "value": round_seconds(value, field),
                    "frames": compact_evidence(frames, detail),
                }
            else:
                result[field] = {
                    "value": None,
                    "reason": reasons.get(field, "Not measured from pixels."),
                }
        return result


def _base_style(base: str | Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve an explicit preset or protected file, otherwise use schema defaults."""
    if base is None:
        return resolve_style_data({"schema_version": 1}), {
            "kind": "schema_defaults",
            "name": "Schema defaults",
        }
    if str(base).startswith("preset:"):
        resolved = resolve(str(base))
        descriptor = {"kind": "preset", "name": resolved.name, "id": str(base)[7:]}
    else:
        source = paths.validate_input(Path(base).expanduser())
        resolved = resolve(source)
        descriptor = {"kind": "file", "name": resolved.name, "file_name": source.name}
    return resolved.data, descriptor


def _outside_style(field: str, value: Any) -> str | None:
    """Reject only this leaf when a measured value cannot fit the style schema."""
    try:
        validate_style_field(field, value, "measurement")
    except TasteError as exc:
        section, key = field.split(".")
        rule = loader._schema()["properties"][section]["properties"][key]
        unit = " s" if field.endswith("_s") else ""
        shown = f"{value:g}{unit}" if isinstance(value, (int, float)) else repr(value)
        for bound, comparison, direction in (
            ("maximum", lambda a, b: a > b, "above"),
            ("minimum", lambda a, b: a < b, "below"),
        ):
            if bound in rule and isinstance(value, (int, float)) and comparison(value, rule[bound]):
                return f"measured {shown}, {direction} the style's {rule[bound]}{unit} limit"
        if "enum" in rule:
            return f"measured {shown}, not an allowed choice for this style field"
        return f"measured {shown}, {exc.detail}"
    return None


def _proposed_style(
    fields: dict[str, Any], base: dict[str, Any], descriptor: dict[str, Any], missing: list[dict]
) -> dict[str, Any]:
    """Build a complete style and a source decision for every editable leaf."""
    style = {key: copy.deepcopy(value) for key, value in base.items() if key != "internal"}
    style.update(schema=RESOLVED_SCHEMA, name="Learned style (proposed)")
    for field, data in fields.items():
        if data["status"] == "proposed":
            reason = _outside_style(field, data["value"])
            if reason:
                data.update(status="outside_style", reason=reason)
    apply_dependencies(fields)
    for field, data in fields.items():
        if data["status"] == "proposed":
            section, key = field.split(".")
            style[section][key] = data["value"]
    resolve_style_data(style)
    sources = {}
    for section in ("captions", "hook", "chapters", "graphics", "broll", "music"):
        for key in style[section]:
            field = f"{section}.{key}"
            data = fields.get(field)
            measured = data is not None and data["status"] == "proposed"
            reasons = list(dict.fromkeys(row["reason"] for row in missing if row["field"] == field))
            reason = (
                None
                if measured
                else (
                    data.get("reason", "Measured values vary across videos.")
                    if data
                    else " ".join(reasons) or "This choice is not measured from finished videos."
                )
            )
            sources[field] = {
                "source": "measured" if measured else "base",
                "status": data["status"] if data else None,
                "reason": reason,
            }
    return {"base": descriptor, "style": style, "sources": sources}


def learn(
    videos: Sequence[Path],
    out_dir: Path,
    *,
    asr: str = "auto",
    force: bool = False,
    base: str | Path | None = None,
    transcripts: Sequence[Path | None] | None = None,
) -> LearnResult:
    """Write measured proposals without saving or approving any creator preference."""
    if not videos:
        raise TasteError("videos", "provide at least one finished vertical video")
    if len(videos) > 50:
        raise TasteError("videos", "provide at most 50 videos per proposal batch")
    if transcripts is not None and len(transcripts) != len(videos):
        raise TasteError("transcripts", "provide one words file or None per video, in video order")
    with paths.operation():
        base_data, base_descriptor = _base_style(base)
        inputs = [paths.validate_input(Path(p).expanduser()) for p in videos]
        words_files = [
            paths.validate_input(Path(p).expanduser()) if p is not None else None
            for p in (transcripts if transcripts is not None else [None] * len(inputs))
        ]
        infos = [ffmpeg.probe(p) for p in inputs]
        for path, info in zip(inputs, infos, strict=True):
            if not info.has_video or not info.fps or not info.width or not info.height:
                raise InputError(str(path), "no usable video stream")
            if info.height <= info.width:
                raise InputError(
                    str(path), "video must be vertical (displayed height greater than width)"
                )
            if not info.video_frames or not info.duration_s:
                raise InputError(str(path), "video must contain frames and a positive duration")
        directory = Path(out_dir).expanduser()
        config = config_root().resolve()
        if directory.resolve() == config or config in directory.resolve().parents:
            raise TasteRefusal(
                "learn cannot write to the creator's config folder.",
                code="taste_learn_config",
                suggested_action="Choose a separate --out folder.",
            )
        targets = [
            directory / name
            for name in ("proposed-style.json", "taste-proposals.json", "measurements.json")
        ]
        for target in targets:
            if os.path.lexists(target) and not force:
                raise TasteRefusal(
                    f"Output already exists: {target}",
                    code="taste_learn_exists",
                    suggested_action="Choose a new --out folder or use --force.",
                )
            paths.validate_output(target)
        backend = resolve_backend(asr) if any(p is None for p in words_files) else None
        descriptions = []
        all_fields: dict[str, list[dict[str, Any]]] = {}
        all_measures: dict[str, list[dict[str, Any]]] = {}
        missing = list(UNMEASURED)
        showing: Counter[str] = Counter()
        for index, (path, info) in enumerate(zip(inputs, infos, strict=True)):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            timing = presentation_timing(path, info)
            descriptions.append(
                {
                    "name": path.name,
                    "sha256": digest,
                    "frames": info.video_frames,
                    "fps": float(info.fps),
                    "width": info.width,
                    "height": info.height,
                    "scale": 1080 / info.width,
                    "duration_s": round(timing.end_s, 3),
                    "timing": {
                        "from_timestamps": timing.from_timestamps,
                        "variable_rate": timing.variable_rate,
                    },
                }
            )
            fields, measures, absent = _visual(path, info)
            shape_measures, shape_missing = measure_shapes.measure(path, info, fields, measures)
            measures.update(shape_measures)
            absent.extend(shape_missing)
            _timed_visual(fields, measures, timing)
            for section, feature in (
                ("captions", "captions.band"),
                ("hook", "hook.box"),
                ("graphics", "graphics.cards"),
            ):
                if feature in measures or any(key.startswith(section + ".") for key in fields):
                    showing[section] += 1
            boxes = [row[0] for key, row in measures.items() if key == "hook.box"]
            if "captions.band" in measures:
                band = measures["captions.band"][0]
                boxes.append(
                    {
                        "x": 0,
                        "y": band["top_px"] - 20,
                        "width": 1080,
                        "height": band["height_px"] + 40,
                    }
                )
            boxes.extend(c["box"] for c in measures.get("graphics.cards", ([], [], {}))[0])
            measures.update(measure_pacing.measure(path, info, boxes, timing=timing))
            transcript = None
            engine = "words file"
            if words_files[index] is not None:
                transcript = Transcript.load(words_files[index])
            elif backend and info.has_audio:
                transcript = backend.transcribe_many([path])[path]
                engine = backend.name
            if transcript is not None:
                for key, value in measure_pacing.transcript_pacing(
                    transcript, timing.end_s
                ).items():
                    measures[key] = (
                        value,
                        (
                            [0, info.video_frames - 1]
                            if key.endswith("words_per_minute")
                            else sorted({timing.frame_at(gap["start_s"]) for gap in value})
                        ),
                        {
                            "engine": engine,
                            **(
                                {"frame_range": [0, info.video_frames - 1]}
                                if key.endswith("words_per_minute")
                                else {}
                            ),
                        },
                    )
            else:
                reason = "ASR disabled." if asr == "none" else "No speech engine installed."
                if not info.has_audio:
                    reason = "No audio stream."
                absent.extend(
                    {"field": field, "reason": reason}
                    for field in ("pacing.words_per_minute", "pacing.word_gap_pauses")
                )
            word_fields, word_measures, word_missing = measure_words.measure(
                path, info, fields, measures, transcript, timing=timing
            )
            fields.update(word_fields)
            measures.update(word_measures)
            absent.extend(word_missing)
            if not info.has_audio:
                absent.extend(
                    {"field": key, "reason": "No audio stream."}
                    for key in (
                        "pacing.pause_s",
                        "loudness.integrated_lufs",
                        "loudness.true_peak_dbtp",
                        "loudness.lra_lu",
                    )
                )
            missing.extend({**row, "video": index, "name": path.name} for row in absent)
            for dest, data in ((all_fields, fields), (all_measures, measures)):
                for field, (value, evidence, detail) in data.items():
                    dest.setdefault(field, []).append(
                        {
                            "video": index,
                            "value": round_seconds(value, field),
                            "frames": compact_evidence(evidence, detail),
                            "detail": round_seconds(detail),
                            **(
                                {"frame_range": detail["frame_range"]}
                                if "frame_range" in detail
                                else {}
                            ),
                        }
                    )
        fields = {
            key: aggregate(
                key,
                rows,
                "Glyph cores, panel geometry and exact frame windows.",
                showing=showing[key.split(".")[0]],
                total=len(inputs),
            )
            for key, rows in all_fields.items()
        }
        apply_dependencies(fields)
        measures = {
            key: aggregate(
                key,
                rows,
                "Decoded picture or sound with original frame evidence.",
                showing={
                    "captions.font_class": showing["captions"],
                    "graphics.corner_radius_px": showing["graphics"],
                }.get(key),
                total=len(inputs),
            )
            for key, rows in all_measures.items()
        }
        font_class = measures.get("captions.font_class")
        missing = [row for row in missing if row["field"] != "captions.font"]
        missing.append(
            {
                "field": "captions.font",
                "reason": _caption_font_reason(
                    font_class["value"]
                    if font_class and font_class["status"] == "proposed"
                    else None,
                    missing,
                ),
            }
        )
        style_proposal = _proposed_style(fields, base_data, base_descriptor, missing)
        style = style_proposal["style"]
        proposals = {"schema": PROPOSALS_SCHEMA, "proposals": _proposals(measures, descriptions)}
        proposed = {key for key, value in fields.items() if value["status"] == "proposed"}
        unique_missing = {}
        for row in missing:
            if "video" not in row and row["field"] in proposed:
                continue
            if row["field"] in {
                "graphics.paper_color",
                "graphics.ink_color",
                "graphics.accent_color",
                "graphics.card_width_px",
                "graphics.hold_s",
            } and not all_fields.get("graphics.paper_color"):
                row = {**row, "reason": "No card was found."}
            unique_missing[(row["field"], row.get("video"), row["reason"])] = row
        measurements = {
            "schema": MEASUREMENTS_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "videos": descriptions,
            "fields": fields,
            "measures": measures,
            "not_measured": list(unique_missing.values()),
            "proposed_style": style_proposal,
        }
        for target, data in zip(targets, (style, proposals, measurements), strict=True):
            try:
                paths.atomic_write_text(target, json.dumps(data, indent=2, allow_nan=False) + "\n")
            except OSError as exc:
                raise OutputPathError(
                    f"Cannot write proposed measurements: {exc}", code="unsafe_path"
                ) from exc
        return LearnResult(
            *targets, measurements, tuple(p["sentence"] for p in proposals["proposals"])
        )
