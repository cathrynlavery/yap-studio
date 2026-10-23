"""Measure caption stroke classes and card curvature from original picture evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.stages.captions import CAPTION_LINE_HEIGHT
from yap_studio.taste.frames import color_mask, interior_components, iter_rgb, rgb, runs
from yap_studio.taste.measure_text import line_pitch

# Regular stems occupy roughly 1/12 em; heavy stems approach 1/5 em.
# Diagonal monoline strokes broaden by sqrt(2), whereas hairlines are several
# times thinner than serif stems. Keep a refusal band around both decisions.
WEIGHT_THRESHOLD = 0.14
WEIGHT_AMBIGUITY = 0.025
CONTRAST_THRESHOLD = 2.1
CONTRAST_AMBIGUITY = 0.3
MIN_GLYPH_PIXELS = 500
FONT_FRAME_LIMIT = 24
CORNER_SPREAD_TOLERANCE = 6.0
CORNER_SQUARE_PX = 64
Observation = tuple[Any, list[int], dict[str, Any]]


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return the complete horizontal foreground run length at each pixel."""
    columns = np.arange(mask.shape[1])[None, :]
    starts = np.maximum.accumulate(np.where(mask, -1, columns), axis=1)
    ends = np.minimum.accumulate(np.where(mask, mask.shape[1], columns)[:, ::-1], axis=1)
    return np.where(mask, ends[:, ::-1] - starts - 1, 0)


def _glyph_height(mask: np.ndarray) -> float | None:
    """Estimate x-height from the median height of separate glyph column bands."""
    heights = []
    for first, last in runs(mask.any(axis=0), gap=1):
        rows = np.flatnonzero(mask[:, first:last].any(axis=1))
        if len(rows) >= 8:
            heights.append(int(rows[-1] - rows[0] + 1))
    return float(np.median(heights)) if heights else None


def font_class(
    masks: Iterable[tuple[int, np.ndarray]], *, size_px: float | None = None
) -> Observation:
    """Classify full-resolution glyph masks by stem weight and stroke contrast."""
    samples = []
    thicknesses = []
    heights = []
    pitches = []
    for index, mask in masks:
        if len(samples) >= FONT_FRAME_LIMIT:
            break
        horizontal = _run_lengths(mask)
        vertical = _run_lengths(mask.T).T
        shorter = np.minimum(horizontal, vertical)
        longer = np.maximum(horizontal, vertical)
        # Junctions and counters measure geometry rather than stem thickness.
        stems = mask & (longer >= shorter * 2) & (shorter > 0)
        values = shorter[stems]
        if len(values) < 50:
            continue
        thicknesses.append(values)
        height = _glyph_height(mask)
        pitch = line_pitch(mask)
        if height:
            heights.append(height)
        if pitch:
            pitches.append(pitch)
        samples.append(
            {
                "frame": index,
                "glyph_pixels": int(mask.sum()),
                "stem_pixels": len(values),
                "median_stem_px": float(np.median(values)),
            }
        )
    frames = sorted({sample["frame"] for sample in samples})
    detail: dict[str, Any] = {
        "frames": frames,
        "samples": samples,
        "minimum_glyph_pixels": MIN_GLYPH_PIXELS,
        "stroke_measure": "shorter of horizontal and vertical runs; aspect ratio >= 2",
        "contrast_percentiles": [15, 85],
    }
    count = sum(len(values) for values in thicknesses)
    detail["stem_pixels"] = count
    if count < MIN_GLYPH_PIXELS:
        detail["reason"] = "Too few caption glyph pixels to measure font class."
        return None, frames, detail
    values = np.concatenate(thicknesses)
    thin, median, thick = np.percentile(values, [15, 50, 85])
    if size_px is not None and size_px > 0:
        denominator, source, factor = float(size_px), "captions.size_px", 1.0
    elif pitches:
        denominator = float(np.median(pitches))
        source, factor = "measured line pitch", CAPTION_LINE_HEIGHT
    elif heights:
        denominator = float(np.median(heights))
        # Without recognised letters this is an x-height estimate; typical
        # lowercase bodies span about 0.55 em. Widen the refusal band below.
        source, factor = "estimated glyph x-height", 0.55
    else:
        detail["reason"] = "No caption size, line pitch or glyph x-height was measurable."
        return None, frames, detail
    weight, contrast = float(median / denominator), float(thick / thin)
    threshold = WEIGHT_THRESHOLD / factor
    ambiguity = WEIGHT_AMBIGUITY / factor
    if source == "estimated glyph x-height":
        ambiguity *= 1.5
    weight_margin, contrast_margin = weight - threshold, contrast - CONTRAST_THRESHOLD
    detail.update(
        thin_stroke_px=float(thin),
        median_stem_px=float(median),
        thick_stroke_px=float(thick),
        weight_ratio=weight,
        contrast_ratio=contrast,
        size_source=source,
        size_used_px=denominator,
        thresholds={"weight_ratio": threshold, "contrast_ratio": CONTRAST_THRESHOLD},
        ambiguity={"weight_ratio": ambiguity, "contrast_ratio": CONTRAST_AMBIGUITY},
        margins={"weight_ratio": weight_margin, "contrast_ratio": contrast_margin},
        clear_margins={
            "weight_ratio": abs(weight_margin) - ambiguity,
            "contrast_ratio": abs(contrast_margin) - CONTRAST_AMBIGUITY,
        },
    )
    uncertain = []
    if abs(weight_margin) <= ambiguity:
        uncertain.append("weight")
    if abs(contrast_margin) <= CONTRAST_AMBIGUITY:
        uncertain.append("family")
    if uncertain:
        detail["reason"] = "Ambiguous font " + " and ".join(uncertain) + " near a stroke threshold."
        return None, frames, detail
    return (
        {
            "family": "serif" if contrast_margin > 0 else "sans",
            "weight": "heavy" if weight_margin > 0 else "regular",
        },
        frames,
        detail,
    )


def _corner(
    image: np.ndarray, x: int, y: int, dx: int, dy: int, square: int, fill: np.ndarray
) -> dict[str, Any]:
    """Project a corner onto its local background-to-paper colour segment."""
    positions = np.arange(square)
    columns, rows = x + dx * positions, y + dy * positions
    outside = np.arange(2, 6)
    outside_x, outside_y = x - dx * outside, y - dy * outside
    height, width = image.shape[:2]
    if min(columns.min(), outside_x.min()) < 0 or max(columns.max(), outside_x.max()) >= width:
        return {"reason": "Card corner has no adjacent background inside the picture."}
    if min(rows.min(), outside_y.min()) < 0 or max(rows.max(), outside_y.max()) >= height:
        return {"reason": "Card corner has no adjacent background inside the picture."}
    corner = image[np.ix_(rows, columns)].astype(float)
    # Fit the background gradient from strips immediately outside both edges.
    # Those pixels include the local soft shadow instead of assuming bare scenery.
    u, v = np.meshgrid(positions, -outside)
    a, b = np.meshgrid(-outside, positions)
    design = np.column_stack(
        (np.r_[u.ravel(), a.ravel()], np.r_[v.ravel(), b.ravel()], np.ones(u.size + a.size))
    )
    colors = np.concatenate(
        (
            image[np.ix_(outside_y, columns)].reshape(-1, 3),
            image[np.ix_(rows, outside_x)].reshape(-1, 3),
        )
    )
    coefficients = np.linalg.lstsq(design, colors, rcond=None)[0]
    u, v = np.meshgrid(positions, positions)
    background = np.stack((u, v, np.ones_like(u)), axis=-1) @ coefficients
    # Refine the measured paper colour locally to absorb codec rounding without
    # allowing dark ink or a translucent fade to become the fill reference.
    paper = corner[np.linalg.norm(corner - fill, axis=-1) <= 12]
    if len(paper) < square * square / 2:
        return {"reason": "Too little settled paper fill at the card corner."}
    local_fill = np.median(paper, axis=0)
    direction = local_fill - background
    squared = np.sum(direction**2, axis=-1)
    if float(squared.min()) < 24**2:
        return {"reason": "Card paper and local background have too little colour contrast."}
    coverage = np.clip(np.sum((corner - background) * direction, axis=-1) / squared, 0, 1)
    # A two-percent endpoint band removes codec noise over the much larger flat
    # part of the square; intermediate antialiasing coverage remains fractional.
    coverage[coverage >= 0.98] = 1
    coverage[coverage <= 0.02] = 0
    # Card ink can enter the far edge of a generous corner square. Only the
    # missing fill connected to the geometric corner belongs to its curvature.
    void = coverage < 1
    connected = np.zeros_like(void)
    connected[0, 0] = void[0, 0]
    while True:
        expanded = connected.copy()
        expanded[1:] |= connected[:-1]
        expanded[:-1] |= connected[1:]
        expanded[:, 1:] |= connected[:, :-1]
        expanded[:, :-1] |= connected[:, 1:]
        expanded &= void
        if np.array_equal(expanded, connected):
            break
        connected = expanded
    missing = float(np.sum((1 - coverage)[connected]))
    radius = math.sqrt(missing / (1 - math.pi / 4))
    if radius > square - 4:
        return {"reason": "Card radius exceeds the measurable corner square.", "radius_px": radius}
    return {
        "radius_px": radius,
        "missing_area_px": missing,
        "square_px": square,
        "excluded_ink_area_px": float(np.sum((1 - coverage)[~connected])),
        "paper_rgb": local_fill.tolist(),
        "background_rgb": background[0, 0].tolist(),
    }


def corner_radius(
    pictures: Iterable[tuple[int, np.ndarray]],
    cards: Sequence[Mapping[str, Any]],
    paper_color: str,
) -> Observation:
    """Estimate card radii from missing fill area and require agreement at every corner."""
    detail: dict[str, Any] = {
        "corners": [],
        "frames": [],
        "spread_tolerance_px": CORNER_SPREAD_TOLERANCE,
        "endpoint_snap": 0.02,
    }
    if not cards:
        detail["reason"] = "No card was found."
        return None, [], detail
    fill = rgb(paper_color)
    seen = set()
    attempted = set()
    for index, image in pictures:
        for card_index, card in enumerate(cards):
            # Inclusive windows leave room for both entry and exit fades.
            if not card["start_frame"] + 3 <= index <= card["end_frame"] - 3:
                continue
            attempted.add(index)
            scale = image.shape[1] / 1080
            x, y, width, height = (
                round(card["box"][key] * scale) for key in ("x", "y", "width", "height")
            )
            square = min(round(CORNER_SQUARE_PX * scale), width // 2, height // 2)
            if square < 8:
                detail["reason"] = "Card is too small to measure its corners."
                continue
            seen.add(card_index)
            for name, cx, cy, dx, dy in (
                ("top_left", x, y, 1, 1),
                ("top_right", x + width - 1, y, -1, 1),
                ("bottom_left", x, y + height - 1, 1, -1),
                ("bottom_right", x + width - 1, y + height - 1, -1, -1),
            ):
                result = _corner(image, cx, cy, dx, dy, square, fill)
                if "radius_px" in result:
                    result["radius_px"] /= scale
                detail["corners"].append(
                    {"frame": index, "card": card_index, "corner": name, **result}
                )
    frames = sorted(attempted)
    detail["frames"] = frames
    reasons = [corner["reason"] for corner in detail["corners"] if "reason" in corner]
    if reasons:
        detail["reason"] = reasons[0]
        return None, frames, detail
    if len(seen) < len(cards):
        detail.setdefault("reason", "No settled interior frames were available for every card.")
        return None, frames, detail
    values = [corner["radius_px"] for corner in detail["corners"]]
    spread = max(values) - min(values)
    detail["spread_px"] = spread
    detail["spread_margin_px"] = CORNER_SPREAD_TOLERANCE - spread
    if spread > CORNER_SPREAD_TOLERANCE:
        detail["reason"] = "Card corners disagree by more than 6 px."
        return None, frames, detail
    return round(float(np.median(values))), frames, detail


def _caption_masks(
    pictures: Iterable[tuple[int, np.ndarray]], band: Mapping[str, int], color: str
) -> Iterable[tuple[int, np.ndarray]]:
    """Select complete caption-colour glyphs inside the measured canvas band."""
    for index, image in pictures:
        scale = image.shape[1] / 1080
        top = max(0, round((band["top_px"] - 12) * scale))
        bottom = min(image.shape[0], round((band["bottom_px"] + 12) * scale))
        crop = image[top:bottom]
        if crop.size:
            yield index, interior_components(color_mask(crop, rgb(color), 24))


def measure(
    path: Path,
    info: ffmpeg.MediaInfo,
    fields: Mapping[str, Observation],
    measures: Mapping[str, Observation],
) -> tuple[dict[str, Observation], list[dict[str, Any]]]:
    """Decode bounded full-resolution samples after visual discovery, retaining refusals."""
    observations = {}
    missing = []
    band, color = measures.get("captions.band"), fields.get("captions.text_color")
    if band is None or color is None or not color[1]:
        value, frames, detail = (
            None,
            [],
            {"reason": "No measured caption band and text colour to measure font class."},
        )
    else:
        evidence = sorted(set(color[1]))
        indices = [
            evidence[i]
            for i in np.linspace(
                0, len(evidence) - 1, min(FONT_FRAME_LIMIT, len(evidence)), dtype=int
            )
        ]
        size = fields.get("captions.size_px", (None, [], {}))[0]
        value, frames, detail = font_class(
            _caption_masks(iter_rgb(path, info, indices, width=1080), band[0], color[0]),
            size_px=size,
        )
    if value is None:
        missing.append(
            {
                "field": "captions.font_class",
                "reason": detail["reason"],
                "frames": frames,
                "detail": detail,
            }
        )
    else:
        observations["captions.font_class"] = value, frames, detail
    cards = measures.get("graphics.cards", ([], [], {}))[0]
    paper = fields.get("graphics.paper_color")
    if not cards:
        value, frames, detail = None, [], {"reason": "No card was found."}
    elif paper is None:
        value, frames, detail = None, [], {"reason": "No measured card paper colour."}
    else:
        fade_margin = max(3, round(float(info.fps or 30) * 0.25))
        indices = sorted(
            {
                int(index)
                for card in cards
                if card["end_frame"] - card["start_frame"] >= 2 * fade_margin
                for index in np.linspace(
                    card["start_frame"] + fade_margin,
                    card["end_frame"] - fade_margin,
                    3,
                )
            }
        )
        value, frames, detail = corner_radius(
            iter_rgb(path, info, indices, width=1080), cards, paper[0]
        )
    if value is None:
        missing.append(
            {
                "field": "graphics.corner_radius_px",
                "reason": detail["reason"],
                "frames": frames,
                "detail": detail,
            }
        )
    else:
        observations["graphics.corner_radius_px"] = value, frames, detail
    return observations, missing
