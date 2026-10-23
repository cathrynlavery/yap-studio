"""Measure glyph cores, caption bands, highlight geometry and repeated line pitch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yap_studio.taste.frames import color_mask, core, hex_color, interior_components, palette, runs

# Encoding can leave a sparse colour fringe up to 4 px beside an edge, even across
# an empty line. The block's shaded outermost lines can fall outside the colour mask.
BLOCK_FRINGE_PX = 4
# The template extends the block 0.08 em past its word and makes it about 0.96 em tall.
# Its padding is about 8% of its height, keeping the allowance away from stems and serifs.
BLOCK_PADDING_SHARE = 0.083
# A block encloses its active glyphs while coloured glyphs leave their gaps open.
BLOCK_ENCLOSED_SHARE = 0.4


@dataclass(frozen=True)
class TextSample:
    """One caption observation in decoded pixels, retaining first and last glyph-row bounds."""

    color: str
    highlight: str
    highlight_color: str | None
    highlight_text_color: str | None
    top: int
    bottom: int
    pitch: float | None
    line_top: float
    glyph_height: int
    pitch_detail: dict[str, Any] = field(default_factory=dict)
    last_line_top: float | None = None
    last_glyph_height: int | None = None


def line_pitch(
    mask: np.ndarray, *, minimum: int = 12, detail: dict[str, Any] | None = None
) -> float | None:
    """Use repeated glyph rows, with matching row profiles, to find line pitch."""
    profile = mask.sum(axis=1).astype(float)
    threshold = max(2, profile.max(initial=0) * 0.06)
    bands = runs(profile > threshold, gap=2)
    heights = [b - a for a, b in bands]
    bands = [(a, b) for a, b in bands if b - a >= minimum]
    detail = detail if detail is not None else {}
    detail.update(
        band_count=len(bands),
        minimum_height=minimum,
        row_threshold=round(threshold, 3),
        glyph_heights=heights[:8],
    )
    if len(bands) < 2:
        detail["reason"] = "band_count"
        return None
    if len(bands) > 2:
        # Three-line bold titles still supply repeated baseline spacing.
        pairs = []
        pair_details = []
        for first, second in zip(bands, bands[1:], strict=False):
            pair = mask[first[0]:second[1]].copy()
            pair[first[1] - first[0]:second[0] - first[0]] = False
            pair_detail: dict[str, Any] = {}
            pitch = line_pitch(pair, minimum=minimum, detail=pair_detail)
            if pitch is None:
                detail["reason"] = "inconsistent_pitch"
                return None
            pairs.append(pitch)
            pair_details.append(pair_detail)
        if max(pairs) - min(pairs) > 4:
            detail["reason"] = "inconsistent_pitch"
            return None
        detail.update(reason="measured", adjacent_pitches=pairs)
        for key in ("height_margin", "height_ratio_margin", "pitch_search_margin"):
            detail[key] = min(pair[key] for pair in pair_details)
        return float(np.median(pairs))
    first, second = bands
    ratio = (second[1] - second[0]) / (first[1] - first[0])
    detail["height_ratio"] = round(ratio, 3)
    detail["height_margin"] = min(first[1] - first[0], second[1] - second[0]) - minimum
    detail["height_ratio_margin"] = round(min(ratio - 0.5, 2 - ratio), 3)
    if not 0.5 <= ratio <= 2:
        detail["reason"] = "height_ratio"
        return None
    # Baseline-aligned lower glyph edges survive capitals and italic stress.
    separation = (second[0] + second[1] - first[0] - first[1]) / 2
    low, high = max(minimum, round(separation * 0.75)), round(separation * 1.25)
    scores = [
        (float(np.dot(profile[:-lag], profile[lag:])), lag)
        for lag in range(low, min(high + 1, len(profile)))
    ]
    detail["pitch_search"] = [low, min(high, len(profile) - 1)]
    detail["reason"] = "measured" if scores else "pitch_search"
    if scores:
        pitch = max(scores)[1]
        # Equally tall capital rows share a baseline, even when M and W have
        # different mass profiles. Correlating those profiles can shift the
        # peak several pixels away from their actual line spacing.
        baseline_pitch = second[1] - first[1]
        if 0.95 <= ratio <= 1.05 and low <= baseline_pitch <= high:
            pitch = baseline_pitch
        detail["pitch_search_margin"] = min(pitch - low, high - pitch)
        return float(pitch)
    return None


def _edge_fill(lines: np.ndarray, fringe: int) -> float:
    """Find the fullest adjacent pair of lines within the edge's fringe."""
    if len(lines) <= 2:
        return float(lines.mean())
    return max(
        float(lines[start:start + 2].mean())
        for start in range(min(fringe + 1, len(lines) - 1))
    )


def _fringe_allowance(rectangle: np.ndarray) -> int:
    """Allow a fringe within padding when the accent encloses its holes."""
    fringe = min(BLOCK_FRINGE_PX, round(len(rectangle) * BLOCK_PADDING_SHARE) - 2)
    if fringe <= 0:
        return 0
    holes = ~rectangle
    hole_count = holes.sum()
    if not hole_count or interior_components(holes).sum() / hole_count < BLOCK_ENCLOSED_SHARE:
        return 0
    return fringe


def classify_highlight(image: np.ndarray, text: np.ndarray, accent: np.ndarray | None) -> str:
    """Distinguish a bar, glyph colour, background block, or no second colour."""
    if accent is None or accent.sum() < 8:
        return "none"
    ys, xs = np.nonzero(accent)
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    occupancy = accent.sum() / (width * height)
    if height <= max(10, image.shape[0] * 0.04) and width >= height * 3:
        return "underline"
    # A solid horizontal interior, with glyph holes, distinguishes a block.
    row_fill = accent.sum(axis=1).max(initial=0) / max(width, 1)
    rectangle = accent[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def block(fringe: int) -> bool:
        edges = (
            _edge_fill(rectangle, fringe), _edge_fill(rectangle[::-1], fringe),
            _edge_fill(rectangle.T, fringe), _edge_fill(rectangle.T[::-1], fringe),
        )
        border_fill = min(edges)
        # A fringe only hides a block's real edges, so restoring them fills all four sides.
        # Letters such as B and E keep a curved or open side.
        if fringe > 0:
            return bool(row_fill > 0.85 and border_fill > 0.65)
        horizontal_edges = edges[:2]
        return bool(row_fill > 0.85 and (occupancy > 0.55 or border_fill > 0.65
                                       or (occupancy > 0.25 and (
                                           min(horizontal_edges) > 0.8
                                           or (max(horizontal_edges) > 0.8 and width >= height * 3)
                                       ))))

    # Read the outer pair first, then allow a fringe only within enclosed block padding.
    if block(0) or ((fringe := _fringe_allowance(rectangle)) > 0 and block(fringe)):
        return "moving_block"
    return "color"


def stroke_mask(image: np.ndarray) -> np.ndarray:
    """Find thin flat strokes with contrasting pixels on both horizontal sides.

    Wide walls and shirts have few such pixels. Flatness on either side keeps
    narrow serif strokes at discovery resolution, while colour recurrence
    across frames distinguishes overlay ink from moving picture texture.
    """
    values = image.astype(np.int16)
    flat = (np.max(abs(values - np.roll(values, 1, axis=1)), axis=2) < 10) | (
        np.max(abs(values - np.roll(values, -1, axis=1)), axis=2) < 10
    )
    contrast = np.zeros(image.shape[:2], dtype=bool)
    for distance in (max(1, round(image.shape[1] * d / 360)) for d in (2, 4, 6, 10)):
        left = np.max(abs(values - np.roll(values, distance, axis=1)), axis=2) > 40
        right = np.max(abs(values - np.roll(values, -distance, axis=1)), axis=2) > 40
        contrast |= left & right
    contrast[:, : max(1, round(image.shape[1] / 60))] = False
    contrast[:, -max(1, round(image.shape[1] / 60)) :] = False
    return flat & contrast


def glyph_mask(
    image: np.ndarray, color: np.ndarray, radius: float = 14, strokes: np.ndarray | None = None
) -> np.ndarray:
    """Keep colour pixels near thin strokes, including their flat interiors."""
    mask = color_mask(image, color, radius)
    strokes = mask & (stroke_mask(image) if strokes is None else strokes)
    near = strokes.copy()
    for distance in range(1, max(2, round(image.shape[1] * 4 / 1080)) + 1):
        for axis in (0, 1):
            near |= np.roll(strokes, distance, axis=axis)
            near |= np.roll(strokes, -distance, axis=axis)
    return mask & near


class CaptionDiscovery:
    """Accumulate colour recurrence by row, retaining no decoded pictures."""

    def __init__(self, height: int, width: int) -> None:
        self.width = width
        self.offset = round(height * 0.46)
        self.histogram = np.zeros((round(height * 0.94) - self.offset, 22**3), np.int32)
        self.count = 0

    def add(self, image: np.ndarray) -> None:
        """Record only thin contrasting strokes in a sparse discovery frame."""
        crop = image[self.offset : self.offset + len(self.histogram)]
        mask = stroke_mask(crop)
        q = crop.astype(np.int32) // 12
        keys = q[:, :, 0] * 22**2 + q[:, :, 1] * 22 + q[:, :, 2]
        y, x = np.nonzero(mask)
        np.add.at(self.histogram, (y, keys[y, x]), 1)
        self.count += 1

    def band(self) -> tuple[float, float] | None:
        """Return recurrent stroke rows in canvas coordinates, with discovery room."""
        if not self.count:
            return None
        scores = self.histogram.sum(axis=0)
        best = None
        for key in np.argsort(scores)[-30:][::-1]:
            rows = self.histogram[:, key]
            bands = runs(rows > max(self.count * 0.15, rows.max(initial=0) * 0.08), gap=4)
            for top, bottom in bands:
                if not 12 <= (bottom - top) * 1080 / self.width <= 260:
                    continue
                mass = int(rows[top:bottom].sum())
                if mass < self.count * 6 or mass < scores[key] * 0.35:
                    continue
                if best is None or mass > best[0]:
                    best = mass, top, bottom
        if best is None:
            return None
        # A rare second line can grow above or below the recurrent line.
        # Keep equal room in both directions before measuring its pitch.
        _, top, bottom = best
        scale = 1080 / self.width
        return (self.offset + top) * scale - 260, (self.offset + bottom) * scale + 260


def caption_band(
    image: np.ndarray,
    band: tuple[float, float] | None = None,
    text_hint: str | None = None,
    accent_hint: str | None = None,
    highlight_hint: str | None = None,
) -> TextSample | None:
    """Measure caption ink using thin strokes inside recurrent discovery rows."""
    height, width = image.shape[:2]
    scale = width / 1080
    offset = max(0, round(band[0] * scale)) if band else round(height * 0.46)
    end = min(height, round(band[1] * scale)) if band else round(height * 0.94)
    region = image[offset:end]
    if not region.size:
        return None
    strokes = stroke_mask(region)
    colors = (
        [(np.array([int(text_hint[i : i + 2], 16) for i in (1, 3, 5)]), 1)]
        if text_hint
        else (
            palette(region, strokes & (region.min(axis=2) > 180), count=8)
            + palette(region, strokes, count=8)
        )
    )
    if not colors and not text_hint:
        colors = palette(region, strokes, count=8)
    choices = []
    for color, _ in colors:
        full_mask = color_mask(region, color, 14)
        mask = full_mask & strokes
        # A bright wall can make thin counters around dark letters. Its colour
        # occupies a broad area; foreground ink occupies compact strokes.
        if not text_hint and mask.sum() < full_mask.sum() * 0.08:
            continue
        transitions = np.count_nonzero(np.diff(mask.astype(np.int8), axis=1), axis=1)
        rows = runs((mask.sum(axis=1) >= max(5, width * 0.01)) & (transitions >= 6), gap=2)
        for top, bottom in rows:
            if 10 * scale <= bottom - top <= 120 * scale:
                mass = int((mask & strokes)[top:bottom].sum())
                choices.append((mass * (10 if color.min() > 180 else 1), color, top, bottom))
    if not choices:
        return None
    _, text_color, a, b = max(choices, key=lambda item: item[0])
    siblings = [
        (lo, hi)
        for _, c, lo, hi in choices
        if np.linalg.norm(c - text_color) < 20 and abs(lo - a) < 260 * scale
    ]
    top = max(0, min(lo for lo, _ in siblings) - round(25 * scale))
    bottom = min(len(region), max(hi for _, hi in siblings) + round(140 * scale))
    crop = region[top:bottom]
    crop_strokes = strokes[top:bottom]
    text = glyph_mask(crop, text_color, 18, crop_strokes)
    transitions = np.count_nonzero(np.diff(text.astype(np.int8), axis=1), axis=1)
    dense_rows = transitions >= 6
    text &= dense_rows[:, None]
    ink = core(text)
    if ink.any():
        text_color = np.median(crop[ink], axis=0)
        text = glyph_mask(crop, text_color, 18, crop_strokes) & dense_rows[:, None]
    # Accent candidates must form ink or a compact flat bar beside the glyphs.
    spread = crop.max(axis=2).astype(float) - crop.min(axis=2)
    accent_core = core(spread > 40)
    accent_colors = palette(crop, accent_core if accent_core.any() else spread > 40, count=20)
    if accent_hint:
        accent_colors = [(np.array([int(accent_hint[i : i + 2], 16) for i in (1, 3, 5)]), 1)]
    accent_color = accent = None
    candidates = []
    for color, _ in accent_colors:
        if np.linalg.norm(color - text_color) < 35:
            continue
        mask = color_mask(crop, color, 7)
        yy, xx = np.nonzero(mask)
        if len(yy) < max(8, text.sum() * 0.01):
            continue
        rows = runs(mask.sum(axis=1) > max(2, width * 0.004), gap=2)
        if not rows:
            continue
        lo, hi = max(rows, key=lambda ab: int(mask[ab[0] : ab[1]].sum()))
        local = mask.copy()
        local[:lo] = local[hi:] = False
        yy, xx = np.nonzero(local)
        w, h = int(np.ptp(xx)) + 1, hi - lo
        occupancy = local.sum() / max(1, w * h)
        transition_count = np.count_nonzero(np.diff(local.astype(np.int8), axis=1), axis=1)
        glyph_like = (
            np.count_nonzero(transition_count >= 4) >= max(3, round(10 * scale))
            and (local & crop_strokes).sum() > local.sum() * 0.15
        )
        bar = h <= max(3, round(10 * scale)) and w >= h * 3 and occupancy > 0.65
        look = classify_highlight(crop, text, local)
        block = look == "moving_block"
        if not (glyph_like or bar or (block and h < 120 * scale and w < width * 0.5)):
            continue
        if highlight_hint and look != highlight_hint:
            continue
        # Underlines and blocks need a flat core, unlike skin gradients.
        flat_count = int(core(local).sum())
        if flat_count < local.sum() * 0.2 and not bar:
            continue
        candidates.append(
            (int(local.sum()) if block else
             int((local & crop_strokes).sum()) + (local.sum() * 8 if bar else 0), color, local)
        )
    if candidates:
        _, accent_color, accent = max(candidates, key=lambda item: item[0])
    highlight = classify_highlight(crop, text, accent)
    active_text = None
    active_glyphs = np.zeros_like(text)
    if highlight == "moving_block" and accent is not None:
        yy, xx = np.nonzero(accent)
        interior = crop[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
        inks = []
        for color, _ in palette(interior, count=8):
            if np.linalg.norm(color - accent_color) < 55:
                continue
            ink = core(interior_components(color_mask(interior, color, 14)))
            if ink.sum() > 10:
                inks.append((int(ink.sum()), color, ink))
        if inks:
            _, _, ink = max(inks, key=lambda item: item[0])
            active_text = hex_color(np.median(interior[ink], axis=0))
            active_glyphs[yy.min():yy.max() + 1, xx.min():xx.max() + 1] = ink
    glyphs = text | active_glyphs | (
        glyph_mask(crop, accent_color, 10, crop_strokes)
        if highlight == "color" and accent is not None
        else False
    )
    pitch_detail: dict[str, Any] = {}
    pitch = line_pitch(glyphs, minimum=max(3, round(width / 1080 * 12)), detail=pitch_detail)
    yy = np.flatnonzero(glyphs.sum(axis=1) > max(2, width * 0.005))
    if not len(yy):
        return None
    bands = runs(glyphs.sum(axis=1) > max(2, width * 0.005), gap=2)
    return TextSample(
        hex_color(text_color),
        highlight,
        hex_color(accent_color) if accent_color is not None else None,
        active_text,
        offset + top + int(yy[0]),
        offset + top + int(yy[-1]) + 1,
        pitch,
        float(offset + top + yy[0]),
        bands[0][1] - bands[0][0],
        pitch_detail,
        float(offset + top + bands[-1][0]),
        bands[-1][1] - bands[-1][0],
    )


def panel_text(
    image: np.ndarray,
    fill: np.ndarray,
    *,
    rotated: bool = False,
    detail: dict[str, Any] | None = None,
    require_glyphs: bool = False,
) -> dict[str, Any]:
    """Measure interior ink; opening titles also require thin glyph strokes."""
    detail = detail if detail is not None else {}
    detail["reason"] = "text_colour"
    colors = palette(image, count=10)
    contrast = np.linalg.norm(image.astype(float) - fill, axis=2) > 55
    chroma = image.max(axis=2).astype(float) - image.min(axis=2) > 35
    colors += palette(image, core(contrast & chroma), count=4)
    candidates = [(color, n) for color, n in colors if np.linalg.norm(color - fill) > 55]
    if not candidates:
        return {}
    # Glyph cores have a contrasting colour and occur away from the outer edges.
    inset = max(3, round(image.shape[1] * 0.055))
    valid = np.zeros(image.shape[:2], dtype=bool)
    valid[8:-8, inset:-inset] = True
    masks = [(c, core(interior_components(color_mask(image, c, 14))) & valid)
             for c, _ in candidates]
    masks = [(c, m) for c, m in masks if m.sum() > 10]
    if not masks:
        return {}
    strokes = stroke_mask(image)
    inks = []
    for color, mask in masks:
        supported = glyph_mask(image, color, 14, strokes) & mask
        # A wall, silhouette or scenery edge can share an ink colour. A text
        # core must have thin strokes with contrast on both horizontal sides.
        fraction = float(supported.sum() / mask.sum())
        if require_glyphs and fraction < 0.15:
            continue
        inks.append((color, mask, fraction))
    if not inks:
        return {}
    # Navy and plum ink are still text. Chroma alone cannot distinguish them
    # from an accent; use interior glyph support and area instead.
    detail.update(ink_colours=len(inks), reason="measured")
    color, glyphs, fraction = max(inks, key=lambda item: int(item[1].sum()))
    result: dict[str, Any] = {"text_color": hex_color(np.median(image[glyphs], axis=0))}
    result["colour_margins"] = {
        "contrast": round(float(np.linalg.norm(color - fill)) - 55, 3),
        "neutral": round(32 - float(np.ptp(color)), 3),
        "core_pixels": int(glyphs.sum()) - 10,
        "stroke_support": round(fraction - 0.15, 3),
    }
    accents = [(c, core(color_mask(image, c, 14)) & valid) for c, _ in candidates
               if np.ptp(c) >= 32 and np.linalg.norm(c - color) > 35]
    # A quiet title's underline may have only two settled RGB rows after
    # chroma subsampling. Full erosion erases both; a long, flat bar supplies
    # color evidence after horizontal erosion alone.
    for candidate, _ in candidates:
        if np.ptp(candidate) < 32 or np.linalg.norm(candidate - color) <= 35:
            continue
        mask = color_mask(image, candidate, 14) & valid
        yy, xx = np.nonzero(mask)
        if not len(yy):
            continue
        w, h = int(np.ptp(xx)) + 1, int(np.ptp(yy)) + 1
        if h <= max(3, round(image.shape[1] * 8 / 1080)) and w >= h * 3:
            if mask.sum() / (w * h) > 0.7:
                horizontal = mask & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1)
                accents.append((candidate, horizontal))
    accents = [(c, m) for c, m in accents if m.sum() > 10]
    if accents:
        accent, mask = max(accents, key=lambda item: int(item[1].sum()))
        result["accent_color"] = hex_color(np.median(image[mask], axis=0))
        if rotated:
            glyphs |= mask
    if rotated:
        # Undo the template's 2.5 degree band rotation for row projections.
        y, x = np.indices(glyphs.shape)
        yy = np.rint(y + np.tan(np.radians(2.5)) * (x - glyphs.shape[1] / 2)).astype(int)
        flat = np.zeros_like(glyphs)
        keep = (yy >= 0) & (yy < glyphs.shape[0]) & glyphs
        flat[yy[keep], x[keep]] = True
        glyphs = flat
        glyphs[glyphs.sum(axis=1) > image.shape[1] * 0.75] = False
    pitch_detail: dict[str, Any] = {}
    pitch = line_pitch(
        glyphs, minimum=max(3, round(image.shape[1] / 1080 * 20)), detail=pitch_detail
    )
    detail["pitch"] = pitch_detail
    result["pitch_detail"] = pitch_detail
    if pitch is not None:
        result["pitch"] = pitch
    return result
