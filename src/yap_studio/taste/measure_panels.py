"""Find flat title and card panels without text recognition or scene metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yap_studio.taste.frames import color_mask, core, interior_components, palette, runs


@dataclass(frozen=True)
class Panel:
    """Visible fill bounds and core colour in the decoded picture's pixels."""

    x: int
    y: int
    width: int
    height: int
    fill: tuple[float, float, float]
    rotated: bool = False
    margins: dict[str, float] = field(default_factory=dict, compare=False)

    def box(self, scale: float = 1) -> dict[str, int]:
        """Return a box scaled to the engine canvas."""
        return {
            key: round(value * scale)
            for key, value in (
                ("x", self.x),
                ("y", self.y),
                ("width", self.width),
                ("height", self.height),
            )
        }


def find_panels(
    image: np.ndarray, *, detail: dict[str, Any] | None = None, discovery: bool = False
) -> list[Panel]:
    """Find bounded fills, deferring lower-edge checks during coarse discovery."""
    height, width = image.shape[:2]
    region = image[: round(height * 0.62)]
    result = []
    detail = detail if detail is not None else {}
    rejections = detail.setdefault("rejections", {})

    def reject(check: str) -> None:
        rejections[check] = rejections.get(check, 0) + 1

    edge = max(1, round(width * 0.08))
    colors = palette(region, count=12)
    # The central subject can dominate a nearby dark fill's palette bin.
    # Sample the outer edges too, where full-width title bands remain visible.
    colors += palette(region[:, :edge], count=4)
    colors += palette(region[:, -edge:], count=4)
    seen_colors: list[np.ndarray] = []
    for color, _ in colors:
        if any(np.linalg.norm(color - seen) < 3 for seen in seen_colors):
            continue
        seen_colors.append(color)
        mask = color_mask(region, color, 30 if np.mean(color) > 120 else 5)
        # Heavy title glyphs can occupy most of a row inside a real panel.
        # Keep that row connected; coverage and parallel edges still validate
        # the complete candidate below.
        rows = runs(mask.sum(axis=1) > width * 0.15, gap=max(1, round(width / 360)))
        if np.mean(color) < 60:
            # A dark subject can share a bold band's fill in the centre.
            # Full-width bands also have fill at the picture's outer edges.
            outer = (mask[:, :edge].mean(axis=1) > 0.7) | (
                mask[:, -edge:].mean(axis=1) > 0.7
            )
            edge_rows = [(a, b) for a, b in runs(outer, gap=max(1, round(width / 360)))
                         if 65 * width / 1080 <= b - a <= 650 * width / 1080]
            if edge_rows:
                rows = edge_rows
        if not rows:
            reject("fill")
        for top, bottom in rows:
            band_height = bottom - top
            detail["candidates_examined"] = detail.get("candidates_examined", 0) + 1
            if not 65 * width / 1080 <= bottom - top <= 650 * width / 1080:
                reject("fill")
                continue
            ys, xs = np.nonzero(mask[top:bottom])
            if not len(xs):
                reject("fill")
                continue
            columns = np.flatnonzero(mask[top:bottom].mean(axis=0) > 0.6)
            if not len(columns):
                reject("fill")
                continue
            left, right = int(columns.min()), int(columns.max()) + 1
            band_width = right - left
            if right - left < width * 0.45:
                reject("fill")
                continue
            crop_mask = mask[top:bottom, left:right]
            # Text is a hole in the fill, not evidence against a panel. Measure
            # fill coverage outside enclosed ink so rasterisation and
            # glyph weight do not decide whether a dense title exists.
            ink = interior_components(~crop_mask)
            coverage = float(crop_mask.sum() / max(1, np.count_nonzero(~ink)))
            minimum_coverage = 0.65 if width < 540 else 0.7
            if bottom >= len(region) - 1 or coverage < minimum_coverage:
                reject("fill")
                continue
            # A gradient background produces stripes with little edge contrast.
            inside = region[top:bottom, left:right][core(crop_mask)]
            if not len(inside):
                reject("fill")
                continue
            fill = np.median(inside, axis=0)
            # Translucent fills occupy a compact colour distribution. A scenery
            # gradient spans a continuum even when a broad colour mask makes
            # one segment look rectangular.
            clusters = palette(inside.reshape(-1, 1, 3), count=2)
            clustered = np.zeros(len(inside), dtype=bool)
            for centre, _ in clusters:
                clustered |= color_mask(inside.reshape(-1, 1, 3), centre, 10).ravel()
            if clustered.mean() < 0.75:
                reject("fill")
                continue
            above = image[max(0, top - 8) : top, left:right]
            below = image[bottom : min(height, bottom + 8), left:right]
            edges = [
                float(np.linalg.norm(np.median(v.reshape(-1, 3), axis=0) - fill))
                for v in (above, below)
                if v.size
            ]
            if top < 2 or top > height * 0.5 or not edges or max(edges) < 24:
                reject("edge_contrast")
                continue
            # Full-width title bands have a sloping upper edge. For centred
            # rounded panels, avoid corner pixels when locating that edge.
            tops = []
            edge_columns = []
            for x in range(left + (right - left) // 5, right - (right - left) // 5):
                yy = np.flatnonzero(mask[max(0, top - 30) : bottom, x])
                if len(yy):
                    tops.append(int(yy[0]))
                    edge_columns.append(x)
            rotated = bool(tops and np.ptp(tops) > width * 0.015)
            if not rotated:
                # Compare immediately across each edge. A gradual wall or sky
                # gradient can contrast with the median fill without a boundary.
                contrasts = []
                for boundary in (top, bottom):
                    upper = image[max(0, boundary - 4) : boundary, left:right]
                    lower = image[boundary : boundary + 4, left:right]
                    if upper.size and lower.size:
                        contrasts.append(
                            float(
                                np.linalg.norm(
                                    np.median(upper.reshape(-1, 3), axis=0)
                                    - np.median(lower.reshape(-1, 3), axis=0)
                                )
                            )
                        )
                if len(contrasts) != 2 or min(contrasts) < 7:
                    reject("edge_contrast")
                    continue
            # Overlay boundaries are straight apart from rounded corners.
            # Walls, faces and furniture seldom maintain two parallel edges.
            residual = 0.0
            slope = 0.0
            parallel_margin = 0.0
            if tops:
                xx = np.arange(len(tops))
                slope, intercept = np.polyfit(xx, tops, 1)
                residual = np.median(abs(np.asarray(tops) - (slope * xx + intercept)))
                if residual > max(1.5, width * 0.003):
                    reject("slope_or_residual")
                    continue
                # Row coverage clips the low corner of a rotated rectangle.
                # Extend by its measured slope before locating the lower edge.
                edge_end = min(len(mask), bottom + math.ceil(abs(slope) * (right - left)) + 4)
                lower_columns = edge_columns
                if rotated and band_width > width * 0.95:
                    lower_columns = [x for x in edge_columns
                                     if x < width * 0.25 or x > width * 0.75]
                bottoms = [
                    int(np.flatnonzero(mask[max(0, top - 30) : edge_end, x])[-1])
                    for x in lower_columns
                ]
                lower_x = np.asarray(lower_columns) - edge_columns[0]
                lower_slope, lower_intercept = np.polyfit(lower_x, bottoms, 1)
                lower_residual = np.median(
                    abs(np.asarray(bottoms) - (lower_slope * lower_x + lower_intercept))
                )
                parallel_margin = max(3, width * 0.01) - abs(slope - lower_slope) * len(tops)
                # Scenery stripes can inherit the straight upper edge of a
                # title or card. Their lower edge must also be straight and
                # parallel before the region can establish an overlay panel.
                if not discovery and (
                    lower_residual > max(1.5, width * 0.003) or parallel_margin < 0
                ):
                    reject("slope_or_residual")
                    continue
            if rotated:
                expanded = color_mask(region, fill, 9)
                if band_width > width * 0.95:
                    expanded[:, round(width * 0.15):round(width * 0.85)] = False
                # The rotated look's coloured strip is part of its visible box.
                # It is not the dominant fill and would otherwise be clipped.
                chroma = region.max(axis=2).astype(float) - region.min(axis=2)
                expanded |= chroma > 60
                yy, xx = np.nonzero(expanded[max(0, top - 40) : min(len(region), bottom + 40)])
                # Only expand within the local band, never another dark object.
                if len(yy):
                    top0 = max(0, top - 40)
                    top, bottom = int(yy.min()) + top0, int(yy.max()) + top0 + 1
                    left, right = int(xx.min()), int(xx.max()) + 1
            panel = Panel(
                left,
                top,
                right - left,
                bottom - top,
                tuple(float(v) for v in fill),
                rotated,
                {
                    "height_min": round(band_height - 65 * width / 1080, 3),
                    "height_max": round(650 * width / 1080 - band_height, 3),
                    "width": round(band_width - width * 0.45, 3),
                    "fill_coverage": round(coverage - minimum_coverage, 3),
                    "raw_fill_coverage": round(float(crop_mask.mean()), 3),
                    "excluded_ink_fraction": round(float(ink.mean()), 3),
                    "cluster_coverage": round(float(clustered.mean()) - 0.75, 3),
                    "edge_contrast": round(max(edges) - 24, 3),
                    "rotation": round(float(np.ptp(tops)) - width * 0.015, 3) if tops else 0,
                    "residual": round(max(1.5, width * 0.003) - float(residual), 3),
                    "slope": round(float(slope), 5),
                    "parallel_edges": round(float(parallel_margin), 3),
                },
            )
            if any(
                abs(panel.y - old.y) < 15 and abs(panel.width - old.width) < 20 for old in result
            ):
                continue
            result.append(panel)
    detail["panels_found"] = len(result)
    return sorted(result, key=lambda panel: panel.y)


def panel_edge_match(
    image: np.ndarray,
    panel: Panel,
    *,
    reference_width: int,
    reference: np.ndarray,
    search_rise: bool = False,
) -> tuple[float, float]:
    """Return ink-edge amplitude and correlation, retaining faint fade evidence."""
    ratio = image.shape[1] / reference_width
    x, y, w = (round(v * ratio) for v in (panel.x, panel.y, panel.width))
    target = np.diff(reference.astype(np.float32).mean(axis=2), axis=1)
    valid = abs(target) > 18
    if valid.sum() < 10:
        return 0.0, 0.0
    expected = target[valid]
    norm = float(np.dot(expected, expected))
    rise = max(2, round(w * 30 / 1080)) if search_rise else 0
    matches = []
    for shift in range(-rise, rise + 1):
        yy = y + shift
        current = image[max(0, yy) : yy + reference.shape[0], x : x + reference.shape[1]]
        if current.shape != reference.shape:
            continue
        edges = np.diff(current.astype(np.float32).mean(axis=2), axis=1)[valid]
        product = float(np.dot(edges, expected))
        correlation = product / max(float(np.linalg.norm(edges)) * np.sqrt(norm), 1)
        matches.append((product / max(norm, 1), correlation))
    return max(
        matches,
        key=lambda match: (match[0] > 0.025 and match[1] > 0.55, match[1]),
        default=(0.0, 0.0),
    )


def fade_end(samples: list[tuple[int, float]], end: int) -> int:
    """Extrapolate a declining edge amplitude to the post-overlay baseline.

    Linear, quadratic and cubic declines cover abrupt and eased fades. Require
    several declining frames and a quiet tail; a one-frame card removal cannot
    qualify. The tail removes any scenery edges shared with the reference ink.
    """
    if len(samples) < 8:
        return end
    indices, amplitudes = np.asarray(samples, dtype=float).T
    tail = amplitudes[-max(4, len(samples) // 5) :]
    baseline = float(np.median(tail))
    noise = float(np.median(abs(tail - baseline))) * 1.4826
    signal = amplitudes - baseline
    peak = float(signal.max())
    floor = max(0.004, noise * 3)
    selected = np.flatnonzero((signal > floor) & (signal < peak * 0.6))
    if len(selected) < 3 or peak < 0.2 or np.ptp(tail) > peak * 0.1:
        return end
    # Only the final uninterrupted decline belongs to this disappearance.
    breaks = np.flatnonzero(np.diff(selected) > 1)
    if len(breaks):
        selected = selected[breaks[-1] + 1 :]
    if len(selected) < 3:
        return end
    x = indices[selected] - indices[selected[0]]
    values = signal[selected]
    if np.count_nonzero(np.diff(values) < 0) < len(values) - 2:
        return end
    fits = []
    for power in (1.0, 0.5, 1 / 3):
        y = values**power
        slope, intercept = np.polyfit(x, y, 1)
        if slope >= 0:
            continue
        error = float(np.mean((y - (slope * x + intercept)) ** 2) / max(np.var(y), 1e-9))
        zero = indices[selected[0]] - intercept / slope
        if indices[selected[-1]] <= zero <= indices[-1] and error < 0.1:
            fits.append((error, zero))
    if not fits:
        return end
    zero = round(min(fits)[1])
    return max(end, zero - 1)


def panel_present(
    image: np.ndarray,
    panel: Panel,
    *,
    reference_width: int,
    reference: np.ndarray | None = None,
) -> bool:
    """Recognise a discovered panel during boundary refinement, including fades."""
    ratio = image.shape[1] / reference_width
    x, y, w, h = (round(v * ratio) for v in (panel.x, panel.y, panel.width, panel.height))
    crop = image[y : y + h, x : x + w]
    if not crop.size:
        return False
    if reference is not None:
        # Match the panel's unchanged ink edges through its fade. Fill colour
        # and a scenery edge alone cannot establish that an overlay is present.
        amplitude, correlation = panel_edge_match(
            image, panel, reference_width=reference_width, reference=reference, search_rise=True
        )
        return correlation > 0.55 and amplitude > 0.025
    # Use the settled panel palette for exact later cards. Title fading needs
    # the geometric edge test below, as its fill is mixed with the background.
    if (
        color_mask(crop, np.asarray(panel.fill), 30 if np.mean(panel.fill) > 120 else 15).mean()
        > 0.35
    ):
        return True
    if y < 2 or w < 4 or h < 4:
        return False
    if panel.rotated:
        y += round(w * np.tan(np.radians(2.5)) / 2)
    left, right = x + w // 4, x + 3 * w // 4
    inside = np.median(image[y + 2 : y + 6, left:right].reshape(-1, 3), axis=0)
    outside = np.median(image[y - 6 : y - 2, left:right].reshape(-1, 3), axis=0)
    return bool(np.linalg.norm(inside - outside) > 7)
