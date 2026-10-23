"""Pure vertical placement hints for elements over a window of face boxes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yap_studio.faces.clearance import DEFAULT_HAIR, HairAllowance, _margin, gap_px, head_box
from yap_studio.faces.model import Box


@dataclass(frozen=True)
class Band:
    """Vertical limits containing the entire element, in display pixels."""

    top: int
    bottom: int

    def __post_init__(self) -> None:
        if (
            type(self.top) is not int
            or type(self.bottom) is not int
            or not 0 <= self.top < self.bottom
        ):
            raise ValueError("band limits must be integers with 0 <= top < bottom")


HEADROOM_BAND = Band(280, 640)
TOP_BAND = Band(0, 640)


def top_band(style: Mapping[str, Any]) -> Band:
    """Return the top band where cards sit by default, the owner's decision.

    A card layer carries its own top padding, so a layer at y 0 keeps the visible
    card off the frame edge. Callers pass preferred_y=band.top to top-align.
    """
    return Band(0, style["internal"]["key_text_bottom_px"])


def headroom_band(style: Mapping[str, Any]) -> Band:
    """Return the headroom limits from resolved style data."""
    internal = style["internal"]
    return Band(internal["key_text_top_px"], internal["key_text_bottom_px"])


def caption_band(style: Mapping[str, Any]) -> Band:
    """Allow captions below the headroom band and above the caption safe bottom.

    The limits are key_text_bottom_px and caption_safe_bottom_px in the resolved
    style's internal data, giving 640 to 1740 for the shipped styles.
    """
    internal = style["internal"]
    return Band(internal["key_text_bottom_px"], internal["caption_safe_bottom_px"])


@dataclass(frozen=True)
class PlacementHint:
    """A clear top edge, or the best available gap when the band is blocked."""

    y: int | None
    offset_px: int | None
    preferred_y: int
    band: Band
    margin_px: float
    hair: HairAllowance
    reason: str
    best_y: int | None
    best_gap_px: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible placement fields and band limits."""
        return {
            "y": self.y,
            "offset_px": self.offset_px,
            "preferred_y": self.preferred_y,
            "band": {"top": self.band.top, "bottom": self.band.bottom},
            "margin_px": self.margin_px,
            "hair": self.hair.to_dict(),
            "reason": self.reason,
            "best_y": self.best_y,
            "best_gap_px": self.best_gap_px,
        }


def _forbidden_intervals(
    faces: set[Box], element: Box, margin: float, low: int, high: int
) -> list[tuple[int, int]]:
    """Merge forbidden integer ranges, retaining clear open interval ends."""
    intervals = []
    for face in faces:
        horizontal_gap = max(face.x - element.x1, element.x - face.x1)
        if horizontal_gap >= margin:
            continue
        # Strict inequalities keep a gap exactly equal to the margin available.
        start = math.floor(max(low - 1, min(high, face.y - margin - element.h))) + 1
        end = math.ceil(max(low, min(high + 1, face.y1 + margin))) - 1

        # Rearranging decimal coordinates can round differently from gap_px.
        # Refine a boundary by bisection only when its original gap arithmetic
        # disagrees, keeping the result exact without scanning the band.
        if (start <= high and face.y - (start + element.h) >= margin) or (
            start > low and face.y - (start - 1 + element.h) < margin
        ):
            left, right = low, high + 1
            while left < right:
                middle = (left + right) // 2
                if face.y - (middle + element.h) >= margin:
                    left = middle + 1
                else:
                    right = middle
            start = left
        if (end >= low and end - face.y1 >= margin) or (end < high and end + 1 - face.y1 < margin):
            left, right = low, high + 1
            while left < right:
                middle = (left + right) // 2
                if middle - face.y1 < margin:
                    left = middle + 1
                else:
                    right = middle
            end = left - 1
        if start <= end:
            intervals.append((start, end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _closest_allowed_y(
    forbidden: Sequence[tuple[int, int]], low: int, high: int, preferred: int
) -> int | None:
    """Pick the closest integer in the complement, breaking ties upwards."""
    candidates = []
    start = low
    for first, last in forbidden:
        if start < first:
            candidates.append(max(start, min(preferred, first - 1)))
        start = last + 1
    if start <= high:
        candidates.append(max(start, min(preferred, high)))
    return min(candidates, key=lambda y: (abs(y - preferred), y), default=None)


def suggest_offset(
    faces: Sequence[Sequence[Box]],
    *,
    x: float,
    width: float,
    height: float,
    band: Band,
    margin_px: float,
    preferred_y: int | None = None,
    hair: HairAllowance = DEFAULT_HAIR,
) -> PlacementHint:
    """Suggest the closest clear integer top edge across every supplied frame.

    Preferences default to the centred position and are clamped to the fitting
    range. Identical faces are collapsed and expanded to head boxes before their intervals
    are merged. Only a fully blocked band needs a scan for the best remaining gap.
    """
    element = Box(x, band.top, width, height)
    margin = _margin(margin_px, "margin_px")
    if preferred_y is not None and type(preferred_y) is not int:
        raise ValueError("preferred_y must be an integer")
    preferred = (
        band.top + int((band.bottom - band.top - height) // 2)
        if preferred_y is None
        else preferred_y
    )
    if height > band.bottom - band.top:
        return PlacementHint(
            None,
            None,
            preferred,
            band,
            margin,
            hair,
            f"a {height:.0f} px element does not fit the band {band.top} to {band.bottom}",
            None,
            None,
        )

    low, high = band.top, math.floor(band.bottom - height)
    preferred = max(low, min(preferred, high))
    unique_heads = {head_box(face, hair) for face in {face for frame in faces for face in frame}}
    preferred_box = Box(x, preferred, width, height)
    if not unique_heads:
        reason = "no face in the window; keep the preferred position"
    elif all(gap_px(preferred_box, head) >= margin for head in unique_heads):
        reason = "clear of the head at the preferred position"
    else:
        reason = None
    if reason is not None:
        return PlacementHint(preferred, 0, preferred, band, margin, hair, reason, None, None)

    forbidden = _forbidden_intervals(unique_heads, element, margin, low, high)
    y = _closest_allowed_y(forbidden, low, high, preferred)
    if y is not None:
        offset = y - preferred
        direction = "up" if offset < 0 else "down"
        return PlacementHint(
            y,
            offset,
            preferred,
            band,
            margin,
            hair,
            f"moved {direction} {abs(offset)} px to stay {margin:.0f} px clear of the head",
            None,
            None,
        )

    best_gap, best_y = max(
        (
            (min(gap_px(Box(x, y, width, height), head) for head in unique_heads), y)
            for y in range(low, high + 1)
        ),
        key=lambda item: (item[0], -abs(item[1] - preferred), -item[1]),
    )
    gap_description = (
        f"an overlap of {-best_gap:.0f} px" if best_gap < 0 else f"a gap of {best_gap:.0f} px"
    )
    reason = (
        f"no position in the band {band.top} to {band.bottom} keeps a {height:.0f} px "
        f"element {margin:.0f} px clear of the head; "
        f"the best is y {best_y} with {gap_description}"
    )
    return PlacementHint(None, None, preferred, band, margin, hair, reason, best_y, best_gap)
