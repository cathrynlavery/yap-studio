"""Pure geometry and frame timing for pages behind the speaker."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from fractions import Fraction

import numpy as np

from yap_studio.matte.refine import dilate

# Fit a tall page above the caption band.
TALL_BOX_PX = (230, 140, 620, 860)
# Shrink the person without changing the page.
SHRINK_SCALE = 0.8
# Keep the complete scaled frame inside the canvas.
SHRINK_BOX_PX = (108, 384, 864, 1536)
# Soften the person's left, right and top frame edges.
PERSON_EDGE_FADE_PX = 24
# Give narrow pages a readable background width.
SHRINK_PAGE_MIN_W = 620
# Measure every third local matte frame.
AUTO_SAMPLE_EVERY = 3
# Keep at least half the page clear of the person.
AUTO_VISIBLE_MIN = 0.5
# Leave room above the person inside the page columns.
AUTO_ABOVE_MIN_PX = 300
# Hold the top of the page before its first flick.
SCROLL_LEAD_FRAMES = 15
# Ease each flick over 1.2 seconds.
SCROLL_FLICK_FRAMES = 36
# Leave four still frames between flicks.
SCROLL_PERIOD_FRAMES = 40
# Hold after the final flick.
SCROLL_TAIL_FRAMES = 12
# Prefer steps of at least a quarter of the box height.
SCROLL_STEP_MIN = 0.25
# Limit a step to one box height.
SCROLL_STEP_MAX = 1.0
# Give the page enough time to be read.
TALL_MIN_S = 2.5
# Allow longer holds than ordinary overlays.
TALL_MAX_S = 8.0
# Snap anchors only to nearby picture cuts.
CUT_SNAP_FRAMES = 15
# Reserve the named placement for long screenshots.
LONG_PAGE_RATIO = 1.6


def rule_constants() -> dict:
    """Identify every rule that affects the rendered page."""
    return {key: value for key, value in globals().items() if key.isupper()}


def chosen_framing(payload: Mapping) -> str | None:
    """Use the recorded decision once assembly has measured automatic framing."""
    return payload.get("framing", {}).get("chosen", payload.get("tall_framing"))


def shrink_page_box(src_w: int) -> tuple[int, int, int, int]:
    """Centre the width-fit page on the full-height paper canvas."""
    width = min(1080, max(SHRINK_PAGE_MIN_W, src_w))
    return (1080 - width) // 2, 0, width, 1920


def auto_framing(frames: Iterable[np.ndarray], clearance: int) -> dict:
    """Measure the least clear sampled box and its highest person row."""
    x, y, width, height = TALL_BOX_PX
    visible_min, above_min = 1.0, 1920 - y
    samples = 0
    for index, alpha in enumerate(frames):
        if index % AUTO_SAMPLE_EVERY:
            continue
        person = dilate(alpha >= 128, math.ceil(clearance))
        visible_min = min(visible_min, 1 - float(person[y:y + height, x:x + width].mean()))
        rows = np.flatnonzero(person[:, x:x + width].any(axis=1))
        above_min = min(above_min, int(rows[0]) - y if len(rows) else 1920 - y)
        samples += 1
    if not samples:
        raise ValueError("Automatic tall framing needs sampled matte frames")
    return {"chosen": "keep" if (visible_min >= AUTO_VISIBLE_MIN
                                and above_min >= AUTO_ABOVE_MIN_PX) else "shrink",
            "visible_min": visible_min, "above_min_px": above_min}


def is_long_page(item: Mapping) -> bool:
    """Recognise catalog screenshots that suit the tall placement."""
    return (item.get("kind") == "image" and item.get("media_type") == "screenshot"
            and (item.get("width") or 0) > 0
            and (item.get("height") or 0) >= LONG_PAGE_RATIO * item["width"])


def page_height(src_w: int, src_h: int, box_w: int) -> int:
    """Fit a page to the box width on the integer pixel grid."""
    return round(src_h * box_w / src_w)


def _steps(frames: int, page_h: int, box_h: int) -> tuple[int, Fraction]:
    travel = page_h - box_h
    fit = max(0, 1 + (frames - SCROLL_TAIL_FRAMES - SCROLL_LEAD_FRAMES
                      - SCROLL_FLICK_FRAMES) // SCROLL_PERIOD_FRAMES)
    if travel <= 0 or box_h <= 0 or not fit:
        return 0, Fraction(0)
    count = min(fit, max(1, math.floor(travel / (SCROLL_STEP_MIN * box_h))))
    return count, min(Fraction(travel, count), Fraction(str(SCROLL_STEP_MAX)) * box_h)


def scroll_plan(frames: int, page_h: int, box_h: int) -> list[tuple[int, float, float]]:
    """Return local flick starts and page offsets without exceeding its bottom."""
    count, step = _steps(frames, page_h, box_h)
    return [(SCROLL_LEAD_FRAMES + index * SCROLL_PERIOD_FRAMES,
             float(index * step), float((index + 1) * step)) for index in range(count)]


def scroll_offsets(frames: int, page_h: int, box_h: int) -> list[int]:
    """Evaluate the quadratic ease on the same integer grid as crop."""
    count, step = _steps(frames, page_h, box_h)
    offsets = []
    for frame in range(frames):
        value = Fraction(0)
        for index in range(count):
            start = SCROLL_LEAD_FRAMES + index * SCROLL_PERIOD_FRAMES
            if frame < start:
                break
            t = min(Fraction(frame - start, SCROLL_FLICK_FRAMES), 1)
            value = step * (index + 1 - (1 - t) ** 2)
        offsets.append(math.floor(value))
    return offsets


def scroll_expression(frames: int, page_h: int, box_h: int) -> str:
    """Build crop's piecewise expression with exact rational numerators."""
    count, step = _steps(frames, page_h, box_h)
    result = str(math.floor(count * step))
    duration = SCROLL_FLICK_FRAMES
    for index in reversed(range(count)):
        start = SCROLL_LEAD_FRAMES + index * SCROLL_PERIOD_FRAMES
        numerator = step.numerator
        ease = (f"floor(({numerator}*({(index + 1) * duration ** 2}"
                f"-pow({duration}-(n-{start}),2)))/{step.denominator * duration ** 2})")
        result = (f"if(lt(n,{start}),{math.floor(index * step)},"
                  f"if(lt(n,{start + duration}),{ease},{result}))")
    return result
