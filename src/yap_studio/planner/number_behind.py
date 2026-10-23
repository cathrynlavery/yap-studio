"""Fit bare count-up numbers and prove their placement against every matte frame."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import DEFAULT_HAIR, gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.faces.placement import Band
from yap_studio.matte.check import PERSON_ALPHA_THRESHOLD
from yap_studio.render import fonts
from yap_studio.render.text_metrics import text_width_em, widest_digit_em

if TYPE_CHECKING:
    from yap_studio.cards import CardPlan
    from yap_studio.style import ResolvedStyle

BIG_NUMBER_MAX_PX = 300  # Start with large digits above the speaker.
BIG_NUMBER_MIN_PX = 120  # Smaller digits lose the intended big-number look.
BIG_NUMBER_WIDTH_PX = 1000  # Leave space at both edges of the vertical canvas.
COUNT_S = 1.0  # Finish the count quickly enough to read the landed value.
HOLD_S = 0.8  # Reserve a readable rest after the count.
FADE_IN_S = 0.15  # Fade the first count frames to keep the full hold in the 2.1 s minimum.
FADE_OUT_S = 0.3  # Leave the frame gently at the end of the card.
MIN_DURATION_S = COUNT_S + HOLD_S + FADE_OUT_S  # Keep the shortest card readable.
MAX_DECIMALS = 2  # Keep decimal values readable at a glance.
WEIGHT = 700  # Use the bundled heavy face consistently in text and measurements.
CAP_HEIGHT_EM = 0.75  # Bounds the bundled Latin cap heights, including Inter's 0.7275 em.
DESCENT_EM = 0.2  # Reserve comma, affix and label descenders before the shadow.
LABEL_SHARE = 0.16  # Keep the label visibly secondary to the number.
LABEL_MIN_PX = 24  # Keep a scaled label readable on a phone.
LABEL_GAP_PX = 12  # Separate the label from the digits without a panel.
SHADOW_PAD_PX = 24  # Contain the soft shadow and glyph descenders inside the layer.
SHADOW_BLUR_PX = 6  # A soft dark edge keeps text readable over varied rooms.
SHADOW_Y_PX = 4  # Put a little more shadow beneath the text.
SHADOW_ALPHA = 0.7  # Support bright accent colours without a visible backing panel.
MIN_VISIBLE_SHARE = 0.9  # Keep the rendered glyph ink readable behind the person.
PLAN_VISIBLE_SHARE = 0.95  # Keep a conservative fallback when rendered ink is unavailable.
HEAD_TUCK_SHARE = 0.4  # Limit how far the lowest glyph row reaches into the hair allowance.
UP_STEP_PX = 16  # Search upwards in small, stable steps.
INK_UP_STEP_PX = 8  # Refine real ink in smaller steps to preserve the approved hair tuck.
SIDE_STEP_PX = 40  # Search both side corridors without an expensive pixel scan.
FIT_STEP_PX = 1  # Fit a side corridor at the largest whole-pixel digit size.
SCALES = (1.0, 0.9, 0.8, 0.7)  # Prefer movement before reducing the digit size.
CANVAS_WIDTH = 1080  # Match the delivery canvas.
EVEN_PX = 2  # Composite coordinates and layer dimensions use even pixels.


def _even(value: float) -> int:
    return math.ceil(value / EVEN_PX) * EVEN_PX


def _decimals(value: int | float) -> int:
    return max(0, -Decimal(str(value)).normalize().as_tuple().exponent)


def number_string(v: dict, frame: int) -> str:
    """Match the template's cubic count, half-up rounding and fixed decimal places."""
    progress = max(0.0, min(1.0, frame / v["fps"] / v["count_s"]))
    value = v["start"] + (v["end"] - v["start"]) * (1 - (1 - progress) ** 3)
    factor = 10 ** v["decimals"]
    rounded = math.floor(value * factor + 0.5)
    whole, fraction = divmod(rounded, factor)
    text = f"{whole:,}"
    if v["decimals"]:
        text += f".{fraction:0{v['decimals']}d}"
    return f"{v['prefix']}{text}{v['suffix']}"


def _advances(text: str, v: dict) -> list[float]:
    digit = widest_digit_em(v["font_name"], WEIGHT)
    return [digit if char.isdigit() else text_width_em(char, v["font_name"], WEIGHT)
            for char in text]


def resize_number(plan: CardPlan, size: int) -> CardPlan:
    """Rebuild the tight canvas and typography together when placement scales down."""
    v = dict(plan.variables)
    label_px = max(LABEL_MIN_PX, math.floor(size * LABEL_SHARE)) if v["label"] else 0
    width = max(v["widest_em"] * size, v["label_em"] * label_px)
    height = (CAP_HEIGHT_EM + v["digit_descent_em"]) * size
    if v["label"]:
        height += LABEL_GAP_PX + (CAP_HEIGHT_EM + DESCENT_EM) * label_px
    v.update(value_px=size, label_px=label_px)
    return replace(plan, width=_even(width + 2 * SHADOW_PAD_PX),
                   height=_even(height + 2 * SHADOW_PAD_PX), variables=v)


def number_plan(values: dict, style: ResolvedStyle, *, frames: int, fps: int,
                tokens: dict) -> CardPlan:
    """Validate decimal precision and size the widest actual count-up string."""
    from yap_studio.cards import CardPlan

    for key in ("start", "end"):
        if _decimals(values[key]) > MAX_DECIMALS:
            raise ValidationError(f"slots.{key}", "round the number to at most 2 decimal places")
    graphics = style.data["graphics"]
    v = {**tokens, **values, "font_name": graphics["font"], "weight": WEIGHT,
         "font_family": fonts.font_stack(graphics["font"]),
         "accent_color": graphics["accent_color"],
         "label_color": style.data["captions"]["text_color"],
         "fps": fps, "count_s": COUNT_S, "fade_in_s": FADE_IN_S, "fade_out_s": FADE_OUT_S,
         "decimals": max(_decimals(values[key]) for key in ("start", "end")),
         "cap_height_em": CAP_HEIGHT_EM, "shadow_pad_px": SHADOW_PAD_PX,
         "shadow_blur_px": SHADOW_BLUR_PX, "shadow_y_px": SHADOW_Y_PX,
         "shadow_alpha": SHADOW_ALPHA, "label_gap_px": LABEL_GAP_PX}
    v["label"] = values["label"].upper()
    v["label_em"] = text_width_em(v["label"], graphics["font"], WEIGHT)
    rows = [number_string(v, frame) for frame in range(frames)]
    v["widest_em"] = max(sum(_advances(row, v)) for row in rows)
    v["digit_descent_em"] = (DESCENT_EM if values["prefix"] or values["suffix"]
                             or any("," in row for row in rows) else 0)
    size = min(BIG_NUMBER_MAX_PX, math.floor(BIG_NUMBER_WIDTH_PX / v["widest_em"]))
    if size < BIG_NUMBER_MIN_PX:
        raise ValidationError("slots.end",
                              "shorten the number, prefix or suffix to fit at 120 px")
    # Supply the same advance widths to the browser, including fixed tabular digit cells.
    characters = sorted(set("0123456789,." + v["prefix"] + v["suffix"] + v["label"]))
    v["advances"] = json.dumps({char: _advances(char, v)[0] for char in characters})
    base = CardPlan("big_number", 0, 0, 0, 0, frames, fps, v, (graphics["font"],))
    fitted = resize_number(base, size)
    while fitted.width > BIG_NUMBER_WIDTH_PX + 2 * SHADOW_PAD_PX:
        size -= FIT_STEP_PX
        if size < BIG_NUMBER_MIN_PX:
            raise ValidationError("slots.label", "shorten the label to fit beside the number")
        fitted = resize_number(base, size)
    return replace(fitted, x=(CANVAS_WIDTH - fitted.width) // EVEN_PX // EVEN_PX * EVEN_PX)


def glyph_rows(plan: CardPlan, frame: int) -> list[tuple[str, float, float]]:
    """Return the number and optional label with their top and pixel size."""
    v = plan.variables
    top = plan.y + SHADOW_PAD_PX
    rows = [(number_string(v, frame), top, v["value_px"])]
    if v["label"]:
        rows.append((v["label"], top + (CAP_HEIGHT_EM + v["digit_descent_em"])
                     * v["value_px"] + LABEL_GAP_PX,
                     v["label_px"]))
    return rows


def _visible_share(plan: CardPlan, alpha: np.ndarray, frame: int) -> float:
    """Count person pixels in each centred advance row at cap height."""
    worst = 1.0
    for text, top, size in glyph_rows(plan, frame):
        advances = _advances(text, plan.variables)
        cursor = plan.x + (plan.width - sum(advances) * size) / 2
        total = covered = 0
        for char, advance in zip(text, advances, strict=True):
            left, right = math.floor(cursor), math.ceil(cursor + advance * size)
            first, last = math.floor(top), math.ceil(top + CAP_HEIGHT_EM * size)
            if not char.isspace():
                total += (right - left) * (last - first)
                covered += int(np.count_nonzero(alpha[first:last, left:right]
                                                >= PERSON_ALPHA_THRESHOLD))
            cursor += advance * size
        worst = min(worst, 1 - covered / total if total else 1.0)
    return worst


def provisional_number(plan: CardPlan, *, margin_px: float) -> tuple[CardPlan, str, dict]:
    """Keep cards plan matte-free; graphics owns the final head placement."""
    return replace(plan, y=0), "top", {
        "status": "provisional", "margin_px": margin_px, "hair": DEFAULT_HAIR.to_dict(),
        "reason": "graphics places this number against the cut person matte", "gap_px": None,
    }


def _positions(
    plan: CardPlan, head_top: float, band_top: int, *, step: int, first: str = "tucked",
) -> Iterator[tuple[CardPlan, str]]:
    """Try up before both side corridors while respecting the same top bound."""
    ys = []
    y = plan.y
    while y >= band_top:
        ys.append(y)
        if y + plan.height - SHADOW_PAD_PX <= head_top:
            break
        y -= step
    for index, y in enumerate(ys):
        yield replace(plan, y=y), first if index == 0 else "up"
    for distance in range(SIDE_STEP_PX, CANVAS_WIDTH, SIDE_STEP_PX):
        for x in (plan.x - distance, plan.x + distance):
            if 0 <= x <= CANVAS_WIDTH - plan.width:
                for y in ys:
                    yield replace(plan, x=x, y=y), "side"


def _box_visible_share(plan: CardPlan, mattes: Sequence[np.ndarray], minimum: float) -> float:
    worst = 1.0
    for frame, alpha in enumerate(mattes):
        worst = min(worst, _visible_share(plan, alpha, frame))
        if worst < minimum:
            break
    return worst


def place_number(
    plan: CardPlan, faces: Sequence[Sequence[Box]] | None,
    mattes: Sequence[np.ndarray], *, caption_top: int, band: Band, margin_px: float,
    identifier: str, face_windows: tuple[tuple[int, int], ...] = (),
    ink_visible_share: Callable[[CardPlan], float] | None = None,
) -> tuple[CardPlan, str, dict]:
    """Choose the first readable tucked position, then a clear front side corridor."""
    if faces is None or len(faces) != plan.frames or len(mattes) != plan.frames:
        raise ValidationError("big_number.clearance", "measure faces and the cut matte again")
    heads = [head_box(face) for frame in faces for face in frame]
    if not heads:
        raise ValidationError("big_number.clearance", "choose another moment with a measured head")
    centre = sum(head.x + head.w / 2 for head in heads) / len(heads)
    tuck_bottom = min(head.y + HEAD_TUCK_SHARE * head.h for head in heads)
    original_size = plan.variables["value_px"]
    head_top = min(head.y for head in heads)
    box_minimum = MIN_VISIBLE_SHARE if ink_visible_share is not None else PLAN_VISIBLE_SHARE
    for scale in (() if face_windows else SCALES):
        size = math.floor(original_size * scale)
        if size < BIG_NUMBER_MIN_PX:
            continue
        fitted = resize_number(plan, size)
        x = max(0, min(CANVAS_WIDTH - fitted.width, _even(centre - fitted.width / 2)))
        lowest = fitted.height - SHADOW_PAD_PX
        # Tucked numbers may leave the top band; only captions limit their bottom edge.
        y = math.floor(min(tuck_bottom - lowest, caption_top - fitted.height) / EVEN_PX) * EVEN_PX
        start = replace(fitted, x=x, y=y)
        for candidate, _position in _positions(start, head_top, band.top, step=UP_STEP_PX):
            box_share = _box_visible_share(candidate, mattes, box_minimum)
            if box_share >= box_minimum:
                break
        else:
            continue
        positions = (_positions(candidate, head_top, band.top, step=INK_UP_STEP_PX, first=_position)
                     if ink_visible_share is not None else ((candidate, _position),))
        for candidate, position in positions:
            worst = ink_visible_share(candidate) if ink_visible_share is not None else box_share
            if worst >= box_minimum:
                measured = ink_visible_share is not None
                return candidate, "behind_head", {
                    "status": "behind_number", "position": position, "scale": size / original_size,
                    "worst_visible_share": round(worst, 4) if measured else worst,
                    "box_visible_share": _box_visible_share(candidate, mattes, 0),
                    "ink_status": "verified" if measured else "not_measured",
                    "x": candidate.x, "y": candidate.y,
                    "value_px": candidate.variables["value_px"], "margin_px": margin_px,
                    "hair": DEFAULT_HAIR.to_dict(),
                    "reason": (("rendered glyphs" if measured else "glyph boxes")
                               + f" stay at least {box_minimum:.0%} visible on every frame"),
                }
    corridors = [(0, min(head.x for head in heads) - margin_px),
                 (max(head.x1 for head in heads) + margin_px, CANVAS_WIDTH)]
    # A narrow corridor can need more shrinking than the behind search permits.
    for left, right in corridors:
        for size in range(original_size, BIG_NUMBER_MIN_PX - FIT_STEP_PX, -FIT_STEP_PX):
            side = resize_number(plan, size)
            x = _even(max(0, left))
            preferred_y = min(min(head.y for head in heads), caption_top - side.height)
            y = math.floor(max(0, preferred_y) / EVEN_PX) * EVEN_PX
            if x + side.width > right or y + side.height > caption_top:
                continue
            side = replace(side, x=x, y=y)
            if min(gap_px(Box(x, y, side.width, side.height), head) for head in heads) < margin_px:
                continue
            reason = ("a zoom, insert or transition overlaps the number" if face_windows else
                      "the number would be hidden behind her head")
            return side, "top", {
                "status": "beside_front", "position": "side", "x": x, "y": y,
                "scale": size / original_size, "value_px": size,
                "worst_visible_share": 1.0, "margin_px": margin_px,
                "ink_status": "front",
                "hair": DEFAULT_HAIR.to_dict(), "reason": reason,
                "warning": f"{identifier}: {reason}, so it sits beside her head in front",
            }
    raise ValidationError("big_number.clearance",
                          "use a shorter number, a smaller label, or another moment")
