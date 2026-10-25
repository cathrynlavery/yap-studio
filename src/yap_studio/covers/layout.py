"""Measure cover copy and verify the visible text."""

from __future__ import annotations

import copy
import itertools
import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from yap_studio.cards import _contrast, _luminance
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import head_box
from yap_studio.faces.model import Box
from yap_studio.render.text_metrics import text_width_em
from yap_studio.sfx import bleep

SAFE_X = (60, 1020)
SAFE_Y = (280, 1640)
TEXT_WIDTH = SAFE_X[1] - SAFE_X[0]
TEXT_X = 72
GLYPH_WIDTH = SAFE_X[1] - TEXT_X
NUMBER_WIDTH = 460
KICKER_OFFSET = math.ceil(48 * 1.2) + 8
CAPS = {"headline": 60, "kicker": 32, "number": 6, "label": 40,
        "before": 10, "after": 10}
USED = {
    "photo": {"headline", "kicker", "inset"},
    "number": {"number", "label"},
    "hook": {"headline", "kicker", "label"},
    "before-after": {"before", "after", "kicker", "label"},
}
REQUIRED = {"photo": {"headline"}, "number": {"number", "label"},
            "hook": {"headline", "label"}, "before-after": {"before", "after"}}


@dataclass(frozen=True)
class LayoutPlan:
    """Measured text boxes and non-text placement derived from the face."""

    blocks: tuple[dict, ...]
    speaker_side: str | None = None
    speaker_shift_px: int = 0
    inset_x: int | None = None
    inset_y: int | None = None
    inset_side: str | None = None
    inset_scale: float | None = None
    inset_width: int | None = None
    inset_height: int | None = None
    inset_padding: int | None = None
    inset_bottom_padding: int | None = None
    inset_box: tuple[int, int, int, int] | None = None


def parse_accent(value: str) -> tuple[str, str | None]:
    """Remove one whole-word italic span from a headline."""
    plain, phrase, _, _ = _accent_span(value)
    return plain, phrase


def _accent_span(value: str) -> tuple[str, str | None, int, int]:
    """Return plain text and the marked span within it."""
    if "*" not in value:
        return value, None, -1, -1
    if value.count("*") != 2:
        raise ValidationError("headline", "use one *accent phrase* with whole words")
    match = re.fullmatch(r"(.*?)\*([^*]+)\*(.*?)", value)
    if match is None or not match[2].strip() or match[2] != match[2].strip():
        raise ValidationError("headline", "use one *accent phrase* with whole words")
    left, accent, right = match.groups()
    if (not accent[0].isalnum() or not accent[-1].isalnum() or
            (left and not (left[-1].isspace() or left[-1] in '"\'“‘([{')) or
            (right and not (right[0].isspace() or right[0] in '"\'”’,.;:!?)]}'))):
        raise ValidationError("headline", "accent must surround whole words")
    return left + accent + right, accent, len(left), len(left) + len(accent)


def balance_lines(
    words: list[str], width_of: Callable[[str], float], max_width: float, max_lines: int,
    *, forbidden_breaks: tuple[int, ...] = (),
) -> list[str]:
    """Choose the smallest widest line, then the most even line widths."""
    # This can merge with the shared caption text rules later.
    if not words:
        raise ValidationError("headline", "enter a shorter headline")
    possibilities = []
    for count in range(1, min(max_lines, len(words)) + 1):
        for breaks in itertools.combinations(range(1, len(words)), count - 1):
            if any(edge in forbidden_breaks for edge in breaks):
                continue
            edges = (0, *breaks, len(words))
            groups = [words[edges[i]:edges[i + 1]] for i in range(count)]
            if len(words) > 1 and any(len(group) == 1 for group in groups):
                continue
            lines = [" ".join(group) for group in groups]
            widths = [width_of(line) for line in lines]
            if max(widths) <= max_width:
                possibilities.append((max(widths), max(widths) - min(widths), count, lines))
    if not possibilities:
        raise ValidationError("headline", "shorten to fit without an orphan word")
    return min(possibilities)[3]


def _copy(look: str, values: dict[str, str | None]) -> dict[str, str]:
    if look not in USED:
        raise ValidationError("look", "choose photo, number, hook or before-after")
    supplied = {key: value for key, value in values.items() if value is not None}
    for key in supplied:
        if key not in USED[look]:
            raise ValidationError(key, f"{key} is not used by the {look} look")
    for key in REQUIRED[look]:
        if not supplied.get(key, "").strip():
            raise ValidationError(key, f"{key} is required for the {look} look")
    clean = {}
    for key, value in supplied.items():
        if not isinstance(value, str):
            raise ValidationError(key, "expected text")
        if any(ord(char) < 32 or 127 <= ord(char) <= 159 or char in "\u2028\u2029"
               for char in value):
            raise ValidationError(key, "remove control characters")
        if key == "inset":
            clean[key] = value
            continue
        if not value.strip():
            raise ValidationError(key, "enter text or omit this flag")
        plain, _ = parse_accent(value) if key == "headline" else (value, None)
        if value != value.strip() or re.search(r"\s{2,}", plain):
            raise ValidationError(key, "use single spaces and remove outer spaces")
        cap = CAPS[key]
        if len(plain) > cap:
            raise ValidationError(key, f"shorten to at most {cap} characters")
        clean[key] = value
    return clean


def _width(text: str, font: str, size: int, *, small_caps: bool = False) -> float:
    return text_width_em(text.upper() if small_caps else text, font, 400,
                         tracking_em=0.12 if small_caps else 0) * size


def _single_limit(text: str, font: str, width: int, minimum: int, *,
                  small_caps: bool = False) -> int:
    return max((count for count in range(len(text) + 1)
                if _width(text[:count], font, minimum, small_caps=small_caps) <= width),
               default=0)


def _fit_single(name: str, text: str, font: str, width: int, minimum: int,
                maximum: int, *, small_caps: bool = False) -> int:
    for size in range(maximum, minimum - 1, -1):
        if _width(text, font, size, small_caps=small_caps) <= width:
            return size
    limit = _single_limit(text, font, width, minimum, small_caps=small_caps)
    raise ValidationError(name, f"shorten to at most {limit} characters to fit at {minimum} px")


def _headline_at(text: str, font: str, start: int, bottom: int, size: int) -> list[str]:
    line_height = math.ceil(size * 1.12)
    max_lines = min(3, (bottom - start) // line_height)
    if max_lines < 1:
        raise ValidationError("headline", "shorten the headline to fit the top third")
    return balance_lines(text.split(), lambda line: _width(line, font, size),
                         GLYPH_WIDTH, max_lines)


def _headline_fits(text: str, font: str, start: int, bottom: int) -> bool:
    try:
        _headline_at(text, font, start, bottom, 72)
    except ValidationError:
        return False
    return True


def _headline_limit(text: str, font: str, start: int, bottom: int) -> int:
    words = text.split()
    candidates = [" ".join(words[:count]) for count in range(1, len(words) + 1)]
    if words:
        candidates.extend(words[0][:count] for count in range(1, len(words[0])))
    return max((len(candidate) for candidate in candidates
                if _headline_fits(candidate, font, start, bottom)), default=0)


def headline_lines(text: str, font: str, band: tuple[int, int], *, maximum: int = 150,
                   minimum: int = 72, kicker: bool = False,
                   literal: bool = False) -> tuple[list[str], int]:
    """Choose the largest measured size that fits the headline band."""
    plain = text if literal else parse_accent(text)[0]
    start = band[0] + (KICKER_OFFSET if kicker else 0)
    for size in range(maximum, minimum - 1, -1):
        try:
            return _headline_at(plain, font, start, band[1], size), size
        except ValidationError:
            continue
    limit = _headline_limit(plain, font, start, band[1])
    raise ValidationError("headline", f"shorten to at most {limit} characters "
                          "to fit the top third at 72 px")


def _block(role: str, text: str, x: int, y: int, width: int, size: int,
           color: str, font: str, *, small_caps: bool = False,
           opacity: float = 1.0, strike: bool = False) -> dict:
    height = math.ceil(size * (1.2 if small_caps else 1.12))
    return {"role": role, "text": text, "x": x, "y": y, "width": width,
            "height": height, "font_size": size, "color": color, "font": font,
            "small_caps": small_caps, "opacity": opacity, "strike": strike}


def _check_box(block: dict, band: tuple[int, int]) -> None:
    x, y = block["x"], block["y"]
    right, bottom = x + block["width"], y + block["height"]
    if x < SAFE_X[0] or right > SAFE_X[1] or y < SAFE_Y[0] or bottom > SAFE_Y[1]:
        raise ValidationError(block["role"], "3:4 crop safe area failed; "
                              "shorten text or choose another frame")
    if block["role"] == "headline" and (y < band[0] or bottom > band[1]):
        raise ValidationError("headline", "top third check failed; shorten the headline")


def _headline_blocks(look: str, clean: dict[str, str], style: dict,
                     band: tuple[int, int], title_font: str,
                     graphics_font: str) -> list[dict]:
    text_color = style["captions"]["text_color"]
    plain, phrase, accent_start, accent_end = _accent_span(clean["headline"])
    censored = bleep.enabled(style)
    if censored:
        plain = bleep.mask_text(plain, bleep.listed(style["music"]))
    accent = readable_accent(style, "#101010") if phrase else text_color
    lines, size = headline_lines(plain if censored else clean["headline"], title_font, band,
                                 kicker=bool(clean.get("kicker")), literal=censored)
    blocks = []
    if kicker := clean.get("kicker"):
        kicker_size = _fit_single("kicker", kicker, graphics_font, GLYPH_WIDTH, 40, 48,
                                  small_caps=True)
        block = _block("kicker", kicker, SAFE_X[0], band[0], TEXT_WIDTH,
                       kicker_size, text_color, graphics_font, small_caps=True)
        block["text_inset"] = TEXT_X - SAFE_X[0]
        blocks.append(block)
    y = band[0] + (KICKER_OFFSET if kicker else 0)
    offset = 0
    if phrase:
        accent_start = len(re.sub(r"\s+", " ", plain[:accent_start]))
        accent_end = len(re.sub(r"\s+", " ", plain[:accent_end]))
    for line in lines:
        block = _block("headline", line, SAFE_X[0], y, TEXT_WIDTH, size,
                       text_color, title_font)
        block["text_inset"] = TEXT_X - SAFE_X[0]
        block["accent_color"] = accent
        overlaps = accent_start < offset + len(line) and accent_end > offset
        block["accent_start"] = max(0, accent_start - offset) if overlaps else -1
        block["accent_end"] = min(len(line), accent_end - offset) if overlaps else -1
        blocks.append(block)
        y += block["height"]
        offset += len(line) + 1
    if look == "hook":
        proof = clean["label"]
        proof_size = _fit_single("label", proof, graphics_font, GLYPH_WIDTH, 40, 48,
                                 small_caps=True)
        block = _block("proof", proof, SAFE_X[0], y + 10, TEXT_WIDTH,
                       proof_size, text_color, graphics_font, small_caps=True)
        block["text_inset"] = TEXT_X - SAFE_X[0]
        blocks.append(block)
    return blocks


def _number_blocks(clean: dict[str, str], style: dict,
                   face_box: tuple[float, float, float, float] | None,
                   title_font: str, graphics_font: str) -> tuple[list[dict], str, int]:
    if face_box is None:
        raise ValidationError("face", "face box unavailable; choose another frame")
    centre = face_box[0] + face_box[2] / 2
    side = "left" if centre < 540 else "right"
    x = 560 if side == "left" else 60
    shift = round((300 if side == "left" else 780) - centre)
    paper = style["graphics"]["paper_color"]
    ink = style["graphics"]["ink_color"]
    try:
        accent = readable_accent(style, paper)
    except ValidationError:
        accent = None
    number_color = accent if accent and _contrast(accent, paper) >= 3 else ink
    if _contrast(number_color, paper) < 3:
        raise ValidationError("number", "style colors lack 3:1 contrast; choose another style")
    label_color = next((color for color in (ink, accent)
                        if color and _contrast(color, paper) >= 4.5), None)
    if label_color is None:
        raise ValidationError("label", "style colors lack 4.5:1 contrast; choose another style")
    size = _fit_single("number", clean["number"], title_font, NUMBER_WIDTH, 220, 440)
    label_size = _fit_single("label", clean["label"], graphics_font, NUMBER_WIDTH, 40, 48,
                             small_caps=True)
    total_height = math.ceil(size * 1.12) + 12 + math.ceil(label_size * 1.2)
    face_y = face_box[1] + face_box[3] / 2
    y = round(max(SAFE_Y[0], min(SAFE_Y[1] - total_height,
                                 face_y - total_height / 2)))
    number = _block("number", clean["number"], x, y, NUMBER_WIDTH, size,
                    number_color, title_font)
    label = _block("label", clean["label"], x, number["y"] + number["height"] + 12,
                   NUMBER_WIDTH, label_size, label_color, graphics_font, small_caps=True)
    return [number, label], side, shift


def _before_after_blocks(clean: dict[str, str], style: dict,
                         face_box: tuple[float, float, float, float] | None,
                         title_font: str, graphics_font: str) -> list[dict]:
    if face_box is None:
        raise ValidationError("face", "face box unavailable; choose another frame")
    before_size = _fit_single("before", clean["before"], title_font, GLYPH_WIDTH, 96, 140)
    after_size = _fit_single("after", clean["after"], title_font, GLYPH_WIDTH, 160, 300)
    kicker = clean.get("kicker")
    label = clean.get("label")
    kicker_size = (_fit_single("kicker", kicker, graphics_font, GLYPH_WIDTH, 40, 48,
                               small_caps=True) if kicker else 0)
    label_size = (_fit_single("label", label, graphics_font, GLYPH_WIDTH, 40, 48,
                              small_caps=True) if label else 0)
    above_end = min(SAFE_Y[1], math.floor(face_box[1] - 40))
    below_start = max(SAFE_Y[0], math.ceil(face_box[1] + face_box[3] + 40))
    regions = ((SAFE_Y[0], above_end), (below_start, SAFE_Y[1]))
    accent = readable_accent(style, "#000000")
    text_color = style["captions"]["text_color"]
    for top, bottom in regions:
        for after in range(after_size, 159, -1):
            for before in range(before_size, 95, -1):
                height = (math.ceil(kicker_size * 1.2) + 6 if kicker else 0)
                height += math.ceil(before * 1.12) + 16 + math.ceil(after * 1.12)
                height += 10 + math.ceil(label_size * 1.2) if label else 0
                if top + height > bottom:
                    continue
                y = top
                blocks = []
                if kicker:
                    block = _block("kicker", kicker, SAFE_X[0], y, TEXT_WIDTH,
                                   kicker_size, text_color, graphics_font, small_caps=True)
                    blocks.append(block)
                    y += block["height"] + 6
                block = _block("before", clean["before"], SAFE_X[0], y, TEXT_WIDTH,
                               before, text_color, title_font, opacity=0.55, strike=True)
                blocks.append(block)
                y += block["height"] + 16
                block = _block("after", clean["after"], SAFE_X[0], y, TEXT_WIDTH,
                               after, accent, title_font)
                blocks.append(block)
                y += block["height"] + 10
                if label:
                    blocks.append(_block("label", label, SAFE_X[0], y, TEXT_WIDTH,
                                         label_size, text_color, graphics_font,
                                         small_caps=True))
                for item in blocks:
                    item["text_inset"] = TEXT_X - SAFE_X[0]
                return blocks
    raise ValidationError("before-after", "no clear space around the face; "
                          "choose another frame or shorter values")


def _overlap(a: Box, b: Box) -> bool:
    return a.x < b.x1 and b.x < a.x1 and a.y < b.y1 and b.y < a.y1


def _inset_placement(face_box: tuple[float, float, float, float],
                     blocks: list[dict]) -> dict:
    side = "right" if face_box[0] + face_box[2] / 2 < 540 else "left"
    head = head_box(Box(*face_box))
    clearance = Box(head.x - 24, head.y - 24, head.w + 48, head.h + 48)
    angle = math.radians(6)
    for scale in (1, .95, .9, .85, .8, .75, .7):
        width, height = round(300 * scale), round(360 * scale)
        rotated_width = width * math.cos(angle) + height * math.sin(angle)
        rotated_height = width * math.sin(angle) + height * math.cos(angle)
        x = (math.ceil(SAFE_X[0] + (rotated_width - width) / 2)
             if side == "left" else
             math.floor(SAFE_X[1] - (rotated_width + width) / 2))
        y = math.floor(SAFE_Y[1] - (rotated_height + height) / 2)
        left = math.floor(x + (width - rotated_width) / 2)
        top = math.floor(y + (height - rotated_height) / 2)
        right = math.ceil(x + (width + rotated_width) / 2)
        bottom = math.ceil(y + (height + rotated_height) / 2)
        rotated = Box(left, top, right - left, bottom - top)
        if not (SAFE_X[0] <= left and right <= SAFE_X[1] and
                SAFE_Y[0] <= top and bottom <= SAFE_Y[1]):
            continue
        if _overlap(rotated, clearance):
            continue
        if any(_overlap(rotated, Box(block["x"], block["y"], block["width"],
                                     block["height"])) for block in blocks):
            continue
        return {"inset_x": x, "inset_y": y, "inset_side": side,
                "inset_scale": scale, "inset_width": width, "inset_height": height,
                "inset_padding": round(14 * scale),
                "inset_bottom_padding": round(58 * scale),
                "inset_box": (left, top, right - left, bottom - top)}
    raise ValidationError("inset", "inset cannot clear the head and text inside the 3:4 "
                          "crop; try another eligible frame with --frame N or omit --inset")


def plan_blocks(look: str, values: dict[str, str | None], style: dict, *,
                face_box: tuple[float, float, float, float] | None = None,
                shape: str = "vertical") -> LayoutPlan:
    """Compute all text boxes from one style, copy, crop, and face box."""
    clean = _copy(look, values)
    if shape == "horizontal":
        from yap_studio.covers.wide import plan_wide

        return plan_wide(look, clean, style, face_box)
    if bleep.enabled(style):
        vocabulary = bleep.listed(style["music"])
        clean = {key: bleep.mask_text(value, vocabulary) if key not in {"headline", "inset"}
                 else value for key, value in clean.items()}
    title_font = style["hook"]["font"]
    graphics_font = style["graphics"]["font"]
    band = (style["internal"]["key_text_top_px"],
            style["internal"]["key_text_bottom_px"])
    side = None
    shift = 0
    inset = {}
    if look in {"photo", "hook"}:
        blocks = _headline_blocks(look, clean, style, band, title_font, graphics_font)
    elif look == "number":
        blocks, side, shift = _number_blocks(clean, style, face_box, title_font, graphics_font)
    else:
        blocks = _before_after_blocks(clean, style, face_box, title_font, graphics_font)
    if "inset" in clean:
        if face_box is None:
            raise ValidationError("inset", "face box unavailable; choose another frame")
        inset = _inset_placement(face_box, blocks)
    for block in blocks:
        _check_box(block, band)
    return LayoutPlan(tuple(blocks), side, shift, **inset)


def validate_text(look: str, values: dict[str, str | None], *, font: str = "Instrument Serif",
                  graphics_font: str = "Inter", band: tuple[int, int] = (280, 640),
                  face_box: tuple[float, float, float, float] | None = None,
                  measure: bool = True) -> dict[str, str]:
    """Validate flags, caps, and optionally the shared measured layout."""
    clean = _copy(look, values)
    if measure:
        from yap_studio.style import resolve

        style = copy.deepcopy(resolve("preset:editorial_serif").data)
        style["hook"]["font"] = font
        style["graphics"]["font"] = graphics_font
        style["internal"]["key_text_top_px"], style["internal"]["key_text_bottom_px"] = band
        plan_blocks(look, clean, style, face_box=face_box or (450, 350, 500, 700))
    return clean


def check_geometry(mask: np.ndarray, *, blocks: tuple[dict, ...] | list[dict] | None = None,
                   headline_box: tuple[int, int, int, int] | None = None,
                   look: str = "photo", band: tuple[int, int] = (280, 640),
                   shape: str = "vertical") -> dict:
    """Reject glyphs outside the planned blocks and the profile crop inset."""
    yy, xx = np.nonzero(mask > 16)
    if not len(xx):
        raise ValidationError("text", "no text rendered; choose another look")
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    if shape == "horizontal":
        from yap_studio.covers.wide import check_box

        check_box(Box(box[0], box[1], box[2] - box[0], box[3] - box[1]))
        for block in blocks or ():
            check_box(Box(*(block[key] for key in ("x", "y", "width", "height"))))
        covered = np.zeros(mask.shape, dtype=bool)
        for block in blocks or ():
            x, y = block["x"], block["y"]
            covered[y:y + block["height"], x:x + block["width"]] = True
        if blocks is not None and np.any((mask > 16) & ~covered):
            raise ValidationError("text", "glyphs escaped a planned block; shorten the text")
        return {"passed": True, "text_box": box, "badge_clear": True}
    if box[0] < SAFE_X[0] or box[2] > SAFE_X[1] or box[1] < SAFE_Y[0] or box[3] > SAFE_Y[1]:
        raise ValidationError("text", f"3:4 crop safe area failed at {box}; shorten the text")
    if blocks is not None:
        covered = np.zeros(mask.shape, dtype=bool)
        for block in blocks:
            _check_box(block, band)
            x, y = block["x"], block["y"]
            covered[y:y + block["height"], x:x + block["width"]] = True
        if np.any((mask > 16) & ~covered):
            raise ValidationError("text", "glyphs escaped a planned block; shorten the text")
    elif look in {"photo", "hook"} and headline_box is not None:
        _, top, _, height = headline_box
        if top < band[0] or top + height > band[1]:
            raise ValidationError("headline", "top third check failed; shorten the headline")
        if np.any(mask[:band[0]] > 16) or np.any(mask[band[1]:] > 16):
            raise ValidationError("headline", "top third glyph check failed; shorten the headline")
    return {"passed": True, "text_box": box, "headline_box": headline_box}


def check_contrast(mask: np.ndarray, backdrop: np.ndarray, color: str, *,
                   minimum: float) -> float:
    """Check the tenth percentile contrast at glyph pixels."""
    pixels = backdrop[mask > 16] if mask.dtype != bool else backdrop[mask]
    if not len(pixels):
        raise ValidationError("contrast", "no glyph pixels rendered")
    rgb = pixels[:, :3].astype(np.float64) / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = linear @ np.array([0.2126, 0.7152, 0.0722])
    text_lum = _luminance(color)
    ratios = (np.maximum(lum, text_lum) + 0.05) / (np.minimum(lum, text_lum) + 0.05)
    measured = float(np.percentile(ratios, 10))
    if measured < minimum:
        raise ValidationError("contrast", f"contrast {measured:.2f}:1 is below {minimum}:1; "
                              "choose another frame or look")
    return measured


def readable_accent(style: dict, backdrop: str) -> str:
    """Choose the first style accent readable on a given ground."""
    candidates = ((style.get("brand") or {}).get("accent_color"),
                  style["captions"].get("highlight_color"),
                  style["hook"].get("accent_color"),
                  style["graphics"].get("accent_color"))
    chosen = next((color for color in candidates if color and _contrast(color, backdrop) >= 3),
                  None)
    if chosen is None:
        raise ValidationError("accent", "no style accent reaches 3:1 contrast; "
                              "choose another look or style")
    return chosen
