"""Independent 1280x720 layout tables and YouTube preview checks."""
from __future__ import annotations

import math

from yap_studio.covers.layout import (
    LayoutPlan,
    _accent_span,
    _block,
    _fit_single,
    _overlap,
    _width,
    balance_lines,
    readable_accent,
)
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import head_box
from yap_studio.faces.model import Box
from yap_studio.sfx import bleep

# Each look owns its column and type hierarchy; no portrait dimensions are scaled.
WIDE_LAYOUTS = {
    'photo': {'left': 56, 'right': 656, 'width': 568, 'top': 112, 'bottom': 592,
              'maximum': 146, 'minimum': 104, 'lines': 3},
    'number': {'left': 56, 'right': 656, 'width': 568, 'top': 112, 'bottom': 592,
               'maximum': 300, 'minimum': 160, 'lines': 1},
    'before-after': {'left': 56, 'right': 656, 'width': 568, 'top': 112, 'bottom': 592,
                     'maximum': 142, 'minimum': 104, 'lines': 2},
}
BADGE = Box(1080, 620, 200, 100)


def thumbnail_face(box: Box, width: int, height: int) -> Box:
    """Map source face pixels through the same cover scale/crop as extraction."""
    scale = max(1280 / width, 720 / height)
    return Box(box.x * scale - (width * scale - 1280) / 2,
               box.y * scale - (height * scale - 720) / 2,
               box.w * scale, box.h * scale)


def check_box(box: Box) -> None:
    """Keep important content inside the safe area and outside the duration badge."""
    if _overlap(box, BADGE):
        raise ValidationError('text', 'YouTube duration badge corner failed; shorten the text')
    if box.x < 48 or box.y < 48 or box.x1 > 1232 or box.y1 > 672:
        raise ValidationError('text', 'wide safe area failed; shorten text or choose another frame')


def check_face(blocks: list[dict] | tuple[dict, ...], face: tuple) -> dict:
    """Reserve the head and a little breathing room, including hair."""
    head = head_box(Box(*face))
    clearance = Box(head.x - 24, head.y - 24, head.w + 48, head.h + 48)
    if _overlap(head, BADGE):
        raise ValidationError('face', 'face overlaps the duration badge; choose another frame')
    for block in blocks:
        if _overlap(clearance, Box(*(block[k] for k in ('x', 'y', 'width', 'height')))):
            raise ValidationError('face', 'headline needs room beside the face; '
                                  'choose another frame')
    return {'passed': True, 'head_box': [head.x, head.y, head.w, head.h]}


def check_legibility(blocks: list[dict], *, look: str = 'photo') -> dict:
    """Require measured glyphs to survive reduction to the 168x94 browse size."""
    rows = []
    for block in blocks:
        height = (block['measured_box'][3] - block['measured_box'][1]) * 94 / 720
        secondary = block['role'] in {'kicker', 'label', 'proof'}
        minimum = 4 if secondary and not (look == 'number' and block['role'] == 'label') else 8
        if height < minimum:
            raise ValidationError('text', f"{block['role']} glyph height {height:.1f} px at "
                                  f'168x94 is below {minimum}; shorten text')
        rows.append({'role': block['role'], 'glyph_height_px': round(height, 2),
                     'minimum': minimum})
    return {'passed': True, 'preview': [168, 94], 'blocks': rows}


def _lines(role: str, text: str, font: str, table: dict, *, maximum: int | None = None,
           minimum: int | None = None, lines: int | None = None,
           height: int = 720) -> tuple[list[str], int]:
    for size in range(maximum or table['maximum'], (minimum or table['minimum']) - 1, -1):
        try:
            result = balance_lines(text.split(), lambda line, size=size: _width(line, font, size),
                                   table['width'] - 40, lines or table['lines'])
            if len(result) * math.ceil(size * 1.12) <= height:
                return result, size
        except ValidationError:
            continue
    raise ValidationError(role, 'shorten text to fit the thumbnail without an orphan word')


def _inset(face: tuple, blocks: list[dict]) -> dict:
    # The rotated bounds, rather than its unrotated CSS box, must clear all content.
    angle = math.radians(6)
    for width in (210, 180, 150):
        height = round(width * 1.2)
        rw = math.ceil(width * math.cos(angle) + height * math.sin(angle))
        rh = math.ceil(height * math.cos(angle) + width * math.sin(angle))
        for x in (56, 1224 - rw):
            for y in (56, 664 - rh):
                box = Box(x, y, rw, rh)
                try:
                    check_box(box)
                    check_face([{'x': x, 'y': y, 'width': rw, 'height': rh}], face)
                except ValidationError:
                    continue
                if any(_overlap(box, Box(*(b[k] for k in ('x', 'y', 'width', 'height'))))
                       for b in blocks):
                    continue
                return {'inset_x': x + (rw - width) // 2,
                        'inset_y': y + (rh - height) // 2,
                        'inset_width': width, 'inset_height': height,
                        'inset_padding': 9, 'inset_bottom_padding': 30,
                        'inset_scale': width / 300,
                        'inset_side': 'left' if x < 640 else 'right',
                        'inset_box': (x, y, rw, rh)}
    raise ValidationError('inset', 'inset cannot clear face, text and badge; omit inset')


def plan_wide(look: str, clean: dict, style: dict, face: tuple | None) -> LayoutPlan:
    """Lay copy on the open side of an unchanged photo."""
    if face is None:
        raise ValidationError('face', 'wide thumbnail needs a face; choose another frame')
    if look == 'hook':
        table = WIDE_LAYOUTS['photo']
    else:
        table = WIDE_LAYOUTS[look]
    if bleep.enabled(style):
        vocabulary = bleep.listed(style['music'])
        clean = {k: bleep.mask_text(v, vocabulary) if k != 'inset' else v
                 for k, v in clean.items()}
    side = 'left' if face[0] + face[2] / 2 < 640 else 'right'
    x = table['right'] if side == 'left' else table['left']
    font, small_font = style['hook']['font'], style['graphics']['font']
    color = style['captions']['text_color']
    accent = readable_accent(style, '#101010')
    y = table['top']
    blocks = []

    def add(role, text, size, *, ink=color, caps=False, italic=False, strike=False):
        nonlocal y
        block = _block(role, text, x, y, table['width'], size, ink,
                       small_font if caps else font, small_caps=caps, strike=strike)
        block['text_inset'] = 24
        if look == 'number':
            block['text_inset'] = round((table['width'] - _width(text, font, size)) / 2)
        block['italic'] = italic
        blocks.append(block)
        y += block['height']
        return block

    if kicker := clean.get('kicker'):
        size = _fit_single('kicker', kicker, small_font, table['width'] - 40,
                           42, 44, small_caps=True)
        add('kicker', kicker, size, caps=True)
        y += 12
    if look in {'photo', 'hook'}:
        plain, phrase, first, last = _accent_span(clean['headline'])
        lines, size = _lines('headline', plain, font, table,
                             height=table['bottom'] - y - (80 if clean.get('label') else 0))
        offset = 0
        for line in lines:
            block = add('headline', line, size)
            if phrase and first < offset + len(line) and last > offset:
                block.update(accent_start=max(0, first - offset),
                             accent_end=min(len(line), last - offset), accent_color=accent)
            offset += len(line) + 1
    elif look == 'number':
        labels, label_size = _lines('label', clean['label'], font, table,
                                    maximum=112, minimum=104, lines=2)
        available = table['bottom'] - y - len(labels) * math.ceil(label_size * 1.12) - 16
        size = _fit_single('number', clean['number'], font, table['width'] - 40,
                           table['minimum'], min(table['maximum'], math.floor(available / 1.12)))
        number = add('number', clean['number'], size, ink=accent)
        # Trim both fonts' whitespace so the visible gap is about a quarter cap height.
        trim = round((size + label_size) * .12)
        number['height'] -= trim
        y -= trim
        y += 8
    else:
        for role in ('before', 'after'):
            lines, size = _lines(role, clean[role], font, table, lines=1)
            for line in lines:
                add(role, line, size, ink=accent if role == 'after' else color,
                    italic=role == 'after', strike=role == 'before')
            y += 12
    if label := clean.get('label'):
        y += 8
        if look == 'number':
            for line in labels:
                add('label', line, label_size)
        else:
            lines, size = _lines('label', label.upper(), small_font, table,
                                 maximum=48, minimum=44, lines=2)
            for line in lines:
                # Small-caps tracking must fit as well as the untracked line breaks.
                size = _fit_single('label', line, small_font, table['width'] - 40,
                                   42, size, small_caps=True)
                add('proof' if look == 'hook' else 'label', line, size, caps=True)
    if y > table['bottom']:
        raise ValidationError('text', 'shorten text to fit the wide safe area')
    for block in blocks:
        check_box(Box(*(block[k] for k in ('x', 'y', 'width', 'height'))))
    check_face(blocks, face)
    inset = _inset(face, blocks) if clean.get('inset') else {}
    return LayoutPlan(tuple(blocks), side, **inset)
