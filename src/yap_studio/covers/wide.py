"""Independent 1280x720 layout tables and YouTube preview checks."""
from __future__ import annotations

import math
import re

import numpy as np

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

MINIMUM_GLYPH_PX = 8
# Search no lower than the source height needed at browse size; ink is checked after rendering.
MINIMUM_TYPE_PX = math.ceil(MINIMUM_GLYPH_PX * 720 / 94)

# Each look owns its column and type hierarchy; no portrait dimensions are scaled.
WIDE_LAYOUTS = {
    'photo': {'left': 56, 'right': 656, 'width': 568, 'top': 112, 'bottom': 592,
              'maximum': 146, 'minimum': MINIMUM_TYPE_PX, 'lines': 3},
    'number': {'left': 56, 'right': 656, 'width': 568, 'top': 112, 'bottom': 592,
               'maximum': 300, 'minimum': 160, 'lines': 1},
    'before-after': {'left': 56, 'right': 656, 'width': 568, 'top': 112, 'bottom': 592,
                     'maximum': 142, 'minimum': MINIMUM_TYPE_PX, 'lines': 2},
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


def check_face(blocks: list[dict] | tuple[dict, ...], face: tuple, *,
               require_full_head: bool = True) -> dict:
    """Reserve the head and a little breathing room, including hair."""
    head = head_box(Box(*face))
    if require_full_head and (head.x < -1e-6 or head.y < -1e-6 or
                              head.x1 > 1280 + 1e-6 or head.y1 > 720 + 1e-6):
        raise ValidationError('face', 'face or head clipped by canvas; choose another frame')
    clearance = Box(head.x - 24, head.y - 24, head.w + 48, head.h + 48)
    if _overlap(head, BADGE):
        raise ValidationError('face', 'face overlaps the duration badge; choose another frame')
    for block in blocks:
        if _overlap(clearance, Box(*(block[k] for k in ('x', 'y', 'width', 'height')))):
            error = ValidationError('face', 'headline needs room beside every face; '
                                    'choose another frame')
            error.suggested_action = 'Shorten the text or choose another speaker frame.'
            raise error
    return {'passed': True, 'head_box': [head.x, head.y, head.w, head.h]}


def check_faces(blocks: list[dict] | tuple[dict, ...], face: tuple,
                face_boxes: tuple = ()) -> dict:
    """Keep copy clear of every visible head, including partially cropped guests."""
    result = check_face(blocks, face)
    result['heads'] = [check_face(blocks, box, require_full_head=False)['head_box']
                       for box in (face_boxes or (face,))]
    return result


def _preview(pixels: np.ndarray) -> np.ndarray:
    """Area-average pixels, retaining fractional coverage at the browse size."""
    for axis, size in ((1, 168), (0, 94)):
        source = np.moveaxis(pixels, axis, 0)
        edges = np.linspace(0, len(source), size + 1)
        indices = np.minimum(edges.astype(int), len(source) - 1)
        sums = np.concatenate((np.zeros_like(source[:1]), np.cumsum(source, axis=0)))
        integral = sums[indices] + source[indices] * (
            edges - indices).reshape((-1,) + (1,) * (source.ndim - 1))
        pixels = np.moveaxis(np.diff(integral, axis=0) / (len(source) / size), 0, axis)
    return pixels


def check_legibility(blocks: list[dict], text: np.ndarray, backdrop: np.ndarray, *,
                     look: str = 'photo', accent_text: np.ndarray | None = None) -> dict:
    """Require measured glyphs to survive reduction to the 168x94 browse size."""
    alpha = text[:, :, 3:4].astype(float) / 255
    back = backdrop[:, :, :3].astype(float) / 255
    visible = _preview(text[:, :, :3] / 255 * alpha + back * (1 - alpha))
    ground = _preview(back)
    coverage = _preview(alpha)[:, :, 0]

    def linear(rgb):
        return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)

    weights = np.array([.2126, .7152, .0722])
    a, b = linear(visible) @ weights, linear(ground) @ weights
    contrast = (np.maximum(a, b) + .05) / (np.minimum(a, b) + .05)
    rows = []
    for block in blocks:
        height = (block['measured_box'][3] - block['measured_box'][1]) * 94 / 720
        # An isolated italic phrase can have only lowercase x-height, without line capitals.
        secondary = block['role'] in {'kicker', 'label', 'proof', 'accent'}
        minimum = (4 if secondary and not (look == 'number' and block['role'] == 'label')
                   else MINIMUM_GLYPH_PX)
        if height < minimum:
            error = ValidationError('text', f"{block['role']} glyph height {height:.1f} px at "
                                    f'168x94 is below {minimum}')
            error.suggested_action = 'Try --look before-after or --look number.'
            raise error
        x, y, w, h = block['box']
        x0, y0 = math.floor(x * 168 / 1280), math.floor(y * 94 / 720)
        x1, y1 = math.ceil((x + w) * 168 / 1280), math.ceil((y + h) * 94 / 720)
        mask = coverage[y0:y1, x0:x1]
        strokes = mask >= .5
        yy, _ = np.nonzero(strokes)
        if not len(yy) or yy.max() - yy.min() + 1 < minimum:
            error = ValidationError('text', f"{block['role']} strokes vanish at 168x94")
            error.suggested_action = (
                'Choose a heavier font or try --look before-after or --look number.')
            raise error
        # Measure stroke cores; antialiased edges blend into the backdrop by design.
        cores = mask >= max(.5, .9 * float(mask.max()))
        p10 = float(np.percentile(contrast[y0:y1, x0:x1][cores], 10))
        if p10 < 3:
            raise ValidationError('text', f"{block['role']} contrast {p10:.2f}:1 at 168x94 "
                                  'is below 3:1; choose another frame or heavier font')
        rows.append({'role': block['role'], 'glyph_height_px': round(height, 2),
                     'minimum': minimum, 'surviving_pixels': int(strokes.sum()),
                     'contrast_p10': round(p10, 3)})
        if block.get('accent_color'):
            try:
                if accent_text is None:
                    raise ValidationError('accent', 'accent did not render')
                yy, xx = np.nonzero(accent_text[y:y + h, x:x + w, 3] >= 128)
                if not len(xx):
                    raise ValidationError('accent', 'accent did not render')
                measured = [x + int(xx.min()), y + int(yy.min()),
                            x + int(xx.max()) + 1, y + int(yy.max()) + 1]
                result = check_legibility([{**block, 'role': 'accent', 'accent_color': None,
                                            'measured_box': measured}],
                                           accent_text, backdrop, look=look)
                rows.extend(result['blocks'])
            except ValidationError as error:
                error.suggested_action = 'Mark a shorter accent phrase or choose a heavier font.'
                raise
    return {'passed': True, 'preview': [168, 94], 'blocks': rows}


def _lines(role: str, text: str, font: str, table: dict, *, maximum: int | None = None,
           minimum: int | None = None, lines: int | None = None,
           height: int = 720, forbidden_breaks: tuple[int, ...] = ()) -> tuple[list[str], int]:
    for size in range(maximum or table['maximum'], (minimum or table['minimum']) - 1, -1):
        try:
            result = balance_lines(text.split(), lambda line, size=size: _width(line, font, size),
                                   table['width'] - 40, lines or table['lines'],
                                   forbidden_breaks=forbidden_breaks)
            if len(result) * math.ceil(size * 1.12) <= height:
                return result, size
        except ValidationError:
            continue
    error = ValidationError(role, 'text cannot fit the thumbnail without an orphan word')
    error.suggested_action = (
        'Each line needs two or more words; try --look before-after or --look number.'
        + (' Mark a shorter accent phrase.' if forbidden_breaks else ''))
    raise error


def _inset(face: tuple, blocks: list[dict], face_boxes: tuple) -> dict:
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
                    check_faces([{'x': x, 'y': y, 'width': rw, 'height': rh}], face, face_boxes)
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
    error = ValidationError('inset', 'inset cannot clear face, text and badge; omit inset')
    error.suggested_action = 'Omit --inset or choose another speaker frame.'
    raise error


def plan_wide(look: str, clean: dict, style: dict, face: tuple | None,
              face_boxes: tuple = ()) -> LayoutPlan:
    """Lay copy on the open side of an unchanged photo."""
    if look not in WIDE_LAYOUTS:
        error = ValidationError('look', 'horizontal thumbnails use photo, number or before-after')
        error.suggested_action = 'Choose --look photo, --look number or --look before-after.'
        raise error
    if face is None:
        raise ValidationError('face', 'wide thumbnail needs a face; choose another frame')
    table = WIDE_LAYOUTS[look]
    plain, phrase, first, last = (_accent_span(clean['headline'])
                                if look == 'photo' else ('', None, -1, -1))
    if bleep.enabled(style):
        vocabulary = bleep.listed(style['music'])
        plain = bleep.mask_text(plain, vocabulary)
        clean = {k: bleep.mask_text(v, vocabulary) if k not in {'headline', 'inset'} else v
                 for k, v in clean.items()}
    side = 'left' if face[0] + face[2] / 2 < 640 else 'right'
    x = table['right'] if side == 'left' else table['left']
    font, small_font = style['hook']['font'], style['graphics']['font']
    color = style['captions']['text_color']
    accent = readable_accent(style, '#101010') if phrase or look in {
        'number', 'before-after'} else color
    y = table['top']
    blocks = []

    def add(role, text, size, *, ink=color, caps=False, italic=False, strike=False):
        nonlocal y
        block = _block(role, text, x, y, table['width'], size, ink,
                       small_font if caps else font, small_caps=caps, strike=strike)
        if look == 'number' and font == 'Gelasio':
            # Gelasio's deeper baseline and descenders need the full line box.
            block['height'] = math.ceil(size * 1.2)
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
    if look == 'photo':
        first = len(re.sub(r'\s+', ' ', plain[:first])) if phrase else -1
        last = len(re.sub(r'\s+', ' ', plain[:last])) if phrase else -1
        plain = ' '.join(plain.split())
        forbidden = tuple(i for i, word in enumerate(re.finditer(r'\S+', plain))
                          if first < word.start() < last)
        lines, size = _lines('headline', plain, font, table,
                             forbidden_breaks=forbidden,
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
                add('label', line, size, caps=True)
    if y > table['bottom']:
        raise ValidationError('text', 'shorten text to fit the wide safe area')
    for block in blocks:
        check_box(Box(*(block[k] for k in ('x', 'y', 'width', 'height'))))
    check_faces(blocks, face, face_boxes)
    inset = _inset(face, blocks, face_boxes) if clean.get('inset') else {}
    return LayoutPlan(tuple(blocks), side, **inset)
