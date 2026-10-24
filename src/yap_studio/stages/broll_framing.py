"""Wide insert framing using the engine's existing face measurements."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from yap_studio.faces.clearance import head_box
from yap_studio.faces.model import Box

if TYPE_CHECKING:
    from pathlib import Path

    from yap_studio.ffmpeg import MediaInfo
    from yap_studio.stages.assemble_picture import _Source
    from yap_studio.stages.timeline import ResolvedElement


def insert_fit(media: dict, info: MediaInfo, size: tuple[int, int]) -> str:
    """Show portrait and square sources whole on a wide delivery canvas."""
    fit = media.get('fit', 'cover')
    if size[0] > size[1] and fit == 'cover' and info.width <= info.height:
        return 'contain'
    return fit


def wide_insert(item: ResolvedElement, source: _Source, root: Path,
                size: tuple[int, int]) -> ResolvedElement:
    """Keep a measured face inside an automatic cover crop; contain if it cannot fit."""
    media = item.payload['media']
    fit = insert_fit(media, source.info, size)
    width, height = source.info.width, source.info.height
    if fit == 'cover' and (width * size[1] != height * size[0]
                           or item.payload.get('motion') == 'push'):
        if source.image:
            from yap_studio.gallery.still import default_landmark_detector

            detector = default_landmark_detector()
            if detector is None:
                fit = 'contain'
            faces = (detector.measure([source.path])[0] if detector is not None else [])
            boxes = [Box(face.x, face.y, face.w, face.h) for face in faces]
        else:
            from yap_studio.faces.project import cut_faces

            faces = cut_faces(root, source.path)
            if not faces.frames:
                fit = 'contain'
            first = math.floor(media.get('in_s', 0) * faces.video.fps)
            last = math.ceil(first + (item.end_frame - item.start_frame) / 30
                             * media.get('speed', 1) * faces.video.fps)
            boxes = [face.box for frame in faces.frames[first:last] for face in frame]
        if boxes:
            if item.payload.get('motion') == 'push':
                item = replace(item, payload={**item.payload, 'motion': 'still'})
            boxes = [head_box(box) for box in boxes]
            left, top = min(box.x for box in boxes), min(box.y for box in boxes)
            right, bottom = max(box.x1 for box in boxes), max(box.y1 for box in boxes)
            crop_w = min(width, height * size[0] / size[1])
            crop_h = min(height, width * size[1] / size[0])
            w, h = max(2, int(crop_w) // 2 * 2), max(2, int(crop_h) // 2 * 2)
            if right - left > w or bottom - top > h:
                fit = 'contain'
            else:
                x = max(0, min(width - w, left, max(right - w, (width - w) / 2)))
                y = max(0, min(height - h, top, max(bottom - h, (height - h) / 2)))
                media = {**media, 'crop': [round(x), round(y), w, h]}
                fit = 'crop'
    return replace(item, payload={**item.payload, 'media': {**media, 'fit': fit}})
