"""Fit delete chips and tiles and sample their motion on the card clock."""

from __future__ import annotations

import json
import math

from yap_studio.cards import MIN_TEXT_PX, _cue_window
from yap_studio.errors import ValidationError
from yap_studio.render.hyperframes import layer_duration_s
from yap_studio.render.text_metrics import text_width_em

_BASE_LABEL_PX = 34
_MAX_LABEL_PX = 56


def removal_cues(slots: dict, duration: float) -> list[float]:
    """Use one removal clock for the composition and its item sounds."""
    wiggle = slots.get("wiggle_at", 0.3)
    latest = duration - min(0.3, duration * 0.15) - 0.95
    if not 0 <= wiggle <= latest:
        raise ValidationError("slots.wiggle_at", f"use seconds within 0..{max(0, latest):g}")
    count, explicit = len(slots["remove"]), slots.get("cues", [])
    _cue_window({"cues": explicit}, duration, count, reserve=0.95, lower=wiggle, unit="removal")
    cues = []
    for index in range(count):
        at = (explicit[index] if index < len(explicit) else
              cues[-1] + 0.5 if cues else wiggle + 0.6)
        cues.append(at)
    _cue_window({"cues": cues}, duration, count, reserve=0.95, lower=wiggle, unit="removal")
    return cues


def jiggle_variables(values: dict, *, frames: int, fps: int, width: int,
                     height: int, font: str) -> dict[str, object]:
    """Measure labels, centre every remaining row and retain frame-exact poses."""
    labels, remove, images = values["items"], values["remove"], values["images"]
    count = len(labels)
    if any(position != int(position) for position in remove):
        raise ValidationError("slots.remove", "use whole item positions")
    remove = [int(position) for position in remove]
    if any(not 1 <= position <= count for position in remove):
        raise ValidationError("slots.remove", f"use item positions within 1..{count}")
    if len(set(remove)) != len(remove):
        raise ValidationError("slots.remove", "list each item position once")
    if len(remove) >= count:
        raise ValidationError("slots.remove", "keep at least one item")
    if images and len(images) != count:
        raise ValidationError("slots.images", "use the same count as items or leave images empty")
    cues = removal_cues(values, layer_duration_s(frames, fps))
    inner = width - 48
    units = [text_width_em(label, font, 600) for label in labels]
    for size in range(_MAX_LABEL_PX, MIN_TEXT_PX - 1, -1):
        text_scale = size / _BASE_LABEL_PX
        gap, pad_x, pad_y = (math.ceil(value * text_scale) for value in (16, 18, 10))
        label_gap = math.ceil(8 * text_scale)
        picture = (min(height - 48 - label_gap - math.ceil(size * 1.2),
                       (inner - (count - 1) * gap) // count) if images else 0)
        if images and picture < 80:
            continue
        text_room = picture if images else inner - 2 * pad_x
        if max(units) * size > text_room:
            continue
        widths = [picture if images else math.ceil(unit * size) + 2 * pad_x for unit in units]
        item_height = (picture + label_gap + math.ceil(size * 1.2) if images else
                       math.ceil(size * 1.2) + 2 * pad_y)
        rows = _rows(list(range(1, count + 1)), widths, inner, gap)
        panel_height = len(rows) * item_height + (len(rows) - 1) * gap + 48
        if panel_height <= height:
            break
    else:
        if images and picture < 80:
            raise ValidationError("slots.items", "remove items to fit readable image tiles")
        if max(units) * MIN_TEXT_PX > text_room:
            widest = max(text_width_em(char, font, 600) for label in labels for char in label)
            capacity = max(0, math.floor(text_room / (widest * MIN_TEXT_PX)))
            raise ValidationError("slots.items",
                                  f"shorten labels to {capacity} characters to fit at 30 px")
        raise ValidationError("slots.items", "remove items to fit the panel at 30 px")
    panel_width = max(sum(widths[number - 1] for number in row) + (len(row) - 1) * gap
                      for row in rows) + 48
    remaining, layouts = list(range(1, count + 1)), []
    for removed in [None, *remove]:
        if removed is not None:
            remaining.remove(removed)
        rows = _rows(remaining, widths, panel_width - 48, gap)
        layout_width = max(sum(widths[number - 1] for number in row) + (len(row) - 1) * gap
                           for row in rows) + 48
        layout_height = len(rows) * item_height + (len(rows) - 1) * gap + 48
        top = 24
        positions = {}
        for row in rows:
            row_width = sum(widths[number - 1] for number in row) + (len(row) - 1) * gap
            left = (layout_width - row_width) / 2
            for number in row:
                positions[str(number)] = {"x_px": left, "y_px": top}
                left += widths[number - 1] + gap
            top += item_height + gap
        layouts.append({"remaining": remaining.copy(), "positions": positions,
                        "width_px": layout_width, "height_px": layout_height})
    panel_points = [{key: layout[key] for key in ("width_px", "height_px")} for layout in layouts]
    panel_poses = [{"at_s": max(0, (frame - 0.5) / fps),
                    **{key: round(value, 6) for key, value in
                       _reflow(frame / fps, panel_points, cues).items()}}
                   for frame in range(frames)]
    panel_poses.append({**panel_poses[-1], "at_s": frames / fps})
    settle = cues[-1] + 0.3
    items = []
    for index, label in enumerate(labels):
        number = index + 1
        removed_at = cues[remove.index(number)] if number in remove else math.inf
        poses = []
        for frame in range(frames):
            time = frame / fps
            position = _position(number, min(time, removed_at), layouts, cues)
            out = _progress(time, removed_at, 0.25)
            scale = 1 - 0.8 * out
            badge = _progress(time, values["wiggle_at"], 0.15)
            badge *= 1 - _progress(time, settle, 0.2)
            badge_scale = 1 - 0.25 * _progress(time, removed_at, 0.1)
            amplitude = (1 - _progress(time, settle, 0.2)) if time >= values["wiggle_at"] else 0
            angle = 2 * math.sin((time - values["wiggle_at"]) * math.tau / 0.25 + index * 0.7)
            poses.append({
                "at_s": max(0, (frame - 0.5) / fps),
                "x_px": position["x_px"], "y_px": position["y_px"],
                "angle": angle * amplitude, "opacity": 1 - out,
                "width_px": widths[index] * scale, "height_px": item_height * scale,
                "inset_x_px": widths[index] * (1 - scale) / 2,
                "inset_y_px": item_height * (1 - scale) / 2,
                "font_px": size * scale, "picture_px": picture * scale,
                "label_top_px": (picture + label_gap) * scale if images else 0,
                "label_height_px": math.ceil(size * 1.2) * scale if images else item_height * scale,
                "badge_opacity": badge,
                "badge_px": 26 * text_scale * badge * badge_scale * scale,
                "badge_dip_px": 4 * text_scale * _progress(time, removed_at, 0.1),
            })
        poses = [{key: value if key == "at_s" else round(value, 6)
                  for key, value in pose.items()} for pose in poses]
        poses.append({**poses[-1], "at_s": frames / fps})
        items.append({"label": label, "width_px": widths[index], "height_px": item_height,
                      "picture_px": picture, "src": f"item-{number}{images[index].suffix.lower()}"
                      if images else "", "keyframes": poses})
    return {"items": json.dumps(items), "layouts": json.dumps(layouts), "cues": json.dumps(cues),
            "wiggle_at": values["wiggle_at"], "settle_at_s": settle, "label_px": size,
            "badge_offset_px": math.ceil(8 * text_scale),
            "badge_stroke_px": max(1, round(2 * text_scale)),
            "delete_panel_width_px": panel_width, "panel_height_px": panel_height,
            "panel_keyframes": json.dumps(panel_poses),
            "motion_duration_s": frames / fps}


def _rows(remaining: list[int], widths: list[int], available: int, gap: int) -> list[list[int]]:
    rows, current, used = [], [], 0
    for number in remaining:
        width = widths[number - 1]
        if current and used + gap + width > available:
            rows.append(current)
            current, used = [], 0
        used += (gap if current else 0) + width
        current.append(number)
    if current:
        rows.append(current)
    return rows


def _progress(time: float, start: float, duration: float) -> float:
    return max(0, min(1, (time - start) / duration))


def _position(number: int, time: float, layouts: list[dict], cues: list[float]) -> dict:
    return _reflow(time, [layout["positions"].get(str(number)) for layout in layouts], cues)


def _reflow(time: float, points: list[dict | None], cues: list[float]) -> dict:
    position = points[0]
    for cue, target in zip(cues, points[1:], strict=True):
        if time < cue + 0.25:
            break
        if target is None:
            break
        progress = _progress(time, cue + 0.25, 0.3)
        progress = progress * progress * (3 - 2 * progress)
        position = {key: position[key] + (target[key] - position[key]) * progress
                    for key in position}
    return position
