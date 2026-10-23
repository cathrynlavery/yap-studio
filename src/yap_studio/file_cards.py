"""Fit file cards and sample their fall on the output frame clock."""

from __future__ import annotations

import json
import math
import re

from yap_studio import ffmpeg
from yap_studio.cards import MIN_TEXT_PX, PAD_BOTTOM, PAD_TOP, PAD_X
from yap_studio.errors import ValidationError
from yap_studio.render import hyperframes
from yap_studio.render.text_metrics import text_width_em

FILE_DROP_LAND_S = 0.38
FILE_DROP_SETTLE_S = 0.7
FILE_DROP_REST_S = 0.4
FILE_DROP_BADGE_POP_PX = 6


def _layout(info: ffmpeg.MediaInfo, *, width: int, height: int, name_units: float,
            duration_units: float, badge_units: float, badge: str, play: bool) -> dict | None:
    pad, gap = 24, 28
    poster_height = height - 2 * pad
    poster_width = math.floor(poster_height * info.width / info.height)
    poster_width = min(poster_width, math.floor((width - 2 * pad - gap) * 0.5))
    poster_height = max(1, round(poster_width * info.height / info.width))
    if play:
        poster_width = max(2, poster_width // 2 * 2)
        poster_height = max(2, poster_height // 2 * 2)
    room = width - 2 * pad - gap - poster_width
    name_height = height - 2 * pad - 12 - 36 - (14 + 44 if badge else 0)
    name_px = min(48, math.floor(room / max(name_units, 0.01)), math.floor(name_height / 1.2))
    if name_px < MIN_TEXT_PX:
        return None
    duration_px = 30
    badge_px = 30
    badge_width = math.ceil(badge_units * badge_px) + 24 if badge else 0
    if duration_units * duration_px > room or badge_width > room:
        return None
    text_width = math.ceil(max(name_units * name_px,
                               duration_units * duration_px,
                               badge_width))
    panel_width = 2 * pad + poster_width + gap + text_width
    text_height = math.ceil(name_px * 1.2) + 12 + 36 + (14 + 44 if badge else 0)
    panel_height = max(poster_height, text_height) + 2 * pad
    if play:
        panel_width = math.ceil(panel_width / 2) * 2
        panel_height = math.ceil(panel_height / 2) * 2
    return {"name_px": name_px, "duration_px": duration_px, "badge_px": badge_px,
            "file_panel_width_px": panel_width, "panel_height_px": panel_height,
            "poster_width_px": poster_width, "poster_height_px": poster_height,
            "text_left_px": pad + poster_width + gap, "text_width_px": text_width}


def _motion(layout: dict, *, drop: float, frames: int, fps: int) -> list[dict]:
    panel_width, panel_height = layout["file_panel_width_px"], layout["panel_height_px"]
    anchors = [(0, -4, -6, 1.12, 1.12), (FILE_DROP_LAND_S, 0, 1, 1.025, 0.94),
               (0.46, -3, -1, 0.99, 1.025), (0.56, 0, 0.4, 1.005, 0.99),
               (FILE_DROP_SETTLE_S, 0, 0, 1, 1)]
    motion = []
    motion_duration = frames / fps

    def px(value: float) -> float:
        return round(value, 3)

    for frame in range(frames + 1):
        elapsed = max(0, frame / fps - drop)
        pose = anchors[-1][1:]
        for first, last in zip(anchors, anchors[1:], strict=False):
            if elapsed <= last[0]:
                fraction = max(0, (elapsed - first[0]) / (last[0] - first[0]))
                if first[0] == 0:
                    fraction *= fraction
                pose = tuple(a + (b - a) * fraction
                             for a, b in zip(first[1:], last[1:], strict=True))
                break
        y, angle, sx, sy = pose
        text_scale = min(sx, sy)
        w, h = round(panel_width * sx, 3), round(panel_height * sy, 3)
        shadow = min(1, elapsed / FILE_DROP_LAND_S)
        shadow_width, shadow_height = w * (0.7 + 0.3 * shadow), h * (0.7 + 0.3 * shadow)
        motion.append({"at_s": max(0, (frame - 0.5) / fps) if frame < frames else motion_duration,
                       "left_px": round((panel_width - w) / 2, 3),
                       "top_px": round(y + (panel_height - h) / 2, 3),
                       "width_px": w, "height_px": h, "angle": round(angle, 4),
                       "opacity": round(min(1, max(0, (frame / fps - drop) / 0.1)), 4),
                       "shadow": round(shadow, 4),
                       "shadow_width_px": round(shadow_width, 3),
                       "shadow_height_px": round(shadow_height, 3),
                       "shadow_left_px": round((w - shadow_width) / 2, 3),
                       "shadow_top_px": round((h - shadow_height) / 2, 3),
                       "poster_left_px": px(24 * sx),
                       "poster_width_px": px(layout["poster_width_px"] * sx),
                       "poster_height_px": px(layout["poster_height_px"] * sy),
                       "text_left_px": px(layout["text_left_px"] * sx),
                       "text_width_px": px(layout["text_width_px"] * sx),
                       "name_px": px(max(MIN_TEXT_PX, layout["name_px"] * text_scale)),
                       "duration_px": px(max(MIN_TEXT_PX, layout["duration_px"] * text_scale)),
                       "badge_px": px(max(MIN_TEXT_PX, layout["badge_px"] * text_scale)),
                       "name_line_px": px(layout["name_px"] * 1.2 * sy),
                       "duration_line_px": px(36 * sy), "duration_gap_px": px(12 * sy),
                       "badge_line_px": px(36 * sy), "badge_gap_px": px(14 * sy),
                       "badge_pad_x_px": px(12 * sx), "badge_pad_y_px": px(4 * sy)})
    return motion


def _shadows(shadow: str) -> list[tuple[float, float, float]]:
    number = r"(-?\d+(?:\.\d+)?)(?:px)?"
    values = re.findall(r"(?:^|,\s*)" + number + r"\s+" + number
                        + r"(?:\s+" + number + r")?(?:\s+" + number + r")?", shadow)
    # Two blur radii also reserve the faint edge of the browser's shadow raster.
    return [(float(x), float(y), 2 * float(blur or 0) + float(spread or 0))
            for x, y, blur, spread in values]


def _motion_bounds(motion: list[dict], layout: dict, *, width: int, height: int,
                   shadows: list[tuple[float, float, float]], badge_units: float,
                   badge: str) -> list[float]:
    origin_x = (width + 2 * PAD_X - layout["file_panel_width_px"]) / 2
    origin_y = PAD_TOP + (height - layout["panel_height_px"]) / 2
    bounds = [math.inf, math.inf, -math.inf, -math.inf]
    for frame in motion:
        w, h = frame["width_px"], frame["height_px"]
        boxes = [(0, 0, w, h)]
        for x, y, spread in shadows:
            left, top = frame["shadow_left_px"] + x - spread, frame["shadow_top_px"] + y - spread
            boxes.append((left, top, left + frame["shadow_width_px"] + 2 * spread,
                          top + frame["shadow_height_px"] + 2 * spread))
        if badge:
            badge_height = frame["badge_line_px"] + 2 * frame["badge_pad_y_px"]
            text_height = (frame["name_line_px"] + frame["duration_gap_px"]
                           + frame["duration_line_px"] + frame["badge_gap_px"] + badge_height)
            bottom = (h + text_height) / 2 + FILE_DROP_BADGE_POP_PX
            boxes.append((frame["text_left_px"], bottom - badge_height,
                          frame["text_left_px"] + badge_units * frame["badge_px"]
                          + 2 * frame["badge_pad_x_px"], bottom))
        angle = math.radians(frame["angle"])
        cosine, sine = math.cos(angle), math.sin(angle)
        for left, top, right, bottom in boxes:
            for x, y in ((left, top), (right, top), (right, bottom), (left, bottom)):
                dx, dy = x - w / 2, y - h / 2
                x = origin_x + frame["left_px"] + w / 2 + dx * cosine - dy * sine
                y = origin_y + frame["top_px"] + h / 2 + dx * sine + dy * cosine
                bounds = [min(bounds[0], x), min(bounds[1], y),
                          max(bounds[2], x), max(bounds[3], y)]
    return bounds


def file_variables(values: dict, info: ffmpeg.MediaInfo, *, frames: int, fps: int, width: int,
                   height: int, font: str, mono: str, shadow: str) -> dict:
    """Fit every moving pixel before preparing or rendering the card."""
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - FILE_DROP_SETTLE_S - FILE_DROP_REST_S
    drop = values["drop_at"]
    if not 0 <= drop <= latest:
        raise ValidationError("slots.drop_at", f"use seconds within 0..{max(0, latest):g}; "
                              "leave 0.7 s to settle and 0.4 s of rest before the fade-out")
    name = values["name"] or values["clip"].name
    badge = values["badge"].upper()
    seconds = max(1, math.floor(info.video_duration_s + 0.5))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_text = (f"{hours}:{minutes:02}:{seconds:02}" if hours else
                     f"{minutes}:{seconds:02}")
    name_units = text_width_em(name, font, 600)
    duration_units = text_width_em(duration_text, mono, 400)
    badge_units = text_width_em(badge, font, 600)
    shadows = _shadows(shadow)
    for available in range(height, 47, -1):
        layout = _layout(info, width=width, height=available, name_units=name_units,
                          duration_units=duration_units, badge_units=badge_units, badge=badge,
                          play=values["playback"] == "play")
        if layout is None:
            continue
        motion = _motion(layout, drop=drop, frames=frames, fps=fps)
        bounds = _motion_bounds(motion, layout, width=width, height=height,
                                shadows=shadows, badge_units=badge_units, badge=badge)
        if (bounds[0] >= 2 and bounds[1] >= 2 and bounds[2] <= width + 2 * PAD_X - 2
                and bounds[3] <= height + PAD_TOP + PAD_BOTTOM - 2):
            return {**layout, "name": name, "badge": badge, "duration_text": duration_text,
                    "poster_src": "poster.jpg", "drop_at": drop,
                    "land_at_s": drop + FILE_DROP_LAND_S, "motion_duration_s": frames / fps,
                    "motion": json.dumps(motion), "play_window": "{}",
                    "motion_bounds": json.dumps([round(value, 6) for value in bounds])}
    raise ValidationError("slots.name", "set name to a shorter label")
