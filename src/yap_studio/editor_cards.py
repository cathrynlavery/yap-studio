"""Fit editor cards and map their output frames to panel pixels."""

from __future__ import annotations

import json
import math

from yap_studio.card_project import CardProject
from yap_studio.cards import MIN_TEXT_PX, _fit_measured
from yap_studio.errors import ValidationError
from yap_studio.render.text_metrics import text_width_em


def timeline_variables(values: dict, project: CardProject, *, frames: int, fps: int,
                       width: int, height: int, font: str, mono: str) -> dict[str, object]:
    """Keep the fitted box fixed while drawing the project's visible clips."""
    size = MIN_TEXT_PX
    padding = 24
    timecode_px, timecode_row = 40, 48
    ruler_top = 20 + timecode_row + 10
    tracks_top = ruler_top + 14 + 8
    row_height, row_gap = 44, 8
    panel_height = (tracks_top + len(project.tracks) * row_height
                    + max(0, len(project.tracks) - 1) * row_gap + padding)
    timecode_width = math.ceil(text_width_em("0000:00:00", mono, 400) * timecode_px)
    title_width = width - 48 - timecode_width - 24
    title_px = _fit_measured("title", values["title"], title_width, font, 600, size)
    label_width = math.ceil(max(text_width_em(name, font, 400)
                                for name in ("Video", "B-roll", "Cards")) * size)
    lane_width = width - 48 - label_width - 20
    if lane_width < 48 or height < tracks_top + 3 * row_height + 2 * row_gap + padding:
        raise ValidationError("graphics.card_width_px", "use a wider card and a 360 px key band")
    if project.fps != fps or not 0 <= project.start_frame < project.total_frames:
        raise ValidationError("project", "use the card's frame rate and an output start frame")
    if project.start_frame + frames > project.total_frames:
        raise ValidationError("project", "keep the card inside the output frames")
    span = (project.total_frames if values["view"] == "whole" else
            min(project.total_frames, max(8 * fps, 3 * frames)))
    first = (0 if values["view"] == "whole" else
             max(0, min(project.total_frames - span, project.start_frame + frames / 2 - span / 2)))
    last = first + span

    def pixel(frame: float) -> float:
        return (frame - first) / span * lane_width

    tracks = []
    for row, track in enumerate(project.tracks):
        clips = [{"x_px": pixel(max(first, start)),
                  "width_px": pixel(min(last, end)) - pixel(max(first, start)),
                  "tint": ("video-a" if index % 2 == 0 else "video-b")
                  if track.name == "Video" else "broll" if track.name == "B-roll" else "cards"}
                 for index, (start, end) in enumerate(track.clips) if end > first and start < last]
        tracks.append({"name": track.name, "y_px": tracks_top + row * (row_height + row_gap),
                       "clips": clips})
    labels = []
    for frame in range(project.start_frame, project.start_frame + frames):
        seconds, subframe = divmod(frame, fps)
        minutes, seconds = divmod(seconds, 60)
        labels.append(f"{minutes:02}:{seconds:02}:{subframe:02}")
    ticks = [{"frame": frame, "x_px": pixel(frame), "major": frame % (5 * fps) == 0}
             for frame in range(math.ceil(first / fps) * fps, math.floor(last) + 1, fps)]
    return {
        "title": values["title"], "title_px": title_px, "text_px": size,
        "timecode_px": timecode_px, "panel_height_px": panel_height,
        "timecode_width_px": timecode_width, "label_width_px": label_width,
        "lane_left_px": 24 + label_width + 20, "lane_width_px": lane_width,
        "window_start_frame": first, "window_end_frame": last,
        "playhead": json.dumps({"start_px": pixel(project.start_frame),
                                "end_px": pixel(project.start_frame + frames - 1)}),
        "ruler_top_px": ruler_top, "playhead_top_px": ruler_top - 2,
        "playhead_height_px": panel_height - padding - (ruler_top - 2),
        "timecode_row_px": timecode_row, "timecodes": json.dumps(labels),
        "tracks": json.dumps(tracks), "ticks": json.dumps(ticks), "fps": fps,
    }


def storyboard_variables(values: dict, project: CardProject, *, frames: int, fps: int,
                         width: int, height: int) -> dict[str, object]:
    """Fit portrait thumbnails and schedule the active shot on the output clock."""
    count, gap, padding = values["count"], 16, 20
    capacity = max(0, (width - 2 * padding + gap) // (81 + 12 + gap))
    if count > capacity:
        raise ValidationError("slots.count", f"lower count to {capacity} or use a wider card")
    number_row = 38 if values["numbers"] == "show" else 0
    available = height - 2 * padding - number_row - 12
    thumb_width = min(math.floor(available * 9 / 16),
                      (width - 2 * padding - (count - 1) * gap) // count - 12)
    thumb_width = thumb_width // 9 * 9
    if thumb_width < 81:
        raise ValidationError("slots.count", "use a taller key band to fit readable thumbnails")
    thumb_height = thumb_width * 16 // 9
    if not project.shots:
        raise ValidationError("project.shots", "add named segments to the cut before rendering")
    if project.fps != fps or not 0 <= project.start_frame < project.start_frame + frames <= (
        project.total_frames
    ):
        raise ValidationError("project", "keep the card inside the output at its frame rate")
    on_screen = [index for index, shot in enumerate(project.shots)
                 if shot.end_frame > project.start_frame
                 and shot.start_frame < project.start_frame + frames]
    if not on_screen:
        raise ValidationError("project.shots", "cover the card's output frames with shots")
    if len(on_screen) > count:
        raise ValidationError("slots.count", f"shorten the card to show at most {count} shots")
    shown = min(count, len(project.shots))
    first = max(0, on_screen[-1] - shown + 1,
                min(on_screen[0] - shown // 2, len(project.shots) - shown))
    selected = project.shots[first:first + shown]
    row_width = shown * (thumb_width + 12) + (shown - 1) * gap
    left = padding + 6
    top = (height - thumb_height - number_row) // 2
    thumbnails = [{"index": first + index, "label": shot.label,
                   "src": f"shot-{index + 1:02}.jpg" if shot.source is not None else "",
                   "x_px": left + index * (thumb_width + 12 + gap)}
                  for index, shot in enumerate(selected)]
    changes = [{"frame": 0, "at_s": 0, "index": on_screen[0] - first}]
    for index in on_screen[1:]:
        frame = project.shots[index].start_frame - project.start_frame
        changes.append({"frame": frame, "at_s": frame / fps, "index": index - first})
    duration = frames / fps
    ring = [{"at_s": 0, "x_px": thumbnails[changes[0]["index"]]["x_px"] - 4}]
    scales = [[{"at_s": 0, "scale": 1.06 if index == changes[0]["index"] else 1}]
              for index in range(shown)]
    for index, change in enumerate(changes[1:], 1):
        at = change["at_s"]
        end = min(at + 0.2, changes[index + 1]["at_s"] if index + 1 < len(changes) else duration)
        ring.extend([{"at_s": at, "x_px": ring[-1]["x_px"]},
                     {"at_s": end, "x_px": thumbnails[change["index"]]["x_px"] - 4}])
        for number, motion in enumerate(scales):
            motion.extend([{"at_s": at, "scale": motion[-1]["scale"]},
                           {"at_s": end, "scale": 1.06 if number == change["index"] else 1}])
    ring.append({"at_s": duration, "x_px": ring[-1]["x_px"]})
    for number, (thumbnail, motion) in enumerate(zip(thumbnails, scales, strict=True)):
        motion.append({"at_s": duration, "scale": motion[-1]["scale"]})
        poses = _frame_poses(motion, "scale", frames, fps)
        for frame, pose in enumerate(poses):
            scale = pose.pop("scale")
            active = next(change["index"] for change in reversed(changes)
                          if change["frame"] <= min(frame, frames - 1))
            pose.update(width_px=round(thumb_width * scale, 6),
                        height_px=round(thumb_height * scale, 6),
                        left_px=round(thumb_width * (1 - scale) / 2, 6),
                        top_px=round(thumb_height * (1 - scale) / 2, 6),
                        radius_px=round(6 * scale, 6), active=number == active)
        thumbnail["scale_keyframes"] = poses
    return {"numbers": values["numbers"], "number_px": MIN_TEXT_PX,
            "panel_height_px": height, "storyboard_width_px": row_width + 2 * padding,
            "thumb_width_px": thumb_width,
            "thumb_height_px": thumb_height, "thumb_top_px": top,
            "thumbnails": json.dumps(thumbnails), "changes": json.dumps(changes),
            "ring_keyframes": json.dumps(_frame_poses(ring, "x_px", frames, fps)),
            "motion_duration_s": duration}


def _frame_poses(points: list[dict], key: str, frames: int, fps: int) -> list[dict]:
    """Place fixed poses half a frame early so seeks avoid interpolation rounding."""
    poses, index = [], 0
    for frame in range(frames):
        at = frame / fps
        while index + 1 < len(points) - 1 and points[index + 1]["at_s"] <= at:
            index += 1
        first, last = points[index], points[index + 1]
        span = last["at_s"] - first["at_s"]
        progress = (at - first["at_s"]) / span if span else 1
        value = first[key] + (last[key] - first[key]) * progress
        poses.append({"at_s": max(0, (frame - 0.5) / fps), key: round(value, 6)})
    poses.append({"at_s": frames / fps, key: poses[-1][key]})
    return poses
