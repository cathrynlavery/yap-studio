"""Draw five-second plan samples over the padded plate for pilot review."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/camera3d_plan_sheet.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/camera3d_plan_sheet.py"
    )

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg
from yap_studio.render.camera3d import (
    _head_at,
    _metrics,
    _project,
    project_plate_corners,
    project_real_picture_box,
)

THUMB_W, THUMB_H = 216, 384


def _frame(plate: Path, seconds: float) -> np.ndarray:
    result = ffmpeg.run(
        ["ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(plate),
         "-frames:v", "1", "-vf", "scale=390:660", "-pix_fmt", "rgb24",
         "-f", "rawvideo", "pipe:1"],
        timeout_s=90, binary=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(660, 390, 3)


def _outline(image: np.ndarray, box: list[float], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = [round(value * THUMB_W / 1080) for value in box]
    x0, x1 = max(0, x0), min(THUMB_W - 1, x1)
    y0, y1 = max(0, y0), min(THUMB_H - 1, y1)
    if x0 < x1 and y0 < y1:
        image[y0:y0 + 2, x0:x1] = color
        image[y1 - 2:y1, x0:x1] = color
        image[y0:y1, x0:x0 + 2] = color
        image[y0:y1, x1 - 2:x1] = color


def _picture_edge(image: np.ndarray, box: list[float]) -> None:
    """Draw only the real-picture edges that actually enter the output frame."""
    x0, x1 = (round(box[i] * THUMB_W / 1080) for i in (0, 2))
    y0, y1 = (round(box[i] * THUMB_H / 1920) for i in (1, 3))
    color = (255, 221, 40)
    left, right = max(0, x0), min(THUMB_W, x1)
    top, bottom = max(0, y0), min(THUMB_H, y1)
    if top < bottom:
        if 0 <= x0 < THUMB_W:
            image[top:bottom, x0] = color
        if 0 <= x1 < THUMB_W:
            image[top:bottom, x1] = color
    if left < right:
        if 0 <= y0 < THUMB_H:
            image[y0, left:right] = color
        if 0 <= y1 < THUMB_H:
            image[y1, left:right] = color


_DIGITS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ":": ("0", "1", "0", "1", "0"),
}


def _time_label(image: np.ndarray, second: int) -> None:
    label = f"{second // 60}:{second % 60:02}"
    image[4:18, 4:44] = (0, 0, 0)
    x = 6
    for char in label:
        for row, pattern in enumerate(_DIGITS[char]):
            for col, ink in enumerate(pattern):
                if ink == "1":
                    image[6 + row * 2:8 + row * 2, x + col * 2:x + col * 2 + 2] = 255
        x += 8 if char != ":" else 4


def make_sheet(plan: dict, plate: Path, output: Path) -> Path:
    """Project the camera crop and overlay planned face, caption, and hero boxes."""
    moments = range(0, math.ceil(plan["frames"] / plan["fps"]), 5)
    columns = 4
    canvas = np.zeros((math.ceil(len(moments) / columns) * THUMB_H,
                       columns * THUMB_W, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:THUMB_H, :THUMB_W]
    screen_x = (xx + 0.5) * 1080 / THUMB_W
    screen_y = (yy + 0.5) * 1920 / THUMB_H
    for index, second in enumerate(moments):
        frame = min(plan["frames"] - 1, second * plan["fps"])
        cam = plan["camera"]["samples"][frame]
        corners = project_plate_corners(plan["plate"], cam, 1080, 1920)
        scale = (corners[1][0] - corners[0][0]) / plan["plate"]["width"]
        source = _frame(plate, second)
        u = np.clip(((screen_x - corners[0][0]) / scale * 0.25).astype(int), 0, 389)
        v = np.clip(((screen_y - corners[0][1]) / scale * 0.25).astype(int), 0, 659)
        thumb = source[v, u].copy()
        _picture_edge(thumb, project_real_picture_box(plan["plate"], cam, 1080, 1920))
        section = next(
            (item for item in plan["silhouettes"] if item["start"] <= second < item["end"]),
            plan["silhouettes"][-1],
        )
        _outline(thumb, _head_at(plan["plate"], cam, section["head_box"]), (35, 210, 220))
        for group in plan["groups"]:
            if group["on_frame"] <= frame < group["off_frame"]:
                pose = group["poses"][frame - group["on_frame"]]
                _outline(thumb, pose["box"], (244, 239, 230))
        for sentence in plan["sentences"]:
            hero = sentence["hero"]
            if hero and hero["on_frame"] <= frame < hero["off_frame"]:
                x, baseline, scale = _project(cam, hero["ref_x"], hero["ref_y"],
                                               hero["depth"])
                size = hero["ref_size"] * scale
                top = baseline - _metrics()["base"]["hero"] * size
                accent = plan.get("style", {}).get("data", {}).get("brand", {}).get(
                    "accent_color", "") or "#F7471B"
                color = tuple(bytes.fromhex(accent.removeprefix("#")))
                _outline(thumb, [x, top, x + sum(hero["advances"]) * size,
                                 top + size], color)
        _time_label(thumb, second)
        row, col = divmod(index, columns)
        canvas[row * THUMB_H:(row + 1) * THUMB_H,
               col * THUMB_W:(col + 1) * THUMB_W] = thumb
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg.find_tool("ffmpeg"), "-hide_banner", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
               f"{canvas.shape[1]}x{canvas.shape[0]}", "-i", "pipe:0",
               "-frames:v", "1", str(output)]
    subprocess.run(command, input=canvas.tobytes(), capture_output=True, check=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("plate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    print(make_sheet(plan, args.plate, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
