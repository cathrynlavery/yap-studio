"""Drawn layer bounds and caption geometry on the final video canvas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.faces.clearance import DEFAULT_HAIR, LAYOUT_SCHEMA, Layout
from yap_studio.faces.model import Box


def _alpha_frames(layer: Path, width: int, height: int, *,
                  crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Decode the layer alpha on its original frame clock."""
    filters = "alphaextract"
    if crop is not None:
        left, top, width, height = crop
        filters = f"crop={width}:{height}:{left}:{top}:exact=1," + filters
    data = ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(layer), "-map", "0:v:0", "-an",
        "-vf", filters, "-pix_fmt", "gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, binary=True).stdout
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, height, width)


def drawn_boxes(
    layer: Path, *, x: float, y: float, width: int, height: int, threshold: int = 64,
    region: Box | None = None,
) -> list[tuple[int, int, Box]]:
    """Group consecutive frames with identical visible alpha bounds in output coordinates."""
    runs: list[tuple[int, int, Box]] = []
    left = max(0, int(region.x - x)) if region else 0
    top = max(0, int(region.y - y)) if region else 0
    right = min(width, int(region.x1 - x)) if region else width
    bottom = min(height, int(region.y1 - y)) if region else height
    if right <= left or bottom <= top:
        return runs
    crop = (left, top, right - left, bottom - top) if region else None
    for frame, alpha in enumerate(_alpha_frames(layer, width, height, crop=crop)):
        rows, columns = np.nonzero(alpha >= threshold)
        if not rows.size:
            continue
        first_x, first_y = int(columns.min()), int(rows.min())
        box = Box(x + left + first_x, y + top + first_y,
                  int(columns.max()) - first_x + 1, int(rows.max()) - first_y + 1)
        if runs and runs[-1][1] == frame and runs[-1][2] == box:
            runs[-1] = (runs[-1][0], frame + 1, box)
        else:
            runs.append((frame, frame + 1, box))
    return runs


def drawn_box(
    layer: Path, *, x: float, y: float, width: int, height: int, threshold: int = 64,
) -> Box | None:
    """Union visible alpha pixels over every layer frame in output coordinates."""
    alpha = _alpha_frames(layer, width, height)
    if not alpha.size:
        return None
    rows, columns = np.nonzero(alpha.max(axis=0) >= threshold)
    if not rows.size:
        return None
    left, top = int(columns.min()), int(rows.min())
    return Box(x + left, y + top, int(columns.max()) - left + 1, int(rows.max()) - top + 1)


def final_layout(
    captions_layout: Mapping[str, Any], elements: Sequence[dict], *,
    width: int, height: int, clearance_px: float, caption_gap_px: float,
) -> dict:
    """Keep caption windows and drawn elements with their clearance policy."""
    document = {
        "schema": LAYOUT_SCHEMA, "width": width, "height": height,
        "measured": captions_layout["measured"], "reason": captions_layout["reason"],
        **({"caption_placement": captions_layout["caption_placement"]}
           if "caption_placement" in captions_layout else {}),
        "captions": captions_layout["captions"], "elements": list(elements),
        "clearance_px": clearance_px, "caption_gap_px": caption_gap_px,
        "hair": DEFAULT_HAIR.to_dict(),
    }
    Layout.from_dict(document)
    return document
