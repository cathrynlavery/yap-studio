"""Shared constant-rate checks for frame-indexed mattes and composites."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from yap_studio import ffmpeg
from yap_studio.errors import InputError


def require_constant_rate(path: Path, info: ffmpeg.MediaInfo) -> None:
    """Require matching reported rates and packet PTS steps within a quarter frame."""
    error = InputError(str(path), "video needs constant-rate timestamps")
    if info.variable_frame_rate or info.fps is None or info.fps <= 0:
        raise error
    result = ffmpeg.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time", "-of", "json", str(path)],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    try:
        times = sorted(Fraction(packet["pts_time"]) for packet in json.loads(result.stdout)[
            "packets"
        ])
    except (ValueError, ZeroDivisionError, KeyError, TypeError) as exc:
        raise error from exc
    step = 1 / info.fps
    if not times or any(
        abs(second - first - step) > step / 4
        for first, second in zip(times, times[1:], strict=False)
    ):
        raise error
