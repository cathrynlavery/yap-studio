"""Streaming colour thumbnails, labelled contact strips, and deterministic PNGs."""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Iterator, Sequence
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import InputError, ValidationError

LONG_SIDE = 320
LABEL_HEIGHT = 24
TILE_GAP = 2
MAX_TILES = 5000
BACKGROUND = (22, 25, 30)
FOREGROUND = (240, 242, 245)
BEFORE_COLOUR = (255, 170, 40)
AFTER_COLOUR = (40, 205, 225)
FONT_SCALE = 2
BORDER = 3
PNG_COMPRESSION = 9

# Original 5 by 7 glyphs. Each row's five low bits describe its lit pixels.
FONT = {
    "0": (14, 17, 19, 21, 25, 17, 14),
    "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31),
    "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2),
    "5": (31, 16, 16, 30, 1, 1, 30),
    "6": (14, 16, 16, 30, 17, 17, 14),
    "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 1, 14),
    ":": (0, 4, 4, 0, 4, 4, 0),
    ".": (0, 0, 0, 0, 0, 6, 6),
    "C": (15, 16, 16, 16, 16, 16, 15),
    "<": (1, 2, 4, 8, 4, 2, 1),
    ">": (16, 8, 4, 2, 4, 8, 16),
    " ": (0, 0, 0, 0, 0, 0, 0),
}


def format_time(time_s: float) -> str:
    """Format a video time to the nearest hundredth, with carry into minutes."""
    hundredths = max(0, math.floor(time_s * 100 + 0.5))
    minutes, remainder = divmod(hundredths, 6000)
    return f"{minutes}:{remainder // 100:02d}.{remainder % 100:02d}"


def tile_dimensions(width: int, height: int) -> tuple[int, int]:
    """Scale the long picture side to 320, rounding both dimensions to even pixels."""
    factor = LONG_SIDE / max(width, height)
    return tuple(max(2, math.floor(value * factor / 2 + 0.5) * 2) for value in (width, height))


def timeline_tiles(
    frames: int,
    fps: Fraction,
    every_s: float,
    cuts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge regular samples and cut sides, deduplicating on canonical frame index."""
    rate = float(fps)
    tiles = {}

    def check_limit() -> None:
        if len(tiles) > MAX_TILES:
            error = ValidationError(
                "timeline", f"watch pack would contain more than {MAX_TILES} tiles"
            )
            error.suggested_action = "Pass a larger --every to reduce the number of timeline tiles."
            raise error

    # Sub-frame intervals visit every frame after deduplication. Enumerate frames
    # directly so an arbitrarily small --every cannot create unbounded work.
    step = every_s * rate
    indexes = (
        range(frames)
        if step < 1
        else (
            min(frames - 1, round(sample * every_s * rate))
            for sample in range(max(1, math.ceil(frames / step)))
        )
    )
    for frame in indexes:
        tiles[frame] = {
            "frame": frame,
            "time_s": round(frame / rate, 3),
            "kind": "sample",
            "cut": None,
            "cut_roles": [],
        }
        check_limit()
    for cut in cuts:
        for frame, kind in ((cut["frame"] - 1, "cut_before"), (cut["frame"], "cut_after")):
            role = {"cut": cut["index"], "kind": kind}
            if tiles.get(frame, {}).get("cut") is not None:
                tiles[frame]["cut_roles"].append(role)
            else:
                tiles[frame] = {
                    "frame": frame,
                    "time_s": round(frame / rate, 3),
                    "kind": kind,
                    "cut": cut["index"],
                    "cut_roles": [role],
                }
            check_limit()
    return [tiles[frame] for frame in sorted(tiles)]


def layout_strips(tiles: Sequence[dict[str, Any]], columns: int) -> list[list[dict[str, Any]]]:
    """Keep cut pairs in one row, including shared sides of consecutive cuts.

    A connected group wider than columns occupies its own row. This also lets a
    two-tile cut pair stay together when the requested sample column count is one.
    """
    groups: list[list[dict[str, Any]]] = []
    for tile in tiles:
        roles = tile["cut_roles"]
        after = {role["cut"] for role in roles if role["kind"] == "cut_after"}
        previous = groups[-1][-1] if groups else None
        before = (
            {role["cut"] for role in previous["cut_roles"] if role["kind"] == "cut_before"}
            if previous
            else set()
        )
        if after & before:
            groups[-1].append(tile)
        else:
            groups.append([tile])
    rows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for group in groups:
        if current and len(current) + len(group) > columns:
            rows.append(current)
            current = []
        current.extend(group)
    if current:
        rows.append(current)
    return [[{**tile, "column": column} for column, tile in enumerate(row, 1)] for row in rows]


def decode_thumbnails(
    path: Path,
    indexes: Sequence[int],
    *,
    info: ffmpeg.MediaInfo,
    width: int,
    height: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield selected RGB frames in order on the grey decoder's frame clock."""
    wanted = set(indexes)
    if not wanted:
        return
    if min(wanted) < 0 or info.fps is None or info.video_index is None:
        raise ValidationError("thumbnails", "require non-negative indexes and a video stream")
    fps = f"{info.fps.numerator}/{info.fps.denominator}"
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:{info.video_index}",
        "-an",
        "-vf",
        f"setpts=PTS-STARTPTS,fps={fps}:start_time=0:round=near,"
        f"scale={width}:{height}:flags=area,format=rgb24",
        "-frames:v",
        str(max(wanted) + 1),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    frame_size = width * height * 3
    storage = bytearray()
    frame = 0
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as chunks:
        for chunk in chunks:
            storage.extend(chunk)
            consumed = 0
            while len(storage) - consumed >= frame_size:
                if frame in wanted:
                    wanted.remove(frame)
                    yield frame, (
                        np.frombuffer(storage, dtype=np.uint8, count=frame_size, offset=consumed)
                        .reshape(height, width, 3)
                        .copy()
                    )
                frame += 1
                consumed += frame_size
            del storage[:consumed]
    if wanted:
        raise InputError(str(path), "video ended before the requested thumbnail frames")


def draw_label(band: np.ndarray, text: str, *, colour: tuple[int, int, int] = FOREGROUND) -> None:
    """Draw into a label band only, clipping deterministically at its right edge."""
    for char in text:
        if char not in FONT:
            raise ValidationError("label", f"unsupported character {char!r}")
    x = 4
    y = (len(band) - 7 * FONT_SCALE) // 2
    for char in text:
        for row, bits in enumerate(FONT[char]):
            for column in range(5):
                if bits & (1 << (4 - column)):
                    band[
                        y + row * FONT_SCALE : y + (row + 1) * FONT_SCALE,
                        x + column * FONT_SCALE : x + (column + 1) * FONT_SCALE,
                    ] = colour
        x += 6 * FONT_SCALE


def compose_strip(
    tiles: Sequence[dict[str, Any]],
    thumbnails: dict[int, np.ndarray],
) -> np.ndarray:
    """Compose one contact row with fixed backgrounds, borders, and bitmap labels."""
    height, width, _ = thumbnails[tiles[0]["frame"]].shape
    strip_width = width * len(tiles) + TILE_GAP * (len(tiles) - 1)
    strip = np.full((height + LABEL_HEIGHT, strip_width, 3), BACKGROUND, dtype=np.uint8)
    for column, tile in enumerate(tiles):
        start = column * (width + TILE_GAP)
        region = strip[:, start : start + width]
        region[:height] = thumbnails[tile["frame"]]
        prefix = ""
        if tile["kind"] != "sample":
            before = tile["kind"] == "cut_before"
            colour = BEFORE_COLOUR if before else AFTER_COLOUR
            region[:BORDER] = colour
            region[height - BORDER : height] = colour
            region[:height, :BORDER] = colour
            region[:height, width - BORDER :] = colour
            prefix = f"C{tile['cut']}{'<' if before else '>'} "
        draw_label(region[height:], prefix + format_time(tile["time_s"]))
    return strip


def png_bytes(pixels: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as a deterministic PNG using filter zero."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValidationError("PNG", "expected an RGB uint8 array")
    height, width, _ = pixels.shape
    if not width or not height:
        raise ValidationError("PNG", "dimensions must be positive")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=PNG_COMPRESSION))
        + chunk(b"IEND", b"")
    )
