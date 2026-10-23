"""Bounded RGB decoding and small image primitives for finished-video measurements."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg


@dataclass(frozen=True)
class FrameTiming:
    """Presentation-order frame starts and the end of the final displayed frame."""

    starts: tuple[float, ...]
    end_s: float
    from_timestamps: bool
    variable_rate: bool

    def boundary(self, frame: int) -> float:
        """Return a frame start, or the final end for the boundary after all frames."""
        return self.starts[frame] if frame < len(self.starts) else self.end_s

    def window_s(self, first: int, last: int) -> float:
        """Measure the complete visible interval, including its final frame."""
        return self.boundary(last + 1) - self.boundary(first)

    def frame_at(self, seconds: float) -> int:
        """Map an audio event to its nearest presentation frame, clamped to the video."""
        after = min(len(self.starts) - 1, int(np.searchsorted(self.starts, seconds)))
        before = max(0, after - 1)
        return before if seconds - self.starts[before] < self.starts[after] - seconds else after


def presentation_timing(path: Path, info: ffmpeg.MediaInfo) -> FrameTiming:
    """Probe presentation timestamps once, in the same order as decoded RGB frames."""
    result = ffmpeg.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_streams",
            "-show_entries",
            "frame=pts_time,best_effort_timestamp_time,duration_time,pkt_duration_time:stream=time_base",
            "-of",
            "json",
            str(path),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    payload = json.loads(result.stdout)
    frames = payload.get("frames", [])
    count = info.video_frames or len(frames)
    fps = float(info.fps or 30)
    duration = info.video_duration_s or info.duration_s or count / fps
    timed = []
    for frame in frames:
        pts = frame.get("pts_time", frame.get("best_effort_timestamp_time"))
        if pts is None:
            return FrameTiming(tuple(i / fps for i in range(count)), duration, False, False)
        span = frame.get("duration_time", frame.get("pkt_duration_time", 0))
        timed.append((float(pts), float(span)))
    timed.sort(key=lambda row: row[0])
    if len(timed) != count or not timed:
        return FrameTiming(tuple(i / fps for i in range(count)), duration, False, False)
    origin = timed[0][0]
    starts = tuple(pts - origin for pts, _ in timed)
    intervals = np.diff(starts)
    last_duration = timed[-1][1]
    end = starts[-1] + last_duration if last_duration > 0 else duration
    if end <= starts[-1]:
        end = starts[-1] + (float(np.median(intervals)) if len(intervals) else 1 / fps)
    streams = payload.get("streams", [])
    tick = float(Fraction(streams[0].get("time_base", "0"))) if streams else 0
    # A constant cadence can alternate by one timestamp tick after rounding.
    # Detect actual rate variation rather than imposing a percentage cutoff.
    variable = bool(len(intervals) and np.ptp(intervals) > max(0.00001, 2 * tick))
    return FrameTiming(starts, end, True, variable)


def rgb(color: str) -> np.ndarray:
    """Convert a hex display colour to three float channels."""
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def hex_color(value: np.ndarray | Sequence[float]) -> str:
    """Round and clamp display RGB channels to a portable hex colour."""
    return "#" + "".join(f"{int(v):02X}" for v in np.clip(np.rint(value), 0, 255))


def lab(value: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert display sRGB to CIE Lab under the D65 reference white."""
    channels = np.asarray(value, dtype=float) / 255
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    xyz = (
        linear
        @ np.array(
            [
                [0.4124564, 0.2126729, 0.0193339],
                [0.3575761, 0.7151522, 0.1191920],
                [0.1804375, 0.0721750, 0.9503041],
            ]
        )
        / np.array([0.95047, 1, 1.08883])
    )
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        (116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])),
        axis=-1,
    )


def delta_e(first: str | np.ndarray, second: str | np.ndarray) -> float:
    """Return the CIE76 distance between two display colours."""
    a = rgb(first) if isinstance(first, str) else first
    b = rgb(second) if isinstance(second, str) else second
    return float(np.linalg.norm(lab(a) - lab(b)))


def runs(mask: np.ndarray, *, gap: int = 0) -> list[tuple[int, int]]:
    """Return half-open runs of true values, merging small internal gaps."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > gap + 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks] + 1, indices[-1] + 1]
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def core(mask: np.ndarray) -> np.ndarray:
    """Erode one pixel to exclude antialiasing and subsampled chroma edges."""
    result = mask.copy()
    result[1:] &= mask[:-1]
    result[:-1] &= mask[1:]
    result[:, 1:] &= mask[:, :-1]
    result[:, :-1] &= mask[:, 1:]
    result[[0, -1]] = False
    result[:, [0, -1]] = False
    return result


def interior_components(mask: np.ndarray) -> np.ndarray:
    """Remove colour regions connected to a crop boundary using horizontal runs.

    Run-based filling visits spans rather than individual pixels, so a large
    wall or silhouette remains cheap to exclude from a panel's interior ink.
    """
    result = mask.copy()
    if not result.size:
        return result
    height, width = result.shape
    stack = [(y, a) for y in (0, height - 1) for a, _ in runs(result[y])]
    stack.extend((a, x) for x in (0, width - 1) for a, _ in runs(result[:, x]))
    while stack:
        y, x = stack.pop()
        if not result[y, x]:
            continue
        before = np.flatnonzero(~result[y, :x])
        after = np.flatnonzero(~result[y, x:])
        left = int(before[-1]) + 1 if len(before) else 0
        right = x + int(after[0]) if len(after) else width
        result[y, left:right] = False
        for neighbour in (y - 1, y + 1):
            if 0 <= neighbour < height:
                stack.extend((neighbour, left + a) for a, _ in runs(result[neighbour, left:right]))
    return result


def palette(
    image: np.ndarray, mask: np.ndarray | None = None, *, count: int = 8
) -> list[tuple[np.ndarray, int]]:
    """Find frequent colour clusters and retain the median unquantized channels."""
    pixels = image.reshape(-1, 3) if mask is None else image[mask]
    if not len(pixels):
        return []
    # Bound histogram work, independently of source resolution.
    pixels = pixels[:: max(1, len(pixels) // 100_000)].astype(np.int32)
    quantized = pixels // 12
    keys = quantized[:, 0] * 22**2 + quantized[:, 1] * 22 + quantized[:, 2]
    bins = np.bincount(keys, minlength=22**3)
    result = []
    for key in np.argsort(bins)[::-1]:
        if not bins[key]:
            break
        color = np.median(pixels[keys == key], axis=0)
        if any(np.linalg.norm(color - previous) < 16 for previous, _ in result):
            continue
        result.append((color, int(bins[key])))
        if len(result) == count:
            break
    return result


def color_mask(image: np.ndarray, color: np.ndarray, radius: float = 10) -> np.ndarray:
    """Select pixels close to a colour in display RGB, without unsigned overflow."""
    return np.sum((image.astype(np.float32) - color) ** 2, axis=-1) <= radius**2


def frame_selection(indices: Sequence[int]) -> str:
    """Compress strides and balance sums to stay below ffmpeg's parser depth limit."""
    chosen = sorted(set(indices))
    clauses = []
    first = 0
    while first < len(chosen):
        last = first
        step = chosen[first + 1] - chosen[first] if first + 1 < len(chosen) else 1
        while last + 1 < len(chosen) and chosen[last + 1] - chosen[last] == step:
            last += 1
        if last > first and (step == 1 or last - first >= 2):
            clause = f"between(n,{chosen[first]},{chosen[last]})"
            if step != 1:
                clause += f"*not(mod(n-{chosen[first]},{step}))"
            clauses.append(clause)
            first = last + 1
        else:
            clauses.append(f"eq(n,{chosen[first]})")
            first += 1
    while len(clauses) > 1:
        clauses = [
            f"({clauses[i]}+{clauses[i + 1]})" if i + 1 < len(clauses) else clauses[i]
            for i in range(0, len(clauses), 2)
        ]
    return clauses[0] if clauses else "0"


def iter_rgb(
    path: Path, info: ffmpeg.MediaInfo, indices: Sequence[int], *, width: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream selected original frame indices using the stream's matrix and range.

    Selection precedes scaling and uses passthrough timestamps; neither an fps
    filter nor input seeking can change the evidence frame numbers. Only one RGB
    picture is retained. ffmpeg autorotation supplies displayed coordinates.
    """
    chosen = sorted(set(indices))
    if not chosen:
        return
    output_width = width or info.width
    assert output_width and info.width and info.height
    output_height = round(info.height * output_width / info.width)
    matrix = info.color_space or "bt709"
    matrix = {"bt2020nc": "bt2020", "smpte170m": "bt601", "bt470bg": "bt601"}.get(matrix, matrix)
    if matrix not in {"bt709", "bt601", "bt2020", "smpte240m", "fcc"}:
        matrix = "bt709"
    color_range = "pc" if info.color_range in {"pc", "jpeg"} else "tv"
    selection = frame_selection(chosen)
    filters = (
        f"select='{selection}',scale={output_width}:{output_height}:"
        f"in_color_matrix={matrix}:in_range={color_range},format=rgb24"
    )
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        filters,
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(len(chosen)),
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    picture = np.empty((output_height, output_width, 3), dtype=np.uint8)
    target = memoryview(picture).cast("B")
    used = index = 0
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as stream:
        for chunk in stream:
            raw = memoryview(chunk)
            while raw:
                size = min(len(raw), len(target) - used)
                target[used : used + size] = raw[:size]
                raw = raw[size:]
                used += size
                if used == len(target):
                    yield chosen[index], picture
                    index += 1
                    picture = np.empty_like(picture)
                    target = memoryview(picture).cast("B")
                    used = 0
