"""Small-frame video decoding, matching, and picture mapping."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Generator
from contextlib import closing
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import InputError, ValidationError
from yap_studio.grade import GradeSpec, graph_filter, write_cube
from yap_studio.hdr import conversion


@dataclass(frozen=True)
class Mapping:
    """A geometric relationship between source and rendered picture."""

    kind: Literal["full", "crop", "fit"]
    scale: float
    x: float
    y: float
    width: float
    height: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _even(value: float) -> int:
    return max(2, math.floor(value / 2.0 + 0.5) * 2)


def source_rate(info: ffmpeg.MediaInfo, plan_fps: Fraction) -> Fraction | None:
    """Use the cut's frame grid for VFR sources and the native grid for CFR."""
    return plan_fps if info.variable_frame_rate else info.fps


def _grey_command(
    path: str | os.PathLike[str], w: int, h: int | None, rate: Fraction | None = None,
    cube: Path | None = None,
) -> tuple[list[str], int]:
    source = Path(path)
    info = ffmpeg.probe(source, count_frames=False)
    if (
        not info.has_video
        or info.video_index is None
        or info.fps is None
        or info.width is None
        or info.height is None
    ):
        raise InputError(str(source), "no video stream")
    if w <= 0:
        raise ValidationError("w", "must be positive")
    if h is None:
        h = _even(w * info.height / info.width)
    if h <= 0:
        raise ValidationError("h", "must be positive")
    decode_rate = info.fps if rate is None else rate
    if decode_rate <= 0:
        raise ValidationError("rate", "must be positive")
    rate_text = f"{decode_rate.numerator}/{decode_rate.denominator}"
    filters = (
        f"setpts=PTS-STARTPTS,fps={rate_text}:start_time=0:round=near,"
        f"scale={w}:{h}:flags=area,format=gray"
    )
    selected = conversion(info) if cube is not None else None
    if cube is not None:
        filters = ((selected.filter_prefix + "," if selected else "")
                   + filters.replace(",format=gray", f",{graph_filter(cube)},format=gray"))
    return (
        [
            "ffmpeg",
            "-loglevel",
            "error",
            *(selected.input_args if selected else []),
            "-i",
            str(source),
            "-map",
            f"0:{info.video_index}",
            "-an",
            "-vf",
            filters,
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "-",
        ],
        h,
    )


def grey_frames(
    path: str | os.PathLike[str],
    *,
    w: int = 64,
    h: int | None = None,
    start_frame: int | None = None,
    count: int | None = None,
    rate: Fraction | None = None,
    grade: GradeSpec | None = None,
) -> np.ndarray:
    """Decode upright grey frames as a writable uint8 array of shape ``(n, h, w)``.

    Stream frame ``n`` starts at ``video_offset_s + n / rate`` on the container
    clock; ``rate`` defaults to the probed fps. A window starts at ``start_frame``
    on that same clock. Only the returned frames are retained; ``count`` stops
    decoding after that window.
    """
    if grade is not None and grade.active:
        with tempfile.TemporaryDirectory(prefix="yap-grade-") as folder:
            cube = write_cube(grade, Path(folder))
            return _grey_frames(path, w=w, h=h, start_frame=start_frame, count=count,
                                rate=rate, cube=cube)
    return _grey_frames(path, w=w, h=h, start_frame=start_frame, count=count, rate=rate)


def _grey_frames(
    path: str | os.PathLike[str], *, w: int, h: int | None, start_frame: int | None,
    count: int | None, rate: Fraction | None, cube: Path | None = None,
) -> np.ndarray:
    command, h = _grey_command(path, w, h, rate, cube)
    first = 0 if start_frame is None else start_frame
    if first < 0:
        raise ValidationError("start_frame", "must be non-negative")
    if count is not None and count < 0:
        raise ValidationError("count", "must be non-negative")
    if count == 0:
        return np.empty((0, h, w), dtype=np.uint8)
    if count is not None:
        command[-1:-1] = ["-frames:v", str(first + count)]
    frame_size = w * h
    skip = first * frame_size
    storage = bytearray()
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as chunks:
        for chunk in chunks:
            skipped = min(skip, len(chunk))
            skip -= skipped
            storage.extend(memoryview(chunk)[skipped:])
    # The bytearray becomes the array's backing store, without a second full copy.
    size = len(storage) - len(storage) % frame_size
    return np.frombuffer(storage, dtype=np.uint8, count=size).reshape(-1, h, w)


def iter_grey_frames(
    path: str | os.PathLike[str],
    *,
    w: int = 64,
    h: int | None = None,
    block: int = 128,
    rate: Fraction | None = None,
    grade: GradeSpec | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield ``(first_index, uint8 frames)`` blocks on ``grey_frames``' clock.

    Each independent, writable array has shape ``(n, h, w)``, with at most
    ``block`` frames. Close the iterator when stopping before the end.
    """
    if block <= 0:
        raise ValidationError("block", "must be positive")
    if grade is not None and grade.active:
        with tempfile.TemporaryDirectory(prefix="yap-grade-") as folder:
            cube = write_cube(grade, Path(folder))
            yield from _iter_grey_frames(path, w=w, h=h, block=block, rate=rate, cube=cube)
        return
    yield from _iter_grey_frames(path, w=w, h=h, block=block, rate=rate)


def _iter_grey_frames(
    path: str | os.PathLike[str], *, w: int, h: int | None, block: int,
    rate: Fraction | None, cube: Path | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    command, h = _grey_command(path, w, h, rate, cube)
    frames = np.empty((block, h, w), dtype=np.uint8)
    target = memoryview(frames).cast("B")
    used = first = 0
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as chunks:
        for chunk in chunks:
            raw = memoryview(chunk)
            while raw:
                size = min(len(raw), len(target) - used)
                target[used : used + size] = raw[:size]
                used += size
                raw = raw[size:]
                if used == len(target):
                    yield first, frames
                    first += block
                    frames = np.empty((block, h, w), dtype=np.uint8)
                    target = memoryview(frames).cast("B")
                    used = 0
    count = used // (w * h)
    if count:
        yield first, frames[:count]


def _normalise(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def rgb_frames(path: Path, frames: list[int], *, w: int = 90, h: int = 160,
               rate: Fraction | None = None) -> np.ndarray:
    """Seek to each selected output-grid frame as a small BT.709 RGB picture."""
    frames = sorted({frame for frame in frames if frame >= 0})
    if not frames:
        return np.empty((0, h, w, 3), dtype=np.uint8)
    if rate is None:
        rate = ffmpeg.probe(path, count_frames=False).fps
    if not rate:
        raise InputError(str(path), "video has no frame rate for RGB sampling")
    filters = f"scale={w}:{h}:flags=area:in_color_matrix=bt709:in_range=tv,format=rgb24"
    storage = bytearray()
    size = w * h * 3
    for frame in frames:
        command = ["ffmpeg", "-v", "error", "-ss", f"{float(frame / rate):.12f}",
                   "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an",
                   "-filter_threads", "1", "-vf", filters, "-fps_mode", "passthrough",
                   "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as chunks:
            for chunk in chunks:
                storage.extend(chunk)
        if len(storage) % size:
            raise InputError(str(path), "RGB decode ended with a partial frame")
    return np.frombuffer(storage, dtype=np.uint8).reshape(-1, h, w, 3)


def frame_distance(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Compare two frames after normalising brightness and contrast.

    ``valid`` optionally limits the comparison, normalisation included, to the
    pixels where it is true.
    """
    first = np.asarray(a)
    second = np.asarray(b)
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValidationError("frames", "a and b must be two-dimensional with equal shapes")
    if valid is None:
        return float(np.mean(np.abs(_normalise(first) - _normalise(second))))
    keep = np.asarray(valid, dtype=bool)
    if keep.shape != first.shape:
        raise ValidationError("valid", "must match the frame shape")
    if not keep.any():
        raise ValidationError("valid", "must keep at least one pixel")
    return float(np.mean(np.abs(_normalise(first[keep]) - _normalise(second[keep]))))


def exclusion_mask(
    shape: tuple[int, int],
    rects: list[tuple[float, float, float, float]],
    width: float,
    height: float,
) -> np.ndarray:
    """Mark small-frame pixels touched by output-pixel rectangles ``(x, y, w, h)``.

    ``width`` and ``height`` are the output picture size the rectangles use.
    A pixel only partly covered is excluded too.
    """
    rows, columns = shape
    excluded = np.zeros(shape, dtype=bool)
    if width <= 0 or height <= 0:
        return excluded
    scale_x, scale_y = columns / width, rows / height
    for x, y, w, h in rects:
        left = max(0, math.floor(x * scale_x))
        right = min(columns, math.ceil((x + w) * scale_x))
        top = max(0, math.floor(y * scale_y))
        bottom = min(rows, math.ceil((y + h) * scale_y))
        if right > left and bottom > top:
            excluded[top:bottom, left:right] = True
    return excluded


def valid_in_comparison(
    excluded: np.ndarray, mapping: Mapping, size: tuple[int, int]
) -> np.ndarray:
    """Carry a render-space exclusion into a mapping's comparison space.

    Returns the pixels to compare. Any comparison pixel that samples an excluded
    render pixel is left out.
    """
    weight = apply_mapping(excluded.astype(np.float32), mapping, role="render", size=size)
    return weight <= 1e-6


def locate_sequence(
    needle: np.ndarray,
    hay: np.ndarray,
    *,
    expect: int,
    search: int,
    valid: np.ndarray | None = None,
) -> tuple[int, float, float]:
    """Locate a short frame sequence near an expected index, comparing ``valid`` pixels."""
    query = np.asarray(needle)
    signal = np.asarray(hay)
    if query.ndim != 3 or signal.ndim != 3 or query.shape[1:] != signal.shape[1:]:
        raise ValidationError("frames", "needle and hay must be compatible frame arrays")
    if not len(query):
        raise ValidationError("needle", "must not be empty")
    if search < 0:
        raise ValidationError("search", "must be non-negative")
    first = max(0, expect - search)
    last = min(len(signal) - len(query), expect + search)
    if last < first:
        raise ValidationError("hay", "search window cannot fit the needle")
    starts = np.arange(first, last + 1)
    distances = np.asarray(
        [
            np.mean([frame_distance(signal[q + i], query[i], valid) for i in range(len(query))])
            for q in starts
        ]
    )
    selected = int(np.argmin(distances))
    best_index = int(starts[selected])
    eligible = np.abs(starts - best_index) >= 2
    second = float(np.min(distances[eligible])) if np.any(eligible) else math.inf
    return best_index, float(distances[selected]), second


def best_of(frame: np.ndarray, candidates: np.ndarray) -> tuple[int, list[float]]:
    """Return the closest candidate frame and all candidate distances."""
    options = np.asarray(candidates)
    if options.ndim != 3 or not len(options):
        raise ValidationError("candidates", "must be a non-empty frame array")
    distances = [frame_distance(frame, candidate) for candidate in options]
    return int(np.argmin(distances)), distances


def _as_frames(frames: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim == 2:
        return values[None, ...], True
    if values.ndim == 3:
        return values, False
    raise ValidationError("frames", "must have shape [h, w] or [n, h, w]")


def _sample_rect(
    frames: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    size: tuple[int, int],
) -> np.ndarray:
    """Bilinearly sample a pixel-center rectangle into a requested size."""
    values, _ = _as_frames(frames)
    count, source_h, source_w = values.shape
    output_h, output_w = size
    if output_h <= 0 or output_w <= 0 or width <= 0 or height <= 0:
        raise ValidationError("size", "sampled rectangles and output dimensions must be positive")
    xs = x + (np.arange(output_w, dtype=np.float64) + 0.5) * width / output_w - 0.5
    ys = y + (np.arange(output_h, dtype=np.float64) + 0.5) * height / output_h - 0.5
    xs = np.clip(xs, 0.0, source_w - 1.0)
    ys = np.clip(ys, 0.0, source_h - 1.0)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, source_w - 1)
    y1 = np.minimum(y0 + 1, source_h - 1)
    wx = (xs - x0)[None, None, :]
    wy = (ys - y0)[None, :, None]
    top = values[:, y0[:, None], x0[None, :]] * (1.0 - wx)
    top += values[:, y0[:, None], x1[None, :]] * wx
    bottom = values[:, y1[:, None], x0[None, :]] * (1.0 - wx)
    bottom += values[:, y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32).reshape(
        count, output_h, output_w
    )


def _resize(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    values, _ = _as_frames(frames)
    return _sample_rect(values, 0.0, 0.0, values.shape[2], values.shape[1], size)


def apply_mapping(
    frames: np.ndarray,
    mapping: Mapping,
    *,
    role: Literal["source", "render"],
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Transform source or render frames into their common comparison space."""
    values, single = _as_frames(frames)
    _, height, width = values.shape
    if role not in {"source", "render"}:
        raise ValidationError("role", "must be 'source' or 'render'")
    if mapping.scale <= 0:
        raise ValidationError("mapping", "scale must be positive")
    if mapping.width <= 0 or mapping.height <= 0:
        raise ValidationError("mapping", "width and height must be positive")

    if mapping.kind == "crop" and role == "source":
        target = size or (max(1, round(mapping.height)), max(1, round(mapping.width)))
        output = _sample_rect(
            values,
            mapping.x,
            mapping.y,
            mapping.width,
            mapping.height,
            target,
        )
    elif mapping.kind == "fit" and role == "render":
        target = size or (max(1, round(mapping.height)), max(1, round(mapping.width)))
        output = _sample_rect(
            values,
            mapping.x,
            mapping.y,
            mapping.width,
            mapping.height,
            target,
        )
    else:
        target = size or (height, width)
        output = _resize(values, target)
    return output[0] if single else output


def _pair_distance(
    first: np.ndarray, second: np.ndarray, valid: np.ndarray | None = None
) -> float:
    return min(frame_distance(a, b, valid) for a in first for b in second)


def _score(distance: float) -> float:
    return float(np.clip(1.0 - distance / 1.128, 0.0, 1.0))


def estimate_mapping(
    source_frames: np.ndarray,
    render_frames: np.ndarray,
    render_valid: np.ndarray | None = None,
) -> Mapping:
    """Estimate full, crop, or fit mapping from small nearby frame sets.

    ``render_valid`` optionally limits every comparison to render pixels where
    it is true, for example to leave out a burned-in caption band.
    """
    source, _ = _as_frames(source_frames)
    render, _ = _as_frames(render_frames)
    if not len(source) or not len(render):
        raise ValidationError("frames", "source_frames and render_frames must not be empty")
    _, source_h, source_w = source.shape
    _, render_h, render_w = render.shape
    valid = None if render_valid is None else np.asarray(render_valid, dtype=bool)
    excluded = None if valid is None else (~valid).astype(np.float32)
    source_aspect = source_w / source_h
    render_aspect = render_w / render_h
    candidates: list[Mapping] = []

    if abs(source_aspect / render_aspect - 1.0) <= 0.03:
        transformed = _resize(source, (render_h, render_w))
        distance = _pair_distance(transformed, render, valid)
        candidates.append(
            Mapping(
                "full",
                render_w / source_w,
                0.0,
                0.0,
                float(render_w),
                float(render_h),
                _score(distance),
            )
        )

    def crop_candidate(x: float, y: float, window_w: float, window_h: float) -> Mapping:
        transformed = _sample_rect(source, x, y, window_w, window_h, (render_h, render_w))
        distance = _pair_distance(transformed, render, valid)
        scale = 0.5 * (render_w / window_w + render_h / window_h)
        return Mapping(
            "crop",
            scale,
            float(x),
            float(y),
            float(window_w),
            float(window_h),
            _score(distance),
        )

    coarse: list[Mapping] = []
    if render_aspect < source_aspect:
        window_h = float(source_h)
        window_w = window_h * render_aspect
        maximum = source_w - window_w
        positions = list(np.arange(0.0, maximum + 1e-9, 1.0))
        if not positions or positions[-1] < maximum:
            positions.append(maximum)
        coarse = [crop_candidate(x, 0.0, window_w, window_h) for x in positions]
    elif render_aspect > source_aspect:
        window_w = float(source_w)
        window_h = window_w / render_aspect
        maximum = source_h - window_h
        positions = list(np.arange(0.0, maximum + 1e-9, 1.0))
        if not positions or positions[-1] < maximum:
            positions.append(maximum)
        coarse = [crop_candidate(0.0, y, window_w, window_h) for y in positions]
    if coarse:
        best_coarse = max(coarse, key=lambda item: item.score)
        candidates.append(best_coarse)
        if best_coarse.y == 0.0:
            maximum = source_w - window_w
            refine = np.arange(best_coarse.x - 1.0, best_coarse.x + 1.001, 0.25)
            candidates.extend(
                crop_candidate(float(np.clip(x, 0.0, maximum)), 0.0, window_w, window_h)
                for x in refine
            )
        else:
            maximum = source_h - window_h
            refine = np.arange(best_coarse.y - 1.0, best_coarse.y + 1.001, 0.25)
            candidates.extend(
                crop_candidate(0.0, float(np.clip(y, 0.0, maximum)), window_w, window_h)
                for y in refine
            )

    fit_scale = min(render_w / source_w, render_h / source_h)
    placed_w = source_w * fit_scale
    placed_h = source_h * fit_scale
    center_x = (render_w - placed_w) / 2.0
    center_y = (render_h - placed_h) / 2.0
    for dx in (-2.0, 0.0, 2.0):
        for dy in (-2.0, 0.0, 2.0):
            x = center_x + dx
            y = center_y + dy
            if x < 0 or y < 0 or x + placed_w > render_w or y + placed_h > render_h:
                continue
            region = _sample_rect(render, x, y, placed_w, placed_h, (source_h, source_w))
            region_valid = None
            if excluded is not None:
                weight = _sample_rect(excluded, x, y, placed_w, placed_h, (source_h, source_w))[0]
                region_valid = weight <= 1e-6
                if not region_valid.any():
                    continue
            distance = _pair_distance(source, region, region_valid)
            candidates.append(
                Mapping("fit", fit_scale, x, y, placed_w, placed_h, _score(distance))
            )

    if not candidates:
        raise ValidationError("frames", "no valid mapping candidates")
    return max(candidates, key=lambda item: item.score)
