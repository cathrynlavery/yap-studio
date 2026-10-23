"""Join-aware temporal person matte cleanup and edge measurements."""

from __future__ import annotations

import bisect
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path

import numpy as np

from yap_studio.edl import load_edit_list
from yap_studio.errors import ProcessingError, ValidationError
from yap_studio.matte.model import Cleanup
from yap_studio.plan import Plan

STILL_THRESHOLD = 6.0  # Six luma levels tolerate blurred sensor noise without accepting motion.
STILL_BLUR_RADIUS = 1  # A three-pixel box on the half grid suppresses fine sensor noise.
STILL_SUBSAMPLE = 2  # Half resolution keeps the motion test cheap and spatially local.
TEMPORAL_RADIUS = 2  # Five median results settle a stationary edge with short lookahead.
BAND_LOW = 0.03  # Keep the solid exterior out of edge measurements.
BAND_HIGH = 0.97  # Keep the solid interior out of edge measurements.
BAND_DILATION_SCALE = 360  # Three pixels at 1080 measure changes beside the uncertain rim.
_DEFAULT_CLEANUP = Cleanup()


def shot_starts(spans: Path, *, frames: int | None = None) -> tuple[int, ...]:
    """Return shot and hold boundaries on the cut's 30 fps output clock."""
    plan = Plan.build(load_edit_list(spans), fps=Fraction(30), sample_rate=48_000)
    if frames is not None and plan.total_frames != frames:
        raise ValidationError("cut spans", f"edit list plans {plan.total_frames} frames but the "
                              f"cut contains {frames} frames. Re-run cut, then assemble again.")
    joins = {right.out_f0 for left, right in zip(plan.segments, plan.segments[1:], strict=False)
             if left.source_id != right.source_id or left.src_f1 != right.src_f0}
    joins.update(frame for hold in plan.holds for frame in (hold.out_f0, hold.out_f1))
    return tuple(sorted(frame for frame in joins if 0 < frame < plan.total_frames))


def box_mean(array: np.ndarray, radius: int) -> np.ndarray:
    """Separable box mean with cloned spatial ends and float32 sums."""
    result = array.astype(np.float32, copy=False)
    size = 2 * radius + 1
    for axis in (0, 1):
        padding = [(0, 0)] * result.ndim
        padding[axis] = (radius + 1, radius)
        sums = np.cumsum(np.pad(result, padding, mode="edge"), axis=axis, dtype=np.float32)
        left, right = [slice(None)] * result.ndim, [slice(None)] * result.ndim
        left[axis], right[axis] = slice(None, -size), slice(size, None)
        result = (sums[tuple(right)] - sums[tuple(left)]) / size
    return result


def downsample(array: np.ndarray, factor: int) -> np.ndarray:
    """Average blocks, extending odd image sizes at their right and bottom edges."""
    h, w = array.shape[:2]
    padding = [(0, (-h) % factor), (0, (-w) % factor)] + [(0, 0)] * (array.ndim - 2)
    padded = np.pad(array, padding, mode="edge")
    result = np.zeros((padded.shape[0] // factor, padded.shape[1] // factor,
                       *array.shape[2:]), dtype=np.float32)
    for y in range(factor):
        for x in range(factor):
            result += padded[y::factor, x::factor]
    return result / (factor * factor)


def bilinear_at(array: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a small grid at fractional coordinates with cloned spatial ends."""
    y = np.clip(y, 0, array.shape[0] - 1)
    x = np.clip(x, 0, array.shape[1] - 1)
    y0, x0 = y.astype(np.intp), x.astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, array.shape[0] - 1), np.minimum(x0 + 1, array.shape[1] - 1)
    fy, fx = (y - y0).astype(np.float32), (x - x0).astype(np.float32)
    if array.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    return ((array[y0, x0] * (1 - fx) + array[y0, x1] * fx) * (1 - fy)
            + (array[y1, x0] * (1 - fx) + array[y1, x1] * fx) * fy)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary band with a square neighbourhood."""
    result = mask
    for axis in (0, 1):
        expanded = result.copy()
        for offset in range(1, min(radius, result.shape[axis] - 1) + 1):
            left, right = [slice(None)] * 2, [slice(None)] * 2
            left[axis], right[axis] = slice(None, -offset), slice(offset, None)
            expanded[tuple(left)] |= result[tuple(right)]
            expanded[tuple(right)] |= result[tuple(left)]
        result = expanded
    return result


def edge_band(alpha: np.ndarray) -> np.ndarray:
    """Include the uncertain alpha rim and its immediate spatial neighbours."""
    return dilate((alpha > BAND_LOW * 255) & (alpha < BAND_HIGH * 255),
                  max(1, round(min(alpha.shape) / BAND_DILATION_SCALE)))


def _still_guide(luma: np.ndarray) -> np.ndarray:
    return box_mean(downsample(luma, STILL_SUBSAMPLE), STILL_BLUR_RADIUS)


def _expand_still(a: np.ndarray, b: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.repeat(np.repeat(np.abs(a - b) <= STILL_THRESHOLD, STILL_SUBSAMPLE, axis=0),
                     STILL_SUBSAMPLE, axis=1)[:shape[0], :shape[1]]


def still_pixels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the same noise-tolerant still test used by smoothing and measurements."""
    return _expand_still(_still_guide(a), _still_guide(b), a.shape)


def temporal_mean(center: np.ndarray, guide: np.ndarray,
                  neighbours: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Average only pixels whose median changes; identical values need no arithmetic."""
    active = np.zeros(center.shape, dtype=bool)
    for alpha, _ in neighbours:
        active |= alpha != center
    result = center.copy()
    y, x = np.nonzero(active)
    if not len(y):
        return result
    sy, sx = y // STILL_SUBSAMPLE, x // STILL_SUBSAMPLE
    total = center[y, x].astype(np.float32)
    weights = np.ones(len(y), dtype=np.uint8)
    current = guide[sy, sx]
    for alpha, other in neighbours:
        eligible = np.abs(current - other[sy, sx]) <= STILL_THRESHOLD
        np.add(total, alpha[y, x], out=total, where=eligible)
        weights += eligible
    result[y, x] = np.round(total / weights).astype(np.uint8)
    return result


def refine_frames(
    masks: Iterator[np.ndarray], guides: Iterator[np.ndarray], *,
    cleanup: Cleanup = _DEFAULT_CLEANUP, joins: Sequence[int] = (),
) -> Iterator[np.ndarray]:
    """Stream temporal cleanup with at most seven raw frames and five cached medians."""
    yield from refine_pairs(iter(zip_longest(masks, guides)), cleanup=cleanup, joins=joins)


def refine_pairs(
    pairs: Iterator[tuple[np.ndarray | None, np.ndarray | None]], *,
    cleanup: Cleanup = _DEFAULT_CLEANUP, joins: Sequence[int] = (),
) -> Iterator[np.ndarray]:
    """Return median masks with a still-only mean clipped to each shot."""
    raw, still, medians = {}, {}, {}
    table = np.clip((np.arange(256) - cleanup.low) * 255 / (cleanup.high - cleanup.low),
                    0, 255).astype(np.uint8)
    read = 0
    ended = False
    t = 0
    while True:
        shot = bisect.bisect_right(joins, t)
        first = joins[shot - 1] if shot else 0
        for key in list(raw):
            if key < first:
                del raw[key]
        end = joins[shot] if shot < len(joins) else None
        limit = min(t + TEMPORAL_RADIUS + 1, end - 1) if end else t + TEMPORAL_RADIUS + 1
        while not ended and read <= limit:
            pair = next(pairs, None)
            if pair is None:
                ended = True
                break
            mask, guide = pair
            if mask is None or guide is None:
                raise ProcessingError("ffmpeg", 0, "matte and guide frame counts differ",
                                      code="matte_frame_count")
            raw[read] = table[mask]
            still[read] = _still_guide(guide)
            read += 1
            del mask, guide, pair
        if t >= read:
            return
        last = min(read, end) - 1 if end else read - 1
        neighbours = range(max(first, t - TEMPORAL_RADIUS), min(last, t + TEMPORAL_RADIUS) + 1)
        for n in neighbours:
            if n not in medians:
                a, b, c = raw[max(first, n - 1)], raw[n], raw[min(last, n + 1)]
                medians[n] = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c)) \
                    if cleanup.median else b
                del a, b, c
        yield temporal_mean(medians[t], still[t],
                            [(medians[n], still[n]) for n in neighbours if n != t])
        t += 1
        for cache, retain in ((raw, t + TEMPORAL_RADIUS - 1),
                              (still, t - TEMPORAL_RADIUS), (medians, t - TEMPORAL_RADIUS)):
            for key in list(cache):
                if key < retain:
                    del cache[key]


@dataclass(frozen=True)
class EdgeMeasurements:
    """Alpha change on still edges, transition width and decoded window length."""

    flicker: float
    edge_width_px: float
    frames: int


def measure_edges(matte: Path, video: Path, *, first: int = 0, count: int | None = None,
                  joins: Sequence[int] = ()) -> EdgeMeasurements:
    """Measure an exact frame window, excluding pairs across shot boundaries."""
    from yap_studio import ffmpeg
    from yap_studio.matte.frames import decode_frames

    info = ffmpeg.probe(video, count_frames=count is None)
    count = info.video_frames - first if count is None else count
    deadline = time.monotonic() + 120 + count
    previous = None
    change, pixels, width, used = 0.0, 0, 0.0, 0
    with ExitStack() as stack:
        masks = stack.enter_context(closing(decode_frames(matte, info.width, info.height,
            deadline, first=first, count=count)))
        guides = stack.enter_context(closing(decode_frames(video, info.width, info.height,
            deadline, first=first, count=count)))
        for alpha, guide in zip_longest(masks, guides):
            if alpha is None or guide is None:
                raise ProcessingError("ffmpeg", 0, "matte and video frame counts differ",
                                      code="matte_frame_count")
            solid = alpha >= 128
            boundary = np.zeros_like(solid)
            boundary[1:] |= solid[1:] & ~solid[:-1]
            boundary[:-1] |= solid[:-1] & ~solid[1:]
            boundary[:, 1:] |= solid[:, 1:] & ~solid[:, :-1]
            boundary[:, :-1] |= solid[:, :-1] & ~solid[:, 1:]
            width += np.count_nonzero((alpha > BAND_LOW * 255) & (alpha < BAND_HIGH * 255)) \
                / max(1, np.count_nonzero(boundary))
            band = edge_band(alpha)
            if previous is not None and first + used not in joins:
                old, old_guide, old_band = previous
                eligible = (band | old_band) & still_pixels(guide, old_guide)
                change += np.abs(alpha[eligible].astype(np.int16) - old[eligible]).sum()
                pixels += np.count_nonzero(eligible)
            previous = alpha, guide, band
            used += 1
    if used != count:
        raise ProcessingError("ffmpeg", 0, f"expected {count} frames, decoded {used}",
                              code="matte_frame_count")
    return EdgeMeasurements(float(change / max(1, pixels)), width / max(1, used), used)
