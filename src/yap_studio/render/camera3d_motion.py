"""Deterministic background translation measurements for camera captions."""

from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg

WIDTH, HEIGHT, SAMPLE_FPS = 216, 384, 5
MIN_PEAK = .08


def _samples(source: Path) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:v:0", "-an",
               "-vf", f"select=not(mod(n\\,6)),scale={WIDTH}:{HEIGHT},format=gray",
               "-fps_mode", "passthrough",
               "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    pending = bytearray()
    samples = []
    size = WIDTH * HEIGHT
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as chunks:
        for chunk in chunks:
            pending.extend(chunk)
            while len(pending) >= size:
                samples.append(np.frombuffer(bytes(pending[:size]), dtype=np.uint8)
                               .reshape(HEIGHT, WIDTH))
                del pending[:size]
    if pending or not samples:
        raise ValueError("could not sample complete camera motion frames")
    return np.stack(samples)


def _erode(mask: np.ndarray) -> np.ndarray:
    result = mask
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (4, 4)
        padded = np.pad(result, padding, constant_values=False)
        result = np.logical_and.reduce([
            np.take(padded, range(offset, offset + result.shape[axis]), axis=axis)
            for offset in range(9)
        ])
    return result


def _translation(a: np.ndarray, b: np.ndarray, background: np.ndarray) -> tuple[float, float]:
    images = []
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    for source in (a, b):
        values = source.astype(float)
        mean = values[background].mean()
        values = np.where(background, values - mean, 0)
        if values[background].std() < 1:
            return 0, 0
        images.append(np.fft.rfft2(values * window))
    cross = images[0] * images[1].conj()
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.irfft2(cross, s=a.shape)
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    offsets = []
    for axis, index in enumerate(peak):
        left, right = list(peak), list(peak)
        left[axis] = (index - 1) % a.shape[axis]
        right[axis] = (index + 1) % a.shape[axis]
        low, middle, high = correlation[tuple(left)], correlation[peak], correlation[tuple(right)]
        denominator = low - 2 * middle + high
        fraction = .5 * (low - high) / denominator if abs(denominator) > 1e-12 else 0
        offsets.append((index if index <= a.shape[axis] / 2 else index - a.shape[axis])
                       + float(np.clip(fraction, -.5, .5)))
    return float(np.hypot(*offsets) * 5), float(correlation[peak])


def measure_samples(picture: np.ndarray, matte: np.ndarray, *,
                    shot_starts: Sequence[int]) -> dict:
    """Measure adjacent 5 fps samples, six output frames apart within each shot."""
    if picture.shape != matte.shape or picture.ndim != 3:
        raise ValueError("motion picture and matte samples must have the same shape")
    pairs = []
    for index in range(1, len(picture)):
        first, last = (index - 1) * 6, index * 6
        if any(first < start <= last for start in shot_starts):
            continue
        background = _erode((matte[index - 1] < 32) & (matte[index] < 32))
        if background.mean() < .25:
            continue
        shift, peak = _translation(picture[index - 1], picture[index], background)
        if peak >= MIN_PEAK:
            pairs.append({"start_frame": first, "end_frame": last,
                          "shift_px": round(shift, 6), "peak": round(peak, 6)})
    percentile = float(np.percentile([pair["shift_px"] for pair in pairs], 75)) if pairs else None
    starts = sorted({0, *(start for start in shot_starts if 0 < start < len(picture) * 6)})
    shots = []
    for first, last in zip(starts, [*starts[1:], len(picture) * 6], strict=True):
        shifts = [pair["shift_px"] for pair in pairs
                  if first <= pair["start_frame"] and pair["end_frame"] < last]
        shift = float(np.percentile(shifts, 75)) if shifts else None
        shots.append({"start_frame": first, "end_frame": last,
                      "status": "measured" if shifts else "not measurable",
                      "shift_p75_px": shift, "moving": shift is not None and shift > 4})
    return {"status": "measured" if pairs else "not measurable", "pairs": pairs,
            "shift_p75_px": percentile, "shots": shots,
            "moving": any(shot["moving"] for shot in shots)}


def measure_camera_motion(cut: Path, matte: Path, shot_starts: Sequence[int], *,
                          matte_samples: np.ndarray | None = None) -> dict:
    """Sample the room independently of the moving speaker."""
    picture = _samples(cut)
    if matte_samples is None:
        matte_samples = _samples(matte)
    return measure_samples(picture, matte_samples, shot_starts=shot_starts)
