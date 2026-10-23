"""Audio decoding and signal measurements on the container clock."""

from __future__ import annotations

import functools
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import InputError, ProcessingError, ValidationError


def _samples(seconds: float, sr: int) -> int:
    return math.floor(seconds * sr + 0.5)


def _validate_rate(sr: int) -> None:
    if sr <= 0:
        raise ValidationError("sr", "must be positive")


@dataclass(frozen=True)
class Discontinuity:
    """A break in the audio stream's own timestamps.

    ``samples`` is positive for a gap (the next frame starts later than the
    previous one ends) and negative for an overlap. ``time_s`` is the container
    time where the previous frame's timestamps end. ``position`` counts decoded
    samples before the break, at the stream's native rate.
    """

    frame: int
    time_s: float
    position: int
    samples: int
    ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {**asdict(self), "kind": "gap" if self.samples > 0 else "overlap"}


@dataclass(frozen=True)
class Timeline:
    """Decoded audio frames compared with their container timestamps.

    ``decoded_samples`` is the audio content: every decoded sample, in order.
    ``span_samples`` runs from the first frame's timestamp to the end of the
    last frame by its timestamp. Breaks no larger than ``tolerance_samples``
    (one sample, or one tick of a coarser stream time base) are rounding.

    ``fills`` places the content on the container clock the way a player that
    never drops decoded audio does: each frame starts at its timestamp or where
    the previous frame ends, whichever is later. Each entry is a decoded sample
    position and the silence inserted before it. Overlaps insert nothing, so the
    sound after an overlap runs late by its size.
    """

    sample_rate: int
    frames: int
    decoded_samples: int
    span_samples: int
    tolerance_samples: int
    discontinuities: tuple[Discontinuity, ...]
    fills: tuple[tuple[int, int], ...] = ()

    @property
    def fill_samples(self) -> int:
        """Return the silence inserted to keep frames on their timestamps."""
        return sum(count for _, count in self.fills)

    @property
    def gap_samples(self) -> int:
        """Return the total of all gaps, in native samples."""
        return sum(item.samples for item in self.discontinuities if item.samples > 0)

    @property
    def overlap_samples(self) -> int:
        """Return the total of all overlaps as a positive number of native samples."""
        return -sum(item.samples for item in self.discontinuities if item.samples < 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "sample_rate": self.sample_rate,
            "frames": self.frames,
            "decoded_samples": self.decoded_samples,
            "span_samples": self.span_samples,
            "tolerance_samples": self.tolerance_samples,
            "gap_samples": self.gap_samples,
            "overlap_samples": self.overlap_samples,
            "fill_samples": self.fill_samples,
            "discontinuities": [item.to_dict() for item in self.discontinuities],
        }


def _parse_frames(text: str) -> tuple[Fraction | None, list[tuple[int | None, int]]]:
    time_base: Fraction | None = None
    frames: list[tuple[int | None, int]] = []
    for line in text.splitlines():
        fields = line.strip().split(",")
        if fields[0] == "stream" and len(fields) >= 2:
            try:
                time_base = Fraction(fields[1])
            except (ValueError, ZeroDivisionError):
                time_base = None
        elif fields[0] == "frame" and len(fields) >= 3:
            try:
                count = int(fields[2])
            except ValueError:
                continue
            try:
                pts = int(fields[1])
            except ValueError:
                pts = None
            frames.append((pts, count))
    return time_base, frames


@functools.lru_cache(maxsize=32)
def _timeline(
    path: str,
    size: int,
    modified_ns: int,
    inode: int,
    changed_ns: int,
    stream_index: int,
    sample_rate: int,
    format_start_s: float,
) -> Timeline:
    """Probe decoded frame timestamps once per file version."""
    # Cache keys only: a rewritten or replaced file is probed again, even at the same size and time.
    del size, modified_ns, inode, changed_ns
    result = ffmpeg.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            str(stream_index),
            "-show_entries",
            "stream=time_base:frame=pts,nb_samples",
            "-of",
            "csv=p=1",
            path,
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    output = result.stdout
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    time_base, frames = _parse_frames(output or "")
    ticks = time_base * sample_rate if time_base else Fraction(1)
    tolerance = max(1, math.ceil(ticks))
    breaks: list[Discontinuity] = []
    fills: list[tuple[int, int]] = []
    decoded = 0
    first: Fraction | None = None
    end: Fraction | None = None
    played: Fraction | None = None
    for index, (pts, count) in enumerate(frames):
        stamp = Fraction(pts) * ticks if pts is not None and time_base else None
        if played is not None and stamp is not None and stamp - played > tolerance:
            silence = round(stamp - played)
            fills.append((decoded, silence))
            played += silence
        if end is None:
            start = stamp if stamp is not None else Fraction(0)
            first = start
            played = start
        elif stamp is None:
            start = end
        else:
            delta = stamp - end
            if abs(delta) > tolerance:
                size_samples = round(delta)
                breaks.append(
                    Discontinuity(
                        frame=index,
                        time_s=float(end / sample_rate) - format_start_s,
                        position=decoded,
                        samples=size_samples,
                        ms=size_samples * 1000 / sample_rate,
                    )
                )
                start = stamp
            else:
                # Rounded timestamps stay on the running clock.
                start = end
        end = start + count
        played += count
        decoded += count
    span = round(end - first) if end is not None and first is not None else 0
    return Timeline(
        sample_rate=sample_rate,
        frames=len(frames),
        decoded_samples=decoded,
        span_samples=span,
        tolerance_samples=tolerance,
        discontinuities=tuple(breaks),
        fills=tuple(fills),
    )


def _clock_for(source: Path, info: ffmpeg.MediaInfo) -> Timeline:
    status = source.stat()
    return _timeline(
        str(source.resolve()),
        status.st_size,
        status.st_mtime_ns,
        status.st_ino,
        status.st_ctime_ns,
        info.audio_index,
        info.sample_rate,
        info.format_start_s,
    )


def timeline(path: str | os.PathLike[str]) -> Timeline:
    """Compare the selected audio stream's decoded frames with their timestamps."""
    source = Path(path)
    info = ffmpeg.probe(source, count_frames=False)
    if not info.has_audio or info.audio_index is None or info.sample_rate is None:
        raise InputError(str(source), "no audio stream")
    return _clock_for(source, info)


def _fill_gaps(
    audio: np.ndarray, clock: Timeline, native_rate: int, output_rate: int, source: Path
) -> np.ndarray:
    """Insert the timeline's silences into sequentially decoded audio."""
    if not clock.fills:
        return audio
    scale = Fraction(output_rate, native_rate)
    expected = round(clock.decoded_samples * scale)
    # Resampling can round the stream length by a sample or two.
    slack = 0 if output_rate == native_rate else 4
    if abs(len(audio) - expected) > slack:
        raise ProcessingError(
            f"decode {source}",
            1,
            f"decoded {len(audio)} audio samples but the frame timestamps describe "
            f"{expected}; the audio cannot be placed on the container clock",
            code="audio_clock_mismatch",
        )
    pieces: list[np.ndarray] = []
    previous = 0
    for position, count in clock.fills:
        at = min(len(audio), round(position * scale))
        pieces.append(audio[previous:at])
        pieces.append(np.zeros((round(count * scale), *audio.shape[1:]), dtype=np.float32))
        previous = at
    pieces.append(audio[previous:])
    return np.concatenate(pieces, axis=0)


@dataclass(frozen=True)
class Located:
    """A signal location and the confidence measurements for its peak."""

    time_s: float
    corr: float
    ambiguity: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _selected_audio_info(source: Path, stream: int | None) -> ffmpeg.MediaInfo:
    """Probe facts for an audio ordinal, retaining the container clock."""
    if stream is not None and (
        isinstance(stream, bool) or not isinstance(stream, int) or stream < 0
    ):
        raise ValidationError("stream", "must be a non-negative audio ordinal")
    info = ffmpeg.probe(source, count_frames=False)
    if stream is None:
        return info
    result = ffmpeg.run([
        "ffprobe", "-v", "error", "-select_streams", f"a:{stream}",
        "-show_streams", "-of", "json", str(source),
    ], timeout_s=ffmpeg.PROBE_TIMEOUT_S)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise InputError(str(source), "no audio stream for the selected ordinal")
    selected = streams[0]
    start = ffmpeg._optional_float(selected.get("start_time"))
    return replace(
        info,
        has_audio=True,
        audio_index=ffmpeg._optional_int(selected.get("index")),
        audio_codec=selected.get("codec_name"),
        sample_rate=ffmpeg._optional_int(selected.get("sample_rate")),
        channels=ffmpeg._optional_int(selected.get("channels")),
        audio_start_s=start,
        audio_offset_s=round((start or 0.0) - info.format_start_s, 6),
        audio_duration_s=ffmpeg._optional_float(selected.get("duration")),
    )


def decode(
    path: str | os.PathLike[str],
    *,
    stream: int | None = None,
    sr: int | None = None,
    mono: bool = True,
    start_s: float | None = None,
    dur_s: float | None = None,
) -> tuple[int, np.ndarray]:
    """Decode the selected audio stream on the container clock.

    ``stream`` is an audio ordinal (``0:a:N``); None selects the first stream.
    Sample zero is container time zero, and sample N plays at N / sr for the
    whole stream. Where the audio timestamps jump ahead, the gap is filled with
    silence so later sound stays on its timestamps. Where they step back, the
    decoded sound keeps playing in order and nothing is dropped, the way a
    player that never discards decoded audio plays it; the sound after such an
    overlap runs late by its size. ``timeline`` reports both kinds of break.
    Raises ProcessingError when the decoded length does not match the frame
    timestamps, because the gaps could not then be placed.
    """
    source = Path(path)
    info = _selected_audio_info(source, stream)
    if (
        not info.has_audio
        or info.audio_index is None
        or info.sample_rate is None
        or info.channels is None
    ):
        raise InputError(str(source), "no audio stream")
    output_rate = info.sample_rate if sr is None else sr
    _validate_rate(output_rate)
    if start_s is not None and start_s < 0:
        raise ValidationError("start_s", "must be non-negative")
    if dur_s is not None and dur_s < 0:
        raise ValidationError("dur_s", "must be non-negative")

    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        f"0:{info.audio_index}",
        "-vn",
    ]
    channels = 1 if mono else info.channels
    if mono:
        command.extend(["-ac", "1"])
    if sr is not None:
        command.extend(["-ar", str(output_rate)])
    command.extend(["-f", "f32le", "-"])
    result = ffmpeg.run(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, binary=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if channels > 1:
        audio = audio[: len(audio) - len(audio) % channels].reshape(-1, channels)
    # The raw PCM pipe drops timestamps; restore gaps from the frame timestamps.
    audio = _fill_gaps(audio, _clock_for(source, info), info.sample_rate, output_rate, source)

    offset = _samples(info.audio_offset_s, output_rate)
    if offset > 0:
        shape = (offset,) if mono else (offset, channels)
        audio = np.concatenate((np.zeros(shape, dtype=np.float32), audio), axis=0)
    elif offset < 0:
        audio = audio[min(-offset, len(audio)) :]

    first = _samples(start_s or 0.0, output_rate)
    if dur_s is None:
        return output_rate, audio[first:].copy()
    count = _samples(dur_s, output_rate)
    shape = (count,) if mono else (count, channels)
    sliced = np.zeros(shape, dtype=np.float32)
    available = max(0, min(count, len(audio) - first))
    if available:
        sliced[:available] = audio[first : first + available]
    return output_rate, sliced


def raised_cosine(n: int) -> np.ndarray:
    """Return a half raised-cosine ramp from zero to one."""
    if n < 0:
        raise ValidationError("n", "must be non-negative")
    if n == 0:
        return np.empty(0, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)


def rms_db(sig: np.ndarray, center_s: float, sr: int, half_s: float = 0.005) -> float:
    """Measure RMS level in a window centered at a time."""
    _validate_rate(sr)
    if half_s < 0:
        raise ValidationError("half_s", "must be non-negative")
    first = max(0, _samples(center_s - half_s, sr))
    last = _samples(center_s + half_s, sr)
    segment = np.asarray(sig)[first:last]
    if len(segment) == 0:
        return -120.0
    rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))
    return float(20 * np.log10(rms + 1e-9))


def click_db(m: np.ndarray, j: int, sr: int) -> float:
    """Measure join sharpness against nearby high-frequency signal peaks."""
    _validate_rate(sr)
    signal = np.asarray(m)
    if signal.ndim != 1:
        raise ValidationError("m", "must be a mono array")
    if not 0 <= j < len(signal):
        raise ValidationError("j", "must index the signal")
    difference = np.abs(np.diff(signal.astype(np.float64)))
    w2, w100, w5 = int(0.002 * sr), int(0.1 * sr), int(0.005 * sr)
    near_values = difference[max(0, j - w2) : min(len(difference), j + w2)]
    if not len(near_values):
        return -120.0
    near = near_values.max()
    low, high = max(0, j - w100), min(len(difference), j + w100)
    mask = np.ones(high - low, dtype=bool)
    mask[max(0, j - w5 - low) : max(0, j + w5 - low)] = False
    reference_values = difference[low:high][mask]
    reference = (np.percentile(reference_values, 99.5) if len(reference_values) else 0.0) + 1e-12
    return float(20 * np.log10(near / reference + 1e-12))


def pause_at(
    m: np.ndarray,
    j: int,
    sr: int,
    thr_db: float = -45.0,
    step_s: float = 0.01,
) -> float:
    """Measure the run of quiet windows touching a sample position."""
    _validate_rate(sr)
    signal = np.asarray(m)
    if signal.ndim != 1:
        raise ValidationError("m", "must be a mono array")
    if not 0 <= j < len(signal):
        raise ValidationError("j", "must index the signal")
    step = _samples(step_s, sr)
    if step <= 0:
        raise ValidationError("step_s", "must span at least one sample")
    half = step // 2

    def level(index: int) -> float:
        return rms_db(signal, index / sr, sr, half / sr)

    left = 0
    index = j
    while index - step > 0 and level(index - half) < thr_db:
        left += step
        index -= step
    right = 0
    index = j
    while index + step < len(signal) and level(index + half) < thr_db:
        right += step
        index += step
    return (left + right) / sr


def splice(
    x: np.ndarray,
    pieces_samples: list[tuple[int, int]],
    fade: int,
) -> tuple[np.ndarray, list[int]]:
    """Concatenate ranges with non-overlapping raised-cosine fades."""
    signal = np.asarray(x)
    if signal.ndim not in {1, 2}:
        raise ValidationError("x", "must have one or two dimensions")
    if fade < 0:
        raise ValidationError("fade", "must be non-negative")
    if not pieces_samples:
        return signal[:0].copy(), []
    ramp = raised_cosine(fade)
    if signal.ndim == 2:
        ramp = ramp[:, None]
    output: list[np.ndarray] = []
    joins: list[int] = []
    accumulated = 0
    for index, (start, end) in enumerate(pieces_samples):
        if start < 0 or end < start or end > len(signal):
            raise ValidationError("pieces_samples", "ranges must be ordered within x")
        segment = signal[start:end].copy()
        if fade and index > 0 and len(segment) >= fade:
            segment[:fade] *= ramp
        if fade and index < len(pieces_samples) - 1 and len(segment) >= fade:
            segment[-fade:] *= ramp[::-1]
        output.append(segment)
        accumulated += len(segment)
        if index < len(pieces_samples) - 1:
            joins.append(accumulated)
    return np.concatenate(output), joins


def _fft_valid_correlation(hay: np.ndarray, needle: np.ndarray) -> np.ndarray:
    size = len(hay) + len(needle) - 1
    nfft = 1 << size.bit_length()
    convolution = np.fft.irfft(
        np.fft.rfft(hay, nfft) * np.fft.rfft(needle[::-1], nfft), nfft
    )[:size]
    return convolution[len(needle) - 1 : len(hay)]


def _whiten(signal: np.ndarray) -> np.ndarray:
    """First-difference and robustly compress magnitude so tones do not dominate."""
    difference = np.diff(signal)
    return np.sign(difference) * np.sqrt(np.sqrt(np.abs(difference)))


def locate(
    needle: np.ndarray,
    hay: np.ndarray,
    sr: int,
    *,
    expect_s: float,
    search_s: float,
) -> Located:
    """Locate audio with a whitened, energy-normalised correlation."""
    _validate_rate(sr)
    if search_s < 0:
        raise ValidationError("search_s", "must be non-negative")
    query = np.asarray(needle, dtype=np.float64)
    signal = np.asarray(hay, dtype=np.float64)
    if query.ndim != 1 or signal.ndim != 1:
        raise ValidationError("audio", "needle and hay must be mono arrays")
    if len(query) < 2:
        raise ValidationError("needle", "must contain at least two samples")
    whitened_query = _whiten(query)
    query_energy = float(np.dot(whitened_query, whitened_query))
    if query_energy < 1e-12:
        return Located(expect_s, 0.0, 1.0)

    first = max(0, _samples(expect_s - search_s, sr))
    last = min(len(signal) - len(query), _samples(expect_s + search_s, sr))
    if last < first:
        raise ValidationError("hay", "search window cannot fit the needle")
    # One extra original sample is needed because first differences preserve start indices.
    region = signal[first : last + len(query)]
    whitened_hay = _whiten(region)
    products = _fft_valid_correlation(whitened_hay, whitened_query)
    squared = whitened_hay * whitened_hay
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    width = len(whitened_query)
    energies = cumulative[width:] - cumulative[:-width]
    correlations = products / (np.sqrt(energies * query_energy) + 1e-12)

    best = int(np.argmax(correlations))
    peak = float(correlations[best])
    fraction = 0.0
    if 0 < best < len(correlations) - 1:
        left, center, right = correlations[best - 1 : best + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-15:
            fraction = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))

    local = np.ones(len(correlations), dtype=bool)
    if len(correlations) > 1:
        local[1:] &= correlations[1:] >= correlations[:-1]
        local[:-1] &= correlations[:-1] >= correlations[1:]
    exclusion = _samples(0.005, sr)
    indices = np.flatnonzero(local & (np.abs(np.arange(len(correlations)) - best) > exclusion))
    second = float(np.max(correlations[indices])) if len(indices) else 0.0
    ambiguity = float(np.clip(second / peak, 0.0, 1.0)) if peak > 0 else 1.0
    return Located((first + best + fraction) / sr, peak, ambiguity)


def lag(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    max_lag_s: float = 0.5,
) -> tuple[float, float]:
    """Return lag of b against a, positive when b plays late, and peak correlation."""
    _validate_rate(sr)
    if max_lag_s < 0:
        raise ValidationError("max_lag_s", "must be non-negative")
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1 or not len(first) or not len(second):
        raise ValidationError("audio", "a and b must be non-empty mono arrays")
    first = first - first.mean()
    second = second - second.mean()
    nfft = 1 << (len(first) + len(second) - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(second, nfft) * np.conj(np.fft.rfft(first, nfft)), nfft
    )
    radius = min(_samples(max_lag_s, sr), len(correlation) // 2)
    candidates = (
        correlation[:1]
        if radius == 0
        else np.concatenate((correlation[-radius:], correlation[: radius + 1]))
    )
    index = int(np.argmax(candidates)) - radius
    energy = np.sqrt(np.dot(first, first) * np.dot(second, second)) + 1e-12
    return index / sr, float(candidates.max() / energy)
