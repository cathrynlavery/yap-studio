"""Pure picture, sound, and pacing measurements on the output video clock."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from yap_studio.transcript import Word

STILL_THRESHOLD = 0.25
STILL_MIN_S = 0.5
MOTION_THRESHOLD = 6.0
MOTION_MIN_S = 0.5
MOMENTARY_S = 0.4
LOUDNESS_FLOOR_LUFS = -70.0
SPEECH_GATE_LUFS = -50.0
RELATIVE_GATE_LU = 10.0
CUT_SIDE_S = 2.0
SILENCE_DBFS = -45.0
SILENCE_MIN_S = 0.3
SILENCE_WINDOW_S = 0.01
SILENCE_SAMPLE_RATE = 16000
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+|inf)"


def motion_values(frames: np.ndarray, previous: np.ndarray | None = None) -> list[float]:
    """Return per-frame mean absolute differences, optionally across a block seam."""
    if not len(frames):
        return []
    values = np.asarray(frames, dtype=np.int16)
    first = 0.0 if previous is None else float(np.abs(values[0] - previous).mean())
    return [first, *np.abs(np.diff(values, axis=0)).mean(axis=(1, 2)).tolist()]


def _runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, keep in enumerate([*mask, False]):
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            runs.append((start, index))
            start = None
    return runs


def motion_spans(
    values: Sequence[float], fps: float, *, threshold: float, minimum_s: float, still: bool
) -> list[dict[str, Any]]:
    """Find maximal runs, including the initial picture of each still stretch."""
    mask = [value < threshold if still else value >= threshold for value in values]
    runs = [(max(0, start - 1) if still else start, end) for start, end in _runs(mask)]
    return [
        {
            "start_s": round(start / fps, 3),
            "end_s": round(end / fps, 3),
            "duration_s": round((end - start) / fps, 3),
            "start_frame": start,
            "end_frame": end,
        }
        for start, end in runs
        if (end - start) / fps >= minimum_s
    ]


def words_on_picture_clock(
    words: Sequence[Word],
    audio_offset_s: float,
    video_offset_s: float,
) -> list[Word]:
    """Shift ASR's WAV-relative times to picture zero, clipping words at frame zero."""
    offset = audio_offset_s - video_offset_s
    return [
        replace(word, start=max(0.0, word.start + offset), end=word.end + offset)
        for word in words
        if word.end + offset >= 0
    ]


def motion_per_second(values: Sequence[float], fps: float) -> list[dict[str, Any]]:
    """Summarize frame differences by the second containing each frame start."""
    buckets: dict[int, list[float]] = {}
    for frame, value in enumerate(values):
        buckets.setdefault(math.floor(frame / fps), []).append(value)
    return [
        {
            "second": second,
            "mean": round(sum(bucket) / len(bucket), 3),
            "max": round(max(bucket), 3),
        }
        for second, bucket in buckets.items()
    ]


def parse_ebur128(stderr: str) -> tuple[list[tuple[float, float]], float | None]:
    """Parse momentary block ends and the final integrated loudness summary."""
    blocks = [
        (float(match[1]), float(match[2]))
        for match in re.finditer(rf"\bt:\s*({_NUMBER})[^\n]*?\bM:\s*({_NUMBER})", stderr)
        if all(math.isfinite(float(value)) for value in match.groups())
    ]
    summary = stderr.rsplit("Summary:", 1)[-1] if "Summary:" in stderr else ""
    match = re.search(rf"\bI:\s*({_NUMBER})\s+LUFS", summary)
    integrated = float(match[1]) if match else None
    if integrated is not None and (
        not math.isfinite(integrated) or integrated <= LOUDNESS_FLOOR_LUFS
    ):
        integrated = None
    return blocks, integrated


def energy_mean(levels: Sequence[float]) -> float | None:
    """Average LUFS in energy space, returning None for an empty selection."""
    if not levels:
        return None
    return 10 * math.log10(sum(10 ** (level / 10) for level in levels) / len(levels))


def window_loudness(
    blocks: Sequence[tuple[float, float]],
    start: float,
    end: float,
    *,
    speech: bool = False,
    epsilon: float = 0.0001,
) -> float | None:
    """Measure wholly contained blocks, with an optional speech and relative gate."""
    levels = [
        level
        for t, level in blocks
        if t - MOMENTARY_S >= start - epsilon
        and t <= end + epsilon
        and (level >= SPEECH_GATE_LUFS if speech else level > LOUDNESS_FLOOR_LUFS)
    ]
    mean = energy_mean(levels)
    if speech and mean is not None:
        mean = energy_mean([level for level in levels if level >= mean - RELATIVE_GATE_LU])
    return None if mean is None else round(mean, 1)


def loudness_per_second(
    blocks: Sequence[tuple[float, float]], duration_s: float, *, epsilon: float = 0.0001
) -> list[dict[str, Any]]:
    """Return one energy mean per video second, with nulls for silent windows."""
    return [
        {
            "second": second,
            "lufs": window_loudness(
                blocks,
                second,
                min(second + 1, duration_s),
                epsilon=epsilon,
            ),
        }
        for second in range(math.ceil(duration_s))
    ]


def cut_loudness(
    blocks: Sequence[tuple[float, float]],
    cuts: Sequence[float],
    duration_s: float,
    *,
    epsilon: float = 0.0001,
) -> list[dict[str, float | None]]:
    """Measure speech on both sides of cuts, clipped to adjacent shots."""
    results = []
    for index, time in enumerate(cuts):
        start = max(time - CUT_SIDE_S, cuts[index - 1] if index else 0.0)
        end = min(time + CUT_SIDE_S, cuts[index + 1] if index + 1 < len(cuts) else duration_s)
        before = window_loudness(blocks, start, time, speech=True, epsilon=epsilon)
        after = window_loudness(blocks, time, end, speech=True, epsilon=epsilon)
        results.append(
            {
                "loudness_before_lufs": before,
                "loudness_after_lufs": after,
                "loudness_jump_lu": None
                if before is None or after is None
                else round(after - before, 1),
            }
        )
    return results


def silences(samples: np.ndarray, sr: int) -> list[dict[str, float]]:
    """Find quiet RMS runs in 10 ms windows, including a partial final window."""
    size = max(1, round(sr * SILENCE_WINDOW_S))
    values = np.asarray(samples, dtype=np.float64)
    full = len(values) // size
    rms = np.sqrt(np.mean(values[: full * size].reshape(full, size) ** 2, axis=1)).tolist()
    if len(values) % size:
        rms.append(float(np.sqrt(np.mean(values[full * size :] ** 2))))
    quiet = [value < 10 ** (SILENCE_DBFS / 20) for value in rms]
    result = []
    for start, end in _runs(quiet):
        start_s, end_s = start * size / sr, min(end * size, len(values)) / sr
        if end_s - start_s >= SILENCE_MIN_S - 1e-9:
            result.append(
                {
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "duration_s": round(end_s - start_s, 3),
                }
            )
    return result


def segment_words(words: Sequence[Word], start: float, end: float) -> list[Word]:
    """Assign words to a segment using their midpoint, with an exclusive end."""
    return sorted(
        (word for word in words if start <= (word.start + word.end) / 2 < end),
        key=lambda word: (word.start, word.end, word.text),
    )


def pacing(
    start: float,
    end: float,
    words: Sequence[Word] | None,
    silence_spans: Sequence[dict[str, float]] | None,
) -> dict[str, Any]:
    """Measure word rate, pauses and the longest intersecting audio silence."""
    result: dict[str, Any] = dict.fromkeys(
        ["words", "words_per_s", "longest_pause_s", "longest_pause_at_s", "text"]
    )
    result["longest_silence_s"] = (
        None
        if silence_spans is None
        else round(
            max(
                [
                    0.0,
                    *(
                        max(0.0, min(end, span["end_s"]) - max(start, span["start_s"]))
                        for span in silence_spans
                    ),
                ]
            ),
            3,
        )
    )
    if words is None:
        return result
    selected = segment_words(words, start, end)
    cursor = start
    gaps = []
    for word in selected:
        gaps.append((max(0.0, min(word.start, end) - cursor), cursor))
        cursor = max(cursor, min(word.end, end))
    gaps.append((max(0.0, end - cursor), cursor))
    gap, at = max(gaps, key=lambda item: item[0])
    result.update(
        words=len(selected),
        words_per_s=round(len(selected) / (end - start), 3),
        longest_pause_s=round(gap, 3),
        longest_pause_at_s=round(at, 3),
        text=" ".join(word.text for word in selected),
    )
    return result
