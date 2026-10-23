"""Find speech without transcript words and merge a short second pass."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from yap_studio.transcript import Word, norm_words

# Measured on a real take with dropped speech and checked against steady noise fixtures.
ALLOW_BASE_S = 0.3
ALLOW_PER_LETTER_S = 0.1
COVER_PAD_S = 0.1
BELOW_SPEECH_DB = 20.0
FLOOR_MARGIN_DB = 15.0
REFERENCE_MARGIN_DB = 10.0
QUIET_DBFS = -70.0
HOLE_S = 0.3
MIN_SPEECH_S = 0.6
DIP_DB = 10.0
MIN_DIP_SHARE = 0.15
WINDOW_PAD_S = 2.0
WINDOW_MIN_S = 1.2
MATCH_MIDPOINT_S = 0.3

THRESHOLDS = {
    "allow_base_s": ALLOW_BASE_S,
    "allow_per_letter_s": ALLOW_PER_LETTER_S,
    "cover_pad_s": COVER_PAD_S,
    "below_speech_db": BELOW_SPEECH_DB,
    "floor_margin_db": FLOOR_MARGIN_DB,
    "reference_margin_db": REFERENCE_MARGIN_DB,
    "quiet_dbfs": QUIET_DBFS,
    "hole_s": HOLE_S,
    "min_speech_s": MIN_SPEECH_S,
    "dip_db": DIP_DB,
    "min_dip_share": MIN_DIP_SHARE,
    "window_pad_s": WINDOW_PAD_S,
    "window_min_s": WINDOW_MIN_S,
    "match_midpoint_s": MATCH_MIDPOINT_S,
}


@dataclass
class _Cluster:
    """Track one run of uncovered loud frames and short quiet holes."""

    start: int
    last_loud: int
    loud_frames: int
    quiet_frames: int = 0


def _allowance(word: Word) -> float:
    """Bound one word's coverage by its letters, even when its end is stretched."""
    letters = sum(char.isalnum() for char in word.text)
    return ALLOW_BASE_S + ALLOW_PER_LETTER_S * letters


def speech_warning(start: float, end: float, seconds: float) -> str:
    """Describe a source-clock stretch that still needs a human listen."""
    first, last = (int(time + 0.5) for time in (start, end))
    return (
        f"speech with no words at {first // 60}:{first % 60:02d}-"
        f"{last // 60}:{last % 60:02d} ({seconds:.1f} s); "
        "listen before you use that part"
    )


def uncovered_speech(
    level: np.ndarray, words: Sequence[Word], *, hop_s: float = 0.01,
) -> list[tuple[float, float]]:
    """Return speech-shaped uncovered spans on the WAV clock."""
    if len(level) == 0:
        return []
    floor = float(np.percentile(level, 10))
    midpoints = [
        round((word.start + word.end) / (2 * hop_s))
        for word in words
        if math.isfinite(word.start) and math.isfinite(word.end)
    ]
    midpoints = [index for index in midpoints if 0 <= index < len(level)]
    reference = (
        float(np.median(level[midpoints])) if len(midpoints) >= 10
        else float(np.percentile(level, 95)) - 6.0
    )
    threshold = max(
        reference - BELOW_SPEECH_DB,
        min(floor + FLOOR_MARGIN_DB, reference - REFERENCE_MARGIN_DB),
    )
    loud = (level >= threshold) & (level >= QUIET_DBFS)
    covered = np.zeros(len(level), dtype=bool)
    for word in words:
        if not (math.isfinite(word.start) and math.isfinite(word.end)):
            continue
        first = max(0, math.ceil((word.start - COVER_PAD_S) / hop_s))
        last_s = min(word.end, word.start + _allowance(word)) + COVER_PAD_S
        last = min(len(level), math.ceil(last_s / hop_s))
        if first < last:
            covered[first:last] = True

    clusters: list[tuple[float, float]] = []
    current: _Cluster | None = None

    def finish() -> None:
        """Keep a long, modulated cluster and clear scanner state."""
        nonlocal current
        if current is None:
            return
        if current.loud_frames * hop_s >= MIN_SPEECH_S:
            span = level[current.start : current.last_loud + 1]
            peak = float(np.percentile(span, 90))
            if float(np.mean(span <= peak - DIP_DB)) >= MIN_DIP_SHARE:
                clusters.append((current.start * hop_s, (current.last_loud + 1) * hop_s))
        current = None

    max_hole = round(HOLE_S / hop_s)
    for index in range(len(level)):
        if covered[index]:
            finish()
        elif loud[index]:
            if current is None:
                current = _Cluster(index, index, 1)
            else:
                current.last_loud = index
                current.loud_frames += 1
                current.quiet_frames = 0
        elif current is not None:
            current.quiet_frames += 1
            if current.quiet_frames > max_hole:
                finish()
    finish()
    return clusters


def windows(
    clusters: Sequence[tuple[float, float]], words: Sequence[Word], duration_s: float,
) -> list[tuple[float, float]]:
    """Pad speech spans, include complete edge words, and merge overlaps."""
    ranges = []
    for start, end in clusters:
        left = max(0.0, start - WINDOW_PAD_S)
        right = min(duration_s, end + WINDOW_PAD_S)
        extra = max(0.0, (WINDOW_MIN_S - (right - left)) / 2)
        left, right = max(0.0, left - extra), min(duration_s, right + extra)
        if right - left < WINDOW_MIN_S:
            left = max(0.0, min(left, duration_s - WINDOW_MIN_S))
            right = min(duration_s, max(right, left + WINDOW_MIN_S))
        overlapping = [word for word in words if word.start < right and word.end > left]
        if overlapping:
            left = max(0.0, min(left, min(word.start for word in overlapping)))
            right = min(duration_s, max(right, max(word.end for word in overlapping)))
        ranges.append((left, right))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def merge(
    words: Sequence[Word], window: tuple[float, float], clip_words: Sequence[Word],
) -> tuple[list[Word], int]:
    """Keep original words while inserting safe words from a retry clip."""
    start, end = window
    kept = list(words)
    shifted = [
        Word(word.text, word.start + start, word.end + start, word.conf)
        for word in clip_words
        if math.isfinite(word.start) and math.isfinite(word.end) and word.end > word.start
        and start <= start + (word.start + word.end) / 2 <= end
    ]
    old_keys = [
        (word, min(word.end, word.start + _allowance(word)), tuple(norm_words(word.text)))
        for word in words
    ]
    candidates = []
    for word in shifted:
        key = tuple(norm_words(word.text))
        if any(
            key == old_key
            and (
                word.start < old_end and word.end > old.start
                or abs((word.start + word.end - old.start - old_end) / 2) <= MATCH_MIDPOINT_S
            )
            for old, old_end, old_key in old_keys
        ):
            continue
        candidates.append(word)
    added: list[Word] = []
    for candidate in candidates:
        overlaps = [
            (index, word) for index, word in enumerate(kept)
            if candidate.start < word.end and candidate.end > word.start
        ]
        if any(
            word.end - word.start <= _allowance(word)
            or candidate.start - word.start < 0.05
            for _, word in overlaps
        ):
            continue
        for index, word in overlaps:
            kept[index] = Word(word.text, word.start, candidate.start, word.conf)
        added.append(candidate)
    return sorted([*kept, *added], key=lambda word: (word.start, word.end)), len(added)
