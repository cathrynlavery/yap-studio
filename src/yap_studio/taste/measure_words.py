"""Recover caption windows from glyph ink and identify limits from timed word partitions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.style.loader import resolve_style_data
from yap_studio.taste.frames import FrameTiming, color_mask, iter_rgb, rgb, runs
from yap_studio.taste.measure_text import glyph_mask, stroke_mask
from yap_studio.transcript.model import Transcript

LIMIT_FIELDS = ("captions.max_words_per_phrase", "captions.max_characters_per_phrase")
NEEDS_WORDS = "Needs word timings: a speech engine or a words file for this video."
ENTRY_TOLERANCE_S = 0.15
Observation = tuple[Any, list[int], dict[str, Any]]
Layout = tuple[tuple[int, int], ...]


def _ink_layout(
    image: np.ndarray, band: Mapping[str, int], colors: Mapping[str, str]
) -> tuple[int, Layout]:
    """Measure glyph mass and horizontal line extents, excluding highlight decorations."""
    scale = image.shape[1] / 1080
    top = max(0, round((band["top_px"] - 24) * scale))
    bottom = min(image.shape[0], round((band["bottom_px"] + 24) * scale))
    crop = image[top:bottom]
    if not crop.size:
        return 0, ()
    strokes = stroke_mask(crop)
    ink = glyph_mask(crop, rgb(colors["captions.text_color"]), 24, strokes)
    style = colors.get("captions.highlight_style", "none")
    if style == "color" and "captions.highlight_color" in colors:
        ink |= glyph_mask(crop, rgb(colors["captions.highlight_color"]), 24, strokes)
    if style == "moving_block" and "captions.highlight_text_color" in colors:
        active = glyph_mask(crop, rgb(colors["captions.highlight_text_color"]), 24, strokes)
        if "captions.highlight_color" in colors:
            block = color_mask(crop, rgb(colors["captions.highlight_color"]), 24)
            near = block.copy()
            # Restrict dark glyphs to the block; dark picture texture is not active text.
            for distance in range(1, 11):
                near |= np.roll(block, distance, axis=1) | np.roll(block, -distance, axis=1)
            active &= near
        ink |= active
    profile = ink.sum(axis=1)
    lines = []
    for first, last in runs(profile > max(2, profile.max(initial=0) * 0.06), gap=2):
        line = ink[first:last]
        mass = int(line.sum())
        height = last - first
        if height < 3 or mass < 12:
            continue
        columns = np.flatnonzero(line.sum(axis=0) >= 2)
        if len(columns):
            lines.append(((int(columns[0]), int(columns[-1])), mass, height))
    largest = max((mass for _, mass, _ in lines), default=0)
    tallest = max((height for _, _, height in lines), default=0)
    # Picture texture and decoration fragments can share the caption colour.
    # Real caption lines have comparable ink mass and glyph height.
    lines = [
        line for line in lines if line[1] >= largest * 0.2 and line[2] >= tallest * 0.4
    ]
    return sum(mass for _, mass, _ in lines), tuple(layout for layout, _, _ in lines)


def _layout_changed(first: Layout, second: Layout, tolerance: int) -> bool:
    """Compare line count and horizontal bounds without treating entry rise as a change."""
    return len(first) != len(second) or any(
        abs(a - b) > tolerance
        for line_a, line_b in zip(first, second, strict=False)
        for a, b in zip(line_a, line_b, strict=True)
    )


def caption_windows(
    pictures: Iterable[tuple[int, np.ndarray]],
    band: Mapping[str, int],
    colors: Mapping[str, str],
    *,
    reference_frames: Sequence[int] = (),
    fps: float | None = None,
) -> list[dict[str, int]]:
    """Group ink using settled layouts; without a cadence allow three entry frames."""
    samples = []
    tolerance = 6
    for index, image in pictures:
        mass, layout = _ink_layout(image, band, colors)
        samples.append((index, mass, layout))
        tolerance = max(2, round(image.shape[1] / 60))
    references = set(reference_frames)
    masses = [mass for index, mass, _ in samples if index in references and mass]
    if not masses:
        masses = [mass for _, mass, _ in samples if mass]
        typical = float(np.median(masses)) if masses else 0
        masses = [mass for mass in masses if mass >= typical]
    if not masses:
        return []
    threshold = max(12, float(np.median(masses)) * 0.15)
    entry_frames = max(3, math.ceil(ENTRY_TOLERANCE_S * (fps or 0)))
    windows: list[dict[str, int]] = []
    previous = -2
    layout: Layout = ()
    pending: tuple[int, Layout, int] | None = None
    for index, mass, current in samples:
        if mass < threshold or not current:
            previous = -2
            pending = None
            continue
        if index != previous + 1:
            windows.append({"start_frame": index, "end_frame": index})
            layout, pending = current, None
        elif index - windows[-1]["start_frame"] < entry_frames:
            # Faint/rising entry ink can initially contain only part of a line.
            # Keep replacing the reference until the entry period has settled.
            layout, pending = current, None
            windows[-1]["end_frame"] = index
        elif _layout_changed(layout, current, tolerance):
            windows[-1]["end_frame"] = index
            if pending and not _layout_changed(pending[1], current, tolerance):
                pending = pending[0], pending[1], pending[2] + 1
                if pending[2] >= 3:
                    windows[-1]["end_frame"] = pending[0] - 1
                    windows.append({"start_frame": pending[0], "end_frame": index})
                    layout, pending = current, None
            else:
                pending = index, current, 1
        else:
            pending = None
            windows[-1]["end_frame"] = index
        previous = index
    # Fold short fragments forward only when no blank or missing frame separates
    # them. Work backwards so chains of fragments share the following window.
    for position in range(len(windows) - 2, -1, -1):
        window, following = windows[position : position + 2]
        if (
            window["end_frame"] - window["start_frame"] + 1 < entry_frames
            and window["end_frame"] + 1 == following["start_frame"]
        ):
            following["start_frame"] = window["start_frame"]
            del windows[position]
    return windows


def assign_words(
    windows: Sequence[Mapping[str, int]], transcript: Transcript, timing: FrameTiming
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    """Assign word starts to displayed windows, allowing a short entry detection delay."""
    result = [{**window, "words": 0} for window in windows]
    intervals = [
        (timing.boundary(window["start_frame"]), timing.boundary(window["end_frame"] + 1))
        for window in windows
    ]
    last_frame_durations = [
        timing.window_s(window["end_frame"], window["end_frame"])
        for window in windows
    ]
    outside = []
    for index, word in enumerate(sorted(transcript.words, key=lambda word: word.start)):
        match = next(
            (i for i, (start, end) in enumerate(intervals) if start <= word.start < end), None
        )
        if (
            match is not None
            and match + 1 < len(intervals)
            and intervals[match][1] - word.start <= last_frame_durations[match] + 1e-9
            and 0 <= intervals[match + 1][0] - word.start <= ENTRY_TOLERANCE_S + 1e-9
        ):
            # Entry ink can lag the next phrase's first word by a frame or two.
            # Reconsider only starts in the previous caption's last displayed frame.
            match += 1
        if match is None:
            match = next(
                (
                    i
                    for i, (start, _) in enumerate(intervals)
                    if 0 <= start - word.start <= ENTRY_TOLERANCE_S + 1e-9
                ),
                None,
            )
        if match is None:
            outside.append({"index": index, "text": word.text, "start_s": word.start})
        else:
            result[match]["words"] += 1
    return result, outside


def _ignore_empty_windows(
    windows: list[dict[str, int]],
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    """Merge ink without word starts into touching phrases, dropping isolated ink."""
    windows = [window.copy() for window in windows]
    ignored = []
    position = 0
    while position < len(windows):
        window = windows[position]
        if window["words"]:
            position += 1
            continue
        action = "dropped"
        if position and windows[position - 1]["end_frame"] + 1 == window["start_frame"]:
            windows[position - 1]["end_frame"] = window["end_frame"]
            action = "merged_previous"
        elif (
            position + 1 < len(windows)
            and window["end_frame"] + 1 == windows[position + 1]["start_frame"]
        ):
            windows[position + 1]["start_frame"] = window["start_frame"]
            action = "merged_next"
        ignored.append({**window, "reason": "No word starts in this window.", "action": action})
        del windows[position]
    return windows, ignored


def consistent_limits(transcript: Transcript, counts: Sequence[int]) -> list[tuple[int, int]]:
    """Find every schema limit pair that reproduces the observed phrase word counts."""
    words = [
        CaptionWord(f"w{i}", word.text, word.start, word.end)
        for i, word in enumerate(sorted(transcript.words, key=lambda word: word.start))
    ]
    defaults = resolve_style_data({"schema_version": 1})["captions"]
    pairs = []
    for max_words in range(1, 9):
        # A character limit can change splitting only when it admits another word span.
        thresholds = {8}
        for first in range(len(words)):
            characters = len(words[first].text)
            for last in range(first + 1, min(len(words), first + max_words)):
                characters += 1 + len(words[last].text)
                if characters > 48:
                    break
                if characters >= 8:
                    thresholds.add(characters)
        matches = False
        for max_chars in range(8, 49):
            if max_chars in thresholds:
                captions = {
                    **defaults,
                    "max_words_per_phrase": max_words,
                    "max_characters_per_phrase": max_chars,
                }
                matches = [len(phrase.words) for phrase in build_phrases(words, captions)] == list(
                    counts
                )
            if matches:
                pairs.append((max_words, max_chars))
    return pairs


def _range(values: Sequence[int]) -> str:
    """Describe contiguous candidate values as a range and preserve any holes."""
    values = sorted(set(values))
    if values == list(range(values[0], values[-1] + 1)):
        return str(values[0]) if len(values) == 1 else f"{values[0]} to {values[-1]}"
    return ", ".join(str(value) for value in values)


def infer_limits(
    transcript: Transcript, counts: Sequence[int], evidence: Sequence[int]
) -> tuple[dict[str, Observation], list[dict[str, Any]]]:
    """Propose only uniquely identified limits, retaining ambiguous candidates in detail."""
    pairs = consistent_limits(transcript, counts)
    word_limits = sorted({words for words, _ in pairs})
    char_limits = sorted({characters for _, characters in pairs})
    detail = {
        "observed_counts": list(counts),
        "consistent_max_words": word_limits,
        "consistent_max_chars": char_limits,
        "consistent_max_chars_range": [min(char_limits), max(char_limits)] if char_limits else [],
    }
    fields: dict[str, Observation] = {}
    missing = []
    if not pairs:
        reason = "Observed caption word counts do not match any phrase limits in the schema."
        return {}, [{"field": field, "reason": reason, "detail": detail} for field in LIMIT_FIELDS]
    if len(word_limits) == 1:
        fields[LIMIT_FIELDS[0]] = word_limits[0], list(evidence), detail
    else:
        missing.append(
            {
                "field": LIMIT_FIELDS[0],
                "reason": f"Consistent with {_range(word_limits)} words; "
                "no caption was limited by its word count.",
                "detail": detail,
            }
        )
    if len(char_limits) == 1:
        fields[LIMIT_FIELDS[1]] = char_limits[0], list(evidence), detail
    else:
        missing.append(
            {
                "field": LIMIT_FIELDS[1],
                "reason": f"Consistent with {_range(char_limits)} characters; "
                "the partition does not identify an exact character limit.",
                "detail": detail,
            }
        )
    return fields, missing


def measure(
    path: Path,
    info: ffmpeg.MediaInfo,
    fields: Mapping[str, Observation],
    measures: Mapping[str, Observation],
    transcript: Transcript | None,
    *,
    timing: FrameTiming,
) -> tuple[dict[str, Observation], dict[str, Observation], list[dict[str, Any]]]:
    """Measure caption partitions and limit evidence using one final-timeline transcript."""
    if transcript is None:
        return {}, {}, [{"field": field, "reason": NEEDS_WORDS} for field in LIMIT_FIELDS]
    band = measures.get("captions.band")
    if band is None or "captions.text_color" not in fields:
        reason = "No measured caption band and text colour to identify caption windows."
        return {}, {}, [{"field": field, "reason": reason} for field in LIMIT_FIELDS]
    colors = {key: row[0] for key, row in fields.items() if key.startswith("captions.")}
    windows = caption_windows(
        iter_rgb(path, info, range(len(timing.starts)), width=360),
        band[0],
        colors,
        reference_frames=band[1],
        fps=float(getattr(info, "fps", None) or len(timing.starts) / timing.end_s),
    )
    if not windows:
        reason = "No caption ink windows were found in the measured band."
        return {}, {}, [{"field": field, "reason": reason} for field in LIMIT_FIELDS]
    windows, outside = assign_words(windows, transcript, timing)
    windows, ignored = _ignore_empty_windows(windows)
    counts = [window["words"] for window in windows]
    evidence = [
        index
        for window in windows
        for index in range(window["start_frame"], window["end_frame"] + 1)
    ]
    detail = {"observed_counts": counts, "outside_words": outside, "ignored_windows": ignored}
    observations = {
        "captions.words_per_caption": ({"counts": counts, "windows": windows}, evidence, detail)
    }
    if outside:
        word = outside[0]
        reason = (
            f"Word {word['index'] + 1} ({word['text']!r}) at {word['start_s']:g} s "
            "is outside every caption window."
        )
        return (
            {},
            observations,
            [{"field": field, "reason": reason, "detail": detail} for field in LIMIT_FIELDS],
        )
    proposed, missing = infer_limits(transcript, counts, evidence)
    return proposed, observations, missing
