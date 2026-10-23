"""Pure pacing and named-word timing warnings for resolved B-roll."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

from yap_studio.media.select import content_terms
from yap_studio.project.model import CutWords, Take

if TYPE_CHECKING:
    from yap_studio.stages.timeline import ResolvedElement

FPS = 30
# Three starts per minute leave room for the creator between supporting shots.
MAX_STARTS_PER_MINUTE = 3
# Full-screen coverage beyond a quarter makes the supporting footage dominate.
MAX_FULL_SCREEN_PERCENT = 25
# Keep the creator visible for most of a talking-head video.
MIN_FACE_PERCENT = 65
# Shorter standalone inserts are hard to read before the next cut.
MIN_INSERT_S = Fraction(6, 5)
# A named thing should appear with the word rather than visibly after it.
MAX_LATE_S = Fraction(15, 100)


def _covered(elements: Sequence[ResolvedElement], total: int) -> int:
    windows = sorted((max(0, item.start_frame), min(total, item.end_frame)) for item in elements)
    covered, end = 0, 0
    for first, last in windows:
        covered += max(0, last - max(end, first))
        end = max(end, last)
    return covered


def _insert_units(inserts: Sequence[ResolvedElement]) -> list[list[ResolvedElement]]:
    """Count each contiguous montage run at its own start."""
    units: list[list[ResolvedElement]] = []
    groups: dict[tuple[str, str], list[ResolvedElement]] = {}
    for item in inserts:
        group = item.payload.get("group")
        key = (item.lane, group)
        run = groups.get(key) if group else None
        if run is not None and item.start_frame <= max(part.end_frame for part in run) + 1:
            run.append(item)
        else:
            run = [item]
            units.append(run)
            if group:
                groups[key] = run
    return units


def broll_limits(
    elements: Sequence[ResolvedElement], total_frames: int, words: CutWords,
    takes: Sequence[Take], finding: Callable[..., None], *,
    insert_length_ids: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Report guidance without changing the validity of a timeline."""
    inserts = sorted((item for item in elements if item.kind == "insert"),
                     key=lambda item: (item.start_frame, item.id))
    if total_frames > 0 and not (takes and all(take.voiceover for take in takes)):
        units = _insert_units(inserts)
        densest: list[list[ResolvedElement]] = []
        end = 0
        window = min(total_frames, 60 * FPS)
        for index, unit in enumerate(units):
            while end < len(units) and units[end][0].start_frame < unit[0].start_frame + window:
                end += 1
            if end - index > len(densest):
                densest = units[index:end]
        if len(densest) > MAX_STARTS_PER_MINUTE:
            finding("broll_rate", f"{len(densest)} insert starts in one minute window",
                    *(item for unit in densest for item in unit), severity="warning",
                    suggested_action="Drop or merge inserts so no minute has more than 3.")
        full = [item for item in inserts if item.payload.get("media", {}).get("fit") != "split"]
        full_frames = _covered(full, total_frames)
        action = "Shorten inserts, or use split or an overlay for some."
        if full_frames * 100 > MAX_FULL_SCREEN_PERCENT * total_frames:
            finding("broll_full_screen",
                    f"full-screen inserts cover {100 * full_frames / total_frames:.1f}% of runtime",
                    *full, severity="warning", suggested_action=action)
        absent = [*full, *(item for item in elements if item.kind == "transition")]
        face_frames = total_frames - _covered(absent, total_frames)
        if face_frames * 100 < MIN_FACE_PERCENT * total_frames:
            finding("face_on_screen",
                    f"her face shows for {100 * face_frames / total_frames:.1f}% of runtime",
                    *absent, severity="warning", suggested_action=action)
    for item in inserts:
        if (not item.payload.get("group") and item.id not in insert_length_ids
                and Fraction(item.end_frame - item.start_frame, FPS) < MIN_INSERT_S):
            finding("insert_short", "standalone insert is shorter than 1.2 s", item,
                    severity="warning", suggested_action="Hold this insert for at least 1.2 s.")
    for item in elements:
        if item.kind not in {"insert", "overlay"} or not item.payload.get("matches"):
            continue
        terms = set(content_terms(item.payload["matches"]))
        start, end_s = Fraction(item.start_frame, FPS), Fraction(item.end_frame, FPS)
        word = next((word for word in sorted(words.words, key=lambda word: word.start)
                     if start - 1 <= Fraction(str(word.start)) < end_s
                     and terms.intersection(content_terms(word.text))), None)
        if word is not None:
            gap = start - Fraction(str(word.start))
            if gap > MAX_LATE_S:
                finding("broll_late", f"starts {float(gap):.2f} s after {word.text!r}", item,
                        severity="warning", suggested_action=(
                            f"Anchor the start on {word.text!r} with offset_s -0.1."))
