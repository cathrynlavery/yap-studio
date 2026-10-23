"""Small word leads for shots that show the named thing."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from yap_studio.media.select import content_terms
from yap_studio.planner.anchors import frame_at
from yap_studio.project.model import CutWords


def lead_named_start(element: dict, words: CutWords, planned: Sequence[dict] = (),
                     *, end_s: float, max_frames: int | None = None, fps: int = 30) -> None:
    """Lead a matching word within prior-shot and source limits; preserve authored offsets."""
    anchor = element.get("start")
    if (not isinstance(anchor, dict) or "word" not in anchor or "offset_s" in anchor
            or anchor.get("edge", "start") != "start"
            or not isinstance(element.get("matches"), str)):
        return
    occurrences = [word for word in words.words if word.id == anchor["word"]
                   and ("segment" not in anchor or word.segment == anchor["segment"])]
    if len(occurrences) != 1:
        return
    word = occurrences[0]
    if word.start >= 0.1 and set(content_terms(word.text)).intersection(
        content_terms(element["matches"]),
    ):
        word_start = Fraction(str(word.start))
        start = word_start - Fraction(1, 10)
        for previous in planned:
            if (Fraction(previous["start_frame"], fps) < word_start
                    and Fraction(previous["end_frame"], fps) > start):
                start = min(word_start, Fraction(previous["end_frame"], fps))
        while start < word_start:
            offset = float(start - word_start)
            # Match anchor resolution and window validation, including float rounding.
            start_s = word.start + offset
            end = end_s if "end" in element else start_s + element["hold_s"]
            first, last = frame_at(start_s, fps), frame_at(end, fps)
            if first < last and (max_frames is None or last - first <= max_frames):
                anchor["offset_s"] = offset
                return
            # Try the next frame center, dropping the lead if none fits before the word.
            start = Fraction(first + 1, fps)
