"""Resolve output-time anchors without running a project stage."""

from __future__ import annotations

import math
from fractions import Fraction

from yap_studio.errors import ValidationError
from yap_studio.project.model import Anchor, CutWords


def resolve_anchor(anchor: Anchor, cut_words: CutWords, *, start: bool, location: str) -> float:
    """Resolve a kept anchor to output seconds."""
    return float(resolve_anchor_time(anchor, cut_words, start=start, location=location))


def resolve_anchor_time(
    anchor: Anchor, cut_words: CutWords, *, start: bool, location: str,
) -> Fraction:
    """Keep word edges and offsets exact until they reach the frame grid.

    The caller parses default edges (start for starts, end for ends). This
    resolver preserves the parsed edge, including explicit edge overrides.
    """
    if anchor.snap is not None:
        raise ValidationError(f"{location}.snap", "snap needs the timeline stage")
    if anchor.time is not None:
        return Fraction(str(anchor.time))
    occurrences = [word for word in cut_words.words if word.id == anchor.word]
    if not occurrences:
        take, index = str(anchor.word).split(":")
        index = int(index)
        kept = {
            int(word.id.split(":")[1]): word
            for word in cut_words.words
            if word.id.split(":")[0] == take
        }
        before = max((i for i in kept if i < index), default=None)
        after = min((i for i in kept if i > index), default=None)

        def describe(index: int | None) -> str:
            return "none" if index is None else f"{kept[index].id} ({kept[index].text!r})"

        raise ValidationError(
            f"{location}.word",
            f"word {anchor.word!r} was removed from the cut; nearest kept words in {take}: "
            f"before {describe(before)}, after {describe(after)}",
        )
    if anchor.segment is not None:
        occurrences = [word for word in occurrences if word.segment == anchor.segment]
        if not occurrences:
            raise ValidationError(
                f"{location}.segment",
                f"word {anchor.word!r} has no occurrence in segment {anchor.segment!r}",
            )
    if len(occurrences) > 1:
        segments = ", ".join(repr(word.segment) for word in occurrences)
        raise ValidationError(
            f"{location}.word",
            f"word {anchor.word!r} is ambiguous in segments {segments}; specify segment",
        )
    word = occurrences[0]
    if not anchor.matches(word.text):
        raise ValidationError(
            f"{location}.text",
            f"guard {anchor.text!r} does not match word {anchor.word!r}: {word.text!r}",
        )
    return (Fraction(str(word.start if anchor.edge == "start" else word.end))
            + Fraction(str(anchor.offset_s)))


def frame_at(seconds: float, fps: int) -> int:
    """Use the verify rule: floor(t * fps + 1/2), including ties."""
    return math.floor(seconds * fps + 0.5)
