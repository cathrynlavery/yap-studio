"""Metadata-only source anchor geometry shared by freshness checks and cut rendering."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from yap_studio.plan import snap_frame
from yap_studio.project.model import (
    Anchor,
    CutWord,
    CutWords,
    Edit,
    ProjectFileError,
    Segment,
    Words,
)
from yap_studio.transcript.words import norm_words

CONTINUOUS_GAP_S = 1.0
START_PAD_S = Fraction(12, 100)
END_PAD_S = Fraction(20, 100)
MATCH_WORDS = 2


@dataclass(frozen=True)
class ResolvedAnchor:
    """One anchor on the output grid, with an explicit record of any snap."""

    frame: int
    word: str | None = None
    segment: str | None = None
    snapped_from: str | None = None


def resolve_anchor(
    anchor: Anchor, cut_words: CutWords, source_words: dict[str, Words], edit: Edit,
) -> ResolvedAnchor:
    """Resolve through cut words; removed words may snap only within their story segment."""
    if anchor.time is not None:
        return ResolvedAnchor(snap_frame(Fraction(str(anchor.time)), Fraction(30)))
    source = source_words.get(cast(str, anchor.word).split(":")[0])
    original = next((word for word in source.words if word.id == anchor.word), None) \
        if source is not None else None
    if original is None:
        raise ProjectFileError("timeline", "anchor", f"missing source word {anchor.word!r}")
    if not anchor.matches(original.text):
        raise ProjectFileError(
            "timeline", "anchor",
            f"guard {anchor.text!r} disagrees with source word {original.text!r}",
        )
    occurrences = [word for word in cut_words.words if word.id == anchor.word
                   and (anchor.segment is None or word.segment == anchor.segment)]
    if len(occurrences) > 1:
        segments = ", ".join(word.segment for word in occurrences)
        raise ProjectFileError(
            "timeline", "anchor",
            f"word {anchor.word!r} is ambiguous in segments {segments}; specify segment",
        )
    snapped_from = None
    if not occurrences:
        segments = []
        for segment in edit.segments:
            if segment.take != source.take or (anchor.segment is not None
                                              and segment.id != anchor.segment):
                continue
            first = nominal_edge(segment, segment.from_, source, start=True)
            last = nominal_edge(segment, segment.to, source, start=False)
            if Fraction(str(original.end)) > first and Fraction(str(original.start)) < last:
                segments.append(segment.id)
        if len(segments) > 1:
            raise ProjectFileError(
                "timeline", "anchor",
                f"removed word {anchor.word!r} is ambiguous in segments "
                f"{', '.join(segments)}; specify segment",
            )
        by_id = {word.id: word for word in source.words}
        kept = [word for word in cut_words.words
                if word.segment in segments and word.id in by_id]
        before = [word for word in kept if by_id[word.id].start < original.start]
        after = [word for word in kept if by_id[word.id].start > original.start]
        left = max(before, key=lambda word: by_id[word.id].start, default=None)
        right = min(after, key=lambda word: by_id[word.id].start, default=None)
        neighbours = [word for word in (left, right) if word is not None]
        names = ", ".join(f"{word.id} ({word.text!r}, segment {word.segment})"
                          for word in neighbours) or "none in the same segment"
        if anchor.snap != "nearest" or not neighbours:
            raise ProjectFileError(
                "timeline", "anchor", f"removed word {anchor.word!r}; nearest kept words: {names}",
            )
        occurrences = [min(neighbours, key=lambda word: abs(
            Fraction(str(by_id[word.id].start)) - Fraction(str(original.start)),
        ))]
        snapped_from = anchor.word
    occurrence: CutWord = occurrences[0]
    point = Fraction(str(occurrence.start if anchor.edge == "start" else occurrence.end))
    point += Fraction(str(anchor.offset_s))
    return ResolvedAnchor(snap_frame(point, Fraction(30)), occurrence.id,
                          occurrence.segment, snapped_from)


def edge_is_exact(anchor: Anchor, *, start: bool) -> bool:
    """Source times and explicit word-edge overrides need no speech confirmation."""
    return anchor.word is None or bool(anchor.offset_s) or anchor.edge != (
        "start" if start else "end"
    )


def nominal_edge(
    segment: Segment,
    anchor: Anchor,
    words: Words,
    duration: Fraction | None = None,
    *,
    start: bool,
) -> Fraction:
    """Place a nominal edge from metadata, optionally checking the picture duration.

    Without a duration, the final word gets its full end pad. Continuous joins
    lie between adjacent words, which bound the pads on both sides of the join.
    Their shared point stays inside the source and both valid segments, so
    clamping an outer final-word pad to the picture duration cannot change a join.
    """
    location = f"segment {segment.id!r}.{'from' if start else 'to'}"
    if anchor.word is not None:
        word = next((word for word in words.words if word.id == anchor.word), None)
        if word is None:
            raise ProjectFileError("edit.json", location, f"missing word {anchor.word!r}")
        if not anchor.matches(word.text):
            raise ProjectFileError(
                "edit.json",
                location,
                f"guard {anchor.text!r} disagrees with source word {word.text!r}",
            )
        if duration is not None and Fraction(str(word.start)) >= duration:
            raise ProjectFileError("edit.json", location, "anchor is outside the source picture")
        point = Fraction(str(word.start if anchor.edge == "start" else word.end))
        point += Fraction(str(anchor.offset_s))
        if (duration is not None and point > duration and not anchor.offset_s
                and 0 <= Fraction(str(word.start)) < duration):
            point = duration
    else:
        point = Fraction(str(cast(float, anchor.time)))
    if point < 0 or (duration is not None and point > duration):
        raise ProjectFileError("edit.json", location, "anchor is outside the source picture")
    if edge_is_exact(anchor, start=start):
        return point
    ordered = sorted(words.words, key=lambda word: (word.start, word.end))
    for index, word in enumerate(ordered):
        first, last = Fraction(str(word.start)), Fraction(str(word.end))
        if start and first <= point < last:
            previous = max((Fraction(str(item.end)) for item in ordered[:index]),
                           default=Fraction(0))
            gap = max(Fraction(0), first - previous)
            return max(Fraction(0), first - min(gap / 2, START_PAD_S))
        if not start and first < point <= last:
            # Without picture metadata, allow the full final-word pad. Continuous
            # joins lie between source words, so this cannot change a valid join.
            following = min(
                (Fraction(str(item.start)) for item in ordered[index + 1 :]),
                default=duration if duration is not None else last + 2 * END_PAD_S,
            )
            gap = max(Fraction(0), following - last)
            padded = last + min(gap / 2, END_PAD_S)
            return min(duration, padded) if duration is not None else padded
    return point


def edge_words(words: Words, anchor: Anchor, *, start: bool) -> tuple[str, ...]:
    """Normalised words a word-anchored edge must start or end on."""
    ordered = sorted(words.words, key=lambda word: (word.start, word.end))
    index = next((i for i, word in enumerate(ordered) if word.id == anchor.word), None)
    if index is None or not norm_words(ordered[index].text):
        return ()
    tokens: list[str] = []
    step = 1 if start else -1
    while 0 <= index < len(ordered) and len(tokens) < MATCH_WORDS:
        found = norm_words(ordered[index].text)
        tokens = tokens + found if start else found + tokens
        index += step
    return tuple(tokens)


def continues(left: Segment, right: Segment, words: Words, warnings: list[str]) -> bool:
    """Join neighbouring words only when their gap is short and neither is overridden."""
    if left.take != right.take or left.to.word is None or right.from_.word is None:
        return False
    if edge_is_exact(left.to, start=False) or edge_is_exact(right.from_, start=True):
        return False
    ordered = sorted(words.words, key=lambda word: (word.start, word.end))
    ids = [word.id for word in ordered]
    try:
        index = ids.index(left.to.word)
    except ValueError:
        return False
    if index + 1 >= len(ids) or ids[index + 1] != right.from_.word:
        return False
    gap = ordered[index + 1].start - ordered[index].end
    if gap >= CONTINUOUS_GAP_S:
        warnings.append(
            f"Segments {left.id!r} and {right.id!r} use neighbouring words {gap:.1f} s apart; "
            "the transcript may be missing words there, so the gap is cut."
        )
        return False
    return True


def join_edges(
    segments: tuple[Segment, ...],
    source_words: dict[str, Words],
    nominal: dict[str, tuple[Fraction, Fraction]],
    warnings: list[str],
) -> set[tuple[str, bool]]:
    """Share valid continuous boundaries and identify the edges requiring no check."""
    joined: set[tuple[str, bool]] = set()
    for left, right in zip(segments, segments[1:], strict=False):
        if continues(left, right, source_words[left.take], warnings):
            point = (nominal[left.id][1] + nominal[right.id][0]) / 2
            if not nominal[left.id][0] < point < nominal[right.id][1]:
                continue
            nominal[left.id] = (nominal[left.id][0], point)
            nominal[right.id] = (point, nominal[right.id][1])
            joined |= {(left.id, False), (right.id, True)}
    return joined


def edit_needs_edge_engine(
    segments: tuple[Segment, ...],
    source_words: dict[str, Words],
    *,
    joined: set[tuple[str, bool]] | None = None,
) -> bool:
    """Whether any unjoined, non-exact edge has words to confirm, using only metadata.

    The renderer can reuse joins already calculated with checked picture durations.
    Status computes the same joins from the edit and source words without probing media.
    """
    if joined is None:
        nominal = {
            segment.id: (
                nominal_edge(segment, segment.from_, source_words[segment.take], start=True),
                nominal_edge(segment, segment.to, source_words[segment.take], start=False),
            )
            for segment in segments
        }
        joined = join_edges(segments, source_words, nominal, [])
    return any(
        (segment.id, start) not in joined
        and not edge_is_exact(anchor, start=start)
        and bool(edge_words(source_words[segment.take], anchor, start=start))
        for segment in segments
        for anchor, start in ((segment.from_, True), (segment.to, False))
    )
