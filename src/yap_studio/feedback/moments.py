"""Map review moments through source word occurrences across immutable cuts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from yap_studio.project.model import CutWords


@dataclass(frozen=True)
class Occurrence:
    """One source word in a particular segment, placed on output time."""

    id: str
    segment: str
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class MomentMap:
    """A destination and the occurrence evidence used to choose it."""

    status: Literal["mapped", "moved", "ambiguous", "removed", "time"]
    time_s: float
    word: Occurrence | None
    target: Occurrence | None
    candidates: tuple[Occurrence, ...]

    def to_dict(self) -> dict:
        """Return JSON-ready occurrence records and an ordered candidate list."""
        return {
            "status": self.status,
            "time_s": self.time_s,
            "word": asdict(self.word) if self.word is not None else None,
            "target": asdict(self.target) if self.target is not None else None,
            "candidates": [asdict(item) for item in self.candidates],
        }


def occurrences(words: CutWords) -> tuple[Occurrence, ...]:
    """Return occurrences in the cut's output order."""
    return tuple(Occurrence(w.id, w.segment, w.text, w.start, w.end) for w in words.words)


def occurrence_at(words: CutWords, t: float) -> tuple[Occurrence | None, float, str]:
    """Anchor inside speech, after speech in silence, or before the first word."""
    items = occurrences(words)
    if not items:
        return None, 0.0, "none"
    for item in items:
        if item.start <= t < item.end:
            return item, t - item.start, "inside"
    for item in reversed(items):
        if item.end <= t:
            return item, t - item.end, "after"
    return items[0], t - items[0].start, "before"


def _match(anchor: Occurrence, targets: tuple[Occurrence, ...]) -> Occurrence | None:
    exact = next(
        (item for item in targets if (item.id, item.segment) == (anchor.id, anchor.segment)),
        None,
    )
    if exact is not None:
        return exact
    same_id = [item for item in targets if item.id == anchor.id]
    return same_id[0] if len(same_id) == 1 else None


def map_moment(
    source: CutWords | None,
    target: CutWords | None,
    t: float,
    *,
    target_duration: float | None = None,
) -> MomentMap:
    """Map a moment by word id and segment, reporting deletion or ambiguity."""
    def clamp(value: float) -> float:
        value = max(0.0, value)
        return min(value, target_duration) if target_duration is not None else value

    if source is None or target is None or not source.words or not target.words:
        return MomentMap("time", clamp(t), None, None, ())
    sources, targets = occurrences(source), occurrences(target)
    anchor, offset, where = occurrence_at(source, t)
    assert anchor is not None
    destination = _match(anchor, targets)
    if destination is None:
        candidates = tuple(item for item in targets if item.id == anchor.id)
        if candidates:
            return MomentMap("ambiguous", clamp(candidates[0].start), anchor, None, candidates)
        position = sources.index(anchor)
        for item in sources[position + 1:]:
            found = _match(item, targets)
            if found is not None:
                return MomentMap("removed", clamp(found.start), anchor, found, ())
        for item in reversed(sources[:position]):
            found = _match(item, targets)
            if found is not None:
                return MomentMap("removed", clamp(found.end), anchor, found, ())
        return MomentMap("removed", 0.0, anchor, None, ())

    status = "mapped" if destination.segment == anchor.segment else "moved"
    if where == "inside":
        time_s = min(destination.start + offset, math.nextafter(destination.end, -math.inf))
        time_s = max(destination.start, time_s)
    elif where == "after":
        time_s = destination.end + offset
        position = targets.index(destination)
        if position + 1 < len(targets):
            time_s = min(time_s, math.nextafter(targets[position + 1].start, -math.inf))
    else:
        time_s = max(0.0, destination.start + offset)
    return MomentMap(status, clamp(time_s), anchor, destination, ())
