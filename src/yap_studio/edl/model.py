"""Exact edit-list model shared by all readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from yap_studio.errors import EditListError

EditListFormat = Literal["yap", "ranges", "fcpxml", "xmeml"]


def _fraction_fields(name: str, value: Fraction) -> dict[str, str | float]:
    return {name: str(value), f"{name}_s": float(value)}


@dataclass(frozen=True)
class Source:
    """A source referenced by an edit list."""

    id: str
    path: Path | None
    audio_stream: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-compatible source representation."""
        return {
            "id": self.id, "path": str(self.path) if self.path is not None else None,
            "audio_stream": self.audio_stream,
        }


@dataclass(frozen=True)
class Segment:
    """One source range in output order, or a gap, with optional edge word intent."""

    index: int
    source_id: str | None
    start: Fraction
    end: Fraction
    label: str | None = None
    output_start: Fraction | None = None
    first_word: str | None = None
    last_word: str | None = None

    def __post_init__(self) -> None:
        for name in ("first_word", "last_word"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise EditListError(
                    f"Segment {self.index} {name} must be a non-empty string.",
                    suggested_action=f"Set {name} to the intended spoken word or omit it.",
                )

    @property
    def is_gap(self) -> bool:
        """Return whether the segment has no source."""
        return self.source_id is None

    @property
    def duration(self) -> Fraction:
        """Return the exact segment duration."""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible segment representation."""
        result: dict[str, Any] = {
            "index": self.index,
            "source_id": self.source_id,
            "label": self.label,
            "is_gap": self.is_gap,
        }
        for name in ("first_word", "last_word"):
            if (value := getattr(self, name)) is not None:
                result[name] = value
        result.update(_fraction_fields("start", self.start))
        result.update(_fraction_fields("end", self.end))
        result.update(_fraction_fields("duration", self.duration))
        if self.output_start is None:
            result["output_start"] = None
            result["output_start_s"] = None
        else:
            result.update(_fraction_fields("output_start", self.output_start))
        return result


@dataclass(frozen=True)
class Cut:
    """A join between two consecutive edit-list segments."""

    index: int
    left: Segment
    right: Segment

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible cut representation."""
        return {"index": self.index, "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass(frozen=True)
class Hold:
    """A declared still stretch of the output, in output frames.

    ``end_frame`` is exclusive. ``audio`` is the planned sound under the hold;
    only ``"silence"`` is supported.
    """

    index: int
    start_frame: int
    end_frame: int
    audio: Literal["silence"] = "silence"

    @property
    def frames(self) -> int:
        """Return the number of held output frames."""
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible hold representation."""
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "audio": self.audio,
        }


@dataclass(frozen=True)
class Still:
    """An insert declared to hold its picture on output frames."""

    id: str
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "start_frame": self.start_frame,
                "end_frame": self.end_frame}


@dataclass(frozen=True)
class JoinAssertion:
    """Words that must be heard across a join in the output.

    ``out_frame`` is the first output frame after the join. The normalised words
    of ``before`` then ``after`` must be heard in a row across it. Each phrase in
    ``absent`` must not be heard anywhere in the render, and each phrase in
    ``once`` must be heard exactly once.
    """

    index: int
    out_frame: int
    before: str
    after: str
    absent: tuple[str, ...] = ()
    once: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible assertion representation."""
        return {
            "index": self.index,
            "out_frame": self.out_frame,
            "before": self.before,
            "after": self.after,
            "absent": list(self.absent),
            "once": list(self.once),
        }


@dataclass(frozen=True)
class PictureRegion:
    """An output-time window where an overlay covers one rectangle of the picture.

    ``x``, ``y``, ``w`` and ``h`` are output pixels. Picture checks leave the
    rectangle out instead of skipping the whole window.
    """

    start: Fraction
    end: Fraction
    x: float
    y: float
    w: float
    h: float
    camera_3d: bool = False

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """Return the covered rectangle as ``(x, y, w, h)``."""
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible region representation."""
        result: dict[str, Any] = {"region": [self.x, self.y, self.w, self.h]}
        if self.camera_3d:
            result["camera_3d"] = True
        result.update(_fraction_fields("start", self.start))
        result.update(_fraction_fields("end", self.end))
        return result


@dataclass(frozen=True)
class Evidence:
    """A file recording cut checks made outside Yap Studio, with its expected hash."""

    path: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible evidence representation."""
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class EditList:
    """A normalized edit list with exact source-container times.

    ``composite`` says the ranges describe an already finished render, so its
    cuts are not exposed. ``evidence`` lists files attesting cut checks made
    elsewhere.
    """

    format: EditListFormat
    path: Path
    sources: dict[str, Source]
    segments: list[Segment]
    fps: Fraction | None
    picture_masks: list[tuple[Fraction, Fraction]]
    warnings: list[str]
    holds: list[Hold] = field(default_factory=list)
    joins: list[JoinAssertion] = field(default_factory=list)
    composite: bool = False
    evidence: list[Evidence] = field(default_factory=list)
    picture_regions: list[PictureRegion] = field(default_factory=list)
    stills: list[Still] = field(default_factory=list)
    grade: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for segment in self.segments:
            if segment.end <= segment.start:
                raise EditListError(
                    f"Segment {segment.index} end must be greater than start.",
                    path=str(self.path),
                )
        ordered = sorted(self.holds, key=lambda hold: hold.start_frame)
        for hold in ordered:
            if hold.start_frame < 0 or hold.end_frame <= hold.start_frame:
                raise EditListError(
                    f"Hold {hold.index} must start at frame 0 or later and end after it starts.",
                    path=str(self.path),
                )
            if hold.audio != "silence":
                raise EditListError(
                    f"Hold {hold.index} audio must be \"silence\".", path=str(self.path)
                )
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right.start_frame < left.end_frame:
                raise EditListError(
                    f"Holds {left.index} and {right.index} overlap.", path=str(self.path)
                )
        for still in self.stills:
            if (not still.id.strip() or still.start_frame < 0
                    or still.end_frame <= still.start_frame):
                raise EditListError(
                    f"Still {still.id!r} needs an id, a non-negative start frame and a later end.",
                    path=str(self.path),
                )

    def cuts(self) -> list[Cut]:
        """Return joins between every pair of consecutive segments."""
        return [
            Cut(index=index, left=left, right=right)
            for index, (left, right) in enumerate(
                zip(self.segments, self.segments[1:], strict=False), 1
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible edit-list representation."""
        masks = []
        for start, end in self.picture_masks:
            mask: dict[str, str | float] = {}
            mask.update(_fraction_fields("start", start))
            mask.update(_fraction_fields("end", end))
            masks.append(mask)
        return {
            "format": self.format,
            "path": str(self.path),
            "sources": {source_id: source.to_dict() for source_id, source in self.sources.items()},
            "segments": [segment.to_dict() for segment in self.segments],
            "fps": str(self.fps) if self.fps is not None else None,
            "fps_float": float(self.fps) if self.fps is not None else None,
            "picture_masks": masks,
            "picture_regions": [region.to_dict() for region in self.picture_regions],
            "holds": [hold.to_dict() for hold in self.holds],
            "stills": [still.to_dict() for still in self.stills],
            "joins": [join.to_dict() for join in self.joins],
            "composite": self.composite,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
            **({"grade": dict(self.grade)} if self.grade is not None else {}),
        }
