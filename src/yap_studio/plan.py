"""Exact frame and sample plan derived from an edit list."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from yap_studio.edl.model import EditList, Still
from yap_studio.errors import ValidationError


def snap_frame(time: Fraction, fps: Fraction) -> int:
    """Snap time to the nearest frame, with exact half-frame ties upward."""
    return (time * fps + Fraction(1, 2)).numerator // (time * fps + Fraction(1, 2)).denominator


def _sample_at(frame: int, fps: Fraction, sample_rate: int) -> int:
    position = Fraction(frame * sample_rate, 1) / fps + Fraction(1, 2)
    return position.numerator // position.denominator


@dataclass(frozen=True)
class PlanSegment:
    """One edit-list segment placed on the output clock, preserving edge word intent."""

    index: int
    edl_index: int
    source_id: str | None
    src_f0: int
    src_f1: int
    n_frames: int
    out_f0: int
    out_time: Fraction
    sample_start: int
    sample_end: int
    label: str | None
    first_word: str | None = None
    last_word: str | None = None
    first_word_anchor_s: Fraction | None = None
    last_word_anchor_s: Fraction | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible planned segment."""
        return {
            "index": self.index,
            "edl_index": self.edl_index,
            "source_id": self.source_id,
            "src_f0": self.src_f0,
            "src_f1": self.src_f1,
            "n_frames": self.n_frames,
            "out_f0": self.out_f0,
            "out_time": str(self.out_time),
            "out_time_s": float(self.out_time),
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "label": self.label,
            "first_word": self.first_word,
            "last_word": self.last_word,
            **{name: float(value) for name in ("first_word_anchor_s", "last_word_anchor_s")
               if (value := getattr(self, name)) is not None},
        }


@dataclass(frozen=True)
class PlanCut:
    """A planned output cut between consecutive retained segments."""

    index: int
    left: PlanSegment
    right: PlanSegment
    out_frame: int
    out_time: Fraction
    out_sample: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible planned cut."""
        return {
            "index": self.index,
            "left_segment": self.left.index,
            "right_segment": self.right.index,
            "out_frame": self.out_frame,
            "out_time": str(self.out_time),
            "out_time_s": float(self.out_time),
            "out_sample": self.out_sample,
        }


@dataclass(frozen=True)
class PlanHold:
    """A declared still stretch placed on the output clock, with planned silence."""

    index: int
    out_f0: int
    out_f1: int
    out_time: Fraction
    end_time: Fraction
    sample_start: int
    sample_end: int
    audio: str

    @property
    def n_frames(self) -> int:
        """Return the number of held frames."""
        return self.out_f1 - self.out_f0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible planned hold."""
        return {
            "index": self.index,
            "out_f0": self.out_f0,
            "out_f1": self.out_f1,
            "out_time_s": float(self.out_time),
            "end_time_s": float(self.end_time),
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "audio": self.audio,
        }


@dataclass(frozen=True)
class Plan:
    """Exact output placement for edit-list segments, holds and cuts.

    Declared holds occupy output frames: a segment placed in sequence starts
    after any hold that covers its start, and hold frames count toward the
    planned totals.
    """

    fps: Fraction
    sample_rate: int
    segments: list[PlanSegment]
    cuts: list[PlanCut]
    total_frames: int
    total_samples: int
    warnings: list[str]
    holds: list[PlanHold] = field(default_factory=list)
    stills: list[Still] = field(default_factory=list)

    @classmethod
    def build(cls, edit_list: EditList, fps: Fraction, sample_rate: int) -> Plan:
        """Build an exact output plan from an edit list."""
        fps = Fraction(fps)
        if fps <= 0:
            raise ValidationError("fps", "must be greater than zero")
        if sample_rate <= 0:
            raise ValidationError("sample_rate", "must be greater than zero")

        warnings = list(edit_list.warnings)
        holds = sorted(edit_list.holds, key=lambda hold: hold.start_frame)
        segments: list[PlanSegment] = []
        running_frame = 0
        for segment in edit_list.segments:
            src_f0 = snap_frame(segment.start, fps)
            src_f1 = snap_frame(segment.end, fps)
            n_frames = src_f1 - src_f0
            if n_frames < 1:
                warnings.append(
                    f"Segment {segment.index} is shorter than one frame and was dropped."
                )
                continue
            if segment.output_start is not None:
                out_f0 = snap_frame(segment.output_start, fps)
            else:
                out_f0 = running_frame
                # A hold covering the next output frame pushes the segment past it.
                for hold in holds:
                    if hold.start_frame <= out_f0 < hold.end_frame:
                        out_f0 = hold.end_frame
            if segments and out_f0 < running_frame:
                warnings.append(f"Segment {segment.index} overlaps the previous segment in output.")
            for hold in holds:
                if hold.start_frame < out_f0 + n_frames and hold.end_frame > out_f0:
                    warnings.append(
                        f"Hold {hold.index} overlaps segment {segment.index} in output."
                    )
            planned = PlanSegment(
                index=len(segments) + 1,
                edl_index=segment.index,
                source_id=segment.source_id,
                src_f0=src_f0,
                src_f1=src_f1,
                n_frames=n_frames,
                out_f0=out_f0,
                out_time=Fraction(out_f0, 1) / fps,
                sample_start=_sample_at(out_f0, fps, sample_rate),
                sample_end=_sample_at(out_f0 + n_frames, fps, sample_rate),
                label=segment.label,
                first_word=segment.first_word,
                last_word=segment.last_word,
                first_word_anchor_s=segment.first_word_anchor_s,
                last_word_anchor_s=segment.last_word_anchor_s,
            )
            segments.append(planned)
            running_frame = out_f0 + n_frames

        boundaries = (
            (left, right)
            for left, right in zip(segments, segments[1:], strict=False)
            if (left.source_id is None or left.source_id != right.source_id
                or left.src_f1 != right.src_f0
                or left.out_f0 + left.n_frames != right.out_f0)
        )
        cuts = [
            PlanCut(
                index=index,
                left=left,
                right=right,
                out_frame=right.out_f0,
                out_time=right.out_time,
                out_sample=right.sample_start,
            )
            for index, (left, right) in enumerate(boundaries, 1)
        ]
        planned_holds = [
            PlanHold(
                index=hold.index,
                out_f0=hold.start_frame,
                out_f1=hold.end_frame,
                out_time=Fraction(hold.start_frame, 1) / fps,
                end_time=Fraction(hold.end_frame, 1) / fps,
                sample_start=_sample_at(hold.start_frame, fps, sample_rate),
                sample_end=_sample_at(hold.end_frame, fps, sample_rate),
                audio=hold.audio,
            )
            for hold in holds
        ]
        total_frames = max(
            [running_frame if segments else 0, *(hold.end_frame for hold in holds)]
        )
        return cls(
            fps=fps,
            sample_rate=sample_rate,
            segments=segments,
            cuts=cuts,
            total_frames=total_frames,
            total_samples=_sample_at(total_frames, fps, sample_rate),
            warnings=warnings,
            holds=planned_holds,
            stills=list(edit_list.stills),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible plan representation."""
        return {
            "fps": str(self.fps),
            "fps_float": float(self.fps),
            "sample_rate": self.sample_rate,
            "segments": [segment.to_dict() for segment in self.segments],
            "cuts": [cut.to_dict() for cut in self.cuts],
            "total_frames": self.total_frames,
            "total_samples": self.total_samples,
            "holds": [hold.to_dict() for hold in self.holds],
            "stills": [still.to_dict() for still in self.stills],
            "warnings": list(self.warnings),
        }
