"""Locate review spans in the word, piece, and layer snapshots they describe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Literal

from yap_studio.errors import ValidationError
from yap_studio.feedback.moments import Occurrence, occurrences
from yap_studio.feedback.versions import LayerWindow
from yap_studio.project.model import CutWords
from yap_studio.review.model import Comment


@dataclass(frozen=True)
class Location:
    """Evidence on output time, without interpreting the reviewer's request."""

    status: Literal["located", "silence", "whole_video", "outside"]
    start_s: float
    end_s: float
    start_frame: int | None
    end_frame: int | None
    words: tuple[Occurrence, ...]
    before: Occurrence | None
    after: Occurrence | None
    segments: tuple[str, ...]
    layers: tuple[LayerWindow, ...]
    layers_known: bool

    def to_dict(self) -> dict:
        """Return JSON-ready occurrence and layer records in output order."""
        return {
            "status": self.status, "start_s": self.start_s, "end_s": self.end_s,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "words": [asdict(word) for word in self.words],
            "before": asdict(self.before) if self.before is not None else None,
            "after": asdict(self.after) if self.after is not None else None,
            "segments": list(self.segments), "layers": [asdict(layer) for layer in self.layers],
            "layers_known": self.layers_known,
        }


def locate(
    comment: Comment, *, fps: Fraction | None, duration_s: float | None,
    words: CutWords, pieces: dict, layers: tuple[LayerWindow, ...] | None,
) -> Location:
    """Resolve a half-open span or instant through this comment's own snapshots."""
    if fps is not None:
        if fps <= 0 or comment.start_frame is None:
            raise ValidationError("comment", "constant rate requires fps and a start frame")
        start_frame = comment.start_frame
        end_frame = comment.end_frame if comment.end_frame is not None else start_frame + 1
        start, end = Fraction(start_frame) / fps, Fraction(end_frame) / fps
    else:
        start_frame = end_frame = None
        start = Fraction(str(comment.start_s))
        end = Fraction(str(comment.end_s)) if comment.end_s is not None else start
    if start < 0 or end < start:
        raise ValidationError("comment", "span must be non-negative and in time order")

    def overlaps(a: float | Fraction, b: float | Fraction) -> bool:
        a = a if isinstance(a, Fraction) else Fraction(str(a))
        b = b if isinstance(b, Fraction) else Fraction(str(b))
        return a <= start < b if start == end else a < end and b > start

    items = occurrences(words)
    found = tuple(word for word in items if overlaps(word.start, word.end))
    piece_fps = Fraction(pieces["fps"])
    segments = tuple(dict.fromkeys(
        piece["segment"] for piece in pieces["pieces"]
        if overlaps(Fraction(piece["out_f0"]) / piece_fps,
                    Fraction(piece["out_f0"] + piece["n_frames"]) / piece_fps)
    ))
    found_layers = tuple(layer for layer in layers or () if overlaps(layer.start, layer.end))
    before = after = None
    if duration_s is not None and start >= Fraction(str(duration_s)):
        state = "outside"
    elif comment.target == "video":
        state = "whole_video"
    elif found:
        state = "located"
    else:
        state = "silence"
        before = next((word for word in reversed(items) if Fraction(str(word.end)) <= start), None)
        after = next((word for word in items if Fraction(str(word.start)) >= end), None)
    return Location(state, float(start), float(end), start_frame, end_frame, found,
                    before, after, segments, found_layers, layers is not None)
