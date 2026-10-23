"""Review comments, exact frame positions, and input validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from yap_studio.checks.model import format_time
from yap_studio.errors import ValidationError

Kind = Literal["keep", "fix"]
Target = Literal["story", "speech", "caption", "element", "video"]
TARGETS: tuple[Target, ...] = ("story", "speech", "caption", "element", "video")


def legacy_kind(label: str) -> Kind:
    """Map an old like or change label to its current comment kind."""
    if label == "like":
        return "keep"
    if label == "change":
        return "fix"
    raise ValueError("label must be 'like' or 'change'")


@dataclass(frozen=True)
class MediaFacts:
    """Facts that identify the reviewed video and its timeline."""

    name: str
    sha256: str
    duration_s: float | None
    fps: Fraction | None
    cfr: bool


@dataclass(frozen=True)
class Comment:
    """A point or half-open range with a kind and optional scope.

    Story covers the words in the cut and their order over the range. Speech
    covers delivery and pauses. Caption covers caption text or appearance.
    Element covers an on-screen title, card, insert, or zoom. Video covers the
    whole video. A keep protects only its target. A keep with no target is read as story.
    """

    id: str
    kind: Kind
    target: Target | None
    text: str
    start_frame: int | None
    end_frame: int | None
    start_s: float
    end_s: float | None
    created_at: str
    updated_at: str
    request: dict | None = None
    pin: dict | None = None


_PIN_SPAN_ERROR = "must be on a frame inside the note; move or remove the pin"
_PIN_TIME_TOLERANCE_S = 0.0005


def _pin_coordinates(value: dict) -> dict:
    coordinates = {}
    for key in ("x", "y"):
        number = value[key]
        try:
            valid = (type(number) in (int, float) and math.isfinite(number)
                     and 0 <= number <= 1)
        except OverflowError:
            valid = False
        if not valid:
            raise ValidationError("pin", "x and y must be finite numbers from 0 to 1")
        coordinates[key] = round(number, 4)
    return coordinates


def _pin_seconds(value: object) -> float:
    try:
        return _seconds(value, "pin")
    except OverflowError as exc:
        raise ValidationError("pin", "must be a finite non-negative number") from exc


def _pin_span(pin: dict, *, start_frame: int | None, end_frame: int | None,
              start_s: float, end_s: float | None) -> None:
    if start_frame is not None:
        frame = pin["frame"]
        valid = frame is not None and (frame == start_frame if end_frame is None
                                       else start_frame <= frame < end_frame)
    else:
        valid = (abs(pin["time_s"] - start_s) <= _PIN_TIME_TOLERANCE_S if end_s is None
                 else start_s <= pin["time_s"] <= end_s)
    if not valid:
        raise ValidationError("pin", _PIN_SPAN_ERROR)


def normalize_pin(value: object, *, facts: MediaFacts, start_frame: int | None,
                  end_frame: int | None, start_s: float, end_s: float | None) -> dict | None:
    """Validate a page pin and attach its canonical frame and seconds."""
    if value is None:
        return None
    timing = "frame" if facts.cfr else "time_s"
    if not isinstance(value, dict) or set(value) != {"x", "y", timing}:
        raise ValidationError("pin", f"must contain exactly x, y and {timing}")
    if facts.cfr and (facts.fps is None or facts.fps <= 0):
        raise ValidationError("pin", "requires a positive constant frame rate")
    coordinates = _pin_coordinates(value)
    frame = _frame(value["frame"], "pin") if facts.cfr else None
    time = start_s if facts.cfr else _pin_seconds(value["time_s"])
    pin = {**coordinates, "frame": frame, "time_s": time}
    _pin_span(pin, start_frame=start_frame, end_frame=end_frame, start_s=start_s, end_s=end_s)
    if facts.cfr:
        pin["time_s"] = round(float(frame_start(frame, facts.fps)), 4)
    return pin


def validate_stored_pin(value: object, *, start_frame: int | None, end_frame: int | None,
                        start_s: float, end_s: float | None) -> dict:
    """Validate the complete stored shape against its comment's span."""
    if not isinstance(value, dict) or set(value) != {"x", "y", "frame", "time_s"}:
        raise ValidationError("pin", "must contain exactly x, y, frame and time_s")
    _pin_coordinates(value)
    if value["frame"] is not None:
        _frame(value["frame"], "pin")
        if start_frame is None:
            raise ValidationError("pin", "frame must be null for a time-only record")
    _pin_seconds(value["time_s"])
    _pin_span(value, start_frame=start_frame, end_frame=end_frame, start_s=start_s, end_s=end_s)
    return dict(value)


def _cut_request(value: dict, kind: Kind, target: Target | None) -> dict:
    if kind != "fix" or target != "story":
        raise ValidationError("request", "a cut request needs a Fix note about the story")
    if set(value) != {"type", "start", "end", "words"}:
        raise ValidationError("request", "a cut request needs start, end and words")
    try:
        start = _seconds(value["start"], "request")
        end = _seconds(value["end"], "request")
    except OverflowError as exc:
        raise ValidationError("request", "times must be finite non-negative numbers") from exc
    if start >= end:
        raise ValidationError("request", "end must be greater than start")
    if not isinstance(value["words"], list) or not 1 <= len(value["words"]) <= 200:
        raise ValidationError("request", "words must list 1 to 200 words")
    words = []
    seen = set()
    for word in value["words"]:
        if not isinstance(word, dict) or set(word) != {"id", "segment", "text"}:
            raise ValidationError("request", "word must contain id, segment and text")
        word_id, segment, text = word["id"], word["segment"], word["text"]
        if not isinstance(word_id, str) or not re.fullmatch(r"t[1-9][0-9]*:(0|[1-9][0-9]*)",
                                                          word_id):
            raise ValidationError("request", "word id must look like t1:12")
        if not isinstance(segment, str) or not segment.strip() or len(segment) > 200:
            raise ValidationError("request", "segment must contain 1 to 200 characters")
        if (not isinstance(text, str) or not 1 <= len(text.strip()) <= 200
                or re.search(r"[\r\n\v\f\x85\u2028\u2029]", text)):
            raise ValidationError("request", "text must contain 1 to 200 characters, no newlines")
        occurrence = (word_id, segment.strip())
        if occurrence in seen:
            raise ValidationError("request", "words must not repeat an occurrence")
        seen.add(occurrence)
        words.append({"id": word_id, "segment": segment.strip(), "text": text.strip()})
    return {"type": "cut", "start": start, "end": end, "words": words}


def normalize_request(value: object, *, kind: Kind, target: Target | None) -> dict | None:
    """Validate a typed request independently of comment timing and patch fields."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("type") not in (
        "spelling", "restore", "cut", "caption_fix",
    ):
        raise ValidationError("request", "must be a spelling, restore, cut or caption fix request")
    if value["type"] == "caption_fix":
        if kind != "fix" or target != "caption" or set(value) != {
            "type", "word", "heard", "from", "to", "previous",
        }:
            raise ValidationError("request", "a caption fix needs a Fix note about a caption")
        if not isinstance(value["word"], str) or not re.fullmatch(r"t[1-9][0-9]*:[0-9]+",
                                                               value["word"]):
            raise ValidationError("request", "invalid caption word id")
        for key in ("from", "to", "heard", "previous"):
            text = value[key]
            if key in ("heard", "previous") and text is None:
                continue
            if (not isinstance(text, str) or not text.strip()
                    or re.search(r"[\r\n\v\f\x85\u2028\u2029]", text)):
                raise ValidationError("request", "caption text must be nonempty with no newlines")
        if len(value["to"]) > 80:
            raise ValidationError("request", "caption text must have at most 80 characters")
        return dict(value)
    if value["type"] == "cut":
        return _cut_request(value, kind, target)
    if value["type"] == "restore":
        if kind != "fix" or target != "story":
            raise ValidationError("request", "a restore request needs a Fix note about the story")
        if set(value) != {"type", "range", "take", "start", "end", "text"}:
            raise ValidationError("request", "a restore request needs range, take, start, end, "
                                  "text")
        for key in ("range", "take"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValidationError("request", f"{key} must be a nonempty string")
        try:
            start = _seconds(value["start"], "request")
            end = _seconds(value["end"], "request")
        except OverflowError as exc:
            raise ValidationError("request", "times must be finite non-negative numbers") from exc
        if start >= end:
            raise ValidationError("request", "end must be greater than start")
        text = value["text"]
        if text is not None and (not isinstance(text, str) or len(text) > 500):
            raise ValidationError("request", "text must be null or a string of at most "
                                  "500 characters")
        return {**value, "start": start, "end": end}
    if kind != "fix" or target != "caption":
        raise ValidationError("request", "a spelling fix needs a Fix note about a caption")
    if set(value) != {"type", "from", "to"}:
        raise ValidationError("request", "a spelling request needs type, from and to")
    words = {}
    for key in ("from", "to"):
        word = value[key]
        if (not isinstance(word, str) or not 1 <= len(word.strip()) <= 80
                or re.search(r"[\r\n\v\f\x85\u2028\u2029]", word)):
            raise ValidationError("request", f"{key} must contain 1 to 80 characters, no newlines")
        words[key] = word.strip()
    if words["from"] == words["to"]:
        raise ValidationError("request", "from and to must differ")
    return {"type": "spelling", **words}


def frame_at(t: float, fps: Fraction) -> int:
    """Map a time to its frame, tolerating a one-thousandth-frame boundary error."""
    return max(0, math.floor(Fraction(t) * fps + Fraction(1, 1000)))


def frame_start(frame: int, fps: Fraction) -> Fraction:
    """Return the exact start time of a frame."""
    return Fraction(frame) / fps


def frame_mid(frame: int, fps: Fraction) -> Fraction:
    """Return an exact seek position in the middle of a frame."""
    return (Fraction(frame) + Fraction(1, 2)) / fps


def frame_time_text(frame: int, fps: Fraction) -> str:
    """Format a frame boundary by flooring its exact time to a tenth."""
    return format_time(frame_start(frame, fps))


def _frame_count(duration_s: float, fps: Fraction) -> int:
    return max(1, math.floor(Fraction(duration_s) * fps + Fraction(1, 2)))


def conform_comment(comment: Comment, facts: MediaFacts) -> Comment:
    """Represent a saved comment on the video's current timeline."""
    if not facts.cfr or facts.fps is None:
        return replace(comment, start_frame=None, end_frame=None,
                       pin={**comment.pin, "frame": None} if comment.pin is not None else None)
    if comment.start_frame is not None:
        return comment

    fps = facts.fps
    start_frame = frame_at(comment.start_s, fps)
    end_frame = (
        max(start_frame + 1, math.floor(Fraction(comment.end_s) * fps + Fraction(1, 2)))
        if comment.end_s is not None
        else None
    )
    if facts.duration_s is not None:
        frame_count = _frame_count(facts.duration_s, fps)
        start_frame = min(start_frame, frame_count - 1)
        if end_frame is not None:
            end_frame = min(end_frame, frame_count)
    if end_frame is not None and end_frame <= start_frame + 1:
        end_frame = None
    pin = comment.pin
    if pin is not None:
        pin_frame = (start_frame if end_frame is None else
                     max(start_frame, min(end_frame - 1, frame_at(pin["time_s"], fps))))
        pin = {**pin, "frame": pin_frame,
               "time_s": round(float(frame_start(pin_frame, fps)), 4)}
    return replace(
        comment,
        start_frame=start_frame,
        end_frame=end_frame,
        start_s=round(float(frame_start(start_frame, fps)), 4),
        end_s=round(float(frame_start(end_frame, fps)), 4) if end_frame is not None else None,
        pin=pin,
    )


def _frame(value: object, parameter: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(parameter, "must be a non-negative integer")
    return value


def _seconds(value: object, parameter: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(parameter, "must be a finite non-negative number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValidationError(parameter, "must be a finite non-negative number")
    return seconds


def _validate_cut_range(request: dict | None, start_s: float, end_s: float | None) -> None:
    if request is None or request["type"] != "cut":
        return
    if end_s is None:
        raise ValidationError("request", "a cut request needs a range")
    if (round(request["start"], 4) != round(start_s, 4)
            or round(request["end"], 4) != round(end_s, 4)):
        raise ValidationError("request", "a cut request must match the note's range; "
                              "select the range again")


def normalize_input(
    data: dict, facts: MediaFacts, *, partial: bool = False, current: Comment | None = None
) -> dict:
    """Validate a create or patch payload and return canonical comment fields."""
    if not isinstance(data, dict):
        raise ValidationError("data", "must be an object")
    if partial and current is None:
        raise ValidationError("current", "is required for a patch")
    allowed = (
        {"kind", "target", "label", "text", "request", "pin", "start_frame", "end_frame"}
        if facts.cfr
        else {"kind", "target", "label", "text", "request", "pin", "start_s", "end_s"}
    )
    for key in data:
        if key not in allowed:
            raise ValidationError(key, "is not an accepted field")

    kind = data.get("kind", current.kind if partial and current else "fix")
    if kind not in ("keep", "fix"):
        raise ValidationError("kind", "must be 'keep' or 'fix'")
    if "label" in data:
        try:
            mapped = legacy_kind(data["label"])
        except ValueError as exc:
            raise ValidationError("label", "must be 'like' or 'change'") from exc
        if "kind" in data and mapped != kind:
            raise ValidationError("label", "disagrees with kind")
        kind = mapped
    target = data.get("target", current.target if partial and current else None)
    if target is not None and target not in TARGETS:
        raise ValidationError(
            "target", "must be 'story', 'speech', 'caption', 'element' or 'video'"
        )
    request = normalize_request(
        data.get("request", current.request if partial and current else None),
        kind=kind, target=target,
    )
    text = data.get("text", current.text if current else None)
    if not isinstance(text, str):
        raise ValidationError("text", "must be a string")
    text = text.replace("\r\n", "\n").strip()
    if not 1 <= len(text) <= 5000:
        raise ValidationError("text", "must contain 1 to 5000 characters")

    if facts.cfr:
        fps = facts.fps
        if fps is None or fps <= 0:
            raise ValidationError("fps", "must be positive for constant frame rate media")
        start_frame = _frame(
            data.get("start_frame", current.start_frame if current else None), "start_frame"
        )
        end_value = data.get("end_frame", current.end_frame if current else None)
        end_frame = None if end_value is None else _frame(end_value, "end_frame")
        if end_frame is not None and end_frame <= start_frame:
            raise ValidationError("end_frame", "must be greater than start_frame")
        if facts.duration_s is not None:
            frame_count = _frame_count(facts.duration_s, fps)
            if start_frame > frame_count - 1:
                raise ValidationError("start_frame", "must be within the frame count")
            if end_frame is not None and end_frame > frame_count:
                raise ValidationError("end_frame", "must be within the frame count")
        if end_frame == start_frame + 1:
            end_frame = None
        start_s = round(float(frame_start(start_frame, fps)), 4)
        end_s = round(float(frame_start(end_frame, fps)), 4) if end_frame is not None else None
        _validate_cut_range(request, start_s, end_s)
        return {
            "kind": kind,
            "target": target,
            "text": text,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": start_s,
            "end_s": end_s,
            "request": request,
            "pin": _input_pin(data, facts, current if partial else None,
                              start_frame, end_frame, start_s, end_s),
        }

    start_s = _seconds(data.get("start_s", current.start_s if current else None), "start_s")
    end_value = data.get("end_s", current.end_s if current else None)
    end_s = None if end_value is None else _seconds(end_value, "end_s")
    if end_s is not None and end_s <= start_s:
        raise ValidationError("end_s", "must be greater than start_s")
    if facts.duration_s is not None:
        duration_limit = facts.duration_s + 0.1
        if start_s > duration_limit:
            raise ValidationError("start_s", "must be within the duration plus 0.1 seconds")
        if end_s is not None and end_s > duration_limit:
            raise ValidationError("end_s", "must be within the duration plus 0.1 seconds")
    _validate_cut_range(request, start_s, end_s)
    return {
        "kind": kind,
        "target": target,
        "text": text,
        "start_frame": None,
        "end_frame": None,
        "start_s": start_s,
        "end_s": end_s,
        "request": request,
        "pin": _input_pin(data, facts, current if partial else None, None, None, start_s, end_s),
    }


def _input_pin(data: dict, facts: MediaFacts, current: Comment | None,
               start_frame: int | None, end_frame: int | None,
               start_s: float, end_s: float | None) -> dict | None:
    span = dict(start_frame=start_frame, end_frame=end_frame, start_s=start_s, end_s=end_s)
    if "pin" not in data and current is not None and current.pin is not None:
        _pin_span(current.pin, **span)
        return dict(current.pin)
    return normalize_pin(data.get("pin"), facts=facts, **span)


def sort_key(comment: Comment) -> tuple[int | float, int | float, int]:
    """Sort by start, then end boundary, then the creation number in the id."""
    match = re.search(r"\d+$", comment.id)
    if match is None:
        raise ValueError(f"comment id has no creation number: {comment.id}")
    start = comment.start_frame if comment.start_frame is not None else comment.start_s
    end = (
        comment.end_frame
        if comment.end_frame is not None
        else (comment.start_frame if comment.start_frame is not None else comment.end_s)
    )
    return start, start if end is None else end, int(match.group())


def sorted_comments(comments: list[Comment]) -> list[Comment]:
    """Return comments in timeline order without changing the input list."""
    return sorted(comments, key=sort_key)
