"""Face rectangles and the portable faces.json document."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from yap_studio.errors import InputError, ValidationError

FACES_SCHEMA = "yap-studio/faces@1"


def _object(data: Any, parameter: str) -> dict[str, Any]:
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValidationError(parameter, "expected an object with string keys")
    return data


def _field(data: dict[str, Any], key: str, kind: type, parameter: str) -> Any:
    if key not in data:
        raise ValidationError(f"{parameter}.{key}", "missing required key")
    value = data[key]
    if type(value) is not kind:
        raise ValidationError(f"{parameter}.{key}", f"expected {kind.__name__}")
    return value


def _number(data: dict[str, Any], key: str, parameter: str) -> float:
    value = data.get(key)
    if type(value) not in (int, float):
        raise ValidationError(f"{parameter}.{key}", "expected a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(f"{parameter}.{key}", "expected a finite number") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{parameter}.{key}", "expected a finite number")
    return result


@dataclass(frozen=True)
class Box:
    """A rectangle in display pixels with its origin at the top left."""

    x: float
    y: float
    w: float
    h: float
    start_frame: int | None = None
    end_frame: int | None = None

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.w, self.h)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("box coordinates must be finite numbers")
        if self.w < 0 or self.h < 0:
            raise ValueError("box width and height must be nonnegative")
        if self.start_frame is not None or self.end_frame is not None:
            if (type(self.start_frame) is not int or type(self.end_frame) is not int
                    or not 0 <= self.start_frame < self.end_frame):
                raise ValueError("box window needs integers with 0 <= start_frame < end_frame")

    def overlaps_frames(self, start: int, end: int) -> bool:
        """Whether this box applies within a half-open frame window."""
        return self.start_frame is None or (self.start_frame < end and self.end_frame > start)

    @property
    def x1(self) -> float:
        """Return the right edge in display pixels."""
        return self.x + self.w

    @property
    def y1(self) -> float:
        """Return the bottom edge in display pixels."""
        return self.y + self.h

    def to_dict(self) -> dict[str, float | int]:
        """Return rounded JSON coordinates."""
        return {**{key: round(getattr(self, key), 2) for key in ("x", "y", "w", "h")},
                **({"start_frame": self.start_frame, "end_frame": self.end_frame}
                   if self.start_frame is not None else {})}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Box:
        """Parse coordinates without coercing strings or booleans."""
        obj = _object(data, "box")
        try:
            window = ({key: _field(obj, key, int, "box") for key in ("start_frame", "end_frame")}
                      if "start_frame" in obj or "end_frame" in obj else {})
            return cls(*(_number(obj, key, "box") for key in ("x", "y", "w", "h")), **window)
        except ValueError as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("box", str(exc)) from exc


@dataclass(frozen=True)
class FaceBox:
    """A face rectangle, its confidence, and how it was obtained."""

    box: Box
    confidence: float
    source: Literal["detected", "interpolated"]

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            raise ValueError("face box must be a Box")
        if (
            type(self.confidence) not in (int, float)
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("face confidence must be finite and between 0 and 1")
        if self.source not in ("detected", "interpolated"):
            raise ValueError("face source must be detected or interpolated")

    def to_dict(self) -> dict[str, Any]:
        """Return the flat face record used in faces.json."""
        return {
            **self.box.to_dict(),
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceBox:
        """Parse a face record with validated coordinates and confidence."""
        obj = _object(data, "face")
        try:
            return cls(
                Box.from_dict(obj),
                _number(obj, "confidence", "face"),
                _field(obj, "source", str, "face"),
            )
        except ValueError as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("face", str(exc)) from exc


@dataclass(frozen=True)
class VideoIdentity:
    """Identify the input without recording its directory."""

    name: str
    sha256: str
    frames: int
    fps: Fraction
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("video name must be a bare file name")
        if not isinstance(self.sha256, str):
            raise ValueError("video sha256 must be a string")
        if any(
            type(value) is not int or value < 1 for value in (self.frames, self.width, self.height)
        ):
            raise ValueError("video frames, width, and height must be positive integers")
        if not isinstance(self.fps, Fraction) or self.fps <= 0:
            raise ValueError("video fps must be a positive Fraction")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON identity fields, including an exact rational frame rate."""
        return {
            "name": self.name,
            "sha256": self.sha256,
            "frames": self.frames,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoIdentity:
        """Parse an identity using the document's strict field types."""
        obj = _object(data, "video")
        fps = _field(obj, "fps", str, "video")
        if re.fullmatch(r"[0-9]+/[1-9][0-9]*", fps) is None:
            raise ValidationError("video.fps", "expected a positive rational such as 30/1")
        try:
            return cls(
                _field(obj, "name", str, "video"),
                _field(obj, "sha256", str, "video"),
                _field(obj, "frames", int, "video"),
                Fraction(fps),
                _field(obj, "width", int, "video"),
                _field(obj, "height", int, "video"),
            )
        except ValueError as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("video", str(exc)) from exc


@dataclass(frozen=True)
class FacesData:
    """Per-frame faces and optional presentation seconds, or an explicit unavailable result."""

    status: Literal["measured", "unmeasured"]
    reason: str | None
    video: VideoIdentity
    detector: dict[str, Any]
    sampling: dict[str, Any]
    frames: tuple[tuple[FaceBox, ...], ...]
    gaps: tuple[tuple[int, int], ...]
    warnings: tuple[str, ...] = ()
    times: tuple[float, ...] | None = None
    end_time: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.video, VideoIdentity):
            raise ValueError("video must be a VideoIdentity")
        for name, value in (("detector", self.detector), ("sampling", self.sampling)):
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise ValueError(f"{name} must be an object with string keys")
        if not isinstance(self.frames, tuple) or any(
            not isinstance(faces, tuple) or any(not isinstance(face, FaceBox) for face in faces)
            for faces in self.frames
        ):
            raise ValueError("frames must be a tuple of face tuples")
        if not isinstance(self.gaps, tuple) or any(
            not isinstance(gap, tuple)
            or len(gap) != 2
            or any(type(edge) is not int for edge in gap)
            for gap in self.gaps
        ):
            raise ValueError("gaps must be a tuple of integer pairs")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(warning, str) for warning in self.warnings
        ):
            raise ValueError("warnings must be a tuple of strings")
        if self.times is None:
            if self.end_time is not None:
                raise ValueError("times and end_time must both be present or absent")
        else:
            if (
                not isinstance(self.times, tuple)
                or len(self.times) != self.video.frames
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in self.times
                )
            ):
                raise ValueError("times must contain one finite presentation time per frame")
            if any(a > b for a, b in zip(self.times, self.times[1:], strict=False)):
                raise ValueError("times must be non-decreasing")
            if (
                type(self.end_time) not in (int, float)
                or not math.isfinite(self.end_time)
                or self.end_time < self.times[-1]
            ):
                raise ValueError(
                    "end_time must be finite and at or after the last presentation time"
                )
        if self.status == "unmeasured":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("unmeasured faces need a non-empty reason")
            if self.frames != () or self.gaps != ():
                raise ValueError("unmeasured faces must have no frames or gaps")
            if self.times is not None:
                raise ValueError("unmeasured faces must have no presentation times")
        elif self.status == "measured":
            if self.reason is not None or len(self.frames) != self.video.frames:
                raise ValueError("measured faces need no reason and must cover every video frame")
            expected: list[tuple[int, int]] = []
            start: int | None = None
            for index, faces in enumerate(self.frames):
                if not faces and start is None:
                    start = index
                elif faces and start is not None:
                    expected.append((start, index))
                    start = None
            if start is not None:
                expected.append((start, len(self.frames)))
            if self.gaps != tuple(expected):
                raise ValueError("gaps must equal the maximal runs of frames without faces")
        else:
            raise ValueError("status must be measured or unmeasured")

    def faces_at(self, frame: int) -> tuple[FaceBox, ...]:
        """Return the faces at a measured frame, rejecting indices outside the video."""
        if type(frame) is not int or not 0 <= frame < self.video.frames:
            raise IndexError("frame is outside the video")
        return self.frames[frame] if self.status == "measured" else ()

    def time_at(self, frame: int) -> float:
        """Return presentation seconds at a frame or the video's exclusive end."""
        if type(frame) is not int or not 0 <= frame <= self.video.frames:
            raise IndexError("frame is outside the video")
        if self.times is None:
            return float(frame / self.video.fps)
        if frame == self.video.frames:
            assert self.end_time is not None
            return self.end_time
        return self.times[frame]

    def to_dict(self) -> dict[str, Any]:
        """Return the faces.json layout in its defined key order."""
        return {
            "schema": FACES_SCHEMA,
            "status": self.status,
            "reason": self.reason,
            "video": self.video.to_dict(),
            "detector": self.detector,
            "sampling": self.sampling,
            "frames": [
                {
                    "frame": index,
                    **({"t": round(self.times[index], 6)} if self.times is not None else {}),
                    "faces": [face.to_dict() for face in faces],
                }
                for index, faces in enumerate(self.frames)
            ],
            "gaps": [{"start_frame": start, "end_frame": end} for start, end in self.gaps],
            "end_time": round(self.end_time, 6) if self.end_time is not None else None,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FacesData:
        """Validate the schema, required fields, types, frame order, and gaps."""
        obj = _object(data, "faces")
        if _field(obj, "schema", str, "faces") != FACES_SCHEMA:
            raise ValidationError("faces.schema", f"expected {FACES_SCHEMA}")
        if "reason" not in obj or (obj["reason"] is not None and type(obj["reason"]) is not str):
            raise ValidationError("faces.reason", "expected a required string or null")
        per_frame = []
        times = []
        for index, record in enumerate(_field(obj, "frames", list, "faces")):
            location = f"faces.frames[{index}]"
            record = _object(record, location)
            if _field(record, "frame", int, location) != index:
                raise ValidationError(location, "frames must appear in order starting at zero")
            if "t" in record:
                times.append(_number(record, "t", location))
            per_frame.append(
                tuple(FaceBox.from_dict(face) for face in _field(record, "faces", list, location))
            )
        if times and len(times) != len(per_frame):
            raise ValidationError("faces.frames", "presentation times must appear on all frames")
        end_time = _number(obj, "end_time", "faces") if obj.get("end_time") is not None else None
        gaps = []
        for index, gap in enumerate(_field(obj, "gaps", list, "faces")):
            location = f"faces.gaps[{index}]"
            gap = _object(gap, location)
            gaps.append(
                (_field(gap, "start_frame", int, location), _field(gap, "end_frame", int, location))
            )
        warnings = _field(obj, "warnings", list, "faces")
        if any(type(warning) is not str for warning in warnings):
            raise ValidationError("faces.warnings", "expected strings")
        try:
            return cls(
                _field(obj, "status", str, "faces"),
                obj["reason"],
                VideoIdentity.from_dict(_field(obj, "video", dict, "faces")),
                _field(obj, "detector", dict, "faces"),
                _field(obj, "sampling", dict, "faces"),
                tuple(per_frame),
                tuple(gaps),
                tuple(warnings),
                tuple(times) if times else None,
                end_time,
            )
        except ValueError as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("faces", str(exc)) from exc


def load_faces(path: Path) -> FacesData:
    """Read a faces document, reporting its path on unreadable or invalid input."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(str(path), f"cannot read faces JSON: {exc}") from exc
    except UnicodeError as exc:
        raise ValidationError(str(path), f"invalid faces JSON: {exc}") from exc
    try:
        return FacesData.from_dict(json.loads(text))
    except (ValueError, RecursionError) as exc:
        raise ValidationError(str(path), f"invalid faces JSON: {exc}") from exc
