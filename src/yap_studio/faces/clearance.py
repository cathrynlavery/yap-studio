"""Pure face and caption clearance checks in display pixels.

Face clearance measures head boxes with room for hair. Margins can come from
options, layouts, or resolved styles; the defaults apply at 1080x1920.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from yap_studio.errors import InputError, ValidationError
from yap_studio.faces.model import Box, FacesData, _field, _number, _object

LAYOUT_SCHEMA = "yap-studio/clearance-layout@1"
DEFAULT_FACE_CLEARANCE_PX = 24
DEFAULT_CAPTION_GAP_PX = 100


def _hair_ratio(value: Any, parameter: str) -> float:
    detail = "must be a finite number from 0 to 2"
    if type(value) not in (int, float):
        raise ValidationError(parameter, detail)
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(parameter, detail) from exc
    if not math.isfinite(result) or not 0 <= result <= 2:
        raise ValidationError(parameter, detail)
    return result


@dataclass(frozen=True)
class HairAllowance:
    """Room added above a face as a height fraction and on each side as a width fraction.

    The bottom at the chin is unchanged.
    """

    above: float = 0.6
    sides: float = 0.15

    def __post_init__(self) -> None:
        _hair_ratio(self.above, "hair.above")
        _hair_ratio(self.sides, "hair.sides")

    def to_dict(self) -> dict[str, float]:
        """Return the hair allowance ratios without rounding."""
        return {"above": self.above, "sides": self.sides}

    @classmethod
    def from_dict(cls, data: Any, parameter: str = "hair") -> HairAllowance:
        """Parse optional ratios strictly, preserving contextual field paths."""
        obj = _object(data, parameter)
        for key in obj:
            if key not in ("above", "sides"):
                raise ValidationError(f"{parameter}.{key}", "unknown field")
        above = obj.get("above", DEFAULT_HAIR.above)
        sides = obj.get("sides", DEFAULT_HAIR.sides)
        _hair_ratio(above, f"{parameter}.above")
        _hair_ratio(sides, f"{parameter}.sides")
        return cls(above, sides)


DEFAULT_HAIR = HairAllowance()
NO_HAIR = HairAllowance(0, 0)


def head_box(face: Box, hair: HairAllowance = DEFAULT_HAIR) -> Box:
    """Add room for hair above and beside a face, allowing negative frame coordinates."""
    return Box(
        face.x - hair.sides * face.w,
        face.y - hair.above * face.h,
        face.w * (1 + 2 * hair.sides),
        face.h * (1 + hair.above),
    )


def face_clearance_px(style: Mapping[str, Any]) -> float:
    """Return the face clearance margin from resolved style data."""
    return style["graphics"]["face_clearance_px"]


def caption_gap_px(style: Mapping[str, Any]) -> float:
    """Return the caption gap from resolved style data."""
    return style["graphics"]["caption_gap_px"]


def gap_px(a: Box, b: Box) -> float:
    """Return the signed Chebyshev gap, with negative values for overlap."""
    dx = max(b.x - a.x1, a.x - b.x1)
    dy = max(b.y - a.y1, a.y - b.y1)
    return max(dx, dy)


def _margin(value: float, parameter: str) -> float:
    if type(value) not in (int, float):
        raise ValidationError(parameter, "must be a finite number >= 0")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(parameter, "must be a finite number >= 0") from exc
    if not math.isfinite(result) or result < 0:
        raise ValidationError(parameter, "must be a finite number >= 0")
    return result


def _window(start: int, end: int, parameter: str) -> None:
    if type(start) is not int or start < 0:
        raise ValidationError(f"{parameter}.start_frame", "must be an integer >= 0")
    if type(end) is not int or end <= start:
        raise ValidationError(f"{parameter}.end_frame", "must be an integer > start_frame")


def _id(value: str, parameter: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{parameter}.id", "must be a non-empty string")


@dataclass(frozen=True)
class Element:
    """A graphic layer in a half-open frame window."""

    id: str
    kind: Literal["card", "title", "overlay", "background"]
    box: Box
    start_frame: int
    end_frame: int
    face_windows: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        _id(self.id, "element")
        if self.kind not in ("card", "title", "overlay", "background"):
            raise ValidationError("element.kind", "must be card, title, overlay, or background")
        if not isinstance(self.box, Box):
            raise ValidationError("element.box", "must be a Box")
        _window(self.start_frame, self.end_frame, "element")
        if self.face_windows is not None:
            if not isinstance(self.face_windows, tuple):
                raise ValidationError("element.face_windows", "must be a tuple of frame pairs")
            previous_end = self.start_frame
            for window in self.face_windows:
                if (not isinstance(window, tuple) or len(window) != 2
                        or any(type(frame) is not int for frame in window)
                        or not previous_end <= window[0] < window[1] <= self.end_frame):
                    raise ValidationError(
                        "element.face_windows", "use ordered frame pairs within the element window",
                    )
                previous_end = window[1]


@dataclass(frozen=True)
class CaptionPhrase:
    """A caption's actual glyph rectangles in a half-open frame window."""

    id: str
    text: str
    boxes: tuple[Box, ...]
    start_frame: int
    end_frame: int
    face_windows: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        _id(self.id, "caption")
        if not isinstance(self.text, str):
            raise ValidationError("caption.text", "must be a string")
        if (
            not isinstance(self.boxes, tuple)
            or not self.boxes
            or any(not isinstance(box, Box) for box in self.boxes)
        ):
            raise ValidationError("caption.boxes", "must be a non-empty tuple of Boxes")
        _window(self.start_frame, self.end_frame, "caption")
        for box in self.boxes:
            if (box.start_frame is not None
                    and not self.start_frame <= box.start_frame < box.end_frame <= self.end_frame):
                raise ValidationError("caption.boxes", "box windows must fit the caption window")
        if self.face_windows is not None:
            previous_end = self.start_frame
            if not isinstance(self.face_windows, tuple):
                raise ValidationError("caption.face_windows", "must be a tuple of frame pairs")
            for window in self.face_windows:
                if (not isinstance(window, tuple) or len(window) != 2
                        or any(type(frame) is not int for frame in window)
                        or not previous_end <= window[0] < window[1] <= self.end_frame):
                    raise ValidationError("caption.face_windows",
                                          "use ordered frame pairs within the caption window")
                previous_end = window[1]


def _layout_box(data: Any, parameter: str, *, timed: bool = False) -> Box:
    obj = _object(data, parameter)
    coordinates = {key: _number(obj, key, parameter) for key in ("x", "y", "w", "h")}
    for key in ("w", "h"):
        if coordinates[key] < 0:
            raise ValidationError(f"{parameter}.{key}", "must be nonnegative")
    window = {}
    if timed and ("start_frame" in obj or "end_frame" in obj):
        window = {key: _field(obj, key, int, parameter) for key in ("start_frame", "end_frame")}
        _window(window["start_frame"], window["end_frame"], parameter)
    return Box(**coordinates, **window)


def _face_windows(record: dict, parameter: str) -> tuple[tuple[int, int], ...] | None:
    if "face_windows" not in record:
        return None
    windows = record["face_windows"]
    if (not isinstance(windows, list)
            or any(not isinstance(window, list) or len(window) != 2 for window in windows)):
        raise ValidationError(f"{parameter}.face_windows", "use a list of integer frame pairs")
    return tuple(tuple(window) for window in windows)


@dataclass(frozen=True)
class Layout:
    """Display geometry, timed targets, and clearance margins."""

    width: int
    height: int
    elements: tuple[Element, ...]
    captions: tuple[CaptionPhrase, ...]
    clearance_px: float | None = None
    caption_gap_px: float = DEFAULT_CAPTION_GAP_PX
    hair: HairAllowance | None = None
    measured: bool | None = None

    def __post_init__(self) -> None:
        for key in ("width", "height"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValidationError(f"layout.{key}", "must be a positive integer")
        if self.clearance_px is not None:
            _margin(self.clearance_px, "layout.clearance_px")
        _margin(self.caption_gap_px, "layout.caption_gap_px")
        if self.hair is not None and not isinstance(self.hair, HairAllowance):
            raise ValidationError("layout.hair", "must be a HairAllowance")
        if self.measured is not None and type(self.measured) is not bool:
            raise ValidationError("layout.measured", "must be a boolean")
        seen: set[str] = set()
        for key, kind in (("elements", Element), ("captions", CaptionPhrase)):
            items = getattr(self, key)
            if not isinstance(items, tuple):
                raise ValidationError(f"layout.{key}", "must be a tuple")
            for index, item in enumerate(items):
                parameter = f"layout.{key}[{index}]"
                if not isinstance(item, kind):
                    raise ValidationError(parameter, f"must be a {kind.__name__}")
                if item.id in seen:
                    raise ValidationError(f"{parameter}.id", "ids must be unique across the layout")
                seen.add(item.id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layout:
        """Parse the layout with strict field types and contextual error paths."""
        obj = _object(data, "layout")
        if _field(obj, "schema", str, "layout") != LAYOUT_SCHEMA:
            raise ValidationError("layout.schema", f"expected {LAYOUT_SCHEMA}")
        elements = []
        captions = []
        for key in ("elements", "captions"):
            records = _field(obj, key, list, "layout") if key in obj else []
            for index, record in enumerate(records):
                parameter = f"layout.{key}[{index}]"
                record = _object(record, parameter)
                target_id = _field(record, "id", str, parameter)
                _id(target_id, parameter)
                start = _field(record, "start_frame", int, parameter)
                end = _field(record, "end_frame", int, parameter)
                _window(start, end, parameter)
                if key == "elements":
                    kind = _field(record, "kind", str, parameter)
                    if kind not in ("card", "title", "overlay", "background"):
                        raise ValidationError(
                            f"{parameter}.kind", "must be card, title, overlay, or background"
                        )
                    elements.append(
                        Element(target_id, kind, _layout_box(record, parameter), start, end,
                                _face_windows(record, parameter))
                    )
                else:
                    text = _field(record, "text", str, parameter)
                    boxes = _field(record, "boxes", list, parameter)
                    if not boxes:
                        raise ValidationError(
                            f"{parameter}.boxes", "must contain at least one glyph box"
                        )
                    captions.append(
                        CaptionPhrase(
                            target_id,
                            text,
                            tuple(
                                _layout_box(box, f"{parameter}.boxes[{box_index}]", timed=True)
                                for box_index, box in enumerate(boxes)
                            ),
                            start,
                            end,
                            _face_windows(record, parameter),
                        )
                    )
        return cls(
            _field(obj, "width", int, "layout"),
            _field(obj, "height", int, "layout"),
            tuple(elements),
            tuple(captions),
            _number(obj, "clearance_px", "layout") if obj.get("clearance_px") is not None else None,
            _number(obj, "caption_gap_px", "layout")
            if "caption_gap_px" in obj
            else DEFAULT_CAPTION_GAP_PX,
            HairAllowance.from_dict(obj["hair"], "layout.hair")
            if obj.get("hair") is not None
            else None,
            obj.get("measured"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the portable layout document."""
        return {
            "schema": LAYOUT_SCHEMA,
            "width": self.width,
            "height": self.height,
            "clearance_px": self.clearance_px,
            "hair": self.hair.to_dict() if self.hair is not None else None,
            "caption_gap_px": self.caption_gap_px,
            **({"measured": self.measured} if self.measured is not None else {}),
            "elements": [
                {
                    "id": element.id,
                    "kind": element.kind,
                    **element.box.to_dict(),
                    "start_frame": element.start_frame,
                    "end_frame": element.end_frame,
                    **({"face_windows": [list(window) for window in element.face_windows]}
                       if element.face_windows is not None else {}),
                }
                for element in self.elements
            ],
            "captions": [
                {
                    "id": phrase.id,
                    "text": phrase.text,
                    "start_frame": phrase.start_frame,
                    "end_frame": phrase.end_frame,
                    "boxes": [box.to_dict() for box in phrase.boxes],
                    **({"face_windows": [list(window) for window in phrase.face_windows]}
                       if phrase.face_windows is not None else {}),
                }
                for phrase in self.captions
            ],
        }


def load_layout(path: Path) -> Layout:
    """Read a layout, reporting unreadable files and invalid JSON as Yap errors."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(str(path), f"cannot read layout JSON: {exc}") from exc
    except UnicodeError as exc:
        raise ValidationError(str(path), f"invalid layout JSON: {exc}") from exc
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ValidationError(str(path), f"invalid layout JSON: {exc}") from exc
    return Layout.from_dict(data)


@dataclass(frozen=True)
class Violation:
    """A consecutive failing run and the boxes at its first worst frame."""

    kind: Literal["face", "caption_gap"]
    target_type: Literal["element", "caption"]
    target_id: str
    target_kind: str
    text: str | None
    start_frame: int
    end_frame: int
    frame: int
    gap_px: float
    required_px: float
    target_box: Box
    face: Box | None
    head: Box | None
    other_id: str | None

    def to_dict(self, time_at: Callable[[int], float]) -> dict[str, Any]:
        """Return frame ranges, rounded seconds, and the involved rectangles."""
        return {
            "kind": self.kind,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_kind": self.target_kind,
            "text": self.text,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame": self.frame,
            "start_s": round(time_at(self.start_frame), 3),
            "end_s": round(time_at(self.end_frame), 3),
            "time_s": round(time_at(self.frame), 3),
            "gap_px": round(self.gap_px, 2),
            "required_px": round(self.required_px, 2),
            "target_box": self.target_box.to_dict(),
            "face": self.face.to_dict() if self.face is not None else None,
            "head": self.head.to_dict() if self.head is not None else None,
            "other_id": self.other_id,
        }


def _caption_violations(
    elements: Sequence[Element], captions: Sequence[CaptionPhrase], required: float
) -> list[Violation]:
    violations = []
    for phrase in captions:
        for element in elements:
            if element.kind == "background":
                continue
            start = max(phrase.start_frame, element.start_frame)
            end = min(phrase.end_frame, element.end_frame)
            if start >= end:
                continue
            box = min((box for box in phrase.boxes if box.overlaps_frames(start, end)),
                      key=lambda box: gap_px(box, element.box), default=None)
            if box is None:
                continue
            start = max(start, box.start_frame if box.start_frame is not None else start)
            end = min(end, box.end_frame if box.end_frame is not None else end)
            gap = gap_px(box, element.box)
            if gap < required:
                violations.append(
                    Violation(
                        "caption_gap",
                        "caption",
                        phrase.id,
                        "caption",
                        phrase.text,
                        start,
                        end,
                        start,
                        gap,
                        required,
                        box,
                        None,
                        None,
                        element.id,
                    )
                )
    return violations


def _ordered(violations: list[Violation]) -> list[Violation]:
    return sorted(
        violations,
        key=lambda item: (
            item.start_frame,
            item.kind != "face",
            item.target_id,
            item.other_id or "",
        ),
    )


def check_clearance(
    faces: Sequence[Sequence[Box]],
    elements: Sequence[Element],
    captions: Sequence[CaptionPhrase],
    *,
    margin_px: float,
    caption_gap_px: float | None = None,
    hair: HairAllowance = DEFAULT_HAIR,
) -> list[Violation]:
    """Check timed targets against head boxes and optionally captions against elements."""
    margin = _margin(margin_px, "margin_px")
    caption_gap = None if caption_gap_px is None else _margin(caption_gap_px, "caption_gap_px")
    # The person matte and captions are drawn above a background.
    targets = [*(element for element in elements if element.kind != "background"), *captions]
    for target in targets:
        if target.end_frame > len(faces):
            raise ValueError(f"{target.id}: end_frame exceeds the face frame count")
    heads = [tuple((face, head_box(face, hair)) for face in frame) for frame in faces]
    violations = []
    for target in targets:
        is_caption = isinstance(target, CaptionPhrase)
        boxes = target.boxes if is_caption else (target.box,)
        active: Violation | None = None
        for frame in range(target.start_frame, target.end_frame):
            if (target.face_windows is not None
                    and not any(start <= frame < end for start, end in target.face_windows)):
                if active is not None:
                    violations.append(replace(active, end_frame=frame))
                    active = None
                continue
            closest = min(
                (
                    (gap_px(box, head), box, face, head)
                    for box in boxes
                    if box.overlaps_frames(frame, frame + 1)
                    for face, head in heads[frame]
                ),
                key=lambda pair: pair[0],
                default=None,
            )
            if closest is None or closest[0] >= margin:
                if active is not None:
                    violations.append(replace(active, end_frame=frame))
                    active = None
                continue
            gap, box, face, head = closest
            if active is None:
                active = Violation(
                    "face",
                    "caption" if is_caption else "element",
                    target.id,
                    "caption" if is_caption else target.kind,
                    target.text if is_caption else None,
                    frame,
                    frame + 1,
                    frame,
                    gap,
                    margin,
                    box,
                    face,
                    head,
                    None,
                )
            elif gap < active.gap_px:
                active = replace(
                    active, frame=frame, gap_px=gap, target_box=box, face=face, head=head
                )
        if active is not None:
            violations.append(replace(active, end_frame=target.end_frame))
    if caption_gap is not None:
        violations.extend(_caption_violations(elements, captions, caption_gap))
    return _ordered(violations)


def face_boxes_per_frame(data: FacesData) -> list[tuple[Box, ...]]:
    """Convert measured face records without treating unmeasured faces as empty."""
    if data.status != "measured":
        raise ValueError("face boxes require measured FacesData")
    return [tuple(face.box for face in frame) for frame in data.frames]


@dataclass(frozen=True)
class ClearanceReport:
    """Clearance results with explicit detection availability and video timing."""

    status: Literal["pass", "fail", "unmeasured"]
    reason: str | None
    margin_px: float
    hair: HairAllowance
    margin_source: Literal["option", "layout", "style", "default"]
    caption_gap_px: float
    fps: Fraction
    frames: int
    elements: int
    captions: int
    violations: tuple[Violation, ...]
    times: tuple[float, ...] | None = None
    end_time: float | None = None
    frames_with_faces: int | None = None
    frames_without_faces: int | None = None

    def time_at(self, frame: int) -> float:
        """Return presentation seconds at a frame or the video's exclusive end."""
        if type(frame) is not int or not 0 <= frame <= self.frames:
            raise IndexError("frame is outside the video")
        if self.times is None:
            return float(frame / self.fps)
        if frame == self.frames:
            assert self.end_time is not None
            return self.end_time
        return self.times[frame]

    def to_dict(self) -> dict[str, Any]:
        """Return the clearance report document with exact rational fps."""
        return {
            "schema": "yap-studio/clearance-report@1",
            "status": self.status,
            "reason": self.reason,
            "margin_px": self.margin_px,
            "hair": self.hair.to_dict(),
            "margin_source": self.margin_source,
            "caption_gap_px": self.caption_gap_px,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "timing": "pts" if self.times is not None else "fps",
            "frames": self.frames,
            "elements": self.elements,
            "captions": self.captions,
            "coverage": {
                "frames_with_faces": self.frames_with_faces,
                "frames_without_faces": self.frames_without_faces,
            }
            if self.frames_with_faces is not None
            else None,
            "violations": [violation.to_dict(self.time_at) for violation in self.violations],
        }


def check_layout(
    faces: FacesData,
    layout: Layout,
    *,
    margin_px: float | None = None,
    caption_gap_px: float | None = None,
    hair: HairAllowance | None = None,
    margin_source: str | None = None,
) -> ClearanceReport:
    """Validate video geometry and check both margins, retaining unmeasured status.

    A pass means no overlap with the faces Vision detected. It does not mean every
    visible face was found. The check measures head boxes (face plus hair allowance).
    """
    for key in ("width", "height"):
        if getattr(layout, key) != getattr(faces.video, key):
            raise ValidationError(
                f"layout.{key}", f"must equal video {key} {getattr(faces.video, key)}"
            )
    for key in ("elements", "captions"):
        for index, target in enumerate(getattr(layout, key)):
            if target.end_frame > faces.video.frames:
                raise ValidationError(
                    f"layout.{key}[{index}].end_frame",
                    f"must be at or before video frame count {faces.video.frames}",
                )
    margin = _margin(
        margin_px
        if margin_px is not None
        else layout.clearance_px
        if layout.clearance_px is not None
        else DEFAULT_FACE_CLEARANCE_PX,
        "margin_px",
    )
    source = (
        margin_source
        if margin_source is not None
        else "option"
        if margin_px is not None
        else "layout"
        if layout.clearance_px is not None
        else "default"
    )
    if source not in ("option", "layout", "style", "default"):
        raise ValidationError("margin_source", "must be option, layout, style, or default")
    allowance = (
        hair if hair is not None else layout.hair if layout.hair is not None else DEFAULT_HAIR
    )
    caption_gap = _margin(
        layout.caption_gap_px if caption_gap_px is None else caption_gap_px, "caption_gap_px"
    )
    captions = () if layout.measured is False else layout.captions
    if faces.status == "measured":
        violations = check_clearance(
            face_boxes_per_frame(faces),
            layout.elements,
            captions,
            margin_px=margin,
            caption_gap_px=caption_gap,
            hair=allowance,
        )
    else:
        violations = _ordered(_caption_violations(layout.elements, captions, caption_gap))
    status = "fail" if violations else "unmeasured" if faces.status == "unmeasured" else "pass"
    with_faces = sum(bool(frame) for frame in faces.frames) if faces.status == "measured" else None
    return ClearanceReport(
        status,
        faces.reason,
        margin,
        allowance,
        source,
        caption_gap,
        faces.video.fps,
        faces.video.frames,
        len(layout.elements),
        len(captions),
        tuple(violations),
        faces.times,
        faces.end_time,
        with_faces,
        faces.video.frames - with_faces if with_faces is not None else None,
    )
