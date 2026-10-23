"""Pure camera curves shared by picture rendering and face geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yap_studio.faces.model import Box

SNAP_IN_FRAMES = 2
SNAP_OUT_FRAMES = 4
EASES = ("cut", "snap", "linear", "ease-in-out")
SCALE_LIMIT = 2.0
HEAD_TOP = 0.4
FACE_MARGIN_PX = 24
CAPTION_SLACK_PX = 4  # Face mapping and picture rounding.
MOTION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ZoomKey:
    frame: int
    scale: float
    center: tuple[float, float]
    ease: str = "cut"


def snap_frames(a: ZoomKey, b: ZoomKey) -> int:
    """Enter faster than returning to the wider picture."""
    return SNAP_IN_FRAMES if b.scale > a.scale else SNAP_OUT_FRAMES


@dataclass(frozen=True)
class ZoomMove:
    start_frame: int
    end_frame: int
    from_scale: float
    to_scale: float


@dataclass(frozen=True)
class ZoomCurve:
    start_frame: int
    end_frame: int
    keys: tuple[ZoomKey, ...] = ()
    peak: float | None = None
    swell_center: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        by_frame = {key.frame: key for key in self.keys}
        object.__setattr__(self, "keys", tuple(by_frame[frame] for frame in sorted(by_frame)))

    @classmethod
    def swell(cls, start: int, end: int, peak: float,
              center: tuple[float, float]) -> ZoomCurve:
        """Keep the original symmetric swell."""
        return cls(start, end, peak=peak, swell_center=tuple(center))

    @classmethod
    def from_payload(cls, start_frame: int, end_frame: int,
                     payload: Mapping[str, Any]) -> ZoomCurve:
        """Read a resolved timeline payload."""
        if "curve" in payload:
            return cls(start_frame, end_frame, tuple(
                ZoomKey(key["frame"], key["scale"], tuple(key["center"]), key["ease"])
                for key in payload["curve"]))
        return cls.swell(start_frame, end_frame, payload["peak_scale"], tuple(payload["center"]))

    def at(self, frame: int) -> tuple[float, float, float]:
        """Evaluate the framing; callers apply the exclusive picture window."""
        if self.peak is not None:
            assert self.swell_center is not None
            scale = 1 + (self.peak - 1) * math.sin(
                math.pi * (frame - self.start_frame) / (self.end_frame - self.start_frame)) ** 2
            return scale, *self.swell_center
        if not self.keys:
            return 1.0, .5, .42
        a = self.keys[0]
        for b in self.keys[1:]:
            if frame < b.frame:
                if b.ease == "cut":
                    break
                start = max(a.frame, b.frame - snap_frames(a, b)) if b.ease == "snap" else a.frame
                if frame <= start:
                    break
                t = (frame - start) / (b.frame - start)
                weight = (1 - (1 - t)**3 if b.ease == "snap" else
                          (1 - math.cos(math.pi * t)) / 2 if b.ease == "ease-in-out" else t)
                return (a.scale * (b.scale / a.scale) ** weight,
                        *(va + (vb - va) * weight
                          for va, vb in zip(a.center, b.center, strict=True)))
            a = b
        return a.scale, *a.center

    def scales(self, first: int, last: int) -> list[float]:
        """Sample the exclusive picture window on the output clock."""
        return [self.at(frame)[0] if self.start_frame <= frame < self.end_frame else 1.0
                for frame in range(first, last)]

    def max_scale(self) -> float:
        """Return the largest framing scale."""
        return self.peak if self.peak is not None else max((k.scale for k in self.keys), default=1)

    def moves(self) -> tuple[ZoomMove, ...]:
        """Describe scale changes, excluding the implicit return at the window end."""
        if self.peak is not None:
            middle = (self.start_frame + self.end_frame) // 2
            return (ZoomMove(self.start_frame, middle, 1, self.peak),
                    ZoomMove(middle, self.end_frame, self.peak, 1))
        result = []
        if self.keys and self.keys[0].scale != 1:
            first = self.keys[0]
            result.append(ZoomMove(first.frame, first.frame, 1, first.scale))
        for a, b in zip(self.keys, self.keys[1:], strict=False):
            if a.scale == b.scale:
                continue
            start = (b.frame if b.ease == "cut" else max(a.frame, b.frame - snap_frames(a, b))
                     if b.ease == "snap" else a.frame)
            result.append(ZoomMove(start, b.frame, a.scale, b.scale))
        return tuple(result)

    def expressions(self, first: int) -> tuple[str, str, str]:
        """Build ffmpeg expressions on a piece's local frame clock."""
        if self.peak is not None:
            assert self.swell_center is not None
            return (f"(1+{self.peak - 1:.12f}*sin(PI*(n+{first - self.start_frame})/"
                    f"{self.end_frame - self.start_frame})^2)",
                    *(f"{value:.12f}" for value in self.swell_center))
        if not self.keys:
            return "1.000000000000", "0.500000000000", "0.420000000000"
        frame = f"(n+{first})"

        def expression(axis: int) -> str:
            values = [(key.scale, *key.center)[axis] for key in self.keys]
            if all(value == values[0] for value in values):
                return f"{values[0]:.12f}"
            result = f"{values[-1]:.12f}"
            for index in reversed(range(len(self.keys) - 1)):
                a, b = self.keys[index:index + 2]
                va, vb = values[index:index + 2]
                segment = f"{va:.12f}"
                if b.ease != "cut" and va != vb:
                    start = (max(a.frame, b.frame - snap_frames(a, b))
                             if b.ease == "snap" else a.frame)
                    t = f"(({frame}-{start})/{b.frame - start})"
                    weight = (f"(1-pow(1-{t},3))" if b.ease == "snap" else
                              f"((1-cos(PI*{t}))/2)" if b.ease == "ease-in-out" else t)
                    segment = (f"({va:.12f}*pow({vb / va:.12f},{weight}))" if axis == 0 else
                               f"({va:.12f}+({vb:.12f}-{va:.12f})*{weight})")
                    segment = f"if(lt({frame},{start}),{va:.12f},{segment})"
                result = f"if(lt({frame},{b.frame}),{segment},{result})"
            return f"if(lt({frame},{self.keys[0].frame}),{values[0]:.12f},{result})"

        return tuple(expression(axis) for axis in range(3))


def punch_keys(start: int, end: int, scale: float, center: tuple[float, float],
               ease: str = "snap") -> tuple[ZoomKey, ...]:
    """Punch in, hold, and land back on the end word."""
    if ease == "cut":
        return ZoomKey(start, scale, center), ZoomKey(end, 1, center)
    a, b = ZoomKey(start, 1, center), ZoomKey(end, scale, center, "snap")
    return (a, ZoomKey(start + snap_frames(a, b), scale, center, "snap"),
            ZoomKey(end - snap_frames(b, a), scale, center), ZoomKey(end, 1, center, "snap"))


def push_keys(start: int, end: int, scale: float, center: tuple[float, float],
              ease: str = "linear", ramp_frames: int | None = None) -> tuple[ZoomKey, ...]:
    """Reach a framing and hold it until the window ends."""
    return (ZoomKey(start, 1, center),
            ZoomKey(end if ramp_frames is None else start + ramp_frames, scale, center, ease))


@dataclass(frozen=True)
class MotionFlaw:
    frame: int
    kind: str
    change: float


def declared_moves(curve: ZoomCurve) -> tuple[ZoomMove, ...]:
    """Include the cut back to 1 when the picture window ends."""
    moves = curve.moves()
    last = curve.at(curve.end_frame)[0]
    if last != 1:
        moves += (ZoomMove(curve.end_frame, curve.end_frame, last, 1),)
    return moves


def scale_motion(scales: Sequence[float], first_frame: int, moves: Sequence[ZoomMove], *,
                 tolerance: float = MOTION_TOLERANCE) -> tuple[MotionFlaw, ...]:
    """Flag scale steps that no declared cut, snap or move explains."""
    logs = [math.log(scale) for scale in scales]
    steps = [b - a for a, b in zip(logs, logs[1:], strict=False)]
    cuts = {move.start_frame for move in moves if move.start_frame == move.end_frame}
    flaws = []
    for index, change in enumerate(steps):
        frame = first_frame + index + 1
        if frame in cuts:
            continue
        move = next((move for move in moves if move.start_frame < frame <= move.end_frame), None)
        if move is None:
            if abs(change) > tolerance:
                flaws.append(MotionFlaw(frame, "jump", change))
            continue
        direction = (move.to_scale > move.from_scale) - (move.to_scale < move.from_scale)
        backwards = change * direction < -tolerance
        stall = (0 < index < len(steps) - 1
                 and move.start_frame < frame - 1 and frame + 1 <= move.end_frame
                 and min(abs(steps[index - 1]), abs(steps[index + 1])) > tolerance
                 and abs(change) <= .25 * min(abs(steps[index - 1]), abs(steps[index + 1])))
        if backwards or stall:
            flaws.append(MotionFlaw(frame, "jerk", change))
    return tuple(flaws)


def _boxes_at(boxes_by_frame: Mapping[int, Sequence[Box]] | Sequence[Sequence[Box]],
              frame: int) -> Sequence[Box]:
    if frame < 0:
        return ()
    if isinstance(boxes_by_frame, Mapping):
        return boxes_by_frame.get(frame, ())
    try:
        return boxes_by_frame[frame]
    except IndexError:
        return ()


def face_center(boxes_by_frame: Mapping[int, Sequence[Box]] | Sequence[Sequence[Box]],
                frames: Sequence[int],
                scale: float, *, bottom: float = 1920,
                bottom_margin: float = FACE_MARGIN_PX) -> tuple[tuple[float, float], bool] | None:
    """Aim at the speaker's head and keep the face clear of edges and captions."""
    boxes = [max(per_frame, key=lambda box: box.w * box.h)
             for frame in frames if (per_frame := _boxes_at(boxes_by_frame, frame))]
    if not boxes:
        return None
    center = []
    fits = True
    for size, limit, lows, highs, preferred, end_margin in (
        (1080, 1080, [box.x for box in boxes], [box.x1 for box in boxes],
         [(box.x + box.x1) / 2 for box in boxes], FACE_MARGIN_PX),
        (1920, bottom, [box.y for box in boxes], [box.y1 for box in boxes],
         [(box.y - HEAD_TOP * box.h + box.y1) / 2 for box in boxes], bottom_margin),
    ):
        middle = median(preferred)
        axis_fits = False
        for low_margin, high_margin in ((FACE_MARGIN_PX, end_margin), (0, 0)):
            lo, hi = (max(0, (max(highs) * scale - limit + high_margin) / (scale - 1)),
                      min(size, (min(lows) * scale - low_margin) / (scale - 1))) if scale > 1 else (
                          0, size)
            axis_fits = (lo <= hi if scale > 1 else
                         min(lows) >= low_margin and max(highs) <= limit - high_margin)
            if axis_fits:
                break
        fits = fits and axis_fits
        center.append(min(hi, max(lo, middle)) / size if axis_fits
                      else min(size, max(0, middle)) / size)
    return tuple(center), fits


def face_leaves_frame(
    curve: ZoomCurve, boxes_by_frame: Mapping[int, Sequence[Box]] | Sequence[Sequence[Box]],
    *, bottom: float = 1920,
) -> tuple[int, str] | None:
    """Find the first speaker face edge pushed outside or into captions."""
    for frame in range(curve.start_frame, curve.end_frame):
        scale, cx, cy = curve.at(frame)
        cx, cy = cx * 1080, cy * 1920
        per_frame = _boxes_at(boxes_by_frame, frame)
        if not per_frame:
            continue
        box = max(per_frame, key=lambda box: box.w * box.h)
        left = cx + (box.x - cx) * scale
        top = cy + (box.y - cy) * scale
        right = cx + (box.x1 - cx) * scale
        chin = cy + (box.y1 - cy) * scale
        if ((left < 0 and left < box.x - 1e-9)
                or (top < 0 and top < box.y - 1e-9)
                or (right > 1080 and right > box.x1 + 1e-9)
                or (chin > 1920 and chin > box.y1 + 1e-9)):
            return frame, "frame"
        if chin > bottom and chin > box.y1 + 1e-9:
            return frame, "captions"
    return None
