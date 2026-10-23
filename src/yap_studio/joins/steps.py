"""Plan still framing changes on the output cut clock."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from yap_studio.faces.clearance import gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.zoom_curve import ZoomCurve, ZoomKey, face_center, face_leaves_frame

if TYPE_CHECKING:
    from yap_studio.project.model import CutWord

LADDER = (1.15, 1.3, 1.45)
STEP_MIN_FRAMES = 15
PAUSE_RESET_S = 0.5
NO_FACE = "No face was measured in this shot."


def jumps(pieces: Sequence[dict]) -> Iterator[tuple[int, str, dict, dict]]:
    """Yield actual picture jumps, including section and take changes."""
    for previous, following in zip(pieces, pieces[1:], strict=False):
        take_changed = previous["take"] != following["take"]
        if not take_changed and following["src_f0"] == previous["src_f1"]:
            continue
        kind = "take" if take_changed else (
            "story" if previous["segment_id"] != following["segment_id"] else "removal")
        yield following["out_f0"], kind, previous, following


@dataclass(frozen=True)
class StepShot:
    """One still framing between picture jumps."""

    start: int
    end: int
    level: int
    scale: float
    center: tuple[float, float] | None
    reason: str


@dataclass(frozen=True)
class StepJump:
    """The framing decision at a cut."""

    frame: int
    kind: str
    scale_before: float
    scale_after: float
    handling: str
    reason: str


@dataclass(frozen=True)
class StepPlan:
    """Deterministic shots and cut decisions without media or I/O."""

    shots: tuple[StepShot, ...]
    jumps: tuple[StepJump, ...]

    def summary(self) -> dict[str, int]:
        """Count framing changes and holds."""
        counts = {kind: sum(j.handling == kind for j in self.jumps)
                  for kind in ("step", "reset", "shift", "hold")}
        return {"jumps": len(self.jumps),
                "changed": counts["step"] + counts["reset"] + counts["shift"], **counts}

    def to_dict(self) -> dict:
        """Return the resolved timeline's plain framing record."""
        return {"summary": self.summary(), "jumps": [asdict(j) for j in self.jumps],
                "shots": [{**asdict(s), "center": list(s.center) if s.center else None}
                          for s in self.shots]}


def _blocking_card(cards: Sequence[tuple[int, int, str, Box]],
                   boxes: Mapping[int, Sequence[Box]], start: int, end: int, scale: float,
                   center: tuple[float, float], margin: float) -> str | None:
    """Check each overlapping face with the supplied card's head clearance rule."""
    cx, cy = center[0] * 1080, center[1] * 1920
    for first, last, card_id, card_box in cards:
        for frame in range(max(start, first), min(end, last)):
            for face in boxes.get(frame, ()):
                mapped = Box(cx + (face.x - cx) * scale, cy + (face.y - cy) * scale,
                             face.w * scale, face.h * scale)
                if gap_px(card_box, head_box(mapped)) < margin:
                    return card_id
    return None


def plan_steps(pieces: Sequence[dict], words: Sequence[dict | CutWord], total_frames: int,
               covers: Sequence[tuple[int, int, str, str]],
               boxes: Mapping[int, Sequence[Box]] | None,
               cards: Sequence[tuple[int, int, str, Box]], *, bottom: float,
               bottom_margin: float, card_margin: float,
               wide_from: Sequence[tuple[int, str]] = ()) -> StepPlan:
    """Walk in on story and removal cuts and return wide when the picture needs it."""
    boundaries = list(jumps(pieces))
    wide = dict(wide_from)
    starts = sorted(w["start"] if isinstance(w, dict) else w.start for w in words)
    ends = sorted(w["end"] if isinstance(w, dict) else w.end for w in words)
    shots = [StepShot(0, boundaries[0][0] if boundaries else total_frames,
                      0, 1.0, None, wide.get(0, NO_FACE if boxes is None else "Start wide."))]
    decisions = []
    for index, (frame, kind, _, _) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else total_frames
        previous = shots[-1]
        level, center = 0, None
        wide_reason = wide.get(frame)
        cover = next((c for c in covers if frame < c[1] and c[0] < end), None)
        before = bisect_right(ends, frame / 30 + 1e-9) - 1
        after = bisect_left(starts, frame / 30 - 1e-9)
        pause = (before >= 0 and after < len(starts)
                 and starts[after] - ends[before] >= PAUSE_RESET_S - 1e-9)
        short = end - frame < STEP_MIN_FRAMES
        if wide_reason is not None:
            reason = wide_reason
        elif cover:
            reason = f"A {cover[3]} covers this shot ({cover[2]})."
        elif kind == "take":
            reason = "The take changes here."
        elif pause:
            reason = "A long pause comes before this cut."
        elif previous.level == 3:
            reason = "Back to wide after three steps."
        elif short:
            level, center = previous.level, previous.center
            reason = "The next shot is shorter than 0.5 s."
        else:
            level = previous.level + 1
            reason = f"Steps in to {LADDER[level - 1]}x."
        if boxes is None and wide_reason is None:
            level, reason = 0, NO_FACE
        elif level:
            scale = LADDER[level - 1]
            aimed = face_center(boxes, range(frame, end), scale, bottom=bottom,
                                bottom_margin=bottom_margin)
            fits = aimed is not None and aimed[1]
            if short and center is not None and fits:
                curve = ZoomCurve(frame, end, (ZoomKey(frame, scale, center),))
                fits = face_leaves_frame(curve, boxes, bottom=bottom) is None
            if not fits:
                level = 0
                reason = NO_FACE if aimed is None else (
                    f"The face would leave the safe area at {scale}x.")
            elif not short:
                center = aimed[0]
            if level and (card_id := _blocking_card(
                cards, boxes, frame, end, scale, center, card_margin,
            )) is not None:
                reason = f"A card needs the room around your head ({card_id})."
                lower_levels = range(level - 1, 0, -1) if not short else ()
                for lower in lower_levels:
                    scale = LADDER[lower - 1]
                    aimed = face_center(boxes, range(frame, end), scale, bottom=bottom,
                                        bottom_margin=bottom_margin)
                    if aimed is None or not aimed[1]:
                        continue
                    blocked = _blocking_card(cards, boxes, frame, end, scale, aimed[0],
                                             card_margin)
                    if blocked is None:
                        level, center = lower, aimed[0]
                        break
                    reason = f"A card needs the room around your head ({blocked})."
                else:
                    level = 0
        scale = LADDER[level - 1] if level else 1.0
        if level and level == previous.level and previous.center is not None:
            curve = ZoomCurve(frame, end, (ZoomKey(frame, scale, previous.center),))
            if (face_leaves_frame(curve, boxes, bottom=bottom) is None
                    and _blocking_card(cards, boxes, frame, end, scale, previous.center,
                                       card_margin) is None):
                center = previous.center
        center = center if level else None
        shots.append(StepShot(frame, end, level, scale, center, reason))
        handling = ("reset" if wide_reason is not None or scale < previous.scale
                    else "step" if scale > previous.scale
                    else "shift" if center != previous.center else "hold")
        decisions.append(StepJump(frame, kind, previous.scale, scale, handling, reason))
    return StepPlan(tuple(shots), tuple(decisions))
