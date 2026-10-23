"""Find picture jumps and choose a bounded interpolation window."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.faces.model import Box, FacesData
from yap_studio.faces.project import cut_faces
from yap_studio.joins import helper
from yap_studio.joins.helper import JoinError
from yap_studio.joins.steps import jumps
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

# Picture change guards against ghosts from entering or leaving hands or objects;
# face shift and size limits keep the glide from reading as a swoop.
WINDOW_FRAMES = 6
FACE_SHIFT_MAX = 0.25
FACE_SCALE_MAX = 0.10
PICTURE_CHANGE_MAX = 5.5
COVERS = frozenset({"zoom", "insert", "transition", "background", "moment", "grade_compare"})


def _takes_over_picture(item: ResolvedElement) -> bool:
    return item.kind in COVERS or (
        item.kind == "overlay" and item.payload.get("placement") == "tall_behind")


def limits() -> dict[str, float]:
    """Return the smoothing limits."""
    return {"face_shift": FACE_SHIFT_MAX, "face_scale": FACE_SCALE_MAX,
            "picture_change": PICTURE_CHANGE_MAX}


def fallback(moves: dict) -> str:
    """List a punch for smooth or punch jump cuts, even with zooms off."""
    return "punch" if moves.get("jump_cuts") in {"smooth", "punch"} else "hard"


@dataclass
class JoinDecision:
    """A picture boundary, its measured changes and chosen handling."""

    index: int
    kind: str
    join_frame: int
    time_s: float
    start_frame: int
    frames: int
    before_word: dict | None
    after_word: dict | None
    handling: str
    reason: str
    covered_by: str | None = None
    face_shift: float | None = None
    face_scale: float | None = None
    picture_change: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Round measurements for the receipt without changing threshold decisions."""
        return {key: round(value, 6) if isinstance(value, float) else value
                for key, value in asdict(self).items()}


@dataclass
class JoinPlan:
    """Serializable decisions without retained full-resolution frames."""

    mode: str
    fallback: str
    joins: list[JoinDecision]
    width: int = 1080
    height: int = 1920

    def to_dict(self) -> dict[str, Any]:
        """Return the public plan and counts, without raw pixels."""
        return {"schema": "yap-studio/joins@1", "mode": self.mode, "fallback": self.fallback,
                "window_frames": WINDOW_FRAMES, "limits": limits(),
                "joins": [join.to_dict() for join in self.joins],
                "counts": {kind: sum(j.handling == kind for j in self.joins)
                           for kind in ("smooth", "hard", "punch", "covered", "step_in")}}


def decode_frames(cut: Path, indices: list[int], *, width: int,
                  height: int, extra_timeout_s: float = 0) -> Iterator[tuple[int, bytes]]:
    """Stream only the requested frames; allow time for work between reads."""
    ordered = sorted(set(indices))
    if not ordered:
        return
    select = "+".join(f"eq(n\\,{index})" for index in ordered)
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(cut), "-map", "0:v:0", "-an",
        "-vf", f"select={select},"
        "scale=in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int,"
        "format=rgb48le", "-fps_mode", "passthrough", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "rgb48le", "-frames:v", str(len(ordered)), "pipe:1",
    ]
    size = width * height * 6
    timeout = max(ffmpeg.DEFAULT_TIMEOUT_S, 30 + (ordered[-1] + 1) / 30 * 12) + extra_timeout_s
    with closing(ffmpeg.run_stream(command, timeout_s=timeout,
                                   chunk_size=min(size, 64 * 1024))) as stream:
        buffer = bytearray()
        count = 0
        for chunk in stream:
            buffer.extend(chunk)
            while len(buffer) >= size:
                if count == len(ordered):
                    raise JoinError("Smooth join frame extraction returned an extra "
                                    "frame sequence.")
                raw = bytes(memoryview(buffer)[:size])
                del buffer[:size]
                index = ordered[count]
                count += 1
                yield index, raw
                del raw
        if buffer or count != len(ordered):
            raise JoinError("Smooth join frame extraction returned an incomplete frame sequence.")


def _nearest(faces: FacesData | None, frame: int, piece: dict) -> Box | None:
    if faces is None or faces.status != "measured":
        return None
    first, last = piece["out_f0"], piece["out_f0"] + piece["n_frames"]
    for index in sorted(range(max(first, frame - 2), min(last, frame + 3)),
                        key=lambda i: (abs(i - frame), i)):
        boxes = [face.box for face in faces.faces_at(index)
                 if face.source == "detected" and face.box.h > 0]
        if boxes:
            return max(boxes, key=lambda box: box.w * box.h)
    return None


def _averaged_luma(raw: bytes, width: int, height: int) -> np.ndarray:
    rgb = np.frombuffer(raw, dtype="<u2").reshape(height, width, 3).astype(np.float64) / 257
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    h, w = height // 8, width // 8
    return luma[:h * 8, :w * 8].reshape(h, 8, w, 8).mean(axis=(1, 3))


def _word(words: list[dict], join: int, *, before: bool) -> dict | None:
    candidates = [word for word in words if (word["end"] * 30 <= join + 1e-6 if before
                                           else word["start"] * 30 >= join - 1e-6)]
    if not candidates:
        return None
    chosen = (max(candidates, key=lambda w: w["end"]) if before
              else min(candidates, key=lambda w: w["start"]))
    return {key: chosen[key] for key in ("id", "text", "segment")}


def build_plan(project_dir: Path, timeline: ResolvedTimeline, pieces: list[dict],
               words: list[dict], moves: dict, *, faces: FacesData | None = None,
               frames: dict[int, bytes] | None = None, cut: Path | None = None) -> JoinPlan:
    """Decide every actual picture jump, measuring only eligible smooth windows."""
    moves = {**moves, "jump_cuts": "hard"} if timeline.camera_3d else moves
    mode = moves.get("jump_cuts", "hard")
    if mode == "smooth" and (reason := helper.unavailable_reason()) is not None:
        raise JoinError(reason, fix=reason)
    assert WINDOW_FRAMES > 0 and WINDOW_FRAMES % 2 == 0
    result = JoinPlan(mode, fallback(moves), [])
    source = cut or Path(project_dir) / "cut/cut.mp4"
    pending = []
    half = WINDOW_FRAMES // 2
    step_jumps = {j["frame"]: j for j in (timeline.steps or {}).get("jumps", [])}
    step_shots = (timeline.steps or {}).get("shots", [])
    step_changes = {after["start"]: any(before[key] != after[key] for key in ("scale", "center"))
                    for before, after in zip(step_shots, step_shots[1:], strict=False)}
    authored = [item for item in timeline.elements if item.lane != "engine"]
    for join, kind, previous, next_piece in jumps(pieces):
        a, b = join - half - 1, join + half
        decision = JoinDecision(len(result.joins), kind, join, join / 30, join - half,
                                WINDOW_FRAMES, _word(words, join, before=True),
                                _word(words, join, before=False), mode, "")
        result.joins.append(decision)
        if mode == "step_in":
            step = step_jumps.get(join)
            if step is None:
                raise JoinError("The resolved timeline has no Step in decision for this cut.",
                                fix="Run yap-studio cut DIR again.")
            decision.handling = "step_in" if step_changes[join] else "hard"
            decision.reason = step["reason"]
            continue
        covering = next((item for item in authored if _takes_over_picture(item)
                         and max(join - 1, item.start_frame) < min(join + 1, item.end_frame)),
                        None)
        overlapping = next((item for item in authored if _takes_over_picture(item)
                            and max(a, item.start_frame) < min(b + 1, item.end_frame)), None)
        if covering is not None:
            decision.handling, decision.covered_by = "covered", covering.id
            decision.reason = f"A {covering.kind} covers this join ({covering.id})."
        elif mode != "smooth":
            decision.reason = ("The style keeps hard cuts." if mode == "hard"
                               else "The style covers jump cuts with a punch.")
        elif overlapping is not None:
            decision.handling = result.fallback
            decision.reason = (f"A {overlapping.kind} ({overlapping.id}) "
                               "overlaps the smoothing window.")
        elif (a < previous["out_f0"] or a >= previous["out_f0"] + previous["n_frames"]
              or b < next_piece["out_f0"] or b >= next_piece["out_f0"] + next_piece["n_frames"]):
            decision.handling = result.fallback
            decision.reason = (f"A piece next to the join is shorter than the "
                               f"{WINDOW_FRAMES}-frame window.")
        else:
            pending.append((decision, previous, next_piece, a, b))
    if not pending:
        return result
    measured = faces if faces is not None else cut_faces(Path(project_dir), source)
    result.width, result.height = measured.video.width, measured.video.height
    eligible = []
    for decision, previous, next_piece, a, b in pending:
        box_a, box_b = _nearest(measured, a, previous), _nearest(measured, b, next_piece)
        if box_a is None or box_b is None:
            decision.handling = result.fallback
            decision.reason = "No face was detected next to the join."
            continue
        decision.face_shift = math.hypot((box_b.x + box_b.w / 2) - (box_a.x + box_a.w / 2),
                                         (box_b.y + box_b.h / 2) - (box_a.y + box_a.h / 2)) / (
                                             (box_a.h + box_b.h) / 2)
        decision.face_scale = abs(math.log(box_b.h / box_a.h))
        eligible.append((decision, a, b))
    needed = [frame for _, a, b in eligible for frame in (a, b)]
    reader = (decode_frames(source, needed, width=result.width, height=result.height)
              if frames is None else ((index, frames[index]) for index in sorted(set(needed))))
    with closing(reader):
        averages = {index: _averaged_luma(raw, result.width, result.height)
                    for index, raw in reader}
    for decision, a, b in eligible:
        decision.picture_change = float(np.abs(averages[a] - averages[b]).mean())
        assert decision.face_shift is not None and decision.face_scale is not None
        if decision.face_shift > FACE_SHIFT_MAX:
            reason = (f"The face moved {decision.face_shift:.2f} face heights; "
                      f"the limit is {FACE_SHIFT_MAX:g}.")
        elif decision.face_scale > FACE_SCALE_MAX:
            reason = (f"The face size changed {math.expm1(decision.face_scale):.0%}; "
                      f"the limit is {FACE_SCALE_MAX:.0%}.")
        elif decision.picture_change > PICTURE_CHANGE_MAX:
            reason = (f"The picture changed {decision.picture_change:.1f}; "
                      f"the limit is {PICTURE_CHANGE_MAX:g}. "
                      "A hand or object entering or leaving would ghost.")
        else:
            decision.reason = (f"The face moved {decision.face_shift:.2f} face heights "
                               f"and the picture changed {decision.picture_change:.1f}.")
            continue
        decision.handling, decision.reason = result.fallback, reason
    previous_after = -1
    for decision in result.joins:
        if decision.handling != "smooth":
            continue
        if decision.start_frame - 1 < previous_after:
            decision.handling = result.fallback
            decision.reason = "Another smoothed join is too close."
        else:
            previous_after = decision.start_frame + decision.frames
    return result
