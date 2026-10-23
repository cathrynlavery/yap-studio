"""Find one or two foreign frames at a join using nearby picture motion.

Score one-frame distances against a median and MAD from about one second on
each side, excluding steps within touch_frames + 1 of the join. A flash has
two changes within touch_frames of the join, enclosing one or two frames.
Each enclosed frame must differ from both neighbours by at least 0.85 of their
distance. Measured motion ratios were 0.51 to 0.74, while foreign frames were
about 1.0 to 1.33. Motion's summed distances were 1.07 to 1.49 times the
neighbour distance, with one at 1.91. Blend residuals were 0.04 to 0.23, so a
blend test alone cannot reject motion. Reject a blend that matches a mixture
of those neighbours within local motion, since contrast normalization can
inflate its two distances.
Confirm changes across an adjacent change too, so a nearby flash cannot make
ordinary frames look foreign. A clean cut supplies one change. Repeats,
missing frames and small catch-up steps alone are not flashes. Short reference
windows are skipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from yap_studio import video
from yap_studio.checks.model import Finding, format_time
from yap_studio.errors import ValidationError

REFERENCE_SECONDS = 1.0  # Local motion spans about one second on each side.
MIN_REFERENCE_STEPS = 3  # A median needs several steps outside the join.
MAD_SCALE = 1.4826  # Convert the median absolute deviation to a robust scale.
MEDIAN_SCALE_FLOOR = 0.05  # Keep nearly uniform motion from exaggerating scores.
DISTANCE_SCALE_FLOOR = 0.002  # Keep static pictures and codec noise finite.
CHANGE_SCORE = 5.0  # Match the detector's robust picture-change gate.
CHANGE_MOTION_RATIO = 3.0  # Changes must exceed ordinary motion by this factor.
MAX_FLASH_FRAMES = 2  # Only one or two stray frames count as a flash.
NEIGHBOUR_RATIO = 0.85  # Motion measured 0.51 to 0.74; foreign frames about 1.0 to 1.33.
BLEND_MOTION_RATIO = 1.0  # A fitted blend may differ by ordinary local motion.


def reference_window(
    count: int, frame: int, *, fps: float, touch_frames: int = 2
) -> tuple[int, int]:
    """Return the frame slice used by the steps and their motion reference."""
    radius = max(touch_frames + MAX_FLASH_FRAMES, round(REFERENCE_SECONDS * fps))
    return max(0, frame - radius - 1), min(count, frame + radius + 1)


def _blend_distance(
    before: np.ndarray, picture: np.ndarray, after: np.ndarray, valid: np.ndarray | None
) -> float:
    """Measure the residual from the closest mixture before normalization."""
    first, middle, last = (np.asarray(item, dtype=np.float64) for item in (before, picture, after))
    delta, offset = last - first, middle - first
    if valid is not None:
        delta, offset = delta[valid], offset[valid]
    weight = float(np.clip(np.sum(offset * delta) / max(float(np.sum(delta * delta)), 1e-12), 0, 1))
    return video.frame_distance(picture, (1 - weight) * first + weight * last, valid)


def flash_at(
    frames: np.ndarray,
    frame: int,
    *,
    fps: float,
    touch_frames: int = 2,
    valid: np.ndarray | None = None,
    ignored_windows: Sequence[tuple[int, int]] = (),
) -> dict:
    """Judge a flash near the first incoming frame, returning JSON-safe evidence.

    A reason describes an unmeasured window. An evaluated window has no reason,
    even when its steps do not enclose a flash. Step indices name the incoming
    frame of each pair, so a run [a, b] has change steps a and b + 1.
    """
    if frames.ndim != 3 or min(frames.shape[1:], default=0) < 1:
        raise ValidationError("frames", "must have shape [n, h, w] with non-empty pictures")
    if not isinstance(frame, (int, np.integer)):
        raise ValidationError("frame", "must be an integer frame index")
    if not math.isfinite(fps) or fps <= 0:
        raise ValidationError("fps", "must be positive and finite")
    if not isinstance(touch_frames, int) or touch_frames < 0:
        raise ValidationError("touch_frames", "must be a non-negative integer")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != frames.shape[1:]:
            raise ValidationError("valid", "must match the frame shape")
    frame = int(frame)
    first, stop = reference_window(len(frames), frame, fps=fps, touch_frames=touch_frames)
    result = {
        "flash": False,
        "reason": None,
        "run": None,
        "changes": [],
        "steps": [],
        "local_motion": None,
        "window": {"first_frame": first, "last_frame": stop - 1},
        "rule": {
            "reference_seconds": REFERENCE_SECONDS,
            "minimum_reference_steps": MIN_REFERENCE_STEPS,
            "reference_exclusion_frames": touch_frames + 1,
            "mad_scale": MAD_SCALE,
            "median_scale_floor": MEDIAN_SCALE_FLOOR,
            "distance_scale_floor": DISTANCE_SCALE_FLOOR,
            "change_score": CHANGE_SCORE,
            "change_motion_ratio": CHANGE_MOTION_RATIO,
            "maximum_flash_frames": MAX_FLASH_FRAMES,
            "neighbour_ratio": NEIGHBOUR_RATIO,
            "blend_motion_ratio": BLEND_MOTION_RATIO,
            "touch_frames": touch_frames,
        },
    }
    if not 0 <= frame < len(frames) or len(frames) < 3:
        return {**result, "reason": "not enough frames around the join"}
    if valid is not None and not valid.any():
        return {**result, "reason": "picture masked"}
    if not np.isfinite(frames[first:stop]).all():
        raise ValidationError("frames", "must contain finite grey values")
    distances = {
        index: video.frame_distance(frames[index - 1], frames[index], valid)
        for index in range(first + 1, stop)
    }
    reference = np.asarray(
        [value for index, value in distances.items() if abs(index - frame) > touch_frames + 1]
    )
    if len(reference) < MIN_REFERENCE_STEPS:
        return {**result, "reason": "not enough motion reference frames around the join"}
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(MAD_SCALE * mad, MEDIAN_SCALE_FLOOR * median, DISTANCE_SCALE_FLOOR)
    result["local_motion"] = {"median": median, "mad": mad, "scale": scale, "steps": len(reference)}

    def changed(distance: float) -> bool:
        return (
            distance - median >= CHANGE_SCORE * scale and distance >= CHANGE_MOTION_RATIO * median
        )

    steps = {
        index: {
            "frame": index,
            "distance": value,
            "score": (value - median) / scale,
            "change": changed(value),
        }
        for index, value in distances.items()
        if abs(index - frame) <= touch_frames
    }
    result["steps"] = list(steps.values())
    for start, into in steps.items():
        if not into["change"]:
            continue
        for length in range(1, MAX_FLASH_FRAMES + 1):
            end = start + length
            if any(first <= start and end <= stop for first, stop in ignored_windows):
                continue
            out = steps.get(end)
            if out is None or not out["change"]:
                continue
            # A neighbouring flash must not make its returning picture look stray.
            if (
                start >= 2
                and changed(distances.get(start - 1, 0.0))
                and not changed(video.frame_distance(frames[start - 2], frames[start], valid))
            ):
                continue
            if (
                end + 1 < len(frames)
                and changed(distances.get(end + 1, 0.0))
                and not changed(video.frame_distance(frames[end - 1], frames[end + 1], valid))
            ):
                continue
            neighbour_distance = video.frame_distance(frames[start - 1], frames[end], valid)
            sides = [
                {
                    "frame": index,
                    "before_distance": video.frame_distance(
                        frames[start - 1], frames[index], valid
                    ),
                    "after_distance": video.frame_distance(frames[index], frames[end], valid),
                    "blend_distance": _blend_distance(
                        frames[start - 1], frames[index], frames[end], valid
                    ),
                }
                for index in range(start, end)
            ]
            if any(
                min(side["before_distance"], side["after_distance"])
                < NEIGHBOUR_RATIO * neighbour_distance
                for side in sides
            ):
                continue
            if any(
                side["blend_distance"] <= BLEND_MOTION_RATIO * max(median, scale) for side in sides
            ):
                continue
            result.update(
                flash=True,
                run={"first_frame": start, "last_frame": end - 1, "length": length},
                changes=[into, out],
                neighbour_distance=neighbour_distance,
                run_distances=sides,
            )
            return result
    return result


def flash_finding(
    evidence: dict, cut: int, time_s: float, *, fps: float, video_offset_s: float = 0.0
) -> Finding:
    """Describe a measured flash on the render's container clock."""
    run = evidence["run"]
    count = run["length"]
    start = video_offset_s + run["first_frame"] / fps
    return Finding(
        "picture.flash",
        "fail",
        f"Cut {cut} at {format_time(time_s)}: a stray frame flashes at the join "
        f"({count} {'frame' if count == 1 else 'frames'} at {format_time(start)} "
        f"{'matches' if count == 1 else 'match'} neither side).",
        cut=cut,
        time_s=time_s,
        suggested_action="Re-render the cut so no stray frame shows between the two segments.",
        evidence=evidence,
    )
