"""Check the final's colour against its retained ungraded output-grid picture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yap_studio import ffmpeg, receipt, video
from yap_studio.checks.model import Finding, format_time
from yap_studio.edl.model import EditList
from yap_studio.grade import GradeSpec, apply_rgb
from yap_studio.plan import Plan

# Allow half the visible grade change, bounded for encode noise and scaling error.
GRADE_ERROR_FLOOR = 2.5
GRADE_ERROR_CEILING = 10.0
GRADE_ERROR_MAGNITUDE_FRACTION = 0.5
GRADE_SAMPLE_LIMIT = 24
GRADE_MIN_FRAMES = 6
GRADE_WIDTH = 90
GRADE_HEIGHT = 160


def check_grade(render: Path, edits: EditList, plan: Plan,
                info: ffmpeg.MediaInfo) -> tuple[list[Finding], dict]:
    """Measure visible picture colour, warning when a declared grade cannot be checked."""
    def skipped(reason: str) -> tuple[list[Finding], dict]:
        return [], {"ran": False, "reason": reason}

    if edits.grade is None:
        return skipped("no grade in the edit list")
    spec = GradeSpec.from_dict({key: edits.grade[key]
                               for key in ("look", "strength", "version", "color")
                               if key in edits.grade})
    if not spec.active:
        return skipped("no active grade in the edit list")

    def valid_pixels(frame: int) -> np.ndarray:
        rects = [region.rect for region in edits.picture_regions
                 if not region.camera_3d and region.start <= frame / plan.fps < region.end]
        return ~video.exclusion_mask((GRADE_HEIGHT, GRADE_WIDTH), rects,
                                      info.width or 0, info.height or 0)

    count = info.video_frames or 0
    candidates = []
    for frame in range(count):
        time = frame / plan.fps
        if any(start <= time < end for start, end in edits.picture_masks):
            continue
        if any(hold.start_frame <= frame < hold.end_frame for hold in edits.holds):
            continue
        if any(start <= frame < end for start, end in edits.grade.get("skip_frames", [])):
            continue
        if edits.picture_regions and not valid_pixels(frame).any():
            continue
        candidates.append(frame)
    usable = len(candidates)

    def not_measured(reason: str) -> tuple[list[Finding], dict]:
        evidence = {"usable_frames": usable, "minimum_frames": GRADE_MIN_FRAMES}
        action = "Check the final by eye or re-render with yap-studio assemble."
        finding = Finding(
            "picture.grade", "warn", f"Picture colour grade not measured: {reason}. {action}",
            suggested_action=action, evidence=evidence,
        )
        return [finding], {"ran": False, "reason": reason, "evidence": evidence}

    if usable < GRADE_MIN_FRAMES:
        return not_measured(f"{usable} usable frames (needs {GRADE_MIN_FRAMES})")
    reference = Path(edits.grade["reference"])
    if not reference.is_file():
        return not_measured(f"grade reference is missing ({usable} usable frames)")
    if receipt.sha256_file(reference) != edits.grade["reference_sha256"]:
        return not_measured(f"grade reference sha256 has changed ({usable} usable frames)")
    frames = [candidates[i] for i in np.unique(np.linspace(
        0, usable - 1, min(GRADE_SAMPLE_LIMIT, usable), dtype=int))]
    masks = [valid_pixels(frame) for frame in frames]
    final = video.rgb_frames(render, frames, rate=plan.fps)
    original = video.rgb_frames(reference, frames, rate=plan.fps)
    if len(final) != len(frames) or len(original) != len(frames):
        return not_measured("grade reference or final does not cover the sampled output frames "
                            f"({usable} usable frames)")
    expected = apply_rgb(original / 255, spec) * 255
    errors = [float(np.abs(actual.astype(float) - wanted)[valid].mean())
              for actual, wanted, valid in zip(final, expected, masks, strict=True)]
    plain_errors = [float(np.abs(actual.astype(float) - plain)[valid].mean())
                    for actual, plain, valid in zip(final, original, masks, strict=True)]
    magnitudes = [float(np.abs(wanted - plain)[valid].mean())
                  for wanted, plain, valid in zip(expected, original, masks, strict=True)]
    median = float(np.median(errors))
    magnitude = float(np.median(magnitudes))
    limit = min(GRADE_ERROR_CEILING, max(GRADE_ERROR_FLOOR,
                                       GRADE_ERROR_MAGNITUDE_FRACTION * magnitude))
    looks_ungraded = float(np.median(plain_errors)) < median
    evidence = {"frames": frames, "frame_errors": errors, "median_error": median,
                "grade_magnitude": magnitude, "limit": limit, "threshold": limit,
                "looks_ungraded": looks_ungraded,
                "reference_error": float(np.median(plain_errors))}
    coverage = {"ran": True, "reason": None, "evidence": evidence}
    if median <= limit:
        return [], coverage
    first = next(frame for frame, error in zip(frames, errors, strict=True)
                 if error > limit)
    time_s = float(first / plan.fps)
    appearance = ("The final looks ungraded." if looks_ungraded
                  else "The final has different colour.")
    finding = Finding(
        "picture.grade", "fail",
        f"Colour differs from {format_time(time_s)} by {median:.2f} RGB levels "
        f"(median; limit {limit:g}). {appearance}",
        time_s=time_s, suggested_action="Re-render with yap-studio assemble.", evidence=evidence,
    )
    return [finding], coverage
