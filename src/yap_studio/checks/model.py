"""Shared findings, creator-facing timestamps, and check thresholds."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from yap_studio.errors import ValidationError

Severity = Literal["fail", "warn", "info"]

CHECK_IDS: tuple[str, ...] = (
    "join.click",
    "join.gap",
    "join.words",
    "join.frames",
    "join.listen",
    "join.relisten",
    "join.second_model",
    "sync.av_offset",
    "sync.drift",
    "sync.unmeasured",
    "plan.picture",
    "plan.frames",
    "plan.samples",
    "plan.holds",
    "edit_list.evidence",
    "structure.av_length",
    "structure.audio_timestamps",
    "loudness.integrated",
    "loudness.true_peak",
    "loudness.parts",
    "picture.black",
    "picture.freeze",
    "picture.flash",
    "picture.transition",
    "picture.grade",
    "captions.timing",
    "captions.cards",
    "captions.contrast",
    "captions.words",
    "faces.clearance",
    "detect.cuts",
    "platform.shape",
    "platform.duration",
    "platform.resolution",
    "platform.fps",
    "platform.chapters",
    "taste.rules",
)

PILOT_CHECK_IDS: tuple[str, ...] = (
    "captions3d.behind_words",
    "captions3d.contrast",
)


@dataclass(frozen=True)
class TransitionWindow:
    """A declared transition on the output frame grid, with an exclusive end."""

    join_frame: int
    start_frame: int
    end_frame: int
    id: str

    def __post_init__(self) -> None:
        frames = (self.join_frame, self.start_frame, self.end_frame)
        if any(type(frame) is not int for frame in frames):
            raise ValidationError("transitions", "window frames must be integers")
        if not 0 <= self.start_frame <= self.join_frame < self.end_frame:
            raise ValidationError("transitions", "window must contain its join frame")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValidationError("transitions", "window needs a transition id")


@dataclass(frozen=True)
class Finding:
    """One actionable result on the output timeline."""

    check: str
    severity: Severity
    message: str
    cut: int | None = None
    segment: int | None = None
    time_s: float | None = None
    suggested_action: str | None = None
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.check not in CHECK_IDS and self.check not in PILOT_CHECK_IDS:
            raise ValueError(f"unknown check id: {self.check}")
        if self.severity not in {"fail", "warn", "info"}:
            raise ValueError(f"unknown severity: {self.severity}")
        # Reject non-finite numbers and objects before they reach a report writer.
        json.dumps(self.evidence, allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class Thresholds:
    """Default measurement limits, with sound late expressed as positive."""

    av_early_fail_ms: float = 40.0
    av_late_fail_ms: float = 60.0
    av_warn_frames: float = 0.5
    drift_warn_ms_per_cut: float = 2.0
    # Midpoint between <= 0 dB faded splices and >= 6 dB planted steps.
    click_fail_db: float = 3.0
    # A join whose sharpest sample step near the cut is below this level is
    # inaudible, so it passes whatever its relative sharpness.
    click_floor_dbfs: float = -60.0
    gap_warn_ms: float = 5.0
    loudness_target_lufs: float = -14.0
    loudness_tolerance_lu: float = 1.0
    true_peak_max_dbtp: float = -1.0
    parts_loudness_warn_lu: float = 3.0
    parts_min_s: float = 1.0
    black_warn_s: float = 0.5
    freeze_warn_s: float = 1.0
    touch_cut_frames: int = 2
    frames_vs_plan_warn: int = 1
    # First-word starts add at most one frame; three frames at 30 fps is generous, but 0.3 s fails.
    caption_offset_fail_ms: float = 100.0
    # 3:1 is WCAG AA for large text and the renderer's contrast level.
    # Below 1.5:1 letters disappear into the picture; real gold words measured 1.0 to 1.4.
    caption_contrast_warn: float = 3.0
    caption_contrast_fail: float = 1.5

    def with_target(self, lufs: float) -> Thresholds:
        """Return these thresholds with a different integrated loudness target."""
        return replace(self, loudness_target_lufs=lufs)


def format_time(t: float) -> str:
    """Format an output time as m:ss.s, flooring to a tenth of a second."""
    tenths = math.floor(t * 10)
    minutes, remainder = divmod(tenths, 600)
    seconds, fraction = divmod(remainder, 10)
    return f"{minutes}:{seconds:02d}.{fraction}"


def status_of(findings: Iterable[Finding]) -> Literal["pass", "warn", "fail"]:
    """Return the most severe actionable result; information alone passes."""
    severities = {finding.severity for finding in findings}
    return "fail" if "fail" in severities else "warn" if "warn" in severities else "pass"
