"""Score the timing of labeled defect spans against planted faults."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yap_studio.errors import ValidationError

DEFAULT_TOLERANCES_FRAMES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 30)


@dataclass(frozen=True)
class Span:
    """One labeled interval or point event on a video's timeline."""

    video: str
    label: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValidationError("span", "start_s and end_s must be finite")
        if not 0 <= self.start_s <= self.end_s:
            raise ValidationError("span", "require 0 <= start_s <= end_s")


def _metrics(
    true_positives: int, truth_count: int, predicted_count: int
) -> dict[str, int | float]:
    """Compute counts and scores, treating a wholly empty group as correct."""
    precision = (
        true_positives / predicted_count if predicted_count else float(truth_count == 0)
    )
    recall = true_positives / truth_count if truth_count else float(predicted_count == 0)
    return {
        "true_positives": true_positives,
        "false_positives": predicted_count - true_positives,
        "false_negatives": truth_count - true_positives,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _match_count(candidates: Sequence[tuple[float, int, int]], tolerance_s: float) -> int:
    """Greedily assign sorted candidates within the boundary tolerance."""
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for error, truth_index, prediction_index in candidates:
        if error > tolerance_s + 1e-9:
            break
        if truth_index not in matched_truth and prediction_index not in matched_predictions:
            matched_truth.add(truth_index)
            matched_predictions.add(prediction_index)
    return len(matched_truth)


def score_spans(
    truth: Sequence[Span],
    predicted: Sequence[Span],
    *,
    fps: float = 30.0,
    tolerances_frames: Sequence[int] = DEFAULT_TOLERANCES_FRAMES,
    videos: Collection[str] | None = None,
) -> dict[str, Any]:
    """Return overall and per-label scores across frame tolerances.

    Candidates are ordered by maximum boundary error, then original truth and
    prediction indices. Each tolerance makes its own greedy assignment within
    each video and label. Explicit videos include controls without truth spans;
    otherwise the truth spans alone define the allowed video set.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValidationError("fps", "must be positive and finite")
    fps = float(fps)
    frames_values = tuple(tolerances_frames)
    if not frames_values or any(
        not isinstance(frames, int) or isinstance(frames, bool) or frames < 0
        for frames in frames_values
    ):
        raise ValidationError("tolerances_frames", "must contain non-negative integers")
    seconds_values = []
    for frames in frames_values:
        try:
            seconds = frames / fps
        except OverflowError as exc:
            raise ValidationError("tolerances_frames", "seconds must be finite") from exc
        if not math.isfinite(seconds):
            raise ValidationError("tolerances_frames", "seconds must be finite")
        seconds_values.append(seconds)

    video_ids = set(videos) if videos is not None else {span.video for span in truth}
    unknown = ({span.video for span in truth} | {span.video for span in predicted}) - video_ids
    if unknown:
        error = ValidationError("videos", "unknown video ids: " + ", ".join(sorted(unknown)))
        error.code = "span_video_mismatch"
        raise error

    truth_groups: dict[tuple[str, str], list[tuple[int, Span]]] = defaultdict(list)
    predicted_groups: dict[tuple[str, str], list[tuple[int, Span]]] = defaultdict(list)
    for index, span in enumerate(truth):
        truth_groups[span.video, span.label].append((index, span))
    for index, span in enumerate(predicted):
        predicted_groups[span.video, span.label].append((index, span))
    candidates = {
        group: sorted(
            (
                max(abs(actual.start_s - guess.start_s), abs(actual.end_s - guess.end_s)),
                truth_index,
                prediction_index,
            )
            for truth_index, actual in truth_groups.get(group, ())
            for prediction_index, guess in predicted_groups.get(group, ())
        )
        for group in sorted(truth_groups.keys() | predicted_groups.keys())
    }
    truth_counts = Counter(span.label for span in truth)
    predicted_counts = Counter(span.label for span in predicted)
    labels = sorted(truth_counts.keys() | predicted_counts.keys())
    tolerances = []
    for frames, seconds in zip(frames_values, seconds_values, strict=True):
        matched: Counter[str] = Counter()
        for (_, label), pairs in candidates.items():
            matched[label] += _match_count(pairs, seconds)
        tolerances.append({
            "frames": frames,
            "seconds": seconds,
            **_metrics(sum(matched.values()), len(truth), len(predicted)),
            "labels": {
                label: _metrics(matched[label], truth_counts[label], predicted_counts[label])
                for label in labels
            },
        })
    return {
        "schema": "yap-studio/span-score@1",
        "fps": fps,
        "videos": sorted(video_ids),
        "tolerances": tolerances,
        "mean_f1": sum(row["f1"] for row in tolerances) / len(tolerances),
    }


def spans_from_report(
    report: Mapping[str, Any],
    *,
    video: str,
    severities: Collection[str] = ("fail",),
) -> list[Span]:
    """Convert findings with selected severities and finite times into points."""
    spans = []
    for finding in report.get("findings", ()):
        time_s = finding.get("time_s")
        if (
            finding["severity"] in severities
            and time_s is not None
            and math.isfinite(time_s)
        ):
            spans.append(Span(video, finding["check"], time_s, time_s))
    return spans
