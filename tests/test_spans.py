"""Hand-made defect spans exercise timing, assignment, and report conversion."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest

from yap_studio.checks.model import Finding
from yap_studio.checks.spans import DEFAULT_TOLERANCES_FRAMES, Span, score_spans, spans_from_report
from yap_studio.errors import ValidationError


def test_exact_match() -> None:
    span = Span("fault", "join.frames", 7.2, 7.4)
    score = score_spans([span], [span])

    assert score["schema"] == "yap-studio/span-score@1"
    assert score["fps"] == 30.0
    assert score["videos"] == ["fault"]
    assert DEFAULT_TOLERANCES_FRAMES == (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 30)
    assert [row["frames"] for row in score["tolerances"]] == list(DEFAULT_TOLERANCES_FRAMES)
    expected = {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    for row in score["tolerances"]:
        assert row == {
            "frames": row["frames"],
            "seconds": row["frames"] / 30.0,
            **expected,
            "labels": {"join.frames": expected},
        }
    assert score["mean_f1"] == 1.0


def test_one_frame_offset_and_mean_f1() -> None:
    truth = [Span("fault", "join.click", 7.2, 7.2)]
    predicted = [Span("fault", "join.click", 7.2 + 1 / 30, 7.2 + 1 / 30)]
    score = score_spans(truth, predicted, tolerances_frames=(0, 1))
    exact, one_frame = score["tolerances"]

    assert (exact["true_positives"], exact["false_positives"], exact["false_negatives"]) == (
        0, 1, 1,
    )
    assert exact["precision"] == exact["recall"] == exact["f1"] == 0.0
    assert one_frame["true_positives"] == 1
    assert one_frame["precision"] == one_frame["recall"] == one_frame["f1"] == 1.0
    assert score["mean_f1"] == 0.5


def test_both_boundaries_must_be_within_tolerance() -> None:
    truth = [Span("fault", "join.frames", 1.0, 2.0)]
    predicted = [Span("fault", "join.frames", 1.0 + 1 / 30, 2.0 + 2 / 30)]
    row = score_spans(truth, predicted, tolerances_frames=(1,))["tolerances"][0]
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (0, 1, 1)


@pytest.mark.parametrize("extra_s,expected", [(0.5e-9, 1), (2e-9, 0)])
def test_boundary_comparison_has_one_nanosecond_slack(extra_s: float, expected: int) -> None:
    truth = [Span("fault", "join.click", 1.0, 1.0)]
    time_s = 1.0 + 1 / 30 + extra_s
    predicted = [Span("fault", "join.click", time_s, time_s)]
    row = score_spans(truth, predicted, tolerances_frames=(1,))["tolerances"][0]
    assert row["true_positives"] == expected


def test_label_mismatch_never_matches() -> None:
    truth = [Span("fault", "join.frames", 1.0, 1.0)]
    predicted = [Span("fault", "picture.flash", 1.0, 1.0)]
    row = score_spans(truth, predicted, tolerances_frames=(0, 30))["tolerances"][-1]
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (0, 1, 1)
    assert row["labels"] == {
        "join.frames": {
            "true_positives": 0, "false_positives": 0, "false_negatives": 1,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        },
        "picture.flash": {
            "true_positives": 0, "false_positives": 1, "false_negatives": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        },
    }


def test_different_videos_never_match() -> None:
    truth = [Span("fault", "join.click", 1.0, 1.0)]
    predicted = [Span("clean", "join.click", 1.0, 1.0)]
    row = score_spans(
        truth, predicted, videos={"fault", "clean"}, tolerances_frames=(30,)
    )["tolerances"][0]
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (0, 1, 1)


@pytest.mark.parametrize("reverse", [False, True], ids=["extra_prediction", "extra_truth"])
def test_matching_is_one_to_one(reverse: bool) -> None:
    truth = [Span("fault", "join.click", 1.0, 1.0)]
    predicted = [
        Span("fault", "join.click", 1.0, 1.0),
        Span("fault", "join.click", 1.0 + 1 / 30, 1.0 + 1 / 30),
    ]
    if reverse:
        truth, predicted = predicted, truth
    row = score_spans(truth, predicted, tolerances_frames=(1,))["tolerances"][0]
    assert row["true_positives"] == 1
    assert row["false_positives"] == (0 if reverse else 1)
    assert row["false_negatives"] == (1 if reverse else 0)
    assert row["precision"] == (1.0 if reverse else 0.5)
    assert row["recall"] == (0.5 if reverse else 1.0)
    assert row["f1"] == pytest.approx(2 / 3)


def test_deterministic_tie_order() -> None:
    truth = [Span("fault", "join.frames", time, time) for time in (1.0, 3.0)]
    predicted = [Span("fault", "join.frames", time, time) for time in (2.0, 0.0)]
    # All candidate errors are equal. Assigning indices (0, 0) first leaves one match.
    score = score_spans(truth, predicted, fps=1.0, tolerances_frames=(1,))
    assert score["tolerances"][0]["true_positives"] == 1
    assert score_spans(truth, predicted, fps=1.0, tolerances_frames=(1,)) == score
    for actual, guesses in ((truth[::-1], predicted), (truth, predicted[::-1])):
        row = score_spans(actual, guesses, fps=1.0, tolerances_frames=(1,))["tolerances"][0]
        assert row["true_positives"] == 2


def test_nearest_candidate_precedes_index_order() -> None:
    truth = [Span("fault", "join.frames", time, time) for time in (1.0, 4.0)]
    predicted = [Span("fault", "join.frames", time, time) for time in (3.0, 6.0)]
    row = score_spans(truth, predicted, fps=1.0, tolerances_frames=(3,))["tolerances"][0]
    # The nearest pair consumes the prediction that the first truth could use.
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (1, 1, 1)


@pytest.mark.parametrize(
    "truth_videos,predicted_videos,videos,unknown",
    [
        (["known"], ["zeta", "alpha"], None, ("alpha", "zeta")),
        (["unknown"], [], {"known"}, ("unknown",)),
        (["known"], ["unknown"], {"known", "clean"}, ("unknown",)),
        ([], ["unknown"], None, ("unknown",)),
    ],
)
def test_mismatched_video_sets_are_refused(
    truth_videos: list[str],
    predicted_videos: list[str],
    videos: set[str] | None,
    unknown: tuple[str, ...],
) -> None:
    truth = [Span(video, "join.click", 1.0, 1.0) for video in truth_videos]
    predicted = [Span(video, "join.click", 1.0, 1.0) for video in predicted_videos]
    with pytest.raises(ValidationError) as caught:
        score_spans(truth, predicted, videos=videos)
    assert caught.value.code == "span_video_mismatch"
    assert ", ".join(unknown) in caught.value.message


def test_explicit_clean_video_with_no_spans() -> None:
    score = score_spans([], [], videos={"clean"})
    assert score["videos"] == ["clean"]
    assert score["mean_f1"] == 1.0
    for row in score["tolerances"]:
        assert row["true_positives"] == row["false_positives"] == row["false_negatives"] == 0
        assert row["precision"] == row["recall"] == row["f1"] == 1.0
        assert row["labels"] == {}


def test_false_prediction_on_explicit_clean_video() -> None:
    predicted = [Span("clean", "join.click", 0.0, 0.0)]
    score = score_spans([], predicted, videos={"clean"})
    for row in score["tolerances"]:
        assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (
            0, 1, 0,
        )
        assert row["precision"] == row["recall"] == row["f1"] == 0.0
    assert score["mean_f1"] == 0.0


def test_point_spans_are_valid_and_frozen() -> None:
    span = Span("fault", "join.click", 0.0, 0.0)
    assert score_spans([span], [span])["mean_f1"] == 1.0
    with pytest.raises(FrozenInstanceError):
        span.start_s = 1.0


@pytest.mark.parametrize(
    "start,end",
    [(-0.1, 1.0), (math.nan, 1.0), (0.0, math.nan), (math.inf, math.inf),
     (0.0, math.inf), (0.0, -math.inf), (2.0, 1.0)],
)
def test_invalid_spans_are_refused(start: float, end: float) -> None:
    with pytest.raises(ValidationError):
        Span("fault", "join.click", start, end)


@pytest.mark.parametrize("fps", [0.0, -30.0, math.nan, math.inf])
def test_invalid_fps_is_refused(fps: float) -> None:
    with pytest.raises(ValidationError):
        score_spans([], [], fps=fps)


@pytest.mark.parametrize("frames", [(), (-1,), (1.5,), (True,), (10**400,)])
def test_invalid_tolerances_are_refused(frames: tuple) -> None:
    with pytest.raises(ValidationError):
        score_spans([], [], tolerances_frames=frames)


def test_report_conversion_filters_severity_and_time() -> None:
    report = {"findings": [
        Finding("join.click", "fail", "click", time_s=7.2).to_dict(),
        Finding("join.frames", "fail", "frame", time_s=0.0).to_dict(),
        Finding("sync.av_offset", "warn", "offset", time_s=7.2).to_dict(),
        Finding("join.listen", "info", "listen", time_s=7.2).to_dict(),
        Finding("plan.frames", "fail", "length").to_dict(),
        {"check": "join.words", "severity": "fail"},
        {"check": "join.words", "severity": "fail", "time_s": math.nan},
        {"check": "join.words", "severity": "fail", "time_s": math.inf},
        {"check": "join.words", "severity": "fail", "time_s": -math.inf},
    ]}
    assert spans_from_report(report, video="fault") == [
        Span("fault", "join.click", 7.2, 7.2),
        Span("fault", "join.frames", 0.0, 0.0),
    ]
    assert spans_from_report(report, video="fault", severities={"warn", "info"}) == [
        Span("fault", "sync.av_offset", 7.2, 7.2),
        Span("fault", "join.listen", 7.2, 7.2),
    ]
    assert spans_from_report({}, video="clean") == []


def test_label_scores_and_json_safety() -> None:
    truth = [
        Span("fault", "join.click", 7.2, 7.2),
        Span("fault", "join.words", 7.2, 7.2),
    ]
    predicted = [
        Span("fault", "join.click", 7.2, 7.2),
        Span("clean", "picture.flash", 1.0, 1.0),
    ]
    score = score_spans(truth, predicted, videos=["fault", "clean", "fault"])
    assert score["videos"] == ["clean", "fault"]
    assert json.loads(json.dumps(score, allow_nan=False)) == score
    row = score["tolerances"][0]
    assert row["precision"] == row["recall"] == row["f1"] == 0.5
    assert list(row["labels"]) == ["join.click", "join.words", "picture.flash"]
    assert row["labels"]["join.click"]["f1"] == 1.0
    assert row["labels"]["join.words"]["false_negatives"] == 1
    assert row["labels"]["picture.flash"]["false_positives"] == 1
    assert score["mean_f1"] == 0.5
