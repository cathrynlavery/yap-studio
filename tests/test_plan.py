"""Exact frame and sample plan tests."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from yap_studio.edl import EditList, Hold, Segment, Source
from yap_studio.plan import Plan, snap_frame


def _rounded(value: Fraction) -> int:
    value += Fraction(1, 2)
    return value.numerator // value.denominator


def _edit_list(segments: list[Segment]) -> EditList:
    return EditList(
        "yap",
        Path("edit.json"),
        {"main": Source("main", Path("/media/project/take.mov"))},
        segments,
        None,
        [],
        [],
    )


def test_only_source_and_output_continuity_skip_a_cut() -> None:
    plan = Plan.build(_edit_list([
        Segment(1, "main", Fraction(0), Fraction(1)),
        Segment(2, "main", Fraction(1), Fraction(2)),
        Segment(3, "main", Fraction(3), Fraction(4)),
        Segment(4, "other", Fraction(4), Fraction(5)),
        Segment(5, "other", Fraction(5), Fraction(6), output_start=Fraction(5)),
    ]), Fraction(30), 48_000)
    assert [(cut.index, cut.left.index, cut.right.index) for cut in plan.cuts] == [
        (1, 2, 3), (2, 3, 4), (3, 4, 5),
    ]
    assert plan.total_frames == 180 and plan.total_samples == 288_000


def test_half_frame_ties_snap_up() -> None:
    fps = Fraction(30000, 1001)
    for frame in (0, 1, 17, 100):
        time = Fraction(2 * frame + 1, 2) / fps
        assert snap_frame(time, fps) == frame + 1


def test_plan_carries_planned_boundary_words() -> None:
    """Frame snapping preserves the editor's word intent."""
    plan = Plan.build(_edit_list([
        Segment(1, "main", Fraction(0), Fraction(1), first_word="Bright", last_word="hills"),
        Segment(2, "main", Fraction(2), Fraction(3)),
    ]), Fraction(30), 48_000)
    assert (plan.segments[0].first_word, plan.segments[0].last_word) == ("Bright", "hills")
    assert plan.segments[1].first_word is plan.segments[1].last_word is None
    assert plan.to_dict()["segments"][0]["first_word"] == "Bright"
    assert plan.to_dict()["segments"][0]["last_word"] == "hills"


def test_explicit_output_positions_win_and_gaps_are_planned() -> None:
    fps = Fraction(24)
    edit_list = EditList(
        "fcpxml",
        Path("edit.fcpxml"),
        {"main": Source("main", None)},
        [
            Segment(1, "main", Fraction(0), Fraction(1), output_start=Fraction(2)),
            Segment(2, None, Fraction(0), Fraction(1), output_start=Fraction(3)),
            Segment(3, "main", Fraction(1), Fraction(2)),
        ],
        fps,
        [],
        [],
    )
    plan = Plan.build(edit_list, fps, 48_000)

    assert [segment.out_f0 for segment in plan.segments] == [48, 72, 96]
    assert plan.segments[1].source_id is None
    assert plan.cuts[0].out_frame == 72
    assert plan.cuts[0].out_sample == 144_000
    assert plan.total_frames == 120
    assert plan.total_samples == 240_000


def test_short_segment_is_dropped_and_remaining_segments_renumbered() -> None:
    fps = Fraction(24)
    edit_list = _edit_list(
        [
            Segment(1, "main", Fraction(0), Fraction(1, 1000)),
            Segment(2, "main", Fraction(1), Fraction(2), label="kept"),
        ]
    )
    plan = Plan.build(edit_list, fps, 48_000)

    assert len(plan.segments) == 1
    assert plan.segments[0].index == 1
    assert plan.segments[0].edl_index == 2
    assert "Segment 1" in plan.warnings[0]


def test_explicit_overlap_adds_warning() -> None:
    plan = Plan.build(
        _edit_list(
            [
                Segment(1, "main", Fraction(0), Fraction(2)),
                Segment(2, "main", Fraction(2), Fraction(3), output_start=Fraction(1)),
            ]
        ),
        Fraction(24),
        48_000,
    )
    assert any("overlaps" in warning for warning in plan.warnings)


def _held(segments: list[Segment], holds: list[Hold]) -> EditList:
    return replace(_edit_list(segments), holds=holds)


def test_end_hold_counts_toward_planned_frames_and_samples() -> None:
    plan = Plan.build(
        _held([Segment(1, "main", Fraction(0), Fraction(1))], [Hold(1, 24, 36)]),
        Fraction(24),
        48_000,
    )
    assert plan.total_frames == 36
    assert plan.total_samples == 72_000
    assert plan.cuts == []
    hold = plan.holds[0]
    assert (hold.out_f0, hold.out_f1, hold.n_frames) == (24, 36, 12)
    assert (hold.sample_start, hold.sample_end) == (48_000, 72_000)
    assert hold.out_time == 1 and hold.end_time == Fraction(3, 2)
    assert plan.to_dict()["holds"][0]["sample_end"] == 72_000
    assert not plan.warnings


def test_hold_between_segments_pushes_the_next_segment() -> None:
    plan = Plan.build(
        _held(
            [
                Segment(1, "main", Fraction(0), Fraction(1)),
                Segment(2, "main", Fraction(2), Fraction(3)),
            ],
            [Hold(1, 24, 36)],
        ),
        Fraction(24),
        48_000,
    )
    assert [segment.out_f0 for segment in plan.segments] == [0, 36]
    assert plan.cuts[0].out_frame == 36
    assert plan.total_frames == 60
    assert plan.total_samples == 120_000


def test_hold_inside_a_segment_is_warned() -> None:
    plan = Plan.build(
        _held([Segment(1, "main", Fraction(0), Fraction(2))], [Hold(1, 24, 36)]),
        Fraction(24),
        48_000,
    )
    assert plan.segments[0].out_f0 == 0
    assert plan.total_frames == 48
    assert any("Hold 1 overlaps segment 1" in warning for warning in plan.warnings)


def test_2997_plan_stays_exact_over_100_segments() -> None:
    fps = Fraction(30000, 1001)
    sample_rate = 48_000
    frame_duration = 1 / fps
    segment_duration = frame_duration * 45
    segments = [
        Segment(
            index=index + 1,
            source_id="main",
            start=segment_duration * index,
            end=segment_duration * (index + 1),
        )
        for index in range(100)
    ]
    plan = Plan.build(_edit_list(segments), fps, sample_rate)

    expected_frame = 0
    for segment in plan.segments:
        expected_sample = _rounded(Fraction(expected_frame * sample_rate, 1) / fps)
        assert segment.out_f0 == expected_frame
        assert segment.sample_start == expected_sample
        expected_frame += 45
    assert plan.total_frames == 4_500
    assert plan.total_samples == _rounded(Fraction(4_500 * sample_rate, 1) / fps)
    assert plan.to_dict()["fps"] == "30000/1001"
