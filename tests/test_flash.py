"""Flash frames at joins, using only synthetic pictures and sound."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg, video
from yap_studio.checks import detect, joins, sync
from yap_studio.checks.flash import flash_at
from yap_studio.checks.model import Thresholds
from yap_studio.edl import PictureRegion, load_edit_list
from yap_studio.errors import ValidationError
from yap_studio.plan import Plan
from yap_studio.verify import _edit_coverage, verify

_JOIN = 90
_FPS = 30.0


@pytest.fixture
def pictures():
    rng = np.random.default_rng(35)
    first, second, foreign = rng.normal(120, 35, (3, 32, 48)).astype(np.float32)
    drift = np.linspace(-1, 1, 48)[None, :]
    frames = np.asarray(
        [
            (first if index < _JOIN else second)
            + rng.normal(0, 0.6, first.shape)
            + index * 0.025 * drift
            for index in range(180)
        ],
        dtype=np.float32,
    )
    return frames, foreign


def test_clean_cut_and_single_scene_change_have_no_flash(pictures):
    frames, _ = pictures
    evidence = flash_at(frames, _JOIN, fps=_FPS)
    assert not evidence["flash"] and evidence["reason"] is None
    assert [step["frame"] for step in evidence["steps"] if step["change"]] == [_JOIN]
    assert evidence["run"] is None and evidence["changes"] == []
    assert not flash_at(frames, _JOIN - 30, fps=_FPS)["flash"]
    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence


@pytest.mark.parametrize(
    "start,length", [(_JOIN - 1, 1), (_JOIN, 1), (_JOIN - 2, 2), (_JOIN - 1, 2), (_JOIN, 2)]
)
def test_one_or_two_foreign_frames_at_the_join_fail(pictures, start, length):
    frames, foreign = pictures
    frames[start : start + length] = foreign
    evidence = flash_at(frames, _JOIN, fps=_FPS)
    assert evidence["flash"] and evidence["reason"] is None
    assert evidence["run"] == {
        "first_frame": start,
        "last_frame": start + length - 1,
        "length": length,
    }
    assert [change["frame"] for change in evidence["changes"]] == [start, start + length]
    assert all(change["score"] >= 5 for change in evidence["changes"])
    assert all(
        change["distance"] >= 3 * evidence["local_motion"]["median"]
        for change in evidence["changes"]
    )
    assert all(
        min(side["before_distance"], side["after_distance"])
        >= 0.85 * evidence["neighbour_distance"]
        for side in evidence["run_distances"]
    )
    assert evidence["rule"]["neighbour_ratio"] == 0.85
    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence


@pytest.mark.parametrize("window,expected", [((89, 90), False), ((87, 89), True), ((90, 92), True)])
def test_flash_ignores_only_runs_inside_transition_windows(pictures, window, expected):
    frames, foreign = pictures
    frames[_JOIN - 1] = foreign
    assert flash_at(frames, _JOIN, fps=_FPS)["flash"]
    evidence = flash_at(frames, _JOIN, fps=_FPS, ignored_windows=[window])
    assert evidence["flash"] is expected
    assert evidence["reason"] is None


@pytest.mark.parametrize("offset", [-3, 3])
def test_foreign_frame_three_frames_away_is_not_a_join_flash(pictures, offset):
    frames, foreign = pictures
    frames[_JOIN + offset] = foreign
    evidence = flash_at(frames, _JOIN, fps=_FPS)
    assert not evidence["flash"] and evidence["reason"] is None


def test_repeated_frame_is_not_a_flash(pictures):
    frames, _ = pictures
    frames[_JOIN - 1] = frames[_JOIN - 2]
    evidence = flash_at(frames, _JOIN, fps=_FPS)
    assert not evidence["flash"] and evidence["reason"] is None


@pytest.fixture
def moving_texture():
    rng = np.random.default_rng(38)
    noise = rng.normal(size=(64, 128))
    rows = np.fft.fftfreq(64)[:, None]
    columns = np.fft.fftfreq(128)[None, :]
    smoothing = np.exp(-2 * np.pi**2 * 6**2 * (rows**2 + columns**2))
    texture = np.fft.ifft2(np.fft.fft2(noise) * smoothing).real
    return (120 + 35 * texture / texture.std()).astype(np.float32)


def _motion_burst(texture, repeated=False):
    shifts = np.full(180, 2)
    shifts[_JOIN] = 8
    shifts[_JOIN + 1] = 0 if repeated else 8
    if repeated:
        shifts[_JOIN + 2] = 8
    return np.asarray([np.roll(texture, int(offset), axis=1) for offset in np.cumsum(shifts)])


def _assert_motion_is_not_a_flash(frames, length):
    evidence = flash_at(frames, _JOIN, fps=_FPS)
    assert not evidence["flash"] and evidence["reason"] is None
    assert [step["frame"] for step in evidence["steps"] if step["change"]] == [
        _JOIN,
        _JOIN + length,
    ]
    before, after = frames[_JOIN - 1], frames[_JOIN + length]
    neighbour_distance = video.frame_distance(before, after)
    for index in range(_JOIN, _JOIN + length):
        ratio = min(
            video.frame_distance(before, frames[index]),
            video.frame_distance(frames[index], after),
        ) / neighbour_distance
        assert 0.5 < ratio < 0.85
    return evidence


def test_steady_fast_motion_is_not_a_flash(moving_texture):
    frames = _motion_burst(moving_texture)
    _assert_motion_is_not_a_flash(frames, 1)


def test_repeated_first_incoming_frame_then_fast_motion_is_not_a_flash(moving_texture):
    frames = _motion_burst(moving_texture, repeated=True)
    assert np.array_equal(frames[_JOIN], frames[_JOIN + 1])
    evidence = _assert_motion_is_not_a_flash(frames, 2)
    repeat = next(step for step in evidence["steps"] if step["frame"] == _JOIN + 1)
    assert repeat["distance"] == 0 and not repeat["change"]


def test_missing_frame_is_not_a_flash(pictures):
    frames, _ = pictures
    frames = np.delete(frames, _JOIN - 2, axis=0)
    evidence = flash_at(frames, _JOIN - 1, fps=_FPS)
    assert not evidence["flash"] and evidence["reason"] is None


def test_halfway_blend_is_not_a_flash(pictures):
    frames, _ = pictures
    frames[_JOIN] = (frames[_JOIN - 1] + frames[_JOIN + 1]) / 2
    evidence = flash_at(frames, _JOIN, fps=_FPS)
    assert not evidence["flash"] and evidence["reason"] is None


def test_small_codec_catch_up_step_is_not_a_flash(pictures):
    frames, foreign = pictures
    frames[_JOIN] = 0.98 * frames[_JOIN + 1] + 0.02 * foreign
    evidence = flash_at(frames, _JOIN, fps=_FPS)
    assert not evidence["flash"] and evidence["reason"] is None
    assert not next(step for step in evidence["steps"] if step["frame"] == _JOIN + 1)["change"]


@pytest.mark.parametrize("count,frame", [(0, 0), (1, 0), (2, 1), (3, 1), (180, 0), (180, 179)])
def test_edges_and_short_arrays_do_not_raise(pictures, count, frame):
    frames, _ = pictures
    evidence = flash_at(frames[:count], frame, fps=_FPS)
    assert not evidence["flash"]
    if count <= 3:
        assert evidence["reason"]
    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence


def test_reference_excludes_join_and_flash_steps(pictures):
    frames, foreign = pictures
    clean = flash_at(frames, _JOIN, fps=_FPS)
    frames[_JOIN - 1 : _JOIN + 1] = foreign
    flashed = flash_at(frames, _JOIN, fps=_FPS)
    assert flashed["flash"]
    assert flashed["local_motion"] == clean["local_motion"]
    assert flashed["local_motion"]["steps"] == 54


def test_overlay_pixels_are_excluded(pictures):
    frames, foreign = pictures
    frames[_JOIN - 1, :30] = foreign[:30]
    assert flash_at(frames, _JOIN, fps=_FPS)["flash"]
    valid = np.ones(frames.shape[1:], dtype=bool)
    valid[:30] = False
    evidence = flash_at(frames, _JOIN, fps=_FPS, valid=valid)
    assert not evidence["flash"] and evidence["reason"] is None
    masked = flash_at(frames, _JOIN, fps=_FPS, valid=np.zeros_like(valid))
    assert not masked["flash"] and masked["reason"] == "picture masked"


@pytest.mark.parametrize(
    "argument,value",
    [
        ("fps", 0),
        ("fps", float("nan")),
        ("touch_frames", -1),
        ("touch_frames", 1.5),
        ("frame", 1.5),
        ("valid", np.ones((2, 2))),
    ],
)
def test_bad_arguments_raise_validation_error(pictures, argument, value):
    frames, _ = pictures
    arguments = {"frame": _JOIN, "fps": _FPS, argument: value}
    with pytest.raises(ValidationError):
        flash_at(frames, **arguments)


def test_invalid_pictures_raise_validation_error(pictures):
    frames, _ = pictures
    with pytest.raises(ValidationError):
        flash_at(frames[:, 0], _JOIN, fps=_FPS)
    frames[_JOIN, 0, 0] = np.nan
    with pytest.raises(ValidationError):
        flash_at(frames, _JOIN, fps=_FPS)


def _edit_list(path: Path, source: Path, spans, **extra) -> Path:
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "segments": [{"start": start, "end": end} for start, end in spans],
                **extra,
            }
        )
    )
    return path


@pytest.fixture
def source_comparison(pictures):
    frames, _ = pictures
    source_frames = np.random.default_rng(36).normal(120, 35, (40, 32, 48)).astype(np.float32)
    source = joins._Media(SimpleNamespace(fps=Fraction(24)), None, source_frames, Fraction(60))
    for index, wanted in ((_JOIN - 2, 8), (_JOIN - 1, 10), (_JOIN, 28), (_JOIN + 1, 30)):
        frames[index] = source_frames[wanted]
    return SimpleNamespace(
        rendered=joins._Media(
            SimpleNamespace(
                fps=Fraction(30),
                video_offset_s=0.25,
                width=48,
                height=32,
            ),
            None,
            frames,
        ),
        left=source,
        right=source,
        left_sync=sync.SegmentSync(1, "left"),
        right_sync=sync.SegmentSync(2, "right"),
        cut=SimpleNamespace(left=SimpleNamespace(src_f1=6), right=SimpleNamespace(src_f0=14)),
    )


def _source_evidence(case, start, length, regions=()):
    frames = case.rendered.frames
    flash = {
        "flash": True,
        "reason": None,
        "run": {"first_frame": start, "last_frame": start + length - 1, "length": length},
        "window": {"first_frame": _JOIN - 30, "last_frame": _JOIN + 30},
        "changes": [
            {
                "frame": index,
                "distance": video.frame_distance(frames[index - 1], frames[index]),
            }
            for index in (start, start + length)
        ],
    }
    return joins._flash_in_source(
        flash,
        case.cut,
        _JOIN,
        case.rendered,
        case.left,
        case.right,
        case.left_sync,
        case.right_sync,
        _FPS,
        regions,
    )


@pytest.mark.parametrize(
    "start,length,expected",
    [
        (_JOIN - 1, 1, [("left", 10)]),
        (_JOIN, 1, [("right", 28)]),
        (_JOIN - 2, 2, [("left", 8), ("left", 10)]),
        (_JOIN - 1, 2, [("left", 10), ("right", 28)]),
        (_JOIN, 2, [("right", 28), ("right", 30)]),
    ],
)
def test_source_frames_follow_each_side_and_the_decoded_rate(
    source_comparison, start, length, expected
):
    evidence = _source_evidence(source_comparison, start, length)
    assert evidence["in_source"] and evidence["source_checked"]
    matches = evidence["source_matches"]
    assert [(match["side"], match["planned_frame"]) for match in matches] == expected
    assert all(match["best_frame"] == match["planned_frame"] for match in matches)
    assert all(match["matched"] is True and match["reason"] is None for match in matches)
    assert json.loads(json.dumps(evidence, allow_nan=False))["in_source"] is True


def test_one_wrong_frame_in_a_run_keeps_the_failure(source_comparison):
    case = source_comparison
    case.rendered.frames[_JOIN] = case.right.frames[29]
    evidence = _source_evidence(case, _JOIN - 1, 2)
    assert evidence["flash"] and not evidence["in_source"] and evidence["source_checked"]
    assert [match["matched"] for match in evidence["source_matches"]] == [True, False]
    assert evidence["source_matches"][1]["best_frame"] == 29


def test_source_candidates_as_far_as_flash_jumps_do_not_exempt_the_flash(source_comparison):
    case = source_comparison
    planned = case.right.frames[30].copy()
    foreign = np.random.default_rng(39).normal(120, 35, planned.shape).astype(np.float32)
    picture = 0.75 * planned + 0.25 * foreign
    for index in range(27, 34):
        case.right.frames[index] = planned + (index - 30) * 0.01 * (picture - planned)
    case.rendered.frames[_JOIN] = case.right.frames[28]
    case.rendered.frames[_JOIN + 1] = picture
    case.rendered.frames[_JOIN + 2] = case.right.frames[32]
    comparison = joins._frame_match(case.rendered, case.right, case.right_sync, 30, _JOIN + 1)
    assert comparison["matched"] and comparison["best_frame"] == 33
    assert 0 < comparison["distances"][30] - comparison["distances"][33] < 0.02

    evidence = _source_evidence(case, _JOIN + 1, 1)
    smaller_change = min(change["distance"] for change in evidence["changes"])
    assert all(
        0.9 * smaller_change < distance < 1.1 * smaller_change
        for distance in comparison["distances"].values()
    )
    assert evidence["flash"] and evidence["source_checked"] and not evidence["in_source"]
    match = evidence["source_matches"][0]
    assert match["matched"] is False and match["planned_frame"] == 30
    assert match["in_source_ratio"] == 0.25
    assert match["in_source_limit"] == pytest.approx(0.25 * smaller_change)
    assert match["distances"][30] > match["in_source_limit"]
    assert json.loads(json.dumps(evidence, allow_nan=False))["in_source"] is False
    print(json.dumps({"unit_source_match": match, "changes": evidence["changes"]}))


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("missing", "source not found"),
        ("no_picture", "no source picture"),
        ("no_mapping", "picture mapping not measured"),
        ("outside", "planned source frame not found"),
    ],
)
def test_source_that_cannot_be_compared_keeps_the_failure(source_comparison, kind, reason):
    case = source_comparison
    if kind == "missing":
        case.right = None
    elif kind == "no_picture":
        case.right = replace(case.right, frames=None)
    elif kind == "no_mapping":
        case.right = replace(case.right, frames=case.right.frames[:, :, ::2])
    else:
        case.cut.right.src_f0 = 100
    evidence = _source_evidence(case, _JOIN - 1, 2)
    assert evidence["flash"] and not evidence["in_source"] and not evidence["source_checked"]
    assert evidence["source_matches"][0]["matched"] is True
    match = evidence["source_matches"][1]
    assert match["reason"] == reason and match["matched"] is None and not match["checked"]


def test_source_flash_uses_the_measured_picture_mapping(source_comparison):
    case = source_comparison
    case.right = replace(case.right, frames=np.pad(case.right.frames, ((0, 0), (2, 2), (4, 4))))
    case.right_sync = replace(case.right_sync, mapping=video.Mapping("crop", 1, 4, 2, 48, 32, 1))
    evidence = _source_evidence(case, _JOIN, 1)
    assert evidence["in_source"] and evidence["source_checked"]
    assert evidence["source_matches"][0]["distances"][28] == 0


def test_source_flash_excludes_overlay_regions_over_the_reference_window(source_comparison):
    case = source_comparison
    case.rendered.frames[_JOIN, :20] = case.right.frames[29, :20]
    assert not _source_evidence(case, _JOIN, 1)["in_source"]
    regions = [PictureRegion(Fraction(9, 4), Fraction(23, 10), 0, 0, 48, 20)]
    evidence = _source_evidence(case, _JOIN, 1, regions)
    assert evidence["in_source"] and evidence["source_checked"]
    assert evidence["source_matches"][0]["distances"][28] == 0


def test_source_flash_accepts_the_existing_tie_tolerance(source_comparison):
    case = source_comparison
    case.right.frames[29] = case.right.frames[28] + np.random.default_rng(37).normal(
        0, 0.2, (32, 48)
    )
    case.rendered.frames[_JOIN] = case.right.frames[29]
    evidence = _source_evidence(case, _JOIN, 1)
    assert evidence["in_source"] and evidence["source_checked"]
    match = evidence["source_matches"][0]
    assert match["planned_frame"] == 28 and match["best_frame"] == 29
    assert 0 < match["distances"][28] - match["distances"][29] < 0.02


@pytest.fixture(scope="module")
def calibration(synth_media, tmp_path_factory):
    directory = tmp_path_factory.mktemp("flash")
    source = synth_media.path("calib")
    clean = synth_media.path("clean")
    edit = _edit_list(directory / "edit.json", source, synth_media.spans)
    join_frame = 144
    flashed = synth_media.cached(
        "flash-foreign", lambda out: synth.replace_frame(clean, out, join_frame, 333)
    )
    scene_source = synth_media.cached(
        "flash-scene-source", lambda out: synth.scene_change_source(source, out, 1.6), suffix=".mov"
    )
    scene_render = synth_media.cached(
        "flash-scene-render", lambda out: synth.render_clean(scene_source, synth_media.spans, out)
    )
    scene_edit = _edit_list(directory / "scene.json", scene_source, synth_media.spans)
    source_spans = [(3.6, 125 / _FPS), (5, 7.4)]
    source_render = synth_media.cached(
        "flash-in-source", lambda out: synth.render_clean(source, source_spans, out)
    )
    source_edit = _edit_list(directory / "in-source.json", source, source_spans)
    reports = {}
    for name, media, edl in (
        ("clean", clean, edit),
        ("foreign", flashed, edit),
        ("scene", scene_render, scene_edit),
        ("in_source", source_render, source_edit),
    ):
        for mode in ("edit", "detect"):
            reports[name, mode] = verify(
                str(media), edl=str(edl) if mode == "edit" else None, asr="none", detail="full"
            )
    measurements = {
        name: flash_at(video.grey_frames(media, w=64), frame, fps=_FPS)
        for name, media, frame in (
            ("clean", clean, join_frame),
            ("foreign", flashed, join_frame),
            ("scene", scene_render, 36),
            ("in_source", source_render, 17),
        )
    }
    diagnostics = directory / "flash-measurements.json"
    diagnostics.write_text(json.dumps(measurements, indent=2, allow_nan=False))
    print(f"Flash measurements: {diagnostics}")
    return SimpleNamespace(
        reports=reports,
        measurements=measurements,
        edit=edit,
        clean=clean,
        flashed=flashed,
        source=source,
        join_frame=join_frame,
        diagnostics=diagnostics,
    )


@pytest.mark.parametrize("mode", ["edit", "detect"])
def test_clean_render_runs_and_passes(calibration, mode):
    report = calibration.reports["clean", mode]
    assert report["checks"]["picture.flash"] == {"ran": True, "status": "pass", "reason": None}
    assert not [row for row in report["findings"] if row["check"] == "picture.flash"]
    if mode == "edit":
        assert all(row["detail"]["flash"]["reason"] is None for row in report["cuts"])
    else:
        assert all(row["flash"] is False for row in report["cuts"])


@pytest.mark.parametrize("mode", ["edit", "detect"])
def test_foreign_frame_fails_at_exactly_one_cut(calibration, mode):
    report = calibration.reports["foreign", mode]
    failures = [row for row in report["findings"] if row["check"] == "picture.flash"]
    assert report["checks"]["picture.flash"]["ran"]
    assert report["checks"]["picture.flash"]["status"] == "fail"
    assert len(failures) == 1 and failures[0]["severity"] == "fail"
    failure = failures[0]
    assert failure["cut"] == 2
    assert abs(failure["time_s"] * _FPS - calibration.join_frame) <= 1
    assert failure["evidence"]["run"] == {
        "first_frame": calibration.join_frame,
        "last_frame": calibration.join_frame,
        "length": 1,
    }
    assert "matches neither side" in failure["message"] and failure["suggested_action"]
    assert (
        ffmpeg.probe(calibration.flashed).video_frames
        == ffmpeg.probe(calibration.clean).video_frames
    )
    if mode == "detect":
        assert report["cuts"][1]["flash"] is True
    else:
        assert failure["evidence"]["source_checked"]
        assert not failure["evidence"]["in_source"]
        assert any(match["matched"] is False for match in failure["evidence"]["source_matches"])


def test_planned_source_flash_passes_with_an_edit_list(calibration):
    report = calibration.reports["in_source", "edit"]
    assert report["checks"]["picture.flash"] == {"ran": True, "status": "pass", "reason": None}
    assert not [row for row in report["findings"] if row["check"] == "picture.flash"]
    flash = report["cuts"][0]["detail"]["flash"]
    assert flash["flash"] and flash["in_source"] and flash["source_checked"]
    assert flash["run"] == {"first_frame": 17, "last_frame": 17, "length": 1}
    match = flash["source_matches"][0]
    assert match["side"] == "right" and match["render_frame"] == 17
    assert match["planned_frame"] == match["best_frame"] == 150
    assert match["matched"] and match["reason"] is None
    assert match["distances"][150] < 0.02
    assert match["in_source_ratio"] == 0.25
    assert match["distances"][150] <= match["in_source_limit"]
    assert match["in_source_limit"] == pytest.approx(
        0.25 * min(change["distance"] for change in flash["changes"])
    )
    assert all(change["score"] >= 5 for change in flash["changes"])


def test_second_incoming_foreign_frame_fails_at_cut_three(calibration, synth_media):
    flashed = synth_media.cached(
        "flash-foreign-second", lambda out: synth.replace_frame(calibration.clean, out, 217, 333)
    )
    report = verify(
        str(flashed), edl=str(calibration.edit), asr="none", loudness="note", detail="full"
    )
    assert report["checks"]["picture.flash"] == {"ran": True, "status": "fail", "reason": None}
    failures = [row for row in report["findings"] if row["check"] == "picture.flash"]
    assert len(failures) == 1 and failures[0]["cut"] == 3
    assert failures[0]["severity"] == "fail"
    flash = failures[0]["evidence"]
    assert flash["run"] == {"first_frame": 217, "last_frame": 217, "length": 1}
    assert flash["source_checked"] and not flash["in_source"]
    match = flash["source_matches"][0]
    assert match["render_frame"] == 217 and match["side"] == "right"
    assert match["planned_frame"] == 316 and match["matched"] is False
    assert match["in_source_ratio"] == 0.25
    assert match["in_source_limit"] == pytest.approx(
        0.25 * min(change["distance"] for change in flash["changes"])
    )
    assert match["distances"][316] > match["in_source_limit"]
    assert ffmpeg.probe(flashed).video_frames == ffmpeg.probe(calibration.clean).video_frames
    diagnostics = json.loads(calibration.diagnostics.read_text())
    diagnostics["second_incoming_foreign"] = flash
    diagnostics["planned_source"] = calibration.reports["in_source", "edit"]["cuts"][0][
        "detail"
    ]["flash"]
    calibration.diagnostics.write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    print(json.dumps({"media_source_match": match, "changes": flash["changes"]}))


def test_source_flash_without_an_edit_list_keeps_the_detector_result(calibration):
    report = calibration.reports["in_source", "detect"]
    # The returning white frame fails the detector's cut persistence gate.
    assert report["cuts"] == []
    assert report["checks"]["picture.flash"] == {
        "ran": False,
        "status": "skipped",
        "reason": "no cuts found in the picture",
    }
    assert not [row for row in report["findings"] if row["check"] == "picture.flash"]
    measured = calibration.measurements["in_source"]
    assert measured["flash"] and "in_source" not in measured


def test_foreign_flash_with_unavailable_sources_still_fails(calibration, tmp_path):
    plan = Plan.build(load_edit_list(calibration.edit), Fraction(30), 48_000)
    result = joins.check_joins(
        calibration.flashed,
        plan,
        {"main": None},
        sync.SyncResult([], None, []),
        backend=None,
        work_dir=tmp_path,
    )
    findings = [row for row in result.findings if row.check == "picture.flash"]
    assert len(findings) == 1 and findings[0].cut == 2
    flash = findings[0].evidence
    assert not flash["in_source"] and not flash["source_checked"]
    assert flash["source_matches"][0]["reason"] == "source not found"
    assert flash["source_matches"][0]["matched"] is None


@pytest.mark.parametrize("mode", ["edit", "detect"])
def test_real_scene_change_in_a_segment_passes(calibration, mode):
    report = calibration.reports["scene", mode]
    assert report["checks"]["picture.flash"]["ran"]
    assert report["checks"]["picture.flash"]["status"] == "pass"
    assert not [row for row in report["findings"] if row["check"] == "picture.flash"]
    if mode == "detect":
        assert any(abs(row["time_s"] - 1.2) <= 1 / _FPS for row in report["cuts"])
    measured = calibration.measurements["scene"]
    assert not measured["flash"] and measured["reason"] is None
    assert [step["frame"] for step in measured["steps"] if step["change"]] == [36]


def test_all_masked_cuts_are_skipped(calibration, tmp_path):
    data = json.loads(calibration.edit.read_text())
    data = {
        "sources": {"main": data["source"]},
        "ranges": [{"source": "main", **segment} for segment in data["segments"]],
        "overlays": [{"start_in_output": 0, "duration": 19.2}],
    }
    edit = tmp_path / "masked.json"
    edit.write_text(json.dumps(data))
    report = verify(str(calibration.flashed), edl=str(edit), asr="none", detail="full")
    assert report["checks"]["picture.flash"] == {
        "ran": False,
        "status": "skipped",
        "reason": "picture masked",
    }
    assert not [row for row in report["findings"] if row["check"] == "picture.flash"]
    assert all(row["detail"]["flash"]["reason"] == "picture masked" for row in report["cuts"])


def test_no_cuts_skip_in_both_modes(synth_media, tmp_path):
    source = synth_media.path("calib")
    edit = _edit_list(tmp_path / "uncut.json", source, synth_media.spans[:1])
    for edl, reason in ((str(edit), "cut coverage: none"), (None, "no cuts found in the picture")):
        report = verify(str(source), edl=edl, asr="none")
        assert report["checks"]["picture.flash"] == {
            "ran": False,
            "status": "skipped",
            "reason": reason,
        }


def test_join_masks_and_regions_use_the_measured_window(pictures):
    frames, foreign = pictures
    frames[_JOIN - 1] = foreign
    rendered = joins._Media(
        SimpleNamespace(
            video_offset_s=0.25,
            width=48,
            height=32,
        ),
        None,
        frames,
    )
    # This mask covers reference motion rather than the boundary frames.
    masks = [(Fraction(9, 4), Fraction(23, 10))]
    evidence = joins._flash(rendered, _JOIN, _FPS, 2, masks, ())
    assert evidence == {"flash": False, "reason": "picture masked"}
    regions = [PictureRegion(Fraction(9, 4), Fraction(23, 10), 0, 0, 48, 32)]
    assert joins._flash(rendered, _JOIN, _FPS, 2, (), regions)["reason"] == "picture masked"
    frames[_JOIN - 1] = frames[_JOIN - 2]
    frames[_JOIN - 1, :30] = foreign[:30]
    assert joins._flash(rendered, _JOIN, _FPS, 2, (), ())["flash"]
    band = [replace(regions[0], h=30)]
    assert not joins._flash(rendered, _JOIN, _FPS, 2, (), band)["flash"]


def test_measured_picture_landing_and_shared_decode(calibration, monkeypatch, tmp_path):
    plan = Plan.build(load_edit_list(calibration.edit), Fraction(30), 48_000)
    landings = sync.check_sync(calibration.clean, plan, {"main": calibration.source})
    original = video.grey_frames
    decodes = []
    measured = []
    original_flash = joins.flash_at

    def decode(media, **kwargs):
        decodes.append(media)
        return original(media, **kwargs)

    def measure(frames, frame, **kwargs):
        measured.append(frame)
        return original_flash(frames, frame, **kwargs)

    monkeypatch.setattr(video, "grey_frames", decode)
    monkeypatch.setattr(joins, "flash_at", measure)
    moved = replace(
        landings,
        segments=[
            replace(row, video_frame=row.video_frame + 1) if row.video_frame is not None else row
            for row in landings.segments
        ],
    )
    result = joins.check_joins(
        calibration.clean,
        plan,
        {"main": calibration.source},
        moved,
        backend=None,
        work_dir=tmp_path,
    )
    assert measured == [cut.frames.render_right for cut in result.cuts]
    assert measured == [cut.out_frame + 1 for cut in plan.cuts]
    assert sum(Path(media) == calibration.clean for media in decodes) == 1
    assert all(cut.flash is not None for cut in result.cuts)
    measured.clear()
    decodes.clear()
    original_boundary = joins._frames

    def missing_landing(*args, **kwargs):
        return replace(original_boundary(*args, **kwargs), render_right=None)

    monkeypatch.setattr(joins, "_frames", missing_landing)
    fallback = joins.check_joins(
        calibration.clean,
        plan,
        {"main": calibration.source},
        landings,
        backend=None,
        work_dir=tmp_path,
    )
    assert all(cut.frames.render_right is None for cut in fallback.cuts)
    assert measured == [cut.out_frame for cut in plan.cuts]
    assert sum(Path(media) == calibration.clean for media in decodes) == 1
    assert all(cut.flash["reason"] is None for cut in fallback.cuts)


def test_all_short_detected_windows_are_skipped(pictures, monkeypatch, tmp_path):
    frames, _ = pictures
    cut = detect.DetectedCut(1, 2, 2 / _FPS, 1.0, {})
    monkeypatch.setattr(detect, "detect_cuts", lambda _: [cut])
    monkeypatch.setattr(video, "grey_frames", lambda *args, **kwargs: frames[:6])
    monkeypatch.setattr(
        ffmpeg,
        "probe",
        lambda *args, **kwargs: SimpleNamespace(
            has_audio=False,
            fps=Fraction(30),
            video_offset_s=0,
        ),
    )
    monkeypatch.setattr(
        detect.structure, "check_structure", lambda *args, **kwargs: SimpleNamespace(findings=[])
    )
    result = detect.check_without_edl(Path("short.mp4"), backend=None, work_dir=tmp_path)
    assert result.results[0].flash is None
    assert result.checks["picture.flash"]["ran"] is False
    assert "not enough" in result.checks["picture.flash"]["reason"]
    assert not [finding for finding in result.findings if finding.check == "picture.flash"]


def test_coverage_skips_unavailable_picture_and_keeps_partial_measurements():
    missing = SimpleNamespace(
        flash={"flash": False, "reason": "no render picture"},
        frames=SimpleNamespace(status="skipped", reason="no render picture"),
        words=SimpleNamespace(status="skipped", reason="no audio"),
        gap_ms=None,
    )
    empty_sync = sync.SyncResult([], None, [])
    coverage = _edit_coverage(
        SimpleNamespace(has_audio=False), empty_sync, SimpleNamespace(cuts=[missing]), None
    )
    assert coverage["picture.flash"] == {"ran": False, "reason": "no render picture"}
    measured = SimpleNamespace(**{**vars(missing), "flash": {"flash": False, "reason": None}})
    coverage = _edit_coverage(
        SimpleNamespace(has_audio=False),
        empty_sync,
        SimpleNamespace(cuts=[missing, measured]),
        None,
    )
    assert coverage["picture.flash"] == {"ran": True, "reason": None}


def test_touch_frames_limits_the_search(pictures):
    frames, foreign = pictures
    frames[_JOIN - 2] = foreign
    assert flash_at(frames, _JOIN, fps=_FPS, touch_frames=2)["flash"]
    assert not flash_at(
        frames, _JOIN, fps=_FPS, touch_frames=Thresholds(touch_cut_frames=1).touch_cut_frames
    )["flash"]
