"""Synthetic colour damage and independent picture landing checks."""

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from yap_studio import ffmpeg, receipt, video
from yap_studio.edl import load_edit_list
from yap_studio.errors import EditListError
from yap_studio.grade import GRADE_VERSION, GradeSpec, apply_rgb, graph_filter, write_cube
from yap_studio.verify import verify


@pytest.fixture(scope="module")
def graded(tmp_path_factory, synth_media) -> dict:
    root = tmp_path_factory.mktemp("film-verify")
    reference = synth_media.path("clean")
    source = synth_media.path("calib")
    spec = GradeSpec("film", 1, "style")
    cube = write_cube(spec, root)
    outputs = {"ungraded": reference}
    for name, filters in (("film", graph_filter(cube)),
                          ("wrong", graph_filter(write_cube(GradeSpec("none"), root / "none"))),
                          ("weak", graph_filter(write_cube(GradeSpec("film", 0.15),
                                                            root / "weak"))),
                          ("eyedropper", graph_filter(write_cube(GradeSpec(
                              "eyedropper", 1, color="#1E6BD6"), root / "eyedropper"))),
                          ("swapped", graph_filter(cube)
                           + ",format=gbrp,colorchannelmixer=rr=0:rb=1:bb=0:br=1"),
                          ("shift", graph_filter(cube)
                           + ",tpad=start=5:start_mode=clone,"
                           "trim=end_frame=576,setpts=PTS-STARTPTS"),
                          ("plain-shift", "tpad=start=5:start_mode=clone,"
                           "trim=end_frame=576,setpts=PTS-STARTPTS")):
        out = root / f"{name}.mp4"
        ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(reference), "-vf", filters,
                    "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
                    "-c:a", "copy", str(out)], timeout_s=120)
        outputs[name] = out
    contract = {"sources": {"main": str(source)},
                "ranges": [{"source": "main", "start": a, "end": b}
                           for a, b in synth_media.spans],
                "grade": {"look": "film", "strength": 1, "version": GRADE_VERSION,
                          "reference": str(reference),
                          "reference_sha256": receipt.sha256_file(reference)}}
    edl = root / "ranges.json"
    edl.write_text(json.dumps(contract))
    return {"edl": edl, "contract": contract, **outputs}


def test_graded_final_passes(graded: dict) -> None:
    report = verify(str(graded["film"]), edl=str(graded["edl"]), asr="none", detail="full",
                    loudness="note")
    assert report["status"] == "pass", report["findings"]
    assert report["checks"]["picture.grade"]["ran"]
    rows = report["segments"]
    assert all(row["confidence"] == "high" for row in rows)
    evidence = report["checks"]["picture.grade"]["evidence"]
    assert evidence["median_error"] < evidence["limit"] / 2
    assert evidence["grade_magnitude"] > 20
    assert evidence["limit"] == 10
    print("Saturated Film correct:", evidence)


def test_camera_region_still_measures_grade(graded: dict, tmp_path: Path) -> None:
    contract = {**graded["contract"], "overlays": [
        {"start_in_output": 0, "duration": 19.2, "region": [0, 0, 1080, 1920],
         "camera_3d": True},
    ]}
    edl = tmp_path / "camera.json"
    edl.write_text(json.dumps(contract))
    report = verify(str(graded["film"]), edl=str(edl), asr="none", loudness="note")
    assert report["checks"]["picture.grade"]["ran"]
    evidence = report["checks"]["picture.grade"]["evidence"]
    assert evidence["median_error"] < evidence["limit"]


@pytest.mark.parametrize("damage", ["ungraded", "wrong", "weak", "swapped", "eyedropper"])
def test_colour_damage_fails(graded: dict, damage: str) -> None:
    report = verify(str(graded[damage]), edl=str(graded["edl"]), asr="none")
    assert report["checks"]["picture.grade"]["status"] == "fail"
    row = next(row for row in report["findings"] if row["check"] == "picture.grade")
    assert row["suggested_action"] == "Re-render with yap-studio assemble."
    if damage == "ungraded":
        assert row["evidence"]["looks_ungraded"]


def test_shift_is_still_detected(graded: dict, tmp_path: Path) -> None:
    report = verify(str(graded["shift"]), edl=str(graded["edl"]), asr="none")
    assert any(row["check"] in {"sync.av_offset", "plan.picture"}
               and row["severity"] in {"warn", "fail"} for row in report["findings"])
    plain_edl = tmp_path / "plain.json"
    plain_edl.write_text(json.dumps({key: value for key, value in graded["contract"].items()
                                     if key != "grade"}))
    ungraded = verify(str(graded["plain-shift"]), edl=str(plain_edl), asr="none")
    checks = {"sync.av_offset", "plan.picture"}
    assert {(row["check"], row["severity"]) for row in report["findings"]
            if row["check"] in checks} == {
                (row["check"], row["severity"]) for row in ungraded["findings"]
                if row["check"] in checks}


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_reference_unavailable_warns(graded: dict, tmp_path: Path, damage: str) -> None:
    contract = json.loads(json.dumps(graded["contract"]))
    if damage == "missing":
        contract["grade"]["reference"] = str(tmp_path / "missing.mp4")
    else:
        contract["grade"]["reference_sha256"] = "0" * 64
    edl = tmp_path / "ranges.json"
    edl.write_text(json.dumps(contract))
    report = verify(str(graded["film"]), edl=str(edl), asr="none")
    coverage = report["checks"]["picture.grade"]
    assert not coverage["ran"]
    assert damage in coverage["reason"]
    assert "576 usable frames" in coverage["reason"]
    assert coverage["evidence"] == {"usable_frames": 576, "minimum_frames": 6}
    rows = [row for row in report["findings"] if row["check"] == "picture.grade"]
    assert len(rows) == 1 and rows[0]["severity"] == "warn"
    assert "Picture colour grade not measured:" in rows[0]["message"]
    assert rows[0]["suggested_action"] == (
        "Check the final by eye or re-render with yap-studio assemble."
    )
    assert report["status"] == "warn", report["findings"]


def test_too_few_grade_frames_warn_in_report(graded: dict, tmp_path: Path) -> None:
    contract = {**graded["contract"], "overlays": [{"start_in_output": 0, "duration": 19.1}]}
    edl = tmp_path / "covered.json"
    edl.write_text(json.dumps(contract))
    report = verify(str(graded["film"]), edl=str(edl), asr="none", loudness="note")
    coverage = report["checks"]["picture.grade"]
    assert not coverage["ran"]
    assert coverage["reason"] == "3 usable frames (needs 6)"
    assert coverage["evidence"]["usable_frames"] == 3
    rows = [row for row in report["findings"] if row["check"] == "picture.grade"]
    assert len(rows) == 1 and rows[0]["severity"] == "warn"
    assert "not measured: 3 usable frames (needs 6)" in rows[0]["message"]
    assert report["status"] == "warn", report["findings"]


@pytest.mark.parametrize("bad", [{}, {"version": 999}, {"version": True}, {"strength": True},
                                  {"reference_sha256": "bad"}, {"look": "bad"},
                                  {"skip_frames": None}, {"skip_frames": [[1, 1]]},
                                  {"skip_frames": [[-1, 2]]}, {"skip_frames": [[True, 2]]},
                                  {"skip_frames": [[1, 2.5]]}, {"skip_frames": [1, 2]},
                                  {"look": "eyedropper"},
                                  {"look": "eyedropper", "color": "#808080"},
                                  {"look": "eyedropper", "color": "invalid"}])
def test_bad_grade_edl(graded: dict, tmp_path: Path, bad: dict) -> None:
    contract = json.loads(json.dumps(graded["contract"]))
    contract["grade"] = {**contract["grade"], **bad} if bad else {}
    edl = tmp_path / "bad.json"
    edl.write_text(json.dumps(contract))
    with pytest.raises(EditListError) as error:
        load_edit_list(edl)
    assert error.value.suggested_action


def test_null_grade_edl_is_invalid(graded: dict, tmp_path: Path) -> None:
    edl = tmp_path / "null.json"
    edl.write_text(json.dumps({**graded["contract"], "grade": None}))
    with pytest.raises(EditListError) as error:
        load_edit_list(edl)
    assert error.value.suggested_action


def test_source_grade_decode_and_legacy_identity(graded: dict) -> None:
    from yap_studio.checks.sync import check_sync
    from yap_studio.plan import Plan

    edits = load_edit_list(graded["edl"])
    plan = Plan.build(edits, fps=ffmpeg.probe(graded["film"]).fps, sample_rate=48000)
    sources = {name: source.path for name, source in edits.sources.items()}
    aware = check_sync(graded["film"], plan, sources, grade=GradeSpec("film", 1))
    unaware = check_sync(graded["film"], plan, sources)
    assert all(row.confidence == "high" for row in aware.segments)
    print("Grade-aware distances:", [row.frame_distance for row in aware.segments])
    print("Without grade-aware decode:", [(row.confidence, row.frame_distance)
                                           for row in unaware.segments])
    np.testing.assert_array_equal(video.grey_frames(graded["ungraded"], count=2),
                                  video.grey_frames(graded["ungraded"], count=2,
                                                    grade=GradeSpec("none")))


def test_grade_skips_masks_holds_and_covered_rectangles(graded: dict, monkeypatch) -> None:
    from yap_studio.checks.grade import check_grade
    from yap_studio.edl.model import Hold, PictureRegion
    from yap_studio.plan import Plan

    edits = replace(load_edit_list(graded["edl"]),
                    picture_masks=[(Fraction(0), Fraction(3))],
                    holds=[Hold(1, 500, 576)],
                    picture_regions=[PictureRegion(Fraction(3), Fraction(20),
                                                   0, 0, 160, 180)])
    edits = replace(edits, grade={**edits.grade, "skip_frames": [[100, 200]]})
    info = ffmpeg.probe(graded["film"])
    plan = Plan.build(edits, fps=info.fps, sample_rate=48000)
    decoded = []

    def rgb(path, frames, **kwargs):
        decoded.append(frames)
        original = np.full((len(frames), 160, 90, 3), (32, 80, 110), dtype=np.uint8)
        if path == graded["film"]:
            final = np.round(apply_rgb(original / 255, GradeSpec("film", 1)) * 255)
            final[:, :80, :45] = (255, 0, 0)
            return final.astype(np.uint8)
        return original

    monkeypatch.setattr(video, "rgb_frames", rgb)
    findings, coverage = check_grade(graded["film"], edits, plan, info)
    assert not findings and coverage["ran"]
    assert coverage["evidence"]["median_error"] < 0.5
    assert len(decoded[0]) == 24 and decoded[0] == decoded[1]
    assert all(90 <= frame < 500 for frame in decoded[0])
    assert all(not 100 <= frame < 200 for frame in decoded[0])
    assert edits.picture_masks == [(Fraction(0), Fraction(3))]
    hidden = replace(edits, picture_regions=[PictureRegion(Fraction(0), Fraction(20),
                                                           0, 0, 320, 180)])
    findings, coverage = check_grade(graded["film"], hidden, plan, info)
    assert not coverage["ran"]
    assert coverage["evidence"]["usable_frames"] == 0
    assert len(findings) == 1 and findings[0].severity == "warn"
