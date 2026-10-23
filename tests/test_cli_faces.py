"""Face detection and clearance command output, documents, and input protection."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_faces_detect import FakeDetector, VariableRateDetector, face, make_clip

from yap_studio.cli import main
from yap_studio.faces import DEFAULT_HAIR, DetectorRun, command, detect, load_faces
from yap_studio.style import add


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


@pytest.fixture
def detector(monkeypatch: pytest.MonkeyPatch) -> FakeDetector:
    fake = FakeDetector()
    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    monkeypatch.setattr(detect, "default_detector", lambda: fake)
    return fake


def test_faces_detect_writes_document_and_summary(
    clip: Path, tmp_path: Path, detector: FakeDetector, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "faces.json"
    assert main(["faces", "detect", str(clip), "--out", str(out), "--every", "5"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Faces: 19 of 30 frames have a face, 1 gaps, every 5 frame(s), " in captured.out
    assert "gap: frames 10 to 21 (0:00.33 to 0:00.70)" in captured.out
    assert f"Wrote {out}\n" in captured.out
    data = load_faces(out)
    assert data.status == "measured"
    assert len(data.frames) == 30
    assert detector.calls == [(clip.resolve(), 5)]


def test_faces_detect_json_output(
    clip: Path, tmp_path: Path, detector: FakeDetector, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "faces.json"
    assert main(["faces", "detect", str(clip), "--out", str(out), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    assert set(result) == {
        "out",
        "status",
        "reason",
        "frames",
        "frames_with_faces",
        "gaps",
        "seconds",
    }
    assert result["out"] == str(out)
    assert result["status"] == "measured"
    assert result["reason"] is None
    assert result["frames"] == 30
    assert result["frames_with_faces"] == 11
    assert result["gaps"] == [{"start_frame": 6, "end_frame": 25}]
    assert isinstance(result["seconds"], float) and result["seconds"] >= 0
    assert load_faces(out).status == "measured"
    assert detector.calls == [(clip.resolve(), 1)]


@pytest.mark.parametrize("json_output", [False, True])
def test_faces_unmeasured_writes_reason(
    clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    reason = "face detection needs the Xcode command line tools"
    monkeypatch.setattr(detect, "unavailable_reason", lambda: reason)

    def unexpected() -> None:
        pytest.fail("unmeasured CLI invoked the default detector")

    monkeypatch.setattr(detect, "default_detector", unexpected)
    out = tmp_path / "faces.json"
    args = ["faces", "detect", str(clip), "--out", str(out)]
    assert main(args + (["--json"] if json_output else [])) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        result = json.loads(captured.out)
        assert result["status"] == "unmeasured"
        assert result["reason"] == reason
        assert result["frames_with_faces"] == 0
    else:
        assert captured.out == f"Faces unmeasured: {reason}\nWrote {out}\n"
    data = load_faces(out)
    assert data.status == "unmeasured"
    assert data.reason == reason
    assert data.frames == ()


@pytest.mark.parametrize("json_output", [False, True])
def test_faces_refuses_existing_unowned_output(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    out = tmp_path / "faces.json"
    out.write_text('{"keep": "this"}\n')
    args = ["faces", "detect", str(clip), "--out", str(out)]
    assert main(args + (["--json"] if json_output else [])) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsafe_path" in captured.err if json_output else "error:" in captured.err
    assert out.read_text() == '{"keep": "this"}\n'
    assert detector.calls


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_faces_never_overwrites_video(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    capsys: pytest.CaptureFixture[str],
    alias: str,
) -> None:
    original = clip.read_bytes()
    out = clip if alias == "same" else tmp_path / "alias.json"
    if alias == "symlink":
        out.symlink_to(clip)
    elif alias == "hardlink":
        out.hardlink_to(clip)
    assert main(["faces", "detect", str(clip), "--out", str(out)]) == 2
    assert "input file" in capsys.readouterr().err
    assert clip.read_bytes() == original
    assert detector.calls


@pytest.mark.parametrize("json_output", [False, True])
def test_faces_errors_follow_cli_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool
) -> None:
    args = ["faces", "detect", str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "faces.json")]
    assert main(args + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        assert json.loads(captured.err)["error"]["code"] == "invalid_input"
    else:
        assert "error:" in captured.err and "fix:" in captured.err


def test_faces_every_zero_is_rejected_by_argparse(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["faces", "detect", "video.mp4", "--out", "faces.json", "--every", "0"]) == 2
    assert "every must be an integer >= 1" in capsys.readouterr().err


def test_faces_subcommand_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["faces"]) == 2
    assert "required" in capsys.readouterr().err


def write_layout(
    tmp_path: Path, *, y: int = 20, captions: bool = False, clearance_px: float | None = 5
) -> Path:
    """Write a layout in the synthetic clip's display coordinates."""
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "schema": "yap-studio/clearance-layout@1",
                "width": 108,
                "height": 192,
                "clearance_px": clearance_px,
                "caption_gap_px": 10,
                "elements": [
                    {
                        "id": "card-1",
                        "kind": "card",
                        "x": 10,
                        "y": y,
                        "w": 10,
                        "h": 12,
                        "start_frame": 0,
                        "end_frame": 30,
                    }
                ],
                "captions": [
                    {
                        "id": "phrase-1",
                        "text": "visible words",
                        "start_frame": 3,
                        "end_frame": 15,
                        "boxes": [{"x": 10, "y": 116, "w": 10, "h": 8}],
                    }
                ]
                if captions
                else [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_faces_check_names_card_time_and_face_and_moving_card_passes(
    clip: Path, tmp_path: Path, detector: FakeDetector, capsys: pytest.CaptureFixture[str]
) -> None:
    original = clip.read_bytes()
    layout = write_layout(tmp_path)
    assert main(["faces", "check", str(clip), "--layout", str(layout)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "FAIL face clearance: 1 violation(s), margin 5 px (layout), "
        "hair 0.6 above and 0.15 each side, caption gap 10 px; "
        "faces found in 11 of 30 frames\n"
        "  card card-1 covers the head from 0:00.00 to 0:00.20 (frames 0 to 6): "
        "overlaps by 12 px, needs 5 px; face x 10 y 20 w 10 h 12, head top y 13 at 0:00.00\n"
    )
    assert detector.calls == [(clip.resolve(), 1)]
    layout = write_layout(tmp_path, y=100)
    assert main(["faces", "check", str(clip), "--layout", str(layout)]) == 0
    assert capsys.readouterr().out == (
        "PASS face clearance: no overlap with detected faces; 1 elements and 0 phrases "
        "in 30 frames, faces found in 11, none in 19, margin 5 px (layout), "
        "hair 0.6 above and 0.15 each side\n"
    )
    assert clip.read_bytes() == original
    assert set(item.name for item in tmp_path.iterdir()) == {"clip.mp4", "layout.json"}


@pytest.mark.parametrize("json_output", [False, True])
def test_faces_check_rejects_document_from_a_different_video(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    data = detect.detect_faces(clip)
    faces_path = tmp_path / "faces.json"
    detect.write_faces(faces_path, data)
    different = tmp_path / "different.mp4"
    different.write_bytes(clip.read_bytes() + b"different")
    layout = write_layout(tmp_path)

    def unexpected() -> None:
        pytest.fail("--faces invoked the default detector")

    monkeypatch.setattr(detect, "default_detector", unexpected)
    args = ["faces", "check", str(different), "--layout", str(layout), "--faces", str(faces_path)]
    assert main(args + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "different video" in captured.err
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["code"] == "invalid_parameter"
        assert (
            f"run yap-studio faces detect {different} --out faces.json" == error["suggested_action"]
        )
    else:
        assert f"fix: run yap-studio faces detect {different} --out faces.json\n" in captured.err


def test_faces_check_matching_document_skips_detection(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = detect.detect_faces(clip)
    faces_path = tmp_path / "faces.json"
    detect.write_faces(faces_path, data)
    layout = write_layout(tmp_path, y=100)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("matching --faces invoked detection")

    monkeypatch.setattr(detect, "detect_faces", unexpected)
    assert (
        main(["faces", "check", str(clip), "--layout", str(layout), "--faces", str(faces_path)])
        == 0
    )
    assert "PASS face clearance" in capsys.readouterr().out
    assert detector.calls == [(clip.resolve(), 1)]


@pytest.mark.parametrize("every", [1, 3])
def test_faces_check_detects_requested_samples_despite_matching_sparse_sidecar(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], every: int,
) -> None:
    class BriefFaceDetector(FakeDetector):
        def detect(self, video: Path, *, every: int) -> DetectorRun:
            run = super().detect(video, every=every)
            return replace(run, samples={i: (face(),) if i == 3 else () for i in run.samples})

    fake = BriefFaceDetector()
    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    monkeypatch.setattr(detect, "default_detector", lambda: fake)
    sparse = detect.detect_faces(clip, every=2)
    assert not any(sparse.frames)
    sidecar = clip.with_suffix(".faces.json")
    detect.write_faces(sidecar, sparse)
    original = sidecar.read_bytes()
    layout = write_layout(tmp_path)
    assert main(["faces", "check", str(clip), "--layout", str(layout),
                 "--every", str(every), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "fail"
    violation = report["violations"][0]
    assert violation["start_frame"] <= 3 < violation["end_frame"]
    assert fake.calls == [(clip.resolve(), 2), (clip.resolve(), every)]
    assert sidecar.read_bytes() == original


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_faces_check_unmeasured_status_and_strict_exit(
    clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    strict: bool,
    json_output: bool,
) -> None:
    reason = "face detector unavailable"
    monkeypatch.setattr(detect, "unavailable_reason", lambda: reason)
    layout = write_layout(tmp_path)
    args = ["faces", "check", str(clip), "--layout", str(layout)]
    if strict:
        args.append("--strict")
    if json_output:
        args.append("--json")
    assert main(args) == int(strict)
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        report = json.loads(captured.out)
        assert report["status"] == "unmeasured" and report["reason"] == reason
        assert report["violations"] == []
        assert report["coverage"] is None
    else:
        assert captured.out == f"UNMEASURED face clearance: {reason}\n"


def test_faces_check_json_report_and_every_option(
    clip: Path, tmp_path: Path, detector: FakeDetector, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = write_layout(tmp_path)
    args = [
        "faces",
        "check",
        str(clip),
        "--layout",
        str(layout),
        "--every",
        "5",
        "--json",
        "--hair-above",
        "0",
        "--hair-sides",
        "0",
    ]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["schema"] == "yap-studio/clearance-report@1"
    assert report["timing"] == "fps"
    assert report["hair"] == {"above": 0, "sides": 0} and report["margin_source"] == "layout"
    assert report["coverage"] == {"frames_with_faces": 19, "frames_without_faces": 11}
    assert (
        report["status"],
        report["fps"],
        report["frames"],
        report["elements"],
        report["captions"],
    ) == (
        "fail",
        "30/1",
        30,
        1,
        0,
    )
    (violation,) = report["violations"]
    assert violation["target_id"] == "card-1" and violation["kind"] == "face"
    assert violation["start_s"] == violation["time_s"] == 0
    assert violation["end_s"] == 0.333
    assert violation["face"] == {"x": 10, "y": 20, "w": 10, "h": 12}
    assert violation["head"] == violation["face"]
    assert detector.calls == [(clip.resolve(), 5)]


@pytest.mark.parametrize("caption_y", [116, 110])
def test_faces_check_caption_gap_can_fail_with_unmeasured_faces(
    clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caption_y: int,
) -> None:
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "face detector unavailable")
    layout = write_layout(tmp_path, y=100, captions=True)
    document = json.loads(layout.read_text())
    document["captions"][0]["boxes"][0]["y"] = caption_y
    layout.write_text(json.dumps(document))
    args = ["faces", "check", str(clip), "--layout", str(layout)]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    distance = "is 4 px from card card-1" if caption_y == 116 else "overlaps card card-1 by 2 px"
    assert captured.out == (
        "FAIL face clearance: 1 violation(s), margin 5 px (layout), "
        "hair 0.6 above and 0.15 each side, caption gap 10 px\n"
        "  faces unmeasured: face detector unavailable; the face check did not run\n"
        f'  caption phrase-1 "visible words" {distance} '
        "from 0:00.10 to 0:00.50 (frames 3 to 15), needs 10 px\n"
    )
    assert main(args + ["--caption-gap", "0"]) == (0 if caption_y == 116 else 1)
    capsys.readouterr()


def test_faces_check_caption_face_text_and_positive_gap_output(
    clip: Path, tmp_path: Path, detector: FakeDetector, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = write_layout(tmp_path)
    document = json.loads(layout.read_text())
    document["elements"] = []
    document["captions"] = [
        {
            "id": "phrase-1",
            "text": "visible words",
            "start_frame": 1,
            "end_frame": 3,
            "boxes": [{"x": 10, "y": 34, "w": 10, "h": 10}],
        }
    ]
    layout.write_text(json.dumps(document))
    args = ["faces", "check", str(clip), "--layout", str(layout)]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert captured.out == (
        "FAIL face clearance: 1 violation(s), margin 5 px (layout), "
        "hair 0.6 above and 0.15 each side, caption gap 10 px; "
        "faces found in 11 of 30 frames\n"
        '  caption phrase-1 "visible words" covers the head from 0:00.03 to 0:00.10 '
        "(frames 1 to 3): gap 2 px, needs 5 px; face x 11 y 20 w 10 h 12, "
        "head top y 13 at 0:00.03\n"
    )
    assert main(args + ["--margin", "2"]) == 0
    assert "PASS face clearance" in capsys.readouterr().out


def test_faces_check_cli_uses_presentation_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = make_clip(tmp_path / "variable.mp4", frames=60, fps=Fraction(45, 2))
    fake = VariableRateDetector()
    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    monkeypatch.setattr(detect, "default_detector", lambda: fake)
    layout = write_layout(tmp_path)
    document = json.loads(layout.read_text())
    document["elements"][0].update(start_frame=30, end_frame=60)
    layout.write_text(json.dumps(document))
    args = [
        "faces",
        "check",
        str(clip),
        "--layout",
        str(layout),
        "--hair-above",
        "0",
        "--hair-sides",
        "0",
    ]
    assert main(args) == 1
    assert capsys.readouterr().out == (
        "FAIL face clearance: 1 violation(s), margin 5 px (layout), "
        "hair 0 above and 0 each side, caption gap 10 px; "
        "faces found in 60 of 60 frames\n"
        "  card card-1 covers the head from 0:02.00 to 0:03.00 (frames 30 to 60): "
        "overlaps by 10 px, needs 5 px; face x 10 y 20 w 10 h 12, head top y 20 at 0:02.00\n"
    )
    assert main(args + ["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["timing"] == "pts"
    assert report["violations"][0]["start_s"] == 2.0


def test_faces_detect_gap_cli_uses_presentation_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class GappedDetector(VariableRateDetector):
        def detect(self, video: Path, *, every: int) -> DetectorRun:
            run = super().detect(video, every=every)
            return replace(
                run, samples={i: boxes if i < 30 else () for i, boxes in run.samples.items()}
            )

    clip = make_clip(tmp_path / "variable.mp4", frames=60, fps=Fraction(45, 2))
    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    monkeypatch.setattr(detect, "default_detector", GappedDetector)
    out = tmp_path / "faces.json"
    assert main(["faces", "detect", str(clip), "--out", str(out)]) == 0
    assert "gap: frames 30 to 60 (0:02.00 to 0:03.00)" in capsys.readouterr().out
    data = load_faces(out)
    assert data.time_at(30) == 2.0 and data.time_at(60) == 3.0


def test_faces_check_pass_with_no_detections_names_zero_coverage(
    clip: Path, tmp_path: Path, detector: FakeDetector, capsys: pytest.CaptureFixture[str]
) -> None:
    data = detect.detect_faces(clip)
    data = replace(data, frames=((),) * 30, gaps=((0, 30),))
    faces_path = detect.write_faces(tmp_path / "faces.json", data)
    layout = write_layout(tmp_path)
    assert (
        main(["faces", "check", str(clip), "--layout", str(layout), "--faces", str(faces_path)])
        == 0
    )
    assert capsys.readouterr().out == (
        "PASS face clearance: no overlap with detected faces; 1 elements and 0 phrases "
        "in 30 frames, faces found in 0, none in 30, margin 5 px (layout), "
        "hair 0.6 above and 0.15 each side\n"
    )


@pytest.mark.parametrize("input_kind", ["video", "layout", "faces", "invalid_layout"])
@pytest.mark.parametrize("json_output", [False, True])
def test_faces_check_errors_follow_cli_contract(
    clip: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    input_kind: str,
    json_output: bool,
) -> None:
    layout = write_layout(tmp_path)
    video = tmp_path / "missing.mp4" if input_kind == "video" else clip
    if input_kind == "layout":
        layout = tmp_path / "missing.json"
    elif input_kind == "invalid_layout":
        layout.write_text('{"schema": "wrong"}')
    args = ["faces", "check", str(video), "--layout", str(layout)]
    if input_kind == "faces":
        args += ["--faces", str(tmp_path / "missing-faces.json")]
    assert main(args + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["code"] == (
            "invalid_parameter" if input_kind == "invalid_layout" else "invalid_input"
        )
    else:
        assert "error:" in captured.err and "fix:" in captured.err


@pytest.mark.parametrize("option", ["--margin", "--caption-gap"])
@pytest.mark.parametrize("value", ["-1", "nan", "inf", "text"])
def test_faces_check_invalid_margins_are_rejected(
    option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["faces", "check", "clip.mp4", "--layout", "layout.json", option, value]) == 2
    assert "pixels must be a finite number >= 0" in capsys.readouterr().err


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize(
    ("layout_margin", "option_margin", "expected_margin", "expected_source"),
    [(None, None, 50, "style"), (5, None, 5, "layout"), (5, 10, 10, "option"), (5, 0, 0, "option")],
)
def test_faces_check_style_margin_precedence_in_json_and_text(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
    layout_margin: float | None,
    option_margin: float | None,
    expected_margin: float,
    expected_source: str,
) -> None:
    style = tmp_path / "style.json"
    style.write_text(json.dumps({"schema_version": 1, "graphics": {"face_clearance_px": 50}}))
    layout = write_layout(tmp_path, y=100, clearance_px=layout_margin)
    if layout_margin is None:
        document = json.loads(layout.read_text())
        del document["clearance_px"]
        layout.write_text(json.dumps(document))
    args = ["faces", "check", str(clip), "--layout", str(layout), "--style", str(style)]
    if option_margin is not None:
        args += ["--margin", str(option_margin)]
    assert main(args + (["--json"] if json_output else [])) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        report = json.loads(captured.out)
        assert report["margin_px"] == expected_margin
        assert report["margin_source"] == expected_source
        assert report["hair"] == DEFAULT_HAIR.to_dict()
    else:
        assert captured.out == (
            "PASS face clearance: no overlap with detected faces; 1 elements and 0 phrases "
            "in 30 frames, faces found in 11, none in 19, "
            f"margin {expected_margin:g} px ({expected_source}), "
            "hair 0.6 above and 0.15 each side\n"
        )


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("explicit_null", [False, True])
def test_faces_check_default_margin_never_resolves_style(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
    explicit_null: bool,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("faces check without --style resolved a style")

    monkeypatch.setattr(command, "resolve", unexpected)
    layout = write_layout(tmp_path, y=100, clearance_px=None)
    if not explicit_null:
        document = json.loads(layout.read_text())
        del document["clearance_px"]
        layout.write_text(json.dumps(document))
    args = ["faces", "check", str(clip), "--layout", str(layout)]
    assert main(args + (["--json"] if json_output else [])) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        report = json.loads(captured.out)
        assert report["margin_px"] == 24 and report["margin_source"] == "default"
        assert report["hair"] == DEFAULT_HAIR.to_dict()
    else:
        assert captured.out == (
            "PASS face clearance: no overlap with detected faces; 1 elements and 0 phrases "
            "in 30 frames, faces found in 11, none in 19, margin 24 px (default), "
            "hair 0.6 above and 0.15 each side\n"
        )


def test_faces_check_zero_hair_flags_restore_face_box_only_clearance(
    clip: Path, tmp_path: Path, detector: FakeDetector, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = write_layout(tmp_path, y=0)
    args = ["faces", "check", str(clip), "--layout", str(layout)]
    assert main(args + ["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["hair"] == DEFAULT_HAIR.to_dict() and report["margin_source"] == "layout"
    (violation,) = report["violations"]
    assert violation["face"] == {"x": 10, "y": 20, "w": 10, "h": 12}
    assert violation["head"] == {"x": 8.5, "y": 12.8, "w": 13, "h": 19.2}
    assert violation["gap_px"] == 0.8 and violation["required_px"] == 5
    flags = ["--hair-above", "0", "--hair-sides", "0"]
    assert main(args + flags + ["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["hair"] == {"above": 0, "sides": 0} and report["violations"] == []
    assert main(args + flags) == 0
    assert "hair 0 above and 0 each side\n" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("layout_hair", "flags", "expected"),
    [
        (None, [], {"above": 0.6, "sides": 0.15}),
        (None, ["--hair-above", "0"], {"above": 0, "sides": 0.15}),
        ({"above": 0.4, "sides": 0.2}, [], {"above": 0.4, "sides": 0.2}),
        ({"above": 0.4, "sides": 0.2}, ["--hair-above", "0"], {"above": 0, "sides": 0.2}),
        ({"above": 0.4, "sides": 0.2}, ["--hair-sides", "0"], {"above": 0.4, "sides": 0}),
        ({"above": 0}, ["--hair-sides", "1"], {"above": 0, "sides": 1}),
    ],
)
def test_faces_check_hair_precedence_per_component(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    capsys: pytest.CaptureFixture[str],
    layout_hair: dict[str, float] | None,
    flags: list[str],
    expected: dict[str, float],
) -> None:
    layout = write_layout(tmp_path)
    document = json.loads(layout.read_text())
    document["hair"] = layout_hair
    layout.write_text(json.dumps(document))
    assert main(["faces", "check", str(clip), "--layout", str(layout), "--json", *flags]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["hair"] == expected
    face = report["violations"][0]["face"]
    head = report["violations"][0]["head"]
    assert head["y"] == pytest.approx(face["y"] - expected["above"] * face["h"])
    assert head["x"] == pytest.approx(face["x"] - expected["sides"] * face["w"])


@pytest.mark.parametrize("spec_kind", ["preset", "library"])
def test_faces_check_resolves_preset_and_library_style_specs(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    spec_kind: str,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    expected_margin = 24
    spec = "preset:clean_minimal"
    if spec_kind == "library":
        source = tmp_path / "library-style.json"
        source.write_text(json.dumps({"schema_version": 1, "graphics": {"face_clearance_px": 50}}))
        spec = add(source).id
        expected_margin = 50
    layout = write_layout(tmp_path, y=100, clearance_px=None)
    assert (
        main(["faces", "check", str(clip), "--layout", str(layout), "--style", spec, "--json"]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["margin_px"] == expected_margin and report["margin_source"] == "style"


@pytest.mark.parametrize("json_output", [False, True])
def test_faces_check_bad_style_exits_two_with_error(
    clip: Path,
    tmp_path: Path,
    detector: FakeDetector,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    layout = write_layout(tmp_path)
    args = ["faces", "check", str(clip), "--layout", str(layout), "--style", "preset:missing"]
    assert main(args + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "unknown preset" in captured.err
    if json_output:
        assert json.loads(captured.err)["error"]["code"] == "invalid_parameter"
    else:
        assert "error:" in captured.err
    assert detector.calls == []


@pytest.mark.parametrize("option", ["--hair-above", "--hair-sides"])
@pytest.mark.parametrize("value", ["-1", "3", "nan", "inf", "text"])
def test_faces_check_invalid_hair_ratios_are_rejected_by_argparse(
    option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["faces", "check", "clip.mp4", "--layout", "layout.json", option, value]) == 2
    assert "ratio must be a finite number from 0 to 2" in capsys.readouterr().err
