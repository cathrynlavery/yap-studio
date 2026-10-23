"""Person matte CLI output, option forwarding, and structured errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import test_matte_check as check_tests
import test_matte_composite as composite_tests
import test_matte_hyperframes as hyperframes_tests
from test_matte_composite import SyntheticMedia, assert_planes, decode
from test_matte_hyperframes import FakeRenderer, vp9_encoder  # noqa: F401
from test_matte_make import PIXELS, FakeHelper, make_clip

from yap_studio.cli import main
from yap_studio.errors import OutputPathError
from yap_studio.matte import check, hyperframes, vision
from yap_studio.matte.composite import CARDS_SCHEMA
from yap_studio.matte.model import load_matte

hyperframes_clip = hyperframes_tests.clip
composite_media = composite_tests.synthetic


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHelper:
    return FakeHelper(tmp_path, monkeypatch).install([bytes([255]) * PIXELS] * 3)


@pytest.mark.parametrize("json_output", [False, True])
def test_matte_measured_output(
    clip: Path,
    tmp_path: Path,
    fake: FakeHelper,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    out = tmp_path / "out"
    args = ["matte", "make", str(clip), "--out", str(out)]
    assert main(args + (["--json"] if json_output else [])) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        result = json.loads(captured.out)
        assert set(result) == {"out", "status", "reason", "frames", "backend", "seconds"}
        assert result["out"] == str(out) and result["status"] == "measured"
        assert result["reason"] is None and result["frames"] == 3
        assert result["backend"]["name"] == "vision-person-segmentation"
        assert result["backend"]["quality"] == "accurate"
        assert isinstance(result["seconds"], float) and result["seconds"] >= 0
    else:
        assert "Matte: 3 frames, vision-person-segmentation (accurate), median on, " in captured.out
        assert " s.\n" in captured.out
        assert captured.out.endswith(f"Wrote {out / 'matte.mkv'}\nWrote {out / 'matte.json'}\n")
    assert load_matte(out / "matte.json").status == "measured"
    fake.assert_reaped()


def test_matte_no_median_and_quality_reach_pipeline(
    clip: Path, tmp_path: Path, fake: FakeHelper, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert (
        main(
            [
                "matte",
                "make",
                str(clip),
                "--out",
                str(out),
                "--no-median",
                "--quality",
                "balanced",
                "--backend",
                "vision",
            ]
        )
        == 0
    )
    data = load_matte(out / "matte.json")
    assert data.cleanup.median is False and data.backend["quality"] == "balanced"
    assert "(balanced), median off," in capsys.readouterr().out


@pytest.mark.parametrize("json_output", [False, True])
def test_matte_unavailable_output(
    clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    reason = "Vision needs macOS; no renderer"
    monkeypatch.setattr(vision, "unavailable_reason", lambda: "Vision needs macOS")
    monkeypatch.setattr(hyperframes, "unavailable_reason", lambda: "no renderer")
    out = tmp_path / "out"
    args = ["matte", "make", str(clip), "--out", str(out)]
    assert main(args + (["--json"] if json_output else [])) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        result = json.loads(captured.out)
        assert result["status"] == "unavailable" and result["reason"] == reason
        assert result["frames"] == 3 and result["backend"] == {}
        assert result["out"] == str(out) and result["seconds"] >= 0
    else:
        assert captured.out == f"Matte unavailable: {reason}\nWrote {out / 'matte.json'}\n"
    assert load_matte(out / "matte.json").status == "unavailable"
    assert not (out / "matte.mkv").exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_matte_input_error_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool
) -> None:
    args = ["matte", "make", str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "out")]
    assert main(args + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        assert json.loads(captured.err)["error"]["code"] == "invalid_input"
    else:
        assert "error:" in captured.err and "fix:" in captured.err


@pytest.mark.parametrize("json_output", [False, True])
def test_matte_unsafe_path_exit_three(
    clip: Path,
    tmp_path: Path,
    fake: FakeHelper,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "matte.mkv").write_bytes(b"keep")
    args = ["matte", "make", str(clip), "--out", str(out)]
    assert main(args + (["--json"] if json_output else [])) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (out / "matte.mkv").read_bytes() == b"keep"
    if json_output:
        assert json.loads(captured.err)["error"]["code"] == "unsafe_path"
    else:
        assert "error:" in captured.err


def test_matte_required_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["matte"]) == 2
    assert "required" in capsys.readouterr().err


@pytest.mark.parametrize("option", [["--backend", "other"], ["--quality", "fast"]])
def test_matte_rejects_unknown_options(
    option: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["matte", "make", "clip.mp4", "--out", "out", *option]) == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.usefixtures("vp9_encoder")
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("model_exists", [False, True])
def test_hyperframes_backend_guard_and_model_notice(
    hyperframes_clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
    model_exists: bool,
) -> None:
    fake = FakeRenderer(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_HF_STRIPES", "2,5")
    if model_exists:
        model = hyperframes.model_path()
        model.parent.mkdir(parents=True)
        model.touch()
    out = tmp_path / "out"
    args = [
        "matte",
        "make",
        str(hyperframes_clip),
        "--out",
        str(out),
        "--backend",
        "hyperframes",
        "--no-median",
    ]
    assert main(args + (["--json"] if json_output else [])) == 0
    captured = capsys.readouterr()
    assert captured.err == (
        hyperframes.MODEL_NOTICE + "\n" if not model_exists and not json_output else ""
    )
    if json_output:
        result = json.loads(captured.out)
        assert result["status"] == "measured"
        assert result["backend"]["name"] == "hyperframes-remove-background"
    else:
        assert "Matte: 12 frames, hyperframes-remove-background, median off," in captured.out
        assert "Guard: 2 frame(s) had no person and were left empty.\n" in captured.out
    assert load_matte(out / "matte.json").empty_frames == (2, 5)
    fake.assert_reaped()


def test_model_notice_precedes_cli_execution_and_error(
    hyperframes_clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeRenderer(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_HF_MODE", "exit")
    assert (
        main(
            [
                "matte",
                "make",
                str(hyperframes_clip),
                "--out",
                str(tmp_path / "out"),
                "--backend",
                "hyperframes",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err.startswith(hyperframes.MODEL_NOTICE + "\nerror:")


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("use_matte", [False, True])
def test_composite_output_and_relative_layer_paths(
    composite_media: SyntheticMedia,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
    use_matte: bool,
) -> None:
    card = composite_media.card.to_dict()
    card["layer"] = composite_media.layer.name
    document = tmp_path / "cards.json"
    document.write_text(json.dumps({"schema": CARDS_SCHEMA, "cards": [card]}))
    out = tmp_path / "composite.mp4"
    args = ["matte", "composite", str(composite_media.plate), "--cards", str(document),
            "--out", str(out), "--lossless"]
    if use_matte:
        args += ["--matte", str(composite_media.matte)]
    assert main(args + (["--json"] if json_output else [])) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    warning = "person cutout unavailable; card top renders in front of the person"
    if json_output:
        result = json.loads(captured.out)
        assert result["path"] == str(out) and result["frames"] == composite_tests.FRAMES
        assert result["matte_used"] == use_matte
        assert result["cards"] == [composite_media.card.to_dict()]
        assert result["warnings"] == ([] if use_matte else [warning])
    else:
        matte = "matte used" if use_matte else "no matte"
        expected = f"Composite: 12 frames, 1 card(s) (1 behind the head), {matte}.\n"
        if not use_matte:
            expected += f"warning: {warning}\n"
        assert captured.out == expected + f"Wrote {out}\n"
    plate = decode(composite_media.plate)
    expected_pixels = tuple(plane.copy() for plane in plate)
    composite_tests.expected_card(expected_pixels, plate, composite_media.card, keyed=use_matte)
    assert_planes(decode(out), expected_pixels)


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("kind", ["missing", "invalid-json", "schema", "record", "unowned"])
def test_composite_error_exits(
    composite_media: SyntheticMedia,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
    kind: str,
) -> None:
    document = tmp_path / "cards.json"
    out = tmp_path / "out.mp4"
    card = composite_media.card.to_dict()
    payload = {"schema": CARDS_SCHEMA, "cards": [card]}
    if kind == "record":
        card["x"] = 3
    if kind == "schema":
        payload["schema"] = "wrong"
    if kind != "missing":
        document.write_text("not JSON" if kind == "invalid-json" else json.dumps(payload))
    if kind == "unowned":
        out.write_bytes(b"keep")
    args = ["matte", "composite", str(composite_media.plate), "--cards", str(document),
            "--out", str(out)]
    assert main(args + (["--json"] if json_output else [])) == (3 if kind == "unowned" else 2)
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        code = json.loads(captured.err)["error"]["code"]
        assert code == {"missing": "invalid_input", "unowned": "unsafe_path"}.get(
            kind, "invalid_parameter"
        )
    else:
        assert "error:" in captured.err
        if kind != "unowned":
            assert "fix:" in captured.err
    if kind == "unowned":
        assert out.read_bytes() == b"keep"
    else:
        assert not out.exists()


def test_composite_cannot_overwrite_cards_document(
    composite_media: SyntheticMedia, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = tmp_path / "cards.json"
    document.write_text(json.dumps({"schema": CARDS_SCHEMA, "cards": []}))
    before = document.read_bytes()
    assert main(["matte", "composite", str(composite_media.plate), "--cards", str(document),
                 "--out", str(document)]) == 3
    assert "error:" in capsys.readouterr().err
    assert document.read_bytes() == before


@pytest.mark.parametrize("status", ["pass", "fail", "review", "skipped"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_cutout_output_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    status: str, strict: bool, json_output: bool,
) -> None:
    media = check_tests.make_media(
        tmp_path, moving=status == "review", defect="dropout" if status == "review" else None
    )
    out = media.composite
    if status == "fail":
        pixels = composite_tests.decode(out, check_tests.WIDTH, check_tests.HEIGHT)[0].copy()
        pixels[6, 250:266, :16] = 200
        out = check_tests.encode(tmp_path / "bad.mp4", pixels)
    document = tmp_path / "cards.json"
    card = media.card.to_dict()
    card["layer"] = media.card.layer.name
    document.write_text(json.dumps({"schema": CARDS_SCHEMA, "cards": [card]}))
    args = ["matte", "check", str(media.plate), "--cards", str(document),
            "--composite", str(out)]
    if status != "skipped":
        args += ["--matte", str(media.matte)]
    if strict:
        args += ["--strict"]
    if json_output:
        args += ["--json"]
    expected_exit = int(status == "fail" or (strict and status in ("review", "skipped")))
    assert main(args) == expected_exit
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        report = json.loads(captured.out)
        assert report["schema"] == "yap-studio/cutout-report@1" and report["status"] == status
        assert report["frames"] == check_tests.FRAMES and report["behind"] == report["cards"] == 1
        assert len(report["checks"]) == 5
    else:
        expected = ("SKIPPED person cutout: person matte is missing" if status == "skipped" else
                    f"{status.upper()} person cutout: 1 card behind the person, 16 frames")
        lines = captured.out.splitlines()
        assert len(lines) == 6 and lines[0] == expected
        assert lines[1].startswith("  matte.frames: ")
        assert lines[2].startswith("  matte.doubled_edges: ")
        assert lines[3].startswith("  matte.flicker: ")
        assert lines[4].startswith("  matte.hand_dropout: ")
        assert lines[5].startswith("  matte.occlusion: ")
        if status == "review":
            assert "frame 7 at 0:00.70, dip 0.125" in lines[4]
        elif status == "fail":
            assert "outside share" in lines[2] and "frame 6 (0:00.60)" in lines[2]
        elif status == "pass":
            assert "p95 0, max 0 static pixels per frame pair in card hero" in lines[3]
        else:
            assert all(": skipped," in line for line in lines[1:])


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_cutout_occlusion_review_output_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], strict: bool, json_output: bool,
) -> None:
    media = check_tests.make_ink_media(tmp_path)
    document = tmp_path / "cards.json"
    card = media.card.to_dict()
    card["layer"] = media.card.layer.name
    document.write_text(json.dumps({"schema": CARDS_SCHEMA, "cards": [card]}))
    args = ["matte", "check", str(media.plate), "--cards", str(document),
            "--matte", str(media.matte), "--composite", str(media.composite)]
    if strict:
        args += ["--strict"]
    if json_output:
        args += ["--json"]
    assert main(args) == int(strict)
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        report = json.loads(captured.out)
        assert report["status"] == "review"
        result = next(item for item in report["checks"] if item["id"] == "matte.occlusion")
        assert result["status"] == "review"
        assert result["data"]["cards"][0]["share"] == 0.25
        assert result["data"]["cards"][0]["frame"] == 7
        assert report["thresholds"]["ink_diff"] == 40
        assert report["thresholds"]["occlusion_max"] == 0.02
    else:
        assert captured.out.startswith("REVIEW person cutout:")
        assert captured.out.endswith(
            "  matte.occlusion: review, card g-skill hides 25.0% of its text "
            "at frame 7 (0:00.70) (limit 2%)\n"
        )


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("kind", ["missing", "invalid-json", "schema", "record", "composite"])
def test_cutout_input_errors_exit_two(
    composite_media: SyntheticMedia, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    kind: str, json_output: bool,
) -> None:
    document = tmp_path / "cards.json"
    card = composite_media.card.to_dict()
    payload = {"schema": CARDS_SCHEMA, "cards": [card]}
    if kind == "record":
        card["x"] = 3
    elif kind == "schema":
        payload["schema"] = "wrong"
    if kind != "missing":
        document.write_text("not JSON" if kind == "invalid-json" else json.dumps(payload))
    args = ["matte", "check", str(composite_media.plate), "--cards", str(document),
            "--composite", str(tmp_path / "absent.mp4" if kind == "composite"
                               else composite_media.plate)]
    assert main(args + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        assert json.loads(captured.err)["error"]["code"] == (
            "invalid_input" if kind in ("missing", "composite") else "invalid_parameter"
        )
    else:
        assert "error:" in captured.err and "fix:" in captured.err


@pytest.mark.parametrize("json_output", [False, True])
def test_cutout_unsafe_path_exit_three(
    composite_media: SyntheticMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], json_output: bool,
) -> None:
    document = tmp_path / "cards.json"
    document.write_text(json.dumps({"schema": CARDS_SCHEMA,
                                    "cards": [composite_media.card.to_dict()]}))

    def unsafe(*args, **kwargs):
        raise OutputPathError("Unsafe media path.", code="unsafe_path")

    monkeypatch.setattr(check, "check_cutout", unsafe)
    args = ["matte", "check", str(composite_media.plate), "--cards", str(document),
            "--composite", str(composite_media.plate)]
    assert main(args + (["--json"] if json_output else [])) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        assert json.loads(captured.err)["error"]["code"] == "unsafe_path"
    else:
        assert "error: Unsafe media path." in captured.err


def test_cutout_requires_cards_and_composite(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["matte", "check", "plate.mp4"]) == 2
    assert "--cards" in capsys.readouterr().err
