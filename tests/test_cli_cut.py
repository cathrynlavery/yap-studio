"""CLI cut outcomes and read-only story inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import synth
from test_cut import make_project, snapshot, time_segment, write_edit, write_json

from yap_studio.cli import main
from yap_studio.stages import cut as cut_stage


@pytest.mark.parametrize(("enabled", "headroom", "gains", "expected"), [
    (True, 0, (0.0, 8.0), "Take levels matched to t1: t2 +8.0 dB"),
    (True, -2, (-2.0, 6.0), "Take levels matched to t1: t1 -2.0 dB, t2 +6.0 dB"),
    (False, 0, (0.0, 0.0), "Take levels kept as recorded (match_takes is false)."),
    (True, 0, (0.0, 0.0), None),
])
def test_cut_prints_take_levels(tmp_path: Path, capsys: pytest.CaptureFixture,
                               monkeypatch: pytest.MonkeyPatch, enabled: bool,
                               headroom: float, gains: tuple, expected: str | None) -> None:
    result = {"status": "verified", "noop": False, "out": "cut/cut.mp4", "warnings": [],
              "take_levels": {"match_takes": enabled, "reference": "t1",
                              "headroom_db": headroom, "takes": {
                                  "t1": {"gain_db": gains[0]}, "t2": {"gain_db": gains[1]},
                              }}}
    monkeypatch.setattr(cut_stage, "cut", lambda *args, **kwargs: result)
    assert main(["cut", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    if expected:
        assert lines[:2] == ["Cut verified.", expected]
    else:
        assert not any(line.startswith("Take levels") for line in lines)


@pytest.mark.parametrize("limited", ["gain", "peak"])
def test_cut_names_partial_take_matching(tmp_path: Path, capsys: pytest.CaptureFixture,
                                        monkeypatch: pytest.MonkeyPatch, limited: str) -> None:
    result = {"status": "verified", "noop": False, "out": "cut/cut.mp4", "warnings": [],
              "take_levels": {"match_takes": True, "reference": "t1", "headroom_db": -12,
                              "takes": {"t1": {"gain_db": -12, "limited": None},
                                        "t2": {"gain_db": -1, "limited": limited}}}}
    monkeypatch.setattr(cut_stage, "cut", lambda *args, **kwargs: result)
    assert main(["cut", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[1] == (
        "Take levels adjusted toward t1: t1 -12.0 dB, t2 -1.0 dB"
    )


def test_edit_check_json_and_text_are_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = synth.calib_source(tmp_path / "source.mov", seconds=4)
    root = make_project(tmp_path / "project", [source])
    write_edit(root, [time_segment("story", "t1", 0.3, 2.7)])
    before = snapshot(root)
    assert main(["edit", "check", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_frames"] == 72
    assert result["segments"][0]["text"] == "bright morning light"
    assert main(["edit", "check", str(root)]) == 0
    assert "Total: 72 frames, 115200 samples" in capsys.readouterr().out
    assert snapshot(root) == before


def test_repeat_warnings_are_advisory_and_saved_in_cut_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    source = synth.calib_source(tmp_path / "source.mov", seconds=7)
    rows = [
        (text, start + offset * 0.2, start + offset * 0.2 + 0.15)
        for start in (0.5, 4.5)
        for offset, text in enumerate("Amber lantern glows.".split())
    ]
    root = make_project(tmp_path / "project", [source], word_rows=[rows])
    write_edit(root, [time_segment("story", "t1", 0, 7)])
    before = snapshot(root)
    assert main(["edit", "check", str(root), "--json"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert len(checked["repeats"]) == 1
    warning = next(warning for warning in checked["warnings"] if warning.startswith("Repeated:"))
    assert main(["edit", "check", str(root)]) == 0
    assert warning in capsys.readouterr().out
    assert snapshot(root) == before
    assert main(["cut", str(root), "--json"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert warning in rendered["warnings"]
    stored = json.loads((root / "cut/receipt.json").read_bytes())["plan"]
    assert stored["repeats"] == checked["repeats"]


def test_cut_cli_verified_noop_force_and_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = synth.calib_source(tmp_path / "source.mov", seconds=4)
    root = make_project(tmp_path / "project", [source])
    write_edit(root, [time_segment("story", "t1", 0.3, 2.7)])
    assert main(["cut", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert main(["cut", str(root)]) == 0
    assert "Cut unchanged." in capsys.readouterr().out
    monkeypatch.setattr(cut_stage, "verify", lambda **kwargs: {"status": "fail", "findings": []})
    assert main(["cut", str(root), "--force", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert main(["cut", str(root), "--force"]) == 1
    assert "failed: verify (expected pass, actual fail)" in capsys.readouterr().out


def test_cut_cli_prints_contract_values(tmp_path: Path, capsys: pytest.CaptureFixture,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cut_stage, "cut", lambda *_args, **_kwargs: {
        "status": "failed", "noop": False, "out": "cut/cut.mp4", "warnings": [],
        "asserts": [{"id": "picture_contract", "pass": False,
                     "expected": {"width": 1080}, "actual": {"width": 1}}],
    })
    assert main(["cut", str(tmp_path)]) == 1
    assert 'failed: picture_contract (expected {"width":1080}, actual {"width":1})' \
        in capsys.readouterr().out


@pytest.mark.parametrize("command", [["cut"], ["edit", "check"]])
def test_cli_missing_input_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture, command: list[str]
) -> None:
    assert main([*command, str(tmp_path), "--json"]) == 2
    assert "error" in json.loads(capsys.readouterr().err)


def test_invalid_edit_reports_segment_and_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = synth.calib_source(tmp_path / "source.mov", seconds=4)
    root = make_project(tmp_path / "project", [source])
    write_edit(
        root, [{"id": "story", "take": "t1", "from": {"word": "t1:99"}, "to": {"word": "t1:2"}}]
    )
    assert main(["edit", "check", str(root), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "story" in error["message"]
    assert error["suggested_action"]
    write_json(root / "edit.json", {"schema": "bad"})
    assert main(["edit", "check", str(root)]) == 2
    assert "fix:" in capsys.readouterr().err


@pytest.mark.parametrize("authored", [None, 0, 4])
def test_edit_check_reports_effective_end_hold(
    tmp_path: Path, capsys: pytest.CaptureFixture, authored: float | None,
) -> None:
    source = synth.calib_source(tmp_path / "source.mov", seconds=4)
    root = make_project(tmp_path / "project", [source])
    project = json.loads((root / "project.json").read_bytes())
    project["style"] = "style.json"
    write_json(root / "project.json", project)
    write_json(root / "style.json", {
        "schema_version": 1, "name": "fixture", "internal": {"end_hold_s": 1.5},
    })
    write_edit(root, [time_segment("story", "t1", 0.3, 2.7)], end_hold_s=authored)
    before = snapshot(root)
    assert main(["edit", "check", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    value = 1.5 if authored is None else authored
    assert result["end_hold_s"] == value
    origin = "style.internal.end_hold_s" if authored is None else "edit.end_hold_s"
    assert result["end_hold_source"] == origin
    assert not any("end hold" in warning.lower() for warning in result["warnings"])
    assert main(["edit", "check", str(root)]) == 0
    text = capsys.readouterr().out
    lines = text.splitlines()
    total_index = next(index for index, line in enumerate(lines) if line.startswith("Total:"))
    assert lines[total_index + 1] == (
        f"End hold: {value:g} s from {origin}, applied at assembly"
    )
    assert not any(line.startswith("warning:") and "end hold" in line.lower() for line in lines)
    assert snapshot(root) == before
