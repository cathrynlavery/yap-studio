"""Command line verification behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from yap_studio.cli import main

ROOT = Path(__file__).parents[1]


def _json_error(stderr: str) -> dict:
    return json.loads(stderr.splitlines()[-1])["error"]


def _edit_list(path: Path, source: Path, spans) -> Path:
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "segments": [{"start": start, "end": end} for start, end in spans],
            }
        )
    )
    return path


def test_clean_with_edit_list_exits_zero(synth_media, tmp_path: Path, capsys) -> None:
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    assert main(["verify", str(synth_media.path("clean")), "--edl", str(edl), "--asr", "none"]) == 0
    captured = capsys.readouterr()
    assert "Yap Studio verify: clean.mp4" in captured.out
    assert "Checking sync" in captured.err


def test_loudness_note_reaches_verify(synth_media, capsys) -> None:
    assert main([
        "verify", "--loudness", "note", str(synth_media.path("calib")),
        "--asr", "none", "--json", "--strict",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    notes = [row for row in report["findings"] if row["check"].startswith("loudness.")]
    assert {row["check"] for row in notes} == {"loudness.integrated", "loudness.true_peak"}
    assert all(row["severity"] == "info" and row["suggested_action"] is None for row in notes)
    assert report["status"] == "pass"


def test_invalid_loudness_value_is_usage_error(capsys) -> None:
    assert main(["verify", "render.mp4", "--loudness", "invalid"]) == 2
    error = capsys.readouterr().err
    assert "usage:" in error
    assert "--loudness: invalid choice" in error


def test_shifted_audio_exits_one_with_sync_failure(synth_media, tmp_path: Path, capsys) -> None:
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    code = main(
        [
            "verify",
            str(synth_media.path("clean_shift_samples")),
            "--edl",
            str(edl),
            "--asr",
            "none",
            "--json",
        ]
    )
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert any(
        row["check"] == "sync.av_offset" and row["severity"] == "fail"
        for row in report["findings"]
    )


def test_detected_click_exits_one_and_explains_skipped_sync(
    synth_media,
    capsys,
) -> None:
    code = main(["verify", str(synth_media.path("perclip_click3")), "--asr", "none"])
    assert code == 1
    text = capsys.readouterr().out
    assert "Click at the join" in text
    assert "7 clear cuts found in the picture" in text
    assert "Not checked:" in text
    assert "Jump cuts inside one take: may be missed without an edit list" in text
    assert "Sync and words against the source: need the edit list" in text


def test_missing_media_is_json_error(synth_media, tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.mp4"
    assert main(["verify", str(missing), "--json", "--asr", "none"]) == 2
    error = _json_error(capsys.readouterr().err)
    assert error["code"] == "invalid_input"


def test_unsupported_fcpxml_ref_clip_exits_two(
    synth_media,
    tmp_path: Path,
    capsys,
) -> None:
    edl = tmp_path / "edit.fcpxml"
    edl.write_text('<fcpxml version="1.10"><library><ref-clip/></library></fcpxml>')
    assert main(
        ["verify", str(synth_media.path("clean")), "--edl", str(edl), "--json", "--asr", "none"]
    ) == 2
    error = _json_error(capsys.readouterr().err)
    assert error["code"] == "unsupported_edit_list"


@pytest.mark.parametrize("alias", ["media", "edl"])
def test_report_cannot_alias_an_input(
    alias: str,
    synth_media,
    tmp_path: Path,
    capsys,
) -> None:
    media = synth_media.path("clean")
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    target = media if alias == "media" else edl
    code = main(
        [
            "verify",
            str(media),
            "--edl",
            str(edl),
            "--report",
            str(target),
            "--json",
            "--asr",
            "none",
        ]
    )
    assert code == 2
    assert _json_error(capsys.readouterr().err)["code"] == "invalid_output_path"


def test_no_outputs_leaves_working_directory_unchanged(
    synth_media,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    work = tmp_path / "empty"
    work.mkdir()
    monkeypatch.chdir(work)
    before = os.listdir(work)
    assert main(["verify", str(synth_media.path("clean")), "--asr", "none"]) == 0
    capsys.readouterr()
    assert os.listdir(work) == before


def test_report_is_valid_json_without_leftover_temp_files(
    synth_media,
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "out.json"
    assert main(
        [
            "verify",
            str(synth_media.path("clean")),
            "--asr",
            "none",
            "--report",
            str(output),
        ]
    ) == 0
    capsys.readouterr()
    assert json.loads(output.read_text())["schema"] == "yap-studio/verify-report@1"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.e2e
def test_console_script_verify(synth_media, tmp_path: Path) -> None:
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("yap-studio")),
            "verify",
            str(synth_media.path("clean")),
            "--edl",
            str(edl),
            "--asr",
            "none",
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0
    assert "Yap Studio verify: clean.mp4" in result.stdout
