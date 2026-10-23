"""Synthetic music stems, exact silent holds, receipts and narrow invalidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
from test_timeline import write

from yap_studio import audio, cli, ffmpeg, receipt
from yap_studio.checks.structure import _loudness
from yap_studio.errors import OutputPathError
from yap_studio.project.lock import ConflictError
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import (
    ProjectContext,
    cut_inputs,
    ingest_inputs,
    music_inputs,
    status,
)
from yap_studio.render.beats import BeatsError
from yap_studio.stages import music as stage
from yap_studio.stages.music import MusicError, music
from yap_studio.style import ResolvedStyle


@pytest.fixture(autouse=True)
def canned_beats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep music tests offline with a synthetic source-time beat grid."""
    def detect(audio: Path, *, duration_s: float) -> dict[str, Any]:
        assert audio.is_file() and duration_s > 0
        return {"bpm": 120.0, "beats": [
            {"time_s": index / 2 + 0.001, "strength": 0.8}
            for index in range(int(duration_s * 2))
        ]}

    monkeypatch.setattr(stage, "detect_beats", detect)


def make_project(root: Path, *, frames: int = 300, hold_s: float = 1.5) -> Path:
    """Publish synthetic source words and a cut frame plan through real journals."""
    source = synth.calib_source(root / "source.mov", seconds=frames / 30, size="160x90")
    write(root / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample", "style": "style.json",
        "asr": {"engine": "none"}, "takes": [{"id": "t1", "path": source.name}],
        "music": {"mode": "file", "file": "track.wav"},
    })
    write(root / "style.json", {"schema_version": 1, "name": "Sample style"})
    inputs = ingest_inputs(ProjectContext(root))
    words = [{"id": "t1:0", "text": "Sample", "start": 0, "end": frames / 30}]
    staged = write(root / "staged-words.json", {
        "schema": "yap-studio/words@1", "take": "t1", "take_sha256": inputs["take:t1"],
        "time_base": "source", "audio_offset_s": 0, "engine": "none", "model": None,
        "transcript_sha256": "1" * 64, "words": words,
    })
    with StageRun(root, "ingest") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "ingest/t1/words.json", staged)
        run.finish()
    hashes = {"t1": receipt.sha256_file(root / "ingest/t1/words.json")}
    write(root / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": hashes, "end_hold_s": hold_s,
        "segments": [{"id": "s1", "take": "t1", "from": {"word": "t1:0"},
                      "to": {"word": "t1:0"}}],
    })
    publish_cut(root, frames)
    return root


def publish_cut(root: Path, frames: int) -> None:
    """Replace the cut's word map and frame plan after an authored length change."""
    words = json.loads((root / "ingest/t1/words.json").read_bytes())["words"]
    staged = write(root / "staged-cut-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output",
        "words_sha256": {"t1": receipt.sha256_file(root / "ingest/t1/words.json")},
        "words": [{**word, "segment": "s1"} for word in words],
    })
    inputs = cut_inputs(ProjectContext(root))
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "cut/cut-words.json", staged)
        run.finish(extra={"plan": {"total_frames": frames}})


def tone(target: Path, *, seconds: float = 20, frequency: float = 220,
         rate: int = 48_000) -> Path:
    """Write numpy tone samples using the synthetic fixture PCM writer."""
    samples = 0.1 * np.sin(np.arange(round(seconds * rate)) * (2 * np.pi * frequency / rate))
    synth._write_wav(target, samples, rate)
    return target


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Provide a ten-second cut and a twenty-second tone track."""
    root = make_project(tmp_path)
    tone(root / "track.wav")
    return root


def set_music(root: Path, **settings: Any) -> None:
    """Change authored music settings without modifying stage-owned outputs."""
    data = json.loads((root / "project.json").read_bytes())
    data["music"] = settings
    write(root / "project.json", data)


def set_beat_snap(root: Path, *, enabled: bool = True) -> None:
    """Toggle source beat analysis without changing the remaining style settings."""
    style = json.loads((root / "style.json").read_bytes())
    style.setdefault("music", {})["beat_snap"] = enabled
    write(root / "style.json", style)


def state(root: Path) -> dict[str, Any]:
    """Return music's public stage status."""
    return next(item for item in status(root)["stages"] if item["name"] == "music")


def test_bed_contract_exact_samples_fade_silent_hold_and_loudness(project: Path) -> None:
    source_hash = receipt.sha256_file(project / "track.wav")
    result = music(project)
    rate, samples = audio.decode(project / "music/bed.wav", mono=False)
    assert rate == 48_000 and samples.shape == (345 * 1600, 2)
    assert result["cut_frames"] == 300 and result["end_hold_frames"] == 45
    assert result["cut_samples"] == 480_000 and result["final_samples"] == len(samples)
    assert np.max(np.abs(samples[480_000:])) < 10 ** (-90 / 20)
    assert samples[479_999].tolist() == [0, 0]
    rms = [np.sqrt(np.mean(samples[first:first + 4800] ** 2))
           for first in (432_000, 451_200, 470_400)]
    assert rms[0] > rms[1] > rms[2] > 0
    measured = ffmpeg.run(
        ["ffmpeg", "-v", "info", "-i", str(project / "music/bed.wav"), "-map", "0:a:0",
         "-af", "ebur128=peak=true", "-f", "null", "-"], timeout_s=60,
    )
    assert _loudness(measured.stderr)[0] == pytest.approx(-16, abs=0.2)
    record = read_receipt(project / "music")
    assert record["status"] == "verified"
    assert record["source"] == {"path": "track.wav", "sha256": source_hash, "samples": 960_000}
    assert record["trim"] == {"start_sample": 0, "end_sample": 480_000,
                              "discarded_samples": 480_000}
    assert record["fade"] == {"curve": "raised_cosine", "start_sample": 432_000,
                              "end_sample": 480_000, "duration_s": 1}
    assert record["normalisation"]["target_lufs"] == -16
    assert record["normalisation"]["target_source"] == "style.internal.music_lufs"
    assert record["inputs"] == music_inputs(ProjectContext(project))
    assert [item["path"] for item in record["outputs"]] == ["music/bed.wav"]
    assert state(project)["state"] == "done"
    assert receipt.sha256_file(project / "track.wav") == source_hash


def test_bed_and_beats_stage_inside_project_work_folder(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_beat_snap(project)
    original = StageRun.replace_output
    staged_files: dict[str, Path] = {}
    work_root = project / ".yap/work"

    def publish(run: StageRun, target: Path, staged: Path) -> None:
        assert staged.is_file()
        assert staged.parent.parent == work_root
        assert staged.parent.name.startswith("music-")
        assert staged.stat().st_dev == target.parent.stat().st_dev
        staged_files[target.name] = staged
        original(run, target, staged)

    monkeypatch.setattr(StageRun, "replace_output", publish)
    assert not music(project)["noop"]
    assert set(staged_files) == {"bed.wav", "beats.json"}
    assert len({staged.parent for staged in staged_files.values()}) == 1
    assert all(not staged.parent.exists() for staged in staged_files.values())
    assert all((project / "music" / name).is_file() for name in staged_files)
    assert state(project)["state"] == "done"


@pytest.mark.parametrize("seconds", [9, 10, 11.499])
def test_short_track_exits_two_with_longer_or_loop_fix(
    project: Path, capsys: pytest.CaptureFixture[str], seconds: float,
) -> None:
    tone(project / "track.wav", seconds=seconds)
    assert cli.main(["music", str(project), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "music_too_short"
    assert "longer track" in error["suggested_action"]
    assert "loop it yourself" in error["suggested_action"]
    assert not (project / "music/bed.wav").exists()
    assert not (project / "music/receipt.json").exists()


@pytest.mark.parametrize("mode", ["none", "auto"])
def test_none_without_track_writes_no_bed_and_reports_done(project: Path, mode: str) -> None:
    set_beat_snap(project)
    set_music(project, mode=mode, file=None, generate=True)
    (project / "track.wav").unlink()
    result = music(project)
    assert result["mode"] == "none" and result["bed"] is None
    assert not (project / "music/bed.wav").exists()
    assert not (project / "music/beats.json").exists()
    record = read_receipt(project / "music")
    assert record["mode"] == "none" and record["bed"] is None and record["outputs"] == []
    assert record["source"] is None and record["normalisation"] is None
    assert state(project)["state"] == "done"
    before = (project / "music/receipt.json").read_bytes()
    assert music(project)["noop"]
    assert (project / "music/receipt.json").read_bytes() == before


def test_auto_uses_absolute_file_without_generation(project: Path) -> None:
    set_music(project, mode="auto", file=str(project / "track.wav"), generate=True)
    assert music(project)["mode"] == "file"
    assert read_receipt(project / "music")["source"]["path"] == str(project / "track.wav")


def test_unchanged_noop_source_edit_rerenders_and_force_rerenders(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    music(project)
    before = (project / "music/receipt.json").read_bytes()
    bed_before = (project / "music/bed.wav").read_bytes()
    with monkeypatch.context() as guarded:
        guarded.setattr(stage, "_decode", lambda source: pytest.fail("noop must not decode"))
        assert music(project)["noop"]
    assert (project / "music/receipt.json").read_bytes() == before
    assert (project / "music/bed.wav").read_bytes() == bed_before
    tone(project / "track.wav", frequency=440)
    assert state(project)["state"] == "stale"
    assert "music:source" in state(project)["reason"]
    assert not music(project)["noop"]
    assert (project / "music/bed.wav").read_bytes() != bed_before
    assert not music(project, force=True)["noop"]


def test_mix_style_and_music_windows_do_not_invalidate_bed(project: Path) -> None:
    set_beat_snap(project)
    music(project)
    before = (project / "music/receipt.json").read_bytes()
    style = json.loads((project / "style.json").read_bytes())
    style["music"] = {"level": 0.3, "duck_ratio": 6, "beat_snap": True,
                      "beat_snap_window_ms": 300}
    write(project / "style.json", style)
    write(project / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1",
        "words_sha256": {"t1": receipt.sha256_file(project / "ingest/t1/words.json")},
        "elements": [{"id": "agent-music", "kind": "music_level", "start": {"time": 1},
                      "end": {"time": 2}, "level": 0.5}],
    })
    assert state(project)["state"] == "done"
    assert music(project)["noop"]
    assert (project / "music/receipt.json").read_bytes() == before


def test_none_retires_owned_bed_and_file_can_return(project: Path) -> None:
    set_beat_snap(project)
    music(project)
    assert (project / "music/beats.json").is_file()
    set_music(project, mode="none")
    assert not music(project)["noop"]
    assert not (project / "music/bed.wav").exists()
    assert not (project / "music/beats.json").exists()
    assert read_receipt(project / "music")["outputs"] == []
    set_music(project, mode="file", file="track.wav")
    assert not music(project)["noop"]
    assert (project / "music/bed.wav").is_file()
    assert (project / "music/beats.json").is_file()


def test_music_does_not_require_unrelated_timeline_validation(project: Path) -> None:
    music(project)
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1"})
    assert music(project)["noop"]
    assert state(project)["state"] == "done"


@pytest.mark.parametrize("boundary", ["before_unlink", "after_unlink"])
def test_interrupted_bed_retirement_can_resume(
    project: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    set_beat_snap(project)
    music(project)
    set_music(project, mode="none")
    with monkeypatch.context() as interrupted:
        if boundary == "before_unlink":
            original = StageRun._journal_output

            def journal(run: StageRun, entry: dict[str, Any]) -> None:
                original(run, entry)
                raise RuntimeError("synthetic interruption")

            interrupted.setattr(StageRun, "_journal_output", journal)
        else:
            original_remove = stage._remove_bed

            def remove(run: StageRun) -> None:
                original_remove(run)
                raise RuntimeError("synthetic interruption")

            interrupted.setattr(stage, "_remove_bed", remove)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            music(project)
    assert (project / "music/receipt.pending.json").exists()
    assert not music(project)["noop"]
    assert not (project / "music/bed.wav").exists()
    assert not (project / "music/receipt.pending.json").exists()
    assert state(project)["state"] == "done"


def test_edit_during_retirement_is_preserved(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    music(project)
    set_music(project, mode="none")
    original = StageRun._journal_output

    def edited(run: StageRun, entry: dict[str, Any]) -> None:
        original(run, entry)
        (project / "music/bed.wav").write_bytes(b"hand edited during retirement")

    monkeypatch.setattr(StageRun, "_journal_output", edited)
    with pytest.raises(ConflictError):
        music(project)
    assert (project / "music/bed.wav").read_bytes() == b"hand edited during retirement"


@pytest.mark.parametrize("mode", ["none", "file"])
def test_modified_bed_is_preserved_even_with_force(project: Path, mode: str) -> None:
    music(project)
    target = project / "music/bed.wav"
    target.write_bytes(b"hand edited output")
    set_music(project, mode=mode, file="track.wav")
    with pytest.raises(OutputPathError):
        music(project, force=True)
    assert target.read_bytes() == b"hand edited output"


def test_unconfigured_generation_and_file_without_path_exit_two(
    project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    for mode, code in [("elevenlabs", "elevenlabs_not_set_up"),
                       ("file", "music_file_missing")]:
        set_music(project, mode=mode, file=None)
        assert cli.main(["music", str(project), "--json"]) == 2
        error = json.loads(capsys.readouterr().err)["error"]
        assert error["code"] == code and error["suggested_action"]
        if mode == "elevenlabs":
            assert "isn't set up" in error["message"]


def test_short_cut_fade_default_directory_and_resampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path, frames=60, hold_s=0.5)
    tone(root / "track.wav", seconds=2.5, rate=44_100)
    monkeypatch.chdir(root)
    assert cli.main(["music", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["final_samples"] == 75 * 1600
    record = read_receipt(root / "music")
    assert record["fade"]["duration_s"] == 0.5
    rate, samples = audio.decode(root / "music/bed.wav", mono=False)
    assert rate == 48_000 and samples.shape == (120_000, 2)
    assert np.max(np.abs(samples[96_000:])) == 0


def test_hold_and_cut_frame_changes_rerender(project: Path) -> None:
    music(project)
    edit = json.loads((project / "edit.json").read_bytes())
    edit["end_hold_s"] = 2
    write(project / "edit.json", edit)
    publish_cut(project, 300)
    assert "end_hold:frames" in state(project)["reason"]
    assert not music(project)["noop"]
    publish_cut(project, 270)
    assert "cut:frames" in state(project)["reason"]
    assert not music(project)["noop"]
    assert audio.decode(project / "music/bed.wav", mono=False)[1].shape == (330 * 1600, 2)


def test_source_snapshot_hash_change_refused(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stage.shutil.copyfile

    def changing_copy(source: Path, target: Path) -> None:
        tone(project / "track.wav", frequency=440)
        original(source, target)

    monkeypatch.setattr(stage.shutil, "copyfile", changing_copy)
    with pytest.raises(MusicError, match="changed while being read"):
        music(project)
    assert not (project / "music/bed.wav").exists()


def test_internal_target_fallback_is_recorded(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ProjectContext.render_style

    def without_target(context: ProjectContext) -> ResolvedStyle:
        resolved = original(context)
        resolved.data["internal"].pop("music_lufs", None)
        return resolved

    monkeypatch.setattr(ProjectContext, "render_style", without_target)
    music(project)
    normalisation = read_receipt(project / "music")["normalisation"]
    assert normalisation["target_lufs"] == -16 and normalisation["target_source"] == "default"


def test_source_edit_during_decode_leaves_old_snapshot_stale(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stage._decode
    source_hash = receipt.sha256_file(project / "track.wav")

    def edited(source: Path) -> np.ndarray:
        if source.name == "source":
            tone(project / "track.wav", frequency=440)
        return original(source)

    monkeypatch.setattr(stage, "_decode", edited)
    music(project)
    assert read_receipt(project / "music")["source"]["sha256"] == source_hash
    assert state(project)["state"] == "stale"
    assert "music:source" in state(project)["reason"]


def test_beats_map_source_time_drop_dry_open_trim_and_hold_and_own_output(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_beat_snap(project)
    source_hash = receipt.sha256_file(project / "track.wav")
    calls = []

    def detect(audio: Path, *, duration_s: float) -> dict[str, Any]:
        calls.append((audio, duration_s))
        return {"bpm": 120.0, "beats": [
            {"time_s": time, "strength": 0.75}
            for time in (0.001, 0.699, 0.7, 0.75, 9.501, 10, 10.5, 19.999)
        ]}

    monkeypatch.setattr(stage, "detect_beats", detect)
    music(project)
    assert calls == [(project / "track.wav", 20.0)]
    target = project / "music/beats.json"
    assert json.loads(target.read_bytes()) == {
        "schema": "yap-studio/beats@1", "bpm": 120.0, "music_sha256": source_hash,
        "beats": [{"time_s": 0.7, "frame": 21, "strength": 0.75},
                  {"time_s": 0.75, "frame": 22, "strength": 0.75},
                  {"time_s": 9.501, "frame": 285, "strength": 0.75}],
    }
    record = read_receipt(project / "music")
    assert record["beats"] == {"status": "verified", "path": "music/beats.json",
                               "bpm": 120.0, "count": 3}
    output = next(item for item in record["outputs"] if item["path"] == "music/beats.json")
    assert output["sha256"] == receipt.sha256_file(target)
    assert output["bytes"] == target.stat().st_size
    assert state(project)["state"] == "done"


def test_dry_open_change_refreshes_hints_without_delaying_beats(project: Path) -> None:
    set_beat_snap(project)
    music(project)
    style = json.loads((project / "style.json").read_bytes())
    style["music"] = {"dry_open_s": 2, "beat_snap": True}
    write(project / "style.json", style)
    assert "style:music:dry_open" in state(project)["reason"]
    assert not music(project)["noop"]
    hints = json.loads((project / "music/beats.json").read_bytes())
    assert hints["beats"][0] == {"time_s": 2.001, "frame": 60, "strength": 0.8}
    assert all(2 <= beat["time_s"] < 10 for beat in hints["beats"])


@pytest.mark.parametrize("existing", [False, True])
def test_detection_failure_is_skipped_and_retires_old_hints(
    project: Path, monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    set_beat_snap(project)
    if existing:
        music(project)
        assert (project / "music/beats.json").exists()

    def failing(audio: Path, *, duration_s: float) -> dict[str, Any]:
        raise BeatsError("synthetic detector failure")

    monkeypatch.setattr(stage, "detect_beats", failing)
    result = music(project, force=True)
    assert not result["noop"] and (project / "music/bed.wav").is_file()
    assert not (project / "music/beats.json").exists()
    record = read_receipt(project / "music")
    assert record["status"] == "verified"
    assert record["beats"] == {"status": "skipped",
                               "reason": "Beat detection failed: synthetic detector failure"}
    assert [item["path"] for item in record["outputs"]] == ["music/bed.wav"]
    assert state(project)["state"] == "done"


@pytest.mark.parametrize("mode", ["none", "file"])
def test_modified_beat_hints_are_preserved_even_with_force(project: Path, mode: str) -> None:
    set_beat_snap(project)
    music(project)
    target = project / "music/beats.json"
    target.write_bytes(b"hand edited beat hints")
    set_music(project, mode=mode, file="track.wav")
    with pytest.raises(OutputPathError):
        music(project, force=True)
    assert target.read_bytes() == b"hand edited beat hints"


def test_source_edit_during_detection_skips_hints_for_the_old_bed(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_beat_snap(project)
    original = stage.detect_beats
    source_hash = receipt.sha256_file(project / "track.wav")

    def changing(audio: Path, *, duration_s: float) -> dict[str, Any]:
        tone(audio, frequency=440)
        return original(audio, duration_s=duration_s)

    monkeypatch.setattr(stage, "detect_beats", changing)
    music(project)
    record = read_receipt(project / "music")
    assert record["source"]["sha256"] == source_hash
    assert record["beats"]["status"] == "skipped"
    assert not (project / "music/beats.json").exists()
    assert state(project)["state"] == "stale"


def test_default_project_never_runs_beat_detection(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(audio: Path, *, duration_s: float) -> dict[str, Any]:
        pytest.fail("the default style must not run beat detection")

    monkeypatch.setattr(stage, "detect_beats", forbidden)
    assert not music(project)["noop"]
    assert music(project)["noop"]
    assert not music(project, force=True)["noop"]
    assert (project / "music/bed.wav").is_file()
    assert not (project / "music/beats.json").exists()
    record = read_receipt(project / "music")
    assert record["status"] == "verified"
    assert [output["path"] for output in record["outputs"]] == ["music/bed.wav"]
    assert "style:music:beat_snap" in record["inputs"]


def test_beat_flag_transitions_rerun_detection_and_journal_hint_retirement(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_detect = stage.detect_beats
    calls = []

    def detect(audio: Path, *, duration_s: float) -> dict[str, Any]:
        calls.append((audio, duration_s))
        return original_detect(audio, duration_s=duration_s)

    monkeypatch.setattr(stage, "detect_beats", detect)
    music(project)
    disabled = read_receipt(project / "music")
    assert calls == [] and not (project / "music/beats.json").exists()
    set_beat_snap(project)
    assert state(project)["state"] == "stale"
    assert "style:music:beat_snap" in state(project)["reason"]
    assert not music(project)["noop"]
    enabled = read_receipt(project / "music")
    assert calls == [(project / "track.wav", 20.0)]
    assert enabled["beats"]["status"] == "verified"
    key = "style:music:beat_snap"
    assert enabled["inputs"][key] != disabled["inputs"][key]
    hints_hash = receipt.sha256_file(project / "music/beats.json")
    original_journal = StageRun._journal_output
    journal_entries = []

    def journal(run: StageRun, entry: dict[str, Any]) -> None:
        journal_entries.append(dict(entry))
        original_journal(run, entry)

    monkeypatch.setattr(StageRun, "_journal_output", journal)
    set_beat_snap(project, enabled=False)
    assert state(project)["state"] == "stale"
    assert "style:music:beat_snap" in state(project)["reason"]
    assert not music(project)["noop"]
    assert calls == [(project / "track.wav", 20.0)]
    assert not (project / "music/beats.json").exists()
    retired = read_receipt(project / "music")
    assert retired["status"] == "verified"
    assert retired["inputs"][key] == disabled["inputs"][key]
    assert [output["path"] for output in retired["outputs"]] == ["music/bed.wav"]
    assert {"path": "music/beats.json", "old_sha256": hints_hash,
            "new_sha256": None} in journal_entries
    assert not (project / "music/receipt.pending.json").exists()
    assert state(project)["state"] == "done" and music(project)["noop"]


def test_renderer_pin_change_refreshes_beat_hints_only_with_beat_snap(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    music(project)
    with monkeypatch.context() as version:
        version.setattr(stage.hyperframes, "HYPERFRAMES_VERSION", "9.9.9")
        assert state(project)["state"] == "done"
    set_beat_snap(project)
    music(project)
    assert "renderer:beats" in read_receipt(project / "music")["inputs"]
    calls = []
    original_detect = stage.detect_beats

    def detect(audio: Path, *, duration_s: float) -> dict[str, Any]:
        calls.append(audio)
        return original_detect(audio, duration_s=duration_s)

    monkeypatch.setattr(stage, "detect_beats", detect)
    monkeypatch.setattr(stage.hyperframes, "HYPERFRAMES_VERSION", "9.9.9")
    assert state(project)["state"] == "stale"
    assert "renderer:beats" in state(project)["reason"]
    assert not music(project)["noop"]
    assert calls == [project / "track.wav"]
