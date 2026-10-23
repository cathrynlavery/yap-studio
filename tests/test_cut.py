"""Exact story resolution, source clocks, publication, and real verification."""

from __future__ import annotations

import json
import os
import shutil
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth

from yap_studio import audio, ffmpeg, receipt, transcript, video
from yap_studio.edl import load_edit_list
from yap_studio.errors import DependencyError, ValidationError
from yap_studio.project import status as project_status
from yap_studio.project.anchors import edge_words, edit_needs_edge_engine
from yap_studio.project.model import (
    Anchor,
    Edit,
    ProjectFileError,
    Word,
    Words,
    load_cut_words,
    parse_document,
)
from yap_studio.project.receipts import StageRun, audio_stream_sha256, projection_sha256
from yap_studio.project.status import (
    ProjectContext,
    TakeChangedError,
    cut_inputs,
    ingest_inputs,
    status,
)
from yap_studio.stages import cut as cut_stage
from yap_studio.stages import edges, take_levels
from yap_studio.stages.cut import _edge_backend, cut, edit_check, resolve_cut
from yap_studio.transcript import ScriptedBackend
from yap_studio.transcript.model import Transcript
from yap_studio.transcript.model import Word as HeardWord


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write portable project fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def make_project(
    root: Path,
    takes: list[Path],
    *,
    word_rows: list[list[tuple[str, float, float]]] | None = None,
    ingest_manifest: bool = False,
) -> Path:
    """Publish known source words through a matching ingest stage receipt."""
    root.mkdir(parents=True, exist_ok=True)
    source_dir = root / "takes"
    source_dir.mkdir()
    copied = []
    for index, source in enumerate(takes, 1):
        target = source_dir / f"t{index}{source.suffix}"
        shutil.copyfile(source, target)
        copied.append(target)
    write_json(
        root / "project.json",
        {
            "schema": "yap-studio/project@1",
            "name": "story",
            "style": "preset:clean_minimal",
            "takes": [
                {"id": f"t{index}", "path": f"takes/{source.name}", "crop_x": 0.5}
                for index, source in enumerate(copied, 1)
            ],
        },
    )
    inputs = ingest_inputs(ProjectContext(root))
    with StageRun(root, "ingest") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        records = []
        for index, source in enumerate(copied, 1):
            take_id = f"t{index}"
            info = ffmpeg.probe(source)
            rows = (
                word_rows[index - 1]
                if word_rows is not None
                else [
                    ("bright", 0.5, 0.8),
                    ("morning", 1.2, 1.5),
                    ("light", 1.9, 2.1),
                    ("quiet", 3.4, 3.7),
                    ("garden", 4.3, 4.6),
                ]
            )
            staged = root / f"staged-{take_id}.json"
            write_json(
                staged,
                {
                    "schema": "yap-studio/words@1",
                    "take": take_id,
                    "take_sha256": inputs[f"take:{take_id}"],
                    "time_base": "source",
                    "audio_offset_s": info.audio_offset_s,
                    "engine": "none",
                    "model": None,
                    "transcript_sha256": "1" * 64,
                    "words": [
                        {
                            "id": f"{take_id}:{number}",
                            "text": text,
                            "start": first,
                            "end": last,
                            "conf": 1.0,
                        }
                        for number, (text, first, last) in enumerate(rows)
                    ],
                },
            )
            run.replace_output(root / f"ingest/{take_id}/words.json", staged)
            if ingest_manifest:
                records.append({
                    "id": take_id,
                    "path": f"takes/{source.name}",
                    "sha256": inputs[f"take:{take_id}"],
                    "requested_audio_stream": None,
                    "audio_stream": 0,
                    "audio_offset_s": info.audio_offset_s,
                    "audio_sha256": audio_stream_sha256(source, 0),
                })
        if ingest_manifest:
            staged = root / "staged-ingest.json"
            write_json(staged, {"schema": "yap-studio/ingest@1", "takes": records})
            run.replace_output(root / "ingest/ingest.json", staged)
        run.finish()
    return root


def write_edit(root: Path, segments: list[dict], **fields: Any) -> None:
    """Bind story anchors to the exact source word snapshots."""
    write_json(
        root / "edit.json",
        {
            "schema": "yap-studio/edit@1",
            "words_sha256": {
                take: receipt.sha256_file(root / f"ingest/{take}/words.json")
                for take in {segment["take"] for segment in segments}
            },
            "segments": segments,
            **fields,
        },
    )


def time_segment(identifier: str, take: str, first: float, last: float) -> dict:
    """Create an explicit source-time segment."""
    return {"id": identifier, "take": take, "from": {"time": first}, "to": {"time": last}}


@pytest.fixture(scope="module")
def calibration(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.calib_source(tmp_path_factory.mktemp("cut-source") / "source.mov", seconds=7)


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Include bookkeeping and mtimes in the no-write assertion."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def replace_take(source: Path, *, copy_audio: bool) -> str:
    """Replace a take with a smaller picture and copied or re-encoded audio."""
    stand_in = source.with_name(f"{source.stem}-stand-in{source.suffix}")
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=160:90",
            "-c:v", "libx264", "-c:a", "copy" if copy_audio else "aac",
            *([] if copy_audio else ["-b:a", "64k"]), str(stand_in),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    stand_in.replace(source)
    return receipt.sha256_file(source)


def test_cut_plan_keeps_exact_pauses_and_rounds_only_the_report(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    pause = edges.PauseThreshold(-50.04, "floor", -65.04, None, 5)
    monkeypatch.setattr(edges, "choose_pause_threshold", lambda level, midpoints_s: pause)
    plan = resolve_cut(root)
    assert plan.pauses["t1"] is pause
    assert plan.pauses["t1"].threshold_db == -50.04
    report = json.loads(json.dumps(plan.to_dict()))
    assert report["pause_thresholds"] == {"t1": pause.to_dict()}
    assert report["pause_thresholds"]["t1"]["threshold_db"] == -50.0
    assert "pauses" not in report


def test_two_takes_offset_exact_counts_and_verify(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shifted = synth.shift_audio(calibration, tmp_path / "offset.mov", 100, "start_time")
    root = make_project(tmp_path / "project", [calibration, shifted, calibration],
                        ingest_manifest=True)
    write_edit(
        root, [time_segment("s1", "t1", 0.3, 2.7), time_segment("s2", "t2", 3.1, 5.5)], end_hold_s=4
    )
    plan = resolve_cut(root)
    assert set(plan.pause_thresholds) == {"t1", "t2"}
    assert all(pause["words"] == 5 for pause in plan.pause_thresholds.values())
    resolutions = []
    original_resolve = cut_stage._resolve

    def resolve_once(context: ProjectContext) -> cut_stage.CutPlan:
        result = original_resolve(context)
        resolutions.append(result.to_dict())
        return result

    monkeypatch.setattr(cut_stage, "_resolve", resolve_once)
    result = cut(root)
    assert len(resolutions) == 1
    assert result["status"] == "verified", result
    assert plan.total_frames == 144
    assert plan.total_samples == 230_400
    info = ffmpeg.probe(root / "cut/cut.mp4")
    assert info.video_frames == plan.total_frames
    assert info.fps == 30
    assert info.audio_duration_s * 48_000 == plan.total_samples
    with wave.open(str(root / "cut/voice.wav")) as voice:
        assert voice.getnframes() == plan.total_samples
        assert voice.getframerate() == 48_000
        assert voice.getsampwidth() == 2
        assert voice.getnchannels() == 1
    _, actual = audio.decode(root / "cut/voice.wav")
    stage_receipt = json.loads((root / "cut/receipt.json").read_bytes())
    assert list(stage_receipt["take_levels"]["takes"]) == ["t1", "t2"]
    expected_parts = []
    for piece in plan.pieces:
        _, source = audio.decode(plan.sources[piece.take], sr=48_000)
        first = piece.src_f0 * 1600
        part = source[first : first + piece.n_frames * 1600]
        gain = stage_receipt["take_levels"]["takes"][piece.take]["gain_db"]
        if gain != 0.0:
            part = part * np.float32(10 ** (gain / 20))
        expected_parts.append(part)
    expected, joins = audio.splice(
        np.concatenate(expected_parts), [(0, 115_200), (115_200, 230_400)], 240
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert actual[joins[0] - 1] == actual[joins[0]] == 0
    spans = load_edit_list(root / "cut/spans.json")
    assert set(spans.sources) == {"t1", "t2"}
    assert spans.segments[1].start == Fraction(31, 10)
    report = json.loads((root / "cut/verify.json").read_bytes())
    assert report["summary"]["fail"] == 0, report["findings"]
    assert report["status"].lower() != "fail"
    stage_receipt = json.loads((root / "cut/receipt.json").read_bytes())
    manifest = json.loads((root / "ingest/ingest.json").read_bytes())
    assert list(stage_receipt["takes"]) == ["t1", "t2"]
    assert stage_receipt["takes"] == {
        item["id"]: {key: item[key] for key in
                     ("path", "sha256", "audio_stream", "audio_sha256")}
        for item in manifest["takes"] if item["id"] in plan.sources
    }
    assert stage_receipt["plan"] == json.loads(json.dumps(resolutions[0]))
    assert set(stage_receipt["plan"]["pause_thresholds"]) == {"t1", "t2"}
    for pause in stage_receipt["plan"]["pause_thresholds"].values():
        assert pause["rule"] == "floor"  # The fixture supplies only five words per take.
        assert pause["threshold_db"] == edges.SILENCE_CEILING_DBFS
        assert pause["share"] is None
        assert pause["words"] == 5
    assert result["edges"] == resolutions[0]["edges"]
    assert all(item["pass"] for item in stage_receipt["asserts"])
    assert stage_receipt["voice"] == {"codec": "pcm_s16le", "sample_rate": 48_000, "channels": 1}
    cut_state = next(stage for stage in status(root)["stages"] if stage["name"] == "cut")
    assert cut_state["state"] == "done"
    before = snapshot(root)
    noop = cut(root)
    assert noop["noop"] is True
    assert noop["take_levels"] == stage_receipt["take_levels"]
    assert len(resolutions) == 1
    expected_noop = {
        **stage_receipt["plan"],
        "warnings": [*stage_receipt["plan"]["warnings"],
                     *take_levels.warnings(stage_receipt["take_levels"])],
    }
    assert {key: noop[key] for key in stage_receipt["plan"]} == expected_noop
    assert snapshot(root) == before
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][1]["to"]["time"] = 5.8
    write_json(root / "edit.json", edit)
    cut_state = next(stage for stage in status(root)["stages"] if stage["name"] == "cut")
    assert cut_state["state"] == "stale"
    rerender = cut(root)
    assert len(resolutions) == 2
    assert rerender["status"] == "verified"
    assert rerender["noop"] is False
    assert rerender["total_frames"] == 153
    assert not list((root / ".yap/work").glob("cut-*"))


@pytest.mark.parametrize("record_audio", [True, False], ids=["audio-changed", "legacy-ingest"])
def test_changed_take_refuses_cut_without_writing(
    tmp_path: Path, calibration: Path, record_audio: bool,
) -> None:
    root = make_project(tmp_path / "project", [calibration], ingest_manifest=True)
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    if not record_audio:
        manifest = json.loads((root / "ingest/ingest.json").read_bytes())
        manifest["takes"][0].pop("audio_sha256")
        write_json(root / "ingest/ingest.json", manifest)
    replace_take(root / "takes/t1.mov", copy_audio=not record_audio)
    shutil.rmtree(root / ".yap")
    before = snapshot(root)
    for operation in (edit_check, cut):
        with pytest.raises(TakeChangedError) as caught:
            operation(root)
        error = caught.value
        assert error.message == (
            "Take t1 (takes/t1.mov) changed since ingest: "
            + ("its audio stream differs" if record_audio else
               "no audio-stream hash was recorded, so a stand-in cannot be accepted") + "."
        )
        assert error.error_type == error.code == "take_changed"
        assert "takes/t1.mov" in error.suggested_action
        assert "yap-studio ingest DIR --force" in error.suggested_action
        assert "yap-studio cut DIR" in error.suggested_action
        assert snapshot(root) == before


def test_proxy_keeps_words_and_rerenders_cut(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration], ingest_manifest=True)
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    assert cut(root)["status"] == "verified"
    original = json.loads((root / "ingest/ingest.json").read_bytes())["takes"][0]
    words = (root / "ingest/t1/words.json").read_bytes()
    digest = replace_take(root / "takes/t1.mov", copy_audio=True)
    assert digest != original["sha256"]
    shutil.rmtree(root / ".yap")
    before = snapshot(root)
    checked = edit_check(root)
    warning = "Take t1 renders from a stand-in file with the identical audio stream."
    assert warning in checked["warnings"]
    assert snapshot(root) == before
    result = cut(root)
    assert result["status"] == "verified"
    assert result["noop"] is False
    assert warning in result["warnings"]
    recorded = json.loads((root / "cut/receipt.json").read_bytes())["takes"]
    assert recorded == {"t1": {
        "path": "takes/t1.mov", "sha256": digest, "audio_stream": 0,
        "audio_sha256": original["audio_sha256"],
    }}
    assert (root / "ingest/t1/words.json").read_bytes() == words
    assert cut(root)["noop"] is True


def test_take_touch_keeps_cut_noop(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration], ingest_manifest=True)
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    assert cut(root)["status"] == "verified"
    pauses = json.loads((root / "cut/receipt.json").read_bytes())["plan"]["pause_thresholds"]
    assert set(pauses) == {"t1"}
    assert pauses["t1"]["rule"] == "floor"
    assert pauses["t1"]["threshold_db"] == edges.SILENCE_CEILING_DBFS
    assert pauses["t1"]["words"] == 5
    source = root / "takes/t1.mov"
    identity = source.stat()
    os.utime(source, ns=(identity.st_atime_ns, identity.st_mtime_ns + 1_000_000))
    before = snapshot(root)
    result = cut(root)
    assert result["noop"] is True
    assert result["pause_thresholds"] == pauses
    assert snapshot(root) == before


@pytest.mark.parametrize("matrix", ["bt470bg", "bt709"])
def test_full_range_conforms_and_verifies(
    tmp_path: Path, calibration: Path, matrix: str,
) -> None:
    source = synth.full_range_source(calibration, tmp_path / "full-range.mov", matrix=matrix)
    source_info = synth._stream(source, "video")
    assert (source_info["pix_fmt"], source_info["color_range"], source_info["color_space"]) == (
        "yuvj420p", "pc", matrix,
    )

    expected = synth.patch_colours(source).astype(int)
    np.testing.assert_allclose(expected, [(40, 40, 40), (200, 200, 200), (210, 40, 80)], atol=3)
    root = make_project(tmp_path / "project", [source])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    result = cut(root)
    assert result["status"] == "verified", result
    report = json.loads((root / "cut/receipt.json").read_bytes())
    assert next(row for row in report["asserts"] if row["id"] == "picture_contract")["pass"]
    output = root / "cut/cut.mp4"
    info = synth._stream(output, "video")
    assert tuple(info[key] for key in (
        "pix_fmt", "color_range", "color_space", "color_primaries", "color_transfer",
    )) == ("yuv420p", "tv", "bt709", "bt709", "bt709")
    actual = synth.patch_colours(output).astype(int)
    np.testing.assert_allclose(actual[:2], expected[:2], atol=3)
    # Matrix conversion and 4:2:0 rounding can add a few levels on saturated colours.
    np.testing.assert_allclose(actual[2], expected[2], atol=5)


def test_vfr_conforms_and_verifies(tmp_path: Path, synth_media) -> None:
    source = synth_media.cached("vfr", synth.vfr_source, suffix=".mov")
    root = make_project(tmp_path / "project", [source])
    write_edit(root, [
        time_segment("s1", "t1", 16.3, 18.7),
        time_segment("s2", "t1", 20.1, 22.5),
        time_segment("s3", "t1", 25.3, 27.7),
    ])
    plan = resolve_cut(root)
    result = cut(root)
    report = json.loads((root / "cut/verify.json").read_bytes())
    frame_findings = [item for item in report["findings"] if item["check"] == "join.frames"]
    assert frame_findings == [], frame_findings
    assert result["status"] == "verified", result
    assert not any(item["check"] == "sync.av_offset" for item in report["findings"])
    assert len(report["segments"]) == 3
    assert all(
        row["measured"] and row["av_offset_ms"] is not None
        and abs(row["av_offset_ms"]) <= 1000 / 30
        for row in report["segments"]
    )
    assert plan.total_frames == 216
    assert plan.total_samples == 345_600
    info = ffmpeg.probe(root / "cut/cut.mp4")
    assert info.video_frames == plan.total_frames
    assert info.fps == 30
    assert info.audio_duration_s * 48_000 == plan.total_samples
    with wave.open(str(root / "cut/voice.wav")) as voice:
        assert voice.getnframes() == plan.total_samples


@pytest.mark.parametrize("change", ["language", "engine", "model"])
def test_cut_rerenders_after_speech_settings_change(
    tmp_path: Path, calibration: Path, change: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test changes unavailable settings to exercise stage invalidation only.
    # The real-ASR coverage belongs to test_known_speech_spans_cut_verifies_with_project_asr.
    real_verify = cut_stage.verify
    monkeypatch.setattr(
        cut_stage, "verify", lambda **kwargs: real_verify(**{**kwargs, "asr": "none"})
    )
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:0"}, "to": {"word": "t1:2"},
    }])
    assert cut(root)["status"] == "verified"
    assert cut(root)["noop"] is True
    data = json.loads((root / "project.json").read_bytes())
    if change == "language":
        data["language"] = "es"
    else:
        data["asr"] = {"engine": "auto", "model": None}
        data["asr"][change] = "faster-whisper" if change == "engine" else "large-v3"
    write_json(root / "project.json", data)
    result = cut(root)
    assert result["status"] == "verified"
    assert result["noop"] is False
    assert cut(root)["noop"] is True


@pytest.mark.parametrize("segments", [
    pytest.param([time_segment("s1", "t1", 0.3, 2)], id="time-only"),
    pytest.param([{
        "id": "s1", "take": "t1", "from": {"word": "t1:0", "offset_s": 0.1},
        "to": {"word": "t1:2", "edge": "start"},
    }], id="word-overrides"),
    pytest.param([
        {"id": "s1", "take": "t1", "from": {"time": 0.3}, "to": {"word": "t1:0"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"time": 2}},
    ], id="continuous-word-join"),
])
def test_cut_without_word_confirmation_ignores_engine_changes(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch, segments: list[dict],
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, segments)
    identities = []

    def identity(context: ProjectContext) -> dict[str, str | None]:
        identities.append(context)
        return {"engine": "parakeet-onnx", "model": "default-model"}

    monkeypatch.setattr(project_status, "_edge_engine_identity", identity)
    inputs = cut_inputs(ProjectContext(root))
    assert inputs["asr:resolved"] == projection_sha256({"engine": "not-needed"})
    assert cut(root)["status"] == "verified"
    before = snapshot(root / "cut")
    for resolved in ({"engine": "faster-whisper", "model": "small.en"},
                     {"engine": "none", "model": None}):
        monkeypatch.setattr(project_status, "_edge_engine_identity",
                            lambda context, resolved=resolved: resolved)
        assert cut_inputs(ProjectContext(root)) == inputs
        stages = {stage["name"]: stage for stage in status(root)["stages"]}
        assert stages["cut"]["state"] == "done"
    assert snapshot(root / "cut") == before
    assert not identities


def test_silent_tail_word_survives_cut_and_caption_projection(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("wanted", 0.5, 1.5), ("one.", 1.8, 3.8),
    ]])
    write_edit(root, [time_segment("s1", "t1", 0, 1.75)])
    sound = np.zeros(7 * edges.RATE, dtype=np.float32)
    sound[round(0.5 * edges.RATE):round(1.65 * edges.RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    monkeypatch.setattr(edges, "load_take", lambda *args, **kwargs: take)
    monkeypatch.setattr(cut_stage, "_edge_backend", lambda context: ScriptedBackend(
        lambda _: Transcript([HeardWord("wanted", 0.1, 0.95),
                              HeardWord("one.", 1.0, 1.22)]),
    ))

    plan = resolve_cut(root)
    assert "t1:1" in plan.pieces[0].words
    assert "t1:1" in plan.segments[0]["words"]
    assert [row["word"] for row in plan.word_corrections] == ["t1:0", "t1:1"]
    assert any(word["id"] == "t1:1" for word in cut_stage._cut_words(plan)["words"])
    assert cut(root)["status"] == "verified"
    stored = json.loads((root / "cut/receipt.json").read_bytes())["plan"]
    assert [row["word"] for row in stored["word_corrections"]] == ["t1:0", "t1:1"]


def test_silent_tail_cut_identity_matches_runtime_candidate_threshold(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("wanted", 0.5, 0.9), ("one.", 1.1, 2.1),
    ]])
    write_edit(root, [time_segment("s1", "t1", 0, 2)])
    monkeypatch.setattr(project_status, "_edge_engine_identity",
                        lambda context: {"engine": "scripted", "model": "fixture"})
    identity = cut_inputs(ProjectContext(root))["asr:resolved"]
    assert identity == projection_sha256({"engine": "scripted", "model": "fixture"})


@pytest.mark.parametrize(("segments", "word_rows", "needs_engine"), [
    pytest.param([time_segment("s1", "t1", 0.3, 2)], None, False, id="time-only"),
    pytest.param([{
        "id": "s1", "take": "t1", "from": {"word": "t1:0"}, "to": {"time": 2},
    }], None, True, id="word-start"),
    pytest.param([{
        "id": "s1", "take": "t1", "from": {"time": 0.3}, "to": {"word": "t1:2"},
    }], None, True, id="word-end"),
    *[
        pytest.param([{
            "id": "s1", "take": "t1", "from": {"time": 0.3}, "to": {"time": 2},
            side: {"word": "t1:1", **override},
        }], None, False, id=f"{side}-{name}")
        for side, name, override in (
            ("from", "offset", {"offset_s": 0.1}), ("to", "offset", {"offset_s": -0.1}),
            ("from", "edge", {"edge": "end"}), ("to", "edge", {"edge": "start"}),
        )
    ],
    pytest.param([
        {"id": "s1", "take": "t1", "from": {"time": 0.3}, "to": {"word": "t1:0"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"time": 2}},
    ], None, False, id="continuous"),
    pytest.param([
        {"id": "s1", "take": "t1", "from": {"time": 0.3}, "to": {"word": "t1:0"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"word": "t1:1"}},
        {"id": "s3", "take": "t1", "from": {"word": "t1:2"}, "to": {"time": 2.7}},
    ], None, False, id="continuous-chain"),
    pytest.param([
        {"id": "s1", "take": "t1", "from": {"time": 0.3},
         "to": {"word": "t1:0", "offset_s": 0.1}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"time": 2}},
    ], None, True, id="override-breaks-join"),
    pytest.param([
        {"id": "s1", "take": "t1", "from": {"time": 0.3}, "to": {"word": "t1:0"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"time": 3}},
    ], [[("one", 0.5, 0.8), ("two", 1.8, 2.2)]], True, id="one-second-gap"),
    pytest.param([
        {"id": "s1", "take": "t1", "from": {"time": 1.3}, "to": {"word": "t1:0"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"time": 2.5}},
    ], [[("one", 0.5, 1.5), ("two", 1.2, 2)]], True, id="join-outside-segment"),
    pytest.param([{
        "id": "s1", "take": "t1", "from": {"word": "t1:0"}, "to": {"time": 2},
    }], [[("...", 0.5, 0.8), ("welcome", 1.2, 1.5)]], False, id="empty-expected-words"),
])
def test_edge_engine_predicate_agrees_with_settlement(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
    segments: list[dict], word_rows: list[list[tuple[str, float, float]]] | None,
    needs_engine: bool,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=word_rows)
    write_edit(root, segments)
    context = ProjectContext(root)
    edit = parse_document("edit", context.json("edit.json"))
    words = parse_document("words", context.json("ingest/t1/words.json"))
    assert isinstance(edit, Edit) and isinstance(words, Words)
    assert edit_needs_edge_engine(edit.segments, {"t1": words}) is needs_engine
    calls = []

    def backend(context: ProjectContext):
        calls.append(context)
        return None

    original_place = edges.place

    def place(requests, takes, backend, *, language=None):
        assert bool(any(not request.exact and request.expected for request in requests)) == (
            needs_engine
        )
        return original_place(requests, takes, backend, language=language)

    monkeypatch.setattr(cut_stage, "_edge_backend", backend)
    monkeypatch.setattr(edges, "place", place)
    resolve_cut(root)
    assert len(calls) == int(needs_engine)


def test_time_anchors_are_exact_and_flagged_in_sound(tmp_path: Path, calibration: Path) -> None:
    root = make_project(
        tmp_path / "project",
        [calibration],
        word_rows=[
            [("before", 0.8, 1.2), ("no", 1.2, 1.4), ("after", 1.4, 1.8)],
        ],
    )
    write_edit(root, [time_segment("s1", "t1", 0, 0.5), time_segment("s2", "t1", 1.25, 1.35)])
    before = snapshot(root)
    result = edit_check(root)
    assert result["pieces"][1]["src_t0"] == "19/15"
    assert result["pieces"][1]["src_t1"] == "41/30"
    # The calibration tone never falls silent, so both exact edges ask for a listen.
    assert result["listen_joins"] == [{"frame": 15, "time_s": 0.5, "left": "s1", "right": "s2"}]
    assert {row["how"] for row in result["edges"]} == {"time"}
    assert set(result["pause_thresholds"]) == {"t1"}
    assert result["pause_thresholds"]["t1"]["rule"] == "floor"
    assert result["pause_thresholds"]["t1"]["threshold_db"] == edges.SILENCE_CEILING_DBFS
    assert result["pause_thresholds"]["t1"]["words"] == 3
    assert "holds a word" in result["pieces"][1]["notes"][0]
    assert snapshot(root) == before


def test_word_edges_use_full_transcript_neighbours(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(
        root,
        [
            {
                "id": "s1",
                "take": "t1",
                "from": {"word": "t1:1"},
                "to": {"word": "t1:1"},
                "text": "MORNING!",
            }
        ],
    )
    plan = resolve_cut(root)
    assert plan.pieces[0].src_f0 == 32  # 1.2 - min(0.4/2, 0.12), snapped.
    assert plan.pieces[0].src_f1 == 51  # 1.5 + min(0.4/2, 0.20), snapped.
    assert plan.text_mismatches == ()


def test_snapped_touching_edge_excludes_next_word(tmp_path: Path, calibration: Path) -> None:
    root = make_project(
        tmp_path / "project",
        [calibration],
        word_rows=[[("watch", 3.76, 4.16), ("something.", 4.16, 4.56)]],
    )
    write_edit(
        root,
        [
            {
                "id": "s3",
                "take": "t1",
                "from": {"word": "t1:0"},
                "to": {"word": "t1:0"},
                "text": "watch",
            },
            time_segment("tail", "t1", 5, 6),
        ],
    )
    plan = resolve_cut(root)
    piece = plan.pieces[0]
    assert piece.src_f1 == 125
    assert piece.src_t1 - Fraction("4.16") == Fraction(1, 150)
    assert piece.words == ("t1:0",)
    assert piece.listen_out
    assert plan.listen_joins[0]["left"] == "s3"
    assert plan.text_mismatches == ()
    assert not any("transcript is" in warning for warning in plan.warnings)
    checked = edit_check(root)
    assert checked["pieces"][0]["words"] == ("t1:0",)
    assert checked["segments"][0]["text"] == "watch"
    assert checked["segments"][0]["words"] == ["t1:0"]
    result = cut(root)
    assert result["status"] == "verified", result
    assert result["text_mismatches"] == []
    assert not any("transcript is" in warning for warning in result["warnings"])
    words = load_cut_words(root / "cut/cut-words.json")
    assert [(word.id, word.segment) for word in words.words] == [("t1:0", "s3")]
    assert words.words[0].start == pytest.approx(float(Fraction("3.76") - piece.src_t0))
    assert words.words[0].end == pytest.approx(float(Fraction("4.16") - piece.src_t0))
    assert words.words[0].end <= piece.n_frames / 30


def test_midpoint_membership_controls_short_pieces_and_clipped_words(
    tmp_path: Path, calibration: Path
) -> None:
    root = make_project(
        tmp_path / "project",
        [calibration],
        word_rows=[[("watch", 0.1, 0.7), ("something.", 0.7, 1.1)]],
    )
    write_edit(root, [time_segment("s1", "t1", 0, 1.4)])
    write_json(
        root / "removals.json",
        {
            "schema": "yap-studio/removals@1",
            "removals": [
                {
                    "id": "r1",
                    "take": "t1",
                    "start": 0.2,
                    "end": 0.4,
                    "kind": "manual",
                    "status": "applied",
                }
            ],
        },
    )
    plan = resolve_cut(root)
    assert len(plan.pieces) == 2
    assert plan.pieces[0].src_t0 == 0
    assert plan.pieces[1].src_t0 == Fraction("0.4")
    assert "short piece kept: no contiguous neighbour" in plan.pieces[0].notes
    assert cut_stage._word_ids(plan.source_words["t1"], Fraction("0.4"), Fraction("0.9")) == (
        "t1:0",
    )
    words = cut_stage._cut_words(plan)["words"]
    assert [word["id"] for word in words] == ["t1:0", "t1:1"]
    assert words[0]["start"] == pytest.approx(0.2)
    assert words[0]["end"] == pytest.approx(0.5)


def test_silence_trim_keeps_stretched_word_and_ends_its_caption_at_voice(tmp_path: Path) -> None:
    source = synth.voiced_source(
        tmp_path / "source.mov", [(0.5, 0.8), (2.5, 2.8)], seconds=3.5,
    )
    root = make_project(tmp_path / "project", [source], word_rows=[[
        ("tank.", 0.5, 2.2), ("next", 2.5, 2.8),
    ]])
    write_edit(root, [time_segment("s1", "t1", 0, 3.5)])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1",
        "removals": [{
            "id": "r1", "take": "t1", "start": 0.9, "end": 2.2,
            "kind": "manual", "status": "applied",
        }],
    })

    plan = resolve_cut(root)
    assert plan.pieces[0].words == ("t1:0",)
    assert plan.source_words["t1"].words[0].end == pytest.approx(0.87)
    assert [word["id"] for word in cut_stage._cut_words(plan)["words"]] == ["t1:0", "t1:1"]
    result = cut(root)
    assert result["status"] == "verified", result
    words = load_cut_words(root / "cut/cut-words.json").words
    assert [word.id for word in words] == ["t1:0", "t1:1"]
    assert words[0].end == pytest.approx(0.87)


def test_settled_word_edges_keep_planned_words_and_report_moves(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("before", 0.9, 1.6),
        ("first", 0.94, 1.1),
        ("middle", 1.8, 2.1),
        ("last", 2.7, 5.0),
        ("after", 2.94, 3.02),
    ]])
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:1"},
        "to": {"word": "t1:3"}, "text": "first middle last",
    }])

    def settled(
        requests: list[edges.EdgeRequest], takes: dict, backend: Any, *, language: str | None,
    ) -> tuple[list[edges.EdgeResult], list[str]]:
        return [edges.EdgeResult(
            request.segment, request.start, Fraction("1.2" if request.start else "3"),
            request.nominal, "silence", False, "first middle" if request.start else "middle last",
        ) for request in requests], []

    monkeypatch.setattr(edges, "place", settled)
    result = cut(root)
    assert result["status"] == "verified", result
    words = load_cut_words(root / "cut/cut-words.json").words
    assert [word.id for word in words] == ["t1:1", "t1:2", "t1:3"]
    assert result["segments"][0]["words"] == [word.id for word in words]
    assert list(result["pieces"][0]["words"]) == [word.id for word in words]
    assert result["text_mismatches"] == []
    piece = result["pieces"][0]
    assert all(0 <= word.start <= word.end <= piece["n_frames"] / 30 for word in words)
    assert all(left.end <= right.start for left, right in zip(words, words[1:], strict=False))
    assert words[0].start == 0
    assert words[0].end == pytest.approx(0.16)
    assert words[-1].end == pytest.approx(1.8)
    assert {(row["edge"], row.get("moved_word")) for row in result["edges"]} == {
        ("from", "t1:1"), ("to", "t1:3"),
    }
    stage_receipt = json.loads((root / "cut/receipt.json").read_bytes())
    assert stage_receipt["plan"]["edges"] == result["edges"]


@pytest.mark.parametrize("side", ["from", "to"])
def test_wholly_outside_edge_word_borrows_one_frame(
    tmp_path: Path, calibration: Path, side: str,
) -> None:
    rows = (
        [("first", 0.94, 1.1), ("next", 1.2, 1.5)] if side == "from"
        else [("previous", 1.5, 2), ("last", 2.1, 2.3)]
    )
    root = make_project(tmp_path / "project", [calibration], word_rows=[rows])
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:0", "offset_s": 0.26},
        "to": {"time": 2}, "text": "first next",
    }] if side == "from" else [{
        "id": "s1", "take": "t1", "from": {"time": 1},
        "to": {"word": "t1:1", "offset_s": -0.3}, "text": "previous last",
    }])
    plan = resolve_cut(root)
    words = cut_stage._cut_words(plan)["words"]
    assert [word["id"] for word in words] == ["t1:0", "t1:1"]
    edge_word = words[0 if side == "from" else -1]
    assert edge_word["end"] - edge_word["start"] == pytest.approx(1 / 30)
    assert 0 <= words[0]["start"] <= words[0]["end"] <= words[1]["start"] <= words[1]["end"]
    assert words[1]["end"] <= plan.total_frames / 30
    if side == "from":
        assert words[0]["start"] == 0
        assert words[1]["start"] == pytest.approx(1 / 30)
        assert words[1]["end"] == pytest.approx(0.3)
    else:
        assert words[0]["start"] == pytest.approx(0.5)
        assert words[0]["end"] == pytest.approx(1 - 1 / 30)
        assert words[1]["end"] == 1
    assert plan.text_mismatches == ()
    row = next(row for row in plan.edges if row["edge"] == side)
    assert row["moved_word"] == edge_word["id"]
    assert row["moved_words"] == ["t1:0", "t1:1"]


@pytest.mark.parametrize("removal", [(2.0, 2.3), (2.09, 2.11)])
def test_word_anchored_removals_keep_inner_midpoint_membership(
    tmp_path: Path, calibration: Path, removal: tuple[float, float],
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("first", 0.94, 1.1), ("clipped", 1.5, 2.1),
        ("removed", 2.08, 2.11), ("last", 2.5, 2.8), ("after", 2.99, 3.01),
    ]])
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:0", "offset_s": 0.26},
        "to": {"word": "t1:3", "offset_s": 0.2}, "text": "first clipped last",
    }])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "gap", "take": "t1", "start": removal[0], "end": removal[1],
             "kind": "manual", "status": "applied"},
        ],
    })
    plan = resolve_cut(root)
    assert [piece.words for piece in plan.pieces] == [("t1:0", "t1:1"), ("t1:3",)]
    assert plan.segments[0]["words"] == ["t1:0", "t1:1", "t1:3"]
    assert plan.text_mismatches == ()
    words = cut_stage._cut_words(plan)["words"]
    assert [word["id"] for word in words] == ["t1:0", "t1:1", "t1:3"]
    assert words[1]["end"] == pytest.approx(float(plan.pieces[0].src_t1 - Fraction("1.2")))
    for piece in plan.pieces:
        start, end = piece.out_f0 / 30, (piece.out_f0 + piece.n_frames) / 30
        assert all(start <= word["start"] <= word["end"] <= end
                   for word in words if word["id"] in piece.words)


def test_cut_word_clocks_are_clamped_and_nonoverlapping(
    tmp_path: Path, calibration: Path,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("first", 1.1, 1.4), ("middle", 1.35, 1.7), ("last", 2.1, 2.3),
    ]])
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:0", "offset_s": 0.1},
        "to": {"word": "t1:2", "offset_s": -0.3}, "text": "first middle last",
    }])
    plan = resolve_cut(root)
    words = cut_stage._cut_words(plan)["words"]
    assert [word["id"] for word in words] == ["t1:0", "t1:1", "t1:2"]
    assert [(word["start"], word["end"]) for word in words] == pytest.approx([
        (0, 0.15), (0.15, 0.5), (0.6, 0.8),
    ])
    assert words[0]["end"] <= words[1]["start"] <= words[1]["end"] <= words[2]["start"]
    assert {(row["edge"], row["moved_word"]) for row in plan.edges} == {
        ("from", "t1:0"), ("to", "t1:2"),
    }
    assert plan.text_mismatches == ()


@pytest.mark.parametrize("neighbour_start,neighbour_end,piece_end,interval", [
    (1.19, 1.213, 2, 0.013), (1.209, 1.213, 2, 0.013),
    (1.19, 1.25, 2, 0.05), (1.19, 1.213, 37 / 30, 1 / 30),
])
def test_tight_edge_words_share_available_interval(
    tmp_path: Path, calibration: Path, neighbour_start: float, neighbour_end: float,
    piece_end: float, interval: float,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("first", 0.94, 1.1), ("next", neighbour_start, neighbour_end),
    ]])
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:0", "offset_s": 0.26},
        "to": {"time": piece_end}, "text": "first next",
    }])
    result = cut(root)
    assert result["status"] == "verified", result
    words = load_cut_words(root / "cut/cut-words.json").words
    assert [word.id for word in words] == ["t1:0", "t1:1"]
    assert words[0].start == 0
    assert words[0].end == pytest.approx(interval / 2)
    assert words[1].start == words[0].end
    assert words[1].end == pytest.approx(interval)
    assert all(word.end - word.start == pytest.approx(interval / 2) for word in words)
    assert all(0 <= word.start <= word.end <= result["total_frames"] / 30 for word in words)
    assert result["text_mismatches"] == []
    row = next(row for row in result["edges"] if row["edge"] == "from")
    assert row["moved_words"] == ["t1:0", "t1:1"]


def test_wordless_short_piece_merges_and_spoken_short_piece_stays(
    tmp_path: Path,
    calibration: Path,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[("no", 0.4, 0.6)]])
    write_edit(
        root, [time_segment("silence", "t1", 0, 0.2), time_segment("speech", "t1", 0.2, 1.0)]
    )
    plan = resolve_cut(root)
    assert len(plan.pieces) == 1
    assert plan.total_frames == 30
    assert "merged" in plan.pieces[0].notes[0]
    write_edit(root, [time_segment("s1", "t1", 0.4, 0.6)])
    # Touching neighbours leave exactly 0.2 s around the one retained word.
    raw = json.loads((root / "ingest/t1/words.json").read_bytes())
    raw["words"] = [
        {"id": "t1:0", "text": "before", "start": 0.1, "end": 0.4},
        {"id": "t1:1", "text": "no", "start": 0.4, "end": 0.6},
        {"id": "t1:2", "text": "after", "start": 0.6, "end": 1.0},
    ]
    write_json(root / "ingest/t1/words.json", raw)
    write_edit(root, [time_segment("s1", "t1", 0.4, 0.6)])
    plan = resolve_cut(root)
    assert plan.total_frames == 6
    assert plan.pieces[0].words == ("t1:1",)
    assert "holds a word" in plan.pieces[0].notes[0]


def test_applied_removals_and_repeated_word_occurrences(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7), time_segment("s2", "t1", 0.3, 2.7)])
    write_json(
        root / "removals.json",
        {
            "schema": "yap-studio/removals@1",
            "removals": [
                {
                    "id": "r1",
                    "take": "t1",
                    "start": 1.0,
                    "end": 1.7,
                    "kind": "manual",
                    "status": "applied",
                },
                {
                    "id": "r2",
                    "take": "t1",
                    "start": 1.9,
                    "end": 2.1,
                    "kind": "manual",
                    "status": "proposed",
                },
            ],
        },
    )
    result = cut(root)
    assert result["status"] == "verified", result
    words = load_cut_words(root / "cut/cut-words.json")
    assert [(word.id, word.segment) for word in words.words] == [
        ("t1:0", "s1"),
        ("t1:2", "s1"),
        ("t1:0", "s2"),
        ("t1:2", "s2"),
    ]
    assert words.words[2].start > words.words[0].end
    assert words.words[1].start == pytest.approx(0.9)


def _authored_filler_project(tmp_path: Path, calibration: Path, text: str,
                             *, removal_status: str = "applied", words: tuple[str, ...] = ("t1:3",),
                             ) -> Path:
    """Author both occurrences of like and name only the second for removal."""
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("I", 0.3, 0.5), ("like", 0.6, 0.8), ("really", 0.9, 1.2),
        ("like", 1.3, 1.5), ("it", 1.6, 1.9),
    ]])
    segment = time_segment("s1", "t1", 0.1, 2.3)
    segment["text"] = text
    write_edit(root, [segment])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "r1", "take": "t1", "start": 1.3, "end": 1.5, "kind": "filler",
            "status": removal_status, "words": list(words),
        }],
    })
    return root


@pytest.mark.parametrize("authored", [
    "I like really like it", "I LIKE, really like it!", "I like really it",
], ids=["both-occurrences", "normalized-tokens", "already-trimmed"])
def test_authored_text_ignores_applied_filler_by_word_id(tmp_path: Path, calibration: Path,
                                                       authored: str) -> None:
    root = _authored_filler_project(tmp_path, calibration, authored)
    plan = resolve_cut(root)
    assert plan.segments[0]["text"] == "I like really it"
    assert plan.segments[0]["words"] == ["t1:0", "t1:1", "t1:2", "t1:4"]
    assert plan.text_mismatches == ()
    assert not any("transcript is" in warning for warning in plan.warnings)
    assert json.loads((root / "edit.json").read_bytes())["segments"][0]["text"] == authored


@pytest.mark.parametrize("word_from,word_to", [
    (True, True), (True, False), (False, True), (False, False),
], ids=["word-word", "word-time", "time-word", "time-time"])
def test_authored_text_removal_uses_planned_word_bounds(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
    word_from: bool, word_to: bool,
) -> None:
    """Align filler tokens even when planned edge words lie beyond settled edges."""
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("before", 0.5, 0.8), ("first", 0.94, 1.1), ("middle", 1.5, 1.7),
        ("um", 1.8, 2.0), ("last", 2.7, 2.9), ("after", 3.4, 3.6),
    ]])
    authored_words = [*(["first"] if word_from else []), "middle", "um",
                      *(["last"] if word_to else [])]
    authored = " ".join(authored_words)
    write_edit(root, [{
        "id": "s1", "take": "t1",
        "from": {"word": "t1:1", "offset_s": 0.26} if word_from else {"time": 1.2},
        "to": {"word": "t1:4", "offset_s": -0.3} if word_to else {"time": 2.6},
        "text": authored,
    }])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "r1", "take": "t1", "start": 1.8, "end": 2.0, "kind": "filler",
            "status": "applied", "words": ["t1:3"],
        }],
    })
    expected_texts: list[str] = []
    original_expected = cut_stage._expected_segment_text

    def expected_segment_text(text: str, words: tuple[Word, ...], removed: set[str]) -> str:
        """Record the guard's actual expected text without changing its alignment."""
        expected = original_expected(text, words, removed)
        expected_texts.append(expected)
        return expected

    monkeypatch.setattr(cut_stage, "_expected_segment_text", expected_segment_text)
    result = cut(root)
    expected = " ".join(word for word in authored_words if word != "um")
    assert result["status"] == "verified", result
    assert result["text_mismatches"] == []
    assert expected_texts == [expected]
    assert result["segments"][0]["text"] == expected
    assert not any("transcript is" in warning for warning in result["warnings"])
    assert json.loads((root / "edit.json").read_bytes())["segments"][0]["text"] == authored


def test_authored_text_disagreement_still_warns_after_filler_removal(tmp_path: Path,
                                                                   calibration: Path) -> None:
    authored = "I like rarely like it"
    root = _authored_filler_project(tmp_path, calibration, authored)
    plan = resolve_cut(root)
    assert plan.text_mismatches == ({
        "segment": "s1", "expected": authored, "actual": "I like really it",
    },)
    assert any("rarely" in warning and "transcript is" in warning for warning in plan.warnings)


@pytest.mark.parametrize("removal_status", ["proposed", "rejected"])
def test_unapplied_removal_words_do_not_change_expected_text(tmp_path: Path, calibration: Path,
                                                            removal_status: str) -> None:
    root = _authored_filler_project(tmp_path, calibration, "I like really like it",
                                    removal_status=removal_status)
    plan = resolve_cut(root)
    assert plan.segments[0]["text"] == "I like really like it"
    assert plan.text_mismatches == ()


def test_missing_removal_word_ids_preserves_text_warning(tmp_path: Path, calibration: Path) -> None:
    authored = "I like really like it"
    root = _authored_filler_project(tmp_path, calibration, authored, words=())
    plan = resolve_cut(root)
    assert plan.text_mismatches == ({
        "segment": "s1", "expected": authored, "actual": "I like really it",
    },)


def test_removal_between_touching_words_requires_listening(
    tmp_path: Path, calibration: Path
) -> None:
    root = make_project(
        tmp_path / "project",
        [calibration],
        word_rows=[
            [("before", 0.1, 0.5), ("middle", 0.5, 0.8), ("after", 0.8, 1.2)],
        ],
    )
    write_edit(root, [time_segment("s1", "t1", 0, 2)])
    write_json(
        root / "removals.json",
        {
            "schema": "yap-studio/removals@1",
            "removals": [
                {
                    "id": "r1",
                    "take": "t1",
                    "start": 0.5,
                    "end": 0.8,
                    "kind": "manual",
                    "status": "applied",
                },
            ],
        },
    )
    plan = resolve_cut(root)
    assert plan.pieces[0].listen_out and plan.pieces[1].listen_in
    assert plan.listen_joins[0]["frame"] == 15
    assert plan.pieces[0].words == ("t1:0",)
    assert plan.pieces[1].words == ("t1:2",)


def test_60fps_conforms_and_verifies(tmp_path: Path) -> None:
    source = synth.calib_source(tmp_path / "source.mov", seconds=7, fps="60/1")
    root = make_project(tmp_path / "project", [source])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7), time_segment("s2", "t1", 3.1, 5.5)])
    result = cut(root)
    assert result["status"] == "verified", result
    assert any("60" in warning for warning in result["warnings"])
    assert ffmpeg.probe(root / "cut/cut.mp4").video_frames == 144


@pytest.mark.parametrize("problem", ["missing", "guard", "stale", "backwards", "outside"])
def test_invalid_anchors_and_snapshot_are_actionable(
    tmp_path: Path,
    calibration: Path,
    problem: str,
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    segment = {"id": "story", "take": "t1", "from": {"word": "t1:0"}, "to": {"word": "t1:2"}}
    if problem == "missing":
        segment["from"] = {"word": "t1:99"}
    elif problem == "guard":
        segment["from"] = {"word": "t1:0", "text": "wrong"}
    elif problem == "backwards":
        segment["from"] = {"time": 5}
        segment["to"] = {"time": 2}
    elif problem == "outside":
        segment["to"] = {"time": 8}
    write_edit(root, [segment])
    if problem == "stale":
        data = json.loads((root / "edit.json").read_bytes())
        data["words_sha256"]["t1"] = "0" * 64
        write_json(root / "edit.json", data)
    before = snapshot(root)
    with pytest.raises(ProjectFileError) as caught:
        edit_check(root)
    assert caught.value.suggested_action
    if problem in {"missing", "guard"}:
        assert "story" in caught.value.message
    if problem == "guard":
        assert "wrong" in caught.value.message and "bright" in caught.value.message
    assert snapshot(root) == before


def test_text_mismatch_and_platform_hold_warning(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration])
    segment = time_segment("story", "t1", 0.3, 2.7)
    segment["text"] = "something else"
    write_edit(root, [segment])
    project = json.loads((root / "project.json").read_bytes())
    project["platform"] = "shorts"
    write_json(root / "project.json", project)
    write_edit(
        root,
        [segment]
        + [dict(segment, id=f"s{index}") for index in range(73)]
        + [time_segment("tail", "t1", 3.0, 4.2)],
    )
    plan = resolve_cut(root)
    assert plan.total_frames == 5364
    assert any("maximum" in warning for warning in plan.warnings)
    assert plan.text_mismatches[0]["actual"] == "bright morning light"


def test_cut_loudness_is_information(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    result = cut(root)
    assert result["status"] == "verified", result
    report = json.loads((root / "cut/verify.json").read_bytes())
    notes = [row for row in report["findings"] if row["check"].startswith("loudness.")]
    assert {row["check"] for row in notes} == {"loudness.integrated", "loudness.true_peak"}
    assert all(row["severity"] == "info" and row["suggested_action"] is None for row in notes)
    assert not any(
        row["check"].startswith("loudness.") and row["severity"] == "warn"
        for row in report["structure"]["findings"]
    )
    assert not any(
        "loudness" in warning.lower() or "sound peaks" in warning.lower()
        for warning in result["warnings"]
    )


@pytest.mark.parametrize("platform", ["reels", "none"])
def test_cut_platform_coverage_is_advisory(tmp_path, calibration, monkeypatch, platform):
    import importlib

    root = make_project(tmp_path / "project", [calibration])
    data = json.loads((root / "project.json").read_bytes())
    data["platform"] = platform
    write_json(root / "project.json", data)
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    presets = cut_stage.load_presets()
    presets["reels"]["duration"]["max_s"] = 1
    monkeypatch.setattr(cut_stage, "load_presets", lambda: presets)
    monkeypatch.setattr(importlib.import_module("yap_studio.verify"), "load_presets",
                        lambda: presets)
    result = cut(root)
    assert result["status"] == "verified"
    report = json.loads((root / "cut/verify.json").read_bytes())
    coverage = report["checks"]["platform.duration"]
    assert coverage["ran"] is (platform != "none")
    if platform == "none":
        assert coverage["reason"] == "no platform chosen"
    else:
        assert coverage["status"] == "warn"
        assert sum("maximum" in item or "long" in item for item in result["warnings"]) == 1
        assert all(item["severity"] != "fail" for item in report["findings"]
                   if item["check"].startswith("platform."))


def test_verify_failure_is_retained_and_failed(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    monkeypatch.setattr(cut_stage, "verify", lambda **kwargs: {"status": "FAIL", "findings": []})
    assert cut(root)["status"] == "failed"
    assert (root / "cut/cut.mp4").is_file()
    assert json.loads((root / "cut/verify.json").read_bytes())["status"] == "FAIL"
    assert json.loads((root / "cut/receipt.json").read_bytes())["status"] == "failed"
    checks = {item["id"]: item for item in
              json.loads((root / "cut/receipt.json").read_bytes())["asserts"]}
    for name in ("picture_contract", "audio_contract"):
        assert checks[name]["expected"] and checks[name]["actual"]
        assert checks[name]["expected"].keys() == checks[name]["actual"].keys()
    record = json.loads((root / "cut/receipt.json").read_bytes())
    picture = next(item for item in record["asserts"] if item["id"] == "picture_contract")
    picture["pass"] = False
    picture["actual"]["width"] = 1
    write_json(root / "cut/receipt.json", record)
    cut_state = next(stage for stage in status(root)["stages"] if stage["name"] == "cut")
    assert cut_state["reason"].startswith("failed: picture_contract (expected ")
    assert '"width":1' in cut_state["reason"]
    cut_state = next(stage for stage in status(root)["stages"] if stage["name"] == "cut")
    assert cut_state["state"] == "failed"


def test_check_without_identity_cache_writes_nothing(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7)])
    shutil.rmtree(root / ".yap")
    before = snapshot(root)
    assert edit_check(root)["total_frames"] == 72
    assert snapshot(root) == before
    assert not (root / ".yap").exists()


def test_picture_offset_and_sample_clock_are_verified(tmp_path: Path, calibration: Path) -> None:
    shifted = tmp_path / "picture-offset.mov"
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(calibration),
            "-itsoffset",
            "0.1",
            "-i",
            str(calibration),
            "-map",
            "1:v:0",
            "-map",
            "0:a:0",
            "-c",
            "copy",
            str(shifted),
        ],
        timeout_s=60,
    )
    root = make_project(tmp_path / "project", [shifted])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7), time_segment("s2", "t1", 3.1, 5.5)])
    plan = resolve_cut(root)
    assert plan.media["t1"].video_offset_s == 0.1
    result = cut(root)
    assert result["status"] == "verified", result
    _, source = audio.decode(plan.sources["t1"], sr=48_000)
    _, voice = audio.decode(root / "cut/voice.wav")
    np.testing.assert_allclose(voice[1000:3000], source[20_200:22_200], atol=2e-5)


def test_audio_timestamp_gap_keeps_the_verify_clock(tmp_path: Path) -> None:
    source = synth.audio_gap_render(
        tmp_path / "gap.mp4", tmp_path / "reference.wav", seconds=7, gap_samples=4800
    )
    root = make_project(tmp_path / "project", [source])
    write_edit(root, [time_segment("s1", "t1", 0.3, 2.7), time_segment("s2", "t1", 3.1, 5.5)])
    plan = resolve_cut(root)
    assert audio.timeline(plan.sources["t1"]).fill_samples > 0
    result = cut(root)
    assert result["status"] == "verified", result
    _, reference = audio.decode(plan.sources["t1"], sr=48_000)
    _, actual = audio.decode(root / "cut/voice.wav")
    parts = [reference[piece.src_f0 * 1600 : piece.src_f1 * 1600] for piece in plan.pieces]
    expected, _ = audio.splice(np.concatenate(parts), [(0, 115_200), (115_200, 230_400)], 240)
    np.testing.assert_allclose(actual, expected, atol=2e-5)


def test_half_frame_ties_use_the_engine_grid(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[]])
    write_edit(root, [time_segment("s1", "t1", 0.05, 0.35)])
    plan = resolve_cut(root)
    assert (plan.pieces[0].src_f0, plan.pieces[0].src_f1) == (2, 11)
    assert plan.total_frames == 9
    assert plan.total_samples == 14_400


def test_last_word_past_picture_keeps_the_final_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error",
        "-f", "lavfi", "-i", "color=c=black:s=180x320:r=30:d=3",
        "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=3.023",
        "-vf", "drawbox=c=white:t=fill:enable='eq(n,89)'",
        "-c:v", "libx264", "-c:a", "pcm_s16le", str(source),
    ], timeout_s=120)
    info = ffmpeg.probe(source)
    assert info.video_duration_s is not None
    assert info.audio_duration_s > info.video_duration_s
    duration = info.video_duration_s
    root = make_project(tmp_path / "project", [source], word_rows=[[
        ("Strawberry.", duration - 1, duration + 0.003),
    ]])
    write_edit(root, [{"id": "ending", "take": "t1", "from": {"time": 0},
                      "to": {"word": "t1:0"}}])
    monkeypatch.setattr(cut_stage, "_edge_backend", lambda _: None)
    checked = edit_check(root)
    assert Fraction(checked["pieces"][-1]["src_t1"]) == Fraction(str(duration))
    result = cut(root)
    assert result["status"] == "verified", result
    assert result["pieces"][-1]["src_f1"] == info.video_frames
    assert ffmpeg.probe(root / "cut/cut.mp4").video_frames == info.video_frames
    source_last = video.grey_frames(source, start_frame=info.video_frames - 1, count=1)
    cut_end = video.grey_frames(root / "cut/cut.mp4", start_frame=info.video_frames - 2, count=2)
    assert source_last.mean() > 230
    assert cut_end[0].mean() < 30
    assert cut_end[1].mean() > 230


@pytest.mark.parametrize("last,expected", [(0.249, 1), (0.251, 2)])
def test_short_rule_precedes_frame_snapping(
    tmp_path: Path, calibration: Path, last: float, expected: int
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(
        root, [time_segment("silence", "t1", 0, last), time_segment("speech", "t1", last, 2.7)]
    )
    assert len(resolve_cut(root).pieces) == expected


def _known_speech_cut_project(tmp_path: Path) -> tuple[Path, list[synth.WordSpan]]:
    """Build a word-anchored cut from generated speech and picture."""
    source, spans = synth.speech_source(
        tmp_path / "speech.mov",
        [
            "Bright planets circle slowly over quiet hills on calm winter evenings",
            "Seven green boxes wait beside the open door near our garden",
        ],
    )
    root = make_project(
        tmp_path / "project",
        [source],
        word_rows=[
            [(word.text, word.start, word.end) for word in spans],
        ],
    )
    write_edit(
        root,
        [
            {"id": "s1", "take": "t1", "from": {"word": "t1:0"}, "to": {"word": "t1:10"}},
            {"id": "s2", "take": "t1", "from": {"word": "t1:11"}, "to": {"word": "t1:21"}},
        ],
    )
    return root, spans


def test_known_speech_spans_cut_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_verify = cut_stage.verify
    monkeypatch.setattr(
        cut_stage, "verify", lambda **kwargs: real_verify(**{**kwargs, "asr": "none"})
    )
    root, spans = _known_speech_cut_project(tmp_path)
    result = cut(root)
    assert result["status"] == "verified", result
    assert len(load_cut_words(root / "cut/cut-words.json").words) == len(spans)
    saved = load_edit_list(root / "cut/spans.json")
    assert [(row.first_word, row.last_word) for row in saved.segments] == [
        (spans[0].text, spans[10].text), (spans[11].text, spans[21].text),
    ]


@pytest.mark.asr
def test_known_speech_spans_cut_verifies_with_project_asr(tmp_path: Path) -> None:
    """The cut verifier uses the configured speech engine, like final assembly."""
    if transcript.resolve_backend("auto") is None:
        pytest.skip("no speech engine is installed")
    root, _ = _known_speech_cut_project(tmp_path)
    result = cut(root)
    assert result["status"] == "verified", result
    report = json.loads((root / "cut/verify.json").read_bytes())
    assert report["checks"]["join.relisten"]["ran"] is True


@pytest.mark.parametrize("word_from,word_to", [(True, True), (True, False), (False, True),
                                            (False, False)])
def test_span_intent_stays_on_authored_edges_after_removals(
    tmp_path: Path, calibration: Path, word_from: bool, word_to: bool,
) -> None:
    """Only outer word anchors name intent; removal-created edges have none."""
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [{
        "id": "s1", "take": "t1",
        "from": {"word": "t1:0"} if word_from else {"time": 0.3},
        "to": {"word": "t1:4"} if word_to else {"time": 5.0},
    }])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1",
        "removals": [
            {"id": "r1", "take": "t1", "start": 1.0, "end": 1.7,
             "kind": "manual", "status": "applied"},
            {"id": "r2", "take": "t1", "start": 2.5, "end": 3.0,
             "kind": "manual", "status": "applied"},
        ],
    })
    plan = resolve_cut(root)
    assert len(plan.pieces) == 3
    written = cut_stage._spans(plan, root / "cut")
    assert [(row.get("first_word"), row.get("last_word")) for row in written["segments"]] == [
        ("bright" if word_from else None, None), (None, None),
        (None, "garden" if word_to else None),
    ]
    for row in written["segments"][1:-1]:
        assert "first_word" not in row and "last_word" not in row


@pytest.mark.parametrize("consume_from,consume_to", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("split", [False, True])
def test_removing_an_anchored_edge_does_not_move_its_word_intent(
    tmp_path: Path, calibration: Path, consume_from: bool, consume_to: bool, split: bool,
) -> None:
    """Intent survives only at the settled authored frame, including fractional anchors."""
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [{
        "id": "s1", "take": "t1",
        "from": {"word": "t1:0", "offset_s": 0.01},
        "to": {"word": "t1:4", "offset_s": 0.01},
    }])
    ranges = []
    if consume_from:
        ranges.append((0.3, 1.0))
    if consume_to:
        ranges.append((4.0, 4.8))
    if split:
        ranges.append((2.5, 3.0))
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": f"r{index}", "take": "t1", "start": start, "end": end,
             "kind": "manual", "status": "applied"}
            for index, (start, end) in enumerate(ranges, 1)
        ],
    })

    plan = resolve_cut(root)
    assert len(plan.pieces) == (2 if split else 1)
    rows = cut_stage._spans(plan, root / "cut")["segments"]
    assert rows[0].get("first_word") == (None if consume_from else "bright")
    assert rows[-1].get("last_word") == (None if consume_to else "garden")
    assert all("first_word" not in row for row in rows[1:])
    assert all("last_word" not in row for row in rows[:-1])


_FIRST_TAKE_SENTENCES = [
    "Bright planets circle slowly over quiet hills on calm winter evenings",
    "Gentle rivers carry silver stones through peaceful valleys beneath autumn clouds",
]
_SECOND_TAKE_SENTENCE = "Seven green boxes wait beside the open door near our garden"


@pytest.fixture
def second_take_cut(tmp_path: Path) -> cut_stage.CutPlan:
    """Render a word-anchored second take between two pieces of the first take."""
    # Vertical fixtures keep source and output geometry equal for frame comparisons.
    first, first_spans = synth.speech_source(
        tmp_path / "first.mov", _FIRST_TAKE_SENTENCES, size="180x320"
    )
    second, second_spans = synth.speech_source(
        tmp_path / "second.mov", [_SECOND_TAKE_SENTENCE], picture="testsrc", size="180x320"
    )
    root = make_project(
        tmp_path / "project", [first, second],
        word_rows=[
            [(word.text, word.start, word.end) for word in spans]
            for spans in (first_spans, second_spans)
        ],
    )
    segments = []
    for number, (take, spans, sentence) in enumerate(
        [("t1", first_spans, 0), ("t2", second_spans, 0), ("t1", first_spans, 1)], 1
    ):
        indices = [index for index, word in enumerate(spans) if word.sentence == sentence]
        segments.append({
            "id": f"s{number}", "take": take,
            "from": {"word": f"{take}:{indices[0]}"},
            "to": {"word": f"{take}:{indices[-1]}"},
        })
    write_edit(root, segments)
    plan = resolve_cut(root)
    assert [(piece.segment_id, piece.take) for piece in plan.pieces] == [
        ("s1", "t1"), ("s2", "t2"), ("s3", "t1"),
    ]
    result = cut(root)
    assert result["status"] == "verified", result
    return plan


def test_second_take_segment_renders_its_picture_audio_and_words(
    second_take_cut: cut_stage.CutPlan,
) -> None:
    """Reject first-take substitution independently in picture, PCM, and word ids."""
    plan = second_take_cut
    root = plan.context.project_dir
    middle = plan.pieces[1]
    report = json.loads((root / "cut/verify.json").read_bytes())
    assert report["summary"]["fail"] == 0, report["findings"]
    rendered = video.grey_frames(
        root / "cut/cut.mp4", start_frame=middle.out_f0, count=middle.n_frames,
    )
    pictures = {
        take: video.grey_frames(
            source, start_frame=middle.src_f0, count=middle.n_frames, rate=Fraction(30),
        )
        for take, source in plan.sources.items()
    }
    assert len(rendered) == middle.n_frames
    assert all(len(frames) == middle.n_frames for frames in pictures.values())
    for index in (middle.n_frames // 4, middle.n_frames // 2, 3 * middle.n_frames // 4):
        correct = video.frame_distance(rendered[index], pictures["t2"][index])
        wrong = video.frame_distance(rendered[index], pictures["t1"][index])
        assert correct < 0.1, (index, correct, wrong)
        assert wrong > correct + 0.5, (index, correct, wrong)

    sources = {take: audio.decode(source, sr=48_000)[1] for take, source in plan.sources.items()}
    levels = json.loads((root / "cut/receipt.json").read_bytes())["take_levels"]["takes"]
    for take, sound in sources.items():
        gain = levels[take]["gain_db"]
        if gain != 0.0:
            sources[take] = sound * np.float32(10 ** (gain / 20))
    _, actual = audio.decode(root / "cut/voice.wav", sr=48_000)
    expected_parts = [
        sources[piece.take][piece.src_f0 * 1600 : piece.src_f1 * 1600]
        for piece in plan.pieces
    ]
    expected, _ = audio.splice(
        np.concatenate(expected_parts),
        [(piece.sample_start, piece.sample_end) for piece in plan.pieces], 240,
    )
    assert len(actual) == plan.total_samples
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
    # Compare unfaded samples so the join ramps cannot mask a source substitution.
    window = actual[middle.sample_start + 240 : middle.sample_end - 240]
    first = middle.src_f0 * 1600 + 240
    last = middle.src_f1 * 1600 - 240
    correct_error = float(np.sqrt(np.mean((window - sources["t2"][first:last]) ** 2)))
    wrong_error = float(np.sqrt(np.mean((window - sources["t1"][first:last]) ** 2)))
    assert wrong_error > 0.05, (correct_error, wrong_error)
    assert wrong_error > 100 * correct_error, (correct_error, wrong_error)
    words = load_cut_words(root / "cut/cut-words.json")
    assert [(word.id, word.text) for word in words.words if word.segment == "s2"] == [
        (f"t2:{index}", text) for index, text in enumerate(_SECOND_TAKE_SENTENCE.split())
    ]


@pytest.mark.asr
def test_second_take_segment_transcribes_its_sentence(
    second_take_cut: cut_stage.CutPlan, tmp_path: Path,
) -> None:
    """Listen only to the middle output window and reject the first take's words."""
    if not transcript.ParakeetOnnxBackend.available():
        pytest.skip("onnx-asr is not importable")
    backend = transcript.ParakeetOnnxBackend()
    plan = second_take_cut
    middle = plan.pieces[1]
    window = tmp_path / "middle.wav"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i",
        str(plan.context.project_dir / "cut/voice.wav"), "-map", "0:a:0", "-af",
        f"atrim=start_sample={middle.sample_start}:end_sample={middle.sample_end},"
        "asetpts=PTS-STARTPTS", "-c:a", "pcm_s16le", str(window),
    ], timeout_s=60)
    recovered = backend.transcribe_many([window])[window]
    expected = Counter(transcript.norm_words(_SECOND_TAKE_SENTENCE))
    got = Counter(transcript.norm_words(recovered.text))
    recall = sum((expected & got).values()) / sum(expected.values())
    assert recall >= 0.9, (recovered.text, recall)
    first_take_words = set(transcript.norm_words(" ".join(_FIRST_TAKE_SENTENCES)))
    assert not first_take_words.intersection(got), recovered.text


def test_short_piece_merges_with_contiguous_next_take(tmp_path: Path, calibration: Path) -> None:
    root = make_project(
        tmp_path / "project", [calibration, calibration],
        word_rows=[[("first", 0.4, 0.6)], [("next", 2.4, 2.6)]],
    )
    write_edit(root, [
        time_segment("a", "t1", 0, 1),
        time_segment("short", "t2", 2, 2.2),
        time_segment("next", "t2", 2.2, 3),
    ])
    plan = resolve_cut(root)
    assert plan.total_frames == 60
    assert [(piece.segment_id, piece.src_t0, piece.src_t1) for piece in plan.pieces] == [
        ("a", Fraction(0), Fraction(1)), ("next", Fraction(2), Fraction(3)),
    ]
    assert [(word["id"], word["segment"]) for word in cut_stage._cut_words(plan)["words"]] == [
        ("t1:0", "a"), ("t2:0", "next"),
    ]
    assert [row["id"] for row in plan.segments] == ["a", "short", "next"]


def test_short_piece_kept_across_applied_removal(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[]])
    write_edit(root, [time_segment("story", "t1", 0, 1)])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "gap", "take": "t1", "start": 0.2, "end": 0.4,
             "kind": "manual", "status": "applied"},
        ],
    })
    plan = resolve_cut(root)
    assert plan.total_frames == 24
    assert [(piece.src_t0, piece.src_t1) for piece in plan.pieces] == [
        (Fraction(0), Fraction("0.2")), (Fraction("0.4"), Fraction(1)),
    ]
    assert "short piece kept: no contiguous neighbour" in plan.pieces[0].notes
    piece_warnings = [warning for warning in plan.warnings
                      if "short piece kept" in warning or "holds no word after a trim" in warning]
    assert piece_warnings == ["Segment 'story': short piece kept: no contiguous neighbour."]


@pytest.mark.parametrize("middle_end,worded,expected", [
    (0.9, False, True), (1.2, False, False), (0.9, True, False),
])
def test_trim_warns_for_short_wordless_piece(
    tmp_path: Path, calibration: Path, middle_end: float, worded: bool, expected: bool,
) -> None:
    words = [("first", 0.1, 0.3), ("last", 1.5, 1.8)]
    if worded:
        words.append(("middle", 0.7, 0.8))
    root = make_project(tmp_path / "project", [calibration], word_rows=[words])
    write_edit(root, [time_segment("story", "t1", 0, 2)])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "before", "take": "t1", "start": 0.4, "end": 0.6,
             "kind": "manual", "status": "applied"},
            {"id": "after", "take": "t1", "start": middle_end, "end": 1.3,
             "kind": "manual", "status": "applied"},
        ],
    })
    plan = resolve_cut(root)
    middle = next(piece for piece in plan.pieces if piece.src_t0 == Fraction("0.6"))
    warning = [note for note in middle.notes if "holds no word after a trim" in note]
    assert bool(warning) is expected
    plan_warnings = [note for note in plan.warnings if "holds no word after a trim" in note]
    assert bool(plan_warnings) is expected
    if expected:
        assert warning[0] in plan.warnings
        assert f"in take {middle.take}" in warning[0]


def test_short_piece_prefers_previous_contiguous_neighbour(
    tmp_path: Path, calibration: Path,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[
        [("first", 0.2, 0.4), ("last", 1.6, 1.8)],
    ])
    write_edit(root, [
        time_segment("left", "t1", 0, 1), time_segment("short", "t1", 1, 1.2),
        time_segment("right", "t1", 1.2, 2),
    ])
    plan = resolve_cut(root)
    assert [(piece.segment_id, piece.src_t0, piece.src_t1) for piece in plan.pieces] == [
        ("left", Fraction(0), Fraction("1.2")), ("right", Fraction("1.2"), Fraction(2)),
    ]
    assert [(word["id"], word["segment"]) for word in cut_stage._cut_words(plan)["words"]] == [
        ("t1:0", "left"), ("t1:1", "right"),
    ]


@pytest.mark.parametrize("authored", [None, 0, 4])
def test_platform_warning_uses_effective_end_hold(
    tmp_path: Path, calibration: Path, authored: float | None,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[]])
    project = json.loads((root / "project.json").read_bytes())
    project.update(platform="shorts", style="style.json")
    write_json(root / "project.json", project)
    write_json(root / "style.json", {
        "schema_version": 1, "name": "fixture", "internal": {"end_hold_s": 1.5},
    })
    write_edit(root, [time_segment(f"s{index}", "t1", 0, 1) for index in range(178)],
               end_hold_s=authored)
    result = edit_check(root)
    assert result["end_hold_s"] == (1.5 if authored is None else authored)
    assert result["end_hold_source"] == (
        "style.internal.end_hold_s" if authored is None else "edit.end_hold_s"
    )
    assert any("maximum" in warning for warning in result["warnings"]) == (authored == 4)
    assert not any("Assembly will apply" in warning for warning in result["warnings"])
    assert result["total_frames"] == 178 * 30


def test_selected_speech_stream_cuts_and_verifies(tmp_path: Path) -> None:
    speech, words = synth.speech_source(tmp_path / "speech.mov", [
        "Bright planets circle slowly over quiet hills on calm winter evenings",
        "Seven green boxes wait beside the open door near our garden",
    ])
    source = tmp_path / "streams.mov"
    duration = ffmpeg.probe(speech).duration_s
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(speech), "-f", "lavfi", "-i",
        f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0",
        "-c:v", "copy", "-c:a", "pcm_s16le", str(source),
    ], timeout_s=120)
    root = make_project(tmp_path / "project", [source], word_rows=[
        [(word.text, word.start, word.end) for word in words],
    ])
    project = json.loads((root / "project.json").read_bytes())
    project["takes"][0]["audio_stream"] = 1
    write_json(root / "project.json", project)
    write_edit(root, [
        {"id": "s1", "take": "t1", "from": {"word": "t1:0"}, "to": {"word": "t1:10"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:11"}, "to": {"word": "t1:21"}},
    ])
    result = cut(root)
    assert result["status"] == "verified", result
    spans = json.loads((root / "cut/spans.json").read_bytes())
    assert spans["sources"]["t1"] == {"path": "../takes/t1.mov", "audio_stream": 1}
    report = json.loads((root / "cut/verify.json").read_bytes())
    assert all(row["audio_landing_s"] is not None for row in report["segments"])
    assert all(row["audio_vs_plan_ms"] == pytest.approx(0, abs=2) for row in report["segments"])
    assert report["checks"]["join.words"]["status"] == "pass"


def test_neighbouring_word_segments_join_only_when_close(
    tmp_path: Path, calibration: Path
) -> None:
    root = make_project(
        tmp_path / "project",
        [calibration],
        word_rows=[[("one", 0.5, 0.8), ("two", 0.9, 1.2), ("three", 4.0, 4.3)]],
    )

    def words(identifier: str, first: str, last: str) -> dict:
        return {"id": identifier, "take": "t1", "from": {"word": first}, "to": {"word": last}}

    write_edit(root, [words("s1", "t1:0", "t1:0"), words("s2", "t1:1", "t1:1"),
                      words("s3", "t1:2", "t1:2")])
    plan = resolve_cut(root)
    # s1 runs straight into s2: one shared point, no join to listen to.
    assert plan.pieces[0].src_f1 == plan.pieces[1].src_f0
    assert not plan.pieces[0].listen_out and not plan.pieces[1].listen_in
    assert {(row["segment"], row["how"]) for row in plan.edges} >= {
        ("s1", "continuous"), ("s2", "continuous")
    }
    # "two" and "three" are neighbours 2.8 s apart, so the gap is cut and flagged.
    assert plan.pieces[2].src_f0 > plan.pieces[1].src_f1
    assert any("2.8 s apart" in warning for warning in plan.warnings)


@pytest.mark.parametrize("edge,offset,point", [
    ("start", 0.1, Fraction("0.6")), ("end", 0, Fraction("0.8")),
])
def test_word_anchor_overrides_are_exact(
    tmp_path: Path, calibration: Path, edge: str, offset: float, point: Fraction,
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [{
        "id": "s1", "take": "t1",
        "from": {"word": "t1:0", "edge": edge, "offset_s": offset}, "to": {"time": 2},
    }])
    plan = resolve_cut(root)
    assert plan.pieces[0].src_t0 == point
    assert plan.edges[0]["how"] == "time" and plan.edges[0]["listen"]


@pytest.mark.parametrize("side,override", [
    ("to", {"edge": "start"}), ("from", {"edge": "end"}),
    ("to", {"offset_s": -0.1}), ("from", {"offset_s": 0.1}),
])
def test_neighbouring_word_segments_respect_anchor_overrides(
    tmp_path: Path, calibration: Path, side: str, override: dict,
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    left = {"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"word": "t1:0"}}
    right = {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"time": 2}}
    (left if side == "to" else right)[side].update(override)
    write_edit(root, [left, right])
    plan = resolve_cut(root)
    assert all(row["how"] != "continuous" for row in plan.edges)
    assert plan.pieces[0].src_f1 != plan.pieces[1].src_f0
    assert plan.listen_joins


def test_exactly_one_second_gap_is_not_continuous(tmp_path: Path, calibration: Path) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("one", 0.5, 0.8), ("two", 1.8, 2.2),
    ]])
    write_edit(root, [
        {"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"word": "t1:0"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:1"}, "to": {"time": 3}},
    ])
    plan = resolve_cut(root)
    assert all(row["how"] != "continuous" for row in plan.edges)
    assert any("1.0 s apart" in warning for warning in plan.warnings)


def test_empty_normalised_anchor_cannot_be_confirmed_as_its_neighbour(
    tmp_path: Path, calibration: Path,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[
        ("...", 0.5, 0.8), ("welcome", 1.2, 1.5),
    ]])
    words = parse_document("words", json.loads((root / "ingest/t1/words.json").read_bytes()))
    assert isinstance(words, Words)
    assert edge_words(words, Anchor(word="t1:0"), start=True) == ()
    assert edge_words(words, Anchor(word="t1:0"), start=False) == ()
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:0"}, "to": {"time": 2},
    }])
    plan = resolve_cut(root)
    assert plan.edges[0]["how"] == "unconfirmed" and plan.edges[0]["listen"]


def test_crossed_placements_fall_back_without_losing_removals(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [calibration])
    write_edit(root, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:0"}, "to": {"word": "t1:2"},
    }])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "gap", "take": "t1", "start": 1.2, "end": 1.5,
             "kind": "manual", "status": "applied", "listen": True},
        ],
    })

    def crossed(requests, takes, backend, *, language=None):
        assert len(requests) == 2
        assert requests[0].other == requests[1].nominal
        assert requests[1].other == requests[0].nominal
        return [edges.EdgeResult(
            request.segment, request.start, Fraction("1.7" if request.start else "1.0"),
            request.nominal, "silence", False,
        ) for request in requests], []

    monkeypatch.setattr(edges, "place", crossed)
    plan = resolve_cut(root)
    assert [(piece.src_t0, piece.src_t1) for piece in plan.pieces] == [
        (Fraction(11, 30), Fraction("1.2")), (Fraction("1.5"), Fraction("2.3")),
    ]
    assert all(row["how"] == "unconfirmed" and row["listen"] for row in plan.edges)
    assert all(piece.listen_in and piece.listen_out for piece in plan.pieces)
    assert any("placements crossed" in warning for warning in plan.warnings)


@pytest.mark.parametrize("error", [DependencyError.missing_engine("speech"),
                                     ValidationError("asr", "invalid engine")])
def test_backend_resolution_only_swallows_missing_engine(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    root = make_project(tmp_path / "project", [calibration])

    def fail(name: str, model: str | None):
        raise error

    monkeypatch.setattr(cut_stage, "resolve_backend", fail)
    if isinstance(error, DependencyError):
        assert _edge_backend(ProjectContext(root)) is None
    else:
        with pytest.raises(ValidationError) as caught:
            _edge_backend(ProjectContext(root))
        assert caught.value is error


def test_removal_listen_survives_when_it_meets_a_silent_segment_edge(
    tmp_path: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [calibration], word_rows=[[]])
    write_edit(root, [time_segment("left", "t1", 0, 1), time_segment("right", "t1", 1.5, 2)])
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "gap", "take": "t1", "start": 1, "end": 1.5,
             "kind": "manual", "status": "applied", "listen": True},
        ],
    })
    sound = np.zeros(7 * edges.RATE, dtype=np.float32)
    quiet = edges._Take(sound, edges._levels(sound), edges.SILENCE_CEILING_DBFS)
    monkeypatch.setattr(edges, "load_take", lambda source, stream, *, words=(): quiet)
    plan = resolve_cut(root)
    assert all(not row["listen"] for row in plan.edges)
    assert plan.pieces[0].listen_out and plan.pieces[1].listen_in
    assert plan.listen_joins == ({"frame": 30, "time_s": 1, "left": "left", "right": "right"},)


@pytest.mark.parametrize("remove_repeat", [False, True])
def test_edit_check_repeats_use_retained_words_in_cut_order(
    tmp_path: Path, calibration: Path, remove_repeat: bool,
) -> None:
    rows = [
        (text, start + offset * 0.2, start + offset * 0.2 + 0.15)
        for start in (0.5, 4.5)
        for offset, text in enumerate("Amber lantern glows.".split())
    ]
    root = make_project(tmp_path / "project", [calibration], word_rows=[rows])
    write_edit(root, [time_segment("later", "t1", 4, 7), time_segment("earlier", "t1", 0, 2)])
    if remove_repeat:
        write_json(root / "removals.json", {
            "schema": "yap-studio/removals@1", "removals": [
                {"id": "repeat", "take": "t1", "start": 0.4, "end": 1.2,
                 "words": ["t1:0", "t1:1", "t1:2"], "kind": "manual", "status": "applied"},
            ],
        })
    before = snapshot(root)
    checked = edit_check(root)
    assert snapshot(root) == before
    if remove_repeat:
        assert checked["repeats"] == []
        assert not any(warning.startswith("Repeated:") for warning in checked["warnings"])
    else:
        assert checked["repeats"] == [{
            "kind": "phrase", "shared": "amber lantern glows",
            "first": {"time_s": 0.5, "sentence": "Amber lantern glows.",
                      "words": ["t1:3", "t1:4", "t1:5"]},
            "second": {"time_s": 3.5, "sentence": "Amber lantern glows.",
                       "words": ["t1:0", "t1:1", "t1:2"]},
        }]
        assert (
            'Repeated: "amber lantern glows" at 0:00 and again at 0:03 '
            '("Amber lantern glows."). Keep it once unless the repeat is on purpose.'
        ) in checked["warnings"]
