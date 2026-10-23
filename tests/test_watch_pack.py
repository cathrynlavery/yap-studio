"""Real offline watch pack measurements, publication safety, and CLI tests."""

from __future__ import annotations

import json
import os
import weakref
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_watch_images import decode_png

from yap_studio import cli, ffmpeg, paths, transcript
from yap_studio.edl import load_edit_list
from yap_studio.edl.model import Hold, PictureRegion
from yap_studio.errors import (
    DependencyError,
    InputError,
    OutputPathError,
    ProcessingError,
    ValidationError,
)
from yap_studio.receipt import sha256_file
from yap_studio.transcript import ScriptedBackend, Transcript, Word
from yap_studio.watch import KINDS, PACK_SCHEMA, SEVERITIES, build_pack, images, load_pack, measure
from yap_studio.watch import pack as pack_module


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("watch-media")
    video = root / "render.mp4"
    cuts = synth.watch_render(video)
    edl = synth.watch_spans(root / "spans.json", root / "missing-source.mp4", cuts)
    words = synth.watch_words(
        root / "words.json",
        [
            ("First", 0.5, 1.0),
            ("piece", 1.5, 2.0),
            ("Boundary", 2.8, 3.2),
            ("held", 6.5, 7.0),
            ("Louder", 8.0, 8.5),
        ],
    )
    silent = root / "no-audio.mp4"
    delayed = synth.watch_audio_offset(root / "delayed-audio.mp4", video)
    ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            str(silent),
        ],
        timeout_s=60,
    )
    return {
        "root": root,
        "video": video,
        "edl": edl,
        "words": words,
        "silent": silent,
        "delayed": delayed,
    }


@pytest.fixture(scope="module")
def prepared(media: dict[str, Path]) -> tuple[dict, Path]:
    out = media["root"] / "prepared"
    result = build_pack(media["video"], edl=media["edl"], out=out, words=media["words"])
    return result, out


def test_planned_cuts_and_video_facts(prepared: tuple[dict, Path], media: dict[str, Path]) -> None:
    pack, out = prepared
    assert pack["schema"] == PACK_SCHEMA
    assert pack == load_pack(out)
    assert pack["video"]["frames"] == 315
    assert pack["video"]["fps"] == "30/1"
    assert pack["video"]["duration_s"] == 10.5
    assert pack["video"]["path"] == str(media["video"].resolve())
    assert pack["edl"]["sha256"] == sha256_file(media["edl"])
    assert [cut["frame"] for cut in pack["cuts"]] == [90, 225]
    assert [cut["time_s"] for cut in pack["cuts"]] == [3.0, 7.5]
    assert all(
        cut["source"] == "edl"
        and cut["before"]["frame"] == cut["frame"] - 1
        and cut["after"]["frame"] == cut["frame"]
        for cut in pack["cuts"]
    )
    assert [(segment["start_frame"], segment["end_frame"]) for segment in pack["segments"]] == [
        (0, 90),
        (90, 225),
        (225, 315),
    ]


def test_second_pack_reuses_transcript_without_download(tmp_path, media, monkeypatch):
    from test_transcript_shared import CountingBackend

    from yap_studio import receipt
    from yap_studio.transcript import shared

    backend = CountingBackend()
    first, second = tmp_path / "first", tmp_path / "second"
    hashes = []
    hash_file = receipt.sha256_file

    def count(path):
        if Path(path) == media["video"]:
            hashes.append(path)
        return hash_file(path)

    monkeypatch.setattr(receipt, "sha256_file", count)
    build_pack(media["video"], out=first, backend=backend)
    assert shared.known_digest(media["video"]) == sha256_file(media["video"])
    backend.notice = "model unavailable"
    result = build_pack(media["video"], out=second, backend=backend)
    assert result["transcript"]["status"] == "ok"
    assert len(backend.calls) == 1
    assert hashes == [media["video"]]
    assert (first / "transcript.json").read_bytes() == (second / "transcript.json").read_bytes()


def test_frozen_span_and_motion(prepared: tuple[dict, Path]) -> None:
    pack, out = prepared
    spans = pack["motion"]["still_spans"]
    assert len(spans) == 1
    assert pack["settings"]["still_threshold"] == 0.25
    motion = json.loads((out / "motion.json").read_text())
    assert motion["schema"] == "yap-studio/watch-motion@1"
    assert motion["frames"] == len(motion["values"]) == 315
    assert motion["values"][0] == 0.0
    if (
        motion["values"][90] >= 0.25
        and motion["values"][225] >= 0.25
        and max(motion["values"][91:225]) < 0.25
    ):
        assert spans[0] == {
            "start_s": 3.0,
            "end_s": 7.5,
            "duration_s": 4.5,
            "start_frame": 90,
            "end_frame": 225,
        }
    else:
        assert abs(spans[0]["start_frame"] - 90) <= 1
        assert abs(spans[0]["end_frame"] - 225) <= 1
        assert abs(spans[0]["duration_s"] - 4.5) <= 1 / 30 + 0.001
    assert max(motion["values"][100:220]) < 0.25
    assert np.median(motion["values"][1:90]) > 0.25
    assert np.median(motion["values"][226:315]) > 0.25
    assert len(pack["motion"]["per_second"]) == 11


def test_loudness_jump_and_silence(prepared: tuple[dict, Path]) -> None:
    pack, _ = prepared
    assert pack["loudness"]["status"] == "ok"
    assert pack["loudness"]["integrated_lufs"] is not None
    assert 10 < pack["cuts"][1]["loudness_jump_lu"] < 20
    spans = pack["loudness"]["silences"]
    assert len(spans) == 1
    assert abs(spans[0]["start_s"] - 4.0) < 0.05
    assert abs(spans[0]["end_s"] - 6.0) < 0.05
    assert spans[0]["duration_s"] > 1.9
    seconds = {item["second"]: item["lufs"] for item in pack["loudness"]["per_second"]}
    assert seconds[5] is None
    assert seconds[2] is not None and seconds[8] > seconds[2] + 10


def test_short_audio_silent_ending(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = synth.watch_short_audio(tmp_path / "short-audio.mp4")
    info = ffmpeg.probe(source, count_frames=True)
    assert info.video_frames == 300
    assert info.audio_duration_s == pytest.approx(7, abs=0.05)
    out = tmp_path / "pack"
    assert cli.main([
        "watch", "pack", str(source), "--out", str(out), "--asr", "none", "--json",
    ]) == 0
    pack = json.loads(capsys.readouterr().out)
    assert pack == load_pack(out)
    assert pack["transcript"]["status"] == "skipped"
    ending = pack["loudness"]["silences"][-1]
    assert ending["start_s"] == pytest.approx(7, abs=0.05)
    assert ending["end_s"] == pack["video"]["duration_s"] == 10
    assert ending["duration_s"] == pytest.approx(3, abs=0.05)
    seconds = {item["second"]: item["lufs"] for item in pack["loudness"]["per_second"]}
    assert seconds[6] is not None
    assert all(seconds[second] is None for second in (7, 8, 9))


def test_pacing_and_transcript_files(prepared: tuple[dict, Path]) -> None:
    pack, out = prepared
    first, held, last = pack["segments"]
    assert first["words"] == 2 and first["words_per_s"] == 0.667
    assert first["longest_pause_s"] == 1.0 and first["longest_pause_at_s"] == 2.0
    assert first["text"] == "First piece"
    assert held["words"] == 2 and held["words_per_s"] == 0.444
    assert held["longest_pause_s"] == 3.3 and held["longest_pause_at_s"] == 3.2
    assert held["text"] == "Boundary held"
    assert held["longest_silence_s"] > 1.9
    assert last["longest_pause_s"] == 2.0 and last["longest_pause_at_s"] == 8.5
    assert pack["transcript"]["status"] == "ok"
    assert pack["transcript"]["source"] == "file"
    assert pack["transcript"]["words"] == 5
    assert json.loads((out / "transcript.json").read_text())["words"][2]["start"] == 2.8
    assert (out / "transcript.txt").read_text().splitlines() == [
        "Transcript, engine: prepared",
        "",
        "[0:00.50] First piece",
        "[0:02.80] Boundary held",
        "[0:08.00] Louder",
    ]


def test_strips_hashes_and_guide(prepared: tuple[dict, Path]) -> None:
    pack, out = prepared
    strips = pack["timeline"]["strips"]
    assert len(strips) == 2
    assert sum(len(strip["tiles"]) for strip in strips) == 10
    assert pack["timeline"]["tile"] == {"width": 180, "height": 320, "label_height": 24, "gap": 2}
    assert pack["settings"]["gap"] == 2
    assert pack["settings"]["max_tiles"] == 5000
    assert all(isinstance(tile["cut_roles"], list) for strip in strips for tile in strip["tiles"])
    for cut in pack["cuts"]:
        pair = [
            (index, tile)
            for index, strip in enumerate(strips)
            for tile in strip["tiles"]
            if tile["cut"] == cut["index"]
        ]
        assert pair[0][0] == pair[1][0]
        assert pair[0][1]["kind"] == "cut_before" and pair[1][1]["kind"] == "cut_after"
        assert pair[0][1]["column"] + 1 == pair[1][1]["column"]
    for strip in strips:
        assert decode_png((out / strip["image"]).read_bytes()).shape == (
            344,
            180 * len(strip["tiles"]) + 2 * (len(strip["tiles"]) - 1),
            3,
        )
    assert [item["path"] for item in pack["outputs"]] == sorted(
        item["path"] for item in pack["outputs"]
    )
    assert {item["path"] for item in pack["outputs"]} == {
        item.relative_to(out).as_posix()
        for item in out.rglob("*")
        if item.is_file() and item.name != "pack.json"
    }
    assert all(sha256_file(out / item["path"]) == item["sha256"] for item in pack["outputs"])
    guide = (out / "pack.md").read_text()
    assert "Labels show M:SS.ss." in guide
    assert "0:03.00 to 0:07.50" in guide
    assert "column 3" in guide and "## Segments" in guide and "## Silences" in guide
    assert chr(0x2013) not in guide and chr(0x2014) not in guide
    assert "## Reporting findings" in guide
    assert "yap-studio watch scan ." in guide
    assert f"yap-studio watch list {pack['video']['path']}" in guide
    assert f"yap-studio watch note {pack['video']['path']} --file FINDINGS.json" in guide
    assert f"Kinds: {', '.join(KINDS)}" in guide
    assert f"Severities: {', '.join(SEVERITIES)}" in guide
    assert '"reviewer": "<your name>"' in guide


def test_determinism_and_owned_rerun(media: dict[str, Path], tmp_path: Path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    pack = build_pack(media["video"], edl=media["edl"], out=one, words=media["words"])
    again = build_pack(media["video"], edl=media["edl"], out=two, words=media["words"])
    assert pack == again
    names = ["pack.json", *(item["path"] for item in pack["outputs"])]
    original = {name: (one / name).read_bytes() for name in names}
    assert all((two / name).read_bytes() == content for name, content in original.items())
    dense = build_pack(media["video"], edl=media["edl"], out=one, every_s=0.5, words=media["words"])
    stale = one / dense["timeline"]["strips"][-1]["image"]
    assert stale.name > pack["timeline"]["strips"][-1]["image"].split("/")[-1]
    build_pack(media["video"], edl=media["edl"], out=one, words=media["words"])
    assert not stale.exists()
    assert all((one / name).read_bytes() == content for name, content in original.items())


def test_foreign_folder_refused_untouched(media: dict[str, Path], tmp_path: Path, capsys) -> None:
    out = tmp_path / "foreign"
    out.mkdir()
    (out / "mine.txt").write_bytes(b"keep this")
    assert cli.main(["watch", "pack", str(media["video"]), "--out", str(out), "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsafe_path"
    assert list(out.iterdir()) == [out / "mine.txt"]
    assert (out / "mine.txt").read_bytes() == b"keep this"


@pytest.mark.parametrize("change", ["foreign", "modified", "missing", "symlink", "receipt"])
def test_previous_pack_refusals(
    prepared: tuple[dict, Path],
    media: dict[str, Path],
    tmp_path: Path,
    change: str,
) -> None:
    pack, original = prepared
    out = tmp_path / "pack"
    out.mkdir()
    for name in ["pack.json", *(item["path"] for item in pack["outputs"])]:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    if change == "foreign":
        (out / "nested").mkdir()
        (out / "nested" / "foreign.json").write_text("{}")
    elif change == "modified":
        (out / "pack.md").write_text("changed externally")
    elif change == "missing":
        (out / "motion.json").unlink()
    elif change == "symlink":
        (out / "linked").symlink_to(media["video"])
    else:
        data = load_pack(out)
        data["outputs"].append({"path": "../outside", "sha256": "wrong"})
        (out / "pack.json").write_text(json.dumps(data))
    before = {item.relative_to(out): item.read_bytes() for item in out.rglob("*") if item.is_file()}
    with pytest.raises(OutputPathError) as exc:
        build_pack(media["video"], out=out, asr="none")
    assert exc.value.code == "unsafe_path"
    assert before == {
        item.relative_to(out): item.read_bytes() for item in out.rglob("*") if item.is_file()
    }


def test_output_cannot_contain_any_input(media: dict[str, Path], tmp_path: Path, capsys) -> None:
    original = media["video"].read_bytes()
    for out in (media["video"], media["root"]):
        assert cli.main(["watch", "pack", str(media["video"]), "--out", str(out)]) == 3
        assert "contains an input" in capsys.readouterr().err
    assert media["video"].read_bytes() == original
    words = synth.watch_words(tmp_path / "input.json", [("input", 0.1, 0.5)])
    with pytest.raises(OutputPathError):
        build_pack(media["video"], words=words, out=tmp_path)
    assert list(tmp_path.iterdir()) == [words]


@pytest.mark.parametrize("state", ["none", "unavailable", "download", "ok", "failed", "named"])
def test_transcript_statuses(
    media: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    calls = []
    previous = "previous-value"
    monkeypatch.setenv("HF_HUB_OFFLINE", previous)

    def transcribe(source: Path) -> Transcript:
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        calls.append(source)
        if state == "failed":
            raise ProcessingError("scripted speech", 1, "engine stopped")
        return Transcript([Word("Hello", 0.1, 0.5)])

    backend = ScriptedBackend(transcribe)
    if state == "download":
        monkeypatch.setattr(backend, "download_notice", lambda: "not cached")
    if state == "unavailable":
        monkeypatch.setattr(transcript, "resolve_backend", lambda name: None)
    elif state == "named":

        def missing(name: str):
            raise DependencyError.missing_engine(name)

        monkeypatch.setattr(transcript, "resolve_backend", missing)
    result = build_pack(
        media["video"],
        edl=media["edl"],
        out=tmp_path / "pack",
        asr="none" if state == "none" else "parakeet-onnx" if state == "named" else "auto",
        backend=None if state in {"unavailable", "named"} else backend,
    )
    expected = {
        "none": "skipped",
        "unavailable": "unavailable",
        "download": "skipped",
        "ok": "ok",
        "failed": "failed",
        "named": "unavailable",
    }
    assert result["transcript"]["status"] == expected[state]
    assert os.environ["HF_HUB_OFFLINE"] == previous
    assert bool(calls) == (state in {"ok", "failed"})
    if state == "unavailable":
        assert result["transcript"]["reason"] == (
            "no local speech engine installed; pip install 'yap-studio[asr]' or pass --words"
        )
    if state == "download":
        assert "verify or yap-studio ingest once while online" in result["transcript"]["reason"]
        assert "doctor" not in result["transcript"]["reason"]
    if state == "ok":
        assert result["transcript"]["engine"] == "scripted"
        assert result["segments"][0]["text"] == "Hello"
    else:
        assert result["segments"][0]["words"] is None


def test_offline_environment_removed_after_asr(media: dict[str, Path], tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    backend = ScriptedBackend(lambda _: Transcript([]))
    build_pack(media["video"], edl=media["edl"], out=tmp_path / "pack", backend=backend)
    assert "HF_HUB_OFFLINE" not in os.environ


def test_cut_words_and_text_wrapping(media: dict[str, Path], tmp_path: Path) -> None:
    words = tmp_path / "cut-words.json"
    chosen = [
        {"text": f"word{index}", "start": index * 0.1, "end": index * 0.1 + 0.05}
        for index in range(15)
    ]
    words.write_text(
        json.dumps({"schema": "yap-studio/cut-words@1", "time_base": "output", "words": chosen})
    )
    out = tmp_path / "pack"
    pack = build_pack(media["video"], edl=media["edl"], out=out, words=words, asr="none")
    assert pack["transcript"]["status"] == "ok" and pack["transcript"]["words"] == 15
    lines = (out / "transcript.txt").read_text().splitlines()[2:]
    assert len(lines) == 2
    assert lines[0].startswith("[0:00.00]") and len(lines[0].split()) == 13
    assert lines[1].startswith("[0:01.20]") and len(lines[1].split()) == 4
    words.write_text(
        json.dumps({"schema": "yap-studio/cut-words@1", "time_base": "source", "words": chosen})
    )
    with pytest.raises(ValidationError, match="output time"):
        build_pack(media["video"], edl=media["edl"], out=out, words=words)


@pytest.mark.parametrize("state", ["found", "stale", "missing"])
def test_verify_reports(media: dict[str, Path], tmp_path: Path, state: str) -> None:
    report = tmp_path / "verify-input.json"
    data = {
        "schema": "yap-studio/verify-report@1",
        "status": "warn",
        "summary": {"status": "warn", "text": "Check this cut", "fail": 0, "warn": 1},
        "findings": [
            {
                "check": "audio.level",
                "severity": "warn",
                "time_s": 7.5,
                "message": "Level changes",
                "suggested_action": "Listen",
            }
        ],
        "receipt": {
            "inputs": [
                {
                    "path": "render.mp4",
                    "sha256": sha256_file(media["video"]) if state == "found" else "0" * 64,
                }
            ]
        },
    }
    report.write_text(json.dumps(data))
    out = tmp_path / "pack"
    pack = build_pack(
        media["video"],
        edl=media["edl"],
        out=out,
        asr="none",
        verify=report if state != "missing" else None,
    )
    assert pack["verify"]["status"] == state
    if state == "found":
        assert (out / "verify.json").read_bytes() == report.read_bytes()
        assert pack["verify"]["result"] == "warn"
        assert pack["verify"]["summary_text"] == "Check this cut"
        assert pack["verify"]["findings"] == [
            {"check": "audio.level", "severity": "warn", "time_s": 7.5, "message": "Level changes"}
        ]
        assert "Level changes" in (out / "pack.md").read_text()
    else:
        assert not (out / "verify.json").exists()
        assert pack["verify"]["file"] is None


def test_detected_cuts(media: dict[str, Path], tmp_path: Path) -> None:
    pack = build_pack(media["video"], out=tmp_path / "pack", asr="none")
    assert pack["edl"] is None
    assert all(cut["source"] == "detected" for cut in pack["cuts"])
    assert {cut["frame"] for cut in pack["cuts"]} == {90, 225}
    assert len(pack["segments"]) == 3


def test_no_audio_and_no_cuts(media: dict[str, Path], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pack_module, "detect_cuts", lambda _: [])
    pack = build_pack(media["silent"], out=tmp_path / "pack", asr="none")
    assert pack["loudness"] == {
        "status": "no_audio",
        "integrated_lufs": None,
        "per_second": [],
        "silences": [],
    }
    assert len(pack["segments"]) == 1
    assert pack["segments"][0]["longest_silence_s"] is None
    assert pack["segments"][0]["words"] is None


def test_graphics_and_plan_warnings(media: dict[str, Path], tmp_path: Path, monkeypatch, capsys):
    edits = replace(
        load_edit_list(media["edl"]),
        picture_masks=[(Fraction(1), Fraction(2))],
        picture_regions=[PictureRegion(Fraction(2), Fraction(3), 1, 2, 3, 4)],
        holds=[Hold(1, 315, 330)],
        warnings=["Review declared hold"],
    )
    monkeypatch.setattr(pack_module, "load_edit_list", lambda _: edits)
    out = tmp_path / "pack"
    assert (
        cli.main(
            [
                "watch",
                "pack",
                str(media["video"]),
                "--edl",
                str(media["edl"]),
                "--out",
                str(out),
                "--asr",
                "none",
            ]
        )
        == 0
    )
    pack = load_pack(out)
    assert pack["graphics"] == [
        {"kind": "mask", "start_s": 1.0, "end_s": 2.0, "rect": None},
        {"kind": "region", "start_s": 2.0, "end_s": 3.0, "rect": [1, 2, 3, 4]},
        {"kind": "hold", "start_s": 10.5, "end_s": 11.0},
    ]
    text = capsys.readouterr().out
    assert "warning: Review declared hold" in text and "plans 330 frames" in text


def test_cli_text_json_and_parser(media: dict[str, Path], tmp_path: Path, capsys) -> None:
    out = tmp_path / "pack"
    args = [
        "watch",
        "pack",
        str(media["video"]),
        "--edl",
        str(media["edl"]),
        "--out",
        str(out),
        "--asr",
        "none",
    ]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "10 frames, 2 strips written" in text
    assert "Cuts: 2, source: edit list" in text
    assert "transcript: skipped" in text and "verify: missing" in text
    assert f"next: yap-studio watch scan {out}, then read {out / 'pack.md'} and the strips" in text
    assert cli.main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == load_pack(out)
    assert "watch" in cli._parser().format_help()
    assert cli.main(["watch"]) == 2
    capsys.readouterr()


@pytest.mark.parametrize(
    "options", [["--every", "0"], ["--every", "nan"], ["--columns", "0"], ["--columns", "13"]]
)
def test_cli_invalid_options(media: dict[str, Path], tmp_path: Path, capsys, options: list[str]):
    out = tmp_path / "pack"
    assert (
        cli.main(["watch", "pack", str(media["video"]), "--out", str(out), "--json", *options]) == 2
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_parameter"
    assert not out.exists()


def test_cli_invalid_inputs(tmp_path: Path, capsys) -> None:
    assert cli.main(["watch", "pack", str(tmp_path / "missing.mp4")]) == 2
    text = capsys.readouterr().err
    assert "error:" in text and "fix:" in text


def test_pure_measurements() -> None:
    frames = np.array([[[0]], [[10]], [[30]]], dtype=np.uint8)
    assert measure.motion_values(frames) == [0.0, 10.0, 20.0]
    assert measure.motion_values(frames, np.array([[255]], dtype=np.uint8))[0] == 255.0
    blocks, integrated = measure.parse_ebur128(
        "t: 0.399979 M: -20.0\nt: 0.499979 M: -80.0\nt: 0.599979 M: -40.0\n"
        "t: 1.0 M: -inf\nSummary:\nI: -70.0 LUFS\nSummary:\nI: -21.0 LUFS\n"
    )
    assert len(blocks) == 3 and integrated == -21.0
    assert measure.window_loudness(blocks, 0, 1, speech=True) == -20.0
    assert measure.window_loudness(blocks, 0.3, 0.6, speech=True) is None
    assert measure.pacing(0, 3, [], None)["longest_pause_s"] == 3.0
    assert measure.silences(np.zeros(4800, dtype=np.float32), 16000) == [
        {"start_s": 0.0, "end_s": 0.3, "duration_s": 0.3}
    ]
    assert (
        measure.cut_loudness([(0.4, -20), (0.5, -20), (1.4, -5), (1.5, -5)], [1.0, 1.2], 2.0)[0][
            "loudness_after_lufs"
        ]
        is None
    )


def test_index_written_last(media: dict[str, Path], tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "pack"
    replace_file = paths.os.replace
    published = []

    def observe(source, target):
        if Path(target).name == "pack.json":
            index = json.loads(Path(source).read_text())
            assert all((out / item["path"]).exists() for item in index["outputs"])
        else:
            assert not (out / "pack.json").exists()
        published.append(Path(target).name)
        replace_file(source, target)

    monkeypatch.setattr(paths.os, "replace", observe)
    build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    assert published[-1] == "pack.json"


@pytest.mark.parametrize("report_name", ["verify.json", "render.verify.json"])
def test_default_folder_and_report_discovery(
    media: dict[str, Path],
    tmp_path: Path,
    report_name: str,
) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(media["video"].read_bytes())
    report = {
        "schema": "yap-studio/verify-report@1",
        "status": "pass",
        "summary": {"text": "Verified"},
        "findings": [],
        "receipt": {"inputs": [{"sha256": sha256_file(source)}]},
    }
    (tmp_path / report_name).write_text(json.dumps(report))
    pack = build_pack(source, edl=media["edl"], asr="none")
    assert pack == load_pack(tmp_path / "render-watch")
    assert pack["verify"]["status"] == "found"
    assert pack["settings"]["verify"]["path"] == str(tmp_path / report_name)


@pytest.mark.parametrize(
    "stem,folder,expected,status",
    [
        ("matching", "stale", "render.verify.json", "found"),
        ("matching", "matching", "render.verify.json", "found"),
        (None, "stale", "verify.json", "stale"),
        (None, None, None, "missing"),
        ("stale", "matching", "verify.json", "found"),
        ("stale", "stale", "render.verify.json", "stale"),
    ],
)
def test_report_discovery_matches_video_hash(
    media: dict[str, Path],
    tmp_path: Path,
    stem: str | None,
    folder: str | None,
    expected: str | None,
    status: str,
) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(media["video"].read_bytes())
    report = {
        "schema": "yap-studio/verify-report@1",
        "status": "pass",
        "summary": {"text": "Verified"},
        "findings": [],
        "receipt": {"inputs": [{"sha256": sha256_file(source)}]},
    }
    for name, state in (("render.verify.json", stem), ("verify.json", folder)):
        if state is not None:
            report["receipt"]["inputs"][0]["sha256"] = (
                sha256_file(source) if state == "matching" else "0" * 64
            )
            report["summary"]["text"] = name
            (tmp_path / name).write_text(json.dumps(report))
    pack = build_pack(source, edl=media["edl"], asr="none")
    assert pack["verify"]["status"] == status
    if expected is None:
        assert pack["settings"]["verify"] is None
    else:
        selected = tmp_path / expected
        assert pack["settings"]["verify"] == {
            "path": str(selected), "sha256": sha256_file(selected),
        }
    copied = tmp_path / "render-watch" / "verify.json"
    if status == "found":
        assert pack["verify"]["summary_text"] == expected
        assert copied.read_bytes() == (tmp_path / expected).read_bytes()
    else:
        assert not copied.exists()


@pytest.mark.parametrize("payload", ["{", "[]", "{}"])
@pytest.mark.parametrize("matching", [False, True])
def test_report_discovery_skips_invalid_json(
    media: dict[str, Path], tmp_path: Path, payload: str, matching: bool
) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(media["video"].read_bytes())
    invalid = tmp_path / "render.verify.json"
    invalid.write_text(payload)
    if matching:
        (tmp_path / "verify.json").write_text(json.dumps({
            "schema": "yap-studio/verify-report@1",
            "receipt": {"inputs": [{"sha256": sha256_file(source)}]},
        }))
    pack = build_pack(source, edl=media["edl"], asr="none")
    assert pack["verify"]["status"] == ("found" if matching else "stale")
    assert pack["settings"]["verify"]["path"] == str(
        tmp_path / ("verify.json" if matching else "render.verify.json")
    )
    with pytest.raises((InputError, ValidationError)):
        build_pack(source, edl=media["edl"], asr="none", verify=invalid)


def test_report_discovery_skips_unreadable_json(
    media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(media["video"].read_bytes())
    unreadable = tmp_path / "render.verify.json"
    unreadable.write_text("{}")
    report = tmp_path / "verify.json"
    report.write_text(json.dumps({
        "schema": "yap-studio/verify-report@1",
        "receipt": {"inputs": [{"sha256": sha256_file(source)}]},
    }))
    read_text = Path.read_text

    def read_or_refuse(path: Path, *args, **kwargs) -> str:
        if path == unreadable:
            raise PermissionError("unreadable report")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_or_refuse)
    pack = build_pack(source, edl=media["edl"], asr="none")
    assert pack["verify"]["status"] == "found"
    assert pack["settings"]["verify"]["path"] == str(report)
    report.unlink()
    pack = build_pack(source, edl=media["edl"], asr="none")
    assert pack["verify"]["status"] == "stale"
    with pytest.raises(InputError, match="unreadable JSON"):
        build_pack(source, edl=media["edl"], asr="none", verify=unreadable)


@pytest.mark.parametrize("columns", [1, 2, 3])
def test_thumbnail_memory_is_bounded_by_largest_row(
    media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, columns: int
) -> None:
    frames = 60
    width, height = images.tile_dimensions(90, 160)
    references: list[weakref.ReferenceType[np.ndarray]] = []
    retained: list[int] = []
    decoded: list[int] = []
    written: list[tuple[str, int]] = []
    frombuffer = np.frombuffer
    write_bytes = Path.write_bytes

    class Thumbnail(np.ndarray):
        def copy(self, *args, **kwargs) -> np.ndarray:
            result = super().copy(*args, **kwargs)
            references.append(weakref.ref(result))
            retained.append(sum(reference() is not None for reference in references))
            return result

    def tracked_buffer(*args, **kwargs) -> np.ndarray:
        return frombuffer(*args, **kwargs).view(Thumbnail)

    def grey_frames(*args, **kwargs) -> Iterator[tuple[int, np.ndarray]]:
        yield 0, np.zeros((frames, 2, 2), dtype=np.uint8)

    def frame_source(*args, **kwargs) -> Iterator[bytes]:
        for frame in range(frames):
            decoded.append(frame)
            yield bytes([frame]) * (width * height * 3)

    def observe_write(path: Path, content: bytes) -> int:
        result = write_bytes(path, content)
        if path.parent.name == "strips":
            written.append((path.name, len(decoded)))
        return result

    monkeypatch.setattr(pack_module.video_module, "iter_grey_frames", grey_frames)
    monkeypatch.setattr(pack_module, "detect_cuts", lambda _: [
        SimpleNamespace(index=index, frame=frame)
        for index, frame in enumerate((30, 31, 32), 1)
    ])
    monkeypatch.setattr(images.ffmpeg, "run_stream", frame_source)
    monkeypatch.setattr(images.np, "frombuffer", tracked_buffer)
    monkeypatch.setattr(Path, "write_bytes", observe_write)
    out = tmp_path / "pack"
    pack = build_pack(media["silent"], out=out, asr="none", every_s=1 / 30, columns=columns)
    strips = pack["timeline"]["strips"]
    largest_row = max(len(strip["tiles"]) for strip in strips)
    assert largest_row == 4 > columns
    assert len(references) == len(decoded) == frames
    assert max(retained) <= largest_row
    assert not any(reference() is not None for reference in references)
    assert written[0][1] == len(strips[0]["tiles"]) < frames
    assert len(written) == len(strips)
    for strip in strips:
        thumbnails = {
            tile["frame"]: np.full((height, width, 3), tile["frame"], dtype=np.uint8)
            for tile in strip["tiles"]
        }
        assert (out / strip["image"]).read_bytes() == images.png_bytes(
            images.compose_strip(strip["tiles"], thumbnails)
        )


@pytest.mark.parametrize(
    "payload",
    ["[]", "{}", '{"schema":"yap-studio/transcript@1","words":[{"text":"bad","start":1,"end":0}]}'],
)
def test_invalid_words_preserve_existing_pack(
    media: dict[str, Path],
    tmp_path: Path,
    capsys,
    payload: str,
) -> None:
    out = tmp_path / "pack"
    build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    before = {item.relative_to(out): item.read_bytes() for item in out.rglob("*") if item.is_file()}
    words = tmp_path / "bad.json"
    words.write_text(payload)
    assert (
        cli.main(
            [
                "watch",
                "pack",
                str(media["video"]),
                "--out",
                str(out),
                "--edl",
                str(media["edl"]),
                "--words",
                str(words),
                "--json",
            ]
        )
        == 2
    )
    assert "error" in json.loads(capsys.readouterr().err)
    assert before == {
        item.relative_to(out): item.read_bytes() for item in out.rglob("*") if item.is_file()
    }


def test_colour_thumbnail_indexes(media: dict[str, Path]) -> None:
    info = ffmpeg.probe(media["video"], count_frames=True)
    wanted = [89, 90, 224, 225]
    frames = dict(
        images.decode_thumbnails(media["video"], wanted, info=info, width=180, height=320)
    )
    reference = ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(media["video"]),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "setpts=PTS-STARTPTS,fps=30/1:start_time=0:round=near,"
            "select='eq(n,89)+eq(n,90)+eq(n,224)+eq(n,225)',scale=180:320:flags=area,format=rgb24",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        timeout_s=60,
        binary=True,
    )
    expected = np.frombuffer(reference.stdout, dtype=np.uint8).reshape(4, 320, 180, 3)
    assert set(frames) == set(wanted)
    for index, frame in enumerate(wanted):
        np.testing.assert_array_equal(frames[frame], expected[index])


def test_still_spans_include_first_identical_frame() -> None:
    identical = np.full((120, 2, 2), 128, dtype=np.uint8)
    values = measure.motion_values(identical, np.zeros((2, 2), dtype=np.uint8))
    assert measure.motion_spans(values, 30, threshold=0.25, minimum_s=0.5, still=True) == [
        {"start_s": 0.0, "end_s": 4.0, "duration_s": 4.0, "start_frame": 0, "end_frame": 120},
    ]
    assert (
        measure.motion_spans([10.0, 0.0, 0.0], 30, threshold=0.25, minimum_s=0.1, still=True)[0][
            "duration_s"
        ]
        == 0.1
    )
    assert (
        measure.motion_spans([0.0, 6.0, 6.0, 0.0], 30, threshold=6, minimum_s=2 / 30, still=False)[
            0
        ]["start_frame"]
        == 1
    )


def test_asr_times_use_picture_clock(media: dict[str, Path], tmp_path: Path) -> None:
    source = media["delayed"]
    info = ffmpeg.probe(source, count_frames=True)
    shift = info.audio_offset_s - info.video_offset_s
    assert 0.9 < shift < 1.1
    raw = Transcript([Word("Shifted", 2.1, 2.4, 0.9)], backend="scripted")
    backend = ScriptedBackend(lambda _: raw)
    out = tmp_path / "asr-pack"
    pack = build_pack(source, edl=media["edl"], out=out, backend=backend)
    word = json.loads((out / "transcript.json").read_text())["words"][0]
    assert word["start"] == round(2.1 + shift, 3)
    assert word["end"] == round(2.4 + shift, 3)
    assert word["conf"] == 0.9
    assert pack["settings"]["asr_word_offset_s"] == round(shift, 3)
    assert pack["segments"][0]["words"] == 0
    assert pack["segments"][1]["text"] == "Shifted"
    assert (
        (out / "transcript.txt")
        .read_text()
        .splitlines()[2]
        .startswith(f"[{images.format_time(2.1 + shift)}]")
    )
    assert raw.words[0].start == 2.1

    # File timestamps already use the picture clock and receive no audio shift.
    words = synth.watch_words(tmp_path / "words.json", [("Unchanged", 2.1, 2.4)])
    file_out = tmp_path / "file-pack"
    pack = build_pack(source, edl=media["edl"], out=file_out, words=words)
    word = json.loads((file_out / "transcript.json").read_text())["words"][0]
    assert word["start"] == 2.1 and word["end"] == 2.4
    assert pack["segments"][0]["text"] == "Unchanged"


def test_asr_mapping_clips_words_before_picture_zero() -> None:
    words = [Word("Earlier", 0.1, 0.4), Word("Crosses", 0.8, 1.2, 0.9), Word("Later", 1.5, 2.0)]
    mapped = measure.words_on_picture_clock(words, 0.5, 1.5)
    assert len(mapped) == 2
    assert mapped[0].text == "Crosses" and mapped[0].start == 0.0
    assert mapped[0].end == pytest.approx(0.2) and mapped[0].conf == 0.9
    assert mapped[1] == Word("Later", 0.5, 1.0)
    assert words[1].start == 0.8


def test_cli_transcript_engine_and_count(
    media: dict[str, Path], tmp_path: Path, monkeypatch, capsys
):
    backend = ScriptedBackend(lambda _: Transcript([Word("Hello", 0.1, 0.5)]), name="parakeet-mlx")
    monkeypatch.setattr(transcript, "resolve_backend", lambda _: backend)
    assert (
        cli.main(
            [
                "watch",
                "pack",
                str(media["video"]),
                "--edl",
                str(media["edl"]),
                "--out",
                str(tmp_path / "pack"),
            ]
        )
        == 0
    )
    assert "transcript: ok, parakeet-mlx, 1 words" in capsys.readouterr().out
    assert (
        cli.main(
            [
                "watch",
                "pack",
                str(media["video"]),
                "--edl",
                str(media["edl"]),
                "--words",
                str(media["words"]),
                "--out",
                str(tmp_path / "file-pack"),
            ]
        )
        == 0
    )
    assert "transcript: ok, prepared, 5 words" in capsys.readouterr().out


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_tile_limit_refusal_preserves_output(
    media: dict[str, Path],
    tmp_path: Path,
    monkeypatch,
    capsys,
    json_output: bool,
) -> None:
    out = tmp_path / "pack"
    build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    before = {item.relative_to(out): item.read_bytes() for item in out.rglob("*") if item.is_file()}

    def grey_frames(*args, **kwargs):
        yield 0, np.zeros((5001, 2, 2), dtype=np.uint8)

    monkeypatch.setattr(
        pack_module.video_module,
        "iter_grey_frames",
        grey_frames,
    )
    monkeypatch.setattr(pack_module, "detect_cuts", lambda _: [])
    monkeypatch.setattr(
        pack_module, "_sound", lambda *args: pytest.fail("limit must run before sound")
    )
    args = ["watch", "pack", str(media["video"]), "--out", str(out), "--every", str(1 / 30)]
    assert cli.main([*args, "--json"] if json_output else args) == 2
    error = capsys.readouterr().err
    if json_output:
        data = json.loads(error)["error"]
        assert data["code"] == "invalid_parameter" and "--every" in data["suggested_action"]
    else:
        assert "error:" in error and "5000" in error and "fix:" in error and "--every" in error
    assert before == {
        item.relative_to(out): item.read_bytes() for item in out.rglob("*") if item.is_file()
    }
