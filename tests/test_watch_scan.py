"""Pack-only heuristic scans, real media acceptance, and findings preservation."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import synth

from yap_studio import cli, ffmpeg
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.receipt import sha256_file
from yap_studio.transcript import Word
from yap_studio.watch import (
    LOUDNESS_JUMP_LU,
    PAUSE_S,
    STILL_S,
    build_pack,
    default_findings_path,
    load_findings,
    load_pack,
    note_from_file,
    run_scan,
    scan_pack,
)
from yap_studio.watch.scan import GRAPHIC_MIN_AREA


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("watch-scan")
    video = root / "render.mp4"
    cuts = synth.watch_render(video)
    edl = synth.watch_spans(root / "spans.json", video, cuts)
    words = synth.watch_words(
        root / "words.json",
        [("First", 0.5, 1.0), ("piece", 1.5, 4.0), ("Boundary", 6.0, 8.5)],
    )
    packs = {"video": video}
    for name, word_file in (("silence", None), ("words", words)):
        packs[name] = root / name
        build_pack(video, edl=edl, out=packs[name], words=word_file, asr="none")
    for name, overlays, holds in (
        ("mask", [{"start_in_output": 7.1, "duration": 0.4}], []),
        ("region", [{"start_in_output": 3.5, "duration": 1, "region": [0, 0, 40, 40]}], []),
        ("hold", [], [(90, 225)]),
    ):
        ranges = synth.watch_ranges(
            root / f"{name}.json", video, cuts, overlays=overlays, holds=holds
        )
        packs[name] = root / name
        build_pack(video, edl=ranges, out=packs[name], asr="none")
    exact = synth.watch_four_second_freeze(root / "exact.mp4")
    exact_edl = synth.watch_spans(root / "exact-spans.json", exact, [], end_s=8)
    packs["exact_video"] = exact
    packs["exact"] = root / "exact"
    build_pack(exact, edl=exact_edl, out=packs["exact"], asr="none")
    return packs


def entries(result: dict, check: str) -> list[dict]:
    return [entry for entry in result["findings"] if entry["check"] == check]


def pack_dict(*, still=None, silences=None, cuts=None, graphics=None, has_audio=True) -> dict:
    return {
        "video": {
            "fps": "30/1",
            "duration_s": 12.0,
            "has_audio": has_audio,
            "width": 100,
            "height": 200,
        },
        "motion": {"still_spans": still or []},
        "loudness": {"status": "ok" if has_audio else "no_audio", "silences": silences or []},
        "cuts": cuts or [],
        "graphics": graphics or [],
    }


def still_span(start: int, end: int) -> dict:
    return {"start_frame": start, "end_frame": end, "start_s": start / 30, "end_s": end / 30}


def cut(index: int, frame: int, jump: float | None) -> dict:
    return {
        "index": index,
        "frame": frame,
        "time_s": frame / 30,
        "loudness_jump_lu": jump,
        "loudness_before_lufs": -31.2,
        "loudness_after_lufs": -31.2 + jump if jump is not None else None,
    }


def agent_note(video: Path, tmp_path: Path, findings_path: Path | None = None) -> dict:
    file = tmp_path / "agent.json"
    file.write_text(
        json.dumps(
            {
                "reviewer": "Reviewer A",
                "findings": [
                    {
                        "start": 2.0,
                        "end": 6.0,
                        "kind": "face_covered",
                        "severity": "medium",
                        "reason": "A card covers the face.",
                        "fix": "Move the card away from the face.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return note_from_file(video, file, findings_path=findings_path)


def test_exact_four_second_acceptance(prepared: dict[str, Path], tmp_path: Path, capsys) -> None:
    pack = load_pack(prepared["exact"])
    assert pack["cuts"] == []
    assert pack["video"]["frames"] == 240
    result = run_scan(prepared["exact"])
    findings = entries(result, "watch.still")
    assert len(findings) == 1 and findings[0]["kind"] == "slow"
    assert findings[0]["start_s"] == pytest.approx(2, abs=1 / 30)
    assert findings[0]["end_s"] == pytest.approx(6, abs=1 / 30)
    assert findings[0]["evidence"]["duration_s"] == pytest.approx(4, abs=1 / 30)
    agent_note(prepared["exact_video"], tmp_path)
    assert cli.main(["watch", "list", str(prepared["exact_video"])]) == 0
    text = capsys.readouterr().out
    assert "2 findings" in text
    assert "0:02.00 to 0:06.00 | slow | medium | machine" in text
    assert "0:02.00 to 0:06.00 | face_covered | medium | agent (Reviewer A)" in text
    assert default_findings_path(prepared["exact_video"]).exists()


def test_real_still_loudness_and_silence(prepared: dict[str, Path]) -> None:
    result = run_scan(prepared["silence"], dry_run=True)
    still = entries(result, "watch.still")
    assert len(still) == 1
    assert still[0]["start_s"] == pytest.approx(3, abs=1 / 30)
    assert still[0]["end_s"] == pytest.approx(7.5, abs=1 / 30)
    assert still[0]["severity"] == "medium"
    jumps = entries(result, "watch.cut_loudness")
    assert len(jumps) == 1
    assert jumps[0]["start_s"] == 7.5 and jumps[0]["end_s"] is None
    assert jumps[0]["kind"] == "audio" and jumps[0]["severity"] == "high"
    assert jumps[0]["evidence"]["cut"] == 2
    assert jumps[0]["evidence"]["jump_lu"] == pytest.approx(15, abs=0.8)
    pauses = entries(result, "watch.pause")
    assert len(pauses) == 1
    assert pauses[0]["start_s"] == pytest.approx(4, abs=0.05)
    assert pauses[0]["end_s"] == pytest.approx(6, abs=0.05)
    assert "of silence" in pauses[0]["reason"]
    assert result["rules"]["watch.pause"]["basis"] == "silence"


def test_scan_short_audio_silent_ending(tmp_path: Path) -> None:
    source = synth.watch_short_audio(tmp_path / "short-audio.mp4")
    out = tmp_path / "pack"
    build_pack(source, out=out, asr="none")
    target = tmp_path / "findings.json"
    result = run_scan(out, dry_run=True, findings_path=target)
    pauses = entries(result, "watch.pause")
    assert len(pauses) == 1
    assert pauses[0]["start_s"] == pytest.approx(7, abs=0.05)
    assert pauses[0]["end_s"] == 10
    assert pauses[0]["evidence"]["duration_s"] == pytest.approx(3, abs=0.05)
    assert "of silence" in pauses[0]["reason"]
    assert result["rules"]["watch.pause"]["basis"] == "silence"
    assert result["recorded"] is False and not target.exists()


def test_real_word_pauses(prepared: dict[str, Path]) -> None:
    result = run_scan(prepared["words"], dry_run=True)
    pauses = entries(result, "watch.pause")
    assert [(entry["start_s"], entry["end_s"]) for entry in pauses] == [(4, 6), (8.5, 10.5)]
    assert 'between "piece" and "Boundary"' in pauses[0]["reason"]
    assert 'after the last word "Boundary"' in pauses[1]["reason"]
    assert result["rules"]["watch.pause"]["basis"] == "words"


@pytest.mark.parametrize("name,remaining", [("mask", [(3, 7.1)]), ("region", []), ("hold", [])])
def test_real_graphics_and_holds(prepared: dict[str, Path], name: str, remaining: list) -> None:
    pack = load_pack(prepared[name])
    assert pack["graphics"][0]["kind"] == name
    result = run_scan(prepared[name], dry_run=True)
    still = entries(result, "watch.still")
    assert len(still) == len(remaining)
    for finding, (start, end) in zip(still, remaining, strict=True):
        assert finding["start_s"] == pytest.approx(start, abs=1 / 30)
        assert finding["end_s"] == pytest.approx(end, abs=1 / 30)
    assert bool(entries(result, "watch.pause")) == (name != "hold")


def test_subpixel_region_packs_and_scans_with_still_finding(
    prepared: dict[str, Path], tmp_path: Path
) -> None:
    video = prepared["exact_video"]
    ranges = synth.watch_ranges(
        tmp_path / "ranges.json",
        video,
        [],
        end_s=8,
        overlays=[{"start_in_output": 2, "duration": 4, "region": [0, 0, 0.0004, 20]}],
    )
    out = tmp_path / "pack"
    build_pack(video, edl=ranges, out=out, asr="none")
    assert load_pack(out)["graphics"][0]["rect"] == [0, 0, 0.0, 20]
    findings = entries(run_scan(out, dry_run=True), "watch.still")
    assert len(findings) == 1
    assert findings[0]["start_s"] == pytest.approx(2, abs=1 / 30)
    assert findings[0]["end_s"] == pytest.approx(6, abs=1 / 30)


def test_cut_splits_still_span() -> None:
    pack = pack_dict(still=[still_span(0, 240)], cuts=[cut(1, 120, None)])
    result = scan_pack(pack, words=None)
    assert [(f["start_s"], f["end_s"]) for f in result["findings"]] == [(0, 4), (4, 8)]
    assert all(f["severity"] == "medium" for f in result["findings"])
    pack["cuts"][0]["frame"] = 105
    assert [(f["start_s"], f["end_s"]) for f in scan_pack(pack, words=None)["findings"]] == [
        (3.5, 8)
    ]


@pytest.mark.parametrize(
    "rect,suppresses",
    [([0, 0, 10, 100], False), ([0, 0, 40, 100], True), ([0, 0, 20, 100], True)],
    ids=["five_percent", "twenty_percent", "ten_percent_boundary"],
)
def test_region_area_controls_still_suppression(rect: list[int], suppresses: bool) -> None:
    pack = pack_dict(
        still=[still_span(0, 240)],
        graphics=[{"kind": "region", "start_s": 0, "end_s": 8, "rect": rect}],
    )
    findings = entries(scan_pack(pack, words=None), "watch.still")
    assert bool(findings) is not suppresses
    if findings:
        assert (findings[0]["start_s"], findings[0]["end_s"]) == (0, 8)


@pytest.mark.parametrize("rect", [[0, 0, 0.0, 20], [0, 0, 20, 0.0], [0, 0, 0.0, 0.0]])
def test_zero_area_region_keeps_still_finding(rect: list[float]) -> None:
    pack = pack_dict(
        still=[still_span(0, 240)],
        graphics=[{"kind": "region", "start_s": 0, "end_s": 8, "rect": rect}],
    )
    findings = entries(scan_pack(pack, words=None), "watch.still")
    assert len(findings) == 1
    assert (findings[0]["start_s"], findings[0]["end_s"]) == (0, 8)


@pytest.mark.parametrize(
    "rect",
    [
        [0, 0, -1, 20], [0, 0, 20, -1], [0, 0, 0.0, -1], [0, 0, -1, 0.0],
        [0, 0, float("nan"), 20], [0, 0, 0.0, float("inf")],
        [float("-inf"), 0, 0.0, 20], [0, 0, True, 20], [0, 0, 20], None,
    ],
)
def test_invalid_region_rectangles_still_raise(rect: list[float] | None) -> None:
    pack = pack_dict(
        still=[still_span(0, 240)],
        graphics=[{"kind": "region", "start_s": 0, "end_s": 8, "rect": rect}],
    )
    with pytest.raises(ValidationError, match="graphics.rect"):
        scan_pack(pack, words=None)


@pytest.mark.parametrize(
    "rect,suppresses",
    [
        ([90, 0, 40, 100], False),
        ([0, 190, 100, 40], False),
        ([-30, 0, 40, 100], False),
        ([0, -30, 100, 40], False),
        ([100, 0, 40, 100], False),
        ([80, 0, 40, 200], True),
    ],
    ids=["right", "bottom", "left", "top", "outside", "clipped_twenty_percent"],
)
def test_region_area_is_clipped_to_frame(rect: list[int], suppresses: bool) -> None:
    pack = pack_dict(
        still=[still_span(0, 240)],
        graphics=[{"kind": "region", "start_s": 0, "end_s": 8, "rect": rect}],
    )
    assert bool(entries(scan_pack(pack, words=None), "watch.still")) is not suppresses


@pytest.mark.parametrize("kind", ["mask", "hold"])
def test_whole_picture_masks_and_holds_suppress_stillness(kind: str) -> None:
    pack = pack_dict(
        still=[still_span(0, 240)],
        graphics=[{"kind": kind, "start_s": 0, "end_s": 8, "rect": None}],
    )
    assert entries(scan_pack(pack, words=None), "watch.still") == []


def test_defaults_rounding_and_determinism() -> None:
    assert (STILL_S, PAUSE_S, LOUDNESS_JUMP_LU) == (4, 1, 8)
    assert GRAPHIC_MIN_AREA == 0.10
    pack = pack_dict(still=[still_span(90, 226)], cuts=[cut(2, 300, 15.1)])
    words = [Word("piece", 0, 2), Word("Boundary", 4.1, 12)]
    original, original_words = copy.deepcopy(pack), list(words)
    result = scan_pack(pack, words=words)
    assert result == scan_pack(pack, words=words)
    assert pack == original and words == original_words
    assert result["settings"] == {
        "still_s": 4, "pause_s": 1, "loudness_jump_lu": 8, "graphic_min_area": 0.10
    }
    assert entries(result, "watch.still")[0]["reason"] == (
        "The picture barely changes for 4.53 s, from 0:03.00 to 0:07.53, "
        "with no cut or large graphic."
    )
    assert entries(result, "watch.cut_loudness")[0]["reason"] == (
        "Speech after cut 2 is 15.1 LU louder than before it (-31.2 to -16.1 LUFS)."
    )
    assert entries(result, "watch.pause")[0]["reason"] == (
        'A 2.10 s pause between "piece" and "Boundary", from 0:02.00 to 0:04.10.'
    )
    assert [(f["start_s"], f["check"]) for f in result["findings"]] == sorted(
        (f["start_s"], f["check"]) for f in result["findings"]
    )
    for finding in result["findings"]:
        assert all(type(value) in (int, float) for value in finding["evidence"].values())
        assert chr(0x2013) not in finding["reason"] and chr(0x2014) not in finding["reason"]


@pytest.mark.parametrize(
    "length,threshold,severity",
    [(4, 4, "medium"), (3.99, 4, "medium"), (3.98, 4, None), (8, 4, "high"), (4, 2, "high")],
)
def test_still_threshold_tolerance_and_severity(length, threshold, severity) -> None:
    pack = pack_dict(
        still=[still_span(0, 360)],
        graphics=[{"kind": "mask", "start_s": length, "end_s": 12}],
    )
    result = entries(scan_pack(pack, words=None, still_s=threshold), "watch.still")
    assert len(result) == (0 if severity is None else 1)
    if severity:
        assert result[0]["severity"] == severity
        assert result[0]["evidence"]["threshold_s"] == threshold


@pytest.mark.parametrize(
    "jump,threshold,severity",
    [
        (None, 8, None),
        (7.99, 8, None),
        (8, 8, "medium"),
        (-8, 8, "medium"),
        (12, 8, "high"),
        (-12, 8, "high"),
        (6, 4, "high"),
    ],
)
def test_loudness_threshold_direction_and_severity(jump, threshold, severity) -> None:
    result = entries(
        scan_pack(pack_dict(cuts=[cut(1, 60, jump)]), words=None, loudness_jump_lu=threshold),
        "watch.cut_loudness",
    )
    assert len(result) == (0 if severity is None else 1)
    if severity:
        assert result[0]["severity"] == severity
        assert ("quieter" if jump < 0 else "louder") in result[0]["reason"]


@pytest.mark.parametrize(
    "length,threshold,severity",
    [(0.99, 1, None), (1, 1, "low"), (2, 1, "medium"), (1, 0.5, "medium")],
)
def test_pause_threshold_and_severity(length, threshold, severity) -> None:
    result = entries(
        scan_pack(
            pack_dict(silences=[{"start_s": 0, "end_s": length}]), words=None, pause_s=threshold
        ),
        "watch.pause",
    )
    assert len(result) == (0 if severity is None else 1)
    if severity:
        assert result[0]["severity"] == severity


def test_word_edges_overlaps_empty_and_unsorted() -> None:
    pack = pack_dict()
    words = [Word("last", 8, 10), Word("overlap", 2, 3), Word("first", 1, 5)]
    pauses = entries(scan_pack(pack, words=words), "watch.pause")
    assert [(f["start_s"], f["end_s"]) for f in pauses] == [(0, 1), (5, 8), (10, 12)]
    assert 'before the first word "first"' in pauses[0]["reason"]
    assert 'between "first" and "last"' in pauses[1]["reason"]
    assert 'after the last word "last"' in pauses[2]["reason"]
    assert entries(scan_pack(pack, words=[]), "watch.pause")[0]["end_s"] == 12


@pytest.mark.parametrize("start,end,severity", [(0.4, 1.4, "low"), (2.1, 4.1, "medium")])
def test_word_pause_inclusive_boundaries(start: float, end: float, severity: str) -> None:
    words = [Word("before", 0, start), Word("after", end, 12)]
    pauses = entries(scan_pack(pack_dict(), words=words), "watch.pause")
    assert len(pauses) == 1
    assert pauses[0]["severity"] == severity


def test_hold_subtraction_and_half_open_windows() -> None:
    graphics = [
        {"kind": "hold", "start_s": 4, "end_s": 6},
        {"kind": "hold", "start_s": 5, "end_s": 7},
        {"kind": "region", "start_s": 8, "end_s": 9, "rect": [0, 0, 40, 100]},
        {"kind": "mask", "start_s": 12, "end_s": 14},
        {"kind": "mask", "start_s": 2, "end_s": 2},
    ]
    pack = pack_dict(still=[still_span(0, 360)], graphics=graphics)
    result = scan_pack(pack, words=[Word("first", 0, 1), Word("last", 11, 12)])
    assert [(f["start_s"], f["end_s"]) for f in entries(result, "watch.still")] == [(0, 4)]
    assert [(f["start_s"], f["end_s"]) for f in entries(result, "watch.pause")] == [(1, 4), (7, 11)]
    # The same holds also trim the measured-silence fallback.
    pack["loudness"]["silences"] = [{"start_s": 1, "end_s": 11}]
    silences = entries(scan_pack(pack, words=None), "watch.pause")
    assert [(f["start_s"], f["end_s"]) for f in silences] == [(1, 4), (7, 11)]
    assert all("of silence" in f["reason"] for f in silences)


def test_skipped_rules_and_same_time_order() -> None:
    result = scan_pack(pack_dict(has_audio=False), words=None)
    assert result["findings"] == []
    assert result["rules"]["watch.still"]["status"] == "ran"
    for check in ("watch.cut_loudness", "watch.pause"):
        assert result["rules"][check]["status"] == "skipped"
        assert result["rules"][check]["reason"]
    pack = pack_dict(still=[still_span(0, 120)], cuts=[cut(1, 0, 8)])
    assert [f["check"] for f in scan_pack(pack, words=[])["findings"]] == [
        "watch.cut_loudness",
        "watch.pause",
        "watch.still",
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_loudness_measurements(value) -> None:
    pack = pack_dict(cuts=[cut(1, 60, 8)])
    pack["cuts"][0]["loudness_jump_lu"] = value
    with pytest.raises(ValidationError, match="cuts"):
        scan_pack(pack, words=None)
    pack["cuts"][0]["loudness_jump_lu"] = 8
    pack["cuts"][0]["loudness_after_lufs"] = value
    with pytest.raises(ValidationError, match="cuts"):
        scan_pack(pack, words=None)


@pytest.mark.parametrize("name", ["still_s", "pause_s", "loudness_jump_lu"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf"), True])
def test_invalid_thresholds(name: str, value: float, tmp_path: Path, capsys) -> None:
    with pytest.raises(ValidationError, match=name):
        scan_pack(pack_dict(), words=None, **{name: value})
    with pytest.raises(ValidationError, match=name):
        run_scan(tmp_path / "missing-pack", **{name: value})
    if value is not True:
        option = "--" + name.replace("_", "-")
        assert cli.main(["watch", "scan", str(tmp_path), f"{option}={value}", "--json"]) == 2
        assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_parameter"


def test_dry_run_no_writes_or_decoding(
    prepared: dict[str, Path], tmp_path: Path, monkeypatch
) -> None:
    original_run = ffmpeg.run

    def metadata_only(argv, *args, **kwargs):
        assert Path(argv[0]).name == "ffprobe"
        assert "-count_frames" not in argv
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", metadata_only)
    target = tmp_path / "findings.json"
    before = {p: p.read_bytes() for p in prepared["silence"].rglob("*") if p.is_file()}
    result = run_scan(prepared["silence"], dry_run=True, findings_path=target)
    assert result["path"] is None and result["recorded"] is False
    assert not target.exists()
    assert before == {p: p.read_bytes() for p in before}
    saved = run_scan(prepared["silence"], findings_path=target)
    assert saved["recorded"] is True
    assert saved["findings"] == result["findings"]


def test_rescan_replaces_machine_and_keeps_agent(prepared: dict[str, Path], tmp_path: Path) -> None:
    target = tmp_path / "findings.json"
    first = run_scan(prepared["silence"], findings_path=target)
    assert (first["added"], first["removed"], first["total"]) == (3, 0, 3)
    noted = agent_note(prepared["video"], tmp_path, target)
    agent = next(f for f in noted["findings"] if f["source"] == "agent")
    result = run_scan(prepared["silence"], findings_path=target)
    assert (result["added"], result["removed"], result["total"]) == (3, 3, 4)
    saved = load_findings(prepared["video"], findings_path=target)["findings"]
    assert [f for f in saved if f["source"] == "agent"] == [agent]
    assert all(f["reviewer"] is None for f in saved if f["source"] == "machine")
    empty = run_scan(
        prepared["silence"], findings_path=target, still_s=20, pause_s=20, loudness_jump_lu=40
    )
    assert (empty["added"], empty["removed"], empty["total"]) == (0, 3, 1)
    assert load_findings(prepared["video"], findings_path=target)["findings"] == [agent]


@pytest.mark.parametrize("missing", [False, True])
def test_stale_video_cli_refuses(
    prepared: dict[str, Path], tmp_path: Path, capsys, missing: bool
) -> None:
    directory = tmp_path / "pack"
    shutil.copytree(prepared["silence"], directory)
    video = tmp_path / "changed.mp4"
    shutil.copyfile(prepared["video"], video)
    pack = load_pack(directory)
    pack["video"]["path"] = str(video)
    (directory / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
    if missing:
        video.unlink()
    else:
        with video.open("ab") as stream:
            stream.write(b"changed")
    target = tmp_path / "findings.json"
    args = ["watch", "scan", str(directory), "--findings", str(target)]
    assert cli.main(args) == 3
    assert "watch pack again" in capsys.readouterr().err
    assert cli.main([*args, "--json", "--dry-run"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "stale_pack"
    assert not target.exists()


@pytest.mark.parametrize(
    "problem",
    [
        "edited_transcript", "edited_motion", "missing", "symlink", "pending",
        "pending_without_index", "dangling_pending", "unlisted_transcript", "unknown_output",
        "strip_symlink", "directory_output",
    ],
)
def test_unverified_pack_evidence_cli_refuses(
    prepared: dict[str, Path], tmp_path: Path, capsys, problem: str
) -> None:
    directory = tmp_path / "pack"
    shutil.copytree(prepared["words"], directory)
    target = tmp_path / "findings.json"
    run_scan(directory, findings_path=target)
    before = target.read_bytes()
    name = "transcript.json"
    if problem == "edited_transcript":
        file = directory / name
        data = json.loads(file.read_text())
        data["words"][1]["end"] = 5.0
        file.write_text(json.dumps(data), encoding="utf-8")
    elif problem == "edited_motion":
        name = "motion.json"
        file = directory / name
        data = json.loads(file.read_text())
        data["values"][0] += 1
        file.write_text(json.dumps(data), encoding="utf-8")
    elif problem == "missing":
        name = "motion.json"
        (directory / name).unlink()
    elif problem == "symlink":
        (directory / name).unlink()
        (directory / name).symlink_to(prepared["words"] / name)
    elif problem in {"pending", "pending_without_index", "dangling_pending"}:
        name = "pack.pending.json"
        if problem == "dangling_pending":
            (directory / name).symlink_to(tmp_path / "missing-journal.json")
        else:
            (directory / name).write_text("{}", encoding="utf-8")
        if problem == "pending_without_index":
            (directory / "pack.json").unlink()
    elif problem in {"unlisted_transcript", "unknown_output"}:
        pack = load_pack(directory)
        if problem == "unlisted_transcript":
            pack["outputs"] = [item for item in pack["outputs"] if item["path"] != name]
        else:
            name = "unknown.json"
            (directory / name).write_text("{}", encoding="utf-8")
            pack["outputs"].append({"path": name, "sha256": sha256_file(directory / name)})
        (directory / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
    elif problem == "strip_symlink":
        name = "strips/strip-001.png"
        shutil.rmtree(directory / "strips")
        (directory / "strips").symlink_to(prepared["words"] / "strips")
    else:
        name = "motion.json"
        (directory / name).unlink()
        (directory / name).mkdir()
    args = ["watch", "scan", str(directory), "--findings", str(target)]
    assert cli.main(args) == 3
    text = capsys.readouterr().err
    assert name in text and "Run yap-studio watch pack again" in text
    assert target.read_bytes() == before
    new_target = tmp_path / "new-findings.json"
    assert cli.main(["watch", "scan", str(directory), "--findings", str(new_target), "--json"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "stale_pack" and name in error["message"]
    assert "Run yap-studio watch pack again" in error["suggested_action"]
    assert not new_target.exists()
    with pytest.raises(OutputPathError) as raised:
        run_scan(directory, findings_path=target, dry_run=True)
    assert raised.value.code == "stale_pack"
    assert target.read_bytes() == before


def test_scan_reads_verified_transcript_bytes(
    prepared: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "pack"
    shutil.copytree(prepared["words"], directory)
    file = directory / "transcript.json"
    read_bytes = Path.read_bytes
    reads = 0

    def read_then_change(target: Path) -> bytes:
        nonlocal reads
        content = read_bytes(target)
        if target == file:
            reads += 1
            data = json.loads(content)
            data["words"][1]["end"] = 5.0
            file.write_text(json.dumps(data), encoding="utf-8")
        return content

    monkeypatch.setattr(Path, "read_bytes", read_then_change)
    result = run_scan(directory, dry_run=True)
    assert reads == 1
    assert [(f["start_s"], f["end_s"]) for f in entries(result, "watch.pause")] == [
        (4, 6), (8.5, 10.5)
    ]


@pytest.mark.parametrize("problem", ["stale", "malformed", "unsafe"])
def test_findings_refusals(prepared: dict[str, Path], tmp_path: Path, capsys, problem: str) -> None:
    target = tmp_path / "findings.json"
    if problem == "stale":
        run_scan(prepared["exact"], findings_path=target)
    elif problem == "malformed":
        target.write_text("{}", encoding="utf-8")
    else:
        target.symlink_to(prepared["video"])
    before = target.read_bytes()
    assert (
        cli.main(["watch", "scan", str(prepared["silence"]), "--findings", str(target), "--json"])
        == 3
    )
    code = {"stale": "stale_findings", "malformed": "invalid_findings", "unsafe": "unsafe_path"}
    assert json.loads(capsys.readouterr().err)["error"]["code"] == code[problem]
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "problem",
    ["schema", "missing", "negative", "nan", "text", "boolean", "beyond", "path", "duration"],
)
def test_invalid_packed_transcript(
    prepared: dict[str, Path], tmp_path: Path, capsys, problem: str
) -> None:
    directory = tmp_path / "pack"
    shutil.copytree(prepared["words"], directory)
    file = directory / "transcript.json"
    data = json.loads(file.read_text())
    if problem == "schema":
        data["schema"] = "wrong"
    elif problem == "missing":
        data.pop("words")
    elif problem == "text":
        data["words"][0]["text"] = None
    elif problem == "duration":
        data["duration_s"] = float("nan")
    elif problem == "path":
        pack = load_pack(directory)
        pack["transcript"]["file"] = "../words.json"
        (directory / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
    else:
        data["words"][0]["start"] = {
            "negative": -1,
            "nan": float("nan"),
            "boolean": True,
            "beyond": 20,
        }[problem]
    file.write_text(json.dumps(data), encoding="utf-8")
    # Keep the receipt current so malformed content reaches transcript validation.
    pack = load_pack(directory)
    for item in pack["outputs"]:
        if item["path"] == file.name:
            item["sha256"] = sha256_file(file)
    (directory / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
    target = tmp_path / "findings.json"
    expected = 3 if problem == "path" else 2
    assert cli.main(["watch", "scan", str(directory), "--findings", str(target)]) == expected
    assert capsys.readouterr().err
    assert not target.exists()


def test_cli_text_and_json(prepared: dict[str, Path], tmp_path: Path, capsys) -> None:
    args = [
        "watch",
        "scan",
        str(prepared["silence"]),
        "--findings",
        str(tmp_path / "findings.json"),
    ]
    assert cli.main([*args, "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "watch.still: ran, 1 finding" in text
    assert "watch.pause: ran on silence, no transcript" in text
    assert "0:07.50 | audio | high | machine" in text
    assert text.endswith("Dry run: nothing written.\n")
    assert cli.main([*args, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recorded"] is True and result["added"] == 3
    assert result["settings"] == {
        "still_s": 4, "pause_s": 1, "loudness_jump_lu": 8, "graphic_min_area": 0.10
    }
    assert cli.main(args) == 0
    assert "Recorded 3 machine findings" in capsys.readouterr().out


def test_cli_skipped_rules(prepared: dict[str, Path], tmp_path: Path, capsys) -> None:
    directory = tmp_path / "pack"
    shutil.copytree(prepared["silence"], directory)
    pack = load_pack(directory)
    pack["video"]["has_audio"] = False
    pack["loudness"].update(status="no_audio", silences=[])
    (directory / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
    assert cli.main(["watch", "scan", str(directory), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "watch.cut_loudness: skipped, video has no audio" in text
    assert "watch.pause: skipped, no transcript words or measured silences" in text
