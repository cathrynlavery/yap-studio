"""Calibration and structure checks on actual decoded synthetic media."""

from __future__ import annotations

import inspect
import json
import math
import re
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import audio, ffmpeg
from yap_studio.checks import structure
from yap_studio.checks.model import Thresholds
from yap_studio.edl.model import EditList, Hold, Segment, Source, Still
from yap_studio.plan import Plan
from yap_studio.report import _check_status
from yap_studio.stages.assemble_picture import CRF, PRESET

# The encoder's 1024-sample frames put the jump at the first frame starting at or
# after sample 96,000.
GAP_AT_SAMPLE = math.ceil(96_000 / 1024) * 1024


@pytest.fixture(scope="module")
def audio_gap(synth_media):
    """AAC renders of the same pink noise, with and without a 1600-sample timestamp jump."""
    gapped = synth_media.path("audio_gap")
    clean = synth_media.path("audio_clean_priming")
    return SimpleNamespace(
        gapped=gapped,
        clean=clean,
        long_picture=synth_media.path("audio_gap_long_picture"),
        reference=gapped.with_suffix(".wav"),
        clean_reference=clean.with_suffix(".wav"),
    )


def _single_range_plan(render: Path, seconds: float) -> Plan:
    edit = EditList(
        "ranges",
        render.with_suffix(".json"),
        {"main": Source("main", render)},
        [Segment(1, "main", Fraction(0), Fraction(str(seconds)))],
        None,
        [],
        [],
    )
    return Plan.build(edit, fps=Fraction(30), sample_rate=48000)


@pytest.fixture(scope="module")
def structure_cases(synth_media):
    source = synth_media.path("calib")
    edit = EditList(
        "yap",
        source.with_suffix(".json"),
        {"main": Source("main", source)},
        [
            Segment(i, "main", Fraction(str(a)), Fraction(str(b)))
            for i, (a, b) in enumerate(synth_media.spans, 1)
        ],
        Fraction(30),
        [],
        [],
    )
    plan = Plan.build(edit, fps=Fraction(30), sample_rate=48000)
    second_source = synth_media.cached(
        "t7-calib-copy",
        lambda out: shutil.copyfile(source, out),
        suffix=".mov",
    )
    different_source_segments = list(edit.segments)
    different_source_segments[3] = replace(different_source_segments[3], source_id="second")
    different_source_plan = Plan.build(
        replace(
            edit,
            sources={"main": edit.sources["main"], "second": Source("second", second_source)},
            segments=different_source_segments,
        ),
        fps=Fraction(30),
        sample_rate=48000,
    )
    clean = synth_media.path("clean")

    def filtered(out: Path, vf: str | None = None, af: str | None = None) -> None:
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clean),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
        ]
        command += ["-vf", vf, "-c:v", "libx264"] if vf else ["-c:v", "copy"]
        command += ["-af", af, "-c:a", "aac", "-b:a", "192k"] if af else ["-c:a", "copy"]
        ffmpeg.run([*command, str(out)], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)

    cut = plan.cuts[3]
    part = plan.segments[3]
    black = synth_media.cached(
        "t7-black-cut4",
        lambda out: filtered(
            out,
            vf="drawbox=color=black:t=fill:enable="
            f"'between(n,{cut.out_frame},{cut.out_frame + 9})'",
        ),
    )
    quiet = synth_media.cached(
        "t7-quiet-part4",
        lambda out: filtered(
            out,
            af=f"volume=0.5:enable='between(t,{float(part.out_time)},"
            f"{float(part.out_time + part.n_frames / plan.fps)})'",
        ),
    )
    quiet_half = synth_media.cached(
        "t7-quiet-half",
        lambda out: filtered(out, af="volume=0.5:enable='between(t,9.6,19.2)'"),
    )

    def no_audio(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x36:r=30:d=2",
                "-c:v",
                "libx264",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    def no_video(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clean),
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    paths = {
        "clean": clean,
        "calib": source,
        "black": black,
        "quiet": quiet,
        "quiet_half": quiet_half,
        "perclip": synth_media.path("perclip"),
        "freeze": synth_media.cached("t7-freeze-eof", no_audio),
        "audio_only": synth_media.cached("t7-audio-only", no_video, suffix=".m4a"),
    }
    # Reuse measurements when checking alternate plans and detected-part inputs.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audio, "decode", lru_cache(None)(audio.decode))
        patch.setattr(structure, "_decode_pass", lru_cache(None)(structure._decode_pass))
        patch.setattr(structure, "_packet_count", lru_cache(None)(structure._packet_count))
        results = {
            name: structure.check_structure(
                path,
                plan=plan
                if name
                in {
                    "clean",
                    "black",
                    "quiet",
                    "perclip",
                }
                else None,
            )
            for name, path in paths.items()
        }
        results["quiet_different_source"] = structure.check_structure(
            quiet,
            plan=different_source_plan,
        )
        direct = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-af",
                "ebur128",
                "-f",
                "null",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        loudness = float(re.findall(r"I:\s*([-\d.]+) LUFS", direct.stderr)[-1])
        (synth_media.root / "t7-structure-report.json").write_text(
            json.dumps(
                {
                    "calib_lufs": results["calib"].integrated_lufs,
                    "direct_lufs": loudness,
                    "delta_lu": results["calib"].integrated_lufs - loudness,
                    "quiet_parts_lufs": results["quiet"].parts_lufs,
                    "quiet_parts_cuts": [
                        f.cut for f in results["quiet"].findings if f.check == "loudness.parts"
                    ],
                },
                indent=2,
            )
        )
        yield paths, plan, results, loudness


def test_loudness_calibrates_against_direct_ffmpeg(structure_cases) -> None:
    _, _, results, direct_lufs = structure_cases
    result = results["calib"]
    assert result.integrated_lufs == pytest.approx(direct_lufs, abs=0.1)
    assert result.lra_lu is not None
    assert result.true_peak_dbtp is not None


def test_clean_counts_and_no_failures(structure_cases) -> None:
    _, plan, results, _ = structure_cases
    result = results["clean"]
    assert result.frames == plan.total_frames == 576
    assert result.sample_rate == 48000
    assert abs(result.samples - plan.total_samples) <= 1600
    assert result.video_duration_s == pytest.approx(19.2)
    assert result.audio_duration_s == pytest.approx(19.2, abs=1 / 30)
    assert not any(f.severity == "fail" for f in result.findings)
    assert not any(
        f.check in {"plan.frames", "plan.samples", "structure.av_length", "loudness.parts"}
        for f in result.findings
    )
    assert len(result.parts_lufs) == 8
    assert all(value is not None for value in result.parts_lufs)
    for result in results.values():
        json.dumps(result.to_dict(), allow_nan=False)


def test_av_length_compares_stream_end_times(synth_media) -> None:
    clean = synth_media.path("clean")

    def shifted_picture(out: Path, *, pad_audio: bool) -> None:
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-itsoffset",
            "0.5",
            "-i",
            str(clean),
            "-i",
            str(clean),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-af",
            "apad=pad_dur=0.5" if pad_audio else "anull",
            "-c:a",
            "aac",
            str(out),
        ]
        ffmpeg.run(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)

    matched = synth_media.cached(
        "structure-picture-offset-matched",
        lambda out: shifted_picture(out, pad_audio=True),
    )
    short = synth_media.cached(
        "structure-picture-offset-short-audio",
        lambda out: shifted_picture(out, pad_audio=False),
    )

    matched_info = ffmpeg.probe(matched, count_frames=False)
    assert matched_info.video_offset_s == pytest.approx(0.5, abs=0.01)
    assert not any(
        finding.check == "structure.av_length"
        for finding in structure.check_structure(matched).findings
    )

    short_result = structure.check_structure(short)
    finding = next(
        finding for finding in short_result.findings if finding.check == "structure.av_length"
    )
    assert "the sound ends 0.5" in finding.message
    assert "earlier than the picture" in finding.message
    assert finding.evidence["picture_end_s"] == pytest.approx(
        0.5 + short_result.video_duration_s
    )
    assert finding.evidence["sound_end_s"] == pytest.approx(short_result.audio_duration_s)


def test_plan_samples_excludes_late_audio_stream_offset(structure_cases, synth_media) -> None:
    paths, plan, _, _ = structure_cases

    def build(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(paths["clean"]),
                "-itsoffset",
                "0.5",
                "-i",
                str(paths["clean"]),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    render = synth_media.cached("structure-audio-offset", build)
    info = ffmpeg.probe(render, count_frames=False)
    result = structure.check_structure(render, plan=plan)
    offset_samples = round(info.audio_offset_s * result.sample_rate)
    assert info.audio_offset_s > 0.4
    assert result.samples - plan.total_samples > result.sample_rate / 3
    assert abs(result.samples - offset_samples - plan.total_samples) <= result.sample_rate / 30
    assert not any(finding.check == "plan.samples" for finding in result.findings)


def test_ten_black_frames_at_cut_four_fail(structure_cases) -> None:
    _, _, results, _ = structure_cases
    result = results["black"]
    failures = [f for f in result.findings if f.check == "picture.black"]
    assert len(failures) == 1
    assert failures[0].severity == "fail" and failures[0].cut == 4
    assert failures[0].time_s == pytest.approx(9.6)
    assert len(result.black) == 1
    assert result.black[0].start_s == pytest.approx(9.6)
    assert result.black[0].duration_s == pytest.approx(10 / 30, abs=0.001)


def test_one_black_frame_at_cut_with_late_video_fails(structure_cases, synth_media) -> None:
    paths, plan, _, _ = structure_cases
    cut = plan.cuts[3]

    def build(out: Path) -> None:
        black = out.with_name(out.stem + "-black.mp4")
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(paths["clean"]),
                "-map",
                "0:v:0",
                "-vf",
                f"drawbox=color=black:t=fill:enable='eq(n,{cut.out_frame})'",
                "-c:v",
                "libx264",
                "-an",
                str(black),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )
        try:
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-itsoffset",
                    "0.5",
                    "-i",
                    str(black),
                    "-i",
                    str(paths["clean"]),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c",
                    "copy",
                    str(out),
                ],
                timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
            )
        finally:
            black.unlink(missing_ok=True)

    render = synth_media.cached("structure-late-video-black-cut-v2", build)
    info = ffmpeg.probe(render, count_frames=False)
    result = structure.check_structure(render, plan=plan)
    failures = [finding for finding in result.findings if finding.check == "picture.black"]

    assert info.video_offset_s == pytest.approx(0.5, abs=0.01)
    assert len(failures) == 1
    assert failures[0].severity == "fail"
    assert failures[0].cut == cut.index
    assert failures[0].time_s == pytest.approx(0.5 + float(cut.out_time), abs=0.01)


def test_quiet_part_from_same_source_does_not_warn(structure_cases) -> None:
    _, _, results, _ = structure_cases
    result = results["quiet"]
    assert not any(f.check == "loudness.parts" for f in result.findings)
    assert result.parts_lufs[3] - results["clean"].parts_lufs[3] == pytest.approx(-6, abs=0.2)


def test_quiet_part_from_different_source_warns_at_both_cuts(structure_cases) -> None:
    _, _, results, _ = structure_cases
    findings = [
        f for f in results["quiet_different_source"].findings if f.check == "loudness.parts"
    ]
    assert [f.cut for f in findings] == [3, 4]
    assert "quieter" in findings[0].message
    assert "louder" in findings[1].message
    assert [(row.evidence["left_source"], row.evidence["right_source"]) for row in findings] == [
        ("main", "second"), ("second", "main"),
    ]
    for finding in findings:
        assert finding.severity == "warn"
        assert finding.evidence["reason"] == "different sources"
        assert abs(finding.evidence["difference_lu"]) == pytest.approx(6, abs=0.5)
        assert finding.suggested_action == "Normalize the loudness of this part to match the rest."


def test_perclip_audio_length_exposes_extra_samples(structure_cases) -> None:
    _, plan, results, _ = structure_cases
    result = results["perclip"]
    finding = next(f for f in result.findings if f.check == "plan.samples")
    assert finding.evidence["expected"] == plan.total_samples
    assert finding.evidence["actual"] == result.samples
    assert finding.evidence["difference"] == result.samples - plan.total_samples > 10000
    assert any(f.check == "structure.av_length" for f in result.findings)


def test_planless_short_detected_parts_do_not_warn(structure_cases) -> None:
    paths, plan, results, _ = structure_cases
    cuts = [float(c.out_time) for c in plan.cuts]
    result = structure.check_structure(paths["quiet"], cuts_s=cuts)
    assert result.parts_lufs == pytest.approx(results["quiet"].parts_lufs, abs=0.1)
    assert not any(f.check == "loudness.parts" for f in result.findings)
    assert not any(f.check.startswith("plan.") for f in result.findings)


def test_planless_long_parts_warn(structure_cases) -> None:
    paths, _, _, _ = structure_cases
    result = structure.check_structure(paths["quiet_half"], parts_s=[(0, 9.6), (9.6, 19.2)])
    findings = [f for f in result.findings if f.check == "loudness.parts"]
    assert len(findings) == 1
    assert findings[0].cut == 1
    assert findings[0].evidence["reason"] == "parts at least 3 s"
    assert findings[0].evidence["left_lufs"] == pytest.approx(result.parts_lufs[0])
    assert findings[0].evidence["right_lufs"] == pytest.approx(result.parts_lufs[1])
    assert findings[0].evidence["difference_lu"] == pytest.approx(-6, abs=0.5)


def test_open_ended_freeze_and_absent_streams(structure_cases) -> None:
    paths, _, results, _ = structure_cases
    frozen = results["freeze"]
    assert frozen.freeze == [structure.Interval(0.0, None, None)]
    assert frozen.samples is None and frozen.sample_rate is None
    assert frozen.integrated_lufs is None and frozen.parts_lufs == [None]
    assert next(f for f in frozen.findings if f.check == "picture.freeze").severity == "warn"
    skipped = next(f for f in frozen.findings if f.check == "structure.av_length")
    assert skipped.severity == "info" and skipped.evidence["has_audio"] is False
    at_cut = structure.check_structure(paths["freeze"], cuts_s=[1.0])
    fail = next(f for f in at_cut.findings if f.check == "picture.freeze")
    assert fail.severity == "fail" and fail.cut == 1
    sound = results["audio_only"]
    assert sound.frames is None and sound.video_duration_s is None
    assert sound.black == sound.freeze == []
    assert sound.samples > 0 and sound.integrated_lufs is not None
    skipped = next(f for f in sound.findings if f.check == "structure.av_length")
    assert skipped.evidence["has_video"] is False


def test_interval_parsers_handle_interleaved_lines_and_eof() -> None:
    log = """
black_start:0.25 black_end:0.75 black_duration:0.5
freeze_start: 1
unrelated filter output
freeze_duration: 2
freeze_end: 3
black_start:4 black_end:5 black_duration:1
freeze_start: 6.5
"""
    assert structure._intervals(log, "black") == [
        structure.Interval(0.25, 0.75, 0.5),
        structure.Interval(4, 5, 1),
    ]
    assert structure._intervals(log, "freeze") == [
        structure.Interval(1, 3, 2),
        structure.Interval(6.5, None, None),
    ]
    assert structure._intervals("freeze_start:1.2 freeze_end:1.7", "freeze") == [
        structure.Interval(1.2, 1.7, 0.5),
    ]


def test_final_summary_and_nonfinite_loudness() -> None:
    log = """
Summary:
  Integrated loudness:
    I: -70.0 LUFS
  Loudness range:
    LRA: 0.0 LU
  True peak:
    Peak: -inf dBFS
[ebur128] t: 0.399979 TARGET:-23 LUFS M: -14.0 S:-120.7 I: -14.0 LUFS
Summary:
  Integrated loudness:
    I: -14.1 LUFS
  Loudness range:
    LRA: 4.2 LU
  True peak:
    Peak: -0.5 dBFS
"""
    assert structure._loudness(log) == (-14.1, 4.2, -0.5)
    assert structure._momentary(log) == [(0.399979, -14)]
    assert structure._loudness(log.split("[ebur128]")[0]) == (-70, 0, None)
    assert structure._loudness("") == (None, None, None)


def test_part_gates_and_boundary_blocks() -> None:
    blocks = [
        (0.3, 0),
        *[(0.4 + i * 0.1, -20) for i in range(10)],
        *[(1.4 + i * 0.1, -40) for i in range(10)],
        (2.4, -80),
    ]
    assert structure._part_loudness(blocks, [(0, 3), (4, 5), (0, 0.99)], 48000, Thresholds()) == [
        pytest.approx(-20),
        None,
        None,
    ]
    # Blocks spanning the cut are excluded from the second part.
    assert structure._part_loudness([(1.1, -10), (1.4, -20)], [(1, 2)], 48000, Thresholds()) == [
        -20
    ]


@pytest.mark.parametrize("kind,minimum", [("black", 0.5), ("freeze", 1.0)])
def test_picture_severity_duration_and_cut_tolerance(kind: str, minimum: float) -> None:
    thresholds = Thresholds()
    check = f"picture.{kind}"
    assert (
        structure._picture_findings(
            check,
            [structure.Interval(2, 2.1, 0.1)],
            [],
            10,
            30,
            thresholds,
        )
        == []
    )
    finding = structure._picture_findings(
        check,
        [structure.Interval(2, 2 + minimum, minimum)],
        [],
        10,
        30,
        thresholds,
    )[0]
    assert finding.severity == "warn" and finding.cut is None
    for gap, expected in ((2 / 30, "fail"), (3 / 30, None)):
        findings = structure._picture_findings(
            check,
            [structure.Interval(2 + gap, 2.2, 0.1)],
            [(7, 2)],
            10,
            30,
            thresholds,
        )
        assert (findings[0].severity if findings else None) == expected


def test_declared_still_freeze_at_cut_is_information() -> None:
    interval = structure.Interval(2.0, 4.0, 2.0)
    cuts = [(1, 2.0)]
    stills = [Still("broll-diagram", 60, 120)]
    declared = structure._picture_findings(
        "picture.freeze", [interval], cuts, 5.0, 30, Thresholds(), stills=stills,
    )[0]
    assert declared.severity == "info"
    assert "insert broll-diagram" in declared.message
    assert declared.evidence["declared_still"] == {
        "id": "broll-diagram", "start_frame": 60, "end_frame": 120,
    }
    assert structure._picture_findings(
        "picture.freeze", [interval], cuts, 5.0, 30, Thresholds(),
    )[0].severity == "fail"
    partial = structure._picture_findings(
        "picture.freeze", [structure.Interval(1.5, 2.5, 1.0)], cuts,
        5.0, 30, Thresholds(), stills=stills,
    )[0]
    assert partial.severity == "fail"


@pytest.mark.parametrize("second_start", [50, 60])
def test_freeze_across_adjacent_stills_is_information(second_start: int) -> None:
    stills = [Still("broll-a", 0, 60), Still("broll-b", second_start, 120)]
    findings = structure._picture_findings(
        "picture.freeze", [structure.Interval(0.0, 4.0, 4.0)], [(1, 2.0)],
        5.0, 30, Thresholds(), stills=stills,
    )
    assert len(findings) == 1
    assert findings[0].severity == "info"
    assert "broll-a" in findings[0].message
    assert "broll-b" in findings[0].message
    assert [still["id"] for still in findings[0].evidence["declared_stills"]] == [
        "broll-a", "broll-b",
    ]
    assert findings[0].evidence["declared_span"] == {"start_s": 0.0, "end_s": 4.0}


def test_freeze_past_adjacent_stills_still_fails() -> None:
    findings = structure._picture_findings(
        "picture.freeze", [structure.Interval(0.0, 130 / 30, 130 / 30)],
        [(1, 2.0)], 5.0, 30, Thresholds(),
        stills=[Still("broll-a", 0, 60), Still("broll-b", 60, 120)],
    )
    assert findings[0].severity == "fail"


def test_freeze_across_gap_between_stills_still_fails() -> None:
    findings = structure._picture_findings(
        "picture.freeze", [structure.Interval(0.0, 4.0, 4.0)],
        [(1, 2.0)], 5.0, 30, Thresholds(),
        stills=[Still("broll-a", 0, 50), Still("broll-b", 60, 120)],
    )
    assert findings[0].severity == "fail"


def test_freeze_across_still_and_touching_edl_hold_is_information() -> None:
    hold = structure._HoldWindow(1, 60, 120, 2.0, 4.0, 96_000, 192_000)
    findings = structure._picture_findings(
        "picture.freeze", [structure.Interval(0.0, 4.0, 4.0)],
        [(1, 2.0)], 5.0, 30, Thresholds(), holds=[hold],
        stills=[Still("broll-a", 0, 60)],
    )
    assert findings[0].severity == "info"
    assert "broll-a" in findings[0].message
    assert findings[0].evidence["declared_hold"]["index"] == 1


def test_count_thresholds_and_loudness_targets(structure_cases, monkeypatch) -> None:
    paths, plan, results, _ = structure_cases
    # Use real decoded samples, changing only the expected plan and thresholds.
    clean = results["clean"]
    same_rate = replace(plan, total_samples=clean.samples)
    boundary = replace(same_rate, total_frames=clean.frames - 1, total_samples=clean.samples - 1600)
    result = structure.check_structure(
        paths["clean"],
        plan=boundary,
        thresholds=Thresholds().with_target(clean.integrated_lufs),
    )
    assert not any(
        f.check in {"plan.frames", "plan.samples", "loudness.integrated"} for f in result.findings
    )
    outside = replace(boundary, total_frames=clean.frames - 2, total_samples=clean.samples - 1601)
    result = structure.check_structure(paths["clean"], plan=outside)
    assert {"plan.frames", "plan.samples"}.issubset({f.check for f in result.findings})
    equivalent = replace(same_rate, sample_rate=24000, total_samples=clean.samples // 2)
    assert not any(
        f.check == "plan.samples"
        for f in structure.check_structure(paths["clean"], plan=equivalent).findings
    )
    monkeypatch.setattr(
        structure,
        "_decode_pass",
        lambda *a: (
            """Summary:
    I: -14.0 LUFS
    LRA: 0.0 LU
    True peak:
      Peak: -0.9 dBFS
    """
        ),
    )
    findings = structure.check_structure(paths["clean"]).findings
    assert any(f.check == "loudness.true_peak" for f in findings)
    monkeypatch.setattr(audio, "decode", lambda *a: (48000, np.zeros(48000)))
    shorter = structure.check_structure(paths["clean"], plan=plan)
    assert next(f for f in shorter.findings if f.check == "plan.samples").evidence["difference"] < 0
    assert "earlier" in next(
        f for f in shorter.findings if f.check == "structure.av_length"
    ).message


def test_audio_timestamp_gap_is_reported_where_it_happens(audio_gap, aac_tail) -> None:
    result = structure.check_structure(audio_gap.gapped)
    findings = [f for f in result.findings if f.check == "structure.audio_timestamps"]
    assert len(findings) == 1
    finding = findings[0]
    at = GAP_AT_SAMPLE / 48000
    assert finding.severity == "warn"
    assert finding.time_s == pytest.approx(at, abs=1e-9)
    assert "jump 1,600 samples (33.3 ms) ahead" in finding.message
    assert finding.message.startswith(f"At {structure.format_time(at)}, ")
    assert finding.evidence["count"] == 1
    assert finding.evidence["discontinuities"] == [
        {
            "time_s": pytest.approx(at, abs=1e-9),
            "samples": 1600,
            "ms": pytest.approx(1600 / 48),
            "kind": "gap",
            "frame": GAP_AT_SAMPLE // 1024,
            "decoded_position": GAP_AT_SAMPLE,
        }
    ]
    assert finding.evidence["decoded_samples"] == 4 * 48000 + aac_tail
    assert finding.evidence["timestamp_span_samples"] == 4 * 48000 + 1600 + aac_tail
    assert finding.evidence["gap_samples"] == 1600
    assert finding.evidence["overlap_samples"] == 0
    assert result.audio_timeline["discontinuities"][0]["samples"] == 1600
    json.dumps(result.to_dict(), allow_nan=False)


def test_clean_aac_with_priming_has_no_timestamp_break_and_no_offset(audio_gap, aac_tail) -> None:
    info = ffmpeg.probe(audio_gap.clean, count_frames=False)
    clock = audio.timeline(audio_gap.clean)
    assert info.audio_offset_s == 0
    assert clock.discontinuities == ()
    assert clock.decoded_samples == clock.span_samples == 4 * 48000 + aac_tail
    result = structure.check_structure(audio_gap.clean)
    assert not [f for f in result.findings if f.check == "structure.audio_timestamps"]
    # Priming is removed: decoded sample zero lines up with the PCM reference.
    sr, decoded = audio.decode(audio_gap.clean)
    _, reference = synth.read_wav(audio_gap.clean_reference)
    window = slice(sr, 2 * sr)
    lag_s, corr = audio.lag(reference[window], decoded[window], sr, max_lag_s=0.05)
    assert lag_s == 0
    assert corr > 0.9


def test_length_findings_separate_content_span_and_breaks(audio_gap, aac_tail) -> None:
    # The picture runs 0.5 s past the sound, and the plan asks for 4.5 s of both.
    render = audio_gap.long_picture
    longer = _single_range_plan(render, 4.5)
    result = structure.check_structure(render, plan=longer)
    samples = next(f for f in result.findings if f.check == "plan.samples")
    length = next(f for f in result.findings if f.check == "structure.av_length")
    for finding in (samples, length):
        assert finding.evidence["decoded_samples"] == 4 * 48000 + aac_tail
        assert finding.evidence["timestamp_span_samples"] == 4 * 48000 + 1600 + aac_tail
        assert finding.evidence["gap_samples"] == 1600
        assert finding.evidence["timestamp_breaks"] == 1
        assert "structure.audio_timestamps" in finding.evidence["note"]
        assert finding.evidence["filled_samples"] == 1600
    # The sample count is the content; the sound end is on the container clock.
    assert samples.evidence["actual"] == 4 * 48000 + aac_tail
    assert length.evidence["sound_end_s"] == pytest.approx((4 * 48000 + 1600 + aac_tail) / 48000)
    assert result.samples == 4 * 48000 + 1600 + aac_tail
    clean = structure.check_structure(audio_gap.clean, plan=longer)
    clean_samples = next(f for f in clean.findings if f.check == "plan.samples")
    assert "note" not in clean_samples.evidence
    assert clean_samples.evidence["timestamp_breaks"] == 0


@pytest.fixture(scope="module")
def hold_renders(synth_media):
    """Two seconds of motion and noise, then a 1.5 s declared hold."""
    delivery = ("-crf", str(CRF), "-preset", PRESET)
    variants = {
        "held": {},
        "moving": {"picture": "moving"},
        "noisy": {"sound": "noise"},
        "short": {"sound": "short"},
        "keyframe-delivery": {"keyframe": 87, "x264": delivery},
        "early-keyframe-delivery": {"keyframe": 66, "x264": delivery},
        "shifted-keyframe-delivery": {
            "keyframe": 87, "x264": delivery, "output_ts_offset": 1.5,
        },
        **{
            f"{picture}-keyframe-delivery": {
                "picture": picture, "keyframe": 87, "change_frame": 87, "x264": delivery,
            }
            for picture in ("box", "cut", "resumes", "tint")
        },
    }
    return {
        name: synth_media.cached(
            f"structure-hold-{name}",
            lambda out, options=options: synth.hold_render(out, **options),
        )
        for name, options in variants.items()
    }


def _hold_plan(render: Path, *, declared: bool = True) -> Plan:
    edit = EditList(
        "ranges",
        render.with_suffix(".json"),
        {"main": Source("main", render)},
        [Segment(1, "main", Fraction(0), Fraction(2))],
        None,
        [],
        [],
        holds=[Hold(1, 60, 105, "silence")] if declared else [],
    )
    return Plan.build(edit, fps=Fraction(30), sample_rate=48000)


_HOLD_CHECKS = {"plan.frames", "plan.samples", "plan.holds", "structure.av_length"}


def _keyframe_hold_note(result):
    holds = [f for f in result.findings if f.check == "plan.holds"]
    assert [f.severity for f in holds] == ["info"]
    assert holds[0].evidence["still_picture"] is True
    steps = holds[0].evidence["keyframe_steps"]
    assert steps and all(step["accepted"] for step in steps)
    assert _check_status("plan.holds", [f.to_dict() for f in result.findings],
                         {"plan.holds": {"ran": True}}) == {
        "ran": True, "status": "pass", "reason": None,
    }
    return holds[0]


def test_freeze_inside_a_declared_hold_is_information(hold_renders) -> None:
    render = hold_renders["held"]
    plan = _hold_plan(render)
    assert plan.total_frames == 105
    result = structure.check_structure(render, plan=plan)
    freezes = [f for f in result.findings if f.check == "picture.freeze"]
    assert [f.severity for f in freezes] == ["info"]
    assert "as the edit list declares" in freezes[0].message
    assert freezes[0].evidence["declared_hold"]["index"] == 1
    assert freezes[0].suggested_action is None
    assert not [
        f for f in result.findings if f.check in _HOLD_CHECKS and f.severity != "info"
    ], result.findings


def test_undeclared_freeze_still_warns(hold_renders) -> None:
    render = hold_renders["held"]
    for plan in (None, _hold_plan(render, declared=False)):
        result = structure.check_structure(render, plan=plan)
        freeze = next(f for f in result.findings if f.check == "picture.freeze")
        assert freeze.severity == "warn"
        assert "declared_hold" not in freeze.evidence
        assert not [f for f in result.findings if f.check == "plan.holds"]


def test_declared_hold_that_keeps_moving_warns(hold_renders) -> None:
    render = hold_renders["moving"]
    result = structure.check_structure(render, plan=_hold_plan(render))
    holds = [f for f in result.findings if f.check == "plan.holds"]
    assert [f.severity for f in holds] == ["warn"]
    assert "the picture keeps moving" in holds[0].message
    assert holds[0].time_s == pytest.approx(2.0)
    assert holds[0].evidence["still_picture"] is False
    assert holds[0].evidence["hold"]["start_frame"] == 60


def test_declared_silent_hold_with_sound_warns(hold_renders) -> None:
    render = hold_renders["noisy"]
    result = structure.check_structure(render, plan=_hold_plan(render))
    holds = [f for f in result.findings if f.check == "plan.holds"]
    assert [f.severity for f in holds] == ["warn"]
    assert "the sound peaks at" in holds[0].message
    assert holds[0].evidence["peak_dbfs"] > -90
    assert holds[0].evidence["silence_limit_dbfs"] == -90
    assert holds[0].evidence["still_picture"] is True
    freeze = next(f for f in result.findings if f.check == "picture.freeze")
    assert freeze.severity == "info"
    quiet = structure.check_structure(hold_renders["held"], plan=_hold_plan(hold_renders["held"]))
    assert not [f for f in quiet.findings if f.check == "plan.holds"]


def test_audio_shortfall_inside_a_silent_hold_is_information(hold_renders) -> None:
    render = hold_renders["short"]
    result = structure.check_structure(render, plan=_hold_plan(render))
    samples = next(f for f in result.findings if f.check == "plan.samples")
    length = next(f for f in result.findings if f.check == "structure.av_length")
    for finding in (samples, length):
        assert finding.severity == "info"
        assert "inside the declared silent hold" in finding.message
        assert finding.suggested_action is None
        assert finding.evidence["declared_hold"]["index"] == 1
    assert samples.evidence["difference"] < -48000
    assert not [f for f in result.findings if f.check == "plan.holds"]
    # Without the declared hold the same file still warns.
    plain = structure.check_structure(render)
    assert next(f for f in plain.findings if f.check == "structure.av_length").severity == "warn"


def test_keyframe_inside_a_declared_hold_is_still(hold_renders) -> None:
    render = hold_renders["keyframe-delivery"]
    plan = _hold_plan(render)
    result = structure.check_structure(render, plan=plan)
    assert any(interval.end_s == pytest.approx(2.9) for interval in result.freeze)
    finding = _keyframe_hold_note(result)
    steps = finding.evidence["keyframe_steps"]
    assert len(steps) == 1
    assert steps[0]["time_s"] == pytest.approx(2.9)
    assert steps[0]["mafd_db"] <= -45
    assert steps[0]["tile_change"] <= 12


def test_keyframe_early_in_a_declared_hold_is_still(hold_renders) -> None:
    render = hold_renders["early-keyframe-delivery"]
    result = structure.check_structure(render, plan=_hold_plan(render))
    assert result.freeze and all(interval.start_s > 2.0 for interval in result.freeze)
    _keyframe_hold_note(result)


def test_box_that_appears_at_a_keyframe_in_a_hold_warns(hold_renders) -> None:
    render = hold_renders["box-keyframe-delivery"]
    result = structure.check_structure(render, plan=_hold_plan(render))
    holds = [f for f in result.findings if f.check == "plan.holds"]
    assert [f.severity for f in holds] == ["warn"]
    assert "the picture keeps moving" in holds[0].message
    assert holds[0].evidence["freeze_intervals"] == [
        {"start_s": i.start_s, "end_s": i.end_s, "duration_s": i.duration_s}
        for i in result.freeze
    ]
    assert len(holds[0].evidence["hold_segments"]) == 2
    step = holds[0].evidence["keyframe_steps"][0]
    assert step["time_s"] == pytest.approx(2.9)
    assert step["accepted"] is False
    assert step["tile_change"] > 12


def test_colour_that_appears_at_a_keyframe_in_a_hold_warns(hold_renders) -> None:
    render = hold_renders["tint-keyframe-delivery"]
    raw = ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-ss", "2.850000", "-i", str(render),
        "-map", "0:v:0", "-frames:v", "2", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
    ], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, binary=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(2, 160 * 90 * 3 // 2).astype(np.int16)
    change = frames[1] - frames[0]
    assert np.mean(np.abs(change[:160 * 90].reshape(90, 160)[36:54, 48:112])) < 2
    chroma = change[160 * 90:].reshape(2, 45, 80)[:, 18:27, 24:56]
    assert np.mean(chroma[0]) > 20
    assert np.mean(chroma[1]) < -20
    result = structure.check_structure(render, plan=_hold_plan(render))
    holds = [f for f in result.findings if f.check == "plan.holds"]
    assert [f.severity for f in holds] == ["warn"]
    assert "the picture keeps moving" in holds[0].message
    step = holds[0].evidence["keyframe_steps"][0]
    assert step["accepted"] is False
    assert step["tile_change"] > 12


def test_cut_to_another_still_at_a_keyframe_warns(hold_renders) -> None:
    render = hold_renders["cut-keyframe-delivery"]
    result = structure.check_structure(render, plan=_hold_plan(render))
    holds = [f for f in result.findings if f.check == "plan.holds"]
    assert [f.severity for f in holds] == ["warn"]
    assert "the picture keeps moving" in holds[0].message
    assert holds[0].evidence["keyframe_steps"][0]["accepted"] is False


def test_hold_that_moves_after_a_keyframe_warns(hold_renders) -> None:
    render = hold_renders["resumes-keyframe-delivery"]
    result = structure.check_structure(render, plan=_hold_plan(render))
    holds = [f for f in result.findings if f.check == "plan.holds"]
    assert [f.severity for f in holds] == ["warn"]
    assert "the picture keeps moving" in holds[0].message


def test_picture_freeze_is_unchanged_by_the_hold_pass(hold_renders) -> None:
    render = hold_renders["keyframe-delivery"]
    plan = _hold_plan(render)
    result = structure.check_structure(render, plan=plan)
    direct = ffmpeg.run([
        "ffmpeg", "-nostdin", "-i", str(render), "-map", "0:v:0",
        "-vf", "freezedetect=n=-60dB:d=0.5", "-f", "null", "-",
    ], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
    intervals = structure._intervals(direct.stderr, "freeze")
    assert result.freeze == intervals
    info = ffmpeg.probe(render)
    expected = structure._picture_findings(
        "picture.freeze", intervals,
        [(cut.index, float(cut.out_time) + info.video_offset_s) for cut in plan.cuts],
        info.video_offset_s + result.video_duration_s, float(info.fps), Thresholds(),
        structure._hold_windows(plan, info, info.sample_rate),
        stills=plan.stills, picture_offset_s=info.video_offset_s,
    )
    assert [f for f in result.findings if f.check == "picture.freeze"] == expected


def test_decode_pass_is_unchanged(hold_renders, monkeypatch) -> None:
    commands = []
    info = replace(ffmpeg.probe(hold_renders["held"]), fps=Fraction(30))
    monkeypatch.setattr(structure.ffmpeg, "run", lambda cmd, **kw: (
        commands.append(cmd) or SimpleNamespace(stderr="")
    ))
    structure._decode_pass(Path("render.mp4"), info)
    original = "blackdetect=d=0.033333333:pix_th=0.10,freezedetect=n=-60dB:d=0.5"
    assert commands == [[
        "ffmpeg", "-loglevel", "verbose", "-nostats", "-i", "render.mp4",
        "-map", "0:0", "-map", "0:1", "-vf", original,
        "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-",
    ]]
    assert list(inspect.signature(structure._decode_pass).parameters) == ["render", "info"]


def test_hold_segments_are_on_the_main_clock(hold_renders) -> None:
    render = hold_renders["keyframe-delivery"]
    info = ffmpeg.probe(render)
    hold = structure._hold_windows(_hold_plan(render), info, info.sample_rate)[0]
    segments = structure._hold_segments(render, info, hold, 3.5)
    assert len(segments) == 2
    assert segments[0].start_s <= 2.0 + 1 / 30
    assert segments[0].end_s == pytest.approx(2.9, abs=1e-6)
    assert segments[1].start_s == pytest.approx(2.9, abs=1e-6)
    assert segments[1].end_s == pytest.approx(3.5)


def test_keyframe_hold_with_a_shifted_start_is_still(hold_renders) -> None:
    render = hold_renders["shifted-keyframe-delivery"]
    info = ffmpeg.probe(render)
    assert info.format_start_s == pytest.approx(1.5, abs=0.03)
    result = structure.check_structure(render, plan=_hold_plan(render))
    assert any(interval.end_s == pytest.approx(info.video_offset_s + 2.9, abs=1e-4)
               for interval in result.freeze)
    _keyframe_hold_note(result)


@pytest.mark.parametrize("picture_end", [None, 3.5])
def test_hold_segments_limit_the_decode_and_close_open_ends(monkeypatch, picture_end) -> None:
    calls = []

    def decode(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stderr="freeze_start: 0.966667\n")

    monkeypatch.setattr(structure.ffmpeg, "run", decode)
    info = SimpleNamespace(fps=Fraction(30), video_index=2)
    hold = structure._HoldWindow(1, 60, 105, 2.0, 3.5, 96_000, 168_000)
    segments = structure._hold_segments(Path("render.mp4"), info, hold, picture_end)
    assert calls == [[
        "ffmpeg", "-nostdin", "-loglevel", "info", "-nostats", "-ss", "1.933333",
        "-t", "1.633333", "-i", "render.mp4", "-map", "0:2",
        "-vf", "freezedetect=n=-60dB:d=0.050000000", "-f", "null", "-",
    ]]
    assert len(segments) == 1
    assert segments[0].start_s == pytest.approx(2.9, abs=1e-6)
    end = 3.5 if picture_end is not None else 3.5 + 2 / 30
    assert segments[0].end_s == pytest.approx(end)
    assert segments[0].duration_s == pytest.approx(end - segments[0].start_s)


@pytest.mark.parametrize(("fps", "video_index"), [(None, 0), (Fraction(30), None)])
def test_hold_segments_need_fps_and_video(monkeypatch, fps, video_index) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("A hold without a video clock must not decode.")

    monkeypatch.setattr(structure.ffmpeg, "run", unexpected)
    hold = structure._HoldWindow(1, 60, 105, 2.0, 3.5, 96_000, 168_000)
    assert structure._hold_segments(
        Path("render.mp4"), SimpleNamespace(fps=fps, video_index=video_index), hold, 3.5,
    ) == []


def test_hold_measurements_are_lazy_and_keyframes_are_probed_once(
    hold_renders, monkeypatch,
) -> None:
    render = hold_renders["keyframe-delivery"]
    plan = _hold_plan(render)
    info = ffmpeg.probe(render)
    hold = structure._hold_windows(plan, info, info.sample_rate)[0]
    monkeypatch.setattr(structure, "_hold_windows", lambda *a: [hold, replace(hold, index=2)])
    stderr = "freeze_start: 1.96667\n"
    segment_calls, probe_calls, step_calls = [], [], []

    def segments(*args):
        segment_calls.append(args)
        return [structure.Interval(1.96667, 2.9, 0.933333), structure.Interval(2.9, 3.5, 0.6)]

    def keyframes(*args):
        probe_calls.append(args)
        return [2.9]

    def measure(*args):
        step_calls.append(args)
        return structure._KeyframeStep(args[-1], -54.0, 1.0, True)

    monkeypatch.setattr(structure, "_decode_pass", lambda *a: stderr)
    monkeypatch.setattr(structure, "_hold_segments", segments)
    monkeypatch.setattr(structure, "_keyframe_times", keyframes)
    monkeypatch.setattr(structure, "_keyframe_step", measure)
    result = structure.check_structure(render, plan=plan)
    assert not [f for f in result.findings if f.check == "plan.holds" and f.severity == "warn"]
    assert segment_calls == probe_calls == step_calls == []
    stderr += "freeze_duration: 0.933333\nfreeze_end: 2.9\n"
    result = structure.check_structure(render, plan=plan)
    assert result.freeze == [structure.Interval(1.96667, 2.9, 0.933333)]
    assert not [f for f in result.findings if f.check == "plan.holds" and f.severity == "warn"]
    assert len(segment_calls) == 2
    assert len(probe_calls) == 1
    assert len(step_calls) == 1


@pytest.mark.parametrize("breaks", [(2.9,), (2.4, 2.7), (2.3, 2.6, 2.9)])
def test_accepted_keyframe_steps_are_reported_as_information(breaks) -> None:
    hold = structure._HoldWindow(1, 60, 105, 2.0, 3.5, 96_000, 168_000)
    edges = [2.0, *breaks, 3.5]
    segments = [structure.Interval(start, end, end - start)
                for start, end in zip(edges, edges[1:], strict=False)]
    findings = structure._hold_findings(
        [hold], [], 3.5, None, None, 30, Thresholds(), lambda hold: segments,
        lambda: breaks, lambda time: structure._KeyframeStep(time, -54.0, 1.0, True),
    )
    assert [f.severity for f in findings] == ["info"]
    times = [structure.format_time(time) for time in breaks]
    label = times[0] if len(times) == 1 else ", ".join(times[:-1]) + " and " + times[-1]
    noun = "keyframe" if len(times) == 1 else "keyframes"
    assert findings[0].message == (
        "At 0:02.0, hold 1 is still. The encoder redraws the picture "
        f"at the {noun} at {label}, a small step spread across the frame."
    )
    assert findings[0].evidence["still_picture"] is True
    assert findings[0].evidence["hold_segments"]
    assert len(findings[0].evidence["keyframe_steps"]) == len(breaks)
    assert all(step["accepted"] for step in findings[0].evidence["keyframe_steps"])
    assert _check_status("plan.holds", [f.to_dict() for f in findings], {})["status"] == "pass"


@pytest.mark.parametrize(
    ("segments", "keyframes", "measured", "picture_end", "expected", "step_count"),
    [
        ([(2.0, 2.9), (2.9, 3.5)], [2.9], True, 3.5, True, 1),
        ([(2.0, 2.9), (2.9, 3.5)], [2.8], True, 3.5, False, 0),
        ([(2.0, 2.9), (3.0, 3.5)], [2.9], True, 3.5, False, 0),
        ([(2.0, 2.9), (2.9, 3.5)], [2.9], False, 3.5, False, 1),
        ([(2.0, 2.9), (2.9, 3.5)], [2.9], None, 3.5, False, 0),
        ([(2.0, 2.5), (2.5, 2.9), (2.9, 3.5)], [2.5, 2.9], True, 3.5, True, 2),
        ([(2.0, 2.9), (2.9, None)], [2.9], True, 3.5, True, 1),
        ([(2.0, 2.9), (2.9, None)], [2.9], True, 3.0, False, 1),
        ([(2.1, 2.9), (2.9, 3.5)], [2.9], True, 3.5, False, 0),
        ([(1.0, 1.9), (2.1, 3.5)], [1.9], True, 3.5, False, 0),
    ],
    ids=["accepted", "off-keyframe", "gap", "rejected", "unmeasured", "two-breaks",
         "open-end", "short-open-end", "late-start", "ends-before-hold"],
)
def test_still_across_keyframes(segments, keyframes, measured, picture_end, expected,
                               step_count) -> None:
    hold = structure._HoldWindow(1, 60, 105, 2.0, 3.5, 96_000, 168_000)
    calls = []

    def measure(time):
        calls.append(time)
        return (structure._KeyframeStep(time, -54.0, 1.0, measured)
                if measured is not None else None)

    still, steps = structure._still_across_keyframes(
        hold, [structure.Interval(start, end, None) for start, end in segments],
        picture_end, 1 / 30, 30, lambda: keyframes, measure,
    )
    assert still is expected
    assert len(steps) == step_count
    assert calls == ([step.time_s for step in steps] if measured is not None else [2.9])


def test_keyframe_times_use_packet_pts_on_the_format_clock(monkeypatch) -> None:
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"packets": [
            {"pts_time": "10.5", "flags": "K__"},
            {"pts_time": "10.6", "flags": "___"},
            {"pts_time": "N/A", "flags": "K__"},
            {"flags": "K__"},
            {"pts_time": "13.4", "flags": "K_D"},
        ]}))

    monkeypatch.setattr(structure.ffmpeg, "run", probe)
    info = SimpleNamespace(video_index=2, format_start_s=10.5)
    assert structure._keyframe_times(Path("render.mp4"), info) == pytest.approx([0.0, 2.9])
    assert calls == [[
        "ffprobe", "-v", "error", "-select_streams", "2", "-show_entries",
        "packet=pts_time,flags", "-of", "json", "render.mp4",
    ]]


def _step_output(raw: bytes, times, *, binary_stderr: bool = True):
    stderr = "\n".join(
        f"[Parsed_showinfo_0 @ 0x1] n: {index} pts: 0 pts_time:{time:.9f}"
        for index, time in enumerate(times)
    )
    return SimpleNamespace(stdout=raw, stderr=stderr.encode() if binary_stderr else stderr)


@pytest.mark.parametrize("short", [False, True])
def test_keyframe_step_measures_all_planes_and_partial_tiles(monkeypatch, short) -> None:
    width, height = 35, 33
    samples = width * height + 2 * math.ceil(width / 2) * math.ceil(height / 2)
    before = np.full(samples, 100, dtype=np.uint8)
    after = before.copy()
    after[:width * height].reshape(height, width)[32:, 32:] += 15
    after[width * height:] += 2
    raw = before.tobytes() + after.tobytes()
    calls = []

    def decode(command, **kwargs):
        calls.append(command)
        assert kwargs["binary"] is True
        return _step_output(raw[:-1] if short else raw, [0.216667, 0.25])

    monkeypatch.setattr(structure.ffmpeg, "run", decode)
    info = SimpleNamespace(fps=Fraction(30), width=width, height=height, video_index=2)
    step = structure._keyframe_step(Path("render.mp4"), info, 2.9)
    assert calls == [[
        "ffmpeg", "-nostdin", "-loglevel", "info", "-nostats", "-ss", "2.650000",
        "-t", "0.266667", "-i", "render.mp4", "-map", "0:2", "-vf", "showinfo",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
    ]]
    if short:
        assert step is None
    else:
        assert step.mafd_db == pytest.approx(20 * math.log10(
            (15 * 3 + 2 * (samples - width * height)) / samples / 256
        ))
        assert step.tile_change == 15
        assert step.accepted is False


def test_identical_keyframe_step_has_json_safe_evidence(monkeypatch) -> None:
    monkeypatch.setattr(structure.ffmpeg, "run", lambda *a, **kw: _step_output(
        bytes(2 * (32 * 32 + 2 * 16 * 16)), [0.216667, 0.25],
    ))
    info = SimpleNamespace(fps=Fraction(30), width=32, height=32, video_index=0)
    step = structure._keyframe_step(Path("render.mp4"), info, 2.9)
    assert step.accepted is True
    assert step.mafd_db == -math.inf
    assert step.tile_change == 0.0
    assert step.to_dict()["mafd_db"] is None
    json.dumps(step.to_dict(), allow_nan=False)


@pytest.mark.parametrize("plane", [1, 2], ids=["u", "v"])
def test_keyframe_step_rejects_a_change_in_a_partial_chroma_tile(monkeypatch, plane) -> None:
    width, height = 35, 33
    chroma_samples = 18 * 17
    samples = width * height + 2 * chroma_samples
    before = np.full(samples, 100, dtype=np.uint8)
    after = before.copy()
    offset = width * height + (plane - 1) * chroma_samples
    after[offset:offset + chroma_samples].reshape(17, 18)[16:, 16:] += 60
    assert np.array_equal(before[:width * height], after[:width * height])
    monkeypatch.setattr(structure.ffmpeg, "run", lambda *a, **kw: _step_output(
        before.tobytes() + after.tobytes(), [0.216667, 0.25],
    ))
    info = SimpleNamespace(fps=Fraction(30), width=width, height=height, video_index=0)
    step = structure._keyframe_step(Path("render.mp4"), info, 2.9)
    assert step.mafd_db < -45
    assert step.accepted is False
    assert step.tile_change == 60


def test_keyframe_step_compares_the_frame_before_an_irregular_timestamp(monkeypatch) -> None:
    samples = 32 * 32 + 2 * 16 * 16
    before = bytes([100]) * samples
    keyframe = bytes([140]) * samples
    monkeypatch.setattr(structure.ffmpeg, "run", lambda *a, **kw: _step_output(
        before + keyframe + keyframe, [0.05, 0.25, 0.26], binary_stderr=False,
    ))
    info = SimpleNamespace(fps=Fraction(30), width=32, height=32, video_index=0)
    step = structure._keyframe_step(Path("render.mp4"), info, 2.9)
    assert step is not None
    assert step.time_s == 2.9
    assert step.mafd_db == pytest.approx(20 * math.log10(40 / 256))
    assert step.tile_change == 40
    assert step.accepted is False


@pytest.mark.parametrize("times", [(0.25, 0.26), (0.05, 0.06)], ids=["first-frame", "no-match"])
def test_keyframe_step_needs_a_keyframe_with_a_preceding_frame(monkeypatch, times) -> None:
    monkeypatch.setattr(structure.ffmpeg, "run", lambda *a, **kw: _step_output(
        bytes(2 * (32 * 32 + 2 * 16 * 16)), times,
    ))
    info = SimpleNamespace(fps=Fraction(30), width=32, height=32, video_index=0)
    assert structure._keyframe_step(Path("render.mp4"), info, 2.9) is None


def test_keyframe_step_requires_matching_frame_bytes_and_timestamps(monkeypatch) -> None:
    monkeypatch.setattr(structure.ffmpeg, "run", lambda *a, **kw: _step_output(
        bytes(2 * (32 * 32 + 2 * 16 * 16)), [0.05, 0.25, 0.26],
    ))
    info = SimpleNamespace(fps=Fraction(30), width=32, height=32, video_index=0)
    assert structure._keyframe_step(Path("render.mp4"), info, 2.9) is None
