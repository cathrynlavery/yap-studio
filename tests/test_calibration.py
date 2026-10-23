"""Synthetic finished videos that calibrate verify against isolated known faults.

Join faults are planted at cut 3, frame 216 (7.2 s), of the clean render.
Sound-only edits copy picture; picture-only edits copy sound.

Fault                   Intended check   Outcome Collateral warnings
Clean control           none             pass    none
Incoming frame missing  join.frames      fail    sync.av_offset, sync.drift
Outgoing frame repeated join.frames      fail    sync.av_offset, sync.drift, structure.av_length
Dropped word (0.3 s)    join.words       fail    none
Click at the join       join.click       fail    none
Sound one frame early   sync.av_offset   warn    none beyond the intended check
Sound 41 ms early       sync.av_offset   fail    none
Foreign incoming frame  picture.flash    fail    none
Caption 0.3 s late      captions.timing  fail    none
Card over caption       captions.cards   fail    none
Card over face          faces.clearance  fail    none
Caption over face       faces.clearance  fail    none

The repeated picture makes the picture end one frame after the sound, producing
the collateral structure.av_length warning. The flash replaces the second
incoming frame; the join's own two pictures stay intact.

The R37 sound-early fail limit is more than 40 ms. One frame at 30 fps is
33.3 ms, above the half-frame warning limit, so that fault fails only under
--strict by design. The default-fail case advances 1968 samples (41 ms), the
smallest whole-millisecond advance beyond that limit. Measured segment offsets
in ms are [-40.999953, -40.999998, -40.999971, -41.000024, -41.000020,
-41.000037, -41.000022, -41.000012]. Every segment clears the fail limit by
about 1 ms, so no larger advance is needed. No threshold is changed.

Caption faults are byte copies of the clean render with different hash-bound
layout records. Every final has measured layout and faces records so all three
layout checks run without invoking the system face detector.
"""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pytest
import synth

from yap_studio.checks.caption_layout import write_layout_record
from yap_studio.checks.spans import Span, score_spans, spans_from_report
from yap_studio.edl import load_edit_list
from yap_studio.faces.detect import write_faces
from yap_studio.plan import Plan
from yap_studio.report import exit_code
from yap_studio.verify import verify

_CUT = 3


@pytest.fixture(scope="module")
def calibration(synth_media, tmp_path_factory):
    """Build each final once and share one real verify report per file."""
    started = perf_counter()
    directory = tmp_path_factory.mktemp("calibration")
    source = synth_media.path("calib")
    clean = directory / "clean.mp4"
    shutil.copyfile(synth_media.path("clean"), clean)
    edl = directory / "ranges.json"
    edl.write_text(json.dumps({
        "sources": {"main": str(source)},
        "ranges": [
            {"source": "main", "start": start, "end": end}
            for start, end in synth_media.spans
        ],
    }))
    plan = Plan.build(load_edit_list(edl), Fraction(30), 48_000)
    cut = plan.cuts[_CUT - 1]
    assert cut.index == _CUT and cut.out_frame == 216 and float(cut.out_time) == 7.2
    frame, at = cut.out_frame, float(cut.out_time)
    paths = {"clean": clean}
    builders = {
        "missing": lambda out: synth.drop_frame(clean, out, frame),
        "repeated": lambda out: synth.repeat_frame(clean, out, frame - 1),
        "dropped_word": lambda out: synth.mute_audio(clean, out, at + 0.04, 0.3),
        "click": lambda out: synth.plant_click(clean, out, at),
        "sound_early": lambda out: synth.shift_audio_early(clean, out, 1600),
        "sound_early_fail": lambda out: synth.shift_audio_early(clean, out, 1968),
        # Frame 333 is from segment 5; the join's own two pictures stay intact.
        "flash": lambda out: synth.plant_flash(clean, out, frame + 1, 333),
    }
    for name, build in builders.items():
        paths[name] = directory / f"{name}.mp4"
        shutil.copyfile(synth_media.cached(f"calibration-{name}", build), paths[name])
    for name in ("caption_late", "card_over_caption", "card_over_face", "caption_over_face"):
        paths[name] = directory / f"{name}.mp4"
        shutil.copyfile(clean, paths[name])
    reports, timings = {}, {}
    for name, media in paths.items():
        document = synth.calibration_layout(
            late_phrase="caption-2" if name == "caption_late" else None,
            late_frames=9, card_over_caption=name == "card_over_caption",
            card_over_face=name == "card_over_face", caption_over_face=name == "caption_over_face",
        )
        if name in {"card_over_face", "caption_over_face"}:
            # Isolate face faults on the small calibration canvas.
            document["caption_gap_px"] = 0
        write_layout_record(media, document)
        write_faces(media.with_suffix(".faces.json"), synth.calibration_faces(
            media, over_face=name in {"card_over_face", "caption_over_face"},
        ))
        before = perf_counter()
        reports[name] = verify(str(media), edl=str(edl), asr="none", loudness="note")
        timings[name] = perf_counter() - before
    elapsed = perf_counter() - started
    (directory / "reports.json").write_text(json.dumps({
        "reports": reports, "verify_seconds": timings, "module_seconds": elapsed,
    }, indent=2))
    return SimpleNamespace(reports=reports, paths=paths, timings=timings, elapsed=elapsed)


def _checks(report: dict, severity: str) -> set[str]:
    return {finding["check"] for finding in report["findings"] if finding["severity"] == severity}


@pytest.mark.parametrize(
    "name,intended,warnings,reason",
    [
        ("missing", "join.frames", {"sync.av_offset", "sync.drift"}, "frame is missing"),
        (
            "repeated", "join.frames",
            {"sync.av_offset", "sync.drift", "structure.av_length"}, "frame is repeated",
        ),
        ("dropped_word", "join.words", set(), None),
        ("click", "join.click", set(), None),
        ("sound_early", "sync.av_offset", {"sync.av_offset"}, None),
        ("sound_early_fail", "sync.av_offset", set(), None),
        ("flash", "picture.flash", set(), None),
        ("caption_late", "captions.timing", set(), None),
        ("card_over_caption", "captions.cards", set(), None),
        ("card_over_face", "faces.clearance", set(), None),
        ("caption_over_face", "faces.clearance", set(), None),
    ],
    ids=["missing", "repeated", "dropped_word", "click", "sound_early", "sound_early_fail",
         "flash", "caption_late", "card_over_caption", "card_over_face", "caption_over_face"],
)
def test_planted_fault(calibration, name, intended, warnings, reason):
    report = calibration.reports[name]
    diagnostic = json.dumps(report["findings"], indent=2)
    expected_fails = set() if name == "sound_early" else {intended}
    assert _checks(report, "fail") == expected_fails, diagnostic
    assert _checks(report, "warn") == warnings, diagnostic
    assert report["status"] == ("warn" if name == "sound_early" else "fail"), diagnostic
    findings = [row for row in report["findings"] if row["severity"] == "fail"]
    if name == "sound_early":
        offset_warnings = [
            row for row in report["findings"]
            if row["check"] == "sync.av_offset" and row["severity"] == "warn"
        ]
        assert offset_warnings, diagnostic
        for warning in offset_warnings:
            assert "sound" in warning["message"].lower(), diagnostic
            assert "early" in warning["message"].lower(), diagnostic
            offsets = warning["evidence"]["offsets_ms"]
            assert offsets, diagnostic
            for offset in offsets:
                assert offset == pytest.approx(-1000 / 30, abs=2), diagnostic
        assert exit_code(report, strict=False) == 0
        assert exit_code(report, strict=True) == 1
    else:
        if name in {"sound_early_fail", "caption_late", "card_over_caption",
                    "card_over_face", "caption_over_face"}:
            assert len(findings) == 1, diagnostic
        else:
            assert [row["cut"] for row in findings] == [_CUT], diagnostic
        assert report["checks"][intended]["ran"] is True
        assert report["checks"][intended]["status"] == "fail"
        if name == "sound_early_fail":
            offsets = findings[0]["evidence"]["offsets_ms"]
            assert len(offsets) == len(synth.default_spans()), diagnostic
            assert all(offset < -40 for offset in offsets), diagnostic
            assert exit_code(report, strict=False) == 1
        if reason is not None:
            assert reason in findings[0]["message"], diagnostic
    for check in ("captions.timing", "captions.cards", "faces.clearance"):
        assert report["checks"][check]["ran"] is True
        expected = "fail" if check == intended else "pass"
        assert report["checks"][check]["status"] == expected


def test_clean_control(calibration):
    report = calibration.reports["clean"]
    diagnostic = json.dumps(report["findings"], indent=2)
    assert report["status"] == "pass", diagnostic
    assert _checks(report, "fail") == set(), diagnostic
    assert _checks(report, "warn") == set(), diagnostic
    assert exit_code(report, strict=True) == 0
    for check in ("captions.timing", "captions.cards", "faces.clearance"):
        assert report["checks"][check]["ran"] is True
        assert report["checks"][check]["status"] == "pass"


def test_span_score(calibration: SimpleNamespace, tmp_path: Path) -> None:
    """Grade fail findings against planted times without weakening fault assertions."""
    planted = {
        "missing": ("join.frames", 7.2),
        "repeated": ("join.frames", 7.2),
        "dropped_word": ("join.words", 7.2),
        "click": ("join.click", 7.2),
        # The second incoming frame is foreign; the finding reports the planned join.
        "flash": ("picture.flash", 217 / 30),
        "sound_early_fail": ("sync.av_offset", 0.0),
        "caption_late": ("captions.timing", 7.2),
        "card_over_caption": ("captions.cards", 217 / 30),
        "card_over_face": ("faces.clearance", 216 / 30),
        "caption_over_face": ("faces.clearance", 217 / 30),
    }
    truth = [Span(video, check, time_s, time_s) for video, (check, time_s) in planted.items()]
    predicted = [
        span
        for video, report in calibration.reports.items()
        for span in spans_from_report(report, video=video, severities=("fail",))
    ]
    score = score_spans(truth, predicted, videos=calibration.reports.keys())
    # Keep both artifacts together here without changing the shared fixture.
    (tmp_path / "reports.json").write_text(json.dumps({
        "reports": calibration.reports,
        "verify_seconds": calibration.timings,
        "module_seconds": calibration.elapsed,
    }, indent=2, allow_nan=False))
    (tmp_path / "span-score.json").write_text(json.dumps(score, indent=2, allow_nan=False))
    exact = next(row for row in score["tolerances"] if row["frames"] == 0)
    two_frames = next(row for row in score["tolerances"] if row["frames"] == 2)
    # Only flash misses at zero frames: its finding points at the join, one frame earlier.
    assert len(calibration.reports) == 12
    assert exact["recall"] == pytest.approx(9 / 10), score
    assert two_frames["precision"] == two_frames["recall"] == 1.0, score
    assert two_frames["true_positives"] == len(planted)
    assert two_frames["false_positives"] == two_frames["false_negatives"] == 0
