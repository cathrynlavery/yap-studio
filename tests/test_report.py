"""Pinned verification report layout and creator summary rendering."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from yap_studio.checks.model import CHECK_IDS, Finding
from yap_studio.report import TOP_LEVEL_KEYS, build_report, exit_code, summarize


def _report(*, mode: str = "edit_list", status: str = "fail") -> dict:
    checks = {
        check: {"ran": True, "status": "pass", "reason": None}
        for check in CHECK_IDS
    }
    for check in CHECK_IDS:
        if check.startswith("platform."):
            checks[check] = {"ran": False, "status": "skipped", "reason": "no platform chosen"}
    if mode == "edit_list":
        checks["detect.cuts"] = {
            "ran": False,
            "status": "skipped",
            "reason": "edit list supplied",
        }
    if mode == "detected":
        for check in (
            "sync.av_offset",
            "sync.drift",
            "plan.picture",
            "plan.frames",
            "plan.samples",
            "join.words",
            "join.gap",
        ):
            checks[check] = {
                "ran": False,
                "status": "skipped",
                "reason": "needs the source and edit list",
            }
    findings = [
        {
            "check": "join.click",
            "severity": "fail",
            "message": "Cut 3 at 0:12.4: click at the join (+8.1 dB).",
            "cut": 3,
            "segment": None,
            "time_s": 12.4,
            "suggested_action": "Re-render this cut with a short audio fade.",
            "evidence": {},
        }
    ]
    return {
        "schema": "yap-studio/verify-report@1",
        "tool": {"name": "yap-studio", "version": "test"},
        "status": status,
        "media": {
            "path": "/tmp/final.mp4",
            "file": "final.mp4",
            "width": 1080,
            "height": 1920,
            "orientation": "vertical",
            "fps": "30/1",
            "duration_s": 48.2,
            "frames": 1446,
            "sample_rate": 48000,
            "has_audio": True,
        },
        "edit_list": None if mode == "detected" else {"path": "/tmp/edit.json"},
        "sources": [],
        "platform": None,
        "summary": {
            "mode": mode,
            "status": status,
            "cuts": 12,
            "fail": 1 if status == "fail" else 0,
            "warn": 1 if status == "warn" else 0,
            "info": 0,
            "text": "",
        },
        "findings": findings if status == "fail" else [],
        "cuts": [
            {
                "index": index,
                "time_s": float(index),
                "click_db": -20.0,
                "gap_ms": 0.0,
                "frames": "pass",
                "words": "pass",
                "landing": "measured",
            }
            for index in range(1, 13)
        ],
        "segments": None,
        "structure": {
            "integrated_lufs": -15.8,
            "lra_lu": 2.0,
            "true_peak_dbtp": -2.0,
            "target_lufs": -14.0,
            "black": 0,
            "freeze": 0,
        },
        "checks": checks,
        "receipt": {},
    }


def test_top_level_keys_are_pinned() -> None:
    assert TOP_LEVEL_KEYS == [
        "schema",
        "tool",
        "status",
        "media",
        "edit_list",
        "sources",
        "platform",
        "summary",
        "findings",
        "cuts",
        "segments",
        "structure",
        "checks",
        "receipt",
    ]


def test_summary_prints_failing_cut_and_fix_on_separate_lines() -> None:
    text = summarize(_report())
    lines = text.splitlines()
    failure = next(index for index, line in enumerate(lines) if "Cut 3 at 0:12.4" in line)
    assert "Click at the join" in lines[failure]
    assert lines[failure + 1].strip() == "Re-render this cut with a short audio fade."


def test_passing_word_check_says_audio_matches_the_edit() -> None:
    text = summarize(_report(status="pass"))
    assert "audio matches the edit" in text
    assert "words intact" not in text


def test_summary_separates_failures_and_warnings_with_matching_counts() -> None:
    report = _report()
    report["summary"]["warn"] = 1
    report["findings"].append(
        {
            "check": "loudness.integrated",
            "severity": "warn",
            "message": "At 0:00.0, overall loudness is -19.4 LUFS.",
            "cut": None,
            "segment": None,
            "time_s": 0.0,
            "suggested_action": "Normalize the whole video to -14 LUFS.",
            "evidence": {},
        }
    )
    lines = summarize(report).splitlines()
    fail_heading = lines.index("FAIL: 1 problem")
    warn_heading = lines.index("WARN: 1 thing to check")
    failure = next(index for index, line in enumerate(lines) if "Click at the join" in line)
    warning = next(index for index, line in enumerate(lines) if "Overall loudness" in line)
    assert fail_heading < failure < warn_heading < warning
    assert lines[warn_heading - 1] == ""


def test_edit_list_summary_omits_inapplicable_cut_detection() -> None:
    report = _report()
    assert report["checks"]["detect.cuts"] == {
        "ran": False,
        "status": "skipped",
        "reason": "edit list supplied",
    }
    assert "Picture cut detection" not in summarize(report)


def test_detected_summary_names_cut_discovery_and_skipped_source_checks() -> None:
    text = summarize(_report(mode="detected"))
    assert "12 clear cuts found in the picture" in text
    assert "Not checked:" in text
    assert "  Jump cuts inside one take: may be missed without an edit list" in text
    assert "Sync and words against the source: need the edit list" in text


def test_summary_labels_missing_loudness_as_not_measured() -> None:
    report = _report(status="pass")
    report["structure"]["integrated_lufs"] = None
    report["checks"]["loudness.integrated"] = {
        "ran": False,
        "status": "skipped",
        "reason": "no audio in the render",
    }
    assert "  Loudness not measured" in summarize(report)


def test_summary_reports_loudness_notes_without_target_or_action() -> None:
    report = _report(status="pass")
    report["findings"] = [
        Finding(
            "loudness.integrated", "info",
            "At 0:00.0, overall loudness is -20.0 LUFS. "
            "Loudness is set when the video is assembled.",
            time_s=0.0,
        ).to_dict(),
        Finding(
            "loudness.true_peak", "info",
            "At 0:00.0, sound peaks at -0.8 dBTP. "
            "Loudness is set when the video is assembled.",
            time_s=0.0,
        ).to_dict(),
    ]
    report["summary"]["info"] = 2
    text = summarize(report)
    assert "PASS" in text
    assert "Overall loudness is -20.0 LUFS." in text
    assert "Sound peaks at -0.8 dBTP." in text
    assert "Loudness is set when the video is assembled." in text
    assert "target" not in text
    assert "Normalize" not in text and "Limit" not in text
    assert "WARN" not in text


@pytest.mark.parametrize("check,severity", [
    ("join.click", "fail"), ("taste.rules", "warn"),
])
def test_actionable_finding_overrides_not_run_coverage(check: str, severity: str) -> None:
    checks = {check: {"ran": True, "reason": None} for check in CHECK_IDS}
    checks[check] = {"ran": False, "reason": "input unavailable"}
    finding = Finding(
        check,
        severity,
        "Cut 1 at 0:01.0: click at the join (+8.1 dB).",
        cut=1,
        time_s=1.0,
    )
    report = build_report(
        media_info=SimpleNamespace(
            path="/tmp/final.mp4",
            width=1080,
            height=1920,
            orientation="vertical",
            fps=Fraction(30),
            duration_s=2.0,
            video_frames=60,
            sample_rate=48000,
            has_audio=True,
        ),
        edit_list=None,
        sources=[],
        findings=[finding],
        structure=SimpleNamespace(
            integrated_lufs=-14.0,
            lra_lu=1.0,
            true_peak_dbtp=-2.0,
            black=[],
            freeze=[],
        ),
        checks=checks,
        receipt={},
        target_lufs=-14.0,
        detected=SimpleNamespace(results=[]),
    )
    assert report["checks"][check] == {
        "ran": True,
        "status": severity,
        "reason": None,
    }
    assert report["status"] == severity
    assert report["findings"] == [finding.to_dict()]
    assert exit_code(report, strict=False) == int(severity == "fail")


@pytest.mark.parametrize("ran", [False, True])
def test_taste_info_findings_follow_measurement_coverage(ran: bool) -> None:
    reason = (
        "no applied rule could be measured from the picture: "
        "captions.max_words_per_phrase (Words per caption need text recognition.)"
    )
    finding = Finding(
        "taste.rules", "info",
        "Rule r-123456789abc, captions max words per phrase: "
        "Words per caption need text recognition.",
    )
    coverage = (
        {"ran": True, "reason": None, "note": "1 of 2 rules not measured"}
        if ran else {"ran": False, "reason": reason}
    )
    report = build_report(
        media_info=SimpleNamespace(
            path="/tmp/final.mp4", width=1080, height=1920, orientation="vertical",
            fps=Fraction(30), duration_s=2.0, video_frames=60, sample_rate=48000, has_audio=True,
        ),
        edit_list=None, sources=[], findings=[finding],
        structure=SimpleNamespace(
            integrated_lufs=-14.0, lra_lu=1.0, true_peak_dbtp=-2.0, black=[], freeze=[],
        ),
        checks={"taste.rules": coverage}, receipt={}, target_lufs=-14.0,
        detected=SimpleNamespace(results=[]),
    )
    assert report["findings"] == ([finding.to_dict()] if ran else [])
    assert report["summary"]["info"] == int(ran)
    assert report["checks"]["taste.rules"] == {
        **coverage, "status": "pass" if ran else "skipped",
    }
    if not ran:
        assert f"Approved taste rules: {reason}" in report["summary"]["text"]


def test_not_checked_fallback_uses_friendly_label() -> None:
    report = _report(status="pass")
    report["checks"]["picture.freeze"] = {
        "ran": False,
        "status": "skipped",
        "reason": "picture unavailable",
    }
    text = summarize(report)
    assert "  Frozen picture: picture unavailable" in text
    assert "picture.freeze:" not in text


@pytest.mark.parametrize(
    "state,reason,expected",
    [
        ("none", "the edit list is one continuous range", "Cut coverage: none."),
        ("not_exposed", "the edit list says the render is a composite", "not exposed."),
        ("declared", "1 join declared in the edit list was checked", "Cut coverage: declared."),
        ("attested", "cuts were checked outside Yap Studio", "Cut coverage: attested."),
    ],
)
def test_summary_names_the_cut_coverage_state(state, reason, expected) -> None:
    report = _report(status="pass")
    report["cuts"] = []
    report["summary"]["cuts"] = 0
    report["edit_list"] = {
        "path": "/tmp/edit.json",
        "cut_coverage": state,
        "cut_coverage_reason": reason,
        "attested": (
            [
                {
                    "path": "/tmp/review.txt",
                    "found": True,
                    "match": True,
                    "label": "checked outside Yap Studio",
                }
            ]
            if state == "attested"
            else []
        ),
    }
    for check in ("join.click", "join.gap", "join.words", "join.frames", "join.listen"):
        report["checks"][check] = {
            "ran": False,
            "status": "skipped",
            "reason": f"cut coverage: {state.replace('_', ' ')}",
        }
    text = summarize(report)
    assert expected in text
    assert reason[:1].upper() + reason[1:] in text
    assert "no cuts in the edit list" not in text
    assert "Cut checks (clicks, gaps, words, frames, listening): cut coverage: " in text
    if state == "attested":
        assert "review.txt: hash matches (checked outside Yap Studio)" in text


def test_checked_edit_list_has_no_cut_coverage_line() -> None:
    report = _report()
    report["edit_list"] = {"path": "/tmp/edit.json", "cut_coverage": "checked"}
    assert "Cut coverage" not in summarize(report)


def test_exit_codes_allow_warnings_unless_strict() -> None:
    passing = _report(status="pass")
    warning = _report(status="warn")
    failing = _report(status="fail")
    assert exit_code(passing, False) == 0
    assert exit_code(warning, False) == 0
    assert exit_code(warning, True) == 1
    assert exit_code(failing, False) == 1
