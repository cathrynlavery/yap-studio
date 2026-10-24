"""Published platform preset checks."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pytest

from yap_studio import ffmpeg
from yap_studio.checks.platform import PLATFORMS, check_platform, load_presets
from yap_studio.cli import main
from yap_studio.report import summarize
from yap_studio.verify import verify


def _finding(findings, check: str):
    return next((finding for finding in findings if finding.check == check), None)


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


def test_bundled_presets_have_pinned_official_sources() -> None:
    presets = load_presets()
    assert tuple(presets) == PLATFORMS
    domains = {
        "youtube": "support.google.com",
        "shorts": "support.google.com",
        "tiktok": "support.tiktok.com",
        "reels": "help.instagram.com",
    }
    for name, preset in presets.items():
        assert date.fromisoformat(preset["checked"]).isoformat() == preset["checked"]
        for rule in preset.values():
            if not isinstance(rule, dict):
                continue
            if any(isinstance(value, (int, float, list)) for value in rule.values()):
                assert urlparse(rule["source_url"]).scheme == "https"
                assert urlparse(rule["source_url"]).hostname == domains[name]
                assert date.fromisoformat(rule["checked"]).isoformat() == rule["checked"]
            for key, value in rule.items():
                if key.endswith("source_url"):
                    assert urlparse(value).scheme == "https"
                    assert urlparse(value).hostname == domains[name]


@pytest.mark.parametrize(
    ("name", "horizontal", "vertical", "resolution"),
    [
        ("youtube", None, "fail", "warn"),
        ("shorts", "fail", None, None),
        ("tiktok", "fail", None, "warn"),
        ("reels", None, None, "warn"),
    ],
)
def test_presets_check_horizontal_and_vertical_shape_and_resolution(
    synth_media,
    name: str,
    horizontal: str | None,
    vertical: str | None,
    resolution: str | None,
) -> None:
    preset = load_presets()[name]
    for fixture, expected_shape in (("clean", horizontal), ("clean_crop", vertical)):
        findings = check_platform(ffmpeg.probe(synth_media.path(fixture)), preset, name=name)
        shape = _finding(findings, "platform.shape")
        assert (shape.severity if shape else None) == expected_shape
        resolution_finding = _finding(findings, "platform.resolution")
        assert (resolution_finding.severity if resolution_finding else None) == resolution
        for finding in findings:
            assert finding.evidence["source_url"].startswith("https://")
            assert finding.evidence["checked"] == "2026-09-28"


def test_reels_accepts_sixteen_by_nine(synth_media) -> None:
    findings = check_platform(
        ffmpeg.probe(synth_media.path("clean")),
        load_presets()["reels"],
        name="reels",
    )
    assert _finding(findings, "platform.shape") is None


def test_injected_short_duration_limits_fail_and_warn(synth_media) -> None:
    info = ffmpeg.probe(synth_media.path("clean"))
    preset = deepcopy(load_presets()["youtube"])
    preset["duration"]["max_s"] = 5
    preset["duration"]["warn_over_s"] = 3
    failure = _finding(check_platform(info, preset, name="test"), "platform.duration")
    assert failure.severity == "fail"
    assert failure.evidence["rule"] == preset["duration"]["rule"]

    preset["duration"]["max_s"] = 30
    warning = _finding(check_platform(info, preset, name="test"), "platform.duration")
    assert warning.severity == "warn"
    assert warning.evidence["rule"] == preset["duration"]["warn_rule"]


@pytest.mark.parametrize(("name", "duration_s", "severity", "action"), [
    ("youtube", 900, None, None),
    ("youtube", 1731.97, "warn",
     "Uploads longer than 15 minutes need a verified YouTube account. "
     "Verify your account, or trim the video to 15 minutes or less if you cannot verify."),
    ("youtube", 43201, "fail", "Trim the video to 12 hours or less."),
    ("tiktok", 1801, "warn", "Trim the video to 30 minutes or less."),
    ("reels", 181, "warn", "Trim the video to 3 minutes or less."),
])
def test_duration_actions(synth_media, name, duration_s, severity, action) -> None:
    info = replace(ffmpeg.probe(synth_media.path("clean")), duration_s=duration_s)
    preset = load_presets()[name]
    finding = _finding(check_platform(info, preset, name=name), "platform.duration")
    if severity is None:
        assert finding is None
        return
    assert finding.severity == severity
    assert finding.suggested_action == action
    assert finding.evidence["source_url"] == preset["duration"].get(
        "warn_source_url", preset["duration"]["source_url"]
    )


def test_verify_summary_uses_duration_warning_action(synth_media, monkeypatch) -> None:
    presets = load_presets()
    presets["youtube"]["duration"]["warn_over_s"] = 3
    action = "Verify your YouTube account for longer uploads."
    presets["youtube"]["duration"]["warn_action"] = action
    monkeypatch.setattr("yap_studio.verify.load_presets", lambda: presets)
    result = verify(str(synth_media.path("clean")), asr="none", platform="youtube")
    warning = next(row for row in result["findings"] if row["check"] == "platform.duration")
    assert warning["severity"] == "warn"
    assert warning["suggested_action"] == action
    assert action in " ".join(result["summary"]["text"].split())
    assert "Trim the video" not in result["summary"]["text"]


def test_reels_warns_for_twenty_four_fps(synth_media) -> None:
    def build(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(synth_media.path("clean")),
                "-r",
                "24",
                "-c:v",
                "libx264",
                "-c:a",
                "copy",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    path = synth_media.cached("t11-24-fps", build)
    findings = check_platform(ffmpeg.probe(path), load_presets()["reels"], name="reels")
    fps = _finding(findings, "platform.fps")
    assert fps.severity == "warn"
    assert "24 fps" in fps.message


@pytest.mark.parametrize("with_edit_list", [False, True])
def test_verify_records_shorts_preset_and_summary(
    synth_media,
    tmp_path: Path,
    with_edit_list: bool,
) -> None:
    kwargs = {"asr": "none", "platform": "shorts"}
    if with_edit_list:
        edl = _edit_list(
            tmp_path / "edit.json",
            synth_media.path("calib"),
            synth_media.spans,
        )
        kwargs["edl"] = str(edl)
    result = verify(str(synth_media.path("clean_crop")), **kwargs)
    assert result["platform"]["name"] == "shorts"
    assert result["platform"]["checked"] == "2026-09-28"
    assert result["platform"]["rules"]["label"] == "YouTube Shorts"
    assert all(result["checks"][f"platform.{check}"]["ran"] for check in (
        "shape",
        "duration",
        "resolution",
        "fps",
    ))
    assert "for YouTube Shorts" in summarize(result).splitlines()[0]


def test_cli_accepts_youtube_platform(synth_media, capsys) -> None:
    code = main(
        [
            "verify",
            str(synth_media.path("clean")),
            "--asr",
            "none",
            "--platform",
            "youtube",
        ]
    )
    assert code == 0
    assert "for YouTube" in capsys.readouterr().out.splitlines()[0]
