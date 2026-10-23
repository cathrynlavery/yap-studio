"""Published platform rules for finished video renders."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from yap_studio.checks.model import Finding
from yap_studio.ffmpeg import MediaInfo

PLATFORMS = ("youtube", "shorts", "tiktok", "reels")


def load_presets() -> dict[str, dict[str, Any]]:
    """Load the bundled, source-pinned platform presets."""
    resource = resources.files("yap_studio").joinpath("platforms.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if payload.get("schema") != "yap-studio/platforms@1":
        raise ValueError("unsupported platform preset schema")
    return payload["platforms"]


def _source_name(url: str) -> str:
    if "support.google.com" in url:
        return "YouTube Help"
    if "support.tiktok.com" in url:
        return "TikTok Help"
    return "Instagram Help"


def _evidence(rule: dict[str, Any], *, warning: bool = False) -> dict[str, Any]:
    source_key = "warn_source_url" if warning and rule.get("warn_source_url") else "source_url"
    text_key = "warn_rule" if warning else "rule"
    evidence = {
        "source_url": rule[source_key],
        "rule": rule[text_key],
        "checked": rule["checked"],
    }
    additional = rule.get("additional_source_url")
    if additional:
        evidence["additional_source_url"] = additional
    return evidence


def _citation(rule: dict[str, Any], *, warning: bool = False) -> str:
    evidence = _evidence(rule, warning=warning)
    source = _source_name(evidence["source_url"])
    return f"{evidence['rule']} ({source}, checked {evidence['checked']})."


def _number(value: float) -> str:
    return f"{value:g}"


def _duration(value: float) -> str:
    if value % 3600 == 0:
        hours = value / 3600
        return f"{_number(hours)} {'hour' if hours == 1 else 'hours'}"
    if value % 60 == 0:
        minutes = value / 60
        return f"{_number(minutes)} {'minute' if minutes == 1 else 'minutes'}"
    return f"{_number(value)} {'second' if value == 1 else 'seconds'}"


def _expected_shape(rule: dict[str, Any]) -> str:
    accepted = rule.get("accept")
    if accepted:
        return " or ".join(accepted)
    return "9:16 to 1.91:1 aspect ratio"


def _shape_finding(info: MediaInfo, rule: dict[str, Any], label: str) -> Finding | None:
    if info.width is None or info.height is None or info.orientation is None:
        return None
    severity = None
    if "min_aspect" in rule:
        aspect = info.width / info.height
        if not rule["min_aspect"] <= aspect <= rule["max_aspect"]:
            severity = "fail"
        measured = f"{aspect:.3f}:1"
        description = f"This video's aspect ratio is {measured}"
    else:
        if info.orientation in rule.get("fail", []):
            severity = "fail"
        elif info.orientation in rule.get("warn", []):
            severity = "warn"
        description = f"This video is {info.orientation}"
    if severity is None:
        return None
    return Finding(
        check="platform.shape",
        severity=severity,
        message=f"{description}. {_citation(rule)}",
        suggested_action=f"Export a {_expected_shape(rule)} version for {label}.",
        evidence=_evidence(rule),
    )


def _duration_finding(info: MediaInfo, rule: dict[str, Any], label: str) -> Finding | None:
    if info.duration_s is None:
        return None
    maximum = rule.get("max_s")
    warning = rule.get("warn_over_s")
    if maximum is not None and info.duration_s > maximum:
        severity = "fail"
        limit = maximum
        use_warning_rule = False
    elif warning is not None and info.duration_s > warning:
        severity = "warn"
        limit = warning
        use_warning_rule = True
    else:
        return None
    return Finding(
        check="platform.duration",
        severity=severity,
        message=(
            f"This video is {_duration(info.duration_s)} long. "
            f"{_citation(rule, warning=use_warning_rule)}"
        ),
        suggested_action=f"Trim the video to {_duration(limit)} or less.",
        evidence=_evidence(rule, warning=use_warning_rule),
    )


def _resolution_finding(info: MediaInfo, rule: dict[str, Any], label: str) -> Finding | None:
    if info.width is None or info.height is None:
        return None
    short_side = min(info.width, info.height)
    minimum = rule.get("min_short_side")
    maximum = rule.get("max_short_side")
    if minimum is not None and short_side < minimum:
        return Finding(
            check="platform.resolution",
            severity="warn",
            message=(
                f"This video's short side is {short_side} pixels. {_citation(rule)}"
            ),
            suggested_action=f"Export at {minimum} pixels or more on the short side.",
            evidence=_evidence(rule),
        )
    if maximum is not None and short_side > maximum:
        return Finding(
            check="platform.resolution",
            severity="info",
            message=(
                f"This video's short side is {short_side} pixels. {_citation(rule)}"
            ),
            evidence=_evidence(rule),
        )
    return None


def _fps_finding(info: MediaInfo, rule: dict[str, Any], label: str) -> Finding | None:
    minimum = rule.get("min")
    if minimum is None or info.fps is None or float(info.fps) >= minimum * 0.999:
        return None
    measured = float(info.fps)
    return Finding(
        check="platform.fps",
        severity="warn",
        message=f"This video is {_number(measured)} fps. {_citation(rule)}",
        suggested_action=f"Export at {minimum} fps or more.",
        evidence=_evidence(rule),
    )


def check_platform(info: MediaInfo, preset: dict, *, name: str) -> list[Finding]:
    """Check one preset; the 0.999 fps tolerance lets 29.97 pass a 30 fps minimum."""
    label = preset.get("label", name)
    findings = []
    shape = preset.get("shape")
    if shape and (finding := _shape_finding(info, shape, label)):
        findings.append(finding)
    duration = preset.get("duration")
    if duration and (finding := _duration_finding(info, duration, label)):
        findings.append(finding)
    resolution = preset.get("resolution")
    if resolution and (finding := _resolution_finding(info, resolution, label)):
        findings.append(finding)
    fps = preset.get("fps")
    if fps and (finding := _fps_finding(info, fps, label)):
        findings.append(finding)
    return findings
