"""Pinned verification reports and creator-facing summaries."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Literal

from yap_studio import __version__
from yap_studio.checks.model import CHECK_IDS, Finding, format_time

REPORT_SCHEMA = "yap-studio/verify-report@1"
TOP_LEVEL_KEYS = [
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

_SEVERITY_ORDER = {"fail": 0, "warn": 1, "info": 2}
_WHOLE_FILE_CHECKS = {
    "plan.frames",
    "plan.samples",
    "structure.av_length",
    "loudness.integrated",
    "loudness.true_peak",
    "platform.shape",
    "platform.duration",
    "platform.resolution",
    "platform.fps",
}
_SOUND_CHECKS = (
    "join.click",
    "join.gap",
    "join.words",
    "join.listen",
    "sync.av_offset",
    "sync.drift",
    "plan.samples",
    "structure.av_length",
    "structure.audio_timestamps",
    "loudness.integrated",
    "loudness.true_peak",
    "loudness.parts",
)
_CHECK_LABELS = {
    "join.click": "Clicks at the cuts",
    "join.gap": "Gaps at the cuts",
    "join.words": "Words at the cuts",
    "join.frames": "Frames at the cuts",
    "join.listen": "Listening at the cuts",
    "join.relisten": "Re-listening at the cuts",
    "join.second_model": "Second speech engine at the cuts",
    "sync.av_offset": "Picture and sound sync",
    "sync.drift": "Sound drift",
    "sync.unmeasured": "Sync measurement",
    "plan.picture": "Picture against the edit list",
    "plan.frames": "Picture length",
    "plan.samples": "Sound length",
    "plan.holds": "Declared holds",
    "edit_list.evidence": "Attested evidence",
    "structure.av_length": "Picture and sound length",
    "structure.audio_timestamps": "Audio timestamps",
    "loudness.integrated": "Overall loudness",
    "loudness.true_peak": "True peak",
    "loudness.parts": "Loudness between parts",
    "picture.black": "Black picture",
    "picture.freeze": "Frozen picture",
    "picture.flash": "Flash frames at the cuts",
    "picture.transition": "Frames around declared transitions",
    "picture.grade": "Picture colour grade",
    "captions.timing": "Caption timing against the words",
    "captions.cards": "Cards near caption text",
    "captions.contrast": "Captions readable on the picture",
    "captions.words": "Caption words against the speech",
    "faces.clearance": "Captions and cards clear of the face",
    "detect.cuts": "Picture cut detection",
    "platform.shape": "Platform shape",
    "platform.duration": "Platform duration",
    "platform.resolution": "Platform resolution",
    "platform.fps": "Platform frame rate",
    "taste.rules": "Approved taste rules",
}


def _finding_dict(finding: Finding | dict[str, Any]) -> dict[str, Any]:
    return finding.to_dict() if isinstance(finding, Finding) else dict(finding)


def _check_status(
    check: str,
    findings: list[dict[str, Any]],
    coverage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    state = coverage.get(check, {"ran": True, "reason": None})
    if not state.get("ran", False):
        result = {"ran": False, "status": "skipped", "reason": state.get("reason")}
    else:
        severities = {row["severity"] for row in findings if row["check"] == check}
        status = "fail" if "fail" in severities else "warn" if "warn" in severities else "pass"
        result = {"ran": True, "status": status, "reason": None}
    if state.get("note"):
        result["note"] = state["note"]
    if state.get("evidence"):
        result["evidence"] = state["evidence"]
    if state.get("click_source"):
        result["click_source"] = state["click_source"]
    if state.get("words_source"):
        result["words_source"] = state["words_source"]
    return result


def _compact_edit_cuts(joins: Any, detail: str) -> list[dict[str, Any]]:
    rows = []
    for cut in joins.cuts:
        row = {
            "index": cut.index,
            "time_s": cut.out_time_s,
            "click_db": cut.click_db,
            "gap_ms": cut.gap_ms,
            "frames": "covered by transition" if cut.frames.reason == "covered by transition"
            else cut.frames.status,
            "words": cut.words.status,
            "landing": cut.landing,
        }
        if detail == "full":
            row["detail"] = cut.to_dict()
        rows.append(row)
    return rows


def _compact_detected_cuts(detected: Any, detail: str) -> list[dict[str, Any]]:
    rows = []
    for result in detected.results:
        row = {
            "index": result.cut.index,
            "time_s": result.cut.time_s,
            "confidence": result.cut.confidence,
            "click_db": result.click_db,
            "duplicate": result.duplicate,
            "black": result.black,
            "flash": result.flash,
        }
        if detail == "full":
            row["detail"] = result.to_dict()
        rows.append(row)
    return rows


def build_report(
    *,
    media_info: Any,
    edit_list: Any | None,
    sources: list[dict[str, Any]],
    findings: list[Finding],
    structure: Any,
    checks: dict[str, dict[str, Any]],
    receipt: dict[str, Any],
    target_lufs: float,
    platform: dict[str, Any] | None = None,
    detail: Literal["summary", "full"] = "summary",
    joins: Any | None = None,
    sync: Any | None = None,
    detected: Any | None = None,
    cut_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-compatible report with a stable top-level layout.

    With an edit list, ``edit_list.cut_coverage`` says how the render's cuts were
    covered: ``checked`` (every planned cut was compared with its sources),
    ``partial`` (some were), ``unavailable`` (none could be, for example because the
    sources are missing),
    ``declared`` (joins declared in the edit list were checked by their words),
    ``attested`` (checked outside Yap Studio; never counted as a pass),
    ``not_exposed`` (the edit list describes a finished render) or ``none``.
    """
    if detail not in {"summary", "full"}:
        raise ValueError("detail must be summary or full")
    coverage = {check: dict(state) for check, state in checks.items()}
    all_finding_rows = [_finding_dict(finding) for finding in findings]
    for row in all_finding_rows:
        # An unavailable grade warns without claiming a colour measurement.
        if row["check"] == "picture.grade" and row["severity"] == "warn":
            continue
        # A refused contrast record warns without claiming a completed measurement.
        if (row["check"] != "captions.contrast" and row["severity"] in {"warn", "fail"}
                and not coverage.get(row["check"], {"ran": True}).get("ran", True)):
            coverage[row["check"]] = {"ran": True, "reason": None}
    finding_rows = [
        row
        for row in all_finding_rows
        if row["severity"] != "info"
        or coverage.get(row["check"], {"ran": True}).get("ran", True)
    ]
    finding_rows.sort(
        key=lambda row: (
            _SEVERITY_ORDER[row["severity"]],
            float("inf") if row.get("time_s") is None else row["time_s"],
            row.get("cut") or 0,
        )
    )
    check_rows = {
        check: _check_status(check, finding_rows, coverage)
        for check in CHECK_IDS
    }
    mode = "edit_list" if edit_list is not None else "detected"
    cuts = (
        _compact_edit_cuts(joins, detail)
        if joins is not None
        else _compact_detected_cuts(detected, detail)
    )
    media = {
        "path": str(media_info.path),
        "file": Path(media_info.path).name,
        "width": media_info.width,
        "height": media_info.height,
        "orientation": media_info.orientation,
        "fps": (
            f"{media_info.fps.numerator}/{media_info.fps.denominator}"
            if media_info.fps is not None
            else None
        ),
        "duration_s": media_info.duration_s,
        "frames": media_info.video_frames,
        "sample_rate": media_info.sample_rate,
        "has_audio": media_info.has_audio,
    }
    edit_row = None
    if edit_list is not None:
        edit_row = {
            "path": str(edit_list.path),
            "format": edit_list.format,
            "segments": len(edit_list.segments),
            "cuts": len(cuts),
            "warnings": list(edit_list.warnings),
        }
        if cut_coverage is not None:
            edit_row.update(
                {
                    "cut_coverage": cut_coverage["state"],
                    "cut_coverage_reason": cut_coverage["reason"],
                    "declared_joins": cut_coverage["declared_joins"],
                    "composite": cut_coverage["composite"],
                    "attested": cut_coverage["attested"],
                    "planned_cuts": cut_coverage["planned_cuts"],
                    "checked_cuts": cut_coverage["checked_cuts"],
                    "declared_cuts": cut_coverage["declared_cuts"],
                }
            )
    structure_row = structure.to_dict() if detail == "full" else {
        "integrated_lufs": structure.integrated_lufs,
        "lra_lu": structure.lra_lu,
        "true_peak_dbtp": structure.true_peak_dbtp,
        "target_lufs": target_lufs,
        "black": len(structure.black),
        "freeze": len(structure.freeze),
    }
    severities = {row["severity"] for row in finding_rows}
    status = "fail" if "fail" in severities else "warn" if "warn" in severities else "pass"
    counts = {
        severity: sum(row["severity"] == severity for row in finding_rows)
        for severity in ("fail", "warn", "info")
    }
    result = {
        "schema": REPORT_SCHEMA,
        "tool": {"name": "yap-studio", "version": __version__},
        "status": status,
        "media": media,
        "edit_list": edit_row,
        "sources": sources,
        "platform": platform,
        "summary": {
            "mode": mode,
            "status": status,
            "cuts": len(cuts),
            **counts,
            "text": "",
        },
        "findings": finding_rows,
        "cuts": cuts,
        "segments": (
            [row.to_dict() for row in sync.segments]
            if detail == "full" and sync is not None
            else None
        ),
        "structure": structure_row,
        "checks": check_rows,
        "receipt": receipt,
    }
    result["summary"]["text"] = summarize(result)
    return result


def _duration(value: float | None) -> str:
    seconds = max(0, round(value or 0))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _label_and_message(finding: dict[str, Any]) -> tuple[str, str]:
    message = finding["message"]
    time_s = finding.get("time_s")
    timestamp = format_time(time_s) if time_s is not None else None
    if finding["check"] in _WHOLE_FILE_CHECKS:
        label = "Whole video"
        message = re.sub(r"^At \d+:\d{2}\.\d,\s*", "", message)
    elif finding.get("cut") is not None:
        message_label = re.match(
            rf"^(Cut {finding['cut']} at \d+:\d{{2}}\.\d):\s*",
            message,
        )
        label = (
            message_label.group(1)
            if message_label
            else f"Cut {finding['cut']} at {timestamp or '0:00.0'}"
        )
        message = re.sub(rf"^{re.escape(label)}:\s*", "", message, count=1)
    elif message.startswith("From "):
        match = re.match(r"^From (\d+:\d{2}\.\d)(?: to \d+:\d{2}\.\d)?,\s*", message)
        label = f"From {match.group(1)}" if match else f"From {timestamp or '0:00.0'}"
        if match:
            message = message[match.end() :]
    elif timestamp is not None:
        label = f"At {timestamp}"
        message = re.sub(rf"^(?:At )?{re.escape(timestamp)}[:,]?\s*", "", message, count=1)
    else:
        label = "Whole video"
    return label, message[:1].upper() + message[1:]


def _finding_lines(finding: dict[str, Any]) -> list[str]:
    label, message = _label_and_message(finding)
    column = 20
    prefix = f"  {label:<18}"
    lines = textwrap.wrap(
        message,
        width=98,
        initial_indent=prefix,
        subsequent_indent=" " * column,
        break_long_words=False,
        break_on_hyphens=False,
    )
    action = finding.get("suggested_action")
    if action and not message.rstrip().endswith(action.rstrip()):
        lines.extend(
            textwrap.wrap(
                action,
                width=98,
                initial_indent=" " * column,
                subsequent_indent=" " * column,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return lines


def _target_lufs(report: dict[str, Any]) -> float | None:
    if "target_lufs" in report["structure"]:
        return report["structure"]["target_lufs"]
    return report.get("receipt", {}).get("thresholds", {}).get("loudness_target_lufs")


def _ok_cut_line(report: dict[str, Any]) -> str | None:
    checks = report["checks"]
    properties = []
    if checks["join.click"]["ran"]:
        properties.append("no click")
    if checks["join.gap"]["ran"]:
        properties.append("no gap")
    if checks["join.frames"]["ran"]:
        properties.append(
            "right frames outside transitions" if checks["join.frames"].get("evidence")
            else "right frames"
        )
    if checks.get("picture.transition", {}).get("ran"):
        properties.append("transition boundaries checked")
    if checks["join.words"]["ran"]:
        properties.append("audio matches the edit")
    else:
        properties.append("words not checked")
    actionable = {
        row.get("cut")
        for row in report["findings"]
        if row["severity"] in {"fail", "warn"} and row.get("cut") is not None
    }
    count = sum(row["index"] not in actionable for row in report["cuts"])
    if not count or not properties:
        return None
    return f"  {count} cuts OK: {', '.join(properties)}"


_COVERAGE_TITLES = {
    "partial": "partial",
    "unavailable": "unavailable",
    "none": "none",
    "not_exposed": "not exposed",
    "declared": "declared",
    "attested": "attested",
}


def _cut_coverage_line(report: dict[str, Any]) -> str | None:
    """Say how the cuts were covered, unless every planned cut was checked."""
    edit_list = report.get("edit_list") or {}
    state = edit_list.get("cut_coverage")
    if state not in _COVERAGE_TITLES:
        return None
    reason = edit_list.get("cut_coverage_reason") or ""
    line = f"  Cut coverage: {_COVERAGE_TITLES[state]}. {reason[:1].upper()}{reason[1:]}."
    for row in edit_list.get("attested") or []:
        verdict = (
            "not found"
            if not row.get("found")
            else "hash matches"
            if row.get("match")
            else "hash does not match"
        )
        line += f"\n    {Path(row['path']).name}: {verdict} ({row.get('label')})"
    return line


def _not_checked(report: dict[str, Any]) -> list[str]:
    skipped = {
        check: row["reason"]
        for check, row in report["checks"].items()
        if not row["ran"]
    }
    if report["summary"]["mode"] == "edit_list":
        skipped.pop("detect.cuts", None)
    # Holds are checked only where an edit list declares them.
    if skipped.get("plan.holds") in {"no holds in the edit list", "needs the edit list"}:
        skipped.pop("plan.holds")
    if skipped.get("edit_list.evidence") in {"no evidence in the edit list", "needs the edit list"}:
        skipped.pop("edit_list.evidence")
    if skipped.get("picture.transition") == "no declared transitions":
        skipped.pop("picture.transition")
    lines = []
    if report["summary"]["mode"] == "detected":
        lines.append("  Jump cuts inside one take: may be missed without an edit list")
    if set(_SOUND_CHECKS) <= skipped.keys() and {
        skipped[check] for check in _SOUND_CHECKS
    } == {"no audio in the render"}:
        lines.append(
            "  Sound checks (clicks, gaps, words, sync, loudness): no audio in the render"
        )
        for check in _SOUND_CHECKS:
            skipped.pop(check)
    source_checks = {
        "sync.av_offset",
        "sync.drift",
        "plan.picture",
        "plan.frames",
        "plan.samples",
        "join.words",
        "join.gap",
    }
    if source_checks <= skipped.keys() and len({skipped[key] for key in source_checks}) == 1:
        reason = skipped[next(iter(source_checks))]
        if reason == "needs the source and edit list":
            reason = "need the edit list"
        lines.append(f"  Sync and words against the source: {reason}")
        for key in source_checks:
            skipped.pop(key)
    cut_checks = {
        "join.click": "clicks",
        "join.gap": "gaps",
        "join.words": "words",
        "join.frames": "frames",
        "join.listen": "listening",
    }
    by_coverage = [
        key
        for key in cut_checks
        if key in skipped and str(skipped[key]).startswith("cut coverage:")
    ]
    if by_coverage and len({skipped[key] for key in by_coverage}) == 1:
        names = ", ".join(cut_checks[key] for key in by_coverage)
        lines.append(f"  Cut checks ({names}): {skipped[by_coverage[0]]}")
        for key in by_coverage:
            skipped.pop(key)
    groups = [
        (
            ("platform.shape", "platform.duration", "platform.resolution", "platform.fps"),
            "Platform fit",
        ),
        (("join.words",), "Words at the cuts"),
        (("join.listen",), "Listening at the cuts"),
        (("join.relisten", "join.second_model"), "Re-listening with speech engines at the cuts"),
    ]
    for keys, label in groups:
        present = [key for key in keys if key in skipped]
        if present and len(present) == len(keys):
            reasons = {skipped[key] for key in present}
            reason = next(iter(reasons)) if len(reasons) == 1 else "; ".join(sorted(reasons))
            lines.append(f"  {label}: {reason}")
            for key in present:
                skipped.pop(key)
    for check, reason in skipped.items():
        lines.append(f"  {_CHECK_LABELS[check]}: {reason}")
    return lines


def summarize(report: dict[str, Any]) -> str:
    """Render a plain-ASCII verification summary for video creators."""
    media = report["media"]
    summary = report["summary"]
    cut_text = f"{summary['cuts']} {'cut' if summary['cuts'] == 1 else 'cuts'}"
    if summary["mode"] == "detected":
        noun = "cut" if summary["cuts"] == 1 else "cuts"
        cut_text = f"{summary['cuts']} clear {noun} found in the picture"
    header = (
        f"Yap Studio verify: {media['file']} "
        f"({media['width']}x{media['height']} {media['orientation']}, "
        f"{_duration(media['duration_s'])}, {cut_text})"
    )
    if report.get("platform"):
        header += f" for {report['platform']['label']}"
    lines = [header]
    sections = []
    if summary["fail"]:
        noun = "problem" if summary["fail"] == 1 else "problems"
        sections.append((f"FAIL: {summary['fail']} {noun}", "fail"))
    if summary["warn"]:
        noun = "thing" if summary["warn"] == 1 else "things"
        sections.append((f"WARN: {summary['warn']} {noun} to check", "warn"))
    if not sections:
        lines.extend(["", "PASS"])
    for heading, severity in sections:
        findings = [
            finding for finding in report["findings"] if finding["severity"] == severity
        ]
        lines.extend(["", heading])
        if findings:
            lines.append("")
        for finding in findings:
            lines.extend(_finding_lines(finding))
    ok_line = _ok_cut_line(report)
    extras = []
    if ok_line:
        extras.append(ok_line)
    coverage_line = _cut_coverage_line(report)
    if coverage_line:
        extras.append(coverage_line)
    integrated = report["structure"].get("integrated_lufs")
    target = _target_lufs(report)
    loudness_notes = [
        finding for finding in report["findings"]
        if finding["check"] in {"loudness.integrated", "loudness.true_peak"}
        and finding["severity"] == "info"
    ]
    if loudness_notes:
        for finding in loudness_notes:
            extras.extend(_finding_lines(finding))
    elif integrated is not None and target is not None:
        loudness_status = report["checks"]["loudness.integrated"]["status"]
        extras.append(
            f"  Loudness {integrated:.1f} LUFS, target {target:g} "
            f"({'ok' if loudness_status == 'pass' else loudness_status})"
        )
    elif not report["checks"]["loudness.integrated"]["ran"]:
        extras.append("  Loudness not measured")
    black = report["structure"].get("black", 0)
    freeze = report["structure"].get("freeze", 0)
    black = len(black) if isinstance(black, list) else black
    freeze = len(freeze) if isinstance(freeze, list) else freeze
    if not black and not freeze:
        extras.append("  No black or frozen picture")
    else:
        extras.append(f"  Picture: {black} black, {freeze} frozen intervals")
    if extras:
        lines.extend(["", *extras])
    skipped = _not_checked(report)
    if skipped:
        lines.extend(["", "Not checked:", *skipped])
    return "\n".join(lines)


def exit_code(report: dict[str, Any], strict: bool) -> int:
    """Return the command exit status for a completed verification report."""
    if report["status"] == "fail" or (strict and report["status"] == "warn"):
        return 1
    return 0
