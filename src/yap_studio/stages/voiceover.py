"""Verification policy for an internal audio take picture."""

from __future__ import annotations

import copy
from typing import Any


def uncovered_frames(total_frames: int, elements: list[dict[str, Any]],
                     kinds: frozenset[str] = frozenset({"insert", "graphic"})
                     ) -> list[tuple[int, int]]:
    """Find inclusive picture gaps in content; the end hold shares its last frame."""
    windows = sorted((max(0, item["start_frame"]), min(total_frames, item["end_frame"]))
                     for item in elements if item["kind"] in kinds
                     and item.get("media", {}).get("fit") != "split")
    gaps = []
    cursor = 0
    for first, last in windows:
        if last <= first:
            continue
        if first > cursor:
            gaps.append((cursor, first - 1))
        cursor = max(cursor, last)
    if cursor < total_frames:
        gaps.append((cursor, total_frames - 1))
    return gaps


def frame_ranges_text(gaps: list[tuple[int, int]]) -> str:
    """Format inclusive frame ranges for creator messages."""
    return ", ".join(f"{first}-{last}" for first, last in gaps)


def skip_standin_picture_checks(report: dict) -> dict:
    """Remove only expected black and held picture findings from a cut."""
    result = copy.deepcopy(report)
    checks = {"picture.black", "picture.freeze"}
    result["findings"] = [row for row in result["findings"] if row["check"] not in checks]
    for check in checks:
        if check in result.get("checks", {}):
            result["checks"][check] = {"ran": False, "status": "skipped",
                                        "reason": "voiceover stand-in"}
    severities = {row["severity"] for row in result["findings"]}
    severities.update(row["status"] for row in result.get("checks", {}).values())
    status = "fail" if "fail" in severities else "warn" if "warn" in severities else "pass"
    result["status"] = status
    if "summary" in result:
        result["summary"]["status"] = status
        for severity in ("fail", "warn", "info"):
            if severity in result["summary"]:
                result["summary"][severity] = sum(
                    row["severity"] == severity for row in result["findings"])
        if "media" in result:
            from yap_studio.report import summarize

            result["summary"]["text"] = (
                summarize(result) + "\n  Voiceover stand-in black/freeze checks skipped"
            )
    return result
