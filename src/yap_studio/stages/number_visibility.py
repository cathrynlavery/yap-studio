"""Measure rendered number ink and add the same result to final checks and verification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from yap_studio import ffmpeg
from yap_studio.matte.check import measure_behind_layers
from yap_studio.planner.number_behind import MIN_VISIBLE_SHARE

if TYPE_CHECKING:
    from yap_studio.stages.timeline import ResolvedTimeline

MAX_HIDDEN_FRAMES = 3  # Allow at most 0.1 seconds of weak visibility at 30 fps.
MAX_HIDDEN_SHARE = 0.1  # Keep exactly 90% visible ink on the passing side of the boundary.
NUMBER_INK_ALPHA_THRESHOLD = 230  # Exclude shadow alpha up to 179 while retaining solid glyph ink.
NUMBER_VISIBLE_CHECK = "graphics.number_visible"
NUMBER_VISIBLE_FIX = "re-run graphics so the number moves up or beside her head"


def measure_number_layers(
    root: Path, matte: Path, info: ffmpeg.MediaInfo, layers: list[dict],
    timeline: ResolvedTimeline, skipped: tuple[tuple[int, int], ...],
) -> tuple[dict, ...]:
    """Measure real ink for every number, including clear front fallbacks."""
    numbers = {item.id for item in timeline.elements
               if item.kind == "graphic" and item.payload.get("template") == "big_number"}
    records = []
    for layer in layers:
        if layer["id"] not in numbers:
            continue
        measured = measure_behind_layers(
            root, matte, info, [layer], skipped_windows=skipped,
            hidden_glyph_share=MAX_HIDDEN_SHARE,
            hidden_time_share=MAX_HIDDEN_FRAMES / layer["frames"],
            ink_alpha_threshold=NUMBER_INK_ALPHA_THRESHOLD,
            require_ink=True, report_worst=True,
            covered_pixels=(lambda *args: 0) if layer["layer"] == "front" else None,
        )
        records.extend({**item, "template": "big_number"} for item in measured)
    return tuple(records)


def number_visibility_assert(records: list[dict]) -> dict:
    """Require every real number layer to stay readable apart from three frames."""
    return {"id": NUMBER_VISIBLE_CHECK,
            "pass": all(item["hidden_frames"] <= MAX_HIDDEN_FRAMES for item in records),
            "expected": f"at most {MAX_HIDDEN_FRAMES} frames below {MIN_VISIBLE_SHARE:.0%} visible",
            "actual": records, "fix": NUMBER_VISIBLE_FIX}


def add_number_visibility(report: dict, asserts: list[dict], records: tuple[dict, ...]) -> None:
    """Retain number visibility in both the delivery asserts and the verify report."""
    numbers = [item for item in records if item.get("template") == "big_number"]
    if not numbers:
        return
    check = number_visibility_assert(numbers)
    asserts.append(check)
    report.setdefault("checks", {})[NUMBER_VISIBLE_CHECK] = {
        "ran": True, "status": "pass" if check["pass"] else "fail", "cards": numbers,
    }
    if not check["pass"]:
        report["status"] = "fail"
        report.setdefault("findings", []).append({
            "check": NUMBER_VISIBLE_CHECK, "severity": "fail", "time_s": None,
            "message": NUMBER_VISIBLE_FIX, "evidence": {"cards": numbers},
            "suggested_action": NUMBER_VISIBLE_FIX,
        })
