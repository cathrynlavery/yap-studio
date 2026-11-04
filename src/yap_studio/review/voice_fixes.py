"""Creator-facing voice checks and verified review clips."""

from __future__ import annotations

import hashlib
from pathlib import Path

from yap_studio import voice
from yap_studio.voice.model import (
    VoiceError,
    checked_file,
    generation_fingerprint,
    require_generation,
)


def check_lines(checks: dict) -> list[str]:
    """Describe each measured check without exposing its internal keys."""
    lines = []
    for item in checks["items"]:
        kind, status = item["id"], item["status"]
        if kind == "splice.click":
            lines.append("Join: clean." if status == "pass" else "Join: listen at the edges.")
        elif kind == "splice.level":
            difference = item.get("difference_lu")
            lines.append(f"Loudness: within {difference:.1f} LU of your voice."
                         if difference is not None else "Loudness: could not be checked.")
        elif kind == "splice.room_tone":
            lines.append({"pass": "Room tone: matched.",
                          "skipped": "Room tone: no quiet gap to check."}.get(
                              status, "Room tone: listen for a change in background sound."))
        elif kind == "splice.pace":
            change = round(abs(1 - 1 / item["stretch"]) * 100)
            direction = "slower" if item["stretch"] > 1 else "faster"
            text = f"Pace: {change}% {direction} to fit." if change else "Pace: unchanged."
            if item["gap_s"] > 0.25:
                text += " A short pause remains."
            lines.append(text)
        elif kind == "splice.heard":
            text = (f"Heard as: {item['heard'].rstrip('.')}." if item.get("heard")
                    else "Heard as: could not be checked.")
            lines.append(text)
            if item.get("message"):
                lines.append(item["message"])
    return lines


def review_fixes(root: Path, words: list[dict]) -> list[dict]:
    """Use the reviewed version's output word clock, never the source clock."""
    fixes = voice.list_fixes(root)["fixes"]
    result = []
    for fix in fixes:
        if fix["status"] not in {"ready", "approved", "rejected"} or "clips" not in fix:
            continue
        starts = [word["start"] for word in words if word["id"] == fix["words"][0]]
        time_s = min(starts) if starts else None
        seconds = int(time_s) if time_s is not None else None
        cost = fix.get("generation", {}).get("cost", fix["estimate"])
        generation = generation_fingerprint(fix)
        origin = fix.get("from", {"kind": "clone"})
        own = origin["kind"] == "own"
        donor_seconds = int(origin["start_s"]) if own else None
        result.append({
            "id": fix["id"], "said": fix["said"], "say": fix["say"],
            "time_s": time_s, "time": f"{seconds // 60}:{seconds % 60:02}" if seconds is not None
            else None,
            "generation": generation,
            "kind": origin["kind"],
            "from_time": (f"{donor_seconds // 60}:{donor_seconds % 60:02}" if own else None),
            "before_url": f"voice-fix-audio?fix={fix['id']}&clip=before&generation={generation}",
            "after_url": f"voice-fix-audio?fix={fix['id']}&clip=after&generation={generation}",
            "checks": check_lines(fix["checks"]), "credits": 0 if own else cost["credits"],
            "credits_source": "free" if own else (
                "charged" if cost["source"] == "history" else "estimated"),
            "cost_note": "Credits left could not be read"
            if "plan" in fix["estimate"].get("unavailable", {}) else "",
            "status": fix["status"], "decision": fix.get("decision"),
        })
    return result


def clip_bytes(root: Path, fix_id: str, clip: str, *, generation: str | None = None) -> bytes:
    """Read only a recorded clip in its own fix directory and hash the served bytes."""
    fix = next((row for row in voice.list_fixes(root)["fixes"] if row["id"] == fix_id), None)
    if fix is None or clip not in fix.get("clips", {}):
        raise VoiceError(f"Voice fix {fix_id} was not found.", code="voice_fix_missing")
    if generation is not None:
        require_generation(fix, generation)
    row = fix["clips"][clip]
    target = checked_file(root, row)
    if target.resolve().parent != (root / fix["patch"]["path"]).resolve().parent:
        raise VoiceError("The voice clip must stay in its fix folder.", code="voice_patch_changed")
    try:
        data = target.read_bytes()
    except OSError:
        raise VoiceError("The voice clip is missing or changed.",
                         code="voice_patch_changed") from None
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise VoiceError("The voice clip is missing or changed.", code="voice_patch_changed")
    return data
