"""Deterministic review hints from measurements already in a watch pack."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio.errors import InputError, OutputPathError, ValidationError
from yap_studio.receipt import sha256_file
from yap_studio.review.model import frame_at
from yap_studio.transcript import Transcript, Word

from . import load_pack
from .findings import record_findings
from .images import format_time
from .pack import pack_output_name

STILL_S = 4.0
PAUSE_S = 1.0
LOUDNESS_JUMP_LU = 8.0
GRAPHIC_MIN_AREA = 0.10


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _settings(still_s: float, pause_s: float, loudness_jump_lu: float) -> dict[str, float]:
    settings = {"still_s": still_s, "pause_s": pause_s, "loudness_jump_lu": loudness_jump_lu}
    for name, value in settings.items():
        if not _number(value) or value <= 0:
            raise ValidationError(name, "must be finite and greater than zero")
    return {**settings, "graphic_min_area": GRAPHIC_MIN_AREA}


def _at_least(value: float, threshold: float) -> bool:
    """Keep exact boundaries inclusive despite floating-point subtraction."""
    return value >= threshold or math.isclose(value, threshold, rel_tol=1e-9, abs_tol=0)


def _subtract(
    start: float, end: float, windows: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Subtract half-open windows, keeping the uncovered pieces in time order."""
    pieces = [(start, end)] if end > start else []
    for left, right in sorted(windows):
        if right <= left:
            continue
        remaining = []
        for a, b in pieces:
            if right <= a or left >= b:
                remaining.append((a, b))
            else:
                if a < left:
                    remaining.append((a, left))
                if right < b:
                    remaining.append((right, b))
        pieces = remaining
    return pieces


def _suppresses_stillness(graphic: dict, video: dict) -> bool:
    """Always suppress holds and whole-picture masks; clip regions to 10 percent coverage."""
    if graphic["kind"] == "hold" or graphic["kind"] == "mask" and graphic.get("rect") is None:
        return True
    if graphic["kind"] != "region":
        return False
    width, height = video["width"], video["height"]
    if not _number(width) or not _number(height) or width <= 0 or height <= 0:
        raise ValidationError("video", "expected positive finite frame dimensions")
    rect = graphic["rect"]
    if not isinstance(rect, (list, tuple)) or len(rect) != 4 or not all(map(_number, rect)):
        raise ValidationError("graphics.rect", "expected finite [x, y, w, h] pixels")
    x, y, w, h = rect
    if w < 0 or h < 0:
        raise ValidationError("graphics.rect", "expected non-negative rectangle dimensions")
    # Pack rounding can turn a subpixel region into an empty rectangle.
    if w == 0 or h == 0:
        return False
    covered_width = max(0, min(width, x + w) - max(0, x))
    covered_height = max(0, min(height, y + h) - max(0, y))
    return _at_least((covered_width / width) * (covered_height / height), GRAPHIC_MIN_AREA)


def _still_findings(pack: dict, fps: Fraction, threshold: float) -> list[dict]:
    windows = [
        (graphic["start_s"], graphic["end_s"])
        for graphic in pack["graphics"]
        if _suppresses_stillness(graphic, pack["video"])
    ]
    cuts = sorted({cut["frame"] for cut in pack["cuts"]})
    findings = []
    for span in pack["motion"]["still_spans"]:
        first, last = span["start_frame"], span["end_frame"]
        edges = [first, *(cut for cut in cuts if first < cut < last), last]
        for a, b in zip(edges, edges[1:], strict=False):
            for start, end in _subtract(float(a / fps), float(b / fps), windows):
                duration = end - start
                if duration + float(0.5 / fps) < threshold:
                    continue
                findings.append(
                    {
                        "kind": "slow",
                        "severity": "high" if _at_least(duration, 2 * threshold) else "medium",
                        "start_s": start,
                        "end_s": end,
                        "reason": f"The picture barely changes for {duration:.2f} s, "
                        f"from {format_time(start)} to {format_time(end)}, "
                        "with no cut or large graphic.",
                        "fix": "Cut to another shot, add a graphic or B-roll, "
                        "or shorten this stretch.",
                        "check": "watch.still",
                        "evidence": {
                            "duration_s": duration,
                            "start_frame": frame_at(start, fps),
                            "end_frame": math.ceil(end * fps - Fraction(1, 1000)),
                            "threshold_s": threshold,
                        },
                    }
                )
    return findings


def _cut_findings(pack: dict, threshold: float) -> list[dict]:
    findings = []
    for cut in pack["cuts"]:
        jump = cut["loudness_jump_lu"]
        if jump is not None and not _number(jump):
            raise ValidationError("cuts", "loudness jumps must be finite numbers or null")
        if jump is None or abs(jump) < threshold:
            continue
        before, after = cut["loudness_before_lufs"], cut["loudness_after_lufs"]
        if not _number(before) or not _number(after):
            raise ValidationError("cuts", "a loudness jump requires both finite speech levels")
        direction = "louder" if jump > 0 else "quieter"
        findings.append(
            {
                "kind": "audio",
                "severity": "high" if abs(jump) >= 1.5 * threshold else "medium",
                "start_s": cut["time_s"],
                "end_s": None,
                "reason": f"Speech after cut {cut['index']} is {abs(jump):.1f} LU "
                f"{direction} than before it ({before:.1f} to {after:.1f} LUFS).",
                "fix": "Match the speech level on both sides of the cut before assembly.",
                "check": "watch.cut_loudness",
                "evidence": {
                    "cut": cut["index"],
                    "before_lufs": before,
                    "after_lufs": after,
                    "jump_lu": jump,
                },
            }
        )
    return findings


def _validate_words(words: list[Word], duration: float, fps: Fraction) -> None:
    for word in words:
        if (
            not isinstance(word, Word)
            or not isinstance(word.text, str)
            or not word.text.strip()
            or not _number(word.start)
            or not _number(word.end)
            or not 0 <= word.start <= word.end <= duration + float(1 / fps)
            or word.conf is not None
            and not _number(word.conf)
        ):
            raise ValidationError("words", "expected text and finite ordered video-clock times")


def _word_gaps(words: list[Word], duration: float) -> list[tuple]:
    cursor = 0.0
    previous = None
    gaps = []
    for word in sorted(words, key=lambda word: (word.start, word.end, word.text)):
        if word.start > cursor:
            gaps.append((cursor, min(word.start, duration), previous, word.text))
        if word.end >= cursor:
            cursor = min(word.end, duration)
            previous = word.text
    if cursor < duration:
        gaps.append((cursor, duration, previous, None))
    return gaps


def _neighbours(before: str | None, after: str | None) -> str:
    def quote(text: str) -> str:
        return (
            json.dumps(text, ensure_ascii=False)
            .replace(chr(0x2013), "-")
            .replace(chr(0x2014), ", ")
        )

    if before is not None and after is not None:
        return f" between {quote(before)} and {quote(after)}"
    if after is not None:
        return f" before the first word {quote(after)}"
    if before is not None:
        return f" after the last word {quote(before)}"
    return " with no spoken words"


def _pause_findings(pack: dict, words: list[Word] | None, threshold: float) -> list[dict]:
    windows = [
        (graphic["start_s"], graphic["end_s"])
        for graphic in pack["graphics"]
        if graphic["kind"] == "hold"
    ]
    gaps = (
        _word_gaps(words, pack["video"]["duration_s"])
        if words is not None
        else [(span["start_s"], span["end_s"], None, None) for span in pack["loudness"]["silences"]]
    )
    findings = []
    for a, b, before, after in gaps:
        for start, end in _subtract(a, b, windows):
            duration = end - start
            if not _at_least(duration, threshold):
                continue
            description = (
                f"A {duration:.2f} s pause{_neighbours(before, after)}, "
                if words is not None
                else f"{duration:.2f} s of silence "
            )
            findings.append(
                {
                    "kind": "slow",
                    "severity": "medium" if _at_least(duration, 2 * threshold) else "low",
                    "start_s": start,
                    "end_s": end,
                    "reason": f"{description}from {format_time(start)} to {format_time(end)}.",
                    "fix": "Tighten the pause to about 0.3 s, "
                    "or cover it with a graphic or B-roll.",
                    "check": "watch.pause",
                    "evidence": {"duration_s": duration, "threshold_s": threshold},
                }
            )
    return findings


def scan_pack(
    pack: dict,
    *,
    words: list[Word] | None,
    still_s: float = STILL_S,
    pause_s: float = PAUSE_S,
    loudness_jump_lu: float = LOUDNESS_JUMP_LU,
) -> dict:
    """Return sorted machine entries without reading, writing, or decoding media.

    Holds and whole-picture masks suppress stillness. Region overlays suppress only
    when their rectangle, clipped to the frame, covers at least 10 percent of its area
    (GRAPHIC_MIN_AREA = 0.10).
    """
    settings = _settings(still_s, pause_s, loudness_jump_lu)
    try:
        fps = Fraction(pack["video"]["fps"])
        duration = pack["video"]["duration_s"]
        if fps <= 0 or not _number(duration) or duration <= 0:
            raise ValidationError("video", "expected positive fps and duration")
        if words is not None:
            _validate_words(words, duration, fps)
        findings = _still_findings(pack, fps, still_s)
        rules = {"watch.still": {"status": "ran", "reason": None, "basis": "motion"}}
        has_audio = pack["video"]["has_audio"] and pack["loudness"]["status"] != "no_audio"
        rules["watch.cut_loudness"] = {
            "status": "ran" if has_audio else "skipped",
            "reason": None if has_audio else "video has no audio",
            "basis": "cut_loudness" if has_audio else None,
        }
        if has_audio:
            findings.extend(_cut_findings(pack, loudness_jump_lu))
        basis = "words" if words is not None else "silence"
        has_pauses = words is not None or bool(pack["loudness"]["silences"])
        rules["watch.pause"] = {
            "status": "ran" if has_pauses else "skipped",
            "reason": None if has_pauses else "no transcript words or measured silences",
            "basis": basis if has_pauses else None,
        }
        if has_pauses:
            findings.extend(_pause_findings(pack, words, pause_s))
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValidationError("pack", "invalid scan measurements or video facts") from exc
    findings.sort(key=lambda finding: (finding["start_s"], finding["check"]))
    return {"findings": findings, "rules": rules, "settings": settings}


def _stale_pack(name: str, reason: str) -> OutputPathError:
    error = OutputPathError(f"Watch pack {name} {reason}.", code="stale_pack")
    error.suggested_action = "Run yap-studio watch pack again for this video."
    return error


def _check_pending(directory: Path) -> None:
    """Refuse interrupted publications, including dangling journal links."""
    name = "pack.pending.json"
    try:
        (directory / name).lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _stale_pack(name, "cannot be checked") from exc
    raise _stale_pack(name, "remains from an interrupted publication")


def _verify_outputs(directory: Path, pack: dict) -> bytes | None:
    """Verify all listed outputs and keep the exact transcript bytes that were hashed."""
    outputs = pack.get("outputs")
    if not isinstance(outputs, list):
        raise _stale_pack("pack.json", "has no valid output receipt")
    transcript_bytes = None
    seen: set[str] = set()
    for item in outputs:
        if not isinstance(item, dict):
            raise _stale_pack("pack.json", "has an invalid output receipt")
        try:
            name = pack_output_name(item.get("path"))
        except OutputPathError as exc:
            raise _stale_pack(str(item.get("path")), "is an unknown or unsafe output name") from exc
        if name in seen:
            raise _stale_pack(name, "is listed more than once")
        seen.add(name)
        target = directory / name
        try:
            if not stat.S_ISDIR(directory.lstat().st_mode) or not stat.S_ISDIR(
                target.parent.lstat().st_mode
            ):
                raise _stale_pack(name, "has a symlink or non-directory parent")
            if not stat.S_ISREG(target.lstat().st_mode):
                raise _stale_pack(name, "is a symlink or non-regular file")
            if not target.resolve().is_relative_to(directory.resolve()):
                raise _stale_pack(name, "resolves outside the pack folder")
            content = target.read_bytes()
        except (OSError, RuntimeError) as exc:
            raise _stale_pack(name, "is missing or unreadable") from exc
        if hashlib.sha256(content).hexdigest() != item.get("sha256"):
            raise _stale_pack(name, "has changed or has an invalid sha256")
        if name == "transcript.json":
            transcript_bytes = content
    return transcript_bytes


def _read_words(directory: Path, pack: dict, transcript_bytes: bytes | None) -> list[Word] | None:
    if pack["transcript"]["status"] != "ok":
        return None
    name = pack["transcript"]["file"]
    if name != "transcript.json":
        raise _stale_pack(str(name), "must refer to transcript.json in the pack")
    if transcript_bytes is None:
        raise _stale_pack(name, "is not listed in outputs")
    target = directory / name
    try:
        data = json.loads(transcript_bytes.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InputError(str(target), "unreadable packed transcript JSON") from exc
    try:
        if not isinstance(data, dict) or not isinstance(data.get("words"), list):
            raise ValueError("expected a transcript object with a words list")
        # Validate raw types before the transcript reader coerces numbers and text.
        raw_words = [Word(**item) for item in data["words"]]
        _validate_words(raw_words, pack["video"]["duration_s"], Fraction(pack["video"]["fps"]))
        duration = data.get("duration_s")
        if duration is not None and (not _number(duration) or duration <= 0):
            raise ValidationError("transcript.duration_s", "must be finite and greater than zero")
        return Transcript.from_dict(data).words
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValidationError("transcript", "invalid packed transcript words or times") from exc


def run_scan(
    pack_dir: str | Path,
    *,
    findings_path: str | Path | None = None,
    dry_run: bool = False,
    still_s: float = STILL_S,
    pause_s: float = PAUSE_S,
    loudness_jump_lu: float = LOUDNESS_JUMP_LU,
) -> dict:
    """Check pack freshness and replace machine findings, preserving agent notes."""
    _settings(still_s, pause_s, loudness_jump_lu)
    directory = Path(pack_dir)
    _check_pending(directory)
    pack = load_pack(directory)
    transcript_bytes = _verify_outputs(directory, pack)
    try:
        video = Path(pack["video"]["path"])
        digest = pack["video"]["sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("pack", "missing or invalid video facts") from exc
    try:
        fresh = video.is_file() and sha256_file(video) == digest
    except OSError:
        fresh = False
    if not fresh:
        raise _stale_pack(f"video {video}", "is missing or has changed")
    try:
        words = _read_words(directory, pack, transcript_bytes)
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("pack", "missing or invalid transcript facts") from exc
    result = scan_pack(
        pack, words=words, still_s=still_s, pause_s=pause_s, loudness_jump_lu=loudness_jump_lu
    )
    result.update(path=None, recorded=False)
    if not dry_run:
        recorded = record_findings(
            video, result["findings"], source="machine", reviewer=None, findings_path=findings_path
        )
        result.update(
            {key: recorded[key] for key in ("path", "added", "duplicates", "removed", "total")}
        )
        result["recorded"] = True
    return result
