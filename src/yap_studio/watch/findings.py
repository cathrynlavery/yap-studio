"""Validated, deterministic reviewer findings for one exact render."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg, paths
from yap_studio.errors import InputError, OutputPathError, ValidationError, YapError
from yap_studio.receipt import sha256_file
from yap_studio.review.model import frame_at

from .images import format_time

FINDINGS_SCHEMA = "yap-studio/watch-findings@1"
KINDS = ("slow", "jarring_cut", "face_covered", "caption_issue", "audio", "other")
SEVERITIES = ("low", "medium", "high")
SOURCES = ("agent", "machine")
_MAX_FINDINGS = 500
_TIME = re.compile(r"(?:(\d+):)?(\d+):([0-5]\d(?:\.\d+)?)")
_ENTRY_KEYS = {"at", "start", "end", "kind", "severity", "reason", "fix"}


@dataclass(frozen=True)
class Finding:
    """A point or half-open range, kept separate from creator comments."""

    id: str
    source: str
    reviewer: str | None
    kind: str
    severity: str
    start_s: float
    end_s: float | None
    start_frame: int
    end_frame: int | None
    reason: str
    fix: str
    check: str | None
    evidence: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        """Return the schema's JSON fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        """Read a finding, refusing malformed fields."""
        try:
            finding = cls(**data)
        except (TypeError, ValueError) as exc:
            raise ValidationError("finding", "expected all finding fields") from exc
        problems: list[str] = []
        if not isinstance(finding.id, str) or not re.fullmatch(r"f[1-9]\d*", finding.id):
            problems.append("id: must be f followed by a positive integer")
        _choice(finding.source, SOURCES, "source", problems)
        _choice(finding.kind, KINDS, "kind", problems)
        _choice(finding.severity, SEVERITIES, "severity", problems)
        for key in ("reason", "fix"):
            _text(getattr(finding, key), key, 1000, problems, multiline=True)
        if finding.reviewer is not None:
            _text(finding.reviewer, "reviewer", 64, problems)
        for key in ("start_s", "end_s"):
            value = getattr(finding, key)
            if key == "end_s" and value is None:
                continue
            if not _number(value) or value < 0:
                problems.append(f"{key}: must be a finite non-negative number")
        if _number(finding.start_s) and _number(finding.end_s):
            if finding.end_s <= finding.start_s:
                problems.append("end_s: must be greater than start_s")
        for key in ("start_frame", "end_frame"):
            value = getattr(finding, key)
            if key == "end_frame" and value is None:
                continue
            if not _integer(value) or value < 0:
                problems.append(f"{key}: must be a non-negative integer")
        if _integer(finding.start_frame) and _integer(finding.end_frame):
            if finding.end_frame <= finding.start_frame:
                problems.append("end_frame: must be greater than start_frame")
        if (finding.end_s is None) != (finding.end_frame is None):
            problems.append("end_frame: must be null exactly when end_s is null")
        _metadata(finding.check, finding.evidence, finding.source, "finding", problems)
        _raise(problems)
        return finding


def default_findings_path(video: str | Path) -> Path:
    """Place the findings beside the video."""
    video = Path(video)
    return video.with_name(f"{video.stem}.findings.json")


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _raise(problems: list[str]) -> None:
    if problems:
        raise ValidationError("findings", "\n" + "\n".join(problems))


def _choice(value: Any, choices: tuple[str, ...], key: str, problems: list[str]) -> None:
    if value not in choices:
        problems.append(f"{key}: must be one of {', '.join(choices)}")


def _text(
    value: Any, key: str, limit: int, problems: list[str], *, multiline: bool = False
) -> str | None:
    if not isinstance(value, str):
        problems.append(f"{key}: must be a string")
        return None
    if any(
        unicodedata.category(char) in {"Cc", "Cf", "Cs"} and not (multiline and char == "\n")
        for char in value
    ):
        problems.append(f"{key}: must not contain control characters")
    value = value.strip()
    if not 1 <= len(value) <= limit:
        problems.append(f"{key}: must contain 1 to {limit} characters")
    return value


def _seconds(value: Any, key: str, limit: float, problems: list[str]) -> float | None:
    if isinstance(value, str):
        match = _TIME.fullmatch(value.strip())
        if match is None:
            problems.append(f"{key}: must be seconds or M:SS(.f) / H:MM:SS(.f)")
            return None
        try:
            if match[1] is not None and (len(match[2]) != 2 or int(match[2]) >= 60):
                problems.append(f"{key}: minutes must be MM from 00 to 59 in H:MM:SS(.f)")
                return None
            value = int(match[1] or 0) * 3600 + int(match[2]) * 60 + float(match[3])
        except (ValueError, OverflowError):
            value = float("inf")
    if not _number(value) or value < 0:
        problems.append(f"{key}: must be a finite non-negative time")
        return None
    if value > limit:
        problems.append(f"{key}: is past the video end (including one frame of tolerance)")
    return float(value)


def _position(start: float, end: float | None, fps: Fraction, frames: int) -> dict[str, Any]:
    start_frame = min(frame_at(start, fps), frames - 1)
    end_frame = (
        min(frames, max(start_frame + 1, math.ceil(Fraction(end) * fps - Fraction(1, 1000))))
        if end is not None
        else None
    )
    return {
        "start_s": start,
        "end_s": end,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def parse_findings_input(
    data: Any, *, fps: Fraction, duration_s: float, frames: int
) -> tuple[str | None, list[dict[str, Any]]]:
    """Validate the entire agent batch, reporting all indexed problems together."""
    fps = Fraction(fps)
    if fps <= 0 or not _number(duration_s) or duration_s <= 0:
        raise ValidationError("video", "must have a positive fps and duration")
    if not _integer(frames) or frames < 1:
        raise ValidationError("frames", "must be a positive integer")
    problems: list[str] = []
    reviewer = None
    if isinstance(data, dict):
        for key in sorted(data.keys() - {"reviewer", "findings"}):
            problems.append(f"{key}: unknown key")
        if "reviewer" in data:
            reviewer = _text(data["reviewer"], "reviewer", 64, problems)
        data = data.get("findings")
    if not isinstance(data, list):
        problems.append("findings: must be a list")
        _raise(problems)
    if len(data) > _MAX_FINDINGS:
        problems.append(f"findings: must contain at most {_MAX_FINDINGS} findings")
    entries = []
    for index, entry in enumerate(data):
        key = f"findings[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{key}: must be an object")
            continue
        for extra in sorted(entry.keys() - _ENTRY_KEYS):
            problems.append(f"{key}.{extra}: unknown key")
        _choice(entry.get("kind"), KINDS, f"{key}.kind", problems)
        _choice(entry.get("severity"), SEVERITIES, f"{key}.severity", problems)
        reason = _text(entry.get("reason"), f"{key}.reason", 1000, problems, multiline=True)
        fix = _text(entry.get("fix"), f"{key}.fix", 1000, problems, multiline=True)
        point = "at" in entry
        if point and ("start" in entry or "end" in entry):
            problems.append(f"{key}.at: use either at or start and end, not both")
        limit = duration_s + float(1 / fps)
        times = {
            name: _seconds(entry[name], f"{key}.{name}", limit, problems)
            for name in ("at", "start", "end")
            if name in entry
        }
        if not point:
            for name in ("start", "end"):
                if name not in entry:
                    problems.append(f"{key}.{name}: is required for a range")
        start = times.get("at" if point else "start")
        end = None if point else times.get("end")
        if not point and start is not None and end is not None and start >= end:
            problems.append(f"{key}.end: must be greater than start")
        if start is not None:
            entries.append(
                {
                    "kind": entry.get("kind"),
                    "severity": entry.get("severity"),
                    **_position(start, end, fps, frames),
                    "reason": reason,
                    "fix": fix,
                }
            )
    _raise(problems)
    return reviewer, entries


def _metadata(check: Any, evidence: Any, source: str, key: str, problems: list[str]) -> None:
    if check is not None:
        if source == "agent":
            problems.append(f"{key}.check: must be null for agent findings")
        _text(check, f"{key}.check", 128, problems)
    if not isinstance(evidence, dict) or any(
        not isinstance(name, str) or not _number(value) for name, value in evidence.items()
    ):
        problems.append(f"{key}.evidence: must be an object containing only finite numbers")


def _video_facts(video: str | Path) -> tuple[Path, Fraction, dict[str, Any]]:
    video = paths.validate_input(video)
    info = ffmpeg.probe(video, count_frames=False)
    fps = info.fps
    duration = info.video_duration_s or info.duration_s
    if not info.has_video or fps is None or fps <= 0 or not _number(duration) or duration <= 0:
        raise InputError(str(video), "video must have a positive frame rate and duration")
    frames = info.video_frames if info.video_frames is not None else round(duration * fps)
    if frames < 1:
        raise InputError(str(video), "video must contain at least one frame")
    return (
        video,
        fps,
        {
            "file": video.name,
            "fps": f"{fps.numerator}/{fps.denominator}",
            "frames": frames,
            "duration_s": duration,
        },
    )


def _read_existing(target: Path) -> dict[str, Any] | None:
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != FINDINGS_SCHEMA:
            raise ValueError(f"expected schema {FINDINGS_SCHEMA}")
        video = data["video"]
        if not isinstance(video, dict) or not isinstance(video["file"], str):
            raise ValueError("invalid video metadata")
        if not isinstance(video["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", video["sha256"]
        ):
            raise ValueError("invalid video sha256")
        fps = Fraction(video["fps"])
        frames, duration = video["frames"], video["duration_s"]
        if fps <= 0 or not _integer(frames) or frames < 1 or not _number(duration) or duration <= 0:
            raise ValueError("invalid video timeline")
        if not isinstance(data["findings"], list):
            raise ValueError("findings must be a list")
        findings = [Finding.from_dict(entry) for entry in data["findings"]]
        ids = [int(finding.id[1:]) for finding in findings]
        if len(set(ids)) != len(ids):
            raise ValueError("finding ids must be unique")
        if not _integer(data["next_id"]) or data["next_id"] <= max(ids, default=0):
            raise ValueError("next_id must be greater than every finding id")
        for finding in findings:
            if finding.start_frame >= frames or (
                finding.end_frame is not None and finding.end_frame > frames
            ):
                raise ValueError(f"{finding.id}: frames are outside the video")
            if max(finding.start_s, finding.end_s or 0) > duration + float(1 / fps):
                raise ValueError(f"{finding.id}: times are outside the video")
        data["findings"] = [finding.to_dict() for finding in findings]
        return data
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        RecursionError,
        ZeroDivisionError,
    ) as exc:
        raise InputError(str(target), f"malformed findings file: {exc}") from exc


def read_findings(path: Path) -> dict[str, Any] | None:
    """Read a findings file strictly without inspecting or hashing its video."""
    return _read_existing(path)


def _sort_key(finding: dict[str, Any]) -> tuple[int, int]:
    return finding["start_frame"], int(finding["id"][1:])


def _identity(finding: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        finding[key]
        for key in ("source", "kind", "severity", "start_frame", "end_frame", "reason", "fix")
    )


def note_from_file(
    video: str | Path,
    file: str | Path,
    *,
    findings_path: str | Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Read and validate an offline agent note, protecting both input files."""
    with paths.operation():
        source = paths.validate_input(file)
        maximum = 4 * 1024 * 1024
        try:
            with source.open("rb") as stream:
                payload = stream.read(maximum + 1)
            if len(payload) > maximum:
                raise InputError(str(source), "findings JSON exceeds the 4 MB size limit")
            data = json.loads(payload.decode("utf-8"))
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            raise InputError(str(source), "findings must be readable UTF-8 JSON") from exc
        video, fps, facts = _video_facts(video)
        reviewer, entries = parse_findings_input(
            data, fps=fps, duration_s=facts["duration_s"], frames=facts["frames"]
        )
        target = Path(findings_path) if findings_path is not None else default_findings_path(video)
        if "\x00" in str(target):
            raise OutputPathError("Findings path contains null bytes.", code="unsafe_path")
        if target.resolve() == source or (target.exists() and target.samefile(source)):
            raise OutputPathError("Findings path refers to the JSON input.", code="unsafe_path")
        return record_findings(
            video,
            entries,
            source="agent",
            reviewer=reviewer,
            findings_path=target,
            replace=replace,
        )


def record_findings(
    video: str | Path,
    entries: list[dict[str, Any]],
    *,
    source: str,
    reviewer: str | None = None,
    findings_path: str | Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Append agent findings or replace machine findings without reusing ids."""
    with paths.operation():
        video, fps, facts = _video_facts(video)
        target = Path(findings_path) if findings_path is not None else default_findings_path(video)
        if "\x00" in str(target):
            raise OutputPathError("Findings path contains null bytes.", code="unsafe_path")
        if target.is_symlink():
            raise OutputPathError("Findings path must not be a symlink.", code="unsafe_path")
        # Guard input aliases before attempting to read a supposed findings file.
        if target.resolve() == video or (target.exists() and target.samefile(video)):
            raise OutputPathError("Findings path refers to the video input.", code="unsafe_path")
        facts["sha256"] = sha256_file(video)
        problems: list[str] = []
        _choice(source, SOURCES, "source", problems)
        if reviewer is not None:
            reviewer = _text(reviewer, "reviewer", 64, problems)
        if not isinstance(entries, list):
            raise ValidationError("findings", "must be a list")
        raw = []
        metadata = []
        for index, entry in enumerate(entries):
            key = f"findings[{index}]"
            if not isinstance(entry, dict):
                raw.append(entry)
                metadata.append((None, {}))
                continue
            item = dict(entry)
            check, evidence = item.pop("check", None), item.pop("evidence", {})
            _metadata(check, evidence, source, key, problems)
            metadata.append((check, evidence))
            if "start_s" in item:
                if item.keys() & {"at", "start", "end"}:
                    problems.append(f"{key}: cannot mix start_s/end_s with at or start/end")
                start, end = item.pop("start_s"), item.pop("end_s", None)
                item.pop("start_frame", None)
                item.pop("end_frame", None)
                if end is None:
                    item["at"] = start
                else:
                    item["start"], item["end"] = start, end
            raw.append(item)
        try:
            _, validated = parse_findings_input(
                raw, fps=fps, duration_s=facts["duration_s"], frames=facts["frames"]
            )
        except ValidationError as exc:
            problems.append(exc.detail.lstrip())
            validated = []
        _raise(problems)
        try:
            previous = _read_existing(target)
        except YapError as exc:
            error = OutputPathError(exc.message, code="invalid_findings")
            error.suggested_action = "Move the old file aside or pass --findings NEW_PATH."
            raise error from exc
        if previous is not None and previous["video"]["sha256"] != facts["sha256"]:
            error = OutputPathError(
                "Findings were recorded for a different video render.", code="stale_findings"
            )
            error.suggested_action = "Move the old file aside or pass --findings NEW_PATH."
            raise error
        paths.validate_output(target)
        findings = previous["findings"] if previous is not None else []
        next_id = previous["next_id"] if previous is not None else 1
        kept = [
            finding
            for finding in findings
            if not (
                finding["source"] == source
                and (source == "machine" or (replace and finding["reviewer"] == reviewer))
            )
        ]
        removed = len(findings) - len(kept)
        identities = {_identity(finding) for finding in kept}
        added = duplicates = 0
        for entry, (check, evidence) in zip(validated, metadata, strict=True):
            finding = Finding(
                id=f"f{next_id}",
                source=source,
                reviewer=reviewer,
                **entry,
                check=check,
                evidence=evidence,
            ).to_dict()
            identity = _identity(finding)
            if identity in identities:
                duplicates += 1
                continue
            identities.add(identity)
            kept.append(finding)
            next_id += 1
            added += 1
        kept.sort(key=_sort_key)
        data = {"schema": FINDINGS_SCHEMA, "video": facts, "next_id": next_id, "findings": kept}
        try:
            paths.atomic_write_text(
                target, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
        except OSError as exc:
            raise OutputPathError(f"Could not write findings: {exc}", code="unsafe_path") from exc
        return {
            "path": str(target),
            "added": added,
            "duplicates": duplicates,
            "removed": removed,
            "total": len(kept),
            "findings": kept,
        }


def load_findings(video: str | Path, *, findings_path: str | Path | None = None) -> dict[str, Any]:
    """Load a render's findings and report whether its content hash has changed."""
    with paths.operation():
        video = paths.validate_input(video)
        target = Path(findings_path) if findings_path is not None else default_findings_path(video)
        data = _read_existing(target)
        return {
            "path": str(target),
            "exists": data is not None,
            "stale": data is not None and data["video"]["sha256"] != sha256_file(video),
            "video": data["video"] if data is not None else None,
            "findings": sorted(data["findings"], key=_sort_key) if data is not None else [],
        }


def format_findings(findings: list[dict[str, Any] | Finding]) -> list[str]:
    """Format points and ranges using watch's hundredth-second clock."""
    entries = [
        finding.to_dict() if isinstance(finding, Finding) else finding for finding in findings
    ]
    lines = []
    for finding in sorted(entries, key=lambda entry: (entry["start_s"], int(entry["id"][1:]))):
        time = format_time(finding["start_s"])
        if finding["end_s"] is not None:
            time += f" to {format_time(finding['end_s'])}"
        source = finding["source"]
        if finding["reviewer"] is not None:
            source += f" ({finding['reviewer']})"
        reason = finding["reason"].replace("\n", " ")
        fix = finding["fix"].replace("\n", " ")
        lines.append(
            f"{time} | {finding['kind']} | {finding['severity']} | {source} | {reason} | fix: {fix}"
        )
    return lines
