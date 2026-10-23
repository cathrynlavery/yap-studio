"""Plain-text and JSON representations of review feedback."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from yap_studio.checks.model import format_time
from yap_studio.review.model import (
    TARGETS,
    Comment,
    MediaFacts,
    frame_start,
    frame_time_text,
    legacy_kind,
    normalize_request,
    sorted_comments,
    validate_stored_pin,
)

if TYPE_CHECKING:
    from yap_studio.feedback.versions import LayerWindow
    from yap_studio.review.series import Thread
    from yap_studio.review.server import ReviewSeries
    from yap_studio.review.store import ReviewStore

_COMMENT_KEYS = (
    "id",
    "start_frame",
    "end_frame",
    "start_s",
    "end_s",
    "kind",
    "target",
    "text",
    "created_at",
    "updated_at",
    "request",
)
_LEGACY_COMMENT_KEYS = tuple(
    key for key in _COMMENT_KEYS if key not in ("kind", "target", "request")
) + (
    "label",
)
_TARGET_TEXT = {"element": "graphic", "insert": "B-roll", "overlay": "overlay",
                "video": "whole video"}


def _target_text(comment: Comment, facts: MediaFacts,
                 layers: Sequence[LayerWindow] | None) -> str:
    target = comment.target
    if target == "element" and layers is not None:
        start = (float(frame_start(comment.start_frame, facts.fps))
                 if facts.cfr and facts.fps is not None and comment.start_frame is not None
                 else comment.start_s)
        end = (float(frame_start(comment.end_frame, facts.fps))
               if facts.cfr and facts.fps is not None and comment.end_frame is not None
               else comment.end_s)
        kinds = {layer.kind for layer in layers if layer.kind != "caption"
                 and (layer.start <= start < layer.end if end is None or end == start else
                      layer.start < end and layer.end > start)}
        if len(kinds) == 1:
            target = kinds.pop()
    return _TARGET_TEXT.get(target, target or "")


def comment_to_dict(comment: Comment) -> dict:
    """Keep field order and append a pin only when it is present."""
    record = {key: getattr(comment, key) for key in _COMMENT_KEYS}
    record["request"] = dict(comment.request) if comment.request is not None else None
    if comment.pin is not None:
        record["pin"] = dict(comment.pin)
    return record


def comment_from_dict(data: dict, *, legacy: bool | None = None) -> Comment:
    """Rebuild an exact current or legacy record, optionally requiring one shape."""
    if isinstance(data, dict):
        if legacy is not True and set(_COMMENT_KEYS) - {"request"} <= data.keys() <= (
            set(_COMMENT_KEYS) | {"pin"}
        ):
            if data["kind"] not in ("keep", "fix"):
                raise ValueError("kind must be 'keep' or 'fix'")
            if data["target"] is not None and data["target"] not in TARGETS:
                raise ValueError("target must be None or one of the comment targets")
            request = normalize_request(data.get("request"), kind=data["kind"],
                                        target=data["target"])
            pin = validate_stored_pin(data["pin"], **{
                key: data[key] for key in ("start_frame", "end_frame", "start_s", "end_s")
            }) if "pin" in data else None
            return Comment(**{**data, "request": request, "pin": pin})
        if legacy is not False and set(data) == set(_LEGACY_COMMENT_KEYS):
            fields = {key: value for key, value in data.items() if key != "label"}
            return Comment(**fields, kind=legacy_kind(data["label"]), target=None)
    raise ValueError("comment record must contain exactly the comment fields")


def _plural(count: int, word: str) -> str:
    suffix = "" if count == 1 else ("es" if word == "fix" else "s")
    return f"{count} {word}{suffix}"


def _request_text(comment: Comment) -> str:
    request = comment.request
    if request is not None and request["type"] == "spelling":
        return f' (spelling: "{request["from"]}" -> "{request["to"]}")'
    if request is not None and request["type"] == "restore":
        return (f' (restore: {request["take"]} '
                f'{format_time(request["start"])}-{format_time(request["end"])})')
    if request is not None and request["type"] == "cut":
        text = " ".join(word["text"] for word in request["words"])
        return f' (cut: "{text[:117] + "..." if len(text) > 120 else text}")'
    return ""


def _time_span(comment: Comment, facts: MediaFacts) -> str:
    if facts.cfr:
        if facts.fps is None or comment.start_frame is None:
            raise ValueError("constant frame rate comments require fps and a start frame")
        start = frame_time_text(comment.start_frame, facts.fps)
        end = (
            frame_time_text(comment.end_frame, facts.fps) if comment.end_frame is not None else None
        )
    else:
        start = format_time(comment.start_s)
        end = format_time(comment.end_s) if comment.end_s is not None else None
    return f"{start}-{end}" if end is not None else start


def _pin_text(comment: Comment, facts: MediaFacts) -> str:
    pin = comment.pin
    if pin is None:
        return ""
    across = math.floor(pin["x"] * 100 + 0.5)
    down = math.floor(pin["y"] * 100 + 0.5)
    suffix = f" (pinned {across}% across, {down}% down"
    if facts.cfr and facts.fps is not None and pin["frame"] is not None:
        if pin["frame"] != comment.start_frame:
            suffix += f" at {frame_time_text(pin['frame'], facts.fps)}"
    elif pin["time_s"] != comment.start_s and comment.end_s is not None:
        suffix += f" at {format_time(pin['time_s'])}"
    return suffix + ")"


def _export_comment(comment: Comment, facts: MediaFacts) -> dict:
    record = comment_to_dict(comment)
    if facts.cfr:
        if facts.fps is None or comment.start_frame is None:
            raise ValueError("constant frame rate comments require fps and a start frame")
        record["start_s"] = round(float(frame_start(comment.start_frame, facts.fps)), 4)
        record["end_s"] = (
            round(float(frame_start(comment.end_frame, facts.fps)), 4)
            if comment.end_frame is not None
            else None
        )
    else:
        record["start_frame"] = None
        record["end_frame"] = None
        if comment.pin is not None:
            record["pin"]["frame"] = None
    return record


def format_text(
    facts: MediaFacts, comments: list[Comment], reviewer: str | None, exported_on: date,
    *, version: tuple[int, int] | None = None,
    layers: Sequence[LayerWindow] | None = None,
) -> str:
    """Format feedback as a readable, chronologically ordered text document."""
    duration = format_time(facts.duration_s) if facts.duration_s is not None else "unknown duration"
    rate = (
        f"{float(facts.fps):.2f}".rstrip("0").rstrip(".") + " fps"
        if facts.cfr and facts.fps is not None
        else "variable frame rate"
    )
    ordered = sorted_comments(comments)
    version_text = f", version {version[0]} of {version[1]}" if version is not None else ""
    lines = [f"Feedback on {facts.name}{version_text} ({duration}, {rate})"]
    if reviewer:
        lines.append(f"From {reviewer}, {exported_on.isoformat()}")
    counts = _plural(len(ordered), "comment")
    if ordered:
        fixes = sum(comment.kind == "fix" for comment in ordered)
        keeps = len(ordered) - fixes
        parts = []
        if fixes:
            parts.append(_plural(fixes, "fix"))
        if keeps:
            parts.append(_plural(keeps, "keep"))
        counts += ": " + ", ".join(parts)
    lines.append(counts)
    if ordered:
        lines.append("")
    for comment in ordered:
        span = _time_span(comment, facts)
        text_lines = comment.text.split("\n")
        tag = comment.kind.capitalize()
        if comment.target is not None:
            tag += ": " + _target_text(comment, facts, layers)
        lines.append(f"{span} [{tag}]{_request_text(comment)}{_pin_text(comment, facts)} "
                     f"{text_lines[0]}")
        lines.extend(" " * (len(span) + 1) + line for line in text_lines[1:])
    return "\n".join(lines) + "\n"


def _evidence_fields(
    value: object, required: set[str], optional: set[str] | frozenset[str] = frozenset(),
) -> bool:
    return isinstance(value, dict) and required <= value.keys() <= required | optional


def _evidence_seconds(value: object) -> bool:
    try:
        return type(value) in (int, float) and value >= 0 and math.isfinite(value)
    except OverflowError:
        return False


def _reply_evidence(value: object) -> dict | None:
    """Read the writer's complete typed schema, leaving legacy or malformed evidence alone."""
    if not _evidence_fields(value, {
        "schema", "batch", "comment_version", "version", "media_sha256", "disposition",
        "operations", "moment", "unmapped", "change", "keep", "question",
    }) or value["schema"] != "yap-studio/reply-evidence@1":
        return None

    def strings(record: dict, keys: tuple[str, ...]) -> bool:
        return all(isinstance(record[key], str) for key in keys)

    def nullable_string(item: object) -> bool:
        return item is None or isinstance(item, str)

    def frame(item: object) -> bool:
        return item is None or (type(item) is int and 0 <= item <= 2**53 - 1)

    if (not strings(value, ("batch", "comment_version", "version", "media_sha256", "disposition"))
            or not nullable_string(value["unmapped"])
            or not isinstance(value["operations"], list)
            or not all(_evidence_fields(item, {"id"}, {"type"}) and strings(item, ("id",))
                       and ("type" not in item or strings(item, ("type",)))
                       for item in value["operations"])):
        return None
    moment = value["moment"]
    if moment is not None and (
        not _evidence_fields(moment, {"status", "start_s", "end_s", "start_frame", "end_frame"})
        or moment["status"] not in ("mapped", "moved", "ambiguous", "partial", "removed", "time")
        or not _evidence_seconds(moment["start_s"]) or not _evidence_seconds(moment["end_s"])
        or moment["end_s"] < moment["start_s"]
        or not frame(moment["start_frame"]) or not frame(moment["end_frame"])
        or (moment["start_frame"] is not None and moment["end_frame"] is not None
            and moment["end_frame"] < moment["start_frame"])
    ):
        return None

    def words(items: object) -> bool:
        return isinstance(items, list) and all(
            _evidence_fields(item, {"id", "segment", "text"})
            and strings(item, ("id", "segment", "text")) for item in items
        )

    change = value["change"]
    if change is not None and (
        not _evidence_fields(change, {"before", "after", "differences"})
        or not words(change["before"]) or not words(change["after"])
        or not isinstance(change["differences"], list)
        or not all(item in ("removed", "words", "order", "timing", "pause", "layers")
                   for item in change["differences"])
    ):
        return None
    keep = value["keep"]
    if keep is not None and (
        not _evidence_fields(keep, {"status", "detail"})
        or keep["status"] not in ("kept", "broken", "unmeasured")
        or not strings(keep, ("detail",))
    ):
        return None
    question = value["question"]
    if question is not None and (
        not _evidence_fields(question, {"question", "answer", "option"})
        or not strings(question, ("question",)) or not nullable_string(question["answer"])
        or not nullable_string(question["option"])
    ):
        return None
    return value


def format_threads_summary(threads: Sequence[Thread]) -> str:
    """Summarize earlier notes without copying their text."""
    if not threads:
        return ""
    count = len(threads)
    open_count = sum(item.state in ("open", "needs_clarification") for item in threads)
    state = f"{open_count} still open" if open_count else "all replied"
    return f"Earlier versions: {count} {'note' if count == 1 else 'notes'}, {state}\n"


def earlier_threads(threads: Sequence[Thread], as_of: int | None) -> list[Thread]:
    """Keep open notes and replies made in the selected version or later."""
    if as_of is None:
        return list(threads)
    return [item for item in threads if item.state in ("open", "needs_clarification")
            or item.reply is None or item.reply.version >= as_of]


def format_threads(threads: Sequence[Thread], facts_by_version: Mapping[int, MediaFacts], *,
                   layers_by_version: Mapping[int, Sequence[LayerWindow]] | None = None,
                   as_of: int | None = None) -> str:
    """Format earlier comments with replies or open markers using their own timelines."""
    threads = earlier_threads(threads, as_of)
    if not threads:
        return ""
    lines = []
    if as_of is None:
        open_count = sum(item.state in ("open", "needs_clarification") for item in threads)
        lines.append(f"Earlier versions: {open_count} open, {len(threads) - open_count} replied")
    labels = {"changed": "Changed", "kept": "Kept", "declined": "Declined",
              "needs_clarification": "Question"}
    for thread in threads:
        comment = thread.comment
        span = _time_span(comment, facts_by_version[thread.version])
        tag = comment.kind.capitalize()
        if comment.target is not None:
            tag += ": " + _target_text(comment, facts_by_version[thread.version],
                                       (layers_by_version or {}).get(thread.version))
        prefix = f"v{thread.version} {span} "
        text_lines = comment.text.split("\n")
        lines.append(f"{prefix}[{tag}]{_request_text(comment)}"
                     f"{_pin_text(comment, facts_by_version[thread.version])} {text_lines[0]}")
        lines.extend(" " * len(prefix) + line for line in text_lines[1:])
        if thread.reply is None:
            lines.append("  Open")
        else:
            reply = thread.reply
            prefix = f"  {labels[reply.status]} in v{reply.version}"
            evidence = _reply_evidence(reply.evidence)
            if evidence is not None:
                moment = evidence["moment"]
                if moment is not None:
                    facts = facts_by_version[reply.version]
                    time = (frame_time_text(moment["start_frame"], facts.fps)
                            if facts.cfr and facts.fps is not None
                            and moment["start_frame"] is not None
                            else format_time(moment["start_s"]))
                    if moment["status"] == "removed":
                        prefix += f" (removed, join at {time})"
                    elif moment["status"] == "partial":
                        prefix += f" (partly removed, at {time})"
                    else:
                        prefix += f" at {time}"
                if evidence["keep"] is not None:
                    check = evidence["keep"]["status"]
                    prefix += f" (checked: {'not measured' if check == 'unmeasured' else check})"
            prefix += ": "
            text_lines = reply.text.split("\n")
            lines.append(prefix + text_lines[0])
            lines.extend(" " * len(prefix) + line for line in text_lines[1:])
    return "\n".join(lines) + "\n"


def to_export_json(
    facts: MediaFacts, comments: list[Comment], reviewer: str | None, exported_at: datetime
) -> dict:
    """Build the versioned feedback export with a UTC timestamp."""
    if exported_at.tzinfo is None or exported_at.utcoffset() is None:
        raise ValueError("exported_at must be timezone-aware")
    timestamp = exported_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "format": "yap-studio-feedback",
        "version": 2,
        "media": {
            "name": facts.name,
            "sha256": facts.sha256,
            "duration_s": facts.duration_s,
            "fps": f"{facts.fps.numerator}/{facts.fps.denominator}"
            if facts.cfr and facts.fps is not None
            else None,
        },
        "reviewer": reviewer,
        "exported_at": timestamp,
        "comments": [_export_comment(comment, facts) for comment in sorted_comments(comments)],
    }


def export_filename(media_name: str, extension: str) -> str:
    """Return the default feedback file name for a video and extension."""
    return f"{Path(media_name).stem}-feedback.{extension.lstrip('.')}"


def series_export(store: ReviewStore, series: ReviewSeries, version: int) -> dict:
    """Share the exact version identities and earlier threads used by JSON exports."""
    from yap_studio.review.series import thread_to_dict

    return {
        "id": series.id,
        "version": version,
        "versions": [{"number": item.number, "name": item.media.name,
                      "sha256": item.sha256, "version_id": item.version_id}
                     for item in series.versions],
        "threads": [thread_to_dict(item) for item in store.threads(series.id, version)],
    }
