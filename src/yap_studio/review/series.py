"""Strict review series records, replies, and pure thread construction."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime

from yap_studio.errors import ValidationError
from yap_studio.review.export import comment_to_dict
from yap_studio.review.model import Comment, sorted_comments

SERIES_SCHEMA = "yap-studio/review-series@1"
REPLY_STATUSES = ("changed", "kept", "declined", "needs_clarification")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPLY_ID = re.compile(r"r([1-9][0-9]*)\Z")
_COMMENT_ID = re.compile(r"c[1-9][0-9]*\Z")


@dataclass(frozen=True)
class SeriesEntry:
    """Media identity and optional immutable project version provenance."""

    sha256: str
    name: str
    version_id: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class SeriesVersion:
    """A numbered video added to a review series."""

    number: int
    sha256: str
    name: str
    version_id: str | None
    project: str | None
    added_at: str


@dataclass(frozen=True)
class Reply:
    """A response in a later version to an earlier version's comment."""

    id: str
    comment_version: int
    comment_id: str
    version: int
    status: str
    text: str
    evidence: dict | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Series:
    """An ordered video history and its programmatically attached responses."""

    id: str
    name: str
    versions: tuple[SeriesVersion, ...]
    replies: tuple[Reply, ...]


@dataclass(frozen=True)
class Thread:
    """An earlier comment with its latest applicable reply or open state."""

    version: int
    comment: Comment
    reply: Reply | None
    state: str


def reply_to_dict(reply: Reply) -> dict:
    """Return a reply with a qualified comment reference."""
    return {
        "id": reply.id,
        "comment": {"version": reply.comment_version, "id": reply.comment_id},
        "version": reply.version,
        "status": reply.status,
        "text": reply.text,
        "evidence": copy.deepcopy(reply.evidence),
        "created_at": reply.created_at,
        "updated_at": reply.updated_at,
    }


def series_to_dict(series: Series, *, next_reply_id: int | None = None) -> dict:
    """Return the persisted series, optionally retaining a higher id counter."""
    if next_reply_id is None:
        next_reply_id = max((int(item.id[1:]) for item in series.replies), default=0) + 1
    return {
        "schema": SERIES_SCHEMA,
        "id": series.id,
        "name": series.name,
        "versions": [asdict(item) for item in series.versions],
        "next_reply_id": next_reply_id,
        "replies": [reply_to_dict(item) for item in series.replies],
    }


def _fields(data: object, keys: set[str]) -> dict:
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"expected exactly these fields: {', '.join(sorted(keys))}")
    return data


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value: object) -> None:
    if (not isinstance(value, str)
            or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value) is None):
        raise ValueError("expected a UTC timestamp")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def validate_reply_text(text: object) -> str:
    """Strip and validate the public reply text input."""
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
        raise ValidationError("text", "must contain 1 to 2000 characters")
    return text.strip()


def validate_evidence(evidence: object) -> dict | None:
    """Validate finite JSON evidence and copy it away from caller-owned data."""
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValidationError("evidence", "must be an object or null")
    try:
        payload = json.dumps(evidence, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValidationError("evidence", "must contain JSON values") from exc
    if len(payload.encode("utf-8")) > 16384:
        raise ValidationError("evidence", "compact JSON must be at most 16384 bytes")
    return json.loads(payload)


def series_from_dict(data: object) -> Series:
    """Strictly parse a series, rejecting invalid sequences and reply references."""
    data = _fields(data, {"schema", "id", "name", "versions", "next_reply_id", "replies"})
    if data["schema"] != SERIES_SCHEMA or not _text(data["name"]):
        raise ValueError("invalid series schema or name")
    if not isinstance(data["id"], str) or _SHA256.fullmatch(data["id"]) is None:
        raise ValueError("invalid series id")
    if not isinstance(data["versions"], list) or not data["versions"]:
        raise ValueError("versions must be a nonempty array")
    versions = []
    hashes = set()
    for number, raw in enumerate(data["versions"], 1):
        item = _fields(raw, {"number", "sha256", "name", "version_id", "project", "added_at"})
        if type(item["number"]) is not int or item["number"] != number:
            raise ValueError("version numbers must be contiguous from 1")
        sha = item["sha256"]
        if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None or sha in hashes:
            raise ValueError("invalid or duplicate media hash")
        hashes.add(sha)
        if not _text(item["name"]) or (item["project"] is not None and not _text(item["project"])):
            raise ValueError("invalid version name or project")
        if item["version_id"] is not None and (
            not isinstance(item["version_id"], str)
            or re.fullmatch(r"v[1-9][0-9]*", item["version_id"]) is None
        ):
            raise ValueError("invalid project version id")
        _timestamp(item["added_at"])
        versions.append(SeriesVersion(**item))
    if data["id"] != versions[0].sha256:
        raise ValueError("series id must be the first media hash")
    counter = data["next_reply_id"]
    if type(counter) is not int or counter < 1 or not isinstance(data["replies"], list):
        raise ValueError("invalid reply counter or array")
    replies = []
    ids, references = set(), set()
    for raw in data["replies"]:
        item = _fields(raw, {
            "id", "comment", "version", "status", "text", "evidence", "created_at", "updated_at"
        })
        comment = _fields(item["comment"], {"version", "id"})
        reply_id = item["id"]
        if not isinstance(reply_id, str) or _REPLY_ID.fullmatch(reply_id) is None:
            raise ValueError("invalid reply id")
        if reply_id in ids or int(reply_id[1:]) >= counter:
            raise ValueError("invalid reply id sequence")
        ids.add(reply_id)
        cv, version = comment["version"], item["version"]
        if (type(cv) is not int or type(version) is not int
                or not 1 <= cv < version <= len(versions)):
            raise ValueError("reply must refer to an earlier comment version")
        if not isinstance(comment["id"], str) or _COMMENT_ID.fullmatch(comment["id"]) is None:
            raise ValueError("invalid comment id")
        reference = (cv, comment["id"], version)
        if reference in references or item["status"] not in REPLY_STATUSES:
            raise ValueError("duplicate reply reference or invalid status")
        references.add(reference)
        try:
            text = validate_reply_text(item["text"])
            evidence = validate_evidence(item["evidence"])
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        if text != item["text"]:
            raise ValueError("reply text must be stripped")
        _timestamp(item["created_at"])
        _timestamp(item["updated_at"])
        replies.append(Reply(reply_id, cv, comment["id"], version, item["status"], text,
                             evidence, item["created_at"], item["updated_at"]))
    return Series(data["id"], data["name"], tuple(versions), tuple(replies))


def thread_to_dict(thread: Thread) -> dict:
    """Return a comment record with its version, reply, and state."""
    return {
        **comment_to_dict(thread.comment),
        "version": thread.version,
        "reply": reply_to_dict(thread.reply) if thread.reply is not None else None,
        "state": thread.state,
    }


def build_threads(
    series: Series, comments_by_version: dict[int, list[Comment]], as_of: int
) -> list[Thread]:
    """Show each earlier comment with its latest reply at the selected version."""
    if type(as_of) is not int or not 1 <= as_of <= len(series.versions):
        raise ValidationError("as_of", "must name a version in the series")
    latest: dict[tuple[int, str], Reply] = {}
    for reply in series.replies:
        key = (reply.comment_version, reply.comment_id)
        if reply.version <= as_of and (key not in latest or reply.version > latest[key].version):
            latest[key] = reply
    threads = []
    for version in range(1, as_of):
        for comment in sorted_comments(comments_by_version.get(version, [])):
            reply = latest.get((version, comment.id))
            threads.append(Thread(version, comment, reply, reply.status if reply else "open"))
    return threads


def is_resolved(state: str) -> bool:
    """Return whether a thread has a completed disposition."""
    return state in ("changed", "kept", "declined")
