"""Typed taste records, canonical identities, and hand validation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from yap_studio.errors import ValidationError, YapError
from yap_studio.style import loader

TASTE_SCHEMA = "yap-studio/taste@1"
PROPOSALS_SCHEMA = "yap-studio/taste-proposals@1"
SNAPSHOT_SCHEMA = "yap-studio/taste-snapshot@1"
_BODY_FIELDS = ("scope", "condition", "sentence", "machine", "sources", "context")
_IDENTITY_FIELDS = ("scope", "condition", "sentence", "machine")
_SECTION_SCOPES = {
    "captions": "captions",
    "hook": "cards",
    "chapters": "cards",
    "graphics": "cards",
    "broll": "broll",
    "music": "music",
}
_ID = re.compile(r"r-[0-9a-f]{12}")
_HASH = re.compile(r"[0-9a-f]{64}")
_TAG = re.compile(r"[a-z0-9_]{1,40}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


class TasteError(ValidationError):
    """Report a taste field or file that needs correction."""


class TasteRefusal(YapError):
    """Refuse a decision that cannot apply to the current taste records."""

    def __init__(self, message: str, *, code: str, suggested_action: str) -> None:
        super().__init__(
            message, error_type="taste_refusal", code=code, suggested_action=suggested_action
        )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rule_id(body: Mapping[str, Any]) -> str:
    """Hash preference content, excluding evidence and audit metadata."""
    content = {key: body.get(key) for key in _IDENTITY_FIELDS}
    return "r-" + hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()[:12]


def _object(value: Any, keys: tuple[str, ...], location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TasteError(location, "expected object")
    for key in value:
        if key not in keys:
            raise TasteError(f"{location}.{key}", "unknown field")
    for key in keys:
        if key not in value:
            raise TasteError(f"{location}.{key}", "missing required field")
    return dict(value)


def _string(
    value: Any, location: str, *, minimum: int = 0, maximum: int, stripped: bool = False
) -> None:
    if not isinstance(value, str):
        raise TasteError(location, "expected string")
    if not minimum <= len(value) <= maximum:
        raise TasteError(location, "length is outside the allowed range")
    if stripped and (value != value.strip() or any(char in value for char in "\r\n")):
        raise TasteError(location, "must be stripped with no newlines")


def _choice(value: Any, choices: tuple[str, ...], location: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise TasteError(location, f"choose one of {', '.join(choices)}")


def _integer(value: Any, location: str, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TasteError(location, f"expected integer at least {minimum}")


def _pattern(value: Any, pattern: re.Pattern[str], location: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TasteError(location, f"must match {pattern.pattern}")


def _timestamp(value: Any, location: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    _pattern(value, _TIMESTAMP, location)
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise TasteError(location, "expected a valid UTC timestamp") from exc


def validate_style_field(field: Any, value: Any, location: str) -> str:
    """Validate an editable schema leaf and its value; return its section."""
    if not isinstance(field, str):
        raise TasteError(f"{location}.field", "expected a dotted style field")
    parts = field.split(".")
    if parts[0] == "internal" or field in {"schema_version", "name"}:
        raise TasteError(f"{location}.field", "engine constraints and identity cannot be changed")
    if len(parts) != 2 or parts[0] not in _SECTION_SCOPES:
        raise TasteError(f"{location}.field", "unknown style field")
    section, key = parts
    rule = loader._schema()["properties"].get(section, {}).get("properties", {}).get(key)
    if rule is None or rule["type"] == "object":
        raise TasteError(f"{location}.field", "unknown style field")
    try:
        loader._check(value, rule, f"{location}.value")
        if field == "graphics.card_kinds_on":
            loader.check_card_kinds_on(value, f"{location}.value")
    except loader.StyleError as exc:
        raise TasteError(exc.parameter, exc.detail) from exc
    return section


def _condition(value: Any, location: str) -> None:
    if value is None:
        return
    data = _object(value, ("text", "tags"), location)
    _string(data["text"], f"{location}.text", minimum=1, maximum=200, stripped=True)
    tags = data["tags"]
    if not isinstance(tags, list) or len(tags) > 8:
        raise TasteError(f"{location}.tags", "expected a list of at most 8 tags")
    for index, tag in enumerate(tags):
        item = f"{location}.tags[{index}]"
        _pattern(tag, _TAG, item)
        if tag in tags[:index]:
            raise TasteError(item, "tags must be unique")


def _machine(value: Any, scope: str, location: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TasteError(location, "expected object or null")
    kind = value.get("kind")
    _choice(kind, ("style_field", "filler_word", "placement"), f"{location}.kind")
    if kind == "style_field":
        data = _object(value, ("kind", "field", "value"), location)
        section = validate_style_field(data["field"], data["value"], location)
        expected_scope = _SECTION_SCOPES[section]
    elif kind == "filler_word":
        data = _object(value, ("kind", "word", "position"), location)
        word = data["word"]
        _pattern(word, re.compile(r"[a-z][a-z' ]{0,39}"), f"{location}.word")
        if word.endswith(" ") or "  " in word:
            raise TasteError(f"{location}.word", "no double or trailing spaces")
        _choice(data["position"], ("leading", "anywhere"), f"{location}.position")
        expected_scope = "cut"
    else:
        data = _object(value, ("kind", "layer", "rule"), location)
        _choice(data["layer"], ("captions", "title", "cards"), f"{location}.layer")
        _choice(data["rule"], ("clear_of_face",), f"{location}.rule")
        expected_scope = "captions" if data["layer"] == "captions" else "cards"
    if scope != expected_scope:
        raise TasteError(location.rsplit(".", 1)[0] + ".scope", f"must be {expected_scope!r}")


def _sources(value: Any, location: str) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 50:
        raise TasteError(location, "expected 1 to 50 sources")
    for index, source in enumerate(value):
        item = f"{location}[{index}]"
        if not isinstance(source, Mapping):
            raise TasteError(item, "expected object")
        kind = source.get("type")
        _choice(kind, ("comment", "measurement", "answer"), f"{item}.type")
        if kind == "answer":
            data = _object(source, ("type", "question", "answer"), item)
            _choice(data["question"], ("never",), f"{item}.question")
            _string(data["answer"], f"{item}.answer", minimum=1, maximum=200, stripped=True)
        elif kind == "comment":
            data = _object(
                source,
                (
                    "type",
                    "media_sha256",
                    "comment_id",
                    "label",
                    "start_frame",
                    "end_frame",
                    "quote",
                ),
                item,
            )
            _string(data["comment_id"], f"{item}.comment_id", minimum=1, maximum=80)
            if data["label"] is not None:
                _choice(data["label"], ("keep", "fix"), f"{item}.label")
            if data["start_frame"] is not None:
                _integer(data["start_frame"], f"{item}.start_frame")
            if data["end_frame"] is not None:
                _integer(data["end_frame"], f"{item}.end_frame", data["start_frame"] or 0)
            if data["quote"] is not None:
                _string(data["quote"], f"{item}.quote", maximum=500)
        else:
            data = _object(
                source, ("type", "media_sha256", "media_name", "measure", "frames"), item
            )
            _string(data["media_name"], f"{item}.media_name", minimum=1, maximum=255)
            if any(char in data["media_name"] for char in "/\\\x00"):
                raise TasteError(f"{item}.media_name", "expected a file name without separators")
            _pattern(data["measure"], re.compile(r"[a-z0-9_.]{1,80}"), f"{item}.measure")
            frames = data["frames"]
            if not isinstance(frames, list) or not 1 <= len(frames) <= 10000:
                raise TasteError(f"{item}.frames", "expected 1 to 10000 frames")
            for offset, frame in enumerate(frames):
                _integer(frame, f"{item}.frames[{offset}]")
                if offset and frame <= frames[offset - 1]:
                    raise TasteError(f"{item}.frames[{offset}]", "frames must be sorted and unique")
        if kind != "answer":
            _pattern(data["media_sha256"], _HASH, f"{item}.media_sha256")


def validate_body(value: Any, location: str = "proposal", *, defaults: bool = False) -> dict:
    """Validate a preference body, optionally filling nullable input fields."""
    if defaults and isinstance(value, Mapping):
        value = {"condition": None, "machine": None, "context": None, **value}
    data = _object(value, _BODY_FIELDS, location)
    _choice(
        data["scope"], ("cut", "captions", "cards", "broll", "pacing", "music"), f"{location}.scope"
    )
    _condition(data["condition"], f"{location}.condition")
    _string(data["sentence"], f"{location}.sentence", minimum=1, maximum=280, stripped=True)
    _machine(data["machine"], data["scope"], f"{location}.machine")
    _sources(data["sources"], f"{location}.sources")
    if data["context"] is not None:
        _string(data["context"], f"{location}.context", maximum=500)
    return copy.deepcopy(data)


def _record(value: Any, audit: tuple[str, ...], location: str) -> dict:
    data = _object(value, (*_BODY_FIELDS, "id", *audit), location)
    validate_body({key: data[key] for key in _BODY_FIELDS}, location)
    _pattern(data["id"], _ID, f"{location}.id")
    if data["id"] != rule_id(data):
        raise TasteError(f"{location}.id", "does not match preference content")
    _timestamp(data["proposed_at"], f"{location}.proposed_at")
    return data


def _records(value: Any, location: str) -> list:
    if not isinstance(value, list):
        raise TasteError(location, "expected array")
    return value


def _unique(data: dict, seen: set[str], location: str) -> None:
    if data["id"] in seen:
        raise TasteError(f"{location}.id", "ids must be unique")
    seen.add(data["id"])


def validate_proposals(value: Any) -> Proposals:
    """Validate a persisted proposals file, including decision invariants."""
    data = _object(value, ("schema", "proposals"), "proposals")
    _choice(data["schema"], (PROPOSALS_SCHEMA,), "proposals.schema")
    seen: set[str] = set()
    records = []
    for index, value in enumerate(_records(data["proposals"], "proposals.proposals")):
        location = f"proposals.proposals[{index}]"
        record = _record(value, ("status", "proposed_at", "decided_at"), location)
        _unique(record, seen, location)
        _choice(record["status"], ("proposed", "rejected"), f"{location}.status")
        _timestamp(record["decided_at"], f"{location}.decided_at", nullable=True)
        if (record["decided_at"] is None) != (record["status"] == "proposed"):
            raise TasteError(f"{location}.decided_at", "must be null exactly when proposed")
        records.append(copy.deepcopy(record))
    return Proposals(tuple(records))


def validate_taste(value: Any) -> Taste:
    """Validate persisted taste and all approval and ending invariants."""
    legacy = isinstance(value, Mapping) and "history" not in value
    if legacy:
        value = {**value, "history": []}
    data = _object(value, ("schema", "revision", "updated_at", "rules", "history"), "taste")
    _choice(data["schema"], (TASTE_SCHEMA,), "taste.schema")
    _integer(data["revision"], "taste.revision")
    _timestamp(data["updated_at"], "taste.updated_at", nullable=True)
    seen: set[str] = set()
    records = []
    for index, value in enumerate(_records(data["rules"], "taste.rules")):
        location = f"taste.rules[{index}]"
        record = _record(
            value,
            (
                "status",
                "proposed_at",
                "approved_at",
                "approved_revision",
                "ended_at",
                "ended_revision",
                "superseded_by",
            ),
            location,
        )
        _unique(record, seen, location)
        _choice(record["status"], ("approved", "revoked", "superseded"), f"{location}.status")
        _timestamp(record["approved_at"], f"{location}.approved_at")
        _integer(record["approved_revision"], f"{location}.approved_revision", 1)
        if record["approved_revision"] > data["revision"]:
            raise TasteError(f"{location}.approved_revision", "must not exceed taste revision")
        _timestamp(record["ended_at"], f"{location}.ended_at", nullable=True)
        if record["status"] == "approved":
            for key in ("ended_at", "ended_revision", "superseded_by"):
                if record[key] is not None:
                    raise TasteError(f"{location}.{key}", "must be null when approved")
        else:
            if record["ended_at"] is None:
                raise TasteError(f"{location}.ended_at", "must be set when ended")
            _integer(record["ended_revision"], f"{location}.ended_revision", 1)
            if not record["approved_revision"] < record["ended_revision"] <= data["revision"]:
                raise TasteError(
                    f"{location}.ended_revision", "must follow approval within revision"
                )
            if record["status"] == "revoked" and record["superseded_by"] is not None:
                raise TasteError(f"{location}.superseded_by", "must be null when revoked")
            if record["status"] == "superseded":
                _pattern(record["superseded_by"], _ID, f"{location}.superseded_by")
        records.append(copy.deepcopy(record))
    for index, record in enumerate(records):
        if record["status"] == "superseded" and (
            record["superseded_by"] not in seen or record["superseded_by"] == record["id"]
        ):
            raise TasteError(
                f"taste.rules[{index}].superseded_by", "must name another rule in taste"
            )
    if legacy:
        data["history"] = _legacy_history(records)
    history = _history(data, records)
    return Taste(data["revision"], data["updated_at"], tuple(records), history)


def _legacy_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild only the approvals and endings retained by the earlier file format."""
    history = []
    for record in records:
        history.append(
            {
                "revision": record["approved_revision"],
                "at": record["approved_at"],
                "action": "approved",
                "rule_id": record["id"],
                "by": None,
                "rule": {key: copy.deepcopy(record[key]) for key in _BODY_FIELDS},
                "migrated": True,
            }
        )
        if record["status"] != "approved":
            history.append(
                {
                    "revision": record["ended_revision"],
                    "at": record["ended_at"],
                    "action": record["status"],
                    "rule_id": record["id"],
                    "by": record["superseded_by"],
                    "rule": None,
                    "migrated": True,
                }
            )
    return sorted(
        history,
        key=lambda entry: (entry["revision"], entry["action"] != "approved", entry["rule_id"]),
    )


def _history(data: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Replay immutable decisions and require current records to match their last decision."""
    history = []
    states: dict[str, dict[str, Any]] = {}
    approvals: dict[str, dict[str, Any]] = {}
    revisions: set[int] = set()
    decisions: set[int] = set()
    previous = 0
    last_migrated = 0
    normal_seen = False
    for index, value in enumerate(_records(data["history"], "taste.history")):
        location = f"taste.history[{index}]"
        keys = ("revision", "at", "action", "rule_id", "by", "rule")
        migrated = isinstance(value, Mapping) and "migrated" in value
        entry = _object(value, (*keys, "migrated") if migrated else keys, location)
        if migrated and entry["migrated"] is not True:
            raise TasteError(f"{location}.migrated", "must be true; omit for normal decisions")
        if migrated and normal_seen:
            raise TasteError(f"{location}.migrated", "migrated entries must precede normal entries")
        revision, identity, action = entry["revision"], entry["rule_id"], entry["action"]
        _integer(revision, f"{location}.revision", 1)
        if not previous <= revision <= data["revision"]:
            raise TasteError(f"{location}.revision", "must be ordered within taste revision")
        previous = revision
        if migrated:
            last_migrated = revision
        else:
            normal_seen = True
            if revision <= last_migrated:
                raise TasteError(f"{location}.revision", "must follow the migrated prefix")
        revisions.add(revision)
        _timestamp(entry["at"], f"{location}.at")
        _pattern(identity, _ID, f"{location}.rule_id")
        _choice(action, ("approved", "reapproved", "revoked", "superseded"), f"{location}.action")
        if action != "superseded":
            if revision in decisions:
                raise TasteError(f"{location}.revision", "one approval or revocation per revision")
            decisions.add(revision)
        old = states.get(identity)
        active = old is not None and old["action"] in {"approved", "reapproved"}
        if action in {"approved", "reapproved"}:
            if (action == "approved" and old is not None) or (
                action == "reapproved" and (old is None or active)
            ):
                raise TasteError(f"{location}.action", "approval does not follow decision history")
            body = validate_body(entry["rule"], f"{location}.rule")
            if rule_id(body) != identity:
                raise TasteError(f"{location}.rule", "does not match rule_id")
            approvals[identity] = entry
        else:
            if not active or old["revision"] >= revision:
                raise TasteError(f"{location}.action", "ending must follow an active approval")
            if entry["rule"] is not None:
                raise TasteError(f"{location}.rule", "must be null for an ending")
        if action == "superseded":
            _pattern(entry["by"], _ID, f"{location}.by")
            if entry["by"] == identity or entry["by"] not in {record["id"] for record in records}:
                raise TasteError(f"{location}.by", "must name another rule in taste")
            replacement = states.get(entry["by"])
            # A migrated replacement may retain only a later reapproval.
            # Its earlier approval cannot be inferred from the current records.
            if not migrated and (
                entry["by"] == identity
                or replacement is None
                or (
                    replacement["action"] not in {"approved", "reapproved"}
                    or replacement["revision"] != revision
                )
            ):
                raise TasteError(f"{location}.by", "must name the approval in this revision")
        elif entry["by"] is not None:
            raise TasteError(f"{location}.by", "must be null unless superseded")
        states[identity] = entry
        history.append(copy.deepcopy(entry))
    normal_revisions = {revision for revision in revisions if revision > last_migrated}
    normal_decisions = {revision for revision in decisions if revision > last_migrated}
    if (
        len(normal_revisions) != data["revision"] - last_migrated
        or normal_decisions != normal_revisions
    ):
        raise TasteError("taste.history", "every taste revision must have a decision")
    if set(states) != {record["id"] for record in records}:
        raise TasteError("taste.history", "must describe every rule record")
    for index, record in enumerate(records):
        last, approval = states[record["id"]], approvals[record["id"]]
        status = "approved" if last["action"] in {"approved", "reapproved"} else last["action"]
        expected = {
            "status": status,
            "approved_at": approval["at"],
            "approved_revision": approval["revision"],
            "ended_at": None if status == "approved" else last["at"],
            "ended_revision": None if status == "approved" else last["revision"],
            "superseded_by": last["by"],
            **approval["rule"],
        }
        for key, value in expected.items():
            if record[key] != value:
                raise TasteError(f"taste.rules[{index}].{key}", "does not match decision history")
    if history and data["updated_at"] != history[-1]["at"]:
        raise TasteError("taste.updated_at", "must match the last decision")
    return tuple(history)


@dataclass(frozen=True)
class TasteSnapshot:
    """Active preference content and its evidence-independent hash."""

    revision: int
    sha256: str
    rules: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        """Return the pinnable snapshot and its revision and hash."""
        return {
            "schema": SNAPSHOT_SCHEMA,
            "rules": copy.deepcopy(list(self.rules)),
            "revision": self.revision,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class Taste:
    """Validated taste, including retained approval history."""

    revision: int = 0
    updated_at: str | None = None
    rules: tuple[dict[str, Any], ...] = ()
    history: tuple[dict[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return a separate persisted taste object."""
        return {
            "schema": TASTE_SCHEMA,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "rules": copy.deepcopy(list(self.rules)),
            "history": copy.deepcopy(list(self.history)),
        }

    def snapshot(self) -> TasteSnapshot:
        """Sort and hash only active preference content for version pinning."""
        rules = tuple(
            {key: copy.deepcopy(rule[key]) for key in ("id", *_IDENTITY_FIELDS)}
            for rule in sorted(self.rules, key=lambda rule: rule["id"])
            if rule["status"] == "approved"
        )
        payload = {"schema": SNAPSHOT_SCHEMA, "rules": list(rules)}
        digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        return TasteSnapshot(self.revision, digest, rules)


def snapshot_at(taste: Taste, revision: int) -> TasteSnapshot:
    """Replay approvals and endings through a revision using current snapshot hashing.

    Migrated history may omit earlier decisions. Callers must compare the recovered
    sha256 with the pinned sha256 and report a mismatch.
    """
    _integer(revision, "revision")
    if revision > taste.revision:
        raise TasteError("revision", "must not exceed the current taste revision")
    active = {}
    for entry in taste.history:
        if entry["revision"] > revision:
            break
        identity = entry["rule_id"]
        if entry["action"] in {"approved", "reapproved"}:
            active[identity] = {
                **copy.deepcopy(entry["rule"]), "id": identity, "status": "approved"
            }
        else:
            active.pop(identity, None)
    return Taste(revision=revision, rules=tuple(active.values())).snapshot()


@dataclass(frozen=True)
class Proposals:
    """Validated pending and rejected preference proposals."""

    proposals: tuple[dict[str, Any], ...] = ()

    @property
    def pending(self) -> tuple[dict[str, Any], ...]:
        """Return only proposals awaiting a decision."""
        return tuple(copy.deepcopy(item) for item in self.proposals if item["status"] == "proposed")

    def to_json(self) -> dict[str, Any]:
        """Return a separate persisted proposals object."""
        return {"schema": PROPOSALS_SCHEMA, "proposals": copy.deepcopy(list(self.proposals))}


@dataclass(frozen=True)
class AddResult:
    """One proposal import outcome."""

    id: str
    status: str


@dataclass(frozen=True)
class ApproveResult:
    """An approved rule, the rules it superseded, and the resulting revision."""

    rule: dict[str, Any]
    superseded_ids: tuple[str, ...]
    revision: int
