"""Validated agent descriptions and append-only footage judgments."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from yap_studio.errors import InputError, ValidationError
from yap_studio.media import catalog

DESCRIPTION_SCHEMA = "yap-studio/media-description@1"
_CHANGES = {"description", "tags", "media_type", "provenance", "ranges", "revoke"}


def read_description_file(path: str | Path) -> object:
    """Read an agent's JSON file, reporting missing or malformed input clearly."""
    target = Path(path).expanduser()
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(str(target), "description file not found") from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(str(target), "description file is not valid JSON") from exc
    except OSError as exc:
        raise InputError(str(target), f"cannot read description file: {exc}") from exc


def _object(value: object, field: str, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(field, "must be an object")
    for key in value:
        if key not in allowed:
            raise ValidationError(f"{field}.{key}", "unknown field")
    for key in sorted(required - value.keys()):
        raise ValidationError(f"{field}.{key}", "required field is missing")
    return value


def _text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValidationError(field, "must be a string")
    value = value.strip()
    if not 1 <= len(value) <= limit:
        raise ValidationError(field, f"must contain 1 to {limit} characters after trimming")
    return value


def _list(value: object, field: str) -> list:
    if not isinstance(value, list):
        raise ValidationError(field, "must be a list")
    return value


def _finite_number(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _provenance(value: object) -> dict:
    fields = {"kind", "source", "license", "url"}
    result = dict(_object(value, "provenance", fields, fields))
    if result["kind"] not in ("own", "licensed", "generated"):
        raise ValidationError("provenance.kind", "choose own, licensed, or generated")
    for key in ("source", "license", "url"):
        entry = result[key]
        limit = 500 if key == "url" else 200
        if entry is not None and (
            not isinstance(entry, str) or len(entry) > limit
        ):
            raise ValidationError(f"provenance.{key}", f"must be null or a string up to {limit}")
    url = result["url"]
    if url is not None:
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                and url.isprintable() and not any(char.isspace() for char in url)
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValidationError("provenance.url", "must be null or an http or https URL")
    return result


def _validate(data: object, item: dict) -> dict:
    source = _object(data, "data", _CHANGES | {"schema", "by"}, {"schema"})
    if source["schema"] != DESCRIPTION_SCHEMA:
        raise ValidationError("schema", f"must be {DESCRIPTION_SCHEMA!r}")
    if not _CHANGES.intersection(source):
        raise ValidationError("data", "provide at least one field to change")
    by = source.get("by", "agent")
    if not isinstance(by, str) or not 1 <= len(by) <= 64 or not by.isprintable():
        raise ValidationError("by", "must contain 1 to 64 printable characters")
    result = {"by": by}
    if "description" in source:
        text = _text(source["description"], "description", 2000)
        if any(not char.isprintable() and char != "\n" for char in text):
            raise ValidationError(
                "description", "must not contain control characters except newline"
            )
        result["description"] = text
    if "tags" in source:
        tags = _list(source["tags"], "tags")
        if len(tags) > 50:
            raise ValidationError("tags", "must contain at most 50 strings")
        normalized = [_text(tag, f"tags[{i}]", 40).lower() for i, tag in enumerate(tags)]
        result["tags"] = list(dict.fromkeys(normalized))
    if "media_type" in source:
        choices = ("clip", "screen_recording") if item["kind"] == "video" else (
            "photo", "screenshot"
        )
        if source["media_type"] not in choices:
            raise ValidationError("media_type", f"this item accepts {', '.join(choices)}")
        result["media_type"] = source["media_type"]
    if "provenance" in source:
        result["provenance"] = _provenance(source["provenance"])
    result["ranges"] = []
    for i, value in enumerate(_list(source.get("ranges", []), "ranges")):
        field = f"ranges[{i}]"
        entry = _object(
            value, field, {"verdict", "reason", "start_s", "end_s"}, {"verdict", "reason"}
        )
        if entry["verdict"] not in ("approved", "rejected"):
            raise ValidationError(f"{field}.verdict", "choose approved or rejected")
        reason = _text(entry["reason"], f"{field}.reason", 500)
        start, end = entry.get("start_s"), entry.get("end_s")
        if ("start_s" in entry) != ("end_s" in entry):
            raise ValidationError(field, "start_s and end_s must both be present or both absent")
        if start is not None or end is not None:
            for key, number in (("start_s", start), ("end_s", end)):
                if not _finite_number(number):
                    raise ValidationError(f"{field}.{key}", "must be a finite number")
            if item["kind"] != "video":
                raise ValidationError(field, "stills accept only whole-item ranges")
            if not 0 <= start < end <= item["duration_s"] + 0.001:
                raise ValidationError(field, "require 0 <= start_s < end_s <= duration_s + 0.001")
        result["ranges"].append({
            "verdict": entry["verdict"], "start_s": start, "end_s": end, "reason": reason,
        })
    decisions = {entry["id"]: entry for entry in item["decisions"]}
    result["revoke"] = []
    for i, decision_id in enumerate(_list(source.get("revoke", []), "revoke")):
        field = f"revoke[{i}]"
        if not isinstance(decision_id, str) or decision_id not in decisions:
            raise ValidationError(field, "unknown decision id for this item")
        if decisions[decision_id]["revoked_at"] is not None or decision_id in result["revoke"]:
            raise ValidationError(field, "decision is already revoked")
        result["revoke"].append(decision_id)
    return result


def describe_item(
    item_id: str, data: object, *, now: Callable[[], datetime] | None = None,
) -> dict:
    """Store a description revision and append or revoke validated decisions atomically."""
    # Reject bad input before creating storage, then revalidate under the catalog lock.
    _validate(data, catalog.get_item(item_id))

    def mutate(document: dict) -> dict:
        item = next((entry for entry in document["items"] if entry["id"] == item_id), None)
        if item is None:
            raise catalog.UnknownMediaError(item_id)
        # Validate everything against the current item before changing any field.
        values = _validate(data, item)
        instant = (now or (lambda: datetime.now(UTC)))()
        at = instant.replace(tzinfo=instant.tzinfo or UTC).astimezone(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        by = values["by"]
        if "description" in values or "tags" in values:
            previous = item["description"] or {"text": None, "tags": [], "revision": 0}
            item["description"] = {
                "text": values.get("description", previous["text"]),
                "tags": values.get("tags", previous["tags"]),
                "by": by, "at": at, "revision": previous["revision"] + 1,
            }
        for key in ("media_type", "provenance"):
            if key in values:
                item[key] = values[key]
        if "media_type" in values:
            item["media_type_by"] = "agent"
        added = []
        for entry in values["ranges"]:
            decision_id = f"d{item['next_decision']}"
            item["decisions"].append({
                "id": decision_id, **entry, "by": by, "at": at,
                "revoked_at": None, "revoked_by": None,
            })
            item["next_decision"] += 1
            added.append(decision_id)
        for decision in item["decisions"]:
            if decision["id"] in values["revoke"]:
                decision.update(revoked_at=at, revoked_by=by)
        return {
            "id": item_id, "description": item["description"], "decisions_added": added,
            "revoked": values["revoke"], "item": item,
        }

    return catalog.update_catalog(mutate)
