"""Apply approved taste at the shared render boundary and record its identities.

Where the project render pipeline resolves its render style, call
render_style(style, explicit=<project instructions>). After publishing the final
render, call write_taste_record(final_media, application) so versions save pins
the revision. Records contain ids, revisions, hashes, fields, and decision codes.
Rule values and sentences stay in the private config folder.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from yap_studio import paths
from yap_studio.errors import InputError
from yap_studio.receipt import sha256_file
from yap_studio.style.loader import ResolvedStyle, resolve
from yap_studio.taste.model import TasteSnapshot
from yap_studio.taste.resolve import TasteApplication, apply_taste

TASTE_RECORD_SCHEMA = "yap-studio/taste-record@1"
_KEYS = {
    "schema", "media_sha256", "revision", "sha256", "style_sha256", "applied", "skipped",
    "explicit_fields",
}
_HASH = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"r-[0-9a-f]{12}")
_FIELD = re.compile(r"(?:captions|hook|chapters|graphics|broll|music)\.[a-z][a-z0-9_]*")
_REASON = re.compile(r"[a-z][a-z0-9_]*")


def render_style(
    spec: str | os.PathLike[str] | ResolvedStyle | None = None,
    *,
    explicit: Mapping[str, Any] | None = None,
    tags: Collection[str] = (),
    snapshot: TasteSnapshot | None = None,
) -> TasteApplication:
    """Apply base style, then approved taste rules, then project instructions.

    Schema limits always hold, including engine limits and style identity.
    """
    style = spec if isinstance(spec, ResolvedStyle) else resolve(spec)
    return apply_taste(style, snapshot, tags=tags, explicit=explicit)


def _matches(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _record(data: Any) -> dict[str, Any]:
    """Validate the content-free record shape before writing or trusting it."""
    if not isinstance(data, dict) or set(data) not in (_KEYS, _KEYS | {"enforced"}):
        raise ValueError("taste record must have exactly the required fields")
    if data["schema"] != TASTE_RECORD_SCHEMA:
        raise ValueError("unknown taste record schema")
    if type(data["revision"]) is not int or data["revision"] < 0:
        raise ValueError("taste revision must be an integer at least 0")
    for key in ("media_sha256", "sha256", "style_sha256"):
        if not _matches(data[key], _HASH):
            raise ValueError(f"taste record {key} must be a SHA256 digest")
    for key, leaf, pattern in (("applied", "field", _FIELD), ("skipped", "reason", _REASON)):
        if not isinstance(data[key], list):
            raise ValueError(f"taste record {key} must be an array")
        for item in data[key]:
            if not isinstance(item, dict) or set(item) != {"id", leaf}:
                raise ValueError(f"taste record {key} entries must have exactly id and {leaf}")
            if not _matches(item["id"], _ID) or not _matches(item[leaf], pattern):
                raise ValueError(f"invalid taste record {key} entry")
    fields = data["explicit_fields"]
    if not isinstance(fields, list) or any(not _matches(field, _FIELD) for field in fields):
        raise ValueError("taste record explicit_fields must be an array of style fields")
    enforced = data.get("enforced", [])
    if not isinstance(enforced, list) or any(
        not isinstance(item, dict) or set(item) != {"id", "check"}
        or not _matches(item["id"], _ID) or item["check"] != "faces.clearance"
        for item in enforced
    ):
        raise ValueError("taste record enforced must name rule ids and faces.clearance")
    return data


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate taste record field")
        result[key] = value
    return result


def write_taste_record(
    media: Path, application: TasteApplication, *, out: Path | None = None,
) -> Path:
    """Hash published media, check its receipt, and atomically record its taste.

    Publishers with a receipt must hold their publication lock through this call.
    An explicit output lets project stages journal the record before publication.
    """
    target = out if out is not None else media.with_suffix(".taste.json")
    source = paths.validate_input(media)
    paths.validate_output(target)
    try:
        digest = sha256_file(source)
        record = _record({
            "schema": TASTE_RECORD_SCHEMA, "media_sha256": digest, **application.pin()
        })
        receipt_path = media.with_suffix(".receipt.json")
        if receipt_path.exists() or receipt_path.is_symlink():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"),
                                     object_pairs_hook=_json_object)
                style = receipt.get("style") if isinstance(receipt, dict) else None
                outputs = receipt.get("outputs") if isinstance(receipt, dict) else None
                matches = (
                    isinstance(style, dict) and style.get("sha256") == record["style_sha256"]
                    and isinstance(outputs, list) and any(
                        isinstance(output, dict) and output.get("sha256") == digest
                        for output in outputs
                    )
                )
            except (OSError, UnicodeError, ValueError, RecursionError):
                matches = False
            if not matches:
                raise InputError(
                    str(media), "render changed while its taste was being recorded; render again",
                )
        return paths.atomic_write_text(
            target, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    except (OSError, ValueError) as exc:
        raise InputError(str(target), f"cannot write taste record: {exc}") from exc


def read_taste_record(media: Path) -> dict[str, Any] | None:
    """Read an optional record, refusing corrupt data or a different media hash."""
    target = media.with_suffix(".taste.json")
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        if target.is_symlink():
            raise InputError(str(target), "taste record points to a missing file") from exc
        return None
    except (OSError, UnicodeError) as exc:
        raise InputError(str(target), "cannot read taste record") from exc
    try:
        record = _record(json.loads(text, object_pairs_hook=_json_object))
    except (ValueError, RecursionError) as exc:
        raise InputError(str(target), f"corrupt taste record: {exc}") from exc
    source = paths.validate_input(media)
    try:
        digest = sha256_file(source)
    except OSError as exc:
        raise InputError(str(media), "cannot hash media for taste record") from exc
    if digest != record["media_sha256"]:
        raise InputError(str(target), "taste record media hash does not match the media")
    return record
