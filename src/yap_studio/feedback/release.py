"""Restore saved project inputs and approve immutable version downloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from yap_studio import paths
from yap_studio.errors import InputError
from yap_studio.feedback.versions import (
    VersionError,
    VersionManifest,
    _digest,
    _fields,
    _read_json,
    _snapshot_json,
    current_version,
    load_version,
    version_report,
)
from yap_studio.project.lock import project_lock, write_json_cas
from yap_studio.project.model import parse_document
from yap_studio.receipt import sha256_file

APPROVAL_SCHEMA = "yap-studio/version-approval@1"
POINTER_SCHEMA = "yap-studio/version-pointer@1"
ROLLBACK_SCHEMA = "yap-studio/rollback-journal@1"


def _timestamp(now: Callable[[], datetime] | None) -> str:
    value = now() if now is not None else datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _verify_media(manifest: VersionManifest) -> None:
    if _hash(manifest.media_path) != manifest.media["sha256"]:
        raise InputError(str(manifest.media_path), "media changed after it was saved")


def _clear_rollback(root: Path) -> None:
    (root / "versions" / ".rollback.json").unlink(missing_ok=True)
    shutil.rmtree(root / "versions" / ".rollback", ignore_errors=True)


def _json_payload(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_rollback_copy(path: Path, payload: bytes) -> None:
    paths._guard_write(path)
    path.write_bytes(payload)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _restore_rollback(root: Path, journal: dict) -> None:
    """Undo our replacements while preserving any later edits by another writer."""
    entries = dict(journal["files"])
    if journal.get("pointer_after_sha256") is not None:
        entries["versions/current.json"] = {
            "before_sha256": journal["pointer_before_sha256"],
            "after_sha256": journal["pointer_after_sha256"],
            "backup": journal["pointer_backup"],
        }
    for name, entry in entries.items():
        target = root / name
        if _hash(target) == entry["after_sha256"] and (
            entry["before_sha256"] != entry["after_sha256"]
        ):
            paths._guard_write(target)
            if entry["before_sha256"] is None:
                target.unlink()
            else:
                backup = root / entry["backup"]
                if _hash(backup) != entry["before_sha256"]:
                    raise VersionError(f"Rollback backup changed: {name}.")
                os.replace(backup, target)


def _complete_pending_rollback(root: Path) -> dict | None:
    """Finish a journal while the caller holds the project lock."""
    journal_path = root / "versions" / ".rollback.json"
    if not journal_path.exists():
        # A crash before publishing a journal leaves only disposable staged copies.
        _clear_rollback(root)
        return None
    journal = _read_json(journal_path)
    try:
        if (not isinstance(journal, dict) or journal.get("schema") != ROLLBACK_SCHEMA
                or not isinstance(journal.get("files"), dict) or not journal["files"]):
            raise ValueError("invalid rollback journal")
        load_version(root, journal["version"])
        for name, entry in journal["files"].items():
            if name not in {"edit.json", "removals.json"}:
                raise ValueError("invalid rollback target")
            if (entry["before_sha256"] is not None and not _digest(entry["before_sha256"])
                    or not _digest(entry["after_sha256"])):
                raise ValueError("invalid rollback hashes")
            for key in ("staged", "backup"):
                expected = f"versions/.rollback/{name}.{key}"
                if entry[key] != expected or (root / expected).is_symlink():
                    raise ValueError("invalid rollback copy path")
    except (KeyError, TypeError, ValueError) as exc:
        raise VersionError("Cannot read a valid rollback journal.") from exc
    try:
        conflicts = [name for name, entry in journal["files"].items()
                     if _hash(root / name) not in (entry["before_sha256"], entry["after_sha256"])]
        if conflicts:
            raise VersionError(f"Rollback conflicts: {', '.join(conflicts)}.")
        for name, entry in journal["files"].items():
            if _hash(root / name) == entry["after_sha256"]:
                continue
            staged = root / entry["staged"]
            if _hash(staged) != entry["after_sha256"]:
                raise VersionError(f"Rollback staged copy changed: {name}.")
            if _hash(root / name) != entry["before_sha256"]:
                raise VersionError(f"Rollback conflicts: {name}.")
            paths._guard_write(root / name)
            os.replace(staged, root / name)
        if journal["pointer"] is not None:
            pointer = root / "versions" / "current.json"
            # A pointer already published before a crash needs no second write.
            if not pointer.exists() or _read_json(pointer) != journal["pointer"]:
                write_json_cas(pointer, journal["pointer"],
                               base_sha256=journal["pointer_before_sha256"])
        _clear_rollback(root)
        return journal["result"]
    except Exception:
        _restore_rollback(root, journal)
        _clear_rollback(root)
        raise


def complete_pending_rollback(project_dir: Path) -> dict | None:
    """Complete an interrupted rollback forward, or undo it on a content conflict."""
    root = Path(project_dir).resolve()
    with project_lock(root):
        return _complete_pending_rollback(root)


def rollback(
    project_dir: Path, version_id: str, *, now: Callable[[], datetime] | None = None,
) -> dict:
    """Restore verified edit and removals snapshots, then publish the current pointer."""
    root = Path(project_dir).resolve()
    with project_lock(root):
        _complete_pending_rollback(root)
        manifest = load_version(root, version_id)
        targets = {"edit.json": root / "edit.json", "removals.json": root / "removals.json"}
        bases = {name: _hash(path) for name, path in targets.items()}
        _verify_media(manifest)
        documents = {}
        for key, entry in manifest.snapshots.items():
            if entry is None or key in {"timeline", "taste"}:
                continue
            if key == "style" and entry["file"] is None:
                continue
            documents[key] = _snapshot_json(manifest, key)
        version_report(manifest)
        restore = {"edit.json": documents["edit"]}
        if "removals" in documents or bases["removals.json"] is not None:
            restore["removals.json"] = documents.get(
                "removals", {"schema": "yap-studio/removals@1", "removals": []},
            )
        for name, data in restore.items():
            parse_document(name.removesuffix(".json"), cast(dict, data))
        timeline = manifest.snapshots["timeline"]
        lanes = timeline["lanes"] if timeline else {}
        not_restored = []
        current_lanes = {path.stem for path in (root / "timeline").glob("*.json")}
        for lane in sorted(current_lanes | lanes.keys()):
            entry = lanes.get(lane)
            if entry is not None:
                snapshot = manifest.path / entry["file"]
                if _hash(snapshot) != entry["sha256"]:
                    raise InputError(str(snapshot), "snapshot changed after it was saved")
            if _hash(root / "timeline" / f"{lane}.json") != (
                entry["sha256"] if entry else None
            ):
                not_restored.append(f"timeline/{lane}.json")
        result = {"version": manifest.id, "restored": list(restore),
                  "not_restored": not_restored, "cut_stale": True}
        folder = root / "versions" / ".rollback"
        journal = {"schema": ROLLBACK_SCHEMA, "version": manifest.id, "files": {},
                   "pointer": None, "result": result}
        pointer = root / "versions" / "current.json"
        journal["pointer_before_sha256"] = _hash(pointer)
        journal["pointer_backup"] = "versions/.rollback/current.json.backup"
        try:
            paths._guard_write(folder)
            folder.mkdir()
            if journal["pointer_before_sha256"] is not None:
                _write_rollback_copy(root / journal["pointer_backup"], pointer.read_bytes())
                if _hash(root / journal["pointer_backup"]) != journal["pointer_before_sha256"]:
                    raise VersionError("Rollback targets changed: versions/current.json.")
            changed = False
            for name, data in restore.items():
                target = targets[name]
                paths._guard_write(target)
                try:
                    before = target.read_bytes() if bases[name] is not None else None
                except FileNotFoundError as exc:
                    raise VersionError(f"Rollback targets changed: {name}.") from exc
                backup = folder / f"{name}.backup"
                if before is not None:
                    _write_rollback_copy(backup, before)
                    if _hash(backup) != bases[name]:
                        raise VersionError(f"Rollback targets changed: {name}.")
                try:
                    same = before is not None and json.loads(before) == data
                except (ValueError, UnicodeError):
                    same = False
                staged = folder / f"{name}.staged"
                payload = before if same else _json_payload(data)
                _write_rollback_copy(staged, payload)
                after = _hash(staged)
                changed |= after != bases[name]
                journal["files"][name] = {
                    "before_sha256": bases[name], "after_sha256": after,
                    "staged": staged.relative_to(root).as_posix(),
                    "backup": backup.relative_to(root).as_posix(),
                }
            conflicts = [name for name in restore if _hash(targets[name]) != bases[name]]
            if conflicts:
                raise VersionError(f"Rollback targets changed: {', '.join(conflicts)}.")
            if changed or current_version(root) != manifest:
                journal["pointer"] = {"schema": POINTER_SCHEMA, "current": manifest.id,
                                      "set_at": _timestamp(now), "reason": "rollback"}
            journal["pointer_after_sha256"] = hashlib.sha256(
                _json_payload(journal["pointer"]),
            ).hexdigest() if journal["pointer"] is not None else None
            write_json_cas(root / "versions" / ".rollback.json", journal, base_sha256=None)
        except Exception:
            _clear_rollback(root)
            raise
        return _complete_pending_rollback(root)


def approve(
    project_dir: Path, version_id: str, *, now: Callable[[], datetime] | None = None,
) -> dict:
    """Approve a hash-checked version with a report that has not failed."""
    root = Path(project_dir).resolve()
    with project_lock(root):
        _complete_pending_rollback(root)
        manifest = load_version(root, version_id)
        _verify_media(manifest)
        report = version_report(manifest)
        action = "Run the checks before approving this version."
        if report is None:
            raise VersionError("This version has no check report; run the checks before approving.",
                               suggested_action=action)
        if any(status is None for status in report["statuses"].values()):
            raise VersionError("This version's check report has no valid status; "
                               "run the checks before approving.", suggested_action=action)
        if report["status"] == "fail":
            source = next(report[name] for name, status in report["statuses"].items()
                          if status == "fail")
            findings = source.get("findings", [])
            details = "; ".join(str(item["message"]) for item in findings
                                if isinstance(item, dict) and item.get("message")) if (
                isinstance(findings, list)
            ) else ""
            raise VersionError(f"This version failed its checks: {details or manifest.id}.",
                               suggested_action="Fix the failed checks and save a new version.")
        approval = {"schema": APPROVAL_SCHEMA, "version": manifest.id,
                    "media_sha256": manifest.media["sha256"], "approved_at": _timestamp(now),
                    "report_status": report["status"]}
        path = root / "versions" / "approval.json"
        write_json_cas(path, approval, base_sha256=_hash(path))
        return approval


def clear_approval(project_dir: Path) -> None:
    """Clear the project's approval under its writer lock."""
    root = Path(project_dir).resolve()
    with project_lock(root):
        (root / "versions" / "approval.json").unlink(missing_ok=True)


def approval_record(project_dir: Path, *, verify_media: bool = True) -> dict | None:
    """Match approval to the manifest and optionally verify the saved media file."""
    path = Path(project_dir) / "versions" / "approval.json"
    if not path.exists():
        return None
    try:
        data = _fields(_read_json(path), {
            "schema", "version", "media_sha256", "approved_at", "report_status",
        })
        if data["schema"] != APPROVAL_SCHEMA or not _digest(data["media_sha256"]):
            raise ValueError("invalid version approval")
        datetime.strptime(data["approved_at"], "%Y-%m-%dT%H:%M:%SZ")
        if data["report_status"] is not None and not isinstance(data["report_status"], str):
            raise ValueError("report_status must be a string or null")
    except (TypeError, ValueError) as exc:
        raise InputError(str(path), str(exc)) from exc
    manifest = load_version(project_dir, data["version"])
    if data["media_sha256"] != manifest.media["sha256"] or (
        verify_media and _hash(manifest.media_path) != manifest.media["sha256"]
    ):
        return None
    return data


def approved_version(project_dir: Path, *, verify_media: bool = True) -> VersionManifest | None:
    """Return the approved version, optionally verifying its saved media file."""
    record = approval_record(project_dir, verify_media=verify_media)
    return load_version(project_dir, record["version"]) if record else None
