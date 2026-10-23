"""Private creator taste persistence and locked approval decisions."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yap_studio import paths
from yap_studio.style.library import config_root
from yap_studio.taste.model import (
    PROPOSALS_SCHEMA,
    AddResult,
    ApproveResult,
    Proposals,
    Taste,
    TasteError,
    TasteRefusal,
    rule_id,
    validate_body,
    validate_proposals,
    validate_taste,
)


def taste_path() -> Path:
    """Return the creator's taste file without creating it."""
    return config_root() / "taste.json"


def proposals_path() -> Path:
    """Return the creator's proposal file without creating it."""
    return config_root() / "taste-proposals.json"


def _now() -> str:
    """Return the current UTC timestamp, rounded to seconds."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TasteError(str(path), f"cannot read JSON: {exc}") from exc


def load_taste() -> Taste:
    """Read and validate taste, or return empty taste when absent."""
    target = taste_path()
    return validate_taste(_read(target)) if target.exists() else Taste()


def _read_proposals() -> Proposals:
    target = proposals_path()
    return validate_proposals(_read(target)) if target.exists() else Proposals()


def _reconcile(taste: Taste, proposals: Proposals) -> Proposals:
    approved = {rule["id"] for rule in taste.rules if rule["status"] == "approved"}
    return Proposals(
        tuple(
            proposal
            for proposal in proposals.proposals
            if not (proposal["status"] == "proposed" and proposal["id"] in approved)
        )
    )


def load_proposals() -> Proposals:
    """Read proposals, hiding pending records already committed as approved."""
    return _reconcile(load_taste(), _read_proposals())


def _assert_private() -> None:
    root = config_root()
    home = Path.home().resolve()
    # Check the selected folder and its actual destination to prevent symlink bypasses.
    for candidate in (root, root.resolve()):
        under_home = candidate.is_relative_to(home)
        for ancestor in (candidate, *candidate.parents):
            if under_home and ancestor == home and ancestor != candidate:
                break
            git_entry = ancestor / ".git"
            if (
                git_entry.exists()
                or git_entry.is_symlink()
                or (ancestor / "project.json").is_file()
            ):
                error = TasteError("taste", "config folder is inside a repository or project")
                error.suggested_action = (
                    "Set YAP_STUDIO_CONFIG to a private folder outside repositories and projects."
                )
                raise error
            if under_home and ancestor == home:
                break


@contextmanager
def _lock(*, timeout_s: float = 30) -> Iterator[None]:
    """Hold a small OS lock, released automatically if the writer exits."""
    _assert_private()
    root = paths.validate_output_dir(config_root())
    target = root / ".taste.lock"
    paths._guard_write(target)
    descriptor = os.open(target, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b" ")
        else:
            import fcntl

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                pass
            except OSError:
                if os.name != "nt":
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TasteRefusal(
                    "Another writer holds the taste lock.",
                    code="taste_locked",
                    suggested_action="Wait for the writer and retry.",
                )
            time.sleep(min(0.05, remaining))
        yield
    finally:
        try:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def _mutation() -> Iterator[tuple[Taste, Proposals]]:
    """Re-read both files inside one guarded write operation and lock."""
    try:
        with paths.operation(), _lock():
            yield load_taste(), _read_proposals()
    except OSError as exc:
        raise TasteError("taste", f"cannot save taste: {exc}") from exc


def _write(target: Path, data: dict[str, Any]) -> None:
    _assert_private()
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    paths.atomic_write_text(target, text)


def _save(taste: Taste, proposals: Proposals, before: tuple[Taste, Proposals]) -> None:
    # Validate both results before the first write. Approval must commit taste first.
    taste_data, proposal_data = taste.to_json(), proposals.to_json()
    validate_taste(taste_data)
    validate_proposals(proposal_data)
    target = taste_path()
    needs_upgrade = target.exists() and "history" not in _read(target)
    if taste_data != before[0].to_json() or needs_upgrade:
        _write(taste_path(), taste_data)
    if proposal_data != before[1].to_json():
        _write(proposals_path(), proposal_data)


def _input(source: Path | Mapping[str, Any]) -> list[dict[str, Any]]:
    data = _read(source) if isinstance(source, Path) else source
    if not isinstance(data, Mapping):
        raise TasteError("proposal", "expected object")
    if "schema" in data:
        for key in data:
            if key not in {"schema", "proposals"}:
                raise TasteError(f"proposals.{key}", "unknown field")
        if data["schema"] != PROPOSALS_SCHEMA:
            raise TasteError("proposals.schema", f"must be {PROPOSALS_SCHEMA!r}")
        if not isinstance(data.get("proposals"), list):
            raise TasteError("proposals.proposals", "expected array")
        inputs = [(body, f"proposals[{index}]") for index, body in enumerate(data["proposals"])]
    else:
        inputs = [(data, "proposal")]
    bodies = []
    for value, location in inputs:
        if not isinstance(value, Mapping):
            raise TasteError(location, "expected object")
        body = validate_body(
            {key: item for key, item in value.items() if key not in {"id", "status"}},
            location,
            defaults=True,
        )
        identity = rule_id(body)
        if "id" in value and value["id"] != identity:
            raise TasteError(f"{location}.id", "does not match preference content")
        if "status" in value and value["status"] != "proposed":
            raise TasteError(f"{location}.status", "must be 'proposed'")
        bodies.append({**body, "id": identity})
    return bodies


def add_proposals(source: Path | Mapping[str, Any]) -> list[AddResult]:
    """Validate all input bodies, then add or merge evidence under the lock."""
    bodies = _input(source)
    with _mutation() as before:
        taste, raw = before
        proposals = _reconcile(taste, raw)
        results = []
        now = _now()
        for body in bodies:
            proposals, result = _import_proposal(taste, proposals, body, now)
            results.append(result)
        _save(taste, proposals, before)
        return results


def _import_proposal(
    taste: Taste, proposals: Proposals, body: Mapping[str, Any], now: str, *, reopen: bool = False
) -> tuple[Proposals, AddResult]:
    """Return imported records without mutating either input or touching files."""
    identity = rule_id(body)
    if any(rule["id"] == identity and rule["status"] == "approved" for rule in taste.rules):
        return proposals, AddResult(identity, "already_approved")
    existing = next((item for item in proposals.proposals if item["id"] == identity), None)
    if existing is not None and existing["status"] == "rejected" and not reopen:
        return proposals, AddResult(identity, "previously_rejected")
    if existing is None:
        record = {**copy.deepcopy(body), "id": identity, "status": "proposed",
                  "proposed_at": now, "decided_at": None}
        return Proposals((*proposals.proposals, record)), AddResult(identity, "added")
    sources = []
    for source in (*existing["sources"], *body["sources"]):
        if source not in sources and len(sources) < 50:
            sources.append(copy.deepcopy(source))
    record = {**existing, "sources": sources, "status": "proposed", "decided_at": None,
              "proposed_at": min(existing["proposed_at"], now)}
    records = tuple(record if item["id"] == identity else item for item in proposals.proposals)
    return Proposals(records), AddResult(identity, "merged")


def _pending(identity: str, taste: Taste, proposals: Proposals) -> dict[str, Any]:
    pending = next((item for item in proposals.pending if item["id"] == identity), None)
    if pending is not None:
        return pending
    if any(rule["id"] == identity and rule["status"] == "approved" for rule in taste.rules):
        action = f"Use yap-studio taste revoke {identity} to end the approved rule."
    else:
        action = "Use yap-studio taste show to choose a pending proposal."
    raise TasteRefusal(
        f"Taste rule {identity!r} is not pending.",
        code="taste_not_pending",
        suggested_action=action,
    )


def _target(rule: Mapping[str, Any]) -> tuple | None:
    machine = rule["machine"]
    if machine is None:
        return None
    kind = machine["kind"]
    if kind == "style_field":
        return kind, machine["field"]
    if kind == "filler_word":
        return kind, machine["word"]
    return kind, machine["layer"], machine["rule"]


def approve(rule_id: str) -> ApproveResult:
    """Approve one proposal, superseding equal-condition rules on its target."""
    with _mutation() as before:
        taste, raw = before
        taste, proposals, result = _approve(taste, _reconcile(taste, raw), rule_id, _now())
        _save(taste, proposals, before)
        return result


def _approve(
    taste: Taste, proposals: Proposals, identity: str, now: str
) -> tuple[Taste, Proposals, ApproveResult]:
    pending = _pending(identity, taste, proposals)
    revision = taste.revision + 1
    rule = {key: copy.deepcopy(value) for key, value in pending.items() if key != "decided_at"}
    rule.update(status="approved", approved_at=now, approved_revision=revision,
                ended_at=None, ended_revision=None, superseded_by=None)
    records, superseded = [], []
    target, replaced = _target(rule), False
    for old in taste.rules:
        if old["id"] == identity:
            records.append(rule)
            replaced = True
        elif (old["status"] == "approved" and target is not None
              and _target(old) == target and old["condition"] == rule["condition"]):
            records.append({**old, "status": "superseded", "ended_at": now,
                            "ended_revision": revision, "superseded_by": identity})
            superseded.append(old["id"])
        else:
            records.append(old)
    if not replaced:
        records.append(rule)
    remaining = Proposals(tuple(item for item in proposals.proposals if item["id"] != identity))
    history = (
        *taste.history,
        _decision(revision, now, "reapproved" if replaced else "approved", rule),
        *(_decision(revision, now, "superseded", {"id": old_id}, by=identity)
          for old_id in superseded),
    )
    result = ApproveResult(copy.deepcopy(rule), tuple(sorted(superseded)), revision)
    return Taste(revision, now, tuple(records), history), remaining, result


def reject(rule_id: str) -> dict[str, Any]:
    """Reject a pending proposal, keeping its evidence and decision time."""
    with _mutation() as before:
        taste, raw = before
        proposals, result = _reject(taste, _reconcile(taste, raw), rule_id, _now())
        _save(taste, proposals, before)
        return result


def _reject(
    taste: Taste, proposals: Proposals, identity: str, now: str
) -> tuple[Proposals, dict[str, Any]]:
    pending = _pending(identity, taste, proposals)
    rejected = {**pending, "status": "rejected", "decided_at": now}
    records = tuple(rejected if item["id"] == identity else item for item in proposals.proposals)
    return Proposals(records), copy.deepcopy(rejected)


def _import_rejected(
    taste: Taste, proposals: Proposals, body: dict, now: str
) -> tuple[Proposals, dict[str, Any]]:
    proposals, result = _import_proposal(taste, proposals, body, now)
    if result.status == "already_approved":
        raise TasteRefusal(
            f"Taste rule {result.id!r} is already approved.", code="taste_already_approved",
            suggested_action=f"Use yap-studio taste revoke {result.id} to end the approved rule.",
        )
    if result.status == "previously_rejected":
        record = next(item for item in proposals.proposals if item["id"] == result.id)
        return proposals, copy.deepcopy(record)
    return _reject(taste, proposals, result.id, now)


def approve_proposal(body: dict, *, reject: dict | None = None) -> ApproveResult:
    """Import and approve a body, optionally rejecting original wording in one save."""
    body = validate_body(body)
    reject_body = validate_body(reject) if reject is not None else None
    with _mutation() as before:
        taste, raw = before
        proposals, now = _reconcile(taste, raw), _now()
        proposals, imported = _import_proposal(taste, proposals, body, now, reopen=True)
        if reject_body is not None:
            proposals, _ = _import_rejected(taste, proposals, reject_body, now)
        if imported.status == "already_approved":
            approved = next(rule for rule in taste.rules if rule["id"] == imported.id)
            result = ApproveResult(copy.deepcopy(approved), (), approved["approved_revision"])
        else:
            taste, proposals, result = _approve(taste, proposals, imported.id, now)
        _save(taste, proposals, before)
        return result


def reject_proposal(body: dict) -> dict:
    """Import and reject a body in one save, refusing an approved rule."""
    body = validate_body(body)
    with _mutation() as before:
        taste, raw = before
        proposals, result = _import_rejected(taste, _reconcile(taste, raw), body, _now())
        _save(taste, proposals, before)
        return result


def revoke(rule_id: str) -> dict[str, Any]:
    """End an approved rule in a new revision, retaining its audit record."""
    with _mutation() as before:
        taste, raw = before
        active = next(
            (
                rule
                for rule in taste.rules
                if rule["id"] == rule_id and rule["status"] == "approved"
            ),
            None,
        )
        if active is None:
            raise TasteRefusal(
                f"Taste rule {rule_id!r} is not approved.",
                code="taste_not_approved",
                suggested_action="Use yap-studio taste show to choose an approved rule.",
            )
        revision, now = taste.revision + 1, _now()
        revoked = {**active, "status": "revoked", "ended_at": now, "ended_revision": revision}
        records = tuple(revoked if rule["id"] == rule_id else rule for rule in taste.rules)
        history = (*taste.history, _decision(revision, now, "revoked", revoked))
        _save(Taste(revision, now, records, history), _reconcile(taste, raw), before)
        return copy.deepcopy(revoked)


def _decision(
    revision: int, at: str, action: str, rule: Mapping[str, Any], *, by: str | None = None
) -> dict[str, Any]:
    """Copy an approval body into an immutable decision entry."""
    return {
        "revision": revision,
        "at": at,
        "action": action,
        "rule_id": rule["id"],
        "by": by,
        "rule": validate_body(
            {
                key: rule[key]
                for key in ("scope", "condition", "sentence", "machine", "sources", "context")
            }
        )
        if action in {"approved", "reapproved"}
        else None,
    }
