"""Comment-backed project proposals and decisions resolved against private taste."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback.taste import ProposalInbox, _comment_source, _validate_body
from yap_studio.paths import atomic_write_text
from yap_studio.project.lock import project_lock
from yap_studio.review.store import ReviewStore, data_dir
from yap_studio.taste import store as taste_store
from yap_studio.taste.model import TasteError, TasteRefusal, rule_id, validate_body

DECISIONS_SCHEMA = "yap-studio/taste-decisions@1"
_ID = re.compile(r"r-[0-9a-f]{12}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_PROPOSAL_KEYS = {"scope", "sentence", "comments", "condition", "machine", "context"}


class UnknownProposalError(ValidationError):
    """Report a proposal id absent from this project's inbox."""

    def __init__(self, proposal_id: str) -> None:
        super().__init__("proposal_id", f"no inbox proposal {proposal_id}")
        self.code = "unknown_proposal"
        self.suggested_action = "Choose a proposal from this project's inbox."


def load_version(project_dir: Path, version_id: str):  # Lazy: CLI parsing must stay light.
    from yap_studio.feedback.versions import load_version as load
    return load(project_dir, version_id)


def _input(source: Path | Mapping[str, Any]) -> list[tuple[Mapping, str]]:
    if isinstance(source, Path):
        try:
            source = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise InputError(str(source), "not a valid proposal JSON file") from exc
    if not isinstance(source, Mapping):
        raise ValidationError("proposal", "expected object")
    if "proposals" not in source:
        return [(source, "proposal")]
    if set(source) != {"proposals"}:
        raise ValidationError("proposals", "unknown field")
    if not isinstance(source["proposals"], list):
        raise ValidationError("proposals", "expected array")
    return [(value, f"proposals[{index}]") for index, value in enumerate(source["proposals"])]


def _build(project_dir: Path, value: Mapping, location: str, store: ReviewStore) -> dict:
    if not isinstance(value, Mapping):
        raise ValidationError(location, "expected object")
    for key in value:
        if key not in _PROPOSAL_KEYS:
            raise ValidationError(f"{location}.{key}", "unknown field")
    for key in ("scope", "sentence", "comments"):
        if key not in value:
            raise ValidationError(f"{location}.{key}", "missing required field")
    comments = value["comments"]
    if not isinstance(comments, list) or not 1 <= len(comments) <= 50:
        raise ValidationError(f"{location}.comments", "expected 1 to 50 comments")
    sources, seen = [], set()
    for index, ref in enumerate(comments):
        item = f"{location}.comments[{index}]"
        if not isinstance(ref, Mapping) or set(ref) != {"version", "id"}:
            raise ValidationError(item, "expected exactly version and id")
        for key, pattern in (("version", r"v[1-9][0-9]*"), ("id", r"c[1-9][0-9]*")):
            if not isinstance(ref[key], str) or re.fullmatch(pattern, ref[key]) is None:
                raise ValidationError(f"{item}.{key}", f"must match {pattern}")
        identity = (ref["version"], ref["id"])
        if identity in seen:
            raise ValidationError(item, "duplicate comment reference")
        seen.add(identity)
        try:
            version = load_version(project_dir, ref["version"])
        except InputError as exc:
            error = ValidationError(item, f"no version {ref['version']}")
            error.suggested_action = "Choose an existing saved version and review comment."
            raise error from exc
        sha = version.media["sha256"]
        comment = next((note for note in store.comments(sha) if note.id == ref["id"]), None)
        if comment is None:
            error = ValidationError(item, f"no comment {ref['id']} on version {ref['version']}")
            error.suggested_action = "Choose a saved review comment on the cited version."
            raise error
        fps = Fraction(version.media["fps"]) if version.media.get("fps") is not None else None
        sources.append(_comment_source(media_sha256=sha, comment=comment, fps=fps))
    body = validate_body({
        "scope": value["scope"], "sentence": value["sentence"],
        "condition": value.get("condition"), "machine": value.get("machine"),
        "context": value.get("context", "Proposed from review notes."), "sources": sources,
    }, location)
    try:
        _validate_body(body)
    except ValueError as exc:
        raise TasteError(location, str(exc)) from exc
    return body


def propose(
    project_dir: Path, source: Path | Mapping[str, Any], *, store: ReviewStore | None = None
) -> list[dict]:
    """Resolve cited review comments, validate all bodies, and merge the project inbox."""
    root = Path(project_dir).resolve()
    review_store = store if store is not None else ReviewStore(data_dir())
    bodies = [_build(root, value, location, review_store) for value, location in _input(source)]
    outcomes = ProposalInbox(root).propose(bodies)
    return [{"id": rule_id(body), "status": status}
            for body, status in zip(bodies, outcomes, strict=True)]


def _ledger_path(project_dir: Path) -> Path:
    return project_dir / "feedback" / "taste-decisions.json"


def _read_ledger(project_dir: Path) -> list[dict]:
    path = _ledger_path(project_dir)
    try:
        if not path.exists() and not path.is_symlink():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or set(data) != {"schema", "decisions"}
                or data["schema"] != DECISIONS_SCHEMA or not isinstance(data["decisions"], list)):
            raise ValueError("not a taste decisions ledger")
        for entry in data["decisions"]:
            if not isinstance(entry, dict) or set(entry) != {"proposal", "action", "rule", "at"}:
                raise ValueError("invalid decision fields")
            for key in ("proposal", "rule"):
                if not isinstance(entry[key], str) or _ID.fullmatch(entry[key]) is None:
                    raise ValueError("invalid decision id")
            if entry["action"] not in ("approved", "edited", "rejected", "revoked"):
                raise ValueError("invalid decision action")
            if not isinstance(entry["at"], str) or _TIMESTAMP.fullmatch(entry["at"]) is None:
                raise ValueError("invalid decision timestamp")
            datetime.strptime(entry["at"], "%Y-%m-%dT%H:%M:%SZ")
        return data["decisions"]
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise InputError(str(path), "not a valid taste decisions file") from exc


def _view(bodies: list[dict], decisions: list[dict]) -> dict:
    taste, proposals = taste_store.load_taste(), taste_store.load_proposals()
    rules = {rule["id"]: rule for rule in taste.rules}
    rejected = {item["id"]: item for item in proposals.proposals if item["status"] == "rejected"}
    latest = {entry["proposal"]: entry for entry in decisions}
    items = []
    for body in bodies:
        identity = rule_id(body)
        decision = latest.get(identity)
        target = decision["rule"] if decision is not None else identity
        rule = rules.get(target)
        item = {**copy.deepcopy(body), "id": identity, "state": "pending", "edited": False,
                "rule": None, "decided_at": None, "decided_here": False}
        if rule is not None:
            item.update(state=rule["status"], edited=target != identity,
                        rule={key: rule[key] for key in (
                            "id", "sentence", "status", "approved_revision", "ended_revision"
                        )}, decided_at=rule["ended_at"] or rule["approved_at"])
        elif identity in rejected:
            item.update(state="rejected", decided_at=rejected[identity]["decided_at"])
        if decision is not None:
            expected = {"approved": "approved", "edited": "approved", "rejected": "rejected",
                        "revoked": "revoked"}[decision["action"]]
            item["decided_here"] = (decision["rule"] == (rule["id"] if rule else identity)
                                    and item["state"] == expected)
        items.append(item)
    return {"revision": taste.revision, "proposals": items}


def inbox_view(project_dir: Path) -> dict:
    """Derive proposal states from private records and ledger targets without writing."""
    root = Path(project_dir).resolve()
    return _view(ProposalInbox(root).read(), _read_ledger(root))


def _ended() -> TasteRefusal:
    return TasteRefusal(
        "This taste rule has ended.", code="taste_ended",
        suggested_action="Leave a new review note to propose a new taste rule.",
    )


def decide(
    project_dir: Path, proposal_id: str, action: str, *, sentence: str | None = None
) -> dict:
    """Record intent then apply one creator decision while holding the project lock."""
    root = Path(project_dir).resolve()
    if action not in ("approve", "reject", "revoke"):
        raise ValidationError("action", "choose one of approve, reject, revoke")
    # The private store acquires only its taste lock, never this project lock.
    with project_lock(root):
        bodies, decisions = ProposalInbox(root).read(), _read_ledger(root)
        item = next((item for item in _view(bodies, decisions)["proposals"]
                     if item["id"] == proposal_id), None)
        if item is None:
            raise UnknownProposalError(proposal_id)
        body = next(body for body in bodies if rule_id(body) == proposal_id)
        latest = next((entry for entry in reversed(decisions)
                       if entry["proposal"] == proposal_id), None)
        # Finish an edited approval whose proposals write was interrupted.
        if (latest is not None and latest["action"] == "edited" and item["state"] == "approved"
                and item["rule"]["id"] == latest["rule"]
                and not any(rule["id"] == proposal_id and rule["status"] == "approved"
                            for rule in taste_store.load_taste().rules)
                and not any(proposal["id"] == proposal_id and proposal["status"] == "rejected"
                            for proposal in taste_store.load_proposals().proposals)):
            taste_store.reject_proposal(body)
        state, target, edited_body = item["state"], proposal_id, None
        if action == "approve":
            if sentence is not None and not isinstance(sentence, str):
                raise TasteError("sentence", "expected string")
            wording = sentence.strip() if sentence is not None else body["sentence"]
            if wording != body["sentence"]:
                if state not in {"pending", "rejected"}:
                    raise TasteRefusal(
                        "Only a pending or rejected proposal can be edited.",
                        code="taste_not_pending", suggested_action="Leave a new review note.",
                    )
                edited_body = validate_body({**body, "sentence": wording})
                target, ledger_action = rule_id(edited_body), "edited"
            else:
                if state == "approved":
                    return item
                if state in {"revoked", "superseded"}:
                    raise _ended()
                ledger_action = "approved"
        elif action == "reject":
            if state == "rejected":
                return item
            if state == "approved":
                raise TasteRefusal(
                    "This taste rule is approved.", code="taste_approved",
                    suggested_action="Use the revoke action to stop this approved rule.",
                )
            if state != "pending":
                raise _ended()
            ledger_action = "rejected"
        else:
            if state != "approved":
                raise TasteRefusal(
                    "This taste rule is not approved.", code="taste_not_approved",
                    suggested_action="Choose an approved proposal to revoke.",
                )
            target, ledger_action = item["rule"]["id"], "revoked"
        decisions.append({"proposal": proposal_id, "action": ledger_action, "rule": target,
                          "at": taste_store._now()})
        atomic_write_text(_ledger_path(root), json.dumps(
            {"schema": DECISIONS_SCHEMA, "decisions": decisions}, indent=2, allow_nan=False,
        ) + "\n")
        if action == "approve":
            if edited_body is not None:
                taste_store.approve_proposal(edited_body, reject=body)
            else:
                taste_store.approve_proposal(body)
        elif action == "reject":
            taste_store.reject_proposal(body)
        else:
            taste_store.revoke(target)
        return next(item for item in _view(bodies, decisions)["proposals"]
                    if item["id"] == proposal_id)
