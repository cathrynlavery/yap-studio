"""Strict round records and pure comment send states."""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from yap_studio.review.model import Comment

ROUNDS_SCHEMA = "yap-studio/review-rounds@1"
INTERRUPTED_ERROR = {
    "code": "interrupted",
    "message": "The review server stopped while sending these notes. Send them again.",
}


@dataclass(frozen=True)
class Round:
    """A version-qualified collection of exact comment revisions to send together."""

    id: str
    version: int
    comments: tuple[dict, ...]
    status: str
    step: str | None
    batch: str | None
    error: dict | None
    attempts: int
    created_at: str
    updated_at: str
    answers: tuple[dict, ...] = ()
    owner: dict | None = None


def _fields(value: object, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"expected exactly these fields: {', '.join(sorted(keys))}")
    return value


def _id(value: object, prefix: str) -> bool:
    return isinstance(value, str) and re.fullmatch(prefix + r"[1-9][0-9]*", value) is not None


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _timestamp(value: object) -> None:
    if (not isinstance(value, str)
            or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?Z", value) is None):
        raise ValueError("expected a UTC timestamp")
    datetime.fromisoformat(value)


def round_from_dict(data: object) -> Round:
    """Refuse unknown fields, invalid references, and inconsistent send states."""
    if isinstance(data, dict) and "answers" not in data:
        data = {**data, "answers": []}
    if isinstance(data, dict) and "owner" not in data:
        data = {**data, "owner": None}
    data = _fields(data, {
        "id", "version", "comments", "status", "step", "batch", "error", "attempts",
        "created_at", "updated_at", "answers", "owner",
    })
    owner = data["owner"]
    if owner is not None:
        owner = _fields(owner, {"pid", "instance"})
        if (type(owner["pid"]) is not int or owner["pid"] <= 0
                or not isinstance(owner["instance"], str)
                or re.fullmatch(r"[0-9a-f]+", owner["instance"]) is None):
            raise ValueError("invalid round owner")
    if not _id(data["id"], "n"):
        raise ValueError("invalid round id")
    if type(data["version"]) is not int or data["version"] < 1:
        raise ValueError("version must be a positive integer")
    if type(data["attempts"]) is not int or data["attempts"] < 1:
        raise ValueError("attempts must be a positive integer")
    if not isinstance(data["answers"], list):
        raise ValueError("answers must be an array")
    answers, answer_refs = [], set()
    for raw in data["answers"]:
        item = _fields(raw, {"batch", "comment"})
        ref = _fields(item["comment"], {"version", "id"})
        if (not _id(item["batch"], "b") or not _id(ref["version"], "v")
                or not _id(ref["id"], "c")):
            raise ValueError("invalid answer reference")
        key = (item["batch"], ref["version"], ref["id"])
        if key in answer_refs:
            raise ValueError("duplicate answer reference")
        answer_refs.add(key)
        answers.append(copy.deepcopy(item))
    if (not isinstance(data["comments"], list) or not data["comments"] and not answers):
        raise ValueError("comments may be empty only when there are answers")
    comments, ids = [], set()
    for raw in data["comments"]:
        item = _fields(raw, {"id", "revision"})
        if not _id(item["id"], "c") or item["id"] in ids:
            raise ValueError("invalid or duplicate comment id")
        _timestamp(item["revision"])
        ids.add(item["id"])
        comments.append(dict(item))
    status, step, batch, error = (data[key] for key in ("status", "step", "batch", "error"))
    if not isinstance(status, str) or status not in ("sending", "sent", "failed"):
        raise ValueError("invalid round status")
    if step is not None and (not _text(step) or len(step) > 100):
        raise ValueError("step must be short text or null")
    if batch is not None and not _id(batch, "b"):
        raise ValueError("invalid batch id")
    if error is not None:
        error = _fields(error, {"code", "message"})
        if not _text(error["code"]) or not _text(error["message"]):
            raise ValueError("error must have a code and message")
    if (status == "sending" and (batch is not None or error is not None)
            or status == "sent" and (batch is None or error is not None or step is not None)
            or status == "failed" and (error is None or batch is not None or step is not None)):
        raise ValueError("inconsistent round state")
    _timestamp(data["created_at"])
    _timestamp(data["updated_at"])
    if datetime.fromisoformat(data["updated_at"]) < datetime.fromisoformat(data["created_at"]):
        raise ValueError("updated_at precedes created_at")
    return Round(data["id"], data["version"], tuple(comments), status, step, batch,
                 copy.deepcopy(error), data["attempts"], data["created_at"], data["updated_at"],
                 tuple(answers), copy.deepcopy(owner))


def round_to_dict(round: Round) -> dict:
    """Return a validated JSON record without sharing mutable references."""
    value = {
        "id": round.id, "version": round.version, "comments": copy.deepcopy(list(round.comments)),
        "status": round.status, "step": round.step, "batch": round.batch,
        "error": copy.deepcopy(round.error), "attempts": round.attempts,
        "created_at": round.created_at, "updated_at": round.updated_at,
        "answers": copy.deepcopy(list(round.answers)),
        "owner": copy.deepcopy(round.owner),
    }
    round_from_dict(value)
    return value


def comment_states(comments: Sequence[Comment], rounds: Sequence[Round]) -> dict[str, dict]:
    """Compute states for one version, preferring locks over matching failed revisions."""
    states = {item.id: {"state": "draft", "round": None, "batch": None, "error": None}
              for item in comments}
    revisions = {item.id: item.updated_at for item in comments}
    for round in sorted(rounds, key=lambda item: int(item.id[1:])):
        for ref in round.comments:
            if ref["id"] not in states:
                continue
            state = states[ref["id"]]
            if state["state"] in ("sending", "sent"):
                continue
            if round.status == "failed" and revisions[ref["id"]] != ref["revision"]:
                continue
            states[ref["id"]] = {
                "state": round.status, "round": round.id, "batch": round.batch,
                "error": copy.deepcopy(round.error),
            }
    return states
