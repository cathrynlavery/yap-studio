"""Portable taste proposals for import by the separate taste feature."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from yap_studio.errors import InputError
from yap_studio.paths import atomic_write_text
from yap_studio.project.lock import project_lock
from yap_studio.review.model import Comment, MediaFacts, conform_comment

PROPOSALS_SCHEMA = "yap-studio/taste-proposals@1"
_SOURCE_KEYS = {"type", "media_sha256", "comment_id", "label", "start_frame", "end_frame", "quote"}


class TasteProposalSink(Protocol):
    """Accept proposal bodies without depending on a taste implementation."""

    def propose(self, bodies: Sequence[dict]) -> Sequence[str] | None: ...


def _validate_body(body: object) -> None:
    # Import lazily so the public taste API can also expose the project inbox.
    from yap_studio.taste.model import validate_body

    body = validate_body(body)
    if not isinstance(body["context"], str) or not 1 <= len(body["context"]) <= 500:
        raise ValueError("invalid context")
    seen = set()
    for source in body["sources"]:
        if not isinstance(source, dict) or set(source) != _SOURCE_KEYS:
            raise ValueError("invalid source fields")
        if source["type"] != "comment" or source["label"] not in {"keep", "fix"}:
            raise ValueError("expected a keep or fix comment source")
        if (not isinstance(source["media_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", source["media_sha256"]) is None
                or not isinstance(source["comment_id"], str)
                or re.fullmatch(r"c[1-9][0-9]*", source["comment_id"]) is None):
            raise ValueError("invalid source identity")
        for key in ("start_frame", "end_frame"):
            if source[key] is not None and (type(source[key]) is not int or source[key] < 0):
                raise ValueError("source frames must be non-negative integers or null")
        if not isinstance(source["quote"], str) or len(source["quote"]) > 500:
            raise ValueError("invalid source quote")
        identity = (source["media_sha256"], source["comment_id"])
        if identity in seen:
            raise ValueError("duplicate source identity")
        seen.add(identity)


def _words_text(template: str, wrong: str, right: str, limit: int) -> str:
    available = limit - len(template.format(wrong="", right=""))
    left = min(len(wrong), max(available // 2, available - len(right)))
    return template.format(wrong=wrong[:left], right=right[:available - left])


def _comment_source(
    *, media_sha256: str, comment: Comment, fps: Fraction | None,
) -> dict:
    start, end = comment.start_frame, comment.end_frame
    if start is None and fps is not None:
        timed = conform_comment(comment, MediaFacts("render", media_sha256, None, fps, True))
        start, end = timed.start_frame, timed.end_frame
    return {"type": "comment", "media_sha256": media_sha256,
            "comment_id": comment.id, "label": comment.kind, "start_frame": start,
            "end_frame": end, "quote": comment.text[:500]}


def spelling_proposal(
    *, wrong: str, right: str, media_sha256: str, comment: Comment, fps: Fraction | None,
) -> dict:
    """Build the exact import body, retaining the comment's own timeline and identity."""
    source = _comment_source(media_sha256=media_sha256, comment=comment, fps=fps)
    source["label"] = "fix"
    return {
        "scope": "captions", "condition": None,
        "sentence": _words_text('Spell "{wrong}" as "{right}" in captions.', wrong, right, 280),
        "machine": None,
        "sources": [source],
        "context": _words_text(
            'Caption spelling fix from review. Word fix: "{wrong}" -> "{right}".',
            wrong, right, 500,
        ),
    }


class ProposalInbox:
    """Merge validated import bodies under the project lock and replace atomically."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir).resolve()

    @property
    def path(self) -> Path:
        return self.project_dir / "feedback" / "taste-proposals.json"

    def read(self) -> list[dict]:
        """Read validated bodies without writing, returning an empty absent inbox."""
        try:
            if not self.path.exists() and not self.path.is_symlink():
                return []
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or set(data) != {"schema", "proposals"}
                    or data["schema"] != PROPOSALS_SCHEMA
                    or not isinstance(data["proposals"], list)):
                raise ValueError("not a taste proposals inbox")
            for body in data["proposals"]:
                _validate_body(body)
            return data["proposals"]
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise InputError(str(self.path), "not a valid taste proposals file") from exc

    def propose(self, bodies: Sequence[dict]) -> list[str]:
        """Merge under the project lock, returning one outcome per incoming body."""
        incoming = copy.deepcopy(list(bodies))
        with project_lock(self.project_dir):
            try:
                for body in incoming:
                    _validate_body(body)
                data = {"schema": PROPOSALS_SCHEMA, "proposals": self.read()}
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                raise InputError(str(self.path), "not a valid taste proposals file") from exc
            original = copy.deepcopy(data)
            outcomes = []
            for body in incoming:
                match = next((proposal for proposal in data["proposals"] if all(
                    proposal[key] == body[key]
                    for key in ("scope", "condition", "sentence", "machine")
                )), None)
                if match is None:
                    data["proposals"].append(body)
                    outcomes.append("added")
                    continue
                source_count = len(match["sources"])
                identities = {(source["media_sha256"], source["comment_id"])
                              for source in match["sources"]}
                for source in body["sources"]:
                    identity = (source["media_sha256"], source["comment_id"])
                    if identity not in identities and len(match["sources"]) < 50:
                        match["sources"].append(source)
                        identities.add(identity)
                outcomes.append("merged" if len(match["sources"]) != source_count else "unchanged")
            if data != original:
                atomic_write_text(self.path, json.dumps(data, indent=2, ensure_ascii=False,
                                                       allow_nan=False) + "\n")
            return outcomes
