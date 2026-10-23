"""Apply caption spelling requests once and hand off their proposed taste rules."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from yap_studio.errors import InputError
from yap_studio.feedback.batch import (
    _batch_numbers,
    _write_batch,
    check_inputs,
    load_batch,
    set_disposition,
)
from yap_studio.feedback.taste import TasteProposalSink, spelling_proposal
from yap_studio.feedback.versions import load_version
from yap_studio.project.lock import project_lock, write_json_cas
from yap_studio.project.model import parse_document
from yap_studio.receipt import sha256_file
from yap_studio.review.model import Comment, normalize_request


def apply_word_fixes(
    project_dir: Path, batch_id: str, *, sink: TasteProposalSink | None = None, now=None,
) -> list[dict]:
    """Record project changes under one lock, then deliver any pending proposal handoff."""
    root = Path(project_dir).resolve()
    path = root / "project.json"
    applied, proposals, pending = [], [], []
    with project_lock(root):
        batch = load_batch(root, batch_id)
        newest, seen = {}, {}
        for number in sorted(_batch_numbers(root)):
            saved = load_batch(root, f"b{number}")
            for order, item in enumerate(saved["items"]):
                raw = item["comment"]
                for operation in item["operations"]:
                    if operation.get("op") != "word_fix":
                        continue
                    ref = {key: raw[key] for key in ("version", "id", "revision")}
                    key = (raw["version"], raw["id"], raw["revision"])
                    seen.setdefault(key, saved["id"])
                    rank = (raw["revision"], number, order)
                    previous = newest.get(operation["from"])
                    if previous is None or rank > previous[0]:
                        newest[operation["from"]] = (rank, ref)
        for order, item in sorted(enumerate(batch["items"]),
                                  key=lambda entry: (entry[1]["comment"]["revision"], entry[0])):
            raw = item["comment"]
            request = normalize_request(item.get("request", raw.get("request")),
                                        kind=raw["kind"], target=raw["target"])
            if (request is None or request["type"] != "spelling"
                    or item["disposition"] not in ("open", "applied")):
                continue
            operation_id = f'word-fix:{raw["version"]}:{raw["id"]}'
            if item["disposition"] == "open":
                key = (raw["version"], raw["id"], raw["revision"])
                prior_batch = seen.get(key)
                if prior_batch is not None and prior_batch != batch_id:
                    set_disposition(root, batch_id, raw["version"], raw["id"], "applied",
                                    evidence={"already_applied_in": prior_batch}, _locked=True)
                    continue
                previous = newest.get(request["from"])
                if previous is not None and raw["revision"] < previous[0][0]:
                    set_disposition(root, batch_id, raw["version"], raw["id"], "declined",
                                    evidence={"superseded_by": previous[1]}, _locked=True)
                    continue
                check_inputs(root, batch)
                before = sha256_file(path)
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, ValueError) as exc:
                    raise InputError(str(path), "cannot read valid project JSON") from exc
                parse_document("project", data)
                fixes = data.setdefault("word_fixes", {})
                old = fixes.get(request["from"])
                operation = {"id": operation_id, "op": "word_fix",
                             "from": request["from"], "to": request["to"]}
                if old is not None and old != request["to"]:
                    operation["replaced"] = old
                after = before
                if old != request["to"]:
                    fixes[request["from"]] = request["to"]
                    parse_document("project", data)
                    after = write_json_cas(path, data, base_sha256=before)
                set_disposition(root, batch_id, raw["version"], raw["id"], "applied",
                                operations=[operation],
                                evidence={"project_json": {"before": before, "after": after}},
                                _locked=True)
                applied.append(operation)
                item["disposition"] = "applied"
                item["operations"].append(operation)
                seen[key] = batch_id
                newest[request["from"]] = (
                    (raw["revision"], int(batch_id[1:]), order),
                    {key: raw[key] for key in ("version", "id", "revision")},
                )
            if (sink is not None and not item.get("taste_proposed", False)
                    and any(op["id"] == operation_id for op in item["operations"])):
                version = load_version(root, raw["version"]) if raw["version"] else None
                comment = Comment(
                    **{key: value for key, value in raw.items()
                       if key not in ("version", "revision")},
                    created_at=raw["revision"], updated_at=raw["revision"],
                )
                proposals.append(spelling_proposal(
                    wrong=request["from"], right=request["to"], comment=comment,
                    media_sha256=version.media["sha256"] if version else
                    batch["review"]["media_sha256"],
                    fps=Fraction(version.media["fps"])
                    if version and comment.start_frame is not None else None,
                ))
                pending.append((raw["version"], raw["id"]))
    if proposals and sink is not None:
        sink.propose(proposals)
        with project_lock(root):
            batch = load_batch(root, batch_id)
            for item in batch["items"]:
                if (item["comment"]["version"], item["comment"]["id"]) in pending:
                    item["taste_proposed"] = True
            _write_batch(root, batch)
    return applied
