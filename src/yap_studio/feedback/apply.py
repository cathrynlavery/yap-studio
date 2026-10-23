"""Apply a saved feedback plan once, resuming interrupted fillers work."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Protocol, cast

from yap_studio import paths
from yap_studio.errors import InputError
from yap_studio.feedback.batch import (
    FeedbackError,
    StaleBatchError,
    _changed_inputs,
    _write_batch,
    current_inputs,
    load_batch,
)
from yap_studio.feedback.broll import (
    KINDS,
    LANE,
    decline_kept,
    plan_broll,
    publish_broll,
    require_hash,
    require_targets,
    saved_hash,
    verify_media,
)
from yap_studio.feedback.edits import EditPlan, plan_edit, validate_operations
from yap_studio.feedback.removed import _words, removed_ranges
from yap_studio.feedback.versions import (
    VersionManifest,
    _snapshot_json,
    current_version,
    load_version,
    version_cut_words,
    version_pieces,
)
from yap_studio.project.lock import project_lock, timestamp
from yap_studio.project.model import ProjectFileError, parse_document


class RemovalWriter(Protocol):
    """The fillers stage owns writes and takes its own project lock."""

    def add_manual(self, project_dir: Path, take: str, start: float, end: float, *,
                   reason: str | None = None) -> dict: ...

    def set_removal(self, project_dir: Path, identifier: str, status: str, *,
                    reason: str | None = None) -> dict: ...


class JoinTester(Protocol):
    """The fillers stage tests joins without writing project files."""

    def test_cuts(self, project_dir: Path, spans: list[dict], *, edit: dict | None = None,
                  removals: Sequence[dict] = (), rejects: Sequence[str] = ()) -> list[dict]: ...


class _FillersWriter:
    def __init__(self, module) -> None:
        self.module = module

    def add_manual(self, project_dir: Path, take: str, start: float, end: float, *,
                   reason: str | None = None) -> dict:
        return self.module.add_manual(project_dir, take, start, end, reason=reason)

    def set_removal(self, project_dir: Path, identifier: str, status: str, *,
                    reason: str | None = None) -> dict:
        return self.module.set_removal(project_dir, identifier, status, reason=reason)


def fillers_writer() -> RemovalWriter:
    """Load the documented pipeline step 2b seam only for removal work."""
    try:
        module = importlib.import_module("yap_studio.stages.fillers")
        if not all(callable(getattr(module, name, None))
                   for name in ("add_manual", "set_removal")):
            raise ImportError("missing fillers writer functions")
    except ImportError as exc:
        error = FeedbackError(
            "Removing words inside a segment and bringing back removed fillers need the "
            "fillers stage, which this build does not have yet.",
            suggested_action="Use trim, remove whole segments, or restore story cuts, "
            "or leave the note open.",
        )
        error.code = "needs_fillers"
        raise error from exc
    return _FillersWriter(module)


def join_tester() -> JoinTester:
    """Load the fillers join test only for requested cuts."""
    try:
        module = importlib.import_module("yap_studio.stages.fillers")
        if not callable(getattr(module, "test_cuts", None)):
            raise ImportError("missing fillers join test")
    except ImportError as exc:
        error = FeedbackError(
            "Testing requested cuts needs the fillers stage, which this build does not have yet.",
            suggested_action="Leave the cut note open until the fillers stage is available.",
        )
        error.code = "needs_fillers"
        raise error from exc
    return module


def _json_payload(edit: dict) -> bytes:
    return (json.dumps(edit, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _base(root: Path, batch: dict) -> VersionManifest:
    base_id = batch.get("base_version")
    if base_id is None:
        raise FeedbackError("This batch reviews the current cut without a saved base version.")
    current = current_version(root)
    if current is None or current.id != base_id:
        current_id = current.id if current else "none"
        raise FeedbackError(
            f"Batch base {base_id} is not the current version ({current_id}).",
            suggested_action="Review the current saved version and recompute the operations.",
        )
    return current


def _unique_operations(batch: dict, document: dict) -> None:
    recorded = {operation["id"] for item in batch["items"] for operation in item["operations"]}
    for item in document["items"]:
        for operation in item["operations"]:
            if operation["id"] in recorded:
                raise FeedbackError(f"Operation {operation['id']} was already applied.",
                                    suggested_action="Do not apply an operation twice.")


def _plan(root: Path, batch: dict, base: VersionManifest, document: dict) -> EditPlan:
    edit = cast(dict, _snapshot_json(base, "edit"))
    saved_pieces = version_pieces(base)
    inputs = saved_pieces.get("inputs", {})
    takes = {segment["take"] for segment in edit["segments"]}
    sources = {take: _words(root, take, inputs) for take in sorted(takes)}
    ranges, pieces = {}, {}
    for item in document["items"]:
        if not any(operation["kind"] == "restore" for operation in item["operations"]):
            continue
        version_id = item["comment"].split(":", 1)[0]
        if version_id in pieces:
            continue
        try:
            version = load_version(root, version_id)
        except InputError as exc:
            raise FeedbackError(
                f"Restore comment {item['comment']} needs saved version {version_id}.",
                suggested_action="Save versions before reviewing or leave the note open.",
            ) from exc
        pieces[version_id] = version_pieces(version)
        ranges[version_id] = removed_ranges(root, version)
    return plan_edit(edit, version_cut_words(base), sources, batch["items"], document,
                     ranges=ranges, pieces=pieces)


def _test_cut_plan(root: Path, batch: dict, base: VersionManifest, document: dict,
                   plan: EditPlan, joins: JoinTester | None) -> tuple[EditPlan, list[dict]]:
    declined, results = [], {}
    while plan.cut_spans:
        if joins is None:
            joins = join_tester()
        # Preserve indexes when reject actions are omitted from the audio context.
        removals = [{"action": index, "take": action.take, "start": action.start, "end": action.end}
                    for index, action in enumerate(plan.removal_actions)
                    if action.action == "add_manual"]
        rejects = tuple(action.removal_id for action in plan.removal_actions
                        if action.action == "reject")
        tested = joins.test_cuts(root, list(plan.cut_spans), edit=plan.edit,
                                 removals=removals, rejects=rejects)
        tested = [{**result, "combined_with": list(span["combined_with"])}
                  if span.get("combined_with") else result
                  for span, result in zip(plan.cut_spans, tested, strict=True)]
        results = {(span["operation"], span["segment"]): result
                   for span, result in zip(plan.cut_spans, tested, strict=True)}
        failed = [(span, result) for span, result in zip(plan.cut_spans, tested, strict=True)
                  if not result["pass"]]
        if not failed:
            break
        options = [option for span, _ in failed for option in span["options"]]
        clean: dict[str, list[dict]] = {}
        if options:
            alternatives = joins.test_cuts(root, options, edit=plan.edit,
                                           removals=removals, rejects=rejects)
            for option, result in zip(options, alternatives, strict=True):
                if result["pass"]:
                    clean.setdefault(option["operation"], []).append({
                        key: result[key] if key in {"start", "end"} else option[key]
                        for key in ("label", "words", "texts", "edge", "start", "end")
                    })
        for change in plan.changes:
            failures = [(span, result) for span, result in failed
                        if span["operation"] == change["operation"]]
            if not failures:
                continue
            descriptions = []
            for span, result in failures:
                text = " ".join(span["texts"])
                detail = "; ".join(result["diff"]) or "join check failed"
                descriptions.append(f'the join at "{text}" failed ({detail})')
            passing = clean.get(change["operation"], [])
            summary = "Not cut: " + "; ".join(descriptions) + "."
            summary += (" Clean options: " + "; ".join(option["label"] for option in passing) + "."
                        if passing else " No clean cut within one word; ask the creator.")
            declined.append({"operation": change["operation"], "comment": change["comment"],
                             "kind": "cut", "disposition": "declined", "segments": [], "words": [],
                             "join": [result for _, result in failures], "options": passing,
                             "summary": summary})
        comments = {span["comment"] for span, _ in failed}
        document = {**document, "items": [entry for entry in document["items"]
                                         if entry["comment"] not in comments]}
        plan = _plan(root, batch, base, document)
    actions = list(plan.removal_actions)
    for span in plan.cut_spans:
        key = span["operation"], span["segment"]
        result = results[key]
        if span["action"] is not None and any(result["nudge"]):
            at = span["action"]
            actions[at] = replace(actions[at], start=result["start"], end=result["end"])
    changes = []
    for change in plan.changes:
        if change["kind"] == "cut":
            tested = [results[span["operation"], span["segment"]] for span in plan.cut_spans
                      if span["operation"] == change["operation"]]
            change = {**change, "join": tested}
            if any(result["listen"] for result in tested):
                change["summary"] += " Please listen to this join."
        changes.append(change)
    return replace(plan, removal_actions=tuple(actions), changes=tuple(changes)), declined


def _application(batch: dict, digest: str) -> dict | None:
    application = batch.get("application")
    if application is None:
        return None
    if application["status"] == "pending":
        if application["operations_sha256"] != digest:
            raise FeedbackError(
                "This batch has a pending application for a different operations file. "
                "There is no clear command; rerun the same file to finish it.",
                suggested_action="Rerun the same operations file to finish the pending "
                "application.",
            )
        return application
    raise FeedbackError(
        "This batch already has an applied plan; further notes go into the next round.",
        suggested_action="Put further notes into the next feedback round.",
    )


def _result(batch: dict, application: dict, *, dry_run: bool) -> dict:
    result = {"batch": batch["id"], "dry_run": dry_run, "changes": application["changes"],
              "edit": application["edit"]["segments"],
              "removal_actions": application["removal_actions"],
              "status": "planned" if dry_run else "applied"}
    broll = application.get("broll")
    if broll is not None:
        result.update(broll_operations=broll["operations"], broll=broll["lane"]["elements"],
                      findings=broll["findings"])
        if not broll["valid"]:
            result["status"] = "refused"
    declined = [change for change in application["changes"]
                if change.get("disposition") == "declined"]
    if declined:
        result["dispositions"] = declined
    return result


def _removals_state(root: Path) -> tuple[dict | None, str | None]:
    """Read one document and hash those same bytes while the caller holds the lock."""
    try:
        payload = (root / "removals.json").read_bytes()
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        raise StaleBatchError(["removals.json"]) from exc
    try:
        document = json.loads(payload)
        parse_document("removals", document)
    except (ValueError, UnicodeError, ProjectFileError) as exc:
        raise StaleBatchError(["removals.json"]) from exc
    return document, hashlib.sha256(payload).hexdigest()


def _own_removal_change(before: dict | None, current: dict | None, action: dict) -> str:
    """Accept exactly one action's delta, preserving every unrelated entry."""
    before = before if before is not None else {"schema": "yap-studio/removals@1", "removals": []}
    if current is None or before["schema"] != current["schema"]:
        raise StaleBatchError(["removals.json"])
    previous = {entry["id"]: entry for entry in before["removals"]}
    entries = {entry["id"]: entry for entry in current["removals"]}
    if action["action"] == "add_manual":
        added = entries.keys() - previous.keys()
        if len(added) == 1:
            identifier = added.pop()
            entry = entries[identifier]
            if (entry["status"] == "applied" and entry["kind"] == "manual"
                    and (entry["take"], entry["start"], entry["end"])
                    == (action["take"], action["start"], action["end"])
                    and action["operation"] in (entry.get("reason") or "")
                    and [entry for entry in current["removals"] if entry["id"] != identifier]
                    == before["removals"]):
                return identifier
    else:
        identifier = action["removal_id"]
        if (previous.keys() == entries.keys() and identifier in entries
                and previous[identifier]["status"] == "applied"
                and entries[identifier]["status"] == "rejected"
                and [entry for entry in current["removals"] if entry["id"] != identifier]
                == [entry for entry in before["removals"] if entry["id"] != identifier]):
            entry = entries[identifier]
            expected = copy.deepcopy(previous[identifier])
            expected["status"] = "rejected"
            if "reason" in entry:
                expected["reason"] = entry["reason"]
            else:
                expected.pop("reason", None)
            if entry.get("evidence", {}).get("decision") is True:
                expected["evidence"] = {**expected.get("evidence", {}), "decision": True}
            if entry == expected:
                return identifier
    raise StaleBatchError(["removals.json"])


def _record_removal_change(application: dict, action: dict, current: dict | None,
                           digest: str | None) -> None:
    identifier = _own_removal_change(application["removals"], current, action)
    action.update(done=True, removal_id=identifier)
    application.update(removals=current, removals_sha256=digest)


def apply_operations(project_dir: Path, operations: object, *, dry_run: bool = False,
                     removals: RemovalWriter | None = None,
                     joins: JoinTester | None = None) -> dict:
    """Journal and apply operations to their saved base, or resume that journal.

    A dry run only reads project files. Writers own removals.json and run outside
    our short lock sections. The journal preserves the edit and summaries so a
    retry never plans against partially applied inputs.
    """
    document = validate_operations(operations)
    digest = hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    root = Path(project_dir).resolve()
    # project_lock writes its bookkeeping; a read-only preview must not take it.
    with nullcontext() if dry_run else project_lock(root):
        batch = load_batch(root, document["batch"])
        application = _application(batch, digest)
        _unique_operations(batch, document)
        base = _base(root, batch)
        if application is None:
            before = current_inputs(root)
            has_broll = any(operation["kind"] in KINDS for item in document["items"]
                            for operation in item["operations"])
            if has_broll:
                require_hash(root, saved_hash(base))
                require_hash(root, batch["expected_inputs"].get(LANE))
            changed = _changed_inputs(before, batch["expected_inputs"])
            if changed:
                raise StaleBatchError(changed)
            if before["edit.json"] != base.snapshots["edit"]["sha256"]:
                raise StaleBatchError(["edit.json"])
            accepted, declined = decline_kept(document, batch["constraints"])
            plan = _plan(root, batch, base, accepted)
            plan, failed_cuts = _test_cut_plan(root, batch, base, accepted, plan, joins)
            declined.extend(failed_cuts)
            if plan.broll_operations:
                require_targets(root, plan.broll_operations)
            broll = (plan_broll(root, plan.broll_operations, edit=plan.edit,
                                story_changed=plan.edit != _snapshot_json(base, "edit"),
                                removal_actions=plan.removal_actions)
                     if plan.broll_operations else None)
            # Preserve bytes for a plan that only records keeps or removal work.
            after = (before["edit.json"] if plan.edit == _snapshot_json(base, "edit") else
                     hashlib.sha256(_json_payload(plan.edit)).hexdigest())
            removals_document, removals_digest = _removals_state(root)
            if removals_digest != before["removals.json"]:
                raise StaleBatchError(["removals.json"])
            application = {
                "status": "pending", "operations": document, "operations_sha256": digest,
                "inputs_before": before, "edit_sha256_before": before["edit.json"],
                "edit_sha256_after": after,
                "removals": removals_document, "removals_sha256": removals_digest,
                "removal_actions": [{**asdict(action), "done": False}
                                    for action in plan.removal_actions],
                "started_at": timestamp(), "edit": plan.edit,
                "changes": [*plan.changes, *(broll["changes"] if broll else []), *declined],
            }
            if broll is not None:
                application["broll"] = broll
                if not broll["valid"] and not dry_run:
                    raise FeedbackError("B-roll timeline refused: " + "; ".join(
                        finding["message"] for finding in broll["findings"]
                        if finding["severity"] == "violation"))
        if (batch.get("application") is not application and removals is None
                and any(not action["done"] for action in application["removal_actions"])):
            removals = fillers_writer()
        if dry_run:
            if application.get("broll") is not None:
                require_hash(root, application["broll"]["sha256_before"],
                             after=application["broll"]["sha256_after"])
            return _result(batch, application, dry_run=True)
        if batch.get("application") is not application:
            paths.validate_output(root / "edit.json")
            batch["application"] = application
            _write_batch(root, batch)

    for index in range(len(application["removal_actions"])):
        with project_lock(root):
            batch = load_batch(root, document["batch"])
            application = _application(batch, digest)
            action = application["removal_actions"][index]
            if action["done"]:
                continue
            current_removals, current_digest = _removals_state(root)
            if current_digest != application["removals_sha256"]:
                # A writer may have succeeded before its completion flag was saved.
                _record_removal_change(application, action, current_removals, current_digest)
                _write_batch(root, batch)
                continue
            # The last verified document is this action's durable pre-action state.
            _write_batch(root, batch)
        if removals is None:
            removals = fillers_writer()
        reason = f"Operation {action['operation']} ({action['comment']}): {action['reason']}"
        if action["action"] == "add_manual":
            removals.add_manual(root, action["take"], action["start"], action["end"], reason=reason)
        else:
            removals.set_removal(root, action["removal_id"], "rejected", reason=reason)
        with project_lock(root):
            batch = load_batch(root, document["batch"])
            application = _application(batch, digest)
            action = application["removal_actions"][index]
            current_removals, current_digest = _removals_state(root)
            _record_removal_change(application, action, current_removals, current_digest)
            _write_batch(root, batch)

    with project_lock(root):
        batch = load_batch(root, document["batch"])
        application = _application(batch, digest)
        _base(root, batch)
        _unique_operations(batch, document)
        current = current_inputs(root)
        if current["edit.json"] not in (
            application["edit_sha256_before"], application["edit_sha256_after"],
        ):
            raise StaleBatchError(["edit.json"])
        if current["removals.json"] != application["removals_sha256"]:
            raise StaleBatchError(["removals.json"])
        broll = application.get("broll")
        if broll is not None:
            require_hash(root, broll["sha256_before"], after=broll["sha256_after"])
            verify_media(root, broll["lane"])
        changed = [name for name in _changed_inputs(current, application["inputs_before"])
                   if name not in {"edit.json", "removals.json"}
                   and not (broll is not None and name == LANE)]
        if changed:
            raise StaleBatchError(changed)
        if current["edit.json"] != application["edit_sha256_after"]:
            paths.atomic_write_text(root / "edit.json", _json_payload(application["edit"]).decode())
        if broll is not None:
            publish_broll(root, broll)
        after = current_inputs(root)
        summaries = {change["operation"]: change["summary"] for change in application["changes"]}
        by_comment = {f"{item['comment']['version']}:{item['comment']['id']}": item
                      for item in batch["items"]}
        for entry in document["items"]:
            item = by_comment[entry["comment"]]
            conflict = next((change for change in application["changes"]
                             if change["comment"] == entry["comment"]
                             and change.get("disposition") == "declined"), None)
            if conflict is not None:
                evidence = ({"join": conflict["join"], "options": conflict["options"],
                             "reason": conflict["summary"]} if "join" in conflict else
                            {"protected_by": conflict["keep"], "reason": conflict["summary"]})
                item.update(disposition="declined", evidence=evidence)
                continue
            item["operations"].extend({
                **copy.deepcopy(operation), "inputs_before": copy.deepcopy(
                    application["inputs_before"]), "inputs_after": copy.deepcopy(after),
                "summary": summaries[operation["id"]],
            } for operation in entry["operations"])
            item["disposition"] = "applied"
        batch["expected_inputs"] = after
        application.update(status="applied", finished_at=timestamp())
        _write_batch(root, batch)
        return _result(batch, application, dry_run=False)
