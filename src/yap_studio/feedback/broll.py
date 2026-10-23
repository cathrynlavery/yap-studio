"""Validate and stage review changes to the B-roll lane before publication."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import ValidationError
from yap_studio.feedback.batch import FeedbackError
from yap_studio.feedback.keeps import element_keep
from yap_studio.feedback.versions import VersionManifest
from yap_studio.project.lock import write_json_cas
from yap_studio.project.model import parse_anchor, parse_document
from yap_studio.receipt import sha256_file
from yap_studio.stages import cut, timeline

LANE = "timeline/broll.json"
KINDS = frozenset({"broll_remove", "broll_media", "broll_retime", "broll_place"})


def _fragment(*elements: dict) -> dict:
    return {"schema": "yap-studio/timeline@1", "elements": list(elements)}


def validate_operation(operation: dict, comment: str) -> None:
    """Use the fragment schema for anchors, media and placement field values."""
    parameter = f"operation {operation['id']} ({comment})"

    def invalid(detail: str) -> None:
        raise ValidationError(parameter, detail)

    kind = operation["kind"]
    required = {"id", "kind", "element"}
    optional = {"comment"}
    if kind == "broll_media":
        required.add("media")
    elif kind == "broll_retime":
        required.add("start")
        optional |= {"end", "hold_s"}
        if ("end" in operation) == ("hold_s" in operation):
            invalid("retime requires exactly one of end or hold_s")
    elif kind == "broll_place":
        optional |= {"placement", "box", "fit", "motion", "crop"}
        if not optional.difference({"comment"}).intersection(operation):
            invalid("place requires at least one placement field")
    elif kind != "broll_remove":
        invalid(f"unknown kind {kind!r}")
    if not required <= operation.keys() <= required | optional:
        invalid("unexpected or missing fields")
    if not isinstance(operation["element"], str) or not operation["element"].strip():
        invalid("element must be a nonempty string")
    if "comment" in operation and operation["comment"] != comment:
        invalid("comment must match the containing item")
    sample = {"id": "broll-sample", "kind": "insert", "start": {"time": 0},
              "hold_s": 1, "media": {"path": "sample.mov", "sha256": "0" * 64}}
    try:
        if kind == "broll_media":
            sample["media"] = operation["media"]
        elif kind == "broll_retime":
            parse_anchor(operation["start"])
            sample.pop("hold_s")
            sample.update({key: operation[key] for key in ("start", "end", "hold_s")
                           if key in operation})
        elif kind == "broll_place":
            fields = {key: operation[key] for key in ("placement", "box", "fit", "motion")
                      if key in operation}
            if {"placement", "box", "motion"}.intersection(fields):
                sample.update(kind="overlay", placement="headroom")
                sample.update(fields)
                if "box" in fields and "placement" not in fields:
                    sample["placement"] = "box"
                if "crop" in operation:
                    invalid("crop is only available for inserts")
            else:
                sample["media"].update(fields)
                if "crop" in operation:
                    if operation.get("fit") != "crop":
                        invalid("crop requires fit: crop")
                    sample["media"]["crop"] = operation["crop"]
        parse_document("timeline-fragment", _fragment(sample), lane="broll")
    except ValidationError as exc:
        invalid(str(exc))


def require_hash(root: Path, expected: str | None, *, after: str | None = None) -> None:
    """Refuse unrelated lane changes, including changes made during recovery."""
    source = root / LANE
    current = sha256_file(source) if source.is_file() else None
    if current != expected and (after is None or current != after):
        raise FeedbackError(f"{LANE} changed after B-roll planning.",
                            suggested_action="Rebuild the batch and recompute its operations.")


def saved_hash(base: VersionManifest) -> str | None:
    """Return the reviewed lane identity recorded in the version manifest."""
    snapshot = base.snapshots.get("timeline") or {}
    return snapshot.get("lanes", {}).get("broll", {}).get("sha256")


def decline_kept(document: dict, constraints: list[dict]) -> tuple[dict, list[dict]]:
    """Decline a conflicting comment as a unit, naming its protecting keep."""
    accepted = copy.deepcopy(document)
    accepted["items"], declined = [], []
    for entry in document["items"]:
        conflict = next((keep for operation in entry["operations"]
                         if operation["kind"] in KINDS
                         and (keep := element_keep(constraints, operation["element"]))), None)
        if conflict is None:
            accepted["items"].append(copy.deepcopy(entry))
            continue
        ref = conflict["comment"]
        summary = f"Declined: protected by keep comment {ref['version']}:{ref['id']}."
        declined.extend({"operation": operation["id"], "comment": entry["comment"],
                         "kind": operation["kind"], "summary": summary, "segments": [],
                         "words": [], "disposition": "declined", "keep": copy.deepcopy(ref)}
                        for operation in entry["operations"])
    return accepted, declined


def _change(element: dict, operation: dict) -> None:
    kind = operation["kind"]
    if kind == "broll_media":
        element["media"] = copy.deepcopy(operation["media"])
    elif kind == "broll_retime":
        element.pop("end", None)
        element.pop("hold_s", None)
        element.update({key: copy.deepcopy(operation[key]) for key in ("start", "end", "hold_s")
                        if key in operation})
    elif element["kind"] == "overlay":
        if "crop" in operation:
            raise ValidationError(operation["id"], "overlays cannot crop")
        if operation.get("placement") not in (None, "box"):
            element.pop("box", None)
        if "box" in operation and "placement" not in operation:
            element["placement"] = "box"
        element.update({key: copy.deepcopy(operation[key])
                        for key in ("placement", "box", "fit", "motion") if key in operation})
    else:
        if {"placement", "box", "motion"}.intersection(operation):
            raise ValidationError(operation["id"], "inserts accept only fit and crop")
        element["media"].update({key: copy.deepcopy(operation[key]) for key in ("fit", "crop")
                                 if key in operation})
        if operation.get("fit") not in (None, "crop"):
            element["media"].pop("crop", None)
    parse_document("timeline-fragment", _fragment(element), lane="broll")


def require_targets(root: Path, operations: tuple[dict, ...]) -> None:
    """Keep every operation outside the supported lane behind the named refusal."""
    from yap_studio.feedback.edits import NeedsTimelineError

    source = root / LANE
    lane = json.loads(source.read_bytes()) if source.is_file() else {"elements": []}
    by_id = {element["id"]: element for element in lane["elements"]}
    for operation in operations:
        element = by_id.get(operation["element"])
        if (element is None or element.get("lane", "broll") != "broll"
                or element["kind"] not in {"insert", "overlay"}):
            raise NeedsTimelineError(operation["id"], operation["comment"])


def _write(source: Path, data: dict) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(data) + "\n", encoding="utf-8")


def _shadow(root: Path, shadow: Path) -> None:
    """Stage small writable inputs while retaining read-only links to media."""
    for source in root.iterdir():
        if source.name in {".git", ".codegraph", ".yap", "feedback", "versions"}:
            continue
        target = shadow / source.name
        if source.name in {"timeline", "cut", "ingest"} and source.is_dir():
            _stage_tree(source, target)
        elif source.is_file() and source.suffix == ".json":
            shutil.copyfile(source, target)
        else:
            target.symlink_to(source, target_is_directory=source.is_dir())


def _stage_tree(source: Path, target: Path) -> None:
    target.mkdir()
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            _stage_tree(child, destination)
        elif child.suffix == ".json":
            shutil.copyfile(child, destination)
        else:
            try:
                os.link(child.resolve(), destination)
            except OSError:
                shutil.copyfile(child, destination)


def _project_cut(shadow: Path, edit: dict, actions: tuple) -> None:
    """Resolve the proposed story without rendering or touching project receipts."""
    _write(shadow / "edit.json", edit)
    source = shadow / "removals.json"
    removals = json.loads(source.read_bytes()) if source.exists() else {
        "schema": "yap-studio/removals@1", "removals": [],
    }
    for action in actions:
        if action.action == "reject":
            next(entry for entry in removals["removals"]
                 if entry["id"] == action.removal_id)["status"] = "rejected"
        else:
            removals["removals"].append({"id": f"feedback:{action.operation}",
                                         "kind": "manual", "status": "applied",
                                         **{key: asdict(action)[key]
                                            for key in ("take", "start", "end")}})
    if actions:
        _write(source, removals)
    plan = cut.resolve_cut(shadow)
    _write(shadow / "cut/cut-words.json", cut._cut_words(plan))
    _write(shadow / "cut/spans.json", cut._spans(plan, shadow / "cut"))
    receipt = json.loads((shadow / "cut/receipt.json").read_bytes())
    receipt.update(inputs=cut.cut_inputs(plan.context), plan=plan.to_dict())
    for output in receipt["outputs"]:
        output["sha256"] = sha256_file(shadow / output["path"])
    _write(shadow / "cut/receipt.json", receipt)


def plan_broll(root: Path, operations: tuple[dict, ...], *, edit: dict,
               story_changed: bool, removal_actions: tuple) -> dict:
    """Check a complete candidate through the timeline stage in an isolated snapshot."""
    from yap_studio.feedback.edits import NeedsTimelineError

    payload = (root / LANE).read_bytes() if (root / LANE).is_file() else None
    lane = json.loads(payload) if payload is not None else {
        "schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [],
    }
    candidate = copy.deepcopy(lane)
    changed, removed, changes = set(), [], []
    for operation in operations:
        element = next((item for item in candidate["elements"]
                        if item["id"] == operation["element"]), None)
        if (element is None or element.get("lane", "broll") != "broll"
                or element["kind"] not in {"insert", "overlay"}):
            raise NeedsTimelineError(operation["id"], operation["comment"])
        if operation["kind"] == "broll_remove":
            candidate["elements"].remove(element)
            removed.append(element["id"])
            changed.discard(element["id"])
        else:
            _change(element, operation)
            changed.add(element["id"])
        changes.append({"operation": operation["id"], "comment": operation["comment"],
                        "kind": operation["kind"], "segments": [], "words": [],
                        "element": element["id"],
                        "summary": f"{operation['kind']}: {element['kind']} {element['id']}."})
    verify_media(root, candidate)
    with tempfile.TemporaryDirectory(prefix="yap-feedback-broll-") as directory:
        shadow = Path(directory)
        _shadow(root, shadow)
        if story_changed or removal_actions:
            _project_cut(shadow, edit, removal_actions)
        if removed:
            timeline.remove_timeline(shadow, lane="broll", ids=tuple(removed))
        if changed:
            fragment = shadow / "feedback-fragment.json"
            _write(fragment, _fragment(*(element for element in candidate["elements"]
                                        if element["id"] in changed)))
            result = timeline.add_timeline(shadow, fragment, lane="broll")
        else:
            result = timeline.resolve_timeline(shadow)
        after = json.loads((shadow / LANE).read_bytes()) if result.valid else candidate
        findings = [{**finding.to_dict(), "suggested_action": finding.suggested_action.replace(
            shlex.quote(str(shadow)), shlex.quote(str(root)),
        )} for finding in result.findings]
    return {"sha256_before": hashlib.sha256(payload).hexdigest() if payload else None,
            "sha256_after": hashlib.sha256(_payload(after)).hexdigest(), "lane": after,
            "operations": list(operations), "changes": changes, "valid": result.valid,
            "findings": findings, "done": False}


def _payload(lane: dict) -> bytes:
    return (json.dumps(lane, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def verify_media(root: Path, lane: dict) -> None:
    """Hash media directly so a cached identity cannot authorize a changed source."""
    for element in lane["elements"]:
        media = element["media"]
        source = paths.validate_input(root / media["path"])
        if sha256_file(source) != media["sha256"]:
            raise FeedbackError(
                f"B-roll element {element['id']}: media sha256 does not match file.")


def publish_broll(root: Path, plan: dict) -> None:
    """Publish the stage-validated lane once under the caller's project lock."""
    require_hash(root, plan["sha256_before"], after=plan["sha256_after"])
    verify_media(root, plan["lane"])
    if sha256_file(root / LANE) != plan["sha256_after"]:
        write_json_cas(root / LANE, plan["lane"], base_sha256=plan["sha256_before"])
    plan["done"] = True
