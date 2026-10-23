"""Build durable feedback batches from exact render and comment identities."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from yap_studio.errors import InputError, ValidationError, YapError
from yap_studio.feedback import pins
from yap_studio.feedback.audio import media_envelope
from yap_studio.feedback.keeps import keep_constraint
from yap_studio.feedback.locate import locate
from yap_studio.feedback.versions import (
    LayerWindow,
    VersionError,
    VersionManifest,
    _receipt_pieces,
    current_version,
    find_version_by_media,
    load_version,
    load_versions,
    parse_layers,
    verify_cut,
    version_cut_words,
    version_envelope,
    version_layers,
    version_pieces,
)
from yap_studio.paths import atomic_write_text
from yap_studio.project import load_project
from yap_studio.project.lock import project_lock
from yap_studio.project.model import load_cut_words
from yap_studio.project.receipts import read_receipt
from yap_studio.receipt import sha256_file
from yap_studio.review.export import comment_from_dict
from yap_studio.review.model import TARGETS, Comment

BATCH_SCHEMA = "yap-studio/feedback@1"
_BATCH_ID = re.compile(r"b([1-9][0-9]*)\Z")
_DISPOSITIONS = {"open", "applied", "needs_clarification", "declined", "awaiting_acceptance"}


class FeedbackError(YapError):
    """Refuse feedback that cannot be safely attached to this project."""

    def __init__(self, reason: str, *, suggested_action: str =
                 "Save versions with yap-studio versions save before reviewing.") -> None:
        super().__init__(reason, error_type="feedback_error", code="feedback_refused",
                         suggested_action=suggested_action)


class StaleBatchError(FeedbackError):
    """Refuse operations computed from inputs that have since changed."""

    def __init__(self, changed: list[str]) -> None:
        self.changed = changed
        super().__init__(f"Batch inputs changed: {', '.join(changed)}.",
                         suggested_action="Rebuild the batch and recompute its operations.")
        self.code = "stale_batch"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, UnicodeError, ValueError) as exc:
        raise InputError(str(path), "cannot read a valid JSON object") from exc


def _comment(raw: dict, *, legacy: bool) -> Comment:
    comment = comment_from_dict(raw, legacy=legacy)
    for name in ("id", "text", "created_at", "updated_at"):
        if not isinstance(getattr(comment, name), str) or not getattr(comment, name).strip():
            raise ValueError(f"comment {name} must be a nonempty string")
    for name in ("start_s", "end_s"):
        value = getattr(comment, name)
        if value is None and name == "end_s":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"comment {name} must be finite and non-negative")
    for name in ("start_frame", "end_frame"):
        value = getattr(comment, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"comment {name} must be a non-negative integer or null")
    if comment.end_s is not None and comment.end_s <= comment.start_s:
        raise ValueError("comment end_s must be greater than start_s")
    if comment.end_frame is not None and (
        comment.start_frame is None or comment.end_frame <= comment.start_frame
    ):
        raise ValueError("comment end_frame must be greater than start_frame")
    return comment


def _export(data: dict) -> tuple[list[Comment], Fraction | None]:
    try:
        if (not isinstance(data, dict) or data.get("format") != "yap-studio-feedback"
                or type(data.get("version")) is not int or data["version"] not in (1, 2)):
            raise ValueError("expected yap-studio-feedback version 1 or 2")
        media = data["media"]
        if not isinstance(media, dict):
            raise ValueError("media must be an object")
        if not isinstance(media["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                 media["sha256"]):
            raise ValueError("media.sha256 must be a SHA256 digest")
        if not isinstance(media["name"], str) or not media["name"].strip():
            raise ValueError("media.name must be a nonempty string")
        duration = media["duration_s"]
        if duration is not None and (
            type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
        ):
            raise ValueError("media.duration_s must be finite and non-negative or null")
        rate = media["fps"]
        if rate is not None and not isinstance(rate, str):
            raise ValueError("media.fps must be a fraction string or null")
        fps = Fraction(rate) if rate is not None else None
        if fps is not None and fps <= 0:
            raise ValueError("media.fps must be positive")
        if data["reviewer"] is not None and not isinstance(data["reviewer"], str):
            raise ValueError("reviewer must be a string or null")
        if not isinstance(data["exported_at"], str):
            raise ValueError("exported_at must be a timestamp")
        if not isinstance(data["comments"], list):
            raise ValueError("comments must be an array")
        comments = [_comment(raw, legacy=data["version"] == 1) for raw in data["comments"]]
        if len({comment.id for comment in comments}) != len(comments):
            raise ValueError("comment ids must be unique within a version")
        if fps is not None and any(comment.start_frame is None for comment in comments):
            raise ValueError("constant rate comments require a start frame")
        return comments, fps
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise InputError("review export", str(exc)) from exc


def layers_for(
    project_dir: Path, base: VersionManifest | None, layers_path: Path | None,
) -> tuple[LayerWindow, ...] | None:
    """Load the single layer seam from the saved version or an explicit file."""
    saved = version_layers(base) if base is not None else None
    if saved is not None and layers_path is not None:
        raise FeedbackError("Both the base version and --layers supply layer windows.",
                            suggested_action="Use the base version's saved layers.")
    if layers_path is None:
        return saved
    return parse_layers(_read_json(Path(layers_path)), source=str(layers_path))


def _inputs(root: Path, *, require_edit: bool = False) -> dict:
    paths = [root / "edit.json", root / "removals.json",
             *sorted((root / "timeline").glob("*.json"))]
    if require_edit and not paths[0].is_file():
        raise InputError("edit.json")
    try:
        return {path.relative_to(root).as_posix(): sha256_file(path) if path.is_file() else None
                for path in paths}
    except OSError as exc:
        raise InputError("project inputs", "cannot hash current inputs") from exc


def check_inputs(project_dir: Path, batch: dict) -> None:
    """Name every changed, removed, or newly added input before applying operations."""
    changed = _changed_inputs(current_inputs(project_dir), batch["expected_inputs"])
    if changed:
        raise StaleBatchError(changed)


def current_inputs(project_dir: Path) -> dict:
    """Capture the current edit, removals and lane hashes before applying an operation."""
    return _inputs(Path(project_dir).resolve())


def _changed_inputs(current: dict, expected: dict) -> list[str]:
    return sorted(name for name in current.keys() | expected.keys()
                  if name not in current or name not in expected
                  or current[name] != expected[name])


def _supersedes(constraint: dict, item: dict) -> bool:
    """Release a carried keep only for a fresh fix of the same target and identity."""
    comment = item["comment"]
    target = constraint.get("target") or "story"
    if (item["carried"] or item.get("disposition") == "needs_clarification"
            or comment["kind"] != "fix"
            or (comment["target"] or "story") != target):
        return False
    if target == "video":
        return True
    location = item["location"]
    if location is None:
        return False
    if target in ("story", "speech"):
        kept = {(word["id"], word["segment"]) for word in constraint["words"]}
        if kept.intersection((word["id"], word["segment"]) for word in location["words"]):
            return True
        if (constraint.get("pause", False) and location["status"] == "silence"
                and len(constraint["words"]) == 2):
            neighbours = (location["before"], location["after"])
            return all(neighbour is not None and (
                neighbour["id"], neighbour["segment"]
            ) == (word["id"], word["segment"]) for neighbour, word in zip(
                neighbours, constraint["words"], strict=True
            ))
        return False
    return bool({layer["id"] for layer in constraint["layers"]}.intersection(
        layer["id"] for layer in location["layers"]
    ))


def _batch_numbers(root: Path) -> list[int]:
    try:
        return [int(match[1]) for path in (root / "feedback").glob("b*.json")
                if (match := _BATCH_ID.fullmatch(path.stem)) is not None]
    except OSError as exc:
        raise InputError("feedback", "cannot list feedback batches") from exc


def batch_ids(project_dir: Path) -> list[str]:
    """List a project's feedback batches in numeric order."""
    return [f"b{number}" for number in sorted(_batch_numbers(Path(project_dir)))]


def _batch_path(root: Path, batch_id: str) -> Path:
    if not isinstance(batch_id, str) or _BATCH_ID.fullmatch(batch_id) is None:
        raise ValidationError("batch_id", "must be b followed by a positive integer")
    return root / "feedback" / f"{batch_id}.json"


def load_batch(project_dir: Path, batch_id: str) -> dict:
    """Read a numbered feedback batch without changing it."""
    path = _batch_path(Path(project_dir), batch_id)
    data = _read_json(path)
    if (data.get("schema") != BATCH_SCHEMA or data.get("id") != batch_id
            or not isinstance(data.get("items"), list)
            or not isinstance(data.get("constraints"), list)
            or not isinstance(data.get("expected_inputs"), dict)):
        raise InputError(str(path), "not a feedback batch")
    for item in data["items"]:
        item.setdefault("clarification", None)
    return data


def latest_batch(project_dir: Path) -> dict | None:
    """Read the highest existing batch number, or return None."""
    numbers = _batch_numbers(Path(project_dir))
    return load_batch(project_dir, f"b{max(numbers)}") if numbers else None


def _write_batch(root: Path, batch: dict) -> None:
    atomic_write_text(_batch_path(root, batch["id"]),
                      json.dumps(batch, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _series(data: dict) -> tuple[dict | None, dict[int, dict]]:
    series = data.get("series") if data["version"] == 2 else None
    if series is None:
        return None, {}
    try:
        if not isinstance(series, dict) or not isinstance(series["id"], str):
            raise ValueError("series must have an id")
        if type(series["version"]) is not int or series["version"] < 1:
            raise ValueError("series.version must be a positive integer")
        if not isinstance(series["versions"], list) or not isinstance(series["threads"], list):
            raise ValueError("series versions and threads must be arrays")
        entries = {}
        for entry in series["versions"]:
            number = entry["number"]
            if type(number) is not int or number < 1 or number in entries:
                raise ValueError("series version numbers must be unique positive integers")
            if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                     entry["sha256"]):
                raise ValueError("series version needs a media hash")
            if entry.get("version_id") is not None and not isinstance(entry["version_id"], str):
                raise ValueError("series version_id must be a string or null")
            entries[number] = entry
        if entries[series["version"]]["sha256"] != data["media"]["sha256"]:
            raise ValueError("series current version disagrees with export media")
        return series, entries
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError("review export", f"invalid series: {exc}") from exc


def build_batch(
    project_dir: Path, export: dict, *, layers: Path | None = None,
    now: Callable[[], datetime] | None = None,
    origin: dict | None = None, answers: Sequence[dict] | None = None,
) -> dict:
    """Locate review comments under the project lock and atomically save a new batch."""
    from yap_studio.feedback.clarify import automatic_clarification

    comments, fps = _export(export)
    series, entries = _series(export)
    root = Path(project_dir).resolve()
    with project_lock(root):
        numbers = _batch_numbers(root)
        history = [load_batch(root, f"b{number}") for number in sorted(numbers)]
        if origin is not None:
            for existing in history:
                if existing.get("origin") == origin:
                    return existing
        selected_answers = None
        if answers is not None:
            selected_answers = set()
            for answer in answers:
                try:
                    if not isinstance(answer, dict) or set(answer) != {"batch", "comment"}:
                        raise ValueError("invalid answer")
                    ref = answer["comment"]
                    if (not isinstance(ref, dict) or set(ref) != {"version", "id"}
                            or any(not isinstance(value, str) for value in (
                                answer["batch"], ref["version"], ref["id"],
                            ))):
                        raise ValueError("invalid answer reference")
                    key = (answer["batch"], ref["version"], ref["id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise FeedbackError("Invalid answer reference.") from exc
                source = next((item for old in history if old["id"] == key[0]
                               for item in old["items"]
                               if (item["comment"]["version"], item["comment"]["id"])
                               == key[1:]), None)
                question = source.get("clarification") if source else None
                if (question is None or question["answered_at"] is None
                        or question["carried_to"] is not None):
                    raise FeedbackError("This answer is no longer answered or was already carried.")
                selected_answers.add(key)
        project = load_project(root / "project.json")
        base = find_version_by_media(root, export["media"]["sha256"])
        versions = load_versions(root)
        current = current_version(root)
        if base is not None and current is not None and base.id != current.id:
            raise FeedbackError(f"{base.id} is not the current version ({current.id}).",
                                suggested_action="Review the current saved version.")
        if base is not None:
            words, pieces = version_cut_words(base), version_pieces(base)
            envelope = version_envelope(base)
        else:
            receipt = read_receipt(root / "cut")
            outputs = receipt.get("outputs", []) if receipt is not None else []
            if not isinstance(outputs, list):
                raise InputError("cut/receipt.json", "outputs must be an array")
            hashes = [item.get("sha256") for item in outputs
                      if isinstance(item, dict) and item.get("path") == "cut/cut.mp4"]
            if hashes != [export["media"]["sha256"]]:
                raise FeedbackError("The review is of a render this project does not have.")
            try:
                verify_cut(root, receipt)
            except VersionError as exc:
                raise FeedbackError(str(exc), suggested_action=exc.suggested_action) from exc
            words = load_cut_words(root / "cut" / "cut-words.json")
            _read_json(root / "cut" / "spans.json")
            pieces = _receipt_pieces(receipt, root / "cut" / "receipt.json")
            envelope = None
        envelope_measured = False
        layer_windows = layers_for(root, base, layers)
        duration = export["media"]["duration_s"]
        items, refs = [], set()
        batch_id = f"b{max(numbers, default=0) + 1}"
        pin_context = pins.PinContext(root, versions, current, current_cut=base is None,
                                      cut_layers=layer_windows)
        consumed = []
        unanswered_refs = {(item["comment"]["version"], item["comment"]["id"])
                           for batch in history for item in batch["items"]
                           if (question := item.get("clarification")) is not None
                           and question["carried_to"] is None
                           and (question["answered_at"] is None or selected_answers is not None
                                and (batch["id"], item["comment"]["version"],
                                     item["comment"]["id"]) not in selected_answers)}
        sent_answers = {(item["comment"]["version"], item["comment"]["id"],
                         item["comment"]["revision"])
                        for batch in history for item in batch["items"]
                        if (question := item.get("clarification")) is not None
                        and question["answered_at"] is not None
                        and question["carried_to"] is not None}

        def add(comment: Comment, version_id: str | None, carried: bool, location,
                unlocated: str | None = None, *, own_envelope: dict | None = None,
                own_inputs: dict | None = None, clarification: dict | None = None) -> None:
            nonlocal envelope, envelope_measured
            ref = (version_id, comment.id)
            if ref in refs:
                return
            refs.add(ref)
            clarification = (copy.deepcopy(clarification) if clarification is not None else
                             automatic_clarification(comment, version_id, location))
            needs_answer = clarification is not None and clarification["answered_at"] is None
            if (base is None and version_id is None and comment.kind == "keep"
                    and comment.target == "speech" and location is not None
                    and location.status != "outside" and not needs_answer):
                if not envelope_measured:
                    envelope = media_envelope(root / "cut" / "cut.mp4")
                    envelope_measured = True
                own_envelope = envelope
            constraint = keep_constraint(comment, location, version=version_id,
                                         envelope=own_envelope, inputs=own_inputs) if (
                not needs_answer and comment.kind == "keep" and location is not None
                and location.status != "outside"
            ) else None
            items.append({
                "comment": {"version": version_id, "id": comment.id,
                            "revision": comment.updated_at, "kind": comment.kind,
                            "target": comment.target, "text": comment.text,
                            "request": copy.deepcopy(comment.request),
                            "start_frame": comment.start_frame, "end_frame": comment.end_frame,
                            "start_s": comment.start_s, "end_s": comment.end_s},
                "request": copy.deepcopy(comment.request),
                "carried": carried, "location": location.to_dict() if location else None,
                "clarification": clarification,
                "unlocated": unlocated, "constraint": constraint,
                "disposition": "needs_clarification" if needs_answer else "open",
                "operations": [], "evidence": None,
            })
            if comment.pin is not None:
                items[-1]["comment"]["pin"] = dict(comment.pin)
                items[-1]["pin"] = pins.pin_spot(
                    comment, batch_id=batch_id, version_id=version_id, context=pin_context,
                )

        for old_batch in history:
            for item in old_batch["items"]:
                clarification = item.get("clarification")
                if (clarification is None or clarification["answered_at"] is None
                        or clarification["carried_to"] is not None):
                    continue
                raw = item["comment"]
                if (selected_answers is not None
                        and (old_batch["id"], raw["version"], raw["id"]) not in selected_answers):
                    continue
                comment = _comment({
                    **{key: value for key, value in raw.items()
                       if key not in ("version", "revision")},
                    "created_at": raw["revision"], "updated_at": raw["revision"],
                }, legacy=False)
                if comment.target is None and clarification["option"] in TARGETS:
                    comment = replace(comment, target=clarification["option"])
                saved = next((version for version in versions if version.id == raw["version"]),
                             None)
                # Both copies name their destination so the answer is consumed only once.
                carried_question = {**clarification, "carried_to": batch_id}
                if saved is None:
                    add(comment, raw["version"], True, None,
                        "The comment's version is not saved in this project.",
                        clarification=carried_question)
                else:
                    own_pieces = version_pieces(saved)
                    add(comment, saved.id, True, locate(
                        comment, fps=Fraction(saved.media["fps"])
                        if comment.start_frame is not None else None,
                        duration_s=saved.media["duration_s"], words=version_cut_words(saved),
                        pieces=own_pieces, layers=layers_for(root, saved, None),
                    ), own_envelope=version_envelope(saved), own_inputs=own_pieces.get("inputs"),
                        clarification=carried_question)
                consumed.append((old_batch, clarification))

        for comment in comments:
            add(comment, base.id if base else None, False,
                locate(comment, fps=fps, duration_s=duration, words=words, pieces=pieces,
                       layers=layer_windows), own_envelope=envelope,
                own_inputs=pieces.get("inputs"))
        for thread in series["threads"] if series else ():
            try:
                if thread["state"] not in ("open", "needs_clarification"):
                    continue
                number = thread["version"]
                if type(number) is not int or number not in entries:
                    raise ValueError("thread must name a series version")
                comment = _comment({key: value for key, value in thread.items()
                                    if key not in ("version", "reply", "state")}, legacy=False)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise InputError("review export", f"invalid thread: {exc}") from exc
            entry = entries[number]
            saved = next((version for version in versions if version.id == entry.get("version_id")
                          and version.media["sha256"] == entry["sha256"]), None)
            if saved is None:
                saved = find_version_by_media(root, entry["sha256"])
            version_ref = (saved.id if saved is not None else
                           entry.get("version_id") or f"series:{series['id']}:{number}")
            # Waiting threads, including keeps, add neither an item nor a constraint.
            if (version_ref, comment.id) in unanswered_refs:
                continue
            if saved is None:
                if (version_ref, comment.id, comment.updated_at) in sent_answers:
                    continue
                add(comment, version_ref, True, None, "The comment's version is not saved in "
                    "this project; save versions before reviewing.")
            else:
                if (saved.id, comment.id, comment.updated_at) in sent_answers:
                    continue
                add(comment, saved.id, True, locate(
                    comment, fps=Fraction(saved.media["fps"]) if comment.start_frame is not None
                    else None,
                    duration_s=saved.media["duration_s"], words=version_cut_words(saved),
                    pieces=version_pieces(saved), layers=layers_for(root, saved, None),
                ), own_envelope=version_envelope(saved),
                    own_inputs=version_pieces(saved).get("inputs"))

        previous = history[-1] if history else None
        constraints = [item["constraint"] for item in items
                       if item["constraint"] is not None and not item["carried"]]
        constraint_refs = {(item["comment"]["version"], item["comment"]["id"])
                           for item in constraints}
        fresh_refs = set(constraint_refs)
        carried_constraints = [item["constraint"] for item in items
                               if item["constraint"] is not None and item["carried"]]
        carried_constraints.extend(previous["constraints"] if previous else ())
        released = copy.deepcopy(previous.get("released", [])) if previous else []
        released = [entry for entry in released if (
            entry["constraint"]["comment"]["version"], entry["constraint"]["comment"]["id"]
        ) not in fresh_refs]
        handled = set()
        for constraint in carried_constraints:
            ref = (constraint["comment"]["version"], constraint["comment"]["id"])
            # A waiting keep must not restore a constraint from an earlier batch.
            if ref in unanswered_refs or ref in fresh_refs or ref in handled:
                continue
            handled.add(ref)
            prior = [entry for entry in released if (
                entry["constraint"]["comment"]["version"], entry["constraint"]["comment"]["id"]
            ) == ref]
            if any(entry["constraint"]["comment"].get("revision")
                   == constraint["comment"].get("revision") for entry in prior):
                continue
            # A changed revision is a new keep from the creator.
            released = [entry for entry in released if entry not in prior]
            fix = next((item for item in items if _supersedes(constraint, item)), None)
            if fix is not None:
                released.append({"constraint": copy.deepcopy(constraint), "by": {
                    "version": fix["comment"]["version"], "id": fix["comment"]["id"]},
                    "batch": batch_id})
            elif ref not in constraint_refs:
                constraints.append(copy.deepcopy(constraint))
                constraint_refs.add(ref)
        value = now() if now is not None else datetime.now(UTC)
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        batch = {
            "schema": BATCH_SCHEMA, "id": batch_id,
            "project": project.name, "base": "version" if base else "cut",
            "base_version": base.id if base else None,
            "created_at": value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "review": {"media_sha256": export["media"]["sha256"],
                       "media_name": export["media"]["name"], "reviewer": export["reviewer"],
                       "exported_at": export["exported_at"],
                       "series_id": series["id"] if series else None},
            "expected_inputs": _inputs(root, require_edit=True), "items": items,
            "constraints": constraints, "released": released, "resulting_version": None,
        }
        if origin is not None:
            batch["origin"] = copy.deepcopy(origin)
        if _batch_path(root, batch["id"]).exists():
            raise FeedbackError(f"{batch['id']} already exists.")
        _write_batch(root, batch)
        for _old_batch, clarification in consumed:
            clarification["carried_to"] = batch_id
        for old_batch in history:
            if any(old_batch is source for source, _ in consumed):
                _write_batch(root, old_batch)
        return batch


def set_disposition(
    project_dir: Path, batch_id: str, comment_version: str | None, comment_id: str,
    disposition: str, *, operations: list[dict] | None = None, evidence=None,
    inputs_before: dict | None = None,
    _locked: bool = False,
) -> dict:
    """Record unique operations; internal transactions can supply their held project lock."""
    if not isinstance(disposition, str) or disposition not in _DISPOSITIONS:
        raise ValidationError("disposition", "unknown feedback disposition")
    try:
        json.dumps({"operations": operations, "evidence": evidence, "inputs_before": inputs_before},
                   allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("feedback", "operations and evidence must be valid JSON") from exc
    root = Path(project_dir).resolve()
    with nullcontext() if _locked else project_lock(root):
        batch = load_batch(root, batch_id)
        item = next((item for item in batch["items"]
                     if (item["comment"]["version"], item["comment"]["id"])
                     == (comment_version, comment_id)), None)
        if item is None:
            raise ValidationError("comment", "comment reference is not in this batch")
        if operations is not None:
            if not isinstance(operations, list):
                raise ValidationError("operations", "must be an array")
            ids = {operation["id"] for entry in batch["items"] for operation in entry["operations"]}
            for operation in operations:
                if (not isinstance(operation, dict) or not isinstance(operation.get("id"), str)
                        or not operation["id"].strip()):
                    raise ValidationError("operations", "each operation needs a nonempty string id")
                if operation["id"] in ids:
                    raise FeedbackError(f"operation {operation['id']} was already applied",
                                        suggested_action="Do not apply an operation twice.")
                ids.add(operation["id"])
        if operations or disposition == "applied":
            current = current_inputs(root)
            if inputs_before is not None and not isinstance(inputs_before, dict):
                raise ValidationError("inputs_before", "must be an input hash object")
            before = current if inputs_before is None else inputs_before
            changed = _changed_inputs(before, batch["expected_inputs"])
            if changed:
                raise StaleBatchError(changed)
            item["operations"].extend({**copy.deepcopy(operation),
                                       "inputs_before": copy.deepcopy(batch["expected_inputs"]),
                                       "inputs_after": copy.deepcopy(current)}
                                      for operation in operations or ())
            batch["expected_inputs"] = current
        item["disposition"] = disposition
        if evidence is not None:
            item["evidence"] = copy.deepcopy(evidence)
        _write_batch(root, batch)
        return batch


def set_resulting_version(project_dir: Path, batch_id: str, version_id: str) -> dict:
    """Attach one saved outcome that is newer than this batch's reviewed base."""
    root = Path(project_dir).resolve()
    with project_lock(root):
        batch = load_batch(root, batch_id)
        if batch["resulting_version"] is not None:
            raise FeedbackError("The resulting version can be set only once.",
                                suggested_action="Build a new batch for further feedback.")
        version = load_version(root, version_id)
        if ((batch["base_version"] is not None
             and version.number <= load_version(root, batch["base_version"]).number)
                or (batch["base_version"] is None and version.created_at < batch["created_at"])
                or version.media["sha256"] == batch["review"]["media_sha256"]):
            raise FeedbackError("The resulting version must be newer than the base.",
                                suggested_action="Save a changed version after applying feedback.")
        batch["resulting_version"] = version.id
        _write_batch(root, batch)
        return batch
