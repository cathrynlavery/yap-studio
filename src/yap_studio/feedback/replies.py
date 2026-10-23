"""Attach agent replies backed by measurements from the resulting saved version."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback.batch import (
    FeedbackError,
    latest_batch,
    load_batch,
    set_resulting_version,
)
from yap_studio.feedback.keeps import check_version
from yap_studio.feedback.moments import map_moment, occurrences
from yap_studio.feedback.versions import (
    VersionManifest,
    load_version,
    load_versions,
    version_cut_words,
    version_layers,
)
from yap_studio.receipt import sha256_file
from yap_studio.review.model import frame_at, frame_start
from yap_studio.review.series import (
    REPLY_STATUSES,
    Reply,
    SeriesEntry,
    validate_evidence,
    validate_reply_text,
)
from yap_studio.review.store import ReviewStore

_MAP_STATUSES = ("removed", "ambiguous", "partial", "moved", "mapped", "time")


def _fps(version: VersionManifest) -> Fraction | None:
    value = version.media.get("fps")
    return Fraction(value) if value is not None else None


def _identities(words: list[dict]) -> list[tuple[str, str]]:
    return [(word["id"], word["segment"]) for word in words]


def _overlaps(start: float, end: float, a: float, b: float) -> bool:
    return a <= start < b if start == end else a < end and b > start


def _keep_results(project_dir: Path, batch: dict, version: VersionManifest) -> dict:
    report = check_version(project_dir, batch_id=batch["id"], version_id=version.id)
    return {(result.comment["version"], result.comment["id"]):
            {"status": result.status, "detail": result.detail} for result in report.results}


def _measure_moment(project_dir: Path, item: dict, version: VersionManifest) -> dict:
    comment, location = item["comment"], item.get("location")
    own = next((saved for saved in load_versions(project_dir)
                if saved.id == comment["version"]), None)
    if own is None:
        return {"moment": None, "change": None,
                "unmapped": "This comment's version is not saved in the project."}
    if location is None:
        return {"moment": None, "change": None,
                "unmapped": item.get("unlocated") or "This comment has no located moment."}
    if location["status"] == "whole_video":
        return {"moment": None, "change": None, "unmapped": "whole video"}

    own_words, next_words = version_cut_words(own), version_cut_words(version)
    request = comment.get("request")
    requested = request["words"] if request and request["type"] == "cut" else None
    requested_ids = set(_identities(requested)) if requested else set()
    cut_words = [word for word in occurrences(own_words)
                 if (word.id, word.segment) in requested_ids]
    start = location["start_s"]
    last = start
    if comment["end_frame"] is not None and _fps(own) is not None:
        last = float(frame_start(comment["end_frame"] - 1, _fps(own)))
    elif comment["end_s"] is not None:
        last = max(start, math.nextafter(comment["end_s"], -math.inf))
    if cut_words:
        start = cut_words[0].start
        last = math.nextafter(cut_words[-1].end, -math.inf)
    maps = [map_moment(own_words, next_words, time,
                       target_duration=version.media["duration_s"]) for time in (start, last)]
    mapped_start = maps[0].time_s
    mapped_last = max(mapped_start, maps[1].time_s)
    fps = _fps(version)
    tolerance = float(1 / fps) if fps is not None else 1 / 30
    moment = {
        "status": min((maps[0].status,
                       "partial" if maps[1].status == "removed" else maps[1].status),
                      key=_MAP_STATUSES.index),
        "start_s": float(mapped_start),
        "end_s": float(mapped_last + tolerance if fps is not None else mapped_last),
        "start_frame": frame_at(mapped_start, fps) if fps is not None else None,
        "end_frame": frame_at(mapped_last, fps) + 1 if fps is not None else None,
    }
    silence = location["status"] == "silence"
    before = ([word for word in (location["before"], location["after"]) if word is not None]
              if silence else location["words"])
    if cut_words:
        before = [asdict(word) for word in cut_words]
        requested_maps = [map_moment(own_words, next_words, word.start,
                                    target_duration=version.media["duration_s"])
                          for word in cut_words]
        removed = sum(mapping.status == "removed" or not next_words.words
                      for mapping in requested_maps)
        if removed:
            moment["status"] = "removed" if removed == len(cut_words) else "partial"
        maps = requested_maps
    targets = [asdict(word) for word in occurrences(next_words)]
    if silence:
        left = next((word for word in reversed(targets) if word["end"] <= mapped_start), None)
        right = next((word for word in targets if word["start"] >= mapped_start), None)
        after = [word for word in (left, right) if word is not None]
    elif cut_words:
        after = [asdict(mapping.target) for mapping in maps
                 if mapping.status in {"mapped", "moved"} and mapping.target is not None]
        after.sort(key=targets.index)
    else:
        after = [word for word in targets if _overlaps(
            moment["start_s"], moment["end_s"], word["start"], word["end"]
        )]
    old_ids, new_ids = _identities(before), _identities(after)
    differences = set()
    if (any(mapping.status == "removed" for mapping in maps)
            or cut_words and moment["status"] in {"removed", "partial"}):
        differences.add("removed")
    if Counter(old_ids) != Counter(new_ids):
        differences.add("words")
    elif old_ids != new_ids:
        differences.add("order")
    else:
        if any(a["text"] != b["text"] for a, b in zip(before, after, strict=True)):
            differences.add("words")
        if (any(abs((a["end"] - a["start"]) - (b["end"] - b["start"])) > tolerance + 1e-9
                for a, b in zip(before, after, strict=True))
                or any(abs((b["start"] - a["end"]) - (d["start"] - c["end"])) > tolerance + 1e-9
                       for a, b, c, d in zip(before, before[1:], after, after[1:], strict=False))):
            differences.add("timing")
    if silence and len(before) == 2:
        target_ids = _identities(targets)
        pairs = [(a, b) for index, (a, b) in enumerate(zip(targets, targets[1:], strict=False))
                 if target_ids[index:index + 2] == old_ids]
        old_gap = before[1]["start"] - before[0]["end"]
        if not pairs or all(abs(b["start"] - a["end"] - old_gap) > tolerance + 1e-9
                            for a, b in pairs):
            differences.add("pause")
    own_layers, next_layers = version_layers(own), version_layers(version)
    if own_layers is None and own.media["source"] == "cut/cut.mp4":
        own_layers = ()
    if own_layers is not None and next_layers is not None:
        old_offsets = {(layer.id, layer.kind, layer.text):
                       (layer.start - location["start_s"], layer.end - location["start_s"])
                       for layer in own_layers if _overlaps(
                           location["start_s"], location["end_s"], layer.start, layer.end
                       )}
        new_offsets = {(layer.id, layer.kind, layer.text):
                       (layer.start - moment["start_s"], layer.end - moment["start_s"])
                       for layer in next_layers if _overlaps(
                           moment["start_s"], moment["end_s"], layer.start, layer.end
                       )}
        if old_offsets.keys() != new_offsets.keys() or any(
            abs(a - b) > tolerance + 1e-9 for key in old_offsets
            for a, b in zip(old_offsets[key], new_offsets[key], strict=True)
        ):
            differences.add("layers")
    return {"moment": moment, "unmapped": None, "change": {
        "before": [{key: word[key] for key in ("id", "segment", "text")} for word in before[:40]],
        "after": [{key: word[key] for key in ("id", "segment", "text")} for word in after[:40]],
        "differences": sorted(differences),
    }}


def _evidence(
    project_dir: Path, batch: dict, item: dict, version: VersionManifest, keeps: dict,
) -> dict:
    operations = []
    for operation in item["operations"][:20]:
        record = {"id": operation["id"]}
        if "type" in operation or "kind" in operation:
            record["type"] = operation.get("type", operation.get("kind"))
        operations.append(record)
    question = item.get("clarification")
    unmeasured = {"status": "unmeasured", "detail": "no keep constraint for this comment"}
    evidence = {
        "schema": "yap-studio/reply-evidence@1", "batch": batch["id"],
        "comment_version": item["comment"]["version"], "version": version.id,
        "media_sha256": version.media["sha256"], "disposition": item["disposition"],
        "operations": operations, **_measure_moment(project_dir, item, version),
        "keep": (keeps.get((item["comment"]["version"], item["comment"]["id"]), unmeasured)
                 if item["comment"]["kind"] == "keep" else None),
        "question": ({key: question.get(key) for key in ("question", "answer", "option")}
                     if question is not None else None),
    }
    change = evidence["change"]
    while len(json.dumps(evidence, separators=(",", ":"), ensure_ascii=False).encode()) > 16384:
        if change is None or not (change["before"] or change["after"]):
            break
        words = max((change["before"], change["after"]), key=len)
        words.pop()
    validate_evidence(evidence)
    return evidence


def reply_evidence(project_dir: Path, batch: dict, item: dict, version: VersionManifest) -> dict:
    """Measure a comment's moment and protected material in the next saved version."""
    keeps = (_keep_results(project_dir, batch, version)
             if item["comment"]["kind"] == "keep" else {})
    return _evidence(project_dir, batch, item, version, keeps)


def _unanswered(item: dict) -> bool:
    question = item.get("clarification")
    return question is not None and question.get("answered_at") is None


def _changed(item: dict, evidence: dict) -> bool:
    return bool((evidence["change"] or {}).get("differences") or item["operations"])


def _check_status(item: dict, version: VersionManifest, status: str, evidence: dict) -> None:
    kind = item["comment"]["kind"]
    if status == "changed":
        if kind != "fix":
            raise FeedbackError("Only a fix comment can have a changed reply.",
                                suggested_action="Reply kept or declined to this keep comment.")
        if item["disposition"] == "declined":
            raise FeedbackError("This item was declined.",
                                suggested_action="Reply declined with the reason.")
        if item["disposition"] == "needs_clarification" and _unanswered(item):
            raise FeedbackError("This item still needs an answer to its clarification.",
                                suggested_action="Get an answer or reply needs_clarification.")
        if not _changed(item, evidence):
            raise FeedbackError(f"Nothing changed at this moment in {version.id}.",
                                suggested_action="Make the edit and save a new version, "
                                "or reply declined with the reason.")
    elif status == "kept":
        if kind != "keep":
            raise FeedbackError("Only a keep comment can have a kept reply.",
                                suggested_action="Reply changed or declined to this fix comment.")
        if evidence["keep"]["status"] == "broken":
            raise FeedbackError(f"The keep is broken: {evidence['keep']['detail']}",
                                suggested_action="Restore the kept material or ask the creator.")
    elif status == "declined" and item["operations"]:
        raise FeedbackError("This item has recorded operations, so it was applied.",
                            suggested_action="Reply changed with what the operations did.")
    elif status == "needs_clarification" and item.get("clarification") is None:
        raise FeedbackError("This item has no clarification question.",
                            suggested_action="Use yap-studio feedback ask to record the question.")


def write_reply(
    project_dir: Path, store: ReviewStore, batch_id: str, comment_version: str,
    comment_id: str, status: str, text: str, *, version_id: str | None = None,
) -> Reply:
    """Validate measured next-version evidence and add or replace one agent reply."""
    batch = load_batch(project_dir, batch_id)
    item = next((item for item in batch["items"] if
                 (item["comment"]["version"], item["comment"]["id"])
                 == (comment_version, comment_id)), None)
    if item is None:
        raise ValidationError("comment", "comment reference is not in this batch")
    if status not in REPLY_STATUSES:
        raise ValidationError("status", f"must be one of {', '.join(REPLY_STATUSES)}")
    text = validate_reply_text(text)
    result_id = batch["resulting_version"]
    if result_id is None and version_id is None:
        raise FeedbackError("This batch has no resulting version.",
                            suggested_action="Save the next version, then pass --version vN.")
    version = load_version(project_dir, version_id or result_id)
    try:
        matches = sha256_file(version.media_path) == version.media["sha256"]
    except OSError as exc:
        raise FeedbackError(
            f"Cannot read the saved media file {version.media_path}.",
            suggested_action="Restore the saved media file before replying.",
        ) from exc
    if not matches:
        raise FeedbackError(f"The saved media file {version.media_path} changed after saving.",
                            suggested_action="Restore the saved media file before replying.")
    versions = load_versions(project_dir)
    numbers = {saved.id: number for number, saved in enumerate(versions, 1)}
    if result_id is not None and numbers[version.id] < numbers[result_id]:
        raise FeedbackError(f"The reply version must be {result_id} or a later saved version.",
                            suggested_action=f"Use --version {result_id} or a later version.")
    if comment_version not in numbers:
        raise FeedbackError("This comment's version is not saved in the project.")
    if numbers[comment_version] >= numbers[version.id]:
        raise FeedbackError("The reply version must follow this comment's saved version.",
                            suggested_action="Save the next version, then pass --version vN.")
    evidence = reply_evidence(project_dir, batch, item, version)
    _check_status(item, version, status, evidence)
    entries = [SeriesEntry(saved.media["sha256"],
                           f"{saved.project}-{saved.id}{saved.media_path.suffix}",
                           version_id=saved.id, project=saved.project) for saved in versions]
    expected_id = batch["review"].get("series_id")
    if expected_id is not None and expected_id != entries[0].sha256:
        raise FeedbackError("This batch belongs to a different review series.",
                            suggested_action="Use the batch from this project's review series.")
    if not any(comment.id == comment_id for comment in
               store.comments(versions[numbers[comment_version] - 1].media["sha256"])):
        raise ValidationError("comment", "comment reference is not in the review store")
    series = store.open_series(entries)
    if expected_id is not None and series.id != expected_id:
        raise FeedbackError("This batch belongs to a different review series.",
                            suggested_action="Use the batch from this project's review series.")
    if result_id is None:
        set_resulting_version(project_dir, batch_id, version.id)
    return store.add_reply(series.id, comment_version=numbers[comment_version],
                           comment_id=comment_id, version=numbers[version.id],
                           status=status, text=text, evidence=evidence)


def reply_status(project_dir: Path, store: ReviewStore, batch_id: str | None = None) -> dict:
    """Read replies for a batch and suggest missing statuses without writing a series."""
    batch = load_batch(project_dir, batch_id) if batch_id is not None else latest_batch(project_dir)
    if batch is None:
        raise FeedbackError("No feedback batch exists.",
                            suggested_action="Run yap-studio feedback build first.")
    versions = load_versions(project_dir)
    series_id = batch["review"].get("series_id") or (
        versions[0].media["sha256"] if versions else None
    )
    series = None
    if series_id is not None:
        try:
            series = store.series(series_id)
        except InputError as exc:
            if exc.reason != "series does not exist":
                raise
    result_id = batch["resulting_version"]
    version = load_version(project_dir, result_id) if result_id is not None else None
    saved_by_id = {saved.id: saved for saved in versions}
    numbers = {saved.sha256: saved.number for saved in series.versions} if series else {}
    replies = series.replies if series else ()
    keeps = (_keep_results(project_dir, batch, version) if version is not None
             and any(item["comment"]["kind"] == "keep" for item in batch["items"]) else {})
    items = []
    for item in batch["items"]:
        comment = item["comment"]
        own = saved_by_id.get(comment["version"])
        first_result = numbers.get(version.media["sha256"]) if version else None
        reply = max((reply for reply in replies
                     if own is not None and first_result is not None
                     and reply.comment_version == numbers.get(own.media["sha256"])
                     and reply.comment_id == comment["id"] and reply.version >= first_result
                     and reply.evidence is not None
                     and reply.evidence.get("schema") == "yap-studio/reply-evidence@1"
                     and reply.evidence.get("batch") == batch["id"]),
                    key=lambda reply: reply.version, default=None)
        suggested = None
        if version is not None and reply is None:
            evidence = _evidence(project_dir, batch, item, version, keeps)
            if _unanswered(item):
                suggested = "needs_clarification"
            elif item["disposition"] == "declined":
                suggested = "declined"
            elif comment["kind"] == "keep":
                suggested = "kept"
            else:
                suggested = "changed"
            try:
                _check_status(item, version, suggested, evidence)
            except FeedbackError:
                suggested = None
        items.append({
            "comment": {key: comment[key] for key in ("version", "id")},
            **{key: comment[key] for key in ("kind", "target", "text")},
            "disposition": item["disposition"],
            "reply": ({"id": reply.id, "status": reply.status, "text": reply.text,
                       "evidence": reply.evidence} if reply is not None else None),
            "suggested": suggested,
        })
    return {"batch": batch["id"], "version": result_id,
            "complete": result_id is not None and all(item["reply"] is not None for item in items),
            "items": items}
