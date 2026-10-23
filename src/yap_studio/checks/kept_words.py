"""Account for every word occurrence missing from the reviewed cut."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from yap_studio.feedback.versions import (
    _snapshot_json,
    current_version,
    load_version,
    version_cut_words,
)
from yap_studio.project.model import CutWord, load_cut_words
from yap_studio.project.receipts import read_receipt
from yap_studio.receipt import sha256_file


def _removals(root: Path) -> list[dict]:
    try:
        return json.loads((root / "removals.json").read_bytes())["removals"]
    except FileNotFoundError:
        return []


def _unmatched(words: tuple[CutWord, ...], other: tuple[CutWord, ...]) -> list[CutWord]:
    """Match segment occurrences before allowing renamed segments."""
    remaining = Counter((word.segment, word.id) for word in other)
    pending = []
    for word in words:
        key = (word.segment, word.id)
        if remaining[key]:
            remaining[key] -= 1
        else:
            pending.append(word)
    remaining_ids: Counter[str] = Counter()
    for (_, word_id), count in remaining.items():
        remaining_ids[word_id] += count
    unmatched = []
    for word in pending:
        if remaining_ids[word.id]:
            remaining_ids[word.id] -= 1
        else:
            unmatched.append(word)
    return unmatched


def check_kept_words(root: Path, final: Path) -> dict[str, Any]:
    """Compare word ids with the reviewed version and explain every missing occurrence."""
    baseline = current_version(root)
    if baseline is not None:
        outputs = (read_receipt(root / "final") or {}).get("outputs", [])
        digest = next((item["sha256"] for item in outputs
                       if root / item["path"] == final and item.get("sha256")), None)
        if baseline.media["sha256"] == (digest or sha256_file(final)):
            baseline = load_version(root, baseline.parent) if baseline.parent else None
    result: dict[str, Any] = {
        "status": "skipped", "baseline": baseline.id if baseline else None,
        "total": 0, "kept": 0, "dropped": [], "added": [], "reason": None,
    }
    if baseline is None:
        result["reason"] = "no saved version to compare with"
        return result
    before = version_cut_words(baseline)
    after = load_cut_words(root / "cut/cut-words.json")
    result["total"] = len(before.words)
    changed = [take for take in sorted(before.words_sha256.keys() & after.words_sha256.keys())
               if before.words_sha256[take] != after.words_sha256[take]]
    if changed:
        result["reason"] = (f"transcript changed since {baseline.id} for {', '.join(changed)}; "
                            "the words were not compared")
        return result
    applied = [item for item in _removals(root) if item["status"] == "applied"]
    previous = (_snapshot_json(baseline, "removals")["removals"]
                if baseline.snapshots.get("removals") else [])
    saved = {(item["take"], item["start"], item["end"])
             for item in previous if item["status"] == "applied"}
    before_edit = _snapshot_json(baseline, "edit")
    after_edit = json.loads((root / "edit.json").read_bytes())
    before_segments = {segment["id"]: segment for segment in before_edit["segments"]}
    after_segments = {segment["id"]: segment for segment in after_edit["segments"]}
    source = {}
    for word in _unmatched(before.words, after.words):
        take = word.id.split(":", 1)[0]
        if take not in source:
            source[take] = {item["id"]: item for item in json.loads(
                (root / f"ingest/{take}/words.json").read_bytes())["words"]}
        original = source[take].get(word.id)
        midpoint = (original["start"] + original["end"]) / 2 if original else None
        covering = [item for item in applied if item["take"] == take and (
            word.id in item.get("words", [])
            or midpoint is not None and item["start"] <= midpoint < item["end"]
        )]
        approved = next((item for item in covering if item.get("by") == "manual"
                         or item.get("evidence", {}).get("decision") is True
                         or (item["take"], item["start"], item["end"]) in saved), None)
        removal = approved or next(iter(covering), None)
        story_changed = word.segment in before_segments and (
            word.segment not in after_segments or any(
                before_segments[word.segment][key] != after_segments[word.segment][key]
                for key in ("take", "from", "to")
            )
        )
        why = ("approved" if approved else "unapproved" if covering else
               "story" if story_changed else "unexplained")
        result["dropped"].append({"id": word.id, "segment": word.segment,
                                  "text": word.text, "time": round(word.start, 2),
                                  "why": why, "removal": removal["id"] if removal else None})
    for word in _unmatched(after.words, before.words):
        result["added"].append({"id": word.id, "segment": word.segment,
                                "text": word.text, "time": round(word.start, 2)})
    result["kept"] = result["total"] - len(result["dropped"])
    result["status"] = ("fail" if any(item["why"] in {"unapproved", "unexplained"}
                                    for item in result["dropped"]) else "pass")
    return result


def words_summary(result: dict) -> str:
    """Describe retained words and the reasons for approved changes."""
    if result["status"] == "skipped":
        return f"skipped: {result['reason']}"
    total, kept, baseline = result["total"], result["kept"], result["baseline"]
    if not result["dropped"]:
        return f"all {total} words of {baseline} kept"
    reasons = Counter(item["why"] for item in result["dropped"])
    descriptions = {"approved": "at the creator's request", "story": "by the story edit",
                    "unapproved": "without approval", "unexplained": "without explanation"}
    detail = ", ".join(f"{reasons[key]} {description}" for key, description in descriptions.items()
                       if reasons[key])
    return (f"{kept} of {total} words of {baseline} kept; "
            f"{len(result['dropped'])} removed: {detail}")
