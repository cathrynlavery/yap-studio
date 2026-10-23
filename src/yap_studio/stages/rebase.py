"""Move authored word references onto a newer source transcript."""

from __future__ import annotations

import copy
import hashlib
import json
from bisect import bisect_left, bisect_right
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from yap_studio import paths
from yap_studio.errors import InputError, YapError
from yap_studio.project.anchors import nominal_edge
from yap_studio.project.lock import (
    ConflictError,
    pid_alive,
    project_lock,
    timestamp,
    write_json_cas,
)
from yap_studio.project.model import Edit, Word, Words, parse_document
from yap_studio.project.status import ProjectContext, cut_inputs
from yap_studio.stages.cut import resolve_cut
from yap_studio.stages.timeline import _referenced_takes
from yap_studio.transcript.words import norm_words

MATCH_START_TOLERANCE_S = 0.3
_FILES = ("edit.json", "removals.json", "timeline/agent.json", "timeline/broll.json")


def _old_cut_plan(receipt_bytes: bytes | None, edit_bytes: bytes,
                  edit: dict[str, Any], context: ProjectContext,
                  ) -> tuple[dict[str, Any] | None, str]:
    """Use a cut receipt only when all its non-word inputs still match."""
    absent = "no cut for the old edit"
    if receipt_bytes is None:
        return None, absent
    try:
        record = json.loads(receipt_bytes)
        inputs = record["inputs"]
        plan = record["plan"]
        takes = {item["take"] for item in edit["segments"]}
        matches = (inputs.get("edit.json") == hashlib.sha256(edit_bytes).hexdigest()
                   and all(inputs.get(f"words:{take}") == edit["words_sha256"].get(take)
                           for take in takes))
        if matches and isinstance(plan["pieces"], list) and isinstance(
                plan["total_frames"], int):
            try:
                old_context = ProjectContext(context.project_dir)
                old_context._snapshots["edit.json"] = edit_bytes
                for take in takes:
                    digest = edit["words_sha256"][take]
                    relative = f"ingest/{take}/words.json"
                    words = context.read(relative)
                    if hashlib.sha256(words).hexdigest() != digest:
                        try:
                            words = context.read(f"ingest/{take}/previous/{digest}.json")
                        except FileNotFoundError:
                            pass
                        else:
                            if hashlib.sha256(words).hexdigest() != digest:
                                return None, (
                                    "cut is older than other changes; run cut before rebase")
                    old_context._snapshots[relative] = words
                current = cut_inputs(old_context)
            except (YapError, OSError, ValueError):
                return None, "cut is older than other changes; run cut before rebase"
            if all(current.get(name) == digest for name, digest in inputs.items()
                   if name != "edit.json" and not name.startswith("words:")):
                return plan, "resolved"
            return None, "cut is older than other changes; run cut before rebase"
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return None, absent


def _cut_changes(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare every segment's settled pieces on the source frame grid."""
    old = before["pieces"]
    new = after["pieces"]
    ids = dict.fromkeys(piece["segment_id"] for piece in [*old, *new])
    changed = []
    for segment in ids:
        earlier = [piece for piece in old if piece["segment_id"] == segment]
        later = [piece for piece in new if piece["segment_id"] == segment]
        old_ranges = [(piece["src_t0"], piece["src_t1"], piece["src_f0"], piece["src_f1"])
                      for piece in earlier]
        new_ranges = [(piece["src_t0"], piece["src_t1"], piece["src_f0"], piece["src_f1"])
                      for piece in later]
        frames_old = sum(piece["n_frames"] for piece in earlier)
        frames_new = sum(piece["n_frames"] for piece in later)
        if old_ranges == new_ranges and frames_old == frames_new:
            continue
        old_first, new_first = earlier[0] if earlier else None, later[0] if later else None
        old_last, new_last = earlier[-1] if earlier else None, later[-1] if later else None
        changed.append({
            "segment": segment,
            "start_old_s": float(Fraction(old_first["src_t0"])) if old_first else None,
            "start_new_s": float(Fraction(new_first["src_t0"])) if new_first else None,
            "start_frames": (new_first["src_f0"] - old_first["src_f0"]
                             if old_first and new_first else None),
            "end_old_s": float(Fraction(old_last["src_t1"])) if old_last else None,
            "end_new_s": float(Fraction(new_last["src_t1"])) if new_last else None,
            "end_frames": (new_last["src_f1"] - old_last["src_f1"]
                           if old_last and new_last else None),
            "frames_old": frames_old, "frames_new": frames_new,
            "pieces_old": len(earlier), "pieces_new": len(later),
            "source_ranges_old": [[piece["src_f0"], piece["src_f1"]] for piece in earlier],
            "source_ranges_new": [[piece["src_f0"], piece["src_f1"]] for piece in later],
        })
    return changed


def _edge_pins(before: dict[str, Any], moved: list[dict[str, Any]],
               edit: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose old resolved outer edges that new source words would move."""
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for piece in before["pieces"]:
        by_segment.setdefault(piece["segment_id"], []).append(piece)
    segments = {item["id"]: item for item in edit["segments"]}
    pins = []
    for change in moved:
        segment = change["segment"]
        pieces = by_segment.get(segment)
        if not pieces or segment not in segments:
            continue
        for edge, side, old_piece, field in (
            ("from", "start", pieces[0], "src_t0"),
            ("to", "end", pieces[-1], "src_t1"),
        ):
            if change[f"{side}_old_s"] == change[f"{side}_new_s"]:
                continue
            point = float(Fraction(old_piece[field]))
            if segments[segment][edge] == {"time": point}:
                continue
            pins.append({"segment": segment, "edge": edge, "time": point,
                         "frames": change[f"{side}_frames"]})
    return pins


def _check_fillers_writer(root: Path) -> None:
    """Do not change removals while fillers owns an active write or journal."""
    folder = root / "fillers"
    progress = folder / "progress.json"
    if progress.exists() and pid_alive(json.loads(progress.read_bytes()).get("pid")):
        raise YapError("fillers is already running.", code="stage_running",
                       suggested_action="Wait for the stage and retry.")
    if (folder / "receipt.pending.json").exists():
        raise InputError("fillers", "unfinished stage journal; rerun fillers first")


def _alignment(old: Words, new: Words) -> dict[str, Word]:
    """Find forced pairs in the closest maximum-size monotonic alignment."""
    Score = tuple[int, Decimal, int]  # pairs, total distance, optimal paths

    def combine(left: Score, right: Score) -> Score:
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        if left[1] != right[1]:
            return left if left[1] < right[1] else right
        return left[0], left[1], left[2] + right[2]

    class BestPrefix:
        def __init__(self, size: int) -> None:
            self.values: list[Score] = [(0, Decimal(0), 0)] * (size + 1)

        def update(self, index: int, value: Score) -> None:
            index += 1
            while index < len(self.values):
                self.values[index] = combine(self.values[index], value)
                index += index & -index

        def query(self, index: int) -> Score:
            result: Score = (0, Decimal(0), 1)
            index += 1
            while index > 0:
                result = combine(result, self.values[index])
                index -= index & -index
            return result

    target_starts = [Decimal(str(word.start)) for word in new.words]
    by_token: dict[tuple[str, ...], list[tuple[Decimal, int]]] = {}
    for index, word in enumerate(new.words):
        by_token.setdefault(tuple(norm_words(word.text)), []).append(
            (target_starts[index], index))
    for positions in by_token.values():
        positions.sort()
    tolerance = Decimal(str(MATCH_START_TOLERANCE_S))
    candidates: list[tuple[int, int, Decimal]] = []
    for index, word in enumerate(old.words):
        start = Decimal(str(word.start))
        positions = by_token.get(tuple(norm_words(word.text)), [])
        low = bisect_left(positions, (start - tolerance, -1))
        high = bisect_right(positions, (start + tolerance, len(new.words)))
        candidates.extend((index, new_index, abs(start - new_start))
                          for new_start, new_index in positions[low:high])
    candidates.sort(key=lambda pair: (pair[0], pair[1]))
    if not candidates:
        return {}

    size = len(new.words)
    forward: list[Score] = [(0, Decimal(0), 0)] * len(candidates)
    prefix = BestPrefix(size)
    position = 0
    while position < len(candidates):
        end = position + 1
        while end < len(candidates) and candidates[end][0] == candidates[position][0]:
            end += 1
        for entry in range(position, end):
            _, new_index, distance = candidates[entry]
            prior = prefix.query(new_index - 1)
            forward[entry] = prior[0] + 1, prior[1] + distance, prior[2]
        for entry in range(position, end):
            prefix.update(candidates[entry][1], forward[entry])
        position = end
    optimum = prefix.query(size - 1)

    backward: list[Score] = [(0, Decimal(0), 0)] * len(candidates)
    suffix = BestPrefix(size)
    position = len(candidates)
    while position:
        start = position - 1
        while start and candidates[start - 1][0] == candidates[position - 1][0]:
            start -= 1
        for entry in range(position - 1, start - 1, -1):
            _, new_index, distance = candidates[entry]
            later = suffix.query(size - new_index - 2)
            backward[entry] = later[0] + 1, later[1] + distance, later[2]
        for entry in range(position - 1, start - 1, -1):
            suffix.update(size - candidates[entry][1] - 1, backward[entry])
        position = start

    choices: dict[int, list[tuple[int, int]]] = {}
    for entry, (old_index, new_index, distance) in enumerate(candidates):
        before, after = forward[entry], backward[entry]
        if (before[0] + after[0] - 1 == optimum[0]
                and before[1] + after[1] - distance == optimum[1]):
            choices.setdefault(old_index, []).append((new_index, before[2] * after[2]))
    return {old.words[index].id: new.words[options[0][0]]
            for index, options in choices.items()
            if len(options) == 1 and options[0][1] == optimum[2]}


def _nearest(old: Word | None, words: Words) -> list[dict[str, Any]]:
    if old is None:
        return []
    return [{"id": word.id, "text": word.text, "start": word.start, "end": word.end}
            for word in sorted(words.words, key=lambda word: (
                abs(word.start - old.start), word.start,
            ))[:3]]


def _removal_ids(item: dict[str, Any]) -> set[str]:
    """Collect decision and refined evidence ids for one removal."""
    return set(item.get("words", [])) | {
        word["id"] for word in item.get("evidence", {}).get("refined_source_words", [])
    }


def _word_fits_removal(item: dict[str, Any], word_id: str,
                       by_id: dict[str, Word]) -> bool:
    word = by_id.get(word_id)
    return bool(word and word.start <= item["end"] and word.end >= item["start"])


def _fits_removal(item: dict[str, Any], words: Words) -> bool:
    """Check that each named word still touches the source-time cut range."""
    by_id = {word.id: word for word in words.words}
    return all(_word_fits_removal(item, word_id, by_id)
               for word_id in _removal_ids(item))


def _restore_published(root: Path, original: dict[str, bytes | None],
                       published: dict[str, bytes]) -> list[str]:
    """Restore exact prior bytes only if every rebase output is still ours."""
    with project_lock(root):
        _check_fillers_writer(root)
        for relative, payload in published.items():
            target = root / relative
            if not target.is_file() or target.read_bytes() != payload:
                raise ConflictError(target)
        for relative, payload in original.items():
            target = root / relative
            if payload is None:
                paths._guard_write(target)
                target.unlink()
            else:
                with paths.atomic_output(target) as temporary:
                    temporary.write_bytes(payload)
    return list(original)


def rebase_edit(project_dir: Path, *, dry_run: bool = False,
                pin_unmatched: bool = False) -> dict[str, Any]:
    """Plan all word moves, then compare every input before writing any document."""
    with paths.operation():
        context = ProjectContext(project_dir)
        root = context.project_dir
        _check_fillers_writer(root)
        raw: dict[str, bytes] = {}
        authored: dict[str, dict[str, Any]] = {}
        for relative in _FILES:
            try:
                payload = context.read(relative)
            except FileNotFoundError:
                if relative == "edit.json":
                    raise InputError(relative, "missing edit; create edit.json first") from None
                continue
            data = json.loads(payload)
            name = ("timeline" if relative.startswith("timeline/")
                    else relative.removesuffix(".json"))
            parse_document(name, data, filename=relative,
                           lane=relative.split("/")[1].removesuffix(".json")
                           if name == "timeline" else None)
            raw[relative], authored[relative] = payload, data
        changes = copy.deepcopy(authored)
        inputs = dict(raw)
        try:
            cut_receipt = context.read("cut/receipt.json")
        except FileNotFoundError:
            cut_receipt = None
        else:
            inputs["cut/receipt.json"] = cut_receipt
        old_plan, old_cut_reason = _old_cut_plan(
            cut_receipt, raw["edit.json"], authored["edit.json"], context)
        missing: list[dict[str, str]] = []
        unmatched: list[dict[str, Any]] = []
        pinned: list[dict[str, Any]] = []
        shifted: list[dict[str, Any]] = []
        files = {relative: {"reanchored": 0, "unchanged": 0, "pinned": 0,
                            "unmatched": 0, "moved": [], "mapping": []}
                 for relative in authored}
        current: dict[str, tuple[str, Words]] = {}
        snapshots: dict[tuple[str, str], Words | None] = {}
        alignments: dict[tuple[int, int], dict[str, Word]] = {}

        def aligned(old: Words, new: Words) -> dict[str, Word]:
            key = id(old), id(new)
            if key not in alignments:
                alignments[key] = _alignment(old, new)
            return alignments[key]

        def new_words(take: str) -> tuple[str, Words]:
            if take not in current:
                relative = f"ingest/{take}/words.json"
                payload = context.read(relative)
                inputs[relative] = payload
                current[take] = (hashlib.sha256(payload).hexdigest(), cast(Words,
                    parse_document("words", json.loads(payload), filename=relative)))
            return current[take]

        def old_words(relative: str, take: str, recorded: str | None) -> Words | None:
            now_hash, now = new_words(take)
            if recorded == now_hash:
                return now
            if not recorded:
                issue = {"file": relative, "take": take, "hash": "missing"}
                if issue not in missing:
                    missing.append(issue)
                return None
            key = take, recorded
            if key not in snapshots:
                snapshot = f"ingest/{take}/previous/{recorded}.json"
                try:
                    payload = context.read(snapshot)
                except FileNotFoundError:
                    snapshots[key] = None
                else:
                    inputs[snapshot] = payload
                    if hashlib.sha256(payload).hexdigest() != recorded:
                        raise InputError(snapshot, "snapshot hash differs from its name")
                    snapshots[key] = cast(Words, parse_document(
                        "words", json.loads(payload), filename=snapshot,
                    ))
            if snapshots[key] is None:
                issue = {"file": relative, "take": take, "hash": recorded}
                if issue not in missing:
                    missing.append(issue)
            return snapshots[key]

        def move_id(relative: str, owner: dict[str, str], anchor: str, word_id: str,
                    old: Words, new: Words, *, guard: str | None = None,
                    edge: str = "start", offset: float = 0,
                    transition: bool = False) -> tuple[str | None, float | None, bool]:
            source = next((word for word in old.words if word.id == word_id), None)
            target = aligned(old, new).get(word_id) if source is not None else None
            if guard is not None and source is not None and (
                    norm_words(guard) != norm_words(source.text)
                    or target is not None and norm_words(guard) != norm_words(target.text)):
                target = None
            item = {"file": relative, **owner, "anchor": anchor, "word": word_id,
                    "text": source.text if source else None,
                    "time": source.start if source else None,
                    "nearest": _nearest(source, new)}
            if target is None:
                if transition:
                    item["note"] = "transition needs a word at a join; place it again"
                if pin_unmatched and source is not None and relative.startswith(
                        "timeline/") and not transition and new.words:
                    nearest = min(new.words, key=lambda word: (
                        abs(Fraction(str(word.start)) - Fraction(str(source.start))),
                        Fraction(str(word.start)),
                    ))
                    pinned.append({**item, "new_word": nearest.id, "new_text": nearest.text})
                    files[relative]["pinned"] += 1
                    files[relative]["mapping"].append({**owner, "anchor": anchor,
                                                         "old": word_id, "new": nearest.id})
                    return nearest.id, None, True
                if pin_unmatched and source is not None and not transition:
                    point = (source.start if edge == "start" else source.end) + offset
                    if anchor.startswith("words") or anchor.startswith("evidence"):
                        point = None
                    if point is not None and point >= 0:
                        pinned.append(item)
                        files[relative]["pinned"] += 1
                        files[relative]["mapping"].append({**owner, "anchor": anchor,
                                                             "old": word_id, "time": point})
                        return None, point, False
                    if point is None:
                        pinned.append(item)
                        files[relative]["pinned"] += 1
                        files[relative]["mapping"].append({**owner, "anchor": anchor,
                                                             "old": word_id, "new": None})
                        return None, None, False
                unmatched.append(item)
                files[relative]["unmatched"] += 1
                files[relative]["mapping"].append({**owner, "anchor": anchor,
                                                     "old": word_id, "new": None})
                return None, None, False
            files[relative]["mapping"].append({**owner, "anchor": anchor,
                                                 "old": word_id, "new": target.id})
            if target.id == word_id:
                files[relative]["unchanged"] += 1
            else:
                files[relative]["reanchored"] += 1
                files[relative]["moved"].append(f"{word_id} -> {target.id}")
            return target.id, None, False

        def anchor(relative: str, owner: dict[str, str], label: str,
                   value: dict[str, Any], old: Words, new: Words, *,
                   transition: bool = False) -> None:
            word_id = value.get("word")
            if word_id is None:
                return
            target, point, pinned_word = move_id(
                relative, owner, label, word_id, old, new, guard=value.get("text"),
                edge=value.get("edge", "end" if label in {"to", "end"} else "start"),
                offset=value.get("offset_s", 0), transition=transition,
            )
            if target is not None:
                value["word"] = target
                if pinned_word:
                    value.pop("text", None)
            elif point is not None:
                value.clear()
                value["time"] = point

        edit_hashes = authored["edit.json"]["words_sha256"]
        for relative, data in changes.items():
            if relative == "removals.json":
                continue
            hashes = data["words_sha256"]
            if relative == "edit.json":
                referenced = {item["take"] for item in data["segments"]
                              if any(item[label].get("word") for label in ("from", "to"))}
            else:
                lane = cast(Any, parse_document("timeline", authored[relative],
                                                filename=relative))
                referenced = {take for element in lane.elements
                              for take in _referenced_takes(element)}
            available = {take: old_words(relative, take, hashes.get(take))
                         for take in sorted(referenced)}
            for take in sorted(set(hashes) | referenced):
                if take not in referenced or available[take] is not None:
                    hashes[take] = new_words(take)[0]
            if relative == "edit.json":
                for item in data["segments"]:
                    old = available.get(item["take"])
                    if old is None:
                        continue
                    new = new_words(item["take"])[1]
                    for label in ("from", "to"):
                        anchor(relative, {"segment": item["id"]}, label, item[label], old, new)
                old_edit = cast(Edit, parse_document("edit", authored[relative]))
                new_edit = cast(Edit, parse_document("edit", data))
                for before, after in zip(old_edit.segments, new_edit.segments, strict=True):
                    old = available.get(before.take)
                    if old is None or (authored[relative]["words_sha256"].get(before.take)
                                       == new_words(before.take)[0]):
                        continue
                    for label, old_anchor, new_anchor, start in (
                        ("from", before.from_, after.from_, True),
                        ("to", before.to, after.to, False),
                    ):
                        if old_anchor.word is None or new_anchor.word is None or any(
                            item.get("segment") == before.id and item["anchor"] == label
                            for item in unmatched
                        ):
                            continue
                        first = nominal_edge(before, old_anchor, old, start=start)
                        second = nominal_edge(after, new_anchor, new_words(before.take)[1],
                                              start=start)
                        if abs(second - first) >= Fraction(1, 30):
                            shifted.append({"segment": before.id, "anchor": label,
                                            "old_s": float(first), "new_s": float(second)})
            else:
                def nested(value: Any, item: dict[str, Any], location: str, *,
                           available: dict[str, Words | None] = available,
                           relative: str = relative) -> None:
                    if isinstance(value, list):
                        for index, child in enumerate(value):
                            nested(child, item, f"{location}[{index}]")
                    elif isinstance(value, dict):
                        for key, child in value.items():
                            here = f"{location}.{key}"
                            if key == "at" and isinstance(child, dict):
                                word_id = child.get("word")
                                if word_id:
                                    take = word_id.split(":", 1)[0]
                                    old = available.get(take)
                                    if old is not None:
                                        anchor(relative, {"element": item["id"]}, here,
                                               child, old, new_words(take)[1],
                                               transition=item["kind"] == "transition")
                            else:
                                nested(child, item, here)

                for item in data["elements"]:
                    for label in ("start", "end"):
                        value = item.get(label)
                        word_id = value.get("word") if value else None
                        if word_id:
                            take = word_id.split(":", 1)[0]
                            old = available.get(take)
                            if old is not None:
                                anchor(relative, {"element": item["id"]}, label,
                                       value, old, new_words(take)[1],
                                       transition=item["kind"] == "transition")
                    nested(item.get("slots", {}), item, "slots")
                    nested(item.get("keys", []), item, "keys")

        if "removals.json" in changes:
            relative = "removals.json"

            def snapshot_for(item: dict[str, Any], now: Words) -> tuple[Words | None, bool]:
                take = item["take"]
                folder = root / f"ingest/{take}/previous"
                candidates: list[tuple[tuple[int, int, str], Words]] = []
                evidence = item.get("evidence", {}).get("refined_source_words", [])
                for path in sorted(folder.glob("*.json")):
                    digest = path.stem
                    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                        continue
                    name = path.relative_to(root).as_posix()
                    payload = context.read(name)
                    inputs[name] = payload
                    if hashlib.sha256(payload).hexdigest() != digest:
                        raise InputError(name, "snapshot hash differs from its name")
                    old = cast(Words, parse_document("words", json.loads(payload), filename=name))
                    if not _fits_removal(item, old):
                        continue
                    by_id = {word.id: word for word in old.words}
                    matching_text = sum(norm_words(by_id[row["id"]].text) == norm_words(
                        row.get("text", "")) for row in evidence)
                    matched_ids = sum(word_id in aligned(old, now)
                                      for word_id in _removal_ids(item))
                    candidates.append(((matching_text, matched_ids, digest), old))
                if not candidates:
                    return None, False
                best_score = max(score[:2] for score, _ in candidates)
                best = [(score, old) for score, old in candidates
                        if score[:2] == best_score]
                removal_ids = _removal_ids(item)
                mappings = [{word_id: target.id for word_id, target in aligned(old, now).items()
                             if word_id in removal_ids} for _, old in best]
                if any(mapping != mappings[0] for mapping in mappings[1:]):
                    unmatched.append({"file": relative, "removal": item["id"],
                                      "take": take, "anchor": "words/evidence",
                                      "reason": "ambiguous previous snapshots map removal words "
                                                "to different current words; resolve the "
                                                "removal IDs manually"})
                    files[relative]["unmatched"] += 1
                    return None, True
                return max(best, key=lambda entry: entry[0])[1], False

            def stale_id(item: dict[str, Any], label: str, word_id: str,
                         now: Words, evidence_text: str | None = None) -> None:
                near = sorted(now.words, key=lambda word: (
                    abs(word.start - item["start"]), word.start,
                ))[:3]
                issue = {"file": relative, "removal": item["id"], "anchor": label,
                         "word": word_id, "text": evidence_text, "time": item["start"],
                         "nearest": [{"id": word.id, "text": word.text,
                                      "start": word.start, "end": word.end} for word in near]}
                if pin_unmatched:
                    pinned.append(issue)
                    files[relative]["pinned"] += 1
                else:
                    unmatched.append(issue)
                    files[relative]["unmatched"] += 1
                files[relative]["mapping"].append({"removal": item["id"],
                                                     "anchor": label, "old": word_id,
                                                     "new": None})

            for item in changes[relative]["removals"]:
                take = item["take"]
                if take not in edit_hashes or not (item.get("words") or item.get(
                        "evidence", {}).get("refined_source_words")):
                    continue
                now_hash, new = new_words(take)
                if edit_hashes[take] != now_hash:
                    old = old_words(relative, take, edit_hashes[take])
                    if old is None:
                        continue
                else:
                    if _fits_removal(item, new):
                        continue
                    old, ambiguous = snapshot_for(item, new)
                    if ambiguous:
                        continue
                    if old is None:
                        current_ids = {word.id: word for word in new.words}
                        kept_ids = []
                        for index, word_id in enumerate(item.get("words", [])):
                            if _word_fits_removal(item, word_id, current_ids):
                                kept_ids.append(word_id)
                            else:
                                stale_id(item, f"words[{index}]", word_id, new)
                        if pin_unmatched:
                            item["words"] = kept_ids
                        kept_evidence = []
                        for index, word in enumerate(item.get("evidence", {}).get(
                                "refined_source_words", [])):
                            if _word_fits_removal(item, word["id"], current_ids):
                                kept_evidence.append(word)
                            else:
                                stale_id(item, f"evidence.refined_source_words[{index}].id",
                                         word["id"], new, word.get("text"))
                        if pin_unmatched:
                            item.get("evidence", {}).get("refined_source_words", [])[:] = (
                                kept_evidence)
                        continue
                kept = []
                for index, word_id in enumerate(item.get("words", [])):
                    target, _, _ = move_id(relative, {"removal": item["id"]},
                                           f"words[{index}]", word_id, old, new)
                    if target is not None:
                        kept.append(target)
                item["words"] = kept
                for index, word in enumerate(item.get("evidence", {}).get(
                        "refined_source_words", [])):
                    target, _, _ = move_id(relative, {"removal": item["id"]},
                                           f"evidence.refined_source_words[{index}].id",
                                           word["id"], old, new)
                    if target is not None:
                        word["id"] = target
                    elif pin_unmatched:
                        word["id"] = None
                if pin_unmatched:
                    item.get("evidence", {}).get("refined_source_words", [])[:] = [
                        word for word in item.get("evidence", {}).get("refined_source_words", [])
                        if word["id"] is not None
                    ]

        for relative, data in changes.items():
            name = ("timeline" if relative.startswith("timeline/")
                    else relative.removesuffix(".json"))
            parse_document(name, data, filename=relative,
                           lane=relative.split("/")[1].removesuffix(".json")
                           if name == "timeline" else None)
            files[relative]["moved"] = list(dict.fromkeys(files[relative]["moved"]))
        changed = [relative for relative in changes if changes[relative] != authored[relative]]
        refused = bool(unmatched or missing)
        status = "unmatched" if refused else "dry_run" if dry_run else (
            "rebased" if changed else "unchanged")
        report = {"status": status, "files": files, "missing_snapshots": missing,
                  "shifted": [{**item, "estimated": True} for item in shifted],
                  "kept": [],
                  "edges_compared": ("dry run" if dry_run else old_cut_reason
                                     if old_plan is None else "no changes" if not changed
                                     else "rebase not written"),
                  "cut_change": None,
                  "unmatched": unmatched, "pinned": pinned, "changed": changed,
                  "next": "Run yap-studio cut DIR next."}
        if refused or dry_run or not changed:
            _check_fillers_writer(root)
            return report
        receipt_relative = "fillers/receipt.json"
        receipt_path = root / receipt_relative
        receipt_raw = context.read(receipt_relative) if receipt_path.is_file() else None
        if receipt_raw is not None:
            inputs[receipt_relative] = receipt_raw
        receipt_data = None
        if "removals.json" in changed:
            payload = (json.dumps(changes["removals.json"], sort_keys=True, indent=2,
                                  ensure_ascii=False, allow_nan=False) + "\n").encode()
            receipt_data = json.loads(receipt_raw) if receipt_raw is not None else {
                "schema": "yap-studio/stage-receipt@1", "stage": "fillers",
                "status": "verified", "started": timestamp(), "inputs": {},
                "outputs": [], "asserts": [],
            }
            receipt_data["outputs"] = [
                item for item in receipt_data["outputs"] if item["path"] != "removals.json"
            ] + [{"path": "removals.json", "sha256": hashlib.sha256(payload).hexdigest(),
                  "bytes": len(payload)}]
            receipt_data["finished"] = timestamp()
        original: dict[str, bytes | None] = {relative: raw[relative] for relative in changed}
        if receipt_data is not None:
            original[receipt_relative] = receipt_raw
        expected: dict[str, bytes | None] = {**inputs, receipt_relative: receipt_raw}
        published: dict[str, bytes] = {}
        with project_lock(root):
            _check_fillers_writer(root)
            if receipt_raw is None and receipt_path.exists():
                raise ConflictError(receipt_path)
            for relative, payload in inputs.items():
                target = root / relative
                if not target.is_file() or target.read_bytes() != payload:
                    raise ConflictError(target)
            for relative in changed:
                write_json_cas(root / relative, changes[relative],
                               base_sha256=hashlib.sha256(raw[relative]).hexdigest())
                expected[relative] = published[relative] = (root / relative).read_bytes()
            if receipt_data is not None:
                write_json_cas(receipt_path, receipt_data,
                               base_sha256=hashlib.sha256(receipt_raw).hexdigest()
                               if receipt_raw is not None else None)
                expected[receipt_relative] = published[receipt_relative] = (
                    receipt_path.read_bytes())
        if old_plan is not None:
            resolve_error: Exception | None = None
            for attempt in range(4):
                try:
                    new_plan = resolve_cut(root).to_dict()
                    moved = _cut_changes(old_plan, new_plan)
                except Exception as exc:
                    resolve_error = exc
                    break
                if not moved and old_plan["total_frames"] == new_plan["total_frames"]:
                    break
                if attempt == 3:
                    break
                edit_path = root / "edit.json"
                edit_bytes = expected["edit.json"]
                assert edit_bytes is not None
                edit_data = json.loads(edit_bytes)
                pins = _edge_pins(old_plan, moved, edit_data)
                if not pins:
                    break
                for item in pins:
                    segment = next(row for row in edit_data["segments"]
                                   if row["id"] == item["segment"])
                    segment[item["edge"]] = {"time": item["time"]}
                parse_document("edit", edit_data, filename="edit.json")
                with project_lock(root):
                    _check_fillers_writer(root)
                    for relative, payload in expected.items():
                        target = root / relative
                        if ((payload is None and target.exists()) or
                                (payload is not None and (not target.is_file() or
                                                          target.read_bytes() != payload))):
                            raise ConflictError(target)
                    write_json_cas(edit_path, edit_data,
                                   base_sha256=hashlib.sha256(edit_bytes).hexdigest())
                    expected["edit.json"] = published["edit.json"] = edit_path.read_bytes()
                original.setdefault("edit.json", raw["edit.json"])
                report["kept"].extend(pins)
                if "edit.json" not in report["changed"]:
                    report["changed"].append("edit.json")
            if resolve_error is not None:
                message = (resolve_error.reason if isinstance(resolve_error, InputError)
                           else resolve_error.message if isinstance(resolve_error, YapError)
                           else str(resolve_error))
                report["restored"] = _restore_published(root, original, published)
                report["status"] = "cut_failed"
                report["resolve_error"] = message
                report["edges_compared"] = message
                report["kept"] = []
            else:
                report["edges_compared"] = "resolved"
                report["shifted"] = moved
                report["cut_change"] = {"frames_old": old_plan["total_frames"],
                                        "frames_new": new_plan["total_frames"]}
                if moved or old_plan["total_frames"] != new_plan["total_frames"]:
                    report["status"] = "moved"
        return report
