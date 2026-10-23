"""Plain words for the agent's story, creator notes, and the pre-render wait."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Literal

from yap_studio.feedback.batch import (
    BATCH_SCHEMA,
    FeedbackError,
    _batch_numbers,
    _batch_path,
    _inputs,
    _write_batch,
    load_batch,
)
from yap_studio.feedback.keeps import keep_constraint
from yap_studio.feedback.locate import Location
from yap_studio.feedback.moments import Occurrence
from yap_studio.paths import atomic_write_text, validate_output
from yap_studio.project.anchors import nominal_edge
from yap_studio.project.lock import project_lock
from yap_studio.project.model import Word, Words, load_edit, load_project, load_removals, load_words
from yap_studio.receipt import sha256_file
from yap_studio.review.model import Comment

OUTLINE_SCHEMA = "yap-studio/story-outline@1"
NOTES_SCHEMA = "yap-studio/outline-notes@1"
STATE_SCHEMA = "yap-studio/outline-state@1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_NOTE_ID = re.compile(r"o([1-9][0-9]*)\Z")
_BATCH_ID = re.compile(r"b[1-9][0-9]*\Z")
_WORD_ID = re.compile(r"t[1-9][0-9]*:(0|[1-9][0-9]*)\Z")


def _refuse(reason: str, action: str = "Correct the outline input and retry.") -> FeedbackError:
    return FeedbackError(reason, suggested_action=action)


def _now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _canonical(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _outline_id(data: dict) -> str:
    content = {key: value for key, value in data.items() if key not in {"id", "created_at"}}
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


def _write(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        return data
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise _refuse(f"{path.name}: cannot read valid JSON.",
                      "Restore the file from a valid copy and retry.") from exc


def _keys(data: object, keys: set[str]) -> bool:
    return isinstance(data, dict) and set(data) == keys


def _hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _time(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        return datetime.fromisoformat(value).utcoffset() == timedelta(0)
    except ValueError:
        return False


def _seconds(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _word_pairs(words: list[Word]) -> list[dict]:
    return [{"id": word.id, "text": word.text} for word in words]


def _midpoint(word: Word) -> Fraction:
    return (Fraction(str(word.start)) + Fraction(str(word.end))) / 2


def _source_words(root: Path, take: str, expected: str | None) -> Words:
    path = root / "ingest" / take / "words.json"
    if expected is not None:
        try:
            digest = sha256_file(path)
        except OSError as exc:
            raise _refuse(f"{take}: cannot read ingest words.",
                          "Run ingest again or update the edit.") from exc
        if digest != expected:
            raise _refuse(f"{take}: words.json changed since the edit was written.",
                          "Run ingest again or update the edit.")
    words = load_words(path)
    if words.take != take:
        raise _refuse(f"{take}: words.json belongs to {words.take}.")
    return words


def build_outline(project_dir: Path, *, summary: str | None = None) -> dict:
    """Write the current edit's source geometry and spoken words under its lock."""
    if summary is not None and (not isinstance(summary, str) or len(summary) > 1000):
        raise _refuse("summary must be null or at most 1000 characters.")
    root = Path(project_dir)
    with project_lock(root):
        project = load_project(root / "project.json")
        edit = load_edit(root / "edit.json")
        removals_path = root / "removals.json"
        inputs = {"edit.json": sha256_file(root / "edit.json"),
                  "removals.json": sha256_file(removals_path) if removals_path.exists() else None}
        sources = {}
        for take in project.takes:
            sources[take.id] = _source_words(root, take.id, edit.words_sha256.get(take.id))
            inputs[f"words:{take.id}"] = sha256_file(root / "ingest" / take.id / "words.json")
        removals = load_removals(removals_path).removals if inputs["removals.json"] else ()
        removed = {
            take: {word.id for word in source.words if any(
                removal.take == take and removal.status == "applied"
                and Fraction(str(removal.start)) <= _midpoint(word) < Fraction(str(removal.end))
                for removal in removals
            )} for take, source in sources.items()
        }
        ordered = {take: sorted(source.words, key=lambda word: (word.start, word.end, word.id))
                   for take, source in sources.items()}
        beats, used = [], {take: set() for take in sources}
        for index, segment in enumerate(edit.segments):
            if segment.take not in sources:
                raise _refuse(f"{segment.take}: segment {segment.id} names an unknown take.")
            source = sources[segment.take]
            start = nominal_edge(segment, segment.from_, source, None, start=True)
            end = nominal_edge(segment, segment.to, source, None, start=False)
            if end <= start:
                raise _refuse(f"{segment.id}: the beat's end must follow its start.")
            from_word, to_word = segment.from_.word, segment.to.word
            candidates = ordered[segment.take]
            if from_word is not None or to_word is not None:
                # Cut membership follows file order within inclusive word anchors.
                ids = [word.id for word in source.words]
                first = ids.index(from_word) if from_word is not None else 0
                last = ids.index(to_word) + 1 if to_word is not None else len(ids)
                candidates = source.words[first:last]
            words = [word for word in candidates
                     if (start <= _midpoint(word) < end
                         or from_word is not None and _midpoint(word) < start
                         or to_word is not None and _midpoint(word) >= end)
                     and word.id not in removed[segment.take]]
            used[segment.take].update(word.id for word in words)
            role = "opening" if index == 0 else (
                "ending" if index == len(edit.segments) - 1 else "beat"
            )
            beats.append({"id": segment.id, "role": role, "take": segment.take,
                          "start": float(start), "end": float(end), "note": segment.note,
                          "text": " ".join(word.text for word in words),
                          "words": _word_pairs(words)})
        left_out, truncated = [], False

        def add_run(take: str, number: int, run: list[Word]) -> None:
            nonlocal truncated
            if len(left_out) >= 200:
                truncated = True
                return
            truncated |= len(run) > 400
            left_out.append({"id": f"{take}.{number}", "take": take,
                             "start": min(word.start for word in run),
                             "end": max(word.end for word in run),
                             "text": " ".join(word.text for word in run),
                             "words": _word_pairs(run[:400])})

        used_takes = {segment.take for segment in edit.segments}
        for take, words in ordered.items():
            if take not in used_takes:
                if words:
                    add_run(take, 0, words)
                continue
            runs: list[list[Word]] = []
            run: list[Word] = []
            for word in words:
                if word.id not in used[take] and word.id not in removed[take]:
                    run.append(word)
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)
            number = 0
            for run in runs:
                if len(run) >= 3 or max(w.end for w in run) - min(w.start for w in run) >= 2:
                    add_run(take, number, run)
                    number += 1
        if {beat["id"] for beat in beats}.intersection(entry["id"] for entry in left_out):
            raise _refuse("A beat id conflicts with a left-out id.", "Give the beat a unique id.")
        data = {"schema": OUTLINE_SCHEMA, "project": project.name,
                "edit_sha256": inputs["edit.json"], "inputs": inputs,
                "created_at": _timestamp(_now()),
                "summary": summary, "beats": beats, "left_out": left_out,
                "left_out_truncated": truncated,
                "duration_s": round(sum(beat["end"] - beat["start"] for beat in beats), 2)}
        data["id"] = _outline_id(data)
        _write(root / "outline" / "outline.json", data)
        return data


def _load_outline(root: Path) -> dict | None:
    path = root / "outline" / "outline.json"
    if not path.exists():
        return None
    data = _read(path)
    valid = (
        _keys(data, {"schema", "id", "project", "edit_sha256", "inputs", "created_at", "summary",
                     "beats", "left_out", "left_out_truncated", "duration_s"})
        and data["schema"] == OUTLINE_SCHEMA and _hash(data["id"])
        and _hash(data["edit_sha256"]) and _time(data["created_at"])
        and isinstance(data["inputs"], dict)
        and {"edit.json", "removals.json"} <= set(data["inputs"])
        and data["inputs"]["edit.json"] == data["edit_sha256"]
        and all((key in {"edit.json", "removals.json"}
                 or re.fullmatch(r"words:t[1-9][0-9]*", key) is not None)
                and (_hash(digest) or key == "removals.json" and digest is None)
                for key, digest in data["inputs"].items())
        and isinstance(data["project"], str) and bool(data["project"])
        and (data["summary"] is None or isinstance(data["summary"], str)
             and len(data["summary"]) <= 1000)
        and isinstance(data["beats"], list) and isinstance(data["left_out"], list)
        and len(data["left_out"]) <= 200 and type(data["left_out_truncated"]) is bool
        and _seconds(data["duration_s"])
    )
    seen = set()
    for group in ("beats", "left_out") if valid else ():
        for entry in data[group]:
            fields = {"id", "take", "start", "end", "text", "words"}
            if group == "beats":
                fields |= {"role", "note"}
            if (not _keys(entry, fields) or not isinstance(entry["id"], str)
                    or not entry["id"] or entry["id"] in seen
                    or not isinstance(entry["take"], str)
                    or re.fullmatch(r"t[1-9][0-9]*", entry["take"]) is None
                    or f"words:{entry['take']}" not in data["inputs"]
                    or not _seconds(entry["start"]) or not _seconds(entry["end"])
                    or entry["end"] < entry["start"] or not isinstance(entry["text"], str)
                    or not isinstance(entry["words"], list)):
                valid = False
                break
            seen.add(entry["id"])
            if group == "beats" and (
                entry["role"] not in ("opening", "beat", "ending")
                or entry["note"] is not None and not isinstance(entry["note"], str)
            ):
                valid = False
            word_ids = set()
            for word in entry["words"]:
                if (not _keys(word, {"id", "text"}) or not isinstance(word["id"], str)
                        or _WORD_ID.fullmatch(word["id"]) is None
                        or word["id"].split(":")[0] != entry["take"]
                        or word["id"] in word_ids or not isinstance(word["text"], str)):
                    valid = False
                    break
                word_ids.add(word["id"])
            if group == "left_out" and len(entry["words"]) > 400:
                valid = False
    if not valid or data["id"] != _outline_id(data):
        raise _refuse("outline.json: invalid story outline.", "Rebuild the outline first.")
    return data


def _current(root: Path, *, fresh: bool = False) -> dict:
    outline = _load_outline(root)
    if outline is None:
        raise _refuse("No story outline exists.", "Rebuild the outline first.")
    if fresh and (changed := _changed_input(root, outline)) is not None:
        error = _refuse(f"{changed} changed since the outline was built.",
                        "Rebuild the outline first.")
        error.code = "outline_changed"
        raise error
    return outline


def _changed_input(root: Path, outline: dict) -> str | None:
    for name, expected in sorted(outline["inputs"].items()):
        path = root / "ingest" / name.split(":")[1] / "words.json" if (
            name.startswith("words:")
        ) else root / name
        try:
            digest = sha256_file(path)
        except FileNotFoundError:
            digest = None
        except OSError as exc:
            raise _refuse(f"{path.relative_to(root)}: cannot read the current input.") from exc
        if digest != expected:
            return str(path.relative_to(root))
    return None


def _load_notes(root: Path) -> dict:
    path = root / "outline" / "notes.json"
    if not path.exists():
        return {"schema": NOTES_SCHEMA, "next_id": 1, "notes": []}
    data = _read(path)
    valid = (_keys(data, {"schema", "next_id", "notes"}) and data["schema"] == NOTES_SCHEMA
             and type(data["next_id"]) is int and data["next_id"] >= 1
             and isinstance(data["notes"], list))
    ids = set()
    for note in data["notes"] if valid else ():
        if (not _keys(note, {"id", "outline", "beat", "kind", "text", "created_at",
                            "updated_at", "batch"})
                or not isinstance(note["id"], str)
                or (match := _NOTE_ID.fullmatch(note["id"])) is None
                or int(match[1]) >= data["next_id"] or note["id"] in ids
                or not _hash(note["outline"])
                or note["beat"] is not None and (
                    not isinstance(note["beat"], str) or not note["beat"])
                or note["kind"] not in ("keep", "fix") or not isinstance(note["text"], str)
                or note["text"] != note["text"].strip() or not 1 <= len(note["text"]) <= 5000
                or not _time(note["created_at"]) or not _time(note["updated_at"])
                or datetime.fromisoformat(note["updated_at"])
                < datetime.fromisoformat(note["created_at"])
                or note["batch"] is not None and (
                    not isinstance(note["batch"], str)
                    or _BATCH_ID.fullmatch(note["batch"]) is None)):
            valid = False
            break
        ids.add(note["id"])
    if not valid:
        raise _refuse("notes.json: invalid outline notes.",
                      "Restore the notes file from a valid copy and retry.")
    return data


def _note_fields(outline: dict, beat: str | None, kind: str, text: str) -> str:
    if kind not in ("keep", "fix"):
        raise _refuse("kind must be keep or fix.")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 5000:
        raise _refuse("text must contain 1 to 5000 characters after stripping spaces.")
    if beat is not None:
        if not isinstance(beat, str) or not any(
            entry["id"] == beat for entry in outline["beats"] + outline["left_out"]
        ):
            raise _refuse("beat must exist in the current outline.")
        if kind == "keep" and any(entry["id"] == beat for entry in outline["left_out"]):
            raise _refuse("A left-out entry needs a fix note, not a keep.")
    return text.strip()


def add_note(project_dir: Path, *, beat: str | None, kind: Literal["keep", "fix"],
             text: str) -> dict:
    """Add a creator note with an id that will never be reused."""
    root = Path(project_dir)
    with project_lock(root):
        outline, data = _current(root), _load_notes(root)
        text = _note_fields(outline, beat, kind, text)
        now = _timestamp(_now())
        note = {"id": f"o{data['next_id']}", "outline": outline["id"], "beat": beat,
                "kind": kind, "text": text, "created_at": now, "updated_at": now, "batch": None}
        data["next_id"] += 1
        data["notes"].append(note)
        _write(root / "outline" / "notes.json", data)
        return note


def _editable(data: dict, note_id: str) -> dict:
    note = next((note for note in data["notes"] if note["id"] == note_id), None)
    if note is None:
        raise _refuse(f"Unknown outline note {note_id}.")
    if note["batch"] is not None:
        raise _refuse(f"{note_id} was sent in {note['batch']} and is locked.",
                      "Add a new note on the current outline.")
    return note


def update_note(project_dir: Path, note_id: str, **fields) -> dict:
    """Update an unsent note's beat, kind, or text on its current outline."""
    if not fields or set(fields) - {"beat", "kind", "text"}:
        raise _refuse("Only beat, kind, and text can be updated.")
    root = Path(project_dir)
    with project_lock(root):
        outline, data = _current(root), _load_notes(root)
        note = _editable(data, note_id)
        if note["outline"] != outline["id"]:
            raise _refuse("The note belongs to an older outline.",
                          "Add a new note on the current outline.")
        updated = {**note, **fields}
        updated["text"] = _note_fields(outline, updated["beat"], updated["kind"], updated["text"])
        updated["updated_at"] = _timestamp(_now())
        note.update(updated)
        _write(root / "outline" / "notes.json", data)
        return note


def delete_note(project_dir: Path, note_id: str) -> dict:
    """Delete an unsent note without moving the next id backwards."""
    root = Path(project_dir)
    with project_lock(root):
        data = _load_notes(root)
        note = _editable(data, note_id)
        data["notes"].remove(note)
        _write(root / "outline" / "notes.json", data)
        return note


def list_notes(project_dir: Path) -> list[dict]:
    """Read all outline notes, including those on earlier outlines."""
    return _load_notes(Path(project_dir))["notes"]


def _location(outline: dict, note: dict, sources: dict[str, dict[str, Word]]) -> Location:
    if note["beat"] is None:
        return Location("whole_video", 0.0, 0.0, None, None, (), None, None, (), (), False)
    entry = next(entry for entry in outline["beats"] + outline["left_out"]
                 if entry["id"] == note["beat"])
    words = []
    for pair in entry["words"]:
        source = sources[entry["take"]].get(pair["id"])
        if source is None or source.text != pair["text"]:
            raise _refuse(f"{entry['take']}: source words changed since the outline was built.",
                          "Rebuild the outline first.")
        words.append(Occurrence(source.id, entry["id"], source.text, source.start, source.end))
    segments = (entry["id"],) if entry in outline["beats"] else ()
    return Location("located", 0.0, 0.0, None, None, tuple(words), None, None,
                    segments, (), False)


def send_notes(project_dir: Path) -> dict:
    """Send current unsent notes through the normal numbered feedback batch path."""
    root = Path(project_dir)
    with project_lock(root):
        outline, data = _current(root, fresh=True), _load_notes(root)
        notes = [note for note in data["notes"]
                 if note["outline"] == outline["id"] and note["batch"] is None]
        if not notes:
            raise _refuse("No unsent notes on the current outline.", "Add notes before sending.")
        numbers = _batch_numbers(root)
        for number in sorted(numbers):
            batch = load_batch(root, f"b{number}")
            if batch.get("base") != "outline" or batch["review"].get("outline") != outline["id"]:
                continue
            revisions = {(item["comment"]["id"], item["comment"]["revision"])
                         for item in batch["items"] if item["comment"]["version"] == "outline"}
            recovered = [note for note in notes if (note["id"], note["updated_at"]) in revisions]
            if recovered:
                for note in recovered:
                    note["batch"] = batch["id"]
                _write(root / "outline" / "notes.json", data)
                return batch
        sources = {}
        edit = load_edit(root / "edit.json")
        for note in notes:
            _note_fields(outline, note["beat"], note["kind"], note["text"])
        for take in {entry["take"] for entry in outline["beats"] + outline["left_out"]}:
            words = _source_words(root, take, edit.words_sha256.get(take))
            sources[take] = {word.id: word for word in words.words}
        items, constraints = [], []
        for note in notes:
            comment = Comment(note["id"], note["kind"], "story", note["text"], None, None,
                              0.0, None, note["created_at"], note["updated_at"])
            location = _location(outline, note, sources)
            constraint = keep_constraint(comment, location, version="outline") if (
                note["kind"] == "keep" and note["beat"] is not None
            ) else None
            if constraint is not None:
                constraints.append(constraint)
            items.append({"comment": {"version": "outline", "id": note["id"],
                                      "revision": note["updated_at"], "kind": note["kind"],
                                      "target": "story", "text": note["text"], "request": None,
                                      "start_frame": None, "end_frame": None,
                                      "start_s": 0.0, "end_s": None},
                          "request": None, "carried": False, "location": location.to_dict(),
                          "constraint": constraint, "clarification": None, "unlocated": None,
                          "disposition": "open", "operations": [], "evidence": None})
        batch_id = f"b{max(numbers, default=0) + 1}"
        now = _timestamp(_now())
        batch = {"schema": BATCH_SCHEMA, "id": batch_id, "project": outline["project"],
                 "base": "outline", "base_version": None, "created_at": now,
                 "review": {"media_sha256": None, "media_name": "story outline",
                            "reviewer": None, "exported_at": now, "series_id": None,
                            "outline": outline["id"]},
                 "expected_inputs": _inputs(root, require_edit=True), "items": items,
                 "constraints": constraints, "released": [], "resulting_version": None}
        validate_output(root / "outline" / "notes.json")
        validate_output(_batch_path(root, batch_id))
        _write_batch(root, batch)
        for note in notes:
            note["batch"] = batch_id
        _write(root / "outline" / "notes.json", data)
        return batch


def _load_state(root: Path) -> dict:
    path = root / "outline" / "state.json"
    if not path.exists():
        return {"schema": STATE_SCHEMA, "outline": None, "approved_at": None,
                "deadline": None, "wait_minutes": None}
    data = _read(path)
    if _keys(data, {"schema", "outline", "approved_at", "deadline"}):
        data["wait_minutes"] = None
    if (not _keys(data, {"schema", "outline", "approved_at", "deadline", "wait_minutes"})
            or data["schema"] != STATE_SCHEMA
            or data["outline"] is not None and not _hash(data["outline"])
            or any(data[key] is not None and not _time(data[key])
                   for key in ("approved_at", "deadline"))
            or data["wait_minutes"] is not None and (
                type(data["wait_minutes"]) is not int or not 1 <= data["wait_minutes"] <= 10080
                or data["deadline"] is None)
            or data["outline"] is None and (
                data["approved_at"] is not None or data["deadline"] is not None)):
        raise _refuse("state.json: invalid outline state.",
                      "Restore the state file from a valid copy and retry.")
    return data


def _current_state(root: Path, outline: dict) -> dict:
    state = _load_state(root)
    if state["outline"] != outline["id"]:
        state = {"schema": STATE_SCHEMA, "outline": outline["id"],
                 "approved_at": None, "deadline": None, "wait_minutes": None}
    return state


def approve_outline(project_dir: Path) -> dict:
    """Record creator approval of the current, unchanged outline."""
    root = Path(project_dir)
    with project_lock(root):
        outline = _current(root, fresh=True)
        state = _current_state(root, outline)
        state["approved_at"] = _timestamp(_now())
        _write(root / "outline" / "state.json", state)
        return state


def set_wait(project_dir: Path, minutes: int | None) -> dict:
    """Set or clear the creator's deadline for the current outline only."""
    if minutes is not None and (type(minutes) is not int or not 1 <= minutes <= 10080):
        raise _refuse("minutes must be null or an integer from 1 to 10080.")
    root = Path(project_dir)
    with project_lock(root):
        outline = _current(root, fresh=True)
        state = _current_state(root, outline)
        state["deadline"] = _timestamp(_now() + timedelta(minutes=minutes)) if (
            minutes is not None
        ) else None
        state["wait_minutes"] = minutes
        _write(root / "outline" / "state.json", state)
        return state


def outline_status(project_dir: Path, *, now=None) -> dict:
    """Report whether approval or the creator's timeout permits rendering."""
    root = Path(project_dir)
    outline = _load_outline(root)
    result = {"outline": outline["id"] if outline else None, "state": "missing",
              "approved_at": None, "deadline": None, "wait_minutes": None,
              "open_notes": 0, "sent_notes": 0,
              "edit_changed": False}
    if outline is None:
        return result
    state = _current_state(root, outline)
    notes = [note for note in list_notes(root) if note["outline"] == outline["id"]]
    changed = _changed_input(root, outline)
    result.update(approved_at=state["approved_at"], deadline=state["deadline"],
                  wait_minutes=state["wait_minutes"],
                  open_notes=sum(note["batch"] is None for note in notes),
                  sent_notes=sum(note["batch"] is not None for note in notes),
                  edit_changed=changed == "edit.json")
    moment = _utc(now() if callable(now) else now if now is not None else _now())
    if changed is not None:
        result["state"] = "stale"
    elif result["approved_at"] is not None:
        result["state"] = "approved"
    elif any(note["batch"] is not None and note["kind"] == "fix" for note in notes):
        result["state"] = "notes_sent"
    elif result["deadline"] is not None and moment >= datetime.fromisoformat(result["deadline"]):
        result["state"] = "timed_out"
    else:
        result["state"] = "waiting"
    return result
