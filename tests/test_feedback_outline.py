"""Story outlines and creator notes work with synthetic words and no media."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio.cli import main
from yap_studio.feedback import outline as outline_module
from yap_studio.feedback.batch import BATCH_SCHEMA, FeedbackError, _inputs, load_batch
from yap_studio.feedback.keeps import check_keeps
from yap_studio.feedback.locate import Location
from yap_studio.feedback.moments import Occurrence
from yap_studio.feedback.outline import (
    NOTES_SCHEMA,
    OUTLINE_SCHEMA,
    STATE_SCHEMA,
    add_note,
    approve_outline,
    build_outline,
    delete_note,
    list_notes,
    outline_status,
    send_notes,
    set_wait,
    update_note,
)
from yap_studio.project.anchors import nominal_edge
from yap_studio.project.model import CutWord, CutWords, load_edit, load_words
from yap_studio.receipt import sha256_file
from yap_studio.stages.cut import Piece, _segment_piece_words, _segment_word_ids

NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)


def _json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def _words(root: Path, take: str, spans: list[tuple[float, float]]) -> None:
    _json(root / "ingest" / take / "words.json", {
        "schema": "yap-studio/words@1", "take": take, "take_sha256": "a" * 64,
        "time_base": "source", "audio_offset_s": 0, "engine": "none", "model": None,
        "transcript_sha256": "b" * 64,
        "words": [{"id": f"{take}:{i}", "text": f"word{i}", "start": start, "end": end}
                  for i, (start, end) in enumerate(spans)],
    })


def _segment(id: str, first: int, last: int, take: str = "t1", **fields) -> dict:
    return {"id": id, "take": take, "from": {"word": f"{take}:{first}"},
            "to": {"word": f"{take}:{last}"}, **fields}


def _edit(root: Path, segments: list[dict], **fields) -> None:
    _json(root / "edit.json", {
        "schema": "yap-studio/edit@1",
        "words_sha256": {take: sha256_file(root / "ingest" / take / "words.json")
                         for take in ("t1", "t2")},
        "segments": segments, **fields,
    })


@pytest.fixture
def outline_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "story"
    _json(root / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample-story",
        "takes": [{"id": take, "path": f"takes/{take}.mov"} for take in ("t1", "t2")],
    })
    for take, count in (("t1", 21), ("t2", 4)):
        _words(root, take, [(i * 0.5 + 0.2, i * 0.5 + 0.4) for i in range(count)])
    _edit(root, [_segment("s1", 2, 4, note="Introduce the idea"),
                 _segment("s2", 8, 10), _segment("s3", 14, 16)])
    monkeypatch.setattr(outline_module, "_now", lambda: NOW)
    return root


def test_outline_beats_roles_nominal_spans_and_applied_removals(outline_project: Path) -> None:
    root = outline_project
    _json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": f"r{i}", "take": "t1", "start": index * 0.5 + 0.2,
             "end": index * 0.5 + 0.4, "kind": "manual", "status": status}
            for i, (index, status) in enumerate([(3, "applied"), (9, "proposed"),
                                                (15, "rejected")])
        ],
    })
    result = build_outline(root, summary="Start with an idea and show why it matters.")
    assert set(result) == {"schema", "id", "project", "edit_sha256", "inputs", "created_at",
                           "summary", "beats", "left_out", "left_out_truncated", "duration_s"}
    assert result["schema"] == OUTLINE_SCHEMA and result["project"] == "sample-story"
    assert result["edit_sha256"] == sha256_file(root / "edit.json")
    assert result["inputs"] == {
        "edit.json": result["edit_sha256"], "removals.json": sha256_file(root / "removals.json"),
        **{f"words:{take}": sha256_file(root / "ingest" / take / "words.json")
           for take in ("t1", "t2")},
    }
    assert result["created_at"] == "2026-10-06T12:00:00Z"
    assert result == json.loads((root / "outline" / "outline.json").read_text())
    assert [beat["role"] for beat in result["beats"]] == ["opening", "beat", "ending"]
    assert [beat["text"] for beat in result["beats"]] == [
        "word2 word4", "word8 word9 word10", "word14 word15 word16",
    ]
    assert result["beats"][0]["note"] == "Introduce the idea"
    assert result["beats"][1]["note"] is None
    edit, words = load_edit(root / "edit.json"), load_words(root / "ingest/t1/words.json")
    for beat, segment in zip(result["beats"], edit.segments, strict=True):
        assert set(beat) == {"id", "role", "take", "start", "end", "note", "text", "words"}
        assert beat["id"] == segment.id and beat["take"] == "t1"
        assert beat["start"] == float(nominal_edge(segment, segment.from_, words, None, start=True))
        assert beat["end"] == float(nominal_edge(segment, segment.to, words, None, start=False))
        assert beat["words"] == [{"id": word.id, "text": word.text} for word in words.words
                                 if beat["start"] <= (word.start + word.end) / 2 < beat["end"]
                                 and word.id != "t1:3"]
    assert result["beats"][0]["start"] == pytest.approx(1.08)
    assert result["beats"][0]["end"] == pytest.approx(2.55)
    assert result["duration_s"] == round(sum(b["end"] - b["start"] for b in result["beats"]), 2)
    assert "t1:3" not in {w["id"] for entry in result["left_out"] for w in entry["words"]}


def test_outline_midpoints_half_open_and_single_opening(outline_project: Path) -> None:
    root = outline_project
    _words(root, "t1", [(i * 0.5, (i + 1) * 0.5) for i in range(4)])
    _edit(root, [{"id": "s1", "take": "t1", "from": {"time": 0.25}, "to": {"time": 1.25}}])
    outline = build_outline(root)
    assert outline["beats"] == [{
        "id": "s1", "role": "opening", "take": "t1", "start": 0.25, "end": 1.25,
        "note": None, "text": "word0 word1",
        "words": [{"id": "t1:0", "text": "word0"}, {"id": "t1:1", "text": "word1"}],
    }]
    _edit(root, [_segment("s1", 1, 2)])
    assert [beat["role"] for beat in build_outline(root)["beats"]] == ["opening"]


@pytest.mark.parametrize("from_word,to_word", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("removed", [False, True])
@pytest.mark.parametrize("shifted", [False, True])
def test_outline_anchored_words_match_cut_membership_in_file_order(
    outline_project: Path, from_word: bool, to_word: bool, removed: bool, shifted: bool,
) -> None:
    root = outline_project
    _words(root, "t1", [(0, .2), (1, 2.4), (1.6, 1.8), (1.4, 1.5),
                        (3, 3.2), (2.5, 3.1)])
    _edit(root, [{
        "id": "s1", "take": "t1",
        "from": {"word": "t1:2", "offset_s": .3 if shifted else 0}
        if from_word else {"time": 1.9},
        "to": {"word": "t1:4", "offset_s": -.3 if shifted else 0}
        if to_word else {"time": 2.9},
    }])
    if removed:
        _json(root / "removals.json", {"schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 1.4, "end": 1.5,
             "kind": "manual", "status": "applied"},
        ]})
    outline = build_outline(root)
    beat = outline["beats"][0]
    expected = ["t1:2", "t1:3", "t1:4"] if from_word and to_word else (
        ["t1:2", "t1:3", "t1:5"] if from_word else ["t1:4"]
    )
    if removed:
        expected = [word for word in expected if word != "t1:3"]
    assert [word["id"] for word in beat["words"]] == expected
    segment = load_edit(root / "edit.json").segments[0]
    words = load_words(root / "ingest/t1/words.json")
    piece = Piece("s1", "t1", Fraction(str(beat["start"])), Fraction(str(beat["end"])),
                  0, 0, 0, 0, 0, 0, (), (), False, False)
    removals = [(Fraction("1.4"), Fraction("1.5"))] if removed else []
    planned = _segment_word_ids(segment, words, piece.src_t0, piece.src_t1)
    cut_words = _segment_piece_words(segment, words, [piece], removals,
                                     planned_ids=planned)[0].words
    assert tuple(expected) == cut_words
    add_note(root, beat="s1", kind="keep", text="Keep this beat.")
    batch = send_notes(root)
    unchanged_cut = CutWords("output", {}, tuple(
        CutWord(word, "s1", f"word{word.split(':')[1]}", i * .4, i * .4 + .2)
        for i, word in enumerate(cut_words)
    ))
    assert check_keeps(batch["constraints"], unchanged_cut, None, fps=Fraction(30)).ok


def test_outline_left_out_runs_unused_take_and_union_of_repeated_beats(
    outline_project: Path,
) -> None:
    root = outline_project
    result = build_outline(root)
    assert [(entry["id"], entry["text"]) for entry in result["left_out"]] == [
        ("t1.0", "word5 word6 word7"), ("t1.1", "word11 word12 word13"),
        ("t1.2", "word17 word18 word19 word20"), ("t2.0", "word0 word1 word2 word3"),
    ]
    assert all(set(entry) == {"id", "take", "start", "end", "text", "words"}
               for entry in result["left_out"])
    assert result["left_out"][-1]["start"] == 0.2
    assert result["left_out"][-1]["end"] == 1.9
    assert result["left_out_truncated"] is False
    segments = json.loads((root / "edit.json").read_text())["segments"]
    _edit(root, segments + [_segment("repeated", 2, 4), _segment("overlap", 4, 7)])
    outline = build_outline(root)
    assert outline["beats"][-2]["text"] == "word2 word3 word4"
    assert [entry["id"] for entry in outline["left_out"]] == ["t1.0", "t1.1", "t2.0"]
    assert outline["left_out"][0]["text"] == "word11 word12 word13"


def test_outline_left_out_three_words_or_two_seconds_and_removal_splits(
    outline_project: Path,
) -> None:
    root = outline_project
    spans = [(0, .2), (.3, .4), (.5, .6), (.7, .9), (1, 1.1), (3, 3.1),
             (3.2, 3.4), (3.5, 3.6), (3.7, 3.8), (3.9, 4), (4.1, 4.3),
             (4.4, 4.5), (6.3, 6.4), (6.6, 6.8), (6.9, 7), (7.1, 7.2), (7.3, 7.5)]
    _words(root, "t1", spans)
    _edit(root, [_segment(f"s{i}", index, index)
                 for i, index in enumerate((0, 3, 6, 10, 13))])
    entries = build_outline(root)["left_out"]
    assert [entry["text"] for entry in entries if entry["take"] == "t1"] == [
        "word4 word5", "word7 word8 word9", "word11 word12", "word14 word15 word16",
    ]
    _json(root / "removals.json", {"schema": "yap-studio/removals@1", "removals": [
        {"id": "r1", "take": "t1", "start": 3.7, "end": 3.8,
         "kind": "manual", "status": "applied"},
    ]})
    assert "word7 word8 word9" not in [entry["text"] for entry in build_outline(root)["left_out"]]


def test_outline_left_out_truncates_words_and_entries(outline_project: Path) -> None:
    root = outline_project
    _words(root, "t2", [(i, i + .2) for i in range(401)])
    _edit(root, [_segment("s1", 2, 4)])
    result = build_outline(root)
    assert result["left_out_truncated"] is True
    whole = result["left_out"][-1]
    assert whole["take"] == "t2" and len(whole["words"]) == 400
    assert whole["end"] == 400.2 and whole["text"].endswith("word400")
    _words(root, "t1", [(i, i + .2) for i in range(805)])
    _edit(root, [_segment(f"s{i}", index, index) for i, index in enumerate(range(0, 805, 4))])
    result = build_outline(root)
    assert len(result["left_out"]) == 200 and result["left_out_truncated"] is True
    assert result["left_out"][0]["id"] == "t1.0"
    assert result["left_out"][-1]["id"] == "t1.199"


def test_outline_unused_take_is_whole_even_when_short(outline_project: Path) -> None:
    root = outline_project
    _words(root, "t2", [(0, .2)])
    _edit(root, [_segment("s1", 2, 4)])
    _json(root / "removals.json", {"schema": "yap-studio/removals@1", "removals": [
        {"id": "r1", "take": "t2", "start": 0, "end": .2,
         "kind": "manual", "status": "applied"},
    ]})
    assert build_outline(root)["left_out"][-1] == {
        "id": "t2.0", "take": "t2", "start": 0, "end": .2, "text": "word0",
        "words": [{"id": "t2:0", "text": "word0"}],
    }


def test_outline_id_canonical_stable_and_changes_with_summary_or_edit(
    outline_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = outline_project
    first = build_outline(root)
    canonical = json.dumps({k: v for k, v in first.items() if k not in {"id", "created_at"}},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert first["id"] == hashlib.sha256(canonical.encode()).hexdigest()
    monkeypatch.setattr(outline_module, "_now", lambda: NOW + timedelta(minutes=1))
    second = build_outline(root)
    assert first["id"] == second["id"] and first["created_at"] != second["created_at"]
    assert build_outline(root, summary="A clearer opening")["id"] != first["id"]
    _edit(root, [_segment("s1", 2, 5)])
    assert build_outline(root)["id"] != first["id"]


@pytest.mark.parametrize("take", ["t1", "t2"])
def test_outline_words_hash_mismatch_names_take_and_preserves_outline(
    outline_project: Path, take: str,
) -> None:
    root = outline_project
    build_outline(root)
    path = root / "outline" / "outline.json"
    before = path.read_bytes()
    words_path = root / "ingest" / take / "words.json"
    words_path.write_bytes(words_path.read_bytes() + b"\n")
    with pytest.raises(FeedbackError, match=take) as error:
        build_outline(root)
    assert error.value.suggested_action == "Run ingest again or update the edit."
    assert path.read_bytes() == before
    edit = json.loads((root / "edit.json").read_text())
    edit["words_sha256"].pop(take)
    _json(root / "edit.json", edit)
    assert build_outline(root)["beats"]


@pytest.mark.parametrize("summary", ["x" * 1001, 1])
def test_outline_summary_limit(outline_project: Path, summary) -> None:
    with pytest.raises(FeedbackError, match="1000"):
        build_outline(outline_project, summary=summary)
    assert build_outline(outline_project, summary="x" * 1000)["summary"] == "x" * 1000


def test_outline_hash_guard_precedes_parse_and_applies_when_sending(outline_project: Path) -> None:
    root = outline_project
    build_outline(root)
    add_note(root, beat="s1", kind="keep", text="Keep this.")
    _json(root / "ingest/t1/words.json", {"schema": "invalid"})
    before = (root / "outline/notes.json").read_bytes()
    for action in (lambda: build_outline(root), lambda: send_notes(root)):
        with pytest.raises(FeedbackError, match="t1"):
            action()
    assert (root / "outline/notes.json").read_bytes() == before
    assert not (root / "feedback").exists()


def test_outline_notes_add_update_delete_ids_never_reused(
    outline_project: Path, monkeypatch,
) -> None:
    root = outline_project
    outline = build_outline(root)
    assert list_notes(root) == []
    first = add_note(root, beat="s1", kind="keep", text="  Keep this opening.  ")
    assert first == {"id": "o1", "outline": outline["id"], "beat": "s1", "kind": "keep",
                     "text": "Keep this opening.", "created_at": "2026-10-06T12:00:00Z",
                     "updated_at": "2026-10-06T12:00:00Z", "batch": None}
    monkeypatch.setattr(outline_module, "_now", lambda: NOW + timedelta(seconds=1))
    updated = update_note(root, "o1", beat="t1.0", kind="fix", text="  Put this back. ")
    assert updated["created_at"] == first["created_at"]
    assert updated["updated_at"] == "2026-10-06T12:00:01Z"
    assert updated["text"] == "Put this back." and updated["beat"] == "t1.0"
    assert delete_note(root, "o1") == updated
    second = add_note(root, beat=None, kind="fix", text="Reorder the story.")
    assert second["id"] == "o2" and list_notes(root) == [second]
    assert json.loads((root / "outline/notes.json").read_text()) == {
        "schema": NOTES_SCHEMA, "next_id": 3, "notes": [second],
    }
    with pytest.raises(FeedbackError, match="Unknown"):
        delete_note(root, "o1")


@pytest.mark.parametrize("fields,match", [
    ({"beat": "missing"}, "current outline"), ({"beat": 3}, "current outline"),
    ({"beat": "t1.0", "kind": "keep"}, "left-out"), ({"kind": "other"}, "kind"),
    ({"text": " \n "}, "5000"), ({"text": "x" * 5001}, "5000"), ({"text": None}, "5000"),
])
def test_outline_notes_validate_add_and_update(outline_project: Path, fields, match) -> None:
    root = outline_project
    build_outline(root)
    note = add_note(root, beat="s1", kind="fix", text="Change the opening.")
    before = (root / "outline/notes.json").read_bytes()
    with pytest.raises(FeedbackError, match=match):
        add_note(root, **{"beat": "s1", "kind": "fix", "text": "Fix it.", **fields})
    with pytest.raises(FeedbackError, match=match):
        update_note(root, note["id"], **fields)
    assert (root / "outline/notes.json").read_bytes() == before


def test_outline_notes_only_editable_fields_and_old_outline_not_rebound(
    outline_project: Path,
) -> None:
    root = outline_project
    build_outline(root)
    note = add_note(root, beat="s1", kind="fix", text="x" * 5000)
    for fields in ({"batch": "b1"}, {"id": "o9"}, {"outline": "a" * 64}, {}):
        with pytest.raises(FeedbackError, match="Only"):
            update_note(root, note["id"], **fields)
    build_outline(root, summary="A new story")
    with pytest.raises(FeedbackError, match="older outline"):
        update_note(root, note["id"], text="Different note")
    assert list_notes(root) == [note]


@pytest.mark.parametrize("corruption", [
    "json", "duplicate_key", "schema", "extra", "next_id", "bool_id", "duplicate_note",
    "outline", "kind", "text", "created_at", "updated_at", "batch", "note_extra",
])
def test_outline_corrupt_notes_refused_untouched(outline_project: Path, corruption: str) -> None:
    root = outline_project
    build_outline(root)
    add_note(root, beat="s1", kind="fix", text="Fix it.")
    path = root / "outline/notes.json"
    data = json.loads(path.read_text())
    if corruption == "json":
        path.write_text("{invalid")
    elif corruption == "duplicate_key":
        path.write_text('{"schema": "bad", ' + path.read_text()[1:])
    else:
        if corruption == "schema":
            data["schema"] = "other"
        elif corruption == "extra":
            data["extra"] = True
        elif corruption == "next_id":
            data["next_id"] = 1
        elif corruption == "bool_id":
            data["next_id"] = True
        elif corruption == "duplicate_note":
            data["notes"].append(copy.deepcopy(data["notes"][0]))
        else:
            key, value = {"outline": ("outline", "bad"), "kind": ("kind", "other"),
                          "text": ("text", "  not stripped "), "created_at": ("created_at", "bad"),
                          "updated_at": ("updated_at", "2020-01-01T00:00:00Z"),
                          "batch": ("batch", "b0"), "note_extra": ("extra", True)}[corruption]
            data["notes"][0][key] = value
        _json(path, data)
    before = path.read_bytes()
    for action in (lambda: list_notes(root),
                   lambda: add_note(root, beat=None, kind="fix", text="x"),
                   lambda: update_note(root, "o1", text="x"), lambda: delete_note(root, "o1"),
                   lambda: send_notes(root)):
        with pytest.raises(FeedbackError, match="notes.json"):
            action()
        assert path.read_bytes() == before
    assert not (root / "feedback").exists()


def test_outline_send_batch_shape_locations_constraints_and_numbering(
    outline_project: Path,
) -> None:
    root = outline_project
    outline = build_outline(root)
    notes = [add_note(root, beat=beat, kind=kind, text=text) for beat, kind, text in (
        ("s1", "keep", "Keep this opening."), ("s2", "fix", "Explain this sooner."),
        ("t1.0", "fix", "Put this back."), (None, "fix", "Make the story clearer."),
        (None, "keep", "Keep the main idea."),
    )]
    _json(root / "feedback/b1.json", {"schema": BATCH_SCHEMA, "id": "b1", "items": [],
                                      "constraints": [], "expected_inputs": {}})
    batch = send_notes(root)
    assert load_batch(root, "b2") == batch
    assert set(batch) == {"schema", "id", "project", "base", "base_version", "created_at",
                          "review", "expected_inputs", "items", "constraints", "released",
                          "resulting_version"}
    assert batch["schema"] == BATCH_SCHEMA and batch["id"] == "b2"
    assert batch["project"] == "sample-story" and batch["base"] == "outline"
    assert batch["base_version"] is None and batch["created_at"] == "2026-10-06T12:00:00Z"
    assert batch["review"] == {"media_sha256": None, "media_name": "story outline",
                               "reviewer": None, "exported_at": batch["created_at"],
                               "series_id": None, "outline": outline["id"]}
    assert batch["expected_inputs"] == _inputs(root, require_edit=True)
    assert batch["released"] == [] and batch["resulting_version"] is None
    source = {word.id: word for word in load_words(root / "ingest/t1/words.json").words}
    for item, note in zip(batch["items"], notes, strict=True):
        assert set(item) == {"comment", "request", "carried", "location", "constraint",
                             "clarification", "unlocated", "disposition", "operations", "evidence"}
        assert item["comment"] == {"version": "outline", "id": note["id"],
                                   "revision": note["updated_at"], "kind": note["kind"],
                                   "target": "story", "text": note["text"], "request": None,
                                   "start_frame": None, "end_frame": None, "start_s": 0.0,
                                   "end_s": None}
        assert item["request"] is None and item["carried"] is False
        assert item["clarification"] is None and item["unlocated"] is None
        assert item["disposition"] == "open" and item["operations"] == []
        assert item["evidence"] is None
        if note["beat"] is None:
            location = Location("whole_video", 0.0, 0.0, None, None, (), None, None, (), (), False)
        else:
            entry = next(e for e in outline["beats"] + outline["left_out"]
                         if e["id"] == note["beat"])
            words = tuple(Occurrence(w["id"], entry["id"], w["text"], source[w["id"]].start,
                                     source[w["id"]].end) for w in entry["words"])
            segments = (note["beat"],) if entry in outline["beats"] else ()
            location = Location("located", 0.0, 0.0, None, None, words, None, None,
                                segments, (), False)
        assert item["location"] == location.to_dict()
    assert batch["constraints"] == [batch["items"][0]["constraint"]]
    assert all(item["constraint"] is None for item in batch["items"][1:])
    cut_words = CutWords("output", {}, tuple(
        CutWord(w["id"], "s1", w["text"], i * .4, i * .4 + .2)
        for i, w in enumerate(outline["beats"][0]["words"])
    ))
    kept = check_keeps(batch["constraints"], cut_words, None, fps=Fraction(30))
    assert kept.ok and kept.results[0].status == "kept"
    missing = CutWords("output", {}, cut_words.words[:-1])
    broken = check_keeps(batch["constraints"], missing, None, fps=Fraction(30))
    assert not broken.ok and broken.results[0].status == "broken"
    wrong_segment = CutWords("output", {}, tuple(
        CutWord(w.id, "other", w.text, w.start, w.end) for w in cut_words.words
    ))
    assert not check_keeps(batch["constraints"], wrong_segment, None, fps=Fraction(30)).ok
    assert list_notes(root) == [{**note, "batch": "b2"} for note in notes]
    before = (root / "outline/notes.json").read_bytes()
    for action in (lambda: update_note(root, "o1", text="Changed"),
                   lambda: delete_note(root, "o1")):
        with pytest.raises(FeedbackError, match="locked"):
            action()
    assert (root / "outline/notes.json").read_bytes() == before
    with pytest.raises(FeedbackError, match="No unsent"):
        send_notes(root)
    add_note(root, beat="s3", kind="fix", text="Finish sooner.")
    assert send_notes(root)["id"] == "b3"


def test_outline_send_refuses_changed_edit_and_skips_older_notes(outline_project: Path) -> None:
    root = outline_project
    build_outline(root)
    old = add_note(root, beat="s1", kind="fix", text="Old note.")
    _edit(root, [_segment("s1", 2, 5)])
    before = (root / "outline/notes.json").read_bytes()
    with pytest.raises(FeedbackError, match="edit.json changed") as error:
        send_notes(root)
    assert error.value.suggested_action == "Rebuild the outline first."
    assert (root / "outline/notes.json").read_bytes() == before
    assert not (root / "feedback").exists()
    build_outline(root)
    with pytest.raises(FeedbackError, match="No unsent"):
        send_notes(root)
    current = add_note(root, beat="s1", kind="fix", text="Current note.")
    assert [item["comment"]["id"] for item in send_notes(root)["items"]] == [current["id"]]
    assert list_notes(root)[0] == old


def test_outline_interrupted_send_reuses_batch_and_marks_notes(
    outline_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = outline_project
    build_outline(root)
    notes = [add_note(root, beat=beat, kind=kind, text="Creator note.")
             for beat, kind in [("s1", "keep"), (None, "fix")]]
    write = outline_module._write

    def fail_notes(path, data):
        if path == root / "outline/notes.json":
            raise OSError("interrupted notes write")
        write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(outline_module, "_write", fail_notes)
        with pytest.raises(OSError, match="interrupted notes write"):
            send_notes(root)
    saved_batch = load_batch(root, "b1")
    assert list_notes(root) == notes
    retried = send_notes(root)
    assert retried["id"] == "b1"
    assert retried == saved_batch
    assert list((root / "feedback").glob("b*.json")) == [root / "feedback/b1.json"]
    assert list_notes(root) == [{**note, "batch": "b1"} for note in notes]


@pytest.mark.parametrize("mismatch", ["base", "outline", "revision"])
def test_outline_send_reconciliation_requires_same_outline_and_revision(
    outline_project: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str,
) -> None:
    root = outline_project
    build_outline(root)
    note = add_note(root, beat="s1", kind="fix", text="Original note.")
    batch = send_notes(root)
    _json(root / "outline/notes.json", {"schema": NOTES_SCHEMA, "next_id": 2, "notes": [note]})
    if mismatch == "base":
        batch["base"] = "v1"
    elif mismatch == "outline":
        batch["review"]["outline"] = "0" * 64
    else:
        monkeypatch.setattr(outline_module, "_now", lambda: NOW + timedelta(seconds=1))
        note = update_note(root, note["id"], text="Revised note.")
    _json(root / "feedback/b1.json", batch)
    retried = send_notes(root)
    assert retried["id"] == "b2"
    assert retried["items"][0]["comment"]["revision"] == note["updated_at"]
    assert list_notes(root) == [{**note, "batch": "b2"}]


def test_outline_concurrent_notes_and_sends_are_serialized(outline_project: Path) -> None:
    root = outline_project
    build_outline(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        notes = list(pool.map(lambda i: add_note(root, beat=None, kind="fix", text=f"Note {i}"),
                              range(8)))
    assert {note["id"] for note in notes} == {f"o{i}" for i in range(1, 9)}

    def send(_):
        try:
            return send_notes(root)
        except FeedbackError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send, range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(load_batch(root, "b1")["items"]) == 8
    assert {note["batch"] for note in list_notes(root)} == {"b1"}


def test_outline_status_every_state_and_now_injection(outline_project: Path) -> None:
    root = outline_project
    assert outline_status(root) == {"outline": None, "state": "missing", "approved_at": None,
                                    "deadline": None, "wait_minutes": None,
                                    "open_notes": 0, "sent_notes": 0,
                                    "edit_changed": False}
    outline = build_outline(root)
    assert outline["inputs"]["removals.json"] is None
    assert outline_status(root)["state"] == "waiting"
    add_note(root, beat="s1", kind="fix", text="Change it.")
    assert outline_status(root)["open_notes"] == 1
    assert outline_status(root)["state"] == "waiting"
    send_notes(root)
    result = outline_status(root)
    assert result["outline"] == outline["id"] and result["state"] == "notes_sent"
    assert result["open_notes"] == 0 and result["sent_notes"] == 1
    state = set_wait(root, 5)
    assert state == {"schema": STATE_SCHEMA, "outline": outline["id"], "approved_at": None,
                     "deadline": "2026-10-06T12:05:00Z", "wait_minutes": 5}
    assert json.loads((root / "outline/state.json").read_text()) == state
    assert outline_status(root, now=NOW + timedelta(minutes=4, seconds=59))["state"] == "notes_sent"
    assert outline_status(root, now=NOW + timedelta(minutes=5))["state"] == "notes_sent"
    assert outline_status(root, now=lambda: NOW + timedelta(minutes=6))["state"] == "notes_sent"
    set_wait(root, None)
    assert outline_status(root, now=NOW + timedelta(days=1))["state"] == "notes_sent"
    assert outline_status(root)["deadline"] is None
    set_wait(root, 1)
    approval = approve_outline(root)
    assert approval["approved_at"] == "2026-10-06T12:00:00Z"
    assert approval["deadline"] == "2026-10-06T12:01:00Z"
    assert outline_status(root, now=NOW + timedelta(days=1))["state"] == "approved"
    _edit(root, [_segment("s1", 2, 5)])
    result = outline_status(root)
    assert result["state"] == "stale" and result["edit_changed"] is True


def test_outline_sent_keep_alone_does_not_block_timeout(outline_project: Path) -> None:
    root = outline_project
    build_outline(root)
    add_note(root, beat="s1", kind="keep", text="Keep this opening.")
    send_notes(root)
    set_wait(root, 5)
    assert outline_status(root)["state"] == "waiting"
    assert outline_status(root)["sent_notes"] == 1
    assert outline_status(root, now=NOW + timedelta(minutes=5))["state"] == "timed_out"
    add_note(root, beat=None, kind="fix", text="This draft is not sent yet.")
    assert outline_status(root, now=NOW + timedelta(minutes=6))["state"] == "timed_out"
    send_notes(root)
    assert outline_status(root, now=NOW + timedelta(minutes=6))["state"] == "notes_sent"


def test_outline_approval_and_wait_bound_to_outline_id(outline_project: Path) -> None:
    root = outline_project
    first = build_outline(root)
    set_wait(root, 1)
    approve_outline(root)
    add_note(root, beat=None, kind="fix", text="Older note.")
    send_notes(root)
    assert build_outline(root)["id"] == first["id"]
    assert outline_status(root)["state"] == "approved"
    build_outline(root, summary="New summary")
    assert outline_status(root, now=NOW + timedelta(days=1)) == {
        "outline": build_outline(root, summary="New summary")["id"], "state": "waiting",
        "approved_at": None, "deadline": None, "wait_minutes": None,
        "open_notes": 0, "sent_notes": 0,
        "edit_changed": False,
    }
    state = set_wait(root, 10080)
    assert state["approved_at"] is None
    assert outline_status(root, now=NOW + timedelta(days=7))["state"] == "timed_out"
    build_outline(root, summary="Another summary")
    state = approve_outline(root)
    assert state["deadline"] is None


@pytest.mark.parametrize("changed", ["removals.json", "words:t1", "words:t2"])
def test_outline_changed_inputs_invalidate_approval_and_actions(
    outline_project: Path, capsys, changed: str,
) -> None:
    root = outline_project
    edit = json.loads((root / "edit.json").read_text())
    edit["words_sha256"] = {}
    _json(root / "edit.json", edit)
    outline = build_outline(root)
    add_note(root, beat="s1", kind="fix", text="Change this beat.")
    set_wait(root, 15)
    approve_outline(root)
    before = {name: (root / "outline" / name).read_bytes()
              for name in ("notes.json", "state.json")}
    if changed == "removals.json":
        _json(root / changed, {"schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 1.7, "end": 1.9,
             "kind": "manual", "status": "applied"},
        ]})
    else:
        words_path = root / "ingest" / changed.split(":")[1] / "words.json"
        words_path.write_bytes(words_path.read_bytes() + b"\n")
    assert outline_status(root)["state"] == "stale"
    assert outline_status(root)["edit_changed"] is False
    assert main(["outline", "status", str(root), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["state"] == "stale"
    for action in (lambda: send_notes(root), lambda: approve_outline(root),
                   lambda: set_wait(root, 60)):
        with pytest.raises(FeedbackError, match="changed") as error:
            action()
        assert error.value.suggested_action == "Rebuild the outline first."
    assert all((root / "outline" / name).read_bytes() == content
               for name, content in before.items())
    assert not (root / "feedback").exists()
    rebuilt = build_outline(root)
    assert rebuilt["id"] != outline["id"]
    assert outline_status(root)["state"] == "waiting"


def test_outline_wait_minutes_persist_and_clear_with_deadline(outline_project: Path) -> None:
    root = outline_project
    build_outline(root)
    state = set_wait(root, 15)
    assert state.get("wait_minutes") == 15
    assert json.loads((root / "outline/state.json").read_text())["wait_minutes"] == 15
    assert outline_status(root)["wait_minutes"] == 15
    assert approve_outline(root)["wait_minutes"] == 15
    state = set_wait(root, None)
    assert state["deadline"] is None and state["wait_minutes"] is None
    assert outline_status(root)["wait_minutes"] is None
    set_wait(root, 5)
    build_outline(root, summary="A revised story")
    assert outline_status(root)["wait_minutes"] is None


def test_outline_legacy_wait_keeps_deadline_without_minutes(outline_project: Path) -> None:
    root = outline_project
    build_outline(root)
    state = set_wait(root, 5)
    state.pop("wait_minutes")
    _json(root / "outline/state.json", state)
    status = outline_status(root)
    assert status["deadline"] == state["deadline"] and status["wait_minutes"] is None
    assert status["state"] == "waiting"
    assert outline_status(root, now=NOW + timedelta(minutes=5))["state"] == "timed_out"
    approved = approve_outline(root)
    assert approved["deadline"] == state["deadline"] and approved["wait_minutes"] is None


@pytest.mark.parametrize("minutes", [0, -1, 10081, True, 1.0, "5"])
def test_outline_wait_validation(outline_project: Path, minutes) -> None:
    build_outline(outline_project)
    with pytest.raises(FeedbackError, match="minutes"):
        set_wait(outline_project, minutes)
    assert not (outline_project / "outline/state.json").exists()


def test_outline_missing_or_stale_refuses_actions(outline_project: Path) -> None:
    root = outline_project
    for action in (lambda: approve_outline(root), lambda: set_wait(root, 1),
                   lambda: send_notes(root),
                   lambda: add_note(root, beat=None, kind="fix", text="x")):
        with pytest.raises(FeedbackError, match="No story outline"):
            action()
    build_outline(root)
    (root / "edit.json").unlink()
    assert outline_status(root)["state"] == "stale"
    for action in (lambda: approve_outline(root), lambda: set_wait(root, 1)):
        with pytest.raises(FeedbackError, match="edit.json changed"):
            action()


@pytest.mark.parametrize("filename", ["outline.json", "state.json"])
def test_outline_corrupt_review_files_refused_untouched(outline_project: Path, filename) -> None:
    root = outline_project
    build_outline(root)
    set_wait(root, 1)
    path = root / "outline" / filename
    data = json.loads(path.read_text())
    data["extra"] = "unexpected"
    _json(path, data)
    before = path.read_bytes()
    with pytest.raises(FeedbackError, match=filename):
        outline_status(root)
    with pytest.raises(FeedbackError, match=filename):
        approve_outline(root)
    assert path.read_bytes() == before
