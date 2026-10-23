"""Series ordering, durable replies, and unresolved comment carry-forward."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio.errors import InputError, ValidationError
from yap_studio.review.model import Comment, MediaFacts
from yap_studio.review.series import (
    REPLY_STATUSES,
    Reply,
    SeriesEntry,
    build_threads,
    is_resolved,
    reply_to_dict,
    series_from_dict,
    series_to_dict,
    thread_to_dict,
)
from yap_studio.review.store import ReviewStore, UnknownCommentError

NOW = datetime(2026, 10, 5, 22, 0, tzinfo=UTC)
ENTRIES = tuple(SeriesEntry(char * 64, f"render-{number}.mp4", f"v{number}", "sample-video")
                for number, char in enumerate("abc", 1))


def _comment(store: ReviewStore, number: int, frame: int = 0) -> Comment:
    entry = ENTRIES[number - 1]
    facts = MediaFacts(entry.name, entry.sha256, 10.0, Fraction(30), True)
    return store.add_comment(entry.sha256, facts,
                             {"kind": "fix", "target": "element", "text": "Move the card",
                              "start_frame": frame})


def _reply(store: ReviewStore, version: int = 2, **fields: object) -> Reply:
    return store.add_reply(ENTRIES[0].sha256, **{
        "comment_version": 1, "comment_id": "c1", "version": version, "status": "changed",
        "text": "Moved the card", **fields,
    })


def test_create_extend_reopen_and_plain_entries(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES[:2])
    assert series.id == ENTRIES[0].sha256 and series.name == ENTRIES[1].name
    assert [item.number for item in series.versions] == [1, 2]
    assert [item.sha256 for item in series.versions] == [item.sha256 for item in ENTRIES[:2]]
    assert series.versions[0].added_at == "2026-10-05T22:00:00Z"
    assert series.versions[0].version_id == "v1" and series.versions[0].project == "sample-video"
    assert series.replies == ()
    path = tmp_path / "series" / f"{series.id}.json"
    original = path.read_bytes()
    assert store.open_series(ENTRIES[:2]) == series
    assert path.read_bytes() == original
    extended = ReviewStore(tmp_path, now=lambda: NOW + timedelta(hours=1)).open_series(ENTRIES)
    assert extended.versions[:2] == series.versions
    assert extended.versions[2].number == 3 and extended.name == ENTRIES[2].name
    assert extended.versions[2].added_at == "2026-10-05T23:00:00Z"
    assert ReviewStore(tmp_path).series(series.id) == extended
    assert series_from_dict(json.loads(path.read_text())) == extended
    assert json.loads(path.read_text()) == series_to_dict(extended)
    plain = store.open_series([SeriesEntry("d" * 64, "plain.mp4")])
    assert plain.versions[0].version_id is None and plain.versions[0].project is None


@pytest.mark.parametrize("case", [
    "wrong_order", "missing_middle", "shorter", "duplicates", "metadata", "project"
])
def test_refuses_order_duplicates_and_provenance_conflicts(tmp_path: Path, case: str) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES)
    path = tmp_path / "series" / f"{series.id}.json"
    original = path.read_bytes()
    inputs = {
        "wrong_order": (ENTRIES[1], ENTRIES[0], ENTRIES[2]),
        "missing_middle": (ENTRIES[0], ENTRIES[2]),
        "shorter": ENTRIES[:2],
        "duplicates": (ENTRIES[0], SeriesEntry(ENTRIES[0].sha256, "duplicate.mp4")),
        "metadata": (SeriesEntry(ENTRIES[0].sha256, ENTRIES[0].name, "v9", "sample-video"),
                     *ENTRIES[1:]),
        "project": (SeriesEntry(ENTRIES[0].sha256, ENTRIES[0].name, "v1", "other-project"),
                    *ENTRIES[1:]),
    }
    with pytest.raises(ValidationError) as error:
        store.open_series(inputs[case])
    if case == "wrong_order":
        assert "expected order" in str(error.value) and "given" in str(error.value)
        assert ENTRIES[0].name in str(error.value) and ENTRIES[1].name in str(error.value)
    elif case in ("missing_middle", "shorter"):
        missing_number = 2 if case == "missing_middle" else 3
        assert f"version {missing_number}" in str(error.value)
        assert ENTRIES[missing_number - 1].name in str(error.value)
    elif case == "duplicates":
        assert "identical contents" in str(error.value)
        assert ENTRIES[0].name in str(error.value) and "duplicate.mp4" in str(error.value)
    assert path.read_bytes() == original


def test_refuses_entries_from_two_series_and_empty_input(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    store.open_series(ENTRIES[:1])
    store.open_series(ENTRIES[1:2])
    with pytest.raises(ValidationError, match="more than one series"):
        store.open_series(ENTRIES[:2])
    with pytest.raises(ValidationError, match="at least one"):
        store.open_series([])


def test_reply_replacement_retains_id_created_at_and_counter(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES[:2])
    comment = _comment(store, 1)
    evidence = {"frames": [4, 8], "note": "confirmed"}
    first = _reply(store, text="  Moved the card  ", evidence=evidence)
    assert first.id == "r1" and first.text == "Moved the card"
    assert first.comment_version == 1 and first.comment_id == comment.id
    assert first.created_at == first.updated_at == "2026-10-05T22:00:00Z"
    evidence["frames"].append(10)
    assert first.evidence["frames"] == [4, 8]
    later = ReviewStore(tmp_path, now=lambda: NOW + timedelta(hours=1))
    replaced = _reply(later, status="kept", text="Kept the card", evidence=None)
    assert replaced.id == first.id and replaced.created_at == first.created_at
    assert replaced.updated_at == "2026-10-05T23:00:00Z"
    assert later.series(series.id).replies == (replaced,)
    assert json.loads((tmp_path / "series" / f"{series.id}.json").read_text())["next_reply_id"] == 2
    later.open_series(ENTRIES)
    second = _reply(later, version=3)
    assert second.id == "r2"
    assert ReviewStore(tmp_path).series(series.id).replies == (replaced, second)
    assert reply_to_dict(second)["comment"] == {"version": 1, "id": "c1"}


@pytest.mark.parametrize("fields", [
    {"status": "unknown"}, {"comment_version": 2}, {"comment_version": 0}, {"version": 1},
    {"version": 4}, {"comment_version": True}, {"comment_id": "c99"}, {"text": "   "},
    {"text": "x" * 2001}, {"text": None}, {"evidence": []},
    {"evidence": {"large": "x" * 16384}}, {"evidence": {"invalid": float("nan")}},
])
def test_invalid_replies_are_refused_without_mutation(tmp_path: Path, fields: dict) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES)
    _comment(store, 1)
    path = tmp_path / "series" / f"{series.id}.json"
    original = path.read_bytes()
    with pytest.raises(ValidationError):
        _reply(store, **fields)
    assert path.read_bytes() == original


def test_unknown_comment_on_selected_version_is_refused(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    store.open_series(ENTRIES)
    _comment(store, 1)
    with pytest.raises(UnknownCommentError):
        _reply(store, version=3, comment_version=2)
    with pytest.raises(InputError):
        store.add_reply("f" * 64, comment_version=1, comment_id="c1", version=2,
                        status="changed", text="Changed")


def test_threads_open_carry_forward_latest_reply_and_timeline_order(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES)
    late = _comment(store, 1, frame=8)
    early = _comment(store, 1, frame=2)
    middle = _comment(store, 2)
    assert store.threads(series.id, 1) == []
    threads = store.threads(series.id, 2)
    assert [(item.version, item.comment.id, item.state) for item in threads] == [
        (1, early.id, "open"), (1, late.id, "open"),
    ]
    assert all(item.reply is None for item in threads)
    reply = _reply(store)
    clarification = _reply(
        store, comment_id=early.id, status="needs_clarification", text="Which card?"
    )
    threads = store.threads(series.id, 3)
    assert [(item.version, item.comment.id, item.state) for item in threads] == [
        (1, early.id, "needs_clarification"), (1, late.id, "changed"), (2, middle.id, "open"),
    ]
    assert threads[0].reply == clarification and not is_resolved(threads[0].state)
    assert threads[1].reply == reply and is_resolved(threads[1].state)
    latest = _reply(store, version=3, status="declined", text="Retained after review")
    assert store.threads(series.id, 3)[1].reply == latest
    assert store.threads(series.id, 2)[1].reply == reply
    assert thread_to_dict(threads[0]) == {
        "id": early.id, "start_frame": 2, "end_frame": None, "start_s": 0.0667,
        "end_s": None, "kind": "fix", "target": "element", "text": "Move the card",
        "created_at": "2026-10-05T22:00:00Z", "updated_at": "2026-10-05T22:00:00Z",
        "request": None,
        "version": 1, "reply": reply_to_dict(clarification), "state": "needs_clarification",
    }
    pure = build_threads(store.series(series.id), {1: [late, early], 2: [middle]}, 3)
    assert pure == store.threads(series.id, 3)
    store.delete_comment(ENTRIES[0].sha256, late.id)
    assert [item.comment.id for item in store.threads(series.id, 3)] == [early.id, middle.id]
    assert len(store.series(series.id).replies) == 3


@pytest.mark.parametrize("state", ["open", *REPLY_STATUSES])
def test_resolution_states(state: str) -> None:
    assert is_resolved(state) == (state in ("changed", "kept", "declined"))


@pytest.mark.parametrize("as_of", [0, 4, True, "2"])
def test_thread_version_must_exist(tmp_path: Path, as_of: object) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES)
    with pytest.raises(ValidationError):
        store.threads(series.id, as_of)
    with pytest.raises(ValidationError):
        build_threads(series, {}, as_of)


@pytest.mark.parametrize("corruption", [
    "json", "schema", "extra", "number", "counter", "reply", "duplicate"
])
def test_corrupt_series_is_refused_and_untouched(tmp_path: Path, corruption: str) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES[:2])
    _comment(store, 1)
    _reply(store)
    path = tmp_path / "series" / f"{series.id}.json"
    data = json.loads(path.read_text())
    if corruption == "json":
        path.write_text("{broken", encoding="utf-8")
    else:
        if corruption == "schema":
            data["schema"] = "unknown"
        elif corruption == "extra":
            data["extra"] = True
        elif corruption == "number":
            data["versions"][1]["number"] = 3
        elif corruption == "counter":
            data["next_reply_id"] = 1
        elif corruption == "reply":
            data["replies"][0]["comment"]["version"] = 2
        else:
            data["replies"].append(copy.deepcopy(data["replies"][0]))
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            series_from_dict(data)
    original = path.read_bytes()
    for action in (lambda: store.series(series.id), lambda: store.open_series(ENTRIES),
                   lambda: _reply(store), lambda: store.threads(series.id, 2)):
        with pytest.raises(InputError) as error:
            action()
        assert error.value.path == str(path)
        assert path.read_bytes() == original


def test_preserves_high_counter_when_extending_and_replacing(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES[:2])
    _comment(store, 1)
    _reply(store)
    path = tmp_path / "series" / f"{series.id}.json"
    data = json.loads(path.read_text())
    data["next_reply_id"] = 20
    path.write_text(json.dumps(data), encoding="utf-8")
    store.open_series(ENTRIES)
    assert _reply(store).id == "r1"
    assert _reply(store, version=3).id == "r20"
    assert json.loads(path.read_text())["next_reply_id"] == 21


def test_separate_store_instances_serialize_reply_replacement(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    series = store.open_series(ENTRIES[:2])
    _comment(store, 1)

    def write_reply(number: int) -> Reply:
        return _reply(ReviewStore(tmp_path, now=lambda: NOW), text=f"Revision {number}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(write_reply, range(8)))
    assert {item.id for item in replies} == {"r1"}
    persisted = ReviewStore(tmp_path).series(series.id)
    assert len(persisted.replies) == 1
    assert persisted.replies[0].text in {item.text for item in replies}
