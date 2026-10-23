"""Agent replies are proved by saved next-version evidence and scoped keeps."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_feedback_batch import NOW, _cut, _export, _json, _layers
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_locate import WORDS

from yap_studio.cli import main
from yap_studio.errors import ValidationError
from yap_studio.feedback import replies as replies_module
from yap_studio.feedback.batch import (
    FeedbackError,
    build_batch,
    load_batch,
    set_disposition,
    set_resulting_version,
)
from yap_studio.feedback.clarify import answer_clarification, ask_clarification
from yap_studio.feedback.replies import reply_evidence, reply_status, write_reply
from yap_studio.feedback.versions import save_version
from yap_studio.project.model import CutWord, CutWords
from yap_studio.review.export import series_export
from yap_studio.review.launch import version_video
from yap_studio.review.send import send_round
from yap_studio.review.series import SeriesEntry, validate_evidence
from yap_studio.review.store import ReviewStore

COMMENTS = [
    {"kind": "fix", "target": "story", "text": "Remove the repeated opening",
     "start_frame": 6, "end_frame": 33},
    {"kind": "keep", "target": "story", "text": "Keep the ending",
     "start_frame": 66, "end_frame": 93},
    {"kind": "fix", "target": "story", "text": "Change this moment\nKeep its meaning",
     "start_frame": 69},
]
REMOVED = CutWords("output", {}, WORDS.words[2:])


def _round(
    root: Path, data: Path, *, comments=None, own_words=WORDS, next_words=REMOVED,
    layers=False, own_layers=None, next_layers=None,
):
    """Send a real v1 round to b1, then save the requested next-version words."""
    if own_words != WORDS:
        _cut(root, words=own_words)
    first = save_version(root, layers=own_layers or (_layers(root) if layers else None))
    store = ReviewStore(data, now=lambda: NOW)
    video = version_video([str(root)], store, lambda _: [])
    notes = [store.add_comment(first.media["sha256"], video.media.facts(video.sha256), comment)
             for comment in (COMMENTS if comments is None else comments)]
    round = store.start_round(video.series.id, 1, notes)
    send_round(store, video.series, video.series.versions[0], round.id, now=NOW)
    assert store.rounds(video.series.id)[0].status == "sent"
    assert load_batch(root, "b1")["origin"] == {"series": video.series.id, "round": round.id}
    _cut(root, words=next_words, label="next render")
    second = save_version(root, layers=next_layers or (_layers(root) if layers else None))
    return store, first, second


@pytest.fixture
def reply_project(feedback_project: Path, tmp_path: Path):
    store, first, second = _round(feedback_project, tmp_path / "data")
    return feedback_project, store, first, second


def _reply(root, store, comment="c1", status="changed", text="Removed the repeated opening",
           version="v2"):
    return write_reply(root, store, "b1", "v1", comment, status, text, version_id=version)


def _tree(root: Path) -> dict:
    return {str(path.relative_to(root)): None if path.is_dir() else path.read_bytes()
            for path in root.rglob("*")} if root.exists() else {}


def test_round_replies_measure_removed_kept_and_unchanged_moments(reply_project) -> None:
    root, store, first, second = reply_project
    reply = _reply(root, store)
    assert reply.status == "changed" and reply.comment_version == 1 and reply.version == 2
    evidence = reply.evidence
    assert evidence["schema"] == "yap-studio/reply-evidence@1"
    assert evidence["batch"] == "b1" and evidence["comment_version"] == "v1"
    assert evidence["version"] == "v2" and evidence["media_sha256"] == second.media["sha256"]
    assert evidence["disposition"] == "open" and evidence["operations"] == []
    assert evidence["moment"]["status"] == "partial"
    assert "removed" in evidence["change"]["differences"]
    assert evidence["change"]["before"] == [
        {"id": "t1:0", "segment": "s1", "text": "Start"},
        {"id": "t1:1", "segment": "s1", "text": "really"},
    ]
    kept = _reply(root, store, "c2", "kept", "The ending is intact", None)
    assert kept.evidence["keep"] == {"status": "kept", "detail": "kept material is present"}
    with pytest.raises(FeedbackError, match="Nothing changed at this moment in v2") as error:
        _reply(root, store, "c3")
    assert error.value.suggested_action == (
        "Make the edit and save a new version, or reply declined with the reason."
    )
    declined = _reply(root, store, "c3", "declined", "The wording fits the story")
    assert declined.evidence["change"]["differences"] == []
    result = reply_status(root, store)
    assert result["complete"] and result["batch"] == "b1" and result["version"] == "v2"
    assert [item["reply"]["status"] for item in result["items"]] == ["changed", "kept", "declined"]
    assert all(item["suggested"] is None for item in result["items"])
    assert first.media["sha256"] != second.media["sha256"]


def test_range_maps_to_retimed_word_frames(feedback_project: Path, tmp_path: Path) -> None:
    words = CutWords("output", {}, (replace(WORDS.words[2], segment="s3", start=3.2, end=3.5),
                                  replace(WORDS.words[3], start=3.8, end=3.9)))
    comment = {**COMMENTS[2], "end_frame": 72}
    store, _, _ = _round(feedback_project, tmp_path / "data", comments=[comment], next_words=words)
    evidence = _reply(feedback_project, store).evidence
    assert evidence["moment"] == {
        "status": "moved", "start_s": pytest.approx(3.3), "end_s": pytest.approx(3.4),
        "start_frame": 99, "end_frame": 102,
    }
    assert evidence["change"]["after"][0]["segment"] == "s3"


def test_broken_keep_refuses_reply_with_check_detail(
    feedback_project: Path, tmp_path: Path,
) -> None:
    store, _, _ = _round(feedback_project, tmp_path / "data",
                         next_words=CutWords("output", {}, WORDS.words[:2]))
    with pytest.raises(FeedbackError, match="The keep is broken") as error:
        _reply(feedback_project, store, "c2", "kept", "The ending is intact")
    assert 'word t1:0 "Start" in s2 is missing from the new cut' in error.value.message
    assert error.value.suggested_action
    assert load_batch(feedback_project, "b1")["resulting_version"] is None


def test_range_maps_last_frame_independently(feedback_project: Path, tmp_path: Path) -> None:
    words = CutWords("output", {}, (replace(WORDS.words[0], start=0.5, end=0.8),
                                  replace(WORDS.words[1], start=1.8, end=2.1), *WORDS.words[2:]))
    store, _, _ = _round(feedback_project, tmp_path / "data", comments=[COMMENTS[0]],
                         next_words=words)
    moment = _reply(feedback_project, store).evidence["moment"]
    assert moment["start_frame"] == 15 and moment["end_frame"] == 63
    assert moment["start_s"] == pytest.approx(0.5) and moment["end_s"] == pytest.approx(2.1)


@pytest.mark.parametrize("word_count", [1, 2])
@pytest.mark.parametrize("edit", ["text", "duration", "both"])
def test_word_text_and_duration_edits_read_as_unchanged(
    feedback_project: Path, tmp_path: Path, word_count: int, edit: str,
) -> None:
    words = CutWords("output", {}, (
        CutWord("t1:4", "s1", "there", 1.0, 1.3),
        CutWord("t1:5", "s1", "again", 1.6, 1.9),
    )[:word_count])
    edited = replace(words.words[-1],
                     text="their" if edit in ("text", "both") else words.words[-1].text,
                     end=words.words[-1].end + (0.1 if edit in ("duration", "both") else 0))
    after = CutWords("output", {}, (*words.words[:-1], edited))
    comment = {**COMMENTS[0], "start_frame": 30, "end_frame": 39 if word_count == 1 else 57}
    store, _, second = _round(feedback_project, tmp_path / "data", comments=[comment],
                             own_words=words, next_words=after)
    batch = load_batch(feedback_project, "b1")
    evidence = reply_evidence(feedback_project, batch, batch["items"][0], second)
    expected = {"text": ["words"], "duration": ["timing"], "both": ["timing", "words"]}
    assert evidence["change"]["differences"] == expected[edit]
    assert _reply(feedback_project, store).evidence == evidence


@pytest.mark.parametrize("rate,delta,changed", [
    ("30/1", 1 / 30, False), ("30/1", 1 / 30 + 0.001, True),
    ("60/1", 0.02, True), ("60/1", 1 / 60, False),
    (None, 1 / 30, False), (None, 1 / 30 + 0.001, True),
])
def test_word_text_and_duration_edits_read_as_unchanged_frame_tolerance(
    feedback_project: Path, tmp_path: Path, rate: str | None, delta: float, changed: bool,
) -> None:
    words = CutWords("output", {}, (CutWord("t1:4", "s1", "there", 1.0, 1.3),))
    after = CutWords("output", {}, (replace(words.words[0], end=1.3 + delta),))
    store, _, second = _round(feedback_project, tmp_path / "data", own_words=words,
                             next_words=after, comments=[{**COMMENTS[0], "start_frame": 30,
                                                        "end_frame": 39}])
    batch = load_batch(feedback_project, "b1")
    version = replace(second, media={**second.media, "fps": rate})
    evidence = reply_evidence(feedback_project, batch, batch["items"][0], version)
    assert evidence["change"]["differences"] == (["timing"] if changed else [])
    if not changed:
        with pytest.raises(FeedbackError, match="Nothing changed"):
            _reply(feedback_project, store)


@pytest.mark.parametrize("rate", ["30/1", "60/1", None])
@pytest.mark.parametrize("offsets,shift,changed", [
    ((0.3, 0.3), 0.0, True), ((0.3, 0.0), 0.0, True), ((0.0, 0.3), 0.0, True),
    ((0.0, 0.0), 0.3, False), ((1 / 60, 1 / 60), 0.0, False),
])
def test_layer_moves_inside_comment_span_read_as_unchanged(
    feedback_project: Path, tmp_path: Path, rate: str | None,
    offsets: tuple[float, float], shift: float, changed: bool,
) -> None:
    words = CutWords("output", {}, (CutWord("t1:0", "s1", "A", 0.5, 0.8),
                                    CutWord("t1:1", "s1", "B", 2.7, 3.0)))
    after = CutWords("output", {}, tuple(replace(word, start=word.start + shift,
                                               end=word.end + shift) for word in words.words))
    old_path, new_path = feedback_project / "old-layers.json", feedback_project / "new-layers.json"
    layer = {"id": "title", "kind": "caption", "text": "Title", "start": 1.0, "end": 2.0}
    for path, start_offset, end_offset in ((old_path, 0, 0), (new_path, *offsets)):
        _json(path, {"schema": "yap-studio/layer-windows@1", "time_base": "output", "layers": [
            {**layer, "start": layer["start"] + start_offset + (shift if path == new_path else 0),
             "end": layer["end"] + end_offset + (shift if path == new_path else 0)},
        ]})
    store, _, second = _round(feedback_project, tmp_path / "data", own_words=words,
                             next_words=after, own_layers=old_path, next_layers=new_path,
                             comments=[{**COMMENTS[0], "start_frame": 15, "end_frame": 90}])
    batch = load_batch(feedback_project, "b1")
    version = replace(second, media={**second.media, "fps": rate})
    evidence = reply_evidence(feedback_project, batch, batch["items"][0], version)
    assert evidence["change"]["differences"] == (["layers"] if changed else [])
    if changed:
        assert _reply(feedback_project, store).evidence["change"]["differences"] == ["layers"]
    else:
        with pytest.raises(FeedbackError, match="Nothing changed"):
            _reply(feedback_project, store)


@pytest.mark.parametrize("removed", ["last", "start", "both"])
@pytest.mark.parametrize("rate", ["30/1", None])
def test_range_whose_last_word_was_cut_shows_wrong_join(
    feedback_project: Path, tmp_path: Path, removed: str, rate: str | None,
) -> None:
    words = CutWords("output", {}, (CutWord("t1:0", "s1", "A", 1.0, 1.3),
                                    CutWord("t1:1", "s1", "B", 1.7, 2.0),
                                    CutWord("t1:2", "s1", "C", 2.7, 3.0)))
    after = CutWords("output", {}, tuple(word for index, word in enumerate(words.words)
                                        if index not in ({1} if removed == "last" else
                                                         {0} if removed == "start" else {0, 1})))
    store, _, second = _round(feedback_project, tmp_path / "data", own_words=words,
                             next_words=after, comments=[{**COMMENTS[0], "start_frame": 30,
                                                        "end_frame": 60}])
    batch = load_batch(feedback_project, "b1")
    version = replace(second, media={**second.media, "fps": rate})
    evidence = reply_evidence(feedback_project, batch, batch["items"][0], version)
    start = 1.0 if removed == "last" else 1.7 if removed == "start" else 2.7
    end = 59 / 30 if removed == "start" else 2.7
    tolerance = 1 / 30 if rate is not None else 0
    assert evidence["moment"] == {
        "status": "partial" if removed == "last" else "removed", "start_s": pytest.approx(start),
        "end_s": pytest.approx(end + tolerance),
        "start_frame": round(start * 30) if rate is not None else None,
        "end_frame": round((end + tolerance) * 30) if rate is not None else None,
    }
    assert "removed" in evidence["change"]["differences"]
    assert _reply(feedback_project, store).evidence["moment"]["status"] == (
        "partial" if removed == "last" else "removed"
    )


def test_range_with_ambiguous_start_and_removed_last_word_preserves_ambiguity(
    feedback_project: Path, tmp_path: Path,
) -> None:
    words = CutWords("output", {}, (CutWord("t1:0", "s1", "A", 1.0, 1.3),
                                    CutWord("t1:1", "s1", "B", 1.7, 2.0)))
    after = CutWords("output", {}, (
        replace(words.words[0], segment="s2", start=0.5, end=0.8),
        replace(words.words[0], segment="s3", start=2.7, end=3.0),
    ))
    store, _, _ = _round(feedback_project, tmp_path / "data", own_words=words,
                         next_words=after, comments=[{**COMMENTS[0], "start_frame": 30,
                                                    "end_frame": 60}])
    evidence = _reply(feedback_project, store, text="Removed the final word").evidence
    assert evidence["moment"]["status"] == "ambiguous"
    assert "removed" in evidence["change"]["differences"]


@pytest.mark.parametrize("difference", ["order", "timing", "pause", "layers"])
def test_differences_measure_order_timing_pause_and_layers(
    feedback_project: Path, tmp_path: Path, difference: str,
) -> None:
    root = feedback_project
    words = CutWords("output", {}, (
        CutWord("t1:0", "s1", "A", 0.1, 0.3), CutWord("t1:1", "s1", "B", 0.5, 0.7),
        CutWord("t1:2", "s1", "C", 1.0, 1.2), CutWord("t1:3", "s1", "D", 1.5, 1.7),
        CutWord("t1:4", "s1", "E", 2.7, 3.0),
    ))
    comment = {**COMMENTS[0], "start_frame": 3, "end_frame": 90}
    next_layers = None
    if difference == "order":
        after = CutWords("output", {}, (words.words[0],
                                       replace(words.words[2], start=0.5, end=0.7),
                                       replace(words.words[1], start=1.0, end=1.2),
                                       *words.words[3:]))
    elif difference == "timing":
        after = CutWords("output", {}, (words.words[0],
                                       replace(words.words[1], start=0.8, end=0.9),
                                       *words.words[2:]))
    elif difference == "pause":
        comment = {**COMMENTS[0], "start_frame": 12, "end_frame": None}
        after = CutWords("output", {}, (words.words[0],
                                       CutWord("t1:99", "s1", "Inserted", 0.35, 0.38),
                                       *words.words[1:]))
    else:
        after = words
        next_layers = root / "next-windows.json"
        _json(next_layers, {"schema": "yap-studio/layer-windows@1", "time_base": "output",
                            "layers": [{"id": "caption", "kind": "caption", "start": 2.1,
                                        "end": 3.2, "text": "Updated ending"}]})
    store, _, _ = _round(root, tmp_path / "data", comments=[comment], own_words=words,
                         next_words=after, layers=difference == "layers", next_layers=next_layers)
    change = _reply(root, store).evidence["change"]
    assert difference in change["differences"]
    assert change["differences"] == sorted(set(change["differences"]))
    if difference in ("order", "timing", "layers"):
        assert "words" not in change["differences"]
    if difference == "pause":
        assert [word["id"] for word in change["after"]] == ["t1:0", "t1:99"]


def test_variable_rate_range_excludes_removed_word_at_end(
    feedback_project: Path, tmp_path: Path,
) -> None:
    words = CutWords("output", {}, (
        CutWord("t1:0", "s1", "A", 1.0, 1.3),
        CutWord("t1:1", "s1", "B", 1.3, 1.7),
        CutWord("t1:2", "s1", "C", 1.7, 2.0),
    ))
    _, _, second = _round(
        feedback_project, tmp_path / "data", own_words=words,
        next_words=CutWords("output", {}, words.words[:2]),
        comments=[{**COMMENTS[0], "start_frame": 30, "end_frame": 51}],
    )
    batch = load_batch(feedback_project, "b1")
    item = copy.deepcopy(batch["items"][0])
    item["comment"].update(start_frame=None, end_frame=None, start_s=1.0, end_s=1.7)
    item["location"].update(start_frame=None, end_frame=None, start_s=1.0, end_s=1.7)
    variable = replace(second, media={**second.media, "fps": None})
    evidence = reply_evidence(feedback_project, batch, item, variable)
    assert evidence["moment"]["status"] not in ("partial", "removed")
    assert "removed" not in evidence["change"]["differences"]
    assert evidence["change"]["differences"] == []


def test_variable_rate_moment_uses_seconds_and_no_frames(reply_project) -> None:
    root, _, _, second = reply_project
    batch = load_batch(root, "b1")
    item = copy.deepcopy(batch["items"][2])
    item["comment"].update(start_frame=None, end_frame=None, start_s=2.3, end_s=2.45)
    item["location"].update(start_frame=None, end_frame=None, start_s=2.3, end_s=2.45)
    variable = replace(second, media={**second.media, "fps": None})
    moment = reply_evidence(root, batch, item, variable)["moment"]
    assert moment == {"status": "mapped", "start_s": 2.3, "end_s": pytest.approx(2.45),
                      "start_frame": None, "end_frame": None}
    assert 2.3 <= moment["end_s"] < 2.45
    item["comment"]["end_s"] = None
    item["location"]["end_s"] = 2.3
    evidence = reply_evidence(root, batch, item, variable)
    assert evidence["moment"]["start_s"] == evidence["moment"]["end_s"] == 2.3
    assert evidence["change"]["differences"] == []


def test_clarification_replies_require_question_and_block_unanswered_changes(reply_project) -> None:
    root, store, _, _ = reply_project
    ask_clarification(root, "b1", "v1", "c1", "Which words should go?",
                      options=[{"value": "opening", "label": "The opening"}])
    with pytest.raises(FeedbackError, match="still needs an answer"):
        _reply(root, store)
    reply = _reply(root, store, status="needs_clarification", text="Which words should go?")
    assert reply.evidence["question"] == {
        "question": "Which words should go?", "answer": None, "option": None,
    }
    with pytest.raises(FeedbackError, match="no clarification question"):
        _reply(root, store, "c3", "needs_clarification", "What should change?")
    answer_clarification(root, "b1", "v1", "c1", option="opening")
    changed = _reply(root, store)
    assert changed.evidence["question"]["option"] == "opening"
    assert changed.id == reply.id


def test_operations_prove_change_and_refuse_declined(reply_project) -> None:
    root, store, _, _ = reply_project
    set_disposition(root, "b1", "v1", "c3", "applied", operations=[
        {"id": "op1", "kind": "caption", "private_detail": "do not copy"},
        {"id": "op2", "type": "style"}, {"id": "op3"},
    ])
    with pytest.raises(FeedbackError, match="recorded operations"):
        _reply(root, store, "c3", "declined", "Left it alone")
    reply = _reply(root, store, "c3", text="Updated the caption styling")
    assert reply.evidence["change"]["differences"] == []
    assert reply.evidence["operations"] == [
        {"id": "op1", "type": "caption"}, {"id": "op2", "type": "style"}, {"id": "op3"},
    ]
    assert reply.evidence["disposition"] == "applied"


def test_resulting_version_required_recorded_once_and_cannot_change(reply_project) -> None:
    root, store, _, _ = reply_project
    before = _tree(store.root)
    with pytest.raises(FeedbackError, match="no resulting version") as error:
        _reply(root, store, version=None)
    assert error.value.suggested_action == "Save the next version, then pass --version vN."
    assert _tree(store.root) == before
    _reply(root, store)
    assert load_batch(root, "b1")["resulting_version"] == "v2"
    with pytest.raises(FeedbackError, match="must be v2 or a later saved version"):
        _reply(root, store, version="v1")
    assert _reply(root, store, version=None).version == 2


def test_replies_land_in_later_project_review_series(reply_project) -> None:
    root, store, first, _ = reply_project
    reply = _reply(root, store)
    video = version_video([str(root)], store, lambda _: [])
    assert video.series.id == first.media["sha256"]
    assert [version.name for version in store.series(video.series.id).versions] == [
        "sample-video-v1.mp4", "sample-video-v2.mp4",
    ]
    threads = store.threads(video.series.id, 2)
    thread = next(thread for thread in threads if thread.comment.id == "c1")
    assert thread.version == 1 and thread.reply == reply
    assert thread.reply.evidence["moment"]["status"] == "partial"


def test_carried_comment_uses_own_version_instead_of_newer_batch_base(reply_project) -> None:
    root, store, first, second = reply_project
    video = version_video([str(root)], store, lambda _: [])
    export = _export(root, [], version=second)
    export["series"] = series_export(store, video.series, 2)
    batch = build_batch(root, export)
    assert batch["base_version"] == "v2" and batch["id"] == "b2"
    item = next(item for item in batch["items"] if item["comment"]["id"] == "c1")
    assert item["carried"] and item["comment"]["version"] == "v1"
    _cut(root, words=REMOVED, label="third render")
    third = save_version(root)
    reply = write_reply(root, store, "b2", "v1", "c1", "changed", "Removed the opening",
                        version_id="v3")
    assert reply.comment_version == 1 and reply.version == 3
    assert reply.evidence["media_sha256"] == third.media["sha256"]
    assert reply.evidence["moment"]["status"] == "partial"
    assert [word["segment"] for word in reply.evidence["change"]["before"]] == ["s1", "s1"]
    assert next(thread for thread in store.threads(first.media["sha256"], 3)
                if thread.comment.id == "c1").reply == reply


def test_other_review_series_is_refused(reply_project) -> None:
    root, store, _, _ = reply_project
    other = store.open_series([SeriesEntry("a" * 64, "other.mp4")])
    batch = load_batch(root, "b1")
    batch["review"]["series_id"] = other.id
    _json(root / "feedback" / "b1.json", batch)
    before = _tree(store.root)
    with pytest.raises(FeedbackError, match="different review series"):
        _reply(root, store)
    assert _tree(store.root) == before
    assert load_batch(root, "b1")["resulting_version"] is None


def test_rewriting_reply_keeps_id_and_updates_text(reply_project) -> None:
    root, store, first, _ = reply_project
    reply = _reply(root, store)
    updated = _reply(root, store, text="Removed both repeated opening words")
    assert updated.id == reply.id and updated.created_at == reply.created_at
    assert updated.text == "Removed both repeated opening words"
    assert store.series(first.media["sha256"]).replies == (updated,)


def test_another_batch_reply_completes_this_batch(
    feedback_project: Path, tmp_path: Path,
) -> None:
    root = feedback_project
    store, first, second = _round(root, tmp_path / "data", comments=[COMMENTS[0]])
    video = version_video([str(root)], store, lambda _: [])
    export = _export(root, [], version=second)
    export["series"] = series_export(store, video.series, 2)
    batch = build_batch(root, export)
    assert batch["id"] == "b2" and len(batch["items"]) == 1
    _cut(root, words=REMOVED, label="third render")
    save_version(root)
    set_resulting_version(root, "b2", "v3")
    earlier = _reply(root, store, version="v3")
    missing = reply_status(root, store, "b2")
    assert missing["items"][0]["reply"] is None
    assert not missing["complete"] and missing["items"][0]["suggested"] == "changed"
    newest = write_reply(root, store, "b2", "v1", "c1", "changed", "Removed it for this batch")
    assert newest.id == earlier.id and newest.created_at == earlier.created_at
    assert newest.evidence["batch"] == "b2"
    assert reply_status(root, store, "b2")["complete"]
    assert reply_status(root, store, "b2")["items"][0]["reply"]["id"] == newest.id
    missing = reply_status(root, store, "b1")
    assert not missing["complete"] and missing["items"][0]["reply"] is None
    assert store.series(first.media["sha256"]).replies == (newest,)
    assert store.threads(first.media["sha256"], 3)[0].reply == newest


@pytest.mark.parametrize("evidence", [None, {"batch": "b1"},
                                      {"schema": "yap-studio/reply-evidence@2", "batch": "b1"}])
def test_another_batch_reply_completes_this_batch_requires_evidence_schema(
    feedback_project: Path, tmp_path: Path, evidence: dict | None,
) -> None:
    root = feedback_project
    store, first, _ = _round(root, tmp_path / "data", comments=[COMMENTS[0]])
    _reply(root, store)
    store.add_reply(first.media["sha256"], comment_version=1, comment_id="c1", version=2,
                    status="changed", text="Legacy reply", evidence=evidence)
    result = reply_status(root, store, "b1")
    assert result["items"][0]["reply"] is None
    assert not result["complete"] and result["items"][0]["suggested"] == "changed"


@pytest.mark.parametrize("kind,disposition,operations,question,expected", [
    ("fix", "declined", False, None, "declined"),
    ("fix", "declined", True, None, None),
    ("keep", "declined", False, None, "declined"),
    ("keep", "declined", True, None, None),
    ("fix", "open", False, None, "changed"),
    ("keep", "open", False, None, None),
    ("fix", "needs_clarification", False, "unanswered", "needs_clarification"),
    ("fix", "needs_clarification", False, "answered", "changed"),
])
def test_suggestions_agent_cannot_follow(
    feedback_project: Path, tmp_path: Path, kind: str, disposition: str,
    operations: bool, question: str | None, expected: str | None,
) -> None:
    root = feedback_project
    store, _, second = _round(root, tmp_path / "data", comments=[{**COMMENTS[0], "kind": kind}])
    if question is not None:
        ask_clarification(root, "b1", "v1", "c1", "Which words?")
        if question == "answered":
            answer_clarification(root, "b1", "v1", "c1", answer="The opening")
    set_disposition(root, "b1", "v1", "c1", disposition,
                    operations=[{"id": "op1", "type": "cut"}] if operations else None)
    set_resulting_version(root, "b1", "v2")
    batch = load_batch(root, "b1")
    evidence = reply_evidence(root, batch, batch["items"][0], second)
    assert evidence["change"]["differences"]
    before = _tree(tmp_path)
    result = reply_status(root, store, "b1")
    assert result["items"][0]["suggested"] == expected
    assert _tree(tmp_path) == before
    if expected is not None:
        reply = _reply(root, store, status=expected, text="Measured response")
        assert reply.status == expected and reply.evidence == evidence


def test_evidence_size_truncates_prefixes_after_measuring_full_range(
    feedback_project: Path, tmp_path: Path,
) -> None:
    words = tuple(CutWord(f"t1:{index}", "s1", "\U0001f642" * 250,
                          index / 100, (index + 0.5) / 100) for index in range(350))
    own = CutWords("output", {}, words)
    next_words = CutWords("output", {}, (*words[:200], *words[201:]))
    store, _, second = _round(feedback_project, tmp_path / "data", own_words=own,
                             next_words=next_words, comments=[{
                                 **COMMENTS[0], "start_frame": 0, "end_frame": 105,
                             }])
    batch = load_batch(feedback_project, "b1")
    evidence = reply_evidence(feedback_project, batch, batch["items"][0], second)
    payload = json.dumps(evidence, separators=(",", ":"), ensure_ascii=False).encode()
    assert len(payload) <= 16384
    assert 0 < len(evidence["change"]["before"]) < 40
    assert 0 < len(evidence["change"]["after"]) < 40
    assert evidence["change"]["before"][0]["id"] == "t1:0"
    assert "words" in evidence["change"]["differences"]
    assert validate_evidence(evidence) == evidence
    assert _reply(feedback_project, store).evidence == evidence


def test_reply_status_writes_nothing_and_suggests_measured_statuses(
    feedback_project: Path, tmp_path: Path, monkeypatch,
) -> None:
    comments = [*COMMENTS,
                {**COMMENTS[0], "kind": "keep", "text": "Keep the opening"},
                {**COMMENTS[1], "target": "caption", "text": "Keep this caption"}]
    store, first, _ = _round(feedback_project, tmp_path / "data", comments=comments)
    fresh = ReviewStore(tmp_path / "uncreated-data")
    before = _tree(tmp_path)
    result = reply_status(feedback_project, fresh)
    assert not result["complete"] and result["version"] is None
    assert all(item["reply"] is None and item["suggested"] is None for item in result["items"])
    assert _tree(tmp_path) == before and not fresh.root.exists()
    ask_clarification(feedback_project, "b1", "v1", "c3", "Which part?")
    set_resulting_version(feedback_project, "b1", "v2")
    check = replies_module.check_version
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return check(*args, **kwargs)

    monkeypatch.setattr(replies_module, "check_version", counted)
    before = _tree(tmp_path)
    for selected in (fresh, store):
        result = reply_status(feedback_project, selected, "b1")
        assert not result["complete"]
        assert all(item["reply"] is None for item in result["items"])
        assert {item["comment"]["id"]: item["suggested"] for item in result["items"]} == {
            "c1": "changed", "c2": "kept", "c3": "needs_clarification", "c4": None, "c5": "kept",
        }
        assert _tree(tmp_path) == before
    assert len(calls) == 2 and all(call == {"batch_id": "b1", "version_id": "v2"}
                                   for call in calls)
    assert len(store.series(first.media["sha256"]).versions) == 1
    reply = _reply(feedback_project, store)
    before = _tree(tmp_path)
    assert reply_status(feedback_project, store)["items"][0]["reply"]["id"] == reply.id
    assert _tree(tmp_path) == before


def test_unmeasured_and_missing_keep_constraints_are_recorded(reply_project, monkeypatch) -> None:
    root, store, _, second = reply_project
    ask_clarification(root, "b1", "v1", "c2", "Which material should stay?")
    batch = load_batch(root, "b1")
    check = replies_module.check_version
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return check(*args, **kwargs)

    monkeypatch.setattr(replies_module, "check_version", counted)
    evidence = reply_evidence(root, batch, batch["items"][1], second)
    assert evidence["keep"] == {
        "status": "unmeasured", "detail": "no keep constraint for this comment",
    }
    assert len(calls) == 1
    assert _reply(root, store, "c2", "kept", "Awaiting the scoped keep check").evidence == evidence


@pytest.mark.parametrize("status,comment,reason", [
    ("changed", "c2", "Only a fix"), ("kept", "c1", "Only a keep"),
    ("changed", "c1", "was declined"),
])
def test_reply_status_rules_refuse_wrong_kind_and_declined_fixes(
    reply_project, status, comment, reason,
) -> None:
    root, store, _, _ = reply_project
    if reason == "was declined":
        set_disposition(root, "b1", "v1", "c1", "declined")
    with pytest.raises(FeedbackError, match=reason) as error:
        _reply(root, store, comment, status)
    assert error.value.suggested_action


@pytest.mark.parametrize("version", [None, "series:unsaved:1", "v99"])
def test_unsaved_comment_versions_have_no_mapped_evidence_and_refuse_replies(
    reply_project, version,
) -> None:
    root, store, _, second = reply_project
    batch = load_batch(root, "b1")
    item = batch["items"][0]
    item["comment"]["version"] = version
    evidence = reply_evidence(root, batch, item, second)
    assert evidence["moment"] is None and evidence["change"] is None
    assert "not saved in the project" in evidence["unmapped"]
    _json(root / "feedback" / "b1.json", batch)
    with pytest.raises(FeedbackError, match="This comment's version is not saved in the project"):
        write_reply(root, store, "b1", version, "c1", "declined", "Cannot locate this version",
                    version_id="v2")


@pytest.mark.parametrize("missing", [False, True])
def test_changed_or_missing_saved_media_refuses_reply(reply_project, missing) -> None:
    root, store, _, second = reply_project
    before = _tree(store.root)
    if missing:
        second.media_path.unlink()
    else:
        second.media_path.write_bytes(b"changed saved render")
    with pytest.raises(FeedbackError) as error:
        _reply(root, store)
    assert str(second.media_path) in error.value.message and error.value.suggested_action
    assert _tree(store.root) == before
    assert load_batch(root, "b1")["resulting_version"] is None


def test_invalid_comment_status_and_text_leave_reply_store_unchanged(reply_project) -> None:
    root, store, _, _ = reply_project
    before = _tree(store.root)
    for kwargs, parameter in [({"comment": "c99"}, "comment"),
                              ({"status": "invalid"}, "status"),
                              ({"text": " "}, "text")]:
        with pytest.raises(ValidationError) as error:
            _reply(root, store, **kwargs)
        assert error.value.parameter == parameter
        assert _tree(store.root) == before


def test_evidence_whole_video_unlocated_and_operation_limit(reply_project) -> None:
    root, _, _, second = reply_project
    batch = load_batch(root, "b1")
    item = copy.deepcopy(batch["items"][0])
    item["location"]["status"] = "whole_video"
    item["operations"] = [{"id": f"op{index}", "type": "edit", "inputs_after": {}}
                          for index in range(25)]
    evidence = reply_evidence(root, batch, item, second)
    assert evidence["moment"] is None and evidence["change"] is None
    assert evidence["unmapped"] == "whole video"
    assert evidence["operations"] == [{"id": f"op{index}", "type": "edit"} for index in range(20)]
    item["location"] = None
    item["unlocated"] = "Missing the reviewed location"
    assert reply_evidence(root, batch, item, second)["unmapped"] == item["unlocated"]


def test_feedback_reply_cli_text_json_refusal_and_listing(
    reply_project, monkeypatch, capsys,
) -> None:
    root, store, _, _ = reply_project
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    listing = ["feedback", "replies", str(root)]
    assert main(listing) == 3
    assert capsys.readouterr().out == (
        "missing v1:c1 fix: Remove the repeated opening\n"
        "missing v1:c2 keep: Keep the ending\n"
        "missing v1:c3 fix: Change this moment\n"
    )
    command = ["feedback", "reply", str(root), "--batch", "b1", "--comment", "v1:c1",
               "--status", "changed", "--text", "Removed the opening", "--version", "v2"]
    assert main(command) == 0
    assert capsys.readouterr().out == "Replied to v1:c1 in v2: changed\n"
    assert main([*command, "--json"]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["id"] == "r1" and reply["comment"] == {"version": 1, "id": "c1"}
    assert reply["status"] == "changed" and reply["evidence"]["moment"]["status"] == "partial"
    assert main([*listing, "--batch", "b1"]) == 3
    assert capsys.readouterr().out == (
        "changed v1:c1 fix: Remove the repeated opening\n"
        "missing v1:c2 keep: Keep the ending\n  suggested: kept\n"
        "missing v1:c3 fix: Change this moment\n"
    )
    refused = ["feedback", "reply", str(root), "--batch", "b1", "--comment", "v1:c3",
               "--status", "changed", "--text", "Changed this moment"]
    assert main(refused) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: Nothing changed" in captured.err and "fix:" in captured.err
    _reply(root, store, "c2", "kept", "Kept the ending")
    _reply(root, store, "c3", "declined", "The wording fits")
    assert main(listing) == 0
    assert "declined v1:c3" in capsys.readouterr().out
    assert main([*listing, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["complete"] is True
    assert main([*listing, "--batch", "b99"]) == 2
    assert "error:" in capsys.readouterr().err
    invalid = command.copy()
    invalid[invalid.index("v1:c1")] = "invalid"
    assert main(invalid) == 2
    assert "fix:" in capsys.readouterr().err


def test_reply_parser_keeps_implementation_imports_lazy() -> None:
    result = subprocess.run([
        sys.executable, "-c",
        "import sys\n"
        "from yap_studio.cli import _parser\n"
        "parser = _parser()\n"
        "args = parser.parse_args(['feedback', 'replies', 'project'])\n"
        "assert args.feedback_command == 'replies'\n"
        "parser.parse_args(['feedback', 'reply', 'project', '--batch', 'b1', '--comment', 'v1:c1', "
        "'--status', 'changed', '--text', 'Changed'])\n"
        "for name in ('replies', 'batch', 'keeps', 'versions'):\n"
        "    assert 'yap_studio.feedback.' + name not in sys.modules, name\n",
    ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
