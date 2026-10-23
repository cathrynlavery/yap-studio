"""Round persistence, exact revision locks, and contained send failures."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest
from test_feedback_batch import _cut, _export, _json
from test_feedback_spelling import assert_proposals_shape

from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback.batch import FeedbackError, build_batch, load_batch
from yap_studio.feedback.clarify import answer_clarification, answered_questions, ask_clarification
from yap_studio.feedback.versions import save_version
from yap_studio.receipt import sha256_file
from yap_studio.review import send as send_module
from yap_studio.review.export import comment_to_dict, series_export
from yap_studio.review.launch import version_video
from yap_studio.review.model import MediaFacts
from yap_studio.review.rounds import (
    INTERRUPTED_ERROR,
    ROUNDS_SCHEMA,
    comment_states,
    round_from_dict,
    round_to_dict,
)
from yap_studio.review.series import SeriesEntry
from yap_studio.review.store import (
    CommentSentError,
    ReviewStore,
    RoundConflictError,
    UnknownRoundError,
)

SHA = "a" * 64
OTHER_SHA = "b" * 64
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
ERROR = {"code": "feedback_refused", "message": "Try again later."}


def test_regression_retry_after_batch_write_crash_reuses_origin(
    round_project: Path, tmp_path: Path, monkeypatch,
) -> None:
    store = ReviewStore(tmp_path / "data")
    series = version_video([str(round_project)], store, lambda _: []).series
    version = series.versions[-1]
    round = store.start_round(series.id, version.number, [note(store, version.sha256)])
    original = send_module.build_batch

    class ProcessDied(BaseException):
        pass

    def crash_after_build(*args, **kwargs):
        batch = original(*args, **kwargs)
        assert batch["id"] == "b1"
        raise ProcessDied()

    with monkeypatch.context() as patch:
        patch.setattr(send_module, "build_batch", crash_after_build)
        with pytest.raises(ProcessDied):
            send_module.send_round(store, series, version, round.id)
    assert store.rounds(series.id)[0].batch is None
    store.update_round(series.id, round.id, status="failed", step=None, error=INTERRUPTED_ERROR)
    store.retry_round(series.id, round.id)
    send_module.send_round(store, series, version, round.id)
    sent = store.rounds(series.id)[0]
    assert sent.status == "sent" and sent.batch == "b1"
    assert not (round_project / "feedback" / "b2.json").exists()
    assert load_batch(round_project, "b1")["origin"] == {"series": series.id, "round": round.id}


def test_regression_answer_saved_after_round_start_waits_for_next_round(
    round_project: Path, tmp_path: Path,
) -> None:
    store = ReviewStore(tmp_path / "data")
    series = version_video([str(round_project)], store, lambda _: []).series
    version = series.versions[-1]
    question_note = note(store, version.sha256)
    build_batch(round_project, _export(round_project, [question_note]))
    ask_clarification(round_project, "b1", "v2", question_note.id, "Which caption?")
    draft = note(store, version.sha256)
    round = store.start_round(series.id, version.number, [draft])
    assert round.answers == ()
    answer_clarification(round_project, "b1", "v2", question_note.id, answer="The ending")
    send_module.send_round(store, series, version, round.id)
    assert store.rounds(series.id)[0].status == "sent"
    batch = load_batch(round_project, "b2")
    assert [item["comment"]["id"] for item in batch["items"]] == [draft.id]
    question = load_batch(round_project, "b1")["items"][0]["clarification"]
    assert question["answered_at"] is not None and question["carried_to"] is None
    assert len(answered_questions(round_project)) == 1


@pytest.fixture
def round_project(tmp_path: Path) -> Path:
    """Build two saved versions using the real batch project's tiny media."""
    root = tmp_path / "project"
    _json(root / "project.json", {"schema": "yap-studio/project@1", "name": "sample-video",
                                 "takes": [{"id": "t1", "path": "takes/t1.mp4"}]})
    _json(root / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {}, "segments": []})
    _json(root / "timeline" / "agent.json", {"schema": "yap-studio/timeline@1", "elements": []})
    _cut(root)
    save_version(root)
    _cut(root, label="next render")
    save_version(root)
    return root


@pytest.fixture
def store(tmp_path: Path) -> ReviewStore:
    store = ReviewStore(tmp_path / "data", now=lambda: NOW)
    store.open_series([SeriesEntry(SHA, "first.mp4"), SeriesEntry(OTHER_SHA, "next.mp4")])
    return store


def note(store: ReviewStore, sha: str = SHA):
    return store.add_comment(sha, MediaFacts("render.mp4", sha, 4.0, Fraction(30), True),
                             {"text": "Move this caption", "start_frame": 15, "target": "caption"})


def fail(store: ReviewStore, round_id: str):
    return store.update_round(SHA, round_id, status="failed", step=None, error=ERROR)


def test_round_states_locks_retry_and_ids(store: ReviewStore) -> None:
    first = note(store)
    other = note(store, OTHER_SHA)
    with pytest.raises(ValidationError, match="no draft notes to send"):
        store.start_round(SHA, 1, [])
    round = store.start_round(SHA, 1, [first])
    assert round.id == "n1" and round.status == "sending" and round.attempts == 1
    assert round_from_dict(round_to_dict(round)) == round
    with pytest.raises(CommentSentError):
        store.start_round(SHA, 1, [first])
    with pytest.raises(CommentSentError):
        store.update_comment(SHA, store.review_facts(SHA), first.id, {"text": "Changed"})
    with pytest.raises(CommentSentError):
        store.delete_comment(SHA, first.id)
    assert store.start_round(SHA, 2, [other]).id == "n2"
    assert comment_states([first], [round])[first.id] == {
        "state": "sending", "round": "n1", "batch": None, "error": None,
    }
    failed = fail(store, "n1")
    assert comment_states([first], [failed])[first.id]["state"] == "failed"
    retry = store.retry_round(SHA, "n1")
    assert retry.status == "sending" and retry.error is None and retry.attempts == 2
    with pytest.raises(RoundConflictError):
        store.retry_round(SHA, "n1")
    sent = store.update_round(SHA, "n1", status="sent", step=None, batch="b3")
    assert comment_states([first], [failed, sent])[first.id] == {
        "state": "sent", "round": "n1", "batch": "b3", "error": None,
    }
    with pytest.raises(CommentSentError):
        store.delete_comment(SHA, first.id)
    with pytest.raises(UnknownRoundError):
        store.retry_round(SHA, "n99")
    assert ReviewStore(store.root).start_round(SHA, 1, [note(store)]).id == "n3"
    document = json.loads((store.root / "rounds" / f"{SHA}.json").read_text())
    assert document["schema"] == ROUNDS_SCHEMA and document["next_id"] == 4
    assert "send" not in comment_to_dict(first)


def test_edited_failed_revision_is_draft_even_in_same_second(store: ReviewStore) -> None:
    first = note(store)
    store.start_round(SHA, 1, [first])
    failed = fail(store, "n1")
    edited = store.update_comment(SHA, store.review_facts(SHA), first.id, {"text": "New note"})
    assert edited.updated_at != first.updated_at
    assert comment_states([edited], [failed])[first.id] == {
        "state": "draft", "round": None, "batch": None, "error": None,
    }
    new = store.start_round(SHA, 1, [edited])
    assert new.id == "n2"
    with pytest.raises(RoundConflictError):
        store.retry_round(SHA, "n1")
    assert comment_states([edited], [new, failed])[first.id]["state"] == "sending"


def test_comments_and_round_serializers_copy_mutable_data(store: ReviewStore) -> None:
    first = note(store)
    round = store.start_round(SHA, 1, [first])
    document = round_to_dict(round)
    parsed = round_from_dict(document)
    document["comments"][0]["id"] = "c99"
    assert parsed.comments[0]["id"] == round.comments[0]["id"] == "c1"
    failed = fail(store, round.id)
    states = comment_states([first], [failed])
    states[first.id]["error"]["message"] = "Changed"
    assert failed.error == ERROR


def test_answer_only_round_serialization_locks_and_legacy_records(store: ReviewStore) -> None:
    answer = {"batch": "b1", "comment": {"version": "v1", "id": "c1"}}
    round = store.start_round(SHA, 2, [], answers=[answer])
    answer["comment"]["id"] = "c99"
    assert round.comments == ()
    assert round.answers == ({"batch": "b1", "comment": {"version": "v1", "id": "c1"}},)
    with pytest.raises(RoundConflictError):
        store.start_round(SHA, 1, [], answers=list(round.answers))
    fail(store, round.id)
    replacement = store.start_round(SHA, 1, [], answers=list(round.answers))
    with pytest.raises(RoundConflictError):
        store.retry_round(SHA, round.id)
    sent = store.update_round(SHA, replacement.id, status="sent", step=None, batch="b2")
    assert round_from_dict(round_to_dict(sent)) == sent
    with pytest.raises(RoundConflictError):
        store.start_round(SHA, 2, [], answers=list(round.answers))
    legacy = round_to_dict(store.start_round(SHA, 1, [note(store)]))
    del legacy["answers"]
    assert round_from_dict(legacy).answers == ()


@pytest.mark.parametrize("answers", [
    None, {}, [{"batch": "b0", "comment": {"version": "v1", "id": "c1"}}],
    [{"batch": "b1", "comment": {"version": 1, "id": "c1"}}],
    [{"batch": "b1", "comment": {"version": "v1", "id": "c0"}}],
    [{"batch": "b1", "comment": {"version": "v1", "id": "c1", "extra": 1}}],
    [{"batch": "b1", "comment": {"version": "v1", "id": "c1"}, "extra": 1}],
    [{"batch": "b1", "comment": {"version": "v1", "id": "c1"}}] * 2,
])
def test_round_refuses_invalid_answers(store: ReviewStore, answers) -> None:
    round = store.start_round(SHA, 1, [note(store)])
    with pytest.raises(ValueError):
        round_from_dict({**round_to_dict(round), "answers": answers})


@pytest.mark.parametrize("change", [
    {"extra": 1}, {"id": "n0"}, {"version": True}, {"attempts": 0}, {"attempts": True},
    {"comments": []}, {"comments": [{"id": "c1", "revision": "invalid"}]},
    {"comments": [{"id": "c0", "revision": "2026-10-05T12:00:00Z"}]},
    {"comments": [{"id": "c1", "revision": "2026-10-05T12:00:00Z", "extra": 1}]},
    {"status": "draft"}, {"status": []}, {"step": " "}, {"step": "x" * 101},
    {"batch": "b1"}, {"error": ERROR}, {"created_at": "2026-02-30T12:00:00Z"},
    {"updated_at": "2025-10-05T12:00:00Z"},
    {"status": "sent"}, {"status": "failed"},
    {"status": "failed", "error": {"code": "", "message": "Bad"}},
    {"status": "failed", "error": {"code": "bad", "message": "Bad", "extra": 1}},
])
def test_strict_round_parsing(store: ReviewStore, change: dict) -> None:
    round = store.start_round(SHA, 1, [note(store)])
    with pytest.raises(ValueError):
        round_from_dict({**round_to_dict(round), **change})
    with pytest.raises(ValueError):
        round_to_dict(replace(round, status="unknown"))


@pytest.mark.parametrize("corruption", ["json", "schema", "series", "counter", "duplicate",
                                       "locked", "extra", "round", "bool"])
def test_corrupt_rounds_refused_and_untouched(store: ReviewStore, corruption: str) -> None:
    first = note(store)
    round = store.start_round(SHA, 1, [first])
    path = store.root / "rounds" / f"{SHA}.json"
    document = json.loads(path.read_text())
    if corruption == "schema":
        document["schema"] = "unknown"
    elif corruption == "series":
        document["series"] = OTHER_SHA
    elif corruption == "counter":
        document["next_id"] = 1
    elif corruption == "duplicate":
        document["rounds"].append(copy.deepcopy(document["rounds"][0]))
    elif corruption == "locked":
        document["rounds"].append({**document["rounds"][0], "id": "n2"})
        document["next_id"] = 3
    elif corruption == "extra":
        document["extra"] = 1
    elif corruption == "round":
        document["rounds"][0]["comments"] = []
    elif corruption == "bool":
        document["next_id"] = True
    path.write_text("{broken" if corruption == "json" else json.dumps(document))
    before = path.read_bytes()
    operations = [lambda: store.rounds(SHA), lambda: store.start_round(SHA, 1, [first]),
                  lambda: store.update_round(SHA, round.id, step="Locating notes"),
                  lambda: store.retry_round(SHA, round.id), lambda: store.recover_rounds(SHA)]
    for operation in operations:
        with pytest.raises(InputError, match="not a valid rounds file"):
            operation()
        assert path.read_bytes() == before


def test_round_start_refuses_stale_duplicate_and_invalid_version(store: ReviewStore) -> None:
    first = note(store)
    with pytest.raises(ValidationError):
        store.start_round(SHA, 1, [first, first])
    for version in (0, 3, True):
        with pytest.raises(ValidationError):
            store.start_round(SHA, version, [first])
    store.update_comment(SHA, store.review_facts(SHA), first.id, {"text": "Changed"})
    with pytest.raises(RoundConflictError):
        store.start_round(SHA, 1, [first])


def test_concurrent_stores_cannot_send_the_same_comment(store: ReviewStore) -> None:
    first = note(store)

    def start(_):
        try:
            return ReviewStore(store.root).start_round(SHA, 1, [first]).id
        except CommentSentError:
            return "locked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(start, range(2))) == ["locked", "n1"]
    assert len(store.rounds(SHA)) == 1


def test_recovery_is_idempotent_and_preserves_attempts(store: ReviewStore) -> None:
    store.start_round(SHA, 1, [note(store)])
    store.start_round(SHA, 2, [note(store, OTHER_SHA)])
    fail(store, "n1")
    assert store.recover_rounds(SHA) == 1
    recovered = store.rounds(SHA)[1]
    assert recovered.status == "failed" and recovered.error == INTERRUPTED_ERROR
    assert recovered.step is None and recovered.attempts == 1
    assert store.recover_rounds(SHA) == 0


@pytest.mark.parametrize("error", [FeedbackError("Temporary cause."), RuntimeError("detail")])
def test_send_contains_failures_and_retry_succeeds(
    round_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], error: Exception,
) -> None:
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(round_project)], store, lambda _: [])
    series = video.series
    assert series.project_dir == round_project.resolve()
    version = series.versions[-1]
    comment = note(store, version.sha256)
    round = store.start_round(series.id, 2, [comment])
    build = send_module.build_batch
    calls = []

    def flaky(project_dir, export, **kwargs):
        progress = store.rounds(series.id)[0]
        assert progress.status == "sending" and progress.step == "Writing the batch"
        assert export["series"] == series_export(store, series, 2)
        assert "send" not in export["comments"][0]
        calls.append(export)
        if len(calls) == 1:
            raise error
        return build(project_dir, export, **kwargs)

    monkeypatch.setattr(send_module, "build_batch", flaky)
    send_module.send_round(store, series, version, round.id, now=NOW)
    failed = store.rounds(series.id)[0]
    assert failed.status == "failed"
    assert failed.error == ({"code": "feedback_refused", "message": "Temporary cause."}
                            if isinstance(error, FeedbackError) else
                            {"code": "internal_error",
                             "message": "Could not send these notes. Try again."})
    assert capsys.readouterr().out == ""
    assert store.retry_round(series.id, round.id).attempts == 2
    send_module.send_round(store, series, version, round.id, now=NOW)
    sent = store.rounds(series.id)[0]
    assert sent.status == "sent" and sent.batch == "b1" and sent.error is None
    assert calls[1]["exported_at"] == "2026-10-05T12:00:00Z"


def test_retry_changed_or_deleted_notes_fails_without_building(
    round_project: Path, tmp_path: Path,
) -> None:
    store = ReviewStore(tmp_path / "data")
    series = version_video([str(round_project)], store, lambda _: []).series
    version = series.versions[-1]
    first = note(store, version.sha256)
    round = store.start_round(series.id, 2, [first])
    store.update_round(series.id, round.id, status="failed", error=ERROR)
    store.delete_comment(version.sha256, first.id)
    store.retry_round(series.id, round.id)
    send_module.send_round(store, series, version, round.id)
    failed = store.rounds(series.id)[0]
    assert failed.status == "failed" and "notes changed" in failed.error["message"]
    assert not (round_project / "feedback").exists()


def test_sent_spelling_round_writes_word_fix_and_merges_another_source(
    round_project: Path, tmp_path: Path,
) -> None:
    store = ReviewStore(tmp_path / "data")
    series = version_video([str(round_project)], store, lambda _: []).series
    version = series.versions[-1]
    facts = version.media.facts(version.sha256)
    request = {"type": "spelling", "from": "tail net", "to": "tailnet"}
    before = sha256_file(round_project / "project.json")
    for number in (1, 2):
        comment = store.add_comment(version.sha256, facts, {
            "kind": "fix", "target": "caption", "text": "Correct this spelling",
            "start_frame": 18, "request": request,
        })
        round = store.start_round(series.id, version.number, [comment])
        send_module.send_round(store, series, version, round.id)
        sent = store.rounds(series.id)[-1]
        assert sent.status == "sent" and sent.batch == f"b{number}"
        item = load_batch(round_project, sent.batch)["items"][0]
        assert item["disposition"] == "applied"
        assert item["operations"][0]["id"] == f"word-fix:{version.version_id}:{comment.id}"
        assert item["operations"][0]["op"] == "word_fix"
        assert item["evidence"]["project_json"] == {
            "before": before, "after": sha256_file(round_project / "project.json"),
        }
        before = sha256_file(round_project / "project.json")
    data = json.loads((round_project / "feedback" / "taste-proposals.json").read_text())
    assert_proposals_shape(data)
    assert len(data["proposals"]) == 1
    assert [source["comment_id"] for source in data["proposals"][0]["sources"]] == ["c1", "c2"]
    assert {source["media_sha256"] for source in data["proposals"][0]["sources"]} == {
        version.sha256,
    }
    assert json.loads((round_project / "project.json").read_text())["word_fixes"] == {
        "tail net": "tailnet",
    }


@pytest.mark.parametrize("failure", ["apply", "handoff"])
def test_spelling_failure_retains_batch_and_retry_only_applies_existing_batch(
    round_project: Path, tmp_path: Path, monkeypatch, failure: str,
) -> None:
    store = ReviewStore(tmp_path / "data")
    series = version_video([str(round_project)], store, lambda _: []).series
    version = series.versions[-1]
    comment = store.add_comment(version.sha256, version.media.facts(version.sha256), {
        "text": "Correct the caption spelling", "kind": "fix", "target": "caption",
        "start_frame": 18, "request": {"type": "spelling", "from": "tail net", "to": "tailnet"},
    })
    round = store.start_round(series.id, version.number, [comment])
    build, apply = send_module.build_batch, send_module.apply_word_fixes
    builds, applies = [], []

    def counted_build(*args, **kwargs):
        builds.append(True)
        return build(*args, **kwargs)

    def flaky_apply(project_dir, batch_id, *, sink):
        progress = store.rounds(series.id)[0]
        assert progress.batch == batch_id == "b1" and progress.step == "Applying spelling fixes"
        applies.append(True)
        if len(applies) == 1:
            if failure == "handoff":
                apply(project_dir, batch_id)
            raise RuntimeError("Spelling handoff unavailable.")
        return apply(project_dir, batch_id, sink=sink)

    monkeypatch.setattr(send_module, "build_batch", counted_build)
    monkeypatch.setattr(send_module, "apply_word_fixes", flaky_apply)
    send_module.send_round(store, series, version, round.id)
    failed = store.rounds(series.id)[0]
    assert failed.status == "failed" and failed.batch == "b1"
    assert failed.error["message"] == "Spelling handoff unavailable."
    assert load_batch(round_project, "b1")["id"] == "b1"
    reopened = ReviewStore(store.root)
    assert reopened.retry_round(series.id, round.id).batch == "b1"
    send_module.send_round(reopened, series, version, round.id)
    sent = reopened.rounds(series.id)[0]
    assert sent.status == "sent" and sent.batch == "b1" and sent.attempts == 2
    assert len(builds) == 1 and len(applies) == 2
    assert not (round_project / "feedback" / "b2.json").exists()
    assert len(load_batch(round_project, "b1")["items"][0]["operations"]) == 1
    assert_proposals_shape(json.loads(
        (round_project / "feedback" / "taste-proposals.json").read_text(),
    ))


def test_interrupted_round_retains_written_batch_for_retry(store: ReviewStore) -> None:
    round = store.start_round(SHA, 1, [note(store)])
    store.update_round(SHA, round.id, batch="b1", step="Applying spelling fixes")
    assert store.recover_rounds(SHA) == 1
    assert store.rounds(SHA)[0].batch == "b1"
    assert store.retry_round(SHA, round.id).batch == "b1"


def test_carried_spelling_proposal_uses_its_own_render(round_project: Path, tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data")
    series = version_video([str(round_project)], store, lambda _: []).series
    old, new = series.versions
    store.add_comment(old.sha256, old.media.facts(old.sha256), {
        "kind": "fix", "target": "caption", "text": "Correct the earlier spelling",
        "start_frame": 18, "request": {"type": "spelling", "from": "tail net", "to": "tailnet"},
    })
    round = store.start_round(series.id, new.number, [note(store, new.sha256)])
    send_module.send_round(store, series, new, round.id)
    assert store.rounds(series.id)[0].status == "sent"
    proposal = json.loads((round_project / "feedback" / "taste-proposals.json").read_text())
    assert_proposals_shape(proposal)
    assert proposal["proposals"][0]["sources"][0]["media_sha256"] == old.sha256
    assert old.sha256 != new.sha256
