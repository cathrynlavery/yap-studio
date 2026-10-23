"""Clarification rules, answer validation, and carrying saved answers once."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from test_feedback_batch import _cut, _export, _layers
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_locate import _comment, _locate

from yap_studio.errors import ValidationError
from yap_studio.feedback.batch import FeedbackError, build_batch, load_batch, set_disposition
from yap_studio.feedback.clarify import (
    CarriedQuestionError,
    UnknownQuestionError,
    answer_clarification,
    answered_questions,
    ask_clarification,
    automatic_clarification,
    pending_questions,
)
from yap_studio.feedback.versions import LayerWindow, save_version
from yap_studio.review.export import comment_to_dict


def test_ambiguous_fix_asks_three_options_from_saved_layers(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    batch = build_batch(root, _export(root, [_comment()]))
    item = batch["items"][0]
    assert item["disposition"] == "needs_clarification"
    assert item["constraint"] is None and item["operations"] == []
    assert item["clarification"] == {
        "question": "Which do you want changed here?", "reason": "ambiguous_target",
        "options": [{"value": "story", "label": "The words"},
                    {"value": "caption", "label": 'The caption "Start End"'},
                    {"value": "element", "label": "The graphic card"}],
        "answer": None, "option": None, "answered_at": None, "carried_to": None,
    }
    assert pending_questions(root) == [{
        "batch": "b1", "comment": {"version": "v1", "id": "c1"},
        **{key: item["clarification"][key] for key in ("question", "options", "reason")},
    }]
    assert answered_questions(root) == []


@pytest.mark.parametrize("target", ["story", "speech", "caption", "element", "video"])
def test_explicit_targets_do_not_ask(target: str) -> None:
    comment = _comment(target=target)
    assert automatic_clarification(comment, "v1", _locate(comment)) is None


def test_keeps_unknown_layers_and_single_candidates_do_not_ask() -> None:
    keep = _comment(kind="keep")
    assert automatic_clarification(keep, "v1", _locate(keep)) is None
    assert automatic_clarification(keep, "v1", None) is None
    comment = _comment()
    location = _locate(comment)
    assert automatic_clarification(comment, "v1", replace(location, layers_known=False)) is None
    assert automatic_clarification(comment, "v1", replace(location, layers=())) is None
    no_words = replace(location, words=(), layers=(LayerWindow("c1", "caption", 2, 3, None),
                                                 LayerWindow("g1", "graphic", 2, 3, None)))
    question = automatic_clarification(comment, "v1", no_words)
    assert question["options"] == [{"value": "caption", "label": "The caption"},
                                   {"value": "element", "label": "The graphic g1"}]


def test_outside_fix_asks_for_its_version(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root)
    item = build_batch(root, _export(root, [_comment(start_frame=150)]))["items"][0]
    assert item["location"]["status"] == "outside"
    assert item["clarification"]["reason"] == "unlocated"
    assert item["clarification"]["options"] == []
    assert item["clarification"]["question"] == (
        "I could not find this moment in v1. Which part do you mean?"
    )


@pytest.mark.parametrize("question", ["", " ", "x" * 501, "x" * 500 + " ", None])
def test_ask_refuses_bad_question(feedback_project: Path, question) -> None:
    root = feedback_project
    build_batch(root, _export(root, [_comment()]))
    before = (root / "feedback" / "b1.json").read_bytes()
    with pytest.raises(ValidationError, match="1 to 500"):
        ask_clarification(root, "b1", None, "c1", question)
    assert (root / "feedback" / "b1.json").read_bytes() == before


def test_agent_question_and_answer_rules(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root)
    build_batch(root, _export(root, [_comment(), _comment("c2", kind="keep")]))
    options = [{"value": "caption", "label": "The caption"}]
    batch = ask_clarification(root, "b1", "v1", "c1", "  Which caption?  ", options=options)
    options[0]["label"] = "Changed elsewhere"
    item = batch["items"][0]
    assert item["disposition"] == "needs_clarification"
    assert item["clarification"]["reason"] == "agent"
    assert item["clarification"]["question"] == "Which caption?"
    assert item["clarification"]["options"][0]["label"] == "The caption"
    before = (root / "feedback" / "b1.json").read_bytes()
    for kwargs in ({}, {"answer": " "}, {"answer": "x" * 2001}, {"answer": 1},
                   {"option": "bad"}, {"option": []}):
        with pytest.raises(ValidationError):
            answer_clarification(root, "b1", "v1", "c1", **kwargs)
        assert (root / "feedback" / "b1.json").read_bytes() == before
    with pytest.raises(ValidationError, match="no open clarification"):
        answer_clarification(root, "b1", "v1", "c2", answer="Leave it")
    with pytest.raises(UnknownQuestionError):
        answer_clarification(root, "b1", "v1", "c99", answer="This one")
    with pytest.raises(UnknownQuestionError):
        answer_clarification(root, "b99", "v1", "c1", answer="This one")
    batch = answer_clarification(root, "b1", "v1", "c1", answer="  The last caption  ")
    clarification = batch["items"][0]["clarification"]
    assert clarification["answer"] == "The last caption" and clarification["option"] is None
    assert clarification["answered_at"].endswith("Z")
    assert pending_questions(root) == []
    assert answered_questions(root)[0]["answer"] == "The last caption"
    with pytest.raises(ValidationError, match="no open clarification"):
        answer_clarification(root, "b1", "v1", "c1", option="caption")


def test_agent_cannot_ask_after_operations(feedback_project: Path) -> None:
    root = feedback_project
    build_batch(root, _export(root, [_comment()]))
    set_disposition(root, "b1", None, "c1", "applied", operations=[{"id": "op1"}])
    with pytest.raises(ValidationError, match="operations were recorded"):
        ask_clarification(root, "b1", None, "c1", "Which caption?")


def test_answered_keep_builds_constraints_for_chosen_target(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    batch = build_batch(root, _export(root, [_comment(kind="keep")]))
    assert batch["constraints"][0]["target"] == "story"
    asked = ask_clarification(root, "b1", "v1", "c1", "Which part should stay?",
                              options=[{"value": "caption", "label": "The caption"}])
    assert asked["constraints"] == [] and asked["items"][0]["constraint"] is None
    _cut(root, label="next render")
    second = save_version(root)
    export = _export(root, [], version=second)
    export["series"] = {
        "id": "sample-series", "version": 2, "versions": [
            {"number": 1, "sha256": batch["review"]["media_sha256"], "version_id": "v1"},
            {"number": 2, "sha256": second.media["sha256"], "version_id": "v2"},
        ], "threads": [{**comment_to_dict(_comment(kind="keep")), "version": 1,
                        "reply": None, "state": "open"}],
    }
    waiting = build_batch(root, export)
    assert waiting["items"] == [] and waiting["constraints"] == []
    assert len(pending_questions(root)) == 1
    answer_clarification(root, "b1", "v1", "c1", option="caption")
    carried = build_batch(root, export)
    assert carried["constraints"][0]["target"] == "caption"
    assert carried["constraints"][0]["words"] == []
    assert carried["constraints"][0]["layers"] == [
        {"id": "caption", "kind": "caption", "text": "Start End"},
    ]


def test_unlocated_thread_waits_for_one_answer_before_carry(feedback_project: Path) -> None:
    root = feedback_project
    first = save_version(root)
    _cut(root, label="next render")
    second = save_version(root)
    comment = _comment(start_frame=150)
    export = _export(root, [], version=second)
    export["series"] = {
        "id": "sample-series", "version": 2, "versions": [
            {"number": 1, "sha256": first.media["sha256"], "version_id": "v1"},
            {"number": 2, "sha256": second.media["sha256"], "version_id": "v2"},
        ], "threads": [{**comment_to_dict(comment), "version": 1,
                        "reply": None, "state": "open"}],
    }
    first_batch = build_batch(root, export)
    item = first_batch["items"][0]
    assert item["carried"] and item["comment"]["version"] == "v1"
    assert item["clarification"]["reason"] == "unlocated"
    question = pending_questions(root)
    assert len(question) == 1 and question[0]["batch"] == "b1"
    _cut(root, label="third render")
    third = save_version(root)
    later_export = _export(root, [], version=third)
    later_export["series"] = {**export["series"], "version": 3, "versions": [
        *export["series"]["versions"],
        {"number": 3, "sha256": third.media["sha256"], "version_id": "v3"},
    ]}
    second_batch = build_batch(root, later_export)
    assert second_batch["id"] == "b2" and second_batch["items"] == []
    assert second_batch["constraints"] == []
    assert pending_questions(root) == question
    answer_clarification(root, "b1", "v1", "c1", answer="The ending in the first version")
    third_batch = build_batch(root, later_export)
    assert third_batch["id"] == "b3" and len(third_batch["items"]) == 1
    carried = third_batch["items"][0]
    assert carried["carried"] and carried["comment"] == item["comment"]
    assert carried["disposition"] == "open"
    assert carried["clarification"]["answer"] == "The ending in the first version"
    assert load_batch(root, "b1")["items"][0]["clarification"]["carried_to"] == "b3"
    assert pending_questions(root) == answered_questions(root) == []
    assert build_batch(root, later_export)["items"] == []


@pytest.mark.parametrize("options", ["bad", [{}], [{"value": "", "label": "Caption"}],
                                     [{"value": "caption", "label": " "}],
                                     [{"value": "caption", "label": "One"},
                                      {"value": "caption", "label": "Two"}]])
def test_agent_refuses_invalid_options(feedback_project: Path, options) -> None:
    root = feedback_project
    build_batch(root, _export(root, [_comment()]))
    with pytest.raises(ValidationError):
        ask_clarification(root, "b1", None, "c1", "Which caption?", options=options)


def test_answer_carried_once_with_target_and_own_snapshots(feedback_project: Path) -> None:
    root = feedback_project
    first = save_version(root, layers=_layers(root))
    original = build_batch(root, _export(root, [_comment()]))
    answer_clarification(root, "b1", "v1", "c1", option="caption")
    _cut(root, label="next render")
    second = save_version(root)
    export = _export(root, [], version=second)
    export["series"] = {
        "id": "sample-series", "version": 2, "versions": [
            {"number": 1, "sha256": first.media["sha256"], "version_id": "v1"},
            {"number": 2, "sha256": second.media["sha256"], "version_id": "v2"},
        ], "threads": [{**comment_to_dict(_comment()), "version": 1,
                        "reply": None, "state": "needs_clarification"}],
    }
    batch = build_batch(root, export)
    assert len(batch["items"]) == 1
    item = batch["items"][0]
    assert item["carried"] and item["disposition"] == "open"
    assert item["comment"] == {**original["items"][0]["comment"], "target": "caption"}
    assert item["location"] == original["items"][0]["location"]
    assert item["location"]["layers_known"]
    assert item["clarification"]["option"] == "caption"
    assert item["clarification"]["answer"] is None
    assert load_batch(root, "b1")["items"][0]["clarification"]["carried_to"] == "b2"
    assert pending_questions(root) == answered_questions(root) == []
    for batch_id in ("b1", "b2"):
        with pytest.raises(CarriedQuestionError):
            answer_clarification(root, batch_id, "v1", "c1", answer="Another answer")
    assert build_batch(root, export)["items"] == []


def test_answers_from_all_earlier_batches_and_concurrent_carry(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    build_batch(root, _export(root, [_comment()]))
    build_batch(root, _export(root, [_comment("c2")]))
    answer_clarification(root, "b1", "v1", "c1", answer="The background graphic")
    answer_clarification(root, "b2", "v1", "c2", option="story")
    with ThreadPoolExecutor(max_workers=2) as executor:
        batches = list(executor.map(lambda _: build_batch(root, _export(root, [])), range(2)))
    assert sorted(len(batch["items"]) for batch in batches) == [0, 2]
    assert answered_questions(root) == []
    assert load_batch(root, "b1")["items"][0]["clarification"]["carried_to"] == "b3"
    carried = load_batch(root, "b3")["items"]
    assert carried[0]["clarification"]["answer"] == "The background graphic"
    assert carried[1]["comment"]["target"] == "story"


def test_ambiguous_fix_does_not_release_an_existing_keep(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    kept = build_batch(root, _export(root, [_comment(kind="keep")]))
    batch = build_batch(root, _export(root, [_comment("c2")]))
    assert batch["constraints"] == kept["constraints"]
    assert batch["released"] == []


def test_batch_carries_only_selected_answers_and_refuses_stale_references(
    feedback_project: Path,
) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    build_batch(root, _export(root, [_comment(), _comment("c2")]))
    first = {"batch": "b1", "comment": {"version": "v1", "id": "c1"}}
    second = {"batch": "b1", "comment": {"version": "v1", "id": "c2"}}
    with pytest.raises(FeedbackError, match="no longer answered"):
        build_batch(root, _export(root, []), answers=[first])
    answer_clarification(root, "b1", "v1", "c1", option="caption")
    answer_clarification(root, "b1", "v1", "c2", option="element")
    batch = build_batch(root, _export(root, []), answers=[first])
    assert [item["comment"]["id"] for item in batch["items"]] == ["c1"]
    assert [item["comment"] for item in answered_questions(root)] == [second["comment"]]
    with pytest.raises(FeedbackError, match="already carried"):
        build_batch(root, _export(root, []), answers=[first])
    assert not (root / "feedback" / "b3.json").exists()
    assert build_batch(root, _export(root, []), answers=[])["items"] == []
    assert len(answered_questions(root)) == 1


def test_round_origin_returns_existing_batch_without_consuming_answers_again(
    feedback_project: Path,
) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    build_batch(root, _export(root, [_comment()]))
    answer_clarification(root, "b1", "v1", "c1", option="caption")
    ref = {"batch": "b1", "comment": {"version": "v1", "id": "c1"}}
    origin = {"series": "series", "round": "n1"}
    batch = build_batch(root, _export(root, []), answers=[ref], origin=origin)
    files = [root / "feedback" / "b1.json", root / "feedback" / "b2.json"]
    before = [(file.read_bytes(), file.stat().st_mtime_ns) for file in files]
    assert build_batch(root, _export(root, [_comment("c2")]),
                       answers=[ref], origin=origin) == batch
    assert [(file.read_bytes(), file.stat().st_mtime_ns) for file in files] == before
    assert not (root / "feedback" / "b3.json").exists()
