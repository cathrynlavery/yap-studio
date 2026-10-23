"""Compare the final's word occurrences with the reviewed version."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_feedback_versions import _json, _update_cut
from test_feedback_versions import project as version_project

from yap_studio.feedback.versions import save_version
from yap_studio.receipt import sha256_file


def saved_project(tmp_path: Path, *, baseline_removal: bool = False,
                  repeated: bool = False) -> Path:
    root = version_project.__wrapped__(tmp_path)
    _json(root / "ingest/t1/words.json", {"words": [
        {"id": "t1:0", "text": "First", "start": 0.1, "end": 0.5},
        {"id": "t1:1", "text": "Second", "start": 2.1, "end": 2.5},
    ]})
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"] = [
        {"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"time": 1}},
        {"id": "s2", "take": "t1", "from": {"time": 2}, "to": {"time": 3}},
    ]
    if baseline_removal:
        _removal(root)
    if repeated:
        words = json.loads((root / "cut/cut-words.json").read_bytes())
        words["words"].append({**words["words"][0], "segment": "s3", "start": 1.6, "end": 1.9})
        _json(root / "cut/cut-words.json", words)
        edit["segments"].append({**edit["segments"][0], "id": "s3"})
    _json(root / "edit.json", edit)
    _update_cut(root, content=(root / "cut/cut.mp4").read_bytes())
    save_version(root, note="Story cut")
    final = root / "final/result.mp4"
    final.parent.mkdir()
    final.write_bytes(b"synthetic final")
    return root


def _removal(root, **fields):
    entry = {"id": "r13", "take": "t1", "start": 0, "end": 1,
             "kind": "filler", "status": "applied", "words": ["t1:0"],
             "by": "fillers", "listen": False, "evidence": {"pass": True}, **fields}
    _json(root / "removals.json", {"schema": "yap-studio/removals@1", "removals": [entry]})


def _drop(root, index=0):
    words = json.loads((root / "cut/cut-words.json").read_bytes())
    words["words"].pop(index)
    _json(root / "cut/cut-words.json", words)


def _check(root):
    from yap_studio.checks.kept_words import check_kept_words

    return check_kept_words(root, root / "final/result.mp4")


def test_new_automatic_removal_fails_and_names_word_time_and_removal(tmp_path):
    root = saved_project(tmp_path)
    _removal(root)
    _drop(root)
    result = _check(root)
    assert result["status"] == "fail" and result["baseline"] == "v1"
    assert result["total"] == 2 and result["kept"] == 1
    assert result["dropped"] == [{"id": "t1:0", "segment": "s1", "text": "First", "time": 0.1,
                                   "why": "unapproved", "removal": "r13"}]


@pytest.mark.parametrize("fields", [
    {"evidence": {"decision": True}}, {"by": "manual"},
    {"words": [], "by": "manual"},
])
def test_creator_approved_removal_passes(tmp_path, fields):
    root = saved_project(tmp_path)
    _removal(root, **fields)
    _drop(root)
    result = _check(root)
    assert result["status"] == "pass" and result["dropped"][0]["why"] == "approved"


def test_removal_already_applied_in_baseline_is_approved(tmp_path):
    root = saved_project(tmp_path, baseline_removal=True)
    _removal(root, id="r99")
    _drop(root)
    assert _check(root)["dropped"][0]["why"] == "approved"


def test_removal_coverage_uses_source_clock_and_report_uses_baseline_clock(tmp_path):
    root = saved_project(tmp_path)
    _removal(root, start=2, end=3, words=[], by="manual")
    _drop(root, 1)
    result = _check(root)
    assert result["status"] == "pass"
    assert result["dropped"] == [{"id": "t1:1", "segment": "s2", "text": "Second", "time": 1.1,
                                   "why": "approved", "removal": "r13"}]


@pytest.mark.parametrize("field,value", [
    ("take", "t2"), ("from", {"time": 0.2}), ("to", {"time": 0.2}),
])
def test_edit_change_classifies_missing_word_as_story(tmp_path, field, value):
    root = saved_project(tmp_path)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0][field] = value
    _json(root / "edit.json", edit)
    _drop(root)
    _drop(root)
    result = _check(root)
    assert result["status"] == "fail"
    assert [(word["segment"], word["why"]) for word in result["dropped"]] == [
        ("s1", "story"), ("s2", "unexplained"),
    ]


@pytest.mark.parametrize("change", ["end_hold_s", "note", "other_segment"])
def test_unrelated_edit_change_does_not_explain_missing_word(tmp_path, change):
    root = saved_project(tmp_path)
    edit = json.loads((root / "edit.json").read_bytes())
    if change == "end_hold_s":
        edit["end_hold_s"] = 0.1
    elif change == "note":
        edit["segments"][0]["note"] = "Keep the opening words"
    else:
        edit["segments"][1]["to"] = {"time": 2.6}
    _json(root / "edit.json", edit)
    _drop(root)

    result = _check(root)

    assert result["status"] == "fail"
    assert result["dropped"][0]["why"] == "unexplained"
    assert result["dropped"][0]["segment"] == "s1"


def test_removed_segment_explains_only_its_missing_word(tmp_path):
    root = saved_project(tmp_path)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"].pop(0)
    _json(root / "edit.json", edit)
    _drop(root)
    _drop(root)

    result = _check(root)

    assert result["status"] == "fail"
    assert [(word["segment"], word["why"]) for word in result["dropped"]] == [
        ("s1", "story"), ("s2", "unexplained"),
    ]


def test_changed_transcript_skips_comparison(tmp_path):
    root = saved_project(tmp_path)
    words = json.loads((root / "cut/cut-words.json").read_bytes())
    words["words_sha256"]["t1"] = "c" * 64
    _json(root / "cut/cut-words.json", words)
    result = _check(root)
    assert result["status"] == "skipped"
    assert "transcript changed since v1" in result["reason"]
    assert "t1" in result["reason"]


@pytest.mark.parametrize("story_changed", [False, True])
def test_take_only_in_baseline_still_compares_missing_words(tmp_path, story_changed):
    root = saved_project(tmp_path)
    words = json.loads((root / "cut/cut-words.json").read_bytes())
    words["words_sha256"] = {}
    words["words"] = []
    _json(root / "cut/cut-words.json", words)
    if story_changed:
        edit = json.loads((root / "edit.json").read_bytes())
        edit["words_sha256"] = {}
        edit["segments"] = []
        _json(root / "edit.json", edit)

    result = _check(root)

    assert result["status"] == ("pass" if story_changed else "fail")
    assert result["total"] == 2 and result["kept"] == 0
    why = "story" if story_changed else "unexplained"
    assert result["dropped"] == [
        {"id": "t1:0", "segment": "s1", "text": "First", "time": 0.1,
         "why": why, "removal": None},
        {"id": "t1:1", "segment": "s2", "text": "Second", "time": 1.1,
         "why": why, "removal": None},
    ]
    assert result["added"] == []


def test_take_only_in_final_adds_words_without_skipping(tmp_path):
    root = saved_project(tmp_path)
    words = json.loads((root / "cut/cut-words.json").read_bytes())
    words["words_sha256"]["t2"] = "b" * 64
    words["words"].append({"id": "t2:0", "text": "Added", "segment": "s3",
                           "start": 5.678, "end": 5.9})
    _json(root / "cut/cut-words.json", words)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t2"] = "b" * 64
    _json(root / "edit.json", edit)

    result = _check(root)

    assert result["status"] == "pass"
    assert result["total"] == result["kept"] == 2
    assert result["dropped"] == []
    assert result["added"] == [{"id": "t2:0", "segment": "s3", "text": "Added", "time": 5.68}]


@pytest.mark.parametrize("recorded", [False, True])
def test_rerun_compares_with_parent(tmp_path, monkeypatch, recorded):
    root = saved_project(tmp_path)
    _drop(root)
    _update_cut(root)
    newest = save_version(root)
    final = root / "final/result.mp4"
    final.write_bytes(newest.media_path.read_bytes())
    if recorded:
        _json(root / "final/receipt.json", {"schema": "yap-studio/stage-receipt@1", "outputs": [
            {"path": "final/result.mp4", "sha256": sha256_file(final)},
        ]})
        monkeypatch.setattr("yap_studio.checks.kept_words.sha256_file",
                            lambda _: pytest.fail("use the recorded final hash"))
    result = _check(root)
    assert result["baseline"] == "v1" and result["status"] == "fail"
    assert result["dropped"][0]["why"] == "unexplained"


def test_rollback_pointer_selects_reviewed_baseline(tmp_path):
    root = saved_project(tmp_path)
    _drop(root)
    _update_cut(root)
    save_version(root)
    _json(root / "versions/current.json", {
        "schema": "yap-studio/version-pointer@1", "current": "v1",
        "set_at": "2026-10-19T12:00:00Z", "reason": "rollback",
    })
    result = _check(root)
    assert result["baseline"] == "v1" and result["status"] == "fail"


def test_occurrences_are_counted_and_additions_use_final_clock(tmp_path):
    root = saved_project(tmp_path, repeated=True)
    _drop(root, 2)
    words = json.loads((root / "cut/cut-words.json").read_bytes())
    words["words"].append({"id": "t1:2", "segment": "s4", "text": "Added",
                           "start": 1.678, "end": 1.9})
    _json(root / "cut/cut-words.json", words)
    result = _check(root)
    assert result["total"] == 3 and result["kept"] == 2
    assert len(result["dropped"]) == 1 and result["dropped"][0]["time"] == 1.6
    assert result["added"] == [{"id": "t1:2", "segment": "s4", "text": "Added", "time": 1.68}]


@pytest.mark.parametrize("renamed", [False, True])
def test_lost_first_occurrence_keeps_later_segment_match(tmp_path, renamed):
    root = saved_project(tmp_path, repeated=True)
    _drop(root)
    if renamed:
        words = json.loads((root / "cut/cut-words.json").read_bytes())
        words["words"][0]["segment"] = "s4"
        _json(root / "cut/cut-words.json", words)
        edit = json.loads((root / "edit.json").read_bytes())
        edit["segments"][1]["id"] = "s4"
        _json(root / "edit.json", edit)

    result = _check(root)

    assert result["total"] == 3 and result["kept"] == 2
    assert len(result["dropped"]) == 1
    assert result["dropped"][0]["time"] == 0.1
    assert result["dropped"][0]["segment"] == "s1"
    assert result["dropped"][0]["why"] == "unexplained"
    assert result["added"] == []


def test_added_first_occurrence_keeps_existing_segment_match(tmp_path):
    root = saved_project(tmp_path)
    words = json.loads((root / "cut/cut-words.json").read_bytes())
    words["words"].insert(0, {**words["words"][0], "segment": "s3", "start": 0.01, "end": 0.05})
    _json(root / "cut/cut-words.json", words)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"].insert(0, {**edit["segments"][0], "id": "s3"})
    _json(root / "edit.json", edit)

    result = _check(root)

    assert result["status"] == "pass"
    assert result["total"] == result["kept"] == 2
    assert result["dropped"] == []
    assert len(result["added"]) == 1
    assert result["added"][0]["time"] == 0.01
    assert result["added"][0]["segment"] == "s3"


def test_no_version_skips(tmp_path):
    from yap_studio.checks.kept_words import check_kept_words

    result = check_kept_words(tmp_path, tmp_path / "final/result.mp4")
    assert result["status"] == "skipped"
    assert result["reason"] == "no saved version to compare with"
