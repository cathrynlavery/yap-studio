"""Feedback plans apply once against saved bases and recover from their journal."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_feedback_batch import _cut, _export, _json
from test_feedback_locate import _comment

from yap_studio.feedback import apply as apply_module
from yap_studio.feedback.apply import apply_operations, fillers_writer
from yap_studio.feedback.batch import (
    FeedbackError,
    StaleBatchError,
    _write_batch,
    build_batch,
    current_inputs,
    load_batch,
)
from yap_studio.feedback.edits import OPERATIONS_SCHEMA, plan_edit
from yap_studio.feedback.removed import _words, removed_ranges
from yap_studio.feedback.versions import (
    _snapshot_json,
    load_version,
    save_version,
    version_cut_words,
    version_pieces,
)
from yap_studio.project.lock import project_lock
from yap_studio.project.model import CutWord, CutWords, Word, Words
from yap_studio.receipt import sha256_file


def _project(root: Path, *, filler: bool = False, save: bool = True,
             request: dict | None = None) -> Path:
    _json(root / "project.json", {"schema": "yap-studio/project@1", "name": "sample-video",
                                 "takes": [{"id": "t1", "path": "takes/t1.mp4"}]})
    texts = ("Start", "um,", "so", "about.", "and", "that", "is", "end.",
             "One", "small", "pause", "finish.")
    source = Words("t1", "b" * 64, "source", 0, "none", None, "c" * 64, tuple(
        Word(f"t1:{index}", text, index * 2 + 0.5, index * 2 + 1)
        for index, text in enumerate(texts)
    ))
    _json(root / "ingest" / "t1" / "words.json", source.to_dict())
    digest = sha256_file(root / "ingest" / "t1" / "words.json")
    _json(root / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {"t1": digest},
                              "segments": [
                                  {"id": segment, "take": "t1", "from": {"word": f"t1:{first}"},
                                   "to": {"word": f"t1:{last}"}, "note": "preserve this metadata"}
                                  for segment, first, last in (("s1", 0, 3), ("s2", 7, 7),
                                                               ("s3", 8, 11))
                              ], "end_hold_s": 0.5})
    _json(root / "timeline" / "agent.json", {"schema": "yap-studio/timeline@1", "elements": []})
    if filler:
        _json(root / "removals.json", {"schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 1.75, "end": 5.75,
             "kind": "filler", "status": "applied"},
        ]})
    kept = ([("s1", 0, 0.1), ("s1", 3, 1.1), ("s2", 7, 2.1)] if filler else
            [("s1", index, 0.1 + index * 0.3) for index in range(4)] + [("s2", 7, 1.5)])
    kept += [("s3", index, (3.05 + (index - 8) * 0.2) if filler else
              2.75 + (index - 8) * 0.3) for index in range(8, 12)]
    cut_words = CutWords("output", {"t1": digest}, tuple(
        CutWord(f"t1:{index}", segment, texts[index], start, start + 0.15)
        for segment, index, start in kept
    ))
    _cut(root, words=cut_words, segments=("s1", "s2", "s3"))
    receipt_path = root / "cut" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"]["words:t1"] = digest
    spans = ([("s1", "0", "7.2"), ("s2", "14", "15.2"), ("s3", "16", "24")]
             if not filler else [("s1", "0", "1.75"), ("s1", "5.75", "7.2"),
                                 ("s2", "14", "15.2"), ("s3", "16", "24")])
    frames = 120 // len(spans)
    receipt["plan"]["pieces"] = [
        {"segment_id": segment, "take": "t1", "out_f0": index * frames, "n_frames": frames,
         "src_t0": start, "src_t1": end}
        for index, (segment, start, end) in enumerate(spans)
    ]
    _json(receipt_path, receipt)
    if save:
        save_version(root)
    comments = [_comment("c1", target="story"), _comment("c2", target="story"),
                _comment("c3", kind="keep", target="story")]
    if request is not None:
        comments[0] = replace(comments[0], request=request)
    build_batch(root, _export(root, comments))
    return root


@pytest.fixture
def apply_project(tmp_path: Path) -> Path:
    return _project(tmp_path / "project")


def _document(*operations: dict, comment: str = "v1:c1") -> dict:
    return {"schema": OPERATIONS_SCHEMA, "batch": "b1",
            "items": [{"comment": comment, "operations": list(operations)}]}


def _trim() -> dict:
    return {"id": "trim", "kind": "trim", "segment": "s1", "edge": "end", "word": "t1:2"}


def _middle() -> dict:
    return {"id": "middle", "kind": "remove",
            "from": {"word": "t1:9", "segment": "s3"},
            "to": {"word": "t1:10", "segment": "s3"}}


def _tree(root: Path, *, lock: bool = True) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*")
            if path.is_file() and (lock or path.relative_to(root).as_posix() != ".yap/lock")}


def _planned(root: Path, document: dict):
    base = load_version(root, "v1")
    return plan_edit(_snapshot_json(base, "edit"), version_cut_words(base),
                     {"t1": _words(root, "t1", version_pieces(base)["inputs"])},
                     load_batch(root, "b1")["items"], document,
                     ranges={"v1": removed_ranges(root, base)},
                     pieces={"v1": version_pieces(base)})


def test_trim_and_whole_segment_remove_applied(apply_project: Path, monkeypatch) -> None:
    root = apply_project
    monkeypatch.setattr(apply_module, "fillers_writer",
                        lambda: pytest.fail("a story plan must not load fillers"))
    document = _document(_trim())
    drop = {"id": "drop", "kind": "remove", "segment": "s3"}
    document["items"].append({"comment": "v1:c2", "operations": [drop]})
    document["items"].append({"comment": "v1:c3", "operations": [{"id": "keep", "kind": "keep"}]})
    plan = _planned(root, document)
    before = current_inputs(root)
    result = apply_operations(root, document)
    assert json.loads((root / "edit.json").read_text()) == plan.edit
    after = current_inputs(root)
    assert before["edit.json"] != after["edit.json"]
    assert result == {"batch": "b1", "dry_run": False, "changes": list(plan.changes),
                      "edit": plan.edit["segments"], "removal_actions": [], "status": "applied"}
    batch = load_batch(root, "b1")
    for entry in document["items"]:
        item = next(item for item in batch["items"]
                    if item["comment"]["id"] == entry["comment"].split(":")[1])
        assert item["disposition"] == "applied"
        assert item["operations"] == [
            {**operation, "inputs_before": before, "inputs_after": after,
             "summary": next(change["summary"] for change in plan.changes
                             if change["operation"] == operation["id"])}
            for operation in entry["operations"]
        ]
    assert batch["expected_inputs"] == after
    application = batch["application"]
    assert set(application) == {"status", "operations", "operations_sha256", "inputs_before",
                                "edit_sha256_before", "edit_sha256_after", "removal_actions",
                                "started_at", "finished_at", "edit", "changes", "removals",
                                "removals_sha256"}
    assert application["status"] == "applied"
    assert application["operations"] == document
    assert application["operations_sha256"] == hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    assert application["inputs_before"] == before
    assert application["edit_sha256_before"] == before["edit.json"]
    assert application["edit_sha256_after"] == after["edit.json"]
    assert application["removals"] is None
    assert application["removals_sha256"] == before["removals.json"] is None
    assert application["edit"] == plan.edit and application["changes"] == list(plan.changes)
    assert application["started_at"] <= application["finished_at"]


def test_same_file_is_already_applied_and_bytes_unchanged(apply_project: Path) -> None:
    document = _document(_trim())
    apply_operations(apply_project, document)
    before = _tree(apply_project, lock=False)
    with pytest.raises(FeedbackError, match="already has an applied plan"):
        apply_operations(apply_project, document)
    assert _tree(apply_project, lock=False) == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_keep_only_application_refuses_a_second_plan(apply_project: Path, dry_run: bool) -> None:
    apply_operations(apply_project, _document({"id": "keep", "kind": "keep"}, comment="v1:c3"))
    before = _tree(apply_project, lock=False)
    error = None
    try:
        apply_operations(apply_project, _document(_trim()), dry_run=dry_run)
    except FeedbackError as caught:
        error = caught
    assert error is not None, "an applied batch must refuse every new operations file"
    assert str(error) == (
        "This batch already has an applied plan; further notes go into the next round."
    )
    assert _tree(apply_project, lock=False) == before


def test_operation_id_recorded_on_another_item_is_refused(apply_project: Path) -> None:
    batch = load_batch(apply_project, "b1")
    batch["items"][1]["operations"] = [{"id": "trim"}]
    _write_batch(apply_project, batch)
    before = _tree(apply_project, lock=False)
    with pytest.raises(FeedbackError, match="Operation trim was already applied"):
        apply_operations(apply_project, _document(_trim()))
    assert _tree(apply_project, lock=False) == before


def test_stale_base_names_reviewed_and_current_versions(apply_project: Path) -> None:
    _cut(apply_project, label="next render")
    save_version(apply_project)
    before = _tree(apply_project, lock=False)
    with pytest.raises(FeedbackError) as error:
        apply_operations(apply_project, _document(_trim()))
    assert str(error.value) == "Batch base v1 is not the current version (v2)."
    assert _tree(apply_project, lock=False) == before


@pytest.mark.parametrize("filenames", [
    ["edit.json"], ["timeline/agent.json"], ["edit.json", "timeline/agent.json"],
    ["timeline/new.json"],
])
def test_stale_inputs_name_every_changed_file(apply_project: Path, filenames: list[str]) -> None:
    for filename in filenames:
        path = apply_project / filename
        if path.exists():
            path.write_bytes(path.read_bytes() + b" ")
        else:
            _json(path, {"schema": "yap-studio/timeline@1", "elements": []})
    before = _tree(apply_project, lock=False)
    with pytest.raises(StaleBatchError) as error:
        apply_operations(apply_project, _document(_trim()))
    assert error.value.changed == sorted(filenames)
    assert _tree(apply_project, lock=False) == before


def test_base_edit_snapshot_must_match_even_when_expected_inputs_match(apply_project: Path) -> None:
    path = apply_project / "edit.json"
    path.write_bytes(path.read_bytes() + b" ")
    batch = load_batch(apply_project, "b1")
    batch["expected_inputs"] = current_inputs(apply_project)
    _write_batch(apply_project, batch)
    with pytest.raises(StaleBatchError) as error:
        apply_operations(apply_project, _document(_trim()))
    assert error.value.changed == ["edit.json"]
    assert "application" not in load_batch(apply_project, "b1")


def test_current_cut_batch_requires_saved_base(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", save=False)
    with pytest.raises(FeedbackError) as error:
        apply_operations(root, _document(_trim()))
    assert str(error.value) == "This batch reviews the current cut without a saved base version."
    assert error.value.suggested_action == (
        "Save versions with yap-studio versions save before reviewing."
    )
    assert "application" not in load_batch(root, "b1")


def test_dry_run_leaves_every_tree_byte_unchanged(apply_project: Path) -> None:
    document = _document(_trim())
    plan = _planned(apply_project, document)
    before = _tree(apply_project)
    assert apply_operations(apply_project, document, dry_run=True) == {
        "batch": "b1", "dry_run": True, "changes": list(plan.changes),
        "edit": plan.edit["segments"], "removal_actions": [], "status": "planned",
    }
    assert _tree(apply_project) == before


def test_story_restore_uses_items_request_and_extends_previous_segment(tmp_path: Path) -> None:
    request = {"type": "restore", "range": "s1.0", "take": "t1", "start": 7.2, "end": 14.0,
               "text": "and that is"}
    root = _project(tmp_path / "project", request=request)
    document = _document({"id": "back", "kind": "restore"})
    plan = _planned(root, document)
    result = apply_operations(root, document)
    assert result["changes"] == [{
        "operation": "back", "comment": "v1:c1", "kind": "restore",
        "summary": 'Brought back "and that is" after s1.', "segments": ["s1"],
        "words": [{"id": f"t1:{index}", "text": text}
                  for index, text in ((4, "and"), (5, "that"), (6, "is"))],
    }]
    assert json.loads((root / "edit.json").read_text()) == plan.edit
    assert plan.edit["segments"][0]["to"] == {"word": "t1:6"}
    assert result["removal_actions"] == []


def test_restore_from_unsaved_comment_version_is_refused(apply_project: Path) -> None:
    batch = load_batch(apply_project, "b1")
    batch["items"][0]["comment"]["version"] = "v9"
    _write_batch(apply_project, batch)
    before = _tree(apply_project, lock=False)
    with pytest.raises(FeedbackError) as error:
        apply_operations(apply_project, _document(
            {"id": "back", "kind": "restore", "range": "s1.0"}, comment="v9:c1"))
    assert str(error.value) == "Restore comment v9:c1 needs saved version v9."
    assert _tree(apply_project, lock=False) == before


class FakeRemovalWriter:
    """Write decisions under our own lock, just as the fillers stage will."""

    def __init__(self, *, crash: bool = False) -> None:
        self.calls: list[tuple] = []
        self.crash = crash

    def add_manual(self, project_dir: Path, take: str, start: float, end: float, *,
                   reason: str | None = None) -> dict:
        self.calls.append(("add_manual", project_dir, take, start, end, reason))
        with project_lock(project_dir, timeout_s=0):
            path = project_dir / "removals.json"
            data = json.loads(path.read_text()) if path.exists() else {
                "schema": "yap-studio/removals@1", "removals": [],
            }
            removal = {"id": "manual1", "take": take, "start": start, "end": end,
                       "kind": "manual", "status": "applied", "reason": reason}
            data["removals"].append(removal)
            _json(path, data)
        if self.crash:
            self.crash = False
            raise RuntimeError("crash after first writer call")
        return removal

    def set_removal(self, project_dir: Path, identifier: str, status: str, *,
                    reason: str | None = None) -> dict:
        self.calls.append(("set_removal", project_dir, identifier, status, reason))
        with project_lock(project_dir, timeout_s=0):
            path = project_dir / "removals.json"
            data = json.loads(path.read_text())
            removal = next(item for item in data["removals"] if item["id"] == identifier)
            removal.update(status=status, reason=reason)
            removal.setdefault("evidence", {})["decision"] = True
            _json(path, data)
        return removal


def _cut_note() -> dict:
    return {"type": "cut", "start": 0.4, "end": 0.85, "words": [
        {"id": "t1:1", "segment": "s1", "text": "um,"},
        {"id": "t1:2", "segment": "s1", "text": "so"},
    ]}


class FakeJoinTester:
    def __init__(self, *, fail: bool = False, nudge: bool = False) -> None:
        self.calls: list[list[dict]] = []
        self.contexts: list[dict] = []
        self.fail = fail
        self.nudge = nudge

    def test_cuts(self, project_dir: Path, spans: list[dict], *,
                  edit=None, removals=(), rejects=()) -> list[dict]:
        self.calls.append(copy.deepcopy(spans))
        self.contexts.append({"edit": copy.deepcopy(edit), "removals": copy.deepcopy(removals),
                              "rejects": tuple(rejects)})
        return [{"segment": span["segment"], "edge": span["edge"],
                 "pass": not self.fail or "label" in span,
                 "start": span["start"] + (1 / 30 if self.nudge else 0),
                 "end": span["end"], "nudge": [1, 0] if self.nudge else [0, 0],
                 "words": "fail" if self.fail and "label" not in span else "pass",
                 "diff": ["missing join words: about"] if self.fail else [],
                 "removed_words": span["texts"], "click_db": -40, "listen": self.nudge}
                for span in spans]


@pytest.mark.parametrize("nudge", [False, True])
def test_cut_applies_tested_middle_boundaries(tmp_path, nudge) -> None:
    root = _project(tmp_path / "project", request=_cut_note())
    joins, writer = FakeJoinTester(nudge=nudge), FakeRemovalWriter()
    result = apply_operations(root, _document({"id": "cut", "kind": "cut"}),
                              joins=joins, removals=writer)
    change, = result["changes"]
    assert len(joins.calls) == 1
    assert change["join"][0]["pass"] is True
    assert writer.calls[0][3:5] == (1.75 + (1 / 30 if nudge else 0), 5.75)
    assert ("listen to this join" in change["summary"]) == nudge
    assert load_batch(root, "b1")["items"][0]["disposition"] == "applied"


def test_cut_declines_with_options_and_other_note_still_applies(tmp_path) -> None:
    root = _project(tmp_path / "project", request=_cut_note())
    before = json.loads((root / "edit.json").read_bytes())
    joins, writer = FakeJoinTester(fail=True), FakeRemovalWriter()
    document = _document({"id": "cut", "kind": "cut"})
    document["items"].append({"comment": "v1:c2", "operations": [
        {"id": "drop", "kind": "remove", "segment": "s2"},
    ]})
    result = apply_operations(root, document, joins=joins, removals=writer)
    decline, = result["dispositions"]
    assert decline["kind"] == "cut" and decline["segments"] == decline["words"] == []
    assert decline["join"][0]["pass"] is False
    assert [option["label"] for option in decline["options"]] == [
        'also cut "Start"', 'also cut "about."', 'keep "um,"', 'keep "so"',
    ]
    assert decline["summary"].startswith(
        'Not cut: the join at "um, so" failed (missing join words: about). Clean options: ')
    assert len(joins.calls) == 2 and len(joins.calls[1]) == 4
    assert not writer.calls and not (root / "removals.json").exists()
    after = json.loads((root / "edit.json").read_bytes())
    assert after["segments"] == [before["segments"][0], before["segments"][2]]
    item = load_batch(root, "b1")["items"][0]
    assert item["operations"] == [] and item["disposition"] == "declined"
    assert item["evidence"] == {"join": decline["join"], "options": decline["options"],
                                "reason": decline["summary"]}


def test_cut_dry_run_tests_and_writes_nothing(tmp_path) -> None:
    root = _project(tmp_path / "project", request=_cut_note())
    joins = FakeJoinTester()
    before = _tree(root)
    result = apply_operations(root, _document({"id": "cut", "kind": "cut"}), dry_run=True,
                              joins=joins, removals=FakeRemovalWriter())
    assert result["status"] == "planned" and len(joins.calls) == 1
    assert _tree(root) == before


def test_cut_tester_receives_final_edit_and_all_manual_actions(tmp_path) -> None:
    root = _project(tmp_path / "project", request=_cut_note())
    document = _document({"id": "cut", "kind": "cut"})
    document["items"].append({"comment": "v1:c2", "operations": [_middle()]})
    plan = _planned(root, document)
    joins = FakeJoinTester()
    apply_operations(root, document, dry_run=True, joins=joins, removals=FakeRemovalWriter())
    assert joins.contexts == [{"edit": plan.edit, "rejects": (), "removals": [
        {"action": index, "take": action.take, "start": action.start, "end": action.end}
        for index, action in enumerate(plan.removal_actions) if action.action == "add_manual"
    ]}]
    assert joins.calls[0][0]["action"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_cut_tester_receives_reject_ids_for_spans_and_options(tmp_path, fail) -> None:
    request = {"type": "cut", "start": 3.05, "end": 3.5, "words": [
        {"id": "t1:9", "segment": "s3", "text": "small"},
        {"id": "t1:10", "segment": "s3", "text": "pause"},
    ]}
    root = _project(tmp_path / "project", filler=True, request=request)
    document = _document({"id": "cut", "kind": "cut"})
    document["items"].append({"comment": "v1:c2", "operations": [
        {"id": "back", "kind": "restore", "range": "s1.0"},
    ]})
    joins = FakeJoinTester(fail=fail)
    apply_operations(root, document, dry_run=True, joins=joins, removals=FakeRemovalWriter())
    assert len(joins.calls) == (2 if fail else 1)
    assert all(context["rejects"] == ("r1",) for context in joins.contexts)


def test_cut_tester_receives_combined_edge_span_and_evidence(tmp_path) -> None:
    request = {"type": "cut", "start": 0.1, "end": 0.25, "words": [
        {"id": "t1:0", "segment": "s1", "text": "Start"},
    ]}
    root = _project(tmp_path / "project", request=request)
    joins = FakeJoinTester()
    document = _document({"id": "cut", "kind": "cut"},
                         {"id": "trim-more", "kind": "trim", "segment": "s1",
                          "edge": "start", "word": "t1:2"})
    result = apply_operations(root, document, dry_run=True, joins=joins)
    span, = joins.calls[0]
    assert span["words"] == ["t1:0", "t1:1"] and span["end"] == 3.75
    assert span["combined_with"] == ["trim-more"] and span["options"] == []
    assert result["changes"][0]["join"][0]["combined_with"] == ["trim-more"]


def test_cut_pending_resume_reuses_tested_plan(tmp_path, monkeypatch) -> None:
    root = _project(tmp_path / "project", request=_cut_note())
    joins, writer = FakeJoinTester(nudge=True), FakeRemovalWriter(crash=True)
    document = _document({"id": "cut", "kind": "cut"})
    with pytest.raises(RuntimeError, match="crash after first writer call"):
        apply_operations(root, document, joins=joins, removals=writer)
    monkeypatch.setattr(apply_module, "join_tester", lambda: pytest.fail("must reuse join"))
    result = apply_operations(root, document, removals=writer)
    assert len(joins.calls) == len(writer.calls) == 1
    assert result["changes"][0]["join"][0]["nudge"] == [1, 0]


def test_plans_without_cuts_never_load_join_tester(apply_project, monkeypatch) -> None:
    monkeypatch.setattr(apply_module, "join_tester", lambda: pytest.fail("no cut spans"))
    apply_operations(apply_project, _document(_trim()))


def test_cut_drop_records_empty_join_without_loading_tester(tmp_path, monkeypatch) -> None:
    request = {"type": "cut", "start": 1.5, "end": 1.65, "words": [
        {"id": "t1:7", "segment": "s2", "text": "end."},
    ]}
    root = _project(tmp_path / "project", request=request)
    monkeypatch.setattr(apply_module, "join_tester", lambda: pytest.fail("no cut spans"))
    result = apply_operations(root, _document({"id": "cut", "kind": "cut"}))
    assert result["changes"][0]["join"] == []
    assert [segment["id"] for segment in result["edit"]] == ["s1", "s3"]


def test_failed_cut_replans_whole_note_and_retests_other_cuts(tmp_path) -> None:
    request = _cut_note()
    request["words"].append({"id": "t1:7", "segment": "s2", "text": "end."})
    root = _project(tmp_path / "project", request=request)
    batch = load_batch(root, "b1")
    batch["items"][1]["request"] = {
        "type": "cut", "start": 3.05, "end": 3.5, "words": [
            {"id": "t1:9", "segment": "s3", "text": "small"},
            {"id": "t1:10", "segment": "s3", "text": "pause"},
        ],
    }
    _write_batch(root, batch)

    class MixedJoins(FakeJoinTester):
        def test_cuts(self, project_dir, spans, **kwargs):
            results = super().test_cuts(project_dir, spans, **kwargs)
            for span, result in zip(spans, results, strict=True):
                result["pass"] = span["segment"] == "s3"
            return results

    joins, writer = MixedJoins(nudge=True), FakeRemovalWriter()
    document = _document({"id": "cut", "kind": "cut"},
                         {"id": "tail", "kind": "trim", "segment": "s1",
                          "edge": "end", "word": "t1:0"})
    document["items"].append({"comment": "v1:c2", "operations": [{"id": "cut2", "kind": "cut"}]})
    result = apply_operations(root, document, joins=joins, removals=writer)
    assert len(joins.calls) == 3 and len(joins.calls[0]) == 2
    assert all(span["segment"] == "s1" for span in joins.calls[1])
    assert [span["operation"] for span in joins.calls[2]] == ["cut2"]
    assert [len(context["removals"]) for context in joins.contexts] == [2, 2, 1]
    assert [change["operation"] for change in result["changes"]] == ["cut2", "cut"]
    assert [segment["id"] for segment in result["edit"]] == ["s1", "s2", "s3"]
    assert result["edit"][2]["to"] == {"word": "t1:11"}
    assert writer.calls[0][3:5] == (17.75 + 1 / 30, 21.75)
    assert result["dispositions"][0]["options"] == []
    assert result["dispositions"][0]["summary"].endswith(
        "No clean cut within one word; ask the creator.")


def test_cut_retests_when_decline_changes_another_cut_span(tmp_path) -> None:
    root = _project(tmp_path / "project", request=_cut_note())
    batch = load_batch(root, "b1")
    batch["items"][1]["request"] = {
        "type": "cut", "start": 3.05, "end": 3.5, "words": [
            {"id": "t1:9", "segment": "s3", "text": "small"},
            {"id": "t1:10", "segment": "s3", "text": "pause"},
        ],
    }
    _write_batch(root, batch)

    class MixedJoins(FakeJoinTester):
        def test_cuts(self, project_dir, spans, **kwargs):
            results = super().test_cuts(project_dir, spans, **kwargs)
            for span, result in zip(spans, results, strict=True):
                result["pass"] = span["segment"] == "s3"
            return results

    joins, writer = MixedJoins(nudge=True), FakeRemovalWriter()
    document = _document({"id": "cut", "kind": "cut"},
                         {"id": "tail", "kind": "trim", "segment": "s3",
                          "edge": "end", "word": "t1:10"})
    document["items"].append({"comment": "v1:c2", "operations": [{"id": "cut2", "kind": "cut"}]})
    result = apply_operations(root, document, joins=joins, removals=writer)
    assert len(joins.calls) == 3
    assert joins.calls[0][1]["edge"] == "end" and joins.calls[2][0]["edge"] is None
    applied = next(change for change in result["changes"] if change["operation"] == "cut2")
    assert applied["join"][0]["edge"] is None and applied["join"][0]["end"] == 21.75
    assert writer.calls[0][3:5] == (17.75 + 1 / 30, 21.75)


def test_cut_second_round_failure_declines_remaining_note(tmp_path) -> None:
    root = _project(tmp_path / "project", request=_cut_note())
    batch = load_batch(root, "b1")
    batch["items"][1]["request"] = {
        "type": "cut", "start": 3.05, "end": 3.5, "words": [
            {"id": "t1:9", "segment": "s3", "text": "small"},
            {"id": "t1:10", "segment": "s3", "text": "pause"},
        ],
    }
    _write_batch(root, batch)

    class CascadingJoins(FakeJoinTester):
        round = 0

        def test_cuts(self, project_dir, spans, **kwargs):
            results = super().test_cuts(project_dir, spans, **kwargs)
            if "label" not in spans[0]:
                self.round += 1
            for span, result in zip(spans, results, strict=True):
                result.update({"pass": "label" not in span and self.round == 1
                               and span["operation"] == "cut2"},
                              diff=[f"failed in round {self.round}"])
            return results

    document = _document({"id": "cut", "kind": "cut"})
    document["items"].append({"comment": "v1:c2", "operations": [{"id": "cut2", "kind": "cut"}]})
    joins, writer = CascadingJoins(), FakeRemovalWriter()
    before = (root / "edit.json").read_bytes()
    result = apply_operations(root, document, joins=joins, removals=writer)
    assert joins.round == 2 and len(joins.calls) == 4
    assert [len(context["removals"]) for context in joins.contexts] == [2, 2, 1, 1]
    assert {change["operation"] for change in result["dispositions"]} == {"cut", "cut2"}
    second = next(change for change in result["dispositions"] if change["operation"] == "cut2")
    assert second["join"][0]["diff"] == ["failed in round 2"]
    assert not writer.calls and (root / "edit.json").read_bytes() == before
    assert all(item["disposition"] == "declined" for item in load_batch(root, "b1")["items"][:2])


@pytest.mark.parametrize("missing", ["module", "function"])
def test_join_tester_missing_module_or_function_has_named_error(monkeypatch, missing) -> None:
    def import_module(name):
        assert name == "yap_studio.stages.fillers"
        if missing == "module":
            raise ModuleNotFoundError(name)
        return SimpleNamespace()

    monkeypatch.setattr(apply_module.importlib, "import_module", import_module)
    with pytest.raises(FeedbackError) as error:
        apply_module.join_tester()
    assert error.value.code == "needs_fillers"


def _removal_document() -> dict:
    document = _document(_middle(), {"id": "tail", "kind": "trim", "segment": "s3",
                                    "edge": "end", "word": "t1:8"})
    document["items"].append({"comment": "v1:c2", "operations": [
        {"id": "back", "kind": "restore", "range": "s1.0"},
    ]})
    return document


@pytest.mark.parametrize("crash", [False, True])
def test_middle_remove_and_filler_restore_call_writer_once_and_resume(
    tmp_path: Path, monkeypatch, crash: bool,
) -> None:
    root = _project(tmp_path / "project", filler=True)
    document = _removal_document()
    plan = _planned(root, document)
    before = current_inputs(root)
    writer = FakeRemovalWriter(crash=crash)
    writes = []
    atomic_write = apply_module.paths.atomic_write_text

    def record_write(path, text, *args, **kwargs):
        if path == root / "edit.json":
            writes.append(text)
        return atomic_write(path, text, *args, **kwargs)

    monkeypatch.setattr(apply_module.paths, "atomic_write_text", record_write)
    if crash:
        with pytest.raises(RuntimeError, match="crash after first writer call"):
            apply_operations(root, document, removals=writer)
        journal = load_batch(root, "b1")["application"]
        assert journal["status"] == "pending"
        assert journal["inputs_before"] == before
        assert journal["removals"]["removals"] == [
            {"id": "r1", "take": "t1", "start": 1.75, "end": 5.75,
             "kind": "filler", "status": "applied"},
        ]
        assert journal["removals_sha256"] == before["removals.json"]
        assert [action["done"] for action in journal["removal_actions"]] == [False, False]
        assert [action["removal_id"] for action in journal["removal_actions"]] == [None, "r1"]
        assert current_inputs(root)["edit.json"] == before["edit.json"]
        assert len(writer.calls) == 1 and writes == []
        monkeypatch.setattr(apply_module, "plan_edit",
                            lambda *args, **kwargs: pytest.fail("resume must not replan"))
    result = apply_operations(root, document, removals=writer)
    assert writer.calls == [
        ("add_manual", root, "t1", 17.75, 21.75,
         'Operation middle (v1:c1): Remove "small pause" for v1:c1.'),
        ("set_removal", root, "r1", "rejected",
         "Operation back (v1:c2): Bring back the removed filler for v1:c2."),
    ]
    assert len(writes) == 1
    assert json.loads(writes[0]) == plan.edit
    assert result["changes"] == list(plan.changes)
    assert json.loads((root / "edit.json").read_text()) == plan.edit
    actions = result["removal_actions"]
    assert actions == [
        {"action": "add_manual", "operation": "middle", "comment": "v1:c1",
         "reason": 'Remove "small pause" for v1:c1.', "take": "t1", "start": 17.75,
         "end": 21.75, "removal_id": "manual1", "done": True},
        {"action": "reject", "operation": "back", "comment": "v1:c2",
         "reason": "Bring back the removed filler for v1:c2.", "take": None,
         "start": None, "end": None, "removal_id": "r1", "done": True},
    ]
    batch = load_batch(root, "b1")
    assert batch["application"]["status"] == "applied"
    assert batch["application"]["removals"] == json.loads((root / "removals.json").read_text())
    assert batch["application"]["removals_sha256"] == current_inputs(root)["removals.json"]
    assert batch["expected_inputs"] == current_inputs(root)
    for item in batch["items"][:2]:
        assert item["disposition"] == "applied"
        for operation in item["operations"]:
            assert operation["inputs_before"] == before
            assert operation["inputs_after"] == current_inputs(root)
    before_tree = _tree(root, lock=False)
    with pytest.raises(FeedbackError, match="already has an applied plan"):
        apply_operations(root, document, removals=writer)
    assert _tree(root, lock=False) == before_tree and len(writer.calls) == 2


def test_crash_after_edit_write_resumes_without_second_write(
    apply_project: Path, monkeypatch,
) -> None:
    document = _document(_trim())
    write_batch = apply_module._write_batch
    writes = []
    atomic_write = apply_module.paths.atomic_write_text

    def crash_on_completion(root, batch):
        if batch["application"]["status"] == "applied":
            raise RuntimeError("crash after edit write")
        return write_batch(root, batch)

    def record_write(path, text):
        writes.append(path)
        return atomic_write(path, text)

    monkeypatch.setattr(apply_module, "_write_batch", crash_on_completion)
    monkeypatch.setattr(apply_module.paths, "atomic_write_text", record_write)
    with pytest.raises(RuntimeError, match="crash after edit write"):
        apply_operations(apply_project, document)
    batch = load_batch(apply_project, "b1")
    assert batch["application"]["status"] == "pending"
    assert batch["items"][0]["operations"] == []
    assert sha256_file(apply_project / "edit.json") == batch["application"]["edit_sha256_after"]
    monkeypatch.setattr(apply_module, "_write_batch", write_batch)
    result = apply_operations(apply_project, document)
    assert result["status"] == "applied" and writes == [apply_project / "edit.json"]
    assert len(load_batch(apply_project, "b1")["items"][0]["operations"]) == 1


@pytest.mark.parametrize("missing", ["module", "add_manual", "set_removal"])
def test_fillers_writer_missing_module_or_function_has_named_error(
    monkeypatch, missing: str,
) -> None:
    def import_module(name):
        assert name == "yap_studio.stages.fillers"
        if missing == "module":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(**{key: lambda *args, **kwargs: {}
                                  for key in ("add_manual", "set_removal") if key != missing})

    monkeypatch.setattr(apply_module.importlib, "import_module", import_module)
    with pytest.raises(FeedbackError) as error:
        fillers_writer()
    assert error.value.code == "needs_fillers"
    assert str(error.value) == (
        "Removing words inside a segment and bringing back removed fillers need the "
        "fillers stage, which this build does not have yet."
    )
    assert error.value.suggested_action == (
        "Use trim, remove whole segments, or restore story cuts, or leave the note open."
    )


def test_fillers_adapter_passes_exact_arguments(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def add_manual(*args, **kwargs):
        calls.append((args, kwargs))
        return {"id": "r1"}

    def set_removal(*args, **kwargs):
        calls.append((args, kwargs))
        return {"id": "r1", "status": "rejected"}

    monkeypatch.setattr(apply_module.importlib, "import_module", lambda name: SimpleNamespace(
        add_manual=add_manual, set_removal=set_removal))
    writer = fillers_writer()
    assert writer.add_manual(tmp_path, "t1", 1.0, 2.0, reason="cut") == {"id": "r1"}
    assert writer.set_removal(tmp_path, "r1", "rejected", reason="back") == {
        "id": "r1", "status": "rejected",
    }
    assert calls == [((tmp_path, "t1", 1.0, 2.0), {"reason": "cut"}),
                     ((tmp_path, "r1", "rejected"), {"reason": "back"})]


@pytest.mark.parametrize("filler", [False, True])
def test_missing_fillers_refuses_before_journal_or_edit(
        monkeypatch, tmp_path: Path, filler: bool) -> None:
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(apply_module.importlib, "import_module", missing)
    root = _project(tmp_path / "project", filler=filler)
    document = (_document({"id": "back", "kind": "restore", "range": "s1.0"})
                if filler else _document(_middle()))
    before = _tree(root, lock=False)
    with pytest.raises(FeedbackError) as error:
        apply_operations(root, document)
    assert error.value.code == "needs_fillers"
    assert _tree(root, lock=False) == before


def test_timeline_kind_refuses_before_any_write(apply_project: Path) -> None:
    before = _tree(apply_project)
    with pytest.raises(FeedbackError) as error:
        apply_operations(apply_project, _document({"id": "caption", "kind": "caption"}))
    assert error.value.code == "needs_timeline"
    assert _tree(apply_project) == before


def test_pending_application_refuses_different_file(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", filler=True)
    writer = FakeRemovalWriter(crash=True)
    with pytest.raises(RuntimeError):
        apply_operations(root, _removal_document(), removals=writer)
    before = _tree(root, lock=False)
    with pytest.raises(FeedbackError) as error:
        apply_operations(root, _document(_trim()), removals=writer)
    assert str(error.value) == (
        "This batch has a pending application for a different operations file. "
        "There is no clear command; rerun the same file to finish it."
    )
    assert _tree(root, lock=False) == before and len(writer.calls) == 1


def test_pending_application_preserves_conflicting_edit(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", filler=True)
    writer = FakeRemovalWriter(crash=True)
    document = _removal_document()
    with pytest.raises(RuntimeError):
        apply_operations(root, document, removals=writer)
    path = root / "edit.json"
    changed = path.read_bytes() + b" "
    path.write_bytes(changed)
    with pytest.raises(StaleBatchError) as error:
        apply_operations(root, document, removals=writer)
    assert error.value.changed == ["edit.json"] and path.read_bytes() == changed
    assert load_batch(root, "b1")["application"]["status"] == "pending"


def _unrelated_removals_change(root: Path, mutation: str, *, identifier: str = "r1") -> None:
    path = root / "removals.json"
    data = json.loads(path.read_text())
    if mutation == "add":
        data["removals"].append({"id": "unrelated", "take": "t1", "start": 30.0, "end": 31.0,
                                 "kind": "silence", "status": "proposed"})
    elif mutation == "remove":
        data["removals"] = [entry for entry in data["removals"] if entry["id"] != identifier]
    else:
        next(entry for entry in data["removals"] if entry["id"] == identifier)["reason"] = (
            "Unrelated user decision"
        )
    _json(path, data)


@pytest.mark.parametrize("mutation", ["add", "remove", "change"])
def test_pending_application_refuses_unrelated_removals_change(
    tmp_path: Path, mutation: str,
) -> None:
    root = _project(tmp_path / "project", filler=True)
    document = _document(_middle())
    writer = FakeRemovalWriter(crash=True)
    with pytest.raises(RuntimeError, match="crash after first writer call"):
        apply_operations(root, document, removals=writer)
    _unrelated_removals_change(root, mutation)
    before = _tree(root, lock=False)
    error = None
    try:
        apply_operations(root, document, removals=writer)
    except StaleBatchError as caught:
        error = caught
    assert error is not None, "recovery must not absorb unrelated removals changes"
    assert error.changed == ["removals.json"]
    assert load_batch(root, "b1")["application"]["status"] == "pending"
    assert _tree(root, lock=False) == before
    assert len(writer.calls) == 1


@pytest.mark.parametrize("action", ["add_manual", "reject"])
@pytest.mark.parametrize("mutation", ["add", "remove", "change"])
def test_writer_cannot_absorb_unrelated_removals_changes(
    tmp_path: Path, action: str, mutation: str,
) -> None:
    root = _project(tmp_path / "project", filler=True)
    path = root / "removals.json"
    data = json.loads(path.read_text())
    data["removals"].append({"id": "other", "take": "t1", "start": 30.0, "end": 31.0,
                             "kind": "silence", "status": "proposed"})
    _json(path, data)
    batch = load_batch(root, "b1")
    batch["expected_inputs"] = current_inputs(root)
    _write_batch(root, batch)

    class ConflictingWriter(FakeRemovalWriter):
        def add_manual(self, *args, **kwargs):
            result = super().add_manual(*args, **kwargs)
            if action == "add_manual":
                with project_lock(root):
                    _unrelated_removals_change(root, mutation, identifier="other")
            return result

        def set_removal(self, *args, **kwargs):
            result = super().set_removal(*args, **kwargs)
            if action == "reject":
                with project_lock(root):
                    _unrelated_removals_change(root, mutation, identifier="other")
            return result

    document = (_document(_middle()) if action == "add_manual" else
                _document({"id": "back", "kind": "restore", "range": "s1.0"}))
    writer = ConflictingWriter()
    before = current_inputs(root)
    error = None
    try:
        apply_operations(root, document, removals=writer)
    except StaleBatchError as caught:
        error = caught
    assert error is not None, "a writer may change only its own removal entry"
    assert error.changed == ["removals.json"]
    journal = load_batch(root, "b1")["application"]
    assert journal["status"] == "pending" and not journal["removal_actions"][0]["done"]
    assert current_inputs(root)["edit.json"] == before["edit.json"]
    assert len(writer.calls) == 1


def test_completion_refuses_removals_change_after_last_verified_action(
    tmp_path: Path, monkeypatch,
) -> None:
    root = _project(tmp_path / "project", filler=True)
    write_batch = apply_module._write_batch

    def change_after_action(root, batch):
        write_batch(root, batch)
        journal = batch["application"]
        if journal["status"] == "pending" and all(a["done"] for a in journal["removal_actions"]):
            _unrelated_removals_change(root, "add")

    monkeypatch.setattr(apply_module, "_write_batch", change_after_action)
    error = None
    try:
        apply_operations(root, _document(_middle()), removals=FakeRemovalWriter())
    except StaleBatchError as caught:
        error = caught
    assert error is not None, "completion must compare removals with the last verified hash"
    assert error.changed == ["removals.json"]
    assert load_batch(root, "b1")["application"]["status"] == "pending"


def test_pending_story_application_refuses_new_removals_document(
    apply_project: Path, monkeypatch,
) -> None:
    write_batch = apply_module._write_batch

    def crash_on_completion(root, batch):
        if batch["application"]["status"] == "applied":
            raise RuntimeError("crash after edit write")
        return write_batch(root, batch)

    monkeypatch.setattr(apply_module, "_write_batch", crash_on_completion)
    document = _document(_trim())
    with pytest.raises(RuntimeError, match="crash after edit write"):
        apply_operations(apply_project, document)
    monkeypatch.setattr(apply_module, "_write_batch", write_batch)
    _json(apply_project / "removals.json", {"schema": "yap-studio/removals@1", "removals": []})
    error = None
    try:
        apply_operations(apply_project, document)
    except StaleBatchError as caught:
        error = caught
    assert error is not None, "plans without removal actions must still track removals.json"
    assert error.changed == ["removals.json"]
    assert load_batch(apply_project, "b1")["application"]["status"] == "pending"


def test_reject_retry_recognizes_only_its_own_change_without_second_writer_call(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "project", filler=True)

    class CrashingRejectWriter(FakeRemovalWriter):
        def set_removal(self, *args, **kwargs):
            result = super().set_removal(*args, **kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("crash after reject writer call")
            return result

    writer = CrashingRejectWriter()
    document = _document({"id": "back", "kind": "restore", "range": "s1.0"})
    with pytest.raises(RuntimeError, match="crash after reject writer call"):
        apply_operations(root, document, removals=writer)
    result = apply_operations(root, document, removals=writer)
    assert result["status"] == "applied"
    assert len(writer.calls) == 1, "a verified reject must be journaled without calling again"


@pytest.mark.parametrize("crash", [False, True])
def test_reject_refuses_target_span_change_on_apply_and_retry(
    tmp_path: Path, crash: bool,
) -> None:
    root = _project(tmp_path / "project", filler=True)

    class ConflictingRejectWriter(FakeRemovalWriter):
        def set_removal(self, *args, **kwargs):
            result = super().set_removal(*args, **kwargs)
            path = root / "removals.json"
            data = json.loads(path.read_text())
            data["removals"][0]["start"] = 2.0
            _json(path, data)
            if crash:
                raise RuntimeError("crash after conflicting reject")
            return result

    writer = ConflictingRejectWriter()
    document = _document({"id": "back", "kind": "restore", "range": "s1.0"})
    before = current_inputs(root)
    if crash:
        with pytest.raises(RuntimeError, match="crash after conflicting reject"):
            apply_operations(root, document, removals=writer)
    error = None
    try:
        apply_operations(root, document, removals=writer)
    except StaleBatchError as caught:
        error = caught
    assert error is not None, "reject must refuse changes to its target's source span"
    assert error.changed == ["removals.json"]
    journal = load_batch(root, "b1")["application"]
    assert journal["status"] == "pending" and not journal["removal_actions"][0]["done"]
    assert current_inputs(root)["edit.json"] == before["edit.json"]
    assert len(writer.calls) == 1


def _reject_state() -> dict:
    return {"schema": "yap-studio/removals@1", "removals": [{
        "id": "r1", "take": "t1", "start": 2.0, "end": 10.0,
        "kind": "filler", "status": "applied", "words": ["t1:1"],
        "by": "fillers", "listen": False, "reason": "detected filler",
        "evidence": {"score": 0.9, "decision": False},
    }]}


@pytest.mark.parametrize("change", [
    {"take": "t2"}, {"start": 8.0}, {"end": 11.0}, {"kind": "silence"},
    {"words": ["t1:2"]}, {"by": "creator"}, {"listen": True},
    {"evidence": {"score": 0.1, "decision": True}},
    {"evidence": {"decision": True}},
    {"evidence": {"score": 0.9, "decision": True, "extra": "changed"}},
    {"evidence": {"score": 0.9, "decision": "yes"}},
])
def test_reject_refuses_changes_beyond_status_reason_and_decision(change: dict) -> None:
    before = _reject_state()
    current = copy.deepcopy(before)
    current["removals"][0].update(status="rejected", reason="bring back filler",
                                  evidence={"score": 0.9, "decision": True})
    current["removals"][0].update(change)
    error = None
    try:
        apply_module._own_removal_change(before, current,
                                         {"action": "reject", "removal_id": "r1"})
    except StaleBatchError as caught:
        error = caught
    assert error is not None, "reject may change only status, reason, and evidence.decision"
    assert error.changed == ["removals.json"]


@pytest.mark.parametrize("status", ["proposed", "rejected"])
def test_reject_requires_prior_applied_status(status: str) -> None:
    before = _reject_state()
    before["removals"][0]["status"] = status
    current = copy.deepcopy(before)
    current["removals"][0].update(status="rejected", reason="bring back filler")
    error = None
    try:
        apply_module._own_removal_change(before, current,
                                         {"action": "reject", "removal_id": "r1"})
    except StaleBatchError as caught:
        error = caught
    assert error is not None, "reject must start from an applied removal"
    assert error.changed == ["removals.json"]


@pytest.mark.parametrize("evidence", [None, {}, {"score": 0.9},
                                      {"score": 0.9, "decision": False},
                                      {"score": 0.9, "decision": True}])
@pytest.mark.parametrize("decide", [False, True])
def test_reject_accepts_only_permitted_decision_changes(evidence: dict | None,
                                                       decide: bool) -> None:
    before = _reject_state()
    if evidence is None:
        before["removals"][0].pop("evidence")
    else:
        before["removals"][0]["evidence"] = evidence
    current = copy.deepcopy(before)
    current["removals"][0].update(status="rejected", reason="bring back filler")
    if decide:
        current["removals"][0].setdefault("evidence", {})["decision"] = True
    original = copy.deepcopy((before, current))
    assert apply_module._own_removal_change(before, current, {
        "action": "reject", "removal_id": "r1",
    }) == "r1"
    assert (before, current) == original


@pytest.mark.parametrize("action", ["add_manual", "reject"])
def test_written_action_can_resume_without_loading_a_writer(
    tmp_path: Path, monkeypatch, action: str,
) -> None:
    root = _project(tmp_path / "project", filler=True)

    class CrashingWriter(FakeRemovalWriter):
        def set_removal(self, *args, **kwargs):
            super().set_removal(*args, **kwargs)
            raise RuntimeError("crash after writer call")

    document = (_document(_middle()) if action == "add_manual" else
                _document({"id": "back", "kind": "restore", "range": "s1.0"}))
    writer = CrashingWriter(crash=True)
    with pytest.raises(RuntimeError):
        apply_operations(root, document, removals=writer)
    monkeypatch.setattr(apply_module, "fillers_writer",
                        lambda: pytest.fail("an already written action needs no writer"))
    assert apply_operations(root, document)["status"] == "applied"
    assert len(writer.calls) == 1


@pytest.mark.parametrize("change", [
    {"status": "proposed"}, {"kind": "silence"}, {"take": "t2"},
    {"start": 18.0}, {"end": 22.0}, {"reason": "Unrelated user decision"},
])
def test_pending_manual_action_requires_its_exact_own_change(tmp_path: Path, change: dict) -> None:
    root = _project(tmp_path / "project", filler=True)
    writer = FakeRemovalWriter(crash=True)
    document = _document(_middle())
    with pytest.raises(RuntimeError):
        apply_operations(root, document, removals=writer)
    path = root / "removals.json"
    data = json.loads(path.read_text())
    data["removals"][-1].update(change)
    _json(path, data)
    before = _tree(root, lock=False)
    with pytest.raises(StaleBatchError) as error:
        apply_operations(root, document, removals=writer)
    assert error.value.changed == ["removals.json"]
    assert _tree(root, lock=False) == before and len(writer.calls) == 1


def test_action_retries_writer_when_removals_still_match_recorded_state(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", filler=True)

    class CrashBeforeWriter(FakeRemovalWriter):
        def add_manual(self, *args, **kwargs):
            if self.crash:
                self.crash = False
                raise RuntimeError("crash before writer change")
            return super().add_manual(*args, **kwargs)

    writer = CrashBeforeWriter(crash=True)
    document = _document(_middle())
    before = current_inputs(root)
    with pytest.raises(RuntimeError, match="crash before writer change"):
        apply_operations(root, document, removals=writer)
    assert load_batch(root, "b1")["application"]["removals_sha256"] == before["removals.json"]
    assert apply_operations(root, document, removals=writer)["status"] == "applied"
    assert len(writer.calls) == 1


def test_source_words_are_hash_verified_before_manual_remove(apply_project: Path) -> None:
    path = apply_project / "ingest" / "t1" / "words.json"
    path.write_bytes(path.read_bytes() + b" ")
    before = _tree(apply_project, lock=False)
    writer = FakeRemovalWriter()
    with pytest.raises(FeedbackError, match="words file does not match the saved version"):
        apply_operations(apply_project, _document(_middle()), removals=writer)
    assert _tree(apply_project, lock=False) == before and writer.calls == []


def test_apply_does_not_mutate_operations_document(apply_project: Path) -> None:
    document = _document(_trim())
    before = copy.deepcopy(document)
    apply_operations(apply_project, document)
    assert document == before
