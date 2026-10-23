"""Caption word fixes and strict, independent taste handoff contract checks."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_feedback_batch import _cut, _export, _json
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_locate import _comment

from yap_studio.errors import InputError
from yap_studio.feedback import spelling as spelling_module
from yap_studio.feedback.batch import (
    StaleBatchError,
    build_batch,
    check_inputs,
    load_batch,
    set_disposition,
)
from yap_studio.feedback.spelling import apply_word_fixes
from yap_studio.feedback.taste import ProposalInbox, spelling_proposal
from yap_studio.feedback.versions import save_version
from yap_studio.project.lock import ConflictError
from yap_studio.project.model import parse_document
from yap_studio.receipt import sha256_file
from yap_studio.review.export import comment_to_dict

REQUEST = {"type": "spelling", "from": "tail net", "to": "tailnet"}


def test_regression_carried_spelling_does_not_overwrite_new_revision(
    feedback_project: Path,
) -> None:
    root = feedback_project
    first = save_version(root)
    old = spelling_note()
    build_batch(root, _export(root, [old]))
    apply_word_fixes(root, "b1")
    _cut(root, label="next spelling render")
    second = save_version(root)
    new = replace(spelling_note("c2", request={**REQUEST, "to": "tail-net"}),
                  updated_at="2026-10-05T23:00:00Z")
    export = _export(root, [new])
    export["series"] = {
        "id": "series", "version": 2, "versions": [
            {"number": 1, "sha256": first.media["sha256"], "version_id": "v1"},
            {"number": 2, "sha256": second.media["sha256"], "version_id": "v2"},
        ], "threads": [{**comment_to_dict(old), "version": 1, "reply": None, "state": "open"}],
    }
    build_batch(root, export)
    apply_word_fixes(root, "b2")
    assert json.loads((root / "project.json").read_text())["word_fixes"]["tail net"] == "tail-net"
    carried = next(item for item in load_batch(root, "b2")["items"] if item["carried"])
    assert carried["disposition"] == "applied" and carried["operations"] == []
    assert carried["evidence"] == {"already_applied_in": "b1"}


def test_regression_fresh_spelling_applies_in_revision_order(feedback_project: Path) -> None:
    root = feedback_project
    newer = replace(spelling_note("c2", request={**REQUEST, "to": "tail-net"}),
                    updated_at="2026-10-05T23:00:00Z")
    batch = build_batch(root, _export(root, [newer, spelling_note()]))
    batch["items"].sort(key=lambda item: item["comment"]["revision"], reverse=True)
    _json(root / "feedback" / "b1.json", batch)
    apply_word_fixes(root, "b1")
    assert json.loads((root / "project.json").read_text())["word_fixes"]["tail net"] == "tail-net"


def test_regression_older_fresh_spelling_is_declined(feedback_project: Path) -> None:
    root = feedback_project
    newer = replace(spelling_note("c2", request={**REQUEST, "to": "tail-net"}),
                    updated_at="2026-10-05T23:00:00Z")
    build_batch(root, _export(root, [newer]))
    apply_word_fixes(root, "b1")
    build_batch(root, _export(root, [spelling_note()]))
    assert apply_word_fixes(root, "b2") == []
    item = load_batch(root, "b2")["items"][0]
    assert item["disposition"] == "declined" and item["operations"] == []
    assert item["evidence"] == {"superseded_by": {
        "version": None, "id": "c2", "revision": newer.updated_at,
    }}


def assert_proposals_shape(data: object) -> None:
    """Check exact importer keys and types without relying on production validation."""
    assert isinstance(data, dict) and set(data) == {"schema", "proposals"}
    assert data["schema"] == "yap-studio/taste-proposals@1"
    assert isinstance(data["proposals"], list)
    for body in data["proposals"]:
        assert isinstance(body, dict)
        assert set(body) == {"scope", "condition", "sentence", "machine", "sources", "context"}
        assert body["scope"] == "captions"
        assert body["condition"] is None and body["machine"] is None
        assert isinstance(body["sentence"], str) and 1 <= len(body["sentence"]) <= 280
        assert isinstance(body["context"], str) and 1 <= len(body["context"]) <= 500
        assert isinstance(body["sources"], list) and 1 <= len(body["sources"]) <= 50
        identities = set()
        for source in body["sources"]:
            assert isinstance(source, dict)
            assert set(source) == {"type", "media_sha256", "comment_id", "label",
                                   "start_frame", "end_frame", "quote"}
            assert source["type"] == "comment" and source["label"] == "fix"
            assert isinstance(source["media_sha256"], str)
            assert re.fullmatch(r"[0-9a-f]{64}", source["media_sha256"])
            assert isinstance(source["comment_id"], str)
            assert re.fullmatch(r"c[1-9][0-9]*", source["comment_id"])
            for field in ("start_frame", "end_frame"):
                assert source[field] is None or (type(source[field]) is int and source[field] >= 0)
            assert isinstance(source["quote"], str) and len(source["quote"]) <= 500
            identity = (source["media_sha256"], source["comment_id"])
            assert identity not in identities
            identities.add(identity)


def spelling_note(id="c1", *, request=None):
    return replace(_comment(id, kind="fix", target="caption", start_frame=18, start_s=0.6),
                   text="Correct this caption spelling.", request=request or REQUEST)


def test_apply_hashes_operation_exact_handoff_merge_and_idempotence(feedback_project: Path) -> None:
    root = feedback_project
    version = save_version(root)
    note = spelling_note()
    batch = build_batch(root, _export(root, [note]))
    assert batch["items"][0]["request"] == batch["items"][0]["comment"]["request"] == REQUEST
    before = sha256_file(root / "project.json")
    inbox = ProposalInbox(root)
    operations = apply_word_fixes(root, batch["id"], sink=inbox)
    operation = {"id": f"word-fix:{version.id}:c1", "op": "word_fix",
                 "from": "tail net", "to": "tailnet"}
    assert operations == [operation]
    project = json.loads((root / "project.json").read_text())
    assert project["word_fixes"] == {"tail net": "tailnet"}
    parse_document("project", project)
    applied = load_batch(root, batch["id"])
    item = applied["items"][0]
    assert item["disposition"] == "applied"
    assert item["evidence"] == {"project_json": {"before": before,
                                              "after": sha256_file(root / "project.json")}}
    assert before != item["evidence"]["project_json"]["after"]
    assert item["operations"] == [{**operation, "inputs_before": batch["expected_inputs"],
                                    "inputs_after": batch["expected_inputs"]}]
    assert applied["expected_inputs"] == batch["expected_inputs"]
    check_inputs(root, applied)
    proposal = json.loads(inbox.path.read_text())
    assert_proposals_shape(proposal)
    assert proposal["proposals"] == [{
        "scope": "captions", "condition": None,
        "sentence": 'Spell "tail net" as "tailnet" in captions.', "machine": None,
        "sources": [{"type": "comment", "media_sha256": version.media["sha256"],
                     "comment_id": "c1", "label": "fix", "start_frame": note.start_frame,
                     "end_frame": note.end_frame, "quote": note.text}],
        "context": 'Caption spelling fix from review. Word fix: "tail net" -> "tailnet".',
    }]
    watched = [root / "project.json", root / "feedback" / "b1.json", inbox.path]
    unchanged = [(file.read_bytes(), file.stat().st_mtime_ns) for file in watched]
    assert apply_word_fixes(root, batch["id"], sink=inbox) == []
    assert [(file.read_bytes(), file.stat().st_mtime_ns) for file in watched] == unchanged
    another = build_batch(root, _export(root, [spelling_note("c2")]))
    assert len(apply_word_fixes(root, another["id"], sink=inbox)) == 1
    merged = json.loads(inbox.path.read_text())
    assert_proposals_shape(merged)
    assert len(merged["proposals"]) == 1
    assert [source["comment_id"] for source in merged["proposals"][0]["sources"]] == ["c1", "c2"]
    assert watched[0].read_bytes() == unchanged[0][0]
    evidence = load_batch(root, another["id"])["items"][0]["evidence"]["project_json"]
    assert evidence["before"] == evidence["after"]


def test_conflicting_mapping_replaced_and_multiple_operations_unique(
    feedback_project: Path,
) -> None:
    root = feedback_project
    project = json.loads((root / "project.json").read_text())
    project["word_fixes"] = {"tail net": "previous spelling", "untouched": "kept"}
    _json(root / "project.json", project)
    notes = [spelling_note(), spelling_note("c2", request={**REQUEST, "to": "tail-net"})]
    batch = build_batch(root, _export(root, notes))
    operations = apply_word_fixes(root, batch["id"])
    assert [operation["replaced"] for operation in operations] == ["previous spelling", "tailnet"]
    assert len({operation["id"] for operation in operations}) == 2
    assert json.loads((root / "project.json").read_text())["word_fixes"] == {
        "tail net": "tail-net", "untouched": "kept",
    }
    assert apply_word_fixes(root, batch["id"]) == []


def test_only_open_spelling_items_apply(feedback_project: Path) -> None:
    root = feedback_project
    notes = [spelling_note(), spelling_note("c2"), spelling_note("c3"), _comment("c4")]
    batch = build_batch(root, _export(root, notes))
    set_disposition(root, batch["id"], None, "c1", "declined")
    set_disposition(root, batch["id"], None, "c2", "needs_clarification")
    operations = apply_word_fixes(root, batch["id"])
    assert [operation["id"] for operation in operations] == ["word-fix:None:c3"]
    assert not ProposalInbox(root).path.exists()


def test_mixed_restore_and_spelling_batch_applies_only_spelling(feedback_project: Path) -> None:
    root = feedback_project
    version = save_version(root)
    request = {"type": "restore", "range": "s1.0", "take": "t1", "start": 1.0,
               "end": 2.0, "text": "um well"}
    restore = replace(_comment("c1", target="story", start_frame=15, start_s=0.5),
                      text='Bring back: "um well"', request=request)
    batch = build_batch(root, _export(root, [restore, spelling_note("c2")]))
    original = copy.deepcopy(next(item for item in batch["items"]
                                  if item["comment"]["id"] == "c1"))
    inbox = ProposalInbox(root)
    operation = {"id": f"word-fix:{version.id}:c2", "op": "word_fix",
                 "from": "tail net", "to": "tailnet"}
    assert apply_word_fixes(root, batch["id"], sink=inbox) == [operation]
    assert json.loads((root / "project.json").read_text())["word_fixes"] == {
        "tail net": "tailnet",
    }
    items = {item["comment"]["id"]: item for item in load_batch(root, batch["id"])["items"]}
    assert items["c1"] == original
    assert items["c1"]["disposition"] == "open" and items["c1"]["operations"] == []
    assert items["c1"]["request"] == items["c1"]["comment"]["request"] == request
    assert items["c2"]["disposition"] == "applied"
    assert [item["id"] for item in items["c2"]["operations"]] == [operation["id"]]
    proposals = json.loads(inbox.path.read_text())["proposals"]
    assert len(proposals) == 1
    assert [source["comment_id"] for source in proposals[0]["sources"]] == ["c2"]
    assert apply_word_fixes(root, batch["id"], sink=inbox) == []
    assert next(item for item in load_batch(root, batch["id"])["items"]
                if item["comment"]["id"] == "c1") == original


def test_sink_failure_retries_handoff_without_reapplying_project(feedback_project: Path) -> None:
    root = feedback_project
    batch = build_batch(root, _export(root, [spelling_note()]))

    class UnavailableSink:
        def propose(self, bodies):
            assert len(bodies) == 1
            raise RuntimeError("Proposal handoff unavailable.")

    with pytest.raises(RuntimeError, match="handoff unavailable"):
        apply_word_fixes(root, batch["id"], sink=UnavailableSink())
    item = load_batch(root, batch["id"])["items"][0]
    assert item["disposition"] == "applied" and len(item["operations"]) == 1
    assert not item.get("taste_proposed", False)
    before = (root / "project.json").read_bytes()
    inbox = ProposalInbox(root)
    assert apply_word_fixes(root, batch["id"], sink=inbox) == []
    assert (root / "project.json").read_bytes() == before
    assert load_batch(root, batch["id"])["items"][0]["operations"] == item["operations"]
    assert_proposals_shape(json.loads(inbox.path.read_text()))


def test_stale_batch_refused_before_project_write(feedback_project: Path) -> None:
    root = feedback_project
    batch = build_batch(root, _export(root, [spelling_note()]))
    _json(root / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {},
                               "segments": [], "end_hold_s": 2})
    before = (root / "project.json").read_bytes()
    with pytest.raises(StaleBatchError):
        apply_word_fixes(root, batch["id"], sink=ProposalInbox(root))
    assert (root / "project.json").read_bytes() == before
    assert load_batch(root, batch["id"])["items"][0]["operations"] == []


def test_cas_failure_leaves_item_open(feedback_project: Path, monkeypatch) -> None:
    root = feedback_project
    batch = build_batch(root, _export(root, [spelling_note()]))
    before = (root / "project.json").read_bytes()

    def conflict(path, data, *, base_sha256):
        assert base_sha256 == sha256_file(path)
        assert data["word_fixes"] == {"tail net": "tailnet"}
        raise ConflictError(path)

    monkeypatch.setattr(spelling_module, "write_json_cas", conflict)
    with pytest.raises(ConflictError):
        apply_word_fixes(root, batch["id"])
    assert (root / "project.json").read_bytes() == before
    assert load_batch(root, batch["id"])["items"][0]["disposition"] == "open"


def test_proposal_limits_and_frames_for_variable_and_seconds_only_comments() -> None:
    note = replace(spelling_note(), text="q" * 1000)
    body = spelling_proposal(wrong="w" * 1000, right="r" * 1000,
                             media_sha256="a" * 64, comment=note, fps=Fraction(30))
    assert_proposals_shape({"schema": "yap-studio/taste-proposals@1", "proposals": [body]})
    assert len(body["sentence"]) == 280 and len(body["context"]) == 500
    assert len(body["sources"][0]["quote"]) == 500
    seconds = replace(note, start_frame=None, end_frame=None, start_s=0.6, end_s=1.2)
    variable = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                                 comment=seconds, fps=None)
    assert variable["sources"][0]["start_frame"] is None
    assert variable["sources"][0]["end_frame"] is None
    fixed = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                              comment=seconds, fps=Fraction(30))
    assert fixed["sources"][0]["start_frame"] == 18
    assert fixed["sources"][0]["end_frame"] == 36


@pytest.mark.parametrize("bad", [
    "{broken", [], {}, {"schema": "foreign", "proposals": []},
    {"schema": "yap-studio/taste-proposals@1", "proposals": [], "extra": 1},
    {"schema": "yap-studio/taste-proposals@1", "proposals": {}},
])
def test_inbox_refuses_foreign_or_corrupt_file_without_overwrite(tmp_path: Path, bad) -> None:
    inbox = ProposalInbox(tmp_path)
    inbox.path.parent.mkdir()
    inbox.path.write_text(bad if isinstance(bad, str) else json.dumps(bad))
    before = inbox.path.read_bytes()
    body = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                             comment=spelling_note(), fps=Fraction(30))
    with pytest.raises(InputError):
        inbox.propose([body])
    assert inbox.path.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"machine": {}}, {"condition": "something"}, {"scope": "unknown"}, {"sentence": 1},
    {"sentence": "s" * 281}, {"context": "c" * 501}, {"sources": []}, {"extra": None},
])
def test_inbox_refuses_invalid_bodies(tmp_path: Path, change: dict) -> None:
    inbox = ProposalInbox(tmp_path)
    body = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                             comment=spelling_note(), fps=Fraction(30))
    invalid = {**body, **change}
    with pytest.raises(InputError):
        inbox.propose([invalid])
    assert not inbox.path.exists()
    _json(inbox.path, {"schema": "yap-studio/taste-proposals@1", "proposals": [invalid]})
    before = inbox.path.read_bytes()
    with pytest.raises(InputError):
        inbox.propose([body])
    assert inbox.path.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"media_sha256": "bad"}, {"comment_id": 1}, {"label": None}, {"start_frame": True},
    {"end_frame": 1.5}, {"quote": "q" * 501}, {"extra": 1},
])
def test_inbox_refuses_invalid_source(tmp_path: Path, change: dict) -> None:
    inbox = ProposalInbox(tmp_path)
    body = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                             comment=spelling_note(), fps=None)
    body["sources"][0].update(change)
    with pytest.raises(InputError):
        inbox.propose([body])
    assert not inbox.path.exists()


def test_inbox_merges_identity_caps_sources_and_appends_distinct_rule(tmp_path: Path) -> None:
    inbox = ProposalInbox(tmp_path)
    bodies = [spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                               comment=spelling_note(f"c{number}"), fps=None)
              for number in range(1, 56)]
    inbox.propose(bodies)
    first = json.loads(inbox.path.read_text())
    assert_proposals_shape(first)
    assert len(first["proposals"]) == 1 and len(first["proposals"][0]["sources"]) == 50
    original = inbox.path.read_bytes()
    inbox.propose(bodies)
    assert inbox.path.read_bytes() == original
    another = copy.deepcopy(bodies[0])
    another["sentence"] = 'Spell "wrong" as "another" in captions.'
    inbox.propose([another])
    assert len(json.loads(inbox.path.read_text())["proposals"]) == 2


def test_source_identity_includes_media_hash_and_keeps_first_quote(tmp_path: Path) -> None:
    inbox = ProposalInbox(tmp_path)
    body = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                             comment=spelling_note(), fps=None)
    same = copy.deepcopy(body)
    same["sources"][0]["quote"] = "Edited quote for the same comment"
    another = copy.deepcopy(body)
    another["sources"][0]["media_sha256"] = "b" * 64
    assert inbox.propose([body, same, another]) == ["added", "unchanged", "merged"]
    data = json.loads(inbox.path.read_text())
    assert_proposals_shape(data)
    assert len(data["proposals"]) == 1
    assert data["proposals"][0]["sources"] == [body["sources"][0], another["sources"][0]]


def test_inbox_accepts_style_keep_and_placement_proposals(tmp_path: Path) -> None:
    inbox = ProposalInbox(tmp_path)
    assert inbox.read() == [] and not inbox.path.exists()
    body = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                             comment=spelling_note(), fps=None)
    body.update(sentence="Keep captions short.",
                condition={"text": "On screen recordings.", "tags": ["screen"]},
                machine={"kind": "style_field", "field": "captions.max_words_per_phrase",
                         "value": 3})
    body["sources"][0]["label"] = "keep"
    another = copy.deepcopy(body)
    another["sources"][0]["comment_id"] = "c2"
    placement = copy.deepcopy(body)
    placement.update(scope="cards", sentence="Keep titles clear of faces.", condition=None,
                     machine={"kind": "placement", "layer": "title", "rule": "clear_of_face"})
    inbox.propose([body, another, placement])
    result = inbox.read()
    assert len(result) == 2
    assert result[0] == {**body, "sources": [*body["sources"], *another["sources"]]}
    assert result[1] == placement
    before = inbox.path.read_bytes()
    inbox.propose([body, another, placement])
    assert inbox.path.read_bytes() == before


@pytest.mark.parametrize("bad", ["measurement", "null_label", "uppercase", "duplicate",
                                "null_context", "empty_context"])
def test_inbox_strict_comment_evidence_on_write_and_read(tmp_path: Path, bad: str) -> None:
    inbox = ProposalInbox(tmp_path)
    body = spelling_proposal(wrong="wrong", right="right", media_sha256="a" * 64,
                             comment=spelling_note(), fps=None)
    if bad == "measurement":
        body["sources"] = [{"type": "measurement", "media_sha256": "a" * 64,
                            "media_name": "render.mp4", "measure": "face_overlap", "frames": [1]}]
    elif bad == "null_label":
        body["sources"][0]["label"] = None
    elif bad == "uppercase":
        body["sources"][0]["media_sha256"] = "A" * 64
    elif bad == "duplicate":
        body["sources"].append(copy.deepcopy(body["sources"][0]))
    else:
        body["context"] = None if bad == "null_context" else ""
    with pytest.raises(InputError):
        inbox.propose([body])
    assert not inbox.path.exists()
    _json(inbox.path, {"schema": "yap-studio/taste-proposals@1", "proposals": [body]})
    before = inbox.path.read_bytes()
    with pytest.raises(InputError):
        inbox.read()
    assert inbox.path.read_bytes() == before
