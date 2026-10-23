"""Project taste proposals, private decisions, and interrupted intent recovery."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_review_rounds import round_project as round_project

from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback.taste import ProposalInbox
from yap_studio.feedback.versions import load_version
from yap_studio.review.model import MediaFacts
from yap_studio.review.store import ReviewStore
from yap_studio.taste import (
    TasteError,
    TasteRefusal,
    UnknownProposalError,
    add_proposals,
    approve,
    decide,
    inbox_view,
    load_proposals,
    load_taste,
    proposals_path,
    propose,
    rule_id,
    store,
    taste_path,
)
from yap_studio.taste import inbox as inbox_module

NOW = "2026-10-06T12:00:00Z"


@pytest.fixture(autouse=True)
def isolated_taste(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(store, "_now", lambda: NOW)


@pytest.fixture
def notes(round_project: Path, tmp_path: Path) -> tuple[ReviewStore, list]:
    review = ReviewStore(tmp_path / "data")
    first, second = (load_version(round_project, version) for version in ("v1", "v2"))
    sha1, sha2 = first.media["sha256"], second.media["sha256"]
    frame = review.add_comment(sha1, MediaFacts("first.mp4", sha1, 4, Fraction(60), True),
                               {"kind": "keep", "start_frame": 12, "end_frame": 42,
                                "text": "Keep this rhythm " + "q" * 600})
    seconds = review.add_comment(sha2, MediaFacts("second.mp4", sha2, 4, None, False),
                                 {"kind": "fix", "start_s": 0.6, "end_s": 1.2,
                                  "text": "Fewer words please."})
    return review, [frame, seconds]


def _proposal(words: int = 3, *, comments: list | None = None) -> dict:
    return {"scope": "captions", "sentence": f"Use at most {words} caption words.",
            "machine": {"kind": "style_field", "field": "captions.max_words_per_phrase",
                        "value": words},
            "comments": comments if comments is not None else [{"version": "v1", "id": "c1"}]}


def _seed(root: Path, notes: tuple, words: int = 3) -> str:
    return propose(root, _proposal(words), store=notes[0])[0]["id"]


def _ledger(root: Path) -> list[dict]:
    path = root / "feedback" / "taste-decisions.json"
    return json.loads(path.read_text())["decisions"] if path.exists() else []


def test_propose_saved_comments_timing_labels_quotes_and_outcomes(
    round_project: Path, notes: tuple, tmp_path: Path,
) -> None:
    root, (review, comments) = round_project, notes
    first = _proposal()
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"proposals": [first]}))
    identity = rule_id(first)
    assert propose(root, source, store=review) == [{"id": identity, "status": "added"}]
    body = ProposalInbox(root).read()[0]
    assert body["condition"] is None and body["context"] == "Proposed from review notes."
    assert body["sources"] == [{"type": "comment", "media_sha256": load_version(root, "v1").media[
        "sha256"], "comment_id": comments[0].id, "label": "keep", "start_frame": 12,
        "end_frame": 42, "quote": comments[0].text[:500]}]
    both = _proposal(comments=[{"version": "v1", "id": "c1"}, {"version": "v2", "id": "c1"}])
    assert propose(root, both, store=review) == [{"id": identity, "status": "merged"}]
    sources = ProposalInbox(root).read()[0]["sources"]
    assert sources[1] == {"type": "comment", "media_sha256": load_version(root, "v2").media[
        "sha256"], "comment_id": comments[1].id, "label": "fix", "start_frame": 18,
        "end_frame": 36, "quote": comments[1].text}
    before = ProposalInbox(root).path.read_bytes()
    assert propose(root, both, store=review) == [{"id": identity, "status": "unchanged"}]
    assert ProposalInbox(root).path.read_bytes() == before
    assert not taste_path().exists() and not proposals_path().exists()


def test_propose_unknown_fps_leaves_seconds_frames_null(
    round_project: Path, notes: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = load_version(round_project, "v2")
    monkeypatch.setattr(inbox_module, "load_version", lambda *args: replace(
        version, media={**version.media, "fps": None},
    ))
    propose(round_project, _proposal(comments=[{"version": "v2", "id": "c1"}]), store=notes[0])
    source = ProposalInbox(round_project).read()[0]["sources"][0]
    assert source["start_frame"] is None and source["end_frame"] is None


@pytest.mark.parametrize(("different_notes", "status"), [(True, "merged"), (False, "unchanged")])
def test_propose_duplicate_file_entries_report_each_merge_outcome(
    round_project: Path, notes: tuple, tmp_path: Path, different_notes: bool, status: str,
) -> None:
    first = _proposal()
    second = _proposal(comments=[{"version": "v2", "id": "c1"}]) if different_notes else first
    source = tmp_path / "duplicates.json"
    source.write_text(json.dumps({"proposals": [first, second]}))
    identity = rule_id(first)
    assert propose(round_project, source, store=notes[0]) == [
        {"id": identity, "status": "added"}, {"id": identity, "status": status},
    ]
    bodies = ProposalInbox(round_project).read()
    assert len(bodies) == 1 and len(bodies[0]["sources"]) == (2 if different_notes else 1)


@pytest.mark.parametrize(("change", "message"), [
    ({"comments": [{"version": "v99", "id": "c1"}]}, "no version v99"),
    ({"comments": [{"version": "v2", "id": "c7"}]}, "no comment c7 on version v2"),
    ({"extra": 1}, "unknown field"),
    ({"comments": [{"version": "v1", "id": "c1"}] * 2}, "duplicate comment"),
    ({"comments": []}, "1 to 50"),
    ({"comments": [{"version": "v1", "id": "c1"}] * 51}, "1 to 50"),
    ({"comments": [{"version": "v0", "id": "c1"}]}, "must match"),
    ({"comments": [{"version": "v1", "id": "c0"}]}, "must match"),
    ({"comments": [{"version": "v1", "id": "c1", "extra": 1}]}, "exactly"),
    ({"context": None}, "context"),
    ({"sentence": "Bad\nwording"}, "newlines"),
])
def test_propose_validates_entire_batch_before_writing(
    round_project: Path, notes: tuple, change: dict, message: str,
) -> None:
    source = {"proposals": [_proposal(), {**_proposal(4), **change}]}
    with pytest.raises(ValidationError, match=message) as caught:
        propose(round_project, source, store=notes[0])
    assert "proposals[1]" in caught.value.parameter
    assert caught.value.suggested_action
    assert not ProposalInbox(round_project).path.exists()
    assert not taste_path().exists() and not proposals_path().exists()


@pytest.mark.parametrize("source", [[], {"proposals": {}}, {"proposals": [], "extra": 1},
                                   {"proposals": [None]}, {"scope": "captions"}])
def test_propose_refuses_invalid_input_objects(round_project: Path, notes: tuple, source) -> None:
    with pytest.raises(ValidationError):
        propose(round_project, source, store=notes[0])
    assert not ProposalInbox(round_project).path.exists()


def test_reading_absent_inbox_is_read_only(tmp_path: Path) -> None:
    root = tmp_path / "missing-project"
    assert inbox_view(root) == {"revision": 0, "proposals": []}
    assert not root.exists() and not taste_path().parent.exists()


def test_approve_revoke_states_history_snapshots_and_idempotency(
    round_project: Path, notes: tuple,
) -> None:
    root = round_project
    identity = _seed(root, notes)
    pending = inbox_view(root)
    assert pending["revision"] == 0
    assert pending["proposals"][0]["state"] == "pending"
    assert pending["proposals"][0]["rule"] is None
    assert pending["proposals"][0]["decided_at"] is None
    empty = load_taste().snapshot()
    approved = decide(root, identity, "approve")
    assert approved["state"] == "approved" and approved["edited"] is False
    assert approved["decided_at"] == NOW
    assert approved["rule"] == {"id": identity, "sentence": _proposal()["sentence"],
                                "status": "approved", "approved_revision": 1,
                                "ended_revision": None}
    active = load_taste().snapshot()
    assert active != empty and active.sha256 != empty.sha256
    ledger_bytes = (root / "feedback" / "taste-decisions.json").read_bytes()
    private_bytes = taste_path().read_bytes()
    wording = "  " + approved["sentence"] + "  "
    assert decide(root, identity, "approve", sentence=wording) == approved
    assert (root / "feedback" / "taste-decisions.json").read_bytes() == ledger_bytes
    assert taste_path().read_bytes() == private_bytes and load_taste().revision == 1
    revoked = decide(root, identity, "revoke")
    assert revoked["state"] == "revoked" and revoked["rule"]["ended_revision"] == 2
    assert load_taste().snapshot().sha256 == empty.sha256
    assert [entry["action"] for entry in load_taste().history] == ["approved", "revoked"]
    assert [entry["action"] for entry in _ledger(root)] == ["approved", "revoked"]
    assert _ledger(root)[:1] == json.loads(ledger_bytes)["decisions"]


def test_reject_is_snapshot_neutral_idempotent_and_can_reopen(
    round_project: Path, notes: tuple,
) -> None:
    root, identity = round_project, _seed(round_project, notes)
    before = load_taste().snapshot()
    rejected = decide(root, identity, "reject")
    assert rejected["state"] == "rejected" and rejected["decided_at"] == NOW
    assert rejected["rule"] is None and rejected["edited"] is False
    assert load_taste().snapshot() == before and load_taste().history == ()
    ledger = (root / "feedback" / "taste-decisions.json").read_bytes()
    private = proposals_path().read_bytes()
    assert decide(root, identity, "reject") == rejected
    assert (root / "feedback" / "taste-decisions.json").read_bytes() == ledger
    assert proposals_path().read_bytes() == private
    approved = decide(root, identity, "approve")
    assert approved["state"] == "approved" and load_taste().revision == 1
    assert load_proposals().proposals == ()
    assert [entry["action"] for entry in _ledger(root)] == ["rejected", "approved"]


@pytest.mark.parametrize("reject_first", [False, True])
def test_edit_approves_new_wording_rejects_original_and_revokes_target(
    round_project: Path, notes: tuple, reject_first: bool,
) -> None:
    root, identity = round_project, _seed(round_project, notes)
    if reject_first:
        decide(root, identity, "reject")
    before = load_taste().snapshot()
    edited = decide(root, identity, "approve", sentence="  Keep phrases to three words.  ")
    target = rule_id({**ProposalInbox(root).read()[0], "sentence": "Keep phrases to three words."})
    assert target != identity
    assert edited["state"] == "approved" and edited["edited"] is True
    assert edited["sentence"] == _proposal()["sentence"]
    assert edited["rule"]["sentence"] == "Keep phrases to three words."
    assert edited["rule"]["id"] == target and load_taste().revision == 1
    assert load_proposals().proposals[0]["id"] == identity
    assert load_proposals().proposals[0]["status"] == "rejected"
    assert load_taste().snapshot() != before
    assert _ledger(root)[-1] == {
        "proposal": identity, "action": "edited", "rule": target, "at": NOW,
    }
    assert decide(root, identity, "approve") == edited
    assert _ledger(root)[-1]["action"] == "edited"
    revoked = decide(root, identity, "revoke")
    assert revoked["state"] == "revoked" and revoked["edited"] is True
    assert _ledger(root)[-1]["rule"] == target
    assert [entry["rule_id"] for entry in load_taste().history] == [target, target]


def test_later_same_target_approval_supersedes_inbox_rule(
    round_project: Path, notes: tuple,
) -> None:
    first = _seed(round_project, notes)
    second = _seed(round_project, notes, 4)
    decide(round_project, first, "approve")
    decide(round_project, second, "approve")
    view = inbox_view(round_project)
    assert view["revision"] == 2
    assert [item["id"] for item in view["proposals"]] == [first, second]
    assert [item["state"] for item in view["proposals"]] == ["superseded", "approved"]
    assert view["proposals"][0]["rule"]["ended_revision"] == 2
    assert [entry["action"] for entry in load_taste().history] == [
        "approved", "approved", "superseded",
    ]


def test_edit_to_existing_approved_rule_reuses_it_without_new_revision(
    round_project: Path, notes: tuple,
) -> None:
    root, identity = round_project, _seed(round_project, notes)
    original = ProposalInbox(root).read()[0]
    wording = "Keep phrases to three words."
    approved_body = {**original, "sentence": wording,
                     "sources": [{**original["sources"][0], "media_sha256": "a" * 64,
                                  "comment_id": "c7", "quote": "Evidence from another project."}]}
    target = store.approve_proposal(approved_body).rule["id"]
    before = load_taste(), taste_path().read_bytes()
    prior_ledger = _ledger(root)
    item = decide(root, identity, "approve", sentence=wording)
    assert item["state"] == "approved" and item["edited"] is True
    assert item["sentence"] == original["sentence"]
    assert item["rule"]["id"] == target and item["rule"]["sentence"] == wording
    assert inbox_view(root)["proposals"][0] == item
    assert (load_taste(), taste_path().read_bytes()) == before
    assert load_taste().revision == 1
    rejected = load_proposals().proposals[0]
    assert rejected["id"] == identity and rejected["status"] == "rejected"
    assert _ledger(root) == [*prior_ledger, {
        "proposal": identity, "action": "edited", "rule": target, "at": NOW,
    }]


def test_cli_approval_without_ledger_is_visible(round_project: Path, notes: tuple, capsys) -> None:
    from yap_studio.cli import main

    identity = _seed(round_project, notes)
    add_proposals(ProposalInbox(round_project).read()[0])
    assert main(["taste", "approve", identity]) == 0
    capsys.readouterr()
    assert inbox_view(round_project)["proposals"][0]["state"] == "approved"
    assert _ledger(round_project) == []
    assert not (round_project / "feedback" / "taste-decisions.json").exists()


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_decisions_are_shared_across_projects_with_local_completion_flag(
    round_project: Path, notes: tuple, tmp_path: Path, action: str,
) -> None:
    identity = _seed(round_project, notes)
    other = tmp_path / "other-project"
    ProposalInbox(other).propose(ProposalInbox(round_project).read())
    assert inbox_view(other)["proposals"][0]["decided_here"] is False
    decided = decide(round_project, identity, action)
    assert decided["decided_here"] is True
    elsewhere = inbox_view(other)["proposals"][0]
    assert elsewhere["state"] == decided["state"]
    assert elsewhere["rule"] == decided["rule"]
    assert elsewhere["decided_here"] is False and _ledger(other) == []
    if action == "reject":
        approved = decide(other, identity, "approve")
        assert approved["state"] == "approved" and approved["decided_here"] is True
        first = inbox_view(round_project)["proposals"][0]
        assert first["state"] == "approved" and first["decided_here"] is False


@pytest.mark.parametrize(("state", "action", "sentence", "code"), [
    ("pending", "revoke", None, "taste_not_approved"),
    ("rejected", "revoke", None, "taste_not_approved"),
    ("approved", "reject", None, "taste_approved"),
    ("approved", "approve", "New wording.", "taste_not_pending"),
    ("revoked", "approve", None, "taste_ended"),
    ("superseded", "approve", None, "taste_ended"),
    ("revoked", "approve", "New wording.", "taste_not_pending"),
    ("superseded", "approve", "New wording.", "taste_not_pending"),
    ("revoked", "revoke", None, "taste_not_approved"),
    ("superseded", "revoke", None, "taste_not_approved"),
    ("revoked", "reject", None, "taste_ended"),
    ("superseded", "reject", None, "taste_ended"),
])
def test_decision_refusal_codes_leave_records_unchanged(
    round_project: Path, notes: tuple, state: str, action: str, sentence: str | None, code: str,
) -> None:
    root, identity = round_project, _seed(round_project, notes)
    if state == "rejected":
        decide(root, identity, "reject")
    elif state != "pending":
        decide(root, identity, "approve")
        if state == "revoked":
            decide(root, identity, "revoke")
        elif state == "superseded":
            decide(root, _seed(root, notes, 4), "approve")
    before = _ledger(root), load_taste(), load_proposals()
    with pytest.raises(TasteRefusal) as caught:
        decide(root, identity, action, sentence=sentence)
    assert caught.value.code == code and caught.value.suggested_action
    if code == "taste_ended":
        assert "new review note" in caught.value.suggested_action
    if code == "taste_approved":
        assert "revoke" in caught.value.suggested_action
    assert (_ledger(root), load_taste(), load_proposals()) == before


def test_unknown_proposal_and_invalid_action(round_project: Path, notes: tuple) -> None:
    identity = _seed(round_project, notes)
    with pytest.raises(UnknownProposalError) as caught:
        decide(round_project, "r-000000000000", "approve")
    assert isinstance(caught.value, ValidationError) and caught.value.code == "unknown_proposal"
    with pytest.raises(ValidationError, match="action"):
        decide(round_project, identity, "delete")
    assert _ledger(round_project) == [] and load_taste().revision == 0


@pytest.mark.parametrize("wording", ["", "   ", "x" * 281, "One\nTwo", "One\rTwo", 3])
def test_invalid_edited_wording_is_refused_before_intent(
    round_project: Path, notes: tuple, wording,
) -> None:
    identity = _seed(round_project, notes)
    with pytest.raises(TasteError):
        decide(round_project, identity, "approve", sentence=wording)
    assert _ledger(round_project) == [] and load_taste().revision == 0
    assert inbox_view(round_project)["proposals"][0]["state"] == "pending"


@pytest.mark.parametrize("action", ["approve", "edit", "reject"])
def test_interrupted_intent_stays_pending_and_retry_succeeds(
    round_project: Path, notes: tuple, monkeypatch: pytest.MonkeyPatch, action: str,
) -> None:
    root, identity = round_project, _seed(round_project, notes)
    helper = "reject_proposal" if action == "reject" else "approve_proposal"
    kwargs = {"sentence": "Use short phrases."} if action == "edit" else {}
    command = "approve" if action == "edit" else action

    def interrupted(*args, **kw):
        assert len(_ledger(root)) == 1
        raise RuntimeError("private save interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(store, helper, interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            decide(root, identity, command, **kwargs)
    pending = inbox_view(root)["proposals"][0]
    assert pending["state"] == "pending" and pending["decided_here"] is False
    assert not taste_path().exists() and not proposals_path().exists()
    intent = copy.deepcopy(_ledger(root))
    item = decide(root, identity, command, **kwargs)
    assert item["state"] == ("rejected" if action == "reject" else "approved")
    assert item["decided_here"] is True
    assert _ledger(root)[:1] == intent and len(_ledger(root)) == 2
    assert load_taste().revision == (0 if action == "reject" else 1)


@pytest.mark.parametrize("original_pending", [False, True])
@pytest.mark.parametrize("action", ["approve", "reject", "revoke"])
def test_interrupted_edited_approval_repairs_original_before_any_retry(
    round_project: Path, notes: tuple, monkeypatch: pytest.MonkeyPatch,
    original_pending: bool, action: str,
) -> None:
    from contextlib import contextmanager

    root, identity = round_project, _seed(round_project, notes)
    body = ProposalInbox(root).read()[0]
    if original_pending:
        add_proposals(body)
    real_write, real_lock = store._write, inbox_module.project_lock
    locked, proposal_writes = False, 0

    @contextmanager
    def tracked_lock(project):
        nonlocal locked
        with real_lock(project):
            locked = True
            try:
                yield
            finally:
                locked = False

    def interrupted_write(target, data):
        nonlocal proposal_writes
        assert locked
        if target == proposals_path():
            proposal_writes += 1
            if proposal_writes == 1:
                raise OSError("proposals write interrupted")
        real_write(target, data)

    monkeypatch.setattr(inbox_module, "project_lock", tracked_lock)
    monkeypatch.setattr(store, "_write", interrupted_write)
    with pytest.raises(TasteError, match="proposals write interrupted"):
        decide(root, identity, "approve", sentence="Use short phrases.")
    item = inbox_view(root)["proposals"][0]
    assert item["state"] == "approved" and item["edited"]
    assert load_taste().revision == 1
    assert not any(proposal["status"] == "rejected" for proposal in load_proposals().proposals)
    before = _ledger(root), taste_path().read_bytes()
    if action == "reject":
        with pytest.raises(TasteRefusal) as caught:
            decide(root, identity, action)
        assert caught.value.code == "taste_approved"
    else:
        result = decide(root, identity, action)
        assert result["state"] == ("revoked" if action == "revoke" else "approved")
    assert load_proposals().proposals == ({
        **body, "id": identity, "status": "rejected", "proposed_at": NOW, "decided_at": NOW,
    },)
    assert proposal_writes == 2 and not locked
    if action != "revoke":
        assert (_ledger(root), taste_path().read_bytes()) == before
        assert decide(root, identity, "approve") == item
    else:
        with pytest.raises(TasteRefusal):
            decide(root, identity, "revoke")
        assert _ledger(root)[-1]["rule"] == item["rule"]["id"]
    assert proposal_writes == 2
    with pytest.raises(TasteRefusal, match="not pending"):
        approve(identity)


def test_project_files_only_hold_inbox_and_id_only_append_ledger(
    round_project: Path, notes: tuple, tmp_path: Path,
) -> None:
    root = round_project
    before = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    identity = _seed(root, notes)
    decide(root, identity, "reject")
    prefix = _ledger(root)
    decide(root, identity, "approve", sentence="Prefer brief caption phrases.")
    assert _ledger(root)[:len(prefix)] == prefix
    decide(root, identity, "revoke")
    ledger_path = root / "feedback" / "taste-decisions.json"
    text = ledger_path.read_text()
    assert _proposal()["sentence"] not in text
    assert "Prefer brief caption phrases." not in text and notes[1][0].text not in text
    assert "Keep this rhythm" not in text and "value" not in text and "quote" not in text
    assert [entry["action"] for entry in _ledger(root)] == ["rejected", "edited", "revoked"]
    assert all(set(entry) == {"proposal", "action", "rule", "at"} for entry in _ledger(root))
    after = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    # Saved versions already created the project's required lock bookkeeping.
    assert after - before == {"feedback/taste-proposals.json", "feedback/taste-decisions.json"}
    assert taste_path().parent == tmp_path / "config"
    assert proposals_path().parent == tmp_path / "config"
    assert taste_path().exists() and proposals_path().exists()


@pytest.mark.parametrize("bad", [
    "{broken", [], {}, {"schema": "foreign", "decisions": []},
    {"schema": inbox_module.DECISIONS_SCHEMA, "decisions": [], "extra": 1},
    {"schema": inbox_module.DECISIONS_SCHEMA, "decisions": {}},
    *[{"schema": inbox_module.DECISIONS_SCHEMA, "decisions": [entry]} for entry in [
        {}, {"proposal": "r-000000000000", "action": "approved", "rule": "r-000000000000",
             "at": NOW, "sentence": "private wording"},
        {"proposal": "bad", "action": "approved", "rule": "r-000000000000", "at": NOW},
        {"proposal": "r-000000000000", "action": "approve", "rule": "r-000000000000", "at": NOW},
        {"proposal": "r-000000000000", "action": "approved", "rule": "bad", "at": NOW},
        {"proposal": "r-000000000000", "action": "approved", "rule": "r-000000000000",
         "at": "2026-02-30T12:00:00Z"},
        {"proposal": "r-000000000000", "action": "approved", "rule": "r-000000000000",
         "at": "2026-10-06T12:00:00+00:00"},
    ]],
])
def test_corrupt_ledger_refused_and_byte_identical(
    round_project: Path, notes: tuple, bad,
) -> None:
    identity = _seed(round_project, notes)
    path = round_project / "feedback" / "taste-decisions.json"
    path.write_text(bad if isinstance(bad, str) else json.dumps(bad))
    before = path.read_bytes()
    with pytest.raises(InputError):
        inbox_view(round_project)
    with pytest.raises(InputError):
        decide(round_project, identity, "approve")
    assert path.read_bytes() == before and load_taste().revision == 0


def test_reading_view_does_not_change_files(round_project: Path, notes: tuple) -> None:
    identity = _seed(round_project, notes)
    decide(round_project, identity, "approve")
    paths = [ProposalInbox(round_project).path, round_project / "feedback" / "taste-decisions.json",
             round_project / ".yap" / "lock", taste_path()]
    before = {path: path.read_bytes() for path in paths}
    assert inbox_view(round_project)["proposals"][0]["state"] == "approved"
    assert {path: path.read_bytes() for path in paths} == before


def test_ledger_target_uses_private_state_not_intent(round_project: Path, notes: tuple) -> None:
    identity = _seed(round_project, notes)
    add_proposals(ProposalInbox(round_project).read()[0])
    approve(identity)
    path = round_project / "feedback" / "taste-decisions.json"
    path.write_text(json.dumps({"schema": inbox_module.DECISIONS_SCHEMA, "decisions": [
        {"proposal": identity, "action": "revoked", "rule": identity, "at": NOW},
    ]}))
    item = inbox_view(round_project)["proposals"][0]
    assert item["state"] == "approved" and item["decided_here"] is False


def test_decision_holds_project_lock_through_private_mutation(
    round_project: Path, notes: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    identity = _seed(round_project, notes)
    real_lock, real_approve = inbox_module.project_lock, store.approve_proposal
    state = SimpleNamespace(locked=False)

    @contextmanager
    def tracked_lock(root):
        with real_lock(root):
            state.locked = True
            try:
                yield
            finally:
                state.locked = False

    def tracked_approve(body, **kwargs):
        assert state.locked and len(_ledger(round_project)) == 1
        return real_approve(body, **kwargs)

    monkeypatch.setattr(inbox_module, "project_lock", tracked_lock)
    monkeypatch.setattr(store, "approve_proposal", tracked_approve)
    assert decide(round_project, identity, "approve")["state"] == "approved"
    assert not state.locked
