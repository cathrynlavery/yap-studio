"""Taste review HTTP contract and the same dispatcher without loopback sockets."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_review_server import _in_process_request
from test_review_server import project_server as project_server
from test_review_server import request as http_request
from test_review_server import round_project as round_project
from test_review_server import socket_access as socket_access

from yap_studio.feedback.taste import ProposalInbox
from yap_studio.review import server as server_module
from yap_studio.review.launch import version_video
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore
from yap_studio.taste import (
    decide,
    inbox_view,
    load_proposals,
    load_taste,
    proposals_path,
    propose,
    taste_path,
)


@pytest.fixture(autouse=True)
def private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep private taste records outside the project for every transport."""
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "private-config"))


@pytest.fixture(params=["http", "dispatch"])
def taste_client(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch):
    """Run every contract over real HTTP and over the socket free dispatcher."""
    if request.param == "http":
        server = request.getfixturevalue("project_server")
        send = http_request
    else:
        root = request.getfixturevalue("round_project")
        monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer", lambda *_args:
                            SimpleNamespace(server_port=0, server_close=lambda: None))
        store = ReviewStore(tmp_path / "data")
        video = version_video([str(root)], store, lambda _: [])
        server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                              series=video.series)
        send = _in_process_request
    return server, partial(send, server)


@pytest.fixture
def proposed(taste_client) -> list[str]:
    server, _ = taste_client
    refs = []
    for version, kind in zip(server.series.versions, ("keep", "fix"), strict=True):
        comment = server.store.add_comment(version.sha256, version.media.facts(version.sha256), {
            "kind": kind, "text": f"A {kind} note for the rule.", "start_frame": 12,
        })
        refs.append({"version": version.version_id, "id": comment.id})
    source = {"proposals": [
        {"scope": "captions", "sentence": "Keep captions short.", "comments": refs},
        {"scope": "pacing", "sentence": "Keep a pause before the ending.", "comments": refs},
        {"scope": "cards", "sentence": "Use fewer cards.", "comments": refs},
    ]}
    return [item["id"] for item in propose(server.series.project_dir, source, store=server.store)]


def post_decision(client, identity: str, action: str, **extra) -> dict:
    server, send = client
    status, _, payload = send("POST", "/api/taste", {
        "proposal": identity, "action": action, **extra,
    })
    assert status == 200, payload
    response = json.loads(payload)
    assert set(response) == {"taste"}
    taste = response["taste"]
    assert taste == json.loads(send("GET", "/api/review")[2])["series"]["taste"]
    expected = inbox_view(server.series.project_dir)
    assert taste["revision"] == expected["revision"] and taste["available"]
    assert [item["state"] for item in taste["proposals"]] == [
        item["state"] for item in expected["proposals"]
    ]
    return next(item for item in taste["proposals"] if item["id"] == identity)


def ledger(server: ReviewServer) -> list[dict]:
    return json.loads((server.series.project_dir / "feedback" / "taste-decisions.json")
                      .read_text())["decisions"]


def test_get_taste_states_and_source_versions(taste_client, proposed: list[str]) -> None:
    server, send = taste_client
    root = server.series.project_dir
    decide(root, proposed[1], "reject")
    decide(root, proposed[2], "approve")
    status, _, payload = send("GET", "/api/review?version=1")
    assert status == 200
    review = json.loads(payload)
    taste = review["series"]["taste"]
    assert taste["available"] and taste["revision"] == 1
    assert [item["id"] for item in taste["proposals"]] == proposed
    assert [item["state"] for item in taste["proposals"]] == ["pending", "rejected", "approved"]
    assert [item["decided_here"] for item in taste["proposals"]] == [False, True, True]
    assert [source["version"] for source in taste["proposals"][0]["sources"]] == [1, 2]
    assert review["comments"][0]["text"] == "A keep note for the rule."
    assert all("version" not in source for body in ProposalInbox(root).read()
               for source in body["sources"])


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_review_exposes_creator_decision_from_another_project(
    taste_client, proposed, tmp_path: Path, action: str,
) -> None:
    server, send = taste_client
    other = tmp_path / "other-project"
    ProposalInbox(other).propose(ProposalInbox(server.series.project_dir).read())
    decide(other, proposed[0], action)
    status, _, payload = send("GET", "/api/review")
    assert status == 200
    item = json.loads(payload)["series"]["taste"]["proposals"][0]
    assert item["state"] == ("approved" if action == "approve" else "rejected")
    assert item["decided_here"] is False
    if action == "reject":
        assert post_decision(taste_client, proposed[0], "approve")["decided_here"] is True


def test_source_outside_series_has_null_version(taste_client, proposed, monkeypatch) -> None:
    server, send = taste_client
    view = inbox_view(server.series.project_dir)
    view["proposals"][0]["sources"][0]["media_sha256"] = "f" * 64
    monkeypatch.setattr(server_module, "inbox_view", lambda _root: view)
    status, _, payload = send("GET", "/api/review?version=2")
    assert status == 200
    sources = json.loads(payload)["series"]["taste"]["proposals"][0]["sources"]
    assert [item["version"] for item in sources] == [None, 2]


@pytest.mark.parametrize("plain_file", [False, True])
def test_non_project_has_no_taste_and_refuses_decisions(
    taste_client, proposed, monkeypatch, plain_file: bool,
) -> None:
    server, send = taste_client
    series = None if plain_file else replace(server.series, project_dir=None)
    monkeypatch.setattr(server, "series", series)
    status, _, payload = send("GET", "/api/review")
    assert status == 200
    review = json.loads(payload)
    assert review["series"] is None if plain_file else "taste" not in review["series"]
    status, _, payload = send("POST", "/api/taste", {"proposal": proposed[0], "action": "approve"})
    assert status == 409 and json.loads(payload)["error"]["code"] == "no_project"


@pytest.mark.parametrize("broken", ["ledger", "inbox", "taste", "private_proposals"])
def test_corrupt_taste_does_not_hide_comments(taste_client, proposed, broken: str) -> None:
    server, send = taste_client
    root = server.series.project_dir
    target = {"ledger": root / "feedback" / "taste-decisions.json",
              "inbox": ProposalInbox(root).path, "taste": taste_path(),
              "private_proposals": proposals_path()}[broken]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("broken json")
    status, _, payload = send("GET", "/api/review")
    assert status == 200
    review = json.loads(payload)
    assert review["comments"][0]["text"] == "A fix note for the rule."
    taste = review["series"]["taste"]
    assert set(taste) == {"available", "error", "proposals"}
    assert taste["available"] is False and taste["proposals"] == []
    assert taste["error"]["message"]
    assert target.read_text() == "broken json"


def test_approve_and_revoke_write_private_history(taste_client, proposed) -> None:
    server, _ = taste_client
    identity = proposed[0]
    approved = post_decision(taste_client, identity, "approve")
    assert approved["state"] == "approved" and approved["edited"] is False
    assert approved["rule"]["approved_revision"] == 1
    assert taste_path().parent != server.series.project_dir
    assert load_taste().rules[0]["sentence"] == "Keep captions short."
    revoked = post_decision(taste_client, identity, "revoke")
    assert revoked["state"] == "revoked" and revoked["rule"]["ended_revision"] == 2
    assert [entry["action"] for entry in load_taste().history] == ["approved", "revoked"]
    assert [entry["action"] for entry in ledger(server)] == ["approved", "revoked"]
    assert all(set(entry) == {"proposal", "action", "rule", "at"} for entry in ledger(server))


def test_reject_then_approve_instead(taste_client, proposed) -> None:
    server, _ = taste_client
    rejected = post_decision(taste_client, proposed[0], "reject")
    assert rejected["state"] == "rejected" and rejected["rule"] is None
    assert not taste_path().exists()
    assert load_proposals().proposals[0]["status"] == "rejected"
    assert proposals_path().parent != server.series.project_dir
    approved = post_decision(taste_client, proposed[0], "approve")
    assert approved["state"] == "approved" and load_taste().revision == 1
    assert load_proposals().proposals == ()
    assert [entry["action"] for entry in ledger(server)] == ["rejected", "approved"]


def test_edited_approve_and_revoke_keep_original_proposal(taste_client, proposed) -> None:
    server, _ = taste_client
    edited = post_decision(taste_client, proposed[0], "approve", sentence="  Use short phrases.  ")
    assert edited["sentence"] == "Keep captions short." and edited["edited"]
    assert edited["rule"]["sentence"] == "Use short phrases."
    target = edited["rule"]["id"]
    assert target != proposed[0] and load_taste().rules[0]["id"] == target
    assert load_proposals().proposals[0]["id"] == proposed[0]
    assert ledger(server)[0]["action"] == "edited" and ledger(server)[0]["rule"] == target
    revoked = post_decision(taste_client, proposed[0], "revoke")
    assert revoked["state"] == "revoked" and revoked["rule"]["id"] == target
    assert load_taste().rules[0]["status"] == "revoked"


def test_taste_auth_methods_unknown_and_refusals(taste_client, proposed) -> None:
    server, send = taste_client
    body = {"proposal": proposed[0], "action": "approve"}
    assert send("POST", "/api/taste", body, token=None)[0] == 403
    assert send("POST", "/api/taste?t=test-token", body, token=None)[0] == 403
    assert send("POST", "/api/taste", body, host="wrong.example")[0] == 403
    status, headers, _ = send("GET", "/api/taste?t=test-token", token=None)
    assert status == 405 and headers["Allow"] == "POST"
    for method in ("HEAD", "PATCH", "DELETE", "PUT"):
        assert send(method, "/api/taste", body)[0] == 405
    status, _, payload = send("POST", "/api/taste", {**body, "proposal": "r-000000000000"})
    assert status == 404 and json.loads(payload)["error"]["code"] == "unknown_proposal"
    status, _, payload = send("POST", "/api/taste", {**body, "action": "revoke"})
    assert status == 409 and json.loads(payload)["error"]["code"] == "taste_not_approved"
    post_decision(taste_client, proposed[0], "approve")
    status, _, payload = send("POST", "/api/taste", {**body, "action": "reject"})
    assert status == 409 and json.loads(payload)["error"]["code"] == "taste_approved"
    assert len(ledger(server)) == 1


@pytest.mark.parametrize("change", [
    {"extra": True}, {"sentence": "Changed", "action": "reject"},
    {"sentence": "Changed", "action": "revoke"}, {"sentence": 7}, {"sentence": None},
    {"sentence": ""}, {"sentence": "x" * 281}, {"sentence": "Two\nlines"},
    {"proposal": 7}, {"proposal": None}, {"action": 7}, {"action": []}, {"action": "edit"},
])
def test_bad_taste_body_is_400(taste_client, proposed, change: dict) -> None:
    server, send = taste_client
    status, _, payload = send("POST", "/api/taste", {
        "proposal": proposed[0], "action": "approve", **change,
    })
    assert status == 400 and json.loads(payload)["error"]["type"] == "validation_error"
    assert not taste_path().exists() and not proposals_path().exists()
    assert not (server.series.project_dir / "feedback" / "taste-decisions.json").exists()


@pytest.mark.parametrize("body", [{}, {"action": "approve"}, {"proposal": "r-000000000000"}, [], 7])
def test_incomplete_taste_body_is_400(taste_client, body) -> None:
    _, send = taste_client
    assert send("POST", "/api/taste", body)[0] == 400


def test_taste_is_not_media_guarded(taste_client, proposed) -> None:
    server, send = taste_client
    server.media.path.write_bytes(b"changed media")
    assert json.loads(send("GET", "/api/review")[2])["media_changed"] is True
    assert post_decision(taste_client, proposed[0], "approve")["state"] == "approved"


def test_exports_never_include_taste(taste_client, proposed) -> None:
    _, send = taste_client
    edited = post_decision(taste_client, proposed[0], "approve", sentence="Use short phrases.")
    for route in ("/api/export.json", "/api/export.txt"):
        status, _, payload = send("GET", route)
        assert status == 200
        assert b"A fix note for the rule." in payload
        for private in (b"taste", b"Keep captions short.", b"Use short phrases.",
                        proposed[0].encode(), edited["rule"]["id"].encode()):
            assert private not in payload
