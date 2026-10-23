"""Detached home discovery and links on the existing pages."""

from __future__ import annotations

import json

from test_feedback_batch import feedback_project as feedback_project
from test_gallery_pick import gallery as gallery
from test_review_server import _in_process_request
from test_review_server import removed_server as removed_server

from yap_studio import background
from yap_studio.homepage import bridge, link


def test_link_only_live_containing_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    record = tmp_path / "home.json"
    background.write_json(
        record,
        {
            "pid": 123,
            "workspace": str(workspace),
            "document": {
                "url": "http://127.0.0.1:42/?t=test",
                "phone": {"url": "https://phone.test/?t=test", "via": "tailscale"},
            },
        },
    )
    monkeypatch.setattr(background, "records", lambda kind: [record])
    monkeypatch.setattr(background, "command_matches", lambda *args: True)
    assert link.for_path(workspace / "projects/demo") == {
        "url": "http://127.0.0.1:42/",
        "phone_url": "https://phone.test/",
        "phone_via": "tailscale",
        "listening": False,
        "delivered": [],
        "final_requested": False,
    }
    (workspace / "projects/demo").mkdir(parents=True)
    bridge.write_json(
        bridge.directory(workspace) / "events.json",
        [
            {
                "id": "e1",
                "kind": "notes",
                "project": "projects/demo",
                "detail": {"batch": "b2"},
                "created_at": "2026-10-19T10:00:00Z",
                "delivered_at": "2026-10-19T10:00:00Z",
            },
            {
                "id": "e2",
                "kind": "notes",
                "project": "projects/other",
                "detail": {"batch": "b9"},
                "created_at": "2026-10-19T10:00:00Z",
                "delivered_at": "2026-10-19T10:00:00Z",
            },
        ],
    )
    assert link.for_path(workspace / "projects/demo")["delivered"] == ["b2"]
    assert link.for_path(workspace)["delivered"] == []
    assert link.for_path(tmp_path / "outside") is None
    monkeypatch.setattr(background, "command_matches", lambda *args: False)
    assert link.for_path(workspace) is None


def test_link_final_requested_tracks_project_and_active_state(
    tmp_path, feedback_project, monkeypatch
):
    from yap_studio.feedback.versions import save_version

    (tmp_path / "inbox").mkdir()
    project = tmp_path / "projects/demo"
    project.parent.mkdir()
    feedback_project.rename(project)
    save_version(project)
    record = tmp_path / "home.json"
    background.write_json(
        record,
        {
            "pid": 123,
            "workspace": str(tmp_path),
            "document": {"url": "http://127.0.0.1:42/?t=test", "phone": None},
        },
    )
    monkeypatch.setattr(background, "records", lambda kind: [record] if kind == "home" else [])
    monkeypatch.setattr(background, "command_matches", lambda *args: True)
    assert link.workspace_for_path(project) == tmp_path
    assert link.for_path(project)["final_requested"] is False
    row = bridge.create_request(tmp_path, "final", {"project": "projects/demo"})
    assert link.for_path(project)["final_requested"] is True
    assert link.for_path(project / "cut/cut.mp4")["final_requested"] is True
    assert link.for_path(tmp_path / "projects/other")["final_requested"] is False
    assert link.for_path(tmp_path)["final_requested"] is False
    bridge.wait(tmp_path, timeout=1)
    assert link.for_path(project)["final_requested"] is True
    bridge.ack(tmp_path, row["id"], "claimed")
    assert link.for_path(project)["final_requested"] is True
    bridge.ack(tmp_path, row["id"], "done")
    assert link.for_path(project)["final_requested"] is False


def test_gallery_exposes_home(gallery, monkeypatch):
    expected = {"url": "http://127.0.0.1:42/?t=test", "phone_url": None, "listening": True}
    seen = []
    monkeypatch.setattr(link, "for_path", lambda p: seen.append(p) or expected)
    assert gallery.state()["home"] == expected
    assert seen == [gallery.workspace]


def test_review_exposes_home(removed_server, monkeypatch):
    expected = {"url": "http://127.0.0.1:42/?t=test", "phone_url": None, "listening": True}
    seen = []
    monkeypatch.setattr(link, "for_path", lambda p: seen.append(p) or expected)
    code, _, body = _in_process_request(removed_server, "GET", "/api/review")
    assert code == 200 and json.loads(body)["home"] == expected
    assert seen == [removed_server.series.project_dir]


def detached_home(tmp_path, workspace, monkeypatch):
    token = "home-only-private-secret-288"
    record = tmp_path / "private-home.json"
    background.write_json(record, {
        "pid": 123, "workspace": str(workspace),
        "document": {"url": f"http://127.0.0.1:42/?t={token}",
                     "phone": {"url": f"https://phone.test/?t={token}", "via": "tailscale"}},
    })
    monkeypatch.setattr(background, "records", lambda kind: [record] if kind == "home" else [])
    monkeypatch.setattr(background, "command_matches", lambda *args: True)
    return token


def test_review_body_never_contains_home_token(removed_server, tmp_path, monkeypatch):
    token = detached_home(tmp_path, tmp_path, monkeypatch)
    code, _, body = _in_process_request(removed_server, "GET", "/api/review")
    assert code == 200
    assert json.loads(body)["home"]["url"] == "http://127.0.0.1:42/"
    assert token.encode() not in body
    assert "t=" not in json.dumps(json.loads(body)["home"])


def test_style_state_never_contains_home_token(gallery, tmp_path, monkeypatch):
    from test_gallery_pick import _request

    token = detached_home(tmp_path, gallery.workspace, monkeypatch)
    code, body = _request(gallery, "GET", f"/api/state?t={gallery.token}")
    assert code == 200
    assert json.loads(body)["home"]["phone_url"] == "https://phone.test/"
    assert token.encode() not in body
    assert "t=" not in json.dumps(json.loads(body)["home"])
