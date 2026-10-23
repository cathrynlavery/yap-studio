"""Home HTTP security, bounded uploads and in-process engine writes."""

from __future__ import annotations

import http.client
import json
import socket
import threading
from datetime import timedelta

import pytest
from test_feedback_batch import feedback_project as feedback_project
from test_home_bridge import clock as clock

from yap_studio.homepage import server as module


@pytest.fixture
def home(tmp_path, monkeypatch):
    (tmp_path / "inbox").mkdir()
    static = tmp_path / "static"
    static.mkdir()
    (static / "home.html").write_text("<!doctype html><title>Home test</title>")
    (static / "home.js").write_text("void 0;")
    monkeypatch.setattr(module, "_STATIC", static)
    page = module.HomeServer(tmp_path)
    thread = threading.Thread(target=page.serve_forever, daemon=True)
    thread.start()
    yield page
    page.shutdown()
    page.server_close()
    thread.join(timeout=5)


def request(home, method="GET", route="/api/state", data=None, headers=None, auth=True):
    sent = {"Cookie": f"yap_home_{home.port}={home.token}"} if auth else {}
    if method == "POST":
        sent.update({"X-Home-Token": home.token, "Content-Type": "application/json"})
    sent.update(headers or {})
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    conn = http.client.HTTPConnection("127.0.0.1", home.port, timeout=5)
    conn.request(method, route, body=body, headers=sent)
    response = conn.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    conn.close()
    return result


@pytest.fixture
def home_story(home, feedback_project):
    from yap_studio.feedback.versions import save_version

    project = home.workspace / "projects/demo"
    project.parent.mkdir()
    feedback_project.rename(project)
    save_version(project)
    return home


@pytest.mark.parametrize("action", ["resume", "cancel"])
def test_stalled_request_actions_over_http(home, clock, action):
    from yap_studio.homepage import bridge

    code, _, body = request(home, "POST", "/api/requests", {"kind": "learn"})
    row = json.loads(body)
    assert code == 201
    route = f"/api/requests/{row['id']}/{action}"
    assert request(home, "POST", route.replace(action, "resume"), {})[0] == 409
    bridge.wait(home.workspace, timeout=1)
    claimed = bridge.ack(home.workspace, row["id"], "claimed", message="Examples checked.")
    code, _, body = request(home, "POST", route, {})
    assert code == 409 and json.loads(body)["error"]["code"] == "home_conflict"
    clock[0] += timedelta(minutes=10)
    assert json.loads(request(home)[2])["requests"][0]["stalled"] is True
    code, _, body = request(home, "POST", route, {})
    assert code == 200
    updated = json.loads(body)
    assert updated["state"] == ("pending" if action == "resume" else "cancelled")
    assert updated["ack"] == claimed["ack"]
    if action == "resume":
        assert updated["resumes"] == 1
        assert request(home, "POST", route, {})[0] == 409
        assert bridge.wait(home.workspace, timeout=1)["resumed"] is True


def test_final_request_and_state(home_story):
    home = home_story
    assert json.loads(request(home)[2])["projects"][0]["final_request"] is None
    payload = {"kind": "final", "project": "projects/demo"}
    code, _, body = request(home, "POST", "/api/requests", payload)
    row = json.loads(body)
    assert code == 201 and row["kind"] == "final"
    assert row["payload"] == {"project": "projects/demo"}
    assert request(home, "POST", "/api/requests", payload)[0] == 409
    assert request(home, "POST", f"/api/requests/{row['id']}/cancel", {})[0] == 200
    assert json.loads(request(home)[2])["projects"][0]["final_request"] == {
        "id": row["id"],
        "state": "cancelled",
    }
    code, _, body = request(home, "POST", "/api/requests", payload)
    newer = json.loads(body)
    assert code == 201
    assert json.loads(request(home)[2])["projects"][0]["final_request"] == {
        "id": newer["id"],
        "state": "pending",
    }


@pytest.mark.parametrize(
    "project",
    [None, 42, "projects/missing", "../outside", "inbox", "projects/link", "projects/file"],
)
def test_final_request_requires_real_workspace_project(home_story, project):
    home = home_story
    (home.workspace / "projects/link").symlink_to(home.workspace / "projects/demo")
    (home.workspace / "projects/file").write_text("synthetic")
    assert request(home, "POST", "/api/requests", {"kind": "final", "project": project})[0] == 400


@pytest.mark.parametrize("saved", ["missing", "final"])
def test_final_request_requires_newest_story_cut(home_story, saved):
    import shutil

    from test_feedback_versions import _render_media

    from yap_studio.feedback.versions import save_version

    home = home_story
    project = home.workspace / "projects/demo"
    if saved == "missing":
        shutil.rmtree(project / "versions")
    else:
        final = project / "final/final.mp4"
        final.parent.mkdir()
        _render_media(final, 120, label="synthetic final")
        save_version(project, media=final)
    code, _, body = request(
        home, "POST", "/api/requests", {"kind": "final", "project": "projects/demo"}
    )
    assert code == 409 and json.loads(body)["error"]["message"]


@pytest.mark.parametrize("project", [None, "projects/demo"])
def test_message_request_accepts_global_or_project(home_story, project):
    payload = {"kind": "message", "project": project, "text": "  A creator message \n"}
    for _ in range(2):
        code, _, body = request(home_story, "POST", "/api/requests", payload)
        assert code == 201
        assert json.loads(body)["payload"] == {"project": project, "text": "A creator message"}


@pytest.mark.parametrize(
    "text,code",
    [
        ("x", 201),
        (" \n" + "é" * 2000 + "  ", 201),
        ("x" * 2001, 400),
        (" \t\n", 400),
        (None, 400),
        (42, 400),
    ],
)
def test_message_request_text_boundaries(home, text, code):
    assert (
        request(home, "POST", "/api/requests", {"kind": "message", "project": None, "text": text})[
            0
        ]
        == code
    )


@pytest.mark.parametrize(
    "project", [42, "projects/missing", "../outside", "inbox", "projects/link", "projects/file"]
)
def test_message_request_rejects_invalid_project(home_story, project):
    home = home_story
    (home.workspace / "projects/link").symlink_to(home.workspace / "projects/demo")
    (home.workspace / "projects/file").write_text("synthetic")
    assert (
        request(
            home, "POST", "/api/requests", {"kind": "message", "project": project, "text": "Hello"}
        )[0]
        == 400
    )


def test_token_cookie_host_origin_csp(home):
    assert request(home, auth=False)[0] == 403
    code, headers, _ = request(home, route=f"/?t={home.token}", auth=False)
    assert code == 200 and f"yap_home_{home.port}=" in headers["Set-Cookie"]
    csp = headers["Content-Security-Policy"]
    assert "img-src 'self' data: blob:" in csp and "unsafe-inline" not in csp
    assert "media-src 'self'" in csp and "font-src 'self'" in csp
    assert request(home)[0] == 200
    assert request(home, headers={"Host": "attacker.test"})[0] == 403
    assert (
        request(
            home,
            "POST",
            "/api/requests",
            {"kind": "learn"},
            headers={"Origin": "https://attacker.test"},
        )[0]
        == 403
    )
    assert (
        request(
            home, "POST", "/api/requests", {"kind": "learn"}, headers={"X-Home-Token": "wrong"}
        )[0]
        == 403
    )
    assert request(home, route="/static/home.js")[0] == 200
    assert request(home, route="/static/../home.html")[0] == 404


@pytest.mark.parametrize(
    "route,data,headers,code",
    [
        ("/api/requests", b"{}", {"Content-Type": "text/plain"}, 415),
        ("/api/requests", b"{", {}, 400),
        ("/api/requests", b"[]", {}, 400),
        ("/api/requests", b"", {"Content-Length": "1048577"}, 413),
        ("/api/nope", {}, {}, 404),
        ("/api/video-type", {"video_type": "green_screen", "broll_sources": []}, {}, 400),
        (
            "/api/requests",
            {"kind": "edit", "video_type": "talking_head", "takes": [], "broll_sources": []},
            {},
            400,
        ),
        (
            "/api/requests",
            {"kind": "edit", "video_type": "voiceover", "takes": [], "broll_sources": ["stock"]},
            {},
            400,
        ),
    ],
)
def test_errors(home, route, data, headers, code):
    actual, _, body = request(home, "POST", route, data, headers)
    assert actual == code and json.loads(body)["error"]["message"]


def test_voiceover_request_conflict_and_cancel(home):
    data = {
        "kind": "edit",
        "video_type": "voiceover",
        "takes": [],
        "broll_sources": ["folder"],
        "album": None,
    }
    code, _, body = request(home, "POST", "/api/requests", data)
    assert code == 201
    row = json.loads(body)
    assert request(home, "POST", "/api/requests", data)[0] == 409
    assert request(home, "POST", f"/api/requests/{row['id']}/cancel", {})[0] == 200
    assert request(home, "POST", f"/api/requests/{row['id']}/cancel", {})[0] == 409


def test_video_type_answers_and_style_page(home, monkeypatch):
    from yap_studio.style.library import get_default, library_dir
    from yap_studio.video_type import load_default

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(home.workspace / "config"))
    assert (
        request(
            home,
            "POST",
            "/api/video-type",
            {"video_type": "talking_head", "broll_sources": ["folder"]},
        )[0]
        == 200
    )
    assert load_default()["broll_sources"] == ["folder"]
    code, _, body = request(
        home,
        "POST",
        "/api/answers",
        {"answers": {"cuts": "tight", "brand": {"accent_color": "#114488"}}},
    )
    assert code == 200 and "brand.accent_color" in json.loads(body)["applied"]
    assert not (home.workspace / "style/my-style.json").exists()
    assert get_default() is None and not library_dir().exists()
    assert json.loads(request(home)[2])["style"]["saved"] is False
    assert request(home, "POST", "/api/answers", {"answers": {"music": "none"}})[0] == 200
    saved = json.loads((home.workspace / "style/answers.json").read_text())
    assert saved["cuts"] == "tight" and saved["music"] == "none"
    launched = []
    monkeypatch.setattr(
        module,
        "launch_style_page",
        lambda workspace, phone: (
            launched.append(phone) or {"url": "http://127.0.0.1:42/?t=test", "phone": None}
        ),
    )
    assert request(home, "POST", "/api/style-page", {})[0] == 200
    home.state.phone = {"via": "wifi", "url": "http://192.168.1.4:9/?t=test"}
    assert request(home, "POST", "/api/style-page", {})[0] == 200
    # Requests from the Mac keep the style page local, even when home has a phone link.
    assert launched == [False, False]


def test_answers_update_saved_look_in_place(home, monkeypatch):
    from yap_studio.style import resolve, save_pick
    from yap_studio.style.library import library_dir

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(home.workspace / "config"))
    entry = save_pick(resolve("preset:clean_minimal").unbranded_data, name="My look",
                      workspace=home.workspace)
    assert request(home, "POST", "/api/answers", {"answers": {"music": "none"}})[0] == 200
    assert resolve(entry.id).data["music"]["level"] == 0
    assert list(library_dir().glob("*.json")) == [entry.path]
    assert (home.workspace / "style/my-style.json").read_bytes() == entry.path.read_bytes()
    assert json.loads(request(home)[2])["style"]["saved"] is True


def upload(home, name="clip.mp4", body=b"synthetic", headers=None):
    return request(
        home,
        "POST",
        "/api/inbox",
        body,
        {"Content-Type": "application/octet-stream", "X-File-Name": name, **(headers or {})},
    )


@pytest.mark.parametrize("shape,width,height,code", [
    ("horizontal", 320, 180, 201), ("horizontal", 180, 320, 400),
    ("vertical", 320, 180, 400), ("vertical", 180, 320, 201),
    ("horizontal", 180, 180, 400), ("horizontal", None, None, 400),
    ("vertical", None, None, 201), ("square", 320, 180, 400),
])
def test_edit_shape_validation_and_payload(home, monkeypatch, shape, width, height, code):
    (home.workspace / "inbox/clip.mp4").write_bytes(b"synthetic")
    monkeypatch.setattr(home.state, "metadata", lambda p: ("take", 1.0, width, height))
    actual, _, body = request(home, "POST", "/api/requests", {
        "kind": "edit", "video_type": "talking_head", "broll_sources": [],
        "takes": ["clip.mp4"], "shape": shape,
    })
    assert actual == code
    row = json.loads(body)
    if code == 201:
        assert row["payload"]["shape"] == shape
        if shape == "horizontal":
            assert row["payload"]["captions"] is False
        else:
            assert "captions" not in row["payload"]
    else:
        assert any(word in row["error"]["message"] for word in ("shape", "take", "Choose"))


def test_upload_and_collision(home, monkeypatch):
    monkeypatch.setattr(home.state, "metadata", lambda p: ("take", 1.0, 1080, 1920))
    for expected in ("clip.mp4", "clip (2).mp4"):
        code, _, body = upload(home)
        assert code == 201 and json.loads(body)["name"] == expected
        assert (home.workspace / "inbox" / expected).read_bytes() == b"synthetic"


@pytest.mark.parametrize(
    "sent,expected",
    [
        ("My%20take%203.mp4", "My take 3.mp4"),
        ("Caf%C3%A9.mov", "Café.mov"),
        ("plain.mp4", "plain.mp4"),
        ("literal%252F.mp4", "literal%2F.mp4"),
        ("100%25.mp4", "100%.mp4"),
    ],
)
def test_upload_decodes_file_name_once(home, monkeypatch, sent, expected):
    monkeypatch.setattr(home.state, "metadata", lambda p: ("take", 1.0, 1080, 1920))
    code, _, body = upload(home, sent)
    assert code == 201
    assert json.loads(body)["name"] == expected
    assert (home.workspace / "inbox" / expected).read_bytes() == b"synthetic"


@pytest.mark.parametrize(
    "name",
    [
        "a%2Fb.mp4",
        "a%5Cb.mp4",
        "%2Ehidden.mp4",
        "bad%00.mp4",
        "bad%FF.mp4",
        "bad%C3.mp4",
        "bad%ZZ.mp4",
        "bad%2.mp4",
    ],
)
def test_upload_refuses_invalid_encoded_name(home, name):
    assert upload(home, name)[0] == 400
    assert not list((home.workspace / "inbox").iterdir())


@pytest.mark.parametrize(
    "name", ["../x.mp4", ".hidden.mp4", "a/b.mp4", "a\\b.mp4", "bad.txt", "control\t.mp4"]
)
def test_upload_bad_names(home, name):
    assert upload(home, name)[0] == 400


def test_upload_limits_and_free_space(home, monkeypatch):
    assert upload(home, headers={"Content-Length": str(16 * 1024**3 + 1)})[0] == 413
    assert upload(home, headers={"Content-Length": "no"})[0] == 400
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    assert upload(home)[0] == 413
    assert not list((home.workspace / "inbox").iterdir())


def test_broken_upload_removes_part(home):
    conn = socket.create_connection(("127.0.0.1", home.port), timeout=5)
    conn.sendall(
        (
            f"POST /api/inbox HTTP/1.0\r\nHost: 127.0.0.1:{home.port}\r\n"
            f"X-Home-Token: {home.token}\r\nContent-Type: application/octet-stream\r\n"
            "X-File-Name: broken.mp4\r\nContent-Length: 100\r\n\r\nshort"
        ).encode()
    )
    conn.shutdown(socket.SHUT_WR)
    while conn.recv(4096):
        pass
    conn.close()
    assert not list((home.workspace / "inbox").iterdir())


def test_thumbnail_routes_and_static_whitelist(home):
    from test_feedback_versions import _render_media

    _render_media(home.workspace / "inbox/clip.mp4", 30)
    assert request(home, route="/thumb/clip.mp4")[0] == 200
    (home.workspace / "inbox/memo.m4a").write_bytes(b"synthetic memo")
    assert request(home, route="/thumb/memo.m4a")[0] == 404
    assert request(home, route="/thumb/absent.mp4")[0] == 404
    assert request(home, route="/static/home.html")[0] == 404
    assert request(home, route="/api/download/../anything")[0] == 404
    assert request(home, "POST", "/api/requests/r999/cancel", {})[0] == 404


@pytest.mark.parametrize(
    "video,names,code",
    [
        ("talking_head", ["video.mp4"], 201),
        ("voiceover", ["memo.m4a"], 201),
        ("voiceover", ["video.mp4"], 400),
        ("talking_head", ["memo.m4a"], 400),
        ("talking_head", ["video.mp4", "memo.m4a"], 400),
        ("talking_head", ["absent.mp4"], 400),
        ("talking_head", ["video.mp4", "video.mp4"], 400),
    ],
)
def test_request_take_fit(home, video, names, code):
    for name in ("video.mp4", "memo.m4a"):
        (home.workspace / "inbox" / name).write_bytes(b"synthetic")
    assert (
        request(
            home,
            "POST",
            "/api/requests",
            {"kind": "edit", "video_type": video, "takes": names, "broll_sources": ["folder"]},
        )[0]
        == code
    )


def test_json_and_upload_require_length(home):
    for route, mime in (
        ("/api/requests", "application/json"),
        ("/api/inbox", "application/octet-stream"),
    ):
        conn = http.client.HTTPConnection("127.0.0.1", home.port, timeout=5)
        conn.putrequest("POST", route)
        conn.putheader("X-Home-Token", home.token)
        conn.putheader("Content-Type", mime)
        conn.putheader("X-File-Name", "clip.mp4")
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 400
        response.read()
        conn.close()


def test_download_and_reveal_require_approved_and_loopback(home, monkeypatch):
    assert request(home, route="/api/download/demo")[0] == 404
    assert request(home, "POST", "/api/reveal", {"project": "projects/demo"})[0] == 404
    final = home.workspace / "projects/demo/versions/v1/media.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"synthetic final")
    monkeypatch.setattr(home.state, "approved_file", lambda name: final)
    opened = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw: opened.append(args))
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert request(home, "POST", "/api/reveal", {"project": "projects/demo"})[0] == 200
    assert opened == [["open", "-R", str(final)]]
    home.allowed_hosts.add("phone.test")
    assert (
        request(home, "POST", "/api/reveal", {"project": "projects/demo"}, {"Host": "phone.test"})[
            0
        ]
        == 403
    )
    code, headers, body = request(home, route="/api/download/demo")
    assert code == 200 and body == b"synthetic final"
    assert int(headers["Content-Length"]) == len(body)
    assert headers["Content-Disposition"].startswith("attachment;")
    assert 'filename="demo.mp4"' in headers["Content-Disposition"]


def test_open_approved_video_on_this_mac(home, monkeypatch):
    assert request(home, "POST", "/api/open", {"project": "projects/demo"})[0] == 404
    final = home.workspace / "projects/demo/versions/v1/media.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"synthetic final")
    monkeypatch.setattr(home.state, "approved_file", lambda name: final)
    opened = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw: opened.append(args))
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert request(home, "POST", "/api/open", {"project": "projects/demo"})[0] == 200
    assert opened == [["open", str(final)]]
    home.allowed_hosts.add("phone.test")
    assert (
        request(home, "POST", "/api/open", {"project": "projects/demo"}, {"Host": "phone.test"})[0]
        == 403
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    assert request(home, "POST", "/api/open", {"project": "projects/demo"})[0] == 400
    assert opened == [["open", str(final)]]


@pytest.fixture
def approved_home(home_story):
    from test_feedback_versions import _json, _render_media

    from yap_studio.feedback.release import approve
    from yap_studio.feedback.versions import save_version

    project = home_story.workspace / "projects/demo"
    _json(project / "cut/verify.json", {"status": "PASS", "findings": []})
    final = project / "final/final.mp4"
    final.parent.mkdir()
    _render_media(final, 120, label="synthetic final")
    saved = save_version(project, media=final)
    approve(project, saved.id)
    return home_story, saved


def assert_not_approved(home):
    code, _, body = request(home)
    assert code == 200
    project = json.loads(body)["projects"][0]
    assert project["phase"] == "final_ready"
    assert project["approved"] is None
    assert home.state.approved_file("demo") is None
    assert request(home, route="/api/download/demo")[0] == 404
    for route in ("/api/open", "/api/reveal"):
        assert request(home, "POST", route, {"project": "projects/demo"})[0] == 404


def test_older_approval_cannot_serve_newest_final(approved_home):
    from test_feedback_versions import _render_media

    from yap_studio.feedback.versions import save_version

    home, saved = approved_home
    assert json.loads(request(home)[2])["projects"][0]["phase"] == "approved"
    assert request(home, route="/api/download/demo")[0] == 200
    final = saved.path.parents[1] / "final/final.mp4"
    _render_media(final, 120, label="newer synthetic final")
    save_version(saved.path.parents[1], media=final)
    assert_not_approved(home)


def test_approval_hash_cached_and_tampering_invalidates(approved_home, monkeypatch):
    import os

    from yap_studio.homepage import bridge

    home, saved = approved_home
    original = bridge.sha256_file
    calls = []

    def hashed(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(bridge, "sha256_file", hashed)
    for _ in range(2):
        assert json.loads(request(home)[2])["projects"][0]["phase"] == "approved"
        assert home.state.approved_file("demo") == saved.media_path
        assert bridge._history(home.workspace)[1] == [("projects/demo", saved.id)]
        assert request(home, route="/api/download/demo")[0] == 200
    assert calls == [saved.media_path.resolve()]
    before = saved.media_path.stat()
    with saved.media_path.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    os.utime(saved.media_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert saved.media_path.stat().st_size == before.st_size
    assert_not_approved(home)
    assert len(calls) == 2
    assert bridge._history(home.workspace)[1] == []


@pytest.mark.parametrize("content", ["{", "{}", '{"schema": "broken"}'])
def test_malformed_approval_does_not_break_state(approved_home, content):
    home, saved = approved_home
    (saved.path.parent / "approval.json").write_text(content)
    assert_not_approved(home)


def test_server_seeds_history_before_first_wait(home):
    from yap_studio.homepage import bridge

    root = bridge.directory(home.workspace)
    assert bridge.read_json(root / "seen.json") == {}
    feedback = home.workspace / "projects/demo/feedback"
    feedback.mkdir(parents=True)
    (feedback / "b1.json").write_text("{}")
    # Reopening home must preserve the original baseline too.
    second = module.HomeServer(home.workspace)
    second.server_close()
    assert bridge.wait(home.workspace, timeout=1) == {
        "event": "notes", "project": "projects/demo", "batch": "b1"
    }


def test_cookie_redirect_and_private_link_help(home):
    code, headers, body = request(home, route="/")
    assert code == 303 and headers["Location"] == f"/?t={home.token}"
    assert body == b""
    code, headers, body = request(home, route="/", auth=False)
    assert code == 403 and headers["Content-Type"].startswith("text/html")
    assert b"This page needs its private link. Ask Claude to open Yap Studio for you." in body
    assert b"<script" not in body and b"style=" not in body
    assert home.token.encode() not in body
    assert "unsafe-inline" not in headers["Content-Security-Policy"]
    assert b"Open Yap Studio on this Mac" not in body
    code, headers, _ = request(home, auth=False)
    assert code == 403 and headers["Content-Type"].startswith("application/json")


def test_tailnet_visitor_without_cookie_gets_a_token_free_link_to_this_mac(tmp_path):
    (tmp_path / "inbox").mkdir()
    page = module.HomeServer(tmp_path, allowed_hosts=("box.tailnet.example.invalid",))
    thread = threading.Thread(target=page.serve_forever, daemon=True)
    thread.start()
    try:
        code, _, body = request(page, route="/", auth=False,
                                headers={"Host": f"box.tailnet.example.invalid:{page.port}"})
        assert code == 403
        assert f'<a href="http://127.0.0.1:{page.port}/">Open Yap Studio on this Mac</a>'.encode() \
            in body
        assert page.token.encode() not in body
    finally:
        page.shutdown()
        page.server_close()
        thread.join(timeout=5)


def test_home_cookies_are_isolated_by_port(home):
    second = module.HomeServer(home.workspace)
    thread = threading.Thread(target=second.serve_forever, daemon=True)
    thread.start()
    try:
        first_cookie = request(home, route=f"/?t={home.token}", auth=False)[1]["Set-Cookie"]
        second_cookie = request(second, route=f"/?t={second.token}", auth=False)[1]["Set-Cookie"]
        first_cookie, second_cookie = (
            cookie.split(";", 1)[0] for cookie in (first_cookie, second_cookie)
        )
        assert first_cookie.split("=")[0] != second_cookie.split("=")[0]
        cookies = {"Cookie": first_cookie + "; " + second_cookie}
        assert request(home, headers=cookies, auth=False)[0] == 200
        assert request(second, headers=cookies, auth=False)[0] == 200
        assert request(home, headers={"Cookie": second_cookie}, auth=False)[0] == 403
    finally:
        second.shutdown()
        second.server_close()
        thread.join(timeout=5)
