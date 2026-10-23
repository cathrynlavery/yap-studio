"""Story outline HTTP contracts, with live and socket-free coverage."""

from __future__ import annotations

import errno
import http.client
import io
import json
import socket
import stat
import threading
from email.message import Message
from pathlib import Path

import pytest
from test_feedback_outline import outline_project as outline_project
from test_review_js import PageParser

from yap_studio.errors import ValidationError
from yap_studio.feedback.outline import add_note, build_outline, list_notes
from yap_studio.review import outline_server as server_module
from yap_studio.review.outline_server import OutlineServer
from yap_studio.review.server import _CSP


@pytest.fixture(scope="module")
def socket_access() -> None:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("sandbox blocked loopback socket binding")


@pytest.fixture
def fake_binding(monkeypatch):
    bindings = set()

    class Binding:
        def __init__(self, address, handler):
            requested = address[1]
            if requested in bindings:
                raise OSError(errno.EADDRINUSE, "port busy")
            self.server_port = requested or next(
                port for port in range(20000, 20100) if port not in bindings
            )
            bindings.add(self.server_port)

        def server_close(self):
            bindings.remove(self.server_port)

    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer", Binding)
    return bindings


def _request(server, method, route, body=None, *, token="test-token", host=None, headers=None):
    fields = dict(headers or {})
    fields["Host"] = host if host is not None else f"127.0.0.1:{server.port}"
    if token is not None:
        fields["X-Yap-Token"] = token
    payload = body if isinstance(body, bytes) else (
        json.dumps(body).encode() if body is not None else b""
    )
    if body is not None and "Content-Length" not in fields:
        fields["Content-Length"] = str(len(payload))
    if getattr(server, "live", False):
        connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        try:
            connection.request(method, route, body=payload, headers=fields)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()
    handler_type = server._handler_type()
    handler = handler_type.__new__(handler_type)
    handler.command = method
    handler.path = route
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {route} HTTP/1.1"
    handler.headers = Message()
    for key, value in fields.items():
        handler.headers[key] = value
    handler.rfile = io.BytesIO(payload)
    handler.wfile = io.BytesIO()
    handler._dispatch()
    head, _, content = handler.wfile.getvalue().partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    return int(lines[0].split()[1]), dict(line.split(": ", 1) for line in lines[1:]), content


@pytest.fixture(params=["in_process", "live"])
def outline_http(outline_project, request):
    if request.param == "live":
        request.getfixturevalue("socket_access")
    else:
        request.getfixturevalue("fake_binding")
    build_outline(outline_project, summary="A plain story.")
    server = OutlineServer(outline_project, token="test-token", allowed_hosts=["phone.test:8080"])
    server.live = request.param == "live"
    thread = None
    if server.live:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        assert server._http.daemon_threads
    try:
        yield server
    finally:
        if thread is not None:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
        server.server_close()


def _json_request(server, method, route, body=None, **options):
    status, headers, raw = _request(server, method, route, body, **options)
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Cache-Control"] == "no-store"
    return status, json.loads(raw) if raw else None


@pytest.mark.parametrize("token", [None, "wrong-token", ""])
def test_outline_token_missing_or_wrong_is_forbidden(outline_http, token):
    code, payload = _json_request(outline_http, "GET", "/api/outline", token=token)
    assert code == 403 and payload["error"]["code"] == "forbidden"


def test_outline_query_token_reads_and_header_only_writes(outline_http):
    route = "/api/outline?t=test-token"
    assert _json_request(outline_http, "GET", route, token=None)[0] == 200
    code, headers, content = _request(outline_http, "HEAD", route, token=None)
    assert code == 200 and not content and int(headers["Content-Length"]) > 0
    assert _json_request(outline_http, "GET", route, token="wrong-token")[0] == 403
    for method, path, body in [
        ("POST", "/api/outline/notes", {"beat": None, "kind": "fix", "text": "Fix it"}),
        ("PATCH", "/api/outline/notes/o1", {"text": "Update"}),
        ("DELETE", "/api/outline/notes/o1", None),
        ("POST", "/api/outline/send", None),
        ("POST", "/api/outline/approve", None),
        ("POST", "/api/outline/wait", {"minutes": 15}),
    ]:
        code, _ = _json_request(outline_http, method, path + "?t=test-token", body, token=None)
        assert code == 403
    assert list_notes(outline_http.project_dir) == []


@pytest.mark.parametrize("host", ["evil.test", "localhost", "127.0.0.1:1",
                                     "phone.test:8081", "phone.test:0", "bad/host"])
def test_outline_wrong_host_is_forbidden(outline_http, host):
    code, payload = _json_request(outline_http, "GET", "/api/outline", host=host)
    assert code == 403 and payload["error"]["message"] == "invalid Host header"


def test_outline_allowed_host_and_localhost(outline_http):
    for host in [f"localhost:{outline_http.port}", "PHONE.test:8080"]:
        assert _json_request(outline_http, "GET", "/api/outline", host=host)[0] == 200


def test_outline_all_routes_success_shapes_and_sent_note_lock(outline_http):
    server = outline_http
    code, view = _json_request(server, "GET", "/api/outline")
    assert code == 200 and set(view) == {"project", "outline", "status", "notes", "older_notes"}
    assert view["project"] == "sample-story" and view["outline"]["summary"] == "A plain story."
    assert view["status"]["state"] == "waiting" and view["notes"] == []
    assert view["older_notes"] == 0
    note_ids = []
    for beat, kind in [(None, "fix"), ("s1", "keep"), ("t1.0", "fix")]:
        code, note = _json_request(server, "POST", "/api/outline/notes", {
            "beat": beat, "kind": kind, "text": "  A creator note.  ",
        })
        assert code == 201 and note["text"] == "A creator note."
        assert note["beat"] == beat and note["kind"] == kind and note["batch"] is None
        assert note["outline"] == view["outline"]["id"]
        note_ids.append(note["id"])
    code, note = _json_request(server, "PATCH", "/api/outline/notes/o1", {"kind": "keep"})
    assert code == 200 and note["kind"] == "keep"
    code, note = _json_request(server, "PATCH", "/api/outline/notes/o1", {"text": "Keep it"})
    assert code == 200 and note["text"] == "Keep it"
    assert _request(server, "DELETE", "/api/outline/notes/o3")[::2] == (204, b"")
    code, batch = _json_request(server, "POST", "/api/outline/send")
    assert code == 200 and batch == {"batch": "b1", "notes": 2}
    notes = _json_request(server, "GET", "/api/outline")[1]["notes"]
    assert len(notes) == 2 and all(note["batch"] == "b1" for note in notes)
    for method, body in [("PATCH", {"text": "Locked"}), ("DELETE", None)]:
        assert _json_request(server, method, "/api/outline/notes/o1", body)[0] == 409
    assert _json_request(server, "POST", "/api/outline/send")[0] == 409
    code, state = _json_request(server, "POST", "/api/outline/wait", {"minutes": 15})
    assert code == 200 and state == {
        "schema": "yap-studio/outline-state@1", "outline": view["outline"]["id"],
        "approved_at": None, "deadline": "2026-10-06T12:15:00Z",
        "wait_minutes": 15,
    }
    code, state = _json_request(server, "POST", "/api/outline/wait", {"minutes": None})
    assert code == 200 and state["deadline"] is None and state["wait_minutes"] is None
    code, state = _json_request(server, "POST", "/api/outline/approve")
    assert code == 200 and state["approved_at"] == "2026-10-06T12:00:00Z"
    assert _json_request(server, "GET", "/api/outline")[1]["status"]["state"] == "approved"


@pytest.mark.parametrize("path,body", [
    ("/api/outline/notes", {"beat": None, "kind": "other", "text": "Text"}),
    ("/api/outline/notes", {"beat": "absent", "kind": "fix", "text": "Text"}),
    ("/api/outline/notes", {"beat": "t1.0", "kind": "keep", "text": "Text"}),
    ("/api/outline/notes", {"beat": None, "kind": "fix", "text": " "}),
    ("/api/outline/notes", {"beat": None, "kind": "fix", "text": "x" * 5001}),
    ("/api/outline/notes", {"text": "Text"}),
    ("/api/outline/notes", []),
    ("/api/outline/notes", b"{invalid"),
    ("/api/outline/wait", {"minutes": True}),
    ("/api/outline/wait", {"minutes": 0}),
    ("/api/outline/wait", {"minutes": 10081}),
    ("/api/outline/wait", {"minutes": "15"}),
    ("/api/outline/wait", {}),
])
def test_outline_validation_is_400(outline_http, path, body):
    code, payload = _json_request(outline_http, "POST", path, body)
    assert code == 400 and payload["error"]["message"]


def test_outline_patch_validation_unknown_note_and_empty_send(outline_http):
    add_note(outline_http.project_dir, beat=None, kind="fix", text="Original")
    for body in [{}, {"beat": "s1"}, {"kind": None}, {"text": " "}]:
        assert _json_request(outline_http, "PATCH", "/api/outline/notes/o1", body)[0] == 400
    for method, body in [("PATCH", {"text": "Missing"}), ("DELETE", None)]:
        assert _json_request(outline_http, method, "/api/outline/notes/unknown", body)[0] == 404
    _request(outline_http, "DELETE", "/api/outline/notes/o1")
    assert _json_request(outline_http, "POST", "/api/outline/send")[0] == 409


@pytest.mark.parametrize("changed", ["edit.json", "removals.json",
                                     "ingest/t1/words.json", "ingest/t2/words.json"])
def test_outline_stale_refuses_writes_without_changing_notes(outline_http, changed):
    root = outline_http.project_dir
    add_note(root, beat=None, kind="fix", text="Original")
    before = (root / "outline/notes.json").read_bytes()
    if changed == "removals.json":
        (root / changed).write_text(json.dumps({
            "schema": "yap-studio/removals@1", "removals": [
                {"id": "r1", "take": "t1", "start": 1.7, "end": 1.9,
                 "kind": "manual", "status": "applied"},
            ],
        }))
    else:
        (root / changed).write_bytes((root / changed).read_bytes() + b"\n")
    assert _json_request(outline_http, "GET", "/api/outline")[1]["status"]["state"] == "stale"
    for method, path, body in [
        ("POST", "/api/outline/notes", {"beat": None, "kind": "fix", "text": "New"}),
        ("PATCH", "/api/outline/notes/o1", {"text": "Changed"}),
        ("DELETE", "/api/outline/notes/o1", None),
        ("POST", "/api/outline/send", None),
        ("POST", "/api/outline/approve", None),
        ("POST", "/api/outline/wait", {"minutes": 15}),
    ]:
        code, payload = _json_request(outline_http, method, path, body)
        assert code == 409
        assert payload["error"]["message"] == (
            "The story changed. The agent is updating the outline."
        )
    assert (root / "outline/notes.json").read_bytes() == before
    assert not (root / "feedback").exists()
    assert not (root / "outline/state.json").exists()


def test_outline_missing_and_older_notes_are_reported(outline_http):
    root = outline_http.project_dir
    add_note(root, beat=None, kind="fix", text="Earlier story")
    build_outline(root, summary="A revised story")
    view = _json_request(outline_http, "GET", "/api/outline")[1]
    assert view["notes"] == [] and view["older_notes"] == 1
    for method, body in [("PATCH", {"text": "Earlier"}), ("DELETE", None)]:
        assert _json_request(outline_http, method, "/api/outline/notes/o1", body)[0] == 409
    (root / "outline/outline.json").unlink()
    view = _json_request(outline_http, "GET", "/api/outline")[1]
    assert view["outline"] is None and view["status"]["state"] == "missing"
    assert view["notes"] == [] and view["older_notes"] == 1


def test_outline_large_body_is_413_and_bad_content_length_is_400(outline_http):
    for path in ["/api/outline/notes", "/api/outline/send", "/api/outline/approve",
                 "/api/outline/wait"]:
        assert _json_request(outline_http, "POST", path, b"x" * 65537)[0] == 413
    assert _json_request(outline_http, "POST", "/api/outline/notes", b"{}",
                         headers={"Content-Length": "invalid"})[0] == 400


@pytest.mark.parametrize("method,path", [
    ("PUT", "/api/outline"), ("GET", "/api/outline/send"),
    ("POST", "/"), ("OPTIONS", "/api/outline"),
    ("GET", "/api/outline/notes/o1"), ("POST", "/static/app.css"),
])
def test_outline_wrong_method_is_405(outline_http, method, path):
    code, headers, raw = _request(outline_http, method, path)
    assert code == 405 and headers["Allow"]
    assert json.loads(raw)["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize("path", ["/static/../outline.html", "/static/%2e%2e/app.css",
                                  "/static/nested/app.css", "/static/outline.html",
                                  "/static/missing.mjs", "/static/app.py", "/missing"])
def test_outline_static_traversal_and_unknown_routes_are_404(outline_http, path):
    assert _json_request(outline_http, "GET", path)[0] == 404


def test_outline_page_and_static_security_headers(outline_http):
    code, headers, content = _request(outline_http, "GET", "/?t=test-token", token=None)
    assert code == 200 and b"Whole story" in content
    assert headers["Content-Security-Policy"] == _CSP
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    for name in ["app.css", "outline.mjs", "outline-view.mjs"]:
        code, headers, content = _request(outline_http, "GET", "/static/" + name, token=None)
        assert code == 200 and content and headers["X-Content-Type-Options"] == "nosniff"
    assert _request(outline_http, "HEAD", "/")[::2] == (200, b"")


@pytest.mark.parametrize("live", [False, True])
def test_outline_durable_link_token_mode_saved_port_and_busy_fallback(
    outline_project, request, live,
):
    request.getfixturevalue("socket_access" if live else "fake_binding")
    first = OutlineServer(outline_project)
    first_url, first_token, first_port = first.url, first.token, first.port
    link = outline_project / ".yap/outline-link.json"
    assert stat.S_IMODE(link.stat().st_mode) == 0o600
    assert json.loads(link.read_text()) == {
        "schema": "yap-studio/outline-link@1", "token": first_token, "port": first_port,
    }
    first.server_close()
    second = OutlineServer(outline_project)
    try:
        assert second.token == first_token and second.url == first_url
        third = OutlineServer(outline_project)
        try:
            assert third.token == first_token and third.port != first_port
            assert json.loads(link.read_text())["port"] == third.port
            assert stat.S_IMODE(link.stat().st_mode) == 0o600
        finally:
            third.server_close()
    finally:
        second.server_close()


def test_outline_explicit_port_wins_and_does_not_fall_back(outline_project, fake_binding):
    first = OutlineServer(outline_project)
    first.server_close()
    second = OutlineServer(outline_project, port=23456)
    try:
        assert second.port == 23456
        with pytest.raises(OSError, match="port busy"):
            OutlineServer(outline_project, port=23456)
    finally:
        second.server_close()


def test_outline_link_write_failure_closes_socket(outline_project, fake_binding, monkeypatch):
    def fail(*_args):
        raise OSError("write failed")

    monkeypatch.setattr(server_module, "atomic_write_text", fail)
    with pytest.raises(OSError, match="write failed"):
        OutlineServer(outline_project)
    assert not fake_binding


def test_outline_invalid_allowed_host_is_rejected(outline_project):
    with pytest.raises(ValidationError, match="allow-host"):
        OutlineServer(outline_project, allowed_hosts=["https://phone.test"])


def test_outline_page_csp_text_safety_and_phone_styles():
    static = Path(server_module.__file__).parent / "static"
    parser = PageParser()
    parser.feed((static / "outline.html").read_text())
    assert not parser.inline_script
    for tag, attrs in parser.tags:
        assert tag != "style" and "style" not in attrs
        assert not any(name.startswith("on") for name in attrs)
    for name in ["outline.mjs", "outline-view.mjs"]:
        source = (static / name).read_text()
        for forbidden in ["innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "new Function"]:
            assert forbidden not in source
    css = (static / "app.css").read_text()
    assert ".outline-page button, .outline-page select { min-height: 44px;" in css
    assert ".outline-bar { position: fixed;" in css
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in css
