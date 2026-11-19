"""Offline review: notes saved once per client id, the worker and the Home Screen manifest."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

import pytest
from test_review_server import SHA, request
from test_review_server import multi_server as multi_server
from test_review_server import server as server
from test_review_server import socket_access as socket_access

from yap_studio.errors import ValidationError
from yap_studio.review.model import MediaFacts
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore, UnknownCommentError

FACTS = MediaFacts("sample.mp4", SHA, 12.0, Fraction(30000, 1001), True)


def test_store_saves_a_client_note_once_and_keeps_the_review_file_shape(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    body = {"kind": "fix", "text": "Trim the start", "start_frame": 3}
    first, created = store.add_client_comment(SHA, FACTS, body, "a1b2c3d4e5f6")
    again, repeated = store.add_client_comment(SHA, FACTS, body, "a1b2c3d4e5f6")
    other, _ = store.add_client_comment(SHA, FACTS, body, "f6e5d4c3b2a1")
    assert (created, repeated) == (True, False)
    assert again == first and other.id != first.id
    assert [comment.id for comment in store.comments(SHA)] == [first.id, other.id]
    review = json.loads((tmp_path / "reviews" / f"{SHA}.json").read_text())
    # Older builds read the review file strictly, so the ids stay out of it.
    assert set(review) == {"schema", "media", "next_id", "comments"}
    assert all("client_id" not in item for item in review["comments"])
    assert store.add_comment(SHA, FACTS, body).id == "c3"


def test_store_refuses_bad_ids_and_a_note_deleted_after_it_was_saved(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    body = {"kind": "fix", "text": "Trim", "start_frame": 0}
    for bad in ("short", "x" * 65, "has space", "../etc/passwd1", 12345678):
        with pytest.raises(ValidationError):
            store.add_client_comment(SHA, FACTS, body, bad)
    comment, _ = store.add_client_comment(SHA, FACTS, body, "abcdefgh")
    store.delete_comment(SHA, comment.id)
    with pytest.raises(UnknownCommentError):
        store.add_client_comment(SHA, FACTS, body, "abcdefgh")
    assert store.comments(SHA) == []


def test_comment_api_dedupes_a_retried_client_id(server: ReviewServer) -> None:
    note = {"kind": "fix", "text": "Offline note", "start_frame": 10,
            "client_id": "0123456789abcdef"}
    status, _, first = request(server, "POST", "/api/comments", note)
    assert status == 201
    status, _, again = request(server, "POST", "/api/comments", note)
    assert status == 200 and json.loads(again) == json.loads(first)
    plain = {key: value for key, value in note.items() if key != "client_id"}
    assert request(server, "POST", "/api/comments", plain)[0] == 201
    review = json.loads(request(server, "GET", "/api/review")[2])
    assert [item["text"] for item in review["comments"]] == ["Offline note", "Offline note"]
    status, _, body = request(server, "POST", "/api/comments", {**note, "client_id": "bad id"})
    assert status == 400 and "client_id" in json.loads(body)["error"]["message"]
    assert request(server, "POST", "/api/comments", note, token="wrong")[0] == 403


@pytest.mark.parametrize("fixture", ["server", "multi_server"])
def test_worker_is_served_at_the_root_with_every_static_file(
    fixture: str, request_fixture: pytest.FixtureRequest,
) -> None:
    review = request_fixture.getfixturevalue(fixture)
    status, headers, body = request(review, "GET", "/sw.js", token=None)
    assert status == 200 and headers["Content-Type"].startswith("text/javascript")
    worker = body.decode()
    files = json.loads(re.search(r"const STATIC_FILES = (\[.*?\]);", worker).group(1))
    assert {"/static/app.mjs", "/static/offline.mjs", "/static/list.mjs",
            "/static/timeline.mjs", "/static/app.css"} <= set(files)
    for path in files:
        assert request(review, "GET", path, token=None)[0] == 200
    assert re.search(r'const VERSION = "[0-9a-f]{16}";', worker)
    assert request(review, "POST", "/sw.js", {}, token=None)[0] == 405
    assert request(review, "GET", "/static/sw.js", token=None)[0] == 404


@pytest.fixture
def request_fixture(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    return request


def test_pages_link_a_manifest_that_keeps_the_token(multi_server: ReviewServer) -> None:
    token = multi_server.list_token
    page = request(multi_server, "GET", f"/?t={token}", token=None)[2].decode()
    link = re.search(r'<link rel="manifest" href="([^"]+)">', page).group(1)
    assert link == f"/manifest.webmanifest?start={quote(f'/?t={token}', safe='')}"
    assert 'src="static/list.mjs"' in page
    status, headers, body = request(multi_server, "GET", link, token=None)
    manifest = json.loads(body)
    assert status == 200 and headers["Content-Type"] == "application/manifest+json"
    assert manifest["start_url"] == f"/?t={token}"
    assert (manifest["scope"], manifest["display"]) == ("/", "standalone")
    video = multi_server.videos[0].token
    page = request(multi_server, "GET", f"/v/1/?t={video}", token=None)[2].decode()
    assert quote(f"/v/1/?t={video}", safe="") in page
    for start in ("https://evil.example/", "//evil.example/", "/\\evil", "x" * 10):
        route = f"/manifest.webmanifest?start={quote(start, safe='')}"
        assert json.loads(request(multi_server, "GET", route, token=None)[2])["start_url"] == "/"
