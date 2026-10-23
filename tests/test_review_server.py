"""Loopback review server contract tests."""

from __future__ import annotations

import http.client
import io
import json
import os
import socket
import threading
from dataclasses import replace
from datetime import date
from email.message import Message
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from test_feedback_batch import _cut, _json, _layers
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_removed import _source_words
from test_review_rounds import round_project as _round_project

from yap_studio import ffmpeg, receipt
from yap_studio.errors import ValidationError
from yap_studio.feedback import release as release_module
from yap_studio.feedback.batch import FeedbackError, load_batch
from yap_studio.feedback.clarify import ask_clarification
from yap_studio.feedback.versions import LAYERS_SCHEMA, save_version
from yap_studio.project.lock import pid_alive
from yap_studio.project.model import CutWord, CutWords
from yap_studio.review import media as media_module
from yap_studio.review import send as send_module
from yap_studio.review import server as server_module
from yap_studio.review.export import format_text
from yap_studio.review.launch import version_video
from yap_studio.review.media import ReviewMedia, file_identity
from yap_studio.review.model import MediaFacts
from yap_studio.review.series import SeriesEntry
from yap_studio.review.server import ReviewSeries, ReviewServer, ReviewVersion
from yap_studio.review.store import ReviewStore, RoundConflictError
from yap_studio.watch.findings import default_findings_path, record_findings

SHA = "a" * 64
OLD_SHA = "b" * 64
VIDEO = b"0123456789abcdefghijklmnopqrstuvwxyz"
round_project = _round_project


@pytest.mark.parametrize("cfr", [True, False])
def test_pin_post_get_patch_and_next_version_thread(
    removed_server: ReviewServer, cfr: bool,
) -> None:
    server = removed_server
    if not cfr:
        server.series = replace(server.series, versions=tuple(
            replace(item, media=replace(item.media, cfr=False, fps=None))
            for item in server.series.versions
        ))
    timing = ({"start_frame": 10, "end_frame": 30} if cfr else
              {"start_s": 0.333333333, "end_s": 1})
    status, _, payload = _in_process_request(server, "POST", "/api/comments?version=1", {
        "text": "Move the card", **timing,
        "pin": {"x": 0.42134, "y": 0.31054, **({"frame": 15} if cfr else {"time_s": 0.5})},
    })
    assert status == 201
    pin = json.loads(payload)["pin"]
    assert pin == {"x": 0.4213, "y": 0.3105, "frame": 15 if cfr else None, "time_s": 0.5}
    for version, key in ((1, "comments"), (2, "threads")):
        data = json.loads(_in_process_request(server, "GET", f"/api/review?version={version}")[2])
        records = data[key] if key == "comments" else data["series"][key]
        assert records[0]["pin"] == pin
    route = "/api/comments/c1?version=1"
    stranded = {"end_frame": 15} if cfr else {"end_s": 0.4}
    assert _in_process_request(server, "PATCH", route, stranded)[0] == 400
    assert json.loads(_in_process_request(server, "PATCH", route, {"kind": "keep"})[2])[
        "pin"] == pin
    moved = {"x": 0.8, "y": 0.2, **({"frame": 20} if cfr else {"time_s": 0.7})}
    assert json.loads(_in_process_request(server, "PATCH", route, {"pin": moved})[2])[
        "pin"]["x"] == 0.8
    assert "pin" not in json.loads(_in_process_request(server, "PATCH", route, {"pin": None})[2])


@pytest.fixture(scope="session")
def socket_access() -> None:
    """Skip live fixtures only when the execution profile refuses loopback binding."""
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("sandbox blocked loopback socket binding")


def test_regression_second_server_preserves_live_owner_and_recovers_dead_owner(
    round_project: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(round_project)], store, lambda _: [])
    queued = []
    first = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                         series=video.series, send_runner=queued.append)
    assert _in_process_request(first, "POST", "/api/comments", {
        "text": "Fix the caption", "target": "caption", "start_frame": 18,
    })[0] == 201
    assert _in_process_request(first, "POST", "/api/rounds", {})[0] == 202
    second = ReviewServer(video.media, video.sha256, None, ReviewStore(store.root),
                          token="test-token", series=video.series)
    live = store.rounds(video.series.id)[0]
    assert live.status == "sending"
    assert live.owner == first.owner and first.owner["instance"] != second.owner["instance"]
    with pytest.raises(RoundConflictError, match="owner"):
        store.update_round(video.series.id, live.id, status="sent", batch="b1",
                           instance=second.owner["instance"])
    queued[0]()
    assert store.rounds(video.series.id)[0].status == "sent"
    assert _in_process_request(first, "POST", "/api/comments", {
        "text": "Another caption", "target": "caption", "start_frame": 21,
    })[0] == 201
    assert _in_process_request(first, "POST", "/api/rounds", {})[0] == 202
    dead_pid = 1_000_000
    while pid_alive(dead_pid):
        dead_pid += 1
    file = store.root / "rounds" / f"{video.series.id}.json"
    data = json.loads(file.read_text())
    data["rounds"][-1]["owner"]["pid"] = dead_pid
    _json(file, data)
    ReviewServer(video.media, video.sha256, None, store, series=video.series)
    dead = store.rounds(video.series.id)[-1]
    assert dead.status == "failed" and dead.error["code"] == "interrupted"
    assert live.owner["pid"] == os.getpid()


@pytest.fixture
def removed_server(feedback_project: Path, tmp_path: Path, monkeypatch, request) -> ReviewServer:
    root = feedback_project
    words_path = _source_words(root)
    words = json.loads(words_path.read_text())
    if getattr(request, "param", None) == "long":
        words["words"][0]["text"] = "x" * 700
    for word in words["words"]:
        word["start"] += 1
        word["end"] += 1
    _json(words_path, words)
    for version in (1, 2):
        if version == 2:
            _cut(root, label="next removed render")
        _json(root / "removals.json", {"schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 2, "end": 3,
             "kind": "filler", "status": "applied"},
        ] if version == 1 else []})
        receipt_path = root / "cut" / "receipt.json"
        data = json.loads(receipt_path.read_text())
        data["inputs"]["words:t1"] = receipt.sha256_file(words_path)
        data["plan"]["pieces"][1].update(
            segment_id="s1" if version == 1 else "s2",
            src_t0="3" if version == 1 else "4", src_t1="5" if version == 1 else "6",
        )
        _json(receipt_path, data)
        save_version(root)
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(root)], store, lambda _: [])
    return ReviewServer(video.media, video.sha256, None, store, token="test-token",
                        series=video.series, send_runner=lambda work: work())


@pytest.mark.parametrize("via", ["tailscale", "wifi"])
@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("version", [1, 2, None])
def test_phone_link_preserves_selected_version(
    removed_server: ReviewServer, via: str, multiple: bool, version: int | None,
) -> None:
    server = removed_server
    server.phone = {"via": via, "url": "http://synthetic.test/?t=list-only-token"}
    prefix, token = "/", server.token
    if multiple:
        token = "second-only-token"
        server.videos = (server.videos[0], replace(server.videos[0], token=token))
        prefix = "/v/2/"
    query = f"?version={version}" if version is not None else ""
    code, _, body = _in_process_request(server, "GET", prefix + "api/review" + query,
                                       token=token)
    assert code == 200
    original = json.loads(body)
    expected = version if version is not None else 2
    assert original["series"]["current"] == expected
    assert original["phone"]["via"] == via
    phone_url = urlsplit(original["phone"]["url"])
    assert phone_url.path == prefix
    assert parse_qs(phone_url.query) == {"t": [token], "version": [str(expected)]}
    code, _, body = _in_process_request(
        server, "GET", phone_url.path + "api/review?" + phone_url.query, token=None,
    )
    assert code == 200
    opened = json.loads(body)
    assert opened["series"]["current"] == expected
    assert opened["media"]["sha256"] == original["media"]["sha256"]


def test_removed_api_uses_each_saved_version(removed_server: ReviewServer) -> None:
    for number, kind, text in ((1, "filler", "um well"), (2, "story", "um well kept")):
        status, _, payload = _in_process_request(
            removed_server, "GET", f"/api/removed?version={number}",
        )
        assert status == 200
        data = json.loads(payload)
        assert data["available"] and data["version"] == number
        assert len(data["ranges"]) == 1
        item = data["ranges"][0]
        assert item["kind"] == kind and item["text"] == text and item["words_known"]
        assert item["at_frame"] == 60 and item["at_s"] == 2.0
    assert _in_process_request(removed_server, "GET", "/api/removed?version=3")[0] == 400
    assert _in_process_request(removed_server, "GET", "/api/removed", token=None)[0] == 403
    assert _in_process_request(removed_server, "POST", "/api/removed", {})[0] == 405


def test_project_series_title_is_the_folder_name(removed_server: ReviewServer) -> None:
    root = removed_server.series.project_dir
    project = json.loads((root / "project.json").read_text())
    assert project["name"] != root.name
    code, _, body = _in_process_request(removed_server, "GET", "/api/review")
    assert code == 200 and json.loads(body)["series"]["title"] == root.name


def test_review_versions_include_saved_notes(removed_server):
    root = removed_server.series.project_dir
    manifest = root / "versions/v2/manifest.json"
    document = json.loads(manifest.read_text())
    document["note"] = "The story is ready for your review."
    _json(manifest, document)
    code, _, body = _in_process_request(removed_server, "GET", "/api/review")
    assert code == 200
    assert [v["note"] for v in json.loads(body)["series"]["versions"]] == [None, document["note"]]


@pytest.fixture
def final_review_server(feedback_project, tmp_path, monkeypatch, request):
    from yap_studio import background

    (tmp_path / "inbox").mkdir()
    project = tmp_path / "projects/demo"
    project.parent.mkdir()
    feedback_project.rename(project)
    save_version(project)
    if getattr(request, "param", None) == "final":
        from test_feedback_versions import _render_media

        final = project / "final/final.mp4"
        final.parent.mkdir()
        _render_media(final, 120, label="synthetic final")
        save_version(project, media=final)
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(project)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series)
    record = tmp_path / "home.json"
    background.write_json(record, {"pid": 123, "workspace": str(tmp_path),
        "document": {"url": "http://127.0.0.1:42/?t=test", "phone": None}})
    monkeypatch.setattr(background, "records", lambda kind: [record] if kind == "home" else [])
    monkeypatch.setattr(background, "command_matches", lambda *args: True)
    return server


def test_request_final_from_review_and_deliver_to_agent(final_review_server, tmp_path):
    from yap_studio.homepage import bridge

    server = final_review_server
    code, _, body = _in_process_request(server, "POST", "/api/final", {"version": 1})
    assert code == 200
    row = json.loads(body)["request"]
    assert row["kind"] == "final" and row["payload"] == {"project": "projects/demo"}
    assert _in_process_request(server, "POST", "/api/final", {"version": 1})[0] == 409
    assert json.loads(_in_process_request(server, "GET", "/api/review")[2])[
        "home"]["final_requested"] is True
    assert bridge.wait(tmp_path, timeout=1)["request"]["id"] == row["id"]


@pytest.mark.parametrize("body", [{}, {"version": True}, {"version": "1"},
                                  {"version": 1, "extra": True}, {"version": 99}])
def test_request_final_validates_review_version(final_review_server, body):
    assert _in_process_request(final_review_server, "POST", "/api/final", body)[0] == 400


def test_request_final_review_security_and_missing_workspace(final_review_server, monkeypatch):
    from yap_studio import background

    server = final_review_server
    assert _in_process_request(server, "POST", "/api/final", {"version": 1}, token=None)[0] == 403
    assert _in_process_request(server, "POST", "/api/final?t=test-token", {
        "version": 1}, token=None)[0] == 403
    assert _in_process_request(server, "GET", "/api/final")[0] == 405
    monkeypatch.setattr(background, "records", lambda kind: [])
    code, _, body = _in_process_request(server, "POST", "/api/final", {"version": 1})
    assert code == 409 and json.loads(body)["error"]["message"]


def test_request_final_refuses_newer_version_not_loaded_in_review(final_review_server):
    server = final_review_server
    root = server.series.project_dir
    _cut(root, label="newer synthetic story")
    save_version(root)
    assert _in_process_request(server, "POST", "/api/final", {"version": 1})[0] == 409


@pytest.mark.parametrize("final_review_server", ["final"], indirect=True)
def test_review_version_final_flags_and_story_cut_requirement(final_review_server):
    server = final_review_server
    code, _, body = _in_process_request(server, "GET", "/api/review")
    assert code == 200
    assert [v["final"] for v in json.loads(body)["series"]["versions"]] == [False, True]
    for number in (1, 2):
        assert _in_process_request(server, "POST", "/api/final", {"version": number})[0] == 409


def test_request_final_requires_project_series(removed_server):
    removed_server.series = replace(removed_server.series, project_dir=None)
    assert _in_process_request(removed_server, "POST", "/api/final", {"version": 2})[0] == 409


def test_removed_unavailable_for_plain_files_and_plain_series(removed_server: ReviewServer) -> None:
    for series in (None, replace(removed_server.series, project_dir=None)):
        plain = ReviewServer(removed_server.media, removed_server.sha256, None,
                             removed_server.store, token="test-token", series=series)
        for route in ("/api/removed", "/api/removed?version=1"):
            status, _, payload = _in_process_request(plain, "GET", route)
            assert status == 200 and json.loads(payload) == {"available": False, "ranges": []}
        assert _in_process_request(plain, "POST", "/api/restore?version=1",
                                   {"range": "s1.0"})[0] == 409


@pytest.mark.parametrize("cfr", [True, False])
def test_restore_creates_editable_draft_at_join_and_sends_request(
    removed_server: ReviewServer, cfr: bool,
) -> None:
    server = removed_server
    if not cfr:
        server.series = replace(server.series, versions=tuple(
            replace(item, media=replace(item.media, cfr=False, fps=None))
            for item in server.series.versions
        ))
    request = {"type": "restore", "range": "s1.0", "take": "t1",
               "start": 2.0, "end": 4.0, "text": "um well kept"}
    status, _, payload = _in_process_request(server, "POST", "/api/restore?version=2",
                                            {"range": "s1.0"})
    assert status == 201
    comment = json.loads(payload)
    assert comment["kind"] == "fix" and comment["target"] == "story"
    assert comment["start_frame"] == (60 if cfr else None) and comment["start_s"] == 2.0
    assert comment["end_frame"] is None and comment["end_s"] is None
    assert comment["text"] == 'Bring back: "um well kept"' and comment["request"] == request
    document = json.loads(_in_process_request(server, "GET", "/api/review?version=2")[2])
    assert document["comments"][0]["send"]["state"] == "draft"
    assert json.loads(_in_process_request(server, "GET", "/api/review?version=1")[2])[
        "comments"] == []
    status, _, payload = _in_process_request(server, "PATCH", "/api/comments/c1?version=2",
                                            {"text": "Keep this line in the story"})
    assert status == 200 and json.loads(payload)["request"] == request
    status, _, payload = _in_process_request(server, "GET", "/api/export.json?version=2")
    assert status == 200 and json.loads(payload)["comments"][0]["request"] == request
    assert b'(restore: t1 0:02.0-0:04.0)' in _in_process_request(
        server, "GET", "/api/export.txt?version=2",
    )[2]
    project = server.series.project_dir
    edit_before = (project / "edit.json").read_bytes()
    removals_before = (project / "removals.json").read_bytes()
    assert _in_process_request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    round = json.loads(_in_process_request(server, "GET", "/api/review?version=2")[2])[
        "series"]["rounds"][0]
    assert round["status"] == "sent", round["error"]
    item = load_batch(project, round["batch"])["items"][0]
    assert item["request"] == item["comment"]["request"] == request
    assert item["disposition"] == "open" and item["operations"] == []
    assert (project / "edit.json").read_bytes() == edit_before
    assert (project / "removals.json").read_bytes() == removals_before


def test_restore_unknown_range_bad_body_guards_and_clear(removed_server: ReviewServer) -> None:
    server = removed_server
    assert _in_process_request(server, "POST", "/api/restore?version=1",
                               {"range": "missing"})[0] == 404
    for body in ({}, {"range": 1}, {"range": "s1.0", "text": "injected"}):
        assert _in_process_request(server, "POST", "/api/restore", body)[0] == 400
    assert _in_process_request(server, "GET", "/api/restore")[0] == 405
    assert _in_process_request(server, "POST", "/api/restore", {"range": "s1.0"},
                               token=None)[0] == 403
    assert _in_process_request(server, "POST", "/api/restore", {"range": "s1.0"},
                               host="evil.example")[0] == 403
    assert server.store.comments(server.sha256) == []
    assert _in_process_request(server, "POST", "/api/restore", {"range": "s1.0"})[0] == 201
    status, _, payload = _in_process_request(server, "PATCH", "/api/comments/c1", {"request": None})
    assert status == 200 and json.loads(payload)["request"] is None
    server.series.versions[-1].media.path.write_bytes(b"changed render")
    assert _in_process_request(server, "POST", "/api/restore", {"range": "s1.0"})[0] == 409


@pytest.fixture
def cut_server(removed_server: ReviewServer) -> ReviewServer:
    words = (
        CutWord("t1:0", "s1", "before", 0.7, 0.9),
        CutWord("t1:1", "s1", "so", 0.9, 1.1),
        CutWord("t1:2", "s1", "basically", 1.2, 1.8),
        CutWord("t1:1", "s2", "yeah", 1.9, 2.1),
    )
    removed_server.series = replace(removed_server.series, versions=tuple(
        replace(item, words=CutWords("output", {}, words if item.number == 2 else words[:1]))
        for item in removed_server.series.versions
    ))
    return removed_server


def test_words_api_uses_selected_map_and_plain_review_fallback(cut_server: ReviewServer) -> None:
    server = cut_server
    for version, count in ((1, 1), (2, 4)):
        status, _, payload = _in_process_request(server, "GET", f"/api/words?version={version}")
        data = json.loads(payload)
        assert status == 200 and data["available"] and data["version"] == version
        assert data["words"] == [
            {key: getattr(word, key) for key in ("id", "segment", "text", "start", "end")}
            for word in server.series.versions[version - 1].words.words
        ]
        assert len(data["words"]) == count
    assert _in_process_request(server, "GET", "/api/words", token=None)[0] == 403
    assert _in_process_request(server, "POST", "/api/words", {})[0] == 405
    plain = ReviewServer(server.media, server.sha256, None, server.store, token="test-token")
    for route in ("/api/words", "/api/words?version=1"):
        status, _, payload = _in_process_request(plain, "GET", route)
        assert status == 200 and json.loads(payload) == {
            "available": False, "version": None, "words": [],
        }
    server.series = replace(server.series, versions=tuple(
        replace(item, words=None) for item in server.series.versions
    ))
    assert json.loads(_in_process_request(server, "GET", "/api/words?version=1")[2]) == {
        "available": False, "version": 1, "words": [],
    }


@pytest.mark.parametrize("cfr", [True, False])
@pytest.mark.parametrize("note", [None, "  Keep the pace.  "])
def test_cut_creates_one_story_note_with_half_open_words_and_survives_reload(
    cut_server: ReviewServer, cfr: bool, note: str | None,
) -> None:
    server = cut_server
    if not cfr:
        server.series = replace(server.series, versions=tuple(
            replace(item, media=replace(item.media, cfr=False, fps=None))
            for item in server.series.versions
        ))
    body = {"start_frame": 30, "end_frame": 60} if cfr else {"start_s": 1, "end_s": 2}
    if note is not None:
        body["text"] = note
    status, _, payload = _in_process_request(server, "POST", "/api/cut?version=2", body)
    assert status == 201
    comment = json.loads(payload)
    expected = {"type": "cut", "start": 1.0, "end": 2.0, "words": [
        {"id": "t1:1", "segment": "s1", "text": "so"},
        {"id": "t1:2", "segment": "s1", "text": "basically"},
    ]}
    assert comment["kind"] == "fix" and comment["target"] == "story"
    assert comment["request"] == expected
    assert comment["text"] == (note.strip() if note else "Cut this.")
    assert (comment["start_s"], comment["end_s"]) == (1, 2)
    assert (comment["start_frame"], comment["end_frame"]) == ((30, 60) if cfr else (None, None))
    server.store = ReviewStore(server.store.root)
    data = json.loads(_in_process_request(server, "GET", "/api/review?version=2")[2])
    assert len(data["comments"]) == 1 and data["comments"][0]["request"] == expected
    assert json.loads(_in_process_request(server, "GET", "/api/review?version=1")[2])[
        "comments"] == []
    exported = json.loads(_in_process_request(server, "GET", "/api/export.json?version=2")[2])
    assert exported["comments"][0]["request"] == expected
    assert b' (cut: "so basically")' in _in_process_request(
        server, "GET", "/api/export.txt?version=2",
    )[2]
    status, _, payload = _in_process_request(server, "PATCH", "/api/comments/c1?version=2",
                                            {"text": "A shorter note"})
    assert status == 200 and json.loads(payload)["request"] == expected
    edge = {"end_frame": 61} if cfr else {"end_s": 2.1}
    assert _in_process_request(server, "PATCH", "/api/comments/c1?version=2", edge)[0] == 400
    for change in ({"kind": "keep"}, {"target": "caption"}):
        assert _in_process_request(server, "PATCH", "/api/comments/c1?version=2", change)[0] == 400
    assert _in_process_request(server, "PATCH", "/api/comments/c1?version=2",
                               {"request": None})[0] == 200


def test_cut_uses_exact_fractional_frame_edges(cut_server: ReviewServer) -> None:
    server = cut_server
    words = (CutWord("t1:0", "s1", "before", 0.0333, 0.0334),
             CutWord("t1:1", "s1", "inside", 0.03336, 0.03338),
             CutWord("t1:2", "s1", "after", 0.13347, 0.13349))
    server.series = replace(server.series, versions=tuple(
        replace(item, media=replace(item.media, fps=Fraction(30000, 1001)),
                words=CutWords("output", {}, words)) for item in server.series.versions
    ))
    status, _, payload = _in_process_request(server, "POST", "/api/cut",
                                            {"start_frame": 1, "end_frame": 4})
    comment = json.loads(payload)
    assert status == 201
    assert comment["request"] == {"type": "cut", "start": 0.0334, "end": 0.1335,
                                   "words": [{"id": word.id, "segment": word.segment,
                                              "text": word.text} for word in words[1:2]]}


@pytest.mark.parametrize("body", [
    {}, {"start_frame": 30}, {"end_frame": 60}, {"start_frame": 30, "end_frame": 60, "extra": 1},
    {"start_frame": True, "end_frame": 60}, {"start_frame": "30", "end_frame": 60},
    {"start_frame": 30.5, "end_frame": 60}, {"start_frame": 30, "end_frame": "60"},
    {"start_frame": 30, "end_frame": 60, "text": None},
    {"start_s": 1, "end_s": 2}, [],
])
def test_cut_rejects_bad_body(cut_server: ReviewServer, body: object) -> None:
    assert _in_process_request(cut_server, "POST", "/api/cut", body)[0] == 400
    assert cut_server.store.comments(cut_server.sha256) == []


@pytest.mark.parametrize("body,message", [
    ({"start_frame": 30, "end_frame": None}, "Cut this needs a range with a start and an end"),
    ({"start_frame": 30, "end_frame": 31}, "Cut this needs a range with a start and an end"),
    ({"start_frame": 90, "end_frame": 120}, "No spoken words in this range"),
])
def test_cut_rejects_points_and_empty_ranges(
    cut_server: ReviewServer, body: dict, message: str,
) -> None:
    status, _, payload = _in_process_request(cut_server, "POST", "/api/cut", body)
    assert status == 400 and message in json.loads(payload)["error"]["message"]


def test_cut_limits_word_count_and_uses_default_for_blank_note(cut_server: ReviewServer) -> None:
    server = cut_server
    for count, expected_status in ((201, 400), (200, 201)):
        server.series = replace(server.series, versions=tuple(
            replace(item, words=CutWords("output", {}, tuple(
                CutWord(f"t1:{index}", "s1", "word", 1.1, 1.2) for index in range(count)
            ))) for item in server.series.versions
        ))
        status, _, payload = _in_process_request(server, "POST", "/api/cut",
                                                {"start_frame": 30, "end_frame": 60, "text": "  "})
        assert status == expected_status
        data = json.loads(payload)
        if count == 201:
            assert "Select at most 200 words to cut in one note" in data["error"]["message"]
        else:
            assert data["text"] == "Cut this." and len(data["request"]["words"]) == 200


def test_cut_requires_header_token_words_post_and_unchanged_media(cut_server: ReviewServer) -> None:
    server = cut_server
    body = {"start_frame": 30, "end_frame": 60}
    assert _in_process_request(server, "POST", "/api/cut?t=test-token", body, token=None)[0] == 403
    assert _in_process_request(server, "GET", "/api/cut")[0] == 405
    assert _in_process_request(server, "POST", "/api/cut", body, host="evil.example")[0] == 403
    plain = ReviewServer(server.media, server.sha256, None, server.store, token="test-token")
    status, _, payload = _in_process_request(plain, "POST", "/api/cut?version=1", body)
    error = json.loads(payload)["error"]
    assert status == 409 and error["code"] == "no_words"
    assert error["message"] == ("This version has no word list. Open the project folder "
                                "with --versions to cut by words.")
    server.series.versions[-1].media.path.write_bytes(b"changed render")
    status, _, payload = _in_process_request(server, "POST", "/api/cut", body)
    assert status == 409 and json.loads(payload)["error"]["code"] == "media_changed"


@pytest.mark.parametrize("body", [
    {"start_s": True, "end_s": 2}, {"start_s": 1, "end_s": "2"},
    {"start_s": -1, "end_s": 2}, {"start_s": 1}, {"end_s": 2},
    {"start_s": 1, "end_s": 2, "request": {}},
])
def test_cut_rejects_bad_seconds(cut_server: ReviewServer, body: dict) -> None:
    server = cut_server
    server.series = replace(server.series, versions=tuple(
        replace(item, media=replace(item.media, cfr=False, fps=None))
        for item in server.series.versions
    ))
    assert _in_process_request(server, "POST", "/api/cut", body)[0] == 400
    assert server.store.comments(server.sha256) == []


def test_cut_live_http_selects_words_saves_and_reloads(series_server: ReviewServer) -> None:
    server = series_server
    status, _, payload = request(server, "GET", "/api/words?version=1&t=test-token", token=None)
    assert status == 200 and [word["text"] for word in json.loads(payload)["words"]] == [
        "Removed", "Kept",
    ]
    status, _, payload = request(server, "POST", "/api/cut?version=1",
                                 {"start_frame": 12, "end_frame": 45})
    comment = json.loads(payload)
    assert status == 201 and comment["text"] == "Cut this."
    assert comment["request"] == {"type": "cut", "start": 0.4, "end": 1.5, "words": [
        {"id": "t1:0", "segment": "s1", "text": "Removed"},
        {"id": "t1:1", "segment": "s1", "text": "Kept"},
    ]}
    assert json.loads(request(server, "GET", "/api/review?version=1")[2])["comments"][0][
        "request"] == comment["request"]
    assert b' (cut: "Removed Kept")' in request(server, "GET", "/api/export.txt?version=1")[2]


def test_restore_uses_unknown_audio_fallback(removed_server: ReviewServer) -> None:
    server = removed_server
    words = server.series.project_dir / "ingest" / "t1" / "words.json"
    words.write_text("changed transcript", encoding="utf-8")
    data = json.loads(_in_process_request(server, "GET", "/api/removed?version=1")[2])
    assert not data["ranges"][0]["words_known"] and data["ranges"][0]["text"] is None
    status, _, payload = _in_process_request(server, "POST", "/api/restore?version=1",
                                            {"range": "s1.0"})
    assert status == 201
    comment = json.loads(payload)
    assert comment["text"] == "Bring back the removed part at 0:02.0"
    assert comment["request"]["text"] is None


@pytest.mark.parametrize("removed_server", ["long"], indirect=True)
def test_restore_caps_the_request_quote(removed_server: ReviewServer) -> None:
    server = removed_server
    data = json.loads(_in_process_request(server, "GET", "/api/removed?version=1")[2])
    assert len(data["ranges"][0]["text"]) > 500
    status, _, payload = _in_process_request(server, "POST", "/api/restore?version=1",
                                            {"range": "s1.0"})
    assert status == 201
    comment = json.loads(payload)
    assert comment["request"]["text"] == "x" * 500
    assert comment["text"] == f'Bring back: "{"x" * 500}"'


def test_spelling_http_create_patch_export_send_and_lock(
    round_project: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(round_project)], store, lambda _: []).series
    version = video.versions[-1]
    server = ReviewServer(version.media, version.sha256, None, store, token="test-token",
                          series=video, send_runner=lambda work: work())
    request = {"type": "spelling", "from": "tail net", "to": "tailnet"}
    status, _, payload = _in_process_request(server, "POST", "/api/comments", {
        "kind": "fix", "target": "caption", "text": "Correct spelling",
        "start_frame": 18, "request": request,
    })
    assert status == 201 and json.loads(payload)["request"] == request
    status, _, _ = _in_process_request(server, "PATCH", "/api/comments/c1", {"target": "story"})
    assert status == 400
    status, _, payload = _in_process_request(server, "PATCH", "/api/comments/c1", {"request": None})
    assert status == 200 and json.loads(payload)["request"] is None
    status, _, payload = _in_process_request(
        server, "PATCH", "/api/comments/c1", {"request": request},
    )
    assert status == 200 and json.loads(payload)["request"] == request
    status, _, payload = _in_process_request(server, "GET", "/api/export.json")
    assert status == 200 and json.loads(payload)["comments"][0]["request"] == request
    status, _, payload = _in_process_request(server, "GET", "/api/export.txt")
    assert status == 200 and b'(spelling: "tail net" -> "tailnet")' in payload
    assert _in_process_request(server, "POST", "/api/rounds", {})[0] == 202
    status, _, payload = _in_process_request(server, "GET", "/api/review")
    assert status == 200 and json.loads(payload)["series"]["rounds"][0]["status"] == "sent"
    assert _in_process_request(server, "PATCH", "/api/comments/c1", {"request": None})[0] == 409


def test_spelling_http_retry_serializes_and_reuses_written_batch(
    round_project: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(round_project)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series, send_runner=lambda work: work())
    apply = send_module.apply_word_fixes
    calls = []

    def flaky(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("Spelling temporarily unavailable.")
        return apply(*args, **kwargs)

    monkeypatch.setattr(send_module, "apply_word_fixes", flaky)
    assert _in_process_request(server, "POST", "/api/comments", {
        "target": "caption", "text": "Correct spelling", "start_frame": 18,
        "request": {"type": "spelling", "from": "tail net", "to": "tailnet"},
    })[0] == 201
    assert _in_process_request(server, "POST", "/api/rounds", {})[0] == 202
    failed = json.loads(_in_process_request(server, "GET", "/api/review")[2])["series"]["rounds"][0]
    assert failed["status"] == "failed" and failed["batch"] == "b1"
    status, _, body = _in_process_request(server, "POST", "/api/rounds/n1/retry", {})
    assert status == 202 and json.loads(body)["batch"] == "b1"
    sent = json.loads(_in_process_request(server, "GET", "/api/review")[2])["series"]["rounds"][0]
    assert sent["status"] == "sent" and sent["batch"] == "b1"
    assert not (round_project / "feedback" / "b2.json").exists()


def _in_process_request(
    server: ReviewServer, method: str, route: str, body: object = None, *,
    token: str | None = "test-token", host: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Exercise the real HTTP dispatcher with byte streams when sockets are unavailable."""
    handler_type = server._handler_type()
    handler = handler_type.__new__(handler_type)
    handler.command = method
    handler.path = route
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {route} HTTP/1.1"
    handler.headers = Message()
    supplied = dict(headers or {})
    supplied["Host"] = host if host is not None else f"127.0.0.1:{server.port}"
    if token is not None:
        supplied["X-Yap-Token"] = token
    payload = b"" if body is None else json.dumps(body).encode()
    if body is not None and not any(key.lower() == "content-length" for key in supplied):
        supplied["Content-Length"] = str(len(payload))
    for key, value in supplied.items():
        handler.headers[key] = value
    handler.rfile = io.BytesIO(payload)
    handler.wfile = io.BytesIO()
    handler._dispatch()
    response = handler.wfile.getvalue()
    head, _, content = response.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    status = int(lines[0].split()[1])
    fields = dict(line.split(": ", 1) for line in lines[1:])
    return status, fields, content


@pytest.mark.parametrize("save_layers", [True, False])
def test_server_text_exports_use_each_saved_versions_layer_windows(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_layers: bool,
) -> None:
    root = feedback_project
    layer_path = root / "windows.json"
    for number in (1, 2):
        if number == 2:
            _cut(root, label="next layer render")
        kinds = ("insert", "overlay") if number == 1 else ("overlay", "insert")
        _json(layer_path, {"schema": LAYERS_SCHEMA, "time_base": "output", "layers": [
            {"id": f"broll-{kind}", "kind": kind, "start": index, "end": index + 1,
             "text": None} for index, kind in enumerate(kinds)
        ]})
        save_version(root, layers=layer_path if save_layers else None)
    layer_path.unlink()
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(root)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series)
    for version in video.series.versions:
        for frame in (6, 36):
            facts = version.media.facts(version.sha256)
            store.add_comment(version.sha256, facts, {
                "kind": "fix", "target": "element", "start_frame": frame,
                "text": f"Version {version.number} at {frame}",
            })
    expected = ("B-roll", "overlay") if save_layers else ("graphic", "graphic")
    for number in (1, 2):
        status, _, payload = _in_process_request(server, "GET", f"/api/export.txt?version={number}")
        assert status == 200
        text = payload.decode()
        labels = expected if number == 1 else expected[::-1]
        assert f"[Fix: {labels[0]}] Version {number} at 6" in text
        assert f"[Fix: {labels[1]}] Version {number} at 36" in text
        if number == 2:
            assert "Earlier versions: 2 notes, 2 still open" in text
            assert "Version 1 at 6" not in text
            _, _, included = _in_process_request(
                server, "GET", "/api/export.txt?version=2&earlier=1",
            )
            assert f"[Fix: {expected[0]}] Version 1 at 6" in included.decode()
            assert f"[Fix: {expected[1]}] Version 1 at 36" in included.decode()
    old, current = video.series.versions
    plain = ReviewServer(current.media, current.sha256, old.sha256, store, token="test-token")
    status, _, payload = _in_process_request(plain, "GET", "/api/export.txt?version=previous")
    assert status == 200
    assert f"[Fix: {expected[0]}] Version 1 at 6" in payload.decode()
    status, _, payload = _in_process_request(plain, "GET", "/api/review")
    assert status == 200
    previous = json.loads(payload)["previous"]["text"]
    assert f"[Fix: {expected[1]}] Version 1 at 36" in previous


@pytest.fixture(params=["in_process", "live"])
def release_server(feedback_project: Path, tmp_path: Path, monkeypatch, request):
    if request.param == "live":
        request.getfixturevalue("socket_access")
    root = feedback_project
    for number, status in ((1, "PASS"), (2, "WARN")):
        if number == 2:
            _cut(root, label="second release render")
        _json(root / "cut" / "verify.json", {"status": status, "findings": []})
        save_version(root)
    if request.param == "in_process":
        monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer", lambda *_args:
                            SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(root)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series, send_runner=lambda work: work())
    call = _in_process_request if request.param == "in_process" else globals()["request"]
    thread = None
    if request.param == "live":
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    try:
        yield server, call
    finally:
        if thread is not None:
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()


def test_regression_download_link_tokens_and_shared_method_policy(release_server, monkeypatch):
    server, http = release_server
    assert http(server, "POST", "/api/approve", {"version": 1})[0] == 200
    for route in ("/api/download", "/api/download/report"):
        assert http(server, "GET", route + "?version=1&t=test-token", token=None)[0] == 200
        assert http(server, "GET", route + "?version=1&t=wrong", token=None)[0] == 403
        assert http(server, "GET", route + "?version=1", token=None)[0] == 403
    assert http(server, "HEAD", "/api/download?version=1&t=test-token", token=None)[0] == 200
    for route in ("/api/approve", "/api/rollback"):
        assert http(server, "POST", route + "?t=test-token", {"version": 1}, token=None)[0] == 403
    # Every route must consult the documented method policy for link tokens.
    monkeypatch.setattr(server_module, "accepts_query_token", lambda _: False, raising=False)
    assert http(server, "GET", "/api/download?version=1&t=test-token", token=None)[0] == 403
    assert http(server, "HEAD", "/api/download?version=1&t=test-token", token=None)[0] == 403
    assert http(server, "GET", "/api/download/report?version=1&t=test-token", token=None)[0] == 403
    assert http(server, "GET", "/api/review?t=test-token", token=None)[0] == 403
    assert http(server, "GET", "/media?t=test-token", token=None)[0] == 403


def test_regression_empty_qa_report_refuses_approval_and_download(
    feedback_project: Path, tmp_path: Path, monkeypatch,
) -> None:
    _json(feedback_project / "qa" / "report.json", {})
    _json(feedback_project / "cut" / "verify.json", {"status": "pass"})
    save_version(feedback_project)
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(feedback_project)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series)
    status, _, body = _in_process_request(server, "POST", "/api/approve", {"version": 1})
    assert status == 409
    assert json.loads(body)["error"]["message"] == (
        "This version's check report has no valid status; run the checks before approving."
    )
    for route in ("/api/download", "/api/download/report"):
        assert _in_process_request(server, "GET", route + "?t=test-token", token=None)[0] == 403


def test_rollback_approve_and_download_issue_acceptance(release_server) -> None:
    server, http = release_server
    assert http(server, "GET", "/api/download?version=1")[0] == 403
    status, _, payload = http(server, "POST", "/api/rollback", {"version": 1})
    assert status == 200 and json.loads(payload)["version"] == "v1"
    status, _, payload = http(server, "POST", "/api/approve", {"version": 1})
    assert status == 200 and json.loads(payload)["report_status"] == "pass"
    first = server.series.versions[0]
    assert first.media.path.read_bytes() != server.series.versions[1].media.path.read_bytes()
    status, headers, payload = http(server, "GET", "/api/download?version=1")
    assert status == 200 and payload == first.media.path.read_bytes()
    assert headers["Content-Disposition"] == 'attachment; filename="sample-video-v1.mp4"'
    status, headers, payload = http(server, "GET", "/api/download?version=1",
                                   headers={"Range": "bytes=2-15"})
    assert status == 206 and payload == first.media.path.read_bytes()[2:16]
    assert headers["Content-Range"] == f"bytes 2-15/{first.media.path.stat().st_size}"
    assert headers["Content-Disposition"] == 'attachment; filename="sample-video-v1.mp4"'
    assert http(server, "GET", "/api/download?version=1", headers={"Range": "bytes=-4"})[
        2] == first.media.path.read_bytes()[-4:]
    assert http(server, "HEAD", "/api/download?version=1")[2] == b""
    assert http(server, "GET", "/api/download?version=1", headers={"Range": "bytes=9-2"})[
        0] == 416
    status, headers, payload = http(server, "GET", "/api/download/report?version=1")
    assert status == 200 and json.loads(payload) == {
        "status": "pass", "subject": "render", "qa": None,
        "statuses": {"verify": "pass"},
        "verify": {"status": "PASS", "findings": []},
    }
    assert headers["Content-Disposition"] == 'attachment; filename="sample-video-v1-checks.json"'
    for route in ("/api/download?version=2", "/api/download/report?version=2"):
        status, _, payload = http(server, "GET", route)
        assert status == 403 and json.loads(payload)["error"] == {
            "type": "not_approved", "code": "not_approved",
            "message": "Approve this version to download it.",
        }
    series = json.loads(http(server, "GET", "/api/review")[2])["series"]
    assert series["newest"] == 2 and series["current_version"] == 1
    assert series["approved"]["version"] == 1 and series["approved"]["report_status"] == "pass"
    assert [item["report_status"] for item in series["versions"]] == ["pass", "warn"]
    assert http(server, "POST", "/api/approve", {"version": 2})[0] == 200
    assert http(server, "GET", "/api/download?version=1")[0] == 403
    assert http(server, "GET", "/api/download?version=2")[0] == 200
    assert http(server, "DELETE", "/api/approve")[0] == 204
    assert http(server, "GET", "/api/download?version=2")[0] == 403
    assert json.loads(http(server, "GET", "/api/review")[2])["series"]["approved"] is None


def test_review_polls_and_downloads_do_not_hash_version_media(release_server, monkeypatch) -> None:
    server, http = release_server
    assert http(server, "POST", "/api/approve", {"version": 1})[0] == 200
    media_paths = {item.media.path for item in server.series.versions}
    media_hash_calls = []
    original = release_module.sha256_file

    def counted_hash(path):
        if Path(path) in media_paths:
            media_hash_calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(release_module, "sha256_file", counted_hash)
    for _ in range(3):
        for version in (1, 2):
            status, _, payload = http(server, "GET", f"/api/review?version={version}")
            assert status == 200 and json.loads(payload)["series"]["approved"]["version"] == 1
    for route in ("/api/download?version=1", "/api/download/report?version=1"):
        assert http(server, "GET", route)[0] == 200
    assert len(media_hash_calls) == 0

    # Writes still verify the immutable media even after serving it without rehashing.
    assert http(server, "POST", "/api/approve", {"version": 1})[0] == 200
    assert http(server, "POST", "/api/rollback", {"version": 1})[0] == 200
    assert media_hash_calls == [server.series.versions[0].media.path] * 2


def test_downloads_guard_approved_media_identity_without_rehashing(release_server) -> None:
    server, http = release_server
    assert http(server, "POST", "/api/approve", {"version": 1})[0] == 200
    server.series.versions[0].media.path.write_bytes(b"changed media")
    status, _, payload = http(server, "GET", "/api/review?version=1")
    assert status == 200 and json.loads(payload)["media_changed"]
    for route in ("/api/download?version=1", "/api/download/report?version=1"):
        status, _, payload = http(server, "GET", route)
        assert status == 409 and json.loads(payload)["error"]["code"] == "media_changed"


def test_release_routes_require_project_header_and_valid_body(release_server) -> None:
    server, http = release_server
    for route in ("/api/rollback", "/api/approve"):
        for body in ({}, {"version": "1"}, {"version": True}, {"version": 0}, {"version": 3},
                     {"version": 1, "extra": 1}):
            assert http(server, "POST", route, body)[0] == 400
        assert http(server, "GET", route)[0] == 405
        assert http(server, "POST", route + "?t=test-token", {"version": 1}, token=None)[0] == 403
        assert http(server, "POST", route, {"version": 1}, host="evil.example")[0] == 403
    assert http(server, "DELETE", "/api/approve?t=test-token", token=None)[0] == 403
    assert http(server, "GET", "/api/download?version=1", token=None)[0] == 403
    assert http(server, "POST", "/api/download", {})[0] == 405
    for series in (None, replace(server.series, project_dir=None)):
        original = server.series
        server.series = series
        try:
            for method, route in (("POST", "/api/rollback"), ("POST", "/api/approve"),
                                  ("DELETE", "/api/approve"), ("GET", "/api/download"),
                                  ("GET", "/api/download/report")):
                status, _, payload = http(server, method, route + "?version=1", {"version": 1})
                assert status == 409 and json.loads(payload)["error"]["code"] == "no_project"
        finally:
            server.series = original


def test_rounds_build_on_current_after_rollback(release_server) -> None:
    server, http = release_server
    assert http(server, "POST", "/api/rollback", {"version": 1})[0] == 200
    for version in (1, 2):
        assert http(server, "POST", f"/api/comments?version={version}", {
            "text": "Move this caption", "target": "caption", "start_frame": 18,
        })[0] == 201
        assert http(server, "POST", f"/api/rounds?version={version}", {})[0] == 202
    rounds = server.store.rounds(server.series.id)
    assert rounds[0].status == "sent" and load_batch(server.series.project_dir, rounds[0].batch)[
        "base_version"] == "v1"
    assert rounds[1].status == "failed" and rounds[1].error == {
        "code": "feedback_refused", "message": "v2 is not the current version (v1).",
    }


def test_retry_written_round_refuses_non_current_version(release_server, monkeypatch) -> None:
    server, http = release_server
    assert http(server, "POST", "/api/comments?version=2", {
        "text": "Fix this caption", "target": "caption", "start_frame": 18,
    })[0] == 201

    def failed_apply(*_args, **_kwargs):
        raise FeedbackError("Temporary failure")

    monkeypatch.setattr(send_module, "apply_word_fixes", failed_apply)
    assert http(server, "POST", "/api/rounds?version=2", {})[0] == 202
    round = server.store.rounds(server.series.id)[0]
    assert round.status == "failed" and round.batch == "b1"
    assert http(server, "POST", "/api/rollback", {"version": 1})[0] == 200
    assert http(server, "POST", "/api/rounds/n1/retry?version=2", {})[0] == 202
    round = server.store.rounds(server.series.id)[0]
    assert round.batch == "b1" and round.error == {
        "code": "feedback_refused", "message": "v2 is not the current version (v1).",
    }


@pytest.mark.parametrize("scenario", [
    "acceptance", "stale", "temporary", "guards", "recovery", "plain", "clarification",
    "earlier_question",
])
def test_round_http_dispatch_without_sockets(
    round_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str,
) -> None:
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    monkeypatch.setattr(__import__(__name__), "request", _in_process_request)
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(round_project)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series, send_runner=lambda work: work())
    if scenario == "acceptance":
        test_send_three_notes_as_one_batch(server)
    elif scenario == "stale":
        test_send_old_version_fails_and_retry_fails_again(server)
    elif scenario == "temporary":
        test_retry_temporary_failure_sends(server, monkeypatch)
    elif scenario == "guards":
        test_round_routes_guards_and_sending_locks(server)
    elif scenario == "recovery":
        test_server_recovers_sending_round_on_startup(server)
    elif scenario == "clarification":
        test_questions_answer_only_round(server)
    elif scenario == "earlier_question":
        test_questions_match_saved_versions_and_earlier_threads(server)
    else:
        server.series = replace(server.series, project_dir=None)
        plain = ReviewServer(video.media, video.sha256, None, store, token="test-token")
        test_plain_video_and_series_cannot_send(plain, server)


def test_automatic_question_answer_and_round_without_sockets(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(root)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series, send_runner=lambda work: work())
    http = _in_process_request
    assert http(server, "POST", "/api/comments?version=1",
                {"text": "Change this", "start_frame": 69})[0] == 201
    assert http(server, "POST", "/api/rounds?version=1", {})[0] == 202
    questions = json.loads(http(server, "GET", "/api/review")[2])["series"]["questions"]
    assert len(questions) == 1 and questions[0]["state"] == "asked"
    assert questions[0]["reason"] == "ambiguous_target"
    assert [option["value"] for option in questions[0]["options"]] == ["story", "caption",
                                                                    "element"]
    assert http(server, "POST", "/api/questions/answer?version=1", {
        "batch": "b1", "comment_version": "v1", "comment_id": "c1", "option": "element",
    })[0] == 200
    answered = json.loads(http(server, "GET", "/api/review")[2])["series"]["questions"]
    assert answered[0]["state"] == "answered"
    assert http(server, "POST", "/api/rounds?version=1", {})[0] == 202
    document = json.loads(http(server, "GET", "/api/review")[2])
    assert document["series"]["rounds"][0]["comments"] == []
    assert len(document["series"]["rounds"][0]["answers"]) == 1
    assert document["series"]["questions"] == []
    carried = load_batch(root, "b2")["items"][0]
    assert carried["carried"] and carried["comment"]["target"] == "element"
    assert carried["location"]["layers_known"] and carried["disposition"] == "open"


@pytest.fixture
def server(tmp_path: Path, socket_access) -> ReviewServer:
    video = tmp_path / "sample.mp4"
    video.write_bytes(VIDEO)
    media = ReviewMedia(
        video.resolve(), video.name, 12.0, Fraction(30000, 1001), True, "h264", 64, 36
    )
    review = ReviewServer(media, SHA, None, ReviewStore(tmp_path / "data"), token="test-token")
    thread = threading.Thread(target=review.serve_forever, daemon=True)
    thread.start()
    yield review
    review.shutdown()
    thread.join(timeout=5)
    review.server_close()


@pytest.fixture
def content_server(tmp_path: Path, socket_access) -> ReviewServer:
    video = tmp_path / "synced.mp4"
    video.write_bytes(VIDEO)
    media = ReviewMedia(video, video.name, 12.0, Fraction(30), True, "h264", 64, 36)
    review = ReviewServer(media, receipt.sha256_file(video), None,
                          ReviewStore(tmp_path / "data"), token="test-token")
    thread = threading.Thread(target=review.serve_forever, daemon=True)
    thread.start()
    yield review
    review.shutdown()
    thread.join(timeout=5)
    review.server_close()


def test_identical_media_replacement_is_served_and_hashed_once(
    content_server: ReviewServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = content_server.media.path
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(VIDEO)
    os.utime(replacement, ns=(video.stat().st_atime_ns, video.stat().st_mtime_ns + 1_000_000))
    os.replace(replacement, video)
    calls = []
    original = media_module._hash_stream

    def counted(stream):
        calls.append(stream.name)
        return original(stream)

    monkeypatch.setattr(media_module, "_hash_stream", counted)
    assert request(content_server, "GET", "/media")[2] == VIDEO
    status, _, body = request(content_server, "GET", "/media", headers={"Range": "bytes=2-5"})
    assert status == 206 and body == VIDEO[2:6]
    assert json.loads(request(content_server, "GET", "/api/review")[2])["media_changed"] is False
    assert len(calls) == 1


def test_same_size_media_replacement_is_rejected(content_server: ReviewServer) -> None:
    video = content_server.media.path
    video.write_bytes(b"x" * len(VIDEO))
    status, _, body = request(content_server, "GET", "/media")
    assert status == 409
    assert json.loads(body)["error"]["code"] == "media_changed"


def request(
    server: ReviewServer,
    method: str,
    route: str,
    body: object = None,
    *,
    token: str | None = "test-token",
    host: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    supplied = dict(headers or {})
    supplied["Host"] = host if host is not None else f"127.0.0.1:{server.port}"
    if token is not None:
        supplied["X-Yap-Token"] = token
    payload = None if body is None else json.dumps(body).encode()
    if payload is not None:
        supplied["Content-Type"] = "application/json"
        if not any(key.lower() == "content-length" for key in supplied):
            supplied["Content-Length"] = str(len(payload))
    connection.putrequest(method, route, skip_host=True, skip_accept_encoding=True)
    for key, value in supplied.items():
        connection.putheader(key, value)
    connection.endheaders(payload)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    assert result[1]["X-Content-Type-Options"] == "nosniff"
    assert result[1]["Referrer-Policy"] == "no-referrer"
    assert result[1]["Cache-Control"] in {"no-store", "no-cache"}
    assert "Access-Control-Allow-Origin" not in result[1]
    return result


@pytest.fixture(params=["in_process", "live"])
def findings_server(tmp_path: Path, monkeypatch, request):
    if request.param == "live":
        request.getfixturevalue("socket_access")
    path = tmp_path / "render.mp4"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=blue:size=64x36:rate=30", "-frames:v", "90", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ], timeout_s=30)
    media = ReviewMedia(path, path.name, 3.0, Fraction(30), True, "h264", 64, 36)
    store = ReviewStore(tmp_path / "data")
    hashed = store.hash_media(path)
    if request.param == "in_process":
        monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer", lambda *_args:
                            SimpleNamespace(server_port=0, server_close=lambda: None))
    server = ReviewServer(media, hashed.sha256, None, store, token="test-token")
    call = _in_process_request if request.param == "in_process" else globals()["request"]
    thread = None
    if request.param == "live":
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    try:
        yield server, call
    finally:
        if thread is not None:
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()


def test_review_includes_single_video_findings(findings_server) -> None:
    server, call = findings_server
    status, _, body = call(server, "GET", "/api/review")
    assert status == 200 and json.loads(body)["findings"] == {
        "available": False, "file": None, "stale": [], "items": [],
    }
    written = record_findings(server.media.path, [{
        "at": 0.5, "kind": "caption_issue", "severity": "high",
        "reason": "The caption is misspelled.", "fix": "Correct the spelling.",
    }], source="agent", reviewer="Reviewer A")
    assert call(server, "POST", "/api/comments", {
        "text": "My own comment.", "start_frame": 30, "target": "story",
    })[0] == 201
    status, _, body = call(server, "GET", "/api/review")
    assert status == 200
    document = json.loads(body)
    assert document["findings"] == {
        "available": True, "file": "render.findings.json", "stale": [],
        "items": [{**written["findings"][0], "target": "caption"}],
    }
    assert len(document["comments"]) == 1
    assert document["comments"][0]["text"] == "My own comment."
    assert "The caption is misspelled." not in document["text"]
    assert str(server.media.path.parent) not in json.dumps(document["findings"])


def test_review_malformed_findings_returns_200(findings_server) -> None:
    server, call = findings_server
    default_findings_path(server.media.path).write_text("{broken", encoding="utf-8")
    assert call(server, "POST", "/api/comments", {
        "text": "My own comment.", "start_frame": 30,
    })[0] == 201
    status, _, body = call(server, "GET", "/api/review")
    assert status == 200
    document = json.loads(body)
    assert document["findings"] == {
        "available": False, "file": None, "stale": [], "items": [],
        "error": "render.findings.json: malformed findings file",
    }
    assert document["media"]["sha256"] == server.sha256
    assert document["comments"][0]["text"] == "My own comment."
    assert "My own comment." in document["text"]
    assert str(server.media.path.parent) not in json.dumps(document["findings"])


@pytest.mark.parametrize("fixture", ["series_server", "removed_server"])
def test_review_series_findings_use_selected_version(fixture: str, request) -> None:
    server = request.getfixturevalue(fixture)
    call = globals()["request"] if fixture == "series_server" else _in_process_request
    written = []
    for version in server.series.versions:
        written.append(record_findings(version.media.path, [{
            "at": 0.5 * version.number, "kind": "jarring_cut", "severity": "medium",
            "reason": f"Version {version.number} cut needs attention.",
            "fix": f"Adjust version {version.number}.",
        }], source="agent", reviewer="Reviewer A"))
    for version, notes in zip(server.series.versions, written, strict=True):
        status, _, body = call(server, "GET", f"/api/review?version={version.number}")
        assert status == 200
        document = json.loads(body)
        path = default_findings_path(version.media.path)
        name = (path.relative_to(server.series.project_dir).as_posix()
                if server.series.project_dir is not None else path.name)
        assert document["media"]["sha256"] == version.sha256
        assert document["findings"] == {
            "available": True, "file": name, "stale": [],
            "items": [{**notes["findings"][0], "target": "story"}],
        }
        assert str(version.media.path.parent) not in json.dumps(document["findings"])
    latest = json.loads(call(server, "GET", "/api/review")[2])
    assert latest["findings"]["items"][0]["reason"] == "Version 2 cut needs attention."


def test_review_project_cut_findings_for_selected_version(removed_server: ReviewServer) -> None:
    server = removed_server
    cut = server.series.project_dir / "cut" / "cut.mp4"
    written = record_findings(cut, [{
        "at": 0.5, "kind": "slow", "severity": "low",
        "reason": "A pause runs long.", "fix": "Shorten the pause.",
    }], source="agent", reviewer="Reviewer A")
    assert receipt.sha256_file(cut) == server.series.versions[1].sha256
    for number in (1, 2):
        status, _, body = _in_process_request(server, "GET", f"/api/review?version={number}")
        assert status == 200
        document = json.loads(body)
        assert document["findings"] == ({
            "available": False, "file": None, "stale": ["cut/cut.findings.json"], "items": [],
        } if number == 1 else {
            "available": True, "file": "cut/cut.findings.json", "stale": [],
            "items": [{**written["findings"][0], "target": "story"}],
        })


@pytest.fixture
def project_server(round_project: Path, tmp_path: Path, socket_access) -> ReviewServer:
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(round_project)], store, lambda _: [])
    review = ReviewServer(video.media, video.sha256, None, store, token="test-token",
                          series=video.series, send_runner=lambda work: work())
    thread = threading.Thread(target=review.serve_forever, daemon=True)
    thread.start()
    try:
        yield review
    finally:
        review.shutdown()
        thread.join(timeout=5)
        review.server_close()


def test_send_three_notes_as_one_batch(project_server: ReviewServer) -> None:
    server = project_server
    posted = []
    for frame, text in ((3, "Move the first caption"), (33, "Keep this pause"),
                        (66, "Shorten the last caption")):
        status, _, body = request(server, "POST", "/api/comments?version=2",
                                  {"text": text, "start_frame": frame})
        assert status == 201
        posted.append(json.loads(body))
    before = json.loads(request(server, "GET", "/api/review")[2])
    assert before["series"]["project"] is True and before["series"]["rounds"] == []
    assert [item["send"]["state"] for item in before["comments"]] == ["draft"] * 3
    status, _, body = request(server, "POST", "/api/rounds?version=2", {})
    assert status == 202
    started = json.loads(body)
    assert started["id"] == "n1" and started["status"] == "sending"
    document = json.loads(request(server, "GET", "/api/review?version=2")[2])
    assert len(document["series"]["rounds"]) == 1
    round = document["series"]["rounds"][0]
    assert (round["status"], round["batch"], round["step"], round["error"]) == (
        "sent", "b1", None, None,
    )
    assert [item["send"] for item in document["comments"]] == [
        {"state": "sent", "round": "n1", "batch": "b1", "error": None},
    ] * 3
    batch = json.loads((server.series.project_dir / "feedback" / "b1.json").read_text())
    assert len(batch["items"]) == 3
    assert [(item["comment"]["id"], item["comment"]["text"], item["comment"]["revision"])
            for item in batch["items"]] == [
                (item["id"], item["text"], item["updated_at"]) for item in posted
            ]
    assert batch["review"]["series_id"] == server.series.id
    assert all(item["comment"]["version"] == "v2" for item in batch["items"])
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 400
    for method, payload in (("PATCH", {"text": "Changed"}), ("DELETE", None)):
        status, _, body = request(server, method, "/api/comments/c1?version=2", payload)
        assert status == 409
        assert json.loads(body)["error"]["code"] == "comment_sent"
        assert json.loads(body)["error"]["message"] == (
            "This note was sent in round n1 and can no longer be changed. Add a new note instead."
        )
    exported = json.loads(request(server, "GET", "/api/export.json?version=2")[2])
    assert all("send" not in item for item in exported["comments"])
    assert "rounds" not in exported["series"] and "project" not in exported["series"]
    request(server, "POST", "/api/comments?version=2", {"text": "Another", "start_frame": 90})
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    document = json.loads(request(server, "GET", "/api/review")[2])
    assert [item["id"] for item in document["series"]["rounds"]] == ["n2", "n1"]
    assert document["comments"][-1]["send"]["batch"] == "b2"
    assert len(json.loads((server.series.project_dir / "feedback" / "b2.json").read_text())[
        "items"]) == 1


def test_questions_answer_only_round(project_server: ReviewServer) -> None:
    server = project_server
    root = server.series.project_dir
    assert request(server, "POST", "/api/comments?version=2",
                   {"text": "Change this", "start_frame": 69})[0] == 201
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    ask_clarification(root, "b1", "v2", "c1", "Which do you want changed here?",
                      options=[{"value": "caption", "label": "The caption"}])
    document = json.loads(request(server, "GET", "/api/review?version=2")[2])
    assert document["series"]["questions"] == [{
        "batch": "b1", "comment": {"version": 2, "id": "c1"}, "comment_version": "v2",
        "question": "Which do you want changed here?", "state": "asked", "reason": "agent",
        "options": [{"value": "caption", "label": "The caption"}],
        "answer": None, "option": None,
    }]
    route = "/api/questions/answer?version=2"
    body = {"batch": "b1", "comment_version": "v2", "comment_id": "c1", "option": "caption"}
    assert request(server, "POST", route + "&t=test-token", body, token=None)[0] == 403
    assert request(server, "GET", route)[0] == 405
    for change in ({"option": "bad"}, {"answer": " "}, {"comment_version": "v99"},
                   {"extra": True}):
        assert request(server, "POST", route, {**body, **change})[0] == 400
    assert request(server, "POST", route, {**body, "comment_id": "c99"})[0] == 404
    assert request(server, "POST", route, {**body, "batch": "b99"})[0] == 404
    assert request(server, "POST", route, body)[0] == 200
    document = json.loads(request(server, "GET", "/api/review?version=2")[2])
    question = document["series"]["questions"][0]
    assert question["state"] == "answered" and question["option"] == "caption"
    assert question["answer"] is None and question["answered_at"].endswith("Z")
    assert request(server, "POST", route, body)[0] == 400
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    document = json.loads(request(server, "GET", "/api/review?version=2")[2])
    round = document["series"]["rounds"][0]
    assert round["status"] == "sent" and round["batch"] == "b2"
    assert round["comments"] == []
    assert round["answers"] == [{"batch": "b1", "comment": {"version": "v2", "id": "c1"}}]
    assert document["series"]["questions"] == []
    carried = load_batch(root, "b2")["items"][0]
    assert carried["carried"] and carried["disposition"] == "open"
    assert carried["comment"]["target"] == "caption"
    assert carried["clarification"]["option"] == "caption"
    assert load_batch(root, "b1")["items"][0]["clarification"]["carried_to"] == "b2"
    assert request(server, "POST", route, body)[0] == 409
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 400


def test_questions_match_saved_versions_and_earlier_threads(project_server: ReviewServer) -> None:
    server = project_server
    root = server.series.project_dir
    for version in (1, 2):
        assert request(server, "POST", f"/api/comments?version={version}",
                       {"text": "Change this", "start_frame": 69})[0] == 201
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    ask_clarification(root, "b1", "v1", "c1", "Which earlier caption?")
    document = json.loads(request(server, "GET", "/api/review?version=2")[2])
    question = document["series"]["questions"][0]
    assert question["comment"] == {"version": 1, "id": "c1"}
    assert question["comment_version"] == "v1"
    assert document["series"]["threads"][0]["id"] == "c1"
    assert request(server, "POST", "/api/questions/answer?version=2", {
        "batch": "b1", "comment_version": "v1", "comment_id": "c1",
        "answer": "The last caption in the first version",
    })[0] == 200
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    batch = load_batch(root, "b2")
    assert len(batch["items"]) == 1 and batch["items"][0]["comment"]["version"] == "v1"
    assert batch["items"][0]["clarification"]["answer"] == (
        "The last caption in the first version"
    )
    assert request(server, "POST", "/api/comments?version=2",
                   {"text": "Another note", "start_frame": 3})[0] == 201
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    assert len(load_batch(root, "b3")["items"]) == 1


def test_send_old_version_fails_and_retry_fails_again(project_server: ReviewServer) -> None:
    server = project_server
    request(server, "POST", "/api/comments?version=1", {"text": "Old note", "start_frame": 3})
    assert request(server, "POST", "/api/rounds?version=1", {})[0] == 202
    document = json.loads(request(server, "GET", "/api/review?version=1")[2])
    round = document["series"]["rounds"][0]
    assert round["status"] == "failed" and round["attempts"] == 1
    assert round["error"] == {"code": "feedback_refused",
                              "message": "v1 is not the current version (v2)."}
    assert document["comments"][0]["send"]["error"] == round["error"]
    assert request(server, "POST", "/api/rounds?version=1", {})[0] == 400
    queued = []
    server._send_runner = queued.append
    status, _, body = request(server, "POST", "/api/rounds/n1/retry?version=1", {})
    assert status == 202 and json.loads(body)["status"] == "sending"
    assert json.loads(body)["attempts"] == 2 and json.loads(body)["error"] is None
    assert len(queued) == 1
    queued[0]()
    document = json.loads(request(server, "GET", "/api/review?version=1")[2])
    round = document["series"]["rounds"][0]
    assert round["status"] == "failed" and round["attempts"] == 2
    assert "not the current" in round["error"]["message"]
    assert not (server.series.project_dir / "feedback").exists()


def test_retry_temporary_failure_sends(project_server: ReviewServer, monkeypatch) -> None:
    server = project_server
    request(server, "POST", "/api/comments?version=2", {"text": "New note", "start_frame": 3})
    original = send_module.build_batch
    calls = 0

    def flaky(project, export, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FeedbackError("Temporary cause.")
        return original(project, export, **kwargs)

    monkeypatch.setattr(send_module, "build_batch", flaky)
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 202
    document = json.loads(request(server, "GET", "/api/review")[2])
    assert document["comments"][0]["send"]["state"] == "failed"
    assert request(server, "POST", "/api/rounds/n1/retry?version=2")[0] == 202
    document = json.loads(request(server, "GET", "/api/review")[2])
    round = document["series"]["rounds"][0]
    assert round["status"] == "sent" and round["attempts"] == 2 and round["batch"] == "b1"
    assert document["comments"][0]["send"]["state"] == "sent"
    assert request(server, "POST", "/api/rounds/n1/retry?version=2")[0] == 409


def test_round_routes_guards_and_sending_locks(project_server: ReviewServer) -> None:
    server = project_server
    assert request(server, "POST", "/api/rounds?version=2", {})[0] == 400
    assert request(server, "POST", "/api/rounds?t=test-token", {}, token=None)[0] == 403
    assert request(server, "POST", "/api/rounds/n1/retry?t=test-token", {}, token=None)[0] == 403
    assert request(server, "POST", "/api/rounds", {}, host="evil.example")[0] == 403
    assert request(server, "POST", "/api/rounds/n99/retry?version=2")[0] == 404
    assert request(server, "POST", "/api/rounds", {"unknown": 1})[0] == 400
    assert request(server, "GET", "/api/rounds")[0] == 405
    queued = []
    server._send_runner = queued.append
    request(server, "POST", "/api/comments?version=2", {"text": "Note", "start_frame": 3})
    assert request(server, "POST", "/api/rounds", {})[0] == 202
    document = json.loads(request(server, "GET", "/api/review")[2])
    assert document["comments"][0]["send"]["state"] == "sending"
    assert request(server, "POST", "/api/rounds/n1/retry?version=2", {})[0] == 409
    assert request(server, "POST", "/api/rounds/n1/retry?version=1", {})[0] == 404
    assert request(server, "PATCH", "/api/comments/c1", {"kind": "keep"})[0] == 409
    assert request(server, "DELETE", "/api/comments/c1")[0] == 409
    version = server.series.versions[-1]
    version.media.path.write_bytes(b"changed")
    assert request(server, "POST", "/api/rounds", {})[0] == 409
    assert request(server, "POST", "/api/rounds/n1/retry", {})[0] == 409


def test_server_recovers_sending_round_on_startup(project_server: ReviewServer) -> None:
    server = project_server
    series = server.series
    version = series.versions[-1]
    comment = server.store.add_comment(version.sha256, version.media.facts(version.sha256),
                                       {"text": "Note", "start_frame": 0})
    server.store.start_round(series.id, 2, [comment])
    restarted = ReviewServer(server.media, server.sha256, None, server.store, series=series)
    try:
        recovered = server.store.rounds(series.id)[0]
        assert recovered.status == "failed" and recovered.error["code"] == "interrupted"
        assert recovered.error["message"] == (
            "The review server stopped while sending these notes. Send them again."
        )
    finally:
        restarted.server_close()


def test_plain_video_and_series_cannot_send(server: ReviewServer, series_server: ReviewServer):
    for selected in (server, series_server):
        for route in ("/api/rounds", "/api/rounds/n1/retry"):
            status, _, body = request(selected, "POST", route, {})
            assert status == 409
            assert json.loads(body)["error"]["code"] == "no_project"
            assert json.loads(body)["error"]["message"] == (
                "Open the project folder with --versions to send notes."
            )
    assert json.loads(request(series_server, "GET", "/api/review")[2])["series"]["project"] is False


def test_authorization_and_host(server: ReviewServer) -> None:
    assert request(server, "GET", "/api/review", token=None)[0] == 403
    assert request(server, "GET", "/api/review", token="wrong")[0] == 403
    assert request(server, "GET", "/api/review", host="evil.example")[0] == 403
    assert request(server, "GET", "/api/review", host="")[0] == 403
    assert request(server, "GET", "/api/review", host="localhost:" + str(server.port))[0] == 200
    assert request(server, "GET", "/api/review?t=test-token", token=None)[0] == 200
    assert request(server, "GET", "/media", token=None)[0] == 403
    assert request(server, "GET", "/media?t=test-token", token=None)[0] == 200
    assert (
        request(
            server,
            "POST",
            "/api/comments?t=test-token",
            {"text": "ok", "start_frame": 0},
            token=None,
        )[0]
        == 403
    )


@pytest.fixture
def series_server(tmp_path: Path, socket_access) -> ReviewServer:
    store = ReviewStore(tmp_path / "data")
    versions = []
    for number, color in enumerate(("red", "blue"), 1):
        path = tmp_path / f"render-{number}.mp4"
        ffmpeg.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            f"color=c={color}:size=64x36:rate=30", "-frames:v", "90", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(path),
        ], timeout_s=30)
        media = server_module.ReviewMedia(path, path.name, 3.0, Fraction(30), True, "h264", 64, 36)
        hashed = store.hash_media(path)
        words = (CutWord("t1:0", "s1", "Removed", 0.4, 0.8),
                 CutWord("t1:1", "s1", "Kept", 1.2, 1.5)) if number == 1 else (
                     CutWord("t1:1", "s1", "Kept", 2.0, 2.3),
                 )
        versions.append(ReviewVersion(number, media, hashed.sha256, hashed.identity,
                                      CutWords("output", {}, words), f"v{number}"))
    stored = store.open_series([SeriesEntry(item.sha256, item.media.name, item.version_id,
                                           "sample-video") for item in versions])
    newest = versions[-1]
    server = ReviewServer(newest.media, newest.sha256, None, store, token="test-token",
                          series=ReviewSeries(stored.id, tuple(versions)),
                          allowed_hosts=("review.example.net",))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_series_defaults_selection_comments_and_export(series_server: ReviewServer) -> None:
    server = series_server
    for number in (1, 2):
        status, _, body = request(server, "GET", f"/api/review?version={number}")
        assert status == 200
        document = json.loads(body)
        assert document["series"]["current"] == number and document["series"]["newest"] == 2
        assert document["media"]["sha256"] == server.series.versions[number - 1].sha256
        assert document["previous"] is None
        assert [item["word_map"] for item in document["series"]["versions"]] == [True, True]
        assert document["series"]["versions"][0]["fps"] == "30/1"
        status, _, body = request(server, "POST", f"/api/comments?version={number}",
                                  {"text": f"Comment v{number}", "start_frame": 3})
        assert status == 201 and json.loads(body)["id"] == "c1"
    document = json.loads(request(server, "GET", "/api/review")[2])
    assert document["series"]["current"] == 2
    assert [item["comment_count"] for item in document["series"]["versions"]] == [1, 1]
    assert document["comments"][0]["text"] == "Comment v2"
    for number in (1, 2):
        assert server.store.comments(server.series.versions[number - 1].sha256)[0].text == (
            f"Comment v{number}"
        )
        exported = json.loads(request(server, "GET", f"/api/export.json?version={number}")[2])
        assert exported["series"]["version"] == number
        assert exported["comments"][0]["text"] == f"Comment v{number}"
        assert set(exported["series"]["versions"][0]) == {"number", "name", "sha256", "version_id"}
    assert request(server, "PATCH", "/api/comments/c1?version=1", {"text": "Patched old"})[0] == 200
    assert request(server, "DELETE", "/api/comments/c1?version=1")[0] == 204
    assert server.store.comments(server.series.versions[0].sha256) == []
    assert server.store.comments(server.series.versions[1].sha256)[0].text == "Comment v2"


@pytest.mark.parametrize("number", [1, 2])
def test_series_range_tokens_and_proxy_hosts(series_server: ReviewServer, number: int) -> None:
    server = series_server
    content = server.series.versions[number - 1].media.path.read_bytes()
    route = f"/media?version={number}"
    assert request(server, "GET", route)[2] == content
    assert request(server, "HEAD", route)[1]["Content-Length"] == str(len(content))
    status, headers, body = request(server, "GET", route, headers={"Range": "bytes=0-99"})
    assert status == 206 and body == content[:100]
    assert headers["Content-Range"] == f"bytes 0-99/{len(content)}"
    assert request(server, "GET", route, headers={"Range": "bytes=-10"})[2] == content[-10:]
    assert request(server, "GET", route, headers={"Range": f"bytes={len(content)}-"})[0] == 416
    assert request(server, "GET", route, token="wrong")[0] == 403
    assert request(server, "GET", f"/api/review?version={number}", token="wrong")[0] == 403
    assert request(server, "GET", route + "&t=test-token", token=None)[0] == 200
    assert request(server, "GET", route, host="review.example.net:443")[0] == 200
    assert request(server, "GET", f"/api/review?version={number}",
                   host="unknown.example.net")[0] == 403
    assert request(server, "POST", f"/api/comments?version={number}&t=test-token",
                   {"text": "No header", "start_frame": 0}, token=None)[0] == 403


def test_series_changed_old_media_blocks_only_that_version(series_server: ReviewServer) -> None:
    server = series_server
    old = server.series.versions[0]
    old.media.path.write_bytes(b"replaced render")
    assert request(server, "GET", "/media?version=1")[0] == 409
    assert request(server, "POST", "/api/comments?version=1",
                   {"text": "Refused", "start_frame": 0})[0] == 409
    assert request(server, "PATCH", "/api/comments/c1?version=1", {"text": "Refused"})[0] == 409
    assert request(server, "DELETE", "/api/comments/c1?version=1")[0] == 409
    assert json.loads(request(server, "GET", "/api/review?version=1")[2])["media_changed"] is True
    assert request(server, "GET", "/media?version=2")[0] == 200
    assert request(server, "POST", "/api/comments?version=2",
                   {"text": "Accepted", "start_frame": 0})[0] == 201


@pytest.mark.parametrize("value", ["0", "3", "-1", "1.0", "bad", "previous", "", "1&version=2"])
def test_series_invalid_version(series_server: ReviewServer, value: str) -> None:
    for route in ("/media", "/api/review", "/api/export.json"):
        status, _, body = request(series_server, "GET", f"{route}?version={value}")
        assert status == 400 and json.loads(body)["error"]["type"] == "validation_error"
        assert "version" in json.loads(body)["error"]["message"]


def test_plain_version_is_not_found_and_plain_exports_mark_no_series(server: ReviewServer) -> None:
    for route in ("/media", "/api/review", "/api/export.json"):
        status, _, body = request(server, "GET", f"{route}?version=1")
        assert status == 404 and json.loads(body)["error"]["message"] == "version not found"
    assert json.loads(request(server, "GET", "/api/review")[2])["series"] is None
    assert json.loads(request(server, "GET", "/api/export.json")[2])["series"] is None
    assert request(server, "GET", "/api/map?from=1&to=2&frame=0")[0] == 404


def test_series_map_occurrences_removed_and_time(series_server: ReviewServer) -> None:
    server = series_server
    result = json.loads(request(server, "GET", "/api/map?from=1&to=2&frame=42")[2])
    assert (result["status"], result["from"], result["to"], result["frame"]) == ("mapped", 1, 2, 66)
    assert result["time_s"] == pytest.approx(2.2)
    assert result["word"]["id"] == result["target"]["id"] == "t1:1"
    result = json.loads(request(server, "GET", "/api/map?from=1&to=2&frame=18")[2])
    assert result["status"] == "removed" and result["frame"] == 60
    assert result["target"]["id"] == "t1:1"
    server.series = replace(server.series, versions=tuple(replace(item, words=None)
                                                         for item in server.series.versions))
    result = json.loads(request(server, "GET", "/api/map?from=1&to=2&frame=18")[2])
    assert result["status"] == "time" and result["time_s"] == 0.6 and result["frame"] == 18
    assert request(server, "POST", "/api/map?from=1&to=2&frame=0")[0] == 405
    assert request(server, "GET", "/api/map?from=1&to=2&frame=0", token="wrong")[0] == 403
    for query in ("from=0&to=2&frame=0", "from=1&to=3&frame=0", "from=1&to=2&frame=-1",
                  "from=1&to=2&frame=NaN", "from=1&to=2", "from=x&to=2&frame=0"):
        assert request(server, "GET", f"/api/map?{query}")[0] == 400


def test_series_map_variable_rate_seconds(series_server: ReviewServer) -> None:
    server = series_server
    server.series = replace(server.series, versions=tuple(
        replace(item, media=replace(item.media, fps=None, cfr=False))
        for item in server.series.versions
    ))
    result = json.loads(request(server, "GET", "/api/map?from=1&to=2&time=1.4")[2])
    assert result["status"] == "mapped" and result["time_s"] == pytest.approx(2.2)
    assert result["frame"] is None
    for value in ("NaN", "inf", "-1", "bad"):
        assert request(server, "GET", f"/api/map?from=1&to=2&time={value}")[0] == 400


def test_series_map_variable_rate_with_query_token(series_server: ReviewServer) -> None:
    server = series_server
    server.series = replace(server.series, versions=tuple(
        replace(item, media=replace(item.media, fps=None, cfr=False))
        for item in server.series.versions
    ))
    status, _, body = request(
        server, "GET", "/api/map?from=1&to=2&t=test-token&time=1.4", token=None,
    )
    assert status == 200
    result = json.loads(body)
    assert result["status"] == "mapped"
    assert result["time_s"] == pytest.approx(2.2)
    assert result["frame"] is None


def test_series_threads_and_golden_export(series_server: ReviewServer) -> None:
    server = series_server
    request(server, "POST", "/api/comments?version=1",
            {"kind": "fix", "target": "caption", "text": "Move caption", "start_frame": 15})
    document = json.loads(request(server, "GET", "/api/review?version=2")[2])
    thread = document["series"]["threads"][0]
    assert thread["version"] == 1 and thread["state"] == "open" and thread["reply"] is None
    reply = server.store.add_reply(server.series.id, comment_version=1, comment_id="c1", version=2,
                                   status="changed", text="Moved to the word")
    document = json.loads(request(server, "GET", "/api/review")[2])
    assert document["series"]["threads"][0]["reply"]["id"] == reply.id
    assert document["series"]["threads"][0]["state"] == "changed"
    exported = json.loads(request(server, "GET", "/api/export.json?version=2")[2])
    assert exported["series"]["id"] == server.series.id and exported["series"]["version"] == 2
    assert exported["series"]["threads"] == document["series"]["threads"]
    expected = (
        "Feedback on render-2.mp4, version 2 of 2 (0:03.0, 30 fps)\n"
        "0 comments\n\n"
        "Earlier versions: 1 note, all replied\n"
    )
    assert request(server, "GET", "/api/export.txt?version=2")[2].decode() == expected
    assert document["text"] == expected
    assert document["earlier_count"] == 1
    assert "Earlier versions: 1 note, all replied\nv1 0:00.5 [Fix: caption]" in (
        document["text_earlier"]
    )
    assert " open, " not in document["text_earlier"]
    assert "Changed in v2: Moved to the word" in document["text_earlier"]
    first = json.loads(request(server, "GET", "/api/review?version=1")[2])
    assert first["series"]["threads"] == []
    assert first["text_earlier"] is None
    assert first["earlier_count"] == 0
    assert first["text"].splitlines()[0] == (
        "Feedback on render-1.mp4, version 1 of 2 (0:03.0, 30 fps)"
    )


def test_seventh_version_text_excludes_replied_history_by_default(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "data")
    versions = []
    for number in range(1, 8):
        path = tmp_path / f"render-{number}.mp4"
        path.write_bytes(f"synthetic version {number}".encode())
        media = ReviewMedia(path, path.name, 3.0, Fraction(30), True, "h264", 64, 36)
        hashed = store.hash_media(path)
        versions.append(ReviewVersion(number, media, hashed.sha256, hashed.identity,
                                      None, f"v{number}"))
    stored = store.open_series([SeriesEntry(item.sha256, item.media.name, item.version_id)
                                for item in versions])
    server = ReviewServer(versions[-1].media, versions[-1].sha256, None, store,
                          token="test-token", series=ReviewSeries(stored.id, tuple(versions)))
    for number in range(2, 7):
        item = versions[number - 1]
        note = store.add_comment(item.sha256, item.media.facts(item.sha256), {
            "text": f"Old note v{number}", "start_frame": number,
        })
        store.add_reply(stored.id, comment_version=number, comment_id=note.id,
                        version=number + 1, status="changed", text=f"Changed in v{number + 1}")
    latest = versions[-1]
    for text in ("New note one", "New note two"):
        store.add_comment(latest.sha256, latest.media.facts(latest.sha256),
                          {"text": text, "start_frame": 7})
    default = _in_process_request(server, "GET", "/api/export.txt?version=7")[2].decode()
    document = json.loads(_in_process_request(server, "GET", "/api/review?version=7")[2])
    assert document["text"] == default
    assert "New note one" in default and "New note two" in default
    assert "Earlier versions: 5 notes, all replied" in default
    assert all(f"Old note v{number}" not in default for number in range(2, 7))
    included = _in_process_request(server, "GET", "/api/export.txt?version=7&earlier=1")[2]
    assert included.decode() == document["text_earlier"]
    assert document["earlier_count"] == 1
    assert "Old note v6" in document["text_earlier"]
    assert all(f"Old note v{number}" not in document["text_earlier"]
               for number in range(2, 6))
    assert "Earlier versions: 5 notes, all replied\nv6 0:00.2 [Fix]" in (
        document["text_earlier"]
    )
    assert " open, " not in document["text_earlier"]
    for value in ("", "0", "true", "2", "1&earlier=1"):
        assert _in_process_request(server, "GET", f"/api/export.txt?earlier={value}")[0] == 400

    open_note = store.add_comment(versions[4].sha256,
                                  versions[4].media.facts(versions[4].sha256),
                                  {"text": "Still open from v5", "start_frame": 12})
    assert open_note.id == "c2"
    default = _in_process_request(server, "GET", "/api/export.txt?version=7")[2].decode()
    assert "Earlier versions: 6 notes, 1 still open" in default
    assert "Still open from v5" not in default
    included = _in_process_request(server, "GET", "/api/export.txt?version=7&earlier=1")[2]
    assert "Still open from v5" in included.decode()
    assert "Earlier versions: 6 notes, 1 still open\nv5 0:00.4 [Fix]" in included.decode()
    assert " open, " not in included.decode()


def test_series_can_be_another_video_on_the_same_server(series_server: ReviewServer) -> None:
    first = series_server.series.versions[0]
    series_video = server_module.ReviewVideo(
        series_server.media, series_server.sha256, token="series-token", series=series_server.series
    )
    plain = ReviewMedia(first.media.path, "plain.mp4", 3.0, Fraction(30), True, "h264", 64, 36)
    server = ReviewServer(plain, first.sha256, None, series_server.store, token="plain-token",
                          additional_videos=(series_video,))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        document = json.loads(request(server, "GET", "/v/2/api/review?version=1",
                                      token="series-token")[2])
        assert document["series"]["current"] == 1 and document["media"]["sha256"] == first.sha256
        assert request(server, "GET", "/v/2/media?version=1", token="series-token")[0] == 200
        assert request(server, "GET", "/v/2/media?version=1", token="plain-token")[0] == 403
        assert request(server, "GET", "/v/1/media?version=1", token="plain-token")[0] == 404
        assert request(server, "GET", "/v/1/api/review", token="plain-token")[0] == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def multi_server(tmp_path: Path, socket_access) -> ReviewServer:
    videos = []
    for name, source in (
        ("first.mp4", "testsrc2=size=64x36:rate=30"),
        ("second <b>.mp4", "color=c=blue:size=64x36:rate=30"),
    ):
        video = tmp_path / name
        ffmpeg.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", source,
             "-map", "0:v:0", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(video)],
            timeout_s=30,
        )
        media = ReviewMedia(video, name, 1.0, Fraction(30), True, "h264", 64, 36)
        videos.append(server_module.ReviewVideo(media, receipt.sha256_file(video)))
    first, second = videos
    review = ReviewServer(
        first.media, first.sha256, None, ReviewStore(tmp_path / "data"),
        token=first.token, additional_videos=(second,), allowed_hosts=("review.example.net",),
    )
    thread = threading.Thread(target=review.serve_forever, daemon=True)
    thread.start()
    try:
        yield review
    finally:
        review.shutdown()
        thread.join(timeout=5)
        review.server_close()


@pytest.fixture(params=["server", "multi_server"])
def static_server(request: pytest.FixtureRequest) -> ReviewServer:
    """Exercise static routes in both review modes."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("prefix", ["", "/v/1"])
@pytest.mark.parametrize("name", ["index.html", "list.html"])
def test_static_html_is_not_served(
    static_server: ReviewServer, prefix: str, name: str
) -> None:
    for query in ("", f"?t={static_server.token}"):
        status, _, body = request(
            static_server, "GET", f"{prefix}/static/{name}{query}", token=None
        )
        assert status == 404
        assert json.loads(body)["error"]["code"] == "not_found"


def test_review_roots_and_static_assets_still_load(static_server: ReviewServer) -> None:
    roots = [("/", static_server.list_token or static_server.token)]
    if static_server.list_token is not None:
        roots.append(("/v/1/", static_server.videos[0].token))
    for root, token in roots:
        status, headers, body = request(static_server, "GET", f"{root}?t={token}", token=None)
        assert status == 200
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert b'href="static/app.css"' in body
        if root == "/" and static_server.list_token is not None:
            assert b"Choose a video" in body
        else:
            assert b"Loading review" in body
            assert b'src="static/app.mjs"' in body
        for name, content_type in (
            ("app.mjs", "text/javascript; charset=utf-8"),
            ("app.css", "text/css; charset=utf-8"),
        ):
            status, headers, body = request(
                static_server, "GET", f"{root}static/{name}", token=None
            )
            assert status == 200
            assert headers["Content-Type"] == content_type
            assert body


def test_multiple_video_routes_and_token_isolation(multi_server: ReviewServer) -> None:
    for number, video in enumerate(multi_server.videos, 1):
        prefix = f"/v/{number}"
        for route in ("/", "/api/review", "/media", "/static/app.mjs", "/static/app.css"):
            status, _, body = request(
                multi_server, "GET", f"{prefix}{route}?t={video.token}", token=None
            )
            assert status == 200
            if route == "/api/review":
                assert json.loads(body)["media"]["sha256"] == video.sha256
            elif route == "/media":
                assert body == video.media.path.read_bytes()
            elif route == "/":
                assert b"Loading review" in body
        wrong_token = multi_server.videos[2 - number].token
        for method, route, payload in (
            ("GET", "/", None), ("GET", "/api/review", None),
            ("GET", "/media", None), ("HEAD", "/media", None),
            ("GET", "/api/export.txt", None),
            ("GET", "/api/export.json", None),
            ("POST", "/api/comments", {"text": "Point", "start_frame": 0}),
            ("PATCH", "/api/comments/c1", {"text": "Changed"}),
            ("DELETE", "/api/comments/c1", None),
        ):
            assert request(
                multi_server, method, prefix + route, payload, token=wrong_token
            )[0] == 403
            assert request(multi_server, method, prefix + route, payload, token=None)[0] == 403
            assert request(
                multi_server, method, prefix + route, payload, token=video.token,
                host="review.example.net.evil",
            )[0] == 403
        assert request(
            multi_server, "GET", prefix + "/", token=multi_server.list_token
        )[0] == 403
        assert request(multi_server, "GET", prefix + "/static/app.mjs", token=None)[0] == 200
        assert request(
            multi_server, "GET", prefix + "/static/app.mjs", token=None, host="evil.example"
        )[0] == 403
    for route in ("/v/0/", "/v/3/", "/v/01/", "/api/review", "/media"):
        assert request(multi_server, "GET", route)[0] == 404


def test_multiple_video_comments_and_exports_are_separate(multi_server: ReviewServer) -> None:
    first, second = multi_server.videos
    status, _, body = request(
        multi_server, "POST", "/v/1/api/comments",
        {"text": "First video feedback", "start_frame": 2}, token=first.token,
    )
    assert status == 201
    comment_id = json.loads(body)["id"]
    for number, video in enumerate((first, second), 1):
        prefix = f"/v/{number}/api"
        for route in ("review", "export.json"):
            status, _, body = request(
                multi_server, "GET", f"{prefix}/{route}", token=video.token
            )
            assert status == 200
            document = json.loads(body)
            assert document["media"]["sha256"] == video.sha256
            assert [item["text"] for item in document["comments"]] == (
                ["First video feedback"] if number == 1 else []
            )
        status, _, body = request(
            multi_server, "GET", f"{prefix}/export.txt", token=video.token
        )
        assert status == 200
        assert (b"First video feedback" in body) is (number == 1)
    assert request(
        multi_server, "PATCH", f"/v/2/api/comments/{comment_id}",
        {"text": "Other video"}, token=second.token,
    )[0] == 404
    assert request(
        multi_server, "PATCH", f"/v/1/api/comments/{comment_id}",
        {"text": "Updated feedback"}, token=first.token,
    )[0] == 200
    assert multi_server.store.comments(first.sha256)[0].text == "Updated feedback"
    assert multi_server.store.comments(second.sha256) == []
    assert request(
        multi_server, "DELETE", f"/v/1/api/comments/{comment_id}", token=first.token
    )[0] == 204


def test_list_page_is_private_and_escapes_names(multi_server: ReviewServer) -> None:
    assert multi_server.list_token not in {video.token for video in multi_server.videos}
    for token in (None, "wrong", *(video.token for video in multi_server.videos)):
        assert request(multi_server, "GET", "/", token=token)[0] == 403
    parsed = urlsplit(multi_server.list_url)
    status, headers, body = request(
        multi_server, "GET", parsed.path + "?" + parsed.query,
        token=None, host="REVIEW.EXAMPLE.NET:8443",
    )
    assert status == 200
    assert headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert b"first.mp4" in body and b"second &lt;b&gt;.mp4" in body
    assert b"<b>" not in body
    assert b"1.0 s" in body
    for number, video in enumerate(multi_server.videos, 1):
        assert f'v/{number}/?t={video.token}'.encode() in body
    assert request(
        multi_server, "GET", "/", token=multi_server.list_token, host="evil.example"
    )[0] == 403


def test_prefixed_media_range_through_proxy(multi_server: ReviewServer) -> None:
    second = multi_server.videos[1]
    status, headers, body = request(
        multi_server, "GET", "/v/2/media", token=second.token,
        host="REVIEW.EXAMPLE.NET:8443", headers={"Range": "bytes=0-99"},
    )
    assert status == 206
    assert body == second.media.path.read_bytes()[:100]
    assert headers["Content-Range"] == f"bytes 0-99/{second.media.path.stat().st_size}"


@pytest.mark.parametrize("changed_number", [1, 2])
def test_media_change_and_previous_review_are_per_video(
    multi_server: ReviewServer, changed_number: int
) -> None:
    first, second = multi_server.videos
    old_facts = replace(second.media.facts(OLD_SHA), name="earlier.mp4")
    multi_server.store.add_comment(OLD_SHA, old_facts, {"text": "Earlier", "start_frame": 0})
    multi_server.videos = (first, replace(second, previous_sha=OLD_SHA))
    changed = multi_server.videos[changed_number - 1]
    changed.media.path.write_bytes(b"changed synthetic bytes")
    for method, route, payload in (
        ("GET", "media", None), ("HEAD", "media", None),
        ("POST", "api/comments", {"text": "New", "start_frame": 0}),
        ("PATCH", "api/comments/c1", {"text": "Updated"}),
        ("DELETE", "api/comments/c1", None),
    ):
        assert request(
            multi_server, method, f"/v/{changed_number}/" + route, payload, token=changed.token
        )[0] == 409
    one = json.loads(request(multi_server, "GET", "/v/1/api/review", token=first.token)[2])
    two = json.loads(request(multi_server, "GET", "/v/2/api/review", token=second.token)[2])
    assert one["media_changed"] is (changed_number == 1) and one["previous"] is None
    assert two["media_changed"] is (changed_number == 2) and two["previous"]["sha256"] == OLD_SHA
    healthy_number = 3 - changed_number
    healthy = multi_server.videos[healthy_number - 1]
    assert request(
        multi_server, "GET", f"/v/{healthy_number}/media", token=healthy.token
    )[0] == 200
    assert request(
        multi_server, "POST", f"/v/{healthy_number}/api/comments",
        {"text": "Healthy video", "start_frame": 0}, token=healthy.token,
    )[0] == 201
    assert request(multi_server, "GET", "/v/1/api/export.json", token=first.token)[0] == 200
    assert request(
        multi_server, "GET", "/v/1/api/export.json?version=previous", token=first.token
    )[0] == 404
    assert json.loads(request(
        multi_server, "GET", "/v/2/api/export.json?version=previous", token=second.token
    )[2])["comments"][0]["text"] == "Earlier"


@pytest.fixture
def proxy_server(tmp_path: Path, socket_access) -> ReviewServer:
    video = tmp_path / "sample.mp4"
    video.write_bytes(VIDEO * 8)
    media = ReviewMedia(video, video.name, 12.0, Fraction(30), True, "h264", 64, 36)
    review = ReviewServer(
        media,
        SHA,
        None,
        ReviewStore(tmp_path / "data"),
        token="test-token",
        allowed_hosts=("Review.Example.NET", "Pinned.Example.NET:8443", "192.0.2.10"),
    )
    thread = threading.Thread(target=review.serve_forever, daemon=True)
    thread.start()
    try:
        yield review
    finally:
        review.shutdown()
        thread.join(timeout=5)
        review.server_close()


@pytest.mark.parametrize(
    "host",
    [
        "review.example.net",
        "REVIEW.EXAMPLE.NET",
        "Review.Example.Net:443",
        "review.example.net:65535",
        "PINNED.EXAMPLE.NET:8443",
        "192.0.2.10",
        "192.0.2.10:8080",
    ],
)
def test_allowed_hosts_on_review_routes(proxy_server: ReviewServer, host: str) -> None:
    for route in ("/", "/api/review", "/media"):
        status, headers, body = request(
            proxy_server, "GET", route + "?t=test-token", host=host, token=None
        )
        assert status == 200
        assert "Location" not in headers
        if route == "/":
            assert b"Loading review" in body
        elif route == "/api/review":
            assert json.loads(body)["media"]["sha256"] == SHA
        else:
            assert body == VIDEO * 8
    assert proxy_server._http.server_address[0] == "127.0.0.1"
    assert request(proxy_server, "GET", "/api/review")[0] == 200
    assert request(
        proxy_server, "GET", "/api/review", host=f"localhost:{proxy_server.port}"
    )[0] == 200


def test_allowed_hosts_reject_other_names_and_ports(proxy_server: ReviewServer) -> None:
    for host in (
        "evil-review.example.net",
        "review.example.net.evil",
        "evil-review.example.net:443",
        "review.example.net.evil:443",
        "pinned.example.net",
        "pinned.example.net:443",
        "pinned.example.net:8444",
        "review.example.net:0",
        "review.example.net:65536",
        "review.example.net:",
        "review.example.net:abc",
        "review.example.net.",
        "review.example.net/path",
        "review.example.net@evil.example",
        "https://review.example.net",
        "other.example.net",
        "",
    ):
        for route in ("/", "/api/review", "/media"):
            status, _, body = request(proxy_server, "GET", route, host=host)
            assert status == 403
            assert json.loads(body)["error"]["message"] == "invalid Host header"


@pytest.mark.parametrize("token", [None, "wrong"])
def test_allowed_host_still_requires_token(proxy_server: ReviewServer, token: str | None) -> None:
    for route in ("/api/review", "/media"):
        status, _, body = request(
            proxy_server, "GET", route, host="review.example.net", token=token
        )
        assert status == 403
        assert json.loads(body)["error"]["message"] == "invalid review token"
    assert request(
        proxy_server,
        "POST",
        "/api/comments?t=test-token",
        {"text": "Point", "start_frame": 0},
        host="review.example.net",
        token=token,
    )[0] == 403


def test_allowed_host_media_range(proxy_server: ReviewServer) -> None:
    status, headers, body = request(
        proxy_server,
        "GET",
        "/media",
        host="REVIEW.EXAMPLE.NET:8443",
        headers={"Range": "bytes=0-99"},
    )
    assert status == 206
    assert body == (VIDEO * 8)[:100]
    assert headers["Content-Range"] == f"bytes 0-99/{len(VIDEO * 8)}"
    assert headers["Content-Length"] == "100"
    assert headers["Accept-Ranges"] == "bytes"


def test_allowed_host_comments_and_media_change(proxy_server: ReviewServer) -> None:
    host = "review.example.net:443"
    status, _, body = request(
        proxy_server, "POST", "/api/comments", {"text": "Point", "start_frame": 2}, host=host
    )
    assert status == 201
    comment = json.loads(body)
    assert ReviewStore(proxy_server.store.root).comments(SHA)[0].id == comment["id"]
    assert json.loads(request(proxy_server, "GET", "/api/review")[2])["comments"] == [comment]
    proxy_server.media.path.write_bytes(b"changed synthetic bytes")
    for method, route, payload in (
        ("GET", "/media", None),
        ("HEAD", "/media", None),
        ("POST", "/api/comments", {"text": "New", "start_frame": 0}),
        ("PATCH", "/api/comments/c1", {"text": "Updated"}),
        ("DELETE", "/api/comments/c1", None),
    ):
        assert request(proxy_server, method, route, payload, host=host)[0] == 409
    assert json.loads(
        request(proxy_server, "GET", "/api/review", host=host)[2]
    )["media_changed"] is True
    assert request(proxy_server, "GET", "/api/export.json", host=host)[0] == 200
    assert proxy_server.store.comments(SHA)[0].text == "Point"


@pytest.mark.parametrize(
    "entry",
    [
        "", " ", "review example.net", "review.example.net ", "https://review.example.net",
        "review.example.net/path", "review.example.net?x=1", "review.example.net#fragment",
        "*.example.net", "review.example.net:*", "user@review.example.net", "[::1]",
        "review_example.net", ".example.net", "review..example.net", "-review.example.net",
        "review-.example.net", "review.example.net:", "review.example.net:0",
        "review.example.net:65536", "review.example.net:-1", "review.example.net:abc",
        "review.example.net:443:80", "review.example.net:１２３",
    ],
)
def test_invalid_allowed_hosts_fail_before_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    media = ReviewMedia(tmp_path / "sample.mp4", "sample.mp4", None, None, False, "h264", 64, 36)
    monkeypatch.setattr(
        server_module, "_QuietThreadingHTTPServer",
        lambda *_args: pytest.fail("invalid host reached socket binding"),
    )
    with pytest.raises(ValidationError, match="host") as raised:
        ReviewServer(media, SHA, None, ReviewStore(tmp_path / "data"), allowed_hosts=(entry,))
    assert raised.value.suggested_action


def test_allowed_hosts_do_not_allow_other_bind_addresses(tmp_path: Path) -> None:
    media = ReviewMedia(tmp_path / "sample.mp4", "sample.mp4", None, None, False, "h264", 64, 36)
    with pytest.raises(ValueError, match="must bind to 127.0.0.1"):
        ReviewServer(
            media, SHA, None, ReviewStore(tmp_path / "data"),
            host="0.0.0.0", allowed_hosts=("review.example.net",),
        )


def test_comments_and_errors(server: ReviewServer) -> None:
    status, _, body = request(server, "POST", "/api/comments", {"text": "Later", "start_frame": 4})
    assert status == 201
    later = json.loads(body)
    assert later["id"] == "c1"
    assert later["kind"] == "fix"
    status, _, body = request(
        server, "POST", "/api/comments", {"label": "like", "text": "First", "start_frame": 1}
    )
    assert status == 201
    first = json.loads(body)
    assert first["kind"] == "keep" and first["target"] is None and "label" not in first
    status, _, body = request(server, "GET", "/api/review")
    assert status == 200
    review = json.loads(body)
    assert [item["id"] for item in review["comments"]] == [first["id"], later["id"]]
    assert review["media"] == {
        "name": "sample.mp4",
        "sha256": SHA,
        "duration_s": 12.0,
        "fps": "30000/1001",
        "cfr": True,
        "codec": "h264",
        "width": 64,
        "height": 36,
    }
    assert review["previous"] is None
    assert review["media_changed"] is False
    status, _, body = request(server, "PATCH", "/api/comments/c1", {"text": "Updated"})
    assert status == 200 and json.loads(body)["text"] == "Updated"
    assert request(server, "DELETE", "/api/comments/c2")[0] == 204
    assert len(json.loads(request(server, "GET", "/api/review")[2])["comments"]) == 1
    status, _, body = request(server, "POST", "/api/comments", {"text": "", "start_frame": 0})
    assert status == 400 and json.loads(body)["error"]["type"] == "validation_error"
    assert request(server, "PATCH", "/api/comments/c99", {"text": "Gone"})[0] == 404
    assert request(server, "DELETE", "/api/comments/c0")[0] == 404
    assert request(server, "GET", "/api/comments")[0] == 405
    assert request(server, "GET", "/api/comments")[1]["Allow"] == "POST"
    assert request(server, "POST", "/api/comments", headers={"Content-Length": "65537"})[0] == 413
    assert request(server, "POST", "/api/comments", headers={"Content-Length": "bad"})[0] == 400
    assert request(server, "POST", "/api/comments", [1, 2])[0] == 400


def test_exports_and_previous(server: ReviewServer) -> None:
    server.store.set_reviewer("Alex Reviewer")
    server.store.add_comment(SHA, server.media.facts(SHA), {"text": "Current", "start_frame": 3})
    old_facts = MediaFacts("earlier.mp4", OLD_SHA, 8.0, Fraction(24), True)
    server.store.add_comment(OLD_SHA, old_facts, {"text": "Earlier", "start_frame": 1})
    assert request(server, "GET", "/api/export.txt?version=previous")[0] == 404
    server.previous_sha = OLD_SHA
    status, headers, body = request(server, "GET", "/api/export.txt")
    assert status == 200
    assert headers["Content-Disposition"] == (
        'attachment; filename="sample-feedback.txt"; '
        "filename*=UTF-8''sample-feedback.txt"
    )
    assert body.decode() == format_text(
        server.media.facts(SHA), server.store.comments(SHA), "Alex Reviewer", date.today()
    )
    status, headers, body = request(server, "GET", "/api/export.json")
    assert status == 200
    assert headers["Content-Disposition"] == (
        'attachment; filename="sample-feedback.json"; '
        "filename*=UTF-8''sample-feedback.json"
    )
    document = json.loads(body)
    assert document["format"] == "yap-studio-feedback"
    assert document["version"] == 2
    assert document["media"]["sha256"] == SHA
    assert document["comments"][0]["id"] == "c1"
    status, _, body = request(server, "GET", "/api/export.txt?version=previous")
    assert status == 200 and b"Earlier" in body and b"Current" not in body
    assert b"earlier.mp4" in body
    status, headers, body = request(server, "GET", "/api/export.json?version=previous")
    assert status == 200
    assert headers["Content-Disposition"] == (
        'attachment; filename="earlier-feedback.json"; '
        "filename*=UTF-8''earlier-feedback.json"
    )
    assert json.loads(body)["media"]["name"] == "earlier.mp4"
    previous = json.loads(request(server, "GET", "/api/review")[2])["previous"]
    assert previous["sha256"] == OLD_SHA and previous["count"] == 1
    assert "Earlier" in previous["text"]
    assert previous["comments"][0]["kind"] == "fix"
    assert previous["comments"][0]["target"] is None
    assert "label" not in previous["comments"][0]


def test_kind_target_round_trip_and_patch_through_api_and_exports(server: ReviewServer) -> None:
    records = []
    for kind, target, frame in (("keep", "element", 1), ("fix", "caption", 4)):
        status, _, body = request(server, "POST", "/api/comments", {
            "kind": kind, "target": target, "text": f"{kind} note", "start_frame": frame,
            "end_frame": frame + 3,
        })
        assert status == 201
        record = json.loads(body)
        assert record["kind"] == kind and record["target"] == target and "label" not in record
        records.append(record)
    status, _, body = request(server, "GET", "/api/review")
    assert status == 200
    assert json.loads(body)["comments"] == records
    status, _, body = request(server, "GET", "/api/export.txt")
    assert status == 200
    assert b"2 comments: 1 fix, 1 keep" in body
    assert b"[Keep: graphic] keep note" in body and b"[Fix: caption] fix note" in body
    status, _, body = request(server, "GET", "/api/export.json")
    assert status == 200
    exported = json.loads(body)
    assert exported["version"] == 2 and exported["comments"] == records

    status, _, body = request(server, "PATCH", "/api/comments/c1", {"kind": "fix"})
    assert status == 200
    patched = json.loads(body)
    assert patched["kind"] == "fix"
    for key in ("target", "text", "start_frame", "end_frame", "start_s", "end_s", "created_at"):
        assert patched[key] == records[0][key]
    for target in ("story", "speech", "video", None):
        status, _, body = request(server, "PATCH", "/api/comments/c1", {"target": target})
        assert status == 200
        patched = json.loads(body)
        assert patched["target"] == target and patched["kind"] == "fix"
        assert patched["text"] == "keep note" and patched["start_frame"] == 1
        assert json.loads(request(server, "GET", "/api/review")[2])["comments"][0] == patched
    assert "[Fix] keep note" in json.loads(request(server, "GET", "/api/review")[2])["text"]


@pytest.mark.parametrize("kind", ["change", "like", "other", None])
def test_bad_kind_is_400_without_saving(server: ReviewServer, kind: object) -> None:
    status, _, body = request(server, "POST", "/api/comments", {
        "kind": kind, "text": "Note", "start_frame": 0,
    })
    assert status == 400
    assert json.loads(body)["error"]["message"] == "Bad value for kind: must be 'keep' or 'fix'"
    assert server.store.comments(SHA) == []


def test_saved_frame_comments_follow_variable_rate_media(server: ReviewServer) -> None:
    server.store.add_comment(
        SHA, server.media.facts(SHA),
        {"text": "Saved frame range", "start_frame": 30, "end_frame": 60},
    )
    server.media = replace(server.media, cfr=False)

    status, _, body = request(server, "GET", "/api/review")
    assert status == 200
    review = json.loads(body)
    assert review["comments"][0]["start_frame"] is None
    assert review["comments"][0]["end_frame"] is None
    assert "0:01.0-0:02.0 [Fix] Saved frame range" in review["text"]

    status, _, body = request(server, "GET", "/api/export.txt")
    assert status == 200
    assert b"0:01.0-0:02.0 [Fix] Saved frame range" in body
    status, _, body = request(server, "GET", "/api/export.json")
    assert status == 200
    assert json.loads(body)["comments"][0]["start_frame"] is None


def test_saved_seconds_comments_follow_constant_rate_media_and_can_be_patched(
    server: ReviewServer,
) -> None:
    variable_facts = replace(server.media.facts(SHA), fps=None, cfr=False)
    server.store.add_comment(
        SHA, variable_facts, {"text": "Saved seconds", "start_s": 1.2, "end_s": 2.2}
    )

    status, _, body = request(server, "GET", "/api/review")
    assert status == 200
    comment = json.loads(body)["comments"][0]
    assert (comment["start_frame"], comment["end_frame"]) == (35, 66)
    assert (comment["start_s"], comment["end_s"]) == (1.1678, 2.2022)
    assert request(server, "GET", "/api/export.txt")[0] == 200
    status, _, body = request(server, "GET", "/api/export.json")
    assert status == 200
    assert json.loads(body)["comments"][0]["start_frame"] == 35

    status, _, body = request(server, "PATCH", "/api/comments/c1", {"text": "Patched"})
    assert status == 200
    assert json.loads(body)["start_frame"] == 35
    assert json.loads(body)["text"] == "Patched"


def test_seconds_only_comment_past_duration_can_be_patched_on_constant_rate_media(
    server: ReviewServer,
) -> None:
    server.media = replace(server.media, duration_s=1.0, fps=Fraction(30))
    variable_facts = replace(server.media.facts(SHA), fps=None, cfr=False)
    server.store.add_comment(SHA, variable_facts, {"text": "Saved seconds", "start_s": 1.05})

    status, _, body = request(server, "PATCH", "/api/comments/c1", {"text": "Patched"})

    assert status == 200
    comment = json.loads(body)
    assert comment["text"] == "Patched"
    assert comment["start_frame"] == 29
    assert comment["start_s"] == 0.9667


def test_previous_comments_follow_previous_review_facts(server: ReviewServer) -> None:
    old_facts = MediaFacts("earlier.mp4", OLD_SHA, 8.0, Fraction(24), True)
    server.store.add_comment(OLD_SHA, old_facts, {"text": "Frame note", "start_frame": 24})
    server.store.add_comment(
        OLD_SHA, replace(old_facts, cfr=False), {"text": "Seconds note", "start_s": 2.0}
    )
    server.previous_sha = OLD_SHA

    status, _, body = request(server, "GET", "/api/review")
    assert status == 200
    previous = json.loads(body)["previous"]
    assert [item["start_frame"] for item in previous["comments"]] == [None, None]
    assert "Frame note" in previous["text"]
    status, _, body = request(server, "GET", "/api/export.json?version=previous")
    assert status == 200
    assert all(item["start_frame"] is None for item in json.loads(body)["comments"])


def test_export_filename_is_safe_in_header(server: ReviewServer) -> None:
    server.media = replace(server.media, name='a"b\r\nc é.mp4')
    status, headers, _ = request(server, "GET", "/api/export.txt")
    assert status == 200
    disposition = headers["Content-Disposition"]
    assert "\r" not in disposition and "\n" not in disposition
    assert disposition == (
        'attachment; filename="a_b__c _-feedback.txt"; '
        "filename*=UTF-8''a%22b%0D%0Ac%20%C3%A9-feedback.txt"
    )


@pytest.mark.parametrize(
    ("range_header", "expected", "content_range"),
    [
        (None, VIDEO, None),
        ("bytes=2-5", VIDEO[2:6], f"bytes 2-5/{len(VIDEO)}"),
        ("bytes=30-", VIDEO[30:], f"bytes 30-{len(VIDEO) - 1}/{len(VIDEO)}"),
        ("bytes=-4", VIDEO[-4:], f"bytes {len(VIDEO) - 4}-{len(VIDEO) - 1}/{len(VIDEO)}"),
    ],
)
def test_media_ranges(
    server: ReviewServer, range_header: str | None, expected: bytes, content_range: str | None
) -> None:
    headers = {"Range": range_header} if range_header else None
    status, returned, body = request(server, "GET", "/media", headers=headers)
    assert status == (206 if range_header else 200)
    assert body == expected
    assert returned["Content-Length"] == str(len(expected))
    assert returned["Accept-Ranges"] == "bytes"
    assert returned.get("Content-Range") == content_range
    assert returned["Content-Type"] == "video/mp4"
    status, returned, body = request(server, "HEAD", "/media", headers=headers)
    assert status == (206 if range_header else 200)
    assert returned["Content-Length"] == str(len(expected))
    assert body == b""


def test_media_stream_stops_when_open_file_changes(
    tmp_path: Path,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"a" * (3 * 256 * 1024))
    media = ReviewMedia(video, video.name, None, None, False, "h264", 64, 36)
    review = object.__new__(ReviewServer)
    review.media = media
    review.sha256 = SHA
    review.previous_sha = None
    review.store = ReviewStore(tmp_path / "data")
    review.token = "test-token"
    review.port = 0
    review._media_identity = file_identity(video)
    handler_type = review._handler_type()
    handler = object.__new__(handler_type)
    handler.command = "GET"
    handler.path = "/media"
    handler.headers = {"Host": "127.0.0.1:0", "X-Yap-Token": "test-token"}
    lengths: list[int] = []
    handler._headers = lambda _status, _kind, length, _extra: lengths.append(length)

    class Client:
        def __init__(self) -> None:
            self.received = bytearray()

        def write(self, chunk: bytes) -> None:
            self.received.extend(chunk)
            if len(self.received) == 256 * 1024:
                video.write_bytes(b"b" * (3 * 256 * 1024))

    client = Client()
    handler.wfile = client
    handler._route()
    assert lengths == [3 * 256 * 1024]
    assert client.received == b"a" * (256 * 1024)
    assert len(client.received) < lengths[0]
    assert handler.close_connection is True

    next_handler = object.__new__(handler_type)
    next_handler.command = "GET"
    next_handler.path = "/media"
    next_handler.headers = handler.headers
    errors: list[tuple[int, str]] = []
    next_handler._error = lambda status, kind, _message: errors.append((status, kind))
    next_handler._route()
    assert errors == [(409, "media_changed")]


@pytest.mark.parametrize("range_header", ["bytes=999-", "bytes=-0", "bytes=5-2", "bytes=0-1,3-4"])
def test_bad_ranges(server: ReviewServer, range_header: str) -> None:
    status, headers, _ = request(server, "GET", "/media", headers={"Range": range_header})
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{len(VIDEO)}"


def test_replaced_media_blocks_playback_and_comment_writes(server: ReviewServer) -> None:
    assert json.loads(request(server, "GET", "/api/review")[2])["media_changed"] is False
    assert request(server, "GET", "/media")[2] == VIDEO
    assert request(server, "HEAD", "/media")[0] == 200
    status, _, body = request(server, "POST", "/api/comments", {"text": "Old", "start_frame": 0})
    assert status == 201
    server.media.path.write_bytes(b"different video bytes")

    for method in ("GET", "HEAD"):
        status, headers, body = request(server, method, "/media")
        assert status == 409
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        if method == "GET":
            assert json.loads(body)["error"]["code"] == "media_changed"
        else:
            assert body == b""
    for method, route, payload in (
        ("POST", "/api/comments", {"text": "New", "start_frame": 0}),
        ("PATCH", "/api/comments/c1", {"text": "Updated"}),
        ("DELETE", "/api/comments/c1", None),
    ):
        status, _, body = request(server, method, route, payload)
        assert status == 409
        error = json.loads(body)["error"]
        assert error == {
            "type": "media_changed",
            "code": "media_changed",
            "message": (
                "The video changed on disk after this review started. "
                "Restart yap-studio review to review the new version."
            ),
        }
    assert json.loads(request(server, "GET", "/api/review")[2])["media_changed"] is True
    assert request(server, "GET", "/api/export.json")[0] == 200


def test_explicit_launch_identity_rejects_changed_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(VIDEO)
    launch_identity = file_identity(video)
    video.write_bytes(b"replacement")
    media = ReviewMedia(video, video.name, None, None, False, "h264", 64, 36)
    monkeypatch.setattr(
        server_module, "_QuietThreadingHTTPServer",
        lambda _address, _handler: SimpleNamespace(server_port=0),
    )
    review = ReviewServer(
        media, SHA, None, ReviewStore(tmp_path / "data"),
        token="test-token", media_identity=launch_identity,
    )
    handler = object.__new__(review._handler_type())
    handler.command = "GET"
    handler.path = "/media"
    handler.headers = {"Host": "127.0.0.1:0", "X-Yap-Token": "test-token"}
    errors: list[tuple[int, str]] = []
    handler._error = lambda status, kind, _message: errors.append((status, kind))
    handler._route()
    assert errors == [(409, "media_changed")]


def test_deleted_media_blocks_playback(server: ReviewServer) -> None:
    server.media.path.unlink()
    status, _, body = request(server, "GET", "/media")
    assert status == 409
    assert json.loads(body)["error"]["code"] == "media_changed"
    assert request(server, "HEAD", "/media")[0] == 409


def test_static_and_csp(
    server: ReviewServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, headers, body = request(server, "GET", "/", token=None)
    assert status == 200 and b"Loading review" in body
    assert headers["Content-Security-Policy"].startswith("default-src 'self'")
    for route in (
        "/static/../server.py",
        "/static/a/b.html",
        "/static/%2e%2e/server.py",
        "/static/secret.py",
        "/static/a\\b.html",
    ):
        assert request(server, "GET", route, token=None)[0] == 404
    (tmp_path / "example.mjs").write_text("export const x = 1;", encoding="utf-8")
    monkeypatch.setattr(server_module, "_STATIC", tmp_path)
    status, headers, _ = request(server, "GET", "/static/example.mjs", token=None)
    assert status == 200 and headers["Content-Type"] == "text/javascript; charset=utf-8"


def test_corrupt_review_returns_safe_500(server: ReviewServer) -> None:
    path = server.store.root / "reviews" / f"{SHA}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad", encoding="utf-8")
    status, _, body = request(server, "GET", "/api/review")
    assert status == 500
    assert json.loads(body)["error"]["type"] == "input_error"
    assert b"Traceback" not in body


def test_url_contains_token(server: ReviewServer) -> None:
    assert server._http.server_name == "127.0.0.1"
    parsed = urlsplit(server.url)
    assert parsed.hostname == "127.0.0.1"
    assert parsed.port == server.port
    assert parsed.query == "t=test-token"


@pytest.mark.parametrize("disconnect", [ConnectionAbortedError, TimeoutError])
def test_media_disconnect_is_quiet(server: ReviewServer, disconnect: type[OSError]) -> None:
    handler = object.__new__(server._handler_type())
    handler.command = "GET"
    handler.headers = {}
    handler._headers = lambda *_args, **_kwargs: None

    class DroppingWriter:
        def write(self, _data: bytes) -> None:
            raise disconnect("client disconnected")

    handler.wfile = DroppingWriter()
    handler._media()


def test_failed_error_response_is_quiet(server: ReviewServer) -> None:
    handler = object.__new__(server._handler_type())

    def broken_route() -> None:
        raise RuntimeError("route failed")

    def broken_response(*_args: object, **_kwargs: object) -> None:
        raise ConnectionResetError("client disconnected")

    handler._route = broken_route
    handler._error = broken_response
    handler._dispatch()


def test_handle_error_hides_disconnects_only(
    server: ReviewServer, capsys: pytest.CaptureFixture[str]
) -> None:
    try:
        raise ConnectionResetError("client disconnected")
    except ConnectionResetError:
        server._http.handle_error(None, ("127.0.0.1", 0))
    assert capsys.readouterr().err == ""

    try:
        raise RuntimeError("unexpected failure")
    except RuntimeError:
        server._http.handle_error(None, ("127.0.0.1", 0))
    assert "RuntimeError: unexpected failure" in capsys.readouterr().err
