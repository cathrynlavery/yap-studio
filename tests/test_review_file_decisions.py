"""Approve or send notes on each video of a file review, then move to the next one."""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from yap_studio import ffmpeg
from yap_studio.cli import _review_approval
from yap_studio.errors import InputError
from yap_studio.review.media import probe_media
from yap_studio.review.server import ReviewServer, ReviewVideo, decision_status
from yap_studio.review.store import ReviewStore, RoundConflictError

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("testsrc=size=64x36:rate=30", "testsrc2=size=64x36:rate=30",
           "color=c=blue:size=64x36:rate=30")


def make_clip(path: Path, source: str, *, crf: int = 23) -> Path:
    """Write one short synthetic clip."""
    ffmpeg.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", source,
                "-t", "1", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                str(path)], timeout_s=60)
    return path


def make_clips(folder: Path) -> list[Path]:
    """Write three synthetic clips for one review."""
    return [make_clip(folder / f"clip-{number}.mp4", source)
            for number, source in enumerate(SOURCES, 1)]


def start_server(store: ReviewStore, clips: list[Path]) -> ReviewServer:
    """Serve these clips as one file review on a loopback port."""
    videos = []
    for clip in clips:
        hashed = store.hash_media(clip)
        videos.append(ReviewVideo(probe_media(clip), hashed.sha256, hashed.previous_sha256,
                                  media_identity=hashed.identity))
    first = videos[0]
    server = ReviewServer(first.media, first.sha256, first.previous_sha, store,
                          token=first.token, media_identity=first.media_identity,
                          additional_videos=videos[1:])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop_server(server: ReviewServer) -> None:
    server.shutdown()
    server.server_close()


def call(server: ReviewServer, method: str, route: str, token: str,
         body: object = None) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    headers = {"Host": f"127.0.0.1:{server.port}", "X-Yap-Token": token}
    payload = None if body is None else json.dumps(body).encode()
    if payload is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(payload))})
    connection.request(method, route, body=payload, headers=headers)
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


def api(server: ReviewServer, number: int, method: str, route: str, body: object = None):
    status, payload = call(server, method, f"/v/{number}{route}",
                           server.videos[number - 1].token, body)
    assert status in (200, 201), payload
    return json.loads(payload)


def listing(server: ReviewServer) -> dict:
    status, payload = call(server, "GET", "/api/list", server.list_token)
    assert status == 200, payload
    return json.loads(payload)


def list_page(server: ReviewServer) -> str:
    status, payload = call(server, "GET", f"/?t={server.list_token}", server.list_token)
    assert status == 200
    return payload.decode()


def next_open(server: ReviewServer, current: int) -> int | None:
    videos = listing(server)["videos"]
    open_numbers = [item["number"] for item in videos
                    if item["status"]["state"] == "open" and item["number"] != current]
    after = [number for number in open_numbers if number > current]
    return (after or open_numbers or [None])[0]


@pytest.fixture
def review(tmp_path: Path):
    store = ReviewStore(tmp_path / "data")
    store.set_reviewer("Sample Reviewer")
    clips = make_clips(tmp_path)
    server = start_server(store, clips)
    try:
        yield server, store, clips
    finally:
        stop_server(server)


def test_approve_clip_one_then_notes_on_two_then_approve_three(review) -> None:
    server, _, _ = review
    page = list_page(server)
    assert page.count("Not reviewed") == 3
    start = re.search(r'id="start-reviewing" class="button-link primary" href="([^"]+)"', page)
    assert start and start.group(1).startswith("v/1/?t=")
    assert f"#list={server.list_token}" in start.group(1)

    review_one = api(server, 1, "GET", "/api/review")
    assert review_one["decision"]["approved"] is None
    approved = api(server, 1, "POST", "/api/approve", {})["decision"]["approved"]
    assert approved["reviewer"] == "Sample Reviewer" and approved["approved_at"].endswith("Z")
    assert next_open(server, 1) == 2
    statuses = [item["status"]["label"] for item in listing(server)["videos"]]
    assert statuses == ["Approved", "Not reviewed", "Not reviewed"]
    assert 'class="review-status approved">Approved<' in list_page(server)

    api(server, 2, "POST", "/api/comments", {"text": "Trim the start", "start_frame": 3})
    sent = api(server, 2, "POST", "/api/rounds", {})["decision"]
    assert sent["approved"] is None and sent["notes_sent"]["fixes"] == 1
    assert sent["unsent_fixes"] == 0
    assert next_open(server, 2) == 3

    api(server, 3, "POST", "/api/approve", {})
    assert next_open(server, 3) is None
    page = list_page(server)
    assert "Every video has a decision." in page and "Start reviewing" not in page
    assert [item["status"]["label"] for item in listing(server)["videos"]] == [
        "Approved", "Notes sent (1 fix)", "Approved"]

    status, payload = call(server, "GET", "/v/1/api/export.json", server.videos[0].token)
    first = json.loads(payload)
    assert status == 200 and first["approved"] == approved and first["notes_sent"] is None
    status, payload = call(server, "GET", "/v/2/api/export.json", server.videos[1].token)
    second = json.loads(payload)
    assert second["approved"] is None
    assert second["notes_sent"]["fixes"] == 1
    assert second["notes_sent"]["comments"] == [
        {"id": item["id"], "revision": item["updated_at"], "kind": "fix"}
        for item in second["comments"]]
    assert [item["kind"] for item in second["comments"]] == ["fix"]


def test_undo_approval_and_send_needs_a_fix(review) -> None:
    server, store, _ = review
    api(server, 1, "POST", "/api/approve", {})
    undone = api(server, 1, "DELETE", "/api/approve")["decision"]
    assert undone["approved"] is None
    assert listing(server)["videos"][0]["status"]["label"] == "Not reviewed"

    api(server, 1, "POST", "/api/comments", {"text": "Keep this", "kind": "keep",
                                             "start_frame": 0})
    status, payload = call(server, "POST", "/v/1/api/rounds", server.videos[0].token, {})
    assert status == 409 and json.loads(payload)["error"]["code"] == "no_fix_notes"
    api(server, 1, "POST", "/api/comments", {"text": "Fix this", "start_frame": 2})
    assert api(server, 1, "GET", "/api/review")["decision"]["unsent_fixes"] == 1
    api(server, 1, "POST", "/api/approve", {})
    sent = api(server, 1, "POST", "/api/rounds", {})["decision"]
    # Sending fixes replaces an approval: the video needs another version.
    assert sent["approved"] is None and sent["notes_sent"]["fixes"] == 1
    status, _ = call(server, "POST", "/v/1/api/rounds", server.videos[0].token, {})
    assert status == 409
    # Editing a sent Fix makes it unsent again.
    fix = next(item for item in api(server, 1, "GET", "/api/review")["comments"]
               if item["kind"] == "fix")
    store.update_comment(server.videos[0].sha256, server.videos[0].media.facts(
        server.videos[0].sha256), fix["id"], {"text": "Fix this better"})
    assert api(server, 1, "GET", "/api/review")["decision"]["unsent_fixes"] == 1


def test_list_api_needs_the_list_token(review) -> None:
    server, _, _ = review
    assert call(server, "GET", "/api/list", server.videos[0].token)[0] == 403
    assert call(server, "POST", "/api/list", server.list_token, {})[0] == 405
    assert call(server, "POST", "/v/1/api/approve", server.videos[1].token, {})[0] == 403
    status, _ = call(server, "POST", "/v/1/api/approve", server.videos[0].token, {"x": 1})
    assert status == 400


def test_reencoded_clip_does_not_keep_its_approval(review, tmp_path: Path) -> None:
    server, store, clips = review
    approved_sha = server.videos[0].sha256
    api(server, 1, "POST", "/api/approve", {})
    stop_server(server)
    make_clip(clips[0], SOURCES[0], crf=35)
    again = start_server(store, clips)
    try:
        assert again.videos[0].sha256 != approved_sha
        decision = api(again, 1, "GET", "/api/review")["decision"]
        assert decision["approved"] is None and decision["previous_approved"] is True
        assert listing(again)["videos"][0]["status"]["label"] == "Not reviewed"
        assert _review_approval(store, again.videos[0]) is None
        assert store.decision(approved_sha)["approved"] is not None
    finally:
        stop_server(again)


def test_single_file_review_approves_without_a_list(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data")
    clip = make_clip(tmp_path / "only.mp4", SOURCES[0])
    server = start_server(store, [clip])
    try:
        assert server.list_token is None
        status, payload = call(server, "POST", "/api/approve", server.token, {})
        assert status == 200 and json.loads(payload)["decision"]["approved"]["reviewer"] is None
        assert _review_approval(store, server.videos[0]) == json.loads(
            payload)["decision"]["approved"]
        assert call(server, "GET", "/api/list", server.token)[0] == 404
    finally:
        stop_server(server)


def test_decision_status_labels() -> None:
    assert decision_status(None) is None
    assert decision_status({"approved": {"approved_at": "x", "reviewer": None},
                            "notes_sent": None})["label"] == "Approved"
    assert decision_status({"approved": None, "notes_sent": {"fixes": 2}})["label"] == (
        "Notes sent (2 fixes)")
    assert decision_status({"approved": None, "notes_sent": None})["label"] == "Not reviewed"


def test_invalid_decision_file_is_refused(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data")
    sha = "c" * 64
    path = tmp_path / "data" / "decisions" / f"{sha}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "yap-studio/review-decision@1", "sha256": sha,
                                "approved": {"approved_at": "soon"}, "notes_sent": None}))
    with pytest.raises(InputError):
        store.decision(sha)
    with pytest.raises(RoundConflictError):
        ReviewStore(tmp_path / "other").send_media_notes(sha)


@pytest.mark.skipif(
    not os.environ.get("CI") and (
        not os.environ.get("YAP_STUDIO_PUPPETEER") or not os.environ.get("YAP_STUDIO_CHROME")),
    reason="set YAP_STUDIO_PUPPETEER and YAP_STUDIO_CHROME for the real browser flow",
)
def test_browser_approves_and_opens_the_next_video(review, tmp_path: Path) -> None:
    assert os.environ.get("YAP_STUDIO_PUPPETEER"), "Puppeteer is required in CI"
    assert os.environ.get("YAP_STUDIO_CHROME"), "Chrome is required in CI"
    server, store, _ = review
    result_file = tmp_path / "result.json"
    shots = os.environ.get("YAP_STUDIO_APPROVE_SHOTS", "")
    completed = subprocess.run(
        ["node", str(ROOT / "tests/js/review-approve-browser.mjs"), server.list_url,
         str(result_file), shots],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(result_file.read_text())
    assert result["errors"] == []
    assert result["undoShown"] is True
    assert result["approveBlockedByDraft"] is True
    assert result["final"] == ["Approved", "Notes sent (1 fix)", "Approved"]
    assert result["allDecided"] == "Every video has a decision."
    for key in ("phoneUndo", "phoneApprove"):
        box = result[key]
        assert box["width"] >= box["panel"] * 0.85, box
    assert store.decision(server.videos[0].sha256)["approved"] is None
    assert store.decision(server.videos[1].sha256)["notes_sent"]["fixes"] == 1
    assert store.decision(server.videos[2].sha256)["approved"] is not None
