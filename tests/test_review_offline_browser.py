"""Real browser: download a review, review it with the server gone, and sync the note once."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest
from test_review_file_decisions import ROOT, start_server, stop_server

from yap_studio import ffmpeg
from yap_studio.review.media import probe_media
from yap_studio.review.server import ReviewServer, ReviewVideo
from yap_studio.review.store import ReviewStore

SOURCES = ("testsrc2=size=360x640:rate=30", "smptebars=size=360x640:rate=30")


def restart(server: ReviewServer, store: ReviewStore) -> ReviewServer:
    """Serve the same videos, tokens and port again, as a review restored on its old link."""
    first, *rest = server.videos
    again = ReviewServer(first.media, first.sha256, first.previous_sha, store,
                         port=server.port, token=first.token, media_identity=first.media_identity,
                         additional_videos=[ReviewVideo(probe_media(video.media.path), video.sha256,
                                                        video.previous_sha, video.token,
                                                        video.media_identity) for video in rest])
    again.list_token = server.list_token
    threading.Thread(target=again.serve_forever, daemon=True).start()
    return again


@pytest.mark.skipif(
    not os.environ.get("YAP_STUDIO_PLAYWRIGHT") and (
        not os.environ.get("CI") and (not os.environ.get("YAP_STUDIO_PUPPETEER")
                                      or not os.environ.get("YAP_STUDIO_CHROME"))),
    reason="set YAP_STUDIO_PLAYWRIGHT, or YAP_STUDIO_PUPPETEER and YAP_STUDIO_CHROME",
)
def test_browser_reviews_offline_and_syncs_the_note_once(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data")
    clips = [tmp_path / f"light-{number}.mp4" for number in (1, 2)]
    for clip, source in zip(clips, SOURCES, strict=True):
        # Small light review copies: portrait H.264, long enough to watch playback advance.
        ffmpeg.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", source,
                    "-t", "6", "-c:v", "libx264", "-b:v", "600k", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(clip)], timeout_s=60)
    server = start_server(store, clips)
    result_file = tmp_path / "result.json"
    process = subprocess.Popen(
        ["node", str(ROOT / "tests/js/review-offline-browser.mjs"), server.list_url,
         str(result_file)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "offline", process.stderr.read()
        stop_server(server)
        process.stdin.write("ok\n")
        process.stdin.flush()
        assert process.stdout.readline().strip() == "online", process.stderr.read()
        server = restart(server, store)
        process.stdin.write("ok\n")
        process.stdin.flush()
        process.wait(timeout=240)
        assert process.returncode == 0, process.stdout.read() + process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
        stop_server(server)
    result = json.loads(result_file.read_text())
    assert result["errors"] == []
    assert "All 2 videos ready offline" in result["ready"]
    assert [state.startswith("Ready offline") for state in result["states"]] == [True, True]
    assert result["offlineList"] == 2
    for video in result["videos"]:
        assert video["error"] is None and video["banner"] is True, video
        assert video["end"] > video["start"] + 0.5, video
    assert result["pending"].startswith("Waiting on this device")
    assert result["pendingAfterReload"] == "Offline note from the car"
    assert result["synced"].count("Offline note from the car") == 1
    texts = [comment.text for comment in store.comments(server.videos[1].sha256)]
    assert texts == ["Offline note from the car"]
    assert store.comments(server.videos[0].sha256) == []
