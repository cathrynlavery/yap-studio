"""Real phone and desktop pin flows against a loopback review server."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import threading
from pathlib import Path

import pytest
from test_review_join_findings import feedback_project as feedback_project
from test_review_join_findings import join_project as join_project

from yap_studio.feedback.versions import save_version
from yap_studio.receipt import sha256_file
from yap_studio.review.launch import version_video
from yap_studio.review.media import probe_media
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("CI") and (
        not os.environ.get("YAP_STUDIO_PUPPETEER") or not os.environ.get("YAP_STUDIO_CHROME")),
    reason="set YAP_STUDIO_PUPPETEER and YAP_STUDIO_CHROME for the real browser flow",
)
ROOT = Path(__file__).resolve().parents[1]


def test_review_pin_phone_and_desktop(tmp_path: Path) -> None:
    assert os.environ.get("YAP_STUDIO_PUPPETEER"), "Puppeteer is required in CI"
    assert os.environ.get("YAP_STUDIO_CHROME"), "Chrome is required in CI"
    media = tmp_path / "portrait.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=0x345678:s=1080x1920:r=30:d=3", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(media)],
                   stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True)
    store = ReviewStore(tmp_path / "store")
    digest = sha256_file(media)
    server = ReviewServer(probe_media(media), digest, None, store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result_file = tmp_path / "result.json"
    shots = Path(os.environ.get("YAP_STUDIO_PIN_SHOTS", str(tmp_path / "shots")))
    try:
        result = subprocess.run(["node", str(ROOT / "tests/js/review-pin-browser.mjs"),
                                 server.url, str(result_file), str(shots)],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=120, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        results = json.loads(result_file.read_text())
        assert [item["mobile"] for item in results] == [True, False]
        assert all(item["stall"] and item["retry"] for item in results)
        comments = {comment.id: comment for comment in store.comments(digest)}
        assert len(comments) == 2
        for item in results:
            pin = comments[item["comment"]].pin
            assert pin["x"] == pytest.approx(item["x"], abs=0.01)
            assert pin["y"] == pytest.approx(item["y"], abs=0.01)
            assert pin["frame"] == comments[item["comment"]].start_frame == item["frame"]
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


@pytest.mark.parametrize(("agent", "cpu_throttle"), [
    pytest.param(None, 1, id="None"),
    pytest.param("codex", 1, id="codex"),
    pytest.param("codex", 8, id="codex-8x"),
])
def test_review_join_listens_phone_and_desktop(join_project, tmp_path, agent, cpu_throttle):
    root, _, _ = join_project
    if agent:
        folder = root / ".yap-studio"
        folder.mkdir(exist_ok=True)
        (folder / "setup.json").write_text(json.dumps({"agent": agent}))
    save_version(root, note="Story cut. Tightened the opening.")
    store = ReviewStore(tmp_path / "store")
    video = version_video([str(root)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, series=video.series)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result_file = tmp_path / "join-result.json"
    shots = tmp_path / "join-shots"
    try:
        result = subprocess.run(["node", str(ROOT / "tests/js/review-join-browser.mjs"),
                                 server.url, str(result_file), str(shots),
                                 "Codex" if agent == "codex" else "Claude Code",
                                 str(cpu_throttle)],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=120, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        results = json.loads(result_file.read_text())
        assert [item["mobile"] for item in results] == [True, False]
        assert all(len(item["played"]) == 2 for item in results)
        for item in results:
            image = Path(item["screenshot"]).read_bytes()
            assert image[:8] == b"\x89PNG\r\n\x1a\n"
            expected_size = (390, 844) if item["mobile"] else (1440, 900)
            assert struct.unpack(">II", image[16:24]) == expected_size
        assert store.comments(video.sha256) == []
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
