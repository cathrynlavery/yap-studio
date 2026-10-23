"""Real phone and desktop pin flows against a loopback review server."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from yap_studio.receipt import sha256_file
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
