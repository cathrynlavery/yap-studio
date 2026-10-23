"""Real home choices and request delivery on phone and desktop."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from yap_studio.homepage import bridge
from yap_studio.homepage.server import HomeServer

pytestmark = pytest.mark.skipif(
    not os.environ.get("CI") and (
        not os.environ.get("YAP_STUDIO_PUPPETEER") or not os.environ.get("YAP_STUDIO_CHROME")),
    reason="set YAP_STUDIO_PUPPETEER and YAP_STUDIO_CHROME for the real browser flow",
)
ROOT = Path(__file__).resolve().parents[1]


def test_home_shape_phone_and_desktop(tmp_path: Path) -> None:
    workspace = tmp_path / "My videos"
    workspace.mkdir()
    inbox = workspace / "inbox"
    inbox.mkdir()
    for name, size in [("computer.mp4", "320x180"), ("phone.mp4", "180x320")]:
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=s={size}:r=10:d=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(inbox / name),
        ], stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True)
    os.utime(inbox / "computer.mp4", (1, 1))
    server = HomeServer(workspace)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result_file = tmp_path / "result.json"
    shots = os.environ.get("YAP_STUDIO_HOME_SHOTS", str(tmp_path / "shots"))
    try:
        result = subprocess.run([
            "node", str(ROOT / "tests/js/home-browser.mjs"), server.url,
            str(result_file), shots,
        ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        results = json.loads(result_file.read_text())
        assert [row["width"] for row in results] == [390, 1440]
        requests = list(reversed(bridge.requests(workspace)))
        assert len(requests) == 2
        for row, captions in zip(requests, [True, False], strict=True):
            assert row["payload"] == {
                "video_type": "talking_head", "shape": "horizontal", "captions": captions,
                "takes": ["computer.mp4"], "broll_sources": [], "album": None,
            }
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
