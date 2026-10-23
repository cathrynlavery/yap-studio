"""Exercise Home and portrait review in a fresh, independently launched Chrome."""

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from test_feedback_batch import _json
from test_feedback_batch import feedback_project as feedback_project

from yap_studio.feedback.versions import save_version
from yap_studio.homepage import bridge
from yap_studio.homepage import state as home_state
from yap_studio.homepage.server import HomeServer
from yap_studio.receipt import sha256_file
from yap_studio.review.launch import version_video
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore


@pytest.mark.skipif(not os.environ.get("YAP_STUDIO_PUPPETEER"), reason="needs Chrome and Puppeteer")
def test_fresh_user_home_and_review(feedback_project, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    project = workspace / "projects/demo"
    project.parent.mkdir(parents=True)
    source = os.environ.get("YAP_STUDIO_FRESH_PROJECT")
    if source:
        shutil.copytree(source, project)
    else:
        feedback_project.rename(project)
        media = project / "cut/cut.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
            "color=c=teal:size=180x320:rate=30", "-frames:v", "120", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(media),
        ], check=True, capture_output=True, timeout=30)
        receipt_path = project / "cut/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["outputs"][0]["sha256"] = sha256_file(media)
        _json(receipt_path, receipt)
        save_version(project, note="Story cut. Kept the opening and ending in your own words.")
    inbox = workspace / "inbox"
    inbox.mkdir()
    shutil.copyfile(project / "cut/cut.mp4", inbox / "take.mp4")
    if source:
        # Home is observed during the render; review is observed after its final is saved.
        versions = bridge.load_versions(project)
        with monkeypatch.context() as earlier:
            earlier.setattr(bridge, "load_versions", lambda _: [
                version for version in versions if not version.media["source"].startswith("final/")
            ])
            request = bridge.create_request(workspace, "final", {"project": "projects/demo"})
    else:
        request = bridge.create_request(workspace, "final", {"project": "projects/demo"})
    bridge.ack(workspace, request["id"], "claimed")
    monkeypatch.setattr(home_state.bridge, "verified_approval", lambda _: None)
    monkeypatch.setattr(home_state, "status", lambda _: {
        "video": {"type": "talking_head", "broll_sources": []},
        "stages": [{"name": "assemble", "state": "running", "progress": {
            "started": "2099-01-01T00:00:00Z", "step": "Rendering the final",
        }}],
    })
    store = ReviewStore(tmp_path / "review-data")
    video = version_video([str(project)], store, lambda _: [])
    review = ReviewServer(video.media, video.sha256, None, store, series=video.series)
    home = HomeServer(workspace)
    servers = [review, home]
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    shots = Path(os.environ.get("YAP_STUDIO_FRESH_SHOTS", str(tmp_path / "screenshots")))
    result = shots / "browser-results.json"
    try:
        completed = subprocess.run([
            "node", str(Path(__file__).parent / "js/fresh-user-browser.mjs"),
            review.url, home.url, str(shots), str(result),
        ], capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stderr
        checks = json.loads(result.read_text())
        assert checks and all(check["passed"] for check in checks)
    finally:
        for server, thread in zip(servers, threads, strict=True):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
