"""Copy chapter text from real Home and review pages on phone and desktop."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_versions import _render_media

from yap_studio.feedback.versions import save_version
from yap_studio.homepage.server import HomeServer
from yap_studio.receipt import sha256_file
from yap_studio.review.launch import version_video
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("YAP_STUDIO_PUPPETEER") or not os.environ.get("YAP_STUDIO_CHROME"),
    reason="set YAP_STUDIO_PUPPETEER and YAP_STUDIO_CHROME for browser tests",
)


@pytest.mark.parametrize("prior_qa", [False, True])
def test_chapter_copy_phone_desktop_and_frozen_versions(tmp_path, feedback_project, prior_qa):
    workspace = tmp_path / "workspace"
    (workspace / "inbox").mkdir(parents=True)
    project = workspace / "projects/demo"
    project.parent.mkdir()
    feedback_project.rename(project)
    manifest_path = project / "project.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shape"] = "horizontal"
    manifest_path.write_text(json.dumps(manifest))
    final = project / "final/final.mp4"
    final.parent.mkdir()
    _render_media(final, 120, label="Synthetic final")
    text = "0:00 Opening\n0:15 Plan the story\n1:00 Publish\n"
    report = project / "final/receipt.json"
    document = {"schema": "yap-studio/stage-receipt@1", "status": "verified", "verify": {
        "status": "pass", "findings": [], "youtube_chapters": text}, "outputs": [
            {"path": "final/final.mp4", "sha256": sha256_file(final)}]}
    report.write_text(json.dumps(document))
    if prior_qa:
        (project / "qa").mkdir(exist_ok=True)
        (project / "qa/report.json").write_text(json.dumps({
            "status": "pass", "findings": [], "youtube_chapters": "0:00 Old chapter list\n"}))
    save_version(project, media=final)
    # Later edits must not change the selected saved version's list.
    document["verify"].pop("youtube_chapters")
    report.write_text(json.dumps(document))
    store = ReviewStore(tmp_path / "review-store")
    video = version_video([str(project)], store, lambda values: [Path(value) for value in values])
    review = ReviewServer(video.media, video.sha256, video.previous_sha, store,
                          series=video.series, media_identity=video.media_identity)
    home = HomeServer(workspace)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (home, review)]
    for thread in threads:
        thread.start()
    try:
        script = Path(__file__).parent / "fixtures/youtube_chapters_browser.mjs"
        result = subprocess.run(["node", str(script), home.url, review.url, text],
                                capture_output=True, text=True, timeout=90, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        for server in (home, review):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=10)


@pytest.mark.parametrize("mismatch", [False, True])
def test_invalid_or_unbound_final_drops_older_list(feedback_project, mismatch):
    from yap_studio.youtube_chapters import version_chapters

    project = feedback_project
    manifest = json.loads((project / "project.json").read_text())
    manifest["shape"] = "horizontal"
    (project / "project.json").write_text(json.dumps(manifest))
    final = project / "final/final.mp4"
    final.parent.mkdir()
    _render_media(final, 120, label="Synthetic final")
    (project / "qa").mkdir(exist_ok=True)
    text = "0:00 Old list\n0:10 Middle\n0:20 End\n"
    (project / "qa/report.json").write_text(json.dumps({
        "status": "pass", "findings": [], "youtube_chapters": text}))
    report = {"status": "warn", "findings": []}
    if mismatch:
        report["youtube_chapters"] = text
    (project / "final/receipt.json").write_text(json.dumps({
        "schema": "yap-studio/stage-receipt@1", "status": "verified", "verify": report,
        "outputs": [{"path": "final/final.mp4", "sha256": "0" * 64 if mismatch
                     else sha256_file(final)}]}))
    assert version_chapters(save_version(project, media=final)) is None
