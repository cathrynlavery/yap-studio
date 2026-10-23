"""Real HTTP style and home flows at the two supported viewport widths."""

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from yap_studio.gallery.server import GalleryServer
from yap_studio.homepage.server import HomeServer
from yap_studio.render import hyperframes
from yap_studio.style import resolve

ROOT = Path(__file__).parents[1]


@pytest.mark.render
@pytest.mark.skipif(not os.environ.get("YAP_STUDIO_PUPPETEER"), reason="needs browser environment")
@pytest.mark.parametrize("width", [390, 1440])
def test_top_text_style_and_home_browser(tmp_path, monkeypatch, width):
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    workspace = tmp_path / "creator"
    (workspace / "inbox").mkdir(parents=True)
    image = workspace / "creator-still.jpg"
    shutil.copyfile(ROOT / "src/yap_studio/gallery/sample-still.jpg", image)
    prefix = Path(os.environ["YAP_STUDIO_PUPPETEER"]).parents[1]
    install = hyperframes.RendererInstall(hyperframes.HYPERFRAMES_VERSION, prefix,
                                         prefix / "node_modules/.bin/hyperframes",
                                         os.environ["YAP_STUDIO_CHROME"], "test")
    gallery = GalleryServer(image, resolve("preset:clean_minimal").data, tmp_path / "layers",
                            install=install, gsap=hyperframes.gsap_file(install),
                            workspace=workspace, preview={"source": "take", "generation": 0,
                                "sentence": "A clearer opening starts with your own words.",
                                "words": []})
    home = HomeServer(workspace)
    servers = (gallery, home)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        result = subprocess.run([
            shutil.which("node"), str(ROOT / "tests/fixtures/top_text_browser.mjs"),
            gallery.url, home.url, str(width),
        ], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        saved = json.loads((workspace / "style/my-style.json").read_text())
        assert saved["hook"]["answered"] is True
        assert saved["hook"]["style"] == "quiet_serif"
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)
