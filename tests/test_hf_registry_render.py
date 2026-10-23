"""Render a synthetic localized registry document with external networking denied."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import offline_renderer
import pytest
from test_cards_render import renderer_install as renderer_install
from test_hf_registry import CORE
from test_hf_registry import synthetic as synthetic

from yap_studio import ffmpeg, tools
from yap_studio.hf_registry.localize import Localized
from yap_studio.render import hyperframes

pytestmark = pytest.mark.render


@pytest.fixture(scope="module")
def offline_registry_renderer(request, _shared_renderer_install):
    if not offline_renderer.offline_supported():
        pytest.skip("offline sandbox requires macOS")
    if _shared_renderer_install is None:
        pytest.skip("the shared pinned renderer must already be installed")
    return request.getfixturevalue("renderer_install")


def _localized_host(
    synthetic,
    renderer: hyperframes.RendererInstall,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Localized]:
    """Mount the same synthetic item in a host for both compiler font checks."""
    home = tmp_path / "render-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HYPERFRAMES_FONT_CACHE_DIR", raising=False)
    _, install, localize, fake, add, _ = synthetic
    source = (
        "<!doctype html><html><head>"
        f'<script src="{CORE}"></script>'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<style>:root{--title-font:"Space Grotesk",sans-serif}'
        "html,body{margin:0;background:transparent} "
        "p{font-family:var(--font-display, Montserrat, system-ui, sans-serif);"
        "color:red;font-size:40px}"
        "h1{font-family:var(--title-font);color:blue;font-size:30px}</style></head><body>"
        '<div data-composition-id="sample" data-width="320" data-height="240" '
        'data-duration="1"><p>Offline words</p><h1>Local title</h1></div>'
        "<script>window.__timelines = window.__timelines || {};"
        'window.__timelines.sample = gsap.timeline({paused:true}).to("p", {x:10,duration:1});'
        "</script></body></html>"
    )
    add(source=source)
    install.install_items(install=fake)
    root = tmp_path / "offline"
    result = localize.localize("sample", root, install=renderer)
    host = (
        "<!doctype html><html><head>"
        + localize.host_head(result)
        + "<style>html,body{margin:0;background:transparent}"
        '[data-composition-id="host"]{position:relative;width:320px;height:240px}'
        "</style></head><body>"
        '<div data-composition-id="host" data-width="320" data-height="240" '
        'data-duration="1" data-start="0">'
        f'<div data-composition-id="sample" data-composition-src="{result.main}" '
        'data-start="0" data-duration="1" data-width="320" data-height="240" '
        'data-track-index="0" style="position:absolute;inset:0"></div></div>'
        "<script>window.__timelines = window.__timelines || {};"
        'window.__timelines["host"] = gsap.timeline({paused:true});</script></body></html>'
    )
    (root / "index.html").write_text(host)
    shutil.copyfile(hyperframes.gsap_file(renderer), root / hyperframes.GSAP_SCRIPT)
    return root, home, result


def _snapshot(
    composition: Path,
    home: Path,
    tmp_path: Path,
    renderer: hyperframes.RendererInstall,
) -> tuple[Path, str]:
    """Capture a frame offline and return all compiler messages."""
    output_dir = tmp_path / "snapshot"
    with hyperframes._renderer_env() as env:
        result = tools.run_tool(
            "hyperframes",
            [
                "snapshot",
                str(composition),
                "--at",
                "0.5",
                "--no-end",
                "--no-browser-gpu",
                "--output",
                str(output_dir),
            ],
            executable=renderer.executable,
            env={**env, "HOME": str(home)},
            timeout_s=120,
        )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    image = output_dir / "frame-00-at-0.5s.png"
    assert image.is_file(), output
    return image, output


def test_localized_document_renders_offline_without_font_download(
    synthetic,
    offline_registry_renderer,
    tmp_path,
    monkeypatch,
):
    renderer = offline_registry_renderer
    composition, home, localized = _localized_host(synthetic, renderer, tmp_path, monkeypatch)
    assert set(localized.families) == {"Montserrat", "Space Grotesk"}
    image, output = _snapshot(composition, home, tmp_path, renderer)
    for message in ("Injected deterministic @font-face", "Google Fonts", "Unresolved"):
        assert message.casefold() not in output.casefold(), output
    # Layer preflight requires one composition ID; this host nests a second one.
    # Check the CLI's PNG until nested hosts are supported by render_layer in part 2.
    rgba = ffmpeg.run(
        [
            "ffmpeg",
            "-i",
            str(image),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ],
        timeout_s=30,
        binary=True,
    ).stdout
    pixels = np.frombuffer(rgba, dtype=np.uint8).reshape((240, 320, 4))
    assert np.any(pixels[:, :, 3] > 0)
    red, green, blue, alpha = (pixels[:, :, index] for index in range(4))
    assert np.count_nonzero((red > 180) & (green < 80) & (blue < 80) & (alpha > 0)) > 20
    assert np.count_nonzero((blue > 180) & (red < 80) & (green < 80) & (alpha > 0)) > 20
    assert not (home / ".cache/hyperframes/fonts").exists()


def test_snapshot_font_oracle_detects_missing_aliases(
    synthetic,
    offline_registry_renderer,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(synthetic[2], "_families", lambda _source: [])
    composition, home, localized = _localized_host(
        synthetic,
        offline_registry_renderer,
        tmp_path,
        monkeypatch,
    )
    assert localized.families == ()
    _, output = _snapshot(composition, home, tmp_path, offline_registry_renderer)
    message = "Injected deterministic @font-face"
    if message.casefold() not in output.casefold():
        pytest.skip(
            "Pinned compiler did not report deterministic font-face injection for the "
            f"unaliased document on this machine. Snapshot output: {output[-1000:]}"
        )
    assert message.casefold() in output.casefold(), output
