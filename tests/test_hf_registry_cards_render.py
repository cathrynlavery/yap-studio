"""Registry card pixels, clipping, timing and repeated offline renders."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import offline_renderer
import pytest
from test_cards_render import _alpha_frames, _rgba_frames
from test_cards_render import renderer_install as renderer_install
from test_hf_registry import CORE
from test_hf_registry import synthetic as synthetic
from test_hf_registry_cards import card_fields, duration_source
from test_hf_registry_render import offline_registry_renderer as offline_registry_renderer

from yap_studio import cards
from yap_studio.gallery import live
from yap_studio.hf_registry import catalog, install
from yap_studio.render import hyperframes
from yap_studio.style import resolve

pytestmark = pytest.mark.render


def _live_snapshot(composition, plan, renderer, tmp_path):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/index.html":
                source = (composition / "index.html").read_text()
                data = live.live_html(source, {}, "/runtime.js").encode()
                content_type = "text/html"
            elif self.path == "/runtime.js":
                data = live.runtime_file(renderer).read_bytes()
                content_type = "text/javascript"
            elif self.path == "/gsap.min.js":
                data = hyperframes.gsap_file(renderer).read_bytes()
                content_type = "text/javascript"
            else:
                return super().do_GET()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(composition)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    image = tmp_path / "live.png"
    try:
        result = subprocess.run(
            ["/usr/bin/sandbox-exec", "-f",
             str(offline_renderer._profile(tmp_path / "live-offline")), "node",
             str(Path(__file__).parent / "fixtures/hf_card_capture.mjs"),
             str(renderer.prefix / "node_modules"), renderer.browser_path,
             f"http://127.0.0.1:{server.server_port}/index.html",
             str(plan.width), str(plan.height), str(image)],
            capture_output=True, text=True, check=False, timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    (tmp_path / "live-probe.json").write_text(json.dumps(report, indent=2))
    return _alpha_frames(image, replace(plan, frames=1)), report


def owner_style(kind):
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(card_kinds_on=[kind], card_width_px=960,
                                 accent_color="#F34E3F", paper_color="#F6F1E7",
                                 ink_color="#1C1B19", font="Geist")
    style.data["hook"]["font"] = "Geist"
    style.data["internal"].update(key_text_top_px=180, key_text_bottom_px=540)
    return style


def verify_twice(kind, slots, tmp_path, renderer):
    plan = cards.card_layer_plan(kind, slots, owner_style(kind), frames=60)
    hashes = []
    for repeat in ("first", "second"):
        out = tmp_path / f"{repeat}.mov"
        cards.render_card(plan, tmp_path / repeat, out, install=renderer)
        rgba = _rgba_frames(out, plan)
        hashes.append([hashlib.sha256(frame.tobytes()).hexdigest() for frame in rgba])
        alpha = rgba[:, :, :, 3]
        panel = alpha[:, cards.PAD_TOP:-cards.PAD_BOTTOM, cards.PAD_X:-cards.PAD_X]
        assert np.count_nonzero(panel[30]) > 100
        assert not np.any(alpha[:, 0]) and not np.any(alpha[:, -1])
        assert not np.any(alpha[:, :, 0]) and not np.any(alpha[:, :, -1])
        # The last sample precedes D by one frame; match the shared power2.in fade.
        assert alpha[-1].max() < 80
        assert alpha[-1].mean() < 5
        assert alpha[-1].sum() < 0.3 * alpha.sum(axis=(1, 2)).max()
    assert hashes[0] == hashes[1]


def _copy_gate_item(name, tmp_path, monkeypatch, shared_cache_root):
    # Gate items are read-only; all renderer and localization writes stay in the test cache.
    commit = catalog.load_catalog()["registry"]["commit"]
    gate = (shared_cache_root / "registry" /
            f"hyperframes-{hyperframes.HYPERFRAMES_VERSION}-{commit[:12]}")
    with monkeypatch.context() as gate_cache:
        gate_cache.setattr(install, "items_root", lambda: gate)
        source = install.installed(name)
    if source is None:
        pytest.skip(f"{name} is not installed in the gate cache; run yap-studio registry install")
    target = install.item_dir(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    monkeypatch.setenv("HOME", str(tmp_path / "render-home"))


@pytest.mark.parametrize(("name", "slots"), [
    ("char-slam-explode", {"text": "Ship it now!"}),
    ("conic-progress-ring", {"percent": 72}),
])
def test_installed_kind(name, slots, tmp_path, monkeypatch, _shared_cache_root,
                        offline_registry_renderer):
    _copy_gate_item(name, tmp_path, monkeypatch, _shared_cache_root)
    verify_twice(f"hf:{name}", slots, tmp_path, offline_registry_renderer)


@pytest.mark.parametrize(("name", "ink_token"), [
    ("char-slam-explode", "--brand"), ("conic-progress-ring", "--fg"),
])
def test_installed_card_contrast(
    name, ink_token, tmp_path, monkeypatch, _shared_cache_root, offline_registry_renderer,
):
    from yap_studio.checks.caption_contrast import OPAQUE_ALPHA, word_contrast
    from yap_studio.checks.model import Thresholds

    _copy_gate_item(name, tmp_path, monkeypatch, _shared_cache_root)
    entry = catalog.item(name)
    plan = cards.card_layer_plan(f"hf:{name}", entry["preview"], owner_style(f"hf:{name}"),
                                 frames=90)
    frame = 54  # 1.8 s: entrance settled, before either exit or the shared fade.
    layers = {}
    for readability in ("none", "halo"):
        plan.registry["entry"]["readability"] = readability
        out = tmp_path / f"{readability}.mov"
        cards.render_card(plan, tmp_path / readability, out, install=offline_registry_renderer)
        layers[readability] = _rgba_frames(out, plan)[frame].copy()
    ink_color = np.frombuffer(bytes.fromhex(plan.registry["tokens"][ink_token][1:]), dtype=np.uint8)
    # Find the glyphs, excluding the ring and its track from the label's box.
    bare = layers["none"]
    ink = np.all(np.abs(bare[:, :, :3].astype(np.int16) - ink_color) <= 4, axis=2)
    ys, xs = np.nonzero(ink & (bare[:, :, 3] >= OPAQUE_ALPHA))
    assert len(xs) >= 40
    box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
           int(ys.max() - ys.min() + 1))
    backgrounds = {
        "light wall": np.full((plan.height, plan.width, 3), [238, 236, 230], dtype=np.uint8),
        "grey": np.full((plan.height, plan.width, 3), 128, dtype=np.uint8),
        "busy": np.random.default_rng(312).integers(
            0, 256, (plan.height, plan.width, 3), dtype=np.uint8),
    }
    measurements = {name: {} for name in backgrounds}
    for readability, layer in layers.items():
        alpha = layer[:, :, 3:] / 255
        for background_name, background in backgrounds.items():
            final = np.rint(layer[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
            measurements[background_name][readability] = word_contrast(final, layer, box)
    print(f"Registry contrast ({name}, frame {frame}): " + json.dumps(measurements, sort_keys=True))
    assert all(values["halo"] is not None
               and values["halo"] >= Thresholds().caption_contrast_warn
               for values in measurements.values()), measurements


@pytest.mark.parametrize("text", ["https://a.co", "//a.co/x"])
@pytest.mark.parametrize("playback", ["cli", "live"])
def test_url_shaped_text_reaches_item(
    synthetic, offline_registry_renderer, tmp_path, monkeypatch, text, playback,
):
    registry_catalog, registry_install, _, fake, add, _ = synthetic
    # Keep the expected text out of the source's script scan as well as its HTML.
    expected = json.dumps([ord(character) for character in text])
    source = (
        '<!doctype html><html><head><style>'
        'html,body{margin:0;background:transparent}'
        '#root{position:relative;width:100%;height:100%}'
        '.mark{display:none;position:absolute;left:40px;top:40px;width:200px;height:100px;'
        'background:red}</style>'
        f'<script src="{CORE}"></script></head><body>'
        '<div id="root" data-composition-id="sample" data-duration="2" '
        "data-composition-variables='[{\"id\":\"text\",\"type\":\"string\"},"
        "{\"id\":\"progress\",\"type\":\"number\"}]'><div class=\"mark\"></div></div>"
        '<script>(function(){var values=window.__hyperframes.getVariables();'
        f'if(values.text===String.fromCharCode(...{expected})){{'
        'var mark=document.getElementById("root").querySelector(".mark");'
        'mark.style.display="block";mark.textContent=values.text;}'
        'window.__timelines=window.__timelines||{};'
        'window.__timelines.sample=gsap.timeline({paused:true});})();</script></body></html>'
    )
    add(source=source)
    registry_catalog.item("sample").update(card_fields())
    registry_install.install_items(install=fake)
    monkeypatch.setenv("HOME", str(tmp_path / "render-home"))
    plan = cards.card_layer_plan("hf:sample", {"text": text},
                                 owner_style("hf:sample"), frames=60)
    if playback == "cli":
        out = tmp_path / "text.mov"
        cards.render_card(plan, tmp_path / "work", out, install=offline_registry_renderer)
        alpha = _alpha_frames(out, plan)
    else:
        from yap_studio.hf_registry import cards as registry_cards

        composition = registry_cards.prepare(plan, tmp_path / "work",
                                              install=offline_registry_renderer)
        hyperframes.preflight(composition, plan.frames, plan.fps)
        alpha, _ = _live_snapshot(composition, plan, offline_registry_renderer, tmp_path)
    panel = alpha[len(alpha) // 2, cards.PAD_TOP:-cards.PAD_BOTTOM, cards.PAD_X:-cards.PAD_X]
    assert np.count_nonzero(panel) > 100, "the item did not receive the exact creator text"


@pytest.mark.parametrize("template", [False, True])
def test_synthetic_item_holds_for_card_window(
    synthetic, offline_registry_renderer, tmp_path, monkeypatch, template,
):
    registry_catalog, registry_install, _, fake, add, _ = synthetic
    add(source=duration_source(template=template))
    registry_catalog.item("sample").update(card_fields())
    registry_install.install_items(install=fake)
    monkeypatch.setenv("HOME", str(tmp_path / "render-home"))
    plan = cards.card_layer_plan("hf:sample", {"text": "Words"},
                                 owner_style("hf:sample"), frames=90)
    out = tmp_path / "long.mov"
    cards.render_card(plan, tmp_path / "work", out, install=offline_registry_renderer)
    alpha = _alpha_frames(out, plan)
    assert np.count_nonzero(alpha[15]) > 100
    assert np.count_nonzero(alpha[75]) > 100, "the item ended before the shared fade-out"


def test_installed_ring_holds_for_six_second_card(
    tmp_path, monkeypatch, _shared_cache_root, offline_registry_renderer,
):
    _copy_gate_item("conic-progress-ring", tmp_path, monkeypatch, _shared_cache_root)
    kind = "hf:conic-progress-ring"
    plan = cards.card_layer_plan(kind, {"percent": 72}, owner_style(kind), frames=180)
    out = tmp_path / "long-ring.mov"
    cards.render_card(plan, tmp_path / "work", out, install=offline_registry_renderer)
    alpha = _alpha_frames(out, plan)
    assert np.count_nonzero(alpha[60]) > 100
    assert np.count_nonzero(alpha[135]) > 100, "the ring ended before the shared fade-out"


@pytest.mark.parametrize("mount", ["fill", "scale"])
@pytest.mark.parametrize("playback", ["cli", "live"])
def test_item_root_background_stays_inside_panel(
    synthetic, offline_registry_renderer, tmp_path, monkeypatch, mount, playback,
):
    registry_catalog, registry_install, _, fake, add, _ = synthetic
    size = "width:1920px;height:1080px" if mount == "scale" else "width:100%;height:100%"
    source = (
        '<!doctype html><html><head><style>'
        'html,body{margin:0;background:transparent}'
        f'#root{{position:relative;{size};background:#32a852}}'
        f'</style><script src="{CORE}"></script></head><body>'
        '<div id="root" data-composition-id="sample" data-duration="2" '
        "data-composition-variables='[{\"id\":\"text\",\"type\":\"string\"},"
        "{\"id\":\"progress\",\"type\":\"number\"}]'></div>"
        '<script>var root=document.getElementById("root");'
        'window.__itemRootProbe={id:root.id,composition:root.dataset.compositionId,'
        'duration:root.dataset.duration};'
        'window.__timelines=window.__timelines||{};'
        'window.__timelines.sample=gsap.timeline({paused:true});</script></body></html>'
    )
    add(source=source)
    entry = registry_catalog.item("sample")
    entry.update(card_fields(), mount=mount)
    if mount == "scale":
        entry.update(stage=[1920, 1080], content=[0, 0, 1920, 1080], scale_max=1)
    registry_install.install_items(install=fake)
    monkeypatch.setenv("HOME", str(tmp_path / "render-home"))
    plan = cards.card_layer_plan("hf:sample", {"text": "Words"},
                                 owner_style("hf:sample"), frames=60)
    report = None
    if playback == "cli":
        out = tmp_path / "background.mov"
        cards.render_card(plan, tmp_path / "work", out, install=offline_registry_renderer)
        alpha = _alpha_frames(out, plan)
    else:
        from yap_studio.hf_registry import cards as registry_cards

        composition = registry_cards.prepare(plan, tmp_path / "work",
                                              install=offline_registry_renderer)
        alpha, report = _live_snapshot(composition, plan, offline_registry_renderer, tmp_path)
    assert np.count_nonzero(alpha[len(alpha) // 2]) > 1000
    assert not np.any(alpha[:, :cards.PAD_TOP]), "item background reached the top padding"
    assert not np.any(alpha[:, -cards.PAD_BOTTOM:]), "item background reached the bottom padding"
    assert not np.any(alpha[:, :, :cards.PAD_X]), "item background reached the left padding"
    assert not np.any(alpha[:, :, -cards.PAD_X:]), "item background reached the right padding"
    if report is not None:
        assert report["globalRoot"] is None
        assert report["itemRoot"]["id"] == "hf-mount"
        assert report["itemRoot"]["composition"] == "sample"
        mount_root = next(root for root in report["roots"] if root.get("id") == "hf-mount")
        assert mount_root["data-hf-authored-duration"] == "1.999"
        assert report["hostBackground"] == "rgba(0, 0, 0, 0)"


@pytest.mark.parametrize("playback", ["cli", "live"])
def test_hf_0_8_184_bg_fallback_stays_clear(
    synthetic, offline_registry_renderer, tmp_path, monkeypatch, playback,
):
    registry_catalog, registry_install, _, fake, add, _ = synthetic
    source = (
        '<!doctype html><html><head><style>'
        'html,body{margin:0;background:transparent}'
        '#root{position:relative;width:100%;height:100%}'
        '.clip{position:absolute;inset:0;width:100%;height:100%;'
        'background:var(--bg, #090d16);display:flex;align-items:center;justify-content:center}'
        '.tile{width:200px;height:100px;background:var(--surface, #151b2b);'
        'border-radius:var(--radius, 16px)}'
        f'</style><script src="{CORE}"></script></head><body>'
        '<div id="root" data-composition-id="sample" data-duration="2" '
        "data-composition-variables='[{\"id\":\"text\",\"type\":\"string\"},"
        "{\"id\":\"progress\",\"type\":\"number\"}]'>"
        '<div class="clip"><div class="tile"></div></div></div>'
        '<script>window.__timelines=window.__timelines||{};'
        'window.__timelines.sample=gsap.timeline({paused:true});</script></body></html>'
    )
    add(source=source)
    registry_catalog.item("sample").update(card_fields())
    registry_install.install_items(install=fake)
    monkeypatch.setenv("HOME", str(tmp_path / "render-home"))
    plan = cards.card_layer_plan("hf:sample", {"text": "Words"},
                                 owner_style("hf:sample"), frames=60)
    if playback == "cli":
        out = tmp_path / "backdrop.mov"
        cards.render_card(plan, tmp_path / "work", out, install=offline_registry_renderer)
        alpha = _alpha_frames(out, plan)
    else:
        from yap_studio.hf_registry import cards as registry_cards

        composition = registry_cards.prepare(plan, tmp_path / "work",
                                              install=offline_registry_renderer)
        alpha, _ = _live_snapshot(composition, plan, offline_registry_renderer, tmp_path)
    panel = alpha[len(alpha) // 2, cards.PAD_TOP:-cards.PAD_BOTTOM, cards.PAD_X:-cards.PAD_X]
    x, y = (panel.shape[1] - 200) // 2, (panel.shape[0] - 100) // 2
    assert np.any(panel[y:y + 100, x:x + 200]), "the tile did not draw"
    outside = np.ones(panel.shape, dtype=bool)
    outside[y:y + 100, x:x + 200] = False
    assert not np.any(panel[outside]), "the --bg fallback painted outside the tile"
    assert panel[y, x] == 0, "the tile lost its own radius fallback"


def test_synthetic_scale(synthetic, offline_registry_renderer, tmp_path, monkeypatch):
    registry_catalog, registry_install, _, fake, add, _ = synthetic
    source = (
        '<!doctype html><html><head><style>'
        'html,body{margin:0;background:transparent}'
        '#root{position:relative;width:1920px;height:1080px}'
        '.mark{position:absolute;left:400px;top:300px;width:200px;height:100px;'
        'background:var(--brand,red)}</style>'
        f'<script src="{CORE}"></script></head><body>'
        '<div id="root" data-composition-id="sample" data-width="1920" data-height="1080" '
        'data-start="0" data-duration="2" '
        "data-composition-variables='[{\"id\":\"text\",\"type\":\"string\"},"
        "{\"id\":\"progress\",\"type\":\"number\"}]'><div class=\"mark\"></div></div>"
        '<script>window.__timelines=window.__timelines||{};'
        'window.__timelines.sample=gsap.timeline({paused:true});</script></body></html>'
    )
    add(source=source)
    registry_catalog.items()["sample"].update(
        card_fields(), mount="scale", stage=[1920, 1080],
        content=[400, 300, 200, 100], scale_max=1)
    registry_install.install_items(install=fake)
    monkeypatch.setenv("HOME", str(tmp_path / "render-home"))
    verify_twice("hf:sample", {"text": "Ship it"}, tmp_path, offline_registry_renderer)
