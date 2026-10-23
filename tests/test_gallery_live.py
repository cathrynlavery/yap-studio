"""Live layers use the same plans and compositions as a render."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from test_hf_registry import synthetic as synthetic
from test_hf_registry_cards import KIND
from test_hf_registry_cards import registry_card as registry_card

from yap_studio import cards, text_rules
from yap_studio.brand_words import set_words
from yap_studio.errors import DependencyError, ValidationError
from yap_studio.faces.model import Box
from yap_studio.gallery import live, still
from yap_studio.gallery.server import GalleryServer
from yap_studio.render import hyperframes
from yap_studio.stages.captions import (
    caption_band_geometry,
    caption_offsets,
    prepare_layer_plan,
)
from yap_studio.style import resolve

_CAPTURE = Path(__file__).parent / "fixtures" / "live_capture.mjs"


def _install(root: Path) -> hyperframes.RendererInstall:
    runtime = root / "node_modules/hyperframes/dist/hyperframe.runtime.iife.js"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("window.__runtime=1")
    return hyperframes.RendererInstall("0", root, root / "bin", None, "v22")


def _request(server: GalleryServer, method: str, route: str, body: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=15)
    headers = ({"X-Gallery-Token": server.token, "Content-Type": "application/json"}
               if method == "POST" else {})
    connection.request(method, route, body=json.dumps(body) if body else None, headers=headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_registry_live_defaults_use_catalog_preview(registry_card, tmp_path):
    preview = registry_card[0].item("sample")["preview"]
    assert live.card_slots(KIND) == preview
    returned = live.card_slots(KIND)
    returned["text"] = "Changed"
    assert live.card_slots(KIND) == preview
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = [KIND]
    layer, = live.live_layers(style, live.sample_words("Ship it now"), tmp_path,
                              card=KIND, only="card")
    assert layer.variables == {} and layer.template == KIND
    host = (layer.composition / "index.html").read_text()
    assert "Ship it now" in host and 'data-variable-values=' in host


def test_registry_live_preview_extends_short_hold(registry_card, tmp_path):
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(hold_s=1, card_kinds_on=[KIND])
    layer, = live.live_layers(style, live.sample_words("Ship it now"), tmp_path,
                              card=KIND, only="card")
    assert layer.frames == 45 and layer.fps == 30


def test_bleep_masks_creator_text_in_preview_plans(tmp_path: Path) -> None:
    style = resolve("preset:clean_minimal")
    style.data["music"]["bleep_swears"] = True
    layers = live.live_layers(style, live.sample_words("Fuck this."), tmp_path,
                              title=("Fuck", ""), card="quote",
                              slots={"quote": "Shit", "name": "Speaker"})
    captions, title, card = layers
    shown = [word["text"] for phrase in json.loads(captions.variables["phrases"])
             for word in phrase["words"]]
    assert shown == ["F***", "this."]
    assert title.variables["plain"] == "F***"
    assert card.variables["quote"] == "S***"


def test_sample_words_and_render_compositions(tmp_path: Path) -> None:
    words = live.sample_words(live.SAMPLE_SENTENCE)
    assert words[0].start > 0
    assert all(a.end <= b.start for a, b in zip(words, words[1:], strict=False))
    with pytest.raises(ValidationError):
        live.sample_words("   ")
    style = resolve("preset:warm_editorial")
    layers = live.live_layers(style, words, tmp_path / "live", card="text_tag",
                              slots={"text": live.sentence_card_text(words, style)})
    assert [layer.name for layer in layers] == ["captions", "title", "card"]
    plan = live.caption_plan(style, words)
    prepared = prepare_layer_plan(plan, tmp_path / "render")
    assert (prepared / "index.html").read_bytes() == (
        layers[0].composition / "index.html").read_bytes()
    assert layers[0].variables == plan.variables
    assert (layers[0].x, layers[0].y, layers[0].width, layers[0].height) == (
        plan.x, plan.y, plan.width, plan.height)
    assert [layer.name for layer in live.live_layers(style, words, tmp_path / "one",
                                                     only="captions")] == ["captions"]
    assert live.card_slots("stat") == json.loads(
        (cards.template_dir("stat") / "preview.json").read_text())["slots"]


@pytest.mark.parametrize(("sentence", "limit", "expected"), [
    ("Ship it now. Keep going.", 12, ["Ship", "it", "now."]),
    ("Ship it now! Keep going.", 100, ["Ship", "it", "now!"]),
    ("Ship it now? Keep going.", 100, ["Ship", "it", "now?"]),
    ("Ship it now", 7, ["Ship", "it"]),
    ("Ship it now", 6, ["Ship"]),
    ("Unfittableword next", 5, []),
    ("Ship it now", 0, []),
])
def test_leading_preview_words_fit_limit_and_stop_at_sentence_end(sentence, limit, expected):
    assert live.leading_words(live.sample_words(sentence), limit) == expected


def test_leading_preview_words_allow_an_empty_preview():
    assert live.leading_words([], 12) == []


def test_feed_preview_keeps_captions_above_the_backdrop(tmp_path: Path) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["reel_feed"]
    layers = live.live_layers(style, live.sample_words(live.SAMPLE_SENTENCE), tmp_path,
                              card="reel_feed")
    assert [layer.name for layer in layers] == ["card", "captions", "title"]


@pytest.mark.parametrize("kind", ["reel_feed", "playing_grid"])
@pytest.mark.parametrize("hold_s", [1, 2, 5])
def test_live_card_preview_uses_default_preview_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, hold_s: int,
) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(hold_s=hold_s, card_kinds_on=[kind])
    expected = cards.default_preview_frames(kind, live.card_slots(kind), style)
    preview = cards.preview_videos
    calls = []

    def videos(template, slots, work, *, frames, fps):
        calls.append(frames)
        values = preview(template, slots, work, frames=frames, fps=fps)
        for clip in values.get("clips", []):
            assert cards.video_info(Path(clip["path"])).video_duration_s >= expected / fps
        return values

    monkeypatch.setattr(cards, "preview_videos", videos)
    layer, = live.live_layers(style, live.sample_words("Small steps."), tmp_path,
                               only="card", card=kind)
    assert layer.frames == expected
    assert calls == [expected]


def test_file_drop_live_preview_keeps_poster(tmp_path: Path) -> None:
    from test_file_drop import clip_source

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["hold_s"] = 1.5
    layer, = live.live_layers(style, live.sample_words("Small steps."), tmp_path / "live",
                               only="card", card="file_drop",
                               slots={"clip": clip_source(tmp_path), "playback": "play",
                                      "poster_at": 0})
    assert json.loads(layer.variables["play_window"]) == {}
    assert (layer.composition / "poster.jpg").is_file()


def test_injection_is_serve_only(tmp_path: Path) -> None:
    style = resolve("preset:electric")
    gsap = tmp_path / "gsap.min.js"
    gsap.write_text("window.gsap={}")
    server = GalleryServer(still.sample_still(), style.data, tmp_path / "work",
                           install=_install(tmp_path / "renderer"), gsap=gsap)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = _request(server, "POST", "/api/layers", {
            "style": style.data, "tile": {"tab": "captions", "look": "pill"}})
        assert status == 200
        src = json.loads(body)["layers"][0]["src"]
        status, headers, body = _request(server, "GET", src)
        assert status == 200 and b"window.__hfVariables=" in body
        assert b'src="/hf/runtime.js"' in body
        assert "frame-ancestors 'self'" in headers["Content-Security-Policy"]
        digest = src.split("/")[2]
        composition = server._cache[digest][1][0].composition
        assert b"__hfVariables" not in (composition / "index.html").read_bytes()
        assert _request(server, "GET", "/hf/runtime.js")[2] == b"window.__runtime=1"
        assert _request(server, "GET", src + "../../etc/passwd")[0] == 404
        _, _, body = _request(server, "POST", "/api/layers", {
            "style": style.data, "tile": {"tab": "cards", "card": "stat"}})
        card_src = json.loads(body)["layers"][0]["src"]
        assert b"window.__hfVariables=" in _request(server, "GET", card_src)[2]
        assert _request(server, "GET", card_src + "gsap.min.js")[2] == b"window.gsap={}"
        card_key = card_src.split("/")[2]
        card = server._cache[card_key][1][0]
        assert b"__hfVariables" not in (card.composition / "index.html").read_bytes()
    finally:
        server.shutdown()
        server.server_close()


def test_html_escapes_variables_and_runtime_is_pinned(tmp_path: Path) -> None:
    source = '<html><head><meta charset="utf-8"></head><body></body></html>'
    injected = live.live_html(source, {"text": "</script><b>"}, "/hf/runtime.js")
    assert "</script><b>" not in injected
    assert injected.index("__hfVariables") < injected.index("<meta")
    install = _install(tmp_path / "renderer")
    assert live.runtime_file(install).is_file()
    live.runtime_file(install).unlink()
    with pytest.raises(DependencyError):
        live.runtime_file(install)


def test_live_caption_face_avoidance_matches_render_planner(tmp_path: Path) -> None:
    from test_faces_caption_modes import _faces

    from yap_studio.faces.model import FaceBox

    style = resolve("preset:clean_minimal")
    style.data["captions"].update(position="middle", top_px=700, entry_ms=0)
    words = live.sample_words("Captions clear the creator's face.")
    fixed = live.caption_plan(style, words)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    boxes = [Box(400, geometry.y + 40, 200, 180)]
    faces = _faces(tuple((FaceBox(boxes[0], 0.9, "detected"),) for _ in range(fixed.frames)))
    expected = caption_offsets(json.loads(fixed.variables["phrases"]), geometry, faces, style)
    assert any(row["offset_px"] for row in expected)
    layers = live.live_layers(style, words, tmp_path, only="captions", face_boxes=boxes)
    offsets = [slot["offset_px"] for slot in json.loads(layers[0].variables["phrases"])]
    assert offsets == [row["offset_px"] for row in expected]
    style.data["captions"]["placement"] = "fixed"
    assert live.caption_plan(style, words, face_boxes=boxes) == fixed
    assert live.caption_plan(style, words, face_boxes=[]) == fixed


@pytest.mark.render
def test_caption_and_card_tiles_match_render(tmp_path: Path) -> None:
    """Capture each live tile at one moment and compare its rendered layer."""
    install = hyperframes.ensure_installed()
    node = shutil.which("node")
    assert node is not None and install.browser_path is not None
    style = resolve("preset:electric")
    server = GalleryServer(still.sample_still(), style.data, tmp_path / "work",
                           install=install, gsap=hyperframes.gsap_file(install))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for tile in ({"tab": "captions", "look": "pill"},
                     {"tab": "cards", "look": "paper", "card": "stat"}):
            _, _, body = _request(server, "POST", "/api/layers", {"style": style.data,
                                                                     "tile": tile})
            spec = json.loads(body)["layers"][0]
            digest = spec["src"].split("/")[2]
            layer = server._cache[digest][1][0]
            at = round(min(1.5, layer.frames / layer.fps * 0.6) * layer.fps)
            shot = tmp_path / f"live-{layer.name}.png"
            subprocess.run([node, str(_CAPTURE), str(install.prefix / "node_modules"),
                            install.browser_path,
                            f"http://127.0.0.1:{server.port}{spec['src']}",
                            str(layer.width), str(layer.height), str(at / layer.fps), str(shot)],
                           check=True, timeout=120, env={**os.environ})
            work = tmp_path / f"render-{layer.name}"
            shutil.copytree(layer.composition, work)
            movie = tmp_path / f"render-{layer.name}.mov"
            hyperframes.render_layer(work, movie, frames=layer.frames,
                                     variables=layer.variables, fps=layer.fps, checks=False)
            live_rgba = _rgba(shot, layer)
            render_rgba = _rgba(movie, layer, frame=at)
            alpha = live_rgba[..., 3:] / 255
            live_rgb = live_rgba[..., :3] * alpha + 128 * (1 - alpha)
            alpha = render_rgba[..., 3:] / 255
            render_rgb = render_rgba[..., :3] * alpha + 128 * (1 - alpha)
            diff = np.abs(live_rgb - render_rgb).max(axis=2)
            assert (diff > 48).mean() < 0.01
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def viewport_gallery(tmp_path: Path):
    install = hyperframes.ensure_installed()
    node = shutil.which("node")
    assert node is not None and install.browser_path is not None
    yy, xx = np.ogrid[:1920, :1080]
    person = np.zeros((1920, 1080, 4), dtype=np.uint8)
    person[((xx - 540) / 390) ** 2 + ((yy - 1600) / 650) ** 2 <= 1] = (70, 140, 120, 255)
    person[((xx - 540) / 170) ** 2 + ((yy - 740) / 230) ** 2 <= 1] = (215, 160, 113, 255)
    for eye in (480, 600):
        person[(xx - eye) ** 2 + (yy - 720) ** 2 <= 12 ** 2] = (30, 25, 20, 255)
    picture = np.full_like(person, (145, 125, 105, 255))
    picture[..., :3] = np.where(((xx // 180 + yy // 240) % 2)[..., None],
                               (145, 125, 105), (165, 150, 130))
    picture[person[..., 3] > 0] = person[person[..., 3] > 0]
    image, cutout = tmp_path / "picture.png", tmp_path / "person.png"
    for target, pixels in ((image, picture), (cutout, person)):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo",
                        "-pixel_format", "rgba", "-video_size", "1080x1920", "-i", "pipe:0",
                        "-frames:v", "1", "-threads", "1", str(target)],
                       input=pixels.tobytes(), check=True, timeout=30)
    server = GalleryServer(image, resolve("preset:clean_minimal").unbranded_data,
                           tmp_path / "work", install=install, gsap=hyperframes.gsap_file(install))
    server.update_preview({"still": str(image), "source": "take", "sentence": live.SAMPLE_SENTENCE,
                           "face_boxes": [{"x": 370, "y": 510, "w": 340, "h": 460}]})
    generation, _ = server.preview_image()
    server.update_cutout(generation, cutout)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, install
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.render
def test_caption_controls_fit_desktop_and_phone_viewports(viewport_gallery, tmp_path) -> None:
    """Run the shipped controls, engine layers and cutout previews in a real browser."""
    server, install = viewport_gallery
    node = shutil.which("node")
    assert node is not None
    subprocess.run(
        [node, str(_CAPTURE.with_name("gallery_viewport.mjs")),
         str(install.prefix / "node_modules"), install.browser_path, server.url, str(tmp_path)],
        check=True, timeout=120, env={**os.environ},
    )


@pytest.mark.render
def test_caption_spoken_live_builds_on_desktop_and_phone(viewport_gallery, tmp_path) -> None:
    server, install = viewport_gallery
    subprocess.run(
        ["node", str(_CAPTURE.with_name("gallery_viewport.mjs")),
         str(install.prefix / "node_modules"), install.browser_path, server.url, str(tmp_path),
         "spoken"], check=True, timeout=120, env={**os.environ},
    )


@pytest.mark.render
def test_card_motion_desktop_and_phone_walk(
    viewport_gallery, tmp_path, _shared_cache_root, monkeypatch,
) -> None:
    from test_hf_registry_cards_render import _copy_gate_item

    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    server, install = viewport_gallery
    subprocess.run(
        ["node", str(_CAPTURE.with_name("gallery_viewport.mjs")),
         str(install.prefix / "node_modules"), install.browser_path, server.url, str(tmp_path),
         "motion"], check=True, timeout=120, env={**os.environ},
    )


@pytest.mark.render
def test_registry_cards_fit_desktop_and_phone_viewports(
    viewport_gallery, tmp_path, _shared_cache_root, monkeypatch,
) -> None:
    """Reuse installed pilots without fetching items or changing the gate cache."""
    from yap_studio.hf_registry import install as registry_install

    server, install = viewport_gallery
    names = ("char-slam-explode", "conic-progress-ring")
    destinations = {name: registry_install.item_dir(name) for name in names}
    with monkeypatch.context() as shared:
        shared.setenv("YAP_STUDIO_CACHE", str(_shared_cache_root))
        sources = {name: registry_install.installed(name) for name in names}
    missing = [name for name, source in sources.items() if source is None]
    if missing:
        pytest.skip("registry tile walk needs installed gate-cache items: " + ", ".join(missing))
    for name, source in sources.items():
        shutil.copytree(source, destinations[name])
    node = shutil.which("node")
    assert node is not None
    subprocess.run(
        [node, str(_CAPTURE.with_name("gallery_viewport.mjs")),
         str(install.prefix / "node_modules"), install.browser_path, server.url, str(tmp_path),
         "registry"], check=True, timeout=120, env={**os.environ},
    )


def _rgba(source: Path, layer: live.LiveLayer, frame: int | None = None) -> np.ndarray:
    select = [] if frame is None else ["-vf", f"select=eq(n\\,{frame})"]
    data = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), *select,
                           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                          check=True, capture_output=True, timeout=120).stdout
    return np.frombuffer(data, np.uint8).reshape(layer.height, layer.width, 4).astype(float)


def test_default_live_preview_has_no_sample_card(tmp_path):
    words = live.sample_words("These are the creator's own words.")
    layers = live.live_layers(resolve("preset:clean_minimal"), words, tmp_path)
    assert [layer.name for layer in layers] == ["captions", "title"]


def test_sentence_card_uses_first_sentence_and_fits_registry_limit():
    style = resolve("preset:clean_minimal")
    words = live.sample_words("Your first sentence. Keep the second one out.")
    assert live.sentence_card_text(words, style) == "Your first sentence."
    words = live.sample_words(
        "This longer opening sentence has enough words to reach the text sign limit.")
    result = live.sentence_card_text(words, style)
    assert len(result) <= 48
    assert "This longer opening sentence has enough words to reach".startswith(result)


@pytest.mark.parametrize("look", ("paper", "minimal", "dark", "bright"))
@pytest.mark.parametrize("width", (200, 600, 960))
def test_long_first_sentence_sign_fits_engine_limits_in_every_card_style(tmp_path, look, width):
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(style=look, card_width_px=width)
    sentence = "Make a long sentence about useful tools that continues well beyond the sign limit."
    words = live.sample_words(sentence + " Keep this second sentence out.")
    text = live.sentence_card_text(words, style)
    assert text and sentence.startswith(text)
    assert sentence[len(text)] in " ."
    plan = cards.card_layer_plan("text_tag", {"text": text, "variant": "sign"}, style)
    assert plan.variables["text_px"] >= cards.MIN_TEXT_PX
    if len(text.split()) < len(sentence.split()):
        next_word = sentence.split()[len(text.split())]
        with pytest.raises(ValidationError, match="slots.text"):
            cards.card_layer_plan("text_tag", {"text": text + " " + next_word, "variant": "sign"},
                                 style)
    layers = live.live_layers(style, words, tmp_path, card="text_tag",
                              slots={"text": text, "variant": "sign"}, only="card")
    assert len(layers) == 1 and (layers[0].composition / "index.html").is_file()
    assert layers[0].variables["text"] == text
    assert layers[0].variables["look"] == cards.TAG_SIGN_LOOKS[look]


def test_sentence_sign_does_not_cut_an_unfittable_word():
    words = live.sample_words("X" * 60 + ". More words.")
    assert live.sentence_card_text(words, resolve("preset:clean_minimal")) is None


@pytest.mark.parametrize("preset", ("clean_minimal", "editorial_serif", "one_word"))
def test_poster_times_are_after_entry_and_before_exit(tmp_path, preset):
    style = resolve(f"preset:{preset}")
    words = live.sample_words("Make the first sentence visible in the options.")
    layers = live.live_layers(style, words, tmp_path, card="text_tag",
                              slots={"text": live.sentence_card_text(words, style),
                                     "variant": "sign"})
    for layer in layers:
        poster = live.poster_time(layer)
        assert 0 < poster < layer.frames / layer.fps
        if layer.name == "captions":
            phrases = json.loads(layer.variables["phrases"])
            assert any(phrase["start"] + layer.variables["entry_ms"] / 1000 <= poster
                       < phrase["end"] for phrase in phrases)
        elif layer.name == "title":
            duration = layer.frames / layer.fps
            assert min(0.4, duration / 2) <= poster <= duration - min(0.25, duration / 2)
        else:
            assert poster >= 1.01


@pytest.mark.parametrize("entrance", ("slide", "rise"))
@pytest.mark.parametrize("highlight", ("none", "slam"))
def test_short_first_word_poster_waits_for_rise(tmp_path, entrance, highlight):
    style = resolve("preset:clean_minimal")
    style.data["captions"].update(entrance=entrance, entry_ms=0, highlight_style=highlight)
    words = [live.CaptionWord("w1", "A", .3, .35), live.CaptionWord("w2", "clear", .35, .8),
             live.CaptionWord("w3", "sentence.", .8, 1.3)]
    layer = live.live_layers(style, words, tmp_path, only="captions")[0]
    poster = live.poster_time(layer)
    if entrance == "slide":
        assert poster == pytest.approx(.635 if highlight == "slam" else .325)
    elif highlight == "slam":
        assert poster == pytest.approx((.35 + 8 / layer.fps + .8) / 2)
    else:
        # Word two finishes its eight-frame rise after a two-frame stagger.
        assert poster == pytest.approx((.3 + (8 + 2) / layer.fps + .8) / 2)


def test_rise_poster_tracks_template_frame_constants():
    source = (Path(live.__file__).parents[1] / "templates/captions/template.html").read_text()
    for name in ("RISE_FRAMES", "RISE_STAGGER_FRAMES", "RISE_MIN_FRAMES"):
        assert f"const {name} = {getattr(live, '_' + name)};" in source


def test_kinetic_live_title_uses_sample_word_times_and_creator_ink(tmp_path: Path) -> None:
    style = resolve("preset:clean_minimal")
    style.data["hook"].update(style="kinetic", duration_s=3)
    style.data["captions"]["text_color"] = "#EEDDBB"
    words = live.sample_words("Small steps.")
    layer, = live.live_layers(style, words, tmp_path, only="title")
    assert layer.variables["look"] == "kinetic"
    assert layer.variables["ink_color"] == "#EEDDBB"
    assert json.loads(layer.variables["word_times"]) == [
        round(word.start * 30) / 30 for word in words]


def test_kinetic_live_title_rules_use_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    set_words(["SandPiper"])

    def aliases(vocabulary):
        assert tuple(vocabulary) == ("SandPiper",)
        return {"sand piper": "SandPiper"}

    monkeypatch.setattr(text_rules, "vocabulary_fixes", aliases)
    style = resolve("preset:clean_minimal")
    style.data["hook"].update(style="kinetic", duration_s=3)
    layer, = live.live_layers(style, live.sample_words("sand piper"), tmp_path,
                             only="title", title=("SandPiper", ""))
    assert json.loads(layer.variables["word_times"]) == [0.3]


@pytest.mark.parametrize("accent", ["#ABCDEF", ""])
def test_giant_live_captions_use_layout_settings(tmp_path: Path, accent: str) -> None:
    style = resolve("preset:clean_minimal")
    style.data["brand"]["accent_color"] = accent
    style.data["captions"].update(highlight_style="giant", size_px=83, max_words_per_phrase=8)
    words = live.sample_words(live.SAMPLE_SENTENCE)
    layer, = live.live_layers(style, words, tmp_path, only="captions")
    assert layer.variables["highlight_style"] == "giant"
    assert layer.variables["size_px"] == 124
    assert layer.variables["background_mode"] == "box"
    assert layer.variables["background_color"] == (accent or "#000000")
    assert all(len(slot["words"]) <= 3 for slot in json.loads(layer.variables["phrases"]))
    assert style.data["captions"]["size_px"] == 83


def test_chapter_live_preview_only_when_the_style_turns_it_on(tmp_path: Path) -> None:
    style = resolve("preset:clean_minimal")
    words = live.sample_words("Small steps.")
    assert [layer.name for layer in live.live_layers(style, words, tmp_path / "off",
                                                     only="chapter")] == ["captions"]
    style.data["chapters"]["style"] = "title_card"
    layers = live.live_layers(style, words, tmp_path / "on", only="chapter")
    assert [layer.name for layer in layers] == ["chapter"]
    assert layers[0].template == "chapter"
