"""The style gallery serves choices, saves a pick, and protects writes."""

from __future__ import annotations

import http.client
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import numpy as np
import pytest
from test_hf_registry import synthetic as synthetic
from test_hf_registry_cards import KIND, Elements, card_fields
from test_hf_registry_cards import registry_card as registry_card

from yap_studio import cards, grade
from yap_studio.gallery import live, still
from yap_studio.gallery import server as gallery_server
from yap_studio.gallery.server import GalleryServer
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.render import hyperframes
from yap_studio.render.hyperframes import RendererInstall
from yap_studio.review import phone
from yap_studio.style import current_default_data, resolve, save_pick


@pytest.fixture(autouse=True)
def _no_native_cutout(monkeypatch):
    from yap_studio.gallery import cutout

    monkeypatch.setattr(cutout, "prepare_cutout", lambda *_args: None)


@pytest.fixture(autouse=True)
def _no_native_screen(monkeypatch):
    from yap_studio.matte import chroma

    detect = chroma.detect
    monkeypatch.setattr(chroma, "detect", lambda *_args: None)
    return detect


def _install(root: Path) -> RendererInstall:
    runtime = root / "node_modules/hyperframes/dist/hyperframe.runtime.iife.js"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("window.__runtime=1")
    return RendererInstall("0", root, root / "bin", None, "v22")


def _request(server, method, route, body=None, headers=None, *, authenticated=True):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    sent = ({"Content-Type": "application/json"} if method == "POST" else {})
    if method == "POST" and authenticated:
        sent["X-Gallery-Token"] = server.token
    sent.update(headers or {})
    connection.request(method, route, body=body, headers=sent)
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    server = GalleryServer(
        tmp_path / "still.svg", resolve("preset:clean_minimal").data,
        tmp_path / "work", install=_install(tmp_path / "renderer"),
        workspace=tmp_path / "workspace",
    )
    (tmp_path / "still.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_page_state_and_tiles(gallery):
    token = gallery.token
    assert _request(gallery, "GET", f"/?t={token}")[0] == 200
    assert _request(gallery, "GET", f"/static/pick.js?t={token}")[0] == 200
    status, body = _request(gallery, "GET", f"/api/state?t={token}")
    state = json.loads(body)
    assert status == 200
    assert state["presets"] and state["fonts"]["bundled"]
    assert state["from_library"] is False
    from yap_studio.sfx import bleep

    assert state["bleep_builtin"] == list(bleep.builtin_words())
    assert state["card_kinds"]["text_tag"]["category"] == "Words"
    request = {"style": state["style"], "tile": {"tab": "captions", "look": "pill"}}
    status, body = _request(gallery, "POST", "/api/layers", json.dumps(request),
                            {"Content-Type": "application/json"})
    result = json.loads(body)
    assert status == 200
    assert result["layers"][0]["name"] == "captions"
    assert result["layers"][0]["start_s"] > 0
    assert result["layers"][0]["start_s"] < result["layers"][0]["duration_s"]
    assert result["layers"][0]["y"] == live.caption_plan(
        resolve("preset:clean_minimal"), live.sample_words(live.SAMPLE_SENTENCE)
    ).y


@pytest.mark.parametrize("tab,step,names", [
    ("captions", "captions", []),
    ("preview", "captions", ["title"]),
    ("preview", "cards", ["card"]),
    ("title", "opening", ["title"]),
    ("cards", "cards", ["card"]),
])
def test_camera3d_has_no_flat_captions_but_keeps_title_and_card_layers(gallery, tab, step, names):
    style = resolve("preset:clean_minimal").data
    style["captions"]["camera_3d"] = True
    status, body = _request(gallery, "POST", "/api/layers", json.dumps({
        "style": style, "tile": {"tab": tab, "step": step},
    }))
    assert status == 200
    assert [layer["name"] for layer in json.loads(body)["layers"]] == names


@pytest.mark.parametrize("answer", [None, "none", "quiet"])
def test_state_reports_workspace_music_answer(gallery, answer):
    if answer is not None:
        target = gallery.workspace / "style/answers.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"schema": "yap-studio/style-answers@1", "music": answer}))
    assert gallery.state()["music_answer"] == answer


def test_chapter_tab_renders_only_the_requested_chapter(gallery):
    style = resolve("preset:clean_minimal").data
    style["chapters"].update(style="title_card", hold_s=2.7)
    style["hook"]["style"] = "kinetic"
    status, body = _request(gallery, "POST", "/api/layers",
                            json.dumps({"style": style, "tile": {"tab": "chapter"}}))
    assert status == 200
    result = json.loads(body)
    assert [(layer["name"], layer["template"]) for layer in result["layers"]] == [
        ("chapter", "chapter")]
    assert result["layers"][0]["duration_s"] == 2.7
    assert result["layers"][0]["start_s"] > 0


@pytest.mark.parametrize("tab, message", [("chapter", "chapter cards"), ("bad-tab", "tile.tab")])
def test_chapter_tab_off_and_unknown_tabs_are_refused(gallery, tab, message):
    style = resolve("preset:clean_minimal").data
    status, body = _request(gallery, "POST", "/api/layers",
                            json.dumps({"style": style, "tile": {"tab": tab}}))
    assert status == 400
    assert message in json.loads(body)["error"]


def test_chat_card_group_preserves_existing_categories(gallery):
    assert list(gallery_server._CATEGORIES) == [
        "Words", "Numbers", "Media", "Chat", "Editor", "Video",
    ]
    assert gallery_server._CATEGORIES == {
        "Words": ("text_tag", "checklist", "steps", "flow", "terminal", "quote", "specs_list",
                  "name_tag", "success_check", "gravity_drop", "choices"),
        "Numbers": ("stat", "chart", "progress_stat", "big_number"),
        "Media": ("image_card", "image_compare", "transfer", "callout_circle", "icon_strike"),
        "Editor": ("edit_timeline", "storyboard", "jiggle_delete", "file_drop"),
        "Video": ("video_card", "playing_grid", "reel_feed"),
        "Chat": ("prompt_bar", "comment_cta", "comment_drop"),
    }
    status, body = _request(gallery, "GET", f"/api/state?t={gallery.token}")
    assert status == 200
    state = json.loads(body)
    for category, kinds in gallery_server._CATEGORIES.items():
        assert all(state["card_kinds"][kind]["category"] == category for kind in kinds)
    for kind in ("prompt_bar", "comment_cta", "comment_drop", "icon_strike", "playing_grid",
                 "reel_feed", "gravity_drop", "choices"):
        assert state["card_kinds"][kind]["opt_in"] is True
        assert state["card_kinds"][kind]["off"] is True


@pytest.mark.parametrize("kind", ("prompt_bar", "comment_cta", "comment_drop", "icon_strike",
                                  "gravity_drop", "choices"))
def test_chat_kind_save_changes_only_card_kinds_on(gallery, kind):
    style = resolve("preset:clean_minimal").data
    style["graphics"]["card_kinds_off"] = ["stat"]
    before = dict(style["graphics"])
    replacement = {}
    for on in ([kind], []):
        style["graphics"]["card_kinds_on"] = on
        payload = json.dumps({"style": style, "name": "Card choice", **replacement})
        status, body = _request(gallery, "POST", "/api/save", payload)
        assert status == 200
        replacement = {"replace": json.loads(body)["id"]}
        saved = resolve(json.loads(body)["id"])
        assert saved.data["graphics"] == {**before, "card_kinds_on": on}
        assert cards.card_kind_off(saved, kind) is (not bool(on))


@pytest.mark.parametrize("chapters", [None, {"style": "none", "hold_s": 2.7},
                                     {"style": "title_card", "hold_s": 2.7}])
def test_chapter_choice_survives_save_and_reload(gallery, chapters):
    style = resolve("preset:clean_minimal").data
    if chapters is None:
        del style["chapters"]
    else:
        style["chapters"] = chapters
    status, body = _request(gallery, "POST", "/api/save",
                            json.dumps({"style": style, "name": "Chapter choice"}))
    assert status == 200
    saved = resolve(json.loads(body)["id"])
    expected = chapters or {"style": "none", "hold_s": 1.6}
    assert saved.data["chapters"] == expected
    assert gallery.state()["style"]["chapters"] == expected


def test_storyboard_gallery_fills_every_sample_shot(memory_gallery):
    result = memory_gallery.layers({"style": memory_gallery.style,
                                    "tile": {"tab": "cards", "card": "storyboard"}})
    assert result["layers"]
    layers = next(reversed(memory_gallery._cache.values()))[1]
    layer = next(layer for layer in layers if layer.template == "storyboard")
    thumbnails = json.loads(layer.variables["thumbnails"])
    assert len(thumbnails) == 5
    expected = memory_gallery._card_image().read_bytes()
    for thumbnail in thumbnails:
        assert (layer.composition / thumbnail["src"]).read_bytes() == expected


@pytest.mark.parametrize("size,expected", [
    ("640x400", (640, 400)),
    ("1080x1920", (360, 640)),
])
def test_file_drop_gallery_builds_a_clip_from_its_still(
    memory_gallery, tmp_path, monkeypatch, size, expected,
):
    from yap_studio import ffmpeg

    image = tmp_path / "synthetic.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color=c=blue:s={size}", "-frames:v", "1", str(image)], timeout_s=30)
    monkeypatch.setattr(memory_gallery, "_card_image", lambda: image)
    result = memory_gallery.layers({"style": memory_gallery.style,
                                    "tile": {"tab": "cards", "card": "file_drop"}})
    assert result["layers"]
    layers = next(reversed(memory_gallery._cache.values()))[1]
    layer = next(layer for layer in layers if layer.template == "file_drop")
    assert layer.variables["duration_text"] == "0:02"
    poster = layer.composition / "poster.jpg"
    assert ffmpeg.probe(poster).has_video
    info = ffmpeg.probe(poster)
    assert info.width / info.height == pytest.approx(expected[0] / expected[1], abs=1 / info.height)
    assert layer.variables["poster_src"] == poster.name
    pixel = ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(poster),
                        "-vf", "scale=1:1", "-frames:v", "1", "-pix_fmt", "rgb24",
                        "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30).stdout
    assert pixel[2] > 200 and max(pixel[:2]) < 50


def test_state_reports_opt_in_kinds(gallery, monkeypatch):
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("stat",))
    kinds = gallery.state()["card_kinds"]
    assert kinds["stat"] == {"category": "Numbers", "opt_in": True, "off": True}
    assert kinds["text_tag"] == {"category": "Words", "opt_in": False, "off": False}


def test_card_categories_cover_every_template_once():
    kinds = [kind for group in gallery_server._CATEGORIES.values() for kind in group]
    assert len(kinds) == len(set(kinds))
    assert set(kinds) == set(cards.CARD_TEMPLATES)


@pytest.mark.parametrize("category", ["Words", "Numbers"])
def test_state_lists_only_installed_on_registry_kinds(memory_gallery, registry_card, category):
    catalog, install, _, _, add, _ = registry_card
    catalog.items()["sample"]["kind"]["category"] = category
    add("missing")
    catalog.items()["missing"].update(card_fields())
    add("pilot")
    install.install_items(["pilot"])
    kinds = memory_gallery.state()["card_kinds"]
    assert kinds[KIND] == {"category": category, "opt_in": True, "off": True,
                           "label": "Sample", "registry": True}
    assert "hf:missing" not in kinds
    assert "hf:pilot" not in kinds
    memory_gallery.style["graphics"]["card_kinds_on"] = [KIND]
    assert memory_gallery.state()["card_kinds"][KIND]["off"] is False


@pytest.mark.parametrize(("category", "sentence", "expected"), [
    ("Words", "My first sentence. Another one.", "My first"),
    ("Words", "Go now. Keep going.", "Go now."),
    ("Words", "Unfittableword next", "Ship it now"),
    ("Numbers", "My first sentence.", "Ship it now"),
])
def test_registry_layer_uses_preview_words(
    memory_gallery, registry_card, category, sentence, expected,
):
    catalog = registry_card[0]
    entry = catalog.items()["sample"]
    entry["kind"]["category"] = category
    entry["slots"]["text"]["max_length"] = 11
    words = live.sample_words(sentence)
    memory_gallery.preview.update(sentence=sentence, words=[vars(word) for word in words])
    status, body, _ = _memory_request(memory_gallery, "/api/layers", method="POST", data={
        "style": memory_gallery.style, "tile": {"tab": "cards", "card": KIND},
    }, headers={"X-Gallery-Token": memory_gallery.token})
    assert status == 200, body
    assert json.loads(body)["layers"][0]["template"] == KIND
    layer = next(reversed(memory_gallery._cache.values()))[1][0]
    elements = Elements((layer.composition / "index.html").read_text())
    mount = next(attrs for _, attrs in elements.elements if attrs.get("id") == "hf-mount")
    variables = json.loads(mount["data-variable-values"])
    assert variables["text"] == expected
    assert len(variables["text"]) <= 11
    assert variables["progress"] == 37


@pytest.mark.parametrize("kind", ["hf:missing", "hf:unknown", "hf:pilot"])
def test_layers_accept_listed_and_refuse_unlisted_registry_kinds(
    memory_gallery, registry_card, kind,
):
    catalog, _, _, _, add, _ = registry_card
    add("missing")
    catalog.items()["missing"].update(card_fields())
    add("pilot")
    assert _memory_request(memory_gallery, "/api/layers", method="POST", data={
        "style": memory_gallery.style, "tile": {"tab": "cards", "card": KIND},
    }, headers={"X-Gallery-Token": memory_gallery.token})[0] == 200
    status, body, _ = _memory_request(memory_gallery, "/api/layers", method="POST", data={
        "style": memory_gallery.style, "tile": {"tab": "cards", "card": kind},
    }, headers={"X-Gallery-Token": memory_gallery.token})
    assert status == 400
    assert "unknown card kind" in json.loads(body)["error"]


def test_registry_layer_serves_nested_assets_safely(memory_gallery, registry_card, tmp_path):
    result = memory_gallery.layers({"style": memory_gallery.style,
                                    "tile": {"tab": "cards", "card": KIND}})
    src = result["layers"][0]["src"]
    layer = next(reversed(memory_gallery._cache.values()))[1][0]
    files = {
        "compositions/sample.html": "text/html; charset=utf-8",
        "compositions/components/deep/item.html": "text/html; charset=utf-8",
        "fonts/alias.woff2": "font/woff2", "fonts/alias.woff": "font/woff",
        "fonts/alias.ttf": "font/ttf", "fonts/alias.otf": "font/otf",
        "CustomEase.min.js": "text/javascript; charset=utf-8",
        "assets/item.js": "text/javascript; charset=utf-8",
        "assets/item.css": "text/css; charset=utf-8", "assets/item.svg": "image/svg+xml",
        "assets/item.png": "image/png", "assets/item.jpg": "image/jpeg",
        "assets/item.webp": "image/webp", "assets/item.json": "application/json",
        "a/b/c/d/e/item.js": "text/javascript; charset=utf-8",
    }
    for relative, mime in files.items():
        target = layer.composition / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(b"asset")
        status, body, csp = _memory_request(memory_gallery, src + relative)
        assert status == 200, relative
        assert body == target.read_bytes()
        assert memory_gallery.layer_file(src.split("/")[2], 0, relative)[1] == mime
        if relative.endswith(".html"):
            assert csp == gallery_server._LAYER_CSP
    outside = tmp_path / "outside.js"
    outside.write_bytes(b"outside")
    (layer.composition / "escape.js").symlink_to(outside)
    (layer.composition / "inside.js").symlink_to(layer.composition / "CustomEase.min.js")
    (layer.composition / "linked").symlink_to(layer.composition / "assets",
                                               target_is_directory=True)
    for relative in [".hidden.js", "assets/.hidden.js", "a/b/c/d/e/f/item.js", "unknown.txt"]:
        target = layer.composition / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"hidden")
    for relative in ["../outside.js", "/outside.js", ".hidden.js", "assets/.hidden.js",
                     "compositions/../index.html", "escape.js", "inside.js", "linked/item.js",
                     "%2e%2e/outside.js", "%2Foutside.js", "%2ehidden.js", "assets\\item.js",
                     "a/b/c/d/e/f/item.js", "unknown.txt", "missing.js", "assets/", "a//b.js"]:
        assert _memory_request(memory_gallery, src + relative)[0] == 404, relative
    status, body, csp = _memory_request(memory_gallery, src + "index.html")
    assert status == 200 and b'/hf/runtime.js' in body
    assert csp == gallery_server._LAYER_CSP
    builtin = memory_gallery.layers({"style": memory_gallery.style,
                                     "tile": {"tab": "cards", "card": "stat"}})
    for relative in ["compositions/sample.html", "CustomEase.min.js", "unknown.js"]:
        assert _memory_request(memory_gallery, builtin["layers"][0]["src"] + relative)[0] == 404


def test_save_round_trips_opt_in_kinds(gallery, monkeypatch):
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("stat",))
    style = resolve("preset:clean_minimal").data
    style["graphics"]["card_kinds_on"] = ["stat"]
    payload = json.dumps({"style": style, "name": "Opt in"})
    status, body = _request(gallery, "POST", "/api/save", payload)
    assert status == 200
    saved = resolve(json.loads(body)["id"])
    assert saved.data["graphics"]["card_kinds_on"] == ["stat"]
    assert cards.card_kind_off(saved, "stat") is False
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ())
    monkeypatch.setattr(registry_cards, "kinds", lambda: ())
    status, body = _request(gallery, "POST", "/api/save", payload)
    assert status == 400
    assert "card_kinds_on" in json.loads(body)["error"]


def test_music_only_changes_reuse_visual_layers(memory_gallery):
    style = resolve("preset:clean_minimal").data
    first = memory_gallery.layers({"style": style})
    style["music"]["level"] = 0.31
    assert memory_gallery.layers({"style": style}) == first
    assert len(memory_gallery._cache) == 1
    style["captions"]["size_px"] += 10
    assert memory_gallery.layers({"style": style}) != first
    assert len(memory_gallery._cache) == 2


@pytest.mark.parametrize("tab", ["captions", "title", "cards"])
@pytest.mark.parametrize("first_on", [False, True])
def test_bleep_switch_rebuilds_visual_layers(memory_gallery, tab, first_on):
    style = resolve("preset:clean_minimal").data
    memory_gallery.preview["sentence"] = "Fuck this."
    tile = {"tab": tab}
    results = []
    for on in (first_on, not first_on, first_on):
        style["music"]["bleep_swears"] = on
        result = memory_gallery.layers({"style": style, "tile": tile})
        digest = result["layers"][0]["src"].split("/")[2]
        drawn = json.dumps([layer.variables for layer in memory_gallery._cache[digest][1]])
        assert ("F***" in drawn) is on
        results.append(result)
    assert results[0] != results[1]
    assert results[0] == results[2]
    assert len(memory_gallery._cache) == 2


def test_bleep_cache_uses_effective_words_and_ignores_other_music(memory_gallery):
    style = resolve("preset:clean_minimal").data
    style["music"]["bleep_swears"] = True
    first = memory_gallery.layers({"style": style})
    style["music"]["bleep_words_add"] = ["small"]
    custom = memory_gallery.layers({"style": style})
    assert custom != first
    style["music"]["level"] = .31
    style["music"]["bleep_words_add"].append("fuck")
    assert memory_gallery.layers({"style": style}) == custom
    style["music"]["bleep_words_remove"] = ["small"]
    assert memory_gallery.layers({"style": style}) == first


def test_bleep_off_defaults_keep_the_original_visual_identity(memory_gallery):
    import hashlib

    style = resolve("preset:clean_minimal").data
    result = memory_gallery.layers({"style": style})
    visual = {key: value for key, value in gallery_server.resolve_style_data(style).items()
              if key not in {"music", "grade"}}
    expected = hashlib.sha256(json.dumps(
        [visual, [vars(word) for word in memory_gallery._words()], {},
         memory_gallery.preview["generation"], None], sort_keys=True,
    ).encode()).hexdigest()
    assert result["layers"][0]["src"].split("/")[2] == expected
    for key in ("bleep_swears", "bleep_words_add", "bleep_words_remove"):
        style["music"].pop(key)
    assert memory_gallery.layers({"style": style}) == result


def test_grade_only_changes_reuse_visual_layers(memory_gallery):
    style = resolve("preset:clean_minimal").data
    first = memory_gallery.layers({"style": style})
    for strength in (0.5, 1):
        style["grade"] = {"look": "film", "strength": strength}
        assert memory_gallery.layers({"style": style}) == first
    assert len(memory_gallery._cache) == 1
    style["grade"] = {"look": "eyedropper", "strength": 0.5, "color": "#1E6BD6"}
    assert memory_gallery.layers({"style": style}) == first
    style["grade"]["slider"] = True
    assert memory_gallery.layers({"style": style}) == first
    assert len(memory_gallery._cache) == 1


@pytest.mark.parametrize(("look", "strength", "color"), [
    ("film", 1, None), ("film", 0.5, None),
    ("eyedropper", 1, "#1E6BD6"), ("eyedropper", 0.5, "#1E6BD6"),
    ("eyedropper", 1, "#F08A45"), ("eyedropper", 0.5, "#F08A45"),
])
def test_preview_grade_color_matches_engine(look, strength, color):
    colors = ["#F34E3F", "#FFFFFF", "#000000", "#101010", "#909090", "#9F9F9F",
              "#A0A0A0", "#D7A071", "#FF0000", "#00FF00", "#0000FF", "#00FFFF",
              "#FF00FF", "#FFFF00"]
    choices = (Path(__file__).resolve().parents[1]
               / "src/yap_studio/gallery/static/choices.mjs")
    script = """
import {readFileSync} from 'node:fs';
const {gradeColor} = await import(process.argv[1]);
const {colors, grade} = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(colors.map(color => gradeColor(color, grade))));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, choices.as_uri()],
        input=json.dumps({"colors": colors, "grade": {
            "look": look, "strength": strength, "color": color,
        }}), capture_output=True, text=True, check=True, timeout=10,
    )
    actual = np.array([[int(value[i:i+2], 16) for i in (1, 3, 5)]
                       for value in json.loads(result.stdout)])
    rgb = np.array([[int(value[i:i+2], 16) for i in (1, 3, 5)] for value in colors]) / 255
    expected = np.rint(grade.apply_rgb(rgb, grade.GradeSpec(look, strength, color=color)) * 255)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1,
                               err_msg=f"{look} strength={strength} color={color}; {colors}")


def test_still_slider_is_cached_and_uses_style_handle(memory_gallery, monkeypatch):
    from yap_studio import ffmpeg

    server = memory_gallery
    full = _memory_request(server, "/still?look=film")[1]
    query = "/still?look=film&slider=1&text_color=%23F8E4A0&ink_color=%23123456"
    status, body, _ = _memory_request(server, query)
    assert status == 200 and body != full
    assert body.startswith(b"\xff\xd8")
    monkeypatch.setattr(ffmpeg, "run", lambda *a, **kw: pytest.fail("Still was not cached"))
    assert _memory_request(server, query)[1] == body


@pytest.mark.parametrize("query", ["look=film&slider=2", "look=film&slider=true",
                                    "look=film&slider=1&slider=0", "look=none&slider=1",
                                    "look=film&slider=1&text_color=bad"])
def test_bad_still_slider_parameters(memory_gallery, query):
    assert _memory_request(memory_gallery, "/still?" + query)[0] == 400


def test_eyedropper_preview_and_colour_validation(memory_gallery):
    from yap_studio import ffmpeg

    server = memory_gallery
    image = server.work_dir / "scene.png"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=0x20506e:s=90x160,drawbox=x=0:y=54:w=90:h=106:c=red:t=fill",
                "-frames:v", "1", "-threads", "1", str(image)], timeout_s=30)
    server.still = image
    preview = server.state()["preview"]
    assert preview["grade_default_strength"] == 0.5
    assert preview["grade_color"]["colorful"]
    color = preview["grade_color"]["color"]
    decoded = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(image), "-frames:v", "1",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         timeout_s=30, binary=True).stdout
    assert color == "#" + "".join(f"{value:02X}" for value in decoded[:3])
    plain = _memory_request(server, "/still?look=none")[1]
    request = "/still?look=eyedropper&strength=0.5&color=" + color.replace("#", "%23")
    status, body, _ = _memory_request(server, request)
    assert status == 200 and body != plain and body.startswith(b"\xff\xd8")
    assert _memory_request(server, request)[1] == body
    other = _memory_request(server, "/still?look=eyedropper&color=%231E6BD6")[1]
    assert other != body
    status, body, _ = _memory_request(server, "/api/grade-color?color=%23808080")
    info = json.loads(body)
    assert status == 200 and not info["colorful"] and "colourful" in info["message"]
    assert _memory_request(server, "/still?look=eyedropper&color=%23808080")[0] == 400
    assert _memory_request(server, "/api/grade-color?color=bad")[0] == 400


@pytest.mark.parametrize("image_kind", ["svg", "jpg"])
def test_still_grade_cached_and_generation_changes(memory_gallery, monkeypatch, image_kind):
    from yap_studio import ffmpeg

    server = memory_gallery
    if image_kind == "jpg":
        image = server.work_dir / "synthetic.jpg"
        ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=90x160",
                    "-frames:v", "1", str(image)], timeout_s=30)
        server.still = image
    plain = _memory_request(server, "/still?g=0")[1]
    base = (Path(gallery_server.__file__).with_name("sample-still.jpg")
            if image_kind == "svg" else server.still)
    assert plain == server.still.read_bytes()
    assert _memory_request(server, "/still?g=0&look=none")[1] == base.read_bytes()
    assert _memory_request(server, "/still?g=0&look=film&strength=0")[1] == base.read_bytes()
    calls = []
    original = ffmpeg.run

    def measured(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", measured)
    status, full, _ = _memory_request(server, "/still?g=0&look=film&strength=1")
    assert status == 200 and full.startswith(b"\xff\xd8") and full != plain
    decode = next(command for command in calls
                  if command[0] == "ffmpeg" and "-pixel_format" not in command)
    assert decode[decode.index("-i") + 1] == str(base)
    count = len(calls)
    assert _memory_request(server, "/still?g=0&look=film&strength=1")[1] == full
    assert len(calls) == count
    light = _memory_request(server, "/still?g=0&look=film&strength=0.5")[1]
    assert light != full and len(calls) > count
    before = len(calls)
    server.update_preview({"still": str(server.still), "source": "sample"})
    assert _memory_request(server, "/still?g=1&look=film&strength=1")[1] == full
    assert len(calls) > before
    server._still_cache.clear()
    assert _memory_request(server, "/still?g=1&look=film&strength=1")[1] == full


@pytest.mark.parametrize(("size", "expected"), [
    ((2160, 3840), (1080, 1920)), ((3840, 2160), (1080, 608)), ((90, 160), (90, 160)),
])
def test_still_grade_bounds_decoded_pixels(memory_gallery, monkeypatch, size, expected):
    from yap_studio import ffmpeg

    server = memory_gallery
    image = server.work_dir / "synthetic.png"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                f"color=0x20506e:s={size[0]}x{size[1]}", "-frames:v", "1", "-threads", "1",
                str(image)], timeout_s=30)
    server.still = image
    assert _memory_request(server, "/still?g=0")[1] == image.read_bytes()
    shapes = []
    original = gallery_server.apply_rgb

    def measured(rgb, spec):
        shapes.append(rgb.shape)
        return original(rgb, spec)

    monkeypatch.setattr(gallery_server, "apply_rgb", measured)
    status, body, _ = _memory_request(server, "/still?g=0&look=film&strength=1")
    assert status == 200
    assert shapes == [(expected[1], expected[0], 3)]
    result = server.work_dir / "graded.jpg"
    result.write_bytes(body)
    info = ffmpeg.probe(result, count_frames=False)
    assert (info.width, info.height) == expected


@pytest.mark.parametrize("query", ["look=other", "strength=nan", "strength=inf", "strength=2",
                                    "strength=-1", "strength=", "strength=abc", "g=-1", "g=x",
                                    "look=film&look=none", "look=", "extra=1"])
def test_bad_still_grade_parameters(memory_gallery, query):
    status, body, _ = _memory_request(memory_gallery, "/still?" + query)
    assert status == 400 and json.loads(body)["error"]


def test_preview_face_boxes_reach_the_live_caption_plan(memory_gallery):
    style = resolve("preset:clean_minimal").data
    style["captions"].update(position="middle", top_px=700, entry_ms=0)
    memory_gallery.preview["face_boxes"] = [{"x": 400, "y": 650, "w": 200, "h": 180}]
    result = memory_gallery.layers({"style": style})
    digest = result["layers"][0]["src"].split("/")[2]
    caption = memory_gallery._cache[digest][1][0]
    assert any(slot["offset_px"] for slot in json.loads(caption.variables["phrases"]))


def test_state_has_learned_styles_and_creator_fonts(gallery):
    config = gallery.workspace.parent / "config"
    library = config / "styles"
    library.mkdir(parents=True)
    (library / "learned.json").write_text(json.dumps(resolve("preset:electric").data))
    (library / "learned.source.json").write_text(json.dumps({"videos": 3}))
    fonts = config / "fonts"
    fonts.mkdir()
    (fonts / "fonts.json").write_text(json.dumps({
        "schema": "yap-studio/user-fonts@1",
        "fonts": [{"family": "Creator Serif", "file": "creator.woff2", "style": "normal",
                   "weight": "400", "sha256": "a" * 64}],
    }))
    state = gallery.state()
    assert state["library"][0]["videos"] == 3
    assert state["library"][0]["name"] == "Electric"
    assert "Creator Serif" in state["fonts"]["creator"]
    assert "Instrument Serif" in state["fonts"]["bundled"]


def test_http_state_lists_learned_style_with_source_record(gallery):
    library = gallery.workspace.parent / "config" / "styles"
    library.mkdir(parents=True)
    (library / "learned.json").write_text(json.dumps(resolve("preset:electric").data))
    (library / "learned.source.json").write_text(json.dumps({"videos": 7}))
    status, body = _request(gallery, "GET", "/api/state")
    assert status == 200
    assert [(item["id"], item["videos"]) for item in json.loads(body)["library"]] == [
        ("learned", 7)]


def test_brand_view_changes_only_brand_values(gallery):
    style = resolve("preset:electric").data
    style["brand"]["accent_color"] = "#ABCDEF"
    style["brand"]["captions_font"] = "Instrument Serif"
    base = {"style": style, "tile": {"tab": "captions"}}
    _, branded = _request(gallery, "POST", "/api/layers", json.dumps(base))
    base["tile"]["branded"] = False
    _, plain = _request(gallery, "POST", "/api/layers", json.dumps(base))
    branded_key = json.loads(branded)["layers"][0]["src"].split("/")[2]
    plain_key = json.loads(plain)["layers"][0]["src"].split("/")[2]
    branded_vars = gallery._cache[branded_key][1][0].variables
    plain_vars = gallery._cache[plain_key][1][0].variables
    assert branded_vars["highlight_style"] == plain_vars["highlight_style"]
    assert branded_vars["highlight_color"] != plain_vars["highlight_color"]


def test_preview_changes_without_reload(gallery):
    before = json.loads(_request(gallery, "GET", "/api/preview")[1])
    photo = gallery.workspace.parent / "take.jpg"
    photo.write_bytes(b"synthetic")
    gallery.update_preview({"source": "take", "still": str(photo),
                            "sentence": "Your first sentence.",
                            "words": [{"text": "Your", "start": 0.3, "end": 0.6},
                                      {"text": "first", "start": 0.7, "end": 1.0}]})
    after = json.loads(_request(gallery, "GET", "/api/preview")[1])
    assert before["source"] == "sample" and after["source"] == "take"
    assert after["generation"] == before["generation"] + 1
    assert after["sentence"] == "Your first sentence."
    assert _request(gallery, "GET", after["still"])[1] == b"synthetic"


def test_save_and_request_validation(gallery):
    style = resolve("preset:clean_minimal").data
    route = "/api/save"
    assert _request(gallery, "POST", route, "{")[0] == 400
    assert _request(gallery, "POST", route, "", {"Content-Length": "1048577"})[0] == 413
    assert _request(gallery, "POST", route, json.dumps({"style": {}}))[0] == 400
    assert _request(gallery, "POST", route, json.dumps({"style": style}),
                    {"Origin": "http://evil.example"})[0] == 403
    status, body = _request(gallery, "POST", route,
                            json.dumps({"style": style, "name": "My style"}))
    saved = json.loads(body)
    assert status == 200 and saved["id"] == "my_style"
    assert (gallery.workspace / "style/my-style.json").is_file()
    assert saved["workspace_copy"] == "style/my-style.json"
    assert current_default_data()["name"] == "My style"


def test_save_name_conflict_identical_content_and_replace(gallery):
    from yap_studio.style.library import get_default, library_dir

    style = resolve("preset:clean_minimal").data
    first = save_pick(style, name="My style", workspace=gallery.workspace)
    before = first.path.read_bytes()
    payload = {"style": style, "name": "My style"}
    status, body = _request(gallery, "POST", "/api/save", json.dumps(payload))
    assert status == 200 and json.loads(body)["id"] == first.id
    payload["style"]["music"]["level"] = 0
    payload["name"] = "  MY STYLE  "
    status, body = _request(gallery, "POST", "/api/save", json.dumps(payload))
    assert status == 409
    assert json.loads(body) == {
        "error": "You already have a style called “My style”. Update it, or choose another name.",
        "code": "style_name_taken", "id": first.id,
    }
    assert first.path.read_bytes() == before
    payload.update(name="My style", replace=first.id)
    status, body = _request(gallery, "POST", "/api/save", json.dumps(payload))
    assert status == 200 and json.loads(body)["id"] == first.id
    assert get_default() == first.id
    assert resolve(first.id).data["music"]["level"] == 0
    assert first.path.read_bytes() == (gallery.workspace / "style/my-style.json").read_bytes()
    assert len(list(library_dir().glob("*.json"))) == 1


@pytest.mark.parametrize("replacement", ["missing", "other"])
def test_save_replace_requires_matching_name(gallery, replacement):
    first = save_pick({"schema_version": 1}, name="My style")
    save_pick({"schema_version": 1}, name="Other")
    status, body = _request(gallery, "POST", "/api/save", json.dumps({
        "style": resolve("preset:clean_minimal").data,
        "name": "My style", "replace": replacement,
    }))
    assert status == 409
    assert json.loads(body)["code"] == "style_name_taken"
    assert json.loads(body)["id"] == first.id


def test_remote_binding_requires_token_and_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    image = tmp_path / "still.svg"
    image.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    server = GalleryServer(image, resolve("preset:clean_minimal").data,
                           tmp_path / "work", install=_install(tmp_path / "renderer"),
                           host="0.0.0.0")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _request(server, "GET", "/")[0] == 403
        assert _request(server, "GET", f"/?t={server.token}")[0] == 200
        body = json.dumps({"style": resolve("preset:clean_minimal").data,
                           "tile": {"tab": "captions"}})
        assert _request(server, "POST", "/api/layers", body, authenticated=False)[0] == 403
        assert _request(server, "POST", "/api/layers", body,
                        {"X-Gallery-Token": server.token,
                         "Origin": "http://evil.example"})[0] == 403
        assert _request(server, "POST", "/api/layers", body,
                        {"X-Gallery-Token": server.token})[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_parser_is_lazy_and_missing_renderer_has_fix(capsys, monkeypatch):
    code = ("import sys; from yap_studio.cli import _parser; "
            "assert _parser().parse_args(['style','pick']).style_command == 'pick'; "
            "assert 'yap_studio.gallery.server' not in sys.modules; "
            "assert 'yap_studio.gallery.live' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(hyperframes, "installed", lambda: None)
    from yap_studio.cli import main

    assert main(["style", "pick", "--no-open"]) == 2
    assert "fix: run yap-studio renderer install" in capsys.readouterr().err


def test_parallel_layer_requests_succeed(gallery):
    payload = json.dumps({"style": gallery.style, "tile": {"tab": "captions", "look": "pill"}})
    assert _request(gallery, "POST", "/api/layers", payload)[0] == 200
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: _request(gallery, "POST", "/api/layers", payload),
                                range(64)))
    assert all(status == 200 and json.loads(body)["layers"] for status, body in results)


def test_sample_image_card_prepares_without_sips(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    preview = still.prepare_preview(inbox, tmp_path / "preview")
    gallery = GalleryServer(Path(preview["still"]), resolve("preset:clean_minimal").data,
                            tmp_path / "work", install=_install(tmp_path / "renderer"))
    commands = []
    original = subprocess.run

    def planned_only(command, *args, **kwargs):
        commands.append(Path(command[0]).name)
        assert Path(command[0]).name != "sips"
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", planned_only)
    try:
        result = gallery.layers({"style": gallery.style, "tile": {"tab": "cards",
                                                                  "card": "image_card"}})
        digest = result["layers"][0]["src"].split("/")[2]
        image = gallery._cache[digest][1][0].composition / "image.jpg"
        bundled = still.sample_still().with_suffix(".jpg")
        assert image.read_bytes() == bundled.read_bytes()
        assert bundled.stat().st_size < 200_000
        assert set(commands) <= {"ffmpeg", "ffprobe"}
    finally:
        gallery.server_close()


def test_recently_fetched_layer_survives_cache_eviction(gallery):
    result = gallery.layers({"style": gallery.style, "tile": {"tab": "captions"}})
    src = result["layers"][0]["src"]
    digest = src.split("/")[2]
    layer = gallery._cache[digest][1][0]
    with gallery._lock:
        for index in range(gallery._cache_limit - 1):
            key = f"{index:064x}"
            gallery._cache[key] = (tempfile.TemporaryDirectory(dir=gallery.work_dir), [layer])
    assert _request(gallery, "GET", src)[0] == 200
    gallery.layers({"style": gallery.style, "tile": {"tab": "captions", "new": True}})
    assert _request(gallery, "GET", src)[0] == 200


def test_state_and_static_respond_during_slow_layer_preparation(gallery, monkeypatch):
    cached = gallery.layers({"style": gallery.style, "tile": {"tab": "captions"}})
    cached_src = cached["layers"][0]["src"]
    entered, release = threading.Event(), threading.Event()
    original = live.live_layers

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(live, "live_layers", slow)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = pool.submit(gallery.layers, {"style": gallery.style,
                                                "tile": {"tab": "captions", "slow": True}})
        try:
            assert entered.wait(10)
            for route in ("/api/state", "/static/pick.js", "/still", cached_src):
                assert pool.submit(_request, gallery, "GET", route).result(timeout=2)[0] == 200
        finally:
            release.set()
        assert pending.result(timeout=10)["layers"]


def test_local_post_requires_token_and_json_content_type(gallery):
    body = json.dumps({"style": gallery.style, "tile": {"tab": "captions"}})
    assert _request(gallery, "POST", "/api/layers", body,
                    authenticated=False)[0] == 403
    assert _request(gallery, "POST", "/api/layers", body,
                    {"X-Gallery-Token": gallery.token,
                     "Content-Type": "application/x-www-form-urlencoded"})[0] == 415


def test_no_open_json_prints_url_then_serves(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(hyperframes, "installed", lambda: _install(tmp_path / "renderer"))
    script = tmp_path / "gsap.min.js"
    script.write_text("window.gsap={}")
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: script)
    ready = threading.Event()
    holder = {}
    original = GalleryServer.serve_forever

    def serving(server):
        holder["server"] = server
        ready.set()
        original(server)

    monkeypatch.setattr(GalleryServer, "serve_forever", serving)
    from yap_studio.cli import main

    thread = threading.Thread(target=lambda: holder.setdefault(
        "result", main(["style", "pick", "--no-open", "--json"])), daemon=True)
    thread.start()
    assert ready.wait(10)
    url = json.loads(capsys.readouterr().out)["url"]
    assert url.startswith("http://127.0.0.1:")
    holder["server"].shutdown()
    thread.join(timeout=10)
    assert holder["result"] == 0 and not thread.is_alive()


@pytest.fixture
def phone_pick(tmp_path, monkeypatch):
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    install = _install(tmp_path / "renderer")
    monkeypatch.setattr(hyperframes, "installed", lambda: install)
    gsap = tmp_path / "gsap.min.js"
    gsap.write_text("window.gsap={}")
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: gsap)
    binds = []

    def bind(address, _handler):
        binds.append(address)
        return type("Socket", (), {"server_port": address[1] or 12345,
                                    "server_close": lambda self: None,
                                    "serve_forever": lambda self: None})()

    monkeypatch.setattr(gallery_server, "_GalleryHTTPServer", bind)
    return binds


def test_phone_pick_wifi_link_and_json(phone_pick, monkeypatch, capsys):
    from yap_studio.cli import main

    monkeypatch.setattr(phone, "tailscale_name", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: "192.168.1.20")
    assert main(["style", "pick", "--phone", "--no-open"]) == 0
    assert phone_pick == [("192.168.1.20", 0)]
    assert capsys.readouterr().out.startswith(
        "Open on your phone: Not encrypted (same Wi-Fi): "
        "http://192.168.1.20:12345/?t=")
    assert main(["style", "pick", "--phone", "--no-open", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["phone"] == {"via": "wifi", "url": result["url"]}


def test_phone_pick_tailnet_and_mapping_cleanup(phone_pick, monkeypatch, capsys):
    from yap_studio.cli import main

    cleaned = []
    monkeypatch.setattr(phone, "tailscale_name", lambda: "box.tail1234.ts.net")
    monkeypatch.setattr(phone, "tailscale_origin", lambda port: (
        f"https://box.tail1234.ts.net:{port}", lambda: cleaned.append(port)))
    monkeypatch.setattr(phone, "lan_address", lambda: pytest.fail("Wi-Fi not needed"))
    assert main(["style", "pick", "--phone", "--no-open", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert phone_pick == [("127.0.0.1", 0)]
    assert cleaned == [12345]
    assert result["phone"] == {"via": "tailscale", "url": result["url"]}
    assert result["url"].startswith("https://box.tail1234.ts.net:12345/?t=")


def test_phone_pick_falls_back_and_reports_no_network(phone_pick, monkeypatch, capsys):
    from yap_studio.cli import main

    monkeypatch.setattr(phone, "tailscale_name", lambda: "box.tail1234.ts.net")
    monkeypatch.setattr(phone, "tailscale_origin",
                        lambda _port: (_ for _ in ()).throw(OSError("HTTPS unavailable")))
    monkeypatch.setattr(phone, "lan_address", lambda: "10.0.0.7")
    assert main(["style", "pick", "--phone", "--no-open"]) == 0
    assert phone_pick == [("127.0.0.1", 0), ("10.0.0.7", 0)]
    captured = capsys.readouterr()
    assert "Tailscale link failed (HTTPS unavailable)" in captured.err
    assert "http://10.0.0.7:12345/?t=" in captured.out
    monkeypatch.setattr(phone, "tailscale_name", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: None)
    assert main(["style", "pick", "--phone", "--no-open"]) == 2
    captured = capsys.readouterr()
    assert "No network your phone can reach" in captured.err
    assert "Join the same Wi-Fi" in captured.err


def test_phone_pick_and_host_are_exclusive(phone_pick, capsys):
    from yap_studio.cli import main

    assert main(["style", "pick", "--phone", "--host", "127.0.0.1", "--no-open"]) == 2
    assert "use --phone or --host, not both" in capsys.readouterr().err
    assert phone_pick == []


@pytest.mark.parametrize("startup_error", [False, True])
def test_pick_warns_when_take_jobs_close_fails(
    phone_pick, tmp_path, monkeypatch, capsys, startup_error,
):
    from yap_studio import style
    from yap_studio.cli import main
    from yap_studio.gallery import command
    from yap_studio.transcript import take_job

    closed = []

    class FailingJobs:
        def close(self):
            closed.append(True)
            raise OSError("take job process group did not stop")

    work = tempfile.TemporaryDirectory(dir=tmp_path)
    monkeypatch.setattr(command.tempfile, "TemporaryDirectory", lambda **_kwargs: work)
    monkeypatch.setattr(take_job, "TakeJobs", FailingJobs)
    if startup_error:
        def fail():
            raise ValueError("synthetic startup failure")

        monkeypatch.setattr(style, "current_default_data", fail)
    before = signal.getsignal(signal.SIGTERM)
    assert main(["style", "pick", "--no-open"]) == (2 if startup_error else 0)
    assert closed == [True]
    assert not Path(work.name).exists()
    assert signal.getsignal(signal.SIGTERM) is before
    error = capsys.readouterr().err
    assert error.count("warning:") == 1
    assert "take job process group did not stop" in error


def test_pick_marks_library_starting_style(phone_pick, monkeypatch):
    from yap_studio.cli import main

    save_pick(resolve("preset:electric").data, name="My custom look")
    seen = []
    monkeypatch.setattr(GalleryServer, "serve_forever",
                        lambda server: seen.append(server.state()["from_library"]))
    assert main(["style", "pick", "--no-open"]) == 0
    assert seen == [True]


@pytest.mark.parametrize("saved,explicit", [(False, False), (True, False), (False, True)])
def test_pick_starts_with_answer_fields_only_without_a_look(
    phone_pick, monkeypatch, tmp_path, saved, explicit,
):
    from yap_studio.cli import main

    target = tmp_path / "style/answers.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"schema": "yap-studio/style-answers@1", "music": "none",
                                  "never": {"cover_face": False},
                                  "brand": {"accent_color": "#114488"}}))
    if saved:
        save_pick(resolve("preset:clean_minimal").unbranded_data, name="My look")
    seen = []
    monkeypatch.setattr(GalleryServer, "serve_forever", lambda server: seen.append(server.state()))
    assert main(["style", "pick", "--no-open", "--workspace", str(tmp_path),
                 *(["--style", "preset:clean_minimal"] if explicit else [])]) == 0
    assert seen[0]["from_library"] is saved
    style = seen[0]["style"]
    assert style["music"]["level"] == (0 if not saved and not explicit else 0.06)
    assert style["captions"]["placement"] == (
        "fixed" if not saved and not explicit else "avoid_face")
    assert style["brand"]["accent_color"] == ("#114488" if not saved and not explicit else "")


@pytest.mark.parametrize("problem", ["json", "validation", "read", "style"])
def test_pick_ignores_unreadable_or_invalid_answers(phone_pick, monkeypatch, tmp_path, problem):
    from yap_studio.cli import main
    from yap_studio.style import StyleError, answers

    target = tmp_path / "style/answers.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"schema": "yap-studio/style-answers@1", "music": "none"}))
    if problem == "json":
        target.write_text("{")
    elif problem == "validation":
        target.write_text(json.dumps({"schema": "yap-studio/style-answers@1", "music": "invalid"}))
    elif problem == "read":
        def fail_read(_path):
            raise OSError("Synthetic read failure")

        monkeypatch.setattr(answers, "load_answers", fail_read)
    else:
        def fail_style(_answers, _base):
            raise StyleError("answers", "Synthetic style failure")

        monkeypatch.setattr(answers, "style_changes", fail_style)
    seen = []
    monkeypatch.setattr(GalleryServer, "serve_forever", lambda server: seen.append(server.style))
    assert main(["style", "pick", "--no-open", "--workspace", str(tmp_path)]) == 0
    assert seen == [resolve("preset:clean_minimal").unbranded_data]


def test_phone_pick_sigterm_cleans_mapping(phone_pick, monkeypatch):
    from yap_studio.cli import main

    cleaned = []
    monkeypatch.setattr(phone, "tailscale_name", lambda: "box.tail1234.ts.net")
    monkeypatch.setattr(phone, "tailscale_origin",
                        lambda port: (f"https://box.tail1234.ts.net:{port}",
                                      lambda: cleaned.append(port)))
    monkeypatch.setattr(GalleryServer, "serve_forever",
                        lambda _server: os.kill(os.getpid(), signal.SIGTERM))
    before = signal.getsignal(signal.SIGTERM)
    assert main(["style", "pick", "--phone", "--no-open"]) == 0
    assert cleaned == [12345]
    assert signal.getsignal(signal.SIGTERM) is before


def test_tailnet_host_needs_gallery_token(tmp_path, monkeypatch):
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    image = tmp_path / "still.svg"
    image.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    server = GalleryServer(image, resolve("preset:clean_minimal").data,
                           tmp_path / "work", install=_install(tmp_path / "renderer"),
                           allowed_hosts=("box.tail1234.ts.net",))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = {"Host": f"box.tail1234.ts.net:{server.port}"}
        assert _request(server, "GET", "/", headers=host)[0] == 403
        assert _request(server, "GET", f"/?t={server.token}", headers=host)[0] == 200
        assert _request(server, "GET", f"/?t={server.token}",
                        headers={"Host": f"other.tail1234.ts.net:{server.port}"})[0] == 403
    finally:
        server.shutdown()
        server.server_close()


def test_background_preview_replaces_sample(tmp_path, monkeypatch):
    from yap_studio.cli import main

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(hyperframes, "installed", lambda: _install(tmp_path / "renderer"))
    script = tmp_path / "gsap.min.js"
    script.write_text("window.gsap={}")
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: script)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    entered, release, updated, serving = (threading.Event() for _ in range(4))
    holder = {}
    image = tmp_path / "take.jpg"
    image.write_bytes(b"synthetic")

    def fake_preview(_inbox, _work):
        entered.set()
        assert release.wait(10)
        return {"source": "take", "still": str(image), "sentence": "A real sentence.",
                "words": [{"text": "A", "start": 0.3, "end": 0.5}]}

    original_update = GalleryServer.update_preview
    original_serve = GalleryServer.serve_forever

    def update(server, result):
        original_update(server, result)
        updated.set()

    def serve(server):
        holder["server"] = server
        serving.set()
        original_serve(server)

    monkeypatch.setattr(still, "prepare_preview", fake_preview)
    monkeypatch.setattr(GalleryServer, "update_preview", update)
    monkeypatch.setattr(GalleryServer, "serve_forever", serve)
    thread = threading.Thread(target=lambda: holder.setdefault("result", main([
        "style", "pick", "--inbox", str(inbox), "--no-open", "--json"])), daemon=True)
    thread.start()
    assert serving.wait(10) and entered.wait(10)
    server = holder["server"]
    assert json.loads(_request(server, "GET", "/api/preview")[1])["source"] == "sample"
    release.set()
    assert updated.wait(10)
    after = json.loads(_request(server, "GET", "/api/preview")[1])
    assert after["source"] == "take" and after["sentence"] == "A real sentence."
    server.shutdown()
    thread.join(timeout=10)
    assert holder["result"] == 0


@pytest.mark.parametrize("preview_error", [False, True])
def test_page_updates_before_take_job_finishes_and_closes_group(
    tmp_path, monkeypatch, preview_error,
):
    from test_ingest import make_media
    from test_transcript_shared import CountingBackend
    from test_transcript_take_job import FAKE_JOB, group_gone, job_pids, wait_for

    from yap_studio import transcript
    from yap_studio.cli import main
    from yap_studio.transcript.take_job import TakeJobs

    monkeypatch.setattr(hyperframes, "installed", lambda: _install(tmp_path / "renderer"))
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _: None)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    make_media(inbox / "take.mov", seconds=43, sound="silence", fps=2)
    backend = CountingBackend()
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_: backend)
    script = tmp_path / "fake_job.py"
    script.write_text(FAKE_JOB)
    monkeypatch.setattr(TakeJobs, "child_command", [sys.executable, str(script)])
    monkeypatch.setattr(still, "pick_still", lambda *_args, **_kwargs: SimpleNamespace(
        path=still.sample_still(), time_s=1, scores={"faces": []}, reasons=(),
    ))
    prepare = still.prepare_preview

    def preview(inbox, work):
        if preview_error:
            raise OSError("synthetic preview failure")
        return prepare(inbox, work, backend=backend)

    monkeypatch.setattr(still, "prepare_preview", preview)
    observed = {}
    original_serve = GalleryServer.serve_forever

    def serve(server):
        observed["server"] = server
        original_serve(server)

    monkeypatch.setattr(GalleryServer, "serve_forever", serve)
    thread = threading.Thread(target=lambda: observed.setdefault("result", main([
        "style", "pick", "--inbox", str(inbox), "--no-open", "--json",
    ])), daemon=True)
    thread.start()
    try:
        record, pids = wait_for(job_pids)
        server = wait_for(lambda: observed.get("server"))
        result = json.loads(_request(server, "GET", "/api/preview")[1])
        if preview_error:
            assert result["source"] == "sample"
            assert backend.calls == []
        else:
            assert result["source"] == "take"
            assert result["transcript"] == "opening"
            assert backend.durations == [40.0]
        assert _request(server, "GET", f"/?t={server.token}")[0] == 200
        assert not group_gone(pids[0])
    finally:
        if "server" in observed:
            observed["server"].shutdown()
        thread.join(timeout=10)
    assert not thread.is_alive() and observed["result"] == 0
    assert group_gone(pids[0])
    assert not record.with_suffix(".json").exists()


@pytest.mark.parametrize("stale_field", ["take", "take_size", "take_mtime_ns", None])
def test_cached_take_requires_current_identity(tmp_path, monkeypatch, stale_field):
    from yap_studio.cli import main

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(hyperframes, "installed", lambda: _install(tmp_path / "renderer"))
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: None)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    old = inbox / "old.mp4"
    new = inbox / "new.mp4"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, ns=(1, 1))
    os.utime(new, ns=(2, 2))
    workspace = tmp_path / "workspace"
    work = workspace / ".yap-studio" / "gallery"
    work.mkdir(parents=True)
    cached_still = work / "still.jpg"
    cached_still.write_bytes(b"old image")
    cached = {
        "schema": "yap-studio/gallery-preview@1", "source": "take",
        "take": str(new), "take_size": new.stat().st_size,
        "take_mtime_ns": new.stat().st_mtime_ns, "still": str(cached_still),
        "sentence": "Old words", "words": [{"text": "Old", "start": 0.3, "end": 0.6}],
    }
    if stale_field == "take":
        cached["take"] = str(old)
    elif stale_field is not None:
        cached[stale_field] += 1
    (work / "preview.json").write_text(json.dumps(cached))
    entered, release = threading.Event(), threading.Event()
    observed = {}

    def prepare(_inbox, _work):
        entered.set()
        assert release.wait(10)
        return {"source": "sample", "still": str(still.sample_still()), "words": []}

    def serve(server):
        assert entered.wait(10)
        observed["preview"] = server.state()["preview"]
        observed["words"] = [word.text for word in server._words()]
        observed["still"] = server.still
        release.set()

    monkeypatch.setattr(still, "prepare_preview", prepare)
    monkeypatch.setattr(GalleryServer, "serve_forever", serve)
    assert main(["style", "pick", "--inbox", str(inbox), "--workspace", str(workspace),
                 "--no-open"]) == 0
    if stale_field is None:
        assert observed["preview"]["source"] == "take"
        assert observed["still"] == cached_still
        assert "Old" in observed["words"]
    else:
        assert observed["preview"]["source"] == "sample"
        assert observed["preview"]["pending"] is True
        assert observed["still"] == still.sample_still()
        assert "Old" not in observed["words"]


def test_failed_background_preview_keeps_sample_and_message(tmp_path, monkeypatch):
    from yap_studio.cli import main

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(hyperframes, "installed", lambda: _install(tmp_path / "renderer"))
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: None)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "new.mp4").write_bytes(b"synthetic")
    failed = threading.Event()
    observed = {}

    def prepare(_inbox, _work):
        raise OSError("bad take")

    def serve(server):
        assert failed.wait(10)
        observed["preview"] = server.state()["preview"]
        observed["still"] = server.still
        observed["words"] = [word.text for word in server._words()]

    original_fail = GalleryServer.fail_preview

    def fail(server):
        original_fail(server)
        failed.set()

    monkeypatch.setattr(still, "prepare_preview", prepare)
    monkeypatch.setattr(GalleryServer, "fail_preview", fail)
    monkeypatch.setattr(GalleryServer, "serve_forever", serve)
    assert main(["style", "pick", "--inbox", str(inbox), "--no-open"]) == 0
    assert observed["preview"]["message"] == (
        "Could not read your newest take; showing the sample still.")
    assert observed["preview"]["pending"] is False
    assert observed["preview"]["source"] == "sample"
    assert observed["still"] == still.sample_still()
    assert "Most" in observed["words"]


def test_every_card_kind_has_a_live_tile(tmp_path):
    server = GalleryServer(still.sample_still(), resolve("preset:clean_minimal").data,
                           tmp_path / "work", install=_install(tmp_path / "renderer"))
    try:
        for kind in cards.CARD_TEMPLATES:
            result = server.layers({"style": resolve("preset:clean_minimal").data,
                                    "tile": {"tab": "cards", "card": kind}})
            assert result["layers"][0]["template"] == kind
            assert result["layers"][0]["start_s"] > 0
    finally:
        server.server_close()


@pytest.fixture
def memory_gallery(tmp_path, monkeypatch):
    """Exercise the real request handler without binding or opening any sockets."""
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(gallery_server, "_GalleryHTTPServer", lambda *_args: SimpleNamespace(
        server_port=12345, server_close=lambda: None))
    server = GalleryServer(still.sample_still(), resolve("preset:clean_minimal").data,
                           tmp_path / "work", install=_install(tmp_path / "renderer"),
                           workspace=tmp_path / "workspace")
    yield server
    server.server_close()


def _memory_request(server, route, *, method="GET", data=None, headers=None):
    handler = object.__new__(server._handler_type())
    handler.path, handler.command = route, method
    body = json.dumps(data).encode() if data is not None else b""
    handler.headers = {"Host": f"127.0.0.1:{server.port}", **(headers or {})}
    if data is not None:
        handler.headers.update({"Content-Type": "application/json",
                                "Content-Length": str(len(body))})
    handler.rfile = io.BytesIO(body)
    replies = []
    handler.send = lambda status, payload, mime, csp="": replies.append((status, payload, csp))
    handler.dispatch()
    return replies[0]


def test_screen_tracks_only_the_current_preview(memory_gallery):
    server = memory_gallery
    preview = server.state()["preview"]
    assert preview["screen"] is None and preview["screen_checked"] is False
    server.update_screen(0, "green")
    preview = server.state()["preview"]
    assert preview["screen"] == "green" and preview["screen_checked"] is True
    server.update_preview({"still": str(server.still), "source": "take"})
    server.update_screen(0, "blue")
    preview = server.state()["preview"]
    assert preview["screen"] is None and preview["screen_checked"] is False
    server.update_screen(1, None)
    preview = server.state()["preview"]
    assert preview["screen"] is None and preview["screen_checked"] is True
    server.fail_preview()
    server.update_screen(1, "green")
    preview = server.state()["preview"]
    assert preview["screen"] is None and preview["screen_checked"] is False


@pytest.mark.parametrize("saved", [None, "green_screen", "talking_head", "invalid"])
def test_video_type_default_reaches_gallery_state(memory_gallery, saved):
    from yap_studio import video_type

    target = video_type.defaults_path()
    if saved == "invalid":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("invalid")
    elif saved is not None:
        video_type.save_default(saved, [])
    assert memory_gallery.state()["video_type"] == (None if saved == "invalid" else saved)


def test_cutout_tracks_only_the_current_preview(memory_gallery, tmp_path):
    from yap_studio import ffmpeg

    server = memory_gallery
    image = tmp_path / "person.png"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=c=blue:s=16x24,format=rgba", "-frames:v", "1", str(image)],
               timeout_s=30)
    assert server.state()["preview"]["cutout"] is None
    assert _memory_request(server, "/cutout?g=0")[0] == 404
    server.update_cutout(0, image)
    assert server.state()["preview"]["cutout"] == "/cutout?g=0"
    assert _memory_request(server, "/cutout?g=0")[:2] == (200, image.read_bytes())
    assert _memory_request(server, "/cutout?g=1")[0] == 404
    server.update_preview({"still": str(image), "source": "take"})
    assert server.state()["preview"]["cutout"] is None
    assert _memory_request(server, "/cutout?g=0")[0] == 404
    assert _memory_request(server, "/cutout?g=1")[0] == 404
    server.update_cutout(0, image)
    assert server.state()["preview"]["cutout"] is None
    server.update_cutout(1, image)
    assert server.state()["preview"]["cutout"] == "/cutout?g=1"
    assert _memory_request(server, "/cutout?g=1")[0] == 200
    server.fail_preview()
    assert server.state()["preview"]["cutout"] is None
    assert _memory_request(server, "/cutout?g=1")[0] == 404


@pytest.fixture
def rgba_cutout(memory_gallery):
    pixels = np.arange(6 * 8 * 4, dtype=np.uint8).reshape(6, 8, 4)
    pixels[..., 3] = np.resize([0, 63, 128, 255], (6, 8))
    raw = memory_gallery.work_dir / "person.rgba"
    raw.write_bytes(pixels.tobytes())
    image = memory_gallery.work_dir / "person.png"
    gallery_server.ffmpeg.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", "8x6", "-i", str(raw), "-frames:v", "1", "-threads", "1", str(image),
    ], timeout_s=30)
    memory_gallery.update_cutout(0, image)
    return pixels, image


@pytest.mark.parametrize(("look", "strength", "color"), [
    ("film", 1, None), ("film", 0.5, None), ("eyedropper", 0.5, "#1E6BD6"),
])
def test_cutout_grade_matches_rgb_math_and_preserves_alpha(memory_gallery, rgba_cutout,
                                                         look, strength, color):
    pixels, _ = rgba_cutout
    query = {"g": 0, "look": look, "strength": strength}
    if color:
        query["color"] = color
    status, body, _ = _memory_request(memory_gallery, "/cutout?" + urlencode(query))
    assert status == 200 and body.startswith(b"\x89PNG")
    image = memory_gallery.work_dir / "graded-person.png"
    image.write_bytes(body)
    decoded = gallery_server.ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(image), "-frames:v", "1",
        "-pix_fmt", "rgba", "-f", "rawvideo", "-",
    ], timeout_s=30, binary=True).stdout
    actual = np.frombuffer(decoded, np.uint8).reshape(pixels.shape)
    spec = gallery_server.GradeSpec(look, strength, color=color)
    expected = np.rint(gallery_server.apply_rgb(pixels[..., :3] / 255, spec) * 255)
    assert np.max(np.abs(actual[..., :3].astype(float) - expected)) <= 1
    np.testing.assert_array_equal(actual[..., 3], pixels[..., 3])
    assert np.any(actual[..., :3] != pixels[..., :3])
    assert _memory_request(memory_gallery, "/cutout?" + urlencode(query))[1] == body
    memory_gallery.update_preview({"still": str(image), "source": "take"})
    assert _memory_request(memory_gallery, "/cutout?" + urlencode(query))[0] == 404


def test_cutout_grades_are_bounded_and_cache_ungraded_pixels(
    memory_gallery, rgba_cutout, monkeypatch,
):
    _, image = rgba_cutout
    assert _memory_request(memory_gallery, "/cutout?look=none")[1] == image.read_bytes()
    assert _memory_request(memory_gallery, "/cutout?look=film&strength=0")[1] == image.read_bytes()
    calls = []
    decodes = []
    original = gallery_server.apply_rgb
    original_run = gallery_server.ffmpeg.run

    def measured(rgb, spec):
        calls.append(spec.strength)
        return original(rgb, spec)

    def measured_run(command, **kwargs):
        if "-i" in command and command[command.index("-i") + 1] == str(image):
            decodes.append(command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(gallery_server, "apply_rgb", measured)
    monkeypatch.setattr(gallery_server.ffmpeg, "run", measured_run)
    for value in range(1, gallery_server._STILL_CACHE_LIMIT + 1):
        assert _memory_request(memory_gallery, f"/cutout?look=film&strength={value / 10}")[0] == 200
    count = len(calls)
    assert _memory_request(memory_gallery, "/cutout?look=film&strength=0.1")[0] == 200
    assert len(calls) == count
    assert _memory_request(memory_gallery, "/cutout?look=film&strength=0.9")[0] == 200
    assert len(memory_gallery._cutout_cache) == gallery_server._STILL_CACHE_LIMIT
    assert _memory_request(memory_gallery, "/cutout?look=film&strength=0.1")[0] == 200
    assert len(calls) == count + 1
    assert _memory_request(memory_gallery, "/cutout?look=film&strength=0.2")[0] == 200
    assert len(calls) == count + 2
    assert len(decodes) == 1


def test_cutout_grade_discards_a_generation_changed_during_grading(memory_gallery, rgba_cutout,
                                                                 monkeypatch):
    _, image = rgba_cutout
    original = gallery_server.apply_rgb

    def replaced(rgb, spec):
        memory_gallery.update_preview({"still": str(image), "source": "take"})
        return original(rgb, spec)

    monkeypatch.setattr(gallery_server, "apply_rgb", replaced)
    assert _memory_request(memory_gallery, "/cutout?g=0&look=film")[0] == 404


@pytest.mark.parametrize("query", ["look=other", "strength=bad", "strength=nan", "strength=inf",
                                    "strength=-1", "strength=2", "color=bad",
                                    "look=eyedropper", "look=eyedropper&color=%23808080"])
def test_cutout_grade_validation_matches_still(memory_gallery, query):
    cutout = _memory_request(memory_gallery, "/cutout?" + query)
    still = _memory_request(memory_gallery, "/still?" + query)
    assert cutout[0] == still[0] == 400
    assert json.loads(cutout[1])["error"] == json.loads(still[1])["error"]


@pytest.mark.parametrize("query", ["g=-1", "g=no", "g=1&g=1", "look=other",
                                    "look=film&look=film", "strength=1&strength=0.5",
                                    "color=%231E6BD6&color=%23FF0000", "slider=1", "other=1"])
def test_cutout_rejects_bad_parameters(memory_gallery, query):
    assert _memory_request(memory_gallery, "/cutout?" + query)[0] == 400


@pytest.mark.parametrize("remote", [False, True, "tailnet"])
def test_cutout_auth_matches_still(memory_gallery, tmp_path, remote):
    server = memory_gallery
    image = tmp_path / "cutout.png"
    image.write_bytes(b"synthetic png")
    server.update_cutout(0, image)
    if remote == "tailnet":
        server.allowed_hosts.add("preview.test")
    elif remote:
        server.host = "0.0.0.0"
    for route in ("/still?g=0", "/cutout?g=0"):
        assert _memory_request(server, route)[0] == (403 if remote else 200)
        assert _memory_request(server, route + "&t=wrong")[0] == (403 if remote else 200)
        assert _memory_request(server, route + "&t=" + server.token)[0] == 200
        assert _memory_request(server, route, headers={
            "Cookie": f"yap_gallery={server.token}"})[0] == 200


def test_cutout_http_headers_match_still(gallery, tmp_path):
    image = tmp_path / "cutout.png"
    image.write_bytes(b"synthetic png")
    gallery.update_cutout(0, image)
    connection = http.client.HTTPConnection("127.0.0.1", gallery.port, timeout=10)
    for route, mime in (("/still?g=0", "image/svg+xml"), ("/cutout?g=0", "image/png")):
        connection.request("HEAD", route)
        response = connection.getresponse()
        assert response.status == 200 and response.read() == b""
        assert response.getheader("Content-Type") == mime
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert response.getheader("Referrer-Policy") == "no-referrer"
        assert response.getheader("Content-Security-Policy") is None
    connection.close()


@pytest.mark.parametrize("screen", ["green", None, "YapError", "OSError", "ValueError"])
def test_command_detects_screen_after_cutout(
    tmp_path, monkeypatch, _no_native_screen, capsys, screen,
):
    from yap_studio import ffmpeg, transcript
    from yap_studio.cli import main
    from yap_studio.errors import YapError
    from yap_studio.gallery import cutout
    from yap_studio.matte import chroma
    from yap_studio.transcript.take_job import TakeJobs

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    take = inbox / "take.mp4"
    background = "0x00ff00" if screen == "green" else "0x777777"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={background}:s=160x240:r=10:d=1,"
                "drawbox=x=50:y=50:w=60:h=190:color=0x888888:t=fill",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(take)], timeout_s=30)
    image = still.sample_still()
    monkeypatch.setattr(hyperframes, "installed", lambda: _install(tmp_path / "renderer"))
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: None)
    monkeypatch.setattr(still, "prepare_preview", lambda *_args: {
        "still": str(image), "source": "take", "take": str(take)})
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: None)
    monkeypatch.setattr(TakeJobs, "start", lambda *_args: None)
    order, holder = [], {}
    detected = threading.Event()

    def detect(source):
        order.append("detect")
        holder["thread"] = threading.current_thread()
        detected.set()
        assert source == take
        if screen in ("YapError", "OSError", "ValueError"):
            raise {"YapError": YapError, "OSError": OSError,
                   "ValueError": ValueError}[screen]("synthetic screen failure")
        return _no_native_screen(source)

    def serve(server):
        if detected.wait(10):
            holder["thread"].join(timeout=10)
            assert not holder["thread"].is_alive()
            holder["preview"] = server.state()["preview"]

    monkeypatch.setattr(cutout, "prepare_cutout", lambda *_args: order.append("cutout"))
    monkeypatch.setattr(chroma, "detect", detect)
    monkeypatch.setattr(GalleryServer, "serve_forever", serve)
    assert main(["style", "pick", "--inbox", str(inbox), "--no-open"]) == 0
    assert detected.is_set() and order == ["cutout", "detect"]
    preview = holder["preview"]
    if screen in ("YapError", "OSError", "ValueError"):
        assert preview["screen"] is None and preview["screen_checked"] is False
        assert capsys.readouterr().err.splitlines() == [
            "warning: screen could not be checked: synthetic screen failure"]
    else:
        assert preview["screen"] == screen and preview["screen_checked"] is True
        assert not capsys.readouterr().err


@pytest.mark.parametrize("cached", [False, True])
def test_command_prepares_cutout_for_start_and_updated_still(tmp_path, monkeypatch, cached):
    from yap_studio.cli import main
    from yap_studio.gallery import cutout

    monkeypatch.setattr(hyperframes, "installed", lambda: _install(tmp_path / "renderer"))
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: None)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    image, png = tmp_path / "still.png", tmp_path / "cutout.png"
    image.write_bytes(b"synthetic")
    png.write_bytes(b"synthetic")
    workspace = tmp_path / "workspace"
    if cached:
        take = inbox / "new.mp4"
        take.write_bytes(b"synthetic")
        folder = workspace / ".yap-studio" / "gallery"
        folder.mkdir(parents=True)
        (folder / "preview.json").write_text(json.dumps({
            "schema": "yap-studio/gallery-preview@1", "source": "take", "still": str(image),
            "take": str(take), "take_size": take.stat().st_size,
            "take_mtime_ns": take.stat().st_mtime_ns,
        }))
    calls, published, order = [], [], []
    done = threading.Event()

    def prepare(image, folder):
        order.append("cutout")
        calls.append((image, folder))
        return png

    original = GalleryServer.update_cutout

    def update(server, generation, path):
        original(server, generation, path)
        published.append(generation)
        if generation == 1:
            done.set()

    monkeypatch.setattr(cutout, "prepare_cutout", prepare)
    def preview(*_args):
        order.append("preview")
        return {"still": str(image), "source": "take"}

    monkeypatch.setattr(still, "prepare_preview", preview)
    monkeypatch.setattr(GalleryServer, "update_cutout", update)
    monkeypatch.setattr(GalleryServer, "serve_forever", lambda _server: done.wait(10))
    assert main(["style", "pick", "--inbox", str(inbox), "--workspace", str(workspace),
                 "--no-open"]) == 0
    assert done.is_set() and published == [1]
    assert [call[0] for call in calls] == [image]
    assert order == ["preview", "cutout"]


def test_memory_page_has_all_screens_closed_folds_and_strict_csp(memory_gallery):
    import re

    status, body, csp = _memory_request(memory_gallery, "/")
    assert status == 200 and csp == gallery_server._PAGE_CSP
    page = body.decode()
    panels = re.findall(r'class="step-panel" id="panel-([a-z-]+)"', page)
    assert panels == ["looks", "font", "captions", "colors", "top-text", "more", "grade", "opening",
                      "cards", "broll", "moves", "moments", "sound"]
    assert re.findall(r'data-step="([^"]+)"', page) == [
        "font", "captions", "colors", "top-text", "more",
    ]
    assert 'id="customize"' in page and 'id="back-to-looks"' in page
    assert page.count('>Save style</button>') == 2
    assert 'id="save-early"' not in page
    assert "Sets every choice" in page and "Most people save here." in page
    assert "More choices" in page and ">Optional</span>" in page
    assert not re.search(r"<details[^>]*\bopen\b", page)
    assert not re.search(r"\sstyle=", page)
    assert not re.search(r"<script(?![^>]*\bsrc=)", page)
    assert "unsafe-inline" not in csp
    assert 'id="font-mode"' not in page and 'id="font-role"' not in page
    assert 'id="reset"' in page and 'id="change-count"' in page
    assert "Zooms on your face and the cuts between shots." in page
    assert 'id="moves-more" class="more-fold"' in page
    assert 'id="moves-options"' in page
    assert "fonts.googleapis.com" not in page
    assert "Your brand" not in page
    colors = page.split('id="colors-more"', 1)[1].split('id="panel-more"', 1)[0]
    assert 'id="colors-fold"' not in page
    assert "<h3>Your colors</h3>" in colors
    assert colors.index('id="color-fields"') < colors.index("Pull colors from a website")


def test_memory_page_phone_preview_footer_and_save_sheet_contract(memory_gallery):
    page = _memory_request(memory_gallery, "/")[1].decode()
    css = _memory_request(memory_gallery, "/static/pick.css")[1].decode()
    controller = _memory_request(memory_gallery, "/static/pick.js")[1].decode()
    assert 'class="viewer sticky-preview"' in page
    assert 'id="main-frame" aria-expanded="false"' in page
    assert ".sticky-preview[data-editing=true]" in css
    assert ".main-frame.phone-expanded" in css
    assert 'classList.toggle("phone-expanded")' in controller
    footer = page.split('id="savebar"', 1)[1].split("<dialog", 1)[0]
    for control in ("back", "next", "customize", "save", "screen-summary", "change-count"):
        assert f'id="{control}"' in footer
    assert 'id="save" class="primary"' in footer
    assert 'id="next" class="primary"' not in footer
    assert 'id="style-name"' not in footer and 'id="reset"' not in footer
    sheet = page.split('<dialog id="save-sheet"', 1)[1].split("</dialog>", 1)[0]
    assert 'id="style-name"' in sheet and 'value="My style"' in sheet
    assert 'id="cancel-save"' in sheet and 'id="confirm-save"' in sheet
    assert '$("save-sheet").showModal()' in controller
    assert "prefers-reduced-motion: reduce" in css
    assert ".choice-row" in css and ".more-fold" in css and ".sub-fold" in css


def test_shot_photo_is_served_only_from_the_static_allowlist(memory_gallery):
    status, body, _ = _memory_request(memory_gallery, "/static/shot.jpg")
    assert status == 200
    assert gallery_server._STATIC_TYPES["shot.jpg"] == "image/jpeg"
    assert body[:3] == b"\xff\xd8\xff" and len(body) < 200_000
    assert _memory_request(memory_gallery, "/static/shot.svg")[0] == 404
    assert _memory_request(memory_gallery, "/static/missing.jpg")[0] == 404
    assert _memory_request(memory_gallery, "/static/../server.py")[0] == 404


def test_memory_font_names_load_bundled_and_creator_faces_locally(memory_gallery):
    from yap_studio.render import fonts, user_fonts

    directory = user_fonts.user_fonts_dir()
    directory.mkdir(parents=True)
    creator = directory / "creator.woff2"
    creator.write_bytes(b"synthetic creator font")
    (directory / "fonts.json").write_text(json.dumps({
        "schema": "yap-studio/user-fonts@1",
        "fonts": [{"family": "Creator Serif", "file": creator.name, "style": "normal",
                   "weight": "400", "sha256": "a" * 64}],
    }))
    page = _memory_request(memory_gallery, "/")[1].decode()
    assert '<link rel="stylesheet" href="/static/fonts.css">' in page
    assert "font-src 'self'" in gallery_server._PAGE_CSP
    assert "unsafe-inline" not in gallery_server._PAGE_CSP
    status, body, _ = _memory_request(memory_gallery, "/static/fonts.css")
    assert status == 200
    css = body.decode()
    for name in [*fonts.BUNDLED, "Creator Serif"]:
        assert fonts.font_face_css([name], url_prefix="/fonts/") in css
    for filename, source in fonts.font_files([*fonts.BUNDLED, "Creator Serif"]):
        status, body, _ = _memory_request(memory_gallery, f"/fonts/{filename}")
        assert status == 200 and body == source.read_bytes()
    for filename in ("missing.woff2", "fonts.json", "../fonts.json", "%2e%2e%2ffonts.json"):
        assert _memory_request(memory_gallery, f"/fonts/{filename}")[0] == 404
    creator.unlink()
    assert _memory_request(memory_gallery, "/fonts/creator.woff2")[0] == 404
    memory_gallery.allowed_hosts = {"box.tail1234.ts.net"}
    for route in ("/static/fonts.css", "/fonts/inter-variable.woff2"):
        assert _memory_request(memory_gallery, route)[0] == 403
        assert _memory_request(memory_gallery, route,
                               headers={"Cookie": f"yap_gallery={memory_gallery.token}"})[0] == 200


def test_memory_page_caption_controls_stay_inside_more_and_use_shared_posters(memory_gallery):
    page = _memory_request(memory_gallery, "/")[1].decode()
    controller = _memory_request(memory_gallery, "/static/pick.js")[1].decode()
    css = _memory_request(memory_gallery, "/static/pick.css")[1].decode()
    captions = page.split('id="panel-captions"', 1)[1].split('id="panel-colors"', 1)[0]
    assert captions.index('id="caption-look-tiles"') < captions.index('id="caption-animation"')
    more = captions.split('id="caption-animation"', 1)[1]
    for control in ("word-tiles", "highlight-tiles", "italics", "motion", "caption-color",
                    "background-fields", "avoid-face", "position-fields", "sizes", "fine-tune"):
        assert f'id="{control}"' in more
    assert "gallery-fold-" in controller
    assert "captionPoster(host, candidate)" in controller
    assert "drawPreview(host, candidate" in controller
    assert "poster.hidden = true" in controller
    assert "Loading preview" not in css
    for copy in ("Tap a look to try it on your video.", ">Customize</button>",
                 ">Back to looks</button>", "Reset changes", "Background strength",
                 "Italicize emphasized words", "More choices", ">Optional</span>"):
        assert copy in page
    assert '"Caption position"' in controller
    assert "Preferred vertical position" not in controller
    assert 'for="caption-color">Captions' not in captions
    assert page.index('id="font-options"') < page.index('id="width-tiles"')
    assert ': "Preset"' in controller
    assert "Swipe for more" not in page and "overflow-x: auto" not in css


def test_memory_start_library_is_newest_first_with_saved_names(memory_gallery):
    first = save_pick(resolve("preset:electric").data, name="SparrowLight")
    newer = save_pick(resolve("preset:warm_editorial").data, name="Learned custom name")
    library = first.path.parent
    first.path.rename(library / "learned.json")
    (library / "learned.source.json").write_text(json.dumps({"videos": 6}))
    os.utime(library / "learned.json", ns=(1, 1))
    os.utime(newer.path, ns=(2, 2))
    state = memory_gallery.state()
    assert [(item["id"], item["name"]) for item in state["library"]] == [
        (newer.id, "Learned custom name"), ("learned", "SparrowLight")]
    assert state["library"][1]["videos"] == 6
    assert [item["id"] for item in state["presets"]] == list(gallery_server.SHIPPED_PRESETS)
    assert state["library"][1]["style"]["name"] == "SparrowLight"


def test_memory_site_colors_requires_token_and_checks_origin(memory_gallery, monkeypatch):
    calls = []
    monkeypatch.setattr(gallery_server, "site_colors",
                        lambda url: calls.append(url) or {"site": url})
    route = "/api/site-colors?url=example.com"
    assert _memory_request(memory_gallery, route)[0] == 403
    assert _memory_request(memory_gallery, route + "&t=wrong")[0] == 403
    token = memory_gallery.token
    assert _memory_request(memory_gallery, route + f"&t={token}",
                           headers={"Origin": "https://evil.example"})[0] == 403
    assert calls == []
    assert _memory_request(memory_gallery, route + f"&t={token}")[0] == 200
    assert _memory_request(memory_gallery, route,
                           headers={"Cookie": f"yap_gallery={token}"})[0] == 200
    assert calls == ["example.com"]


def test_memory_site_colors_cache_includes_concurrent_requests(memory_gallery, monkeypatch):
    calls = []
    result = {"site": "example.com", "suggestion": {"accent": "#2F3BFF"}}
    monkeypatch.setattr(gallery_server, "site_colors", lambda url: calls.append(url) or result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(memory_gallery.website_colors, ["example.com"] * 4))
    assert replies == [result] * 4
    assert calls == ["example.com"]
    route = "/api/site-colors?" + urlencode({"url": "example.com", "t": memory_gallery.token})
    assert json.loads(_memory_request(memory_gallery, route)[1]) == result
    assert calls == ["example.com"]


@pytest.mark.parametrize("url,message", [
    ("", "Enter a website address."),
    ("ftp://example.com", "Use a website address like example.com."),
    ("https://example.com:bad", "Use a website address like example.com."),
])
def test_memory_site_colors_bad_url_uses_engine_message(memory_gallery, url, message):
    route = "/api/site-colors?" + urlencode({"url": url, "t": memory_gallery.token})
    status, body, _ = _memory_request(memory_gallery, route)
    assert status == 400
    assert json.loads(body) == {"error": message}


def test_memory_site_colors_private_address_refusal_is_from_engine(memory_gallery, monkeypatch):
    from importlib import import_module

    engine = import_module("yap_studio.style.site_colors")
    monkeypatch.setattr(engine.socket, "getaddrinfo", lambda *_args, **_kwargs: [
        (2, 1, 6, "", ("127.0.0.1", 80))])
    monkeypatch.setattr(engine.socket, "create_connection",
                        lambda *_args, **_kwargs: pytest.fail("private address must not connect"))
    route = "/api/site-colors?" + urlencode({"url": "http://private.example",
                                           "t": memory_gallery.token})
    status, body, _ = _memory_request(memory_gallery, route)
    assert status == 400
    assert json.loads(body) == {"error": "That address is not a public website."}


def test_memory_save_roundtrip_all_steps_and_reloaded_server_state(memory_gallery):
    style = resolve("preset:electric").data
    style["captions"].update(max_words_per_phrase=3, max_characters_per_phrase=18,
                             highlight_style="pill", italic_stress=True, width_scale=0.79,
                             font="Instrument Serif", text_color="#FFF4DD",
                             highlight_color="#2F3BFF", highlight_text_color="#FFFFFF",
                             entry_rise_px=40, entry_ms=160)
    style["hook"].update(style="none", font="Instrument Serif", background_color="#F6F1E7")
    style["graphics"].update(style="bright", font="Instrument Serif", card_kinds_off=["stat"],
                             paper_color="#F6F1E7", ink_color="#222222", titlebar_color="#E5E0D7")
    style["music"]["level"] = 0.13
    style["grade"] = {"look": "eyedropper", "strength": 0.5, "color": "#1E6BD6",
                      "slider": True}
    status, body, _ = _memory_request(memory_gallery, "/api/save", method="POST",
                                     data={"style": style, "name": "SparrowLight"},
                                     headers={"X-Gallery-Token": memory_gallery.token})
    assert status == 200
    saved = json.loads(body)
    state = memory_gallery.state()
    assert state["from_library"] is True
    assert state["style"]["name"] == "SparrowLight"
    restored = resolve(saved["id"]).data
    for section in ("brand", "captions", "hook", "graphics", "music", "grade"):
        assert restored[section] == state["style"][section] == style[section]
        assert state["library"][0]["style"][section] == style[section]
    workspace_copy = memory_gallery.workspace / "style/my-style.json"
    assert json.loads(workspace_copy.read_text())["name"] == "SparrowLight"


@pytest.mark.parametrize("mode", ["none", "box", "band"])
@pytest.mark.parametrize("placement", ["avoid_face", "fixed"])
def test_memory_background_and_placement_save_state_and_template_roundtrip(
    memory_gallery, mode, placement,
):
    state = memory_gallery.state()
    for preset in state["presets"]:
        captions = preset["style"]["captions"]
        assert captions["background_mode"] == "none"
        assert captions["placement"] == "avoid_face"
        assert captions["background_color"] == "#000000"
        assert captions["background_opacity"] == 0.85
        assert captions["background_padding_px"] == 18
        assert captions["background_radius_px"] == 12
    style = state["style"]
    expected = {"background_mode": mode, "background_color": "#FFFFFF",
                "background_opacity": 0.63, "background_padding_px": 27,
                "background_radius_px": 23, "placement": placement,
                "position": "middle", "top_px": 700, "text_color": "#ABCDEF"}
    style["captions"].update(expected)
    status, body, _ = _memory_request(memory_gallery, "/api/save", method="POST",
                                     data={"style": style, "name": "Caption background"},
                                     headers={"X-Gallery-Token": memory_gallery.token})
    assert status == 200
    saved = resolve(json.loads(body)["id"])
    state = memory_gallery.state()
    workspace_copy = memory_gallery.workspace / "style/my-style.json"
    raw = json.loads(workspace_copy.read_text())
    for data in (state["style"], state["library"][0]["style"], saved.unbranded_data, raw):
        assert {key: data["captions"][key] for key in expected} == expected
    resolved = current_default_data()["captions"]
    assert resolved["text_color"] == ("#ABCDEF" if mode == "none" else "#111111")
    assert {key: resolved[key] for key in expected if key != "text_color"} == {
        key: value for key, value in expected.items() if key != "text_color"}
    reloaded = GalleryServer(still.sample_still(), resolve().unbranded_data,
                             memory_gallery.work_dir / "reopened",
                             install=_install(memory_gallery.work_dir / "reopened-renderer"),
                             workspace=memory_gallery.workspace)
    try:
        restored = reloaded.state()
        assert {key: restored["style"]["captions"][key] for key in expected} == expected
        assert {key: restored["library"][0]["style"]["captions"][key]
                for key in expected} == expected
    finally:
        reloaded.server_close()


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("color,text", [("#FFFFFF", "#111111"), ("#000000", "#FFFFFF")])
def test_memory_background_preview_resolves_engine_contrast_without_mutating_style(
    memory_gallery, mode, color, text,
):
    style = memory_gallery.style
    style["captions"].update(background_mode=mode, background_color=color,
                             background_opacity=0.63, background_padding_px=27,
                             background_radius_px=23, text_color="#ABCDEF", placement="fixed")
    result = memory_gallery.layers({"style": style, "tile": {"tab": "captions"}})
    digest = result["layers"][0]["src"].split("/")[2]
    variables = memory_gallery._cache[digest][1][0].variables
    for field in ("background_mode", "background_color", "background_opacity",
                  "background_padding_px", "background_radius_px"):
        assert variables[field] == style["captions"][field]
    assert variables["text_color"] == text
    assert style["captions"]["text_color"] == "#ABCDEF"


def test_memory_main_preview_never_uses_stat_samples(memory_gallery):
    memory_gallery.preview.update(sentence="Your own opening words. Another sentence.", words=[])
    for step in ("looks", "start", "templates", "font", "captions", "colors", "more",
                 "words", "highlight", "title", "grade", "opening", "cards", "broll",
                 "moves", "moments", "sound"):
        result = memory_gallery.layers({"style": memory_gallery.style,
                                        "tile": {"tab": "preview", "step": step, "card": "stat"}})
        layers = result["layers"]
        assert all(layer["template"] != "stat" for layer in layers)
        card = next((layer for layer in layers if layer["name"] == "card"), None)
        if step not in {"colors", "cards", "more"}:
            assert card is None
        else:
            assert card["template"] == "text_tag"
            digest = card["src"].split("/")[2]
            composed = next(layer for layer in memory_gallery._cache[digest][1]
                            if layer.name == "card")
            assert composed.variables["text"] == "Your own opening words."
            assert "end" not in composed.variables
    disabled = {**memory_gallery.style, "graphics": {
        **memory_gallery.style["graphics"], "card_kinds_off": ["text_tag"]}}
    result = memory_gallery.layers({"style": disabled, "tile": {"tab": "preview", "step": "cards"}})
    assert all(layer["name"] != "card" for layer in result["layers"])


def test_memory_font_graphics_role_previews_a_card_only_when_requested(memory_gallery):
    style = memory_gallery.style
    style["captions"]["font"] = "Instrument Serif"
    style["hook"]["font"] = style["graphics"]["font"] = "Geist"
    for section in (None, "cards"):
        tile = {"tab": "preview", "step": "font"}
        if section:
            tile["section"] = section
        result = memory_gallery.layers({"style": style, "tile": tile})
        card = next((layer for layer in result["layers"] if layer["name"] == "card"), None)
        if section:
            assert card is not None
            assert card["template"] == "text_tag"
        else:
            assert card is None


@pytest.mark.parametrize("card_state", ["on", "off", "no_text"])
@pytest.mark.parametrize("title_style", ["quiet_serif", "none"])
def test_memory_font_graphics_role_falls_back_to_title_without_a_card(
    memory_gallery, monkeypatch, card_state, title_style,
):
    style = memory_gallery.style
    style["hook"]["style"] = title_style
    style["graphics"]["card_kinds_off"] = ["text_tag"] if card_state == "off" else []
    if card_state == "no_text":
        monkeypatch.setattr(live, "sentence_card_text", lambda *_args: None)
    for step in ("font", "colors", "more"):
        result = memory_gallery.layers({
            "style": style, "tile": {"tab": "preview", "step": step, "section": "cards"},
        })
        names = {layer["name"] for layer in result["layers"]}
        assert "captions" in names
        assert ("card" in names) == (card_state == "on")
        assert ("title" in names) == (
            step == "font" and card_state != "on" and title_style != "none")


def test_memory_each_card_style_and_kind_prepares_real_compositions(memory_gallery):
    for look in ("paper", "minimal", "dark", "bright"):
        result = memory_gallery.layers({"style": memory_gallery.style,
                                        "tile": {"tab": "cards", "look": look}})
        layer = result["layers"][0]
        assert layer["template"] == "text_tag"
        digest = layer["src"].split("/")[2]
        variables = memory_gallery._cache[digest][1][0].variables
        assert variables["text"] == "Most people think this takes hours."
    for kind in cards.CARD_TEMPLATES:
        result = memory_gallery.layers({"style": memory_gallery.style,
                                        "tile": {"tab": "cards", "card": kind}})
        assert result["layers"][0]["template"] == kind
        assert result["layers"][0]["start_s"] > 0


def test_memory_runtime_injection_and_lru_still_work(memory_gallery):
    memory_gallery._cache_limit = 2
    results = []
    for look in ("pill", "underline"):
        result = memory_gallery.layers({"style": memory_gallery.style,
                                        "tile": {"tab": "captions", "look": look}})
        results.append(result["layers"][0]["src"].split("/")[2])
    body, mime, csp = memory_gallery.layer_file(results[0], 0, "")
    assert mime == "text/html; charset=utf-8" and csp == gallery_server._LAYER_CSP
    assert b"window.__hfVariables=" in body and b'src="/hf/runtime.js"' in body
    source = memory_gallery._cache[results[0]][1][0].composition / "index.html"
    assert b"window.__hfVariables=" not in source.read_bytes()
    memory_gallery.layers({"style": memory_gallery.style,
                           "tile": {"tab": "captions", "look": "color"}})
    assert results[0] in memory_gallery._cache and results[1] not in memory_gallery._cache
    with pytest.raises(gallery_server._RequestError, match="layer not found"):
        memory_gallery.layer_file(results[1], 0, "")


@pytest.mark.parametrize(
    "step", ("looks", "start", "templates", "font", "captions", "colors", "more",
             "words", "highlight", "title", "grade", "opening", "cards", "broll",
             "moves", "moments", "sound"),
)
def test_memory_main_preview_replaces_title_with_card_on_colors_and_more(memory_gallery, step):
    result = memory_gallery.layers({"style": memory_gallery.style,
                                    "tile": {"tab": "preview", "step": step}})
    names = [layer["name"] for layer in result["layers"]]
    assert "captions" in names
    assert not ("title" in names and "card" in names)
    assert ("title" in names) == (step not in {"colors", "cards", "more"})
    assert ("card" in names) == (step in {"colors", "cards", "more"})
    if step in {"colors", "cards", "more"}:
        disabled = {**memory_gallery.style, "graphics": {
            **memory_gallery.style["graphics"], "card_kinds_off": ["text_tag"]}}
        result = memory_gallery.layers({"style": disabled,
                                        "tile": {"tab": "preview", "step": step}})
        assert [layer["name"] for layer in result["layers"]] == ["captions"]


def test_memory_template_tiles_have_caption_and_title_poster_metadata(memory_gallery):
    for preset in memory_gallery.state()["presets"]:
        result = memory_gallery.layers({"style": preset["style"],
                                        "tile": {"tab": "preview", "step": "templates"}})
        assert [layer["name"] for layer in result["layers"]] == ["captions", "title"]
        for layer in result["layers"]:
            assert 0 < layer["poster_s"] < layer["duration_s"]


def test_memory_dark_sign_has_dark_paper_when_the_choice_saves_dark_colors(memory_gallery):
    style = resolve("preset:clean_minimal").data
    style["graphics"].update(style="dark", paper_color="#222222", ink_color="#F4F3EE")
    result = memory_gallery.layers({"style": style, "tile": {"tab": "cards"}})
    digest = result["layers"][0]["src"].split("/")[2]
    variables = memory_gallery._cache[digest][1][0].variables
    assert variables["look"] == "card"
    assert variables["paper_color"] == "#222222"
    assert variables["ink_color"] == "#F4F3EE"


@pytest.mark.parametrize("preset", ("hf_bold", "hf_editorial", "hf_neutral"))
def test_memory_theme_template_saves_directly_and_becomes_the_default(memory_gallery, preset):
    state = memory_gallery.state()
    selected = next(item for item in state["presets"] if item["id"] == preset)
    assert selected["name"] == {"hf_bold": "Bold", "hf_editorial": "Editorial",
                                "hf_neutral": "Neutral"}[preset]
    result = memory_gallery.layers({"style": selected["style"],
                                    "tile": {"tab": "preview", "step": "templates"}})
    assert [layer["name"] for layer in result["layers"]] == ["captions", "title"]
    status, body, _ = _memory_request(memory_gallery, "/api/save", method="POST",
                                     data={"style": selected["style"], "name": "My template"},
                                     headers={"X-Gallery-Token": memory_gallery.token})
    assert status == 200
    saved = resolve(json.loads(body)["id"]).data
    assert saved["captions"] == selected["style"]["captions"]
    assert saved["graphics"] == selected["style"]["graphics"]
    assert memory_gallery.state()["from_library"] is True


def test_memory_top_placement_and_shadow_reach_the_engine_composition(memory_gallery):
    style = resolve("preset:clean_minimal").data
    style["captions"].update(position="middle", top_px=700, size_px=118,
                             shadow_opacity=0.8, shadow_blur_px=3,
                             shadow_x_px=0, shadow_y_px=6)
    result = memory_gallery.layers({"style": style,
                                    "tile": {"tab": "preview", "step": "captions"}})
    layer = result["layers"][0]
    digest = layer["src"].split("/")[2]
    composed = memory_gallery._cache[digest][1][0]
    assert composed.variables["position"] == "middle"
    assert layer["y"] < 700
    for field in ("size_px", "shadow_opacity", "shadow_blur_px", "shadow_x_px", "shadow_y_px"):
        assert composed.variables[field] == style["captions"][field]


@pytest.mark.parametrize("section,names", [
    ("title", ["captions", "title"]), ("cards", ["captions", "card"]),
])
def test_memory_more_preview_follows_its_selected_section(memory_gallery, section, names):
    result = memory_gallery.layers({"style": memory_gallery.style,
                                    "tile": {"tab": "preview", "step": "more",
                                             "section": section}})
    assert [layer["name"] for layer in result["layers"]] == names
    assert all(layer["template"] != "stat" for layer in result["layers"])
