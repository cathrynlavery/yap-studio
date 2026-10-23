"""Shipped presets stay usable in the engine and style page."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.render.fonts import BUNDLED
from yap_studio.stages.captions import (
    caption_band_geometry,
    caption_layer_plan,
    caption_layout,
    caption_shrink,
    render_layer_plan,
)
from yap_studio.style import PRESETS_DIR, SHIPPED_PRESETS, resolve

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_shipped_preset_resolves(preset: str) -> None:
    style = resolve(f"preset:{preset}")
    assert style.source == f"preset:{preset}"
    assert style.name
    assert style.data["broll"]["color_match"] == "on"


@pytest.mark.parametrize("preset", ["editorial_serif", "one_word"])
def test_new_presets_anchor_captions_at_the_bottom(preset: str) -> None:
    style = resolve(f"preset:{preset}")
    captions = style.data["captions"]
    assert captions["position"] == "bottom"
    assert captions["top_px"] == 1503
    geometry = caption_band_geometry(
        captions, safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
    )
    assert geometry.warnings == ()


def test_one_word_splits_every_word_into_its_own_phrase() -> None:
    captions = resolve("preset:one_word").data["captions"]
    words = [
        CaptionWord(str(index), text, index * 0.4, index * 0.4 + 0.3)
        for index, text in enumerate(("We", "build", "understanding"))
    ]
    assert captions["max_words_per_phrase"] == 1
    assert [phrase.words[0].text for phrase in build_phrases(words, captions)] == [
        "We", "build", "understanding",
    ]


def test_one_word_long_caption_band_stays_in_safe_area() -> None:
    style = resolve("preset:one_word")
    phrases = build_phrases(
        [CaptionWord("long", "understanding", 0.1, 0.8)], style.data["captions"],
    )
    plan = caption_layer_plan(phrases, style, frames=30)
    assert plan.warnings == ()
    assert plan.y + plan.height <= style.data["internal"]["caption_safe_bottom_px"]
    assert plan.variables["size_px"] == 110


@pytest.mark.render
def test_one_word_long_caption_fits_in_rendered_safe_area(tmp_path: Path) -> None:
    style = resolve("preset:one_word")
    phrases = build_phrases(
        [CaptionWord("long", "understanding", 0.1, 0.8)], style.data["captions"],
    )
    plan = caption_layer_plan(phrases, style, frames=30)
    result = render_layer_plan(plan, tmp_path / "work", tmp_path / "caption.mov")
    assert caption_shrink(result.checks, plan) == {
        "measured": True, "floor_px": 32, "style_size_px": 110, "phrases": [],
    }
    layout = caption_layout(result.checks, plan)
    assert layout["measured"] is True
    boxes = layout["captions"][0]["boxes"]
    assert boxes
    assert all(
        0 <= box["x"] and box["x"] + box["w"] <= 1080
        and 0 <= box["y"] and box["y"] + box["h"] <= 1740
        for box in boxes
    )


def test_editorial_serif_uses_instrument_serif() -> None:
    style = resolve("preset:editorial_serif").data
    assert style["captions"]["font"] == "Instrument Serif"
    assert style["hook"]["font"] == "Instrument Serif"


def test_prototype_includes_new_shipped_presets() -> None:
    source = ROOT / "docs" / "design" / "style-page" / "style-data.js"
    text = source.read_text(encoding="utf-8")
    data = json.loads(text.split("window.YAP_STYLE_DATA = ", 1)[1].removesuffix(";\n"))
    by_id = {preset["id"]: preset for preset in data["presets"]}
    for preset in SHIPPED_PRESETS:
        assert by_id[preset]["proposed"] is False
        assert by_id[preset]["style"]["name"] == resolve(f"preset:{preset}").name
    app = (source.parent / "app.js").read_text(encoding="utf-8")
    order = re.search(r"const PRESET_ORDER = \[(.*?)\];", app)
    assert order is not None
    ids = re.findall(r"'([a-z_]+)'", order[1])
    assert ids[0] == "editorial_serif"
    assert ids[-1] == "hf_neutral"
    assert ids == list(SHIPPED_PRESETS)
    assert set(ids) == set(by_id)


_PROPOSED_DIGESTS = {
    "cobalt": "bc2014ba54905b8f018bd8768584d8088410b402c41b84d098976228b611d0cc",
    "mono": "6cdac333146f515a7c4b8e15e1e6a1dc3cea851e4f003ae2c1188735936b21a4",
    "headline": "59e2a403e4781b79826d365161af458157e68a2305b8f233dc5a850e155cf2e6",
    "candy": "44117b5ba5d462aaaa3aa81647410e5fa8181d45208dfb8fb4ebb1a0bc5ea6d6",
    "marker": "0bacc58c5552f777c78f5aa6faf906581e4507a0d4f7fb888f3bb6bc73450d4d",
}


@pytest.mark.parametrize("preset", _PROPOSED_DIGESTS)
def test_promoted_presets_keep_the_approved_proposed_file_exactly(preset):
    source = PRESETS_DIR / f"{preset}.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == _PROPOSED_DIGESTS[preset]
    assert not (ROOT / "docs/design/style-page/proposed-presets" / source.name).exists()


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_all_preset_fonts_are_bundled(preset):
    style = resolve(f"preset:{preset}").data
    for section in ("captions", "hook", "graphics"):
        assert style[section]["font"] in BUNDLED


def test_all_site_styles_are_shipped_and_no_proposals_remain():
    assert list(SHIPPED_PRESETS) == [
        "editorial_serif", "clean_minimal", "warm_editorial", "cobalt", "mono", "headline",
        "electric", "candy", "marker", "one_word", "hf_bold", "hf_editorial", "hf_neutral",
    ]
    source = ROOT / "docs/design/style-page/proposed-presets/order.json"
    assert json.loads(source.read_text()) == []


@pytest.mark.parametrize("preset,display,body", [
    ("hf_bold", "Anton SC", "Inter"),
    ("hf_editorial", "Gelasio", "Gelasio"),
    ("hf_neutral", "Inter", "Inter"),
])
def test_theme_presets_map_upstream_tokens_with_recorded_bundled_substitutions(
    preset, display, body,
):
    provenance = json.loads((PRESETS_DIR.parent / "PROVENANCE.json").read_text())
    assert provenance["schema"] == "yap-studio/provenance@1"
    upstream = provenance["upstream"]
    assert upstream["commit"] == "6afcb48395632b85663b9b9486e1b47fd175f8bc"
    assert upstream["license"] == "Apache-2.0"
    license_file = ROOT / upstream["license_file"]
    assert hashlib.sha256(license_file.read_bytes()).hexdigest() == upstream["license_sha256"]
    entry = next(item for item in provenance["files"]
                 if item["yap_path"].endswith(f"{preset}.json"))
    assert entry["upstream_path"] == f"themes/{preset.removeprefix('hf_')}.css"
    assert re.fullmatch(r"[0-9a-f]{64}", entry["upstream_sha256"])
    assert entry["font_substitutions"] == {"font-display": display, "font-body": body}
    assert len(entry["tokens"]) == 18
    style = resolve(f"preset:{preset}").data
    assert style["captions"]["font"] == style["hook"]["font"] == display
    assert style["graphics"]["font"] == body
    for field, token in entry["field_tokens"].items():
        section, key = field.split(".")
        assert style[section][key].upper() == entry["tokens"][token].upper()
    assert resolve("preset:hf_editorial").name != resolve("preset:editorial_serif").name
