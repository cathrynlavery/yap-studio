"""Giant captions derive layout values without changing the creator's style."""

import copy
import json
from pathlib import Path

import pytest
from test_captions_render import _caption_dom, _dom_nodes, _phrases, _style

from yap_studio import captions as caption_module
from yap_studio.faces.model import Box
from yap_studio.gallery.live import caption_plan, sample_words
from yap_studio.stages.captions import caption_band_geometry, caption_layer_plan, caption_offsets
from yap_studio.style import ResolvedStyle, loader, resolve, resolve_style_data, save_pick
from yap_studio.taste import Taste, render_style


def test_giant_resolved_document_keeps_source_values_and_is_idempotent(tmp_path: Path) -> None:
    source = {"schema_version": 1, "brand": {"accent_color": "#ABCDEF"}, "captions": {
        "highlight_style": "giant", "size_px": 83, "max_words_per_phrase": 5,
        "max_characters_per_phrase": 24, "background_mode": "none",
        "background_opacity": 0.21, "background_color": "#123456",
        "background_padding_px": 2, "text_color": "#765432",
    }}
    before = copy.deepcopy(source)
    data = resolve_style_data(source)
    assert all(data["captions"][key] == value for key, value in source["captions"].items())
    original = copy.deepcopy(data)
    for _ in range(2):
        document = ResolvedStyle(data["name"], "test", data).to_json()
        data = resolve_style_data(document)
        assert data == original
    file = tmp_path / "resolved.json"
    file.write_text(json.dumps(document))
    loaded = resolve(file)
    assert loaded.data == original
    assert render_style(loaded, snapshot=Taste().snapshot()).style.data == original
    assert source == before


@pytest.mark.parametrize("mode", ["none", "box", "band"])
def test_giant_resolves_like_other_looks(mode: str) -> None:
    source = {"schema_version": 1, "brand": {"accent_color": "#ABCDEF"}, "captions": {
        "highlight_style": "none", "size_px": 83, "max_words_per_phrase": 5,
        "max_characters_per_phrase": 24, "background_mode": mode,
        "background_opacity": 0.8, "background_color": "#765432",
        "background_padding_px": 2, "text_color": "#123456",
    }}
    ordinary = resolve_style_data(source)
    source["captions"]["highlight_style"] = "giant"
    giant = resolve_style_data(source)
    assert giant["captions"] == {**ordinary["captions"], "highlight_style": "giant"}


@pytest.mark.parametrize("has_source", [False, True])
def test_giant_taste_resolution_is_repeatable(has_source: bool) -> None:
    source = {"schema_version": 1, "captions": {
        "highlight_style": "giant", "size_px": 83, "max_characters_per_phrase": 24,
    }}
    data = resolve_style_data(source)
    style = ResolvedStyle(data["name"], "test", data,
                          unbranded_data=loader.unbranded(source) if has_source else None)
    before = copy.deepcopy(style)
    original = style
    for _ in range(3):
        style = render_style(style, snapshot=Taste().snapshot()).style
        assert style.data == data
    assert original == before
    assert data["captions"]["size_px"] == 83
    assert data["captions"]["max_characters_per_phrase"] == 24


@pytest.mark.parametrize("size,expected", [(51, 76), (76, 114), (120, 180), (180, 180)])
@pytest.mark.parametrize("accent", ["", "#ABCDEF"])
def test_giant_layout_is_pure_and_keeps_other_caption_choices(
    size: int, expected: int, accent: str,
) -> None:
    captions = loader.unbranded({"schema_version": 1, "captions": {
        "size_px": size, "highlight_style": "giant", "background_color": "#123456",
        "background_padding_px": 2, "background_radius_px": 27,
        "max_words_per_phrase": 8, "max_characters_per_phrase": 37,
        "width_scale": 0.79, "italic_stress": True, "highlight_color": "#FFFFFF",
    }})["captions"]
    before = copy.deepcopy(captions)
    brand = {"accent_color": accent}
    result = caption_module.caption_layout_settings(captions, brand)
    changes = {
        "size_px": expected, "background_mode": "box", "background_opacity": 1,
        "background_color": accent or "#123456",
        "background_padding_px": round(expected * 0.18),
        "max_words_per_phrase": 3, "max_characters_per_phrase": 24,
        "text_color": "#111111" if accent else "#FFFFFF",
        "highlight_color": "#111111" if accent else "#FFFFFF",
    }
    assert result == {**captions, **changes}
    assert captions == before and brand == {"accent_color": accent}


@pytest.mark.parametrize("look", [look for look in loader._schema()["properties"]["captions"][
    "properties"]["highlight_style"]["enum"] if look != "giant"])
def test_other_looks_keep_their_layout_settings(look: str) -> None:
    captions = _style(highlight=look).data["captions"]
    before = copy.deepcopy(captions)
    assert caption_module.caption_layout_settings(captions, {"accent_color": "#ABCDEF"}) == before
    assert captions == before


@pytest.mark.parametrize("words,chars,expected_words,expected_chars", [
    (1, 8, 1, 8), (2, 16, 2, 10), (3, 18, 3, 12), (8, 48, 3, 32),
])
def test_giant_layout_limits_and_existing_padding(
    words: int, chars: int, expected_words: int, expected_chars: int,
) -> None:
    data = resolve_style_data({"schema_version": 1, "captions": {
        "highlight_style": "giant", "background_padding_px": 80,
        "max_words_per_phrase": words, "max_characters_per_phrase": chars,
    }})
    captions = caption_module.caption_layout_settings(data["captions"], data["brand"])
    assert captions["background_padding_px"] == 80
    assert captions["max_words_per_phrase"] == expected_words
    assert captions["max_characters_per_phrase"] == expected_chars


@pytest.mark.parametrize("block", ["#FFFFFF", "#000000", "#777777", "#FFFF00"])
@pytest.mark.parametrize("accent", [False, True])
def test_giant_layout_has_readable_text_and_stress(block: str, accent: bool) -> None:
    source = {"schema_version": 1, "brand": {"accent_color": block} if accent else {},
              "captions": {"highlight_style": "giant", "background_color": block,
                           "highlight_color": block, "text_color": block}}
    data = resolve_style_data(source)
    before = copy.deepcopy(data)
    captions = caption_module.caption_layout_settings(data["captions"], data["brand"])
    assert captions["background_color"] == block
    assert loader._contrast(captions["text_color"], block) >= 4.5
    assert captions["highlight_color"] == captions["text_color"]
    assert data == before


@pytest.mark.parametrize("highlight,expected", [
    ("#555555", "#555555"), ("#888888", "#888888"), ("#AAAAAA", "#111111"),
])
def test_giant_layout_highlight_fallback_uses_three_to_one(highlight: str, expected: str) -> None:
    data = resolve_style_data({"schema_version": 1, "captions": {
        "highlight_style": "giant", "background_color": "#FFFFFF", "highlight_color": highlight,
    }})
    captions = caption_module.caption_layout_settings(data["captions"], data["brand"])
    assert captions["highlight_color"] == expected


def test_giant_save_and_switch_back_preserve_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    source = {"schema_version": 1, "name": "My captions", "captions": {
        "highlight_style": "color", "size_px": 83, "background_mode": "none",
        "background_color": "#765432", "background_opacity": 0.21,
        "max_words_per_phrase": 7, "max_characters_per_phrase": 39,
    }, "brand": {"accent_color": "#ABCDEF"}}
    original = copy.deepcopy(source)
    previous = resolve_style_data(source)
    source["captions"]["highlight_style"] = "giant"
    entry = save_pick(source)
    style = resolve(entry.path)
    assert style.data["captions"]["size_px"] == 83
    assert json.loads(entry.path.read_text()) == source
    plan = caption_plan(style, sample_words("Small steps bring lasting change."))
    assert plan.variables["size_px"] == 124
    assert style.data["captions"]["size_px"] == 83
    saved = save_pick(style.unbranded_data, name="Giant again")
    restored = json.loads(saved.path.read_text())
    assert restored["captions"] == style.unbranded_data["captions"]
    restored["captions"]["highlight_style"] = "color"
    restored["name"] = source["name"]
    assert resolve_style_data(restored) == previous
    source["captions"]["highlight_style"] = "color"
    assert source == original


def test_giant_band_phrase_limits_and_plain_word_motion() -> None:
    giant = _style(highlight="giant", top_px=1700, max_words_per_phrase=8)
    plain = _style(highlight="none", top_px=1700, max_words_per_phrase=8)
    words = sample_words("Small steps can bring a lasting change today.")
    before = copy.deepcopy(giant.data)
    plan, ordinary = (caption_plan(style, words) for style in (giant, plain))
    assert plan.height > ordinary.height
    assert plan.y >= 0 and plan.y + plan.height == 1740
    assert plan.warnings
    assert all(len(slot["words"]) <= 3 for slot in json.loads(plan.variables["phrases"]))
    phrase = _phrases(giant)[0]
    dom = _caption_dom(caption_layer_plan([phrase], giant, frames=30), [])
    assert _dom_nodes(dom, "cap-ground")[0]["style"]["background"].endswith(" / 1)")
    spans = _dom_nodes(dom, "w")
    assert all(span["animations"] == [] for span in spans)
    assert spans[1]["style"]["fontStyle"] == "italic"
    assert not _dom_nodes(dom, "cap-mark")[0]["animations"]
    assert dom["children"][0]["children"][0]["animations"][0]["keys"][1]["opacity"] == 0
    assert giant.data == before


def test_giant_face_offsets_use_the_layout_band() -> None:
    style = _style(highlight="giant", size_px=83, top_px=1700)
    before = copy.deepcopy(style.data)
    settings = caption_module.caption_layout_settings(style.data["captions"], style.data["brand"])
    geometry = caption_band_geometry(settings, safe_bottom_px=1740)
    plan = caption_plan(style, sample_words("Words stay clear."))
    assert (plan.y, plan.height) == (geometry.y, geometry.height)
    placements = caption_offsets(json.loads(plan.variables["phrases"]), geometry, None, style,
                                 still_faces=[Box(400, geometry.y + 100, 280, 140)])
    assert all(row["offset_px"] < 0 for row in placements)
    assert all(plan.y + row["offset_px"] >= 640 for row in placements)
    assert style.data == before
