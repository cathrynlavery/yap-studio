"""Portable phrase backgrounds, placement defaults and readable ground text."""

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from yap_studio.style import SCHEMA_PATH, StyleError, load_style, resolve, resolve_style_data
from yap_studio.style.loader import _contrast, unbranded


def test_caption_ground_feature_has_a_release_note() -> None:
    changelog = (Path(__file__).parents[1] / "CHANGELOG.md").read_text()
    bullets = [line.lower() for line in changelog.splitlines() if line.startswith("- ")]
    assert any(all(word in bullet for word in ("caption", "box", "band", "placement"))
               for bullet in bullets)


def test_ground_and_placement_loader_defaults(tmp_path: Path) -> None:
    source = tmp_path / "style.json"
    source.write_text('{"schema_version": 1}')
    captions = load_style(source)["captions"]
    expected = {
        "background_mode": "none", "background_color": "#000000",
        "background_opacity": 0.85, "background_padding_px": 18,
        "background_radius_px": 12, "placement": "avoid_face",
    }
    assert {key: captions[key] for key in expected} == expected
    for name in ("clean_minimal", "electric", "warm_editorial"):
        preset = resolve(f"preset:{name}").data["captions"]
        assert preset["background_mode"] == "none" and preset["placement"] == "avoid_face"
        assert not resolve(f"preset:{name}").caption_placement_explicit


@pytest.mark.parametrize("placement", [None, "fixed", "avoid_face"])
def test_loader_retains_explicit_caption_placement(tmp_path: Path, placement: str | None) -> None:
    data = {"schema_version": 1, "captions": {}}
    if placement is not None:
        data["captions"]["placement"] = placement
    source = tmp_path / "style.json"
    source.write_text(json.dumps(data))
    style = resolve(source)
    assert style.data["captions"]["placement"] == (placement or "avoid_face")
    assert style.caption_placement_explicit is (placement is not None)


@pytest.mark.parametrize(("key", "value"), [
    ("background_mode", "pill"), ("background_color", "#fff"),
    ("background_opacity", -0.01), ("background_opacity", 1.01),
    ("background_opacity", True), ("background_padding_px", -1),
    ("background_padding_px", 81), ("background_padding_px", 1.5),
    ("background_radius_px", -1), ("background_radius_px", 81),
    ("background_radius_px", True), ("placement", "auto"),
])
def test_schema_and_loader_reject_invalid_ground_fields(key: str, value: object) -> None:
    source = {"schema_version": 1, "captions": {key: value}}
    with pytest.raises(StyleError, match=rf"style\.captions\.{key}"):
        resolve_style_data(source)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(source, json.loads(SCHEMA_PATH.read_text()))


@pytest.mark.parametrize("mode", ["none", "box", "band"])
@pytest.mark.parametrize("placement", ["fixed", "avoid_face"])
def test_ground_fields_accept_limits_and_round_trip(mode: str, placement: str) -> None:
    source = {"schema_version": 1, "captions": {
        "background_mode": mode, "background_color": "#AbCdEf", "background_opacity": 1,
        "background_padding_px": 80, "background_radius_px": 0, "placement": placement,
    }}
    data = resolve_style_data(source)
    jsonschema.validate(source, json.loads(SCHEMA_PATH.read_text()))
    assert resolve_style_data(data) == data
    assert all(data["captions"][key] == value for key, value in source["captions"].items())


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize(("color", "text"), [
    ("#FFFFFF", "#111111"), ("#000000", "#FFFFFF"), ("#FFFF00", "#111111"),
    ("#123456", "#FFFFFF"),
])
def test_ground_text_uses_highlight_contrast_choice(mode: str, color: str, text: str) -> None:
    source = {"schema_version": 1, "captions": {
        "background_mode": mode, "background_color": color, "text_color": color,
    }, "brand": {"accent_color": "#112233"}}
    original = copy.deepcopy(source)
    data = resolve_style_data(source)
    assert data["captions"]["text_color"] == text
    assert _contrast(text, color) >= 4.5
    assert data["captions"]["highlight_text_color"] == "#FFFFFF"
    assert unbranded(source)["captions"]["text_color"] == color
    assert source == original


@pytest.mark.parametrize(("mode", "opacity"), [
    ("none", 0.85), ("box", 0), ("band", 0), ("box", 0.01), ("band", 0.49),
])
def test_disabled_or_transparent_ground_preserves_text_color(mode: str, opacity: float) -> None:
    data = resolve_style_data({"schema_version": 1, "captions": {
        "background_mode": mode, "background_opacity": opacity, "text_color": "#123456",
    }})
    assert data["captions"]["text_color"] == "#123456"


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("camera", [True, False])
def test_camera_ink_ignores_the_saved_caption_ground(mode: str, camera: bool) -> None:
    source = {"schema_version": 1, "captions": {
        "camera_3d": camera, "background_mode": mode, "background_color": "#FFFFFF",
        "background_opacity": 1, "text_color": "#FFFFFF", "highlight_style": "highlight",
        "highlight_color": "#112233", "highlight_text_color": "#112233",
    }}
    original = copy.deepcopy(source)
    captions = resolve_style_data(source)["captions"]
    assert captions["text_color"] == ("#FFFFFF" if camera else "#111111")
    assert captions["highlight_text_color"] == "#FFFFFF"
    assert captions["background_mode"] == mode
    assert captions["background_color"] == "#FFFFFF" and captions["background_opacity"] == 1
    assert source == original


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("opacity", [0.01, 0.49, 0.5])
def test_faint_white_ground_preserves_white_ink_until_half_strength(
    mode: str, opacity: float,
) -> None:
    source = {"schema_version": 1, "captions": {
        "background_mode": mode, "background_color": "#FFFFFF",
        "background_opacity": opacity, "text_color": "#FFFFFF",
    }}
    original = copy.deepcopy(source)
    resolved = resolve_style_data(source)
    assert resolved["captions"]["text_color"] == ("#FFFFFF" if opacity < 0.5 else "#111111")
    assert source == original
