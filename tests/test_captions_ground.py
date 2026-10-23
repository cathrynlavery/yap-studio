"""Ground bounds, padded fitting and phrase backgrounds under every highlight."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_caption_layout_render import _full_alpha_frames
from test_captions_render import (
    _HIGHLIGHTS,
    _caption_dom,
    _dom_nodes,
    _phrases,
    _style,
    _word_plan,
)

from yap_studio import ffmpeg, paths
from yap_studio.stages.captions import (
    caption_band_geometry,
    caption_layer_plan,
    render_layer_plan,
)


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("position", ["bottom", "middle"])
@pytest.mark.parametrize("padding", [0, 18, 80])
def test_ground_padding_is_reserved_inside_safe_band(
    mode: str, position: str, padding: int,
) -> None:
    style = _style(position=position, background_mode=mode, background_padding_px=padding,
                   top_px=1700, size_px=180, entry_rise_px=80, shadow_blur_px=20)
    plan = caption_layer_plan(_phrases(style), style, frames=90)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    expected = math.ceil(180 * 0.08 + 20 * 3 + 4 + 8) + padding
    assert geometry.padding == plan.variables["padding_px"] == expected
    assert plan.height == math.ceil(2 * 180 * 1.12 + 80 + 2 * expected)
    assert plan.y >= 0 and plan.y + plan.height <= 1740
    assert plan.variables["background_padding_px"] == padding
    if position == "bottom":
        assert plan.warnings and plan.y + plan.height == 1740


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("scale", [0.6, 1])
def test_ground_geometry_and_highlights_share_phrase_visibility(
    highlight: str, mode: str, scale: float,
) -> None:
    style = _style(highlight=highlight, width_scale=scale, background_mode=mode,
                   background_padding_px=18, background_radius_px=80,
                   background_color="#ABCDEF", background_opacity=0.85)
    plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
    dom = _caption_dom(plan, [], seek_times=[0, 0.4, 0.9])
    ground = _dom_nodes(dom, "cap-ground")[0]
    css = ground["style"]
    width = float(css["width"].removesuffix("px"))
    height = float(css["height"].removesuffix("px"))
    assert css["background"] == "rgb(171 205 239 / 0.85)"
    assert width <= (1080 if mode == "band" else 940)
    assert height <= plan.height - plan.variables["entry_rise_px"]
    assert _dom_nodes(dom, "cap-words") and _dom_nodes(dom, "cap-mark")
    line = dom["children"][0]["children"][0]
    assert line["children"][0]["className"] == "cap-ground"
    if mode == "band":
        assert width == 1080 and css["borderRadius"] == "0px"
    else:
        # The harness supplies a 300 px logical line; the ground stays in display pixels.
        text_scale = scale * (0.8 * 1.3 if highlight == "slam" else 1)
        size = float(line["style"].get("fontSize", "48px").removesuffix("px"))
        highlight_pad = size * (0.4 if highlight == "pill" else
                                0.08 if highlight == "moving_block" else 0)
        assert width == pytest.approx(300 * text_scale + 36 + 2 * highlight_pad)
        assert float(css["borderRadius"].removesuffix("px")) == min(80, width / 2, height / 2)
    visible = [frame["children"][0]["children"][0]["style"]["opacity"]
               for frame in dom["seeks"]]
    assert visible == [0, 1, 0]
    assert ground["animations"] == []  # Its parent owns phrase entry and exit.
    assert css["left"] == "50%" and css["transform"] == "translateX(-50%)"


@pytest.mark.parametrize("highlight", ["none", "gradient", "clip_wipe"])
def test_phrase_backing_preserves_unbacked_dom_and_composes_halo(highlight) -> None:
    style = _style(highlight=highlight, background_mode="none", background_color="#FFFFFF",
                   text_color="#FFFFFF", background_opacity=0.1)
    phrases = _phrases(style)
    plain = caption_layer_plan(phrases, style, frames=90)
    plan = caption_layer_plan(phrases, style, frames=90, backing=[None, "shadow", "box"])
    dom = _caption_dom(plan, [])
    original = _caption_dom(plain, [])
    lines = [cap["children"][0] for cap in dom["children"]]
    assert lines[0] == original["children"][0]["children"][0]
    assert [len(_dom_nodes(line, "cap-ground")) for line in lines] == [0, 0, 1]
    words = [_dom_nodes(line, "cap-words")[0] for line in lines]
    assert "filter" not in words[0]["style"] and "filter" not in words[2]["style"]
    halo = words[1]["style"]["filter"]
    assert halo.count("drop-shadow(") >= 2 and "/ 0.9)" in halo
    if highlight in {"gradient", "clip_wipe"}:
        assert halo.count("drop-shadow(") == 3
    assert "filter" not in _dom_nodes(lines[1], "cap-mark")[0]["style"]
    ground = _dom_nodes(lines[2], "cap-ground")[0]["style"]
    assert ground["background"] == "rgb(17 17 17 / 0.85)"


def test_no_ground_preserves_band_and_has_no_extra_dom() -> None:
    style = _style(background_mode="none", background_padding_px=80)
    plain = _style(background_mode="none", background_padding_px=0)
    a = caption_layer_plan(_phrases(style), style, frames=90)
    b = caption_layer_plan(_phrases(plain), plain, frames=90)
    assert (a.y, a.height, a.variables["padding_px"]) == (
        b.y, b.height, b.variables["padding_px"],
    )
    assert not _dom_nodes(_caption_dom(a, []), "cap-ground")


@pytest.mark.parametrize("highlight", ["none", "slam", "weight_shift", "clip_wipe",
                                       "editorial_emphasis"])
@pytest.mark.parametrize("position", ["bottom", "middle"])
def test_backing_keeps_unbacked_fit_and_absolute_anchor(highlight, position) -> None:
    style = _style(highlight=highlight, position=position, top_px=1700, size_px=180,
                   background_mode="none", background_padding_px=80)
    phrases = _phrases(style)
    plain = caption_layer_plan(phrases, style, frames=90)
    backed = caption_layer_plan(phrases, style, frames=90, backing=[None, "shadow", "box"])
    original = _caption_dom(plain, [])["children"][0]
    actual = _caption_dom(backed, [])["children"][0]
    assert actual["children"][0] == original["children"][0]

    def anchor(plan, cap):
        css = cap["style"]
        if css.get("bottom", "auto") != "auto":
            return plan.y + plan.height - float(css["bottom"].removesuffix("px"))
        return plan.y + float(css["top"].removesuffix("px"))

    assert anchor(backed, actual) == pytest.approx(anchor(plain, original))


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("highlight", [*_HIGHLIGHTS, "editorial_emphasis", "pill_karaoke"])
def test_transparent_ground_matches_disabled_layout(mode: str, highlight: str) -> None:
    boxed = _style(highlight=highlight, background_mode=mode, background_opacity=0,
                   background_padding_px=80, width_scale=0.6)
    plain = _style(highlight=highlight, background_mode="none", background_opacity=0,
                   background_padding_px=80, width_scale=0.6)
    a, b = (_word_plan(style, "Seven Caption Words.") for style in (boxed, plain))
    assert (a.y, a.height, a.variables["padding_px"]) == (
        b.y, b.height, b.variables["padding_px"],
    )
    assert _caption_dom(a, []) == _caption_dom(b, [])
    assert a.variables["background_mode"] == mode
    assert a.variables["background_padding_px"] == 80


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
@pytest.mark.parametrize("scale", [0.6, 1])
def test_ground_padding_reduces_text_fit_width(highlight: str, scale: float) -> None:
    style = _style(highlight=highlight, width_scale=scale,
                   background_mode="box", background_padding_px=80)
    plan = _word_plan(style, "Legible captions")
    dom = _caption_dom(plan, [])
    line = dom["children"][0]["children"][0]
    width = float(line["style"]["maxWidth"].removesuffix("px"))
    text_scale = scale * (0.8 if highlight == "slam" else 1)
    assert width * text_scale + 160 <= 940 + 0.001


def test_long_word_shrinks_to_padded_width() -> None:
    style = _style(highlight="none", background_mode="box", background_padding_px=80)
    plan = _word_plan(style, "Supercalifragilisticexpialidocious")
    dom = _caption_dom(plan, [])
    line = dom["children"][0]["children"][0]
    assert float(line["style"]["fontSize"].removesuffix("px")) < 48
    assert float(line["style"]["maxWidth"].removesuffix("px")) == 780
    report = json.loads(dom["warnings"][0].removeprefix("yap-studio caption-shrink "))
    assert report["phrases"][0]["fits"] is True


def test_shadow_backing_uses_fitted_size() -> None:
    plan = _word_plan(_style(highlight="none"), "Supercalifragilisticexpialidocious")
    slots = json.loads(plan.variables["phrases"])
    slots[0]["backing"] = "shadow"
    plan.variables["phrases"] = json.dumps(slots)
    dom = _caption_dom(plan, [])
    line = dom["children"][0]["children"][0]
    size = float(line["style"]["fontSize"].removesuffix("px"))
    assert size < plan.variables["size_px"]
    halo = _dom_nodes(line, "cap-words")[0]["style"]["filter"]
    assert f"0 0 {0.04 * size}px" in halo and f"0 0 {0.12 * size}px" in halo


def test_padding_changes_line_breaks_before_render() -> None:
    plain = _style(highlight="clip_wipe")
    boxed = _style(highlight="clip_wipe", background_mode="box", background_padding_px=80)
    text = "Seven Caption Words."
    assert len(_dom_nodes(_caption_dom(_word_plan(plain, text), []), "cap-row")) == 1
    assert len(_dom_nodes(_caption_dom(_word_plan(boxed, text), []), "cap-row")) == 2


@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
def test_wide_ground_clamps_to_horizontal_safe_area(highlight: str) -> None:
    style = _style(highlight=highlight, background_mode="box", background_padding_px=80,
                   background_radius_px=80, width_scale=0.6, size_px=180)
    plan = _word_plan(style, "A broad caption")
    dom = _caption_dom(plan, [], line_width=2000)
    ground = _dom_nodes(dom, "cap-ground")[0]["style"]
    width = float(ground["width"].removesuffix("px"))
    assert width <= 940
    assert (1080 - width) / 2 >= 70 and (1080 + width) / 2 <= 1010


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("highlight", _HIGHLIGHTS)
@pytest.mark.parametrize("color,outline", [("#000000", "#111111"), ("#FFFFFF", "#111111")])
def test_faint_ground_has_solid_contrasting_outline_without_shadow(
    mode: str, highlight: str, color: str, outline: str,
) -> None:
    style = _style(highlight=highlight, background_mode=mode, background_color=color,
                   background_opacity=0.01, shadow_opacity=0)
    plan = _word_plan(style, "Visible words")
    dom = _caption_dom(plan, [])
    words = _dom_nodes(dom, "cap-words")[0]["style"]
    assert words["webkitTextStroke"] == f"2px {outline}"
    assert words["paintOrder"] == "stroke fill"
    assert _dom_nodes(dom, "cap-ground")[0]["style"]["background"].endswith(" / 0.01)")


@pytest.mark.parametrize("mode,opacity", [("none", 0.01), ("box", 1), ("band", 1)])
def test_opaque_or_disabled_ground_does_not_add_outline(mode: str, opacity: float) -> None:
    plan = _word_plan(_style(background_mode=mode, background_opacity=opacity), "Words")
    words = _dom_nodes(_caption_dom(plan, []), "cap-words")[0]["style"]
    assert "webkitTextStroke" not in words


@pytest.mark.render
@pytest.mark.parametrize("color", ["#000000", "#FFFFFF"])
def test_real_faint_ground_text_stays_visible_on_light_and_dark_video(
    tmp_path: Path, color: str,
) -> None:
    style = _style(highlight="none", background_mode="box", background_color=color,
                   background_opacity=0.01, shadow_opacity=0, italic_stress=False)
    plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path, tmp_path / "faint.mov")
    raw = ffmpeg.run([
        "ffmpeg", "-nostdin", "-i", str(rendered.path), "-vf", "select=eq(n\\,12)",
        "-frames:v", "1", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
    ], timeout_s=120, binary=True).stdout
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(plan.height, plan.width, 4)
    alpha = pixels[:, :, 3:4].astype(float) / 255
    # Composite the same rendered layer over synthetic black and white footage.
    for video in (0, 255):
        composited = pixels[:, :, :3] * alpha + video * (1 - alpha)
        assert np.count_nonzero(np.max(np.abs(composited - video), axis=2) > 100) > 100


@pytest.mark.render
@pytest.mark.parametrize("highlight", ["none", "moving_block", "pill", "slam", "clip_wipe"])
@pytest.mark.parametrize("mode", ["box", "band"])
def test_real_ground_caption_has_stable_padded_alpha(
    tmp_path: Path, highlight: str, mode: str,
) -> None:
    style = _style(highlight=highlight, background_mode=mode, background_opacity=1,
                   background_padding_px=18, background_radius_px=12,
                   shadow_opacity=0, italic_stress=False, width_scale=0.8)
    plan = caption_layer_plan(_phrases(style)[:1], style, frames=30)
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path, tmp_path / "ground.mov")
    alpha = _full_alpha_frames(rendered.path, plan)
    assert not alpha[:6].any() and not alpha[21:].any()
    masks = alpha[10:20] >= 128
    assert masks.all(axis=0).any()
    for mask in masks[1:]:
        np.testing.assert_array_equal(mask, masks[0])
    rows, columns = np.nonzero(masks[0])
    assert rows.min() > 0 and rows.max() < plan.height - 1
    assert plan.y + rows.max() < 1740
    if mode == "box":
        assert columns.min() >= 70 and columns.max() < 1010
    else:
        assert columns.min() == 0 and columns.max() == 1079
