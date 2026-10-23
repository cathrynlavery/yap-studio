"""Solid glyph cores and opt-in phone-scale contrast against their immediate rim."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_captions_render import _TEMPLATES, _caption_dom, _dom_nodes, _rgba_frames, _style

from yap_studio import paths
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.render import fonts
from yap_studio.stages.captions import LayerAssets, LayerPlan, caption_layer_plan, render_layer_plan
from yap_studio.style.loader import _contrast

SOLID_LOOKS = (
    "neon_glow", "neon_accent", "gradient_fill", "glitch_rgb", "particle_burst", "texture",
)
ACCENTS = ("#FF2F92", "#D8FF2F", "#FFFFFF", "#000000", "#777777", "#0000FF", "#FF0000")


def _legibility_plan(look: str, accent: str, text: str = "#FFFFFF") -> LayerPlan:
    # A condensed creator face at 30 px after 1080 -> 360 phone downsampling.
    style = _style(highlight=look, font="Anton SC", size_px=90, highlight_color=accent,
                   text_color=text, shadow_opacity=0, italic_stress=False,
                   entry_ms=0, entry_rise_px=0)
    phrase = build_phrases([
        CaptionWord("w0", "RANDOM", 0.1, 0.4), CaptionWord("w1", "ON", 0.4, 0.9),
    ], style.data["captions"])[0]
    phrase = replace(phrase, start=0.1, end=0.9, stress_id="w0")
    return caption_layer_plan([phrase], style, frames=30)


@pytest.mark.parametrize("look", SOLID_LOOKS)
@pytest.mark.parametrize("accent", ACCENTS)
@pytest.mark.parametrize("text", ["#FFFFFF", "#111111", "#FF2F92"])
def test_computed_glyph_core_contrasts_with_effect_and_never_fades(
    look: str, accent: str, text: str,
) -> None:
    plan = _legibility_plan(look, accent, text)
    # Cover active, future, settled, ramp and backwards seeks. Glitch's burst is
    # deliberately sampled at 0.2, before its 0.26 end; particles peak at 0.22.
    times = [0.2, 0.22, 0.3, 0.43, 0.6, 0.8, 0.2]
    dom = _caption_dom(plan, [], seek_times=times)
    assert dom["seeks"][0] == dom["seeks"][-1]
    initial = [node["style"] for node in _dom_nodes(dom, "cap-ink")]
    assert len(initial) == 2
    for snapshot in dom["seeks"]:
        cores = _dom_nodes(snapshot, "cap-ink")
        assert [core["style"] for core in cores] == initial
        for core in cores:
            css = core["style"]
            assert css["opacity"] == "1"
            assert css["textShadow"] == css["filter"] == css["backgroundImage"] == "none"
            assert _contrast(css["color"], accent) >= 4.5
            assert _contrast(css["color"], css["webkitTextStrokeColor"]) >= 4.5
        for word in _dom_nodes(snapshot, "w"):
            core_color = _dom_nodes(word, "cap-ink")[0]["style"]["color"]
            for effect in _dom_nodes(word, "cap-effect"):
                css = effect["style"]
                if css["color"] != "transparent":
                    assert _contrast(core_color, css["color"]) >= 4.5
                glow_colors = re.findall(r"#[0-9A-Fa-f]{6}", css.get("textShadow", ""))
                if look in {"neon_glow", "neon_accent"}:
                    assert glow_colors == [accent] * 3
                    assert all(_contrast(core_color, glow) >= 4.5 for glow in glow_colors)
            for particle in _dom_nodes(word, "cap-particle"):
                assert _contrast(core_color, particle["style"]["background"]) >= 4.5
    assert all(not node["animations"] for node in _dom_nodes(dom, "cap-ink"))
    if look != "particle_burst":
        assert len(_dom_nodes(dom, "cap-effect")) == (4 if look == "glitch_rgb" else 2)


@pytest.mark.parametrize("look", SOLID_LOOKS)
def test_effect_phrase_entry_never_washes_out_inactive_words(look: str) -> None:
    plan = _legibility_plan(look, "#FF2F92")
    plan = replace(plan, variables={**plan.variables, "entry_ms": 100, "entry_rise_px": 12})
    times = [0, 0.09, 0.1, 0.13, 0.2, 0.6, 0.9]
    dom = _caption_dom(plan, [], seek_times=times)
    lines = [snapshot["children"][0]["children"][0] for snapshot in dom["seeks"]]
    assert [line["style"]["opacity"] for line in lines] == [0, 0, 1, 1, 1, 1, 0]


@pytest.mark.parametrize("look", SOLID_LOOKS)
@pytest.mark.parametrize("text, accent", [
    ("#FF0000", "#FFFFFF"), ("#0000FF", "#D8FF2F"), ("#FFFFFF", "#000000"),
    ("#333333", "#FFFFFF"), ("#FFFFFF", "#D8FF2F"), ("#FF0000", "#D8FF2F"),
])
def test_solid_core_preserves_requested_ink_only_when_it_meets_contrast(
    look: str, text: str, accent: str,
) -> None:
    dom = _caption_dom(_legibility_plan(look, accent, text), [],
                       seek_times=[0.1, 0.2, 0.43, 0.8, 0.1])
    for snapshot in dom["seeks"]:
        for core in _dom_nodes(snapshot, "cap-ink"):
            color = core["style"]["color"]
            assert (color == text) is (_contrast(text, accent) >= 4.5)
            assert _contrast(color, accent) >= 4.5
            assert _contrast(color, core["style"]["webkitTextStrokeColor"]) >= 4.5


def _phone_pixels(pixels: np.ndarray) -> np.ndarray:
    """Area-average to a 360 px wide display, preserving fractional glyph coverage."""
    height, width = pixels.shape[:2]
    pixels = pixels[:height - height % 3]
    return pixels.reshape(height // 3, 3, width // 3, 3, *pixels.shape[2:]).mean(axis=(1, 3))


def _luminance(rgb: np.ndarray) -> np.ndarray:
    channels = rgb / 255
    linear = np.where(channels <= 0.04045, channels / 12.92,
                      ((channels + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def _adjacent(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([
        padded[y:y + mask.shape[0], x:x + mask.shape[1]]
        for y in range(3) for x in range(3)
    ])


def _mask_assets(plan: LayerPlan) -> LayerAssets:
    # Color-code the two glyph silhouettes. Hide effects and all outlines so
    # decoration, bloom or a big opaque tile can never count as legible letters.
    css = """
    .cap-line, .cap-words, .w, .cap-ink { text-shadow: none !important;
      -webkit-text-stroke: 0 !important; filter: none !important; }
    .cap-ink { color: #FF0000 !important; background: none !important;
      opacity: 1 !important; }
    .w:nth-of-type(2) .cap-ink { color: #00FF00 !important; }
    .cap-effect, .cap-particle { visibility: hidden !important; }
    """
    source = (_TEMPLATES / "captions/template.html").read_text().replace(
        "</style>", css + "</style>", 1,
    )
    font = plan.variables["font_name"]
    return LayerAssets(
        html=source, metadata=(_TEMPLATES / "captions/template.json").read_bytes(),
        font_css=fonts.font_face_css([font]), font_files=tuple(
            (face.file, (fonts.FONTS_DIR / face.file).read_bytes())
            for face in fonts.BUNDLED[font].faces
        ),
    )


@pytest.mark.render
@pytest.mark.parametrize("look", SOLID_LOOKS)
@pytest.mark.parametrize("accent", ["#FF2F92", "#0000FF"])
def test_real_phone_glyph_core_and_immediate_surround_contrast(
    tmp_path: Path, look: str, accent: str,
) -> None:
    plan = _legibility_plan(look, accent)
    with paths.operation():
        out = render_layer_plan(plan, tmp_path / "actual", tmp_path / "actual.mov")
        mask = render_layer_plan(plan, tmp_path / "mask", tmp_path / "mask.mov",
                                 assets=_mask_assets(plan))
    actual, reference = _rgba_frames(out.path, plan), _rgba_frames(mask.path, plan)
    dom = _caption_dom(plan, [])
    expected = np.array([
        int(_dom_nodes(dom, "cap-ink")[0]["style"]["color"][i:i + 2], 16)
        for i in (1, 3, 5)
    ])
    metrics = []
    # Both words become active, with checks during and after their effect ramps.
    for time in (0.2, 0.3, 0.43, 0.6, 0.8):
        frame = round(time * plan.variables["fps"])
        rgba = actual[frame].astype(float)
        alpha = rgba[..., 3:4] / 255
        silhouettes = reference[frame].astype(float)
        for word in (0, 1):
            coverage = _phone_pixels(
                silhouettes[..., word] / 255 * silhouettes[..., 3] / 255,
            )
            core = coverage >= 0.97
            surround = (coverage <= 0.03) & _adjacent(coverage > 0.1)
            assert core.sum() >= 12 and surround.sum() >= 12, (look, time, word)
            # Full-opacity, faithful core pixels must survive the resize on each
            # backdrop. Measure only the one-pixel exterior rim, not remote glow.
            assert np.quantile(_phone_pixels(alpha[..., 0])[core], 0.05) >= 0.98
            for ground in (0, 119, 255):
                composite = _phone_pixels(rgba[..., :3] * alpha + ground * (1 - alpha))
                error = np.max(np.abs(composite[core] - expected), axis=1)
                assert np.quantile(error, 0.95) <= 12, (look, accent, time, word, ground)
                core_luma = np.median(_luminance(composite[core]))
                rim_luma = _luminance(composite[surround])
                contrast = (np.maximum(core_luma, rim_luma) + 0.05) / (
                    np.minimum(core_luma, rim_luma) + 0.05)
                # At least half of the immediate exterior samples must meet the
                # same 4.5:1 threshold, including antialiased corners at phone size.
                ratio = float(np.median(contrast))
                metrics.append({"time": time, "word": "RANDOM" if word == 0 else "ON",
                                "active": (time < 0.4) == (word == 0), "backdrop": ground,
                                "contrast": ratio, "core_pixels": int(core.sum()),
                                "surround_pixels": int(surround.sum())})
                assert ratio >= 4.5, (look, accent, metrics[-1])
    (tmp_path / "legibility.json").write_text(json.dumps(metrics, indent=2) + "\n")
