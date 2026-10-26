"""All seventeen registry looks share schema, fit, timing and readable text contracts."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from release_notes import release_bullets
from test_captions_render import (
    _TEMPLATES,
    _caption_dom,
    _Declarations,
    _dom_nodes,
    _phrases,
    _pilot_plan,
    _rgba_frames,
    _shrink_summary,
    _style,
    _validate_variables,
    _word_plan,
    _word_rects,
)

from yap_studio import paths
from yap_studio.render import fonts
from yap_studio.stages.captions import LayerAssets, caption_layer_plan, render_layer_plan
from yap_studio.style import SCHEMA_PATH, resolve_style_data
from yap_studio.style.loader import _contrast, caption_legibility

PORTS = {
    "blend_difference": "Difference blend",
    "camera_follow": "Camera follow",
    "editorial_emphasis": "Large key word",
    "emoji_pop": "Emoji pop",
    "glitch_rgb": "RGB glitch",
    "gradient_fill": "Gradient fill",
    "highlight": "Word background",
    "matrix_decode": "Matrix decode",
    "neon_accent": "Neon accent",
    "neon_glow": "Neon glow",
    "parallax_layers": "Parallax layers",
    "particle_burst": "Particle burst",
    "pill_karaoke": "Rounded karaoke",
    "texture": "Text texture",
}
REGISTRY = {"slam": "caption-kinetic-slam", "weight_shift": "caption-weight-shift",
            "clip_wipe": "caption-clip-wipe", **{
                look: "caption-" + look.replace("_", "-") for look in PORTS
            }}


PORT_DIGESTS = {
    "blend_difference": "82dddc30094cfe03195485f0bf5bad686855732fdadc5a975e4da33eb6447e26",
    "camera_follow": "4a0da1ba065882e95274468c8f3e36f1094fc598c3c93be1fc63182e9e42953c",
    "editorial_emphasis": "6144571b6c73dbdec282284fe2277d3c958e15cddaf362295ba9cca02e705d1c",
    "emoji_pop": "79e7cad5cf54dec70e7af59c8d0583531017bacca6bc9ee057bfad386904eb9b",
    "glitch_rgb": "65e588d27515cde0e361cddd6034a3dad647a86f50c02847c6cc453619d58648",
    "gradient_fill": "6d08dc0a9f54d25b25590721ec579a3521a0079c70f32b292bd93697384c60ff",
    "highlight": "77f877ddcef85e787453daa6724cec5e85a0678fcb098fe94f9b86ce0ef5f2f4",
    "matrix_decode": "c8f703cbe6e8dd547107f2e6298f27c287a9d08b133ceab128e874ce88d36e78",
    "neon_accent": "1c4af30ab35c6e3f754157cce2c364c75386f37059d47fa52446bca19200339d",
    "neon_glow": "dc9d43d58879e1422837c213aa7f8921ca43b58f4e9c77f8463232585e05034c",
    "parallax_layers": "e1c14eb380fa7d7b8c84798e40530e946ed140547c61d2e6f5111d712cb72d78",
    "particle_burst": "48ec0a000e9c40cff891c2ca0e66ed4cf33c358bbeb2ca76fb3bdc8288bf5936",
    "pill_karaoke": "cdbbc72af7fbac440c0bf5750cf9e3d4ded20f26588b00ac79399342ca60b3e3",
    "texture": "dcd2c9c9ba068507c559171264748c129981253fd61ce933c5d0f461431b0d7f",
}


def test_registry_styles_have_a_changelog_entry() -> None:
    bullets = release_bullets()
    entry = next(line for line in bullets if "HyperFrames caption styles" in line)
    assert str(len(PORTS)) in entry


@pytest.mark.parametrize("look", REGISTRY)
def test_registry_schema_loader_and_template_variables(look: str) -> None:
    source = {"schema_version": 1, "captions": {"highlight_style": look}}
    jsonschema.validate(source, json.loads(SCHEMA_PATH.read_text()))
    assert resolve_style_data(source)["captions"]["highlight_style"] == look
    style = _style(highlight=look, font="Instrument Serif", size_px=104, width_scale=0.6)
    plan = _pilot_plan(style)
    assert plan.variables["font_name"] == "Instrument Serif"
    assert plan.variables["size_px"] == 104
    assert plan.variables["width_scale"] == 0.6
    template = (_TEMPLATES / "captions/template.html").read_text()
    _validate_variables(template, plan.variables)
    parser = _Declarations()
    parser.feed(template)
    options = next(v for v in parser.variables if v["id"] == "highlight_style")["options"]
    labels = {option["value"]: option["label"] for option in options}
    assert labels[look]
    if look in PORTS:
        assert labels[look] == PORTS[look]


@pytest.mark.parametrize("look", REGISTRY)
def test_registry_guards_keep_original_looks_outside_port_dom(look: str) -> None:
    source = (_TEMPLATES / "captions/template.html").read_text()
    stylesheet = source.split("<style>", 1)[1].split("</style>", 1)[0]
    stylesheet = re.sub(r"/\*.*?\*/", "", stylesheet, flags=re.DOTALL)
    guard_selectors = {
        selector.strip()
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", stylesheet)
        if "--legibility_ink" in body or "--legibility_halo" in body
    }
    assert guard_selectors == {
        ".cap-port .w", ".cap-port .cap-ink",
        ".cap-slam .w, .cap-weight-shift .w, .cap-clip-wipe .w",
        ".cap-slam .w.is-key, .cap-clip-wipe .w.is-key",
    }
    plan = _word_plan(_style(highlight=look, shadow_opacity=0), "Read me")
    dom = _caption_dom(plan, [])
    line = dom["children"][0]["children"][0]
    ported = look in PORTS
    assert ("cap-port" in line["className"].split()) is ported
    assert "--legibility_ink" in line["style"]
    assert "--legibility_halo" in line["style"]
    assert "--registry_shadow" not in line["style"]
    assert len(_dom_nodes(dom, "cap-ink")) == (2 if ported else 0)


@pytest.mark.parametrize("look", PORTS)
@pytest.mark.parametrize("background", ["none", "box", "band"])
@pytest.mark.parametrize("position", ["bottom", "middle"])
@pytest.mark.parametrize("scale", [0.6, 1])
def test_registry_fit_ground_and_placement(
    look: str, background: str, position: str, scale: float,
) -> None:
    style = _style(highlight=look, position=position, size_px=180, width_scale=scale,
                   background_mode=background, background_padding_px=80, entry_rise_px=80,
                   shadow_opacity=0, italic_stress=False, placement="avoid_face")
    plan = _word_plan(style, "Write documentation")
    dom = _caption_dom(plan, _word_rects(plan), seek_times=[0, 0.5, 0.95])
    cap = dom["children"][0]
    line = cap["children"][0]
    assert "cap-port" in line["className"]
    assert float(line["style"]["maxWidth"].removesuffix("px")) * scale <= 850
    assert plan.y >= 0 and plan.y + plan.height <= 1740
    if position == "middle":
        assert cap["style"]["top"] == f'{plan.variables["padding_px"]}px'
    else:
        assert float(cap["style"]["bottom"].removesuffix("px")) == pytest.approx(
            plan.variables["bottom_padding_px"],
        )
    ground = _dom_nodes(dom, "cap-ground")
    assert bool(ground) is (background != "none")
    if ground:
        css = ground[0]["style"]
        assert float(css["width"].removesuffix("px")) <= (1080 if background == "band" else 940)
        assert float(css["height"].removesuffix("px")) <= plan.height
    assert [frame["children"][0]["children"][0]["style"]["opacity"]
            for frame in dom["seeks"]] == [0, 1, 0]
    assert len(_shrink_summary(dom)["layout"][0]["boxes"]) == 2
    moved = caption_layer_plan(_phrases(style)[:1], style, frames=30, offsets=[-250])
    moved_dom = _caption_dom(moved, _word_rects(moved))
    assert moved_dom["children"][0]["style"]["translate"] == "0 -250px"
    assert moved.variables["band_height_px"] == plan.height


@pytest.mark.parametrize("look", REGISTRY)
@pytest.mark.parametrize("ground", ["#000000", "#FFFFFF", "#777777", "#FFFF00"])
def test_registry_legibility_helper_preserves_source_and_readable_ink(
    look: str, ground: str,
) -> None:
    captions = _style(highlight=look, background_mode="box", background_color=ground,
                      highlight_color=ground, highlight_text_color=ground).data["captions"]
    before = copy.deepcopy(captions)
    result = caption_legibility(captions)
    assert captions == before
    assert _contrast(result["text_color"], ground) >= 4.5
    if look in {"highlight", "pill_karaoke"}:
        assert _contrast(result["highlight_text_color"], ground) >= 4.5
    plan = _word_plan(_style(highlight=look, text_color=ground, shadow_opacity=0), "Read me")
    dom = _caption_dom(plan, [])
    css = dom["children"][0]["children"][0]["style"]
    if look in PORTS:
        assert _contrast(css["--legibility_ink"], ground) >= 4.5
        assert _contrast(css["--legibility_ink"], css["--legibility_halo"]) >= 18
        assert len(_dom_nodes(dom, "cap-ink")) == 2


@pytest.mark.parametrize("look", ["slam", "weight_shift", "clip_wipe"])
@pytest.mark.parametrize("text, accent", [("#FFFFFF", "#000000"), ("#111111", "#FFFFFF")])
@pytest.mark.parametrize("background", ["none", "box", "band"])
def test_original_registry_words_keep_guard_without_ground_or_shadow(
    look: str, text: str, accent: str, background: str,
) -> None:
    plan = _pilot_plan(_style(highlight=look, text_color=text, highlight_color=accent,
                             background_mode=background, background_opacity=0,
                             shadow_opacity=0))
    dom = _caption_dom(plan, [])
    line = dom["children"][0]["children"][0]
    assert "cap-port" not in line["className"].split()
    assert not _dom_nodes(dom, "cap-ink") and not _dom_nodes(dom, "cap-ground")
    assert _contrast(text, line["style"]["--legibility_ink"]) >= 4.5
    assert _contrast(line["style"]["--legibility_ink"],
                     line["style"]["--legibility_halo"]) >= 18
    if look != "weight_shift":
        key = _dom_nodes(dom, "w")[-1]["style"]
        assert _contrast(accent, key["--legibility_ink"]) >= 4.5
        assert _contrast(key["--legibility_ink"], key["--legibility_halo"]) >= 18
    # The shared stylesheet must paint both the edge and halo on plain registry
    # spans, including the parent of the clipped copies, rather than cap-ink.
    source = (_TEMPLATES / "captions/template.html").read_text()
    stylesheet = re.sub(r"/\*.*?\*/", "", source.split("</style>", 1)[0], flags=re.DOTALL)
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", stylesheet)
    body = next(body for selector, body in rules if f".cap-{look.replace('_', '-')} .w"
                in [part.strip() for part in selector.split(',')] and "--legibility_halo" in body)
    assert "-webkit-text-stroke:" in body and "paint-order: stroke fill" in body
    assert "drop-shadow" in body


@pytest.mark.parametrize("look", PORTS)
def test_registry_word_effects_are_finite_seekable_and_keep_actual_copy(look: str) -> None:
    plan = _pilot_plan(_style(highlight=look))
    dom = _caption_dom(plan, _word_rects(plan), seek_times=[0.25, 1.2, 0.25])
    assert dom["seeks"][0] == dom["seeks"][2]
    words = _dom_nodes(dom, "w")
    assert [node["textContent"] for node in _dom_nodes(dom, "cap-ink")] == [
        "Small", "steps", "bring", "change",
    ]
    assert all("opacity" not in key for word in words for animation in word["animations"]
               for key in animation["keys"])
    def check(node: dict) -> None:
        for animation in node.get("animations", []):
            assert animation["options"] == {
                "duration": 2000, "fill": "both", "easing": "linear",
            }
            offsets = [key["offset"] for key in animation["keys"]]
            assert offsets == sorted(offsets) and offsets[0] == 0 and offsets[-1] == 1
            assert animation["currentTime"] == animation["initialTime"]
        for child in node.get("children", []):
            check(child)
    check(dom)
    special = {"blend_difference": ("cap-tile", 4), "highlight": ("cap-tile", 4),
               "emoji_pop": ("cap-emoji", 1), "matrix_decode": ("cap-decode", 8),
               "parallax_layers": ("cap-layer", 4), "particle_burst": ("cap-particle", 10)}
    if look in special:
        cls, count = special[look]
        assert len(_dom_nodes(dom, cls)) == count
    if look == "camera_follow":
        assert _dom_nodes(dom, "cap-words")[0]["animations"]
    if look == "blend_difference":
        assert len(_dom_nodes(dom, "cap-difference")) == 4
    if look in {"editorial_emphasis", "neon_accent"}:
        assert words[-1]["style"]["fontSize"] == (
            "1.45em" if look == "editorial_emphasis" else "1.2em"
        )


@pytest.mark.parametrize("look", PORTS)
def test_registry_port_provenance_and_license(look: str) -> None:
    provenance = json.loads((_TEMPLATES / "PROVENANCE.json").read_text())
    item = REGISTRY[look]
    entries = [entry for entry in provenance["files"] if entry["item"] == item]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["highlight_style"] == look
    assert entry["upstream_commit"] == "6afcb48395632b85663b9b9486e1b47fd175f8bc"
    assert entry["upstream_path"] == f"registry/components/{item}/{item}.html"
    assert entry["upstream_sha256"] == PORT_DIGESTS[look]
    assert entry["changes"].endswith(".")
    meta = json.loads((_TEMPLATES / "captions/template.json").read_text())
    assert {"item": item, "path": entry["upstream_path"]} in meta["derived_from"]
    html = (_TEMPLATES / "captions/template.html").read_text()
    assert f'{entry["upstream_path"]} at commit {entry["upstream_commit"]}' in html
    assert "Apache License 2.0" in html


@pytest.mark.render
@pytest.mark.parametrize("look", REGISTRY)
@pytest.mark.parametrize("background", ["none", "box", "band"])
def test_real_registry_words_differ_from_background(
    tmp_path: Path, look: str, background: str,
) -> None:
    style = _style(highlight=look, background_mode=background, background_opacity=1,
                   background_color="#777777", width_scale=0.8, shadow_opacity=0,
                   italic_stress=False)
    phrase = replace(_phrases(style)[0], end=0.9, stress_id="w1")
    plan = caption_layer_plan([phrase], style, frames=30)
    with paths.operation():
        out = render_layer_plan(plan, tmp_path, tmp_path / "registry.mov").path
    rgba = _rgba_frames(out, plan)
    # Keep identical layout and decorations, hiding only the actual letters. A
    # nonempty pill, box or particle must not satisfy the readable-text assertion.
    selector = ".cap-ink" if look in PORTS else ".w"
    source = (_TEMPLATES / "captions/template.html").read_text().replace(
        "</style>", f"{selector} {{ opacity: 0 !important; }}\n</style>", 1,
    )
    font = plan.variables["font_name"]
    assets = LayerAssets(
        html=source, metadata=(_TEMPLATES / "captions/template.json").read_bytes(),
        font_css=fonts.font_face_css([font]), font_files=tuple(
            (face.file, (fonts.FONTS_DIR / face.file).read_bytes())
            for face in fonts.BUNDLED[font].faces
        ),
    )
    with paths.operation():
        control = render_layer_plan(plan, tmp_path / "control", tmp_path / "control.mov",
                                    assets=assets).path
    without_ink = _rgba_frames(control, plan)[18].astype(float)
    frame = rgba[18].astype(float)
    alpha = frame[..., 3:4] / 255
    assert alpha.max() == 1
    for ground in (0, 119, 255):
        composite = frame[..., :3] * alpha + ground * (1 - alpha)
        control_alpha = without_ink[..., 3:4] / 255
        baseline = without_ink[..., :3] * control_alpha + ground * (1 - control_alpha)
        different = (np.max(np.abs(composite - baseline), axis=2) > 32) & (alpha[..., 0] > 0.9)
        assert np.count_nonzero(different) > 100, (look, background, ground)
    # Seeks on the full layer still leave the phrase gaps transparent.
    assert not rgba[:6, ..., 3].any() and not rgba[27:, ..., 3].any()
