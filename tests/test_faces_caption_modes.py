"""Phrase-wide caption placement choices over synthetic moving heads."""

import json
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import test_stage_captions as caption_test_helpers
from test_caption_layout_render import _full_alpha_frames
from test_captions_render import _caption_dom, _phrases, _style, _word_plan
from test_stage_captions import checked_layers as checked_layers
from test_stage_captions import layer_project as layer_project

from yap_studio import paths
from yap_studio.captions import caption_layout_settings
from yap_studio.faces.clearance import face_clearance_px, gap_px, head_box
from yap_studio.faces.detect import find_gaps
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.project.receipts import read_receipt
from yap_studio.render.hyperframes import render_layer as real_render_layer
from yap_studio.render.text_metrics import text_width_em
from yap_studio.stages import captions as caption_stage
from yap_studio.stages.captions import (
    _phrase_box_width,
    caption_band_geometry,
    caption_layer_plan,
    caption_offsets,
    captions,
    render_layer_plan,
)


@pytest.fixture
def fake_layers(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Opt into the shared CLI fake without importing its autouse behavior."""
    return caption_test_helpers.fake_layers.__wrapped__(monkeypatch)


def _faces(frames: tuple) -> FacesData:
    return FacesData("measured", None,
                     VideoIdentity("picture.mp4", "a" * 64, len(frames), Fraction(30), 1080, 1920),
                     {}, {}, frames, tuple(find_gaps(frames)))


def _moving_faces(y: int) -> FacesData:
    frames = tuple((FaceBox(Box(400, y + (40 if frame < 12 else 80), 200, 180),
                           0.95, "detected"),) if frame < 21 else ()
                   for frame in range(60))
    return _faces(frames)


def test_caption_modes_use_real_renderer_without_explicit_fake_fixture() -> None:
    assert caption_stage.hyperframes.render_layer is real_render_layer


def test_opaque_box_clearance_bounds_contain_template_metrics() -> None:
    # Use the failing render's exact style and the template's own ground calculation.
    style = _style(placement="avoid_face", background_mode="box", background_opacity=1,
                   shadow_opacity=0, highlight="none", entry_ms=0)
    phrases = _phrases(style)[:2]
    fixed = caption_layer_plan(phrases, style, frames=60)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    faces = _moving_faces(geometry.y)
    slots = json.loads(fixed.variables["phrases"])
    placements = caption_offsets(slots, geometry, faces, style)
    offsets = [row["offset_px"] for row in placements]
    assert offsets[0] < 0 and offsets[1] == 0
    plan = caption_layer_plan(phrases, style, frames=60, offsets=offsets)
    # The DOM harness supplies a 300 px laid-out line, independently of the stage estimate.
    dom = _caption_dom(plan, [], line_width=300)
    for phrase, cap, offset in zip(slots, dom["children"], offsets, strict=True):
        line = cap["children"][0]
        ground = line["children"][0]["style"]
        width = float(ground["width"].removesuffix("px"))
        height = float(ground["height"].removesuffix("px"))
        assert width == 336  # Logical line width plus 18 px ground padding on each side.
        assert height == pytest.approx(48 * 1.12 + 36)
        assert "webkitTextStroke" not in line["children"][2]["style"]
        bottom = plan.y + plan.height - float(cap["style"]["bottom"].removesuffix("px"))
        top = bottom - geometry.line_height + offset
        top += float(ground["top"].removesuffix("px"))
        left = (1080 - width) / 2
        # Bound rasterized pixels outward, including fractional CSS coordinates.
        drawn = Box(math.floor(left), math.floor(top),
                    math.ceil(left + width) - math.floor(left),
                    math.ceil(top + height) - math.floor(top))
        estimate_width = _phrase_box_width(phrase, style.data["captions"])
        estimate = Box((1080 - estimate_width) / 2, geometry.y + offset,
                       estimate_width, geometry.height)
        assert estimate.x <= drawn.x and estimate.x1 >= drawn.x1
        assert estimate.y <= drawn.y and estimate.y1 >= drawn.y1
        first, end = math.ceil(phrase["start"] * 30), math.ceil(phrase["end"] * 30)
        assert all(gap_px(drawn, head_box(face.box)) >= face_clearance_px(style.data)
                   for frame in faces.frames[first:end] for face in frame)


def test_moving_face_union_moves_only_the_colliding_phrase() -> None:
    style = _style(placement="avoid_face", entry_ms=0, background_mode="box")
    phrases = _phrases(style)[:2]
    fixed = caption_layer_plan(phrases, style, frames=60)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    faces = _moving_faces(geometry.y)
    slots = json.loads(fixed.variables["phrases"])
    placements = caption_offsets(slots, geometry, faces, style)
    offsets = [row["offset_px"] for row in placements]
    assert offsets[0] < 0 and offsets[1] == 0
    element = Box(70, geometry.y + offsets[0], 940, geometry.height)
    assert all(gap_px(element, head_box(face.box)) >= face_clearance_px(style.data)
               for frame in faces.frames[6:21] for face in frame)
    moved = caption_layer_plan(phrases, style, frames=60, offsets=offsets)
    assert [slot["offset_px"] for slot in json.loads(moved.variables["phrases"])] == offsets
    dom = _caption_dom(moved, [], seek_times=[0.3, 0.5, 0.65, 1.3])
    for seek in dom["seeks"]:
        assert seek["children"][0]["style"]["translate"] == f"0 {offsets[0]}px"
        assert "translate" not in seek["children"][1]["style"]


@pytest.mark.parametrize("case", ["missing", "unmeasured", "empty"])
def test_avoid_face_falls_back_to_exact_fixed_plan(case: str) -> None:
    style = _style(placement="avoid_face")
    phrases = _phrases(style)
    plan = caption_layer_plan(phrases, style, frames=90)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    faces = None if case == "missing" else _faces(((),) * 90)
    if case == "unmeasured":
        faces = replace(faces, status="unmeasured", reason="helper unavailable", frames=(), gaps=())
    offsets = caption_offsets(json.loads(plan.variables["phrases"]), geometry, faces, style)
    assert all(row["offset_px"] == 0 for row in offsets)
    assert caption_layer_plan(phrases, style, frames=90,
                              offsets=[row["offset_px"] for row in offsets]) == plan


def test_fixed_mode_ignores_colliding_face_measurements() -> None:
    style = _style(placement="fixed")
    plan = caption_layer_plan(_phrases(style), style, frames=90)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    offsets = caption_offsets(json.loads(plan.variables["phrases"]), geometry,
                              _moving_faces(geometry.y), style)
    assert all(row["offset_px"] == 0 and row["reason"] == "kept: fixed caption placement"
               for row in offsets)


def test_zero_entry_duration_includes_first_visible_frame() -> None:
    style = _style(placement="avoid_face", entry_ms=0)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    faces = _faces(tuple((FaceBox(Box(400, geometry.y + 40, 200, 180), 0.9, "detected"),)
                         if frame == 6 else () for frame in range(30)))
    slots = [{"start": 0.2, "end": 0.7, "words": [{"text": "Hi"}]}]
    assert caption_offsets(slots, geometry, faces, style)[0]["offset_px"] < 0
    style.data["captions"]["entry_ms"] = 100
    assert caption_offsets(slots, geometry, faces, style)[0]["offset_px"] == 0


def test_rise_skips_transparent_first_frame_even_with_zero_entry_ms() -> None:
    style = _style(placement="avoid_face", entry_ms=0)
    style.data["captions"]["entrance"] = "rise"
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    faces = _faces(tuple((FaceBox(Box(400, geometry.y + 40, 200, 180), 0.9, "detected"),)
                         if frame == 6 else () for frame in range(30)))
    slots = [{"start": 0.2, "end": 0.7, "words": [{"text": "Hi"}]}]
    assert caption_offsets(slots, geometry, faces, style)[0]["offset_px"] == 0


def test_band_checks_faces_outside_text_safe_width() -> None:
    style = _style(placement="avoid_face", background_mode="band")
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    faces = _faces(tuple((FaceBox(Box(0, geometry.y + 40, 10, 40), 0.9, "detected"),)
                         for _ in range(30)))
    slots = [{"start": 0.2, "end": 0.7, "words": [{"text": "Hi"}]}]
    offset = caption_offsets(slots, geometry, faces, style)[0]["offset_px"]
    assert offset != 0
    assert gap_px(Box(0, geometry.y + offset, 1080, geometry.height),
                  head_box(faces.frames[0][0].box)) >= face_clearance_px(style.data)
    style.data["captions"]["background_mode"] = "box"
    assert caption_offsets(slots, geometry, faces, style)[0]["offset_px"] == 0


@pytest.mark.parametrize("mode", ["box", "band"])
def test_transparent_ground_uses_disabled_face_geometry(mode: str) -> None:
    style = _style(placement="avoid_face", background_mode=mode, background_opacity=0,
                   background_padding_px=80)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    slots = json.loads(_word_plan(style, "Hi").variables["phrases"])
    face_x = 0 if mode == "band" else 100
    faces = _faces(tuple((FaceBox(Box(face_x, geometry.y + 40, 30, 40), 0.9, "detected"),)
                         for _ in range(30)))
    transparent = caption_offsets(slots, geometry, faces, style)
    style.data["captions"]["background_mode"] = "none"
    assert transparent == caption_offsets(slots, geometry, faces, style)


@pytest.mark.parametrize("mode,weight,rendered_text", [
    ("none", 400, "mMmM"),
    ("moving_block", 400, "mMmM"),
    ("underline", 400, "mMmM"),
    ("color", 400, "mMmM"),
    ("pill", 400, "mMmM"),
    ("gradient", 400, "mMmM"),
    ("editorial", 700, "mMmM"),
    ("slam", 800, "MMMM"),
    ("clip_wipe", 800, "MMMM"),
    ("weight_shift", 800, "mmmm"),
    ("giant", 400, "mMmM"),
])
def test_fitted_ground_bounds_rendered_weight_case_and_emphasis(
    mode: str, weight: int, rendered_text: str,
) -> None:
    style = _style(highlight=mode, font="Inter", size_px=76, background_mode="box",
                   background_padding_px=18, shadow_opacity=0, width_scale=1)
    settings = caption_layout_settings(style.data["captions"], style.data["brand"])
    slot = {"words": [{"text": "mMmM"}], "stress": 0}
    size = settings["size_px"]
    actual = text_width_em(rendered_text, "Inter", weight) * size
    if mode == "slam":
        actual *= 0.8 * 1.3  # Condensed words at the largest entry punch.
    highlight_pad = size * (0.4 if mode == "pill" else 0.08 if mode == "moving_block" else 0)
    padding = settings["background_padding_px"]
    assert _phrase_box_width(slot, settings) >= actual + 2 * (padding + highlight_pad)


def test_fixed_project_does_not_measure_faces(
    layer_project: Path, fake_layers: list, checked_layers: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = layer_project / "style.json"
    data = json.loads(source.read_text())
    data["captions"]["placement"] = "fixed"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(caption_stage, "cut_faces",
                        lambda *a, **k: pytest.fail("fixed placement must not measure faces"))
    captions(layer_project)
    assert all("offset_px" not in slot
               for slot in json.loads(fake_layers[0]["variables"]["phrases"]))
    assert read_receipt(layer_project / "captions")["caption_placement"] == {
        "faces": {"status": "unmeasured", "reason": "fixed caption placement",
                  "frame_sources": []}, "phrases": [], "kept_phrases": 2,
    }


@pytest.mark.render
@pytest.mark.parametrize("highlight", ["none", "giant"])
def test_real_avoid_face_caption_stays_clear_for_each_phrase(
    tmp_path: Path, highlight: str,
) -> None:
    style = _style(placement="avoid_face", background_mode="box", background_opacity=1,
                   shadow_opacity=0, highlight=highlight, entry_ms=0)
    phrases = _phrases(style)[:2]
    fixed = caption_layer_plan(phrases, style, frames=60)
    settings = caption_layout_settings(style.data["captions"], style.data["brand"])
    geometry = caption_band_geometry(settings, safe_bottom_px=1740)
    faces = _moving_faces(geometry.y)
    placements = caption_offsets(json.loads(fixed.variables["phrases"]), geometry, faces, style)
    offsets = [row["offset_px"] for row in placements]
    assert offsets[0] < 0 and offsets[1] == 0
    plan = caption_layer_plan(phrases, style, frames=60, offsets=offsets)
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path, tmp_path / "avoid-face.mov")
    alpha = _full_alpha_frames(rendered.path, plan)
    for frame in range(6, 21):
        rows, columns = np.nonzero(alpha[frame] >= 128)
        assert rows.size
        visible = Box(int(columns.min()), int(rows.min()) + plan.y,
                      int(columns.max() - columns.min() + 1), int(rows.max() - rows.min() + 1))
        assert all(gap_px(visible, head_box(face.box)) >= face_clearance_px(style.data)
                   for face in faces.frames[frame])
    first = np.nonzero(alpha[12] >= 128)[0]
    second = np.nonzero(alpha[39] >= 128)[0]
    assert first.mean() < second.mean()
