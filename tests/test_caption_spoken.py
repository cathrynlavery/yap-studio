"""Spoken words keep the full phrase layout while building at speech times."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from test_captions_render import (
    _ALL_HIGHLIGHTS,
    _caption_dom,
    _dom_nodes,
    _pilot_plan,
    _shrink_summary,
    _style,
    _word_rects,
)
from test_faces_caption_modes import _faces

from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.faces.model import Box, FaceBox
from yap_studio.gallery import live
from yap_studio.stages.captions import (
    LayerPlan,
    _phrase_box_width,
    caption_band_geometry,
    caption_layer_plan,
    caption_offsets,
)
from yap_studio.style import resolve_style_data


def _plan(highlight: str = "none", **tokens: object) -> LayerPlan:
    style = _style(highlight=highlight, italic_stress=False, **tokens)
    # Test the template even before the schema accepts the new entrance.
    style.data["captions"]["entrance"] = "spoken"
    return _pilot_plan(style)


def _opacities(node: dict, inherited: float = 1) -> list[float]:
    opacity = inherited * float(node.get("style", {}).get("opacity", 1))
    if node.get("className") == "w":
        return [opacity]
    return [value for child in node.get("children", []) for value in _opacities(child, opacity)]


def test_spoken_schema_and_plan() -> None:
    assert resolve_style_data({"schema_version": 1, "captions": {"entrance": "spoken"}})[
        "captions"]["entrance"] == "spoken"
    assert _plan().variables["entrance"] == "spoken"


def test_spoken_slam_face_clearance_reserves_the_full_phrase() -> None:
    style = _style(highlight="slam", entrance="spoken")
    phrase = json.loads(_pilot_plan(style).variables["phrases"])[0]
    width = _phrase_box_width(phrase, style.data["captions"])
    style.data["captions"]["entrance"] = "slide"
    assert width > _phrase_box_width(phrase, style.data["captions"])


def test_single_word_phrase_still_uses_spoken_outside_one_word_mode() -> None:
    style = _style(entrance="spoken", highlight="none")
    phrases = build_phrases([CaptionWord("w0", "Small", 0.2, 0.5)], style.data["captions"])
    plan = caption_layer_plan(phrases, style, frames=30)
    dom = _caption_dom(plan, _word_rects(plan), seek_times=[0.2])
    assert _opacities(dom["seeks"][0])[0] > 0
    assert _dom_nodes(dom, "cap-spoken-word")


def test_spoken_face_clearance_includes_the_first_visible_frame() -> None:
    style = _style(entrance="spoken", placement="avoid_face")
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    phrases = json.loads(_pilot_plan(style).variables["phrases"])
    faces = _faces(tuple((FaceBox(Box(400, geometry.y, 200, 100), 0.9, "detected"),)
                         if frame == 6 else () for frame in range(60)))
    assert caption_offsets(phrases, geometry, faces, style)[0]["offset_px"] != 0
    style.data["captions"]["entrance"] = "slide"
    assert caption_offsets(phrases, geometry, faces, style)[0]["offset_px"] == 0


@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_each_word_first_visible_frame_is_its_spoken_start(highlight: str, fps: int) -> None:
    plan = _plan(highlight)
    phrase = json.loads(plan.variables["phrases"])[0]
    for word, start in zip(phrase["words"], [0.1, 0.511, 0.8, 1.111], strict=True):
        word["start"] = start
    plan = replace(plan, frames=2 * fps, variables={**plan.variables, "fps": fps,
                                                  "phrases": json.dumps([phrase])})
    times = [frame / fps for frame in range(math.ceil(phrase["end"] * fps) + 1)]
    dom = _caption_dom(plan, _word_rects(plan), seek_times=[*times, times[0]])
    assert dom["seeks"][0] == dom["seeks"][-1]
    samples = [_opacities(snapshot) for snapshot in dom["seeks"][:-1]]
    for index, word in enumerate(phrase["words"]):
        first = math.ceil(max(phrase["start"], word["start"]) * fps - 1e-6)
        assert next(frame for frame, values in enumerate(samples) if values[index] > 0) == first
        assert all(values[index] > 0 for values in samples[first:math.ceil(phrase["end"] * fps)])
    wrappers = _dom_nodes(dom, "cap-spoken-word")
    assert len(wrappers) == 4
    for wrapper in wrappers:
        assert all(set(key) <= {"offset", "opacity"}
                   for animation in wrapper["animations"] for key in animation["keys"])
    assert _opacities(dom["seeks"][-2]) == [0] * 4


@pytest.mark.parametrize("mode", ["none", "box", "band"])
@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
def test_full_layout_and_ground_are_reserved_from_first_frame(mode: str, highlight: str) -> None:
    plan = _plan(highlight, size_px=180, width_scale=0.8, background_mode=mode)
    times = [0.2, 0.5, 0.8, 1.1, 1.5]
    dom = _caption_dom(plan, _word_rects(plan), seek_times=times)
    assert _dom_nodes(dom, "cap-spoken-word")
    snapshots = dom["seeks"]
    for snapshot in snapshots:
        assert len(_dom_nodes(snapshot, "w")) == 4
        assert len(_dom_nodes(snapshot, "cap-row")) == len(_dom_nodes(snapshots[0], "cap-row"))
        if mode != "none":
            assert _dom_nodes(snapshot, "cap-ground")[0]["style"] == _dom_nodes(
                snapshots[0], "cap-ground")[0]["style"]
    slide = replace(plan, variables={**plan.variables, "entrance": "slide"})
    if highlight != "slam":
        baseline = _caption_dom(slide, _word_rects(slide))
        assert _shrink_summary(dom) == _shrink_summary(baseline)
        assert _dom_nodes(dom, "cap-ground") == _dom_nodes(baseline, "cap-ground")


@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
def test_one_word_mode_keeps_existing_entrance(highlight: str) -> None:
    style = _style(highlight=highlight, max_words_per_phrase=1)
    phrases = build_phrases([CaptionWord("w0", "Small", 0.2, 0.5),
                            CaptionWord("w1", "steps", 0.5, 0.8)], style.data["captions"])
    slide = caption_layer_plan(phrases, style, frames=30)
    style.data["captions"]["entrance"] = "spoken"
    assert caption_layer_plan(phrases, style, frames=30) == slide


def test_live_spoken_poster_waits_for_the_fade(tmp_path: Path) -> None:
    plan = _plan()
    layer = live.LiveLayer("captions", "captions", tmp_path, 0, plan.y,
                           plan.width, plan.height, plan.frames, 30, plan.variables)
    poster = live.poster_time(layer)
    assert 0.2 + 2 / 30 <= poster < 0.5
    dom = _caption_dom(plan, _word_rects(plan), seek_times=[poster])
    assert _opacities(dom["seeks"][0]) == [1, 0, 0, 0]
