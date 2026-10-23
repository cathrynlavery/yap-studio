"""Rise timing, clipping, and compatibility with existing caption looks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from test_captions_render import (
    _ALL_HIGHLIGHTS,
    _caption_dom,
    _dom_nodes,
    _pilot_plan,
    _shrink_summary,
    _style,
    _word_plan,
    _word_rects,
)

from yap_studio.stages.captions import LayerPlan
from yap_studio.style import StyleError, resolve_style_data


def _plan(highlight: str = "none", **tokens: object) -> LayerPlan:
    style = _style(highlight=highlight, entry_ms=0, entry_rise_px=0,
                   shadow_opacity=0, italic_stress=False, **tokens)
    # Exercise the template independently of schema validation.
    style.data["captions"]["entrance"] = "rise"
    return _pilot_plan(style)


def test_caption_entrance_defaults_to_slide() -> None:
    assert resolve_style_data({"schema_version": 1})["captions"]["entrance"] == "slide"


@pytest.mark.parametrize("entrance", ["slide", "rise"])
def test_caption_entrance_accepts_known_values(entrance: str) -> None:
    data = resolve_style_data({"schema_version": 1, "captions": {"entrance": entrance}})
    assert data["captions"]["entrance"] == entrance


@pytest.mark.parametrize("entrance", ["unknown", "", None, 1, True])
def test_caption_entrance_rejects_other_values(entrance: object) -> None:
    with pytest.raises(StyleError) as caught:
        resolve_style_data({"schema_version": 1, "captions": {"entrance": entrance}})
    assert caught.value.parameter == "style.captions.entrance"
    assert "unknown field" not in caught.value.detail
    assert "choose one of" in caught.value.detail or "expected string" in caught.value.detail


def test_caption_plan_passes_entrance() -> None:
    assert _plan().variables["entrance"] == "rise"


def test_slide_plan_and_line_animation_are_unchanged() -> None:
    plan = _pilot_plan(_style(highlight="none", entry_ms=0, entry_rise_px=0,
                             shadow_opacity=0, italic_stress=False))
    variables = dict(plan.variables)
    assert variables.pop("entrance") == "slide"
    # Word ids were added for caption taps.
    assert hashlib.sha256(json.dumps(variables, sort_keys=True).encode()).hexdigest() == (
        "3196a09d320c128caf41d32df91b131149341c2f41b4a5b3d85f2619bd9a332d"
    )
    dom = _caption_dom(plan, _word_rects(plan))
    line = dom["children"][0]["children"][0]
    assert not _dom_nodes(dom, "cap-rise-clip")
    assert line["animations"][0]["keys"] == [
        {"offset": 0, "opacity": 0, "transform": "translateY(0px)"},
        {"offset": 0.1, "opacity": 0, "transform": "translateY(0px)",
         "easing": "cubic-bezier(.2,.7,.2,1)"},
        {"offset": 0.1, "opacity": 1, "transform": "translateY(0)"},
        {"offset": 0.8, "opacity": 1, "transform": "translateY(0)"},
        {"offset": 0.8, "opacity": 0, "transform": "translateY(0)"},
        {"offset": 1, "opacity": 0, "transform": "translateY(0)"},
    ]


def _motion_times(mover: dict, duration: float) -> tuple[float, float]:
    animation = next(a for a in mover["animations"] if "top" in a["keys"][0])
    keys, options = animation["keys"], animation["options"]
    assert options["fill"] == "backwards"
    assert options["duration"] <= duration * 1000
    start = options["delay"] / 1000
    end = start + options["duration"] / 1000
    assert float(keys[0]["top"].removesuffix("px")) > 0
    assert keys[0]["easing"] == "cubic-bezier(.2,.7,.2,1)"
    assert keys[-1]["top"] == "0px"
    return start, end


def test_rise_staggers_words_and_finishes_by_spoken_start_where_possible() -> None:
    plan = _plan()
    phrase = json.loads(plan.variables["phrases"])[0]
    for word, start in zip(phrase["words"], [0.2, 0.6, 0.9, 1.2], strict=True):
        word["start"] = start
    plan = replace(plan, variables={**plan.variables, "phrases": json.dumps([phrase])})
    dom = _caption_dom(plan, _word_rects(plan), seek_times=[0.2, 0.2 + 1 / 30, 0.7, 0.2])
    movers = _dom_nodes(dom, "cap-rise-mover")
    assert len(movers) == 4
    phrase = json.loads(plan.variables["phrases"])[0]
    for i, (mover, word) in enumerate(zip(movers, phrase["words"], strict=True)):
        start, end = _motion_times(mover, 2)
        assert start == pytest.approx(phrase["start"] + i * 2 / 30)
        assert end == pytest.approx(start + 8 / 30)
        assert float(mover["animations"][0]["keys"][0]["top"].removesuffix("px")) == (
            pytest.approx(48 * 1.12 * 0.6)
        )
        if word["start"] >= phrase["start"] + 8 / 30:
            assert end <= word["start"] + 1e-9
        assert mover["children"][0]["className"] == "w"
    first, partial, rested, backward = dom["seeks"]
    assert first == backward
    assert all(float(m["style"]["opacity"]) == 0
               for m in _dom_nodes(first, "cap-rise-mover"))
    partial_words = _dom_nodes(partial, "cap-rise-mover")
    assert 0 < float(partial_words[0]["style"]["opacity"]) < 1
    assert float(partial_words[-1]["style"]["opacity"]) == 0
    assert all(m["style"].get("top", "auto") == "auto"
               for m in _dom_nodes(rested, "cap-rise-mover"))
    line = partial["children"][0]["children"][0]
    assert float(line["style"]["opacity"]) == 1
    assert line["style"]["transform"] == "translateY(0)"
    assert all(k["transform"] in {"translateY(0)", "translateY(0px)"}
               for k in dom["children"][0]["children"][0]["animations"][0]["keys"])


def test_rise_keeps_word_order_and_stagger_during_fast_speech() -> None:
    plan = _plan()
    phrase = json.loads(plan.variables["phrases"])[0]
    for i, word in enumerate(phrase["words"]):
        word["start"] = phrase["start"] + i * 0.12
        word["end"] = word["start"] + 0.1
    plan = replace(plan, variables={**plan.variables, "phrases": json.dumps([phrase])})
    dom = _caption_dom(plan, _word_rects(plan))
    movers = _dom_nodes(dom, "cap-rise-mover")
    assert len(movers) == 4
    for i, mover in enumerate(movers):
        start, end = _motion_times(mover, 2)
        assert start == pytest.approx(phrase["start"] + i * 2 / 30)
        assert end == pytest.approx(start + 8 / 30)


@pytest.mark.parametrize(("frames", "rise", "stagger"), [(10, 8, 2 / 3), (8, 8, 0),
                                                         (4, 4, 0), (0.3, 1, 0)])
def test_rise_short_phrases_shrink_stagger_then_duration(
    frames: float, rise: float, stagger: float,
) -> None:
    plan = _plan()
    phrase = json.loads(plan.variables["phrases"])[0]
    phrase["end"] = phrase["start"] + frames / 30
    plan = replace(plan, variables={**plan.variables, "phrases": json.dumps([phrase])})
    dom = _caption_dom(plan, _word_rects(plan))
    movers = _dom_nodes(dom, "cap-rise-mover")
    assert len(movers) == 4
    for i, mover in enumerate(movers):
        start, end = _motion_times(mover, 2)
        assert start == pytest.approx(phrase["start"] + i * stagger / 30)
        assert end == pytest.approx(start + rise / 30)


@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
def test_rise_keeps_every_highlight_and_restores_seek_times(highlight: str) -> None:
    plan = _plan(highlight)
    dom = _caption_dom(plan, _word_rects(plan))
    clips = _dom_nodes(dom, "cap-rise-clip")
    assert len(clips) == 4
    assert len(_dom_nodes(dom, "w")) == 4
    for clip in clips:
        assert clip["children"][0]["className"] == "cap-rise-mover"
    for mover in _dom_nodes(dom, "cap-rise-mover"):
        assert all(a["currentTime"] == 0 for a in mover["animations"])
    if highlight == "slam":
        phrase = json.loads(plan.variables["phrases"])[0]
        for i, mover in enumerate(_dom_nodes(dom, "cap-rise-mover")):
            start, end = _motion_times(mover, 2)
            assert start == pytest.approx(phrase["words"][i]["start"])
            assert end <= phrase["end"]


def test_rise_layout_intersects_the_word_with_its_clip() -> None:
    plan = _plan()
    rects = _word_rects(plan)
    clips = [dict(r) for r in rects]
    for rect in rects:
        rect["frames"] = {f: {**rect, "top": 120, "bottom": 170}
                          for f in range(7, 10)}
    dom = _caption_dom(plan, rects, clip_rects=clips)
    boxes = _shrink_summary(dom)["layout"][0]["boxes"]
    assert boxes and all(box["y"] == 100 and box["h"] == 50 for box in boxes)


@pytest.mark.parametrize("highlight", ["moving_block", "underline", "pill", "pill_karaoke"])
def test_rise_highlight_ground_fades_without_competing_opacity(highlight: str) -> None:
    plan = _plan(highlight)
    phrase = json.loads(plan.variables["phrases"])[0]
    times = [phrase["start"] + frame / plan.variables["fps"] for frame in range(4)]
    dom = _caption_dom(plan, _word_rects(plan), seek_times=[*times, times[0]])
    target = "cap-words" if highlight == "pill_karaoke" else "cap-mark"
    opacities = [float(_dom_nodes(snapshot, target)[0]["style"].get("opacity", 1))
                 for snapshot in dom["seeks"][:4]]
    assert opacities == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert dom["seeks"][0] == dom["seeks"][-1]
    # A single opacity track cannot be overridden by animation order.
    ground = _dom_nodes(dom, target)[0]
    assert sum(any("opacity" in key for key in animation["keys"])
               for animation in ground["animations"]) == 1


@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
def test_rise_preserves_each_words_highlight_animation(highlight: str) -> None:
    plan = _plan(highlight)
    rise = _caption_dom(plan, _word_rects(plan))
    assert len(_dom_nodes(rise, "cap-rise-mover")) == 4
    slide = _caption_dom(replace(plan, variables={**plan.variables, "entrance": "slide"}),
                         _word_rects(plan))
    for actual, expected in zip(_dom_nodes(rise, "w"), _dom_nodes(slide, "w"), strict=True):
        assert actual["style"] == expected["style"]
        assert [a["keys"] for a in actual["animations"]] == [
            a["keys"] for a in expected["animations"]
        ]


@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
def test_rise_preserves_shrink_rows_and_ground_size(highlight: str) -> None:
    style = _style(highlight=highlight, size_px=180, background_mode="box", width_scale=0.8)
    style.data["captions"]["entrance"] = "rise"
    plan = _word_plan(style, "A remarkably-long-word follows these thoughtful little moments")
    rise = _caption_dom(plan, _word_rects(plan))
    assert _dom_nodes(rise, "cap-rise-mover")
    slide = _caption_dom(replace(plan, variables={**plan.variables, "entrance": "slide"}),
                         _word_rects(plan))
    assert _shrink_summary(rise)["phrases"] == _shrink_summary(slide)["phrases"]
    assert len(_dom_nodes(rise, "cap-row")) == len(_dom_nodes(slide, "cap-row"))
    assert _dom_nodes(rise, "cap-ground")[0]["style"] == _dom_nodes(slide, "cap-ground")[0][
        "style"
    ]
