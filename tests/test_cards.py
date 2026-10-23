"""Vendored card contracts, slot fitting, and local render preparation without a browser."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import zlib
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from yap_studio import cards, ffmpeg, tools
from yap_studio.errors import (
    DependencyError,
    InputError,
    OutputPathError,
    ProcessingError,
    ValidationError,
)
from yap_studio.render import fonts, hyperframes, text_metrics
from yap_studio.render.text_metrics import text_width_em, widest_digit_em
from yap_studio.style import ResolvedStyle, resolve, resolve_style_data

# The first sixteen cards use the shared panel geometry and scaffold.
# Bare text cards have their own geometry tests.
PANEL_TEMPLATES = tuple(kind for kind in cards.CARD_TEMPLATES
                        if kind not in {"big_number", "choices"})
PRESETS = ("warm_editorial", "electric", "clean_minimal")


@pytest.mark.parametrize(("drops", "message"), [
    ([], "1..4 items"),
    ([{"text": "x", "at": 0.3}] * 5, "1..4 items"),
    ([{"text": "x", "at": 0.3, "extra": 1}], "text and at"),
    ([{"text": "x"}], "text and at"),
    ([{"text": "", "at": 0.3}], "1..20 characters"),
    ([{"text": " ", "at": 0.3}], "1..20 characters"),
    ([{"text": "x" * 21, "at": 0.3}], "shorten"),
    ([{"text": 1, "at": 0.3}], "string"),
    ([{"text": "x", "at": True}], "finite number"),
    ([{"text": "x", "at": math.nan}], "finite number"),
    ([{"text": "x", "at": -0.1}], "0"),
    ([{"text": "x", "at": 1}, {"text": "y", "at": 0.8}], "non-decreasing"),
    ([{"text": "x", "at": 2.3}], "0.5 s before"),
])
def test_gravity_drop_invalid_slots(drops: list, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        cards.card_layer_plan("gravity_drop", {"drops": drops}, _style(), frames=75)


def test_gravity_drop_fonts_layout_and_caption_tokens() -> None:
    style = _style(font="Inter")
    style.data["captions"].update(font="Gelasio", size_px=64, text_color="#AABBCC",
                                  shadow_opacity=0.4, shadow_blur_px=5,
                                  shadow_x_px=2, shadow_y_px=3)
    style.data["hook"]["font"] = "Instrument Serif"
    drops = [{"text": text, "at": at}
             for text, at in zip(("Tokens", "are", "spoken", "words"),
                                 (0, 0.8, 1.3, 1.3), strict=True)]
    plan = cards.card_layer_plan("gravity_drop", {"drops": drops}, style, frames=75)
    planned = json.loads(plan.variables["drops"])
    assert plan.fonts == ("Inter", "Gelasio", "Instrument Serif")
    assert [drop["font_family"] for drop in planned] == [
        fonts.font_stack(name) for name in ("Inter", "Gelasio", "Instrument Serif", "Inter")
    ]
    assert [drop["font_style"] for drop in planned] == ["normal", "italic"] * 2
    assert planned[0]["at"] == 0
    assert len({drop["y"] for drop in planned}) <= 2
    assert cards.MIN_TEXT_PX <= plan.variables["text_px"] <= 64 * 1.35
    assert plan.variables["ink_color"] == "#AABBCC"
    assert plan.variables["text_shadow"] == "2px 3px 5px rgb(0 0 0 / 0.4)"
    for drop in planned:
        assert 0 <= drop["x"] < drop["x"] + drop["width"] <= plan.width
        assert 0 <= drop["y"] < drop["y"] + drop["height"] <= plan.height
    assert cards.card_layer_plan("gravity_drop", {"drops": drops}, style, frames=75) == plan
    style.data["captions"]["font"] = style.data["hook"]["font"] = "Inter"
    assert cards.gravity_drop_fonts(style) == ("Inter",)


@pytest.mark.parametrize("at", [0, 1 / 30, 1 / 15, 0.1, 0.5])
def test_gravity_drop_preserves_landing_and_shortens_fall(at: float) -> None:
    plan = cards.card_layer_plan("gravity_drop", {"drops": [{"text": "Weight", "at": at}]},
                                 _style(), frames=75)
    drop = json.loads(plan.variables["drops"])[0]
    assert drop["at"] == at
    assert drop["fall_s"] == min(0.26, at)


def test_gravity_drop_measured_fit_and_refusal() -> None:
    drops = [{"text": "Wide words", "at": 0.3}] * 4
    style = _style(card_width_px=440)
    style.data["captions"]["size_px"] = 100
    plan = cards.card_layer_plan("gravity_drop", {"drops": drops}, style, frames=75)
    assert cards.MIN_TEXT_PX <= plan.variables["text_px"] < 135
    assert len({drop["y"] for drop in json.loads(plan.variables["drops"])}) == 2
    with pytest.raises(ValidationError, match="shorten drop words"):
        cards.card_layer_plan("gravity_drop", {"drops": [{"text": "W" * 20, "at": 0.3}]},
                              _style(card_width_px=100), frames=75)
    with pytest.raises(ValidationError, match="at least 1.2 s"):
        cards.card_layer_plan("gravity_drop", {"drops": [{"text": "x", "at": 0}]},
                              style, frames=35)

UPSTREAM = {
    "terminal": [("components", "code-terminal-run"), ("components", "typed-prompt")],
    "stat": [
        ("components", "count-up"), ("components", "number-wheel"),
        ("components", "slot-machine-roll"),
    ],
    "checklist": [("blocks", "mk-specs-list"), ("components", "success-check")],
    "text_tag": [
        ("blocks", "lt-clean-bar"),
        ("blocks", "lt-dark-card"),
        ("blocks", "lt-color-block"),
        ("blocks", "lt-soft-pill"),
        ("blocks", "lt-accent-underline"),
        ("blocks", "lt-side-rule"),
        ("components", "titlecard-calm"),
    ],
    "image_card": [
        ("components", "browser-device-stage"), ("components", "paper-collage-reveal"),
    ],
    "transfer": [("components", "offset-path-traveler")],
    "image_compare": [("components", "before-after-wipe")],
    "steps": [("components", "state-chip-rail")],
    "flow": [("blocks", "hw-pipeline"), ("components", "hw-arrow")],
    "chart": [("components", "animated-bar-chart")],
    "quote": [("components", "testimonial-card")],
    "progress_stat": [("blocks", "mk-progress-stat")],
    "specs_list": [("blocks", "mk-specs-list")],
    "callout_circle": [("components", "hw-callout-circle")],
    "name_tag": [("blocks", "lt-clean-bar")],
    "success_check": [("components", "success-check")],
    "edit_timeline": [],
    "storyboard": [],
    "jiggle_delete": [],
    "file_drop": [],
    "video_card": [("components", "browser-device-stage")],
    "prompt_bar": [("components", "typed-prompt"), ("components", "success-check")],
    "comment_cta": [("components", "typed-prompt")],
    "comment_drop": [("components", "chat-message"), ("components", "spring-pop")],
    "icon_strike": [("components", "strikethrough-replace"), ("components", "spring-pop")],
    "playing_grid": [("blocks", "mk-placeholder-grid")],
    "reel_feed": [("components", "browser-device-stage")],
    "gravity_drop": [],
    "choices": [],
}
TEXT_SIZES = {
    "terminal": ("text_px", "bar_px"),
    "stat": ("value_px", "kicker_px"),
    "checklist": ("text_px",),
    "text_tag": ("text_px", "subtext_px"),
    "image_card": ("bar_px",),
    "transfer": ("label_px", "chip_px"),
    "image_compare": ("chip_px",),
    "steps": ("text_px",),
    "flow": ("text_px",),
    "chart": ("title_px", "label_px", "value_px"),
    "quote": ("quote_px", "name_px", "role_px"),
    "progress_stat": ("label_px", "value_px"),
    "specs_list": ("title_px", "row_px"),
    "callout_circle": ("label_px",),
    "name_tag": ("name_px", "role_px"),
    "success_check": ("label_px",),
    "edit_timeline": ("text_px", "timecode_px", "title_px"),
    "storyboard": ("number_px",),
    "jiggle_delete": ("label_px",),
    "file_drop": ("name_px", "duration_px", "badge_px"),
    "video_card": ("label_px",),
    "prompt_bar": ("text_px",),
    "comment_cta": ("text_px", "label_px"),
    "comment_drop": ("text_px",),
    "icon_strike": ("label_px",),
    "playing_grid": ("chip_px",),
    "reel_feed": ("handle_px", "caption_px", "views_px"),
    "gravity_drop": ("text_px",),
    "choices": ("text_px",),
}


def test_choices_slot_contract() -> None:
    metadata = cards.load_template("choices")
    assert metadata["derived_from"] == []
    rules = metadata["slots"]
    assert rules["options"] == {"type": "string_list", "required": True,
                                "min_items": 2, "max_items": 4,
                                "min_length": 1, "max_length": 24}
    assert rules["cues"] == {"type": "number_list", "required": False,
                             "min_items": 0, "max_items": 4, "default": []}
    assert rules["letters"] == {"type": "boolean", "required": False, "default": True}


def test_choices_prepared_contract(tmp_path: Path) -> None:
    metadata = cards.load_template("choices")
    source = (cards.template_dir("choices") / "template.html").read_text()
    assert _HTML(source).variables == metadata["variables"]
    assert "/* BUNDLED_FONT_FACES */" in source
    assert "gsap" not in source.casefold() and "__timelines" not in source
    assert [attrs for tag, attrs in _HTML(source).elements if tag == "script"] == [{}]
    assert "data-no-timeline" in source and ".animate(" in source
    assert "document.fonts.ready.then(animateOptions)" in source
    assert ".panel" not in source
    for token in ("Math.random", "Date", "setTimeout", "requestAnimationFrame", "innerHTML"):
        assert token not in source
    plan = cards.card_layer_plan("choices", {"options": ["One", "Two"]}, _style())
    assert set(plan.variables) == {v["id"] for v in metadata["variables"]}
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert hyperframes.preflight(composition, plan.frames) == hyperframes.layer_duration_s(
        plan.frames)


@pytest.mark.parametrize(("accent", "tint"), [
    ("#00FF00", "#A6FFA6"), ("#E9C46A", "#F7EACB"), ("#102030", "#ABB1B7"),
])
def test_glow_colors(accent: str, tint: str) -> None:
    assert cards.glow_colors(accent.lower()) == (tint, accent)


@pytest.mark.parametrize(("slots", "frames", "field", "fix"), [
    ({"options": ["One"]}, 75, "slots.options", "2..4"),
    ({"options": ["One"] * 5}, 75, "slots.options", "2..4"),
    ({"options": ["x" * 25, "Two"]}, 75, "slots.options", "shorten"),
    ({"cues": [0, 0.29]}, 75, "slots.cues", "space"),
    ({"cues": [0.5, 0.1]}, 75, "slots.cues", "space"),
    ({"cues": [-0.1, 0.5]}, 75, "slots.cues", "0"),
    ({"cues": [0, 1.3]}, 75, "slots.cues", "earlier"),
    ({"cues": [0, 0.4, 0.8]}, 75, "slots.cues", "one per option"),
    ({}, 59, "frames", "raise"),
])
def test_choices_refusals(slots: dict, frames: int, field: str, fix: str) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("choices", {"options": ["One", "Two"], **slots},
                              _style(), frames=frames)
    assert caught.value.parameter.startswith(field)
    assert fix in caught.value.detail


@pytest.mark.parametrize("letters", [[], {}, 0, 1, "no"])
def test_choices_letters_refuses_non_booleans(letters: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("choices", {"options": ["One", "Two"], "letters": letters},
                              _style())
    assert caught.value.parameter == "slots.letters"
    assert caught.value.detail == "use true or false"


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("duration", [2, 2.5])
def test_choices_default_cues(count: int, duration: float) -> None:
    style = _style()
    slots = {"options": ["One"] * count}
    if count >= 3 and duration == 2:
        with pytest.raises(ValidationError, match="end the card later or give fewer options"):
            cards.card_layer_plan("choices", slots, style, frames=round(duration * 30))
        return
    plan = cards.card_layer_plan("choices", slots, style, frames=round(duration * 30))
    latest = hyperframes.layer_duration_s(round(duration * 30)) - (5 / 30 + 0.5 + 0.2 + 0.3 + 0.1)
    cues = json.loads(plan.variables["cues"])
    assert cues == pytest.approx([i * latest / (count - 1) for i in range(count)], abs=1e-6)
    assert all(b - a >= 11 / 30 - 1e-6 for a, b in zip(cues, cues[1:], strict=False))
    assert plan == cards.card_layer_plan("choices", slots, style, frames=round(duration * 30))


def test_choices_refuses_the_reviewers_overlapping_cues() -> None:
    with pytest.raises(ValidationError, match="slots.cues.*space the cues"):
        cards.card_layer_plan("choices", {"options": ["Choose time wisely", "Spend time together"],
                                          "cues": [0.1, 0.4]}, _style(), frames=90)


@pytest.mark.parametrize("fps,gap_frames", [(24, 9), (25, 10), (30, 11), (60, 22)])
def test_choices_minimum_gap_rounds_up_to_whole_frames(fps: int, gap_frames: int) -> None:
    slots = {"options": ["One", "Two"], "cues": [0.1, 0.1 + gap_frames / fps]}
    assert cards.card_layer_plan("choices", slots, _style(), frames=3 * fps, fps=fps)
    slots["cues"][1] -= 1 / fps
    with pytest.raises(ValidationError, match="slots.cues.*space the cues"):
        cards.card_layer_plan("choices", slots, _style(), frames=3 * fps, fps=fps)


def test_choices_partial_cues_keep_spacing() -> None:
    plan = cards.card_layer_plan("choices", {"options": ["One", "Two", "Three"],
                                            "cues": [0.4]}, _style(), frames=90)
    cues = json.loads(plan.variables["cues"])
    assert cues == pytest.approx([0.4, 0.866167, 1.732333], abs=1e-6)
    with pytest.raises(ValidationError, match="end the card later or give fewer options"):
        cards.card_layer_plan("choices", {"options": ["One", "Two", "Three"],
                                          "cues": [1.2]}, _style(), frames=90)


@pytest.mark.parametrize("letters", [True, False])
def test_choices_fits_landing_width(letters: bool) -> None:
    style = _style(card_width_px=600)
    slots = {"options": ["Ten minutes", "One hour"], "letters": letters}
    plan = cards.card_layer_plan("choices", slots, style)
    assert plan.variables["letters"] is letters
    lines = json.loads(plan.variables["lines"])
    for i, option in enumerate(lines):
        assert " ".join(option) == (f"{chr(65 + i)}) " if letters else "") + slots["options"][i]
        assert 1 <= len(option) <= 2
    size = plan.variables["text_px"]
    landed = max(text_width_em(line, style.data["graphics"]["font"], 600) * size * 1.6
                 for option in lines for line in option)
    assert landed <= 864
    assert 0 <= plan.width - landed - 2 * plan.variables["glow_pad_px"] < 4
    assert size == 64 and plan.variables["scale"] == 1.6
    assert plan.x == (1080 - plan.width) // 2
    assert 280 <= plan.y + plan.variables["glow_pad_px"] <= 281
    assert plan.variables["glow_color"] == style.data["graphics"]["accent_color"].upper()


def test_choices_balances_wrapped_lines_with_the_prefix_on_line_one() -> None:
    style = _style()
    text = "Choose time wisely"
    plan = cards.card_layer_plan("choices", {"options": [text, "Two"]}, style)
    words = text.split()
    size = plan.variables["text_px"]
    def measure(text: str) -> float:
        return text_width_em(text, style.data["graphics"]["font"], 600) * size * 1.6
    splits = [["A) " + " ".join(words[:i]), " ".join(words[i:])]
              for i in range(1, len(words))]
    expected = min(splits, key=lambda pair: abs(measure(pair[0]) - measure(pair[1])))
    lines = json.loads(plan.variables["lines"])[0]
    assert measure("A) " + text) > 648
    assert lines == expected and lines[0].startswith("A) ")
    assert all(measure(line) <= 864 for line in lines)


def test_choices_prefers_smaller_single_lines_before_wrapping() -> None:
    style = _style()
    slots = {"options": ["Small", "In between"]}
    plan = cards.card_layer_plan("choices", slots, style)
    labels = ["A) Small", "B) In between"]
    expected = max(size for size in range(56, 65) if all(
        text_width_em(label, style.data["graphics"]["font"], 600) * size * 1.6 <= 648
        for label in labels))
    assert plan.variables["text_px"] == expected < 64
    assert json.loads(plan.variables["lines"]) == [[label] for label in labels]


def test_choices_short_headroom_budget() -> None:
    plan = cards.card_layer_plan("choices", {"options": ["Small", "In between"]}, _style(),
                                 max_height_px=274)
    assert plan.variables["text_px"] == 56
    assert json.loads(plan.variables["lines"]) == [["A) Small"], ["B) In between"]]
    assert plan.variables["glow_pad_px"] == 45
    assert plan.height == 274
    assert plan.y + plan.height - 280 <= 230


@pytest.mark.parametrize("options", [
    ["One", "Two"], ["Choose time wisely", "One"],
    ["One", "Choose time wisely", "Three"],
    ["Choose time wisely", "Spend time together"],
])
def test_choices_tight_lines_and_minimal_row_clearance(options: list[str]) -> None:
    plan = cards.card_layer_plan("choices", {"options": options}, _style(), frames=90)
    v = plan.variables
    assert v["line_height"] == 1.1
    lines = json.loads(v["lines"])
    heights = json.loads(v["row_heights"])
    gaps = json.loads(v["row_gaps"])
    assert heights == pytest.approx([len(row) * 1.1 * v["text_px"] for row in lines])
    for first, second, gap in zip(heights, heights[1:], gaps, strict=False):
        clearance = gap - (v["scale"] - 1) / 2 * max(first, second)
        assert 4 <= clearance < 5


def test_choices_long_landed_lines_obey_the_eighty_percent_limit() -> None:
    style = _style()
    plan = cards.card_layer_plan("choices", {"options": ["WWWWWWW WWWWWWWWW", "Two"]}, style)
    lines = json.loads(plan.variables["lines"])[0]
    size = plan.variables["text_px"]
    widest = max(text_width_em(line, style.data["graphics"]["font"], 600) for line in lines)
    assert len(lines) == 2 and 56 <= size <= 64
    assert 648 < widest * size * 1.6 <= 864
    if size < 64:
        assert widest * (size + 1) * 1.6 > 864


def test_choices_unwrappable_option_is_named_in_the_fix() -> None:
    with pytest.raises(ValidationError, match="slots.options.*shorten option B to fit two lines"):
        cards.card_layer_plan("choices", {"options": ["One", "W" * 24]}, _style())


def test_choices_four_short_options_ignore_the_old_graphics_band() -> None:
    style = _style()
    style.data["internal"]["key_text_bottom_px"] = 400
    slots = {"options": ["One", "Two", "Three", "Four"]}
    plan = cards.card_layer_plan("choices", slots, style, frames=90)
    assert plan.variables["text_px"] == 64 and plan.height > 360
    assert plan == cards.card_layer_plan("choices", slots, style, frames=90)


@pytest.mark.parametrize("height,size", [(314, 64), (292, 60), (274, 56)])
def test_choices_height_limit_keeps_the_largest_readable_size(height: int, size: int) -> None:
    plan = cards.card_layer_plan("choices", {"options": ["One", "Two"]}, _style(),
                                 max_height_px=height)
    assert plan.variables["text_px"] == size and plan.height <= height
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("choices", {"options": ["One", "Two"]}, _style(),
                              max_height_px=272)
    assert caught.value.detail == (
        "use fewer or shorter options, or another moment with more room above the head, "
        "to fit at 56 px")


def test_choices_timeline(tmp_path: Path) -> None:
    cues = [0.1, 0.5, 1.4]
    plan = cards.card_layer_plan("choices", {"options": ["One", "Two", "Three"],
                                            "cues": cues}, _style(), frames=90)
    events = _timeline_events("choices", 3, tmp_path, **plan.variables)
    assert events == _timeline_events("choices", 3, tmp_path, **plan.variables)
    animations = [e for e in events if e["kind"] == "animate"]
    assert len(animations) == 3 * len(cues) + 1
    assert all(e["paused"] and e["currentTime"] == 0 for e in animations)
    assert all(e["options"]["fill"] == "both" and e["options"]["easing"] == "linear"
               for e in animations)
    land = [e for e in animations if e["target"] == "option"
            and "opacity" in e["keyframes"][0]]
    assert [e["options"]["delay"] for e in land] == pytest.approx([cue * 1000 for cue in cues])
    assert all(e["options"]["duration"] == pytest.approx(2 / 30 * 1000) for e in land)
    assert all(e["keyframes"] == [{"opacity": 0}, {"opacity": 1}] for e in land)
    settles = [0.3, 0.5 + 5 / 30 + 0.5, 1.4 + 5 / 30 + 0.5]
    scale = [e for e in animations if "transform" in e["keyframes"][0]]
    assert [e["options"]["delay"] for e in scale] == pytest.approx([s * 1000 for s in settles])
    for event in scale:
        assert event["options"]["duration"] == 200
        keys = event["keyframes"]
        assert [key["offset"] for key in keys] == [0, 0.5, 1]
        assert [float(re.search(r"scale\(([^)]+)\)", key["transform"])[1])
                for key in keys] == pytest.approx([1.6, 1.3, 1])
        assert [float(n) for n in re.findall(r"[0-9.]+", keys[0]["easing"])] == (
            pytest.approx([1 / 3, 0, 2 / 3, 0]))
        assert [float(n) for n in re.findall(r"[0-9.]+", keys[1]["easing"])] == (
            pytest.approx([1 / 3, 1, 2 / 3, 1]))
    glow = [e for e in animations if "textShadow" in e["keyframes"][0]]
    assert len(glow) == len(cues)
    for cue, settle, event in zip(cues, settles, glow, strict=True):
        timing = event["options"]
        keys = event["keyframes"]
        assert timing["delay"] == pytest.approx(cue * 1000)
        assert [timing["delay"] + key["offset"] * timing["duration"] for key in keys] == (
            pytest.approx([cue * 1000, (cue + 5 / 30) * 1000, settle * 1000,
                           (settle + 0.1) * 1000, (settle + 0.2) * 1000]))
        for key, level in zip(keys, [0, 1, 1, 0.825, 0.65], strict=True):
            alphas = [float(alpha) for alpha in re.findall(r"rgba\([^)]*,([^,)]+)\)",
                                                         key["textShadow"])]
            assert alphas == pytest.approx([1, 1, 1, 1, 0.85, 0.75, *[
                level * alpha for alpha in (1, 0.75, 0.5)]])
            assert key["textShadow"].startswith("0.02em 0 0 rgba(0,0,0,1),")
        assert keys[0]["easing"] == keys[3]["easing"] == scale[0]["keyframes"][1]["easing"]
        assert keys[2]["easing"] == scale[0]["keyframes"][0]["easing"]
    out = [e for e in animations if e["target"] == "stack"]
    assert len(out) == 1 and out[0]["options"] == {
        "delay": 2900, "duration": 100, "fill": "both", "easing": "linear",
    }
    assert out[0]["keyframes"] == [{"opacity": 1}, {"opacity": 0}]


@pytest.mark.parametrize("cues", [[0.1, 0.5, 1.4], [0.066667, 0.433333, 0.8]])
def test_choices_is_settled_at_every_next_cue(cues: list[float], tmp_path: Path) -> None:
    plan = cards.card_layer_plan("choices", {"options": ["One", "Two", "Three"],
                                            "cues": cues}, _style(), frames=90)
    events = _timeline_events("choices", 3, tmp_path, **plan.variables)
    scales = [e for e in events if e["kind"] == "animate" and "transform" in e["keyframes"][0]]
    for previous, next_cue in zip(scales, cues[1:], strict=False):
        end = sum(previous["options"][key] for key in ("delay", "duration")) / 1000
        assert end <= next_cue + 1e-9
        assert previous["keyframes"][-1]["transform"] == "scale(1)"


def test_choices_waits_for_fonts_before_animations(tmp_path: Path) -> None:
    events = _timeline_events("choices", 3, tmp_path, defer_fonts=True)
    assert events[0]["kind"] == "fonts-wait"
    assert events[0]["to"] == {"seeks": [], "registered": []}
    assert all(event["kind"] == "animate" for event in events[1:])


class _HTML(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict]] = []
        self.feed(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))

    @property
    def variables(self) -> list[dict]:
        return json.loads(
            next(
                attrs["data-composition-variables"] for tag, attrs in self.elements if tag == "html"
            )
        )


@pytest.fixture
def image(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "sample.PNG"
    source.parent.mkdir()
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=1280x800",
            "-frames:v",
            "1",
            str(source),
        ],
        timeout_s=30,
    )
    return source


def _preview(template: str, image: Path) -> dict:
    slots = json.loads((cards.template_dir(template) / "preview.json").read_text())["slots"]
    if template == "image_card":
        assert "image" not in slots
        slots["image"] = image
    elif template == "image_compare":
        assert not {"before", "after"}.intersection(slots)
        slots.update(before=image, after=image)
    elif template == "callout_circle":
        slots["image"] = image
    elif template == "file_drop":
        from test_file_drop import clip_source

        slots["clip"] = clip_source(image.parent)
    return cards.preview_videos(template, slots, image.parent)


def _preset(name: str) -> ResolvedStyle:
    style = resolve(f"preset:{name}")
    style.data["graphics"]["card_kinds_on"] = list(cards.OPT_IN_KINDS)
    return style


def _style(**graphics: object) -> ResolvedStyle:
    style = _preset("clean_minimal")
    style.data["graphics"].update(graphics)
    return style


def _default_variable_ids(template: str) -> set[str]:
    optional = ({"specs_rows", "cues", "text_color", "glow_color", "glow_pad_px"}
                if template in {"specs_list", "stat"} else set())
    if template == "stat":
        optional |= {"rows", "row_px"}
    if template == "flow":
        optional |= {"flow_look", "tag_max_width_px", "tag_height_px"}
    return {v["id"] for v in cards.load_template(template)["variables"]} - optional


@pytest.fixture
def video_clip(tmp_path: Path) -> dict:
    source = tmp_path / "clip.mp4"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=gray:s=320x180:r=30", "-t", "4", str(source)], timeout_s=30)
    return {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


@pytest.mark.parametrize("changes,fix", [
    ({"sha256": None}, "sha256"), ({"focus": [1.1, 0]}, "focus"),
    ({"unknown": 1}, "unknown"), ({"speed": 2.1}, "0.5"),
    ({"audio": {"gain_db": 7}}, "gain_db"), ({"in_s": -1}, "in_s"),
])
def test_video_slot_refuses_invalid_objects(video_clip: dict, changes: dict, fix: str) -> None:
    with pytest.raises(ValidationError, match=fix):
        cards.card_layer_plan("video_card", {"clip": {**video_clip, **changes}}, _style())


def test_video_slot_requires_hash_and_refuses_self_and_still(video_clip: dict, image: Path) -> None:
    for value, fix in (({"path": video_clip["path"]}, "sha256"),
                       ("self", "self"),
                       ({"path": str(image), "sha256": "a" * 64}, "clip")):
        with pytest.raises(ValidationError, match=fix):
            cards.card_layer_plan("video_card", {"clip": value}, _style())


@pytest.mark.parametrize("crop", [[0, 0, 320, 180], [17, 9, 127, 95]])
def test_video_slot_accepts_pixel_crop(video_clip: dict, crop: list[int]) -> None:
    rule = cards.video_slots("video_card")["clip"]
    value = cards.video_value("clip", {**video_clip, "crop": crop}, rule)
    assert value["crop"] == crop and value["crop"] is not crop
    assert "crop" not in cards.video_value("clip", video_clip, rule)


@pytest.mark.parametrize("crop", [
    [300, 0, 32, 64], [-1, 0, 64, 64], [0, 0, 31, 64], [0, 0, 64, 31],
    [0, 0, 64.0, 64], [False, 0, 64, 64], [0, 0, 64], None,
])
def test_video_slot_refuses_bad_crop_with_picture_size(video_clip: dict, crop: object) -> None:
    with pytest.raises(ValidationError, match="crop must sit inside the 320x180 picture") as caught:
        cards.video_value("clip", {**video_clip, "crop": crop},
                          cards.video_slots("video_card")["clip"])
    assert caught.value.parameter == "slots.clip"


def test_video_self_refuses_crop() -> None:
    with pytest.raises(ValidationError, match="self takes no crop"):
        cards.video_value("post", {"kind": "self", "crop": [0, 0, 64, 64]},
                          cards.video_slots("reel_feed")["post"])


def test_video_crop_uses_rotated_picture_dimensions(video_clip: dict, tmp_path: Path) -> None:
    source = tmp_path / "rotated.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", video_clip["path"],
                "-c", "copy", str(source)], timeout_s=30)
    assert ffmpeg.probe(source, count_frames=False).rotation == 90
    clip = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    rule = cards.video_slots("video_card")["clip"]
    assert cards.video_value("clip", {**clip, "crop": [100, 200, 80, 120]}, rule)["crop"] == [
        100, 200, 80, 120,
    ]
    with pytest.raises(ValidationError, match="crop must sit inside the 180x320 picture"):
        cards.video_value("clip", {**clip, "crop": [200, 100, 80, 120]}, rule)


@pytest.mark.parametrize("aspect,ratio", [("source", 3 / 4), ("4:3", 4 / 3)])
def test_video_card_source_aspect_uses_crop(video_clip: dict, aspect: str, ratio: float) -> None:
    crop = [32, 16, 96, 128]
    plan = cards.card_layer_plan("video_card", {"clip": {**video_clip, "crop": crop},
                                                "aspect": aspect}, _style(), frames=60)
    window, = plan.windows
    assert abs(window["height"] - window["width"] / ratio) < 2
    assert window["source"]["crop"] == crop
    assert "crop" not in plan.variables["windows"]


def test_video_card_windows_and_bounds(video_clip: dict) -> None:
    plan = cards.card_layer_plan("video_card", {"clip": video_clip, "aspect": "4:5"},
                                 _style(), frames=60)
    window, = plan.windows
    for key in ("x", "y", "width", "height"):
        assert type(window[key]) is int and window[key] % 2 == 0
    assert window["x"] + window["width"] <= plan.width
    assert window["y"] + window["height"] <= plan.height
    assert abs(window["width"] / window["height"] - 4 / 5) < 0.005
    assert window["source"]["audio"] == "mute"
    assert plan.variables["window_fill"] == "clear"
    assert "path" not in plan.variables["windows"]
    with pytest.raises(ValidationError, match="clip ends at 4 s; use in_s at most"):
        cards.card_layer_plan("video_card", {"clip": {**video_clip, "in_s": 3}},
                              _style(), frames=60)
    with pytest.raises(ValidationError, match="needs at least 1.5 s"):
        cards.card_layer_plan("video_card", {"clip": video_clip}, _style(), frames=44)
    with pytest.raises(ValidationError, match="shorten to at most"):
        cards.card_layer_plan("video_card", {"clip": video_clip, "label": "W" * 56},
                              _style(card_width_px=400), frames=60)


@pytest.mark.parametrize("accent", ["", "#A13467"])
def test_video_card_label_rule_uses_only_brand_accent(video_clip: dict, accent: str) -> None:
    style = _style(accent_color="#365A58")
    style.data["brand"]["accent_color"] = accent
    plan = cards.card_layer_plan("video_card", {"clip": video_clip, "label": "Alpha"}, style)
    assert plan.variables["rule_color"] == accent


def test_video_list_validates_each_source(video_clip: dict) -> None:
    rules = {"clips": {"type": "video_list", "required": True, "min_items": 2,
                       "max_items": 6, "sources": ["media", "self"]}}
    values, _ = cards._slots({"clips": [video_clip, "self"]}, rules)
    assert values["clips"][0]["speed"] == 1 and values["clips"][1] == "self"
    with pytest.raises(ValidationError, match=r"slots.clips\[1\]"):
        cards._slots({"clips": [video_clip, {"path": video_clip["path"]}]}, rules)


def test_video_source_needs_stream_duration(video_clip: dict, monkeypatch) -> None:
    info = ffmpeg.probe(Path(video_clip["path"]), count_frames=False)
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs:
                        replace(info, video_duration_s=None))
    with pytest.raises(ValidationError, match="known duration"):
        cards.card_layer_plan("video_card", {"clip": video_clip}, _style())


@pytest.mark.parametrize("crop,ratio", [(None, 16 / 9), ([40, 20, 120, 160], 1)])
def test_video_source_display_aspect_uses_anamorphic_pixels(
    tmp_path: Path, crop: list[int] | None, ratio: float,
) -> None:
    source = tmp_path / "wide.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=s=320x240:r=30,setsar=4/3", "-t", "4", str(source)], timeout_s=30)
    clip = {
        "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    if crop is not None:
        clip["crop"] = crop
    plan = cards.card_layer_plan("video_card", {"clip": clip}, _style(), frames=60)
    window, = plan.windows
    assert window["width"] / window["height"] == pytest.approx(ratio, abs=0.005)


@pytest.fixture
def metrics() -> dict:
    return json.loads(Path(text_metrics.__file__).with_suffix(".json").read_text(encoding="utf-8"))


def test_text_metrics_schema_and_bundled_advances(metrics: dict) -> None:
    assert metrics["schema"] == "yap-studio/text-metrics@1"
    assert set(metrics["fonts"]) == set(fonts.BUNDLED)
    ascii_characters = {chr(code) for code in range(0x20, 0x7F)}
    for name, font in fonts.BUNDLED.items():
        weights = set()
        for face in font.faces:
            if face.style != "normal":
                continue
            limits = [int(weight) for weight in face.weight.split()]
            weights.update(str(weight) for weight in (400, 600, 700)
                           if limits[0] <= weight <= limits[-1])
        assert set(metrics["fonts"][name]) == weights
        for table in metrics["fonts"][name].values():
            assert ascii_characters <= table.keys()
            advances = [value for char, value in table.items() if len(char) == 1]
            assert all(type(value) in (int, float) and 0 < value < 2 for value in advances)
            assert table["_max"] >= max(advances)
            if font.generic != "monospace":
                assert table["W"] > table["i"]
            else:
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                assert len({table[char] for char in letters}) == 1


def test_text_metrics_coverage_is_compact_and_sorted(metrics: dict) -> None:
    for tables in metrics["fonts"].values():
        for table in tables.values():
            coverage = table["_coverage"]
            assert isinstance(coverage, list) and coverage
            previous_end = -2
            for pair in coverage:
                assert isinstance(pair, list) and len(pair) == 2
                start, end = pair
                assert type(start) is int and type(end) is int
                assert 0 <= start <= end <= 0x10FFFF
                assert start > previous_end + 1
                previous_end = end


def test_text_metrics_blocks_bound_covered_samples_and_match_coverage(metrics: dict) -> None:
    for tables in metrics["fonts"].values():
        for table in tables.values():
            blocks = table["_blocks"]
            coverage = table["_coverage"]
            assert isinstance(blocks, dict) and blocks
            assert set(blocks) == {
                str(block)
                for start, end in coverage
                for block in range(start // 64, end // 64 + 1)
            }
            for maximum in blocks.values():
                assert type(maximum) in (int, float) and math.isfinite(maximum)
                assert 0 <= maximum <= table["_max"]
                assert maximum == round(maximum, 4)
            for char, advance in table.items():
                if len(char) == 1 and any(start <= ord(char) <= end for start, end in coverage):
                    assert advance <= blocks[str(ord(char) // 64)]


def test_text_metrics_covered_unsampled_characters_use_block_max(metrics: dict) -> None:
    for family, tables in metrics["fonts"].items():
        for weight, table in tables.items():
            char = next(
                chr(codepoint)
                for start, end in table["_coverage"]
                for codepoint in range(start, end + 1)
                if chr(codepoint) not in table
            )
            expected = table["_blocks"][str(ord(char) // 64)]
            assert text_width_em(char, family, int(weight)) == expected


@pytest.mark.parametrize(("char", "expected"), [
    ("A", 0.6), ("\u0416", 1.1), ("\u0440", 2.7), ("\u2603", 1.25),
])
def test_text_metrics_advance_uses_samples_blocks_and_fallbacks(char: str, expected: float) -> None:
    table = text_metrics._FontTable(
        advances={"A": 0.6, "\u2603": 9.0}, maximum=2.7,
        coverage=[[65, 65], [0x0400, 0x047F]], blocks={"1": 1.2, "16": 1.1},
    )
    assert text_metrics._advance(table, char) == expected


@pytest.mark.parametrize(("weight", "recorded"), [
    (399, "400"), (500, "600"), (600, "600"), (650, "700"), (900, "700"),
])
def test_text_metrics_nearest_weight_ties_go_heavier(
    metrics: dict, weight: int, recorded: str,
) -> None:
    text = "Wi A"
    table = metrics["fonts"]["Inter"][recorded]
    assert text_width_em(text, "Inter", weight) == math.fsum(table[char] for char in text)


@pytest.mark.parametrize("family", fonts.BUNDLED)
def test_text_metrics_unknown_characters_and_tracking(metrics: dict, family: str) -> None:
    tables = metrics["fonts"][family]
    recorded = min(tables, key=lambda key: (abs(int(key) - 600), -int(key)))
    table = tables[recorded]
    text = "Wi A\u2603"
    assert "\u2603" not in table
    base = math.fsum(
        table.get(char, table["_blocks"][str(ord(char) // 64)])
        if any(start <= ord(char) <= end for start, end in table["_coverage"])
        else text_metrics.FALLBACK_EM
        for char in text
    )
    assert text_width_em(text, family, 600, tracking_em=0.04) == base + len(text) * 0.04
    assert text_width_em("", family, 600, tracking_em=0.18) == 0
    assert widest_digit_em(family, 600) == max(table[char] for char in "0123456789")


@pytest.mark.parametrize("weight", [400, 500, 550, 600, 650, 700])
@pytest.mark.parametrize("text", ["Wide Wi %", "Caf\u00e9 \u201cTitle\u201d", "\u2603", "\u0416"])
def test_text_metrics_unbundled_family_is_conservative(text: str, weight: int) -> None:
    width = text_width_em(text, "An installed font", weight)
    for name in text_metrics.UNBUNDLED_BOUND_FAMILIES:
        assert width >= text_width_em(text, name, weight)


def test_text_metrics_unbundled_table_includes_monospace(metrics: dict) -> None:
    tables = [metrics["fonts"][name]["400"] for name in text_metrics.UNBUNDLED_BOUND_FAMILIES]
    for char in "Wi ":
        expected = max(text_metrics.UNBUNDLED_MIN_EM,
                       max(table[char] for table in tables) * text_metrics.UNBUNDLED_MARGIN)
        assert text_width_em(char, "An installed font", 400) == expected


def test_text_metrics_unbundled_bounds_use_frozen_families(metrics: dict) -> None:
    families = (
        "Alata", "Anton SC", "Archivo Black", "Gelasio", "Instrument Serif", "Inter",
        "Permanent Marker", "Space Mono", "Titan One",
    )
    assert text_metrics.UNBUNDLED_BOUND_FAMILIES == families
    assert set(families) <= metrics["fonts"].keys()

    def bound(names: tuple[str, ...]) -> float:
        tables = [metrics["fonts"][name] for name in names]
        nearest = [weights[min(weights, key=lambda key: (abs(int(key) - 700), -int(key)))]
                   for weights in tables]
        return math.fsum(
            max(text_metrics.UNBUNDLED_MIN_EM,
                max(table[char] for table in nearest) * text_metrics.UNBUNDLED_MARGIN)
            for char in "$5"
        )

    expected = bound(families)
    assert expected != bound((*families, "Geist"))
    assert text_width_em("$5", "Unbundled Face", 700) == expected


@pytest.mark.parametrize("weight", [400, 500, 600, 700])
def test_text_metrics_installed_monospace_narrow_letters_do_not_underestimate(weight: int) -> None:
    text = "i" * 20
    width = text_width_em(text, "Liberation Mono", weight)
    assert width >= 20 * 0.7
    assert width >= text_width_em(text, "Space Mono", weight)
    assert width * 64 > 536


def test_text_tag_installed_monospace_title_shrinks_or_reports_capacity() -> None:
    text = "i" * 20
    style = _style(font="Liberation Mono", card_width_px=600)
    try:
        plan = cards.card_layer_plan("text_tag", {"text": text, "variant": "title"}, style)
    except ValidationError as error:
        assert error.parameter == "slots.text"
    else:
        size = plan.variables["text_px"]
        assert cards.MIN_TEXT_PX <= size < 64
        assert text_width_em(text, "Liberation Mono", 600) * size <= 536


def test_text_metrics_missing_cyrillic_uses_browser_fallback_bound(metrics: dict) -> None:
    char = "\u0416"
    coverage = metrics["fonts"]["Instrument Serif"]["400"]["_coverage"]
    assert not any(start <= ord(char) <= end for start, end in coverage)
    assert text_width_em(char * 4, "Instrument Serif", 400) == 4 * 1.25


def test_flow_covered_cyrillic_labels_fit_at_readable_size() -> None:
    label = "\u0416\u0416"
    labels = [label, "Edit", "Ship", "Done"]
    assert text_width_em(label, "Inter", 600) < 3
    plan = cards.card_layer_plan(
        "flow", {"nodes": labels}, _style(font="Inter", card_width_px=960)
    )
    assert plan.variables["text_px"] >= 30
    room = plan.variables["box_width_px"] - 2 * cards.FLOW_LABEL_PAD_X
    assert room == 138
    assert all(text_width_em(label, "Inter", 600) * plan.variables["text_px"] <= room
               for label in labels)


def test_flow_missing_cyrillic_fits_or_reports_node_capacity() -> None:
    label = "\u0416" * 4
    labels = [label, "Edit", "Ship", "Done"]
    try:
        plan = cards.card_layer_plan("flow", {"nodes": labels},
                                     _style(font="Instrument Serif", card_width_px=960))
    except ValidationError as error:
        assert error.parameter == "slots.nodes"
    else:
        assert plan.variables["box_width_px"] - 2 * cards.FLOW_LABEL_PAD_X == 138
        assert plan.variables["text_px"] < 40
        assert text_width_em(label, "Instrument Serif", 600) * plan.variables["text_px"] <= 138


def test_text_metrics_unbundled_uncovered_characters_use_fallback_bound(metrics: dict) -> None:
    char = "\U0010ffff"
    for tables in metrics["fonts"].values():
        for table in tables.values():
            assert not any(start <= ord(char) <= end for start, end in table["_coverage"])
    assert text_width_em(char * 3, "An installed font", 600) == 3 * 1.25


@pytest.mark.parametrize("weight", [400, 600, 700])
def test_text_metrics_unbundled_unsampled_characters_use_block_bounds(
    metrics: dict, weight: int,
) -> None:
    char = "\u0416"
    families = [metrics["fonts"][name] for name in text_metrics.UNBUNDLED_BOUND_FAMILIES]
    tables = [tables[min(tables, key=lambda key: (abs(int(key) - weight), -int(key)))]
              for tables in families]
    assert all(char not in table for table in tables)
    expected = max(
        text_metrics.UNBUNDLED_MIN_EM,
        max(
            table["_blocks"][str(ord(char) // 64)]
            if any(start <= ord(char) <= end for start, end in table["_coverage"])
            else text_metrics.FALLBACK_EM
            for table in tables
        ) * text_metrics.UNBUNDLED_MARGIN,
    )
    assert text_width_em(char, "An installed font", weight) == expected


@pytest.mark.parametrize("weight", [400, 500, 600, 700])
def test_text_metrics_unbundled_widest_digit_uses_conservative_rule(
    metrics: dict, weight: int,
) -> None:
    families = [metrics["fonts"][name] for name in text_metrics.UNBUNDLED_BOUND_FAMILIES]
    tables = [tables[min(tables, key=lambda key: (abs(int(key) - weight), -int(key)))]
              for tables in families]
    expected = max(
        max(text_metrics.UNBUNDLED_MIN_EM,
            max(table[char] for table in tables) * text_metrics.UNBUNDLED_MARGIN)
        for char in "0123456789"
    )
    assert widest_digit_em("An installed font", weight) == expected


def test_text_metrics_generator_records_block_and_all_glyph_max_with_compact_cmap() -> None:
    generator = runpy.run_path(str(Path(__file__).parents[1] / "scripts/gen_text_metrics.py"))

    class Font:
        def getBestCmap(self) -> dict[int, str]:
            return {
                0x2001: "space", 66: "B", 33: "bang", 0x2000: "space", 65: "A",
                63: "bang", 64: "A", 0x2040: "zero",
            }

        def __getitem__(self, name: str) -> SimpleNamespace:
            if name == "head":
                return SimpleNamespace(unitsPerEm=2048)
            assert name == "hmtx"
            return SimpleNamespace(metrics={
                "bang": (500, 0), "A": (1000, 0), "B": (2000, 0),
                "space": (2049, 0), "unencoded": (3101, 0), "zero": (0, 0),
            })

    table = generator["_advances"](Font())
    assert table["_max"] == math.ceil(3101 * 10000 / 2048) / 10000
    assert table["_coverage"] == [[33, 33], [63, 66], [0x2000, 0x2001], [0x2040, 0x2040]]
    assert table["_blocks"] == {
        "0": math.ceil(500 * 10000 / 2048) / 10000,
        "1": math.ceil(2000 * 10000 / 2048) / 10000,
        "128": math.ceil(2049 * 10000 / 2048) / 10000,
        "129": 0,
    }
    assert table["A"] == math.ceil(1000 * 10000 / 2048) / 10000
    assert table["B"] == math.ceil(2000 * 10000 / 2048) / 10000
    assert table["i"] == table["B"]


def test_text_metrics_generator_check_when_dependencies_are_available() -> None:
    pytest.importorskip("fontTools")
    pytest.importorskip("brotli")
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/gen_text_metrics.py"), "--check"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fit_measured_has_no_overflow_slack() -> None:
    text = "WWWWW"
    width_em = text_width_em(text, "Inter", 600)
    assert cards._fit_measured("text", text, width_em * 34, "Inter", 600, 40) == 34
    assert cards._fit_measured("text", "", 1, "Inter", 600, 40) == 40
    with pytest.raises(ValidationError) as caught:
        cards._fit_measured("text", text, math.nextafter(width_em * 30, -math.inf),
                            "Inter", 600, 40, look="title")
    assert caught.value.parameter == "slots.text"
    assert "at 30 px in the title look" in caught.value.detail


@pytest.mark.parametrize("template", ["flow", "text_tag"])
def test_fitted_templates_guard_measured_overflow_before_first_seek(template: str) -> None:
    source = (cards.template_dir(template) / "template.html").read_text()
    guard = re.search(r"function fitText\(element\) \{(.*?)\n    \}", source, re.DOTALL)[1]
    assert "const available = element.clientWidth;" in guard
    assert "if (!(element.scrollWidth > available)) return;" in guard
    assert guard.index("return;") < guard.index("element.style.fontSize")
    assert "Math.floor(size * available / element.scrollWidth)" in guard
    assert "while (element.scrollWidth > available && size > 1)" in guard
    assert "size -= 1;" in guard
    finish = re.search(r"function finishTimeline\(\) \{(.*?)\n    \}", source, re.DOTALL)[1]
    if template == "flow":
        assert 'label.className = "label";' in source
        assert "labels.push(label);" in source
        assert finish.index("labels.forEach(fitText);") < finish.index("tl.seek(0);")
        assert _css_rules(source)[".label"]["justify-content"] == "safe center"
    else:
        assert 'if (v.look === "rule" || v.look === "title")' in finish
        assert 'const text = root.querySelector(".tag-text");' in source
        assert 'const subtext = root.querySelector(".subtext");' in source
        assert finish.index("fitText(text);") < finish.index("tl.seek(0);")
        assert finish.index("fitText(subtext);") < finish.index("tl.seek(0);")
    assert finish.index("tl.seek(0);") < finish.index(f'window.__timelines["{template}"] = tl;')
    assert "document.fonts.ready.then(finishTimeline);" in source
    assert source.index("panel.getBoundingClientRect();") < source.index(
        "document.fonts.ready.then(finishTimeline);")
    assert finish.count("tl.seek(0);") == 1


def test_stat_template_widens_reels_on_measured_digit_overflow() -> None:
    source = (cards.template_dir("stat") / "template.html").read_text()
    guard = re.search(r"function fitSlotReels\(\) \{(.*?)\n    \}", source, re.DOTALL)[1]
    assert "if (!reels.length) return;" in guard
    assert 'for (const digit of "0123456789")' in guard
    assert 'probe.style.width = "max-content";' in guard
    assert "stage.appendChild(probe);" in guard
    assert "Math.max(widest, probe.scrollWidth)" in guard
    assert "probe.remove();" in guard
    assert "if (widest > reels[0].clientWidth)" in guard
    assert '"--card-slot-digit-width", `${Math.ceil(widest)}px`, "important"' in guard
    finish = re.search(r"function finishTimeline\(\) \{(.*?)\n    \}", source, re.DOTALL)[1]
    assert finish.index("fitSlotReels();") < finish.index("tl.seek(0);")
    assert finish.index("tl.seek(0);") < finish.index('window.__timelines["stat"] = tl;')
    assert "document.fonts.ready.then(finishTimeline);" in source
    assert source.index("panel.getBoundingClientRect();") < source.index("document.fonts.ready")


@pytest.mark.parametrize("template", ["flow", "text_tag"])
@pytest.mark.parametrize(("available", "width_at_40", "expected"), [
    (138, 138, None), (138, 90, None), (138, 180, 30), (138, 275, 19),
])
def test_text_overflow_guard_keeps_fitting_sizes_and_shrinks_in_whole_pixels(
    template: str, available: int, width_at_40: int, expected: int | None, tmp_path: Path,
) -> None:
    if tools.find_tool("node") is None:
        pytest.skip("a JavaScript runtime is needed for overflow guard tests")
    source = (cards.template_dir(template) / "template.html").read_text()
    guard = re.search(r"function fitText\(element\) \{(.*?)\n    \}", source, re.DOTALL)[0]
    payload_file = tmp_path / "overflow-input.json"
    payload_file.write_text(json.dumps({
        "guard": guard, "available": available, "width_at_40": width_at_40,
    }))
    runner = tmp_path / "overflow-spy.cjs"
    runner.write_text(r"""
const fs = require("fs");
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const writes = [];
const element = {
  clientWidth: input.available,
  style: {
    set fontSize(value) { writes.push(value); this.size = parseFloat(value); },
  },
  get scrollWidth() {
    // A fixed spacing term forces the correction loop after the first proportional guess.
    const size = this.style.size || 40;
    return Math.max(this.clientWidth, Math.ceil((input.width_at_40 - 2) * size / 40 + 2));
  },
};
global.getComputedStyle = () => ({ fontSize: "40px" });
new Function("element", input.guard + "\nfitText(element);")(element);
process.stdout.write(JSON.stringify({ writes, size: element.style.size,
  width: element.scrollWidth }));
""")
    result = json.loads(
        tools.run_tool("node", [str(runner), str(payload_file)], timeout_s=10).stdout
    )
    if expected is None:
        assert result["writes"] == []
    else:
        assert result["size"] == expected
        assert all(re.fullmatch(r"[0-9]+px", value) for value in result["writes"])
        assert result["width"] <= available
        assert int(result["writes"][0][:-2]) == math.floor(40 * available / width_at_40)
        assert [int(value[:-2]) for value in result["writes"]] == list(
            range(int(result["writes"][0][:-2]), expected - 1, -1)
        )


@pytest.mark.parametrize(("widest", "expected"), [(60, None), (70, None), (88.4, "89px")])
def test_stat_reel_guard_measures_all_digits_and_only_widens_overflow(
    widest: float, expected: str | None, tmp_path: Path,
) -> None:
    if tools.find_tool("node") is None:
        pytest.skip("a JavaScript runtime is needed for reel guard tests")
    source = (cards.template_dir("stat") / "template.html").read_text()
    guard = re.search(r"function fitSlotReels\(\) \{(.*?)\n    \}", source, re.DOTALL)[0]
    payload_file = tmp_path / "reel-input.json"
    payload_file.write_text(json.dumps({"guard": guard, "widest": widest}))
    runner = tmp_path / "reel-spy.cjs"
    runner.write_text(r"""
const fs = require("fs");
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const measured = [];
const writes = [];
let removed = false;
const probe = {
  style: {},
  get scrollWidth() {
    measured.push(this.textContent);
    return Math.ceil(this.textContent === "8" ? input.widest : 50);
  },
  remove() { removed = true; },
};
const stage = { appendChild(child) { if (child !== probe) throw new Error("unexpected probe"); } };
const root = { style: { setProperty(...args) { writes.push(args); } } };
global.document = { createElement() { return probe; } };
new Function("reels", "stage", "root", input.guard + "\nfitSlotReels();")(
  [{ clientWidth: 70 }], stage, root);
process.stdout.write(JSON.stringify({ measured, writes, removed }));
""")
    result = json.loads(
        tools.run_tool("node", [str(runner), str(payload_file)], timeout_s=10).stdout
    )
    assert result["measured"] == list("0123456789")
    assert result["removed"] is True
    assert result["writes"] == (
        [["--card-slot-digit-width", expected, "important"]] if expected else []
    )


@pytest.mark.parametrize("template", PANEL_TEMPLATES)
@pytest.mark.parametrize("preset", PRESETS)
def test_metadata_plan_and_prepared_contract(
    template: str,
    preset: str,
    image: Path,
    tmp_path: Path,
) -> None:
    metadata = cards.load_template(template)
    source = (cards.template_dir(template) / "template.html").read_text()
    parser = _HTML(source)
    assert metadata["id"] == template
    assert metadata["version"] == 1
    assert metadata["reads"] == "graphics"
    assert metadata["derived_from"] == [
        {"item": item, "path": f"registry/{kind}/{item}/{item}.html"}
        for kind, item in UPSTREAM[template]
    ]
    assert metadata["variables"] == parser.variables
    assert metadata["box"] == ({"width": 1080, "height": 1920, "x": 0, "y": 0}
                              if template in {"playing_grid", "reel_feed"} else {
        "width": "graphics.card_width_px + 80",
        "height": "window height + label row + 120",
        "y": "internal.key_text_top_px",
    } if template == "video_card" else {
        "width": "graphics.card_width_px + 80",
        "height": 360,
        "y": "internal.key_text_top_px",
        "panel_max_height": 272,
    })
    plan = cards.card_layer_plan(template, _preview(template, image), _preset(preset))
    assert set(plan.variables) == _default_variable_ids(template)
    for variable in parser.variables:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", variable["id"])
        assert variable["type"] in {"string", "number", "color", "boolean", "enum"}
        assert "label" in variable and "default" in variable
        if variable["type"] == "enum":
            assert all(set(option) == {"value", "label"} for option in variable["options"])
            assert plan.variables[variable["id"]] in [o["value"] for o in variable["options"]]
    composition = cards.prepare_card(plan, tmp_path / "work")
    duration = hyperframes.layer_duration_s(plan.frames, plan.fps)
    assert hyperframes.preflight(composition, plan.frames, plan.fps) == duration
    prepared = _HTML((composition / "index.html").read_text())
    original = next(attrs for _, attrs in parser.elements if attrs.get("id") == "root")
    actual = next(attrs for _, attrs in prepared.elements if attrs.get("id") == "root")
    assert actual == {
        **original,
        "data-width": str(plan.width),
        "data-height": str(plan.height),
        "data-duration": str(duration),
    }
    for name in TEXT_SIZES[template]:
        assert type(plan.variables[name]) is int and plan.variables[name] >= cards.MIN_TEXT_PX
    assert not (composition / hyperframes.GSAP_SCRIPT).exists()


@pytest.mark.parametrize("template", [kind for kind, sources in UPSTREAM.items() if sources])
def test_static_template_contract(template: str) -> None:
    source = (cards.template_dir(template) / "template.html").read_text()
    header = ["<!doctype html>", "<!--"]
    for kind, item in UPSTREAM[template]:
        header += [
            f'  Derived from HyperFrames registry item "{item}"',
            f"  (registry/{kind}/{item}/{item}.html at commit "
            "0d5e395cd31d0f491e9b26d0da110da6e231c512)",
        ]
    header += ["  Copyright 2026 HeyGen, Inc. Apache License 2.0 (LICENSES/Apache-2.0.txt)."]
    if template == "gravity_drop":
        assert source.startswith("<!doctype html>\n<html")
        assert "registry" not in source and "gsap" not in source
        assert ".animate([" in source and "fill: 'both'" in source
    else:
        assert source.startswith("\n".join(header) + "\n  Modified for Yap Studio, 2026: ")
        assert re.search(r"Modified for Yap Studio, 2026: [^\n]+\.\n-->\n", source)
    parser = _HTML(source)
    roots = [attrs for _, attrs in parser.elements if "data-composition-id" in attrs]
    assert roots == [
        {
            **({"data-no-timeline": None} if template == "gravity_drop" else {}),
            "id": "root",
            "data-composition-id": template,
            "data-start": "0",
            "data-width": "1080" if template in {"playing_grid", "reel_feed"} else "1040",
            "data-height": "1920" if template in {"playing_grid", "reel_feed"} else (
                "706" if template == "video_card" else "360"),
            "data-duration": "3",
        }
    ]
    assert sum(attrs.get("id") == "root" for _, attrs in parser.elements) == 1
    assert sum(tag == "style" for tag, _ in parser.elements) == 1
    assert "<style>\n/* BUNDLED_FONT_FACES */\n" in source
    assert {"charset": "utf-8"} in [attrs for tag, attrs in parser.elements if tag == "meta"]
    scripts = [attrs for tag, attrs in parser.elements if tag == "script"]
    if template == "gravity_drop":
        assert scripts == [{}]
    else:
        assert scripts == [{"src": "gsap.min.js"}, {}]
        assert "gsap.timeline({ paused: true })" in source
        assert f'window.__timelines["{template}"] = tl;' in source
    assert "window.__hyperframes.getVariables()" in source
    assert 'document.getElementById("root")' in source
    assert '"use strict";' in source
    if template in {"video_card", "playing_grid", "reel_feed"}:
        assert "w.fade_in/fps" in source and 'ease: "none"' in source
    else:
        assert "Number(root.dataset.duration)" in source
        assert "Math.min(0.3, duration * 0.15)" in source
    assert re.search(r"\}\)\(\);\s*</script>\s*</body>", source)
    for _, attrs in parser.elements:
        if attrs.get("id") != "root":
            assert not {"data-start", "data-duration", "data-track-index"}.intersection(attrs)
    families = re.findall(r"font-family:\s*([^;}]+)", source)
    if template == "gravity_drop":
        assert "fontFamily: drop.font_family" in source
    else:
        assert families and set(families) <= {"var(--card-font)", "var(--card-mono)"}
    used = re.findall(r"var\(--([a-zA-Z0-9_-]+)", source)
    assert all(name.startswith("card-") for name in used)
    assert not set(used).intersection(v["id"] for v in parser.variables)
    if template != "gravity_drop":
        assert 'root.style.setProperty(`--card-${name}`, value, "important")' in source
    banned = (
        "timeScale",
        "Math.random",
        "Date",
        "performance.now",
        "setTimeout",
        "setInterval",
        "requestAnimationFrame",
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "http://",
        "https://",
        "\u2013",
        "\u2014",
    )
    assert not any(token in source for token in banned)
    if template == "image_card":
        assert r"/^image\.(png|jpe?g|webp)$/.test(source)" in source
        assert "image.src = source" in source
        assert "width: auto; height: auto;" in source
    if template == "image_compare":
        assert r"/^(image|before|after)\.(png|jpe?g|webp)$/.test(source)" in source
        assert "image.src = source" in source
    if template in {"terminal", "checklist", "steps", "flow"}:
        assert "JSON.parse(value)" in source and "catch (_) { return []; }" in source
    if template in {"checklist", "steps", "flow"}:
        assert "list(v.cues, Number)" in source
    if template == "terminal":
        assert "JSON.parse(v.schedule)" in source
        assert "schedule.rows.forEach(([t, n])" in source
        assert not any(token in source for token in ("lcgState", "chunksOf", "squeeze", ".split("))


@pytest.mark.parametrize("template", ["edit_timeline", "storyboard", "jiggle_delete", "file_drop"])
def test_own_code_template_contract(template: str) -> None:
    source = (cards.template_dir(template) / "template.html").read_text()
    parser = _HTML(source)
    roots = [attrs for _, attrs in parser.elements if "data-composition-id" in attrs]
    assert roots == [{"id": "root", "data-composition-id": template, "data-no-timeline": None,
                      "data-start": "0", "data-width": "1040", "data-height": "360",
                      "data-duration": "3"}]
    assert [attrs for tag, attrs in parser.elements if tag == "script"] == [{}]
    assert "registry" not in source and "gsap" not in source
    assert "<style>\n/* BUNDLED_FONT_FACES */\n" in source
    assert '\"use strict\";' in source
    assert "window.__hyperframes.getVariables()" in source
    assert 'root.style.setProperty(`--card-${name}`, value, "important")' in source
    assert "Number(root.dataset.duration)" in source
    assert "Math.min(0.3, duration * 0.15)" in source
    assert ".animate(" in source
    if template == "edit_timeline":
        assert 'steps(${labels.length}, end)' in source
        assert "-0.5 / v.fps" in source
    families = {"var(--card-font)"}
    if template != "jiggle_delete":
        families.add("var(--card-mono)")
    assert set(re.findall(r"font-family:\s*([^;}]+)", source)) == families
    assert all(name.startswith("card-") for name in re.findall(r"var\(--([\w-]+)", source))
    for banned in ("Math.random", "Date", "getBoundingClientRect", "offsetWidth", "innerHTML",
                   "setTimeout", "requestAnimationFrame", "http://", "https://",
                   "\u2013", "\u2014"):
        assert banned not in source


@pytest.mark.parametrize("template", PANEL_TEMPLATES)
@pytest.mark.parametrize("preset", PRESETS)
def test_geometry_and_frame_window(template: str, preset: str, image: Path) -> None:
    style = _preset(preset)
    slots = _preview(template, image)
    plan = cards.card_layer_plan(template, slots, style, fps=24)
    assert plan.width == (1080 if template in {"playing_grid", "reel_feed"} else
                          style.data["graphics"]["card_width_px"] + 80)
    assert plan.x == (1080 - plan.width) // 2
    assert plan.y == (0 if template in {"playing_grid", "reel_feed"} else 280)
    assert plan.height == (1920 if template in {"playing_grid", "reel_feed"} else (
        plan.variables["panel_max_height_px"] + 88 if template == "video_card" else 360))
    if template != "video_card":
        assert plan.variables["panel_max_height_px"] == 272
    assert (
        plan.variables["pad_x_px"],
        plan.variables["pad_top_px"],
        plan.variables["pad_bottom_px"],
    ) == (40, 24, 64)
    assert plan.frames == max(1, round(style.data["graphics"]["hold_s"] * 24))
    minimum = {"transfer": cards.TRANSFER_MIN_S, "image_compare": cards.COMPARE_MIN_S,
               "steps": cards.STEPS_MIN_S, "flow": cards.FLOW_MIN_S, "reel_feed": 2.5,
               "gravity_drop": 2.1}
    explicit_frames = max(48, math.ceil(minimum.get(template, 0) * 24))
    if template == "jiggle_delete":
        slots = {**slots, "wiggle_at": 0, "cues": [0.5]}
    explicit = cards.card_layer_plan(template, slots, style, frames=explicit_frames, fps=24)
    assert explicit.frames == explicit_frames and explicit.fps == 24
    for invalid in (0, -1, True, 1.2):
        with pytest.raises(ValidationError, match="frames"):
            cards.card_layer_plan(template, slots, style, frames=invalid)
    for invalid in (0, -1, True, 1.2):
        with pytest.raises(ValidationError, match="fps"):
            cards.card_layer_plan(template, slots, style, fps=invalid)


@pytest.mark.parametrize("template", cards.CARD_TEMPLATES)
@pytest.mark.parametrize("preset", PRESETS)
def test_all_pixel_variables_are_integers(template: str, preset: str, image: Path) -> None:
    plan = cards.card_layer_plan(template, _preview(template, image), _preset(preset))
    pixels = {name: value for name, value in plan.variables.items() if name.endswith("_px")}
    assert pixels
    assert all(type(value) is int for value in pixels.values()), pixels


@pytest.mark.parametrize("template", PANEL_TEMPLATES)
def test_one_frame_and_wide_geometry(template: str, image: Path, tmp_path: Path) -> None:
    slots = _preview(template, image)
    style = _style(card_width_px=1080)
    if template in {"terminal", "transfer", "image_compare", "steps", "flow", "edit_timeline",
                    "storyboard", "jiggle_delete", "file_drop", "video_card", "prompt_bar",
                    "comment_cta", "comment_drop", "icon_strike", "playing_grid", "reel_feed",
                    "gravity_drop"}:
        with pytest.raises(ValidationError, match="frames"):
            cards.card_layer_plan(template, slots, style, frames=1)
        minimum = {"terminal": cards.TERMINAL_MIN_S, "transfer": cards.TRANSFER_MIN_S,
                   "image_compare": cards.COMPARE_MIN_S, "steps": cards.STEPS_MIN_S,
                   "flow": cards.FLOW_MIN_S,
                   "edit_timeline": cards.EDIT_TIMELINE_MIN_S,
                   "storyboard": cards.STORYBOARD_MIN_S,
                   "jiggle_delete": cards.JIGGLE_DELETE_MIN_S,
                   "file_drop": cards.FILE_DROP_MIN_S, "video_card": 1.5,
                   "prompt_bar": cards.PROMPT_BAR_MIN_S,
                   "comment_cta": cards.COMMENT_CTA_MIN_S,
                   "comment_drop": cards.COMMENT_DROP_MIN_S,
                   "icon_strike": cards.ICON_STRIKE_MIN_S,
                   "playing_grid": 1.5, "reel_feed": 2.5,
                   "gravity_drop": cards.GRAVITY_MIN_S}[template]
        if template == "gravity_drop":
            slots = {"drops": [{"text": "x", "at": 0}]}
        frames = math.ceil(minimum * 30)
        if template == "jiggle_delete":
            slots = {**slots, "wiggle_at": 0, "cues": [0.5]}
    else:
        frames = 1
    plan = cards.card_layer_plan(template, slots, style, frames=frames)
    assert (plan.width, plan.x) == ((1080, 0) if template in {"playing_grid", "reel_feed"}
                                  else (1160, -40))
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert hyperframes.preflight(composition, frames) == hyperframes.layer_duration_s(frames)


@pytest.mark.parametrize(("fps", "minimum_frames"), [(30, 45), (24, 36)])
def test_terminal_minimum_frame_window(fps: int, minimum_frames: int) -> None:
    for type_at in (0, 0.4, 2):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan(
                "terminal",
                {"command": "x", "type_at": type_at},
                _style(),
                frames=minimum_frames - 1,
                fps=fps,
            )
        assert caught.value.parameter == "frames"
        assert (
            f"needs at least 1.5 s ({minimum_frames} frames at {fps} fps)"
            in caught.value.detail
        )
        assert "raise graphics.hold_s or pass more frames" in caught.value.detail
    plan = cards.card_layer_plan(
        "terminal", {"command": "x"}, _style(), frames=minimum_frames, fps=fps
    )
    assert plan.frames == minimum_frames and plan.fps == fps
    assert json.loads(plan.variables["schedule"])["type_start"] == 0.4


@pytest.mark.parametrize(("fps", "minimum_frames"), [(30, 45), (24, 36)])
def test_terminal_hold_obeys_minimum(fps: int, minimum_frames: int) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("terminal", {"command": "x"}, _style(hold_s=0.5), fps=fps)
    assert caught.value.parameter == "frames"
    plan = cards.card_layer_plan("terminal", {"command": "x"}, _style(hold_s=1.5), fps=fps)
    assert plan.frames == minimum_frames
    assert json.loads(plan.variables["schedule"])["type_start"] == 0.4


def test_steps_slot_contract() -> None:
    slots = cards.load_template("steps")["slots"]
    assert slots == {
        "steps": {"type": "string_list", "required": True, "min_items": 2, "max_items": 6,
                  "min_length": 1, "max_length": 24},
        "cues": {"type": "number_list", "required": False, "min_items": 0, "max_items": 6,
                 "default": []},
    }
    assert json.loads((cards.template_dir("steps") / "preview.json").read_text()) == {
        "slots": {"steps": ["Draft", "Review", "Ship"]},
    }
    for labels in (["x" * 24, "y"], ["A"] * 6):
        plan = cards.card_layer_plan("steps", {"steps": labels}, _style())
        assert json.loads(plan.variables["steps"]) == labels
        assert len(json.loads(plan.variables["cues"])) == len(labels)


@pytest.mark.parametrize("labels", [
    [], ["A"], ["A"] * 7, "A,B", ("A", "B"), None,
    ["", "B"], [" ", "B"], ["x" * 25, "B"], [True, "B"], [1, "B"], [None, "B"],
])
def test_steps_rejects_invalid_labels(labels: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("steps", {"steps": labels}, _style())
    assert caught.value.parameter == "slots.steps"


@pytest.mark.parametrize("cues", [
    [0] * 7, [0] * 3, [-0.001], [math.inf], [-math.inf], [math.nan],
    [True], ["1"], [None], [0.6, 0.5], [3], "1", (1,), None,
])
def test_steps_rejects_invalid_cues(cues: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("steps", {"steps": ["A", "B"], "cues": cues}, _style())
    assert caught.value.parameter == "slots.cues"


@pytest.mark.parametrize(("frames", "fps"), [(60, 30), (48, 24), (75, 30), (240, 24)])
def test_steps_cue_window_boundaries(frames: int, fps: int) -> None:
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - 0.6
    for cues in ([0, 0], [0, latest], [latest, latest]):
        plan = cards.card_layer_plan("steps", {"steps": ["A", "B"], "cues": cues},
                                     _style(), frames=frames, fps=fps)
        assert json.loads(plan.variables["cues"]) == cues
    for cues in ([-0.001], [0, latest + 0.001], [out_start]):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("steps", {"steps": ["A", "B"], "cues": cues},
                                  _style(), frames=frames, fps=fps)
        assert caught.value.parameter == "slots.cues"
        assert "leave 0.6 s before the fade-out" in caught.value.detail
    with pytest.raises(ValidationError, match="one per step"):
        cards.card_layer_plan("steps", {"steps": ["A", "B"], "cues": [0, 0, 0]},
                              _style(), frames=frames, fps=fps)


@pytest.mark.parametrize("count", range(2, 7))
@pytest.mark.parametrize(("frames", "fps"), [(60, 30), (75, 30), (240, 24)])
def test_steps_default_rhythm(count: int, frames: int, fps: int) -> None:
    plan = cards.card_layer_plan("steps", {"steps": ["A"] * count},
                                 _style(), frames=frames, fps=fps)
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    entrance_end = 0.3 + 0.24 + (count - 1) * 0.06
    gap = (out_start - 0.6 - entrance_end) / (count - 1)
    cues = json.loads(plan.variables["cues"])
    assert cues == pytest.approx([entrance_end + i * gap for i in range(count)])
    assert out_start - cues[-1] == pytest.approx(0.6)
    assert all(a <= b for a, b in zip(cues, cues[1:], strict=False))


@pytest.mark.parametrize("given", [[0.1], [0.1, 0.4], [1.5], [1.5, 1.5]])
def test_steps_partial_cues_preserve_order(given: list[float]) -> None:
    slots = {"steps": ["A", "B", "C"]}
    default = cards.card_layer_plan("steps", slots, _style(), frames=75)
    defaults = json.loads(default.variables["cues"])
    plan = cards.card_layer_plan("steps", {**slots, "cues": given}, _style(), frames=75)
    cues = json.loads(plan.variables["cues"])
    assert cues[:len(given)] == given
    for i in range(len(given), len(cues)):
        assert cues[i] == max(cues[i - 1], defaults[i])
    assert all(a <= b for a, b in zip(cues, cues[1:], strict=False))


@pytest.mark.parametrize(("fps", "frames"), [(30, 60), (24, 48)])
def test_steps_minimum_duration(fps: int, frames: int) -> None:
    slots = {"steps": ["A"] * 6}
    with pytest.raises(ValidationError, match="needs at least 2 s"):
        cards.card_layer_plan("steps", slots, _style(), frames=frames - 1, fps=fps)
    with pytest.raises(ValidationError, match="frames"):
        cards.card_layer_plan("steps", slots, _style(hold_s=1), fps=fps)
    plan = cards.card_layer_plan("steps", slots, _style(hold_s=2), fps=fps)
    assert plan.frames == frames
    assert json.loads(plan.variables["cues"])[0] == pytest.approx(0.84)


@pytest.mark.parametrize("preset", PRESETS)
def test_steps_six_long_labels_explain_capacity(preset: str) -> None:
    resolved = _preset(preset)
    body_width = resolved.data["graphics"]["card_width_px"] - 48
    capacity = math.floor((body_width - 6 * 38 - 5 * 12) / (0.76 * 30))
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("steps", {"steps": ["x" * 24] * 6}, resolved)
    assert caught.value.parameter == "slots.steps"
    assert f"at most {capacity} uppercase characters in total" in caught.value.detail
    assert "remove steps" in caught.value.detail and "30 px" in caught.value.detail


def test_steps_one_row_fit_boundary_and_uppercase_expansion() -> None:
    plan = cards.card_layer_plan("steps", {"steps": ["a" * 10, "b" * 10]},
                                 _style(card_width_px=600))
    assert plan.variables["text_px"] == 30
    with pytest.raises(ValidationError, match="at most 20 uppercase characters"):
        cards.card_layer_plan("steps", {"steps": ["a" * 11, "b" * 10]},
                              _style(card_width_px=600))
    with pytest.raises(ValidationError, match="slots.steps"):
        cards.card_layer_plan("steps", {"steps": ["ß" * 10, "b"]}, _style(card_width_px=600))
    roomy = cards.card_layer_plan("steps", {"steps": ["A", "B"]}, _style())
    assert roomy.variables["text_px"] == 36


def test_steps_panel_height_fit() -> None:
    resolved = _style()
    internal = resolved.data["internal"]
    internal["key_text_bottom_px"] = internal["key_text_top_px"] + 88 + 120
    plan = cards.card_layer_plan("steps", {"steps": ["A", "B"]}, resolved)
    assert plan.variables["text_px"] == 35
    internal["key_text_bottom_px"] = internal["key_text_top_px"] + 88 + 100
    with pytest.raises(ValidationError, match="panel height") as caught:
        cards.card_layer_plan("steps", {"steps": ["A", "B"]}, resolved)
    assert caught.value.parameter == "slots.steps"


def test_steps_state_css_and_geometry_match_adapter() -> None:
    source = (cards.template_dir("steps") / "template.html").read_text()
    rules = _css_rules(source)
    assert rules[".panel"]["padding"] == f"{cards.CONTENT_PAD}px"
    assert rules[".rail"]["gap"] == f"{cards.STEPS_CHIP_GAP}px"
    assert rules[".rail"]["flex-wrap"] == "nowrap"
    chip = rules[".chip"]
    assert chip["padding"] == f"{cards.STEPS_CHIP_PAD_Y}px {cards.STEPS_CHIP_PAD_X}px"
    assert chip["border"].startswith(f"{cards.STEPS_CHIP_BORDER}px solid color-mix(")
    assert chip["line-height"] == str(cards.STEPS_LINE_HEIGHT)
    assert chip["letter-spacing"] == f"{cards.STEPS_TRACKING_EM}em"
    assert chip["font-family"] == "var(--card-mono)"
    active = rules['.chip[data-state="active"]']
    assert active["background"] == "var(--card-accent)"
    assert active["color"] == "var(--card-accent-ink)"
    assert "color-mix(" in rules['.chip[data-state="done"]']["color"]
    assert 'chip.setAttribute("data-state", "pending")' in source
    assert "step.toUpperCase()" in source
    assert "Array.from(label.textContent).length * (0.62 + 0.14) * v.text_px" in source
    assert "cues.forEach(function (at, i)" in source
    assert 'tl.set(chips[i], { attr: { "data-state": "active" } }, at);' in source
    assert 'tl.set(chips[i - 1], { attr: { "data-state": "done" } }, at);' in source
    assert not any(token in source for token in ("transition:", "cqw", "cqh", "cqmin", "cqmax"))


@pytest.mark.parametrize("cues", [[0.8, 1.2, 1.5], [0, 0, 0]])
def test_steps_timeline_snaps_at_resolved_cues(cues: list[float], tmp_path: Path) -> None:
    plan = cards.card_layer_plan("steps", {"steps": ["A", "B", "C"], "cues": cues},
                                 _style(), frames=75)
    events = _timeline_events("steps", hyperframes.layer_duration_s(75),
                              tmp_path, **plan.variables)
    active = [e for e in events if e["to"].get("attr", {}).get("data-state") == "active"]
    done = [e for e in events if e["to"].get("attr", {}).get("data-state") == "done"]
    assert [e["at"] for e in active] == cues
    assert [e["at"] for e in done] == cues[1:]
    assert all(e["kind"] == "set" for e in active + done)
    entrance = [e for e in events if e["kind"] == "fromTo" and e["target"] == "chip"]
    assert [e["at"] for e in entrance] == pytest.approx([0.3, 0.36, 0.42])
    assert all(e["from"]["opacity"] == 0 and e["to"]["duration"] == 0.24 for e in entrance)


def test_flow_slot_contract() -> None:
    assert cards.CARD_TEMPLATES[7:9] == ("steps", "flow")
    assert cards.load_template("flow")["slots"] == {
        "nodes": {"type": "string_list", "required": True, "min_items": 2, "max_items": 4,
                  "min_length": 1, "max_length": 18},
        "cues": {"type": "number_list", "required": False, "min_items": 0, "max_items": 4,
                 "default": []},
    }
    assert json.loads((cards.template_dir("flow") / "preview.json").read_text()) == {
        "slots": {"nodes": ["Record", "Edit", "Publish"]},
    }
    for labels in (["x" * 18, "y" * 18], ["A"] * 4, ["录制", "编辑", "发布"]):
        plan = cards.card_layer_plan("flow", {"nodes": labels}, _style())
        assert json.loads(plan.variables["nodes"]) == labels
        assert len(json.loads(plan.variables["cues"])) == len(labels)


def test_flow_look_schema_and_existing_boxes_variables() -> None:
    from yap_studio.style import SHIPPED_PRESETS

    rule = json.loads((Path(cards.__file__).parent / "style/schema-v1.json").read_text())[
        "properties"]["graphics"]["properties"]["flow_look"]
    assert rule["default"] == "boxes" and rule["enum"] == ["boxes", "tags"]
    for preset in SHIPPED_PRESETS:
        assert resolve(f"preset:{preset}").data["graphics"]["flow_look"] == "boxes"
    assert resolve_style_data({"schema_version": 1, "name": "Old"})[
        "graphics"]["flow_look"] == "boxes"
    with pytest.raises(ValidationError, match="flow_look"):
        resolve_style_data({"schema_version": 1, "name": "Bad",
                            "graphics": {"flow_look": "other"}})
    plan = cards.card_layer_plan("flow", {"nodes": ["Record", "Edit", "Publish"]},
                                 _style(flow_look="boxes"))
    assert plan.variables == {
        "paper_color": "#F4F3EE", "titlebar_color": "#E4E2DA", "bar_ink_color": "#222222",
        "ink_color": "#222222", "accent_color": "#365A58", "accent_ink_color": "#F4F3EE",
        "font_family": '"Yap Studio Inter", sans-serif',
        "mono_family": '"Yap Studio Space Mono", monospace', "card_radius_px": 16,
        "card_shadow": "0 0 0 2px color-mix(in srgb, #222222 12%, transparent), "
                       "0 8px 18px rgb(0 0 0 / 0.16)",
        "panel_width_px": 960, "panel_max_height_px": 272, "pad_x_px": 40,
        "pad_top_px": 24, "pad_bottom_px": 64,
        "nodes": '["Record", "Edit", "Publish"]',
        "cues": "[0.54, 0.6947000000000001, 0.8494]", "text_px": 40,
        "box_width_px": 256, "box_height_px": 120, "arrow_gap_px": 72,
    }


@pytest.mark.parametrize("count", [2, 3])
def test_flow_tags_fit_and_compact_geometry(count: int) -> None:
    labels = ["Record", "Edit", "Publish"][:count]
    plan = cards.card_layer_plan("flow", {"nodes": labels}, _style(flow_look="tags"))
    v = plan.variables
    assert v["flow_look"] == "tags" and json.loads(v["nodes"]) == labels
    assert v["text_px"] == 60
    assert v["tag_height_px"] == math.ceil(v["text_px"] * 1.46)
    stack = count * v["tag_height_px"] + (count - 1) * 36
    assert plan.height == stack + v["pad_top_px"] + v["pad_bottom_px"]
    assert plan.height % 2 == 0 and plan.width % 4 == 0
    assert plan.width <= 864 and plan.x == (1080 - plan.width) // 2
    assert plan.y == 280
    declared = {v["id"] for v in cards.load_template("flow")["variables"]}
    assert set(v) <= declared
    narrow = cards.card_layer_plan("flow", {"nodes": ["Record", "Edit"]},
                                   _style(flow_look="tags", card_width_px=200))
    assert narrow.variables["text_px"] == 60
    with pytest.raises(ValidationError, match="shorten.*WWWW.*52 px") as caught:
        cards.card_layer_plan("flow", {"nodes": ["W" * 18, "Edit"]},
                              _style(flow_look="tags", card_width_px=200))
    assert caught.value.parameter == "slots.nodes"


def test_flow_tags_refuses_four_nodes() -> None:
    with pytest.raises(ValidationError, match="use 2 or 3 nodes, or the boxes look") as caught:
        cards.card_layer_plan("flow", {"nodes": ["A"] * 4}, _style(flow_look="tags"))
    assert caught.value.parameter == "slots.nodes"


def test_flow_tags_largest_phone_size_and_readable_refit(monkeypatch) -> None:
    monkeypatch.setattr(cards, "text_width_em", lambda text, *args, **kwargs:
                        12.3 if text == "Long" else 2)
    style = _style(flow_look="tags")
    plan = cards.card_layer_plan("flow", {"nodes": ["Long", "Edit"]}, style)
    assert plan.variables["text_px"] == 55
    assert plan.width <= 864
    assert (12.3 + 2 * 0.42) * 56 * 1.16 + 16 > 864
    compact = cards.card_layer_plan("flow", {"nodes": ["One", "Two"]}, style,
                                    max_height_px=218)
    assert compact.variables["text_px"] == 52
    assert compact.height == 216 and compact.y == 280
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("flow", {"nodes": ["One", "Two"]}, style, max_height_px=214)
    assert caught.value.detail == (
        "use shorter tags, or another moment with more room above the head, to fit at 52 px")
    with pytest.raises(ValidationError, match="shorten.*Long.*52 px"):
        cards.card_layer_plan("flow", {"nodes": ["Long", "Edit"]}, style, max_width_px=800)


@pytest.mark.parametrize(("frames", "fps"), [(45, 30), (36, 24), (75, 30)])
@pytest.mark.parametrize("given", [[], [0.1], [0.2, 0.4, 0.6]])
@pytest.mark.parametrize("count", [2, 3])
def test_flow_tags_cues_and_timeline(
    frames: int, fps: int, given: list[float], count: int, tmp_path: Path,
) -> None:
    given = given[:count]
    plan = cards.card_layer_plan("flow", {"nodes": ["A", "B", "C"][:count], "cues": given},
                                 _style(flow_look="tags"), frames=frames, fps=fps)
    duration = hyperframes.layer_duration_s(frames, fps)
    cues = json.loads(plan.variables["cues"])
    assert cues[:len(given)] == given
    assert cues == sorted(cues)
    assert cues[-1] + 5 / 30 + 0.3 <= duration - 0.1
    events = _timeline_events("flow", duration, tmp_path, **plan.variables)
    pops = [e for e in events if e["kind"] == "fromTo" and e["target"] == "flow-tag"
            and "scale" in e["to"]]
    fades = [e for e in events if e["kind"] == "fromTo" and e["target"] == "flow-tag"
             and "opacity" in e["to"]]
    arrows = [e for e in events if e["kind"] == "fromTo" and e["target"] == "tag-arrow"
              and "strokeDashoffset" in e["to"]]
    assert [e["at"] for e in pops] == cues
    assert [e["at"] for e in fades] == cues
    assert all(e["from"]["scale"] == 0.6 and e["to"]["scale"] == 1
               and e["to"]["duration"] == 5 / 30 and e["to"]["ease"] == "back.out(2.2)"
               for e in pops)
    assert all(e["from"]["opacity"] == 0 and e["to"]["opacity"] == 1
               and e["to"]["duration"] == 2 / 30 for e in fades)
    assert [e["at"] for e in arrows] == pytest.approx([cue + 5 / 30 for cue in cues[:-1]])
    assert all(e["to"]["duration"] == 0.2 for e in arrows)
    if not given:
        assert all(cue + 5 / 30 + 0.2 <= next_cue + 1e-9
                   for cue, next_cue in zip(cues, cues[1:], strict=False))
    out = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "tag-stack"
               and e["to"]["opacity"] == 0)
    assert out["at"] == pytest.approx(duration - 0.1) and out["to"]["duration"] == 0.1
    fallback = _timeline_events("flow", duration, tmp_path,
                                **{**plan.variables, "cues": json.dumps(given)})
    assert [e["at"] for e in fallback if e["kind"] == "fromTo"
            and e["target"] == "flow-tag" and "scale" in e["to"]] == pytest.approx(cues)


def test_flow_tags_cue_window_and_minimum() -> None:
    style = _style(flow_look="tags")
    assert cards.FLOW_TAGS_MIN_S == 1.5
    for fps in (24, 30):
        frames = math.ceil(1.5 * fps)
        with pytest.raises(ValidationError, match=f"1.5 s .*{frames} frames") as caught:
            cards.card_layer_plan("flow", {"nodes": ["A", "B"]}, style,
                                  frames=frames - 1, fps=fps)
        assert caught.value.parameter == "frames"
        duration = hyperframes.layer_duration_s(frames, fps)
        latest = duration - 0.1 - (5 / 30 + 0.3)
        cards.card_layer_plan("flow", {"nodes": ["A", "B"], "cues": [0, latest]},
                              style, frames=frames, fps=fps)
        for cues in ([0, latest + 0.001], [-0.001], [0.5, 0.4], [0, 0, 0]):
            with pytest.raises(ValidationError) as caught:
                cards.card_layer_plan("flow", {"nodes": ["A", "B"], "cues": cues},
                                      style, frames=frames, fps=fps)
            assert caught.value.parameter == "slots.cues"
            assert "use" in caught.value.detail


@pytest.mark.parametrize("look,paper,ink", [
    ("minimal", "#FFFFFF", "#222222"), ("dark", "#161616", "#FFFFFF"),
])
def test_flow_tags_card_colors(look: str, paper: str, ink: str) -> None:
    style = _style(flow_look="tags", style=look, paper_color=paper, ink_color=ink)
    plan = cards.card_layer_plan("flow", {"nodes": ["A", "B"]}, style)
    assert plan.variables["paper_color"] == paper and plan.variables["ink_color"] == ink
    rules = _css_rules((cards.template_dir("flow") / "template.html").read_text())
    assert rules[".flow-tag"]["background"] == "var(--card-paper)"
    assert rules[".flow-tag"]["color"] == "var(--card-ink)"
    assert rules[".tag-arrow"]["stroke"] == "var(--card-accent)"
    assert rules[".tag-arrow"]["stroke-width"] == "3"


@pytest.mark.parametrize("widths,expected", [
    ({"WORKFLOW": 300, "Edit": 90}, 60),
    ({"WORKFLOW": 335, "Edit": 90}, 56),
    ({"WORKFLOW": 90, "Edit": 347}, 54),
])
def test_flow_tags_fit_loaded_font_before_seek(
    widths: dict[str, int], expected: int, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["WORKFLOW", "Edit"]},
                                 _style(flow_look="tags", font="Zapfino", card_width_px=400))
    assert plan.variables["text_px"] == 60
    events = _timeline_events("flow", 2.499, tmp_path, defer_fonts=True,
                              font_widths_at_40=widths, **plan.variables)
    layouts = {event["target"]: event["to"] for event in events
               if event["kind"] == "tag-layout"}
    assert [tag["size"] for tag in layouts["fonts-wait"]] == [60, 60]
    assert [tag["size"] for tag in layouts["first-seek"]] == [expected, expected]
    budget = plan.variables["tag_max_width_px"] - 2 * 0.42 * plan.variables["text_px"]
    loaded = layouts["fonts-loaded"]
    assert any(tag["text_width"] > budget for tag in loaded) == (expected < 60)
    for tag in layouts["first-seek"]:
        assert tag["text_width"] <= budget
        assert tag["scroll_width"] <= plan.variables["tag_max_width_px"]
        assert (tag["scroll_width"] * 1.16 + 16
                <= plan.variables["panel_width_px"])
    phases = [event["target"] for event in events if event["kind"] == "tag-layout"]
    assert phases == ["fonts-wait", "fonts-loaded", "first-seek"]


@pytest.mark.parametrize("max_height", [None, 218])
def test_flow_tags_loaded_font_does_not_throw(
    max_height: int | None, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["WORKFLOW", "Edit"]},
                                 _style(flow_look="tags", font="Zapfino"), max_height_px=max_height)
    events = _timeline_events("flow", 2.499, tmp_path, defer_fonts=True,
                              font_widths_at_40={"WORKFLOW": 600, "Edit": 347}, **plan.variables)
    layout = next(event["to"] for event in events
                  if event["kind"] == "tag-layout" and event["target"] == "first-seek")
    sizes = [tag["size"] for tag in layout]
    assert 0 < sizes[0] <= plan.variables["text_px"] and sizes[0] == sizes[1]


@pytest.mark.parametrize("max_height", [None, 218])
def test_flow_tags_wide_loaded_font_fits_tag_and_layer(
    max_height: int | None, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["WORKFLOW", "Edit"]},
                                 _style(flow_look="tags", font="Zapfino"), max_height_px=max_height)
    events = _timeline_events("flow", 2.499, tmp_path, defer_fonts=True,
                              font_widths_at_40={"WORKFLOW": 444, "Edit": 90}, **plan.variables)
    layouts = {event["target"]: event["to"] for event in events
               if event["kind"] == "tag-layout"}
    loaded = layouts["fonts-loaded"]
    if max_height is None:
        assert loaded[0]["text_width"] == 665
        assert loaded[0]["client_width"] == 522
    for tag in layouts["first-seek"]:
        assert tag["scroll_width"] <= tag["client_width"]
        assert tag["text_width"] <= plan.variables["tag_max_width_px"] - 0.84 * tag["size"]
        assert tag["client_width"] * 1.16 + 16 <= plan.width
    sizes = [tag["size"] for tag in layouts["first-seek"]]
    assert 0 < sizes[0] < 52 and sizes[0] == sizes[1]


def test_flow_tags_headroom_reserves_browser_rounding(tmp_path: Path) -> None:
    label = "X" * 15
    plan = cards.card_layer_plan("flow", {"nodes": [label, "A"]},
                                 _style(flow_look="tags", font="Geist"), max_height_px=218)
    assert plan.variables["text_px"] == 52
    events = _timeline_events("flow", 2.499, tmp_path, defer_fonts=True,
                              font_widths_at_40={label: 396, "A": 25}, **plan.variables)
    layout = next(event["to"] for event in events
                  if event["kind"] == "tag-layout" and event["target"] == "first-seek")
    assert [tag["size"] for tag in layout] == [52, 52]
    assert layout[0]["text_width"] == 515
    budget = plan.variables["tag_max_width_px"] - 2 * math.ceil(0.42 * 52)
    assert budget >= math.ceil(text_width_em(label, "Geist", 600) * 52) + 2
    assert layout[0]["text_width"] <= budget
    assert plan.variables["tag_max_width_px"] * 1.16 + 16 <= plan.width


@pytest.mark.parametrize("nodes", [
    [], ["A"], ["A"] * 5, "A,B", ("A", "B"), None,
    ["", "B"], [" ", "B"], ["x" * 19, "B"], [True, "B"], [1, "B"], [None, "B"],
    ["A\n", "B"], ["A\x7f", "B"], ["A\u2028", "B"],
])
def test_flow_rejects_invalid_nodes(nodes: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("flow", {"nodes": nodes}, _style())
    assert caught.value.parameter == "slots.nodes"


@pytest.mark.parametrize("cues", [
    [0] * 5, [0] * 3, [-0.001], [math.inf], [-math.inf], [math.nan],
    [True], ["1"], [None], [0.6, 0.5], [2], "1", (1,), None,
])
def test_flow_rejects_invalid_cues(cues: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("flow", {"nodes": ["A", "B"], "cues": cues}, _style())
    assert caught.value.parameter == "slots.cues"


@pytest.mark.parametrize(("frames", "fps"), [(63, 30), (51, 24), (75, 30), (240, 24)])
def test_flow_cue_window_boundaries(frames: int, fps: int) -> None:
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    reserve = cards.FLOW_NODE_REVEAL_S + cards.FLOW_EXPLICIT_REST_S
    latest = out_start - reserve
    for cues in ([0, 0], [0, latest], [latest, latest]):
        plan = cards.card_layer_plan("flow", {"nodes": ["A", "B"], "cues": cues},
                                     _style(), frames=frames, fps=fps)
        assert json.loads(plan.variables["cues"]) == cues
    for cues in ([-0.001], [0, latest + 0.001], [out_start]):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("flow", {"nodes": ["A", "B"], "cues": cues},
                                  _style(), frames=frames, fps=fps)
        assert caught.value.parameter == "slots.cues"
        assert f"use seconds within 0..{latest:g}" in caught.value.detail
        assert "leave 0.95 s before the fade-out" in caught.value.detail
        assert f"fade-out starts at {out_start:g} s" in caught.value.detail
    with pytest.raises(ValidationError, match="one per node"):
        cards.card_layer_plan("flow", {"nodes": ["A", "B"], "cues": [0, 0, 0]},
                              _style(), frames=frames, fps=fps)


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize(("frames", "fps"), [(63, 30), (51, 24), (75, 30), (240, 24)])
def test_flow_default_rhythm(count: int, frames: int, fps: int) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["A"] * count},
                                 _style(), frames=frames, fps=fps)
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    first = 0.3 + 0.24
    latest = min(out_start - 0.65 - 0.6, duration * 0.6 - 0.65)
    gap = (latest - first) / (count - 1)
    cues = json.loads(plan.variables["cues"])
    assert cues == pytest.approx([first + i * gap for i in range(count)])
    assert cues[0] - cards.FLOW_ARROW_DRAW_S == pytest.approx(cards.FLOW_PANEL_IN_S)
    assert cues[-1] + cards.FLOW_BOX_DRAW_S + cards.FLOW_LABEL_IN_S == pytest.approx(
        min(out_start - cards.FLOW_REST_S, duration * 0.6)
    )
    assert all(a <= b for a, b in zip(cues, cues[1:], strict=False))


@pytest.mark.parametrize("count", [2, 3, 4])
def test_flow_default_nodes_complete_before_preview_with_rest(count: int, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["A"] * count}, _style())
    assert plan.frames == 75 and plan.fps == 30
    assert cards.FLOW_REST_S == 0.6 and cards.FLOW_EXPLICIT_REST_S == 0.3
    duration = hyperframes.layer_duration_s(plan.frames, plan.fps)
    preview_at = (plan.frames * 3 // 5) / plan.fps
    cues = json.loads(plan.variables["cues"])
    reveal_ends = [cue + cards.FLOW_NODE_REVEAL_S for cue in cues]
    assert all(end <= preview_at for end in reveal_ends)
    out_start = duration - min(0.3, duration * 0.15)
    assert out_start - reveal_ends[-1] >= cards.FLOW_REST_S

    events = _timeline_events("flow", duration, tmp_path, **plan.variables)
    labels = [e for e in events if e["kind"] == "fromTo" and e["target"] == "label"]
    actual_ends = [e["at"] + e["to"]["duration"] for e in labels]
    assert actual_ends == pytest.approx(reveal_ends)
    assert all(end <= preview_at for end in actual_ends)
    assert out_start - actual_ends[-1] >= cards.FLOW_REST_S


@pytest.mark.parametrize("given", [[0], [0.1, 0.4], [1.2], [1.2, 1.2]])
def test_flow_partial_cues_preserve_order(given: list[float]) -> None:
    slots = {"nodes": ["A", "B", "C"]}
    defaults = json.loads(cards.card_layer_plan("flow", slots, _style()).variables["cues"])
    plan = cards.card_layer_plan("flow", {**slots, "cues": given}, _style())
    cues = json.loads(plan.variables["cues"])
    assert cues[:len(given)] == given
    for i in range(len(given), len(cues)):
        assert cues[i] == max(cues[i - 1], defaults[i])
    assert all(a <= b for a, b in zip(cues, cues[1:], strict=False))
    out_start = hyperframes.layer_duration_s(plan.frames, plan.fps) - 0.3
    assert out_start - cues[-1] - cards.FLOW_NODE_REVEAL_S >= cards.FLOW_EXPLICIT_REST_S


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize(("fps", "frames"), [(30, 63), (24, 51), (60, 126)])
def test_flow_minimum_duration(count: int, fps: int, frames: int) -> None:
    slots = {"nodes": ["A"] * count}
    with pytest.raises(ValidationError, match="needs at least 2.1 s") as caught:
        cards.card_layer_plan("flow", slots, _style(), frames=frames - 1, fps=fps)
    assert caught.value.parameter == "frames"
    with pytest.raises(ValidationError, match="frames"):
        cards.card_layer_plan("flow", slots, _style(hold_s=1.5), fps=fps)
    plan = cards.card_layer_plan("flow", slots, _style(hold_s=frames / fps), fps=fps)
    assert plan.frames == frames
    assert json.loads(plan.variables["cues"])[0] == pytest.approx(0.54)
    duration = hyperframes.layer_duration_s(frames, fps)
    last_end = json.loads(plan.variables["cues"])[-1] + cards.FLOW_NODE_REVEAL_S
    assert last_end == pytest.approx(duration - 0.3 - cards.FLOW_REST_S)


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("width", [600, 960, 961, 1080])
def test_flow_geometry_and_largest_fitted_type(count: int, width: int) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["A"] * count},
                                 _style(card_width_px=width))
    v = plan.variables
    body_width = width - 2 * cards.CONTENT_PAD
    box_width = math.floor((body_width - (count - 1) * 72) / count)
    assert v["box_width_px"] == box_width
    assert v["arrow_gap_px"] == 72
    assert v["box_height_px"] == 120
    assert v["text_px"] == 40
    row_width = count * box_width + (count - 1) * v["arrow_gap_px"]
    assert 0 <= body_width - row_width < count
    assert v["text_px"] * text_width_em("A", "Inter", 600) + 2 * cards.FLOW_LABEL_PAD_X <= box_width
    assert v["box_height_px"] + 2 * cards.CONTENT_PAD <= v["panel_max_height_px"]


@pytest.mark.parametrize("count", [2, 3, 4])
def test_flow_fit_error_reports_per_node_capacity(count: int) -> None:
    width = 600
    box_width = math.floor((width - 48 - (count - 1) * 72) / count)
    glyph_em = text_width_em("x", "Inter", 600)
    capacity = math.floor((box_width - 36) / (glyph_em * 30))
    labels = ["x" * capacity] * count
    plan = cards.card_layer_plan("flow", {"nodes": labels}, _style(card_width_px=width))
    expected_size = max(size for size in range(30, 41)
                        if text_width_em(labels[0], "Inter", 600) * size <= box_width - 36)
    assert plan.variables["text_px"] == expected_size
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("flow", {"nodes": [*labels[:-1], "x" * (capacity + 1)]},
                              _style(card_width_px=width))
    assert caught.value.parameter == "slots.nodes"
    assert f"shorten to at most {capacity} characters to fit at 30 px" in caught.value.detail


def test_flow_horizontal_fit_boundary_and_narrow_panel() -> None:
    slots = {"nodes": ["x" * 8, "y"]}
    minimum_width = 48 + 72 + 2 * math.ceil(
        text_width_em(slots["nodes"][0], "Inter", 600) * 30 + 36
    )
    plan = cards.card_layer_plan("flow", slots, _style(card_width_px=minimum_width))
    assert plan.variables["text_px"] == 30
    with pytest.raises(ValidationError, match="at most 7 characters"):
        cards.card_layer_plan("flow", slots, _style(card_width_px=minimum_width - 1))
    with pytest.raises(ValidationError, match="at most 0 characters"):
        cards.card_layer_plan("flow", {"nodes": ["A"] * 4}, _style(card_width_px=200))


def test_flow_wide_labels_fit_or_report_node_capacity() -> None:
    labels = ["WWWWW", "Edit", "Ship", "Done"]
    try:
        plan = cards.card_layer_plan("flow", {"nodes": labels}, _style(card_width_px=960))
    except ValidationError as error:
        assert error.parameter == "slots.nodes"
        assert "at 30 px" in error.detail
    else:
        assert plan.variables["box_width_px"] - 2 * cards.FLOW_LABEL_PAD_X == 138
        assert text_width_em(labels[0], "Inter", 600) * plan.variables["text_px"] <= 138


def test_flow_ordinary_labels_keep_readable_sizes() -> None:
    labels = ["Record", "Edit", "Publish"]
    plan = cards.card_layer_plan("flow", {"nodes": labels}, _style(card_width_px=960))
    size = plan.variables["text_px"]
    assert size >= 30
    room = plan.variables["box_width_px"] - 2 * cards.FLOW_LABEL_PAD_X
    assert all(text_width_em(label, "Inter", 600) * size <= room for label in labels)


def test_flow_panel_height_fit() -> None:
    resolved = _style()
    internal = resolved.data["internal"]
    for panel_height, expected_size in ((130, 38), (120, 30)):
        internal["key_text_bottom_px"] = internal["key_text_top_px"] + 88 + panel_height
        plan = cards.card_layer_plan("flow", {"nodes": ["A", "B"]}, resolved)
        assert plan.variables["box_height_px"] == panel_height - 48
        assert plan.variables["text_px"] == expected_size
    internal["key_text_bottom_px"] = internal["key_text_top_px"] + 88 + 119
    with pytest.raises(ValidationError, match="raise panel height") as caught:
        cards.card_layer_plan("flow", {"nodes": ["A", "B"]}, resolved)
    assert caught.value.parameter == "slots.nodes"


def test_flow_style_and_geometry_match_adapter() -> None:
    source = (cards.template_dir("flow") / "template.html").read_text()
    rules = _css_rules(source)
    assert rules[".panel"]["padding"] == f"{cards.CONTENT_PAD}px"
    assert rules[".row"]["flex-wrap"] == "nowrap"
    assert rules[".node"]["width"] == "var(--card-box-width)"
    assert rules[".node"]["height"] == "var(--card-box-height)"
    assert rules[".connector"]["width"] == "var(--card-arrow-gap)"
    assert rules[".label"]["padding"] == f"{cards.FLOW_LABEL_PAD_X}px"
    assert cards.FLOW_LABEL_PAD_X == cards.FLOW_LABEL_PAD_Y
    assert rules[".label"]["line-height"] == str(cards.FLOW_LINE_HEIGHT)
    assert rules[".label"]["font-family"] == "var(--card-font)"
    assert rules[".label"]["font-weight"] == "600"
    assert rules[".label"]["white-space"] == "nowrap"
    assert rules[".node-stroke"]["stroke"] == "var(--card-ink)"
    assert rules[".arrow-curve, .arrow-head"]["stroke"] == "var(--card-accent)"
    assert f"const STROKE_INSET = {cards.FLOW_STROKE_INSET};" in source
    assert cards.FLOW_STROKE_INSET > 2 + float(rules[".node-stroke"]["stroke-width"]) / 2
    assert "Caveat" not in source
    assert not re.search(r"(?:@font-face|src\s*:\s*url\(|https?://)", source, re.IGNORECASE)
    assert not any(token in source for token in ("cqw", "cqh", "cqmin", "cqmax", "hwBoil"))


@pytest.mark.parametrize("cues", [[0.6, 1, 1.5], [0, 0, 0], [0, 0.1, 0.2]])
def test_flow_timeline_draws_box_then_label_and_incoming_arrow(
    cues: list[float], tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["A", "B", "C"], "cues": cues},
                                 _style(), frames=90)
    events = _timeline_events("flow", hyperframes.layer_duration_s(90), tmp_path, **plan.variables)
    traces = [e for e in events if e["kind"] == "fromTo" and e["target"] == "node-stroke"]
    labels = [e for e in events if e["kind"] == "fromTo" and e["target"] == "label"]
    curves = [e for e in events if e["kind"] == "fromTo" and e["target"] == "arrow-curve"]
    heads = [e for e in events if e["kind"] == "fromTo" and e["target"] == "arrow-head"]
    assert [e["at"] for e in traces] == cues
    assert [e["at"] for e in labels] == pytest.approx([cue + 0.45 for cue in cues])
    assert all(e["to"]["duration"] == 0.45 for e in traces)
    assert all(e["to"]["duration"] == 0.2 for e in labels)
    assert all(e["from"]["strokeDashoffset"] > 0 and e["to"]["strokeDashoffset"] == 0
               for e in traces + curves + heads)
    assert len(curves) == len(heads) == 2
    for cue, curve, head in zip(cues[1:], curves, heads, strict=True):
        assert curve["at"] == pytest.approx(max(0, cue - 0.24))
        assert curve["at"] + curve["to"]["duration"] == pytest.approx(head["at"])
        assert head["at"] + head["to"]["duration"] == pytest.approx(cue)
    hidden = [e for e in events if e["kind"] == "initial" and e["target"] in
              {"node-stroke", "arrow-curve", "arrow-head"}]
    assert len(hidden) == 7 and all(e["to"]["opacity"] == 0 for e in hidden)
    fade = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "panel"
                and e["to"]["opacity"] == 0)
    assert fade["at"] == pytest.approx(hyperframes.layer_duration_s(90) - 0.3)
    assert (labels[-1]["at"] + labels[-1]["to"]["duration"]
            + cards.FLOW_EXPLICIT_REST_S <= fade["at"])


@pytest.mark.parametrize("given", [[], [0.1], [0.8, 0.8]])
@pytest.mark.parametrize("frames", [63, 75, 90, 300])
def test_flow_template_default_rhythm_matches_adapter(
    given: list[float], frames: int, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan("flow", {"nodes": ["A", "B", "C"], "cues": given},
                                 _style(), frames=frames)
    variables = {**plan.variables, "cues": json.dumps(given)}
    events = _timeline_events("flow", hyperframes.layer_duration_s(plan.frames),
                              tmp_path, **variables)
    traces = [e for e in events if e["kind"] == "fromTo" and e["target"] == "node-stroke"]
    assert [e["at"] for e in traces] == pytest.approx(json.loads(plan.variables["cues"]))


def _transfer_slots(**updates: object) -> dict:
    return {"left_label": "Laptop", "right_label": "Desktop", "payload": "notes.txt", **updates}


def test_transfer_slot_contract() -> None:
    slots = cards.load_template("transfer")["slots"]
    for name, maximum in (("left_label", 20), ("right_label", 20), ("payload", 16)):
        assert slots[name] == {
            "type": "string", "required": True, "min_length": 1, "max_length": maximum,
        }
    for name, options, default in (
        ("left_device", ["laptop", "desktop"], "laptop"),
        ("right_device", ["laptop", "desktop"], "desktop"),
        ("payload_icon", ["doc", "code"], "doc"),
    ):
        assert slots[name] == {
            "type": "enum", "required": False, "options": options, "default": default,
        }
    assert slots["travel_at"] == {"type": "number", "required": False, "default": 0.6}
    plan = cards.card_layer_plan("transfer", _transfer_slots(), _style())
    assert (plan.variables["left_device"], plan.variables["right_device"]) == ("laptop", "desktop")
    assert plan.variables["payload_icon"] == "doc" and plan.variables["travel_at"] == 0.6
    assert plan.variables["travel_s"] == cards.TRANSFER_TRAVEL_S == 1.0
    assert plan.variables["land_s"] == cards.TRANSFER_LAND_S == 0.3
    assert cards.TRANSFER_MIN_S == 2.3


@pytest.mark.parametrize(("fps", "minimum_frames"), [(30, 69), (24, 56)])
def test_transfer_minimum_frame_window(fps: int, minimum_frames: int) -> None:
    for travel_at in (0, 0.6, 2):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan(
                "transfer", _transfer_slots(travel_at=travel_at), _style(),
                frames=minimum_frames - 1, fps=fps,
            )
        assert caught.value.parameter == "frames"
        assert f"needs at least 2.3 s ({minimum_frames} frames at {fps} fps)" in caught.value.detail
        assert "raise graphics.hold_s or pass more frames" in caught.value.detail
    plan = cards.card_layer_plan(
        "transfer", _transfer_slots(), _style(), frames=minimum_frames, fps=fps
    )
    assert plan.frames == minimum_frames and plan.fps == fps
    assert plan.variables["travel_at"] == 0.6


@pytest.mark.parametrize(("fps", "minimum_frames"), [(30, 69), (24, 56)])
def test_transfer_hold_obeys_minimum(fps: int, minimum_frames: int) -> None:
    with pytest.raises(
        ValidationError, match="raise graphics.hold_s or pass more frames"
    ) as caught:
        cards.card_layer_plan("transfer", _transfer_slots(), _style(hold_s=2), fps=fps)
    assert caught.value.parameter == "frames"
    plan = cards.card_layer_plan(
        "transfer", _transfer_slots(), _style(hold_s=minimum_frames / fps), fps=fps
    )
    assert plan.frames == minimum_frames and plan.variables["travel_at"] == 0.6


@pytest.mark.parametrize(("frames", "fps"), [(69, 30), (75, 30), (90, 24)])
def test_transfer_travel_window_boundaries(frames: int, fps: int) -> None:
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - cards.TRANSFER_TRAVEL_S - cards.TRANSFER_LAND_S
    for at in (0, latest):
        plan = cards.card_layer_plan(
            "transfer", _transfer_slots(travel_at=at), _style(), frames=frames, fps=fps
        )
        assert plan.variables["travel_at"] == at
    for at in (-0.001, latest + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan(
                "transfer", _transfer_slots(travel_at=at), _style(), frames=frames, fps=fps
            )
        assert caught.value.parameter == "slots.travel_at"
        assert f"within 0..{latest:g}" in caught.value.detail
        assert f"fade-out starts at {out_start:g} s" in caught.value.detail


@pytest.mark.parametrize("left_device", ["laptop", "desktop"])
@pytest.mark.parametrize("right_device", ["laptop", "desktop"])
@pytest.mark.parametrize("payload_icon", ["doc", "code"])
def test_transfer_device_and_icon_variables(
    left_device: str, right_device: str, payload_icon: str
) -> None:
    slots = _transfer_slots(
        left_device=left_device, right_device=right_device, payload_icon=payload_icon
    )
    plan = cards.card_layer_plan("transfer", slots, _style())
    assert all(plan.variables[name] == value for name, value in slots.items())
    for name in TEXT_SIZES["transfer"]:
        assert type(plan.variables[name]) is int and plan.variables[name] >= cards.MIN_TEXT_PX


@pytest.mark.parametrize(
    ("slot", "width", "capacity"),
    [("left_label", 960, 19), ("right_label", 960, 19), ("payload", 600, 8),
     ("payload", 800, 12), ("payload", 920, 15)],
)
def test_transfer_capacity_refusals(slot: str, width: int, capacity: int) -> None:
    style = _style(card_width_px=width)
    slots = _transfer_slots(left_label="Left", right_label="Right", payload="x")
    plan = cards.card_layer_plan("transfer", {**slots, slot: "x" * capacity}, style)
    assert plan.variables[slot] == "x" * capacity
    variable = "chip_px" if slot == "payload" else "label_px"
    assert type(plan.variables[variable]) is int and plan.variables[variable] >= cards.MIN_TEXT_PX
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("transfer", {**slots, slot: "x" * (capacity + 1)}, style)
    assert caught.value.parameter == f"slots.{slot}"
    assert f"at most {capacity} characters to fit at 30 px" in caught.value.detail


@pytest.mark.parametrize("preset", PRESETS)
def test_transfer_default_payload_capacity(preset: str) -> None:
    plan = cards.card_layer_plan("transfer", _transfer_slots(payload="x" * 16),
                                _preset(preset))
    v = plan.variables
    assert v["chip_px"] == cards.MIN_TEXT_PX == 30
    assert v["chip_max_width_px"] == 342
    assert all(v[name] == value for name, value in cards.TRANSFER_GEOMETRY.items())
    chrome = v["chip_icon_px"] + v["chip_gap_px"] + 2 * v["chip_pad_x_px"]
    assert 16 * cards.PROPORTIONAL_EM * v["chip_px"] + chrome <= v["chip_max_width_px"]


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize(
    ("template", "slots"),
    [
        (
            "terminal",
            {"command": "x" * 44, "output": ["x" * 48] * 3, "title": "x" * 24, "host": "node"},
        ),
        ("stat", {"end": -999999999, "start": 999999999, "suffix": "units", "kicker": "x" * 34}),
        ("checklist", {"items": ["x" * 40] * 4}),
        ("text_tag", {"text": "x" * 48, "subtext": "x" * 50, "variant": "bar"}),
        ("image_card", {"title": "x" * 38, "chrome": "browser"}),
        ("transfer", {"left_label": "x" * 19, "right_label": "x" * 19, "payload": "x" * 16}),
        ("image_compare", {"before_label": "x" * 8, "after_label": "x" * 8}),
    ],
)
def test_long_content_keeps_readable_type(
    preset: str,
    template: str,
    slots: dict,
    image: Path,
) -> None:
    if template == "image_card":
        slots = {**slots, "image": image}
    elif template == "image_compare":
        slots = {**slots, "before": image, "after": image}
    plan = cards.card_layer_plan(template, slots, _preset(preset))
    for name in TEXT_SIZES[template]:
        assert plan.variables[name] >= cards.MIN_TEXT_PX
    if template == "stat":
        assert plan.variables["value_px"] >= 2 * cards.MIN_TEXT_PX


def test_stat_kicker_capacity_includes_tracking() -> None:
    style = _style(card_width_px=960)
    plan = cards.card_layer_plan("stat", {"end": 1200, "kicker": "X" * 34}, style)
    assert plan.variables["kicker_px"] == 30
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("stat", {"end": 1200, "kicker": "X" * 35}, style)
    assert caught.value.parameter == "slots.kicker"
    assert "at most 34 characters" in caught.value.detail


def test_text_tag_variant_slot_and_look_contract() -> None:
    metadata = cards.load_template("text_tag")
    assert metadata["slots"]["variant"] == {
        "type": "enum", "required": False,
        "options": ["sign", "tag", "plain", "bar", "card", "block", "rule", "title"],
        "default": "sign",
    }
    look = next(v for v in metadata["variables"] if v["id"] == "look")
    assert look == {
        "id": "look", "type": "enum", "label": "Look", "default": "bar",
        "options": [{"value": name, "label": name.title()}
                    for name in ("bar", "card", "block", "pill", "underline", "rule", "title")],
    }
    assert "variant" not in {v["id"] for v in metadata["variables"]}
    for template in cards.CARD_TEMPLATES:
        declared = {v["id"] for v in cards.load_template(template)["variables"]}
        assert ("plain_ink_color" in declared) == (template == "text_tag")


@pytest.mark.parametrize(
    ("graphics_style", "look"),
    [("paper", "bar"), ("minimal", "bar"), ("dark", "card"), ("bright", "block")],
)
def test_text_tag_sign_uses_graphics_style(graphics_style: str, look: str) -> None:
    data = resolve_style_data({
        "schema_version": 1, "name": "Sign sample", "graphics": {"style": graphics_style},
    })
    style = ResolvedStyle(data["name"], "test", data)
    for slots in ({"text": "Label"}, {"text": "Label", "variant": "sign"}):
        plan = cards.card_layer_plan("text_tag", slots, style)
        assert plan.variables["look"] == look
        assert "variant" not in plan.variables


@pytest.mark.parametrize("graphics_style", ["paper", "minimal", "dark", "bright"])
@pytest.mark.parametrize(
    ("variant", "look"),
    [("tag", "pill"), ("plain", "underline"), ("bar", "bar"),
     ("card", "card"), ("block", "block"), ("rule", "rule"), ("title", "title")],
)
def test_text_tag_explicit_variants(graphics_style: str, variant: str, look: str) -> None:
    data = resolve_style_data({
        "schema_version": 1, "name": "Tag sample", "graphics": {"style": graphics_style},
    })
    plan = cards.card_layer_plan(
        "text_tag", {"text": "Label", "subtext": "Detail", "variant": variant},
        ResolvedStyle(data["name"], "test", data),
    )
    assert plan.variables["look"] == look
    for name, value in plan.variables.items():
        if name.endswith("_px"):
            assert type(value) is int
    assert cards.MIN_TEXT_PX <= plan.variables["text_px"] <= 64
    assert cards.MIN_TEXT_PX <= plan.variables["subtext_px"] <= 36


@pytest.mark.parametrize(
    ("paper", "ink", "expected"),
    [("#F4F3EE", "#222222", "#F4F3EE"), ("#111111", "#FFFFFF", "#FFFFFF"),
     ("#FF0000", "#00FF00", "#00FF00"), ("#FFFFFF", "#FFFFFF", "#FFFFFF"),
     ("#101010", "#111111", "#111111")],
)
def test_text_tag_plain_ink_uses_wcag_luminance(paper: str, ink: str, expected: str) -> None:
    plan = cards.card_layer_plan(
        "text_tag", {"text": "Label", "variant": "plain"},
        _style(paper_color=paper, ink_color=ink),
    )
    assert plan.variables["plain_ink_color"] == expected


@pytest.mark.parametrize(
    ("variant", "look", "text_capacity", "subtext_capacity"),
    [("bar", "bar", 30, 30), ("card", "card", 29, 29), ("block", "block", 23, 26),
     ("tag", "pill", 23, 23), ("plain", "underline", 24, 27)],
)
@pytest.mark.parametrize("slot", ["text", "subtext"])
def test_text_tag_look_capacity_refusals(
    variant: str, look: str, text_capacity: int, subtext_capacity: int, slot: str,
) -> None:
    capacity = text_capacity if slot == "text" else subtext_capacity
    slots = {"text": "x", "subtext": "", "variant": variant, slot: "x" * capacity}
    style = _style(card_width_px=600)
    plan = cards.card_layer_plan("text_tag", slots, style)
    assert plan.variables["look"] == look
    size = plan.variables["text_px" if slot == "text" else "subtext_px"]
    assert type(size) is int and size >= cards.MIN_TEXT_PX
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("text_tag", {**slots, slot: "x" * (capacity + 1)}, style)
    assert caught.value.parameter == f"slots.{slot}"
    assert caught.value.detail == (
        f"shorten to at most {capacity} characters to fit at 30 px in the {look} look"
    )


@pytest.mark.parametrize(("variant", "capacity"), [("block", 23), ("plain", 24)])
def test_text_tag_uppercase_expansion_fits(variant: str, capacity: int) -> None:
    text = "x" * (capacity - 1) + "ß"
    assert len(text) == capacity and len(text.upper()) == capacity + 1
    with pytest.raises(ValidationError, match=f"at most {capacity} characters") as caught:
        cards.card_layer_plan("text_tag", {"text": text, "variant": variant},
                              _style(card_width_px=600))
    assert caught.value.parameter == "slots.text"


@pytest.mark.parametrize(("variant", "slot", "weight", "tracking"), [
    ("rule", "text", 700, 0.01), ("rule", "subtext", 400, 0.04),
    ("title", "text", 600, 0), ("title", "subtext", 600, 0.18),
])
def test_text_tag_measured_capacity_refusals(
    variant: str, slot: str, weight: int, tracking: float,
) -> None:
    left, right, _, _ = cards.TAG_PADDING[variant]
    room = 600 - left - right - cards.TAG_WIDTH_RESERVE[variant]
    glyph_em = text_width_em("W", "Inter", weight, tracking_em=tracking)
    capacity = math.floor(room / (glyph_em * 30))
    slots = {"text": "Label", "subtext": "", "variant": variant, slot: "W" * capacity}
    style = _style(card_width_px=600)
    plan = cards.card_layer_plan("text_tag", slots, style)
    size = plan.variables["text_px" if slot == "text" else "subtext_px"]
    assert size >= 30
    assert text_width_em(slots[slot], "Inter", weight, tracking_em=tracking) * size <= room
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("text_tag", {**slots, slot: "W" * (capacity + 1)}, style)
    assert caught.value.parameter == f"slots.{slot}"
    assert caught.value.detail == (
        f"shorten to at most {capacity} characters to fit at 30 px in the {variant} look"
    )


def test_text_tag_rule_measures_uppercase_expansion() -> None:
    room = 600 - 26 - 32 - 8
    capacity = math.floor(room / (text_width_em("S", "Inter", 700, tracking_em=0.01) * 30))
    text = "s" * (capacity - 1) + "ß"
    with pytest.raises(ValidationError, match=f"at most {capacity} characters") as caught:
        cards.card_layer_plan("text_tag", {"text": text, "variant": "rule"},
                              _style(card_width_px=600))
    assert caught.value.parameter == "slots.text"


@pytest.mark.parametrize(("variant", "weight", "tracking"), [
    ("rule", 700, 0.01), ("title", 600, 0),
])
def test_text_tag_covered_cyrillic_fits_rule_and_title(
    variant: str, weight: int, tracking: float,
) -> None:
    text = "\u0416" * 7
    plan = cards.card_layer_plan(
        "text_tag", {"text": text, "variant": variant}, _style(font="Inter", card_width_px=600)
    )
    assert plan.variables["text_px"] >= 30
    left, right, _, _ = cards.TAG_PADDING[variant]
    room = 600 - left - right - cards.TAG_WIDTH_RESERVE[variant]
    width_em = text_width_em(text, "Inter", weight, tracking_em=tracking)
    assert width_em * plan.variables["text_px"] <= room


def test_text_tag_wide_title_reports_title_look() -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("text_tag", {"text": "W" * 29, "variant": "title"},
                              _style(card_width_px=600))
    assert caught.value.parameter == "slots.text"
    assert "at 30 px in the title look" in caught.value.detail


def test_text_tag_title_kicker_fits_uppercase_expansion_and_tracking() -> None:
    style = _style(card_width_px=600)
    room = 600 - 64
    capacity = math.floor(room / (text_width_em("S", "Inter", 600, tracking_em=0.18) * 30))
    slots = {"text": "x" * 29, "subtext": "s" * (capacity - 2) + "ß", "variant": "title"}
    plan = cards.card_layer_plan("text_tag", slots, style)
    assert plan.variables["text"] == slots["text"]
    assert plan.variables["subtext"] == slots["subtext"]
    assert plan.variables["text_px"] >= cards.MIN_TEXT_PX
    assert plan.variables["subtext_px"] == cards.MIN_TEXT_PX
    assert text_width_em(slots["text"], "Inter", 600) * plan.variables["text_px"] <= room
    assert text_width_em(slots["subtext"].upper(), "Inter", 600, tracking_em=0.18) * 30 <= room
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("text_tag", {**slots, "subtext": "s" * (capacity - 1) + "ß"}, style)
    assert caught.value.parameter == "slots.subtext"
    assert caught.value.detail == (
        f"shorten to at most {capacity} characters to fit at 30 px in the title look"
    )


@pytest.mark.parametrize(("variant", "subtext", "minimum_height"), [
    ("rule", "", 80), ("rule", "Detail", 135),
    ("title", "", 96), ("title", "Kicker", 157),
])
def test_text_tag_rule_and_title_panel_height_refusals(
    variant: str, subtext: str, minimum_height: int,
) -> None:
    style = _style(card_width_px=600)
    internal = style.data["internal"]
    internal["key_text_bottom_px"] = (
        internal["key_text_top_px"] + cards.PAD_TOP + cards.PAD_BOTTOM + minimum_height
    )
    slots = {"text": "Label", "subtext": subtext, "variant": variant}
    plan = cards.card_layer_plan("text_tag", slots, style)
    assert plan.variables["text_px"] == cards.MIN_TEXT_PX
    assert plan.variables["subtext_px"] >= cards.MIN_TEXT_PX
    internal["key_text_bottom_px"] -= 1
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("text_tag", slots, style)
    assert caught.value.parameter == "slots.text"
    assert caught.value.detail == f"shorten text or remove subtext to fit the {variant} look"


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("variant", ["bar", "card", "block", "tag", "plain", "rule", "title"])
@pytest.mark.parametrize("frames", [1, 75])
def test_text_tag_looks_prepare_with_safe_assets(
    preset: str, variant: str, frames: int, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan(
        "text_tag", {"text": "One clear idea", "subtext": "A useful detail", "variant": variant},
        _preset(preset), frames=frames,
    )
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert hyperframes.preflight(composition, frames) == hyperframes.layer_duration_s(frames)
    assert set(plan.variables) == {
        v["id"] for v in cards.load_template("text_tag")["variables"]
    }
    for name in TEXT_SIZES["text_tag"]:
        assert type(plan.variables[name]) is int and plan.variables[name] >= cards.MIN_TEXT_PX


@pytest.mark.parametrize(
    ("template", "name", "updates"),
    [
        ("terminal", "command", {"command": "x" * 61}),
        ("terminal", "command", {"command": ""}),
        ("terminal", "command", {"command": " "}),
        ("terminal", "command", {"command": 12}),
        ("terminal", "output", {"output": ["x"] * 4}),
        ("terminal", "output", {"output": ["x" * 61]}),
        ("terminal", "output", {"output": [""]}),
        ("terminal", "output", {"output": [3]}),
        ("terminal", "output", {"output": ("x",)}),
        ("terminal", "title", {"title": "x" * 41}),
        ("terminal", "host", {"host": "x" * 33}),
        ("terminal", "prompt", {"prompt": "x" * 4}),
        ("terminal", "cadence", {"cadence": "fast"}),
        ("terminal", "type_at", {"type_at": True}),
        ("terminal", "type_at", {"type_at": math.nan}),
        ("terminal", "type_at", {"type_at": -0.1}),
        ("terminal", "type_at", {"type_at": 2}),
        ("terminal", "cues", {"cues": [0.2]}),
        ("terminal", "cues", {"cues": [1.2, 1.1]}),
        ("terminal", "cues", {"cues": [3]}),
        ("terminal", "cues", {"cues": [1, 1, 1, 1]}),
        ("stat", "end", {"end": True}),
        ("stat", "end", {"end": 1.0}),
        ("stat", "end", {"end": 1000000000}),
        ("stat", "start", {"start": -1000000000}),
        ("stat", "prefix", {"prefix": "x" * 4}),
        ("stat", "suffix", {"suffix": "x" * 7}),
        ("stat", "kicker", {"kicker": "x" * 41}),
        ("checklist", "items", {"items": []}),
        ("checklist", "items", {"items": ["x"] * 5}),
        ("checklist", "items", {"items": ["x" * 41]}),
        ("checklist", "items", {"items": [""]}),
        ("checklist", "items", {"items": [True]}),
        ("checklist", "items", {"items": "x"}),
        ("checklist", "cues", {"cues": [1] * 5}),
        ("checklist", "cues", {"cues": [-0.1]}),
        ("checklist", "cues", {"cues": [math.inf]}),
        ("checklist", "cues", {"cues": [True]}),
        ("checklist", "cues", {"cues": ["1"]}),
        ("checklist", "cues", {"cues": [1.1, 1.0]}),
        ("checklist", "cues", {"cues": [3]}),
        ("checklist", "cues", {"cues": "1"}),
        ("text_tag", "text", {"text": "x" * 49}),
        ("text_tag", "text", {"text": ""}),
        ("text_tag", "text", {"text": False}),
        ("text_tag", "subtext", {"subtext": "x" * 61}),
        ("text_tag", "variant", {"variant": "pill"}),
        ("text_tag", "variant", {"variant": "underline"}),
        ("text_tag", "variant", {"variant": True}),
        ("text_tag", "variant", {"variant": ""}),
        ("image_card", "image", {"image": "missing.png"}),
        ("image_card", "image", {"image": "missing.svg"}),
        ("image_card", "image", {"image": 1}),
        ("image_card", "title", {"title": "x" * 49}),
        ("image_card", "chrome", {"chrome": "phone"}),
        ("transfer", "left_label", {"left_label": "x" * 21}),
        ("transfer", "left_label", {"left_label": ""}),
        ("transfer", "left_label", {"left_label": " "}),
        ("transfer", "left_label", {"left_label": 1}),
        ("transfer", "right_label", {"right_label": "x" * 21}),
        ("transfer", "right_label", {"right_label": ""}),
        ("transfer", "right_label", {"right_label": " "}),
        ("transfer", "right_label", {"right_label": False}),
        ("transfer", "payload", {"payload": "x" * 17}),
        ("transfer", "payload", {"payload": ""}),
        ("transfer", "payload", {"payload": " "}),
        ("transfer", "payload", {"payload": []}),
        ("transfer", "left_device", {"left_device": "phone"}),
        ("transfer", "left_device", {"left_device": 1}),
        ("transfer", "right_device", {"right_device": "phone"}),
        ("transfer", "right_device", {"right_device": None}),
        ("transfer", "payload_icon", {"payload_icon": "image"}),
        ("transfer", "payload_icon", {"payload_icon": False}),
        ("transfer", "travel_at", {"travel_at": True}),
        ("transfer", "travel_at", {"travel_at": "0.6"}),
        ("transfer", "travel_at", {"travel_at": math.nan}),
        ("transfer", "travel_at", {"travel_at": math.inf}),
        ("transfer", "travel_at", {"travel_at": -math.inf}),
        ("transfer", "travel_at", {"travel_at": -0.001}),
        ("transfer", "travel_at", {"travel_at": 2}),
    ],
)
def test_bad_slots_raise_named_validation(
    template: str,
    name: str,
    updates: dict,
    image: Path,
) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**_preview(template, image), **updates}, _style())
    assert caught.value.parameter == f"slots.{name}"
    assert caught.value.detail


@pytest.mark.parametrize("template", cards.CARD_TEMPLATES)
def test_required_unknown_and_control_characters(template: str, image: Path) -> None:
    slots = _preview(template, image)
    rules = cards.load_template(template)["slots"]
    for name, rule in rules.items():
        if rule["required"]:
            with pytest.raises(ValidationError) as caught:
                cards.card_layer_plan(
                    template, {k: v for k, v in slots.items() if k != name}, _style()
                )
            assert caught.value.parameter == f"slots.{name}"
        if rule["type"] in {"string", "string_list", "image", "video"}:
            for character in (
                "\x00", "\x1f", "\x7f", "\n", "\t", "\x80", "\x85", "\x9f", "\u2028", "\u2029",
            ):
                value = "x" + character
                if rule["type"] == "string_list":
                    value = [value]
                with pytest.raises(ValidationError) as caught:
                    cards.card_layer_plan(template, {**slots, name: value}, _style())
                assert caught.value.parameter == f"slots.{name}"
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**slots, "unknown": "x"}, _style())
    assert caught.value.parameter == "slots.unknown"


@pytest.mark.parametrize("template", PANEL_TEMPLATES)
def test_width_refusals_explain_what_to_shorten(template: str, image: Path) -> None:
    slots = _preview(template, image)
    updates = {
        "terminal": {"command": "x" * 25},
        "stat": {"end": 999999999},
        "checklist": {"items": ["x" * 20]},
        "text_tag": {"text": "x" * 20},
        "image_card": {"title": "x" * 15},
        "transfer": {"left_label": "x" * 20},
        "image_compare": {"before_label": "x" * 20},
        "steps": {"steps": ["x" * 20, "y"]},
        "flow": {"nodes": ["x" * 18, "y"]},
        "chart": {"title": "x" * 40},
        "quote": {"name": "x" * 30},
        "progress_stat": {"label": "x" * 30},
        "specs_list": {"title": "x" * 40},
        "callout_circle": {"label": "x" * 30},
        "name_tag": {"name": "x" * 30},
        "success_check": {"label": "x" * 30},
        "edit_timeline": {"title": "x" * 28},
        "storyboard": {"count": 7},
        "jiggle_delete": {"items": ["W" * 24, "A"]},
        "file_drop": {"name": "W" * 40},
        "video_card": {"label": "x" * 30},
        "prompt_bar": {"request": "x" * 30},
        "comment_cta": {"keyword": "x" * 30},
        "comment_drop": {"text": "x" * 30},
        "icon_strike": {"label": "x" * 30},
        "playing_grid": {"labels": ["W" * 24]},
        "reel_feed": {"caption": "W" * 80},
        "gravity_drop": {"drops": [{"text": "W" * 20, "at": 0.3}]},
    }
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**slots, **updates[template]}, _style(card_width_px=200))
    assert caught.value.parameter.startswith("slots.")
    if template == "file_drop":
        assert caught.value.detail == "set name to a shorter label"
        return
    assert ("lower count" if template == "storyboard" else "shorten") in caught.value.detail
    assert re.search(r"\d+", caught.value.detail)


def test_bar_capacity_cues_and_defaults(image: Path) -> None:
    with pytest.raises(ValidationError, match="slots.title"):
        cards.card_layer_plan(
            "terminal", {"command": "x", "title": "x" * 40}, _style(card_width_px=600)
        )
    with pytest.raises(ValidationError, match="slots.host"):
        cards.card_layer_plan("terminal", {"command": "x", "host": "x" * 32}, _style())
    for template, required in (
        ("terminal", {"command": "x"}),
        ("stat", {"end": 0}),
        ("checklist", {"items": ["x"]}),
        ("text_tag", {"text": "x"}),
        ("image_card", {"image": str(image)}),
        ("transfer", {"left_label": "Left", "right_label": "Right", "payload": "x"}),
        ("image_compare", {"before": image, "after": image}),
        ("steps", {"steps": ["First", "Next"]}),
    ):
        plan = cards.card_layer_plan(template, required, _style())
        assert set(plan.variables) == _default_variable_ids(template)
    terminal = cards.card_layer_plan("terminal", {"command": "x"}, _style())
    assert terminal.variables["prompt"] == "$" and terminal.variables["output"] == "[]"
    assert json.loads(terminal.variables["schedule"])["output_at"] == []
    for template, slots in (
        ("terminal", {"command": "x", "output": ["x"]}),
        ("checklist", {"items": ["x"]}),
    ):
        with pytest.raises(ValidationError, match="slots.cues"):
            cards.card_layer_plan(template, {**slots, "cues": [1, 1]}, _style())
    out_start = hyperframes.layer_duration_s(75) - 0.3
    plan = cards.card_layer_plan("terminal", {"command": "x", "type_at": out_start - 0.5},
                                 _style())
    assert json.loads(plan.variables["schedule"])["type_start"] == round(out_start - 0.5, 6)


@pytest.mark.parametrize(
    ("template", "slots", "reserve"),
    [
        ("checklist", {"items": ["x"]}, 0.3),
        ("terminal", {"command": "x", "output": ["x"]}, 0.12),
    ],
)
def test_cue_window_reserves_reveal_time(template: str, slots: dict, reserve: float) -> None:
    duration = hyperframes.layer_duration_s(75)
    out_start = duration - min(0.3, duration * 0.15)
    latest_cue = out_start - reserve
    plan = cards.card_layer_plan(
        template, {**slots, "cues": [latest_cue]}, _style(), frames=75
    )
    times = json.loads(plan.variables["schedule"])["output_at"] if template == "terminal" else (
        json.loads(plan.variables["cues"])
    )
    assert times == [round(latest_cue, 6) if template == "terminal" else latest_cue]
    for cue in (latest_cue + 0.001, out_start):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan(template, {**slots, "cues": [cue]}, _style(), frames=75)
        assert caught.value.parameter == "slots.cues"
        assert f"leave {reserve:g} s before the fade-out" in caught.value.detail
        assert f"starts at {out_start:g} s" in caught.value.detail


def _luminance(color: str) -> float:
    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return sum(
        (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) * w
        for c, w in zip(rgb, (0.2126, 0.7152, 0.0722), strict=True)
    )


def _contrast(a: str, b: str) -> float:
    low, high = sorted((_luminance(a), _luminance(b)))
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("template", PANEL_TEMPLATES)
def test_restyle_and_font_order(template: str, image: Path) -> None:
    warm, electric = (_preset(name) for name in PRESETS[:2])
    plans = [
        cards.card_layer_plan(template, _preview(template, image), style)
        for style in (warm, electric)
    ]
    for style, plan in zip((warm, electric), plans, strict=True):
        graphics = style.data["graphics"]
        for name in ("paper_color", "titlebar_color", "ink_color", "accent_color"):
            expected = (style.data["captions"]["text_color"]
                        if template == "gravity_drop" and name == "ink_color" else graphics[name])
            assert plan.variables[name] == expected
        assert plan.fonts == (cards.gravity_drop_fonts(style) if template == "gravity_drop"
                              else (graphics["font"], cards.MONO_FONT))
        assert plan.variables["font_family"] == fonts.font_stack(graphics["font"])
        assert plan.variables["mono_family"] == fonts.font_stack(cards.MONO_FONT)
    assert plans[0].variables["paper_color"] != plans[1].variables["paper_color"]
    assert plans[0].variables["accent_color"] != plans[1].variables["accent_color"]
    mono = cards.card_layer_plan(template, _preview(template, image), _style(font=cards.MONO_FONT))
    assert mono.fonts == (cards.gravity_drop_fonts(_style(font=cards.MONO_FONT))
                          if template == "gravity_drop" else (cards.MONO_FONT,))


@pytest.mark.parametrize(
    ("look", "radius", "shadow"),
    [
        ("paper", 30, "0 12px 22px rgb(18 12 6 / 0.28), 0 3px 8px rgb(18 12 6 / 0.16)"),
        (
            "minimal",
            16,
            "0 0 0 2px color-mix(in srgb, #222222 12%, transparent), 0 8px 18px rgb(0 0 0 / 0.16)",
        ),
        ("dark", 30, "0 12px 22px rgb(0 0 0 / 0.5), 0 0 0 2px rgb(255 255 255 / 0.06)"),
        ("bright", 36, "0 12px 22px rgb(0 0 0 / 0.26)"),
    ],
)
def test_look_table(look: str, radius: int, shadow: str) -> None:
    plan = cards.card_layer_plan("text_tag", {"text": "x"}, _style(style=look))
    assert plan.variables["card_radius_px"] == radius
    assert plan.variables["card_shadow"] == shadow


@pytest.mark.parametrize("look", cards._LOOKS)
def test_look_shadow_extents_fit_layer_padding(look: str) -> None:
    plan = cards.card_layer_plan("text_tag", {"text": "x"}, _style(style=look))
    shadow = plan.variables["card_shadow"]
    # Commas inside color functions do not separate shadow layers.
    layers = []
    depth = 0
    start = 0
    for index, character in enumerate(shadow):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            layers.append(shadow[start:index].strip())
            start = index + 1
    assert depth == 0, shadow
    layers.append(shadow[start:].strip())
    for layer in layers:
        match = re.match(r"((?:[+-]?\d+(?:\.\d+)?(?:px)?\s+){2,4})", layer)
        assert match, f"{look}: cannot parse shadow geometry in {layer}"
        lengths = [float(token.removesuffix("px")) for token in match[1].split()]
        offset_x, offset_y, blur, spread = lengths + [0.0] * (4 - len(lengths))
        assert 1.5 * blur - offset_y + spread <= cards.PAD_TOP, (look, layer)
        assert 1.5 * blur + abs(offset_x) + spread <= cards.PAD_X, (look, layer)
        assert 1.5 * blur + offset_y + spread <= cards.PAD_BOTTOM, (look, layer)


@pytest.mark.parametrize(
    ("paper", "ink", "bar", "accent"),
    [
        ("#FFFFFF", "#111111", "#EEEEEE", "#222222"),
        ("#111111", "#FFFFFF", "#222222", "#EEEEEE"),
        ("#777777", "#888888", "#999999", "#777777"),
        ("#DDDDDD", "#CCCCCC", "#FFFFFF", "#AAAAAA"),
    ],
)
def test_contrast_candidate_priority(paper: str, ink: str, bar: str, accent: str) -> None:
    plan = cards.card_layer_plan(
        "text_tag",
        {"text": "x"},
        _style(paper_color=paper, ink_color=ink, titlebar_color=bar, accent_color=accent),
    )
    expected_bar = next(
        (c for c in (ink, paper) if _contrast(c, bar) >= 3),
        "#111111" if _luminance(bar) > 0.4 else "#FFFFFF",
    )
    expected_accent = next(
        (c for c in (paper, ink, "#111111", "#FFFFFF") if _contrast(c, accent) >= 4.5),
        max(("#111111", "#FFFFFF"), key=lambda c: _contrast(c, accent)),
    )
    assert plan.variables["bar_ink_color"] == expected_bar
    assert plan.variables["accent_ink_color"] == expected_accent


@pytest.mark.parametrize("template", PANEL_TEMPLATES)
@pytest.mark.parametrize("text", ["<b>bold</b>", '"><img src=x onerror=alert(1)>'])
def test_slot_markup_stays_out_of_html(
    template: str,
    text: str,
    image: Path,
    tmp_path: Path,
) -> None:
    if template == "storyboard":
        with pytest.raises(ValidationError, match="slots.numbers"):
            cards.card_layer_plan(template, {"numbers": text}, _style())
        return
    name = {
        "terminal": "command",
        "stat": "kicker",
        "checklist": "items",
        "text_tag": "text",
        "image_card": "title",
        "transfer": "payload",
        "image_compare": "before_label",
        "steps": "steps",
        "flow": "nodes",
        "chart": "title",
        "quote": "quote",
        "progress_stat": "label",
        "specs_list": "title",
        "callout_circle": "label",
        "name_tag": "name",
        "success_check": "label",
        "edit_timeline": "title",
        "jiggle_delete": "items",
        "file_drop": "name",
        "video_card": "label",
        "prompt_bar": "request",
        "comment_cta": "keyword",
        "comment_drop": "text",
        "icon_strike": "label",
        "playing_grid": "labels",
        "reel_feed": "caption_lines",
        "gravity_drop": "drops",
    }[template]
    if template == "transfer" and len(text) > 16:
        text = '"><b>x</b>'
    elif template == "image_compare" and len(text) > 9:
        text = "<b>x</b>"
    elif template in {"steps", "jiggle_delete"} and len(text) > 24:
        text = "<b>x</b>"
    elif template == "flow" and len(text) > 18:
        text = "<b>x</b>"
    elif template == "callout_circle" and len(text) > 18:
        text = '"><img onerror=1>'
    if template == "edit_timeline" or (template == "gravity_drop" and len(text) > 20):
        text = "<b>x</b>"
    if template == "playing_grid" and len(text) > 24:
        text = "<img src=x onerror=x()>"
    value = ([text, "Safe"] if name in {"steps", "nodes"} else
             [text] if name in {"items", "labels", "caption_lines"} else text)
    if template == "jiggle_delete":
        value = [text, "Safe"]
    if name == "drops":
        value = [{"text": text, "at": 0.3}]
    slots = _preview(template, image)
    slots["caption" if template == "reel_feed" else name] = (
        text if template == "reel_feed" else value)
    if template == "playing_grid":
        slots["clips"] = slots["clips"][:2]
    plan = cards.card_layer_plan(template, slots, _style())
    if template == "reel_feed":
        value = json.loads(plan.variables["caption_lines"])
        assert " ".join(value) == text
    actual = (json.loads(plan.variables[name]) if name in {"items", "steps", "nodes", "labels",
                                                        "caption_lines", "drops"}
              else plan.variables[name])
    if template == "jiggle_delete":
        actual = [item["label"] for item in actual]
    if name == "drops":
        actual, value = actual[0]["text"], text
    assert actual == value
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert text not in (composition / "index.html").read_text()


def test_image_collage_slot_contract() -> None:
    slots = cards.load_template("image_card")["slots"]
    assert slots["reveal"] == {
        "type": "enum", "required": False, "options": ["fade", "collage"], "default": "fade",
    }
    assert {k: v for k, v in slots["reveal_at"].items() if k != "description"} == {
        "type": "number", "required": False, "default": 0.3,
    }
    assert {k: v for k, v in slots["panels"].items() if k != "description"} == {
        "type": "integer", "required": False, "minimum": 4, "maximum": 8, "default": 6,
    }


@pytest.mark.parametrize("preset", PRESETS)
def test_image_default_plan_keeps_fade_variables(preset: str, image: Path) -> None:
    resolved = _preset(preset)
    plan = cards.card_layer_plan("image_card", {"image": image}, resolved)
    width = resolved.data["graphics"]["card_width_px"]
    assert plan.variables == {
        **cards._tokens(resolved.data["graphics"], 272),
        "title": "", "chrome": "window", "image_src": "image.png", "bar_px": 30,
        "image_max_width_px": width - 4, "image_max_height_px": 208,
        "reveal": "fade", "reveal_at": 0.3, "panels": 6,
        "collage_span_s": cards.COLLAGE_SPAN_S, "collage_exit_s": cards.COLLAGE_EXIT_S,
    }
    assert cards.card_layer_plan(
        "image_card", {"image": image, "reveal": "fade"}, resolved
    ) == plan


@pytest.mark.parametrize(("slot", "value"), [
    ("reveal", "paper"), ("reveal", ""), ("reveal", True), ("reveal", None),
    ("reveal_at", True), ("reveal_at", "0.3"), ("reveal_at", None),
    ("reveal_at", math.nan), ("reveal_at", math.inf), ("reveal_at", -math.inf),
    ("panels", 3), ("panels", 9), ("panels", 6.0), ("panels", True),
    ("panels", "6"), ("panels", None),
])
def test_image_collage_rejects_invalid_slots(slot: str, value: object, image: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(
            "image_card", {"image": image, "reveal": "collage", slot: value}, _style()
        )
    assert caught.value.parameter == f"slots.{slot}"


@pytest.mark.parametrize("panels", range(4, 9))
@pytest.mark.parametrize(("frames", "fps"), [(66, 30), (53, 24), (75, 30), (240, 24)])
def test_image_collage_reveal_window_boundaries(
    panels: int, frames: int, fps: int, image: Path,
) -> None:
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - cards.COLLAGE_SPAN_S - cards.COLLAGE_REST_S
    slots = {"image": image, "reveal": "collage", "panels": panels}
    for at in (0, 0.3, latest):
        plan = cards.card_layer_plan(
            "image_card", {**slots, "reveal_at": at}, _style(), frames=frames, fps=fps
        )
        assert plan.variables["reveal_at"] == at and plan.variables["panels"] == panels
    for at in (-0.001, latest + 0.001, out_start):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan(
                "image_card", {**slots, "reveal_at": at}, _style(), frames=frames, fps=fps
            )
        assert caught.value.parameter == "slots.reveal_at"
        assert f"within 0..{latest:g}" in caught.value.detail
        assert "1.25 s for the lift and 0.3 s of rest" in caught.value.detail
        assert f"fade-out starts at {out_start:g} s" in caught.value.detail


@pytest.mark.parametrize(("fps", "minimum_frames"), [(30, 66), (24, 53)])
def test_image_collage_minimum_duration(fps: int, minimum_frames: int, image: Path) -> None:
    slots = {"image": image, "reveal": "collage"}
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("image_card", slots, _style(),
                              frames=minimum_frames - 1, fps=fps)
    assert caught.value.parameter == "frames"
    assert f"needs at least 2.2 s ({minimum_frames} frames at {fps} fps)" in caught.value.detail
    assert "raise graphics.hold_s or pass more frames" in caught.value.detail
    with pytest.raises(ValidationError, match="needs at least 2.2 s"):
        cards.card_layer_plan("image_card", slots, _style(hold_s=1), fps=fps)
    plan = cards.card_layer_plan("image_card", slots, _style(hold_s=2.2), fps=fps)
    assert plan.frames == minimum_frames


@pytest.mark.parametrize("at", [-1, 100])
def test_image_fade_ignores_collage_timing(at: int, image: Path, tmp_path: Path) -> None:
    plan = cards.card_layer_plan(
        "image_card", {"image": image, "reveal_at": at, "panels": 8}, _style(), frames=1
    )
    assert plan.variables["reveal"] == "fade" and plan.variables["reveal_at"] == at
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert hyperframes.preflight(composition, 1) == hyperframes.layer_duration_s(1)


def test_image_collage_static_determinism_contract() -> None:
    source = (cards.template_dir("image_card") / "template.html").read_text()
    for banned in ("Math.random", "Date", "performance.now", "requestAnimationFrame", "http",
                   "cqmin", "cqw", "cqh", "@font-face"):
        assert banned not in source
    assert "let state = 0x2c6b8f19;" in source
    assert "state = (Math.imul(1664525, state) + 1013904223) >>> 0;" in source
    assert 'seed="41"' in source and 'seed="7"' in source
    rules = _css_rules(source)
    assert rules[".screen.collage"]["overflow"] == "hidden"
    assert "var(--card-lift)" in rules[".collage-scrap"]["filter"]
    assert "var(--card-ink)" in rules[".collage-scrap"]["filter"]
    assert rules[".collage-fill"]["background"] == "var(--card-scrap-tone)"
    assert rules[".collage-grain"]["background"] == "var(--card-ink)"


def test_image_copy_is_local_identical_and_source_untouched(image: Path, tmp_path: Path) -> None:
    before = image.read_bytes()
    stat = image.stat()
    plan = cards.card_layer_plan("image_card", {"image": image}, _style())
    assert plan.assets == (("image.png", image.resolve()),)
    assert plan.variables["image_src"] == "image.png"
    assert plan.variables["image_max_width_px"] == 956
    assert plan.variables["image_max_height_px"] == 208
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert not image.is_relative_to(composition)
    assert (composition / "image.png").read_bytes() == before
    assert image.read_bytes() == before and image.stat().st_mtime_ns == stat.st_mtime_ns
    assert str(image) not in (composition / "index.html").read_text()
    with pytest.raises(ValidationError, match="slots.image"):
        cards.card_layer_plan("image_card", {"image": image.parent}, _style())


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("chrome", ["window", "browser"])
def test_image_frame_fits_panel_maxima(preset: str, chrome: str, image: Path) -> None:
    plan = cards.card_layer_plan(
        "image_card", {"image": image, "chrome": chrome}, _preset(preset)
    )
    source = (cards.template_dir("image_card") / "template.html").read_text()
    rules = _css_rules(source)
    assert rules["*"]["box-sizing"] == "border-box"
    # The bar's declared height includes its padding and bottom border.
    bar_height = int(rules[".bar"]["height"].removesuffix("px"))
    frame_border = 2
    assert (
        plan.variables["image_max_width_px"] + 2 * frame_border
        <= plan.variables["panel_width_px"]
    )
    assert (
        bar_height + plan.variables["image_max_height_px"] + 2 * frame_border
        <= plan.variables["panel_max_height_px"]
    )


@pytest.mark.parametrize("suffix", [".png", ".JPG", ".jpeg", ".WEBP"])
def test_image_suffix_normalization(suffix: str, image: Path, tmp_path: Path) -> None:
    source = tmp_path / ("asset" + suffix)
    source.write_bytes(image.read_bytes())
    plan = cards.card_layer_plan("image_card", {"image": source}, _style())
    assert plan.variables["image_src"] == "image" + suffix.lower()
    assert plan.assets == (("image" + suffix.lower(), source.resolve()),)


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
@pytest.mark.parametrize("damage", ["random", "truncated", "empty"])
def test_image_slots_reject_corrupt_files(
    template: str, name: str, damage: str, image: Path, tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.png"
    data = image.read_bytes()
    corrupt.write_bytes({
        "random": os.urandom(128), "truncated": data[:len(data) // 2], "empty": b"",
    }[damage])
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**_preview(template, image), name: corrupt}, _style())
    assert caught.value.parameter == f"slots.{name}"
    assert caught.value.detail.startswith("choose a readable image")


def _patterned_image(tmp_path: Path, suffix: str) -> Path:
    source = tmp_path / ("pattern" + suffix)
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=640x400:rate=1",
        "-frames:v", "1", "-threads", "1", str(source),
    ], timeout_s=30)
    return source


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
def test_image_slots_reject_truncated_jpeg(
    template: str, name: str, image: Path, tmp_path: Path,
) -> None:
    data = _patterned_image(tmp_path, ".jpg").read_bytes()
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(data[:len(data) // 2])
    info = ffmpeg.probe(truncated, count_frames=False)
    assert info.has_video and (info.width, info.height) == (640, 400)
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**_preview(template, image), name: truncated}, _style())
    assert caught.value.parameter == f"slots.{name}"
    assert caught.value.detail.startswith("choose a readable image")


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
def test_image_slots_reject_missing_paths(
    template: str, name: str, image: Path, tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jpg"
    assert not missing.exists()
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**_preview(template, image), name: missing}, _style())
    assert caught.value.parameter == f"slots.{name}"
    assert caught.value.detail == "choose an existing, readable image file"


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
@pytest.mark.parametrize("warning", ["malformed_exif", "missing_eoi", "ancillary_chunk"])
def test_image_slots_accept_metadata_and_container_warnings(
    template: str, name: str, warning: str, image: Path, tmp_path: Path,
) -> None:
    suffix = ".png" if warning == "ancillary_chunk" else ".jpg"
    source = _patterned_image(tmp_path, suffix)
    data = source.read_bytes()
    if warning == "malformed_exif":
        assert data.startswith(b"\xff\xd8")
        exif = b"Exif\0\0MM\0*\0\0\0\x08\xff\xff\xff"
        segment = b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
        # Observed stderr, despite a successful decode:
        # not enough bytes remaining in EXIF buffer. entries: 65535
        # error decoding EXIF data: Invalid data found when processing input
        data = data[:2] + segment + data[2:]
    elif warning == "missing_eoi":
        assert data.endswith(b"\xff\xd9")
        data = data[:-2]
    else:
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        # The lowercase first letter marks this unknown chunk as ancillary.
        chunk = b"yaPt" + b"ignored metadata"
        segment = (
            (len(chunk) - 4).to_bytes(4, "big") + chunk
            + zlib.crc32(chunk).to_bytes(4, "big")
        )
        ihdr_end = 8 + 12 + int.from_bytes(data[8:12], "big")
        data = data[:ihdr_end] + segment + data[ihdr_end:]
    source.write_bytes(data)
    plan = cards.card_layer_plan(template, {**_preview(template, image), name: source}, _style())
    filename = name + suffix
    assert (filename, source.resolve()) in plan.assets
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert (composition / filename).read_bytes() == data
    assert source.read_bytes() == data


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
def test_image_slots_decode_with_error_detection(
    template: str, name: str, image: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ffmpeg.run
    decoded = []

    def record(command: list[str], *, timeout_s: float):
        if command[0] == "ffmpeg":
            decoded.append(command)
        return run(command, timeout_s=timeout_s)

    monkeypatch.setattr(ffmpeg, "run", record)
    cards.card_layer_plan(template, {**_preview(template, image), name: image}, _style())
    assert len(decoded) == (2 if template == "image_compare" else 1)
    for command in decoded:
        assert command[command.index("-err_detect") + 1] == "explode"
        assert "-xerror" in command


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
@pytest.mark.parametrize("suffix", [".png", ".jpg", ".webp"])
def test_image_slots_accept_decodable_formats(
    template: str, name: str, suffix: str, image: Path, tmp_path: Path,
) -> None:
    source = tmp_path / ("valid" + suffix)
    generated = source
    codec = []
    if suffix == ".webp":
        encoders = ffmpeg.run(["ffmpeg", "-encoders"], timeout_s=10).stdout
        encoder = next((name for name in ("libwebp", "webp", "libwebp_anim")
                        if re.search(rf"^\s*V\S*\s+{name}\s", encoders, re.MULTILINE)), None)
        if encoder:
            codec = ["-c:v", encoder]
        else:
            # Some local builds decode this format but omit its encoder.
            if shutil.which("cwebp") is None:
                pytest.skip("no WebP encoder in ffmpeg and no cwebp")
            generated = source.with_suffix(".png")
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-f", "lavfi", "-i", "color=c=blue:s=640x400",
        "-frames:v", "1", "-threads", "1", *codec, str(generated),
    ], timeout_s=30)
    if generated != source:
        subprocess.run(
            ["cwebp", "-quiet", str(generated), "-o", str(source)],
            stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=30,
        )
    plan = cards.card_layer_plan(template, {**_preview(template, image), name: source}, _style())
    assert (name + suffix, source.resolve()) in plan.assets


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
@pytest.mark.parametrize("phase", ["decode", "probe"])
@pytest.mark.parametrize("failure", ["input", "timeout", "exit", "exec", "os"])
def test_image_slots_name_decode_and_probe_failures(
    template: str, name: str, phase: str, failure: str, image: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = _preview(template, image)
    # Validate only the target slot so a failure cannot be attributed to another image.
    rules = {name: cards.load_template(template)["slots"][name]}
    tool = "ffmpeg" if phase == "decode" else "ffprobe"
    error = {
        "input": InputError(str(image), "unreadable"),
        "timeout": ProcessingError("decode", -1, "timed out", code="ffmpeg_timeout"),
        "exit": ProcessingError("decode", 1, "decode failed"),
        "exec": ProcessingError(tool, -1, "could not launch", code=f"{tool}_exec_failed"),
        "os": OSError("unreadable"),
    }[failure]

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(ffmpeg, "run" if phase == "decode" else "probe", fail)
    if failure in {"exec", "os"}:
        with pytest.raises(type(error)) as caught:
            cards._slots({name: slots[name]}, rules)
        assert caught.value is error
        return
    with pytest.raises(ValidationError) as caught:
        cards._slots({name: slots[name]}, rules)
    assert caught.value.parameter == f"slots.{name}"
    assert caught.value.detail.startswith("choose a readable image")
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize("failure", ["missing", "unsupported"])
def test_image_slots_propagate_tool_dependency_errors(
    template: str, name: str, tool: str, failure: str, image: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = (DependencyError.missing_tool(tool) if failure == "missing"
             else DependencyError.ffmpeg_too_old(tool, "4.4.2"))
    find_tool = ffmpeg.find_tool

    def unavailable(requested: str) -> str:
        if requested == tool:
            raise error
        return find_tool(requested)

    monkeypatch.setattr(ffmpeg, "find_tool", unavailable)
    with pytest.raises(DependencyError) as caught:
        cards.card_layer_plan(template, {**_preview(template, image), name: image}, _style())
    assert caught.value is error
    assert caught.value.code == error.code
    assert caught.value.suggested_action == error.suggested_action


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
def test_image_slots_accept_stderr_on_success(
    template: str, name: str, image: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ffmpeg.run
    decoded = []

    def warn(command: list[str], *, timeout_s: float):
        if command[0] == "ffmpeg":
            decoded.append(Path(command[command.index("-i") + 1]))
            return subprocess.CompletedProcess(command, 0, "", "harmless metadata warning")
        return run(command, timeout_s=timeout_s)

    monkeypatch.setattr(ffmpeg, "run", warn)
    plan = cards.card_layer_plan(template, {**_preview(template, image), name: image}, _style())
    assert (name + ".png", image.resolve()) in plan.assets
    assert decoded == [image.resolve()] * (2 if template == "image_compare" else 1)


@pytest.mark.parametrize(("template", "name"), [
    ("image_compare", "before"), ("image_compare", "after"), ("image_card", "image"),
])
@pytest.mark.parametrize("updates", [
    {"has_video": False}, {"width": 0}, {"height": 0}, {"width": None}, {"height": -1},
])
def test_image_slots_require_video_and_positive_dimensions(
    template: str, name: str, updates: dict, image: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = replace(ffmpeg.probe(image, count_frames=False), **updates)
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: info)
    rules = {name: cards.load_template(template)["slots"][name]}
    with pytest.raises(ValidationError) as caught:
        cards._slots({name: image}, rules)
    assert caught.value.parameter == f"slots.{name}"
    assert caught.value.detail.startswith("choose a readable image")


def test_compare_reuses_slot_image_probes(
    image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    after = tmp_path / "after.png"
    after.write_bytes(image.read_bytes())
    probe = ffmpeg.probe
    run = ffmpeg.run
    probed = []
    decoded = []

    def inspect_probe(source: Path, *, count_frames: bool) -> ffmpeg.MediaInfo:
        assert count_frames is False
        probed.append(source)
        return probe(source, count_frames=count_frames)

    def inspect_run(command: list[str], *, timeout_s: float):
        if command[0] == "ffmpeg":
            assert command[:8] == [
                "ffmpeg", "-nostdin", "-v", "error", "-err_detect", "explode", "-xerror", "-i",
            ]
            assert command[9:] == ["-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"]
            assert timeout_s <= 10
            decoded.append(Path(command[8]))
        return run(command, timeout_s=timeout_s)

    monkeypatch.setattr(ffmpeg, "probe", inspect_probe)
    monkeypatch.setattr(ffmpeg, "run", inspect_run)
    cards.card_layer_plan("image_compare", {"before": image, "after": after}, _style())
    assert probed == decoded == [image.resolve(), after.resolve()]


def test_compare_slot_contract_and_defaults(image: Path) -> None:
    slots = cards.load_template("image_compare")["slots"]
    assert set(slots) == {"before", "after", "before_label", "after_label", "split", "wipe_at"}
    for name in ("before", "after"):
        assert {k: v for k, v in slots[name].items() if k != "description"} == {
            "type": "image", "required": True, "options": [".png", ".jpg", ".jpeg", ".webp"],
        }
    for name, default in (("before_label", "Before"), ("after_label", "After")):
        assert slots[name] == {
            "type": "string", "required": False, "min_length": 0, "max_length": 20,
            "default": default,
        }
    assert slots["split"] == {
        "type": "integer", "required": False, "minimum": 10, "maximum": 90, "default": 50,
    }
    assert slots["wipe_at"] == {"type": "number", "required": False, "default": 0.4}
    plan = cards.card_layer_plan("image_compare", {"before": image, "after": image}, _style())
    assert plan.variables["before_label"] == "Before" and plan.variables["after_label"] == "After"
    assert plan.variables["split"] == 50 and plan.variables["wipe_at"] == 0.4
    assert cards.CARD_TEMPLATES[5:9] == ("transfer", "image_compare", "steps", "flow")
    assert cards.COMPARE_MIN_S == 2.2
    assert cards.COMPARE_WIPE_S == 1.05 and cards.COMPARE_REST_S == 0.3


@pytest.mark.parametrize(("fps", "minimum_frames"), [(30, 66), (24, 53)])
def test_compare_minimum_frame_window(fps: int, minimum_frames: int, image: Path) -> None:
    slots = {"before": image, "after": image}
    for wipe_at in (0, 0.4, 2):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("image_compare", {**slots, "wipe_at": wipe_at}, _style(),
                                  frames=minimum_frames - 1, fps=fps)
        assert caught.value.parameter == "frames"
        assert f"needs at least 2.2 s ({minimum_frames} frames at {fps} fps)" in caught.value.detail
    plan = cards.card_layer_plan("image_compare", slots, _style(), frames=minimum_frames, fps=fps)
    assert plan.frames == minimum_frames and plan.variables["wipe_at"] == 0.4
    with pytest.raises(ValidationError, match="raise graphics.hold_s or pass more frames"):
        cards.card_layer_plan("image_compare", slots, _style(hold_s=2), fps=fps)
    default = cards.card_layer_plan("image_compare", slots,
                                   _style(hold_s=minimum_frames / fps), fps=fps)
    assert default.frames == minimum_frames and default.variables["wipe_at"] == 0.4


@pytest.mark.parametrize(("frames", "fps"), [(66, 30), (75, 30), (90, 24)])
def test_compare_wipe_window_boundaries(frames: int, fps: int, image: Path) -> None:
    slots = {"before": image, "after": image}
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - cards.COMPARE_WIPE_S - cards.COMPARE_REST_S
    for at in (0, latest):
        plan = cards.card_layer_plan("image_compare", {**slots, "wipe_at": at}, _style(),
                                     frames=frames, fps=fps)
        assert plan.variables["wipe_at"] == at
    for at in (-0.001, latest + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("image_compare", {**slots, "wipe_at": at}, _style(),
                                  frames=frames, fps=fps)
        assert caught.value.parameter == "slots.wipe_at"
        assert f"within 0..{latest:g}" in caught.value.detail
        assert f"fade-out starts at {out_start:g} s" in caught.value.detail


@pytest.mark.parametrize("name", ["before", "after"])
@pytest.mark.parametrize("value", [True, 3, None, "missing.png", "missing.svg", ""])
def test_compare_image_slot_validation(name: str, value: object, image: Path) -> None:
    slots = {"before": image, "after": image, name: value}
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("image_compare", slots, _style())
    assert caught.value.parameter == f"slots.{name}"


@pytest.mark.parametrize(
    ("name", "value"),
    [("before_label", "x" * 21), ("after_label", "x" * 21),
     ("before_label", 1), ("after_label", False),
     ("split", 9), ("split", 91), ("split", 50.0), ("split", True), ("split", "50"),
     ("wipe_at", True), ("wipe_at", "0.4"), ("wipe_at", math.nan),
     ("wipe_at", math.inf), ("wipe_at", -math.inf)],
)
def test_compare_slot_limits_and_types(name: str, value: object, image: Path) -> None:
    slots = {"before": image, "after": image, name: value}
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("image_compare", slots, _style())
    assert caught.value.parameter == f"slots.{name}"


@pytest.mark.parametrize("split", [10, 50, 90])
def test_compare_split_and_empty_labels(split: int, image: Path) -> None:
    plan = cards.card_layer_plan("image_compare", {
        "before": image, "after": image, "split": split, "before_label": "", "after_label": "",
    }, _style())
    assert plan.variables["split"] == split
    assert plan.variables["before_label"] == plan.variables["after_label"] == ""
    assert plan.variables["chip_px"] >= cards.MIN_TEXT_PX


@pytest.mark.parametrize("split", [10, 50, 90])
def test_compare_empty_labels_need_no_frame_room(split: int, tmp_path: Path) -> None:
    image = tmp_path / "thin.png"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-f", "lavfi", "-i", "color=c=red:s=2x2048",
        "-frames:v", "1", str(image),
    ], timeout_s=30)
    plan = cards.card_layer_plan("image_compare", {
        "before": image, "after": image, "split": split, "before_label": "", "after_label": "",
    }, _style())
    assert plan.variables["before_chip_max_width_px"] == 0
    assert plan.variables["after_chip_max_width_px"] == 0
    assert plan.variables["after_chip_reveal_progress"] == 0
    assert plan.variables["chip_px"] == 36


@pytest.mark.parametrize("name", ["before_label", "after_label"])
@pytest.mark.parametrize(("width", "capacity"), [(200, 1), (300, 4), (960, 8)])
def test_compare_label_capacity_refusals(
    name: str, width: int, capacity: int, image: Path
) -> None:
    slots = {"before": image, "after": image, "before_label": "", "after_label": ""}
    style = _style(card_width_px=width)
    plan = cards.card_layer_plan("image_compare", {**slots, name: "x" * capacity}, style)
    assert plan.variables[name] == "x" * capacity
    assert plan.variables["chip_px"] >= cards.MIN_TEXT_PX
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("image_compare", {**slots, name: "x" * (capacity + 1)}, style)
    assert caught.value.parameter == f"slots.{name}"
    assert f"at most {capacity} characters to fit at 30 px" in caught.value.detail
    assert "at split 50%" in caught.value.detail


@pytest.mark.parametrize("split", [10, 50, 90])
@pytest.mark.parametrize("name", ["before_label", "after_label"])
def test_compare_label_capacity_at_each_split(name: str, split: int, image: Path) -> None:
    slots = {"before": image, "after": image, "split": split,
             "before_label": "", "after_label": ""}
    empty = cards.card_layer_plan("image_compare", slots, _style())
    v = empty.variables
    displayed_width = math.floor(min(1280, v["image_max_width_px"],
                                     v["image_max_height_px"] * 1280 / 800))
    share = (split if name == "after_label" else 100 - split) / 100
    chrome = 2 * (cards.COMPARE_CHIP_PAD + cards.COMPARE_CHIP_BORDER)
    max_width = math.floor(share * displayed_width - cards.COMPARE_CHIP_INSET
                           - cards.COMPARE_DIVIDER_KNOB / 2 - cards.COMPARE_CHIP_GAP)
    capacity = max(0, math.floor((max_width - chrome)
                                / (cards.PROPORTIONAL_EM * cards.MIN_TEXT_PX)))
    plan = cards.card_layer_plan("image_compare", {**slots, name: "x" * capacity}, _style())
    assert plan.variables[name] == "x" * capacity
    for side, region in (("after", split / 100), ("before", (100 - split) / 100)):
        maximum = plan.variables[f"{side}_chip_max_width_px"]
        assert maximum == max(0, math.floor(region * displayed_width
                                           - cards.COMPARE_CHIP_INSET
                                           - cards.COMPARE_DIVIDER_KNOB / 2
                                           - cards.COMPARE_CHIP_GAP))
        label = plan.variables[f"{side}_label"]
        if label:
            chip_width = len(label) * cards.PROPORTIONAL_EM * plan.variables["chip_px"] + chrome
            assert chip_width <= maximum
            assert (cards.COMPARE_CHIP_INSET + maximum + cards.COMPARE_DIVIDER_KNOB / 2
                    + cards.COMPARE_CHIP_GAP <= region * displayed_width)
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("image_compare", {**slots, name: "x" * (capacity + 1)}, _style())
    assert caught.value.parameter == f"slots.{name}"
    assert f"at most {capacity} characters to fit at 30 px at split {split}%" in caught.value.detail


@pytest.mark.parametrize("preset", PRESETS)
def test_compare_frame_maxima_and_fitted_labels(preset: str, image: Path) -> None:
    plan = cards.card_layer_plan("image_compare", {"before": image, "after": image},
                                 _preset(preset))
    v = plan.variables
    assert v["image_max_width_px"] + 2 * cards.COMPARE_MAT == v["panel_width_px"]
    assert v["image_max_height_px"] + 2 * cards.COMPARE_MAT == v["panel_max_height_px"]
    for name in ("chip_px", "image_max_width_px", "image_max_height_px"):
        assert type(v[name]) is int and v[name] >= cards.MIN_TEXT_PX
    minimum_width = math.floor(min(1280, v["image_max_width_px"],
                                   v["image_max_height_px"] * 1280 / 800))
    for name in ("before_label", "after_label"):
        chip_width = (len(v[name]) * cards.PROPORTIONAL_EM * v["chip_px"]
                      + 2 * (cards.COMPARE_CHIP_PAD + cards.COMPARE_CHIP_BORDER))
        side = name.removesuffix("_label")
        assert chip_width <= v[f"{side}_chip_max_width_px"]
        assert (chip_width + cards.COMPARE_CHIP_INSET + cards.COMPARE_DIVIDER_KNOB / 2
                + cards.COMPARE_CHIP_GAP <= minimum_width / 2)
    rules = _css_rules((cards.template_dir("image_compare") / "template.html").read_text())
    assert rules[".panel"]["padding"] == "8px" and rules[".panel"]["width"] == "max-content"
    assert rules[".panel"]["overflow"] == "hidden"
    assert rules[".panel"]["border-radius"] == "var(--card-radius)"
    assert rules[".panel"]["box-shadow"] == "var(--card-shadow)"
    assert rules[".before-image"]["width"] == rules[".before-image"]["height"] == "auto"
    assert rules[".frame"]["position"] == "relative"
    assert rules[".before-chip"]["right"] == rules[".after-chip"]["left"] == "12px"
    assert rules[".before-chip"]["max-width"] == "var(--card-before-chip-width)"
    assert rules[".after-chip"]["max-width"] == "var(--card-after-chip-width)"
    assert rules[".chip:empty"]["display"] == "none"
    assert rules[".divider"]["width"] == "4px"


def test_compare_images_share_centered_cover_rule() -> None:
    source = (cards.template_dir("image_compare") / "template.html").read_text()
    rules = _css_rules(source)
    for selector in (".before-image", ".after-image"):
        assert rules[selector]["object-fit"] == "cover"
        assert rules[selector]["object-position"] == "50% 50%"
    for name in ("before", "after"):
        description = cards.load_template("image_compare")["slots"][name]["description"]
        assert "object-fit: cover" in description
        assert "object-position: 50% 50%" in description


def test_compare_chips_are_above_image_layers() -> None:
    source = (cards.template_dir("image_compare") / "template.html").read_text()
    assert re.search(
        r'<div class="panel"><div class="frame"><img class="before-image"[^>]+>\s*'
        r'<div class="after-layer"><img class="after-image"[^>]+></div>\s*'
        r'<div class="chip after-chip"></div><div class="chip before-chip"></div>', source,
    )
    rules = _css_rules(source)
    assert rules[".after-chip"]["opacity"] == "0"
    assert "opacity" not in rules[".before-chip"]
    assert rules[".after-layer"]["inset"] == "0"
    assert rules[".after-image"]["width"] == rules[".after-image"]["height"] == "100%"


@pytest.mark.parametrize("suffix", [".png", ".JPG", ".jpeg", ".WEBP"])
def test_compare_assets_copy_is_local_identical_and_sources_untouched(
    suffix: str, image: Path, tmp_path: Path
) -> None:
    after = tmp_path / ("second" + suffix)
    ffmpeg.run(["ffmpeg", "-nostdin", "-f", "lavfi", "-i", "color=c=blue:s=800x1280",
                "-frames:v", "1", "-c:v", "png", "-f", "image2", str(after)], timeout_s=30)
    snapshots = [(p, p.read_bytes(), p.stat().st_mtime_ns) for p in (image, after)]
    assert snapshots[0][1] != snapshots[1][1]
    plan = cards.card_layer_plan("image_compare", {"before": str(image), "after": after}, _style())
    assert plan.assets == (("before.png", image.resolve()),
                           ("after" + suffix.lower(), after.resolve()))
    composition = cards.prepare_card(plan, tmp_path / "work")
    html = (composition / "index.html").read_text()
    assert plan.variables["before_src"] == "before.png"
    assert plan.variables["after_src"] == "after" + suffix.lower()
    for (filename, source), (original, data, modified) in zip(plan.assets, snapshots, strict=True):
        assert (composition / filename).read_bytes() == data
        assert original.read_bytes() == data and original.stat().st_mtime_ns == modified
        assert str(source) not in html and not source.is_relative_to(composition)
    assert hyperframes.preflight(composition, plan.frames) == hyperframes.layer_duration_s(
        plan.frames
    )


@pytest.mark.parametrize("filename", ["../before.png", "/after.png", "image.PNG", "other.png",
                                      "before.svg", "after.png/x", "before.png\n"])
def test_asset_filename_guard(filename: str, image: Path, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("image_card", {"image": image}, _style())
    with pytest.raises(ValidationError, match="assets"):
        cards.prepare_card(replace(plan, assets=((filename, image),)), tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_asset_copy_failure_is_atomic(
    image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cards.card_layer_plan("image_card", {"image": image}, _style())
    data, modified = image.read_bytes(), image.stat().st_mtime_ns
    copyfile = cards.shutil.copyfile

    def interrupted_copy(source: Path, target: Path) -> None:
        if target.suffix != ".png":
            copyfile(source, target)
            return
        target.write_bytes(b"partial")
        raise OSError("interrupted copy")

    monkeypatch.setattr(cards.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        cards.prepare_card(plan, tmp_path / "work")
    composition = tmp_path / "work" / "image_card"
    assert not (composition / "image.png").exists()
    assert not list(composition.glob(".yap_tmp_*"))
    assert image.read_bytes() == data and image.stat().st_mtime_ns == modified


def test_asset_sources_are_all_protected_before_copy(image: Path, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("image_card", {"image": image}, _style())
    composition = tmp_path / "work" / "image_card"
    composition.mkdir(parents=True)
    second = composition / "before.png"
    second.write_bytes(b"original source")
    forged = replace(plan, assets=(("before.png", image), ("after.png", second)))
    with pytest.raises(OutputPathError, match="input file"):
        cards.prepare_card(forged, tmp_path / "work")
    assert second.read_bytes() == b"original source"
    assert not (composition / "after.png").exists()


@pytest.mark.parametrize(("size", "capacity"), [("64x32", 0), ("1280x800", 8), ("1024x100", 20)])
def test_compare_capacity_uses_before_image_dimensions(
    size: str, capacity: int, image: Path, tmp_path: Path
) -> None:
    before = tmp_path / "before.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-f", "lavfi", "-i", f"color=c=red:s={size}",
                "-frames:v", "1", str(before)], timeout_s=30)
    slots = {"before": before, "after": image, "before_label": "x" * capacity, "after_label": ""}
    plan = cards.card_layer_plan("image_compare", slots, _style())
    assert plan.variables["before_label"] == "x" * capacity
    assert plan.variables["chip_px"] >= cards.MIN_TEXT_PX
    if capacity < 20:
        with pytest.raises(ValidationError, match=f"at most {capacity} characters"):
            cards.card_layer_plan("image_compare", {**slots, "before_label": "x" * (capacity + 1)},
                                  _style())


@pytest.mark.parametrize("template", cards.CARD_TEMPLATES)
def test_render_delegates_once_with_exact_variables_and_fonts(
    template: str,
    image: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = cards.card_layer_plan(template, _preview(template, image), _style(), frames=60, fps=24)
    out = tmp_path / "layer.mov"
    install = hyperframes.RendererInstall("test", tmp_path, tmp_path / "renderer", None, "test")
    result = hyperframes.LayerResult(out, 60, plan.width, 360, hyperframes.layer_duration_s(60, 24))
    seen = []

    def inspect(composition: Path, target: Path, **kwargs: object) -> hyperframes.LayerResult:
        assert composition == tmp_path / "work" / template
        assert target == out
        assert kwargs == {"frames": 60, "fps": 24, "variables": plan.variables, "install": install,
                          "browser_gpu": False}
        assert kwargs["variables"] is plan.variables
        assert hyperframes.preflight(composition, 60, 24) == result.duration_s
        expected = {face.file for name in plan.fonts for face in fonts.BUNDLED[name].faces}
        assert {p.name for p in (composition / "fonts").iterdir()} == expected
        source = (composition / "index.html").read_text()
        assert "BUNDLED_FONT_FACES" not in source
        assert source.count("@font-face") == len(expected)
        seen.append(composition)
        return result

    monkeypatch.setattr(hyperframes, "render_layer", inspect)
    assert cards.render_card(plan, tmp_path / "work", out, install=install) is result
    assert len(seen) == 1


def test_unknown_template_and_image_filename_guard(image: Path, tmp_path: Path) -> None:
    for operation in (cards.template_dir, cards.load_template):
        with pytest.raises(ValidationError, match="template"):
            operation("../title")
    plan = cards.card_layer_plan("image_card", {"image": image}, _style())
    forged = replace(plan, variables={**plan.variables, "image_src": "../image.png"})
    with pytest.raises(ValidationError, match="slots.image"):
        cards.prepare_card(forged, tmp_path / "work")


def test_schema_is_the_source_of_slot_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = cards.load_template("text_tag")
    metadata["slots"]["text"]["max_length"] = 3
    monkeypatch.setattr(cards, "load_template", lambda _: metadata)
    assert cards.card_layer_plan("text_tag", {"text": "abc"}, _style()).variables["text"] == "abc"
    with pytest.raises(ValidationError, match="slots.text"):
        cards.card_layer_plan("text_tag", {"text": "abcd"}, _style())


def test_slot_limits_match_the_contract() -> None:
    terminal = cards.load_template("terminal")["slots"]
    for name, minimum, maximum in (
        ("command", 1, 60),
        ("title", 0, 40),
        ("host", 0, 32),
        ("prompt", 0, 3),
    ):
        assert (terminal[name]["min_length"], terminal[name]["max_length"]) == (minimum, maximum)
    assert terminal["command"]["required"] is True
    assert terminal["output"] == {
        "type": "string_list",
        "required": False,
        "default": [],
        "min_items": 0,
        "max_items": 3,
        "min_length": 1,
        "max_length": 60,
    }
    assert terminal["cadence"]["options"] == ["human", "uniform"]
    assert terminal["type_at"]["default"] == 0.4
    assert terminal["cues"]["max_items"] == 3
    stat = cards.load_template("stat")["slots"]
    assert stat["end"]["required"] is True and stat["start"]["default"] == 0
    for name in ("start", "end"):
        assert stat[name]["type"] == "integer"
        assert (stat[name]["minimum"], stat[name]["maximum"]) == (-999999999, 999999999)
    assert [stat[n]["max_length"] for n in ("prefix", "suffix", "kicker")] == [3, 6, 40]
    checklist = cards.load_template("checklist")["slots"]
    assert checklist["items"] == {
        "type": "string_list",
        "required": True,
        "min_items": 1,
        "max_items": 4,
        "min_length": 1,
        "max_length": 40,
    }
    assert checklist["cues"]["max_items"] == 4
    tag = cards.load_template("text_tag")["slots"]
    assert tag["text"]["required"] is True and tag["text"]["min_length"] == 1
    assert [tag[n]["max_length"] for n in ("text", "subtext")] == [48, 60]
    image = cards.load_template("image_card")["slots"]
    assert image["image"]["required"] is True
    assert image["image"]["options"] == [".png", ".jpg", ".jpeg", ".webp"]
    assert image["title"]["max_length"] == 48
    assert image["chrome"]["options"] == ["window", "browser"]
    assert image["chrome"]["default"] == "window"


def _css_rules(source: str) -> dict[str, dict[str, str]]:
    css = re.search(r"<style>(.*?)</style>", source, re.DOTALL)[1]
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    blocks = re.findall(r"([^{}]+)\{([^{}]+)\}", css)
    selectors = [selector.strip() for selector, _ in blocks]
    assert len(selectors) == len(set(selectors)), "merge duplicate selector blocks"
    return {
        selector.strip(): dict(re.findall(r"([a-z-]+):\s*([^;]+);", body))
        for selector, body in blocks
    }


def test_stat_kicker_tracking_matches_fit() -> None:
    source = (cards.template_dir("stat") / "template.html").read_text()
    spacing = _css_rules(source)[".kicker"]["letter-spacing"]
    assert spacing.endswith("em")
    assert float(spacing.removesuffix("em")) == cards.KICKER_TRACKING_EM


def test_text_tag_look_css_matches_fitting() -> None:
    source = (cards.template_dir("text_tag") / "template.html").read_text()
    rules = _css_rules(source)
    padding = {
        "bar": "24px", "card": "24px 38px 26px 32px", "block": "20px 36px 24px 32px",
        "pill": "20px 64px 20px 40px", "underline": "24px",
        "rule": "24px 32px 24px 26px", "title": "32px",
    }
    assert set(padding) == set(cards.TAG_PADDING) == set(cards.TAG_WIDTH_RESERVE)
    assert set(padding) == set(cards.TAG_GAP) == set(cards.TAG_RULE_HEIGHT)
    for look in padding:
        prefix = f'.tag-wrapper[data-look="{look}"]'
        selector = prefix + (" .panel" if look == "pill" else " .tag-body")
        assert rules[selector]["padding"] == padding[look]
        left, right, top, bottom = cards.TAG_PADDING[look]
        lengths = [int(token.removesuffix("px")) for token in padding[look].split()]
        if len(lengths) == 1:
            lengths *= 4
        assert lengths == [top, right, bottom, left]
        assert float(rules[prefix + " .tag-body"]["gap"].removesuffix("px")) == (
            cards.TAG_GAP[look]
        )
        if look in {"card", "underline"}:
            assert rules[prefix + " .rule"]["height"] == f"{cards.TAG_RULE_HEIGHT[look]}px"
        if look in cards.TAG_TRACKING_EM:
            assert rules[prefix + " .tag-text"]["text-transform"] == "uppercase"
            assert float(rules[prefix + " .tag-text"]["letter-spacing"].removesuffix("em")) == (
                cards.TAG_TRACKING_EM[look]
            )
        if look in cards.TAG_SUB_TRACKING_EM:
            assert float(rules[prefix + " .subtext"]["letter-spacing"].removesuffix("em")) == (
                cards.TAG_SUB_TRACKING_EM[look]
            )
        if look in {"block", "underline"}:
            assert rules[prefix + " .subtext"]["font-family"] == "var(--card-mono)"
    side = '.tag-wrapper[data-look="rule"]'
    title = '.tag-wrapper[data-look="title"]'
    assert rules[side + " .tab"]["display"] == "block"
    assert rules[side + " .tab"]["width"] == f"{cards.TAG_WIDTH_RESERVE['rule']}px"
    assert rules[side + " .tab"]["flex"] == f"0 0 {cards.TAG_WIDTH_RESERVE['rule']}px"
    assert rules[side + " .tab"]["transform-origin"] == "50% 0%"
    assert cards.TAG_RULE_HEIGHT["rule"] == cards.TAG_RULE_HEIGHT["title"] == 0
    assert rules[side + " .subtext"]["color"] == "var(--card-ink)"
    assert rules[title]["width"] == "var(--card-panel-width)"
    assert rules[title + " .panel"]["width"] == "100%"
    assert rules[title + " .tag-body"]["width"] == "100%"
    assert rules[title + " .subtext"]["order"] == "-1"
    assert rules[title + " .subtext"]["text-transform"] == "uppercase"
    assert rules[title + " .subtext"]["color"] == "var(--card-accent)"
    assert rules[title + " .tag-text"]["font-weight"] == "600"
    for look in (side, title):
        for element in (" .tag-text", " .subtext"):
            assert "font-size" not in rules[look + element]
            assert "font-family" not in rules[look + element]
    for selector in (".tag-wrapper", ".tag-text", ".subtext", ".tab", ".rule"):
        assert rules[selector]["will-change"] == "transform"
    assert rules['.tag-wrapper[data-look="pill"] .panel']["border-radius"] == "999px"
    assert rules['.tag-wrapper[data-look="underline"] .panel']["background"] == "transparent"
    assert rules['.tag-wrapper[data-look="underline"] .panel']["color"] == "var(--card-plain-ink)"
    assert "rgb(0 0 0 /" in rules['.tag-wrapper[data-look="underline"] .tag-text']["text-shadow"]
    assert [attrs["data-look"] for _, attrs in _HTML(source).elements if "data-look" in attrs] == [
        "bar"
    ]
    assert "lt-mask-reveal" not in source


def _timeline_events(
    template: str,
    duration: float,
    tmp_path: Path,
    *,
    progress_samples: list[float] | None = None,
    dom_selectors: list[str] | None = None,
    path_y_scale: float = 1,
    defer_fonts: bool = False,
    font_widths_at_40: dict[str, int] | None = None,
    **variables: object,
) -> list[dict]:
    """Execute the template with DOM and timeline spies, without a browser."""
    if tools.find_tool("node") is None:
        pytest.skip("a JavaScript runtime is needed for timeline contract tests")
    source = cards.template_html(template)
    defaults = {v["id"]: v["default"] for v in cards.load_template(template)["variables"]}
    if template == "terminal" and "schedule" not in variables:
        variables["schedule"] = json.dumps(cards._terminal_schedule(
            command=str(variables.get("command", "")),
            output=json.loads(str(variables.get("output", defaults["output"]))),
            cadence=str(variables.pop("cadence", "human")),
            type_at=variables.pop("type_at", 0.4),
            word_cues=json.loads(str(variables.pop("word_cues", "[]"))),
            cues=json.loads(str(variables.pop("cues", "[]"))),
            duration=duration,
        ))
    if template in {"playing_grid", "reel_feed"}:
        windows = json.loads(defaults["windows"])
        for window in windows:
            window["frames"] = round(duration * defaults["fps"]) - window["start"]
            window["fade_out"] = max(1, min(9, round(duration * defaults["fps"] * 0.15)))
        variables.setdefault("windows", json.dumps(windows))
    payload = {
        "script": re.search(r"<script>(.*?)</script>", source, re.DOTALL)[1],
        "duration": duration,
        "variables": {**defaults, **variables},
        "progress_samples": progress_samples or [],
        "dom_selectors": dom_selectors or [],
        "path_y_scale": path_y_scale,
        "defer_fonts": defer_fonts,
        "font_widths_at_40": font_widths_at_40,
        "no_timeline": "data-no-timeline" in source,
    }
    payload_file = tmp_path / "timeline-input.json"
    payload_file.write_text(json.dumps(payload))
    runner = tmp_path / "timeline-spy.cjs"
    runner.write_text("""
const fs = require("fs");
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let fontsReady = !input.defer_fonts;
class Element {
  constructor(label) {
    this.label = label;
    this.className = label.startsWith(".") ? label.slice(1) : "";
    this.children = [];
    this.queries = new Map();
    this.attributes = {};
    this.style = { setProperty() {} };
    this.classList = { add() {} };
  }
  querySelector(selector) {
    if (!this.queries.has(selector)) {
      const child = new Element(selector);
      child.parent = this;
      this.queries.set(selector, child);
    }
    return this.queries.get(selector);
  }
  appendChild(child) { child.parent = this; this.children.push(child); }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  animate(keyframes, options) {
    const event = { kind: "animate", target: this.className || this.label, keyframes, options };
    events.push(event);
    return { pause() { event.paused = true; },
      set currentTime(value) { event.currentTime = value; } };
  }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  insertBefore(child) { this.appendChild(child); }
  setAttribute(name, value) {
    this.attributes[name] = value;
    if (name === "class") this.className = value;
  }
  cloneNode() { return new Element(this.label); }
  get fontSize() {
    return parseFloat(this.style.fontSize) || this.parent?.fontSize || input.variables.text_px;
  }
  get textWidth() {
    const text = this.textContent || this.children[0]?.textContent || "";
    const size = this.children[0]?.fontSize || this.fontSize;
    if (!fontsReady) return Math.ceil(text.length * size * 0.5);
    const width = input.font_widths_at_40[text];
    // Include fixed glyph overhang rather than scaling every width exactly.
    return Math.ceil((width - 2) * size / 40 + 2);
  }
  get clientWidth() {
    if (input.font_widths_at_40 && this.className === "flow-tag") {
      return Math.min(input.variables.tag_max_width_px, this.textWidth + 0.84 * this.fontSize);
    }
    if (input.font_widths_at_40 && this.className === "flow-tag-text") {
      return Math.round(parseFloat(this.style.width) || this.textWidth);
    }
    return input.variables.panel_width_px;
  }
  get scrollWidth() {
    if (!input.font_widths_at_40) return undefined;
    if (this.className === "flow-tag") {
      return Math.max(this.clientWidth, this.textWidth + 0.84 * this.fontSize);
    }
    if (this.className === "flow-tag-text") return Math.max(this.clientWidth, this.textWidth);
  }
  get clientHeight() { return input.variables.panel_max_height_px; }
  getBoundingClientRect() {
    if (this.label === ".panel") return { left: 0, top: 0, width: this.clientWidth,
      height: this.clientHeight };
    if (this.className === "row") {
      const height = input.variables.row_px * 1.15 + 2;
      const top = this.parent.children.indexOf(this) * (height + 4);
      return { left: 0, top, bottom: top + height, width: this.clientWidth, height };
    }
    if (this.label === ".chip") return {
      width: input.variables.chip_icon_px + input.variables.chip_gap_px +
        2 * input.variables.chip_pad_x_px +
        String(input.variables.payload).length * 0.6 * input.variables.chip_px,
      height: Math.max(input.variables.chip_icon_px, input.variables.chip_px * 1.2) +
        2 * input.variables.chip_pad_y_px,
    };
    let parent = this;
    while (parent && parent.label !== ".right-device") parent = parent.parent;
    const inset = input.variables.panel_inner_pad_px;
    const center = parent ? this.clientWidth - inset - input.variables.device_column_width_px / 2 :
      inset + input.variables.device_column_width_px / 2;
    const width = Number(this.attributes.width);
    const diagramLeft = center - input.variables.device_width_px / 2;
    const top = this.clientHeight - inset - input.variables.label_px * 1.2 -
      input.variables.device_label_gap_px - input.variables.device_height_px +
      Number(this.attributes.y);
    return { left: diagramLeft + Number(this.attributes.x), top, width,
      height: Number(this.attributes.height) };
  }
  getTotalLength() { return 600; }
  getPointAtLength(distance) {
    const points = this.attributes.d.match(/-?[0-9]+(?:[.][0-9]+)?/g).map(Number);
    const t = distance / this.getTotalLength();
    const weights = [(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3];
    const x = weights.reduce((sum, weight, i) => sum + weight * points[i * 2], 0);
    const y = weights.reduce((sum, weight, i) => sum + weight * points[i * 2 + 1], 0);
    const baseline = points[1] + (points[7] - points[1]) * t;
    return { x, y: baseline + (y - baseline) * input.path_y_scale };
  }
}
const root = new Element("root");
root.dataset = { width: "1040", height: "360", duration: String(input.duration) };
global.document = {
  getElementById(id) { if (id !== "root") throw new Error("unexpected root"); return root; },
  createElement(tag) { return new Element(tag); },
  createElementNS(namespace, tag) { return new Element(tag); },
  createTextNode(text) { const node = new Element("text"); node.textContent = text; return node; },
};
const fontCallbacks = [];
if (input.defer_fonts) {
  document.fonts = { ready: { then(callback) { fontCallbacks.push(callback); } } };
}
global.window = { __hyperframes: { getVariables() { return input.variables; } } };
global.getComputedStyle = element => ({ fontSize: `${element.fontSize}px`, paddingBottom: "12px" });
const events = [];
const seeks = [];
function tagLayout(phase) {
  if (!input.font_widths_at_40) return;
  const stack = root.children.find(node => node.className === "tag-stack");
  const tags = stack.children.filter(node =>
    node.className === "flow-tag");
  events.push({ kind: "tag-layout", target: phase, to: tags.map(tag => ({
    size: (tag.children[0] || tag).fontSize,
    text_width: tag.textWidth, client_width: tag.clientWidth, scroll_width: tag.scrollWidth,
  })) });
}
let paused = false;
let progressTween;
function record(kind, target, from, to, at) {
  for (const element of Array.isArray(target) ? target : [target]) {
    events.push({ kind, target: element.className || element.label || "progress", from,
      to: { ...to, ...(to.onUpdate ? { onUpdate: true } : {}) }, at });
  }
}
global.gsap = {
  set(target, to) { record("initial", target, {}, to, 0); },
  parseEase() { return progress => (1 - Math.cos(Math.PI * progress)) / 2; },
  timeline(options) {
    paused = options.paused;
    return {
      set(target, to, at) { record("set", target, {}, to, at); return this; },
      fromTo(target, from, to, at) {
        record("fromTo", target, from, to, at);
        if (to.onUpdate) progressTween = { target, render: to.onUpdate };
        return this;
      },
      seek(time) {
        if (!seeks.length) tagLayout("first-seek");
        seeks.push(time); return this;
      },
    };
  },
};
new Function(input.script)();
if (fontCallbacks.length) {
  events.push({ kind: "fonts-wait", target: "timeline", to: {
    seeks: [...seeks], registered: Object.keys(window.__timelines || {}),
  } });
  tagLayout("fonts-wait");
  fontsReady = true;
  tagLayout("fonts-loaded");
  fontCallbacks.forEach(callback => callback());
}
for (const selector of input.dom_selectors) {
  const element = root.querySelector(selector);
  function childrenOf(node) {
    return node.children.map(child => ({
      className: child.className, text: child.textContent, children: childrenOf(child),
    }));
  }
  events.push({ kind: "dom", target: selector, to: {},
    src: element.src, text: element.textContent, attributes: element.attributes,
    children: childrenOf(element) });
}
if (input.progress_samples.length) {
  events.push({ kind: "route", target: "guide", to: {},
    path: root.querySelector(".guide").attributes.d,
    chip: root.querySelector(".chip").getBoundingClientRect(),
    left: root.querySelector(".left-device").children[0]
      .querySelector(`.device-${input.variables.left_device}`).querySelector(".screen")
      .getBoundingClientRect(),
    right: root.querySelector(".right-device").children[0]
      .querySelector(`.device-${input.variables.right_device}`).querySelector(".screen")
      .getBoundingClientRect(),
  });
}
for (const progress of input.progress_samples) {
  progressTween.target.progress = progress;
  progressTween.render();
  events.push({ kind: "drawn", target: "route-progress", to: {},
    offset: root.querySelector(".route-progress").style.strokeDashoffset });
}
process.stdout.write(JSON.stringify({
  events, paused, seeks, registered: Object.keys(window.__timelines || {}),
}));
""")
    result = json.loads(
        tools.run_tool(
            "node",
            [str(runner), str(payload_file)],
            timeout_s=10,
        ).stdout
    )
    if payload["no_timeline"]:
        assert result["paused"] is False and result["seeks"] == result["registered"] == []
    else:
        assert result["paused"] is True and result["seeks"] == [0]
        assert result["registered"] == [template]
    return result["events"]


@pytest.mark.parametrize(("template", "variables"), [
    ("flow", {"nodes": '["A", "B"]'}),
    ("text_tag", {"look": "title", "text": "Label", "subtext": "Kicker"}),
    ("stat", {"reveal": "slot", "end": 12}),
])
def test_fitted_templates_wait_for_fonts_before_seek_and_registration(
    template: str, variables: dict, tmp_path: Path,
) -> None:
    events = _timeline_events(template, 3, tmp_path, defer_fonts=True, **variables)
    waiting = [event for event in events if event["kind"] == "fonts-wait"]
    assert len(waiting) == 1
    assert waiting[0]["to"] == {"seeks": [], "registered": []}


@pytest.mark.parametrize("template", cards.CARD_TEMPLATES)
def test_template_cleanup_keeps_only_used_css_and_list_helpers(template: str) -> None:
    source = (cards.template_dir(template) / "template.html").read_text()
    rules = _css_rules(source)
    if template in {"stat", "checklist", "text_tag", "steps", "flow"}:
        assert not {".bar", ".dots", ".dots i", ".bar-title"}.intersection(rules)
    if template in {"terminal", "checklist", "steps", "flow"}:
        assert "function list(value, coerce)" in source
        assert ("list(v.cues, Number)" in source) == (template in {"checklist", "steps", "flow"})
    else:
        assert "function list(" not in source


NEW_KINDS = (
    "chart", "quote", "progress_stat", "specs_list", "callout_circle", "name_tag",
    "success_check",
    "edit_timeline",
    "storyboard",
    "jiggle_delete",
    "file_drop",
    "video_card",
    "prompt_bar",
    "comment_cta",
    "comment_drop",
    "icon_strike",
    "playing_grid", "reel_feed",
    "gravity_drop",
    "choices",
)


@pytest.mark.parametrize("template", NEW_KINDS)
def test_new_kind_preview_slots_and_safe_box(template: str, image: Path) -> None:
    slots = _preview(template, image)
    plan = cards.card_layer_plan(template, slots, _style())
    assert 0 <= plan.x and plan.x + plan.width <= 1080
    assert 0 <= plan.y and plan.y + plan.height <= 1920
    assert plan.variables["panel_max_height_px"] == (
        plan.height - 88 if template == "video_card" else 272)
    assert set(plan.variables) == _default_variable_ids(template)


@pytest.mark.parametrize("template", NEW_KINDS)
@pytest.mark.parametrize("look", ("paper", "minimal", "dark", "bright"))
def test_new_kind_uses_brand_tokens_and_local_assets(
    template: str, look: str, image: Path,
) -> None:
    style = _style(style=look, font="Gelasio", accent_color="#A13467")
    plan = cards.card_layer_plan(template, _preview(template, image), style)
    assert plan.variables["accent_color"] == "#A13467"
    assert plan.variables["font_family"] == fonts.font_stack("Gelasio")
    assert plan.variables["card_radius_px"] == cards._LOOKS[look][0]
    source = (cards.template_dir(template) / "template.html").read_text()
    assert "http://" not in source and "https://" not in source


def test_chart_planned_lengths_follow_zero_based_values(image: Path) -> None:
    plan = cards.card_layer_plan("chart", _preview("chart", image), _style())
    bars = json.loads(plan.variables["bars"])
    track = plan.variables["track_width_px"]
    assert [bar["value"] for bar in bars] == [42, 68, 91]
    assert bars[-1]["length_px"] == track
    for bar in bars:
        assert abs(bar["length_px"] - track * bar["value"] / 91) <= 1
    assert [bar["length_px"] for bar in bars] == sorted(bar["length_px"] for bar in bars)


def test_quote_preview_has_quote_larger_than_attribution(image: Path) -> None:
    plan = cards.card_layer_plan("quote", _preview("quote", image), _style())
    assert plan.variables["quote_px"] > plan.variables["name_px"]
    assert plan.variables["name_px"] >= plan.variables["role_px"]


@pytest.mark.parametrize(("template", "changes"), [
    ("chart", {"title": "A longer view of progress", "bars": [
        {"label": label, "value": value} for label, value in
        (("Jan", 10), ("Feb", 25), ("Mar", 50), ("Apr", 75), ("May", 100))
    ]}),
    ("quote", {"quote": "A clear process made each decision easier for the whole team."}),
    ("progress_stat", {"label": "Completion across this phase", "value": 100}),
    ("specs_list", {"title": "Specifications", "rows": [
        {"name": name, "value": value} for name, value in
        (("Size", "Small"), ("Weight", "Light"), ("Color", "Blue"),
         ("Finish", "Matte"), ("Use", "Daily"))
    ]}),
    ("callout_circle", {"label": "Notice this small detail", "x": 50, "y": 50, "r": 20}),
    ("name_tag", {"name": "Dr. Maya Chen", "role": "Founder and researcher"}),
    ("success_check", {"label": "Everything is ready to go"}),
])
def test_new_kind_long_slots_fit(template: str, changes: dict, image: Path) -> None:
    plan = cards.card_layer_plan(template, {**_preview(template, image), **changes}, _style())
    assert plan.width <= 1080 and plan.height == 360
    assert all(
        value >= cards.MIN_TEXT_PX for name, value in plan.variables.items()
        if name.endswith("_px") and name in TEXT_SIZES[template]
    )


@pytest.mark.parametrize(("template", "changes", "field"), [
    ("chart", {"bars": [{"label": "Only", "value": 1}]}, "bars"),
    ("chart", {"bars": [{"label": "A", "value": 1}, {"label": "B", "value": True}]},
     "bars[1].value"),
    ("quote", {"name": ""}, "name"),
    ("progress_stat", {"value": 101}, "value"),
    ("specs_list", {"rows": [{"name": "A", "value": "B"}]}, "rows"),
    ("specs_list", {"rows": [{"name": "A", "value": "B"}, {"name": "C"}]}, "rows[1]"),
    ("callout_circle", {"x": 5, "r": 20}, "r"),
    ("name_tag", {"name": ""}, "name"),
    ("success_check", {"label": ""}, "label"),
    ("edit_timeline", {"title": "x" * 29}, "title"),
])
def test_new_kind_bad_slots_have_plain_errors(
    template: str, changes: dict, field: str, image: Path,
) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**_preview(template, image), **changes}, _style())
    assert caught.value.parameter == f"slots.{field}"
    assert caught.value.detail and any(
        word in caught.value.detail for word in ("expected", "use", "keep")
    )


def test_callout_circle_uses_shorter_image_side_for_wide_image(tmp_path: Path) -> None:
    image = tmp_path / "wide.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-f", "lavfi", "-i",
                "color=c=red:s=1920x200", "-frames:v", "1", str(image)], timeout_s=30)
    plan = cards.card_layer_plan(
        "callout_circle", {**_preview("callout_circle", image),
                           "x": 10, "y": 50, "r": 20}, _style(),
    )
    assert plan.variables["x"] == 10
    assert plan.variables["image_width_px"] > plan.variables["image_height_px"]


def test_callout_circle_refuses_radius_outside_narrow_width(tmp_path: Path) -> None:
    image = tmp_path / "tall.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-f", "lavfi", "-i",
                "color=c=red:s=200x1920", "-frames:v", "1", str(image)], timeout_s=30)
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(
            "callout_circle", {**_preview("callout_circle", image),
                               "x": 10, "y": 50, "r": 20}, _style(),
        )
    assert caught.value.parameter == "slots.r"
    assert caught.value.detail == "keep the circle inside the image"


def test_callout_label_fits_displayed_image_width(image: Path) -> None:
    label = "W" * 12
    plan = cards.card_layer_plan(
        "callout_circle", {**_preview("callout_circle", image), "label": label}, _style(),
    )
    # CSS reserves 12 px on each side, then 10 px padding on each side and a 6 px border.
    text_room = plan.variables["image_width_px"] - 24 - 20 - 6
    assert text_width_em(label, "Inter", 700) * plan.variables["label_px"] <= text_room


def test_callout_label_refuses_text_too_wide_for_displayed_image(image: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(
            "callout_circle", {**_preview("callout_circle", image), "label": "W" * 14},
            _style(),
        )
    assert caught.value.parameter == "slots.label"
    assert "shorten" in caught.value.detail


@pytest.mark.parametrize("template", [kind for kind, sources in UPSTREAM.items() if sources])
@pytest.mark.parametrize("short", [False, True])
def test_cards_start_with_transparent_painting_elements(
    template: str,
    short: bool,
    tmp_path: Path,
) -> None:
    duration = (0.7 if template == "terminal" else 0.032) if short else 3
    if template == "video_card":
        duration = max(1.5, duration)
    if template in {"playing_grid", "reel_feed"}:
        duration = max(2, duration)
    events = _timeline_events(
        template,
        duration,
        tmp_path,
        command="x",
        type_at=0,
        items=json.dumps(["First", "Second"]),
        text="Label",
    )
    painting = {"text_tag": "tag-wrapper", "playing_grid": "tile",
                "reel_feed": "scene"}.get(template, "panel")
    initial = [e for e in events if e["kind"] == "initial" and e["target"] == painting]
    assert initial and all(e["to"]["opacity"] == 0 for e in initial)
    if template == "text_tag":
        assert all(
            e["to"]["opacity"] == 0
            for e in events
            if e["kind"] == "initial" and e["target"] == "shadow-backing"
        )
    else:
        entrance = next(
            e
            for e in events
            if e["target"] == painting and e["kind"] == "fromTo" and e["to"].get("opacity") == 1
        )
        assert entrance["at"] == 0 and entrance["from"]["opacity"] == 0
        assert 0 < entrance["to"]["duration"] <= duration


@pytest.mark.parametrize("panels", range(4, 9))
@pytest.mark.parametrize("at", [0, 0.3, 1.1])
def test_image_collage_seeded_arcs_and_reserved_rest(
    panels: int, at: float, tmp_path: Path,
) -> None:
    variables = {"reveal": "collage", "reveal_at": at, "panels": panels}
    events = _timeline_events("image_card", 3, tmp_path, **variables)
    assert events == _timeline_events("image_card", 3, tmp_path, **variables)
    exits = [e for e in events if e["kind"] == "fromTo" and e["target"] == "collage-scrap"]
    arcs = [e for e in events if e["kind"] == "fromTo" and e["target"] == "collage-arc"]
    assert len(exits) == len(arcs) == panels
    assert exits[0]["at"] == at
    assert all(a["at"] < b["at"] for a, b in zip(exits, exits[1:], strict=False))
    assert len({(e["to"]["x"], e["to"]["y"], e["to"]["rotation"]) for e in exits}) == panels
    for exit_event, arc in zip(exits, arcs, strict=True):
        assert exit_event["from"] == {"x": 0, "y": 0, "rotation": 0, "--card-lift": 0}
        assert exit_event["to"]["--card-lift"] == 1
        assert arc["from"] == {"x": 0, "y": 0}
        assert arc["at"] == exit_event["at"]
        assert (arc["to"]["x"], arc["to"]["y"]) != (0, 0)
        for event in (exit_event, arc):
            assert event["to"]["duration"] == cards.COLLAGE_EXIT_S
            assert event["to"]["force3D"] is False
            assert event["at"] + event["to"]["duration"] <= at + cards.COLLAGE_SPAN_S
    fade = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "panel"
                and e["to"].get("opacity") == 0)
    assert fade["at"] - (exits[-1]["at"] + cards.COLLAGE_EXIT_S) >= cards.COLLAGE_REST_S


def test_image_fade_timeline_is_unchanged_by_collage_slots(tmp_path: Path) -> None:
    default = _timeline_events("image_card", 3, tmp_path)
    explicit = _timeline_events("image_card", 3, tmp_path,
                                reveal="fade", reveal_at=-100, panels=8)
    assert default == explicit
    assert not any(e["target"].startswith("collage-") for e in default)
    entrance = [e for e in default if e["kind"] == "fromTo" and e["at"] == 0]
    assert [e["from"] for e in entrance] == [{"opacity": 0}, {"y": 16, "scale": 0.965}]
    assert [e["to"]["duration"] for e in entrance] == [0.9 * 0.66, 0.9]


@pytest.mark.parametrize(("duration", "travel_at"), [(2.299, 0), (2.299, 0.6),
                                                    (2.499, 0.899), (0.4, 0.1)])
def test_transfer_motion_reserves_landing_and_fade(
    duration: float, travel_at: float, tmp_path: Path
) -> None:
    events = _timeline_events("transfer", duration, tmp_path, travel_at=travel_at)
    progress = [e for e in events if e["target"] == "progress"]
    assert len(progress) == 1
    motion = progress[0]
    assert motion["at"] == travel_at and motion["from"] == {"progress": 0}
    assert motion["to"]["progress"] == 1 and motion["to"]["onUpdate"] is True
    assert motion["to"]["ease"] == "power2.inOut"
    out_start = duration - min(0.3, duration * 0.15)
    scale = min(1, (out_start - travel_at) / (cards.TRANSFER_TRAVEL_S + cards.TRANSFER_LAND_S))
    assert motion["to"]["duration"] == pytest.approx(cards.TRANSFER_TRAVEL_S * scale)
    tick = next(e for e in events if e["target"] == "landing-tick" and e["kind"] == "fromTo")
    assert tick["at"] == pytest.approx(
        travel_at + motion["to"]["duration"] + cards.TRANSFER_LAND_S * scale / 2
    )
    assert tick["to"]["duration"] == pytest.approx(cards.TRANSFER_LAND_S * scale / 2)
    assert tick["to"]["strokeDashoffset"] == 0
    assert tick["at"] + tick["to"]["duration"] <= out_start + 1e-12
    pulse = [e for e in events if e["target"] == "landing-outline" and e["kind"] == "fromTo"]
    assert len(pulse) == 2
    assert [e["to"]["opacity"] for e in pulse] == [1, 0]
    assert sum(e["to"]["duration"] for e in pulse) == pytest.approx(tick["to"]["duration"])
    appear = next(e for e in events if e["target"] == "traveler" and e["kind"] == "fromTo")
    assert appear["at"] == max(0, travel_at - 0.15 * scale)
    assert appear["from"] == {"opacity": 0, "scale": 0.4}
    assert appear["to"]["opacity"] == 1 and appear["to"]["scale"] == 1
    receipt = next(e for e in events if e["target"] == "traveler" and e["kind"] == "fromTo"
                   and e["to"].get("opacity") == 0)
    assert receipt["at"] == pytest.approx(travel_at + motion["to"]["duration"])
    assert receipt["at"] + receipt["to"]["duration"] == pytest.approx(tick["at"])
    assert receipt["from"] == {"opacity": 1, "scale": 1}
    assert receipt["to"]["scale"] == 0.4
    assert receipt["to"]["duration"] == tick["to"]["duration"]
    assert receipt["to"]["immediateRender"] is False
    tint = [e for e in events if e["target"] == "landing-tint" and e["kind"] == "fromTo"]
    assert len(tint) == 2 and [e["to"]["opacity"] for e in tint] == [1, 0]
    assert tint[0]["at"] == tick["at"]
    assert sum(e["to"]["duration"] for e in tint) == pytest.approx(tick["to"]["duration"])
    exit_event = next(e for e in events if e["target"] == "panel" and e["to"].get("opacity") == 0
                      and e["kind"] == "fromTo")
    assert exit_event["at"] == out_start
    assert exit_event["at"] + exit_event["to"]["duration"] == pytest.approx(duration)


@pytest.mark.parametrize(("duration", "wipe_at"), [(2.199, 0), (2.199, 0.4),
                                                 (2.499, 0.849), (0.4, 0.1)])
@pytest.mark.parametrize("split", [10, 50, 90])
def test_compare_motion_uses_one_split_and_reserves_rest(
    duration: float, wipe_at: float, split: int, tmp_path: Path
) -> None:
    events = _timeline_events("image_compare", duration, tmp_path, wipe_at=wipe_at, split=split)
    split_events = [e for e in events if e["kind"] == "fromTo"
                    and "--card-split-position" in e["to"]]
    assert len(split_events) == 1
    motion = split_events[0]
    assert motion["target"] == "panel" and motion["at"] == wipe_at
    assert motion["from"] == {"--card-split-position": 0}
    assert motion["to"]["--card-split-position"] == split
    assert motion["to"]["ease"] == "sine.inOut"
    out_start = duration - min(0.3, duration * 0.15)
    scale = min(1, (out_start - wipe_at) / (cards.COMPARE_WIPE_S + cards.COMPARE_REST_S))
    assert motion["to"]["duration"] == pytest.approx(cards.COMPARE_WIPE_S * scale)
    assert (motion["at"] + motion["to"]["duration"] + cards.COMPARE_REST_S * scale
            <= out_start + 1e-12)
    divider_initial = next(e for e in events if e["target"] == "divider"
                           and e["kind"] == "initial")
    assert divider_initial["to"] == {"opacity": 0}
    divider = next(e for e in events if e["target"] == "divider" and e["kind"] == "fromTo")
    assert divider["at"] == wipe_at and divider["from"] == {"opacity": 0}
    assert divider["to"]["opacity"] == 1 and divider["to"]["immediateRender"] is False
    assert divider["to"]["duration"] == pytest.approx(min(0.22 * scale, motion["to"]["duration"]))
    entrance = next(e for e in events if e["kind"] == "fromTo" and "y" in e["from"])
    assert entrance["from"] == {"opacity": 0, "y": 8}
    assert entrance["to"]["duration"] == min(0.32, out_start)
    exits = [e for e in events if e["kind"] == "fromTo" and e["to"].get("opacity") == 0]
    assert len(exits) == 1 and exits[0]["target"] == "panel"
    assert exits[0]["at"] == out_start and exits[0]["to"]["immediateRender"] is False
    assert exits[0]["at"] + exits[0]["to"]["duration"] == pytest.approx(duration)
    source = (cards.template_dir("image_compare") / "template.html").read_text()
    rules = _css_rules(source)
    assert rules[".after-layer"]["clip-path"] == (
        "inset(0 calc((100 - var(--card-split-position)) * 1%) 0 0)"
    )
    assert rules[".divider"]["left"] == "calc(var(--card-split-position) * 1%)"
    assert re.search(r'<div class="after-layer"><img[^>]+></div>\s*'
                     r'<div class="chip after-chip"></div><div class="chip before-chip">', source)
    assert not any(token in source for token in ("dispatchEvent", "CustomEvent", "exit", "baw-"))


@pytest.mark.parametrize("split", [20, 50, 80])
@pytest.mark.parametrize("frames", [66, 75])
def test_compare_after_chip_fades_only_after_reveal(
    split: int, frames: int, image: Path, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan("image_compare", {
        "before": image, "after": image, "split": split, "before_label": "B", "after_label": "A",
    }, _style(), frames=frames)
    v = plan.variables
    events = _timeline_events("image_compare", hyperframes.layer_duration_s(frames), tmp_path,
                              **v)
    initial = next(e for e in events if e["kind"] == "initial" and e["target"] == "after-chip")
    assert initial["to"] == {"opacity": 0}
    fade = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "after-chip")
    assert fade["from"] == {"opacity": 0}
    assert fade["to"]["opacity"] == 1 and fade["to"]["immediateRender"] is False
    progress = v["after_chip_reveal_progress"]
    assert 0.5 < progress < 1
    assert fade["at"] == pytest.approx(v["wipe_at"] + cards.COMPARE_WIPE_S * progress)
    assert fade["at"] + fade["to"]["duration"] == pytest.approx(
        v["wipe_at"] + cards.COMPARE_WIPE_S
    )
    displayed_width = math.floor(min(1280, v["image_max_width_px"],
                                     v["image_max_height_px"] * 1280 / 800))
    divider_at_reveal = displayed_width * split / 100 * (1 - math.cos(math.pi * progress)) / 2
    assert divider_at_reveal == pytest.approx(
        cards.COMPARE_CHIP_INSET + v["after_chip_max_width_px"]
    )
    assert not any(e["target"] == "before-chip" for e in events)


@pytest.mark.parametrize("filename", ["before.png", "after.jpg", "image.webp", "after.jpeg",
                                      "../before.png", "after.PNG", "other.png", "before.png\n"])
def test_compare_dom_uses_guarded_images_and_literal_labels(filename: str, tmp_path: Path) -> None:
    selectors = [".before-image", ".after-image", ".before-chip", ".after-chip"]
    events = _timeline_events("image_compare", 3, tmp_path, dom_selectors=selectors,
                              before_src=filename, after_src=filename,
                              before_label="<b>Before</b>", after_label="")
    elements = {e["target"]: e for e in events if e["kind"] == "dom"}
    allowed = re.fullmatch(r"(image|before|after)\.(png|jpe?g|webp)", filename) is not None
    for selector in selectors[:2]:
        if allowed:
            assert elements[selector]["src"] == filename
        else:
            assert "src" not in elements[selector]
    assert elements[".before-chip"]["text"] == "<b>Before</b>"
    assert elements[".after-chip"]["text"] == ""


@pytest.mark.parametrize("frames", [69, 75])
def test_transfer_arrival_handoff_keeps_tick_and_route(frames: int, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("transfer", _transfer_slots(payload="x" * 16), _style(),
                                frames=frames)
    duration = hyperframes.layer_duration_s(frames)
    events = _timeline_events(
        "transfer", duration, tmp_path, progress_samples=[1], **plan.variables
    )
    arrived = plan.variables["travel_at"] + cards.TRANSFER_TRAVEL_S + cards.TRANSFER_LAND_S
    hold_at = (arrived + duration - min(0.3, duration * 0.15)) / 2

    def completed_state(target: str) -> dict:
        state = {}
        for event in events:
            if event["target"] != target:
                continue
            if event["kind"] == "initial" or (
                event["kind"] == "fromTo"
                and event["at"] + event["to"]["duration"] <= hold_at
            ):
                state.update(event["to"])
        return state

    chip = completed_state("traveler")
    tick = completed_state("landing-tick")
    assert chip["opacity"] == 0 and chip["scale"] == 0.4
    assert tick["opacity"] == 1 and tick["strokeDashoffset"] == 0
    assert completed_state("landing-tint")["opacity"] == 0
    assert completed_state("panel")["opacity"] == 1
    assert float(next(e["offset"] for e in events if e["kind"] == "drawn")) == 0
    source = (cards.template_dir("transfer") / "template.html").read_text()
    tint_fill = _css_rules(source)[".landing-tint"]["fill"]
    assert "var(--card-accent)" in tint_fill and "var(--card-titlebar)" in tint_fill


@pytest.mark.parametrize("path_y_scale", [1, 100])
def test_transfer_progress_sampling_is_repeatable_and_clamps_tangent(
    path_y_scale: float, tmp_path: Path
) -> None:
    samples = [0, 0.25, 0.5, 1, 0.25, 0, 0.5]
    plan = cards.card_layer_plan("transfer", _transfer_slots(), _style())
    events = _timeline_events(
        "transfer", 3, tmp_path, progress_samples=samples, path_y_scale=path_y_scale,
        **plan.variables,
    )
    poses = [e["to"] for e in events if e["target"] == "traveler" and "x" in e["to"]][
        -len(samples):
    ]
    route = next(e for e in events if e["kind"] == "route")
    path = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", route["path"])]
    assert len(poses) == len(samples)
    assert poses[0] == poses[5] and poses[1] == poses[4] and poses[2] == poses[6]
    assert (poses[0]["x"], poses[0]["y"]) == pytest.approx(path[:2])
    assert (poses[3]["x"], poses[3]["y"]) == pytest.approx(path[-2:])
    midpoint = sum(weight * path[i] for weight, i in zip((0.125, 0.375, 0.375, 0.125),
                                                      (1, 3, 5, 7), strict=True))
    assert poses[2]["x"] == pytest.approx((path[0] + path[6]) / 2)
    baseline = (path[1] + path[7]) / 2
    assert poses[2]["y"] == pytest.approx(baseline + (midpoint - baseline) * path_y_scale)
    for progress, pose in zip(samples, poses, strict=True):
        assert (pose["xPercent"], pose["yPercent"]) == (-50, -50)
        assert abs(pose["rotation"]) <= 12 * (1 - progress)
    assert poses[3]["rotation"] == 0
    if path_y_scale == 100:
        assert poses[0]["rotation"] == -12 and poses[1]["rotation"] == -9
    offsets = [float(e["offset"]) for e in events if e["kind"] == "drawn"]
    assert offsets == pytest.approx([(1 - progress) * 600 for progress in samples])


@pytest.mark.parametrize("width", [600, 800, 920, 960, 1080])
@pytest.mark.parametrize("left_device", ["laptop", "desktop"])
@pytest.mark.parametrize("right_device", ["laptop", "desktop"])
@pytest.mark.parametrize("payload_size", ["short", "tall", "longest"])
def test_transfer_route_screen_centres_and_panel_bounds(
    width: int,
    left_device: str,
    right_device: str,
    payload_size: str,
    tmp_path: Path,
) -> None:
    style = _style(card_width_px=width)
    slots = _transfer_slots(left_device=left_device, right_device=right_device, payload="x")
    v = cards.card_layer_plan("transfer", slots, style).variables
    chrome = v["chip_icon_px"] + v["chip_gap_px"] + 2 * v["chip_pad_x_px"]
    capacity = min(16, math.floor((v["chip_max_width_px"] - chrome) /
                                 (cards.PROPORTIONAL_EM * cards.MIN_TEXT_PX)))
    count = {"short": 1, "tall": min(12, capacity), "longest": capacity}[payload_size]
    slots["payload"] = "x" * count
    plan = cards.card_layer_plan("transfer", slots, style)
    if width == 960 and payload_size == "longest":
        assert plan.variables["chip_px"] == cards.MIN_TEXT_PX
    samples = [i / 100 for i in range(101)]
    events = _timeline_events("transfer", 3, tmp_path, progress_samples=samples, **plan.variables)
    route = next(e for e in events if e["kind"] == "route")
    path = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", route["path"])]
    chip = route["chip"]
    assert path[1] - path[3] == pytest.approx(path[7] - path[5])
    assert path[1] - path[3] > 50, "preserve the visible arc above the screens"
    assert path[2] + path[4] == pytest.approx(path[0] + path[6])
    poses = [e["to"] for e in events if e["target"] == "traveler" and "x" in e["to"]][
        -len(samples):
    ]
    for pose in poses:
        angle = math.radians(pose["rotation"])
        half_width = (chip["width"] * math.cos(angle) + chip["height"] * abs(math.sin(angle))) / 2
        half_height = (chip["height"] * math.cos(angle) + chip["width"] * abs(math.sin(angle))) / 2
        assert 0 <= pose["x"] - half_width and pose["x"] + half_width <= width
        assert 0 <= pose["y"] - half_height
        assert pose["y"] + half_height <= plan.variables["panel_max_height_px"]
    for pose, screen in ((poses[0], route["left"]), (poses[-1], route["right"])):
        assert pose["x"] == pytest.approx(screen["left"] + screen["width"] / 2)
        assert pose["y"] == pytest.approx(screen["top"] + screen["height"] / 2)
    assert poses[-1]["rotation"] == 0
    inset = plan.variables["panel_inner_pad_px"]
    left_center = route["left"]["left"] + route["left"]["width"] / 2
    right_center = route["right"]["left"] + route["right"]["width"] / 2
    clearance = min(left_center - inset, width - inset - right_center)
    assert plan.variables["chip_max_width_px"] == math.floor(2 * clearance)
    assert chip["width"] / 2 <= clearance
    assert poses[len(samples) // 2]["y"] < min(path[1], path[7]) - 50


@pytest.mark.parametrize("radius", [16, 30, 36])
@pytest.mark.parametrize("duration", [0.032, 3])
def test_tag_round_wipe_and_shadow_have_separate_paint_and_shared_exit(
    radius: int,
    duration: float,
    tmp_path: Path,
) -> None:
    source = (cards.template_dir("text_tag") / "template.html").read_text()
    assert re.search(
        r'<div class="tag-wrapper" data-look="bar"><div class="shadow-backing"[^>]*></div>\s*'
        r'<div class="panel">',
        source,
    )
    rules = _css_rules(source)
    backing, panel, wrapper = (
        rules[name] for name in (".shadow-backing", ".panel", ".tag-wrapper")
    )
    assert backing["position"] == "absolute" and backing["inset"] == "0"
    assert backing["border-radius"] == panel["border-radius"] == "var(--card-radius)"
    assert backing["box-shadow"] == "var(--card-shadow)"
    assert not {"background", "clip-path", "overflow"}.intersection(backing)
    assert panel["background"] == "var(--card-paper)" and panel["overflow"] == "hidden"
    assert "box-shadow" not in panel
    assert wrapper["position"] == "relative" and "overflow" not in wrapper
    events = _timeline_events("text_tag", duration, tmp_path, card_radius_px=radius, text="Label")
    wipe = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "panel")
    assert wipe["from"]["clipPath"] == f"inset(0 100% 0 0 round {radius}px)"
    assert wipe["to"]["clipPath"] == f"inset(0 0% 0 0 round {radius}px)"
    shadow = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "shadow-backing")
    assert shadow["from"]["opacity"] == 0 and shadow["to"]["opacity"] == 1
    assert (shadow["at"], shadow["to"]["duration"]) == (wipe["at"], wipe["to"]["duration"])
    exits = [e for e in events if e["kind"] == "fromTo" and e["to"].get("opacity") == 0]
    assert len(exits) == 1 and exits[0]["target"] == "tag-wrapper"
    assert exits[0]["at"] + exits[0]["to"]["duration"] == pytest.approx(duration)


@pytest.mark.parametrize("duration", [0.032, 0.4, 3])
@pytest.mark.parametrize(
    ("look", "natural_end", "targets", "times", "lengths", "eases"),
    [
        ("bar", 0.94, ["panel", "shadow-backing", "tab", "tag-text", "subtext"],
         [0.1, 0.1, 0.28, 0.34, 0.44], [0.55, 0.55, 0.45, 0.5, 0.5],
         ["power3.out", "power3.out", "power2.out", "power3.out", "power3.out"]),
        ("card", 1.01, ["tag-wrapper", "tag-text", "rule", "subtext"],
         [0.1, 0.26, 0.42, 0.56], [0.5, 0.45, 0.5, 0.45],
         ["power3.out", "power3.out", "power4.out", "power2.out"]),
        ("block", 0.82, ["tag-wrapper", "tag-text", "subtext"],
         [0.1, 0.3, 0.4], [0.5, 0.42, 0.42],
         ["back.out(1.4)", "power3.out", "power3.out"]),
        ("pill", 0.95, ["tag-wrapper", "tab", "tag-text", "subtext"],
         [0.1, 0.4, 0.42, 0.5], [0.55, 0.4, 0.45, 0.45],
         ["back.out(1.7)", "back.out(3)", "power3.out", "power3.out"]),
        ("underline", 0.96, ["tag-text", "rule", "subtext"],
         [0.1, 0.3, 0.46], [0.55, 0.5, 0.5],
         ["power3.out", "power4.out", "power3.out"]),
        ("rule", 0.88, ["tag-wrapper", "tab", "tag-text", "subtext"],
         [0.1, 0.1, 0.26, 0.38], [0.5, 0.5, 0.5, 0.5],
         ["power3.out", "power3.out", "power3.out", "power3.out"]),
        ("title", 1.0, ["tag-wrapper", "subtext", "tag-text"],
         [0, 0, 0.28], [0.3, 0.5, 0.72],
         ["power2.out", "power2.out", "power3.out"]),
    ],
)
def test_text_tag_motion_preserves_upstream_order_and_shared_exit(
    duration: float, look: str, natural_end: float, targets: list[str], times: list[float],
    lengths: list[float], eases: list[str], tmp_path: Path,
) -> None:
    events = _timeline_events("text_tag", duration, tmp_path, look=look, text="Label",
                              subtext="Detail")
    initial = [e for e in events if e["kind"] == "initial" and e["target"] == "tag-wrapper"]
    assert initial and all(e["to"]["opacity"] == 0 for e in initial)
    tweens = [e for e in events if e["kind"] == "fromTo"]
    entrance, fade = tweens[:-1], tweens[-1]
    out = min(0.3, duration * 0.15)
    out_start = duration - out
    scale = min(1, out_start / natural_end)
    assert [e["target"] for e in entrance] == targets
    assert [e["at"] for e in entrance] == pytest.approx([t * scale for t in times])
    assert [e["to"]["duration"] for e in entrance] == pytest.approx(
        [length * scale for length in lengths]
    )
    assert [e["to"]["ease"] for e in entrance] == eases
    assert all(e["at"] + e["to"]["duration"] <= out_start + 1e-12 for e in entrance)
    assert fade["target"] == "tag-wrapper" and fade["from"] == {"opacity": 1}
    assert fade["to"] == {
        "opacity": 0, "duration": out, "ease": "power2.in", "immediateRender": False,
    }
    assert fade["at"] == out_start
    assert fade["at"] + fade["to"]["duration"] == pytest.approx(duration)
    assert all("visibility" not in e["to"] for e in events)
    if look == "card":
        assert entrance[0]["from"] == {"y": 24, "opacity": 0}
        assert entrance[2]["from"] == {"scaleX": 0} and entrance[2]["to"]["scaleX"] == 1
    elif look == "block":
        assert entrance[0]["from"] == {"x": -24, "opacity": 0}
        assert abs(entrance[0]["from"]["x"]) < cards.PAD_X
    elif look == "pill":
        assert entrance[0]["from"] == {"scale": 0.88, "y": 18, "opacity": 0}
        assert initial[-1]["to"]["transformOrigin"] == "0% 100%"
        assert entrance[1]["from"] == {"scale": 0} and entrance[1]["to"]["scale"] == 1
    elif look == "underline":
        assert entrance[0]["from"] == {"y": 28, "opacity": 0}
        assert entrance[1]["from"] == {"scaleX": 0} and entrance[1]["to"]["scaleX"] == 1
    elif look == "rule":
        assert entrance[1]["from"] == {"scaleY": 0} and entrance[1]["to"]["scaleY"] == 1
        for event in entrance[2:]:
            assert event["from"] == {"x": -24, "opacity": 0}
            assert event["to"]["x"] == 0
        assert entrance[1]["at"] < entrance[2]["at"] < entrance[3]["at"]
    elif look == "title":
        assert entrance[1]["from"] == {"y": 10, "opacity": 0}
        assert entrance[2]["from"] == {"y": 16, "opacity": 0}
        assert entrance[1]["at"] < entrance[2]["at"]


@pytest.mark.parametrize("look", ["bar", "card", "block", "pill", "underline", "rule", "title"])
def test_text_tag_dom_uses_resolved_look_and_literal_text(look: str, tmp_path: Path) -> None:
    selectors = [".tag-wrapper", ".tag-text", ".subtext"]
    events = _timeline_events("text_tag", 3, tmp_path, look=look, text="<b>Label</b>",
                              subtext="", dom_selectors=selectors)
    elements = {e["target"]: e for e in events if e["kind"] == "dom"}
    assert elements[".tag-wrapper"]["attributes"] == {"data-look": look}
    assert elements[".tag-text"]["text"] == "<b>Label</b>"
    assert elements[".subtext"]["text"] == ""


@pytest.mark.parametrize("duration", [0.032, 3])
def test_checklist_panel_arrives_before_default_rows(duration: float, tmp_path: Path) -> None:
    events = _timeline_events(
        "checklist", duration, tmp_path, items=json.dumps(["First", "Second"])
    )
    entrance = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "panel")
    rows = [e for e in events if e["kind"] == "fromTo" and e["target"] == "row"]
    assert entrance["at"] == 0 and entrance["from"] == {"opacity": 0, "y": 8}
    assert entrance["to"]["opacity"] == 1 and entrance["to"]["y"] == 0
    assert 0 < entrance["to"]["duration"] <= 0.3
    assert rows[0]["at"] == pytest.approx(entrance["to"]["duration"])
    assert rows[0]["at"] < rows[1]["at"]
    if duration == 3:
        assert entrance["to"]["duration"] == 0.3
        assert [row["at"] for row in rows] == pytest.approx([0.3, 0.65])


@pytest.mark.parametrize("frames", [15, 75])
def test_checklist_latest_cue_finishes_rise_before_fade_out(frames: int, tmp_path: Path) -> None:
    duration = hyperframes.layer_duration_s(frames)
    out_start = duration - min(0.3, duration * 0.15)
    latest_cue = out_start - cards.CHECKLIST_REVEAL_S
    plan = cards.card_layer_plan(
        "checklist", {"items": ["First"], "cues": [latest_cue]}, _style(), frames=frames
    )
    events = _timeline_events("checklist", duration, tmp_path, **plan.variables)
    row = next(e for e in events if e["kind"] == "fromTo" and e["target"] == "row")
    assert row["at"] == latest_cue
    assert row["from"]["y"] == 10 and row["to"]["y"] == 0
    assert 0 < row["to"]["duration"] <= cards.CHECKLIST_REVEAL_S
    assert row["at"] + row["to"]["duration"] <= out_start


def _output_events(duration: float, tmp_path: Path, cues: list[float]) -> list[dict]:
    events = _timeline_events(
        "terminal",
        duration,
        tmp_path,
        command="x" if cues else "x" * 44,
        cadence="uniform",
        type_at=0,
        output=json.dumps(["First", "Second", "Third"]),
        cues=json.dumps(cues),
    )
    prints = [e for e in events if e["kind"] == "fromTo" and e["target"] == "output"]
    assert len(prints) == 3
    assert [e["at"] for e in prints] == sorted(e["at"] for e in prints)
    out_start = duration - min(0.3, duration * 0.15)
    assert all(e["to"]["duration"] == 0.12 for e in prints)
    assert all(e["at"] + e["to"]["duration"] <= out_start + 1e-12 for e in prints)
    return prints


def test_terminal_empty_cues_preserve_compressed_output_schedule(tmp_path: Path) -> None:
    short = _output_events(0.7, tmp_path, [])
    times = [e["at"] for e in short]
    assert times[0] < times[1] < times[2]
    assert all(0 < b - a < 0.3 for a, b in zip(times, times[1:], strict=False))
    natural = _output_events(4, tmp_path, [])
    assert [b["at"] - a["at"] for a, b in zip(natural, natural[1:], strict=False)] == (
        pytest.approx([0.3, 0.3])
    )


@pytest.mark.parametrize("cues", [[0.5], [0.5, 0.9], [0.5, 0.5, 0.5]])
def test_terminal_partial_cues_pin_rows_then_follow_previous(
    cues: list[float], tmp_path: Path
) -> None:
    times = [e["at"] for e in _output_events(4, tmp_path, cues)]
    assert times[: len(cues)] == cues
    for i in range(len(cues), len(times)):
        assert times[i] - times[i - 1] == pytest.approx(0.3)


@pytest.mark.parametrize("cues", [[0.88], [0.5, 0.88], [0.88, 0.89, 0.9]])
def test_terminal_late_cues_clamp_prints_before_fade_out(cues: list[float], tmp_path: Path) -> None:
    times = [e["at"] for e in _output_events(1.2, tmp_path, cues)]
    assert times[-1] == pytest.approx(0.9)


def test_terminal_word_cue_slot_and_plan(tmp_path: Path) -> None:
    metadata = cards.load_template("terminal")
    rule = metadata["slots"]["word_cues"]
    assert {k: v for k, v in rule.items() if k != "description"} == {
        "type": "number_list", "required": False, "min_items": 0, "max_items": 12,
        "default": [],
    }
    declaration = next(v for v in metadata["variables"] if v["id"] == "schedule")
    assert declaration == {
        "id": "schedule", "type": "string", "label": "Schedule",
        "default": '{"command":"","output":[],"rows":[[0,0]],"type_start":0.4,"type_end":0.41,'
                   '"exec_at":0.63,"output_at":[]}',
    }
    assert "card_layer_plan" in metadata["notes"]
    assert all(slot in metadata["notes"] for slot in ("command", "output", "timing"))
    assert not {"cadence", "type_at", "cues", "word_cues"}.intersection(
        variable["id"] for variable in metadata["variables"]
    )
    cues = [0.5, 1.0, 1.4, 1.8]
    plan = cards.card_layer_plan("terminal", {
        "command": "open the notes file", "word_cues": cues,
        "type_at": 2, "output": ["Ready"], "cues": [2.3],
    }, _style(), frames=90)
    schedule = json.loads(plan.variables["schedule"])
    assert schedule["type_start"] == cues[0]
    assert schedule["output_at"] == [2.3]
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert hyperframes.preflight(composition, 90) == hyperframes.layer_duration_s(90)


@pytest.mark.parametrize(("frames", "fps"), [(45, 30), (75, 30), (90, 24)])
def test_terminal_word_cue_window_boundaries(frames: int, fps: int) -> None:
    duration = hyperframes.layer_duration_s(frames, fps)
    out_start = duration - min(0.3, duration * 0.15)
    latest = out_start - cards.TERMINAL_WORD_TAIL_S
    assert cards.TERMINAL_WORD_IN_S == 0.3 and cards.TERMINAL_WORD_TAIL_S == 0.5
    for cues in ([], [0.3], [latest], [0.3, latest], [latest, latest]):
        plan = cards.card_layer_plan(
            "terminal", {"command": "go now", "word_cues": cues}, _style(),
            frames=frames, fps=fps,
        )
        schedule = json.loads(plan.variables["schedule"])
        assert schedule["type_start"] == round(cues[0] if cues else 0.4, 6)
    for cues in ([-0.001], [0], [0.299], [0.3, latest + 0.001], [0.5, 0.4]):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan(
                "terminal", {"command": "go now", "word_cues": cues}, _style(),
                frames=frames, fps=fps,
            )
        assert caught.value.parameter == "slots.word_cues"
        assert f"within 0.3..{latest:g}" in caught.value.detail
        if cues == [0.5, 0.4]:
            assert "non-decreasing" in caught.value.detail
        else:
            assert "panel entrance at 0.3 s" in caught.value.detail
            assert "leave 0.5 s" in caught.value.detail


@pytest.mark.parametrize("value", [True, "0.5", math.nan, math.inf, -math.inf, None])
def test_terminal_word_cues_require_finite_numbers(value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("terminal", {"command": "go", "word_cues": [value]}, _style())
    assert caught.value.parameter == "slots.word_cues"
    assert "finite number" in caught.value.detail


@pytest.mark.parametrize("value", ["[0.5]", (0.5,), None, True])
def test_terminal_word_cues_require_a_list(value: object) -> None:
    with pytest.raises(ValidationError, match="expected a list") as caught:
        cards.card_layer_plan("terminal", {"command": "go", "word_cues": value}, _style())
    assert caught.value.parameter == "slots.word_cues"


def test_terminal_word_cue_count_limits() -> None:
    command = " ".join(["x"] * 13)
    plan = cards.card_layer_plan("terminal", {
        "command": command, "word_cues": [0.5] * 12,
    }, _style())
    assert json.loads(plan.variables["schedule"])["type_start"] == 0.5
    with pytest.raises(ValidationError, match=r"0\.\.12 items") as caught:
        cards.card_layer_plan("terminal", {
            "command": command, "word_cues": [0.5] * 13,
        }, _style())
    assert caught.value.parameter == "slots.word_cues"
    with pytest.raises(ValidationError, match=r"0\.\.2 cues.*one per U\+0020-separated word"):
        cards.card_layer_plan("terminal", {
            "command": "  go   now  ", "word_cues": [0.5] * 3,
        }, _style())


@pytest.mark.parametrize("frames", [45, 75, 90])
def test_terminal_word_output_cue_boundaries(frames: int) -> None:
    duration = hyperframes.layer_duration_s(frames)
    out_start = duration - min(0.3, duration * 0.15)
    last = out_start - cards.TERMINAL_WORD_TAIL_S
    latest = out_start - cards.TERMINAL_PRINT_S
    slots = {"command": "go now", "word_cues": [0.3, last], "output": ["Ready"]}
    natural = cards.card_layer_plan("terminal", slots, _style(), frames=frames)
    earliest = json.loads(natural.variables["schedule"])["exec_at"]
    for cue in (earliest, latest):
        plan = cards.card_layer_plan("terminal", {**slots, "cues": [cue]}, _style(), frames=frames)
        assert json.loads(plan.variables["schedule"])["output_at"] == [round(cue, 6)]
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("terminal", {**slots, "cues": [earliest - 0.001]}, _style(),
                              frames=frames)
    assert caught.value.parameter == "slots.cues"
    assert f"execution at {earliest:.6f} s" in caught.value.detail
    with pytest.raises(ValidationError, match="leave 0.12 s before the fade-out"):
        cards.card_layer_plan("terminal", {**slots, "cues": [latest + 0.001]}, _style(),
                              frames=frames)


def test_terminal_word_cues_keep_existing_validation() -> None:
    slots = {"command": "go now", "word_cues": [0.5, 0.7], "output": ["Ready", "Done"]}
    for update, name in (
        ({"type_at": -0.001}, "type_at"), ({"type_at": 2}, "type_at"),
        ({"cues": [1.2, 1.1]}, "cues"), ({"cues": [1, 1, 1]}, "cues"),
    ):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("terminal", {**slots, **update}, _style())
        assert caught.value.parameter == f"slots.{name}"


def _typing_rows(events: list[dict]) -> list[dict]:
    return [e for e in events if e["kind"] == "set" and e["target"] == "text"]


@pytest.mark.parametrize("cadence", ["human", "uniform"])
def test_terminal_empty_word_cues_preserve_typing(cadence: str, tmp_path: Path) -> None:
    variables = {"command": "go now", "type_at": 0.4, "cadence": cadence}
    original = _timeline_events("terminal", 3, tmp_path, **variables)
    events = _timeline_events("terminal", 3, tmp_path, word_cues="[]", **variables)
    assert events == original
    if cadence == "uniform":
        rows = _typing_rows(original)
        assert [e["at"] for e in rows] == pytest.approx([0, 0.45, 0.5, 0.61, 0.66, 0.71, 0.76])
        assert [e["to"]["textContent"] for e in rows] == [
            "", "g", "go", "go ", "go n", "go no", "go now",
        ]


@pytest.mark.parametrize("cadence", ["human", "uniform"])
@pytest.mark.parametrize("cues", [[0.5, 1, 1.4, 1.8], [0.5, 0.5, 0.5, 0.5]])
def test_terminal_word_starts_stay_on_cues(
    cadence: str, cues: list[float], tmp_path: Path,
) -> None:
    command = "open the notes file"
    variables = {
        "command": command, "word_cues": json.dumps(cues), "type_at": 2,
        "cadence": cadence, "output": json.dumps(["First", "Second", "Third"]),
    }
    events = _timeline_events("terminal", 2.999, tmp_path, **variables)
    assert events == _timeline_events("terminal", 2.999, tmp_path, **variables)
    rows = _typing_rows(events)
    assert [e["at"] for e in rows] == sorted(e["at"] for e in rows)
    for word, cue in zip(re.finditer(r"[^ ]+", command), cues, strict=True):
        first = next(e for e in rows if len(e["to"]["textContent"]) > word.start())
        assert first["at"] == cue
    assert rows[-1]["to"]["textContent"] == command
    assert rows[-1]["at"] < 2.699


def test_terminal_word_intervals_compress_proportionally(tmp_path: Path) -> None:
    rows = _typing_rows(_timeline_events("terminal", 4, tmp_path,
        command="longer tiny end", cadence="uniform", word_cues="[0.5, 0.6, 1.0]"))
    factor = (0.6 - 0.5) / (5 * 0.05 + 0.11)
    assert [e["at"] for e in rows[1:7]] == pytest.approx([
        0.5 + i * 0.05 * factor for i in range(6)
    ], abs=0.000001)
    assert [e["at"] for e in rows[7:11]] == pytest.approx([0.6, 0.65, 0.7, 0.75])
    assert rows[7]["to"]["textContent"] == "longer t"
    assert rows[12]["at"] == 1.0


def test_terminal_partial_word_cues_follow_natural_cadence(tmp_path: Path) -> None:
    command = "  go  now "
    rows = _typing_rows(_timeline_events("terminal", 4, tmp_path,
        command=command, cadence="uniform", type_at=2, word_cues="[0.5]"))
    assert [e["at"] for e in rows] == pytest.approx([
        0, 0.5, 0.55, 0.66, 0.71, 0.76, 0.81, 0.92,
    ])
    assert rows[1]["to"]["textContent"] == "  g"
    assert rows[-1]["to"]["textContent"] == command


def test_terminal_last_word_tail_squeeze_keeps_anchors(tmp_path: Path) -> None:
    duration = hyperframes.layer_duration_s(90)
    out_start = duration - 0.3
    last = out_start - cards.TERMINAL_WORD_TAIL_S
    command = "open longestremainingword"
    events = _timeline_events("terminal", duration, tmp_path, command=command,
        cadence="uniform", word_cues=json.dumps([0.5, last]),
        output=json.dumps(["First", "Second", "Third"]))
    rows = _typing_rows(events)
    assert [e["at"] for e in rows[1:5]] == pytest.approx([0.5, 0.55, 0.6, 0.65])
    assert rows[6]["at"] == round(last, 6)
    assert rows[-1]["to"]["textContent"] == command
    assert rows[-1]["at"] < out_start - 0.1
    prints = [e for e in events if e["kind"] == "fromTo" and e["target"] == "output"]
    assert all(e["at"] + e["to"]["duration"] <= out_start for e in prints)
    exec_beat = next(e for e in events if e["kind"] == "set" and e["target"] == "caret"
                     and e["to"]["opacity"] == 0 and e["at"] > rows[-1]["at"])
    assert exec_beat["at"] < prints[0]["at"]


@pytest.mark.parametrize("cadence", ["uniform", "human"])
@pytest.mark.parametrize(
    ("command", "output", "word_cues", "cues", "frames", "type_at"),
    [
        ("go now", [], [], [], 45, 0.4),
        ("x" * 44, ["First", "Second", "Third"], [], [], 45, 0.4),
        ("longer tiny end", [], [0.5, 0.6, 1.0], [], 120, 0.4),
        ("open the notes file", ["First", "Second", "Third"], [0.3, 0.4, 0.6, 0.7], [], 45, 0),
        ("go now", ["Done"], [0.5, 0.9], [1.7], 75, 0.4),
        ("go now", ["First", "Second", "Third"], [], [1.8, 1.8], 75, 0.4),
        ("x" * 44, [], [0.5], [], 240, 0.4),
        ("  go  now ", ["Ready"], [0.5], [], 120, 0.4),
        ("open the notes file", ["Done"], [0.5, 0.5, 0.5, 0.5], [], 90, 0.4),
        ("a b", [], [0.3, 0.300001], [], 45, 0.4),
        ("go\u00a0now", [], [0.5], [], 90, 0.4),
        ("go \U0001f680", ["Done"], [0.5, 1.0], [], 90, 0.4),
        ("x", [], [0.5], [], 45, 0.4),
    ],
)
def test_terminal_schedule_properties(
    cadence: str, command: str, output: list[str], word_cues: list[float], cues: list[float],
    frames: int, type_at: float,
) -> None:
    duration = hyperframes.layer_duration_s(frames)
    plan = cards.card_layer_plan("terminal", {
        "command": command, "output": output, "cadence": cadence, "type_at": type_at,
        "word_cues": word_cues, "cues": cues,
    }, _style(), frames=frames)
    schedule = json.loads(plan.variables["schedule"])
    assert set(schedule) == {
        "command", "output", "rows", "type_start", "type_end", "exec_at", "output_at",
    }
    assert schedule["command"] == command
    assert schedule["output"] == output
    rows = schedule["rows"]
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(rows, rows[1:], strict=False))
    for word, cue in zip(re.finditer(r"[^ ]+", command), word_cues, strict=False):
        first = next(row for row in rows if row[1] > word.start())
        assert first[0] == cue
    assert schedule["type_start"] == (word_cues[0] if word_cues else type_at)
    assert rows[-1][1] == len(command)
    assert rows[-1][0] <= schedule["type_end"] < schedule["exec_at"]
    assert len(schedule["output_at"]) == len(output)
    out_start = duration - min(0.3, duration * 0.15)
    assert all(schedule["exec_at"] <= at for at in schedule["output_at"])
    assert all(at + 0.12 <= out_start + 1e-12 for at in schedule["output_at"])
    assert schedule["output_at"][:len(cues)] == cues
    times = [row[0] for row in rows] + schedule["output_at"] + [
        schedule["type_start"], schedule["type_end"], schedule["exec_at"],
    ]
    assert all(math.isfinite(at) and at == round(at, 6) for at in times)


@pytest.mark.parametrize(
    ("cadence", "rows", "type_end", "exec_at"),
    [
        ("uniform", [[0, 0], [0.45, 1], [0.5, 2], [0.61, 3], [0.66, 4], [0.71, 5], [0.76, 6]],
         0.76, 0.98),
        ("human", [[0, 0], [0.511874, 2], [0.68231, 3], [0.844787, 6]], 0.844787, 1.064787),
    ],
)
def test_terminal_schedule_determinism(
    cadence: str, rows: list[list[float | int]], type_end: float, exec_at: float,
) -> None:
    inputs = ("go now", ["Done"], cadence, 0.4, [], [], 4)
    schedule = cards._terminal_schedule(*inputs)
    assert schedule == cards._terminal_schedule(*inputs)
    assert schedule == {
        "command": "go now", "output": ["Done"],
        "rows": rows, "type_start": 0.4, "type_end": type_end,
        "exec_at": exec_at, "output_at": [round(exec_at + 0.3, 6)],
    }


@pytest.mark.parametrize("cadence", ["uniform", "human"])
def test_terminal_output_cue_waits_for_long_last_word(cadence: str) -> None:
    slots = {"command": "x" * 44, "word_cues": [0.5], "output": ["Done"], "cadence": cadence}
    natural = cards.card_layer_plan("terminal", slots, _style(), frames=75, fps=30)
    schedule = json.loads(natural.variables["schedule"])
    earliest = schedule["exec_at"]
    assert earliest > 0.8
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("terminal", {**slots, "cues": [0.8]}, _style(), frames=75, fps=30)
    assert caught.value.parameter == "slots.cues"
    assert f"execution at {earliest:.6f} s" in caught.value.detail
    reported = float(re.search(r"execution at (\d+\.\d+) s", caught.value.detail)[1])
    accepted = cards.card_layer_plan("terminal", {**slots, "cues": [reported]}, _style(),
                                     frames=75, fps=30)
    accepted_schedule = json.loads(accepted.variables["schedule"])
    assert accepted_schedule["output_at"] == [earliest]
    assert all(accepted_schedule[key] == schedule[key] for key in (
        "rows", "type_start", "type_end", "exec_at",
    ))


@pytest.mark.parametrize("command", ["go\u0085now", "go\u2028now", "go\u2029now"])
def test_terminal_word_controls_refused(command: str) -> None:
    with pytest.raises(ValidationError, match="remove control characters") as caught:
        cards.card_layer_plan("terminal", {"command": command, "word_cues": [0.5]}, _style())
    assert caught.value.parameter == "slots.command"


@pytest.mark.parametrize(
    ("command", "words"), [("go\u00a0now", 1), ("go\u2003now", 1), ("a  b", 2)],
)
def test_terminal_word_boundaries(command: str, words: int) -> None:
    cues = [0.5 + index * 0.3 for index in range(words)]
    plan = cards.card_layer_plan("terminal", {"command": command, "word_cues": cues}, _style())
    rows = json.loads(plan.variables["schedule"])["rows"]
    for word, cue in zip(re.finditer(r"[^ ]+", command), cues, strict=True):
        assert next(row[0] for row in rows if row[1] > word.start()) == cue
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("terminal", {"command": command, "word_cues": cues + [1.2]}, _style())
    assert caught.value.parameter == "slots.word_cues"
    assert f"0..{words} cues" in caught.value.detail


def test_terminal_template_plays_planned_schedule(tmp_path: Path) -> None:
    command = "go \U0001f680"
    schedule = {
        "command": command, "output": ["First", "Second"],
        "rows": [[0, 0], [0.6, 1], [0.7, 2], [1, 4]], "type_start": 0.5,
        "type_end": 1.0, "exec_at": 1.22, "output_at": [1.6, 1.9],
    }
    variables = {"command": command, "output": '["First","Second"]',
                 "schedule": json.dumps(schedule)}
    events = _timeline_events("terminal", 3, tmp_path, **variables)
    assert events == _timeline_events("terminal", 3, tmp_path, **variables)
    rows = _typing_rows(events)
    assert [[event["at"], len(event["to"]["textContent"])] for event in rows] == schedule["rows"]
    assert rows[-1]["to"]["textContent"] == command
    caret = [event for event in events if event["kind"] == "set" and event["target"] == "caret"]
    assert [(event["at"], event["to"]["opacity"]) for event in caret] == [
        (0.5, 1), (0.75, 0), (1.0, 1), (1.22, 0),
    ]
    prints = [event for event in events if event["kind"] == "fromTo"
              and event["target"] == "output"]
    assert [event["at"] for event in prints] == schedule["output_at"]


@pytest.mark.parametrize(("variable", "value"), [
    ("command", "go later"), ("output", '["Changed"]'),
    ("output", '["Second", "First"]'),
])
@pytest.mark.parametrize("operation", ["prepare", "render"])
def test_terminal_preparation_refuses_changed_command_or_output(
    variable: str, value: str, operation: str, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan(
        "terminal", {"command": "go now", "output": ["First", "Second"]}, _style(),
    )
    changed = replace(plan, variables={**plan.variables, variable: value})
    work = tmp_path / "work"
    out = tmp_path / "layer.mov"
    with pytest.raises(ValidationError) as caught:
        if operation == "prepare":
            cards.prepare_card(changed, work)
        else:
            cards.render_card(changed, work, out)
    assert caught.value.parameter == "schedule"
    assert caught.value.detail == (
        "schedule must come from card_layer_plan for the same command and output"
    )
    assert not list(tmp_path.iterdir())


def test_terminal_template_rejects_mismatch_before_timeline() -> None:
    source = (cards.template_dir("terminal") / "template.html").read_text()
    comparison = source.index("schedule.command !== v.command")
    output_comparison = source.index(
        "JSON.stringify(schedule.output) !== JSON.stringify(JSON.parse(v.output))"
    )
    refusal = source.index(
        'throw new Error("schedule must come from card_layer_plan for the same command and output")'
    )
    timeline = source.index("gsap.timeline(")
    registration = source.index('window.__timelines["terminal"] = tl;')
    assert comparison < output_comparison < refusal < timeline < registration


def test_terminal_body_geometry_matches_constants() -> None:
    rules = _css_rules((cards.template_dir("terminal") / "template.html").read_text())
    assert rules[".bar"]["height"] == f"{cards.BAR_HEIGHT}px"
    assert rules[".terminal-body"]["padding"] == (
        f"{cards.TERMINAL_BODY_PAD_Y}px {cards.CONTENT_PAD}px"
    )
    assert float(rules[".command, .output"]["line-height"]) == cards.TERMINAL_LINE_HEIGHT
    assert cards.TERMINAL_MONO_ADVANCE_EM == 0.612


def test_prompt_bar_slot_contract() -> None:
    metadata = cards.load_template("prompt_bar")
    rules = metadata["slots"]
    assert next(v for v in metadata["variables"] if v["id"] == "text_px")["default"] == 52
    source = (cards.template_dir("prompt_bar") / "template.html").read_text()
    assert [attrs for tag, attrs in _HTML(source).elements if tag == "meta"] == [
        {"charset": "utf-8"},
    ]
    assert rules["request"] == {
        "type": "string", "required": True, "min_length": 1, "max_length": 60,
    }
    assert rules["placeholder"] == {
        "type": "string", "required": False, "min_length": 0, "max_length": 40, "default": "",
    }
    assert rules["cadence"]["options"] == ["human", "uniform"]
    assert rules["cadence"]["default"] == "human"
    assert rules["type_at"]["default"] == 0.4
    for name, maximum in (("word_cues", 12), ("cues", 1)):
        assert rules[name]["type"] == "number_list"
        assert (rules[name]["min_items"], rules[name]["max_items"]) == (0, maximum)
        assert rules[name]["default"] == []
        assert rules[name]["description"]


@pytest.mark.parametrize(("name", "value"), [
    ("request", ""), ("request", " "), ("request", "x" * 61), ("request", 12),
    ("placeholder", "x" * 41), ("placeholder", None), ("cadence", "fast"),
    ("type_at", True), ("type_at", math.nan), ("type_at", -0.01), ("type_at", 2.0),
    ("word_cues", "0.5"), ("word_cues", [True]), ("word_cues", [math.inf]),
    ("word_cues", [0.5] * 13), ("word_cues", [0.5, 0.6, 0.7]),
    ("word_cues", [0.6, 0.5]), ("cues", [1, 1]), ("cues", [math.nan]), ("cues", [False]),
])
def test_prompt_bar_slot_refusals(name: str, value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("prompt_bar", {"request": "go now", name: value}, _style())
    assert caught.value.parameter == f"slots.{name}"


@pytest.mark.parametrize("name", ["request", "placeholder"])
def test_prompt_bar_fit_refusal_explains_capacity(name: str) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("prompt_bar", {"request": "go", name: "W" * 40},
                              _style(card_width_px=600))
    assert caught.value.parameter == f"slots.{name}"
    assert re.search(r"shorten to at most \d+ characters", caught.value.detail)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("placeholder", ["", "A longer placeholder", "W" * 24])
def test_prompt_bar_shared_size_is_largest_fit(preset: str, placeholder: str) -> None:
    style = resolve(f"preset:{preset}")
    slots = {"request": "Plan my next week", "placeholder": placeholder}
    plan = cards.card_layer_plan("prompt_bar", slots, style)
    width = style.data["graphics"]["card_width_px"] - 2 * 36 - 76 - 24
    em = max(text_width_em(slots[name], style.data["graphics"]["font"], 500)
             for name in ("request", "placeholder"))
    size = plan.variables["text_px"]
    assert 30 <= size <= 52
    assert em * size + max(3, 0.06 * size) <= width
    assert size == 52 or em * (size + 1) + max(3, 0.06 * (size + 1)) > width


@pytest.mark.parametrize("kind", ["prompt_bar", "comment_cta"])
def test_text_field_caret_geometry(kind: str) -> None:
    rules = _css_rules((cards.template_dir(kind) / "template.html").read_text())
    assert rules[".caret"] == {
        "display": "inline-block", "width": "max(3px, 0.06em)", "height": "1.1em",
        "background": "var(--card-accent)", "vertical-align": "text-bottom",
    }
    assert rules[".placeholder"]["left"] == "0"


@pytest.mark.parametrize("fps", [24, 30])
def test_prompt_bar_minimum_duration_and_hold(fps: int) -> None:
    assert cards.PROMPT_BAR_MIN_S == 2.0
    minimum = math.ceil(cards.PROMPT_BAR_MIN_S * fps)
    with pytest.raises(ValidationError, match="raise graphics.hold_s"):
        cards.card_layer_plan("prompt_bar", {"request": "go"}, _style(),
                              frames=minimum - 1, fps=fps)
    with pytest.raises(ValidationError, match="raise graphics.hold_s"):
        cards.card_layer_plan("prompt_bar", {"request": "go"}, _style(hold_s=1.9), fps=fps)
    plan = cards.card_layer_plan("prompt_bar", {"request": "go"}, _style(hold_s=2), fps=fps)
    assert plan.frames == minimum


@pytest.mark.parametrize("frames", [60, 75, 90])
def test_prompt_bar_word_cue_window(frames: int) -> None:
    duration = hyperframes.layer_duration_s(frames)
    latest = duration - min(0.3, duration * 0.15) - 1.2
    for cue in (0.3, latest):
        plan = cards.card_layer_plan("prompt_bar", {"request": "go", "word_cues": [cue]},
                                     _style(), frames=frames)
        assert json.loads(plan.variables["schedule"])["type_start"] == round(cue, 6)
    for cue in (0.299, latest + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("prompt_bar", {"request": "go", "word_cues": [cue]},
                                  _style(), frames=frames)
        assert caught.value.parameter == "slots.word_cues"
        assert f"within 0.3..{latest:g}" in caught.value.detail


def test_prompt_bar_confirm_cue_window() -> None:
    slots = {"request": "go now"}
    plan = cards.card_layer_plan("prompt_bar", slots, _style(), frames=90)
    schedule = json.loads(plan.variables["schedule"])
    latest = hyperframes.layer_duration_s(90) - 0.3 - 0.7
    for cue in (schedule["type_end"], latest):
        cued = cards.card_layer_plan("prompt_bar", {**slots, "cues": [cue]}, _style(), frames=90)
        assert json.loads(cued.variables["schedule"])["confirm_at"] == cue
    for cue in (-0.1, schedule["type_end"] - 0.001, latest + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("prompt_bar", {**slots, "cues": [cue]}, _style(), frames=90)
        assert caught.value.parameter == "slots.cues"
        assert f"within {schedule['type_end']:g}..{latest:g}" in caught.value.detail


@pytest.mark.parametrize("cadence", ["human", "uniform"])
@pytest.mark.parametrize(("text", "word_cues", "frames"), [
    ("go now", [], 90), ("i" * 60, [], 60), ("  go  now ", [0.3, 0.49], 60),
    ("a b", [0.3, 0.3], 60), ("go \U0001f680", [0.3, 0.5], 90),
])
def test_prompt_bar_schedule_determinism_and_reserve(
    cadence: str, text: str, word_cues: list[float], frames: int,
) -> None:
    slots = {"request": text, "cadence": cadence, "word_cues": word_cues}
    plan = cards.card_layer_plan("prompt_bar", slots, _style(), frames=frames)
    schedule = json.loads(plan.variables["schedule"])
    assert schedule == json.loads(cards.card_layer_plan(
        "prompt_bar", slots, _style(), frames=frames,
    ).variables["schedule"])
    duration = hyperframes.layer_duration_s(frames)
    assert schedule == cards._typing_schedule(text, cadence, 0.4, word_cues, [], duration)
    assert set(schedule) == {"text", "rows", "type_start", "type_end", "confirm_at"}
    assert schedule["text"] == text
    assert schedule["rows"][-1][1] == len(text)
    assert schedule["confirm_at"] == round(schedule["type_end"] + 0.3, 6)
    assert schedule["confirm_at"] + 0.4 + 0.3 <= duration - 0.3 + 1e-6
    assert all(a[0] < b[0] and a[1] < b[1]
               for a, b in zip(schedule["rows"], schedule["rows"][1:], strict=False))
    for word, cue in zip(re.finditer(r"[^ ]+", text), word_cues, strict=False):
        assert next(row[0] for row in schedule["rows"] if row[1] > word.start()) == cue


@pytest.mark.parametrize("schedule", [None, "{}", "[]", '{"text":"changed"}'])
def test_prompt_bar_prepare_refuses_schedule_mismatch(schedule: str | None, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("prompt_bar", {"request": "go now"}, _style())
    changed = replace(plan, variables={**plan.variables, "schedule": schedule})
    with pytest.raises(ValidationError, match="same request") as caught:
        cards.prepare_card(changed, tmp_path / "work")
    assert caught.value.parameter == "schedule"
    assert not list(tmp_path.iterdir())


def test_prompt_bar_prepare_refuses_changed_request(tmp_path: Path) -> None:
    plan = cards.card_layer_plan("prompt_bar", {"request": "go now"}, _style())
    changed = replace(plan, variables={**plan.variables, "request": "go later"})
    with pytest.raises(ValidationError, match="same request"):
        cards.prepare_card(changed, tmp_path / "work")
    assert not list(tmp_path.iterdir())


def test_prompt_bar_template_plays_schedule_and_confirms(tmp_path: Path) -> None:
    schedule = {"text": "go \U0001f680", "rows": [[0, 0], [0.5, 2], [0.8, 4]],
                "type_start": 0.4, "type_end": 0.8, "confirm_at": 1.1}
    variables = {"request": schedule["text"], "schedule": json.dumps(schedule)}
    events = _timeline_events("prompt_bar", 3, tmp_path, **variables)
    assert events == _timeline_events("prompt_bar", 3, tmp_path, **variables)
    rows = _typing_rows(events)
    assert [[event["at"], len(event["to"]["textContent"])] for event in rows] == schedule["rows"]
    placeholder = [event for event in events if event["target"] == "placeholder"
                   and event["kind"] == "set"]
    assert any(event["at"] == 0.5 and event["to"]["opacity"] == 0 for event in placeholder)
    caret = [event for event in events if event["target"] == "caret" and event["kind"] == "set"]
    assert caret[-1]["at"] == 1.1 and caret[-1]["to"]["opacity"] == 0
    check = next(event for event in events if event["target"] == "check"
                 and event["kind"] == "fromTo")
    assert check["at"] == 1.1 and check["to"]["duration"] == 0.4
    assert check["from"]["strokeDashoffset"] == 36 and check["to"]["strokeDashoffset"] == 0
    pulses = [event for event in events if event["target"] == "send"
              and event["kind"] == "fromTo"]
    assert [(event["from"]["scale"], event["to"]["scale"]) for event in pulses] == [
        (1, 1.08), (1.08, 1),
    ]
    assert pulses[0]["at"] == 1.1 and pulses[-1]["at"] + pulses[-1]["to"]["duration"] == 1.5


def test_prompt_bar_template_refuses_mismatch_before_timeline(tmp_path: Path) -> None:
    source = (cards.template_dir("prompt_bar") / "template.html").read_text()
    assert source.index("schedule.text !== v.request") < source.index("gsap.timeline(")
    refusal = 'throw new Error("schedule must come from card_layer_plan for the same request")'
    assert refusal in source
    with pytest.raises(ProcessingError):
        _timeline_events("prompt_bar", 3, tmp_path, request="changed")


@pytest.mark.parametrize(("cadence", "index", "expected"), [
    ("human", 0, "7e4b9443430f59e55899d4f82a5a12ea1079a050f8a04cc13bc1ad6974f4291c"),
    ("human", 1, "dfd7283c98b4cb45fd8bebb3b168ddcbdf1e4cee8091dd2a22a643d09443f91e"),
    ("human", 2, "abef7e441acda1ef0fefb4c0f400662cf220d7f2ffd8fdb36027720ea3e5803f"),
    ("uniform", 0, "034eb8bb75e07290a2ba0d559c79eb62d31e8c7161425aabc2b1d96ff499d1a1"),
    ("uniform", 1, "0d9a830182fb26f76790c53f3bcf8c3a1dae36781fe110d5f80c70c3e53bcc97"),
    ("uniform", 2, "026f246728d8b8ffb4b96b455a709222e80ea1b1dda99317f95f24988f9722f2"),
])
def test_terminal_schedule_unchanged_after_shared_typing_helper(
    cadence: str, index: int, expected: str,
) -> None:
    inputs = [
        ("go now", ["Done"], cadence, 0.4, [], [], 4),
        ("x" * 60, [], cadence, 0.4, [], [], 2),
        ("  go  now ", ["Ready"], cadence, 0.4, [0.5, 1.0], [1.8], 3),
    ]
    schedule = json.dumps(cards._terminal_schedule(*inputs[index])).encode()
    assert hashlib.sha256(schedule).hexdigest() == expected


def test_icon_strike_slot_contract() -> None:
    rules = cards.load_template("icon_strike")["slots"]
    assert rules["icon"] == cards.load_template("comment_drop")["slots"]["avatar"]
    assert rules["glyph"] == {
        "type": "enum", "required": False,
        "options": ["app", "bell", "chat", "mail", "cart", "clock"], "default": "app",
    }
    assert rules["label"] == {
        "type": "string", "required": False, "min_length": 0, "max_length": 40, "default": "",
    }
    assert rules["strike_at"] == {"type": "number", "required": False, "default": 0.8}


@pytest.mark.parametrize(("name", "value"), [
    ("glyph", "unknown"), ("glyph", None), ("label", "W" * 41), ("label", None),
    ("strike_at", True), ("strike_at", math.nan), ("strike_at", math.inf),
    ("icon", None), ("icon", "missing.png"), ("icon", "image.svg"), ("icon", "bad\n.png"),
])
def test_icon_strike_slot_refusals(name: str, value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("icon_strike", {name: value}, _style())
    assert caught.value.parameter == f"slots.{name}"


@pytest.mark.parametrize("glyph", ["app", "bell", "chat", "mail", "cart", "clock"])
def test_icon_strike_glyph_plan_and_timeline(glyph: str, tmp_path: Path) -> None:
    slots = {"glyph": glyph, "label": "Less noise"}
    plan = cards.card_layer_plan("icon_strike", slots, _style())
    assert plan.assets == () and plan.variables["icon_src"] == ""
    assert plan.variables["glyph"] == glyph and plan.variables["label"] == "Less noise"
    assert plan == cards.card_layer_plan("icon_strike", slots, _style())
    events = _timeline_events("icon_strike", 3, tmp_path, **plan.variables)
    assert events == _timeline_events("icon_strike", 3, tmp_path, **plan.variables)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("label", ["", "Less noise", "W" * 22])
def test_icon_strike_label_is_largest_fit(preset: str, label: str) -> None:
    style = resolve(f"preset:{preset}")
    plan = cards.card_layer_plan("icon_strike", {"label": label}, style)
    size = plan.variables["label_px"]
    em = text_width_em(label, style.data["graphics"]["font"], 700)
    width = style.data["graphics"]["card_width_px"] - 48 - 160 - 28
    assert 30 <= size <= 56 and em * size <= width
    assert size == 56 or em * (size + 1) > width
    assert plan.variables["content_width_px"] == math.ceil(
        48 + 160 + (28 + em * size if label else 0)
    )


def test_icon_strike_label_fit_refusal() -> None:
    with pytest.raises(ValidationError, match=r"shorten to at most \d+ characters") as caught:
        cards.card_layer_plan("icon_strike", {"label": "W" * 40}, _style(card_width_px=600))
    assert caught.value.parameter == "slots.label"


def test_icon_strike_without_label_fits_compact_panel() -> None:
    plan = cards.card_layer_plan("icon_strike", {}, _style(card_width_px=208))
    assert plan.variables["content_width_px"] == 208
    with pytest.raises(ValidationError, match="raise graphics.card_width_px"):
        cards.card_layer_plan("icon_strike", {}, _style(card_width_px=207))


@pytest.mark.parametrize("height", [207, 208])
def test_icon_strike_panel_height(height: int) -> None:
    style = _style()
    internal = style.data["internal"]
    internal["key_text_bottom_px"] = internal["key_text_top_px"] + 88 + height
    if height < 208:
        with pytest.raises(ValidationError, match="raise panel height to at least 208 px"):
            cards.card_layer_plan("icon_strike", {}, style)
    else:
        assert cards.card_layer_plan("icon_strike", {}, style)


@pytest.mark.parametrize(("accent", "titlebar", "source"), [
    ("#222222", "#ECE5D6", "accent_color"),
    ("#E9C46A", "#ECE5D6", "bar_ink_color"),
])
def test_icon_strike_uses_readable_strike_color(
    accent: str, titlebar: str, source: str,
) -> None:
    style = _style(accent_color=accent, titlebar_color=titlebar, ink_color="#123456")
    plan = cards.card_layer_plan("icon_strike", {}, style)
    assert plan.variables["strike_color"] == plan.variables[source]
    assert plan.variables["accent_color"] == accent
    assert (cards._contrast(accent, titlebar) >= 3) == (source == "accent_color")
    assert cards._contrast(plan.variables["strike_color"], titlebar) >= 3


def test_icon_strike_white_on_white_uses_contrasting_bar_ink() -> None:
    style = _style(accent_color="#FFFFFF", ink_color="#FFFFFF",
                   titlebar_color="#FFFFFF", paper_color="#FFFFFF")
    plan = cards.card_layer_plan("icon_strike", {}, style)
    assert plan.variables["strike_color"] == plan.variables["bar_ink_color"]
    assert cards._contrast(plan.variables["strike_color"], plan.variables["titlebar_color"]) >= 3


@pytest.mark.parametrize("icon", [None, ""])
def test_icon_strike_optional_image(icon: str | None) -> None:
    slots = {} if icon is None else {"icon": icon}
    plan = cards.card_layer_plan("icon_strike", slots, _style())
    assert plan.assets == () and plan.variables["icon_src"] == ""
    assert plan.variables["glyph"] == "app"


def test_icon_strike_image_asset_path(image: Path, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("icon_strike", {"icon": image, "glyph": "bell"}, _style())
    assert plan.assets == (("icon.png", image.resolve()),)
    assert plan.variables["icon_src"] == "icon.png"
    composition = cards.prepare_card(plan, tmp_path / "work")
    assert (composition / "icon.png").read_bytes() == image.read_bytes()


@pytest.mark.parametrize("filename", ["../icon.png", "other.png", "icon.png\n"])
def test_icon_strike_preparation_refuses_nonlocal_image(filename: str, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("icon_strike", {}, _style())
    changed = replace(plan, variables={**plan.variables, "icon_src": filename})
    with pytest.raises(ValidationError, match="local icon filename"):
        cards.prepare_card(changed, tmp_path / "work")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fps", [24, 30])
def test_icon_strike_minimum_duration(fps: int) -> None:
    assert cards.ICON_STRIKE_MIN_S == 1.7
    minimum = math.ceil(1.7 * fps)
    with pytest.raises(ValidationError, match="raise graphics.hold_s"):
        cards.card_layer_plan("icon_strike", {}, _style(), frames=minimum - 1, fps=fps)
    assert cards.card_layer_plan("icon_strike", {}, _style(),
                                 frames=minimum, fps=fps)


@pytest.mark.parametrize("frames", [51, 60, 90])
def test_icon_strike_time_window(frames: int) -> None:
    duration = hyperframes.layer_duration_s(frames)
    latest = duration - min(0.3, duration * 0.15) - 0.28 - 0.3
    for at in (0.6, latest):
        plan = cards.card_layer_plan("icon_strike", {"strike_at": at}, _style(), frames=frames)
        assert plan.variables["strike_at"] == at
    for at in (0.599, latest + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("icon_strike", {"strike_at": at}, _style(), frames=frames)
        assert caught.value.parameter == "slots.strike_at"
        assert f"use seconds within 0.6..{latest:g}" in caught.value.detail


@pytest.mark.parametrize("fps", [24, 30])
@pytest.mark.parametrize("use_preview", [False, True], ids=["defaults", "preview"])
def test_icon_strike_defaults_fit_minimum_frame_count(fps: int, use_preview: bool) -> None:
    minimum = math.ceil(cards.ICON_STRIKE_MIN_S * fps)
    slots = (json.loads((cards.template_dir("icon_strike") / "preview.json").read_text())["slots"]
             if use_preview else {})
    assert "strike_at" not in slots
    plan = cards.card_layer_plan("icon_strike", slots, _style(), frames=minimum, fps=fps)
    assert plan.frames == minimum and plan.fps == fps
    assert plan.variables["strike_at"] == 0.8
    duration = hyperframes.layer_duration_s(minimum, fps)
    assert plan.variables["strike_at"] + 0.28 + 0.3 <= duration - min(0.3, duration * 0.15)


def test_icon_strike_timeline_draws_after_entrance(tmp_path: Path) -> None:
    events = _timeline_events("icon_strike", 3, tmp_path, strike_at=0.9)
    tile = next(event for event in events if event["target"] == "tile"
                and event["kind"] == "fromTo")
    assert tile["at"] == 0 and tile["from"]["scale"] == 0.9 and tile["to"]["scale"] == 1
    assert tile["to"]["duration"] == 0.6 and tile["to"]["ease"] == "back.out(1.7)"
    slash = next(event for event in events if event["target"] == "slash"
                 and event["kind"] == "fromTo")
    assert slash["at"] == 0.9 and slash["from"]["scaleX"] == 0 and slash["to"]["scaleX"] == 1
    assert slash["to"]["duration"] == 0.28 and slash["to"]["ease"] == "power2.out"
    fade = next(event for event in events if event["target"] == "tile-content"
                and event["kind"] == "fromTo")
    assert fade["at"] == 1.0 and fade["from"]["opacity"] == 1 and fade["to"]["opacity"] == 0.72
    assert fade["to"]["duration"] == 0.2


def test_comment_drop_slot_contract() -> None:
    rules = cards.load_template("comment_drop")["slots"]
    assert rules["text"] == {
        "type": "string", "required": True, "min_length": 1, "max_length": 80,
    }
    for name, maximum in (("name", 32), ("initials", 2)):
        assert rules[name] == {
            "type": "string", "required": False, "min_length": 0,
            "max_length": maximum, "default": "",
        }
    assert rules["avatar"] == {
        "type": "image", "required": False,
        "options": [".png", ".jpg", ".jpeg", ".webp"], "default": "",
    }
    assert rules["side"]["options"] == ["left", "right"]
    assert rules["side"]["default"] == "left" and rules["drop_at"]["default"] == 0.1


@pytest.mark.parametrize(("name", "value"), [
    ("text", ""), ("text", " "), ("text", "x" * 81), ("text", 12),
    ("name", "x" * 33), ("name", None), ("initials", "ABC"), ("initials", 12),
    ("side", "middle"), ("drop_at", True), ("drop_at", math.nan),
    ("avatar", None), ("avatar", "missing.png"), ("avatar", "image.svg"),
    ("avatar", "bad\n.png"),
])
def test_comment_drop_slot_refusals(name: str, value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("comment_drop", {"text": "Send it", "initials": "AB", name: value},
                              _style())
    assert caught.value.parameter == f"slots.{name}"


@pytest.mark.parametrize("avatar", [None, ""])
def test_comment_drop_optional_avatar(avatar: str | None) -> None:
    slots = {"text": "Send it", "initials": "AB"}
    if avatar is not None:
        slots["avatar"] = avatar
    plan = cards.card_layer_plan("comment_drop", slots, _style())
    assert plan.assets == () and plan.variables["avatar_src"] == ""
    assert plan.variables["initials"] == "AB"


@pytest.mark.parametrize("source", ["avatar", "initials", "brand", "name"])
def test_comment_drop_avatar_precedence(source: str, image: Path, tmp_path: Path) -> None:
    style = _style()
    style.data["brand"]["logo"] = str(image) if source in {"avatar", "initials", "brand"} else ""
    slots = {"text": "Send the checklist", "name": "Maya Chen"}
    if source in {"avatar", "initials"}:
        slots["initials"] = "XY"
    if source == "avatar":
        slots["avatar"] = image
    plan = cards.card_layer_plan("comment_drop", slots, style)
    if source in {"avatar", "brand"}:
        assert plan.assets == (("avatar.png", image.resolve()),)
        assert plan.variables["avatar_src"] == "avatar.png"
        prepared = cards.prepare_card(plan, tmp_path / "work")
        assert (prepared / "avatar.png").read_bytes() == image.read_bytes()
    else:
        assert plan.assets == () and plan.variables["avatar_src"] == ""
        assert plan.variables["initials"] == ("XY" if source == "initials" else "MC")


@pytest.mark.parametrize("name,expected", [("Maya", "M"), ("  maya   chen third", "MC")])
def test_comment_drop_name_initials(name: str, expected: str) -> None:
    plan = cards.card_layer_plan("comment_drop", {"text": "Send it", "name": name}, _style())
    assert plan.variables["initials"] == expected


@pytest.mark.parametrize("slots", [{"text": "Send it"}, {"text": "Send it", "name": "  "}])
def test_comment_drop_needs_avatar_identity(slots: dict) -> None:
    with pytest.raises(ValidationError, match=(
        "set slots.initials or slots.name, pass slots.avatar, or set brand.logo"
    )):
        cards.card_layer_plan("comment_drop", slots, _style())


@pytest.mark.parametrize("logo", [
    "relative.png", "missing.png", "bad.svg", "damaged.png", "bad\n.png",
    "~yap-studio-missing-user/logo.png",
])
def test_comment_drop_invalid_brand_logo(logo: str, tmp_path: Path) -> None:
    source = tmp_path / logo
    if logo == "damaged.png":
        source.write_bytes(b"not an image")
    style = _style()
    style.data["brand"]["logo"] = (
        logo if logo == "relative.png" or logo.startswith("~") else str(source)
    )
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("comment_drop", {"text": "Send it", "name": "Maya"}, style)
    assert caught.value.parameter == "brand.logo"
    assert any(word in caught.value.detail for word in ("image", "absolute", ".png"))


def test_comment_drop_damaged_avatar_is_refused(tmp_path: Path) -> None:
    avatar = tmp_path / "avatar.png"
    avatar.write_bytes(b"not an image")
    with pytest.raises(ValidationError, match="readable image") as caught:
        cards.card_layer_plan("comment_drop", {"text": "Yes", "avatar": avatar}, _style())
    assert caught.value.parameter == "slots.avatar"


@pytest.mark.parametrize("filename", ["../avatar.png", "other.png", "avatar.png\n"])
def test_comment_drop_preparation_refuses_nonlocal_avatar(filename: str, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("comment_drop", {"text": "Yes", "initials": "MC"}, _style())
    changed = replace(plan, variables={**plan.variables, "avatar_src": filename})
    with pytest.raises(ValidationError, match="local avatar filename"):
        cards.prepare_card(changed, tmp_path / "work")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["avatar", "initials"])
def test_comment_drop_unused_brand_logo_is_not_validated(source: str, image: Path) -> None:
    style = _style()
    style.data["brand"]["logo"] = "missing.png"
    slots = {"text": "Send it", source: image if source == "avatar" else "AB"}
    assert cards.card_layer_plan("comment_drop", slots, style)


def test_comment_drop_brand_logo_expands_user(image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda value: (
        str(image.parent) if value == "~" else original(value)
    ))
    style = _style()
    style.data["brand"]["logo"] = "~/" + image.name
    assert cards.card_layer_plan("comment_drop", {"text": "Send it"}, style).assets == (
        ("avatar.png", image.resolve()),
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_comment_drop_wrap_fit_and_content_width(preset: str) -> None:
    style = resolve(f"preset:{preset}")
    text = "Send the checklist and I will try this with the whole team tomorrow"
    plan = cards.card_layer_plan("comment_drop", {"text": text, "name": "Maya Chen"}, style)
    lines = json.loads(plan.variables["lines"])
    size = plan.variables["text_px"]
    width = style.data["graphics"]["card_width_px"] - 56 - 72 - 20
    assert 1 <= len(lines) <= 2 and 30 <= size <= 40
    assert " ".join(lines) == text
    assert all(text_width_em(line, style.data["graphics"]["font"], 500) * size <= width
               for line in lines)
    assert plan.variables["bubble_width_px"] <= style.data["graphics"]["card_width_px"]
    short = cards.card_layer_plan("comment_drop", {"text": "Yes", "initials": "AB"}, style)
    assert short.variables["text_px"] == 40
    assert short.variables["bubble_width_px"] < plan.variables["bubble_width_px"]


@pytest.mark.parametrize("text", ["W" * 80, " ".join(["WWWW"] * 16)])
def test_comment_drop_text_fit_refusal(text: str) -> None:
    with pytest.raises(ValidationError, match="shorten the comment to fit two lines at 30 px"):
        cards.card_layer_plan("comment_drop", {"text": text, "initials": "AB"},
                              _style(card_width_px=600))


@pytest.mark.parametrize("text", [
    " Hello world", "Hello world ", "Hello  world", "Hello      world",
])
def test_comment_drop_refuses_ambiguous_spaces(text: str) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("comment_drop", {"text": text, "initials": "AB"},
                              _style(card_width_px=400))
    assert caught.value.parameter == "slots.text"
    assert caught.value.detail == "use single spaces between words, with none at the start or end"


@pytest.mark.parametrize("width", [400, 520, 700, 960])
@pytest.mark.parametrize("text", [
    "Hello world", "Send the checklist", "One two three four", "A B C D E F G H", "Short",
])
def test_comment_drop_single_space_text_round_trips(width: int, text: str) -> None:
    plan = cards.card_layer_plan("comment_drop", {"text": text, "initials": "AB"},
                                 _style(card_width_px=width))
    lines = json.loads(plan.variables["lines"])
    assert 1 <= len(lines) <= 2
    assert " ".join(lines) == text


def test_comment_drop_name_fit_refusal() -> None:
    with pytest.raises(ValidationError, match="shorten to at most") as caught:
        cards.card_layer_plan("comment_drop", {"text": "Yes", "name": "W" * 32},
                              _style(card_width_px=600))
    assert caught.value.parameter == "slots.name"


@pytest.mark.parametrize("fps", [24, 30])
def test_comment_drop_minimum_duration(fps: int) -> None:
    assert cards.COMMENT_DROP_MIN_S == 1.6
    with pytest.raises(ValidationError, match="raise graphics.hold_s"):
        cards.card_layer_plan("comment_drop", {"text": "Yes", "initials": "AB"}, _style(),
                              frames=math.ceil(1.6 * fps) - 1, fps=fps)
    assert cards.card_layer_plan("comment_drop", {"text": "Yes", "initials": "AB"},
                                 _style(hold_s=math.ceil(1.6 * fps) / fps), fps=fps)


@pytest.mark.parametrize("frames", [48, 60, 90])
def test_comment_drop_drop_window(frames: int) -> None:
    duration = hyperframes.layer_duration_s(frames)
    latest = duration - min(0.3, duration * 0.15) - 0.6 - 0.6
    for at in (0, latest):
        slots = {"text": "Yes", "initials": "AB", "drop_at": at}
        plan = cards.card_layer_plan("comment_drop", slots, _style(), frames=frames)
        assert plan.variables["drop_at"] == at
        assert plan == cards.card_layer_plan("comment_drop", slots, _style(), frames=frames)
    for at in (-0.001, latest + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("comment_drop", {**slots, "drop_at": at}, _style(), frames=frames)
        assert caught.value.parameter == "slots.drop_at"
        assert f"use seconds within 0..{latest:g}" in caught.value.detail


@pytest.mark.parametrize("side", ["left", "right"])
def test_comment_drop_side_and_motion(side: str, tmp_path: Path) -> None:
    plan = cards.card_layer_plan("comment_drop", {"text": "Yes", "initials": "AB", "side": side},
                                 _style())
    assert plan.variables["side"] == side
    events = _timeline_events("comment_drop", 3, tmp_path, **plan.variables)
    assert events == _timeline_events("comment_drop", 3, tmp_path, **plan.variables)
    drop = next(event for event in events if event["target"] == "bubble"
                and event["kind"] == "fromTo" and "scale" in event["from"])
    assert drop["at"] == 0.1 and drop["from"]["scale"] == 0.75 and drop["from"]["y"] == -140
    assert drop["to"]["scale"] == 1 and drop["to"]["y"] == 0
    assert drop["to"]["duration"] == 0.6 and drop["to"]["ease"] == "back.out(1.7)"
    opacity = next(event for event in events if event["target"] == "bubble"
                   and event["kind"] == "fromTo" and "opacity" in event["from"])
    assert opacity["at"] == 0.1 and opacity["to"]["duration"] == 0.15


def test_comment_cta_slot_contract() -> None:
    metadata = cards.load_template("comment_cta")
    rules = metadata["slots"]
    assert rules["keyword"] == {
        "type": "string", "required": True, "min_length": 1, "max_length": 32,
    }
    for name, maximum, default in (("placeholder", 32, "Add a comment"), ("label", 40, "")):
        assert rules[name] == {
            "type": "string", "required": False, "min_length": 0,
            "max_length": maximum, "default": default,
        }
    prompt = cards.load_template("prompt_bar")["slots"]
    for name in ("cadence", "type_at", "word_cues"):
        assert rules[name] == prompt[name]
    assert rules["cues"]["max_items"] == 1 and rules["cues"]["default"] == []
    assert "Send time in seconds" in rules["cues"]["description"]
    defaults = {v["id"]: v["default"] for v in metadata["variables"]}
    assert defaults["text_px"] == 44 and defaults["label_px"] == 40
    assert defaults["placeholder"] == "Add a comment"


@pytest.mark.parametrize(("name", "value"), [
    ("keyword", ""), ("keyword", " "), ("keyword", "x" * 33), ("keyword", 12),
    ("placeholder", "x" * 33), ("placeholder", None), ("label", "x" * 41), ("label", None),
    ("cadence", "fast"), ("type_at", True), ("type_at", math.nan), ("type_at", -0.01),
    ("type_at", 2.0), ("word_cues", "0.5"), ("word_cues", [True]),
    ("word_cues", [math.inf]), ("word_cues", [0.5] * 13),
    ("word_cues", [0.5, 0.6, 0.7]), ("word_cues", [0.6, 0.5]),
    ("cues", [1, 1]), ("cues", [math.nan]), ("cues", [False]),
])
def test_comment_cta_slot_refusals(name: str, value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("comment_cta", {"keyword": "go now", name: value}, _style())
    assert caught.value.parameter == f"slots.{name}"


@pytest.mark.parametrize(("name", "text"), [
    ("keyword", "W" * 32), ("placeholder", "W" * 32), ("label", "W" * 40),
])
def test_comment_cta_fit_refusal_explains_capacity(name: str, text: str) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("comment_cta", {"keyword": "go", name: text},
                              _style(card_width_px=600))
    assert caught.value.parameter == f"slots.{name}"
    assert re.search(r"shorten to at most \d+ characters", caught.value.detail)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("placeholder", ["", "Add a comment", "W" * 20])
def test_comment_cta_shared_size_is_largest_fit(preset: str, placeholder: str) -> None:
    style = resolve(f"preset:{preset}")
    slots = {"keyword": "Send the checklist", "placeholder": placeholder}
    plan = cards.card_layer_plan("comment_cta", slots, style)
    width = style.data["graphics"]["card_width_px"] - 2 * 24 - 2 * (28 + 2) - 64 - 16
    em = max(text_width_em(slots[name], style.data["graphics"]["font"], 500)
             for name in ("keyword", "placeholder"))
    size = plan.variables["text_px"]
    assert 30 <= size <= 44
    assert em * size + max(3, 0.06 * size) <= width
    assert size == 44 or em * (size + 1) + max(3, 0.06 * (size + 1)) > width


@pytest.mark.parametrize("label", ["", "Reply below", "W" * 28])
def test_comment_cta_label_fit_and_default_placeholder(label: str) -> None:
    style = _style()
    plan = cards.card_layer_plan("comment_cta", {"keyword": "PLAN", "label": label}, style)
    assert plan.variables["placeholder"] == "Add a comment"
    assert plan.variables["label"] == label
    size = plan.variables["label_px"]
    assert 30 <= size <= 40
    width = style.data["graphics"]["card_width_px"] - 48
    em = text_width_em(label, style.data["graphics"]["font"], 700)
    assert em * size <= width
    assert size == 40 or em * (size + 1) > width


@pytest.mark.parametrize(("label", "height", "size"), [
    ("", 144, 40), ("Reply", 208, 40), ("Reply", 196, 30),
])
def test_comment_cta_panel_height_fit(label: str, height: int, size: int) -> None:
    style = _style()
    internal = style.data["internal"]
    internal["key_text_bottom_px"] = internal["key_text_top_px"] + 88 + height
    plan = cards.card_layer_plan("comment_cta", {"keyword": "PLAN", "label": label}, style)
    assert plan.variables["label_px"] == size
    assert plan.variables["text_px"] == 44


@pytest.mark.parametrize(("label", "height"), [("", 143), ("Reply", 195)])
def test_comment_cta_panel_height_refusal(label: str, height: int) -> None:
    style = _style()
    internal = style.data["internal"]
    internal["key_text_bottom_px"] = internal["key_text_top_px"] + 88 + height
    with pytest.raises(ValidationError, match="raise panel height"):
        cards.card_layer_plan("comment_cta", {"keyword": "PLAN", "label": label}, style)


@pytest.mark.parametrize("fps", [24, 30])
def test_comment_cta_minimum_duration_and_hold(fps: int) -> None:
    assert cards.COMMENT_CTA_MIN_S == 2.0
    minimum = math.ceil(cards.COMMENT_CTA_MIN_S * fps)
    with pytest.raises(ValidationError, match="raise graphics.hold_s"):
        cards.card_layer_plan("comment_cta", {"keyword": "go"}, _style(),
                              frames=minimum - 1, fps=fps)
    with pytest.raises(ValidationError, match="raise graphics.hold_s"):
        cards.card_layer_plan("comment_cta", {"keyword": "go"}, _style(hold_s=1.9), fps=fps)
    assert cards.card_layer_plan("comment_cta", {"keyword": "go"}, _style(hold_s=2),
                                 fps=fps).frames == minimum


@pytest.mark.parametrize("frames", [60, 75, 90])
def test_comment_cta_word_and_send_cue_windows(frames: int) -> None:
    duration = hyperframes.layer_duration_s(frames)
    out_start = duration - min(0.3, duration * 0.15)
    latest_word = out_start - 1.2
    for cue in (0.3, latest_word):
        plan = cards.card_layer_plan("comment_cta", {"keyword": "go", "word_cues": [cue]},
                                     _style(), frames=frames)
        assert json.loads(plan.variables["schedule"])["type_start"] == round(cue, 6)
    for cue in (0.299, latest_word + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("comment_cta", {"keyword": "go", "word_cues": [cue]},
                                  _style(), frames=frames)
        assert caught.value.parameter == "slots.word_cues"
        assert f"within 0.3..{latest_word:g}" in caught.value.detail
    base = cards.card_layer_plan("comment_cta", {"keyword": "go now"}, _style(), frames=frames)
    schedule = json.loads(base.variables["schedule"])
    latest_send = out_start - 0.7
    for cue in (schedule["type_end"], latest_send):
        plan = cards.card_layer_plan("comment_cta", {"keyword": "go now", "cues": [cue]},
                                     _style(), frames=frames)
        assert json.loads(plan.variables["schedule"])["confirm_at"] == cue
    for cue in (-0.1, schedule["type_end"] - 0.001, latest_send + 0.001):
        with pytest.raises(ValidationError) as caught:
            cards.card_layer_plan("comment_cta", {"keyword": "go now", "cues": [cue]},
                                  _style(), frames=frames)
        assert caught.value.parameter == "slots.cues"
        assert f"within {schedule['type_end']:g}..{latest_send:g}" in caught.value.detail


@pytest.mark.parametrize("cadence", ["human", "uniform"])
@pytest.mark.parametrize(("text", "word_cues", "frames"), [
    ("go now", [], 90), ("i" * 32, [], 60), ("  go  now ", [0.3, 0.49], 60),
    ("a b", [0.3, 0.3], 60), ("go \U0001f680", [0.3, 0.5], 90),
])
def test_comment_cta_reuses_typing_schedule(
    cadence: str, text: str, word_cues: list[float], frames: int,
) -> None:
    slots = {"keyword": text, "cadence": cadence, "word_cues": word_cues}
    plan = cards.card_layer_plan("comment_cta", slots, _style(), frames=frames)
    schedule = json.loads(plan.variables["schedule"])
    duration = hyperframes.layer_duration_s(frames)
    assert schedule == cards._typing_schedule(text, cadence, 0.4, word_cues, [], duration)
    assert plan.variables == cards.card_layer_plan(
        "comment_cta", slots, _style(), frames=frames,
    ).variables
    assert schedule["text"] == text and schedule["rows"][-1][1] == len(text)
    assert schedule["confirm_at"] == round(schedule["type_end"] + 0.3, 6)
    assert schedule["confirm_at"] + 0.7 <= duration - min(0.3, duration * 0.15) + 1e-6
    for word, cue in zip(re.finditer(r"[^ ]+", text), word_cues, strict=False):
        assert next(row[0] for row in schedule["rows"] if row[1] > word.start()) == cue


@pytest.mark.parametrize(("name", "value"), [
    ("keyword", "changed"), ("schedule", None), ("schedule", "{}"),
    ("schedule", "[]"), ("schedule", '{"text":"changed"}'),
])
@pytest.mark.parametrize("operation", ["prepare", "render"])
def test_comment_cta_preparation_refuses_schedule_mismatch(
    name: str, value: object, operation: str, tmp_path: Path,
) -> None:
    plan = cards.card_layer_plan("comment_cta", {"keyword": "PLAN"}, _style())
    changed = replace(plan, variables={**plan.variables, name: value})
    with pytest.raises(ValidationError, match="same keyword") as caught:
        if operation == "prepare":
            cards.prepare_card(changed, tmp_path / "work")
        else:
            cards.render_card(changed, tmp_path / "work", tmp_path / "layer.mov")
    assert caught.value.parameter == "schedule"
    assert not list(tmp_path.iterdir())


def test_comment_cta_template_plays_schedule_and_sends(tmp_path: Path) -> None:
    schedule = {"text": "go \U0001f680", "rows": [[0, 0], [0.5, 2], [0.8, 4]],
                "type_start": 0.4, "type_end": 0.8, "confirm_at": 1.1}
    variables = {"keyword": schedule["text"], "schedule": json.dumps(schedule)}
    events = _timeline_events("comment_cta", 3, tmp_path, **variables)
    assert events == _timeline_events("comment_cta", 3, tmp_path, **variables)
    rows = _typing_rows(events)
    assert [[event["at"], len(event["to"]["textContent"])] for event in rows] == schedule["rows"]
    assert any(event["target"] == "placeholder" and event["kind"] == "set"
               and event["at"] == 0.5 and event["to"]["opacity"] == 0 for event in events)
    caret = [event for event in events if event["target"] == "caret" and event["kind"] == "set"]
    assert caret[-1]["at"] == 1.1 and caret[-1]["to"]["opacity"] == 0
    fill = next(event for event in events if event["target"] == "send"
                and event["kind"] == "fromTo" and "backgroundColor" in event["to"])
    assert fill["at"] == 1.1 and fill["to"]["duration"] <= 0.4
    assert fill["to"]["backgroundColor"] == "#365A58"
    assert fill["to"]["color"] == "#FFFFFF"
    pulses = [event for event in events if event["target"] == "send"
              and event["kind"] == "fromTo" and "scale" in event["to"]]
    assert [(event["from"]["scale"], event["to"]["scale"]) for event in pulses] == [
        (1, 1.08), (1.08, 1),
    ]
    assert pulses[0]["at"] == 1.1 and pulses[-1]["at"] + pulses[-1]["to"]["duration"] == 1.5


def test_comment_cta_template_refuses_mismatch_before_timeline(tmp_path: Path) -> None:
    source = (cards.template_dir("comment_cta") / "template.html").read_text()
    assert source.index("schedule.text !== v.keyword") < source.index("gsap.timeline(")
    refusal = 'throw new Error("schedule must come from card_layer_plan for the same keyword")'
    assert refusal in source
    with pytest.raises(ProcessingError):
        _timeline_events("comment_cta", 3, tmp_path, keyword="changed")


def test_stat_reveal_slot_and_plan() -> None:
    metadata = cards.load_template("stat")
    assert metadata["slots"]["reveal"] == {
        "type": "enum", "required": False,
        "options": ["count", "wheel", "slot"], "default": "count",
    }
    assert "ignored by wheel and slot" in metadata["slots"]["start"]["description"]
    reveal = next(v for v in metadata["variables"] if v["id"] == "reveal")
    assert reveal["default"] == "count"
    assert [option["value"] for option in reveal["options"]] == ["count", "wheel", "slot"]
    slots = {"end": 1200, "prefix": "+", "suffix": "%", "kicker": "Growth"}
    plans = [cards.card_layer_plan("stat", {**slots, "reveal": reveal}, _style())
             for reveal in ("count", "wheel", "slot")]
    assert [plan.variables["reveal"] for plan in plans] == ["count", "wheel", "slot"]
    assert plans[0].variables["value_px"] == plans[1].variables["value_px"]
    assert plans[2].variables["slot_digit_em"] == (
        widest_digit_em("Inter", 600) + cards.STAT_SLOT_SAFETY_EM
    )
    for plan in plans[1:]:
        changed_start = cards.card_layer_plan(
            "stat", {**slots, "reveal": plan.variables["reveal"], "start": -999999999}, _style()
        )
        assert changed_start.variables == {**plan.variables, "start": -999999999}
    assert cards.card_layer_plan("stat", {"end": 0}, _style()).variables["reveal"] == "count"
    for invalid in ("roll", "", True, None):
        with pytest.raises(ValidationError, match="slots.reveal"):
            cards.card_layer_plan("stat", {"end": 0, "reveal": invalid}, _style())


def test_stat_wheel_static_contract() -> None:
    source = (cards.template_dir("stat") / "template.html").read_text()
    assert 'v.reveal === "wheel"' in source
    assert "10 + Number(char)" in source
    assert 'end.toLocaleString("en-US")' in source
    assert "document.createTextNode(char)" in source
    assert "step.textContent = String(i % 10)" in source
    rules = _css_rules(source)
    for selector in (".wheel-column", ".wheel-strip", ".value-stage"):
        assert rules[selector]["will-change"] == "transform"
    assert rules[".wheel-column"]["height"] == "1em"
    assert rules[".wheel-column"]["overflow"] == "hidden"
    assert (
        rules[".wheel-strip span"]["height"]
        == rules[".wheel-strip span"]["line-height"] == "1em"
    )


def test_stat_slot_static_contract() -> None:
    source = (cards.template_dir("stat") / "template.html").read_text()
    assert 'v.reveal === "slot"' in source
    assert 'valueElement.classList.add("value-slot")' in source
    assert 'column.className = "slot-column"' in source
    assert 'strip.className = "slot-strip"' in source
    assert 'separator.className = "slot-separator"' in source
    assert 'for (const text of ["0", char])' in source
    assert 'y: "-1em", duration: SLOT_ROLL, ease: "power3.inOut"' in source
    rules = _css_rules(source)
    assert rules[".value-slot"]["display"] == "inline-flex"
    assert rules[".slot-column"]["width"] == "var(--card-slot-digit-width)"
    assert rules[".slot-column"]["flex"] == "0 0 var(--card-slot-digit-width)"
    assert "0.68em" not in source
    assert '`${v.slot_digit_em}em`, "important"' in source
    assert rules[".slot-column"]["height"] == "1em"
    assert rules[".slot-column"]["overflow"] == "hidden"
    assert rules[".slot-strip"]["will-change"] == "transform"
    assert rules[".slot-strip"]["width"] == "100%"
    for selector in (".slot-strip span", ".slot-separator"):
        assert rules[selector]["height"] == rules[selector]["line-height"] == "1em"


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("end", [999999999, -999999999])
def test_stat_slot_fits_nine_digits_and_affixes(preset: str, end: int) -> None:
    slots = {"end": end, "prefix": "USD", "suffix": "units%", "kicker": "Growth"}
    style = _preset(preset)
    slot = cards.card_layer_plan("stat", {**slots, "reveal": "slot"}, style)
    assert slot.variables["value_px"] >= 60
    assert slot.variables["slot_digit_em"] == (
        widest_digit_em(style.data["graphics"]["font"], 600) + cards.STAT_SLOT_SAFETY_EM
    )
    digits = sum(char.isdigit() for char in f"{end:,}")
    units = digits * slot.variables["slot_digit_em"] + (len(f"{end:,}") - digits) * cards.DIGIT_EM
    units += 9 * 0.55 * cards.PROPORTIONAL_EM + 2 * 0.55 * 0.14
    assert units * slot.variables["value_px"] * 1.07 <= style.data["graphics"]["card_width_px"] - 48
    for start in (-999999999, 999999999):
        changed_start = cards.card_layer_plan("stat", {**slots, "reveal": "slot", "start": start},
                                               style)
        assert changed_start.variables == {**slot.variables, "start": start}


@pytest.mark.parametrize(("reveal", "minimum_width"), [("wheel", 701), ("slot", None)])
def test_stat_reel_fit_preserves_sixty_pixel_floor(reveal: str, minimum_width: int | None) -> None:
    slots = {"end": 999999999, "prefix": "USD", "suffix": "units%", "reveal": reveal}
    if minimum_width is None:
        digit_em = widest_digit_em("Inter", 600) + cards.STAT_SLOT_SAFETY_EM
        units = 9 * digit_em + 2 * cards.DIGIT_EM
        units += 9 * 0.55 * cards.PROPORTIONAL_EM + 2 * 0.55 * 0.14
        minimum_width = math.ceil(units * 1.07 * 60 + 48)
    plan = cards.card_layer_plan("stat", slots, _style(card_width_px=minimum_width))
    assert plan.variables["value_px"] == 60
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("stat", slots, _style(card_width_px=minimum_width - 1))
    assert caught.value.parameter == "slots.end"
    assert caught.value.detail == "shorten the formatted value or affixes to fit at 60 px"


def test_stat_slot_reports_extra_reel_width_when_wheel_fits() -> None:
    slots = {"end": 999999999, "prefix": "USD", "suffix": "units%"}
    style = _style(card_width_px=710, font="Titan One")
    wheel = cards.card_layer_plan("stat", {**slots, "reveal": "wheel"}, style)
    assert wheel.variables["value_px"] >= 60
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("stat", {**slots, "reveal": "slot"}, style)
    assert caught.value.parameter == "slots.end"
    assert caught.value.detail == "shorten the formatted value or affixes to fit at 60 px"


def test_stat_slot_titan_one_reel_reserves_widest_digit() -> None:
    style = _style(font="Titan One")
    plan = cards.card_layer_plan("stat", {"end": 1000, "reveal": "slot"}, style)
    width = plan.variables["slot_digit_em"]
    assert width >= 0.726
    assert width == widest_digit_em("Titan One", 600) + cards.STAT_SLOT_SAFETY_EM
    assert all(text_width_em(char, "Titan One", 600) < width for char in "0123456789")
    metadata = cards.load_template("stat")
    source = (cards.template_dir("stat") / "template.html").read_text()
    assert metadata["variables"] == _HTML(source).variables
    variable = next(v for v in metadata["variables"] if v["id"] == "slot_digit_em")
    assert variable["type"] == "number"


@pytest.mark.parametrize("duration", [0.032, 0.4, 3])
@pytest.mark.parametrize("end", [0, 1200, -1200, -999999999])
def test_stat_wheel_digits_land_before_pulse_and_fade(
    duration: float, end: int, tmp_path: Path,
) -> None:
    variables = {"reveal": "wheel", "end": end, "dom_selectors": [".value"]}
    events = _timeline_events("stat", duration, tmp_path, start=987, **variables)
    assert events == _timeline_events("stat", duration, tmp_path, start=-654, **variables)
    children = next(e for e in events if e["kind"] == "dom")["children"]
    formatted = f"{end:,}"
    assert len(children) == len(formatted)
    for char, child in zip(formatted, children, strict=True):
        if not char.isdigit():
            assert child["text"] == char and child["children"] == []
        else:
            assert child["className"] == "wheel-column"
            strip = child["children"][0]
            assert strip["className"] == "wheel-strip"
            assert [step["text"] for step in strip["children"]] == [
                str(i % 10) for i in range(11 + int(char))
            ]
    digits = [int(char) for char in formatted if char.isdigit()]
    out_start = duration - min(0.3, duration * 0.15)
    wheel_end = 0.38 + 1.1 + (len(digits) - 1) * 0.05
    scale = min(1, out_start / (wheel_end + 0.33))
    rolls = [e for e in events if e["kind"] == "fromTo" and e["target"] == "wheel-strip"]
    assert len(rolls) == len(digits)
    assert [e["at"] for e in rolls] == pytest.approx([
        (0.38 + i * 0.05) * scale for i in range(len(digits))
    ])
    for event, digit in zip(rolls, digits, strict=True):
        assert event["from"] == {"y": "0em"}
        assert event["to"]["y"] == f"{-10 - digit}em"
        assert event["to"]["duration"] == pytest.approx(1.1 * scale)
        assert event["to"]["ease"] == "power3.out"
    pulses = [e for e in events if e["kind"] == "fromTo"
              and e["target"] == "value-stage" and e["to"].get("scale") == 1.07]
    assert len(pulses) == 1 and pulses[0]["at"] == pytest.approx(wheel_end * scale)
    assert pulses[0]["at"] == pytest.approx(rolls[-1]["at"] + rolls[-1]["to"]["duration"])
    assert pulses[0]["at"] + 2 * pulses[0]["to"]["duration"] <= out_start + 1e-12
    assert not any(e["kind"] == "set" and e["target"] == "value" for e in events)


def test_stat_count_reveal_keeps_frame_table(tmp_path: Path) -> None:
    variables = {"start": -10, "end": 1200}
    events = _timeline_events("stat", 3, tmp_path, **variables)
    assert events == _timeline_events("stat", 3, tmp_path, reveal="count", **variables)
    rows = [e for e in events if e["kind"] == "set" and e["target"] == "value"]
    assert len(rows) == round(1.72 * 30) + 1
    assert rows[0]["at"] == 0 and rows[0]["to"]["textContent"] == "-10"
    assert rows[-1]["at"] == 1.72 and rows[-1]["to"]["textContent"] == "1,200"
    assert not any(e["target"] == "wheel-strip" for e in events)
    assert not any(e["target"] == "slot-strip" for e in events)


@pytest.mark.parametrize("duration", [0.032, 0.4, 3])
@pytest.mark.parametrize("end", [0, 1200, -1200, 999999999, -999999999])
def test_stat_slot_digits_land_before_pulse_and_fade(
    duration: float, end: int, tmp_path: Path,
) -> None:
    variables = {
        "reveal": "slot", "end": end, "prefix": "USD", "suffix": "units%", "kicker": "Growth",
        "dom_selectors": [".value", ".prefix", ".suffix", ".kicker"],
    }
    events = _timeline_events("stat", duration, tmp_path, start=987, **variables)
    assert events == _timeline_events("stat", duration, tmp_path, start=-654, **variables)
    dom = {e["target"]: e for e in events if e["kind"] == "dom"}
    for selector in ("prefix", "suffix", "kicker"):
        assert dom[f".{selector}"]["text"] == variables[selector]
    children = dom[".value"]["children"]
    formatted = f"{end:,}"
    assert len(children) == len(formatted)
    for char, child in zip(formatted, children, strict=True):
        if not char.isdigit():
            assert child["className"] == "slot-separator"
            assert child["text"] == char and child["children"] == []
        else:
            assert child["className"] == "slot-column"
            strip = child["children"][0]
            assert strip["className"] == "slot-strip"
            assert [step["text"] for step in strip["children"]] == ["0", char]
    digits = [char for char in formatted if char.isdigit()]
    out_start = duration - min(0.3, duration * 0.15)
    slot_end = 0.38 + 0.52 + (len(digits) - 1) * 0.06
    scale = min(1, out_start / (slot_end + 0.33))
    initial = [e for e in events if e["kind"] == "initial" and e["target"] == "slot-strip"]
    assert len(initial) == len(digits)
    assert all(e["to"] == {"y": "0em"} for e in initial)
    rolls = [e for e in events if e["kind"] == "fromTo" and e["target"] == "slot-strip"]
    assert len(rolls) == len(digits)
    assert [e["at"] for e in rolls] == pytest.approx([
        (0.38 + i * 0.06) * scale for i in range(len(digits))
    ])
    for event in rolls:
        assert event["from"] == {"y": "0em"}
        assert event["to"]["y"] == "-1em"
        assert event["to"]["duration"] == pytest.approx(0.52 * scale)
        assert event["to"]["ease"] == "power3.inOut"
    pulses = [e for e in events if e["kind"] == "fromTo"
              and e["target"] == "value-stage" and e["to"].get("scale") == 1.07]
    assert len(pulses) == 1 and pulses[0]["at"] == pytest.approx(slot_end * scale)
    assert pulses[0]["at"] == pytest.approx(rolls[-1]["at"] + rolls[-1]["to"]["duration"])
    assert pulses[0]["at"] + 2 * pulses[0]["to"]["duration"] <= out_start + 1e-12
    assert not any(e["kind"] == "set" and e["target"] == "value" for e in events)
    assert not any(e["kind"] == "fromTo" and e["target"] in {
        "slot-separator", "prefix", "suffix", "kicker", "wheel-strip",
    } for e in events)


ROWS_ON_CUE_SLOTS = {
    "specs_list": {"title": "Plan", "rows": [
        {"name": "First", "value": "12"}, {"name": "Second", "value": "25"}]},
    "stat": {"end": 37, "rows": ["12 first", "25 second"]},
}


def test_specs_rows_style_schema_and_presets() -> None:
    from yap_studio.style import SHIPPED_PRESETS, StyleError

    for preset in SHIPPED_PRESETS:
        assert resolve(f"preset:{preset}").data["graphics"]["specs_rows"] == "together"
    for look in ("together", "on_cue", "on_cue_glow"):
        assert resolve_style_data({"schema_version": 1, "graphics": {"specs_rows": look}})[
            "graphics"]["specs_rows"] == look
    with pytest.raises(StyleError, match="specs_rows"):
        resolve_style_data({"schema_version": 1, "graphics": {"specs_rows": "other"}})


@pytest.mark.parametrize("template,slot", [
    ("specs_list", "cues"), ("stat", "rows"), ("stat", "cues"),
])
def test_specs_rows_slots_refused_when_together(template: str, slot: str) -> None:
    slots = dict(ROWS_ON_CUE_SLOTS[template])
    if template == "stat" and slot == "cues":
        slots.pop("rows")
    if slot == "cues":
        slots["cues"] = []
    with pytest.raises(ValidationError, match="turn on Rows on cue.*or remove") as caught:
        cards.card_layer_plan(template, slots, _style())
    assert caught.value.parameter == f"slots.{slot}"


@pytest.mark.parametrize("template", ["specs_list", "stat"])
def test_specs_rows_off_preserves_variables(template: str) -> None:
    slots = dict(ROWS_ON_CUE_SLOTS[template])
    if template == "stat":
        slots.pop("rows")
    current = cards.card_layer_plan(template, slots, _style())
    legacy = _style()
    legacy.data["graphics"].pop("specs_rows", None)
    assert current.variables == cards.card_layer_plan(template, slots, legacy).variables
    expected = {"specs_list": {"title": "Plan", "rows": json.dumps(slots.get("rows")),
                               "title_px": 40, "row_px": 36},
                "stat": {"end": 37, "start": 0, "prefix": "", "suffix": "", "kicker": "",
                         "reveal": "count", "value_px": 160, "kicker_px": 36,
                         "slot_digit_em": 0.75}}[template]
    assert current.variables == {**cards._tokens(legacy.data["graphics"], 272), **expected}


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
def test_specs_rows_defaults_explicit_and_glow(template: str, look: str) -> None:
    style = _style(specs_rows=look, accent_color="#123456")
    slots = ROWS_ON_CUE_SLOTS[template]
    plan = cards.card_layer_plan(template, slots, style, frames=90)
    latest = hyperframes.layer_duration_s(90) - 0.3 - 0.6
    assert json.loads(plan.variables["cues"]) == pytest.approx([0.3, latest])
    explicit = cards.card_layer_plan(template, {**slots, "cues": [0.5, 1.2]}, style, frames=90)
    assert json.loads(explicit.variables["cues"]) == [0.5, 1.2]
    partial = cards.card_layer_plan(template, {**slots, "cues": [latest]}, style, frames=90)
    assert json.loads(partial.variables["cues"]) == pytest.approx([latest, latest])
    if look == "on_cue_glow":
        assert (plan.variables["text_color"], plan.variables["glow_color"]) == cards.glow_colors(
            "#123456")


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("cues", [[-0.01], [2.101], [1, 0.5], [0, 0.2, 0.4]])
def test_specs_rows_cue_refusals(template: str, cues: list[float]) -> None:
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan(template, {**ROWS_ON_CUE_SLOTS[template], "cues": cues},
                              _style(specs_rows="on_cue"), frames=90)
    assert caught.value.parameter == "slots.cues"
    assert any(fix in caught.value.detail for fix in ("use", "move", "order"))


def test_stat_breakdown_rows_fit_and_refusal() -> None:
    style = _style(specs_rows="on_cue")
    ordinary = cards.card_layer_plan("stat", {"end": 37}, style)
    plan = cards.card_layer_plan("stat", ROWS_ON_CUE_SLOTS["stat"], style, max_height_px=372)
    assert 60 <= plan.variables["value_px"] < ordinary.variables["value_px"]
    assert json.loads(plan.variables["rows"]) == ["12 first", "25 second"]
    with pytest.raises(ValidationError, match="slots.rows.*fewer or shorter rows.*another moment"):
        cards.card_layer_plan("stat", ROWS_ON_CUE_SLOTS["stat"], style, max_height_px=256)
    for rows in ([""], ["x" * 33], ["a", "b", "c"]):
        with pytest.raises(ValidationError, match="slots.rows"):
            cards.card_layer_plan("stat", {"end": 37, "rows": rows}, style)


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
def test_specs_rows_timeline_cues(template: str, look: str, tmp_path: Path) -> None:
    plan = cards.card_layer_plan(template, {**ROWS_ON_CUE_SLOTS[template], "cues": [0.7, 1.4]},
                                 _style(specs_rows=look), frames=90)
    events = _timeline_events(template, hyperframes.layer_duration_s(90), tmp_path,
                              **plan.variables)
    rows = [event for event in events if event["kind"] == "fromTo"
            and event["target"] == "row"]
    assert [event["at"] for event in rows] == [0.7, 1.4]
    for event in rows:
        assert event["from"] == {"opacity": 0, "x": -12}
        assert event["to"]["opacity"] == 1 and event["to"]["x"] == 0
        assert event["to"]["duration"] == 0.3


@pytest.mark.parametrize("template,selector", [
    ("specs_list", ".glow .row-name"), ("stat", ".glow .row, .glow .kicker"),
])
def test_specs_rows_glow_white_text_shadow(template: str, selector: str) -> None:
    rules = _css_rules((cards.template_dir(template) / "template.html").read_text())
    assert rules.get(selector, {}).get("text-shadow") == (
        "var(--card-dark-core), 0 0.03em 0.08em rgb(0 0 0 / 0.6), "
        "0 0 0.35em rgb(0 0 0 / 0.35)")


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
def test_specs_rows_reserve_effect_padding(template: str, look: str) -> None:
    plan = cards.card_layer_plan(template, ROWS_ON_CUE_SLOTS[template],
                                 _style(specs_rows=look))
    main_px = plan.variables["title_px" if template == "specs_list" else "value_px"]
    minimum = math.ceil(main_px * 0.5) if look == "on_cue_glow" else 18
    assert plan.variables["pad_top_px"] >= minimum
    assert plan.variables["pad_bottom_px"] >= minimum
    assert plan.variables["glow_pad_px"] == plan.variables["pad_top_px"]
    assert 280 <= plan.y + plan.variables["pad_top_px"] <= 281
    assert plan.height >= (plan.variables["panel_max_height_px"]
                           + plan.variables["pad_top_px"] + plan.variables["pad_bottom_px"])
    with pytest.raises(ValidationError, match="slots.rows.*fewer or shorter") as caught:
        cards.card_layer_plan(template, ROWS_ON_CUE_SLOTS[template], _style(specs_rows=look),
                              max_height_px=180)
    assert caught.value.detail == (
        "use fewer or shorter rows, or another moment with more room above the head, "
        + ("to fit at a readable size" if template == "specs_list"
           else "to keep the figure and rows at a readable size"))


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("cues", [[], [0]])
def test_specs_rows_short_window_names_a_usable_fix(template: str, cues: list) -> None:
    with pytest.raises(ValidationError, match="slots.cues.*end the card later"):
        cards.card_layer_plan(template, {**ROWS_ON_CUE_SLOTS[template], "cues": cues},
                              _style(specs_rows="on_cue"), frames=15)


@pytest.mark.parametrize("look,bottom", [("paper", 42), ("minimal", 34), ("dark", 42),
                                       ("bright", 42)])
def test_specs_rows_panel_shadow_padding_is_per_side(look: str, bottom: int) -> None:
    plan = cards.card_layer_plan("specs_list", ROWS_ON_CUE_SLOTS["specs_list"],
                                 _style(specs_rows="on_cue", style=look))
    assert plan.variables["pad_top_px"] == 18
    assert plan.variables["pad_bottom_px"] == bottom


def test_stat_breakdown_rows_fit_at_the_figure_floor() -> None:
    style = _style(specs_rows="on_cue")
    plan = cards.card_layer_plan("stat", ROWS_ON_CUE_SLOTS["stat"], style, max_height_px=258)
    assert plan.variables["row_px"] == 44 and plan.variables["value_px"] == 60


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("frames", [40, 127, 135])
def test_specs_rows_default_cues_round_trip_at_the_window_edge(template: str, frames: int) -> None:
    style = _style(specs_rows="on_cue")
    slots = ROWS_ON_CUE_SLOTS[template]
    plan = cards.card_layer_plan(template, slots, style, frames=frames)
    cues = json.loads(plan.variables["cues"])
    explicit = cards.card_layer_plan(template, {**slots, "cues": cues}, style, frames=frames)
    assert explicit.variables == plan.variables


@pytest.mark.parametrize("look,title,row", [("on_cue", 60, 52), ("on_cue_glow", 90, 56)])
def test_specs_rows_phone_sizes_and_compact_height(look: str, title: int, row: int) -> None:
    slots = {"title": "Poetry tiles", "rows": [
        {"name": "Some", "value": "Small words"},
        {"name": "Some", "value": "In between words"}]}
    style = _style(specs_rows=look, font="Geist")
    plan = cards.card_layer_plan("specs_list", slots, style)
    assert (plan.variables["title_px"], plan.variables["row_px"]) == (title, row)
    assert 280 <= plan.y + plan.variables["pad_top_px"] <= 281
    assert plan.height == math.ceil((plan.variables["panel_max_height_px"]
                                     + plan.variables["pad_top_px"]
                                     + plan.variables["pad_bottom_px"]) / 2) * 2
    assert plan.x == (1080 - plan.width) // 2
    constrained = cards.card_layer_plan("specs_list", slots, style, max_height_px=300)
    assert constrained.height <= 300
    assert constrained.variables["title_px"] >= (72 if look == "on_cue_glow" else 52)
    assert constrained.variables["row_px"] >= (48 if look == "on_cue_glow" else 44)
    with pytest.raises(ValidationError, match="slots.rows.*fewer or shorter.*another moment"):
        cards.card_layer_plan("specs_list", slots, style, max_height_px=150)


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look,minimum,maximum", [("on_cue", 44, 52), ("on_cue_glow", 48, 56)])
def test_specs_rows_phone_minimums_and_named_width_refusals(
    template: str, look: str, minimum: int, maximum: int,
) -> None:
    style = _style(specs_rows=look)
    plan = cards.card_layer_plan(template, ROWS_ON_CUE_SLOTS[template], style)
    assert plan.variables["row_px"] == maximum
    constrained = cards.card_layer_plan(template, ROWS_ON_CUE_SLOTS[template], style,
                                       max_height_px=360)
    assert minimum <= constrained.variables["row_px"] <= maximum
    assert constrained.height <= 360
    slots = ({"title": "Plan", "rows": [{"name": "First", "value": "12"},
                                          {"name": "Wide", "value": "W" * 24}]}
             if template == "specs_list" else {"end": 37, "rows": ["First", "W" * 24]})
    with pytest.raises(ValidationError, match="slots.rows.*row 2.*one line") as caught:
        cards.card_layer_plan(template, slots, style, max_width_px=400)
    assert str(minimum) in caught.value.detail
