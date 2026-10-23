"""Semantic case exceptions survive display fixes, look styling and font fitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_captions_render import _TEMPLATES, _caption_dom, _dom_nodes, _style

from yap_studio.captions import CaptionWord, build_phrases, caption_word_case
from yap_studio.stages import captions as stage

CASES = json.loads((Path(__file__).parent / "fixtures/caption-case-cases.json").read_text())
LOOKS = {"pill_karaoke": "lower", "weight_shift": "lower", "clip_wipe": "upper",
         "camera_follow": "upper", "glitch_rgb": "upper", "highlight": "upper",
         "neon_glow": "upper", "particle_burst": "upper"}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["text"])
@pytest.mark.parametrize("look", LOOKS)
def test_caption_case_contract(case: dict, look: str) -> None:
    assert caption_word_case(case["text"], look,
                             preserve_case=case.get("preserve_case", False)) == case[LOOKS[look]]


def _case_plan(look: str) -> stage.LayerPlan:
    style = _style(highlight=look, max_words_per_phrase=6)
    words = [CaptionWord(f"w{i}", text, round(0.2 + i * 0.2, 1), round(0.4 + i * 0.2, 1))
             for i, text in enumerate(["API", "I'll", "jordy", "Hello"])]
    phrases = build_phrases(words, style.data["captions"], word_fixes={"jordy": "Jordan"})
    assert len(phrases) == 1
    return stage.caption_layer_plan(phrases, style, frames=60)


@pytest.mark.parametrize("look", LOOKS)
def test_renderer_keeps_protected_words_before_during_and_after_their_turn(look: str) -> None:
    plan = _case_plan(look)
    slot = json.loads(plan.variables["phrases"])[0]
    assert slot["words"][2] == {
        "id": "w2", "source_ids": ["w2"],
        "text": "Jordan", "start": 0.6, "end": 0.8, "preserve_case": True,
    }
    assert all("preserve_case" not in slot["words"][i] for i in (0, 1, 3))
    dom = _caption_dom(plan, [], seek_times=[0.25, 0.65, 1.5, 0.25])
    assert dom["seeks"][0] == dom["seeks"][-1]
    for snapshot in dom["seeks"]:
        words = _dom_nodes(snapshot, "w")
        assert [word["textContent"] for word in words] == ["API", "I'll", "Jordan", "Hello"]
        assert [word["style"].get("textTransform") for word in words] == [
            "none", "none", "none", None,
        ]


def test_multiword_name_fix_carries_case_with_original_ids_and_timing() -> None:
    style = _style(highlight="pill_karaoke")
    phrases = build_phrases([
        CaptionWord("w0", "tail", 0.2, 0.4), CaptionWord("w1", "net.", 0.4, 0.7),
    ], style.data["captions"], word_fixes={"tail net": "Tailnet"})
    word = phrases[0].words[0]
    assert (word.text, word.source_ids, word.start, word.end, word.preserve_case) == (
        "Tailnet.", ("w0", "w1"), 0.2, 0.7, True,
    )


@pytest.mark.parametrize("replacement", ["Jordan", "iPhone", "API"])
def test_case_sensitive_word_fix_still_preserves_name(replacement: str) -> None:
    style = _style(highlight="pill_karaoke")
    phrase = build_phrases([CaptionWord("w0", "name", 0.2, 0.5)], style.data["captions"],
                          word_fixes={"name": replacement})[0]
    assert phrase.words[0].preserve_case
    plan = stage.caption_layer_plan([phrase], style, frames=30)
    word = _dom_nodes(_caption_dom(plan, []), "w")[0]
    assert word["style"]["textTransform"] == "none"
    assert _dom_nodes(word, "cap-ink")[0]["textContent"] == replacement


def test_explicit_name_marking_survives_phrase_building() -> None:
    style = _style(highlight="pill_karaoke")
    phrases = build_phrases([CaptionWord("w0", "Jordan", 0.2, 0.5, preserve_case=True)],
                            style.data["captions"])
    assert phrases[0].words[0].preserve_case
    plan = stage.caption_layer_plan(phrases, style, frames=30)
    dom = _caption_dom(plan, [])
    word = _dom_nodes(dom, "w")[0]
    assert word["style"]["textTransform"] == "none"
    assert _dom_nodes(word, "cap-ink")[0]["textContent"] == "Jordan"


@pytest.mark.parametrize("look, expected", [("clip_wipe", "SIMPLE"), ("pill_karaoke", "simple")])
def test_ordinary_word_fix_follows_look_through_layer_and_ground_measurement(
    look: str, expected: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    style = _style(highlight=look)
    phrases = build_phrases([CaptionWord("w0", "small", 0.2, 0.5)],
                            style.data["captions"], word_fixes={"small": "simple"})
    word = phrases[0].words[0]
    assert not word.preserve_case
    assert (word.id, word.source_ids, word.start, word.end) == ("w0", ("w0",), 0.2, 0.5)
    plan = stage.caption_layer_plan(phrases, style, frames=30)
    slot = json.loads(plan.variables["phrases"])[0]
    assert "preserve_case" not in slot["words"][0]
    dom = _caption_dom(plan, [])
    assert "textTransform" not in _dom_nodes(dom, "w")[0]["style"]
    assert caption_word_case(word.text, look, preserve_case=word.preserve_case) == expected
    measured = []
    monkeypatch.setattr(stage, "text_width_em",
                        lambda text, font, weight: measured.append(text) or 1)
    stage._phrase_box_width(slot, style.data["captions"])
    assert measured == [expected, " "]


@pytest.mark.parametrize("marked", [0, 1])
def test_merged_word_fix_preserves_explicit_case_on_any_source_word(marked: int) -> None:
    style = _style(highlight="clip_wipe")
    words = [CaptionWord(f"w{i}", text, 0.2 + i * 0.2, 0.4 + i * 0.2,
                         preserve_case=i == marked) for i, text in enumerate(["tail", "net"])]
    phrase = build_phrases(words, style.data["captions"], word_fixes={"tail net": "tailnet"})[0]
    assert phrase.words[0].preserve_case


@pytest.mark.parametrize("look", ["weight_shift", "clip_wipe"])
def test_renderer_measures_the_preserved_case_before_fitting(look: str) -> None:
    source = (_TEMPLATES / "captions/template.html").read_text()
    needle = "return Math.max(0, context.measureText(copy).width"
    assert source.count(needle) == 1
    source = source.replace(needle,
                            "(root.caseMeasurements ||= []).push(copy);\n          " + needle)
    dom = _caption_dom(_case_plan(look), [], source=source)
    assert dom["caseMeasurements"][:4] == [
        "API", "I'll", "Jordan", "hello" if look == "weight_shift" else "HELLO",
    ]


@pytest.mark.parametrize("look", LOOKS)
def test_ground_measurements_use_the_same_case_as_the_renderer(
    look: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _case_plan(look)
    slot = json.loads(plan.variables["phrases"])[0]
    measured = []
    def measure(text: str, font: str, weight: int) -> float:
        measured.append(text)
        return 1
    monkeypatch.setattr(stage, "text_width_em", measure)
    stage._phrase_box_width(slot, _style(highlight=look).data["captions"])
    assert measured == ["API", "I'll", "Jordan", "hello" if LOOKS[look] == "lower" else "HELLO",
                        " "]


def test_slam_retains_its_uppercase_design_even_for_fixed_names() -> None:
    assert caption_word_case("I'll", "slam") == "I'LL"
    assert caption_word_case("Jordan", "slam", preserve_case=True) == "JORDAN"
    assert caption_word_case(".env", "slam") == ".ENV"
    dom = _caption_dom(_case_plan("slam"), [])
    assert all("textTransform" not in word["style"] for word in _dom_nodes(dom, "w"))
