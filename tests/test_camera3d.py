"""Synthetic contracts for the vertical camera caption pilot."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yap_studio import ffmpeg
from yap_studio.checks.model import CHECK_IDS, PILOT_CHECK_IDS, Finding
from yap_studio.errors import ProcessingError
from yap_studio.matte import check as matte_check
from yap_studio.render import camera3d, hyperframes
from yap_studio.render.camera3d import (
    _head_at,
    _sentences,
    build_composition,
    check_camera_coverage,
    check_hero_caption_clearance,
    check_hero_intervals,
    check_hero_repetition,
    measure_silhouette,
    plan_camera3d,
    prepare_plate,
    project_plate_corners,
    render_pilot,
)
from yap_studio.sfx import bleep
from yap_studio.style.loader import ResolvedStyle, resolve_style_data

ROOT = Path(__file__).parents[1]
KIT = ROOT / "src/yap_studio/templates/camera3d"


@pytest.mark.parametrize("frames", [1, 7, 8])
def test_short_cut_keeps_a_luminance_sample(tmp_path: Path, frames: int) -> None:
    cut = tmp_path / "short.mkv"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=gray:s=1080x1920:r=30", "-frames:v", str(frames),
                "-c:v", "ffv1", str(cut)], timeout_s=30)
    luminance = camera3d.sample_cut_luminance(cut, frames / 30)
    assert len(luminance) >= 1
    words = {"time_base": "output", "words": [
        {"id": "w0", "text": "Hi", "start": 0, "end": frames / 30}]}
    plan = plan_camera3d(words, frames / 30, _silhouette()[1], onset_lead=.09,
                         luminance=luminance, hero_free=[(0, frames)])
    assert plan["groups"][0]["off_frame"] <= frames


def test_pilot_check_ids_do_not_extend_verify_checks() -> None:
    assert {"captions.contrast", "captions.words", "picture.grade"}.issubset(CHECK_IDS)
    assert set(PILOT_CHECK_IDS).isdisjoint(CHECK_IDS)
    assert all(Finding(check, "info", "pilot result").check == check for check in PILOT_CHECK_IDS)


def _silhouette() -> tuple[np.ndarray, dict]:
    alpha = np.zeros((1920, 1080), dtype=np.uint8)
    yy, xx = np.ogrid[:1920, :1080]
    alpha[((xx - 540) / 165) ** 2 + ((yy - 460) / 205) ** 2 <= 1] = 255
    alpha[800:1920, 190:890] = 255
    return alpha, measure_silhouette(alpha)


def _words() -> dict:
    tokens = [
        ("Beautiful", 0.35),
        ("stories,", 0.61),
        ("and", 0.88),
        ("better", 1.10),
        ("camera", 1.36),
        ("moves.", 1.70),
        ("We", 2.02),
        ("keep", 2.25),
        ("fucking", 2.45),
        ("going", 2.72),
        ("because", 2.97),
        ("motion", 3.19),
        ("matters.", 3.47),
        ("Another", 4.35),
        ("sentence", 4.58),
        ("ends.", 4.83),
    ]
    return {
        "schema": "yap-studio/cut-words@1",
        "time_base": "output",
        "words": [
            {
                "id": f"w{i}",
                "segment": "s0",
                "text": token,
                "start": start,
                "end": start + 0.17,
                "conf": 1.0,
            }
            for i, (token, start) in enumerate(tokens)
        ],
    }


def _plan(*, bleep_on: bool = True) -> dict:
    _, silhouette = _silhouette()
    data = resolve_style_data({"schema_version": 1, "music": {"bleep_swears": bleep_on}})
    return plan_camera3d(_words(), 5.3, silhouette,
                         style=ResolvedStyle(data["name"], "test:bleep", data))


def _branded_style(caption_font: str, title_font: str, accent: str) -> ResolvedStyle:
    source = json.loads((ROOT / "src/yap_studio/style/presets/clean_minimal.json").read_text())
    source["brand"] = {"captions_font": caption_font, "titles_font": title_font,
                       "accent_color": accent}
    data = resolve_style_data(source)
    return ResolvedStyle(data["name"], "test:brand", data)


def _overlap(a: list[float], b: list[float], gap: float = 0) -> bool:
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def test_sentences_groups_timing_and_screen_text() -> None:
    plan = _plan()
    sentences = plan["sentences"]
    assert len(sentences) == 3
    assert len(_sentences([
        {"text": "hello", "start": 0.1, "end": 0.3},
        {"text": "again", "start": 1.0, "end": 1.2},
    ])) == 2
    assert all(1 <= len(group["word_ids"]) <= 4 for group in plan["groups"])
    assert len(sentences[0]["groups"]) >= 2
    assert plan["groups"][0]["text"] == "stories,"
    assert any("f******" in group["text"] for group in plan["groups"])
    assert plan["bleeps"] == [{"word": "fucking", "start": 2.39, "end": 2.68}]
    assert all(not group["text"].endswith(".") for group in plan["groups"])
    for group in plan["groups"]:
        start = next(
            word["start"] for word in _words()["words"] if word["id"] == group["word_ids"][0]
        )
        assert group["on_frame"] == max(0, 2 * round((start - 0.2) * 15))
        assert group["on_frame"] % 2 == 0
        assert group["fade_steps"] == 3


@pytest.mark.parametrize("target", ["group", "hero"])
def test_bleep_refuses_unmasked_camera_text(target: str, monkeypatch) -> None:
    original = camera3d._plan_camera3d

    def expose(*args, **kwargs):
        plan = original(*args, **kwargs)
        item = (plan["groups"][0] if target == "group" else
                next(sentence["hero"] for sentence in plan["sentences"] if sentence["hero"]))
        item["text"] = "Fuck"
        return plan

    monkeypatch.setattr(camera3d, "_plan_camera3d", expose)
    with pytest.raises(ValueError, match="Bleep check failed"):
        _plan()
    assert _plan(bleep_on=False)


@pytest.mark.parametrize("at", [0, 2.45])
def test_camera_bleeps_pad_zero_length_words(at: float) -> None:
    source = _words()
    if at == 0:
        source["words"][0].update(text="fucking", start=0, end=0)
        source["words"][8]["text"] = "ordinary"
    else:
        source["words"][8].update(start=at, end=at)
    data = resolve_style_data({"schema_version": 1, "music": {"bleep_swears": True}})
    _, silhouette = _silhouette()
    plan = plan_camera3d(source, 5.3, silhouette,
                         style=ResolvedStyle(data["name"], "test:bleep", data))
    span, = plan["bleeps"]
    assert span["word"] == "fucking"
    assert span["start"] == max(0, at - bleep.PAD_S)
    assert span["end"] == at + bleep.PAD_S
    samples = np.full((round(5.3 * bleep.RATE), 2), 12000, dtype=np.int16)
    treated = camera3d.bleep_pcm(samples, bleep.RATE, plan["bleeps"])
    middle = round((span["start"] + span["end"]) / 2 * bleep.RATE)
    assert not np.array_equal(treated[middle:middle + 100], samples[middle:middle + 100])


def test_bleep_off_shows_spoken_words_but_never_chooses_them_as_heroes() -> None:
    plan = _plan(bleep_on=False)
    assert any("fucking" in group["text"] for group in plan["groups"])
    assert plan["bleeps"] == []
    listed = bleep.listed(plan["style"]["data"]["music"])
    assert all(not bleep.is_listed(sentence["hero"]["text"], listed)
               for sentence in plan["sentences"] if sentence["hero"])
    assert camera3d._hero_word([{"text": "FUCKING"}], Counter({"fucking": 9})) is None
    assert camera3d._screen_word("fucking.") == "fucking"


@pytest.mark.parametrize("on", [False, True])
def test_camera_uses_added_and_removed_words(on: bool) -> None:
    data = resolve_style_data({"schema_version": 1, "music": {
        "bleep_swears": on, "bleep_words_add": ["beautiful"],
        "bleep_words_remove": ["fucking"],
    }})
    _, silhouette = _silhouette()
    plan = plan_camera3d(_words(), 5.3, silhouette,
                         style=ResolvedStyle(data["name"], "test:bleep", data))
    assert any(("B********" if on else "Beautiful") in group["text"]
               for group in plan["groups"])
    assert any("fucking" in group["text"] for group in plan["groups"])
    assert plan["bleeps"] == ([{"word": "Beautiful", "start": .29, "end": .52 + bleep.PAD_S}]
                              if on else [])
    assert all(sentence["hero"]["text"].casefold() != "beautiful"
               for sentence in plan["sentences"] if sentence["hero"])
    token = camera3d._ACTIVE_STYLE.set(data)
    try:
        assert camera3d._hero_word([{"text": "Beautiful"}], Counter({"beautiful": 9})) is None
        assert camera3d._hero_word([{"text": "fucking"}], Counter({"fucking": 9})) == "fucking"
        assert camera3d._hero_word([{"text": "Shitake"}], Counter({"shitake": 9})) == "Shitake"
        assert camera3d._hero_word([{"text": "Damn"}], Counter({"damn": 9})) == "Damn"
    finally:
        camera3d._ACTIVE_STYLE.reset(token)


@pytest.mark.parametrize(("first", "second", "mask"), [
    ("bull", "shit", ("b***", "s***")), ("god", "damn", ("g**", "d***")),
])
def test_camera_bleep_metadata_uses_shared_compound_detection(first, second, mask) -> None:
    words = _words()
    words["words"][8]["text"] = first
    words["words"][9]["text"] = second
    data = resolve_style_data({"schema_version": 1, "music": {"bleep_swears": True}})
    _, silhouette = _silhouette()
    plan = plan_camera3d(words, 5.3, silhouette,
                         style=ResolvedStyle(data["name"], "test:bleep", data))
    assert plan["bleeps"] == [{"word": f"{first} {second}", "start": 2.39, "end": 2.95}]
    for index, expected in zip((8, 9), mask, strict=True):
        group, = [group for group in plan["groups"] if f"w{index}" in group["word_ids"]]
        assert expected in group["text"]
    assert all(sentence["hero"]["text"].casefold() not in {first, second}
               for sentence in plan["sentences"] if sentence["hero"])


@pytest.mark.parametrize(("raw", "fixed", "on"), [("ship", "shit", True),
                                                ("shit", "ship", True),
                                                ("ship", "shit", False)])
def test_camera_detection_uses_word_fixes(raw, fixed, on) -> None:
    words = _words()
    words["words"][8]["text"] = raw
    data = resolve_style_data({"schema_version": 1, "music": {"bleep_swears": on}})
    _, silhouette = _silhouette()
    plan = plan_camera3d(words, 5.3, silhouette, word_fixes={raw: fixed},
                         style=ResolvedStyle(data["name"], "test:bleep", data))
    group, = [group for group in plan["groups"] if "w8" in group["word_ids"]]
    assert ("s***" if fixed == "shit" and on else fixed) in group["text"]
    assert plan["bleeps"] == ([{"word": "shit", "start": 2.39, "end": 2.68}]
                              if fixed == "shit" and on else [])


def test_camera_corrected_detection_keeps_reused_source_ids_separate() -> None:
    words = _words()
    words["words"][8]["text"] = "ship"
    words["words"][9]["text"] = "it"
    words["words"][13].update(id="w8", segment="s1", text="ship")
    data = resolve_style_data({"schema_version": 1, "music": {"bleep_swears": True}})
    _, silhouette = _silhouette()
    plan = plan_camera3d(words, 5.3, silhouette, word_fixes={"ship it": "shit"},
                         style=ResolvedStyle(data["name"], "test:bleep", data))
    assert plan["bleeps"] == [{"word": "shit", "start": 2.39, "end": 2.95}]
    assert any("ship" in group["text"] and group["on_frame"] > 100 for group in plan["groups"])


def test_caption_boxes_clear_safe_area_face_and_each_other() -> None:
    plan = _plan()
    face = plan["silhouette"]["head_box"]
    for group in plan["groups"]:
        box = group["box"]
        assert 60 <= box[0] < box[2] <= 1020
        assert 220 <= box[1] < box[3] <= 1500
        assert not _overlap(box, face)
    for i, left in enumerate(plan["groups"]):
        for right in plan["groups"][i + 1 :]:
            visible = (
                left["on_frame"] < right["off_frame"] and right["on_frame"] < left["off_frame"]
            )
            if visible:
                assert not _overlap(left["box"], right["box"], 24)
    for frame in range(plan["frames"]):
        active = []
        for group in plan["groups"]:
            if group["on_frame"] <= frame < group["off_frame"]:
                pose = group["poses"][frame - group["on_frame"]]
                box = pose["box"]
                assert 60 <= box[0] < box[2] <= 1020
                assert 220 <= box[1] < box[3] <= 1445
                head = _head_at(plan["plate"], plan["camera"]["samples"][frame],
                                plan["silhouettes"][group["section"]]["head_box"])
                assert not _overlap(box, head, 55)
                assert all(not _overlap(box, other, 24) for other in active)
                active.append(box)


def test_stabilizer_uses_head_section_active_at_each_frame() -> None:
    _, first = _silhouette()
    second = {**first, "head_box": [50, 840, 1030, 1170], "shoulder_y": 1300}
    sections = [{"start": 0.0, "end": 1.0, **first},
                {"start": 1.0, "end": 5.3, **second}]
    plan = plan_camera3d(_words(), 5.3, sections)
    frame = 45
    active = [group for group in plan["groups"]
              if group["on_frame"] <= frame < group["off_frame"]]
    assert active
    head = _head_at(plan["plate"], plan["camera"]["samples"][frame], second["head_box"])
    assert all(not _overlap(group["poses"][frame - group["on_frame"]]["box"], head, 24)
               for group in active)


def test_heroes_are_behind_and_legible() -> None:
    plan = _plan()
    assert any(sentence["hero"] for sentence in plan["sentences"])
    for sentence in plan["sentences"]:
        hero = sentence["hero"]
        if hero:
            assert hero["placement"] in {"behind", "beside", "above", "front"}
            if hero["layer"] == "behind":
                assert hero["depth"] == plan["plate"]["depth"]
                assert hero["min_visible_share"] >= 0.90
            else:
                assert hero["placement"] == "front"
            assert hero["zone"] in {"masthead", "eye_level"}


def test_hero_choice_favors_repeated_terms_without_sentence_capital_bias() -> None:
    choose = camera3d._hero_word
    assert choose([{"text": "Frontend"}, {"text": "Database"}],
                  Counter({"frontend": 1, "database": 1})) == "Database"
    assert choose([{"text": "Extraordinary"}, {"text": "MCP"}],
                  Counter({"extraordinary": 1, "mcp": 3})) == "MCP"
    assert choose([{"text": "Calculations"}, {"text": "API"}],
                  Counter({"calculation": 1, "api": 3})) == "API"
    assert choose([{"text": "Technical"}, {"text": "Terms"}],
                  Counter({"technical": 2, "term": 2})) == "Terms"
    rejected = ("That's", "Whatever", "everything", "something", "anything",
                "really", "actually", "basically", "essentially", "literally",
                "gonna", "wanna", "think", "yourselves", "Welcome", "where")
    for word in rejected:
        assert choose([{"text": word}], Counter({camera3d._term_key(word): 9})) is None


def test_hero_terms_have_a_cut_wide_limit_and_no_adjacent_repeats() -> None:
    _, silhouette = _silhouette()
    words = []
    for sentence in range(5):
        for offset, text in enumerate(("Database", "connectors.")):
            start = sentence * 1.1 + 0.3 + offset * 0.23
            words.append({"id": f"w{len(words)}", "text": text,
                          "start": start, "end": start + 0.16})
    plan = plan_camera3d({"time_base": "output", "words": words}, 5.6, silhouette)
    heroes = [item["hero"]["text"] for item in plan["sentences"] if item["hero"]]
    folded = [camera3d._term_key(text) for text in heroes]
    assert folded == ["connector", "database", "connector", "database"]
    assert all(folded[i] != folded[i + 1] for i in range(len(folded) - 1))
    assert max(Counter(folded).values()) == 2
    check_hero_repetition(plan["sentences"])

    repeated = [{"hero": {"text": "API"}}, {"hero": {"text": "APIs"}}]
    with pytest.raises(ValueError, match="hero repeats too often"):
        check_hero_repetition(repeated)
    exhausted = [{"hero": {"text": text}} for text in ("API", "MCP", "APIs", "api")]
    with pytest.raises(ValueError, match="hero repeats too often"):
        check_hero_repetition(exhausted)


def test_captions_clear_the_visible_hero_box() -> None:
    plan = _plan()
    check_hero_caption_clearance(plan)
    hero = next(sentence["hero"] for sentence in plan["sentences"] if sentence["hero"])
    group = next(group for group in plan["groups"]
                 if group["on_frame"] < hero["off_frame"]
                 and hero["on_frame"] < group["off_frame"])
    frame = max(group["on_frame"], hero["on_frame"])
    broken = json.loads(json.dumps(plan))
    broken_group = next(item for item in broken["groups"] if item["word_ids"] == group["word_ids"])
    broken_group["poses"][frame - group["on_frame"]]["box"] = camera3d._hero_screen_box(
        hero, plan["camera"]["samples"][frame], frame
    )
    with pytest.raises(ValueError, match="caption overlaps hero"):
        check_hero_caption_clearance(broken)

    roomy = {"head_box": [390, 600, 690, 950], "shoulder_y": 1050}
    box = camera3d._choose_box("the", 84, roomy, [], 0, 30, (420, 500),
                               hero_obstacles=[[0, 440, 1080, 670]])
    assert box is not None and box[1] >= 670


def test_hero_word_replaces_only_its_caption_word() -> None:
    _, silhouette = _silhouette()
    tokens = [("We", 0.25), ("found", 0.46), ("chargers,", 0.72),
              ("and", 0.99), ("wires.", 1.22), ("Done.", 1.7)]
    words = {"time_base": "output", "words": [
        {"id": f"w{i}", "text": token, "start": start, "end": start + 0.15}
        for i, (token, start) in enumerate(tokens)
    ]}
    plan = plan_camera3d(words, 2.3, silhouette)
    hero = plan["sentences"][0]["hero"]
    assert hero is not None and hero["text"] == "chargers"
    assert [group["word_ids"] for group in plan["groups"][:2]] == [
        ["w0", "w1"], ["w3", "w4"]
    ]
    assert [group["text"] for group in plan["groups"][:2]] == ["We found", "and wires"]

    alpha = np.zeros((1920, 1080), dtype=np.uint8)
    alpha[600:950, 390:690] = 255
    alpha[1050:, 190:890] = 255
    lone = {"time_base": "output", "words": [
        {"id": "w0", "text": "Innovation.", "start": 0.4, "end": 0.8}
    ]}
    plan = plan_camera3d(lone, 2.0, measure_silhouette(alpha))
    assert plan["sentences"][0]["hero"] is not None
    assert plan["sentences"][0]["groups"] == []
    assert plan["groups"] == []


def test_repeated_source_word_id_keeps_the_other_caption_occurrence() -> None:
    _, silhouette = _silhouette()
    words = {"time_base": "output", "words": [
        {"id": "repeated", "segment": "first", "text": "camera", "start": 0.3, "end": 0.48},
        {"id": "middle", "segment": "first", "text": "and", "start": 0.54, "end": 0.7},
        {"id": "repeated", "segment": "second", "text": "camera.",
         "start": 0.76, "end": 0.95},
    ]}
    plan = plan_camera3d(words, 1.2, silhouette)
    assert plan["sentences"][0]["hero"]["text"] == "camera"
    assert any("camera" in group["text"] for group in plan["groups"])
    assert sum(group["text"].count("camera") for group in plan["groups"]) == 1


def test_heroes_follow_their_word_and_never_overlap() -> None:
    plan = _plan()
    words = _words()["words"]
    heroes = []
    for index, sentence in enumerate(plan["sentences"]):
        hero = sentence["hero"]
        if hero is None:
            continue
        candidates = [word for word in words if word["text"].strip(".,") == hero["text"]
                      and sentence["start"] <= word["start"] <= sentence["end"]]
        assert candidates
        assert hero["on_frame"] == max(0, 2 * round((candidates[0]["start"] - 0.2) * 15))
        assert hero["off_frame"] - hero["fade_start_frame"] == 6
        assert hero["off_frame"] - hero["on_frame"] <= 90
        assert hero["off_frame"] / 30 <= sentence["end"] + 1 / 30
        if index + 1 < len(plan["sentences"]):
            assert hero["off_frame"] / 30 <= plan["sentences"][index + 1]["start"]
        heroes.append(hero)
    assert all(left["off_frame"] <= right["on_frame"]
               for left, right in zip(heroes, heroes[1:], strict=False))
    broken = json.loads(json.dumps(plan["sentences"]))
    first = next(i for i, sentence in enumerate(broken) if sentence["hero"])
    second = next(i for i in range(first + 1, len(broken)) if broken[i]["hero"])
    broken[second]["hero"]["on_frame"] = broken[first]["hero"]["off_frame"] - 2
    with pytest.raises(ValueError, match="overlaps"):
        check_hero_intervals(broken)


def test_long_sentence_does_not_hold_hero_for_its_whole_length() -> None:
    _, silhouette = _silhouette()
    tokens = ["Remarkable", *["we"] * 12, "go.", "Done."]
    word_list = []
    for i, token in enumerate(tokens):
        start = 0.3 + i * 0.45 if i < 14 else 6.8
        word_list.append({"id": f"w{i}", "text": token,
                          "start": start, "end": start + 0.2})
    words = {"time_base": "output", "words": word_list}
    plan = plan_camera3d(words, 7.5, silhouette)
    hero = plan["sentences"][0]["hero"]
    assert hero is not None
    assert hero["text"] == "Remarkable"
    assert hero["on_frame"] == max(0, 2 * round((0.3 - 0.2) * 15))
    assert hero["off_frame"] - hero["on_frame"] == 90
    assert plan["sentences"][0]["end"] > hero["off_frame"] / 30 + 2


def test_light_background_gets_contrast_halo_and_dark_zone_wins() -> None:
    _, silhouette = _silhouette()
    bright = np.full((11, 480, 270), 205, dtype=np.uint8)
    plan = plan_camera3d(_words(), 5.3, silhouette, luminance=bright)
    assert all(group["background_luma"] > 0.55 for group in plan["groups"])
    assert all(group["halo"] and group["halo"]["contrast_ratio"] >= 4.5
               for group in plan["groups"])
    dark = np.full((11, 480, 270), 38, dtype=np.uint8)
    plan = plan_camera3d(_words(), 5.3, silhouette, luminance=dark)
    assert all(group["halo"]["contrast_ratio"] >= 4.5 for group in plan["groups"])

    roomy = {"head_box": [390, 600, 690, 950], "shoulder_y": 1050}
    box = camera3d._choose_box(
        "hello", 72, roomy, [], 0, 30, (420, 1100),
        luma_probe=lambda candidate: 0.2 if candidate[3] < 600 else 0.8,
    )
    assert box is not None and box[3] < 600


def test_masthead_caption_stays_at_least_64_px() -> None:
    roomy = {"head_box": [390, 600, 690, 950], "shoulder_y": 1050}
    small = camera3d._choose_box("hello", 60, roomy, [], 0, 30, (400, 500))
    large = camera3d._choose_box("hello", 64, roomy, [], 0, 30, (400, 500))
    assert small is not None and small[1] >= 600
    assert large is not None and large[3] <= 600


def test_hero_uses_masthead_when_it_fits() -> None:
    alpha = np.zeros((1920, 1080), dtype=np.uint8)
    alpha[600:950, 390:690] = 255
    alpha[1050:, 190:890] = 255
    silhouette = measure_silhouette(alpha)
    words = {"time_base": "output", "words": [
        {"id": "w0", "text": "Innovation.", "start": 0.4, "end": 0.8}
    ]}
    plan = plan_camera3d(words, 2.0, silhouette)
    hero = plan["sentences"][0]["hero"]
    assert hero is not None
    assert hero["zone"] == "masthead"
    assert hero["placement"] in {"behind", "above", "front"}
    assert hero["min_visible_share"] >= 0.90


def test_occluded_masthead_moves_to_readable_placement() -> None:
    alpha = np.zeros((1920, 1080), dtype=np.uint8)
    alpha[600:1000, 400:680] = 255
    alpha[1200:, 200:880] = 255
    silhouette = measure_silhouette(alpha)
    words = {"time_base": "output", "words": [
        {"id": "w0", "text": "Innovation.", "start": 0.3, "end": 0.7},
    ]}
    plan = plan_camera3d(words, 2.0, silhouette)
    hero = plan["sentences"][0]["hero"]
    assert hero is not None
    assert hero["zone"] == "masthead"
    assert hero["placement"] != "behind" or hero["size_fallback"]
    assert hero["min_visible_share"] >= 0.90


def test_hero_comes_to_front_when_all_behind_positions_are_covered() -> None:
    alpha = np.zeros((1920, 1080), dtype=np.uint8)
    alpha[220:1500, 0:1080] = 255
    silhouette = measure_silhouette(alpha)
    plan = plan_camera3d({"time_base": "output", "words": [
        {"id": "w0", "text": "Camera.", "start": 0.3, "end": 0.7},
    ]}, 2.0, silhouette)
    hero = plan["sentences"][0]["hero"]
    assert hero is not None
    assert hero["layer"] == "front"
    assert hero["placement"] == "front"


def test_brand_changes_font_metrics_colors_and_composition(tmp_path: Path) -> None:
    _, silhouette = _silhouette()
    first = plan_camera3d(_words(), 5.3, silhouette,
                          style=_branded_style("Inter", "Gelasio", "#2F6FED"))
    second = plan_camera3d(_words(), 5.3, silhouette,
                           style=_branded_style("Alata", "Instrument Serif", "#E05020"))
    assert first["font_metrics"] != second["font_metrics"]
    assert first["style"]["data"]["brand"] != second["style"]["data"]["brand"]
    plate, person = tmp_path / "plate.mp4", tmp_path / "person.webm"
    plate.write_bytes(b"synthetic")
    person.write_bytes(b"synthetic")
    for name, plan in (("first", first), ("second", second)):
        build_composition(plan, plate, person, tmp_path / name)
    first_html = (tmp_path / "first/index.html").read_text()
    second_html = (tmp_path / "second/index.html").read_text()
    assert "Gelasio" in first_html and "#2F6FED" in first_html
    assert "Instrument Serif" in second_html and "#E05020" in second_html
    assert first_html != second_html


def test_pilot_style_source_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT))
    from scripts import camera3d_pilot as pilot

    project = tmp_path / "project"
    project.mkdir()
    (project / "project.json").write_text("{}")
    project_style = _branded_style("Inter", "Gelasio", "#112233")
    monkeypatch.setattr(pilot, "ProjectContext", lambda _path: SimpleNamespace(
        render_style=lambda: project_style))
    monkeypatch.setattr(pilot, "get_default", lambda: None)
    assert pilot.select_style(project, None) is project_style
    monkeypatch.setattr(pilot, "get_default", lambda: "preset:clean_minimal")
    assert pilot.select_style(project, None).source == "preset:clean_minimal"
    explicit = pilot.select_style(project, "preset:electric")
    assert explicit.source == "preset:electric"


def test_behind_verify_uses_worst_frame_and_ten_percent_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    info = replace(ffmpeg.probe(_make_tiny_video(tmp_path)), width=20, height=20)
    layer = np.zeros((20, 20, 4), dtype=np.uint8)
    layer[2:12, :, :3] = 255
    layer[2:12, :, 3] = 255
    record = {"id": "word", "path": "word.mov", "x": 0, "y": 0, "width": 20,
              "height": 20, "start_frame": 0, "frames": 2}

    def measure(covered: int):
        matte = np.zeros((20, 20), dtype=np.uint8)
        matte[2:12, :covered] = 255

        def frames(_source: Path, *_args, rgba: bool = False, **_kwargs):
            yield layer if rgba else matte
            yield layer if rgba else np.zeros_like(matte)

        monkeypatch.setattr(matte_check, "_window_frames", frames)
        return matte_check.measure_behind_layers(
            tmp_path, tmp_path / "matte.mkv", info, [record],
            hidden_glyph_share=.10, hidden_time_share=0, report_worst=True,
        )[0]

    assert measure(6)["flag"]  # 30 percent of a word on one frame.
    assert measure(6)["worst_visible_share"] == .7
    assert not measure(1)["flag"]
    assert measure(1)["worst_visible_share"] == .95


def test_rendered_ink_maps_back_to_camera_transformed_matte() -> None:
    camera = [{"cz": 0.0, "px": 0.0, "py": 0.0, "tx": 0.0, "ty": 0.0}]
    ink = np.zeros((20, 20), dtype=bool)
    ink[5:15, 5:15] = True
    matte = np.zeros((1920, 1080), dtype=np.uint8)
    yy, xx = np.nonzero(ink)
    left, top, scale = camera3d._person_transform(camera[0])
    source_x = np.floor((xx + 500.5 - left) / scale).astype(int)
    source_y = np.floor((yy + 500.5 - top) / scale).astype(int)
    matte[source_y, source_x] = 255
    assert camera3d._render_person_covered(ink, matte, 0, 500, 500,
                                            camera=camera) == 100


def _make_tiny_video(tmp_path: Path) -> Path:
    target = tmp_path / "tiny.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=black:s=16x16:r=30", "-frames:v", "2", str(target)],
               timeout_s=30)
    return target


def test_caption_contrast_synthetic_frames() -> None:
    readable = np.full((80, 100, 3), 30, dtype=np.uint8)
    readable[30:50, 30:70] = 255
    faint = np.full((80, 100, 3), 245, dtype=np.uint8)
    faint[30:50, 30:70] = 255
    assert camera3d.caption_contrast(readable, [20, 20, 80, 60], "#FFFFFF") >= 4.5
    assert camera3d.caption_contrast(faint, [20, 20, 80, 60], "#FFFFFF") < 4.5


def test_bleep_pcm_changes_only_named_samples() -> None:
    rate = 48000
    t = np.arange(rate) / rate
    audio = np.rint(4000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)[:, None]
    edited = camera3d.bleep_pcm(audio, rate, [{"start": .25, "end": .45}])
    assert np.array_equal(edited[:12000], audio[:12000])
    assert np.array_equal(edited[21600:], audio[21600:])
    middle = edited[12480:21120, 0].astype(float)
    spectrum = np.abs(np.fft.rfft(middle))
    peak = np.fft.rfftfreq(len(middle), 1 / rate)[np.argmax(spectrum)]
    assert abs(peak - 1000) < 10


def test_plate_covers_all_camera_samples_and_plan_is_stable() -> None:
    plan = _plan()
    assert json.dumps(plan, sort_keys=True, allow_nan=False) == json.dumps(
        _plan(), sort_keys=True, allow_nan=False
    )
    for sample in plan["camera"]["samples"]:
        corners = project_plate_corners(plan["plate"], sample, 1080, 1920)
        assert min(point[0] for point in corners) <= 0
        assert max(point[0] for point in corners) >= 1080
        assert min(point[1] for point in corners) <= 0
        assert max(point[1] for point in corners) >= 1920
    stairs = [group for group in plan["groups"] if group["stair_index"] is not None]
    assert len(stairs) == 2
    assert [group["depth"] for group in stairs] == [1500, 1800]
    assert [group["size"] for group in stairs] == sorted(
        [group["size"] for group in stairs], reverse=True
    )


def test_camera_keeps_real_picture_in_normal_frames() -> None:
    plan = _plan()
    plate = plan["plate"]
    assert plate["scale"] <= 1.12
    first = plan["camera"]["samples"][0]
    assert plate["scale"] * first["m"] == pytest.approx(1, abs=1e-6)
    assert (first["px"], first["py"], first["tx"]) == (0, 0, 0)
    assert max(sample["py"] for sample in plan["camera"]["samples"][:8]) <= 120
    whips = plan["camera"]["whips"]
    assert whips
    allowed = {frame for whip in whips
               for frame in range(whip["start_frame"], whip["end_frame"])}
    assert all(whip["end_frame"] - whip["start_frame"] <= 8 for whip in whips)
    for frame, sample in enumerate(plan["camera"]["samples"][:plan["frames"]]):
        corners = project_plate_corners(plate, sample, 1080, 1920)
        scale = (corners[1][0] - corners[0][0]) / plate["width"]
        real = [corners[0][0] + plate["pad_left"] * scale,
                corners[0][1] + plate["pad_top"] * scale,
                corners[0][0] + (plate["pad_left"] + 1080) * scale,
                corners[0][1] + (plate["pad_top"] + 1920) * scale]
        visible_width = max(0, min(1080, real[2]) - max(0, real[0]))
        visible_height = max(0, min(1920, real[3]) - max(0, real[1]))
        padding = 1 - visible_width * visible_height / (1080 * 1920)
        assert padding <= 0.2 + 1e-6
        assert plate["scale"] * sample["m"] <= (1.25 if frame in allowed else 1.12) + 1e-6
        if frame == 0 or frame not in allowed:
            assert padding <= 1e-6
    for whip in whips:
        assert all(frame in range(whip["start_frame"], whip["end_frame"])
                   for frame in whip["padding_frames"])
        assert len(whip["padding_frames"]) <= 8


def test_caption_planned_size_floor() -> None:
    _, silhouette = _silhouette()
    words = {"time_base": "output", "words": [
        {"id": f"w{i}", "text": "we." if i == 11 else "we",
         "start": 0.3 + i * 0.16, "end": 0.42 + i * 0.16}
        for i in range(12)
    ] + [{"id": "last", "text": "Done.", "start": 2.5, "end": 2.7}]}
    plan = plan_camera3d(words, 3.0, silhouette)
    fallback_ids = {tuple(item["word_ids"]) for item in plan["fallbacks"]}
    assert any(group["size"] == 60 for group in plan["groups"])
    for group in plan["groups"]:
        if tuple(group["word_ids"]) not in fallback_ids:
            assert 60 <= group["size"] <= 84


def test_padding_check_rejects_normal_frame_and_long_or_large_whip() -> None:
    plan = _plan()
    plate = plan["plate"]
    samples = [dict(sample) for sample in plan["camera"]["samples"]]
    samples[0]["px"] += 80
    with pytest.raises(ValueError, match="first frame"):
        check_camera_coverage(plate, samples, plan["camera"]["whips"])
    samples = [dict(sample) for sample in plan["camera"]["samples"]]
    samples[40]["px"] += 150
    with pytest.raises(ValueError, match="outside a whip"):
        check_camera_coverage(plate, samples, plan["camera"]["whips"])
    samples = [dict(sample) for sample in plan["camera"]["samples"]]
    samples[4]["py"] += 630
    with pytest.raises(ValueError, match="20 percent"):
        check_camera_coverage(plate, samples, plan["camera"]["whips"])
    samples = [dict(sample) for sample in plan["camera"]["samples"]]
    for frame in range(1, 10):
        samples[frame]["px"] += 150
    whips = [{"kind": "first", "start_frame": 1, "end_frame": 9},
             {"kind": "second", "start_frame": 9, "end_frame": 17}]
    with pytest.raises(ValueError, match="eight consecutive"):
        check_camera_coverage(plate, samples, whips)


def test_long_groups_split_and_flat_fallback_without_stopping() -> None:
    _, silhouette = _silhouette()
    tokens = ["Hello.", *["Intercontinentalinterpretation"] * 4,
              "PseudopseudohypoparathyroidismX,",
              "Pneumonoultramicroscopicsilicovolcanoconiosis",
              "Pneumonoultramicroscopicsilicovolcanoconiosis.", "Done."]
    words = {"time_base": "output", "words": [
        {"id": f"w{i}", "text": token, "start": 0.3 + i * 0.35,
         "end": 0.45 + i * 0.35}
        for i, token in enumerate(tokens)
    ]}
    plan = plan_camera3d(words, 4.0, silhouette)
    assert plan["fallbacks"]
    assert any(item["action"] == "smaller size" for item in plan["fallbacks"])
    assert any(item["action"] == "split" for item in plan["fallbacks"])
    assert any(item["action"] == "flat torso" for item in plan["fallbacks"])
    assert any(group["flat"] and group["size"] == 44 for group in plan["groups"])
    assert plan["silhouette_summary"][0]["head_box"] == silhouette["head_box"]
    assert plan["silhouette_summary"][0]["shoulder_y"] == silhouette["shoulder_y"]


def test_flat_fallback_never_chooses_an_occupied_box() -> None:
    silhouette = {"head_box": [390, 600, 690, 950], "shoulder_y": 1450}
    prior = {"box": [60, 1450, 1020, 1500], "on_frame": 0, "off_frame": 40}
    assert camera3d._flat_box("another caption", silhouette, [prior], 20, 35) is None

    plan = _plan()
    broken = deepcopy(plan)
    left, right = broken["groups"][:2]
    frame = max(left["on_frame"], right["on_frame"])
    left["poses"][frame - left["on_frame"]]["box"] = right["poses"][
        frame - right["on_frame"]
    ]["box"]
    with pytest.raises(ValueError, match="caption groups overlap"):
        camera3d.check_caption_clearance(broken)


def test_replaced_fade_ends_before_new_caption_or_new_group_is_unplaced() -> None:
    box = [100.0, 1100.0, 500.0, 1144.0]

    def group(start: int, fade: int, end: int, sentence: int) -> dict:
        return {"on_frame": start, "fade_start_frame": fade, "off_frame": end,
                "sentence": sentence, "word_ids": [f"w{sentence}"],
                "poses": [{"frame": frame, "box": box} for frame in range(start, end)]}

    earlier = group(0, 20, 26, 0)
    replacement = group(20, 34, 40, 1)
    fallbacks = []
    camera3d._resolve_caption_conflicts([earlier, replacement], fallbacks, 40)
    assert earlier["fade_start_frame"] == 14
    assert earlier["off_frame"] == replacement["on_frame"]
    assert len(earlier["poses"]) == 20
    assert not fallbacks

    persistent = group(0, 34, 40, 0)
    blocked = group(20, 34, 40, 1)
    fallbacks = []
    camera3d._resolve_caption_conflicts([persistent, blocked], fallbacks, 40)
    assert blocked["unplaced"] is True
    assert fallbacks[0]["action"] == "unplaced"


def test_unplaceable_caption_is_left_out_of_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    _, silhouette = _silhouette()
    monkeypatch.setattr(camera3d, "_choose_box", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(camera3d, "_flat_box", lambda *_args, **_kwargs: None)
    plan = plan_camera3d(_words(), 5.3, silhouette)
    assert plan["groups"] == []
    assert any(item["action"] == "unplaced" for item in plan["fallbacks"])
    assert all(sentence["groups"] == [] for sentence in plan["sentences"])


def test_vendor_provenance_and_names() -> None:
    provenance = json.loads((KIT / "PROVENANCE.json").read_text())
    files = sorted(path.name for path in KIT.iterdir() if path.is_file()
                   and path.name != "PROVENANCE.json")
    assert files == sorted(Path(item["yap_path"]).name for item in provenance["files"])
    assert "hyperframes-community-skills" in (ROOT / "NOTICE").read_text()
    upstream = Path.home() / ".agents/skills/camera-3d-captions"
    if upstream.exists():
        for item in provenance["files"]:
            relative = Path(item["upstream_path"]).relative_to("skills/camera-3d-captions")
            source = upstream / relative
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item["upstream_sha256"]
    forbidden = ("After" + " Effects", "ava" + "tar", "g1" + "-")
    for path in [*KIT.iterdir(), ROOT / "docs/camera3d-pilot.md"]:
        if path.suffix in {".js", ".json", ".md"}:
            contents = path.read_text()
            assert all(
                not re.search(re.escape(term), contents, re.IGNORECASE) for term in forbidden
            )
            assert not re.search(r"\b" + "A" + "E" + r"\b", contents)


def test_builder_is_local_and_passes_preflight(tmp_path: Path) -> None:
    plan = _plan()
    assets = tmp_path / "media"
    assets.mkdir()
    (assets / "plate.mp4").write_bytes(b"synthetic")
    (assets / "person.webm").write_bytes(b"synthetic")
    comp = build_composition(
        plan, assets / "plate.mp4", assets / "person.webm", tmp_path / "comp", gsap_source=None
    )
    assert hyperframes.preflight(comp, plan["frames"]) == plan["render_duration"]
    assert not re.search(r"https?://|//[a-z]+\.", (comp / "index.html").read_text())
    assert "<audio" not in (comp / "index.html").read_text()


@pytest.mark.parametrize("mode", ["full", "picture", "front", "behind"])
def test_composition_modes_share_plan_and_declare_no_variables(tmp_path: Path, mode: str) -> None:
    plan = _plan()
    plate, person = tmp_path / "plate.mp4", tmp_path / "person.webm"
    plate.write_bytes(b"synthetic plate")
    person.write_bytes(b"synthetic person")
    comp = build_composition(plan, plate, person, tmp_path / mode, grain=False, mode=mode)
    html = (comp / "index.html").read_text()
    assert "data-composition-variables='[]'" in html
    assert 'id="finish"' not in html
    assert 'src="assets/finish.js"' not in html
    if mode == "picture":
        assert "#front{display:none!important;}" in html
    elif mode in {"front", "behind"}:
        hidden = "behind" if mode == "front" else "front"
        assert f"#plateClip,#personClip,#{hidden}{{display:none!important;}}" in html
        assert "background:transparent!important" in html


@pytest.mark.parametrize("cz", [-100, 0, 200])
@pytest.mark.parametrize("bottom_band", [False, True])
def test_flat_caption_shadow_stays_in_screen_pixels(tmp_path: Path, cz, bottom_band) -> None:
    import subprocess

    plan = _plan()
    group = deepcopy(plan["groups"][0])
    group.update(flat=True, bottom_band=bottom_band, depth=0 if bottom_band else 1500,
                 on_frame=0, off_frame=plan["frames"], fade_start_frame=plan["frames"] - 6)
    plan["groups"], plan["sentences"] = [group], []
    plan["style"]["data"]["captions"].update(
        shadow_x_px=4, shadow_y_px=3, shadow_blur_px=8, shadow_opacity=.12)
    for sample in plan["camera"]["samples"]:
        sample.update(cz=cz, px=0, py=0, tx=0, ty=0)
    plate, person, gsap = (tmp_path / name for name in ("plate.mp4", "person.webm", "gsap.js"))
    for target in (plate, person, gsap):
        target.write_bytes(b"synthetic")
    comp = build_composition(plan, plate, person, tmp_path / "comp", grain=False, gsap_source=gsap)
    runner = tmp_path / "paint.cjs"
    runner.write_text(r"""
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
class Element {
  constructor() { this.style = {}; this.dataset = {}; this.children = []; }
  appendChild(child) { this.children.push(child); child.parentNode = this; }
  remove() { this.parentNode.children = this.parentNode.children.filter(c => c !== this); }
}
const nodes = Object.fromEntries(['plate', 'plateClip', 'person', 'personClip', 'front', 'behind']
  .map(id => [id, new Element()]));
global.window = global;
global.document = {createElement: () => new Element(), getElementById: id => nodes[id]};
global.gsap = {timeline: () => ({to() {}})};
for (const name of ['cam3d.js', 'tables.js', 'metrics.js', 'plan.js', 'paint.js']) {
  vm.runInThisContext(fs.readFileSync(path.join(process.argv[2], 'assets', name), 'utf8'));
}
window.__mainPaint(1);
process.stdout.write(JSON.stringify(nodes.front.children[0].style.filter));
""", encoding="utf-8")
    result = subprocess.run(["node", str(runner), str(comp)], capture_output=True,
                             text=True, check=True, timeout=30)
    shadow = json.loads(result.stdout).strip()
    match = re.fullmatch(r"drop-shadow\(([-.\d]+)px ([-.\d]+)px ([-.\d]+)px "
                          r"rgba\(0,0,0,([.\d]+)\)\)", shadow)
    assert match, shadow
    assert tuple(map(float, match.groups())) == pytest.approx((4, 3, 8, .12))


def test_prepare_plate_uses_and_invalidates_shared_matte(tmp_path: Path, monkeypatch) -> None:
    cut, matte = tmp_path / "cut.mp4", tmp_path / "shared.mkv"
    cut.write_bytes(b"synthetic cut")
    matte.write_bytes(b"synthetic matte")
    info = SimpleNamespace(width=1080, height=1920, fps=30, video_frames=2,
                           variable_frame_rate=False)
    monkeypatch.setattr(ffmpeg, "probe", lambda *a, **kw: info)
    monkeypatch.setattr(camera3d, "_check_media", lambda *a, **kw: None)
    monkeypatch.setattr(camera3d, "make_matte",
                        lambda *a, **kw: pytest.fail("shared prep made another matte"))
    commands = []

    def run(command, **options):
        commands.append(command)
        Path(command[-1]).write_bytes(b"prepared picture")

    monkeypatch.setattr(ffmpeg, "run", run)
    out = tmp_path / "prep"
    plate, person, gray = prepare_plate(cut, out, matte=matte)
    assert gray == matte and plate.is_file() and person.is_file()
    assert len(commands) == 2
    assert prepare_plate(cut, out, matte=matte) == (plate, person, matte)
    assert len(commands) == 2
    matte.write_bytes(b"changed shared matte")
    prepare_plate(cut, out, matte=matte)
    assert len(commands) == 4
    record = json.loads((out / "prep-record.json").read_bytes())
    assert record["matte_sha256"] == hashlib.sha256(matte.read_bytes()).hexdigest()


def test_prepare_plate_keeps_synthetic_media_aligned(tmp_path: Path, monkeypatch) -> None:
    cut = tmp_path / "cut.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=c=blue:s=1080x1920:r=30", "-frames:v", "30",
                "-c:v", "libx264", "-preset", "ultrafast", str(cut)], timeout_s=90)

    def synthetic_matte(_source: Path, folder: Path) -> SimpleNamespace:
        folder.mkdir(exist_ok=True)
        ffmpeg.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=white:s=1080x1920:r=30,format=gray", "-frames:v", "30",
                    "-c:v", "ffv1", "-pix_fmt", "gray", str(folder / "matte.mkv")],
                   timeout_s=90)
        return SimpleNamespace(status="measured")

    monkeypatch.setattr(camera3d, "make_matte", synthetic_matte)
    stages: list[str] = []
    plate, person, gray = prepare_plate(cut, tmp_path / "prep", progress=stages.append)
    assert stages == ["matte", "plate", "person video"]
    assert gray.is_file()
    for path, size in ((plate, (1560, 2640)), (person, (1080, 1920))):
        info = ffmpeg.probe(path, count_frames=True)
        assert (info.width, info.height, info.video_frames, info.fps) == (*size, 30, 30)
    alpha = ffmpeg.run(["ffmpeg", "-v", "error", "-c:v", "libvpx-vp9", "-i",
                        str(person), "-vf", "alphaextract,format=gray", "-frames:v", "1",
                        "-f", "rawvideo", "pipe:1"], timeout_s=60, binary=True).stdout
    assert np.frombuffer(alpha, dtype=np.uint8).mean() > 250
    reused: list[str] = []
    assert prepare_plate(cut, tmp_path / "prep", progress=reused.append) == (plate, person, gray)
    assert reused == ["prep reused"]
    record = tmp_path / "prep/prep-record.json"
    assert record.is_file()
    invalid = json.loads(record.read_text())
    invalid["cut_sha256"] = "0" * 64
    record.write_text(json.dumps(invalid))
    rebuilt: list[str] = []
    prepare_plate(cut, tmp_path / "prep", progress=rebuilt.append)
    assert rebuilt == ["matte", "plate", "person video"]


def test_render_pilot_passes_worker_and_ready_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RenderCalled(Exception):
        pass

    captured: list[str] = []

    def capture(_executable: Path, args: list[str], *, timeout_s: float) -> None:
        captured.extend(args)
        assert timeout_s > 240
        raise RenderCalled

    monkeypatch.setattr(hyperframes, "preflight", lambda _path, _frames: None)
    monkeypatch.setattr(
        hyperframes, "ensure_installed", lambda: SimpleNamespace(executable=tmp_path)
    )
    monkeypatch.setattr(hyperframes, "_run_renderer", capture)
    with pytest.raises(RenderCalled):
        render_pilot(tmp_path, tmp_path / "cut.mp4", tmp_path / "out.mp4", 90)
    assert captured[captured.index("--workers") + 1] == str(hyperframes.RENDER_WORKERS)
    assert "--experimental-fast-capture=false" in captured
    captured.clear()
    with pytest.raises(RenderCalled):
        render_pilot(tmp_path, tmp_path / "cut.mp4", tmp_path / "out.mp4", 90, workers=2)
    assert captured[captured.index("--workers") + 1] == "2"
    assert "--experimental-fast-capture=false" in captured
    assert captured[captured.index("--player-ready-timeout") + 1] == "240000"
    with pytest.raises(ValueError, match="workers must be positive"):
        render_pilot(tmp_path, tmp_path / "cut.mp4", tmp_path / "out.mp4", 90, workers=0)


@pytest.mark.parametrize("error", [
    "ERR_INVALID_HTTP_RESPONSE", "ERR_CONNECTION_RESET", "ERR_EMPTY_RESPONSE",
])
def test_render_pilot_retries_local_server_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    error: str,
) -> None:
    detail = f"net::{error} at http://localhost:57344/index.html"
    calls = []

    def render(_executable: Path, args: list[str], *, timeout_s: float) -> None:
        calls.append(args)
        if len(calls) == 1:
            raise ProcessingError("hyperframes", 1, f"Render failed\n   {detail}\n")
        Path(args[args.index("-o") + 1]).write_bytes(b"synthetic picture")

    def mux(command: list[str], *, timeout_s: float) -> None:
        assert (tmp_path / "camera3d-picture.mp4").read_bytes() == b"synthetic picture"
        Path(command[-1]).write_bytes(b"synthetic output")

    monkeypatch.setattr(hyperframes, "preflight", lambda _path, _frames: None)
    monkeypatch.setattr(
        hyperframes, "ensure_installed", lambda: SimpleNamespace(executable=tmp_path)
    )
    monkeypatch.setattr(hyperframes, "_run_renderer", render)
    monkeypatch.setattr(camera3d, "_check_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ffmpeg, "probe", lambda *_args, **_kwargs:
                        SimpleNamespace(has_audio=False, audio_codec=None))
    monkeypatch.setattr(ffmpeg, "run", mux)
    out = tmp_path / "out.mp4"
    assert render_pilot(tmp_path / "composition", tmp_path / "cut.mp4", out, 90) == out
    assert out.read_bytes() == b"synthetic output"
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert capsys.readouterr().err == f"Renderer retried once: {detail}\n"


def test_render_pilot_does_not_retry_other_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    error = ProcessingError("hyperframes", 1, "encoder failed")
    calls = []

    def render(_executable: Path, args: list[str], *, timeout_s: float) -> None:
        calls.append(args)
        raise error

    monkeypatch.setattr(hyperframes, "preflight", lambda _path, _frames: None)
    monkeypatch.setattr(
        hyperframes, "ensure_installed", lambda: SimpleNamespace(executable=tmp_path)
    )
    monkeypatch.setattr(hyperframes, "_run_renderer", render)
    with pytest.raises(ProcessingError) as caught:
        render_pilot(tmp_path / "composition", tmp_path / "cut.mp4", tmp_path / "out.mp4", 90)
    assert caught.value is error
    assert len(calls) == 1
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("fixes", [None, {"ship": "shit"}])
def test_pilot_rebuilds_same_composition_without_trash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixes,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT))
    from scripts import camera3d_pilot as pilot

    project = tmp_path / "project/cut"
    project.mkdir(parents=True)
    (project / "cut.mp4").write_bytes(b"synthetic")
    (project / "cut-words.json").write_text('{"words": []}')
    output = tmp_path / "out.mp4"
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.setattr(sys, "argv", ["camera3d_pilot.py", str(project.parent), str(output)])
    monkeypatch.setattr(pilot.ffmpeg, "probe", lambda *_args, **_kwargs:
                        SimpleNamespace(video_frames=30))
    monkeypatch.setattr(pilot, "prepare_plate", lambda *_args, **_kwargs:
                        (tmp_path / "plate.mp4", tmp_path / "person.webm", tmp_path / "gray.mkv"))
    monkeypatch.setattr(pilot, "summarize_matte", lambda *_args: {})
    monkeypatch.setattr(pilot, "sample_cut_luminance", lambda *_args: None)
    if fixes is not None:
        (project.parent / "project.json").write_text(json.dumps({"word_fixes": fixes}))
        monkeypatch.setattr(pilot, "ProjectContext", lambda *_args: SimpleNamespace(
            project=SimpleNamespace(word_fixes=fixes)))
        monkeypatch.setattr(pilot, "select_style", lambda *_args:
                            camera3d.resolve_style("preset:clean_minimal"))

    def planned(*_args, **kwargs):
        assert kwargs["word_fixes"] == (fixes or {})
        return {"frames": 30, "fallbacks": []}

    monkeypatch.setattr(pilot, "plan_camera3d", planned)
    builds = []

    def build(_plan: dict, _plate: Path, _person: Path, folder: Path, **_kwargs) -> Path:
        folder.mkdir(exist_ok=True)
        marker = str(len(builds) + 1)
        (folder / "marker").write_text(marker)
        builds.append(folder)
        return folder

    monkeypatch.setattr(pilot, "build_composition", build)
    monkeypatch.setattr(pilot, "render_pilot", lambda *_args, **_kwargs: None)
    assert pilot.main() == 0
    assert pilot.main() == 0
    composition = output.with_suffix(".camera3d") / "composition"
    assert (composition / "marker").read_text() == "2"
    assert builds[0] != composition and builds[1] != composition

    def failed_build(*_args, **_kwargs) -> Path:
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(pilot, "build_composition", failed_build)
    with pytest.raises(RuntimeError, match="synthetic build failure"):
        pilot.main()
    assert (composition / "marker").read_text() == "2"


@pytest.mark.render
def test_short_synthetic_render_has_matched_frames_and_brand_accent(tmp_path: Path) -> None:
    if not __import__("os").environ.get("YAP_STUDIO_RENDER_TESTS"):
        pytest.skip("set YAP_STUDIO_RENDER_TESTS=1")
    if not ffmpeg.has_filters(("alphamerge",))["alphamerge"]:
        pytest.skip("ffmpeg needs alphamerge")
    plate, person, cut = (tmp_path / name for name in ("plate.mp4", "person.webm", "cut.mp4"))
    ffmpeg.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x203050:s=1560x2640:r=30",
            "-frames:v",
            "90",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(plate),
        ],
        timeout_s=120,
    )
    ffmpeg.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0xB030A0:s=1080x1920:r=30",
            "-frames:v",
            "90",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(cut),
        ],
        timeout_s=120,
    )
    mask = (
        "color=c=black:s=1080x1920:r=30,"
        "drawbox=x=390:y=260:w=300:h=440:color=white:t=fill,"
        "drawbox=x=190:y=800:w=700:h=1120:color=white:t=fill,format=gray"
    )
    ffmpeg.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(cut),
            "-f",
            "lavfi",
            "-i",
            mask,
            "-filter_complex",
            "[0:v]format=rgba[p];[p][1:v]alphamerge,format=yuva420p[v]",
            "-map",
            "[v]",
            "-frames:v",
            "90",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "18",
            "-b:v",
            "0",
            "-row-mt",
            "1",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-auto-alt-ref",
            "0",
            "-pix_fmt",
            "yuva420p",
            str(person),
        ],
        timeout_s=300,
    )
    _, silhouette = _silhouette()
    words = {
        "time_base": "output",
        "words": [
            {"id": "w0", "segment": "s0", "text": "Beautiful", "start": 0.35, "end": 0.6},
            {"id": "w1", "segment": "s0", "text": "stories", "start": 0.7, "end": 0.9},
            {"id": "w2", "segment": "s0", "text": "move.", "start": 1.1, "end": 1.3},
        ],
    }
    bright = np.full((6, 480, 270), 205, dtype=np.uint8)
    plan = plan_camera3d(words, 3, silhouette, luminance=bright,
                         style=_branded_style("Inter", "Gelasio", "#2F6FED"))
    assert any(group["halo"] for group in plan["groups"])
    comp = build_composition(plan, plate, person, tmp_path / "composition", grain=False)
    out = tmp_path / "render.mp4"
    stages: list[str] = []
    render_pilot(comp, cut, out, 90, progress=stages.append)
    assert stages == ["render", "mux"]
    assert ffmpeg.probe(out, count_frames=True).video_frames == 90
    raw = ffmpeg.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(out),
            "-vf",
            "select=eq(n\\,30),format=rgb24",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout_s=60,
        binary=True,
    ).stdout
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(1920, 1080, 3)
    assert rgb[420, 540, 0] > 120 and rgb[420, 540, 2] > 100
    top = rgb[50:240]
    assert ((top[:, :, 2].astype(int) > top[:, :, 0].astype(int) + 70) &
            (top[:, :, 2].astype(int) > top[:, :, 1].astype(int) + 60) &
            (top[:, :, 2] > 130)).any()
    red_plan = plan_camera3d(words, 3, silhouette, luminance=bright,
                             style=_branded_style("Inter", "Gelasio", "#E05020"))
    red_comp = build_composition(red_plan, plate, person, tmp_path / "red-composition",
                                 grain=False)
    red_out = tmp_path / "red-render.mp4"
    render_pilot(red_comp, cut, red_out, 90)
    red_raw = ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(red_out), "-vf", "select=eq(n\\,30),format=rgb24",
        "-frames:v", "1", "-f", "rawvideo", "pipe:1",
    ], timeout_s=60, binary=True).stdout
    red_rgb = np.frombuffer(red_raw, dtype=np.uint8).reshape(1920, 1080, 3)
    red_top = red_rgb[50:240]
    assert ((red_top[:, :, 0].astype(int) > red_top[:, :, 1].astype(int) + 50) &
            (red_top[:, :, 0].astype(int) > red_top[:, :, 2].astype(int) + 70) &
            (red_top[:, :, 0] > 130)).any()
    assert not np.array_equal(top, red_top)
