"""Synthetic number sizing, matte placement, speech and delivery contracts."""

from __future__ import annotations

import json
import math
import re
import subprocess

import numpy as np
import pytest
from test_planner_cards import plan, requests

from yap_studio import cards
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.faces.placement import Band
from yap_studio.style import resolve


def number_plan(**slots):
    return cards.card_layer_plan("big_number", {"end": 10000, **slots},
                                 resolve("preset:clean_minimal"), frames=75)


def test_number_is_appended_and_requires_creator_opt_in() -> None:
    # Later kinds may be appended after it; it stays after the original kinds and opt-in.
    assert cards.CARD_TEMPLATES.index("big_number") > cards.CARD_TEMPLATES.index("video_card")
    assert "big_number" in cards.OPT_IN_KINDS
    style = resolve("preset:clean_minimal")
    assert cards.card_kind_off(style, "big_number")
    data = requests(template="big_number", slots={"end": 10000})
    with pytest.raises(ValidationError, match="has not turned on"):
        plan(data, style=style)
    style.data["graphics"]["card_kinds_on"] = ["big_number"]
    result = plan(data, style=style)
    assert result.status == "ok"
    assert result.cards[0].clearance["status"] == "provisional"
    assert "matte" in result.cards[0].clearance["reason"]


@pytest.mark.parametrize("slots", [
    {"end": -1}, {"end": 1000000000}, {"end": 1.234}, {"start": 1.234},
    {"end": True}, {"end": float("nan")}, {"label": "x" * 25},
    {"prefix": "xxxx"}, {"suffix": "xxxxxxx"},
])
def test_number_slots_refuse_invalid_values_with_a_fix(slots: dict) -> None:
    with pytest.raises(ValidationError) as caught:
        number_plan(**slots)
    assert caught.value.parameter == f"slots.{next(iter(slots))}"
    assert caught.value.detail


def test_number_duration_and_width_fit() -> None:
    from yap_studio.planner.number_behind import BIG_NUMBER_MAX_PX, BIG_NUMBER_MIN_PX

    with pytest.raises(ValidationError, match="at least 2.1 s"):
        cards.card_layer_plan("big_number", {"end": 10000},
                              resolve("preset:clean_minimal"), frames=62)
    fitted = number_plan(end=999999999, suffix="%", label="Total")
    assert BIG_NUMBER_MIN_PX <= fitted.variables["value_px"] < BIG_NUMBER_MAX_PX
    assert fitted.width <= 1080
    assert fitted.width % 2 == fitted.height % 2 == 0
    assert fitted.variables["label_color"] == resolve("preset:clean_minimal").data[
        "captions"]["text_color"]
    with pytest.raises(ValidationError, match="shorten"):
        number_plan(end=999999999, prefix="界界界", suffix="界" * 6)


@pytest.mark.parametrize("slots,descent", [
    ({"end": 10}, 0), ({"end": 10000}, .2), ({"end": 10, "prefix": "$"}, .2),
])
def test_number_canvas_reserves_descenders_before_shadow(slots: dict, descent: float) -> None:
    plan = number_plan(**slots)
    v = plan.variables
    assert v["digit_descent_em"] == descent
    height = (v["cap_height_em"] + descent) * v["value_px"] + 2 * v["shadow_pad_px"]
    assert plan.height == math.ceil(height / 2) * 2


def test_number_label_keeps_its_descenders_inside_the_shadow_padding() -> None:
    from yap_studio.planner.number_behind import DESCENT_EM

    plan = number_plan(end=10, label="QUANTITY")
    v = plan.variables
    height = ((v["cap_height_em"] + v["digit_descent_em"]) * v["value_px"]
              + v["label_gap_px"] + (v["cap_height_em"] + DESCENT_EM) * v["label_px"]
              + 2 * v["shadow_pad_px"])
    assert plan.height == math.ceil(height / 2) * 2


def test_python_count_strings_match_the_template_formula() -> None:
    from yap_studio.planner.number_behind import number_string

    html = (cards.template_dir("big_number") / "template.html").read_text()
    formula = re.search(r"function numberString\(v, frame\) \{.*?\n    \}", html, re.S)
    assert formula is not None
    v = number_plan(end=10000.25, start=0.05, prefix="$", suffix="x").variables
    frames = list(range(75))
    script = formula[0] + "\nprocess.stdout.write(JSON.stringify(" + json.dumps(frames)
    script += ".map(frame => numberString(" + json.dumps(v) + ", frame))));"
    actual = json.loads(subprocess.check_output(["node", "-e", script]))
    assert actual == [number_string(v, frame) for frame in frames]
    assert actual[-1] == "$10,000.25x"


def independent_visibility(plan, alpha: np.ndarray, frame: int) -> float:
    """Rasterize advance boxes independently of the candidate's visibility counter."""
    from yap_studio.planner.number_behind import number_string
    from yap_studio.render.text_metrics import text_width_em, widest_digit_em

    v = plan.variables
    text = number_string(v, frame)
    size = v["value_px"]
    widths = [(widest_digit_em(v["font_name"], v["weight"]) if char.isdigit()
               else text_width_em(char, v["font_name"], v["weight"])) * size for char in text]
    left = plan.x + (plan.width - sum(widths)) / 2
    top = plan.y + v["shadow_pad_px"]
    mask = np.zeros_like(alpha, dtype=bool)
    for width in widths:
        mask[math.floor(top):math.ceil(top + size * v["cap_height_em"]),
             math.floor(left):math.ceil(left + width)] = True
        left += width
    return 1 - np.count_nonzero(mask & (alpha >= 128)) / np.count_nonzero(mask)


@pytest.mark.parametrize("hair_rows,moves", [(4, False), (80, True)])
def test_tucked_number_moves_up_only_when_needed(hair_rows: int, moves: bool) -> None:
    from yap_studio.planner.number_behind import place_number

    layer = number_plan(end=10)
    face = Box(440, 600, 200, 200)
    head = head_box(face)
    bottom = math.floor(head.y + .4 * head.h)
    alpha = np.zeros((1920, 1080), np.uint8)
    alpha[bottom - hair_rows:1200, 450:630] = 255
    placed, placement, evidence = place_number(
        layer, [[face]] * 75, [alpha] * 75, caption_top=1500, band=Band(0, 900),
        margin_px=24, identifier="agent-number",
    )
    assert placement == "behind_head" and evidence["status"] == "behind_number"
    assert (evidence["position"] == "up") == moves
    assert evidence["worst_visible_share"] >= .95
    assert min(independent_visibility(placed, alpha, frame) for frame in range(75)) >= .95


@pytest.mark.parametrize("hair_top,position", [(800, "tucked"), (780, "up")])
def test_number_tucks_into_hair_below_the_top_band(hair_top: int, position: str) -> None:
    from yap_studio.planner.number_behind import HEAD_TUCK_SHARE, place_number

    layer = number_plan(end=10)
    face = Box(440, 800, 200, 200)
    head = head_box(face)
    band = Band(0, 640)
    clear = np.zeros((1920, 1080), np.uint8)
    hair = clear.copy()
    hair[hair_top:1200, 450:630] = 255
    mattes = [clear if frame % 2 else hair for frame in range(layer.frames)]
    placed, placement, evidence = place_number(
        layer, [[face]] * layer.frames, mattes, caption_top=1500, band=band,
        margin_px=24, identifier="agent-number",
    )
    v = placed.variables
    lowest_row = placed.y + v["shadow_pad_px"] + v["cap_height_em"] * v["value_px"]
    assert placement == "behind_head" and evidence["position"] == position
    assert band.bottom < hair_top < lowest_row <= head.y + HEAD_TUCK_SHARE * head.h
    assert placed.y >= band.top and placed.y + placed.height <= 1500
    shares = [independent_visibility(placed, alpha, frame)
              for frame, alpha in enumerate(mattes)]
    assert .95 <= min(shares) < 1
    assert max(shares) == 1
    assert .95 <= evidence["worst_visible_share"] < 1


@pytest.mark.parametrize("covered_rows,estimated,position", [(16, .93, "up"), (9, .96, "tucked")])
def test_number_planning_keeps_margin_over_rendered_ink_rule(
    covered_rows: int, estimated: float, position: str,
) -> None:
    from yap_studio.planner.number_behind import place_number

    layer = number_plan(start=3, end=3)
    face = Box(440, 600, 200, 200)
    clear = np.zeros((1920, 1080), np.uint8)
    options = {"caption_top": 1500, "band": Band(0, 640), "margin_px": 24,
               "identifier": "agent-number"}
    default, _, _ = place_number(layer, [[face]] * layer.frames, [clear] * layer.frames, **options)
    v = default.variables
    bottom = math.ceil(default.y + v["shadow_pad_px"] + v["cap_height_em"] * v["value_px"])
    hair = clear.copy()
    hair[bottom - covered_rows:] = 255
    assert independent_visibility(default, hair, 0) == pytest.approx(estimated, abs=.002)
    placed, placement, evidence = place_number(
        layer, [[face]] * layer.frames, [hair] * layer.frames, **options,
    )
    assert placement == "behind_head" and evidence["position"] == position
    assert (placed.y < default.y) == (position == "up")
    assert evidence["worst_visible_share"] >= .95
    assert min(independent_visibility(placed, hair, frame)
               for frame in range(layer.frames)) >= .95


@pytest.mark.parametrize("passing", ["side", "scale", "front"])
def test_number_ink_search_moves_before_scaling_or_front_fallback(passing: str) -> None:
    from yap_studio.planner.number_behind import place_number

    card = number_plan(start=3, end=3)
    face = Box(440, 600, 200, 200)
    clear = np.zeros((1920, 1080), np.uint8)
    tried = []

    def ink(candidate):
        tried.append(candidate)
        if passing == "side" and candidate.x > card.x:
            return .91
        if passing == "scale" and candidate.variables["value_px"] < card.variables["value_px"]:
            return .91
        return .8

    placed, placement, evidence = place_number(
        card, [[face]] * card.frames, [clear] * card.frames,
        caption_top=1500, band=Band(0, 640), margin_px=24, identifier="agent-number",
        ink_visible_share=ink,
    )
    centred = [candidate for candidate in tried
               if candidate.x == card.x and candidate.variables["value_px"] == 300]
    assert centred[1].y == centred[0].y - 8
    assert min(candidate.y for candidate in tried) >= 0
    if passing == "front":
        assert placement == "top" and evidence["status"] == "beside_front"
    else:
        assert placement == "behind_head" and evidence["ink_status"] == "verified"
        assert evidence["worst_visible_share"] == .91
        assert (placed.variables["value_px"] < 300) == (passing == "scale")
        assert evidence["position"] == ("side" if passing == "side" else "tucked")


@pytest.mark.parametrize("blocked", ["matte", "transition"])
def test_number_falls_beside_head_in_front_with_warning(blocked: str) -> None:
    from yap_studio.planner.number_behind import place_number

    face = Box(440, 600, 200, 200)
    alpha = np.full((1920, 1080), 255, np.uint8)
    placed, placement, evidence = place_number(
        number_plan(end=10), [[face]] * 75, [alpha] * 75,
        caption_top=1500, band=Band(0, 900), margin_px=24, identifier="agent-number",
        face_windows=((12, 15),) if blocked == "transition" else (),
    )
    assert placement == "top" and evidence["status"] == "beside_front"
    assert gap_px(Box(placed.x, placed.y, placed.width, placed.height), head_box(face)) >= 24
    assert "agent-number:" in evidence["warning"] and "beside her head in front" in evidence[
        "warning"]


def test_number_refuses_when_no_side_corridor_fits() -> None:
    from yap_studio.planner.number_behind import place_number

    alpha = np.full((1920, 1080), 255, np.uint8)
    with pytest.raises(ValidationError, match="shorter number.*smaller label.*another moment"):
        place_number(number_plan(), [[Box(0, 0, 1080, 1920)]] * 75, [alpha] * 75,
                     caption_top=1500, band=Band(0, 640), margin_px=24,
                     identifier="agent-number")


def test_front_side_corridor_can_move_up_to_clear_captions() -> None:
    from yap_studio.planner.number_behind import place_number

    face = Box(440, 600, 200, 200)
    alpha = np.zeros((1920, 1080), np.uint8)
    placed, placement, _ = place_number(
        number_plan(end=10), [[face]] * 75, [alpha] * 75, caption_top=600,
        band=Band(0, 640), margin_px=24, identifier="agent-number", face_windows=((0, 75),),
    )
    assert placement == "top" and placed.y + placed.height <= 600
    assert gap_px(Box(placed.x, placed.y, placed.width, placed.height), head_box(face)) >= 24


@pytest.mark.parametrize("move", ["side", "scale"])
def test_top_band_forces_side_search_or_scaling(move: str) -> None:
    from yap_studio.planner.number_behind import place_number

    layer = number_plan(end=10)
    face = Box(440, 600, 200, 200)
    alpha = np.zeros((1920, 1080), np.uint8)
    if move == "side":
        # An upright strip covers the middle at every allowed height.
        alpha[:1000, 520:560] = 255
        band = Band(350, 900)
    else:
        # The large row cannot fit this band, but the 90% row can.
        band = Band(360, 620)
    placed, placement, evidence = place_number(
        layer, [[face]] * 75, [alpha] * 75, caption_top=1500, band=band,
        margin_px=24, identifier="agent-number",
    )
    assert placement == "behind_head"
    if move == "side":
        assert evidence["position"] == "side" and evidence["scale"] == 1
    else:
        assert evidence["scale"] < 1
    assert min(independent_visibility(placed, alpha, frame) for frame in range(75)) >= .95


def test_label_visibility_is_checked_separately_on_each_frame() -> None:
    from yap_studio.planner.number_behind import glyph_rows, place_number

    layer = number_plan(end=10, label="Total")
    face = Box(440, 600, 200, 200)
    alpha = np.zeros((1920, 1080), np.uint8)
    alpha[580:610, 350:730] = 255
    placed, _, evidence = place_number(
        layer, [[face]] * 75, [alpha] * 75, caption_top=1500, band=Band(0, 900),
        margin_px=24, identifier="agent-number",
    )
    assert evidence["position"] == "up"
    text, top, size = glyph_rows(placed, 45)[1]
    width = sum(json.loads(placed.variables["advances"])[char] * size for char in text)
    left = math.floor(placed.x + (placed.width - width) / 2)
    box = alpha[math.floor(top):math.ceil(top + .75 * size), left:math.ceil(left + width)]
    assert np.count_nonzero(box >= 128) / box.size <= .05


def test_number_template_metadata_and_preflight(tmp_path) -> None:
    from html.parser import HTMLParser

    from yap_studio.render import hyperframes

    class Variables(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "html":
                self.value = json.loads(dict(attrs)["data-composition-variables"])

    layer = number_plan(label="Total")
    parser = Variables()
    parser.feed((cards.template_dir("big_number") / "template.html").read_text())
    metadata = cards.load_template("big_number")
    assert metadata["derived_from"] == []
    assert parser.value == metadata["variables"]
    assert set(layer.variables) == {v["id"] for v in parser.value}
    folder = cards.prepare_card(layer, tmp_path / "work")
    assert hyperframes.preflight(folder, 75) == hyperframes.layer_duration_s(75, 30)


def test_number_on_switch_round_trips_through_the_style_loader() -> None:
    from yap_studio.style import resolve_style_data

    data = resolve_style_data({"schema_version": 1, "name": "Number style",
                               "graphics": {"card_kinds_on": ["big_number"]}})
    assert not cards.card_kind_off(data, "big_number")
    old = resolve_style_data({"schema_version": 1, "name": "Old style"})
    assert cards.card_kind_off(old, "big_number")


@pytest.mark.parametrize("words,value,suffix,accepted", [
    (["ten", "thousand"], 10000, "", True),
    (["fifty", "percent"], 50, "%", True),
    (["fifty"], 50, "%", False),
    (["10,000", "50%", "$5"], 5, "", True),
    (["ten"], 12, "", False),
    (["minus", "ten"], 10, "", False),
    (["negative", "fifty", "percent"], 50, "%", False),
    (["-10"], 10, "", False),
    (["negative.", "ten"], 10, "", True),
])
def test_number_must_match_anchored_speech(words, value, suffix, accepted) -> None:
    from yap_studio.planner.number_spoken import spoken_number_fix
    from yap_studio.project.model import CutWord

    cut = [CutWord(f"t1:{i}", "s1", text, i / 5, (i + 1) / 5)
           for i, text in enumerate(words)]
    fix = spoken_number_fix({"end": value, "suffix": suffix}, cut, 0, 75)
    assert (fix is None) == accepted
    if not accepted:
        assert "heard:" in fix
        assert ("50" if suffix else "10") in fix


def test_final_number_visibility_allows_only_three_hidden_frames() -> None:
    from yap_studio.stages.number_visibility import number_visibility_assert

    for hidden in (0, 3, 4):
        record = {"id": "agent-number", "hidden_frames": hidden,
                  "worst_visible_share": .89 if hidden else 1}
        check = number_visibility_assert([record])
        assert check["id"] == "graphics.number_visible"
        assert check["pass"] == (hidden <= 3)
        assert "re-run graphics" in check["fix"]
