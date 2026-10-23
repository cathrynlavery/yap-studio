"""Motion streak schema, rest boxes, and swept head clearance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from test_cards import ROWS_ON_CUE_SLOTS, _preview
from test_cards import image as image

from yap_studio import cards
from yap_studio.card_motion import SUPPORTED, drawn_template
from yap_studio.errors import ValidationError
from yap_studio.faces.model import Box
from yap_studio.planner.cards import _render_key, place_card
from yap_studio.style import SHIPPED_PRESETS, resolve
from yap_studio.style.loader import resolve_style_data


def motion_style():
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_motion"] = "streak"
    return style


@pytest.mark.parametrize("kind", sorted(SUPPORTED))
def test_drawn_template_accepts_supported_templates(kind):
    drawn_template(cards.template_html(kind), kind)


@pytest.mark.parametrize("kind,pattern", [
    ("stat", "tl.fromTo(row,"),
    ("steps", "tl.set("),
    ("specs_list", "tl.fromTo(row,"),
    ("image_compare", 'tl.fromTo(panel, { "--card-split-position": 0 }'),
])
def test_drawn_template_rejects_missing_cue_pattern(kind, pattern):
    source = cards.template_html(kind)
    assert source.count(pattern) > 0
    edited = source.replace(pattern, pattern.replace("tl.", "edited."))
    with pytest.raises(ValidationError) as caught:
        drawn_template(edited, kind)
    assert kind in str(caught.value)
    assert pattern in str(caught.value)


def test_card_motion_schema_and_presets():
    from yap_studio.style import StyleError

    for preset in SHIPPED_PRESETS:
        assert resolve(f"preset:{preset}").data["graphics"]["card_motion"] == "none"
    assert resolve_style_data({"schema_version": 1, "graphics": {"card_motion": "streak"}})[
        "graphics"]["card_motion"] == "streak"
    with pytest.raises(StyleError, match="card_motion"):
        resolve_style_data({"schema_version": 1, "graphics": {"card_motion": "other"}})


@pytest.mark.parametrize("kind", [kind for kind in cards.CARD_TEMPLATES
                                  if (cards.template_dir(kind) / "preview.json").is_file()])
def test_none_matches_style_without_motion(kind, image, tmp_path):
    slots = _preview(kind, image)
    if kind == "reel_feed":
        slots["caption"] = "Small steps."
    for preset in SHIPPED_PRESETS:
        style = resolve(f"preset:{preset}")
        assert style.data["graphics"]["card_motion"] == "none"
        without_motion = deepcopy(style)
        del without_motion.data["graphics"]["card_motion"]
        plan = cards.card_layer_plan(kind, slots, style, frames=90)
        expected = cards.card_layer_plan(kind, slots, without_motion, frames=90)
        assert plan.motion is None and expected.motion is None
        assert _render_key(plan) == _render_key(expected), preset
        assert plan.variables == expected.variables, preset
        page = cards.prepare_card(plan, tmp_path / preset / "none") / "index.html"
        expected_page = cards.prepare_card(expected, tmp_path / preset / "without") / "index.html"
        assert page.read_bytes() == expected_page.read_bytes(), preset


@pytest.mark.parametrize("kind", ["specs_list", "stat"])
def test_default_cued_rows_fly_with_padding_above_the_line(kind):
    style = motion_style()
    style.data["graphics"]["specs_rows"] = "on_cue"
    plan = cards.card_layer_plan(kind, ROWS_ON_CUE_SLOTS[kind], style, frames=120)
    assert plan.y < 280 <= plan.y + plan.variables["glow_pad_px"] <= 281
    assert plan.motion == {"side": "right", "enter": True}
    assert plan.layer_box() == {"x": 0, "y": plan.y, "width": 1080, "height": plan.height}


@pytest.mark.parametrize("y,variables", [(278, {}), (260, {"glow_pad_px": 18})])
def test_motion_refuses_visible_top_above_the_line(y, variables):
    from yap_studio.card_motion import plan_motion

    plan = cards.CardPlan("quote", 20, y, 1040, 360, 90, 30, variables, ())
    placed, evidence = plan_motion(plan, motion_style(), None, {})
    assert placed.motion is None
    assert evidence["card_motion"] == {
        "status": "none", "reason": "the card row starts above the 280 px line"}


def test_motion_sweep_keeps_glow_padding_in_the_clearance_box():
    from yap_studio.card_motion import plan_motion

    plan = cards.CardPlan("quote", 400, 262, 280, 100, 90, 30, {"glow_pad_px": 18}, ())
    # The visible top clears both heads, but the full layer's padding does not.
    faces = [[Box(0, 230, 150, 20), Box(900, 230, 150, 20)]] * 90
    placed, evidence = plan_motion(plan, motion_style(), faces, {})
    assert placed.motion is None
    assert evidence["card_motion"] == {
        "status": "none", "reason": "neither swept box keeps the required head clearance"}


@pytest.mark.parametrize("face,side", [(Box(40, 900, 200, 250), "right"),
                                      (Box(840, 900, 200, 250), "left")])
def test_motion_side_away_from_face(face, side):
    style = motion_style()
    plan = cards.card_layer_plan("quote", {"quote": "Ship it", "name": "Sample"}, style, frames=90)
    placed, _, evidence = place_card(plan, {"quote": "Ship it", "name": "Sample"}, style,
                                     [[face]] * 90)
    assert placed.motion["side"] == side
    assert evidence["card_motion"]["status"] == "streak"
    assert placed.x == plan.x and placed.width == plan.width
    assert placed.layer_box() == {"x": 0, "y": placed.y, "width": 1080,
                                  "height": placed.height}


def test_motion_sweep_tries_other_edge_then_falls_back():
    style = motion_style()
    plan = replace(cards.card_layer_plan("quote", {"quote": "Ship it", "name": "Sample"}, style,
                                         frames=90), x=400, width=280, y=600)
    # The rest box is clear. Both approach corridors contain a head during the moves.
    faces = [[Box(40, 650, 160, 120), Box(850, 650, 160, 120)]] * 90
    placed, _, evidence = place_card(plan, {}, style, faces, preserve_box=True)
    assert placed.motion is None
    assert evidence["card_motion"]["status"] == "none"
    assert "swept" in evidence["card_motion"]["reason"]
    placed, _, evidence = place_card(plan, {}, style, [[Box(40, 650, 160, 120)]] * 90,
                                     preserve_box=True)
    assert placed.motion["side"] == "right"
    faces = [[Box(850 if index < 13 or index >= 79 else 40, 650, 160, 120)]
             for index in range(90)]
    placed, _, _ = place_card(plan, {}, style, faces, preserve_box=True)
    assert placed.motion["side"] == "left"


@pytest.mark.parametrize("width,boxes,reason", [
    (960, [Box(300, 800, 480, 600)],
     "no position below the 280 px line keeps the head clear for the fly"),
    (400, [Box(40, 650, 160, 120), Box(850, 650, 160, 120)],
     "neither swept box keeps the required head clearance"),
])
def test_motion_retries_ordinary_placement_above_the_band(
    width, boxes, reason, tmp_path, monkeypatch,
):
    from test_planner_cards import fake_render, requests, synthetic_faces, synthetic_words

    from yap_studio.faces.clearance import face_boxes_per_frame, face_clearance_px, gap_px, head_box
    from yap_studio.faces.model import FaceBox
    from yap_studio.planner import cards as planner
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

    style = motion_style()
    style.data["graphics"]["card_width_px"] = width
    slots = {"quote": "Ship it", "name": "Sample"}
    data = requests(template="quote", slots=slots, start={"time": 0}, end={"time": 3})
    faces = replace(synthetic_faces(frames=90),
                    frames=(tuple(FaceBox(box, 1, "detected") for box in boxes),) * 90)
    zoom = ResolvedElement("zoom", "zoom", "agent", 0, 90, "front",
                           {"peak_scale": 1.1, "center": [0.5, 0.5]})
    timeline = ResolvedTimeline((zoom,), (), (), 90, 0)
    plain_style = deepcopy(style)
    plain_style.data["graphics"]["card_motion"] = "none"
    plain = planner.plan_cards(data, synthetic_words(), plain_style, frames=90,
                               faces=faces, timeline=timeline).cards[0]
    planned = planner.plan_cards(data, synthetic_words(), style, frames=90,
                                 faces=faces, timeline=timeline)
    placed, = planned.cards
    assert plain.y == 0 and plain.clearance["status"] == "clear"
    assert (plain.width, plain.height) == (width + 80, 360)
    assert placed.clearance["card_motion"] == {"status": "none", "reason": reason}
    clearance = {key: value for key, value in placed.clearance.items() if key != "card_motion"}
    assert replace(placed, clearance=clearance) == plain
    box = Box(placed.x, placed.y, placed.width, placed.height)
    assert all(gap_px(box, head_box(face)) >= face_clearance_px(style.data)
               for frame in face_boxes_per_frame(faces)[placed.start_frame:placed.end_frame]
               for face in frame)
    rendered = []

    def render(layer, *args, **kwargs):
        rendered.append(layer)
        return fake_render(layer, *args, **kwargs)

    monkeypatch.setattr(planner, "render_card", render)
    planner.render_layers(planned, style, tmp_path / "plan")
    assert len(rendered) == 1 and rendered[0].motion is None


def test_centred_above_head_uses_nearer_open_edge_and_no_face_uses_right():
    style = motion_style()
    plan = replace(cards.card_layer_plan("quote", {"quote": "Ship it", "name": "Sample"}, style,
                                         frames=90), x=220, width=280)
    placed, _, _ = place_card(plan, {}, style, [[Box(260, 1000, 200, 200)]] * 90,
                              preserve_box=True)
    assert placed.motion["side"] == "left"
    placed, _, _ = place_card(plan, {}, style, None, preserve_box=True)
    assert placed.motion["side"] == "right"


@pytest.mark.parametrize("kind,reason", [
    ("big_number", "behind"), ("choices", "GSAP"), ("edit_timeline", "GSAP"),
    ("file_drop", "GSAP"), ("gravity_drop", "GSAP"), ("jiggle_delete", "GSAP"),
    ("storyboard", "GSAP"), ("playing_grid", "full-frame"), ("reel_feed", "full-frame"),
    ("video_card", "video windows"),
])
def test_unsupported_motion_records_reason(kind, reason):
    from yap_studio.card_motion import plan_motion

    plan = cards.CardPlan(kind, 20, 280, 1040, 360, 90, 30, {}, ())
    placed, evidence = plan_motion(plan, motion_style(), None, {})
    assert placed.motion is None
    assert evidence["card_motion"]["status"] == "none"
    assert reason in evidence["card_motion"]["reason"]


def test_choices_keeps_ordinary_entrance_and_default_placement(tmp_path, monkeypatch):
    from test_planner_cards import fake_render, plan, requests, synthetic_faces

    from yap_studio.planner import cards as planner

    style = motion_style()
    style.data["graphics"]["card_kinds_on"] = ["choices"]
    plain_style = deepcopy(style)
    plain_style.data["graphics"]["card_motion"] = "none"
    data = requests(template="choices", slots={"options": ["One", "Two"]})
    faces = synthetic_faces()
    plain, = plan(data, style=plain_style, faces=faces).cards
    planned = plan(data, style=style, faces=faces)
    placed, = planned.cards
    assert placed.clearance["card_motion"] == {
        "status": "none", "reason": "this template has no GSAP panel timeline",
    }
    clearance = {key: value for key, value in placed.clearance.items() if key != "card_motion"}
    assert replace(placed, clearance=clearance) == plain
    rendered = []

    def render(layer, *args, **kwargs):
        rendered.append(layer)
        return fake_render(layer, *args, **kwargs)

    monkeypatch.setattr(planner, "render_card", render)
    planner.render_layers(planned, style, tmp_path / "plan")
    assert len(rendered) == 1 and rendered[0].motion is None


def test_swept_clearance_uses_only_move_frames():
    from yap_studio.card_motion import plan_motion

    plan = cards.CardPlan("quote", 400, 600, 280, 300, 90, 30, {}, ())
    faces = [[] for _ in range(90)]
    faces[40] = [Box(0, 650, 150, 120), Box(900, 650, 150, 120)]
    placed, _ = plan_motion(plan, motion_style(), faces, {})
    assert placed.motion is not None


def test_standalone_plan_keeps_rest_box_and_renders_band(tmp_path, monkeypatch):
    from test_planner_cards import plan, requests

    from yap_studio.planner import cards as planner
    from yap_studio.render.hyperframes import LayerResult

    style = motion_style()
    result = plan(requests(template="text_tag", slots={"text": "Ship it"}), style=style)
    card, = result.cards
    assert card.clearance["card_motion"]["status"] == "streak"
    assert result.lane()["elements"][0]["box"]["width"] == card.width
    assert result.overlays()[0].x == 0 and result.overlays()[0].width == 1080

    def render(layer, work, out, **kwargs):
        assert layer.motion is not None
        assert layer.x == card.x and layer.width == card.width
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"synthetic band")
        return LayerResult(out, layer.frames, 1080, layer.height, layer.frames / layer.fps)

    monkeypatch.setattr(planner, "render_card", render)
    rendered, = planner.render_layers(result, style, tmp_path / "plan")
    assert rendered.width == 1080
