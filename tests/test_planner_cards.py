"""Synthetic card timing, placement, portable artifacts, and ownership guards."""

from __future__ import annotations

import contextlib
import copy
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_planner_anchors import synthetic_words

from yap_studio import paths
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.faces.clearance import (
    DEFAULT_HAIR,
    check_layout,
    face_boxes_per_frame,
    gap_px,
    head_box,
    load_layout,
)
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.faces.placement import suggest_offset, top_band
from yap_studio.matte.composite import load_cards
from yap_studio.planner import cards
from yap_studio.project.model import CutWord, CutWords, parse_anchor, parse_document
from yap_studio.receipt import sha256_file
from yap_studio.render import fonts, hyperframes
from yap_studio.render.hyperframes import LayerResult
from yap_studio.style import SHIPPED_PRESETS, ResolvedStyle, resolve, validate_style_data

FRAMES = 900
DEFAULT_FACE = Box(458, 629, 396, 396)


def test_card_ceiling_uses_only_overlapping_caption_boxes() -> None:
    style = resolve("preset:clean_minimal")
    layout = {"measured": True, "captions": [{"start_frame": 0, "end_frame": 60, "boxes": [
        {"x": 100, "y": 700, "w": 100, "h": 60, "start_frame": 0, "end_frame": 30},
        {"x": 100, "y": 1200, "w": 100, "h": 60, "start_frame": 30, "end_frame": 60},
    ]}]}
    assert cards.card_caption_top(style, 0, 30, layout) == 700
    assert cards.card_caption_top(style, 30, 60, layout) == 1200
    assert cards.card_caption_top(style, 20, 40, layout) == 700


def test_camera_card_ceiling_ignores_the_flat_caption_band() -> None:
    style = resolve("preset:clean_minimal")
    style.data["captions"].update(camera_3d=True, top_px=700)
    layout = {"measured": True, "captions": [{"start_frame": 0, "end_frame": 60, "boxes": [
        {"x": 100, "y": 820, "w": 600, "h": 60, "start_frame": 0, "end_frame": 30},
        {"x": 100, "y": 1200, "w": 600, "h": 60, "start_frame": 30, "end_frame": 60},
    ]}]}
    ceiling = cards.card_caption_top(style, 0, 30, layout)
    assert ceiling == 820
    assert cards.card_caption_top(style, 30, 60, layout) == 1200
    assert cards.card_caption_top(style, 60, 90, layout) == 1920
    assert cards.card_caption_top(style, 0, 30) == 1920
    slots = {"end": 37, "suffix": "%"}
    layer = replace(cards.card_layer_plan("stat", slots, style, frames=30),
                    x=60, y=280, width=960, height=360)
    placed, _, evidence = cards.place_card(layer, slots, style, [()] * 30,
                                           preserve_box=True, caption_top=ceiling)
    assert evidence["status"] == "clear"
    assert placed.y == 280 and placed.y + placed.height == 640


def requests(**changes: object) -> dict:
    card = {
        "id": "g-stat",
        "template": "stat",
        "slots": {"end": 37, "suffix": "%"},
        "start": {"word": "t1:12", "segment": "story", "text": "percent"},
        "end": {"word": "t1:16"},
    }
    card.update(changes)
    return {"schema": cards.REQUESTS_SCHEMA, "cards": [card]}


def synthetic_faces(
    box: Box | None = DEFAULT_FACE,
    *,
    measured: bool = True,
    frames: int = FRAMES,
    fps: Fraction = Fraction(30),
) -> FacesData:
    face = () if box is None else (FaceBox(box, 1, "detected"),)
    return FacesData(
        "measured" if measured else "unmeasured",
        None if measured else "detector unavailable",
        VideoIdentity("cut.mp4", "c" * 64, frames, fps, 1080, 1920),
        {},
        {},
        tuple(face for _ in range(frames)) if measured else (),
        ((0, frames),) if measured and box is None else (),
    )


def plan(
    data: dict | None = None, *, faces: FacesData | None = None, style: ResolvedStyle | None = None
) -> cards.GraphicsPlan:
    return cards.plan_cards(
        requests() if data is None else data,
        synthetic_words(),
        style or resolve("preset:clean_minimal"),
        frames=FRAMES,
        faces=faces,
    )


def fake_render(layer, work_dir, out, *, install=None) -> LayerResult:
    """Publish deterministic fake pixels without invoking the real renderer."""
    out.write_bytes(f"alpha {layer.variables!r} {layer.frames}".encode())
    return LayerResult(out, layer.frames, layer.width, layer.height, layer.frames / layer.fps)


@pytest.mark.parametrize("template,slot", [("choices", "options"), ("flow", "nodes")])
@pytest.mark.parametrize("wrapped", [False, True])
def test_word_cues_resolve_to_frames_and_written_lane(
    template: str, slot: str, wrapped: bool, tmp_path: Path,
) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = [template]
    cues = [{"word": "t1:12", "segment": "story", "text": "percent", "offset_s": 0.023},
            {"word": "t1:14", "text": "word14", "offset_s": 0.019}]
    lane_cues = [{"at": parse_anchor(cue).to_dict()} for cue in cues]
    supplied = lane_cues if wrapped else cues
    result = plan(requests(template=template, slots={slot: ["One", "Two"], "cues": supplied},
                           start={"time": 12.1}, end={"time": 16.5}), style=style)
    expected = [0.333333, 1.333333]
    assert result.cards[0].slots["cues"] == expected
    cards.write_plan(result, tmp_path / "plan")
    lane = json.loads((tmp_path / "plan/agent-lane.json").read_text())
    assert lane["elements"][0]["slots"]["cues"] == lane_cues
    saved = json.loads((tmp_path / "plan/graphics-plan.json").read_text())
    assert saved["cards"][0]["slots"]["cues"] == expected
    numeric = plan(requests(template=template, slots={slot: ["One", "Two"], "cues": expected},
                            start={"time": 12.1}, end={"time": 16.5}), style=style)
    assert numeric.cards[0].to_dict() == result.cards[0].to_dict()
    assert numeric.cards[0].render_key == result.cards[0].render_key
    assert numeric.lane()["elements"][0]["slots"]["cues"] == expected


@pytest.mark.parametrize("anchor", [{"word": "t1:11"}, {"word": "t1:22"}])
def test_word_cues_outside_card_are_refused(anchor: dict) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["choices"]
    with pytest.raises(ValidationError, match="inside the card window") as caught:
        plan(requests(template="choices", slots={"options": ["One", "Two"], "cues": [anchor]}),
             style=style)
    assert caught.value.parameter == "cards[0].slots.cues[0]"


@pytest.mark.parametrize("word_start", [2.0, 2.4])
def test_half_frame_word_cue_cannot_hide_a_short_gap(word_start: float) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["choices"]
    words = CutWords("output", {"t1": "a" * 64}, (
        CutWord("t1:0", "story", "One", word_start, word_start + 0.2, 1),
    ))
    data = requests(template="choices", start={"time": word_start},
                    end={"time": word_start + 3}, slots={
                        "options": ["One", "Two"],
                        "cues": [{"word": "t1:0", "offset_s": 0.05}, 0.333333],
                    })
    with pytest.raises(ValidationError, match="slots.cues.*space the cues"):
        cards.plan_cards(data, words, style, frames=300)


@pytest.mark.parametrize("face", [Box(350, 650, 180, 200), Box(550, 650, 180, 200)])
def test_choices_uses_wider_side_and_clears_heads(face: Box) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["choices"]
    faces = synthetic_faces(face)
    card = plan(requests(template="choices", slots={"options": ["One", "Two"]}),
                style=style, faces=faces).cards[0]
    head = head_box(face)
    assert card.clearance["position"] == "side"
    assert card.y == pytest.approx(head.y, abs=2)
    if face.x < 500:
        assert card.x >= head.x1 + card.clearance["margin_px"]
    else:
        assert card.x + card.width <= head.x - card.clearance["margin_px"]
    assert gap_px(Box(card.x, card.y, card.width, card.height), head) >= 24
    assert card.y + card.height <= cards.card_caption_top(style, card.start_frame, card.end_frame)
    assert plan(faces=faces).cards[0].clearance["position"] == "top"


@pytest.mark.parametrize("face_y,caption_top,position,size", [
    (820, 1400, "headroom", 64), (654, 1400, "headroom", 56),
    (470, 1400, "lower", 64), (470, 1000, None, 64),
])
def test_choices_safe_positions_refit_and_clear_heads(
    face_y: int, caption_top: int, position: str | None, size: int,
) -> None:
    from yap_studio.cards import card_layer_plan

    style = resolve("preset:clean_minimal")
    slots = {"options": ["One", "Two"]}
    layer = card_layer_plan("choices", slots, style)
    faces = [[Box(330, face_y, 500, 200)]]
    placed, _, evidence = cards.place_card(layer, slots, style, faces,
                                          safe_order=("side", "headroom", "lower"),
                                          caption_top=caption_top)
    if position is None:
        assert evidence["status"] == "blocked"
        assert evidence["reason"] == (
            "no room above or beside the head for the options at a readable size; "
            "use fewer or shorter options or move the card to another moment with more room "
            "above the head")
        assert "another moment" in evidence["reason"]
        return
    head = head_box(faces[0][0])
    assert evidence["position"] == position
    assert placed.variables["text_px"] == size
    assert placed.y + placed.variables["glow_pad_px"] >= 280
    assert placed.y + placed.height <= caption_top - style.data["graphics"]["caption_gap_px"]
    assert gap_px(Box(placed.x, placed.y, placed.width, placed.height), head) >= 24
    assert placed.x + placed.width / 2 == pytest.approx((head.x + head.x1) / 2, abs=1)
    if position == "headroom":
        band_top = 280 - placed.variables["glow_pad_px"]
        band_bottom = head.y - 24
        assert placed.y + placed.height / 2 == pytest.approx((band_top + band_bottom) / 2, abs=1)


def test_choices_without_faces_starts_at_safe_top_unchecked() -> None:
    from yap_studio.cards import card_layer_plan

    style = resolve("preset:clean_minimal")
    slots = {"options": ["One", "Two"]}
    layer = card_layer_plan("choices", slots, style)
    placed, _, evidence = cards.place_card(
        layer, slots, style, None, safe_order=("side", "headroom", "lower"), caption_top=1400)
    assert placed == layer and evidence["status"] == "unchecked"
    assert placed.x == (1080 - placed.width) // 2
    assert 280 <= placed.y + placed.variables["glow_pad_px"] <= 281


def test_choices_small_options_fit_short_headroom() -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["choices"]
    result = plan(requests(template="choices", slots={"options": ["Small", "In between"]}),
                  style=style, faces=synthetic_faces(Box(330, 654, 500, 200)))
    card = result.cards[0]
    layer = cards.fitted_card("choices", card.slots, style,
                              frames=card.end_frame - card.start_frame,
                              width=card.width, height=card.height)
    assert card.clearance["position"] == "headroom"
    assert layer.variables["text_px"] == 56
    assert json.loads(layer.variables["lines"]) == [["A) Small"], ["B) In between"]]
    assert card.height == 274 and card.y + card.height - 280 == 230


def test_choices_readable_refit_survives_standalone_render(tmp_path: Path, monkeypatch) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["choices"]
    result = plan(requests(template="choices", slots={"options": ["One", "Two"]}),
                  style=style, faces=synthetic_faces(Box(330, 654, 500, 200)))
    rendered = []

    def render(layer, *args, **kwargs):
        rendered.append(layer)
        return fake_render(layer, *args, **kwargs)

    monkeypatch.setattr(cards, "render_card", render)
    with paths.operation():
        cards.render_layers(result, style, tmp_path / "plan")
    assert rendered[0].variables["text_px"] == 56
    assert rendered[0].height == result.cards[0].height == 274


def test_choices_all_landing_rows_stay_below_safe_top() -> None:
    from yap_studio.cards import card_layer_plan

    style = resolve("preset:clean_minimal")
    slots = {"options": ["Choose time wisely", "One", "Spend time together"]}
    layer = card_layer_plan("choices", slots, style, frames=90)
    heights = json.loads(layer.variables["row_heights"])
    gaps = json.loads(layer.variables["row_gaps"]) + [0]
    cursor = layer.variables["stack_top_px"]
    for height, gap in zip(heights, gaps, strict=True):
        assert layer.y + cursor - 0.3 * height >= 280
        assert cursor + 1.3 * height <= layer.height - layer.variables["glow_pad_px"]
        cursor += height + gap


def test_choices_side_clears_every_head_and_the_caption_ceiling() -> None:
    from yap_studio.cards import card_layer_plan

    style = resolve("preset:clean_minimal")
    slots = {"options": ["One", "Two"]}
    layer = card_layer_plan("choices", slots, style)
    faces = [[Box(350, 650, 180, 200), Box(370, 620, 160, 190)]]
    placed, _, evidence = cards.place_card(
        layer, slots, style, faces, safe_order=("side", "headroom", "lower"), caption_top=800)
    assert evidence["position"] == "side"
    assert placed.y + placed.height <= 700
    assert placed.y + placed.variables["glow_pad_px"] >= 280
    assert all(gap_px(Box(placed.x, placed.y, placed.width, placed.height), head_box(face)) >= 24
               for face in faces[0])


def test_choices_empty_caption_ceiling_returns_blocked_plan() -> None:
    style = resolve("preset:clean_minimal")
    style.data["captions"].update(position="middle", top_px=700, size_px=180,
                                  shadow_blur_px=20, highlight_style="neon_glow")
    style.data["graphics"].update(caption_gap_px=400, card_kinds_on=["choices"])
    validate_style_data(style.data)
    result = plan(requests(template="choices", slots={"options": ["One", "Two"]}),
                  style=style, faces=synthetic_faces(Box(350, 650, 180, 200)))
    assert result.status == "needs_review"
    assert result.cards[0].clearance["status"] == "blocked"
    assert "no room above or beside the head" in (
        result.cards[0].clearance["reason"])


def test_choices_safe_order_zero_caption_ceiling_is_blocked() -> None:
    from yap_studio.cards import card_layer_plan

    style = resolve("preset:clean_minimal")
    slots = {"options": ["One", "Two"]}
    layer = card_layer_plan("choices", slots, style)
    _, placement, clearance = cards.place_card(
        layer, slots, style, [[Box(350, 650, 180, 200)]],
        safe_order=("side", "headroom", "lower"), caption_top=0,
    )
    assert placement == "top"
    assert clearance["status"] == "blocked"
    assert "no room above or beside the head" in (
        clearance["reason"])


@pytest.fixture
def enabled_card_style(kind: str) -> ResolvedStyle:
    from yap_studio import cards as card_library

    style = resolve("preset:clean_minimal")
    if kind in card_library.OPT_IN_KINDS:
        style.data["graphics"]["card_kinds_on"] = [kind]
    return style

@pytest.mark.parametrize("at,lane_at,relative", [
    ({"word": "t1:14"}, {"word": "t1:14"}, 1.0),
    (0.5, {"time": 12.9}, 0.5),
])
def test_gravity_drop_planner_keeps_anchors_and_fits_relative_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, at: object, lane_at: dict, relative: float,
) -> None:
    slots = {"drops": [{"text": "Words", "at": at}]}
    data = requests(template="gravity_drop", slots=slots)
    style = resolve("preset:clean_minimal")
    with pytest.raises(ValidationError, match="creator has not turned on gravity_drop"):
        plan(data, style=style)
    style.data["graphics"]["card_kinds_on"] = ["gravity_drop"]
    result = plan(data, style=style)
    assert result.lane()["elements"][0]["slots"] == {
        "drops": [{"text": "Words", "at": lane_at}],
    }
    assert result.cards[0].render_slots["drops"][0]["at"] == relative
    captured = []

    def render(layer, *args, **kwargs):
        captured.append(json.loads(layer.variables["drops"])[0]["at"])
        return fake_render(layer, *args, **kwargs)

    monkeypatch.setattr(cards, "render_card", render)
    with paths.operation():
        cards.render_layers(result, style, tmp_path)
    assert captured == [relative]
    assert slots["drops"][0]["at"] == at


@pytest.mark.parametrize("at", [-0.034, 1.9997, 2.0])
def test_gravity_drop_numeric_landing_outside_card_is_refused(at: float) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["gravity_drop"]
    data = requests(template="gravity_drop", start={"time": 0.117}, hold_s=2.0,
                    slots={"drops": [{"text": "Words", "at": at}]})
    del data["cards"][0]["end"]
    with pytest.raises(
        ValidationError, match="inside the card window.*move the card start or end",
    ) as caught:
        plan(data, style=style)
    assert caught.value.parameter == "cards[0].slots.drops[0].at"



@pytest.mark.parametrize("kind", (
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
))
def test_planner_accepts_new_card_kind(
    kind: str, tmp_path: Path, enabled_card_style: ResolvedStyle,
) -> None:
    from yap_studio import cards as card_library
    from yap_studio import ffmpeg

    slots = json.loads((card_library.template_dir(kind) / "preview.json").read_text())["slots"]
    slots = card_library.preview_videos(kind, slots, tmp_path)
    if kind == "callout_circle":
        image = tmp_path / "image.png"
        ffmpeg.run([
            "ffmpeg", "-nostdin", "-f", "lavfi", "-i", "color=c=blue:s=640x400",
            "-frames:v", "1", str(image),
        ], timeout_s=30)
        slots["image"] = str(image)
    if kind == "file_drop":
        from test_file_drop import clip_source

        slots["clip"] = str(clip_source(tmp_path))
    result = plan(requests(template=kind, slots=slots), style=enabled_card_style)
    assert len(result.cards) == 1 and result.cards[0].template == kind


@pytest.mark.parametrize("kind", ("prompt_bar", "comment_cta", "comment_drop", "icon_strike",
                                 "choices"))
def test_planner_requires_creator_opt_in(kind: str) -> None:
    from yap_studio import cards as card_library

    slots = json.loads((card_library.template_dir(kind) / "preview.json").read_text())["slots"]
    data = resolve("preset:clean_minimal").data
    data["graphics"]["card_kinds_off"] = []
    data["graphics"].pop("card_kinds_on", None)
    style = ResolvedStyle(data["name"], "test", data)
    with pytest.raises(ValidationError, match=f"creator has not turned on {kind} cards") as caught:
        plan(requests(template=kind, slots=slots), style=style)
    assert caught.value.parameter == "cards[0].template"


def test_planner_refuses_editor_kind_without_opt_in() -> None:
    with pytest.raises(
        ValidationError, match="creator has not turned on edit_timeline cards",
    ) as caught:
        plan(requests(template="edit_timeline", slots={}))
    assert caught.value.parameter == "cards[0].template"


def test_planner_rejects_switched_off_chart() -> None:
    from yap_studio import cards as card_library

    slots = json.loads((card_library.template_dir("chart") / "preview.json").read_text())["slots"]
    data = resolve("preset:clean_minimal").data
    data["graphics"]["card_kinds_off"] = ["chart"]
    style = ResolvedStyle(data["name"], "test", data)
    with pytest.raises(ValidationError, match="creator turned off chart cards") as caught:
        plan(requests(template="chart", slots=slots), style=style)
    assert caught.value.parameter == "cards[0].template"


@pytest.mark.parametrize("kind", ["playing_grid", "reel_feed"])
def test_scene_plan_exports_parts_and_reconstructs_its_render(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio import cards as card_library

    slots = json.loads((card_library.template_dir(kind) / "preview.json").read_text())["slots"]
    slots = card_library.preview_videos(kind, slots, tmp_path)
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = [kind]
    result = plan(requests(template=kind, slots=slots, start={"time": 0}, end={"time": 2.5}),
                  faces=synthetic_faces(), style=style)
    assert "box" not in result.lane()["elements"][0]
    layout = result.layout()
    if kind == "playing_grid":
        assert len(layout.elements) == len(slots["clips"])
        assert all(element.kind == "card" for element in layout.elements)
        assert check_layout(synthetic_faces(), layout).status == "pass"
    else:
        assert layout.elements[0].kind == "background"
    monkeypatch.setattr(cards, "render_card", fake_render)
    with paths.operation():
        rendered = cards.render_layers(result, style, tmp_path / "scene-plan")
    assert len(rendered) == 1 and rendered[0].width == 1080 and rendered[0].height == 1920


@pytest.mark.parametrize("kind", ["playing_grid", "reel_feed"])
def test_scene_plan_uses_measured_caption_ceiling_and_reconstructs_render(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio import cards as card_library
    from yap_studio.cards_video import caption_ceiling

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = [kind]
    slots = card_library.preview_videos(
        kind, {"labels": ["One", "Two", "Three", "Four", "Five", "Six"]}
        if kind == "playing_grid" else {}, tmp_path)
    faces = synthetic_faces(None)
    caption_top = 850 if kind == "playing_grid" else 1050
    layout = {"measured": True, "captions": [
        {"start_frame": 30, "end_frame": 105, "boxes": [{"y": caption_top}]},
        {"start_frame": 0, "end_frame": 30, "boxes": [{"y": 300}]},
        {"start_frame": 105, "end_frame": 150, "boxes": [{"y": 300}]},
    ]}
    result = cards.plan_cards(
        requests(template=kind, slots=slots, start={"time": 1}, end={"time": 3.5}),
        synthetic_words(), style, frames=FRAMES, faces=faces, caption_layout=layout,
    )
    ceiling = caption_ceiling(style, caption_top)
    if kind == "playing_grid":
        expected, _ = card_library.grid_card_plan(slots, style, [()] * 75, frames=75,
                                                  ceiling=ceiling, id="g-stat", start_frame=30)
    else:
        expected = card_library.card_layer_plan(kind, slots, style, frames=75, ceiling=ceiling)
    card, = result.cards
    assert card.render_key == cards._render_key(expected)
    rendered = []

    def render(layer, work, out, *, install=None):
        assert cards._render_key(layer) == cards._render_key(expected)
        rendered.append(layer)
        return fake_render(layer, work, out, install=install)

    monkeypatch.setattr(cards, "render_card", render)
    with paths.operation():
        cards.render_layers(result, style, tmp_path / "scene-render")
    assert len(rendered) == 1


@pytest.mark.parametrize("kind", ["playing_grid", "reel_feed"])
def test_scene_plan_refuses_when_measured_captions_leave_no_room(kind: str, tmp_path: Path) -> None:
    from yap_studio import cards as card_library

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = [kind]
    slots = card_library.preview_videos(kind, {}, tmp_path)
    layout = {"measured": True, "captions": [
        {"start_frame": 30, "end_frame": 105,
         "boxes": [{"y": 180 if kind == "playing_grid" else 850}]},
    ]}
    with pytest.raises(ValidationError):
        cards.plan_cards(
            requests(template=kind, slots=slots, start={"time": 1}, end={"time": 3.5}),
            synthetic_words(), style, frames=FRAMES, faces=synthetic_faces(None),
            caption_layout=layout,
        )


@pytest.mark.parametrize("kind", ["edit_timeline", "storyboard"])
def test_edit_timeline_plans_same_box_and_refuses_standalone_layers(
    tmp_path: Path, kind: str, enabled_card_style: ResolvedStyle,
) -> None:
    from yap_studio import cards as card_library

    style = enabled_card_style
    result = plan(requests(template=kind, slots={}), style=style)
    placed = result.cards[0]
    layer = card_library.card_layer_plan(kind, {}, style,
                                         frames=placed.end_frame - placed.start_frame)
    assert (placed.width, placed.height) == (layer.width, layer.height)
    with pytest.raises(ValidationError, match=f"render {kind} cards through the project run"):
        cards.render_layers(result, style, tmp_path / "standalone")
    assert not (tmp_path / "standalone").exists()


def test_cards_without_video_keep_the_legacy_render_key(monkeypatch) -> None:
    from yap_studio import cards as card_library

    layer = card_library.card_layer_plan("text_tag", {"text": "A small step"},
                                        resolve("preset:clean_minimal"))
    original = cards.asdict
    with monkeypatch.context() as legacy:
        legacy.setattr(cards, "asdict", lambda value: {
            key: item for key, item in original(value).items() if key != "windows"
        })
        expected = cards._render_key(layer)
    assert cards._render_key(layer) == expected


def test_grid_planner_honors_explicit_behind_placement(tmp_path: Path) -> None:
    from yap_studio import cards as card_library

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["playing_grid"]
    slots = card_library.preview_videos("playing_grid", {}, tmp_path)
    result = plan(requests(template="playing_grid", slots=slots, placement="behind_head"),
                  faces=synthetic_faces(Box(0, 0, 1080, 1920)), style=style)
    assert result.cards[0].placement == "behind_head"
    assert result.lane()["elements"][0]["layer"] == "behind"
    assert all(element.face_windows == () for element in result.layout().elements)


def test_behind_grid_planner_refuses_zoom_over_the_head(tmp_path: Path) -> None:
    from yap_studio import cards as card_library
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["playing_grid"]
    slots = card_library.preview_videos("playing_grid", {}, tmp_path)
    data = requests(template="playing_grid", slots=slots, placement="behind_head")
    faces = synthetic_faces(Box(0, 0, 1080, 1920))
    assert plan(data, faces=faces, style=style).cards[0].placement == "behind_head"
    zoom = ResolvedElement("zoom", "zoom", "agent", 390, 400, "front",
                           {"peak_scale": 1.1, "center": [0.5, 0.5]})
    timeline = ResolvedTimeline((zoom,), (), (), FRAMES, 0)
    with pytest.raises(ValidationError, match="move the grid off the zoom, "
                       "or use the front layer"):
        cards.plan_cards(data, synthetic_words(), style, frames=FRAMES,
                         faces=faces, timeline=timeline)


def fail_publication(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    atomic_output = paths.atomic_output

    @contextlib.contextmanager
    def failing(target, *args, **kwargs):
        with atomic_output(target, *args, **kwargs) as temporary:
            if Path(target).name == name:
                raise OSError("injected publication failure")
            yield temporary

    monkeypatch.setattr(paths, "atomic_output", failing)


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_word_frames_top_placement_and_clearance_agree(preset: str) -> None:
    style, faces = resolve(f"preset:{preset}"), synthetic_faces()
    result = plan(faces=faces, style=style)
    card = result.cards[0]
    assert (card.start_frame, card.end_frame) == (372, 447)
    assert (card.start_s, card.end_s) == pytest.approx((12.4, 14.9))
    assert card.end.edge == "end"
    assert (card.x, card.y, card.width, card.height) == (20, 0, 1040, 360)
    assert card.clearance["status"] == "clear" and result.status == "ok"
    assert card.clearance["gap_px"] == pytest.approx(31.4)
    assert card.clearance["hair"] == DEFAULT_HAIR.to_dict()
    assert card.layer() == "front" and not result.warnings
    assert result.style == style.sha256()
    assert check_layout(faces, result.layout()).status == "pass"


@pytest.mark.parametrize("box", [Box(458, 450, 396, 396), Box(458, 100, 396, 300)])
def test_blocked_top_finds_clear_side_or_lower(box: Box) -> None:
    faces, style = synthetic_faces(box), resolve("preset:clean_minimal")
    result = plan(faces=faces, style=style)
    card = result.cards[0]
    hint = suggest_offset(
        face_boxes_per_frame(faces)[372:447], x=20, width=1040, height=360,
        band=top_band(style.data), margin_px=24, preferred_y=0,
    )
    assert hint.y is None
    assert card.clearance["position"] in {"side", "lower"}
    assert card.clearance["status"] == "clear" and result.status == "ok"
    assert card.clearance["gap_px"] >= 24
    assert check_layout(faces, result.layout()).status == "pass"


@pytest.mark.parametrize("box,position", [
    (Box(458, 1000, 396, 396), "headroom"), (Box(458, 100, 396, 300), "side"),
])
def test_flow_tags_face_and_caption_clearance(box: Box, position: str) -> None:
    from yap_studio import cards as library

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["flow_look"] = "tags"
    faces = synthetic_faces(box)
    slots = {"nodes": ["Record", "Edit", "Publish"]}
    result = plan(requests(template="flow", slots=slots), faces=faces, style=style)
    placed = result.cards[0]
    assert placed.clearance["position"] == position
    assert placed.clearance["status"] == "clear" and placed.clearance["gap_px"] >= 24
    assert check_layout(faces, result.layout()).status == "pass"
    layer = library.card_layer_plan("flow", slots, style)
    candidate, _, evidence = cards.place_card(
        layer, slots, style, face_boxes_per_frame(faces)[372:447], caption_top=1100,
        safe_order=("headroom", "side", "lower"),
    )
    assert evidence["status"] == "clear"
    assert candidate.y + candidate.height <= 1100 - style.data["graphics"]["caption_gap_px"]


@pytest.mark.parametrize("face,caption_top,position,size", [
    (Box(330, 642, 500, 200), 1400, "headroom", 52),
    (Box(458, 100, 396, 300), 1400, "side", 60),
    (Box(180, 400, 720, 200), 1400, "lower", 60),
    (Box(180, 400, 720, 200), 700, None, 60),
])
def test_flow_tags_safe_positions_and_refit(
    face: Box, caption_top: int, position: str | None, size: int,
) -> None:
    from yap_studio.cards import card_layer_plan

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["flow_look"] = "tags"
    slots = {"nodes": ["One", "Two"]}
    layer = card_layer_plan("flow", slots, style)
    placed, _, evidence = cards.place_card(layer, slots, style, [[face]],
                                          safe_order=("headroom", "side", "lower"),
                                          caption_top=caption_top)
    if position is None:
        assert evidence["status"] == "blocked"
        assert evidence["reason"] == (
            "no room above or beside the head for the tags at a readable size; "
            "use shorter tags or move the card to another moment with more room above the head")
        assert "another moment" in evidence["reason"]
        return
    assert evidence["status"] == "clear" and evidence["position"] == position
    assert placed.variables["text_px"] == size and placed.y >= 280
    assert placed.y + placed.height <= caption_top - style.data["graphics"]["caption_gap_px"]
    assert gap_px(Box(placed.x, placed.y, placed.width, placed.height), head_box(face)) >= 24
    if position == "headroom":
        assert placed.height == 216 and placed.y + placed.height - 280 <= 218
        assert placed.x + placed.width / 2 == pytest.approx(head_box(face).x + head_box(face).w / 2,
                                                          abs=1)


def test_flow_tags_readable_refit_survives_standalone_render(tmp_path: Path, monkeypatch) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["flow_look"] = "tags"
    result = plan(requests(template="flow", slots={"nodes": ["One", "Two"]}),
                  style=style, faces=synthetic_faces(Box(330, 642, 500, 200)))
    assert result.cards[0].clearance["position"] == "headroom"
    rendered = []

    def render(layer, *args, **kwargs):
        rendered.append(layer)
        return fake_render(layer, *args, **kwargs)

    monkeypatch.setattr(cards, "render_card", render)
    with paths.operation():
        cards.render_layers(result, style, tmp_path / "plan")
    assert rendered[0].variables["text_px"] == 52
    assert rendered[0].height == result.cards[0].height == 216


@pytest.mark.parametrize("face", [Box(350, 650, 180, 200), Box(550, 650, 180, 200)])
def test_flow_tags_uses_wider_side_corridor(face: Box) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["flow_look"] = "tags"
    card = plan(requests(template="flow", slots={"nodes": ["One", "Two", "Three"]}),
                style=style, faces=synthetic_faces(face)).cards[0]
    assert card.clearance["position"] == "side" and card.y >= 280
    head = head_box(face)
    if face.x < 500:
        assert card.x >= head.x1 + 24
    else:
        assert card.x + card.width <= head.x - 24


@pytest.mark.parametrize("preserve_box,caption_top", [(False, 1400), (True, 1400), (False, 0)])
def test_flow_tags_safe_top_without_faces(preserve_box: bool, caption_top: int) -> None:
    from yap_studio.cards import card_layer_plan

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["flow_look"] = "tags"
    slots = {"nodes": ["One", "Two"]}
    layer = card_layer_plan("flow", slots, style)
    if preserve_box:
        layer = replace(layer, y=278)
    placed, _, evidence = cards.place_card(layer, slots, style, None,
                                          safe_order=("headroom", "side", "lower"),
                                          preserve_box=preserve_box, caption_top=caption_top)
    if preserve_box or caption_top == 0:
        assert evidence["status"] == "blocked" and "use shorter tags" in evidence["reason"]
    else:
        assert evidence["status"] == "unchecked" and placed.y == 280
        assert placed.x + placed.width / 2 == pytest.approx(540, abs=1)


def test_only_faces_inside_card_window_affect_placement() -> None:
    faces = synthetic_faces(Box(458, 450, 396, 396))
    clear = synthetic_faces()
    faces = replace(faces, frames=faces.frames[:372] + clear.frames[372:447] + faces.frames[447:])
    assert plan(faces=faces).cards[0].clearance["status"] == "clear"


def test_odd_hint_is_adjusted_with_clearance_and_loadable_overlays(tmp_path: Path) -> None:
    faces = synthetic_faces(Box(458, 101, 100, 100))
    result = plan(faces=faces)
    card = result.cards[0]
    assert card.y == 226 and card.clearance["status"] == "clear"
    assert "adjusted y 225 to 226" in card.clearance["reason"]
    assert card.clearance["gap_px"] == 25
    assert check_layout(faces, result.layout()).status == "pass"
    with paths.operation():
        outputs = cards.write_plan(result, tmp_path)
        assert load_cards(outputs["cards.json"])[0].y == 226


def test_motion_band_limit_uses_the_visible_top() -> None:
    from yap_studio.cards import CardPlan

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_motion"] = "streak"
    layer = CardPlan("quote", 20, 262, 1040, 360, 90, 30, {"glow_pad_px": 18}, (),
                     motion={"side": "right", "enter": True})
    faces = [[Box(300, 766, 480, 200)]] * 90
    placed, placement, evidence = cards.place_card(
        layer, {}, style, faces, caption_top=1000, refit_sides=False, face_windows=((0, 90),))
    assert placed.motion is not None
    assert placement == "top" and evidence["status"] == "clear"
    assert placed.y == 262 and placed.y + placed.variables["glow_pad_px"] == 280
    assert evidence["gap_px"] == 24 and evidence["card_motion"]["gap_px"] == 24


def test_only_clear_odd_top_position_uses_an_even_side_position() -> None:
    lower, upper = Box(458, 101, 100, 100), Box(458, 669, 100, 100)
    faces = synthetic_faces(lower)
    both = (FaceBox(lower, 1, "detected"), FaceBox(upper, 1, "detected"))
    faces = replace(faces, frames=(both,) * FRAMES)
    hint = suggest_offset(
        face_boxes_per_frame(faces)[372:447], x=20, width=1040, height=360,
        band=top_band(resolve("preset:clean_minimal").data), margin_px=24, preferred_y=0,
    )
    assert hint.y == 225
    result = plan(faces=faces)
    assert result.cards[0].y % 2 == 0 and result.cards[0].x % 2 == 0
    assert result.cards[0].clearance["position"] == "side"
    assert result.cards[0].clearance["status"] == "clear"
    assert result.status == "ok" and check_layout(faces, result.layout()).status == "pass"


def test_empty_measured_faces_keep_top_clear() -> None:
    card = plan(faces=synthetic_faces(None)).cards[0]
    assert card.y == 0 and card.clearance["status"] == "clear"
    assert card.clearance["gap_px"] is None


@pytest.mark.parametrize("faces", [None, synthetic_faces(measured=False)])
def test_unmeasured_is_unchecked(faces: FacesData | None) -> None:
    result = plan(faces=faces)
    card = result.cards[0]
    assert card.y == 0 and card.clearance["status"] == "unchecked"
    assert card.clearance["reason"]
    assert card.clearance["gap_px"] is None
    assert result.status == "ok"
    if faces:
        assert card.clearance["reason"] == faces.reason
        assert check_layout(faces, result.layout()).status == "unmeasured"


@pytest.mark.parametrize("face,y", [
    (DEFAULT_FACE, 126), (Box(458, 900, 396, 396), 280), (Box(458, 450, 396, 396), 0),
])
def test_requested_behind_prefers_clear_geometry(face: Box, y: int) -> None:
    faces = synthetic_faces(face)
    result = plan(requests(placement="behind_head"), faces=faces)
    card = result.cards[0]
    assert card.height == 360 and card.y % 2 == 0
    assert card.layer() == "front" and card.clearance["status"] == "clear"
    assert card.clearance["margin_px"] == 24
    assert card.clearance["hair"] == DEFAULT_HAIR.to_dict()
    assert card.clearance["gap_px"] == pytest.approx(
        gap_px(Box(card.x, card.y, card.width, card.height), head_box(face))
    )
    assert card.clearance["gap_px"] >= 24
    assert result.status == "ok" and not result.warnings
    assert result.overlays()[0].placement == "front"
    assert check_layout(faces, result.layout()).status == "pass"


@pytest.mark.parametrize("faces", [None, synthetic_faces(measured=False), synthetic_faces(None)])
def test_behind_request_without_faces_does_not_assume_safe_occlusion(
    faces: FacesData | None,
) -> None:
    card = plan(requests(placement="behind_head"), faces=faces).cards[0]
    assert card.y == 0 and card.placement == "top"
    assert card.clearance["status"] == (
        "clear" if faces and faces.status == "measured" else "unchecked"
    )
    assert card.clearance["reason"]
    assert card.clearance["gap_px"] is None and card.clearance["margin_px"] == 24
    assert card.clearance["hair"] == DEFAULT_HAIR.to_dict()


def test_behind_request_uses_all_heads_only_inside_card_window() -> None:
    faces = synthetic_faces(Box(458, 100, 396, 396))
    lower = FaceBox(Box(458, 900, 396, 396), 1, "detected")
    highest = FaceBox(DEFAULT_FACE, 1, "detected")
    faces = replace(
        faces,
        frames=faces.frames[:372] + ((lower,),) * 30 + ((lower, highest),)
        + ((lower,),) * 44 + faces.frames[447:],
    )
    card = plan(requests(placement="behind_head"), faces=faces).cards[0]
    assert card.y == 0 and card.clearance["status"] == "clear"
    assert card.clearance["gap_px"] == pytest.approx(31.4)


def test_overlapping_graphics_name_both_ids_and_overlap() -> None:
    data = requests()
    data["cards"].append({**copy.deepcopy(data["cards"][0]), "id": "second"})
    with pytest.raises(ValidationError, match="g-stat.*second.*overlap.*372, 447"):
        plan(data)


def test_touching_windows_do_not_overlap() -> None:
    data = requests(start={"time": 0}, end={"time": 2.5})
    data["cards"].append(
        {
            **copy.deepcopy(data["cards"][0]),
            "id": "second",
            "start": {"time": 2.5},
            "end": {"time": 5},
        }
    )
    assert len(plan(data).cards) == 2


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"slots": {"end": "thirty-seven"}}, "slots.end"),
        (
            {
                "template": "terminal",
                "slots": {"command": "pwd"},
                "start": {"time": 0},
                "end": {"time": 1},
            },
            "frames",
        ),
    ],
)
def test_slot_and_minimum_duration_errors_name_card(changes: dict, field: str) -> None:
    with pytest.raises(ValidationError) as error:
        plan(requests(**changes))
    assert field in error.value.parameter and "g-stat" in error.value.parameter


@pytest.mark.parametrize("duration, phrase", [(1, "shorter"), (4, "longer"), (8, "longer")])
def test_hold_warnings(duration: float, phrase: str) -> None:
    result = plan(requests(start={"time": 0}, end={"time": duration}))
    assert len(result.warnings) == 1 and phrase in result.warnings[0]
    assert "g-stat" in result.warnings[0]


def test_hold_violation() -> None:
    with pytest.raises(ValidationError, match="g-stat.*exceeds 8 s"):
        plan(requests(start={"time": 0}, end={"time": 8.1}))


@pytest.mark.parametrize("duration,frames,warning", [
    (1.9999, 60, None), (59 / 30, 59, "1.96667 s is shorter than 2 s"),
    (3.0001, 90, None), (8.0001, 240, "8 s is longer than 3 s"),
])
def test_hold_checks_use_rendered_frames(duration, frames, warning) -> None:
    result = plan(requests(start={"time": 0}, end={"time": duration}))
    assert result.cards[0].end_frame - result.cards[0].start_frame == frames
    assert result.warnings == (() if warning is None else (f"g-stat: hold {warning}",))


def test_default_and_explicit_hold_are_written_to_lane() -> None:
    data = requests(start={"time": 1})
    del data["cards"][0]["end"]
    result = plan(data)
    assert result.cards[0].hold_s == resolve("preset:clean_minimal").data["graphics"]["hold_s"]
    assert result.lane()["elements"][0]["hold_s"] == result.cards[0].hold_s
    data["cards"][0]["hold_s"] = 2.75
    assert plan(data).cards[0].end_frame == 113


@pytest.mark.parametrize("frames, fps", [(FRAMES - 1, Fraction(30)), (FRAMES, Fraction(24))])
@pytest.mark.parametrize("measured", [True, False])
def test_other_cut_faces_refused(frames: int, fps: Fraction, measured: bool) -> None:
    with pytest.raises(ValidationError, match="faces.video.*same cut"):
        plan(faces=synthetic_faces(frames=frames, fps=fps, measured=measured))


@pytest.mark.parametrize("start, end", [(-0.001, 2), (0, 0.001), (30, 31), (29, 31), (2, 1)])
def test_invalid_frame_windows(start: float, end: float) -> None:
    with pytest.raises(ValidationError):
        plan(requests(start={"time": start}, end={"time": end}))


def test_last_exclusive_frame_is_allowed() -> None:
    assert plan(requests(start={"time": 27.5}, end={"time": 30})).cards[0].end_frame == FRAMES


def test_bounds_are_checked_after_verify_frame_rounding() -> None:
    data = requests(start={"word": "t1:0", "offset_s": -6.401}, end={"time": 2.5})
    assert plan(data).cards[0].start_frame == 0
    assert plan(requests(start={"time": 27.5}, end={"time": 30.001})).cards[0].end_frame == FRAMES


def test_hold_violation_uses_rendered_frames_at_the_limit() -> None:
    result = plan(requests(start={"time": 0}, end={"time": 8.001}))
    assert result.cards[0].end_frame == 240
    with pytest.raises(ValidationError, match="exceeds 8 s"):
        plan(requests(start={"time": 0}, end={"time": 8.02}))


def test_parsed_requests_are_revalidated_and_do_not_share_slots() -> None:
    source = requests()
    parsed = cards.parse_requests(source)
    result = cards.plan_cards(
        parsed, synthetic_words(), resolve("preset:clean_minimal"), frames=FRAMES
    )
    source["cards"][0]["slots"]["end"] = 999
    parsed[0].slots["end"] = 123
    assert result.cards[0].slots["end"] == 37
    invalid = replace(parsed[0], id="../escape")
    with pytest.raises(ValidationError, match="safe layer filename"):
        cards.plan_cards(
            (invalid,), synthetic_words(), resolve("preset:clean_minimal"), frames=FRAMES
        )


@pytest.mark.parametrize("frames, fps", [(0, 30), (FRAMES, 0), (FRAMES, True)])
def test_frame_clock_must_be_positive_integers(frames: int, fps: int) -> None:
    with pytest.raises(ValidationError):
        cards.plan_cards(
            requests(), synthetic_words(), resolve("preset:clean_minimal"), frames=frames, fps=fps
        )


def test_empty_plan_contracts(tmp_path: Path) -> None:
    result = plan({"schema": cards.REQUESTS_SCHEMA, "cards": []})
    assert result.status == "ok" and result.cards == () and result.warnings == ()
    assert parse_document("timeline", result.lane(), lane="agent").elements == ()
    with paths.operation():
        outputs = cards.write_plan(result, tmp_path)
        assert load_cards(outputs["cards.json"]) == ()


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("unknown", 1, "cards[0].unknown"),
        ("slots", [], "cards[0].slots"),
        ("id", "", "cards[0].id"),
        ("id", "../escape", "cards[0].id"),
        ("template", "missing", "cards[0].template"),
        ("placement", "front", "cards[0].placement"),
        ("start", {"word": "bad"}, "cards[0].start.word"),
        ("start", {"word": "t1:12", "unknown": 1}, "cards[0].start.unknown"),
        ("start", {"time": True}, "cards[0].start.time"),
    ],
)
def test_strict_requests(field: str, value: object, expected: str) -> None:
    with pytest.raises(ValidationError) as error:
        cards.parse_requests(requests(**{field: value}))
    assert error.value.parameter == expected


@pytest.mark.parametrize("hold", [0, -1, True, float("nan"), float("inf")])
def test_hold_must_be_positive_finite_number(hold: object) -> None:
    data = requests(hold_s=hold)
    del data["cards"][0]["end"]
    with pytest.raises(ValidationError, match="cards\\[0\\].hold_s"):
        cards.parse_requests(data)


def test_duplicate_ids_and_conflicting_end_hold() -> None:
    data = requests()
    data["cards"].append(copy.deepcopy(data["cards"][0]))
    with pytest.raises(ValidationError, match="duplicate id"):
        cards.parse_requests(data)
    with pytest.raises(ValidationError, match="use end or hold_s"):
        cards.parse_requests(requests(hold_s=2))


def test_missing_start_and_unknown_document_key() -> None:
    data = requests()
    del data["cards"][0]["start"]
    with pytest.raises(ValidationError, match="cards\\[0\\].start"):
        cards.parse_requests(data)
    with pytest.raises(ValidationError, match="requests.foreign"):
        cards.parse_requests({**requests(), "foreign": 1})


@pytest.mark.parametrize("key", ["requests", "foreign"])
def test_unknown_request_document_key_names_the_cards_list(key: str) -> None:
    with pytest.raises(ValidationError, match=rf"requests\.{key}.*cards.*list"):
        cards.parse_requests({**requests(), key: []})


def test_lane_layout_and_overlays_round_trip(tmp_path: Path) -> None:
    data = requests()
    data["cards"].append(
        {
            "id": "behind",
            "template": "stat",
            "slots": {"end": 10},
            "start": {"time": 1},
            "hold_s": 2.5,
            "placement": "behind_head",
        }
    )
    result = plan(data, faces=synthetic_faces())
    lane = parse_document("timeline", result.lane(), lane="agent")
    assert [element.fields["layer"] for element in lane.elements] == ["front", "front"]
    assert lane.words_sha256 == synthetic_words().words_sha256
    assert [element.id for element in result.layout().elements] == ["g-stat", "behind"]
    assert result.layout().clearance_px == 24
    assert result.overlays("alpha")[0].layer == Path("alpha/g-stat.mov")
    with paths.operation():
        outputs = cards.write_plan(result, tmp_path / "out")
    assert set(outputs) == {"agent-lane.json", "graphics-plan.json", "layout.json", "cards.json"}
    assert load_layout(outputs["layout.json"]) == result.layout()
    loaded = load_cards(outputs["cards.json"])
    assert [card.placement for card in loaded] == ["front", "front"]
    assert loaded[0].layer == tmp_path / "out/layers/g-stat.mov"
    assert (
        parse_document(
            "timeline", json.loads(outputs["agent-lane.json"].read_bytes()), lane="agent"
        )
        == lane
    )
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert {k: v for k, v in receipt.items() if k != "outputs"} == result.to_dict()
    assert {r["path"] for r in receipt["outputs"]} == {
        "agent-lane.json",
        "layout.json",
        "cards.json",
    }
    assert all(r["sha256"] == sha256_file(tmp_path / "out" / r["path"]) for r in receipt["outputs"])


def test_rerun_replaces_only_matching_owned_outputs(tmp_path: Path) -> None:
    with paths.operation():
        outputs = cards.write_plan(plan(), tmp_path)
    replacement = plan(requests(slots={"end": 42}))
    with paths.operation():
        cards.write_plan(replacement, tmp_path)
    assert json.loads(outputs["cards.json"].read_bytes())["cards"][0]["id"] == "g-stat"
    assert json.loads(outputs["graphics-plan.json"].read_bytes())["cards"][0]["slots"] == {
        "end": 42
    }
    before = {name: file.read_bytes() for name, file in outputs.items()}
    outputs["layout.json"].write_text('{"schema": "yap-studio/clearance-layout@1"}')
    with paths.operation(), pytest.raises(OutputPathError, match="matching its current hash"):
        cards.write_plan(plan(), tmp_path)
    assert outputs["agent-lane.json"].read_bytes() == before["agent-lane.json"]
    assert outputs["graphics-plan.json"].read_bytes() == before["graphics-plan.json"]


@pytest.mark.parametrize(
    "name",
    ["agent-lane.json", "graphics-plan.json", "layout.json", "cards.json", "layers/g-stat.mov"],
)
def test_foreign_output_refused_before_writes(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"schema": "yap-studio/foreign@1"}')
    with paths.operation(), pytest.raises(OutputPathError):
        cards.write_plan(plan(), tmp_path)
    assert target.read_text() == '{"schema": "yap-studio/foreign@1"}'
    assert not (tmp_path / "agent-lane.json").exists() or name == "agent-lane.json"


def test_input_collision_and_symlink_outputs_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with paths.operation():
        paths.protect_input(out / "cards.json")
        with pytest.raises(OutputPathError, match="input file"):
            cards.write_plan(plan(), out)
    target = tmp_path / "foreign.json"
    target.write_text("foreign")
    (out / "graphics-plan.json").symlink_to(target)
    with paths.operation(), pytest.raises(OutputPathError, match="symlink"):
        cards.write_plan(plan(), out)
    assert target.read_text() == "foreign"


def test_layer_publication_with_stub_renderer_has_hashes_and_owned_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.render.hyperframes import LayerResult

    calls = []

    def render(layer, work_dir, out, *, install=None):
        calls.append((layer, work_dir, install))
        out.write_bytes(f"alpha layer {len(calls)}".encode())
        return LayerResult(out, layer.frames, layer.width, layer.height, layer.frames / layer.fps)

    monkeypatch.setattr(cards, "render_card", render)
    style, result = resolve("preset:clean_minimal"), plan()
    with paths.operation():
        first = cards.render_layers(result, style, tmp_path, install="stub")
    assert len(first) == 1 and first[0].path == tmp_path / "layers/g-stat.mov"
    assert first[0].frames == 75
    assert calls[0][0].frames == 75 and calls[0][2] == "stub"
    assert not calls[0][1].parent.exists() and not (tmp_path / ".work").exists()
    with paths.operation():
        second = cards.render_layers(result, style, tmp_path, install="stub")
        cards.write_plan(result, tmp_path)
    assert second[0].path.read_bytes() == b"alpha layer 2"
    receipt = json.loads((tmp_path / "graphics-plan.json").read_bytes())
    assert {r["path"] for r in receipt["outputs"]} == {
        "agent-lane.json",
        "layout.json",
        "cards.json",
        "layers/g-stat.mov",
    }
    assert all(r["sha256"] == sha256_file(tmp_path / r["path"]) for r in receipt["outputs"])
    second[0].path.write_bytes(b"foreign movie")
    with paths.operation(), pytest.raises(OutputPathError):
        cards.render_layers(result, style, tmp_path)
    assert len(calls) == 2 and not (tmp_path / ".work").exists()


def test_failed_stub_render_removes_work_and_leaves_no_alpha_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def render(layer, work_dir, out, *, install=None):
        raise ValidationError("render", "stub failure")

    monkeypatch.setattr(cards, "render_card", render)
    with paths.operation(), pytest.raises(ValidationError, match="stub failure"):
        cards.render_layers(plan(), resolve("preset:clean_minimal"), tmp_path)
    assert not (tmp_path / ".work").exists()
    assert not (tmp_path / "layers/g-stat.mov").exists()


def test_second_render_failure_preserves_existing_layers_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.render.hyperframes import LayerResult

    data = requests()
    data["cards"].append(
        {
            "id": "second",
            "template": "stat",
            "slots": {"end": 9},
            "start": {"time": 0},
            "hold_s": 2.5,
        }
    )
    result, style = plan(data), resolve("preset:clean_minimal")
    calls = []

    def render(layer, work_dir, out, *, install=None):
        calls.append(layer)
        if len(calls) == 4:
            raise ValidationError("render", "second layer failed")
        out.write_bytes(f"alpha {len(calls)}".encode())
        return LayerResult(out, layer.frames, layer.width, layer.height, layer.frames / layer.fps)

    monkeypatch.setattr(cards, "render_card", render)
    with paths.operation():
        cards.render_layers(result, style, tmp_path)
    before = {
        file: file.read_bytes()
        for file in [
            tmp_path / "layers/g-stat.mov",
            tmp_path / "layers/second.mov",
            tmp_path / "graphics-plan.json",
        ]
    }
    with paths.operation(), pytest.raises(ValidationError, match="second layer failed"):
        cards.render_layers(result, style, tmp_path)
    assert all(file.read_bytes() == content for file, content in before.items())
    assert not (tmp_path / ".work").exists()


def test_render_refuses_changed_plan_or_style(tmp_path: Path) -> None:
    with paths.operation():
        cards.write_plan(plan(), tmp_path)
    with (
        paths.operation(),
        pytest.raises(ValidationError, match="match the planned resolved style"),
    ):
        cards.render_layers(plan(), resolve("preset:electric"), tmp_path)
    with paths.operation(), pytest.raises(ValidationError, match="write this plan"):
        cards.render_layers(
            plan(requests(slots={"end": 42})), resolve("preset:clean_minimal"), tmp_path
        )


@pytest.mark.parametrize("change", ["slots", "frames", "fps", "style"])
def test_replan_changed_pixels_deletes_owned_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    style, original = resolve("preset:clean_minimal"), plan()
    with paths.operation():
        cards.render_layers(original, style, tmp_path)
    data = requests(slots={"end": 42}) if change == "slots" else requests()
    if change == "frames":
        data["cards"][0]["end"] = {"time": 14}
    replacement = cards.plan_cards(
        data, synthetic_words(), resolve("preset:electric") if change == "style" else style,
        frames=FRAMES, fps=24 if change == "fps" else 30,
    )
    assert replacement.cards[0].render_key != original.cards[0].render_key
    with paths.operation():
        outputs = cards.write_plan(replacement, tmp_path)
    assert not (tmp_path / "layers/g-stat.mov").exists()
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert receipt["cards"][0]["layer"] == replacement.cards[0].to_dict()["layer"]
    assert receipt["cards"][0]["layer"]["status"] == "missing"
    assert receipt["cards"][0]["layer"]["sha256"] is None
    assert all(item["path"] != "layers/g-stat.mov" for item in receipt["outputs"])


def test_replan_identical_pixels_keeps_current_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    original = plan()
    with paths.operation():
        cards.render_layers(original, resolve("preset:clean_minimal"), tmp_path)
    target = tmp_path / "layers/g-stat.mov"
    before = target.read_bytes()
    with paths.operation():
        outputs = cards.write_plan(plan(), tmp_path)
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert target.read_bytes() == before
    assert receipt["cards"][0]["layer"] == {
        "path": "layers/g-stat.mov", "render_key": original.cards[0].render_key,
        "sha256": sha256_file(target), "status": "current",
    }
    assert {item["path"]: item["sha256"] for item in receipt["outputs"]}["layers/g-stat.mov"] == (
        sha256_file(target)
    )


def test_render_key_uses_canonical_json_and_asset_contents(tmp_path: Path) -> None:
    layer = cards.card_layer_plan("stat", {"end": 37}, resolve("preset:clean_minimal"), frames=75)
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first.write_bytes(b"image one")
    second.write_bytes(first.read_bytes())
    original = replace(layer, assets=(("image.png", first),))
    reordered = replace(
        layer, variables=dict(reversed(list(layer.variables.items()))),
        assets=(("image.png", second),),
    )
    assert cards._render_key(original) == cards._render_key(reordered)
    assert cards._render_key(original) != cards._render_key(
        replace(layer, assets=(("renamed.png", first),))
    )
    second.write_bytes(b"image two")
    assert cards._render_key(original) != cards._render_key(reordered)


def test_render_key_tracks_browser_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = cards.card_layer_plan("stat", {"end": 37}, resolve("preset:clean_minimal"), frames=75)
    software = cards._render_key(layer)
    monkeypatch.setattr(hyperframes, "CARD_BROWSER_GPU", True)
    assert cards._render_key(layer) != software


def test_replan_retires_layer_with_old_render_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    with monkeypatch.context() as legacy:
        legacy.setattr(hyperframes, "CARD_BROWSER_GPU", True)
        original = plan()
        with paths.operation():
            cards.render_layers(original, resolve("preset:clean_minimal"), tmp_path)
    target = tmp_path / "layers/g-stat.mov"
    assert target.exists()
    previous = json.loads((tmp_path / "graphics-plan.json").read_bytes())
    assert previous["cards"][0]["layer"]["status"] == "current"
    replacement = plan()
    assert replacement.cards[0].render_key != original.cards[0].render_key
    with paths.operation():
        outputs = cards.write_plan(replacement, tmp_path)
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert not target.exists()
    assert receipt["cards"][0]["layer"] == replacement.cards[0].to_dict()["layer"]
    assert receipt["cards"][0]["layer"]["status"] == "missing"
    assert all(item["path"] != "layers/g-stat.mov" for item in receipt["outputs"])


def test_render_key_tracks_template_tree_paths_and_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "template"
    shutil.copytree(cards.template_dir("stat"), root)
    monkeypatch.setattr(cards, "template_dir", lambda template: root)
    monkeypatch.setattr(cards, "render_card", fake_render)
    original = plan()
    with paths.operation():
        cards.render_layers(original, resolve("preset:clean_minimal"), tmp_path / "out")
    asset = root / "nested/asset.txt"
    asset.parent.mkdir()
    asset.write_text("first")
    added = plan()
    assert added.cards[0].render_key != original.cards[0].render_key
    asset.write_text("second")
    edited = plan()
    assert edited.cards[0].render_key != added.cards[0].render_key
    asset.rename(asset.with_name("renamed.txt"))
    renamed = plan()
    assert renamed.cards[0].render_key != edited.cards[0].render_key
    with paths.operation(), pytest.raises(ValidationError, match="pixel inputs changed"):
        cards.render_layers(original, resolve("preset:clean_minimal"), tmp_path / "out")
    with paths.operation():
        outputs = cards.write_plan(renamed, tmp_path / "out")
    assert not (tmp_path / "out/layers/g-stat.mov").exists()
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert receipt["cards"][0]["layer"] == renamed.cards[0].to_dict()["layer"]


@pytest.mark.parametrize(
    "face_name",
    [face.file for name in ("Inter", "Space Mono") for face in fonts.BUNDLED[name].faces],
)
def test_bundled_font_change_invalidates_render_key_and_owned_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, face_name: str
) -> None:
    style, original = resolve("preset:clean_minimal"), plan()
    layer = cards.card_layer_plan("stat", {"end": 37, "suffix": "%"}, style, frames=75)
    root = tmp_path / "fonts"
    fonts.copy_fonts(layer.fonts, root)
    monkeypatch.setattr(fonts, "FONTS_DIR", root)
    monkeypatch.setattr(cards, "render_card", fake_render)
    assert plan() == original
    out = tmp_path / "out"
    with paths.operation():
        cards.render_layers(original, style, out)
    face = root / face_name
    face.write_bytes(face.read_bytes() + b"changed bundled face")
    replacement = plan()
    assert replacement.cards[0].render_key != original.cards[0].render_key
    with paths.operation(), pytest.raises(ValidationError, match="pixel inputs changed"):
        cards.render_layers(original, style, out)
    with paths.operation():
        outputs = cards.write_plan(replacement, out)
    assert not (out / "layers/g-stat.mov").exists()
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert receipt["cards"][0]["layer"] == replacement.cards[0].to_dict()["layer"]
    assert all(item["path"] != "layers/g-stat.mov" for item in receipt["outputs"])


def test_render_key_ignores_unselected_bundled_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = cards.card_layer_plan("stat", {"end": 37}, resolve("preset:clean_minimal"), frames=75)
    layer = replace(layer, fonts=("Inter", "Unbundled Font", "Inter"))
    root = tmp_path / "fonts"
    fonts.copy_fonts(layer.fonts, root)
    monkeypatch.setattr(fonts, "FONTS_DIR", root)
    original = cards._render_key(layer)
    unselected = root / fonts.BUNDLED["Space Mono"].faces[0].file
    unselected.write_bytes(b"unselected bundled face")
    assert cards._render_key(layer) == original


def test_remove_owned_layer_and_add_card_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    original, style = plan(), resolve("preset:clean_minimal")
    with paths.operation():
        cards.render_layers(original, style, tmp_path)
        outputs = cards.write_plan(plan({"schema": cards.REQUESTS_SCHEMA, "cards": []}), tmp_path)
    assert not (tmp_path / "layers/g-stat.mov").exists()
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert receipt["cards"] == []
    assert all(item["path"] != "layers/g-stat.mov" for item in receipt["outputs"])
    with paths.operation():
        cards.write_plan(original, tmp_path)
        cards.render_layers(original, style, tmp_path)
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert receipt["cards"][0]["layer"]["status"] == "current"


@pytest.mark.parametrize("removed", [False, True])
@pytest.mark.parametrize("foreign_kind", ["bytes", "dangling_symlink"])
def test_replan_refuses_foreign_layer_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, removed: bool, foreign_kind: str
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    with paths.operation():
        cards.render_layers(plan(), resolve("preset:clean_minimal"), tmp_path)
    target = tmp_path / "layers/g-stat.mov"
    if foreign_kind == "bytes":
        target.write_bytes(b"foreign layer")
    else:
        target.unlink()
        target.symlink_to(tmp_path / "absent.mov")
    before = {file: file.read_bytes() for file in tmp_path.glob("*.json")}
    replacement = plan(
        {"schema": cards.REQUESTS_SCHEMA, "cards": []} if removed else requests(slots={"end": 42})
    )
    with paths.operation(), pytest.raises(OutputPathError):
        cards.write_plan(replacement, tmp_path)
    if foreign_kind == "bytes":
        assert target.read_bytes() == b"foreign layer"
    else:
        assert target.is_symlink() and target.readlink() == tmp_path / "absent.mov"
    assert all(file.read_bytes() == content for file, content in before.items())
    assert not (tmp_path / "receipt.pending.json").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_document_publication_retries_with_pending_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    if existing:
        monkeypatch.setattr(cards, "render_card", fake_render)
        with paths.operation():
            cards.render_layers(plan(), resolve("preset:clean_minimal"), tmp_path)
    before = {
        file.name: sha256_file(file) for file in tmp_path.rglob("*") if file.is_file()
    }
    replacement = plan(requests(slots={"end": 42}))
    with monkeypatch.context() as fault:
        fail_publication(fault, "layout.json")
        with paths.operation(), pytest.raises(OSError, match="injected publication failure"):
            cards.write_plan(replacement, tmp_path)
    assert json.loads((tmp_path / "agent-lane.json").read_bytes()) == replacement.lane()
    pending = tmp_path / "receipt.pending.json"
    journal = json.loads(pending.read_bytes())
    assert journal["schema"] == cards.PENDING_SCHEMA
    expected = {*cards._DOCUMENTS, "graphics-plan.json"}
    if existing:
        expected.add("g-stat.mov")
    assert {Path(item["path"]).name for item in journal["outputs"]} == expected
    for item in journal["outputs"]:
        target = Path(item["path"])
        assert target.is_absolute()
        assert item["old_sha256"] == before.get(target.name)
        if target.name == "g-stat.mov":
            assert item["new_sha256"] is None
        else:
            assert len(item["new_sha256"]) == 64
    assert not list(tmp_path.glob(".yap-owner-*"))
    with paths.operation():
        outputs = cards.write_plan(replacement, tmp_path)
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert receipt["cards"][0]["slots"]["end"] == 42
    assert receipt["cards"][0]["layer"]["status"] == "missing"
    assert all(
        item["sha256"] == sha256_file(tmp_path / item["path"]) for item in receipt["outputs"]
    )
    assert not (tmp_path / "layers/g-stat.mov").exists()
    assert not pending.exists()


@pytest.mark.parametrize("document", [None, *cards._DOCUMENTS])
def test_render_refuses_interrupted_replan_until_documents_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: str | None
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    style = resolve("preset:clean_minimal")
    original = plan(requests(start={"time": 0}, end={"time": 2.5}))
    replacement = plan(requests(start={"time": 0}, end={"time": 3}))
    with paths.operation():
        outputs = cards.write_plan(original, tmp_path)
    old_documents = {name: outputs[name].read_bytes() for name in cards._DOCUMENTS}
    old_manifest = outputs["graphics-plan.json"].read_bytes()
    with monkeypatch.context() as fault:
        fail_publication(fault, "graphics-plan.json")
        with paths.operation(), pytest.raises(OSError, match="injected publication failure"):
            cards.write_plan(replacement, tmp_path)
    assert load_cards(outputs["cards.json"])[0].end_frame == 90
    assert outputs["graphics-plan.json"].read_bytes() == old_manifest
    assert (tmp_path / "receipt.pending.json").is_file()
    if document is not None:
        # Both sides remain journal-owned; isolate each document comparison.
        for name, content in old_documents.items():
            if name != document:
                outputs[name].write_bytes(content)
        assert outputs[document].read_bytes() != old_documents[document]
    before = {
        file.relative_to(tmp_path): file.read_bytes()
        for file in tmp_path.rglob("*") if file.is_file()
    }

    def unexpected(*args, **kwargs):
        pytest.fail("interrupted plan must refuse before rendering or publishing")

    with monkeypatch.context() as guard:
        guard.setattr(cards, "render_card", unexpected)
        guard.setattr(cards, "_publish", unexpected)
        with paths.operation(), pytest.raises(ValidationError, match="rerun the plan.*recovery"):
            cards.render_layers(original, style, tmp_path)
    assert {
        file.relative_to(tmp_path): file.read_bytes()
        for file in tmp_path.rglob("*") if file.is_file()
    } == before
    assert not (tmp_path / "layers").exists()
    assert not (tmp_path / ".work").exists()
    assert not list(tmp_path.glob(".yap-owner-*"))
    with paths.operation():
        cards.write_plan(replacement, tmp_path)
        rendered = cards.render_layers(replacement, style, tmp_path)
    assert rendered[0].frames == 90
    receipt = json.loads(outputs["graphics-plan.json"].read_bytes())
    assert receipt["cards"][0]["end_frame"] == 90
    assert receipt["cards"][0]["layer"]["status"] == "current"
    assert all(
        item["sha256"] == sha256_file(tmp_path / item["path"]) for item in receipt["outputs"]
    )
    assert not (tmp_path / "receipt.pending.json").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_layer_publication_retries_with_pending_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    from yap_studio import ffmpeg

    calls = []

    def render(layer, work_dir, out, *, install=None):
        calls.append(layer)
        color = ("red", "blue", "green", "yellow", "purple", "orange")[len(calls) - 1]
        ffmpeg.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                f"color=c={color}@0.5:s=16x16:r={layer.fps},format=yuva444p10le",
                "-frames:v", "1", "-c:v", "prores_ks", "-profile:v", "4",
                "-pix_fmt", "yuva444p10le", str(out),
            ],
            timeout_s=10,
        )
        return LayerResult(out, 1, 16, 16, 1 / layer.fps)

    monkeypatch.setattr(cards, "render_card", render)
    data = requests()
    data["cards"].append(
        {"id": "second", "template": "stat", "slots": {"end": 9},
         "start": {"time": 0}, "hold_s": 2.5}
    )
    result, style = plan(data), resolve("preset:clean_minimal")
    with paths.operation():
        cards.write_plan(result, tmp_path)
        if existing:
            cards.render_layers(result, style, tmp_path)
    old_manifest = (tmp_path / "graphics-plan.json").read_bytes()
    before = {file.name: sha256_file(file) for file in (tmp_path / "layers").glob("*.mov")}
    with monkeypatch.context() as fault:
        fail_publication(fault, "second.mov")
        with paths.operation(), pytest.raises(OSError, match="injected publication failure"):
            cards.render_layers(result, style, tmp_path)
    pending = tmp_path / "receipt.pending.json"
    journal = json.loads(pending.read_bytes())
    assert journal["schema"] == cards.PENDING_SCHEMA
    assert {Path(item["path"]).name for item in journal["outputs"]} == {
        "g-stat.mov", "second.mov", "graphics-plan.json",
    }
    assert (tmp_path / "graphics-plan.json").read_bytes() == old_manifest
    first = tmp_path / "layers/g-stat.mov"
    media = ffmpeg.probe(first)
    assert media.video_frames == 1 and media.pix_fmt.startswith("yuva")
    for item in journal["outputs"]:
        target = Path(item["path"])
        assert target.is_absolute()
        if target.name == "g-stat.mov":
            assert item["new_sha256"] == sha256_file(first)
            assert item["new_sha256"] != item["old_sha256"]
        if target.suffix == ".mov":
            assert item["old_sha256"] == before.get(target.name)
    assert not (tmp_path / ".work").exists()
    with paths.operation():
        rendered = cards.render_layers(result, style, tmp_path)
    receipt = json.loads((tmp_path / "graphics-plan.json").read_bytes())
    assert len(rendered) == 2
    assert all(card["layer"]["status"] == "current" for card in receipt["cards"])
    assert all(
        item["sha256"] == sha256_file(tmp_path / item["path"]) for item in receipt["outputs"]
    )
    assert not pending.exists() and not (tmp_path / ".work").exists()


@pytest.mark.parametrize("target_name", ["agent-lane.json", "layers/g-stat.mov"])
@pytest.mark.parametrize("operation", ["plan", "render"])
def test_pending_hash_matching_neither_side_does_not_authorize_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_name: str, operation: str
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    result, style = plan(), resolve("preset:clean_minimal")
    with paths.operation():
        cards.write_plan(result, tmp_path)
    target = tmp_path / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"foreign bytes")
    pending = tmp_path / "receipt.pending.json"
    pending.write_text(json.dumps({
        "schema": cards.PENDING_SCHEMA,
        "outputs": [{"path": str(target.resolve()), "old_sha256": "0" * 64,
                     "new_sha256": "1" * 64}],
    }))
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    with paths.operation(), pytest.raises(OutputPathError, match="matching its current hash"):
        if operation == "plan":
            cards.write_plan(result, tmp_path)
        else:
            cards.render_layers(result, style, tmp_path)
    assert all(file.read_bytes() == content for file, content in before.items())


@pytest.mark.render
def test_render_two_card_layers_and_exact_frame_counts(tmp_path: Path) -> None:
    from yap_studio import ffmpeg
    from yap_studio.render import hyperframes

    data = requests(start={"time": 0}, end={"time": 0.5})
    data["cards"].append(
        {
            "id": "tag",
            "template": "text_tag",
            "slots": {"text": "Hello"},
            "start": {"time": 1},
            "hold_s": 0.5,
        }
    )
    style = resolve("preset:clean_minimal")
    result = plan(data, style=style)
    with paths.operation():
        cards.write_plan(result, tmp_path)
        layers = cards.render_layers(
            result, style, tmp_path, install=hyperframes.ensure_installed()
        )
    assert len(layers) == 2
    for layer in layers:
        assert layer.path.is_file() and layer.frames == 15
        assert ffmpeg.probe(layer.path, count_frames=True).video_frames == 15
    assert not (tmp_path / ".work").exists()


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
def test_specs_rows_word_cues_and_long_hold(template: str, look: str, tmp_path: Path) -> None:
    from test_cards import ROWS_ON_CUE_SLOTS

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = look
    slots = {**ROWS_ON_CUE_SLOTS[template], "cues": [
        {"word": "t1:12", "segment": "story"}, {"word": "t1:14"}]}
    data = requests(template=template, slots=slots, start={"time": 12}, end={"time": 19})
    result = plan(data, style=style)
    assert not result.warnings
    assert result.cards[0].slots["cues"] == pytest.approx([0.4, 1.4])
    cards.write_plan(result, tmp_path / "plan")
    assert json.loads((tmp_path / "plan/agent-lane.json").read_text())["elements"][0][
        "slots"]["cues"] == [{"at": parse_anchor(cue).to_dict()} for cue in slots["cues"]]
    data["cards"][0]["end"] = {"time": 20}
    capped = plan(data, style=style).cards[0]
    assert capped.end_frame - capped.start_frame == 240
    data["cards"][0]["end"] = {"time": 20.1}
    with pytest.raises(ValidationError, match="8 s"):
        plan(data, style=style)
    tag = plan(requests(template="text_tag", slots={"text": "Hello"},
                        start={"time": 12}, end={"time": 19}), style=style)
    assert any("longer than" in warning for warning in tag.warnings)
    plain_stat = plan(requests(start={"time": 12}, end={"time": 19}), style=style)
    assert any("longer than" in warning for warning in plain_stat.warnings)


@pytest.mark.parametrize("template", ["specs_list", "stat"])
def test_specs_rows_word_cues_refused_when_together(template: str) -> None:
    slots = ({"title": "Plan", "rows": [{"name": "First", "value": "12"},
                                       {"name": "Second", "value": "25"}]}
             if template == "specs_list" else {"end": 37})
    with pytest.raises(ValidationError, match="turn on Rows on cue.*or remove cues"):
        plan(requests(template=template, slots={**slots, "cues": [{"word": "t1:11"}]}))


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
def test_specs_rows_safe_headroom_refit_and_render(
    template: str, look: str, tmp_path: Path, monkeypatch,
) -> None:
    from test_cards import ROWS_ON_CUE_SLOTS

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = look
    data = requests(template=template, slots=ROWS_ON_CUE_SLOTS[template])
    result = plan(data, style=style, faces=synthetic_faces(Box(330, 820, 500, 200)))
    card = result.cards[0]
    assert card.clearance["position"] == "headroom"
    fitted = cards.fitted_card(template, card.slots, style,
                               frames=card.end_frame - card.start_frame,
                               width=card.width, height=card.height)
    assert fitted.height == card.height
    assert card.y + fitted.variables["pad_top_px"] >= 280
    assert card.y + card.height <= 676
    rendered = []

    def render(layer, *args, **kwargs):
        rendered.append(layer)
        return fake_render(layer, *args, **kwargs)

    monkeypatch.setattr(cards, "render_card", render)
    with paths.operation():
        cards.render_layers(result, style, tmp_path / "rows-plan")
    assert rendered[0].variables == fitted.variables
    assert rendered[0].height == card.height
    layer = cards.fitted_card(template, card.slots, style,
                              frames=card.end_frame - card.start_frame)
    blocked, _, evidence = cards.place_card(
        layer, card.slots, style, [[Box(330, 450, 500, 700)]],
        safe_order=("headroom", "side", "lower"), caption_top=1000)
    assert evidence["status"] == "blocked"
    assert evidence["reason"] == (
        "no room above or beside the head for the rows at a readable size; "
        "use fewer or shorter rows or move the card to another moment "
        "with more room above the head")
    assert "another moment" in evidence["reason"]
    assert blocked.y + blocked.variables["pad_top_px"] >= 280


@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
def test_specs_rows_poetry_tiles_fit_includes_effect_padding(look: str) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(specs_rows=look, font="Geist", style="paper")
    slots = {"title": "Poetry tiles", "rows": [
        {"name": "Some", "value": "Small words"},
        {"name": "Some", "value": "In between words"}]}
    layer = cards.fitted_card("specs_list", slots, style, frames=90)
    placed, _, evidence = cards.place_card(
        layer, slots, style, [[Box(290, 665, 500, 200)]],
        safe_order=("headroom", "side", "lower"), caption_top=1503)
    assert placed.y + placed.variables["pad_top_px"] >= 280
    if look == "on_cue_glow":
        assert evidence["position"] == "headroom"
        assert (placed.variables["title_px"], placed.variables["row_px"]) == (76, 49)
        assert placed.height == 278 and placed.y + placed.height <= 521
    else:
        assert evidence["position"] == "headroom"
        assert placed.variables["title_px"] > 52 and placed.variables["row_px"] >= 44
        assert placed.y + placed.height <= 521


@pytest.mark.parametrize("reveal", ["wheel", "slot"])
def test_stat_glowing_rows_refuse_reel_reveals(reveal: str) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = "on_cue_glow"
    with pytest.raises(ValidationError) as caught:
        plan(requests(slots={"end": 37, "rows": ["12 first", "25 second"],
                             "reveal": reveal}), style=style)
    assert caught.value.parameter == "cards[0].slots.reveal (g-stat)"
    assert caught.value.detail == "use reveal count with the glowing rows look"


def test_stat_glowing_rows_count_still_plans() -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = "on_cue_glow"
    slots = {"end": 37, "rows": ["12 first", "25 second"], "reveal": "count"}
    result = plan(requests(slots=slots), style=style)
    assert result.cards[0].slots == slots


@pytest.mark.parametrize("look,rows", [
    ("together", None), ("on_cue", None), ("on_cue", ["12 first", "25 second"]),
    ("on_cue_glow", None), ("on_cue_glow", []),
])
@pytest.mark.parametrize("reveal", ["wheel", "slot"])
def test_stat_reel_reveals_still_plan_outside_glowing_rows(
    look: str, rows: list[str] | None, reveal: str,
) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = look
    slots = {"end": 37, "reveal": reveal}
    if rows is not None:
        slots["rows"] = rows
    result = plan(requests(slots=slots), style=style)
    assert result.cards[0].slots == slots


def test_shared_glow_source_changes_only_its_card_render_keys(tmp_path: Path, monkeypatch) -> None:
    from yap_studio import cards as compositions

    templates = tmp_path / "templates"
    shutil.copytree(compositions.template_dir("stat").parent, templates)
    monkeypatch.setattr(compositions, "_TEMPLATES", templates)
    before = plan().cards[0].render_key
    tag = requests(template="text_tag", slots={"text": "Hello"})
    tag_before = plan(tag).cards[0].render_key
    helper = templates / "glow-shadow.js"
    helper.write_text(helper.read_text() + "\n// Shared shadow changed.\n")
    assert plan().cards[0].render_key != before
    assert plan(tag).cards[0].render_key == tag_before
