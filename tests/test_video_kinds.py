"""Synthetic grid placement and inline feed contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import synth
from test_cards import video_clip as video_clip
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_stage_graphics import graphic

from yap_studio import cards, ffmpeg, receipt
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.project.status import ProjectContext
from yap_studio.stages.graphics import graphics_snapshot
from yap_studio.stages.timeline import ResolvedElement
from yap_studio.style import resolve


def _style():
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["playing_grid", "reel_feed"]
    return style


@pytest.mark.parametrize("changes,fix", [
    ({"clips": []}, "2..6"), ({"clips": ["self", "self"]}, "self is not allowed"),
    ({"labels": ["a", "b", "c"]}, "more labels than clips"),
    ({"labels": ["x" * 25]}, "0..24"), ({"stagger_s": 0.41}, "0.4"),
])
def test_grid_slot_refusals(video_clip: dict, changes: dict, fix: str) -> None:
    with pytest.raises(ValidationError, match=fix):
        cards.card_layer_plan("playing_grid", {"clips": [video_clip] * 2, **changes},
                              _style(), frames=75)


def test_grid_count_and_window_timing(video_clip: dict) -> None:
    style = _style()
    with pytest.raises(ValidationError, match="2..6"):
        cards.card_layer_plan("playing_grid", {"clips": [video_clip] * 7}, style)
    with pytest.raises(ValidationError, match="eight frames"):
        cards.card_layer_plan("playing_grid", {"clips": [video_clip] * 2}, style, frames=10)
    plan = cards.card_layer_plan("playing_grid", {"clips": [video_clip] * 3}, style, frames=75)
    assert (plan.x, plan.y, plan.width, plan.height) == (0, 0, 1080, 1920)
    assert [window["start"] for window in plan.windows] == [0, 4, 7]
    assert all(window["fade_in"] == 8 for window in plan.windows)
    assert all(window["start"] + window["frames"] == 75 for window in plan.windows)
    assert len({window["fade_out"] for window in plan.windows}) == 1


@pytest.mark.parametrize("stagger,start", [(None, 4), (0, 0), (0.2, 6), (0.4, 12)])
@pytest.mark.parametrize("short", [False, True])
def test_grid_timeline_uses_each_staggered_clip_window(
    layer_project: Path, stagger: float | None, start: int, short: bool,
) -> None:
    style_file = layer_project / "style.json"
    style = json.loads(style_file.read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = ["playing_grid"]
    style_file.write_text(json.dumps(style))
    clips = []
    for index, frames in enumerate((75, 75 - start - int(short))):
        source = synth.picture_source(layer_project / f"grid-{index}.mov",
                                      "color=c=gray:s=320x180:r=30", frames)
        clips.append({"path": str(source), "sha256": receipt.sha256_file(source)})
    slots = {"clips": clips, **({"stagger_s": stagger} if stagger is not None else {})}
    if short:
        with pytest.raises(ValidationError, match="the clip ends"):
            cards.card_layer_plan("playing_grid", slots, _style(), frames=75)
    else:
        plan = cards.card_layer_plan("playing_grid", slots, _style(), frames=75)
        assert [(window["start"], window["frames"]) for window in plan.windows] == [
            (0, 75), (start, 75 - start),
        ]
    slots["clips"] = [{**clip, "path": Path(clip["path"]).name} for clip in clips]
    set_elements(layer_project, graphic(template="playing_grid", slots=slots,
                                        start={"time": 0}, hold_s=2.5))
    timeline = ProjectContext(layer_project).resolved_timeline(allow_violations=True)
    findings = [item for item in timeline.findings if item.rule == "card_video_bounds"]
    assert bool(findings) == short
    if short:
        finding, = findings
        assert "slots.clips[1]" in finding.message
    else:
        timeline.require_valid()


def test_video_previews_keep_self_and_cache_each_requested_duration(tmp_path: Path) -> None:
    assert cards.preview_videos("reel_feed", {}, tmp_path) == {}
    assert not list(tmp_path.iterdir())
    short = cards.preview_videos("playing_grid", {}, tmp_path, frames=60)
    assert cards.preview_videos("playing_grid", {}, tmp_path, frames=60) == short
    long = cards.preview_videos("playing_grid", {}, tmp_path, frames=150)
    cards.card_layer_plan("playing_grid", long, _style(), frames=150)
    assert receipt.sha256_file(Path(short["clips"][0]["path"])) == short["clips"][0]["sha256"]


@pytest.mark.parametrize("on,expected", [(False, "FU"), (True, "F*")])
def test_feed_masks_derived_initials_only_when_on(on: bool, expected: str) -> None:
    style = _style()
    style.data["music"].update(bleep_swears=on, bleep_words_add=["fu"])
    plan = cards.card_layer_plan("reel_feed", {"handle": "Fred User"}, style, frames=75)
    assert plan.variables["handle"] == "Fred User"
    assert plan.variables["initials"] == expected


@pytest.mark.parametrize("on,expected", [(False, "100 views"), (True, "100 v****")])
def test_feed_masks_and_fits_derived_view_label(on: bool, expected: str) -> None:
    style = _style()
    style.data["music"].update(bleep_swears=on, bleep_words_add=["views"])
    plan = cards.card_layer_plan("reel_feed", {"views": "100"}, style, frames=75)
    assert plan.variables["views_text"] == expected
    from yap_studio.render.text_metrics import text_width_em

    assert text_width_em(expected, style.data["graphics"]["font"], 600) * plan.variables[
        "views_px"] <= plan.windows[0]["width"] - 102


def test_feed_slots_self_and_caption_fit(video_clip: dict) -> None:
    style = _style()
    plan = cards.card_layer_plan("reel_feed", {"caption": "Small steps. One at a time."}, style,
                                 frames=75)
    assert plan.windows[0]["source"] == {"kind": "self"}
    assert plan.variables["sound_on"] is True
    assert len(json.loads(plan.variables["caption_lines"])) <= 2
    media = cards.card_layer_plan("reel_feed", {"post": video_clip, "ending": "fade"},
                                  style, frames=60)
    assert media.variables["sound_on"] is False
    for slots, fix in (({"caption": "W" * 80}, "two lines"),
                       ({"handle": "x" * 31}, "0..30"),
                       ({"views": "x" * 13}, "0..12")):
        with pytest.raises(ValidationError, match=fix):
            cards.card_layer_plan("reel_feed", slots, style, frames=75)
    with pytest.raises(ValidationError, match="needs at least 2 s"):
        cards.card_layer_plan("reel_feed", {"ending": "fade"}, style, frames=59)


def test_reel_feed_handoff_plan_has_shared_even_motion_rects() -> None:
    from yap_studio.cards_video import window_rect

    plan = cards.card_layer_plan("reel_feed", {}, _style(), frames=90)
    assert plan.variables["ending"] == "handoff"
    window, = plan.windows
    assert window["fade_out"] == 0
    assert window["motion"] == {"start": 72, "frames": 18, "to": [0, 0, 1080, 1920],
                                 "ease": "cubic_in_out"}
    initial = tuple(window[key] for key in ("x", "y", "width", "height"))
    rects = [window_rect(window, frame) for frame in range(72, 90)]
    assert window_rect(window, 71) == rects[0] == initial
    assert 0 < rects[9][0] < initial[0] and 0 < rects[9][1] < initial[1]
    assert initial[2] < rects[9][2] < 1080 and initial[3] < rects[9][3] < 1920
    assert rects[-1] == (0, 0, 1080, 1920)
    assert all(type(value) is int and value % 2 == 0 for rect in rects for value in rect)
    template_window, = json.loads(plan.variables["windows"])
    assert template_window["rects"] == [list(rect) for rect in rects]
    assert "source" not in template_window


def test_reel_feed_handoff_requires_self_and_two_and_a_half_seconds(video_clip: dict) -> None:
    with pytest.raises(ValidationError, match="handoff continues this video; "
                       "use post self or ending fade") as caught:
        cards.card_layer_plan("reel_feed", {"post": video_clip}, _style(), frames=75)
    assert caught.value.parameter == "slots.ending"
    with pytest.raises(ValidationError, match="needs at least 2.5 s"):
        cards.card_layer_plan("reel_feed", {}, _style(), frames=74)
    plan = cards.card_layer_plan("reel_feed", {"post": video_clip, "ending": "fade"},
                                 _style(), frames=60)
    assert plan.variables["ending"] == "fade" and "motion" not in plan.windows[0]
    assert plan.windows[0]["fade_out"] > 0


def test_feed_caption_uses_a_light_style_color_over_the_scrim() -> None:
    style = _style()
    style.data["graphics"].update(paper_color="#101010", ink_color="#F4F3EE")
    plan = cards.card_layer_plan("reel_feed", {"caption": "Small steps"}, style, frames=75)
    assert plan.variables["overlay_ink"] == "#F4F3EE"


@pytest.mark.parametrize("text,expected", [("#FFFFFF", "ink"), ("#222222", "paper")])
@pytest.mark.parametrize("slots", [{}, {"backdrop": "auto"}])
def test_feed_auto_backdrop_contrasts_with_spoken_captions(
    text: str, expected: str, slots: dict,
) -> None:
    style = _style()
    style.data["captions"]["text_color"] = text
    style.data["graphics"].update(paper_color="#F6F1E7", ink_color="#222222")
    plan = cards.card_layer_plan("reel_feed", slots, style, frames=75)
    assert plan.variables["backdrop"] == expected
    assert cards._contrast(text, plan.variables[f"{expected}_color"]) >= 3


def test_feed_paper_refuses_unreadable_spoken_captions() -> None:
    style = _style()
    style.data["captions"]["text_color"] = "#FFFFFF"
    with pytest.raises(ValidationError, match="captions would be unreadable on paper; "
                       "use backdrop auto or ink") as caught:
        cards.card_layer_plan("reel_feed", {"backdrop": "paper"}, style, frames=75)
    assert caught.value.parameter == "slots.backdrop"


@pytest.mark.parametrize("slots", [{}, {"backdrop": "auto"}])
def test_feed_auto_refuses_when_neither_backdrop_has_caption_contrast(slots: dict) -> None:
    style = _style()
    style.data["captions"]["text_color"] = "#888888"
    style.data["graphics"].update(paper_color="#888888", ink_color="#888888")
    with pytest.raises(ValidationError, match="captions would be unreadable on paper or ink; "
                       "use backdrop dim or change the caption, paper or ink colour") as caught:
        cards.card_layer_plan("reel_feed", slots, style, frames=75)
    assert caught.value.parameter == "slots.backdrop"


@pytest.mark.parametrize("backdrop", ["paper", "ink", "dim"])
def test_feed_explicit_backdrops_keep_their_choice(backdrop: str) -> None:
    style = _style()
    style.data["captions"]["text_color"] = style.data["graphics"]["ink_color"]
    plan = cards.card_layer_plan("reel_feed", {"backdrop": backdrop}, style, frames=75)
    assert plan.variables["backdrop"] == backdrop


def test_grid_layout_clears_head_and_uses_largest_width() -> None:
    from yap_studio.cards_video import grid_layout

    face = Box(400, 700, 200, 250)
    tiles, evidence = grid_layout([16 / 9] * 6, [[face]], margin=24, ceiling=1500,
                                  top_bottom=640, id="grid-one")
    assert evidence["status"] == "clear"
    assert tiles[0]["width"] == 400
    boxes = [Box(tile["x"], tile["y"], tile["width"], tile["height"]) for tile in tiles]
    assert all(box.y >= 96 and box.y1 <= 1500 for box in boxes)
    assert all(gap_px(box, head_box(face)) >= 24 for box in boxes)
    assert all(gap_px(first, second) >= 16 for index, first in enumerate(boxes)
               for second in boxes[index + 1:])
    wide, _ = grid_layout([16 / 9] * 2, [], margin=24, ceiling=1500,
                          top_bottom=640, id="grid-one")
    assert wide[0]["width"] == 480


def test_grid_window_aspects_are_clamped_to_supported_pictures(tmp_path: Path) -> None:
    clips = []
    for index, size in enumerate(["64x192", "320x64"]):
        source = tmp_path / f"clip-{index}.mp4"
        ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"color=s={size}:r=30", "-t", "3", str(source)], timeout_s=30)
        clips.append({"path": str(source), "sha256": receipt.sha256_file(source)})
    plan = cards.card_layer_plan("playing_grid", {"clips": clips}, _style(), frames=75)
    for window, aspect in zip(plan.windows, [3 / 4, 16 / 9], strict=True):
        assert abs(window["height"] - window["width"] / aspect) < 2


def test_grid_window_uses_cropped_aspect_and_compact_shadow_padding(video_clip: dict) -> None:
    tiles = [{"x": x, "y": 96, "width": 320, "height": 376} for x in (24, 360)]
    clip = {**video_clip, "crop": [32, 16, 96, 96]}
    plan = cards.card_layer_plan("playing_grid", {"clips": [clip, clip], "labels": ["A", "B"]},
                                 _style(), frames=75, tiles=tiles)
    assert [(window["x"], window["y"], window["width"], window["height"])
            for window in plan.windows] == [(54, 118, 260, 260), (390, 118, 260, 260)]
    assert all(window["source"]["crop"] == clip["crop"] for window in plan.windows)


def test_grid_reserved_box_uses_compact_shadow_padding() -> None:
    from yap_studio.cards_video import grid_layout, tile_window

    packed, _ = grid_layout([16 / 9] * 2, [], margin=24, ceiling=1500, top_bottom=640,
                            id="grid", labels=["A", "B"])
    assert [(tile["width"], tile["height"]) for tile in packed] == [(480, 352)] * 2
    assert tile_window(packed[0], 16 / 9) == {"x": 54, "y": 118, "width": 420, "height": 236}
    assert packed[1]["x"] - packed[0]["x"] - packed[0]["width"] == 16


def test_grid_template_defaults_use_the_planned_padding() -> None:
    from yap_studio.cards_video import tile_window

    defaults = {item["id"]: item["default"]
                for item in cards.load_template("playing_grid")["variables"]}
    tiles, windows = (json.loads(defaults[name]) for name in ("tiles", "windows"))
    for tile, window in zip(tiles, windows, strict=True):
        assert {key: window[key] for key in ("x", "y", "width", "height")} == tile_window(
            tile, 16 / 9,
        )
        assert tile["height"] == window["height"] + 12 + 16 + 40 + 48


@pytest.mark.parametrize("accent", [None, "#A13467"])
def test_feed_progress_uses_the_brand_accent_or_ink(accent: str | None) -> None:
    style = _style()
    style.data["brand"]["accent_color"] = accent
    plan = cards.card_layer_plan("reel_feed", {}, style, frames=75)
    assert plan.variables["progress_color"] == (accent or style.data["graphics"]["ink_color"])


def test_grid_layout_refusal_unmeasured_and_behind() -> None:
    from yap_studio.cards_video import grid_layout

    head = Box(0, 0, 1080, 1920)
    options = {"margin": 24, "ceiling": 1500, "top_bottom": 640, "id": "grid-one"}
    with pytest.raises(ValidationError, match="no room around the face for 6 tiles") as caught:
        grid_layout([16 / 9] * 6, [[head]], **options)
    assert caught.value.parameter == "graphic.grid-one.clearance"
    tiles, evidence = grid_layout([16 / 9] * 6, None, reason="detector unavailable", **options)
    assert all(tile["y"] + tile["height"] <= 640 for tile in tiles)
    assert evidence["status"] == "unchecked" and evidence["reason"] == "detector unavailable"
    behind, _ = grid_layout([16 / 9] * 6, [[head]], layer="behind", **options)
    empty, _ = grid_layout([16 / 9] * 6, [], layer="behind", **options)
    assert behind == empty


def test_behind_grid_only_clears_heads_in_person_restoration_gaps() -> None:
    from yap_studio.cards_video import grid_layout

    options = {"margin": 24, "ceiling": 1500, "top_bottom": 640, "id": "grid-one",
               "layer": "behind", "start_frame": 40}
    faces = [[Box(0, 0, 1080, 1920)], [Box(0, 530, 240, 120)]]
    plain, _ = grid_layout([16 / 9] * 2, faces, **options)
    assert plain[0]["width"] == 480
    tiles, _ = grid_layout([16 / 9] * 2, faces, face_windows=((41, 42),), **options)
    assert tiles[0]["width"] == 360
    assert all(gap_px(Box(tile["x"], tile["y"], tile["width"], tile["height"]),
                      head_box(faces[1][0])) >= 24 for tile in tiles)
    for measured in (faces, None):
        with pytest.raises(ValidationError, match="move the grid off the zoom, "
                           "or use the front layer"):
            grid_layout([16 / 9] * 2, measured, face_windows=((40, 41),), **options)


def test_behind_grid_place_card_checks_zoom_heads(video_clip: dict) -> None:
    from yap_studio.planner.cards import place_card

    slots = {"clips": [video_clip, video_clip]}
    layer = cards.card_layer_plan("playing_grid", slots, _style(), frames=60)
    with pytest.raises(ValidationError, match="move the grid off the zoom, "
                       "or use the front layer"):
        place_card(layer, slots, _style(), [[Box(0, 0, 1080, 1920)]] * 60,
                   requested_layer="behind", start_frame=372, face_windows=((390, 400),))


def test_behind_grid_graphics_refuses_zoom_over_the_head(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_planner_cards import synthetic_faces

    from yap_studio.stages import graphics as stage

    style_file = layer_project / "style.json"
    style = json.loads(style_file.read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = ["playing_grid"]
    style_file.write_text(json.dumps(style))
    clip = cards.preview_clip(layer_project)
    clip["path"] = Path(clip["path"]).name
    grid = graphic(template="playing_grid", slots={"clips": [clip, clip]},
                   start={"time": 1.2}, layer="behind")
    set_elements(layer_project, grid)
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(Box(0, 0, 1080, 1920),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    assert snapshot.placement()["layer"] == "behind"
    zoom = ResolvedElement("zoom", "zoom", "agent", 48, 78, "front",
                           {"peak_scale": 1.1, "center": [0.5, 0.5]})
    monkeypatch.setattr(ProjectContext, "resolved_timeline", lambda self: replace(
        timeline, elements=(*timeline.elements, zoom),
    ))
    with pytest.raises(ValidationError, match="move the grid off the zoom, "
                       "or use the front layer") as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == "graphic.agent-tag.clearance"


@pytest.mark.parametrize("kind,layer,post", [
    ("playing_grid", "front", None), ("playing_grid", "behind", None),
    ("reel_feed", "front", None), ("reel_feed", "front", "self"),
])
def test_video_kind_graphics_placement_and_reconstruction(
    layer_project: Path, kind: str, layer: str, post: str | None,
) -> None:
    style = json.loads((layer_project / "style.json").read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = [kind]
    (layer_project / "style.json").write_text(json.dumps(style))
    slots = {}
    if post is not None:
        slots["post"] = post
    if kind == "playing_grid":
        clip = cards.preview_clip(layer_project)
        clip["path"] = Path(clip["path"]).name
        slots["clips"] = [clip, clip]
    set_elements(layer_project, graphic(template=kind, slots=slots,
                                        hold_s=2.5 if kind == "reel_feed" else 2, layer=layer))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    placement = snapshot.placement()
    assert placement["layer"] == layer
    assert (placement["x"], placement["y"], placement["width"], placement["height"]) == (
        0, 0, 1080, 1920,
    )
    if kind == "playing_grid":
        assert len(placement["windows"]) == len(placement["parts"]) == 2
        assert all(tile["gap_px"] is None for tile in placement["clearance"]["tiles"])
    else:
        assert placement["full_frame"] is True
        assert placement["backdrop"] == snapshot.plan.variables["backdrop"]
        assert placement["clearance"]["status"] == "full_frame"
        assert placement["windows"][0]["source"] == {"kind": "self"}
        assert placement["windows"][0]["motion"] == {
            "start_frame": 63, "frames": 18, "to": [0, 0, 1080, 1920], "ease": "cubic_in_out",
        }
    rebuilt, = graphics_snapshot(ProjectContext(layer_project), placements=[placement]).graphics
    assert rebuilt.placement() == placement


def test_feed_caption_color_changes_backdrop_receipt_and_digest(layer_project: Path) -> None:
    style_file = layer_project / "style.json"
    style = json.loads(style_file.read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = ["reel_feed"]
    set_elements(layer_project, graphic(template="reel_feed", slots={}, hold_s=2.5))
    snapshots = []
    for text in ("#FFFFFF", "#222222"):
        style.setdefault("captions", {})["text_color"] = text
        style_file.write_text(json.dumps(style))
        snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
        snapshots.append(snapshot)
    assert [snapshot.placement()["backdrop"] for snapshot in snapshots] == ["ink", "paper"]
    assert snapshots[0].digest != snapshots[1].digest
