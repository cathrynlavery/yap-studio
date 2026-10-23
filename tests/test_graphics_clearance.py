"""Synthetic planner-to-assembly clearance contracts without the renderer."""

from __future__ import annotations

import copy
import json
import math
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_cli_captions import fake_layers as fake_layers
from test_planner_cards import plan, requests, synthetic_faces
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_stage_graphics import graphic, title

from yap_studio import cards, cli, ffmpeg, receipt
from yap_studio.checks.caption_layout import LayoutRecord, check_caption_cards
from yap_studio.checks.face_clearance import check_face_clearance
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import (
    DEFAULT_HAIR,
    CaptionPhrase,
    Element,
    Layout,
    check_layout,
    face_boxes_per_frame,
    face_clearance_px,
    head_box,
)
from yap_studio.faces.detect import write_faces
from yap_studio.faces.mapping import ZoomPiece, behind_face_windows, map_picture_faces
from yap_studio.faces.model import Box, VideoIdentity
from yap_studio.planner import command as planner_command
from yap_studio.planner.cards import fitted_card, place_card, plan_cards
from yap_studio.project.model import load_cut_words
from yap_studio.project.status import ProjectContext, graphics_inputs
from yap_studio.stages import graphics as stage
from yap_studio.stages.assemble import assembly_snapshot
from yap_studio.stages.captions import captions, title_layer_plan
from yap_studio.stages.music import music
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import resolve
from yap_studio.zoom_curve import ZoomCurve


def assembly_media(root: Path) -> None:
    """Complete the caption fixture's cut contract without rendering a final."""
    (root / "cut/spans.json").write_text(json.dumps({
        "sources": {"t1": "../source.mov"}, "fps": "30/1",
        "ranges": [{"source": "t1", "start": "0", "end": "3"}],
    }))
    with wave.open(str(root / "cut/voice.wav"), "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(48_000)
        sound.writeframes(b"\0" * 3 * 48_000 * 2)


@pytest.mark.parametrize("absolute", [False, True])
def test_project_cards_cli_resolves_clip_paths_from_another_directory(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    absolute: bool,
) -> None:
    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {})["card_kinds_on"] = ["video_card"]
    (layer_project / "style.json").write_text(json.dumps(data))
    clip = cards.preview_clip(layer_project)
    source = Path(clip["path"])
    if not absolute:
        clip["path"] = source.name
    request_file = layer_project / "requests.json"
    request_file.write_text(json.dumps(requests(
        id="agent-video", template="video_card", slots={"clip": clip},
        start={"time": 0.2}, end={"time": 1.7},
    )))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    faces = synthetic_faces(None, frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(planner_command, "picture_faces", lambda *args: faces)
    rendered = []

    def render(layer, work, out, *, install=None):
        assert layer.windows[0]["source"]["path"] == str(source)
        rendered.append(layer)
        out.write_bytes(b"synthetic card layer")
        return stage.hyperframes.LayerResult(out, layer.frames, layer.width, layer.height,
                                            layer.frames / layer.fps)

    monkeypatch.setattr("yap_studio.planner.cards.render_card", render)
    elsewhere = layer_project / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    out = layer_project / "plan"
    assert cli.main([
        "cards", "plan", str(request_file), "--cut-words",
        str(layer_project / "cut/cut-words.json"), "--frames", str(timeline.total_frames),
        "--out", str(out), "--render", "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    lane = json.loads((out / "agent-lane.json").read_bytes())
    assert result["cards"][0]["slots"]["clip"] == clip
    assert lane["elements"][0]["slots"]["clip"] == clip
    assert result["cards"][0]["layer"]["status"] == "current" and len(rendered) == 1


@pytest.mark.parametrize(("measured", "caption_top"), [
    (False, None), (True, None), (True, 1000), (True, 0),
])
def test_portrait_video_card_plan_matches_graphics_width_search(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    measured: bool, caption_top: int | None,
) -> None:
    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {})["card_kinds_on"] = ["video_card"]
    (layer_project / "style.json").write_text(json.dumps(data))
    source = layer_project / "portrait.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=180x320:r=30",
                "-t", "4", str(source)], timeout_s=30)
    slots = {"clip": {"path": str(source), "sha256": receipt.sha256_file(source)},
             "label": "Portrait"}
    set_elements(layer_project, graphic(template="video_card", slots=slots, hold_s=1.5))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    style = context.render_style()
    faces = synthetic_faces(Box(458, 1400, 396, 396), measured=measured,
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    monkeypatch.setattr(planner_command, "picture_faces", lambda *args: faces)
    caption_layout = None
    if caption_top is not None:
        caption_layout = {"measured": True, "captions": [{"start_frame": 6, "end_frame": 51,
            "boxes": [{"x": 100, "y": caption_top, "w": 800, "h": 60}]}]}
        (layer_project / "captions").mkdir()
        (layer_project / "captions/layout.json").write_text(json.dumps(caption_layout))
    full = cards.card_layer_plan("video_card", slots, style, frames=45)
    _, _, clearance = place_card(
        replace(full, y=0), slots, style,
        face_boxes_per_frame(faces)[6:51] if measured else None,
    )
    assert clearance["status"] == "blocked"
    request = requests(id="agent-video", template="video_card", slots=slots,
                       start={"time": 0.2}, end={"time": 1.7})
    planned = plan_cards(
        request,
        load_cut_words(layer_project / "cut/cut-words.json"), style,
        frames=timeline.total_frames, faces=faces, timeline=timeline,
        caption_layout=caption_layout,
    )
    card, = planned.cards
    request_file = layer_project / "requests.json"
    request_file.write_text(json.dumps(request))
    assert cli.main([
        "cards", "plan", str(request_file), "--cut-words",
        str(layer_project / "cut/cut-words.json"), "--frames", str(timeline.total_frames),
        "--out", str(layer_project / "plan"), "--json",
    ]) == int(caption_top == 0)
    assert json.loads(capsys.readouterr().out)["cards"] == planned.to_dict()["cards"]
    if caption_top == 0:
        assert planned.status == "needs_review" and card.clearance["status"] == "blocked"
        with pytest.raises(ValidationError, match="face sits lower.*box.*behind"):
            stage.graphics_snapshot(ProjectContext(layer_project))
        return
    assert planned.status == "ok" and card.width < full.width
    if caption_top is not None:
        assert card.y + card.height + style.data["graphics"]["caption_gap_px"] <= caption_top
    snapshot, = stage.graphics_snapshot(ProjectContext(layer_project)).graphics
    box = {"x": card.x, "y": card.y, "width": card.width, "height": card.height}
    assert {key: snapshot.placement()[key] for key in box} == box
    fitted = fitted_card("video_card", slots, style, frames=45, width=card.width)
    assert snapshot.plan.windows == tuple(
        {**window, "source": {**window["source"], "path": source.name}}
        for window in fitted.windows
    )
    set_elements(layer_project, *planned.lane()["elements"])
    preserved, = stage.graphics_snapshot(ProjectContext(layer_project)).graphics
    assert preserved.plan == snapshot.plan


@pytest.mark.parametrize("face,position", [
    (Box(458, 629, 396, 396), "top"),
    (Box(400, 100, 280, 1350), "side"),
    (Box(300, 100, 480, 450), "lower"),
])
def test_accepted_card_plan_passes_assembly_and_exports_its_box(face: Box, position: str) -> None:
    faces = synthetic_faces(face)
    result = plan(requests(template="text_tag", slots={"text": "Hi"}), faces=faces)
    card = result.cards[0]
    assert result.status == "ok" and card.clearance["position"] == position
    assert card.layer() == "front"
    exported = result.lane()["elements"][0]["box"]
    assert exported == {"x": card.x, "y": card.y,
                        "width": card.width, "height": card.height}
    layout = Layout.from_dict(result.layout().to_dict())
    assert check_layout(faces, layout).status == "pass"
    findings, coverage = check_face_clearance(LayoutRecord(layout, True, None, {}),
                                             faces, faces_source="record")
    assert findings == [] and coverage["ran"]
    refitted = fitted_card(card.template, card.slots, resolve("preset:clean_minimal"),
                           frames=card.end_frame - card.start_frame, width=card.width)
    assert refitted.width == card.width
    if position == "side":
        assert refitted.variables["panel_width_px"] == card.width - 80


def test_behind_fallback_keeps_the_entire_animated_text_readable() -> None:
    style = resolve("preset:clean_minimal")
    style.data["captions"]["top_px"] = 850
    faces = synthetic_faces(Box(0, 559.9, 1080, 300))
    result = plan(faces=faces, style=style)
    card = result.cards[0]
    assert result.status == "ok" and card.placement == "behind_head"
    assert card.clearance["gap_px"] >= 0
    assert card.clearance["glyph_box"] == Box(card.x, card.y, card.width, card.height).to_dict()
    assert check_layout(faces, Layout.from_dict(result.layout().to_dict())).status == "pass"
    # A zoom loses the exemption, so a near-head box cannot use this fallback.
    layer = fitted_card(card.template, card.slots, style, frames=75)
    _, placement, clearance = place_card(layer, card.slots, style,
                                         face_boxes_per_frame(faces)[372:447],
                                         face_windows=((0, 75),))
    assert placement == "top" and clearance["status"] == "blocked"


def test_planned_behind_box_survives_lane_snapshot_and_graphics_receipt(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_bytes())
    data["captions"]["top_px"] = 850
    data.setdefault("graphics", {})["face_clearance_px"] = 60
    style_path.write_text(json.dumps(data))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    faces = synthetic_faces(Box(0, 560, 1080, 300),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    planned = plan_cards(
        requests(id="agent-stat", start={"time": 0.2}, end={"time": 1.2},
                 placement="behind_head"),
        load_cut_words(layer_project / "cut/cut-words.json"), context.render_style(),
        frames=timeline.total_frames, faces=faces, timeline=timeline,
    )
    card = planned.cards[0]
    assert planned.status == "ok" and card.placement == "behind_head"
    assert card.y == 0 and card.height == 360
    assert card.clearance["gap_px"] == 20 < card.clearance["margin_px"] == 60
    lane = planned.lane()
    exported = lane["elements"][0]
    assert exported["layer"] == "behind"
    box = {"x": card.x, "y": card.y, "width": card.width, "height": card.height}
    assert exported["box"] == box
    set_elements(layer_project, *lane["elements"])
    resolved = ProjectContext(layer_project).resolved_timeline().elements[0]
    assert resolved.layer == "behind" and resolved.payload["box"] == box
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)

    snapshot = stage.graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert snapshot.layer == "behind"
    assert snapshot.clearance["status"] == "behind_head"
    assert snapshot.clearance["glyph_box"] == card.clearance["glyph_box"]
    assert snapshot.clearance["gap_px"] == 20
    result = stage.graphics(layer_project)
    record = json.loads((layer_project / "graphics/receipt.json").read_bytes())
    assert record["elements"] == result["elements"] == [snapshot.placement()]
    assert record["elements"][0]["layer"] == "behind"
    assert {key: record["elements"][0][key] for key in box} == box
    assert (layer_project / record["elements"][0]["path"]).is_file()
    assert len(fake_layers) == 1


@pytest.mark.parametrize("unsafe", ["front", "head", "caption"])
def test_preserved_behind_box_does_not_bypass_clearance_constraints(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str,
) -> None:
    box = {"x": 20, "y": 0, "width": 1040, "height": 360}
    if unsafe == "head":
        box["y"] = 22
    set_elements(layer_project, graphic(layer="front" if unsafe == "front" else "behind",
                                       box=box))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(Box(0, 560, 1080, 300),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    if unsafe == "caption":
        (layer_project / "captions").mkdir()
        (layer_project / "captions/layout.json").write_text(json.dumps({
            "measured": True, "captions": [{"start_frame": 6, "end_frame": 36,
                "boxes": [{"x": 100, "y": 450, "w": 800, "h": 60}]}],
        }))
    with pytest.raises(ValidationError, match="supplied box"):
        stage.graphics_snapshot(ProjectContext(layer_project))
    assert not (layer_project / "graphics").exists()


@pytest.mark.parametrize("gap", [0, 20, 23])
def test_preserved_behind_box_uses_the_same_boundary_as_planning(gap: int) -> None:
    style = resolve("preset:clean_minimal")
    style.data["captions"]["top_px"] = 850
    faces = synthetic_faces(Box(0, 540 + gap, 1080, 300))
    planned = plan(faces=faces, style=style).cards[0]
    assert planned.placement == "behind_head" and planned.clearance["gap_px"] == gap
    layer = fitted_card(planned.template, planned.slots, style, frames=75)
    layer = replace(layer, x=planned.x, y=planned.y,
                    width=planned.width, height=planned.height)
    placed, placement, evidence = place_card(
        layer, planned.slots, style, face_boxes_per_frame(faces)[372:447],
        preserve_box=True, requested_layer="behind",
    )
    assert placed == layer and placement == "behind_head"
    assert evidence["gap_px"] == gap and evidence["status"] == "behind_head"
    assert evidence["glyph_box"] == planned.clearance["glyph_box"]


@pytest.mark.parametrize("kind", ["zoom", "insert", "transition"])
def test_preserved_behind_box_respects_picture_face_windows(kind: str) -> None:
    style = resolve("preset:clean_minimal")
    slots = {"text": "Hi"}
    layer = replace(fitted_card("text_tag", slots, style, frames=30), y=0)
    faces = [[Box(0, 560, 1080, 300)]]
    timeline = ResolvedTimeline((
        ResolvedElement("picture", kind, "agent", 6, 7, "front", {}),
    ), (), (), 90, 0)
    windows = behind_face_windows(timeline, 0, 30)
    assert windows == ((6, 7),)
    placed, placement, evidence = place_card(
        layer, slots, style, faces, preserve_box=True, requested_layer="behind",
        face_windows=windows,
    )
    assert placed == layer and placement == "top" and evidence["status"] == "blocked"


@pytest.mark.parametrize("unsafe", ["other_head", "side", "lower", "empty", "unmeasured"])
def test_preserved_behind_box_requires_the_entire_box_above_every_measured_head(
    unsafe: str,
) -> None:
    style = resolve("preset:clean_minimal")
    slots = {"text": "Hi"}
    layer = fitted_card("text_tag", slots, style, frames=30)
    layer = replace(layer, y=0)
    faces = [[Box(0, 1200, 1080, 300)]]
    if unsafe == "other_head":
        # The distant head never overlaps horizontally, but is higher than the glyphs.
        faces.append([Box(2000, 530, 100, 300)])
    elif unsafe == "side":
        faces = [[Box(2000, 0, 100, 300)]]
    elif unsafe == "lower":
        faces = [[Box(0, 0, 1080, 300)]]
        layer = replace(layer, y=640)
    elif unsafe == "empty":
        faces = []
    else:
        faces = None
    placed, placement, evidence = place_card(
        layer, slots, style, faces, preserve_box=True, requested_layer="behind",
    )
    assert placed == layer and placement == "top" and evidence["status"] == "blocked"


def test_behind_request_does_not_hide_a_label_when_a_clear_lower_position_exists() -> None:
    faces = synthetic_faces(Box(458, 450, 396, 396))
    result = plan(requests(placement="behind_head"), faces=faces)
    assert result.cards[0].placement == "top"
    assert result.cards[0].clearance["position"] in {"side", "lower"}
    assert check_layout(faces, result.layout()).status == "pass"


def test_no_readable_position_is_not_accepted() -> None:
    result = plan(faces=synthetic_faces(Box(0, 0, 1080, 1920)))
    assert result.status == "needs_review"
    assert result.cards[0].clearance["status"] == "blocked"


@pytest.mark.parametrize("face_y,shrinks", [(700, False), (520, True)])
def test_title_moves_then_shrinks_without_mutating_style(face_y: int, shrinks: bool) -> None:
    style = resolve("preset:clean_minimal")
    before = copy.deepcopy(style.data)
    default = title_layer_plan("", "", style, lines=["The API is the waiter"], frames=60)
    faces = synthetic_faces(Box(458, face_y, 396, 396))
    adjusted, evidence = stage.place_title(default, style, faces)
    assert adjusted.y < default.y
    assert (adjusted.variables["size_px"] < default.variables["size_px"]) == shrinks
    assert "default title collides" in evidence["reason"]
    assert ("shrank" in evidence["reason"]) == shrinks
    assert style.data == before
    layout = Layout(1080, 1920, (Element("title", "title", Box(
        adjusted.x, adjusted.y, adjusted.width, adjusted.height), 0, 60),), ())
    assert check_layout(faces, layout).status == "pass"
    assert check_face_clearance(LayoutRecord(layout, True, None, {}),
                                faces, faces_source="record")[0] == []


def test_clear_title_keeps_its_default_and_uses_only_its_own_window() -> None:
    style = resolve("preset:clean_minimal")
    default = title_layer_plan("", "", style, lines=["Small steps"], frames=30, start_frame=30)
    faces = synthetic_faces(Box(0, 0, 1080, 1920), frames=90)
    clear = synthetic_faces(Box(458, 1100, 396, 396), frames=90)
    faces = replace(faces, frames=faces.frames[:30] + clear.frames[30:60] + faces.frames[60:])
    placed, evidence = stage.place_title(default, style, faces)
    assert placed == default and evidence["status"] == "clear"


@pytest.mark.parametrize("band_ratio", [1.5, .85])
def test_kinetic_title_centres_and_scales_in_clear_headroom(band_ratio: float) -> None:
    style = resolve("preset:clean_minimal")
    style.data["hook"].update(style="kinetic", font="Instrument Serif")
    times = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    default = title_layer_plan("", "", style, lines=["How many R's", "are in strawberry"],
                               frames=60, word_times=times)
    band = math.floor(default.height * band_ratio)
    face = Box(458, band + DEFAULT_HAIR.above * 396 + face_clearance_px(style.data), 396, 396)
    faces = synthetic_faces(face)
    adjusted, evidence = stage.place_title(default, style, faces)
    assert abs(adjusted.y + adjusted.height / 2 - band / 2) <= .5
    assert adjusted.y >= 0 and adjusted.y + adjusted.height <= band
    assert evidence["original_scale"] == 1
    assert (evidence["scale"] < 1) == (band_ratio < 1)
    original = json.loads(default.variables["kinetic_lines"])
    resized = json.loads(adjusted.variables["kinetic_lines"])
    assert all(after["size_px"] == pytest.approx(before["size_px"] * evidence["scale"], abs=.01)
               for before, after in zip(original, resized, strict=True))
    assert all(line["size_px"] >= 110 for line in resized)
    assert json.loads(adjusted.variables["word_times"]) == times
    assert evidence["status"] == "clear"
    style.data["hook"]["size_px"] = 40
    other = title_layer_plan("", "", style, lines=["How many R's", "are in strawberry"],
                             frames=60, word_times=times)
    assert stage.place_title(other, style, faces)[0] == adjusted


def test_kinetic_title_stays_large_in_470_px_clear_band() -> None:
    style = resolve("preset:clean_minimal")
    style.data["hook"].update(style="kinetic", font="Instrument Serif")
    plan = title_layer_plan("", "", style, lines=["How many R's", "are in strawberry"],
                            frames=60)
    band = 470
    face = Box(458, band + DEFAULT_HAIR.above * 396 + face_clearance_px(style.data), 396, 396)
    placed, evidence = stage.place_title(plan, style, synthetic_faces(face))
    sizes = [line["size_px"] for line in json.loads(placed.variables["kinetic_lines"])]
    assert max(sizes) >= 150
    assert min(sizes) >= 110
    assert 0 <= placed.y and placed.y + placed.height <= band
    assert abs(placed.y + placed.height / 2 - band / 2) <= .5
    assert evidence["status"] == "clear"


def test_kinetic_title_records_overlap_at_floor_without_refusing() -> None:
    style = resolve("preset:clean_minimal")
    style.data["hook"]["style"] = "kinetic"
    default = title_layer_plan("", "", style, lines=["Small clear", "steps forward"], frames=60)
    face = Box(458, 350, 396, 396)
    faces = synthetic_faces(face)
    adjusted, evidence = stage.place_title(default, style, faces)
    assert adjusted.y == 0
    assert all(line["size_px"] == 110
               for line in json.loads(adjusted.variables["kinetic_lines"]))
    assert evidence["original_scale"] == 1 and evidence["scale"] < 1
    assert evidence["status"] == "overlap" and "overlaps the head" in evidence["reason"]
    assert adjusted.y + adjusted.height > head_box(face).y - face_clearance_px(style.data)
    layout = Layout(1080, 1920, (Element("title", "title", Box(
        adjusted.x, adjusted.y, adjusted.width, adjusted.height), 0, 60),), ())
    assert check_layout(faces, layout).status == "fail"


def test_kinetic_title_only_uses_faces_in_its_own_window() -> None:
    style = resolve("preset:clean_minimal")
    style.data["hook"]["style"] = "kinetic"
    plan = title_layer_plan("Small", "steps", style, frames=30, start_frame=30)
    outside = synthetic_faces(Box(458, 350, 396, 396), frames=90)
    inside = synthetic_faces(Box(458, 1100, 396, 396), frames=90)
    faces = replace(outside, frames=(outside.frames[:30] + inside.frames[30:60]
                                     + outside.frames[60:]))
    assert stage.place_title(plan, style, faces) == stage.place_title(plan, style, inside)


def test_title_refuses_when_even_the_readable_minimum_cannot_clear() -> None:
    style = resolve("preset:clean_minimal")
    default = title_layer_plan("", "", style, lines=["Small steps"], frames=30)
    with pytest.raises(ValidationError, match="no readable title size"):
        stage.place_title(default, style, synthetic_faces(Box(0, 0, 1080, 1920)))


def test_title_clears_face_through_opening_push() -> None:
    from yap_studio.zoom_curve import push_keys

    style = resolve("preset:clean_minimal")
    default = title_layer_plan("", "", style, lines=["Small steps"], frames=45)
    cut = synthetic_faces(Box(458, 1000, 300, 300), frames=90)
    placed, evidence = stage.place_title(default, style, cut)
    assert placed == default and evidence["status"] == "clear"
    curve = ZoomCurve(0, 45, push_keys(0, 45, 1.3, (.5, .9), "linear"))
    zoom = ResolvedElement("agent-opening", "zoom", "agent", 0, 45, "front",
                           {"move": "opening"})
    timeline = ResolvedTimeline((zoom,), (), (), 90, 0)
    faces, _ = map_picture_faces(cut, timeline, zooms=[ZoomPiece(0, 45, curve)],
                                 video=VideoIdentity("final.mp4", "c" * 64, 90,
                                                     Fraction(30), 1080, 1920))
    placed, evidence = stage.place_title(default, style, faces)
    assert placed.y < default.y or placed.variables["size_px"] < default.variables["size_px"]
    assert "default title collides" in evidence["reason"]


def test_empty_graphics_snapshot_does_not_measure_cut(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage, "picture_faces",
                        lambda *args: pytest.fail("empty graphics must not measure faces"))
    snapshot = stage.graphics_snapshot(ProjectContext(layer_project))
    assert snapshot.titles == () and snapshot.graphics == () and snapshot.inputs == {}


def test_empty_project_card_plan_does_not_measure_cut(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(planner_command, "picture_faces",
                        lambda *args: pytest.fail("empty card requests must not measure faces"))
    request_file = layer_project / "requests.json"
    request_file.write_text(json.dumps({"schema": "yap-studio/card-requests@1", "cards": []}))
    assert cli.main([
        "cards", "plan", str(request_file), "--project", str(layer_project),
        "--cut-words", str(layer_project / "cut/cut-words.json"), "--frames", "90",
        "--out", str(layer_project / "plan"), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["cards"] == []


def test_graphics_receipt_reproduces_cleared_placement_without_detection(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_elements(layer_project, graphic(), title(start={"time": 1.5}, hold_s=0.5))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    original = stage.graphics_snapshot(ProjectContext(layer_project))
    record = stage.graphics(layer_project)
    monkeypatch.setattr(stage, "picture_faces",
                        lambda *args: pytest.fail("receipt inspection must not measure faces"))
    restored = stage.graphics_snapshot(ProjectContext(layer_project), placements=record["elements"])
    assert restored == original
    assert graphics_inputs(ProjectContext(layer_project)) == original.inputs


def test_graphics_inputs_track_tall_framing_windows(layer_project, fake_layers) -> None:
    set_elements(layer_project, title())
    stage.graphics(layer_project)
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    before = graphics_inputs(context)
    tall = ResolvedElement("broll-page", "overlay", "broll", 45, 90, "front",
                           {"placement": "tall_behind", "tall_framing": "shrink"})
    identities = []
    for start, end, framing in [(45, 90, "shrink"), (46, 90, "shrink"),
                                (45, 89, "shrink"), (45, 90, "keep")]:
        item = replace(tall, start_frame=start, end_frame=end,
                       payload={**tall.payload, "tall_framing": framing})
        context._timeline = replace(timeline, elements=(*timeline.elements, item))
        after = graphics_inputs(context)
        assert after["title:agent-title"] == before["title:agent-title"]
        assert after["faces"] != before["faces"]
        identities.append(after["faces"])
    assert len(set(identities)) == 4


def test_project_graphics_replans_without_writing_style(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_elements(layer_project, graphic(), title(start={"time": 1.5}, hold_s=0.5))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    faces = synthetic_faces(frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    before = (layer_project / "style.json").read_bytes()
    snapshot = stage.graphics_snapshot(context)
    assert snapshot.graphics[0].plan.y == 0
    assert snapshot.titles[0].plan.y < 260
    assert snapshot.titles[0].placement()["clearance"]["status"] == "clear"
    assert (layer_project / "style.json").read_bytes() == before
    # The exact box exported by the standalone planner survives the graphics stage.
    set_elements(layer_project, graphic(box={"x": 20, "y": 0, "width": 1040, "height": 360}))
    assert stage.graphics_snapshot(ProjectContext(layer_project)).graphics[0].plan.y == 0


@pytest.mark.parametrize("y", [280, 800])
def test_supplied_clear_card_box_survives_graphics_receipt_and_assembly(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
    y: int,
) -> None:
    assembly_media(layer_project)
    box = {"x": 20, "y": y, "width": 1040, "height": 360}
    set_elements(layer_project, graphic(box=box))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(Box(458, 1200 if y == 280 else 1600, 396, 396),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    captions(layer_project)
    result = stage.graphics(layer_project)
    assert {key: result["elements"][0][key] for key in box} == box
    record = json.loads((layer_project / "graphics/receipt.json").read_text())
    assert record["elements"] == result["elements"]
    music(layer_project)
    assembled = assembly_snapshot(ProjectContext(layer_project))
    assert {key: assembled.title_layers[0][key] for key in box} == box


@pytest.mark.parametrize("unsafe", ["head", "caption"])
def test_supplied_unsafe_card_box_is_refused(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str,
) -> None:
    set_elements(layer_project, graphic(box={"x": 20, "y": 280,
                                           "width": 1040, "height": 360}))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(Box(458, 629 if unsafe == "head" else 1200, 396, 396),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    if unsafe == "caption":
        (layer_project / "captions").mkdir()
        (layer_project / "captions/layout.json").write_text(json.dumps({
            "measured": True, "captions": [{"start_frame": 6, "end_frame": 36,
                "boxes": [{"x": 100, "y": 690, "w": 800, "h": 60}]}],
        }))
    with pytest.raises(ValidationError, match="supplied box"):
        stage.graphics_snapshot(ProjectContext(layer_project))
    assert not (layer_project / "graphics").exists()


@pytest.mark.parametrize("faces", [None, [], [[Box(458, 1200, 396, 396)]]])
@pytest.mark.parametrize("caption_top", [0, 300, 450, 460, 690])
def test_top_card_respects_caption_ceiling_with_or_without_faces(
    faces: list | None, caption_top: int,
) -> None:
    style = resolve("preset:clean_minimal")
    slots = {"text": "Hi"}
    layer = fitted_card("text_tag", slots, style, frames=30)
    placed, _, evidence = place_card(layer, slots, style, faces, caption_top=caption_top)
    if caption_top < (304 if faces else 460):
        assert evidence["status"] == "blocked"
    else:
        assert evidence["status"] == ("unchecked" if faces is None else "clear")
        assert placed.y + placed.height <= caption_top - 100
        layout = Layout(1080, 1920, (Element("card", "card", Box(
            placed.x, placed.y, placed.width, placed.height), 0, 30),), (
                CaptionPhrase("caption", "words", (Box(100, caption_top, 800, 60),), 0, 30),
            ))
        assert check_caption_cards(LayoutRecord(layout, True, None, {}), fps=30) == []


def test_giant_fallback_caption_ceiling_reserves_the_padded_block() -> None:
    from yap_studio import captions as caption_module
    from yap_studio.stages.captions import caption_band_geometry
    from yap_studio.style import ResolvedStyle, resolve_style_data

    data = resolve_style_data({"schema_version": 1, "captions": {
        "highlight_style": "giant", "size_px": 83, "top_px": 1700,
    }})
    style = ResolvedStyle(data["name"], "test", data)
    before = copy.deepcopy(data)
    settings = caption_module.caption_layout_settings(data["captions"], data["brand"])
    ceiling = (caption_band_geometry(settings, safe_bottom_px=1740).y
               - data["graphics"]["caption_gap_px"])
    slots = {"text": "Clear words"}
    layer = fitted_card("text_tag", slots, style, frames=30)
    for delta, expected in [(0, "unchecked"), (1, "blocked")]:
        candidate = replace(layer, y=ceiling - layer.height + delta)
        _, _, evidence = place_card(candidate, slots, style, None, preserve_box=True)
        assert evidence["status"] == expected
    assert data == before


def test_top_card_repositions_below_head_without_entering_caption_gap() -> None:
    style = resolve("preset:clean_minimal")
    slots = {"text": "Hi"}
    layer = fitted_card("text_tag", slots, style, frames=30)
    faces = [[Box(458, 101, 100, 100)]]
    placed, _, evidence = place_card(layer, slots, style, faces, caption_top=650)
    assert evidence["status"] == "clear"
    assert placed.y + placed.height <= 550
    assert evidence["position"] == "side"


@pytest.mark.parametrize("overlaps", [False, True])
def test_graphics_uses_moved_measured_captions_only_in_the_card_window(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, overlaps: bool,
) -> None:
    set_elements(layer_project, graphic())
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(Box(458, 1200, 396, 396),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    (layer_project / "captions").mkdir()
    (layer_project / "captions/layout.json").write_text(json.dumps({
        "measured": True, "captions": [{"start_frame": 35 if overlaps else 36,
            "end_frame": 60, "boxes": [{"x": 100, "y": 420, "w": 800, "h": 60}]}],
    }))
    card, = stage.graphics_snapshot(ProjectContext(layer_project)).graphics
    if overlaps:
        assert card.plan.y + card.plan.height <= 320
    else:
        assert card.plan.height == 360


def test_graphics_refuses_card_that_cannot_clear_configured_middle_captions(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_text())
    data["captions"].update(position="middle", top_px=700, size_px=180)
    style_path.write_text(json.dumps(data))
    set_elements(layer_project, graphic(template="stat", slots={"end": 37}))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(Box(458, 1200, 396, 396),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    with pytest.raises(ValidationError, match="caption limits"):
        stage.graphics_snapshot(ProjectContext(layer_project))


@pytest.mark.parametrize("measured", [False, True])
def test_card_without_head_measurements_stays_front_through_assembly(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
    measured: bool,
) -> None:
    assembly_media(layer_project)
    set_elements(layer_project, graphic(layer="behind"))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(None, measured=measured,
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    captions(layer_project)
    result = stage.graphics(layer_project)
    assert result["elements"][0]["layer"] == "front"
    assert result["elements"][0]["clearance"]["gap_px"] is None
    music(layer_project)
    snapshot = assembly_snapshot(ProjectContext(layer_project))
    assert snapshot.timeline.elements[0].layer == "front"
    assert "picture:behind" not in snapshot.inputs


def test_project_cards_cli_uses_assembly_faces_and_project_style(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    mapped = synthetic_faces(frames=timeline.total_frames + timeline.end_hold_frames)
    calls = []

    def common_faces(root, resolved, context):
        assert context.project_dir == root
        assert context.resolved_timeline() is resolved
        calls.append((root, resolved))
        return mapped

    monkeypatch.setattr(planner_command, "picture_faces", common_faces)
    independent = synthetic_faces(None, frames=timeline.total_frames)
    independent = replace(independent, video=replace(
        independent.video, sha256=receipt.sha256_file(layer_project / "cut/cut.mp4"),
    ))
    face_file = write_faces(layer_project / "independent-faces.json", independent)
    request_file = layer_project / "requests.json"
    request_file.write_text(json.dumps({"schema": "yap-studio/card-requests@1", "cards": [{
        "id": "agent-card", "template": "text_tag", "slots": {"text": "Hi"},
        "start": {"time": 0.2}, "hold_s": 1,
    }]}))
    args = ["cards", "plan", str(request_file), "--cut-words",
            str(layer_project / "cut/cut-words.json"), "--frames", str(timeline.total_frames),
            "--faces", str(face_file), "--out", str(layer_project / "plan"), "--json"]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["style"] == context.render_style().sha256()
    assert result["cards"][0]["clearance"]["gap_px"] is not None
    assert len(calls) == 1 and calls[0][0] == layer_project
    assert cli.main([*args, "--style", "preset:electric"]) == 2
    assert "project's resolved render style" in capsys.readouterr().err


def test_automatic_behind_fallback_reaches_the_assembly_contract(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly_media(layer_project)
    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("captions", {})["top_px"] = 850
    (layer_project / "style.json").write_text(json.dumps(data))
    set_elements(layer_project, graphic(template="stat", slots={"end": 37}))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(Box(0, 559.9, 1080, 300),
                            frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    captions(layer_project)
    result = stage.graphics(layer_project)
    assert result["elements"][0]["layer"] == "behind"
    assert result["elements"][0]["clearance"]["gap_px"] >= 0
    music(layer_project)
    snapshot = assembly_snapshot(ProjectContext(layer_project))
    assert snapshot.title_layers[0]["layer"] == "behind"
    assert next(item for item in snapshot.timeline.elements
                if item.id == "agent-tag").layer == "behind"
    assert "picture:behind" in snapshot.inputs


def test_explicit_behind_card_with_detected_head_uses_a_clear_front_position(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly_media(layer_project)
    set_elements(layer_project, graphic(layer="behind"))
    timeline = ProjectContext(layer_project).resolved_timeline()
    faces = synthetic_faces(frames=timeline.total_frames + timeline.end_hold_frames)
    monkeypatch.setattr(stage, "picture_faces", lambda *args: faces)
    captions(layer_project)
    assert stage.graphics(layer_project)["elements"][0]["layer"] == "front"
    music(layer_project)
    snapshot = assembly_snapshot(ProjectContext(layer_project))
    assert snapshot.timeline.elements[0].layer == "front"
    assert "picture:behind" not in snapshot.inputs


@pytest.mark.parametrize("windows", [[], [[10, 20]]])
@pytest.mark.parametrize("explicit_faces", [False, True])
def test_cli_and_assembly_agree_on_behind_exemptions(
    tmp_path: Path, capsys: pytest.CaptureFixture, windows: list, explicit_faces: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "final.mp4"
    video.write_bytes(b"hash bound synthetic final")
    faces = synthetic_faces(frames=30)
    faces = replace(faces, video=replace(faces.video, sha256=receipt.sha256_file(video)))
    face_file = write_faces(tmp_path / "final.faces.json", faces)
    calls = []

    def measure(video: Path, *, every: int):
        calls.append((video, every))
        return faces

    monkeypatch.setattr("yap_studio.faces.detect.detect_faces", measure)
    document = Layout(1080, 1920, (Element("behind", "card", Box(20, 280, 1040, 360),
                                          0, 30),), ()).to_dict()
    document["elements"][0]["face_windows"] = windows
    layout_file = tmp_path / "final.layout.json"
    layout_file.write_text(json.dumps(document))
    layout = Layout.from_dict(document)
    assert layout.to_dict()["elements"][0]["face_windows"] == windows
    findings, _ = check_face_clearance(LayoutRecord(layout, True, None, {}),
                                       faces, faces_source="record")
    code = cli.main(["faces", "check", str(video), "--layout", str(layout_file), "--json",
                     *(["--faces", str(face_file)] if explicit_faces else [])])
    report = json.loads(capsys.readouterr().out)
    assert (code == 0) == (findings == [])
    assert calls == ([] if explicit_faces else [(video, 1)])
    assert report["hair"] == {"above": 0.6, "sides": 0.15}
    assert [(v["start_frame"], v["end_frame"]) for v in report["violations"]] == [
        (v.evidence["start_frame"], v.evidence["end_frame"]) for v in findings
    ]


@pytest.mark.parametrize("measured", [False, True])
def test_unmeasured_caption_geometry_has_the_same_policy_in_both_checks(measured: bool) -> None:
    faces = synthetic_faces(frames=30)
    phrase = CaptionPhrase("caption", "words", (Box(450, 500, 400, 100),), 0, 30)
    layout = Layout.from_dict({**Layout(1080, 1920, (), (phrase,)).to_dict(),
                               "measured": measured})
    report = check_layout(faces, layout)
    findings, _ = check_face_clearance(LayoutRecord(layout, measured, None, {}),
                                       faces, faces_source="record")
    assert (report.status == "fail") == measured
    assert len(report.violations) == len(findings) == int(measured)


def test_planning_uses_the_serialized_zoom_geometry_and_replacement_windows() -> None:
    cut = synthetic_faces(Box(458, 629, 396, 396), frames=90)
    zoom = ResolvedElement("zoom", "zoom", "agent", 0, 75, "front",
                           {"peak_scale": 1.1, "center": [0.5, 0.8]})
    insert = ResolvedElement("insert", "insert", "agent", 15, 30, "front", {})
    transition = ResolvedElement("transition", "transition", "agent", 45, 50, "front", {})
    timeline = ResolvedTimeline((zoom, insert, transition), (), (), 90, 6)
    piece = ZoomPiece(0, 75, ZoomCurve.swell(0, 75, 1.1, (0.5, 0.8)), 3840, 2160)
    faces, _ = map_picture_faces(cut, timeline, zooms=[piece], video=VideoIdentity(
        "final.mp4", "c" * 64, 96, Fraction(30), 1080, 1920))
    assert faces == type(faces).from_dict(faces.to_dict())
    assert not any(faces.frames[15:30]) and not any(faces.frames[45:50])
    assert faces.frames[90:] == (cut.frames[-1],) * 6
    style = resolve("preset:clean_minimal")
    layer = fitted_card("text_tag", {"text": "Hi"}, style, frames=75)
    placed, _, evidence = place_card(layer, {"text": "Hi"}, style,
                                     face_boxes_per_frame(faces)[:75],
                                     face_windows=((0, 75),))
    assert evidence["status"] == "clear"
    layout = Layout(1080, 1920, (Element("card", "card", Box(
        placed.x, placed.y, placed.width, placed.height), 0, 75),), ())
    assert check_face_clearance(LayoutRecord(layout, True, None, {}),
                                faces, faces_source="record")[0] == []


@pytest.mark.parametrize("windows", [None, True, [[True, 2]], [[0, 31]], [[10, 20], [5, 8]]])
def test_invalid_face_windows_cannot_silently_exempt_a_card(windows: object) -> None:
    document = Layout(1080, 1920, (Element("card", "card", Box(0, 0, 100, 100), 0, 30),),
                      ()).to_dict()
    document["elements"][0]["face_windows"] = windows
    with pytest.raises(ValidationError, match="face_windows"):
        Layout.from_dict(document)
