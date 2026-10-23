"""Persistent Text hooks share picture windows and reserve their painted box."""

import json
from dataclasses import replace

import numpy as np
import pytest
import synth
from test_chapters import chapter
from test_cli_captions import fake_layers as fake_layers
from test_moment_picture import decoded
from test_moment_picture import picture as picture
from test_planner_cards import synthetic_faces
from test_stage_captions import layer_project as layer_project
from test_tall_box import page as page
from test_text_hook import choice
from test_timeline import graphic, insert, overlay, transition, transition_story, write
from test_timeline import project as project

from yap_studio import cards, receipt
from yap_studio.errors import ValidationError
from yap_studio.faces.model import Box
from yap_studio.project.status import ProjectContext
from yap_studio.render.moments import check_moments
from yap_studio.stages import assemble_picture
from yap_studio.stages import graphics as graphics_stage
from yap_studio.stages.captions import caption_band_geometry, caption_offsets, captions
from yap_studio.stages.graphics import graphics_snapshot
from yap_studio.stages.timeline import ResolvedElement
from yap_studio.style import resolve_style_data


@pytest.fixture
def hooked_project(project, monkeypatch):
    data = json.loads((project / "project.json").read_bytes())
    write(project / "project.json", {**data, "style": "style.json",
                                     "rules": {"elements_at_once_max": 1}})
    write(project / "style.json", {"schema_version": 1, "hook": {"style": "text_hook"},
                                   "moments": {"styles": ["color_pop", "sticker_freeze",
                                                          "clone_flash"]}})
    write(project / "plans/text-hook.json", choice("Make it clear", "typed"))
    synth.picture_source(project / "cut/cut.mp4", "color=blue:s=160x90:r=30", 360)
    monkeypatch.setattr("yap_studio.stages.graphics.picture_faces", lambda *args:
                        synthetic_faces(Box(400, 1200, 200, 200), frames=400))
    return project


def lane(root, name, elements):
    write(root / f"timeline/{name}.json", {"schema": "yap-studio/timeline@1",
          "words_sha256": json.loads((root / "cut/cut-words.json").read_bytes())["words_sha256"],
          "elements": elements})


def test_text_hook_shares_insert_background_card_and_moment(hooked_project):
    root = hooked_project
    lane(root, "broll", [
        insert(root, start={"time": 0}, end={"time": 1.5}),
        insert(root, id="broll-room", kind="background", layout="bottom",
               start={"time": 2}, end={"time": 3.5}),
    ])
    lane(root, "agent", [
        graphic(start={"time": 4}, end={"time": 5.5}, layer="front",
                box={"x": 100, "y": 450, "width": 880, "height": 360}),
        {"id": "agent-moment", "kind": "moment", "lane": "agent", "style": "color_pop",
         "start": {"time": 6}, "end": {"time": 7.2}},
    ])
    context = ProjectContext(root)
    assert context.resolved_timeline().valid
    snapshot = graphics_snapshot(context)
    hook, = snapshot.titles
    assert hook.id == "agent-text-hook" and hook.layer == "front"
    assert hook.plan.y + hook.plan.height < snapshot.graphics[0].plan.y
    assert graphics_snapshot(ProjectContext(root), placements=[
        item.placement() for item in (*snapshot.titles, *snapshot.graphics)
    ]).inputs == snapshot.inputs


@pytest.mark.parametrize("text", ["Rant", "A better opening starts with a clear idea"])
@pytest.mark.parametrize("hook_y", [250, 650], ids=["clear", "paint-overlap"])
def test_text_hook_checks_chapter_text_not_its_canvas(hooked_project, monkeypatch, text, hook_y):
    root = hooked_project
    data = json.loads((root / "style.json").read_bytes())
    data["chapters"] = {"style": "title_card"}
    write(root / "style.json", data)
    write(root / "plans/text-hook.json", choice(text, "typed"))
    lane(root, "agent", [chapter(start={"time": 4}, hold_s=1)])
    place_title = graphics_stage.place_title

    def placed(plan, *args):
        plan, clearance = place_title(plan, *args)
        return replace(plan, y=hook_y), clearance

    monkeypatch.setattr(graphics_stage, "place_title", placed)
    if hook_y == 650:
        with pytest.raises(ValidationError, match="agent-ch-tokens.*Text hook.*Move its box"):
            graphics_snapshot(ProjectContext(root))
    else:
        snapshot = graphics_snapshot(ProjectContext(root))
        card = next(item for item in snapshot.titles if item.kind == "chapter")
        assert (card.plan.x, card.plan.y, card.plan.width, card.plan.height) == (0, 0, 1080, 1920)
        assert graphics_snapshot(ProjectContext(root), placements=[
            item.placement() for item in snapshot.titles
        ]).inputs == snapshot.inputs


@pytest.mark.parametrize("tile_y", [600, 220], ids=["clear", "paint-overlap"])
def test_text_hook_checks_grid_tiles_not_their_canvas(hooked_project, monkeypatch, tile_y):
    root = hooked_project
    data = json.loads((root / "style.json").read_bytes())
    data["graphics"] = {"card_kinds_on": ["playing_grid"]}
    write(root / "style.json", data)
    slots = {"clips": [insert(root)["media"]] * 2}
    lane(root, "agent", [graphic(template="playing_grid", slots=slots, layer="front",
                                  start={"time": 4}, end={"time": 6})])

    def grid_plan(slots, style, faces, *, frames, **kwargs):
        plan = cards.card_layer_plan("playing_grid", slots, style, frames=frames,
                                     tiles=[{"x": x, "y": tile_y, "width": 400, "height": 300}
                                            for x in (100, 580)])
        return plan, {"status": "clear"}

    monkeypatch.setattr(cards, "grid_card_plan", grid_plan)
    if tile_y == 220:
        with pytest.raises(ValidationError, match="agent-card.*Text hook.*Move its box"):
            graphics_snapshot(ProjectContext(root))
    else:
        snapshot = graphics_snapshot(ProjectContext(root))
        assert snapshot.graphics[0].plan.height == 1920
        assert graphics_snapshot(ProjectContext(root), placements=[
            item.placement() for item in (*snapshot.titles, *snapshot.graphics)
        ]).inputs == snapshot.inputs


def test_text_hook_refuses_a_front_card_in_its_box(hooked_project):
    root = hooked_project
    lane(root, "agent", [graphic(start={"time": 4}, end={"time": 5.5}, layer="front",
                                  box={"x": 100, "y": 0, "width": 880, "height": 640})])
    context = ProjectContext(root)
    assert context.resolved_timeline().valid
    with pytest.raises(ValidationError, match="Text hook.*[Mm]ove|[Mm]ove.*Text hook"):
        graphics_snapshot(context)


@pytest.mark.parametrize("name", ["color_pop", "sticker_freeze", "clone_flash"])
def test_text_hook_does_not_block_moments(hooked_project, name):
    lane(hooked_project, "agent", [{"id": "agent-moment", "kind": "moment", "lane": "agent",
         "style": name, "start": {"time": 6}, "end": {"time": 7.2}}])
    context = ProjectContext(hooked_project)
    timeline = context.resolved_timeline()
    assert timeline.valid
    findings = []
    check_moments(timeline.elements, context.render_style().data,
                  lambda rule, *args, **kwargs: findings.append(rule))
    assert not findings


def test_text_hook_does_not_block_tall_overlay(hooked_project, page):
    lane(hooked_project, "broll", [page])
    assert ProjectContext(hooked_project).resolved_timeline().valid


def test_text_hook_keeps_the_overlap_limit_for_other_elements(hooked_project):
    lane(hooked_project, "broll", [
        overlay(hooked_project), overlay(hooked_project, id="broll-other"),
    ])
    timeline = ProjectContext(hooked_project).resolved_timeline(allow_violations=True)
    assert not timeline.valid
    finding = next(item for item in timeline.findings if item.rule == "visual_overlap")
    assert set(finding.elements) == {"broll-one", "broll-other"}


def test_text_hook_reserves_its_box_from_overlays(hooked_project):
    lane(hooked_project, "broll", [overlay(hooked_project, placement="box",
                                           box=[100, 200, 880, 400])])
    with pytest.raises(ValidationError, match="Move its box clear of the Text hook"):
        graphics_snapshot(ProjectContext(hooked_project))


@pytest.mark.parametrize("look", ["text_hook", "quiet_serif"])
def test_only_the_generated_title_is_exempt(hooked_project, look):
    root = hooked_project
    data = json.loads((root / "style.json").read_bytes())
    data["hook"] = {"style": look}
    write(root / "style.json", data)
    lane(root, "agent", [graphic(id="agent-text-hook"), graphic()])
    timeline = ProjectContext(root).resolved_timeline(allow_violations=True)
    assert any(item.rule == "visual_overlap" for item in timeline.findings)


def test_ordinary_title_still_blocks_clone_moments():
    style = resolve_style_data({"schema_version": 1, "moments": {"styles": ["clone_flash"]}})
    items = [ResolvedElement("agent-text-hook", "title", "agent", 0, 36, "front", {}),
             ResolvedElement("agent-clone", "moment", "agent", 0, 36, "front",
                             {"style": "clone_flash"})]
    findings = []
    check_moments(items, style, lambda rule, *args, **kwargs: findings.append(rule))
    assert "moment_overlap" in findings


def test_text_hook_reserves_the_flying_card_path(hooked_project):
    root = hooked_project
    write(root / "plans/text-hook.json", choice("Rant", "typed"))
    lane(root, "agent", [graphic(start={"time": 4}, end={"time": 5.5}, layer="front",
                                  box={"x": 0, "y": 280, "width": 300, "height": 360})])
    snapshot = graphics_snapshot(ProjectContext(root))
    assert snapshot.graphics[0].plan.x + snapshot.graphics[0].plan.width < snapshot.titles[0].plan.x
    data = json.loads((root / "style.json").read_bytes())
    data["graphics"] = {"card_motion": "streak"}
    write(root / "style.json", data)
    with pytest.raises(ValidationError, match="Move its box clear of the Text hook"):
        graphics_snapshot(ProjectContext(root))


def test_text_hook_does_not_block_transition(hooked_project):
    transition_story(hooked_project)
    lane(hooked_project, "agent", [transition()])
    assert ProjectContext(hooked_project).resolved_timeline().valid


@pytest.mark.parametrize("placement", ["fixed", "avoid_face"])
def test_text_hook_keeps_high_captions_clear(layer_project, monkeypatch, fake_layers, placement):
    root = layer_project
    write(root / "plans/text-hook.json", choice(
        "A better opening starts with a clear idea", "typed"))
    data = json.loads((root / "style.json").read_bytes())
    data["hook"] = {"style": "text_hook"}
    data["captions"] = {"position": "middle", "top_px": 700, "size_px": 180,
                        "background_mode": "box", "placement": placement}
    write(root / "style.json", data)
    monkeypatch.setattr("yap_studio.stages.captions.cut_faces", lambda *args:
                        synthetic_faces(None, frames=90))
    inputs = ProjectContext(root).render_style().data
    hook, = graphics_snapshot(ProjectContext(root)).titles
    captions(root)
    geometry = caption_band_geometry(inputs["captions"], safe_bottom_px=1740)
    slots = json.loads(fake_layers[-1]["variables"]["phrases"])
    assert all(geometry.y + slot.get("offset_px", 0)
               >= hook.plan.y + hook.plan.height + inputs["graphics"]["face_clearance_px"]
               for slot in slots)
    assert captions(root)["noop"]
    write(root / "plans/text-hook.json", choice("Rant", "typed"))
    assert not captions(root).get("noop", False)


def test_text_hook_keeps_low_caption_placement(hooked_project):
    style = ProjectContext(hooked_project).render_style()
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    slots = [{"start": 0, "end": 1, "words": [{"text": "Synthetic"}]}]
    assert (caption_offsets(slots, geometry, None, style, text_hook_bottom=470)
            == caption_offsets(slots, geometry, None, style))


def test_text_hook_refuses_captions_with_no_clear_position(hooked_project):
    style = ProjectContext(hooked_project).render_style()
    style.data["captions"].update(position="middle", top_px=700, size_px=180,
                                  background_mode="box")
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    slots = [{"start": 0, "end": 1, "words": [{"text": "Synthetic"}]}]
    faces = synthetic_faces(Box(0, 600, 1080, 1200), frames=90)
    with pytest.raises(ValidationError, match="Text hook and face"):
        caption_offsets(slots, geometry, faces, style, text_hook_bottom=470)


@pytest.mark.parametrize("kind", ["insert", "background", "moment", "sticker_freeze"])
def test_text_hook_composite_stays_above_picture_and_cards(picture, kind):
    root, _, timeline, titles, caption = picture
    synth.picture_source(root / "hook.mov", "color=white:s=32x24:r=30", 60, alpha=True)
    hook = {**titles[0], "id": "agent-text-hook", "kind": "title", "layer": "behind",
            "path": "hook.mov"}
    item = timeline.elements[0]
    if kind in {"insert", "background"}:
        item = ResolvedElement("broll-picture", kind, "broll", 12, 48, "front", {
            "media": {"path": "cut/cut.mp4", "in_s": 0, "speed": 1,
                      "sha256": receipt.sha256_file(root / "cut/cut.mp4")},
            "box_px": [0, 0, 256, 192], "fit": "cover",
            **({"layout": "bottom", "person_box_px": [45, 67, 166, 125]}
               if kind == "background" else {}),
        })
    elif kind == "sticker_freeze":
        item = replace(item, payload={"style": "sticker_freeze"})
    selected = replace(timeline, elements=(item, timeline.elements[1], ResolvedElement(
        "agent-text-hook", "title", "agent", 0, 60, "front", {"lines": ["Rant"]})))
    # Both a front card and a caption cross the hook to prove the final layer order.
    caption = {**caption, "x": 16, "y": 0}
    out = root / "hook-picture/picture.mp4"
    assemble_picture.render_picture(root, out, work_dir=out.parent, timeline=selected,
                                    title_layers=[hook, *titles], caption_layer=caption)
    frames = decoded(out)
    for frame in frames:
        np.testing.assert_allclose(frame[4:20, 4:12, 0], 235, atol=3)
        assert frame[4:20, 20:28, 0].mean() < 200
