"""Synthetic Text hook fit, choice, placement and project contracts."""

import json

import pytest
from test_planner_cards import synthetic_faces
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_timeline import write

from yap_studio.errors import ValidationError
from yap_studio.faces.model import Box
from yap_studio.project.model import parse_document
from yap_studio.project.status import ProjectContext
from yap_studio.stages.captions import title_layer_plan
from yap_studio.stages.graphics import graphics_snapshot, place_title
from yap_studio.style import resolve_style_data
from yap_studio.style.loader import ResolvedStyle
from yap_studio.text_hook import fit_text_hook, unspoken_numbers


def style(color="white"):
    data = resolve_style_data({"schema_version": 1, "hook": {
        "style": "text_hook", "text_hook_color": color, "font": "Gelasio", "size_px": 160,
    }})
    return ResolvedStyle("Synthetic", "test", data)


def choice(chosen=None, source="idea"):
    return {"schema": "yap-studio/text-hook@1", "ideas": [
        "A clearer way to begin", "Start with the hard part", "What makes this work?",
    ], "chosen": chosen, "source": source}


def test_title_text_hook_fit():
    short = fit_text_hook("Make it clear")
    assert short.lines == ("Make it clear",) and short.size_px == 72
    balanced = fit_text_hook("A better opening starts with a clear idea")
    assert len(balanced.lines) == 2
    assert all(len(line.split()) >= 2 for line in balanced.lines)
    assert min(balanced.widths) / max(balanced.widths) >= .65
    assert balanced.size_px == 72
    floor = fit_text_hook(("W" + "i" * 43 + " now ") * 2)
    assert floor.size_px == 60 and len(floor.lines) == 2
    for count in range(4, 20):
        try:
            fitted = fit_text_hook(" ".join(["wide"] * count))
        except ValidationError:
            continue
        if 60 <= fitted.size_px < 72:
            break
    else:
        pytest.fail("fit never uses the 60 to 71 px range")
    assert max(fitted.widths) + 52 <= 1080 * .84


@pytest.mark.parametrize("text,size", [("Rant", 72), ("alone", 72), ("W" * 14, 60)])
def test_title_text_hook_single_word(text, size):
    fitted = fit_text_hook(text)
    assert fitted.lines == (text,) and fitted.size_px == size
    assert fitted.widths[0] + 52 <= 1080 * .84


@pytest.mark.parametrize("text", ["", " \t\n", "W" * 15, "W" * 40, "wide " * 40,
                                       "W" * 18 + " a b", "W" * 8 + " " + "W" * 8 + " now"])
def test_title_text_hook_refuses_unreadable_text(text):
    with pytest.raises(ValidationError, match="shorter text|two balanced lines"):
        fit_text_hook(text)


@pytest.mark.parametrize("color,background,ink", [
    ("white", "#FFFFFF", "#000000"), ("black", "#000000", "#FFFFFF"),
    ("accent", "#D8FF2F", "#000000"),
])
def test_title_text_hook_style_and_plan(color, background, ink):
    resolved = style(color)
    resolved.data["hook"]["accent_color"] = "#D8FF2F"
    plan = title_layer_plan("Make it clear", "", resolved, frames=90)
    assert plan.variables["font_name"] == "Inter"
    assert plan.variables["size_px"] == 72
    assert plan.variables["background_color"] == background
    assert plan.variables["text_color"] == ink
    assert plan.y == 250 and plan.frames == 90
    assert plan.width <= 1080 * .84
    assert json.loads(plan.variables["lines"]) == ["Make it clear"]


def test_text_hook_schema_defaults_and_refusal():
    assert resolve_style_data({"schema_version": 1, "hook": {
        "style": "text_hook",
    }})["hook"]["text_hook_color"] == "white"
    with pytest.raises(ValidationError):
        style("pink")
    assert parse_document("text-hook", choice()).chosen is None
    assert parse_document("text-hook", choice("Make it clear", "typed")).chosen == "Make it clear"
    for change in ({"ideas": ["Too few ideas"]}, {"chosen": 1}, {"source": "auto"},
                   {"schema": "wrong"}, {"chosen": "wide " * 40}):
        with pytest.raises(ValidationError):
            parse_document("text-hook", {**choice(), **change})


@pytest.mark.parametrize("fields", [{}, {"whole_video": False}, {"whole_video": True}])
def test_text_hook_whole_video_round_trip(fields):
    original = choice("Make it clear", "typed")
    plan = parse_document("text-hook", {**original, **fields})
    assert plan.whole_video is fields.get("whole_video", False)
    expected = {**original, **({"whole_video": True} if plan.whole_video else {})}
    assert json.dumps(plan.to_dict()).encode() == json.dumps(expected).encode()


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_text_hook_whole_video_requires_boolean(value):
    with pytest.raises(ValidationError):
        parse_document("text-hook", {**choice(), "whole_video": value})


@pytest.mark.parametrize("whole_video", [False, True])
def test_opening_title_refuses_whole_video(whole_video):
    with pytest.raises(ValidationError):
        parse_document("text-hook", {**choice(), "kind": "opening", "whole_video": whole_video})


@pytest.mark.parametrize("text,spoken,refuses", [
    ("Why pay $50 for this?", "It costs fifty dollars.", False),
    ("Why pay $50 for this?", "There are fifty pieces.", True),
    ("Why pay $99 for this?", "It costs fifty dollars.", True),
    ("Save 50% with this", "Save fifty percent.", False),
    ("Save 50% with this", "There are fifty pieces.", True),
    ("Try three clear steps", "There are three steps.", False),
    ("Try three clear steps", "Try again.", True),
])
def test_text_hook_keeps_number_and_money_checks(text, spoken, refuses):
    assert unspoken_numbers(text, spoken) is refuses


def test_title_text_hook_preview_uses_take_words(tmp_path, monkeypatch):
    from yap_studio.gallery import live

    monkeypatch.setattr(live, "prepare_layer_plan", lambda _plan, folder: folder)
    words = live.sample_words("A clearer opening starts with your own words.")
    layer, = live.live_layers(style(), words, tmp_path, only="title")
    assert layer.variables["plain"] == "A clearer opening starts with your own words."
    assert layer.frames == round(4.48 * 30)
    assert layer.variables["font_name"] == "Inter"


def test_title_text_hook_face_floor():
    resolved = style()
    plan = title_layer_plan("Make it clear", "", resolved, frames=90)
    faces = synthetic_faces(Box(400, 420, 240, 240), frames=90)
    moved, evidence = place_title(plan, resolved, faces)
    assert 120 <= moved.y < 250 and evidence["status"] == "clear"
    assert moved.height == plan.height and moved.variables["size_px"] == 72
    blocked = synthetic_faces(Box(400, 230, 240, 240), frames=90)
    with pytest.raises(ValidationError, match="face|shorter text"):
        place_title(plan, resolved, blocked)


@pytest.mark.parametrize("duration,frames", [(2, 60), (2.15, 65)])
@pytest.mark.parametrize("fields", [{}, {"whole_video": False}])
def test_title_text_hook_waits_then_shows_for_opening(
    layer_project, monkeypatch, duration, frames, fields,
):
    assert ProjectContext(layer_project).resolved_timeline().total_frames == 90
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"] = {"style": "text_hook", "duration_s": duration}
    write(layer_project / "style.json", data)
    set_elements(layer_project, {"id": "agent-old-title", "kind": "title",
                                "start": {"time": .2}, "lines": ["Old title"]})
    assert not graphics_snapshot(ProjectContext(layer_project)).titles
    write(layer_project / "plans/text-hook.json", choice())
    assert not graphics_snapshot(ProjectContext(layer_project)).titles
    write(layer_project / "plans/text-hook.json", {**choice("What makes this work?"), **fields})
    monkeypatch.setattr("yap_studio.stages.graphics.picture_faces", lambda *args:
                        synthetic_faces(Box(400, 900, 200, 200), frames=120))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    assert not any(f.rule == "title_words" for f in timeline.findings)
    hook, = (item for item in timeline.elements if item.id == "agent-text-hook")
    assert hook.start_frame == 0 and hook.end_frame == frames < timeline.total_frames
    assert hook.layer == "front"
    title, = graphics_snapshot(context).titles
    assert title.plan.start_frame == 0
    assert title.plan.frames == frames
    assert timeline.valid
    write(layer_project / "plans/text-hook.json", choice("Why pay $999 for this?", "typed"))
    assert any(f.rule == "number_not_spoken" for f in
               ProjectContext(layer_project).resolved_timeline(allow_violations=True).findings)


def test_title_text_hook_opening_is_capped_at_cut_end(layer_project, monkeypatch):
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"] = {"style": "text_hook"}
    write(layer_project / "style.json", data)
    write(layer_project / "plans/text-hook.json", choice("Make it clear", "typed"))
    monkeypatch.setattr("yap_studio.stages.graphics.picture_faces", lambda *args:
                        synthetic_faces(Box(400, 900, 200, 200), frames=90))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    assert timeline.total_frames < round(4.48 * 30)
    assert timeline.end_hold_frames > 0
    hook, = (item for item in timeline.elements if item.id == "agent-text-hook")
    assert hook.start_frame == 0 and hook.end_frame == timeline.total_frames
    assert hook.layer == "front"
    title, = graphics_snapshot(context).titles
    assert title.plan.start_frame == 0 and title.plan.frames == timeline.total_frames
    assert timeline.valid


def test_title_text_hook_whole_video_ends_before_hold(layer_project, monkeypatch):
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"] = {"style": "text_hook", "duration_s": 1}
    write(layer_project / "style.json", data)
    write(layer_project / "plans/text-hook.json", {
        **choice("Make it clear", "typed"), "whole_video": True,
    })
    monkeypatch.setattr("yap_studio.stages.graphics.picture_faces", lambda *args:
                        synthetic_faces(Box(400, 900, 200, 200), frames=90))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    assert timeline.total_frames > 30 and timeline.end_hold_frames > 0
    hook, = (item for item in timeline.elements if item.id == "agent-text-hook")
    assert hook.start_frame == 0 and hook.end_frame == timeline.total_frames
    assert hook.layer == "front"
    title, = graphics_snapshot(context).titles
    assert title.plan.start_frame == 0 and title.plan.frames == timeline.total_frames
    assert not any(f.rule in {"end_hold", "inside_video"} for f in timeline.findings)
    assert timeline.valid


@pytest.mark.render
@pytest.mark.parametrize("text", ["Rant", "A better opening starts with a clear idea"])
@pytest.mark.parametrize("color,rgb,ink", [
    ("white", (255, 255, 255), (0, 0, 0)),
    ("black", (0, 0, 0), (255, 255, 255)),
    ("accent", (216, 255, 47), (0, 0, 0)),
])
def test_title_text_hook_render_pixels(tmp_path, color, rgb, ink, text):
    import numpy as np

    from yap_studio import ffmpeg, paths
    from yap_studio.stages.captions import render_layer_plan

    resolved = style(color)
    resolved.data["hook"]["accent_color"] = "#D8FF2F"
    plan = title_layer_plan(text, "", resolved, frames=6)
    with paths.operation():
        output = render_layer_plan(plan, tmp_path, tmp_path / f"{color}.mov").path
    raw = ffmpeg.run([
        "ffmpeg", "-nostdin", "-i", str(output), "-an", "-pix_fmt", "rgba",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=120, binary=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(6, plan.height, plan.width, 4)
    for frame in (frames[0], frames[-1]):
        solid = frame[:, :, 3] > 250
        assert np.count_nonzero(solid) > plan.height * plan.width * .6
        assert np.count_nonzero(np.max(abs(frame[:, :, :3].astype(int) - rgb), axis=2)
                                [solid] < 12) > 1000
        assert np.count_nonzero(np.max(abs(frame[:, :, :3].astype(int) - ink), axis=2)
                                [solid] < 12) > 100
    assert np.array_equal(frames[0], frames[-1])
