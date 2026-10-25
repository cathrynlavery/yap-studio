"""Floating wide cards keep every face and the caption band clear."""

import json
import subprocess
import threading
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_cards import _preview
from test_cards import image as image
from test_stage_captions import layer_project as layer_project

from yap_studio import cards, ffmpeg, paths
from yap_studio.canvas import require_supported
from yap_studio.errors import ValidationError, YapError
from yap_studio.faces.clearance import gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.style import resolve
from yap_studio.wide_cards import LAYOUTS, REFUSED, place_card


@pytest.fixture(autouse=True)
def installed_registry(monkeypatch, _shared_cache_root):
    from yap_studio.hf_registry import install

    monkeypatch.setattr(install, "cache_root", lambda: _shared_cache_root)


@pytest.mark.render
@pytest.mark.parametrize("kind", [*LAYOUTS, "stat/on_cue", "stat/on_cue_glow",
    "specs_list/on_cue", "specs_list/on_cue_glow", "flow/tags",
    "text_tag/block", "text_tag/underline", "quote/no_role", "name_tag/no_role",
    "image_compare/no_labels"])
def test_wide_browser_content_fills_paper(kind, image, tmp_path):
    from yap_studio.render import hyperframes

    renderer = hyperframes.ensure_installed()
    kind, _, look = kind.partition("/")
    slots = slots_for(kind, image)
    slots.update({"specs_list": {"rows": [{"name": "Format", "value": "Live"},
                                          {"name": "Access", "value": "All"}]},
                  "file_drop": {"name": "Clip"}, "comment_drop": {"text": "Send it"},
                  }.get(kind, {}))
    chosen = resolve("preset:clean_minimal")
    if look:
        if look == "no_role":
            slots["role"] = ""
        elif look == "no_labels":
            slots.update(before_label="", after_label="", split=90)
        elif kind == "text_tag":
            slots["variant"] = "plain" if look == "underline" else look
        else:
            chosen.data["graphics"]["flow_look" if kind == "flow" else "specs_rows"] = look
        if kind == "stat":
            slots["rows"] = ["Film", "Edit"]
    plan = cards.card_layer_plan(kind, slots, chosen,
                                 frames=120, shape="horizontal")
    composition = cards.prepare_card(plan, tmp_path / "browser", install=renderer)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(composition)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run([
            "node", str(Path(__file__).parent / "fixtures/wide_card_probe.mjs")],
            input=json.dumps({"kind": kind, "browser": renderer.browser_path,
                "composition": str(composition),
                "variables": plan.variables,
                "puppeteer": str(renderer.prefix / "node_modules/puppeteer-core"),
                "url": f"http://127.0.0.1:{server.server_port}/index.html"}),
            capture_output=True, text=True, timeout=60, check=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    (tmp_path / "geometry.json").write_text(json.dumps(report, indent=2))
    paper, panel = report["paper"], report["panel"]
    assert 100 < paper["height"] <= plan.variables["wide_paper_height_px"]
    assert panel["y"] - paper["y"] == pytest.approx(48, abs=3)
    assert paper["y"] + paper["height"] - panel["y"] - panel["height"] == pytest.approx(52, abs=3)
    for text in report["text"]:
        minimum = 24 if text["className"] in {
            "wide-kicker", "kicker", "bar-title", "host-text"} else 42
        assert text["size"] >= minimum, text
        if kind == "choices":
            assert text["size"] == 42
            assert all(len(line.split()) >= 2 for line in text["text"].split("\n"))
    if kind == "steps":
        steps = report["steps"]
        assert len(steps) == 3
        assert all(step["x"] == steps[0]["x"] for step in steps)
        assert all(b["y"] - a["y"] - a["height"] == pytest.approx(44)
                   for a, b in zip(steps, steps[1:], strict=False))
        assert sum(step["state"] == "active" for step in steps) == 1
        assert all(step["background"] == "rgb(54, 90, 88)" for step in steps
                   if step["state"] == "active")
        assert all(step["color"] == "rgb(34, 34, 34)" for step in steps
                   if step["state"] != "active")


def slots_for(kind, image):
    if kind.startswith("hf:"):
        from yap_studio.hf_registry.cards import preview_slots

        return preview_slots(kind)
    slots = _preview(kind, image)
    slots.update({
        "steps": {"steps": ["Film", "Edit", "Post"]},
        "flow": {"nodes": ["Film", "Edit", "Post"]},
        "prompt_bar": {"request": "Trim pauses", "placeholder": "Ask for an edit"},
        "quote": {"quote": "Make room for the story.", "name": "Sam", "role": "Writer"},
    }.get(kind, {}))
    return slots


@pytest.mark.parametrize("kind", LAYOUTS)
def test_wide_paper_hugs_content_at_board_scale(kind, image):
    plan = cards.card_layer_plan(kind, slots_for(kind, image), resolve("preset:clean_minimal"),
                                 frames=120, shape="horizontal")
    v = plan.variables
    assert v["wide_paper_width_px"] >= 660
    assert v["wide_paper_height_px"] == v["wide_content_height_px"] + 48 + 52
    assert plan.height == v["wide_paper_height_px"] + cards.PAD_TOP + cards.PAD_BOTTOM


def test_wide_steps_use_board_stack_and_balanced_labels():
    plan = cards.card_layer_plan("steps", {"steps": [
        "You film", "The agent edits", "You comment, it fixes"]},
        resolve("preset:clean_minimal"), frames=120, shape="horizontal")
    assert plan.variables["wide_steps_layout"] == "vertical"
    assert plan.variables["text_px"] == 42
    assert plan.variables["wide_content_height_px"] == 452
    long = cards.card_layer_plan("quote", {"quote":
        "Make more room for the story you really want to tell", "name": "Sam", "role": "Writer"},
        resolve("preset:clean_minimal"), frames=120, shape="horizontal")
    lines = long.variables["quote"].split("\n")
    assert len(lines) == 2 and all(len(line.split()) >= 2 for line in lines)


@pytest.mark.parametrize("kind", LAYOUTS)
def test_every_wide_template_plan(kind, image):
    chosen = resolve("preset:clean_minimal")
    plan = cards.card_layer_plan(kind, slots_for(kind, image), chosen,
                                 frames=120, shape="horizontal")
    plan, evidence = place_card(plan, chosen, [[Box(260, 220, 340, 380)]] * 120,
                                caption_top=864)
    box = Box(**evidence["box"])
    assert plan.shape == "horizontal"
    assert plan.layer_box() == {"x": 0, "y": 0, "width": 1920, "height": 1080}
    assert 96 <= box.x < box.x1 <= 1824
    assert 108 <= box.y < box.y1 <= 840
    assert box.x + box.w / 2 > 960 and gap_px(box, head_box(Box(260, 220, 340, 380))) >= 24
    assert plan.variables.get("font_family", chosen.data["graphics"]["font"])


def test_wide_cards_keep_creator_fonts_and_colours():
    chosen = resolve("preset:clean_minimal")
    chosen.data["graphics"].update(font="Gelasio", paper_color="#FAF0DC",
                                   ink_color="#241F18", accent_color="#7351A8")
    plan = cards.card_layer_plan("stat", {"end": 42}, chosen, shape="horizontal")
    assert plan.fonts[0] == "Gelasio"
    for key in ("paper_color", "ink_color", "accent_color"):
        assert plan.variables[key] == chosen.data["graphics"][key]
    assert plan.variables["card_radius_px"] == 24


@pytest.mark.parametrize("face_x,side", [(260, "right"), (1300, "left")])
@pytest.mark.parametrize("motion", ["none", "streak"])
def test_open_side_and_motion_bounds(face_x, side, motion):
    chosen = resolve("preset:clean_minimal")
    chosen.data["graphics"]["card_motion"] = motion
    plan = cards.card_layer_plan("stat", {"end": 42}, chosen, frames=90, shape="horizontal")
    plan, evidence = place_card(plan, chosen, [[Box(face_x, 200, 300, 400)]] * 90,
                                caption_top=820)
    assert (plan.x > 960) == (side == "right")
    assert plan.y + plan.height <= 796
    assert bool(plan.motion) == (motion == "streak")
    if plan.motion:
        assert plan.motion["side"] == side
        sweep = Box(**evidence["card_motion"]["sweep"])
        assert gap_px(sweep, head_box(Box(face_x, 200, 300, 400))) >= 24


@pytest.mark.parametrize("faces", [None, [[Box(790, 100, 340, 740)]],
                                  [[Box(200, 200, 300, 400)],
                                   [Box(1300, 200, 300, 400)]]])
def test_no_room_or_unmeasured_faces_refuses(faces):
    chosen = resolve("preset:clean_minimal")
    plan = cards.card_layer_plan("stat", {"end": 42}, chosen, shape="horizontal")
    with pytest.raises(ValidationError, match="choose a moment") as error:
        place_card(plan, chosen, faces, caption_top=864)
    assert "choose a moment with room beside the speaker" in error.value.suggested_action.lower()


@pytest.mark.parametrize("kind", REFUSED)
def test_refused_templates_have_their_own_fix(kind):
    with pytest.raises(YapError) as caught:
        cards.card_layer_plan(kind, {}, resolve("preset:clean_minimal"), shape="horizontal")
    assert REFUSED[kind] in caught.value.suggested_action


@pytest.mark.parametrize("kind", REFUSED)
def test_planner_refuses_scene_cards_before_suggesting_they_be_enabled(kind, tmp_path):
    from test_planner_anchors import synthetic_words
    from test_planner_cards import requests

    from yap_studio.planner.cards import plan_cards

    (tmp_path / "project.json").write_text(json.dumps({"shape": "horizontal"}))
    with pytest.raises(YapError) as error:
        plan_cards(requests(template=kind, slots={}), synthetic_words(),
                   resolve("preset:clean_minimal"), frames=900, project_dir=tmp_path)
    assert REFUSED[kind] in error.value.suggested_action


def test_horizontal_allowlist_is_exact(tmp_path):
    from test_landscape import _context

    from yap_studio.project.model import Anchor, Element

    context = _context(tmp_path)
    for kind in LAYOUTS:
        item = Element("agent-card", "graphic", Anchor(time=0), fields={"template": kind})
        require_supported(context, {"agent": SimpleNamespace(elements=[item], lane="agent")})
    for kind in REFUSED:
        item = replace(item, fields={"template": kind})
        with pytest.raises(YapError):
            require_supported(context, {"agent": SimpleNamespace(elements=[item], lane="agent")})
    assert set(cards.CARD_TEMPLATES) == (set(LAYOUTS) - {
        "hf:char-slam-explode", "hf:conic-progress-ring"}) | set(REFUSED)


def test_registry_minimum_panel_refuses_unreadable_fit(monkeypatch):
    from copy import deepcopy

    from yap_studio.hf_registry import catalog

    original = catalog.item
    entry = deepcopy(original("conic-progress-ring"))
    entry["min_panel"] = [800, 220]
    monkeypatch.setattr(catalog, "item", lambda name: entry if name == "conic-progress-ring"
                        else original(name))
    with pytest.raises(YapError) as caught:
        cards.card_layer_plan("hf:conic-progress-ring", {"percent": 72},
                              resolve("preset:clean_minimal"), shape="horizontal")
    assert "Choose another registry card" in caught.value.suggested_action


def wide_plan(tmp_path):
    from test_planner_anchors import synthetic_words
    from test_planner_cards import requests, synthetic_faces

    from yap_studio.planner.cards import plan_cards

    (tmp_path / "project.json").write_text(json.dumps({"shape": "horizontal"}))
    faces = synthetic_faces(Box(260, 220, 340, 380))
    faces = replace(faces, video=replace(faces.video, width=1920, height=1080))
    chosen = resolve("preset:clean_minimal")
    plan = plan_cards(requests(id="agent-stat"), synthetic_words(), chosen, frames=900,
                      faces=faces, project_dir=tmp_path)
    return plan, chosen


def test_wide_planner_exports_automatic_floating_placement(tmp_path):
    plan, _ = wide_plan(tmp_path)
    assert plan.status == "ok"
    assert plan.cards[0].x > 960
    assert plan.cards[0].clearance["layout"] == "floating"
    assert "box" not in plan.lane()["elements"][0]


@pytest.mark.render
def test_wide_planner_layer_reconstructs_and_renders(tmp_path):
    from yap_studio.planner.cards import render_layers, write_plan

    plan, chosen = wide_plan(tmp_path)
    write_plan(plan, tmp_path / "plan")
    result, = render_layers(plan, chosen, tmp_path / "plan")
    info = ffmpeg.probe(result.path)
    assert (info.width, info.height) == (1920, 1080)


@pytest.mark.parametrize("motion", ["none", "streak"])
@pytest.mark.parametrize("captions", ["on", "off"])
def test_graphics_snapshot_replays_wide_placement(layer_project, monkeypatch, motion, captions):
    from test_stage_captions import republish_cut, set_elements

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages import graphics

    project = json.loads((layer_project / "project.json").read_text())
    project.update(shape="horizontal", word_fixes={})
    (layer_project / "project.json").write_text(json.dumps(project))
    chosen = resolve("preset:clean_minimal")
    chosen.data["moves"].update(jump_cuts="hard", opening_push=False)
    chosen.data["graphics"]["card_motion"] = motion
    (layer_project / "style.json").write_text(json.dumps(chosen.data))
    republish_cut(layer_project)
    elements = [{"id": "agent-list", "kind": "graphic", "template": "checklist",
                 "slots": {"items": ["Film", "Edit", "Post"]},
                 "start": {"time": 0}, "hold_s": 2}]
    if captions == "off":
        elements.append({"id": "agent-gap", "kind": "caption_gap",
                         "start": {"time": 0}, "hold_s": 3})
    set_elements(layer_project, *elements)
    face = SimpleNamespace(box=Box(260, 220, 340, 380))
    monkeypatch.setattr(graphics, "picture_faces", lambda *_: SimpleNamespace(
        status="measured", reason=None, frames=[(face,)] * 90))
    snapshot = graphics.graphics_snapshot(ProjectContext(layer_project))
    layer, = snapshot.graphics
    assert layer.plan.shape == "horizontal" and layer.plan.x > 960
    assert layer.plan.y == 206
    assert layer.placement()["width"] == 1920
    monkeypatch.setattr(graphics, "picture_faces", lambda *_: pytest.fail("receipt remeasured"))
    replay = graphics.graphics_snapshot(ProjectContext(layer_project),
                                        placements=[layer.placement()])
    assert replay.inputs == snapshot.inputs
    assert replay.graphics[0].placement() == layer.placement()


@pytest.mark.render
@pytest.mark.parametrize("kind", LAYOUTS)
def test_real_wide_card_at_delivery_size(kind, image, tmp_path):
    chosen = resolve("preset:clean_minimal")
    plan = cards.card_layer_plan(kind, slots_for(kind, image), chosen,
                                 frames=120, shape="horizontal")
    plan, evidence = place_card(plan, chosen, [[Box(260, 220, 340, 380)]] * 120,
                                caption_top=864)
    with paths.operation():
        result = cards.render_card(plan, tmp_path / "work", tmp_path / "card.mov")
    info = ffmpeg.probe(result.path)
    assert (info.width, info.height, info.video_frames) == (1920, 1080, 120)
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-ss", "2.5", "-i", str(result.path),
                      "-frames:v", "1", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                     binary=True, timeout_s=30).stdout
    import numpy as np

    rgba = np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 4).copy()
    box = evidence["box"]
    x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
    assert (rgba[y:y+h, x:x+w, 3] > 240).sum() > 1000
    panel = rgba[y+24:y+h-64, x+40:x+w-40]
    paper = chosen.data["graphics"]["paper_color"].lstrip("#")
    rgb = np.array([int(paper[i:i+2], 16) for i in (0, 2, 4)])
    ink = np.max(np.abs(panel[..., :3].astype(int) - rgb), axis=2) > 40
    assert (ink & (panel[..., 3] > 240)).sum() > 100, "empty paper card"
    for window in plan.windows:
        if window["start"] <= 75 < window["start"] + window["frames"]:
            cx = x + window["x"] + window["width"] // 2
            cy = y + window["y"] + window["height"] // 2
            assert rgba[cy, cx, 3] == 0, "floating paper covered the video window"
    rgba[y:y+h, x:x+w, 3] = 0
    assert not rgba[..., 3].any(), "card ink escaped its face-safe box"
    (tmp_path / "placement.json").write_text(json.dumps(evidence))


@pytest.mark.render
def test_wide_playing_file_keeps_its_window_clear(image, tmp_path):
    import numpy as np

    chosen = resolve("preset:clean_minimal")
    slots = slots_for("file_drop", image)
    slots.update(playback="play", poster_at=0)
    plan = cards.card_layer_plan("file_drop", slots, chosen, frames=60, shape="horizontal")
    plan, _ = place_card(plan, chosen, [[Box(260, 220, 340, 380)]] * 60)
    window, = plan.windows
    assert type(window["x"]) is int and type(window["y"]) is int
    with paths.operation():
        result = cards.render_card(plan, tmp_path / "work", tmp_path / "playing.mov")
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-ss", "1.5", "-i", str(result.path),
                      "-frames:v", "1", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                     binary=True, timeout_s=30).stdout
    rgba = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 4)
    assert rgba[plan.y + window["y"] + window["height"] // 2,
                plan.x + window["x"] + window["width"] // 2, 3] == 0


@pytest.mark.render
@pytest.mark.parametrize("face_x", [260, 1300])
def test_real_wide_motion_stays_in_clear_corridor(face_x, tmp_path):
    import numpy as np

    chosen = resolve("preset:clean_minimal")
    chosen.data["graphics"]["card_motion"] = "streak"
    plan = cards.card_layer_plan("stat", {"end": 42}, chosen, frames=60, shape="horizontal")
    plan, evidence = place_card(plan, chosen, [[Box(face_x, 200, 300, 400)]] * 60,
                                caption_top=864)
    with paths.operation():
        result = cards.render_card(plan, tmp_path / "work", tmp_path / "motion.mov")
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(result.path), "-vf",
                      "select='eq(n,0)+eq(n,6)+eq(n,30)+eq(n,54)+eq(n,59)'",
                      "-fps_mode", "passthrough",
                      "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                     binary=True, timeout_s=30).stdout
    alpha = np.frombuffer(raw, np.uint8).reshape(5, 1080, 1920, 4)[..., 3].copy()
    assert alpha[0].max() == 0 and alpha[2].max() == 255
    assert not np.array_equal(alpha[1], alpha[2])
    assert not np.array_equal(alpha[3], alpha[2])
    sweep = Box(**evidence["card_motion"]["sweep"])
    left, right = max(96, int(sweep.x)), min(1824, int(sweep.x1))
    alpha[:, int(sweep.y):int(sweep.y1), left:right] = 0
    assert not alpha.any(), "motion escaped the face-safe corridor"
