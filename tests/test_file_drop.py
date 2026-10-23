"""File card validation, poster identities and fitted motion."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_cards_render import renderer_install as renderer_install
from test_stage_captions import layer_project as layer_project

from yap_studio import cards, ffmpeg, receipt
from yap_studio.errors import ValidationError
from yap_studio.render.text_metrics import text_width_em
from yap_studio.style import resolve


@pytest.fixture
def file_drop_project(layer_project: Path) -> Path:
    from test_stage_captions import set_elements
    from test_stage_graphics import enable_card_kind, graphic
    from test_timeline import write

    enable_card_kind(layer_project, "file_drop")
    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_text())
    data.setdefault("graphics", {})["card_width_px"] = 800
    write(style_path, data)
    clip_source(layer_project)
    set_elements(layer_project, graphic(template="file_drop", hold_s=1.5,
                                        slots={"clip": "synthetic.mp4", "playback": "play",
                                               "poster_at": 0}))
    return layer_project


@pytest.mark.render
@pytest.mark.parametrize("growth", [(80, 0), (0, 140), (80, 140)])
def test_file_drop_enlarged_box_window_matches_receipt_hole(
    file_drop_project: Path, renderer_install, tmp_path: Path, growth: tuple[int, int],
) -> None:
    from test_cards_render import _rgba_frames
    from test_stage_captions import set_elements
    from test_stage_graphics import graphic

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.graphics import graphics_snapshot

    initial = graphics_snapshot(ProjectContext(file_drop_project)).graphics[0].plan
    box = {"x": 40, "y": 280, "width": initial.width + growth[0],
           "height": initial.height + growth[1]}
    set_elements(file_drop_project, graphic(template="file_drop", hold_s=1.5, box=box,
                                            slots={"clip": "synthetic.mp4", "playback": "play",
                                                   "poster_at": 0}))
    first = graphics_snapshot(ProjectContext(file_drop_project)).graphics[0]
    saved = first.placement()
    rebuilt = graphics_snapshot(ProjectContext(file_drop_project), placements=[saved]).graphics[0]
    assert rebuilt.placement() == saved
    assert rebuilt.plan == first.plan
    movie = cards.render_card(rebuilt.plan, tmp_path / "box-work", tmp_path / "box.mov",
                              install=renderer_install).path
    window, = saved["windows"]
    x, y = window["x"] - saved["x"], window["y"] - saved["y"]
    w, h = window["width"], window["height"]
    frame = window["start_frame"] - saved["start_frame"] + window["fade_in_frames"]
    alpha = _rgba_frames(movie, rebuilt.plan)[frame, ..., 3]
    assert not alpha[y + 10:y + h - 10, x + 10:x + w - 10].any()
    assert np.all(alpha[y - 4:y, x + 10:x + w - 10] == 255)
    assert np.all(alpha[y + h:y + h + 4, x + 10:x + w - 10] == 255)
    assert np.all(alpha[y + 10:y + h - 10, x - 4:x] == 255)
    assert np.all(alpha[y + 10:y + h - 10, x + w:x + w + 4] == 255)


def test_file_drop_box_growth_preserves_fitted_panel_offsets(clip: Path) -> None:
    from yap_studio.stages.graphics import _card_box

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_width_px"] = 800
    fitted = cards.card_layer_plan("file_drop", {"clip": clip, "playback": "play",
                                                "poster_at": 0}, style, frames=45)
    enlarged = _card_box(fitted, {"x": 40, "y": 280, "width": fitted.width + 80,
                                  "height": fitted.height + 140})
    play = json.loads(enlarged.variables["play_window"])
    assert play["panel_left_px"] == (fitted.width - fitted.variables["file_panel_width_px"]) / 2
    assert play["panel_top_px"] == cards.PAD_TOP + (
        fitted.variables["panel_max_height_px"] - fitted.variables["panel_height_px"]) / 2
    assert enlarged.windows == fitted.windows


@pytest.mark.render
@pytest.mark.parametrize("selection", [
    {"look": "film", "strength": 1},
    {"look": "eyedropper", "strength": 0.7, "color": "#2466CC"},
])
def test_file_drop_graded_poster_matches_first_window_frame(
    file_drop_project: Path, tmp_path: Path, renderer_install, monkeypatch, selection: dict,
) -> None:
    from test_cards_render import _rgba_frames
    from test_stage_captions import set_elements
    from test_stage_graphics import graphic
    from test_timeline import write

    from yap_studio import grade
    from yap_studio.project.status import ProjectContext
    from yap_studio.render import hyperframes
    from yap_studio.stages import assemble_picture, graphics

    clip = file_drop_project / "colour.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=0x20506e:s=320x180:r=30",
                "-t", "2", "-c:v", "libx264", "-threads", "1", str(clip)], timeout_s=30)
    project_path = file_drop_project / "project.json"
    data = json.loads(project_path.read_text())
    write(project_path, {**data, "grade": selection})
    set_elements(file_drop_project, graphic(template="file_drop", hold_s=1.5,
                                            slots={"clip": clip.name, "playback": "play",
                                                   "poster_at": 0}))
    context = ProjectContext(file_drop_project)
    snapshot = graphics.graphics_snapshot(context).graphics[0]
    monkeypatch.setattr(hyperframes, "ensure_installed", lambda: renderer_install)
    movie = graphics._render_card(snapshot, tmp_path / "grade-work", tmp_path / "grade.mov").path
    local, = snapshot.plan.windows
    window, = snapshot.placement()["windows"]
    spec = grade.resolve_grade(context.render_style().data, context.project.grade,
                               cut=file_drop_project / "cut/cut.mp4")
    graph = assemble_picture._Graph()
    branch = assemble_picture._window_clip(graph, assemble_picture._source(
        clip, [], [], owner="clip"), window, owner="clip")
    branch = graph.filter(branch, grade.rgba_filter(grade.write_cube(spec, tmp_path / "cube")))
    decoded = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
                          "-filter_complex", ";".join(graph.filters), "-map", f"[{branch}]",
                          "-fps_mode", "passthrough", "-pix_fmt", "rgba", "-f", "rawvideo",
                          "pipe:1"], binary=True, timeout_s=30).stdout
    _, _, w, h = assemble_picture._window_rect(window)
    frames = np.frombuffer(decoded, np.uint8).reshape(-1, h, w, 4)
    expected = frames[window["fade_in_frames"], h // 2, w // 2, :3]
    rgba = _rgba_frames(movie, snapshot.plan)
    x, y = local["x"] + local["width"] // 2, local["y"] + local["height"] // 2
    np.testing.assert_allclose(rgba[local["start"] - 1, y, x, :3], expected, atol=3)
    paper = np.array([int(snapshot.plan.variables["paper_color"][i:i + 2], 16)
                      for i in (1, 3, 5)])
    np.testing.assert_allclose(rgba[local["start"] - 1, y, local["x"] - 10, :3], paper, atol=2)


@pytest.mark.render
@pytest.mark.parametrize("preview", ["templates", "gallery"])
def test_file_drop_play_previews_keep_poster_opaque(
    clip: Path, tmp_path: Path, renderer_install, monkeypatch, preview: str,
) -> None:
    from test_cards_render import _rgba_frames

    from yap_studio import cli
    from yap_studio.gallery import live
    from yap_studio.render import hyperframes

    monkeypatch.setattr(hyperframes, "ensure_installed", lambda: renderer_install)
    slots = {"clip": str(clip), "playback": "play", "poster_at": 0}
    seen = []
    original = cards.render_card

    def render(plan, work, out, **kwargs):
        seen.append(plan)
        return original(plan, work, out, **kwargs)

    monkeypatch.setattr(cards, "render_card", render)
    if preview == "templates":
        source = tmp_path / "slots.json"
        source.write_text(json.dumps(slots))
        out = tmp_path / "preview"
        assert cli.main(["templates", "preview", "file_drop", "--style", "preset:clean_minimal",
                         "--frames", "45", "--slots", str(source), "--out", str(out)]) == 0
        movie = out / "file_drop.mov"
        layer, = seen
    else:
        style = resolve("preset:clean_minimal")
        style.data["graphics"]["hold_s"] = 1.5
        layer, = live.live_layers(style, live.sample_words("Small steps."), tmp_path / "live",
                                   only="card", card="file_drop", slots=slots)
        movie = hyperframes.render_layer(layer.composition, tmp_path / "live.mov",
                                         frames=layer.frames, variables=layer.variables,
                                         fps=layer.fps, install=renderer_install).path
    alpha = _rgba_frames(movie, layer)[..., 3]
    v = layer.variables
    origin_x = (layer.width - v["file_panel_width_px"]) / 2
    origin_y = cards.PAD_TOP + (v["panel_max_height_px"] - v["panel_height_px"]) / 2
    def point(pose, x):
        angle = math.radians(pose["angle"])
        dx = x - pose["width_px"] / 2
        return (round(origin_y + pose["top_px"] + pose["height_px"] / 2 + dx * math.sin(angle)),
                round(origin_x + pose["left_px"] + pose["width_px"] / 2 + dx * math.cos(angle)))

    for frame, pose in enumerate(json.loads(v["motion"])[:-1]):
        poster = point(pose, pose["poster_left_px"] + pose["poster_width_px"] / 2)
        chrome = point(pose, pose["poster_left_px"] - 10)
        assert alpha[frame, *poster] == alpha[frame, *chrome]
    assert alpha[30, *poster] > 0


def clip_source(root: Path, name: str = "synthetic.mp4") -> Path:
    source = root / name
    if not source.exists():
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30", "-t", "2", "-c:v", "libx264",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(source)], timeout_s=30)
    return source


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return clip_source(tmp_path)


def test_bleep_masks_a_direct_card_plan_before_fitting(clip: Path) -> None:
    source = clip.with_name("shitting.mp4")
    source.write_bytes(clip.read_bytes())
    style = resolve("preset:clean_minimal")
    style.data["music"]["bleep_swears"] = True
    style.data["graphics"]["card_width_px"] = 800
    layer = cards.card_layer_plan("file_drop", {"clip": source}, style, frames=45)
    assert layer.variables["name"] == "s*******.mp4"
    assert (text_width_em(layer.variables["name"], style.data["graphics"]["font"], 600)
            * layer.variables["name_px"] <= layer.variables["text_width_px"])


def test_bleep_fits_the_masked_fallback_name(file_drop_project: Path, monkeypatch) -> None:
    from test_stage_captions import set_elements
    from test_stage_graphics import graphic
    from test_timeline import write

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.graphics import graphics_snapshot

    source = clip_source(file_drop_project, "shitting.mp4")
    style_path = file_drop_project / "style.json"
    data = json.loads(style_path.read_bytes())
    data["music"] = {"bleep_swears": True}
    write(style_path, data)
    set_elements(file_drop_project, graphic(template="file_drop", hold_s=1.5,
                                            slots={"clip": source.name, "playback": "play",
                                                   "poster_at": 0}))
    context = ProjectContext(file_drop_project)
    snapshot, = graphics_snapshot(context).graphics
    values = snapshot.plan.variables
    assert values["name"] == "s*******.mp4"
    assert snapshot.plan.poster.source == source
    assert snapshot.placement()["display_text"][0] == values["name"]
    units = text_width_em(values["name"], context.render_style().data["graphics"]["font"], 600)
    assert units * values["name_px"] <= values["text_width_px"]
    for frame in json.loads(values["motion"]):
        assert units * frame["name_px"] <= frame["text_width_px"] + .01
    from yap_studio import file_cards

    file_variables = file_cards.file_variables

    def old_fit(values, *args, **kwargs):
        fitted = file_variables({**values, "name": source.name}, *args, **kwargs)
        return {**fitted, "name": values["name"]}

    monkeypatch.setattr(file_cards, "file_variables", old_fit)
    previous, = graphics_snapshot(ProjectContext(file_drop_project)).graphics
    assert previous.plan.variables["name_px"] != values["name_px"]
    assert previous.digest != snapshot.digest


@pytest.mark.parametrize("on", [False, True])
def test_long_fallback_name_is_fitted_after_slot_validation(file_drop_project: Path, on) -> None:
    from test_stage_captions import set_elements
    from test_stage_graphics import graphic
    from test_timeline import write

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.graphics import graphics_snapshot

    source = clip_source(file_drop_project, "i" * 41 + ".mp4")
    assert len(source.name) == 45
    style_path = file_drop_project / "style.json"
    data = json.loads(style_path.read_bytes())
    data["music"] = {"bleep_swears": on}
    write(style_path, data)
    set_elements(file_drop_project, graphic(template="file_drop", hold_s=1.5,
                                            slots={"clip": source.name}))
    snapshot, = graphics_snapshot(ProjectContext(file_drop_project)).graphics
    values = snapshot.plan.variables
    assert values["name"] == source.name
    assert snapshot.plan.poster.source == source
    units = text_width_em(source.name, ProjectContext(file_drop_project).render_style().data[
        "graphics"]["font"], 600)
    assert units * values["name_px"] <= values["text_width_px"]
    if on:
        assert snapshot.placement()["display_text"][0] == source.name


def plan(clip: Path, **slots) -> cards.CardPlan:
    return cards.card_layer_plan("file_drop", {"clip": clip, **slots},
                                 resolve("preset:clean_minimal"), frames=75)


@pytest.mark.parametrize("fps,frames,drop", [(30, 75, 0.15), (24, 60, 0.3), (30, 45, 0.15)])
@pytest.mark.parametrize("size", [(320, 180), (180, 320)])
def test_play_window_matches_settled_poster(
    clip: Path, monkeypatch, fps: int, frames: int, drop: float, size: tuple[int, int],
) -> None:
    from yap_studio.file_cards import FILE_DROP_SETTLE_S

    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: SimpleNamespace(
        has_video=True, width=size[0], height=size[1], video_duration_s=2, fps=Fraction(30),
    ))
    layer = cards.card_layer_plan("file_drop", {"clip": clip, "playback": "play",
                                              "poster_at": 10, "drop_at": drop},
                                  resolve("preset:clean_minimal"), frames=frames, fps=fps)
    window, = layer.windows
    v = layer.variables
    start = math.ceil((Fraction(str(drop)) + Fraction(str(FILE_DROP_SETTLE_S))) * fps)
    assert window == {
        "slot": "clip", "index": 0,
        "x": (layer.width - v["file_panel_width_px"]) // 2 + 24,
        "y": cards.PAD_TOP + (v["panel_max_height_px"] - v["poster_height_px"]) // 2,
        "width": v["poster_width_px"], "height": v["poster_height_px"], "radius": 8,
        "start": start, "frames": frames - start, "fade_in": 2,
        "fade_out": max(1, min(9, frames * 15 // 100)),
        "source": {"kind": "media", "path": str(clip), "sha256": "", "in_s": 0.2,
                   "speed": 1, "focus": [0.5, 0.5], "audio": "mute"},
    }
    assert Fraction(layer.poster.time_s) == Fraction(1, 5)
    assert window["width"] % 2 == window["height"] % 2 == 0
    assert 0 <= window["x"] < window["x"] + window["width"] <= layer.width
    assert 0 <= window["y"] < window["y"] + window["height"] <= layer.height
    assert 0 < window["radius"] <= min(window["width"], window["height"]) / 2
    pose = json.loads(v["motion"])[start]
    assert pose["top_px"] == pose["angle"] == 0
    assert pose["poster_width_px"] == window["width"]
    assert pose["poster_height_px"] == window["height"]


def test_still_is_the_default_without_windows(clip: Path) -> None:
    default = plan(clip)
    assert default == plan(clip, playback="still")
    assert default.windows == ()


@pytest.mark.parametrize("duration,poster,fix", [
    (2, 50, "poster_at at most 18.3333.*shorter card"), (1, 0, "use a shorter card"),
])
def test_play_refuses_a_clip_that_ends_before_the_card(
    clip: Path, monkeypatch, duration: float, poster: float, fix: str,
) -> None:
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: SimpleNamespace(
        has_video=True, width=320, height=180, video_duration_s=duration, fps=Fraction(30),
    ))
    with pytest.raises(ValidationError, match=fix) as caught:
        plan(clip, playback="play", poster_at=poster)
    assert caught.value.parameter == "slots.poster_at"


def test_play_accepts_the_last_poster_time_that_fits(clip: Path) -> None:
    layer = plan(clip, playback="play", poster_at=18)
    window, = layer.windows
    assert window["source"]["in_s"] + window["frames"] / layer.fps <= 2
    with pytest.raises(ValidationError, match="poster_at"):
        plan(clip, playback="play", poster_at=19)


def test_play_poster_fix_fits_without_rounding_past_the_end(clip: Path) -> None:
    style = resolve("preset:clean_minimal")
    with pytest.raises(ValidationError) as caught:
        cards.card_layer_plan("file_drop", {"clip": clip, "playback": "play"}, style, frames=76)
    latest = float(re.search(r"poster_at at most ([\d.]+)", caught.value.detail)[1])
    layer = cards.card_layer_plan("file_drop", {"clip": clip, "playback": "play",
                                              "poster_at": latest}, style, frames=76)
    window, = layer.windows
    assert 0 <= 2 - (window["source"]["in_s"] + window["frames"] / 30) < 1e-6


def test_playback_slot_refuses_other_values(clip: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        plan(clip, playback="loop")
    assert caught.value.parameter == "slots.playback"


def test_play_keeps_the_file_drop_sound_events() -> None:
    from yap_studio.sfx.place import _event_times, _template_events

    def element(playback: str) -> SimpleNamespace:
        return SimpleNamespace(
            start_s=1, end_s=3.5, kind="graphic",
            payload={"template": "file_drop", "slots": {"playback": playback}},
        )

    play = element("play")
    still = element("still")
    landing = [(("land", "settle"), "file_drop", pytest.approx(1.53))]
    assert _template_events(play) == _template_events(still) == landing
    assert _event_times(play) == _event_times(still) == [
        (("card_in",), "card:in", 1),
        *landing,
        (("card_out",), "card:out", pytest.approx(3.2)),
    ]


@pytest.mark.parametrize("value", [None, "missing.mp4", "wrong.png", 42, "bad\n.mp4"])
def test_video_slot_refuses_invalid_paths(tmp_path: Path, value) -> None:
    if isinstance(value, str) and value == "wrong.png":
        (tmp_path / value).write_bytes(b"synthetic image")
    with pytest.raises(ValidationError) as caught:
        plan(tmp_path / value if isinstance(value, str) else value)
    assert caught.value.parameter == "slots.clip"


@pytest.mark.parametrize("changes,fix", [
    ({"has_video": False}, "video stream"),
    ({"video_duration_s": 0}, "positive video duration"),
    ({"video_duration_s": None}, "positive video duration"),
    ({"video_duration_s": -1}, "positive video duration"),
    ({"video_duration_s": float("nan")}, "positive video duration"),
    ({"video_duration_s": float("inf")}, "positive video duration"),
])
def test_video_slot_checks_stream_and_duration(
    clip: Path, monkeypatch, changes: dict, fix: str,
) -> None:
    info = dict(has_video=True, width=320, height=180, duration_s=3,
                video_duration_s=2, fps=Fraction(30))
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: SimpleNamespace(
        **{**info, **changes},
    ))
    with pytest.raises(ValidationError, match=fix) as caught:
        plan(clip)
    assert caught.value.parameter == "slots.clip"


def test_file_name_default_badge_and_poster_percent(clip: Path) -> None:
    layer = plan(clip, badge="raw", poster_at=25)
    assert layer.variables["name"] == clip.name
    assert layer.variables["badge"] == "RAW"
    assert layer.variables["duration_text"] == "0:02"
    assert layer.poster.time_s == "1/2"
    assert layer.assets == ()
    assert layer.variables["poster_src"] == "poster.jpg"
    assert layer.variables["poster_width_px"] / layer.variables["poster_height_px"] == (
        pytest.approx(320 / 180, abs=1 / layer.variables["poster_height_px"])
    )
    assert plan(clip, badge="").variables["badge"] == ""


@pytest.mark.parametrize("duration,text", [(0.4, "0:01"), (59.6, "1:00"),
                                           (3600, "1:00:00")])
def test_duration_label(clip: Path, monkeypatch, duration: float, text: str) -> None:
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: SimpleNamespace(
        has_video=True, width=320, height=180, duration_s=duration + 1,
        video_duration_s=duration, fps=Fraction(30),
    ))
    assert plan(clip).variables["duration_text"] == text


def test_name_fit_and_drop_window_refusals(clip: Path) -> None:
    with pytest.raises(ValidationError, match="set name to a shorter label") as caught:
        plan(clip, name="W" * 40)
    assert caught.value.parameter == "slots.name"
    for at in (-0.1, 1.2):
        with pytest.raises(ValidationError, match="before the fade-out") as caught:
            plan(clip, drop_at=at)
        assert caught.value.parameter == "slots.drop_at"
    with pytest.raises(ValidationError, match="at least 1.5 s"):
        cards.card_layer_plan("file_drop", {"clip": clip}, resolve("preset:clean_minimal"),
                              frames=44)
    assert cards.card_layer_plan("file_drop", {"clip": clip}, resolve("preset:clean_minimal"),
                                 frames=45).frames == 45


@pytest.mark.parametrize("slot,value", [("poster_at", -1), ("poster_at", 101),
                                       ("drop_at", float("nan")), ("badge", "x" * 9)])
def test_file_slot_limits(clip: Path, slot: str, value) -> None:
    with pytest.raises(ValidationError) as caught:
        plan(clip, **{slot: value})
    assert caught.value.parameter == f"slots.{slot}"


def test_poster_is_cached_and_only_extracted_at_preparation(
    clip: Path, tmp_path: Path, monkeypatch,
) -> None:
    from yap_studio import card_stills

    calls = []
    original = card_stills.extract_frame

    def extract(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(card_stills, "extract_frame", extract)
    layer = plan(clip, poster_at=25)
    layer = replace(layer, poster=replace(layer.poster, cache_root=tmp_path,
                                          sha256=receipt.sha256_file(clip)))
    assert not calls
    for index in range(2):
        folder = cards.prepare_card(layer, tmp_path / f"work-{index}")
        assert (folder / "poster.jpg").is_file()
        info = ffmpeg.probe(folder / "poster.jpg")
        assert info.width == layer.variables["poster_width_px"] * 2
        assert info.height == layer.variables["poster_height_px"] * 2
    assert len(calls) == 1 and calls[0]["time_s"] == 0.5
    clip.write_bytes(clip.read_bytes() + b"changed synthetic clip")
    with pytest.raises(ValidationError, match="source changed"):
        cards.prepare_card(layer, tmp_path / "changed")


def test_standalone_render_key_uses_clip_bytes(clip: Path) -> None:
    from yap_studio.planner.cards import _render_key

    layer = plan(clip)
    first = _render_key(layer)
    clip.write_bytes(clip.read_bytes() + b"synthetic padding")
    assert _render_key(plan(clip)) != first


@pytest.mark.parametrize("unsupported", [False, True])
@pytest.mark.parametrize("retained", [False, True])
def test_file_poster_uses_hdr_conversion_and_its_fallback(
    clip: Path, tmp_path: Path, monkeypatch, unsupported: bool, retained: bool,
) -> None:
    from yap_studio import card_stills, hdr

    layer = plan(clip)
    conversion = hdr.HdrConversion("test", [], "hdr_to_bt709")
    conversions, extractions, hashes = [], [], []
    original_hash = receipt.sha256_file
    if retained:
        layer = replace(layer, poster=replace(layer.poster, sha256=original_hash(clip)))

    def convert(info):
        conversions.append(info)
        if unsupported:
            raise hdr.HdrUnsupportedError()
        return conversion

    def extract(source, info, out, **kwargs):
        extractions.append(kwargs)
        out.write_bytes(b"synthetic poster")

    def hash_source(source):
        hashes.append(source)
        return original_hash(source)

    monkeypatch.setattr(hdr, "conversion", convert)
    monkeypatch.setattr(card_stills, "extract_frame", extract)
    monkeypatch.setattr(receipt, "sha256_file", hash_source)
    cards.prepare_card(layer, tmp_path / "work")
    assert len(conversions) == len(extractions) == 1
    assert extractions[0]["conversion"] == (None if unsupported else conversion)
    assert hashes.count(clip) == 1


def test_fall_and_settle_are_sampled_on_the_frame_clock(clip: Path) -> None:
    layer = plan(clip)
    v = layer.variables
    motion = json.loads(v["motion"])
    assert v["land_at_s"] == pytest.approx(0.53)
    assert motion[0]["width_px"] / v["file_panel_width_px"] == pytest.approx(1.12, abs=0.001)
    assert motion[0]["top_px"] + (motion[0]["height_px"] - v["panel_height_px"]) / 2 == (
        pytest.approx(-4, abs=0.001)
    )
    assert motion[0]["angle"] == -6 and motion[0]["opacity"] == 0
    assert motion[10]["opacity"] == 1
    settled = motion[26]
    assert settled["top_px"] == settled["angle"] == 0
    assert settled["width_px"] == v["file_panel_width_px"]
    assert settled["height_px"] == v["panel_height_px"]
    assert settled["shadow"] == 1
    assert motion[16]["height_px"] < settled["height_px"]
    assert motion[16]["at_s"] == 15.5 / 30


def test_poster_at_the_clip_end_has_a_readable_frame(clip: Path, tmp_path: Path) -> None:
    layer = plan(clip, poster_at=100)
    assert layer.poster.time_s == "59/30"
    composition = cards.prepare_card(layer, tmp_path / "end")
    assert ffmpeg.probe(composition / "poster.jpg").has_video


@pytest.mark.parametrize("fps", [24, 30])
def test_file_poster_and_length_use_video_before_a_longer_audio_tail(
    tmp_path: Path, fps: int,
) -> None:
    clip = tmp_path / "audio-tail.mp4"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"testsrc2=size=320x180:rate={fps}:duration=2", "-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=48000:duration=3", "-c:v", "libx264",
                "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)], timeout_s=30)
    info = ffmpeg.probe(clip)
    assert info.duration_s == pytest.approx(3)
    assert info.video_duration_s == 2
    layer = plan(clip, poster_at=100)
    assert layer.variables["duration_text"] == "0:02"
    assert Fraction(layer.poster.time_s) == Fraction(2) - Fraction(1, fps)
    composition = cards.prepare_card(layer, tmp_path / "work")
    assert ffmpeg.probe(composition / "poster.jpg").has_video


def test_portrait_poster_keeps_the_clips_display_shape(clip: Path, monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: SimpleNamespace(
        has_video=True, width=180, height=320, duration_s=2,
        video_duration_s=2, fps=Fraction(30),
    ))
    v = plan(clip).variables
    assert v["poster_width_px"] / v["poster_height_px"] == pytest.approx(180 / 320, abs=0.01)
    assert v["poster_height_px"] == v["panel_height_px"] - 48


@pytest.mark.parametrize("look", ["paper", "minimal", "dark", "bright"])
@pytest.mark.parametrize("width", [960, 1080])
def test_motion_bounds_include_the_widest_name_and_every_shadow(
    clip: Path, look: str, width: int,
) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(style=look, card_width_px=width)
    room = math.ceil((width - 76) / 2)
    count = min(40, math.floor(room / (30 * text_width_em("W", "Inter", 600))))
    layer = cards.card_layer_plan("file_drop", {"clip": clip, "name": "W" * count},
                                  style, frames=75)
    v = layer.variables
    shadows = {"paper": [(0, 12, 44), (0, 3, 16)],
               "minimal": [(0, 0, 2), (0, 8, 36)],
               "dark": [(0, 12, 44), (0, 0, 2)], "bright": [(0, 12, 44)]}[look]
    points = []
    for frame in json.loads(v["motion"]):
        w, h = frame["width_px"], frame["height_px"]
        angle = math.radians(frame["angle"])
        boxes = [(0, 0, w, h)]
        for x, y, spread in shadows:
            left = frame["shadow_left_px"] + x - spread
            top = frame["shadow_top_px"] + y - spread
            boxes.append((left, top, left + frame["shadow_width_px"] + 2 * spread,
                          top + frame["shadow_height_px"] + 2 * spread))
        for left, top, right, bottom in boxes:
            for x, y in ((left, top), (right, top), (right, bottom), (left, bottom)):
                dx, dy = x - w / 2, y - h / 2
                points.append(((layer.width - v["file_panel_width_px"]) / 2
                               + frame["left_px"] + w / 2
                               + dx * math.cos(angle) - dy * math.sin(angle),
                               24 + (272 - v["panel_height_px"]) / 2 + frame["top_px"] + h / 2
                               + dx * math.sin(angle) + dy * math.cos(angle)))
        assert min(frame["name_px"], frame["duration_px"], frame["badge_px"]) >= 30
    actual = [min(x for x, _ in points), min(y for _, y in points),
              max(x for x, _ in points), max(y for _, y in points)]
    assert actual[0] >= 2 and actual[1] >= 2
    assert actual[2] <= layer.width - 2 and actual[3] <= layer.height - 2
    assert json.loads(v["motion_bounds"]) == pytest.approx(actual, abs=0.001)
    assert (layer.width, layer.height) == (width + 80, 360)
