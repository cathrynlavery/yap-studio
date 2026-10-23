"""Timeline windows, readable geometry and frame-exact animation inputs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from yap_studio import cards, tools
from yap_studio.card_project import CardProject, CardTrack
from yap_studio.errors import ValidationError
from yap_studio.render.text_metrics import text_width_em
from yap_studio.style import resolve


def project(start: int = 720, total: int = 1800) -> CardProject:
    return CardProject(30, start, total, (
        CardTrack("Video", ((0, 750), (750, 1200), (1200, total))),
        CardTrack("Cards", ((start, start + 90),)),
    ))


def plan(*, start: int = 720, total: int = 1800, frames: int = 90,
         view: str = "near") -> cards.CardPlan:
    return cards.card_layer_plan("edit_timeline", {"view": view},
                                 resolve("preset:clean_minimal"), frames=frames,
                                 project=project(start, total))


@pytest.mark.parametrize(("start", "total", "frames", "expected"), [
    (720, 1800, 90, (630, 900)),
    (0, 1800, 90, (0, 270)),
    (1710, 1800, 90, (1530, 1800)),
    (0, 180, 45, (0, 180)),
    (600, 1800, 45, (502.5, 742.5)),
])
def test_near_window(start: int, total: int, frames: int, expected: tuple) -> None:
    variables = plan(start=start, total=total, frames=frames).variables
    assert (variables["window_start_frame"], variables["window_end_frame"]) == expected


def test_whole_window_clips_ticks_and_playhead_pixels() -> None:
    variables = plan(view="whole").variables
    assert (variables["window_start_frame"], variables["window_end_frame"]) == (0, 1800)
    lane = variables["lane_width_px"]
    tracks = json.loads(variables["tracks"])
    assert tracks[0]["clips"][0]["x_px"] == 0
    assert tracks[0]["clips"][0]["width_px"] == pytest.approx(lane * 750 / 1800)
    playhead = json.loads(variables["playhead"])
    assert playhead["start_px"] == pytest.approx(lane * 720 / 1800)
    assert playhead["end_px"] == pytest.approx(lane * 809 / 1800)
    near = plan().variables
    clips = json.loads(near["tracks"])[0]["clips"]
    assert len(clips) == 2
    assert clips[0]["x_px"] == 0
    assert clips[-1]["x_px"] + clips[-1]["width_px"] == pytest.approx(near["lane_width_px"])
    ticks = json.loads(near["ticks"])
    assert [tick["frame"] for tick in ticks] == list(range(630, 901, 30))
    assert [tick["frame"] for tick in ticks if tick["major"]] == [750, 900]


def test_labels_follow_every_output_frame() -> None:
    variables = plan(start=0).variables
    labels = json.loads(variables["timecodes"])
    assert len(labels) == 90
    assert (labels[0], labels[29], labels[30], labels[-1]) == (
        "00:00:00", "00:00:29", "00:01:00", "00:02:29",
    )
    assert json.loads(plan(start=1242).variables["timecodes"])[0] == "00:41:12"


def test_geometry_does_not_depend_on_facts() -> None:
    style = resolve("preset:clean_minimal")
    first = cards.card_layer_plan("edit_timeline", {"title": "My cut"}, style)
    other = cards.card_layer_plan("edit_timeline", {"title": "My cut"}, style,
                                  project=replace(project(), tracks=(CardTrack("Video", ()),)))
    assert (first.x, first.y, first.width, first.height) == (
        other.x, other.y, other.width, other.height,
    )
    for key in ("text_px", "timecode_px", "title_px", "label_width_px", "lane_width_px"):
        assert first.variables[key] == other.variables[key]


@pytest.mark.parametrize(("count", "height"), [(2, 220), (3, 272)])
def test_panel_fits_track_rows(count: int, height: int) -> None:
    facts = project()
    if count == 3:
        facts = replace(facts, tracks=(facts.tracks[0], CardTrack("B-roll", ((720, 750),)),
                                      facts.tracks[-1]))
    layer = cards.card_layer_plan("edit_timeline", {}, resolve("preset:clean_minimal"),
                                   project=facts)
    variables = layer.variables
    tracks = json.loads(variables["tracks"])
    assert variables["panel_height_px"] == height
    assert height - (tracks[-1]["y_px"] + 44) == 24
    assert variables["text_px"] == variables["title_px"] == 30
    assert (layer.x, layer.y, layer.width, layer.height) == (20, 280, 1040, 360)
    source = (cards.template_dir("edit_timeline") / "template.html").read_text()
    assert '"panel-height": `${v.panel_height_px}px`' in source
    assert "align-items: center; justify-content: center;" in source


def test_timecode_uses_forty_pixel_mono_with_a_matching_header() -> None:
    variables = plan().variables
    assert variables["timecode_px"] == 40
    assert variables["timecode_row_px"] == 48
    assert variables["timecode_width_px"] == math.ceil(
        text_width_em("0000:00:00", cards.MONO_FONT, 400) * 40
    )
    source = (cards.template_dir("edit_timeline") / "template.html").read_text()
    assert "right: 24px; height: var(--card-timecode-row);" in source
    assert "font-family: var(--card-mono);" in source
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_width_px"] = 600
    with pytest.raises(ValidationError, match="shorten.*30 px") as caught:
        cards.card_layer_plan("edit_timeline", {"title": "W" * 11}, style)
    assert caught.value.parameter == "slots.title"


def test_template_defaults_keep_sample_facts_in_the_preview() -> None:
    defaults = {item["id"]: item["default"]
                for item in cards.load_template("edit_timeline")["variables"]}
    for name in ("timecodes", "tracks", "ticks"):
        assert defaults[name] == "[]"
        assert json.loads(plan().variables[name])
    assert json.loads(defaults["playhead"]) == {"start_px": 0, "end_px": 0}


def test_title_fit_and_minimum_duration_have_fixes() -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_width_px"] = 600
    with pytest.raises(ValidationError, match="shorten.*30 px") as caught:
        cards.card_layer_plan("edit_timeline", {"title": "W" * 28}, style)
    assert caught.value.parameter == "slots.title"
    with pytest.raises(ValidationError, match="at least 1.5 s.*45 frames.*more frames"):
        cards.card_layer_plan("edit_timeline", {}, style, frames=44)
    assert plan(frames=45).frames == 45


def test_view_choices_have_a_fix() -> None:
    with pytest.raises(ValidationError, match="choose near, whole") as caught:
        cards.card_layer_plan("edit_timeline", {"view": "close"}, resolve("preset:clean_minimal"))
    assert caught.value.parameter == "slots.view"


def test_other_kinds_ignore_project() -> None:
    style = resolve("preset:clean_minimal")
    assert cards.card_layer_plan("stat", {"end": 12}, style, project=project()) == (
        cards.card_layer_plan("stat", {"end": 12}, style)
    )


def test_editor_kinds_share_the_layer_box() -> None:
    style = resolve("preset:clean_minimal")
    timeline = cards.card_layer_plan("edit_timeline", {}, style)
    storyboard = cards.card_layer_plan("storyboard", {}, style)
    assert (timeline.x, timeline.y, timeline.width, timeline.height) == (
        storyboard.x, storyboard.y, storyboard.width, storyboard.height,
    )
    assert "timecodes" in timeline.variables and "thumbnails" not in timeline.variables
    assert "thumbnails" in storyboard.variables and "timecodes" not in storyboard.variables


def test_native_animation_clock_and_shared_motion(tmp_path: Path) -> None:
    if tools.find_tool("node") is None:
        pytest.skip("a JavaScript runtime is needed for animation contracts")
    layer = plan()
    source = (cards.template_dir("edit_timeline") / "template.html").read_text()
    payload = tmp_path / "animation.json"
    payload.write_text(json.dumps({
        "variables": layer.variables, "duration": layer.frames / 30,
        "script": re.findall(r"<script>(.*?)</script>", source, re.S)[0],
    }))
    runner = tmp_path / "animation.cjs"
    runner.write_text("""
const input = JSON.parse(require("node:fs").readFileSync(process.argv[2], "utf8"));
const animations = [];
class Element {
  constructor(name) {
    this.name = name; this.style = {setProperty() {}}; this.children = [];
    this.dataset = {duration: String(input.duration)};
  }
  querySelector(name) { return elements[name]; }
  appendChild(child) { this.children.push(child); }
  animate(frames, options) {
    animations.push({target: this.className || this.name, frames, options});
  }
}
const names = ["root", ".panel", ".title", ".timecode-strip", ".ruler", ".playhead"];
const elements = Object.fromEntries(names.map(name => [name, new Element(name)]));
global.document = {
  getElementById(id) { return elements[id]; },
  createElement(tag) { return new Element(tag); },
};
global.window = {__hyperframes: {getVariables() {return input.variables;}}};
new Function(input.script)();
process.stdout.write(JSON.stringify({animations, labels: elements[".timecode-strip"].children
  .map(row => row.textContent)}));
""")
    result = json.loads(tools.run_tool("node", [str(runner), str(payload)], timeout_s=10).stdout)
    events = result["animations"]
    strip = next(event for event in events if event["target"] == ".timecode-strip")
    assert strip["options"] == {"delay": -0.5 / 30 * 1000, "duration": 3000,
                                 "fill": "both", "easing": "steps(90, end)"}
    for frame in range(layer.frames):
        progress = (frame / 30 * 1000 - strip["options"]["delay"]) / 3000
        index = math.floor(progress * layer.frames)
        assert result["labels"][index] == json.loads(layer.variables["timecodes"])[frame]
    panel = next(event for event in events if event["target"] == ".panel")
    assert panel["frames"] == [{"opacity": 0, "transform": "translateY(8px)"},
                               {"opacity": 1, "transform": "translateY(0)"}]
    assert panel["options"]["duration"] == 300
    fade = next(event for event in events if event["target"] == "root")
    assert fade["frames"] == [{"opacity": 1}, {"opacity": 0}]
    assert fade["options"]["delay"] == 2700 and fade["options"]["duration"] == 300
    blocks = [event for event in events if event["target"].startswith("clip ")]
    assert {event["options"]["delay"] for event in blocks} == {100, 160}
    assert all(event["options"]["duration"] == 350 for event in blocks)
    head = [event for event in events if event["target"] == ".playhead"]
    assert head[0]["frames"] == [{"opacity": 0}, {"opacity": 1}]
    assert head[0]["options"]["duration"] == 200
    position = json.loads(layer.variables["playhead"])
    positions = [float(re.fullmatch(r"translateX\((.*)px\)", frame["transform"])[1])
                 for frame in head[1]["frames"]]
    assert positions == [position["start_px"], position["end_px"]]
    assert head[1]["options"] == {"duration": 89 / 30 * 1000,
                                  "fill": "both", "easing": "linear"}
