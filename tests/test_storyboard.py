"""Shot facts, cached picture sources and storyboard layout."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_card_project import spans
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_stage_graphics import graphic
from test_timeline import write

from yap_studio import cards, ffmpeg, hdr, receipt, tools
from yap_studio.card_project import CardProject, CardShot, project_facts, sample_project
from yap_studio.card_stills import cached_frame, cached_still
from yap_studio.errors import ValidationError
from yap_studio.project.status import ProjectContext
from yap_studio.stages.timeline import ResolvedElement
from yap_studio.style import resolve


def shot_words(root: Path) -> None:
    write(root / "cut/spans.json", {
        "source": "../source.mov", "segments": [
            {"start": 0, "end": 1}, {"start": 1, "end": 1.2}, {"start": 1.2, "end": 3},
        ],
    })
    source = root / "cut/cut-words.json"
    data = json.loads(source.read_text())
    for word, segment in zip(data["words"], ("one", "one", "two", "one"), strict=True):
        word["segment"] = segment
    write(source, data)
    record = json.loads((root / "cut/receipt.json").read_text())
    for output in record["outputs"]:
        if output["path"] == "cut/cut-words.json":
            output["sha256"] = receipt.sha256_file(source)
    write(root / "cut/receipt.json", record)


def test_shots_follow_consecutive_segments_and_output_edges(layer_project: Path) -> None:
    spans(layer_project)
    shot_words(layer_project)
    set_elements(layer_project, graphic())
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    facts = project_facts(context, timeline, timeline.elements[0])
    assert [(shot.start_frame, shot.end_frame, shot.label) for shot in facts.shots] == [
        (0, 30, "01"), (30, 36, "02"), (36, 90, "03"),
    ]
    assert [shot.source_frame for shot in facts.shots] == [15, 33, 63]
    assert all(shot.source == "cut/cut.mp4" for shot in facts.shots)
    assert str(layer_project) not in json.dumps(asdict(facts))
    assert all(shot.source_sha256 == context.identity.sha256(layer_project / "cut/cut.mp4")
               for shot in facts.shots)
    (layer_project / "cut/cut.mp4").write_bytes(b"changed synthetic cut")
    after = project_facts(ProjectContext(layer_project), timeline, timeline.elements[0])
    assert after.shots[0].source_sha256 != facts.shots[0].source_sha256


def test_shots_start_at_cut_clips_before_their_first_words(
    layer_project: Path, monkeypatch,
) -> None:
    shot_words(layer_project)
    write(layer_project / "cut/spans.json", {
        "source": "../source.mov", "segments": [
            {"start": 0, "end": 0.8}, {"start": 0.8, "end": 1.1},
            {"start": 1.1, "end": 3},
        ],
    })
    set_elements(layer_project, graphic())
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: pytest.fail("facts must use JSON"))
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    facts = project_facts(context, timeline, timeline.elements[0])
    assert [(shot.start_frame, shot.end_frame) for shot in facts.shots] == [
        (0, 24), (24, 33), (33, 90),
    ]
    assert [shot.source_frame for shot in facts.shots] == [12, 28, 61]


def test_wordless_cut_clips_join_the_previous_shot(layer_project: Path) -> None:
    shot_words(layer_project)
    write(layer_project / "cut/spans.json", {
        "source": "../source.mov", "segments": [
            {"start": 0, "end": 0.5}, {"start": 0.5, "end": 0.8},
            {"start": 0.8, "end": 1.1}, {"start": 1.1, "end": 3},
        ],
    })
    set_elements(layer_project, graphic())
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    facts = project_facts(context, timeline, timeline.elements[0])
    assert facts.tracks[0].clips[:2] == ((0, 15), (15, 24))
    assert [(shot.start_frame, shot.end_frame) for shot in facts.shots] == [
        (0, 24), (24, 33), (33, 90),
    ]


@pytest.mark.parametrize("kind", ["insert", "background"])
def test_shot_midpoints_choose_full_picture_media(layer_project: Path, kind: str) -> None:
    spans(layer_project)
    shot_words(layer_project)
    set_elements(layer_project, graphic())
    clip = layer_project / "insert.mov"
    clip.write_bytes(b"synthetic clip")
    image = layer_project / "picture.jpg"
    image.write_bytes(b"synthetic image")
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    picture = ResolvedElement("agent-insert", kind, "agent", 16, 40, "front", {
        "media": {"path": clip.name, "in_s": 2, "speed": 2},
    })
    still = ResolvedElement("agent-still", kind, "agent", 50, 80, "front", {
        "media": {"path": image.name},
    })
    overlay = replace(picture, kind="overlay", start_frame=0, end_frame=16)
    timeline = replace(timeline, elements=(*timeline.elements, picture, still, overlay))
    facts = project_facts(context, timeline, timeline.elements[0])
    assert facts.shots[0].source == "cut/cut.mp4"
    if kind == "background":
        assert all(shot.source == "cut/cut.mp4" for shot in facts.shots)
        assert [shot.source_frame for shot in facts.shots] == [15, 33, 63]
        return
    assert facts.shots[1].source == clip.name
    assert facts.shots[1].source_time == "47/15"
    assert facts.shots[2].source == image.name
    assert facts.shots[2].source_time is None and facts.shots[2].source_frame is None


def test_shot_cache_reuses_identity_and_converts_hdr(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic HDR source")
    shot = CardShot(0, 90, "01", source.name, receipt.sha256_file(source), source_frame=45)
    info = SimpleNamespace(video_frames=90, fps=Fraction(30), rotation=90,
                           color_transfer="arib-std-b67")
    conversions, commands = [], []
    monkeypatch.setattr(ffmpeg, "probe", lambda source: info)

    def conversion(value):
        conversions.append(value)
        return hdr.HdrConversion("test", ["-hwaccel", "test"], "hdr_to_bt709", rotation=90)

    def run(args, **kwargs):
        commands.append(args)
        Path(args[-1]).write_bytes(b"synthetic jpeg")

    monkeypatch.setattr(hdr, "conversion", conversion)
    monkeypatch.setattr(ffmpeg, "run", run)
    first = cached_still(tmp_path, shot, width=198, height=352)
    assert first.is_relative_to(tmp_path / ".yap")
    assert cached_frame(tmp_path, shot.source, shot.source_sha256,
                         source_frame=45, width=198, height=352) == first
    assert cached_still(tmp_path, shot, width=198, height=352) == first
    assert len(commands) == 1
    assert len(conversions) == 3
    assert commands[0][commands[0].index("-ss") + 1] == "1.5"
    assert "hdr_to_bt709" in commands[0][commands[0].index("-vf") + 1]
    assert "-hwaccel" in commands[0]
    assert cached_still(tmp_path, replace(shot, source_frame=60), width=198, height=352) != first
    source.write_bytes(b"changed source")
    with pytest.raises(ValidationError, match="run graphics again"):
        cached_still(tmp_path, shot, width=198, height=352)


def shots(start: int = 250) -> CardProject:
    return CardProject(30, start, 900, (), tuple(
        CardShot(index * 100, (index + 1) * 100, f"{index + 1:02}") for index in range(9)
    ))


def test_shot_cache_falls_back_when_hdr_conversion_is_unsupported(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic HDR source")
    shot = CardShot(0, 90, "01", source.name, receipt.sha256_file(source), source_frame=45)
    monkeypatch.setattr(ffmpeg, "probe", lambda source: SimpleNamespace(
        rotation=0, fps=Fraction(30), video_frames=90,
    ))

    def unsupported(info):
        raise hdr.HdrUnsupportedError()

    commands = []

    def run(args, **kwargs):
        commands.append(args)
        Path(args[-1]).write_bytes(b"synthetic jpeg")

    monkeypatch.setattr(hdr, "conversion", unsupported)
    monkeypatch.setattr(ffmpeg, "run", run)
    assert cached_still(tmp_path, shot, width=198, height=352).is_file()
    assert len(commands) == 1
    assert commands[0][commands[0].index("-i") + 1] == str(source)
    assert "-hwaccel" not in commands[0]


@pytest.mark.parametrize("unsupported", [False, True])
def test_still_cache_identity_tracks_hdr_conversion(
    tmp_path: Path, monkeypatch, unsupported: bool,
) -> None:
    from yap_studio import card_stills

    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic HDR source")
    digest = receipt.sha256_file(source)
    selected = None
    extractions = []
    monkeypatch.setattr(ffmpeg, "probe", lambda source: SimpleNamespace())

    def conversion(info):
        if selected is None and unsupported:
            raise hdr.HdrUnsupportedError()
        return selected

    def extract(source, info, out, **kwargs):
        extractions.append(kwargs["conversion"])
        out.write_bytes(b"converted" if kwargs["conversion"] else b"plain")

    monkeypatch.setattr(hdr, "conversion", conversion)
    monkeypatch.setattr(card_stills, "extract_frame", extract)
    plain = cached_frame(tmp_path, source, digest, source_frame=30, width=198, height=352)
    selected = hdr.HdrConversion("test", [], "hdr_to_bt709")
    converted = cached_frame(tmp_path, source, digest, source_frame=30, width=198, height=352)
    assert plain != converted
    assert plain.read_bytes() == b"plain" and converted.read_bytes() == b"converted"
    assert cached_frame(tmp_path, source, digest, source_frame=30,
                         width=198, height=352) == converted
    selected = replace(selected, filter_prefix="different_hdr_to_bt709")
    changed = cached_frame(tmp_path, source, digest, source_frame=30, width=198, height=352)
    assert changed not in {plain, converted}
    assert len(extractions) == 3


def test_image_still_cache_preserves_thumbnail_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=red:s=90x160", "-frames:v", "1", "-threads", "1", str(source)],
               timeout_s=30)
    shot = CardShot(0, 90, "01", source.name, receipt.sha256_file(source))
    output = cached_still(tmp_path, shot, width=198, height=352)
    info = ffmpeg.probe(output)
    assert (info.width, info.height) == (198, 352)
    assert cached_still(tmp_path, shot, width=198, height=352) == output


def test_storyboard_selects_neighbours_and_exact_changes() -> None:
    plan = cards.card_layer_plan("storyboard", {}, resolve("preset:clean_minimal"),
                                 frames=90, project=shots())
    thumbnails = json.loads(plan.variables["thumbnails"])
    assert [item["label"] for item in thumbnails] == ["01", "02", "03", "04", "05"]
    assert all(item["src"] == "" for item in thumbnails)
    assert json.loads(plan.variables["changes"]) == [
        {"frame": 0, "at_s": 0, "index": 2},
        {"frame": 50, "at_s": 50 / 30, "index": 3},
    ]
    assert plan.variables["thumb_width_px"] / plan.variables["thumb_height_px"] == 9 / 16
    assert plan.variables["number_px"] == 30
    ring = json.loads(plan.variables["ring_keyframes"])
    assert ring[50]["at_s"] == 49.5 / 30
    assert ring[50]["x_px"] == ring[49]["x_px"]
    assert ring[56]["x_px"] == ring[-1]["x_px"]
    fewer = replace(shots(0), shots=shots(0).shots[:2], total_frames=200)
    assert len(json.loads(cards.card_layer_plan(
        "storyboard", {}, resolve("preset:clean_minimal"), project=fewer,
    ).variables["thumbnails"])) == 2


def test_storyboard_fit_minimum_and_sample_facts() -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_width_px"] = 450
    with pytest.raises(ValidationError, match="lower count to 3") as caught:
        cards.card_layer_plan("storyboard", {}, style)
    assert caught.value.parameter == "slots.count"
    with pytest.raises(ValidationError, match="at least 1.5 s"):
        cards.card_layer_plan("storyboard", {}, style, frames=44)
    assert len(sample_project("storyboard", 90, 30).shots) == 9


@pytest.mark.parametrize("numbers,thumb_height", [("show", 176), ("hide", 208)])
def test_storyboard_panel_hugs_the_row(numbers: str, thumb_height: int) -> None:
    plan = cards.card_layer_plan("storyboard", {"numbers": numbers},
                                 resolve("preset:clean_minimal"), frames=90, project=shots())
    v = plan.variables
    thumbnails = json.loads(v["thumbnails"])
    row_width = 5 * (v["thumb_width_px"] + 12) + 4 * 16
    assert v["storyboard_width_px"] == row_width + 2 * 20
    assert thumbnails[0]["x_px"] == 20 + 6
    assert thumbnails[-1]["x_px"] + v["thumb_width_px"] + 6 + 20 == (
        v["storyboard_width_px"]
    )
    assert v["thumb_height_px"] == thumb_height
    assert v["panel_height_px"] == 272
    assert (plan.width, plan.height) == (1040, 360)


def test_storyboard_native_motion_uses_the_planned_boundary(tmp_path: Path) -> None:
    layer = cards.card_layer_plan("storyboard", {}, resolve("preset:clean_minimal"),
                                  frames=90, project=shots())
    source = (cards.template_dir("storyboard") / "template.html").read_text()
    payload = tmp_path / "motion.json"
    payload.write_text(json.dumps({
        "variables": layer.variables,
        "script": re.findall(r"<script>(.*?)</script>", source, re.S)[0],
    }))
    runner = tmp_path / "motion.cjs"
    runner.write_text("""
const input = JSON.parse(require("node:fs").readFileSync(process.argv[2], "utf8"));
const events = [];
class Element {
  constructor(name) {
    this.name = name; this.style = {setProperty() {}}; this.dataset = {duration: "2.999"};
  }
  querySelector(name) {return elements[name];}
  appendChild(child) {}
  animate(frames, options) {events.push({name: this.className || this.name, frames, options});}
}
const elements = Object.fromEntries(["root", ".panel", ".ring"].map(n => [n, new Element(n)]));
global.document = {getElementById(n) {return elements[n];},
  createElement(n) {return new Element(n);}};
global.window = {__hyperframes: {getVariables() {return input.variables;}}};
new Function(input.script)();
process.stdout.write(JSON.stringify(events));
""")
    events = json.loads(tools.run_tool("node", [str(runner), str(payload)], timeout_s=10).stdout)
    entrances = [event for event in events if event["name"] == "thumbnail"]
    assert [event["options"]["delay"] for event in entrances] == [100, 150, 200, 250, 300]
    assert all(event["frames"][0]["opacity"] == 0 for event in entrances)
    ring = next(event for event in events if event["name"] == ".ring"
                and "offset" in event["frames"][0])
    assert ring["frames"][50]["offset"] == pytest.approx(49.5 / 90)
    assert ring["frames"][56]["offset"] == pytest.approx(55.5 / 90)
    assert all(frame["easing"] == "steps(1, end)" for frame in ring["frames"])
    assert ring["options"]["easing"] == "linear"
    scales = [event for event in events if event["name"] == "picture"]
    assert float(scales[2]["frames"][0]["width"].removesuffix("px")) == (
        pytest.approx(layer.variables["thumb_width_px"] * 1.06)
    )
    assert float(scales[3]["frames"][56]["width"].removesuffix("px")) == (
        pytest.approx(layer.variables["thumb_width_px"] * 1.06)
    )
    numbers = [event for event in events if event["name"] == "number"]
    accent, ink = layer.variables["accent_color"], layer.variables["ink_color"]
    assert numbers[2]["frames"][49]["color"] == accent
    assert numbers[2]["frames"][50]["color"] == ink
    assert numbers[3]["frames"][49]["color"] == ink
    assert numbers[3]["frames"][50]["color"] == accent
    assert numbers[3]["frames"][50]["offset"] == pytest.approx(49.5 / 90)
