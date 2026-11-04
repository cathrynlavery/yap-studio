"""Shared transition geometry, unchanged portrait HTML and wide picture coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import synth
from test_assemble_picture import project, write_document
from test_cut import make_project, time_segment, write_edit, write_json
from test_landscape import _context
from test_timeline import add, transition, transition_story
from test_timeline import project as _timeline_project
from test_transitions import LOOKS, timeline

from yap_studio import ffmpeg, paths
from yap_studio.canvas import canvas, require_supported
from yap_studio.faces import detect
from yap_studio.project.model import Anchor, Element, TimelineLane
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import ProjectContext, cut_inputs, ingest_inputs
from yap_studio.render import hyperframes
from yap_studio.stages import assemble_picture as picture
from yap_studio.stages import transitions as stage
from yap_studio.stages.assemble import assemble
from yap_studio.stages.captions import captions
from yap_studio.stages.cut import cut
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline, resolve_timeline
from yap_studio.verify import verify

timeline_project = _timeline_project


def test_horizontal_accepts_transition_elements(tmp_path: Path) -> None:
    context = _context(tmp_path)
    require_supported(context, {"agent": TimelineLane((Element(
        "agent-transition", "transition", Anchor(word="t1:2"), fields={"look": "whip_pan"},
    ),), lane="agent")})


@pytest.mark.parametrize("shape,look,frames,expected", [
    ("horizontal", "whip_pan", None, 8),
    ("horizontal", "light_leak", None, 12),
    ("horizontal", "cinematic_zoom", None, 12),
    ("vertical", "whip_pan", None, 12),
    ("horizontal", "whip_pan", 16, 16),
])
def test_transition_frame_window(timeline_project: Path, shape: str, look: str,
                                 frames: int | None, expected: int) -> None:
    data = json.loads((timeline_project / "project.json").read_bytes())
    write_json(timeline_project / "project.json", {**data, "shape": shape})
    transition_story(timeline_project)
    element = transition(look=look)
    if frames is not None:
        element["frames"] = frames
    item = add(timeline_project, element, lane="agent").require_valid().elements[0]
    assert item.payload["frames"] == expected
    join = item.payload["join_frame"]
    assert (item.start_frame, item.end_frame) == (join - expected // 2, join + expected // 2)


@pytest.mark.parametrize("look", LOOKS)
def test_template_size_literals_are_width_height_pairs(look: str) -> None:
    source = stage.template_file(look).read_text(encoding="utf-8")
    literals = stage.SIZE_LITERALS.findall(source)
    assert literals == ["1080", "1920"] * 8


@pytest.mark.parametrize("look", LOOKS)
def test_horizontal_transition_html_uses_the_wide_canvas(tmp_path: Path, monkeypatch,
                                                       look: str) -> None:
    write_json(tmp_path / "project.json", {"shape": "horizontal"})
    source = synth.picture_source(tmp_path / "cut.mov", "color=c=red:s=1920x1080:r=30,"
                                  "drawbox=c=blue:t=fill:enable='gte(n,18)'", 36)
    recorded = []

    def fake(composition: Path, out: Path, **options) -> None:
        assert hyperframes.preflight(composition, 12) == 0.399
        recorded.append((composition / "index.html").read_bytes())
        synth.picture_source(out, "color=c=red:s=1920x1080:r=30", options["frames"])

    monkeypatch.setattr(hyperframes, "render_layer", fake)
    placements = stage.render_transitions(tmp_path, timeline(look), source, tmp_path / "work")
    cached = stage.render_transitions(tmp_path, timeline(look), source, tmp_path / "work")
    assert cached == placements
    assert len(recorded) == 1
    html = recorded[0].decode("utf-8")
    for fragment in ('data-width="1920" data-height="1080"', 'width: 1920px; height: 1080px',
                     'canvas.width = 1920; canvas.height = 1080;', 'gl.viewport(0, 0, 1920, 1080)',
                     "'u_resolution'), 1920, 1080", 'must be 1920x1080',
                     'naturalWidth !== 1920 || image.naturalHeight !== 1080'):
        assert fragment in html
    assert '1080px; height: 1920px' not in html
    expected = stage.template_file(look).read_text(encoding="utf-8").replace(
        'data-duration="0.399"', f'data-duration="{hyperframes.layer_duration_s(12, 30)}"',
    )
    expected = stage.SIZE_LITERALS.sub(
        lambda match: {"1080": "1920", "1920": "1080"}[match[0]], expected,
    )
    assert recorded[0] == expected.encode("utf-8")


@pytest.mark.parametrize("look", LOOKS)
def test_vertical_transition_html_is_the_template(tmp_path: Path, monkeypatch, look: str) -> None:
    source = synth.picture_source(tmp_path / "cut.mov", "color=c=red:s=1080x1920:r=30", 36)
    recorded = []

    def fake(composition: Path, out: Path, **options) -> None:
        assert hyperframes.preflight(composition, 12) == 0.399
        recorded.append((composition / "index.html").read_bytes())
        synth.picture_source(out, "color=c=red:s=1080x1920:r=30", options["frames"])

    monkeypatch.setattr(hyperframes, "render_layer", fake)
    stage.render_transitions(tmp_path, timeline(look), source, tmp_path / "work")
    expected = stage.template_file(look).read_bytes().replace(
        b'data-duration="0.399"',
        f'data-duration="{hyperframes.layer_duration_s(12, 30)}"'.encode(),
    )
    assert recorded == [expected]


def test_horizontal_transition_refuses_a_portrait_movie(tmp_path: Path, monkeypatch) -> None:
    write_json(tmp_path / "project.json", {"shape": "horizontal"})
    source = synth.picture_source(tmp_path / "cut.mov", "color=c=red:s=1920x1080:r=30", 36)

    def fake(composition: Path, out: Path, **options) -> None:
        synth.picture_source(out, "color=c=red:s=1080x1920:r=30", options["frames"])

    monkeypatch.setattr(hyperframes, "render_layer", fake)
    with pytest.raises(stage.TransitionError, match="1920x1080"):
        stage.render_transitions(tmp_path, timeline(), source, tmp_path / "work")
    portrait = synth.picture_source(tmp_path / "portrait.mov", "color=c=red:s=1080x1920:r=30", 36)
    with pytest.raises(stage.TransitionError, match="Verified cut dimensions"):
        stage.render_transitions(tmp_path, timeline(), portrait, tmp_path / "work")


def test_horizontal_transition_layer_fills_the_wide_picture(tmp_path: Path) -> None:
    pictures = {
        "cut": synth.picture_source(tmp_path / "cut.mov", "color=c=blue:s=1920x1080:r=30", 90),
        "take": synth.picture_source(tmp_path / "take.mov", "color=c=blue:s=320x180:r=30", 90),
    }
    root = project(tmp_path / "project", pictures, hold_frames=0)
    data = json.loads((root / "project.json").read_bytes())
    write_document(root / "project.json", {**data, "shape": "horizontal"})
    saved = read_receipt(root / "cut")
    with StageRun(root, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in saved["outputs"]})
        run.finish(extra={"plan": saved["plan"]})
    movie = synth.picture_source(root / "transition.mov", "color=c=red:s=1920x1080:r=30", 12)
    transition = ResolvedElement("agent-transition", "transition", "agent", 36, 48, "front",
                                 {"look": "whip_pan", "frames": 12, "join_frame": 42})
    resolved = ResolvedTimeline((transition,), (), (), 90, 0)
    placement = {"id": transition.id, "path": str(movie), "start_frame": 36, "frames": 12}
    out = root / "work/picture.mp4"
    with paths.operation():
        picture.render_picture(root, out, work_dir=root / "work", timeline=resolved,
                               title_layers=[], caption_layer=None, frame=canvas("horizontal"),
                               transition_layers=[placement])
    for index in range(35, 49):
        pixels = synth.picture_pixels(out, index)
        expected = (0, 0, 255) if index in (35, 48) else (255, 0, 0)
        for x in (8, 948, 1888):
            np.testing.assert_allclose(pixels[300:324, x:x + 24].mean(axis=(0, 1)), expected,
                                       atol=8)


@pytest.mark.render
@pytest.mark.parametrize("look", LOOKS)
def test_real_horizontal_transition_fills_wide_frame(tmp_path: Path, look: str) -> None:
    write_json(tmp_path / "project.json", {"shape": "horizontal"})
    source = synth.picture_source(tmp_path / "cut.mov", "color=c=red:s=1920x1080:r=30,"
                                  "drawbox=c=blue:t=fill:enable='gte(n,7)'", 14)
    placements = stage.render_transitions(tmp_path, timeline(look, join=7, total=14), source,
                                          tmp_path / "work")
    movie = Path(placements[0]["path"])
    info = ffmpeg.probe(movie)
    assert (info.width, info.height, info.video_frames) == (1920, 1080, 12)
    for index in (0, 11):
        pixels = synth.picture_pixels(movie, index)
        for y in (200, 840):
            for x in (0, 944, 1888):
                mean = pixels[y:y + 32, x:x + 32].mean(axis=(0, 1))
                print(f"{look} frame={index} x={x} y={y} RGB={mean.tolist()}")
                if index == 0:
                    assert mean[0] > 180 and mean[0] > 3 * mean[2]
                else:
                    assert mean[2] > 180 and mean[2] > 3 * mean[0]


@pytest.mark.render
@pytest.mark.parametrize("look", LOOKS)
def test_real_horizontal_project_assembles_each_transition(tmp_path: Path, monkeypatch,
                                                         look: str) -> None:
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "Synthetic fixture")
    source = synth.loudness_master(tmp_path / "source.mov", frames=90,
        picture_filter="scale=320:180,drawbox=c=red@0.85:t=fill:enable='lt(t,1.5)',"
                       "drawbox=c=blue@0.85:t=fill:enable='gte(t,1.5)'")
    root = make_project(tmp_path / "project", [source], word_rows=[[
        ("Red", .2, .5), ("ends.", .6, 1.2), ("Blue", 1.7, 2.0),
        ("starts", 2.1, 2.5), ("here.", 2.55, 2.9),
    ]])
    data = json.loads((root / "project.json").read_bytes())
    data.update(shape="horizontal", platform="youtube", asr={"engine": "none"},
                music={"mode": "none"}, audio={"mode": "preserve"}, style="style.json")
    write_json(root / "style.json", {"schema_version": 1, "name": "Render fixture",
                                    "music": {"sound_effects": "none"},
                                    "captions": {"size_px": 64}})
    write_json(root / "project.json", data)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish()
    write_edit(root, [time_segment("s1", "t1", 0, 1.4), time_segment("s2", "t1", 1.6, 3)],
               end_hold_s=.3)
    cut(root)
    words = json.loads((root / "cut/cut-words.json").read_bytes())
    write_json(root / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": words["words_sha256"],
        "elements": [{"id": "agent-transition", "kind": "transition", "lane": "agent",
                      "start": {"word": "t1:2", "segment": "s2"}, "look": look, "frames": 12}],
    })
    captions(root)
    graphics(root)
    music(root)
    result = assemble(root)
    assert result["status"] == "verified", result
    final = root / "final/story.mp4"
    info = ffmpeg.probe(final)
    assert (info.width, info.height) == (1920, 1080)
    report = verify(str(final), edl=str(root / "final/verify-edl.json"), asr="none",
                    detail="full", platform="youtube", loudness="note")
    assert report["status"] == "pass", report["findings"]
    # Only assembly's own verify receives the declared transition windows.
    published = read_receipt(root / "final")
    assert published["verify"]["checks"]["picture.transition"]["ran"]
    assert published["verify"]["checks"]["picture.transition"]["status"] == "pass"
    assert published["transitions"][0]["look"] == look
    resolved = resolve_timeline(root).require_valid()
    window = next(item for item in resolved.elements if item.kind == "transition")
    join = window.payload["join_frame"]
    for index in (window.start_frame - 1, join - 1, join, window.end_frame):
        pixels = synth.picture_pixels(final, index)
        for x in (8, 944, 1880):
            mean = pixels[200:232, x:x + 32].mean(axis=(0, 1))
            print(f"{look} frame={index} x={x} y=200 RGB={mean.tolist()}")
            if index == window.start_frame - 1:
                assert mean[0] > 180 and mean[0] > 3 * mean[2]
            elif index == window.end_frame:
                assert mean[2] > 180 and mean[2] > 3 * mean[0]
            else:
                assert mean[0] > 70 and mean[2] > 70
