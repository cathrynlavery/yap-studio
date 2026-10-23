"""Horizontal phase 1 contracts, real delivery, and frozen vertical artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import synth
from test_cut import make_project, time_segment, write_edit, write_json

from yap_studio import cli, ffmpeg, paths, receipt
from yap_studio.canvas import canvas, require_supported
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.errors import ValidationError, YapError
from yap_studio.faces import detect
from yap_studio.project.model import ProjectFileError, json_schema, parse_document
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import ProjectContext, cut_inputs, ingest_inputs
from yap_studio.stages.assemble import assemble, assembly_snapshot, verification_edl
from yap_studio.stages.captions import (
    caption_layer_plan,
    caption_layout,
    captions,
    prepare_layer_plan,
)
from yap_studio.stages.cut import cut
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.style import resolve
from yap_studio.verify import verify


def project_data(shape: str | None = None) -> dict:
    data = {"schema": "yap-studio/project@1", "name": "sample",
            "takes": [{"id": "t1", "path": "source.mov"}]}
    if shape is not None:
        data["shape"] = shape
    return data


def test_shape_default_validation_and_canvas() -> None:
    default = parse_document("project", project_data())
    vertical = parse_document("project", project_data("vertical"))
    assert default.shape == "vertical" and default.to_dict() == vertical.to_dict()
    assert "shape" not in default.to_dict()
    horizontal = parse_document("project", project_data("horizontal"))
    assert horizontal.to_dict()["shape"] == "horizontal"
    jsonschema.validate(horizontal.to_dict(), json_schema("project"))
    for invalid in ("square", "16:9", None, 1):
        with pytest.raises(ProjectFileError):
            parse_document("project", {**project_data(), "shape": invalid})
    assert (canvas().width, canvas().height, canvas().caption_width) == (1080, 1920, 940)
    wide = canvas("horizontal")
    assert (wide.width, wide.height, wide.caption_width, wide.caption_inset) == (1920, 1080,
                                                                             1600, 160)
    with pytest.raises(ValidationError, match="shape"):
        canvas("square")


@pytest.mark.parametrize("context", [SimpleNamespace(),
                                   SimpleNamespace(project=SimpleNamespace()),
                                   SimpleNamespace(project=SimpleNamespace(shape="vertical"))])
def test_shape_guard_defaults_to_vertical_without_project_reads(context,
                                                               monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("Vertical shape guard must not read project files")

    monkeypatch.setattr(Path, "read_bytes", unexpected)
    require_supported(context)


def test_init_shape_flag(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    source = synth.calib_source(tmp_path / "source.mov", seconds=1, size="320x180")
    root = tmp_path / "project"
    assert cli.main(["init", str(root), "--take", str(source), "--shape", "horizontal",
                     "--video-type", "talking_head", "--broll", "none", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["shape"] == "horizontal"
    assert json.loads((root / "project.json").read_bytes())["shape"] == "horizontal"


def _context(root: Path, *, shape: str = "horizontal", style: dict | None = None,
             **settings) -> ProjectContext:
    write_json(root / "project.json", {**project_data(shape), "style": "style.json", **settings})
    write_json(root / "style.json", {"schema_version": 1, "name": "Sample", **(style or {})})
    return ProjectContext(root)


def test_horizontal_init_refuses_voiceover_before_creating_standin(tmp_path: Path) -> None:
    from yap_studio.project import init_project

    root = tmp_path / "project"
    with pytest.raises(YapError, match="voiceover takes"):
        init_project(root, takes=[], voiceovers=[tmp_path / "memo.wav"], shape="horizontal")
    assert not root.exists()


@pytest.mark.parametrize("settings,style,feature", [
    ({"platform": "shorts"}, {}, "platform"),
    ({"video_type": "green_screen"}, {}, "video type"),
    ({"grade": {"look": "film"}}, {}, "grading"),
    ({}, {"moves": {"jump_cuts": "smooth"}}, "joins"),
    ({}, {"moves": {"opening_push": True}}, "zoom"),
    ({}, {"captions": {"camera_3d": True}}, "3D"),
    ({}, {"hook": {"style": "text_hook"}}, "text hooks"),
])
def test_unsupported_settings_fail_before_media(tmp_path: Path, settings: dict,
                                                 style: dict, feature: str) -> None:
    context = _context(tmp_path, style=style, **settings)
    with pytest.raises(YapError, match=feature) as caught:
        require_supported(context)
    assert caught.value.code == "horizontal_unsupported"
    assert caught.value.suggested_action
    assert not (tmp_path / "cut").exists()


@pytest.mark.parametrize("kind", ["title", "chapter", "graphic", "insert", "overlay",
                                   "background", "zoom", "transition", "moment"])
def test_visual_lanes_refused_even_with_rule_override(tmp_path: Path, kind: str) -> None:
    from yap_studio.project.model import Anchor, Element, TimelineLane
    from yap_studio.stages.timeline import _resolve

    context = _context(tmp_path)
    lanes = {"agent": TimelineLane((Element("visual", kind, Anchor(time=0)),), lane="agent")}
    with pytest.raises(YapError, match=kind) as caught:
        _resolve(context, lanes)
    assert "Remove visual" in caught.value.suggested_action
    assert not (tmp_path / "cut").exists()


def _phrases(style):
    return build_phrases([CaptionWord("t1:0", "Keep", .1, .3),
                          CaptionWord("t1:1", "this", .3, .5),
                          CaptionWord("t1:2", "vertical.", .5, .8)], style.data["captions"])


def test_vertical_caption_plan_and_html_match_baseline_bytes(tmp_path: Path) -> None:
    """Golden hashes were measured from public main 16fa27a's planner and prepared HTML."""
    style = resolve("preset:clean_minimal")
    plans = [caption_layer_plan(_phrases(style), style, frames=30, **options)
             for options in ({}, {"shape": "vertical"})]
    assert plans[0] == plans[1]
    data = asdict(plans[0])
    data.pop("shape")
    assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == (
        "e4a685b46b5e3547e259b2f46418a1b17915ce78f47c95eba91e1d82c09576d3")
    with paths.operation():
        composition = prepare_layer_plan(plans[0], tmp_path)
    assert receipt.sha256_file(composition / "index.html") == (
        "3b2e2e270630a65c7e4712399f34018b30101a5fe873c619065b9b4a9e42304c")


def test_horizontal_caption_layout_safe_area_and_wider_lines(tmp_path: Path) -> None:
    context = _context(tmp_path)
    style = context.render_style()
    settings = style.data["captions"]
    assert settings["position"] == "bottom" and settings["placement"] == "fixed"
    words = [CaptionWord(f"t1:{i}", text, .1 + i * .1, .2 + i * .1)
             for i, text in enumerate(("Horizontal", "captions", "have", "room", "to", "breathe"))]
    phrases = build_phrases(words, settings)
    plan = caption_layer_plan(phrases, style, frames=30, shape="horizontal")
    direct = caption_layer_plan(phrases, resolve("preset:clean_minimal"), frames=30,
                                shape="horizontal")
    assert direct.y == plan.y and direct.height == plan.height
    vertical = caption_layer_plan(phrases, resolve("preset:clean_minimal"), frames=30)
    assert plan.width == 1920 and plan.y > 600
    assert plan.y + plan.height <= 1008
    assert len(json.loads(plan.variables["phrases"])[0].get("breaks", [])) <= len(
        json.loads(vertical.variables["phrases"])[0].get("breaks", []))
    layout = caption_layout(None, plan)
    assert (layout["width"], layout["height"]) == (1920, 1080)
    with paths.operation():
        composition = prepare_layer_plan(plan, tmp_path / "html")
    html = (composition / "index.html").read_text()
    assert 'data-width="1920"' in html and "max-width: 1600px" in html
    assert "left: 160px; right: 160px" in html
    assert "Math.min(1600," in html


def test_shape_change_invalidates_cut_and_portrait_source_is_refused(tmp_path: Path) -> None:
    source = synth.loudness_master(tmp_path / "portrait.mov", frames=30)
    root = make_project(tmp_path / "project", [source], word_rows=[[("Hello.", .1, .8)]])
    data = json.loads((root / "project.json").read_bytes())
    data["asr"] = {"engine": "none"}
    write_json(root / "project.json", data)
    write_edit(root, [time_segment("s1", "t1", 0, 1)])
    vertical_inputs = cut_inputs(ProjectContext(root))
    data["shape"] = "horizontal"
    write_json(root / "project.json", data)
    horizontal_inputs = cut_inputs(ProjectContext(root))
    assert "project:shape" not in vertical_inputs
    assert "project:shape" in horizontal_inputs and horizontal_inputs != vertical_inputs
    with pytest.raises(YapError, match="non-horizontal take"):
        cut(root)
    assert not (root / "cut/cut.mp4").exists()


def test_horizontal_covers_and_studio_export_refuse(tmp_path: Path,
                                                  capsys: pytest.CaptureFixture) -> None:
    from yap_studio.studio.export import export_project

    _context(tmp_path)
    assert cli.main(["cover", str(tmp_path), "--headline", "Horizontal", "--json"]) != 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "horizontal_unsupported" in output and "covers" in output
    with pytest.raises(YapError, match="studio export"):
        export_project(tmp_path)
    assert not (tmp_path / "covers").exists()


def test_horizontal_applied_taste_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    context = _context(tmp_path)
    monkeypatch.setattr(context, "taste_application", lambda: SimpleNamespace(applied=("rule",)))
    with pytest.raises(YapError, match="taste preferences"):
        require_supported(context)


def _render_project(tmp_path: Path, shape: str) -> Path:
    source = synth.loudness_master(tmp_path / "source.mov", frames=90,
                                   picture_filter="scale=320:180")
    root = make_project(tmp_path / "project", [source],
                        word_rows=[[("Readable", .2, .6), ("horizontal", .7, 1.2),
                                    ("captions", 1.3, 1.8), ("work.", 1.9, 2.4)]])
    data = json.loads((root / "project.json").read_bytes())
    data.update(shape=shape, asr={"engine": "none"}, music={"mode": "none"},
                audio={"mode": "preserve"}, platform="youtube" if shape == "horizontal" else "none",
                style="style.json")
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
    write_edit(root, [time_segment("s1", "t1", 0, 3)], end_hold_s=.3)
    cut(root)
    captions(root)
    graphics(root)
    music(root)
    return root


@pytest.mark.render
def test_real_horizontal_cut_captions_assemble_and_verify(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "Synthetic fixture")
    root = _render_project(tmp_path, "horizontal")
    result = assemble(root)
    assert result["status"] == "verified", result
    final = root / "final/story.mp4"
    info = ffmpeg.probe(final)
    assert (info.width, info.height, info.video_frames, info.fps) == (1920, 1080, 99, 30)
    report = verify(str(final), edl=str(root / "final/verify-edl.json"), asr="none",
                    detail="full", platform="youtube", loudness="note")
    assert report["status"] == "pass", report["findings"]
    assert report["checks"]["platform.shape"]["status"] == "pass"
    layout = json.loads((root / "captions/layout.json").read_bytes())
    assert layout["measured"] and layout["captions"]
    for caption in layout["captions"]:
        for box in caption["boxes"]:
            assert 150 <= box["x"] < box["x"] + box["w"] <= 1770
            assert 600 <= box["y"] < box["y"] + box["h"] <= 1008
    edl = verification_edl(assembly_snapshot(ProjectContext(root)), base=root)
    assert any(item["region"][2] == 1920 for item in edl["overlays"])
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", "1", "-i", str(final),
                "-frames:v", "1", str(root / "horizontal-frame.png")], timeout_s=30)
    print(f"Horizontal artifact: {final}")


@pytest.mark.render
def test_explicit_vertical_keeps_render_bytes_and_inputs(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "Synthetic fixture")
    root = _render_project(tmp_path, "vertical")
    assemble(root)
    names = ("cut/cut.mp4", "captions/captions.mov", "final/story.mp4")
    before = {name: receipt.sha256_file(root / name) for name in names}
    inputs = cut_inputs(ProjectContext(root))
    data = json.loads((root / "project.json").read_bytes())
    data.pop("shape")
    write_json(root / "project.json", data)
    assert cut_inputs(ProjectContext(root)) == inputs
    assert cut(root)["noop"]
    cut(root, force=True)
    captions(root, force=True)
    assemble(root, force=True)
    assert {name: receipt.sha256_file(root / name) for name in names} == before
