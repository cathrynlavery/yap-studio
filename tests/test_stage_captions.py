"""Synthetic project caption layers, narrow projections and retained render inputs."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import synth
from test_captions_render import _alpha_frames
from test_cli_captions import clip as clip
from test_cli_captions import fake_layers as fake_layers
from test_cli_captions import words as words
from test_render import _FAKE_RENDERER
from test_timeline import write

from yap_studio import ffmpeg, receipt
from yap_studio.brand_words import set_words
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.errors import OutputPathError, RendererBlockedError
from yap_studio.faces import detect
from yap_studio.faces import project as face_project
from yap_studio.faces.clearance import LAYOUT_SCHEMA, Layout, face_clearance_px, gap_px, head_box
from yap_studio.faces.detect import DetectorRun, find_gaps, sample_frames
from yap_studio.faces.mapping import map_cut_faces
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.faces.project import cut_faces, cut_faces_identity
from yap_studio.plan import snap_frame
from yap_studio.project.receipts import StageRun, projection_sha256, read_receipt
from yap_studio.project.status import (
    ProjectContext,
    captions_inputs,
    cut_inputs,
    fillers_inputs,
    ingest_inputs,
    status,
)
from yap_studio.render import hyperframes, layer_checks
from yap_studio.stages import assemble as assemble_stage
from yap_studio.stages import captions as caption_stage
from yap_studio.stages import green_screen
from yap_studio.stages.captions import (
    LayerPlan,
    caption_band_geometry,
    caption_layout,
    caption_offsets,
    caption_shrink,
    caption_snapshot,
    captions,
)
from yap_studio.stages.graphics import graphics_snapshot
from yap_studio.stages.timeline import ResolvedElement, TimelineRefusal
from yap_studio.style import ResolvedStyle, resolve, resolve_style_data


class _EmptyCutDetector:
    """Supply empty detections for synthetic projects used without the helper fixture."""

    def detect(self, video: Path, *, every: int) -> DetectorRun:
        info = ffmpeg.probe(video)
        return DetectorRun(info.width, info.height, info.video_frames,
                           {frame: () for frame in sample_frames(info.video_frames, every)},
                           {"name": "fake-empty-cut"})


def make_project(root: Path, *, trailing_picture: bool = False) -> Path:
    """Publish two synthetic phrases, optionally leaving picture after their natural tail."""
    source = synth.calib_source(root / "source.mov", seconds=3, size="160x90")
    write(root / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample", "style": "style.json",
        "asr": {"engine": "none"}, "takes": [{"id": "t1", "path": source.name}],
        "word_fixes": {"small": "simple"},
    })
    write(root / "style.json", {
        "schema_version": 1, "name": "Sample style",
        "captions": {"size_px": 48, "highlight_style": "underline"},
        "hook": {"style": "quiet_serif", "duration_s": 1.1, "size_px": 48},
    })
    inputs = ingest_inputs(ProjectContext(root))
    offset = 0 if trailing_picture else 0.5
    words = [{"id": f"t1:{i}", "text": text, "start": start, "end": end}
             for i, (text, start, end) in enumerate([
                 ("Small", 0.1, 0.2), ("steps.", 0.3, 0.45),
                 ("More", 1.0 + offset, 1.1 + offset),
                 ("words.", 1.2 + offset, 1.4 + offset),
             ])]
    staged = write(root / "staged-words.json", {
        "schema": "yap-studio/words@1", "take": "t1", "take_sha256": inputs["take:t1"],
        "time_base": "source", "audio_offset_s": 0, "engine": "none", "model": None,
        "transcript_sha256": "1" * 64, "words": words,
    })
    with StageRun(root, "ingest") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "ingest/t1/words.json", staged)
        run.finish()
    hashes = {"t1": receipt.sha256_file(root / "ingest/t1/words.json")}
    write(root / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": hashes, "end_hold_s": 1.5,
        "segments": [{"id": "s1", "take": "t1", "from": {"word": "t1:0"},
                      "to": {"word": "t1:3"}}],
    })
    inputs = fillers_inputs(ProjectContext(root))
    staged = write(root / "staged-removals.json", {
        "schema": "yap-studio/removals@1", "removals": [],
    })
    with StageRun(root, "fillers") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "removals.json", staged)
        run.finish()
    inputs = cut_inputs(ProjectContext(root))
    staged = write(root / "staged-cut-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output", "words_sha256": hashes,
        "words": [{**word, "segment": "s1"} for word in words],
    })
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "cut/cut-words.json", staged)
        staged_video = root / "staged-cut.mp4"
        staged_video.write_bytes(source.read_bytes())
        run.replace_output(root / "cut/cut.mp4", staged_video)
        run.finish(extra={"plan": {"total_frames": 90}})
    if detect.unavailable_reason() is None:
        # Shared callers may not load this module's autouse fixture. Seed fake detections
        # so the stage can use its normal cache without invoking the system helper.
        cut_faces(root, root / "cut/cut.mp4", detector=_EmptyCutDetector())
    return root


def _republish_cut_picture(root: Path, staged: Path) -> None:
    """Replace the synthetic cut picture while retaining its other receipted outputs."""
    previous = read_receipt(root / "cut")
    with StageRun(root, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(root / "cut/cut.mp4", staged)
        run.finish(extra={"plan": previous["plan"]})


def _disable_face_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    def unmeasured(project_dir: Path, cut: Path) -> FacesData:
        return FacesData("unmeasured", "fake project has no face measurements",
                         VideoIdentity(cut.name, "0" * 64, 90, Fraction(30), 1080, 1920),
                         {}, {"every": 2}, (), ())
    monkeypatch.setattr(caption_stage, "cut_faces", unmeasured)
    monkeypatch.setattr(assemble_stage, "cut_faces", unmeasured)
    monkeypatch.setattr(face_project, "cut_faces", unmeasured)
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "fake detector unavailable")


@pytest.fixture(autouse=True)
def no_face_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic projects never invoke the system detector."""
    _disable_face_helper(monkeypatch)


def test_shared_project_factory_supplies_fake_cut_faces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    monkeypatch.setattr(detect, "default_detector",
                        lambda: pytest.fail("Synthetic project must use fake detections"))
    root = make_project(tmp_path)
    faces = cut_faces(root, root / "cut/cut.mp4")
    assert faces.status == "measured" and faces.detector == {"name": "fake-empty-cut"}
    assert faces.frames == ((),) * 90 and faces.gaps == ((0, 90),)


@pytest.fixture
def layer_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide project contracts backed by lavfi picture and numpy audio."""
    _disable_face_helper(monkeypatch)
    root = make_project(tmp_path, trailing_picture=True)
    data = json.loads((root / "project.json").read_bytes())
    data["rules"] = {"zoom_step_min": 0}
    write(root / "project.json", data)
    return root


def test_bleep_masks_compound_across_phrases_and_cuts(layer_project: Path, fake_layers) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    before = caption_snapshot(ProjectContext(layer_project))
    style["music"] = {"bleep_swears": False, "bleep_words_add": [], "bleep_words_remove": []}
    write(layer_project / "style.json", style)
    off = caption_snapshot(ProjectContext(layer_project))
    assert off.inputs == before.inputs and off.plan == before.plan
    legacy = ProjectContext(layer_project)
    for key in ("bleep_swears", "bleep_words_add", "bleep_words_remove"):
        legacy.render_style().data["music"].pop(key)
    assert caption_snapshot(legacy).inputs == before.inputs
    style["music"]["bleep_swears"] = True
    style["captions"]["max_words_per_phrase"] = 1
    write(layer_project / "style.json", style)
    document = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    for word, text in zip(document["words"], ["Fuck", "bull", "shit", "safe"], strict=True):
        word["text"] = text
    document["words"][2]["segment"] = "s2"
    staged = write(layer_project / "staged-bleep-words.json", document)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    snapshot = caption_snapshot(ProjectContext(layer_project))
    assert "bleep" in snapshot.inputs
    assert [phrase.text for phrase in snapshot.phrases] == ["F***", "b***", "s***", "safe"]
    captions(layer_project)
    phrases = json.loads((layer_project / "captions/phrases.json").read_bytes())
    assert [phrase["text"] for phrase in phrases["phrases"]] == ["F***", "b***", "s***", "safe"]
    assert "F***" in (layer_project / "captions/captions.srt").read_text()


@pytest.mark.parametrize("reuse_id", [False, True])
@pytest.mark.parametrize(("raw", "fixes", "expected"), [
    (("ship", "steps."), {"ship": "shit"}, "s***"),
    (("shit", "steps."), {"shit": "ship"}, "ship"),
    (("good", "dam."), {"good dam": "goddamn"}, "g******."),
    (("ship", "steps."), {"ship": "oh god damn"}, "oh g** d***"),
    (("ship-show", "steps."), {"ship-show": "shit-show"}, "s***-show"),
    (("holy", "ship"), {"holy ship": "holy shit"}, "holy s***"),
    (("shit", "hits", "the", "fan"), {}, "s***"),
    (("god-damn-it", "steps."), {}, "g**-d***-it"),
    (("shıt", "steps."), {}, "s***"),
])
def test_bleep_masks_corrected_caption_words(layer_project: Path, fake_layers,
                                             raw, fixes, expected, reuse_id: bool) -> None:
    from yap_studio.sfx import bleep

    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": True}
    write(layer_project / "style.json", style)
    project = json.loads((layer_project / "project.json").read_bytes())
    project["word_fixes"] = fixes
    write(layer_project / "project.json", project)
    document = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    for word, text in zip(document["words"], raw, strict=False):
        word["text"] = text
    if reuse_id:
        document["words"][2].update(id=document["words"][0]["id"], segment="s2", text=raw[0])
    staged = write(layer_project / "staged-fixed-words.json", document)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    snapshot = caption_snapshot(ProjectContext(layer_project))
    assert snapshot.phrases[0].words[0].text == expected
    shown = " ".join(phrase.text for phrase in snapshot.phrases)
    assert bleep.mask_text(shown, bleep.listed({})) == shown
    legacy = projection_sha256(bleep.projection(ProjectContext(layer_project).render_style().data[
        "music"]))
    assert snapshot.inputs["bleep"] != legacy
    if reuse_id:
        repeated = next(word for phrase in snapshot.phrases for word in phrase.words
                        if word.start >= 1)
        assert repeated.text == (raw[0] if any(" " in key for key in fixes) else expected)
    captions(layer_project)
    assert expected in (layer_project / "captions/captions.srt").read_text()


def test_bleep_keeps_caption_brand_casing_and_punctuation(
    layer_project: Path, fake_layers, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    set_words(["Shit"])
    document = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    document["words"][0]["text"] = "shit,"
    staged = write(layer_project / "staged-cased-words.json", document)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    assert caption_snapshot(ProjectContext(layer_project)).phrases[0].words[0].text == "Shit,"
    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": True}
    write(layer_project / "style.json", style)
    snapshot = caption_snapshot(ProjectContext(layer_project))
    assert snapshot.phrases[0].words[0].text == "S***,"
    captions(layer_project)
    assert "S***," in (layer_project / "captions/captions.srt").read_text()


def test_bleep_masks_zero_length_caption_words(layer_project: Path, fake_layers) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": True}
    write(layer_project / "style.json", style)
    document = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    document["words"][0].update(text="Fuck", start=.1, end=.1)
    staged = write(layer_project / "staged-point-words.json", document)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    snapshot = caption_snapshot(ProjectContext(layer_project))
    word = snapshot.phrases[0].words[0]
    assert word.start == word.end == .1 and word.text == "F***"
    captions(layer_project)
    assert "F***" in (layer_project / "captions/captions.srt").read_text()


@pytest.fixture
def corner_caption_context(layer_project: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectContext:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"].update({
        "highlight_style": "none", "size_px": 104, "top_px": 1503, "position": "bottom",
        "placement": "avoid_face", "background_mode": "box", "background_padding_px": 24,
        "max_words_per_phrase": 6, "max_characters_per_phrase": 48,
    })
    write(layer_project / "style.json", style)
    context = ProjectContext(layer_project)
    document = json.loads(context.read("cut/cut-words.json"))
    document["words"] = [
        {"id": f"t1:{i}", "segment": "s1", "text": text, "start": start, "end": end}
        for i, (text, start, end) in enumerate([
            ("before", .3, .45), ("the", .6, .8),
            ("corner", 1.0, 1.2), ("layout", 1.5, 1.7),
            ("then", 2.3, 2.5), ("back", 2.6, 2.8),
        ])
    ]
    read = context.read
    monkeypatch.setattr(context, "read", lambda name: json.dumps(document).encode()
                        if name == "cut/cut-words.json" else read(name))
    return context


def _corner_caption_window(context: ProjectContext, monkeypatch: pytest.MonkeyPatch,
                           *, layout: str = "corner_right",
                           frames: tuple[int, int] = (30, 60)) -> None:
    timeline = context.resolved_timeline()
    item = ResolvedElement("broll-corner", "background", "broll", *frames, "front",
                           {"layout": layout, "enter": "wipe", "wipe_frames": 9})
    monkeypatch.setattr(context, "resolved_timeline", lambda: replace(timeline, elements=(item,)))


@pytest.mark.parametrize("offset,frames", [(0, (30, 60)), (.03, (31, 61))])
def test_caption_phrases_break_at_layout_boundaries(
    corner_caption_context: ProjectContext, monkeypatch: pytest.MonkeyPatch,
    offset: float, frames: tuple[int, int],
) -> None:
    context = corner_caption_context
    if offset:
        document = json.loads(context.read("cut/cut-words.json"))
        for word in document["words"]:
            word["start"] += offset
            word["end"] += offset
        read = context.read
        monkeypatch.setattr(context, "read", lambda name: json.dumps(document).encode()
                            if name == "cut/cut-words.json" else read(name))
    _corner_caption_window(context, monkeypatch, frames=frames)
    snapshot = caption_snapshot(context)
    assert [phrase.word_ids for phrase in snapshot.phrases] == [
        ("t1:0", "t1:1"), ("t1:2", "t1:3"), ("t1:4", "t1:5"),
    ]
    assert not any(snap_frame(Fraction(str(phrase.start)), Fraction(30)) < boundary
                   < snap_frame(Fraction(str(phrase.end)), Fraction(30))
                   for phrase in snapshot.phrases for boundary in frames)
    assert snapshot.phrases[1].end == frames[1] / 30
    assert snapshot.inputs["picture:layouts"] != projection_sha256(
        green_screen.layout_windows(context.resolved_timeline()))


def test_full_background_keeps_legacy_caption_phrases_and_hashes(
    corner_caption_context: ProjectContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = corner_caption_context
    before = caption_snapshot(context)
    words = json.loads(context.read("cut/cut-words.json"))["words"]
    expected = build_phrases(
        [CaptionWord(word["id"], word["text"], word["start"], word["end"]) for word in words],
        context.render_style().data["captions"], word_fixes=context.project.word_fixes,
        text_rules=True, vocabulary=before.vocabulary,
    )
    assert before.phrases == tuple(replace(phrase, end=min(3.0, phrase.end))
                                   for phrase in expected)
    _corner_caption_window(context, monkeypatch, layout="full")
    after = caption_snapshot(context)
    assert after.phrases == before.phrases
    assert after.inputs == before.inputs
    assert after.plan == before.plan
    assert "picture:layouts" not in after.inputs


def test_corner_layout_phrases_clear_the_moved_head(
    corner_caption_context: ProjectContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = corner_caption_context
    _corner_caption_window(context, monkeypatch)
    snapshot = caption_snapshot(context)
    face = FaceBox(Box(380, 787, 387, 387), .9, "detected")
    cut = FacesData("measured", None,
                    VideoIdentity("cut.mp4", "a" * 64, 90, Fraction(30), 1080, 1920),
                    {}, {}, ((face,),) * 90, ())
    mapped, _ = map_cut_faces(
        cut, total_frames=90, end_hold_frames=0, zooms=(), inserts=(), video=cut.video,
        layouts=green_screen.layout_windows(context.resolved_timeline()),
    )
    style = context.render_style()
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    assert geometry.height == 365
    slots = json.loads(snapshot.plan.variables["phrases"])
    placements = caption_offsets(slots, geometry, mapped, style)
    assert all(not item["reason"].startswith("kept: no position") for item in placements)
    inside = next(i for i, phrase in enumerate(snapshot.phrases) if phrase.start == 1.0)
    assert placements[inside]["offset_px"] != 0
    for slot, placement in zip(slots, placements, strict=True):
        assert not placement["reason"].startswith("kept: no position")
        box = Box(70, geometry.y + placement["offset_px"], 940, geometry.height)
        start, end = caption_stage.caption_frame_window(slot["start"], slot["end"],
                                                       fps=30, layer_start=0)
        assert all(gap_px(box, head_box(item.box)) >= 24
                   for frame in mapped.frames[start:end] for item in frame)


@pytest.mark.parametrize("resolved_document", [False, True])
def test_giant_snapshot_resolves_source_once_before_taste_and_phrase_planning(
    layer_project: Path, resolved_document: bool,
) -> None:
    source = layer_project / "style.json"
    data = json.loads(source.read_text())
    data["captions"].update(highlight_style="giant", size_px=83,
                            max_words_per_phrase=8, max_characters_per_phrase=39)
    if resolved_document:
        resolved = resolve_style_data(data)
        data = ResolvedStyle(resolved["name"], "test", resolved).to_json()
    write(source, data)
    context = ProjectContext(layer_project)
    assert context.render_style().data["captions"]["size_px"] == 83
    snapshot = caption_snapshot(context)
    assert snapshot.plan.variables["size_px"] == 124
    assert snapshot.plan.variables["background_mode"] == "box"
    assert all(len(slot["words"]) <= 3
               for slot in json.loads(snapshot.plan.variables["phrases"]))
    assert json.loads(source.read_text()) == data


def test_caption_snapshot_tracks_creator_vocabulary(
    layer_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    before = caption_snapshot(ProjectContext(layer_project))
    set_words(["Steps"])
    after = caption_snapshot(ProjectContext(layer_project))
    assert before.inputs["brand:words"] != after.inputs["brand:words"]
    assert after.phrases[0].text == "simple Steps."


@pytest.fixture
def checked_layers(
    fake_layers: list[dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    """Combine the alpha fake with canned lint and check JSON from the renderer fake."""
    executable = tmp_path / "check-renderer"
    body = _FAKE_RENDERER.split('assert args[0] == "render"')[0].replace(
        'report = clean if args[0] == "lint" else {',
        'report = {**clean, "warningCount": 1, "findings": [{"severity": "warning", '
        '"message": "Review text bounds", "time": 0.25}]} if args[0] == "lint" else {',
    ).replace('"enabled": True},', '"enabled": False},')
    executable.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    executable.chmod(0o700)
    install = hyperframes.RendererInstall(
        hyperframes.HYPERFRAMES_VERSION, tmp_path, executable, None, "v22.0.0",
    )
    render = hyperframes.render_layer
    results = []

    def checked(composition: Path, out: Path, *, variables: Mapping[str, object],
                **kwargs: Any) -> hyperframes.LayerResult:
        result = render(composition, out, variables=variables, **kwargs)
        checks = layer_checks.check_layer(
            composition, variables, install=install, fps=kwargs.get("fps", 30),
            duration_s=result.duration_s, workers=kwargs.get("workers", hyperframes.RENDER_WORKERS),
            browser_gpu=kwargs.get("browser_gpu", True),
        )
        if "phrases" in variables:
            checks["passes"]["runtime"]["findings"].extend(_layout_checks([
                {"slot": slot, "boxes": [
                    {"x": 100 + i * 90,
                     "y": 20 + variables.get("band_top_px", 0) + phrase.get("offset_px", 0),
                     "w": 80, "h": 40}
                    for i in range(len(phrase["words"]))
                ]}
                for slot, phrase in enumerate(json.loads(variables["phrases"]))
            ])["passes"]["runtime"]["findings"])
        results.append(checks)
        return replace(result, checks=checks)

    monkeypatch.setattr(hyperframes, "render_layer", checked)
    return results


def set_elements(root: Path, *elements: dict[str, Any], lane: str = "agent") -> None:
    """Write a synthetic lane with current source-word identities."""
    write(root / f"timeline/{lane}.json", {
        "schema": "yap-studio/timeline@1",
        "words_sha256": {"t1": receipt.sha256_file(root / "ingest/t1/words.json")},
        "elements": list(elements),
    })


@pytest.mark.parametrize("layer", [None, "front"])
@pytest.mark.parametrize("camera", [True, False])
def test_camera_big_number_refuses_before_the_matte(
    layer_project: Path, monkeypatch, fake_layers, checked_layers, capsys, layer, camera,
) -> None:
    from yap_studio import cli
    from yap_studio.render.camera3d_look import ELEMENT_FIX, Camera3DRefusal
    from yap_studio.stages import captions_3d, timeline

    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = camera
    style["graphics"] = {"card_kinds_on": ["big_number"]}
    write(layer_project / "style.json", style)
    number = {"id": "agent-number", "kind": "graphic", "template": "big_number",
              "slots": {"end": 2}, "start": {"time": 0}, "hold_s": 2.5}
    if layer is not None:
        number["layer"] = layer
    set_elements(layer_project, number)
    monkeypatch.setattr(timeline, "spoken_number_fix", lambda *a: None)
    monkeypatch.setattr(captions_3d, "require_cut", lambda *a: None)
    monkeypatch.setattr(captions_3d, "ensure_cut_matte", lambda *a, **kw: pytest.fail(
        "prepared a matte for a Big number conflict"))
    if camera:
        with pytest.raises(Camera3DRefusal) as error:
            captions(layer_project)
        assert error.value.code == "camera3d_refused"
        assert error.value.suggested_action == ELEMENT_FIX
        findings = ProjectContext(layer_project).resolved_timeline(allow_violations=True).findings
        finding, = [item for item in findings if item.rule == "camera_3d"]
        assert finding.severity == "violation" and finding.elements == ("agent-number",)
        assert finding.message == str(error.value) and finding.suggested_action == ELEMENT_FIX
        assert cli.main(["timeline", "check", str(layer_project)]) == 3
        output = capsys.readouterr().out
        assert finding.message in output and ELEMENT_FIX in output
        assert not (layer_project / "captions/receipt.json").exists()
    else:
        assert cli.main(["timeline", "check", str(layer_project)]) == 0
        assert "camera_3d" not in capsys.readouterr().out
        result = captions(layer_project)
        assert result["status"] == "verified" and "picture" not in result
        item, = ProjectContext(layer_project).resolved_timeline().elements
        assert item.layer == "front" and item.payload["template"] == "big_number"


@pytest.mark.parametrize("owned_picture", [False, True])
def test_camera_gaps_hiding_every_word_publish_the_empty_flat_layer(
    layer_project: Path, monkeypatch, fake_layers, checked_layers, owned_picture,
) -> None:
    from yap_studio.stages import captions_3d

    gaps = [{"id": "agent-gap-one", "kind": "caption_gap",
             "start": {"time": 0}, "end": {"time": .8}},
            {"id": "agent-gap-two", "kind": "caption_gap",
             "start": {"time": .8}, "end": {"time": 3}}]
    set_elements(layer_project, *gaps)
    flat = captions(layer_project)
    empty = {name: (layer_project / "captions" / name).read_bytes()
             for name in ("phrases.json", "captions.srt", "layout.json")}
    if owned_picture:
        previous = read_receipt(layer_project / "captions")
        with StageRun(layer_project, "captions") as run:
            for name, digest in previous["inputs"].items():
                run.record_inputs(name, digest)
            run.outputs.update({item["path"]: item for item in previous["outputs"]})
            for name in ("camera3d.mp4", "camera3d.json"):
                staged = layer_project / f"old-{name}"
                staged.write_bytes(b"synthetic camera output")
                run.replace_output(layer_project / "captions" / name, staged)
            run.finish(extra={"picture": "captions/camera3d.mp4", "layer": flat["layer"]})
    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = True
    write(layer_project / "style.json", style)
    monkeypatch.setattr(captions_3d, "require_cut", lambda *a: None)
    monkeypatch.setattr(captions_3d, "ensure_cut_matte", lambda *a, **kw: pytest.fail(
        "prepared a matte with no visible captions"))
    monkeypatch.setattr(captions_3d, "render_outputs", lambda *a: pytest.fail(
        "rendered a camera picture with no visible captions"))
    result = captions(layer_project)
    record = read_receipt(layer_project / "captions")
    assert result["phrases"] == record["phrases"] == 0 and not result["noop"]
    assert result["layer"] == flat["layer"]
    assert "picture" not in result and "picture" not in record
    assert {item["path"] for item in record["outputs"]} == {
        "captions/captions.mov", "captions/phrases.json", "captions/captions.srt",
        "captions/layout.json"}
    for name, contents in empty.items():
        assert (layer_project / "captions" / name).read_bytes() == contents
    assert not (layer_project / "captions/camera3d.mp4").exists()
    assert not (layer_project / "captions/camera3d.json").exists()
    assert record["inputs"] == captions_inputs(ProjectContext(layer_project))
    assert state(layer_project, "captions")["state"] == "done"
    assert captions(layer_project)["noop"] and len(fake_layers) == 2
    plan = caption_snapshot(ProjectContext(layer_project)).plan
    assert _alpha_frames(layer_project / result["out"], plan).max() == 0
    set_elements(layer_project, gaps[0])
    assert state(layer_project, "captions")["state"] == "stale"


@pytest.mark.parametrize("on", [False, True])
def test_camera_inputs_cover_listed_hero_exclusions(layer_project: Path, on) -> None:
    target = layer_project / "style.json"
    style = json.loads(target.read_bytes())
    style["captions"]["camera_3d"] = True
    style["music"] = {"bleep_swears": on}
    write(target, style)
    before = captions_inputs(ProjectContext(layer_project))
    style["music"]["bleep_words_add"] = ["simple"]
    write(target, style)
    assert captions_inputs(ProjectContext(layer_project)) != before


@pytest.mark.parametrize("on", [False, True])
def test_camera_bleep_publishes_masked_words_and_inputs(layer_project: Path, monkeypatch, on):
    from types import SimpleNamespace

    import numpy as np
    from test_camera3d import _silhouette
    from test_faces_mapping import _cut

    from yap_studio.render import camera3d
    from yap_studio.sfx import bleep
    from yap_studio.stages import captions_3d

    target = layer_project / "style.json"
    style = json.loads(target.read_bytes())
    style["captions"]["camera_3d"] = True
    style["music"] = {"bleep_swears": on, "bleep_words_add": ["simple"]}
    write(target, style)
    before = captions_inputs(ProjectContext(layer_project))
    monkeypatch.setattr(captions_3d, "require_cut", lambda *a: None)
    monkeypatch.setattr(captions_3d, "ensure_cut_matte", lambda *a, **kw: layer_project / "matte")
    monkeypatch.setattr(captions_3d, "sample_matte", lambda *a: (None, [
        {"start": 0, "end": 3, **_silhouette()[1]}]))
    monkeypatch.setattr(captions_3d, "measure_camera_motion", lambda *a, **kw: {
        "status": "not measurable", "moving": False})
    monkeypatch.setattr(camera3d, "sample_cut_luminance", lambda *a: None)
    monkeypatch.setattr(captions_3d, "cut_faces", lambda *a: replace(
        _cut(90), frames=((),) * 90, gaps=((0, 90),)))
    monkeypatch.setattr(captions_3d, "prepare", lambda *a: (layer_project / "plate",
                                                         layer_project / "person"))
    planner = camera3d.plan_camera3d
    monkeypatch.setattr(camera3d, "plan_camera3d", lambda *a, **kw: planner(
        *a, **{**kw, "matte": None}))

    def render(plan, plate, person, cut, work):
        for name in ("camera3d.mp4", "captions.mov"):
            (work / name).write_bytes(b"synthetic camera picture")
        return {"retries": [], "flags": []}

    monkeypatch.setattr(captions_3d, "render_outputs", render)
    monkeypatch.setattr(captions_3d, "check_behind", lambda *a: ())
    captions(layer_project)
    plan = json.loads((layer_project / "captions/camera3d.json").read_bytes())
    phrases = json.loads((layer_project / "captions/phrases.json").read_bytes())
    expected = "s*****" if on else "simple"
    assert any(expected in g["text"].lower() for g in plan["groups"])
    assert all(s["hero"] is None or s["hero"]["text"].lower() != "simple"
               for s in plan["sentences"])
    assert expected in " ".join(p["text"] for p in phrases["phrases"]).lower()
    assert expected in (layer_project / "captions/captions.srt").read_text().lower()
    assert [span["word"].lower() for span in plan["bleeps"]] == (["simple"] if on else [])
    assert ("bleep" in before) is on
    if on:
        original = np.full((3 * bleep.RATE, 1), .1, dtype=np.float32)
        context = ProjectContext(layer_project)
        words = [SimpleNamespace(**w) for w in context.json("cut/cut-words.json")["words"]]
        occurrences = bleep.find_display(words, bleep.listed(style["music"]),
                                         word_fixes=context.project.word_fixes)
        voice, tone, _ = bleep.render(occurrences, original, 3, len(original))
        checked = bleep.check(words, music=style["music"], voice=voice,
                              original_voice=original, tone_stem=tone, phrases=phrases,
                              title_layers=[], timeline=context.resolved_timeline(),
                              word_fixes=context.project.word_fixes)
        assert checked["pass"], checked
    assert captions(layer_project)["noop"]
    style["music"]["bleep_swears"] = not on
    write(target, style)
    assert captions_inputs(ProjectContext(layer_project)) != before
    assert state(layer_project, "captions")["state"] == "stale"


@pytest.mark.parametrize("hidden", [False, True, "overlap", "obstacle", "interrupted",
                                     "replacement"])
def test_camera_stage_publishes_and_reuses(
    layer_project: Path, monkeypatch, hidden, fake_layers,
) -> None:
    from test_camera3d import _silhouette

    from yap_studio.render import camera3d
    from yap_studio.stages import captions_3d

    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_bytes())
    data["captions"]["camera_3d"] = True
    write(style_path, data)
    monkeypatch.setattr(captions_3d, "require_cut", lambda *a: None)
    monkeypatch.setattr(captions_3d, "ensure_cut_matte",
                        lambda *a, **kw: layer_project / "matte.mkv")
    monkeypatch.setattr(captions_3d, "prepare", lambda *a: (layer_project / "plate.mp4",
                                                         layer_project / "person.webm"))
    if hidden == "obstacle":
        monkeypatch.setattr(captions_3d, "planner_windows", lambda *a, **kw: ([{
            "start_frame": 0, "end_frame": 90, "box": [0, 0, 1080, 1000]}], []))
        monkeypatch.setattr(captions_3d, "prepare", lambda *a: pytest.fail(
            "prepared a plate before checking obstacles"))
    monkeypatch.setattr(captions_3d, "sample_matte", lambda *a: (None, [
        {"start": 0, "end": 3, **_silhouette()[1]}]))
    monkeypatch.setattr(captions_3d, "measure_camera_motion", lambda *a, **kw: {
        "status": "not measurable", "moving": False, "pairs": [], "shift_p75_px": None})
    monkeypatch.setattr(camera3d, "sample_cut_luminance", lambda *a: None)
    if hidden == "replacement":
        monkeypatch.setattr(captions_3d, "picture_windows", lambda *a: [(0, 90)], raising=False)
    planner = camera3d.plan_camera3d

    def plan(*args, **kwargs):
        if hidden == "replacement":
            assert kwargs["replacements"] == [(0, 90)]
        result = planner(*args, **{**kwargs, "matte": None})
        if hidden in {"overlap", "obstacle"}:
            group = result["groups"][0]
            group.update(flat=True, box=[0, 0, 1080, 1920], ref_size=1920)
            base = result["font_metrics"]["base"]["caption"]
            for pose in group["poses"]:
                pose["box"] = [0, 0, 1080, 1920]
                pose.update(ref_x=0, ref_y=base * 1920, scale=1)
        return result

    monkeypatch.setattr(camera3d, "plan_camera3d", plan)
    from test_faces_mapping import _cut

    faces = replace(_cut(90), frames=((),) * 90, gaps=((0, 90),))
    if hidden == "overlap":
        faces = replace(_cut(90), frames=((FaceBox(Box(420, 620, 240, 300),
                                                   .9, "detected"),),) * 90)
    elif hidden == "replacement":
        faces = replace(_cut(90), frames=((FaceBox(Box(0, 0, 1080, 1920),
                                                   .9, "detected"),),) * 90)
    monkeypatch.setattr(captions_3d, "cut_faces", lambda *a: faces)
    calls = []

    def render(plan, plate, person, cut, work):
        calls.append(plan)
        for name in ("camera3d.mp4", "captions.mov"):
            (work / name).write_bytes(b"synthetic rendered picture")
        return {"retries": [], "flags": ["--no-browser-gpu"]}

    monkeypatch.setattr(captions_3d, "render_outputs", render)
    monkeypatch.setattr(captions_3d, "check_behind", lambda *a: (
        ({"id": "hero-0", "flag": True, "worst_visible_share": .7},)
        if hidden and len(calls) == 1 else ()))
    if hidden == "interrupted":
        from yap_studio.project.receipts import read_pending

        def stop(*args, **kwargs):
            raise RuntimeError("interrupted before finishing captions")

        with monkeypatch.context() as patch:
            patch.setattr(StageRun, "finish", stop)
            with pytest.raises(RuntimeError, match="interrupted"):
                captions(layer_project)
        assert read_receipt(layer_project / "captions") is None
        pending = read_pending(layer_project / "captions")
        camera_files = {"captions/camera3d.mp4", "captions/camera3d.json"}
        assert camera_files <= {item["path"] for item in pending["outputs"]}
        data["captions"]["camera_3d"] = False
        write(style_path, data)
        assert not captions(layer_project)["noop"]
        assert "picture" not in read_receipt(layer_project / "captions")
        assert all(not (layer_project / name).exists() for name in camera_files)
        data["captions"]["camera_3d"] = True
        write(style_path, data)
        assert captions(layer_project)["picture"] == "captions/camera3d.mp4"
        record = read_receipt(layer_project / "captions")
        for name in sorted(camera_files):
            target = layer_project / name
            assert target.is_file()
            assert any(item["path"] == name and item["sha256"] == receipt.sha256_file(target)
                       for item in record["outputs"])
        return
    if hidden in {"overlap", "obstacle"}:
        from yap_studio.errors import YapError

        message = "face clearance" if hidden == "overlap" else "before rendering"
        with pytest.raises(YapError, match=message):
            captions(layer_project)
        assert not calls
        return
    result = captions(layer_project)
    assert not result["noop"] and result["picture"] == "captions/camera3d.mp4"
    record = read_receipt(layer_project / "captions")
    assert record["camera_3d"]["groups"] > 0
    assert record["layer"] == {"path": "captions/captions.mov", "x": 0, "y": 0,
                                "width": 1080, "height": 1920, "start_frame": 0, "frames": 90}
    assert {Path(item["path"]).name for item in record["outputs"]} == {
        "captions.mov", "camera3d.mp4", "camera3d.json", "layout.json", "phrases.json",
        "captions.srt"}
    plan = json.loads((layer_project / "captions/camera3d.json").read_bytes())
    assert any(item["words"][0]["text"].lower() == "simple" for item in [
        *plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])])
    context = ProjectContext(layer_project)
    retained = captions_3d.snapshot(context)
    expected = build_phrases([CaptionWord(w["id"], w["text"], w["start"], w["end"])
                              for w in retained.words], context.render_style().data["captions"])
    phrases = json.loads((layer_project / "captions/phrases.json").read_bytes())["phrases"]
    assert phrases == [p.to_dict() for p in expected]
    srt = (layer_project / "captions/captions.srt").read_text()
    assert srt == caption_stage._srt_text(expected)
    assert captions(layer_project)["noop"] and len(calls) == (2 if hidden else 1)
    assert record["camera_3d"]["removed_heroes"] == ([0] if hidden else [])
    if hidden:
        assert plan["sentences"][0]["hero"] is None
    baseline = captions_inputs(ProjectContext(layer_project))
    data["brand"] = {"accent_color": "#FF3344"}
    write(style_path, data)
    assert captions_inputs(ProjectContext(layer_project)) != baseline
    data["captions"]["camera_3d"] = False
    write(style_path, data)
    assert captions_inputs(ProjectContext(layer_project)) != baseline
    monkeypatch.setattr(captions_3d, "read_pending", lambda *a: {"outputs": []}, raising=False)
    assert not captions(layer_project)["noop"]
    assert not (layer_project / "captions/camera3d.mp4").exists()
    assert not (layer_project / "captions/camera3d.json").exists()
    assert "picture" not in read_receipt(layer_project / "captions")
    data["captions"]["camera_3d"] = True
    write(style_path, data)
    assert not captions(layer_project)["noop"]
    record = read_receipt(layer_project / "captions")
    assert record["picture"] == "captions/camera3d.mp4"
    for name in ("camera3d.mp4", "camera3d.json"):
        target = layer_project / "captions" / name
        assert target.is_file()
        assert any(o["path"] == f"captions/{name}"
                   and o["sha256"] == receipt.sha256_file(target) for o in record["outputs"])
    set_elements(layer_project, {"id": "agent-gap", "kind": "caption_gap",
                                 "start": {"time": .8}, "end": {"time": 1.4}})
    assert not captions(layer_project)["noop"]
    plan = calls[-1]
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    assert items and all(item["off_frame"] <= 24 or item["on_frame"] >= 42 for item in items)
    phrases = json.loads((layer_project / "captions/phrases.json").read_bytes())["phrases"]
    assert all(p["end"] <= .8 or p["start"] >= 1.4 for p in phrases)
    assert captions(layer_project)["noop"]


@pytest.mark.parametrize("survivor_fails", [None, False, True])
def test_camera_retry_keeps_only_measured_heroes(
    layer_project: Path, monkeypatch, survivor_fails,
) -> None:
    from test_camera3d import _silhouette
    from test_faces_mapping import _cut

    from yap_studio.errors import YapError
    from yap_studio.render import camera3d
    from yap_studio.stages import captions_3d

    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = True
    write(layer_project / "style.json", style)
    document = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    texts = ["Beautiful.", "Beautiful."] + (["stories."] if survivor_fails is not None else [])
    document["words"] = [{"id": f"t1:{i}", "segment": "s1", "text": text,
                           "start": .2 + i, "end": .5 + i} for i, text in enumerate(texts)]
    staged = write(layer_project / "retry-words.json", document)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    monkeypatch.setattr(captions_3d, "require_cut", lambda *a: None)
    monkeypatch.setattr(captions_3d, "ensure_cut_matte", lambda *a, **kw: layer_project / "matte")
    monkeypatch.setattr(captions_3d, "prepare", lambda *a: (layer_project / "plate",
                                                         layer_project / "person"))
    monkeypatch.setattr(captions_3d, "sample_matte", lambda *a: (None, [
        {"start": 0, "end": 3, **_silhouette()[1]}]))
    monkeypatch.setattr(captions_3d, "measure_camera_motion", lambda *a, **kw: {
        "status": "not measurable", "moving": False})
    monkeypatch.setattr(captions_3d, "cut_faces", lambda *a: replace(
        _cut(90), frames=((),) * 90, gaps=((0, 90),)))
    monkeypatch.setattr(camera3d, "sample_cut_luminance", lambda *a: None)
    monkeypatch.setattr(camera3d, "_glyph_visible_share", lambda *a: 1)
    planner = camera3d.plan_camera3d
    monkeypatch.setattr(camera3d, "plan_camera3d", lambda *a, **kw: planner(
        *a, **{**kw, "matte": None}))
    plans = []

    def render(plan, plate, person, cut, work):
        plans.append(plan)
        for name in ("camera3d.mp4", "captions.mov"):
            (work / name).write_bytes(b"synthetic camera output")
        return {"retries": [], "flags": ["--no-browser-gpu"]}

    def behind(plan, *args):
        return tuple({"id": f"hero-{i}", "flag": i != 2 or bool(survivor_fails and len(plans) == 2)}
                     for i, sentence in enumerate(plan["sentences"]) if sentence["hero"])

    monkeypatch.setattr(captions_3d, "render_outputs", render)
    monkeypatch.setattr(captions_3d, "check_behind", behind)
    if survivor_fails:
        with pytest.raises(YapError) as error:
            captions(layer_project)
        assert error.value.code == "camera3d_behind_words"
        assert read_receipt(layer_project / "captions") is None
    else:
        result = captions(layer_project)
        record = read_receipt(layer_project / "captions")
        assert result["status"] == record["status"] == "verified"
        assert record["camera_3d"]["removed_heroes"] == [0]
        assert {item["sentence"] for item in record["camera_3d"]["fallbacks"]
                if item["action"] == "hero removed"} == {0}
    assert len(plans) == 2
    before = {i for i, sentence in enumerate(plans[0]["sentences"]) if sentence["hero"]}
    after = {i for i, sentence in enumerate(plans[1]["sentences"]) if sentence["hero"]}
    assert before == ({0} if survivor_fails is None else {0, 2})
    assert after == before - {0}
    assert [w["id"] for group in plans[1]["groups"] for w in group["words"]].count("t1:0") == 1
    assert [w["id"] for group in plans[1]["groups"] for w in group["words"]].count("t1:1") == 1


def test_title_faces_match_final_camera_mapping(layer_project: Path) -> None:
    from test_camera3d import _silhouette
    from test_faces_mapping import _cut

    from yap_studio.faces.mapping import map_picture_faces
    from yap_studio.render import camera3d
    from yap_studio.stages import captions_3d

    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    words = captions_3d.snapshot(context).words
    faces = _cut(timeline.total_frames)
    timeline = replace(timeline, elements=(
        ResolvedElement("agent-insert", "insert", "agent", 30, 40, "front", {}),
        ResolvedElement("agent-transition", "transition", "agent", 50, 55, "front", {}),
    ))
    plan = camera3d.plan_camera3d({"time_base": "output", "words": words},
                                 timeline.total_frames / 30, _silhouette()[1], onset_lead=.09)
    expected, _ = map_picture_faces(
        faces, timeline, zooms=(), camera=camera3d.face_transforms(plan),
        video=replace(faces.video, name="picture.mp4",
                      frames=timeline.total_frames + timeline.end_hold_frames))
    mapped = captions_3d.title_faces(faces, words, timeline)
    assert mapped.to_dict() == expected.to_dict()
    assert mapped.frames[3][0].box != faces.frames[3][0].box
    assert mapped.frames[30:40] == ((),) * 10 and mapped.frames[50:55] == ((),) * 5


def test_camera_snapshot_tracks_face_identity(layer_project: Path, monkeypatch) -> None:
    from yap_studio.stages import captions_3d

    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_bytes())
    data["captions"]["camera_3d"] = True
    write(style_path, data)
    before = captions_inputs(ProjectContext(layer_project))
    monkeypatch.setattr(captions_3d, "cut_faces_identity", lambda *a: {"helper": "changed"},
                        raising=False)
    assert captions_inputs(ProjectContext(layer_project)) != before


def test_camera_snapshot_tracks_chapter_windows(layer_project: Path) -> None:
    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_bytes())
    data["captions"]["camera_3d"] = True
    data["chapters"] = {"style": "title_card"}
    write(style_path, data)
    context = ProjectContext(layer_project)
    before = caption_snapshot(context).inputs["camera3d:plan"]
    chapter = {"id": "agent-part", "kind": "chapter", "start": {"time": .2},
               "hold_s": 1, "title": "A new chapter"}
    set_elements(layer_project, chapter)
    during = caption_snapshot(ProjectContext(layer_project)).inputs["camera3d:plan"]
    assert during != before
    set_elements(layer_project, {**chapter, "start": {"time": .4}})
    assert caption_snapshot(ProjectContext(layer_project)).inputs["camera3d:plan"] != during


@pytest.mark.parametrize("change", [{"position": "center"}, {"size_px": 100},
                                  {"style": "kinetic"}])
def test_camera_snapshot_tracks_title_geometry(layer_project: Path, change: dict) -> None:
    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_bytes())
    data["captions"]["camera_3d"] = True
    data["hook"].update(style="paper_title", position="top")
    write(style_path, data)
    set_elements(layer_project, {"id": "agent-title", "kind": "title",
                                 "start": {"time": 0}, "hold_s": 1,
                                 "lines": ["A new story"], "accent": "story"})
    before = caption_snapshot(ProjectContext(layer_project)).inputs["camera3d:plan"]
    data["hook"].update(change)
    write(style_path, data)
    assert caption_snapshot(ProjectContext(layer_project)).inputs["camera3d:plan"] != before


def test_camera_back_refuses_before_mutating(layer_project: Path) -> None:
    from yap_studio.render.camera3d_look import Camera3DRefusal

    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = True
    write(layer_project / "style.json", style)
    with pytest.raises(Camera3DRefusal, match="own halo") as error:
        caption_stage.back_weak_captions(layer_project, "shadow")
    assert error.value.suggested_action == "Run yap-studio captions DIR without --back."


@pytest.mark.parametrize("missing", ["cutout unavailable", "person matte has no visible person"])
def test_camera_stage_refuses_missing_person(layer_project: Path, monkeypatch, missing) -> None:
    from yap_studio.matte.project import CutMatteError
    from yap_studio.render.camera3d_look import Camera3DRefusal
    from yap_studio.stages import captions_3d

    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = True
    write(layer_project / "style.json", style)
    monkeypatch.setattr(captions_3d, "require_cut", lambda *a: None)

    def matte(*args, **options):
        if missing == "cutout unavailable":
            raise CutMatteError(missing)
        return layer_project / "matte.mkv"

    def summarize(*args):
        raise ValueError(missing)

    monkeypatch.setattr(captions_3d, "ensure_cut_matte", matte)
    monkeypatch.setattr(captions_3d, "sample_matte", summarize)
    with pytest.raises(Camera3DRefusal, match=missing) as error:
        captions(layer_project)
    assert error.value.code == "camera3d_refused"
    assert error.value.suggested_action == (
        "Pick another caption look, or keep only shots that show you.")


def test_camera_stage_checks_shot_coverage_before_preparation(layer_project: Path, monkeypatch):
    from test_camera3d import _silhouette

    from yap_studio.render import camera3d
    from yap_studio.render.camera3d_look import Camera3DRefusal, person_refusal
    from yap_studio.stages import captions_3d

    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = True
    write(layer_project / "style.json", style)
    monkeypatch.setattr(captions_3d, "require_cut", lambda *a: None)
    monkeypatch.setattr(captions_3d, "ensure_cut_matte",
                        lambda *a, **kw: layer_project / "matte.mkv")
    monkeypatch.setattr(camera3d, "summarize_matte", lambda *a: [
        {"start": 0, "end": 3, **_silhouette()[1]}])
    calls = []

    def sample(*args):
        calls.append("coverage")
        raise person_refusal("no person at 0:00")

    def unexpected(*args, **kwargs):
        pytest.fail("continued before checking person coverage")

    monkeypatch.setattr(captions_3d, "sample_matte", sample, raising=False)
    monkeypatch.setattr(captions_3d, "measure_camera_motion", unexpected)
    monkeypatch.setattr(captions_3d, "prepare", unexpected)
    monkeypatch.setattr(captions_3d, "render_outputs", unexpected)
    with pytest.raises(Camera3DRefusal, match="no person at 0:00"):
        captions(layer_project)
    assert calls == ["coverage"]


def test_camera_snapshot_tracks_windows_without_preparation(
    layer_project: Path, monkeypatch,
) -> None:
    from yap_studio.stages import captions_3d

    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_bytes())
    style["captions"]["camera_3d"] = True
    write(style_path, style)
    monkeypatch.setattr(captions_3d, "ensure_cut_matte",
                        lambda *a, **kw: pytest.fail("snapshot prepared a matte"))
    context = ProjectContext(layer_project)
    original = caption_snapshot(context).inputs
    timeline = context.resolved_timeline()
    card = ResolvedElement("agent-card", "graphic", "agent", 10, 40, "front",
                           {"box": {"x": 60, "y": 200, "width": 960, "height": 400}})
    monkeypatch.setattr(context, "resolved_timeline", lambda **kw: replace(
        timeline, elements=(card,)))
    assert caption_snapshot(context).inputs != original
    with_card = caption_snapshot(context).inputs
    monkeypatch.setattr(context, "resolved_timeline", lambda **kw: replace(
        timeline, elements=(replace(card, end_frame=50),)))
    assert caption_snapshot(context).inputs != with_card
    old = hyperframes.CAPTION_BROWSER_GPU
    monkeypatch.setattr(hyperframes, "CAPTION_BROWSER_GPU", not old)
    assert caption_snapshot(context).inputs["renderer"] != original["renderer"]
    set_words(["BrightFold"])
    assert caption_snapshot(context).inputs["brand:words"] != original["brand:words"]


def test_flat_camera_false_keeps_receipt_and_inputs(layer_project: Path, fake_layers) -> None:
    context = ProjectContext(layer_project)
    before = caption_snapshot(context).inputs
    old_style = context.render_style()
    data = json.loads(json.dumps(old_style.data))
    data["captions"].pop("camera_3d")
    context._render_style = replace(old_style, data=data)
    assert caption_snapshot(context).inputs == before
    captions(layer_project)
    digest = receipt.sha256_file(layer_project / "captions/receipt.json")
    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = False
    write(layer_project / "style.json", style)
    assert captions(layer_project)["noop"]
    assert receipt.sha256_file(layer_project / "captions/receipt.json") == digest


def state(root: Path, stage: str) -> dict[str, Any]:
    """Find a stage in the public status result."""
    return next(item for item in status(root)["stages"] if item["name"] == stage)


def _shrink_checks(phrases: list[dict], *, floor_px: int | float = 32) -> dict:
    """Return normalized runtime evidence with a caption shrink summary."""
    return {"passes": {"runtime": {"status": "passed", "findings": [{
        "severity": "info", "code": "console_warning",
        "message": layer_checks.CAPTION_SHRINK_PREFIX + json.dumps({
            "floor_px": floor_px, "phrases": phrases,
        }),
    }]}}}


def _layout_checks(layout: Any) -> dict:
    """Return a shrink summary carrying the supplied layout evidence."""
    checks = _shrink_checks([])
    checks["passes"]["runtime"]["findings"][0]["message"] = (
        layer_checks.CAPTION_SHRINK_PREFIX + json.dumps({
            "floor_px": 32, "phrases": [], "layout": layout,
        })
    )
    return checks


class _CaptionDetector:
    """Measure a low face during the first phrase only."""

    def __init__(self, box: Box) -> None:
        self.box = box
        self.calls = 0

    def detect(self, video: Path, *, every: int) -> DetectorRun:
        self.calls += 1
        return DetectorRun(1080, 1920, 90, {
            n: (FaceBox(self.box, 0.95, "detected"),) if n < 30 else ()
            for n in sample_frames(90, every)
        }, {"name": "fake-caption-face"})


@pytest.fixture
def measured_faces(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> _CaptionDetector:
    """Use a real cut clock with a fake detector and the production face cache."""
    staged = layer_project / "staged-cut.mp4"
    ffmpeg.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                "color=black:size=1080x1920:rate=30", "-frames:v", "90", "-c:v", "libx264",
                "-preset", "ultrafast", str(staged)], timeout_s=30)
    _republish_cut_picture(layer_project, staged)
    style = ProjectContext(layer_project).render_style()
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    detector = _CaptionDetector(Box(400, geometry.y + 40, 200, 180))
    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    monkeypatch.setattr(caption_stage, "cut_faces",
                        lambda project_dir, cut: cut_faces(project_dir, cut, detector=detector))
    return detector


def test_colliding_phrase_moves_and_glyph_layout_follows(
    layer_project: Path, measured_faces: _CaptionDetector, checked_layers: list[dict],
    fake_layers: list[dict],
) -> None:
    style = ProjectContext(layer_project).render_style()
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    snapshot = caption_snapshot(ProjectContext(layer_project))
    assert snapshot.plan.y == geometry.y and snapshot.plan.height == geometry.height
    assert all("offset_px" not in slot for slot in json.loads(snapshot.plan.variables["phrases"]))
    assert measured_faces.calls == 0
    result = captions(layer_project)
    slots = json.loads(fake_layers[0]["variables"]["phrases"])
    placement = read_receipt(layer_project / "captions")["caption_placement"]
    offset = placement["phrases"][0]["offset_px"]
    layer = result["layer"]
    assert layer["y"] >= style.data["internal"]["key_text_bottom_px"]
    assert layer["y"] == geometry.y + offset
    assert layer["height"] == geometry.height - offset
    assert offset == slots[0]["offset_px"] < 0 and slots[1]["offset_px"] == 0
    element = Box(70, geometry.y + offset, 940, geometry.height)
    assert gap_px(element, head_box(measured_faces.box)) >= face_clearance_px(style.data)
    layout = json.loads((layer_project / "captions/layout.json").read_text())
    assert layout["captions"][0]["boxes"][0]["y"] == geometry.y + 20 + offset
    assert layout["captions"][1]["boxes"][0]["y"] == geometry.y + 20
    assert placement["phrases"] == [{"slot": 0, "text": "simple steps.", "offset_px": offset,
                                      "reason": f"moved up {-offset} px to stay "
                                      f"{face_clearance_px(style.data):.0f} px clear of the head"}]
    assert placement["kept_phrases"] == 1
    assert placement["faces"] == {"status": "measured", "reason": None, "frame_sources": [
        {"start_frame": 0, "end_frame": 90, "source": "cut"},
        {"start_frame": 90, "end_frame": 135, "source": "hold"},
    ]}
    assert measured_faces.calls == 1


def test_caption_worker_setting_changes_only_caption_identity(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_elements(
        layer_project,
        {"id": "agent-title", "kind": "title", "start": {"time": 0.2},
         "lines": ["Small", "steps"], "accent": "steps"},
        {"id": "agent-card", "kind": "graphic", "start": {"time": 1.5},
         "hold_s": 1, "template": "stat", "slots": {"end": 12}},
    )
    caption_before = caption_snapshot(ProjectContext(layer_project)).inputs["renderer"]
    graphics_before = graphics_snapshot(ProjectContext(layer_project)).inputs
    monkeypatch.setattr(hyperframes, "CAPTION_WORKERS", 4)
    assert caption_snapshot(ProjectContext(layer_project)).inputs["renderer"] != caption_before
    assert graphics_snapshot(ProjectContext(layer_project)).inputs == graphics_before


def test_caption_drawing_mode_changes_only_caption_identity(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_elements(
        layer_project,
        {"id": "agent-title", "kind": "title", "start": {"time": 0.2},
         "lines": ["Small", "steps"], "accent": "steps"},
        {"id": "agent-card", "kind": "graphic", "start": {"time": 1.5},
         "hold_s": 1, "template": "stat", "slots": {"end": 12}},
    )
    before = caption_snapshot(ProjectContext(layer_project))
    assert before.inputs["renderer"] == projection_sha256(before.assets.template_identity(
        workers=hyperframes.CAPTION_WORKERS, browser_gpu=False,
    ))
    graphics_before = graphics_snapshot(ProjectContext(layer_project)).inputs
    monkeypatch.setattr(hyperframes, "CAPTION_BROWSER_GPU", True)
    after = caption_snapshot(ProjectContext(layer_project))
    assert after.inputs["renderer"] == projection_sha256(after.assets.template_identity(
        workers=hyperframes.CAPTION_WORKERS, browser_gpu=True,
    ))
    assert after.inputs["renderer"] != before.inputs["renderer"]
    assert graphics_snapshot(ProjectContext(layer_project)).inputs == graphics_before


def test_changed_cut_identity_stales_captions(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict],
) -> None:
    captions(layer_project)
    before = captions_inputs(ProjectContext(layer_project))
    target = layer_project / "cut/cut.mp4"
    staged = layer_project / "staged-cut.mp4"
    staged.write_bytes(target.read_bytes() + b"\n")
    _republish_cut_picture(layer_project, staged)
    measured_faces.box = Box(400, 300, 200, 180)
    assert state(layer_project, "captions")["state"] == "stale"
    assert captions_inputs(ProjectContext(layer_project))["faces"] != before["faces"]
    assert measured_faces.calls == 1
    assert not captions(layer_project)["noop"]
    assert read_receipt(layer_project / "captions")["caption_placement"]["phrases"] == []
    assert len(fake_layers) == 2 and measured_faces.calls == 2


def test_status_and_inputs_do_not_measure_faces_and_noop_skips_placement(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = captions_inputs(ProjectContext(layer_project))
    snapshot = caption_snapshot(ProjectContext(layer_project))
    status(layer_project)
    assert measured_faces.calls == 0
    assert not (layer_project / ".yap/cache/faces").exists()
    assert inputs["faces"] == projection_sha256(cut_faces_identity(
        ProjectContext(layer_project).identity.sha256(layer_project / "cut/cut.mp4"),
    ))
    first = captions(layer_project)
    assert measured_faces.calls == 1
    assert first["layer"]["y"] < snapshot.plan.y

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("status and an up-to-date caption run must not measure or place faces")

    with monkeypatch.context() as patch:
        patch.setattr(caption_stage, "cut_faces", unexpected)
        patch.setattr(caption_stage, "map_cut_faces", unexpected)
        patch.setattr(caption_stage, "caption_offsets", unexpected)
        assert captions_inputs(ProjectContext(layer_project)) == inputs
        assert state(layer_project, "captions")["state"] == "done"
        assert caption_snapshot(ProjectContext(layer_project)).plan == snapshot.plan
        second = captions(layer_project)
        assert second["noop"] and second["layer"] == first["layer"]
    assert measured_faces.calls == 1 and len(fake_layers) == 1
    assert not captions(layer_project, force=True)["noop"]
    assert measured_faces.calls == 1 and len(fake_layers) == 2


def test_face_availability_change_stales_captions(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = captions_inputs(ProjectContext(layer_project))
    captions(layer_project)
    assert state(layer_project, "captions")["state"] == "done"
    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    assert captions_inputs(ProjectContext(layer_project))["faces"] != before["faces"]
    assert state(layer_project, "captions")["state"] == "stale"
    assert not captions(layer_project)["noop"]
    assert state(layer_project, "captions")["state"] == "done" and len(fake_layers) == 2


def test_os_update_stales_captions_and_remeasures_faces(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.7", ("", "", ""), ""))
    captions(layer_project)
    before = captions_inputs(ProjectContext(layer_project))
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.7.1", ("", "", ""), ""))
    assert state(layer_project, "captions")["state"] == "stale"
    assert captions_inputs(ProjectContext(layer_project))["faces"] != before["faces"]
    assert measured_faces.calls == 1
    assert not captions(layer_project)["noop"]
    assert state(layer_project, "captions")["state"] == "done"
    assert measured_faces.calls == len(fake_layers) == 2


def test_unmeasured_faces_keep_every_phrase_and_receipt(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    snapshot = caption_snapshot(ProjectContext(layer_project))
    assert all(slot.get("offset_px", 0) == 0
               for slot in json.loads(snapshot.plan.variables["phrases"]))
    captions(layer_project)
    assert read_receipt(layer_project / "captions")["caption_placement"] == {
        "faces": {"status": "unmeasured", "reason": "fake project has no face measurements",
                  "frame_sources": []}, "phrases": [], "kept_phrases": 2,
    }


@pytest.mark.parametrize("kind", ["zoom", "insert", "chapter", "transition", "background"])
def test_picture_mapping_applies_when_captions_render(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict], kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captions(layer_project)
    if kind == "zoom":
        element = {"id": "agent-zoom", "kind": "zoom", "start": {"time": 0},
                   "hold_s": 2.5, "peak_scale": 1.05, "center": [0.5, 0.5]}
    elif kind in {"insert", "background"}:
        element = {"id": "broll-one", "kind": kind, "start": {"time": 0},
                   "hold_s": 1, "media": {"path": "source.mov", "in_s": 0,
                                           "sha256": receipt.sha256_file(
                                               layer_project / "source.mov")}}
        if kind == "background":
            element.update(layout="bottom", enter="wipe")
    elif kind == "chapter":
        element = {"id": "agent-ch", "kind": "chapter", "start": {"time": 0},
                   "hold_s": 1, "title": "New section"}
    if kind == "transition":
        timeline = ProjectContext(layer_project).resolved_timeline()
        transition = ResolvedElement("agent-transition", "transition", "agent", 0, 30, "front",
                                     {"look": "cinematic_zoom", "frames": 30, "join_frame": 15})
        monkeypatch.setattr(ProjectContext, "resolved_timeline",
                            lambda self, **kwargs: replace(timeline, elements=(transition,)))
    else:
        set_elements(layer_project, element,
                     lane="broll" if kind in {"insert", "background"} else "agent")
    assert state(layer_project, "captions")["state"] == "stale"
    assert not captions(layer_project)["noop"]
    placement = read_receipt(layer_project / "captions")["caption_placement"]
    assert placement["faces"]["frame_sources"][0]["source"] == (
        "zoom_estimate" if kind == "zoom" else "cut" if kind == "chapter" else
        "background_wipe" if kind == "background" else kind
    )
    if kind == "background":
        assert placement["faces"]["frame_sources"][:2] == [
            {"start_frame": 0, "end_frame": 9, "source": "background_wipe"},
            {"start_frame": 9, "end_frame": 30, "source": "background_layout"},
        ]
    if kind in {"insert", "transition"}:
        assert placement["phrases"] == []
    elif kind == "chapter":
        assert placement["phrases"] and placement["phrases"][0]["offset_px"]
    assert measured_faces.calls == 1 and len(fake_layers) == 2


@pytest.mark.parametrize("change", ["add", "remove", "move", "scale", "center"])
def test_zoom_edit_stales_captions_and_recalculates_placement(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict], change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placed: list[FacesData] = []
    offsets = caption_stage.caption_offsets

    def place(*args: Any, **kwargs: Any) -> list[dict]:
        placed.append(args[2])
        return offsets(*args, **kwargs)

    monkeypatch.setattr(caption_stage, "caption_offsets", place)
    zoom = {"id": "agent-zoom", "kind": "zoom", "start": {"time": 0},
            "hold_s": 2.5, "peak_scale": 1.08, "center": [0.5, 0.5]}
    if change != "add":
        set_elements(layer_project, zoom)
    captions(layer_project)
    before = read_receipt(layer_project / "captions")
    if change == "remove":
        set_elements(layer_project)
    else:
        if change == "move":
            zoom["start"] = {"time": 0.5}
        elif change == "scale":
            zoom["peak_scale"] = 1.10
        elif change == "center":
            zoom["center"] = [0.5, 0.9]
        set_elements(layer_project, zoom)
    assert state(layer_project, "captions")["state"] == "stale"
    assert not captions(layer_project)["noop"]
    after = read_receipt(layer_project / "captions")
    assert after["inputs"]["picture:faces"] != before["inputs"]["picture:faces"]
    assert len(placed) == 2 and placed[1].frames != placed[0].frames
    if change != "scale":
        assert after["caption_placement"]["phrases"] != before["caption_placement"]["phrases"]
    assert measured_faces.calls == 1 and len(fake_layers) == 2
    assert state(layer_project, "captions")["state"] == "done"


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("face_x", [100, 500])
@pytest.mark.parametrize("highlight", [
    "none", "moving_block", "underline", "color", "pill", "gradient", "editorial",
    "slam", "weight_shift", "clip_wipe",
])
@pytest.mark.parametrize("scale", [0.6, 1])
def test_fitted_ground_placement_uses_each_phrases_width(
    mode: str, face_x: int, highlight: str, scale: float,
) -> None:
    from yap_studio.style import resolve_style_data

    data = resolve_style_data({"schema_version": 1, "captions": {
        "background_mode": mode, "size_px": 48, "highlight_style": highlight,
        "width_scale": scale,
        "shadow_opacity": 0, "italic_stress": False, "entry_ms": 0,
    }})
    geometry = caption_band_geometry(data["captions"], safe_bottom_px=1740)
    frames = tuple((FaceBox(Box(face_x, geometry.y, 100, 100), 0.9, "detected"),)
                   for _ in range(30))
    faces = FacesData("measured", None,
                      VideoIdentity("cut.mp4", "a" * 64, 30, Fraction(30), 1080, 1920),
                      {}, {}, frames, tuple(find_gaps(frames)))
    phrases = [{"start": 0.2, "end": 0.7, "words": [{"text": text}]} for text in (
        "Hi", "W" * 50, "Hi",
    )]
    placements = caption_offsets(phrases, geometry, faces, data)
    if mode == "box" and face_x == 100:
        assert [item["offset_px"] == 0 for item in placements] == [True, False, True]
        assert placements[0]["reason"] == "kept: clear of the head"
    else:
        assert all(item["offset_px"] != 0 for item in placements)
    for item in placements:
        if item["offset_px"]:
            assert geometry.y + item["offset_px"] >= 640
            assert geometry.y + item["offset_px"] + geometry.height <= 1740


@pytest.mark.parametrize("case", ["no_face", "clear", "unmeasured", "blocked", "boundary"])
def test_caption_offsets_keep_clear_empty_and_blocked_windows(case: str) -> None:
    style = resolve("preset:electric")
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    face = Box(400, 300, 200, 180) if case == "clear" else Box(0, 0, 1080, 1920)
    frames = tuple((FaceBox(face, 0.9, "detected"),) if case != "no_face" else ()
                   for _ in range(30))
    if case == "boundary":
        frames = tuple(faces if n in (6, 21) else () for n, faces in enumerate(frames))
    data = FacesData("measured", None,
                     VideoIdentity("cut.mp4", "a" * 64, 30, Fraction(30), 1080, 1920),
                     {}, {}, frames, tuple(find_gaps(frames)))
    if case == "unmeasured":
        data = replace(data, status="unmeasured", reason="test unavailable", frames=(), gaps=())
    placements = caption_offsets([{"start": 0.2, "end": 0.7, "words": [{"text": "Hi"}]}],
                                 geometry, data, style)
    assert placements[0]["offset_px"] == 0
    assert placements[0]["reason"].startswith({
        "no_face": "kept: no face detected", "clear": "kept: clear of the head",
        "unmeasured": "kept: faces not measured: test unavailable",
        "blocked": "kept: no position in the band", "boundary": "kept: no face detected",
    }[case])


def test_blocked_collision_is_listed_in_receipt(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict],
) -> None:
    measured_faces.box = Box(0, 0, 1080, 1920)
    captions(layer_project)
    placement = read_receipt(layer_project / "captions")["caption_placement"]
    assert placement["phrases"][0]["offset_px"] == 0
    assert placement["phrases"][0]["reason"].startswith("kept: no position in the band")
    assert placement["kept_phrases"] == 2


@pytest.fixture
def layout_plan() -> LayerPlan:
    """Provide exact rendered slots in a placed layer with an output-time offset."""
    return LayerPlan("captions", 13, 1400, 1080, 500, 60, 300, {
        "fps": 30, "phrases": json.dumps([
            {"start": 0.2, "end": 0.9, "words": [
                {"id": "t1:0", "source_ids": ["t1:0"], "text": "Écrire",
                 "start": 0.201234, "end": 0.45},
                {"id": "t1:1", "source_ids": ["t1:1"], "text": "documentation",
                 "start": 0.456789, "end": 0.8},
            ]},
            {"start": 1.0, "end": 1.5, "words": [
                {"id": "t1:2", "source_ids": ["t1:2"], "text": "OK", "start": 1.02, "end": 1.4},
            ]},
        ], ensure_ascii=False),
    })


@pytest.mark.parametrize("raw", [False, True])
def test_flat_caption_layout_retains_rendered_word_ids(raw) -> None:
    style = resolve("preset:clean_minimal")
    words = [CaptionWord("t1:8", "Hello", 0, .3), CaptionWord("t1:9", "again", .4, .7)]
    phrases = build_phrases(words, style.data["captions"])
    if raw:
        phrases = [replace(phrases[0], words=tuple(words))]
    plan = caption_stage.caption_layer_plan(phrases, style, frames=90)
    slots = json.loads(plan.variables["phrases"])
    layout = caption_layout(_layout_checks([
        {"slot": i, "boxes": [{"x": 100, "y": 20, "w": 80, "h": 40}]}
        for i in range(len(slots))
    ]), plan)
    assert layout["measured"]
    assert [word["id"] for phrase in layout["captions"] for word in phrase["words"]] == [
        "t1:8", "t1:9"]
    assert [word["source_ids"] for phrase in layout["captions"] for word in phrase["words"]] == [
        ["t1:8"], ["t1:9"]]


def test_caption_layout_maps_first_summary_to_output_boxes_words_and_frames(
    layout_plan: LayerPlan,
) -> None:
    checks = _layout_checks([
        {"slot": 1, "boxes": [{"x": 10, "y": 20, "w": 0, "h": 0}]},
        {"slot": 0, "boxes": [
            {"x": -25.556, "y": 2.347, "w": 100.123, "h": 50.567},
            {"x": 120, "y": 3, "w": 80, "h": 50},
        ]},
    ])
    findings = checks["passes"]["runtime"]["findings"]
    findings.insert(0, None)
    findings.insert(0, {"message": "Ordinary warning"})
    findings.append({"message": layer_checks.CAPTION_SHRINK_PREFIX + "{"})
    before = json.dumps(checks), json.dumps(layout_plan.variables)
    assert caption_layout(checks, layout_plan) == {
        "schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
        "measured": True, "reason": None, "elements": [], "captions": [
            {"id": "caption-0", "text": "Écrire documentation", "start_frame": 67,
             "end_frame": 87, "boxes": [
                 {"x": -12.56, "y": 1402.35, "w": 100.12, "h": 50.57},
                 {"x": 133, "y": 1403, "w": 80, "h": 50},
             ], "words": [
                 {"id": "t1:0", "source_ids": ["t1:0"], "text": "Écrire",
                  "start_s": 2.201, "end_s": 2.45},
                 {"id": "t1:1", "source_ids": ["t1:1"], "text": "documentation",
                  "start_s": 2.457, "end_s": 2.8},
             ]},
            {"id": "caption-1", "text": "OK", "start_frame": 91, "end_frame": 105,
             "boxes": [{"x": 23, "y": 1420, "w": 0, "h": 0}],
             "words": [{"id": "t1:2", "source_ids": ["t1:2"], "text": "OK",
                        "start_s": 3.02, "end_s": 3.4}]},
        ],
    }
    assert (json.dumps(checks), json.dumps(layout_plan.variables)) == before


def test_caption_layout_round_trips_through_clearance_schema(layout_plan: LayerPlan) -> None:
    checks = _layout_checks([
        {"slot": i, "boxes": [{"x": 100, "y": 20, "w": 80, "h": 40}]}
        for i in range(2)
    ])
    document = caption_layout(checks, layout_plan)
    assert document["measured"] is True and len(document["captions"]) == 2
    layout = Layout.from_dict(document)
    assert (layout.width, layout.height, layout.elements) == (1080, 1920, ())
    portable = layout.to_dict()
    assert portable["captions"] == [
        {key: value for key, value in phrase.items() if key != "words"}
        for phrase in document["captions"]
    ]
    assert Layout.from_dict(portable) == layout


@pytest.mark.parametrize(("start", "end", "offset", "window"), [
    (7.2, 8.0, 0, (217, 240)),
    (7.1, 7.2, 0, (214, 216)),
    (7.2 - 1e-10, 7.3, 0, (217, 219)),
    (7.1, 7.2 + 1e-10, 0, (214, 216)),
    (0.5 / 30, 10.5 / 30, 60, (61, 71)),
])
def test_caption_layout_uses_open_time_edges_on_the_output_frame_grid(
    layout_plan: LayerPlan, start: float, end: float, offset: int, window: tuple[int, int],
) -> None:
    slot = {**json.loads(layout_plan.variables["phrases"])[0], "start": start, "end": end}
    plan = replace(layout_plan, start_frame=offset, variables={
        **layout_plan.variables, "phrases": json.dumps([slot]),
    })
    document = caption_layout(_layout_checks([
        {"slot": 0, "boxes": [{"x": 100, "y": 20, "w": 80, "h": 40}]},
    ]), plan)
    assert document["measured"] is True
    phrase = document["captions"][0]
    assert (phrase["start_frame"], phrase["end_frame"]) == window


@pytest.mark.parametrize("empty_boxes", [False, True])
def test_caption_layout_omits_empty_frame_windows(
    layout_plan: LayerPlan, empty_boxes: bool,
) -> None:
    slots = json.loads(layout_plan.variables["phrases"])
    slots[0].update(start=7.2, end=7.21)
    plan = replace(layout_plan, variables={**layout_plan.variables, "phrases": json.dumps(slots)})
    document = caption_layout(_layout_checks([
        {"slot": i, "boxes": [] if i == 0 and empty_boxes else [
            {"x": 100, "y": 20, "w": 80, "h": 40},
        ]}
        for i in range(2)
    ]), plan)
    assert document["measured"] is True and document["reason"] is None
    assert [phrase["id"] for phrase in document["captions"]] == ["caption-1"]
    assert len(Layout.from_dict(document).captions) == 1


def test_caption_layout_empty_slots_are_measured(layout_plan: LayerPlan) -> None:
    plan = replace(layout_plan, variables={**layout_plan.variables, "phrases": "[]"})
    document = caption_layout(_layout_checks([]), plan)
    assert document == {"schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
                        "measured": True, "reason": None, "elements": [], "captions": []}
    assert Layout.from_dict(document).captions == ()


@pytest.mark.parametrize(("checks", "reason"), [
    (None, "Layer checks are unavailable."),
    ({"passes": {"runtime": {"status": "skipped", "reason": "browser could not start"}}},
     "browser could not start"),
    ({"passes": {"runtime": {"status": "skipped"}}}, "Runtime checks were skipped."),
    ({"passes": {"runtime": {"status": "passed", "findings": []}}},
     "Caption shrink summary is missing."),
    ({"passes": {}}, "Caption shrink summary is missing."),
    ({"passes": None}, "Caption shrink summary is missing."),
    ({"passes": {"runtime": None}}, "Caption shrink summary is missing."),
    ({"passes": {"runtime": {"findings": {}}}}, "Caption shrink summary is missing."),
    (_shrink_checks([]), "Caption layout is missing."),
])
def test_caption_layout_without_evidence_is_not_measured(
    layout_plan: LayerPlan, checks: dict | None, reason: str,
) -> None:
    assert caption_layout(checks, layout_plan) == {
        "schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
        "measured": False, "reason": reason, "elements": [], "captions": [],
    }


@pytest.mark.parametrize("message", [
    "{", "null", "[]", "true", '{"layout": null}', '{"layout": {}}', '{"layout": []}',
    *[json.dumps({"layout": [
        {"slot": slot, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1}]},
        {"slot": 1, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1}]},
    ]}) for slot in (2, -1, True, 0.0, "0", 1, None)],
    *[json.dumps({"layout": [
        {"slot": 0, "boxes": boxes},
        {"slot": 1, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1}]},
    ]}) for boxes in ([], None, {}, [None], [{}])],
    *[json.dumps({"layout": [
        {"slot": 0, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1, key: value}]},
        {"slot": 1, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1}]},
    ]}) for key in ("x", "y", "w", "h")
      for value in (True, "1", float("nan"), float("inf"), -float("inf"), None)],
    *[json.dumps({"layout": [
        {"slot": 0, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1, key: -1}]},
        {"slot": 1, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1}]},
    ]}) for key in ("w", "h")],
    '{"layout": [{"slot": 0, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1}]}]}',
    '{"layout": [null, null]}',
])
def test_caption_layout_invalid_summary_is_not_measured(
    layout_plan: LayerPlan, message: str,
) -> None:
    checks = _layout_checks([])
    checks["passes"]["runtime"]["findings"][0]["message"] = (
        layer_checks.CAPTION_SHRINK_PREFIX + message
    )
    assert caption_layout(checks, layout_plan) == {
        "schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
        "measured": False, "reason": "Caption layout is invalid.", "elements": [], "captions": [],
    }


@pytest.fixture
def reported_layers(
    fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict | None], None]:
    """Attach configurable runtime evidence to the existing alpha renderer fake."""
    render = caption_stage.render_layer_plan

    def configure(checks: dict | None) -> None:
        def reported(plan: LayerPlan, work_dir: Path, out: Path,
                     **kwargs: Any) -> hyperframes.LayerResult:
            return replace(render(plan, work_dir, out, **kwargs), checks=checks)
        monkeypatch.setattr(caption_stage, "render_layer_plan", reported)

    return configure


@pytest.fixture
def shrink_plan() -> LayerPlan:
    """Provide two caption slots without creating media or a project."""
    return LayerPlan("captions", 0, 0, 1080, 500, 0, 90, {
        "size_px": 180, "phrases": json.dumps([
            {"start": 0.2, "end": 0.9, "words": [{"text": "Write"}, {"text": "documentation"}]},
            {"start": 1.0, "end": 1.5, "words": [{"text": "OK"}]},
        ]),
    })


@pytest.mark.parametrize("floor_px", [32, 32.0])
def test_caption_shrink_maps_first_summary_in_slot_order(
    shrink_plan: LayerPlan, floor_px: int | float,
) -> None:
    checks = _shrink_checks([
        {"slot": 1, "size_px": 32, "fits": False},
        {"slot": 0, "size_px": 140.25, "fits": True},
    ], floor_px=floor_px)
    findings = checks["passes"]["runtime"]["findings"]
    findings.insert(0, {"message": "Ordinary warning"})
    findings.append({"message": layer_checks.CAPTION_SHRINK_PREFIX + "{"})
    before = json.dumps(checks), json.dumps(shrink_plan.variables)
    assert caption_shrink(checks, shrink_plan) == {
        "measured": True, "floor_px": floor_px, "style_size_px": 180, "phrases": [
            {"start_s": 0.2, "end_s": 0.9, "text": "Write documentation",
             "size_px": 140.25, "fits": True},
            {"start_s": 1.0, "end_s": 1.5, "text": "OK", "size_px": 32, "fits": False},
        ],
    }
    assert (json.dumps(checks), json.dumps(shrink_plan.variables)) == before


def test_caption_shrink_empty_summary_is_measured(shrink_plan: LayerPlan) -> None:
    assert caption_shrink(_shrink_checks([]), shrink_plan) == {
        "measured": True, "floor_px": 32, "style_size_px": 180, "phrases": [],
    }


@pytest.mark.parametrize(("checks", "reason"), [
    (None, "Layer checks are unavailable."),
    ({"passes": {"runtime": {"status": "skipped", "reason": "browser could not start"}}},
     "browser could not start"),
    ({"passes": {"runtime": {"status": "passed", "findings": []}}},
     "Caption shrink summary is missing."),
    ({"passes": {}}, "Caption shrink summary is missing."),
])
def test_caption_shrink_without_runtime_summary_is_not_measured(
    shrink_plan: LayerPlan, checks: dict | None, reason: str,
) -> None:
    assert caption_shrink(checks, shrink_plan) == {"measured": False, "reason": reason}


@pytest.mark.parametrize("message", [
    "{", "null", '{"floor_px": 32, "phrases": null}',
    *[json.dumps({"floor_px": 32, "phrases": [{"slot": slot, "size_px": 100, "fits": True}]})
      for slot in (2, -1, True, 0.0, "0")],
    '{"floor_px": "32", "phrases": []}',
    '{"floor_px": NaN, "phrases": []}',
    '{"floor_px": 32, "phrases": [{"slot": 0, "size_px": "100", "fits": true}]}',
    '{"floor_px": 32, "phrases": [{"slot": 0, "size_px": 100, "fits": "true"}]}',
])
def test_caption_shrink_invalid_summary_is_not_measured(
    shrink_plan: LayerPlan, message: str,
) -> None:
    checks = _shrink_checks([])
    checks["passes"]["runtime"]["findings"][0]["message"] = (
        layer_checks.CAPTION_SHRINK_PREFIX + message
    )
    assert caption_shrink(checks, shrink_plan) == {
        "measured": False, "reason": "Caption shrink summary is invalid.",
    }


@pytest.mark.parametrize("fits", [True, False])
def test_project_receipt_records_caption_shrink(
    layer_project: Path, reported_layers: Callable[[dict | None], None], fits: bool,
) -> None:
    size_px = 40.25 if fits else 32
    reported_layers(_shrink_checks([{"slot": 1, "size_px": size_px, "fits": fits}]))
    result = captions(layer_project)
    assert "caption_shrink" not in result
    assert read_receipt(layer_project / "captions")["caption_shrink"] == {
        "measured": True, "floor_px": 32, "style_size_px": 48, "phrases": [
            {"start_s": 1.0, "end_s": 2.6, "text": "More words.",
             "size_px": size_px, "fits": fits},
        ],
    }


@pytest.mark.parametrize("checks", [
    None,
    {"passes": {"runtime": {"status": "skipped", "reason": "browser could not start"}}},
])
def test_project_receipt_records_unmeasured_caption_shrink(
    layer_project: Path, reported_layers: Callable[[dict | None], None], checks: dict | None,
) -> None:
    reported_layers(checks)
    captions(layer_project)
    shrink = read_receipt(layer_project / "captions")["caption_shrink"]
    assert shrink == {"measured": False, "reason": (
        "Layer checks are unavailable." if checks is None else "browser could not start"
    )}
    layout = json.loads((layer_project / "captions/layout.json").read_text())
    assert layout == {"schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
                      **shrink, "elements": [], "captions": []}
    assert read_receipt(layer_project / "captions")["caption_layout"] == {
        **shrink, "captions": 0,
    }


@pytest.mark.parametrize("measured", [False, True])
def test_standalone_receipt_records_caption_shrink(
    clip: Path, words: Path, tmp_path: Path,
    reported_layers: Callable[[dict | None], None], measured: bool,
) -> None:
    reported_layers(_shrink_checks([{"slot": 0, "size_px": 32, "fits": False}])
                    if measured else None)
    style = resolve("preset:electric")
    result = caption_stage.caption_clip(clip, style=style, out=tmp_path / "captioned.mp4",
                                        words=words, title=None, srt=False, force=False)
    assert "caption_shrink" not in result
    shrink = json.loads(Path(result["receipt"]).read_text())["caption_shrink"]
    assert shrink == ({
        "measured": True, "floor_px": 32, "style_size_px": style.data["captions"]["size_px"],
        "phrases": [{"start_s": 0.1, "end_s": 0.64, "text": "First words.",
                     "size_px": 32, "fits": False}],
    } if measured else {"measured": False, "reason": "Layer checks are unavailable."})


@pytest.mark.parametrize("late_gap", [False, True])
def test_last_caption_keeps_natural_tail_before_trailing_picture(
    layer_project: Path, late_gap: bool,
) -> None:
    republish_cut(layer_project, frames=900)
    if late_gap:
        set_elements(layer_project, {"id": "agent-gap", "kind": "caption_gap",
                                    "start": {"time": 20}, "hold_s": 1})
    snapshot = caption_snapshot(ProjectContext(layer_project))
    assert len(snapshot.phrases) == 2
    assert snapshot.phrases[-1].end == pytest.approx(2.6)
    assert snapshot.plan.frames == 900


def test_natural_caption_tail_covering_cut_stays_visible_for_hold(
    tmp_path: Path,
) -> None:
    """The hold may retain a caption when its natural phrase already reaches the last frame."""
    project = make_project(tmp_path)
    snapshot = caption_snapshot(ProjectContext(project))
    assert snapshot.phrases[-1].end == 3.0
    result = captions(project)
    alpha = _alpha_frames(project / result["out"], snapshot.plan)
    assert alpha[-1].max() > 0


def test_legacy_extended_caption_receipt_requires_a_new_render(layer_project: Path) -> None:
    """An unchanged timeline must not reuse a layer made with the old tail policy."""
    captions(layer_project)
    target = layer_project / "captions/receipt.json"
    record = json.loads(target.read_bytes())
    record["inputs"]["cut/cut-words.json"] = projection_sha256({
        "sha256": hashlib.sha256((layer_project / "cut/cut-words.json").read_bytes()).hexdigest(),
        "frames": 90,
    })
    write(target, record)
    assert state(layer_project, "captions")["state"] == "stale"
    assert not captions(layer_project)["noop"]


def test_caption_outputs_use_cut_frames_output_times_and_last_frame(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict],
) -> None:
    snapshot = caption_snapshot(ProjectContext(layer_project))
    result = captions(layer_project)
    assert fake_layers[0]["workers"] == "auto"
    layer = result["layer"]
    info = ffmpeg.probe(layer_project / result["out"])
    assert (info.width, info.height, info.video_frames) == (layer["width"], layer["height"], 90)
    assert layer["width"] == 1080 and layer["height"] < 1920 and layer["start_frame"] == 0
    assert (info.color_space, info.color_transfer, info.color_primaries) == ("bt709",) * 3
    assert info.pix_fmt.startswith("yuva444p")
    phrases = json.loads((layer_project / "captions/phrases.json").read_bytes())
    assert phrases["layer"] == layer and phrases["time_base"] == "output"
    assert [phrase["text"] for phrase in phrases["phrases"]] == ["simple steps.", "More words."]
    assert [(phrase["start"], phrase["end"]) for phrase in phrases["phrases"]] == [
        (0.1, 0.94), (1.0, 2.6),
    ]
    assert (layer_project / "captions/captions.srt").read_text() == (
        "1\n00:00:00,100 --> 00:00:00,940\nsimple steps.\n\n"
        "2\n00:00:01,000 --> 00:00:02,600\nMore words.\n\n"
    )
    alpha = _alpha_frames(layer_project / result["out"], snapshot.plan)
    assert alpha[77].max() > 0 and alpha[89].max() == 0 and alpha[29].max() == 0
    record = read_receipt(layer_project / "captions")
    assert record["inputs"] == snapshot.inputs and record["layer"] == layer
    assert record["layer_checks"] == checked_layers[0]
    assert record["layer_checks"]["renderer"]["workers"] == "auto"
    assert record["layer_checks"]["passes"]["lint"]["status"] == "warnings"
    assert record["layer_checks"]["passes"]["layout"]["status"] == "passed"
    assert record["layer_checks"]["passes"]["motion"]["status"] == "skipped"
    assert {item["path"] for item in record["outputs"]} == {
        "captions/captions.mov", "captions/captions.srt", "captions/phrases.json",
        "captions/layout.json",
    }
    assert result["layout"] == "captions/layout.json"
    layout_text = (layer_project / result["layout"]).read_text()
    layout = json.loads(layout_text)
    assert layout_text == json.dumps(layout, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    assert layout == {
        "schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
        "measured": True, "reason": None, "elements": [], "captions": [
            {"id": "caption-0", "text": "simple steps.", "start_frame": 4, "end_frame": 29,
             "boxes": [{"x": 100, "y": layer["y"] + 20, "w": 80, "h": 40},
                       {"x": 190, "y": layer["y"] + 20, "w": 80, "h": 40}],
             "words": [{"id": "t1:0", "source_ids": ["t1:0"], "text": "simple",
                        "start_s": 0.1, "end_s": 0.2},
                       {"id": "t1:1", "source_ids": ["t1:1"], "text": "steps.",
                        "start_s": 0.3, "end_s": 0.45}]},
            {"id": "caption-1", "text": "More words.", "start_frame": 31, "end_frame": 78,
             "boxes": [{"x": 100, "y": layer["y"] + 20, "w": 80, "h": 40},
                       {"x": 190, "y": layer["y"] + 20, "w": 80, "h": 40}],
             "words": [{"id": "t1:2", "source_ids": ["t1:2"], "text": "More",
                        "start_s": 1.0, "end_s": 1.1},
                       {"id": "t1:3", "source_ids": ["t1:3"], "text": "words.",
                        "start_s": 1.2, "end_s": 1.4}]},
        ],
    }
    assert record["caption_layout"] == {"measured": True, "reason": None, "captions": 2}
    assert len(Layout.from_dict(layout).captions) == 2
    assert state(layer_project, "captions")["state"] == "done"
    assert len(fake_layers) == 1


def test_bottom_caption_clamp_warning_survives_project_receipt_and_noop(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_text())
    style["captions"].update({
        "position": "bottom", "top_px": 1650, "size_px": 104,
        "shadow_blur_px": 5, "shadow_x_px": 3, "shadow_y_px": 4, "entry_rise_px": 12,
    })
    write(style_path, style)
    result = captions(layer_project)
    layer = result["layer"]
    warnings = layer["warnings"]
    assert len(warnings) == 1 and "clamped" in warnings[0]
    assert "1650" in warnings[0] and "1740" in warnings[0]
    assert layer["y"] + layer["height"] == 1740
    record = read_receipt(layer_project / "captions")
    assert record["layer"]["warnings"] == warnings
    phrases = json.loads((layer_project / "captions/phrases.json").read_text())
    assert phrases["layer"]["warnings"] == warnings
    repeat = captions(layer_project)
    assert repeat["noop"] and repeat["layer"]["warnings"] == warnings
    assert len(fake_layers) == 1


def test_unrelated_timeline_elements_do_not_stale_captions(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    captions(layer_project)
    before = (layer_project / "captions/receipt.json").read_bytes()
    element = {"id": "agent-card", "kind": "graphic", "start": {"time": 0.8},
               "hold_s": 1, "template": "text_tag", "slots": {"text": "Sample"}}
    set_elements(layer_project, element)
    assert state(layer_project, "captions")["state"] == "done"
    assert captions(layer_project)["noop"]
    assert (layer_project / "captions/receipt.json").read_bytes() == before
    assert len(fake_layers) == 1


@pytest.mark.parametrize("change", ["style", "margin", "fix", "gap", "renderer"])
def test_consumed_inputs_stale_and_rerender_captions(
    layer_project: Path, fake_layers: list[dict], monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    captions(layer_project)
    if change == "style":
        data = json.loads((layer_project / "style.json").read_bytes())
        data["captions"]["highlight_style"] = "color"
        write(layer_project / "style.json", data)
    elif change == "margin":
        assert not ProjectContext(layer_project).resolved_timeline().elements
        data = json.loads((layer_project / "style.json").read_bytes())
        data["graphics"] = {"face_clearance_px": 48}
        write(layer_project / "style.json", data)
    elif change == "fix":
        data = json.loads((layer_project / "project.json").read_bytes())
        data["word_fixes"] = {"small": "short"}
        write(layer_project / "project.json", data)
    elif change == "gap":
        set_elements(layer_project, {"id": "agent-gap", "kind": "caption_gap",
                                    "start": {"time": 0.2}, "hold_s": 0.1})
    else:
        monkeypatch.setattr(hyperframes, "HYPERFRAMES_VERSION", "test-version")
    assert state(layer_project, "captions")["state"] == "stale"
    assert not captions(layer_project)["noop"]
    assert len(fake_layers) == 2 and state(layer_project, "captions")["state"] == "done"


@pytest.mark.parametrize("start,hold,count", [(0.2, 0.1, 1), (2.9, 0.1, 2), (0.0, 3.0, 0)])
def test_whole_phrases_touching_caption_gaps_are_absent(
    layer_project: Path, fake_layers: list[dict], start: float, hold: float, count: int,
) -> None:
    set_elements(layer_project, {"id": "agent-gap", "kind": "caption_gap",
                                "start": {"time": start}, "hold_s": hold})
    result = captions(layer_project)
    slots = json.loads(fake_layers[0]["variables"]["phrases"])
    document = json.loads((layer_project / "captions/phrases.json").read_bytes())
    assert len(slots) == len(document["phrases"]) == result["phrases"] == count
    alpha = _alpha_frames(layer_project / result["out"], caption_snapshot(
        ProjectContext(layer_project)).plan)
    if start == 0.2:
        assert alpha[12].max() == 0 and alpha[45].max() > 0
        assert "simple steps." not in (layer_project / "captions/captions.srt").read_text()
    else:
        assert alpha[89].max() == 0
        assert ("More words." in (layer_project / "captions/captions.srt").read_text()) == (
            start == 2.9
        )


def test_force_renders_but_refuses_a_hand_edited_layer(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    first = captions(layer_project)
    assert captions(layer_project)["noop"]
    assert not captions(layer_project, force=True)["noop"] and len(fake_layers) == 2
    target = layer_project / first["out"]
    target.write_bytes(b"external replacement")
    with pytest.raises(OutputPathError):
        captions(layer_project, force=True)
    assert target.read_bytes() == b"external replacement"


def test_renderer_blocked_leaves_recoverable_receipt(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = hyperframes.render_layer
    with monkeypatch.context() as patch:
        def blocked(*args: Any, **kwargs: Any) -> None:
            raise RendererBlockedError()
        patch.setattr(hyperframes, "render_layer", blocked)
        with pytest.raises(RendererBlockedError):
            captions(layer_project)
    assert state(layer_project, "captions")["reason"] == "interrupted"
    assert not (layer_project / "captions/captions.mov").exists()
    assert hyperframes.render_layer is renderer
    assert not captions(layer_project)["noop"]


def test_render_from_snapshot_is_stale_after_a_mid_render_edit(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    render = hyperframes.render_layer
    before = captions_inputs(ProjectContext(layer_project))

    def edit(composition: Path, out: Path, *, variables: Mapping[str, object],
             **kwargs: Any) -> hyperframes.LayerResult:
        assert variables["highlight_style"] == "underline"
        data = json.loads((layer_project / "style.json").read_bytes())
        data["captions"]["highlight_style"] = "color"
        write(layer_project / "style.json", data)
        return render(composition, out, variables=variables, **kwargs)

    monkeypatch.setattr(hyperframes, "render_layer", edit)
    captions(layer_project)
    assert read_receipt(layer_project / "captions")["inputs"] == before
    assert state(layer_project, "captions")["state"] == "stale"


def test_saved_style_is_read_once_for_timeline_and_render(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    reads = []

    def observe(target: Path) -> bytes:
        if target == layer_project / "style.json":
            reads.append(target)
        return original(target)

    monkeypatch.setattr(Path, "read_bytes", observe)
    captions(layer_project)
    assert reads == [layer_project / "style.json"]


def test_invalid_timeline_refused_before_outputs(layer_project: Path) -> None:
    set_elements(layer_project, {"id": "agent-gap", "kind": "caption_gap",
                                "start": {"time": 2.9}, "hold_s": 1})
    with pytest.raises(TimelineRefusal):
        captions(layer_project)
    assert not (layer_project / "captions").exists()


def republish_cut(root: Path, *, frames: int = 90, shift: float = 0) -> None:
    """Publish a changed synthetic cut clock with the original source-word identities."""
    data = json.loads((root / "cut/cut-words.json").read_bytes())
    for word in data["words"]:
        word["start"] += shift
        word["end"] += shift
    inputs = cut_inputs(ProjectContext(root))
    staged = write(root / "staged-cut-words.json", data)
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "cut/cut-words.json", staged)
        run.finish(extra={"plan": {"total_frames": frames}})


@pytest.mark.parametrize("change", ["word-times", "frames"])
def test_recut_stales_captions_for_word_times_or_changed_cut_length(
    layer_project: Path, change: str,
) -> None:
    captions(layer_project)
    republish_cut(layer_project, shift=0.1 if change == "word-times" else 0,
                  frames=93 if change == "frames" else 90)
    assert state(layer_project, "captions")["state"] == "stale"
    result = captions(layer_project)
    assert not result["noop"] and result["frames"] == (93 if change == "frames" else 90)


def test_changed_end_hold_does_not_extend_or_invalidate_layer(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    captions(layer_project)
    data = json.loads((layer_project / "edit.json").read_bytes())
    data["end_hold_s"] = 3
    write(layer_project / "edit.json", data)
    republish_cut(layer_project)
    assert state(layer_project, "captions")["state"] == "done"
    result = captions(layer_project)
    assert result["noop"] and result["frames"] == 90 and len(fake_layers) == 1


@pytest.mark.parametrize("spec", ["preset:electric", "sample_style"])
def test_preset_and_library_style_bytes_are_shared_with_timeline(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, spec: str,
) -> None:
    from yap_studio.style import PRESETS_DIR

    config = layer_project / "style-library"
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(config))
    if spec == "sample_style":
        target = config / "styles/sample_style.json"
        data = json.loads((layer_project / "style.json").read_bytes())
        write(target, data)
    else:
        target = PRESETS_DIR / "electric.json"
    data = json.loads((layer_project / "project.json").read_bytes())
    data["style"] = spec
    write(layer_project / "project.json", data)
    original = Path.read_bytes
    reads = []

    def observe(current: Path) -> bytes:
        if current == target:
            reads.append(current)
        return original(current)

    monkeypatch.setattr(Path, "read_bytes", observe)
    context = ProjectContext(layer_project)
    snapshot = caption_snapshot(context)
    assert snapshot.plan.variables["font_name"] == context.render_style().data["captions"]["font"]
    assert reads == [target]


def test_punch_caption_clearance_uses_enlarged_faces(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"].update({"position": "middle", "top_px": 960})
    write(layer_project / "style.json", style)
    geometry = caption_band_geometry(
        ProjectContext(layer_project).render_style().data["captions"], safe_bottom_px=1740)
    measured_faces.box = Box(400, geometry.y + 40, 200, 180)
    placed = []
    offsets = caption_stage.caption_offsets

    def place(*args, **kwargs):
        placed.append(args[2])
        return offsets(*args, **kwargs)

    monkeypatch.setattr(caption_stage, "caption_offsets", place)
    captions(layer_project)
    before = read_receipt(layer_project / "captions")
    set_elements(layer_project, {"id": "agent-punch", "kind": "zoom", "start": {"time": 0},
                                 "hold_s": 2.5, "move": "punch", "scale": 1.3,
                                 "center": [.5, .5]})
    captions(layer_project)
    after = read_receipt(layer_project / "captions")
    assert after["inputs"]["picture:faces"] != before["inputs"]["picture:faces"]
    assert placed[1].frames[15][0].box.w == pytest.approx(
        placed[0].frames[15][0].box.w * 1.3 + 4)
    assert after["caption_placement"]["phrases"] != before["caption_placement"]["phrases"]


def test_punch_auto_center_keeps_bottom_caption_placement(
    layer_project: Path, measured_faces: _CaptionDetector, fake_layers: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"].update({"position": "bottom", "placement": "avoid_face",
                              "background_mode": "box"})
    write(layer_project / "style.json", style)
    geometry = caption_band_geometry(
        ProjectContext(layer_project).render_style().data["captions"], safe_bottom_px=1740)
    measured_faces.box = Box(400, geometry.y - 300, 200, 200)
    monkeypatch.setattr(face_project, "cut_faces", lambda project_dir, cut, **kwargs:
                        cut_faces(project_dir, cut, detector=measured_faces, **kwargs))
    set_elements(layer_project, {"id": "agent-punch", "kind": "zoom", "start": {"time": 0},
                                 "hold_s": 2.5, "move": "punch", "scale": 2})
    result = captions(layer_project)
    placement = read_receipt(layer_project / "captions")["caption_placement"]
    assert placement["faces"]["status"] == "measured"
    assert all(phrase["offset_px"] == 0 for phrase in placement["phrases"]), placement
    assert placement["kept_phrases"] == result["phrases"] == 2


def test_legacy_hidden_chapter_faces_require_new_caption_placement(layer_project: Path) -> None:
    set_elements(layer_project, {"id": "agent-ch", "kind": "chapter", "start": {"time": 0},
                                 "hold_s": 1, "title": "New section"})
    captions(layer_project)
    target = layer_project / "captions/receipt.json"
    record = json.loads(target.read_bytes())
    record["inputs"]["picture:faces"] = projection_sha256([
        {"id": "agent-ch", "kind": "chapter", "start_frame": 0, "end_frame": 30},
    ])
    write(target, record)
    assert state(layer_project, "captions")["state"] == "stale"
    assert not captions(layer_project)["noop"]
