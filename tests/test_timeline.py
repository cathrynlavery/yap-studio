"""Synthetic timeline anchors, rule checks, and independent writer transactions."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth

from yap_studio import cli, ffmpeg, receipt
from yap_studio.planner import cards
from yap_studio.project.lock import ConflictError
from yap_studio.project.model import load_cut_words, load_timeline, load_words
from yap_studio.project.receipts import StageRun
from yap_studio.project.status import ProjectContext, cut_inputs, ingest_inputs, music_inputs
from yap_studio.stages import timeline as stage
from yap_studio.stages.timeline import (
    TimelineRefusal,
    add_timeline,
    remove_timeline,
    resolve_timeline,
)
from yap_studio.style import StyleError, resolve_style_data


@pytest.mark.parametrize(("kind", "layer", "payload"), [
    ("zoom", "front", {"move": "opening"}), ("moment", "front", {}),
    ("background", "front", {}), ("graphic", "behind", {"template": "big_number"}),
    ("graphic", "front", {"template": "big_number"}),
    ("graphic", None, {"template": "big_number"}),
    ("overlay", "front", {"placement": "tall_behind"}),
    ("insert", "front", {"media": {"fit": "split"}}),
])
def test_camera_timeline_violation(kind, layer, payload) -> None:
    item = stage.ResolvedElement("agent-one", kind, "agent", 0, 30, layer, payload)
    findings = stage.camera_findings([item], {"jump_cuts": "hard"})
    assert len(findings) == 1 and findings[0].rule == "camera_3d"
    assert findings[0].severity == "violation" and findings[0].elements == (item.id,)
    assert findings[0].suggested_action == (
        "Remove those from the timeline, or pick another caption look.")


@pytest.mark.parametrize("move", ["punch", "smooth", "step_in"])
def test_camera_jump_cut_warning(move: str) -> None:
    findings = stage.camera_findings([], {"jump_cuts": move})
    assert len(findings) == 1 and findings[0].severity == "warning"
    assert findings[0].message == "3D camera captions move the camera, so jump cut moves are off."


@pytest.mark.parametrize("move", ["punch", "smooth", "step_in"])
def test_camera_resolve_skips_jump_cut_framing(project: Path, move: str) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "camera-style.json"
    write(project / "project.json", data)
    write(project / "camera-style.json", {"schema_version": 1, "captions": {"camera_3d": True},
                                          "moves": {"jump_cuts": move}})
    timeline = resolve_timeline(project)
    assert timeline.camera_3d and timeline.steps is None
    assert not any(item.lane == "engine" for item in timeline.elements)
    assert any(f.rule == "camera_3d" and f.severity == "warning" for f in timeline.findings)
    write(project / "camera-style.json", {"schema_version": 1})
    assert not any(f.rule == "camera_3d" for f in resolve_timeline(project).findings)


def test_camera_conflict_cli_sentences_and_run_fix(project: Path, monkeypatch, capsys) -> None:
    from yap_studio.render.camera3d_look import ELEMENT_FIX
    from yap_studio.stages import run as chain

    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "camera-style.json"
    write(project / "project.json", data)
    write(project / "camera-style.json", {"schema_version": 1, "captions": {"camera_3d": True}})
    write(project / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1",
        "words_sha256": {"t1": receipt.sha256_file(project / "ingest/t1/words.json")},
        "elements": [zoom(id="agent-first"),
                     zoom(id="agent-second", start={"time": 5}, end={"time": 7})],
    })
    assert cli.main(["timeline", "check", str(project)]) == 3
    output = capsys.readouterr().out
    for name in ("agent-first", "agent-second"):
        sentence = ("3D camera captions move the camera over your take, so they can't share "
                    f"a video with zoom: {name}.")
        assert output.count(sentence) == 1
    assert ELEMENT_FIX in output
    (project / "cut/cut.mp4").write_bytes(b"synthetic cut")
    for name in ("ingest", "fillers", "cut", "music"):
        monkeypatch.setitem(chain._HANDLERS, name, lambda *a: (True, "already checked"))
    assert cli.main(["run", str(project), "--json"]) == 3
    stopped = json.loads(capsys.readouterr().out)["stopped"]
    assert stopped["code"] == "camera3d_refused"
    assert stopped["suggested_action"] == ELEMENT_FIX
    assert "agent-first" in stopped["message"] and "agent-second" in stopped["message"]
    assert not (project / "captions/receipt.json").exists()
    assert cli.main(["run", str(project)]) == 3
    output = capsys.readouterr()
    assert ELEMENT_FIX in output.out + output.err


@pytest.mark.parametrize("look", ["giant", "none", "pill", "slam"])
@pytest.mark.parametrize("position", ["bottom", "middle"])
def test_caption_band_top_matches_rendered_layout(look: str, position: str) -> None:
    from yap_studio.captions import caption_layout_settings
    from yap_studio.stages.captions import caption_band_geometry

    style = resolve_style_data({"schema_version": 1, "captions": {
        "highlight_style": look, "size_px": 120, "position": position,
    }})
    before = copy.deepcopy(style)
    settings = caption_layout_settings(style["captions"], style["brand"])
    expected = min(caption_band_geometry(
        settings, safe_bottom_px=style["internal"]["caption_safe_bottom_px"],
        backing=backing).y for backing in ((), ("box",), ("shadow",)))
    assert stage._caption_band_top(style) == expected
    if look != "giant":
        assert settings == style["captions"]
    assert style == before


def write(path: Path, data: dict[str, Any]) -> Path:
    """Write an authored synthetic contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def publish_cut(root: Path, *, removed: tuple[str, ...] = (), shift: float = 0,
                repeat: bool = False, total_frames: int = 360) -> None:
    """Publish a synthetic output-time map through the real receipt journal."""
    words = load_words(root / "ingest/t1/words.json")
    segments = json.loads((root / "edit.json").read_bytes())["segments"]
    mapped = []
    for word in words.words:
        segment = next((item for item in segments
                        if item["from"]["time"] <= word.start < item["to"]["time"]), None)
        if word.id not in removed and segment is not None:
            mapped.append({"id": word.id, "text": word.text, "segment": segment["id"],
                           "start": max(0, word.start - shift), "end": max(0, word.end - shift)})
    if repeat:
        mapped += [{**word, "segment": "s2", "start": word["start"] + 6,
                    "end": word["end"] + 6} for word in mapped[:2]]
    inputs = cut_inputs(ProjectContext(root))
    staged = write(root / "staged-cut-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output",
        "words_sha256": {"t1": receipt.sha256_file(root / "ingest/t1/words.json")},
        "words": mapped,
    })
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "cut/cut-words.json", staged)
        run.finish(extra={"plan": {"total_frames": total_frames, "end_hold_s": 1.5}})


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Make lavfi picture and numpy audio, source words, and verified receipts."""
    source = synth.calib_source(tmp_path / "source.mov", seconds=14, size="160x90")
    write(tmp_path / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample",
        "style": "preset:clean_minimal", "asr": {"engine": "none"},
        "takes": [{"id": "t1", "path": source.name}],
    })
    inputs = ingest_inputs(ProjectContext(tmp_path))
    staged = write(tmp_path / "staged-words.json", {
        "schema": "yap-studio/words@1", "take": "t1", "take_sha256": inputs["take:t1"],
        "time_base": "source", "audio_offset_s": 0, "engine": "none", "model": None,
        "transcript_sha256": "1" * 64,
        "words": [{"id": f"t1:{index}", "text": text, "start": index + 0.2,
                   "end": index + 0.6} for index, text in enumerate(
                       ["alpha", "um", "beta", "gamma", "delta", "epsilon",
                        "zeta", "eta", "theta", "iota", "kappa", "lambda"],
                   )],
    })
    with StageRun(tmp_path, "ingest") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(tmp_path / "ingest/t1/words.json", staged)
        run.finish()
    write(tmp_path / "edit.json", {
        "schema": "yap-studio/edit@1",
        "words_sha256": {"t1": receipt.sha256_file(tmp_path / "ingest/t1/words.json")},
        "segments": [{"id": "s1", "take": "t1", "from": {"time": 0},
                      "to": {"time": 12}}],
    })
    publish_cut(tmp_path)
    return tmp_path


def insert(project: Path, **changes: Any) -> dict[str, Any]:
    """Build a hand-authored insert referencing the synthetic source."""
    return {"id": "broll-one", "kind": "insert", "lane": "broll",
            "start": {"word": "t1:2", "text": "BETA!"}, "end": {"word": "t1:3"},
            "media": {"path": "source.mov", "sha256": receipt.sha256_file(project / "source.mov"),
                      "in_s": 0, "speed": 1}, **changes}


def overlay(project: Path, **changes: Any) -> dict[str, Any]:
    """Build a silent overlay referencing the synthetic source."""
    return insert(project, **{"kind": "overlay", "placement": "headroom", **changes})


def graphic(**changes: Any) -> dict[str, Any]:
    """Build an agent graphic with a style-dependent default hold."""
    return {"id": "agent-card", "kind": "graphic", "lane": "agent",
            "template": "text_tag", "slots": {"text": "beta"},
            "start": {"word": "t1:6"}, **changes}


def test_requested_sound_resolves_at_word_and_never_snaps(project: Path) -> None:
    item = {"id": "agent-laugh", "kind": "sound", "lane": "agent", "start": {
        "word": "t1:2"}, "sound": "laugh", "gain_db": -2}
    add(project, item, lane="agent").require_valid()
    resolved = resolve_timeline(project).elements[0]
    assert resolved.kind == "sound" and resolved.start_frame == 66
    assert resolved.end_frame == 67
    assert resolved.payload == {"sound": "laugh", "gain_db": -2}


def test_requested_sound_cannot_start_in_end_hold(project: Path) -> None:
    with pytest.raises(TimelineRefusal, match="outside|hold"):
        add(project, {"id": "agent-laugh", "kind": "sound", "lane": "agent",
            "start": {"time": 12}, "sound": "laugh"}, lane="agent").require_valid()


def test_unknown_graphic_template_keeps_existing_timeline_findings(project: Path) -> None:
    lane = {"schema": "yap-studio/timeline@1",
            "words_sha256": load_cut_words(project / "cut/cut-words.json").words_sha256,
            "elements": [graphic(start={"time": 11}, hold_s=2)]}
    write(project / "timeline/agent.json", lane)
    expected = resolve_timeline(project)
    lane["elements"][0]["template"] = "unknown_card"
    write(project / "timeline/agent.json", lane)
    actual = resolve_timeline(project)
    assert actual.findings == expected.findings
    assert len(actual.elements) == len(expected.elements) == 1


@pytest.mark.parametrize("ending", ["handoff", "fade"])
@pytest.mark.parametrize("zoom_start", [2, 2.5])
def test_reel_feed_handoff_requires_cut_picture_on_its_last_frame(
    project: Path, ending: str, zoom_start: float,
) -> None:
    feed = graphic(template="reel_feed", slots={"ending": ending},
                   start={"time": 0}, hold_s=2.5)
    zoom = {"id": "agent-zoom", "kind": "zoom", "lane": "agent",
            "start": {"time": zoom_start}, "hold_s": 1,
            "peak_scale": 1.05, "center": [0.5, 0.5]}
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
          "words_sha256": load_cut_words(project / "cut/cut-words.json").words_sha256,
          "elements": [feed, zoom]})
    result = resolve_timeline(project)
    findings = [item for item in result.findings if item.rule == "card_video_handoff"]
    assert bool(findings) == (ending == "handoff" and zoom_start == 2)
    if findings:
        assert findings[0].severity == "violation"
        assert "agent-card" in findings[0].elements
        assert "ending fade" in findings[0].message


@pytest.mark.parametrize("kind", ["background", "overlay"])
def test_reel_feed_handoff_refuses_active_picture_layers(project: Path, kind: str) -> None:
    feed = graphic(template="reel_feed", start={"time": 0}, hold_s=2.5, slots={})
    picture = (overlay(project) if kind == "overlay" else insert(project, kind=kind))
    picture.update(start={"time": 2}, end={"time": 3})
    words_sha256 = load_cut_words(project / "cut/cut-words.json").words_sha256
    for lane, element in (("agent", feed), ("broll", picture)):
        write(project / f"timeline/{lane}.json", {"schema": "yap-studio/timeline@1",
              "words_sha256": words_sha256, "elements": [element]})
    result = resolve_timeline(project)
    findings = [item for item in result.findings if item.rule == "card_video_handoff"]
    assert len(findings) == 1
    assert findings[0].severity == "violation"
    assert set(findings[0].elements) == {feed["id"], picture["id"]}
    assert kind in findings[0].message
    assert "ending fade" in findings[0].message


@pytest.mark.parametrize("start,end", [(1, 2), (2.5, 3.5)],
                         ids=["ends-before-last-frame", "starts-at-feed-end"])
def test_reel_feed_handoff_allows_picture_layers_outside_last_frame(
    project: Path, start: float, end: float,
) -> None:
    feed = graphic(template="reel_feed", start={"time": 0}, hold_s=2.5, slots={})
    background = insert(project, kind="background", start={"time": start}, end={"time": end})
    words_sha256 = load_cut_words(project / "cut/cut-words.json").words_sha256
    for lane, element in (("agent", feed), ("broll", background)):
        write(project / f"timeline/{lane}.json", {"schema": "yap-studio/timeline@1",
              "words_sha256": words_sha256, "elements": [element]})
    result = resolve_timeline(project)
    assert {item.id for item in result.elements} == {feed["id"], background["id"]}
    assert not any(item.rule == "card_video_handoff" for item in result.findings)


@pytest.mark.parametrize("effect", [
    "background_blur", "color_pop", "spotlight", "sticker_freeze", "outline", "clone_flash",
    "grade_compare", "letters_title", "title",
])
@pytest.mark.parametrize("ending,start,blocked", [
    ("handoff", 2, True), ("fade", 2, False),
    ("handoff", 1, False), ("handoff", 2.5, False),
])
def test_reel_feed_handoff_checks_picture_effects(
    project: Path, monkeypatch: pytest.MonkeyPatch,
    effect: str, ending: str, start: float, blocked: bool,
) -> None:
    style = resolve_style_data({"schema_version": 1, "moments": {"styles": [
        "background_blur", "color_pop", "spotlight", "sticker_freeze", "outline", "clone_flash",
    ]}, "grade": {"look": "film"}})
    if effect == "letters_title":
        style["hook"]["style"] = "video_letters"
    monkeypatch.setattr(stage, "_style", lambda context: style)
    picture = {"id": "agent-effect", "kind": "moment", "style": effect,
               "start": {"time": start}, "hold_s": 1}
    if effect == "grade_compare":
        picture.update(kind="grade_compare", hold_s=2,
                       start={"time": max(0, start - 1) if start <= 2 else start})
        del picture["style"]
    elif effect in {"letters_title", "title"}:
        picture.update(kind="title", lines=["beta"])
        del picture["style"]
    feed = graphic(template="reel_feed", start={"time": 0}, hold_s=2.5,
                   slots={"ending": ending})
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
          "words_sha256": load_cut_words(project / "cut/cut-words.json").words_sha256,
          "elements": [feed, picture]})
    result = resolve_timeline(project)
    findings = [item for item in result.findings if item.rule == "card_video_handoff"]
    assert bool(findings) == (blocked and effect != "title")
    if findings:
        finding, = findings
        assert finding.severity == "violation"
        assert set(finding.elements) == {feed["id"], picture["id"]}
        assert "move the feed" in finding.message and "ending fade" in finding.message
        assert "Move the feed" in finding.suggested_action
        assert "ending fade" in finding.suggested_action


@pytest.mark.parametrize("ending", ["handoff", "fade"])
def test_reel_feed_handoff_checks_step_in_framing(step_project: Path, ending: str) -> None:
    feed = graphic(template="reel_feed", start={"time": 0}, hold_s=2.5,
                   slots={"ending": ending})
    write(step_project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
          "words_sha256": load_cut_words(step_project / "cut/cut-words.json").words_sha256,
          "elements": [feed]})
    result = resolve_timeline(step_project)
    zoom, = [item for item in result.elements if item.lane == "engine"
             and item.start_frame <= 74 < item.end_frame]
    assert zoom.kind == "zoom" and zoom.payload["move"] == "steps"
    findings = [item for item in result.findings if item.rule == "card_video_handoff"]
    assert bool(findings) == (ending == "handoff")
    if findings:
        finding, = findings
        assert set(finding.elements) == {feed["id"], zoom.id}
        assert "Move the feed" in finding.suggested_action
        assert "ending fade" in finding.suggested_action


@pytest.mark.parametrize("problem,rule", [
    ("hash", "media_hash"), ("outside", "card_video_media"),
    ("picture", "card_video_media"), ("missing", "card_video_media"),
    ("speed", "card_video_speed"), ("bounds", "card_video_bounds"),
    ("audio", "card_video_audio"), ("self", "card_video_media"),
    ("crop", "card_video_media"),
])
def test_card_video_findings(project: Path, problem: str, rule: str) -> None:
    source = project / "source.mov"
    clip = {"path": source.name, "sha256": receipt.sha256_file(source)}
    if problem == "hash":
        clip["sha256"] = "0" * 64
    elif problem == "outside":
        clip["path"] = str(project.parent / "outside.mov")
        shutil.copyfile(source, clip["path"])
    elif problem == "picture":
        audio = project / "audio.wav"
        ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-vn", str(audio)],
                   timeout_s=30)
        clip.update(path=audio.name, sha256=receipt.sha256_file(audio))
    elif problem == "missing":
        clip["path"] = "missing.mov"
    elif problem == "speed":
        clip["speed"] = 3
    elif problem == "bounds":
        clip["in_s"] = 13
    elif problem == "crop":
        clip["crop"] = [0, 0, 10000, 32]
    elif problem == "audio":
        silent = synth.picture_source(project / "silent.mov", "color=s=160x90:r=30", 120)
        clip.update(path=silent.name, sha256=receipt.sha256_file(silent), audio={"gain_db": 0})
    value = "self" if problem == "self" else clip
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
          "words_sha256": load_cut_words(project / "cut/cut-words.json").words_sha256,
          "elements": [
        graphic(template="video_card", slots={"clip": value}, start={"time": 1}, hold_s=2),
    ]})
    result = resolve_timeline(project)
    assert any(item.rule == rule and item.severity == "violation"
               and item.elements == ("agent-card",) for item in result.findings)
    if problem == "crop":
        assert any("crop must sit inside" in item.message for item in result.findings
                   if item.rule == rule)


@pytest.mark.parametrize("template,slot", [
    ("video_card", "clip"), ("playing_grid", "clips"), ("reel_feed", "post"),
])
def test_card_video_timeline_accepts_crop(project: Path, template: str, slot: str) -> None:
    source = project / "source.mov"
    clip = {"path": source.name, "sha256": receipt.sha256_file(source), "crop": [0, 0, 32, 32]}
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
          "words_sha256": load_cut_words(project / "cut/cut-words.json").words_sha256,
          "elements": [graphic(template=template, slots={slot: [clip, clip] if slot == "clips"
                                                        else clip,
                                                        **({"ending": "fade"}
                                                           if template == "reel_feed" else {})},
                               start={"time": 1}, hold_s=2)]})
    result = resolve_timeline(project)
    assert not any(item.rule.startswith("card_video_") for item in result.findings)


def fragment(root: Path, *elements: dict[str, Any], name: str = "fragment.json") -> Path:
    """Write the public fragment contract by hand."""
    return write(root / name, {"schema": "yap-studio/timeline@1", "elements": list(elements)})


def test_voiceover_graphic_only_warning_tracks_frames_without_inserts(project: Path) -> None:
    card = fragment(project, graphic(start={"time": 1}, end={"time": 3}))
    video = add_timeline(project, card, lane="agent")
    assert not any(item.rule.startswith("voiceover_") for item in video.findings)
    document = json.loads((project / "project.json").read_text())
    document["takes"][0]["voiceover"] = True
    write(project / "project.json", document)
    result = resolve_timeline(project)
    warning = next(item for item in result.findings if item.rule == "voiceover_graphic_only")
    assert warning.severity == "warning" and warning.elements == warning.lanes == ()
    assert warning.message == warning.bare_message == (
        "Voiceover picture at frames 30-89 comes only from graphics; assembly counts a frame "
        "only where a graphic is opaque and fills the frame.")
    assert warning.suggested_action == (
        "Cover these frames with B-roll inserts, or make sure the graphic fills the whole frame.")
    assert next(item for item in result.findings if item.rule == "voiceover_uncovered").message == (
        "Voiceover picture uncovered at frames 0-29, 90-359")
    result = add_timeline(project, fragment(
        project, insert(project, id="broll-start", start={"time": 0}, end={"time": 1}),
        insert(project, id="broll-end", start={"time": 3}, end={"time": 12}),
    ), lane="broll")
    assert result.valid
    assert not any(item.rule == "voiceover_uncovered" for item in result.findings)
    assert any(item.rule == "voiceover_graphic_only" for item in result.findings)
    remove_timeline(project, lane="agent")
    result = add_timeline(project, fragment(
        project, insert(project, start={"time": 0}, end={"time": 12}),
    ), lane="broll", replace_lane=True)
    assert result.valid
    assert not any(item.rule.startswith("voiceover_") for item in result.findings)


def test_replace_lane_clears_stale_elements_and_refusal_keeps_old(project: Path) -> None:
    old = insert(project, id="broll-old", start={"word": "t1:2"}, end={"word": "t1:3"})
    path = write(project / "timeline/broll.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {"t1": "0" * 64},
        "elements": [old],
    })
    replacement = fragment(project, insert(project, id="broll-new"))
    with pytest.raises(TimelineRefusal):
        add_timeline(project, replacement, lane="broll")
    result = add_timeline(project, replacement, lane="broll", replace_lane=True)
    assert result.valid
    assert [item["id"] for item in json.loads(path.read_bytes())["elements"]] == ["broll-new"]
    before = path.read_bytes()
    refused = fragment(project, insert(project, id="broll-bad", start={"word": "t1:100"}))
    assert not add_timeline(project, refused, lane="broll", replace_lane=True).valid
    assert path.read_bytes() == before


def transition(**changes: Any) -> dict[str, Any]:
    """Anchor a shader window to the incoming occurrence, independent of its speech pad."""
    return {"id": "agent-transition", "kind": "transition", "lane": "agent",
            "start": {"word": "t1:6", "segment": "s2"}, "look": "whip_pan", **changes}


def transition_story(root: Path, *, sentence: bool = True, join: int = 180) -> None:
    """Republish exact spans and word occurrences around a configurable story boundary."""
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"] = [
        {"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"time": 6}},
        {"id": "s2", "take": "t1", "from": {"time": 6}, "to": {"time": 12}},
    ]
    write(root / "edit.json", edit)
    mapped = {"schema": "yap-studio/cut-words@1", "time_base": "output",
              "words_sha256": {"t1": receipt.sha256_file(root / "ingest/t1/words.json")},
              "words": [{"id": word.id, "text": word.text,
                         "segment": "s1" if word.start < 6 else "s2"}
                        for word in load_words(root / "ingest/t1/words.json").words]}
    for word in mapped["words"]:
        index = int(word["id"].split(":")[1])
        if word["segment"] == "s1":
            word["start"] = join / 30 * (index + 0.1) / 6
            word["end"] = join / 30 * (index + 0.7) / 6
        else:
            duration = 12 - join / 30
            word["start"] = join / 30 + duration * (index - 6 + 0.2) / 6
            word["end"] = join / 30 + duration * (index - 6 + 0.7) / 6
        if word["id"] == "t1:5":
            word["text"] = "epsilon." if sentence else "epsilon"
    words = write(root / "staged-join-words.json", mapped)
    spans = write(root / "staged-spans.json", {
        "sources": {"t1": "../source.mov"}, "fps": "30/1",
        "segments": [{"source": "t1", "start": "0", "end": str(join / 30)},
                     {"source": "t1", "start": str(join / 30), "end": "12"}],
    })
    with StageRun(root, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.replace_output(root / "cut/cut-words.json", words)
        run.replace_output(root / "cut/spans.json", spans)
        run.finish(extra={"plan": {"total_frames": 360}})


@pytest.mark.parametrize("look", ["whip_pan", "cinematic_zoom", "light_leak"])
def test_transition_style_default(project: Path, look: str) -> None:
    transition_story(project)
    result = add(project, transition(look=look), lane="agent").require_valid()
    assert not any(f.rule == "transition_style" for f in result.findings)


@pytest.mark.parametrize("look", ["whip_pan", "cinematic_zoom", "light_leak"])
@pytest.mark.parametrize("allowed", [[], ["light_leak", "cinematic_zoom"]])
def test_transition_style_options(project: Path, monkeypatch: pytest.MonkeyPatch,
                                  look: str, allowed: list[str]) -> None:
    transition_story(project)
    default = add(project, transition(look=look), lane="agent").require_valid()
    saved = (project / "timeline/agent.json").read_bytes()
    (project / "timeline/agent.json").unlink()
    style = resolve_style_data({"schema_version": 1})
    style["moves"]["transitions"] = allowed
    monkeypatch.setattr(stage, "_style", lambda context: style)
    result = add(project, transition(look=look), lane="agent").require_valid()
    assert (project / "timeline/agent.json").read_bytes() == saved
    assert result.elements == default.elements
    item = result.elements[0]
    assert (item.start_frame, item.end_frame, item.payload["join_frame"]) == (174, 186, 180)
    assert item.payload["frames"] == 12
    for checked in (result, resolve_timeline(project).require_valid()):
        warnings = [f for f in checked.findings if f.rule == "transition_style"]
        assert len(warnings) == (look not in allowed)
        if warnings:
            assert warnings[0].severity == "warning"
            assert warnings[0].bare_message == (
                "the style has transitions off" if not allowed else
                "the style does not allow whip_pan; it allows light_leak, cinematic_zoom")
            assert warnings[0].suggested_action == (
                "Use a transition look the style allows, or keep this one only when a review "
                "comment asked for it.")


def test_transition_sentence_join_window_and_recut(project: Path) -> None:
    transition_story(project)
    item = add(project, transition(), lane="agent").require_valid().elements[0]
    assert (item.start_frame, item.end_frame, item.payload["join_frame"]) == (174, 186, 180)
    assert item.payload["frames"] == 12
    saved = (project / "timeline/agent.json").read_bytes()
    transition_story(project, join=150)
    item = resolve_timeline(project).require_valid().elements[0]
    assert (item.start_frame, item.end_frame, item.payload["join_frame"]) == (144, 156, 150)
    assert (project / "timeline/agent.json").read_bytes() == saved


@pytest.mark.parametrize("incoming_start", [6.066, 6.067])
def test_transition_boundary_words_use_segment_order_with_rounded_times(
    project: Path, incoming_start: float,
) -> None:
    """Millisecond rounding across a join cannot replace either boundary word."""
    transition_story(project, join=182)
    mapped = json.loads((project / "cut/cut-words.json").read_bytes())
    mapped["words"][4]["text"] = "a"
    mapped["words"][5].update(text="skill.", start=6.03, end=6.067)
    mapped["words"][6].update(start=incoming_start, end=6.3)
    assert mapped["words"][5]["end"] > 182 / 30
    staged_words = write(project / "staged-join-words.json", mapped)
    staged_spans = write(project / "staged-spans.json", json.loads(
        (project / "cut/spans.json").read_bytes(),
    ))
    with StageRun(project, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.replace_output(project / "cut/cut-words.json", staged_words)
        run.replace_output(project / "cut/spans.json", staged_spans)
        run.finish(extra={"plan": {"total_frames": 360}})
    item = add(project, transition(), lane="agent").require_valid().elements[0]
    assert (item.start_frame, item.end_frame, item.payload["join_frame"]) == (176, 188, 182)


@pytest.mark.parametrize("piece_segment_ids", [True, False])
def test_transition_wordless_outgoing_piece_uses_receipt_segment_id(
    project: Path, piece_segment_ids: bool,
) -> None:
    """A retained pause inherits its story segment from the cut's piece record."""
    transition_story(project)
    mapped = json.loads((project / "cut/cut-words.json").read_bytes())
    mapped["words"][5]["end"] = 5.6
    for word in mapped["words"]:
        if word["segment"] == "s2":
            word["start"] -= 0.1
            word["end"] -= 0.1
    staged_words = write(project / "staged-join-words.json", mapped)
    spans = [{"source": "t1", "start": "0", "end": "5.6"},
             {"source": "t1", "start": "5.7", "end": "6"},
             {"source": "t1", "start": "6", "end": "12"}]
    staged_spans = write(project / "staged-spans.json", {
        "sources": {"t1": "../source.mov"}, "fps": "30/1", "segments": spans,
    })
    pieces = [{"segment_id": identifier, "take": span["source"],
               "src_t0": span["start"], "src_t1": span["end"],
               "out_f0": start, "n_frames": frames}
              for span, identifier, start, frames in zip(
                  spans, ["s1", "s1", "s2"], [0, 168, 177], [168, 9, 180], strict=True,
              )]
    if not piece_segment_ids:
        for piece in pieces:
            del piece["segment_id"]
    with StageRun(project, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.replace_output(project / "cut/cut-words.json", staged_words)
        run.replace_output(project / "cut/spans.json", staged_spans)
        run.finish(extra={"plan": {"total_frames": 357, "pieces": pieces}})
    result = add(project, transition(), lane="agent")
    if piece_segment_ids:
        item = result.require_valid().elements[0]
        assert (item.start_frame, item.end_frame, item.payload["join_frame"]) == (171, 183, 177)
    else:
        assert not result.valid and any(f.rule == "transition_join" for f in result.findings)
        assert not (project / "timeline/agent.json").exists()


def test_transition_inside_sentence_refuses_lane_write(project: Path) -> None:
    transition_story(project, sentence=False)
    result = add(project, transition(), lane="agent")
    assert not result.valid and any(f.rule == "transition_sentence" for f in result.findings)
    assert not (project / "timeline/agent.json").exists()


@pytest.mark.parametrize("start", [
    {"word": "t1:7"}, {"time": 6}, {"word": "t1:6", "offset_s": 0.1},
    {"word": "t1:6", "edge": "end"},
])
def test_transition_requires_first_incoming_word(project: Path, start: dict) -> None:
    transition_story(project)
    result = add(project, transition(start=start), lane="agent")
    assert not result.valid and any(f.rule == "transition_join" for f in result.findings)


@pytest.mark.parametrize("kind", ["title", "chapter", "insert", "overlay", "zoom", "transition"])
def test_transition_overlap_refuses_each_exclusive_kind(project: Path, kind: str) -> None:
    transition_story(project)
    if kind in {"insert", "overlay"}:
        other = (overlay if kind == "overlay" else insert)(
            project, start={"time": 5.9}, end={"time": 6.3},
        )
        add(project, other).require_valid()
        elements = [transition()]
    else:
        fields = {"title": {"lines": ["Join"]}, "chapter": {"title": "Join"},
                  "zoom": {"peak_scale": 1.04, "center": [0.5, 0.5]}}
        other = transition(id="agent-other") if kind == "transition" else {
            "id": "agent-other", "kind": kind, "lane": "agent",
            "start": {"time": 5}, "end": {"time": 8}, **fields[kind],
        }
        elements = [transition(), other]
    result = add(project, *elements, lane="agent")
    assert not result.valid
    assert any(f.rule == "transition_overlap" and len(f.elements) == 2 for f in result.findings)
    if kind == "transition":
        assert any(f.rule == "transition_duplicate" for f in result.findings)


@pytest.mark.parametrize("join", [3, 357])
def test_transition_window_stays_inside_content(project: Path, join: int) -> None:
    transition_story(project, join=join)
    result = add(project, transition(), lane="agent")
    assert not result.valid and any(f.rule == "inside_video" for f in result.findings)


@pytest.mark.parametrize("join,valid", [(6, False), (7, True), (353, True), (354, False)])
def test_transition_window_requires_content_neighbors(
    project: Path, join: int, valid: bool,
) -> None:
    transition_story(project, join=join)
    result = add(project, transition(), lane="agent")
    assert result.valid is valid
    assert any(f.rule == "transition_bounds" for f in result.findings) is not valid
    assert (project / "timeline/agent.json").exists() is valid


def test_transition_refuses_filler_removal_after_full_stop_in_one_segment(project: Path) -> None:
    """Sentence punctuation cannot turn an internal removal into a story join."""
    source_words = json.loads((project / "ingest/t1/words.json").read_bytes())
    source_words["words"][0]["text"] = "alpha."
    staged = write(project / "staged-words.json", source_words)
    with StageRun(project, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.replace_output(project / "ingest/t1/words.json", staged)
        run.finish()
    hashes = {"t1": receipt.sha256_file(project / "ingest/t1/words.json")}
    edit = json.loads((project / "edit.json").read_bytes())
    edit["words_sha256"] = hashes
    write(project / "edit.json", edit)
    write(project / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "filler-one", "take": "t1", "start": 1, "end": 2,
            "kind": "filler", "status": "applied", "words": ["t1:1"],
        }],
    })
    words = write(project / "staged-cut-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output", "words_sha256": hashes,
        "words": [{"id": word["id"], "text": word["text"], "segment": "s1",
                   "start": word["start"] - (1 if word["start"] >= 2 else 0),
                   "end": word["end"] - (1 if word["end"] >= 2 else 0)}
                  for word in source_words["words"] if word["id"] != "t1:1"],
    })
    spans = write(project / "staged-spans.json", {
        "sources": {"t1": "../source.mov"}, "fps": "30/1",
        "segments": [{"source": "t1", "start": "0", "end": "1"},
                     {"source": "t1", "start": "2", "end": "12"}],
    })
    with StageRun(project, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.replace_output(project / "cut/cut-words.json", words)
        run.replace_output(project / "cut/spans.json", spans)
        run.finish(extra={"plan": {"total_frames": 330}})
    result = add(project, transition(start={"word": "t1:2", "segment": "s1"}), lane="agent")
    assert not result.valid and any(f.rule == "transition_join" for f in result.findings)
    assert not (project / "timeline/agent.json").exists()


@pytest.mark.parametrize("second_join", [188, 192])
def test_transition_windows_at_distinct_joins(project: Path, second_join: int) -> None:
    transition_story(project)
    edit = json.loads((project / "edit.json").read_bytes())
    edit["segments"] = [edit["segments"][0],
                        {"id": "s2", "take": "t1", "from": {"time": 6}, "to": {"time": 7}},
                        {"id": "s3", "take": "t1", "from": {"time": 7}, "to": {"time": 12}}]
    write(project / "edit.json", edit)
    words = json.loads((project / "cut/cut-words.json").read_bytes())
    for word in words["words"]:
        if word["id"] == "t1:6":
            word.update(start=6.03, end=6.1, text="zeta!")
        elif int(word["id"].split(":")[1]) >= 7:
            word["segment"] = "s3"
    staged_words = write(project / "staged-join-words.json", words)
    spans = write(project / "staged-spans.json", {
        "sources": {"t1": "../source.mov"}, "fps": "30/1",
        "segments": [{"source": "t1", "start": "0", "end": "6"},
                     {"source": "t1", "start": "6", "end": str(second_join / 30)},
                     {"source": "t1", "start": str(second_join / 30), "end": "12"}],
    })
    with StageRun(project, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.replace_output(project / "cut/cut-words.json", staged_words)
        run.replace_output(project / "cut/spans.json", spans)
        run.finish(extra={"plan": {"total_frames": 360}})
    result = add(project, transition(), transition(
        id="agent-second", start={"word": "t1:7", "segment": "s3"},
    ), lane="agent")
    assert result.valid == (second_join == 192)
    assert not any(f.rule == "transition_duplicate" for f in result.findings)
    assert any(f.rule == "transition_overlap" for f in result.findings) == (second_join == 188)


def test_transition_requires_receipt_owned_spans(project: Path) -> None:
    transition_story(project)
    # Publishing only cut words leaves the old spans outside the new receipt's ownership.
    publish_cut(project)
    with pytest.raises(TimelineRefusal, match="does not own cut/spans.json"):
        add(project, transition(), lane="agent")


def add(root: Path, *elements: dict[str, Any], lane: str = "broll") -> stage.ResolvedTimeline:
    """Submit a hand-authored fragment through the transaction API."""
    return add_timeline(root, fragment(root, *elements), lane=lane)


def beat_style(root: Path, *, enabled: bool = True, window_ms: float = 150) -> None:
    """Select a synthetic style without a muted opening for frame-boundary tests."""
    write(root / "style.json", {"schema_version": 1, "name": "Synthetic beats",
                               "music": {"beat_snap": enabled, "beat_snap_window_ms": window_ms,
                                         "dry_open_s": 0}})
    project = json.loads((root / "project.json").read_bytes())
    project["style"] = "style.json"
    write(root / "project.json", project)


def publish_beats(root: Path, frames: tuple[int, ...], *,
                  changes: dict[str, Any] | None = None) -> None:
    """Publish a synthetic music bed and hints through the real ownership journal."""
    track = root / "track.wav"
    if not track.exists():
        synth._write_wav(track, np.zeros(14 * 48_000), 48_000)
    project = json.loads((root / "project.json").read_bytes())
    project["music"] = {"mode": "file", "file": "track.wav"}
    write(root / "project.json", project)
    context = ProjectContext(root)
    staged = write(root / "staged-beats.json", {
        "schema": "yap-studio/beats@1", "bpm": 120.0,
        "music_sha256": receipt.sha256_file(track),
        "beats": [{"time_s": frame / 30, "frame": frame, "strength": 0.8} for frame in frames],
        **(changes or {}),
    })
    bed = root / "staged-bed.wav"
    bed.write_bytes(track.read_bytes())
    with StageRun(root, "music") as run:
        for name, digest in music_inputs(context).items():
            run.record_inputs(name, digest)
        run.replace_output(root / "music/bed.wav", bed)
        run.replace_output(root / "music/beats.json", staged)
        run.finish(extra={"mode": "file", "bed": "music/bed.wav", "cut_frames": 360,
                          "source": {"sha256": receipt.sha256_file(track)}})


def test_beat_snap_moves_graphic_and_insert_and_preserves_lengths(project: Path) -> None:
    beat_style(project)
    add(project, graphic(), lane="agent").require_valid()
    add(project, insert(project)).require_valid()
    before = resolve_timeline(project).to_dict()
    lane_bytes = [(project / f"timeline/{lane}.json").read_bytes() for lane in stage.LANES]
    publish_beats(project, (64, 188))
    result = resolve_timeline(project).require_valid()
    items = {item.id: item for item in result.elements}
    assert (items["broll-one"].start_frame, items["broll-one"].end_frame) == (64, 106)
    old_graphic = next(item for item in before["elements"] if item["id"] == "agent-card")
    assert items["agent-card"].start_frame == 188
    assert items["agent-card"].end_frame == old_graphic["end_frame"] + 2
    for old in before["elements"]:
        item = items[old["id"]]
        assert item.end_frame - item.start_frame == old["end_frame"] - old["start_frame"]
    assert result.snapped_anchors == ()
    snaps = {entry["element"]: entry for entry in result.beat_snaps}
    assert snaps["broll-one"] == {
        "element": "broll-one", "lane": "broll",
        "from_frame": 66, "to_frame": 64, "from_end_frame": 108,
        "to_end_frame": 106,
    }
    assert snaps["agent-card"]["from_frame"] == 186
    assert snaps["agent-card"]["to_frame"] == 188
    assert lane_bytes == [(project / f"timeline/{lane}.json").read_bytes() for lane in stage.LANES]


def test_generated_bed_beat_hints_snap_like_supplied_track(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio import elevenlabs
    from yap_studio.stages import music as music_stage

    beat_style(project)
    add(project, graphic(), lane="agent").require_valid()
    samples = 0.1 * np.sin(np.arange(14 * 48_000) * (2 * np.pi * 220 / 48_000))
    synth._write_wav(project / "track.wav", samples, 48_000)
    publish_beats(project, (188,))
    supplied = resolve_timeline(project).require_valid()
    settings = json.loads((project / "project.json").read_text())
    settings["music"] = {"mode": "elevenlabs"}
    write(project / "project.json", settings)
    monkeypatch.setattr(elevenlabs, "account_state", lambda: "paid")
    monkeypatch.setattr(elevenlabs, "compose_plan", lambda *_args, **_kwargs: {
        "sections": [{"duration_ms": 27_000, "prompt": "Instrumental"}],
    })

    def generate(_body: dict, target: Path, *, dry_run: bool) -> None:
        assert not dry_run
        shutil.copyfile(project / "track.wav", target)

    monkeypatch.setattr(elevenlabs, "generate_music", generate)
    monkeypatch.setattr(music_stage, "detect_beats", lambda *_args, **_kwargs: {
        "bpm": 120.0, "beats": [{"time_s": 188 / 30, "strength": 0.8}],
    })
    assert music_stage.music(project)["mode"] == "elevenlabs"
    generated = resolve_timeline(project).require_valid()
    assert generated.elements[0].start_frame == supplied.elements[0].start_frame == 188
    assert generated.beat_snaps == supplied.beat_snaps


@pytest.mark.parametrize("window_ms,frames,expected", [
    (150, (70, 190), (70, 190)), (150, (71, 191), (66, 186)),
    (100, (63, 183), (63, 183)), (99.9, (63, 183), (66, 186)),
    (0, (65, 185), (66, 186)), (150, (66, 186), (66, 186)),
])
def test_beat_window_uses_exact_frame_distance(
    project: Path, window_ms: float, frames: tuple[int, ...], expected: tuple[int, int],
) -> None:
    beat_style(project, window_ms=window_ms)
    add(project, graphic(), lane="agent").require_valid()
    add(project, insert(project)).require_valid()
    publish_beats(project, frames)
    result = resolve_timeline(project).require_valid()
    assert tuple(item.start_frame for item in result.elements) == expected
    assert result.snapped_anchors == ()
    if expected == (66, 186):
        assert result.beat_snaps == ()


def test_beat_ties_choose_the_earlier_frame(project: Path) -> None:
    beat_style(project)
    add(project, graphic(), lane="agent").require_valid()
    publish_beats(project, (184, 188))
    result = resolve_timeline(project).require_valid()
    assert result.elements[0].start_frame == 184
    assert result.beat_snaps[0]["to_frame"] == 184


def test_beat_flag_off_leaves_resolved_timeline_bytes_identical(project: Path) -> None:
    beat_style(project, enabled=False)
    add(project, graphic(), lane="agent").require_valid()
    add(project, insert(project)).require_valid()
    before = json.dumps(resolve_timeline(project).to_dict()).encode()
    publish_beats(project, (64, 188))
    assert json.dumps(resolve_timeline(project).to_dict()).encode() == before
    assert resolve_timeline(project).snapped_anchors == ()
    assert resolve_timeline(project).beat_snaps == ()
    assert resolve_timeline(project).to_dict()["beat_snaps"] == []


def test_beat_snaps_work_in_cli_text_mode_and_have_separate_json_records(
    project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    beat_style(project)
    publish_beats(project, (188,))
    file = fragment(project, graphic(hold_s=2))
    assert cli.main(["timeline", "add", str(file), str(project), "--lane", "agent"]) == 0
    assert "passed" in capsys.readouterr().out
    for command in ("show", "check"):
        assert cli.main(["timeline", command, str(project)]) == 0
        text = capsys.readouterr().out
        assert "passed" in text and "beat: agent-card entrance frame 186 to 188" in text
        assert cli.main(["timeline", command, str(project), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["snapped_anchors"] == []
        assert data["beat_snaps"] == [{
            "element": "agent-card", "lane": "agent", "from_frame": 186, "to_frame": 188,
            "from_end_frame": 246, "to_end_frame": 248,
        }]


@pytest.mark.parametrize("beats", [(188,), (184, 187)])
def test_card_with_entrance_step_keeps_word_timing_with_beats(
    project: Path, beats: tuple[int, ...],
) -> None:
    beat_style(project)
    item = graphic(hold_s=2, slots={"steps": [{"kind": "type", "at": {"word": "t1:6"}}]})
    add(project, item, lane="agent").require_valid()
    publish_beats(project, beats)
    result = resolve_timeline(project).require_valid()
    card = result.elements[0]
    assert (card.start_frame, card.end_frame) == (186, 246)
    assert card.payload["slots"]["steps"][0]["at"] == {"frame": 186, "time_s": 6.2}
    assert not any(finding.rule == "step_bounds" for finding in result.findings)
    assert result.beat_snaps == ()


def test_beat_candidates_must_keep_every_card_step_inside_the_window(project: Path) -> None:
    beat_style(project)
    item = graphic(hold_s=2, slots={"steps": [
        {"kind": "type", "at": {"word": "t1:6"}},
        {"kind": "type", "at": {"word": "t1:8", "offset_s": -1 / 30}},
    ]})
    add(project, item, lane="agent").require_valid()
    before = resolve_timeline(project).to_dict()
    publish_beats(project, (184, 188))
    result = resolve_timeline(project).require_valid()
    assert result.to_dict() == before
    assert result.beat_snaps == ()


@pytest.mark.parametrize("kind", ["graphic", "insert"])
def test_beat_snap_uses_an_earlier_candidate_when_nearest_exceeds_cut_end(
    project: Path, kind: str,
) -> None:
    beat_style(project)
    if kind == "graphic":
        item = graphic(start={"time": 11}, hold_s=1)
        lane = "agent"
    else:
        item = insert(project, start={"time": 11}, end={"time": 12})
        lane = "broll"
    add(project, item, lane=lane).require_valid()
    publish_beats(project, (327, 331))
    result = resolve_timeline(project).require_valid()
    assert (result.elements[0].start_frame, result.elements[0].end_frame) == (327, 357)
    assert result.beat_snaps == ({
        "element": item["id"], "lane": lane, "from_frame": 330, "to_frame": 327,
        "from_end_frame": 360, "to_end_frame": 357,
    },)
    assert result.snapped_anchors == ()


def test_titles_zooms_and_music_windows_never_snap(project: Path) -> None:
    beat_style(project)
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"zoom_step_min": 0}
    write(project / "project.json", data)
    title = {"id": "agent-title", "kind": "title", "lane": "agent", "lines": ["alpha"],
             "start": {"word": "t1:0"}, "hold_s": 1}
    zoom = {"id": "agent-zoom", "kind": "zoom", "lane": "agent", "start": {"word": "t1:4"},
            "hold_s": 3, "peak_scale": 1.08, "center": [0.5, 0.5]}
    window = {"id": "agent-music", "kind": "music_level", "lane": "agent", "start": {"time": 1},
              "end": {"time": 2}, "level": 0.5}
    add(project, title, zoom, window, lane="agent").require_valid()
    before = json.dumps(resolve_timeline(project).to_dict()).encode()
    publish_beats(project, (8, 32, 128))
    assert json.dumps(resolve_timeline(project).to_dict()).encode() == before


@pytest.mark.parametrize(("spoken", "lines", "warns"), [
    ("Here's why AI did not know how many R's are in strawberry.",
     ["Here's why AI did not know", "how many R's are in strawberry."], False),
    ("Here's why AI did not know how many R's are in strawberry.",
     ["AI did not know", "how many R's"], False),
    ("Here's why AI did not know how many R's are in strawberry.", ["AI"], False),
    ("Here's why AI did not know how many R's are in strawberry.", ["AI knows nothing"], True),
    ("Here's why AI did not know how many R's are in strawberry.",
     ["How many R's", "are in strawberry?"], True),
    ("Here's why AI did not know how many R's are in strawberry.", ["AI!"], True),
    ("How many R's are in strawberry?", ["How many R's are in strawberry?"], False),
    ("How many R's are in strawberry.", ["How many R's?", "are in strawberry"], True),
    ("Why? Because it works.", ["Why?", "Because it works"], False),
    ("I paid fifty percent.", ["I paid $50"], True),
    ("I paid fifty percent.", ["I paid 50%"], False),
    ("It costs fifty dollars.", ["It costs $50"], False),
    ("There are three letters here.", ["3 letters"], False),
    ("There are twenty three letters here.", ["23 letters"], False),
    ("There are twenty three letters here.", ["three letters"], False),
    ("I use BrightPath every day.", ["brightpath"], False),
    ("Here's why AI did not know how many R's are in strawberry.", ["AI", "strawberry"], True),
])
def test_title_words_follow_a_contiguous_spoken_run(
    project: Path, monkeypatch: pytest.MonkeyPatch, spoken: str, lines: list[str], warns: bool,
) -> None:
    from yap_studio import brand_words

    monkeypatch.setattr(brand_words, "list_words", lambda: ["BrightPath"])
    source = json.loads((project / "ingest/t1/words.json").read_bytes())
    source["words"] = [{"id": f"t1:{index}", "text": text, "start": index * 0.5 + 0.2,
                        "end": index * 0.5 + 0.6} for index, text in enumerate(spoken.split())]
    staged = write(project / "staged-title-words.json", source)
    with StageRun(project, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.replace_output(project / "ingest/t1/words.json", staged)
        run.finish()
    edit = json.loads((project / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = receipt.sha256_file(project / "ingest/t1/words.json")
    write(project / "edit.json", edit)
    publish_cut(project)
    title = {"id": "agent-title", "kind": "title", "lane": "agent", "lines": lines,
             "start": {"word": "t1:0"}, "hold_s": 1}
    result = add(project, title, lane="agent").require_valid()
    findings = [item for item in result.findings if item.rule == "title_words"]
    assert bool(findings) is warns
    for item in findings:
        assert item.severity == "warning"
        assert "the creator's words" in item.suggested_action
        assert "word boundaries" in item.suggested_action
        assert "creator asked" in item.suggested_action


@pytest.mark.parametrize("changes", [
    {"music_sha256": "0" * 64}, {"schema": "wrong"}, {"beats": {}},
    {"beats": [{"time_s": 6.3, "frame": 188, "strength": 0.8}]},
    {"beats": [{"time_s": 188 / 30, "frame": True, "strength": 0.8}]},
    {"beats": [{"time_s": 188 / 30, "frame": 188, "strength": 2}]},
    {"beats": [{"time_s": float("nan"), "frame": 188, "strength": 0.8}]},
])
def test_stale_or_malformed_hints_are_ignored(project: Path, changes: dict[str, Any]) -> None:
    beat_style(project)
    add(project, graphic(), lane="agent").require_valid()
    before = resolve_timeline(project).to_dict()
    publish_beats(project, (188,), changes=changes)
    assert resolve_timeline(project).to_dict() == before


@pytest.mark.parametrize("changed", ["source", "bed", "unowned", "no_bed", "none", "cut"])
def test_beat_hints_need_a_current_owned_file_bed(project: Path, changed: str) -> None:
    beat_style(project)
    add(project, graphic(), lane="agent").require_valid()
    before = resolve_timeline(project).to_dict()
    publish_beats(project, (188,))
    if changed == "source":
        (project / "track.wav").write_bytes(b"changed synthetic source")
    elif changed == "bed":
        (project / "music/bed.wav").write_bytes(b"changed synthetic bed")
    elif changed == "no_bed":
        (project / "music/bed.wav").unlink()
    elif changed == "none":
        data = json.loads((project / "project.json").read_bytes())
        data["music"] = {"mode": "none"}
        write(project / "project.json", data)
    else:
        record = json.loads((project / "music/receipt.json").read_bytes())
        if changed == "unowned":
            record["outputs"] = [item for item in record["outputs"]
                                 if item["path"] != "music/beats.json"]
        else:
            record["inputs"]["cut:frames"] = "0" * 64
        write(project / "music/receipt.json", record)
    assert resolve_timeline(project).to_dict() == before


def test_clip_length_uses_snapped_window_in_planner_and_timeline(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_media_select import STAMP, _item
    from test_planner_broll import requests

    from yap_studio.errors import ValidationError
    from yap_studio.planner import broll

    source = project / "source.mov"
    info = replace(stage.ffmpeg.probe(source, count_frames=False), video_duration_s=1.02)
    monkeypatch.setattr(stage.ffmpeg, "probe", lambda *args, **kwargs: info)
    item = _item(duration=1.02, locations=[{
        "path": str(source), "added_at": STAMP, "seen_at": STAMP,
    }])
    item["sha256"] = receipt.sha256_file(source)
    with pytest.raises(ValidationError, match="source window must fit"):
        broll.plan_broll(requests(start={"time": 0}, end={"time": 1.02}),
                         load_cut_words(project / "cut/cut-words.json"), [item], {},
                         project_dir=project, frames=360)
    result = add(project, insert(project, start={"time": 0}, end={"time": 1.02}))
    assert not result.valid
    assert result.elements[0].end_frame == 31
    assert any(finding.rule == "insert_bounds" for finding in result.findings)
    assert not (project / "timeline/broll.json").exists()


@pytest.mark.parametrize("start,hold,beat,expected", [
    (0.1, 1, 0, 0), (11, 1, 334, 330), (11, 1, 326, 326),
    (0, 1, -1, 0), (11, 1, 361, 330),
])
def test_beat_snap_respects_cut_bounds_and_silent_hold(
    project: Path, start: float, hold: float, beat: int, expected: int,
) -> None:
    beat_style(project)
    add(project, graphic(start={"time": start}, hold_s=hold), lane="agent").require_valid()
    publish_beats(project, (beat,))
    result = resolve_timeline(project).require_valid()
    item = result.elements[0]
    assert item.start_frame == expected
    assert item.end_frame - item.start_frame == round(hold * 30)
    assert 0 <= item.start_frame < item.end_frame <= 360


def test_beat_snap_leaves_nested_word_steps_and_cut_documents_unchanged(project: Path) -> None:
    beat_style(project)
    item = graphic(slots={"steps": [{"kind": "type", "at": {"word": "t1:7"}}]})
    add(project, item, lane="agent").require_valid()
    files = ("edit.json", "cut/cut-words.json", "ingest/t1/words.json", "cut/receipt.json")
    before = {name: (project / name).read_bytes() for name in files}
    publish_beats(project, (188,))
    resolved = resolve_timeline(project).require_valid().elements[0]
    assert resolved.start_frame == 186
    assert resolved.payload["slots"]["steps"][0]["at"] == {"frame": 216, "time_s": 7.2}
    assert {name: (project / name).read_bytes() for name in files} == before


@pytest.mark.parametrize("beats", [(), (188,), (184,)])
def test_planned_word_cues_keep_word_timing_with_beats(
    project: Path, capsys: pytest.CaptureFixture[str], beats: tuple[int, ...],
) -> None:
    beat_style(project)
    style_path = project / "style.json"
    style = json.loads(style_path.read_bytes())
    style["graphics"] = {"card_kinds_on": ["choices"]}
    write(style_path, style)
    planned = cards.plan_cards({"schema": cards.REQUESTS_SCHEMA, "cards": [{
        "id": "agent-card", "template": "choices", "start": {"word": "t1:6"}, "hold_s": 3,
        "slots": {"options": ["One", "Two"], "cues": [0.2, {"word": "t1:7", "text": "eta"}]},
    }]}, cut_words=load_cut_words(project / "cut/cut-words.json"),
        style=ProjectContext(project).render_style(), frames=360)
    assert planned.cards[0].slots["cues"] == [0.2, 1.0]
    cards.write_plan(planned, project / "choices-plan")
    lane = project / "choices-plan/agent-lane.json"
    assert cli.main(["timeline", "add", str(lane), str(project), "--lane", "agent",
                     "--json"]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["elements"][0]["slots"]["cues"] == [0.2, 1.0]
    if beats:
        publish_beats(project, beats)
    assert cli.main(["timeline", "check", str(project), "--json"]) == 0
    checked = json.loads(capsys.readouterr().out)
    card = checked["elements"][0]
    assert card["start_frame"] == 186
    assert card["slots"]["cues"] == [0.2, 1.0]
    assert card["start_frame"] + round(card["slots"]["cues"][1] * 30) == 216
    assert checked["beat_snaps"] == []


@pytest.mark.parametrize("start,timing,cue_offset,numeric_cue,window,expected", [
    ({"time": 2.0}, {"hold_s": 3}, 0.05, 0.466667, (60, 150), [0.066667, 0.466667]),
    ({"word": "t1:2", "offset_s": 0.05}, {"hold_s": 2.05},
     0.35, 0.7, (62, 124), [0.3, 0.7]),
    ({"time": 2.05}, {"end": {"word": "t1:4", "edge": "start", "offset_s": 0.05}},
     0.35, 0.7, (62, 122), [0.3, 0.7]),
])
def test_planned_half_frame_cues_match_the_resolved_lane(
    project: Path, start: dict, timing: dict, cue_offset: float, numeric_cue: float,
    window: tuple[int, int], expected: list[float],
) -> None:
    from yap_studio.cards import card_layer_plan

    publish_cut(project, shift=0.2)
    style = ProjectContext(project).render_style()
    style.data["graphics"]["card_kinds_on"] = ["choices"]
    planned = cards.plan_cards({"schema": cards.REQUESTS_SCHEMA, "cards": [{
        "id": "agent-card", "template": "choices", "start": start, **timing,
        "slots": {"options": ["One", "Two"], "cues": [
            {"word": "t1:2", "text": "beta", "offset_s": cue_offset}, numeric_cue,
        ]},
    }]}, cut_words=load_cut_words(project / "cut/cut-words.json"), style=style, frames=360)
    cards.write_plan(planned, project / "half-frame-plan")
    resolved = add_timeline(project, project / "half-frame-plan/agent-lane.json",
                            lane="agent").require_valid().elements[0]
    card = planned.cards[0]
    assert (card.start_frame, card.end_frame) == window
    assert (resolved.start_frame, resolved.end_frame) == window
    assert card.slots["cues"] == resolved.payload["slots"]["cues"] == expected
    planned_frames = [card.start_frame + round(cue * 30) for cue in card.slots["cues"]]
    resolved_frames = [resolved.start_frame + round(cue * 30)
                       for cue in resolved.payload["slots"]["cues"]]
    assert planned_frames == resolved_frames
    card_layer_plan("choices", resolved.payload["slots"], style,
                    frames=resolved.end_frame - resolved.start_frame)


@pytest.mark.parametrize("beats", [(188,), (184,)])
def test_cue_anchors_prevent_beat_snaps(
    project: Path, beats: tuple[int, ...],
) -> None:
    beat_style(project)
    item = graphic(template="choices", hold_s=3, slots={
        "options": ["One", "Two"], "cues": [{"at": {"word": "t1:6"}}, 1.0],
    })
    add(project, item, lane="agent").require_valid()
    publish_beats(project, beats)
    result = resolve_timeline(project).require_valid()
    card = result.elements[0]
    assert card.start_frame == 186
    assert card.payload["slots"]["cues"] == [0.0, 1.0]
    assert result.beat_snaps == ()


@pytest.mark.parametrize("beat", [184, 188])
@pytest.mark.parametrize("schedule", ["anchored", "mixed", "numeric"])
def test_beat_snap_preserves_valid_cue_schedules(
    project: Path, beat: int, schedule: str,
) -> None:
    from yap_studio.cards import card_layer_plan

    beat_style(project)
    cues = [0.2, 0.6]
    if schedule == "anchored":
        cues = [{"at": {"word": "t1:6", "offset_s": 0.2}}, {"at": {"word": "t1:7"}}]
    elif schedule == "mixed":
        cues = [0.2, {"at": {"word": "t1:6", "offset_s": 0.6}}]
    add(project, graphic(template="choices", hold_s=2.3,
                         slots={"options": ["One", "Two"], "cues": cues}),
        lane="agent").require_valid()
    before = resolve_timeline(project).require_valid().elements[0]
    expected = [0.2, 1.0] if schedule == "anchored" else [0.2, 0.6]
    assert before.payload["slots"]["cues"] == expected
    style = ProjectContext(project).render_style()
    card_layer_plan("choices", before.payload["slots"], style, frames=69, fps=30)
    publish_beats(project, (beat,))
    result = resolve_timeline(project).require_valid()
    card = result.elements[0]
    card_layer_plan("choices", card.payload["slots"], style,
                    frames=card.end_frame - card.start_frame, fps=30)
    start = beat if schedule == "numeric" else 186
    assert (card.start_frame, card.end_frame) == (start, start + 69)
    assert card.payload["slots"]["cues"] == expected
    assert len(result.beat_snaps) == (schedule == "numeric")


def test_beat_style_defaults_and_valid_window_endpoints() -> None:
    settings = resolve_style_data({"schema_version": 1})["music"]
    assert settings["beat_snap"] is False and settings["beat_snap_window_ms"] == 150
    for window in (0, 150.5, 500):
        resolved = resolve_style_data({"schema_version": 1,
                                       "music": {"beat_snap": True, "beat_snap_window_ms": window}})
        assert resolved["music"]["beat_snap"] is True
        assert resolved["music"]["beat_snap_window_ms"] == window


@pytest.mark.parametrize("settings", [
    {"beat_snap": 1}, {"beat_snap": "true"}, {"beat_snap_window_ms": -1},
    {"beat_snap_window_ms": 501}, {"beat_snap_window_ms": True},
    {"beat_snap_window_ms": float("nan")}, {"beat_snap_window_ms": float("inf")},
])
def test_beat_style_rejects_invalid_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(StyleError):
        resolve_style_data({"schema_version": 1, "music": settings})


def test_word_frames_and_recut_removing_filler_before_anchor(project: Path) -> None:
    result = add(project, insert(project))
    item = result.require_valid().elements[0]
    assert (item.start_frame, item.end_frame, item.start_s, item.end_s) == (66, 108, 2.2, 3.6)
    before = (project / "timeline/broll.json").read_bytes()
    write(project / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "trim-one", "take": "t1", "start": 1, "end": 2,
            "kind": "filler", "status": "applied", "words": ["t1:1"],
        }],
    })
    publish_cut(project, removed=("t1:1",), shift=1)
    moved = resolve_timeline(project).require_valid().elements[0]
    assert (moved.start_frame, moved.end_frame) == (36, 78)
    assert (project / "timeline/broll.json").read_bytes() == before


def test_removed_anchor_names_neighbours_and_nearest_snap(project: Path) -> None:
    publish_cut(project, removed=("t1:2",))
    result = add(project, insert(project))
    assert not result.valid
    assert "t1:1 ('um'" in result.findings[0].message
    assert "t1:3 ('gamma'" in result.findings[0].message
    assert not (project / "timeline/broll.json").exists()
    item = insert(project, start={"word": "t1:2", "snap": "nearest"})
    result = add(project, item).require_valid()
    assert result.elements[0].start_frame == 36
    assert result.snapped_anchors == ({"element": "broll-one", "lane": "broll",
                                     "anchor": "start", "from_word": "t1:2",
                                     "word": "t1:1", "segment": "s1", "frame": 36},)


def test_nearest_never_crosses_segment(project: Path) -> None:
    data = json.loads((project / "edit.json").read_bytes())
    data["segments"] = [
        {"id": "s1", "take": "t1", "from": {"time": 2}, "to": {"time": 2.8}},
        {"id": "s2", "take": "t1", "from": {"time": 3}, "to": {"time": 12}},
    ]
    write(project / "edit.json", data)
    publish_cut(project, removed=("t1:2",))
    result = add(project, insert(project, start={"word": "t1:2", "snap": "nearest"}))
    assert not result.valid
    assert "none in the same segment" in result.findings[0].message


def test_reused_word_requires_segment_and_uses_end_edge(project: Path) -> None:
    data = json.loads((project / "edit.json").read_bytes())
    data["segments"].append({"id": "s2", "take": "t1", "from": {"time": 0},
                             "to": {"time": 2}})
    write(project / "edit.json", data)
    publish_cut(project, repeat=True)
    item = insert(project, start={"word": "t1:0"}, end={"word": "t1:1"})
    result = add(project, item)
    assert "ambiguous" in result.findings[0].message
    item["start"]["segment"] = "s2"
    item["end"]["segment"] = "s2"
    result = add(project, item).require_valid()
    assert (result.elements[0].start_frame, result.elements[0].end_frame) == (186, 228)


def test_edges_offsets_and_half_frame_ties(project: Path) -> None:
    result = add(project, insert(project, start={"word": "t1:2", "edge": "end",
                                               "offset_s": 0.05})).require_valid()
    assert result.elements[0].start_frame == 80
    assert result.elements[0].end_frame == 108


def test_guard_is_checked_before_snapping(project: Path) -> None:
    publish_cut(project, removed=("t1:2",))
    result = add(project, insert(project, start={"word": "t1:2", "text": "wrong",
                                               "snap": "nearest"}))
    assert not result.valid
    assert "'wrong'" in result.findings[0].message and "'beta'" in result.findings[0].message


def test_overlap_names_both_lanes_and_writes_nothing(project: Path) -> None:
    add(project, graphic(start={"word": "t1:2"}), lane="agent").require_valid()
    original = (project / "timeline/agent.json").read_bytes()
    result = add(project, insert(project))
    violation = next(item for item in result.findings if item.rule == "visual_overlap")
    assert set(violation.elements) == {"agent-card", "broll-one"}
    assert set(violation.lanes) == {"agent", "broll"}
    assert "--lane agent" in violation.suggested_action
    assert (project / "timeline/agent.json").read_bytes() == original
    assert not (project / "timeline/broll.json").exists()


def test_broll_cannot_replace_or_remove_agent(project: Path) -> None:
    add(project, graphic(), lane="agent").require_valid()
    original = (project / "timeline/agent.json").read_bytes()
    assert not add(project, insert(project, id="agent-card")).valid
    with pytest.raises(TimelineRefusal):
        remove_timeline(project, lane="broll", ids=("agent-card",))
    assert (project / "timeline/agent.json").read_bytes() == original


def test_add_takes_the_cards_plan_lane_only_while_its_words_match(
    project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    requests = write(project / "plans/cards.json", {
        "schema": "yap-studio/card-requests@1",
        "cards": [{"id": "agent-card", "template": "text_tag", "slots": {"text": "beta"},
                   "start": {"word": "t1:6"}, "hold_s": 2}],
    })
    frames = receipt_plan_frames(project)
    assert cli.main(["cards", "plan", str(requests), "--cut-words",
                     str(project / "cut/cut-words.json"), "--frames", str(frames),
                     "--style", "preset:clean_minimal", "--out", str(project / "plans/cards")]) == 0
    lane = project / "plans/cards/agent-lane.json"
    assert "words_sha256" in json.loads(lane.read_bytes())
    assert cli.main(["timeline", "add", str(lane), str(project), "--lane", "agent"]) == 0
    assert "passed" in capsys.readouterr().out
    added = load_timeline(project / "timeline/agent.json")
    assert [element.id for element in added.elements] == ["agent-card"]
    original = (project / "timeline/agent.json").read_bytes()
    stale = json.loads(lane.read_bytes())
    stale["words_sha256"]["t1"] = "0" * 64
    write(lane, stale)
    assert cli.main(["timeline", "add", str(lane), str(project), "--lane", "agent"]) == 3
    assert "planned against older words for t1" in capsys.readouterr().err
    assert (project / "timeline/agent.json").read_bytes() == original


def receipt_plan_frames(root: Path) -> int:
    """Read the cut's frame count where the edit skill tells agents to find it."""
    return json.loads((root / "cut/receipt.json").read_bytes())["plan"]["total_frames"]


def test_bad_element_aborts_whole_fragment_and_same_id_replaces(project: Path) -> None:
    add(project, insert(project)).require_valid()
    original = (project / "timeline/broll.json").read_bytes()
    bad = insert(project, id="broll-bad", start={"word": "t1:99"})
    assert not add(project, insert(project, start={"word": "t1:4"},
                                   end={"word": "t1:5"}), bad).valid
    assert (project / "timeline/broll.json").read_bytes() == original
    add(project, insert(project, start={"word": "t1:4"},
                        end={"word": "t1:5"})).require_valid()
    assert len(load_timeline(project / "timeline/broll.json").elements) == 1
    assert resolve_timeline(project).elements[0].start_frame == 126


def test_stale_cut_refuses(project: Path) -> None:
    edit = json.loads((project / "edit.json").read_bytes())
    edit["segments"][0]["to"]["time"] = 11
    write(project / "edit.json", edit)
    with pytest.raises(TimelineRefusal, match="stale"):
        resolve_timeline(project)


def test_changed_take_refuses_cli_check_with_recovery_action(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from test_stage_fingerprints import make_fingerprinted_project, replace_take

    root = make_fingerprinted_project(tmp_path)
    take = replace_take(root)
    assert cli.main(["timeline", "check", str(root), "--json"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "stale_cut"
    assert "t1" in error["message"] and take.name in error["message"]
    assert "its audio stream differs" in error["message"]
    assert "yap-studio ingest DIR --force" in error["suggested_action"]
    assert "yap-studio cut DIR" in error["suggested_action"]


def test_tampered_cut_map_refuses(project: Path) -> None:
    path = project / "cut/cut-words.json"
    data = json.loads(path.read_bytes())
    data["words"][0]["start"] = 0
    write(path, data)
    with pytest.raises(TimelineRefusal, match="changed output"):
        resolve_timeline(project)


def test_insert_hash_cache_reuses_identity_and_refuses_changed_media(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = project / "insert.mov"
    media.write_bytes((project / "source.mov").read_bytes())
    item = insert(project)
    item["media"]["path"] = media.name
    write(project / "timeline/broll.json", {
        "schema": "yap-studio/timeline@1",
        "words_sha256": load_cut_words(project / "cut/cut-words.json").words_sha256,
        "elements": [item],
    })
    original = receipt.sha256_file
    media_hashes = 0

    def counted_hash(path: Path) -> str:
        nonlocal media_hashes
        if Path(path).resolve() == media:
            media_hashes += 1
        return original(path)

    monkeypatch.setattr(receipt, "sha256_file", counted_hash)
    resolve_timeline(project).require_valid()
    assert media_hashes == 1
    resolve_timeline(project).require_valid()
    assert media_hashes == 1

    before = media.stat()
    changed = bytearray(media.read_bytes())
    changed[-1] ^= 1
    media.write_bytes(changed)
    os.utime(media, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    result = resolve_timeline(project)
    assert media_hashes == 2
    assert not result.valid
    assert any(finding.rule == "media_hash" and finding.severity == "violation"
               for finding in result.findings)
    assert not resolve_timeline(project).valid
    assert media_hashes == 2


def test_add_rechecks_media_exactly_after_cached_resolution(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = project / "insert.mov"
    media.write_bytes((project / "source.mov").read_bytes())
    item = insert(project)
    item["media"]["path"] = media.name
    original = stage._resolve

    def changed_after_resolution(context: ProjectContext, lanes: dict) -> stage.ResolvedTimeline:
        result = original(context, lanes)
        result.require_valid()
        changed = bytearray(media.read_bytes())
        changed[-1] ^= 1
        media.write_bytes(changed)
        return result

    monkeypatch.setattr(stage, "_resolve", changed_after_resolution)
    with pytest.raises(ConflictError):
        add(project, item)
    assert not (project / "timeline/broll.json").exists()


@pytest.mark.parametrize("case,rule", [
    ("hash", "media_hash"), ("missing", "insert_media"), ("bounds", "insert_bounds"),
    ("fast", "insert_speed"), ("slow", "insert_speed"), ("hold", "end_hold"),
    ("reverse", "duration"), ("negative", "inside_video"),
])
def test_insert_and_output_rules(project: Path, case: str, rule: str) -> None:
    item = insert(project)
    if case == "hash":
        item["media"]["sha256"] = "0" * 64
    elif case == "missing":
        item["media"]["path"] = "missing.mov"
    elif case == "bounds":
        item["media"]["in_s"] = 13
    elif case in {"fast", "slow"}:
        item["media"]["speed"] = 25 if case == "fast" else 0.4
    elif case == "hold":
        item["end"] = {"time": 13}
    elif case == "reverse":
        item["end"] = {"time": 1}
    else:
        item["start"] = {"word": "t1:0", "offset_s": -1}
    result = add(project, item)
    assert any(finding.rule == rule and finding.severity == "violation"
               for finding in result.findings)
    assert not (project / "timeline/broll.json").exists()


def test_insert_motion_fields_validate_in_fragment_and_lane(project: Path) -> None:
    result = add(project, insert(project, motion="push", push_scale=1.02))
    assert result.valid
    assert result.elements[0].payload["motion"] == "push"
    assert result.elements[0].payload["push_scale"] == 1.02
    saved = load_timeline(project / "timeline/broll.json")
    assert saved.elements[0].fields["push_scale"] == 1.02


@pytest.mark.parametrize("scale", [1.0, 1.2])
def test_insert_push_scale_outside_limit_is_rejected(project: Path, scale: float) -> None:
    from yap_studio.project.model import ProjectFileError

    with pytest.raises(ProjectFileError, match="push_scale"):
        add(project, insert(project, motion="push", push_scale=scale))


def test_insert_still_refuses_push_scale_with_finding(project: Path) -> None:
    result = add(project, insert(project, motion="still", push_scale=1.02))
    assert not result.valid
    assert any(finding.rule == "insert_motion" and "push_scale" in finding.message
               for finding in result.findings)


def test_image_uses_still_style_while_unmarked_clip_keeps_motion(project: Path) -> None:
    clip = add(project, insert(project)).require_valid().elements[0]
    assert "motion" not in clip.payload
    image = project / "diagram.ppm"
    image.write_bytes(b"P6\n32 32\n255\n" + np.full((32, 32, 3), 128, dtype=np.uint8).tobytes())
    config = json.loads((project / "project.json").read_bytes())
    config["style"] = "style.json"
    write(project / "project.json", config)
    write(project / "style.json", {"schema_version": 1, "name": "Held inserts",
                                   "broll": {"still_push": "none"}})
    still = add(project, insert(project, media={
        "path": image.name, "sha256": receipt.sha256_file(image),
    })).require_valid().elements[0]
    assert still.payload["motion"] == "still"


def test_fit_speed_and_external_media_warning(project: Path, tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external.mov"
    external.write_bytes((project / "source.mov").read_bytes())
    item = insert(project)
    item["media"].update(path=str(external), in_s=12.6, speed="fit")
    result = add(project, item).require_valid()
    assert result.elements[0].payload["media"]["speed"] == pytest.approx(1)
    assert any(finding.rule == "external_media" for finding in result.findings)


@pytest.mark.parametrize("kind", ["insert", "overlay"])
def test_heic_still_is_picture_without_clip_duration(project: Path, kind: str) -> None:
    if shutil.which("sips") is None:
        pytest.skip("sips is needed to generate a tiled HEIC still")
    png = project / "still.png"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=red:s=1280x720", "-frames:v", "1", "-threads", "1", str(png)],
               timeout_s=60)
    heic = project / "still.heic"
    subprocess.run(["sips", "-s", "format", "heic", str(png), "--out", str(heic)],
                   check=True, capture_output=True)
    item = (overlay if kind == "overlay" else insert)(project)
    item["media"].update(path=heic.name, sha256=receipt.sha256_file(heic),
                         in_s=100, speed="fit")
    result = add(project, item).require_valid()
    assert result.elements[0].payload["media"]["speed"] == 1
    assert not any(finding.rule == f"{kind}_media" for finding in result.findings)


@pytest.mark.parametrize(("kind", "speed", "audio", "valid"), [
    ("insert", 12, "mute", True),
    ("insert", 12, {"gain_db": 0}, False),
    ("insert", 25, "mute", False),
    ("overlay", 3, "mute", False),
])
def test_timelapse_speed_requires_muted_insert(
    project: Path, kind: str, speed: int, audio: object, valid: bool,
) -> None:
    item = (overlay if kind == "overlay" else insert)(project)
    item["start"] = {"time": 2}
    item["end"] = {"time": 3}
    item["media"]["speed"] = speed
    if kind == "insert":
        item["audio"] = audio
    result = add(project, item)
    assert result.valid is valid
    findings = [finding for finding in result.findings
                if finding.rule == f"{kind}_speed" and finding.severity == "violation"]
    assert bool(findings) is not valid
    if audio != "mute" and findings:
        assert "mute" in findings[0].message


@pytest.mark.parametrize("audio", ["mute", {"gain_db": 0}])
def test_fit_speed_obeys_timelapse_audio_rule(project: Path, audio: object) -> None:
    item = insert(project, start={"time": 2}, end={"time": 3}, audio=audio)
    item["media"].update(in_s=11, speed="fit")
    result = add(project, item)
    assert result.elements[0].payload["media"]["speed"] == pytest.approx(3)
    assert result.valid is (audio == "mute")
    assert any(f.rule == "insert_speed" for f in result.findings) is (audio != "mute")


@pytest.mark.parametrize("kind,duration,scale,rule,severity", [
    ("zoom", 2, 1.08, "zoom_length", "violation"),
    ("zoom", 3, 1.11, "zoom_peak", "violation"),
    ("graphic", 4, None, "graphic_hold", "warning"),
    ("graphic", 9, None, "graphic_hold", "violation"),
])
def test_visual_length_and_zoom_rules(project: Path, kind: str, duration: float,
                                    scale: float | None, rule: str, severity: str) -> None:
    item = graphic(start={"time": 0}, hold_s=duration)
    if kind == "zoom":
        item = {"id": "agent-zoom", "kind": kind, "lane": "agent", "start": {"time": 0},
                "hold_s": duration, "peak_scale": scale, "center": [0.5, 0.5]}
    result = add(project, item, lane="agent")
    assert any(finding.rule == rule and finding.severity == severity for finding in result.findings)
    assert result.valid == (severity == "warning")
    if kind == "zoom":
        assert sum(f.rule == "zoom_subtle" for f in result.findings) == 1


def test_zoom_count_and_project_rule_override(project: Path) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"zoom_max_per_video": 1, "zoom_min_s": 1, "zoom_max_s": 2,
                     "zoom_step_min": 0}
    write(project / "project.json", data)
    elements = [{"id": f"agent-zoom-{index}", "kind": "zoom", "lane": "agent",
                 "start": {"time": index * 2}, "hold_s": 1,
                 "peak_scale": 1.08, "center": [0.5, 0.5]} for index in range(2)]
    result = add(project, *elements, lane="agent")
    assert [f.rule for f in result.findings if f.rule != "zoom_style"] == ["zoom_count"]


def test_montage_parts_touch_and_warn_on_length(project: Path) -> None:
    one = insert(project, start={"time": 0}, end={"time": 1}, group="sequence")
    two = insert(project, id="broll-two", start={"time": 1}, end={"time": 2}, group="sequence")
    assert not add(project, one, two).require_valid().findings
    two["end"] = {"time": 3}
    result = add(project, two).require_valid()
    assert result.findings[0].rule == "insert_length"


def test_montage_group_exception_and_insert_exclusivity(project: Path) -> None:
    one = insert(project, start={"time": 0}, end={"time": 1}, group="sequence")
    two = insert(project, id="broll-two", start={"time": 0.5}, end={"time": 1.5},
                 group="sequence")
    assert add(project, one, two).valid
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"elements_at_once_max": 2}
    write(project / "project.json", data)
    result = add(project, graphic(start={"time": 0}, hold_s=2), lane="agent")
    assert not result.valid
    assert any(finding.rule == "visual_overlap" for finding in result.findings)


def test_nested_graphic_steps_are_resolved_and_bounded(project: Path) -> None:
    item = graphic(slots={"steps": [{"kind": "type", "at": {"word": "t1:7"}}]})
    result = add(project, item, lane="agent").require_valid()
    assert result.elements[0].payload["slots"]["steps"][0]["at"] == {"frame": 216, "time_s": 7.2}
    item["slots"]["steps"][0]["at"] = {"word": "t1:11"}
    assert any(finding.rule == "step_bounds"
               for finding in add(project, item, lane="agent").findings)


def test_remove_only_own_lane_and_clear_while_cut_stale(project: Path) -> None:
    add(project, graphic(), lane="agent").require_valid()
    add(project, insert(project)).require_valid()
    agent = (project / "timeline/agent.json").read_bytes()
    assert remove_timeline(project, lane="broll", ids=("broll-one",))["removed"] == ["broll-one"]
    add(project, insert(project)).require_valid()
    (project / "edit.json").write_text("{}", encoding="utf-8")
    assert remove_timeline(project, lane="broll")["removed"] == ["broll-one"]
    assert (project / "timeline/agent.json").read_bytes() == agent


def test_concurrent_writers_conflict_instead_of_losing_updates(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    original = stage._resolve

    def coordinated(context: ProjectContext, lanes: dict) -> stage.ResolvedTimeline:
        result = original(context, lanes)
        barrier.wait(timeout=30)
        return result

    monkeypatch.setattr(stage, "_resolve", coordinated)
    fragments = [fragment(project, insert(project, id=f"broll-{index}",
                                          start={"time": index * 3},
                                          end={"time": index * 3 + 1}),
                          name=f"fragment-{index}.json") for index in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(add_timeline, project, file, lane="broll") for file in fragments]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except ConflictError:
                outcomes.append(None)
    assert sum(result is not None for result in outcomes) == 1
    assert len(load_timeline(project / "timeline/broll.json").elements) == 1


def test_resolved_payload_contract_and_validation_gate(project: Path) -> None:
    item = insert(project, matches="beta", provenance={"kind": "generated"})
    result = add(project, item).require_valid()
    payload = result.to_dict()["elements"][0]
    assert {"id", "kind", "lane", "start_frame", "end_frame", "start_s", "end_s", "layer",
            "media", "matches", "provenance"} <= payload.keys()
    payload["media"]["path"] = "changed.mov"
    assert result.elements[0].payload["media"]["path"] == "source.mov"
    invalid = add(project, graphic(start={"word": "t1:2"}), lane="agent")
    with pytest.raises(TimelineRefusal):
        invalid.require_valid()
    assert load_cut_words(project / "cut/cut-words.json").words[2].id == "t1:2"


def test_style_swap_reports_agent_fix_without_changing_broll(project: Path) -> None:
    write(project / "style.json", {"schema_version": 1, "name": "Synthetic",
                                   "hook": {"duration_s": 1}})
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write(project / "project.json", data)
    title = {"id": "agent-title", "kind": "title", "lane": "agent",
             "lines": ["alpha"], "start": {"time": 0}}
    add(project, title, lane="agent").require_valid()
    add(project, insert(project)).require_valid()
    broll = (project / "timeline/broll.json").read_bytes()
    write(project / "style.json", {"schema_version": 1, "name": "Synthetic",
                                   "hook": {"duration_s": 4}})
    from yap_studio.project.status import status

    result = status(project)
    conflicts = [item for item in result["needs_human"] if "visual elements overlap" in item]
    assert len(conflicts) == 1
    assert "agent-title" in conflicts[0] and "broll-one" in conflicts[0]
    assert "--lane agent" in conflicts[0] and "hold_s" in conflicts[0]
    assert (project / "timeline/broll.json").read_bytes() == broll
    fixed = copy.deepcopy(title)
    fixed["hold_s"] = 1
    assert add(project, fixed, lane="agent").valid


def zoom(**changes: Any) -> dict[str, Any]:
    return {"id": "agent-zoom", "kind": "zoom", "lane": "agent", "start": {"time": 1},
            "end": {"time": 3}, "move": "punch", "scale": 1.2, "center": [.5, .42],
            **changes}


def opening(**changes: Any) -> dict[str, Any]:
    item = zoom(**{"id": "agent-opening", "move": "opening", "start": {"word": "t1:0"},
                   **changes})
    item.pop("end")
    item.pop("scale")
    return item


def opening_story(root: Path, rows: list[tuple[str, float, float]],
                  cuts: tuple[int, ...] = (), *, source_gap: float = .5) -> None:
    """Publish synthetic word boundaries and exact cuts on the output clock."""
    words = write(root / "staged-opening-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output",
        "words_sha256": {"t1": receipt.sha256_file(root / "ingest/t1/words.json")},
        "words": [{"id": f"t1:{index}", "segment": "s1", "text": text,
                   "start": start, "end": end}
                  for index, (text, start, end) in enumerate(rows)],
    })
    points = [0, *(frame / 30 for frame in cuts), 12]
    spans = write(root / "staged-opening-spans.json", {
        "sources": {"t1": "../source.mov"}, "fps": "30/1",
        "segments": [{"source": "t1", "start": str(start + index * source_gap),
                      "end": str(end + index * source_gap)}
                     for index, (start, end) in enumerate(
                         zip(points, points[1:], strict=False))],
    })
    with StageRun(root, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.replace_output(root / "cut/cut-words.json", words)
        run.replace_output(root / "cut/spans.json", spans)
        run.finish(extra={"plan": {"total_frames": 360}})


@pytest.mark.parametrize("start", [{"word": "t1:0"}, {"time": 0}])
def test_opening_resolution(project: Path, start: dict) -> None:
    transition_story(project, join=45)
    result = add(project, opening(start=start), lane="agent").require_valid()
    item = result.elements[0]
    assert (item.start_frame, item.end_frame) == (0, 43)
    assert item.payload["opening"] == {"reason": "line", "word": "t1:5", "end_frame": 43,
                                       "end_scale": 1.3}
    assert item.payload["scale"] == 1.3
    assert item.payload["curve"] == [
        {"frame": 0, "scale": 1, "center": [.5, .42], "ease": "cut"},
        {"frame": 43, "scale": 1.3, "center": [.5, .42], "ease": "linear"},
    ]


@pytest.mark.parametrize("source_gap,end,reason,word", [
    (0, 90, "line", "t1:2"), (.5, 45, "cut", "t1:1"),
], ids=["source_contiguous", "picture_jump"])
def test_opening_ends_only_on_picture_jumps(project: Path, source_gap: float, end: int,
                                          reason: str, word: str) -> None:
    opening_story(project, [("alpha", 0, .4), ("um", .5, 1.4), ("beta.", 1.6, 3)],
                  (45,), source_gap=source_gap)
    item = add(project, opening(), lane="agent").require_valid().elements[0]
    assert (item.start_frame, item.end_frame) == (0, end)
    assert item.payload["opening"] == {"reason": reason, "word": word, "end_frame": end,
                                       "end_scale": 1.3}


@pytest.mark.parametrize("rows,cuts,end,reason,word,target", [
    ([("alpha", 0, .3), ("um.", .4, .9)], (), 27, "line", "t1:1", 1.3),
    ([("alpha", 0, 1), ("um.", 1.1, 3)], (), 90, "line", "t1:1", 1.3),
    ([("alpha", 0, .4), ("um", .5, 1.7), ("beta.", 2, 3)], (45,), 45, "cut", "t1:1", 1.3),
    ([("alpha", 0, .5), ("um,", .6, 1.2), ("beta", 1.3, 3), ("gamma.", 4.2, 5)], (),
     36, "clause", "t1:1", 1.3),
    ([("alpha", .2, 1), ("um.", 1.1, 4.2)], (), 126, "line", "t1:1", 1.3),
    ([("alpha", 0, .3), ("um.", .4, .9)], (), 27, "line", "t1:1", 1.5),
], ids=["short", "long", "cut", "clause", "first_word_clock", "custom_scale"])
def test_opening_boundary_and_speed(project: Path, rows: list, cuts: tuple[int, ...],
                                    end: int, reason: str, word: str, target: float) -> None:
    from yap_studio.zoom_curve import ZoomCurve, declared_moves, scale_motion

    opening_story(project, rows, cuts)
    authored = opening()
    if target != 1.3:
        authored["scale"] = target
    item = add(project, authored, lane="agent").require_valid().elements[0]
    maximum = min(target, target ** (end / 31))
    assert (item.start_frame, item.end_frame) == (0, end)
    assert item.payload["opening"] == {"reason": reason, "word": word, "end_frame": end,
                                       "end_scale": maximum}
    assert item.payload["scale"] == target
    curve = ZoomCurve.from_payload(0, end, item.payload)
    assert curve.at(0)[0] == 1 and curve.at(end)[0] == pytest.approx(maximum)
    scales = curve.scales(0, end + 1)
    ratio = target ** (1 / 31) if end <= 31 else target ** (1 / end)
    assert [b / a for a, b in zip(scales[:end - 1], scales[1:end], strict=True)] == (
        pytest.approx([ratio] * (end - 1)))
    assert scales[end - 2] < scales[end - 1] < maximum and scales[end] == 1
    assert scale_motion(scales, 0, declared_moves(curve)) == ()


@pytest.mark.parametrize("move,kind,overlap", [
    ("opening", "title", False), ("punch", "title", True),
    ("opening", "graphic", True),
])
def test_opening_title_exception(project: Path, move: str, kind: str, overlap: bool) -> None:
    transition_story(project, join=45)
    other = ({"id": "agent-title", "kind": "title", "lane": "agent", "lines": ["alpha"]}
             if kind == "title" else graphic())
    other.update(start={"time": 0}, hold_s=1.5)
    item = opening() if move == "opening" else zoom(start={"time": 0}, end={"time": 1.5})
    result = add(project, item, other, lane="agent")
    assert any(f.rule == "visual_overlap" for f in result.findings) is overlap


@pytest.mark.parametrize("extra,rule", [("opening", "opening_count"), ("punch", "opening_overlap")])
def test_opening_conflicting_zooms(project: Path, extra: str, rule: str) -> None:
    transition_story(project, join=45)
    other = opening() if extra == "opening" else zoom()
    other["id"] = "agent-extra"
    result = add(project, opening(), other, lane="agent")
    finding = next(f for f in result.findings if f.rule == rule)
    assert "agent-extra" in finding.elements


def test_opening_later_start(project: Path) -> None:
    transition_story(project, join=45)
    finding = next(f for f in add(project, opening(start={"word": "t1:1"}),
                                 lane="agent").findings if f.rule == "opening_start")
    assert finding.bare_message == "an opening push starts on the video's first word"
    assert "t1:0" in finding.suggested_action


def test_opening_count_uses_the_document_lane(project: Path) -> None:
    transition_story(project, join=45)
    elements = [opening(id=identifier) for identifier in (
        "agent-opening", "agent-second", "agent-third")]
    for item in elements:
        item.pop("lane")
    write(project / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1",
        "words_sha256": load_cut_words(project / "cut/cut-words.json").words_sha256,
        "elements": elements,
    })
    finding = next(f for f in resolve_timeline(project).findings if f.rule == "opening_count")
    assert finding.elements == ("agent-second", "agent-third")
    assert finding.lanes == ("agent", "agent")
    assert "hold_s" not in finding.suggested_action


def test_opening_no_boundary_is_skipped(project: Path) -> None:
    opening_story(project, [("alpha", 0, 1), ("um", 1.1, 3), ("beta", 3.1, 5)])
    title = {"id": "agent-title", "kind": "title", "lane": "agent", "lines": ["alpha"],
             "start": {"time": 0}, "hold_s": 1}
    result = add(project, opening(), title, lane="agent").require_valid()
    assert [item.kind for item in result.elements] == ["title"]
    warning = next(f for f in result.findings if f.rule == "opening_skipped")
    assert warning.severity == "warning"
    assert warning.bare_message == (
        "the first line has no sentence end, clause end or cut between 0.8 and 4 s, "
        "so the opening push is skipped")
    assert warning.suggested_action == (
        "Leave the opening push out, or keep it off for this video.")
    assert not any(f.rule == "opening_fit" for f in result.findings)


def test_opening_skip_keeps_other_findings(project: Path) -> None:
    opening_story(project, [("alpha", 0, 1), ("um", 1.1, 3), ("beta", 3.1, 5)])
    result = add(project, opening(), graphic(start={"time": 11}, hold_s=2), lane="agent")
    assert [item.kind for item in result.elements] == ["graphic"]
    assert {"opening_skipped", "inside_video"} <= {f.rule for f in result.findings}


@pytest.mark.parametrize("mode", ["none", "punches", "punches_and_push"])
@pytest.mark.parametrize("enabled", [False, True])
def test_opening_style(project: Path, monkeypatch: pytest.MonkeyPatch,
                       mode: str, enabled: bool) -> None:
    transition_story(project, join=45)
    style = resolve_style_data({"schema_version": 1})
    style["moves"].update(zoom_mode=mode, opening_push=enabled)
    monkeypatch.setattr(stage, "_style", lambda context: style)
    result = add(project, opening(), lane="agent").require_valid()
    warnings = [f for f in result.findings if f.rule == "zoom_style"]
    assert len(warnings) == (not enabled)
    if warnings:
        assert warnings[0].bare_message == "the style has the opening push off"


@pytest.mark.parametrize("field,value", [
    ("end", {"time": 1}), ("hold_s", 1), ("ease", "linear"), ("push_s", 1),
    ("keys", [{"at": {"time": 1}, "scale": 1.3}]), ("peak_scale", 1.3),
])
def test_opening_schema_refuses_authored_timing(field: str, value: Any) -> None:
    from yap_studio.project.model import ProjectFileError, parse_document

    with pytest.raises(ProjectFileError):
        parse_document("timeline", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                    "elements": [{**opening(), field: value}]})


def test_opening_schema_optional_framing() -> None:
    from yap_studio.project.model import parse_document

    item = opening()
    item.pop("center")
    parse_document("timeline", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                "elements": [item]})


@pytest.mark.parametrize("move,extra,frames", [
    ("punch", {}, [30, 32, 86, 90]),
    ("push", {}, [30, 90]),
    ("push", {"push_s": .5, "ease": "ease-in-out"}, [30, 45]),
    ("keys", {"keys": [{"at": {"word": "t1:2"}, "scale": 1.2},
                        {"at": {"time": 2.7}, "scale": 1.35, "ease": "linear"}]},
     [30, 66, 81]),
])
def test_move_zoom_resolution(project: Path, monkeypatch: pytest.MonkeyPatch,
                              move: str, extra: dict, frames: list[int]) -> None:
    calls = []
    monkeypatch.setattr(stage, "_zoom_faces", lambda context, *, measure: calls.append(measure))
    item = zoom(move=move, **extra)
    if move == "keys":
        item.pop("scale")
    result = add(project, item, lane="agent")
    payload = result.elements[0].payload
    assert [key["frame"] for key in payload["curve"]] == frames
    assert payload["center"] == [.5, .42] and payload["center_from"] == "set"
    assert calls == [False]
    assert not any(f.rule in {"zoom_keys", "words_hash", "zoom_motion"} for f in result.findings)


@pytest.mark.parametrize("measured", [False, True])
def test_zoom_auto_center(project: Path, monkeypatch: pytest.MonkeyPatch, measured: bool) -> None:
    from types import SimpleNamespace

    from yap_studio.faces.model import Box, FaceBox

    calls = []
    faces = SimpleNamespace(status="measured", frames=tuple(
        (FaceBox(Box(400, 400, 200, 200), 1, "detected"),) for _ in range(360)))

    def load(context, *, measure):
        calls.append(measure)
        return faces if measured else None

    monkeypatch.setattr(stage, "_zoom_faces", load)
    item = zoom()
    item.pop("center")
    result = add(project, item, zoom(id="agent-two", start={"time": 4}, end={"time": 6}),
                 lane="agent")
    assert calls == [True]
    payload = result.elements[0].payload
    assert payload["center_from"] == ("face" if measured else "default")
    assert payload["center"] == pytest.approx([500 / 1080, 460 / 1920] if measured else [.5, .42])
    assert any(f.rule == "zoom_center" and f.severity == "warning"
               for f in result.findings) is not measured


@pytest.mark.parametrize("fields,subtle", [
    ({"move": "punch", "scale": 1.15}, False),
    ({"move": "push", "scale": 1.2, "end": {"time": 6}}, False),
    ({"move": "push", "scale": 1.15, "end": {"time": 11}}, True),
    ({"peak_scale": 1.08}, True),
])
def test_zoom_deliberate_move_check(project: Path, fields: dict, subtle: bool) -> None:
    item = zoom(**fields)
    if "peak_scale" in fields:
        item.pop("move")
        item.pop("scale")
        item["end"] = {"time": 4}
    result = add(project, item, lane="agent")
    assert any(f.rule == "zoom_subtle" for f in result.findings) is subtle
    assert result.valid
    assert result.require_valid() is result
    assert all(f.severity == "warning" for f in result.findings if f.rule == "zoom_subtle")


@pytest.mark.parametrize("fields,message", [
    ({"peak_scale": 1.08, "start": {"time": 25.9}, "end": {"time": 28.9}},
     "the zoom at 0:25.9 changes the framing by 8%; under 15% reads as camera drift"),
    ({"move": "push", "scale": 1.15, "start": {"time": 90.8}, "end": {"time": 99.8}},
     "the zoom at 1:30.8 moves 15% over 9.0 s; slower than 0.03 per second reads as drift"),
    ({"move": "keys", "start": {"time": 25.9}, "end": {"time": 28.9}, "keys": [
        {"at": {"time": 25.9}, "scale": 1},
        {"at": {"time": 27}, "scale": 1.08, "ease": "snap"},
    ]}, "the zoom at 0:26.9 changes the framing by 8%; under 15% reads as camera drift"),
])
def test_zoom_subtle_warning_keeps_timeline_valid(
    project: Path, fields: dict, message: str,
) -> None:
    publish_cut(project, total_frames=3300)
    item = zoom(**fields)
    if "peak_scale" in fields:
        item.pop("move")
        item.pop("scale")
    elif fields["move"] == "keys":
        item.pop("scale")
    result = add(project, item, lane="agent")
    assert result.valid
    assert result.require_valid() is result
    warnings = [f for f in result.findings if f.rule != "zoom_style"]
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.rule == "zoom_subtle" and warning.severity == "warning"
    assert warning.bare_message == message
    assert warning.message == f"agent-zoom (lane agent): {message}"
    assert warning.suggested_action == "Use a punch or a push, or remove the zoom."


@pytest.mark.parametrize("keys", [
    [{"at": {"time": 3}, "scale": 1.2}],
    [{"at": {"time": 2}, "scale": 1.2}, {"at": {"time": 1.5}, "scale": 1.35}],
    [{"at": {"time": 2}, "scale": 1.2}, {"at": {"time": 2}, "scale": 1.35}],
])
def test_zoom_keys_reject_bad_frames(project: Path, keys: list) -> None:
    item = zoom(move="keys", keys=keys)
    item.pop("scale")
    assert any(f.rule == "zoom_keys" for f in add(project, item, lane="agent").findings)


@pytest.mark.parametrize("changes,rule", [
    ({"move": "push", "push_s": 3}, "zoom_keys"),
    ({"end": {"time": 1.7}}, "zoom_length"),
    ({"scale": 2.05}, "zoom_peak"),
])
def test_zoom_move_limits(project: Path, changes: dict, rule: str) -> None:
    assert any(f.rule == rule for f in add(project, zoom(**changes), lane="agent").findings)


def test_zoom_face_check(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from yap_studio.faces.model import Box, FaceBox

    faces = SimpleNamespace(status="measured", frames=tuple(
        (FaceBox(Box(0, 400, 200, 200), 1, "detected"),) for _ in range(360)))
    monkeypatch.setattr(stage, "_zoom_faces", lambda context, *, measure: faces)
    finding = next(f for f in add(project, zoom(), lane="agent").findings if f.rule == "zoom_face")
    assert "0:01.0" in finding.message and "lower the scale or set center" in finding.message


@pytest.mark.parametrize("changes", [
    {"peak_scale": 1.08}, {"keys": [{"at": {"time": 2}, "scale": 1.2}]},
    {"push_s": .5}, {"ease": "linear"},
    {"move": "keys", "keys": [{"at": {"time": 2}, "scale": 1.2}]},
])
def test_zoom_move_schema_refuses_mixed_fields(changes: dict) -> None:
    from yap_studio.project.model import ProjectFileError, parse_document

    with pytest.raises(ProjectFileError):
        parse_document("timeline", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                    "elements": [zoom(**changes)]}, lane="agent")


def test_legacy_zoom_schema_still_validates() -> None:
    from yap_studio.project.model import parse_document

    item = zoom(peak_scale=1.08)
    item.pop("move")
    item.pop("scale")
    assert parse_document("timeline", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                      "elements": [item]}, lane="agent")


def test_zoom_defaults_and_legacy_does_not_load_faces(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.project.model import RULE_DEFAULTS

    assert {key: RULE_DEFAULTS[key] for key in (
        "zoom_max_per_video", "zoom_scale_max", "zoom_step_min", "zoom_rate_min", "zoom_hold_min_s",
    )} == {"zoom_max_per_video": 8, "zoom_scale_max": 2.0, "zoom_step_min": .15,
           "zoom_rate_min": .03, "zoom_hold_min_s": .6}
    monkeypatch.setattr(stage, "_zoom_faces", lambda *args, **kwargs: pytest.fail("face load"))
    item = zoom(peak_scale=1.08)
    item.pop("move")
    item.pop("scale")
    result = add(project, item, lane="agent")
    assert [f.rule for f in result.findings if f.rule != "zoom_style"] == [
        "zoom_length", "zoom_subtle"]
    assert "under 15% reads as camera drift" in result.findings[-1].message


@pytest.mark.parametrize("kind", ["zoom", "graphic"])
def test_nested_word_hash_is_referenced(kind: str) -> None:
    from yap_studio.project.model import parse_document

    if kind == "zoom":
        item = zoom(move="keys", keys=[{"at": {"word": "t2:1"}, "scale": 1.2}])
        item.pop("scale")
    else:
        item = graphic(start={"time": 0}, template="choices", slots={
            "options": ["One", "Two"], "cues": [0, {"at": {"word": "t2:1"}}],
        })
    lane = parse_document("timeline", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                       "elements": [item]}, lane="agent")
    assert stage._referenced_takes(lane.elements[0]) == {"t2"}


@pytest.mark.parametrize("measure", [False, True])
def test_zoom_faces_passes_cached_identity_digest(
    project: Path, monkeypatch: pytest.MonkeyPatch, measure: bool,
) -> None:
    from types import SimpleNamespace

    from yap_studio.faces import project as face_project

    cut = project / "cut/cut.mp4"
    shutil.copyfile(project / "source.mov", cut)
    context = ProjectContext(project)
    digest = context.identity.sha256(cut)
    faces = SimpleNamespace(status="measured")
    calls = []
    sha256_file = receipt.sha256_file

    def hash_file(source):
        if Path(source) == cut:
            pytest.fail("cut must use the cached identity digest")
        return sha256_file(source)

    def load(root, source, *, digest=None):
        calls.append((root, source, digest))
        return faces

    def unexpected(*args, **kwargs):
        pytest.fail("wrong face cache loader")

    monkeypatch.setattr(receipt, "sha256_file", hash_file)
    monkeypatch.setattr(face_project, "cut_faces", load if measure else unexpected)
    monkeypatch.setattr(face_project, "cached_cut_faces", unexpected if measure else load)
    assert stage._zoom_faces(context, measure=measure) is faces
    assert stage._zoom_faces(context, measure=measure) is faces
    assert calls == [(project, cut, digest)] * 2


@pytest.mark.parametrize("ease,valid", [("cut", True), ("snap", False)])
def test_short_punch_hold_depends_on_ease(project: Path, ease: str, valid: bool) -> None:
    result = add(project, zoom(ease=ease, end={"time": 1.7}), lane="agent")
    assert result.valid is valid
    assert any(f.rule == "zoom_length" for f in result.findings) is not valid


@pytest.mark.parametrize("scale,rule", [(1, "zoom_peak"), (1.01, "zoom_subtle")])
def test_single_start_key_checks_scale(project: Path, scale: float, rule: str) -> None:
    item = zoom(move="keys", keys=[{"at": {"time": 1}, "scale": scale}])
    item.pop("scale")
    result = add(project, item, lane="agent")
    assert any(f.rule == rule for f in result.findings)
    assert result.valid is (rule == "zoom_subtle")


@pytest.mark.parametrize("measured", [False, True])
def test_zoom_key_centers_follow_moving_face(
    project: Path, monkeypatch: pytest.MonkeyPatch, measured: bool,
) -> None:
    from types import SimpleNamespace

    from yap_studio.faces.model import Box, FaceBox

    faces = SimpleNamespace(status="measured", frames=tuple(
        (FaceBox(Box(20 if frame < 60 else 960, 800, 100, 100), 1, "detected"),)
        for frame in range(360)))
    calls = []

    def load(context, *, measure):
        calls.append(measure)
        return faces if measured else None

    monkeypatch.setattr(stage, "_zoom_faces", load)
    item = zoom(move="keys", keys=[
        {"at": {"time": 1}, "scale": 1.2, "center": [0, .5]},
        {"at": {"time": 2}, "scale": 1.2, "center": [1, .5], "ease": "cut"},
    ])
    item.pop("scale")
    item.pop("center")
    result = add(project, item, lane="agent")
    assert not any(f.rule in {"zoom_face", "zoom_center"} for f in result.findings)
    assert calls == [False]
    assert [key["center"] for key in result.elements[0].payload["curve"]] == [[0, .5], [1, .5]]


def test_zoom_impossible_fit_reports_first_frame(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from yap_studio.faces.model import Box, FaceBox

    faces = SimpleNamespace(status="measured", frames=tuple(
        (FaceBox(Box(0, 0, 1000, 1800), 1, "detected"),) for _ in range(360)))
    monkeypatch.setattr(stage, "_zoom_faces", lambda context, *, measure: faces)
    item = zoom(scale=1.3)
    item.pop("center")
    result = add(project, item, lane="agent")
    findings = [f for f in result.findings if f.rule == "zoom_face"]
    assert len(findings) == 1
    assert "the face leaves the frame at 0:01.0" in findings[0].message


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("rule_limit,limit,scale", [(2.1, 2.0, 2.05), (1.25, 1.25, 1.3)])
def test_zoom_scale_limit_matches_renderer(
    project: Path, legacy: bool, rule_limit: float, limit: float, scale: float,
) -> None:
    data = json.loads((project / "project.json").read_text())
    data.setdefault("rules", {})["zoom_peak_max" if legacy else "zoom_scale_max"] = rule_limit
    write(project / "project.json", data)
    item = zoom(scale=scale)
    if legacy:
        item.pop("move")
        item["peak_scale"] = item.pop("scale")
        item["end"] = {"time": 4}
    result = add(project, item, lane="agent")
    findings = [f for f in result.findings if f.rule == "zoom_peak"]
    assert len(findings) == 1
    assert f"at most {limit:g}" in findings[0].message


@pytest.mark.parametrize("scale", [None, 2])
def test_punch_uses_style_scale(project: Path, monkeypatch: pytest.MonkeyPatch, scale) -> None:
    style = resolve_style_data({"schema_version": 1, **(
        {"moves": {"punch_scale": scale}} if scale is not None else {})})
    monkeypatch.setattr(stage, "_style", lambda context: style)
    item = zoom()
    item.pop("scale")
    result = add(project, item, lane="agent").require_valid()
    assert result.elements[0].payload["scale"] == (scale or 1.5)
    assert max(key["scale"] for key in result.elements[0].payload["curve"]) == (scale or 1.5)


def test_two_x_zoom_is_valid(project: Path) -> None:
    result = add(project, zoom(scale=2), lane="agent").require_valid()
    assert not any(f.rule in {"zoom_peak", "zoom_motion"} for f in result.findings)


@pytest.mark.parametrize("highlight", ["none", "giant"])
@pytest.mark.parametrize("position", ["bottom", "middle"])
def test_zoom_chin_clearance(
    project: Path, monkeypatch: pytest.MonkeyPatch, position, highlight,
) -> None:
    from types import SimpleNamespace

    from yap_studio.captions import caption_layout_settings
    from yap_studio.faces.model import Box, FaceBox
    from yap_studio.stages.captions import caption_band_geometry

    style = resolve_style_data({"schema_version": 1, "captions": {
        "position": position, "highlight_style": highlight,
        **({"size_px": 120} if highlight == "giant" else {}),
    }})
    settings = caption_layout_settings(style["captions"], style["brand"])
    bottom = min(caption_band_geometry(
        settings, safe_bottom_px=style["internal"]["caption_safe_bottom_px"],
        backing=backing).y for backing in ((), ("box",), ("shadow",)))
    face = Box(400, (bottom + 960) / 2 - 190, 200, 200)
    faces = SimpleNamespace(frames=((FaceBox(face, 1, "detected"),),) * 360)
    monkeypatch.setattr(stage, "_style", lambda context: style)
    monkeypatch.setattr(stage, "_zoom_faces", lambda context, *, measure: faces)
    item = zoom(scale=2, center=[.5, .5])
    findings = [f for f in add(project, item, lane="agent").findings if f.rule == "zoom_face"]
    if position == "bottom":
        assert len(findings) == 1
        assert "the chin reaches the captions at" in findings[0].message
        assert "use a 1.5x punch here or set center" in findings[0].message
    else:
        assert not findings
    item.pop("center")
    result = add(project, item, lane="agent").require_valid()
    assert not any(f.rule in {"zoom_face", "zoom_motion"} for f in result.findings)
    if position == "bottom":
        cy = result.elements[0].payload["center"][1] * 1920
        assert 2 * face.y1 - cy <= bottom - 24 + 1e-9


def test_zoom_auto_center_uses_style_face_clearance(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from yap_studio.faces.model import Box, FaceBox
    from yap_studio.stages.captions import caption_band_geometry

    style = resolve_style_data({"schema_version": 1, "captions": {"position": "bottom"},
                                "graphics": {"face_clearance_px": 40}})
    bottom = min(caption_band_geometry(
        style["captions"], safe_bottom_px=style["internal"]["caption_safe_bottom_px"],
        backing=backing).y for backing in ((), ("box",), ("shadow",)))
    face = Box(400, bottom - 200, 200, 200)
    faces = SimpleNamespace(frames=((FaceBox(face, 1, "detected"),),) * 360)
    monkeypatch.setattr(stage, "_style", lambda context: style)
    monkeypatch.setattr(stage, "_zoom_faces", lambda context, *, measure: faces)
    item = zoom(scale=2)
    item.pop("center")
    result = add(project, item, lane="agent").require_valid()
    payload = result.elements[0].payload
    assert payload["center_from"] == "face"
    cy = payload["center"][1] * 1920
    assert 2 * face.y1 - cy <= bottom - 44 + 1e-9


def test_zoom_auto_center_lifts_chin_already_in_caption_band(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from yap_studio.faces.model import Box, FaceBox
    from yap_studio.stages.captions import caption_band_geometry

    style = resolve_style_data({"schema_version": 1})
    bottom = min(caption_band_geometry(
        style["captions"], safe_bottom_px=style["internal"]["caption_safe_bottom_px"],
        backing=backing).y for backing in ((), ("box",), ("shadow",)))
    face = Box(400, bottom - 140, 200, 200)
    faces = SimpleNamespace(frames=((FaceBox(face, 1, "detected"),),) * 360)
    monkeypatch.setattr(stage, "_zoom_faces", lambda context, *, measure: faces)
    item = zoom(scale=1.5)
    item.pop("center")
    result = add(project, item, lane="agent").require_valid()
    assert not any(f.rule == "zoom_face" for f in result.findings)
    payload = result.elements[0].payload
    assert payload["center_from"] == "face" and payload["curve"][0]["scale"] == 1
    cy = payload["center"][1] * 1920
    assert 1.5 * face.y1 - .5 * cy <= bottom - 24 + 1e-9


@pytest.mark.parametrize("mode,message,allowed", [
    ("none", "the style has zooms off", ()),
    ("punches", "the style allows punches only", ("punch",)),
    ("punches_and_push", "the style allows punches and a slow push only", ("punch", "push")),
])
def test_zoom_style_modes(project: Path, monkeypatch: pytest.MonkeyPatch,
                          mode: str, message: str, allowed: tuple) -> None:
    style = resolve_style_data({"schema_version": 1})
    style["moves"] = {"zoom_mode": mode, "punch_scale": 1.5}
    monkeypatch.setattr(stage, "_style", lambda context: style)
    for move in ("punch", "push", "keys", "legacy"):
        item = zoom(move=move)
        if move in {"keys", "legacy"}:
            item.pop("scale")
            if move == "keys":
                item["keys"] = [{"at": {"time": 2}, "scale": 1.5}]
            else:
                item.pop("move")
                item.update(peak_scale=1.08, end={"time": 4})
        result = add(project, item, lane="agent").require_valid()
        assert not any(f.rule == "zoom_motion" for f in result.findings)
        warnings = [f for f in result.findings if f.rule == "zoom_style"]
        assert len(warnings) == (move not in allowed)
        if warnings:
            assert warnings[0].severity == "warning" and warnings[0].bare_message == message
            assert warnings[0].suggested_action == (
                "Use zooms the style allows, or keep this one only when a review "
                "comment asked for it.")


@pytest.mark.parametrize("kind", ["jump", "jerk"])
def test_zoom_motion_violation(project: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    from yap_studio.zoom_curve import MotionFlaw

    monkeypatch.setattr(stage, "scale_motion", lambda *args: (MotionFlaw(32, kind, .2),))
    result = add(project, zoom(), lane="agent")
    finding = next(f for f in result.findings if f.rule == "zoom_motion")
    assert finding.severity == "violation" and "0:01.1" in finding.message
    assert ("jumps 22%" if kind == "jump" else "stutters") in finding.message
    assert finding.suggested_action == (
        "Build the move from punch, push or keys with cut or snap eases.")


def stock_style(project: Path, enabled: bool = True) -> None:
    write(project / "stock-style.json", {"schema_version": 1,
                                         "broll": {"stock_clips": "ask" if enabled else "off"}})
    document = json.loads((project / "project.json").read_bytes())
    document["style"] = "stock-style.json"
    write(project / "project.json", document)


def approve_stock(project: Path, digest: str) -> None:
    write(project / "stock/clips.json", {"schema": "yap-studio/stock-clips@1",
                                         "clips": [{"sha256": digest}]})


@pytest.mark.parametrize("kind", ["insert", "overlay", "background"])
def test_timeline_add_requires_stock_approval_and_style(project: Path, kind: str) -> None:
    from yap_studio.style.library import config_root

    stock_style(project)
    cache = config_root() / "assets/stock/pexels/101.mp4"
    cache.parent.mkdir(parents=True)
    shutil.copyfile(project / "source.mov", cache)
    digest = receipt.sha256_file(cache)
    item = insert(project, id="broll-stock-dog", kind=kind,
                  media={"path": str(cache), "sha256": digest})
    if kind == "overlay":
        item["placement"] = "headroom"
    incoming = fragment(project, item)
    target = project / "timeline/broll.json"
    with pytest.raises(TimelineRefusal,
                       match="broll-stock-dog uses a stock clip.*not approved") as error:
        add_timeline(project, incoming, lane="broll")
    assert error.value.suggested_action == "Ask with yap-studio stock search, then stock get."
    assert not target.exists()
    approve_stock(project, digest)
    add_timeline(project, incoming, lane="broll").require_valid()
    before = target.read_bytes()
    stock_style(project, enabled=False)
    with pytest.raises(TimelineRefusal, match="Stock clips are off in this project's style"):
        add_timeline(project, incoming, lane="broll")
    assert target.read_bytes() == before


def test_timeline_add_recognizes_catalog_stock_at_another_path(project: Path) -> None:
    from yap_studio.media.catalog import add_stock_item

    stock_style(project)
    source = project / "source.mov"
    digest = receipt.sha256_file(source)
    add_stock_item(source, stock={"provider": "pexels", "clip_id": "101",
        "page_url": "https://www.pexels.com/video/101/", "author": "Alex",
        "author_url": "https://www.pexels.com/@alex/", "license": "Pexels License",
        "license_url": "https://www.pexels.com/license/",
        "file_url": "https://videos.pexels.com/video-files/101/clip.mp4",
        "downloaded_at": "2026-10-18T12:00:00Z"}, description_text="stock clip")
    incoming = fragment(project, insert(project))
    with pytest.raises(TimelineRefusal, match="not approved"):
        add_timeline(project, incoming, lane="broll")
    approve_stock(project, digest)
    add_timeline(project, incoming, lane="broll").require_valid()


def test_timeline_add_surfaces_unreadable_catalog(project: Path) -> None:
    from yap_studio.errors import InputError
    from yap_studio.media.catalog import catalog_path

    catalog = catalog_path()
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("invalid catalog")
    with pytest.raises(InputError, match="not a valid media catalog"):
        add_timeline(project, fragment(project, insert(project)), lane="broll")


@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow", "together"])
@pytest.mark.parametrize("template,rows", [
    ("specs_list", [{"name": "First", "value": "12"}, {"name": "Second", "value": "25"}]),
    ("stat", ["12 first", "25 second"]), ("stat", []), ("text_tag", []),
])
@pytest.mark.parametrize("duration", [7, 8, 8.1])
def test_specs_rows_timeline_hold(project: Path, look: str, template: str,
                                 rows: list, duration: float) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write(project / "project.json", data)
    write(project / "style.json", {"schema_version": 1, "graphics": {"specs_rows": look}})
    slots = ({"title": "Plan", "rows": rows} if template == "specs_list"
             else {"end": 37, **({"rows": rows} if rows and look != "together" else {})}
             if template == "stat" else {"text": "Hello"})
    result = add(project, graphic(template=template, slots=slots, start={"time": 0},
                                  hold_s=duration), lane="agent")
    holds = [finding for finding in result.findings if finding.rule == "graphic_hold"]
    allowed = look != "together" and (template == "specs_list" or template == "stat" and rows)
    assert [finding.severity for finding in holds] == (
        ["violation"] if duration > 8 else [] if allowed else ["warning"])


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
@pytest.mark.parametrize("beats", [(), (28,), (32,)])
def test_specs_rows_plan_and_resolve_across_jump_cut(
    project: Path, template: str, look: str, beats: tuple[int, ...],
) -> None:
    from test_cards import ROWS_ON_CUE_SLOTS

    from yap_studio.planner.cards import REQUESTS_SCHEMA, plan_cards
    from yap_studio.project.model import parse_anchor
    from yap_studio.style import resolve

    edit = json.loads((project / "edit.json").read_bytes())
    edit["segments"] = [
        {"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"time": 4}},
        {"id": "s2", "take": "t1", "from": {"time": 6}, "to": {"time": 12}},
    ]
    write(project / "edit.json", edit)
    publish_cut(project, total_frames=300)
    mapped = json.loads((project / "cut/cut-words.json").read_bytes())
    for word in mapped["words"]:
        if word["segment"] == "s2":
            word["start"] -= 2
            word["end"] -= 2
    staged = write(project / "staged-jump-words.json", mapped)
    with StageRun(project, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.replace_output(project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": {"total_frames": 300, "end_hold_s": 1.5}})
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = look
    style.data["music"]["dry_open_s"] = 0
    style.data["music"]["beat_snap"] = True
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write(project / "project.json", data)
    write(project / "style.json", style.data)
    request = {"id": "agent-answer", "template": template, "start": {"time": 1},
               "end": {"time": 8}, "slots": {**ROWS_ON_CUE_SLOTS[template],
                   "cues": [{"word": "t1:2"}, {"word": "t1:6"}]}}
    planned = plan_cards({"schema": REQUESTS_SCHEMA, "cards": [request]},
                         load_cut_words(project / "cut/cut-words.json"), style, frames=300)
    assert not planned.warnings
    assert planned.cards[0].slots["cues"] == [1.2, 3.2]
    cards.write_plan(planned, project / "rows-plan")
    lane = json.loads((project / "rows-plan/agent-lane.json").read_bytes())
    assert lane["elements"][0]["slots"]["cues"] == [
        {"at": parse_anchor(cue).to_dict()} for cue in request["slots"]["cues"]]
    if beats:
        publish_beats(project, beats)
    resolved = add(project, lane["elements"][0], lane="agent").require_valid()
    assert (resolved.elements[0].start_frame, resolved.elements[0].end_frame) == (30, 240)
    assert resolved.elements[0].payload["slots"]["cues"] == [1.2, 3.2]
    assert resolved.beat_snaps == ()
    checked = resolve_timeline(project).require_valid()
    assert checked.elements[0].payload["slots"]["cues"] == [1.2, 3.2]
    assert not any(f.rule == "graphic_hold" for f in resolved.findings)


@pytest.mark.parametrize("handling,warnings", [("punch", 0), ("smooth", 0), ("hard", 1)])
def test_jump_cut_punch_allows_zoom_with_zooms_off(
    project: Path, monkeypatch: pytest.MonkeyPatch, handling: str, warnings: int,
) -> None:
    style = resolve_style_data({"schema_version": 1})
    style["moves"].update(zoom_mode="none", jump_cuts=handling)
    monkeypatch.setattr(stage, "_style", lambda context: style)
    result = add(project, zoom(move="punch"), lane="agent").require_valid()
    assert sum(f.rule == "zoom_style" for f in result.findings) == warnings


@pytest.fixture
def step_project(project, monkeypatch):
    from types import SimpleNamespace

    from test_step_in import pieces

    from yap_studio.faces.model import Box, FaceBox

    mapped = json.loads((project / "cut/cut-words.json").read_bytes())
    for index, word in enumerate(mapped["words"]):
        word.update(start=index, end=index + 1)
    staged = write(project / "step-words.json", mapped)
    inputs = cut_inputs(ProjectContext(project))
    with StageRun(project, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": {"total_frames": 360, "pieces": pieces((60,) * 6)}})
    style = resolve_style_data({"schema_version": 1, "moves": {"jump_cuts": "step_in"}})
    monkeypatch.setattr(stage, "_style", lambda context: style)
    measured = SimpleNamespace(frames=tuple(
        (FaceBox(Box(400, 400, 200, 250), 1, "detected"),) for _ in range(360)))
    monkeypatch.setattr(stage, "_zoom_faces", lambda context, *, measure: measured)
    return project


def test_step_engine_keys_and_clean_runs(step_project):
    from yap_studio.faces.model import Box
    from yap_studio.zoom_curve import ZoomCurve, declared_moves, face_leaves_frame, scale_motion

    result = resolve_timeline(step_project).require_valid()
    runs = [e for e in result.elements if e.lane == "engine"]
    assert [e.id for e in runs] == ["steps-1", "steps-2"]
    assert [k["frame"] for k in runs[0].payload["curve"]] == [60, 120, 180]
    assert all(k["ease"] == "cut" for e in runs for k in e.payload["curve"])
    assert result.to_dict()["steps"]["summary"] == dict(
        jumps=5, changed=5, step=4, reset=1, shift=0, hold=0)
    boxes = {i: (Box(400, 400, 200, 250),) for i in range(360)}
    for item in runs:
        curve = ZoomCurve.from_payload(item.start_frame, item.end_frame, item.payload)
        assert face_leaves_frame(curve, boxes, bottom=1500) is None
        assert scale_motion(curve.scales(item.start_frame - 1, item.end_frame + 1),
                            item.start_frame - 1, declared_moves(curve)) == ()


@pytest.mark.parametrize("kind", ["title", "graphic"])
def test_step_under_front_card_or_title(step_project, kind):
    data = json.loads((step_project / "project.json").read_bytes())
    data["rules"] = {"elements_at_once_max": 1, "zoom_max_per_video": 1}
    write(step_project / "project.json", data)
    item = graphic(start={"time": 2}, end={"time": 3}, layer="front")
    if kind == "title":
        item = dict(id="agent-title", kind="title", lane="agent", lines=["alpha"],
                     start={"time": 2}, end={"time": 3})
    result = add(step_project, item, lane="agent")
    assert any(e.lane == "engine" for e in result.elements)
    assert not any(f.rule in {"visual_overlap", "zoom_count", "zoom_style", "zoom_subtle"}
                   for f in result.findings)


@pytest.mark.parametrize("height,margin,scale", [
    (180, 24, 1.15), (200, 24, 1), (150, 24, 1.3), (180, 40, 1),
])
def test_step_preserves_front_card_box_clearance(step_project, monkeypatch, height, margin, scale):
    style = resolve_style_data({"schema_version": 1, "moves": {"jump_cuts": "step_in"},
                                "graphics": {"face_clearance_px": margin}})
    monkeypatch.setattr(stage, "_style", lambda context: style)
    box = dict(x=0, y=0, width=1080, height=height)
    result = add(step_project, graphic(start={"time": 6}, end={"time": 8}, layer="front",
                                       box=box), lane="agent").require_valid()
    assert [shot["scale"] for shot in result.steps["shots"][:3]] == [1, 1.15, 1.3]
    assert result.steps["shots"][3]["scale"] == scale
    assert next(e for e in result.elements if e.id == "agent-card").payload["box"] == box


def test_step_punch_covers_shot_and_keeps_own_overlap_rule(step_project):
    add(step_project, dict(id="agent-title", kind="title", lane="agent", lines=["alpha"],
                     start={"time": 2}, end={"time": 3}), lane="agent")
    result = add(step_project, zoom(start={"time": 2}, end={"time": 3}), lane="agent")
    assert result.steps["shots"][1]["scale"] == 1
    assert result.steps["shots"][2]["scale"] == 1.15
    assert any(f.rule == "visual_overlap" for f in result.findings)
    assert any(f.rule == "zoom_style" for f in result.findings)


def test_step_opening_push_covers_early_shots(step_project, monkeypatch):
    from test_step_in import pieces

    opening_story(step_project, [("alpha", 0, .3), ("um", .4, 1), ("beta.", 1.1, 3)],
                  (20, 60, 120, 180, 240), source_gap=4 / 30)
    record = json.loads((step_project / "cut/receipt.json").read_bytes())
    record["plan"]["pieces"] = pieces((20, 40, 60, 60, 60, 120))
    write(step_project / "cut/receipt.json", record)
    style = resolve_style_data({"schema_version": 1, "moves": {
        "jump_cuts": "step_in", "opening_push": True,
    }})
    monkeypatch.setattr(stage, "_style", lambda context: style)
    result = add(step_project, opening(), lane="agent").require_valid()
    push = next(item for item in result.elements if item.id == "agent-opening")
    assert (push.start_frame, push.end_frame) == (0, 60)
    runs = [item for item in result.elements if item.id.startswith("steps-")]
    assert runs
    assert all(max(item.start_frame, push.start_frame) >= min(item.end_frame, push.end_frame)
               for item in runs)
    assert [shot["level"] for shot in result.steps["shots"][:2]] == [0, 0]
    assert result.steps["shots"][1]["reason"] == (
        "A zoom covers this shot (agent-opening).")
    assert result.findings == ()


def test_step_opening_push_cuts_back_wide(step_project, monkeypatch):
    from test_step_in import pieces

    opening_story(step_project, [("alpha", 0, .3), ("um.", .4, 1.4), ("beta", 1.5, 3)],
                  (45, 90, 150, 210, 270), source_gap=4 / 30)
    record = json.loads((step_project / "cut/receipt.json").read_bytes())
    record["plan"]["pieces"] = pieces((45, 45, 60, 60, 60, 90))
    write(step_project / "cut/receipt.json", record)
    style = resolve_style_data({"schema_version": 1, "moves": {
        "jump_cuts": "step_in", "opening_push": True,
    }})
    monkeypatch.setattr(stage, "_style", lambda context: style)
    result = add(step_project, opening(), lane="agent").require_valid()
    push = next(item for item in result.elements if item.id == "agent-opening")
    assert push.end_frame == 45
    runs = [item for item in result.elements if item.id.startswith("steps-")]
    assert runs and min(item.start_frame for item in runs) == 90
    assert all(item.start_frame != push.end_frame for item in runs)
    assert result.steps["shots"][1]["scale"] == 1
    jump = next(item for item in result.steps["jumps"] if item["frame"] == push.end_frame)
    assert jump["handling"] == "reset"
    assert jump["reason"] == "The opening push cuts back wide here."
    assert result.findings == ()


@pytest.mark.parametrize("changes", [{"layer": "behind"}, {"template": "big_number",
                                                                "slots": {"number": "2"}}])
def test_step_behind_head_card_covers_shot(step_project, changes):
    result = add(step_project, graphic(start={"time": 2}, end={"time": 3}, **changes),
                 lane="agent")
    assert result.steps["shots"][1]["level"] == 0


def test_step_missing_piece_receipt(step_project):
    record = json.loads((step_project / "cut/receipt.json").read_bytes())
    record["plan"].pop("pieces")
    write(step_project / "cut/receipt.json", record)
    result = resolve_timeline(step_project)
    finding = next(f for f in result.findings if f.rule == "step_in_cut")
    assert finding.severity == "violation"
    assert finding.message == "The cut receipt has no pieces for Step in."
    assert finding.suggested_action == "Run yap-studio cut DIR again."


@pytest.mark.parametrize("mode", ["hard", "punch", "smooth"])
def test_other_jump_modes_keep_timeline(project, monkeypatch, mode):
    original = resolve_timeline(project).to_dict()
    style = resolve_style_data({"schema_version": 1, "moves": {"jump_cuts": mode}})
    monkeypatch.setattr(stage, "_style", lambda context: style)
    assert resolve_timeline(project).to_dict() == original
    assert "steps" not in original


@pytest.mark.parametrize("command", ["check", "show"])
def test_step_cli_summary(step_project, capsys, command):
    assert cli.main(["timeline", command, str(step_project)]) == 0
    assert "Step in: 5 of 5 jump cuts change the framing." in capsys.readouterr().out
