"""Final assembly through real media, receipts, CLI and declared-hold verification."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
from test_assemble_picture import heic_quadrants, overlay_image, textured_image
from test_cli_captions import fake_layers as fake_layers
from test_cut import make_project, time_segment, write_edit, write_json
from test_matte_project import synthetic_matte
from test_music_elevenlabs import fake as elevenlabs_fake  # noqa: F401
from test_music_file import set_music
from test_stage_captions import set_elements
from test_stage_graphics import graphic, title

from yap_studio import cli, elevenlabs, ffmpeg, receipt, sfx
from yap_studio.checks.caption_layout import check_caption_cards, read_layout_record
from yap_studio.checks.face_clearance import check_face_clearance
from yap_studio.checks.model import TransitionWindow
from yap_studio.edl import load_edit_list
from yap_studio.errors import OutputPathError, ProcessingError
from yap_studio.faces import detect
from yap_studio.faces.clearance import DEFAULT_HAIR, Layout, caption_gap_px, face_clearance_px
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity, load_faces
from yap_studio.matte import project as matte_cache
from yap_studio.project.receipts import StageRun, projection_sha256, read_receipt
from yap_studio.project.status import ProjectContext, cut_inputs, human_items, ingest_inputs, status
from yap_studio.render import hyperframes
from yap_studio.stages import assemble as stage
from yap_studio.stages import assemble_picture
from yap_studio.stages import graphics as graphics_stage
from yap_studio.stages import transitions as transition_stage
from yap_studio.stages.assemble_audio import decode, measure, render_audio
from yap_studio.stages.captions import captions
from yap_studio.stages.cut import cut
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.stages.timeline import (
    ResolvedElement,
    TimelineFinding,
    TimelineRefusal,
    add_timeline,
)
from yap_studio.style import resolve


@pytest.fixture(autouse=True)
def refuse_unexpected_matte_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary assembly must never invoke segmentation."""
    monkeypatch.setattr(matte_cache, "make_matte",
                        lambda *args, joins=(): pytest.fail("Unexpected person matte work"))
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "Vision unavailable in this test")

    def unexpected_detector() -> None:
        pytest.fail("Synthetic assembly must never invoke the system face detector")

    monkeypatch.setattr(detect, "default_detector", unexpected_detector)


@pytest.fixture
def project(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Publish a synthetic master with one retained range and current stage receipts."""
    settings = getattr(request, "param", 8)
    cut_frames = settings.get("cut_frames", 120) if isinstance(settings, dict) else 120
    hold_frames = settings.get("hold", 8) if isinstance(settings, dict) else settings
    picture_filter = settings.get("picture_filter") if isinstance(settings, dict) else None
    source = synth.loudness_master(tmp_path / "master.mov",
                                   frames=cut_frames + hold_frames if cut_frames != 120 else 120,
                                   picture_filter=picture_filter)
    selected = settings.get("audio_stream", 0) if isinstance(settings, dict) else 0
    if selected:
        quiet = synth.float_audio(tmp_path / "quiet.wav", np.zeros(192000, dtype=np.float32))
        multiple = tmp_path / "multiple.mov"
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-i", str(quiet),
                    "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0", "-c", "copy",
                    str(multiple)], timeout_s=60)
        source = multiple
    root = make_project(tmp_path / "project", [source],
                        word_rows=[[("First", 0.1, 0.5), ("words", 0.6, 1.0)]])
    data = json.loads((root / "project.json").read_bytes())
    fixture_style = resolve(data["style"]).data
    fixture_style["broll"]["color_match"] = "off"
    fixture_style["music"]["sound_effects_fallback"] = "none"
    write_json(root / "assembly-style.json", fixture_style)
    data["style"] = "assembly-style.json"
    data["asr"] = {"engine": "none"}
    data["music"] = {"mode": "none"}
    data["audio"] = {"mode": "preserve"}
    if isinstance(settings, dict):
        data["platform"] = settings.get("platform", "none")
    if selected:
        data["takes"][0]["audio_stream"] = selected
    write_json(root / "project.json", data)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish()
    duration = cut_frames / 30
    write_edit(root, [time_segment("s1", "t1", 0, duration)], end_hold_s=hold_frames / 30)
    picture = root / "picture.mp4"
    picture_args = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
                    "-map", "0:v:0", "-vf", "scale=1080:1920,format=yuv420p", "-an",
                    "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    if cut_frames == 120:
        ffmpeg.run([*picture_args, str(picture)], timeout_s=120)
    else:
        still = root / "still.mp4"
        ffmpeg.run([*picture_args, "-frames:v", "1", str(still)], timeout_s=120)
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-stream_loop",
                    str(cut_frames - 1), "-i", str(still), "-c", "copy", "-frames:v",
                    str(cut_frames), str(picture)], timeout_s=120)
    voice = root / "voice.wav"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map",
                f"0:a:{selected}", "-af", f"atrim=end_sample={cut_frames * 1600}",
                "-c:a", "pcm_f32le", str(voice)], timeout_s=120)
    spans = root / "spans.json"
    source_entry = {"path": "../takes/t1.mov", "audio_stream": selected} \
        if selected else "../takes/t1.mov"
    write_json(spans, {"sources": {"t1": source_entry}, "fps": "30/1",
                       "ranges": [{"source": "t1", "start": "0", "end": str(duration)}]})
    words = root / "words.json"
    hashes = {"t1": receipt.sha256_file(root / "ingest/t1/words.json")}
    write_json(words, {"schema": "yap-studio/cut-words@1", "time_base": "output",
                       "words_sha256": hashes,
                       "words": [{"id": "t1:0", "text": "First", "segment": "s1",
                                  "start": 0.1, "end": 0.5},
                                 {"id": "t1:1", "text": "words", "segment": "s1",
                                  "start": 0.6, "end": 1.0}]})
    inputs = cut_inputs(ProjectContext(root))
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        for name, staged in (("cut.mp4", picture), ("voice.wav", voice),
                             ("spans.json", spans), ("cut-words.json", words)):
            run.replace_output(root / "cut" / name, staged)
        run.finish(extra={"plan": {"total_frames": cut_frames}})
    captions(root)
    graphics(root)
    music(root)
    return root


@pytest.mark.parametrize("project", [{"audio_stream": 1}], indirect=True)
def test_selected_audio_has_owned_portable_verification_reference(project: Path) -> None:
    """Ranges verification must hear the dialogue chosen by the cut stage."""
    stage.assemble(project)
    record = read_receipt(project / "final")
    edl = load_edit_list(project / "final/verify-edl.json")
    reference = edl.sources["t1"].path
    assert reference.is_file() and reference.parent.name == "source-audio"
    assert measure(reference, []).integrated_lufs == pytest.approx(-16, abs=0.1)
    assert record["status"] == "verified"
    assert any(item["path"] == "final/source-audio/source-0.mov" for item in record["outputs"])
    assert final_stage(project)["state"] == "done"


@pytest.mark.parametrize("prior_sidecar,interrupt", [(False, False), (True, False),
                                                    (True, "before"), (True, "after")])
def test_remove_chapter_sidecar_and_recover(project, monkeypatch, prior_sidecar, interrupt):
    original_verify = stage.verify

    def with_chapters(*args, **kwargs):
        return {**original_verify(*args, **kwargs),
                "youtube_chapters": "0:00 Start\n0:10 Middle\n0:20 End\n"}

    with monkeypatch.context() as patch:
        if prior_sidecar:
            patch.setattr(stage, "verify", with_chapters)
        stage.assemble(project)
    sidecar = project / "final/story.chapters.txt"
    assert sidecar.exists() == prior_sidecar
    if interrupt:
        original_journal = StageRun._journal_output
        original_replace = StageRun.replace_output

        def stop_before_delete(run, entry):
            original_journal(run, entry)
            if entry.get("new_sha256") is None:
                raise KeyboardInterrupt

        def stop_after_delete(run, target, staged):
            if target == (project / "final/story.mp4").with_suffix(".layout.json"):
                raise KeyboardInterrupt
            original_replace(run, target, staged)

        with monkeypatch.context() as patch:
            patch.setattr(stage, "_mark_stale", lambda *args: None)
            if interrupt == "before":
                patch.setattr(StageRun, "_journal_output", stop_before_delete)
            else:
                patch.setattr(StageRun, "replace_output", stop_after_delete)
            with pytest.raises(KeyboardInterrupt):
                stage.assemble(project, force=True)
        assert sidecar.exists() == (interrupt == "before")
        assert (project / "final/receipt.pending.json").exists()
    result = stage.assemble(project, force=True)
    record = read_receipt(project / "final")
    assert result["status"] == record["status"] == "verified"
    assert not sidecar.exists()
    assert not any(item["path"].endswith(".chapters.txt") for item in record["outputs"])
    assert not (project / "final/receipt.pending.json").exists()
    assert stage.assemble(project)["noop"]


def test_fully_gapped_captions_assemble_without_a_layer(project: Path, capsys) -> None:
    from yap_studio.review.findings import findings_for_media

    set_elements(project, {"id": "agent-gap", "kind": "caption_gap",
                           "start": {"time": 0}, "hold_s": 4.0})
    result = captions(project)
    assert result["layer"] is None
    assert not (project / "captions/captions.mov").exists()
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert snapshot.caption_layer is None
    assert "captions/captions.mov" not in snapshot.media
    assert not stage.verification_edl(snapshot, base=project / "final")["overlays"]
    assembled = stage.assemble(project)
    record = read_receipt(project / "final")
    assert record["status"] == "verified" and record["verify"]["status"] != "fail"
    assert record["caption_contrast"]["measured"]
    assert record["caption_contrast"]["captions"] == 0
    assert "captions.mov" not in json.dumps(record["picture"])
    assert ffmpeg.probe(project / assembled["out"]).video_frames == 128
    assert final_stage(project)["state"] == "done"
    assert stage.assemble(project)["noop"]
    final = project / assembled["out"]
    review = findings_for_media(final, receipt.sha256_file(final), project_dir=project)
    assert not review.get("layer_stale")
    assert cli.main(["captions", str(project), "--force"]) == 0
    assert "No caption phrases; skipped caption layer." in capsys.readouterr().out
    set_elements(project)
    with pytest.raises(stage.StaleStageError):
        stage.assembly_snapshot(ProjectContext(project))
    assert captions(project)["layer"] is not None
    assert (project / "captions/captions.mov").is_file()


def test_editorial_override_cannot_bypass_keyed_clone_flash(project: Path) -> None:
    style_file = project / "assembly-style.json"
    style = json.loads(style_file.read_text())
    style["moments"]["styles"] = ["clone_flash"]
    style["broll"]["cutout"] = "screen"
    style_file.write_text(json.dumps(style))
    set_elements(project, {"id": "agent-moment", "kind": "moment", "lane": "agent",
                           "style": "clone_flash", "start": {"time": 1}, "hold_s": 1.2})
    context = ProjectContext(project)
    findings = context.resolved_timeline(allow_violations=True).findings
    keyed = [item for item in findings if item.rule == "clone_keyed"]
    assert len(keyed) == 1 and keyed[0].severity == "violation"
    message = "clone_flash cannot clean the green or blue screen edge"
    with pytest.raises(TimelineRefusal, match=message):
        stage.assembly_snapshot(context, allow_violations=True)


def test_grade_snapshot_receipt_and_portable_reference(project: Path) -> None:
    from yap_studio.grade import GRADE_VERSION

    plain = stage.assembly_snapshot(ProjectContext(project))
    assert "grade" not in plain.inputs
    assert "grade" not in stage.verification_edl(plain, base=project / "final")
    assert cli.main(["grade", "set", str(project), "--look", "film"]) == 0
    graded = stage.assembly_snapshot(ProjectContext(project))
    assert graded.inputs["grade"] == projection_sha256({
        "look": "film", "strength": 1, "source": "project", "version": GRADE_VERSION})
    assert {key: value for key, value in graded.inputs.items() if key != "grade"} == plain.inputs
    transition = ResolvedElement("agent-transition", "transition", "agent", 30, 42, "front",
                                 {"look": "whip_pan", "frames": 12, "join_frame": 36})
    changed = replace(graded, timeline=replace(graded.timeline, elements=(transition,)))
    transition_edl = stage.verification_edl(changed, base=project / "final")
    assert transition_edl["grade"]["skip_frames"] == [[30, 42]]
    assert transition_edl["overlays"] == stage.verification_edl(
        plain, base=project / "final")["overlays"]
    nested = stage.verification_edl(graded, base=project / "review/deep")
    assert nested["grade"]["reference"] == "../../cut/cut.mp4"
    result = stage.assemble(project)
    final = project / result["out"]
    record = read_receipt(project / "final")
    assert record["status"] == "verified"
    assert record["grade"] == record["picture"]["grade"]
    assert record["grade"]["version"] == GRADE_VERSION
    assert record["verify"]["checks"]["picture.grade"]["ran"]
    portable = json.loads((project / "final/verify-edl.json").read_bytes())
    assert portable["grade"]["reference"] == "../cut/cut.mp4"
    assert portable["grade"]["reference_sha256"] == receipt.sha256_file(project / "cut/cut.mp4")
    report = stage.verify(str(final), edl=str(project / "final/verify-edl.json"), asr="none",
                          loudness="note")
    assert report["checks"]["picture.grade"]["status"] == "pass"
    assert cli.main(["grade", "set", str(project), "--clear"]) == 0
    assert stage.assembly_snapshot(ProjectContext(project)).inputs == plain.inputs


def test_camera_edl_region_and_grade_reference(project: Path) -> None:
    from yap_studio.grade import GradeSpec

    snapshot = stage.assembly_snapshot(ProjectContext(project))
    camera = project / "captions/camera3d.mp4"
    shutil.copyfile(project / "cut/cut.mp4", camera)
    changed = replace(snapshot, picture="captions/camera3d.mp4",
                       grade=GradeSpec("film", 1, "style"),
                       media={**snapshot.media, "captions/camera3d.mp4": camera},
                       inputs={**snapshot.inputs,
                               "media:captions/camera3d.mp4": receipt.sha256_file(camera)})
    edl = stage.verification_edl(changed, base=project / "final")
    assert edl["grade"]["reference"] == "../captions/camera3d.mp4"
    assert edl["grade"]["reference_sha256"] == receipt.sha256_file(camera)
    region, = [item for item in edl["overlays"] if item.get("camera_3d")]
    assert region == {"camera_3d": True, "region": [0, 0, 1080, 1920],
                      "start_in_output": 0, "duration": snapshot.timeline.total_frames / 30}


def test_camera_edl_masks_caption_boxes_only_in_their_windows(project: Path) -> None:
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    layout = {"captions": [{"start_frame": 0, "end_frame": 60, "boxes": [
        {"x": 100, "y": 500, "w": 200, "h": 60, "start_frame": 0, "end_frame": 30},
        {"x": 100, "y": 1400, "w": 200, "h": 60, "start_frame": 30, "end_frame": 60},
    ]}]}
    changed = replace(snapshot, picture="captions/camera3d.mp4",
                       small_files={**snapshot.small_files,
                                    "captions/layout.json": json.dumps(layout).encode()})
    masks = stage.verification_edl(changed, base=project / "final")["overlays"]
    assert {tuple(m["region"]): (m["start_in_output"], m["duration"]) for m in masks
            if m["region"] in [[100, 500, 200, 60], [100, 1400, 200, 60]]} == {
                (100, 500, 200, 60): (0, 1), (100, 1400, 200, 60): (1, 1)}


@pytest.mark.render
def test_eyedropper_sample_snapshot_and_receipt(project: Path) -> None:
    from yap_studio.grade import GRADE_VERSION, sample_color

    sample = {"time_s": 0.5, "x": 0.2, "y": 0.2}
    color = sample_color(project / "cut/cut.mp4", **sample)
    data = json.loads((project / "project.json").read_bytes())
    data["grade"] = {"look": "eyedropper", "sample": sample}
    write_json(project / "project.json", data)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert snapshot.grade.color == color
    assert snapshot.inputs["grade"] == projection_sha256({
        "look": "eyedropper", "strength": 0.5, "source": "project", "color": color,
        "sample": sample, "version": GRADE_VERSION})
    result = stage.assemble(project)
    record = read_receipt(project / "final")
    assert record["grade"]["color"] == color and record["grade"]["sample"] == sample
    edl = json.loads((project / "final/verify-edl.json").read_bytes())
    assert edl["grade"]["color"] == color and "sample" not in edl["grade"]
    report = stage.verify(str(project / result["out"]),
                          edl=str(project / "final/verify-edl.json"), asr="none", loudness="note")
    assert report["status"] == "pass", report["findings"]
    assert report["checks"]["picture.grade"]["status"] == "pass"
    evidence = report["checks"]["picture.grade"]["evidence"]
    assert evidence["median_error"] < evidence["limit"] / 2
    assert evidence["limit"] == pytest.approx(evidence["grade_magnitude"] / 2)
    print("Sampled project correct:", evidence)
    from yap_studio.checks.grade import check_grade
    from yap_studio.plan import Plan

    edits = load_edit_list(project / "final/verify-edl.json")
    reference = project / "cut/cut.mp4"
    info = ffmpeg.probe(reference)
    plan = Plan.build(edits, fps=info.fps, sample_rate=48000)
    findings, coverage = check_grade(reference, edits, plan, info)
    assert any(row.check == "picture.grade" for row in findings), coverage
    assert coverage["evidence"]["looks_ungraded"]
    print("Sampled project ungraded:", coverage["evidence"])


def test_eyedropper_outside_cut_refused_before_render(project: Path, monkeypatch) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["grade"] = {"look": "eyedropper", "sample": {"time_s": 999, "x": 0.5, "y": 0.5}}
    write_json(project / "project.json", data)
    monkeypatch.setattr(stage, "render_picture",
                        lambda *args, **kwargs: pytest.fail("render started"))
    from yap_studio.grade import GradeError

    with pytest.raises(GradeError) as error:
        stage.assemble(project)
    assert error.value.suggested_action


@pytest.mark.render
def test_grade_compare_receipt_and_verification(project: Path) -> None:
    from yap_studio.grade import describe_compare

    assert cli.main(["grade", "set", str(project), "--look", "film"]) == 0
    fragment = project / "compare.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "agent-compare", "kind": "grade_compare", "lane": "agent",
        "start": {"time": 0.5}, "hold_s": 2.5}]})
    assert add_timeline(project, fragment, lane="agent").valid
    captions(project)
    graphics(project)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert snapshot.inputs["grade:compare"] == projection_sha256(
        describe_compare(snapshot.grade_style))
    result = stage.assemble(project)
    record = read_receipt(project / "final")
    assert record["picture"]["grade_compare"] == [
        {"id": "agent-compare", "start_frame": 15, "end_frame": 90}]
    edl = json.loads((project / "final/verify-edl.json").read_text())
    assert {"start_in_output": 0.5, "duration": 2.5, "region": [0, 0, 1080, 1920]} \
        in edl["overlays"]
    report = stage.verify(str(project / result["out"]), edl=str(project / "final/verify-edl.json"),
                          asr="none", loudness="note")
    assert report["status"] == "pass", report["findings"]
    assert report["checks"]["picture.grade"]["status"] == "pass"
    assert all(not 15 <= frame < 90 for frame in
               report["checks"]["picture.grade"]["evidence"]["frames"])


@pytest.mark.parametrize("rule", ["grade_compare_grade", "grade_compare_overlap",
                                 "grade_compare_length"])
def test_grade_compare_unsafe_rules_cannot_be_overridden(project, monkeypatch, rule):
    context = ProjectContext(project)
    timeline = context.resolved_timeline()
    finding = TimelineFinding(rule, "violation", "Unsafe grade compare", (), (),
                              "Move or remove the compare window.")
    monkeypatch.setattr(context, "resolved_timeline", lambda **kw: replace(
        timeline, findings=(finding,)))
    with pytest.raises(TimelineRefusal, match="Unsafe grade compare"):
        stage.assembly_snapshot(context, allow_violations=True)


def test_one_frame_grade_compare_is_never_overridden(project, monkeypatch):
    data = json.loads((project / "project.json").read_bytes())
    data["grade"] = {"look": "film"}
    write_json(project / "project.json", data)
    write_json(project / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [{
            "id": "agent-too-short", "kind": "grade_compare", "lane": "agent",
            "start": {"time": 1}, "hold_s": 1 / 30}]})
    context = ProjectContext(project)
    finding = next(row for row in context.resolved_timeline(allow_violations=True).findings
                   if row.rule == "grade_compare_length")
    monkeypatch.setattr(stage, "render_picture", lambda *a, **kw: pytest.fail("Render started"))
    with pytest.raises(TimelineRefusal,
                       match="grade compare length must be 2 to 8 seconds") as error:
        stage.assemble(project, allow_violations=True)
    assert str(error.value) == finding.message


def test_grade_reaches_pilot(project: Path) -> None:
    assert cli.main(["grade", "set", str(project), "--look", "film"]) == 0
    result = stage.assemble(project, frames=(0, 89))
    assert result["pilot"] is True
    record = read_receipt(project / "pilot")
    assert record["picture"]["grade"]["look"] == "film"
    assert "grade" in record["inputs"]


@pytest.mark.parametrize("frames", [None, (0, 89)])
def test_card_clip_assemble_uses_project_local_style(
    project: Path, frames: tuple[int, int] | None,
) -> None:
    style = ProjectContext(project).render_style().data
    style["broll"]["color_match"] = "on"
    style["graphics"]["card_kinds_on"] = ["video_card"]
    write_json(project / "styles/owner-review.json", style)
    document = json.loads((project / "project.json").read_bytes())
    document["style"] = "styles/owner-review.json"
    write_json(project / "project.json", document)
    source = synth.picture_source(project / "clip.mov", "color=c=red:s=320x180:r=30", 120)
    clip = {"path": source.name, "sha256": receipt.sha256_file(source)}
    set_elements(project, graphic(template="video_card", slots={"clip": clip}, hold_s=1.5))
    captions(project)
    graphics(project)
    result = stage.assemble(project, frames=frames)
    record = read_receipt(project / ("pilot" if frames else "final"))
    entry, = record["picture"]["broll_match"]
    assert entry["id"] == "agent-tag window 0" and entry["kind"] == "card_window"
    assert "take" in entry and "broll" in entry
    assert ffmpeg.probe(project / result["out"]).video_frames == (90 if frames else 128)


@pytest.mark.parametrize("frames", [None, (0, 89)])
def test_assemble_refuses_behind_grid_over_smooth_join(
    project: Path, monkeypatch: pytest.MonkeyPatch, frames: tuple[int, int] | None,
) -> None:
    from yap_studio.joins import plan as joins_plan
    from yap_studio.joins import render as joins_render

    style_file = project / "assembly-style.json"
    style = json.loads(style_file.read_bytes())
    style["moves"]["jump_cuts"] = "smooth"
    write_json(style_file, style)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    grid = ResolvedElement("agent-grid", "graphic", "agent", 0, 90, "behind",
                           {"template": "playing_grid", "slots": {}})
    small = dict(snapshot.small_files)
    record = json.loads(small["cut/receipt.json"])
    record["plan"]["pieces"] = []
    small["cut/receipt.json"] = json.dumps(record).encode()
    snapshot = replace(snapshot, small_files=small,
                       timeline=replace(snapshot.timeline, elements=(grid,)), title_layers=[{
                           "id": grid.id, "kind": "graphic", "layer": "behind",
                           "path": "cut/cut.mp4", "start_frame": 0, "frames": 90,
                           "x": 0, "y": 0, "width": 1080, "height": 1920,
                           "clearance": {"status": "behind_head", "gap_px": 1,
                                         "tiles": [{"gap_px": 1}]},
                       }])
    monkeypatch.setattr(stage, "assembly_snapshot", lambda *args, **kwargs: snapshot)
    decisions = joins_plan.JoinPlan("smooth", "punch", [
        joins_plan.JoinDecision(0, "removal", 78, 2.6, 75, 6, None, None, "smooth", ""),
    ])
    monkeypatch.setattr(joins_render, "build_plan", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(joins_render, "render_windows", lambda *args, **kwargs: [])
    monkeypatch.setattr(stage, "render_picture", lambda *args, **kwargs:
                        pytest.fail("Unsafe behind grid reached picture rendering"))
    with pytest.raises(TimelineRefusal, match="move the grid.*layer front") as caught:
        stage.assemble(project, frames=frames)
    assert caught.value.code == "card_grid_behind_join"
    assert "too close to the speaker" in str(caught.value)


def test_card_clip_snapshot_retains_source_and_refuses_changed_bytes(
    project: Path, fake_layers: list[dict],  # noqa: F811
) -> None:
    style = ProjectContext(project).render_style().data
    style["graphics"]["card_kinds_on"] = ["video_card"]
    write_json(project / "style.json", style)
    document = json.loads((project / "project.json").read_bytes())
    document["style"] = "style.json"
    write_json(project / "project.json", document)
    source = synth.picture_source(project / "clip.mov", "color=c=red:s=320x180:r=30", 120)
    clip = {"path": source.name, "sha256": receipt.sha256_file(source)}
    set_elements(project, graphic(template="video_card", slots={"clip": clip}, hold_s=1.5))
    captions(project)
    graphics(project)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    key = "card-clips/agent-tag/clip-0.mov"
    assert snapshot.media[key] == source
    assert snapshot.inputs[f"media:{key}"] == clip["sha256"]
    window, = snapshot.title_layers[0]["windows"]
    assert window["source"]["path"] == key and window["source"]["source_name"] == source.name
    assert not stage._audio_clips(snapshot)
    retained = project / "retained-card"
    stage._retained(snapshot, retained)
    assert receipt.sha256_file(retained / key) == clip["sha256"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(stage.AssemblyError, match="changed while"):
        stage._retained(snapshot, project / "changed-card")
    with pytest.raises(TimelineRefusal, match="clip changed"):
        stage.assembly_snapshot(ProjectContext(project), allow_violations=True)


@pytest.mark.parametrize("rule", ["card_video_media", "card_video_speed", "card_video_bounds",
                                  "card_video_audio"])
def test_card_clip_unsafe_findings_cannot_be_overridden(project: Path, rule: str) -> None:
    context = ProjectContext(project)
    context._timeline = replace(context.resolved_timeline(), findings=(
        TimelineFinding(rule, "violation", "broken card video", ("agent-card",), ("agent",),
                        "fix the card", "broken card video"),
    ))
    with pytest.raises(TimelineRefusal, match="broken card video"):
        stage.assembly_snapshot(context, allow_violations=True)
def final_stage(root: Path) -> dict[str, Any]:
    """Read the assembly stage's public status independently of ordering."""
    return next(item for item in status(root)["stages"] if item["name"] == "assemble")


@pytest.mark.parametrize(("raw", "fixes", "expected", "count", "zero_length", "lettered"), [
    ("Fuck", {}, "F***", 1, False, False), ("Fuck", {}, "F***", 1, True, False),
    ("ship", {"ship": "shit"}, "s***", 1, False, False),
    ("shit", {"shit": "ship"}, "ship", 0, False, False),
    ("shit-show", {}, "s***-show", 1, False, False),
    ("Fuck", {}, "F***", 1, False, True),
])
def test_bleep_reaches_assembly_receipt_and_masks_phrases(
    project: Path, raw, fixes, expected, count, zero_length: bool, lettered: bool,
    fake_layers: list[dict],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style_path = project / "assembly-style.json"
    style = json.loads(style_path.read_bytes())
    style["music"].update(sound_effects="none", bleep_swears=True)
    if lettered:
        style["hook"]["style"] = "video_letters"
    write_json(style_path, style)
    document = json.loads((project / "cut/cut-words.json").read_bytes())
    document["words"][0]["text"] = raw
    project_data = json.loads((project / "project.json").read_bytes())
    project_data["word_fixes"] = fixes
    write_json(project / "project.json", project_data)
    if zero_length:
        document["words"][0]["end"] = document["words"][0]["start"]
    staged = project / "staged-bleep-words.json"
    write_json(staged, document)
    previous = read_receipt(project / "cut")
    with StageRun(project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    if lettered:
        import test_cli_captions

        validate = test_cli_captions._validate_variables
        monkeypatch.setattr(test_cli_captions, "_validate_variables", lambda source, variables:
                            None if variables.get("look") == "video_letters"
                            else validate(source, variables))
        source = project / "takes/t1.mov"
        fragment = project / "letters-fragment.json"
        write_json(fragment, {"schema": "yap-studio/timeline@1",
            "words_sha256": document["words_sha256"], "elements": [{
                "id": "broll-letters", "kind": "insert", "start": {"time": 0}, "hold_s": 1.6,
                "media": {"path": "takes/t1.mov", "sha256": receipt.sha256_file(source)},
                "letters": {"text": raw},
            }]})
        add_timeline(project, fragment, lane="broll").require_valid()
    stage.assemble(project)
    record = read_receipt(project / "final")
    if count:
        assert record["audio"]["bleep"]["count"] == count
    else:
        assert "bleep" not in record["audio"]
    check, = [item for item in record["asserts"] if item["id"] == "bleep.words"]
    assert check["pass"] and check["words"] == count
    assert not check["audible"] and not check["unmasked"]
    if lettered:
        assert "letters:broll-letters" in record["inputs"]
        plates = [call["variables"] for call in fake_layers
                  if call["variables"].get("look") == "video_letters"]
        assert plates and all(json.loads(plate["lines"]) == [expected] for plate in plates)
    assert "style:bleep" in record["inputs"] and "cut/cut-words.json" in record["inputs"]
    legacy_bleep = {**stage.bleep.projection(style["music"]), "tone_hz": stage.bleep.TONE_HZ,
                   "pad_s": stage.bleep.PAD_S}
    assert record["inputs"]["style:bleep"] != projection_sha256(legacy_bleep)
    phrases = json.loads((project / "captions/phrases.json").read_bytes())
    assert expected in phrases["phrases"][0]["text"]


def test_bleep_default_does_not_change_assembly_inputs(project: Path) -> None:
    before = stage.assembly_snapshot(ProjectContext(project)).inputs
    style_path = project / "assembly-style.json"
    style = json.loads(style_path.read_bytes())
    for key in ("bleep_swears", "bleep_words_add", "bleep_words_remove"):
        style["music"].pop(key, None)
    write_json(style_path, style)
    assert stage.assembly_snapshot(ProjectContext(project)).inputs == before
    legacy = ProjectContext(project)
    for key in ("bleep_swears", "bleep_words_add", "bleep_words_remove"):
        legacy.render_style().data["music"].pop(key)
    assert stage.assembly_snapshot(legacy).inputs == before
    assert "style:bleep" not in before


@pytest.mark.render
@pytest.mark.parametrize("fallback", ["none", "made"])
def test_sfx_cache_hashes_reach_final_receipt_and_invalidate_it(
    project: Path, fallback: str,
) -> None:
    """Synthetic user-cache sounds bind the final without entering the project tree."""
    style_path = project / "assembly-style.json"
    style_data = json.loads(style_path.read_text())
    style_data["music"]["sound_effects_fallback"] = fallback
    write_json(style_path, style_data)
    rate = 48_000
    time = np.arange(rate) / rate
    tone = synth.float_audio(project.parent / "fixture-tone.wav",
                             np.column_stack([0.1 * np.sin(2 * np.pi * 440 * time)] * 2))
    index = {}
    paths = []
    for sound in sfx.load_kit()["sounds"]:
        digest = elevenlabs.sound_request_sha256(sfx.request(sound))
        path = sfx.sfx_cache_root() / "audio" / f"{digest}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tone, path)
        paths.append(path)
        index[f"soft/{sound['name']}"] = sfx.recipe_sha256(sound)
    (sfx.sfx_cache_root() / "generated.json").write_text(json.dumps(index))
    before = stage.assembly_snapshot(ProjectContext(project))
    assert before.sfx["ready"] and before.sfx["kit"] == "soft"
    assert "fallback_from" not in before.sfx
    assert len([key for key in before.inputs if key.startswith("sfx:audio:")]) == 4
    stage.assemble(project)
    record = read_receipt(project / "final")
    details = record["audio"]["sfx"]
    assert details["kit"] == "soft"
    assert set(details["recipes"]) == {"whoosh", "click", "tick-roll", "chime"}
    assert all(cue["relative_db"] < 0 for cue in details["cues"])
    paths[0].write_bytes(paths[0].read_bytes() + b"changed")
    after = stage.assembly_snapshot(ProjectContext(project))
    assert after.inputs != before.inputs


@pytest.mark.usefixtures("elevenlabs_fake")
def test_cutout_request_is_an_input_only_when_overridden(project: Path) -> None:
    before = stage.assembly_snapshot(ProjectContext(project)).inputs
    assert "picture:cutout" not in before
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write_json(project / "project.json", data)
    for cutout in ("screen", "person", "auto"):
        write_json(project / "style.json", {"schema_version": 1, "broll": {"cutout": cutout},
                                           "music": {"sound_effects_fallback": "none"}})
        after = stage.assembly_snapshot(ProjectContext(project)).inputs
        if cutout == "auto":
            assert before == after
        else:
            assert after.pop("picture:cutout") == projection_sha256(cutout)
            assert before == after


@pytest.mark.usefixtures("elevenlabs_fake")
def test_generated_music_bed_enters_assembly_snapshot(project: Path) -> None:
    set_music(project, mode="elevenlabs")
    assert music(project)["mode"] == "elevenlabs"
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert snapshot.bed == "music/bed.wav"
    assert snapshot.media["music/bed.wav"] == project / "music/bed.wav"


def test_chapter_final_background_face_mapping_and_verification_mask(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    style = ProjectContext(project).render_style().data
    style["chapters"]["style"] = "title_card"
    write_json(project / "style.json", style)
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write_json(project / "project.json", data)
    set_elements(project, {"id": "agent-ch", "kind": "chapter", "title": "New section",
                           "start": {"time": 1}, "hold_s": 1.6})
    captions(project)
    graphics(project)
    music(project)
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    assert {"start_in_output": 1, "duration": 1.6, "region": [0, 240, 1080, 1320]} \
        in stage.verification_edl(snapshot, base=project / "final")["overlays"]
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    layout_file = retained / "captions/layout.json"
    document = json.loads(layout_file.read_bytes())
    document.update(measured=True, reason=None, captions=[{
        "id": "caption-chapter", "text": "New section", "start_frame": 40, "end_frame": 60,
        "boxes": [{"x": 200, "y": 800, "w": 600, "h": 100}],
        "words": [{"text": "New section", "start_s": 40 / 30, "end_s": 2}],
    }])
    write_json(layout_file, document)
    final = project / "record.mp4"
    final.write_bytes(b"synthetic final identity")
    face = FaceBox(Box(300, 500, 200, 200), .9, "detected")
    cut = FacesData("measured", None, VideoIdentity("cut.mp4", "0" * 64, 120,
                    Fraction(30), 1080, 1920), {}, {}, ((face,),) * 120, ())
    monkeypatch.setattr(stage, "cut_faces", lambda *args, **kwargs: cut)
    _, faces_path = stage._final_records(context, snapshot, retained, timeline, final)
    saved_layout = json.loads(final.with_suffix(".layout.json").read_bytes())
    assert saved_layout["caption_phrases"] == json.loads(
        (retained / "captions/phrases.json").read_bytes())
    record = read_layout_record(final, width=1080, height=1920)
    (element,) = record.layout.elements
    assert (element.kind, element.box, element.start_frame, element.end_frame) == (
        "background", Box(0, 0, 1080, 1920), 30, 78)
    assert record.labels == {
        "agent-ch": "chapter 01: text sits over the live picture by design",
    }
    faces = load_faces(faces_path)
    assert faces.frames[29] and faces.frames[78]
    assert all(frame for frame in faces.frames[30:78])
    findings, _ = check_face_clearance(record, faces, faces_source="record")
    assert not findings
    assert not check_caption_cards(record, fps=30)


def test_chapter_renders_above_insert_and_below_captions(project: Path) -> None:
    style = ProjectContext(project).render_style().data
    style["chapters"]["style"] = "title_card"
    style["broll"].update(photo_placement="full_frame", color_match="off")
    write_json(project / "style.json", style)
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write_json(project / "project.json", data)
    image = overlay_image(project / "insert.png", "color=red:s=1080x1920")
    fragment = project / "insert-fragment.json"
    write_json(fragment, {
        "schema": "yap-studio/timeline@1", "elements": [{
            "id": "broll-picture", "kind": "insert", "start": {"time": 0},
            "end": {"time": 4}, "media": {
                "path": image.name, "sha256": receipt.sha256_file(image),
            },
        }],
    })
    add_timeline(project, fragment, lane="broll").require_valid()
    set_elements(project, {"id": "agent-ch", "kind": "chapter", "title": "New section",
                           "start": {"time": 0}, "hold_s": 1})
    synth.picture_source(project / "chapter.mov",
        "color=blue@0.5:s=1080x1920:r=30,format=rgba", 30, alpha=True)
    synth.picture_source(project / "caption.mov",
        "color=lime:s=100x80:r=30,format=rgba", 30, alpha=True)
    out = project / "layer-order.mp4"
    assemble_picture.render_picture(project, out, work_dir=project / "layer-order-work",
        title_layers=[{"id": "agent-ch", "kind": "chapter", "layer": "front",
                       "path": "chapter.mov", "x": 0, "y": 0, "width": 1080, "height": 1920,
                       "start_frame": 0, "frames": 30}],
        caption_layer={"path": "caption.mov", "x": 20, "y": 20, "width": 100, "height": 80,
                       "start_frame": 0, "frames": 30}, frames=(15, 15))
    raw = ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(out),
                      "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                     timeout_s=120, binary=True).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(1920, 1080, 3)
    np.testing.assert_allclose(frame[400:450, 400:450].mean(axis=(0, 1)), (127, 0, 127), atol=10)
    np.testing.assert_allclose(frame[40:60, 40:60].mean(axis=(0, 1)), (0, 255, 0), atol=10)


def _add_overlay(project: Path, *, box: tuple[int, int, int, int] = (60, 400, 400, 400),
                 fit: str = "cover", motion: str = "none",
                 lavfi: str = "color=red:s=320x160") -> None:
    image = overlay_image(project / "photo.png", lavfi)
    fragment = project / "overlay-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-photo", "kind": "overlay", "start": {"word": "t1:0", "text": "First"},
        "hold_s": 1, "media": {"path": image.name, "sha256": receipt.sha256_file(image)},
        "placement": "box", "box": list(box), "fit": fit, "motion": motion,
    }]})
    add_timeline(project, fragment, lane="broll").require_valid()
    captions(project)
    graphics(project)
    music(project)


def test_overlay_retained_media_paths_hash_and_verification_region(project: Path) -> None:
    _add_overlay(project)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    key = "overlays/broll-photo"
    assert snapshot.media[key] == project / "photo.png"
    assert snapshot.inputs[f"media:{key}"] == receipt.sha256_file(project / "photo.png")
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    assert timeline.elements[0].payload["media"]["path"] == str(retained / key)
    assert timeline.elements[0].payload["media"]["source_name"] == "photo.png"
    assert "source_name" not in snapshot.timeline.elements[0].payload["media"]
    assert receipt.sha256_file(retained / key) == snapshot.inputs[f"media:{key}"]
    assert snapshot.timeline.elements[0].payload["media"]["path"] == "photo.png"
    edl = stage.verification_edl(snapshot, base=project / "final")
    assert {"start_in_output": 0.1, "duration": 1.0, "region": [60, 400, 400, 400]} \
        in edl["overlays"]
    (project / "photo.png").write_bytes(b"changed after snapshot")
    with pytest.raises(stage.AssemblyError, match="overlays/broll-photo changed"):
        stage._retained(snapshot, project / "changed")
    with pytest.raises(TimelineRefusal, match="overlay media hash changed"):
        stage.assembly_snapshot(ProjectContext(project), allow_violations=True)


@pytest.mark.parametrize("suffix", [".heic", ".heif", ".HEIC", ".HEIF"])
def test_retained_heic_insert_preserves_extension_and_crop_geometry(
    project: Path, suffix: str,
) -> None:
    image = heic_quadrants(project / f"photo{suffix}")
    fragment = project / "insert-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-photo", "kind": "insert", "start": {"time": 0},
        "end": {"time": 1},
        "media": {"path": image.name, "sha256": receipt.sha256_file(image),
                  "fit": "crop", "crop": [0, 0, 1280, 720]},
    }]})
    add_timeline(project, fragment, lane="broll").require_valid()
    captions(project)
    graphics(project)
    music(project)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    key = f"heif-inserts/broll-photo{suffix}"
    assert snapshot.media[key] == image
    assert snapshot.inputs[f"media:{key}"] == receipt.sha256_file(image)
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    assert timeline.elements[0].payload["media"]["path"] == str(retained / key)
    assert snapshot.timeline.elements[0].payload["media"]["path"] == image.name
    assert timeline.elements[0].payload["media"]["source_name"] == image.name
    assert "source_name" not in snapshot.timeline.elements[0].payload["media"]

    source = assemble_picture._insert_inputs(retained, timeline, [], [])["broll-photo"]
    assert source.path.suffix == image.suffix
    assert source.grid
    assert (source.info.width, source.info.height) == (1280, 720)
    graph = assemble_picture._Graph()
    graph.input(source, element=timeline.elements[0])
    assert graph.owners[0][0] == f"broll-photo ({image.name})"


def test_retained_heic_insert_key_does_not_collide_with_another_insert_id(
    project: Path,
) -> None:
    image = heic_quadrants(project / "photo.heic")
    other = overlay_image(project / "other.png")
    expected_hashes = {source: receipt.sha256_file(source) for source in (image, other)}
    fragment = project / "insert-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": identifier, "kind": "insert", "start": {"time": index},
        "end": {"time": index + 1},
        "media": {"path": source.name, "sha256": expected_hashes[source]},
    } for index, (identifier, source) in enumerate((
        ("broll-photo", image), ("broll-photo.heic", other),
    ))]})
    add_timeline(project, fragment, lane="broll").require_valid()
    captions(project)
    graphics(project)
    music(project)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    expected_keys = {"heif-inserts/broll-photo.heic": image,
                     "inserts/broll-photo.heic": other}
    inserts = {key: source for key, source in snapshot.media.items()
               if key.startswith(("heif-inserts/", "inserts/"))}
    assert inserts == expected_keys
    assert {key: snapshot.inputs[f"media:{key}"] for key in inserts} == {
        key: expected_hashes[source] for key, source in expected_keys.items()
    }
    assert expected_hashes[image] != expected_hashes[other]

    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    assert {item.id: item.payload["media"]["path"] for item in timeline.elements} == {
        "broll-photo": str(retained / "heif-inserts/broll-photo.heic"),
        "broll-photo.heic": str(retained / "inserts/broll-photo.heic"),
    }
    for key, source in expected_keys.items():
        assert receipt.sha256_file(retained / key) == expected_hashes[source]

    sources = assemble_picture._insert_inputs(retained, timeline, [], [])
    assert sources["broll-photo"].path == retained / "heif-inserts/broll-photo.heic"
    assert sources["broll-photo"].grid
    assert (sources["broll-photo"].info.width, sources["broll-photo"].info.height) == (1280, 720)
    assert sources["broll-photo.heic"].path == retained / "inserts/broll-photo.heic"
    assert not sources["broll-photo.heic"].grid
    assert (sources["broll-photo.heic"].info.width,
            sources["broll-photo.heic"].info.height) == (320, 160)


def test_retained_source_name_stays_out_of_published_records(project: Path) -> None:
    image = overlay_image(project / "IMG_0264.png")
    fragment = project / "insert-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-photo", "kind": "insert", "motion": "still", "start": {"time": 0},
        "end": {"time": 1}, "media": {"path": image.name, "sha256": receipt.sha256_file(image)},
    }]})
    add_timeline(project, fragment, lane="broll").require_valid()
    result = stage.assemble(project)
    for record in ("receipt.json", "verify-edl.json", "story.layout.json", "story.faces.json"):
        assert '"source_name"' not in (project / "final" / record).read_text()
    assert result["status"] == "verified"
    assert '"source_name"' not in (project / "timeline/broll.json").read_text()


@pytest.mark.parametrize("rule", [
    "overlay_bounds", "overlay_media", "overlay_speed", "overlay_hdr", "overlay_box",
    "tall_box_overlap", "tall_box_media", "overlay_scroll",
])
def test_overlay_unsafe_rules_cannot_be_overridden(project: Path, rule: str) -> None:
    context = ProjectContext(project)
    timeline = context.resolved_timeline()
    context._timeline = replace(timeline, findings=(TimelineFinding(
        rule, "violation", f"unsafe {rule}", (), (), "Correct media.",
    ),))
    with pytest.raises(TimelineRefusal, match=f"unsafe {rule}"):
        stage.assembly_snapshot(context, allow_violations=True)


@pytest.mark.parametrize("rule", ["background_motion", "background_enter"])
def test_background_effect_rules_cannot_be_overridden(project: Path, rule: str) -> None:
    context = ProjectContext(project)
    timeline = context.resolved_timeline()
    context._timeline = replace(timeline, findings=(TimelineFinding(
        rule, "violation", f"unsafe {rule}", (), (), "Correct background effects.",
    ),))
    with pytest.raises(TimelineRefusal, match=f"unsafe {rule}"):
        stage.assembly_snapshot(context, allow_violations=True)


@pytest.mark.parametrize("fit,motion,expected", [
    ("cover", "none", Box(60, 400, 400, 400)),
    ("contain", "none", Box(60, 500, 400, 200)),
])
def test_overlay_final_layout_drawn_bounds_identity_and_label(
    project: Path, fit: str, motion: str, expected: Box,
) -> None:
    _add_overlay(project, fit=fit, motion=motion)
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    final = project / "record.mp4"
    final.write_bytes(b"final record identity")
    stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    (element,) = record.layout.elements
    assert (element.id, element.kind, element.box, element.start_frame, element.end_frame) == (
        "broll-photo", "overlay", expected, 5, 32,
    )
    assert record.element_ids == {"broll-photo": "broll-photo"}
    assert record.labels == {"broll-photo": "photo.png"}
    assert record.face_windows == {}


@pytest.mark.parametrize("measured,caption_y,chosen,clear", [
    (True, 1700, "shrink", True),
    (True, 880, "keep", False),
    (False, 880, "shrink", None),
    (True, 904, "shrink", True),
    (True, 903, "keep", False),
])
def test_auto_tall_framing_checks_rendered_captions(
    project: Path, monkeypatch, measured, caption_y, chosen, clear,
) -> None:
    style_path = project / "assembly-style.json"
    style = json.loads(style_path.read_bytes())
    style["broll"]["tall_framing"] = "auto"
    style["graphics"]["face_clearance_px"] = 40
    write_json(style_path, style)
    image = overlay_image(project / "page.png", "color=white:s=620x2400")
    fragment = project / "tall-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-page", "kind": "overlay", "start": {"time": 0}, "hold_s": 4,
        "placement": "tall_behind", "motion": "scroll",
        "media": {"path": image.name, "sha256": receipt.sha256_file(image)},
    }]})
    add_timeline(project, fragment, lane="broll").require_valid()
    captions(project)
    graphics(project)
    matte = project / "synthetic-matte.mp4"
    matte.write_bytes(b"synthetic matte identity")
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: matte)
    monkeypatch.setattr(stage, "decode_frames", lambda *args, **kwargs:
                        iter([np.full((1920, 1080), 255, dtype=np.uint8)]))
    # The face's shrunken chin is at y=864; test the style margin's exact boundary.
    face = FaceBox(Box(400, 300, 200, 300), 1, "detected")
    faces = FacesData("measured" if measured else "unmeasured",
                      None if measured else "synthetic detector unavailable",
                      VideoIdentity("cut.mp4", receipt.sha256_file(project / "cut/cut.mp4"),
                                    120, Fraction(30), 1080, 1920), {}, {},
                      tuple((face,) if frame == 118 else () for frame in range(120))
                      if measured else (), ((0, 118), (119, 120)) if measured else ())
    monkeypatch.setattr(stage, "cut_faces", lambda *args, **kwargs: faces)
    layout_path = project / "captions/layout.json"
    layout = json.loads(layout_path.read_bytes())
    layout.update(measured=True, reason=None, captions=[{
        "id": "phrase", "text": "First words", "start_frame": 118, "end_frame": 128,
        "boxes": [Box(0, 800, 50, 40).to_dict(), Box(450, caption_y, 200, 40).to_dict()],
        "words": [{"text": "First", "start_s": .1, "end_s": .5}],
    }, {
        "id": "hold-only", "text": "After the box", "start_frame": 120, "end_frame": 128,
        "boxes": [Box(450, 800, 200, 40).to_dict()],
        "words": [{"text": "After", "start_s": 4, "end_s": 4.2}],
    }])

    def snapshot_at(y):
        layout["captions"][0]["boxes"][1]["y"] = y
        write_json(layout_path, layout)
        record = read_receipt(project / "captions")
        next(item for item in record["outputs"] if item["path"] == "captions/layout.json")[
            "sha256"] = receipt.sha256_file(layout_path)
        write_json(project / "captions/receipt.json", record)
        return stage.assembly_snapshot(ProjectContext(project))

    baseline = snapshot_at(1700)
    snapshot = snapshot_at(caption_y)
    framing = snapshot.timeline.elements[0].payload["framing"]
    assert framing["chosen"] == chosen
    assert framing["captions_clear"] is clear
    assert framing.get("reason") == ("captions" if clear is False else None)
    assert framing["visible_min"] == 0
    assert snapshot.inputs["timeline"] == baseline.inputs["timeline"]
    assert (snapshot.inputs["picture:tall_framing"] != baseline.inputs["picture:tall_framing"]) \
        is (clear is False)


@pytest.mark.parametrize("framing", ["auto", "keep", "shrink"])
def test_tall_box_snapshot_mask_and_final_face_windows(project: Path, monkeypatch, framing) -> None:
    original_style = ProjectContext.render_style

    def render_style(context):
        style = original_style(context)
        return replace(style, data={**style.data, "broll": {
            **style.data["broll"], "tall_framing": framing}})

    monkeypatch.setattr(ProjectContext, "render_style", render_style)
    monkeypatch.setattr(matte_cache, "make_matte", lambda source, folder, *, joins=(), **kwargs:
                        synthetic_matte(source, folder, box=(400, 600, 100, 200), joins=joins))
    image = overlay_image(project / "page.png", "color=white:s=620x2400")
    fragment = project / "tall-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-page", "kind": "overlay", "start": {"time": 0}, "hold_s": 4,
        "placement": "tall_behind", "motion": "scroll",
        "media": {"path": image.name, "sha256": receipt.sha256_file(image)},
    }]})
    add_timeline(project, fragment, lane="broll").require_valid()
    captions(project)
    graphics(project)
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    payload = snapshot.timeline.elements[0].payload
    chosen = "shrink" if framing == "shrink" else "keep"
    assert payload["framing"]["chosen"] == chosen
    assert payload["framing"]["from"] == ("auto" if framing == "auto" else "style")
    assert snapshot.inputs["timeline"] == projection_sha256(context.resolved_timeline().to_dict())
    assert snapshot.inputs["picture:tall_framing"] == projection_sha256([{
        "id": "broll-page", **{key: payload.get(key) for key in ("framing", "box_px", "scroll")},
    }])
    if framing == "auto":
        assert payload["framing"]["visible_min"] >= .5
        assert payload["framing"]["above_min_px"] >= 300
        monkeypatch.setattr(stage.tall_box, "auto_framing", lambda *args, **kwargs: {
            "chosen": "shrink", "visible_min": .2, "above_min_px": 100})
        changed = stage.assembly_snapshot(ProjectContext(project))
        assert changed.inputs["timeline"] == snapshot.inputs["timeline"]
        assert changed.inputs["picture:tall_framing"] != snapshot.inputs["picture:tall_framing"]
        assert changed.timeline.elements[0].payload["box_px"] != payload["box_px"]
        assert changed.timeline.elements[0].payload["scroll"] != payload["scroll"]
    assert "picture:tall_box" in snapshot.inputs
    mask = stage.verification_edl(snapshot, base=project / "final")["overlays"][-1]
    assert mask == {"start_in_output": 0, "duration": 4, "region": (
        [0, 0, 1080, 1920] if chosen == "shrink" else [230, 140, 620, 860])}
    retained = project / "tall-retained"
    timeline = stage._retained(snapshot, retained)
    final = project / "record.mp4"
    final.write_bytes(b"synthetic final identity")
    stage._final_records(context, snapshot, retained, timeline, final)
    layout = json.loads(final.with_suffix(".layout.json").read_bytes())
    element, = layout["elements"]
    if chosen == "keep":
        assert element["face_windows"] == []
        assert element["kind"] == "overlay"
    else:
        assert element["kind"] == "background"
        assert [element[key] for key in ("x", "y", "w", "h")] == [0, 0, 1080, 1920]


@pytest.mark.parametrize("motion", ["none", "drift"])
def test_ordinary_overlay_keeps_tall_filters_and_inputs_out(project, monkeypatch, motion) -> None:
    image = overlay_image(project / "ordinary.png")
    fragment = project / "ordinary-overlay.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-photo", "kind": "overlay", "placement": "headroom", "motion": motion,
        "start": {"time": 0}, "hold_s": 1,
        "media": {"path": image.name, "sha256": receipt.sha256_file(image)},
    }]})
    add_timeline(project, fragment, lane="broll").require_valid()
    captions(project)
    graphics(project)
    music(project)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert "picture:tall_box" not in snapshot.inputs
    retained = project / "ordinary-retained"
    timeline = stage._retained(snapshot, retained)
    monkeypatch.setattr(assemble_picture, "ensure_cut_matte", lambda *args, **kwargs:
                        pytest.fail("Ordinary overlays must not request a cut matte"))
    assemble_picture.render_picture(retained, project / "ordinary.mp4", work_dir=project / "work",
        timeline=timeline, title_layers=list(snapshot.title_layers),
        caption_layer=snapshot.caption_layer)
    graph = (project / "work/picture.ffgraph").read_text()
    assert all(value not in graph for value in
               ("fillborders", "tall_edge", "person_tall", "matte_tall"))


@pytest.mark.parametrize("kind", ["title", "front", "behind"])
def test_tall_box_refuses_cards_and_titles_even_with_override(project, monkeypatch, kind) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"elements_at_once_max": 2}
    write_json(project / "project.json", data)
    image = overlay_image(project / "page.png", "color=white:s=620x2400")
    fragment = project / "tall-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-page", "kind": "overlay", "start": {"time": 0}, "hold_s": 4,
        "placement": "tall_behind", "media": {"path": image.name,
                                               "sha256": receipt.sha256_file(image)},
    }]})
    add_timeline(project, fragment, lane="broll").require_valid()
    other = (title(start={"time": 0}, hold_s=1) if kind == "title" else
             graphic(start={"time": 0}, hold_s=1, layer=kind))
    set_elements(project, other)
    context = ProjectContext(project)
    timeline = context.resolved_timeline(allow_violations=True)
    finding = next(f for f in timeline.findings if f.rule == "tall_box_overlap")
    assert finding.severity == "violation"
    with pytest.raises(TimelineRefusal, match="tall box cannot overlap"):
        stage.assembly_snapshot(context, allow_violations=True)
    # Retained placement must remain guarded even if the timeline finding is missing.
    monkeypatch.setattr(ProjectContext, "resolved_timeline", lambda self, **kwargs:
                        replace(timeline, findings=()))
    captions(project)
    graphics(project)
    if kind == "behind":
        record = read_receipt(project / "graphics")
        record["elements"][0]["layer"] = "behind"
        write_json(project / "graphics/receipt.json", record)
    with pytest.raises(TimelineRefusal, match="tall box cannot overlap") as error:
        stage.assembly_snapshot(ProjectContext(project), allow_violations=True)
    assert error.value.suggested_action == "Move the card or title outside the tall box window."


def test_overridden_tall_box_render_stays_fresh(project: Path, monkeypatch) -> None:
    from yap_studio.project.status import fillers_inputs

    monkeypatch.setattr(matte_cache, "make_matte", lambda source, folder, *, joins=(), **kwargs:
                        synthetic_matte(source, folder, box=(400, 600, 100, 200), joins=joins))
    page = overlay_image(project / "page.png", "color=white:s=620x2400")
    set_elements(project, {"id": "broll-page", "kind": "overlay", "start": {"time": 0},
                          "hold_s": 4, "placement": "tall_behind", "motion": "scroll",
                          "media": {"path": page.name, "sha256": receipt.sha256_file(page)}},
                 lane="broll")
    captions(project)
    graphics(project)
    with StageRun(project, "fillers") as run:
        for key, value in fillers_inputs(ProjectContext(project)).items():
            run.record_inputs(key, value)
        run.finish()
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"elements_at_once_max": 0}
    write_json(project / "project.json", data)
    assert any(f.rule == "visual_overlap" for f in
               ProjectContext(project).resolved_timeline(allow_violations=True).findings)
    result = stage.assemble(project, allow_violations=True)
    assert result["status"] == "verified"
    record = read_receipt(project / "final")
    assert record["allow_violations"] and record["status"] == "verified"
    states = {item["name"]: item["state"] for item in status(project)["stages"]}
    assert states == dict.fromkeys(states, "done")
    assert stage.assemble(project, allow_violations=True)["noop"]


def test_overlay_drift_layout_pieces_have_drawn_bounds_and_identity(project: Path) -> None:
    _add_overlay(project, fit="contain", motion="drift")
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    document = json.loads((retained / "captions/layout.json").read_bytes())
    document["captions"] = [{
        "id": "broll-photo#2", "text": "First words", "start_frame": 3, "end_frame": 30,
        "boxes": [Box(10, 1900, 80, 10).to_dict()],
        "words": [{"text": "First words", "start_s": 0.1, "end_s": 1.0}],
    }]
    write_json(retained / "captions/layout.json", document)
    final = project / "record.mp4"
    final.write_bytes(b"final record identity")
    stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    elements = record.layout.elements
    assert len(elements) > 1
    assert [element.id for element in elements] == [
        "broll-photo", *(f"broll-photo#{number}" for number in range(3, len(elements) + 2)),
    ]
    assert (elements[0].start_frame, elements[-1].end_frame) == (5, 32)
    assert elements[0].box.h < elements[-1].box.h
    assert all(first.end_frame == second.start_frame and first.box != second.box
               for first, second in zip(elements, elements[1:], strict=False))
    assert record.element_ids == {element.id: "broll-photo" for element in elements}
    assert record.labels == {element.id: "photo.png" for element in elements}


@pytest.mark.parametrize("opaque", [False, True])
def test_transparent_overlay_half_clears_face_and_caption(
    project: Path, monkeypatch: pytest.MonkeyPatch, opaque: bool,
) -> None:
    _add_overlay(project, box=(100, 300, 400, 200), lavfi="color=red:s=400x200,format=rgba,"
                 "geq=r=255:g=0:b=0:a='255*lt(X,200)'")
    target = Box(160 if opaque else 400, 360, 30, 40)
    _measured_caption_layout(project, box=Box(target.x, 480, 30, 10))

    def measured_cut(root: Path, cut: Path) -> FacesData:
        frames = tuple((FaceBox(target, 0.9, "detected"),) for _ in range(120))
        return FacesData(
            "measured", None,
            VideoIdentity(cut.name, receipt.sha256_file(cut), 120, Fraction(30), 1080, 1920),
            {"name": "fake"}, {}, frames, (),
        )

    monkeypatch.setattr(stage, "cut_faces", measured_cut)
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    final = project / "record.mp4"
    final.write_bytes(b"final record identity")
    _, faces_path = stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    (element,) = record.layout.elements
    assert element.box == Box(100, 300, 200, 200)
    findings, coverage = check_face_clearance(record, load_faces(faces_path), faces_source="record")
    assert coverage == {"ran": True, "reason": None}
    assert bool(findings) == opaque
    assert bool(check_caption_cards(record, fps=30)) == opaque
    if opaque:
        assert findings[0].evidence["target_kind"] == "overlay"


def test_overlay_piece_id_avoids_caption_ids_and_keeps_original_name(project: Path) -> None:
    _add_overlay(project)
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    document = json.loads((retained / "captions/layout.json").read_bytes())
    document.update(measured=True, reason=None, captions=[{
        "id": name, "text": "First words", "start_frame": 3, "end_frame": 30,
        "boxes": [Box(10, 1900, 80, 10).to_dict()],
        "words": [{"text": "First words", "start_s": 0.1, "end_s": 1.0}],
    } for name in ("broll-photo", "broll-photo#2")])
    write_json(retained / "captions/layout.json", document)
    final = project / "record.mp4"
    final.write_bytes(b"final record identity")
    stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    (element,) = record.layout.elements
    assert element.id == "broll-photo#3"
    assert record.element_ids == {"broll-photo#3": "broll-photo"}
    assert record.labels == {"broll-photo#3": "photo.png"}


def test_card_piece_ids_reserve_literal_overlay_names(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_overlay(project)
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "retained"
    timeline = stage._retained(snapshot, retained)
    original = replace(snapshot.timeline.elements[0], id="agent-card#2")
    timeline = replace(timeline, elements=(replace(timeline.elements[0], id=original.id),))
    snapshot = replace(snapshot, timeline=replace(snapshot.timeline, elements=(original,)),
                       title_layers=[{
                           "id": "agent-card", "kind": "graphic", "path": "card.mov",
                           "x": 10, "y": 300, "width": 100, "height": 40,
                           "start_frame": 3, "frames": 30,
                       }])
    monkeypatch.setattr(stage, "drawn_boxes", lambda *args, **kwargs: [
        (0, 15, Box(10, 300, 40, 40)), (15, 30, Box(20, 300, 40, 40)),
    ])
    final = project / "record.mp4"
    final.write_bytes(b"final record identity")
    stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    assert [element.id for element in record.layout.elements] == [
        "agent-card", "agent-card#3", "agent-card#2",
    ]
    assert record.element_ids["agent-card#2"] == "agent-card#2"
    assert record.labels == {"agent-card#2": "photo.png"}


@pytest.mark.parametrize("check", ["faces.clearance", "captions.cards"])
def test_overlay_overlap_blocks_with_label_and_time_and_override_publishes(
    project: Path, monkeypatch: pytest.MonkeyPatch, check: str,
) -> None:
    _add_overlay(project, box=(100, 300, 400, 200))
    if check == "captions.cards":
        _measured_caption_layout(project, box=Box(150, 350, 80, 10))

    def measured_cut(root: Path, cut: Path) -> FacesData:
        face = Box(160, 360, 30, 40) if check == "faces.clearance" else Box(700, 900, 40, 40)
        frames = tuple((FaceBox(face, 0.9, "detected"),) for _ in range(120))
        return FacesData(
            "measured", None,
            VideoIdentity(cut.name, receipt.sha256_file(cut), 120, Fraction(30), 1080, 1920),
            {"name": "fake"}, {}, frames, (),
        )

    monkeypatch.setattr(stage, "cut_faces", measured_cut)
    with pytest.raises(stage.AssemblyError) as caught:
        stage.assemble(project)
    assert 'Overlay "broll-photo" (photo.png)' in caught.value.message
    assert "0:00.1" in caught.value.message
    assert ("from 0:00.1 to 0:01.0" if check == "faces.clearance" else "at 0:00.1") \
        in caught.value.message
    failed = read_receipt(project / "final")
    failures = [row for row in failed["verify"]["findings"] if row["severity"] == "fail"]
    assert {row["check"] for row in failures} == {check}
    assert failures[0]["evidence"]["label"] == "photo.png"
    assert failures[0]["evidence"]["start_frame"] == 5
    assert failures[0]["evidence"]["end_frame"] == (32 if check == "faces.clearance" else 30)
    assert not (project / "final/story.mp4").exists()
    result = stage.assemble(project, allow_violations=True)
    final = project / result["out"]
    assert final.is_file()
    record = read_layout_record(final, width=1080, height=1920)
    assert record.labels == {"broll-photo": "photo.png"} and record.face_windows == {}
    published = read_receipt(project / "final")
    assert published["status"] == "verified" and published["verify"]["status"] == "fail"
    assert published["allowed_layout_violations"] == [
        {key: row[key] for key in ("check", "time_s", "message")} for row in failures
    ]


@pytest.mark.parametrize("camera", [False, True, "empty"])
def test_assembly_renders_transition_from_retained_cut_and_records_window(
    project: Path, monkeypatch: pytest.MonkeyPatch, camera: bool | str,
) -> None:
    """Exercise retained rendering, transition verification and receipt publication."""
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    item = ResolvedElement("agent-transition", "transition", "agent", 54, 66, "front",
                           {"look": "cinematic_zoom", "frames": 12, "join_frame": 60})
    snapshot = replace(snapshot, timeline=replace(snapshot.timeline, elements=(item,),
                                                 camera_3d=bool(camera)))
    if camera == "empty":
        monkeypatch.setattr(assemble_picture, "captions", lambda *a: pytest.fail(
            "rendered captions again instead of using the retained empty layer"))
    if camera is True:
        picture = project / "captions/camera3d.mp4"
        shutil.copyfile(project / "cut/cut.mp4", picture)
        plan = {"frames": 120, "camera": {"samples": [
            {"cz": 0, "px": 0, "py": 0, "tx": 0, "ty": 0} for _ in range(120)]}}
        snapshot = replace(snapshot, picture="captions/camera3d.mp4",
                           media={**snapshot.media, "captions/camera3d.mp4": picture},
                           inputs={**snapshot.inputs,
                                   "media:captions/camera3d.mp4": receipt.sha256_file(picture)},
                           small_files={**snapshot.small_files,
                                        "captions/camera3d.json": json.dumps(plan).encode()})
        monkeypatch.setattr(stage, "_measure_contrast", lambda *a, **kw: (
            {"schema": "yap-studio/caption-contrast@1", "measured": True, "reason": None,
             "captions": []}, {"measured": True, "reason": None, "captions": 0, "weak": []}))
    monkeypatch.setattr(stage, "assembly_snapshot", lambda *args, **kwargs: snapshot)
    calls = []

    def fake(root: Path, timeline: Any, cut: Path, work: Path) -> list[dict]:
        assert root == project and cut != project / "cut/cut.mp4"
        assert cut.name == ("camera3d.mp4" if camera is True else "cut.mp4")
        assert receipt.sha256_file(cut) == receipt.sha256_file(project / "cut/cut.mp4")
        assert timeline.elements == (item,)
        movie = synth.picture_source(work / "transition.mov",
                                     "color=c=red:s=1080x1920:r=30", 12)
        calls.append(movie)
        return [{"id": item.id, "path": str(movie), "start_frame": 54, "frames": 12}]

    monkeypatch.setattr(stage, "render_transitions", fake)
    verified = []

    def verify_final(*args: Any, **kwargs: Any) -> dict:
        assert kwargs["transitions"] == [TransitionWindow(60, 54, 66, item.id)]
        verified.append(args[0])
        return {"status": "pass"}

    monkeypatch.setattr(stage, "verify", verify_final)
    result = stage.assemble(project)
    record = read_receipt(project / "final")
    assert len(calls) == 1 and record["status"] == "verified"
    assert len(verified) == 1
    assert record["transitions"] == [{"id": item.id, "look": "cinematic_zoom", "frames": 12,
                                       "start_frame": 54, "join_frame": 60}]
    assert record["audio"]["policy"] == {"mode": "preserve"}
    pixels = synth.picture_pixels(project / result["out"], 60, box=(500, 700, 24, 24))
    np.testing.assert_allclose(pixels.mean(axis=(0, 1)), (255, 0, 0), atol=8)


@pytest.mark.parametrize("bed", [False, True])
@pytest.mark.parametrize("effects", [False, True])
def test_final_verify_receives_click_stem_when_present(
    project: Path, monkeypatch: pytest.MonkeyPatch, bed: bool, effects: bool,
) -> None:
    if effects:
        style_path = project / "assembly-style.json"
        style = json.loads(style_path.read_text())
        style["music"]["sound_effects"] = "made"
        write_json(style_path, style)
    if bed:
        time = np.arange(8 * 48000) / 48000
        synth.float_audio(project / "track.wav", 0.1 * np.sin(2 * np.pi * 880 * time))
        data = json.loads((project / "project.json").read_bytes())
        data["music"] = {"mode": "file", "file": "track.wav"}
        write_json(project / "project.json", data)
        music(project)

    seen = []

    def verify_final(*args: Any, **kwargs: Any) -> dict:
        seen.append(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(stage, "verify", verify_final)
    stage.assemble(project)
    assert len(seen) == 1
    assert ("click_audio" in seen[0]) is (bed or effects)
    assert ("words_audio" in seen[0]) is effects
    if effects:
        assert seen[0]["words_audio"].endswith("/no-effects-stem.wav")
        if not bed:
            assert seen[0]["click_audio"] == seen[0]["words_audio"]
    if bed:
        assert seen[0]["click_audio"].endswith("/voice-stem.wav")


def test_no_transition_assembly_does_not_render_shaders(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An empty transition timeline must not invoke its renderer")

    monkeypatch.setattr(stage, "render_transitions", unexpected)
    stage.assemble(project)
    assert "transitions" not in read_receipt(project / "final")


def test_allow_violations_with_transition_publishes_final(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authorized editorial finding must survive the real transition render path."""
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    item = ResolvedElement("agent-transition", "transition", "agent", 54, 66, "front",
                           {"look": "cinematic_zoom", "frames": 12, "join_frame": 60})
    finding = TimelineFinding("zoom_count", "violation", "too many zooms in the video",
                              (), (), "Remove an extra zoom.")
    snapshot = replace(snapshot, timeline=replace(snapshot.timeline, elements=(item,),
                                                findings=(finding,)))
    assert not snapshot.timeline.valid

    def authorized_snapshot(context: Any, *, allow_violations: bool = False) -> Any:
        assert allow_violations
        return snapshot

    calls = []

    def render_layer(composition: Path, out: Path, **options: Any) -> None:
        assert len(list((composition / "from").glob("*.png"))) == 6
        assert len(list((composition / "to").glob("*.png"))) == 6
        calls.append(out)
        synth.picture_source(out, "color=c=red:s=1080x1920:r=30", options["frames"])

    monkeypatch.setattr(stage, "assembly_snapshot", authorized_snapshot)
    monkeypatch.setattr(transition_stage.hyperframes, "render_layer", render_layer)
    monkeypatch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "pass"})
    result = stage.assemble(project, allow_violations=True)
    record = read_receipt(project / "final")
    assert len(calls) == 1 and (project / result["out"]).is_file()
    assert record["status"] == "verified" and record["allow_violations"] is True
    assert record["timeline"]["findings"][0]["rule"] == "zoom_count"
    assert record["transitions"][0]["id"] == item.id


def test_timeline_cards_reach_verified_final(project: Path) -> None:
    """Placed text and numeric card movies must both survive final assembly."""
    set_elements(project, graphic(), graphic(
        id="agent-stat", start={"time": 1.5}, template="stat", slots={"end": 12},
    ))
    graphics(project)
    result = stage.assemble(project)
    items = read_receipt(project / "graphics")["elements"]
    assert len(items) == 2
    for item in items:
        pixels = synth.picture_pixels(project / result["out"], item["start_frame"] + 3,
                                      box=(item["x"] + 50, item["y"] + 50, 20, 20))
        np.testing.assert_allclose(pixels.mean(axis=(0, 1)), (230, 90, 40), atol=5)
    assert read_receipt(project / "final")["status"] == "verified"
    assert final_stage(project)["state"] == "done"


def _measured_caption_layout(project: Path, *, box: Box | None = None) -> dict:
    document = json.loads((project / "captions/layout.json").read_bytes())
    document.update({"measured": True, "reason": None, "captions": [
        {"id": "caption:0", "text": "First words", "start_frame": 3, "end_frame": 30,
         "boxes": [(box or Box(10, 1900, 80, 10)).to_dict()],
         "words": [{"text": "First", "start_s": 0.1, "end_s": 0.5},
                   {"text": "words", "start_s": 0.6, "end_s": 1.0}]},
    ]})
    write_json(project / "captions/layout.json", document)
    record = read_receipt(project / "captions")
    for item in record["outputs"]:
        if item["path"] == "captions/layout.json":
            item["sha256"] = receipt.sha256_file(project / item["path"])
    write_json(project / "captions/receipt.json", record)
    return document


def test_final_records_bind_retained_captions_and_drawn_cards(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    render = hyperframes.render_layer

    def padded(composition: Path, out: Path, **kwargs: Any) -> Any:
        result = render(composition, out, **kwargs)
        text = kwargs["variables"].get("text")
        if text is not None:
            temporary = out.with_name("padded.mov")
            alpha = "0" if text == "Empty" else \
                "if(between(X,10,89)*between(Y,20,59),255,51)"
            synth.picture_source(
                temporary, f"color=c=black@0:s={result.width}x{result.height}:r=30,"
                f"format=rgba,geq=r=230:g=90:b=40:a='{alpha}'", result.frames, alpha=True,
            )
            temporary.replace(out)
        return result

    monkeypatch.setattr(hyperframes, "render_layer", padded)
    set_elements(project, graphic(), title(start={"time": 1.5}, hold_s=0.5),
                 graphic(id="agent-empty-card", start={"time": 3}, hold_s=0.5,
                         slots={"text": "Empty"}))
    captions(project)
    document = _measured_caption_layout(project)
    graphics(project)
    layout_sha = receipt.sha256_file(project / "captions/layout.json")
    original = stage.render_picture

    def changed(*args: Any, **kwargs: Any) -> Any:
        (project / "captions/layout.json").write_bytes(b"changed during rendering")
        return original(*args, **kwargs)

    monkeypatch.setattr(stage, "render_picture", changed)
    original_verify = stage.verify

    def verify_records(media: str, **kwargs: Any) -> dict:
        assert Path(media).with_suffix(".layout.json").is_file()
        assert Path(media).with_suffix(".faces.json").is_file()
        return original_verify(media, **kwargs)

    monkeypatch.setattr(stage, "verify", verify_records)
    result = stage.assemble(project)
    final = project / result["out"]
    layout = json.loads(final.with_suffix(".layout.json").read_bytes())
    faces = load_faces(final.with_suffix(".faces.json"))
    digest = receipt.sha256_file(final)
    assert layout["media_sha256"] == faces.video.sha256 == digest
    assert faces.video.name == final.name
    assert layout["captions"] == document["captions"]
    assert (layout["measured"], layout["reason"]) == (True, None)
    parsed = Layout.from_dict(layout)
    assert len(parsed.elements) == 2
    assert all("face_windows" not in item for item in layout["elements"])
    assert {item["id"]: item["element_id"] for item in layout["elements"]} == {
        "agent-tag": "agent-tag", "agent-title": "agent-title",
    }
    card = next(item for item in parsed.elements if item.kind == "card")
    heading = next(item for item in parsed.elements if item.kind == "title")
    layers = read_receipt(project / "graphics")["elements"]
    card_layer = next(item for item in layers if item["id"] == "agent-tag")
    title_layer = next(item for item in layers if item["id"] == "agent-title")
    assert (card.id, card.kind, card.box, card.start_frame, card.end_frame) == (
        "agent-tag", "card", Box(card_layer["x"] + 10, card_layer["y"] + 20, 80, 40), 6, 36,
    )
    assert (heading.id, heading.kind, heading.box, heading.start_frame, heading.end_frame) == (
        "agent-title", "title", Box(title_layer["x"], title_layer["y"], title_layer["width"],
                                    title_layer["height"]), 45, 60,
    )
    assert parsed.hair == DEFAULT_HAIR
    assert parsed.clearance_px == face_clearance_px(ProjectContext(project).render_style().data)
    assert parsed.caption_gap_px == caption_gap_px(ProjectContext(project).render_style().data)
    assert (faces.status, faces.reason) == ("unmeasured", "Vision unavailable in this test")
    record = read_receipt(project / "final")
    assert record["inputs"]["captions/layout.json"] == layout_sha
    assert {str(final.with_suffix(suffix).relative_to(project))
            for suffix in (".layout.json", ".faces.json")} <= \
        {item["path"] for item in record["outputs"]}


@pytest.mark.parametrize("feed", [False, True])
def test_final_records_keep_grid_parts_and_feed_background(
    project: Path, monkeypatch: pytest.MonkeyPatch, feed: bool,
) -> None:
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "scene-retained"
    retained.mkdir()
    stage._retained(snapshot, retained)
    synth.picture_source(
        retained / "scene.mov", "color=black@0:s=1080x1920:r=30,format=rgba,"
        "geq=r=255:g=255:b=255:a='255*between(Y,320,359)*(between(X,50+4*N,99+4*N)+"
        "between(X,700,749))'", 2, alpha=True,
    )
    card = ResolvedElement("agent-scene", "graphic", "agent", 6, 8, "front", {})
    timeline = replace(snapshot.timeline, elements=(card,))
    layer = {"id": card.id, "kind": "graphic", "layer": "front", "path": "scene.mov",
             "x": 0, "y": 0, "width": 1080, "height": 1920, "start_frame": 6, "frames": 2}
    if feed:
        layer["full_frame"] = True
    else:
        layer["parts"] = [{"x": 0, "y": 300, "width": 200, "height": 200},
                          {"x": 600, "y": 300, "width": 200, "height": 200}]
    snapshot = replace(snapshot, timeline=timeline, title_layers=[layer])

    def measured_cut(root: Path, cut: Path) -> FacesData:
        frames = ((FaceBox(Box(400, 360, 100, 100), 0.9, "detected"),),) * 120
        return FacesData("measured", None,
                         VideoIdentity(cut.name, receipt.sha256_file(cut), 120, Fraction(30),
                                       1080, 1920), {"name": "fake"}, {}, frames, ())

    monkeypatch.setattr(stage, "cut_faces", measured_cut)
    final = project / "scene-record.mp4"
    final.write_bytes(b"scene record identity")
    _, faces_path = stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    assert record is not None
    if feed:
        element, = record.layout.elements
        assert element.kind == "background" and element.box == Box(0, 0, 1080, 1920)
        assert record.labels == {card.id: card.id}
    else:
        assert [element.id for element in record.layout.elements] == [card.id, card.id + "#2"]
        assert [element.box for element in record.layout.elements] == [
            Box(50, 320, 54, 40), Box(700, 320, 50, 40),
        ]
    findings, _ = check_face_clearance(record, load_faces(faces_path), faces_source="record")
    assert not findings


def _card_final_records(
    project: Path, monkeypatch: pytest.MonkeyPatch, *, behind: bool = False,
    picture: tuple[ResolvedElement, ...] = (), moving: bool = False, face_start: int = 6,
):
    """Decode real alpha frames and publish records with deterministic cut face geometry."""
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "retained"
    retained.mkdir()
    stage._retained(snapshot, retained)
    shift = "400*gte(N,15)" if moving else "0"
    synth.picture_source(
        retained / "card.mov", "color=c=black@0:s=512x48:r=30,format=rgba,"
        f"geq=r=230:g=90:b=40:a='if(between(X,10+{shift},49+{shift})*"
        "between(Y,12,39),255,0)'", 30, alpha=True,
    )
    card = ResolvedElement("agent-card", "graphic", "agent", 6, 36,
                           "behind" if behind else "front", {})
    timeline = replace(snapshot.timeline, elements=tuple(sorted(
        (card, *picture), key=lambda item: (item.start_frame, item.id),
    )))
    snapshot = replace(snapshot, timeline=timeline, title_layers=[{
        "id": card.id, "kind": "graphic", "layer": card.layer, "path": "card.mov",
        "x": 100, "y": 300, "width": 512, "height": 48, "start_frame": 6, "frames": 30,
    }])

    def measured_cut(root: Path, cut: Path) -> FacesData:
        frames = tuple((FaceBox(Box(120, 320, 12, 12), 0.9, "detected"),)
                       if face_start <= frame < 36 else () for frame in range(120))
        return FacesData(
            "measured", None,
            VideoIdentity(cut.name, receipt.sha256_file(cut), 120, Fraction(30), 1080, 1920),
            {"name": "fake"}, {}, frames, tuple(detect.find_gaps(frames)),
        )

    monkeypatch.setattr(stage, "cut_faces", measured_cut)
    final = project / "record.mp4"
    final.write_bytes(b"final record identity")
    _, faces_path = stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    assert record is not None
    faces = load_faces(faces_path)
    findings, coverage = check_face_clearance(record, faces, faces_source="record")
    assert coverage["ran"] is True
    return record, faces, findings


@pytest.mark.parametrize("kind,start,end", [
    (None, 0, 0), ("zoom", 40, 60), ("zoom", 15, 30),
    ("insert", 15, 30), ("transition", 15, 30),
])
def test_behind_card_checked_only_where_person_restoration_is_skipped(
    project: Path, monkeypatch: pytest.MonkeyPatch, kind: str | None, start: int, end: int,
) -> None:
    picture = () if kind is None else (ResolvedElement(
        f"agent-{kind}", kind, "agent", start, end, "front",
        {"peak_scale": 1.05, "center": [0.12, 0.17]} if kind == "zoom" else {},
    ),)
    record, faces, findings = _card_final_records(project, monkeypatch, behind=True,
                                                 picture=picture)
    windows = [(item.start_frame, item.end_frame) for item in record.layout.elements]
    assert windows == [(6, 36)]
    assert record.face_windows == {"agent-card": ((15, 30),) if start == 15 else ()}
    if kind == "zoom" and start == 15:
        assert [(row.evidence["start_frame"], row.evidence["end_frame"])
                for row in findings] == [(15, 30)]
    else:
        assert findings == []
    if kind in {"insert", "transition"}:
        assert all(not faces.frames[frame] for frame in range(15, 30))
        assert {"start_frame": 15, "end_frame": 30, "source": kind} in \
            faces.sampling["frame_sources"]


def test_animated_card_moves_away_before_face_arrives(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, _, findings = _card_final_records(project, monkeypatch, moving=True, face_start=21)
    assert [(item.id, item.start_frame, item.end_frame, item.box)
            for item in record.layout.elements] == [
        ("agent-card", 6, 21, Box(110, 312, 40, 28)),
        ("agent-card#2", 21, 36, Box(510, 312, 40, 28)),
    ]
    assert record.face_windows == {}
    assert findings == []


def test_behind_animated_pieces_intersect_merged_skipped_windows(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zoom = ResolvedElement("agent-zoom", "zoom", "agent", 12, 27, "front",
                           {"peak_scale": 1.05, "center": [0.12, 0.17]})
    insert = ResolvedElement("agent-insert", "insert", "agent", 24, 30, "front", {})
    transition = ResolvedElement("agent-transition", "transition", "agent", 32, 34, "front", {})
    record, _, findings = _card_final_records(project, monkeypatch, behind=True, moving=True,
                                              picture=(zoom, insert, transition))
    assert [(item.id, item.start_frame, item.end_frame, item.box)
            for item in record.layout.elements] == [
        ("agent-card", 6, 21, Box(110, 312, 40, 28)),
        ("agent-card#2", 21, 36, Box(510, 312, 40, 28)),
    ]
    assert record.face_windows == {"agent-card": ((12, 21),),
                                   "agent-card#2": ((21, 30), (32, 34))}
    assert [(row.evidence["start_frame"], row.evidence["end_frame"])
            for row in findings] == [(12, 21)]


def test_final_piece_ids_and_verify_preserve_literal_card_names(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    retained = project / "retained"
    retained.mkdir()
    stage._retained(snapshot, retained)
    box = Box(110, 312, 40, 28)
    document = json.loads((retained / "captions/layout.json").read_bytes())
    document.update(measured=True, reason=None, captions=[{
        "id": "card#3", "text": "First words", "start_frame": 6, "end_frame": 36,
        "boxes": [box.to_dict()],
        "words": [{"text": "First words", "start_s": 0.2, "end_s": 1.2}],
    }])
    write_json(retained / "captions/layout.json", document)
    names = ("card", "card~2", "card#2", "intro~alt")
    cards = tuple(ResolvedElement(name, "graphic", "agent", 6, 36,
                                  "behind" if name == "card" else "front", {})
                  for name in names)
    zooms = tuple(ResolvedElement(
        f"zoom-{start}", "zoom", "agent", start, end, "front",
        {"peak_scale": 1.05, "center": [0.12, 0.17]},
    ) for start, end in ((8, 12), (16, 30)))
    timeline = replace(snapshot.timeline, elements=(*cards, *zooms), end_hold_frames=0)
    snapshot = replace(snapshot, timeline=timeline, title_layers=[{
        "id": name, "kind": "graphic", "path": f"{number}.mov", "x": 100, "y": 300,
        "width": 512, "height": 48, "start_frame": 6, "frames": 30,
    } for number, name in enumerate(names)])

    def drawn_pieces(layer: Path, **kwargs: Any) -> list[tuple[int, int, Box]]:
        return [(0, 15, box), (15, 30, Box(115, 312, 40, 28))] if layer.name == "0.mov" \
            else [(0, 30, box)]

    def measured_cut(root: Path, cut: Path) -> FacesData:
        frames = tuple((FaceBox(Box(120, 320, 12, 12), 0.9, "detected"),)
                       if 6 <= frame < 36 else () for frame in range(120))
        return FacesData(
            "measured", None,
            VideoIdentity(cut.name, receipt.sha256_file(cut), 120, Fraction(30), 1080, 1920),
            {"name": "fake"}, {}, frames, tuple(detect.find_gaps(frames)),
        )

    monkeypatch.setattr(stage, "drawn_boxes", drawn_pieces)
    monkeypatch.setattr(stage, "cut_faces", measured_cut)
    final = project / "record.mp4"
    shutil.copyfile(retained / "cut/cut.mp4", final)
    stage._final_records(context, snapshot, retained, timeline, final)
    record = read_layout_record(final, width=1080, height=1920)
    assert record is not None
    piece_ids = [item.id for item in record.layout.elements]
    assert piece_ids == ["card", "card#4", "card~2", "card#2", "intro~alt"]
    assert len(set(piece_ids + [caption.id for caption in record.layout.captions])) == 6
    assert record.element_ids == {
        "card": "card", "card#4": "card", "card~2": "card~2",
        "card#2": "card#2", "intro~alt": "intro~alt",
    }
    assert record.face_windows == {"card": ((8, 12), (16, 21)), "card#4": ((21, 30),)}
    report = stage.verify(str(final), asr="none", loudness="note")
    for check in ("captions.cards", "faces.clearance"):
        assert report["checks"][check]["ran"] is True
        assert report["checks"][check]["status"] == "fail"
        findings = [row for row in report["findings"] if row["check"] == check
                    and row["evidence"].get("element_id")]
        piece_key = "element_piece" if check == "captions.cards" else "target_id"
        assert {row["evidence"][piece_key] for row in findings} == set(piece_ids)
        for row in findings:
            element_id = record.element_ids[row["evidence"][piece_key]]
            assert row["evidence"]["element_id"] == element_id
            assert f'Card "{element_id}"' in row["message"]
        if check == "faces.clearance":
            assert sorted((row["evidence"]["start_frame"], row["evidence"]["end_frame"])
                          for row in findings if row["evidence"]["target_id"] == "card") == [
                (8, 12), (16, 21),
            ]


def _behind_graphics_contract(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze a behind receipt to exercise assembly independently of card planning."""
    snapshot = graphics_stage.graphics_snapshot(ProjectContext(project))
    snapshot = replace(snapshot, graphics=tuple(
        replace(card, layer="behind") for card in snapshot.graphics
    ))
    monkeypatch.setattr(graphics_stage, "graphics_snapshot",
                        lambda context, placements=None: snapshot)


def test_behind_card_over_caption_outside_skipped_windows_blocks_delivery(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_elements(project, graphic(layer="behind"))
    _behind_graphics_contract(project, monkeypatch)
    graphics(project)
    layer = read_receipt(project / "graphics")["elements"][0]
    _measured_caption_layout(project, box=Box(layer["x"] + 50, layer["y"] + 50, 80, 10))
    face = Box(layer["x"] + 50, layer["y"] + layer["height"] - 50, 40, 30)

    def measured_cut(root: Path, cut: Path) -> FacesData:
        frames = tuple((FaceBox(face, 0.9, "detected"),) for _ in range(120))
        return FacesData(
            "measured", None,
            VideoIdentity(cut.name, receipt.sha256_file(cut), 120, Fraction(30), 1080, 1920),
            {"name": "fake"}, {}, frames, (),
        )

    monkeypatch.setattr(stage, "cut_faces", measured_cut)
    monkeypatch.setattr(matte_cache, "make_matte",
                        lambda source, folder, *, joins=(): synthetic_matte(
        source, folder, box=(int(face.x), int(face.y), int(face.w), int(face.h)), joins=joins,
    ))
    with pytest.raises(stage.AssemblyError, match="covers the caption"):
        stage.assemble(project)
    failed = read_receipt(project / "final")
    failures = [row for row in failed["verify"]["findings"] if row["severity"] == "fail"]
    assert {row["check"] for row in failures} == {"captions.cards"}
    assert failed["verify"]["checks"]["faces.clearance"]["status"] == "pass"
    assert failures[0]["evidence"]["start_frame"] == 6
    assert not (project / "final/story.mp4").exists()

    result = stage.assemble(project, allow_violations=True)
    final = project / result["out"]
    record = read_layout_record(final, width=1080, height=1920)
    assert record is not None and record.face_windows == {layer["id"]: ()}
    assert [(item.start_frame, item.end_frame) for item in record.layout.elements] == [(6, 36)]
    published = read_receipt(project / "final")
    assert published["status"] == "verified" and published["verify"]["status"] == "fail"
    assert published["allowed_layout_violations"] == [
        {key: row[key] for key in ("check", "time_s", "message")} for row in failures
    ]


@pytest.mark.parametrize("take_size", [None, (1602, 3004)])
def test_final_faces_map_exact_zoom_insert_and_hold(
    project: Path, monkeypatch: pytest.MonkeyPatch, take_size: tuple[int, int] | None,
) -> None:
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    zoom = ResolvedElement("agent-zoom", "zoom", "agent", 15, 75, "front",
                           {"peak_scale": 1.08, "center": [0.38, 0.62]})
    insert_media = synth.picture_source(project / "insert.mov",
                                        "color=c=red:s=108x192:r=30", 15)
    insert = ResolvedElement("agent-insert", "insert", "agent", 45, 60, "front", {
        "media": {"path": "insert.mov", "sha256": receipt.sha256_file(insert_media)},
    })
    media = {**snapshot.media, "inserts/agent-insert": insert_media}
    inputs = {**snapshot.inputs, "media:inserts/agent-insert": receipt.sha256_file(insert_media)}
    if take_size is not None:
        width, height = take_size
        take = synth.picture_source(project / "native.mov",
                                     f"color=c=blue:s={width}x{height}:r=30", 120)
        key = snapshot.sources["t1"]
        media[key] = take
        inputs[f"media:{key}"] = receipt.sha256_file(take)
    snapshot = replace(snapshot, media=media, inputs=inputs,
                       timeline=replace(snapshot.timeline, elements=(zoom, insert)))
    monkeypatch.setattr(stage, "assembly_snapshot", lambda *args, **kwargs: snapshot)
    calls = []

    class Detector:
        def detect(self, video: Path, *, every: int) -> detect.DetectorRun:
            assert video != project / "cut/cut.mp4"
            assert receipt.sha256_file(video) == receipt.sha256_file(project / "cut/cut.mp4")
            calls.append((video, every))
            samples = {frame: (FaceBox(Box(200 + frame, 900, 120, 150), 0.8, "detected"),)
                       for frame in detect.sample_frames(120, every)}
            return detect.DetectorRun(1080, 1920, 120, samples, {"name": "fake"})

    cut_faces = stage.cut_faces
    monkeypatch.setattr(stage, "cut_faces",
                        lambda root, cut: cut_faces(root, cut, detector=Detector()))
    monkeypatch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "pass"})
    result = stage.assemble(project)
    final = project / result["out"]
    faces = load_faces(final.with_suffix(".faces.json"))
    assert len(calls) == 1 and calls[0][1] == 2
    assert faces.status == "measured" and faces.video.sha256 == receipt.sha256_file(final)
    assert (faces.video.frames, faces.video.width, faces.video.height, str(faces.video.fps)) == \
        (128, 1080, 1920, "30")
    frame = 34
    scale = 1 + 0.08 * math.sin(math.pi * (frame - 15) / 60) ** 2
    base_w, base_h = 1080, 1920
    if take_size is not None:
        factor = max(1080 / take_size[0], 1920 / take_size[1])
        base_w, base_h = (math.ceil(dimension * factor / 2) * 2 for dimension in take_size)
    rx, ry = math.trunc(base_w * scale / 2) * 2 / base_w, \
        math.trunc(base_h * scale / 2) * 2 / base_h
    original = Box(200 + frame, 900, 120, 150)
    expected = Box(1080 * 0.38 + (original.x - 1080 * 0.38) * rx,
                   1920 * 0.62 + (original.y - 1920 * 0.62) * ry,
                   original.w * rx, original.h * ry)
    assert faces.frames[frame][0].box == Box.from_dict(expected.to_dict())
    assert all(not faces.frames[frame] for frame in range(45, 60))
    assert faces.frames[120:] == (faces.frames[119],) * 8
    assert faces.sampling["frame_sources"] == [
        {"start_frame": 0, "end_frame": 15, "source": "cut"},
        {"start_frame": 15, "end_frame": 45, "source": "zoom"},
        {"start_frame": 45, "end_frame": 60, "source": "insert"},
        {"start_frame": 60, "end_frame": 75, "source": "zoom"},
        {"start_frame": 75, "end_frame": 120, "source": "cut"},
        {"start_frame": 120, "end_frame": 128, "source": "hold"},
    ]
    assert faces.detector["cut_sha256"] == receipt.sha256_file(project / "cut/cut.mp4")
    assert faces.gaps == ((45, 60),)


@pytest.mark.parametrize("check", ["faces.clearance", "captions.cards"])
def test_layout_violation_blocks_and_explicit_override_publishes(
    project: Path, monkeypatch: pytest.MonkeyPatch, check: str,
) -> None:
    finding = {"check": check, "severity": "fail", "time_s": 0.5,
               "message": 'Card "agent-tag" covers a face.',
               "evidence": {"element_id": "agent-tag", "target_id": "opaque-piece",
                            "element_piece": "opaque-piece"}}
    report = {"status": "fail", "findings": [finding, {
        "check": "picture.freeze", "severity": "warn", "time_s": 2, "message": "Held picture.",
    }]}
    monkeypatch.setattr(stage, "verify", lambda *args, **kwargs: report)
    with pytest.raises(stage.AssemblyError) as caught:
        stage.assemble(project)
    error = caught.value
    assert all(text in error.message for text in (
        "0:00.5", "agent-tag", "re-run captions", "re-run graphics", "--allow-violations",
    ))
    assert "0:00.5 agent-tag:" in error.message and "opaque-piece" not in error.message
    assert "re-run captions" in error.suggested_action
    assert read_receipt(project / "final")["status"] == "failed"
    assert not (project / "final/story.mp4").exists()
    result = stage.assemble(project, allow_violations=True)
    final = project / result["out"]
    assert final.is_file() and final.with_suffix(".faces.json").is_file()
    assert final.with_suffix(".layout.json").is_file()
    record = read_receipt(project / "final")
    assert record["status"] == "verified" and record["verify"]["status"] == "fail"
    assert next(item for item in record["asserts"] if item["id"] == "verify.status")["pass"]
    assert record["allowed_layout_violations"] == [
        {key: finding[key] for key in ("check", "time_s", "message")},
    ]
    assert stage.assemble(project, allow_violations=True)["noop"]
    with pytest.raises(stage.AssemblyError, match="agent-tag"):
        stage.assemble(project)


def test_allow_violations_keeps_other_verification_failures_blocking(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"status": "fail", "findings": [
        {"check": "faces.clearance", "severity": "fail", "time_s": 0.5,
         "message": 'Card "agent-tag" covers a face.'},
        {"check": "captions.timing", "severity": "fail", "message": "Caption arrives late."},
    ]}
    monkeypatch.setattr(stage, "verify", lambda *args, **kwargs: report)
    with pytest.raises(stage.AssemblyError, match="verify.status"):
        stage.assemble(project, allow_violations=True)
    record = read_receipt(project / "final")
    assert record["status"] == "failed" and "allowed_layout_violations" not in record
    assert not (project / "final/story.mp4").exists()


@pytest.mark.parametrize("explicit", [None, "none", "box"])
def test_final_contrast_halo_redo_and_warning_delivery(project, monkeypatch, explicit) -> None:
    from test_caption_contrast import contrast_document

    from yap_studio.checks.caption_contrast import check_caption_contrast, read_contrast_record
    from yap_studio.checks.model import Thresholds
    from yap_studio.stages import captions as caption_stage

    # Keep a verified predecessor while the halo pass changes only the caption layer.
    monkeypatch.setattr(stage, "verify", lambda *a, **k: {"status": "pass"})
    stage.assemble(project)
    phrase = json.loads((project / "captions/phrases.json").read_bytes())["phrases"][0]
    key = "caption:t1:0-t1:1"
    if explicit is None:
        prior = read_receipt(project / "final")
        prior["caption_contrast"].update(measured=True, weak=[{
            "key": key, "ids": phrase["ids"], "text": phrase["text"], "ratio": 1.2,
        }])
        write_json(project / "final/receipt.json", prior)
    previous = (project / "final/receipt.json").read_bytes()
    previous_media = receipt.sha256_file(project / "final/story.mp4")
    if explicit is not None:
        project_path = project / "project.json"
        data = json.loads(project_path.read_bytes())
        data["caption_backing"] = {key: explicit}
        project_path.write_text(json.dumps(data))
    calls = {"measure": 0, "captions": 0, "audio": 0}
    modes = []
    render_captions, render_audio = caption_stage.captions, stage.render_audio

    def refresh_captions(*args, **kwargs):
        calls["captions"] += 1
        assert (project / "final/receipt.json").read_bytes() == previous
        assert not (project / "final/receipt.pending.json").exists()
        return render_captions(*args, **kwargs)

    def audio(*args, **kwargs):
        calls["audio"] += 1
        return render_audio(*args, **kwargs)

    def measure(picture, layer, placement, layout, **kwargs):
        calls["measure"] += 1
        modes.append(kwargs.get("halo_ground", True))
        style = ProjectContext(project).render_style().data["captions"]
        assert kwargs["caption_style"] == {key: style[key] for key in (
            "text_color", "highlight_color", "highlight_text_color", "highlight_style",
        )}
        assert picture.name == "picture.mp4" and picture.is_file() and layer.is_file()
        assert calls["audio"] == 0
        assert not (project / "final/stale.json").exists()
        assert receipt.sha256_file(project / "final/story.mp4") == previous_media
        document = contrast_document()
        caption = document["captions"][0]
        caption.update(id="caption-0", text=phrase["text"], start_frame=3, end_frame=30,
                       frame=15, word=0)
        caption["words"] = [{"index": 0, "text": "First", "ratio": 1.2,
                             "frame": 15, "samples": 5}]
        return document

    def verify_final(media, **kwargs):
        document = read_contrast_record(Path(media))
        findings, coverage = check_caption_contrast(document, fps=30, thresholds=Thresholds())
        return {"status": "warn", "findings": [item.to_dict() for item in findings],
                "checks": {"captions.contrast": coverage}}

    monkeypatch.setattr(caption_stage, "captions", refresh_captions)
    monkeypatch.setattr(stage, "render_audio", audio)
    monkeypatch.setattr(stage, "measure_caption_contrast", measure)
    monkeypatch.setattr(stage, "verify", verify_final)
    result = stage.assemble(project, force=explicit is not None)
    automatic = explicit is None
    assert result["status"] == "verified" and result["caption_halo"] == {"added": int(automatic)}
    assert calls == {"measure": 2 if automatic else 1,
                     "captions": int(automatic or explicit == "box"), "audio": 1}
    assert modes == ([False, True] if automatic else [False])
    record = read_receipt(project / "final")
    assert record["status"] == "verified" and record["verify"]["status"] == "warn"
    assert record["caption_contrast"]["halo_added"] == ([key] if automatic else [])
    weak = record["caption_contrast"]["weak"][0]
    assert weak["key"] == key and weak["ids"] == phrase["ids"]
    assert weak["backing"] == ("shadow" if automatic else explicit if explicit == "box" else None)
    assert weak["auto"] == automatic
    assert read_contrast_record(project / result["out"])["captions"][0]["auto"] == automatic
    assert not (project / "final/receipt.pending.json").exists()
    halo_path = project / ".yap/caption-halo.json"
    if automatic:
        assert json.loads(halo_path.read_bytes()) == {
            "schema": "yap-studio/caption-halo@1",
            "phrases": {key: {"text": phrase["text"], "ratio": 1.2, "window":
                              caption_stage.caption_snapshot(ProjectContext(project)).windows[0]}}}
        assert "caption_backing" not in json.loads((project / "project.json").read_bytes())
    else:
        assert not halo_path.exists()
    calls.update(measure=0, captions=0, audio=0)
    modes.clear()
    previous_media = receipt.sha256_file(project / "final/story.mp4")
    again = stage.assemble(project, force=True)
    assert again["caption_halo"] == {"added": 0}
    assert calls == {"measure": 2 if automatic else 1, "captions": 0, "audio": 1}
    assert modes == ([False, True] if automatic else [False])
    if automatic:
        project_path = project / "project.json"
        data = json.loads(project_path.read_bytes())
        data["caption_backing"] = {key: "shadow"}
        project_path.write_text(json.dumps(data))
        calls.update(measure=0, captions=0, audio=0)
        modes.clear()
        previous_media = receipt.sha256_file(project / "final/story.mp4")
        assert stage.assemble(project)["caption_halo"] == {"added": 0}
        assert calls == {"measure": 2, "captions": 0, "audio": 1}
        assert modes == [False, True]
        weak = read_receipt(project / "final")["caption_contrast"]["weak"][0]
        assert weak["backing"] == "shadow" and weak["auto"] is False


@pytest.mark.parametrize("still_weak", [False, True])
def test_final_contrast_remeasures_plain_after_caption_color_change(
    project, monkeypatch, fake_layers, still_weak,
) -> None:
    from test_caption_contrast import contrast_document

    from yap_studio.stages import captions as caption_stage

    style = ProjectContext(project).render_style().data
    write_json(project / "style.json", style)
    document = json.loads((project / "project.json").read_bytes())
    document["style"] = "style.json"
    write_json(project / "project.json", document)
    measured = []
    ratio = 1.2
    phrase = json.loads((project / "captions/phrases.json").read_bytes())["phrases"][0]

    def measure(*args, **kwargs):
        layer = next(item for item in reversed(fake_layers) if "phrases" in item["variables"])
        measured.append(json.loads(layer["variables"]["phrases"])[0].get("backing"))
        document = contrast_document((ratio,))
        document["captions"][0].update(id="caption-0", text=phrase["text"],
                                       start_frame=3, end_frame=30,
                                       frame=15, word=0)
        document["captions"][0]["words"] = [
            {"index": 0, "text": "First", "ratio": ratio, "frame": 15, "samples": 5}]
        return document

    monkeypatch.setattr(stage, "measure_caption_contrast", measure)
    monkeypatch.setattr(stage, "verify", lambda *a, **k: {"status": "pass"})
    assert stage.assemble(project)["caption_halo"] == {"added": 1}
    assert measured == [None, "shadow"]
    previous = caption_stage.caption_halos(ProjectContext(project))
    measured.clear()
    style["captions"]["text_color"] = "#123456"
    write_json(project / "style.json", style)
    ratio = 1.2 if still_weak else 5.0
    result = stage.assemble(project)
    assert measured == ([None, "shadow"] if still_weak else [None])
    assert result["caption_halo"] == {"added": int(still_weak)}
    halos = caption_stage.caption_halos(ProjectContext(project))
    if still_weak:
        key = "caption:t1:0-t1:1"
        assert halos[key]["window"] != previous[key]["window"]
    else:
        assert halos == {}
    assert stage.assemble(project)["noop"] is True


@pytest.mark.parametrize("caption_id,expected_ids,expected_key", [
    ("caption-2", ["w0"], "caption:w0#2"), ("caption-3", None, None),
    ("caption-x", None, None),
])
def test_final_contrast_maps_phrase_slot_after_empty_slots(
    tmp_path, monkeypatch, caption_id, expected_ids, expected_key,
) -> None:
    from types import SimpleNamespace

    from test_caption_contrast import contrast_document

    retained = tmp_path / "snapshot"
    (retained / "captions").mkdir(parents=True)
    write_json(retained / "captions/layout.json", {"measured": True, "captions": []})
    write_json(retained / "captions/phrases.json", {"phrases": [
        {"text": "And I said, no.", "ids": ["w0"]}, {"text": "", "ids": []},
        {"text": "And I said, no.", "ids": ["w0"]},
    ]})
    document = contrast_document()
    document["captions"][0]["id"] = caption_id
    monkeypatch.setattr(stage, "measure_caption_contrast", lambda *a, **k: document)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"synthetic final")
    snapshot = SimpleNamespace(caption_layer={"path": "captions/captions.mov"})
    _, summary = stage._final_contrast(snapshot, retained, final)
    assert summary["weak"][0].get("ids") == expected_ids
    assert summary["weak"][0].get("key") == expected_key


def test_contrast_measurement_error_does_not_break_assembly(project, monkeypatch) -> None:
    from yap_studio.checks.caption_contrast import read_contrast_record

    def broken(*args, **kwargs):
        raise ValueError("incomplete contrast decode")

    monkeypatch.setattr(stage, "measure_caption_contrast", broken)
    monkeypatch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "pass"})
    result = stage.assemble(project)
    record = read_contrast_record(project / result["out"])
    assert not record["measured"] and record["reason"] == "incomplete contrast decode"
    summary = read_receipt(project / "final")["caption_contrast"]
    assert summary == {"measured": False, "reason": record["reason"], "captions": 0,
                       "weak": [], "halo_added": []}


def test_final_verification_projection_tracks_platform(project: Path) -> None:
    context = ProjectContext(project)
    before = stage.assembly_snapshot(context).inputs["project:verification"]
    assert context.project.platform == "none"
    assert before == projection_sha256({"asr": context.project.asr})
    data = json.loads((project / "project.json").read_bytes())
    data["platform"] = "reels"
    write_json(project / "project.json", data)
    previous = read_receipt(project / "cut")
    with StageRun(project, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish(extra={"plan": previous["plan"]})
    after = stage.assembly_snapshot(ProjectContext(project)).inputs["project:verification"]
    assert after == projection_sha256({"asr": context.project.asr, "platform": "reels"})
    assert after != before


@pytest.mark.parametrize("project,platform", [
    ({"platform": "reels"}, "reels"), ({"platform": "none"}, "none"),
], indirect=["project"])
def test_final_platform_coverage(project, platform) -> None:
    stage.assemble(project)
    report = read_receipt(project / "final")["verify"]
    assert report["checks"]["platform.duration"]["ran"] is (platform != "none")
    if platform == "none":
        assert report["checks"]["platform.duration"]["reason"] == "no platform chosen"


@pytest.mark.parametrize("project", [{"platform": "reels"}], indirect=True)
def test_final_platform_failure_keeps_its_severity(project, monkeypatch) -> None:
    import importlib

    from yap_studio.checks.platform import load_presets

    presets = load_presets()
    presets["reels"]["duration"]["max_s"] = 1
    monkeypatch.setattr(importlib.import_module("yap_studio.verify"), "load_presets",
                        lambda: presets)
    with pytest.raises(stage.AssemblyError, match="verify.status"):
        stage.assemble(project)
    report = read_receipt(project / "final")["verify"]
    assert report["checks"]["platform.duration"]["status"] == "fail"


@pytest.mark.parametrize("faces", [True, False])
def test_setup_taste_record_precedes_verification(project, monkeypatch, faces) -> None:
    import importlib

    from yap_studio.style.answers import ANSWERS_SCHEMA, taste_rules
    from yap_studio.taste import read_taste_record
    from yap_studio.taste.store import approve_proposal

    for rule in taste_rules({"schema": ANSWERS_SCHEMA,
                            "never": {"cover_face": faces, "emoji": True}}):
        approve_proposal(rule)
    monkeypatch.setattr(importlib.import_module("yap_studio.taste.check"), "_face_clearance",
                        lambda _media: ([], {"ran": True, "reason": None}))
    original = stage.verify

    def verify_with_record(media, **kwargs):
        record = read_taste_record(Path(media))
        assert record is not None
        assert len(record.get("enforced", [])) == (3 if faces else 0)
        assert record["skipped"][0]["reason"] == "followed_by_agent"
        return original(media, **kwargs)

    monkeypatch.setattr(stage, "verify", verify_with_record)
    stage.assemble(project)
    assert read_receipt(project / "final")["verify"]["checks"]["taste.rules"]["ran"]


@pytest.mark.parametrize("covered,allow", [(True, False), (False, False), (True, True)])
def test_measured_face_inside_drawn_card_controls_delivery(
    project: Path, monkeypatch: pytest.MonkeyPatch, covered: bool, allow: bool,
) -> None:
    set_elements(project, graphic())
    graphics(project)
    layer = read_receipt(project / "graphics")["elements"][0]
    card_box = Box(layer["x"], layer["y"], layer["width"], layer["height"])
    face_box = Box(card_box.x + 50, card_box.y + 50, 40, 30) if covered else \
        Box(800, 1300, 40, 30)
    if covered:
        head = Box(face_box.x - DEFAULT_HAIR.sides * face_box.w,
                   face_box.y - DEFAULT_HAIR.above * face_box.h,
                   face_box.w * (1 + 2 * DEFAULT_HAIR.sides),
                   face_box.h * (1 + DEFAULT_HAIR.above))
        assert card_box.x < head.x < head.x1 < card_box.x1
        assert card_box.y < head.y < head.y1 < card_box.y1
    calls = []

    def measured_cut(root: Path, cut: Path) -> FacesData:
        assert root == project and cut != project / "cut/cut.mp4"
        calls.append(cut)
        start, end = layer["start_frame"], layer["start_frame"] + layer["frames"]
        frames = tuple((FaceBox(face_box, 0.9, "detected"),) if start <= frame < end else ()
                       for frame in range(120))
        return FacesData(
            "measured", None,
            VideoIdentity(cut.name, receipt.sha256_file(cut), 120, Fraction(30), 1080, 1920),
            {"name": "fake"}, {}, frames, tuple(detect.find_gaps(frames)),
        )

    monkeypatch.setattr(stage, "cut_faces", measured_cut)
    if covered and not allow:
        with pytest.raises(stage.AssemblyError) as caught:
            stage.assemble(project)
        assert "0:00.2" in caught.value.message and layer["id"] in caught.value.message
        record = read_receipt(project / "final")
        assert record["status"] == "failed"
        assert not (project / "final/story.mp4").exists()
    else:
        result = stage.assemble(project, allow_violations=allow)
        final = project / result["out"]
        assert final.is_file() and final.with_suffix(".faces.json").is_file()
        record = read_receipt(project / "final")
        assert record["status"] == "verified"
        assert record["verify"]["checks"]["faces.clearance"]["ran"] is True
        if covered:
            pixels = synth.picture_pixels(final, layer["start_frame"] + 3,
                                          box=(int(face_box.x), int(face_box.y), 20, 20))
            np.testing.assert_allclose(pixels.mean(axis=(0, 1)), (230, 90, 40), atol=5)
            assert record["verify"]["status"] == "fail"
            assert record["allowed_layout_violations"] == [{
                "check": "faces.clearance", "time_s": layer["start_frame"] / 30,
                "message": 'Card "agent-tag" is over the speaker\'s face '
                           'from 0:00.2 to 0:01.2.',
            }]
        else:
            assert record["verify"]["checks"]["faces.clearance"]["status"] == "pass"
            assert "allowed_layout_violations" not in record
    assert len(calls) == 1
    failures = [row for row in record["verify"]["findings"] if row["severity"] == "fail"]
    assert {row["check"] for row in failures} == ({"faces.clearance"} if covered else set())
    if covered:
        assert len(failures) == 1 and failures[0]["evidence"]["element_id"] == layer["id"]


def test_behind_graphic_uses_retained_cut_cache_and_records_hidden_frames(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_render = hyperframes.render_layer

    def render_with_ink(composition_dir: Path, out: Path, **kwargs: Any):
        result = original_render(composition_dir, out, **kwargs)
        if "text" in kwargs["variables"]:
            temporary = out.with_name(f"{out.stem}-ink.mov")
            synth.picture_source(
                temporary, f"color=0xe65a28:s={result.width}x{result.height}:r=30,"
                "format=rgba,drawbox=x=40:y=40:w=120:h=8:c=black:t=fill,"
                "drawbox=x=40:y=64:w=120:h=8:c=black:t=fill",
                result.frames, alpha=True,
            )
            temporary.replace(out)
        return result

    monkeypatch.setattr(hyperframes, "render_layer", render_with_ink)
    element = graphic(layer="behind")
    set_elements(project, element, title(start={"time": 1.5}, hold_s=1))
    _behind_graphics_contract(project, monkeypatch)
    graphics(project)
    placement = next(item for item in read_receipt(project / "graphics")["elements"]
                     if item["id"] == element["id"])
    calls = []
    digest = receipt.sha256_file(project / "cut/cut.mp4")

    def make(source: Path, folder: Path, *, joins=()):
        assert source != project / "cut/cut.mp4"
        assert receipt.sha256_file(source) == digest
        calls.append(source)
        # Cover both ink rows, with most of the opaque panel still visible.
        return synthetic_matte(source, folder,
                               box=(placement["x"] + 40, placement["y"] + 40, 120, 32),
                               active_windows=((6, 21),))

    monkeypatch.setattr(matte_cache, "make_matte", make)
    result = stage.assemble(project)
    record = read_receipt(project / "final")
    expected = [{"id": element["id"], "start_frame": 6, "frames": 30, "hidden_frames": 15,
                 "hidden_share": 0.5, "flag": True}]
    assert record["behind_layers"] == record["picture"]["behind_layers"] == expected
    projection = {"hidden_glyph_share": stage.HIDDEN_GLYPH_SHARE,
                  "hidden_time_share": stage.HIDDEN_TIME_SHARE,
                  "overlap": "skip_zoom_and_insert", "cut_matte": 2, "measurement": 2}
    assert record["inputs"]["picture:behind"] == projection_sha256(projection)
    assert record["inputs"]["picture:behind"] != projection_sha256({**projection, "cut_matte": 1})
    assert record["status"] == "verified" and final_stage(project)["state"] == "done"
    assert ffmpeg.probe(project / result["out"]).video_frames == 128
    assert (project / ".yap/cache/matte" / digest / "matte.json").is_file()
    assert len(calls) == 1
    assert stage.assemble(project)["noop"] and len(calls) == 1


@pytest.mark.parametrize("look", [False, True])
def test_background_final_layout_checks_and_verification_mask(
    project: Path, monkeypatch: pytest.MonkeyPatch, look: bool,
) -> None:
    image = overlay_image(project / "page.png", "color=lime:s=320x180")
    fragment = project / "background-fragment.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-page", "kind": "background",
        "start": {"word": "t1:0", "text": "First"},
        "end": {"word": "t1:1", "edge": "end"},
        "media": {"path": image.name, "sha256": receipt.sha256_file(image)},
        "fit": "cover",
        **({"layout": "bottom", "blend": "wrap", "enter": "wipe"} if look else {}),
    }]})
    timeline = add_timeline(project, fragment, lane="broll").require_valid()
    assert (timeline.elements[0].start_frame, timeline.elements[0].end_frame) == (3, 30)
    captions(project)
    graphics(project)
    monkeypatch.setattr(matte_cache, "make_matte",
                        lambda source, folder, *, joins=(): synthetic_matte(
        source, folder, box=(80, 100, 100, 100), joins=joins,
    ))
    result = stage.assemble(project)
    final = project / result["out"]
    record = read_receipt(project / "final")
    if look:
        assert record["green_screen"] == record["picture"]["green_screen"]
        assert record["green_screen"][0]["person_box_px"] == [189, 672, 702, 1248]
        assert "picture:green_screen" in record["inputs"]
        from yap_studio.stages import green_screen
        monkeypatch.setattr(green_screen, "WRAP_STRENGTH", .2)
        changed = stage.assembly_snapshot(ProjectContext(project)).inputs
        assert changed["picture:green_screen"] != record["inputs"]["picture:green_screen"]
    else:
        assert "green_screen" not in record and "green_screen" not in record["picture"]
        assert "picture:green_screen" not in record["inputs"]
    layout = read_layout_record(final, width=1080, height=1920)
    assert record["inputs"]["picture:behind"] == projection_sha256({
        "hidden_glyph_share": stage.HIDDEN_GLYPH_SHARE,
        "hidden_time_share": stage.HIDDEN_TIME_SHARE,
        "overlap": "skip_zoom_and_insert", "cut_matte": 2, "measurement": 2,
    })
    assert layout is not None
    (element,) = [item for item in layout.layout.elements if item.kind == "background"]
    assert (element.id, element.start_frame, element.end_frame) == ("broll-page", 3, 30)
    assert layout.labels[element.id] == "page.png"
    assert not check_caption_cards(layout, fps=30)
    assert not any(item["check"] in {"faces.clearance", "captions.cards"}
                   and item["severity"] == "fail" for item in record["verify"]["findings"])
    edl = json.loads((project / "final/verify-edl.json").read_text())
    assert {"start_in_output": 0.1, "duration": 0.9, "region": [0, 0, 1080, 1920]} \
        in edl["overlays"]
    for frame, inside in ((2, False), (3, True), (29, True), (30, False)):
        pixel = synth.picture_pixels(final, frame, box=(210, 110, 20, 20)).mean(axis=(0, 1))
        original = synth.picture_pixels(project / "cut/cut.mp4", frame,
                                        box=(210, 110, 20, 20)).mean(axis=(0, 1))
        if not look or frame != 3:
            np.testing.assert_allclose(pixel, (0, 255, 0) if inside else original, atol=10)


def test_titles_never_trigger_matte_or_receipt_measurements(project: Path) -> None:
    set_elements(project, title(hold_s=1))
    graphics(project)
    stage.assemble(project)
    record = read_receipt(project / "final")
    assert "behind_layers" not in record and "behind_layers" not in record["picture"]
    assert "picture:behind" not in record["inputs"]
    assert not (project / ".yap/cache/matte").exists()


def test_bold_band_verification_regions_stay_inside_frame(project: Path) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write_json(project / "project.json", data)
    write_json(project / "style.json", {
        "schema_version": 1, "name": "Sample style", "hook": {"style": "bold_band"},
    })
    set_elements(project, title(hold_s=1))
    captions(project)
    graphics(project)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    placement = snapshot.title_layers[0]
    assert placement["x"] == -40 and placement["width"] == 1160
    edl = stage.verification_edl(snapshot, base=project / "final")
    for overlay in edl["overlays"]:
        x, y, width, height = overlay["region"]
        assert 0 <= x < x + width <= 1080 and 0 <= y < y + height <= 1920
    assert edl["overlays"][0]["region"] == [0, placement["y"], 1080, placement["height"]]
    target = project / "check-edl.json"
    write_json(target, edl)
    parsed = load_edit_list(target)
    assert len(parsed.picture_regions) == len(edl["overlays"]) and not parsed.picture_masks
    placements = [
        {**placement, "x": -20, "y": -30, "width": 40, "height": 60},
        {**placement, "x": 1080, "y": 0},
    ]
    clipped = stage.verification_edl(replace(
        snapshot, title_layers=placements,
        caption_layer={**snapshot.caption_layer, "x": 1070, "y": 1910,
                       "width": 40, "height": 60},
    ), base=project / "final")
    assert [item["region"] for item in clipped["overlays"]] == [
        [0, 0, 20, 30], [1070, 1910, 10, 10],
    ]


def test_final_exact_samples_receipt_hold_and_preserved_master(project: Path) -> None:
    result = stage.assemble(project)
    out = project / result["out"]
    facts = ffmpeg.probe(out)
    assert facts.video_frames == 128
    assert len(decode(out, [], channels=1)) == facts.video_frames * 1600
    assert abs(measure(out, []).integrated_lufs + 16) <= 0.10001
    record = read_receipt(project / "final")
    assert record["stage"] == "assemble" and record["status"] == "verified"
    assert all(check["pass"] for check in record["asserts"])
    assert record["audio"]["policy"] == {"mode": "preserve"}
    assert record["audio"]["target"] is None
    assert record["audio"]["input_voice"]["integrated_lufs"] == pytest.approx(-16, abs=0.1)
    original = decode(project / "cut/voice.wav", [], channels=1)[:, 0]
    finished = decode(out, [], channels=1)[:, 0]
    correlations = [np.dot(original[100:10000], finished[100 + lag:10000 + lag])
                    for lag in range(-20, 21)]
    assert np.argmax(correlations) - 20 == 0
    parsed = load_edit_list(project / "final/verify-edl.json")
    assert parsed.holds[0].start_frame == 120 and parsed.holds[0].end_frame == 128
    assert parsed.picture_regions and not parsed.picture_masks
    assert not [finding for finding in record["verify"]["findings"]
                if finding["check"] in {"plan.holds", "picture.freeze"}
                and finding["severity"] in {"warn", "fail"}]
    for item in record["outputs"]:
        assert receipt.sha256_file(project / item["path"]) == item["sha256"]
    commands = record["commands"]
    mux = next(command for command in commands if "-c" in command)
    assert mux[mux.index("-c") + 1] == "copy" and "-shortest" not in mux
    assert sum("aac" in command for command in commands) == 1
    assert any("ebur128=peak=true" in command for command in commands)
    assert any("-show_streams" in command for command in commands)
    assert final_stage(project)["state"] == "done"
    assert stage.assemble(project)["noop"]


@pytest.mark.render
def test_photos_and_held_diagram_verify_without_motion_files(project: Path) -> None:
    write_edit(project, [time_segment(f"s{index}", "t1", start, end)
                         for index, (start, end) in enumerate(((0, 1), (1, 2), (2, 4)), 1)],
               end_hold_s=0)
    spans = project / "new-spans.json"
    write_json(spans, {"sources": {"t1": "../takes/t1.mov"}, "fps": "30/1",
                       "ranges": [{"source": "t1", "start": start, "end": end}
                                  for start, end in ((0, 1), (1, 2), (2, 4))]})
    previous = read_receipt(project / "cut")
    with StageRun(project, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(project)).items():
            run.record_inputs(name, digest)
        run.outputs.update({row["path"]: row for row in previous["outputs"]
                            if row["path"] != "cut/spans.json"})
        run.replace_output(project / "cut/spans.json", spans)
        run.finish(extra={"plan": {"total_frames": 120}})
    photos = [textured_image(project / f"photo-{index}.ppm", seed=index)
              for index in (1, 2)]
    clip = project / "diagram.mov"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=s=270x480:r=30", "-vf",
                "trim=end_frame=20,tpad=stop_mode=clone:stop_duration=3,trim=end_frame=90,"
                "setpts=PTS-STARTPTS", "-frames:v", "90", "-c:v", "libx264",
                "-crf", "0", "-preset", "ultrafast", str(clip)], timeout_s=120)
    lane = [{"id": f"broll-photo-{index}", "kind": "insert", "lane": "broll",
             "start": {"time": start}, "end": {"time": end},
             "media": {"path": photo.name, "sha256": receipt.sha256_file(photo)}}
            for index, (photo, start, end) in enumerate(zip(
                photos, (0, 1), (1, 2), strict=True), 1)]
    lane.append({"id": "broll-diagram", "kind": "insert", "lane": "broll",
                 "start": {"time": 2}, "end": {"time": 4},
                 "motion": "still",
                 "media": {"path": clip.name, "sha256": receipt.sha256_file(clip)}})
    write_json(project / "timeline/broll.json", {"schema": "yap-studio/timeline@1",
                                                  "words_sha256": {},
                                                  "elements": lane})
    captions(project)
    graphics(project)
    music(project)
    result = stage.assemble(project)
    assert result["status"] == "verified"
    report = read_receipt(project / "final")["verify"]
    assert report["status"] != "fail"
    assert not [f for f in report["findings"] if f["check"] == "picture.freeze"
                and f["severity"] == "fail"]
    assert any(f["check"] == "picture.freeze" and f["severity"] == "info"
               and f["evidence"]["declared_still"]["id"] == "broll-diagram"
               for f in report["findings"])
    edl = json.loads((project / "final/verify-edl.json").read_bytes())
    assert {"id": "broll-diagram", "start_frame": 60, "end_frame": 120} in edl["stills"]


def test_pilot_receipt_marking_noop_and_final_untouched(project: Path) -> None:
    before = (project / "final/receipt.json").read_bytes() if (
        project / "final/receipt.json").exists() else None
    result = stage.assemble(project, frames=(30, 89))
    media = project / result["out"]
    record = read_receipt(project / "pilot")
    assert result["pilot"] is True and result["frames"] == [30, 89]
    assert ffmpeg.probe(media).video_frames == 60
    assert len(decode(media, [], channels=1)) == 96_000
    assert record["stage"] == "pilot" and record["pilot"] == {
        "frames": [30, 89], "frame_count": 60, "samples": 96_000,
        "of_frames": 128, "final": False,
    }
    assert {phase["name"] for phase in record["timings"]["phases"]} >= {
        "assembling picture", "mixing audio", "checking pilot audio",
    }
    assert stage.pilot_record(media) == {"frames": [30, 89], "of_frames": 128}
    assert stage.assemble(project, frames=(30, 89))["noop"] is True
    second = stage.assemble(project, frames=(90, 119))
    assert media.is_file() and (project / second["out"]).is_file()
    assert {item["path"] for item in read_receipt(project / "pilot")["outputs"]} == {
        result["out"], second["out"],
    }
    if before is None:
        assert not (project / "final/receipt.json").exists()
    else:
        assert (project / "final/receipt.json").read_bytes() == before


def test_pilot_refuses_caption_and_graphic_mid_window(project: Path) -> None:
    layout = _measured_caption_layout(project)
    phrase = layout["captions"][0]
    end = phrase["end_frame"] - 2
    with pytest.raises(stage.PilotRangeError) as caught:
        stage.assemble(project, frames=(0, end))
    assert caught.value.suggested_end == phrase["end_frame"] - 1
    assert f"--frames 0-{phrase['end_frame'] - 1}" in caught.value.suggested_action
    assert not list((project / "pilot").glob("*.mp4"))
    set_elements(project, graphic(start={"time": 1.2}, hold_s=1))
    graphics(project)
    placement = read_receipt(project / "graphics")["elements"][0]
    end = placement["start_frame"] + placement["frames"] - 2
    with pytest.raises(stage.PilotRangeError) as caught:
        stage.assemble(project, frames=(30, end))
    assert caught.value.suggested_end == placement["start_frame"] + placement["frames"] - 1
    assert "agent-tag" in str(caught.value)


def test_pilot_unmeasured_layout_uses_verified_phrase_windows(project: Path) -> None:
    layout = _measured_caption_layout(project)
    phrase = json.loads((project / "captions/phrases.json").read_bytes())["phrases"][0]
    layout["captions"][0]["start_frame"] = math.floor(round(phrase["start"] * 30, 6)) + 1
    layout["captions"][0]["end_frame"] = math.ceil(round(phrase["end"] * 30, 6))
    write_json(project / "captions/layout.json", layout)
    record = read_receipt(project / "captions")
    next(item for item in record["outputs"] if item["path"] == "captions/layout.json")[
        "sha256"] = receipt.sha256_file(project / "captions/layout.json")
    write_json(project / "captions/receipt.json", record)
    with pytest.raises(stage.PilotRangeError) as measured:
        stage.assemble(project, frames=(0, 20))
    layout.update(measured=False, reason="Layer checks are unavailable.", captions=[])
    write_json(project / "captions/layout.json", layout)
    record = read_receipt(project / "captions")
    next(item for item in record["outputs"] if item["path"] == "captions/layout.json")[
        "sha256"] = receipt.sha256_file(project / "captions/layout.json")
    write_json(project / "captions/receipt.json", record)
    with pytest.raises(stage.PilotRangeError) as unmeasured:
        stage.assemble(project, frames=(0, 20))
    assert unmeasured.value.suggested_end == measured.value.suggested_end
    assert not list((project / "pilot").glob("*.mp4"))


@pytest.mark.parametrize("missing", ["file", "receipt_output", "hash"])
def test_pilot_requires_verified_phrase_file_when_layout_unmeasured(
    project: Path, missing: str,
) -> None:
    assert json.loads((project / "captions/layout.json").read_bytes())["measured"] is False
    record = read_receipt(project / "captions")
    if missing == "file":
        (project / "captions/phrases.json").unlink()
    elif missing == "hash":
        (project / "captions/phrases.json").write_bytes(b"changed")
    else:
        record["outputs"] = [item for item in record["outputs"]
                             if item["path"] != "captions/phrases.json"]
        write_json(project / "captions/receipt.json", record)
    if missing == "file":
        result = stage.assemble(project, frames=(30, 89))
        assert result["status"] == "verified" and "captions" in result["refreshed"]
    else:
        with pytest.raises(stage.AssemblyError, match="captions/phrases.json"):
            stage.assemble(project, frames=(30, 89))


def test_pilot_reuses_earlier_range_and_invalidates_changed_inputs(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = stage.assemble(project, frames=(30, 89))
    stage.assemble(project, frames=(90, 119))
    ranges = read_receipt(project / "pilot")["ranges"]
    assert set(ranges) == {"30-89", "90-119"}
    assert all(len(item["inputs_sha256"]) == 64 for item in ranges.values())
    with monkeypatch.context() as patch:
        patch.setattr(stage, "render_picture", lambda *args, **kwargs: pytest.fail(
            "An earlier unchanged pilot range must not render again"))
        assert stage.assemble(project, frames=(30, 89))["noop"] is True
    data = json.loads((project / "project.json").read_bytes())
    data["audio"] = {"mode": "loudnorm", "integrated_lufs": -14,
                     "true_peak_dbtp": -1.7, "lra_lu": 11}
    write_json(project / "project.json", data)
    assert stage.assemble(project, frames=(30, 89))["noop"] is False
    (project / first["out"]).unlink()
    assert stage.assemble(project, frames=(30, 89))["noop"] is False
    assert stage.assemble(project, frames=(30, 89), force=True)["noop"] is False


def test_pilot_range_cli_refusals(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _measured_caption_layout(project)
    assert cli.main(["assemble", str(project), "--frames", "0-999"]) == 3
    assert cli.main(["assemble", str(project), "--frames", "0-20"]) == 3
    assert "--frames 0-" in capsys.readouterr().err
    assert cli.main(["assemble", str(project), "--frames", "5-2"]) == 2
    assert cli.main(["assemble", str(project), "--frames", "x"]) == 2


def test_pilot_cli_success_text_and_json(project: Path,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["assemble", str(project), "--frames", "30-89"]) == 0
    output = capsys.readouterr().out
    assert "Verified pilot (frames 30-89, not a final): pilot/" in output
    assert "Receipt: pilot/receipt.json" in output
    assert cli.main(["assemble", str(project), "--frames", "30-89", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pilot"] is True and result["noop"] is True


@pytest.mark.parametrize("crf", [0, 16])
def test_pilot_picture_matches_full_frame_clock(
    project: Path, monkeypatch: pytest.MonkeyPatch, crf: int,
) -> None:
    set_elements(project, graphic())
    _measured_caption_layout(project)
    graphics(project)
    monkeypatch.setattr(assemble_picture, "CRF", crf)
    full = project / stage.assemble(project)["out"]
    final_receipt = (project / "final/receipt.json").read_bytes()
    pilot = project / stage.assemble(project, frames=(10, 50))["out"]
    assert (project / "final/receipt.json").read_bytes() == final_receipt
    if crf == 0:
        assert synth.picture_hashes(pilot) == synth.picture_hashes(full, first=10)[:41]
    else:
        def small_frames(path: Path, first: int, last: int) -> np.ndarray:
            decoded = ffmpeg.run([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                "-vf", f"trim=start_frame={first}:end_frame={last},scale=108:192,"
                "format=rgb24", "-fps_mode", "passthrough", "-f", "rawvideo", "-",
            ], timeout_s=120, binary=True).stdout
            return np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 192, 108, 3).astype(
                np.int16)

        original = small_frames(full, 10, 52)
        sliced = small_frames(pilot, 0, 41)
        assert len(sliced) == 41
        same = np.abs(sliced - original[:41]).mean(axis=(1, 2, 3))
        shifted = np.abs(sliced - original[1:42]).mean(axis=(1, 2, 3))
        assert np.all(same < shifted)


def test_pilot_audio_pcm_is_exact_full_mix_slice(project: Path) -> None:
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    retained = project / "retained-pcm"
    timeline = stage._retained(snapshot, retained)
    context = ProjectContext(project)
    options = {"timeline": timeline, "policy": context.project.audio,
               "music": context.render_style().data["music"],
               "voice": context.render_style().data["voice"]}
    full_work = project / "full-pcm"
    pilot_work = project / "pilot-pcm"
    render_audio(retained, full_work / "audio.m4a", work_dir=full_work, **options)
    result = render_audio(retained, pilot_work / "audio.m4a", work_dir=pilot_work,
                          frames=(30, 89), **options)
    complete = decode(full_work / "mix.wav", [], channels=1)
    sliced = np.fromfile(pilot_work / "pilot.f32le", dtype="<f4").reshape(-1, 1)
    assert result.samples == 96_000
    assert np.array_equal(sliced, complete[30 * 1600:90 * 1600])
    assert len(decode(pilot_work / "audio.m4a", [], channels=1)) == 96_000


@pytest.mark.render
@pytest.mark.parametrize("project", [{"cut_frames": 4588, "hold": 8}], indirect=True)
def test_pilot_4596_frame_acceptance(project: Path) -> None:
    result = stage.assemble(project, frames=(0, 396))
    media = project / result["out"]
    assert ffmpeg.probe(media).video_frames == 397
    assert len(decode(media, [], channels=1)) == 635_200
    assert read_receipt(project / "pilot")["pilot"]["of_frames"] == 4596


def test_policy_change_stales_final_and_loudnorm_matches_default(project: Path) -> None:
    stage.assemble(project)
    data = json.loads((project / "project.json").read_bytes())
    del data["audio"]
    write_json(project / "project.json", data)
    assert final_stage(project)["state"] == "stale"
    assert "project:audio" in final_stage(project)["reason"]
    stage.assemble(project)
    record = read_receipt(project / "final")
    assert record["audio"]["target"]["integrated_lufs"] == -14
    assert record["audio"]["output"]["integrated_lufs"] == pytest.approx(-14, abs=0.5)


def test_face_margin_stales_captions_and_final_without_graphics(project: Path) -> None:
    context = ProjectContext(project)
    assert not context.resolved_timeline().elements
    style = context.render_style().data
    write_json(project / "style.json", style)
    document = json.loads((project / "project.json").read_bytes())
    document["style"] = "style.json"
    write_json(project / "project.json", document)
    result = stage.assemble(project)
    before = read_receipt(project / "final")
    style["graphics"]["face_clearance_px"] = 48
    write_json(project / "style.json", style)
    stages = {item["name"]: item for item in status(project)["stages"]}
    assert stages["captions"]["state"] == "stale"
    assert "style:face_clearance" in stages["captions"]["reason"]
    assert stages["graphics"]["state"] == "done"
    captions(project)
    after = stage.assembly_snapshot(ProjectContext(project)).inputs
    assert after["style:face_clearance"] == projection_sha256(48)
    assert after["style:face_clearance"] != before["inputs"]["style:face_clearance"]
    assert final_stage(project)["state"] == "stale"
    assert "style:face_clearance" in final_stage(project)["reason"]
    assert not stage.assemble(project)["noop"]
    layout = json.loads((project / result["out"]).with_suffix(".layout.json").read_bytes())
    assert layout["clearance_px"] == 48
    assert final_stage(project)["state"] == "done"


def test_caption_gap_changes_final_snapshot_only(project: Path) -> None:
    context = ProjectContext(project)
    before = stage.assembly_snapshot(context).inputs
    style = context.render_style().data
    style["graphics"]["caption_gap_px"] = 60
    write_json(project / "style.json", style)
    document = json.loads((project / "project.json").read_bytes())
    document["style"] = "style.json"
    write_json(project / "project.json", document)
    after = stage.assembly_snapshot(ProjectContext(project)).inputs
    assert before["style:caption_gap"] == projection_sha256(100)
    assert after["style:caption_gap"] == projection_sha256(60)
    assert {key for key in before if before[key] != after[key]} == {"style:caption_gap"}


def test_default_loudnorm_keeps_variable_speech_verifiable_at_cuts(tmp_path: Path) -> None:
    """A delivery gain must not make unchanged, uneven speech fail edit-list verification."""
    sentences = [
        "Bright garden flowers open beside the sunny door.",
        "Quiet evening rain settles softly over every leaf.",
        "Morning voices carry through the open kitchen window.",
    ]
    source, spans = synth.speech_source(tmp_path / "speech.mov", sentences)
    second_start = spans[len(sentences[0].split())].start
    third_start = spans[len(sentences[0].split()) + len(sentences[1].split())].start
    variable = tmp_path / "variable-speech.mov"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-filter:a",
        f"volume=0.125:enable='between(t,{second_start:.6f},{third_start:.6f})'",
        "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", str(variable),
    ], timeout_s=60)
    root = make_project(tmp_path / "project", [variable], word_rows=[[
        (word.text, word.start, word.end) for word in spans
    ]])
    data = json.loads((root / "project.json").read_bytes())
    data["asr"] = {"engine": "none"}
    data["music"] = {"mode": "none"}
    write_json(root / "project.json", data)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({row["path"]: row for row in previous["outputs"]})
        run.finish()
    first_end = spans[len(sentences[0].split()) - 1].end + 0.05
    second_end = spans[len(sentences[0].split()) + len(sentences[1].split()) - 1].end + 0.05
    write_edit(root, [
        time_segment("first", "t1", 0.4, first_end),
        time_segment("second", "t1", second_start - 0.05, second_end),
        time_segment("third", "t1", third_start - 0.05, spans[-1].end + 0.1),
    ], end_hold_s=0.2)
    from yap_studio.stages.cut import cut

    cut(root)
    captions(root)
    graphics(root)
    music(root)
    stage.assemble(root)
    record = read_receipt(root / "final")
    assert record["status"] == "verified"
    assert record["audio"]["normalization"]["mode"] == "constant_gain_limiter"
    assert record["verify"]["checks"]["join.words"]["status"] == "pass"
    assert record["audio"]["output"]["true_peak_dbtp"] <= -1.7


@pytest.mark.parametrize("project", [1, 17, 45], indirect=True)
def test_music_final_exact_samples_and_declared_hold(project: Path) -> None:
    """Short and default holds stay sample-exact with a partial last AAC packet."""
    time = np.arange(8 * 48000) / 48000
    synth.float_audio(project / "track.wav", 0.1 * np.sin(2 * np.pi * 880 * time))
    data = json.loads((project / "project.json").read_bytes())
    data["music"] = {"mode": "file", "file": "track.wav"}
    del data["audio"]
    write_json(project / "project.json", data)
    music(project)
    stage.assemble(project)
    out = project / "final/story.mp4"
    hold = round(json.loads((project / "edit.json").read_bytes())["end_hold_s"] * 30)
    frames = 120 + hold
    assert ffmpeg.probe(out).video_frames == frames
    assert round(ffmpeg.probe(out).audio_duration_s * 48000) == frames * 1600
    record = read_receipt(project / "final")
    counts = next(item for item in record["asserts"] if item["id"] == "audio.samples")
    assert counts["pass"] and counts["terminal_padding"] == 0
    assert len(decode(out, [], channels=2)) == frames * 1600
    assert record["verify"]["checks"]["plan.holds"]["status"] == "pass"
    assert not [item for item in record["verify"]["findings"]
                if item["check"] in {"plan.holds", "picture.freeze"}
                and item["severity"] in {"warn", "fail"}]


def test_verify_failure_has_failed_receipt_and_no_verified_final(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "fail"})
    with pytest.raises(stage.AssemblyError, match="verify.status"):
        stage.assemble(project)
    record = read_receipt(project / "final")
    assert record["status"] == "failed"
    assert not (project / "final/story.mp4").exists()
    assert cli.main(["assemble", str(project), "--json"]) == 1


def test_cut_change_assemble_refreshes_stale_stages(project: Path) -> None:
    first = stage.assemble(project)
    before = receipt.sha256_file(project / first["out"])
    write_edit(project, [time_segment("s1", "t1", 0, 3)], end_hold_s=8 / 30)
    assert cut(project)["status"] == "verified"
    result = stage.assemble(project)
    assert result["status"] == "verified" and not result["noop"]
    assert result["refreshed"] == ["captions", "music"]
    assert receipt.sha256_file(project / result["out"]) != before
    assert read_receipt(project / "final")["status"] == "verified"


def test_missing_caption_output_is_refreshed(project: Path) -> None:
    stage.assemble(project)
    (project / "captions/layout.json").unlink()
    result = stage.assemble(project)
    assert result["status"] == "verified"
    assert "captions" in result["refreshed"]
    assert (project / "captions/layout.json").is_file()
    assert read_receipt(project / "final")["status"] == "verified"


def test_stale_file_music_refreshes_and_auto_refuses(project: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    time = np.arange(8 * 48000) / 48000
    synth.float_audio(project / "track.wav", 0.1 * np.sin(2 * np.pi * 440 * time))
    data = json.loads((project / "project.json").read_bytes())
    data["music"] = {"mode": "file", "file": "track.wav"}
    write_json(project / "project.json", data)
    assert stage.assemble(project)["refreshed"] == ["music"]
    data["music"] = {"mode": "auto"}
    write_json(project / "project.json", data)
    monkeypatch.setattr("yap_studio.stages.music.music",
                        lambda *_args: pytest.fail("paid music must not run"))
    with pytest.raises(stage.AssemblyError, match="music has no current verified receipt"):
        stage.assemble(project)


def test_graphics_refresh_is_reported_in_cli(project: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    record = read_receipt(project / "graphics")
    record["inputs"]["stale-test"] = "0" * 64
    write_json(project / "graphics/receipt.json", record)
    assert cli.main(["assemble", str(project)]) == 0
    output = capsys.readouterr().out
    assert "Re-ran stale stages: graphics." in output
    assert "Verified final: final/story.mp4" in output


def test_stopped_failed_then_successful_assemble(project: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    original = StageRun.replace_output
    interrupted = False

    def stop_after_first(run: StageRun, target: Path, staged: Path) -> None:
        nonlocal interrupted
        original(run, target, staged)
        if not interrupted and run.stage == "assemble":
            interrupted = True
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(StageRun, "replace_output", stop_after_first)
        with pytest.raises(KeyboardInterrupt):
            stage.assemble(project)
    assert json.loads((project / "final/stale.json").read_bytes())["reason"] == "stopped"
    assert (project / "final/receipt.pending.json").exists()
    write_json(project / "final/progress.json", {"pid": 2147483647,
                                                   "step": "abandoned"})
    with monkeypatch.context() as patch:
        patch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "fail"})
        with pytest.raises(stage.AssemblyError, match="verify.status"):
            stage.assemble(project)
    assert stage.assemble(project)["status"] == "verified"


@pytest.mark.parametrize("with_chapters", [False, True])
def test_failed_assemble_stales_old_final_and_clears_it_on_success(
    project: Path, monkeypatch: pytest.MonkeyPatch, with_chapters: bool,
) -> None:
    chapter_text = "0:00 Start\n0:10 Middle\n0:20 End\n"
    original_verify = stage.verify

    def verify_with_chapters(*args, **kwargs):
        return {**original_verify(*args, **kwargs), "youtube_chapters": chapter_text}

    with monkeypatch.context() as patch:
        if with_chapters:
            patch.setattr(stage, "verify", verify_with_chapters)
        assert stage.assemble(project)["status"] == "verified"
    final = project / "final/story.mp4"
    sidecars = [final.with_suffix(suffix) for suffix in
                (".layout.json", ".contrast.json", ".faces.json", ".taste.json")]
    chapters = final.with_suffix(".chapters.txt")
    if with_chapters:
        assert chapters.read_text() == chapter_text
        sidecars.append(chapters)
    expected = {f"story.stale{path.name[len('story'):]}": {
        "path": f"story.stale{path.name[len('story'):]}",
        "sha256": receipt.sha256_file(path), "bytes": path.stat().st_size,
    } for path in [final, *sidecars]}
    with monkeypatch.context() as patch:
        patch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "fail"})
        with pytest.raises(stage.AssemblyError, match="previous final was moved"):
            stage.assemble(project, force=True)
    manifest = json.loads((project / "final/stale.json").read_bytes())
    assert manifest["schema"] == "yap-studio/stale-final@1"
    assert not final.exists() and final.with_name("story.stale.mp4").is_file()
    assert all(not path.exists() and path.with_name(
        path.name.replace("story.", "story.stale.")).exists() for path in sidecars)
    assert {item["path"]: item for item in manifest["outputs"]} == expected
    if with_chapters:
        assert final.with_suffix(".stale.chapters.txt").read_text() == chapter_text
    assert stage.assemble(project)["status"] == "verified"
    assert final.is_file() and not (project / "final/stale.json").exists()
    assert all(not (final.parent / name).exists() for name in expected)
    assert not chapters.exists()


@pytest.mark.parametrize("frames", [None, (0, 20)])
def test_processing_failure_gets_doctor_action(
    project: Path, monkeypatch: pytest.MonkeyPatch, frames: tuple[int, int] | None,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ProcessingError("ffmpeg", 1, "mix failed")

    monkeypatch.setattr(stage, "_assemble_impl", fail)
    with pytest.raises(ProcessingError) as caught:
        stage.assemble(project, frames=frames)
    assert caught.value.code == "ffmpeg_exit_1"
    assert "yap-studio doctor" in caught.value.suggested_action
    rerun = f"yap-studio run {project}"
    if frames:
        rerun += f" --frames {frames[0]}-{frames[1]}"
    assert f"{rerun} again" in caught.value.suggested_action


def test_running_assemble_keeps_owned_final(project: Path) -> None:
    stage.assemble(project)
    write_json(project / "final/progress.json", {"pid": os.getpid(), "step": "rendering"})
    with pytest.raises(Exception, match="already running"):
        stage.assemble(project, force=True)
    assert (project / "final/story.mp4").is_file()
    assert not (project / "final/stale.json").exists()


def test_live_assemble_keeps_final_when_snapshot_refuses(project: Path) -> None:
    stage.assemble(project)
    final = project / "final/story.mp4"
    sidecars = [final.with_suffix(suffix) for suffix in
                (".layout.json", ".contrast.json", ".faces.json", ".taste.json")]
    write_json(project / "final/progress.json", {"pid": os.getpid(), "step": "rendering"})
    data = json.loads((project / "project.json").read_bytes())
    data["music"] = {"mode": "auto"}
    write_json(project / "project.json", data)
    with pytest.raises(stage.StaleStageError, match="music has no current verified receipt"):
        stage.assemble(project)
    assert final.is_file() and all(path.is_file() for path in sidecars)
    assert not (project / "final/stale.json").exists()


def test_assemble_cleans_only_old_assemble_work(project: Path) -> None:
    work = project / ".yap/work"
    (work / "assemble-abandoned").mkdir(parents=True)
    (work / "pilot-abandoned").mkdir()
    outside = project / "outside"
    outside.mkdir()
    (work / "assemble-link").symlink_to(outside, target_is_directory=True)
    stage.assemble(project)
    assert not (work / "assemble-abandoned").exists()
    assert (work / "pilot-abandoned").is_dir()
    assert (work / "assemble-link").is_symlink() and outside.is_dir()
    stage.assemble(project, frames=(30, 89))
    assert not (work / "pilot-abandoned").exists()


def test_unremovable_old_work_does_not_block_assemble(project: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    old = project / ".yap/work/assemble-unremovable"
    old.mkdir(parents=True)
    original = shutil.rmtree

    def refuse_old(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) == old:
            if not kwargs.get("ignore_errors"):
                raise PermissionError("cannot remove old work")
            return
        original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "rmtree", refuse_old)
        assert stage.assemble(project)["status"] == "verified"
    assert old.is_dir()


def test_second_failure_keeps_first_stale_set(project: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    stage.assemble(project)
    with monkeypatch.context() as patch:
        patch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "fail"})
        with pytest.raises(stage.AssemblyError):
            stage.assemble(project, force=True)
    first = json.loads((project / "final/stale.json").read_bytes())
    stale = project / "final/story.stale.mp4"
    first_hash = receipt.sha256_file(stale)
    original = StageRun.replace_output
    published = 0

    def fail_after_publish(run: StageRun, target: Path, staged: Path) -> None:
        nonlocal published
        original(run, target, staged)
        if run.stage == "assemble":
            published += 1
            if published == 3:
                raise RuntimeError("stopped after publication")

    with monkeypatch.context() as patch:
        patch.setattr(StageRun, "replace_output", fail_after_publish)
        with pytest.raises(RuntimeError, match="stopped after publication"):
            stage.assemble(project)
    second = json.loads((project / "final/stale.json").read_bytes())
    assert second == first
    assert receipt.sha256_file(stale) == first_hash
    assert not (project / "final/story.mp4").exists()
    assert not (project / "final/story.layout.json").exists()
    assert not (project / "final/story.faces.json").exists()
    assert stage.assemble(project)["status"] == "verified"
    assert not stale.exists() and not (project / "final/stale.json").exists()


@pytest.mark.parametrize("completed_moves", range(6))
@pytest.mark.parametrize("changed", [False, True])
def test_mark_stale_resumes_interrupted_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_moves: int, changed: bool,
) -> None:
    folder = tmp_path / "final"
    folder.mkdir()
    sources = [folder / f"story{suffix}" for suffix in
               (".mp4", ".layout.json", ".contrast.json", ".faces.json", ".taste.json")]
    for index, source in enumerate(sources):
        source.write_bytes(f"owned {index}".encode())
    expected_hashes = {f"story.stale{source.name[len('story'):]}":
                       receipt.sha256_file(source) for source in sources}
    (folder / "receipt.json").write_text(json.dumps({
        "schema": "yap-studio/stage-receipt@1", "status": "verified",
        "outputs": [{"path": f"final/{source.name}",
                     "sha256": receipt.sha256_file(source)} for source in sources],
    }))
    unrelated = folder / "unrelated.mp4"
    unrelated.write_bytes(b"unrelated")
    original_replace = os.replace
    moves = 0

    def interrupt(source: Path, target: Path) -> None:
        nonlocal moves
        if not Path(target).name.startswith("story.stale"):
            original_replace(source, target)
            return
        if completed_moves == 0:
            raise KeyboardInterrupt
        original_replace(source, target)
        moves += 1
        if moves == completed_moves:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(stage.os, "replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            stage._mark_stale(tmp_path, "story", "stopped")

    manifest_path = folder / "stale.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert {item["path"]: item["sha256"] for item in manifest["outputs"]} == expected_hashes
    changed_source = sources[-1] if completed_moves < len(sources) else None
    changed_stale = folder / "story.stale.taste.json" if completed_moves == len(sources) else None
    if changed:
        (changed_source or changed_stale).write_bytes(b"changed")
    assert stage._mark_stale(tmp_path, "story", "retry") == "final/story.stale.mp4"
    assert manifest_path.read_bytes() == manifest_bytes
    assert all(not source.exists() for source in sources if source != changed_source or not changed)
    if changed and changed_source:
        assert changed_source.read_bytes() == b"changed"
    assert all((folder / item["path"]).is_file() and
               receipt.sha256_file(folder / item["path"]) == item["sha256"]
               for item in manifest["outputs"]
               if not changed or item["path"] != "story.stale.taste.json")
    if changed and changed_stale:
        assert changed_stale.read_bytes() == b"changed"
    assert unrelated.read_bytes() == b"unrelated"


def test_mark_stale_does_not_report_missing_stale_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "final"
    folder.mkdir()
    final = folder / "story.mp4"
    sidecar = folder / "story.faces.json"
    final.write_bytes(b"owned final")
    sidecar.write_bytes(b"owned sidecar")
    (folder / "receipt.json").write_text(json.dumps({
        "schema": "yap-studio/stage-receipt@1", "status": "verified",
        "outputs": [{"path": f"final/{source.name}",
                     "sha256": receipt.sha256_file(source)}
                    for source in (final, sidecar)],
    }))
    original_replace = os.replace

    def interrupt(source: Path, target: Path) -> None:
        if Path(target).name.startswith("story.stale"):
            raise KeyboardInterrupt
        original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(stage.os, "replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            stage._mark_stale(tmp_path, "story", "stopped")

    manifest = folder / "stale.json"
    manifest_bytes = manifest.read_bytes()
    final.write_bytes(b"changed final")
    assert stage._mark_stale(tmp_path, "story", "retry") is None
    assert final.read_bytes() == b"changed final"
    assert not (folder / "story.stale.mp4").exists()
    assert manifest.read_bytes() == manifest_bytes
    assert (folder / "story.stale.faces.json").read_bytes() == b"owned sidecar"


def test_failed_retry_removes_partial_owned_by_failed_receipt(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage.assemble(project)
    with monkeypatch.context() as patch:
        patch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "fail"})
        with pytest.raises(stage.AssemblyError):
            stage.assemble(project, force=True)
    final = project / "final/story.mp4"
    stale = project / "final/story.stale.mp4"
    first_hash = receipt.sha256_file(stale)
    original = StageRun.replace_output

    def stop_after_mp4(run: StageRun, target: Path, staged: Path) -> None:
        original(run, target, staged)
        if run.stage == "assemble" and target == final:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(StageRun, "replace_output", stop_after_mp4)
        patch.setattr(stage, "_mark_stale", lambda *_args: None)
        with pytest.raises(KeyboardInterrupt):
            stage.assemble(project)
    assert final.is_file() and (project / "final/receipt.pending.json").is_file()
    with monkeypatch.context() as patch:
        patch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "fail"})
        with pytest.raises(stage.AssemblyError):
            stage.assemble(project)
    assert read_receipt(project / "final")["status"] == "failed"
    assert not (project / "final/receipt.pending.json").exists()
    assert not final.exists()
    assert receipt.sha256_file(stale) == first_hash
    assert stage.assemble(project)["status"] == "verified"
    assert final.is_file() and not stale.exists()
    assert not (project / "final/stale.json").exists()


def test_noop_clears_stale_left_after_verified_finish(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage.assemble(project)
    with monkeypatch.context() as patch:
        patch.setattr(stage, "verify", lambda *args, **kwargs: {"status": "fail"})
        with pytest.raises(stage.AssemblyError):
            stage.assemble(project, force=True)
    stale = project / "final/story.stale.mp4"
    assert stale.is_file()
    with monkeypatch.context() as patch:
        patch.setattr(stage, "_clear_stale", lambda _root: None)
        assert not stage.assemble(project)["noop"]
    assert read_receipt(project / "final")["status"] == "verified"
    assert stale.is_file() and (project / "final/stale.json").is_file()
    assert stage.assemble(project)["noop"]
    assert not stale.exists() and not (project / "final/stale.json").exists()


def test_pilot_recovers_pending_output_after_failed_retry(project: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    original = StageRun.replace_output
    stopped = False

    def interrupt(run: StageRun, target: Path, staged: Path) -> None:
        nonlocal stopped
        original(run, target, staged)
        if run.stage == "pilot" and not stopped:
            stopped = True
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(StageRun, "replace_output", interrupt)
        with pytest.raises(KeyboardInterrupt):
            stage.assemble(project, frames=(30, 89))
    assert (project / "pilot/receipt.pending.json").exists()
    original_checks = stage.delivery_asserts

    def fail_checks(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
        asserts, measured = original_checks(*args, **kwargs)
        asserts[0] = {**asserts[0], "pass": False}
        return asserts, measured

    with monkeypatch.context() as patch:
        patch.setattr(stage, "delivery_asserts", fail_checks)
        with pytest.raises(stage.AssemblyError, match="Pilot delivery failed"):
            stage.assemble(project, frames=(30, 89))
    assert read_receipt(project / "pilot")["status"] == "failed"
    assert stage.assemble(project, frames=(30, 89))["status"] == "verified"


@pytest.mark.parametrize("suffix", [".mp4", ".layout.json", ".contrast.json", ".faces.json"])
def test_force_refuses_unowned_final(project: Path, suffix: str) -> None:
    target = (project / "final/story.mp4").with_suffix(suffix)
    target.parent.mkdir()
    target.write_bytes(b"unowned output")
    with pytest.raises(OutputPathError):
        stage.assemble(project, force=True)
    assert target.read_bytes() == b"unowned output"


def test_short_music_file_cli_exits_two_with_fix(project: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    synth.float_audio(project / "short.wav", np.ones(1000, dtype=np.float32) * 0.05)
    assert cli.main(["music", str(project), "--file", "short.wav", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "music_too_short"
    assert "longer track" in error["suggested_action"]


@pytest.mark.render
def test_real_render_final_delivery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the pinned browser layer and final mux with a synthetic project."""
    from yap_studio.render import hyperframes
    from yap_studio.stages import captions as caption_stage

    # The module's alpha fixture is deterministic; opt-in coverage restores the actual renderer.
    monkeypatch.undo()
    source = synth.loudness_master(tmp_path / "master.mov", frames=48)
    root = make_project(tmp_path / "project", [source],
                        word_rows=[[("Small", 0.1, 0.3), ("steps", 0.4, 0.7)]])
    data = json.loads((root / "project.json").read_bytes())
    data["asr"] = {"engine": "none"}
    write_json(root / "project.json", data)
    ingest_record = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({row["path"]: row for row in ingest_record["outputs"]})
        run.finish()
    write_edit(root, [time_segment("s1", "t1", 0, 1.6)], end_hold_s=0.2)
    from yap_studio.stages.cut import cut

    cut(root)
    assert caption_stage.hyperframes.render_layer is hyperframes.render_layer
    captions(root)
    graphics(root)
    music(root)
    # Bright synthetic pictures exercise the automatic halo and saved box choice.
    result = stage.assemble(root)
    assert result["caption_halo"]["added"] >= 1
    assert read_receipt(root / "final")["status"] == "verified"
    assert (root / result["out"]).with_suffix(".contrast.json").is_file()
    if read_receipt(root / "final")["caption_contrast"]["weak"]:
        assert caption_stage.back_weak_captions(root, "box")["backed"]
        graphics(root)
        music(root)
        result = stage.assemble(root)
    final = root / result["out"]
    contrast = json.loads(final.with_suffix(".contrast.json").read_bytes())
    ratios = [caption["ratio"] for caption in contrast["captions"]]
    assert ratios and all(ratio is not None and ratio >= 3 for ratio in ratios), ratios
    layout = json.loads(final.with_suffix(".layout.json").read_bytes())
    assert layout["measured"] and layout["captions"]
    assert layout["media_sha256"] == receipt.sha256_file(final)
    assert load_faces(final.with_suffix(".faces.json")).video.sha256 == receipt.sha256_file(final)
    assert read_receipt(root / "final")["status"] == "verified"


def test_mid_render_voice_change_uses_snapshot_and_leaves_final_stale(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent stem edit cannot leak into an in-progress final render."""
    render = stage.render_picture

    def changed(*args: Any, **kwargs: Any) -> Any:
        (project / "cut/voice.wav").unlink()
        synth.float_audio(project / "cut/voice.wav", np.zeros(192000, dtype=np.float32))
        return render(*args, **kwargs)

    monkeypatch.setattr(stage, "render_picture", changed)
    stage.assemble(project)
    assert abs(measure(project / "final/story.mp4", []).integrated_lufs + 16) <= 0.10001
    assert final_stage(project)["state"] in {"missing", "stale"}
    assert read_receipt(project / "final")["status"] == "verified"


def test_allow_violations_is_explicit_and_end_hold_is_never_overridden(project: Path) -> None:
    """Editorial overrides must be visible in the receipt while media bounds stay enforced."""
    from yap_studio.stages.timeline import TimelineRefusal

    data = json.loads((project / "project.json").read_bytes())
    data.setdefault("rules", {})["zoom_step_min"] = 0
    write_json(project / "project.json", data)
    lane = {"schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [
        {"id": "agent-zoom", "lane": "agent", "kind": "zoom", "start": {"time": 0.5},
         "end": {"time": 3}, "peak_scale": 1.04, "center": [0.5, 0.5]},
    ]}
    write_json(project / "timeline/agent.json", lane)
    captions(project)
    data["rules"]["zoom_max_per_video"] = 0
    write_json(project / "project.json", data)
    with pytest.raises(TimelineRefusal, match="too many zooms"):
        stage.assemble(project)
    stage.assemble(project, allow_violations=True)
    assert read_receipt(project / "final")["allow_violations"]
    assert final_stage(project)["state"] == "done"
    lane["elements"][0]["end"] = {"time": 4.1}
    write_json(project / "timeline/agent.json", lane)
    with pytest.raises(TimelineRefusal, match="hold"):
        stage.assemble(project, allow_violations=True)


def test_hard_jump_cuts_snapshot_has_no_new_inputs(project: Path) -> None:
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert "picture:jump_cuts" not in snapshot.inputs


@pytest.mark.parametrize("mode", ["hard", "punch", "smooth"])
def test_jump_cut_snapshot_tracks_piece_plan_only_for_smooth(project: Path, mode: str) -> None:
    baseline = stage.assembly_snapshot(ProjectContext(project)).inputs
    style_path = project / "assembly-style.json"
    style = json.loads(style_path.read_bytes())
    style["moves"]["jump_cuts"] = mode
    write_json(style_path, style)
    record_path = project / "cut/receipt.json"
    record = json.loads(record_path.read_bytes())
    record["plan"]["pieces"] = [
        dict(segment_id="s1", take="t1", src_f0=0, src_f1=60, out_f0=0, n_frames=60),
        dict(segment_id="s1", take="t1", src_f0=60, src_f1=120, out_f0=60, n_frames=60),
    ]
    write_json(record_path, record)
    before = stage.assembly_snapshot(ProjectContext(project))
    record["plan"]["pieces"][1].update(src_f0=30, src_f1=90)
    write_json(record_path, record)
    after = stage.assembly_snapshot(ProjectContext(project))
    if mode == "smooth":
        assert before.inputs != after.inputs
        assert {key for key in before.inputs if before.inputs[key] != after.inputs[key]} == {
            "cut/receipt.json"}
        for relative in ("cut/receipt.json", "cut/cut-words.json"):
            assert after.inputs[relative] == receipt.sha256_file(project / relative)
            assert after.small_files[relative] == (project / relative).read_bytes()
    else:
        assert before.inputs == after.inputs == baseline
        assert "cut/receipt.json" not in after.inputs
        assert "picture:jump_cuts" not in after.inputs


@pytest.mark.render
@pytest.mark.parametrize("crf", [0, 16])
@pytest.mark.parametrize("project", [{"picture_filter": "eq=contrast=0.3:saturation=0.3"}],
                         indirect=True)
def test_smooth_jump_cuts_preserve_clock_audio_and_other_frames(
    project: Path, monkeypatch, crf: int,
) -> None:
    from test_smooth_joins import Blend

    from yap_studio.joins import helper
    from yap_studio.joins import plan as joins_plan

    monkeypatch.setattr(assemble_picture, "CRF", crf)
    if crf == 16:
        data = json.loads((project / "project.json").read_bytes())
        data["grade"] = {"look": "film"}
        write_json(project / "project.json", data)
    write_edit(project, [time_segment("s1", "t1", 0, 1),
                         time_segment("s2", "t1", 1.1, 3.5)], end_hold_s=0)
    cut(project)
    captions(project)
    graphics(project)
    music(project)
    hard_inputs = stage.assembly_snapshot(ProjectContext(project)).inputs
    assert "picture:jump_cuts" not in hard_inputs
    hard = stage.assemble(project)
    hard_snapshot = stage.assembly_snapshot(ProjectContext(project))
    hard_inputs = hard_snapshot.inputs
    hard_path = project / hard["out"]
    hard_frames = synth.picture_hashes(hard_path)
    hard_audio = decode(hard_path, [], channels=2)
    expected_frames = ffmpeg.probe(hard_path).video_frames
    style_path = project / "assembly-style.json"
    style = json.loads(style_path.read_bytes())
    style["moves"]["jump_cuts"] = "punch"
    write_json(style_path, style)
    punch_snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert punch_snapshot.inputs == hard_inputs
    assert punch_snapshot.small_files == hard_snapshot.small_files
    hard_receipt = (project / "final/receipt.json").read_bytes()
    assert stage.assemble(project)["noop"]
    assert "jump_cuts" not in read_receipt(project / "final")
    assert (project / "final/receipt.json").read_bytes() == hard_receipt
    assert synth.picture_hashes(hard_path) == hard_frames
    if crf == 0:
        stage.assemble(project, frames=(0, expected_frames - 1))
        assert "jump_cuts" not in read_receipt(project / "pilot")
    measured = FacesData("measured", None,
                         VideoIdentity("cut.mp4", receipt.sha256_file(project / "cut/cut.mp4"),
                                       expected_frames, Fraction(30), 1080, 1920), {}, {},
                         tuple((FaceBox(Box(400, 300, 200, 300), 1, "detected"),)
                               for _ in range(expected_frames)), ())
    monkeypatch.setattr(joins_plan, "cut_faces", lambda *args: measured)
    monkeypatch.setattr(helper, "unavailable_reason", lambda: None)
    blend = Blend()
    monkeypatch.setattr(helper, "AppleInterpolator", lambda: blend)
    style["moves"]["jump_cuts"] = "smooth"
    write_json(style_path, style)
    smooth = stage.assemble(project)
    smooth_path = project / smooth["out"]
    assert ffmpeg.probe(smooth_path).video_frames == expected_frames
    np.testing.assert_array_equal(decode(smooth_path, [], channels=2), hard_audio)
    record = read_receipt(project / "final")
    assert record["jump_cuts"]["counts"]["smooth"] == 1
    assert record["verify"]["status"] != "fail"
    assert len(blend.calls) == 1
    windows = record["jump_cuts"]["joins"]
    excluded = {i for j in windows if j["handling"] == "smooth"
                for i in range(j["start_frame"], j["start_frame"] + j["frames"])}
    smooth_frames = synth.picture_hashes(smooth_path)
    if crf == 0:
        # Lossless encoding isolates compositor pixels from temporal compression noise.
        assert all(a == b for i, (a, b) in enumerate(zip(hard_frames, smooth_frames, strict=True))
                   if i not in excluded)
        stage.assemble(project, frames=(0, expected_frames - 1))
        pilot = read_receipt(project / "pilot")
        assert pilot["jump_cuts"] == record["jump_cuts"]
        assert len(blend.calls) == 1
    else:
        edl = json.loads((project / "final/verify-edl.json").read_bytes())
        assert edl["grade"]["skip_frames"] == [[j["start_frame"], j["start_frame"] + j["frames"]]
                                                for j in windows if j["handling"] == "smooth"]
    style["moves"]["jump_cuts"] = "punch"
    write_json(style_path, style)
    assert stage.assembly_snapshot(ProjectContext(project)).inputs == hard_inputs
    monkeypatch.setattr(helper, "unavailable_reason", lambda: pytest.fail("Punch checked helper"))
    stage.assemble(project, force=True)
    assert "jump_cuts" not in read_receipt(project / "final")
    assert synth.picture_hashes(project / smooth["out"]) == hard_frames


@pytest.mark.parametrize(("kit", "fallback"), [("made", "none"), ("soft", "made")])
def test_made_receipt_without_account_or_cache(
    project: Path, monkeypatch: pytest.MonkeyPatch, kit: str, fallback: str,
) -> None:
    monkeypatch.setattr(elevenlabs, "account_state", lambda: pytest.fail("Account queried"))
    style_path = project / "assembly-style.json"
    data = json.loads(style_path.read_text())
    data["music"].update(sound_effects=kit, sound_effects_fallback=fallback)
    write_json(style_path, data)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert snapshot.sfx["ready"] and snapshot.sfx["kit"] == "made"
    assert len([key for key in snapshot.inputs if key.startswith("sfx:recipe:")]) == 9
    assert len([key for key in snapshot.inputs if key.startswith("sfx:audio:")]) == 9
    first = stage.assemble(project)
    details = read_receipt(project / "final")["audio"]["sfx"]
    assert details["kit"] == "made" and len(details["audio"]) == len(details["recipes"]) == 9
    assert not sfx.sfx_cache_root().exists()
    second = stage.assemble(project)
    if fallback == "made":
        assert details["fallback_from"] == "soft"
        assert any("Made by Yap Studio" in line and "sfx kit" in line for line in first["notices"])
        assert not second["notices"]
    else:
        assert not first["notices"]
    if kit == "made":
        assert not any("sound effects:" in item for item in human_items(ProjectContext(project)))


def test_no_fallback_keeps_sfx_projection(project: Path) -> None:
    context = ProjectContext(project)
    style = context.render_style().data
    before = stage.assembly_snapshot(context)
    assert not before.sfx["ready"] and before.sfx["kit"] == "soft"
    assert before.inputs["style:music"] == projection_sha256({
        key: value for key, value in style["music"].items()
        if key not in {"sound_effects_fallback", "sound_extras", "bleep_swears",
                       "bleep_words_add", "bleep_words_remove"}})
    assert before.inputs["style:sfx"] == projection_sha256({
        "kit": "soft", "peak_below_voice_db": style["internal"]["sfx_peak_below_voice_db"],
        "min_gap_s": style["internal"]["sfx_min_gap_s"]})


@pytest.mark.parametrize("kit", ["soft", "none"])
def test_default_made_fallback_and_effects_off(project: Path, kit: str) -> None:
    style_path = project / "assembly-style.json"
    data = json.loads(style_path.read_text())
    data["music"].pop("sound_effects_fallback", None)
    data["music"]["sound_effects"] = kit
    write_json(style_path, data)
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    if kit == "none":
        assert not snapshot.sfx["ready"] and snapshot.sfx["sounds"] == []
    else:
        assert snapshot.sfx["ready"] and snapshot.sfx["kit"] == "made"
        assert snapshot.sfx["fallback_from"] == "soft"
    result = stage.assemble(project)
    assert result
    cues = read_receipt(project / "final")["audio"]["sfx"]["cues"]
    assert bool(cues) == (kit != "none")


def test_ready_kit_keeps_legacy_inputs_with_new_defaults(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_listing = sfx.list_kit

    def listing(name: str = "soft") -> dict:
        result = original_listing(name)
        return {**result, "sounds": [{**sound, "state": "fresh", "audio_sha256": "1" * 64}
                                     for sound in result["sounds"]]}

    monkeypatch.setattr(sfx, "list_kit", listing)
    style_path = project / "assembly-style.json"
    data = json.loads(style_path.read_text())
    data["music"].update(sound_effects_fallback="none")
    data["music"].pop("sound_extras", None)
    write_json(style_path, data)
    before = stage.assembly_snapshot(ProjectContext(project))
    data["music"].pop("sound_effects_fallback")
    data["music"]["sound_extras"] = []
    write_json(style_path, data)
    after = stage.assembly_snapshot(ProjectContext(project))
    assert before.inputs == after.inputs and before.sfx == after.sfx


def test_extras_readiness_uses_only_moments_and_requests_and_never_generates(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs, "generate_sound", lambda *args, **kwargs:
                        pytest.fail("Assembly generated a paid sound"))
    style_path = project / "assembly-style.json"
    data = json.loads(style_path.read_text())
    data["music"].update(sound_extras=["typing", "camera-shutter"])
    write_json(style_path, data)
    context = ProjectContext(project)
    context._timeline = replace(context.resolved_timeline(), elements=(
        ResolvedElement("agent-shutter", "sound", "agent", 30, 31, "front", {
            "sound": "camera-shutter"}),
        ResolvedElement("agent-laugh", "sound", "agent", 60, 61, "front", {"sound": "laugh"}),
    ))
    snapshot = stage.assembly_snapshot(context)
    assert {sound["name"] for sound in snapshot.sfx["additional_sounds"]} == {
        "camera-shutter", "laugh"}
    assert snapshot.sfx["missing_extras"] == ["camera-shutter", "laugh"]
    assert "sfx:recipe:typing" not in snapshot.inputs
    assert "sfx:recipe:applause" not in snapshot.inputs
    before = snapshot.inputs
    original_listing = sfx.list_kit

    def listing(name: str = "soft") -> dict:
        result = original_listing(name)
        return {**result, "sounds": [{**sound, "state": "fresh", "audio_sha256": "1" * 64}
            if sound["name"] in {"camera-shutter", "laugh"} else sound
            for sound in result["sounds"]]}

    monkeypatch.setattr(sfx, "list_kit", listing)
    ready = stage.assembly_snapshot(context)
    assert ready.sfx["missing_extras"] == []
    assert ready.inputs["style:music"] == before["style:music"]
    assert ready.inputs != before
    assert {key for key in ready.inputs if key.startswith("sfx:audio:")} >= {
        "sfx:audio:camera-shutter", "sfx:audio:laugh"}


def test_missing_extras_notice_once(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    snapshot.sfx["missing_extras"] = ["typing"]
    monkeypatch.setattr(stage, "assembly_snapshot", lambda *args, **kwargs:
                        replace(snapshot, sfx=json.loads(json.dumps(snapshot.sfx))))
    monkeypatch.setattr(stage, "_pilot_phrases", lambda context: (b"", "hash"))
    monkeypatch.setattr(stage, "_pilot_window", lambda *args: None)
    monkeypatch.setattr(stage, "_pilot_assemble", lambda context, snapshot, *args, **kwargs:
                        {"notices": snapshot.sfx.get("notices", [])})
    first = stage.assemble(project, frames=(0, 0))
    assert any("sfx kit --kit extras --yes" in line for line in first["notices"])
    assert stage.assemble(project, frames=(0, 0))["notices"] == []


def test_auto_extra_readiness_ignores_every_unused_sound(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    style_path = project / "assembly-style.json"
    data = json.loads(style_path.read_text())
    data["music"]["sound_extras"] = ["typing", "camera-shutter"]
    write_json(style_path, data)
    context = ProjectContext(project)
    before = stage.assembly_snapshot(context)
    assert before.sfx["additional_sounds"] == [] and before.sfx["missing_extras"] == []
    # Isolate sound readiness from the unrelated graphic layer receipt.
    graphic_inputs = context.json("graphics/receipt.json")["inputs"]
    monkeypatch.setattr(stage, "STAGES", [
        replace(spec, inputs=lambda context: graphic_inputs) if spec.name == "graphics" else spec
        for spec in stage.STAGES])
    context._timeline = replace(context.resolved_timeline(), elements=(
        ResolvedElement("agent-terminal", "graphic", "agent", 30, 110, "front", {
            "template": "terminal", "slots": {"command": "echo hi"}}),))
    original_listing = sfx.list_kit

    def listing(name: str = "soft") -> dict:
        result = original_listing(name)
        return {**result, "sounds": [{**sound, "state": "fresh", "audio_sha256": "1" * 64}
            if sound["name"] == "typing" else sound for sound in result["sounds"]]}

    monkeypatch.setattr(sfx, "list_kit", listing)
    after = stage.assembly_snapshot(context)
    assert [sound["name"] for sound in after.sfx["additional_sounds"]] == ["typing"]
    assert after.sfx["missing_extras"] == []
    assert "sfx:audio:typing" in after.inputs and "sfx:recipe:camera-shutter" not in after.inputs


@pytest.mark.parametrize("account", [None, "missing_cli", "signed_out", "free"])
def test_requested_extra_reaches_verified_assembly_without_generation(
    project: Path, monkeypatch: pytest.MonkeyPatch, account: str | None,
) -> None:
    monkeypatch.setattr(elevenlabs, "generate_sound", lambda *args, **kwargs:
                        pytest.fail("Assembly generated a sound"))
    monkeypatch.setattr(elevenlabs, "account_state", lambda:
                        pytest.fail("Account queried") if account is None else account)
    style_path = project / "assembly-style.json"
    data = json.loads(style_path.read_text())
    data["music"].pop("sound_effects_fallback")
    if account is not None:
        data["music"]["sound_effects_fallback"] = "none"
    write_json(style_path, data)
    sound = next(item for item in sfx.load_kit("extras")["sounds"] if item["name"] == "laugh")
    digest = elevenlabs.sound_request_sha256(sfx.request(sound))
    target = sfx.sfx_cache_root() / "audio" / f"{digest}.mp3"
    audio = np.zeros((48000, 2), dtype=np.float32)
    audio[12000] = .8
    source = synth.float_audio(project.parent / "extra.wav", audio)
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    sfx._write_json(sfx.sfx_cache_root() / "generated.json", {
        "extras/laugh": sfx.recipe_sha256(sound)})
    set_elements(project, {"id": "agent-laugh", "kind": "sound", "lane": "agent",
        "start": {"time": 2}, "sound": "laugh", "align": "peak"})
    result = stage.assemble(project)
    record = read_receipt(project / "final")
    assert record["status"] == "verified"
    cues = record["audio"]["sfx"]["cues"]
    assert len(cues) == 1 and cues[0]["sound"] == "laugh"
    assert cues[0]["time_s"] == 1.75 and cues[0]["peak_at_s"] == 2
    assert cues[0]["relative_db"] <= -8
    if account is not None:
        notices = " ".join(result["notices"])
        assert "no automatic sound effects" in notices
        assert "has no sound effects" not in notices


@pytest.mark.parametrize(("kit", "name"), [("soft", "whoosh"), ("extras", "laugh")])
@pytest.mark.parametrize("cached", [False, True])
def test_requested_cache_is_independent_of_made_fallback(
    project: Path, monkeypatch: pytest.MonkeyPatch, kit: str, name: str, cached: bool,
) -> None:
    monkeypatch.setattr(elevenlabs, "generate_sound", lambda *args, **kwargs:
                        pytest.fail("Assembly generated a sound"))
    monkeypatch.setattr(elevenlabs, "account_state", lambda: pytest.fail("Account queried"))
    style_path = project / "assembly-style.json"
    data = json.loads(style_path.read_text())
    data["music"].update(sound_effects="soft", sound_effects_fallback="made")
    write_json(style_path, data)
    recipe = next(sound for sound in sfx.load_kit(kit)["sounds"] if sound["name"] == name)
    target = Path(next(sound for sound in sfx.list_kit(kit)["sounds"]
                       if sound["name"] == name)["cache_path"])
    if cached:
        audio = np.zeros((48000, 2), dtype=np.float32)
        audio[12000] = .8
        source = synth.float_audio(project.parent / "requested.wav", audio)
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        sfx._write_json(sfx.sfx_cache_root() / "generated.json", {
            f"{kit}/{name}": sfx.recipe_sha256(recipe)})
    set_elements(project, {"id": "agent-request", "kind": "sound", "lane": "agent",
        "start": {"time": 2}, "sound": name})
    before = stage.assembly_snapshot(ProjectContext(project))
    assert before.sfx["kit"] == "made" and before.sfx["fallback_from"] == "soft"
    result = stage.assemble(project)
    details = read_receipt(project / "final")["audio"]["sfx"]
    requested = [cue for cue in details["cues"] if cue["element_id"] == "agent-request"]
    if cached:
        assert len(requested) == 1
        assert requested[0]["time_s"] == 2 and requested[0]["peak_at_s"] == 2.25
        digest = receipt.sha256_file(target)
        assert details["requests"][name]["audio_sha256"] == digest
        assert before.inputs[f"sfx:request:audio:{name}"] == digest
        if kit == "soft":
            made_sound = next(sound for sound in sfx.list_kit("made")["sounds"]
                              if sound["name"] == name)
            assert details["audio"][name] == made_sound["audio_sha256"]
            assert before.inputs[f"sfx:audio:{name}"] == made_sound["audio_sha256"]
        # Replacing just this cache file invalidates assembly while fallback stays active.
        target.write_bytes(synth.float_audio(project.parent / "changed.wav",
            np.full((4800, 2), .1, dtype=np.float32)).read_bytes())
        after = stage.assembly_snapshot(ProjectContext(project))
        assert after.sfx["kit"] == "made" and after.inputs != before.inputs
        assert after.inputs[f"sfx:request:audio:{name}"] == receipt.sha256_file(target)
    else:
        assert requested == []
        assert any(f"some {kit} sounds are missing" in notice for notice in result["notices"])
        assert any(f"sfx kit --kit {kit} --yes" in notice for notice in result["notices"])
        assert f"sfx:request:audio:{name}" not in before.inputs
