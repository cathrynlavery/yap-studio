"""Synthetic title stage windows, saved style isolation and receipt ownership."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_captions_render import _alpha_frames
from test_cli_captions import fake_layers as fake_layers
from test_stage_captions import checked_layers as checked_layers
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements, state
from test_timeline import write

from yap_studio import cards, cli, ffmpeg, text_rules
from yap_studio.brand_words import set_words
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import ProjectContext, graphics_inputs
from yap_studio.render import fonts, hyperframes
from yap_studio.stages.captions import LayerAssets, captions
from yap_studio.stages.graphics import graphics, graphics_snapshot


def title(**changes: Any) -> dict[str, Any]:
    """Build a two-line timeline title without an explicit hold."""
    return {"id": "agent-title", "kind": "title", "start": {"time": 0.2},
            "lines": ["Small", "steps"], "accent": "steps", **changes}


def test_graphics_inputs_include_camera_plan(layer_project: Path) -> None:
    baseline = graphics_inputs(ProjectContext(layer_project))
    target = layer_project / "captions/camera3d.json"
    write(target, {"camera": {"samples": []}, "frames": 0})
    assert graphics_inputs(ProjectContext(layer_project)) == baseline
    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = True
    write(layer_project / "style.json", style)
    changed = graphics_inputs(ProjectContext(layer_project))
    assert changed["captions/camera3d.json"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert changed != baseline
    write(target, {"camera": {"samples": []}, "frames": 30})
    assert graphics_inputs(ProjectContext(layer_project)) != changed


@pytest.mark.parametrize("change", [None, "length", "inputs", "unverified", "pending",
                                     "edited", "missing", "unowned"])
@pytest.mark.parametrize("retained", [False, True])
def test_picture_faces_require_current_camera_captions(
    layer_project: Path, change, retained,
) -> None:
    from test_faces_mapping import _cut
    from test_stage_captions import republish_cut

    from yap_studio.faces.project import picture_faces
    from yap_studio.project.status import captions_inputs
    from yap_studio.render.camera3d import face_transforms
    from yap_studio.stages.assemble import StaleStageError

    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = True
    write(layer_project / "style.json", style)
    camera = {"frames": 90, "camera": {"samples": [
        {"px": 5, "py": -10, "tx": 0, "ty": 0, "cz": 0}] * 90}}
    write(layer_project / "captions/camera3d.json", camera)
    target = layer_project / "captions/camera3d.json"
    record = {"schema": "yap-studio/stage-receipt@1", "status": "verified",
              "picture": "captions/camera3d.mp4",
              "outputs": [{"path": "captions/camera3d.json",
                           "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}],
              "inputs": captions_inputs(ProjectContext(layer_project))}
    if change == "length":
        republish_cut(layer_project, frames=93)
    elif change == "inputs":
        style["brand"] = {"accent_color": "#F34E3F"}
        write(layer_project / "style.json", style)
    elif change == "unverified":
        record["status"] = "failed"
    elif change == "pending":
        write(layer_project / "captions/receipt.pending.json", {})
    elif change == "edited":
        camera["camera"]["samples"][0]["px"] = 20
        write(target, camera)
    elif change == "missing":
        target.unlink()
    elif change == "unowned":
        record["outputs"] = []
    write(layer_project / "captions/receipt.json", record)
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    options = {"context": context} if retained else {}
    cut = _cut(timeline.total_frames)
    if change:
        with pytest.raises(StaleStageError, match="run captions first") as error:
            picture_faces(layer_project, timeline, cut=cut, **options)
        assert error.value.stage == "captions"
        return
    mapped = picture_faces(layer_project, timeline, cut=cut, **options)
    scale, x, y = face_transforms(camera)[3]
    original, actual = cut.frames[3][0].box, mapped.frames[3][0].box
    assert (actual.x, actual.y, actual.w, actual.h) == pytest.approx((
        x + scale * original.x, y + scale * original.y, scale * original.w, scale * original.h))
    assert mapped.sampling["frame_sources"][0]["source"] == "camera3d"


@pytest.mark.parametrize("camera", [False, True])
def test_picture_faces_without_a_camera_picture_use_the_cut(layer_project: Path, camera) -> None:
    from test_faces_mapping import _cut

    from yap_studio.faces.project import picture_faces
    from yap_studio.project.status import captions_inputs

    style = json.loads((layer_project / "style.json").read_bytes())
    style["captions"]["camera_3d"] = camera
    write(layer_project / "style.json", style)
    record = {"schema": "yap-studio/stage-receipt@1", "status": "verified", "outputs": [],
              "inputs": captions_inputs(ProjectContext(layer_project))}
    write(layer_project / "captions/receipt.json", record)
    if not camera:
        write(layer_project / "captions/camera3d.json", {"frames": 1})
    cut = _cut(90)
    timeline = ProjectContext(layer_project).resolved_timeline()
    mapped = picture_faces(layer_project, timeline, cut=cut)
    assert mapped.frames[:90] == cut.frames
    assert all(frame == cut.frames[-1] for frame in mapped.frames[90:])
    assert mapped.sampling["frame_sources"][0]["source"] == "cut"


def test_bleep_masks_title_chapter_and_card_and_restores_placements(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": True}
    style["chapters"] = {"style": "title_card"}
    write(layer_project / "style.json", style)
    set_elements(layer_project,
                 title(start={"time": 0}, lines=["Fuck", "this"], accent="Fuck", hold_s=.5),
                 {"id": "agent-chapter", "kind": "chapter", "start": {"time": .5},
                  "title": "Shit happens", "hold_s": 1},
                 graphic(start={"time": 1.6}, template="quote",
                         slots={"quote": "bull shit", "name": "Fucker"}))
    snapshot = graphics_snapshot(ProjectContext(layer_project))
    first, chapter = snapshot.titles
    assert first.placement()["display_text"] == ["F***", "this", "F***"]
    assert chapter.placement()["display_text"] == ["S*** happens"]
    assert "F***" in json.dumps(first.plan.variables)
    assert "S*** happens" in json.dumps(chapter.plan.variables)
    card, = snapshot.graphics
    assert card.placement()["display_text"] == ["b*** s***", "F*****", ""]
    assert "b*** s***" in json.dumps(card.plan.variables)
    placements = [item.placement() for item in (*snapshot.titles, *snapshot.graphics)]
    restored = graphics_snapshot(ProjectContext(layer_project), placements=placements)
    assert restored.inputs == snapshot.inputs


def test_bleep_default_does_not_change_graphics_inputs(layer_project: Path) -> None:
    set_elements(layer_project, title())
    before = graphics_snapshot(ProjectContext(layer_project))
    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": False, "bleep_words_add": [], "bleep_words_remove": []}
    write(layer_project / "style.json", style)
    after = graphics_snapshot(ProjectContext(layer_project))
    assert before.inputs == after.inputs
    assert before.titles[0].placement() == after.titles[0].placement()
    legacy = ProjectContext(layer_project)
    for key in ("bleep_swears", "bleep_words_add", "bleep_words_remove"):
        legacy.render_style().data["music"].pop(key)
    assert graphics_snapshot(legacy).inputs == before.inputs


def test_bleep_keeps_kinetic_times_matched_to_original_text(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    style["music"] = {"bleep_swears": True, "bleep_words_add": ["simple"]}
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Simple steps"],
                                      accent="Simple", hold_s=1.5))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).titles
    assert json.loads(snapshot.plan.variables["word_times"]) == [.1, .3]
    assert json.loads(snapshot.plan.variables["lines"]) == ["S***** steps"]
    assert snapshot.placement()["display_text"] == ["S***** steps", "S*****"]


def test_bleep_records_a_masked_derived_monogram(layer_project: Path) -> None:
    enable_card_kind(layer_project, "comment_drop")
    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": True, "bleep_words_add": ["fu"]}
    write(layer_project / "style.json", style)
    set_elements(layer_project, graphic(template="comment_drop",
                                        slots={"text": "Safe", "name": "Fred User"}, hold_s=1.6))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    assert snapshot.plan.variables["initials"] == "F*"
    assert "F*" in snapshot.placement()["display_text"]


def test_bleep_records_reel_feed_initials(layer_project: Path) -> None:
    enable_card_kind(layer_project, "reel_feed")
    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": True, "bleep_words_add": ["fu", "views"]}
    write(layer_project / "style.json", style)
    set_elements(layer_project, graphic(template="reel_feed",
                                        slots={"handle": "Fred User", "views": "100"}, hold_s=2.5))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    assert snapshot.plan.variables["initials"] == "F*"
    assert "F*" in snapshot.placement()["display_text"]
    assert snapshot.plan.variables["views_text"] == "100 v****"
    assert "100 v****" in snapshot.placement()["display_text"]
    restored, = graphics_snapshot(ProjectContext(layer_project),
                                   placements=[snapshot.placement()]).graphics
    assert restored.placement() == snapshot.placement()


def test_bleep_letters_title_records_the_lines_drawn_by_the_plate(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "video_letters"
    style["music"] = {"bleep_swears": True, "bleep_words_add": ["small"]}
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Small", "steps"], accent="Small"))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).titles
    drawn = json.loads(snapshot.plan.variables["lines"])
    assert "S****" in drawn and "SMALL" not in drawn
    assert snapshot.placement()["display_text"] == drawn


def test_letters_graphics_receipt_stays_current_and_reuses_render(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import graphics as graphics_module

    rendered = []

    def render(plan, work, out, *, assets):
        rendered.append(plan)
        out.write_bytes(b"letters plate")
        return hyperframes.LayerResult(out, plan.frames, plan.width, plan.height, plan.frames / 30,
                                       checks={})

    monkeypatch.setattr(graphics_module, "render_layer_plan", render)
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "video_letters"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}))
    first = graphics(layer_project)
    record = read_receipt(layer_project / "graphics")
    assert first["elements"][0]["kind"] == "title"
    assert first["elements"][0]["letters"] is True
    assert set(record["inputs"]) == {"title:agent-title", "faces"}
    assert graphics_inputs(ProjectContext(layer_project)) == record["inputs"]
    assert state(layer_project, "graphics")["state"] == "done"
    assert graphics(layer_project)["noop"] is True
    assert len(rendered) == 1


def test_letters_only_snapshot_retains_face_inputs_without_measuring(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import graphics as graphics_module

    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "video_letters"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}))
    monkeypatch.setattr(graphics_module, "picture_faces",
                        lambda *args: pytest.fail("Letters should not measure faces"))
    first = graphics_snapshot(ProjectContext(layer_project))
    assert set(first.inputs) == {"title:agent-title", "faces"}
    restored = graphics_snapshot(ProjectContext(layer_project),
                                 placements=[first.titles[0].placement()])
    assert restored.inputs == first.inputs


def test_kinetic_title_uses_cut_words_fixes_and_hashes_timing(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Simple steps"], hold_s=1.5))
    first = graphics_snapshot(ProjectContext(layer_project)).titles[0]
    assert first.layer == "front"
    assert json.loads(first.plan.variables["word_times"]) == [0.1, 0.3]
    words = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    words["words"][1]["start"] = 0.4
    staged = write(layer_project / "staged-words-updated.json", words)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    changed = graphics_snapshot(ProjectContext(layer_project)).titles[0]
    assert json.loads(changed.plan.variables["word_times"]) == [0.1, 0.4]
    assert changed.digest != first.digest
    style["captions"]["shadow_blur_px"] = 10
    write(layer_project / "style.json", style)
    assert graphics_snapshot(ProjectContext(layer_project)).titles[0].digest != changed.digest


@pytest.mark.parametrize("selector", ["--word", "--heard"])
def test_word_fixes_stale_kinetic_title(layer_project: Path, fake_layers, capsys, selector) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Simple steps"], hold_s=1.5))
    graphics(layer_project)
    assert state(layer_project, "graphics")["state"] == "done"
    before = graphics_inputs(ProjectContext(layer_project))
    assert cli.main(["words", "fix", str(layer_project), selector,
                     "t1:2" if selector == "--word" else "more", "--to", "Extra", "--json"]) == 0
    capsys.readouterr()
    assert graphics_inputs(ProjectContext(layer_project)) != before
    assert state(layer_project, "graphics")["state"] == "stale"
    assert not graphics(layer_project)["noop"]
    assert state(layer_project, "graphics")["state"] == "done"


def test_kinetic_title_keeps_its_rounded_up_word_anchor(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    words = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    words["words"][0].update(start=0.22, end=0.27)
    staged = write(layer_project / "staged-anchor-words.json", words)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    set_elements(layer_project, title(start={"word": "t1:0"}, lines=["Simple steps"], hold_s=1.5))
    context = ProjectContext(layer_project)
    assert context.resolved_timeline().elements[0].start_frame == 7
    layer, = graphics_snapshot(context).titles
    assert json.loads(layer.plan.variables["word_times"]) == [0, 2 / 30]


@pytest.mark.parametrize("lines", [["twenty five steps"], ["25 steps"]])
def test_kinetic_title_number_phrase_renders_one_time_per_shown_word(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict], lines: list[str],
) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    words = json.loads((layer_project / "cut/cut-words.json").read_bytes())
    for word, text in zip(words["words"], ("twenty", "five", "steps.", "words."), strict=True):
        word["text"] = text
    staged = write(layer_project / "staged-number-words.json", words)
    previous = read_receipt(layer_project / "cut")
    with StageRun(layer_project, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.replace_output(layer_project / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    set_elements(layer_project, title(start={"time": 0}, lines=lines, hold_s=2))

    result = graphics(layer_project)
    assert result["elements"][0]["id"] == "agent-title"
    assert len(fake_layers) == 1
    variables = fake_layers[0]["variables"]
    shown = json.loads(variables["lines"])
    assert shown == ["25 steps"]
    times = json.loads(variables["word_times"])
    assert times == [0.1, 1.0]
    assert len(times) == sum(len(line.split()) for line in shown)


def test_kinetic_title_rules_use_project_vocabulary(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(layer_project / "config"))
    set_words(["SimpleSteps"])

    def aliases(vocabulary):
        assert tuple(vocabulary) == ("SimpleSteps",)
        return {"simple steps": "SimpleSteps"}

    monkeypatch.setattr(text_rules, "vocabulary_fixes", aliases)
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["SimpleSteps"], hold_s=1.5))
    layer, = graphics_snapshot(ProjectContext(layer_project)).titles
    assert json.loads(layer.plan.variables["word_times"]) == [0.1]


def test_kinetic_title_receipt_resize_keeps_word_times(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Simple steps"], hold_s=1.5))
    first = graphics_snapshot(ProjectContext(layer_project)).titles[0]
    placement = first.placement()
    placement["clearance"]["scale"] = .75
    restored = graphics_snapshot(ProjectContext(layer_project), placements=[placement]).titles[0]
    assert restored.plan.variables["kinetic_lines"] != first.plan.variables["kinetic_lines"]
    before = json.loads(first.plan.variables["kinetic_lines"])
    after = json.loads(restored.plan.variables["kinetic_lines"])
    assert after[0]["size_px"] == pytest.approx(before[0]["size_px"] * .75, abs=.01)
    assert restored.plan.variables["word_times"] == first.plan.variables["word_times"]


def test_kinetic_project_refusal_names_title_element(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Unspoken"], hold_s=1.5))
    with pytest.raises(ValidationError) as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == "title.agent-title.kinetic"
    assert '"Unspoken" is not spoken between 0 s and 1.5 s' in caught.value.detail


def test_kinetic_legacy_receipt_can_be_checked_for_staleness(layer_project: Path) -> None:
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Simple steps"], hold_s=1.5))
    placement = graphics_snapshot(ProjectContext(layer_project)).titles[0].placement()
    placement["clearance"].pop("original_scale")
    placement["clearance"].pop("scale")
    placement["clearance"]["original_size_px"] = 48
    restored = graphics_snapshot(ProjectContext(layer_project), placements=[placement]).titles[0]
    assert restored.plan.variables["kinetic_lines"]


def graphic(**changes: Any) -> dict[str, Any]:
    """Build a short card window on the synthetic output clock."""
    return {"id": "agent-tag", "kind": "graphic", "start": {"time": 0.2},
            "hold_s": 1, "template": "text_tag", "slots": {"text": "Small steps"}, **changes}


def turn_on_card_kind(project: Path, kind: str) -> None:
    style_path = project / "style.json"
    style = json.loads(style_path.read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = [kind]
    style_path.write_text(json.dumps(style))


def enable_card_kind(root: Path, kind: str) -> None:
    """Turn on a card kind in the synthetic project's style."""
    style_path = root / "style.json"
    style = json.loads(style_path.read_text())
    style.setdefault("graphics", {}).setdefault("card_kinds_on", []).append(kind)
    write(style_path, style)


def test_template_identity_keeps_browser_gpu_optional() -> None:
    assets = LayerAssets("<div>Card</div>", b'{"version": 1}', "", ())
    expected = {
        "renderer": hyperframes.HYPERFRAMES_VERSION, "workers": 1,
        "capture": hyperframes.RENDER_CAPTURE, "version": 1,
        "html": hashlib.sha256(assets.html.encode("utf-8")).hexdigest(),
        "metadata": hashlib.sha256(assets.metadata).hexdigest(),
    }
    assert assets.template_identity(workers=1) == expected
    assert assets.template_identity(workers=1, browser_gpu=None) == expected
    assert assets.template_identity(workers=1, browser_gpu=False) == {
        **expected, "browser_gpu": False,
    }


def test_file_drop_play_receipt_and_digest(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict], monkeypatch,
) -> None:
    from test_file_drop import clip_source

    from yap_studio import receipt
    from yap_studio.stages import graphics as module

    projections = []
    original_projection = module.projection_sha256

    def projection(data):
        if "element" in data:
            projections.append(data)
        return original_projection(data)

    monkeypatch.setattr(module, "projection_sha256", projection)

    enable_card_kind(layer_project, "file_drop")
    clip = clip_source(layer_project)
    slots = {"clip": clip.name, "poster_at": 25}
    set_elements(layer_project, graphic(template="file_drop", slots=slots, hold_s=1.5))
    still = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    set_elements(layer_project, graphic(template="file_drop", slots={**slots, "playback": "still"},
                                        hold_s=1.5))
    explicit = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert explicit.digest == still.digest
    assert "windows" not in explicit.placement()
    assert all("windows" not in item for item in projections)
    assert all(item["template"]["browser_gpu"] is False for item in projections)
    set_elements(layer_project, graphic(template="file_drop", slots={**slots, "playback": "play"},
                                        hold_s=1.5))
    expected_hash = receipt.sha256_file(clip)
    with monkeypatch.context() as patch:
        patch.setattr(receipt, "sha256_file", lambda *args: pytest.fail("uncached window hash"))
        snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    local, = snapshot.plan.windows
    assert projections[-1]["windows"] == snapshot.plan.windows
    assert local["source"]["sha256"] == snapshot.plan.poster.sha256 == expected_hash
    assert snapshot.digest != still.digest
    result = graphics(layer_project)
    placement, = result["elements"]
    window, = placement["windows"]
    assert window["x"] == placement["x"] + local["x"]
    assert window["y"] == placement["y"] + local["y"]
    assert window["start_frame"] == placement["start_frame"] + local["start"] == 32
    assert window["frames"] == 19
    assert window["source"] == {"kind": "media", "path": clip.name, "sha256": expected_hash,
                                 "in_s": 0.5, "speed": 1, "focus": [0.5, 0.5], "audio": "mute"}
    rebuilt = graphics_snapshot(ProjectContext(layer_project), placements=result["elements"])
    assert rebuilt.graphics[0].placement() == placement
    moved = layer_project.parent / "moved-play-project"
    shutil.copytree(layer_project, moved)
    assert graphics_snapshot(ProjectContext(moved)).graphics[0].digest == snapshot.digest
    assert graphics(layer_project)["noop"]


def test_file_drop_grade_changes_only_play_digest(layer_project: Path, monkeypatch) -> None:
    from test_file_drop import clip_source

    from yap_studio import grade

    project_path = layer_project / "project.json"
    project = json.loads(project_path.read_text())
    project["rules"] = {**project.get("rules", {}), "elements_at_once_max": 2}
    write(project_path, project)
    enable_card_kind(layer_project, "file_drop")
    clip = clip_source(layer_project)
    set_elements(layer_project,
                 graphic(id="agent-play", template="file_drop", hold_s=1.5,
                         slots={"clip": clip.name, "playback": "play", "poster_at": 0}),
                 graphic(id="agent-still", template="file_drop", hold_s=1.5,
                         slots={"clip": clip.name}),
                 graphic(id="agent-other", start={"time": 2}, hold_s=0.5))
    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_text())

    def snapshot(selection):
        write(style_path, {**data, "grade": selection})
        return {card.id: card for card in graphics_snapshot(ProjectContext(layer_project)).graphics}

    plain = snapshot({"look": "none"})
    film = snapshot({"look": "film", "strength": 0.5})
    strong = snapshot({"look": "film", "strength": 1})
    for previous, changed in ((plain, film), (film, strong)):
        assert previous["agent-play"].digest != changed["agent-play"].digest
        assert previous["agent-still"].digest == changed["agent-still"].digest
        assert previous["agent-other"].digest == changed["agent-other"].digest
    with monkeypatch.context() as patch:
        patch.setattr(grade, "GRADE_VERSION", grade.GRADE_VERSION + 1)
        version = snapshot({"look": "film", "strength": 1})
    assert version["agent-play"].digest != strong["agent-play"].digest
    assert version["agent-still"].digest == strong["agent-still"].digest
    assert version["agent-other"].digest == strong["agent-other"].digest
    project_path = layer_project / "project.json"
    project = json.loads(project_path.read_text())
    write(project_path, {**project, "grade": {"look": "none"}})
    override = snapshot({"look": "film", "strength": 1})
    assert override["agent-play"].digest == plain["agent-play"].digest
    write(project_path, {**project, "grade": {"look": "film", "strength": 1}})
    active = snapshot({"look": "none"})
    assert active["agent-play"].digest != override["agent-play"].digest
    assert active["agent-still"].digest == override["agent-still"].digest


def test_file_drop_graded_poster_cache_tracks_grade_and_version(
    layer_project: Path, tmp_path: Path, monkeypatch,
) -> None:
    from test_file_drop import clip_source

    from yap_studio import card_stills, grade
    from yap_studio.stages import graphics as module

    enable_card_kind(layer_project, "file_drop")
    clip = clip_source(layer_project)
    set_elements(layer_project, graphic(template="file_drop", hold_s=1.5,
                                        slots={"clip": clip.name, "playback": "play",
                                               "poster_at": 0}))
    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_text())
    extracted, posters = [], []
    original = card_stills.extract_frame

    def extract(*args, **kwargs):
        extracted.append(kwargs)
        return original(*args, **kwargs)

    def render(plan, work, out):
        folder = cards.prepare_card(plan, work)
        posters.append((folder / plan.variables["poster_src"]).read_bytes())
        return folder

    monkeypatch.setattr(card_stills, "extract_frame", extract)
    monkeypatch.setattr(cards, "render_card", render)
    for index, strength in enumerate((0.5, 1, 0.5)):
        write(style_path, {**data, "grade": {"look": "film", "strength": strength}})
        snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
        if index == 0:
            assert not extracted
        module._render_card(snapshot, tmp_path / f"grade-{index}", tmp_path / f"grade-{index}.mov")
    assert posters[0] == posters[2] and posters[0] != posters[1]
    assert len(extracted) == 2
    assert len(list((layer_project / ".yap/cache/card-stills").glob("*.png"))) == 2
    monkeypatch.setattr(grade, "GRADE_VERSION", grade.GRADE_VERSION + 1)
    version = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    module._render_card(version, tmp_path / "new-version", tmp_path / "new-version.mov")
    assert len(extracted) == 3
    assert len(list((layer_project / ".yap/cache/card-stills").glob("*.png"))) == 3


@pytest.mark.parametrize("badge", ["RAW", ""])
def test_bleep_records_the_file_label_derived_from_a_path(
    layer_project: Path, badge: str, monkeypatch,
) -> None:
    from test_file_drop import clip_source

    from yap_studio import file_cards
    from yap_studio.sfx import bleep
    from yap_studio.stages.timeline import ResolvedTimeline

    enable_card_kind(layer_project, "file_drop")
    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_bytes())
    style["music"] = {"bleep_swears": True}
    write(style_path, style)
    clip = clip_source(layer_project, "fuck.mp4")
    set_elements(layer_project, graphic(template="file_drop", slots={"clip": clip.name,
                                                                   "badge": badge}, hold_s=1.5))
    card, = graphics_snapshot(ProjectContext(layer_project)).graphics
    assert card.plan.variables["name"] == "f***.mp4"
    assert card.plan.poster.source == clip
    shown = card.placement()["display_text"]
    assert shown == ["f***.mp4", badge, card.plan.variables["duration_text"]]
    audio = np.zeros((3 * bleep.RATE, 1), dtype=np.float32)
    result = bleep.check([], music={}, voice=audio, original_voice=audio, tone_stem=None,
                         phrases={}, title_layers=[card.placement()],
                         timeline=ResolvedTimeline((), (), (), 90, 0))
    assert result["pass"]
    exposed = {**card.placement(), "display_text": ["fuck.mp4", badge]}
    assert not bleep.check([], music={}, voice=audio, original_voice=audio, tone_stem=None,
                           phrases={}, title_layers=[exposed],
                           timeline=ResolvedTimeline((), (), (), 90, 0))["pass"]
    missing = card.placement()
    missing.pop("display_text")
    assert not bleep.check([], music={}, voice=audio, original_voice=audio, tone_stem=None,
                           phrases={}, title_layers=[missing],
                           timeline=ProjectContext(layer_project).resolved_timeline(),
                           slot_rules={"file_drop": cards.load_template("file_drop")["slots"]}
                           )["pass"]
    file_variables = file_cards.file_variables

    def renamed(*args, **kwargs):
        return {**file_variables(*args, **kwargs), "name": "safe.mp4"}

    monkeypatch.setattr(file_cards, "file_variables", renamed)
    changed, = graphics_snapshot(ProjectContext(layer_project)).graphics
    assert changed.digest != card.digest


def test_file_drop_resolves_clip_and_digests_identity_without_extraction(
    layer_project: Path, fake_layers: list[dict], monkeypatch,
) -> None:
    from test_file_drop import clip_source

    from yap_studio import card_stills

    enable_card_kind(layer_project, "file_drop")
    clip = clip_source(layer_project)
    set_elements(layer_project, graphic(template="file_drop", slots={"clip": clip.name},
                                        hold_s=1.5),
                 graphic(id="agent-other", start={"time": 2}, hold_s=0.5))
    calls = []
    original = card_stills.extract_frame

    def extract(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(card_stills, "extract_frame", extract)
    first = graphics_snapshot(ProjectContext(layer_project))
    assert not calls
    card = first.graphics[0]
    assert card.plan.poster.source == clip
    assert card.plan.poster.cache_root == layer_project
    assert card.images == ()
    moved = layer_project.parent / "moved-file-project"
    shutil.copytree(layer_project, moved)
    after_move = graphics_snapshot(ProjectContext(moved),
                                    placements=[item.placement() for item in first.graphics])
    assert after_move.graphics[0].digest == card.digest
    graphics(layer_project)
    assert len(calls) == 1 and calls[0]["time_s"] == 1
    assert graphics(layer_project)["noop"] and len(calls) == 1
    clip.write_bytes(clip.read_bytes() + b"synthetic padding")
    after = graphics_snapshot(ProjectContext(layer_project))
    assert after.graphics[0].digest != first.graphics[0].digest
    assert after.graphics[1].digest == first.graphics[1].digest
    assert len(calls) == 1


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_file_drop_refuses_clip_outside_project(layer_project: Path, escape: str) -> None:
    from test_file_drop import clip_source

    enable_card_kind(layer_project, "file_drop")
    outside = clip_source(layer_project.parent, f"{layer_project.name}-external.mp4")
    if escape == "parent":
        clip = f"../{outside.name}"
    elif escape == "absolute":
        clip = str(outside)
    else:
        (layer_project / "escape.mp4").symlink_to(outside)
        clip = "escape.mp4"
    set_elements(layer_project, graphic(template="file_drop", slots={"clip": clip}, hold_s=1.5))
    with pytest.raises(ValidationError) as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == "graphic.slots.clip"
    assert caught.value.detail == "use a file inside the project folder"


def test_edit_timeline_uses_real_facts_and_narrow_digest(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    from test_card_project import spans

    enable_card_kind(layer_project, "edit_timeline")
    spans(layer_project)
    card = graphic(id="agent-edit", template="edit_timeline", slots={}, hold_s=1.5)
    tag = graphic(start={"time": 2}, hold_s=0.5)
    set_elements(layer_project, card, tag)
    first = graphics_snapshot(ProjectContext(layer_project))
    variables = first.graphics[0].plan.variables
    assert json.loads(variables["timecodes"])[0] == "00:00:06"
    assert variables["window_end_frame"] == 90
    assert [clip["width_px"] for clip in json.loads(variables["tracks"])[0]["clips"]] == (
        pytest.approx([variables["lane_width_px"] * n / 90 for n in (15, 30, 45)])
    )
    graphics(layer_project)
    assert fake_layers[0]["variables"] == variables
    set_elements(layer_project, card, tag, title(start={"time": 2.6}, hold_s=0.3))
    after = graphics_snapshot(ProjectContext(layer_project))
    assert first.inputs["graphic:agent-edit"] != after.inputs["graphic:agent-edit"]
    assert first.inputs["graphic:agent-tag"] == after.inputs["graphic:agent-tag"]
    write(layer_project / "cut/spans.json", {
        "source": "../source.mov", "segments": [{"start": 0, "end": 3}],
    })
    recut = graphics_snapshot(ProjectContext(layer_project))
    assert after.inputs["graphic:agent-edit"] != recut.inputs["graphic:agent-edit"]
    assert after.inputs["graphic:agent-tag"] == recut.inputs["graphic:agent-tag"]


def test_edit_timeline_keeps_facts_with_saved_box(layer_project: Path) -> None:
    from test_card_project import spans

    enable_card_kind(layer_project, "edit_timeline")
    spans(layer_project)
    set_elements(layer_project, graphic(template="edit_timeline", slots={}, hold_s=1.5,
                                        box={"x": 40, "y": 280, "width": 800, "height": 360}))
    first = graphics_snapshot(ProjectContext(layer_project))
    saved = graphics_snapshot(ProjectContext(layer_project),
                               placements=[first.graphics[0].placement()])
    assert saved.graphics[0].plan == first.graphics[0].plan
    assert json.loads(saved.graphics[0].plan.variables["timecodes"])[0] == "00:00:06"


def test_edit_timeline_uses_planner_box_in_project(layer_project: Path) -> None:
    from test_card_project import spans

    from yap_studio.planner.cards import plan_cards
    from yap_studio.project.model import parse_document

    enable_card_kind(layer_project, "edit_timeline")
    spans(layer_project)
    context = ProjectContext(layer_project)
    planned = plan_cards(
        {"schema": "yap-studio/card-requests@1", "cards": [{
            "id": "agent-edit", "template": "edit_timeline", "slots": {},
            "start": {"time": 0.2}, "hold_s": 1.5,
        }]}, parse_document("cut-words", context.json("cut/cut-words.json")),
        context.render_style(), frames=90,
    )
    set_elements(layer_project, *planned.lane()["elements"])
    actual = graphics_snapshot(ProjectContext(layer_project)).graphics[0].plan
    expected = planned.cards[0]
    assert (actual.x, actual.y, actual.width, actual.height) == (
        expected.x, expected.y, expected.width, expected.height,
    )
    assert json.loads(actual.variables["timecodes"])[0] == "00:00:06"


def test_graphics_stage_refuses_editor_kind_without_opt_in(layer_project: Path) -> None:
    set_elements(layer_project, graphic(template="edit_timeline", slots={}, hold_s=1.5))
    with pytest.raises(
        ValidationError, match="creator has not turned on edit_timeline cards",
    ) as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == "graphic.agent-tag.template"
    assert not (layer_project / "graphics").exists()


@pytest.mark.parametrize("crop", [None, [16, 8, 288, 162]])
def test_video_card_receipt_windows_and_reconstruction(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict],
    crop: list[int] | None,
) -> None:
    from yap_studio import receipt

    style = json.loads((layer_project / "style.json").read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = ["video_card"]
    write(layer_project / "style.json", style)
    source = layer_project / "clip.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:r=30",
                "-t", "4", str(source)], timeout_s=30)
    clip = {"path": source.name, "sha256": receipt.sha256_file(source)}
    if crop is not None:
        clip["crop"] = crop
    set_elements(layer_project, graphic(template="video_card", slots={"clip": clip}, hold_s=1.5))
    result = graphics(layer_project)
    placement, = result["elements"]
    window, = placement["windows"]
    assert window["x"] == placement["x"] + cards.PAD_X + 16
    assert window["y"] == placement["y"] + cards.PAD_TOP + 16
    assert window["start_frame"] == 6 and window["frames"] == 45
    assert window["source"]["path"] == "clip.mp4"
    assert window["source"]["sha256"] == clip["sha256"]
    if crop is None:
        assert "crop" not in window["source"]
    else:
        assert window["source"]["crop"] == crop
    rebuilt = graphics_snapshot(ProjectContext(layer_project), placements=result["elements"])
    assert rebuilt.graphics[0].placement() == placement
    assert graphics(layer_project)["noop"]
    source.unlink()
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30",
                "-t", "4", str(source)], timeout_s=30)
    clip["sha256"] = receipt.sha256_file(source)
    set_elements(layer_project, graphic(template="video_card", slots={"clip": clip}, hold_s=1.5))
    assert not graphics(layer_project)["noop"] and len(fake_layers) == 2


def test_video_card_width_search_and_blocked_fix(layer_project: Path, monkeypatch) -> None:
    from yap_studio import receipt
    from yap_studio.planner import cards as module

    source = layer_project / "clip.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=180x320:r=30",
                "-t", "4", str(source)], timeout_s=30)
    slots = {"clip": {"path": str(source), "sha256": receipt.sha256_file(source)}}
    style = ProjectContext(layer_project).render_style()
    plan = cards.card_layer_plan("video_card", slots, style, frames=45)
    seen = []

    def place(candidate, *args, **kwargs):
        seen.append(candidate.width)
        return candidate, "top", {"status": "clear" if candidate.width <= 640 else "blocked"}

    monkeypatch.setattr(module, "place_card", place)
    result, _, _ = module.place_video_card(plan, slots, style, None)
    assert result.width <= 640 and result.x == (1080 - result.width) // 2
    assert all(a - b == 80 for a, b in zip(seen, seen[1:], strict=False))
    monkeypatch.setattr(module, "place_card", lambda candidate, *args, **kwargs:
                        (candidate, "top", {"status": "blocked"}))
    _, _, clearance = module.place_video_card(plan, slots, style, None)
    assert clearance["status"] == "blocked"
    assert "face sits lower" in clearance["reason"]
    assert "box" in clearance["reason"] and "behind" in clearance["reason"]


def test_video_card_explicit_box_refits_window_before_checking_bounds(layer_project: Path) -> None:
    from yap_studio import receipt
    from yap_studio.planner.cards import fitted_card

    style = json.loads((layer_project / "style.json").read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = ["video_card"]
    write(layer_project / "style.json", style)
    source = layer_project / "clip.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:r=30",
                "-t", "4", str(source)], timeout_s=30)
    clip = {"path": source.name, "sha256": receipt.sha256_file(source)}
    expected = fitted_card("video_card", {"clip": {**clip, "path": str(source)}},
                           ProjectContext(layer_project).render_style(), frames=45, width=560)
    box = {"x": 260, "y": 80, "width": 560, "height": expected.height}
    set_elements(layer_project, graphic(template="video_card", slots={"clip": clip},
                                         hold_s=1.5, box=box))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    assert (snapshot.plan.x, snapshot.plan.y, snapshot.plan.width, snapshot.plan.height) == (
        box["x"], box["y"], box["width"], box["height"],
    )
    window, = snapshot.placement()["windows"]
    assert window["x"] + window["width"] <= box["x"] + box["width"]


@pytest.mark.parametrize("height_delta", [-1, 0])
def test_video_card_explicit_box_requires_full_panel_height(
    layer_project: Path, height_delta: int,
) -> None:
    from yap_studio.planner.cards import fitted_card

    style = json.loads((layer_project / "style.json").read_bytes())
    style.setdefault("graphics", {})["card_kinds_on"] = ["video_card"]
    write(layer_project / "style.json", style)
    clip = cards.preview_clip(layer_project)
    expected = fitted_card("video_card", {"clip": clip, "label": "A label"},
                           ProjectContext(layer_project).render_style(), frames=45, width=560)
    box = {"x": 260, "y": 80, "width": 560, "height": expected.height + height_delta}
    window, = expected.windows
    assert window["y"] + window["height"] < box["height"]
    clip["path"] = Path(clip["path"]).name
    set_elements(layer_project, graphic(template="video_card", slots={"clip": clip,
                                       "label": "A label"}, hold_s=1.5, box=box))
    if height_delta < 0:
        with pytest.raises(ValidationError, match=rf"height.*{expected.height}"):
            graphics_snapshot(ProjectContext(layer_project))
    else:
        snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
        assert snapshot.plan.height == expected.height
        assert snapshot.plan.variables["label"] == "A label"


@pytest.mark.parametrize("change", ["geometry", "source", "audio"])
def test_video_card_receipted_windows_must_match_the_plan(layer_project: Path, change: str) -> None:
    from yap_studio import receipt

    style = json.loads((layer_project / "style.json").read_text())
    style.setdefault("graphics", {})["card_kinds_on"] = ["video_card"]
    write(layer_project / "style.json", style)
    source = layer_project / "clip.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:r=30",
                "-t", "4", str(source)], timeout_s=30)
    clip = {"path": source.name, "sha256": receipt.sha256_file(source)}
    set_elements(layer_project, graphic(template="video_card", slots={"clip": clip}, hold_s=1.5))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    saved = snapshot.placement()
    if change == "geometry":
        saved["windows"][0]["x"] += 2
    elif change == "source":
        saved["windows"][0]["source"] = {"kind": "self"}
    else:
        saved["windows"][0]["source"]["audio"] = {"gain_db": 6}
    with pytest.raises(ValidationError, match="windows changed.*rerun graphics"):
        graphics_snapshot(ProjectContext(layer_project), placements=[saved])


def test_video_card_requires_opt_in_from_default_preset(layer_project: Path) -> None:
    from yap_studio.style import resolve

    style = resolve("preset:clean_minimal").to_json()
    write(layer_project / "style.json", style)
    clip = cards.preview_clip(layer_project)
    clip["path"] = Path(clip["path"]).name
    set_elements(layer_project, graphic(template="video_card", slots={"clip": clip}, hold_s=1.5))
    with pytest.raises(
        ValidationError, match="creator has not turned on video_card cards",
    ) as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == "graphic.agent-tag.template"
    style["graphics"]["card_kinds_on"] = ["video_card"]
    write(layer_project / "style.json", style)
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    assert snapshot.plan.template == "video_card"


@pytest.mark.parametrize("template,slots,lane_slots,landing_frame,relative", [
    ("gravity_drop", {"drops": [{"text": "Words", "at": 0.5}]},
     {"drops": [{"text": "Words", "at": {"time": 0.617}}]}, 19, 0.5),
    ("gravity_drop", {"drops": [{"text": "Words", "at": 0.51}]},
     {"drops": [{"text": "Words", "at": {"time": 0.627}}]}, 19, 0.5),
    ("gravity_drop", {"drops": [{"text": "Words", "at": 0.033}]},
     {"drops": [{"text": "Words", "at": {"time": 0.15}}]}, 5, 1 / 30),
    ("gravity_drop", {"drops": [{"text": "Words", "at": {"word": "t1:2"}}]},
     {"drops": [{"text": "Words", "at": {"word": "t1:2"}}]}, 30, 26 / 30),
    ("choices", {"options": ["One", "Two"], "cues": [0, {"word": "t1:2"}]},
     {"options": ["One", "Two"], "cues": [0, {"at": {
         "word": "t1:2", "edge": "start", "offset_s": 0.0,
         }}]}, None, 0.9),
    ("choices", {"options": ["One", "Two"], "cues": [0, 0.5]},
     {"options": ["One", "Two"], "cues": [0, 0.5]}, None, 0.5),
])
def test_planned_card_lane_anchor_round_trip(
    layer_project: Path, template: str, slots: dict, lane_slots: dict,
    landing_frame: int | None, relative: float,
) -> None:
    from yap_studio.planner import cards as planner
    from yap_studio.project.model import parse_document

    source = layer_project / "style.json"
    data = json.loads(source.read_bytes())
    data.setdefault("graphics", {})["card_kinds_on"] = [template]
    write(source, data)
    context = ProjectContext(layer_project)
    words = parse_document("cut-words", context.json("cut/cut-words.json"))
    result = planner.plan_cards({
        "schema": planner.REQUESTS_SCHEMA,
        "cards": [{"id": "agent-planned-card", "template": template, "slots": slots,
                   "start": {"time": 0.117 if template == "gravity_drop" else 0.1},
                   "hold_s": 2.5}],
    }, words, context.render_style(), frames=90)
    lane = result.lane()
    assert lane["elements"][0]["slots"] == lane_slots
    if template == "gravity_drop":
        assert result.cards[0].render_slots["drops"][0]["at"] == relative
    write(layer_project / "timeline/agent.json", lane)
    context = ProjectContext(layer_project)
    element, = context.resolved_timeline().elements
    if landing_frame is not None:
        resolved_at = element.payload["slots"]["drops"][0]["at"]
        assert resolved_at == {"frame": landing_frame, "time_s": landing_frame / 30}
    else:
        assert element.payload["slots"]["cues"] == [0, relative]
    snapshot, = graphics_snapshot(context).graphics
    if template == "gravity_drop":
        assert json.loads(snapshot.plan.variables["drops"])[0]["at"] == relative
    else:
        assert json.loads(snapshot.plan.variables["cues"]) == [0, relative]
    assert planner._render_key(snapshot.plan) == result.cards[0].render_key


def test_gravity_drop_graphics_resolves_nested_landing_anchors(layer_project: Path) -> None:
    set_elements(layer_project, graphic(
        template="gravity_drop", start={"time": 0.1}, hold_s=2.5,
        slots={"drops": [{"text": "More", "at": {"word": "t1:2"}}]},
        box={"x": 20, "y": 0, "width": 1040, "height": 360},
    ))
    with pytest.raises(ValidationError, match="creator has not turned on gravity_drop"):
        graphics_snapshot(ProjectContext(layer_project))
    source = layer_project / "style.json"
    data = json.loads(source.read_text())
    data.setdefault("graphics", {})["card_kinds_on"] = ["gravity_drop"]
    write(source, data)
    snapshot = graphics_snapshot(ProjectContext(layer_project))
    drop = json.loads(snapshot.graphics[0].plan.variables["drops"])[0]
    element = ProjectContext(layer_project).resolved_timeline().elements[0]
    expected = (element.payload["slots"]["drops"][0]["at"]["frame"]
                - element.start_frame) / 30
    assert drop["at"] == expected


@pytest.mark.parametrize("template", ["gravity_drop", "text_tag"])
@pytest.mark.parametrize("setting,value", [
    ("text_color", "#AABBCC"), ("size_px", 64), ("shadow_opacity", 0.7),
    ("shadow_blur_px", 9), ("shadow_x_px", 4), ("shadow_y_px", 5),
])
def test_gravity_drop_digest_tracks_caption_style_only(
    layer_project: Path, template: str, setting: str, value: Any,
) -> None:
    source = layer_project / "style.json"
    data = json.loads(source.read_text())
    data.setdefault("graphics", {})["card_kinds_on"] = ["gravity_drop"]
    write(source, data)
    slots = {"drops": [{"text": "More", "at": {"time": 0.7}}]} if template == "gravity_drop" else {
        "text": "More"}
    set_elements(layer_project, graphic(template=template, slots=slots, hold_s=2.5,
        box={"x": 20, "y": 0, "width": 1040, "height": 360}))
    first = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    data["captions"][setting] = value
    write(source, data)
    changed = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert (changed.digest != first.digest) == (template == "gravity_drop")


def test_gravity_drop_digest_tracks_font_cycle_order(layer_project: Path) -> None:
    source = layer_project / "style.json"
    data = json.loads(source.read_text())
    data.setdefault("graphics", {})["card_kinds_on"] = ["gravity_drop"]
    data["captions"]["font"], data["hook"]["font"] = "Gelasio", "Instrument Serif"
    write(source, data)
    set_elements(layer_project, graphic(template="gravity_drop", hold_s=2.5,
        slots={"drops": [{"text": "More", "at": {"time": 0.7}}]},
        box={"x": 20, "y": 0, "width": 1040, "height": 360}))
    first = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    data["captions"]["font"], data["hook"]["font"] = "Instrument Serif", "Gelasio"
    write(source, data)
    assert graphics_snapshot(ProjectContext(layer_project)).graphics[0].digest != first.digest


def test_giant_card_ceiling_uses_caption_layout_settings(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio import captions as caption_module
    from yap_studio.stages import graphics as stage
    from yap_studio.stages.captions import caption_band_geometry

    source = layer_project / "style.json"
    data = json.loads(source.read_text())
    data["captions"].update(highlight_style="giant", size_px=83, top_px=1700)
    write(source, data)
    set_elements(layer_project, graphic())
    context = ProjectContext(layer_project)
    style = context.render_style()
    assert style.data["captions"]["size_px"] == 83
    settings = caption_module.caption_layout_settings(style.data["captions"], style.data["brand"])
    expected = caption_band_geometry(settings, safe_bottom_px=1740).y
    ceilings = []
    original = stage.place_card

    def capture(*args, **kwargs):
        ceilings.append(kwargs["caption_top"])
        return original(*args, **kwargs)

    monkeypatch.setattr(stage, "place_card", capture)
    graphics_snapshot(context)
    assert ceilings == [expected]
    assert json.loads(source.read_text()) == data


def test_graphics_stage_refuses_a_card_kind_turned_off(layer_project: Path) -> None:
    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_text())
    style.setdefault("graphics", {})["card_kinds_off"] = ["text_tag"]
    style_path.write_text(json.dumps(style))
    set_elements(layer_project, graphic())
    with pytest.raises(ValidationError, match="creator turned off text_tag cards"):
        graphics_snapshot(ProjectContext(layer_project))


def test_storyboard_reads_identities_and_extracts_only_on_render(
    layer_project: Path, fake_layers: list[dict], monkeypatch,
) -> None:
    from test_card_project import spans

    from yap_studio import receipt
    from yap_studio.stages import graphics as stage

    enable_card_kind(layer_project, "storyboard")
    spans(layer_project)
    set_elements(layer_project, graphic(template="storyboard", slots={}, hold_s=1.5))
    calls = []

    def still(root, shot, **size):
        calls.append((shot, size))
        output = root / "synthetic-shot.jpg"
        output.write_bytes(b"synthetic thumbnail")
        return output

    monkeypatch.setattr(stage, "cached_still", still)
    first = graphics_snapshot(ProjectContext(layer_project))
    assert not calls
    assert len(first.graphics[0].shot_sources) == 1
    graphics(layer_project)
    assert len(calls) == 1
    assert json.loads(fake_layers[0]["variables"]["thumbnails"])[0]["src"] == "shot-01.jpg"
    assert calls[0][1]["width"] == first.graphics[0].plan.variables["thumb_width_px"] * 2
    assert graphics(layer_project)["noop"] and len(calls) == 1
    cut = layer_project / "cut/cut.mp4"
    cut.write_bytes(cut.read_bytes() + b"synthetic padding")
    record = json.loads((layer_project / "cut/receipt.json").read_text())
    for output in record["outputs"]:
        if output["path"] == "cut/cut.mp4":
            output["sha256"] = receipt.sha256_file(cut)
    write(layer_project / "cut/receipt.json", record)
    after = graphics_snapshot(ProjectContext(layer_project))
    assert first.graphics[0].digest != after.graphics[0].digest
    assert len(calls) == 1


def test_storyboard_digest_survives_a_project_move(layer_project: Path, tmp_path: Path) -> None:
    from test_card_project import spans

    enable_card_kind(layer_project, "storyboard")
    spans(layer_project)
    set_elements(layer_project, graphic(template="storyboard", slots={}, hold_s=1.5))
    first = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    moved = tmp_path / "moved-project"
    shutil.copytree(layer_project, moved)
    after = graphics_snapshot(ProjectContext(moved), placements=[first.placement()]).graphics[0]
    assert first.shot_sources == after.shot_sources
    assert all(not Path(shot.source).is_absolute() for _, shot in first.shot_sources)
    assert first.digest == after.digest


def test_storyboard_hashes_each_source_once_per_render(
    layer_project: Path, tmp_path: Path, fake_layers: list[dict], monkeypatch,
) -> None:
    from test_card_project import spans

    from yap_studio import card_stills, receipt
    from yap_studio.stages import graphics as stage

    enable_card_kind(layer_project, "storyboard")
    spans(layer_project)
    set_elements(layer_project, graphic(template="storyboard", slots={}, hold_s=1.5))
    snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    shot = snapshot.shot_sources[0][1]
    snapshot = replace(snapshot, shot_sources=tuple(
        (f"shot-{index + 1:02}.jpg", replace(shot, source_frame=index * 15))
        for index in range(5)
    ))
    original_hash = receipt.sha256_file
    hashes, extractions = [], []

    def hash_source(source):
        hashes.append(source)
        return original_hash(source)

    def extract(source, info, out, **kwargs):
        extractions.append(kwargs)
        out.write_bytes(b"synthetic thumbnail")

    monkeypatch.setattr(receipt, "sha256_file", hash_source)
    monkeypatch.setattr(card_stills, "extract_frame", extract)
    monkeypatch.setattr(card_stills.hdr, "conversion", lambda info: None)
    for index in range(2):
        stage._render_card(snapshot, tmp_path / f"work-{index}", tmp_path / f"card-{index}.mov")
        assert hashes.count(layer_project / "cut/cut.mp4") == index + 1
        assert len(extractions) == 5
    (layer_project / "cut/cut.mp4").write_bytes(b"changed synthetic cut")
    with pytest.raises(ValidationError, match="shot source changed; run graphics again"):
        stage._render_card(snapshot, tmp_path / "changed", tmp_path / "changed.mov")


def test_graphics_stage_requires_opt_in(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("text_tag",))
    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_text())
    style.setdefault("graphics", {})["card_kinds_off"] = []
    style["graphics"].pop("card_kinds_on", None)
    style_path.write_text(json.dumps(style))
    set_elements(layer_project, graphic())
    with pytest.raises(ValidationError, match="creator has not turned on text_tag cards"):
        graphics_snapshot(ProjectContext(layer_project))
    style["graphics"]["card_kinds_on"] = ["text_tag"]
    style_path.write_text(json.dumps(style))
    assert graphics_snapshot(ProjectContext(layer_project)).graphics[0].plan.template == "text_tag"


def test_choices_graphics_stage_uses_safe_order(layer_project: Path, monkeypatch) -> None:
    from yap_studio.stages import graphics as module

    turn_on_card_kind(layer_project, "choices")
    set_elements(layer_project, graphic(template="choices", slots={"options": ["One", "Two"]},
                                        hold_s=2.5))
    calls = []
    original = module.place_card

    def place(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "place_card", place)
    assert graphics_snapshot(ProjectContext(layer_project)).graphics[0].plan.template == "choices"
    assert calls[0]["safe_order"] == ("side", "headroom", "lower")


def test_flow_tags_graphics_safe_refit_boxes_and_receipts(layer_project: Path, monkeypatch) -> None:
    from test_planner_cards import synthetic_faces

    from yap_studio.faces.model import Box
    from yap_studio.stages import graphics as module

    style_path = layer_project / "style.json"
    data = json.loads(style_path.read_text())
    data.setdefault("graphics", {})["flow_look"] = "tags"
    style_path.write_text(json.dumps(data))
    monkeypatch.setattr(module, "picture_faces", lambda *args: synthetic_faces(
        Box(330, 642, 500, 200), frames=90))
    calls = []
    original = module.place_card

    def place(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "place_card", place)
    element = graphic(template="flow", slots={"nodes": ["One", "Two"]}, hold_s=2.5)
    set_elements(layer_project, element)
    snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert calls[0]["safe_order"] == ("headroom", "side", "lower")
    assert snapshot.clearance["position"] == "headroom"
    assert snapshot.plan.variables["text_px"] == 52 and snapshot.plan.height == 216
    box = {name: getattr(snapshot.plan, name) for name in ("x", "y", "width", "height")}
    set_elements(layer_project, {**element, "box": box})
    authored = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert authored.plan == snapshot.plan
    cached = graphics_snapshot(ProjectContext(layer_project), placements=[authored.placement()])
    assert cached.graphics[0].plan == snapshot.plan


@pytest.mark.parametrize("template", ["specs_list", "stat"])
@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
def test_specs_rows_graphics_safe_order_and_refit(
    layer_project: Path, monkeypatch, template: str, look: str,
) -> None:
    from test_cards import ROWS_ON_CUE_SLOTS
    from test_planner_cards import synthetic_faces

    from yap_studio.faces.model import Box
    from yap_studio.stages import graphics as module

    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_bytes())
    style.setdefault("graphics", {})["specs_rows"] = look
    style_path.write_text(json.dumps(style))
    monkeypatch.setattr(module, "picture_faces", lambda *args: synthetic_faces(
        Box(330, 820, 500, 200), frames=90))
    calls = []
    original = module.place_card

    def place(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "place_card", place)
    element = graphic(template=template, slots=ROWS_ON_CUE_SLOTS[template], hold_s=2.5)
    set_elements(layer_project, element)
    snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert calls[0]["safe_order"] == ("headroom", "side", "lower")
    assert snapshot.clearance["position"] == "headroom"
    assert snapshot.plan.y + snapshot.plan.variables["pad_top_px"] >= 280
    assert snapshot.plan.y + snapshot.plan.height <= 676
    box = {name: getattr(snapshot.plan, name) for name in ("x", "y", "width", "height")}
    set_elements(layer_project, {**element, "box": box})
    authored = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert authored.plan == snapshot.plan
    cached = graphics_snapshot(ProjectContext(layer_project), placements=[authored.placement()])
    assert cached.graphics[0].plan == snapshot.plan


def test_choices_graphics_refit_survives_boxes_and_receipts(
    layer_project: Path, monkeypatch,
) -> None:
    from test_planner_cards import synthetic_faces

    from yap_studio.faces.model import Box
    from yap_studio.stages import graphics as module

    turn_on_card_kind(layer_project, "choices")
    monkeypatch.setattr(module, "picture_faces", lambda *args: synthetic_faces(
        Box(330, 654, 500, 200), frames=90))
    element = graphic(template="choices", slots={"options": ["One", "Two"]}, hold_s=2.5)
    set_elements(layer_project, element)
    snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert snapshot.clearance["position"] == "headroom"
    assert snapshot.plan.variables["text_px"] == 56
    box = {name: getattr(snapshot.plan, name) for name in ("x", "y", "width", "height")}
    set_elements(layer_project, {**element, "box": box})
    authored = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert authored.plan == snapshot.plan
    cached = graphics_snapshot(ProjectContext(layer_project), placements=[authored.placement()])
    assert cached.graphics[0].plan == snapshot.plan


@pytest.mark.parametrize("kind", ("prompt_bar", "comment_cta", "comment_drop", "icon_strike",
                                 "choices"))
def test_graphics_stage_requires_creator_opt_in(layer_project: Path, kind: str) -> None:
    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_text())
    style.setdefault("graphics", {})["card_kinds_off"] = []
    style["graphics"].pop("card_kinds_on", None)
    style_path.write_text(json.dumps(style))
    slots = json.loads((cards.template_dir(kind) / "preview.json").read_text())["slots"]
    set_elements(layer_project, graphic(template=kind, slots=slots, hold_s=2.5))
    with pytest.raises(ValidationError, match=f"creator has not turned on {kind} cards"):
        graphics_snapshot(ProjectContext(layer_project))
    turn_on_card_kind(layer_project, kind)
    assert graphics_snapshot(ProjectContext(layer_project)).graphics[0].plan.template == kind


def test_cards_publish_movies_and_complete_placements(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict],
) -> None:
    set_elements(layer_project, graphic(), graphic(
        id="agent-stat", start={"time": 1.5}, template="stat", slots={"end": 12},
    ))
    result = graphics(layer_project)
    assert [item["id"] for item in result["elements"]] == ["agent-tag", "agent-stat"]
    for item, start in zip(result["elements"], (6, 45), strict=True):
        plan = cards.card_layer_plan(
            "text_tag" if start == 6 else "stat",
            {"text": "Small steps"} if start == 6 else {"end": 12},
            ProjectContext(layer_project).render_style(), frames=30,
        )
        assert (item["kind"], item["layer"], item["start_frame"], item["frames"]) == (
            "graphic", "front", start, 30,
        )
        assert tuple(item[key] for key in ("x", "y", "width", "height")) == (
            plan.x, plan.y, plan.width, plan.height,
        )
        assert len(item["input_sha256"]) == 64
        assert "windows" not in item
        info = ffmpeg.probe(layer_project / item["path"])
        assert (info.width, info.height, info.video_frames) == (plan.width, plan.height, 30)
        assert info.pix_fmt.startswith("yuva444p")
    record = read_receipt(layer_project / "graphics")
    assert record["elements"] == result["elements"]
    assert record["layer_checks"] == dict(zip(
        (item["id"] for item in result["elements"]), checked_layers, strict=True,
    ))
    assert all(checks["status"] == "findings" for checks in checked_layers)
    assert {item["path"] for item in record["outputs"]} == {
        "graphics/agent-tag.mov", "graphics/agent-stat.mov",
    }
    assert fake_layers[0]["variables"]["text"] == "Small steps"
    assert fake_layers[1]["variables"]["end"] == 12
    assert state(layer_project, "graphics")["state"] == "done"


def test_unchecked_behind_card_plans_and_renders_in_front(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    set_elements(layer_project, graphic())
    front = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    set_elements(layer_project, graphic(layer="behind"))
    behind = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert behind.layer == "front" and behind.plan == front.plan
    result = graphics(layer_project)
    item = result["elements"][0]
    assert item["layer"] == "front" and item["frames"] == 30
    info = ffmpeg.probe(layer_project / item["path"])
    assert (info.width, info.height, info.video_frames) == (
        front.plan.width, front.plan.height, 30,
    )
    assert fake_layers[0]["variables"]["text"] == "Small steps"
    assert read_receipt(layer_project / "graphics")["status"] == "verified"


def test_card_slot_edit_reuses_other_card_and_unchanged_project_is_noop(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict],
) -> None:
    second = graphic(id="agent-stat", start={"time": 1.5}, template="stat", slots={"end": 12})
    set_elements(layer_project, graphic(), second)
    graphics(layer_project)
    previous = read_receipt(layer_project / "graphics")["layer_checks"]
    target = layer_project / "graphics/agent-stat.mov"
    before = target.stat().st_mtime_ns
    assert graphics(layer_project)["noop"] and len(fake_layers) == 2
    set_elements(layer_project, graphic(slots={"text": "Changed steps"}), second)
    assert state(layer_project, "graphics")["state"] == "stale"
    assert not graphics(layer_project)["noop"]
    assert len(fake_layers) == 3 and fake_layers[-1]["variables"]["text"] == "Changed steps"
    assert target.stat().st_mtime_ns == before
    current = read_receipt(layer_project / "graphics")["layer_checks"]
    assert current["agent-stat"] == previous["agent-stat"]
    assert current["agent-tag"] == checked_layers[-1]
    assert len(checked_layers) == 3
    assert graphics(layer_project)["noop"]


@pytest.mark.parametrize("change", ["graphics", "internal", "template", "font"])
def test_card_digest_covers_style_template_and_all_fonts(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    templates = layer_project / "templates"
    shutil.copytree(cards.template_dir("text_tag"), templates / "text_tag")
    monkeypatch.setattr(cards, "_TEMPLATES", templates)
    bundled = layer_project / "fonts"
    shutil.copytree(fonts.FONTS_DIR, bundled)
    monkeypatch.setattr(fonts, "FONTS_DIR", bundled)
    set_elements(layer_project, graphic())
    before = graphics_inputs(ProjectContext(layer_project))
    if change == "internal":
        context = ProjectContext(layer_project)
        context.resolved_timeline()
        # These tokens are fixed in today's schema; simulate a future token revision.
        context.render_style().data["internal"]["key_text_top_px"] += 1
        after = graphics_snapshot(context).inputs
    elif change == "graphics":
        data = json.loads((layer_project / "style.json").read_bytes())
        data[change] = {"accent_color": "#008888"}
        write(layer_project / "style.json", data)
    elif change == "template":
        target = templates / "text_tag/template.html"
        target.write_bytes(target.read_bytes() + b"\n")
    else:
        target = bundled / "space-mono-regular.woff2"
        target.write_bytes(target.read_bytes() + b"\0")
    if change != "internal":
        after = graphics_inputs(ProjectContext(layer_project))
    assert before["graphic:agent-tag"] != after["graphic:agent-tag"]


def test_card_digest_ignores_caption_gap_but_tracks_card_width(layer_project: Path) -> None:
    set_elements(layer_project, graphic(), graphic(
        id="agent-stat", start={"time": 1.5}, template="stat", slots={"end": 12},
    ))
    before = graphics_snapshot(ProjectContext(layer_project)).inputs
    card_inputs = {key: digest for key, digest in before.items() if key.startswith("graphic:")}
    assert len(card_inputs) == 2

    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {})["caption_gap_px"] = 60
    write(layer_project / "style.json", data)
    after_gap = graphics_snapshot(ProjectContext(layer_project)).inputs
    assert {key: after_gap[key] for key in card_inputs} == card_inputs

    data["graphics"]["card_width_px"] = 900
    write(layer_project / "style.json", data)
    after_width = graphics_snapshot(ProjectContext(layer_project)).inputs
    assert all(after_width[key] != digest for key, digest in card_inputs.items())


@pytest.mark.parametrize("template,slots", [
    ("text_tag", {"text": "Small steps"}),
    ("specs_list", {"title": "Size", "rows": [
        {"name": "Fit", "value": "Small"}, {"name": "Count", "value": "Two"},
    ]}),
    ("flow", {"nodes": ["A", "B"]}),
])
def test_flow_look_changes_only_flow_digest(
    layer_project: Path, template: str, slots: dict,
) -> None:
    set_elements(layer_project, graphic(template=template, slots=slots, hold_s=2.5))
    before = graphics_snapshot(ProjectContext(layer_project)).inputs
    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {})["flow_look"] = "boxes"
    write(layer_project / "style.json", data)
    assert graphics_snapshot(ProjectContext(layer_project)).inputs == before
    data["graphics"]["flow_look"] = "tags"
    write(layer_project / "style.json", data)
    after = graphics_snapshot(ProjectContext(layer_project)).inputs
    assert (after["graphic:agent-tag"] != before["graphic:agent-tag"]) == (template == "flow")


@pytest.mark.parametrize("template", ["image_card", "image_compare", "icon_strike"])
def test_image_slot_bytes_rerender_only_the_consuming_card(
    layer_project: Path, fake_layers: list[dict], template: str,
) -> None:
    if template in cards.OPT_IN_KINDS:
        turn_on_card_kind(layer_project, template)

    def image(name: str, colour: str) -> None:
        ffmpeg.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=640x360", "-map", "0:v:0", "-frames:v", "1",
                    str(layer_project / name)], timeout_s=30)

    image("before.png", "red")
    image("after.png", "blue")
    slots = ({"icon": "before.png"} if template == "icon_strike" else
             {"image": "before.png"} if template == "image_card" else
             {"before": "before.png", "after": "after.png"})
    card = graphic(id="agent-image", start={"time": 0.1}, hold_s=2.2,
                   template=template, slots=slots)
    tag = graphic(start={"time": 2.4}, hold_s=0.5)
    set_elements(layer_project, card, tag)
    graphics(layer_project)
    unchanged = layer_project / "graphics/agent-tag.mov"
    modified = unchanged.stat().st_mtime_ns
    assert graphics(layer_project)["noop"]
    image("before.png" if template in {"image_card", "icon_strike"} else "after.png", "green")
    assert state(layer_project, "graphics")["state"] == "stale"
    assert not graphics(layer_project)["noop"]
    assert len(fake_layers) == 3 and unchanged.stat().st_mtime_ns == modified


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
@pytest.mark.parametrize("template,name", [
    ("image_card", "image"), ("image_compare", "before"), ("image_compare", "after"),
    ("comment_drop", "avatar"),
    ("icon_strike", "icon"),
])
def test_image_slots_refuse_paths_outside_project(
    layer_project: Path, escape: str, template: str, name: str,
) -> None:
    """Every image slot must reject paths and symlinks reaching outside the project."""
    if template in cards.OPT_IN_KINDS:
        turn_on_card_kind(layer_project, template)
    outside = layer_project.parent / f"{layer_project.name}-external.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=red:s=640x360", "-map", "0:v:0", "-frames:v", "1",
                str(outside)], timeout_s=30)
    shutil.copyfile(outside, layer_project / "inside.png")
    if escape == "parent":
        image_path = f"../{outside.name}"
    elif escape == "absolute":
        image_path = str(outside)
    else:
        (layer_project / "escape.png").symlink_to(outside)
        image_path = "escape.png"
    slots = ({"icon": "inside.png"} if template == "icon_strike" else
             {"text": "Send it", "avatar": "inside.png"} if template == "comment_drop" else
             {"image": "inside.png"} if template == "image_card" else
             {"before": "inside.png", "after": "inside.png"})
    slots[name] = image_path
    set_elements(layer_project, graphic(template=template, slots=slots, hold_s=2.2))
    with pytest.raises(ValidationError) as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == f"graphic.slots.{name}"
    assert caught.value.detail == "use a file inside the project folder"
    assert not (layer_project / "graphics").exists()


@pytest.mark.parametrize("avatar", [None, ""])
def test_comment_drop_snapshot_skips_empty_optional_avatar(
    layer_project: Path, avatar: str | None,
) -> None:
    turn_on_card_kind(layer_project, "comment_drop")
    slots = {"text": "Send it", "initials": "MC"}
    if avatar is not None:
        slots["avatar"] = avatar
    set_elements(layer_project, graphic(template="comment_drop", slots=slots, hold_s=1.6))
    snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert snapshot.images == () and snapshot.plan.variables["avatar_src"] == ""


def test_comment_drop_snapshot_reads_external_brand_logo(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    turn_on_card_kind(layer_project, "comment_drop")
    outside = layer_project.parent / f"{layer_project.name}-brand.png"

    def write_logo(colour: str) -> None:
        ffmpeg.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=96x96", "-frames:v", "1", str(outside)], timeout_s=30)

    write_logo("red")
    style_file = layer_project / "style.json"
    style = json.loads(style_file.read_text())
    style.setdefault("brand", {})["logo"] = str(outside)
    style_file.write_text(json.dumps(style))
    set_elements(layer_project, graphic(template="comment_drop", slots={"text": "Send it"},
                                        hold_s=1.6))
    context = ProjectContext(layer_project)
    first = graphics_snapshot(context).graphics[0]
    assert first.images == (("avatar.png", context.read(str(outside))),)
    graphics(layer_project)
    assert fake_layers[0]["variables"]["avatar_src"] == "avatar.png"
    write_logo("blue")
    assert context.read(str(outside)) == first.images[0][1]
    second = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert second.images != first.images and second.digest != first.digest


def test_image_list_resolves_project_paths_and_tracks_each_image(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    enable_card_kind(layer_project, "jiggle_delete")
    for name, colour in (("one.png", "red"), ("two.jpg", "blue")):
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=180x180", "-frames:v", "1",
                    str(layer_project / name)], timeout_s=30)
    set_elements(layer_project, graphic(template="jiggle_delete", hold_s=2.5,
                                        slots={"items": ["One", "Two"], "remove": [1],
                                               "images": ["one.png", "two.jpg"]}))
    first = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert [name for name, _ in first.images] == ["item-1.png", "item-2.jpg"]
    assert first.shot_sources == ()
    graphics(layer_project)
    assert fake_layers
    ffmpeg.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                "color=c=green:s=180x180", "-frames:v", "1",
                str(layer_project / "two.jpg")], timeout_s=30)
    after = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert first.digest != after.digest


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_image_list_refuses_paths_outside_project(layer_project: Path, escape: str) -> None:
    enable_card_kind(layer_project, "jiggle_delete")
    outside = layer_project.parent / f"{layer_project.name}-external.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=red:s=180x180", "-frames:v", "1", str(outside)], timeout_s=30)
    shutil.copyfile(outside, layer_project / "inside.png")
    if escape == "parent":
        image_path = f"../{outside.name}"
    elif escape == "absolute":
        image_path = str(outside)
    else:
        (layer_project / "escape.png").symlink_to(outside)
        image_path = "escape.png"
    set_elements(layer_project, graphic(template="jiggle_delete", hold_s=2.5, slots={
        "items": ["One", "Two"], "remove": [1], "images": ["inside.png", image_path],
    }))
    with pytest.raises(ValidationError) as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == "graphic.slots.images[1]"
    assert caught.value.detail == "use a file inside the project folder"


@pytest.mark.parametrize("box", [
    {"x": 0, "y": 900, "width": 1040, "height": 360},
    {"x": -1, "y": 900, "width": 1040, "height": 360},
    {"x": 41, "y": 900, "width": 1040, "height": 360},
    {"x": 0, "y": 1561, "width": 1040, "height": 360},
    {"x": 0, "y": 900, "width": 0, "height": 360},
])
def test_future_card_box_is_honoured_only_inside_frame(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, box: dict[str, int],
) -> None:
    set_elements(layer_project, graphic())
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    item = timeline.elements[0]
    monkeypatch.setattr(context, "resolved_timeline", lambda: replace(
        timeline, elements=(replace(item, payload={**item.payload, "box": box}),),
    ))
    if box["x"] == 0 and box["y"] == 900 and box["width"] > 0:
        snapshot = graphics_snapshot(context)
        placement = snapshot.graphics[0].placement()
        assert {key: placement[key] for key in box} == box
    else:
        with pytest.raises(ValidationError, match="graphic.box"):
            graphics_snapshot(context)


@pytest.mark.render
def test_real_text_tag_graphic_layer(layer_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise a real card layer independently of the synthetic renderer fixture."""
    monkeypatch.undo()
    set_elements(layer_project, graphic())
    result = graphics(layer_project)
    item = result["elements"][0]
    info = ffmpeg.probe(layer_project / item["path"])
    assert (info.width, info.height, info.video_frames) == (
        item["width"], item["height"], item["frames"],
    )
    assert info.pix_fmt.startswith("yuva444p")
    plan = graphics_snapshot(ProjectContext(layer_project)).graphics[0].plan
    alpha = _alpha_frames(layer_project / item["path"], plan)
    assert alpha[plan.frames // 2].max() > 0 and alpha[plan.frames // 2].min() == 0


def test_title_falls_back_to_hook_duration_and_records_box_window(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict],
) -> None:
    set_elements(layer_project, title())
    result = graphics(layer_project)
    assert result["elements"][0]["id"] == "agent-title"
    item = result["elements"][0]
    assert (item["start_frame"], item["frames"], item["layer"]) == (6, 33, "behind")
    assert (item["width"], item["x"]) == (940, 70)
    info = ffmpeg.probe(layer_project / item["path"])
    assert (info.video_frames, info.width, info.height) == (33, item["width"], item["height"])
    assert (info.color_space, info.color_transfer, info.color_primaries) == ("bt709",) * 3
    record = read_receipt(layer_project / "graphics")
    assert record["elements"] == result["elements"]
    assert record["layer_checks"] == {"agent-title": checked_layers[0]}
    assert [output["path"] for output in record["outputs"]] == ["graphics/agent-title.mov"]
    variables = fake_layers[0]["variables"]
    assert json.loads(variables["lines"]) == ["Small", "steps"]
    assert variables["key"] == "steps" and variables["look"] == "quiet_serif"
    assert state(layer_project, "graphics")["state"] == "done"


def test_title_applies_text_rules_and_tracks_vocabulary(
    layer_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    set_words(["SparrowLight"])
    set_elements(layer_project, title(lines=["sparrowlight fifty percent"],
                                      accent="five dollars"))
    before = graphics_snapshot(ProjectContext(layer_project))
    variables = before.titles[0].plan.variables
    assert json.loads(variables["lines"]) == ["SparrowLight 50%"]
    assert variables["key"] == "$5"
    set_words(["SparrowLight", "PaperKettle"])
    after = graphics_snapshot(ProjectContext(layer_project))
    assert before.titles[0].digest != after.titles[0].digest


def test_title_accent_uses_converted_line_text(
    layer_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    set_elements(layer_project, title(lines=["five dollars"], accent="dollars"))
    plan = graphics_snapshot(ProjectContext(layer_project)).titles[0].plan
    assert json.loads(plan.variables["lines"]) == ["$5"]
    assert plan.variables["key"] == "$5"


@pytest.mark.parametrize("window,frames", [({"hold_s": 0.5}, 15), ({"end": {"time": 1}}, 24)])
def test_title_explicit_frame_window_overrides_style_duration(
    layer_project: Path, window: dict[str, Any], frames: int,
) -> None:
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"]["duration_s"] = 0
    write(layer_project / "style.json", data)
    set_elements(layer_project, title(**window))
    item = graphics(layer_project)["elements"][0]
    assert item["frames"] == frames
    assert ffmpeg.probe(layer_project / item["path"]).video_frames == frames


@pytest.mark.parametrize("section", ["hook", "captions"])
def test_style_changes_rerender_only_the_consuming_stage(
    layer_project: Path, fake_layers: list[dict], section: str,
) -> None:
    set_elements(layer_project, title())
    captions(layer_project)
    graphics(layer_project)
    other = "captions" if section == "hook" else "graphics"
    previous = (layer_project / other / "receipt.json").read_bytes()
    data = json.loads((layer_project / "style.json").read_bytes())
    if section == "hook":
        data["hook"]["style"] = "paper_title"
    else:
        data["captions"]["highlight_style"] = "color"
    write(layer_project / "style.json", data)
    affected = "graphics" if section == "hook" else "captions"
    assert state(layer_project, affected)["state"] == "stale"
    assert state(layer_project, other)["state"] == "done"
    cap_result = captions(layer_project)
    title_result = graphics(layer_project)
    assert cap_result["noop"] == (section == "hook")
    assert title_result["noop"] == (section == "captions")
    assert (layer_project / other / "receipt.json").read_bytes() == previous
    assert len(fake_layers) == 3


def test_title_edit_renders_only_the_changed_element(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict],
) -> None:
    second = title(id="agent-second", start={"time": 1.5}, lines=["More words"])
    set_elements(layer_project, title(), second)
    graphics(layer_project)
    previous = read_receipt(layer_project / "graphics")["layer_checks"]
    target = layer_project / "graphics/agent-second.mov"
    before = target.stat().st_mtime_ns
    set_elements(layer_project, title(lines=["Changed", "steps"]), second)
    graphics(layer_project)
    assert len(fake_layers) == 3
    assert target.stat().st_mtime_ns == before
    assert read_receipt(layer_project / "graphics")["layer_checks"]["agent-second"] == (
        previous["agent-second"]
    )
    assert len(checked_layers) == 3
    assert state(layer_project, "graphics")["state"] == "done"


@pytest.mark.parametrize("kind", ["title", "graphic"])
def test_empty_stage_and_removal_preserve_layer_ownership(
    layer_project: Path, fake_layers: list[dict], checked_layers: list[dict], kind: str,
) -> None:
    element = title() if kind == "title" else graphic()
    assert graphics(layer_project)["elements"] == []
    assert graphics(layer_project)["noop"] and fake_layers == []
    set_elements(layer_project, element)
    graphics(layer_project)
    assert read_receipt(layer_project / "graphics")["layer_checks"] == {
        element["id"]: checked_layers[0],
    }
    set_elements(layer_project)
    assert graphics(layer_project)["elements"] == []
    assert read_receipt(layer_project / "graphics")["outputs"][0]["path"] == (
        f"graphics/{element['id']}.mov"
    )
    assert read_receipt(layer_project / "graphics")["layer_checks"] == {}
    assert state(layer_project, "graphics")["state"] == "done"
    set_elements(layer_project, element)
    assert not graphics(layer_project)["noop"]
    assert len(fake_layers) == 2


@pytest.mark.parametrize("kind", ["title", "graphic"])
def test_force_refuses_unowned_layer(layer_project: Path, kind: str) -> None:
    element = title() if kind == "title" else graphic()
    set_elements(layer_project, element)
    target = layer_project / f"graphics/{element['id']}.mov"
    target.parent.mkdir()
    target.write_bytes(b"unowned")
    with pytest.raises(OutputPathError):
        graphics(layer_project, force=True)
    assert target.read_bytes() == b"unowned"


def test_force_renders_again_but_refuses_changed_title(
    layer_project: Path, fake_layers: list[dict],
) -> None:
    set_elements(layer_project, title())
    result = graphics(layer_project)
    assert graphics(layer_project)["noop"]
    assert not graphics(layer_project, force=True)["noop"] and len(fake_layers) == 2
    target = layer_project / result["elements"][0]["path"]
    target.write_bytes(b"hand edited")
    with pytest.raises(OutputPathError):
        graphics(layer_project, force=True)
    assert target.read_bytes() == b"hand edited"


def test_none_hook_refuses_authored_title(layer_project: Path) -> None:
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"]["style"] = "none"
    write(layer_project / "style.json", data)
    set_elements(layer_project, title())
    with pytest.raises(ValidationError, match="hook.style is none"):
        graphics(layer_project)
    assert not (layer_project / "graphics").exists()


@pytest.mark.parametrize("element_id", ["agent-../escape", "agent-..\\escape"])
def test_title_id_cannot_escape_output_folder(layer_project: Path, element_id: str) -> None:
    set_elements(layer_project, title(id=element_id))
    with pytest.raises(ValidationError, match="portable file name"):
        graphics(layer_project)
    assert not (layer_project / "graphics").exists()


def test_mid_render_hook_edit_keeps_snapshot_and_reports_stale(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_elements(layer_project, title())
    before = graphics_inputs(ProjectContext(layer_project))
    render = hyperframes.render_layer

    def edit(composition: Path, out: Path, *, variables: Mapping[str, object],
             **kwargs: Any) -> hyperframes.LayerResult:
        assert variables["look"] == "quiet_serif"
        data = json.loads((layer_project / "style.json").read_bytes())
        data["hook"]["style"] = "bold_band"
        write(layer_project / "style.json", data)
        return render(composition, out, variables=variables, **kwargs)

    monkeypatch.setattr(hyperframes, "render_layer", edit)
    graphics(layer_project)
    assert read_receipt(layer_project / "graphics")["inputs"] == before
    assert state(layer_project, "graphics")["state"] == "stale"


def test_graphics_cli_defaults_json_force_and_refusals(
    layer_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    fake_layers: list[dict],
) -> None:
    set_elements(layer_project, title())
    monkeypatch.chdir(layer_project)
    assert cli.main(["graphics", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["elements"]) == 1
    assert cli.main(["graphics", str(layer_project), "--force"]) == 0
    assert "Rendered 1 title layers" in capsys.readouterr().out and len(fake_layers) == 2
    set_elements(layer_project, title(start={"time": 2.9}))
    assert cli.main(["graphics", "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "timeline_refused"


@pytest.mark.parametrize("template,slots,changed", [
    ("text_tag", {"text": "Hello"}, False),
    ("flow", {"nodes": ["One", "Two"]}, False),
    ("specs_list", {"title": "Plan", "rows": [
        {"name": "First", "value": "12"}, {"name": "Second", "value": "25"}]}, True),
    ("stat", {"end": 37}, True),
])
def test_specs_rows_digest_scope(layer_project: Path, template: str, slots: dict,
                                changed: bool) -> None:
    set_elements(layer_project, graphic(template=template, slots=slots, hold_s=2.5))
    before = graphics_snapshot(ProjectContext(layer_project)).graphics[0].digest
    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {})["specs_rows"] = "on_cue"
    write(layer_project / "style.json", data)
    after = graphics_snapshot(ProjectContext(layer_project)).graphics[0].digest
    assert (before != after) is changed


@pytest.mark.parametrize("width,boxes,reason", [
    (960, [(300, 800, 480, 600)],
     "no position below the 280 px line keeps the head clear for the fly"),
    (400, [(40, 650, 160, 120), (850, 650, 160, 120)],
     "neither swept box keeps the required head clearance"),
])
def test_card_motion_stage_accepts_ordinary_placement_fallback(
    layer_project: Path, monkeypatch, width, boxes, reason,
):
    from test_planner_cards import synthetic_faces

    from yap_studio.faces.clearance import face_boxes_per_frame, face_clearance_px, gap_px, head_box
    from yap_studio.faces.model import Box, FaceBox
    from yap_studio.stages import graphics as module

    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {}).update(card_motion="none", card_width_px=width)
    write(layer_project / "style.json", data)
    faces = replace(synthetic_faces(frames=90),
                    frames=(tuple(FaceBox(Box(*box), 1, "detected") for box in boxes),) * 90)
    monkeypatch.setattr(module, "picture_faces", lambda *args: faces)
    monkeypatch.setattr(module, "behind_face_windows", lambda *args: ((0, 90),))
    set_elements(layer_project, graphic(template="quote",
                                        slots={"quote": "Ship it", "name": "Sample"},
                                        start={"time": 0}, hold_s=3))
    plain = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    data["graphics"]["card_motion"] = "streak"
    write(layer_project / "style.json", data)
    snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert (plain.plan.width, plain.plan.height, plain.plan.y) == (width + 80, 360, 0)
    assert snapshot.plan.motion is None and snapshot.clearance["status"] == "clear"
    assert snapshot.clearance["card_motion"] == {"status": "none", "reason": reason}
    assert snapshot.plan == plain.plan and snapshot.layer == plain.layer
    clearance = {key: value for key, value in snapshot.clearance.items() if key != "card_motion"}
    assert clearance == plain.clearance
    box = Box(snapshot.plan.x, snapshot.plan.y, snapshot.plan.width, snapshot.plan.height)
    margin = face_clearance_px(ProjectContext(layer_project).render_style().data)
    assert all(gap_px(box, head_box(face)) >= margin
               for frame in face_boxes_per_frame(faces)[:90] for face in frame)
    restored = graphics_snapshot(ProjectContext(layer_project), placements=[snapshot.placement()])
    assert restored.graphics[0].plan == snapshot.plan
    assert restored.graphics[0].clearance == snapshot.clearance


def test_card_motion_none_matches_main_cache_and_placement(
    layer_project: Path, fake_layers, monkeypatch,
):
    from yap_studio.planner.cards import _render_key
    from yap_studio.style import resolve

    # Pins from main 3abf21f5, on Darwin.
    # Regenerate in a pristine main archive with the lane environment sourced.
    # Copy this test there; omit the card_motion assignment if main lacks the field.
    # Print actual_key, actual_placement and record["inputs"]["graphic:agent-tag"]
    # before the assertions, then run the archive's own uv:
    # uv run pytest tests/test_stage_graphics.py -k none_matches_main -s
    # Hold the historical renderer identity fixed so release bumps keep this baseline valid.
    monkeypatch.setattr(hyperframes, "HYPERFRAMES_VERSION", "0.8.184")
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_motion"] = "none"
    slots = json.loads((cards.template_dir("quote") / "preview.json").read_text())["slots"]
    plan = cards.card_layer_plan("quote", slots, style, frames=90)
    actual_key = _render_key(plan)
    write(layer_project / "style.json", style.unbranded_data)
    set_elements(layer_project, graphic(template="quote", slots=slots,
                                        start={"time": 0}, hold_s=3))
    result = graphics(layer_project)
    record = read_receipt(layer_project / "graphics")
    actual_placement = json.dumps(record["elements"][0], sort_keys=True, separators=(",", ":"))
    assert record["status"] == "verified" and not result["noop"]
    assert len(fake_layers) == 1
    assert record["elements"] == result["elements"]
    # Pinned sha256 render key, not a secret.
    assert actual_key == (
        "6b04a7f2415eefc0bea5404ac1817276999e91481fa60b296ccf478290cb21a0"  # gitleaks:allow
    )
    assert actual_placement == (
        '{"clearance":{"gap_px":null,"hair":{"above":0.6,"sides":0.15},"margin_px":24,'
        '"reason":"fake project has no face measurements","status":"unchecked"},'
        '"frames":90,"height":360,"id":"agent-tag","input_sha256":'
        '"e9702b2cf9b284a0a505b64b191432991df82a3584a43f5219dc0a8e9b5c0e00",'
        '"kind":"graphic","layer":"front","path":"graphics/agent-tag.mov",'
        '"start_frame":0,"width":1040,"x":20,"y":280}'
    )
    assert record["inputs"]["graphic:agent-tag"] == (
        "e9702b2cf9b284a0a505b64b191432991df82a3584a43f5219dc0a8e9b5c0e00"
    )


def test_renderer_change_invalidates_verified_receipts(
    layer_project: Path, fake_layers, monkeypatch,
):
    set_elements(layer_project, graphic())
    graphics(layer_project)
    before = read_receipt(layer_project / "graphics")
    monkeypatch.setattr(hyperframes, "HYPERFRAMES_VERSION", "0.0.0")
    result = graphics(layer_project)
    after = read_receipt(layer_project / "graphics")
    assert not result["noop"] and len(fake_layers) == 2
    assert before["status"] == after["status"] == "verified"
    assert before["inputs"] != after["inputs"]


@pytest.mark.parametrize("fallback", [False, True])
def test_card_motion_changes_invalidate_verified_receipts(
    layer_project: Path, monkeypatch, fallback,
):
    from test_planner_cards import synthetic_faces

    from yap_studio.faces.model import Box
    from yap_studio.stages import graphics as module

    face = Box(300, 800, 480, 600) if fallback else None
    monkeypatch.setattr(module, "picture_faces", lambda *args: synthetic_faces(face, frames=90))
    monkeypatch.setattr(module, "behind_face_windows", lambda *args: ((0, 90),))
    set_elements(layer_project, graphic(template="quote",
                                        slots={"quote": "Ship it", "name": "Sample"},
                                        start={"time": 0}, hold_s=3))
    rendered = []

    def render(snapshot, work, out):
        rendered.append(snapshot.plan)
        out.write_bytes(f"synthetic card {len(rendered)}".encode())
        box = snapshot.plan.layer_box()
        return hyperframes.LayerResult(out, snapshot.plan.frames, box["width"], box["height"],
                                       snapshot.plan.frames / snapshot.plan.fps, checks={})

    monkeypatch.setattr(module, "_render_card", render)
    data = json.loads((layer_project / "style.json").read_bytes())
    records = []
    for index, motion in enumerate(("none", "streak", "none")):
        data.setdefault("graphics", {})["card_motion"] = motion
        write(layer_project / "style.json", data)
        result = graphics(layer_project)
        assert not result["noop"]
        assert len(rendered) == index + 1
        record = read_receipt(layer_project / "graphics")
        assert record["status"] == "verified"
        assert record["elements"] == result["elements"]
        note = record["elements"][0]["clearance"].get("card_motion")
        if motion == "streak":
            assert note["status"] == ("none" if fallback else "streak")
            assert (rendered[-1].motion is None) == fallback
            if fallback:
                assert "280 px line" in note["reason"]
        else:
            assert note is None and rendered[-1].motion is None
        if records:
            context = ProjectContext(layer_project)
            restored = graphics_snapshot(context, placements=records[-1]["elements"]).graphics[0]
            fresh = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
            assert restored.plan == fresh.plan and restored.clearance == fresh.clearance
            assert restored.digest == fresh.digest
            assert record["inputs"] != records[-1]["inputs"]
        records.append(record)
        with monkeypatch.context() as saved:
            saved.setattr(module, "picture_faces", lambda *args: pytest.fail("remeasured faces"))
            assert graphics(layer_project)["noop"]
        assert len(rendered) == index + 1
    assert records[0]["inputs"] == records[-1]["inputs"]


@pytest.mark.parametrize("kind", ["specs_list", "stat"])
def test_default_cued_row_motion_uses_a_band(layer_project: Path, kind):
    from test_cards import ROWS_ON_CUE_SLOTS

    from yap_studio.style import resolve

    style = resolve("preset:clean_minimal").unbranded_data
    style["graphics"].update(card_motion="streak", specs_rows="on_cue")
    write(layer_project / "style.json", style)
    set_elements(layer_project, graphic(template=kind, slots=ROWS_ON_CUE_SLOTS[kind],
                                        start={"time": 0}, hold_s=3))
    snapshot, = graphics_snapshot(ProjectContext(layer_project)).graphics
    plan = snapshot.plan
    assert plan.motion == {"side": "right", "enter": True}
    assert plan.y < 280 <= plan.y + plan.variables["glow_pad_px"] <= 281
    assert snapshot.clearance["card_motion"]["status"] == "streak"
    record = snapshot.placement()
    assert record["x"] == 0 and record["width"] == 1080 and record["y"] == plan.y


def test_card_motion_requested_setting_is_hashed_for_fallback(layer_project: Path, monkeypatch):
    from test_planner_cards import synthetic_faces

    from yap_studio.faces.model import Box
    from yap_studio.stages import graphics as module

    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {})["card_motion"] = "streak"
    write(layer_project / "style.json", data)
    monkeypatch.setattr(module, "picture_faces", lambda *args: synthetic_faces(
        Box(300, 800, 480, 600), frames=90))
    monkeypatch.setattr(module, "behind_face_windows", lambda *args: ((0, 90),))
    set_elements(layer_project, graphic(template="quote",
                                        slots={"quote": "Ship it", "name": "Sample"},
                                        start={"time": 0}, hold_s=3))
    without_request = []
    original = module.projection_sha256

    def digest(projection):
        if "graphics" in projection:
            graphics_values = {key: value for key, value in projection["graphics"].items()
                               if key != "card_motion"}
            without_request.append(original({**projection, "graphics": graphics_values}))
        return original(projection)

    monkeypatch.setattr(module, "projection_sha256", digest)
    snapshot = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert snapshot.plan.motion is None
    assert len(without_request) == 1 and snapshot.digest != without_request[0]


def test_card_motion_digest_scope_and_band(layer_project: Path, monkeypatch) -> None:
    set_elements(layer_project, graphic(template="text_tag", slots={"text": "Hello"}, hold_s=2.5))
    original = ProjectContext.render_style

    def legacy(context):
        style = original(context)
        style.data["graphics"].pop("card_motion", None)
        return style

    with monkeypatch.context() as old:
        old.setattr(ProjectContext, "render_style", legacy)
        before = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    none = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert none.digest == before.digest
    data = json.loads((layer_project / "style.json").read_bytes())
    data.setdefault("graphics", {})["card_motion"] = "streak"
    write(layer_project / "style.json", data)
    streak = graphics_snapshot(ProjectContext(layer_project)).graphics[0]
    assert streak.digest != none.digest
    assert streak.plan.motion is not None
    assert streak.placement()["x"] == 0 and streak.placement()["width"] == 1080
    restored = graphics_snapshot(ProjectContext(layer_project), placements=[streak.placement()])
    assert restored.graphics[0].digest == streak.digest
    assert restored.graphics[0].plan == streak.plan
