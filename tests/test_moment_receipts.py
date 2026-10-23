"""Synthetic assembly snapshots bind moment style, anchors and renderer version."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_assemble import fake_layers as fake_layers
from test_assemble import project as project
from test_assemble import refuse_unexpected_matte_work as refuse_unexpected_matte_work
from test_stage_captions import set_elements

from yap_studio.project.receipts import StageRun, projection_sha256
from yap_studio.project.status import ProjectContext
from yap_studio.stages import assemble as stage
from yap_studio.stages.captions import captions
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.stages.timeline import TimelineRefusal


def prepare(root: Path) -> None:
    data = json.loads((root / "project.json").read_bytes())
    data["style"] = "style.json"
    (root / "project.json").write_text(json.dumps(data))
    (root / "style.json").write_text(json.dumps({"schema_version": 1,
        "moments": {"styles": ["color_pop", "sticker_freeze", "outline"]}}))
    set_elements(root, {"id": "agent-moment", "kind": "moment", "lane": "agent",
                        "style": "outline", "start": {"time": 1}, "hold_s": 1})
    captions(root)
    graphics(root)
    music(root)


@pytest.mark.parametrize("style", ["color_pop", "background_blur", "spotlight",
                                   "sticker_freeze", "outline"])
def test_old_projection_binds_moments_version_and_accent(project: Path, style: str,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    prepare(project)
    data = json.loads((project / "style.json").read_text())
    data["moments"]["styles"] = [style]
    (project / "style.json").write_text(json.dumps(data))
    set_elements(project, {"id": "agent-moment", "kind": "moment", "lane": "agent",
                           "style": style, "start": {"time": 1}, "hold_s": 1})
    context = ProjectContext(project)
    first = stage.assembly_snapshot(context)
    assert first.inputs["picture:moments"] == projection_sha256({
        "version": stage.MOMENT_RENDER_VERSION,
        "moments": context.render_style().data["moments"],
        "accent_color": context.render_style().data["graphics"]["accent_color"],
        "elements": [first.timeline.elements[0].to_dict()],
    })
    monkeypatch.setattr(stage, "MOMENT_RENDER_VERSION", stage.MOMENT_RENDER_VERSION + 1)
    versioned = stage.assembly_snapshot(ProjectContext(project))
    assert versioned.inputs["picture:moments"] != first.inputs["picture:moments"]
    data = json.loads((project / "style.json").read_bytes())
    data["graphics"] = {"accent_color": "#FF0000"}
    (project / "style.json").write_text(json.dumps(data))
    graphics(project)
    after = stage.assembly_snapshot(ProjectContext(project))
    assert after.inputs["picture:moments"] != versioned.inputs["picture:moments"]
    assert after.inputs["timeline"] == versioned.inputs["timeline"]
    data = json.loads((project / "timeline/agent.json").read_bytes())
    data["elements"][0]["hold_s"] = 1.2
    (project / "timeline/agent.json").write_text(json.dumps(data))
    moved = stage.assembly_snapshot(ProjectContext(project))
    assert moved.inputs["picture:moments"] != after.inputs["picture:moments"]
    assert moved.inputs["timeline"] != after.inputs["timeline"]


def test_snapshot_keeps_moments_and_cutout_inputs(project: Path) -> None:
    prepare(project)
    style = json.loads((project / "style.json").read_bytes())
    style["broll"] = {"cutout": "screen"}
    (project / "style.json").write_text(json.dumps(style))
    snapshot = stage.assembly_snapshot(ProjectContext(project))
    assert "picture:moments" in snapshot.inputs
    assert snapshot.inputs["picture:cutout"] == projection_sha256("screen")


@pytest.mark.parametrize("change", ["off", "overlap", "length"])
def test_editorial_override_cannot_bypass_moment_rules(project: Path, change: str) -> None:
    prepare(project)
    data = json.loads((project / "timeline/agent.json").read_bytes())
    item = data["elements"][0]
    if change == "off":
        item["style"] = "background_blur"
    elif change == "length":
        item["hold_s"] = 4
    else:
        data["elements"].append({**item, "id": "agent-other"})
    (project / "timeline/agent.json").write_text(json.dumps(data))
    with pytest.raises(TimelineRefusal):
        stage.assembly_snapshot(ProjectContext(project), allow_violations=True)


def test_clone_projection_adds_only_clone_version(project: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    prepare(project)
    style = json.loads((project / "style.json").read_text())
    style["moments"]["styles"].append("clone_flash")
    (project / "style.json").write_text(json.dumps(style))
    item = {"id": "agent-moment", "kind": "moment", "lane": "agent",
            "style": "clone_flash", "start": {"word": "t1:0"}, "hold_s": 1.2}
    set_elements(project, item)
    context = ProjectContext(project)
    first = stage.assembly_snapshot(context)
    assert stage.CLONE_RENDER_VERSION == 2
    assert stage.MOMENT_RENDER_VERSION == 3
    assert first.inputs["picture:moments"] == projection_sha256({
        "version": stage.MOMENT_RENDER_VERSION,
        "moments": context.render_style().data["moments"],
        "accent_color": context.render_style().data["graphics"]["accent_color"],
        "elements": [first.timeline.elements[0].to_dict()],
        "clone_version": stage.CLONE_RENDER_VERSION,
    })
    monkeypatch.setattr(stage, "CLONE_RENDER_VERSION", stage.CLONE_RENDER_VERSION + 1)
    changed = stage.assembly_snapshot(ProjectContext(project))
    assert changed.inputs["picture:moments"] != first.inputs["picture:moments"]


def test_clone_refusal_invalidates_version_one_final_receipt(project: Path) -> None:
    prepare(project)
    style = json.loads((project / "style.json").read_text())
    style["moments"]["styles"].append("clone_flash")
    (project / "style.json").write_text(json.dumps(style))
    set_elements(project, {"id": "agent-moment", "kind": "moment", "lane": "agent",
                           "style": "clone_flash", "start": {"word": "t1:0"}, "hold_s": 1.2})
    context = ProjectContext(project)
    snapshot = stage.assembly_snapshot(context)
    old_projection = projection_sha256({
        "version": stage.MOMENT_RENDER_VERSION,
        "moments": context.render_style().data["moments"],
        "accent_color": context.render_style().data["graphics"]["accent_color"],
        "elements": [snapshot.timeline.elements[0].to_dict()],
        "clone_version": 1,
    })
    old_inputs = {**snapshot.inputs, "picture:moments": old_projection}
    staged = project / "old-final.txt"
    staged.write_text("synthetic final from clone version 1\n")
    with StageRun(project, "final") as run:
        for name, digest in old_inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(project / "final/old-final.txt", staged)
        run.finish()
    cached = StageRun(project, "final")
    assert cached.up_to_date(old_inputs)
    assert snapshot.inputs["picture:moments"] != old_projection
    assert not cached.up_to_date(snapshot.inputs)
