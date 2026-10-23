"""Background media contracts and word-anchored timeline rules."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from test_assemble_picture import overlay_image
from test_timeline import add, insert, overlay, write
from test_timeline import project as project

from yap_studio import receipt
from yap_studio.project.model import ProjectFileError, json_schema, parse_document
from yap_studio.project.status import ProjectContext
from yap_studio.stages.assemble import assembly_snapshot
from yap_studio.stages.timeline import TimelineRefusal, resolve_timeline


def background(project: Path, **changes: object) -> dict:
    return {**insert(project), "id": "broll-page", "kind": "background", **changes}


@pytest.mark.parametrize("lane", ["agent", "broll"])
def test_background_schema_and_word_anchors(project: Path, lane: str) -> None:
    item = background(project, id=f"{lane}-page", lane=lane)
    data = {"schema": "yap-studio/timeline@1", "elements": [item]}
    jsonschema.validate(data, json_schema("timeline-fragment"))
    result = add(project, item, lane=lane).require_valid()
    assert (result.elements[0].start_frame, result.elements[0].end_frame) == (66, 108)
    assert result.elements[0].payload["box_px"] == [0, 0, 1080, 1920]
    assert result.elements[0].payload["fit"] == "cover"
    assert resolve_timeline(project).elements[0].payload["box_px"] == [0, 0, 1080, 1920]


@pytest.mark.parametrize("change", [
    {"media": None}, {"fit": "crop"}, {"placement": "headroom"},
    {"motion": "drift"}, {"audio": "mute"},
    {"layout": "middle"}, {"blend": "soft"}, {"enter": "fade"},
    {"media": {"path": "page.png", "sha256": "a" * 64, "crop": [0, 0, 10, 10]}},
    {"end": {"time": 3}, "hold_s": 1},
])
@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
def test_invalid_background_fields_fail_loader_and_schema(
    project: Path, change: dict, name: str,
) -> None:
    item = background(project, **change)
    data = {"schema": "yap-studio/timeline@1", "elements": [item]}
    if name == "timeline":
        data["words_sha256"] = {}
    with pytest.raises(ProjectFileError):
        parse_document(name, data, lane="broll")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, json_schema(name))


def test_background_media_rules_and_visual_overlap(project: Path) -> None:
    item = background(project)
    item["media"]["path"] = "missing.mov"
    result = add(project, item)
    assert not result.valid
    assert any(f.rule == "background_media" for f in result.findings)
    first = background(project)
    add(project, first).require_valid()
    second = background(project, id="broll-second")
    assert any(f.rule == "visual_overlap" for f in add(project, second).findings)


@pytest.mark.parametrize("kind", ["zoom", "insert"])
def test_background_over_zoom_or_insert_is_refused_even_with_override(
    project: Path, kind: str,
) -> None:
    add(project, background(project)).require_valid()
    other = ({"id": "agent-zoom", "kind": "zoom", "lane": "agent", "start": {"time": 2.5},
              "end": {"time": 3}, "peak_scale": 1.04, "center": [0.5, 0.5]}
             if kind == "zoom" else
             insert(project, id="agent-clip", lane="agent", start={"time": 2.5},
                    end={"time": 3}))
    result = add(project, other, lane="agent")
    assert any(f.rule == "background_overlap" and f.elements == (other["id"], "broll-page")
               for f in result.findings)
    # Write the lane by hand, as an agent could, and ask assembly to override.
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
                                            "words_sha256": {}, "elements": [other]})
    with pytest.raises(TimelineRefusal, match="backgrounds cannot overlap zooms or inserts"):
        assembly_snapshot(ProjectContext(project), allow_violations=True)


@pytest.mark.parametrize("extra", [
    {"media_fit": "split"}, {"letters": {"text": "Dog"}},
])
def test_green_screen_background_refuses_split_and_letters_inserts(
    project: Path, extra: dict,
) -> None:
    add(project, background(project, layout="bottom")).require_valid()
    other = insert(project, id="agent-clip", lane="agent", start={"time": 2.5},
                   end={"time": 3})
    if "media_fit" in extra:
        other["media"]["fit"] = extra["media_fit"]
    else:
        other["letters"] = extra["letters"]
    result = add(project, other, lane="agent")
    assert any(f.rule == "background_overlap" and f.elements == (other["id"], "broll-page")
               for f in result.findings)


def test_background_may_sit_under_an_overlay(project: Path) -> None:
    add(project, background(project)).require_valid()
    result = add(project, overlay(project, id="agent-photo", lane="agent"), lane="agent")
    assert not any(f.rule == "background_overlap" for f in result.findings)


def test_background_look_resolves_only_requested_fields(project: Path) -> None:
    legacy = add(project, background(project)).require_valid().elements[0].payload
    assert set(legacy) == {"media", "fit", "box_px"}
    item = background(project, layout="bottom", blend="wrap", enter="wipe")
    resolved = add(project, item).require_valid().elements[0].payload
    assert resolved["person_box_px"] == [189, 672, 702, 1248]
    assert resolved["wipe_frames"] == 9
    assert "push_scale" not in resolved


def test_background_look_findings(project: Path) -> None:
    result = add(project, background(project, motion="push", enter="wipe",
                                     start={"time": 2}, end={"time": 2.5}))
    assert {"background_motion", "background_enter"} <= {f.rule for f in result.findings}


def test_background_explicit_legacy_values_add_no_resolved_fields(project: Path) -> None:
    item = background(project, layout="full", blend="none", enter="cut", motion="none")
    payload = add(project, item).require_valid().elements[0].payload
    assert set(payload) == {"media", "fit", "box_px", "layout", "blend", "enter", "motion"}


@pytest.mark.parametrize("direction", ["in", "out", "none"])
def test_background_photo_push_uses_style(project: Path, direction: str) -> None:
    image = overlay_image(project / "page.png")
    write(project / "style.json", {"schema_version": 1, "broll": {
        "still_push": direction, "still_push_scale": 1.04}})
    data = json.loads((project / "project.json").read_bytes())
    write(project / "project.json", {**data, "style": "style.json"})
    result = add(project, background(project, motion="push", media={
        "path": image.name, "sha256": receipt.sha256_file(image)})).require_valid()
    payload = result.elements[0].payload
    assert payload["push_scale"] == 1.04
    assert payload["push_direction"] == ("out" if direction == "out" else "in")
    assert "person_box_px" not in payload and "wipe_frames" not in payload
