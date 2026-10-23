"""Opt-in moments on synthetic word maps and generated contracts."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from test_timeline import add, graphic, insert, overlay, transition, transition_story, write
from test_timeline import project as project

from yap_studio import receipt
from yap_studio.project.model import ProjectFileError, json_schema, parse_document
from yap_studio.stages.timeline import resolve_timeline
from yap_studio.style import SHIPPED_PRESETS, StyleError, resolve, resolve_style_data

STYLES = ("color_pop", "background_blur", "spotlight", "sticker_freeze", "outline",
          "clone_flash")


def moment(**changes: object) -> dict:
    return {"id": "agent-moment", "kind": "moment", "lane": "agent", "style": "color_pop",
            "start": {"word": "t1:2"}, "end": {"word": "t1:3", "edge": "end"}, **changes}


def enable(root: Path) -> None:
    write(root / "style.json", {"schema_version": 1, "moments": {"styles": list(STYLES)}})
    data = json.loads((root / "project.json").read_bytes())
    write(root / "project.json", {**data, "style": "style.json"})


@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
@pytest.mark.parametrize("style", STYLES)
def test_moment_contract_round_trip(name: str, style: str) -> None:
    data = {"schema": "yap-studio/timeline@1", "elements": [moment(style=style)]}
    if name == "timeline":
        data["words_sha256"] = {}
    jsonschema.validate(data, json_schema(name))
    assert parse_document(name, data, lane="agent").elements[0].fields["style"] == style


@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
@pytest.mark.parametrize("change", [{"style": "unknown"}, {"lane": "broll"},
                                   {"style": None}, {"end": None},
                                   {"style": "_".join(("silhouette", "window"))}, {"clips": []}])
def test_moment_contract_refuses_invalid_fields(name: str, change: dict) -> None:
    data = {"schema": "yap-studio/timeline@1", "words_sha256": {},
            "elements": [moment(**change)]}
    if name == "timeline-fragment":
        data.pop("words_sha256")
    with pytest.raises(ProjectFileError):
        parse_document(name, data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, json_schema(name))


def test_moment_requires_style_and_ending() -> None:
    for field in ("style", "end"):
        item = moment()
        item.pop(field)
        with pytest.raises(ProjectFileError):
            parse_document("timeline-fragment", {"schema": "yap-studio/timeline@1",
                                                 "elements": [item]})


@pytest.mark.parametrize(("style", "duration", "valid"), [
    ("color_pop", 0.7, False), ("color_pop", 0.8, True), ("color_pop", 3, True),
    ("color_pop", 3.1, False), ("sticker_freeze", 0.5, False),
    ("sticker_freeze", 0.6, False), ("sticker_freeze", 0.7, False),
    ("sticker_freeze", 0.8, True), ("sticker_freeze", 1.4, True),
    ("sticker_freeze", 1.5, False),
])
def test_moment_duration_limits(project: Path, style: str, duration: float, valid: bool) -> None:
    enable(project)
    item = moment(style=style, start={"time": 2}, end={"time": 2 + duration})
    result = add(project, item, lane="agent")
    assert result.valid is valid
    if not valid:
        finding = next(f for f in result.findings if f.rule == "moment_length")
        assert finding.severity == "violation" and finding.suggested_action


def test_moment_off_has_concrete_fix(project: Path) -> None:
    result = add(project, moment(), lane="agent")
    finding = next(f for f in result.findings if f.rule == "moment_off")
    assert not result.valid
    assert finding.suggested_action == (
        "turn on color_pop under moments in the style, or remove the moment")
    assert not (project / "timeline/agent.json").exists()


@pytest.mark.parametrize("cutout,video_type,backend,requested,keyed", [
    ("screen", "talking_head", None, None, True),
    ("auto", "green_screen", None, None, True),
    ("auto", "talking_head", "green", None, True),
    ("auto", "talking_head", "blue", None, True),
    ("auto", "green_screen", "person-detection", None, False),
    ("auto", "talking_head", None, None, False),
    ("person", "green_screen", None, None, False),
    ("person", "green_screen", "green", None, False),
    ("screen", "talking_head", "person-detection", None, True),
    ("auto", "talking_head", "green", "chroma", False),
    ("auto", "green_screen", "green", "chroma", True),
    ("auto", "talking_head", "person-detection", "person", False),
    ("auto", "green_screen", "person-detection", "person", True),
    ("auto", "talking_head", "green", "auto", True),
    ("auto", "green_screen", "green", "auto", True),
    ("auto", "green_screen", "green", None, True),
])
def test_clone_flash_checks_project_keyed_take(
    project: Path, cutout: str, video_type: str, backend: str | None,
    requested: str | None, keyed: bool,
) -> None:
    enable(project)
    style = json.loads((project / "style.json").read_text())
    write(project / "style.json", {**style, "broll": {"cutout": cutout}})
    data = json.loads((project / "project.json").read_text())
    write(project / "project.json", {**data, "video_type": video_type})
    if backend is not None:
        (project / "cut/cut.mp4").write_bytes(b"synthetic cut identity")
        digest = receipt.sha256_file(project / "cut/cut.mp4")
        folder = project / f".yap/cache/matte/{digest}"
        folder.mkdir(parents=True)
        (folder / "matte.mkv").write_bytes(b"synthetic matte")
        metadata = ({"name": "chroma-key", "screen": backend, "border_share": 1.0}
                    if backend in {"green", "blue"} else {"name": backend})
        if requested is not None:
            metadata["requested"] = requested
        write(folder / "matte.json", {"status": "measured", "backend": metadata,
                                      "matte": {"file": "matte.mkv"}})
    result = add(project, moment(style="clone_flash", start={"time": 2}, end={"time": 3.2}),
                 lane="agent")
    findings = [f for f in result.findings if f.rule == "clone_keyed"]
    assert result.valid is not keyed
    assert bool(findings) is keyed
    if keyed:
        assert len(findings) == 1 and findings[0].severity == "violation"
        screen = (backend if backend in {"green", "blue"} and cutout == "auto"
                  and requested in (None, "auto") else "green or blue")
        assert findings[0].message.endswith(
            f"clone_flash cannot clean the {screen} screen edge on the copies, "
            f"so they would carry a {screen} fringe")
        assert findings[0].suggested_action == (
            "pick another Moment style, or remove the clone flash; clone flash needs a take "
            "filmed without a green or blue screen")
        assert not (project / "timeline/agent.json").exists()


@pytest.mark.parametrize("name", STYLES[:-1])
def test_other_moments_allow_keyed_take(project: Path, name: str) -> None:
    enable(project)
    style = json.loads((project / "style.json").read_text())
    write(project / "style.json", {**style, "broll": {"cutout": "screen"}})
    assert add(project, moment(style=name, start={"time": 2}, end={"time": 2.8}),
               lane="agent").valid


@pytest.mark.parametrize("kind", ["insert", "background", "zoom", "transition", "graphic",
                                  "moment"])
def test_moment_overlap_names_other(project: Path, kind: str) -> None:
    if kind == "transition":
        transition_story(project)
    enable(project)
    other = {
        "insert": lambda: insert(project, id="agent-other", lane="agent"),
        "background": lambda: insert(project, id="agent-other", lane="agent", kind="background"),
        "zoom": lambda: {"id": "agent-other", "kind": "zoom", "lane": "agent",
                         "start": {"time": 2}, "hold_s": 3, "peak_scale": 1.04,
                         "center": [0.5, 0.5]},
        "transition": lambda: transition(id="agent-other"),
        "graphic": lambda: graphic(id="agent-other", layer="behind", start={"time": 2}),
        "moment": lambda: moment(id="agent-other"),
    }[kind]()
    item = moment(start={"time": 5.9}, end={"time": 6.7}) if kind == "transition" else moment()
    result = add(project, other, item, lane="agent")
    finding = next(f for f in result.findings if f.rule == "moment_overlap")
    assert not result.valid and finding.severity == "violation"
    assert "agent-other" in finding.suggested_action
    assert set(finding.elements) == {"agent-other", "agent-moment"}


@pytest.mark.parametrize("style", STYLES[:5])
@pytest.mark.parametrize("kind", ["graphic", "title", "overlay"])
def test_old_moments_allow_front_layers(project: Path, style: str, kind: str) -> None:
    enable(project)
    other = {
        "graphic": lambda: graphic(start={"time": 2}, layer="front"),
        "title": lambda: {"id": "agent-title", "kind": "title", "lane": "agent",
                          "lines": ["Alpha"], "start": {"time": 2}, "hold_s": 2},
        "overlay": lambda: overlay(project, id="agent-overlay", lane="agent"),
    }[kind]()
    assert add(project, other, moment(style=style), lane="agent").valid


@pytest.mark.parametrize("kind,layer", [("title", "front"), ("graphic", "front"),
                                       ("graphic", "behind"), ("overlay", "front")])
@pytest.mark.parametrize("overlap", [True, False])
def test_clone_flash_refuses_overlapping_layers(project: Path, kind: str, layer: str,
                                               overlap: bool) -> None:
    enable(project)
    other = {
        "graphic": lambda: graphic(id="agent-other", layer=layer,
                                    start={"time": 2}, hold_s=2),
        "title": lambda: {"id": "agent-other", "kind": "title", "lane": "agent",
                          "lines": ["Alpha"], "start": {"time": 2}, "hold_s": 2},
        "overlay": lambda: overlay(project, id="agent-other", lane="agent",
                                    start={"time": 2}, end={"time": 4}),
    }[kind]()
    start = 2 if overlap else 4
    item = moment(style="clone_flash", start={"time": start}, end={"time": start + 1.2})
    result = add(project, other, item, lane="agent")
    findings = [finding for finding in result.findings if finding.rule == "moment_overlap"]
    assert result.valid is not overlap
    if overlap:
        assert len(findings) == 1 and findings[0].severity == "violation"
        assert set(findings[0].elements) == {"agent-other", "agent-moment"}
        assert findings[0].suggested_action == (
            "move or remove agent-moment so it does not overlap agent-other")
    else:
        assert not findings


def test_count_and_close_are_warnings(project: Path) -> None:
    enable(project)
    items = [moment(id=f"agent-moment-{i}", start={"time": start}, end={"time": start + 0.8})
             for i, start in enumerate((0, 5, 10))]
    result = add(project, *items, lane="agent")
    assert result.valid
    counts = [f for f in result.findings if f.rule == "moment_count"]
    assert len(counts) == 1 and counts[0].elements == ("agent-moment-2",)
    close = [f for f in result.findings if f.rule == "moment_close"]
    assert len(close) == 2 and all(f.severity == "warning" for f in [*counts, *close])
    assert all(f.suggested_action == "keep one, or move one at least 10 s away" for f in close)
    assert resolve_timeline(project).valid


def test_ten_second_gap_is_allowed(project: Path) -> None:
    enable(project)
    result = add(project, moment(start={"time": 0}, end={"time": 0.8}),
                 moment(id="agent-second", start={"time": 10}, end={"time": 10.8}), lane="agent")
    assert result.valid and not any(f.rule == "moment_close" for f in result.findings)


def test_old_styles_keep_moments_off() -> None:
    assert resolve_style_data({"schema_version": 1})["moments"] == {"styles": []}
    for preset in SHIPPED_PRESETS:
        assert resolve(f"preset:{preset}").data["moments"] == {"styles": []}


@pytest.mark.parametrize("styles", [["unknown"], ["outline", "outline"], "color_pop"])
def test_invalid_moment_style_choices(styles: object) -> None:
    with pytest.raises(StyleError):
        resolve_style_data({"schema_version": 1, "moments": {"styles": styles}})


def test_moment_refuses_grade_compare(project: Path) -> None:
    enable(project)
    style = json.loads((project / "style.json").read_text())
    write(project / "style.json", {**style, "grade": {"look": "film"}})
    result = add(project, {"id": "agent-compare", "kind": "grade_compare", "lane": "agent",
                          "start": {"time": 2}, "hold_s": 2}, moment(), lane="agent")
    assert any(f.rule == "moment_overlap" for f in result.findings)
