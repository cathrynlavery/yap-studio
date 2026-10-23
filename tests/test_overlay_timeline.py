"""Overlay contracts, placement resolution, and silent image and clip rules."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from test_timeline import add, graphic, insert, overlay, publish_cut, write
from test_timeline import project as project

from yap_studio import receipt
from yap_studio.project.lock import ConflictError
from yap_studio.project.model import (
    ProjectFileError,
    json_schema,
    load_timeline,
    load_timeline_fragment,
    parse_document,
)
from yap_studio.project.status import ProjectContext
from yap_studio.stages import timeline as stage


def authored_overlay(**changes: Any) -> dict[str, Any]:
    return {"id": "broll-overlay", "kind": "overlay", "start": {"time": 0},
            "hold_s": 2, "media": {"path": "picture.png", "sha256": "a" * 64},
            "placement": "headroom", **changes}


@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
def test_old_timeline_v1_still_loads_unchanged(name: str, tmp_path: Path) -> None:
    data = {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-old", "kind": "insert", "lane": "broll", "start": {"time": 0},
        "hold_s": 2, "media": {"path": "clip.mov", "sha256": "a" * 64},
    }]}
    if name == "timeline":
        data["words_sha256"] = {}
    target = write(tmp_path / "broll.json", data)
    before = target.read_bytes()
    loader = load_timeline if name == "timeline" else load_timeline_fragment
    assert loader(target, lane="broll").to_dict() == data
    jsonschema.validate(data, json_schema(name))
    assert target.read_bytes() == before


@pytest.mark.parametrize("lane", ["broll", "agent"])
@pytest.mark.parametrize("placement", ["headroom", "side_left", "side_right", "tall_behind", "box"])
@pytest.mark.parametrize("explicit_lane", [False, True])
def test_overlay_fragments_accept_each_lane_and_placement(
    tmp_path: Path, lane: str, placement: str, explicit_lane: bool,
) -> None:
    item = authored_overlay(id=f"{lane}-overlay", placement=placement)
    if placement == "box":
        item["box"] = [60, 700, 440, 560]
    if explicit_lane:
        item["lane"] = lane
    data = {"schema": "yap-studio/timeline@1", "elements": [item]}
    jsonschema.validate(data, json_schema("timeline-fragment"))
    loaded = load_timeline_fragment(write(tmp_path / "fragment.json", data), lane=lane)
    assert loaded.elements[0].fields["fit"] == "cover"
    assert loaded.elements[0].fields["motion"] == "none"
    normalized = loaded.to_dict()
    jsonschema.validate(normalized, json_schema("timeline-fragment"))
    assert parse_document("timeline-fragment", normalized, lane=lane) == loaded


@pytest.mark.parametrize("changes", [
    {"placement": "box"},
    *({"placement": placement, "box": [0, 0, 100, 100]}
      for placement in ("headroom", "side_left", "side_right", "tall_behind")),
    {"placement": "box", "box": [-1, 0, 100, 100]},
    {"placement": "box", "box": [0, -1, 100, 100]},
    {"placement": "box", "box": [0, 0, 0, 100]},
    {"placement": "box", "box": [0, 0, 100, -1]},
    {"placement": "box", "box": [0, 0, 100]},
    {"placement": "other"}, {"fit": "crop"}, {"motion": "zoom"}, {"audio": "mute"},
    {"media": {"path": "image.png", "sha256": "a" * 64, "crop": [0, 0, 100, 100]}},
    {"media": {"path": "image.png", "sha256": "a" * 64, "blur": []}},
    {"media": {"path": "image.png", "sha256": "a" * 64, "fit": "cover"}},
    {"end": {"time": 2}}, {"hold_s": None},
])
@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
def test_invalid_overlays_fail_loader_and_schema(changes: dict, name: str) -> None:
    data = {"schema": "yap-studio/timeline@1", "elements": [authored_overlay(**changes)]}
    if name == "timeline":
        data["words_sha256"] = {}
    with pytest.raises(ProjectFileError):
        parse_document(name, data, lane="broll")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, json_schema(name))


@pytest.mark.parametrize("field", ["media", "placement", "hold_s"])
def test_overlay_requires_media_placement_and_an_ending(field: str) -> None:
    item = authored_overlay()
    del item[field]
    data = {"schema": "yap-studio/timeline@1", "elements": [item]}
    with pytest.raises(ProjectFileError):
        parse_document("timeline-fragment", data, lane="broll")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, json_schema("timeline-fragment"))


@pytest.mark.parametrize("placement,expected", [
    ("headroom", (60, 320, 960, 400)),
    ("side_left", (60, 760, 440, 560)),
    ("side_right", (580, 760, 440, 560)),
    ("box", (12, 731, 401, 200)),
])
def test_overlay_box_uses_render_style_and_word_anchors(
    project: Path, monkeypatch: pytest.MonkeyPatch, placement: str, expected: tuple,
) -> None:
    original_style = ProjectContext.render_style

    def render_style(context: ProjectContext):
        style = original_style(context)
        data = copy.deepcopy(style.data)
        data["internal"].update(key_text_top_px=320, key_text_bottom_px=720)
        return replace(style, data=data)

    monkeypatch.setattr(ProjectContext, "render_style", render_style)
    item = overlay(project, placement=placement)
    if placement == "box":
        item["box"] = [12.4, 730.6, 400.7, 200.2]
    style_data = render_style(ProjectContext(project)).data
    before = copy.deepcopy(item)
    assert stage.overlay_box(item, style_data) == expected
    assert item == before
    result = add(project, item).require_valid()
    resolved = result.elements[0]
    assert (resolved.start_frame, resolved.end_frame) == (66, 108)
    assert resolved.payload["box_px"] == list(expected)
    assert resolved.to_dict()["box_px"] == list(expected)
    assert "box_px" not in load_timeline(project / "timeline/broll.json").elements[0].fields
    assert stage.resolve_timeline(project).elements[0].payload["box_px"] == list(expected)


def test_overlay_follows_words_after_a_recut(project: Path) -> None:
    add(project, overlay(project)).require_valid()
    publish_cut(project, shift=1)
    result = stage.resolve_timeline(project).require_valid()
    assert (result.elements[0].start_frame, result.elements[0].end_frame) == (36, 78)


@pytest.mark.parametrize("duration,warns", [(0.5, True), (1.0, False), (6.0, False), (6.5, True)])
def test_overlay_length_is_a_warning(project: Path, duration: float, warns: bool) -> None:
    item = overlay(project, start={"time": 0}, end={"time": duration})
    result = add(project, item).require_valid()
    warnings = [finding for finding in result.findings if finding.rule == "overlay_length"]
    assert bool(warnings) == warns
    assert all(finding.severity == "warning" for finding in warnings)


@pytest.mark.parametrize("box,valid", [
    ([0, 0, 1080, 1920], True), ([1000, 0, 81, 100], False),
    ([0, 1900, 100, 21], False), ([0, 0, 0.1, 100], False),
])
def test_overlay_box_must_fit_the_canvas(project: Path, box: list, valid: bool) -> None:
    result = add(project, overlay(project, placement="box", box=box))
    assert result.valid == valid
    assert any(finding.rule == "overlay_box" and finding.severity == "violation"
               for finding in result.findings) == (not valid)
    assert (project / "timeline/broll.json").exists() == valid


@pytest.mark.parametrize("kind,limit,valid", [
    ("insert", 1, False), ("insert", 3, False), ("graphic", 1, False),
    ("graphic", 2, True), ("overlay", 1, False), ("overlay", 2, True),
])
def test_overlay_visual_overlap_and_insert_exclusivity(
    project: Path, kind: str, limit: int, valid: bool,
) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"elements_at_once_max": limit}
    write(project / "project.json", data)
    if kind == "graphic":
        add(project, graphic(start={"word": "t1:2"}), lane="agent").require_valid()
    else:
        other = (insert if kind == "insert" else overlay)(project, id="broll-other")
        add(project, other).require_valid()
    result = add(project, overlay(project))
    assert result.valid == valid
    assert any(finding.rule == "visual_overlap" for finding in result.findings) == (not valid)


@pytest.mark.parametrize("case,rule", [
    ("missing", "overlay_media"), ("bounds", "overlay_bounds"),
    ("fast", "overlay_speed"), ("slow", "overlay_speed"), ("hdr", "overlay_hdr"),
    ("no_duration", "overlay_media"),
])
def test_overlay_clip_media_rules(
    project: Path, monkeypatch: pytest.MonkeyPatch, case: str, rule: str,
) -> None:
    item = overlay(project)
    if case == "missing":
        item["media"]["path"] = "missing.mov"
    elif case == "bounds":
        item["media"].update(in_s=12, speed=2)
    elif case in {"fast", "slow"}:
        item["media"]["speed"] = 2.1 if case == "fast" else 0.4
    else:
        info = stage.ffmpeg.probe(project / "source.mov", count_frames=False)
        if case == "hdr":
            item["media"]["hdr"] = "off"
            info = replace(info, color_transfer="smpte2084")
        else:
            info = replace(info, video_duration_s=None)
        monkeypatch.setattr(stage.ffmpeg, "probe", lambda *args, **kwargs: info)
    result = add(project, item)
    assert not result.valid
    assert any(finding.rule == rule and finding.severity == "violation"
               for finding in result.findings)
    assert not (project / "timeline/broll.json").exists()


def test_overlay_hash_change_is_refused(project: Path) -> None:
    source = project / "overlay.mov"
    source.write_bytes((project / "source.mov").read_bytes())
    item = overlay(project)
    item["media"]["path"] = source.name
    add(project, item).require_valid()
    source.write_bytes(source.read_bytes() + b"changed")
    result = stage.resolve_timeline(project)
    assert not result.valid
    assert any(finding.rule == "media_hash" and finding.severity == "violation"
               for finding in result.findings)


def test_overlay_add_rechecks_media_under_the_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = project / "overlay.mov"
    source.write_bytes((project / "source.mov").read_bytes())
    item = overlay(project)
    item["media"]["path"] = source.name
    original = stage._resolve

    def changed_after_resolution(*args, **kwargs):
        result = original(*args, **kwargs).require_valid()
        source.write_bytes(source.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(stage, "_resolve", changed_after_resolution)
    with pytest.raises(ConflictError):
        add(project, item)
    assert not (project / "timeline/broll.json").exists()


def test_overlay_fit_speed_and_external_media_warning(project: Path, tmp_path: Path) -> None:
    source = tmp_path.parent / f"{tmp_path.name}-overlay.mov"
    source.write_bytes((project / "source.mov").read_bytes())
    item = overlay(project)
    item["media"].update(path=str(source), in_s=12.6, speed="fit")
    result = add(project, item).require_valid()
    assert result.elements[0].payload["media"]["speed"] == pytest.approx(1)
    assert any(finding.rule == "external_media" and finding.severity == "warning"
               for finding in result.findings)


@pytest.mark.parametrize("extension,speed", [("png", 1), ("jpg", "fit")])
@pytest.mark.parametrize("kind", ["overlay", "insert"])
def test_image_has_no_source_duration_limit(
    project: Path, extension: str, speed: int | str, kind: str,
) -> None:
    source = project / f"overlay.{extension}"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=80x60",
        "-frames:v", "1", str(source),
    ], check=True, timeout=30)
    item = (overlay if kind == "overlay" else insert)(
        project, start={"time": 0}, end={"time": 6.5},
    )
    if kind == "overlay":
        item.update(fit="contain", motion="none")
    else:
        item["media"]["fit"] = "contain"
    item["media"].update(path=source.name, sha256=receipt.sha256_file(source),
                         in_s=100, speed=speed)
    result = add(project, item).require_valid()
    expected = [f"{kind}_length"]
    if kind == "insert":
        expected.extend(["broll_full_screen", "face_on_screen"])
    assert [finding.rule for finding in result.findings] == expected
    if kind == "overlay":
        assert result.elements[0].payload["fit"] == "contain"
        assert result.elements[0].payload["motion"] == "none"
    else:
        assert result.elements[0].payload["media"]["fit"] == "contain"
    assert result.elements[0].payload["media"]["speed"] == 1


def test_overlay_media_needs_picture(project: Path) -> None:
    source = project / "audio.wav"
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(project / "source.mov"),
        "-vn", "-c:a", "pcm_s16le", str(source),
    ], check=True, timeout=30)
    item = overlay(project)
    item["media"].update(path=source.name, sha256=receipt.sha256_file(source))
    result = add(project, item)
    assert not result.valid
    assert any(finding.rule == "overlay_media" for finding in result.findings)
