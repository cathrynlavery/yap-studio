"""Synthetic stage receipts, person redraw pixels and real number ink checks."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synth
from test_assemble_picture import pictures as pictures
from test_assemble_picture import project as picture_project
from test_cards_render import renderer_install as renderer_install
from test_matte_project import synthetic_matte
from test_planner_cards import synthetic_faces
from test_stage_captions import layer_project as layer_project
from test_timeline import project as project
from test_timeline import publish_cut, write

from yap_studio import cards, ffmpeg, receipt
from yap_studio.errors import DependencyError, ValidationError
from yap_studio.faces.model import Box
from yap_studio.faces.placement import Band
from yap_studio.planner.number_behind import place_number, resize_number
from yap_studio.project.model import parse_document
from yap_studio.project.receipts import StageRun
from yap_studio.project.status import ProjectContext, ingest_inputs
from yap_studio.render import hyperframes
from yap_studio.stages import graphics, number_graphics, run, timeline
from yap_studio.stages.assemble_picture import _Graph, render_picture
from yap_studio.stages.behind_window import number_redraw
from yap_studio.stages.number_visibility import (
    add_number_visibility,
    measure_number_layers,
    number_visibility_assert,
)
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import resolve

SYNTHETIC_INK_ROWS = 20  # Concentrate synthetic glyph ink near its lower bowl.


def bottom_heavy_layer(
    plan, out: Path, *, rows: int = SYNTHETIC_INK_ROWS,
) -> hyperframes.LayerResult:
    v = plan.variables
    bottom = math.ceil(v["shadow_pad_px"] + v["cap_height_em"] * v["value_px"])
    synth.picture_source(out, f"color=white:s={plan.width}x{plan.height}:r={plan.fps},format=rgba,"
                         f"geq=r=255:g=255:b=255:a='255*gte(X,{v['shadow_pad_px']})"
                         f"*lt(X,{plan.width - v['shadow_pad_px']})"
                         f"*gte(Y,{bottom - rows})*lt(Y,{bottom})'",
                         plan.frames, alpha=True)
    return hyperframes.LayerResult(out, plan.frames, plan.width, plan.height,
                                    hyperframes.layer_duration_s(plan.frames, plan.fps))


@pytest.fixture
def number_renderer(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = []

    def render(plan, work, out, **kwargs):
        calls.append(plan.variables["value_px"])
        return bottom_heavy_layer(plan, out)

    monkeypatch.setattr(cards, "render_card", render)
    return calls


@pytest.fixture
def number_scene(layer_project: Path):
    style = resolve("preset:clean_minimal")
    card = cards.card_layer_plan("big_number", {"start": 3, "end": 3}, style, frames=75)
    face = Box(440, 600, 200, 200)
    clear = np.zeros((1920, 1080), np.uint8)
    default, _, _ = place_number(card, [[face]] * card.frames, [clear] * card.frames,
                                caption_top=1400, band=Band(0, 640), margin_px=24,
                                identifier="agent-number")
    v = default.variables
    bottom = math.ceil(default.y + v["shadow_pad_px"] + v["cap_height_em"] * v["value_px"])
    hair = clear.copy()
    hair[bottom - 10:] = 255
    plate = synth.picture_source(layer_project / "number-plate.mov",
                                 "color=blue:s=1080x1920:r=30", card.frames)
    synthetic_matte(plate, layer_project / "number-matte", box=(0, bottom - 10, 1080,
                    1920 - bottom + 10), active_windows=((70, 75),))
    item = ResolvedElement("agent-number", "graphic", "agent", 0, card.frames, "behind",
                           {"template": "big_number", "slots": {"start": 3, "end": 3}})
    return style, card, face, clear, hair, plate, item, default


def measure_placed_number(root, plate, item, plan, path):
    layer = {"id": item.id, "path": str(path), "start_frame": 0, "frames": plan.frames,
             "x": plan.x, "y": plan.y, "width": plan.width, "height": plan.height,
             "layer": "behind"}
    return measure_number_layers(root, root / "number-matte/matte.mkv", ffmpeg.probe(plate),
                                  [layer], ResolvedTimeline((item,), (), (), plan.frames, 0), ())


def test_rendered_ink_moves_bottom_heavy_number_and_matches_final(
    layer_project: Path, number_scene, number_renderer: list[int],
) -> None:
    from test_number_behind import independent_visibility

    style, card, face, clear, hair, plate, item, default = number_scene
    assert .95 <= independent_visibility(default, hair, 70) < 1
    before = bottom_heavy_layer(default, layer_project / "before-number.mov")
    failed = measure_placed_number(layer_project, plate, item, default, before.path)
    assert failed[0]["hidden_frames"] == 5
    assert not number_visibility_assert(list(failed))["pass"]
    placed, placement, evidence = number_graphics.place_number_graphic(
        card, item, style, [[face]] * card.frames, layer_project / "number-matte/matte.mkv",
        1500, project_dir=layer_project,
    )
    assert placement == "behind_head" and placed.y == default.y - 8
    assert evidence["ink_status"] == "verified" and evidence["render_count"] == 1
    assert number_renderer == [card.variables["value_px"]]
    published = number_graphics.publish_number_layer(
        placed, layer_project, layer_project / "published-number.mov")
    cached, created = number_graphics.cached_number_layer(placed, layer_project)
    assert not created and published.path.read_bytes() == cached.path.read_bytes()
    final = measure_placed_number(layer_project, plate, item, placed, published.path)
    assert final[0]["hidden_frames"] == 0
    assert final[0]["worst_visible_share"] == evidence["worst_visible_share"] == .9
    assert number_visibility_assert(list(final))["pass"]
    assert number_renderer == [card.variables["value_px"]]


def test_number_layer_cache_ignores_position_and_reuses_one_render_per_size(
    layer_project: Path, number_renderer: list[int],
) -> None:
    card = cards.card_layer_plan("big_number", {"end": 3}, resolve("preset:clean_minimal"),
                                 frames=75)
    first, rendered = number_graphics.cached_number_layer(card, layer_project)
    moved, rerendered = number_graphics.cached_number_layer(
        replace(card, x=40, y=400), layer_project)
    assert rendered and not rerendered and first == moved
    smaller, rendered = number_graphics.cached_number_layer(resize_number(card, 270), layer_project)
    assert rendered and smaller.path != first.path
    assert number_renderer == [300, 270]
    assert first.path.is_relative_to(layer_project / ".yap/cache")
    changed = replace(card, variables={**card.variables, "end": 4})
    other, rendered = number_graphics.cached_number_layer(changed, layer_project)
    assert rendered and other.path != first.path
    first.path.write_bytes(b"changed synthetic cache")
    with pytest.raises(ValidationError, match="cached number layer changed"):
        number_graphics.cached_number_layer(card, layer_project)


def test_number_key_includes_browser_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    card = cards.card_layer_plan("big_number", {"end": 3}, resolve("preset:clean_minimal"),
                                 frames=75)
    assets = number_graphics._number_assets(card)
    projections = []
    original_projection = number_graphics.projection_sha256

    def projection(data):
        projections.append(data)
        return original_projection(data)

    monkeypatch.setattr(number_graphics, "projection_sha256", projection)
    software = number_graphics._number_key(card, assets)
    assert projections[-1]["template"]["browser_gpu"] is False
    monkeypatch.setattr(hyperframes, "CARD_BROWSER_GPU", True)
    assert number_graphics._number_key(card, assets) != software
    assert projections[-1]["template"]["browser_gpu"] is True


def test_number_ink_search_renders_only_once_for_each_scale(
    layer_project: Path, number_scene, monkeypatch: pytest.MonkeyPatch,
) -> None:
    style, card, face, _, _, plate, item, _ = number_scene
    calls = []

    def render(plan, work, out, **kwargs):
        calls.append(plan.variables["value_px"])
        return bottom_heavy_layer(plan, out, rows=10 if calls[-1] == 300 else 20)

    monkeypatch.setattr(cards, "render_card", render)
    monkeypatch.setattr(number_graphics, "top_band", lambda style: Band(350, 640))
    placed, placement, evidence = number_graphics.place_number_graphic(
        card, item, style, [[face]] * card.frames, layer_project / "number-matte/matte.mkv",
        1500, project_dir=layer_project,
    )
    assert placement == "behind_head" and placed.variables["value_px"] == 270
    assert calls == [300, 270] and evidence["render_count"] == 2
    published = number_graphics.publish_number_layer(
        placed, layer_project, layer_project / "published-scaled-number.mov")
    measured = measure_placed_number(layer_project, plate, item, placed, published.path)
    assert measured[0]["hidden_frames"] == 0 and calls == [300, 270]


def test_number_without_renderer_keeps_conservative_boxes_and_evidence(
    layer_project: Path, number_scene, monkeypatch: pytest.MonkeyPatch,
) -> None:
    style, card, face, _, _, _, item, default = number_scene

    def unavailable(*args, **kwargs):
        raise DependencyError("renderer unavailable", code="renderer_missing")

    monkeypatch.setattr(cards, "render_card", unavailable)
    placed, placement, evidence = number_graphics.place_number_graphic(
        card, item, style, [[face]] * card.frames, layer_project / "number-matte/matte.mkv",
        1500, project_dir=layer_project,
    )
    assert placement == "behind_head" and placed.y == default.y
    assert evidence["ink_status"] == "unavailable" and evidence["render_count"] == 0
    assert evidence["worst_visible_share"] >= .95
    assert "renderer unavailable" in evidence["reason"]
    before = bottom_heavy_layer(placed, layer_project / "unchecked-number.mov")
    final = measure_placed_number(layer_project, number_scene[5], item, placed, before.path)
    assert not number_visibility_assert(list(final))["pass"]


@pytest.mark.render
def test_real_number_search_and_final_measure_the_same_cached_layer(
    layer_project: Path, number_scene, renderer_install: hyperframes.RendererInstall,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style, card, face, _, _, plate, item, _ = number_scene
    render = cards.render_card
    calls = []

    def actual(plan, work, out, **kwargs):
        calls.append(plan.variables["value_px"])
        return render(plan, work, out, install=renderer_install)

    monkeypatch.setattr(cards, "render_card", actual)
    placed, placement, evidence = number_graphics.place_number_graphic(
        card, item, style, [[face]] * card.frames, layer_project / "number-matte/matte.mkv",
        1500, project_dir=layer_project,
    )
    assert placement == "behind_head" and evidence["ink_status"] == "verified"
    assert evidence["render_count"] == len(calls) == 1
    published = number_graphics.publish_number_layer(
        placed, layer_project, layer_project / "published-real-number.mov")
    measured = measure_placed_number(layer_project, plate, item, placed, published.path)
    assert measured[0]["hidden_frames"] == 0
    assert measured[0]["worst_visible_share"] == evidence["worst_visible_share"]
    assert number_visibility_assert(list(measured))["pass"]
    assert len(calls) == 1


@pytest.mark.parametrize("blocker", ["clear", "matte", "transition"])
def test_number_graphics_uses_matte_and_reuses_scaled_placement_and_warnings(
    layer_project: Path, number_renderer: list[int], monkeypatch: pytest.MonkeyPatch,
    blocker: str,
) -> None:
    blocked = blocker != "clear"
    data = json.loads((layer_project / "style.json").read_text())
    data["graphics"] = {"card_kinds_on": ["big_number"]}
    write(layer_project / "style.json", data)
    item = ResolvedElement("agent-number", "graphic", "agent", 0, 75, "front",
                           {"template": "big_number", "slots": {"end": 10}})
    elements = (item,)
    if blocker == "transition":
        elements += (ResolvedElement("agent-picture", blocker, "agent", 0, 75, "front", {}),)
    resolved = ResolvedTimeline(elements, (), (), 90, 0)
    monkeypatch.setattr(ProjectContext, "resolved_timeline", lambda self: resolved)
    monkeypatch.setattr(graphics, "picture_faces", lambda *args: synthetic_faces(
        None if blocker == "transition" else Box(440, 600, 200, 200), frames=90))
    face_calls = []

    def original_faces(*args):
        face_calls.append(args)
        return synthetic_faces(Box(440, 600, 200, 200), frames=90)

    monkeypatch.setattr(number_graphics, "cut_faces", original_faces)
    matte = layer_project / "number-matte.mkv"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color={'white' if blocker == 'matte' else 'black'}:s=1080x1920:r=30,format=gray",
                "-frames:v", "90", "-c:v", "ffv1", "-threads", "1", str(matte)], timeout_s=60)
    calls = []

    def ensure(root, cut):
        calls.append((root, cut))
        return matte

    monkeypatch.setattr(graphics, "ensure_cut_matte", ensure)
    first = graphics.graphics(layer_project)
    assert len(calls) == 1
    assert bool(face_calls) == (blocker == "transition")
    element = first["elements"][0]
    assert element["layer"] == ("front" if blocked else "behind")
    assert element["clearance"]["status"] == ("beside_front" if blocked else "behind_number")
    assert element["clearance"]["render_count"] == (0 if blocked else 1)
    assert len(number_renderer) == 1
    assert bool(first["warnings"]) == blocked
    legacy = json.loads(json.dumps(first["elements"]))
    del legacy[0]["clearance"]["ink_status"]
    with pytest.raises(ValidationError, match="ink was not checked"):
        graphics.graphics_snapshot(ProjectContext(layer_project), placements=legacy)
    original = graphics.graphics_snapshot(ProjectContext(layer_project),
                                          placements=first["elements"])
    monkeypatch.setattr(graphics, "ensure_cut_matte", lambda *args: pytest.fail("receipt matte"))
    monkeypatch.setattr(graphics, "picture_faces", lambda *args: pytest.fail("receipt faces"))
    second = graphics.graphics(layer_project)
    assert second["noop"] and second["warnings"] == first["warnings"]
    assert second["elements"] == first["elements"]
    assert graphics.graphics_snapshot(ProjectContext(layer_project),
                                      placements=second["elements"]) == original
    state = {"warnings": []}
    assert run._graphics(layer_project, state)[0]
    assert state["warnings"] == first["warnings"]


@pytest.mark.parametrize("spoken,value,suffix,valid", [
    (["ten", "thousand"], 10000, "", True),
    (["fifty", "percent"], 50, "%", True),
    (["ten"], 12, "", False),
])
@pytest.mark.parametrize("picture", [None, "zoom", "insert"])
def test_timeline_checks_spoken_number_and_refuses_zoom_or_insert_overlap(
    project: Path, spoken, value, suffix, valid, picture: str | None,
) -> None:
    root = project
    words = json.loads((root / "ingest/t1/words.json").read_bytes())
    for word, text in zip(words["words"], spoken, strict=False):
        word["text"] = text
    staged = write(root / "updated-words.json", words)
    with StageRun(root, "ingest") as journal:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            journal.record_inputs(name, digest)
        journal.replace_output(root / "ingest/t1/words.json", staged)
        journal.finish()
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = receipt.sha256_file(root / "ingest/t1/words.json")
    write(root / "edit.json", edit)
    publish_cut(root)
    elements = [{"id": "agent-number", "kind": "graphic",
        "start": {"time": 0.5}, "hold_s": 2.5, "template": "big_number",
        "slots": {"end": value, "suffix": suffix}}]
    if picture:
        elements.append({"id": "agent-picture", "kind": picture,
                         "start": {"time": 1}, "hold_s": 2,
                         **({"peak_scale": 1.05, "center": [.5, .5]} if picture == "zoom"
                            else {"media": {"path": "source.mov",
                                           "sha256": receipt.sha256_file(root / "source.mov")}})})
    lane = parse_document("timeline", {"schema": "yap-studio/timeline@1",
                                      "words_sha256": {}, "elements": elements}, lane="agent")
    result = timeline._resolve(ProjectContext(root), {"agent": lane})
    findings = [item for item in result.findings if item.rule == "number_not_spoken"]
    assert bool(findings) != valid
    overlaps = [item for item in result.findings if item.rule == "visual_overlap"]
    assert bool(overlaps) == (picture is not None)
    if picture:
        assert overlaps[0].severity == "violation"
        assert set(overlaps[0].elements) == {"agent-number", "agent-picture"}
    if not valid:
        assert findings[0].severity == "violation"
        assert "heard: 10" in findings[0].suggested_action


@pytest.mark.parametrize("kind", ["big_number", "stat"])
@pytest.mark.parametrize("first,last", [(30, 60), (0, 60), (30, 90)])
def test_person_redraw_has_padded_window_and_nine_frame_fades(
    tmp_path: Path, kind: str, first: int, last: int,
) -> None:
    # A translucent red person over a yellow card exposes the redraw envelope directly.
    graph = _Graph()
    graph.filters = ["[0:v]format=yuva444p[person]"]
    element = ResolvedElement("agent-number", "graphic", "agent", first, last, "behind",
                              {"template": kind})
    resolved = ResolvedTimeline((element,), (), (), 90, 0)
    person, enable = number_redraw(graph, "person", [{"id": element.id,
        "start_frame": first, "frames": last - first}], resolved, width=64, height=96, fps=30)
    base = graph.overlay("1:v", person, 0, 90, enable=enable)
    out = tmp_path / "redraw.mov"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=red:s=64x96:r=30,format=rgba", "-f", "lavfi", "-i",
                "color=yellow:s=64x96:r=30", "-filter_complex_threads", "1",
                "-filter_complex", ";".join(graph.filters), "-map", f"[{base}]",
                "-frames:v", "90", "-threads", "1", "-c:v", "qtrle", str(out)], timeout_s=60)
    for frame in (0, 1, 8, 14, 15, 19, 24, 29, 30, 59, 60, 66, 70, 74, 75, 89):
        green = synth.picture_pixels(out, frame)[:, :, 1].mean()
        if kind == "big_number":
            opacity = max(0, min(1, (frame - first + 15) / 9, (last + 15 - frame) / 9))
        else:
            opacity = float(first <= frame < last)
        assert green == pytest.approx(255 * (1 - opacity), abs=3)
    if kind == "big_number":
        assert enable == f"between(n,{max(0, first - 15)},{min(90, last + 15) - 1})"
    else:
        assert enable == f"between(n,{first},{last - 1})"


@pytest.mark.parametrize("hidden", [3, 4])
@pytest.mark.parametrize("covered_width", [2, 20])
def test_real_ink_visibility_counts_frames_and_populates_verify(
    tmp_path: Path, hidden: int, covered_width: int,
) -> None:
    plate = synth.picture_source(tmp_path / "plate.mov", "color=blue:s=64x96:r=30", 8)
    synthetic_matte(plate, tmp_path / "matte", box=(10, 10, covered_width, 20),
                    active_windows=((0, hidden),))
    synth.picture_source(tmp_path / "number.mov",
                         "color=white:s=40x40:r=30,format=rgba,"
                         "geq=r=255:g=255:b=255:a='255*between(X,10,29)*between(Y,10,29)'",
                         8, alpha=True)
    item = ResolvedElement("agent-number", "graphic", "agent", 0, 8, "behind",
                           {"template": "big_number"})
    layer = {"id": item.id, "path": "number.mov", "start_frame": 0, "frames": 8,
             "x": 0, "y": 0, "width": 40, "height": 40, "layer": "behind"}
    records = measure_number_layers(tmp_path, tmp_path / "matte/matte.mkv", ffmpeg.probe(plate),
                                     [layer], ResolvedTimeline((item,), (), (), 8, 0), ())
    expected_hidden = hidden if covered_width == 20 else 0
    assert records[0]["hidden_frames"] == expected_hidden
    assert records[0]["worst_visible_share"] == (0 if covered_width == 20 else .9)
    assert number_visibility_assert(list(records))["pass"] == (expected_hidden <= 3)
    report = {"status": "pass", "checks": {}, "findings": []}
    asserts = []
    add_number_visibility(report, asserts, records)
    assert report["checks"]["graphics.number_visible"]["status"] == (
        "pass" if expected_hidden <= 3 else "fail")
    assert (report["status"] == "pass") == (expected_hidden <= 3)
    assert asserts[0]["pass"] == (expected_hidden <= 3)
    # A front fallback is measured as drawn in front, not hidden by the unused matte.
    front = ResolvedTimeline((replace(item, layer="front"),), (), (), 8, 0)
    records = measure_number_layers(tmp_path, tmp_path / "matte/matte.mkv", ffmpeg.probe(plate),
                                     [{**layer, "layer": "front"}], front, ())
    assert records[0]["hidden_frames"] == 0 and records[0]["worst_visible_share"] == 1


@pytest.mark.parametrize("covered", ["shadow", "digits", "label"])
def test_number_visibility_counts_solid_glyphs_without_their_soft_shadow(
    tmp_path: Path, covered: str,
) -> None:
    from yap_studio.matte.check import measure_behind_layers

    plate = synth.picture_source(tmp_path / "plate.mov", "color=blue:s=64x96:r=30", 8)
    box = {"shadow": (10, 25, 20, 10), "digits": (10, 10, 20, 15),
           "label": (12, 38, 16, 4)}[covered]
    synthetic_matte(plate, tmp_path / "matte", box=box)
    solid = "between(X,10,29)*between(Y,10,24)+between(X,12,27)*between(Y,38,41)"
    shadow = "between(X,10,29)*between(Y,25,34)*(179-4*(Y-25))"
    synth.picture_source(tmp_path / "number.mov", "color=black:s=48x48:r=30,format=rgba,"
                         f"geq=r='255*({solid})':g='255*({solid})':b='255*({solid})':"
                         f"a='255*({solid})+{shadow}'", 8, alpha=True)
    item = ResolvedElement("agent-number", "graphic", "agent", 0, 8, "behind",
                           {"template": "big_number"})
    layer = {"id": item.id, "path": "number.mov", "start_frame": 0, "frames": 8,
             "x": 0, "y": 0, "width": 48, "height": 48, "layer": "behind"}
    matte = tmp_path / "matte/matte.mkv"
    info = ffmpeg.probe(plate)
    records = measure_number_layers(tmp_path, matte, info, [layer],
                                     ResolvedTimeline((item,), (), (), 8, 0), ())
    assert records[0]["hidden_frames"] == (0 if covered == "shadow" else 8)
    assert number_visibility_assert(list(records))["pass"] == (covered == "shadow")
    if covered == "shadow":
        assert records[0]["worst_visible_share"] == 1
        ordinary = measure_behind_layers(tmp_path, matte, info, [layer],
                                         hidden_glyph_share=.1, report_worst=True)
        assert ordinary[0]["hidden_frames"] == 8
        assert ordinary[0]["worst_visible_share"] < .9


def test_assembly_redraws_person_over_number_and_measures_encoded_picture(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.matte import project as matte_cache

    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = picture_project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    synth.picture_source(root / "number.mov",
                         "color=yellow:s=200x100:r=30,format=rgba,"
                         "geq=r=255:g=255:b=0:a='255*between(X,20,179)*between(Y,20,79)'",
                         63, alpha=True)
    element = ResolvedElement("agent-number", "graphic", "agent", 15, 78, "behind",
                              {"template": "big_number", "slots": {"end": 10}})
    layer = {"id": element.id, "kind": "graphic", "layer": "behind", "path": "number.mov",
             "x": 80, "y": 100, "width": 200, "height": 100, "start_frame": 15, "frames": 63}
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder: synthetic_matte(
        cut, folder, box=(80, 100, 100, 100)))
    out = root / "picture-final.mp4"
    result = render_picture(root, out, work_dir=root / "work",
        timeline=ResolvedTimeline((element,), (), (), 90, 0), title_layers=[layer],
        caption_layer={"path": str(pictures["captions"]), "x": 60, "y": 1800,
                       "width": 100, "height": 80, "start_frame": 0, "frames": 90})
    for frame in (15, 45, 77):
        person = synth.picture_pixels(out, frame, box=(110, 130, 20, 20)).mean(axis=(0, 1))
        assert person == pytest.approx((255, 0, 0), abs=8)
        number = synth.picture_pixels(out, frame, box=(220, 130, 20, 20)).mean(axis=(0, 1))
        assert number == pytest.approx((255, 255, 0), abs=8)
    for frame in (14, 78):
        room = synth.picture_pixels(out, frame, box=(220, 130, 20, 20)).mean(axis=(0, 1))
        assert room == pytest.approx((0, 0, 255), abs=8)
    assert len(result.behind_layers) == 1
    assert result.behind_layers[0]["template"] == "big_number"
    assert result.behind_layers[0]["hidden_frames"] == 63
    assert result.behind_layers[0]["worst_visible_share"] == .5
    assert not number_visibility_assert(list(result.behind_layers))["pass"]
