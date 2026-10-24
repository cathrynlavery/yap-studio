"""Wide title geometry, chapter navigation, and real alpha layers."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements

from yap_studio import ffmpeg, paths
from yap_studio.canvas import require_supported
from yap_studio.errors import ValidationError, YapError
from yap_studio.faces.clearance import gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.stages.captions import render_layer_plan
from yap_studio.style.loader import ResolvedStyle, resolve_style_data
from yap_studio.wide_titles import wide_layer_plan
from yap_studio.youtube_chapters import chapter_list


def style():
    data = resolve_style_data({"schema_version": 1, "name": "Wide sample",
                               "hook": {"style": "quiet_serif", "font": "Inter"},
                               "chapters": {"style": "title_card"}})
    return ResolvedStyle(data["name"], "test", data)


@pytest.mark.parametrize("kind", ["title", "chapter", "name_tag"])
@pytest.mark.parametrize("face", [Box(300, 240, 320, 350), Box(1240, 240, 320, 350),
                                  Box(800, 240, 320, 350)])
def test_wide_plans_clear_faces_and_stay_title_safe(kind, face):
    plan, evidence = wide_layer_plan(kind, {"title": "Cutting the rambling",
                                           "name": "Sam Porter", "role": "Writer and founder"},
                                     style(), frames=48, faces=[[face]], number=2, total=4)
    assert (plan.width, plan.height, plan.shape) == (1920, 1080, "horizontal")
    box = Box(**evidence["box"])
    assert 96 <= box.x < box.x1 <= 1824
    assert 108 <= box.y < box.y1 <= 972
    assert gap_px(box, head_box(face)) >= 24
    assert evidence["status"] == "clear"
    if kind == "chapter":
        assert (box.x > face.x) == (face.x < 800)


def test_wide_title_balances_lines_and_keeps_brand():
    chosen = style()
    chosen.data["hook"]["font"] = "Gelasio"
    chosen.data["graphics"]["accent_color"] = "#AB1256"
    plan, _ = wide_layer_plan("title", {"title": "How I edit a twenty minute video "
                                             "without opening an editor"},
                              chosen, frames=60)
    lines = json.loads(plan.variables["lines"])
    assert 1 <= len(lines) <= 2
    assert all(len(line.split()) >= 2 for line in lines)
    assert plan.variables["font_name"] == "Gelasio"
    assert plan.variables["accent_color"] == "#AB1256"
    with pytest.raises(ValidationError, match="shorten"):
        wide_layer_plan("title", {"title": "W" * 300}, chosen, frames=60)


def test_chapter_bar_done_current_later():
    for number in range(1, 5):
        plan, _ = wide_layer_plan("chapter", {"title": "The next step"}, style(),
                                  frames=48, number=number, total=4)
        assert json.loads(plan.variables["states"]) == (
            ["done"] * (number - 1) + ["current"] + ["later"] * (4 - number))


def test_chapter_uses_fitting_wide_card_before_narrow_fallback():
    plan, evidence = wide_layer_plan(
        "chapter", {"title": "Understanding the fundamentals of editing"}, style(), frames=48)
    assert evidence["box"]["w"] == 640
    assert plan.variables["size_px"] == 46
    assert json.loads(plan.variables["lines"]) == ["Understanding the", "fundamentals of editing"]


def test_chapter_with_no_fitting_width_offers_plain_fix():
    with pytest.raises(ValidationError, match="shorten"):
        wide_layer_plan("chapter", {"title": "W" * 300}, style(), frames=48)


def chapter(seconds, title):
    return {"start_frame": round(seconds * 30), "title": title}


def test_youtube_chapters_format_and_opening():
    rows = [chapter(15, "Plan the story"), chapter(75, "Make the cut"), chapter(3610, "Publish")]
    text, warning = chapter_list(rows, 3620 * 30, fps=30, opening="Opening title")
    assert warning is None
    assert text == "0:00 Opening title\n0:15 Plan the story\n1:15 Make the cut\n1:00:10 Publish\n"
    text, warning = chapter_list([chapter(0, "Start"), chapter(10, "Middle"),
                                 chapter(20, "End")], 900, fps=30)
    assert warning is None and text == "0:00 Start\n0:10 Middle\n0:20 End\n"


@pytest.mark.parametrize("rows,duration,fix", [
    ([chapter(0, "Only one")], 900, "three"),
    ([chapter(5, "Early"), chapter(20, "Next"), chapter(40, "Last")], 1800, "10 seconds"),
    ([chapter(0, "Start"), chapter(10, "Middle"), chapter(20, "End")], 899, "10 seconds"),
    ([chapter(0, "Start"), chapter(0, "Duplicate"), chapter(20, "End")], 1200, "10 seconds"),
])
def test_youtube_invalid_chapters_offer_only_a_plain_fix(rows, duration, fix):
    text, warning = chapter_list(rows, duration, fps=30)
    assert text is None and fix in warning


def test_no_chapters_offer_nothing():
    assert chapter_list([], 18000, fps=30) == (None, None)


@pytest.mark.parametrize("kind,payload", [("title", {}), ("chapter", {}),
                                         ("graphic", {"template": "name_tag"})])
def test_horizontal_refusal_lifts_only_supported_layers(tmp_path, kind, payload):
    from test_landscape import _context

    from yap_studio.project.model import Anchor, Element

    context = _context(tmp_path)
    item = Element("agent-test", kind, Anchor(time=0), fields=payload)
    require_supported(context, {"agent": SimpleNamespace(elements=[item], lane="agent")})
    item = replace(item, kind="graphic", fields={"template": "steps"})
    with pytest.raises(YapError, match="graphic"):
        require_supported(context, {"agent": SimpleNamespace(elements=[item], lane="agent")})


@pytest.mark.render
@pytest.mark.parametrize("kind", ["title", "chapter", "name_tag"])
def test_real_wide_layer_at_delivery_size(tmp_path: Path, kind):
    plan, evidence = wide_layer_plan(kind, {"title": "Cutting the rambling", "name": "Sam Porter",
                                           "role": "Writer and founder"}, style(), frames=30,
                                     number=2, total=4)
    with paths.operation():
        result = render_layer_plan(plan, tmp_path / "work", tmp_path / "layer.mov")
    info = ffmpeg.probe(result.path)
    assert (info.width, info.height, info.video_frames) == (1920, 1080, 30)
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(result.path),
                      "-frames:v", "1", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                     binary=True, timeout_s=30).stdout
    import numpy as np

    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 4)
    x, y, w, h = (round(evidence["box"][key]) for key in ("x", "y", "w", "h"))
    assert (pixels[y:y+h, x:x+w, 3] > 240).sum() > 1000
    assert pixels[0, 0, 3] == 0
    opaque = pixels[..., 3] > 240
    opaque[y:y+h, x:x+w] = False
    if kind == "chapter":
        opaque[1010:1030, 116:1804] = False
    assert not opaque.any(), "Opaque content escaped its planned title-safe bounds"


@pytest.mark.parametrize("kind,text,expected", [
    ("name_tag", {"name": "Shit Porter", "role": "Writer"}, ["S*** Porter", "Writer"]),
    ("name_tag", {"name": "Sam Porter", "role": "Shit happens"}, ["Sam Porter", "S*** happens"]),
    ("title", {"lines": ["Shit happens"]}, ["S*** happens"]),
    ("chapter", {"title": "Shit happens"}, ["S*** happens"]),
])
def test_wide_masked_layers_pass_bleep_check(layer_project, kind, text, expected):
    import numpy as np
    from test_stage_captions import republish_cut

    from yap_studio.project.status import ProjectContext
    from yap_studio.sfx import bleep
    from yap_studio.stages.graphics import graphics_snapshot

    project = json.loads((layer_project / "project.json").read_text())
    project.update(shape="horizontal", word_fixes={})
    (layer_project / "project.json").write_text(json.dumps(project))
    chosen = style().data
    chosen["music"]["bleep_swears"] = True
    (layer_project / "style.json").write_text(json.dumps(chosen))
    republish_cut(layer_project)
    payload = {"template": kind, "slots": text} if kind == "name_tag" else text
    set_elements(layer_project, {"id": "agent-masked", "kind": "graphic"
                                 if kind == "name_tag" else kind,
                                 "start": {"time": 0}, "hold_s": 1, **payload})
    context = ProjectContext(layer_project)
    snapshot = graphics_snapshot(context)
    layer, = snapshot.titles
    drawn = ([layer.plan.variables["name"], layer.plan.variables["role"]]
             if kind == "name_tag" else json.loads(layer.plan.variables["lines"]))
    assert drawn == expected
    placements = [layer.placement()]
    audio = np.zeros((3 * bleep.RATE, 1), dtype=np.float32)
    result = bleep.check([], music=chosen["music"], voice=audio, original_voice=audio,
                         tone_stem=None, phrases={}, title_layers=placements,
                         timeline=context.resolved_timeline())
    assert result["unmasked"] == []
    assert result["pass"]
    assert placements[0]["display_text"] == expected
    restored = graphics_snapshot(ProjectContext(layer_project), placements=placements)
    assert restored.titles[0].placement()["display_text"] == expected
    assert restored.inputs == snapshot.inputs


@pytest.mark.parametrize("captions", ["on", "off", "partial_gap"])
def test_wide_snapshot_round_trips_real_timeline_and_receipts(layer_project, captions):
    from test_stage_captions import republish_cut

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.captions import caption_layer_plan
    from yap_studio.stages.graphics import graphics_snapshot

    project = json.loads((layer_project / "project.json").read_text())
    project.update(shape="horizontal", word_fixes={})
    (layer_project / "project.json").write_text(json.dumps(project))
    (layer_project / "style.json").write_text(json.dumps(style().data))
    republish_cut(layer_project)
    elements = [{"id": "agent-title", "kind": "title", "start": {"time": 0},
                  "hold_s": .5, "lines": ["Small clear steps"]},
                 {"id": "agent-chapter", "kind": "chapter", "start": {"time": .6},
                  "hold_s": 1, "title": "Make it clear"},
                 {"id": "agent-name", "kind": "graphic", "template": "name_tag",
                  "slots": {"name": "Sam Porter", "role": "Writer and founder"},
                  "start": {"time": 1.7}, "end": {"time": 2.7}}]
    if captions != "on":
        elements.append({"id": "agent-gap", "kind": "caption_gap", "start": {"time": 0},
                         "hold_s": 3 if captions == "off" else .5})
    set_elements(layer_project, *elements)
    snapshot = graphics_snapshot(ProjectContext(layer_project))
    assert [item.kind for item in snapshot.titles] == ["title", "chapter", "graphic"]
    chosen = ProjectContext(layer_project).render_style()
    band = caption_layer_plan([], chosen, frames=90, shape="horizontal")
    for item, old_top in zip(snapshot.titles, (863, 300, 776), strict=True):
        box = Box(**item.clearance["box"])
        if captions == "off":
            assert box.y == old_top
        else:
            assert box.y1 + chosen.data["graphics"]["caption_gap_px"] <= band.y
    restored = graphics_snapshot(ProjectContext(layer_project),
                                 placements=[item.placement() for item in snapshot.titles])
    assert restored.inputs == snapshot.inputs
    blocked = style().data
    blocked["graphics"]["card_kinds_off"] = ["name_tag"]
    (layer_project / "style.json").write_text(json.dumps(blocked))
    with pytest.raises(ValidationError, match="name_tag"):
        graphics_snapshot(ProjectContext(layer_project))


@pytest.mark.render
def test_wide_layers_with_captions_pass_final_clearance(tmp_path, monkeypatch):
    from test_landscape import _render_project

    from yap_studio.faces import detect
    from yap_studio.project.receipts import read_receipt
    from yap_studio.stages.assemble import assemble
    from yap_studio.stages.captions import captions
    from yap_studio.stages.graphics import graphics

    monkeypatch.setattr(detect, "unavailable_reason", lambda: "Synthetic fixture")
    root = _render_project(tmp_path, "horizontal")
    chosen = json.loads((root / "style.json").read_text())
    chosen.update(hook={"style": "quiet_serif", "font": "Inter"},
                  chapters={"style": "title_card"})
    (root / "style.json").write_text(json.dumps(chosen))
    set_elements(root,
                 {"id": "agent-title", "kind": "title", "start": {"time": 0},
                  "hold_s": .6, "lines": ["Small clear steps"]},
                 {"id": "agent-chapter", "kind": "chapter", "start": {"time": .7},
                  "hold_s": 1, "title": "Understanding the fundamentals of editing"},
                 {"id": "agent-name", "kind": "graphic", "template": "name_tag",
                  "slots": {"name": "Sam Porter", "role": "Writer and founder"},
                  "start": {"time": 1.8}, "end": {"time": 2.8}})
    captions(root)
    graphics(root)
    result = assemble(root)
    assert result["status"] == "verified", result
    report = read_receipt(root / "final")["verify"]
    assert report["checks"]["captions.cards"]["status"] == "pass"
    assert not any(item["check"] == "captions.cards" for item in report["findings"])


@pytest.mark.render
@pytest.mark.parametrize("kind,digest", [
    ("title", "5f6673bf486d0eaaaacfc73f77519959c9ca7654ba037a77d8d2b09c4fa19e89"),
    ("chapter", "344d8526150233517dba1cf13090ec474faddc9d83f72f85e872c47b8bcdee61"),
    ("name_tag", "9667561b01a1e88f7a87e443afe49d62ded648cddd3596437372a1b1485f2a0e"),
])
def test_vertical_layer_bytes_match_main_5e1876b(tmp_path, kind, digest):
    """Measured by rendering the same inputs from an untouched archive of main."""
    from yap_studio import cards, receipt
    from yap_studio.stages.captions import chapter_layer_plan, title_layer_plan

    chosen = style()
    if kind == "name_tag":
        plan = cards.card_layer_plan(kind, {"name": "Sam Porter", "role": "Writer and founder"},
                                     chosen, frames=30)
        with paths.operation():
            result = cards.render_card(plan, tmp_path / "work", tmp_path / "layer.mov")
    else:
        plan = (title_layer_plan("Small", "steps", chosen, frames=30) if kind == "title"
                else chapter_layer_plan("Make it clear", 2, chosen, frames=30))
        with paths.operation():
            result = render_layer_plan(plan, tmp_path / "work", tmp_path / "layer.mov")
    assert receipt.sha256_file(result.path) == digest


@pytest.mark.parametrize("mask", [False, True])
def test_chapter_evidence_uses_retained_graphics_text(layer_project, mask):
    from test_stage_captions import republish_cut

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.assemble import verification_edl
    from yap_studio.stages.graphics import graphics_snapshot

    project = json.loads((layer_project / "project.json").read_text())
    project.update(shape="horizontal", word_fixes={})
    (layer_project / "project.json").write_text(json.dumps(project))
    chosen = style().data
    chosen["music"].update(bleep_swears=mask, bleep_words_add=["rambling"])
    (layer_project / "style.json").write_text(json.dumps(chosen))
    republish_cut(layer_project)
    set_elements(layer_project,
                 {"id": "agent-title", "kind": "title", "start": {"time": 0},
                  "hold_s": .5, "lines": ["Twenty rambling steps"]},
                 {"id": "agent-chapter", "kind": "chapter", "start": {"time": 1},
                  "hold_s": 1, "title": "Less rambling"})
    context = ProjectContext(layer_project)
    rendered = graphics_snapshot(context)
    placements = [item.placement() for item in rendered.titles]
    expected = ["20 r******* steps", "Less r*******"] if mask else [
        "20 rambling steps", "Less rambling"]
    assert [" ".join(item["display_text"]) for item in placements] == expected
    assert [" ".join(json.loads(item.plan.variables["lines"]))
            for item in rendered.titles] == expected
    # Later lane and style edits cannot rewrite this delivery's retained snapshot.
    snapshot = SimpleNamespace(timeline=context.resolved_timeline(), title_layers=placements,
                               caption_layer=None, grade=None, media={}, sources={},
                               spans={"ranges": []}, small_files={
                                   "project.json": json.dumps(project).encode()})
    set_elements(layer_project)
    chosen["music"]["bleep_swears"] = not mask
    (layer_project / "style.json").write_text(json.dumps(chosen))
    evidence = verification_edl(snapshot, base=layer_project / "final")["youtube_chapters"]
    assert evidence["opening"] == expected[0]
    assert evidence["chapters"] == [{"start_frame": 30, "title": expected[1]}]
    assert evidence["fps"] == "30/1"


def test_verify_warns_with_fix_and_no_invalid_list(tmp_path):
    import synth
    from test_verify import _ranges

    from yap_studio.verify import verify

    media = synth.calib_source(tmp_path / "source.mov", seconds=1, size="192x108")
    edl = _ranges(tmp_path / "edit.json", media, [(0, 1)], youtube_chapters={
        "chapters": [chapter(0, "Start"), chapter(.3, "Middle"), chapter(.6, "End")],
        "opening": "Opening", "total_frames": 30,
    })
    report = verify(str(media), edl=str(edl), asr="none", loudness="note", detail="full")
    finding, = [item for item in report["findings"] if item["check"] == "platform.chapters"]
    assert finding["severity"] == "warn" and "10 seconds" in finding["suggested_action"]
    assert "youtube_chapters" not in report
    assert report["checks"]["platform.chapters"]["status"] == "warn"
