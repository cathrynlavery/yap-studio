"""Synthetic tall page timing, cut framing and picture contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from test_planner_broll import catalog_item, phrase_words, requests
from test_timeline import add, overlay
from test_timeline import project as project

from yap_studio.planner import broll
from yap_studio.project.model import ProjectFileError, parse_document
from yap_studio.stages import green_screen, tall_box
from yap_studio.stages import timeline as stage
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline


def test_auto_framing_measures_every_third_frame_and_reports_minimums() -> None:
    import numpy as np

    small = np.zeros((1920, 1080), dtype=np.uint8)
    small[600:1000, 400:600] = 255
    large = np.zeros_like(small)
    large[200:1500, 200:900] = 255
    measured = tall_box.auto_framing([small, large, large, small], 0)
    assert measured["chosen"] == "keep"
    assert measured["visible_min"] == pytest.approx(1 - 80000 / (620 * 860))
    assert measured["above_min_px"] == 460
    refused = tall_box.auto_framing([small, small, small, large], 0)
    assert refused == {"chosen": "shrink", "visible_min": pytest.approx(60 / 860),
                       "above_min_px": 60}


def test_auto_framing_dilates_person_before_deciding() -> None:
    import numpy as np

    matte = np.zeros((1920, 1080), dtype=np.uint8)
    matte[450:900, 400:600] = 128
    assert tall_box.auto_framing([matte], 0)["chosen"] == "keep"
    result = tall_box.auto_framing([matte], 24)
    assert result["chosen"] == "shrink"
    assert result["above_min_px"] == 286
    matte[:] = 127
    assert tall_box.auto_framing([matte], 24)["visible_min"] == 1


def test_auto_framing_requires_half_the_box_clear_even_with_headroom() -> None:
    import numpy as np

    matte = np.zeros((1920, 1080), dtype=np.uint8)
    matte[580:1500, 230:850] = 255
    assert tall_box.auto_framing([matte], 0)["chosen"] == "keep"
    result = tall_box.auto_framing([matte], 24)
    assert result["chosen"] == "shrink"
    assert result["visible_min"] == pytest.approx(416 / 860)
    assert result["above_min_px"] == 416


@pytest.mark.parametrize("source,width,x", [(200, 620, 230), (800, 800, 140), (1400, 1080, 0)])
def test_shrink_page_width_is_clamped_and_centred(source, width, x) -> None:
    assert tall_box.shrink_page_box(source) == (x, 0, width, 1920)


@pytest.mark.parametrize("framing,window", [("auto", False), ("keep", False), ("shrink", True)])
def test_tall_framing_layout_windows(framing, window) -> None:
    item = ResolvedElement("broll-page", "overlay", "broll", 30, 120, "front",
        {"placement": "tall_behind", "tall_framing": framing})
    timeline = ResolvedTimeline((item,), (), (), 150, 0)
    expected = [(30, 120, (108, 384, 864, 1536), 0)] if window else []
    assert green_screen.layout_windows(timeline) == expected
    item.payload["framing"] = {"chosen": "shrink"}
    assert green_screen.layout_windows(timeline) == [(30, 120, (108, 384, 864, 1536), 0)]


@pytest.mark.parametrize("framing", ["keep", "shrink"])
def test_tall_composite_has_no_room_fill(framing) -> None:
    from yap_studio.stages import assemble_picture as picture

    item = ResolvedElement("broll-page", "overlay", "broll", 6, 81, "front", {
        "tall_framing": framing, "box_px": [230, 140, 620, 860], "fill_color": "#F2D8AA"})
    graph = picture._Graph()
    picture._tall_composite(graph, "room", "person", "page", item)
    filters = ";".join(graph.filters)
    assert "fillborders" not in filters and "gblur" not in filters
    if framing == "keep":
        assert "scale=" not in filters
        assert "overlay=x=230:y=140" in filters
        assert "crop=620:860:230:140:exact=1" in filters
    else:
        assert "scale=864:1536" in filters
        assert "color=c=0xF2D8AA:s=1080x1920" in filters
        assert "X/24" in filters
        assert "[room]nullsink" in filters


def test_scroll_timing_and_step_bounds() -> None:
    plan = tall_box.scroll_plan(240, 5000, 860)
    assert [start for start, _, _ in plan] == [15, 55, 95, 135, 175]
    assert all(to_y - from_y <= 860 for _, from_y, to_y in plan)
    offsets = tall_box.scroll_offsets(240, 5000, 860)
    assert offsets[:16] == [0] * 16
    assert offsets[33] == math.floor(plan[0][2] * .75)
    assert offsets == sorted(offsets)
    assert max(offsets) <= 4140
    assert offsets[51:56] == [math.floor(plan[0][2])] * 5
    assert offsets == tall_box.scroll_offsets(240, 5000, 860)
    assert tall_box.scroll_plan(75, 900, 860) == [(15, 0, 40)]
    assert tall_box.scroll_plan(75, 10000, 860) == [(15, 0, 860)]
    assert tall_box.scroll_plan(62, 1000, 860) == []
    assert tall_box.scroll_offsets(75, 860, 860) == [0] * 75


@pytest.mark.parametrize("kind,media_type,height,expected", [
    ("image", "screenshot", 1599, False), ("image", "screenshot", 1600, True),
    ("image", "photo", 2400, False), ("video", "screenshot", 2400, False),
])
def test_long_page_catalog_rule(kind, media_type, height, expected) -> None:
    assert tall_box.is_long_page(dict(kind=kind, media_type=media_type,
                                      width=1000, height=height)) == expected


@pytest.mark.parametrize("frames,page_h,box_h", [(63, 900, 860), (240, 5000, 860),
                                                   (200, 1921, 400), (240, 900, 860)])
def test_crop_expression_matches_every_integer(frames: int, page_h: int, box_h: int) -> None:
    expression = tall_box.scroll_expression(frames, page_h, box_h)
    expression = expression.replace("if(", "choose(")
    values = [eval(expression, {"__builtins__": {}, "choose": lambda c, a, b: a if c else b,
                                "lt": lambda a, b: a < b, "floor": math.floor,
                                "pow": pow, "n": n}) for n in range(frames)]
    assert values == tall_box.scroll_offsets(frames, page_h, box_h)


@pytest.mark.parametrize("kind,media_type,height,box,motion", [
    ("image", "screenshot", 2400, "tall_behind", "scroll"),
    ("image", "photo", 300, "headroom", "none"),
])
def test_planner_defaults_and_explicit_values(tmp_path, kind, media_type, height, box, motion):
    item = catalog_item(tmp_path, kind=kind)
    item.update(width=1000, height=height, media_type=media_type)
    style = {"broll": {"overlay_box": "tall_behind", "overlay_motion": "scroll",
                       "photo_placement": "overlay", "screenshot_placement": "overlay"}}
    words = phrase_words("Dog", "park", "with", "a", "page.")
    request = requests(kind="overlay", end={"word": "t1:4", "text": "page."})
    result = broll.plan_broll(request, words, [item], style,
                              project_dir=tmp_path, frames=90).lane()["elements"][0]
    assert (result["placement"], result["motion"]) == (box, motion)
    explicit = broll.plan_broll(requests(kind="overlay", box="tall_behind", motion="scroll"),
                                phrase_words(), [item], style, project_dir=tmp_path,
                                frames=90).lane()["elements"][0]
    assert (explicit["placement"], explicit["motion"]) == ("tall_behind", "scroll")
    suggestion = broll.suggest_broll(words, [item], style)["phrases"][0]["candidates"][0]
    assert (suggestion["box"], suggestion["motion"]) == (box, motion)


def test_named_tall_box_forbids_custom_box() -> None:
    item = {"id": "broll-page", "kind": "overlay", "start": {"time": 0}, "hold_s": 3,
            "media": {"path": "page.png", "sha256": "a" * 64}, "placement": "tall_behind",
            "motion": "scroll"}
    assert parse_document("timeline-fragment", {"schema": "yap-studio/timeline@1",
                                               "elements": [item]}).elements
    with pytest.raises(ProjectFileError):
        parse_document("timeline-fragment", {"schema": "yap-studio/timeline@1",
                                             "elements": [{**item, "box": [0, 0, 20, 30]}]})


def test_tall_layout_windows() -> None:
    item = ResolvedElement("broll-page", "overlay", "broll", 30, 120, "front",
                           {"placement": "tall_behind", "tall_framing": "shrink"})
    timeline = ResolvedTimeline((item,), (), (), 150, 0)
    assert green_screen.layout_windows(timeline) == [(30, 120, (108, 384, 864, 1536), 0)]


@pytest.fixture
def page(project: Path, monkeypatch):
    from test_assemble_picture import overlay_image

    source = overlay_image(project / "page.png", "color=white:s=620x2400")
    from yap_studio import receipt

    monkeypatch.setattr(stage, "_picture_cuts", lambda *args: (0, 90, 180, 270, 360))
    return overlay(project, placement="tall_behind", motion="scroll", start={"time": .2},
                   end={"time": 3.2}, media={"path": source.name,
                                           "sha256": receipt.sha256_file(source)})


def test_tall_snaps_both_edges_and_resolves_scroll(project, page) -> None:
    result = add(project, page).require_valid()
    item, = result.elements
    assert (item.start_frame, item.end_frame) == (0, 90)
    assert item.payload["cut_frames"] == [0, 90]
    assert "wide_box_px" not in item.payload
    from yap_studio.style.loader import _schema
    assert item.payload["tall_framing"] == (
        _schema()["properties"]["broll"]["properties"]["tall_framing"]["default"])
    assert item.payload["scroll"]["page_px"] == [620, 2400]
    assert item.payload["scroll"]["flicks"] == [list(row) for row in
                                                tall_box.scroll_plan(90, 2400, 860)]
    assert not any(f.rule == "overlay_length" for f in result.findings)


@pytest.mark.parametrize("end,rule", [(2, "tall_box_length"), (9, "tall_box_length"),
                                      (4.5, "tall_box_cut")])
def test_tall_cut_and_length_refusals(project, page, end, rule) -> None:
    result = add(project, {**page, "end": {"time": end}})
    finding = next(f for f in result.findings if f.rule == rule)
    assert finding.severity == "violation"
    if rule == "tall_box_cut":
        assert "the end is 45 frames (1.50 s) from the nearest picture cut" in finding.message
        assert "gamma" in finding.suggested_action
        assert "zeta" in finding.suggested_action
        assert "first word after a cut" in finding.suggested_action


def test_tall_start_cut_message_includes_frames_and_seconds(project, page) -> None:
    result = add(project, {**page, "start": {"time": 1.5}, "end": {"time": 6}})
    finding = next(f for f in result.findings if f.rule == "tall_box_cut")
    assert "the start is 45 frames (1.50 s) from the nearest picture cut" in finding.message


@pytest.mark.parametrize("frames,valid", [(74, False), (75, True), (240, True), (241, False)])
def test_tall_length_boundaries(project, page, monkeypatch, frames, valid) -> None:
    monkeypatch.setattr(stage, "_picture_cuts", lambda *args: (0, frames, 360))
    result = add(project, {**page, "start": {"time": 0}, "end": {"time": frames / 30}})
    assert any(f.rule == "tall_box_length" for f in result.findings) != valid
    assert not any(f.rule == "overlay_length" for f in result.findings)


@pytest.mark.parametrize("case,rule", [("clip", "tall_box_media"), ("short", "tall_box_media"),
                                      ("scroll_clip", "overlay_scroll"),
                                      ("scroll_short", "overlay_scroll"),
                                      ("scroll_brief", "overlay_scroll")])
def test_media_and_scroll_refusals(project, page, monkeypatch, case, rule) -> None:
    from test_assemble_picture import overlay_image

    from yap_studio import receipt

    item = {**page, "start": {"time": 0}, "end": {"time": 3}}
    if case.endswith("clip"):
        item["media"] = overlay(project)["media"]
    if case.endswith("short"):
        source = overlay_image(project / "short.png", "color=white:s=620x180")
        item["media"] = {"path": source.name, "sha256": receipt.sha256_file(source)}
    if case.startswith("scroll_"):
        item["placement"] = "headroom"
    if case == "scroll_brief":
        item["end"] = {"time": 62 / 30}
    result = add(project, item)
    assert any(f.rule == rule and f.severity == "violation" for f in result.findings)


def test_scroll_in_other_box_resolves_and_exact_height_tall_can_stay_still(project, page) -> None:
    from test_assemble_picture import overlay_image

    from yap_studio import receipt

    item = {**page, "placement": "side_left", "start": {"time": 0}, "end": {"time": 3}}
    result = add(project, item).require_valid()
    assert result.elements[0].payload["scroll"]["page_px"] == [440, 1703]
    source = overlay_image(project / "exact.png", "color=white:s=620x860")
    result = add(project, {**page, "id": "broll-exact", "motion": "none",
                           "start": {"time": 6}, "end": {"time": 9},
                           "media": {"path": source.name,
                                     "sha256": receipt.sha256_file(source)}}).require_valid()
    assert len(result.elements) == 2


@pytest.mark.parametrize("position,top,size,refused", [
    ("top", 999, 76, True), ("middle", 1180, 118, True),
    ("middle", 1380, 118, False), ("bottom", 1500, 118, False),
])
@pytest.mark.parametrize("framing", ["auto", "keep", "shrink"])
def test_tall_caption_clearance(
    project, page, monkeypatch, position, top, size, refused, framing,
) -> None:
    from yap_studio.project.status import ProjectContext

    original = ProjectContext.render_style

    def render_style(context):
        style = original(context)
        return replace(style, data={**style.data, "broll": {
            **style.data["broll"], "tall_framing": framing}, "captions": {**style.data["captions"],
            "position": position, "top_px": top, "size_px": size}})

    monkeypatch.setattr(ProjectContext, "render_style", render_style)
    result = add(project, page)
    findings = [f for f in result.findings if f.rule == "tall_box_captions"]
    assert bool(findings) == (refused and framing != "shrink")
    if findings:
        assert findings[0].suggested_action == "Use Bottom captions, or move captions lower."


@pytest.mark.parametrize("kind", ["zoom", "insert", "background", "transition", "moment",
                                  "grade_compare", "behind", "overlay", "front", "title",
                                  "opening"])
def test_tall_overlap_rules(project, page, kind) -> None:
    import json

    from test_timeline import graphic, insert, opening, opening_story, write

    from yap_studio.project.model import load_cut_words

    base = {"id": "agent-other", "kind": kind, "start": {"time": 0}, "hold_s": 3}
    if kind in {"behind", "front"}:
        base = graphic(id="agent-other", start={"time": 0}, hold_s=3, layer=kind)
    elif kind == "zoom":
        base.update(move="punch", scale=1.5, ease="cut", center=[.5, .5])
    elif kind == "opening":
        opening_story(project, [("alpha", 0, .4), ("um.", .5, 1.1)])
        base = opening(id="agent-other")
    elif kind == "moment":
        base["style"] = "outline"
    elif kind == "title":
        base["lines"] = ["alpha"]
    elif kind == "transition":
        base = {"id": "agent-other", "kind": "transition", "look": "whip_pan",
                "start": {"word": "t1:3", "edge": "start"}, "frames": 12}
    elif kind in {"insert", "overlay", "background"}:
        base = insert(project, id="agent-other", kind=kind, lane="agent",
                      start={"time": 0}, end={"time": 3})
        if kind == "overlay":
            base["placement"] = "headroom"
    hashes = load_cut_words(project / "cut/cut-words.json").words_sha256
    # Write both lanes so unrelated rule refusals do not hide overlap findings.
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
                                            "words_sha256": hashes, "elements": [base]})
    write(project / "timeline/broll.json", {"schema": "yap-studio/timeline@1",
                                            "words_sha256": hashes, "elements": [page]})
    if kind == "transition":
        # The cut itself is covered separately by the receipt-owned spans test.
        from unittest.mock import patch

        with patch.object(stage, "_transition_joins", return_value={
            ("t1:3", "s1"): (90, replace(load_cut_words(
                project / "cut/cut-words.json").words[2], text="beta.")),
        }):
            result = stage.resolve_timeline(project)
    else:
        result = stage.resolve_timeline(project)
    finding = next(f for f in result.findings if f.rule == "tall_box_overlap")
    assert "card or title" in finding.message
    assert "card or title" in finding.suggested_action
    if kind == "opening":
        assert any(f.rule == "visual_overlap" for f in result.findings)
    assert json.loads((project / "timeline/broll.json").read_text())["elements"] == [page]


@pytest.mark.parametrize("hold_s,overlaps", [(2.5, False), (3.5, True)])
def test_choices_tall_box_overlap(project, page, hold_s, overlaps) -> None:
    from test_timeline import graphic

    add(project, {**page, "start": {"time": 3}, "end": {"time": 6}}).require_valid()
    card = graphic(template="choices", slots={"options": ["One", "Two"]},
                   start={"time": 0}, hold_s=hold_s)
    result = add(project, card, lane="agent")
    findings = [finding for finding in result.findings if finding.rule == "tall_box_overlap"]
    assert bool(findings) == overlaps
    assert result.valid == (not overlaps)
    if overlaps:
        assert findings[0].severity == "violation"
        assert set(findings[0].elements) == {card["id"], page["id"]}
    else:
        result.require_valid()
        choices = next(item for item in result.elements if item.id == card["id"])
        tall = next(item for item in result.elements if item.id == page["id"])
        assert choices.end_frame < tall.start_frame


def test_picture_cuts_use_jumps_not_contiguous_span_joins(project) -> None:
    from test_timeline import write

    from yap_studio.project.receipts import StageRun
    from yap_studio.project.status import ProjectContext

    context = ProjectContext(project)
    record = context.json("cut/receipt.json")
    spans = write(project / "staged-spans.json", {"sources": {"t1": "../source.mov"},
        "fps": "30/1", "segments": [{"source": "t1", "start": 0, "end": 3},
                                      {"source": "t1", "start": 3, "end": 6},
                                      {"source": "t1", "start": 8, "end": 14}]})
    with StageRun(project, "cut") as run:
        run.inputs.update(record["inputs"])
        run.outputs.update({output["path"]: output for output in record["outputs"]})
        run.replace_output(project / "cut/spans.json", spans)
        run.finish(extra={"plan": record["plan"]})
    assert stage._picture_cuts(ProjectContext(project), 360) == (0, 180, 360)
    with pytest.raises(stage.TimelineRefusal, match="frame count"):
        stage._picture_cuts(ProjectContext(project), 359)


def test_tall_cut_requires_receipt_ownership(project) -> None:
    with pytest.raises(stage.TimelineRefusal, match="does not own"):
        stage._picture_cuts(stage.ProjectContext(project), 360)


def test_page_scales_before_loop_and_tall_never_fades(tmp_path) -> None:
    from test_assemble_picture import overlay_image

    from yap_studio.stages import assemble_picture as picture

    source = picture._source(overlay_image(tmp_path / "page.png", "color=white:s=1240x4800"),
                             [], [], owner="page")
    item = ResolvedElement("broll-page", "overlay", "broll", 30, 120, "front", {
        "media": {}, "placement": "tall_behind", "motion": "scroll",
        "box_px": list(tall_box.TALL_BOX_PX)})
    graph = picture._Graph()
    picture._overlay(graph, source, item)
    filters = graph.filters[0]
    assert filters.index("scale=620:2400") < filters.index("loop=loop=-1:size=1")
    assert "fade=" not in filters
    assert "-loop" not in graph.inputs
    assert picture.overlay_alpha_runs(source.path, item) == [
        (30, 120, picture.Box(230, 140, 620, 860))]
    still = replace(item, payload={**item.payload, "motion": "none"})
    assert "y='0':exact=1" in picture._overlay_filters(source, still, image=True)


@pytest.mark.render
@pytest.mark.parametrize("framing", ["keep", "shrink"])
def test_tall_render_person_scroll_framing_and_repeatability(
    tmp_path, monkeypatch, framing,
) -> None:
    import numpy as np
    import synth
    from test_assemble_picture import overlay_image, project
    from test_matte_project import synthetic_matte

    from yap_studio import ffmpeg, receipt
    from yap_studio.matte import project as matte_cache
    from yap_studio.stages import assemble_picture as picture

    active = "enable='between(n,6,80)'"
    plate = synth.picture_source(tmp_path / "plate.mov", "color=black:s=1080x1920:r=30,"
        "drawbox=x=96:y=1200:w=192:h=80:c=white:t=fill,"
        f"drawbox=x=0:y=0:w=1080:h=1920:c=blue:t=fill:{active},"
        f"drawbox=x=0:y=0:w=600:h=1300:c=red:t=fill:{active},"
        f"drawbox=x=100:y=1500:w=80:h=100:c=lime:t=fill:{active}", 90)
    take = synth.picture_source(tmp_path / "take.mov", "color=blue:s=160x90:r=30", 90)
    root = project(tmp_path / "project", {"cut": plate, "take": take}, hold_frames=0)
    page = overlay_image(root / "page.png", "nullsrc=s=620x2400,"
                          "geq=lum='16+mod(Y*17+floor(Y/23)*37,220)':cb=128:cr=128")
    item = ResolvedElement("broll-page", "overlay", "broll", 6, 81, "front", {
        "placement": "tall_behind", "motion": "scroll", "tall_framing": framing,
        "fill_color": "#F2D8AA", "media": {
            "path": page.name, "sha256": receipt.sha256_file(page)},
        "box_px": list(tall_box.TALL_BOX_PX), "cut_frames": [6, 81],
        })
    if framing == "shrink":
        item.payload["box_px"] = [230, 0, 620, 1920]
    timeline = ResolvedTimeline((item,), (), (), 90, 0)
    transparent = synth.picture_source(root / "transparent.mov",
        "color=black@0:s=2x2:r=30,format=rgba", 90, alpha=True)
    caption = {"path": transparent.name, "x": 0, "y": 0, "width": 2, "height": 2,
               "start_frame": 0, "frames": 90}
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(), **kwargs:
                        synthetic_matte(cut, folder, box=(0, 0, 600, 1300),
                                        soft_box=(598, 0, 2, 1300), joins=joins))
    outputs = []
    for index in range(2):
        output = root / f"render-{index}.mp4"
        result = picture.render_picture(root, output, work_dir=root / f"work-{index}",
            timeline=timeline, title_layers=[], caption_layer=caption, cutout="auto")
        outputs.append(output)
        graph = (root / f"work-{index}/picture.ffgraph").read_text()
        assert "person_tall" in graph
        assert "fillborders" not in graph
        assert ("scale=864:1536" in graph) == (framing == "shrink")
        assert result.total_frames == 90
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    baseline = root / "cut-picture.mp4"
    picture.render_picture(root, baseline, work_dir=root / "cut-work",
        timeline=replace(timeline, elements=()), title_layers=[], caption_layer=caption)
    for frame in (5, 81):
        np.testing.assert_array_equal(synth.picture_pixels(outputs[0], frame),
                                      synth.picture_pixels(baseline, frame))
        # The normal compositor's colour conversion can round white by one code value.
        np.testing.assert_allclose(synth.picture_pixels(outputs[0], frame),
                                    synth.picture_pixels(root / "cut/cut.mp4", frame), atol=1)
    pixels = synth.picture_pixels(outputs[0], 30)
    np.testing.assert_allclose(pixels[700:720, 450:470].mean(axis=(0, 1)), [255, 0, 0], atol=8)
    if framing == "keep":
        cut = synth.picture_pixels(baseline, 30)
        outside = np.ones((1920, 1080), dtype=bool)
        outside[138:1002, 228:852] = False
        # Runner encoders can shift a few channel values beyond the usual tolerance.
        difference = np.abs(pixels[outside].astype(float) - cut[outside])
        assert difference.max() <= 16
        assert np.count_nonzero(difference > 10) <= difference.size * 0.00001
    else:
        offset = tall_box.scroll_offsets(75, 2400, 1920)[24]
        expected = np.full((1920, 1080, 3), [242, 216, 170], dtype=np.uint8)
        page_image = synth.picture_pixels(page, 0)
        expected[:, 230:850] = page_image[offset:offset + 1920]
        outside = np.ones((1920, 1080), dtype=bool)
        outside[382:1426, 106:590] = False
        # Chroma subsampling mixes the narrow page borders by up to a few code values.
        np.testing.assert_allclose(pixels[outside].astype(float), expected[outside], atol=25)
        assert (pixels[outside, 2].astype(int) - pixels[outside, 0]).max() <= 12
        # The frame edges fade into the page while the soft matte rim still blends.
        for x, y in ((108, 700), (450, 384)):
            np.testing.assert_allclose(pixels[y, x].astype(float), expected[y, x], atol=10)
        for x, y in ((120, 700), (450, 396), (587, 700)):
            assert 0 < int(pixels[y, x, 0]) - int(pixels[y, x, 1]) < 250
        assert "blend=all_mode=multiply" in graph
        assert "tall_shrink_edge_raw" in graph
    # Decode a thin page strip across every frame to measure its integer translation.
    decoded = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(outputs[0]), "-vf",
        "crop=8:200:800:160,format=gray", "-f", "rawvideo", "-"],
        binary=True, timeout_s=60)
    strips = np.frombuffer(decoded.stdout, np.uint8).reshape(90, 200, 8).mean(axis=2)
    page_pixels = synth.picture_pixels(page, 0)[:, 30, :].mean(axis=1)
    top, height = (160, 1920) if framing == "shrink" else (20, 860)
    candidates = np.stack([page_pixels[top + y:top + 200 + y]
                           for y in range(2400 - height + 1)])
    measured = [int(np.argmin(np.abs(candidates - strip).mean(axis=1)))
                for strip in strips[6:81]]
    assert measured == tall_box.scroll_offsets(75, 2400, height)
