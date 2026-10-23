"""Split geometry, picture pixels and caption bounds with synthetic inputs."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_assemble_picture import assert_colour, insert, lane, layers, picture_command, project
from test_assemble_picture import pictures as pictures
from test_captions_render import _style
from test_cut import make_project, time_segment, write_edit, write_json
from test_media_select import _item
from test_planner_broll import catalog_item, phrase_words, requests
from test_style_broll import BROLL_DEFAULTS

from yap_studio import receipt
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import gap_px, head_box
from yap_studio.faces.detect import find_gaps
from yap_studio.faces.mapping import map_cut_faces, map_picture_faces
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.grade import GradeSpec, apply_rgb
from yap_studio.media.select import REQUEST_SCHEMA, parse_request, rank_candidates
from yap_studio.planner.broll import plan_broll, suggest_broll
from yap_studio.project.model import ProjectFileError, parse_document
from yap_studio.project.receipts import StageRun, projection_sha256
from yap_studio.project.status import ProjectContext, cut_inputs
from yap_studio.stages import assemble_picture as picture
from yap_studio.stages import broll_split as split
from yap_studio.stages.captions import caption_band_geometry, caption_offsets, caption_snapshot
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline, resolve_timeline
from yap_studio.style import resolve, validate_style_data


def measured(boxes: tuple[tuple[Box, ...], ...]) -> FacesData:
    frames = tuple(tuple(FaceBox(box, 0.95, "detected") for box in frame) for frame in boxes)
    return FacesData("measured", None,
                     VideoIdentity("cut.mp4", "a" * 64, len(frames), Fraction(30), 1080, 1920),
                     {}, {}, frames, tuple(find_gaps(frames)))


def element(first: int = 10, last: int = 40, **payload) -> ResolvedElement:
    return ResolvedElement("broll-one", "insert", "broll", first, last, "front",
                           {"media": {"fit": "split"}, **payload})


@pytest.mark.parametrize("face", [Box(440, 650, 200, 280), Box(20, 420, 180, 250),
                                  Box(900, 1000, 160, 300)])
def test_layout_is_integer_deterministic_and_inside_take(face: Box) -> None:
    layout = split.calculate_layout("broll-one", [face])
    assert layout == split.calculate_layout("broll-one", [face])
    assert layout.seam_y == 960
    assert 1 <= layout.scale <= 1.25
    x, y, w, h = layout.crop
    assert all(type(value) is int for value in (*layout.crop, layout.seam_y))
    assert w % 2 == h % 2 == layout.seam_y % 2 == 0
    assert 0 <= x <= 1080 - w and 0 <= y <= 1920 - h
    mapped = layout.map_box(face)
    assert mapped.y + mapped.h / 2 == pytest.approx(
        layout.seam_y + 0.4 * (1920 - layout.seam_y), abs=2,
    )
    assert layout.map_box(head_box(face)).y >= layout.seam_y + 24
    assert json.loads(json.dumps(layout.to_dict()))["crop"] == list(layout.crop)


def test_high_face_moves_crop_up() -> None:
    low = split.calculate_layout("broll-low", [Box(400, 700, 200, 280)])
    high = split.calculate_layout("broll-high", [Box(400, 200, 200, 280)])
    assert high.crop[1] < low.crop[1]
    assert high.map_box(head_box(Box(400, 200, 200, 280))).y >= high.seam_y + 24


def test_highest_head_wins_over_median_position() -> None:
    boxes = [Box(400, y, 200, 280) for y in (300, 600, 620)]
    layout = split.calculate_layout("broll-moving", boxes)
    assert layout.scale == pytest.approx(0.34 * 960 / 280)
    assert layout.crop[1] == 111
    assert all(layout.map_box(head_box(box)).y >= layout.seam_y + 24 for box in boxes)


@pytest.mark.parametrize("boxes", [
    [Box(50, 650, 160, 250), Box(900, 650, 160, 250)],
    [Box(400, 300, 160, 250), Box(400, 1100, 160, 250)],
])
def test_moving_face_refuses_when_one_crop_cannot_hold_window(boxes) -> None:
    with pytest.raises(picture.PictureError, match=(
        "broll-moving: she moves too much for one crop; "
        "use fit cover for this shot or shorten it"
    )):
        split.calculate_layout("broll-moving", boxes)


def test_moving_face_crop_centres_horizontal_union() -> None:
    boxes = [Box(x, 650, 160, 250) for x in (300, 300, 300, 500)]
    layout = split.calculate_layout("broll-moving", boxes)
    x, y, w, h = layout.crop
    assert x + w / 2 == 480
    assert all(x <= box.x and box.x + box.w <= x + w for box in boxes)
    assert all(y <= box.y and box.y + box.h <= y + h for box in boxes)


def test_moving_face_crop_keeps_every_bottom_inside() -> None:
    boxes = [Box(400, y, 160, 250) for y in (600, 600, 600, 940)]
    layout = split.calculate_layout("broll-moving", boxes)
    _, y, _, h = layout.crop
    assert all(box.y + box.h <= y + h for box in boxes)
    assert all(layout.map_box(head_box(box)).y >= layout.seam_y + 24 for box in boxes)


@pytest.fixture
def split_take_project(tmp_path, pictures):
    root = make_project(tmp_path / "project", [pictures["take"], pictures["take"]],
                        word_rows=[[("alpha", 1.1, 1.2)], [("beta", 0.1, 0.2)]])
    document = json.loads((root / "project.json").read_bytes())
    document["takes"][1]["voiceover"] = True
    write_json(root / "project.json", document)
    write_edit(root, [time_segment("s1", "t2", 0, 1),
                      time_segment("s2", "t1", 1, 2),
                      time_segment("s3", "t2", 1, 2)], end_hold_s=0)
    word_map = root / "words.json"
    write_json(word_map, {"schema": "yap-studio/cut-words@1", "time_base": "output",
                          "words_sha256": {take: receipt.sha256_file(
                              root / f"ingest/{take}/words.json") for take in ("t1", "t2")},
                          "words": []})
    spans = root / "spans.json"
    write_json(spans, {"sources": {take: f"../takes/{take}.mov" for take in ("t1", "t2")},
                       "fps": "30/1", "segments": [
                           {"source": "t2", "start": "0", "end": "1"},
                           {"source": "t1", "start": "1", "end": "2"},
                           {"source": "t2", "start": "1", "end": "2"}]})
    staged = root / "picture.mp4"
    shutil.copyfile(pictures["cut"], staged)
    with StageRun(root, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        for name, source in (("cut.mp4", staged), ("cut-words.json", word_map),
                             ("spans.json", spans)):
            run.replace_output(root / "cut" / name, source)
        run.finish(extra={"plan": {"total_frames": 90}})
    return root


@pytest.mark.parametrize(("voiceover_only", "first", "last", "refused"), [
    (True, 30, 60, True), (False, 0, 30, True), (False, 29, 31, True),
    (False, 30, 60, False), (False, 60, 90, True),
])
def test_split_voiceover_timeline_maps_output_spans(
    split_take_project, pictures, voiceover_only, first, last, refused,
) -> None:
    root = split_take_project
    if voiceover_only:
        document = json.loads((root / "project.json").read_bytes())
        document["takes"][0]["voiceover"] = True
        write_json(root / "project.json", document)
    lane(root, [insert(root, pictures["insert"], first, last, fit="split")], name="broll")
    timeline = resolve_timeline(root)
    findings = [item for item in timeline.findings if item.rule == "split_voiceover"]
    assert bool(findings) is refused
    if refused:
        assert not timeline.valid
        assert findings[0].severity == "violation"
        assert findings[0].elements == ("broll-one",)
        assert findings[0].bare_message == (
            "split needs the take under it, and this part is voiceover")
        assert findings[0].suggested_action == "set fit cover for this shot"
    else:
        assert timeline.valid


def test_split_voiceover_picture_guard_refuses_supplied_timeline(
    split_take_project, monkeypatch,
) -> None:
    monkeypatch.setattr(picture, "_source", lambda *args, **kwargs: pytest.fail(
        "Voiceover split must be refused before decoding picture"))
    root = split_take_project
    timeline = ResolvedTimeline((element(29, 31),), (), (), 90, 0)
    with pytest.raises(picture.PictureError, match="broll-one.*voiceover.*set fit cover"):
        picture.render_picture(root, root / "out.mp4", work_dir=root / "work", timeline=timeline)


@pytest.mark.parametrize(("fit", "letters", "refused"), [
    ("split", True, True), ("cover", True, False), ("split", False, False),
])
def test_letters_split_timeline_violation(tmp_path, pictures, fit, letters, refused) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    selected = json.loads((root / "picture-style.json").read_bytes())
    selected["broll"]["emphasis_on"] = ["video_letters"]
    write_json(root / "picture-style.json", selected)
    shot = insert(root, pictures["insert"], 0, 60, fit=fit)
    if letters:
        shot["letters"] = {"text": "alpha"}
    lane(root, [shot], name="broll")
    timeline = resolve_timeline(root)
    findings = [finding for finding in timeline.findings if finding.rule == "letters_split"]
    assert bool(findings) is refused
    if refused:
        finding, = findings
        assert not timeline.valid and finding.severity == "violation"
        assert finding.elements == ("broll-one",) and finding.lanes == ("broll",)
        assert finding.bare_message == (
            "Video in letters fills the frame, so it cannot be a split")
        assert finding.suggested_action == "use fit cover for this shot"
    else:
        assert timeline.valid


def test_letters_split_picture_guard_refuses_before_decoding(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(picture, "_source", lambda *args, **kwargs: pytest.fail(
        "Letters split must be refused before decoding picture"))
    timeline = ResolvedTimeline((element(0, 60, letters={"text": "alpha"}),), (), (), 90, 0)
    with pytest.raises(picture.PictureError, match=(
        "broll-one: Video in letters fills the frame, so it cannot be a split; "
        "use fit cover for this shot"
    )):
        picture.render_picture(tmp_path / "absent", tmp_path / "out.mp4",
                               work_dir=tmp_path / "work", timeline=timeline)


@pytest.mark.parametrize("elements", [
    (), (replace(element(), payload={"media": {"fit": "cover"}}),),
    (replace(element(), payload={}),),
])
def test_split_guard_without_split_never_opens_project(tmp_path, elements) -> None:
    split.check_split_voiceover(tmp_path / "absent", elements)


def test_split_guard_without_voiceover_avoids_context(tmp_path, monkeypatch) -> None:
    from yap_studio.project import status

    write_json(tmp_path / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample",
        "style": "preset:clean_minimal", "takes": [{"id": "t1", "path": "take.mov"}],
    })
    monkeypatch.setattr(status, "ProjectContext", lambda *args: pytest.fail(
        "A talking-head split must not construct a project context"))
    split.check_split_voiceover(tmp_path, (element(),))


def test_top_edge_head_refuses() -> None:
    face = Box(400, 180, 200, 300)
    assert head_box(face).y == 0
    with pytest.raises(picture.PictureError, match="use fit cover"):
        split.calculate_layout("broll-top-edge", [face])


def test_impossible_head_refuses_with_element_and_fix() -> None:
    with pytest.raises(picture.PictureError, match=(
        "broll-impossible.*use fit cover for this shot, or start it where she sits lower"
    )):
        split.calculate_layout("broll-impossible", [Box(400, 0, 200, 300)])


def test_no_face_fallback() -> None:
    layout = split.calculate_layout("broll-empty", [])
    assert layout.seam_y == 960
    assert layout.crop == (0, 0, 1080, 960) and layout.scale == 1
    assert "broll-empty" in layout.warnings[0]


def test_loader_uses_largest_face_median_and_minimum_head(monkeypatch, tmp_path: Path) -> None:
    from yap_studio.faces import project as face_project

    boxes = [Box(400, 650, 200, 280), Box(420, 630, 220, 300), Box(410, 640, 210, 290)]
    faces = measured(tuple((Box(0, 0, 10, 10), box) for box in boxes))
    calls = []
    monkeypatch.setattr(face_project, "cut_faces", lambda *args: calls.append(args) or faces)
    timeline = ResolvedTimeline((element(0, 3), replace(element(0, 3), id="broll-two")),
                                (), (), 3, 0)
    layouts = split.split_layouts(tmp_path, timeline)
    assert len(calls) == 1 and len(layouts) == 2
    assert layouts["broll-one"] == split.calculate_layout("broll-one", boxes)
    assert split.split_layouts(tmp_path, replace(timeline, elements=())) == {}
    assert len(calls) == 1


def test_loader_falls_back_when_vision_is_unavailable(monkeypatch, tmp_path):
    from yap_studio.faces import project as face_project

    faces = replace(measured(((),) * 3), status="unmeasured", reason="unavailable",
                    frames=(), gaps=())
    monkeypatch.setattr(face_project, "cut_faces", lambda *args: faces)
    layout = split.split_layouts(tmp_path, ResolvedTimeline((element(0, 3),), (), (), 3, 0))[
        "broll-one"]
    assert layout.crop == (0, 0, 1080, 960) and layout.warnings


@pytest.mark.parametrize("payload", [{}, {"media": {}}])
def test_missing_media_or_fit_is_a_full_screen_insert(payload, monkeypatch, tmp_path):
    from yap_studio.faces import project as face_project

    monkeypatch.setattr(face_project, "cut_faces",
                        lambda *args: pytest.fail("Full-screen inserts must not load split faces"))
    timeline = ResolvedTimeline((replace(element(0, 3), payload=payload),), (), (), 6, 0)
    assert split.split_layouts(tmp_path, timeline) == {}
    cut = measured(((Box(440, 650, 200, 280),),) * 6)
    mapped, sources = map_picture_faces(cut, timeline, zooms=(), video=cut.video)
    assert mapped.frames[:3] == ((),) * 3
    assert mapped.frames[3:] == cut.frames[3:]
    assert sources[0] == {"start_frame": 0, "end_frame": 3, "source": "insert"}


@pytest.mark.parametrize("name", ["cover", "contain", "crop", "push"])
def test_default_insert_graphs_match_captured_filters(name: str) -> None:
    captured = json.loads((Path(__file__).parent / "fixtures/broll_insert_graphs.json").read_text())
    info = SimpleNamespace(video_index=0, color_range=None, color_space=None, pix_fmt=None)
    source = picture._Source(Path("photo.png"), info, None, image=True)
    media = {"fit": "cover" if name == "push" else name}
    if name == "crop":
        media["crop"] = [8, 6, 300, 160]
    payload = {"media": media}
    if name == "push":
        payload.update(motion="push", push_scale=1.08, push_direction="in")
    graph = picture._Graph()
    picture._insert(graph, source, element(**payload))
    assert graph.filters == captured[name]


@pytest.mark.parametrize("push", [False, True])
def test_split_render_pixels_window_and_repeated_bytes(tmp_path, pictures, monkeypatch, push):
    from yap_studio.faces import project as face_project

    pictures = {**pictures, "cut": synth.picture_source(
        tmp_path / "pattern.mov", "color=c=blue:s=1080x1920:r=30,"
        "drawbox=x=400:y=600:w=300:h=300:c=yellow:t=fill", 90,
    )}
    root = project(tmp_path / "project", pictures, hold_frames=0)
    source = pictures["insert"]
    if push:
        source = tmp_path / "photo.png"
        from test_assemble_picture import overlay_image
        overlay_image(source)
    shot = insert(root, source, 10, 35, fit="split")
    if push:
        shot.update(motion="push", push_scale=1.08)
    lane(root, [shot], name="broll")
    faces = measured(((Box(440, 650, 200, 280),),) * 90)
    monkeypatch.setattr(face_project, "cut_faces", lambda *args: faces)
    monkeypatch.setattr(picture, "PRESET", "ultrafast")
    layers(root, pictures, monkeypatch, titles=[])
    timeline = resolve_timeline(root).require_valid()
    outputs = []
    for number in range(2):
        out = root / f"work-{number}/picture.mp4"
        result = picture.render_picture(root, out, work_dir=out.parent, timeline=timeline)
        outputs.append(out)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert_colour(synth.picture_pixels(outputs[0], 9, box=(500, 500, 20, 20)), (0, 0, 255))
    assert_colour(synth.picture_pixels(outputs[0], 35, box=(500, 500, 20, 20)), (0, 0, 255))
    for frame in (10, 22, 34):
        assert_colour(synth.picture_pixels(outputs[0], frame, box=(500, 500, 20, 20)), (255, 0, 0))
        assert_colour(synth.picture_pixels(outputs[0], frame, box=(900, 1300, 20, 20)), (0, 0, 255))
    assert synth.picture_pixels(outputs[0], 22, box=(400, 959, 200, 2)).mean() > 120
    marker = split.calculate_layout("broll-one", [Box(440, 650, 200, 280)]).map_box(
        Box(500, 700, 20, 20),
    )
    marker_pixels = synth.picture_pixels(outputs[0], 22,
                                         box=(round(marker.x), round(marker.y), 10, 10))
    assert_colour(marker_pixels, (255, 255, 0))
    assert result.to_dict()["splits"][0]["id"] == "broll-one"
    assert "splits" not in picture.PictureResult(1, 0, (), (), ()).to_dict()


def test_split_film_grade_runs_once_after_compositing(tmp_path, pictures, monkeypatch) -> None:
    from yap_studio.faces import project as face_project

    pictures = {**pictures, "cut": synth.picture_source(
        tmp_path / "swatch.mov", "color=c=0x20506e:s=1080x1920:r=30", 90,
    )}
    source = synth.picture_source(tmp_path / "insert.mov", "color=c=0x20506e:s=320x180:r=30", 90)
    root = project(tmp_path / "project", pictures, hold_frames=0)
    lane(root, [insert(root, source, 10, 35, fit="split")], name="broll")
    faces = measured(((Box(440, 650, 200, 280),),) * 90)
    monkeypatch.setattr(face_project, "cut_faces", lambda *args: faces)
    monkeypatch.setattr(picture, "PRESET", "ultrafast")
    layers(root, pictures, monkeypatch, titles=[])
    timeline = resolve_timeline(root).require_valid()
    spec = GradeSpec("film", 1)
    outputs = []
    for name, grade in (("plain", None), ("film", spec)):
        out = root / name / "picture.mp4"
        result = picture.render_picture(root, out, work_dir=out.parent, timeline=timeline,
                                        grade=grade, frames=(5, 35))
        outputs.append(out)
    command = picture_command(result)
    script = command[command.index("-filter_complex") + 1]
    assert script.count("lut3d=") == 1
    filters = script.split(";")
    stacked = next(row for row in filters if "vstack=inputs=2" in row)
    split_pad = stacked.rsplit("[", 1)[1].removesuffix("]")
    composited = next(row for row in filters if f"][{split_pad}]overlay=" in row)
    base_pad = composited.rsplit("[", 1)[1].removesuffix("]")
    graded = next(row for row in filters if "lut3d=" in row)
    assert graded.startswith(f"[{base_pad}]")
    assert filters.index(stacked) < filters.index(composited) < filters.index(graded)
    assert result.grade["look"] == "film" and result.splits[0]["id"] == "broll-one"
    for frame, y in ((0, 500), (17, 500), (17, 1300), (30, 500)):
        raw, actual = [synth.picture_pixels(out, frame, box=(450, y, 80, 80)) for out in outputs]
        expected = apply_rgb(raw / 255, spec)
        assert np.abs(actual.astype(float) - expected * 255).mean() < 3
        assert np.abs(actual.astype(float) - raw).mean() > 3
        assert np.abs(actual.astype(float) - apply_rgb(expected, spec) * 255).mean() > 3


@pytest.mark.parametrize("placement", ["fixed", "avoid_face"])
def test_split_caption_band_and_mapped_faces(placement: str) -> None:
    face = Box(440, 650, 200, 280)
    cut = measured(((face,),) * 60)
    layout = split.calculate_layout("broll-one", [face])
    mapped, sources = map_cut_faces(cut, total_frames=60, end_hold_frames=0, zooms=(),
                                   inserts=(), splits=[(0, 30, layout)], video=cut.video)
    assert mapped.frames[0][0].box == layout.map_box(face)
    assert sources[0]["source"] == "split" and mapped.frames[30] == cut.frames[30]
    style = _style(placement=placement, top_px=900, entry_ms=0)
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    slots = [{"start": 0, "end": 0.9, "words": [{"text": "Hello"}]},
             {"start": 1.1, "end": 1.9, "words": [{"text": "Again"}]}]
    ordinary = caption_offsets(slots, geometry, mapped, style)
    actual = caption_offsets(slots, geometry, mapped, style, splits=[(0, 30, layout.seam_y)])
    top = geometry.y + actual[0]["offset_px"]
    assert layout.seam_y + 24 <= top <= 1740 - geometry.height
    if placement == "avoid_face":
        assert gap_px(Box(70, top, 940, geometry.height), head_box(layout.map_box(face))) >= 24
    assert actual[1] == ordinary[1]


def test_final_face_mapping_preserves_split_faces() -> None:
    face = Box(440, 650, 200, 280)
    cut = measured(((face,),) * 60)
    layout = split.calculate_layout("broll-one", [face])
    timeline = ResolvedTimeline((element(0, 30),), (), (), 60, 0)
    mapped, sources = map_picture_faces(cut, timeline, zooms=(), video=cut.video,
                                        splits=[(0, 30, layout)])
    assert sources[0]["source"] == "split"
    assert mapped.frames[0][0].box == Box.from_dict(layout.map_box(face).to_dict())


def test_caption_receipt_changes_with_split_layout_only_when_split_exists(
    tmp_path, pictures, monkeypatch,
):
    from yap_studio.faces import project as face_project

    root = project(tmp_path / "project", pictures)
    shot = insert(root, pictures["insert"], 0, 30)
    lane(root, [shot], name="broll")
    original = caption_snapshot(ProjectContext(root)).inputs
    assert original["picture:faces"] == projection_sha256([{
        "id": "broll-one", "kind": "insert", "start_frame": 0, "end_frame": 30,
    }])
    shot["media"]["fit"] = "split"
    lane(root, [shot], name="broll")
    monkeypatch.setattr(face_project, "cut_faces", lambda *args: measured(
        ((Box(440, 650, 200, 280),),) * 90,
    ))
    before = caption_snapshot(ProjectContext(root)).inputs
    monkeypatch.setattr(face_project, "cut_faces", lambda *args: measured(
        ((Box(540, 750, 200, 280),),) * 90,
    ))
    after = caption_snapshot(ProjectContext(root)).inputs
    assert original["picture:faces"] != before["picture:faces"] != after["picture:faces"]
    assert {key for key in before if before[key] != after[key]} == {"picture:faces"}


@pytest.mark.parametrize("media_type", ["clip", "screen_recording", "photo", "screenshot"])
def test_split_style_selection_and_planner(tmp_path, media_type):
    style = resolve()
    style.data["broll"][f"{media_type}_placement"] = "split"
    validate_style_data(style.data)
    assert resolve().data["broll"] == BROLL_DEFAULTS
    item = catalog_item(tmp_path)
    item["media_type"] = media_type
    candidate = suggest_broll(phrase_words(), [item], style.data)["phrases"][0]["candidates"][0]
    assert candidate["kind"] == "insert" and candidate["fit"] == "split"
    plan = plan_broll(requests(placement="split"), phrase_words(), [item], style.data,
                      project_dir=tmp_path, frames=90)
    assert plan.broll[0]["element"]["media"]["fit"] == "split"
    request = parse_request({"schema": REQUEST_SCHEMA, "words": "dog", "length_s": 2,
                             "placement": "split"})
    selected = rank_candidates([_item()], request)[0]
    assert selected["fit"] == "split" and selected["kind"] == "video"


@pytest.mark.parametrize("extra", [{"crop": [0, 0, 30, 30]}, {"fit": "cover"}])
def test_planner_refuses_split_crop_or_other_fit(tmp_path, extra):
    item = catalog_item(tmp_path)
    with pytest.raises(ValidationError, match="split"):
        plan_broll(requests(placement="split", **extra), phrase_words(), [item], {},
                   project_dir=tmp_path, frames=90)


def test_schema_accepts_split_and_refuses_crop() -> None:
    doc = {"schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [{
        "id": "broll-one", "kind": "insert", "start": {"time": 0}, "hold_s": 2,
        "media": {"path": "shot.png", "sha256": "a" * 64, "fit": "split"},
    }]}
    assert parse_document("timeline", doc, lane="broll").elements[0].fields["media"]["fit"]
    invalid = copy.deepcopy(doc)
    invalid["elements"][0]["media"]["crop"] = [0, 0, 30, 30]
    with pytest.raises(ProjectFileError):
        parse_document("timeline", invalid, lane="broll")


def test_split_blur_bounds_use_top_canvas() -> None:
    info = SimpleNamespace(video_index=0, color_range=None, color_space=None, pix_fmt=None)
    source = picture._Source(Path("photo.png"), info, None, image=True)
    item = element(media={"fit": "cover", "blur": [[20, 950, 80, 80]]})
    with pytest.raises(picture.PictureError, match="1080x960"):
        picture._insert(picture._Graph(), source, item, size=(1080, 960))
