"""Synthetic clone measurements and moving, clean-edged copies on the cut clock."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import synth
from test_matte_edge_colour import write_planes
from test_moment_picture import decoded
from test_moment_timeline import enable, moment
from test_timeline import add
from test_timeline import project as project

from yap_studio.captions import caption_layout_settings
from yap_studio.edl import EditList, Segment
from yap_studio.faces import project as face_project
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.matte.model import Cleanup, MatteData
from yap_studio.plan import Plan
from yap_studio.render import moment_clones as clones
from yap_studio.stages import assemble_picture as stage
from yap_studio.stages.captions import caption_band_geometry
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline, TimelineRefusal
from yap_studio.style import resolve_style_data


def scene(*, x: int = 230, width: int = 80, shift: int = 0) -> tuple:
    rng = np.random.default_rng(3)
    room = np.empty((240, 540, 3), np.uint8)
    y, xx = np.mgrid[:240, :540]
    room[..., 0] = np.rint(100 + 20 * np.sin(xx / 10) + 20 * np.cos(y / 9)
                           + rng.integers(-2, 3, (240, 540)))
    room[..., 1:] = 128
    alpha = np.zeros(room.shape[:2])
    alpha[55:220, x:x + width] = 1
    alpha[55:220, x - 1] = 0.5
    live = np.rint(room * (1 - alpha[..., None]) + (150, 90, 190) * alpha[..., None])
    live = live.astype(np.uint8)
    frames = dict.fromkeys(range(240), live)
    masks = dict.fromkeys(range(240), alpha)
    if shift:
        for i in range(60, 240):
            frames[i] = np.roll(frames[i], shift, axis=1)
            masks[i] = np.roll(masks[i], shift, axis=1)
    face = FaceBox(Box(x * 2, 120, width * 2, 90), 1, "detected")
    faces = FacesData("measured", None, VideoIdentity("cut.mp4", "a" * 64, 240,
        Fraction(30), 1080, 480), {}, {}, ((face,),) * 240, ())
    plan = Plan.build(EditList("yap", Path("spans.json"), {},
        [Segment(1, "t1", Fraction(0), Fraction(8))], None, [], []), Fraction(30), 48000)
    item = ResolvedElement("agent-clone", "moment", "agent", 0, 36, "front",
                           {"style": "clone_flash"})
    return frames, masks, faces, plan.segments, item


def measure(data: tuple) -> dict:
    pictures, mattes, faces, pieces, item = data
    return clones.measure_clones(pictures, mattes, faces, pieces, item,
                                  size=(1080, 480), caption_band=(460, 480), caption_gap_px=100)


@pytest.mark.parametrize("duration,valid", [(0.9, False), (1.0, True),
                                         (2.0, True), (2.1, False)])
def test_clone_duration(project: Path, duration: float, valid: bool) -> None:
    enable(project)
    item = moment(style="clone_flash", hold_s=duration)
    item.pop("end")
    result = add(project, item, lane="agent")
    assert result.valid is valid
    if not valid:
        finding = next(f for f in result.findings if f.rule == "moment_length")
        assert "1 to 2 s" in finding.message and finding.suggested_action


def test_clone_style_stays_off(project: Path) -> None:
    result = add(project, moment(style="clone_flash"), lane="agent")
    assert any(f.rule == "moment_off" for f in result.findings)


def test_still_room_has_two_copies_and_source_separation() -> None:
    record = measure(scene())
    assert len(record["copies"]) == 2
    assert {c["side"] for c in record["copies"]} == {"left", "right"}
    assert record["copies"][1]["source_start_frame"] >= (
        record["copies"][0]["source_start_frame"] + 30)
    assert [c["appear_frame"] for c in record["copies"]] == [0, 9]
    assert [c["frames"] for c in record["copies"]] == [36, 27]
    assert all("source_frames" not in c and "take_source_frames" not in c
               for c in record["copies"])
    assert record["camera"]["shift_px"] == 0


@pytest.mark.parametrize("shift,valid", [(3, False), (1, True)])
def test_camera_six_pixels_refused_two_pixels_pass(shift: int, valid: bool) -> None:
    data = scene(shift=shift)
    if valid:
        assert measure(data)["copies"]
    else:
        with pytest.raises(TimelineRefusal) as caught:
            measure(data)
        assert caught.value.code == "clone_camera" and caught.value.suggested_action


def test_camera_movement_inside_live_window_is_refused() -> None:
    # Later source frames match the window's middle, so endpoint checks miss the bump.
    data = scene(shift=3)
    pictures, mattes, _, _, _ = data
    for frame in range(12, 25):
        pictures[frame] = np.roll(pictures[frame], 3, axis=1)
        mattes[frame] = np.roll(mattes[frame], 3, axis=1)
    np.testing.assert_array_equal(pictures[0], pictures[35])
    with pytest.raises(TimelineRefusal) as caught:
        measure(data)
    assert caught.value.code == "clone_camera" and caught.value.suggested_action


@pytest.mark.parametrize("later_still", [True, False])
def test_camera_movement_inside_candidate_is_skipped(later_still: bool) -> None:
    data = list(scene())
    pictures, mattes, _, pieces, item = data
    candidates = clones.candidate_frames(pieces, item)
    first = candidates[0][0]
    length = item.end_frame - item.start_frame
    for frame in range(first + 12, first + 15):
        pictures[frame] = np.roll(pictures[frame], 3, axis=1)
        mattes[frame] = np.roll(mattes[frame], 3, axis=1)
    np.testing.assert_array_equal(pictures[first], pictures[first + length - 1])
    if later_still:
        assert measure(tuple(data))["copies"][0]["source_start_frame"] == candidates[1][0]
    else:
        data[3] = [replace(pieces[0], n_frames=first + length, src_f1=first + length)]
        with pytest.raises(TimelineRefusal) as caught:
            measure(tuple(data))
        assert caught.value.code == "clone_camera" and caught.value.suggested_action


def test_camera_records_worst_sample_by_shift_then_difference() -> None:
    data = scene()
    pictures, mattes, _, pieces, item = data
    first = clones.candidate_frames(pieces, item)[0][0]
    for start in (0, first):
        for offset, shift, difference in ((6, 1, 0), (12, 1, 3), (24, 0, 5)):
            frame = start + offset
            picture = np.roll(pictures[frame], shift, axis=1).copy()
            picture[..., 0] += difference
            pictures[frame] = picture
            mattes[frame] = np.roll(mattes[frame], shift, axis=1)
    record = measure(data)
    for start, camera in ((0, record["camera"]), (first, record["copies"][0]["camera"])):
        pairs = [clones.camera_pair(pictures[0], mattes[0], pictures[start + offset],
                                    mattes[start + offset]) for offset in (0, 6, 12, 24, 35)]
        assert camera == max(pairs, key=lambda pair: (pair["shift_px"], pair["difference"]))
        assert camera["shift_px"] == 2


@pytest.mark.parametrize("problem,code", [("faces", "clone_faces"),
    ("source", "clone_source"), ("room", "clone_room"), ("window", "clone_camera")])
def test_refusals_have_fixes(problem: str, code: str) -> None:
    data = list(scene(x=15, width=510) if problem == "room" else scene())
    if problem == "faces":
        data[2] = replace(data[2], status="unmeasured", reason="unavailable", frames=())
    elif problem == "source":
        data[3] = [replace(data[3][0], n_frames=36, src_f1=36)]
    elif problem == "window":
        data[0][35] = np.roll(data[0][35], 3, axis=1)
        data[1][35] = np.roll(data[1][35], 3, axis=1)
    with pytest.raises(TimelineRefusal) as caught:
        measure(tuple(data))
    assert caught.value.code == code and caught.value.suggested_action


def test_one_free_side_and_face_boxes_do_not_overlap() -> None:
    record = measure(scene(x=10))
    assert len(record["copies"]) == 1 and record["copies"][0]["side"] == "right"
    for copy in record["copies"]:
        placed, live = copy["placed_face"], record["face"]
        assert placed["x"] >= live["x"] + live["w"] + 24
        assert copy["scale"] >= 0.55


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("unit", [0.5, 1.0])
@pytest.mark.parametrize("clearance,valid", [(-1, False), (10, False), (21, True)])
def test_scaled_copy_checks_moving_head_and_caption_gap(
    side: str, unit: float, clearance: float, valid: bool,
) -> None:
    scale = 0.7
    # The moving face clears the band even when its hair or required gap does not.
    y = 220 + clearance + 0.6 * 110 * scale - (90 * (1 - scale) / 2 - 10 * scale)
    face = Box((160 if side == "left" else 760) * unit, y * unit, 160 * unit, 90 * unit)
    source = Box(460 * unit, 300 * unit, 160 * unit, 90 * unit)
    moving = Box(460 * unit, 290 * unit, 160 * unit, 110 * unit)
    face_top = face.y + (face.h - source.h * scale) / 2 - 10 * unit * scale
    assert face_top > 220 * unit
    result = clones._place(side, face, face, source, moving, np.zeros((240, 540)),
        (round(1080 * unit), round(480 * unit)), (180 * unit, 220 * unit),
        caption_gap_px=20)
    if valid:
        assert result is not None
        assert result["side"] == side and result["scale"] == pytest.approx(scale)
    else:
        assert result is None


def test_source_candidates_respect_takes_joins_order_and_search_bound() -> None:
    _, _, _, pieces, item = scene()
    piece = pieces[0]
    later = replace(piece, source_id="t2", src_f0=0, out_f0=36, n_frames=90, src_f1=90)
    short = replace(piece, src_f0=66, out_f0=126, n_frames=20, src_f1=86)
    good = replace(piece, src_f0=120, out_f0=146, n_frames=600, src_f1=720)
    candidates = clones.candidate_frames([replace(piece, n_frames=36, src_f1=36),
                                          later, short, good], item)
    assert len(candidates) == clones.CLONE_MAX_CANDIDATES
    assert [take for _, _, take in candidates] == ["t1"] * len(candidates)
    assert [source for _, source, _ in candidates] == list(range(120, 300, 15))
    assert all(146 <= cut and cut + 36 <= 746 for cut, _, _ in candidates)


def test_window_faces_need_at_least_half_and_bad_camera_candidate_is_skipped() -> None:
    data = list(scene())
    per_frame = list(data[2].frames)
    for frame in range(18):
        per_frame[frame] = ()
    data[2] = replace(data[2], frames=tuple(per_frame), gaps=((0, 18),))
    assert measure(tuple(data))["copies"]
    per_frame[18] = ()
    data[2] = replace(data[2], frames=tuple(per_frame), gaps=((0, 19),))
    with pytest.raises(TimelineRefusal) as caught:
        measure(tuple(data))
    assert caught.value.code == "clone_faces"
    data = list(scene())
    first = clones.candidate_frames(data[3], data[4])[0][0]
    for frame in (first, first + 35):
        data[0][frame] = np.roll(data[0][frame], 3, axis=1)
        data[1][frame] = np.roll(data[1][frame], 3, axis=1)
    assert measure(tuple(data))["copies"][0]["source_start_frame"] > first


def test_scaled_copy_fades_before_its_picture_bottom() -> None:
    data = scene()
    record = measure(data)
    copy = record["copies"][0].copy()
    scale = 0.6
    face = copy["placed_face"]
    cx, cy = face["x"] + face["w"] / 2, face["y"] + face["h"] / 2
    source_face = copy["source_face"]
    copy.update(scale=scale, placed_face={"x": cx - source_face["w"] * scale / 2,
        "y": cy - source_face["h"] * scale / 2,
        "w": source_face["w"] * scale, "h": source_face["h"] * scale})
    record = {**record, "copies": [copy]}
    live, alpha = data[0][0], data[1][0]
    source = np.full_like(live, (150, 90, 190))
    out = clones.clone_frame(live, alpha, [(source, np.ones_like(alpha))], record,
                             index=0, frames=36)
    bottom = (cy + (480 - source_face["y"] - source_face["h"] / 2) * scale) / 2
    x, y = round(cx / 2), int(bottom) - 1
    assert np.max(abs(out[y, x].astype(int) - live[y, x])) < 10
    np.testing.assert_array_equal(out[y + 2, x], live[y + 2, x])
    np.testing.assert_array_equal(out[y - 35, x], (235, 128, 128))


def test_clone_flash_colour_stagger_motion_and_live_pixels() -> None:
    data = scene()
    pictures, mattes, _, _, _ = data
    record = measure(data)
    live, alpha = pictures[0], mattes[0]
    copy = record["copies"][0]
    x = round((copy["placed_face"]["x"] + copy["placed_face"]["w"] / 2) / 2)
    sources = [(pictures[c["source_start_frame"]], mattes[c["source_start_frame"]])
               for c in record["copies"]]
    for index in range(3):
        out = clones.clone_frame(live, alpha, sources, record, index=index, frames=36)
        np.testing.assert_array_equal(out[75, x], (235, 128, 128))
        np.testing.assert_array_equal(out[alpha == 1], live[alpha == 1])
    out = clones.clone_frame(live, alpha, sources, record, index=6, frames=36)
    np.testing.assert_array_equal(out[75, x], (150, 90, 190))
    second = record["copies"][1]
    x2 = round((second["placed_face"]["x"] + second["placed_face"]["w"] / 2) / 2)
    np.testing.assert_array_equal(out[75, x2], live[75, x2])
    flashed = clones.clone_frame(live, alpha, sources, record, index=9, frames=36)
    np.testing.assert_array_equal(flashed[75, x2], (235, 128, 128))
    moved = [(np.roll(p, 5, axis=1), np.roll(a, 5, axis=1)) for p, a in sources]
    assert np.any(out != clones.clone_frame(live, alpha, moved, record, index=6, frames=36))
    np.testing.assert_array_equal(clones.clone_frame(live, alpha, sources, record,
                                  index=35, frames=36), live)
    np.testing.assert_array_equal(out[alpha == 1], live[alpha == 1])


@pytest.fixture
def clone_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    root = tmp_path / "project"
    (root / "cut").mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({"schema": "yap-studio/project@1",
        "name": "clones", "style": "style.json", "takes": [{"id": "t1", "path": "source.mov"}]}))
    (root / "style.json").write_text(json.dumps({"schema_version": 1,
        "moments": {"styles": ["clone_flash"]}}))
    (root / "cut/spans.json").write_text(json.dumps({"sources": {"t1": "../source.mov"},
        "segments": [{"source": "t1", "start": 0, "end": 8}]}))
    yy, xx = np.mgrid[:192, :256]
    room = np.empty((192, 256, 3), np.uint8)
    room[..., 0] = 95 + 15 * np.sin(xx / 6) + 15 * np.cos(yy / 7)
    room[..., 1:] = 128
    pictures, masks, faces = [], [], []
    for frame in range(240):
        x = 114 + round(2 * np.sin(frame / 8))
        alpha = np.zeros((192, 256), np.uint8)
        alpha[30:180, x:x + 28] = 255
        alpha[30:180, x - 1] = 128
        person = np.array((130 + frame % 20, 100, 180))
        picture = np.rint(room * (1 - alpha[..., None] / 255)
                          + person * (alpha[..., None] / 255)).astype(np.uint8)
        pictures.append(picture)
        masks.append(alpha)
        faces.append((FaceBox(Box(x, 32, 28, 25), 1, "detected"),))
    source = write_planes(root / "source.mkv", pictures, "yuv444p")
    from yap_studio import ffmpeg

    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-c:v", "libx264", "-crf", "0",
        "-pix_fmt", "yuv444p", "-color_range", "tv", "-colorspace", "bt709",
        str(root / "cut/cut.mp4")], timeout_s=60)
    matte = write_planes(root / "matte.mkv", masks, "gray")
    measured = FacesData("measured", None, VideoIdentity("cut.mp4", "a" * 64, 240,
        Fraction(30), 256, 192), {}, {}, tuple(faces), ())
    monkeypatch.setattr(face_project, "cut_faces", lambda *args, **kwargs: measured)
    monkeypatch.setattr(stage, "WIDTH", 256)
    monkeypatch.setattr(stage, "HEIGHT", 192)
    monkeypatch.setattr(stage, "CRF", 0)
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: matte)
    synth.picture_source(root / "front.mov", "color=yellow:s=32x24:r=30", 192, alpha=True)
    synth.picture_source(root / "caption.mov", "color=lime:s=32x24:r=30", 240, alpha=True)
    item = ResolvedElement("agent-clone", "moment", "agent", 12, 48, "front",
                           {"style": "clone_flash"})
    front = ResolvedElement("agent-front", "graphic", "agent", 48, 240, "front", {})
    titles = [{"id": front.id, "kind": "graphic", "layer": "front", "path": "front.mov",
               "x": 76, "y": 80, "width": 32, "height": 24, "start_frame": 48, "frames": 192}]
    caption = {"path": "caption.mov", "x": 148, "y": 168, "width": 32, "height": 24,
               "start_frame": 0, "frames": 240}
    return root, ResolvedTimeline((item, front), (), (), 240, 0), titles, caption


@pytest.mark.parametrize("gray", [False, True])
def test_measure_frames_decodes_large_selection_in_bounded_passes(
    tmp_path: Path, gray: bool,
) -> None:
    def values(frame):
        return frame % 256 if gray else (16 + frame % 220, 64 + frame // 220 * 50, 128)

    source = write_planes(tmp_path / "indexed.mkv", (
        np.full((36, 64) if gray else (36, 64, 3), values(frame), np.uint8)
        for frame in range(500)), "gray" if gray else "yuv444p")
    selected = sorted({*range(0, 400, 2), *range(425, 475)})
    commands = []
    pictures = clones._measure_frames(source, selected, (32, 18), gray=gray, commands=commands)
    assert list(pictures) == selected
    for frame, picture in pictures.items():
        expected = np.full((18, 32) if gray else (18, 32, 3), values(frame), np.uint8)
        np.testing.assert_array_equal(picture, expected.astype(float) / 255 if gray else expected)
    assert len(commands) > 1
    decoded = []
    for command in commands:
        expression = command[command.index("-vf") + 1]
        terms = re.findall(r"(eq|between)\(n,(\d+)(?:,(\d+))?\)", expression)
        assert 0 < len(terms) <= clones.CLONE_SELECT_TERMS_MAX
        frames = [frame for _, first, last in terms
                  for frame in range(int(first), int(last or first) + 1)]
        assert expression.startswith(f"trim=end_frame={frames[-1] + 1},")
        decoded.extend(frames)
    assert decoded == selected


def write_clone_matte(root: Path, backend: dict) -> None:
    video = face_project.cut_faces(root, root / "cut/cut.mp4").video
    data = MatteData("measured", None, video, backend, Cleanup(),
        {"file": "matte.mkv", "sha256": "a" * 64, "codec": "ffv1", "pix_fmt": "gray"},
        tuple(i / 30 for i in range(video.frames)), video.frames / 30,
        (1.0,) * video.frames)
    (root / "matte.json").write_text(json.dumps(data.to_dict()))


@pytest.mark.parametrize("screen", ["green", "blue"])
def test_prepare_refuses_keyed_copies_before_decoding(
    clone_project: tuple, monkeypatch: pytest.MonkeyPatch, screen: str,
) -> None:
    root, timeline, _, _ = clone_project
    write_clone_matte(root, {"name": "chroma-key", "screen": screen,
                            "key": "#00ff00" if screen == "green" else "#0000ff",
                            "border_share": 1.0})
    calls = []

    def measured(*args, **kwargs):
        calls.append(True)
        pytest.fail("Keyed copies must refuse before decoding")

    monkeypatch.setattr(clones, "_measure_frames", measured)
    style = resolve_style_data(json.loads((root / "style.json").read_text()))
    with pytest.raises(TimelineRefusal) as caught:
        clones.prepare(root, root, root / "cut/cut.mp4", root / "matte.mkv",
                       timeline.elements[0], style, [])
    assert caught.value.code == "clone_keyed"
    assert str(caught.value) == (
        f"clone_flash cannot clean the {screen} screen edge on the copies, "
        f"so they would carry a {screen} fringe")
    assert caught.value.suggested_action == (
        "pick another Moment style, or remove the clone flash; clone flash needs a take "
        "filmed without a green or blue screen")
    assert not calls


@pytest.mark.parametrize("person_matte", [False, True])
def test_prepare_samples_candidate_interiors_in_bounded_half_size_passes(
    clone_project: tuple, monkeypatch: pytest.MonkeyPatch, person_matte: bool,
) -> None:
    from yap_studio.edl import load_edit_list

    root, timeline, _, _ = clone_project
    if person_matte:
        write_clone_matte(root, {"name": "person-detection"})
    item = timeline.elements[0]
    style = resolve_style_data(json.loads((root / "style.json").read_text()))
    plan = Plan.build(load_edit_list(root / "cut/spans.json"), Fraction(30), 48000)
    length = item.end_frame - item.start_frame
    candidates = clones.candidate_frames(plan.segments, item)
    expected = sorted({*range(item.start_frame, item.end_frame),
        *(frame for first, _, _ in candidates for frame in range(first, first + length, 6)),
        *(first + length - 1 for first, _, _ in candidates)})
    calls = []
    decode = clones._measure_frames

    def measured(source, selected, size, *, gray, commands):
        calls.append((source, selected, size, gray))
        return decode(source, selected, size, gray=gray, commands=commands)

    monkeypatch.setattr(clones, "_measure_frames", measured)
    commands = []
    effect = clones.prepare(root, root, root / "cut/cut.mp4", root / "matte.mkv", item, style,
                             commands)
    assert effect.record["copies"]
    assert calls == [(root / "cut/cut.mp4", expected, (128, 96), False),
                     (root / "matte.mkv", expected, (128, 96), True)]
    picture_filters, matte_filters = [], []
    for command in commands:
        expression = command[command.index("-vf") + 1]
        assert "scale=128:96" in expression
        selection = expression.split("select='", 1)[1].split("'", 1)[0]
        source = command[command.index("-i") + 1]
        (picture_filters if source.endswith("cut.mp4") else matte_filters).append(selection)
    assert picture_filters == matte_filters
    assert any(f"between(n,{item.start_frame},{item.end_frame - 1})" in selection
               for selection in picture_filters)
    assert all(len(selection.split("+")) <= clones.CLONE_SELECT_TERMS_MAX
               for selection in picture_filters)


@pytest.mark.parametrize("highlight", ["none", "giant"])
@pytest.mark.parametrize("position,placement,valid", [
    ("middle", "above", True), ("middle", "overlap", False), ("middle", "below", True),
    ("middle", "above_gap", False), ("middle", "below_hair", False),
    ("middle", "below_gap", False),
    ("bottom", "above", True), ("bottom", "overlap", False),
    ("bottom", "above_gap", False),
])
def test_prepare_keeps_copy_heads_and_gap_clear_of_caption_span(
    clone_project: tuple, monkeypatch: pytest.MonkeyPatch,
    position: str, placement: str, valid: bool, highlight: str,
) -> None:
    root, timeline, _, _ = clone_project
    style = resolve_style_data({"schema_version": 1, "captions": {
        "position": position, **({"top_px": 960} if position == "middle" else {}),
        "highlight_style": highlight, **({"size_px": 120} if highlight == "giant" else {}),
    }, "graphics": {"caption_gap_px": 40}})
    settings = caption_layout_settings(style["captions"], style["brand"])
    bands = [caption_band_geometry(settings,
        safe_bottom_px=style["internal"]["caption_safe_bottom_px"], backing=backing)
        for backing in ((), ("box",), ("shadow",))]
    faces = face_project.cut_faces(root, root / "cut/cut.mp4")
    factor = faces.video.height / 1920
    top = min(band.y for band in bands) * factor
    bottom = max(band.y + band.height for band in bands) * factor
    height = faces.frames[0][0].box.h
    gap = style["graphics"]["caption_gap_px"] * faces.video.width / 1080
    y = {"above": top - height - gap - 1, "above_gap": top - height - gap / 2,
         "overlap": top - height / 2, "below_hair": bottom + height * 0.3,
         "below_gap": bottom + height * 0.6 + gap / 2,
         "below": bottom + height * 0.6 + gap + 1}[placement]
    faces = replace(faces, frames=tuple(tuple(replace(face, box=replace(face.box, y=y))
        for face in frame) for frame in faces.frames))
    monkeypatch.setattr(face_project, "cut_faces", lambda *args, **kwargs: faces)

    def prepare():
        return clones.prepare(root, root, root / "cut/cut.mp4", root / "matte.mkv",
                               timeline.elements[0], style, [])

    if valid:
        record = prepare().record
        assert record["copies"]
        for copy in record["copies"]:
            assert copy["scale"] == 1
            face = copy["placed_face_union"]
            if placement == "above":
                assert face["y"] + face["h"] == pytest.approx(top - gap - 1, abs=0.01)
            else:
                assert face["y"] - face["h"] * 0.6 == pytest.approx(bottom + gap + 1, abs=0.01)
    else:
        with pytest.raises(TimelineRefusal) as caught:
            prepare()
        assert caught.value.code == "clone_room" and caught.value.suggested_action


@pytest.mark.render
def test_clone_assembly_repeat_pilot_layers_movement_and_receipt(clone_project: tuple) -> None:
    root, timeline, titles, caption = clone_project

    def render(name, elements=timeline.elements, frames=None):
        out = root / name / "picture.mp4"
        result = stage.render_picture(root, out, work_dir=out.parent,
            timeline=replace(timeline, elements=elements), title_layers=titles,
            caption_layer=caption, frames=frames)
        return out, result

    plain, _ = render("plain", timeline.elements[1:])
    full, result = render("full")
    repeat, _ = render("repeat")
    pilot, pilot_result = render("pilot", frames=(25, 40))
    assert full.read_bytes() == repeat.read_bytes()
    before, after = decoded(plain), decoded(full)
    np.testing.assert_array_equal(before[:12], after[:12])
    np.testing.assert_array_equal(before[48:], after[48:])
    np.testing.assert_array_equal(decoded(pilot), after[25:41])
    np.testing.assert_allclose(after[60, 84:100, 80:104], before[60, 84:100, 80:104], atol=3)
    np.testing.assert_allclose(after[25, 172:188, 152:176], before[25, 172:188, 152:176], atol=3)
    record = result.to_dict()["moments"][0]
    assert record == pilot_result.to_dict()["moments"][0]
    assert len(record["copies"]) == 2
    assert all(c["side"] and c["scale"] >= 0.55 and c["frames"] for c in record["copies"])
    assert [c["frames"] for c in record["copies"]] == [36, 27]
    assert all("source_frames" not in c and "take_source_frames" not in c
               for c in record["copies"])
    layer = decoded(root / "full/moment-0.mkv")
    copy = record["copies"][0]
    source = decoded(root / "cut/cut.mp4")
    x = round(copy["placed_face"]["x"] + copy["placed_face"]["w"] / 2)
    sx = round(copy["source_face"]["x"] + copy["source_face"]["w"] / 2)
    for index in (6, 10):
        np.testing.assert_array_equal(layer[index, 40, x],
            source[copy["source_start_frame"] + index, 40, sx])
    assert layer[6, 40, x, 0] != layer[10, 40, x, 0]


def test_overlay_refuses_before_any_layer_encoding(clone_project: tuple,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio.render import moments

    root, timeline, titles, caption = clone_project
    measured = face_project.cut_faces(root, root / "cut/cut.mp4")
    monkeypatch.setattr(face_project, "cut_faces", lambda *args, **kwargs: replace(
        measured, status="unmeasured", reason="unavailable", frames=()))
    monkeypatch.setattr(moments, "render_moments", lambda *args, **kwargs: pytest.fail(
        "layer encoding started before refusal"))
    out = root / "refused/picture.mp4"
    with pytest.raises(TimelineRefusal) as caught:
        stage.render_picture(root, out, work_dir=out.parent, timeline=timeline,
                             title_layers=titles, caption_layer=caption)
    assert caught.value.code == "clone_faces" and caught.value.suggested_action
    assert not out.exists() and not list(out.parent.glob("moment-*.mkv"))
