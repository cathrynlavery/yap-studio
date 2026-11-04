"""Shared camera moves and face framing on the horizontal canvas."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_cut import make_project, time_segment, write_edit, write_json
from test_landscape import _context
from test_smooth_joins import Blend, faces, pieces, timeline
from test_step_in import pieces as step_pieces

from yap_studio import ffmpeg, receipt
from yap_studio.canvas import require_supported
from yap_studio.faces import detect
from yap_studio.faces.mapping import ZoomPiece, map_cut_faces, picture_zooms
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.joins import helper, plan, render
from yap_studio.joins.steps import plan_steps
from yap_studio.project.model import Anchor, Element, TimelineLane
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import ProjectContext, ingest_inputs
from yap_studio.stages.assemble import assemble, verification_edl
from yap_studio.stages.captions import captions
from yap_studio.stages.cut import cut
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.stages.timeline import (
    ResolvedElement,
    ResolvedTimeline,
    add_timeline,
    resolve_timeline,
)
from yap_studio.verify import verify
from yap_studio.zoom_curve import ZoomCurve, ZoomKey, face_center, face_leaves_frame, punch_keys


@pytest.mark.parametrize("moves", [
    {"zoom_mode": "punches"},
    {"zoom_mode": "punches_and_push", "punch_scale": 2},
    {"opening_push": True},
    {"jump_cuts": "punch"},
    {"jump_cuts": "step_in"},
    {"jump_cuts": "smooth"},
])
def test_horizontal_accepts_zoom_moves(tmp_path: Path, moves: dict) -> None:
    context = _context(tmp_path, style={"moves": moves})
    item = Element("agent-punch", "zoom", Anchor(word="t1:0"), end=Anchor(word="t1:1"),
                   fields={"move": "punch"})
    require_supported(context, {"agent": TimelineLane((item,), lane="agent")})


@pytest.mark.parametrize("box,scale", [
    (Box(1700, 260, 180, 220), 1.5),
    (Box(40, 260, 180, 220), 2),
    (Box(850, 550, 180, 140), 2),
])
def test_wide_face_center_keeps_edges_and_chin_clear(box: Box, scale: float) -> None:
    boxes = {frame: (box,) for frame in range(30)}
    center, fits = face_center(boxes, range(30), scale, width=1920, height=1080,
                               bottom=721, bottom_margin=28)
    assert fits
    cx, cy = center[0] * 1920, center[1] * 1080
    assert cx + (box.x - cx) * scale >= 24
    assert cx + (box.x1 - cx) * scale <= 1896
    assert cy + (box.y - cy) * scale >= 24
    assert cy + (box.y1 - cy) * scale <= 721 - 28
    curve = ZoomCurve(0, 30, punch_keys(0, 30, scale, center))
    assert face_leaves_frame(curve, boxes, width=1920, height=1080, bottom=721) is None


def test_portrait_center_can_push_a_wide_face_out_of_frame() -> None:
    boxes = {frame: (Box(40, 260, 180, 220),) for frame in range(30)}
    center, _ = face_center(boxes, range(30), 2)
    curve = ZoomCurve(0, 30, punch_keys(0, 30, 2, center))
    assert face_leaves_frame(curve, boxes, width=1920, height=1080, bottom=721)[1] == "frame"


def test_wide_face_too_big_to_fit() -> None:
    assert face_center({0: (Box(200, 100, 1100, 500),)}, [0], 2,
                       width=1920, height=1080, bottom=721, bottom_margin=28)[1] is False


def test_vertical_face_center_defaults_stay_the_same() -> None:
    boxes = {0: (Box(400, 400, 200, 250),)}
    assert face_center(boxes, [0], 1.5) == face_center(
        boxes, [0], 1.5, width=1080, height=1920)


@pytest.mark.parametrize("card", [None, Box(1500, 260, 180, 220), Box(40, 260, 180, 220)])
def test_wide_step_in_checks_face_and_card(card: Box | None) -> None:
    box = Box(1700, 260, 180, 220)
    boxes = {frame: (box,) for frame in range(120)}
    cards = [] if card is None else [(30, 120, "card", card)]
    result = plan_steps(step_pieces((30, 30, 30, 30)), (), 120, (), boxes, cards,
                        width=1920, height=1080, bottom=721, bottom_margin=28, card_margin=24)
    if card is not None and card.x == 1500:
        assert all(shot.level == 0 for shot in result.shots)
        assert result.shots[1].reason == "A card needs the room around your head (card)."
    else:
        assert [shot.level for shot in result.shots] == [0, 1, 2, 3]
        for shot in result.shots[1:]:
            cx, cy = shot.center[0] * 1920, shot.center[1] * 1080
            assert cx + (box.x - cx) * shot.scale >= 24
            assert cx + (box.x1 - cx) * shot.scale <= 1896
            assert cy + (box.y1 - cy) * shot.scale <= 721 - 28


def test_wide_face_mapping_uses_output_size_without_a_base() -> None:
    video = VideoIdentity("cut.mp4", "a" * 64, 30, Fraction(30), 1920, 1080)
    cut_faces = FacesData("measured", None, video, {}, {},
                          ((FaceBox(Box(1700, 260, 180, 220), 1, "detected"),),) * 30, ())
    zoom = ZoomPiece(0, 30, ZoomCurve(0, 30, (ZoomKey(0, 1.213, (.9, .3)),)))
    mapped, _ = map_cut_faces(cut_faces, total_frames=30, end_hold_frames=0,
                              zooms=[zoom], inserts=(), video=video)
    rx, ry = 2328 / 1920, 1310 / 1080
    assert mapped.frames[0][0].box == Box(1728 + (1700 - 1728) * rx - 2,
                                         324 + (260 - 324) * ry - 2,
                                         180 * rx + 4, 220 * ry + 4)


@pytest.mark.parametrize("take", [False, True])
def test_wide_picture_zooms_use_cut_and_take_cover_bases(take: bool) -> None:
    cut_source = SimpleNamespace(info=SimpleNamespace(width=1920, height=1080))
    source = SimpleNamespace(info=SimpleNamespace(width=3840, height=2160)) if take else cut_source
    item = ResolvedElement("agent-punch", "zoom", "agent", 0, 30, "front",
                           {"peak_scale": 1.5, "center": [.9, .3]})
    piece = SimpleNamespace(out_f0=0, n_frames=30)
    zoom, = picture_zooms([(item, piece, source, .5)], cut_source)
    assert (zoom.base_w, zoom.base_h) == (1920, 1080)


@pytest.mark.parametrize("shape,size", [("horizontal", [1920, 1080]),
                                       ("vertical", [1080, 1920]), (None, [1080, 1920])])
def test_zoom_verification_region_uses_the_canvas(tmp_path: Path, shape, size) -> None:
    item = ResolvedElement("agent-punch", "zoom", "agent", 15, 60, "front", {})
    snapshot = SimpleNamespace(
        small_files={} if shape is None else {
            "project.json": json.dumps({"shape": shape}).encode()},
        caption_layer=None, title_layers=[], grade=None, media={}, sources={},
        spans={"ranges": []}, timeline=ResolvedTimeline((item,), (), (), 90, 0))
    assert verification_edl(snapshot, base=tmp_path)["overlays"] == [
        {"start_in_output": .5, "duration": 1.5, "region": [0, 0, *size]}]


def test_wide_smooth_window_geometry_and_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(helper, "unavailable_reason", lambda: None)
    source = synth.picture_source(tmp_path / "cut.mov", "testsrc2=s=1920x1080:r=30", 36)
    raw = dict(plan.decode_frames(source, [14, 21], width=1920, height=1080))
    decisions = plan.build_plan(tmp_path, timeline(), pieces(), [], {"jump_cuts": "smooth"},
                                faces=faces(width=1920, video_height=1080), frames=raw)
    assert decisions.joins[0].handling == "smooth"
    blend = Blend()
    first = render.render_windows(tmp_path, decisions, cut=source, interpolator=blend)
    movie = Path(first[0]["path"])
    info = ffmpeg.probe(movie)
    assert (info.video_frames, info.width, info.height, info.fps) == (6, 1920, 1080, 30)
    assert (info.color_primaries, info.color_transfer, info.color_space) == ("bt709",) * 3
    assert not info.has_audio
    assert blend.calls == [(raw[14], raw[21], 1920, 1080, [k / 7 for k in range(1, 7)])]
    digest = receipt.sha256_file(movie)
    assert render.render_windows(tmp_path, decisions, cut=source, interpolator=blend) == first
    assert len(blend.calls) == 1 and receipt.sha256_file(movie) == digest


@pytest.mark.render
@pytest.mark.parametrize("size", [(1920, 1080), (2880, 1620)])
def test_real_horizontal_punch_zooms_on_the_face(tmp_path: Path, monkeypatch, size) -> None:
    class FakeDetector:
        def detect(self, video: Path, *, every: int) -> detect.DetectorRun:
            info = ffmpeg.probe(video)
            rx, ry = info.width / 1920, info.height / 1080
            face = FaceBox(Box(1700 * rx, 260 * ry, 180 * rx, 220 * ry), 1, "detected")
            samples = {frame: (face,) for frame in detect.sample_frames(info.video_frames, every)}
            return detect.DetectorRun(info.width, info.height, info.video_frames,
                                      samples, {"name": "fake"})

    monkeypatch.setattr(detect, "unavailable_reason", lambda: None)
    monkeypatch.setattr(detect, "default_detector", lambda: FakeDetector())
    w, h = size
    ratio = w / 1920
    source = synth.loudness_master(
        tmp_path / "source.mov", frames=90,
        picture_filter=f"scale={w}:{h},hue=s=0,drawbox=x={1700 * ratio}:y={260 * ratio}:"
                       f"w={180 * ratio}:h={220 * ratio}:color=red:t=fill")
    root = make_project(tmp_path / "project", [source],
                        word_rows=[[("Readable", .2, .6), ("horizontal", .7, 1.2),
                                    ("captions", 1.3, 1.8), ("work.", 1.9, 2.4)]])
    data = json.loads((root / "project.json").read_bytes())
    data.update(shape="horizontal", asr={"engine": "none"}, music={"mode": "none"},
                audio={"mode": "preserve"}, platform="youtube", style="style.json")
    write_json(root / "style.json", {"schema_version": 1, "name": "Render fixture",
                                    "moves": {"zoom_mode": "punches"},
                                    "music": {"sound_effects": "none"},
                                    "captions": {"size_px": 64}})
    write_json(root / "project.json", data)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish()
    write_edit(root, [time_segment("s1", "t1", 0, 3)], end_hold_s=.3)
    cut(root)
    fragment = root / "punch.json"
    write_json(fragment, {"schema": "yap-studio/timeline@1", "elements": [
        {"id": "agent-punch", "kind": "zoom", "lane": "agent", "start": {"word": "t1:1"},
         "end": {"word": "t1:3"}, "move": "punch"}]})
    add_timeline(root, fragment, lane="agent").require_valid()
    captions(root)
    graphics(root)
    music(root)
    result = assemble(root)
    assert result["status"] == "verified", result
    final = root / "final/story.mp4"
    info = ffmpeg.probe(final)
    assert (info.width, info.height, info.video_frames, info.fps) == (1920, 1080, 99, 30)
    report = verify(str(final), edl=str(root / "final/verify-edl.json"), asr="none",
                    detail="full", platform="youtube", loudness="note")
    assert report["status"] == "pass", report["findings"]
    zoom = next(item for item in resolve_timeline(root).elements if item.id == "agent-punch")
    assert zoom.payload["center_from"] == "face"
    markers = []
    for frame in (zoom.start_frame - 1, zoom.start_frame + 10):
        pixels = synth.picture_pixels(final, frame)
        ys, xs = np.where((pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 80)
                          & (pixels[:, :, 2] < 80))
        assert len(xs)
        markers.append(Box(int(xs.min()), int(ys.min()),
                           int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    before, hold = markers
    print(f"Horizontal {w}x{h} marker: before={before}, hold={hold}")
    assert hold.w == pytest.approx(before.w * 1.5, abs=3)
    assert hold.h == pytest.approx(before.h * 1.5, abs=3)
    assert hold.x >= 24 and hold.x1 <= 1896 + 2
    assert hold.y1 <= 721 - 28 + 2
    assert hold.x + hold.w / 2 < before.x + before.w / 2
