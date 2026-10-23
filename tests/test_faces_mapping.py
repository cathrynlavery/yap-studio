"""Cut face geometry follows picture zooms, replacements and holds."""

from dataclasses import replace
from fractions import Fraction
from math import ceil, pi, sin, trunc

import pytest

from yap_studio.faces.detect import find_gaps
from yap_studio.faces.mapping import ZoomPiece, map_cut_faces, zoom_ratios
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.zoom_curve import ZoomCurve


def _cut(frames: int = 12) -> FacesData:
    per_frame = tuple((FaceBox(Box(200 + n, 900, 120, 150), 0.8, "interpolated"),)
                      for n in range(frames))
    return FacesData("measured", None,
                     VideoIdentity("cut.mp4", "a" * 64, frames, Fraction(30), 1080, 1920),
                     {"name": "fake"}, {"every": 2}, per_frame, ())


def _mapped(cut: FacesData, *, zooms=(), inserts=(), transitions=(), hold=0, total=12):
    return map_cut_faces(cut, total_frames=total, end_hold_frames=hold, zooms=zooms,
                         inserts=inserts, transitions=transitions,
                         video=replace(cut.video, name="picture.mp4", frames=total + hold))


def test_camera_transform_uses_output_frame_and_skips_replacements() -> None:
    cut = _cut()
    camera = [(1.1, -20 + frame, -50) for frame in range(12)]
    mapped, sources = map_cut_faces(
        cut, total_frames=12, end_hold_frames=2, zooms=(), inserts=[(4, 6)],
        transitions=[(8, 10)], camera=camera,
        video=replace(cut.video, frames=14),
    )
    face = cut.frames[3][0]
    assert mapped.frames[3][0].box == Box(-17 + 1.1 * face.box.x,
                                         -50 + 1.1 * face.box.y,
                                         1.1 * face.box.w, 1.1 * face.box.h)
    assert mapped.frames[4:6] == ((), ()) and mapped.frames[8:10] == ((), ())
    assert mapped.frames[12:] == (mapped.frames[11],) * 2
    assert sources[0]["source"] == "camera3d"


@pytest.mark.parametrize("picture", [False, True])
@pytest.mark.parametrize("replacement", [None, "insert", "transition"])
def test_camera_hold_repeats_mapped_faces_and_keeps_pin_hits(picture, replacement) -> None:
    from types import SimpleNamespace

    from yap_studio.faces.mapping import map_picture_faces
    from yap_studio.feedback.pins import face_hits

    cut = _cut()
    camera = [(1.25, 350 + frame, -300) for frame in range(12)]
    video = replace(cut.video, name="picture.mp4", frames=15)
    elements = (() if replacement is None else (SimpleNamespace(
        kind=replacement, start_frame=11, end_frame=12, payload={}),))
    if picture:
        timeline = SimpleNamespace(total_frames=12, end_hold_frames=3, elements=elements)
        mapped, sources = map_picture_faces(cut, timeline, zooms=(), video=video, camera=camera)
    else:
        mapped, sources = map_cut_faces(
            cut, total_frames=12, end_hold_frames=3, zooms=(), video=video, camera=camera,
            inserts=[(11, 12)] if replacement == "insert" else (),
            transitions=[(11, 12)] if replacement == "transition" else ())
    mapped = FacesData.from_dict(mapped.to_dict())
    assert mapped.frames[12:] == (mapped.frames[11],) * 3
    assert sources[-1] == {"start_frame": 12, "end_frame": 15, "source": "hold"}
    box = Box(361 + 1.25 * 211, -300 + 1.25 * 900, 150, 187.5)
    for frame in range(12, 15):
        hits = face_hits(mapped, frame, round(box.x + box.w / 2), round(box.y + box.h / 2))
        assert [hit["kind"] for hit in hits] == ([] if replacement else ["speaker"])
        if not replacement:
            assert hits[0]["box"] == box.to_dict()


@pytest.mark.parametrize("size", [(1080, 1920), (1920, 1080), (3840, 2160), (1500, 3000)])
@pytest.mark.parametrize("frame", [7, 8, 13, 21, 34, 46])
def test_zoom_ratios_match_picture_scale_and_crop(size: tuple[int, int], frame: int) -> None:
    w, h = size
    factor = max(1080 / w, 1920 / h)
    base_w, base_h = (ceil(dim * factor / 2) * 2 for dim in size)
    if min(w / 1080, h / 1920) <= 1:
        base_w, base_h = 1080, 1920
    piece = ZoomPiece(7, 47, ZoomCurve.swell(7, 47, 1.217, (0.38, 0.62)), base_w, base_h)
    scale = 1 + float(f"{piece.curve.peak - 1:.12f}") * sin(pi * (frame - 7) / 40) ** 2
    scaled_w, scaled_h = (trunc(dim * scale / 2) * 2 for dim in (base_w, base_h))
    rx, ry = zoom_ratios(piece, frame)
    assert (rx, ry) == (scaled_w / base_w, scaled_h / base_h)
    cut = _cut(47)
    mapped, _ = _mapped(cut, zooms=[piece], total=47)
    original = cut.frames[frame][0]
    ox, oy = (base_w - 1080) * 0.3, (base_h - 1920) / 2
    crop_x = ox * rx + 1080 * 0.38 * (rx - 1)
    crop_y = oy * ry + 1920 * 0.62 * (ry - 1)
    expected = Box((original.box.x + ox) * rx - crop_x,
                   (original.box.y + oy) * ry - crop_y,
                   original.box.w * rx, original.box.h * ry)
    actual = mapped.frames[frame][0]
    assert (actual.box.x, actual.box.y, actual.box.w, actual.box.h) == pytest.approx(
        (expected.x, expected.y, expected.w, expected.h), abs=1e-9,
    )
    assert (actual.confidence, actual.source) == (original.confidence, original.source)


def test_zoom_piece_keeps_whole_element_phase() -> None:
    whole = ZoomPiece(0, 12, ZoomCurve.swell(0, 12, 1.2, (0.5, 0.5)), 1080, 1920)
    piece = replace(whole, start_frame=5, end_frame=10)
    assert zoom_ratios(piece, 6) == zoom_ratios(whole, 6)
    mapped, runs = _mapped(_cut(), zooms=[piece])
    assert mapped.frames[4] == _cut().frames[4]
    assert runs == [{"start_frame": 0, "end_frame": 5, "source": "cut"},
                    {"start_frame": 5, "end_frame": 10, "source": "zoom"},
                    {"start_frame": 10, "end_frame": 12, "source": "cut"}]


def test_insert_overrides_zoom_and_hold_repeats_last_cut_frame() -> None:
    cut = _cut()
    piece = ZoomPiece(2, 12, ZoomCurve.swell(2, 12, 1.2, (0.4, 0.7)), 1080, 1920)
    mapped, runs = _mapped(cut, zooms=[piece], inserts=[(4, 6), (6, 8), (11, 12)], hold=3)
    assert all(not mapped.frames[n] for n in [4, 5, 6, 7, 11])
    assert mapped.frames[12:] == (cut.frames[-1],) * 3
    assert mapped.gaps == ((4, 8), (11, 12))
    assert runs == [{"start_frame": 0, "end_frame": 2, "source": "cut"},
                    {"start_frame": 2, "end_frame": 4, "source": "zoom"},
                    {"start_frame": 4, "end_frame": 8, "source": "insert"},
                    {"start_frame": 8, "end_frame": 11, "source": "zoom"},
                    {"start_frame": 11, "end_frame": 12, "source": "insert"},
                    {"start_frame": 12, "end_frame": 15, "source": "hold"}]
    assert mapped.sampling == {"cut": cut.sampling, "frame_sources": runs}
    assert mapped.detector == {**cut.detector, "mapped_from": "cut/cut.mp4",
                               "cut_sha256": cut.video.sha256}
    assert FacesData.from_dict(mapped.to_dict()).gaps == mapped.gaps


def test_transition_overrides_insert_zoom_and_cut_on_exact_window() -> None:
    cut = _cut()
    zoom = ZoomPiece(0, 6, ZoomCurve.swell(0, 6, 1.2, (0.5, 0.5)), 1080, 1920)
    mapped, runs = _mapped(cut, zooms=[zoom], inserts=[(3, 7)],
                           transitions=[(2, 5), (5, 9), (11, 12)], hold=2)
    assert mapped.frames[1] and mapped.frames[9] == cut.frames[9]
    assert all(not mapped.frames[frame] for frame in [*range(2, 9), 11])
    assert mapped.frames[12:] == (cut.frames[-1],) * 2
    assert mapped.gaps == ((2, 9), (11, 12))
    assert runs == [
        {"start_frame": 0, "end_frame": 2, "source": "zoom"},
        {"start_frame": 2, "end_frame": 9, "source": "transition"},
        {"start_frame": 9, "end_frame": 11, "source": "cut"},
        {"start_frame": 11, "end_frame": 12, "source": "transition"},
        {"start_frame": 12, "end_frame": 14, "source": "hold"},
    ]
    assert mapped.sampling["frame_sources"] == runs


@pytest.mark.parametrize("wipe", [0, 2, 9])
def test_layout_wipe_has_no_faces_on_its_exact_window(wipe: int) -> None:
    cut = _cut()
    mapped, sources = map_cut_faces(
        cut, total_frames=12, end_hold_frames=0, zooms=(), inserts=(), video=cut.video,
        layouts=[(3, 9, (594, 1056, 486, 864), wipe)],
    )
    assert mapped.frames[:3] == cut.frames[:3]
    assert mapped.frames[9:] == cut.frames[9:]
    assert mapped.frames[3:min(3 + wipe, 9)] == ((),) * min(wipe, 6)
    for frame in range(min(3 + wipe, 9), 9):
        face = cut.frames[frame][0]
        assert mapped.frames[frame] == (replace(face, box=Box(
            594 + .45 * face.box.x, 1056 + .45 * face.box.y,
            .45 * face.box.w, .45 * face.box.h,
        )),)
    assert [item for item in sources if item["source"] == "background_wipe"] == (
        [{"start_frame": 3, "end_frame": min(3 + wipe, 9), "source": "background_wipe"}]
        if wipe else []
    )


def test_estimate_widens_each_edge_and_recomputes_cut_gaps() -> None:
    cut = _cut()
    frames = tuple(() if n in (0, 1, 11) else faces for n, faces in enumerate(cut.frames))
    cut = replace(cut, frames=frames, gaps=tuple(find_gaps(frames)))
    known = ZoomPiece(2, 9, ZoomCurve.swell(2, 9, 1.15, (0.2, 0.8)), 1080, 1920)
    exact, _ = _mapped(cut, zooms=[known])
    estimated, runs = _mapped(cut, zooms=[replace(known, base_w=None, base_h=None)], hold=2)
    for frame in range(2, 9):
        a, b = exact.frames[frame][0].box, estimated.frames[frame][0].box
        assert b == Box(a.x - 2, a.y - 2, a.w + 4, a.h + 4)
    assert estimated.gaps == ((0, 2), (11, 14))
    assert runs[1] == {"start_frame": 2, "end_frame": 9, "source": "zoom_estimate"}


def test_unmeasured_preserves_plain_reason() -> None:
    cut = replace(_cut(), status="unmeasured", reason="detector unavailable", frames=(), gaps=())
    mapped, runs = _mapped(cut, hold=2)
    assert (mapped.status, mapped.reason, mapped.frames, mapped.gaps, runs) == (
        "unmeasured", "detector unavailable", (), (), [],
    )
    assert mapped.video.frames == 14


def test_frame_count_mismatch_is_unmeasured() -> None:
    mapped, runs = _mapped(_cut(11))
    assert mapped.status == "unmeasured"
    assert "11" in mapped.reason and "12" in mapped.reason
    assert mapped.frames == () and runs == []


def test_punch_enlarges_face_during_hold_and_returns_at_end() -> None:
    from yap_studio.zoom_curve import ZoomCurve, punch_keys

    cut = _cut(90)
    piece = ZoomPiece(10, 70, ZoomCurve(10, 70, punch_keys(10, 70, 1.3, (.5, .42))),
                      1080, 1920)
    mapped, _ = _mapped(cut, zooms=[piece], total=90)
    assert mapped.frames[10] == cut.frames[10]
    assert mapped.frames[40][0].box.w == pytest.approx(cut.frames[40][0].box.w * 1.3)
    assert mapped.frames[70] == cut.frames[70]
