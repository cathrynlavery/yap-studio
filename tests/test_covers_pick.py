"""Cover frame selection on the final picture clock."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from yap_studio.covers.pick import TimelineSnapshot, eligible_frame, pick_frame
from yap_studio.errors import ValidationError
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.gallery import still
from yap_studio.gallery.still import FaceLandmarks


def _faces() -> FacesData:
    frames = []
    for frame in range(90):
        size = {15: 350, 30: 350, 45: 350, 75: 330}.get(frame, 200)
        frames.append((FaceBox(Box(300, 700, size, size), 0.9, "detected"),))
    return FacesData("measured", None, VideoIdentity("cut.mp4", "hash", 90,
                     Fraction(30), 1080, 1920), {}, {}, tuple(frames), ())


def _timeline() -> TimelineSnapshot:
    return TimelineSnapshot(90, 10, (("insert", 30, 40), ("transition", 60, 70)))


def test_snapshot_accepts_non_replacement_ending_in_end_hold() -> None:
    snapshot = TimelineSnapshot.from_document({
        "schema": "yap-studio/resolved-timeline@1",
        "total_frames": 90,
        "end_hold_frames": 10,
        "elements": [
            {"kind": "title", "start_frame": 80, "end_frame": 100},
            {"kind": "card", "start_frame": 90, "end_frame": 100},
            {"kind": "insert", "start_frame": 30, "end_frame": 40},
        ],
    })
    assert snapshot.replacements == (("insert", 30, 40),)
    assert eligible_frame(85, snapshot) is None
    assert "end hold" in eligible_frame(95, snapshot)
    with pytest.raises(ValidationError, match="invalid frame window"):
        TimelineSnapshot.from_document({
            "schema": "yap-studio/resolved-timeline@1",
            "total_frames": 90,
            "end_hold_frames": 10,
            "elements": [{"kind": "title", "start_frame": 80, "end_frame": 101}],
        })


def test_eligibility_and_override() -> None:
    timeline = _timeline()
    assert eligible_frame(15, timeline) is None
    assert "insert" in eligible_frame(30, timeline)
    assert "transition" in eligible_frame(60, timeline)
    assert "end hold" in eligible_frame(90, timeline)
    result = pick_frame(None, _faces(), timeline, frame=45)
    assert result.frame == 45
    with pytest.raises(ValidationError, match="insert"):
        pick_frame(None, _faces(), timeline, frame=30)
    unmeasured = FacesData("unmeasured", "helper absent", _faces().video, {}, {}, (), ())
    assert pick_frame(None, unmeasured, timeline, frame=15).frame == 15


def test_prefers_sharp_open_eyes_and_records_fallback() -> None:
    faces = _faces()
    timeline = _timeline()
    def image(frame: int) -> np.ndarray:
        pixels = np.zeros((1920, 1080), dtype=np.uint8)
        if frame != 15:
            pixels[700:1050, 300:650] = np.indices((350, 350)).sum(axis=0) % 2 * 255
        return pixels
    class Detector:
        def measure(self, images: list) -> list[list[FaceLandmarks]]:
            def eyes(path: object) -> float:
                return 0.01 if Path(str(path)).name == "frame-00045.png" else 0.4
            return [[FaceLandmarks(300, 700, 350, 350, eyes(p), eyes(p), 0, 0, 1)]
                    for p in images]
    result = pick_frame(None, faces, timeline, image_loader=image,
                        landmark_detector=Detector())
    assert result.frame == 75
    by_frame = {row["frame"]: row for row in result.candidates}
    assert by_frame[15]["motion_blur"]
    assert by_frame[45]["eyes_open"] is False
    assert by_frame[75]["face_area"] > by_frame[0]["face_area"]
    assert 30 not in by_frame and 60 not in by_frame
    fallback = pick_frame(None, faces, timeline, image_loader=image, landmark_detector=None)
    assert "unavailable" in fallback.reason.lower()


def test_open_eyes_after_first_area_batch_are_picked() -> None:
    frames = []
    for index in range(450):
        size = 320 - index // 15 if index % 15 == 0 else 200
        frames.append((FaceBox(Box(300, 700, size, size), 0.9, "detected"),))
    faces = FacesData("measured", None, VideoIdentity("cut.mp4", "hash", 450,
                      Fraction(30), 1080, 1920), {}, {}, tuple(frames), ())

    pixels = np.indices((1920, 1080)).sum(axis=0).astype(np.uint8) % 2 * 255

    def image(_frame: int) -> np.ndarray:
        return pixels

    class Detector:
        def measure(self, images: list) -> list[list[FaceLandmarks]]:
            def eyes(path: object) -> float:
                return 0.4 if Path(str(path)).name == "frame-00360.png" else 0.01
            return [[FaceLandmarks(300, 700, 300, 300, eyes(p), eyes(p), 0, 0, 1)]
                    for p in images]

    result = pick_frame(None, faces, TimelineSnapshot(450, 0, ()),
                        image_loader=image, landmark_detector=Detector())
    assert result.frame == 360
    assert len(result.candidates) == 30


def _batch_faces(count: int) -> FacesData:
    frames = tuple((FaceBox(Box(300, 700, 300 - index // 15, 300 - index // 15),
                            0.9, "detected"),)
                   for index in range(count * 15))
    video = VideoIdentity("cut.mp4", "hash", len(frames), Fraction(30), 1080, 1920)
    return FacesData("measured", None, video, {}, {}, frames, ())


def _batch_detector(first: tuple[float | None, float | None],
                    later: tuple[float | None, float | None]):
    class Detector:
        def measure(self, images: list) -> list[list[FaceLandmarks]]:
            return [[FaceLandmarks(300, 700, 300, 300, *(
                first if int(path.stem.split("-")[1]) < 360 else later), 0, 0, 1)]
                for path in images]

    return Detector()


def test_incomplete_eyes_do_not_raise_shared_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(still, "_sharpness", lambda pixels, _face:
                        10.0 if pixels[0, 0] < 24 else 100.0)
    result = pick_frame(None, _batch_faces(25), TimelineSnapshot(375, 0, ()),
                        image_loader=lambda frame: np.array([[frame // 15]], dtype=np.uint8),
                        landmark_detector=_batch_detector((1.0, None), (0.3, 0.3)))
    assert result.frame == 360
    assert result.candidates[-1]["eyes_open"] is True


def test_later_batch_eyes_use_all_measured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(still, "_sharpness", lambda pixels, _face:
                        10.0 if pixels[0, 0] < 24 else 100.0)
    with pytest.raises(ValidationError, match="no sharp frame with open eyes"):
        pick_frame(None, _batch_faces(25), TimelineSnapshot(375, 0, ()),
                   image_loader=lambda frame: np.array([[frame // 15]], dtype=np.uint8),
                   landmark_detector=_batch_detector((0.9, 0.4), (0.25, 0.25)))


def test_rescored_eyes_can_select_an_earlier_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(still, "_sharpness", lambda _pixels, _face: 100.0)
    result = pick_frame(None, _batch_faces(48), TimelineSnapshot(720, 0, ()),
                        image_loader=lambda _frame: np.zeros((1, 1), dtype=np.uint8),
                        landmark_detector=_batch_detector((0.9, 0.4), (0.25, 0.25)))
    assert result.frame == 0
    assert len(result.candidates) == 48
    assert all(row["eyes_open"] for row in result.candidates[:24])
    assert not any(row["eyes_open"] for row in result.candidates[24:])


def test_later_sharpness_rescores_every_measured_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(still, "_sharpness", lambda pixels, _face:
                        10.0 if pixels[0, 0] < 24 else 100.0)
    result = pick_frame(None, _batch_faces(25), TimelineSnapshot(375, 0, ()),
                        image_loader=lambda frame: np.array([[frame // 15]], dtype=np.uint8),
                        landmark_detector=_batch_detector((0.9, 0.4), (0.9, 0.9)))
    assert result.frame == 360
    assert len(result.candidates) == 25
    assert all(row["motion_blur"] for row in result.candidates[:24])
    assert all(not row["eyes_open"] for row in result.candidates[:24])
    assert result.candidates[0]["score"] == 1.1
