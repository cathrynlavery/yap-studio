"""Deterministic face models, sampling, helper protocol, and detector tests."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from yap_studio import ffmpeg, paths, receipt, tools
from yap_studio.cache import cache_dir
from yap_studio.errors import InputError, OutputPathError, ProcessingError, ValidationError
from yap_studio.faces import (
    DETECTOR_NAME,
    FACES_SCHEMA,
    Box,
    DetectorRun,
    FaceBox,
    FacesData,
    VideoIdentity,
    VisionDetector,
    detect,
    detect_faces,
    find_gaps,
    interpolate,
    load_faces,
    sample_frames,
    write_faces,
)


def make_clip(path: Path, *, frames: int = 30, fps: Fraction | int = 30) -> Path:
    """Create a display-space video with an exact decoded frame count."""
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=108x192:rate={fps}",
            "-frames:v",
            str(frames),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        timeout_s=30,
    )
    return path


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


def face(
    x: float = 10, y: float = 20, w: float = 10, h: float = 12, confidence: float = 0.9
) -> FaceBox:
    return FaceBox(Box(x, y, w, h), confidence, "detected")


class FakeDetector:
    """Report faces at the beginning and end, with an empty interval between."""

    def __init__(self, frames: int = 30, width: int = 108, height: int = 192) -> None:
        self.frames = frames
        self.width = width
        self.height = height
        self.calls: list[tuple[Path, int]] = []

    def detect(self, video: Path, *, every: int) -> DetectorRun:
        self.calls.append((video, every))
        samples = {
            index: (face(x=10 + index),) if index <= 5 or index >= 25 else ()
            for index in sample_frames(self.frames, every)
        }
        return DetectorRun(
            self.width, self.height, self.frames, samples, {"name": DETECTOR_NAME, "revision": 3}
        )


class VariableRateDetector(FakeDetector):
    """Supply presentation times with slower playback in the first 30 frames."""

    def __init__(self) -> None:
        super().__init__(frames=60)

    def detect(self, video: Path, *, every: int) -> DetectorRun:
        run = super().detect(video, every=every)
        return replace(
            run,
            samples={index: (face(),) for index in sample_frames(self.frames, every)},
            times=tuple(i / 15 if i < 30 else 2 + (i - 30) / 30 for i in range(60)),
            end_time=3.0,
        )


@pytest.mark.parametrize(
    ("frames", "every", "expected"),
    [
        (1, 1, [0]),
        (5, 1, [0, 1, 2, 3, 4]),
        (10, 4, [0, 4, 8, 9]),
        (9, 4, [0, 4, 8]),
        (3, 20, [0, 2]),
    ],
)
def test_sample_frames(frames: int, every: int, expected: list[int]) -> None:
    assert sample_frames(frames, every) == expected


@pytest.mark.parametrize(("frames", "every"), [(0, 1), (-1, 1), (1, 0), (2, -1)])
def test_sample_frames_bad_arguments(frames: int, every: int) -> None:
    with pytest.raises(ValueError):
        sample_frames(frames, every)


def test_interpolation_uses_union_on_every_intermediate_frame_and_preserves_samples() -> None:
    first = face(x=0, y=0, w=20, h=20)
    last = face(x=10, y=8, w=30, h=40, confidence=0.7)
    result = interpolate({0: (first,), 4: (last,)}, 5)
    assert result[1:4] == [(FaceBox(Box(0, 0, 40, 48), 0.7, "interpolated"),)] * 3
    assert result[0] == (first,)
    assert result[4] == (last,)


def test_sparse_sampling_covers_a_face_that_stays_then_jumps() -> None:
    first, last = face(x=0), face(x=10)
    result = interpolate({0: (first,), 3: (last,)}, 4)
    for frame in (1, 2):
        (held,) = result[frame]
        assert held.box == Box(0, 20, 20, 12)
        for sampled in (first.box, last.box):
            assert held.box.x <= sampled.x and held.box.y <= sampled.y
            assert held.box.x1 >= sampled.x1 and held.box.y1 >= sampled.y1


@pytest.mark.parametrize("left_empty", [False, True])
def test_unmatched_face_is_held_from_either_side(left_empty: bool) -> None:
    box = face()
    samples = {0: () if left_empty else (box,), 3: (box,) if left_empty else ()}
    result = interpolate(samples, 4)
    assert result[1:3] == [(FaceBox(box.box, 0.9, "interpolated"),)] * 2
    assert result[0] == samples[0]
    assert result[3] == samples[3]


def test_unmatched_faces_on_both_sides_are_kept_in_sorted_order() -> None:
    left, right = face(x=0, y=100), face(x=80, y=0)
    result = interpolate({0: (left,), 2: (right,)}, 3)
    assert result[1] == (
        FaceBox(right.box, 0.9, "interpolated"),
        FaceBox(left.box, 0.9, "interpolated"),
    )


def test_empty_sample_pair_stays_empty() -> None:
    assert interpolate({0: (), 4: ()}, 5) == [()] * 5


def test_centre_distance_matches_nonoverlapping_nearby_boxes() -> None:
    result = interpolate({0: (face(x=0),), 2: (face(x=10),)}, 3)
    assert result[1] == (FaceBox(Box(0, 20, 20, 12), 0.9, "interpolated"),)


def test_iou_matching_is_greedy_before_centre_distance() -> None:
    samples = {0: (face(x=0, w=20), face(x=18, w=20)), 2: (face(x=2, w=20), face(x=16, w=20))}
    result = interpolate(samples, 3)
    assert [item.box for item in result[1]] == [Box(0, 20, 22, 12), Box(16, 20, 22, 12)]


@pytest.mark.parametrize(
    ("per_frame", "expected"),
    [
        ([], []),
        ([(), (), ()], [(0, 3)]),
        ([(), (face(),), (), (), (face(),), ()], [(0, 1), (2, 4), (5, 6)]),
        ([(face(),)], []),
    ],
)
def test_find_gaps(per_frame: list[tuple[FaceBox, ...]], expected: list[tuple[int, int]]) -> None:
    assert find_gaps(per_frame) == expected


@pytest.fixture
def measured() -> FacesData:
    return FacesData(
        "measured",
        None,
        VideoIdentity("clip.mp4", "a" * 64, 3, Fraction(30), 108, 192),
        {"name": DETECTOR_NAME, "revision": 3},
        {"every": 1, "sampled_frames": 3, "interpolated_frames": 0},
        ((face(),), (), (face(),)),
        ((1, 2),),
        ("example warning",),
    )


@pytest.mark.parametrize("gaps", [(), ((1, 3),), ((1, 2), (1, 2)), ((0, 1),)])
def test_faces_gap_invariant(measured: FacesData, gaps: tuple[tuple[int, int], ...]) -> None:
    with pytest.raises(ValueError, match="maximal runs"):
        replace(measured, gaps=gaps)


def test_faces_document_roundtrip_and_file_io(measured: FacesData, tmp_path: Path) -> None:
    payload = json.loads(json.dumps(measured.to_dict(), allow_nan=False))
    assert FacesData.from_dict(payload) == measured
    assert list(payload) == [
        "schema",
        "status",
        "reason",
        "video",
        "detector",
        "sampling",
        "frames",
        "gaps",
        "end_time",
        "warnings",
    ]
    assert payload["schema"] == FACES_SCHEMA
    assert payload["video"]["fps"] == "30/1"
    assert payload["frames"][1] == {"frame": 1, "faces": []}
    assert payload["end_time"] is None
    assert measured.time_at(1) == float(Fraction(1, 30))
    assert measured.time_at(3) == 0.1
    assert measured.faces_at(0) == (face(),)
    assert measured.faces_at(1) == ()
    with pytest.raises(IndexError):
        measured.faces_at(-1)
    out = write_faces(tmp_path / "faces.json", measured)
    assert load_faces(out) == measured
    assert out.read_text().endswith("\n")
    assert write_faces(out, measured) == out


def test_faces_presentation_times_roundtrip(measured: FacesData, tmp_path: Path) -> None:
    timed = replace(measured, times=(0.1, 0.123456, 0.2), end_time=0.3)
    payload = timed.to_dict()
    assert payload["frames"][1] == {"frame": 1, "t": 0.123456, "faces": []}
    assert list(payload["frames"][1]) == ["frame", "t", "faces"]
    assert payload["end_time"] == 0.3
    assert FacesData.from_dict(json.loads(json.dumps(payload))) == timed
    assert load_faces(write_faces(tmp_path / "timed.json", timed)) == timed
    assert timed.time_at(0) == 0.1
    assert timed.time_at(1) == 0.123456
    assert timed.time_at(3) == 0.3
    rounded = replace(timed, times=(0.1, 0.12345678, 0.2), end_time=0.34567891).to_dict()
    assert rounded["frames"][1]["t"] == 0.123457
    assert rounded["end_time"] == 0.345679


def test_faces_accept_legacy_documents_without_timing(measured: FacesData) -> None:
    payload = measured.to_dict()
    del payload["end_time"]
    assert FacesData.from_dict(payload) == measured


@pytest.mark.parametrize(
    "updates",
    [
        {"times": None, "end_time": 0.3},
        {"times": (0.0, 0.1, 0.2)},
        {"times": (0.0, 0.1), "end_time": 0.3},
        {"times": [0.0, 0.1, 0.2], "end_time": 0.3},
        {"times": (0.0, 0.2, 0.1), "end_time": 0.3},
        {"times": (0.0, float("nan"), 0.2), "end_time": 0.3},
        {"times": (0.0, 0.1, float("inf")), "end_time": 0.3},
        {"times": (False, 0.1, 0.2), "end_time": 0.3},
        {"times": (0.0, "0.1", 0.2), "end_time": 0.3},
        {"times": (0.0, 0.1, 0.2), "end_time": 0.1},
        {"times": (0.0, 0.1, 0.2), "end_time": float("inf")},
        {"times": (0.0, 0.1, 0.2), "end_time": float("nan")},
        {"times": (0.0, 0.1, 0.2), "end_time": True},
    ],
)
def test_faces_timing_invariants(measured: FacesData, updates: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(measured, **updates)


@pytest.mark.parametrize("failure", ["mixed", "no-end", "no-times", "bad-time", "bad-end"])
def test_faces_document_rejects_incomplete_or_invalid_timing(
    measured: FacesData, failure: str
) -> None:
    payload = replace(measured, times=(0.0, 0.1, 0.2), end_time=0.3).to_dict()
    if failure == "mixed":
        del payload["frames"][1]["t"]
    elif failure == "no-end":
        del payload["end_time"]
    elif failure == "no-times":
        for record in payload["frames"]:
            del record["t"]
    elif failure == "bad-time":
        payload["frames"][1]["t"] = "0.1"
    elif failure == "bad-end":
        payload["end_time"] = "0.3"
    with pytest.raises(ValidationError):
        FacesData.from_dict(payload)


def test_equal_presentation_times_are_valid(measured: FacesData) -> None:
    timed = replace(measured, times=(0.0, 0.1, 0.1), end_time=0.1)
    assert FacesData.from_dict(timed.to_dict()) == timed


@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize("frame", [-1, 4, True, 1.0])
def test_time_at_rejects_indices_outside_video(
    measured: FacesData, timed: bool, frame: Any
) -> None:
    if timed:
        measured = replace(measured, times=(0.0, 0.1, 0.2), end_time=0.3)
    with pytest.raises(IndexError):
        measured.time_at(frame)


@pytest.mark.parametrize("protection", ["validate_input", "protect_input"])
def test_write_faces_preserves_callers_input_protection(
    measured: FacesData, tmp_path: Path, protection: str
) -> None:
    faces_path = tmp_path / "faces.json"
    faces_path.write_text(json.dumps(measured.to_dict()) + "\n")
    original = faces_path.read_bytes()
    with paths.operation():
        getattr(paths, protection)(faces_path)
        with pytest.raises(OutputPathError, match="input file"):
            write_faces(faces_path, measured)
    assert faces_path.read_bytes() == original


def test_write_faces_works_without_enclosing_operation(measured: FacesData, tmp_path: Path) -> None:
    faces_path = tmp_path / "new.json"
    assert write_faces(faces_path, measured) == faces_path
    assert load_faces(faces_path) == measured


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("schema",), "yap-studio/faces@2"),
        (("status",), "unknown"),
        (("video", "frames"), "3"),
        (("video", "width"), True),
        (("video", "fps"), 30),
        (("video", "fps"), "30/0"),
        (("video", "name"), "/folder/clip.mp4"),
        (("detector",), []),
        (("sampling",), None),
        (("warnings",), [12]),
        (("reason",), 4),
        (("frames", 0, "frame"), True),
        (("frames", 0, "frame"), 1),
        (("frames", 0, "faces", 0, "x"), "10"),
        (("frames", 0, "faces", 0, "h"), -1),
        (("frames", 0, "faces", 0, "confidence"), float("nan")),
        (("frames", 0, "faces", 0, "source"), "guessed"),
        (("gaps", 0, "start_frame"), 1.0),
    ],
)
def test_document_rejects_wrong_schema_and_types(
    measured: FacesData, location: tuple[str | int, ...], value: Any
) -> None:
    payload = measured.to_dict()
    target = payload
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(ValidationError):
        FacesData.from_dict(payload)


@pytest.mark.parametrize(
    "key",
    ["schema", "status", "reason", "video", "detector", "sampling", "frames", "gaps", "warnings"],
)
def test_document_requires_all_keys(measured: FacesData, key: str) -> None:
    payload = measured.to_dict()
    del payload[key]
    with pytest.raises(ValidationError):
        FacesData.from_dict(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"reason": None},
        {"reason": ""},
        {"reason": " "},
        {"frames": ((face(),),)},
        {"gaps": ((0, 3),)},
        {"times": (0.0, 0.1, 0.2), "end_time": 0.3},
    ],
)
def test_unmeasured_invariants(measured: FacesData, updates: dict[str, Any]) -> None:
    unmeasured = replace(
        measured, status="unmeasured", reason="Vision unavailable", frames=(), gaps=()
    )
    assert FacesData.from_dict(unmeasured.to_dict()) == unmeasured
    assert unmeasured.faces_at(0) == ()
    with pytest.raises(ValueError):
        replace(unmeasured, **updates)


@pytest.mark.parametrize(
    "box", [(0, 0, -1, 2), (0, 0, 1, -2), (float("nan"), 0, 1, 2), (0, float("inf"), 1, 2)]
)
def test_box_validation(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        Box(*box)


def test_box_edges_and_rounding() -> None:
    box = Box(1.234, 2.345, 3.456, 4.567)
    assert box.x1 == pytest.approx(4.69)
    assert box.y1 == pytest.approx(6.912)
    assert box.to_dict() == {"x": 1.23, "y": 2.35, "w": 3.46, "h": 4.57}
    assert FaceBox(box, 0.89234, "detected").to_dict()["confidence"] == 0.8923


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf"), float("nan")])
def test_face_confidence_validation(confidence: float) -> None:
    with pytest.raises(ValueError):
        face(confidence=confidence)


def test_load_reports_unreadable_and_invalid_paths(tmp_path: Path) -> None:
    source = tmp_path / "faces.json"
    with pytest.raises(InputError, match="faces.json"):
        load_faces(source)
    for text in ("{bad", "[]", '{"schema":"unknown"}'):
        source.write_text(text)
        with pytest.raises(ValidationError, match="faces.json"):
            load_faces(source)


def test_detect_faces_identity_sampling_and_gaps(
    clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        detect, "unavailable_reason", lambda: "should not affect an injected detector"
    )
    fake = FakeDetector()
    data = detect_faces(clip, every=5, detector=fake)
    assert data.status == "measured"
    assert data.video == VideoIdentity(
        clip.name, receipt.sha256_file(clip), 30, Fraction(30), 108, 192
    )
    assert len(data.frames) == 30
    assert data.gaps == ((10, 21),)
    assert data.sampling == {"every": 5, "sampled_frames": 7, "interpolated_frames": 23}
    assert data.faces_at(2)[0].box == Box(10, 20, 15, 12)
    assert data.faces_at(2)[0].source == "interpolated"
    assert data.faces_at(5)[0].source == "detected"
    assert data.faces_at(29)[0].source == "detected"
    assert fake.calls == [(clip.resolve(), 5)]
    assert data.times is None and data.end_time is None
    assert data.time_at(30) == 1.0


def test_detect_faces_copies_presentation_times(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "variable.mp4", frames=60, fps=Fraction(45, 2))
    fake = VariableRateDetector()
    data = detect_faces(clip, detector=fake)
    assert data.times == tuple(i / 15 if i < 30 else 2 + (i - 30) / 30 for i in range(60))
    assert data.time_at(30) == 2.0
    assert data.time_at(60) == data.end_time == 3.0
    assert float(30 / data.video.fps) == pytest.approx(4 / 3)


@pytest.mark.parametrize("fake", [FakeDetector(frames=29), FakeDetector(width=192, height=108)])
def test_detector_mismatch_is_explicit(clip: Path, fake: FakeDetector) -> None:
    with pytest.raises(ProcessingError) as caught:
        detect_faces(clip, detector=fake)
    assert caught.value.code == "face_frame_mismatch"
    assert "30 frames at 108x192" in caught.value.full_stderr
    assert f"{fake.frames} frames at {fake.width}x{fake.height}" in caught.value.full_stderr


@pytest.mark.parametrize("missing", ["has_video", "video_frames", "fps"])
def test_detection_requires_video_count_and_rate(
    clip: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    info = replace(ffmpeg.probe(clip), **{missing: False if missing == "has_video" else None})
    monkeypatch.setattr(ffmpeg, "probe", lambda _: info)
    with pytest.raises(InputError, match="known frame count and fps"):
        detect_faces(clip, detector=FakeDetector())


def test_variable_frame_rate_warning(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = replace(ffmpeg.probe(clip), nominal_fps=Fraction(60))
    monkeypatch.setattr(ffmpeg, "probe", lambda _: info)
    data = detect_faces(clip, detector=FakeDetector())
    assert data.warnings == (
        "variable frame rate: frame numbers follow decode order, not time x fps",
    )


@pytest.mark.parametrize(("system", "phrase"), [("linux", "macOS"), ("darwin", "Xcode")])
def test_unavailable_detection_never_invokes_default(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, phrase: str
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(detect.sys, "platform", system)
    monkeypatch.setattr(tools, "find_tool", lambda _: None)

    def unexpected() -> None:
        pytest.fail("unavailable detection invoked the default detector")

    monkeypatch.setattr(detect, "default_detector", unexpected)
    data = detect_faces(clip, every=5)
    assert data.status == "unmeasured"
    assert phrase in data.reason
    assert data.frames == data.gaps == ()
    assert data.video.sha256 == receipt.sha256_file(clip)
    assert data.detector == {"name": DETECTOR_NAME}
    assert data.sampling == {"every": 5}


def executable(path: Path, body: str) -> Path:
    """Write a test-local helper with a known interpreter."""
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o700)
    return path


def test_helper_compile_is_cached_by_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = tmp_path / "calls.jsonl"
    executable(
        tmp_path / "xcrun",
        "import json, pathlib, sys\n"
        f"with pathlib.Path({str(calls)!r}).open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "out.write_text('#!/bin/sh\\nexit 0\\n')\n"
        "out.chmod(0o700)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))
    helper = detect.helper_executable()
    digest = hashlib.sha256(detect.helper_source().read_bytes()).hexdigest()
    assert helper == cache_dir("helpers") / f"yap-face-boxes-{digest[:16]}"
    assert os.access(helper, os.X_OK)
    assert detect.helper_executable() == helper
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(recorded) == 1
    assert recorded[0][:3] == ["swiftc", "-O", "-o"]
    assert Path(recorded[0][3]).parent == helper.parent
    assert not Path(recorded[0][3]).exists()
    assert recorded[0][-1] == str(detect.helper_source())


def test_helper_compile_failure_propagates_and_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable(tmp_path / "xcrun", "import sys\nsys.stderr.write('compiler failed')\nsys.exit(7)\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))
    with pytest.raises(ProcessingError) as caught:
        detect.helper_executable()
    assert caught.value.code == "xcrun_exit_7"
    assert list(cache_dir("helpers").iterdir()) == []


def test_unavailable_when_swift_lookup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable(tmp_path / "xcrun", "import sys\nsys.exit(1)\n")
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr(detect.sys, "platform", "darwin")
    assert "Xcode command line tools" in detect.unavailable_reason()


def protocol(every: int = 5) -> list[dict[str, Any]]:
    records = [{"type": "header", "width": 108, "height": 192, "revision": 3}]
    records.extend(
        {
            "type": "frame",
            "frame": index,
            "pts": index / 30,
            "faces": [{"x": 10, "y": 20, "w": 10, "h": 12, "confidence": 0.9}]
            if index in sample_frames(30, every)
            else None,
        }
        for index in range(30)
    )
    records.append({"type": "end", "frames": 30, "end_pts": 1.0})
    return records


def test_parse_output_retains_skipped_frame_times_and_end() -> None:
    records = protocol()
    header, total, samples, times, end_time = detect._parse_output(
        "\n".join(json.dumps(record) for record in records), "helper", 5
    )
    assert header == records[0]
    assert total == 30
    assert list(samples) == sample_frames(30, 5)
    assert times == tuple(i / 30 for i in range(30))
    assert end_time == 1.0
    assert records[2]["faces"] is None and times[1] == 1 / 30


def test_helper_path_does_not_create_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(root))
    digest = hashlib.sha256(detect.helper_source().read_bytes()).hexdigest()
    assert detect.helper_path() == root / "helpers" / f"yap-face-boxes-{digest[:16]}"
    assert not root.exists()


def test_cached_helper_runs_without_build_tools(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))
    helper = detect.helper_path()
    helper.parent.mkdir(parents=True)
    stdout = "\n".join(json.dumps(record) for record in protocol()) + "\n"
    executable(helper, f"import sys\nsys.stdout.write({stdout!r})\n")
    monkeypatch.setattr(detect.sys, "platform", "darwin")
    monkeypatch.setattr(tools, "find_tool", lambda _: None)
    original = tools.run_tool
    calls = []

    def run(name: str, args: list[str], **kwargs: Any) -> Any:
        assert name != "xcrun"
        calls.append(name)
        return original(name, args, **kwargs)

    monkeypatch.setattr(tools, "run_tool", run)
    assert detect.unavailable_reason() is None
    data = detect_faces(clip, every=5)
    assert data.status == "measured"
    assert data.times == tuple(i / 30 for i in range(30))
    assert data.end_time == 1.0
    assert calls == ["yap-face-boxes"]


@pytest.mark.parametrize("nonexecutable", [False, True])
def test_missing_cached_helper_requires_build_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonexecutable: bool
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(root))
    monkeypatch.setattr(detect.sys, "platform", "darwin")
    monkeypatch.setattr(tools, "find_tool", lambda _: None)
    if nonexecutable:
        helper = detect.helper_path()
        helper.parent.mkdir(parents=True)
        helper.write_text("not executable")
        helper.chmod(0o600)
    assert "Xcode command line tools" in detect.unavailable_reason()
    if not nonexecutable:
        assert not root.exists()


def install_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str) -> Path:
    helper = executable(tmp_path / "helper", f"import sys\nsys.stdout.write({stdout!r})\n")
    monkeypatch.setattr(detect, "helper_executable", lambda: helper)
    return helper


def test_vision_detector_parses_protocol_and_uses_frame_count_deadline(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = install_protocol(
        tmp_path, monkeypatch, "\n".join(json.dumps(record) for record in protocol()) + "\n"
    )
    original = tools.run_tool
    calls = []

    def run(name: str, args: list[str], **kwargs: Any) -> Any:
        calls.append((name, args, kwargs))
        return original(name, args, **kwargs)

    monkeypatch.setattr(tools, "run_tool", run)
    result = VisionDetector().detect(clip, every=5)
    assert (result.width, result.height, result.frames) == (108, 192, 30)
    assert result.samples[0] == (face(),)
    assert list(result.samples) == sample_frames(30, 5)
    assert result.times == tuple(i / 30 for i in range(30))
    assert result.end_time == 1.0
    assert result.detector == {
        "name": DETECTOR_NAME,
        "revision": 3,
        "os": "macOS " + detect.platform.mac_ver()[0],
        "helper_sha256": hashlib.sha256(detect.helper_source().read_bytes()).hexdigest(),
    }
    assert calls == [
        ("yap-face-boxes", ["--every", "5", str(clip)], {"timeout_s": 135, "executable": helper})
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "missing-end",
        "missing-header",
        "bad-json",
        "wrong-set",
        "duplicate",
        "out-of-range",
        "bad-box",
        "after-end",
        "missing-skipped",
        "decreasing-pts",
        "missing-end-pts",
        "invalid-pts",
        "invalid-end-pts",
        "early-end-pts",
        "missing-faces",
        "invalid-faces",
        "wrong-total",
    ],
)
def test_malformed_helper_output_raises_processing_error(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    records = protocol()
    if failure == "missing-end":
        records.pop()
    elif failure == "missing-header":
        records.pop(0)
    elif failure == "wrong-set":
        records[6]["faces"] = None
    elif failure == "duplicate":
        records[2]["frame"] = 0
    elif failure == "out-of-range":
        records[-2]["frame"] = 30
    elif failure == "bad-box":
        records[1]["faces"][0]["w"] = -1
    elif failure == "after-end":
        records.append(records[-1])
    elif failure == "missing-skipped":
        records.pop(2)
    elif failure == "decreasing-pts":
        records[3]["pts"] = 0.0
    elif failure == "missing-end-pts":
        del records[-1]["end_pts"]
    elif failure == "invalid-pts":
        records[2]["pts"] = float("nan")
    elif failure == "invalid-end-pts":
        records[-1]["end_pts"] = float("inf")
    elif failure == "early-end-pts":
        records[-1]["end_pts"] = 0.5
    elif failure == "missing-faces":
        del records[2]["faces"]
    elif failure == "invalid-faces":
        records[2]["faces"] = {}
    elif failure == "wrong-total":
        records[-1]["frames"] = 31
    stdout = "\n".join(json.dumps(record) for record in records) + "\n"
    if failure == "bad-json":
        stdout = "not JSON\n" + stdout
    install_protocol(tmp_path, monkeypatch, stdout)
    with pytest.raises(ProcessingError) as caught:
        VisionDetector().detect(clip, every=5)
    assert caught.value.code == "face_helper_output"
    assert caught.value.returncode == 0


def test_helper_nonzero_exit_remains_processing_failure(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = executable(
        tmp_path / "helper", "import sys\nsys.stderr.write('failed')\nsys.exit(3)\n"
    )
    monkeypatch.setattr(detect, "helper_executable", lambda: helper)
    with pytest.raises(ProcessingError) as caught:
        VisionDetector().detect(clip, every=1)
    assert caught.value.code == "yap-face-boxes_exit_3"


_REAL_HELPER = pytest.mark.skipif(
    sys.platform != "darwin" or tools.find_tool("xcrun") is None,
    reason="real Vision helper needs macOS and xcrun",
)


@pytest.fixture
def real_helper(_owner_folders: Path) -> Path:
    """Reuse the digest-keyed helper through this test's cache link."""
    return detect.helper_executable()


@pytest.mark.helpers
@_REAL_HELPER
@pytest.mark.parametrize("every", [1, 5, 7, 100])
def test_real_vision_helper_covers_every_frame(
    clip: Path, real_helper: Path, every: int
) -> None:
    assert detect.helper_executable() == real_helper
    data = detect_faces(clip, every=every)
    assert data.status == "measured"
    assert data.video.frames == 30
    assert len(data.frames) == 30
    assert [record["frame"] for record in data.to_dict()["frames"]] == list(range(30))
    assert data.times == pytest.approx(tuple(i / 30 for i in range(30)), abs=0.000001)
    assert data.end_time == pytest.approx(1.0, abs=0.000001)
    assert data.sampling["sampled_frames"] == len(sample_frames(30, every))


@pytest.mark.helpers
@_REAL_HELPER
@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ((1, 0, 0, 1), "up"),
        ((-1, 0, 0, 1), "upMirrored"),
        ((-1, 0, 0, -1), "down"),
        ((1, 0, 0, -1), "downMirrored"),
        ((0, 1, 1, 0), "leftMirrored"),
        ((0, 1, -1, 0), "right"),
        ((0, -1, -1, 0), "rightMirrored"),
        ((0, -1, 1, 0), "left"),
        ((0, 2, -2, 0), "right"),
        ((0, 1, -3, 0), "right"),
        ((3, 0, 0, 0.5), "up"),
        ((-0.25, 0, 0, 4), "upMirrored"),
        ((-2, 0, 0, -7), "down"),
        ((5, 0, 0, -0.1), "downMirrored"),
        ((0, 0.5, 6, 0), "leftMirrored"),
        ((0, -9, -0.3, 0), "rightMirrored"),
        ((0, -1, 2.5, 0), "left"),
        ((1, 1, -1, 1), "up"),
        ((0, 0, 0, 0), "up"),
    ],
)
def test_real_helper_orientation(
    real_helper: Path, matrix: tuple[float, float, float, float], expected: str
) -> None:
    result = tools.run_tool(
        "yap-face-boxes",
        ["--orientation", *(str(value) for value in matrix)],
        executable=real_helper,
        timeout_s=30,
    )
    assert result.stdout.splitlines() == [f'{{"orientation":"{expected}"}}']
    if matrix in ((1, 1, -1, 1), (0, 0, 0, 0)):
        assert "warning" in result.stderr.lower()
    else:
        assert result.stderr == ""


@pytest.mark.helpers
@_REAL_HELPER
@pytest.mark.parametrize("bad_number", ["bad", "nan", "inf"])
def test_real_helper_orientation_rejects_bad_numbers(real_helper: Path, bad_number: str) -> None:
    with pytest.raises(ProcessingError) as caught:
        tools.run_tool(
            "yap-face-boxes",
            ["--orientation", bad_number, "0", "0", "1"],
            executable=real_helper,
            timeout_s=30,
        )
    assert caught.value.returncode == 2


@pytest.mark.helpers
@_REAL_HELPER
@pytest.mark.parametrize(
    ("sample", "step", "nominal", "expected"),
    [
        ("0", "0.04", "0.033333", 0.04),
        ("0.05", "0.04", "0.033333", 0.05),
        ("nan", "0.04", "0.033333", 0.04),
        ("0", "none", "0.033333", 0.033333),
        ("-1", "none", "inf", 0),
    ],
)
def test_real_helper_final_duration(
    real_helper: Path, sample: str, step: str, nominal: str, expected: float
) -> None:
    result = tools.run_tool(
        "yap-face-boxes",
        ["--final-duration", sample, step, nominal],
        executable=real_helper,
        timeout_s=30,
    )
    assert result.stdout.splitlines() == [f'{{"duration":{expected:.6f}}}']
    if expected == 0:
        assert "warning" in result.stderr.lower()
    else:
        assert result.stderr == ""


@pytest.mark.helpers
@_REAL_HELPER
def test_real_helper_final_duration_rejects_bad_step(real_helper: Path) -> None:
    with pytest.raises(ProcessingError) as caught:
        tools.run_tool(
            "yap-face-boxes",
            ["--final-duration", "0", "bad", "0.033333"],
            executable=real_helper,
            timeout_s=30,
        )
    assert caught.value.returncode == 2


_REAL_CLIP = os.environ.get("YAP_STUDIO_FACE_CLIP", "")


@pytest.mark.skipif(
    not _REAL_CLIP or not Path(_REAL_CLIP).is_file(),
    reason="set YAP_STUDIO_FACE_CLIP to a short real clip with a face",
)
@pytest.mark.helpers
@_REAL_HELPER
def test_local_real_face_clip(real_helper: Path) -> None:
    assert detect.helper_executable() == real_helper
    data = detect_faces(Path(_REAL_CLIP), every=1)
    assert data.status == "measured"
    assert sum(bool(boxes) for boxes in data.frames) >= 0.8 * data.video.frames
