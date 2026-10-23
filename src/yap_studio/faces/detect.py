"""Build the Vision helper and measure or conservatively interpolate face boxes.

Between samples a face is assumed to stay within the box spanning its two sampled
positions. The default --every 1 measures every frame.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from yap_studio import ffmpeg, paths, receipt, tools
from yap_studio.cache import cache_dir, cache_root
from yap_studio.errors import DependencyError, InputError, ProcessingError, ValidationError
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity

DETECTOR_NAME = "vision-face-rectangles"


def unavailable_reason() -> str | None:
    """Explain why Vision cannot run here, or return None when its tools are available."""
    if sys.platform != "darwin":
        return (
            "face detection uses the macOS Vision framework; "
            f"this computer runs {platform.system()}"
        )
    cached = helper_path()
    if cached.is_file() and os.access(cached, os.X_OK):
        return None
    reason = (
        "face detection needs the Xcode command line tools "
        "(xcode-select --install) to build its helper"
    )
    if tools.find_tool("xcrun") is None:
        return reason
    try:
        tools.run_tool("xcrun", ["--find", "swiftc"], timeout_s=30)
    except (DependencyError, ProcessingError):
        return reason
    return None


def helper_source() -> Path:
    """Return the packaged Swift source used to build the face detector."""
    return Path(__file__).with_name("face_boxes.swift")


def helper_path() -> Path:
    """Return the digest-named cached executable path without creating folders."""
    digest = hashlib.sha256(helper_source().read_bytes()).hexdigest()
    return cache_root() / "helpers" / f"yap-face-boxes-{digest[:16]}"


def helper_executable() -> Path:
    """Compile once per source digest and atomically publish the cached executable."""
    source = helper_source()
    executable = helper_path()
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    folder = cache_dir("helpers")
    descriptor, name = tempfile.mkstemp(prefix=".yap-face-boxes-", dir=folder)
    os.close(descriptor)
    temporary = Path(name)
    try:
        tools.run_tool("xcrun", ["swiftc", "-O", "-o", str(temporary), str(source)], timeout_s=600)
        os.replace(temporary, executable)
    finally:
        temporary.unlink(missing_ok=True)
    return executable


@dataclass(frozen=True)
class DetectorRun:
    """Decoded geometry and samples, with optional per-frame and end presentation seconds."""

    width: int
    height: int
    frames: int
    samples: dict[int, tuple[FaceBox, ...]]
    detector: dict[str, Any]
    times: tuple[float, ...] | None = None
    end_time: float | None = None


class FaceDetector(Protocol):
    """A detector that returns samples indexed by decoded frame order."""

    def detect(self, video: Path, *, every: int) -> DetectorRun:
        """Measure faces in the requested frames, including the final frame."""
        ...


def _integer(record: dict[str, Any], key: str, minimum: int) -> int:
    value = record.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _parse_output(
    stdout: str, command: str, every: int
) -> tuple[dict[str, Any], int, dict[int, tuple[FaceBox, ...]], tuple[float, ...], float]:
    header: dict[str, Any] | None = None
    total: int | None = None
    samples: dict[int, tuple[FaceBox, ...]] = {}
    times: list[float] = []
    end_time: float | None = None
    try:
        for line_number, line in enumerate(stdout.splitlines(), 1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            kind = record.get("type")
            if total is not None:
                raise ValueError("record after end")
            if header is None:
                if kind != "header":
                    raise ValueError("missing first header record")
                for key in ("width", "height", "revision"):
                    _integer(record, key, 1)
                header = record
            elif kind == "frame":
                index = _integer(record, "frame", 0)
                if index != len(times):
                    raise ValueError("frames must appear in order starting at zero, without gaps")
                pts = record.get("pts")
                if type(pts) not in (int, float) or not math.isfinite(pts):
                    raise ValueError("frame pts must be a finite number")
                if times and pts < times[-1]:
                    raise ValueError("frame pts must be non-decreasing")
                times.append(float(pts))
                if "faces" not in record:
                    raise ValueError("frame faces must be an array or null")
                faces = record["faces"]
                if faces is None:
                    continue
                if not isinstance(faces, list):
                    raise ValueError("frame faces must be an array or null")
                boxes = []
                for face in faces:
                    if not isinstance(face, dict):
                        raise ValueError("face must be an object")
                    boxes.append(FaceBox.from_dict({**face, "source": "detected"}))
                samples[index] = tuple(boxes)
            elif kind == "end":
                total = _integer(record, "frames", 1)
                end_pts = record.get("end_pts")
                if type(end_pts) not in (int, float) or not math.isfinite(end_pts):
                    raise ValueError("end_pts must be a finite number")
                end_time = float(end_pts)
            else:
                raise ValueError(f"unexpected record type: {kind!r}")
        if header is None or total is None:
            raise ValueError("missing header or end record")
        if len(times) != total:
            raise ValueError(f"expected a record for every frame in total frame count {total}")
        if end_time is None or end_time < times[-1]:
            raise ValueError("end_pts must be at or after the last frame pts")
        expected = sample_frames(total, every)
        if list(samples) != expected:
            raise ValueError(f"sampled frames {list(samples)} differ from expected {expected}")
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ProcessingError(
            command, 0, f"invalid face helper output: {exc}", code="face_helper_output"
        ) from exc
    return header, total, samples, tuple(times), end_time


class VisionDetector:
    """Detect display-space rectangles through the system Vision framework."""

    def detect(self, video: Path, *, every: int) -> DetectorRun:
        """Run the cached helper with a frame-count deadline and validate its protocol."""
        info = ffmpeg.probe(video)
        frames_hint = info.video_frames or 0
        helper = helper_executable()
        args = ["--every", str(every), str(video)]
        result = tools.run_tool(
            "yap-face-boxes", args, timeout_s=120 + 0.5 * frames_hint, executable=helper
        )
        header, total, samples, times, end_time = _parse_output(
            result.stdout, " ".join([str(helper), *args]), every
        )
        return DetectorRun(
            header["width"],
            header["height"],
            total,
            samples,
            {
                "name": DETECTOR_NAME,
                "revision": header["revision"],
                "os": "macOS " + platform.mac_ver()[0],
                "helper_sha256": hashlib.sha256(helper_source().read_bytes()).hexdigest(),
            },
            times,
            end_time,
        )


def default_detector() -> FaceDetector:
    """Return the system face rectangle detector."""
    return VisionDetector()


def sample_frames(frames: int, every: int) -> list[int]:
    """Select regular frames and always include the final frame."""
    if type(frames) is not int or frames < 1 or type(every) is not int or every < 1:
        raise ValueError("frames and every must be positive integers")
    selected = list(range(0, frames, every))
    if selected[-1] != frames - 1:
        selected.append(frames - 1)
    return selected


def _iou(a: Box, b: Box) -> float:
    area = max(0.0, min(a.x1, b.x1) - max(a.x, b.x)) * max(0.0, min(a.y1, b.y1) - max(a.y, b.y))
    union = a.w * a.h + b.w * b.h - area
    return area / union if union > 0 else 0.0


def _matches(left: Sequence[FaceBox], right: Sequence[FaceBox]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    overlaps = sorted(
        (-_iou(a.box, b.box), i, j)
        for i, a in enumerate(left)
        for j, b in enumerate(right)
        if _iou(a.box, b.box) > 0
    )
    for _, i, j in overlaps:
        if i not in used_left and j not in used_right:
            pairs.append((i, j))
            used_left.add(i)
            used_right.add(j)
    distances = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if i in used_left or j in used_right:
                continue
            distance = math.hypot(
                a.box.x + a.box.w / 2 - b.box.x - b.box.w / 2,
                a.box.y + a.box.h / 2 - b.box.y - b.box.h / 2,
            )
            if distance <= max(a.box.w, a.box.h, b.box.w, b.box.h):
                distances.append((distance, i, j))
    for _, i, j in sorted(distances):
        if i not in used_left and j not in used_right:
            pairs.append((i, j))
            used_left.add(i)
            used_right.add(j)
    return pairs


def interpolate(samples: Mapping[int, Sequence[FaceBox]], frames: int) -> list[tuple[FaceBox, ...]]:
    """Assume faces stay within the boxes spanning their sampled positions.

    Hold unmatched faces from both endpoints. The default --every 1 measures every
    frame and leaves all sampled boxes unchanged.
    """
    if type(frames) is not int or frames < 1:
        raise ValueError("frames must be a positive integer")
    if any(type(index) is not int or not 0 <= index < frames for index in samples):
        raise ValueError("sample frame is outside the video")
    result: list[tuple[FaceBox, ...]] = [()] * frames
    indices = sorted(samples)
    for index in indices:
        result[index] = tuple(sorted(samples[index], key=lambda face: (face.box.y, face.box.x)))
    for a, b in zip(indices, indices[1:], strict=False):
        if b == a + 1:
            continue
        left, right = result[a], result[b]
        pairs = _matches(left, right)
        used_left = {i for i, _ in pairs}
        used_right = {j for _, j in pairs}
        held = [
            FaceBox(face.box, face.confidence, "interpolated")
            for i, face in enumerate(left)
            if i not in used_left
        ]
        held.extend(
            FaceBox(face.box, face.confidence, "interpolated")
            for j, face in enumerate(right)
            if j not in used_right
        )
        boxes = list(held)
        for i, j in pairs:
            first, last = left[i], right[j]
            x, y = min(first.box.x, last.box.x), min(first.box.y, last.box.y)
            box = Box(
                x, y, max(first.box.x1, last.box.x1) - x, max(first.box.y1, last.box.y1) - y
            )
            boxes.append(FaceBox(box, min(first.confidence, last.confidence), "interpolated"))
        for frame in range(a + 1, b):
            result[frame] = tuple(sorted(boxes, key=lambda face: (face.box.y, face.box.x)))
    return result


def find_gaps(per_frame: Sequence[Sequence[FaceBox]]) -> list[tuple[int, int]]:
    """Return maximal half-open frame ranges with no face boxes."""
    gaps = []
    start: int | None = None
    for index, faces in enumerate(per_frame):
        if not faces and start is None:
            start = index
        elif faces and start is not None:
            gaps.append((start, index))
            start = None
    if start is not None:
        gaps.append((start, len(per_frame)))
    return gaps


def detect_faces(video: Path, *, every: int = 1, detector: FaceDetector | None = None) -> FacesData:
    """Measure all video frames or record why the system detector is unavailable."""
    with paths.operation():
        if type(every) is not int or every < 1:
            raise ValidationError("every", "must be an integer >= 1")
        source = paths.validate_input(video)
        info = ffmpeg.probe(source)
        if (
            not info.has_video
            or info.video_frames is None
            or info.video_frames < 1
            or info.fps is None
            or info.fps <= 0
            or not info.width
            or not info.height
        ):
            raise InputError(
                str(video), "face detection needs video with a known frame count and fps"
            )
        identity = VideoIdentity(
            Path(video).name,
            receipt.sha256_file(source),
            info.video_frames,
            info.fps,
            info.width,
            info.height,
        )
        warnings = (
            ("variable frame rate: frame numbers follow decode order, not time x fps",)
            if ffmpeg.variable_frame_rate(info.fps, info.nominal_fps)
            else ()
        )
        if detector is None:
            reason = unavailable_reason()
            if reason is not None:
                return FacesData(
                    "unmeasured",
                    reason,
                    identity,
                    {"name": DETECTOR_NAME},
                    {"every": every},
                    (),
                    (),
                    warnings,
                )
            detector = default_detector()
        run = detector.detect(source, every=every)
        if run.frames != identity.frames or (run.width, run.height) != (info.width, info.height):
            raise ProcessingError(
                "yap-face-boxes",
                0,
                f"face helper decoded {run.frames} frames at {run.width}x{run.height}; "
                f"probe reported {identity.frames} frames at {info.width}x{info.height}",
                code="face_frame_mismatch",
            )
        per_frame = interpolate(run.samples, run.frames)
        sampling = {
            "every": every,
            "sampled_frames": len(run.samples),
            "interpolated_frames": run.frames - len(run.samples),
        }
        return FacesData(
            "measured",
            None,
            identity,
            run.detector,
            sampling,
            tuple(per_frame),
            tuple(find_gaps(per_frame)),
            warnings,
            run.times,
            run.end_time,
        )


def write_faces(path: Path, data: FacesData) -> Path:
    """Atomically write a faces document using the engine's overwrite guard."""
    return paths.atomic_write_text(
        path, json.dumps(data.to_dict(), indent=2, allow_nan=False) + "\n"
    )
