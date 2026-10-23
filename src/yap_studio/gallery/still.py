"""Pick an opening frame and words for the style gallery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import string
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from yap_studio import ffmpeg, paths, tools, transcript
from yap_studio.cache import cache_dir, cache_root
from yap_studio.captions import CaptionWord, caption_layout_settings
from yap_studio.errors import DependencyError, ProcessingError, ValidationError
from yap_studio.stages.captions import caption_band_geometry
from yap_studio.style import resolve
from yap_studio.transcript import shared
from yap_studio.transcript.backends import ENGINES

TAKE_SUFFIXES = frozenset({".mov", ".mp4", ".m4v"})
EYES_OPEN = 0.2
EYES_RELATIVE = 0.8
MIN_WORDS = 5
MAX_WORDS = 40
OPENING_SECONDS = 40.0
OPENERS = frozenset({
    "all right", "alright", "okay", "ok", "so", "hi", "hey", "hello",
    "um", "uh", "umm", "uhh", "well", "yeah",
})
_SCHEMA = "yap-studio/gallery-still@1"
_PREVIEW_SCHEMA = "yap-studio/gallery-preview@1"
_OUTPUT_SCHEMAS = {"still.json": _SCHEMA, "preview.json": _PREVIEW_SCHEMA}
_OUTPUT_NAMES = ("still.jpg", "still.svg", "still.json", "preview.json")


def _check_output_folder(out_dir: Path) -> None:
    """Only replace gallery files when a gallery record identifies the folder."""
    if not any((out_dir / name).exists() or (out_dir / name).is_symlink()
               for name in _OUTPUT_NAMES):
        return
    for name, schema in _OUTPUT_SCHEMAS.items():
        record = out_dir / name
        if not record.is_file() or record.is_symlink():
            continue
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, RecursionError):
            continue
        if isinstance(data, dict) and data.get("schema") == schema:
            return
    raise ValidationError(
        "--out", "output folder has files this command did not write; choose an empty folder"
    )


def newest_take(inbox: Path) -> Path | None:
    """Return the newest visible video directly inside an inbox."""
    if not inbox.is_dir():
        return None
    files = (
        child for child in inbox.iterdir()
        if child.is_file() and not child.name.startswith(".")
        and child.suffix.lower() in TAKE_SUFFIXES
    )
    return max(files, key=lambda child: (child.stat().st_mtime_ns, child.name), default=None)


def sample_still() -> Path:
    """Return the bundled sample presenter used when an inbox is empty."""
    return Path(__file__).with_name("sample-still.jpg")


def first_sentence(
    words: Sequence[transcript.Word], *, max_words: int = MAX_WORDS,
    pause_s: float = 1.5, window_end_s: float | None = None,
) -> list[CaptionWord]:
    """Choose a complete useful sentence and move its first word to 0.3 seconds."""
    if not words or max_words <= 0:
        return []
    sentences: list[tuple[list[transcript.Word], bool]] = []
    current: list[transcript.Word] = []
    limit = min(max_words, MAX_WORDS)
    for word in words:
        if current and word.start - current[-1].end > pause_s:
            sentences.append((current, True))
            current = []
        current.append(word)
        if word.text.rstrip("\"')]}’”»").endswith((".", "!", "?")) or len(current) == limit:
            sentences.append((current, True))
            current = []
    if current:
        complete = window_end_s is None or current[-1].end < window_end_s - 1.0
        sentences.append((current, complete))

    def has_opener(sentence: list[transcript.Word]) -> bool:
        opening = [word.text.strip(string.punctuation + "“”‘’«»").casefold()
                   for word in sentence[:2]]
        return opening[0] in OPENERS or " ".join(opening) in OPENERS

    selected = next(
        (sentence for sentence, complete in sentences
         if complete and len(sentence) >= MIN_WORDS and not has_opener(sentence)),
        None,
    )
    if selected is None:
        selected = max((sentence for sentence, _complete in sentences), key=len)
    shift = 0.3 - selected[0].start
    return [
        CaptionWord(f"w{index}", word.text, round(word.start + shift, 3),
                    round(word.end + shift, 3))
        for index, word in enumerate(selected)
    ]


def transcribe_opening(
    take: Path, out_dir: Path, *, seconds: float = OPENING_SECONDS,
    backend: transcript.Backend | None = None,
) -> transcript.Transcript | None:
    """Transcribe a temporary mono opening WAV with the available local engine."""
    engine = backend if backend is not None else transcript.resolve_backend("auto")
    if engine is None:
        return None
    if not ffmpeg.probe(take).has_audio:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".opening-", suffix=".wav", dir=out_dir)
    os.close(descriptor)
    wav = Path(name)
    try:
        ffmpeg.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(take), "-t", str(seconds),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            timeout_s=300,
        )
        return engine.transcribe_many([wav])[wav]
    finally:
        wav.unlink(missing_ok=True)


@dataclass(frozen=True)
class FaceLandmarks:
    """A detected face and its eye shape in display pixels."""

    x: float
    y: float
    w: float
    h: float
    left_eye: float | None
    right_eye: float | None
    yaw: float | None
    roll: float | None
    confidence: float


class LandmarkDetector(Protocol):
    """Measure faces in display-upright still images."""

    def measure(self, images: Sequence[Path]) -> list[list[FaceLandmarks]]:
        """Return one face list per image in argument order."""
        ...


def _helper_source() -> Path:
    return Path(__file__).with_name("face_landmarks.swift")


def _helper_path() -> Path:
    digest = hashlib.sha256(_helper_source().read_bytes()).hexdigest()
    return cache_root() / "helpers" / f"yap-face-landmarks-{digest[:16]}"


def landmarks_unavailable_reason() -> str | None:
    """Explain why the gallery landmark helper cannot run on this computer."""
    if sys.platform != "darwin":
        return (
            "face detection uses the macOS Vision framework; "
            f"this computer runs {platform.system()}"
        )
    cached = _helper_path()
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


def _helper_executable() -> Path:
    executable = _helper_path()
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    folder = cache_dir("helpers")
    descriptor, name = tempfile.mkstemp(prefix=".yap-face-landmarks-", dir=folder)
    os.close(descriptor)
    temporary = Path(name)
    try:
        tools.run_tool(
            "xcrun", ["swiftc", "-O", "-o", str(temporary), str(_helper_source())],
            timeout_s=600,
        )
        os.replace(temporary, executable)
    finally:
        temporary.unlink(missing_ok=True)
    return executable


def _number(record: dict[str, Any], key: str, minimum: float, maximum: float) -> float:
    value = record.get(key)
    if (
        type(value) not in (int, float) or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{key} must be a finite number from {minimum} to {maximum}")
    return float(value)


def _optional_angle(record: dict[str, Any], key: str) -> float | None:
    if key not in record:
        raise ValueError(f"missing {key}")
    if record[key] is None:
        return None
    return _number(record, key, -2 * math.pi, 2 * math.pi)


def _optional_eye(record: dict[str, Any], key: str) -> float | None:
    if key not in record:
        raise ValueError(f"missing {key}")
    if record[key] is None:
        return None
    return _number(record, key, 0, 10)


def _parse_output(stdout: str, images: Sequence[Path]) -> list[list[FaceLandmarks]]:
    """Reject malformed or reordered helper records before using any face data."""
    try:
        lines = stdout.splitlines()
        if len(lines) != len(images):
            raise ValueError(f"expected {len(images)} image records, got {len(lines)}")
        output = []
        for line, image in zip(lines, images, strict=True):
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("image") != str(image):
                raise ValueError("image record is missing or out of order")
            width = record.get("width")
            height = record.get("height")
            if type(width) is not int or width < 1 or type(height) is not int or height < 1:
                raise ValueError("image dimensions must be positive integers")
            faces = record.get("faces")
            if not isinstance(faces, list):
                raise ValueError("faces must be an array")
            measured = []
            for face in faces:
                if not isinstance(face, dict):
                    raise ValueError("face must be an object")
                x = _number(face, "x", 0, width)
                y = _number(face, "y", 0, height)
                w = _number(face, "w", 0, width - x)
                h = _number(face, "h", 0, height - y)
                if w == 0 or h == 0:
                    raise ValueError("face box must have positive size")
                measured.append(FaceLandmarks(
                    x, y, w, h,
                    _optional_eye(face, "left_eye"), _optional_eye(face, "right_eye"),
                    _optional_angle(face, "yaw"), _optional_angle(face, "roll"),
                    _number(face, "confidence", 0, 1),
                ))
            output.append(measured)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ProcessingError(
            "face-landmarks", 0, f"invalid face helper output: {exc}",
            code="face_landmark_output",
        ) from exc
    return output


class VisionLandmarkDetector:
    """Use the cached macOS Vision helper on still images."""

    def measure(self, images: Sequence[Path]) -> list[list[FaceLandmarks]]:
        """Compile the helper when needed and validate each JSON line."""
        if not images:
            return []
        helper = _helper_executable()
        arguments = [str(image) for image in images]
        result = tools.run_tool(
            "face-landmarks", arguments, timeout_s=120 + len(images), executable=helper
        )
        return _parse_output(result.stdout, images)


def default_landmark_detector() -> LandmarkDetector | None:
    """Return Vision when its macOS tools are available."""
    return VisionLandmarkDetector() if landmarks_unavailable_reason() is None else None


@dataclass(frozen=True)
class StillPick:
    """The written gallery still and the measures behind its choice."""

    path: Path
    source: str
    take: Path | None
    time_s: float | None
    scores: dict[str, Any]
    reasons: tuple[str, ...]


def _candidate_times(duration: float, seconds: float, step_s: float) -> list[float]:
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    limit = min(duration, seconds) - 0.25
    if limit < 1:
        return [max(0.0, min(duration, seconds) / 2)]
    count = int(math.floor((limit - 1) / step_s + 1e-9)) + 1
    return [round(1 + index * step_s, 6) for index in range(count)]


def _sharpness(frame: np.ndarray, face: FaceLandmarks | None) -> float:
    if face is not None:
        x0 = max(0, int(face.x))
        y0 = max(0, int(face.y))
        x1 = min(frame.shape[1], int(math.ceil(face.x + face.w)))
        y1 = min(frame.shape[0], int(math.ceil(face.y + face.h)))
        frame = frame[y0:y1, x0:x1]
    if min(frame.shape) < 3:
        return 0.0
    grey = frame.astype(np.int16)
    laplacian = (
        4 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1]
        - grey[1:-1, :-2] - grey[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _default_band() -> tuple[int, int]:
    style = resolve("preset:clean_minimal")
    geometry = caption_band_geometry(
        caption_layout_settings(style.data["captions"], style.data["brand"]),
        safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
    )
    return geometry.y, geometry.y + geometry.height


def pick_still(
    take: Path, out_dir: Path, *, words: Sequence[transcript.Word] = (),
    detector: LandmarkDetector | None = None, seconds: float = 20.0,
    step_s: float = 0.5, band: tuple[int, int] | None = None,
) -> StillPick:
    """Score opening frames for a clear, open-eyed face and save the best JPEG."""
    _check_output_folder(out_dir)
    info = ffmpeg.probe(take)
    duration = info.video_duration_s or info.duration_s or 0.0
    if duration <= 0:
        raise ProcessingError(str(take), 0, "video has no measurable duration")
    times = _candidate_times(duration, seconds, step_s)
    first_seek = 0.0 if info.video_frames == 1 else times[0]
    caption_top, caption_bottom = band if band is not None else _default_band()
    out_dir.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    selected_detector = detector if detector is not None else default_landmark_detector()
    if selected_detector is None:
        reasons.append("Face landmarks are unavailable; picked by sharpness and speech timing.")
    with tempfile.TemporaryDirectory(prefix="still-frames-", dir=out_dir) as temporary:
        folder = Path(temporary)
        small_pattern = folder / "frame-%03d.png"
        full_pattern = folder / "frame-%03d.jpg"
        sampling = (
            f"fps=fps={1 / step_s}:start_time=0," if len(times) > 1 else ""
        )
        graph = (
            f"[0:v]{sampling}split=2[small][full];"
            "[small]scale=540:960:force_original_aspect_ratio=increase,"
            "crop=540:960[score];"
            "[full]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920[still]"
        )
        ffmpeg.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", str(first_seek), "-i", str(take),
             "-filter_complex", graph, "-map", "[score]", "-frames:v", str(len(times)),
             str(small_pattern), "-map", "[still]", "-frames:v", str(len(times)),
             "-q:v", "2", str(full_pattern)],
            timeout_s=300,
        )
        images = [folder / f"frame-{index:03d}.png" for index in range(1, len(times) + 1)]
        full_images = [folder / f"frame-{index:03d}.jpg" for index in range(1, len(times) + 1)]
        if not all(image.is_file() for image in [*images, *full_images]):
            raise ProcessingError(str(take), 0, "could not extract all candidate frames")
        faces_by_frame = (
            selected_detector.measure(images) if selected_detector else [[] for _ in images]
        )
        if len(faces_by_frame) != len(images):
            raise ProcessingError("face-landmarks", 0, "detector returned the wrong frame count")
        largest_faces = [
            max(faces, key=lambda found: found.w * found.h, default=None)
            for faces in faces_by_frame
        ]
        eye_values = [
            min(face.left_eye, face.right_eye)
            for face in largest_faces
            if face is not None and face.left_eye is not None and face.right_eye is not None
        ]
        eyes_open_threshold = max(
            EYES_OPEN, EYES_RELATIVE * statistics.median(eye_values)
        ) if eye_values else EYES_OPEN
        raw = ffmpeg.run(
            ["ffmpeg", "-v", "error", "-i", str(small_pattern), "-pix_fmt", "gray",
             "-f", "rawvideo", "-"],
            timeout_s=300, binary=True,
        ).stdout
        frame_bytes = 540 * 960
        if len(raw) != len(times) * frame_bytes:
            raise ProcessingError("ffmpeg", 0, "could not read candidate pixels")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(len(times), 960, 540)
        candidates: list[dict[str, Any]] = []
        for index, time in enumerate(times):
            face = largest_faces[index]
            eyes_open = (
                face is not None and face.left_eye is not None and face.right_eye is not None
                and face.left_eye >= eyes_open_threshold
                and face.right_eye >= eyes_open_threshold
            )
            band_clear = face is not None and (
                (face.y + face.h) * 2 <= caption_top or face.y * 2 >= caption_bottom
            )
            speaking = any(word.start <= time <= word.end for word in words)
            sharpness = _sharpness(pixels[index], face)
            candidates.append({
                "time_s": round(time, 3), "face": asdict(face) if face else None,
                "faces": [asdict(found) for found in faces_by_frame[index]],
                "sharpness": sharpness, "eyes_open": eyes_open,
                "band_clear": band_clear, "mid_sentence": speaking,
            })
        best_sharpness = max(candidate["sharpness"] for candidate in candidates)
        for candidate in candidates:
            yaw = candidate["face"]["yaw"] if candidate["face"] else None
            candidate["score"] = (
                candidate["sharpness"] / best_sharpness if best_sharpness > 0 else 0
            ) + (0.5 if candidate["mid_sentence"] else 0) + (
                0.25 * (1 - min(1, abs(yaw) / 0.5)) if yaw is not None else 0
            )
        sharpest_only = False
        if selected_detector is None:
            eligible = candidates
        else:
            eligible = [candidate for candidate in candidates if candidate["face"]
                        and candidate["eyes_open"] and candidate["band_clear"]]
            if not eligible:
                reasons.append("No open-eyed face clear of the caption band; allowed band overlap.")
                eligible = [candidate for candidate in candidates if candidate["face"]
                            and candidate["eyes_open"]]
            if not eligible:
                reasons.append("No face with open eyes; allowed closed or unknown eyes.")
                eligible = [candidate for candidate in candidates if candidate["face"]]
            if not eligible:
                reasons.append("No face in the opening; picked the sharpest frame.")
                eligible = candidates
                sharpest_only = True
        chosen = max(
            eligible, key=lambda candidate: candidate["sharpness" if sharpest_only else "score"]
        )
        path = out_dir / "still.jpg"
        os.replace(full_images[candidates.index(chosen)], path)
    result = StillPick(path, "take", take, chosen["time_s"], chosen, tuple(reasons))
    record = {
        "schema": _SCHEMA, "source": result.source, "take": str(take),
        "time_s": result.time_s, "scores": result.scores,
        "reasons": list(result.reasons), "candidates": candidates,
        "eyes_open_threshold": eyes_open_threshold,
    }
    paths.atomic_write_text(
        out_dir / "still.json", json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )
    return result


def prepare_preview(
    inbox: Path, out_dir: Path, *, detector: LandmarkDetector | None = None,
    backend: transcript.Backend | None = None, transcribe: bool = True,
) -> dict[str, Any]:
    """Prepare a still and first sentence from the newest take, or the sample presenter."""
    _check_output_folder(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    take = newest_take(inbox)
    if take is None:
        path = out_dir / "still.jpg"
        descriptor, name = tempfile.mkstemp(prefix=".still-", suffix=".jpg", dir=out_dir)
        os.close(descriptor)
        temporary = Path(name)
        try:
            shutil.copyfile(sample_still(), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        result = {
            "schema": _PREVIEW_SCHEMA, "source": "sample", "still": str(path),
            "sentence": None, "words": [], "transcript": None,
        }
        paths.atomic_write_text(
            out_dir / "preview.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        return result
    return prepare_take(take, out_dir, detector=detector, backend=backend,
                        transcribe=transcribe)


def prepare_take(
    take: Path, out_dir: Path, *, detector: LandmarkDetector | None = None,
    backend: transcript.Backend | None = None, transcribe: bool = True,
) -> dict[str, Any]:
    """Prepare one explicit take for the gallery or command line."""
    _check_output_folder(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    take_stat = take.stat()
    opening = None
    source = None
    if transcribe:
        engine = backend if backend is not None else transcript.resolve_backend("auto")
        if engine is not None:
            if engine.name in ENGINES:
                info = ffmpeg.probe(take)
                duration = info.audio_duration_s or info.duration_s
                if info.has_audio:
                    if duration is not None and duration <= OPENING_SECONDS:
                        request = shared.Request(shared.take_digest(take), 0, take)
                        opening, _hit = shared.transcribe(engine, [request])[take]
                    else:
                        digest = shared.known_digest(take)
                        if digest is not None:
                            opening = shared.lookup(engine, shared.Request(digest, 0, take))
                    if opening is not None:
                        source = "take"
                    else:
                        opening = transcribe_opening(take, out_dir, backend=engine)
            else:
                opening = transcribe_opening(take, out_dir, backend=engine)
    if opening is not None and source is None:
        source = "opening"
    words = opening.words if opening is not None else []
    window_end_s = opening.duration_s if opening is not None else None
    if source == "take":
        window_end_s = min(
            OPENING_SECONDS, info.audio_duration_s or info.duration_s or OPENING_SECONDS,
        )
        words = [word for word in words if word.start < window_end_s]
    elif opening is not None and window_end_s is None:
        info = ffmpeg.probe(take)
        window_end_s = min(
            OPENING_SECONDS, info.audio_duration_s or info.duration_s or OPENING_SECONDS,
        )
    chosen = pick_still(take, out_dir, words=words, detector=detector)
    sentence = first_sentence(words, window_end_s=window_end_s)
    result = {
        "schema": _PREVIEW_SCHEMA, "source": "take", "take": str(take),
        "take_size": take_stat.st_size, "take_mtime_ns": take_stat.st_mtime_ns,
        "still": str(chosen.path),
        "time_s": chosen.time_s,
        "transcript": source,
        # Detection uses the 540x960 thumbnail; the selected still is 1080x1920.
        "face_boxes": [{key: found[key] * 2 for key in ("x", "y", "w", "h")}
                       for found in chosen.scores["faces"]],
        "sentence": " ".join(word.text for word in sentence) or None,
        "words": [{"text": word.text, "start": word.start, "end": word.end}
                  for word in sentence],
        "reasons": list(chosen.reasons),
    }
    paths.atomic_write_text(
        out_dir / "preview.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result
