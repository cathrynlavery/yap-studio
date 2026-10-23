"""Stream person masks through ffmpeg and publish validated matte artifacts."""

from __future__ import annotations

import json
import math
import os
import platform
import selectors
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import DependencyError, InputError, ProcessingError, ValidationError
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte import chroma, hyperframes, vision
from yap_studio.matte.model import Cleanup, MatteData

BACKEND_NAME = "vision-person-segmentation"
_DEFAULT_CLEANUP = Cleanup()


def _deadline_seconds(frames: int) -> float:
    return 120 + 0.5 * frames


def _stderr(stream: BinaryIO) -> str:
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - 64 * 1024))
    return stream.read().decode("utf-8", errors="replace").strip()


def _output_error(command: str, detail: str) -> ProcessingError:
    return ProcessingError(
        command, 0, f"invalid person matte helper output: {detail}", code="person_matte_output"
    )


def _parse_records(
    records: Path, video: VideoIdentity, quality: str, command: str
) -> tuple[dict[str, Any], tuple[float, ...], float, tuple[float, ...]]:
    header: dict[str, Any] | None = None
    total: int | None = None
    end: float | None = None
    times: list[float] = []
    coverage: list[float] = []

    def integer(record: dict[str, Any], key: str, minimum: int) -> int:
        value = record.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        return value

    def number(record: dict[str, Any], key: str) -> float:
        value = record.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        return float(value)

    try:
        with records.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                if type(record) is not dict:
                    raise ValueError(f"line {line_number} must be a JSON object")
                kind = record.get("type")
                if total is not None:
                    raise ValueError("record after end")
                if header is None:
                    if kind != "header":
                        raise ValueError("missing first header record")
                    for key in ("width", "height", "mask_width", "mask_height", "revision"):
                        integer(record, key, 1)
                    if (record["width"], record["height"]) != (video.width, video.height):
                        raise ValueError("header dimensions differ from the input video")
                    if record.get("quality") != quality:
                        raise ValueError("header quality differs from the requested quality")
                    header = record
                elif kind == "frame":
                    if integer(record, "frame", 0) != len(times):
                        raise ValueError(
                            "frames must appear in order starting at zero without gaps"
                        )
                    pts = number(record, "pts")
                    if times and pts < times[-1]:
                        raise ValueError("pts must be non-decreasing")
                    fraction = number(record, "coverage")
                    if not 0 <= fraction <= 1:
                        raise ValueError("coverage must be between 0 and 1")
                    times.append(pts)
                    coverage.append(fraction)
                elif kind == "end":
                    total = integer(record, "frames", 1)
                    end = number(record, "end_pts")
                else:
                    raise ValueError(f"unexpected record type: {kind!r}")
        if header is None or total is None or end is None:
            raise ValueError("missing header or end record")
        if total != video.frames or len(times) != total:
            raise ValueError("record count must equal the input video frame count")
        if end < times[-1]:
            raise ValueError("end_pts must be at or after the last pts")
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise _output_error(command, str(exc)) from exc
    return header, tuple(times), end, tuple(coverage)


def _encoder_command(
    temporary: Path, video: VideoIdentity, cleanup: Cleanup, *, temporal: bool = True,
    refined: bool = False,
) -> list[str]:
    chain = "null" if refined else cleanup.filter_chain(temporal=temporal)
    return ffmpeg._safe_command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{video.width}x{video.height}",
            "-framerate",
            f"{video.fps.numerator}/{video.fps.denominator}",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-vf",
            chain,
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "gray",
            *(["-fflags", "+bitexact", "-flags:v", "+bitexact"] if refined else []),
            str(temporary),
        ]
    )


def _encode(
    stream: vision.HelperStream,
    temporary: Path,
    video: VideoIdentity,
    cleanup: Cleanup,
    deadline: float,
) -> None:
    command = _encoder_command(temporary, video, cleanup)
    helper = stream.process
    assert helper.stdout is not None
    with tempfile.TemporaryFile() as stderr:
        try:
            encoder = subprocess.Popen(
                command, stdin=helper.stdout, stdout=subprocess.DEVNULL, stderr=stderr
            )
        except OSError as exc:
            raise ProcessingError(" ".join(command), -1, str(exc)) from exc
        helper.stdout.close()
        # Wait concurrently for either child, so a failed encoder cannot leave the
        # producer blocked on a full pipe until the deadline.
        with ThreadPoolExecutor(max_workers=2) as executor:
            children = {executor.submit(child.wait): child for child in (helper, encoder)}
            try:
                while children:
                    done, _ = wait(
                        children,
                        timeout=max(0, deadline - time.monotonic()),
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        raise ProcessingError(
                            " ".join(stream.command),
                            -1,
                            "person matte pipeline timed out",
                            code="person_matte_timeout",
                        )
                    # Prefer the helper's reason when both children fail together.
                    if helper.poll() not in (None, 0):
                        raise ProcessingError(
                            " ".join(stream.command),
                            helper.returncode,
                            _stderr(stream.stderr),
                            code=f"person_matte_exit_{helper.returncode}",
                        )
                    for future in done:
                        child = children.pop(future)
                        if future.result() != 0:
                            raise ProcessingError(
                                " ".join(command), child.returncode, _stderr(stderr)
                            )
            finally:
                for child in (helper, encoder):
                    if child.poll() is None:
                        child.kill()
                for child in (helper, encoder):
                    child.wait()


def _encode_frames(
    frames: Iterator[bytes],
    temporary: Path,
    video: VideoIdentity,
    cleanup: Cleanup,
    deadline: float,
) -> None:
    """Feed temporally cleaned, guarded frames through spatial cleanup and FFV1."""
    command = _encoder_command(temporary, video, cleanup, temporal=False)
    with tempfile.TemporaryFile() as stderr:
        try:
            encoder = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr, bufsize=0
            )
        except OSError as exc:
            raise ProcessingError(" ".join(command), -1, str(exc)) from exc
        assert encoder.stdin is not None
        try:
            os.set_blocking(encoder.stdin.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(encoder.stdin, selectors.EVENT_WRITE)
                for frame in frames:
                    raw = memoryview(frame)
                    while raw:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise subprocess.TimeoutExpired(command, 0)
                        try:
                            size = os.write(encoder.stdin.fileno(), raw)
                        except BlockingIOError:
                            continue
                        raw = raw[size:]
            encoder.stdin.close()
            encoder.wait(timeout=max(0, deadline - time.monotonic()))
            if encoder.returncode:
                raise ProcessingError(" ".join(command), encoder.returncode, _stderr(stderr))
        except BrokenPipeError:
            encoder.wait(timeout=max(0, deadline - time.monotonic()))
            raise ProcessingError(" ".join(command), encoder.returncode, _stderr(stderr)) from None
        except subprocess.TimeoutExpired:
            raise ProcessingError(
                " ".join(command),
                -1,
                "HyperFrames matte pipeline timed out",
                code="hyperframes_matte_timeout",
            ) from None
        finally:
            encoder.stdin.close()
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()


def _model_notice(message: str) -> None:
    print(message, file=sys.stderr)


def make_matte(
    video: Path,
    out_dir: Path,
    *,
    backend: str = "auto",
    cleanup: Cleanup = _DEFAULT_CLEANUP,
    quality: str = "accurate",
    notice: Callable[[str], None] | None = _model_notice,
    requested: str = "auto",
    joins: Sequence[int] = (),
) -> MatteData:
    """Make a lossless gray matte with a screen key or person detection."""
    if backend not in ("auto", "vision", "hyperframes", "chroma"):
        raise ValidationError("backend", "expected auto, vision, hyperframes or chroma")
    if quality not in ("accurate", "balanced"):
        raise ValidationError("quality", "expected accurate or balanced")
    if not isinstance(cleanup, Cleanup):
        raise ValidationError("cleanup", "expected Cleanup")
    with paths.operation():
        paths.protect_input(video)
        source = paths.validate_input(video)
        info = ffmpeg.probe(source, count_frames=True)
        if (
            not info.has_video
            or not info.video_frames
            or not info.width
            or not info.height
            or info.fps is None
            or info.fps <= 0
        ):
            raise InputError(
                str(video), "person matte needs video with a known frame count and fps"
            )
        identity = VideoIdentity(
            video.name,
            receipt.sha256_file(source),
            info.video_frames,
            info.fps,
            info.width,
            info.height,
        )
        joins = tuple(joins)
        if (any(type(j) is not int or not 0 < j < identity.frames for j in joins)
                or any(a >= b for a, b in zip(joins, joins[1:], strict=False))):
            raise ValidationError("joins", "expected sorted, unique interior frame indices")
        if backend in ("auto", "chroma"):
            screen = chroma.make_screen_matte(source, out_dir, backend=backend, cleanup=cleanup,
                                             info=info, identity=identity)
            if screen is not None:
                return screen
        warnings = (
            ("variable frame rate: frame numbers follow decode order, not time x fps",)
            if info.variable_frame_rate
            else ()
        )
        reason = None
        selected = backend
        if backend == "auto":
            reason = vision.unavailable_reason()
            selected = "vision"
            if reason is not None:
                fallback_reason = hyperframes.unavailable_reason()
                selected = "hyperframes"
                reason = f"{reason}; {fallback_reason}" if fallback_reason is not None else None
        elif backend == "vision":
            reason = vision.unavailable_reason()
        else:
            reason = hyperframes.unavailable_reason()
        if reason is not None and backend != "auto":
            raise DependencyError(
                reason,
                error_type="dependency_error",
                code=(
                    "person_matte_unavailable"
                    if backend == "vision"
                    else "hyperframes_matte_unavailable"
                ),
                suggested_action=(
                    "Use macOS with the Xcode command line tools installed."
                    if backend == "vision"
                    else "Run yap-studio doctor."
                ),
            )
        if reason is None and selected == "hyperframes":
            hyperframes.require_constant_rate(info, source)
            cleanup = replace(cleanup, refine=False)
        folder = paths.validate_output_dir(out_dir)
        matte_path = paths.validate_output(folder / "matte.mkv")
        document = paths.validate_output(folder / "matte.json")
        if reason is not None:
            data = MatteData(
                "unavailable", reason, identity, {}, cleanup, None, None, None, None, warnings
            )
            paths.atomic_write_text(document, json.dumps(data.to_dict(), indent=2) + "\n")
            return data
        published = False
        try:
            with paths.atomic_output(matte_path) as temporary:
                empty_frames: tuple[int, ...] = ()
                if selected == "vision":
                    with vision.stream_helper(source, quality=quality) as stream:
                        deadline = time.monotonic() + _deadline_seconds(identity.frames)
                        if cleanup.refine:
                            from yap_studio.matte.frames import encode_vision

                            encode_vision(stream, source, temporary, identity, cleanup, deadline,
                                          joins)
                        else:
                            _encode(stream, temporary, identity, cleanup, deadline)
                        header, times, end, coverage = _parse_records(
                            stream.records, identity, quality, " ".join(stream.command)
                        )
                    metadata = {
                        "name": BACKEND_NAME,
                        "quality": quality,
                        "revision": header["revision"],
                        "os": platform.mac_ver()[0] or platform.platform(),
                        "helper_sha256": receipt.sha256_file(vision.helper_source()),
                        "mask_width": header["mask_width"],
                        "mask_height": header["mask_height"],
                    }
                else:
                    if notice is not None and not hyperframes.model_path().exists():
                        notice(hyperframes.MODEL_NOTICE)
                    with hyperframes.stream_matte(source, folder, identity) as masks:
                        _encode_frames(
                            hyperframes.cleaned_frames(masks, median=cleanup.median),
                            temporary, identity, cleanup, masks.deadline,
                        )
                        coverage = tuple(masks.coverage)
                        empty_frames = tuple(masks.empty_frames)
                        metadata = {
                            "name": hyperframes.BACKEND_NAME,
                            "version": masks.install.version,
                            "alignment": masks.alignment,
                        }
                    start = info.video_start_s or 0.0
                    times = tuple(start + float(i / identity.fps) for i in range(identity.frames))
                    end = start + float(identity.frames / identity.fps)
                    if empty_frames:
                        examples = ", ".join(f"{times[i]:.3f} s" for i in empty_frames[:5])
                        warnings += (
                            f"no person: {len(empty_frames)} frame(s) left empty at {examples}",
                        )
                matte_info = ffmpeg.probe(temporary, count_frames=True)
                if matte_info.video_frames != identity.frames or (
                    matte_info.width,
                    matte_info.height,
                ) != (identity.width, identity.height):
                    raise ProcessingError(
                        "ffprobe",
                        0,
                        "matte frame count or dimensions differ from the input video",
                        code="matte_frame_count",
                    )
                if requested != "auto":
                    metadata["requested"] = requested
                data = MatteData(
                    "measured",
                    None,
                    identity,
                    metadata,
                    cleanup,
                    {
                        "file": "matte.mkv",
                        "sha256": receipt.sha256_file(temporary),
                        "codec": "ffv1",
                        "pix_fmt": "gray",
                    },
                    times,
                    end,
                    coverage,
                    warnings,
                    empty_frames,
                    joins,
                )
                payload = json.dumps(data.to_dict(), indent=2, allow_nan=False) + "\n"
                paths.validate_output(document)
            published = True
            paths.atomic_write_text(document, payload)
        except BaseException:
            if published:
                matte_path.unlink(missing_ok=True)
            raise
        return data
