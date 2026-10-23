"""Stream and validate the alpha from an installed HyperFrames renderer."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import DependencyError, InputError, ProcessingError
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte import timing
from yap_studio.matte.frames import gray_decode_command
from yap_studio.render import hyperframes as renderer

BACKEND_NAME = "hyperframes-remove-background"
MODEL_NOTICE = "The first HyperFrames matte downloads about 390 MB (model and runtime)."
_UNAVAILABLE = "the HyperFrames matte needs the renderer; run yap-studio doctor"


def unavailable_reason() -> str | None:
    """Require an existing renderer installation, without installing anything."""
    return None if renderer.installed() is not None else _UNAVAILABLE


def model_path() -> Path:
    """Resolve the model cache at call time, honoring the current home directory."""
    return Path.home() / ".cache/hyperframes/background-removal/models/u2net_human_seg.onnx"


def require_constant_rate(info: ffmpeg.MediaInfo, video: Path) -> None:
    """Reject inputs which the renderer would resample onto a different clock."""
    try:
        timing.require_constant_rate(video, info)
    except InputError:
        error = InputError(str(video), "the HyperFrames matte needs constant-rate video")
        error.code = "matte_variable_rate"
        error.suggested_action = "run it on the cut, which is constant rate"
        raise error from None


def _deadline_seconds(frames: int) -> float:
    return 120 + 1.5 * frames


def _run(install: renderer.RendererInstall, video: Path, output: Path, deadline: float) -> None:
    command = [
        str(install.executable),
        "remove-background",
        str(video),
        "-o",
        str(output),
        "--json",
    ]
    # Diagnostics go to disk so progress messages cannot grow working memory.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, **renderer.renderer_env()},
            )
        except OSError as exc:
            raise ProcessingError(
                " ".join(command), -1, str(exc), code="hyperframes_matte_output"
            ) from exc
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise ProcessingError(
                " ".join(command),
                -1,
                "HyperFrames matte timed out",
                code="hyperframes_matte_timeout",
            ) from None
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
        if child.returncode:
            stderr.seek(0, 2)
            stderr.seek(max(0, stderr.tell() - 64 * 1024))
            raise ProcessingError(
                " ".join(command),
                child.returncode,
                stderr.read().decode("utf-8", errors="replace"),
                code=f"hyperframes_matte_exit_{child.returncode}",
            )
        stdout.seek(0)
        # Progress may precede the single result line. framesProcessed is never
        # used: it reports the renderer's resampled count, not the plate's count.
        result = None
        for line in stdout:
            try:
                value = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if type(value) is dict and "ok" in value:
                result = value
        if result is None or result.get("ok") is not True or not output.is_file():
            raise ProcessingError(
                " ".join(command),
                0,
                "HyperFrames did not return ok: true and a WebM",
                code="hyperframes_matte_output",
            )


def _frames(
    video: Path,
    identity: VideoIdentity,
    deadline: float,
    *,
    alpha: bool = False,
    webm: bool = False,
) -> Iterator[np.ndarray]:
    command = gray_decode_command(video, alpha=alpha, webm=webm)
    frame = np.empty((identity.height, identity.width), dtype=np.uint8)
    target = memoryview(frame).cast("B")
    used = 0
    with closing(
        ffmpeg.run_stream(command, timeout_s=max(0, deadline - time.monotonic()))
    ) as chunks:
        for chunk in chunks:
            raw = memoryview(chunk)
            while raw:
                size = min(len(raw), len(target) - used)
                target[used : used + size] = raw[:size]
                used += size
                raw = raw[size:]
                if used == len(target):
                    yield frame
                    frame = np.empty_like(frame)
                    target = memoryview(frame).cast("B")
                    used = 0
    if used:
        raise ProcessingError("ffmpeg", 0, "partial decoded matte frame", code="matte_frame_count")


def _no_person(alpha: np.ndarray) -> bool:
    keyed = alpha >= 128
    height = keyed.shape[0]
    top = max(1, math.ceil(height * 0.02))
    bottom = max(1, math.ceil(height * 0.01))
    return bool(
        keyed[:top].any(axis=0).mean() > 0.25 or not keyed[-bottom:].any() or keyed.mean() > 0.85
    )


def _solid_samples(alpha: np.ndarray) -> np.ndarray:
    padded = np.pad(alpha, 4, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (9, 9))
    return windows[::4, ::4].min(axis=(-2, -1)) == 255


@dataclass
class MatteStream:
    """Bounded frame stream and the measurements collected as it is consumed."""

    install: renderer.RendererInstall
    deadline: float
    frames: Iterator[bytes]
    coverage: list[float] = field(default_factory=list)
    empty_frames: list[int] = field(default_factory=list)
    aligned: int = 0
    compared: int = 0

    @property
    def alignment(self) -> float:
        """A stream with no eligible comparisons has no evidence of a shift."""
        return self.aligned / self.compared if self.compared else 1.0


def temporal_median(frames: Iterator[bytes]) -> Iterator[bytes]:
    """Stream a three-frame pixel median with cloned first and last neighbours."""
    raw = next(frames, None)
    if raw is None:
        return
    previous = current = np.frombuffer(raw, dtype=np.uint8)
    while True:
        raw = next(frames, None)
        following = current if raw is None else np.frombuffer(raw, dtype=np.uint8)
        yield np.minimum(
            np.maximum(previous, current), np.maximum(np.minimum(previous, current), following)
        ).tobytes()
        if raw is None:
            break
        previous, current = current, following


def cleaned_frames(stream: MatteStream, *, median: bool) -> Iterator[bytes]:
    """Apply temporal cleanup to guarded masks, then restore every guarded zero."""
    frames = temporal_median(stream.frames) if median else stream.frames
    empty_index = 0
    for index, frame in enumerate(frames):
        # The guard records indices as frames are read, including median lookahead.
        if empty_index < len(stream.empty_frames) and stream.empty_frames[empty_index] == index:
            empty_index += 1
            yield bytes(len(frame))
        else:
            yield frame


def _guarded_frames(
    video: Path, webm: Path, identity: VideoIdentity, stream: MatteStream
) -> Iterator[bytes]:
    def count_error(detail: str) -> ProcessingError:
        return ProcessingError("ffmpeg", 0, detail, code="matte_frame_count")

    with ExitStack() as stack:
        plate = stack.enter_context(closing(_frames(video, identity, stream.deadline)))
        masks = stack.enter_context(
            closing(_frames(webm, identity, stream.deadline, alpha=True, webm=True))
        )
        colour = stack.enter_context(closing(_frames(webm, identity, stream.deadline, webm=True)))
        previous = None
        current = next(plate, None)
        following = next(plate, None)
        index = 0
        while current is not None:
            alpha = next(masks, None)
            luma = next(colour, None)
            if alpha is None or luma is None:
                raise count_error(
                    f"decoded matte ended at frame {index}; plate needs {identity.frames}"
                )
            if previous is not None and following is not None:
                solid = _solid_samples(alpha)
                if solid.any():
                    samples = luma[::4, ::4][solid].astype(np.int16)
                    differences = [
                        float(np.abs(samples - frame[::4, ::4][solid].astype(np.int16)).mean())
                        for frame in (previous, current, following)
                    ]
                    stream.compared += 1
                    stream.aligned += differences[1] <= min(differences) + 0.25
            stream.coverage.append(float((alpha >= 128).mean()))
            if _no_person(alpha):
                stream.empty_frames.append(index)
                yield bytes(identity.width * identity.height)
            else:
                yield alpha.tobytes()
            index += 1
            previous, current = current, following
            following = next(plate, None) if current is not None else None
        if (
            index != identity.frames
            or next(masks, None) is not None
            or next(colour, None) is not None
        ):
            raise count_error(f"decoded matte and plate must both contain {identity.frames} frames")
        if stream.alignment < 0.99:
            raise ProcessingError(
                "ffmpeg",
                0,
                f"matte alignment share {stream.alignment:.2%} is below 99%",
                code="matte_alignment",
            )


@contextmanager
def stream_matte(video: Path, folder: Path, identity: VideoIdentity) -> Iterator[MatteStream]:
    """Run only the existing CLI and clean up its WebM on every exit path."""
    install = renderer.installed()
    if install is None:
        raise DependencyError(
            _UNAVAILABLE, error_type="dependency_error", code="hyperframes_matte_unavailable"
        )
    deadline = time.monotonic() + _deadline_seconds(identity.frames)
    with tempfile.TemporaryDirectory(prefix=".yap-hyperframes-matte-", dir=folder) as temporary:
        webm = Path(temporary) / "person.webm"
        _run(install, video, webm, deadline)
        info = ffmpeg.probe(webm, count_frames=False)
        if (info.width, info.height) != (identity.width, identity.height):
            raise ProcessingError(
                "ffprobe", 0, "matte dimensions differ from the plate", code="matte_frame_count"
            )
        stream = MatteStream(install, deadline, iter(()))
        stream.frames = _guarded_frames(video, webm, identity, stream)
        with closing(stream.frames):
            yield stream
