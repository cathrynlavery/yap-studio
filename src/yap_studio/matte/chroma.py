"""Detect screen borders, stream ratio keys and clean restored edge colours."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import ProcessingError, ValidationError, YapError
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte.model import Cleanup, MatteData, load_matte

if TYPE_CHECKING:
    from yap_studio.matte.project import _CachedMatte
    from yap_studio.project.status import ProjectContext


class _Graph(Protocol):
    filters: list[str]

    def filter(self, source: str, expression: str) -> str: ...


class _FilterGraph:
    """Append shared despill filters to a caller's existing filter list."""

    def __init__(self, filters: list[str]) -> None:
        self.filters = filters

    def filter(self, source: str, expression: str) -> str:
        label = f"chroma_p{len(self.filters)}"
        self.filters.append(f"[{source}]{expression}[{label}]")
        return label

# Eight samples cover changes in pose and light without retaining the take.
SAMPLE_FRAMES = 8
# Small portrait samples keep border detection cheap and repeatable.
SAMPLE_SIZE = (108, 192)
# Side borders avoid the centre where the speaker usually stands.
SIDE_RING = 0.10
# The top catches a screen above the head; the body often fills the bottom.
TOP_RING = 0.06
# A clear channel lead rejects grey rooms and pale green clothing.
SCREEN_EXCESS = 30
# Half the ring must agree before auto replaces person detection.
BORDER_SHARE = 0.5
# Nearby RGB values allow mild noise and light falloff on one screen.
KEY_DISTANCE = 60
# A divisor of at least 40 stops dark hair noise from reading as screen.
CHANNEL_FLOOR = 40


def _rgb_frames(
    video: Path, width: int, height: int, deadline: float, *, filters: str = "",
    limit: int | None = None, seek: float | None = None,
) -> Iterator[np.ndarray]:
    command = ["ffmpeg", "-v", "error"]
    if seek is not None:
        command += ["-ss", str(seek)]
    command += ["-i", str(video), "-map", "0:v:0", "-an"]
    if filters:
        command += ["-vf", filters]
    if limit is not None:
        command += ["-frames:v", str(limit)]
    command += ["-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    target = memoryview(frame).cast("B")
    used = 0
    with closing(ffmpeg.run_stream(
        command, timeout_s=max(0, deadline - time.monotonic()),
    )) as chunks:
        for chunk in chunks:
            raw = memoryview(chunk)
            while raw:
                size = min(len(raw), len(target) - used)
                target[used:used + size] = raw[:size]
                used += size
                raw = raw[size:]
                if used == len(target):
                    yield frame
                    frame = np.empty_like(frame)
                    target = memoryview(frame).cast("B")
                    used = 0
    if used:
        raise ProcessingError("ffmpeg", 0, "partial RGB frame", code="matte_frame_count")


def _detect(video: Path, info: ffmpeg.MediaInfo, *, forced: bool = False) -> dict | None:
    if not info.video_frames or info.fps is None or info.fps <= 0:
        return None
    width, height = SAMPLE_SIZE
    indices = np.unique(np.linspace(0, info.video_frames - 1, SAMPLE_FRAMES, dtype=int))
    filters = f"scale={width}:{height}:flags=area"
    ring = np.zeros((height, width), dtype=bool)
    side = math.ceil(width * SIDE_RING)
    ring[:, :side] = ring[:, -side:] = True
    ring[:math.ceil(height * TOP_RING)] = True
    deadline = time.monotonic() + 120 + 0.5 * info.video_frames
    samples = []
    for index in indices:
        with closing(_rgb_frames(video, width, height, deadline, filters=filters,
                                 limit=1, seek=float(index / info.fps))) as frames:
            samples.extend(frame[ring] for frame in frames)
    if len(samples) != len(indices):
        raise ProcessingError("ffmpeg", 0, "screen samples ended early", code="matte_frame_count")
    pixels = np.concatenate(samples).astype(np.int16)
    green = pixels[:, 1] - np.maximum(pixels[:, 0], pixels[:, 2])
    blue = pixels[:, 2] - np.maximum(pixels[:, 0], pixels[:, 1])
    screen, excesses = ("green", green) if (green >= SCREEN_EXCESS).sum() >= (
        blue >= SCREEN_EXCESS
    ).sum() else ("blue", blue)
    candidates = excesses >= SCREEN_EXCESS
    if not candidates.any():
        return None
    key = np.rint(np.median(pixels[candidates], axis=0)).astype(int)
    near = np.linalg.norm(pixels - key, axis=1) <= KEY_DISTANCE
    share = float((candidates & near).mean())
    if share < BORDER_SHARE and not forced:
        return None
    channel = 1 if screen == "green" else 2
    ratios = excesses[candidates] / np.maximum(pixels[candidates, channel], CHANNEL_FLOOR)
    return {"screen": screen, "key": "#" + "".join(f"{value:02x}" for value in key),
            "border_share": round(share, 3), "excess": float(np.median(excesses[candidates])),
            "ratio": float(np.median(ratios))}


def detect(video: Path) -> dict | None:
    """Return the dominant flat green or blue border from evenly spaced frames."""
    with paths.operation():
        source = paths.validate_input(video)
        return _detect(source, ffmpeg.probe(source, count_frames=True))


def _alpha_frames(
    source: Path, identity: VideoIdentity, metadata: dict, deadline: float,
    coverage: list[float], empty: list[int],
) -> Iterator[bytes]:
    channel = 1 if metadata["screen"] == "green" else 2
    count = 0
    with closing(_rgb_frames(source, identity.width, identity.height, deadline)) as frames:
        for index, frame in enumerate(frames):
            if index >= identity.frames:
                raise ProcessingError("ffmpeg", 0, "extra RGB frame", code="matte_frame_count")
            rgb = frame.astype(np.int16)
            other = np.maximum(rgb[:, :, 0], rgb[:, :, 3 - channel])
            ratio = (rgb[:, :, channel] - other) / np.maximum(rgb[:, :, channel], CHANNEL_FLOOR)
            alpha = np.rint(255 * np.clip((metadata["hi"] - ratio) / (
                metadata["hi"] - metadata["lo"]
            ), 0, 1)).astype(np.uint8)
            coverage.append(float((alpha >= 128).mean()))
            if not alpha.any():
                empty.append(index)
            yield alpha.tobytes()
            count += 1
    if count != identity.frames:
        raise ProcessingError("ffmpeg", 0, "RGB frame count differs", code="matte_frame_count")


def make_screen_matte(
    video: Path, out_dir: Path, *, backend: str, cleanup: Cleanup,
    info: ffmpeg.MediaInfo, identity: VideoIdentity,
) -> MatteData | None:
    """Publish a screen matte, or leave an ordinary auto request to person detection."""
    from yap_studio.matte.make import _deadline_seconds, _encode_frames

    with paths.operation():
        paths.protect_input(video)
        source = paths.validate_input(video)
        found = _detect(source, info, forced=backend == "chroma")
        if backend == "auto" and found is None:
            return None
        if found is None:
            raise YapError(
                "No green or blue screen found at the edges of the take; "
                "use the person cutout instead.",
                error_type="input_error", code="chroma_screen_missing",
                suggested_action="Use yap-studio matte make VIDEO --out DIR --backend auto, "
                "or set broll.cutout to person in the project style.",
            )
        # Vision refinement is not part of a colour key.
        cleanup = replace(cleanup, refine=False)
        # Pale mint sits near 0.04; a lit screen sits near 0.45 to 0.6.
        # Darker screen corners keep the same ratio. The 0.12 floor protects mint.
        lo = max(0.12, 0.4 * found["ratio"])
        # The 0.7 fraction clears screen falloff; 0.05 keeps a soft edge ramp.
        hi = max(lo + 0.05, 0.7 * found["ratio"])
        metadata = {"name": "chroma-key", **found, "lo": lo, "hi": hi, "requested": backend}
        folder = paths.validate_output_dir(out_dir)
        matte_path = paths.validate_output(folder / "matte.mkv")
        document = paths.validate_output(folder / "matte.json")
        coverage: list[float] = []
        empty: list[int] = []
        start = info.video_start_s or 0.0
        times = tuple(start + float(i / identity.fps) for i in range(identity.frames))
        warnings = (("variable frame rate: frame numbers follow decode order, not time x fps",)
                    if info.variable_frame_rate else ())
        published = False
        try:
            with paths.atomic_output(matte_path) as temporary:
                deadline = time.monotonic() + _deadline_seconds(identity.frames)
                with closing(_alpha_frames(source, identity, metadata, deadline,
                                            coverage, empty)) as frames:
                    _encode_frames(frames, temporary, identity, cleanup, deadline)
                encoded = ffmpeg.probe(temporary, count_frames=True)
                if (encoded.video_frames, encoded.width, encoded.height) != (
                    identity.frames, identity.width, identity.height,
                ):
                    raise ProcessingError("ffprobe", 0, "matte frame count or dimensions differ",
                                          code="matte_frame_count")
                if empty:
                    examples = ", ".join(f"{times[i]:.3f} s" for i in empty[:5])
                    warnings += (f"no person: {len(empty)} frame(s) left empty at {examples}",)
                data = MatteData(
                    "measured", None, identity, metadata, cleanup,
                    {"file": "matte.mkv", "sha256": receipt.sha256_file(temporary),
                     "codec": "ffv1", "pix_fmt": "gray"}, times,
                    start + float(identity.frames / identity.fps), tuple(coverage), warnings,
                    tuple(empty),
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


def _matches_request(backend: Mapping, cutout: str) -> bool:
    """Match cached backend metadata to the requested cutout."""
    return backend.get("requested", "auto") == ("chroma" if cutout == "screen" else cutout)


def for_request(cached: _CachedMatte | None, cutout: str) -> _CachedMatte | None:
    """Keep legacy auto entries and reject entries made for another request."""
    if cutout not in ("auto", "screen", "person"):
        raise ValidationError("broll.cutout", "expected auto, screen or person")
    if cached is not None:
        try:
            data = load_matte(cached.path.parent / "matte.json")
            if _matches_request(data.backend, cutout):
                return cached
        except (YapError, OSError, ValueError):
            pass
    return None


def make_cut_matte(
    source: Path, folder: Path, cutout: str, make: Callable[..., MatteData],
    *, joins: Sequence[int] = (),
) -> MatteData:
    """Map style requests to the existing person choice or the screen backend."""
    from yap_studio.matte import vision

    # Omit empty joins so existing stand-in makers keep working.
    kwargs = {"joins": joins} if joins else {}
    if cutout == "auto":
        return make(source, folder, **kwargs)
    backend = "chroma" if cutout == "screen" else (
        "vision" if vision.unavailable_reason() is None else "hyperframes"
    )
    if cutout == "person":
        return make(source, folder, backend=backend, requested="person", **kwargs)
    return make(source, folder, backend=backend, **kwargs)


def project_request(project_dir: Path) -> str:
    """Use the same resolved style as the other project render stages."""
    from yap_studio.project.status import ProjectContext

    if not (project_dir / "project.json").exists():
        return "auto"
    return ProjectContext(project_dir).render_style().data["broll"]["cutout"]


def cutout_metadata(matte: Path | None) -> dict[str, Any] | None:
    """Expose screen evidence only for a chroma matte."""
    if matte is not None and (matte.parent / "matte.json").is_file():
        backend = load_matte(matte.parent / "matte.json").backend
        if backend.get("name") == "chroma-key":
            return {key: backend[key] for key in ("name", "screen", "key", "border_share")}
    return None


def despill(graph: _Graph, person: str, matte: str, screen: str) -> tuple[str, str]:
    """Remove screen spill through a thin band and retain the original alpha."""
    if screen not in ("green", "blue"):
        raise ValidationError("matte.backend.screen", "expected green or blue")
    tag = f"chroma{len(graph.filters)}"
    graph.filters.append(f"[{matte}]split=3[{tag}mask][{tag}positive][{tag}blur]")
    # Keep alpha full range while the merge negotiates its RGB inputs.
    mask = graph.filter(f"{tag}mask", "format=gray")
    positive = graph.filter(f"{tag}positive", "format=gray,lut=c0='if(gt(val,0),255,0)'")
    blurred = graph.filter(f"{tag}blur", "format=gray,gblur=sigma=4,"
                           "format=gray,lut=c0='if(lt(val,250),255,0)'")
    band = graph.filter(f"{positive}][{blurred}", "blend=all_mode=multiply,format=gray")
    rgb = graph.filter(person, "format=gbrp")
    graph.filters.append(f"[{rgb}]split=2[{tag}original][{tag}spill]")
    # The blue screen type still defaults to reducing the green channel.
    channels = ":green=0:blue=-1" if screen == "blue" else ""
    clean = graph.filter(f"{tag}spill", f"despill=type={screen}:mix=0.5{channels},format=gbrp")
    band = graph.filter(band, "format=gbrp")
    merged = graph.filter(f"{tag}original][{clean}][{band}", "maskedmerge")
    return merged, mask


def append_despill(
    filters: list[str], person: str, matte: str, backend: dict | None,
) -> tuple[str, str]:
    """Add edge despill for screen mattes and leave other filter graphs intact."""
    if backend is None or backend.get("name") != "chroma-key":
        return person, matte
    return despill(_FilterGraph(filters), person, matte, backend["screen"])


def despill_people(graph: _Graph, people: dict[str, str], metadata: dict | None) -> dict[str, str]:
    """Clean each restored person before any later edge colour corrections."""
    if metadata is None:
        return people
    result = {}
    for kind, source in people.items():
        tag = f"chroma{len(graph.filters)}"
        rgba = graph.filter(source, "format=rgba")
        graph.filters.append(f"[{rgba}]split=2[{tag}person][{tag}alpha]")
        alpha = graph.filter(f"{tag}alpha", "alphaextract,format=gray")
        person, mask = despill(graph, f"{tag}person", alpha, metadata["screen"])
        result[kind] = graph.filter(f"{person}][{mask}", "alphamerge=ts_sync_mode=nearest")
    return result


def _cached_document(context: ProjectContext) -> tuple[Path, dict] | None:
    """Read the current cut's matte document without decoding picture."""
    cache = context.project_dir / ".yap/cache/matte"
    cut = context.project_dir / "cut/cut.mp4"
    if not cache.is_dir() or not cut.is_file():
        return None
    try:
        digest = context.identity.sha256(cut)
        data = context.json(f".yap/cache/matte/{digest}/matte.json")
        return cache / digest, data
    except (OSError, ValueError, KeyError, TypeError, AttributeError, YapError):
        return None


def keyed_screen(context: ProjectContext, style: Mapping) -> str | None:
    """Decide screen keying from the cutout choice, cached matte or declared take."""
    cutout = style.get("broll", {}).get("cutout", "auto")
    if cutout == "screen":
        return "green or blue"
    if cutout == "person":
        return None
    cached = _cached_document(context)
    if cached is not None:
        try:
            backend = cached[1]["backend"]
            if _matches_request(backend, "auto"):
                return backend["screen"] if backend.get("name") == "chroma-key" else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError, YapError):
            pass
    return "green or blue" if context.project.video_type == "green_screen" else None


def cached_status(context: ProjectContext) -> dict[str, str]:
    """Read the current cut's small matte document through the identity cache."""
    cached = _cached_document(context)
    if cached is None:
        return {}
    folder, data = cached
    try:
        if data.get("status") != "measured":
            return {}
        matte = folder / data["matte"]["file"]
        if not matte.is_file() or matte.stat().st_size == 0:
            return {}
        backend = data["backend"]
        if backend.get("name") == "chroma-key":
            label = (f"{backend['screen']} screen key "
                     f"({backend['border_share']:.0%} of the edges)")
        else:
            label = "person detection"
        return {"cutout": label}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, YapError):
        return {}
