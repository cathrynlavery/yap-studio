"""Safe ffmpeg and ffprobe process helpers."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import subprocess
import tempfile
import time
from collections.abc import Generator, Iterable
from contextlib import closing
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio import process
from yap_studio.errors import DependencyError, InputError, ProcessingError, classify_media_failure

DEFAULT_TIMEOUT_S = 600
PROBE_TIMEOUT_S = 60
MIN_VERSION = (6, 1)
VERSION_TIMEOUT_S = 5


@dataclass(frozen=True)
class _ToolVersion:
    """Keep the executable name and release token from a version banner."""

    name: str
    found: str


_version_cache: dict[tuple[str, int, int, int, int], _ToolVersion | None] = {}


def _read_version(path: str) -> _ToolVersion | None:
    """Read a banner once per executable revision, tolerating failed probes."""
    realpath = os.path.realpath(path)
    try:
        stat = os.stat(realpath)
        key = (realpath, stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_dev)
    except OSError:
        return None
    if key not in _version_cache:
        version = None
        try:
            result = process.spawn([path, "-version"], timeout_s=VERSION_TIMEOUT_S, tool="ffmpeg")
            lines = (result.stdout or "").splitlines()
            if result.returncode == 0 and lines:
                match = re.match(r"^(ffmpeg|ffprobe) version\s+(\S+)", lines[0])
                if match:
                    version = _ToolVersion(match.group(1), match.group(2))
        except (ProcessingError, OSError):
            pass
        _version_cache[key] = version
    return _version_cache[key]


def check_version(path: str, name: str | None = None) -> None:
    """Reject mismatched tools and old releases; allow unknown development builds."""
    version = _read_version(path)
    if version is None:
        return
    if name is not None and version.name != name:
        raise DependencyError(
            f"{path} is not {name}.",
            error_type="dependency_error",
            code=f"{name}_wrong_tool",
            suggested_action=f"Point YAP_STUDIO_{name.upper()} at the {name} executable.",
        )
    match = re.match(r"^n?(\d+)\.(\d+)(?:\.\d+)?(?:\D|$)", version.found)
    if match and (int(match.group(1)), int(match.group(2))) < MIN_VERSION:
        minimum = ".".join(str(part) for part in MIN_VERSION)
        raise DependencyError.ffmpeg_too_old(version.name, version.found, minimum)


def resolve_tool(name: str) -> str:
    """Resolve an executable for diagnostics without enforcing its version."""
    if name not in {"ffmpeg", "ffprobe"}:
        raise ValueError(f"unsupported media tool: {name}")
    override = os.environ.get(f"YAP_STUDIO_{name.upper()}")
    if override:
        candidate = os.path.abspath(os.path.expanduser(override))
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        raise DependencyError.missing_tool(name)
    found = shutil.which(name)
    if found is None:
        raise DependencyError.missing_tool(name)
    return found


def find_tool(name: str) -> str:
    """Find a supported ffmpeg-family executable, honoring its environment override."""
    path = resolve_tool(name)
    check_version(path, name)
    return path


def _safe_command(cmd: list[str]) -> list[str]:
    if not cmd or cmd[0] not in {"ffmpeg", "ffprobe"}:
        raise ValueError("command must start with ffmpeg or ffprobe")
    name = cmd[0]
    flags = ["-hide_banner"]
    if name == "ffmpeg":
        flags.insert(0, "-nostdin")
    missing = [flag for flag in flags if flag not in cmd[1:]]
    return [find_tool(name), *missing, *cmd[1:]]


def run(
    cmd: list[str],
    *,
    timeout_s: float,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run ffmpeg or ffprobe with consistent safety and error handling."""
    command = _safe_command(cmd)
    result = process.spawn(command, timeout_s=timeout_s, tool="ffmpeg", binary=binary)
    if result.returncode:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise classify_media_failure(stderr or "", command)
    return result


def run_stream(
    cmd: list[str],
    *,
    timeout_s: float,
    chunk_size: int = 64 * 1024,
) -> Generator[bytes, None, None]:
    """Yield bounded stdout chunks with the same errors and deadline as ``run``.

    Stderr is captured on disk so it cannot fill a pipe or grow working memory.
    Exhaust or close the generator to reap the child, including on early exit.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    command = _safe_command(cmd)
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr, bufsize=0
        )
        deadline = time.monotonic() + timeout_s
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(command, timeout_s)
                    chunk = os.read(process.stdout.fileno(), chunk_size)
                    if not chunk:
                        break
                    yield chunk
            process.wait(timeout=max(0, deadline - time.monotonic()))
            if process.returncode:
                stderr.seek(0)
                raise classify_media_failure(
                    stderr.read().decode("utf-8", errors="replace"), command
                )
        except subprocess.TimeoutExpired:
            raise ProcessingError(
                " ".join(command),
                -1,
                f"timed out after {timeout_s} s",
                code="ffmpeg_timeout",
            ) from None
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def variable_frame_rate(average: Fraction | None, nominal: Fraction | None) -> bool:
    """Recognise variable timing only when both positive probe rates differ."""
    return (
        average is not None and nominal is not None
        and average > 0 and nominal > 0 and average != nominal
    )


@dataclass(frozen=True)
class MediaInfo:
    """Container and selected-stream facts used by verification."""

    path: str
    format_name: str | None
    duration_s: float | None
    format_start_s: float
    has_video: bool
    video_index: int | None
    video_codec: str | None
    fps: Fraction | None
    width: int | None
    height: int | None
    coded_width: int | None
    coded_height: int | None
    rotation: int
    orientation: str | None
    video_frames: int | None
    video_start_s: float | None
    video_offset_s: float
    video_duration_s: float | None
    has_audio: bool
    audio_index: int | None
    audio_codec: str | None
    sample_rate: int | None
    channels: int | None
    audio_start_s: float | None
    audio_offset_s: float
    audio_duration_s: float | None
    pix_fmt: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_space: str | None = None
    color_range: str | None = None
    format_tags: dict[str, str] | None = field(default=None, hash=False)
    nominal_fps: Fraction | None = None
    video_first_frame_s: float | None = None

    @property
    def variable_frame_rate(self) -> bool:
        """Report whether the average and nominal rates imply variable timing."""
        # If the average is unavailable, fps falls back to the nominal rate.
        return variable_frame_rate(self.fps, self.nominal_fps)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible facts without private container metadata."""
        result = asdict(self)
        result.pop("format_tags")
        result["fps"] = (
            f"{self.fps.numerator}/{self.fps.denominator}" if self.fps is not None else None
        )
        result["nominal_fps"] = (
            f"{self.nominal_fps.numerator}/{self.nominal_fps.denominator}"
            if self.nominal_fps is not None else None
        )
        return result


def _optional_float(value: Any) -> float | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frame_rate(stream: dict[str, Any]) -> Fraction | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if not value:
            continue
        try:
            rate = Fraction(value)
        except (ValueError, ZeroDivisionError):
            continue
        if rate:
            return rate
    return None


def _rotation(stream: dict[str, Any]) -> int:
    raw: Any = None
    for side_data in stream.get("side_data_list", []):
        if side_data.get("rotation") is not None:
            raw = side_data["rotation"]
            break
    if raw is None:
        raw = stream.get("tags", {}).get("rotate", 0)
    try:
        return (int(round(float(raw) / 90.0)) * 90) % 360
    except (TypeError, ValueError):
        return 0


def _first_stream(streams: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("codec_type") != kind:
            continue
        if kind == "video" and stream.get("disposition", {}).get("attached_pic") == 1:
            continue
        return stream
    return None


def _packet_count(path: Path, stream_index: int) -> int | None:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            str(stream_index),
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        timeout_s=PROBE_TIMEOUT_S,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    return _optional_int(streams[0].get("nb_read_packets")) if streams else None


def _first_video_frame_time(path: Path, stream_index: int) -> float | None:
    """Read the decoder's first timestamp, then stop without scanning the source."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", str(stream_index),
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time:frame_side_data=",
        "-of", "csv=p=0", str(path),
    ]
    try:
        with closing(run_stream(command, timeout_s=PROBE_TIMEOUT_S)) as chunks:
            first = b""
            for chunk in chunks:
                first += chunk
                if b"\n" in first:
                    row = first.split(b"\n", 1)[0]
                    return _optional_float(row.split(b",", 1)[0].decode("ascii").strip())
    except (InputError, ProcessingError):
        pass
    return None


def probe(
    path: str | os.PathLike[str], *, count_frames: bool = True, first_frame: bool = False,
) -> MediaInfo:
    """Probe container, stream, display rotation, and stream clock facts."""
    source = Path(path)
    if not source.is_file():
        raise InputError(str(source), "no file at this path")
    try:
        result = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(source),
            ],
            timeout_s=PROBE_TIMEOUT_S,
        )
    except InputError as exc:
        if exc.path:
            raise
        raise InputError(str(source), exc.reason) from exc
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise InputError(str(source), "ffmpeg cannot decode it as audio or video") from exc
    streams = payload.get("streams", [])
    video = _first_stream(streams, "video")
    audio = _first_stream(streams, "audio")
    if video is None and audio is None:
        raise InputError(str(source), "no audio or video stream")

    format_data = payload.get("format", {})
    format_start = _optional_float(format_data.get("start_time")) or 0.0
    rotation = _rotation(video) if video else 0
    source_width = _optional_int(video.get("width")) if video else None
    source_height = _optional_int(video.get("height")) if video else None
    coded_width = _optional_int(video.get("coded_width")) if video else None
    coded_height = _optional_int(video.get("coded_height")) if video else None
    if coded_width is None:
        coded_width = source_width
    if coded_height is None:
        coded_height = source_height
    width, height = source_width, source_height
    if rotation in {90, 270}:
        width, height = height, width
    orientation = None
    if width is not None and height is not None:
        orientation = "vertical" if height > width else "horizontal" if width > height else "square"

    video_start = _optional_float(video.get("start_time")) if video else None
    audio_start = _optional_float(audio.get("start_time")) if audio else None
    video_index = _optional_int(video.get("index")) if video else None
    audio_index = _optional_int(audio.get("index")) if audio else None
    frames = _optional_int(video.get("nb_frames")) if video else None
    if video is not None and video_index is not None and frames is None and count_frames:
        frames = _packet_count(source, video_index)

    return MediaInfo(
        path=str(source),
        format_name=format_data.get("format_name"),
        duration_s=_optional_float(format_data.get("duration")),
        format_start_s=format_start,
        has_video=video is not None,
        video_index=video_index,
        video_codec=video.get("codec_name") if video else None,
        fps=_frame_rate(video) if video else None,
        width=width,
        height=height,
        coded_width=coded_width,
        coded_height=coded_height,
        rotation=rotation,
        orientation=orientation,
        video_frames=frames,
        video_start_s=video_start,
        video_offset_s=round((video_start or 0.0) - format_start, 6) if video else 0.0,
        video_duration_s=_optional_float(video.get("duration")) if video else None,
        has_audio=audio is not None,
        audio_index=audio_index,
        audio_codec=audio.get("codec_name") if audio else None,
        sample_rate=_optional_int(audio.get("sample_rate")) if audio else None,
        channels=_optional_int(audio.get("channels")) if audio else None,
        audio_start_s=audio_start,
        audio_offset_s=round((audio_start or 0.0) - format_start, 6) if audio else 0.0,
        audio_duration_s=_optional_float(audio.get("duration")) if audio else None,
        pix_fmt=video.get("pix_fmt") if video else None,
        color_transfer=video.get("color_transfer") if video else None,
        color_primaries=video.get("color_primaries") if video else None,
        color_space=video.get("color_space") if video else None,
        color_range=video.get("color_range") if video else None,
        format_tags=format_data.get("tags"),
        nominal_fps=_frame_rate({"r_frame_rate": video.get("r_frame_rate")}) if video else None,
        video_first_frame_s=(
            _first_video_frame_time(source, video_index)
            if first_frame and video_index is not None else None
        ),
    )


def tool_versions() -> dict[str, str | None]:
    """Return installed ffmpeg and ffprobe version numbers."""
    versions: dict[str, str | None] = {}
    for name in ("ffmpeg", "ffprobe"):
        try:
            version = _read_version(resolve_tool(name))
        except DependencyError:
            versions[name] = None
            continue
        versions[name] = version.found if version is not None and version.name == name else None
    return versions


def has_filters(names: Iterable[str]) -> dict[str, bool]:
    """Report whether ffmpeg exposes each named filter."""
    requested = list(names)
    result = run(["ffmpeg", "-filters"], timeout_s=PROBE_TIMEOUT_S)
    available: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in requested:
            available.add(parts[1])
    return {name: name in available for name in requested}
