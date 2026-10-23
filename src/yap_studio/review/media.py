"""Video facts needed to review a single media file."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

from yap_studio import ffmpeg
from yap_studio.errors import InputError, YapError
from yap_studio.review.model import MediaFacts

_PACKET_CHECK_COUNT = 240
# Extra packets keep the incomplete B-frame reorder tail outside the checked range.
_PACKET_REORDER_MARGIN = 32


@dataclass(frozen=True)
class ReviewMedia:
    """Resolved video path and timeline facts for a review."""

    path: Path
    name: str
    duration_s: float | None
    fps: Fraction | None
    cfr: bool
    codec: str | None
    width: int | None
    height: int | None

    def facts(self, sha256: str) -> MediaFacts:
        """Combine probed facts with the file's content hash."""
        return MediaFacts(self.name, sha256, self.duration_s, self.fps, self.cfr)


def stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    """Identify a file version from its device, inode, size, and timestamps."""
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def file_identity(path: Path) -> tuple[int, int, int, int, int] | None:
    """Return a file version, or None when the path cannot be stat'ed."""
    try:
        return stat_identity(path.stat())
    except OSError:
        return None


def _hash_stream(stream: BinaryIO) -> str:
    """Hash an open file without changing its read position."""
    position = stream.tell()
    digest = hashlib.sha256()
    try:
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    finally:
        stream.seek(position)
    return digest.hexdigest()


class KnownMedia:
    """Share verified file identities across concurrent media requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._known: dict[tuple[Path, str], set[tuple[int, int, int, int, int]]] = {}

    def changed(self, path: Path, sha256: str, recorded: tuple[int, int, int, int, int],
                stream: BinaryIO | None = None) -> bool:
        """Return whether path or an open stream differs from the launch content."""
        with self._lock:
            known = self._known.setdefault((path, sha256), {recorded})
            for source in (None, stream) if stream is not None else (None,):
                try:
                    if source is None:
                        with path.open("rb") as opened:
                            if self._changed_stream(opened, sha256, recorded, known):
                                return True
                    elif self._changed_stream(source, sha256, recorded, known):
                        return True
                except OSError:
                    return True
            return False

    @staticmethod
    def _changed_stream(stream: BinaryIO, sha256: str,
                        recorded: tuple[int, int, int, int, int],
                        known: set[tuple[int, int, int, int, int]]) -> bool:
        identity = stat_identity(os.fstat(stream.fileno()))
        if identity[2] != recorded[2]:
            return True
        if identity in known:
            return False
        if _hash_stream(stream) != sha256 or stat_identity(os.fstat(stream.fileno())) != identity:
            return True
        known.add(identity)
        return False


def is_constant_rate(r_frame_rate: str | None, avg_frame_rate: str | None) -> bool:
    """Return whether both reported frame rates are valid, positive, and equal."""
    try:
        reference = Fraction(r_frame_rate)
        average = Fraction(avg_frame_rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return reference > 0 and reference == average


def _packet_rate_is_constant(stream: dict, real: Path, fps: Fraction) -> bool:
    """Check presentation gaps when enough packet timestamps are available."""
    try:
        time_base = Fraction(stream["time_base"])
        if time_base <= 0:
            return True
        index = stream["index"]
        result = ffmpeg.run(
            [
                "ffprobe", "-v", "error", "-select_streams", str(index),
                "-read_intervals", f"%+#{_PACKET_CHECK_COUNT + _PACKET_REORDER_MARGIN}",
                "-show_entries", "packet=pts",
                "-of", "json", str(real),
            ],
            timeout_s=ffmpeg.PROBE_TIMEOUT_S,
        )
        packets = json.loads(result.stdout)["packets"]
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError, YapError):
        return True
    if not isinstance(packets, list):
        return True
    points = []
    for packet in packets:
        try:
            value = packet["pts"]
        except (KeyError, TypeError):
            continue
        if type(value) is int or isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            points.append(int(value))
    if len(points) < 3:
        return True
    points.sort()
    points = points[:_PACKET_CHECK_COUNT]
    expected = (1 / fps) / time_base
    return all(
        abs(right - left - expected) <= 1
        for left, right in zip(points, points[1:], strict=False)
    )


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and str(number) == str(value) else None


def _tag_duration(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)(?:\.(\d+))?", value)
    if match is None:
        return None
    hours, minutes, seconds, fraction = match.groups()
    return _positive_float(
        int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        + (float(f"0.{fraction}") if fraction else 0)
    )


def _picture_duration(stream: dict, real: Path, fps: Fraction | None) -> float | None:
    duration = _positive_float(stream.get("duration"))
    if duration is not None:
        return duration
    tags = stream.get("tags") or {}
    duration = _tag_duration(tags.get("DURATION"))
    if duration is not None:
        return duration
    if fps is None:
        return None
    frames = _positive_int(stream.get("nb_frames"))
    if frames is None:
        index = stream.get("index")
        if index is None:
            return None
        result = ffmpeg.run(
            [
                "ffprobe", "-v", "error", "-count_packets", "-select_streams", str(index),
                "-show_entries", "stream=nb_read_packets", "-of", "json", str(real),
            ],
            timeout_s=ffmpeg.PROBE_TIMEOUT_S,
        )
        try:
            packets = json.loads(result.stdout)["streams"][0]["nb_read_packets"]
        except (TypeError, ValueError, KeyError, IndexError):
            return None
        frames = _positive_int(packets)
    return float(frames / fps) if frames is not None else None


def probe_media(path: str | os.PathLike[str]) -> ReviewMedia:
    """Probe a video without counting frames, then check its rate consistency."""
    real = Path(os.path.realpath(path))
    info = ffmpeg.probe(real, count_frames=False)
    if not info.has_video:
        raise InputError(str(real), "no video stream")
    result = ffmpeg.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=index,r_frame_rate,avg_frame_rate,time_base,duration,nb_frames:stream_disposition=attached_pic:stream_tags=DURATION",
            "-of",
            "json",
            str(real),
        ],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    )
    try:
        streams = json.loads(result.stdout)["streams"]
        rates = next(
            stream for stream in streams
            if stream.get("disposition", {}).get("attached_pic") != 1
        )
        reference = rates.get("r_frame_rate")
        average = rates.get("avg_frame_rate")
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, StopIteration) as exc:
        raise InputError(str(real), "not a readable video stream") from exc
    cfr = is_constant_rate(reference, average)
    fps = Fraction(average) if cfr else None
    if fps is not None and not _packet_rate_is_constant(rates, real, fps):
        cfr = False
        fps = None
    return ReviewMedia(
        path=real,
        name=real.name,
        duration_s=_picture_duration(rates, real, fps),
        fps=fps,
        cfr=cfr,
        codec=info.video_codec,
        width=info.width,
        height=info.height,
    )
