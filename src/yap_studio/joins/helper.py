"""Build and call Apple's frame rate conversion helper."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from yap_studio import process, tools
from yap_studio.cache import cache_dir, cache_root
from yap_studio.errors import DependencyError, ProcessingError, YapError

FIX = "Set moves.jump_cuts to hard or punch."
_support: dict[str, bool] = {}


class JoinError(YapError):
    """Refuse an unavailable processor or invalid join output with a fix."""

    def __init__(self, message: str, *, fix: str = FIX) -> None:
        super().__init__(message, error_type="processing_error", code="smooth_join_invalid",
                         suggested_action=fix)


class Interpolator(Protocol):
    """Produce ordered rgb48le frames between two decoded frames."""

    def interpolate(self, before: bytes, after: bytes, *, width: int, height: int,
                    phases: list[float]) -> list[bytes]: ...


def helper_source() -> Path:
    """Return the packaged Swift source."""
    return Path(__file__).with_name("smooth_frames.swift")


def source_digest() -> str:
    """Identify the processor implementation for helpers and movie caches."""
    return hashlib.sha256(helper_source().read_bytes()).hexdigest()


def helper_path() -> Path:
    """Return a digest-named path without creating directories."""
    return cache_root() / "helpers" / f"yap-smooth-frames-{source_digest()[:16]}"


def helper_executable() -> Path:
    """Compile once per source digest and publish atomically."""
    executable = helper_path()
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    descriptor, name = tempfile.mkstemp(prefix=".yap-smooth-frames-", dir=cache_dir("helpers"))
    os.close(descriptor)
    temporary = Path(name)
    try:
        tools.run_tool("xcrun", ["swiftc", "-O", "-o", str(temporary), str(helper_source())],
                       timeout_s=600)
        os.replace(temporary, executable)
    finally:
        temporary.unlink(missing_ok=True)
    return executable


def unavailable_reason() -> str | None:
    """Return a plain explanation and fix when frame conversion is unavailable."""
    if sys.platform != "darwin":
        system = {"linux": "Linux", "win32": "Windows"}.get(sys.platform, platform.system())
        return ("Smooth jump cuts use Apple's frame rate conversion on macOS 15.4 or later; "
                f"this computer runs {system}. {FIX}")
    version = platform.mac_ver()[0]
    numbers = tuple(int(n) for n in (version or "0").split(".")[:2])
    if numbers < (15, 4):
        return (f"Smooth jump cuts found macOS {version or 'unknown'}; update macOS to 15.4 "
                "or later, or set moves.jump_cuts to hard or punch.")
    cached = helper_path()
    if not (cached.is_file() and os.access(cached, os.X_OK)):
        reason = ("Smooth jump cuts need the Xcode command line tools "
                  f"(xcode-select --install) to build their helper. {FIX}")
        if tools.find_tool("xcrun") is None:
            return reason
        try:
            tools.run_tool("xcrun", ["--find", "swiftc"], timeout_s=30)
        except (DependencyError, ProcessingError):
            return reason
    digest = source_digest()
    if digest not in _support:
        try:
            result = process.spawn([str(helper_executable()), "--check"], timeout_s=60,
                                   tool="smooth_frames")
            record = json.loads(result.stdout)
            if (result.returncode or type(record.get("supported")) is not bool
                    or record.get("revision") != 1):
                raise ValueError("invalid support response")
        except (ValueError, TypeError, AttributeError, YapError) as exc:
            raise JoinError(f"Smooth jump cut helper check failed: {exc}",
                            fix="Install current Xcode command line tools, or " + FIX.lower()) \
                from exc
        _support[digest] = record["supported"]
    if not _support[digest]:
        return f"Apple's frame rate conversion processor is not supported on this Mac. {FIX}"
    return None


def interpolate(before: bytes, after: bytes, *, width: int, height: int,
                phases: list[float]) -> list[bytes]:
    """Run one bounded helper request and reject incomplete output."""
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or not phases or any(not math.isfinite(p) or not 0 < p < 1 for p in phases)
            or any(a >= b for a, b in zip(phases, phases[1:], strict=False))):
        raise JoinError("Smooth jump cut dimensions or phases are invalid.")
    size = width * height * 6
    if len(before) != size or len(after) != size:
        raise JoinError(f"Smooth jump cut inputs must each contain {size} bytes.")
    try:
        command = [str(helper_executable()), "--width", str(width), "--height", str(height),
                   "--phases", ",".join(str(p) for p in phases)]
        result = process.spawn(command, input_bytes=before + after, binary=True,
                               timeout_s=120, tool="smooth_frames")
    except (DependencyError, ProcessingError) as exc:
        raise JoinError(f"Smooth jump cut helper failed: {exc}") from exc
    if result.returncode:
        raise JoinError("Smooth jump cut helper failed: " +
                        result.stderr.decode("utf-8", errors="replace").strip())
    expected = size * len(phases)
    if len(result.stdout) != expected:
        raise JoinError(f"Smooth jump cut helper returned {len(result.stdout)} bytes; "
                        f"expected {expected}.")
    return [result.stdout[offset:offset + size] for offset in range(0, expected, size)]


class AppleInterpolator:
    """Use the cached Swift helper in production."""

    def interpolate(self, before: bytes, after: bytes, *, width: int, height: int,
                    phases: list[float]) -> list[bytes]:
        return interpolate(before, after, width=width, height=height, phases=phases)
