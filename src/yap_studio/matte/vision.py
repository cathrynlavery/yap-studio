"""Build and stream the macOS Vision person segmentation helper."""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from yap_studio import tools
from yap_studio.cache import cache_dir, cache_root
from yap_studio.errors import DependencyError, ProcessingError


def unavailable_reason() -> str | None:
    """Explain missing Vision support, allowing an already cached helper."""
    if sys.platform != "darwin":
        return f"person segmentation uses macOS Vision; this computer runs {platform.system()}"
    cached = helper_path()
    if cached.is_file() and os.access(cached, os.X_OK):
        return None
    reason = "person segmentation needs the Xcode command line tools (xcode-select --install)"
    if tools.find_tool("xcrun") is None:
        return reason
    try:
        tools.run_tool("xcrun", ["--find", "swiftc"], timeout_s=30)
    except (DependencyError, ProcessingError):
        return reason
    return None


def helper_source() -> Path:
    """Return the packaged Swift segmentation source."""
    return Path(__file__).with_name("person_matte.swift")


def helper_path() -> Path:
    """Return the source-digest cache path without creating directories."""
    digest = hashlib.sha256(helper_source().read_bytes()).hexdigest()
    return cache_root() / "helpers" / f"yap-person-matte-{digest[:16]}"


def helper_executable() -> Path:
    """Compile once per source digest and atomically publish the helper."""
    executable = helper_path()
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    descriptor, name = tempfile.mkstemp(prefix=".yap-person-matte-", dir=cache_dir("helpers"))
    os.close(descriptor)
    temporary = Path(name)
    try:
        tools.run_tool(
            "xcrun", ["swiftc", "-O", "-o", str(temporary), str(helper_source())], timeout_s=600
        )
        os.replace(temporary, executable)
    finally:
        temporary.unlink(missing_ok=True)
    return executable


@dataclass(frozen=True)
class HelperStream:
    """A streaming child and its disk-backed diagnostics and records."""

    process: subprocess.Popen[bytes]
    command: list[str]
    records: Path
    stderr: BinaryIO


@contextlib.contextmanager
def stream_helper(video: Path, *, quality: str) -> Iterator[HelperStream]:
    """Start the helper with closed stdin and reap it on every exit path."""
    helper = helper_executable()
    with tempfile.TemporaryDirectory(prefix="yap-person-matte-") as folder:
        records = Path(folder) / "records.jsonl"
        command = [str(helper), "--quality", quality, "--records", str(records), str(video)]
        with tempfile.TemporaryFile() as stderr:
            try:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    bufsize=0,
                )
            except OSError as exc:
                raise ProcessingError(
                    " ".join(command), -1, str(exc), code="person_matte_output"
                ) from exc
            try:
                yield HelperStream(child, command, records, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
                assert child.stdout is not None
                child.stdout.close()
