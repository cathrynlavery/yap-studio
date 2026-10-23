"""Short project transactions and atomic compare-and-swap JSON writes."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yap_studio import paths, receipt
from yap_studio.errors import OutputPathError, YapError


class ConflictError(YapError):
    """Refuse a write based on an obsolete file snapshot."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path.name}: the base changed before the write.",
            error_type="project_conflict", code="base_changed",
            suggested_action="Re-read the file and retry with its current hash.",
        )


def timestamp() -> str:
    """Return a UTC timestamp for project bookkeeping."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def pid_alive(pid: object) -> bool:
    """Check whether a progress owner is still alive on this host."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def project_lock(project_dir: Path, *, timeout_s: float = 30) -> Iterator[None]:
    """Lock .yap/lock; the OS releases it even when the owning process dies."""
    project_file = project_dir / "project.json"
    if project_file.is_file():
        from yap_studio.project.model import load_project

        # Protect every take before even the lock metadata is written. Recording
        # missing paths also prevents bookkeeping from creating a future input.
        for take in load_project(project_file).takes:
            paths._record(paths._INPUT_REALPATHS, os.path.realpath(project_dir / take.path))
    root = paths.validate_output_dir(project_dir / ".yap")
    lock_file = root / "lock"
    paths._guard_write(lock_file)
    descriptor = os.open(
        lock_file, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b" ")
        else:
            import fcntl

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                pass
            except OSError:
                if os.name != "nt":
                    raise
            if time.monotonic() >= deadline:
                raise YapError(
                    "Another writer holds the project lock.", error_type="project_conflict",
                    code="project_locked", suggested_action="Wait for the writer and retry.",
                )
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        metadata = {"pid": os.getpid(), "host": socket.gethostname(), "since": timestamp()}
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, (json.dumps(metadata) + "\n").encode("utf-8"))
        os.ftruncate(descriptor, os.lseek(descriptor, 0, os.SEEK_CUR))
        yield
    finally:
        try:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def write_json_cas(path: Path, data: dict, *, base_sha256: str | None) -> str:
    """Under the caller's project lock, replace JSON only if its base still matches.

The base hash authorizes intentional JSON edits, including the unmarked identity
map. All input, symlink, and unsafe-directory guards still apply.
"""
    paths._guard_write(path)
    if path.exists() and not path.is_file():
        raise OutputPathError("JSON output must be a regular file.", code="unsafe_path")
    current = receipt.sha256_file(path) if path.exists() else None
    if current != base_sha256:
        raise ConflictError(path)
    payload = (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".yap_tmp_", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        paths._guard_write(path)
        current = receipt.sha256_file(path) if path.exists() else None
        if current != base_sha256:
            raise ConflictError(path)
        os.replace(temporary, path)
        return hashlib.sha256(payload).hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, data: dict[str, Any]) -> str:
    """Rewrite bookkeeping while its project lock is held."""
    return write_json_cas(
        path, data, base_sha256=receipt.sha256_file(path) if path.exists() else None,
    )
