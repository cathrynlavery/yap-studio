"""Full-take speech work owned by the page that starts it."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from yap_studio import background, cache
from yap_studio.errors import YapError

from . import shared
from .backends import ENGINES, Backend, resolve_backend


def watch_parent() -> None:
    """Stop this private process group when the owning page's pipe closes."""
    def watch() -> None:
        try:
            while os.read(0, 4096):
                pass
        finally:
            os.killpg(os.getpgrp(), signal.SIGTERM)

    threading.Thread(target=watch, daemon=True).start()


def run(take: Path, backend: Backend) -> None:
    """Publish a full raw transcript, or return immediately on a cache hit."""
    digest = shared.take_digest(take)
    shared.transcribe(backend, [shared.Request(digest, 0, take)])


def _group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        from yap_studio.stages.run_detach import _group_running

        return _group_running(pid) is not False
    return True


def _signal_group(pid: int, number: int) -> None:
    try:
        os.killpg(pid, number)
    except ProcessLookupError:
        pass
    except PermissionError:
        if _group_exists(pid):
            raise


class TakeJobs:
    """Hold one page's child and keep start and close safe across threads."""

    child_command = [sys.executable, "-m", "yap_studio.transcript.take_job"]

    def __init__(self, *, child_command: list[str] | None = None) -> None:
        self.child_command = list(
            child_command if child_command is not None else self.child_command,
        )
        self._lock = threading.Lock()
        self._closed = False
        self._child: subprocess.Popen | None = None
        self._record: Path | None = None

    def start(self, take: Path, backend: Backend | None) -> bool:
        """Start without waiting for speech, unless work is unavailable or already done."""
        with self._lock:
            if (self._closed or not background.posix_supported() or backend is None
                    or backend.name not in ENGINES or not take.is_file()):
                return False
            if self._child is not None:
                if self._child.poll() is None:
                    return False
                self._stop(0)
            if backend.download_notice() is not None:
                return False
            digest = shared.known_digest(take)
            if (digest is not None
                    and shared.lookup(backend, shared.Request(digest, 0, take)) is not None):
                return False
            record = cache.cache_dir("transcripts", "jobs") / f"{uuid4().hex}.json"
            child = subprocess.Popen(
                [*self.child_command, str(take), "--engine", backend.name, "--model",
                 shared.model_spec(backend), "--take-job", str(record)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
            )
            self._child, self._record = child, record
            try:
                background.write_json(record, {"pid": child.pid, "take": str(take),
                                               "started_at": datetime.now(UTC).isoformat()})
                if child.poll() is not None:
                    background.remove_record(record)
            except BaseException:
                self._stop(0)
                raise
            return True

    def _stop(self, timeout_s: float) -> None:
        child, record = self._child, self._record
        if child is None:
            return
        try:
            _signal_group(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + timeout_s
            while True:
                child.poll()  # Reap the leader before checking its process group.
                if not _group_exists(child.pid) or time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
            if _group_exists(child.pid):
                _signal_group(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            deadline = time.monotonic() + 5
            while _group_exists(child.pid):
                if time.monotonic() >= deadline:
                    raise OSError(f"take job process group {child.pid} did not stop")
                time.sleep(0.02)
        finally:
            if child.stdin is not None:
                child.stdin.close()
            if record is not None:
                background.remove_record(record)
            self._child, self._record = None, None

    def close(self, timeout_s: float = 5.0) -> None:
        """Prevent future starts and stop the child and all of its descendants."""
        with self._lock:
            self._closed = True
            self._stop(timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("take", type=Path)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--take-job", required=True, type=Path)
    args = parser.parse_args()

    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        watch_parent()
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
            backend = resolve_backend(args.engine, args.model)
            if backend is not None and backend.download_notice() is None:
                run(args.take, backend)
    except (OSError, YapError, ValueError):
        pass
    finally:
        background.remove_record(args.take_job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
