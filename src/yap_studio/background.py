"""Private records and process checks for commands that outlive their caller."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from yap_studio.errors import YapError
from yap_studio.review.store import data_dir


def error(message: str, code: str, *, json_output: bool) -> None:
    """Print a background command error."""
    print_error(YapError(message, code=code), json_output=json_output)


def print_error(exc: YapError, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
    else:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)


def directory() -> Path:
    return (data_dir() / "detached").resolve()


def posix_supported() -> bool:
    return os.name == "posix"


def records(kind: str) -> list[Path]:
    return sorted(directory().glob(f"{kind}-*.json"))


def read_record(path: Path) -> dict | None:
    try:
        document = json.loads(path.read_text())
        return document if isinstance(document, dict) else None
    except (OSError, ValueError):
        return None


def remove_record(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".log").unlink(missing_ok=True)


def command_matches(pid: int, identifier: str | Path, marker: str = "--detached-record"
                    ) -> bool | None:
    """Return None when ps cannot check a process."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    argument = (r"(?<!\S)" + re.escape(marker) + r"(?:=|\s+)"
                + re.escape(str(identifier)) + r"(?=\s|$)")
    return result.returncode == 0 and re.search(argument, result.stdout) is not None


def stop_record(path: Path, document: dict) -> tuple[bool, bool | None]:
    """Return whether a page was signaled and whether it stopped."""
    pid = document.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        remove_record(path)
        return False, True
    matches = command_matches(pid, path)
    if matches is None:
        return False, None
    if not matches:
        remove_record(path)
        return False, True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_record(path)
        return False, True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        matches = command_matches(pid, path)
        if matches is None:
            return True, None
        if not matches:
            remove_record(path)
            return True, True
        time.sleep(0.1)
    return True, False


def private_file(path: Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "w")
    except BaseException:
        os.close(fd)
        raise


def write_json(path: Path, document: dict) -> None:
    """Replace a private JSON file atomically."""
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with private_file(temporary) as output:
            output.write(json.dumps(document, ensure_ascii=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_record(path: Path, document: dict, **fields) -> None:
    """Publish the ready link for the waiting parent."""
    arguments = sys.argv[1:]
    index = arguments.index("--detached-record")
    command = [sys.executable, "-m", "yap_studio", *arguments[:index], *arguments[index + 2:]]
    write_json(path, {
        "pid": os.getpid(), "document": document, "log": str(path.with_suffix(".log")),
        "started_at": datetime.now(UTC).isoformat(), "command": command, **fields,
    })


def log_lines(path: Path) -> list[str]:
    try:
        return path.read_text().splitlines()
    except OSError:
        return []


def relay_child_line(line: str, *, json_output: bool) -> None:
    """Render child errors for a plain parent while preserving other log lines."""
    if not json_output:
        try:
            document = json.loads(line)
        except ValueError:
            document = None
        if isinstance(document, dict) and isinstance(document.get("error"), dict):
            exc = document["error"]
            print(f"error: {exc.get('message', '')}", file=sys.stderr)
            if exc.get("suggested_action"):
                print(f"fix: {exc['suggested_action']}", file=sys.stderr)
            return
    print(line, file=sys.stderr)


def terminate(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def spawn_record(command: list[str], *, kind: str, cwd: str | None = None
                 ) -> tuple[subprocess.Popen, Path, Path]:
    folder = directory()
    folder.mkdir(parents=True, mode=0o700, exist_ok=True)
    folder.chmod(0o700)
    identifier = f"{kind}-{uuid.uuid4().hex}"
    record = folder / f"{identifier}.json"
    log = folder / f"{identifier}.log"
    try:
        with private_file(log) as output:
            child = subprocess.Popen([*command, "--detached-record", str(record)], cwd=cwd,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                     close_fds=True, start_new_session=True)
    except OSError:
        log.unlink(missing_ok=True)
        raise
    return child, record, log


def ready(child: subprocess.Popen, record: Path, log: Path, *, kind: str, label: str,
          json_output: bool) -> dict | None:
    deadline = time.monotonic() + 120
    while not record.exists():
        if child.poll() is not None:
            lines = log_lines(log)
            for line in lines:
                relay_child_line(line, json_output=json_output)
            if not lines and json_output:
                error(f"could not start {label}", f"{kind}_start_failed", json_output=json_output)
            remove_record(record)
            return None
        if time.monotonic() >= deadline:
            terminate(child)
            remove_record(record)
            error(f"{label} did not become ready within 120 seconds; stopped it",
                  f"{kind}_ready_timeout", json_output=json_output)
            return None
        time.sleep(0.1)
    saved = read_record(record)
    if saved is None:
        terminate(child)
        remove_record(record)
        error(f"{label} readiness record could not be read", f"{kind}_record_unreadable",
              json_output=json_output)
    return saved
