"""Private, serialized creator requests and one-shot agent notifications."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from yap_studio.errors import ValidationError, YapError
from yap_studio.feedback.release import approval_record
from yap_studio.feedback.versions import VersionManifest, load_versions
from yap_studio.project.lock import pid_alive, project_lock
from yap_studio.receipt import sha256_file

TERMINAL = {"done", "failed", "cancelled"}
ACTIVE = {"pending", "delivered", "claimed"}
_DELIVERY_LEASE = 120
_STALL_AFTER = 600
_PROJECT_HISTORY_LIMIT = 20
_HASH_LOCK = threading.Lock()


def _media_key(path: Path) -> tuple[Path, int, int, int, int]:
    stat = path.stat()
    return path, stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns


@lru_cache(maxsize=64)
def _media_hash(key: tuple[Path, int, int, int, int]) -> str:
    return sha256_file(key[0])


def verified_approval(project: Path) -> VersionManifest | None:
    """Verify the newest version's approval, caching hashes until its media changes."""
    try:
        record = approval_record(project, verify_media=False)
        if record is None:
            return None
        versions = load_versions(project)
        if not versions or record["version"] != versions[-1].id:
            return None
        manifest = versions[-1]
        path = manifest.media_path.resolve()
        # Serialize cache misses so concurrent state and download requests hash only once.
        with _HASH_LOCK:
            key = _media_key(path)
            digest = _media_hash(key)
            if key != _media_key(path) or digest != manifest.media["sha256"]:
                return None
        return manifest
    except (YapError, OSError, ValueError):
        return None


class ConflictError(YapError):
    """A request cannot make the requested transition."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="home_conflict",
            suggested_action="Finish or cancel the current request first.",
        )


class UnknownRequestError(ValidationError):
    """The requested bridge record does not exist."""


def timestamp() -> str:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def workspace_dir(value: str | Path) -> Path:
    """Require a workspace with an inbox before serving or listening."""
    workspace = Path(value).expanduser().resolve()
    if not (workspace / "inbox").is_dir():
        exc = ValidationError("workspace", "This folder has no inbox folder.")
        exc.suggested_action = "Choose your workspace folder, or create inbox/ inside it."
        raise exc
    return workspace


def private_dir(folder: Path) -> None:
    """Create a private folder without following a directory symlink."""
    if folder.is_symlink():
        raise ValidationError("workspace", "Replace the linked home data folder with a folder.")
    folder.mkdir(mode=0o700, exist_ok=True)
    folder.chmod(0o700)


def directory(workspace: Path) -> Path:
    """Prepare the workspace's private bridge directories."""
    root = Path(workspace) / ".yap-studio"
    private_dir(root)
    private_dir(root / "homepage")
    private_dir(root / "homepage/requests")
    return root / "homepage"


def read_json(path: Path, default: Any = None) -> Any:
    """Read a small record, returning a fallback for a missing or invalid record."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, document: dict | list) -> None:
    """Replace a private record atomically and flush it before publication."""
    if not path.parent.exists():
        directory(path.parents[2])
    fd, temporary = tempfile.mkstemp(prefix=".home-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), 0o600)
            json.dump(document, output, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def writer(workspace: Path, *, launch: bool = False) -> Iterator[Path]:
    """Serialize bridge writers, with a separate lock for detached launches."""
    import fcntl

    root = directory(workspace)
    name = "launch.lock" if launch else "write.lock"
    fd = os.open(root / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def requests(workspace: Path) -> list[dict]:
    """Return complete request records, newest first."""
    rows = [
        read_json(path)
        for path in (Path(workspace) / ".yap-studio/homepage/requests").glob("r*.json")
        if re.fullmatch(r"r[1-9][0-9]*\.json", path.name)
    ]
    return sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: int(row["id"][1:]),
        reverse=True,
    )


def events(workspace: Path) -> list[dict]:
    """Return recent review events, newest first."""
    return list(reversed(read_json(Path(workspace) / ".yap-studio/homepage/events.json", [])))


def _workspace_project(workspace: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValidationError("project", "Choose a project from this workspace.")
    project = workspace / value
    if (
        project.parent != workspace / "projects"
        or Path(value).is_absolute()
        or value != f"projects/{project.name}"
        or project.parent.is_symlink()
        or project.is_symlink()
        or not project.is_dir()
    ):
        raise ValidationError("project", "Choose a real project directly inside projects/.")
    return project


def create_request(
    workspace: Path, kind: str, payload: dict, *, version_id: str | None = None
) -> dict:
    """Allocate a request id, validating messages and preventing duplicate video work."""
    if kind not in {"edit", "learn", "final", "message"} or not isinstance(payload, dict):
        raise ValidationError("request", "Choose an edit, learn, final or message request.")
    if kind == "edit" and "shape" in payload:
        if payload["shape"] not in ("vertical", "horizontal"):
            raise ValidationError("shape", "Choose vertical or horizontal.")
        if "captions" in payload and type(payload["captions"]) is not bool:
            raise ValidationError("captions", "Turn captions on or off.")
        if payload["shape"] == "horizontal":
            if payload.get("video_type") != "talking_head":
                raise ValidationError(
                    "video_type", "Horizontal voiceover and screen effects are coming soon. "
                    "Choose talking head or vertical.",
                )
            if payload.get("broll_sources") or payload.get("text_hook"):
                raise ValidationError(
                    "shape", "Horizontal titles, cards and B-roll are coming soon. "
                    "Remove them or choose vertical.",
                )
            payload = {**payload, "captions": payload.get("captions", False)}
    project = None
    if kind == "final":
        if set(payload) != {"project"}:
            raise ValidationError("project", "Choose a project from this workspace.")
        project = _workspace_project(Path(workspace).resolve(), payload.get("project"))
        if (project / "project.json").is_symlink() or not (project / "project.json").is_file():
            raise ValidationError("project", "Choose a project with a project.json file.")
    elif kind == "message":
        if set(payload) != {"project", "text"}:
            raise ValidationError(
                "message", "Send a message and its project, or null for the workspace."
            )
        text = payload["text"]
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValidationError("text", "Write a message of 1 to 2000 characters.")
        if payload["project"] is not None:
            _workspace_project(Path(workspace).resolve(), payload["project"])
        payload = {"project": payload["project"], "text": text.strip()}
    with project_lock(project) if project else nullcontext(), writer(workspace) as root:
        previous = requests(workspace)
        if kind == "edit" and any(
            row["kind"] == "edit" and row["state"] in ACTIVE for row in previous
        ):
            raise ConflictError("A video request is already in progress.")
        if project is not None:
            if any(
                row["kind"] == "final"
                and row["state"] in ACTIVE
                and row.get("payload", {}).get("project") == payload["project"]
                for row in previous
            ):
                raise ConflictError(
                    "A final video request for this project is already in progress."
                )
            versions = load_versions(project)
            if not versions:
                raise ConflictError("Save a story cut before asking for the final video.")
            newest = versions[-1]
            if version_id is not None and newest.id != version_id:
                raise ConflictError("A newer version is ready. Open its review before continuing.")
            if newest.media["source"].startswith("final/"):
                raise ConflictError("The newest version is already a final video. Review it first.")
        number = int(previous[0]["id"][1:]) + 1 if previous else 1
        row = {
            "schema": "yap-studio/home-request@1",
            "id": f"r{number}",
            "kind": kind,
            "payload": payload,
            "state": "pending",
            "created_at": timestamp(),
            "delivered_at": None,
            "ack": None,
        }
        write_json(root / "requests" / f"r{number}.json", row)
        return row


def _request(root: Path, identifier: str) -> tuple[Path, dict]:
    if re.fullmatch(r"r[1-9][0-9]*", identifier) is None:
        raise ValidationError("request", "Choose an existing request id, such as r1.")
    target = root / "requests" / f"{identifier}.json"
    row = read_json(target)
    if not isinstance(row, dict):
        raise UnknownRequestError("request", "This request does not exist. Refresh the home page.")
    return target, row


def relative_project(workspace: Path, value: str | Path) -> str:
    """Validate a project directory and return its workspace-relative name."""
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else Path(workspace) / candidate
    candidate = candidate.resolve()
    if (
        candidate.parent != (Path(workspace) / "projects").resolve()
        or candidate == (Path(workspace) / "projects").resolve()
    ):
        raise ValidationError("project", "Choose a project directly inside projects/.")
    return f"projects/{candidate.name}"


def project_history(value: object) -> list[str]:
    """Read valid project names from a stored rename history."""
    if not isinstance(value, list):
        return []
    return [
        name
        for name in value
        if isinstance(name, str)
        and re.fullmatch(r"projects/[^/\\\x00]+", name)
        and name not in {"projects/.", "projects/.."}
    ]


def ack(
    workspace: Path,
    identifier: str,
    state: str,
    *,
    project: str | None = None,
    message: str | None = None,
) -> dict:
    """Record the agent's progress without reopening terminal requests."""
    if state not in {"claimed", "done", "failed"}:
        raise ValidationError("state", "Choose claimed, done or failed.")
    if message is not None and (not isinstance(message, str) or len(message) > 2000):
        raise ValidationError("message", "Use a reply of at most 2000 characters.")
    relative = relative_project(workspace, project) if project is not None else None
    with writer(workspace) as root:
        target, row = _request(root, identifier)
        if row["state"] in TERMINAL and row["state"] != state:
            raise ValidationError("state", "This request has already finished.")
        previous = row.get("ack") or {}
        current_project = relative or previous.get("project")
        earlier_projects = project_history(previous.get("renamed_from"))
        if previous.get("project") and previous["project"] != current_project:
            latest = max(
                (other for other in requests(workspace)
                 if other["id"] != identifier
                 and (other.get("ack") or {}).get("project") == previous["project"]),
                key=lambda other: (other["ack"].get("at", ""), int(other["id"][1:])),
                default=None,
            )
            if latest is not None:
                earlier_projects.extend(project_history(latest["ack"].get("renamed_from")))
            earlier_projects.append(previous["project"])
        earlier_projects = [
            name for name in dict.fromkeys(earlier_projects) if name != current_project
        ][-_PROJECT_HISTORY_LIMIT:]
        row.update(
            state=state,
            ack={
                "state": state,
                "message": message,
                "project": current_project,
                "renamed_from": earlier_projects,
                "at": timestamp(),
            },
        )
        write_json(target, row)
        return row


def cancel(workspace: Path, identifier: str) -> dict:
    """Cancel a pending request or one whose session has stalled."""
    with writer(workspace) as root:
        target, row = _request(root, identifier)
        if row["state"] != "pending" and not stalled(workspace, row):
            raise ConflictError("The agent has already received this request.")
        row["state"] = "cancelled"
        write_json(target, row)
        return row


def resume(workspace: Path, identifier: str) -> dict:
    """Queue stalled work again, preserving the earlier session's acknowledgement."""
    with writer(workspace) as root:
        target, row = _request(root, identifier)
        if not stalled(workspace, row):
            raise ConflictError("This request has not stopped. Refresh the home page.")
        row.update(state="pending", resumes=row.get("resumes", 0) + 1)
        write_json(target, row)
        return row


def _project_locked(workspace: Path, value: str) -> bool:
    import fcntl

    try:
        project = _workspace_project(workspace, value)
    except ValidationError:
        return False
    if (project / ".yap").is_symlink():
        return True
    try:
        descriptor = os.open(project / ".yap/lock", os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    with os.fdopen(descriptor, "rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def stalled(workspace: Path, row: dict) -> bool:
    """Require ten quiet minutes and no live listener or project writer."""
    if row["state"] not in {"delivered", "claimed"} or listening(workspace):
        return False
    root = Path(workspace) / ".yap-studio/homepage"
    ack = row.get("ack") or {}
    activity = [value for value in (ack.get("at"), row.get("delivered_at")) if value]
    if not activity:
        return False
    for record in ("last-listener.json", "listener.json"):
        heard = read_json(root / record, {}).get("heartbeat_at")
        if heard:
            activity.append(heard)
    try:
        if any((datetime.now(UTC) - datetime.fromisoformat(value)).total_seconds() < _STALL_AFTER
               for value in activity):
            return False
    except (ValueError, TypeError):
        return False
    return not (ack.get("project") and _project_locked(workspace, ack["project"]))


def listening(workspace: Path) -> bool:
    """A listener is live only while its process and heartbeat are fresh."""
    row = read_json(Path(workspace) / ".yap-studio/homepage/listener.json", {})
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(row["heartbeat_at"])).total_seconds()
        return 0 <= age < 20 and pid_alive(row.get("pid"))
    except (ValueError, KeyError, TypeError):
        return False


def _matches_listener(pid: int, workspace: Path) -> bool:
    """Check the command and workspace before signaling an earlier listener."""
    if type(pid) is not int or pid <= 0 or pid == os.getpid():
        return False
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], text=True, capture_output=True, timeout=5
        )
        command = result.stdout.strip()
        if (
            result.returncode
            or re.search(
                r"(?:^|\s)(?:\S*/)?(?:yap-studio|yap_studio)\s+home\s+wait(?:\s|$)", command
            )
            is None
        ):
            return False
        if re.search(r"--workspace(?:=|\s+)" + re.escape(str(workspace)) + r"(?=\s|$)", command):
            return True
        if sys.platform == "darwin":
            cwd = subprocess.run(
                ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                text=True,
                capture_output=True,
                timeout=5,
            )
            base = next(
                (Path(line[1:]) for line in cwd.stdout.splitlines() if line.startswith("n")), None
            )
        else:
            base = Path(f"/proc/{pid}/cwd").resolve(strict=True)
        supplied = re.search(
            r"(?:^|\s)--workspace(?:=|\s+)(.+?)(?=\s+--[\w-]+(?:\s|=|$)|$)", command
        )
        value = supplied[1] if supplied else "."
        if value.startswith(("'", '"')):
            value = shlex.split(value)[0]
        return base is not None and (base / value).resolve() == workspace
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False


def project_dirs(workspace: Path) -> list[Path]:
    """List real project directories without leaving the workspace."""
    projects = Path(workspace) / "projects"
    if not projects.is_dir() or projects.is_symlink():
        return []
    return sorted(
        (
            p
            for p in projects.iterdir()
            if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")
        ),
        key=lambda p: p.name,
    )


def _history(workspace: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    notes, approvals = [], []
    for project in project_dirs(workspace):
        relative = f"projects/{project.name}"
        notes.extend(
            (relative, p.stem)
            for p in sorted(
                (project / "feedback").glob("b*.json"),
                key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0,
            )
            if re.fullmatch(r"b[1-9][0-9]*", p.stem) and not p.is_symlink()
        )
        try:
            approved = verified_approval(project)
        except (YapError, OSError, ValueError):
            approved = None
        if approved:
            approvals.append((relative, approved.id))
    return notes, approvals


def _seed(notes: list[tuple[str, str]], approvals: list[tuple[str, str]]) -> dict:
    seen: dict = {}
    for key, values in (("batches", notes), ("approvals", approvals)):
        for project, identifier in values:
            seen.setdefault(project, {"batches": [], "approvals": []})[key].append(identifier)
    return seen


def _poll(workspace: Path, root: Path) -> dict | None:
    rows = list(reversed(requests(workspace)))
    # Pending work takes priority over a delivery whose agent never claimed it.
    for state in ("pending", "delivered"):
        for row in rows:
            if row["state"] != state:
                continue
            redelivered = state == "delivered"
            if redelivered:
                if row.get("ack") is not None:
                    continue
                try:
                    delivered = datetime.fromisoformat(row["delivered_at"])
                    age = (datetime.now(UTC) - delivered).total_seconds()
                except (KeyError, TypeError, ValueError):
                    continue
                if age <= _DELIVERY_LEASE:
                    continue
            row.update(
                state="delivered",
                delivered_at=timestamp(),
                deliveries=row.get("deliveries", 1 if redelivered else 0) + 1,
            )
            write_json(root / "requests" / f"{row['id']}.json", row)
            next_action = {
                "edit": "Make the requested video from these choices and acknowledge this request.",
                "learn": (
                    "Learn the creator's style from their examples and acknowledge this request."
                ),
                "final": "Build the final video for this project from its approved story cut "
                "and acknowledge this request.",
                "message": "Read the creator's message, act on it, and reply in one or two plain "
                "sentences with yap-studio home ack ID --state done --message TEXT.",
            }[row["kind"]]
            result = {"event": "request", "request": row, "next": next_action}
            if row.get("ack") is not None:
                result["resumed"] = True
                result["next"] = (
                    "An earlier session claimed this request and stopped before finishing. "
                    "Open the project in the previous ack, if present. Check its receipts and "
                    "yap-studio status to see what already finished, then resume from there. "
                    "Do not redo finished or paid steps. Ack this request claimed and carry on."
                )
            elif redelivered:
                result["redelivered"] = True
                result["next"] = (
                    "This request came back because it was never claimed. "
                    "If you are already working on it, claim it and carry on. " + next_action
                )
            return result
    seen = read_json(root / "seen.json", {})
    notes, approvals = _history(workspace)
    for kind, key, detail, values in (
        ("notes", "batches", "batch", notes),
        ("approved", "approvals", "version", approvals),
    ):
        for project, identifier in values:
            held = seen.setdefault(project, {"batches": [], "approvals": []})
            if identifier in held[key]:
                continue
            held[key].append(identifier)
            previous = list(reversed(events(workspace)))
            number = int(previous[-1]["id"][1:]) + 1 if previous else 1
            now = timestamp()
            row = {
                "id": f"e{number}",
                "kind": kind,
                "project": project,
                "detail": {detail: identifier},
                "created_at": now,
                "delivered_at": now,
            }
            write_json(root / "events.json", [*previous, row][-50:])
            write_json(root / "seen.json", seen)
            return {"event": kind, "project": project, detail: identifier}
    return None


def wait(workspace: Path, timeout: float = 3000) -> dict:
    """Deliver one request or new review event, maintaining a private heartbeat."""
    if not 1 <= timeout <= 3600:
        raise ValidationError("timeout", "Choose a timeout from 1 to 3600 seconds.")
    workspace = workspace_dir(workspace)
    ours = {"pid": os.getpid(), "started_at": timestamp(), "heartbeat_at": timestamp()}
    handlers = {}
    stopping = False

    def interrupted(_signum, _frame):
        nonlocal stopping
        stopping = True

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handlers[signum] = signal.signal(signum, interrupted)
        with writer(workspace) as root:
            previous = read_json(root / "listener.json", {})
            if _matches_listener(previous.get("pid"), workspace):
                try:
                    os.kill(previous["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            write_json(root / "listener.json", ours)
            write_json(root / "last-listener.json", {"heartbeat_at": ours["heartbeat_at"]})
            if not (root / "seen.json").exists():
                write_json(root / "seen.json", _seed(*_history(workspace)))
        deadline = time.monotonic() + timeout
        heartbeat = time.monotonic()
        while not stopping:
            with writer(workspace) as root:
                if read_json(root / "listener.json") != ours:
                    return {"event": "timeout"}
                if time.monotonic() - heartbeat >= 5:
                    ours["heartbeat_at"] = timestamp()
                    write_json(root / "listener.json", ours)
                    write_json(root / "last-listener.json", {"heartbeat_at": ours["heartbeat_at"]})
                    heartbeat = time.monotonic()
                result = _poll(workspace, root)
            if result:
                return result
            remaining = deadline - time.monotonic()
            if stopping or remaining <= 0:
                return {"event": "timeout"}
            threading.Event().wait(min(1, remaining))
        return {"event": "timeout"}
    finally:
        try:
            with writer(workspace) as root:
                if read_json(root / "listener.json") == ours:
                    write_json(root / "last-listener.json", {"heartbeat_at": timestamp()})
                    (root / "listener.json").unlink(missing_ok=True)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
