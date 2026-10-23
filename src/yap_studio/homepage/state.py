"""Read-only creator state assembled from the engine's existing records."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from yap_studio import background, ffmpeg, video_type
from yap_studio.errors import ValidationError, YapError
from yap_studio.feedback.versions import VersionManifest, load_versions
from yap_studio.homepage import bridge
from yap_studio.project.status import status
from yap_studio.style.answers import load_answers, questions

OFFERED_TYPES = ("talking_head", "voiceover")
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm"})
MEMO_SUFFIXES = frozenset({".m4a", ".wav", ".mp3", ".aac", ".caf", ".aif", ".aiff", ".flac"})


def phase(
    snapshot: dict, versions: list[dict], *, newer_feedback: bool = False, approved: bool = False
) -> dict:
    """Map engine status and saved review facts to one creator-facing phase."""
    result = {"phase": "starting", "step": None, "message": None}
    run = snapshot.get("run") or {}
    stages = snapshot.get("stages", [])

    def when(value: str | None) -> datetime:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except (ValueError, AttributeError):
            return datetime.min.replace(tzinfo=UTC)

    running = max(
        (row for row in stages if row["state"] == "running"),
        key=lambda row: when((row.get("progress") or {}).get("started")),
        default=None,
    )
    version_time = when(versions[-1].get("created_at")) if versions else when(None)
    stage_time = when(((running or {}).get("progress") or {}).get("started"))
    run_time = when(
        run.get("finished") or (run.get("progress") or {}).get("started") or run.get("started")
    )
    # Keep the existing priority on ties, including failures recorded in the same second.
    if run_time < max(stage_time, version_time):
        run = {}
    if stage_time < version_time:
        running = None
    if approved:
        result["phase"] = "approved"
    elif run.get("state") in {"failed", "cancelled", "stopped", "lost"}:
        result.update(
            phase="stopped",
            message=(run.get("error") or {}).get("message")
            or "The video stopped. Ask your agent to continue.",
        )
    elif run.get("state") == "running":
        result.update(
            phase="building",
            step=(running or {}).get("reason")
            or (run.get("progress") or {}).get("step")
            or run.get("step")
            or "Working",
        )
    elif running:
        stage = running["name"]
        result.update(
            phase=(
                "transcribing" if stage == "ingest" else "cutting" if stage == "cut" else "building"
            ),
            step=(running.get("progress") or {}).get("step") or running.get("reason") or stage,
        )
    elif newer_feedback:
        result["phase"] = "revising"
    elif versions:
        result["phase"] = "final_ready" if versions[-1]["final"] else "cut_ready"
    elif any(row["name"] == "ingest" and row["state"] == "done" for row in stages):
        result["phase"] = "story"
    return result


def probe(path: Path) -> tuple[str, float | None, int | None, int | None]:
    """Classify media by its streams, falling back to the inbox suffix rules."""
    fallback = "memo" if path.suffix.lower() in MEMO_SUFFIXES else "take"
    try:
        info = ffmpeg.probe(path, count_frames=False)
        duration = info.duration_s
        return (
            "take" if info.has_video else "memo" if info.has_audio else fallback,
            duration if duration is not None and math.isfinite(duration) else None,
            info.width, info.height,
        )
    except (YapError, OSError, ValueError):
        return fallback, None, None, None


def inbox_files(workspace: Path) -> tuple[list[Path], int]:
    """Return unused visible regular media using exact ledger identities."""
    inbox = workspace / "inbox"
    available = [
        p
        for p in inbox.iterdir()
        if p.is_file()
        and not p.is_symlink()
        and not p.name.startswith(".")
        and p.suffix.lower() in VIDEO_SUFFIXES | MEMO_SUFFIXES
    ]
    ledger = inbox / ".used.json"
    used = (
        bridge.read_json(ledger, None)
        if ledger.exists()
        else {"schema": "yap-studio/inbox-used@1", "takes": []}
    )
    keys = {"file", "bytes", "mtime_ns", "project", "used_at"}
    if (
        not isinstance(used, dict)
        or set(used) != {"schema", "takes"}
        or used.get("schema") != "yap-studio/inbox-used@1"
        or not isinstance(used.get("takes"), list)
        or any(
            not isinstance(row, dict)
            or set(row) != keys
            or any(not isinstance(row[key], str) for key in ("file", "project", "used_at"))
            or type(row["bytes"]) is not int
            or type(row["mtime_ns"]) is not int
            for row in used["takes"]
        )
    ):
        raise ValidationError("inbox", "Fix inbox/.used.json before starting a video.")
    identities = {(row["file"], row["bytes"], row["mtime_ns"]) for row in used["takes"]}
    unused = [
        p for p in available if (p.name, p.stat().st_size, p.stat().st_mtime_ns) not in identities
    ]
    return sorted(unused, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True), (
        len(available) - len(unused)
    )


def public_value(value: Any, workspace: Path) -> Any:
    """Keep filesystem paths relative even inside engine messages and answer values."""
    if isinstance(value, dict):
        return {key: public_value(child, workspace) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_value(child, workspace) for child in value]
    if not isinstance(value, str):
        return value
    text = value.replace(str(workspace) + "/", "").replace(str(workspace), workspace.name)
    if text.startswith(("http://", "https://", "/api/", "/thumb/", "/static/")):
        return text
    return re.sub(r"(?<![\w:])/[^\s'\"<>;,)}]+", lambda m: Path(m[0]).name, text)


def _approved_path(project: Path, approved: VersionManifest | None) -> Path | None:
    if approved is None or not approved.media["source"].startswith("final/"):
        return None
    target = approved.media_path.resolve()
    return target if target.is_relative_to(project.resolve()) and target.is_file() else None


def _messages(requests: list[dict], project: str | None) -> list[dict]:
    messages = []
    for row in reversed(requests):
        if row["kind"] != "message" or row["payload"]["project"] != project:
            continue
        ack = row.get("ack") or {}
        messages.append(
            {
                "id": row["id"],
                "text": row["payload"]["text"],
                "state": row["state"],
                "created_at": row["created_at"],
                "reply": ack.get("message"),
                "replied_at": ack.get("at"),
            }
        )
    return messages


def _newer_feedback(batch: Path, version: VersionManifest) -> bool:
    record = bridge.read_json(batch, {})
    created = record.get("created_at")
    if isinstance(created, str):
        try:
            batch_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
            version_time = datetime.fromisoformat(version.created_at.replace("Z", "+00:00"))
            if batch_time != version_time:
                return batch_time > version_time
        except (ValueError, TypeError):
            pass
    # Both engine records use whole seconds. Resolve ties with immutable file times.
    info = batch.stat()
    batch_time_ns = getattr(info, "st_birthtime_ns", None)
    if batch_time_ns is None:
        batch_time_ns = (
            int(info.st_birthtime * 1e9) if hasattr(info, "st_birthtime") else (info.st_mtime_ns)
        )
    return batch_time_ns > (version.path / "manifest.json").stat().st_mtime_ns


class HomeState:
    """Read workspace state and cache media facts by file identity."""

    def __init__(self, workspace: Path, *, phone: dict | None = None) -> None:
        self.workspace = bridge.workspace_dir(workspace)
        self.phone = phone
        self._metadata: dict[tuple, tuple[str, float | None, int | None, int | None]] = {}
        self._lock = threading.RLock()

    def metadata(self, path: Path) -> tuple[str, float | None, int | None, int | None]:
        """Probe a take only when its name, size or modification time changes."""
        info = path.stat()
        key = (path.name, info.st_size, info.st_mtime_ns)
        with self._lock:
            if key not in self._metadata:
                self._metadata[key] = probe(path)
            return self._metadata[key]

    def take_entry(self, path: Path) -> dict:
        """Describe one take using browser routes and a name only."""
        info = path.stat()
        kind, duration, width, height = self.metadata(path)
        return {
            "name": path.name,
            "kind": kind,
            "duration_s": duration,
            "width": width,
            "height": height,
            "bytes": info.st_size,
            "modified": datetime.fromtimestamp(info.st_mtime, UTC).isoformat(),
            "thumb": f"/thumb/{quote(path.name, safe='')}" if kind == "take" else None,
        }

    def thumbnail(self, name: str) -> Path | None:
        """Extract a cached JPEG from a current unused video take."""
        paths, _ = inbox_files(self.workspace)
        source = next((p for p in paths if p.name == name), None)
        if source is None:
            return None
        kind, duration, _, _ = self.metadata(source)
        if kind != "take":
            return None
        info = source.stat()
        key = hashlib.sha256(f"{name}:{info.st_size}:{info.st_mtime_ns}".encode()).hexdigest()
        with self._lock:
            root = bridge.directory(self.workspace) / "thumbs"
            bridge.private_dir(root)
            target = root / f"{key}.jpg"
            if not target.is_file():
                when = min(1.0, duration * 0.1) if duration else 1.0
                result = ffmpeg.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-ss",
                        str(when),
                        "-i",
                        str(source),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=-2:360",
                        "-threads",
                        "1",
                        "-f",
                        "image2pipe",
                        "-c:v",
                        "mjpeg",
                        "-",
                    ],
                    timeout_s=30,
                    binary=True,
                )
                fd, temporary = tempfile.mkstemp(prefix=".thumb-", dir=root)
                try:
                    with os.fdopen(fd, "wb") as output:
                        os.fchmod(output.fileno(), 0o600)
                        output.write(result.stdout)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            return target

    def approved_file(self, name: str) -> Path | None:
        """Resolve only an approved final inside a real workspace project."""
        project = next((p for p in bridge.project_dirs(self.workspace) if p.name == name), None)
        if project is None:
            return None
        return _approved_path(project, bridge.verified_approval(project))

    def _review(self, project: Path) -> dict | None:
        for record in reversed(background.records("review")):
            saved = background.read_record(record)
            if saved is None or background.command_matches(saved.get("pid"), record) is not True:
                continue
            media = saved.get("media", [])
            if not any(Path(p).resolve() == project for p in media if isinstance(p, str)):
                continue
            document = saved.get("document", {})
            videos = document.get("videos", [])
            row = next((row for row in videos if row.get("url")), None)
            if row:
                return {
                    "url": row.get("local_url") or row["url"],
                    "phone_url": (document.get("phone") or {}).get("url"),
                    "phone_via": (document.get("phone") or {}).get("via"),
                }
        return None

    def _project(self, project: Path, requests: list[dict]) -> dict:
        raw = bridge.read_json(project / "project.json", {})
        snapshot = status(project)
        manifests = load_versions(project)
        versions = [
            {
                "id": v.id,
                "number": v.number,
                "note": v.note,
                "created_at": v.created_at,
                "final": v.media["source"].startswith("final/"),
            }
            for v in manifests
        ]
        batches = [
            p
            for p in (project / "feedback").glob("b*.json")
            if re.fullmatch(r"b[1-9][0-9]*\.json", p.name) and not p.is_symlink()
        ]
        newer = (
            any(_newer_feedback(p, manifests[-1]) for p in batches) if manifests else bool(batches)
        )
        approved = bridge.verified_approval(project)
        approved_file = _approved_path(project, approved)
        relative = f"projects/{project.name}"
        earlier_names = list(
            dict.fromkeys(
                Path(name).name
                for row in reversed(requests)
                if (row.get("ack") or {}).get("project") == relative
                for name in bridge.project_history(row["ack"].get("renamed_from"))
                if name != relative
            )
        )
        request = next(
            (row["id"] for row in requests if (row.get("ack") or {}).get("project") == relative),
            None,
        )
        final_request = next(
            (
                row
                for row in requests
                if row["kind"] == "final" and row.get("payload", {}).get("project") == relative
            ),
            None,
        )
        return {
            "name": project.name,
            "renamed_from": earlier_names,
            "title": project.name,
            "shape": raw.get("shape", "vertical"),
            "dir": relative,
            "created_at": raw.get("created_at")
            or datetime.fromtimestamp((project / "project.json").stat().st_mtime, UTC).isoformat(),
            "takes": [Path(row["path"]).name for row in raw.get("takes", [])],
            "video": snapshot.get(
                "video", {"type": raw.get("video_type"), "broll_sources": raw.get("broll_sources")}
            ),
            **phase(snapshot, versions, newer_feedback=newer, approved=bool(approved_file)),
            "stages": [
                {key: row.get(key) for key in ("name", "state", "reason")}
                for row in snapshot["stages"]
            ],
            "versions": versions,
            "review": self._review(project),
            "approved": {
                "version": approved.id,
                "file": str(approved_file.relative_to(self.workspace)),
                "download": f"/api/download/{quote(project.name, safe='')}",
            }
            if approved_file
            else None,
            "request": request,
            "messages": _messages(requests, relative),
            "final_request": {"id": final_request["id"], "state": final_request["state"]}
            if final_request
            else None,
        }

    def snapshot(self, *, on_this_mac: bool = False) -> dict:
        """Return the browser contract without any absolute filesystem paths."""
        paths, used = inbox_files(self.workspace)
        answer_path = self.workspace / "style/answers.json"
        answers = load_answers(answer_path) if answer_path.is_file() else None
        style_path = self.workspace / "style/my-style.json"
        style = bridge.read_json(style_path)
        requests = [
            {**row, "stalled": bridge.stalled(self.workspace, row)}
            for row in bridge.requests(self.workspace)
        ]
        projects = [
            self._project(p, requests)
            for p in bridge.project_dirs(self.workspace)
            if (p / "project.json").is_file()
        ]
        projects.sort(key=lambda p: (p["created_at"], p["dir"]), reverse=True)
        video_questions = copy.deepcopy(video_type.QUESTIONS)
        video_questions[0]["choices"] = [
            row for row in video_questions[0]["choices"] if row["value"] in OFFERED_TYPES
        ]
        default = video_type.load_default()
        if default and default["video_type"] not in OFFERED_TYPES:
            default = None
        result = {
            "schema": "yap-studio/home@1",
            "workspace": self.workspace.name,
            "listening": bridge.listening(self.workspace),
            "on_this_mac": on_this_mac,
            "first_run": not projects and not style_path.exists() and not answer_path.exists(),
            "video_type": {"default": default, "questions": video_questions},
            "answers": {"saved": answers is not None, "questions": questions(answers)},
            "style": {
                "saved": style is not None,
                "name": style.get("name") if style else None,
                "hook_style": style.get("hook", {}).get("style") if style else None,
                "updated_at": datetime.fromtimestamp(style_path.stat().st_mtime, UTC).isoformat()
                if style is not None
                else None,
            },
            "inbox": {"takes": [self.take_entry(p) for p in paths], "used": used},
            "requests": requests[:20],
            "messages": _messages(requests, None),
            "events": bridge.events(self.workspace)[:20],
            "projects": projects,
            "phone": self.phone,
        }
        return public_value(result, self.workspace)
