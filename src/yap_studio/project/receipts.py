"""Snapshot hashes, durable output journals, and recoverable stage receipts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import re
import shutil
import time
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from yap_studio import __version__, ffmpeg, paths, receipt
from yap_studio.errors import InputError, ValidationError, YapError
from yap_studio.project.lock import (
    ConflictError,
    _write_json,
    pid_alive,
    project_lock,
    timestamp,
)


class AudioStreamMissingError(InputError):
    """Report a selected audio stream that is absent or has no packets."""


def projection_sha256(obj: Any) -> str:
    """Hash a canonical JSON input projection, excluding irrelevant settings."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def failed_assert_lines(asserts: list[dict[str, Any]]) -> list[str]:
    """Format failed checks with the values that explain each failure."""
    def value(item: Any) -> str:
        if isinstance(item, (dict, list)):
            return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return str(item)

    lines = []
    for item in asserts:
        if item.get("pass") is not False:
            continue
        if item.get("id") == "bleep.words":
            audible = item.get("audible", [])
            entry = (audible or item.get("unmasked", []))[0]
            minutes, seconds = divmod(int(entry["time_s"]), 60)
            detail = ("is still audible" if audible else f"is still shown in {entry['where']}")
            lines.append(f"Bleep check failed: a swear word at {minutes}:{seconds:02} {detail}")
            continue
        details = [f"{key} {value(item[key])}" for key in ("expected", "actual") if key in item]
        lines.append(str(item.get("id", "check")) +
                     (f" ({', '.join(details)})" if details else ""))
    return lines


def audio_stream_sha256(path: Path, ordinal: int) -> str:
    """Hash selected encoded audio, its start offset, and packet timing gaps."""
    raw = ffmpeg.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    )
    data = json.loads(raw.stdout)
    streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"]
    if ordinal < 0 or ordinal >= len(streams):
        raise AudioStreamMissingError(
            str(path), f"audio stream {ordinal} is missing; choose another stream",
        )
    selected = streams[ordinal]
    format_info = data.get("format", {})
    offset_s = round(
        float(selected.get("start_time") or 0) - float(format_info.get("start_time") or 0), 6,
    )
    duration = max(float(format_info.get("duration") or 0), float(selected.get("duration") or 0))
    result = ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{ordinal}",
            "-c", "copy", "-f", "framehash", "-hash", "sha256", "-",
        ],
        timeout_s=max(ffmpeg.DEFAULT_TIMEOUT_S, 120 + 1.5 * duration),
    )
    packets = hashlib.sha256()
    time_base: Fraction | None = None
    previous_end: Fraction | None = None
    gaps: list[list[int | float]] = []
    count = 0
    for line in result.stdout.splitlines():
        if line.startswith("#tb 0:"):
            time_base = Fraction(line.split(":", 1)[1].strip())
        elif line and not line.startswith("#"):
            if time_base is None:
                raise InputError(
                    str(path), "audio packet time base is missing; re-export and retry",
                )
            # Some codecs append side-data hashes after the six packet fields.
            _, _, pts, packet_duration, _, digest = line.split(",")[:6]
            start = int(pts) * time_base
            if previous_end is not None:
                gap = start - previous_end
                # Container timestamp rounding is smaller than a real timing shift.
                if abs(gap) > Fraction(2, 1000):
                    gaps.append([count, float(round(gap, 3))])
            previous_end = start + int(packet_duration) * time_base
            packets.update(bytes.fromhex(digest.strip()))
            count += 1
    if not count:
        raise AudioStreamMissingError(
            str(path), "audio stream has no packets; re-export the sound and retry",
        )
    return projection_sha256({"packets": packets.hexdigest(), "offset_s": offset_s, "gaps": gaps})


def _identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "ino": stat.st_ino, "dev": stat.st_dev}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError) as exc:
        raise InputError(str(path), "cannot read project bookkeeping; inspect and retry") from exc
    if not isinstance(data, dict):
        raise InputError(str(path), "expected a JSON object")
    return data


class IdentityCache:
    """Hash large takes once per stable size/time/inode/device identity."""

    def __init__(self, project_dir: Path, *, lock_held: bool = False) -> None:
        """Reuse a caller's project lock only while that lock remains held."""
        self.project_dir = Path(project_dir)
        self.path = self.project_dir / ".yap" / "identity.json"
        self._lock_held = lock_held

    def known_entry(self, path: Path) -> dict[str, Any] | None:
        """Read a matching recorded hash and identity without hashing, locking or writing."""
        try:
            real = path.resolve()
            identity = _identity(real)
            entries = _read_json(self.path) or {}
            previous = entries.get(str(real))
            if (isinstance(previous, dict)
                    and all(previous.get(name) == value for name, value in identity.items())
                    and isinstance(previous.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", previous["sha256"])):
                return {"sha256": previous["sha256"],
                        **{name: previous[name] for name in identity}}
        except (OSError, ValueError, YapError):
            pass
        return None

    def known_sha256(self, path: Path) -> str | None:
        """Read a matching file identity without hashing, locking or writing."""
        entry = self.known_entry(path)
        return entry["sha256"] if entry is not None else None

    def sha256(self, path: Path) -> str:
        """Return a cached hash, retrying when the file changes during hashing."""
        with nullcontext() if self._lock_held else project_lock(self.project_dir):
            real = paths.validate_input(path)
            entries = _read_json(self.path) or {}
            key = str(real)
            previous = entries.get(key)
            try:
                identity = _identity(real)
                if isinstance(previous, dict) and all(
                    previous.get(name) == value for name, value in identity.items()
                ) and isinstance(previous.get("sha256"), str):
                    return previous["sha256"]
                for _ in range(3):
                    digest = receipt.sha256_file(real)
                    after = _identity(real)
                    if after == identity:
                        entries[key] = {"sha256": digest, **identity}
                        if isinstance(previous, dict) and all(
                            previous.get(name) == value for name, value in identity.items()
                        ) and isinstance(previous.get("audio"), dict):
                            entries[key]["audio"] = previous["audio"]
                        _write_json(self.path, entries)
                        return digest
                    identity = after
            except OSError as exc:
                raise InputError(
                    str(path), "file changed or disappeared; re-read and retry",
                ) from exc
            raise InputError(str(path), "file kept changing while hashing; wait and retry")

    def audio_sha256(self, path: Path, ordinal: int) -> str:
        """Cache a selected audio hash, retrying when the file changes during hashing."""
        with nullcontext() if self._lock_held else project_lock(self.project_dir):
            real = paths.validate_input(path)
            entries = _read_json(self.path) or {}
            key = str(real)
            stream_key = str(ordinal)
            previous = entries.get(key)
            try:
                identity = _identity(real)
                if isinstance(previous, dict) and all(
                    previous.get(name) == value for name, value in identity.items()
                ) and isinstance(previous.get("audio"), dict):
                    cached = previous["audio"].get(stream_key)
                    if isinstance(cached, str):
                        return cached
                for _ in range(3):
                    digest = audio_stream_sha256(real, ordinal)
                    after = _identity(real)
                    if after == identity:
                        entry: dict[str, Any] = dict(identity)
                        audio = {}
                        if isinstance(previous, dict) and all(
                            previous.get(name) == value for name, value in identity.items()
                        ):
                            if isinstance(previous.get("sha256"), str):
                                entry["sha256"] = previous["sha256"]
                            if isinstance(previous.get("audio"), dict):
                                audio = dict(previous["audio"])
                        audio[stream_key] = digest
                        entry["audio"] = audio
                        entries[key] = entry
                        _write_json(self.path, entries)
                        return digest
                    identity = after
            except OSError as exc:
                raise InputError(
                    str(path), "file changed or disappeared; re-read and retry",
                ) from exc
            raise InputError(str(path), "file kept changing while hashing; wait and retry")


def read_receipt(stage_dir: Path) -> dict | None:
    """Read the final stage receipt, ignoring an unfinished pending journal."""
    data = _read_json(stage_dir / "receipt.json")
    if data is not None and data.get("schema") != "yap-studio/stage-receipt@1":
        raise InputError(str(stage_dir / "receipt.json"), "not a versioned stage receipt")
    return data


def read_pending(stage_dir: Path) -> dict | None:
    """Read a stage journal left by an interrupted run."""
    return _read_json(stage_dir / "receipt.pending.json")


def receipt_output_paths(project_dir: Path, data: dict[str, Any]) -> list[Path]:
    """Resolve every recorded output within the project before reading any file."""
    root = project_dir.resolve()
    targets = []
    for output in data.get("outputs", []):
        target = (root / output["path"]).resolve()
        if not target.is_relative_to(root):
            raise InputError("receipt.json", "receipt names a file outside the project")
        targets.append(target)
    return targets


def retained_outputs(project_dir: Path, *records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep only on-disk outputs whose hash appears in a receipt or journal."""
    kept = {}
    for record in records:
        targets = receipt_output_paths(project_dir, record)
        for output, target in zip(record.get("outputs", []), targets, strict=True):
            if not target.is_file():
                continue
            digest = receipt.sha256_file(target)
            if digest not in (output.get("sha256"), output.get("old_sha256"),
                              output.get("new_sha256")):
                continue
            kept[output["path"]] = {"path": output["path"], "sha256": digest,
                                    "bytes": target.stat().st_size}
    return kept


class StageRun:
    """Journal short write transactions while rendering outside the project lock.

    Call up_to_date with the snapshot projections before entering to skip an unchanged
    run. Rendering must consume the same snapshot whose hashes record_inputs receives.
    """

    def __init__(self, project_dir: Path, stage: str) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_-]*", stage) is None:
            raise YapError("Invalid stage folder.", code="invalid_stage",
                           suggested_action="Choose a registered stage name.")
        self.project_dir = Path(project_dir).resolve()
        self.stage = stage
        self.stage_dir = self.project_dir / stage
        self.receipt_path = self.stage_dir / "receipt.json"
        self.pending_path = self.stage_dir / "receipt.pending.json"
        self.progress_path = self.stage_dir / "progress.json"
        self.started = ""
        self.inputs: dict[str, str] = {}
        self.outputs: dict[str, dict[str, Any]] = {}
        self.owner_receipts: tuple[Path, ...] = ()
        self._pending: dict[str, Any] = {}
        self._active = False
        self._finished = False
        self._operation: Any = None
        self._phase_name = ""
        self._phase_started = 0.0
        self._run_started = 0.0
        self._phases: list[dict[str, str | float]] = []
        self._previous_receipt: bytes | None = None
        self._previous_pending: bytes | None = None
        self._replaced_outputs = False

    def up_to_date(self, inputs: dict[str, str]) -> bool:
        """Check for a verified, unchanged run without mutating its bookkeeping."""
        data = read_receipt(self.stage_dir)
        if data is None:
            return False
        try:
            targets = receipt_output_paths(self.project_dir, data)
        except InputError:
            return False
        if (data.get("status") != "verified" or data.get("inputs") != inputs
                or self.pending_path.exists() or self.progress_path.exists()):
            return False
        return all(
            target.is_file()
            and receipt.sha256_file(target) == output["sha256"]
            for output, target in zip(data.get("outputs", []), targets, strict=True)
        )

    def __enter__(self) -> StageRun:
        if self.started:
            raise RuntimeError("Create a new StageRun for each attempt")
        self._operation = paths.operation()
        self._operation.__enter__()
        try:
            with project_lock(self.project_dir):
                project_path = self.project_dir / "project.json"
                if project_path.exists():
                    from yap_studio.project.status import ProjectContext

                    paths.validate_input(project_path)
                    context = ProjectContext(self.project_dir)
                    context.identity = IdentityCache(self.project_dir, lock_held=True)
                    for take in context.project.takes:
                        # Reserve originals even when only a verified copy is available.
                        paths.protect_input(self.project_dir / take.path)
                    for take in context.project.takes:
                        paths.validate_input(context.take_path(take.id))
                paths.validate_output_dir(self.stage_dir)
                progress = _read_json(self.progress_path)
                if progress and pid_alive(progress.get("pid")):
                    raise YapError(
                        f"{self.stage} is already running.", code="stage_running",
                        error_type="project_conflict",
                        suggested_action="Wait for the stage or stop its owning process and retry.",
                    )
                previous = read_receipt(self.stage_dir)
                pending = _read_json(self.pending_path)
                if pending is not None:
                    receipt_output_paths(self.project_dir, pending)
                    self._previous_pending = self.pending_path.read_bytes()
                if previous is not None:
                    targets = receipt_output_paths(self.project_dir, previous)
                    if pending is not None:
                        # A crash after the final receipt write leaves both files.
                        if previous.get("started") != pending.get("started") or not all(
                            output_path.is_file()
                            and receipt.sha256_file(output_path) == output["sha256"]
                            for output, output_path in zip(
                                previous.get("outputs", []), targets, strict=True,
                            )
                        ):
                            raise InputError(str(self.pending_path), "receipt and journal conflict")
                        self.pending_path.unlink()
                    paths.validate_output(self.receipt_path)
                    paths.validate_output(self.pending_path)
                    self._previous_receipt = self.receipt_path.read_bytes()
                    self._previous_pending = None
                    os.replace(self.receipt_path, self.pending_path)
                    pending = previous
                self.started = timestamp()
                self._pending = pending or {"outputs": []}
                self._pending.update({"schema": "yap-studio/stage-pending@1", "stage": self.stage,
                                      "started": self.started, "inputs": {}})
                _write_json(self.pending_path, self._pending)
                self._active = True
                self._write_progress("starting", 0.0)
                self._phase_name = "starting"
                self._run_started = self._phase_started = time.perf_counter()
            return self
        except BaseException:
            self._operation.__exit__(None, None, None)
            raise

    def _require_active(self) -> None:
        if not self._active or self._finished:
            raise RuntimeError("StageRun must be entered and unfinished")

    def record_inputs(self, name: str, sha256: str) -> None:
        """Record the hash of an input snapshot that the stage actually read."""
        self._require_active()
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValidationError("sha256", "must be a lowercase SHA-256 digest")
        if name in self.inputs and self.inputs[name] != sha256:
            raise ConflictError(Path(name))
        self.inputs[name] = sha256
        with project_lock(self.project_dir):
            self._pending["inputs"] = dict(self.inputs)
            _write_json(self.pending_path, self._pending)

    def _journal_output(self, entry: dict[str, Any]) -> None:
        """Persist ownership before replacing; also provides a crash-test boundary."""
        self._pending.setdefault("outputs", []).append(entry)
        _write_json(self.pending_path, self._pending)

    def replace_output(self, target: Path, staged: Path) -> None:
        """Validate ownership, journal old/new hashes, then atomically replace output."""
        self._require_active()
        target = Path(target).absolute()
        staged = Path(staged)
        try:
            relative = target.relative_to(self.project_dir).as_posix()
            target.resolve().relative_to(self.project_dir)
        except ValueError as exc:
            raise YapError("Stage output must be inside the project.", code="unsafe_path",
                           suggested_action="Choose an output within the stage folder.") from exc
        if target in {self.receipt_path, self.pending_path, self.progress_path}:
            raise YapError("Stage bookkeeping is not an output.", code="unsafe_path",
                           suggested_action="Choose a separate stage output file.")
        paths.validate_input(staged)
        digest = receipt.sha256_file(staged)
        owners = (self.receipt_path, self.pending_path, *self.owner_receipts)
        with project_lock(self.project_dir):
            paths.validate_output(target, owner_receipts=owners)
            old = receipt.sha256_file(target) if target.exists() else None
            self._replaced_outputs = True
            self._journal_output({"path": relative, "old_sha256": old, "new_sha256": digest})
            with paths.atomic_output(target, owner_receipts=owners) as temporary:
                try:
                    os.replace(staged, temporary)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(staged, temporary)
                if receipt.sha256_file(temporary) != digest:
                    raise ConflictError(staged)
                current = receipt.sha256_file(target) if target.exists() else None
                if current != old:
                    raise ConflictError(target)
            self.outputs[relative] = {
                "path": relative, "sha256": digest, "bytes": target.stat().st_size,
            }

    def _write_progress(self, step: str, fraction: float) -> None:
        _write_json(self.progress_path, {
            "schema": "yap-studio/progress@1", "pid": os.getpid(), "started": self.started,
            "stage": self.stage, "step": step, "fraction": fraction,
        })

    def progress(self, step: str, fraction: float) -> None:
        """Publish a stage's current step and fraction for status polling."""
        self._require_active()
        if not 0 <= fraction <= 1:
            raise ValidationError("fraction", "must be between 0 and 1")
        with project_lock(self.project_dir):
            self._write_progress(step, fraction)
            if step != self._phase_name:
                now = time.perf_counter()
                self._phases.append({"name": self._phase_name,
                                     "seconds": round(now - self._phase_started, 3)})
                self._phase_name = step
                self._phase_started = now

    def cancel(self) -> None:
        """Restore bookkeeping when a render stops before replacing any outputs."""
        self._require_active()
        if self._replaced_outputs:
            raise RuntimeError("Cannot cancel after replacing stage outputs")
        with project_lock(self.project_dir):
            for target, previous in ((self.receipt_path, self._previous_receipt),
                                     (self.pending_path, self._previous_pending)):
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    with paths.atomic_output(target) as temporary:
                        temporary.write_bytes(previous)
            self.progress_path.unlink(missing_ok=True)
            self._finished = True

    def finish(
        self, status: Literal["verified", "failed"] = "verified", *,
        asserts: list[Any] | None = None, extra: dict[str, Any] | None = None,
    ) -> dict:
        """Publish the final receipt last, then remove the journal and progress."""
        self._require_active()
        if status not in {"verified", "failed"}:
            raise ValidationError("status", "must be verified or failed")
        result = {
            "schema": "yap-studio/stage-receipt@1", "stage": self.stage, "status": status,
            "started": self.started, "finished": None, "inputs": dict(self.inputs),
            "outputs": list(self.outputs.values()),
            "versions": {"yap_studio": __version__, "python": platform.python_version()},
            "asserts": list(asserts or []),
            "timings": {},
        }
        if extra:
            if result.keys() & extra.keys():
                raise ValidationError("extra", "must not replace receipt contract fields")
            result.update(extra)
        with project_lock(self.project_dir):
            for output in self.outputs.values():
                if receipt.sha256_file(self.project_dir / output["path"]) != output["sha256"]:
                    raise ConflictError(self.project_dir / output["path"])
            now = time.perf_counter()
            result["finished"] = timestamp()
            phases = [*self._phases, {"name": self._phase_name,
                                      "seconds": round(now - self._phase_started, 3)}]
            result["timings"] = {"phases": phases,
                                 "total_s": round(now - self._run_started, 3)}
            _write_json(self.receipt_path, result)
            self.pending_path.unlink(missing_ok=True)
            self.progress_path.unlink(missing_ok=True)
            self._finished = True
        return result

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._active and not self._finished:
                with project_lock(self.project_dir):
                    self.progress_path.unlink(missing_ok=True)
        finally:
            self._active = False
            self._operation.__exit__(exc_type, exc, traceback)
