"""Run the independent speech pass in a disposable child process."""

from __future__ import annotations

import json
import shlex
import signal
import sys
import tempfile
import time
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yap_studio import ffmpeg, process
from yap_studio.errors import ProcessingError, YapError

from .model import SCHEMA, Transcript

if TYPE_CHECKING:
    from .backends import Backend

SECOND_PASS_SCHEMA = "yap-studio/second-pass@1"


class SecondPassUnmeasured(ProcessingError):
    """Keep a failed second pass separate from the primary verification result."""

    def __init__(
        self,
        reason: str,
        *,
        returncode: int = -1,
        stderr: str = "",
        command: str = "",
    ) -> None:
        self.reason = reason if reason.startswith("unmeasured: ") else f"unmeasured: {reason}"
        super().__init__(command, returncode, stderr[-500:], code="second_engine_unmeasured")
        self.args = (self.reason,)


def _timeout_seconds(paths: Sequence[Path]) -> float:
    """Allow model startup, per-file work, and transcription scaled by input duration."""
    durations = []
    for path in paths:
        try:
            with wave.open(str(path), "rb") as stream:
                duration = stream.getnframes() / stream.getframerate()
        except (OSError, EOFError, wave.Error):
            info = ffmpeg.probe(path, count_frames=False)
            duration = info.duration_s if info.duration_s is not None else info.audio_duration_s
            if duration is None:
                raise ValueError("could not determine input duration") from None
        durations.append(duration)
    return 300 + 30 * len(paths) + 1.5 * sum(durations)


def _signal_name(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return None


def _ending(returncode: int) -> str:
    if returncode < 0:
        name = _signal_name(returncode)
        return f"signal {-returncode}" + (f", {name}" if name else "")
    return f"exit status {returncode}"


class IsolatedBackend:
    """Expose engine metadata in the parent and transcribe only in a child."""

    def __init__(self, inner: Backend, *, command: Sequence[str] | None = None) -> None:
        self.inner = inner
        self.command = list(command) if command is not None else [
            sys.executable, "-m", "yap_studio.transcript.child",
        ]
        self.last_run: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    def version(self) -> str | None:
        return self.inner.version()

    def download_notice(self) -> str | None:
        return self.inner.download_notice()

    def prefetch(self) -> None:
        self.inner.prefetch()

    def transcribe_many(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        """Use complete results even after a child crash, recording every run's outcome."""
        started = time.monotonic()
        run: dict[str, Any] = {
            "returncode": None, "signal": None, "used_result": False,
            "reason": None, "seconds": 0.0,
        }
        self.last_run = run
        stderr = ""
        command = self.command
        try:
            originals = list(paths)
            timeout_s = _timeout_seconds(originals)
            with tempfile.TemporaryDirectory(prefix="yap-second-pass-") as temporary:
                directory = Path(temporary)
                request_path, result_path = directory / "request.json", directory / "result.json"
                request = {
                    "engine": self.name, "model": self.model,
                    "paths": [str(path) for path in originals], "result": str(result_path),
                }
                if language is not None:
                    request["language"] = language
                request_path.write_text(json.dumps(request), encoding="utf-8")
                command = [*self.command, str(request_path)]
                try:
                    completed = process.spawn(
                        command, timeout_s=timeout_s, tool="second_engine",
                    )
                except ProcessingError as exc:
                    stderr = exc.full_stderr
                    if exc.code == "second_engine_timeout":
                        detail = f"timed out after {timeout_s:g} s"
                    else:
                        detail = "failed to start"
                    raise SecondPassUnmeasured(
                        f"unmeasured: the second speech engine {self.name} {detail}",
                        stderr=stderr, command=shlex.join(command),
                    ) from None
                run["returncode"] = completed.returncode
                run["signal"] = _signal_name(completed.returncode)
                stderr = completed.stderr
                ending = _ending(completed.returncode)
                if not result_path.is_file():
                    detail = "stopped before writing its result"
                else:
                    try:
                        data = json.loads(result_path.read_text(encoding="utf-8"))
                        if (
                            not isinstance(data, dict)
                            or data.get("schema") != SECOND_PASS_SCHEMA
                            or data.get("engine") != self.name
                            or data.get("complete") is not True
                            or not isinstance(data.get("transcripts"), dict)
                        ):
                            raise ValueError("invalid second-pass envelope")
                        transcripts = {}
                        for path in originals:
                            row = data["transcripts"][str(path)]
                            if (
                                not isinstance(row, dict) or row.get("schema") != SCHEMA
                                or not isinstance(row.get("words"), list)
                                or not all(isinstance(word, dict) for word in row["words"])
                            ):
                                raise ValueError("invalid transcript")
                            transcripts[path] = Transcript.from_dict(row)
                    except (OSError, ValueError, KeyError, TypeError, RecursionError):
                        detail = "wrote an incomplete result"
                    else:
                        run["used_result"] = True
                        return transcripts
                raise SecondPassUnmeasured(
                    f"unmeasured: the second speech engine {self.name} {detail} ({ending})",
                    returncode=completed.returncode, stderr=stderr, command=shlex.join(command),
                )
        except SecondPassUnmeasured as exc:
            run["reason"] = exc.reason
            raise
        except (OSError, ValueError, YapError) as exc:
            reason = f"unmeasured: the second speech engine {self.name} could not run: {exc}"
            run["reason"] = reason
            raise SecondPassUnmeasured(
                reason, stderr=stderr, command=shlex.join(command),
            ) from None
        finally:
            run["seconds"] = time.monotonic() - started
