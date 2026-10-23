"""Run speech engines in a child, accepting complete results even after a crash."""

from __future__ import annotations

import json
import signal as signals
import sys
import tempfile
import time
import wave
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg, process
from yap_studio.errors import ProcessingError, ValidationError, YapError
from yap_studio.paths import atomic_write_text

from .backends import ParakeetOnnxBackend, resolve_backend
from .model import Transcript

SCHEMA = "yap-studio/engine-child@1"
_STDERR_TAIL_CHARS = 4000


@dataclass(frozen=True)
class ChildRun:
    """A child's outcome, with validated transcripts independent of exit status."""

    transcripts: dict[Path, Transcript] | None
    returncode: int | None
    signal: str | None
    used_result: bool
    reason: str | None
    stderr_tail: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe diagnostics without the transcript payload."""
        return {
            "returncode": self.returncode,
            "signal": self.signal,
            "used_result": self.used_result,
            "reason": self.reason,
            "stderr_tail": self.stderr_tail,
            "seconds": self.seconds,
        }


def _timeout(paths: Sequence[Path]) -> float:
    """Budget model loading and recognition using headers before probing media."""
    total_seconds = 0.0
    for path in paths:
        try:
            with wave.open(str(path), "rb") as reader:
                total_seconds += reader.getnframes() / reader.getframerate()
        except (OSError, EOFError, wave.Error):
            info = ffmpeg.probe(path, count_frames=False)
            duration = info.duration_s if info.duration_s is not None else info.audio_duration_s
            total_seconds += duration or 0.0
    count = len(paths)
    return max(300 + 30 * count, 120 + 30 * count + 1.5 * total_seconds)


def _read_result(path: Path, engine: str, paths: Sequence[Path]) -> dict[Path, Transcript]:
    """Validate the protocol and every requested transcript before accepting any."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema") != SCHEMA
        or data.get("engine") != engine
        or data.get("complete") is not True
        or not isinstance(data.get("transcripts"), dict)
    ):
        raise ValueError("invalid child result")
    return {
        original: Transcript.from_dict(data["transcripts"][str(original)]) for original in paths
    }


def run_in_child(
    engine: str,
    model: str,
    paths: Sequence[Path],
    *,
    options: Mapping[str, Any] | None = None,
    language: str | None = None,
    command: Sequence[str] | None = None,
    timeout_s: float | None = None,
) -> ChildRun:
    """Transcribe through private request/result files, never raising for child failure.

    A complete, validated result takes precedence over any child exit status.
    The private directory follows tempfile's TMPDIR policy and is always removed.
    """
    started = time.perf_counter()
    returncode = None
    signal_name = None
    stderr_tail = ""
    reason = None
    transcripts = None
    try:
        with tempfile.TemporaryDirectory(prefix="yap-engine-child-") as temporary:
            directory = Path(temporary)
            request = directory / "request.json"
            result_path = directory / "result.json"
            request.write_text(
                json.dumps({
                    "engine": engine,
                    "model": model,
                    "options": dict(options or {}),
                    "language": language,
                    "paths": [str(path) for path in paths],
                    "result": str(result_path),
                }),
                encoding="utf-8",
            )
            deadline = timeout_s if timeout_s is not None else _timeout(paths)
            child_command = (
                list(command)
                if command is not None
                else [sys.executable, "-m", "yap_studio.transcript.engine_child"]
            )
            try:
                result = process.spawn(
                    [*child_command, str(request)], timeout_s=deadline, tool="speech_engine"
                )
                returncode = result.returncode
                stderr_tail = result.stderr[-_STDERR_TAIL_CHARS:]
                if returncode < 0:
                    number = -returncode
                    try:
                        signal_name = signals.Signals(number).name
                    except ValueError:
                        signal_name = f"SIG{number}"
                    ending = f"signal {number}, {signal_name}"
                else:
                    ending = f"exit status {returncode}"
                reason = f"stopped before writing its result ({ending})"
            except ProcessingError as exc:
                stderr_tail = exc.full_stderr[-_STDERR_TAIL_CHARS:]
                reason = (
                    f"timed out after {deadline:g} s"
                    if exc.code == "speech_engine_timeout"
                    else "failed to start"
                )

            if result_path.exists():
                try:
                    transcripts = _read_result(result_path, engine, paths)
                    reason = None
                except (
                    OSError, ValueError, KeyError, TypeError, AttributeError,
                    OverflowError, ValidationError,
                ):
                    # Preserve timeout/start diagnostics; malformed output is otherwise incomplete.
                    if returncode is not None:
                        reason = "wrote an incomplete result"
    except (OSError, ValueError, TypeError, YapError) as exc:
        reason = "failed to start"
        stderr_tail = str(exc)[-_STDERR_TAIL_CHARS:]

    return ChildRun(
        transcripts=transcripts,
        returncode=returncode,
        signal=signal_name,
        used_result=transcripts is not None,
        reason=reason,
        stderr_tail=stderr_tail,
        seconds=time.perf_counter() - started,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Load one engine in process and atomically publish all requested transcripts."""
    try:
        args = sys.argv[1:] if argv is None else argv
        request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        engine = request["engine"]
        model = request["model"]
        paths = [Path(value) for value in request["paths"]]
        with redirect_stdout(sys.stderr):
            if engine == "parakeet-onnx":
                backend = ParakeetOnnxBackend(
                    model=model,
                    quantization=request["options"]["quantization"],
                    in_process=True,
                )
            else:
                backend = resolve_backend(engine, model)
            if backend is None:
                raise ValueError("no speech engine selected")
            transcripts = backend.transcribe_many(paths, language=request.get("language"))
            payload = {
                "schema": SCHEMA,
                "engine": engine,
                "model": model,
                "version": backend.version(),
                "complete": True,
                "transcripts": {str(path): transcripts[path].to_dict() for path in paths},
            }
            atomic_write_text(request["result"], json.dumps(payload, allow_nan=False) + "\n")
        return 0
    except Exception as exc:
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"speech engine failed: {detail[:500]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
