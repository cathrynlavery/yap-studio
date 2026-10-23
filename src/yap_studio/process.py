"""Internal noninteractive process helper."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

from yap_studio.errors import ProcessingError


def spawn(
    command: list[str],
    *,
    timeout_s: float,
    tool: str,
    input_bytes: bytes | None = None,
    env: Mapping[str, str] | None = None,
    binary: bool = False,
    cwd: str | os.PathLike[str] | None = None,
    stdout_path: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run with a deadline and closed stdin, leaving exit mapping to callers.

    A payload opens stdin only for that input. Environment values overlay the
    inherited environment. Text output uses UTF-8 with replacement on errors.
    A stdout path writes child bytes directly to a file and leaves stdout unset.
    """
    kwargs: dict[str, Any] = {
        "stderr": subprocess.PIPE,
        "timeout": timeout_s,
        "check": False,
        "cwd": cwd,
    }
    if input_bytes is None:
        kwargs["stdin"] = subprocess.DEVNULL
    else:
        kwargs["input"] = input_bytes
    if env is not None:
        kwargs["env"] = {**os.environ, **env}
    if not binary and input_bytes is None:
        kwargs.update(text=True, encoding="utf-8", errors="replace")
    try:
        with ExitStack() as stack:
            kwargs["stdout"] = (stack.enter_context(open(stdout_path, "wb"))
                                if stdout_path is not None else subprocess.PIPE)
            result = subprocess.run(command, **kwargs)
    except subprocess.TimeoutExpired as exc:
        stderr = f"timed out after {timeout_s} s"
        if tool == "wavespeed-pp-cli":
            detail = exc.stderr
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            if detail:
                stderr += "\n" + detail
        error = ProcessingError(
            " ".join(command),
            -1,
            stderr,
            code=f"{tool}_timeout",
        )
        if tool == "wavespeed-pp-cli":
            error.stdout = exc.stdout
        raise error from None
    except OSError as exc:
        raise ProcessingError(
            " ".join(command), -1, str(exc), code=f"{tool}_exec_failed"
        ) from None
    if not binary and input_bytes is not None:
        if result.stdout is not None:
            result.stdout = result.stdout.decode("utf-8", errors="replace")
        result.stderr = result.stderr.decode("utf-8", errors="replace")
    return result
