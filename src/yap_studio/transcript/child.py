"""Transcribe one second-pass request and publish a complete result atomically."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from yap_studio.paths import atomic_write_text

from .backends import resolve_backend
from .isolated import SECOND_PASS_SCHEMA


def main(argv: Sequence[str] | None = None) -> int:
    """Run a request file without letting an engine failure escape into the parent."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("expected one second-pass request file", file=sys.stderr)
        return 2
    try:
        request = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
        engine, model = request["engine"], request["model"]
        paths = [Path(path) for path in request["paths"]]
        with redirect_stdout(sys.stderr):
            backend = resolve_backend(engine, model)
            if backend is None:
                raise ValueError("the requested speech engine is disabled")
            language = request.get("language")
            transcripts = backend.transcribe_many(
                paths, **({"language": language} if language is not None else {}),
            )
            version = backend.version()
        result = {
            "schema": SECOND_PASS_SCHEMA,
            "engine": backend.name,
            "model": backend.model,
            "version": version,
            "complete": True,
            "transcripts": {str(path): transcripts[path].to_dict() for path in paths},
        }
        atomic_write_text(request["result"], json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"second speech pass failed: {str(exc)[-500:]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
