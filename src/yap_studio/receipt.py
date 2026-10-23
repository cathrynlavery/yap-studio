"""Reproducible run receipts."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import __version__
from yap_studio.ffmpeg import tool_versions
from yap_studio.paths import atomic_write_text

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _file_record(path: str | Path, *, required: bool) -> dict[str, Any]:
    item = Path(path)
    if not item.is_file():
        if required:
            sha256_file(item)
        return {"path": str(item), "sha256": None, "bytes": None}
    return {"path": str(item), "sha256": sha256_file(item), "bytes": item.stat().st_size}


def build_receipt(
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    started_at: datetime,
    ended_at: datetime,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run receipt with hashes, timestamps, and tool versions."""
    media_versions = tool_versions()
    receipt: dict[str, Any] = {
        "schema": "yap-studio/receipt@1",
        "started_at": _utc_iso(started_at),
        "ended_at": _utc_iso(ended_at),
        "duration_s": (ended_at - started_at).total_seconds(),
        "inputs": [_file_record(path, required=True) for path in inputs],
        "outputs": [_file_record(path, required=False) for path in outputs],
        "versions": {
            "yap_studio": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "ffmpeg": media_versions["ffmpeg"],
            "ffprobe": media_versions["ffprobe"],
        },
        "speech": None,
    }
    if extra:
        receipt.update(extra)
    return receipt


def write_receipt(path: str | Path, receipt: dict[str, Any]) -> Path:
    """Write a receipt as readable JSON through the guarded writer."""
    payload = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    return atomic_write_text(path, payload)
