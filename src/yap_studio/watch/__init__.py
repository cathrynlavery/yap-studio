"""Offline evidence packs for reviewing a render."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yap_studio.errors import InputError, ValidationError

if TYPE_CHECKING:
    from yap_studio.transcript import Backend

PACK_SCHEMA = "yap-studio/watch-pack@1"


def load_pack(directory: str | Path) -> dict[str, Any]:
    """Read a pack index and require the supported schema."""
    target = Path(directory) / "pack.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise InputError(str(target), "unreadable watch pack JSON") from exc
    if not isinstance(data, dict) or data.get("schema") != PACK_SCHEMA:
        raise ValidationError("pack", f"expected schema {PACK_SCHEMA}")
    return data


def build_pack(
    video: str | Path,
    *,
    edl: str | Path | None = None,
    out: str | Path | None = None,
    every_s: float = 2.0,
    columns: int = 6,
    asr: str = "auto",
    words: str | Path | None = None,
    verify: str | Path | None = None,
    backend: Backend | None = None,
) -> dict[str, Any]:
    """Build an offline, deterministic evidence folder for a render."""
    from .pack import build_pack as build

    return build(
        video,
        edl=edl,
        out=out,
        every_s=every_s,
        columns=columns,
        asr=asr,
        words=words,
        verify=verify,
        backend=backend,
    )


def __getattr__(name: str) -> Any:
    """Load scan and findings support only when requested."""
    if name in {"STILL_S", "PAUSE_S", "LOUDNESS_JUMP_LU", "scan_pack", "run_scan"}:
        from . import scan

        return getattr(scan, name)
    if name in __all__ and name not in {"PACK_SCHEMA", "build_pack", "load_pack"}:
        from . import findings

        return getattr(findings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FINDINGS_SCHEMA",
    "KINDS",
    "LOUDNESS_JUMP_LU",
    "PACK_SCHEMA",
    "PAUSE_S",
    "SEVERITIES",
    "SOURCES",
    "STILL_S",
    "Finding",
    "build_pack",
    "default_findings_path",
    "format_findings",
    "load_findings",
    "load_pack",
    "note_from_file",
    "parse_findings_input",
    "record_findings",
    "run_scan",
    "scan_pack",
]
