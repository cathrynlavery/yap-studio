"""Save a learned style to the library and queue its rules, without making it the default."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from yap_studio import paths
from yap_studio.style.library import StyleError, _read, add, get_default, library_dir
from yap_studio.taste import store
from yap_studio.taste.learn import LearnResult
from yap_studio.taste.store import add_proposals

LEARNED_NAME = "Learned from your videos"
LEARNED_ID = "learned"
SOURCE_SCHEMA = "yap-studio/learned-source@1"


def source_path(style_id: str) -> Path:
    """Name the record of where a learned style came from; style listings skip this name."""
    return library_dir() / f"{style_id}.source.json"


def _owned(style_id: str) -> bool:
    """Recognize a source record written by this flow."""
    record = source_path(style_id)
    if not record.is_file() or record.is_symlink():
        return False
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    return isinstance(data, dict) and data.get("schema") == SOURCE_SCHEMA


def _available(style_id: str, data: dict[str, Any], default: str | None) -> bool:
    """Use empty ids or owned ids that do not change the default."""
    target = library_dir() / f"{style_id}.json"
    record = source_path(style_id)
    if not any((target.exists(), target.is_symlink(), record.exists(), record.is_symlink())):
        return True
    if target.is_symlink() or (target.exists() and not target.is_file()) or not _owned(style_id):
        return False
    if style_id != default:
        return True
    try:
        return target.is_file() and _read(target) == data
    except StyleError:
        return False


def _style_id(data: dict[str, Any]) -> str:
    """Choose the first usable learned style id."""
    default = get_default()
    candidate, number = LEARNED_ID, 1
    while not _available(candidate, data, default):
        number += 1
        candidate = f"{LEARNED_ID}-{number}"
    return candidate


def _restore(path: Path, before: bytes | None) -> None:
    """Put back a saved file exactly, or remove a newly created file."""
    if before is None:
        path.unlink(missing_ok=True)
        return
    with tempfile.TemporaryDirectory(prefix=".yap_tmp_", dir=path.parent) as folder:
        staged = Path(folder) / path.name
        with paths.atomic_output(staged) as temporary:
            temporary.write_bytes(before)
        os.replace(staged, path)


def save_learned(result: LearnResult) -> dict[str, Any]:
    """Add the proposed style as "Learned from your videos" and queue its proposed rules."""
    data = json.loads(result.style_path.read_text(encoding="utf-8"))
    data["name"] = LEARNED_NAME
    store._input(result.proposals_path)
    store.load_taste()
    store.load_proposals()
    style_id = _style_id(data)
    style = library_dir() / f"{style_id}.json"
    record = source_path(style_id)
    before = (style.read_bytes() if style.exists() else None,
              record.read_bytes() if record.exists() else None)
    summary = result.to_dict()
    source = {
        "schema": SOURCE_SCHEMA,
        "style": style_id,
        "videos": len(summary["videos"]),
        "names": [video["name"] for video in summary["videos"]],
        "measured_fields": summary["measured_fields"],
        "style_base": summary["style_base"],
        "learned_at": result.measurements.get("created_at"),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="yap-learned-") as folder:
            staged = Path(folder) / f"{style_id}.json"
            staged.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
            entry = add(staged, make_default=False, replace=True)
        with paths.operation():
            paths.atomic_write_text(record,
                                    json.dumps(source, indent=2, ensure_ascii=False) + "\n")
        queued = add_proposals(result.proposals_path)
    except Exception:
        with paths.operation():
            _restore(style, before[0])
            _restore(record, before[1])
        raise
    return {
        "style": {"id": entry.id, "name": entry.name, "path": str(entry.path),
                  "default": entry.is_default, "videos": source["videos"],
                  "source": str(source_path(entry.id))},
        "rules": [{"id": item.id, "status": item.status} for item in queued],
    }
