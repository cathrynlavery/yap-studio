"""Reader for yap span JSON edit lists."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio.errors import EditListError

from .model import EditList, Segment, Source


def _load(data: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EditListError(f"Unreadable yap span JSON: {exc}", path=str(path)) from exc
    if not isinstance(value, dict):
        raise EditListError("Yap span JSON must contain an object.", path=str(path))
    return value


def _number(value: object, field: str, path: Path) -> Fraction:
    try:
        if isinstance(value, bool) or value is None:
            raise ValueError
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise EditListError(f"Invalid {field} value {value!r}.", path=str(path)) from exc


def _source_path(base: Path, value: object, path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise EditListError("Source paths must be non-empty strings.", path=str(path))
    source = Path(value)
    return source if source.is_absolute() else base / source


def _planned_word(item: dict[str, Any], name: str, index: int, path: Path) -> str | None:
    """Read optional editor intent without coercing invalid values into words."""
    if name not in item:
        return None
    value = item[name]
    if not isinstance(value, str) or not value.strip():
        raise EditListError(
            f"Segment {index} {name} must be a non-empty string.",
            path=str(path),
            suggested_action=f"Set {name} to the intended spoken word or omit it.",
        )
    return value


def _source(source_id: str, value: object, base: Path, path: Path) -> Source:
    """Read a legacy path or a path with an audio stream ordinal."""
    stream = None
    if isinstance(value, dict):
        stream = value.get("audio_stream")
        if stream is not None and (
            isinstance(stream, bool) or not isinstance(stream, int) or stream < 0
        ):
            raise EditListError("audio_stream must be a non-negative integer or null.",
                                path=str(path))
        value = value.get("path")
    return Source(source_id, _source_path(base, value, path), stream)


def sniff(path: Path, data: bytes) -> bool:
    """Return whether data has the yap span JSON shape."""
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and "segments" in value


def read(path: Path, data: bytes) -> EditList:
    """Read yap span JSON into the canonical model."""
    value = _load(data, path)
    base = path.parent
    raw_sources = value.get("sources")
    if raw_sources is not None:
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise EditListError("sources must be a non-empty object.", path=str(path))
        sources = {
            str(source_id): _source(str(source_id), source_path, base, path)
            for source_id, source_path in raw_sources.items()
        }
    elif "source" in value:
        sources = {"main": Source("main", _source_path(base, value["source"], path))}
    else:
        raise EditListError("Yap span JSON must provide source or sources.", path=str(path))

    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list):
        raise EditListError("segments must be a list.", path=str(path))
    segments: list[Segment] = []
    for group_number, item in enumerate(raw_segments, 1):
        if not isinstance(item, dict):
            raise EditListError(f"Segment {group_number} must be an object.", path=str(path))
        source_id = item.get("source")
        if source_id is None:
            if len(sources) != 1:
                raise EditListError(
                    f"Segment {group_number} must name a source in a multi-source list.",
                    path=str(path),
                )
            source_id = next(iter(sources))
        source_id = str(source_id)
        if source_id not in sources:
            raise EditListError(
                f"Segment {group_number} refers to unknown source {source_id!r}.",
                path=str(path),
            )
        label = str(item["text"]) if item.get("text") is not None else None
        first_word = _planned_word(item, "first_word", group_number, path)
        last_word = _planned_word(item, "last_word", group_number, path)
        anchors = {name: _number(item[name], name, path)
                   for name in ("first_word_anchor_s", "last_word_anchor_s") if name in item}
        raw_runs = item.get("runs")
        if raw_runs is None:
            raw_runs = [[item.get("start"), item.get("end")]]
        if not isinstance(raw_runs, list):
            raise EditListError(f"Segment {group_number} runs must be a list.", path=str(path))
        for run_index, raw_run in enumerate(raw_runs):
            if not isinstance(raw_run, list) or len(raw_run) != 2:
                raise EditListError(
                    f"Segment {group_number} contains an invalid run.", path=str(path)
                )
            index = len(segments) + 1
            segments.append(
                Segment(
                    index=index,
                    source_id=source_id,
                    start=_number(raw_run[0], f"segment {index} start", path),
                    end=_number(raw_run[1], f"segment {index} end", path),
                    label=label,
                    first_word=first_word if run_index == 0 else None,
                    last_word=last_word if run_index == len(raw_runs) - 1 else None,
                    first_word_anchor_s=anchors.get("first_word_anchor_s") if run_index == 0
                    else None,
                    last_word_anchor_s=anchors.get("last_word_anchor_s")
                    if run_index == len(raw_runs) - 1 else None,
                )
            )

    fps = _number(value["fps"], "fps", path) if value.get("fps") is not None else None
    return EditList("yap", path, sources, segments, fps, [], [])
