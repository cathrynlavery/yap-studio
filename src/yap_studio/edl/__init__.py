"""Edit-list reader registry and canonical public model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from yap_studio.errors import EditListError, InputError, ValidationError

from .model import (
    Cut,
    EditList,
    Evidence,
    Hold,
    JoinAssertion,
    PictureRegion,
    Segment,
    Source,
    Still,
)

_MAX_EDIT_LIST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Reader:
    """A content-sniffing edit-list reader."""

    name: str
    sniff: Callable[[Path, bytes], bool]
    read: Callable[[Path, bytes], EditList]


READERS: list[Reader] = []


def register_reader(reader: Reader) -> None:
    """Add a reader to the ordered content detector."""
    READERS.append(reader)


def _override_path(edit_list: EditList, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else edit_list.path.parent / path


def _apply_overrides(edit_list: EditList, overrides: Sequence[str]) -> EditList:
    sources = dict(edit_list.sources)
    for override in overrides:
        if "=" in override:
            source_id, raw_path = override.split("=", 1)
            if source_id not in sources:
                raise ValidationError("source", f"unknown source id {source_id!r}")
            if not raw_path:
                raise ValidationError("source", f"source {source_id!r} has an empty path")
        else:
            if len(overrides) != 1 or len(sources) != 1:
                raise ValidationError(
                    "source", "a bare path requires exactly one override and one source"
                )
            source_id = next(iter(sources))
            raw_path = override
            if not raw_path:
                raise ValidationError("source", "source path is empty")
        sources[source_id] = replace(sources[source_id], path=_override_path(edit_list, raw_path))
    return replace(edit_list, sources=sources)


def load_edit_list(
    path: str | Path,
    *,
    source_overrides: Sequence[str] | None = None,
) -> EditList:
    """Load an edit list by content and apply optional source overrides."""
    requested = Path(path)
    source = (
        requested / "Info.fcpxml"
        if requested.is_dir() and requested.suffix == ".fcpxmld"
        else requested
    )
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise InputError(str(source), f"cannot open it for reading ({exc})") from exc
    if size > _MAX_EDIT_LIST_BYTES:
        raise InputError(str(source), "Edit list is larger than 64 MB.")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise InputError(str(source), f"cannot open it for reading ({exc})") from exc
    for reader in READERS:
        if reader.sniff(source, data):
            edit_list = reader.read(source, data)
            if source_overrides:
                edit_list = _apply_overrides(edit_list, source_overrides)
            return edit_list
    raise EditListError(
        "Unsupported edit-list format.",
        code="unsupported_edit_list",
        suggested_action="Export yap span JSON, ranges JSON, FCPXML or xmeml XML.",
        path=str(source),
    )


from . import fcpxml, rangesjson, xmeml, yapjson  # noqa: E402

register_reader(Reader("ranges", rangesjson.sniff, rangesjson.read))
register_reader(Reader("yap", yapjson.sniff, yapjson.read))
register_reader(Reader("fcpxml", fcpxml.sniff, fcpxml.read))
register_reader(Reader("xmeml", xmeml.sniff, xmeml.read))

__all__ = [
    "READERS",
    "Cut",
    "EditList",
    "Evidence",
    "Hold",
    "JoinAssertion",
    "PictureRegion",
    "Reader",
    "Segment",
    "Source",
    "Still",
    "load_edit_list",
    "register_reader",
]
