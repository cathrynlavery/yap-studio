"""Safe stdlib reader for FCPXML edit lists."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote, urlparse

from yap_studio.errors import EditListError

from .model import EditList, Segment, Source

_UNSAFE_DECLARATION = re.compile(br"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_SUGGEST_SAFE_EXPORT = "Export XML without document type or entity declarations."
_SUGGEST_BREAK_APART = "Break apart compound, synchronized or multicam clips before exporting."
_SUGGEST_NO_TRANSITIONS = "Remove dissolves or transitions at cuts, or export without them."
_SUGGEST_RENDER_RETIMES = "Render speed changes into a new source clip first."


def _name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def parse_time(value: str | None) -> Fraction:
    """Parse an XML rational time into exact seconds."""
    if value is None:
        return Fraction(0)
    if not value.endswith("s"):
        raise EditListError(f"Invalid rational time {value!r}.")
    number = value[:-1]
    try:
        return Fraction(number)
    except (ValueError, ZeroDivisionError) as exc:
        raise EditListError(f"Invalid rational time {value!r}.") from exc


def _path_from_src(document: Path, src: str | None) -> Path | None:
    if not src:
        return None
    parsed = urlparse(src)
    if parsed.scheme.lower() == "file":
        host = "" if parsed.netloc.lower() == "localhost" else parsed.netloc
        prefix = f"//{host}" if host else ""
        return Path(prefix + unquote(parsed.path))
    value = Path(unquote(src))
    return value if value.is_absolute() else document.parent / value


def sniff(path: Path, data: bytes) -> bool:
    """Return whether data appears to have an fcpxml root."""
    return re.search(br"<\s*fcpxml(?:\s|>)", data, re.IGNORECASE) is not None


def _unsupported(path: Path, message: str, action: str) -> EditListError:
    return EditListError(
        message,
        code="unsupported_edit_list",
        suggested_action=action,
        path=str(path),
    )


def _find_first(root: ET.Element, wanted: str) -> ET.Element | None:
    return next((element for element in root.iter() if _name(element) == wanted), None)


def _has_primary_time_map(element: ET.Element) -> bool:
    for child in element:
        if child.get("lane") is not None:
            continue
        if _name(child) == "timeMap" or _has_primary_time_map(child):
            return True
    return False


def _connected_masks(
    parent: ET.Element,
    parent_output_start: Fraction,
    warnings: list[str],
) -> list[tuple[Fraction, Fraction]]:
    masks: list[tuple[Fraction, Fraction]] = []
    parent_source_start = parse_time(parent.get("start"))
    for child in parent:
        child_output_start = (
            parent_output_start + parse_time(child.get("offset")) - parent_source_start
        )
        lane_value = child.get("lane")
        if lane_value is not None:
            try:
                lane = int(lane_value)
            except ValueError:
                warnings.append("A connected clip has an unreadable lane and was ignored.")
            else:
                if lane > 0:
                    duration = parse_time(child.get("duration"))
                    masks.append((child_output_start, child_output_start + duration))
                    warnings.append(f"Connected lane {lane} clip masks the primary picture.")
                elif lane < 0:
                    warnings.append(
                        f"Connected lane {lane} clip is below the primary storyline."
                    )
        masks.extend(_connected_masks(child, child_output_start, warnings))
    return masks


def read(path: Path, data: bytes) -> EditList:
    """Read FCPXML into the canonical model."""
    if _UNSAFE_DECLARATION.search(data):
        raise _unsupported(
            path,
            "XML declarations with external structure are not supported.",
            _SUGGEST_SAFE_EXPORT,
        )
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise EditListError(f"Unreadable XML: {exc}", path=str(path)) from exc
    if _name(root) != "fcpxml":
        raise EditListError("XML root must be fcpxml.", path=str(path))

    warnings: list[str] = []
    version = root.get("version")
    try:
        supported_version = version is not None and Fraction(version) in {
            Fraction(18, 10),
            Fraction(19, 10),
            Fraction(110, 100),
            Fraction(111, 100),
        }
    except (ValueError, ZeroDivisionError):
        supported_version = False
    if not supported_version:
        warnings.append(
            f"FCPXML version {version or 'unknown'} is outside the tested range 1.8 to 1.11."
        )

    for element in root.iter():
        element_name = _name(element)
        if element_name == "transition":
            raise _unsupported(path, "Transitions are not supported.", _SUGGEST_NO_TRANSITIONS)
        if element_name in {"ref-clip", "sync-clip", "mc-clip"}:
            raise _unsupported(path, f"{element_name} is not supported.", _SUGGEST_BREAK_APART)

    formats: dict[str, Fraction] = {}
    assets: dict[str, tuple[Source, Fraction]] = {}
    for element in root.iter():
        element_name = _name(element)
        resource_id = element.get("id")
        if element_name == "format" and resource_id:
            if element.get("frameDuration") is not None:
                formats[resource_id] = parse_time(element.get("frameDuration"))
        elif element_name == "asset" and resource_id:
            src = element.get("src")
            if src is None:
                media_reps = [child for child in element if _name(child) == "media-rep"]
                original = next(
                    (child for child in media_reps if child.get("kind") == "original-media"),
                    None,
                )
                selected = (
                    original
                    if original is not None
                    else (media_reps[0] if media_reps else None)
                )
                src = selected.get("src") if selected is not None else None
            assets[resource_id] = (
                Source(resource_id, _path_from_src(path, src)),
                parse_time(element.get("start")),
            )

    sequence = _find_first(root, "sequence")
    if sequence is None:
        raise EditListError("XML has no sequence.", path=str(path))
    spine = next((child for child in sequence if _name(child) == "spine"), None)
    if spine is None:
        spine = _find_first(sequence, "spine")
    if spine is None:
        raise EditListError("Sequence has no spine.", path=str(path))
    format_id = sequence.get("format")
    frame_duration = formats.get(format_id or "")
    if frame_duration is None or frame_duration <= 0:
        raise EditListError("Sequence has no readable frame duration.", path=str(path))
    fps = 1 / frame_duration
    tc_start = parse_time(sequence.get("tcStart"))

    segments: list[Segment] = []
    picture_masks: list[tuple[Fraction, Fraction]] = []
    for primary in spine:
        primary_name = _name(primary)
        if primary_name in {"asset-clip", "clip"} and _has_primary_time_map(primary):
            clip_name = primary.get("name") or primary.get("ref") or "unknown"
            raise _unsupported(
                path,
                f"Clip {clip_name!r} is retimed (its timeline length differs from its "
                "source length). Retimed clips are not supported yet.",
                _SUGGEST_RENDER_RETIMES,
            )
        output_start = parse_time(primary.get("offset")) - tc_start
        duration = parse_time(primary.get("duration"))
        picture_masks.extend(_connected_masks(primary, output_start, warnings))
        source_id: str | None
        source_start: Fraction
        if primary_name == "asset-clip":
            source_id = primary.get("ref")
            if source_id not in assets:
                raise EditListError(f"Clip refers to unknown asset {source_id!r}.", path=str(path))
            source_start = parse_time(primary.get("start")) - assets[source_id][1]
        elif primary_name == "clip":
            video = next((child for child in primary if _name(child) == "video"), None)
            if video is None:
                warnings.append("A clip without primary video was ignored.")
                continue
            source_id = video.get("ref")
            if source_id not in assets:
                raise EditListError(f"Video refers to unknown asset {source_id!r}.", path=str(path))
            source_start = parse_time(video.get("start")) + (
                parse_time(primary.get("start")) - parse_time(video.get("offset"))
            ) - assets[source_id][1]
        elif primary_name == "gap":
            source_id = None
            source_start = Fraction(0)
        else:
            warnings.append(f"Spine element {primary_name} was ignored.")
            continue
        segments.append(
            Segment(
                index=len(segments) + 1,
                source_id=source_id,
                start=source_start,
                end=source_start + duration,
                label=primary.get("name"),
                output_start=output_start,
            )
        )

    sources = {source_id: source for source_id, (source, _) in assets.items()}
    return EditList("fcpxml", path, sources, segments, fps, picture_masks, warnings)
