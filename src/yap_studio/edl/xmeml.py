"""Safe stdlib reader for xmeml edit lists."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from yap_studio.errors import EditListError

from .fcpxml import (
    _SUGGEST_NO_TRANSITIONS,
    _SUGGEST_RENDER_RETIMES,
    _SUGGEST_SAFE_EXPORT,
    _UNSAFE_DECLARATION,
    _path_from_src,
    _unsupported,
)
from .model import EditList, Segment, Source

_SUGGEST_FLATTEN = "Flatten nested sequences before exporting."


def _name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, wanted: str) -> ET.Element | None:
    return next((child for child in element if _name(child) == wanted), None)


def _children(element: ET.Element, wanted: str) -> list[ET.Element]:
    return [child for child in element if _name(child) == wanted]


def _text(element: ET.Element, wanted: str) -> str | None:
    child = _child(element, wanted)
    return child.text.strip() if child is not None and child.text else None


def sniff(path: Path, data: bytes) -> bool:
    """Return whether data appears to have an xmeml root."""
    return re.search(br"<\s*xmeml(?:\s|>)", data, re.IGNORECASE) is not None


def _rate(element: ET.Element | None, path: Path, description: str) -> Fraction:
    if element is None:
        raise EditListError(f"{description} has no rate.", path=str(path))
    timebase = _text(element, "timebase")
    try:
        value = Fraction(timebase or "")
    except (ValueError, ZeroDivisionError) as exc:
        raise EditListError(
            f"{description} has no readable timebase.", path=str(path)
        ) from exc
    if value <= 0:
        raise EditListError(f"{description} timebase must be positive.", path=str(path))
    if (_text(element, "ntsc") or "").upper() == "TRUE":
        value *= Fraction(1000, 1001)
    return value


def _frame(element: ET.Element, wanted: str, path: Path) -> int:
    value = _text(element, wanted)
    try:
        return int(value or "")
    except ValueError as exc:
        clip_id = element.get("id", "unknown")
        raise EditListError(
            f"Clip {clip_id!r} has no readable {wanted} frame.", path=str(path)
        ) from exc


def _is_enabled(clip: ET.Element) -> bool:
    return (_text(clip, "enabled") or "TRUE").upper() != "FALSE"


def _clipitems(track: ET.Element) -> list[ET.Element]:
    return _children(track, "clipitem")


def _resolved_windows(
    clips: list[ET.Element], path: Path
) -> list[tuple[ET.Element, int, int]]:
    values = [(clip, _frame(clip, "start", path), _frame(clip, "end", path)) for clip in clips]
    resolved: list[tuple[ET.Element, int, int]] = []
    for index, (clip, start, end) in enumerate(values):
        if start == -1 and index > 0 and values[index - 1][2] != -1:
            start = values[index - 1][2]
        if end == -1 and index + 1 < len(values) and values[index + 1][1] != -1:
            end = values[index + 1][1]
        if start == -1 or end == -1:
            raise _unsupported(
                path,
                f"Clip {clip.get('id', 'unknown')!r} has an unresolved transition boundary.",
                _SUGGEST_NO_TRANSITIONS,
            )
        resolved.append((clip, start, end))
    return resolved


def _file_definitions(root: ET.Element, path: Path) -> dict[str, ET.Element]:
    definitions: dict[str, ET.Element] = {}
    files: list[ET.Element] = []
    for element in root.iter():
        if _name(element) != "file":
            continue
        files.append(element)
        file_id = element.get("id")
        if file_id and len(element):
            definitions[file_id] = element
    for element in files:
        file_id = element.get("id")
        if not len(element) and (not file_id or file_id not in definitions):
            raise EditListError(
                f"Clip refers to unknown file {file_id!r}.", path=str(path)
            )
    return definitions


def _source_for_clip(
    clip: ET.Element,
    definitions: dict[str, ET.Element],
    path: Path,
) -> tuple[str, Source]:
    file_element = _child(clip, "file")
    file_id = file_element.get("id") if file_element is not None else None
    if not file_id or file_id not in definitions:
        raise EditListError(f"Clip refers to unknown file {file_id!r}.", path=str(path))
    definition = definitions[file_id]
    return file_id, Source(file_id, _path_from_src(path, _text(definition, "pathurl")))


def _internal_cuts(windows: list[tuple[ET.Element, int, int]]) -> list[int]:
    return [
        left_end
        for (_, _, left_end), (_, right_start, _) in zip(windows, windows[1:], strict=False)
        if left_end == right_start
    ]


def _cut_time(frame: int, fps: Fraction) -> str:
    seconds = float(Fraction(frame, 1) / fps)
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - minutes * 60:04.1f}"


def read(path: Path, data: bytes) -> EditList:
    """Read xmeml into the canonical model."""
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
    if _name(root) != "xmeml":
        raise EditListError("XML root must be xmeml.", path=str(path))

    sequence = _child(root, "sequence")
    if sequence is None:
        raise EditListError("XML has no sequence.", path=str(path))
    for clip in (element for element in root.iter() if _name(element) == "clipitem"):
        if _child(clip, "sequence") is not None:
            raise _unsupported(path, "Nested sequences are not supported.", _SUGGEST_FLATTEN)

    fps = _rate(_child(sequence, "rate"), path, "Sequence")
    media = _child(sequence, "media")
    video = _child(media, "video") if media is not None else None
    tracks = _children(video, "track") if video is not None else []
    if not tracks:
        raise EditListError("Sequence has no video track.", path=str(path))

    warnings: list[str] = []
    enabled_tracks: list[tuple[int, ET.Element]] = []
    for track_index, track in enumerate(tracks, 1):
        if _is_enabled(track):
            enabled_tracks.append((track_index, track))
        else:
            warnings.append(f"Disabled video track {track_index} was ignored.")
    if not enabled_tracks:
        raise EditListError("Sequence has no enabled video track.", path=str(path))
    _, primary_track = enabled_tracks[0]
    if any(_name(element) == "transitionitem" for element in primary_track):
        raise _unsupported(path, "Transitions are not supported.", _SUGGEST_NO_TRANSITIONS)

    definitions = _file_definitions(root, path)
    sources = {
        file_id: Source(file_id, _path_from_src(path, _text(element, "pathurl")))
        for file_id, element in definitions.items()
    }

    primary_clips: list[ET.Element] = []
    for clip in _clipitems(primary_track):
        if _is_enabled(clip):
            primary_clips.append(clip)
        else:
            warnings.append(f"Disabled clip {clip.get('id', 'unknown')!r} was ignored.")
    primary_windows = _resolved_windows(primary_clips, path)

    segments: list[Segment] = []
    for clip, output_start, output_end in primary_windows:
        source_id, source = _source_for_clip(clip, definitions, path)
        sources[source_id] = source
        clip_rate_element = _child(clip, "rate")
        clip_rate = (
            _rate(clip_rate_element, path, f"Clip {clip.get('id', 'unknown')!r}")
            if clip_rate_element is not None
            else fps
        )
        source_start = _frame(clip, "in", path)
        source_end = _frame(clip, "out", path)
        timeline_duration = Fraction(output_end - output_start, 1) / fps
        source_duration = Fraction(source_end - source_start, 1) / clip_rate
        if abs(timeline_duration - source_duration) > Fraction(1, 2) / fps:
            clip_name = _text(clip, "name") or clip.get("id", "unknown")
            raise _unsupported(
                path,
                f"Clip {clip_name!r} is retimed (its timeline length differs from its "
                "source length). Retimed clips are not supported yet.",
                _SUGGEST_RENDER_RETIMES,
            )
        segments.append(
            Segment(
                index=len(segments) + 1,
                source_id=source_id,
                start=Fraction(source_start, 1) / clip_rate,
                end=Fraction(source_end, 1) / clip_rate,
                label=_text(clip, "name"),
                output_start=Fraction(output_start, 1) / fps,
            )
        )

    picture_masks: list[tuple[Fraction, Fraction]] = []
    for track_index, track in enabled_tracks[1:]:
        clips: list[ET.Element] = []
        for clip in _clipitems(track):
            if _is_enabled(clip):
                clips.append(clip)
            else:
                warnings.append(f"Disabled clip {clip.get('id', 'unknown')!r} was ignored.")
        if not clips:
            continue
        for _, start, end in _resolved_windows(clips, path):
            picture_masks.append((Fraction(start, 1) / fps, Fraction(end, 1) / fps))
        warnings.append(f"Video track {track_index} clips mask the primary picture.")

    audio = _child(media, "audio") if media is not None else None
    audio_tracks = _children(audio, "track") if audio is not None else []
    if audio_tracks:
        audio_clips = [clip for clip in _clipitems(audio_tracks[0]) if _is_enabled(clip)]
        audio_cuts = set(_internal_cuts(_resolved_windows(audio_clips, path)))
        for cut_index, cut_frame in enumerate(_internal_cuts(primary_windows), 1):
            if cut_frame not in audio_cuts:
                warnings.append(
                    "Audio and video are cut at different points near "
                    f"cut {cut_index} ({_cut_time(cut_frame, fps)}); the video cuts are used."
                )

    return EditList("xmeml", path, sources, segments, fps, picture_masks, warnings)
