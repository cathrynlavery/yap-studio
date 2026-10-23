"""Reader for ranges JSON edit lists."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio.errors import EditListError

from .model import EditList, Evidence, Hold, JoinAssertion, PictureRegion, Segment, Source, Still


def _load(data: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EditListError(f"Unreadable ranges JSON: {exc}", path=str(path)) from exc
    if not isinstance(value, dict):
        raise EditListError("Ranges JSON must contain an object.", path=str(path))
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


def _frame(value: object, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EditListError(f"{field} must be a whole frame number.", path=str(path))
    return value


def _holds(value: object, path: Path) -> list[Hold]:
    """Read declared holds on the output clock, in output frames."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise EditListError("holds must be a list.", path=str(path))
    holds = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise EditListError(f"Hold {index} must be an object.", path=str(path))
        audio = item.get("audio", "silence")
        if audio != "silence":
            raise EditListError(
                f"Hold {index} audio must be \"silence\".",
                path=str(path),
                suggested_action="Declare holds with \"audio\": \"silence\".",
            )
        holds.append(
            Hold(
                index=index,
                start_frame=_frame(item.get("start_frame"), f"Hold {index} start_frame", path),
                end_frame=_frame(item.get("end_frame"), f"Hold {index} end_frame", path),
                audio="silence",
            )
        )
    return sorted(holds, key=lambda hold: hold.start_frame)


def _stills(value: object, path: Path) -> list[Still]:
    if not isinstance(value, list):
        raise EditListError("stills must be a list.", path=str(path))
    stills = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise EditListError(f"Still {index} must be an object.", path=str(path))
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise EditListError(f"Still {index} needs an id.", path=str(path))
        start = _frame(item.get("start_frame"), f"Still {index} start_frame", path)
        end = _frame(item.get("end_frame"), f"Still {index} end_frame", path)
        if start < 0 or end <= start:
            raise EditListError(
                f"Still {index} must start at frame 0 or later and end after it starts.",
                path=str(path),
            )
        stills.append(Still(identifier, start, end))
    return stills


def _phrases(value: object, field: str, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise EditListError(f"{field} must be a list of non-empty phrases.", path=str(path))
    return tuple(item.strip() for item in value)


def _joins(value: object, path: Path) -> list[JoinAssertion]:
    """Read words declared across joins on the output clock."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise EditListError("joins must be a list.", path=str(path))
    joins = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise EditListError(f"Join {index} must be an object.", path=str(path))
        out_frame = _frame(item.get("out_frame"), f"Join {index} out_frame", path)
        if out_frame < 0:
            raise EditListError(
                f"Join {index} out_frame must be 0 or later.", path=str(path)
            )
        before, after = item.get("before", ""), item.get("after", "")
        if not isinstance(before, str) or not isinstance(after, str):
            raise EditListError(
                f"Join {index} before and after must be text.", path=str(path)
            )
        if not before.strip() and not after.strip():
            raise EditListError(
                f"Join {index} needs the words heard before or after it.", path=str(path)
            )
        joins.append(
            JoinAssertion(
                index=index,
                out_frame=out_frame,
                before=before.strip(),
                after=after.strip(),
                absent=_phrases(item.get("absent"), f"Join {index} absent", path),
                once=_phrases(item.get("once"), f"Join {index} once", path),
            )
        )
    return joins


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _region(value: object) -> tuple[float, float, float, float] | None:
    """Read an overlay's ``[x, y, w, h]`` rectangle in output pixels."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        return None
    x, y, w, h = (float(item) for item in value)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _evidence(value: object, base: Path, path: Path) -> list[Evidence]:
    """Read files that attest cut checks made outside Yap Studio."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise EditListError("evidence must be a list.", path=str(path))
    rows = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise EditListError(f"Evidence {index} must be an object.", path=str(path))
        file = item.get("path")
        digest = item.get("sha256")
        if not isinstance(file, str) or not file:
            raise EditListError(f"Evidence {index} needs a path.", path=str(path))
        if not isinstance(digest, str) or not _SHA256.match(digest):
            raise EditListError(
                f"Evidence {index} sha256 must be 64 hexadecimal characters.", path=str(path)
            )
        location = Path(file)
        rows.append(
            Evidence(location if location.is_absolute() else base / location, digest.lower())
        )
    return rows


def sniff(path: Path, data: bytes) -> bool:
    """Return whether data has the ranges JSON shape."""
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and "ranges" in value


def _grade(value: object, base: Path, path: Path) -> dict:
    """Validate the colour contract and resolve its portable reference path."""
    from yap_studio.errors import YapError
    from yap_studio.grade import GradeSpec

    try:
        keys = {"look", "strength", "version", "reference", "reference_sha256"}
        if (not isinstance(value, dict) or not keys <= set(value)
                or set(value) - keys - {"skip_frames", "color"}):
            raise ValueError("needs look, strength, version, reference and reference_sha256")
        GradeSpec.from_dict({key: value[key] for key in ("look", "strength", "version", "color")
                             if key in value})
        reference, digest = value["reference"], value["reference_sha256"]
        if not isinstance(reference, str) or not reference.strip() or "\x00" in reference:
            raise ValueError("reference must be a non-empty path")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("reference_sha256 must be 64 hexadecimal characters")
        skip = value.get("skip_frames", [])
        if not isinstance(skip, list) or any(
            not isinstance(window, list) or len(window) != 2
            or any(type(frame) is not int for frame in window)
            or not 0 <= window[0] < window[1] for window in skip
        ):
            raise ValueError("skip_frames must contain half-open non-negative integer frame pairs")
        location = Path(reference)
        return {**value, "reference": str(location if location.is_absolute() else base / location),
                "reference_sha256": digest.lower()}
    except (ValueError, YapError) as exc:
        raise EditListError(
            f"Invalid grade: {exc}", path=str(path),
            suggested_action="Rebuild the grade EDL with yap-studio assemble.",
        ) from exc


def read(path: Path, data: bytes) -> EditList:
    """Read ranges JSON into the canonical model."""
    value = _load(data, path)
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise EditListError("sources must be a non-empty object.", path=str(path))
    base = path.parent
    sources = {
        str(source_id): Source(str(source_id), _source_path(base, source_path, path))
        for source_id, source_path in raw_sources.items()
    }
    raw_ranges = value.get("ranges")
    if not isinstance(raw_ranges, list):
        raise EditListError("ranges must be a list.", path=str(path))
    segments: list[Segment] = []
    for index, item in enumerate(raw_ranges, 1):
        if not isinstance(item, dict):
            raise EditListError(f"Range {index} must be an object.", path=str(path))
        source_id = str(item.get("source", ""))
        if source_id not in sources:
            raise EditListError(
                f"Range {index} refers to unknown source {source_id!r}.", path=str(path)
            )
        raw_label = item.get("beat") or item.get("note")
        segments.append(
            Segment(
                index=index,
                source_id=source_id,
                start=_number(item.get("start"), f"range {index} start", path),
                end=_number(item.get("end"), f"range {index} end", path),
                label=str(raw_label) if raw_label is not None else None,
            )
        )

    picture_masks: list[tuple[Fraction, Fraction]] = []
    picture_regions: list[PictureRegion] = []
    warnings: list[str] = []
    raw_overlays = value.get("overlays", [])
    if not isinstance(raw_overlays, list):
        warnings.append("Overlays could not be read and were ignored.")
        raw_overlays = []
    for overlay_index, overlay in enumerate(raw_overlays, 1):
        try:
            if not isinstance(overlay, dict):
                raise ValueError
            start = _number(overlay.get("start_in_output"), "overlay start", path)
            duration = _number(overlay.get("duration"), "overlay duration", path)
            if duration < 0:
                raise ValueError
        except (EditListError, ValueError):
            warnings.append(
                f"Overlay {overlay_index} has no readable output window and was ignored."
            )
            continue
        if "region" in overlay:
            region = _region(overlay["region"])
            if region is not None:
                picture_regions.append(PictureRegion(start, start + duration, *region,
                                                      camera_3d=overlay.get("camera_3d") is True))
                continue
            warnings.append(
                f"Overlay {overlay_index} region could not be read; its whole picture is masked."
            )
        picture_masks.append((start, start + duration))

    composite = value.get("composite", False)
    if not isinstance(composite, bool):
        raise EditListError("composite must be true or false.", path=str(path))
    return EditList(
        "ranges",
        path,
        sources,
        segments,
        None,
        picture_masks,
        warnings,
        holds=_holds(value.get("holds"), path),
        joins=_joins(value.get("joins"), path),
        composite=composite,
        evidence=_evidence(value.get("evidence"), base, path),
        picture_regions=picture_regions,
        stills=_stills(value.get("stills", []), path),
        grade=_grade(value["grade"], base, path) if "grade" in value else None,
    )
