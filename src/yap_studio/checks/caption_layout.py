"""Hash-bound caption layout records and checks on the output frame grid."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio import paths
from yap_studio.captions import caption_phrase_end
from yap_studio.checks.model import Finding, Thresholds, format_time
from yap_studio.errors import InputError
from yap_studio.faces.clearance import LAYOUT_SCHEMA, Element, Layout, gap_px
from yap_studio.receipt import sha256_file

LAYOUT_SUFFIX = ".layout.json"
_CHECKS = ("captions.timing", "captions.cards")


@dataclass(frozen=True)
class CaptionWord:
    """A word's spoken interval in output seconds."""

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class LayoutRecord:
    """Measured geometry and spoken words belonging to one render."""

    layout: Layout
    measured: bool
    reason: str | None
    words: dict[str, tuple[CaptionWord, ...]]
    face_windows: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    element_ids: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


def write_layout_record(
    media: Path, document: Mapping[str, Any], out: Path | None = None,
) -> Path:
    """Atomically bind a layout document to the published media bytes."""
    target = out if out is not None else media.with_suffix(LAYOUT_SUFFIX)
    source = paths.validate_input(media)
    paths.validate_output(target)
    try:
        record = {**document, "media_sha256": sha256_file(source)}
        return paths.atomic_write_text(
            target, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
    except (OSError, ValueError) as exc:
        raise InputError(str(target), f"cannot write layout record: {exc}") from exc


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate layout record field")
        result[key] = value
    return result


def _word(data: Any) -> CaptionWord:
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("caption words must have text and finite start_s and end_s")
    times = [data.get(key) for key in ("start_s", "end_s")]
    if any(type(value) not in (int, float) for value in times):
        raise ValueError("caption word times must be finite numbers")
    start, end = map(float, times)
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("caption word times must be finite with end_s >= start_s")
    return CaptionWord(data["text"], start, end)


def read_layout_record(media: Path, *, width: int, height: int) -> LayoutRecord | None:
    """Read an optional layout, refusing corrupt records and stale render geometry."""
    target = media.with_suffix(LAYOUT_SUFFIX)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        if target.is_symlink():
            raise InputError(str(target), "layout record points to a missing file") from exc
        return None
    except (OSError, UnicodeError) as exc:
        raise InputError(str(target), "cannot read layout record") from exc
    try:
        document = json.loads(text, object_pairs_hook=_json_object)
        if not isinstance(document, dict) or document.get("schema") != LAYOUT_SCHEMA:
            raise ValueError("unknown layout record schema")
        layout = Layout.from_dict(document)
        face_windows = {element.id: element.face_windows
                        for element in (*layout.elements, *layout.captions)
                        if element.face_windows is not None}
        element_ids = {}
        labels = {}
        for element in document.get("elements", []):
            element_id = element.get("element_id", element["id"])
            if not isinstance(element_id, str) or not element_id:
                raise ValueError(f'element "{element["id"]}" element_id must be a non-empty string')
            element_ids[element["id"]] = element_id
            if "label" in element:
                label = element["label"]
                if not isinstance(label, str) or not label:
                    raise ValueError(f'element "{element["id"]}" label must be a non-empty string')
                labels[element["id"]] = label
        measured = document.get("measured")
        if type(measured) is not bool:
            raise ValueError("layout record measured must be a boolean")
        reason = document.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("layout record reason must be a string or null")
        words = {}
        if measured:
            for caption in document.get("captions", []):
                items = caption.get("words")
                if not isinstance(items, list) or not items:
                    raise ValueError(f'caption "{caption["id"]}" needs a non-empty words list')
                words[caption["id"]] = tuple(_word(item) for item in items)
    except (ValueError, OverflowError, RecursionError) as exc:
        raise InputError(str(target), f"corrupt layout record: {exc}") from exc
    source = paths.validate_input(media)
    try:
        digest = sha256_file(source)
    except OSError as exc:
        raise InputError(str(media), "cannot hash media for layout record") from exc
    if digest != document.get("media_sha256"):
        raise InputError(str(target), "layout record media hash does not match the media")
    if (layout.width, layout.height) != (width, height):
        raise InputError(str(target), "layout size does not match the render's display size")
    return LayoutRecord(layout, measured, reason, words, face_windows, element_ids, labels)


def check_caption_timing(
    record: LayoutRecord, *, fps: float | Fraction, duration_s: float, thresholds: Thresholds,
) -> list[Finding]:
    """Check each phrase's appearance and departure against the words it shows."""
    findings = []
    limit = thresholds.caption_offset_fail_ms
    captions = sorted(record.layout.captions, key=lambda item: record.words[item.id][0].start_s)
    next_starts = {
        caption.id: record.words[following.id][0].start_s
        for caption, following in zip(captions, captions[1:], strict=False)
    }
    for caption in record.layout.captions:
        words = record.words[caption.id]
        first = words[0]
        offset_ms = (caption.start_frame / float(fps) - first.start_s) * 1000
        message = None
        if offset_ms > limit or offset_ms < -limit:
            direction = "after" if offset_ms > 0 else "before"
            message = (
                f'Caption "{caption.text}" appears {abs(offset_ms) / 1000:.1f} s '
                f"{direction} its first word at {format_time(first.start_s)}."
            )
        else:
            for word in words:
                if (word.start_s < duration_s
                        and (word.start_s - caption.end_frame / float(fps)) * 1000 > limit):
                    message = (
                        f'Caption "{caption.text}" leaves before the word "{word.text}" '
                        f"is said at {format_time(word.start_s)}."
                    )
                    break
            spoken_words = [word for word in words if word.start_s < duration_s]
            last = max(spoken_words, key=lambda word: word.end_s, default=None)
            if message is None and last is not None:
                hold_end = caption_phrase_end(
                    first.start_s, max(word.start_s for word in spoken_words),
                    next_start_s=next_starts.get(caption.id),
                )
                end = min(last.end_s, hold_end, duration_s)
                leaves_s = caption.end_frame / float(fps)
                if (end - leaves_s) * 1000 > limit:
                    message = (
                        f'Caption "{caption.text}" leaves before the word "{last.text}" '
                        f"ends at {format_time(end)}."
                    )
                elif (min(last.end_s, duration_s) - leaves_s) * 1000 > limit:
                    # Transcript ends often carry trailing silence, which must not block
                    # delivery; a word that is really still being said would, so flag it.
                    transcript_end = min(last.end_s, duration_s)
                    findings.append(Finding(
                        "captions.timing", "warn",
                        f'Caption "{caption.text}" leaves at {format_time(leaves_s)}; the '
                        f'transcript has "{last.text}" running to {format_time(transcript_end)}.',
                        time_s=first.start_s,
                        suggested_action=(
                            "Play that moment. Silence after the word is fine; if the word is "
                            "still being said, the caption leaves too early."
                        ),
                        evidence={"caption_id": caption.id, "start_frame": caption.start_frame,
                                  "end_frame": caption.end_frame,
                                  "transcript_end_s": round(transcript_end, 3)},
                    ))
        if message is not None:
            findings.append(Finding(
                "captions.timing", "fail", message, time_s=first.start_s,
                suggested_action=(
                    "Render the captions again from the current cut words, or check where "
                    "the caption layer is placed."
                ),
                evidence={"caption_id": caption.id, "offset_ms": round(offset_ms, 1),
                          "start_frame": caption.start_frame, "end_frame": caption.end_frame},
            ))
    return findings


def _card_elements(record: LayoutRecord) -> list[Element]:
    """Elements drawn above captions; backgrounds sit below them."""
    return [element for element in record.layout.elements if element.kind != "background"]


def check_caption_cards(record: LayoutRecord, *, fps: float | Fraction) -> list[Finding]:
    """Find cards, titles, and overlays too close to glyphs while both are drawn."""
    findings = []
    for element in _card_elements(record):
        element_id = record.element_ids.get(element.id, element.id)
        label = record.labels.get(element.id)
        name = f'{element.kind.capitalize()} "{element_id}"' + (f" ({label})" if label else "")
        for caption in record.layout.captions:
            start = max(element.start_frame, caption.start_frame)
            end = min(element.end_frame, caption.end_frame)
            if start >= end:
                continue
            glyphs = (box for box in caption.boxes if box.w > 0 and box.h > 0
                      and box.overlaps_frames(start, end))
            glyph = min(glyphs, key=lambda box: gap_px(element.box, box), default=None)
            if glyph is None or element.box.w <= 0 or element.box.h <= 0:
                continue
            gap = gap_px(element.box, glyph)
            if gap >= record.layout.caption_gap_px:
                continue
            start = max(start, glyph.start_frame if glyph.start_frame is not None else start)
            end = min(end, glyph.end_frame if glyph.end_frame is not None else end)
            time_s = start / float(fps)
            message = (
                f'{name} covers the caption "{caption.text}" at {format_time(time_s)}.'
                if gap < 0 else
                f'{name} is {gap:g} px from the caption "{caption.text}" '
                f'at {format_time(time_s)}; keep at least {record.layout.caption_gap_px:g} px.'
            )
            findings.append(Finding(
                "captions.cards", "fail",
                message,
                time_s=time_s,
                suggested_action="Move the card, title, or overlay further from the captions.",
                evidence={"element_id": element_id, "element_piece": element.id,
                          "element_kind": element.kind,
                          **({"label": label} if label else {}),
                          "caption_id": caption.id, "start_frame": start, "end_frame": end,
                          "element_box": element.box.to_dict(), "glyph_box": glyph.to_dict(),
                          "gap_px": round(gap, 2),
                          "required_px": round(record.layout.caption_gap_px, 2)},
            ))
    return findings


def caption_coverage(record: LayoutRecord) -> dict[str, dict]:
    """Say which caption checks have measured targets in this render."""
    reason = None
    if not record.measured:
        reason = f"caption layout was not measured: {record.reason}"
    elif not record.layout.captions:
        reason = "no captions in the layout"
    coverage = {check: {"ran": reason is None, "reason": reason} for check in _CHECKS}
    if reason is None and not _card_elements(record):
        coverage["captions.cards"] = {"ran": False, "reason": "no cards in the layout"}
    return coverage
