"""Resolve and render an ordered story on the engine's exact frame and sample grid."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import tempfile
import wave
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np

from yap_studio import audio, ffmpeg, paths, receipt
from yap_studio.canvas import canvas, require_supported, unsupported
from yap_studio.checks.platform import load_presets
from yap_studio.edl.model import EditList, Source
from yap_studio.edl.model import Segment as Span
from yap_studio.errors import InputError, YapError
from yap_studio.hdr import HdrConversion, conversion, hdr_kind
from yap_studio.plan import Plan, snap_frame
from yap_studio.project.anchors import (
    edge_is_exact,
    edge_words,
    edit_needs_edge_engine,
    join_edges,
    nominal_edge,
)
from yap_studio.project.model import (
    Edit,
    ProjectFileError,
    Removals,
    Segment,
    Word,
    Words,
    parse_document,
)
from yap_studio.project.receipts import (
    IdentityCache,
    StageRun,
    _identity,
    audio_stream_sha256,
    read_receipt,
)
from yap_studio.project.status import ProjectContext, cut_inputs
from yap_studio.stages import edges, repeats, take_levels
from yap_studio.style import resolve, resolve_style_data
from yap_studio.transcript import resolve_backend
from yap_studio.transcript.backends import Backend
from yap_studio.transcript.words import norm_words
from yap_studio.verify import verify

FPS = Fraction(30)
SAMPLE_RATE = 48_000
FADE_SAMPLES = 240


@dataclass(frozen=True)
class Piece:
    """One retained source range with exact output placement and review notes."""

    segment_id: str
    take: str
    src_t0: Fraction
    src_t1: Fraction
    src_f0: int
    src_f1: int
    out_f0: int
    n_frames: int
    sample_start: int
    sample_end: int
    words: tuple[str, ...]
    notes: tuple[str, ...]
    listen_in: bool
    listen_out: bool

    def to_dict(self) -> dict[str, Any]:
        """Return exact rational boundaries alongside readable seconds."""
        data = asdict(self)
        data.update(
            src_t0=str(self.src_t0),
            src_t1=str(self.src_t1),
            start_s=float(self.src_t0),
            end_s=float(self.src_t1),
        )
        return data


@dataclass(frozen=True)
class CutPlan:
    """A read-only resolution, including the snapshot consumed by the renderer."""

    pieces: tuple[Piece, ...]
    warnings: tuple[str, ...]
    listen_joins: tuple[dict[str, Any], ...]
    total_frames: int
    total_samples: int
    text_mismatches: tuple[dict[str, str], ...]
    segments: tuple[dict[str, Any], ...]
    context: ProjectContext
    sources: dict[str, Path]
    media: dict[str, ffmpeg.MediaInfo]
    source_words: dict[str, Words]
    words_sha256: dict[str, str]
    end_hold_s: float
    end_hold_source: str
    edges: tuple[dict[str, Any], ...] = ()
    word_corrections: tuple[dict[str, Any], ...] = ()
    # Settled authored ranges before removals and short wordless-piece merges.
    segment_ranges: tuple[tuple[str, Fraction, Fraction], ...] = ()
    pause_thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    pauses: dict[str, edges.PauseThreshold] = field(default_factory=dict)
    repeats: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Describe the resolved story without exposing implementation objects."""
        return {
            "schema": "yap-studio/cut-plan@1",
            "fps": "30/1",
            "sample_rate": SAMPLE_RATE,
            "pieces": [piece.to_dict() for piece in self.pieces],
            "segments": list(self.segments),
            "warnings": list(self.warnings),
            "repeats": list(self.repeats),
            "listen_joins": list(self.listen_joins),
            "edges": list(self.edges),
            "word_corrections": list(self.word_corrections),
            "pause_thresholds": self.pause_thresholds,
            "text_mismatches": list(self.text_mismatches),
            "total_frames": self.total_frames,
            "total_samples": self.total_samples,
            "duration_s": float(Fraction(self.total_frames, 1) / FPS),
            "end_hold_s": self.end_hold_s,
            "end_hold_source": self.end_hold_source,
        }


class _ReadOnlyIdentity(IdentityCache):
    """Honor recorded ingest copies without creating identity or lock files."""

    def sha256(self, path: Path) -> str:
        """Hash a source directly, without updating project bookkeeping."""
        source = paths.validate_input(path)
        for _ in range(3):
            before = _identity(source)
            digest = receipt.sha256_file(source)
            if _identity(source) == before:
                return digest
        raise InputError(source.name, "file changed while hashing; wait and retry")

    def audio_sha256(self, path: Path, ordinal: int) -> str:
        """Hash selected audio directly, without updating project bookkeeping."""
        source = paths.validate_input(path)
        for _ in range(3):
            before = _identity(source)
            digest = audio_stream_sha256(source, ordinal)
            if _identity(source) == before:
                return digest
        raise InputError(source.name, "file changed while hashing; wait and retry")


def _fraction(value: float | int) -> Fraction:
    """Retain decimal source times without introducing binary float arithmetic."""
    return Fraction(str(value))


def _document(context: ProjectContext, name: str, relative: str) -> Any:
    """Validate the bytes held by the shared project snapshot."""
    try:
        return parse_document(name, context.json(relative), filename=relative)
    except (OSError, ValueError, UnicodeError) as exc:
        if isinstance(exc, YapError):
            raise
        raise ProjectFileError(relative, "$", "cannot read a valid project document") from exc


def _subtract(
    first: Fraction,
    last: Fraction,
    ranges: list[tuple[Fraction, Fraction]],
) -> list[tuple[Fraction, Fraction]]:
    """Subtract the union of applied ranges, preserving source order."""
    retained = []
    cursor = first
    for left, right in sorted(ranges):
        if right <= cursor or left >= last:
            continue
        if left > cursor:
            retained.append((cursor, min(left, last)))
        cursor = max(cursor, right)
        if cursor >= last:
            break
    if cursor < last:
        retained.append((cursor, last))
    return retained


def _word_ids(words: Words, first: Fraction, last: Fraction) -> tuple[str, ...]:
    """Retain words whose exact midpoints lie within the piece's half-open range."""
    return tuple(
        word.id
        for word in words.words
        if first <= (_fraction(word.start) + _fraction(word.end)) / 2 < last
    )


def _segment_word_ids(
    segment: Segment, words: Words, first: Fraction, last: Fraction,
) -> tuple[str, ...]:
    """Use planned word bounds and midpoint membership on time-anchored sides."""
    if segment.from_.word is None and segment.to.word is None:
        return _word_ids(words, first, last)
    ids = [word.id for word in words.words]
    first_index = ids.index(segment.from_.word) if segment.from_.word is not None else 0
    last_index = ids.index(segment.to.word) + 1 if segment.to.word is not None else len(ids)
    return tuple(
        word.id for word in words.words[first_index:last_index]
        if (segment.from_.word is not None
            or first <= (_fraction(word.start) + _fraction(word.end)) / 2)
        and (segment.to.word is not None
             or (_fraction(word.start) + _fraction(word.end)) / 2 < last)
    )


def _expected_segment_text(text: str, words: tuple[Word, ...], removed: set[str]) -> str:
    """Drop authored tokens aligned to removed source IDs, preserving unmatched text."""
    if not any(word.id in removed for word in words):
        return text
    authored = norm_words(text)
    source = [(token, word.id) for word in words for token in norm_words(word.text)]
    matcher = difflib.SequenceMatcher(None, authored, [token for token, _ in source],
                                     autojunk=False)
    aligned = {
        block.a + offset: source[block.b + offset][1]
        for block in matcher.get_matching_blocks()
        for offset in range(block.size)
    }
    if len(aligned) != len(authored):
        return text
    kept = [token for index, token in enumerate(authored) if aligned[index] not in removed]
    return " ".join(kept) if len(kept) != len(authored) else text


def _segment_piece_words(
    segment: Segment,
    words: Words,
    pieces: list[Piece],
    removals: list[tuple[Fraction, Fraction]],
    *,
    planned_ids: tuple[str, ...],
) -> list[Piece]:
    """Honor planned word bounds while keeping midpoint ownership at inner joins."""
    if not pieces:
        return []
    if segment.from_.word is None and segment.to.word is None:
        return [replace(piece, words=_word_ids(words, piece.src_t0, piece.src_t1))
                for piece in pieces]
    planned = set(planned_ids)
    memberships: list[list[str]] = [[] for _ in pieces]
    for word in words.words:
        if word.id not in planned:
            continue
        midpoint = (_fraction(word.start) + _fraction(word.end)) / 2
        if any(left <= midpoint < right for left, right in removals):
            continue
        index = next((index for index, piece in enumerate(pieces)
                      if piece.src_t0 <= midpoint < piece.src_t1), None)
        if index is None:
            if segment.from_.word is not None and midpoint < pieces[0].src_t0:
                index = 0
            elif segment.to.word is not None and midpoint >= pieces[-1].src_t1:
                index = len(pieces) - 1
        if index is not None:
            memberships[index].append(word.id)
    return [replace(piece, words=tuple(ids))
            for piece, ids in zip(pieces, memberships, strict=True)]


def _piece_word_spans(piece: Piece, words: Words) -> list[tuple[Word, Fraction, Fraction]]:
    """Clamp clocks, borrowing or sharing edge time while preserving source order."""
    kept = [word for word in words.words if word.id in piece.words]
    spans: list[tuple[Fraction, Fraction]] = []
    for word in kept:
        first = min(piece.src_t1, max(piece.src_t0, _fraction(word.start)))
        last = min(piece.src_t1, max(piece.src_t0, _fraction(word.end)))
        if spans:
            previous_first, previous_last = spans[-1]
            first = max(previous_first, first)
            spans[-1] = (previous_first, min(previous_last, first))
        spans.append((first, max(first, last)))
    for start in (True, False):
        # Mirroring the end edge lets both sides borrow time by the same rule.
        ordered = kept if start else kept[::-1]
        placed = spans if start else [(-last, -first) for first, last in spans[::-1]]
        boundary = piece.src_t0 if start else -piece.src_t1
        opposite = piece.src_t1 if start else -piece.src_t0
        count = 0
        for word in ordered:
            if (_fraction(word.end) <= piece.src_t0 if start
                    else _fraction(word.start) >= piece.src_t1):
                count += 1
            else:
                break
        if not count:
            continue
        shared = min(count + 1, len(placed))
        limit = placed[count][1] if count < len(placed) else opposite
        if opposite - boundary < shared / FPS:
            shared, limit = len(placed), opposite
        if limit - boundary < shared / FPS:
            # Approximate clocks must not prevent a cut when frame-sized spans cannot fit.
            width = (limit - boundary) / shared
            placed[:shared] = [(boundary + index * width, boundary + (index + 1) * width)
                               for index in range(shared)]
        else:
            limit = placed[count][0] if count < len(placed) else opposite
            if limit - boundary < count / FPS and count < len(placed):
                neighbour_end = placed[count][1]
                limit = min(neighbour_end, boundary + count / FPS)
                placed[count] = (limit, neighbour_end)
            cursor = boundary
            for index, word in enumerate(ordered[:count]):
                duration = _fraction(word.end) - _fraction(word.start)
                last = min(cursor + max(duration, 1 / FPS), limit - (count - index - 1) / FPS)
                placed[index] = (cursor, last)
                cursor = last
        spans = placed if start else [(-last, -first) for first, last in placed[::-1]]
    return [(word, first, last) for word, (first, last) in zip(kept, spans, strict=True)]


def _moved_edge_words(
    segment: Segment, words: Words, pieces: list[Piece],
) -> dict[str, tuple[str, ...]]:
    """Expose edge clipping and neighbour timing adjustments in the plan receipt."""
    moved: dict[str, tuple[str, ...]] = {}
    if not pieces:
        return moved
    for anchor, start, piece in ((segment.from_, True, pieces[0]),
                                 (segment.to, False, pieces[-1])):
        if anchor.word is None:
            continue
        ids: list[str] = []
        for word, first, last in _piece_word_spans(piece, words):
            clock = _fraction(word.start if start else word.end)
            clipped = min(piece.src_t1, max(piece.src_t0, clock))
            changed = (clock < piece.src_t0 or first != clipped) if start else (
                clock > piece.src_t1 or last != clipped
            )
            if changed:
                ids.append(word.id)
        if ids:
            moved["from" if start else "to"] = tuple(ids)
    return moved


def _listen_boundary(
    words: Words, point: Fraction, removals: Removals, take: str, *, include_touching: bool = True,
) -> bool:
    """Carry requested listening and touching-word flags onto removal boundaries."""
    touching = any(_fraction(word.end) == point for word in words.words) and any(
        _fraction(word.start) == point for word in words.words
    )
    requested = any(
        item.status == "applied"
        and item.take == take
        and item.listen
        and point in {_fraction(item.start), _fraction(item.end)}
        for item in removals.removals
    )
    return (include_touching and touching) or requested


def _short_pieces(pieces: list[Piece], warnings: list[str]) -> list[Piece]:
    """Fold wordless fragments into adjacent ranges without restoring a removal."""
    index = 0
    while index < len(pieces):
        piece = pieces[index]
        if piece.src_t1 - piece.src_t0 >= Fraction(1, 4):
            index += 1
            continue
        if piece.words:
            note = "Short piece retained because it holds a word."
            pieces[index] = replace(piece, notes=(*piece.notes, note))
            index += 1
            continue
        if (index > 0 and pieces[index - 1].take == piece.take
                and pieces[index - 1].src_t1 == piece.src_t0):
            neighbour_index = index - 1
        elif (index + 1 < len(pieces) and pieces[index + 1].take == piece.take
              and pieces[index + 1].src_t0 == piece.src_t1):
            neighbour_index = index + 1
        else:
            note = "short piece kept: no contiguous neighbour"
            pieces[index] = replace(piece, notes=(*piece.notes, note))
            warnings.append(f"Segment {piece.segment_id!r}: {note}.")
            index += 1
            continue
        neighbour = pieces[neighbour_index]
        note = f"Short wordless piece from segment {piece.segment_id!r} merged at this join."
        pieces[neighbour_index] = replace(
            neighbour,
            src_t0=min(piece.src_t0, neighbour.src_t0),
            src_t1=max(piece.src_t1, neighbour.src_t1),
            src_f0=min(piece.src_f0, neighbour.src_f0),
            src_f1=max(piece.src_f1, neighbour.src_f1),
            listen_in=piece.listen_in if index < neighbour_index else neighbour.listen_in,
            listen_out=piece.listen_out if index > neighbour_index else neighbour.listen_out,
            notes=(*neighbour.notes, *piece.notes, note),
        )
        pieces.pop(index)
        index = max(0, index - 1)
    return pieces


def _edge_backend(context: ProjectContext) -> Backend | None:
    """The project's speech engine for edge checks, or None when none is usable."""
    settings = context.project.asr or {}
    try:
        return resolve_backend(settings.get("engine") or "auto", settings.get("model"))
    except YapError as exc:
        if exc.code != "speech_engine_not_found":
            raise
        return None


def _settle_edges(
    context: ProjectContext,
    segments: tuple[Segment, ...],
    source_words: dict[str, Words],
    media: dict[str, ffmpeg.MediaInfo],
    sources: dict[str, Path],
    warnings: list[str],
) -> tuple[
    dict[str, tuple[tuple[Fraction, bool], tuple[Fraction, bool]]],
    tuple[dict[str, Any], ...], dict[str, edges._Take],
]:
    """Move every story edge into real silence, listening for the words it keeps."""
    nominal: dict[str, tuple[Fraction, Fraction]] = {}
    for segment in segments:
        words = source_words[segment.take]
        duration = _fraction(cast(float, media[segment.take].video_duration_s))
        first = nominal_edge(segment, segment.from_, words, duration, start=True)
        last = nominal_edge(segment, segment.to, words, duration, start=False)
        if last <= first:
            raise ProjectFileError("edit.json", f"segment {segment.id!r}", "end must follow start")
        nominal[segment.id] = (first, last)
    joined = join_edges(segments, source_words, nominal, warnings)
    requests: list[edges.EdgeRequest] = []
    for segment in segments:
        words = source_words[segment.take]
        for anchor, start in ((segment.from_, True), (segment.to, False)):
            if (segment.id, start) in joined:
                continue
            exact = edge_is_exact(anchor, start=start)
            word = next((item for item in words.words if item.id == anchor.word), None)
            requests.append(
                edges.EdgeRequest(
                    segment.id,
                    segment.take,
                    start,
                    nominal[segment.id][0 if start else 1],
                    () if exact else edge_words(words, anchor, start=start),
                    exact,
                    None if word is None else _fraction(word.start),
                    None if word is None else _fraction(word.end),
                    other=nominal[segment.id][1 if start else 0],
                    duration=_fraction(cast(float, media[segment.take].video_duration_s)),
                )
            )
    takes = {
        take_id: edges.load_take(
            sources[take_id], _selected_stream(context, take_id),
            words=[(word.start, word.end) for word in source_words[take_id].words],
        )
        for take_id in sorted({request.take for request in requests})
    }
    needs_engine = edit_needs_edge_engine(segments, source_words, joined=joined)
    backend = _edge_backend(context) if needs_engine else None
    results, notes = edges.place(requests, takes, backend, language=context.project.language)
    warnings.extend(notes)
    settled = {
        segment.id: ((nominal[segment.id][0], False), (nominal[segment.id][1], False))
        for segment in segments
    }
    for result in results:
        first, last = settled[result.segment]
        if result.start:
            settled[result.segment] = ((result.point, result.listen), last)
        else:
            settled[result.segment] = (first, (result.point, result.listen))
    crossed = {segment_id for segment_id, (first, last) in settled.items()
               if first[0] >= last[0]}
    for segment_id in sorted(crossed):
        warnings.append(
            f"Segment {segment_id!r}: silence placements crossed; keeping word-time edges "
            "for listening."
        )
        settled[segment_id] = ((nominal[segment_id][0], True), (nominal[segment_id][1], True))
    results = [
        replace(result, point=result.nominal, how="unconfirmed", listen=True, heard=None)
        if result.segment in crossed else result
        for result in results
    ]
    rows = tuple(result.to_dict() for result in results)
    rows += tuple(
        {"segment": segment_id, "edge": "from" if start else "to", "how": "continuous",
         "listen": False}
        for segment_id, start in sorted(joined)
    )
    return settled, rows, takes


def _resolve(context: ProjectContext) -> CutPlan:
    """Resolve source edges and removals before handing placement to Plan.build."""
    require_supported(context)
    frame = canvas(context.project.shape)
    edit = cast(Edit, _document(context, "edit", "edit.json"))
    try:
        context.read("removals.json")
    except FileNotFoundError:
        removals = Removals(())
    else:
        removals = cast(Removals, _document(context, "removals", "removals.json"))
    sources: dict[str, Path] = {}
    media: dict[str, ffmpeg.MediaInfo] = {}
    source_words: dict[str, Words] = {}
    hashes: dict[str, str] = {}
    warnings: list[str] = []
    for take_id in sorted({segment.take for segment in edit.segments}):
        source = paths.validate_input(context.take_path(take_id))
        sources[take_id] = source
        info = media[take_id] = ffmpeg.probe(source, first_frame=True)
        if context.project.shape == "horizontal" and info.orientation != "horizontal":
            unsupported(f"non-horizontal take {take_id}", "Use horizontal source takes")
        if not info.has_video or not info.fps or info.video_duration_s is None:
            raise InputError(take_id, "source has no readable picture duration")
        raw = context.read(f"ingest/{take_id}/words.json")
        hashes[take_id] = hashlib.sha256(raw).hexdigest()
        words = source_words[take_id] = cast(
            Words, _document(context, "words", f"ingest/{take_id}/words.json")
        )
        if words.take != take_id:
            raise ProjectFileError("edit.json", take_id, "source words are stale; ingest again")
        if words.take_sha256 != context.take_sha256(take_id):
            record = context.ingested_take(take_id)
            if record is None or record.get("sha256") != words.take_sha256:
                raise ProjectFileError("edit.json", take_id, "source words are stale; ingest again")
            if context.check_take(take_id, record, since="ingest"):
                warnings.append(
                    f"Take {take_id} renders from a stand-in file with the identical audio stream."
                )
        if edit.words_sha256.get(take_id) != hashes[take_id]:
            raise ProjectFileError(
                "edit.json",
                f"words_sha256.{take_id}",
                "word snapshot changed; re-read words and update the edit",
            )
        if info.fps != FPS:
            warnings.append(f"Take {take_id} converts from {info.fps} fps to 30 fps.")
        try:
            manifest = context.json("ingest/ingest.json")
        except FileNotFoundError:
            manifest = {}
        if any(item.get("id") == take_id and item.get("vfr") for item in manifest.get("takes", [])):
            warnings.append(f"Take {take_id} has variable frame timing and conforms to 30 fps.")
        if (info.width and info.height
                and max(frame.width / info.width, frame.height / info.height) > 1):
            warnings.append(f"Take {take_id} is upscaled for the "
                            f"{frame.width}x{frame.height} cover crop.")
    pieces: list[Piece] = []
    segment_word_ids: dict[str, tuple[str, ...]] = {}
    segment_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, str]] = []
    settled, edge_rows, takes = _settle_edges(
        context, edit.segments, source_words, media, sources, warnings,
    )
    # Settle authored anchors on their snapshot, then project speech on the applied cut's clock.
    pauses: dict[str, edges.PauseThreshold] = {}
    word_corrections: list[dict[str, Any]] = []
    for take, snapshot in source_words.items():
        take_audio = takes.pop(take, None)
        if take_audio is None:
            take_audio = edges.load_take(
                sources[take], _selected_stream(context, take),
                words=[(word.start, word.end) for word in snapshot.words],
            )
        if take_audio.pause is not None:
            pauses[take] = take_audio.pause
        voiced = edges.voiced_words(take_audio, snapshot.words)
        removed_ranges = [
            (item.start, item.end) for item in removals.removals
            if item.status == "applied" and item.take == take
        ]
        voiced, correction_rows, correction_warnings = edges.realign_silent_words(
            take_audio, voiced, removed_ranges, lambda: _edge_backend(context),
            take_id=take, language=context.project.language,
        )
        word_corrections.extend(correction_rows)
        warnings.extend(correction_warnings)
        refined = {word["id"]: word for item in removals.removals
                   if item.status == "applied" and item.take == take
                   for word in item.evidence.get("refined_source_words", [])}
        source_words[take] = replace(snapshot, words=tuple(
            replace(word, start=refined[word.id]["start"],
                    end=refined[word.id]["end"]) if word.id in refined else word
            for word in voiced
        ))
    applied_ranges = {
        take: [(_fraction(item.start), _fraction(item.end))
               for item in removals.removals if item.status == "applied" and item.take == take]
        for take in source_words
    }
    applied_word_ids = {
        take: {word_id for item in removals.removals
               if item.status == "applied" and item.take == take and item.kind != "silence"
               for word_id in item.words}
        for take in source_words
    }
    for segment in edit.segments:
        words = source_words[segment.take]
        (first, listen_in), (last, listen_out) = settled[segment.id]
        if last <= first:
            raise ProjectFileError("edit.json", f"segment {segment.id!r}", "end must follow start")
        segment_word_ids[segment.id] = _segment_word_ids(
            segment, words, Fraction(snap_frame(first, FPS), 1) / FPS,
            Fraction(snap_frame(last, FPS), 1) / FPS,
        )
        ranges = applied_ranges[segment.take]
        for left, right in _subtract(first, last, ranges):
            f0, f1 = snap_frame(left, FPS), snap_frame(right, FPS)
            if f1 <= f0:
                warnings.append(f"Segment {segment.id!r} has a piece shorter than one frame.")
                continue
            split_in = any(
                _fraction(word.start) < left < _fraction(word.end) for word in words.words
            )
            split_out = any(
                _fraction(word.start) < right < _fraction(word.end) for word in words.words
            )
            pieces.append(
                Piece(
                    segment.id,
                    segment.take,
                    left,
                    right,
                    f0,
                    f1,
                    0,
                    f1 - f0,
                    0,
                    0,
                    _word_ids(words, left, right),
                    (segment.note,) if segment.note else (),
                    listen_in or _listen_boundary(
                        words, left, removals, segment.take, include_touching=False
                    )
                    if left == first
                    else split_in or _listen_boundary(words, left, removals, segment.take),
                    listen_out or _listen_boundary(
                        words, right, removals, segment.take, include_touching=False
                    )
                    if right == last
                    else split_out or _listen_boundary(words, right, removals, segment.take),
                )
            )
    pieces = _short_pieces(pieces, warnings)
    pieces = [
        replace(
            piece,
            src_t0=Fraction(piece.src_f0, 1) / FPS,
            src_t1=Fraction(piece.src_f1, 1) / FPS,
        )
        for piece in pieces
    ]
    moved_words: dict[tuple[str, str], tuple[str, ...]] = {}
    for segment in edit.segments:
        indices = [index for index, piece in enumerate(pieces) if piece.segment_id == segment.id]
        retained = _segment_piece_words(
            segment, source_words[segment.take], [pieces[index] for index in indices],
            applied_ranges[segment.take],
            planned_ids=segment_word_ids[segment.id],
        )
        for index, piece in zip(indices, retained, strict=True):
            pieces[index] = replace(piece, words=tuple(
                word_id for word_id in piece.words if word_id not in applied_word_ids[segment.take]
            ))
        moves = _moved_edge_words(segment, source_words[segment.take], retained)
        moved_words.update({(segment.id, edge): ids for edge, ids in moves.items()})
    for index, piece in enumerate(pieces):
        boundaries = applied_ranges[piece.take]
        length = Fraction(piece.src_f1 - piece.src_f0, 1) / FPS
        touches_trim = any(
            piece.src_f0 == snap_frame(right, FPS) or
            piece.src_f1 == snap_frame(left, FPS)
            for left, right in boundaries
        )
        if not piece.words and Fraction(1, 4) <= length < Fraction(1, 2) and touches_trim:
            seconds = float(length)
            warning = (f"Segment {piece.segment_id!r}: a {seconds:.2f} s piece at "
                       f"{float(piece.src_t0):.2f} s in take {piece.take} "
                       "holds no word after a trim; "
                       "it may be only a breath. Listen, or widen the trim.")
            warnings.append(warning)
            pieces[index] = replace(piece, notes=(*piece.notes, warning))
    edge_rows = tuple(
        {**row, "moved_word": ids[0 if row["edge"] == "from" else -1],
         **({"moved_words": list(ids)} if len(ids) > 1 else {})}
        if (ids := moved_words.get((row["segment"], row["edge"]))) else row
        for row in edge_rows
    )
    for segment in edit.segments:
        words = source_words[segment.take]
        retained_ids = {
            word_id for piece in pieces if piece.segment_id == segment.id for word_id in piece.words
        }
        retained_words = [word for word in words.words if word.id in retained_ids]
        spoken = " ".join(word.text for word in retained_words)
        segment_rows.append(
            {
                "id": segment.id,
                "take": segment.take,
                "text": spoken,
                "words": [word.id for word in retained_words],
            }
        )
        if segment.text is not None:
            source_ids = set(segment_word_ids[segment.id])
            removed = {word_id for item in removals.removals
                       if item.status == "applied" and item.take == segment.take
                       and item.kind != "silence"
                       for word_id in item.words}
            expected = _expected_segment_text(
                segment.text, tuple(word for word in words.words if word.id in source_ids), removed,
            )
            if norm_words(expected) != norm_words(spoken):
                mismatches.append({"segment": segment.id, "expected": expected, "actual": spoken})
                warnings.append(
                    f"Segment {segment.id!r} expects {expected!r}, transcript is {spoken!r}."
                )
    if not pieces:
        raise ProjectFileError("edit.json", "segments", "the cut retains no picture frames")
    engine = Plan.build(
        EditList(
            "yap",
            context.project_dir / "edit.json",
            {key: Source(key, source, _selected_stream(context, key))
             for key, source in sources.items()},
            [
                Span(index, piece.take, piece.src_t0, piece.src_t1)
                for index, piece in enumerate(pieces, 1)
            ],
            FPS,
            [],
            [],
        ),
        FPS,
        SAMPLE_RATE,
    )
    placed = tuple(
        replace(
            piece,
            out_f0=planned.out_f0,
            n_frames=planned.n_frames,
            sample_start=planned.sample_start,
            sample_end=planned.sample_end,
        )
        for piece, planned in zip(pieces, engine.segments, strict=True)
    )
    retained_words: dict[tuple[str, str], repeats.RepeatWord] = {}
    for piece in placed:
        offset = Fraction(piece.out_f0, 1) / FPS - piece.src_t0
        for word, first, last in _piece_word_spans(piece, source_words[piece.take]):
            key = (piece.segment_id, word.id)
            if key not in retained_words:
                retained_words[key] = repeats.RepeatWord(
                    word.id, word.text, float(offset + first), float(offset + last),
                    piece.segment_id,
                )
    repeated = repeats.find_repeats(list(retained_words.values()))
    warnings.extend(repeats.repeat_warnings(repeated))
    cut_frames = {cut.out_frame for cut in engine.cuts}
    listen = tuple(
        {
            "frame": right.out_f0,
            "time_s": right.out_f0 / 30,
            "left": left.segment_id,
            "right": right.segment_id,
        }
        for left, right in zip(placed, placed[1:], strict=False)
        if right.out_f0 in cut_frames and (left.listen_out or right.listen_in)
    )
    if edit.end_hold_s is not None:
        hold = _fraction(edit.end_hold_s)
        hold_source = "edit.end_hold_s"
    else:
        spec = context.project.style or "preset:clean_minimal"
        style_path = context.project_dir / spec
        style = (
            resolve_style_data(context.json(spec)) if style_path.is_file() else resolve(spec).data
        )
        hold = _fraction(style["internal"]["end_hold_s"])
        hold_source = "style.internal.end_hold_s"
    if context.project.platform != "none":
        limit = load_presets()[context.project.platform].get("duration", {}).get("max_s")
        length = Fraction(engine.total_frames, 1) / FPS + hold
        if limit is not None and length > _fraction(limit):
            warnings.append(
                f"Cut plus end hold is {float(length):g} s, exceeding the "
                f"{context.project.platform} maximum of {limit:g} s."
            )
    return CutPlan(
        placed,
        tuple(warnings),
        listen,
        engine.total_frames,
        engine.total_samples,
        tuple(mismatches),
        tuple(segment_rows),
        context,
        sources,
        media,
        source_words,
        hashes,
        float(hold),
        hold_source,
        edge_rows,
        tuple(word_corrections),
        tuple(
            (identifier, Fraction(snap_frame(first[0], FPS), 1) / FPS,
             Fraction(snap_frame(last[0], FPS), 1) / FPS)
            for identifier, (first, last) in settled.items()
        ),
        pause_thresholds={take: pause.to_dict() for take, pause in pauses.items()},
        pauses=pauses,
        repeats=tuple(repeated),
    )


def resolve_cut(project_dir: Path) -> CutPlan:
    """Read a project snapshot and resolve its story without writing any files."""
    with paths.operation():
        context = ProjectContext(project_dir)
        context.identity = _ReadOnlyIdentity(context.project_dir)
        return _resolve(context)


def edit_check(project_dir: Path) -> dict:
    """Describe retained pieces, segment words, length, and joins requiring listening."""
    return resolve_cut(project_dir).to_dict()


def _timeout(seconds: float) -> float:
    """Scale each new media process deadline to the duration it may decode."""
    return max(60.0, 30.0 + 6.0 * seconds)


def _picture(piece: Piece, plan: CutPlan, target: Path, hdr: HdrConversion | None) -> None:
    """Conform on the source frame clock before selecting and scaling a piece."""
    info = plan.media[piece.take]
    take = next(take for take in plan.context.project.takes if take.id == piece.take)
    frame = canvas(plan.context.project.shape)
    scale = (f"scale={frame.width}:{frame.height}:force_original_aspect_ratio=increase:"
             "force_divisible_by=2:out_range=tv")
    if hdr is None and info.color_space in {"bt470bg", "smpte170m"}:
        scale += ":in_color_matrix=bt601:out_color_matrix=bt709"
    # Other containers may correct discontinuities; stream start can precede decoded picture.
    can_seek = ("mov" in (info.format_name or "").split(",")
                and info.video_start_s is not None
                and info.video_first_frame_s == info.video_start_s)
    # Keep keyframe preroll and absolute timestamps so seeking preserves the source FPS grid.
    seek = max(0, piece.src_f0 // 30 - 1) if can_seek else 0
    first, last = piece.src_f0 - seek * 30, piece.src_f1 - seek * 30
    filters = [] if hdr is None else [hdr.filter_prefix]
    filters.extend(
        [
            f"setpts=PTS-round({info.video_start_s}/TB)" if seek else "setpts=PTS-STARTPTS",
            f"fps=30:start_time={seek}:round=near",
            f"trim=start_frame={first}:end_frame={last}",
            "setpts=PTS-STARTPTS",
            scale,
            f"crop={frame.width}:{frame.height}:x=(iw-ow)*{take.crop_x:.9f}:y=(ih-oh)/2",
            "setsar=1",
            "format=yuv420p",
        ]
    )
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            *(hdr.input_args if hdr else []),
            *(["-ss", str(seek + info.video_offset_s), "-noaccurate_seek", "-copyts"]
              if seek else []),
            "-i",
            str(plan.sources[piece.take]),
            "-map",
            f"0:{info.video_index}",
            "-an",
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(piece.n_frames),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-bf",
            "0",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            str(target),
        ],
        timeout_s=_timeout(info.duration_s or float(piece.src_t1)),
    )
    if ffmpeg.probe(target).video_frames != piece.n_frames:
        raise YapError(
            "Piece frame count disagrees with the plan.",
            code="cut_frame_mismatch",
            suggested_action="Inspect source timestamps and retained work files.",
        )


def _selected_stream(context: ProjectContext, take_id: str) -> int:
    """Select the same audio ordinal for rendering and source verification."""
    return context.selected_audio_stream(take_id)


def _audio_piece(piece: Piece, plan: CutPlan) -> np.ndarray:
    """Trim selected PCM on the same container clock that verification decodes."""
    info = plan.media[piece.take]
    selected = _selected_stream(plan.context, piece.take)
    result = ffmpeg.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            f"a:{selected}",
            "-show_streams",
            "-of",
            "json",
            str(plan.sources[piece.take]),
        ],
        timeout_s=_timeout(info.duration_s or 0),
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise InputError(piece.take, "selected audio stream is missing; ingest again")
    stream = streams[0]
    channels = int(stream["channels"])
    if channels not in {1, 2}:
        raise InputError(piece.take, "cut requires mono or stereo audio; supply a matching take")
    offset = _fraction(float(stream.get("start_time", info.format_start_s)))
    offset -= _fraction(info.format_start_s)
    offset_samples = snap_frame(offset, Fraction(SAMPLE_RATE))
    start = snap_frame(_fraction(info.video_offset_s) + piece.src_t0, Fraction(SAMPLE_RATE))
    count = piece.sample_end - piece.sample_start
    source = plan.sources[piece.take]
    source_stat = source.stat()
    native_rate = int(stream["sample_rate"])
    clock = audio._timeline(
        str(source.resolve()),
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ino,
        source_stat.st_ctime_ns,
        int(stream["index"]),
        native_rate,
        info.format_start_s,
    )
    filters = [f"aresample={SAMPLE_RATE}", "asetpts=PTS-STARTPTS"]
    if clock.fills:
        # Timestamp gaps need insertion before trimming. Reuse the verifier's
        # native-clock arithmetic rather than allowing a resampler to drop overlaps.
        expected = round(Fraction(clock.decoded_samples * SAMPLE_RATE, native_rate))
        filters.append(f"atrim=start_sample=0:end_sample={expected}")
    elif offset_samples > 0:
        filters.append(f"adelay={offset_samples}S:all=1")
    elif offset_samples < 0:
        filters.append(f"atrim=start_sample={-offset_samples}")
    if not clock.fills:
        filters.extend(
            [
                f"apad=whole_len={start + count}",
                f"atrim=start_sample={start}:end_sample={start + count}",
                "asetpts=PTS-STARTPTS",
            ]
        )
    result = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(plan.sources[piece.take]),
            "-map",
            f"0:a:{selected}",
            "-vn",
            "-af",
            ",".join(filters),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(channels),
            "-f",
            "f32le",
            "-",
        ],
        timeout_s=_timeout(info.duration_s or float(piece.src_t1)),
        binary=True,
    )
    sound = np.frombuffer(result.stdout, dtype="<f4").copy().reshape(-1, channels)
    if clock.fills:
        sound = audio._fill_gaps(sound, clock, native_rate, SAMPLE_RATE, source)
        first = start - offset_samples
        exact = np.zeros((count, channels), dtype=np.float32)
        left, right = max(0, first), min(len(sound), first + count)
        if right > left:
            exact[left - first : right - first] = sound[left:right]
        sound = exact
    if len(sound) != count:
        raise YapError(
            "Piece audio count disagrees with the plan.",
            code="cut_sample_mismatch",
            suggested_action="Inspect the selected audio stream and its timestamps.",
        )
    return sound


def _spans(plan: CutPlan, directory: Path) -> dict[str, Any]:
    """Write the engine's yap span shape with portable paths and exact fractions."""
    edit = cast(Edit, _document(plan.context, "edit", "edit.json"))
    authored = {segment.id: segment for segment in edit.segments}
    authored_frames = {
        identifier: (snap_frame(first, FPS), snap_frame(last, FPS))
        for identifier, first, last in plan.segment_ranges
    }
    words = {take: {word.id: word.text for word in snapshot.words}
             for take, snapshot in plan.source_words.items()}
    segments = []
    for index, piece in enumerate(plan.pieces):
        segment = authored[piece.segment_id]
        row = {"source": piece.take, "start": str(piece.src_t0), "end": str(piece.src_t1)}
        first = index == 0 or plan.pieces[index - 1].segment_id != piece.segment_id
        last = (index == len(plan.pieces) - 1
                or plan.pieces[index + 1].segment_id != piece.segment_id)
        frames = authored_frames.get(piece.segment_id)
        # Removal-created edges have no authored word anchor.
        if (first and frames is not None and piece.src_f0 == frames[0]
                and segment.from_.word is not None and piece.words):
            row["first_word"] = words[piece.take][piece.words[0]]
        if (last and frames is not None and piece.src_f1 == frames[1]
                and segment.to.word is not None and piece.words):
            row["last_word"] = words[piece.take][piece.words[-1]]
        segments.append(row)
    return {
        "sources": {
            key: {"path": os.path.relpath(source, directory), "audio_stream": selected}
            if (selected := _selected_stream(plan.context, key)) != 0
            else os.path.relpath(source, directory)
            for key, source in plan.sources.items()
        },
        "fps": "30/1",
        "segments": segments,
    }


def _cut_words(plan: CutPlan) -> dict[str, Any]:
    """Place source word occurrences on output time, clipping removed portions."""
    occurrences: dict[tuple[str, str], dict[str, Any]] = {}
    for piece in plan.pieces:
        for word, first, last in _piece_word_spans(piece, plan.source_words[piece.take]):
            output = Fraction(piece.out_f0, 1) / FPS
            item = {
                "id": word.id,
                "segment": piece.segment_id,
                "text": word.text,
                "start": float(output + first - piece.src_t0),
                "end": float(output + last - piece.src_t0),
                "conf": word.conf,
            }
            key = (word.id, piece.segment_id)
            if key in occurrences:
                occurrences[key]["end"] = item["end"]
            else:
                occurrences[key] = item
    data = {
        "schema": "yap-studio/cut-words@1",
        "time_base": "output",
        "words_sha256": plan.words_sha256,
        "words": list(occurrences.values()),
    }
    parse_document("cut-words", data)
    return data


def _write_json(path: Path, data: dict) -> None:
    """Write a staged document that StageRun will publish atomically."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _story_platform_report(report: dict) -> dict:
    """Keep delivery requirements advisory during story review."""
    changed = False
    for item in report.get("findings", []):
        if item["check"].startswith("platform.") and item["severity"] == "fail":
            item["severity"] = "warn"
            changed = True
    if not changed:
        return report
    for check, row in report.get("checks", {}).items():
        if check.startswith("platform.") and row.get("status") == "fail":
            row["status"] = "warn"
    severities = {item["severity"] for item in report.get("findings", [])}
    report["status"] = ("fail" if "fail" in severities else
                        "warn" if "warn" in severities else "pass")
    if "summary" in report:
        from yap_studio.report import summarize

        report["summary"].update(status=report["status"], **{
            severity: sum(item["severity"] == severity for item in report["findings"])
            for severity in ("fail", "warn", "info")
        })
        report["summary"]["text"] = summarize(report)
    return report


def _portable_report(value: Any, root: Path) -> Any:
    """Keep persisted verification evidence relative to its project."""
    if isinstance(value, dict):
        return {key: _portable_report(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_report(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), ".")
    return value


def cut(project_dir: Path, *, force: bool = False) -> dict:
    """Render exact frames and PCM, encode AAC once, and publish verified stage outputs."""
    with paths.operation():
        context = ProjectContext(project_dir)
        context.identity = _ReadOnlyIdentity(context.project_dir)
        require_supported(context)
        frame = canvas(context.project.shape)
        root = context.project_dir
        inputs = cut_inputs(context)
        run = StageRun(root, "cut")
        if not force and run.up_to_date(inputs):
            previous = read_receipt(root / "cut") or {}
            if isinstance(previous.get("plan"), dict):
                return {
                    **previous["plan"],
                    "status": "verified",
                    "noop": True,
                    "out": "cut/cut.mp4",
                    "receipt": "cut/receipt.json",
                    "verify": "cut/verify.json",
                    "conversions": previous.get("conversions", {}),
                    **({"take_levels": previous["take_levels"],
                        "warnings": [*previous["plan"]["warnings"],
                                     *take_levels.warnings(previous["take_levels"])]}
                       if "take_levels" in previous else {}),
                }
        plan = _resolve(context)
        conversions = {key: conversion(info) for key, info in plan.media.items()}
        conversion_rows = {
            key: {"kind": hdr_kind(plan.media[key]), **item.describe()}
            for key, item in conversions.items()
            if item is not None
        }
        work_root = paths.validate_output_dir(root / ".yap" / "work")
        work = Path(tempfile.mkdtemp(prefix=f"cut-{os.getpid()}-", dir=work_root))
        try:
            with run:
                for name, digest in inputs.items():
                    run.record_inputs(name, digest)
                pictures: list[Path] = []
                sounds: list[np.ndarray] = []
                for index, piece in enumerate(plan.pieces):
                    run.progress(
                        f"rendering piece {index + 1} of {len(plan.pieces)}",
                        index / len(plan.pieces) * 0.8,
                    )
                    target = work / f"piece-{index:04d}.mp4"
                    _picture(piece, plan, target, conversions[piece.take])
                    pictures.append(target)
                    sounds.append(_audio_piece(piece, plan))
                channels = max(sound.shape[1] for sound in sounds)
                sounds = [
                    np.repeat(sound, channels, axis=1) if sound.shape[1] == 1 else sound
                    for sound in sounds
                ]
                take_level_record = None
                by_take: dict[str, list[np.ndarray]] = {}
                for piece, sound in zip(plan.pieces, sounds, strict=True):
                    by_take.setdefault(piece.take, []).append(sound)
                if len(by_take) > 1:
                    words = {take: {word.id: word for word in snapshot.words}
                             for take, snapshot in plan.source_words.items()}
                    speech: dict[str, list[np.ndarray]] = {}
                    for piece, sound in zip(plan.pieces, sounds, strict=True):
                        spans = [(_fraction(words[piece.take][word].start),
                                  _fraction(words[piece.take][word].end)) for word in piece.words]
                        speech.setdefault(piece.take, []).append(
                            take_levels.word_audio(sound, spans, piece.src_t0, piece.src_t1),
                        )
                    measurements = {}
                    for take, parts in by_take.items():
                        measurements[take] = take_levels.measure_take(
                            np.concatenate(parts), np.concatenate(speech[take]), work, take,
                        )
                    take_level_record = take_levels.plan(
                        measurements, enabled=take_levels.match_enabled(context.project),
                    )
                    for index, piece in enumerate(plan.pieces):
                        gain = take_level_record["takes"][piece.take]["gain_db"]
                        if gain != 0.0:
                            sounds[index] = sounds[index] * np.float32(10 ** (gain / 20))
                unfaded = np.concatenate(sounds)
                splice_edges = [0, *(
                    right.sample_start
                    for left, right in zip(plan.pieces, plan.pieces[1:], strict=False)
                    if left.take != right.take or left.src_f1 != right.src_f0
                ), plan.total_samples]
                voice, _ = audio.splice(
                    unfaded,
                    list(zip(splice_edges, splice_edges[1:], strict=False)),
                    FADE_SAMPLES,
                )
                voice_path = work / "voice.wav"
                with wave.open(str(voice_path), "wb") as output:
                    output.setnchannels(channels)
                    output.setsampwidth(2)
                    output.setframerate(SAMPLE_RATE)
                    pcm = np.clip(np.rint(voice * 32768), -32768, 32767).astype("<i2")
                    output.writeframes(pcm.tobytes())
                listing = work / "pieces.txt"
                listing.write_text(
                    "".join(f"file '{picture.name}'\n" for picture in pictures), encoding="utf-8"
                )
                rendered = work / "cut.mp4"
                run.progress("encoding cut", 0.8)
                ffmpeg.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-f",
                        "concat",
                        "-safe",
                        "1",
                        "-i",
                        str(listing),
                        "-i",
                        str(voice_path),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-ar",
                        str(SAMPLE_RATE),
                        "-colorspace",
                        "bt709",
                        "-color_primaries",
                        "bt709",
                        "-color_trc",
                        "bt709",
                        "-movie_timescale",
                        "48000",
                        "-movflags",
                        "+faststart",
                        str(rendered),
                    ],
                    timeout_s=_timeout(plan.total_frames / 30),
                )
                run.progress("checking encoded audio", 0.85)
                info = ffmpeg.probe(rendered)
                _, decoded = audio.decode(rendered, sr=SAMPLE_RATE)
                audio_clock = audio.timeline(rendered)
                encoded_samples = round((info.audio_duration_s or 0) * SAMPLE_RATE)
                assertions = [
                    {
                        "id": "frames",
                        "pass": info.video_frames == plan.total_frames,
                        "expected": plan.total_frames,
                        "actual": info.video_frames,
                    },
                    {
                        "id": "pcm_samples",
                        "pass": len(voice) == plan.total_samples,
                        "expected": plan.total_samples,
                        "actual": len(voice),
                    },
                    {
                        "id": "aac_samples",
                        "pass": encoded_samples == plan.total_samples
                        and 0 <= len(decoded) - plan.total_samples < 1024
                        and not audio_clock.discontinuities,
                        "expected": plan.total_samples,
                        "actual": encoded_samples,
                        "decoded": len(decoded),
                        "terminal_padding": len(decoded) - plan.total_samples,
                    },
                    {
                        "id": "audio_contract",
                        "pass": (
                            info.audio_codec,
                            info.sample_rate,
                            info.channels,
                            info.audio_offset_s,
                            info.video_offset_s,
                        )
                        == ("aac", SAMPLE_RATE, channels, 0, 0),
                        "expected": {"codec": "aac", "sample_rate": SAMPLE_RATE,
                                     "channels": channels, "audio_offset_s": 0,
                                     "video_offset_s": 0},
                        "actual": {"codec": info.audio_codec,
                                   "sample_rate": info.sample_rate,
                                   "channels": info.channels,
                                   "audio_offset_s": info.audio_offset_s,
                                   "video_offset_s": info.video_offset_s},
                    },
                    {
                        "id": "picture_contract",
                        "pass": (
                            info.width,
                            info.height,
                            info.fps,
                            info.video_codec,
                            info.pix_fmt,
                            info.color_space,
                            info.color_primaries,
                            info.color_transfer,
                        )
                        == (frame.width, frame.height, FPS, "h264", "yuv420p",
                            "bt709", "bt709", "bt709"),
                        "expected": {"width": frame.width, "height": frame.height, "fps": str(FPS),
                                     "codec": "h264", "pix_fmt": "yuv420p",
                                     "color_space": "bt709", "color_primaries": "bt709",
                                     "color_transfer": "bt709"},
                        "actual": {"width": info.width, "height": info.height,
                                   "fps": str(info.fps), "codec": info.video_codec,
                                   "pix_fmt": info.pix_fmt, "color_space": info.color_space,
                                   "color_primaries": info.color_primaries,
                                   "color_transfer": info.color_transfer},
                    },
                ]
                staged_spans = work / "spans.json"
                staged_words = work / "cut-words.json"
                _write_json(staged_spans, _spans(plan, root / "cut"))
                _write_json(staged_words, _cut_words(plan))
                for name, staged in (
                    ("cut.mp4", rendered),
                    ("voice.wav", voice_path),
                    ("spans.json", staged_spans),
                    ("cut-words.json", staged_words),
                ):
                    run.replace_output(root / "cut" / name, staged)
                run.progress("verifying cut", 0.9)
                asr = context.project.asr
                report = verify(
                    media=str(root / "cut/cut.mp4"),
                    edl=str(root / "cut/spans.json"),
                    asr=asr["engine"],
                    asr_model=asr.get("model"),
                    detail="full",
                    loudness="note",
                    platform=context.project.platform,
                )
                report = _story_platform_report(report)
                report = take_levels.explain_part_jumps(report, take_level_record)
                voiceover_ids = {take.id for take in context.project.takes if take.voiceover}
                if all(piece.take in voiceover_ids for piece in plan.pieces):
                    from yap_studio.stages.voiceover import skip_standin_picture_checks

                    report = skip_standin_picture_checks(report)
                staged_report = work / "verify.json"
                _write_json(staged_report, _portable_report(report, root))
                run.replace_output(root / "cut/verify.json", staged_report)
                verified = str(report["status"]).lower() != "fail" and all(
                    assertion["pass"] for assertion in assertions
                )
                outcome = "verified" if verified else "failed"
                assertions.append({"id": "verify", "pass": str(report["status"]).lower() != "fail",
                                   "expected": "pass", "actual": str(report["status"]).lower()})
                run.finish(
                    outcome,
                    asserts=assertions,
                    extra={
                        "conversions": conversion_rows,
                        "voice": {
                            "codec": "pcm_s16le",
                            "sample_rate": SAMPLE_RATE,
                            "channels": channels,
                        },
                        "plan": plan.to_dict(),
                        "takes": {take_id: context.take_fingerprint(take_id)
                                  for take_id in sorted(plan.sources)},
                        **({"take_levels": take_level_record}
                           if take_level_record is not None else {}),
                    },
                )
                has_duration_warning = any(item["check"] == "platform.duration"
                                           for item in report.get("findings", []))
                warnings = [warning for warning in plan.warnings
                            if not (has_duration_warning
                                    and warning.startswith("Cut plus end hold is "))]
                if take_level_record is not None:
                    warnings.extend(take_levels.warnings(take_level_record))
                warnings.extend(
                    item["message"]
                    for item in report.get("findings", [])
                    if item.get("severity") == "warn"
                )
                return {
                    **plan.to_dict(),
                    "status": outcome,
                    "noop": False,
                    "warnings": warnings,
                    "out": "cut/cut.mp4",
                    "receipt": "cut/receipt.json",
                    "verify": "cut/verify.json",
                    "conversions": conversion_rows,
                    "asserts": assertions,
                    "failures": [item for item in report.get("findings", [])
                                 if item["severity"] == "fail"],
                    **({"take_levels": take_level_record} if take_level_record is not None else {}),
                }
        finally:
            if os.environ.get("YAP_STUDIO_KEEP_WORK") != "1":
                shutil.rmtree(work, ignore_errors=True)
