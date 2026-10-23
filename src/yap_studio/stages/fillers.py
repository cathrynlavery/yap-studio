"""Pure source-time filler rules and pause-aware removal boundary planning."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import tempfile
import time
import wave
from collections import Counter
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from yap_studio import audio, paths, receipt
from yap_studio.checks.joins import click_at, pause_len_at
from yap_studio.errors import InputError, YapError
from yap_studio.plan import snap_frame
from yap_studio.project.lock import ConflictError, _write_json, pid_alive, project_lock, timestamp
from yap_studio.project.model import (
    Anchor,
    ProjectFileError,
    Removal,
    Removals,
    Word,
    Words,
    parse_document,
)
from yap_studio.project.receipts import StageRun, projection_sha256, read_receipt
from yap_studio.project.status import ProjectContext, fillers_inputs, removal_decisions
from yap_studio.transcript import Backend, Transcript, resolve_backend
from yap_studio.transcript.words import OPTIONAL, compare_words, norm_words, words_between

_FPS = Fraction(30)
_SOUNDS = OPTIONAL
_KNOW_BEFORE = {"do", "did", "don't", "didn't", "if", "you", "they", "we", "i"}
_KNOW_AFTER = {
    "what", "that", "how", "why", "who", "where", "when", "if", "the", "him", "her", "them",
    "it",
}
_BE = {"was", "were", "am", "i'm", "is", "are", "we're", "they're", "you're", "he's", "she's"}
_CLAUSE = {
    "i", "i'm", "i've", "i'll", "i'd", "we", "we're", "we'll", "we've", "you", "you're",
    "they", "they're", "he", "she", "it's", "there's", "that's", "oh", "okay", "ok", "well",
    "wow", "yeah", "yes", "no", "what", "this", "hey", "alright", "sure", "wait", "let's",
    "let", "why", "how", "maybe", "please", "um", "uh",
}
_LIKE_BEFORE = {
    "i", "you", "we", "they", "feel", "feels", "felt", "feeling", "look", "looks", "looked",
    "looking", "sound", "sounds", "sounded", "seem", "seems", "seemed", "something",
    "anything", "nothing", "stuff", "things", "more", "less", "much", "don't", "didn't",
    "doesn't", "do", "did", "does", "not",
}
_WOULD = {"would", "i'd", "we'd", "you'd", "they'd", "he'd", "she'd"}
_WOULD_AFTER = {
    "to", "a", "an", "the", "some", "that", "this", "it", "you", "me", "him", "her", "them",
    "us", "one",
}
_OBJECT = {
    "that", "this", "it", "a", "an", "the", "him", "her", "them", "me", "us", "my", "your",
    "our", "their", "his", "you",
}
_SUBJECT = {
    "i", "i've", "i'm", "i'd", "i'll", "we", "we've", "we're", "you", "you're", "they",
    "they're", "he", "she",
}
_SO_AFTER = {"much", "many", "far", "long", "that", "good", "bad", "cool"}
_STUTTER_SKIP = {"very", "really", "no", "yeah", "bye", "ha", "na", "so"}
_CONTRACTIONS = {
    "i": {"i'm", "i'd", "i've", "i'll"},
    "we": {"we're", "we'd", "we've", "we'll"},
    "you": {"you're", "you'd", "you've", "you'll"},
    "they": {"they're", "they'd", "they've", "they'll"},
    "he": {"he's", "he'd", "he'll"},
    "she": {"she's", "she'd", "she'll"},
    "it": {"it's", "it'd", "it'll"},
}
_FODDER = {
    "um", "uh", "uhm", "umm", "mm", "erm", "hmm", "you", "know", "i", "i'd", "i'm", "we",
    "we're", "like", "so", "and", "but", "the", "a", "it", "it's", "they", "they're", "just",
    "basically",
}
_STOP = {"the", "a", "an", "and", "i", "to", "of", "it", "so", "but", "we", "you", "that",
         "is", "was"}
_PRIORITY = {"filler": 0, "stutter": 1, "restart": 2, "silence": -1}
# A restart needs a sustained pause below its own peak after audible left-word speech.
PAUSE_BELOW_PEAK_DB = 30
RESTART_PAUSE_MIN_S = 0.15


@dataclass(frozen=True)
class Candidate:
    """One proposed removal on a source clock, before any join test."""

    take: str
    kind: Literal["filler", "stutter", "restart", "silence", "manual"]
    word_ids: tuple[str, ...]
    start: float
    end: float
    reasons: tuple[str, ...]
    segment: str
    listen: bool
    plan: dict[str, Any]


@dataclass(frozen=True)
class DetectResult:
    """Deterministic proposals and the language scope of the rules."""

    candidates: tuple[Candidate, ...]
    english_rules: bool
    notes: tuple[str, ...]
    restart_rules: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class _Span:
    """An inclusive range of word indexes, prior to boundary planning."""

    first: int
    last: int
    kind: Literal["filler", "stutter", "restart"]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _RefinedSpan:
    """Transient source-clock words for detection and source occurrence alignment."""

    start: float
    end: float
    words: tuple[Word, ...]


def candidate_id(c: Candidate) -> str:
    """Identify a removal by take, millisecond boundaries and kind only."""
    identity = f"{c.take}|{c.start:.3f}|{c.end:.3f}|{c.kind}"
    return "r" + hashlib.sha256(identity.encode()).hexdigest()[:8]


def _norm(text: str) -> str:
    """Lowercase and trim surrounding punctuation, preserving contractions."""
    text = text.lower().replace("\u2019", "'").strip()
    while text and not text[0].isalnum():
        text = text[1:]
    while text and not text[-1].isalnum():
        text = text[:-1]
    return text


def _ending(text: str) -> str:
    """Ignore closing quotes when inspecting sentence punctuation."""
    return text.rstrip().rstrip("\"'\u201d\u2019\u00bb\u203a)]}")


def _sentence_end(word: Word) -> bool:
    """Recognize sentence punctuation before any closing quote."""
    return _ending(word.text).endswith((".", "!", "?"))


def _like_span(words: Sequence[Word], tokens: list[str], i: int) -> _Span | None:
    """Apply the ordered keep rules, then choose the filler phrase extent."""
    previous = tokens[i - 1] if i else ""
    following = tokens[i + 1] if i + 1 < len(tokens) else ""
    if previous != "it's" and _ending(words[i].text).endswith(","):
        return None
    meaningful = i - 1
    while meaningful >= 0 and tokens[meaningful] in _SOUNDS | {"just", "basically"}:
        meaningful -= 1
    if meaningful >= 0 and tokens[meaningful] in _BE and following in _CLAUSE:
        return None
    if following in {"that", "this"} and _sentence_end(words[i + 1]):
        return None
    if previous in _LIKE_BEFORE:
        return None
    if previous in _WOULD and following in _WOULD_AFTER:
        return None
    if previous == "just":
        if following in _OBJECT:
            return None
        first = i - 1
        if i >= 2 and tokens[i - 2] == "it's" and following in _SUBJECT:
            first = i - 2
        return _Span(first, i, "filler", ("filler just like",))
    if i >= 2 and tokens[i - 2:i] in (["kind", "of"], ["sort", "of"]):
        if (i >= 3 and tokens[i - 3] in {"i", "we", "you", "they", "he", "she"}
                or following in _OBJECT):
            return None
        return _Span(i - 2, i, "filler", ("filler kind/sort of like",))
    if previous == "it's" and following in _SUBJECT:
        return _Span(i - 1, i, "filler", ("filler it's like",))
    return _Span(i, i, "filler", ("filler like",))


def _word_spans(words: Sequence[Word], beat_breath_s: float, *,
                refined: Sequence[_RefinedSpan] = ()) -> list[_Span]:
    """Write English detections solely from word text, punctuation and timing."""
    tokens = [_norm(word.text) for word in words]
    spans = []
    for i, token in enumerate(tokens):
        if token in _SOUNDS | {"basically"}:
            spans.append(_Span(i, i, "filler", (f"filler {token}",)))
        if tokens[i:i + 2] == ["you", "know"]:
            previous = tokens[i - 1] if i else ""
            following = tokens[i + 2] if i + 2 < len(tokens) else ""
            if previous not in _KNOW_BEFORE and following not in _KNOW_AFTER:
                spans.append(_Span(i, i + 1, "filler", ("filler you know",)))
        if token == "like":
            span = _like_span(words, tokens, i)
            if span is not None:
                spans.append(span)
        if token == "so":
            first = i - 1 if i and tokens[i - 1] == "and" else i
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            if (first > 0 and _sentence_end(words[first - 1]) and following not in _SO_AFTER
                    and Fraction(str(words[first].start)) - Fraction(str(words[first - 1].end))
                    < Fraction(str(beat_breath_s))):
                spans.append(_Span(first, i, "filler", ("sentence-initial so",)))
        if i + 1 < len(tokens) and token and token not in _STUTTER_SKIP:
            following = tokens[i + 1]
            next_text = words[i + 1].text.strip()
            while next_text and not next_text[0].isalnum():
                next_text = next_text[1:]
            capitalized = bool(next_text and next_text[0].isupper())
            repeat = following == token or following in _CONTRACTIONS.get(token, set())
            if (repeat and not _sentence_end(words[i])
                    and (not capitalized or following.split("'")[0] == "i")):
                spans.append(_Span(i, i, "stutter", ("repeated word or contraction",)))
    restart_removed: set[int] = set()
    for size in (3, 2):
        for first in range(len(tokens) - size + 1):
            if first in restart_removed:
                continue
            gram = tokens[first:first + size]
            if not all(gram) or all(token in _STOP for token in gram):
                continue
            for second in range(first + size, min(first + size + 5, len(tokens) - size + 1)):
                if (tokens[second:second + size] == gram
                        and Fraction(str(words[second].start))
                        - Fraction(str(words[first + size - 1].end)) <= 2
                        and all(token in _FODDER for token in tokens[first + size:second])
                        and not any(_sentence_end(word)
                                    for word in words[first:second + size - 1])):
                    spans.append(_Span(first, second - 1, "restart", (f"repeated {size}-gram",)))
                    restart_removed.update(range(first, second))
                    break
    for size in (3, 2):
        for first in range(len(tokens) - 2 * size):
            second = first + size + 1
            gram = tokens[first:first + size]
            if (not all(gram) or all(token in _STOP for token in gram)
                    or tokens[first + size] in _FODDER
                    or tokens[second:second + size] != gram
                    or any(_sentence_end(word) for word in words[first:second + size - 1])):
                continue
            if any(clip.start <= words[first].start
                   and words[second + size - 1].end <= clip.end for clip in refined):
                spans.append(_Span(first, second - 1, "restart",
                                   (f"hidden repeated {size}-gram",)))
    for first, token in enumerate(tokens):
        if len(token) < 2 or not token.isalpha() or first == 0:
            continue
        for second in range(first + 2, len(tokens)):
            if words[second].start - words[first].start > 2:
                break
            if (not tokens[second].startswith(token) or len(tokens[second]) <= len(token)
                    or tokens[first - 1] != tokens[second - 1]
                    or not tokens[first - 1]
                    or any(_sentence_end(word) for word in words[first - 1:second])):
                continue
            left, right = first - 1, second - 1
            while (left > 0 and right > first + 1
                   and tokens[left - 1] == tokens[right - 1]
                   and not _sentence_end(words[left - 1])):
                left, right = left - 1, right - 1
            spans.append(_Span(left, right - 1, "restart", ("fragment restart",)))
            break
    return spans


def _protected(start: float, end: float, protect: Sequence[tuple[float, float]]) -> bool:
    """Reserve an additional tenth of a second around every proposed span."""
    return any(start - 0.1 < last and end + 0.1 > first for first, last in protect)


def _merge_spans(spans: list[_Span]) -> list[_Span]:
    """Unify overlapping or adjacent removed words, retaining every reason."""
    merged: list[_Span] = []
    for span in sorted(spans, key=lambda item: (item.first, item.last, item.kind)):
        if merged and span.first <= merged[-1].last + 1:
            previous = merged.pop()
            kind = max((previous.kind, span.kind), key=_PRIORITY.__getitem__)
            merged.append(_Span(previous.first, max(previous.last, span.last), kind,
                                tuple(sorted(set(previous.reasons + span.reasons)))))
        else:
            merged.append(span)
    return merged


def _silence_frame(
    want: float, low: float, high: float, silence_db: float,
    level_at: Callable[[float], float] | None,
) -> float | None:
    """Find the nearest silent frame boundary inside a gap, never inside a word."""
    first = math.ceil(low * 30 - 1e-9)
    last = math.floor(high * 30 + 1e-9)
    while first <= last and first / 30 < low:
        first += 1
    while first <= last and last / 30 > high:
        last -= 1
    if first > last:
        return None
    want_time = Fraction(str(want))
    nearest = min(last, max(first, snap_frame(want_time, _FPS)))
    if level_at is None:
        return float(Fraction(nearest, 1) / _FPS)
    frames = sorted(range(first, last + 1), key=lambda frame: (
        abs(Fraction(frame, 1) / _FPS - want_time), frame != nearest, frame,
    ))
    for frame in frames:
        point = float(Fraction(frame, 1) / _FPS)
        if level_at(point) < silence_db:
            return point
    return None


def _snap_time(point: float) -> float:
    """Snap a source boundary to the nearest frame, with half-frame ties upward."""
    return float(Fraction(snap_frame(Fraction(str(point)), _FPS), 1) / _FPS)


def _restart_times(segment: _PlannedAudio, first: Word, right: Word, left: Word | None,
                   low: float, silence_db: float) -> tuple[float, float, float | None]:
    """Locate restart voice onsets on segment-relative ten millisecond PCM frames."""
    frame = _RATE // 100
    count = math.ceil(len(segment.sound) / frame)
    if not count:
        return first.start, right.start, None

    def power(index: int) -> float:
        """Measure one ten millisecond frame, including a short final frame."""
        samples = segment.sound[index * frame:(index + 1) * frame]
        return float(np.mean(samples.astype(np.float64) ** 2)) if len(samples) else 0.0

    def voiced(index: int) -> bool:
        """Use the same floor for onset searches and silent boundary planning."""
        return power(index) >= 10 ** (silence_db / 10)

    def index_at(point: float) -> int:
        """Keep approximate clip clocks inside the decoded PCM."""
        return min(count - 1, max(0, math.floor((point - segment.start) * 100 + 1e-8)))

    start = first.start
    left_end = None
    limit = max(segment.start, low, first.start - 0.4)
    first_frame = max(0, math.ceil((limit - segment.start) * 100 - 1e-8))
    # A late clip clock can land in quiet after the removed word's voice.
    for cursor in range(first_frame, index_at(first.start) + 1):
        if voiced(cursor):
            start = max(limit, min(first.start, segment.start + cursor / 100))
            if (left is not None and cursor == first_frame
                    and low >= first.start - 0.4 - 1e-9):
                back = max(segment.start, first.start - 0.4, left.start)
                earliest = max(0, math.ceil((back - segment.start) * 100 - 1e-8))
                onset = cursor
                while onset > earliest and voiced(onset - 1):
                    onset -= 1
                left_first = max(0, math.ceil((left.start - segment.start) * 100 - 1e-8))
                left_last = min(count, math.ceil((left.end - segment.start) * 100 - 1e-8))
                peak = max(power(index) for index in range(onset, min(count, onset + 20)))
                pause_limit = peak * 10 ** (-PAUSE_BELOW_PEAK_DB / 10)
                pause_first = onset
                while pause_first > left_first and power(pause_first - 1) <= pause_limit:
                    pause_first -= 1
                if (onset - pause_first >= math.ceil(RESTART_PAUSE_MIN_S * 100 - 1e-8)
                        and any(power(index) >= pause_limit
                                for index in range(left_first, min(pause_first, left_last)))):
                    start = segment.start + onset / 100
                    left_end = segment.start + pause_first / 100
            break

    cursor = index_at(right.start)
    end = right.start
    if voiced(cursor):
        while cursor > 0 and voiced(cursor - 1):
            cursor -= 1
        end = min(right.start, segment.start + cursor / 100)
    else:
        # Early clip clocks can land in quiet immediately before the kept copy.
        limit = min(segment.end, right.start + 0.2)
        for following in range(cursor + 1, count):
            point = segment.start + following / 100
            if point > limit + 1e-9:
                break
            if voiced(following):
                end = point
                break
    return round(start, 9), round(end, 9), (round(left_end, 9)
                                              if left_end is not None else None)


def _plan_words(
    words: Sequence[Word], span: _Span, *, beat_breath_s: float, join_target_s: float,
    silence_db: float, level_at: Callable[[float], float] | None,
    take: str | None = None, restart_audio: _PlannedAudio | None = None,
) -> Candidate | None:
    """Keep the longer surrounding pause, preserving a deliberate beat whole."""
    first, last = words[span.first], words[span.last]
    left = words[span.first - 1] if span.first else None
    right = words[span.last + 1] if span.last + 1 < len(words) else None
    low = left.end if left else math.floor(Fraction(str(first.start)) * _FPS) / 30
    high = right.start if right else math.ceil(Fraction(str(last.end)) * _FPS) / 30
    first_start, last_end = first.start, last.end
    snapped: dict[str, float] = {}
    if restart_audio is not None and right is not None:
        low = max(restart_audio.start, left.end if left else restart_audio.start)
        first_start, high, left_end = _restart_times(
            restart_audio, first, right, left, low, silence_db,
        )
        if left_end is not None:
            low = left_end
        # The abandoned pause belongs to the removal; keep the pause before its voice.
        last_end = high
        snapped = {"clip_start_s": first.start, "clip_end_s": right.start,
                   "start_s": first_start, "end_s": high}
        if left_end is not None:
            snapped["left_end_s"] = left_end

        def level_at(point: float) -> float:
            """Keep the retained pause on silent PCM rather than approximate clip words."""
            return audio.rms_db(restart_audio.sound, point - restart_audio.start, _RATE)

    pre = max(0.0, float(Fraction(str(first_start)) - Fraction(str(low)))) if left else 0.0
    post = max(0.0, float(Fraction(str(high)) - Fraction(str(last_end)))) if right else 0.0
    side = "pre" if pre >= post else "post"
    pause = max(pre, post)
    beat = pause >= beat_breath_s
    keep = pause if beat else min(pause, join_target_s)
    want_start = float(Fraction(str(low)) + Fraction(str(keep))) if side == "pre" else low
    want_end = float(Fraction(str(high)) - Fraction(str(keep))) if side == "post" else high
    start = _silence_frame(want_start, low, first_start, silence_db, level_at) if pre else None
    end = _silence_frame(want_end, last_end, high, silence_db, level_at) if post else None
    start_fallback, end_fallback = start is None, end is None
    if start_fallback:
        start = _snap_time(first_start)
    if end_fallback:
        end = _snap_time(last_end)
    if snapped:
        # Floor onsets so video frame rounding cannot retain or clip voiced samples.
        if start_fallback:
            start = max(math.ceil(low * 30 - 1e-9) / 30,
                        math.floor(first_start * 30 + 1e-9) / 30)
        end = min(math.floor(high * 30 + 1e-9) / 30, restart_audio.end)
    elif right is not None and any(reason.startswith(("hidden repeated", "fragment restart"))
                                  for reason in span.reasons):
        start, end = _snap_time(first.start), _snap_time(right.start)
    if start >= end:
        return None
    joined = max(0.0, (start - low if left else 0.0) + (high - end if right else 0.0))
    return Candidate(
        take=take if take is not None else first.id.split(":", 1)[0], kind=span.kind,
        word_ids=tuple(word.id for word in words[span.first:span.last + 1] if word.id),
        start=start, end=end, reasons=span.reasons, segment="",
        listen=(left is not None and start_fallback or right is not None and end_fallback
                or left is not None and right is not None and joined < 1e-9),
        plan={"word_indexes": tuple(range(span.first, span.last + 1)),
              "kept_side": side if pause else "none", "want_pause_s": keep,
              "joined_pause_s": joined, "pre_pause_s": pre, "post_pause_s": post,
              "beat_breath": beat, "silence_db": silence_db,
              "left_word_end_s": left.end if left else None,
              "right_word_start_s": high if right else None,
              **({"restart_audio": snapped} if snapped else {})},
    )


def _pause_candidates(
    words: Sequence[Word], removals: Sequence[Candidate], *, max_pause_s: float,
    silence_db: float, level_at: Callable[[float], float] | None,
    silent_runs: Callable[[float, float], list[tuple[float, float]]] | None = None,
    take: str | None = None,
) -> list[Candidate]:
    """Shorten gaps between kept words, excluding already planned removals."""
    removed = {word_id for candidate in removals for word_id in candidate.word_ids}
    anonymous = {index for candidate in removals
                 for index in candidate.plan.get("word_indexes", ())}
    kept = [word for index, word in enumerate(words)
            if word.id not in removed and (word.id or index not in anonymous)]
    pauses = []
    for left, right in zip(kept, kept[1:], strict=False):
        intervals = []
        cursor = left.end
        for removal in removals:
            if removal.end <= cursor or removal.start >= right.start:
                continue
            if removal.start > cursor:
                intervals.append((cursor, removal.start))
            cursor = max(cursor, removal.end)
        if cursor < right.start:
            intervals.append((cursor, right.start))
        if silent_runs is not None:
            intervals = [
                (max(first, quiet_first), min(last, quiet_last))
                for first, last in intervals
                for quiet_first, quiet_last in silent_runs(first, last)
                if max(first, quiet_first) < min(last, quiet_last)
            ]
        total = sum((Fraction(str(last)) - Fraction(str(first)) for first, last in intervals),
                    start=Fraction(0))
        maximum = Fraction(str(max_pause_s))
        if total <= maximum:
            continue
        offset = Fraction(0)
        for first, last in intervals:
            duration = Fraction(str(last)) - Fraction(str(first))
            want_start = float(Fraction(str(first)) + max(Fraction(0), maximum / 2 - offset))
            want_end = float(Fraction(str(first)) + min(duration, total - maximum / 2 - offset))
            offset += duration
            if want_start >= want_end:
                continue
            start = _silence_frame(want_start, first, last, silence_db, level_at)
            end = _silence_frame(want_end, first, last, silence_db, level_at)
            if silent_runs is not None and (start is None or end is None):
                continue
            fallback = start is None or end is None
            if start is None:
                start = _snap_time(want_start)
            if end is None:
                end = _snap_time(want_end)
            if start >= end:
                continue
            pauses.append(Candidate(
                take=take if take is not None else left.id.split(":", 1)[0],
                kind="silence", word_ids=(),
                start=start, end=end, reasons=("pause exceeds max_pause_s",), segment="",
                listen=fallback or float(total) - (end - start) < 1e-9,
                plan={"word_indexes": (), "kept_side": "centred", "want_pause_s": max_pause_s,
                      "gap_s": float(total), "silence_db": silence_db,
                      "left_word_end_s": left.end, "right_word_start_s": right.start},
            ))
    return pauses


def _merge_candidates(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Combine touching planned spans, including pauses, without losing provenance."""
    merged: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, item.kind)):
        if not merged or candidate.start > merged[-1].end + 1e-9:
            merged.append(candidate)
            continue
        previous = merged.pop()
        primary = max((previous, candidate), key=lambda item: _PRIORITY[item.kind])
        plan = dict(primary.plan)
        plan["word_indexes"] = tuple(sorted(set(previous.plan["word_indexes"])
                                            | set(candidate.plan["word_indexes"])))
        plan["components"] = [*previous.plan.get("components", [previous.plan]), candidate.plan]
        start, end = previous.start, max(previous.end, candidate.end)
        left, right = plan["left_word_end_s"], plan["right_word_start_s"]
        joined = (max(0.0, right - left - (end - start))
                  if left is not None and right is not None else 0.0)
        plan["joined_pause_s"] = joined
        merged.append(replace(
            primary, start=start, end=end,
            word_ids=tuple(dict.fromkeys(previous.word_ids + candidate.word_ids)),
            reasons=tuple(sorted(set(previous.reasons + candidate.reasons))),
            listen=(previous.listen or candidate.listen
                    or left is not None and right is not None and joined < 1e-9), plan=plan,
        ))
    return tuple(merged)


def detect(
    words: Sequence[Word], *, language: str = "en",
    protect: Sequence[tuple[float, float]] = (), max_pause_s: float = 0.3,
    beat_breath_s: float = 0.4, join_target_s: float = 0.2, silence_db: float = -45.0,
    level_at: Callable[[float], float] | None = None,
    silent_runs: Callable[[float, float], list[tuple[float, float]]] | None = None,
    refined: Sequence[_RefinedSpan] = (), take: str | None = None,
    restart_audio: _PlannedAudio | None = None,
) -> DetectResult:
    """Propose removals for ordered words from one segment, without decoding audio.

    The take is inferred from source word IDs; the project wrapper supplies the
    segment ID. Word indexes in each plan refer to the original input sequence.
    Decoded PCM places only restart spans supplied by refined clip words.
    Mixed overlapping detections prefer restart, then stutter, then filler.
    Touching words and gaps without a silent frame use snapped word boundaries
    and flag the proposal for listening. Only empty or reversed word removals
    after frame snapping yield the boundary note.
    """
    for name, value in (("max_pause_s", max_pause_s), ("beat_breath_s", beat_breath_s),
                        ("join_target_s", join_target_s)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(silence_db):
        raise ValueError("silence_db must be finite")
    english = language.lower().replace("_", "-").split("-")[0] == "en"
    notes = [] if english else ["filler rules are English only; pauses only"]
    candidates = []
    restart_rules: Counter[str] = Counter()
    if english:
        spans = [span for span in _word_spans(words, beat_breath_s, refined=refined)
                 if not _protected(words[span.first].start, words[span.last].end, protect)]
        restart_rules.update(reason for span in spans if span.kind == "restart"
                             for reason in span.reasons)
        for span in _merge_spans(spans):
            right = words[span.last + 1] if span.last + 1 < len(words) else None
            audio_restart = (restart_audio if right is not None
                             and any(reason.startswith(("hidden repeated", "fragment restart"))
                                     for reason in span.reasons)
                             and any(words[span.first] in clip.words and right in clip.words
                                     for clip in refined) else None)
            candidate = _plan_words(words, span, beat_breath_s=beat_breath_s,
                                    join_target_s=join_target_s, silence_db=silence_db,
                                    level_at=level_at, take=take, restart_audio=audio_restart)
            if candidate is None:
                notes.append("word removal has no safe silent frame boundary")
            elif not _protected(candidate.start, candidate.end, protect):
                candidates.append(candidate)
    pauses = _pause_candidates(words, candidates, max_pause_s=max_pause_s,
                               silence_db=silence_db, level_at=level_at,
                               silent_runs=silent_runs, take=take)
    candidates.extend(candidate for candidate in pauses
                      if not _protected(candidate.start, candidate.end, protect))
    return DetectResult(_merge_candidates(candidates), english, tuple(dict.fromkeys(notes)),
                        tuple(sorted(restart_rules.items())))


def _anchor_time(anchor: Anchor, words: Words, location: str) -> float:
    """Guard anchors for document-only previews where no audio can settle edges."""
    if anchor.time is not None:
        return anchor.time
    word = next((word for word in words.words if word.id == anchor.word), None)
    if word is None:
        raise ProjectFileError("edit.json", location, f"missing word {anchor.word!r}")
    if not anchor.matches(word.text):
        raise ProjectFileError("edit.json", location, "text guard disagrees with source word")
    return (word.start if anchor.edge == "start" else word.end) + anchor.offset_s


def candidates_for_project(
    project_dir: Path, *, protect: Sequence[tuple[float, float]] = (),
) -> list[Candidate]:
    """Preview document-only edits, using settled ranges when media is present."""
    from yap_studio.stages.cut import _ReadOnlyIdentity

    context = ProjectContext(project_dir)
    context.identity = _ReadOnlyIdentity(context.project_dir)
    edit = parse_document("edit", context.json("edit.json"))
    if all(context.take_path(segment.take).is_file() for segment in edit.segments):
        candidates = _proposals(context, _planned(context), protect=protect)
        return sorted(candidates, key=lambda item: (item.start, item.kind, item.take, item.segment))
    project = context.project
    cache: dict[str, Words] = {}
    candidates = []
    for segment in edit.segments:
        if segment.take not in {take.id for take in project.takes}:
            raise ProjectFileError("edit.json", f"segment {segment.id}.take", "unknown take")
        if segment.take not in cache:
            relative = f"ingest/{segment.take}/words.json"
            try:
                payload = (project_dir / relative).read_bytes()
                source: Words = parse_document("words", json.loads(payload), filename=relative)
            except (OSError, ValueError, UnicodeError) as exc:
                raise ProjectFileError(relative, "$", "cannot read a valid words document") from exc
            if hashlib.sha256(payload).hexdigest() != edit.words_sha256.get(segment.take):
                raise ProjectFileError("edit.json", "words_sha256", "source words have changed")
            if source.take != segment.take:
                raise ProjectFileError(relative, "take", "take disagrees with segment")
            cache[segment.take] = source
        source = cache[segment.take]
        first = _anchor_time(segment.from_, source, f"segment {segment.id}.from")
        last = _anchor_time(segment.to, source, f"segment {segment.id}.to")
        if first < 0 or last <= first:
            raise ProjectFileError("edit.json", f"segment {segment.id}", "invalid source range")
        selected = tuple(word for word in source.words
                         if first <= (word.start + word.end) / 2 < last)
        result = detect(selected, language=project.language, protect=protect,
                        max_pause_s=float(project.rules["max_pause_s"]),
                        beat_breath_s=float(project.rules["beat_breath_s"]))
        frame_first = snap_frame(Fraction(str(first)), _FPS) / 30
        frame_last = snap_frame(Fraction(str(last)), _FPS) / 30
        candidates.extend(replace(candidate, segment=segment.id) for candidate in result.candidates
                          if candidate.start >= frame_first and candidate.end <= frame_last)
    return sorted(candidates, key=lambda item: (item.start, item.kind, item.take, item.segment))


_RATE = 48_000
_NUDGES = (
    ((0, 0),),
    ((-1, 0), (1, 0), (0, -1), (0, 1)),
    ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)),
)


@dataclass(frozen=True)
class _PlannedAudio:
    """A resolved authored segment's PCM and source-clock words."""

    take: str
    start: float
    end: float
    sound: np.ndarray
    words: tuple[Word, ...]
    existing: tuple[Candidate, ...] = ()
    silence_db: float = -45.0
    pause: dict[str, Any] | None = None


@dataclass(frozen=True)
class _Trial:
    """One spliced trial and its uncut control, both with silent leads."""

    candidates: tuple[Candidate, ...]
    nudge: tuple[int, int]
    start: float
    end: float
    cut: np.ndarray
    control: np.ndarray
    joins: tuple[int, ...]
    stored: tuple[Word, ...]
    removed: frozenset[str]
    listen: bool
    quiet: bool
    edge: Literal["start", "end"] | None = None
    edge_removed: tuple[Word, ...] = ()


@dataclass(frozen=True)
class _Tested:
    """A chosen boundary pair and the complete search evidence."""

    candidate: Candidate
    passed: bool
    listen: bool
    evidence: dict[str, Any]


@dataclass(frozen=True)
class _JoinRequest:
    """Source removals to test in every affected authored segment."""

    candidates: tuple[Candidate, ...]
    nudge: tuple[int, int] = (0, 0)
    edge_aware: bool = False


def _write_wav(path: Path, sound: np.ndarray) -> None:
    """Write ephemeral mono PCM for one batched speech round."""
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(_RATE)
        stream.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())


def _decode_segment(context: ProjectContext, take_id: str, start: float,
                    end: float, offset: float) -> np.ndarray:
    """Decode selected mono PCM while retaining the original container clock."""
    from yap_studio.stages.cut import _selected_stream

    return audio.decode(
        context.take_path(take_id), sr=_RATE, mono=True, start_s=start + offset,
        dur_s=end - start, stream=_selected_stream(context, take_id),
    )[1]


def _planned(context: ProjectContext, decisions: Sequence[Removal] = (),
             ) -> dict[str, _PlannedAudio]:
    """Use cut's settled frame ranges before trims and short wordless-piece merges."""
    from yap_studio.stages import edges
    from yap_studio.stages.cut import _resolve, _selected_stream

    base = copy(context)
    base._snapshots = dict(context._snapshots)
    base._snapshots["removals.json"] = b'{"schema":"yap-studio/removals@1","removals":[]}'
    plan = _resolve(base)
    bounds = {identifier: (float(first), float(last))
              for identifier, first, last in plan.segment_ranges}
    segments = {}
    edit = parse_document("edit", context.json("edit.json"))
    decoded: dict[tuple[str, float, float], np.ndarray] = {}
    pauses = dict(plan.pauses)
    pause_reports = {take: pause.to_dict() for take, pause in pauses.items()}
    for authored in edit.segments:
        source = plan.source_words[authored.take]
        if authored.take not in pauses:
            take = edges.load_take(
                context.take_path(authored.take), _selected_stream(context, authored.take),
                words=[(word.start, word.end) for word in source.words],
            )
            assert take.pause is not None
            pauses[authored.take] = take.pause
            pause_reports[authored.take] = take.pause.to_dict()
        pause = pauses[authored.take]
        first, last = bounds[authored.id]
        words = tuple(word for word in source.words
                      if first <= (word.start + word.end) / 2 < last)
        existing = tuple(Candidate(
            item.take, cast(Any, item.kind),
            (() if item.kind == "silence" else tuple(
                word.id for word in words
                if item.start <= (word.start + word.end) / 2 < item.end)),
            item.start, item.end, (), authored.id, item.listen, {"removal_id": item.id},
        ) for item in decisions if item.take == authored.take and item.status == "applied"
                         and item.start < last and item.end > first)
        key = (authored.take, first, last)
        if key not in decoded:
            decoded[key] = (_decode_segment(context, authored.take, first, last,
                                           plan.media[authored.take].video_offset_s)
                            if last > first else np.empty(0, dtype=np.float32))
        segments[authored.id] = _PlannedAudio(
            authored.take, first, last, decoded[key], words, existing,
            silence_db=pause.threshold_db, pause=pause_reports[authored.take],
        )
    return segments


def _suspicious_spots(segment: _PlannedAudio) -> list[tuple[int, int]]:
    """Flag hidden speech above the take's pause threshold on 10 ms frames."""
    frame = _RATE // 100
    count = len(segment.sound) // frame
    if not count:
        return []
    samples = segment.sound[:count * frame].reshape(count, frame)
    power = np.einsum("ij,ij->i", samples, samples, dtype=np.float64) / frame
    voiced = power >= 10 ** (segment.silence_db / 10)

    def frames(first: float, last: float) -> np.ndarray:
        """Use complete frames so audio outside the word cannot make it suspicious."""
        low = max(0, math.ceil((first - segment.start) * 100 - 1e-8))
        high = min(count, math.floor((last - segment.start) * 100 + 1e-8))
        return voiced[low:max(low, high)]

    spots = []
    for index, word in enumerate(segment.words):
        inside = frames(word.start, word.end)
        hot = np.flatnonzero(inside)
        if len(hot) > 1 and np.any(np.diff(hot) - 1 >= 15):
            spots.append((index, index))
        if index + 1 < len(segment.words):
            following = segment.words[index + 1]
            if (following.start - word.end >= 0.2 - 1e-8
                    and np.count_nonzero(frames(word.end, following.start)) >= 8):
                spots.append((index, index + 1))
    return spots


def _measured_silent_runs(
    segment: _PlannedAudio,
) -> Callable[[float, float], list[tuple[float, float]]]:
    """Return sustained quiet 10 ms frame runs on the source clock."""
    from yap_studio.stages import edges

    frame = _RATE // 100
    count = len(segment.sound) // frame
    samples = segment.sound[:count * frame].reshape(count, frame)
    power = np.einsum("ij,ij->i", samples, samples, dtype=np.float64) / frame
    threshold = (edges.pause_quiet_threshold_db(segment.pause["floor_db"])
                 if segment.pause is not None else segment.silence_db)
    quiet = power < 10 ** (threshold / 10)
    boundaries = np.flatnonzero(np.diff(quiet, prepend=False, append=False))
    minimum = round(edges.SILENCE_MIN_S * 100)
    runs = [(segment.start + first / 100, segment.start + last / 100)
            for first, last in zip(boundaries[::2], boundaries[1::2], strict=True)
            if last - first >= minimum]

    def inside(first: float, last: float) -> list[tuple[float, float]]:
        return [(max(first, low), min(last, high)) for low, high in runs
                if min(last, high) - max(first, low) >= edges.SILENCE_MIN_S - 1e-9]

    return inside


def _refinement_ranges(segment: _PlannedAudio,
                       spots: Sequence[tuple[int, int]]) -> list[tuple[float, float]]:
    """Keep neighbouring phrase context, widen to 1.2 s, then merge overlaps."""
    if segment.end - segment.start < 1.2:
        return []
    ranges: list[tuple[float, float]] = []
    for first, last in spots:
        # A long normalized word can hide the repeated prefix too.
        before = max(0, first - (2 if first == last else 1))
        after = min(len(segment.words) - 1, last + 1)
        start = max(segment.start, segment.words[before].start)
        end = min(segment.end, segment.words[after].end)
        padding = max(0.0, (1.2 - (end - start)) / 2)
        start, end = max(segment.start, start - padding), min(segment.end, end + padding)
        start = max(segment.start, min(start, end - 1.2))
        end = min(segment.end, max(end, start + 1.2))
        # Replacement must cover complete source words at the clip's edges.
        overlapping = [word for word in segment.words if word.start < end and word.end > start]
        if overlapping:
            start = max(segment.start, min(start, overlapping[0].start))
            end = min(segment.end, max(end, overlapping[-1].end))
        ranges.append((start, end))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _refine(segments: dict[str, _PlannedAudio], backend: Backend | None,
            directory: Path, language: str,
            ) -> tuple[dict[str, tuple[_RefinedSpan, ...]], dict[str, Any]]:
    """Transcribe all suspicious project clips in one batch without publishing words."""
    summary: dict[str, Any] = {"spots": 0, "suspicious_spots": 0, "clips": 0,
                               "status": "skipped" if backend is None else "complete", "notes": []}
    if backend is None:
        summary["notes"].append("restart refinement skipped: no speech engine")
        return {}, summary
    requests: list[tuple[str, float, float, Path]] = []
    targets: dict[tuple[str, float, float], Path] = {}
    for identifier, segment in segments.items():
        spots = _suspicious_spots(segment)
        summary["suspicious_spots"] += len(spots)
        ranges = _refinement_ranges(segment, spots)
        if ranges:
            summary["spots"] += len(spots)
        if spots and not ranges:
            summary["notes"].append("restart refinement skipped: segment shorter than 1.2 s")
        for start, end in ranges:
            key = (segment.take, start, end)
            if key not in targets:
                target = directory / f"refine-{len(targets)}.wav"
                low = round((start - segment.start) * _RATE)
                high = round((end - segment.start) * _RATE)
                _write_wav(target, segment.sound[low:high])
                targets[key] = target
            requests.append((identifier, start, end, targets[key]))
    summary["clips"] = len(targets)
    transcripts = (backend.transcribe_many(list(targets.values()), language=language)
                   if targets else {})
    if any(target not in transcripts for target in targets.values()):
        raise InputError("speech batch", "missing refinement transcript")
    refined: dict[str, list[_RefinedSpan]] = {}
    for identifier, start, end, target in requests:
        words = tuple(Word("", word.text, round(start + max(0.0, word.start), 9),
                           round(start + min(end - start, word.end), 9), word.conf)
                      for word in transcripts[target].words
                      if math.isfinite(word.start) and math.isfinite(word.end)
                      and 0 <= (word.start + word.end) / 2 < end - start
                      and min(end - start, word.end) > max(0.0, word.start))
        if words:
            refined.setdefault(identifier, []).append(_RefinedSpan(start, end, words))
        else:
            summary["notes"].append("restart refinement kept source words: empty clip transcript")
    summary["notes"] = list(dict.fromkeys(summary["notes"]))
    return {identifier: tuple(clips) for identifier, clips in refined.items()}, summary


def _detection_words(words: Sequence[Word], refined: Sequence[_RefinedSpan]) -> tuple[Word, ...]:
    """Substitute words inside refined spans on an ephemeral source clock."""
    kept = [word for word in words if not any(clip.start <= (word.start + word.end) / 2 < clip.end
                                            for clip in refined)]
    return tuple(sorted([*kept, *(word for clip in refined for word in clip.words)],
                        key=lambda word: (word.start, word.end)))


def _align_refinement(words: Sequence[Word], clip: _RefinedSpan) -> tuple[_RefinedSpan, bool]:
    """Align occurrences in source order, preferring the closest clock for repeats."""
    source = [word for word in words if clip.start <= (word.start + word.end) / 2 < clip.end]
    local = clip.words
    scores = [[(0, 0.0) for _ in range(len(local) + 1)] for _ in range(len(source) + 1)]
    moves = [["" for _ in range(len(local) + 1)] for _ in range(len(source) + 1)]

    def distance(first: Word, second: Word) -> float:
        return abs(first.start - second.start) + abs(first.end - second.end)

    for i, word in enumerate(source, 1):
        for j, heard in enumerate(local, 1):
            options = [(scores[i - 1][j], "source"), (scores[i][j - 1], "local")]
            if _norm(word.text) == _norm(heard.text):
                count, error = scores[i - 1][j - 1]
                options.append(((count + 1, error - distance(word, heard)), "match"))
            scores[i][j], moves[i][j] = max(options, key=lambda option: option[0])
    pairs = []
    i, j = len(source), len(local)
    while i and j:
        move = moves[i][j]
        if move == "match":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "source":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    inferred = False
    # A single replacement between aligned neighbours has an identity, but needs listening.
    anchors = [(-1, -1), *pairs, (len(source), len(local))]
    for (a, b), (c, d) in zip(anchors, anchors[1:], strict=False):
        if c - a == d - b == 2:
            pairs.append((a + 1, b + 1))
            inferred = True
    assigned = {local_index: source[source_index] for source_index, local_index in pairs}
    uncertain = inferred or len(assigned) != len(source) or len(assigned) != len(local)
    for local_index, word in assigned.items():
        if any(index != local_index and _norm(heard.text) == _norm(word.text)
               and abs(distance(word, heard) - distance(word, local[local_index])) < 1e-9
               for index, heard in enumerate(local)):
            uncertain = True
    aligned = tuple(replace(word, id=assigned[index].id) if index in assigned else word
                    for index, word in enumerate(local))
    return replace(clip, words=aligned), uncertain


def _restart_left_word(candidate: Candidate, words: Sequence[Word]) -> Word | None:
    """Find the stretched kept word immediately before a measured restart pause."""
    snapped = candidate.plan.get("restart_audio", {})
    left_end = snapped.get("left_end_s")
    if left_end is None:
        return None
    return max((word for word in words
                if word.start <= left_end < word.end and word.end > candidate.start),
               key=lambda word: word.start, default=None)


def _cut_midpoint(word: Word, left: Word | None, candidate: Candidate) -> float:
    """Use the measured end only for the kept word before this restart."""
    end = (candidate.plan["restart_audio"]["left_end_s"]
           if word is left else word.end)
    return (word.start + end) / 2


def _proposals(context: ProjectContext, segments: dict[str, _PlannedAudio], *,
               protect: Sequence[tuple[float, float]] = (),
               refinements: dict[str, tuple[_RefinedSpan, ...]] | None = None,
               restart_rules: Counter[str] | None = None,
               language: str | None = None) -> list[Candidate]:
    """Detect on transient refined words while preserving source word provenance."""
    candidates = []
    for identifier, segment in segments.items():
        aligned = [_align_refinement(segment.words, clip)
                   for clip in (refinements or {}).get(identifier, ())]
        refined = tuple(clip for clip, _ in aligned)
        words = _detection_words(segment.words, refined)
        silent_runs = _measured_silent_runs(segment)
        result = detect(words, language=language or context.project.language, protect=protect,
                        max_pause_s=float(context.project.rules["max_pause_s"]),
                        beat_breath_s=float(context.project.rules["beat_breath_s"]),
                        refined=refined, take=segment.take, restart_audio=segment,
                        silent_runs=silent_runs)
        boundary_pauses = []
        if words and words[-1].end < segment.end:
            boundary = replace(words[-1], id="", text="", start=segment.end,
                               end=segment.end)
            boundary_pauses = _pause_candidates(
                (words[-1], boundary), segment.existing,
                max_pause_s=float(context.project.rules["max_pause_s"]),
                silence_db=segment.silence_db, level_at=None,
                silent_runs=silent_runs, take=segment.take,
            )
            boundary_pauses = [item for item in boundary_pauses
                               if not _protected(item.start, item.end, protect)]
        if restart_rules is not None:
            restart_rules.update(dict(result.restart_rules))
        for item in (*result.candidates, *boundary_pauses):
            if not segment.start <= item.start < item.end <= segment.end:
                continue
            clips = [clip for clip in refined if clip.start < item.end and clip.end > item.start]
            plan = dict(item.plan)
            ids = item.word_ids
            left_word = _restart_left_word(item, segment.words)
            if left_word is not None:
                ids = tuple(word_id for word_id in ids if word_id != left_word.id)
            listen = item.listen
            if clips:
                assigned = {word.id for clip in clips for word in clip.words if word.id}
                uncertain = any(flag for clip, flag in aligned if clip in clips)
                listen |= uncertain
                plan["refined_provenance"] = "inferred" if uncertain else "aligned"
                plan["refined_source_words"] = [
                    {"id": word.id, "text": word.text, "start": word.start, "end": word.end}
                    for clip in clips for word in clip.words if word.id
                ]
                plan["refined_spans"] = [{"start": clip.start, "end": clip.end} for clip in clips]
                plan["refined_removal"] = {"start": item.start, "end": item.end}
                if "restart_audio" in plan:
                    # Occurrence labels must follow the clip words even when their clock drifts.
                    snapped = plan["restart_audio"]
                    plan["refined_removal"] = {"start": snapped["clip_start_s"],
                                               "end": snapped["clip_end_s"]}
                    for word in plan["refined_source_words"]:
                        if word["start"] == snapped["clip_end_s"]:
                            word["start"] = snapped["end_s"]
                            word["end"] = max(word["end"], word["start"])
                # Stored source IDs follow the actual audio cut after boundary snapping.
                if item.kind != "silence":
                    ids = tuple(dict.fromkeys((*ids, *(word.id for word in segment.words
                        if word.id not in assigned
                        and item.start <= _cut_midpoint(word, left_word, item) < item.end
                        and any(clip.start <= _cut_midpoint(word, left_word, item) < clip.end
                                for clip in clips)))))
                plan["refined_words"] = [{"text": word.text, "start": word.start, "end": word.end}
                                         for clip in clips for word in clip.words]
            if left_word is not None:
                # The left word is kept audio; cut must use its measured end for ownership.
                measured = {"id": left_word.id, "text": left_word.text,
                            "start": left_word.start,
                            "end": plan["restart_audio"]["left_end_s"]}
                source_words = plan.setdefault("refined_source_words", [])
                for index, word in enumerate(source_words):
                    if word["id"] == left_word.id:
                        source_words[index] = measured
                        break
                else:
                    source_words.append(measured)
            candidates.append(replace(item, segment=identifier, word_ids=ids,
                                      listen=listen, plan=plan))
    unique = {candidate_id(item): item for item in candidates}
    return sorted(unique.values(), key=lambda item: (
        item.take, item.segment, item.start, item.kind,
    ))


def _join_inputs(segment: _PlannedAudio, candidates: tuple[Candidate, ...],
                 ) -> tuple[_PlannedAudio, tuple[Candidate, ...]]:
    """Give the existing join test local occurrence labels from candidate evidence."""
    clips: dict[tuple[float, float], _RefinedSpan] = {}
    for candidate in candidates:
        for span in candidate.plan.get("refined_spans", []):
            first, last = span["start"], span["end"]
            words = tuple(Word("", word["text"], word["start"], word["end"])
                          for word in candidate.plan["refined_words"]
                          if first <= (word["start"] + word["end"]) / 2 < last)
            clips[first, last] = _RefinedSpan(first, last, words)
    if not clips:
        return segment, candidates
    # Comparison labels distinguish repeated occurrences and never become source IDs.
    detection = _detection_words(segment.words, tuple(clips.values()))
    onsets = {item.plan["restart_audio"]["clip_end_s"]: item.plan["restart_audio"]["end_s"]
              for item in candidates if "restart_audio" in item.plan}
    # An early clip midpoint must not make the kept copy look removed by the audio cut.
    words = tuple(replace(word, id=f"@join:{index}",
                          start=onsets.get(word.start, word.start),
                          end=max(word.end, onsets.get(word.start, word.start)))
                  if not word.id else word for index, word in enumerate(detection))

    def labelled(candidate: Candidate) -> Candidate:
        """Keep removed occurrences fixed while the join search nudges boundaries."""
        bounds = candidate.plan.get("refined_removal", {"start": candidate.start,
                                                        "end": candidate.end})
        local = tuple(word.id for word in words if word.id.startswith("@join:")
                      and bounds["start"] <= (word.start + word.end) / 2 < bounds["end"])
        return replace(candidate, word_ids=candidate.word_ids + local)

    existing = tuple(labelled(item) for item in segment.existing)
    return (replace(segment, words=words, existing=existing),
            tuple(labelled(item) for item in candidates))


def _trial(segment: _PlannedAudio, candidates: tuple[Candidate, ...],
           nudge: tuple[int, int] = (0, 0), *, edge_aware: bool = False) -> _Trial | None:
    """Splice a source window with five millisecond fades and one second leads."""
    candidates = tuple(sorted(candidates, key=lambda item: item.start))
    if any(not segment.start <= item.start < item.end <= segment.end for item in candidates):
        return None
    if any(left.end >= right.start for left, right in
           zip(candidates, candidates[1:], strict=False)):
        return None
    if any(not item.plan.get("requested_cut")
           and item.start < saved.end and item.end > saved.start
           for item in candidates for saved in segment.existing):
        return None
    first = max(segment.start, candidates[0].start - 1)
    last = min(segment.end, candidates[-1].end + 1)
    edge = None
    edge_removed = ()
    if edge_aware and len(candidates) == 1:
        candidate = candidates[0]
        if candidate.end == segment.end:
            edge, last = "end", candidate.start
        elif candidate.start == segment.start:
            edge, first = "start", candidate.end
        if edge is not None:
            edge_removed = tuple(word for word in segment.words if word.id in candidate.word_ids)
    low = round((first - segment.start) * _RATE)
    high = round((last - segment.start) * _RATE)
    window = segment.sound[low:high]
    baseline = [item for item in segment.existing if item.start < last and item.end > first]

    def spliced(spans: Sequence[Candidate]) -> tuple[np.ndarray, list[int]]:
        """Subtract source-clock spans while retaining fades on every planned join."""
        ranges, cursor = [], 0
        for item in sorted(spans, key=lambda item: item.start):
            start = max(0, round((item.start - first) * _RATE))
            end = min(len(window), round((item.end - first) * _RATE))
            if start > cursor:
                ranges.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(window):
            ranges.append((cursor, len(window)))
        return audio.splice(window, ranges, 240)

    listen = any(item.listen for item in candidates)
    quiet = True
    for item in candidates:
        sound, origin = (segment.sound, segment.start) if edge is not None else (window, first)
        start, end = round((item.start - origin) * _RATE), round((item.end - origin) * _RATE)
        left = audio.rms_db(sound, (start - 480) / _RATE, _RATE, 0.01)
        right = audio.rms_db(sound, (end + 480) / _RATE, _RATE, 0.01)
        quiet &= audio.rms_db(sound, (start + end) / (2 * _RATE), _RATE,
                              (end - start) / (2 * _RATE)) <= segment.silence_db
        listen |= edge is None and left > segment.silence_db and right > segment.silence_db and (
            item.plan.get("joined_pause_s", 0) <= 1e-9
        )
    control, _ = spliced(baseline)
    cut, joins = spliced([*baseline, *candidates])
    baseline_words = {word_id for item in baseline for word_id in item.word_ids}
    lead = np.zeros(_RATE, dtype=np.float32)
    return _Trial(candidates, nudge, first, last, np.concatenate((lead, cut, lead)),
                  np.concatenate((lead, control, lead)), tuple(index + _RATE for index in joins),
                  tuple(word for word in segment.words
                        if first <= (word.start + word.end) / 2 < last
                        and word.id not in baseline_words),
                  frozenset(word_id for item in candidates for word_id in item.word_ids),
                  listen, quiet, edge, edge_removed)


def _tokens(transcript: Transcript, duration: float) -> list[str]:
    """Exclude the digital leads using the transcript's own clip clock."""
    return norm_words(" ".join(word.text for word in words_between(transcript, 1, duration - 1)))


def _evaluate(trial: _Trial, cut: Transcript | None, control: Transcript | None) -> dict[str, Any]:
    """Require word preservation, removal absence, and a click-free splice."""
    clicks = [click_at(trial.cut, _RATE, index / _RATE) for index in trial.joins]
    click = max(clicks, default=-120.0)
    removed_words = tuple(word for word in trial.stored if word.id in trial.removed)
    removed_words += trial.edge_removed
    result: dict[str, Any] = {
        "nudge": list(trial.nudge), "start": trial.candidates[0].start,
        "end": trial.candidates[-1].end, "click_db": click, "clicks_db": clicks,
        "expected": [], "got": [], "diff": [], "pass": False,
        "listen": trial.listen, "words": "skipped",
        "removed_words": [word.text for word in removed_words],
    }
    if trial.edge is not None:
        result["edge"] = trial.edge
    if cut is None or control is None:
        pauses = [pause_len_at(trial.cut, _RATE, index / _RATE) for index in trial.joins]
        result.update(pause_s=pauses, **{"pass": click <= 0 and trial.quiet and all(
            pause <= max(float(item.plan.get("want_pause_s", 0.3))
                         for item in trial.candidates) + 2 / 30 for pause in pauses
        )})
        if not result["pass"]:
            result["diff"] = ["pause or click check failed"]
        return result
    got = _tokens(cut, len(trial.cut) / _RATE)
    heard = _tokens(control, len(trial.control) / _RATE)
    stored = norm_words(" ".join(word.text for word in trial.stored))
    expected_stored = norm_words(" ".join(word.text for word in trial.stored
                                         if word.id not in trial.removed))
    # Align control tokens to stored tokens; delete only occurrences belonging to
    # removed word IDs, so a kept repeated word is never mistaken for a leftover.
    removed_indexes: set[int] = set()
    cursor = 0
    for word in trial.stored:
        tokens = norm_words(word.text)
        if word.id in trial.removed:
            removed_indexes.update(range(cursor, cursor + len(tokens)))
        cursor += len(tokens)
    erase: set[int] = set()
    for block in difflib.SequenceMatcher(None, stored, heard).get_matching_blocks():
        erase.update(block.b + index for index in range(block.size)
                     if block.a + index in removed_indexes)
    expected = [token for index, token in enumerate(heard) if index not in erase]
    comparison = compare_words(expected, got, fillers_optional=True)
    control_ok = bool(compare_words(stored, heard, fillers_optional=True)["core_exact"])
    def interior(word: Word) -> bool:
        """Allow clipping only at the outer source window of a retained-side trial."""
        if trial.edge == "end":
            return trial.start + 0.03 <= word.start
        if trial.edge == "start":
            return word.end <= trial.end - 0.03
        return trial.start + 0.03 <= word.start < word.end <= trial.end - 0.03

    # Preserve interior words and removal neighbours even at the last window token.
    neighbours = {word.id for word in trial.stored if word.id not in trial.removed and (
        interior(word)
        or any(abs(word.end - item.start) <= 2 / 30
               or word.start - 2 / 30 <= item.end <= word.end
               for item in trial.candidates)
    )}
    required = norm_words(" ".join(word.text for word in trial.stored if word.id in neighbours))
    presence = all(got.count(token) >= required.count(token) for token in required
                   if token not in OPTIONAL)
    control_ok &= all(heard.count(token) >= required.count(token) for token in required
                      if token not in OPTIONAL)
    absent = all(got.count(token) <= expected_stored.count(token)
                 for token in {token for word in removed_words for token in norm_words(word.text)})
    passed = bool(comparison["core_exact"]) and presence and absent and click <= 0
    unverifiable = not control_ok
    # A weak control is retained as a fallback only if it cannot establish a
    # contradictory cut transcript and the physical join is safe.
    fallback = unverifiable and click <= 0 and absent and (trial.edge is None or presence) and (
        bool(comparison["core_exact"])
        or bool(compare_words(expected_stored, got, fillers_optional=True)["core_exact"])
    )
    result.update(expected=expected, got=got, control=heard, stored=stored,
                  diff=comparison["diff"], words="unverifiable" if unverifiable else
                  ("pass" if passed else "fail"), unverifiable=fallback,
                  **{"pass": passed and control_ok})
    if not presence:
        result["diff"].append(f"missing join words: {required}")
    if not absent:
        result["diff"].append("removed words remain at the join")
    if click > 0:
        result["diff"].append("click exceeds 0 dB")
    return result


def _round(trials: list[_Trial], backend: Backend | None, directory: Path,
           language: str, round_id: int) -> list[dict[str, Any]]:
    """Transcribe every cut and control of a search round in one model batch."""
    wavs: list[Path] = []
    if backend is not None:
        for index, trial in enumerate(trials):
            for role, sound in (("cut", trial.cut), ("control", trial.control)):
                target = directory / f"round-{round_id}-{index}-{role}.wav"
                _write_wav(target, sound)
                wavs.append(target)
        transcripts = backend.transcribe_many(wavs, language=language) if wavs else {}
        if any(target not in transcripts for target in wavs):
            raise InputError("speech batch", "missing cut or control transcript")
        return [_evaluate(trial, transcripts[wavs[2 * index]], transcripts[wavs[2 * index + 1]])
                for index, trial in enumerate(trials)]
    return [_evaluate(trial, None, None) for trial in trials]


def _affected_segments(candidates: tuple[Candidate, ...],
                       segments: dict[str, _PlannedAudio]) -> list[str]:
    """Find every authored occurrence whose retained source intersects a removal."""
    affected = []
    for identifier, segment in segments.items():
        for item in candidates:
            if item.take != segment.take:
                continue
            first, last = max(item.start, segment.start), min(item.end, segment.end)
            cursor = first
            for saved in sorted(segment.existing, key=lambda entry: entry.start):
                if saved.end <= cursor:
                    continue
                if saved.start > cursor:
                    break
                cursor = max(cursor, saved.end)
            if cursor < last:
                affected.append(identifier)
                break
    return affected


def _source_round(requests: list[_JoinRequest], segments: dict[str, _PlannedAudio],
                  backend: Backend | None, directory: Path, language: str,
                  round_id: int) -> list[dict[str, Any]]:
    """Batch every occurrence and require shared boundaries to pass in all contexts."""
    trials: list[_Trial] = []
    owners: list[tuple[int, str]] = []
    results: list[list[dict[str, Any]]] = [[] for _ in requests]
    for index, request in enumerate(requests):
        inspections = {item.plan["manual_id"] for item in request.candidates
                       if "manual_id" in item.plan}
        contexts = ({identifier: replace(segment, existing=tuple(
            saved for saved in segment.existing if saved.plan.get("removal_id") not in inspections
        )) for identifier, segment in segments.items()} if inspections else segments)
        if any(not contexts[item.segment].start <= item.start < item.end
               <= contexts[item.segment].end for item in request.candidates):
            results[index].append({"segment": request.candidates[0].segment,
                                   "nudge": list(request.nudge), "pass": False,
                                   "diff": ["outside segment or overlaps saved removal"]})
            continue
        for identifier in _affected_segments(request.candidates, contexts):
            segment = contexts[identifier]
            clipped = tuple(replace(item, segment=identifier,
                                    start=max(item.start, segment.start),
                                    end=min(item.end, segment.end)) for item in request.candidates
                            if item.take == segment.take and item.start < segment.end
                            and item.end > segment.start)
            segment, clipped = _join_inputs(segment, clipped)
            removed = {word_id for item in clipped for word_id in item.word_ids}
            if all(item.plan.get("requested_cut") for item in clipped):
                removed.update(word_id for saved in segment.existing for word_id in saved.word_ids)
            unexpected = [word.text for word in segment.words if word.id not in removed
                          and any(item.start <= _cut_midpoint(
                              word, _restart_left_word(item, segment.words), item,
                          ) < item.end
                                  for item in clipped)]
            if unexpected:
                results[index].append({"segment": identifier, "nudge": list(request.nudge),
                                       "pass": False, "words": "fail", "expected": unexpected,
                                       "got": [], "diff": [
                                           f"removal clips kept words: {unexpected}",
                                       ]})
                continue
            trial = _trial(segment, clipped, request.nudge, edge_aware=request.edge_aware)
            if trial is None:
                results[index].append({"segment": identifier, "nudge": list(request.nudge),
                                       "pass": False, "diff": ["overlapping cuts"]})
            else:
                trials.append(trial)
                owners.append((index, identifier))
    for (index, identifier), result in zip(
        owners, _round(trials, backend, directory, language, round_id), strict=True,
    ):
        results[index].append({"segment": identifier, **result})
    combined = []
    for request, occurrences in zip(requests, results, strict=True):
        if not occurrences:
            occurrences = [{"segment": request.candidates[0].segment, "pass": False,
                            "diff": ["removal has no retained source"]}]
        failed = [result for result in occurrences
                  if not result["pass"] and not result.get("unverifiable")]
        weak = [result for result in occurrences if not result["pass"]]
        representative = (failed or weak or sorted(
            occurrences, key=lambda result: -result.get("click_db", -120.0),
        ))[0]
        evidence = {key: value for key, value in representative.items() if key != "segment"}
        evidence.update(
            nudge=list(request.nudge), start=request.candidates[0].start,
            end=request.candidates[-1].end, segments=occurrences,
            click_db=max(result.get("click_db", -120.0) for result in occurrences),
            listen=any(result.get("listen") for result in occurrences) or bool(weak and not failed),
            unverifiable=bool(weak and not failed),
            **{"pass": not weak},
        )
        if weak and not failed:
            evidence["words"] = "unverifiable"
        if failed and len(occurrences) > 1:
            evidence["diff"] = [f"segment {result['segment']}: {difference}" for result in failed
                                for difference in result.get("diff", [])]
        combined.append(evidence)
    return combined


def _test(candidates: list[Candidate], segments: dict[str, _PlannedAudio],
          backend: Backend | None, directory: Path, language: str) -> list[_Tested]:
    """Search zero, one and two frame boundaries, retaining every attempt."""
    attempts: list[list[dict[str, Any]]] = [[] for _ in candidates]
    chosen: dict[int, tuple[Candidate, dict[str, Any]]] = {}
    fallback: dict[int, tuple[Candidate, dict[str, Any]]] = {}
    for level, nudges in enumerate(_NUDGES):
        requests, owners = [], []
        for index, candidate in enumerate(candidates):
            if index in chosen or level and candidate.kind == "manual":
                continue
            for nudge in nudges:
                shifted = replace(candidate, start=candidate.start + nudge[0] / 30,
                                  end=candidate.end + nudge[1] / 30)
                requests.append(_JoinRequest((shifted,), nudge,
                                             edge_aware="saved_id" in candidate.plan))
                owners.append(index)
        results = _source_round(requests, segments, backend, directory, language, level)
        winners: dict[int, list[tuple[Candidate, dict[str, Any]]]] = {}
        for index, request, result in zip(owners, requests, results, strict=True):
            attempts[index].append(result)
            if result["pass"]:
                winners.setdefault(index, []).append((request.candidates[0], result))
            elif result.get("unverifiable"):
                fallback.setdefault(index, (request.candidates[0], result))
        for index, options in winners.items():
            chosen[index] = min(options, key=lambda item: (
                sum(abs(value) for value in item[1]["nudge"]), item[1]["click_db"],
            ))
    tested = []
    for index, candidate in enumerate(candidates):
        selected = chosen.get(index, fallback.get(index))
        final = selected[1] if selected else next(
            (item for item in reversed(attempts[index]) if "got" in item), attempts[index][-1],
        )
        evidence = {**final, "attempts": attempts[index], "identity": candidate_id(candidate)}
        unverifiable = index not in chosen and selected is not None
        if unverifiable:
            evidence["words"] = "unverifiable"
        tested.append(_Tested(selected[0] if selected else candidate, selected is not None,
                              bool(final.get("listen")) or unverifiable, evidence))
    return tested


def test_cuts(project_dir: Path, spans: Sequence[dict], *, edit: dict | None = None,
              removals: Sequence[dict] = (), rejects: Sequence[str] = ()) -> list[dict]:
    """Test requested cuts in their planned edit without writing project files."""
    from yap_studio.stages.cut import _ReadOnlyIdentity

    if not spans:
        return []
    context = ProjectContext(project_dir)
    context.identity = _ReadOnlyIdentity(context.project_dir)
    rejected = set(rejects)
    decisions = [item for item in _removals(context).removals
                 if item.status == "applied" and item.id not in rejected]
    decisions.extend(Removal(
        f"@planned:{item.get('action', index)}", item["take"], item["start"], item["end"],
        "manual", "applied",
    ) for index, item in enumerate(removals))
    final_edit = edit if edit is not None else context.json("edit.json")
    variants = {}
    groups: dict[bytes, list[int]] = {}
    asr = context.project.asr
    backend = None if asr["engine"] == "none" else resolve_backend(
        asr["engine"], asr.get("model"),
    )
    candidates = {}
    attempts: list[list[dict]] = [[] for _ in spans]
    for index, span in enumerate(spans):
        variant = deepcopy(final_edit)
        restore = span.get("restore")
        if restore is not None:
            authored = next((item for item in variant["segments"]
                             if item["id"] == span["segment"]), None)
            if authored is not None:
                authored["from" if restore["edge"] == "start" else "to"] = restore["anchor"]
        key = json.dumps(variant, sort_keys=True).encode()
        if key not in variants:
            snapshot = copy(context)
            snapshot._snapshots = {**context._snapshots, "edit.json": key}
            variants[key] = _planned(snapshot, decisions)
        groups.setdefault(key, []).append(index)
        segments = variants[key]
        segment = segments.get(span["segment"])
        if segment is None:
            attempts[index].append({"pass": False, "start": span["start"], "end": span["end"],
                                    "diff": [f"segment {span['segment']} is not in the saved cut"]})
            continue
        evidence = {"requested_cut": True,
                    "want_pause_s": float(context.project.rules["max_pause_s"])}
        if span.get("action") is not None:
            evidence["manual_id"] = f"@planned:{span['action']}"
        start = segment.start if span["edge"] == "start" else span["start"]
        end = segment.end if span["edge"] == "end" else span["end"]
        word_ids = tuple(span["words"])
        if span["edge"] is not None:
            removed = {word_id for saved in segment.existing
                       if saved.plan.get("removal_id") != evidence.get("manual_id")
                       for word_id in saved.word_ids}
            audible = [word.id for word in segment.words
                       if start <= (word.start + word.end) / 2 < end and word.id not in removed]
            word_ids = tuple(dict.fromkeys((*word_ids, *audible)))
        candidates[index] = Candidate(
            span["take"], "manual", word_ids, start, end,
            (), span["segment"], backend is None, evidence,
        )
    chosen, fallback = {}, {}
    with tempfile.TemporaryDirectory(prefix="yap-cut-joins-") as temporary:
        for level, nudges in enumerate(_NUDGES):
            for group, (key, indexes) in enumerate(groups.items()):
                requests, owners = [], []
                for index in indexes:
                    if index in chosen or index not in candidates:
                        continue
                    candidate, edge = candidates[index], spans[index]["edge"]
                    for nudge in nudges:
                        if edge == "start" and nudge[0] or edge == "end" and nudge[1]:
                            continue
                        shifted = replace(candidate, start=candidate.start + nudge[0] / 30,
                                          end=candidate.end + nudge[1] / 30)
                        requests.append(_JoinRequest((shifted,), nudge,
                                                     edge_aware=edge is not None))
                        owners.append(index)
                if not requests:
                    continue
                directory = Path(temporary) / f"edit-{group}"
                directory.mkdir(exist_ok=True)
                results = _source_round(requests, variants[key], backend, directory,
                                        context.project.language, level)
                winners: dict[int, list[dict]] = {}
                for index, result in zip(owners, results, strict=True):
                    attempts[index].append(result)
                    if result["pass"]:
                        winners.setdefault(index, []).append(result)
                    elif result.get("unverifiable"):
                        fallback.setdefault(index, result)
                for index, options in winners.items():
                    chosen[index] = min(options, key=lambda result: (
                        sum(abs(value) for value in result["nudge"]), result["click_db"],
                    ))
    tested = []
    for index, span in enumerate(spans):
        selected = chosen.get(index, fallback.get(index))
        final = selected if selected is not None else next(
            (attempt for attempt in reversed(attempts[index]) if "got" in attempt),
            attempts[index][-1],
        )
        weak = index not in chosen and selected is not None
        tested.append({
            "segment": span["segment"], "edge": span["edge"], "pass": selected is not None,
            "start": final.get("start", span["start"]), "end": final.get("end", span["end"]),
            "nudge": final.get("nudge", [0, 0]),
            "words": "skipped" if backend is None else
            "unverifiable" if weak else final.get("words", "fail"),
            "diff": final.get("diff", []), "removed_words": final.get("removed_words", []),
            "click_db": final.get("click_db", -120.0),
            "listen": backend is None or weak or bool(final.get("listen")),
            **({"combined_with": list(span["combined_with"])} if span.get("combined_with") else {}),
        })
    return tested


def _components(candidate: Candidate, segment: _PlannedAudio,
                rules: dict[str, Any]) -> list[Candidate]:
    """Recover separate rule spans when a merged candidate fails all nudges."""
    selected = {word.id for word in segment.words if word.id in candidate.word_ids}
    spans = _word_spans(segment.words, float(rules["beat_breath_s"]))
    components = []
    for span in spans:
        if not {word.id for word in segment.words[span.first:span.last + 1]} <= selected:
            continue
        item = _plan_words(segment.words, span, beat_breath_s=float(rules["beat_breath_s"]),
                           join_target_s=0.2, silence_db=-45, level_at=None)
        if item and candidate.start <= item.start < item.end <= candidate.end:
            components.append(replace(item, segment=candidate.segment))
    # U10 also merges adjacent pause trims into a speech span. Recover those from
    # the original gaps without allowing a component to remove a kept word.
    if candidate.plan.get("components"):
        pauses = _pause_candidates(segment.words, segment.existing,
                                   max_pause_s=float(rules["max_pause_s"]),
                                   silence_db=-45, level_at=None,
                                   silent_runs=_measured_silent_runs(segment), take=segment.take)
        components.extend(replace(item, segment=candidate.segment) for item in pauses
                          if candidate.start <= item.start < item.end <= candidate.end)
    unique = {candidate_id(item): item for item in components
              if candidate_id(item) != candidate_id(candidate)}
    return list(unique.values()) if len(unique) > 1 else []


def _wordless_sliver(segment: _PlannedAudio, start: float, end: float) -> bool:
    """Identify short retained spans using the cut's half-open midpoint rule."""
    return (0 < end - start < 0.25
            and not any(start <= (word.start + word.end) / 2 < end for word in segment.words)
            and not any(start < item.end and end > item.start for item in segment.existing))


def _slivers(tested: list[_Tested], segments: dict[str, _PlannedAudio], backend: Backend | None,
             directory: Path, language: str) -> list[_Tested]:
    """Absorb wordless edge fragments and verify merges across short retained gaps."""
    ordered = sorted(tested, key=lambda item: (
        item.candidate.take, item.candidate.segment, item.candidate.start,
    ))
    round_id = 200

    def retest(candidate: Candidate, absorbed: list[list[float]], *, edge_aware: bool = False,
               ) -> dict[str, Any]:
        """Test the final extended span once, without another boundary search."""
        nonlocal round_id
        result = _source_round([_JoinRequest((candidate,), edge_aware=edge_aware)],
                               segments, backend, directory, language, round_id)[0]
        round_id += 1
        return {**result, "absorbed_s": absorbed}

    for index, item in enumerate(ordered):
        if not item.passed or item.candidate.kind == "manual":
            continue
        candidate = item.candidate
        segment = segments[candidate.segment]
        neighbours = [other.candidate for other in ordered if other is not item and other.passed
                      and other.candidate.take == candidate.take
                      and other.candidate.segment == candidate.segment]
        absorbed = []
        if (not any(other.start < candidate.start for other in neighbours)
                and _wordless_sliver(segment, segment.start, candidate.start)):
            absorbed.append([segment.start, candidate.start])
        if (not any(other.end > candidate.end for other in neighbours)
                and _wordless_sliver(segment, candidate.end, segment.end)):
            absorbed.append([candidate.end, segment.end])
        if not absorbed:
            continue
        extended = replace(candidate, start=min(candidate.start, absorbed[0][0]),
                           end=max(candidate.end, absorbed[-1][1]), listen=item.listen)
        safe = all(
            any(word.id in candidate.word_ids and start < word.end and end > word.start
                for word in segment.words)
            or audio.rms_db(segment.sound, (start + end) / 2 - segment.start,
                            _RATE, (end - start) / 2)
            <= float(candidate.plan.get("silence_db", -45)) for start, end in absorbed
        )
        result = (None if safe and len(_affected_segments((extended,), segments)) == 1
                  else retest(extended, absorbed, edge_aware=True))
        evidence = {**item.evidence, "absorbed_s": [
            *item.evidence.get("absorbed_s", []), *absorbed,
        ], "start": extended.start, "end": extended.end}
        if result is not None:
            evidence.update(result, attempts=[*item.evidence.get("attempts", []), result])
        passed = result is None or bool(result["pass"] or result.get("unverifiable"))
        listen = item.listen or bool(result and (result.get("listen")
                                                or result.get("unverifiable")))
        ordered[index] = replace(item, candidate=extended, passed=passed, listen=listen,
                                 evidence=evidence)

    result_items: list[_Tested] = []
    previous_index: int | None = None
    for item in ordered:
        if not item.passed:
            result_items.append(item)
            continue
        earlier = result_items[previous_index] if previous_index is not None else None
        candidate = item.candidate
        segment = segments[candidate.segment]
        if (earlier is not None and earlier.candidate.take == candidate.take
                and earlier.candidate.segment == candidate.segment
                and earlier.candidate.kind != "manual" and candidate.kind != "manual"
                and _wordless_sliver(segment, earlier.candidate.end, candidate.start)):
            absorbed = [[earlier.candidate.end, candidate.start]]
            merged = replace(
                earlier.candidate, end=candidate.end,
                word_ids=tuple(dict.fromkeys((*earlier.candidate.word_ids, *candidate.word_ids))),
                reasons=tuple(dict.fromkeys((*earlier.candidate.reasons, *candidate.reasons))),
                listen=earlier.listen or item.listen,
            )
            checked = retest(merged, absorbed)
            if checked["pass"] or checked.get("unverifiable"):
                identities = list(dict.fromkeys([
                    *earlier.evidence.get("merged_identities", [earlier.evidence["identity"]]),
                    *item.evidence.get("merged_identities", [item.evidence["identity"]]),
                ]))
                evidence = {**earlier.evidence, **checked,
                            "attempts": [*earlier.evidence.get("attempts", []),
                                         *item.evidence.get("attempts", []), checked],
                            "merged_identities": identities, "reasons": list(merged.reasons),
                            "absorbed_s": [*earlier.evidence.get("absorbed_s", []),
                                           *item.evidence.get("absorbed_s", []), *absorbed]}
                result_items[previous_index] = replace(
                    earlier, candidate=merged, listen=merged.listen or bool(
                        checked.get("listen") or checked.get("unverifiable")), evidence=evidence,
                )
                continue
            result_items.append(replace(item, passed=False, evidence={
                **item.evidence, "pass": False, "words": "fail", "sliver_merge": checked,
                "attempts": [*item.evidence.get("attempts", []), checked],
                "diff": ["leaves a sliver under 0.25 s"],
            }))
            continue
        previous_index = len(result_items)
        result_items.append(item)
    return result_items


def _groups(tested: list[_Tested], segments: dict[str, _PlannedAudio], backend: Backend | None,
            directory: Path, language: str) -> list[_Tested]:
    """Retest close accepted cuts together, rejecting the last failing member."""
    groups: list[list[int]] = []
    for identifier, segment in segments.items():
        local: list[list[int]] = []
        for index in sorted(range(len(tested)), key=lambda i: tested[i].candidate.start):
            item = tested[index]
            if (not item.passed or item.candidate.kind == "manual"
                    or identifier not in _affected_segments((item.candidate,), segments)):
                continue
            if local and max(item.candidate.start, segment.start) - min(
                tested[local[-1][-1]].candidate.end, segment.end,
            ) < 0.3:
                local[-1].append(index)
            else:
                local.append([index])
        groups.extend(group for group in local if len(group) > 1 and group not in groups)
    round_id = 100
    while pending := [group for group in groups if len(group) > 1]:
        requests, owners = [], []
        for group in pending:
            group[:] = [index for index in group if tested[index].passed]
            if len(group) < 2:
                continue
            candidates = tuple(tested[index].candidate for index in group)
            requests.append(_JoinRequest(candidates))
            owners.append(group)
        results = _source_round(requests, segments, backend, directory, language, round_id)
        round_id += 1
        for group, result in zip(owners, results, strict=True):
            for index in group:
                tested[index] = replace(tested[index], evidence={
                    **tested[index].evidence,
                    "groups": [*tested[index].evidence.get("groups", []), result],
                })
            if not result["pass"] and not result.get("unverifiable"):
                last = group.pop()
                tested[last] = replace(tested[last], passed=False, evidence={
                    **tested[last].evidence, "diff": result["diff"], "words": "fail", "pass": False,
                })
            else:
                if result.get("unverifiable"):
                    for index in group:
                        tested[index] = replace(tested[index], listen=True, evidence={
                            **tested[index].evidence, "words": "unverifiable",
                        })
                group.clear()
    return tested


def _removals(context: ProjectContext) -> Removals:
    """Load the optional removal snapshot with its existing schema."""
    try:
        data = context.json("removals.json")
    except FileNotFoundError:
        data = {"schema": "yap-studio/removals@1", "removals": []}
    return cast(Removals, parse_document("removals", data))


def _identity(item: Removal) -> str:
    """Match an existing entry by its original or final source boundaries."""
    return candidate_id(cast(Candidate, item))


def _next_id(entries: Sequence[Removal]) -> str:
    """Allocate the next unused sequential removal id."""
    numbers = [int(match[1]) for item in entries if (match := re.fullmatch(r"r(\d+)", item.id))]
    return f"r{max(numbers, default=0) + 1}"


def _covers_removal(entries: Sequence[Removal], previous: Removal) -> bool:
    """Check that the applied source union still covers a previously removed span."""
    cursor = previous.start
    for item in sorted((item for item in entries if item.status == "applied"
                        and item.take == previous.take), key=lambda item: item.start):
        if item.end <= cursor:
            continue
        if item.start > cursor + 1e-9:
            return False
        cursor = max(cursor, item.end)
        if cursor >= previous.end - 1e-9:
            return True
    return False


def _report(entries: Sequence[Removal], backend: Backend | None, skipped: int,
            elapsed: float, *, kept_untested: Sequence[str] = (),
            refinement: dict[str, Any], restart_rules: Counter[str],
            pause_thresholds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Publish readable attempts, decisions, counts and runtime facts."""
    return {
        "schema": "yap-studio/fillers-report@1",
        "removals": [{"id": item.id, "kind": item.kind, "status": item.status,
                      "listen": item.listen, "reason": item.reason,
                      "attempts": item.evidence.get("attempts", []), "evidence": item.evidence}
                     for item in entries],
        "counts": {"kind": dict(Counter(item.kind for item in entries)),
                   "status": dict(Counter(item.status for item in entries)),
                   "restart_rules": dict(restart_rules)},
        "refinement": refinement,
        "pause_thresholds": pause_thresholds,
        "engine": backend.name if backend else "none", "model": backend.model if backend else None,
        "skipped": skipped,
        "kept_untested": list(kept_untested),
        "notes": (["fillers skipped: no speech engine"] if backend is None else []) + (
            [f"speech removals kept untested: no speech engine ({len(kept_untested)})"]
            if kept_untested else []
        ) + refinement["notes"],
        "timings": {"total_s": round(elapsed, 4)},
    }


def _check_removals_receipt(project_dir: Path) -> None:
    """Explain how to recover a removals file changed outside the CLI."""
    target = project_dir / "removals.json"
    if (project_dir / "fillers" / "receipt.pending.json").exists():
        return
    owned = read_receipt(project_dir / "fillers")
    if not target.is_file() or owned is None:
        return
    saved = next((item.get("sha256") for item in owned.get("outputs", [])
                  if item.get("path") == "removals.json"), None)
    if saved is not None and receipt.sha256_file(target) != saved:
        raise YapError(
            "removals.json was changed outside yap-studio.", code="unsafe_path",
            suggested_action=(
                "Put back the saved version or move removals.json aside, then use "
                "yap-studio fillers set DIR ID STATUS and "
                "yap-studio fillers add DIR --manual TAKE START END to make changes."
            ),
        )


def fillers(project_dir: Path, *, apply: bool = False, force: bool = False,
            propose_only: bool = False,
            expected_max_pause_s: float | None = None) -> dict[str, Any]:
    """Test proposals and publish removals; apply passing speech cuts only on request."""
    started = time.monotonic()
    with paths.operation():
        _check_removals_receipt(Path(project_dir))
        context = ProjectContext(project_dir)
        if (expected_max_pause_s is not None
                and context.project.rules["max_pause_s"] != expected_max_pause_s):
            raise ConflictError(context.project_dir / "project.json")
        previous = _removals(context).removals
        inputs = fillers_inputs(context)
        run = StageRun(context.project_dir, "fillers")
        prior_receipt = read_receipt(run.stage_dir)
        if (not force and run.up_to_date(inputs) and prior_receipt
                and (propose_only or prior_receipt.get("apply") == apply)):
            report = json.loads((run.stage_dir / "report.json").read_bytes())
            return {**report, "noop": True, "entries": Removals(previous).to_dict()["removals"]}
        asr = context.project.asr
        backend = None if asr["engine"] == "none" else resolve_backend(
            asr["engine"], asr.get("model"),
        )
        kept_untested = [item for item in previous if backend is None and item.status == "applied"
                        and item.kind in {"filler", "stutter", "restart"}]
        preserved = [item for item in previous if (
            item.status == "rejected" and (
                not force or item.by != "fillers" or "decision" in item.evidence
            )
        ) or "decision" in item.evidence and item.by != "manual"
                     or item.by == "manual" and (not force or backend is None)
                     or item in kept_untested]
        protected = [item for item in previous if item.status == "applied"
                     and (item in preserved or item.by == "manual")]
        while additions := [item for item in previous if item.status == "applied"
                            and item.by == "fillers" and item not in protected
                            and any(item.take == saved.take and item.start < saved.end
                                    and item.end > saved.start for saved in protected)]:
            preserved.extend(additions)
            protected.extend(additions)
        # Keep overlapping decisions as a saved union. Neither decision has an
        # independent control when the other is already removed.
        preserved.extend(item for item in previous if item.by == "manual"
                         and item.status == "applied" and item not in preserved
                         and any(saved.by == "fillers" and item.take == saved.take
                                 and item.start < saved.end and item.end > saved.start
                                 for saved in protected))
        with run, tempfile.TemporaryDirectory(prefix="yap-fillers-") as temporary:
            run.progress("building planned audio", 0.1)
            baseline = [*preserved, *(item for item in previous if item not in preserved
                                     and item.by == "manual" and item.status == "applied")]
            segments = _planned(context, baseline)
            pause_thresholds = {segment.take: segment.pause for segment in segments.values()
                                if segment.pause is not None}
            directory = Path(temporary)
            run.progress("refining suspicious speech", 0.2)
            refinements, refinement = _refine(segments, backend, directory,
                                             context.project.language)
            restart_rules: Counter[str] = Counter({
                "repeated 2-gram": 0, "repeated 3-gram": 0,
                "hidden repeated 2-gram": 0, "hidden repeated 3-gram": 0, "fragment restart": 0,
            })
            proposals = _proposals(context, segments, refinements=refinements,
                                   restart_rules=restart_rules)
            pause_proposals = (_proposals(context, segments, language="und")
                               if backend is None and any(item.kind != "silence"
                                                          for item in proposals) else proposals)
            existing = {_identity(item): item for item in previous}
            existing.update({item.evidence["identity"]: item for item in previous
                             if "identity" in item.evidence})
            existing.update({identity: item for item in previous
                             for identity in item.evidence.get("merged_identities", [])})
            skipped = sum(item.kind != "silence" for item in proposals) if backend is None else 0
            candidates = [item for item in pause_proposals if (
                backend is not None or item.kind == "silence"
            ) and existing.get(candidate_id(item)) not in preserved
                and not any(saved.status == "applied" and saved.take == item.take
                            and item.start < saved.end and item.end > saved.start
                            for saved in baseline)]
            saved_applied: dict[str, Candidate] = {}
            fresh = []
            for candidate in candidates:
                old = existing.get(candidate_id(candidate))
                if old is not None and old.status == "applied":
                    saved_applied[old.id] = replace(candidate, start=old.start, end=old.end,
                                                    kind=cast(Any, old.kind), word_ids=old.words,
                                                    plan={**candidate.plan, "saved_id": old.id})
                else:
                    fresh.append(candidate)
            for old in previous:
                if old.status != "applied" or old in baseline or old.id in saved_applied:
                    continue
                identifier = next((key for key, segment in segments.items()
                                   if segment.take == old.take
                                   and segment.start <= old.start < old.end <= segment.end), None)
                if identifier is not None:
                    saved_applied[old.id] = Candidate(
                        old.take, cast(Any, old.kind), old.words, old.start, old.end, (),
                        identifier, old.listen,
                        {"want_pause_s": float(context.project.rules["max_pause_s"]),
                         "saved_id": old.id},
                    )
            candidates = [*saved_applied.values(), *fresh]
            if force:
                for item in previous:
                    if item.by != "manual" or item in preserved:
                        continue
                    identifier = next((key for key, segment in segments.items()
                                       if segment.take == item.take
                                       and segment.start <= item.start < item.end <= segment.end),
                                      None)
                    if identifier is None:
                        preserved.append(item)
                        continue
                    removed = tuple(word.id for word in segments[identifier].words
                                    if item.start <= (word.start + word.end) / 2 < item.end)
                    candidates.append(Candidate(item.take, cast(Any, "manual"), removed,
                                                item.start, item.end, (), identifier,
                                                item.listen, {"manual_id": item.id}))
            run.progress("testing joins", 0.3)
            tested = _test(candidates, segments, backend, directory, context.project.language)
            saved_failures: dict[str, dict[str, Any]] = {}
            for item in tested:
                old = existing.get(item.evidence["identity"])
                if old is None or old.status != "applied" or old.by != "fillers":
                    continue
                failed = next((attempt for attempt in item.evidence["attempts"]
                               if attempt.get("start") == old.start
                               and attempt.get("end") == old.end
                               and not attempt["pass"] and not attempt.get("unverifiable")), None)
                if failed is not None:
                    saved_failures[old.id] = {
                        **old.evidence, **item.evidence, **failed,
                        "identity": old.evidence.get("identity", _identity(old)),
                    }
            expanded = []
            split_parts: list[Candidate] = []
            merged_attempts: dict[str, list[dict[str, Any]]] = {}
            for item in tested:
                parts = [] if item.passed or item.candidate.kind == "manual" else _components(
                    item.candidate, segments[item.candidate.segment], context.project.rules,
                )
                if parts:
                    split_parts.extend(parts)
                    merged_attempts.update({candidate_id(part): item.evidence["attempts"]
                                            for part in parts})
                else:
                    expanded.append(item)
            for item in _test(split_parts, segments, backend, directory, context.project.language):
                item.evidence["merged_attempts"] = merged_attempts[item.evidence["identity"]]
                expanded.append(item)
            expanded = _slivers(expanded, segments, backend, directory, context.project.language)
            tested = _groups(expanded, segments, backend, directory, context.project.language)
            entries = list(preserved)
            allocated = list(previous)
            inspection_proposals: dict[str, Removal] = {}
            for item in tested:
                candidate = item.candidate
                old = existing.get(item.evidence["identity"], existing.get(candidate_id(candidate)))
                if old is not None and any(entry.id == old.id for entry in entries):
                    old = None
                identifier = old.id if old else _next_id(allocated)
                verified = (item.evidence.get("pass") is True
                            and item.evidence.get("words") != "unverifiable"
                            and not item.evidence.get("unverifiable"))
                state = ("applied" if not propose_only and verified
                         and (apply or backend is None) else "proposed")
                if old is not None and (old.status == "applied" or old.by == "manual"
                                        or "decision" in old.evidence):
                    state = old.status
                if not item.passed and not (old and old.by == "manual"):
                    state = "rejected"
                reason = (old.reason if old and (old.by == "manual"
                                                or old.status == state == "applied") else
                          "join could not be verified; listen before applying"
                          if state == "proposed" and item.passed and not verified else
                          None if item.passed else
                          "; ".join(item.evidence.get("diff", [])) or "join test failed")
                evidence = dict(item.evidence)
                evidence["reasons"] = list(candidate.reasons)
                for key in ("refined_words", "refined_spans", "restart_audio",
                            "refined_source_words", "refined_provenance"):
                    if key in candidate.plan:
                        evidence[key] = candidate.plan[key]
                if old is not None and old.status == "applied":
                    evidence["identity"] = old.evidence.get("identity", _identity(old))
                    if "merged_identities" in old.evidence:
                        evidence["merged_identities"] = list(dict.fromkeys([
                            *old.evidence["merged_identities"],
                            *evidence.get("merged_identities", []),
                        ]))
                    if candidate.start <= old.start and candidate.end >= old.end:
                        for key in ("absorbed_s", "reasons"):
                            values = [*old.evidence.get(key, []), *evidence.get(key, [])]
                            if values:
                                evidence[key] = [value for index, value in enumerate(values)
                                                 if value not in values[:index]]
                entry = Removal(identifier, candidate.take, candidate.start, candidate.end,
                                candidate.kind, state,
                                candidate.word_ids, old.by if old else "fillers",
                                item.listen or (item.passed and not verified)
                                or bool(old and old.by == "manual" and not item.passed),
                                evidence, reason)
                if ((not apply or propose_only) and item.passed and old is not None
                        and old.status == "applied"
                        and old.by != "manual" and "decision" not in old.evidence
                        and (entry.start, entry.end) != (old.start, old.end)):
                    inspection_proposals[old.id] = entry
                entries.append(entry)
                allocated.append(entry)
            for old in previous:
                if old.status != "applied" or _covers_removal(entries, old):
                    continue
                current = next((entry for entry in entries if entry.id == old.id), None)
                if (current is not None and current.status == "rejected"
                        and current.evidence.get("pass") is False):
                    continue
                failed = saved_failures.get(old.id)
                if failed is not None:
                    if current is not None:
                        entries.remove(current)
                        replacement = replace(current, id=_next_id(allocated), evidence={
                            **{key: value for key, value in current.evidence.items()
                               if key != "merged_identities"}, "identity": _identity(current),
                            "rejected_parent": old.id,
                        })
                        entries.append(replacement)
                        allocated.append(replacement)
                        if old.id in inspection_proposals:
                            inspection_proposals[old.id] = replacement
                    entries.append(replace(old, status="rejected", evidence=failed,
                                           reason="; ".join(failed.get("diff", []))
                                           or "join test failed"))
                else:
                    if current is not None:
                        entries.remove(current)
                    entries.append(old)
            # Separate inspection proposals after the saved-boundary rejection guard.
            for old in previous:
                proposal = inspection_proposals.get(old.id)
                if proposal is None:
                    continue
                current = next((entry for entry in entries if entry.id == proposal.id), None)
                if current is not None:
                    entries.remove(current)
                if proposal.id == old.id:
                    entries.append(old)
                if any(saved.status == "rejected"
                       and (saved.by != "fillers" or "decision" in saved.evidence)
                       and saved.evidence.get("applied_parent") == old.id
                       and _identity(saved) == _identity(proposal) for saved in preserved):
                    continue
                entry = replace(proposal, id=_next_id(allocated) if proposal.id == old.id
                                else proposal.id, status="proposed", evidence={
                                    **{key: value for key, value in proposal.evidence.items()
                                       if key != "merged_identities"},
                                    "identity": _identity(proposal), "applied_parent": old.id,
                                })
                entries.append(entry)
                allocated.append(entry)
            entries.sort(key=lambda item: (item.take, item.start, item.end, item.id))
            document = json.loads(json.dumps(Removals(tuple(entries)).to_dict()))
            parse_document("removals", document)
            report = _report(entries, backend, skipped, time.monotonic() - started,
                             kept_untested=[item.id for item in kept_untested],
                             refinement=refinement, restart_rules=restart_rules,
                             pause_thresholds=pause_thresholds)
            # Record preserved rejections and creator decisions in the resulting
            # input projection, avoiding a self-invalidating receipt.
            inputs["removals:decisions"] = projection_sha256(removal_decisions(document))
            for name, digest in inputs.items():
                run.record_inputs(name, digest)
            run.progress("publishing removals", 0.9)
            with project_lock(context.project_dir):
                current = (context.project_dir / "removals.json")
                original = context._snapshots.get("removals.json")
                if (current.read_bytes() if current.exists() else None) != original:
                    raise ConflictError(current)
            for relative, data in (("removals.json", document), ("fillers/report.json", report)):
                staged = directory / Path(relative).name
                staged.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8")
                run.replace_output(context.project_dir / relative, staged)
            run.finish(extra={"apply": apply, "pause_thresholds": pause_thresholds})
        return {**report, "noop": False, "entries": document["removals"]}


def _change(project_dir: Path, change: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """Rewrite decisions and receipt ownership in one locked, journaled transaction."""
    project_dir = Path(project_dir).resolve()
    with paths.operation(), project_lock(project_dir):
        folder = project_dir / "fillers"
        progress = folder / "progress.json"
        if progress.exists() and pid_alive(json.loads(progress.read_bytes()).get("pid")):
            raise YapError("fillers is already running.", code="stage_running",
                           suggested_action="Wait for the stage and retry.")
        pending = folder / "receipt.pending.json"
        if pending.exists():
            raise InputError("fillers", "unfinished stage journal; rerun fillers first")
        _check_removals_receipt(project_dir)
        context = ProjectContext(project_dir)
        original = (context.read("removals.json")
                    if (project_dir / "removals.json").exists() else None)
        data = json.loads(original) if original is not None else Removals(()).to_dict()
        parse_document("removals", data)
        target = project_dir / "removals.json"
        owned = read_receipt(folder)
        paths.validate_output(target, owner_receipts=(folder / "receipt.json",))
        identifier = change(data)
        parse_document("removals", data)
        paths.validate_output_dir(folder)
        payload = _decision_bytes(original, data)
        if original == payload and owned is not None:
            return {"schema": "yap-studio/removals@1", "removals": data["removals"],
                    "id": identifier}
        digest = hashlib.sha256(payload).hexdigest()
        old = receipt.sha256_file(target) if target.exists() else None
        finished = owned or {
            "schema": "yap-studio/stage-receipt@1", "stage": "fillers", "status": "verified",
            "started": timestamp(), "inputs": {}, "outputs": [], "asserts": [],
        }
        outputs = [item for item in finished["outputs"] if item["path"] != "removals.json"]
        outputs.append({"path": "removals.json", "sha256": digest, "bytes": len(payload)})
        if owned is not None:
            paths._guard_write(folder / "receipt.json")
            paths._guard_write(pending)
            os.replace(folder / "receipt.json", pending)
        _write_json(pending, {**finished, "schema": "yap-studio/stage-pending@1", "outputs": [
            *outputs, {"path": "removals.json", "old_sha256": old, "new_sha256": digest},
        ]})
        with paths.atomic_output(
            target, owner_receipts=(folder / "receipt.json", pending),
        ) as staged:
            staged.write_bytes(payload)
        finished.update(outputs=outputs, finished=timestamp())
        _write_json(folder / "receipt.json", finished)
        pending.unlink()
        return {"schema": "yap-studio/removals@1", "removals": data["removals"], "id": identifier}


def _decision_bytes(original: bytes | None, data: dict[str, Any]) -> bytes:
    """Replace changed array entries while preserving other entries byte for byte."""
    if original is None:
        return (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    text = original.decode("utf-8")
    decoder = json.JSONDecoder()

    def skip(index: int) -> int:
        """Advance over JSON whitespace without changing the source text."""
        while text[index].isspace():
            index += 1
        return index

    index = skip(text.index("{") + 1)
    while True:
        key, index = decoder.raw_decode(text, index)
        index = skip(skip(index) + 1)
        value, end = decoder.raw_decode(text, index)
        if key == "removals":
            break
        index = skip(skip(end) + 1)
    ranges: list[tuple[int, int]] = []
    cursor = skip(index + 1)
    while text[cursor] != "]":
        _, last = decoder.raw_decode(text, cursor)
        ranges.append((cursor, last))
        cursor = skip(last)
        if text[cursor] == ",":
            cursor = skip(cursor + 1)
    changes: list[tuple[int, int, str]] = []
    for old, new, (first, last) in zip(value, data["removals"], ranges, strict=False):
        if old == new:
            continue
        indent = text[text.rfind("\n", 0, first) + 1:first]
        indent = indent if indent.isspace() else ""
        rendered = json.dumps(new, indent=2, ensure_ascii=False, allow_nan=False)
        changes.append((first, last, rendered.replace("\n", "\n" + indent)))
    if len(data["removals"]) > len(value):
        rendered = json.dumps(data["removals"][-1], indent=2, ensure_ascii=False, allow_nan=False)
        rendered = rendered.replace("\n", "\n    ")
        insertion = ranges[-1][1] if ranges else index + 1
        changes.append((insertion, insertion, ("," if ranges else "") + "\n    " + rendered))
    for first, last, replacement in sorted(changes, reverse=True):
        text = text[:first] + replacement + text[last:]
    return text.encode("utf-8")


def set_removal(project_dir: Path, identifier: str, status: str,
                *, reason: str | None = None) -> dict[str, Any]:
    """Record a creator decision, replacing an applied inspection parent when needed."""
    if status not in {"proposed", "applied", "rejected"}:
        raise InputError(status, "expected proposed, applied or rejected")

    def change(data: dict[str, Any]) -> str:
        """Update the chosen entry and its applied parent in the same locked write."""
        item = next((item for item in data["removals"] if item["id"] == identifier), None)
        if item is None:
            raise InputError(identifier, "unknown removal id")
        parent_id = item.get("evidence", {}).get("applied_parent") if status == "applied" else None
        parent = next((entry for entry in data["removals"]
                       if entry["id"] == parent_id and entry["status"] == "applied"), None)
        if (item["status"] == status and (reason is None or item.get("reason") == reason)
                and (status != "rejected" or item.get("by") == "manual"
                     or "decision" in item.get("evidence", {})) and parent is None):
            return identifier
        if parent is not None:
            parent["status"] = "rejected"
            parent["reason"] = f"replaced by {identifier}"
            parent.setdefault("evidence", {}).update(decision=True, replaced_by=identifier)
        item["status"] = status
        if reason is not None:
            item["reason"] = reason
        elif status == "rejected":
            item.setdefault("reason", "rejected by creator")
        elif status != "rejected":
            item.pop("reason", None)
        item.setdefault("evidence", {})["decision"] = True
        return identifier

    return _change(project_dir, change)


def add_manual(project_dir: Path, take: str, start: float, end: float,
               *, reason: str | None = None) -> dict[str, Any]:
    """Add an applied manual source removal without running a join test."""
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise InputError("manual removal", "expected finite times with 0 <= start < end")

    def change(data: dict[str, Any]) -> str:
        """Allocate a sequential identity after validating the take."""
        context = ProjectContext(project_dir)
        if take not in {item.id for item in context.project.takes}:
            raise InputError(take, "unknown take")
        entries = cast(Removals, parse_document("removals", data)).removals
        identifier = _next_id(entries)
        entry = Removal(identifier, take, start, end, "manual", "applied", by="manual",
                        reason=reason)
        data["removals"].append(json.loads(json.dumps(Removals((entry,)).to_dict()))["removals"][0])
        return identifier

    return _change(project_dir, change)
