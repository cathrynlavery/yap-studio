"""Place story edges in the take's real silence and confirm the words they keep.

Speech engines time words loosely: on fluent speech a word can be reported a
quarter second early or late, and touching words get no gap at all. An edge
placed only from word times can keep the tail of a dropped word or clip a kept
one. Each word-anchored edge is therefore checked against the take's sound:
candidates sit in silence near the word edge, and a short transcription from
each candidate must start (or end) on the edge's words. An edge that finds no
silence settles on the quietest confirmed point and asks for a listen.
Without a speech engine, word-anchored edges stay at their word times and ask
for a listen unless that point is already in silence.
Each take uses the highest pause threshold that adds at most 2% of word midpoints
to the silence found just above its quiet floor. Takes with too few words or
unreliable timings use the quiet-floor rule.
"""

from __future__ import annotations

import difflib
import math
import tempfile
import unicodedata
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import audio
from yap_studio.errors import YapError
from yap_studio.project.anchors import END_PAD_S, START_PAD_S
from yap_studio.project.anchors import MATCH_WORDS as MATCH_WORDS
from yap_studio.project.model import Word
from yap_studio.transcript.backends import Backend
from yap_studio.transcript.model import Transcript
from yap_studio.transcript.words import norm_words, unicode_words

RATE = 16_000
HOP_S = 0.01
WINDOW_S = 0.02
SEARCH_S = 0.5
SILENCE_MIN_S = 0.04
SILENCE_ABOVE_FLOOR_DB = 15.0
SILENCE_CEILING_DBFS = -45.0
# Bump when the pause rule or the silence-based listen decisions change.
PAUSE_RULE_VERSION = 4
PAUSE_SHARE_MAX = 0.02
PAUSE_BASELINE_MAX = 0.2
PAUSE_MIN_WORDS = 10
PAUSE_SCAN_TOP_DBFS = -10.0
PAUSE_SCAN_STEP_DB = 0.5
PAUSE_ABOVE_FLOOR_MIN_DB = 3.0
CLIP_S = 3.0
CLIP_PAD_S = 0.3
LAST_WORD_LEAD_S = 0.10
MATCH_RATIO = 0.75
LONG_WORD = 4
QUIET_CANDIDATES = 4
DIP_REACH_S = 0.15
DIP_DEPTH_DB = 4.0
DIP_NUDGE_S = 0.03
VOICED_TRIM_MIN_S = 0.25
VOICED_PAD_S = 0.05
VOICED_OWNER_RATIO = 2
VOICED_RULE_VERSION = 5
SILENT_REALIGN_PHRASE_WORD_CAP = 7
SILENT_REALIGN_CONTEXT_VOICE_SHARE = 0.5
SILENT_REALIGN_TARGET_VOICE_SHARE = 0.8


def quiet_floor_threshold_db(floor_db: float) -> float:
    """Keep quiet speech above the conservative silence threshold."""
    return min(floor_db + SILENCE_ABOVE_FLOOR_DB, SILENCE_CEILING_DBFS)


def pause_quiet_threshold_db(floor_db: float) -> float:
    """Keep the pause silence threshold above the measured noise floor."""
    return max(quiet_floor_threshold_db(floor_db), floor_db + PAUSE_ABOVE_FLOOR_MIN_DB)


def pause_rule_identity() -> dict[str, float]:
    """Identify the pause rule and its silence-based listen decisions."""
    return {
        "PAUSE_RULE_VERSION": PAUSE_RULE_VERSION,
        "PAUSE_SHARE_MAX": PAUSE_SHARE_MAX,
        "PAUSE_BASELINE_MAX": PAUSE_BASELINE_MAX,
        "PAUSE_MIN_WORDS": PAUSE_MIN_WORDS,
        "PAUSE_SCAN_TOP_DBFS": PAUSE_SCAN_TOP_DBFS,
        "PAUSE_SCAN_STEP_DB": PAUSE_SCAN_STEP_DB,
        "PAUSE_ABOVE_FLOOR_MIN_DB": PAUSE_ABOVE_FLOOR_MIN_DB,
        "SILENCE_ABOVE_FLOOR_DB": SILENCE_ABOVE_FLOOR_DB,
        "SILENCE_CEILING_DBFS": SILENCE_CEILING_DBFS,
        "SILENCE_MIN_S": SILENCE_MIN_S,
        "HOP_S": HOP_S,
        "VOICED_TRIM_MIN_S": VOICED_TRIM_MIN_S,
        "VOICED_PAD_S": VOICED_PAD_S,
        "VOICED_OWNER_RATIO": VOICED_OWNER_RATIO,
        "VOICED_RULE_VERSION": VOICED_RULE_VERSION,
    }


@dataclass(frozen=True)
class EdgeRequest:
    """One segment edge to place, with the words it must keep."""

    segment: str
    take: str
    start: bool
    nominal: Fraction
    expected: tuple[str, ...]
    exact: bool = False
    word_start: Fraction | None = None
    word_end: Fraction | None = None
    other: Fraction | None = None
    duration: Fraction | None = None


@dataclass(frozen=True)
class EdgeResult:
    """Where an edge landed and why."""

    segment: str
    start: bool
    point: Fraction
    nominal: Fraction
    how: str
    listen: bool
    heard: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return readable seconds for the cut report."""
        return {
            "segment": self.segment,
            "edge": "from" if self.start else "to",
            "time_s": float(self.point),
            "word_time_s": float(self.nominal),
            "shift_s": round(float(self.point - self.nominal), 3),
            "how": self.how,
            "listen": self.listen,
            "heard": self.heard,
        }


@dataclass(frozen=True)
class PauseThreshold:
    """The pause level chosen for a take and the evidence for its rule."""

    threshold_db: float
    rule: str
    floor_db: float
    share: float | None
    words: int
    baseline_share: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return readable levels and the share of words detected as silence."""
        return {
            "threshold_db": round(self.threshold_db, 1),
            "rule": self.rule,
            "floor_db": round(self.floor_db, 1),
            "share": None if self.share is None else round(self.share, 4),
            "baseline_share": (None if self.baseline_share is None
                               else round(self.baseline_share, 4)),
            "words": self.words,
        }


@dataclass(frozen=True)
class _Take:
    sound: np.ndarray
    level: np.ndarray
    threshold: float
    pause: PauseThreshold | None = None
    cautious_threshold: float = field(init=False)

    def __post_init__(self) -> None:
        threshold = self.threshold
        if self.pause is not None:
            threshold = min(threshold, quiet_floor_threshold_db(self.pause.floor_db))
        object.__setattr__(self, "cautious_threshold", threshold)

    def silent(self, index: int, *, threshold: float | None = None) -> bool:
        threshold = self.threshold if threshold is None else threshold
        return 0 <= index < len(self.level) and bool(self.level[index] < threshold)


def _levels(sound: np.ndarray) -> np.ndarray:
    """RMS level in dBFS on a 10 ms hop with a 20 ms window."""
    hop = round(HOP_S * RATE)
    half = round(WINDOW_S * RATE) // 2
    power = np.concatenate(([0.0], np.cumsum(sound.astype(np.float64) ** 2)))
    centres = np.arange(0, len(sound), hop)
    first = np.clip(centres - half, 0, len(sound))
    last = np.clip(centres + half, 0, len(sound))
    width = np.maximum(last - first, 1)
    rms = np.sqrt((power[last] - power[first]) / width)
    return 20 * np.log10(rms + 1e-9)


def voiced_words(take: _Take, words: Sequence[Word]) -> tuple[Word, ...]:
    """Trim silent margins around voice owned unambiguously by one word edge."""
    hop = Fraction(str(HOP_S))
    minimum = Fraction(str(VOICED_TRIM_MIN_S))
    pad = Fraction(str(VOICED_PAD_S))
    threshold = (quiet_floor_threshold_db(take.pause.floor_db)
                 if take.pause is not None else take.cautious_threshold)
    result = []
    for word in words:
        start, end = Fraction(str(word.start)), Fraction(str(word.end))
        first = max(0, math.ceil(start / hop))
        last = min(len(take.level), math.ceil(end / hop))
        voiced = np.flatnonzero(take.level[first:last] >= threshold)
        if not len(voiced):
            result.append(word)
            continue
        runs = np.split(voiced, np.flatnonzero(np.diff(voiced) > math.ceil(minimum / hop)) + 1)
        if len(runs) > 1:
            first_anchored = Fraction(first + int(voiced[0])) * hop - start < minimum
            last_anchored = end - Fraction(first + int(voiced[-1]) + 1) * hop < minimum
            if first_anchored != last_anchored:
                voiced = runs[0] if first_anchored else runs[-1]
            elif first_anchored and last_anchored:
                if len(runs[0]) >= VOICED_OWNER_RATIO * len(runs[-1]):
                    voiced = runs[0]
                elif len(runs[-1]) >= VOICED_OWNER_RATIO * len(runs[0]):
                    voiced = runs[-1]
        voice_start = Fraction(first + int(voiced[0])) * hop
        voice_end = Fraction(first + int(voiced[-1]) + 1) * hop
        new_start = max(start, voice_start - pad) if voice_start - start >= minimum else start
        new_end = min(end, voice_end + pad) if end - voice_end >= minimum else end
        result.append(replace(word, start=float(new_start), end=float(new_end))
                      if new_start < new_end else word)
    return tuple(result)


def _local_words_match(heard: str, expected: str) -> bool:
    """Match local words exactly while retaining Unicode letters and marks."""
    if not any(ord(char) > 127 and (char.isalnum() or unicodedata.category(char).startswith("M"))
               for char in heard + expected):
        heard_words = norm_words(heard)
        return bool(heard_words) and heard_words == norm_words(expected)

    heard_words = unicode_words(heard)
    return bool(heard_words) and heard_words == unicode_words(expected)


def realign_silent_words(
    take: _Take, words: Sequence[Word], removed: Sequence[tuple[float, float]],
    backend: Backend | None | Callable[[], Backend | None], *,
    take_id: str, language: str | None = None,
) -> tuple[tuple[Word, ...], list[dict[str, Any]], list[str]]:
    """Confirm a misplaced long tail word with a short, exact local transcript."""
    candidates: list[tuple[int, float, float, list[Word]]] = []
    warnings: list[str] = []
    threshold = max(take.cautious_threshold, SILENCE_CEILING_DBFS)
    for index, word in enumerate(words):
        if index == 0 or word.end - word.start < 1.0:
            continue
        first = max(0, math.floor(word.start / HOP_S))
        last = min(len(take.level), math.ceil(word.end / HOP_S))
        if first >= last or np.any(take.level[first:last] >= threshold):
            continue
        voiced = np.flatnonzero(take.level[max(0, first - 50):first] >= threshold)
        if not len(voiced):
            continue
        voice_end = (max(0, first - 50) + int(voiced[-1]) + 1) * HOP_S
        if word.start - voice_end > 0.5:
            continue
        clip_start = max(0.0, voice_end - 2.85)
        # Bound the ASR request to nearby context. Keeping the final seven source
        # words excludes earlier crossing timestamps without changing word order.
        local = [source for source in words[:index + 1]
                 if source.start >= clip_start and source.start <= word.start][
                     -SILENT_REALIGN_PHRASE_WORD_CAP:]
        if len(local) < 2 or local[-1].id != word.id:
            continue
        clip_start = max(clip_start, local[0].start - 0.1)
        candidates.append((index, clip_start, voice_end, local))
    if not candidates:
        return tuple(words), [], warnings
    if len(candidates) > 8:
        return tuple(words), [], [
            f"Take {take_id}: too many silent-word candidates ({len(candidates)}); "
            "original timings kept."
        ]
    candidate_words = [
        {word.id for word in local} for _, _, _, local in candidates
    ]
    if any(left & right for index, left in enumerate(candidate_words)
           for right in candidate_words[index + 1:]):
        return tuple(words), [], [
            f"Take {take_id}: silent-word candidate phrases overlap; original timings kept."
        ]
    try:
        if callable(backend):
            backend = backend()
    except (YapError, OSError, RuntimeError) as exc:
        return tuple(words), [], [
            f"Take {take_id}: silent-word engine failed ({exc}); original timings kept."
        ]
    if backend is None:
        return tuple(words), [], [
            f"Take {take_id}: no speech engine for silent-word correction; "
            "original timings kept."
        ]
    corrected = list(words)
    evidence: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="yap-realign-") as temporary:
        clips: list[Path] = []
        for index, start, voice_end, _ in candidates:
            path = Path(temporary) / f"{index}.wav"
            _write_clip(path, take.sound[round(start * RATE):round((voice_end + 0.05) * RATE)])
            clips.append(path)
        try:
            heard = backend.transcribe_many(clips, language=language)
        except (YapError, OSError, RuntimeError) as exc:
            return tuple(words), [], [
                f"Take {take_id}: silent-word engine failed ({exc}); "
                "original timings kept."
            ]
        for (index, start, voice_end, local), clip in zip(candidates, clips, strict=True):
            source = words[index]
            transcript = heard.get(clip)
            reason = None
            if transcript is None or len(transcript.words) != len(local) or any(
                not _local_words_match(got.text, expected.text)
                for got, expected in zip(transcript.words, local, strict=False)
            ):
                reason = "local text or order mismatch"
            else:
                positions = [next(i for i, word in enumerate(words) if word.id == item.id)
                             for item in local]
                proposed = [(start + item.start, start + item.end)
                            for item in transcript.words]
                final_start, final_end = proposed[-1]
                if voice_end < final_end <= voice_end + 0.1 and final_start < voice_end:
                    proposed[-1] = (final_start, voice_end)
                for phrase_index, ((new_start, new_end), item) in enumerate(
                    zip(proposed, transcript.words, strict=True)
                ):
                    first = max(0, math.floor(new_start / HOP_S))
                    last = min(len(take.level), math.ceil(new_end / HOP_S))
                    # Context words can include real pauses ("ever" in the pickup).
                    # The displaced target needs stronger measured voice support.
                    minimum_share = (SILENT_REALIGN_TARGET_VOICE_SHARE
                                     if phrase_index == len(local) - 1
                                     else SILENT_REALIGN_CONTEXT_VOICE_SHARE)
                    if not (0 <= item.start < item.end and new_start < new_end
                            and new_end <= voice_end + 0.02
                            and first < last and
                            np.mean(take.level[first:last] >= threshold) >= minimum_share):
                        reason = "phrase outside measured voice"
                        break
                    if any(new_start < right and left < new_end for left, right in removed):
                        reason = "phrase overlaps applied removal"
                        break
                if reason is None and any(left[1] > right[0]
                                          for left, right in zip(proposed, proposed[1:],
                                                                 strict=False)):
                    reason = "local phrase overlap"
                if (reason is None and positions[0] > 0
                        and proposed[0][0] < corrected[positions[0] - 1].end):
                    reason = "phrase crosses source word order"
                if (reason is None and positions[-1] + 1 < len(words)
                        and proposed[-1][1] > words[positions[-1] + 1].start):
                    reason = "phrase crosses source word order"
                if reason is None:
                    for position, (new_start, new_end) in zip(positions, proposed, strict=True):
                        original = words[position]
                        corrected[position] = replace(original, start=new_start, end=new_end)
                        evidence.append({"word": original.id,
                                         "original": [original.start, original.end],
                                         "corrected": [new_start, new_end],
                                         "clip": [start, voice_end + 0.05],
                                         "engine": backend.name, "model": backend.model})
            if reason:
                warnings.append(
                    f"Take {take_id} word {source.id}: silent-word {reason}; "
                    "original timing kept."
                )
    return tuple(corrected), evidence, warnings


def might_need_silent_word_realign(words: Sequence[Word]) -> bool:
    """Conservative metadata hint; audio determines whether correction is needed."""
    return any(word.end - word.start >= 1.0 for word in words[1:])


def choose_pause_threshold(level: np.ndarray, midpoints_s: Sequence[float]) -> PauseThreshold:
    """Choose the highest level that adds few silent words above the baseline."""
    floor = float(np.percentile(level, 10)) if len(level) else SILENCE_CEILING_DBFS
    fallback = quiet_floor_threshold_db(floor)
    midpoints = np.asarray(midpoints_s, dtype=np.float64)
    midpoints = midpoints[(midpoints >= 0) & (midpoints < len(level) * HOP_S)]
    hops = np.rint(midpoints / HOP_S).astype(np.intp)
    hops = hops[(hops >= 0) & (hops < len(level))]
    words = len(hops)
    if words < PAUSE_MIN_WORDS:
        return PauseThreshold(fallback, "floor", floor, None, words)

    minimum = round(SILENCE_MIN_S / HOP_S)
    bottom = floor + PAUSE_ABOVE_FLOOR_MIN_DB

    def silent_words(threshold: float) -> int:
        silent = level < threshold
        boundaries = np.flatnonzero(np.diff(silent, prepend=False, append=False))
        starts, stops = boundaries[::2], boundaries[1::2]
        long_enough = stops - starts >= minimum
        starts, stops = starts[long_enough], stops[long_enough]
        # Starts count at their hop; exclusive stops do not. Runs span the whole take.
        inside = (np.searchsorted(starts, hops, side="right")
                  != np.searchsorted(stops, hops, side="right"))
        return int(np.count_nonzero(inside))

    baseline_words = silent_words(bottom)
    baseline = baseline_words / words
    if baseline > PAUSE_BASELINE_MAX or bottom > PAUSE_SCAN_TOP_DBFS:
        return PauseThreshold(fallback, "floor", floor, silent_words(fallback) / words,
                              words, baseline)

    steps = int(np.floor((PAUSE_SCAN_TOP_DBFS - bottom) / PAUSE_SCAN_STEP_DB)) + 1
    for step in range(steps):
        threshold = PAUSE_SCAN_TOP_DBFS - step * PAUSE_SCAN_STEP_DB
        count = baseline_words if threshold == bottom else silent_words(threshold)
        # Subtract counts first so the exact 2% allowance survives float rounding.
        if (count - baseline_words) / words <= PAUSE_SHARE_MAX:
            return PauseThreshold(threshold, "words", floor, count / words, words, baseline)
    # Include the exact lower bound when it falls between scan steps.
    return PauseThreshold(bottom, "words", floor, baseline, words, baseline)


def load_take(
    source: Path, stream: int | None, words: Sequence[tuple[float, float]] = (),
) -> _Take:
    """Decode a take on its container clock and choose its pause level from words."""
    _, sound = audio.decode(source, stream=stream, sr=RATE, mono=True)
    level = _levels(sound)
    pause = choose_pause_threshold(level, [(start + end) / 2 for start, end in words])
    return _Take(sound, level, pause.threshold_db, pause)


def _hop(point: Fraction | float) -> int:
    return round(float(point) / HOP_S)


def _runs(take: _Take, first: int, last: int, *,
          threshold: float | None = None) -> list[tuple[int, int]]:
    """Silent hop runs, at least SILENCE_MIN_S long, that touch [first, last)."""
    minimum = round(SILENCE_MIN_S / HOP_S)
    runs = []
    index = max(0, first)
    while index < min(last, len(take.level)):
        if not take.silent(index, threshold=threshold):
            index += 1
            continue
        left = index
        while take.silent(left - 1, threshold=threshold):
            left -= 1
        right = index
        while take.silent(right, threshold=threshold):
            right += 1
        if right - left >= minimum:
            runs.append((left, right))
        index = right
    return runs


def _placed(run: tuple[int, int], start: bool) -> Fraction:
    """Put a start edge just before the sound after a silence, an end edge just after."""
    left = Fraction(run[0]) * Fraction(str(HOP_S))
    right = Fraction(run[1]) * Fraction(str(HOP_S))
    if start:
        return max(left, right - min((right - left) / 2, START_PAD_S))
    return min(right, left + min((right - left) / 2, END_PAD_S))


def _containing(take: _Take, request: EdgeRequest) -> list[Fraction]:
    """The padded placement inside the silence run that holds the word edge, if any."""
    index = _hop(request.nominal)
    return [point for run in _runs(take, index, index + 1)
            if _allowed(take, request, point := _placed(run, request.start))]


def _allowed(take: _Take, request: EdgeRequest, point: Fraction) -> bool:
    """Keep candidates on the take and on the proper side of the segment's other edge."""
    if not 0 <= point <= Fraction(len(take.sound), RATE):
        return False
    if request.duration is not None and point > request.duration:
        return False
    return request.other is None or (
        point < request.other if request.start else point > request.other
    )


def in_silence(take: _Take, point: Fraction) -> bool:
    """True when the edge sits inside a silence run of the minimum length."""
    index = _hop(point)
    return bool(_runs(take, index, index + 1))


def in_cautious_silence(take: _Take, point: Fraction) -> bool:
    """Require a quiet-floor silence run when no engine confirms the edge."""
    index = _hop(point)
    return bool(_runs(take, index, index + 1, threshold=take.cautious_threshold))


def _quietest(take: _Take, first: int, last: int) -> list[int]:
    """The deepest dips between sounds in a window, kept apart from each other."""
    offset = max(0, first)
    window = take.level[offset : min(last, len(take.level))]
    if len(window) < 3:
        return []
    reach = round(DIP_REACH_S / HOP_S)

    def between_sounds(index: int) -> bool:
        # A dip between two sounds, not the fade into a pause: louder on both sides.
        here = offset + index
        left = take.level[max(0, here - reach) : here]
        right = take.level[here + 1 : here + 1 + reach]
        return (
            len(left) > 0
            and len(right) > 0
            and min(left.max(), right.max()) >= take.level[here] + DIP_DEPTH_DB
        )

    minima = [
        index
        for index in range(1, len(window) - 1)
        if window[index] <= window[index - 1]
        and window[index] <= window[index + 1]
        and not take.silent(offset + index)
        and between_sounds(index)
    ]
    chosen: list[int] = []
    for index in sorted(minima, key=lambda item: window[item]):
        if all(abs(index - other) >= 8 for other in chosen):
            chosen.append(index)
        if len(chosen) == QUIET_CANDIDATES:
            break
    return [offset + index for index in chosen]


def candidates(take: _Take, request: EdgeRequest) -> list[tuple[Fraction, str]]:
    """Silence placements nearest the word edge first, then the quietest points."""
    nominal = request.nominal
    reach = Fraction(str(SEARCH_S))
    low, high = nominal - reach, nominal + reach
    # Engines stretch a phrase's last word over the pause after it (and the first word over
    # the pause before), so search the anchor word's whole reported span as well.
    if request.start and request.word_end is not None:
        high = max(high, request.word_end)
    if not request.start and request.word_start is not None:
        low = min(low, request.word_start)
    first, last = _hop(low), _hop(high)
    found: list[tuple[Fraction, str]] = []
    # The pause holding the word edge comes first, padded from its sound, so an edge the
    # engine stretched into a long pause drops the dead air.
    found.extend((point, "silence") for point in _containing(take, request))
    placements = sorted(
        (_placed(run, request.start) for run in _runs(take, first, last)),
        key=lambda point: abs(point - nominal),
    )
    found.extend((point, "silence") for point in placements)
    # A short kept word can start inside the dip, so also try just before (or after) it.
    nudge = Fraction(str(DIP_NUDGE_S)) * (-1 if request.start else 1)
    for index in _quietest(take, first, last):
        point = Fraction(index) * Fraction(str(HOP_S))
        found.extend(((point, "quietest"), (point + nudge, "quietest")))
    unique: list[tuple[Fraction, str]] = []
    for point, how in found:
        if _allowed(take, request, point) and all(
            abs(point - other) >= Fraction(2, 100) for other, _ in unique
        ):
            unique.append((point, how))
    return unique


def _same(left: str, right: str) -> bool:
    return left == right or difflib.SequenceMatcher(None, left, right).ratio() >= MATCH_RATIO


def heard_words(text: str) -> list[str]:
    """Normalised words heard, keeping a placeholder for words in other scripts.

    A multilingual engine can write a clipped syllable in another alphabet; dropping it
    would let a stray sound at the cut pass as silence.
    """
    words: list[str] = []
    for token in text.split():
        found = norm_words(token)
        if found:
            words.extend(found)
        elif any(character.isalnum() for character in token):
            words.append("?")
    return words


def _matches(heard: list[str], expected: Sequence[str], start: bool) -> bool:
    """The word heard right at the cut is the edge's word, and nothing lies beyond it.

    Fillers count at the boundary just like any other word.
    A short edge word ("a", "if") also needs its neighbour, so a stray syllable cannot pass.
    """
    if not expected or not heard:
        return False
    got = heard if start else heard[::-1]
    want = list(expected if start else expected[::-1])
    if not _same(got[0], want[0]):
        return False
    if len(want[0]) >= LONG_WORD or len(want) == 1:
        return True
    return len(got) > 1 and _same(got[1], want[1])


def _inside(transcript: Transcript, cut_s: float, start: bool) -> bool:
    """The last word heard, including a filler, starts before an end cut's silent pad.

    An engine can finish an expected phrase inside the silence padded after a cut
    ("here you" + "go"), stamping the made-up word just before the cut. A spoken last word
    starts at least LAST_WORD_LEAD_S before the cut; one that starts later was not said. Start
    edges are not held to this: with no words before them there is no phrase to finish,
    and the engine's 80 ms timing can place a real first word's end inside the pad.
    """
    spoken = [word for word in transcript.words if heard_words(word.text)]
    if not spoken:
        return False
    return start or spoken[-1].start <= cut_s - LAST_WORD_LEAD_S


def _write_clip(path: Path, sound: np.ndarray) -> None:
    pcm = (np.clip(sound, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())


def _clip(take: _Take, point: Fraction, start: bool) -> np.ndarray:
    """Sound on the kept side of a candidate, with a little silence at the cut.

    The engine can drop a word that runs right to the end of a clip; the pad lets it finish.
    """
    index = round(float(point) * RATE)
    length = round(CLIP_S * RATE)
    first, last = (index, index + length) if start else (max(0, index - length), index)
    pad = np.zeros(round(CLIP_PAD_S * RATE), dtype=np.float32)
    sound = take.sound[first:last]
    return np.concatenate((pad, sound) if start else (sound, pad))


def place(
    requests: Sequence[EdgeRequest],
    takes: dict[str, _Take],
    backend: Backend | None,
    *,
    language: str | None = None,
) -> tuple[list[EdgeResult], list[str]]:
    """Settle every edge, transcribing all candidate clips in one engine run."""
    warnings: list[str] = []
    options = {
        index: candidates(takes[request.take], request)
        for index, request in enumerate(requests)
        if not request.exact and request.expected
        and (request.start or request.nominal != request.duration)
    }
    heard: dict[tuple[int, int], str] = {}
    timed: dict[tuple[int, int], bool] = {}
    listening = backend is not None
    if backend is not None and options:
        with tempfile.TemporaryDirectory(prefix="yap-edges-") as temporary:
            clips: dict[Path, tuple[int, int]] = {}
            cuts: dict[Path, float] = {}
            for index, found in options.items():
                request = requests[index]
                for number, (point, _) in enumerate(found):
                    path = Path(temporary) / f"e{index:04d}-{number:02d}.wav"
                    sound = _clip(takes[request.take], point, request.start)
                    _write_clip(path, sound)
                    clips[path] = (index, number)
                    cuts[path] = CLIP_PAD_S if request.start else len(sound) / RATE - CLIP_PAD_S
            try:
                transcripts = backend.transcribe_many(list(clips), language=language)
            except YapError as exc:
                if exc.code not in {"speech_engine_failed", "speech_engine_not_found"}:
                    raise
                warnings.append(
                    f"Cut edges were placed without listening to the words ({exc.message})."
                )
                transcripts = {}
                listening = False
            if listening and (missing := len(clips.keys() - transcripts.keys())):
                warnings.append(
                    f"Speech engine returned no transcript for {missing} cut-edge clips; "
                    "those candidates remain unconfirmed."
                )
            for path, key in clips.items():
                if path in transcripts:
                    heard[key] = transcripts[path].text
                    timed[key] = _inside(transcripts[path], cuts[path], requests[key[0]].start)
    elif options:
        warnings.append("No speech engine: cut edges were checked for silence only.")
    results = []
    for index, request in enumerate(requests):
        take = takes[request.take]
        silent = in_cautious_silence(take, request.nominal)
        at_end = not request.start and request.nominal == request.duration
        if request.exact or not request.expected or at_end:
            how = "time" if request.exact else "picture_end" if at_end else "unconfirmed"
            results.append(
                EdgeResult(
                    request.segment, request.start, request.nominal, request.nominal,
                    how, not silent if request.exact else True,
                )
            )
            continue
        chosen: EdgeResult | None = None
        dips: list[tuple[float, int, Fraction, str]] = []
        for number, (point, how) in enumerate(options[index]):
            text = heard.get((index, number))
            if (
                text is None
                or not timed.get((index, number), False)
                or not _matches(heard_words(text), request.expected, request.start)
            ):
                continue
            if how == "silence":
                chosen = EdgeResult(
                    request.segment, request.start, point, request.nominal, how, False, text
                )
                break
            dips.append((float(take.level[min(_hop(point), len(take.level) - 1)]), number,
                         point, text))
        if chosen is None and dips:
            # Several dips can pass when the engine misses a clipped fragment at one of them;
            # the quietest is the real word boundary and the least audible join.
            _, _, point, text = min(dips)
            chosen = EdgeResult(
                request.segment, request.start, point, request.nominal, "quietest", True, text
            )
        if chosen is None:
            first_heard = heard.get((index, 0))
            chosen = EdgeResult(
                request.segment, request.start, request.nominal, request.nominal,
                "silence" if not listening and silent else "unconfirmed",
                listening or not silent, first_heard,
            )
        results.append(chosen)
    return results, warnings
