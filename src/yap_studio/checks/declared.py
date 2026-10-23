"""Check joins declared on the output clock by the words heard across them.

A declared join names the words that must play in a row across an output
frame, plus phrases that must be gone from the render or heard exactly once.
Only the render is needed: no source picture or sound. Each join is heard in
two windows that end 3.5 s after it, with 5 s and 9 s of lead-in, so a word
recognised only with more context cannot pass or fail the join alone. The words
must also run across the join itself: the ``before`` words end by the join
frame and the ``after`` words start from it, each within ``NEAR_S`` of it, with
``EDGE_S`` of slack for word timing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from yap_studio import audio
from yap_studio.checks import joins
from yap_studio.checks.model import Finding, format_time
from yap_studio.edl.model import JoinAssertion
from yap_studio.paths import validate_output_dir
from yap_studio.transcript import OPTIONAL, Backend, Transcript, norm_words

LEAD_INS_S = (5.0, 9.0)
AFTER_S = 3.5
# Word timings from a speech engine are coarse: allow this much overlap across the join.
EDGE_S = 0.3
# The words either side must sit this close to the join, so a long silence cannot pass.
NEAR_S = 1.5
# joins._write_wav puts one second of silence before the audio.
_PAD_S = 1.0


@dataclass(frozen=True)
class HeardWindow:
    """One transcribed window around a declared join."""

    lead_in_s: float
    start_s: float
    end_s: float
    heard: str
    contiguous: bool
    in_order: bool = False
    heard_at_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class PhraseCount:
    """How often a phrase is heard in the whole render."""

    phrase: str
    words: list[str]
    count: int
    times_s: list[float]


@dataclass(frozen=True)
class DeclaredJoinResult:
    """The words heard across one declared join."""

    index: int
    out_frame: int
    time_s: float
    expected: list[str]
    status: Literal["pass", "fail"]
    windows: list[HeardWindow]
    absent: list[PhraseCount]
    once: list[PhraseCount]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class DeclaredResult:
    """Results and findings for every declared join."""

    joins: list[DeclaredJoinResult]
    findings: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _words(text: str) -> list[str]:
    return [word for word in norm_words(text) if word not in OPTIONAL]


def _tokens(transcript: Transcript) -> list[tuple[str, float]]:
    """Return normalised words with their start on the render clock."""
    return [(token, start) for token, start, _ in _timed(transcript, 0.0)]


def _timed(transcript: Transcript, offset_s: float) -> list[tuple[str, float, float]]:
    """Return normalised words with their start and end on the render clock.

    ``offset_s`` is the render time where the transcribed clip starts, before the
    WAV padding.
    """
    return [
        (token, offset_s + word.start - _PAD_S, offset_s + word.end - _PAD_S)
        for word in transcript.words
        for token in norm_words(word.text)
        if token not in OPTIONAL
    ]


def _across(
    tokens: Sequence[tuple[str, float, float]], index: int, before: int, after: int, join_s: float
) -> bool:
    """Return whether the matched words at ``index`` run across the join."""
    if before:
        end = tokens[index + before - 1][2]
        if not join_s - NEAR_S <= end <= join_s + EDGE_S:
            return False
    if after:
        start = tokens[index + before][1]
        if not join_s - EDGE_S <= start <= join_s + NEAR_S:
            return False
    return True


def _occurrences(tokens: Sequence[str], phrase: Sequence[str]) -> list[int]:
    if not phrase:
        return []
    size = len(phrase)
    return [
        index
        for index in range(len(tokens) - size + 1)
        if list(tokens[index : index + size]) == list(phrase)
    ]


def _count(phrase: str, tokens: Sequence[tuple[str, float]]) -> PhraseCount:
    words = _words(phrase)
    positions = _occurrences([token for token, _ in tokens], words)
    return PhraseCount(
        phrase=phrase,
        words=words,
        count=len(positions),
        times_s=[max(0.0, tokens[index][1]) for index in positions],
    )


def _quoted(words: Sequence[str]) -> str:
    return '"' + " ".join(words) + '"'


def check_declared_joins(
    render: Path,
    assertions: Sequence[JoinAssertion],
    *,
    backend: Backend,
    fps: Fraction | float,
    work_dir: Path,
    video_offset_s: float = 0.0,
) -> DeclaredResult:
    """Transcribe every declared join and the whole render in one batch."""
    if not assertions:
        return DeclaredResult([], [])
    directory = validate_output_dir(work_dir)
    sr, sound = audio.decode(render)
    rate = float(fps)
    windows: dict[int, list[tuple[float, float, float, Path]]] = {}
    paths: list[Path] = []
    for assertion in assertions:
        join_s = video_offset_s + assertion.out_frame / rate
        rows = []
        for lead in LEAD_INS_S:
            start, end = join_s - lead, join_s + AFTER_S
            path = directory / f"declared-join-{assertion.index:02d}-lead-{round(lead)}.wav"
            joins._write_wav(path, joins._slice(sound, sr, start, end), sr)
            rows.append((lead, start, end, path))
            paths.append(path)
        windows[assertion.index] = rows
    whole = None
    if any(assertion.absent or assertion.once for assertion in assertions):
        whole = directory / "declared-render.wav"
        joins._write_wav(whole, sound, sr)
        paths.append(whole)
    transcripts = backend.transcribe_many(paths)
    tokens = _tokens(transcripts[whole]) if whole is not None else []

    results: list[DeclaredJoinResult] = []
    findings: list[Finding] = []
    for assertion in assertions:
        join_s = video_offset_s + assertion.out_frame / rate
        before, after = _words(assertion.before), _words(assertion.after)
        expected = before + after
        heard_windows = []
        for lead, start, end, path in windows[assertion.index]:
            timed = _timed(transcripts[path], start)
            heard = [token for token, _, _ in timed]
            found = _occurrences(heard, expected)
            heard_windows.append(
                HeardWindow(
                    lead_in_s=lead,
                    start_s=start,
                    end_s=end,
                    heard=" ".join(heard),
                    contiguous=any(
                        _across(timed, index, len(before), len(after), join_s) for index in found
                    ),
                    in_order=bool(found),
                    heard_at_s=tuple(timed[index][1] for index in found),
                )
            )
        absent = [_count(phrase, tokens) for phrase in assertion.absent]
        once = [_count(phrase, tokens) for phrase in assertion.once]
        contiguous = all(window.contiguous for window in heard_windows)
        passed = (
            contiguous
            and all(row.count == 0 for row in absent)
            and all(row.count == 1 for row in once)
        )
        result = DeclaredJoinResult(
            index=assertion.index,
            out_frame=assertion.out_frame,
            time_s=join_s,
            expected=expected,
            status="pass" if passed else "fail",
            windows=heard_windows,
            absent=absent,
            once=once,
        )
        results.append(result)
        evidence = {
            "source": "declared",
            "declared_join": assertion.index,
            "out_frame": assertion.out_frame,
            "time_s": join_s,
            "before": assertion.before,
            "after": assertion.after,
            "expected": expected,
            "windows": [asdict(window) for window in heard_windows],
            "absent": [asdict(row) for row in absent],
            "once": [asdict(row) for row in once],
        }
        where = f"At {format_time(join_s)}, declared join {assertion.index}"
        if contiguous:
            findings.append(
                Finding(
                    "join.words",
                    "info",
                    f"{where}: {_quoted(expected)} is heard across the join, as declared.",
                    time_s=join_s,
                    evidence=evidence,
                )
            )
        else:
            heard = "; ".join(
                f"{window.lead_in_s:g} s lead-in: \"{window.heard}\""
                + (
                    " (in a row at "
                    + ", ".join(format_time(time) for time in window.heard_at_s)
                    + ", away from the join)"
                    if window.in_order and not window.contiguous
                    else ""
                )
                for window in heard_windows
            )
            action = "Listen to this join and re-render it so the declared words play across it."
            findings.append(
                Finding(
                    "join.words",
                    "fail",
                    f"{where}: {_quoted(expected)} is not heard across the join. "
                    f"Heard {heard}.",
                    time_s=join_s,
                    suggested_action=action,
                    evidence=evidence,
                )
            )
        for row in absent:
            if row.count:
                times = ", ".join(format_time(time) for time in row.times_s)
                action = "Remove these words from the edit and re-render."
                findings.append(
                    Finding(
                        "join.words",
                        "fail",
                        f"{where}: {_quoted(row.words)} should be cut but is heard at {times}.",
                        time_s=join_s,
                        suggested_action=action,
                        evidence=evidence,
                    )
                )
        for row in once:
            if row.count != 1:
                where_heard = (
                    f"is heard {row.count} times, at "
                    + ", ".join(format_time(time) for time in row.times_s)
                    if row.count
                    else "is not heard"
                )
                action = "Check for a repeated or missing line near this join and re-render."
                findings.append(
                    Finding(
                        "join.words",
                        "fail",
                        f"{where}: {_quoted(row.words)} {where_heard}; the edit list says once.",
                        time_s=join_s,
                        suggested_action=action,
                        evidence=evidence,
                    )
                )
    return DeclaredResult(results, findings)
