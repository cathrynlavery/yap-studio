"""Compare a cut's render and plan readings, checking controls for doubled words."""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from yap_studio.checks.model import Finding, format_time
from yap_studio.transcript import (
    OPTIONAL,
    Transcript,
    Word,
    compare_words,
    norm_words,
    second_engine_unmeasured_reason,
    words_between,
)

Status = Literal["pass", "warn", "skipped"]
_PAD_S = 1.0  # The existing trial WAVs start with one second of silence.
DOUBLE_NEAR_S = 0.3
PAIR_MATCH_S = 0.3


@dataclass(frozen=True)
class JoinReading:
    """One engine's normalised planned and heard words for the same short span."""

    engine: str
    planned: list[str]
    heard: list[str]
    diff: list[str]
    agreement: bool
    plan_heard: list[str]
    plan_diff: list[str]
    plan_agreement: bool
    heard_words: list[Word]
    planned_words: list[Word] = field(default_factory=list)


@dataclass(frozen=True)
class RelistenResult:
    """Warning verdicts and both readings, with a span on the render clock."""

    status: Status
    reason: str | None
    first: JoinReading
    second: JoinReading | None
    second_status: Status
    second_reason: str | None
    span: dict[str, float]
    doubled_words: list[dict] = field(default_factory=list)
    second_comparison: dict | None = None


def second_model_reason(engine: str) -> str:
    """Explain how to add an engine from another model family."""
    return second_engine_unmeasured_reason(engine)


def _timed_words(
    transcript: Transcript,
    start: float,
    end: float,
    clock_offset_s: float,
    *,
    rule: Literal["midpoint", "overlap"] = "overlap",
) -> list[Word]:
    """Normalise trial words, remove padding, and map their times to the render clock."""
    selected = (
        words_between(transcript, start + _PAD_S, end + _PAD_S)
        if rule == "midpoint" else transcript.words
    )
    mapped = [
        Word(
            token,
            clock_offset_s + word.start - _PAD_S,
            clock_offset_s + word.end - _PAD_S,
            word.conf,
        )
        for word in selected
        for token in norm_words(word.text)
    ]
    if rule == "midpoint":
        return mapped
    # Select on the render clock so offset rounding does not include a boundary word.
    span_start, span_end = clock_offset_s + start, clock_offset_s + end
    return [word for word in mapped if word.start < span_end and word.end > span_start]


def _compare_span(
    first: Sequence[Word],
    second: Sequence[Word],
    span_start: float,
    span_end: float,
    *,
    join_start: float,
    join_end: float,
) -> dict[str, object]:
    """Compare timed words, excusing each changed word clipped away from the join.

    All times share a clock. Every changed word must extend past an outer span
    boundary without overlapping the join interval. Equal landings define a
    point, touched only by words spanning it. Keep excused diffs as evidence.
    """
    a = [word for word in first if word.text not in OPTIONAL]
    b = [word for word in second if word.text not in OPTIONAL]
    expected, got = [word.text for word in a], [word.text for word in b]
    matcher = difflib.SequenceMatcher(None, expected, got)
    operations = [operation for operation in matcher.get_opcodes() if operation[0] != "equal"]
    agreement = True
    for _, i1, i2, j1, j2 in operations:
        changed = a[i1:i2] + b[j1:j2]
        if not all(
            (word.start < span_start or word.end > span_end)
            and not (word.start < join_end and word.end > join_start)
            for word in changed
        ):
            agreement = False
    return {
        "diff": [
            f"{operation} exp{expected[i1:i2]} got{got[j1:j2]}"
            for operation, i1, i2, j1, j2 in operations
        ],
        "agreement": agreement,
    }


def _reading(
    transcripts: Sequence[Transcript],
    engine: str,
    left_s: float,
    right_s: float,
    start_s: float,
    end_s: float,
    window_start_s: float,
    j_end_s: float,
    j_start_s: float,
) -> JoinReading:
    render, plan, left, right = transcripts
    planned_words = _timed_words(
        left, 1.5 - left_s, 1.5, j_end_s - 1.5, rule="midpoint"
    ) + _timed_words(right, 1.0, 1.0 + right_s, j_start_s - 1.0, rule="midpoint")
    planned = [word.text for word in planned_words]
    heard_words = _timed_words(render, start_s, end_s, window_start_s)
    heard = [word.text for word in heard_words]
    plan_words = _timed_words(plan, start_s, end_s, window_start_s)
    plan_heard = [word.text for word in plan_words]
    comparison = compare_words(planned, heard, fillers_optional=True)
    plan_comparison = _compare_span(
        plan_words,
        heard_words,
        window_start_s + start_s,
        window_start_s + end_s,
        join_start=j_end_s,
        join_end=j_start_s,
    )
    return JoinReading(
        engine,
        planned,
        heard,
        comparison["diff"],
        comparison["core_exact"],
        plan_heard,
        plan_comparison["diff"],
        plan_comparison["agreement"],
        heard_words,
        planned_words,
    )


def _doubled_at(reading: JoinReading, j_end_s: float, j_start_s: float) -> list[dict]:
    """Find adjacent equal words near both landings and record their planned match."""
    planned_pairs = list(zip(reading.planned_words, reading.planned_words[1:], strict=False))
    doubled = []
    for first, second in zip(reading.heard_words, reading.heard_words[1:], strict=False):
        if (
            first.text != second.text
            or abs(first.end - j_end_s) > DOUBLE_NEAR_S
            or abs(second.start - j_start_s) > DOUBLE_NEAR_S
        ):
            continue
        first_midpoint = (first.start + first.end) / 2
        second_midpoint = (second.start + second.end) / 2
        planned_pair = next(
            (
                {"first": asdict(left), "second": asdict(right)}
                for left, right in planned_pairs
                if left.text == first.text and right.text == second.text
                and abs((left.start + left.end) / 2 - first_midpoint) <= PAIR_MATCH_S
                and abs((right.start + right.end) / 2 - second_midpoint) <= PAIR_MATCH_S
            ),
            None,
        )
        doubled.append(
            {
                "word": first.text,
                "first": asdict(first),
                "second": asdict(second),
                "planned_pair": planned_pair,
            }
        )
    return doubled


def relisten_at(
    transcripts: Sequence[Transcript],
    *,
    engine: str,
    left_length_s: float,
    right_length_s: float,
    j_end_s: float,
    j_start_s: float,
    window_start_s: float,
    second_transcripts: Sequence[Transcript] | None = None,
    second_engine: str | None = None,
    second_unmeasured_reason: str | None = None,
    words_failed: bool = False,
) -> RelistenResult:
    """Read existing padded trials without crossing a neighbouring planned cut.

    Source controls select a word by its midpoint on the kept side of each
    edge. Render words include anything inserted between the two landings.
    Control differences are evidence only, except for a new doubled word across
    the cut. Render and plan readings select words overlapping the same span
    and map them to the render clock. Each changed word is tolerated only if
    it extends past an outer boundary without touching the join between the
    two landings. Equal landings define a point. The second engine is compared
    within half a second of each landing, with both wider readings retained
    as evidence.
    """
    left_s, right_s = min(1.5, left_length_s), min(1.5, right_length_s)
    start_s, end_s = j_end_s - left_s, j_start_s + right_s
    first = _reading(
        transcripts,
        engine,
        left_s,
        right_s,
        start_s - window_start_s,
        end_s - window_start_s,
        window_start_s,
        j_end_s,
        j_start_s,
    )
    doubled = _doubled_at(first, j_end_s, j_start_s)
    status: Status = "pass"
    reason = None
    if not first.plan_agreement and not words_failed:
        status, reason = "warn", "render words differ from the edit's own audio"
    elif any(pair["planned_pair"] is None for pair in doubled):
        status, reason = "warn", "a word is doubled across the cut that the source does not repeat"
    if not first.planned and not first.heard and not first.plan_heard:
        status, reason = "skipped", "no speech near the cut"
    second = None
    second_status: Status = "skipped"
    second_reason = second_unmeasured_reason or second_model_reason(engine)
    second_comparison = None
    if second_transcripts is not None and second_engine is not None:
        second = _reading(
            second_transcripts,
            second_engine,
            left_s,
            right_s,
            start_s - window_start_s,
            end_s - window_start_s,
            window_start_s,
            j_end_s,
            j_start_s,
        )
        narrow_start, narrow_end = j_end_s - 0.5, j_start_s + 0.5
        narrow_first_words = _timed_words(
            transcripts[0],
            narrow_start - window_start_s,
            narrow_end - window_start_s,
            window_start_s,
        )
        narrow_second_words = _timed_words(
            second_transcripts[0],
            narrow_start - window_start_s,
            narrow_end - window_start_s,
            window_start_s,
        )
        narrow_first = [word.text for word in narrow_first_words]
        narrow_second = [word.text for word in narrow_second_words]
        comparison = _compare_span(
            narrow_first_words,
            narrow_second_words,
            narrow_start,
            narrow_end,
            join_start=j_end_s,
            join_end=j_start_s,
        )
        second_comparison = {
            "span": {"start_s": narrow_start, "end_s": narrow_end},
            "first": narrow_first,
            "second": narrow_second,
            "diff": comparison["diff"],
            "agreement": comparison["agreement"],
        }
        second_status = "pass" if comparison["agreement"] else "warn"
        second_reason = None if comparison["agreement"] else "speech engines hear different words"
        if not narrow_first and not narrow_second:
            second_status, second_reason = "skipped", "no speech near the cut"
    return RelistenResult(
        status,
        reason,
        first,
        second,
        second_status,
        second_reason,
        {"start_s": start_s, "end_s": end_s},
        doubled,
        second_comparison,
    )


def relisten_findings(result: RelistenResult, index: int, time_s: float) -> list[Finding]:
    """Make listening warnings from the readings, never pass or fail findings."""
    label = f"Cut {index} at {format_time(time_s)}"
    first, second = result.first, result.second
    evidence = asdict(result)
    findings = []
    if result.status == "warn":
        if result.reason == "render words differ from the edit's own audio":
            message = (
                f'{label}: heard on its own, the render says "{" ".join(first.heard)}", '
                f'but the edit\'s own audio says "{" ".join(first.plan_heard)}".'
            )
        else:
            message = (
                f'{label}: heard on its own, this join says "{" ".join(first.heard)}", '
                f'but the edit plans "{" ".join(first.planned)}".'
            )
        findings.append(
            Finding(
                "join.relisten",
                "warn",
                message,
                cut=index,
                time_s=time_s,
                suggested_action=(
                    "Listen to the join; if a syllable or word is doubled, clipped or added "
                    "at the cut, move the cut and render again."
                ),
                evidence=evidence,
            )
        )
    if result.second_status == "warn" and second is not None:
        comparison = result.second_comparison
        findings.append(
            Finding(
                "join.second_model",
                "warn",
                f"{label}: two speech engines hear this join differently: "
                f'{first.engine} hears "{" ".join(comparison["first"])}", '
                f'{second.engine} hears "{" ".join(comparison["second"])}".',
                cut=index,
                time_s=time_s,
                suggested_action="Listen to the join and decide which reading is right.",
                evidence={
                    **evidence,
                    "diff": comparison["diff"],
                },
            )
        )
    return findings
