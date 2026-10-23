"""Story edges settle in real silence and keep the words they promise."""

from __future__ import annotations

import re
import wave
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from yap_studio import ffmpeg
from yap_studio.errors import DependencyError, InputError, ProcessingError
from yap_studio.project.model import Word as SourceWord
from yap_studio.stages import edges
from yap_studio.transcript import ScriptedBackend
from yap_studio.transcript.model import Transcript, Word

RATE = edges.RATE

# (start, end, word, level): "alright" runs into "if" through a dip, then a real pause.
BURSTS = [
    (0.50, 0.90, "alright", 0.3),
    (0.95, 1.30, "if", 0.3),
    (1.60, 2.00, "you", 0.3),
    (2.40, 2.80, "ever", 0.3),
]


def test_voiced_words_trims_long_silent_edges_and_preserves_uncertain_spans() -> None:
    level = np.full(240, -80.0)
    level[30:60] = -40.0
    level[100:130] = -40.0
    level[170:200] = -40.0
    pause = edges.PauseThreshold(-20.0, "words", -60.0, 0.0, 12)
    take = edges._Take(np.empty(0), level, pause.threshold_db, pause)
    words = (
        SourceWord("t1:0", "trailing", 0.3, 1.0),
        SourceWord("t1:1", "leading", 0.7, 1.3),
        SourceWord("t1:2", "short", 1.7, 2.1),
        SourceWord("t1:3", "unheard", 2.1, 2.35),
    )

    assert take.cautious_threshold == -45.0
    result = edges.voiced_words(take, words)

    assert level[59] == -40.0 < take.threshold
    assert result[0].start == 0.3
    assert result[0].end == pytest.approx(0.65)
    assert result[1].start == pytest.approx(0.95)
    assert result[1].end == 1.3
    assert result[2] == words[2]
    assert result[3] == words[3]
    assert all(old.start <= new.start < new.end <= old.end
               for old, new in zip(words, result, strict=True))


def test_voiced_words_uses_main_voice_across_long_inside_silence() -> None:
    level = np.full(300, -80.0)
    level[30:60] = -20.0
    level[270:272] = -20.0
    word = SourceWord("t1:0", "word.", 0.3, 3.0)
    take = edges._Take(np.empty(0), level, -45.0)

    result, = edges.voiced_words(take, (word,))
    assert result.start == 0.3
    assert result.end == pytest.approx(0.65)

    level[80:100] = -20.0
    level[270:272] = -80.0
    result, = edges.voiced_words(take, (word,))
    assert result.start == 0.3
    assert result.end == pytest.approx(1.05)


@pytest.mark.parametrize("bounds,runs,expected", [
    pytest.param((0.3, 3.0), ((30, 60), (270, 272)), (0.3, 0.65), id="first-anchored"),
    pytest.param((0.3, 3.0), ((58, 60), (270, 300)), (2.65, 3.0), id="last-anchored"),
    pytest.param((0.3, 3.0), ((30, 50), (280, 300)), (0.3, 3.0), id="both-anchored"),
    pytest.param((0.3, 1.2), ((30, 50), (114, 117)), (0.3, 0.55),
                 id="both-anchored-first-dominant"),
    pytest.param((0.3, 1.2), ((30, 33), (97, 117)), (0.92, 1.2),
                 id="both-anchored-last-dominant"),
    pytest.param((0.3, 3.0), ((30, 50), (267, 297)), (0.3, 3.0),
                 id="both-anchored-not-dominant"),
    pytest.param((0.3, 1.2), ((30, 50), (107, 117)), (0.3, 0.55),
                 id="both-anchored-first-at-owner-ratio"),
    pytest.param((0.3, 1.2), ((30, 40), (97, 117)), (0.92, 1.2),
                 id="both-anchored-last-at-owner-ratio"),
    pytest.param((0.0, 3.0), ((30, 60), (270, 272)), (0.25, 2.77), id="neither-anchored"),
    pytest.param((0.3, 2.0), ((30, 50), (140, 170)), (0.3, 0.55), id="longer-next-word"),
    pytest.param((0.3, 2.0), ((30, 60), (84, 114)), (0.3, 1.19), id="short-inside-gap"),
    pytest.param((0.3, 2.0), ((30, 60), (85, 116)), (0.3, 0.65), id="minimum-inside-gap"),
    pytest.param((0.0, 3.0), ((25, 55), (270, 272)), (0.2, 2.77), id="start-margin-at-minimum"),
    pytest.param((0.3, 3.0), ((58, 60), (245, 275)), (0.53, 2.8), id="end-margin-at-minimum"),
])
def test_voiced_words_uses_only_unambiguous_edge_anchored_runs(
    bounds: tuple[float, float], runs: tuple[tuple[int, int], ...],
    expected: tuple[float, float],
) -> None:
    level = np.full(300, -80.0)
    for first, last in runs:
        level[first:last] = -20.0
    take = edges._Take(np.empty(0), level, -45.0)
    result, = edges.voiced_words(take, (SourceWord("t1:0", "hello", *bounds),))
    assert (result.start, result.end) == pytest.approx(expected)


def test_voiced_words_uses_quiet_floor_when_word_midpoints_lower_threshold() -> None:
    level = np.full(120, -60.0)
    level[30:60] = -25.0
    pause = edges.PauseThreshold(-62.5, "words", -67.5, 0.0, 12)
    take = edges._Take(np.empty(0), level, pause.threshold_db, pause)
    word = SourceWord("t1:0", "tank.", 0.3, 1.1)

    assert take.cautious_threshold == -62.5
    refined, = edges.voiced_words(take, (word,))
    assert refined.start == word.start
    assert refined.end == pytest.approx(0.65)


@pytest.mark.parametrize("source,heard", [
    ("one.", "one."), ("one.", "one"), ("猫", "猫"), ("café", "cafe\u0301"),
    ("café-one", "cafe\u0301-1"),
])
def test_silent_tail_word_realigns_from_exact_local_transcript(
    source: str, heard: str,
) -> None:
    sound = np.zeros(4 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.5),
             SourceWord("t1:1", source, 1.8, 3.8))
    backend = ScriptedBackend(lambda _: Transcript([
        Word("wanted", 0.1, 0.95), Word(heard, 1.0, 1.22),
    ]))

    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], backend, take_id="t1",
    )
    assert not warnings
    assert corrected[1].id == "t1:1"
    assert (corrected[0].start, corrected[0].end) == pytest.approx((0.5, 1.35))
    assert (corrected[1].start, corrected[1].end) == pytest.approx((1.4, 1.62))
    assert [row["word"] for row in evidence] == ["t1:0", "t1:1"]
    assert corrected[0].end <= corrected[1].start
    assert all("heard" not in row for row in evidence)


@pytest.mark.parametrize("source,heard", [
    ("猫", "犬"), ("猫one", "犬one"), ("cafe\u0301", "cafe"),
    ("İ", "i"), (".", "!"), ("...", "???"),
    ("café-one", "caféone"), ("café one", "caféone"),
])
def test_silent_tail_rejects_local_text_mismatch(source: str, heard: str) -> None:
    sound = np.zeros(4 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.5),
             SourceWord("t1:1", source, 1.8, 3.8))
    backend = ScriptedBackend(lambda _: Transcript([
        Word("wanted", 0.1, 0.95), Word(heard, 1.0, 1.22),
    ]))

    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], backend, take_id="t1",
    )
    assert corrected == words
    assert evidence == []
    assert any("local text or order mismatch" in warning for warning in warnings)


def test_silent_tail_rejects_exact_target_mostly_in_silence() -> None:
    sound = np.zeros(4 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.42 * RATE)] = 0.2
    sound[round(1.53 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.5),
             SourceWord("t1:1", "one.", 1.8, 3.8))
    backend = ScriptedBackend(lambda _: Transcript([
        Word("wanted", 0.1, 0.9), Word("one.", 0.96, 1.26),
    ]))

    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], backend, take_id="t1",
    )
    assert corrected == words
    assert evidence == []
    assert any("voice" in warning for warning in warnings)


def test_silent_tail_selects_last_seven_by_source_order_with_overlapping_times() -> None:
    sound = np.zeros(16 * RATE, dtype=np.float32)
    sound[round(9.84 * RATE):round(11.7 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t2:17", "and", 9.6, 9.9),
             SourceWord("t2:18", "ever", 9.84, 10.16),
             SourceWord("t2:19", "since", 10.16, 10.4),
             SourceWord("t2:20", "then", 10.4, 10.56),
             SourceWord("t2:21", "I've", 10.72, 10.96),
             SourceWord("t2:22", "always", 10.96, 11.28),
             SourceWord("t2:23", "wanted", 11.28, 11.84),
             SourceWord("t2:24", "one.", 11.84, 14.64))
    backend = ScriptedBackend(lambda _: Transcript([
        Word(item.text, 0.17 + index * 0.22, 0.17 + (index + 1) * 0.22)
        for index, item in enumerate(words[1:])
    ]))

    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], backend, take_id="t2",
    )
    assert not warnings
    assert [row["word"] for row in evidence] == [item.id for item in words[1:]]
    assert evidence[0]["clip"][0] == pytest.approx(9.74)
    assert corrected[0] == words[0]
    assert evidence[-2]["original"] == [11.28, 11.84]
    assert evidence[-1]["original"] == [11.84, 14.64]
    assert corrected[-2].end <= corrected[-1].start
    assert corrected[-1].end < 11.853


@pytest.mark.parametrize("tail_end,accepted", [(1.74, True), (1.77, False)])
def test_silent_tail_final_end_only_clamp(tail_end: float, accepted: bool) -> None:
    sound = np.zeros(4 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.5),
             SourceWord("t1:1", "one.", 1.8, 3.8))
    backend = ScriptedBackend(lambda _: Transcript([
        Word("wanted", 0.1, 0.95), Word("one.", 1.0, tail_end - 0.4),
    ]))
    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], backend, take_id="t1",
    )
    if accepted:
        assert not warnings
        assert corrected[1].end == pytest.approx(1.66)
        assert corrected[0].end <= corrected[1].start
        assert [row["word"] for row in evidence] == ["t1:0", "t1:1"]
    else:
        assert corrected == words
        assert not evidence
        assert warnings


def test_silent_tail_rejects_target_without_voice_from_start() -> None:
    sound = np.zeros(4 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.5),
             SourceWord("t1:1", "one.", 1.8, 3.8))
    backend = ScriptedBackend(lambda _: Transcript([
        Word("wanted", 0.1, 0.95), Word("one.", 1.26, 1.33),
    ]))
    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], backend, take_id="t1",
    )
    assert corrected == words
    assert not evidence
    assert warnings


def test_silent_tail_rejects_interacting_candidate_phrases() -> None:
    sound = np.zeros(5 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.2),
             SourceWord("t1:1", "one", 1.7, 2.8),
             SourceWord("t1:2", "today.", 1.9, 3.0))
    called = []
    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], lambda: called.append(True), take_id="t1",
    )
    assert corrected == words
    assert not evidence
    assert not called
    assert any("overlap" in warning or "interact" in warning for warning in warnings)


def test_silent_tail_rejects_overlapping_local_phrase() -> None:
    sound = np.zeros(4 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.5),
             SourceWord("t1:1", "one.", 1.8, 3.8))
    backend = ScriptedBackend(lambda _: Transcript([
        Word("wanted", 0.1, 1.1), Word("one.", 1.0, 1.22),
    ]))
    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, [], backend, take_id="t1",
    )
    assert corrected == words
    assert evidence == []
    assert any("overlap" in warning for warning in warnings)


@pytest.mark.parametrize("heard,removed,backend,expected_warning", [
    ("wrong", [], True, "text"),
    ("outside", [], True, "voice"),
    ("one.", [(1.4, 1.6)], True, "removal"),
    ("one.", [], False, "engine"),
])
def test_silent_tail_rejects_unsafe_evidence(
    heard: str, removed: list[tuple[float, float]], backend: bool, expected_warning: str,
) -> None:
    sound = np.zeros(4 * RATE, dtype=np.float32)
    sound[round(0.5 * RATE):round(1.65 * RATE)] = 0.2
    take = edges._Take(sound, edges._levels(sound), -45.0)
    words = (SourceWord("t1:0", "wanted", 0.5, 1.5),
             SourceWord("t1:1", "one.", 1.8, 3.8))
    transcript = Transcript([Word("wanted", 0.1, 0.9),
                             Word("wrong" if heard == "wrong" else "one.",
                                  1.6 if heard == "outside" else 1.0,
                                  1.8 if heard == "outside" else 1.2)])
    listener = ScriptedBackend(lambda _: transcript) if backend else None
    corrected, evidence, warnings = edges.realign_silent_words(
        take, words, removed, listener, take_id="t1",
    )
    assert corrected == words
    assert evidence == []
    assert any(expected_warning in warning.lower() for warning in warnings)


def test_silent_tail_candidate_limit_keeps_all_original_spans() -> None:
    sound = np.zeros(20 * RATE, dtype=np.float32)
    words = []
    for index in range(9):
        base = 2 * index
        sound[round((base + 0.2) * RATE):round((base + 0.8) * RATE)] = 0.2
        words.extend((SourceWord(f"t1:{2 * index}", "wanted", base + 0.2, base + 0.7),
                      SourceWord(f"t1:{2 * index + 1}", "one.", base + 0.9, base + 1.95)))
    take = edges._Take(sound, edges._levels(sound), -45.0)
    called = []
    original = tuple(words)
    corrected, evidence, warnings = edges.realign_silent_words(
        take, original, [], lambda: called.append(True), take_id="t1",
    )
    assert corrected == original
    assert evidence == []
    assert not called
    assert any("too many" in warning for warning in warnings)


def _take() -> edges._Take:
    sound = np.full(round(4.0 * RATE), 1e-4, dtype=np.float32)
    rng = np.random.default_rng(7)
    for first, last, _, level in BURSTS:
        index = slice(round(first * RATE), round(last * RATE))
        sound[index] = level * rng.standard_normal(index.stop - index.start)
    # The dip between "alright" and "if" is quieter than speech but not silent.
    sound[round(0.90 * RATE) : round(0.95 * RATE)] = 0.01 * rng.standard_normal(
        round(0.05 * RATE)
    )
    level = edges._levels(sound)
    floor = float(np.percentile(level, 10))
    return edges._Take(
        sound, level, min(floor + edges.SILENCE_ABOVE_FLOOR_DB, edges.SILENCE_CEILING_DBFS)
    )


def _pause_track(floor: float, words: int = 12) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Speech bursts separated by pauses on a steady room noise bed."""
    level = np.full(words * 50, floor)
    spans = []
    for index in range(words):
        first = index * 50 + 15
        last = index * 50 + 35
        level[first:last] = floor + 20.2
        spans.append((first * edges.HOP_S, last * edges.HOP_S))
    return level, spans


def _noisy_tail_take() -> edges._Take:
    """Loud midpoints miss a soft word tail on a noisy bed, then a real quiet pause."""
    level, spans = _pause_track(-38.0)
    level[35:45] = -25.0
    level[55:65] = -70.0
    pause = edges.choose_pause_threshold(level, [(first + last) / 2 for first, last in spans])
    sound = np.repeat(10 ** (level / 20), round(edges.HOP_S * RATE)).astype(np.float32)
    return edges._Take(sound, level, pause.threshold_db, pause)


def test_pause_threshold_tracks_each_takes_noise_and_keeps_every_word() -> None:
    thresholds = []
    for floor in (-65.0, -38.0):
        level, spans = _pause_track(floor)
        midpoints = [(first + last) / 2 for first, last in spans]
        pause = edges.choose_pause_threshold(level, midpoints)
        take = edges._Take(np.empty(0), level, pause.threshold_db, pause)
        assert pause.rule == "words"
        assert pause.words == len(spans) >= edges.PAUSE_MIN_WORDS
        assert floor < pause.threshold_db < floor + 20.2
        assert pause.floor_db == floor
        assert pause.baseline_share == 0.0
        assert pause.share is not None and pause.share <= edges.PAUSE_SHARE_MAX
        assert all(not edges.in_silence(take, Fraction(str(point))) for point in midpoints)
        assert all(edges.in_silence(take, Fraction(str(last + 0.05))) for _, last in spans)
        # The next higher candidate would put every word in a long silence run.
        higher = replace(take, threshold=pause.threshold_db + edges.PAUSE_SCAN_STEP_DB)
        assert all(edges.in_silence(higher, Fraction(str(point))) for point in midpoints)
        thresholds.append(pause.threshold_db)
        if floor == -38.0:
            old = replace(take, threshold=edges.SILENCE_CEILING_DBFS)
            assert not edges._runs(old, 0, len(level))
            assert edges._runs(take, 0, len(level))
    assert thresholds[0] != thresholds[1]


@pytest.mark.parametrize("floor", [-65.0, -38.0])
@pytest.mark.parametrize("words", [0, edges.PAUSE_MIN_WORDS - 1])
def test_pause_threshold_with_too_few_words_uses_exact_floor_rule(
    floor: float, words: int,
) -> None:
    level, spans = _pause_track(floor)
    midpoints = [(first + last) / 2 for first, last in spans[:words]]
    # These timings cannot qualify the take for the word rule.
    midpoints.extend([-0.001, len(level) * edges.HOP_S, 100.0])
    pause = edges.choose_pause_threshold(level, midpoints)
    assert pause.threshold_db == min(
        floor + edges.SILENCE_ABOVE_FLOOR_DB, edges.SILENCE_CEILING_DBFS,
    )
    assert pause.rule == "floor"
    assert pause.floor_db == floor
    assert pause.share is None
    assert pause.baseline_share is None
    assert pause.words == words


def test_pause_threshold_empty_track_uses_floor_rule() -> None:
    pause = edges.choose_pause_threshold(np.empty(0), [0.0] * edges.PAUSE_MIN_WORDS)
    assert pause == edges.PauseThreshold(-45.0, "floor", -45.0, None, 0)


def test_pause_threshold_with_unreliable_baseline_measures_share_at_floor_rule() -> None:
    level, spans = _pause_track(-65.0, 25)
    midpoints = [(first + last) / 2 for first, last in spans]
    # Six loose timings lie in pauses; two quiet words are above the baseline level
    # but below the fallback threshold, so the receipt must distinguish the shares.
    midpoints[:6] = [index * 0.5 for index in range(6)]
    level[6 * 50 + 15:6 * 50 + 35] = -55.0
    level[7 * 50 + 15:7 * 50 + 35] = -55.0
    pause = edges.choose_pause_threshold(level, midpoints)
    assert pause.threshold_db == -50.0
    assert pause.rule == "floor"
    assert pause.baseline_share == 0.24 > edges.PAUSE_BASELINE_MAX
    assert pause.share == 0.32
    assert pause.words == 25


@pytest.mark.parametrize("quiet_words", [2, 5], ids=["eight-percent", "baseline-limit"])
def test_pause_threshold_tolerates_midpoints_already_in_true_silence(quiet_words: int) -> None:
    level, spans = _pause_track(-62.2, 25)
    midpoints = [(first + last) / 2 for first, last in spans]
    midpoints[:quiet_words] = [index * 0.5 for index in range(quiet_words)]
    pause = edges.choose_pause_threshold(level, midpoints)
    assert pause.rule == "words"
    assert pause.floor_db < pause.threshold_db
    assert pause.baseline_share == pytest.approx(quiet_words / 25)
    assert pause.share is not None and pause.baseline_share is not None
    assert pause.share - pause.baseline_share <= edges.PAUSE_SHARE_MAX
    take = edges._Take(np.empty(0), level, pause.threshold_db)
    assert sum(edges.in_silence(take, Fraction(str(point))) for point in midpoints) == quiet_words
    assert all(not edges.in_silence(take, Fraction(str(point)))
               for point in midpoints[quiet_words:])


@pytest.mark.parametrize("words,threshold,silent_words", [(50, -18.0, 2), (49, -28.5, 1)])
def test_pause_threshold_allows_only_the_small_share_above_baseline(
    words: int, threshold: float, silent_words: int,
) -> None:
    level, spans = _pause_track(-38.0, words)
    # One midpoint is always in a pause; one quieter word uses the additional allowance.
    midpoints = [(first + last) / 2 for first, last in spans]
    midpoints[0] = 0.0
    level[65:85] = -28.2
    pause = edges.choose_pause_threshold(level, midpoints)
    assert pause.rule == "words"
    assert pause.baseline_share == pytest.approx(1 / words)
    assert pause.share == pytest.approx(silent_words / words)
    assert pause.threshold_db == threshold


def test_pause_threshold_keeps_exact_allowance_despite_float_subtraction_rounding() -> None:
    level, spans = _pause_track(-38.0, 100)
    midpoints = [(first + last) / 2 for first, last in spans]
    midpoints[:8] = [index * 0.5 for index in range(8)]
    level[8 * 50 + 15:8 * 50 + 35] = -28.2
    level[9 * 50 + 15:9 * 50 + 35] = -28.2
    pause = edges.choose_pause_threshold(level, midpoints)
    assert pause.rule == "words"
    assert pause.baseline_share == 0.08
    assert pause.share == 0.10
    assert pause.threshold_db == -18.0


def test_pause_threshold_scans_exact_floor_bound_between_step_levels() -> None:
    level, spans = _pause_track(-62.2, 25)
    for first, last in spans:
        level[round(first / edges.HOP_S):round(last / edges.HOP_S)] = -59.1
    pause = edges.choose_pause_threshold(level, [(first + last) / 2 for first, last in spans])
    assert pause.rule == "words"
    assert pause.threshold_db == pytest.approx(-59.2)
    assert pause.baseline_share == pause.share == 0.0
    take = edges._Take(np.empty(0), level, pause.threshold_db, pause)
    assert take.cautious_threshold == take.threshold


@pytest.mark.parametrize("dip_hops,threshold", [(3, -12.0), (4, -55.0)])
def test_pause_threshold_applies_minimum_run_length_at_word_midpoints(
    dip_hops: int, threshold: float,
) -> None:
    level = np.full(150, -12.0)
    level[:20] = -60.0
    level[75:75 + dip_hops] = -55.0
    midpoints = [index * edges.HOP_S for index in range(25, 125, 10)]
    pause = edges.choose_pause_threshold(level, midpoints)
    assert pause.rule == "words"
    assert pause.threshold_db == threshold
    assert pause.share == 0.0
    take = edges._Take(np.empty(0), level, pause.threshold_db)
    assert take.cautious_threshold == take.threshold
    assert take.silent(75) is (dip_hops == 3)
    assert not edges.in_silence(take, Fraction("0.75"))
    assert not edges.in_cautious_silence(take, Fraction("0.75"))
    higher = replace(take, threshold=threshold + edges.PAUSE_SCAN_STEP_DB)
    assert edges.in_silence(higher, Fraction("0.75"))
    assert edges.in_cautious_silence(higher, Fraction("0.75"))


@pytest.mark.parametrize("dip_hops,silent", [(3, False), (4, True)])
def test_cautious_silence_requires_a_full_quiet_run(dip_hops: int, silent: bool) -> None:
    take = _noisy_tail_take()
    take.level[55:65] = -38.0
    take.level[55:55 + dip_hops] = -70.0
    assert edges.in_silence(take, Fraction("0.55"))
    assert edges.in_cautious_silence(take, Fraction("0.55")) is silent


@pytest.mark.parametrize("with_words", [False, True])
def test_load_take_chooses_pause_threshold_from_source_word_spans(
    monkeypatch: pytest.MonkeyPatch, with_words: bool,
) -> None:
    level, spans = _pause_track(-38.0)
    sound = np.repeat(10 ** (level / 20), round(edges.HOP_S * RATE)).astype(np.float32)

    def decode(source: Path, *, stream: int | None, sr: int, mono: bool):
        assert source == Path("take.mov") and stream == 2 and sr == RATE and mono
        return RATE, sound

    monkeypatch.setattr(edges.audio, "decode", decode)
    if with_words:
        take = edges.load_take(Path("take.mov"), 2, words=spans)
        assert take.pause is not None and take.pause.rule == "words"
        assert take.pause.words == len(spans)
        assert take.threshold > -38.0
        assert all(not edges.in_silence(take, Fraction(str((first + last) / 2)))
                   for first, last in spans)
    else:
        take = edges.load_take(Path("take.mov"), 2)
        floor = float(np.percentile(take.level, 10))
        assert take.threshold == min(
            floor + edges.SILENCE_ABOVE_FLOOR_DB, edges.SILENCE_CEILING_DBFS,
        )
        assert take.pause == edges.PauseThreshold(take.threshold, "floor", floor, None, 0)
    assert take.pause is not None and take.threshold == take.pause.threshold_db
    assert take.cautious_threshold == min(
        take.threshold, take.pause.floor_db + edges.SILENCE_ABOVE_FLOOR_DB,
        edges.SILENCE_CEILING_DBFS,
    )


def test_pause_threshold_receipt_has_readable_values() -> None:
    pause = edges.PauseThreshold(-32.25, "words", -38.246, 1 / 51, 51, 1 / 103)
    assert pause.to_dict() == {
        "threshold_db": -32.2, "rule": "words", "floor_db": -38.2, "share": 0.0196, "words": 51,
        "baseline_share": 0.0097,
    }
    assert replace(pause, rule="floor", share=None).to_dict()["share"] is None
    assert replace(pause, baseline_share=None).to_dict()["baseline_share"] is None


class _Listener:
    """Hear the words whose bursts mostly lie inside each candidate clip."""

    name = "listener"
    model = "listener"

    def __init__(
        self, take: edges._Take, requests: Sequence[edges.EdgeRequest], invent: str = ""
    ) -> None:
        self.invent = invent
        self.points = {
            index: edges.candidates(take, request) for index, request in enumerate(requests)
        }
        self.requests = requests

    def version(self) -> str | None:
        return "0"

    def download_notice(self) -> str | None:
        return None

    def transcribe_many(
        self, paths: Sequence[Path], *, language: str | None = None
    ) -> dict[Path, Transcript]:
        del language
        heard = {}
        for path in paths:
            match = re.fullmatch(r"e(\d+)-(\d+)\.wav", path.name)
            assert match
            index, number = int(match[1]), int(match[2])
            point = float(self.points[index][number][0])
            start = self.requests[index].start
            first, last = (point, point + edges.CLIP_S) if start else (point - edges.CLIP_S, point)
            # Clip time: start clips open with the pad; end clips begin CLIP_S before the cut.
            origin = point - edges.CLIP_PAD_S if start else max(0.0, point - edges.CLIP_S)
            words = [
                Word(text, a - origin, b - origin)
                for a, b, text, _ in BURSTS
                if (min(b, last) - max(a, first)) / (b - a) > 0.5
            ]
            if self.invent and not start and words:
                # Finish the phrase inside the silent pad, the way an engine can.
                cut = last - origin
                words.append(Word(self.invent, cut + 0.05, cut + 0.2))
            heard[path] = Transcript(words)
        return heard


def _request(start: bool, nominal: float, expected: tuple[str, ...]) -> edges.EdgeRequest:
    return edges.EdgeRequest("s1", "t1", start, Fraction(str(nominal)), expected)


def test_early_word_time_moves_past_the_dropped_word() -> None:
    take = _take()
    # The engine said "if" starts at 0.60, inside "alright"; nearest silence is before "alright".
    requests = [_request(True, 0.60, ("if", "you"))]
    results, warnings = edges.place(requests, {"t1": take}, _Listener(take, requests))
    assert warnings == []
    result = results[0]
    assert result.how == "quietest"
    assert result.listen
    assert Fraction("0.88") <= result.point <= Fraction("0.97")


def test_end_edge_lands_in_the_pause_and_needs_no_listen() -> None:
    take = _take()
    # "if" really ends at 1.30; the engine stretched it to 1.62, into "you".
    requests = [_request(False, 1.62, ("alright", "if"))]
    results, _ = edges.place(requests, {"t1": take}, _Listener(take, requests))
    result = results[0]
    assert result.how == "silence"
    assert not result.listen
    assert Fraction("1.30") <= result.point <= Fraction("1.55")


def test_correct_silent_edge_stays_put() -> None:
    take = _take()
    requests = [_request(True, 1.48, ("you", "ever"))]
    results, _ = edges.place(requests, {"t1": take}, _Listener(take, requests))
    assert results[0].point == Fraction("1.48")
    assert results[0].how == "silence"
    assert not results[0].listen


def test_unconfirmable_edge_keeps_word_time_and_asks_for_a_listen() -> None:
    take = _take()
    requests = [_request(True, 0.60, ("welcome", "to"))]
    results, _ = edges.place(requests, {"t1": take}, _Listener(take, requests))
    assert results[0].point == Fraction("0.60")
    assert results[0].how == "unconfirmed"
    assert results[0].listen


def test_without_an_engine_only_silence_is_checked() -> None:
    take = _take()
    requests = [_request(True, 1.48, ("you",)), _request(True, 0.60, ("if",))]
    results, warnings = edges.place(requests, {"t1": take}, None)
    assert [result.listen for result in results] == [False, True]
    assert [result.point for result in results] == [Fraction("1.48"), Fraction("0.60")]
    assert [result.how for result in results] == ["silence", "unconfirmed"]
    assert warnings == ["No speech engine: cut edges were checked for silence only."]


@pytest.mark.parametrize("engine_failed", [False, True])
def test_unconfirmed_end_in_soft_tail_requires_a_listen(engine_failed: bool) -> None:
    take = _noisy_tail_take()
    requests = [
        edges.EdgeRequest(
            "s1", "t1", False, Fraction("0.40"), ("hello",),
            word_start=Fraction("0.15"), word_end=Fraction("0.40"),
        ),
        _request(False, 0.60, ("hello",)),
    ]
    assert take.threshold > -25.0
    assert take.cautious_threshold == edges.SILENCE_CEILING_DBFS
    assert edges.in_silence(take, requests[0].nominal)
    assert not edges.in_cautious_silence(take, requests[0].nominal)
    assert edges.in_cautious_silence(take, requests[1].nominal)
    backend = _Failing(DependencyError.missing_engine("speech")) if engine_failed else None
    results, _ = edges.place(requests, {"t1": take}, backend)
    assert [result.point for result in results] == [request.nominal for request in requests]
    assert [result.how for result in results] == ["unconfirmed", "silence"]
    assert [result.listen for result in results] == [True, False]


def test_exact_time_in_soft_tail_requires_a_listen() -> None:
    take = _noisy_tail_take()
    requests = [
        edges.EdgeRequest("s1", "t1", False, Fraction("0.40"), (), exact=True),
        edges.EdgeRequest("s2", "t1", False, Fraction("0.60"), (), exact=True),
    ]
    results, _ = edges.place(requests, {"t1": take}, None)
    assert [(result.point, result.how, result.listen) for result in results] == [
        (Fraction("0.40"), "time", True),
        (Fraction("0.60"), "time", False),
    ]


def test_engine_confirmed_edge_keeps_calibrated_silence_placement() -> None:
    take = _noisy_tail_take()
    request = _request(False, 0.40, ("hello",))
    point, how = edges.candidates(take, request)[0]
    assert how == "silence"
    assert edges.in_silence(take, point)
    assert not edges.in_cautious_silence(take, point)
    backend = ScriptedBackend(lambda _: Transcript([Word("hello", 0.15, 0.45)]))
    results, warnings = edges.place([request], {"t1": take}, backend)
    assert warnings == []
    assert results == [
        edges.EdgeResult("s1", False, point, request.nominal, "silence", False, "hello"),
    ]


def test_exact_time_edges_are_kept_and_flagged_only_in_sound() -> None:
    take = _take()
    requests = [
        edges.EdgeRequest("s1", "t1", False, Fraction("1.45"), (), exact=True),
        edges.EdgeRequest("s2", "t1", True, Fraction("1.10"), (), exact=True),
    ]
    results, _ = edges.place(requests, {"t1": take}, None)
    assert [(result.point, result.how, result.listen) for result in results] == [
        (Fraction("1.45"), "time", False),
        (Fraction("1.10"), "time", True),
    ]


@pytest.mark.parametrize("word_end", ["2.75", "2.753"])
def test_picture_end_does_not_move_inside_the_last_word(word_end: str) -> None:
    take = _take()
    take.sound[round(2.5 * RATE) : round(2.6 * RATE)] = 1e-4
    take = replace(take, level=edges._levels(take.sound))
    request = edges.EdgeRequest(
        "ending", "t1", False, Fraction("2.75"), ("ever",),
        word_start=Fraction("2.4"), word_end=Fraction(word_end), duration=Fraction("2.75"),
    )
    backend = ScriptedBackend(lambda _: Transcript([Word("ever", 0.1, 0.3)]))
    results, _ = edges.place([request], {"t1": take}, backend)
    assert results[0].point == request.duration


def test_stretched_last_word_drops_the_dead_air() -> None:
    take = _take()
    # The engine stretched "ever" (really 2.40 to 2.80) over the pause to 3.80.
    request = edges.EdgeRequest(
        "s1", "t1", False, Fraction("3.80"), ("you", "ever"),
        word_start=Fraction("2.40"), word_end=Fraction("3.80"),
    )
    results, _ = edges.place([request], {"t1": take}, _Listener(take, [request]))
    assert results[0].how == "silence"
    assert not results[0].listen
    assert Fraction("2.95") <= results[0].point <= Fraction("3.05")
    # Without an engine the nominal point stays put, even inside the same pause.
    alone, _ = edges.place([request], {"t1": take}, None)
    assert alone[0].point == request.nominal
    assert alone[0].how == "silence" and not alone[0].listen


@pytest.mark.parametrize("start,heard", [(True, "um hello"), (False, "hello um")])
def test_boundary_filler_cannot_confirm_a_different_anchor(start: bool, heard: str) -> None:
    assert not edges._matches(edges.heard_words(heard), ("hello",), start)


@pytest.mark.parametrize("start,heard", [(True, "um hello"), (False, "hello um")])
def test_authored_filler_anchor_can_be_confirmed(start: bool, heard: str) -> None:
    words = edges.heard_words(heard)
    assert edges._matches(words, ("um",), start)
    assert edges._matches(words, ("um", "hello") if start else ("hello", "um"), start)


@pytest.mark.parametrize("start,heard,expected", [
    (True, "hello um there", ("hello", "there")),
    (False, "there um hello", ("there", "hello")),
    (True, "if you um care", ("if", "you")),
    (False, "care um you if", ("you", "if")),
])
def test_filler_away_from_boundary_match_does_not_block_confirmation(
    start: bool, heard: str, expected: tuple[str, ...],
) -> None:
    assert edges._matches(edges.heard_words(heard), expected, start)


@pytest.mark.parametrize("start,heard", [(True, "if um you"), (False, "you um if")])
def test_short_boundary_word_cannot_skip_a_filler_neighbour(start: bool, heard: str) -> None:
    expected = ("if", "you") if start else ("you", "if")
    assert not edges._matches(edges.heard_words(heard), expected, start)


def test_words_in_another_script_still_count_at_the_cut() -> None:
    assert edges.heard_words("Ара, if you've") == ["?", "if", "you've"]
    assert not edges._matches(edges.heard_words("Ара, if you've"), ("if", "you've"), True)
    assert edges._matches(edges.heard_words("Плетлин да I created."), ("i", "created"), False)


def test_word_finished_inside_the_pad_does_not_confirm_an_edge() -> None:
    take = _take()
    # Cutting at 1.45 drops "you"; an engine that invents "you" in the pad must not pass it.
    request = edges.EdgeRequest(
        "s1", "t1", False, Fraction("1.45"), ("if", "you"),
        word_start=Fraction("1.60"), word_end=Fraction("2.00"),
    )
    results, _ = edges.place([request], {"t1": take}, _Listener(take, [request], invent="you"))
    assert results[0].point >= Fraction("2.00") or results[0].listen


class _Fixed:
    """Hear the same words from every clip cut inside "you", nothing elsewhere."""

    name = model = "fixed"

    def __init__(self, take: edges._Take, request: edges.EdgeRequest) -> None:
        self.points = edges.candidates(take, request)

    def version(self) -> str | None:
        return "0"

    def download_notice(self) -> str | None:
        return None

    def transcribe_many(
        self, paths: Sequence[Path], *, language: str | None = None
    ) -> dict[Path, Transcript]:
        del language
        heard = {}
        for path in paths:
            point = self.points[int(path.stem.split("-")[1])][0]
            inside = Fraction("1.6") < point < Fraction("2.0")
            heard[path] = Transcript([Word("if", 1.0, 1.3)] if inside else [])
        return heard


def test_quietest_confirmed_dip_wins() -> None:
    take = _take()
    # Two dips inside "you": a shallow one at 1.65 s and a deep one at 1.90 s.
    take.level[165] = take.threshold + 20
    take.level[190] = take.threshold + 2
    request = edges.EdgeRequest(
        "s1", "t1", False, Fraction("1.75"), ("if",), word_start=Fraction("1.6"),
        word_end=Fraction("1.75"),
    )
    results, _ = edges.place([request], {"t1": take}, _Fixed(take, request))
    assert results[0].how == "quietest" and results[0].listen
    assert Fraction("1.85") <= results[0].point <= Fraction("1.95")


def test_end_word_stamped_just_before_the_cut_is_refused() -> None:
    late = Transcript([Word("here", 2.0, 2.3), Word("go.", 2.95, 3.1)])
    early = Transcript([Word("here", 2.0, 2.3), Word("go.", 2.6, 2.85)])
    assert not edges._inside(late, 3.0, False)
    assert edges._inside(early, 3.0, False)


def test_end_word_with_exactly_the_minimum_lead_is_inside() -> None:
    assert edges._inside(Transcript([Word("go", 2.9, 3.0)]), 3.0, False)


def test_end_timing_guard_uses_the_last_filler() -> None:
    late = Transcript([Word("hello", 2.0, 2.3), Word("um", 2.95, 3.1)])
    assert not edges._inside(late, 3.0, False)
    assert edges._inside(Transcript([Word("um", 2.6, 2.85)]), 3.0, False)
    assert edges._inside(Transcript([Word("um", 0.3, 0.5)]), 0.3, True)


@pytest.mark.parametrize("start,point", [(False, 0.4), (True, 3.9)])
def test_short_clip_keeps_its_sound_and_pad_on_the_correct_clock(
    tmp_path: Path, start: bool, point: float,
) -> None:
    take = _take()
    sound = edges._clip(take, Fraction(str(point)), start)
    pad = round(edges.CLIP_PAD_S * RATE)
    index = round(point * RATE)
    expected = take.sound[index:] if start else take.sound[:index]
    assert len(sound) == pad + len(expected)
    np.testing.assert_array_equal(sound[pad:] if start else sound[:-pad], expected)
    assert not np.any(sound[:pad] if start else sound[-pad:])
    path = tmp_path / "short.wav"
    edges._write_clip(path, sound)
    with wave.open(str(path), "rb") as clip:
        assert clip.getnframes() == len(sound)
        assert clip.getframerate() == RATE
    cut = edges.CLIP_PAD_S if start else len(expected) / RATE
    spoken = Transcript([Word("word", 0.1, 0.3)])
    assert edges._inside(spoken, cut, start)


@pytest.mark.parametrize("start,nominal,other", [(True, 1.2, 1.35), (False, 1.4, 1.25)])
def test_candidates_cannot_cross_the_segments_other_edge(
    start: bool, nominal: float, other: float,
) -> None:
    request = replace(_request(start, nominal, ("if",)), other=Fraction(str(other)))
    found = edges.candidates(_take(), request)
    assert found
    assert all(point < request.other if start else point > request.other for point, _ in found)


def test_candidates_stay_within_picture_duration_even_if_audio_is_longer() -> None:
    request = replace(_request(False, 2.8, ("ever",)), duration=Fraction("2.9"))
    found = edges.candidates(_take(), request)
    assert found
    assert all(point <= request.duration for point, _ in found)
    assert not edges._containing(_take(), request)


def test_an_edge_without_matchable_words_is_unconfirmed_even_in_silence() -> None:
    request = _request(True, 1.48, ())
    results, _ = edges.place([request], {"t1": _take()}, None)
    assert results[0].point == request.nominal
    assert results[0].how == "unconfirmed" and results[0].listen


class _Failing(_Listener):
    """Fail at the backend boundary without involving audio or file helpers."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def transcribe_many(
        self, paths: Sequence[Path], *, language: str | None = None,
    ) -> dict[Path, Transcript]:
        raise self.error


@pytest.mark.parametrize("error", [
    ProcessingError("speech", 1, "failed", code="speech_engine_failed"),
    DependencyError.missing_engine("speech"),
])
def test_engine_failure_degrades_to_silence_only_with_a_warning(error: Exception) -> None:
    requests = [_request(True, 1.48, ("you",)), _request(True, 0.6, ("if",))]
    results, warnings = edges.place(requests, {"t1": _take()}, _Failing(error))
    alone, _ = edges.place(requests, {"t1": _take()}, None)
    assert results == alone
    assert len(warnings) == 1 and "without listening" in warnings[0]


@pytest.mark.parametrize("engine_failed", [False, True])
def test_late_start_in_silence_keeps_word_time_without_confirmation(engine_failed: bool) -> None:
    sound = np.full(2 * RATE, 1e-4, dtype=np.float32)
    sound[RATE : round(1.2 * RATE)] = 0.3
    sound[round(1.7 * RATE) :] = 0.3
    take = edges._Take(sound, edges._levels(sound), edges.SILENCE_CEILING_DBFS)
    # The word was spoken at 1.00 to 1.20, but its transcript start is in the next pause.
    request = _request(True, 1.30, ("hello",))
    assert edges._containing(take, request)[0] > request.nominal
    backend = _Failing(DependencyError.missing_engine("speech")) if engine_failed else None
    results, _ = edges.place([request], {"t1": take}, backend)
    assert results[0].point == request.nominal
    assert results[0].how == "silence" and not results[0].listen


@pytest.mark.parametrize("error", [
    InputError("source", "missing"), ProcessingError("ffmpeg", 1, "failed"),
    OSError("disk"), RuntimeError("programming error"),
])
def test_non_engine_errors_propagate(error: Exception) -> None:
    with pytest.raises(type(error)) as caught:
        edges.place([_request(True, 1.48, ("you",))], {"t1": _take()}, _Failing(error))
    assert caught.value is error


def test_clip_write_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    take = _take()
    requests = [_request(True, 1.48, ("you",))]

    def fail(path: Path, sound: np.ndarray) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(edges, "_write_clip", fail)
    with pytest.raises(OSError, match="disk full"):
        edges.place(requests, {"t1": take}, _Listener(take, requests))


class _Missing(_Listener):
    """Return no results despite completing the batch call."""

    def transcribe_many(
        self, paths: Sequence[Path], *, language: str | None = None,
    ) -> dict[Path, Transcript]:
        return {}


def test_missing_transcripts_cannot_confirm_a_silent_edge() -> None:
    take = _take()
    requests = [_request(True, 1.48, ("you",))]
    results, warnings = edges.place(requests, {"t1": take}, _Missing(take, requests))
    assert results[0].point == requests[0].nominal
    assert results[0].how == "unconfirmed" and results[0].listen
    assert len(warnings) == 1 and "no transcript" in warnings[0]


def test_delayed_selected_audio_is_measured_on_container_time(tmp_path: Path) -> None:
    source = tmp_path / "streams.mov"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=size=32x32:rate=30:duration=2", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=2", "-itsoffset", "0.4",
        "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=1",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
        "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(source),
    ], timeout_s=60)
    take = edges.load_take(source, 1)
    assert edges.in_silence(take, Fraction("0.2"))
    assert not edges.in_silence(take, Fraction("0.6"))
    assert not edges.in_silence(edges.load_take(source, 0), Fraction("0.2"))
    requests = [_request(True, 0.2, ("word",))]
    results, _ = edges.place(requests, {"t1": take}, None)
    assert results[0].point == requests[0].nominal
    assert results[0].how == "silence"
    assert not results[0].listen
