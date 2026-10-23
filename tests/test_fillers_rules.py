"""Synthetic source words verify filler rules without decoding any audio."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from yap_studio.project.model import ProjectFileError, Word, Words
from yap_studio.stages import edges
from yap_studio.stages import fillers as stage
from yap_studio.stages.fillers import (
    Candidate,
    DetectResult,
    _PlannedAudio,
    _RefinedSpan,
    candidate_id,
    candidates_for_project,
    detect,
)


def _words(text: str, *, take: str = "t1") -> tuple[Word, ...]:
    """Place each word on exact frames with short gaps between words."""
    return tuple(Word(f"{take}:{i}", token, (3 + 12 * i) / 30, (12 + 12 * i) / 30)
                 for i, token in enumerate(text.split()))


def _indexes(result: DetectResult) -> set[int]:
    return {index for candidate in result.candidates for index in candidate.plan["word_indexes"]}


def _lexical(result: DetectResult) -> tuple[Candidate, ...]:
    return tuple(candidate for candidate in result.candidates if candidate.word_ids)


@pytest.mark.parametrize("token", ["um", "uh", "uhm", "umm", "erm", "hmm", "mm", "basically",
                                   "(UM!)", "\u201cUh\u201d"])
def test_always_removed_fillers(token: str) -> None:
    words = _words(f"before {token} after")
    result = detect(words)
    assert _indexes(result) == {1}
    assert _lexical(result)[0].word_ids == ("t1:1",)
    assert _lexical(result)[0].kind == "filler"
    assert _lexical(result)[0].reasons


@pytest.mark.parametrize(("text", "removed"), [
    ("I feel like staying", set()),
    ("it's like, you know", {0, 1, 2, 3}),
    ("you know what happens", set()),
    ("well you know things happen", {1, 2}),
    ("don't you know things", set()),
    ("if you know something", set()),
    ("well you know her", set()),
    ("I was like, I can't", set()),
    ("I was like I can't", set()),
    ("I was just basically um like I can't", {3, 4}),
    ("that is like wow", set()),
    ("sounds like that.", set()),
    ("moving like this!", set()),
    ("something like enough", set()),
    ("I would like to stay", set()),
    ("I'd like some water", set()),
    ("I\u2019d like some water", set()),
    ("would like perhaps leave", {1}),
    ("just like that", set()),
    ("just like water flows", {0, 1}),
    ("it's just like I think", {0, 1, 2}),
    ("it's just like that", set()),
    ("kind of like that", set()),
    ("we kind of like swimming", set()),
    ("she sort of like swimming", set()),
    ("perhaps kind of like swimming", {1, 2, 3}),
    ("perhaps sort of like swimming", {1, 2, 3}),
    ("it's like I think", {0, 1}),
    ("it's like, a river", {1}),
    ("well like things happen", {1}),
])
def test_like_and_you_know_rules(text: str, removed: set[int]) -> None:
    assert _indexes(detect(_words(text))) == removed


@pytest.mark.parametrize(("text", "removed"), [
    ("I I think", {0}),
    ("I I'm thinking", {0}),
    ("we we're thinking", {0}),
    ("we we we think", {0, 1}),
    ("very very good", set()),
    ("really really good", set()),
    ("no no thanks", set()),
    ("yeah yeah sure", set()),
    ("bye bye", set()),
    ("ha ha", set()),
    ("na na", set()),
    ("so so", set()),
    ("go. go again", set()),
    ("go Go again", set()),
    ("go [Go] again", set()),
    ("\u201cgo!\u201d go again", set()),
])
def test_stutters_and_sentence_guards(text: str, removed: set[int]) -> None:
    result = detect(_words(text))
    assert _indexes(result) == removed
    assert all(candidate.kind == "stutter" for candidate in _lexical(result))


@pytest.mark.parametrize(("gap", "removed"), [(0.2, True), (0.4, False), (0.5, False)])
@pytest.mark.parametrize("phrase", ["So.", "And so"])
def test_sentence_initial_so_uses_the_preceding_gap(gap: float, removed: bool, phrase: str) -> None:
    words = (Word("t1:0", 'done."', 0.0, 0.3), *_words(f"{phrase} next"))
    words = (words[0], *(replace(word, id=f"t1:{i + 1}",
                                start=round(word.start + 0.2 + gap, 10),
                                end=round(word.end + 0.2 + gap, 10))
                         for i, word in enumerate(words[1:])))
    result = detect(words)
    expected = set(range(1, 1 + len(phrase.split()))) if removed else set()
    assert _indexes(result) == expected


@pytest.mark.parametrize("following", ["much", "many", "far", "long", "that", "good", "bad",
                                        "cool"])
def test_sentence_initial_so_keeps_qualifiers(following: str) -> None:
    assert not _lexical(detect(_words(f"done. So {following}")))


def test_so_inside_sentence_or_at_segment_start_is_kept() -> None:
    assert not _lexical(detect(_words("So we begin")))
    assert not _lexical(detect(_words("done so we begin")))
    words = _words("done. So we begin")
    words = tuple(replace(word, start=word.start + (0.2 if i else 0),
                          end=word.end + (0.2 if i else 0)) for i, word in enumerate(words))
    assert _indexes(detect(words, beat_breath_s=0.25)) == set()


@pytest.mark.parametrize(("text", "removed"), [
    ("we should leave um uh we should leave soon", {0, 1, 2, 3, 4}),
    ("we should leave we should leave soon", {0, 1, 2}),
    ("leave soon um leave soon please", {0, 1, 2}),
    ("we should leave um uh like basically we should leave", set(range(7))),
    ("we should leave um uh like basically hmm we should leave", {3, 4, 5, 6, 7}),
    ("we should leave because we should leave", set()),
    ("we should leave. we should leave", set()),
    ("we should leave um. we should leave", {3}),
    ("the a and the a and", set()),
    ("we should leave we should leave.", {0, 1, 2}),
])
def test_restarts_fodder_distance_stop_words_and_sentences(text: str, removed: set[int]) -> None:
    result = detect(_words(text))
    assert _indexes(result) == removed
    if 0 in removed:
        assert _lexical(result)[0].kind == "restart"


@pytest.mark.parametrize("text", ["very, very good", "really really good",
                                 "make a cake, make a pie"])
def test_emphasis_and_fluent_parallel_phrases_are_kept(text: str) -> None:
    assert not _lexical(detect(_words(text), max_pause_s=10))


def test_fragment_restart_removes_the_first_matching_attempt() -> None:
    words = _words("build a gard, build a garden plan")
    candidate, = detect(words, max_pause_s=10).candidates
    assert candidate.kind == "restart"
    assert candidate.word_ids == ("t1:0", "t1:1", "t1:2")
    assert candidate.start == words[0].start
    assert candidate.end == words[3].start
    assert "fragment restart" in candidate.reasons


@pytest.mark.parametrize("text", ["a gar the garden", "a g a garden",
                                 "a garden a gar", "a gard. a garden"])
def test_fragment_restart_requires_context_length_and_sentence(text: str) -> None:
    assert not _lexical(detect(_words(text), max_pause_s=10))


def test_fragment_restart_has_a_two_second_limit() -> None:
    words = _words("a gard a garden")
    words = (*words[:2], *(replace(word, start=word.start + 2, end=word.end + 2)
                          for word in words[2:]))
    assert not _lexical(detect(words, max_pause_s=10))


@pytest.mark.parametrize(("gap", "removed"), [(0.5, True), (2.0, True), (2.1, False)])
def test_repeated_ngram_has_a_two_second_limit(gap: float, removed: bool) -> None:
    words = _words("we should leave um we should leave soon")
    shift = words[2].end + gap - words[4].start
    words = (*words[:4], *(replace(word, start=round(word.start + shift, 10),
                                  end=round(word.end + shift, 10)) for word in words[4:]))
    restarts = [candidate for candidate in detect(words).candidates if candidate.kind == "restart"]
    assert {index for candidate in restarts for index in candidate.plan["word_indexes"]} == (
        {0, 1, 2, 3} if removed else set()
    )


def test_abandoned_word_requires_a_refined_span() -> None:
    assert not _lexical(detect(_words("bring a basket bring a blanket"), max_pause_s=10))


@pytest.mark.parametrize(("text", "size"), [
    ("bring a basket bring a blanket", 2),
    ("carry the small basket carry the small blanket", 3),
])
def test_hidden_restart_requires_both_copies_inside_one_refined_span(text: str, size: int) -> None:
    words = _words(text)
    refined = _RefinedSpan(words[0].start, words[-1].end, words)
    result = detect(words, refined=(refined,), max_pause_s=10)
    candidate, = result.candidates
    assert candidate.kind == "restart"
    assert candidate.word_ids == tuple(word.id for word in words[:size + 1])
    assert candidate.start == words[0].start
    assert candidate.end == words[size + 1].start
    assert candidate.reasons == (f"hidden repeated {size}-gram",)
    assert not detect(words, refined=(replace(refined, start=words[1].start),),
                      max_pause_s=10).candidates
    assert not detect(words, refined=(replace(refined, end=words[2 * size].end - 0.1),),
                      max_pause_s=10).candidates


def test_hidden_restart_requires_exactly_one_abandoned_content_word() -> None:
    words = _words("bring a basket outside bring a blanket")
    refined = _RefinedSpan(words[0].start, words[-1].end, words)
    assert not detect(words, refined=(refined,), max_pause_s=10).candidates
    words = _words("bring a um bring a blanket")
    result = detect(words, refined=(_RefinedSpan(words[0].start, words[-1].end, words),),
                    max_pause_s=10)
    candidate, = result.candidates
    assert "repeated 2-gram" in candidate.reasons
    assert "hidden repeated 2-gram" not in candidate.reasons


@pytest.mark.parametrize(("voice_start", "left_end", "expected"), [
    (0.5, 0.3, 0.5),
    (0.2, 0.3, 0.35),
    (0.5, 0.6, 0.6),
    (0.76, 0.6, 0.75),
])
def test_refined_restart_backtracking_bounds_and_pause_target(
    voice_start: float, left_end: float, expected: float,
) -> None:
    """Respect both search limits while using audio to measure the retained pause."""
    words = (Word("t1:0", "before", 0.1, left_end), Word("", "build", 0.75, 0.85),
             Word("", "a", 0.9, 1), Word("", "basket", 1.05, 1.2),
             Word("", "build", 1.6, 1.8), Word("", "a", 1.9, 2),
             Word("", "garden", 2.1, 2.3))
    sound = np.zeros(3 * 48_000, dtype=np.float32)
    sound[round(voice_start * 48_000):48_000] = 0.05
    sound[round(1.6 * 48_000):round(1.8 * 48_000)] = 0.05
    segment = _PlannedAudio("t1", 0, 3, sound, words)
    refined = _RefinedSpan(0.7, 2.3, words[1:])
    candidate, = detect(words, refined=(refined,), restart_audio=segment,
                        max_pause_s=10).candidates
    assert candidate.plan["restart_audio"]["start_s"] == pytest.approx(expected)
    assert candidate.start >= left_end
    assert candidate.plan["want_pause_s"] == pytest.approx(min(expected - left_end, 0.2))
    assert candidate.plan["right_word_start_s"] == pytest.approx(1.6)


def test_refined_restart_recovers_voice_under_stretched_left_word() -> None:
    """Start at the first copy after a silent gap inside the left word's clock."""
    words = (Word("t1:0", "company", 0.1, 0.75), Word("", "make", 0.75, 0.85),
             Word("", "a", 0.9, 1.0), Word("", "comp", 1.05, 1.2),
             Word("", "make", 1.6, 1.8), Word("", "a", 1.9, 2.0),
             Word("", "brand", 2.1, 2.3))
    sound = np.zeros(3 * 48_000, dtype=np.float32)
    sound[round(0.1 * 48_000):round(0.4 * 48_000)] = 0.05
    sound[round(0.55 * 48_000):48_000] = 0.05
    sound[round(1.6 * 48_000):round(1.8 * 48_000)] = 0.05
    segment = _PlannedAudio("t1", 0, 3, sound, words)
    refined = _RefinedSpan(0.7, 2.3, words[1:])

    candidate, = detect(words, refined=(refined,), restart_audio=segment,
                        max_pause_s=10).candidates
    snapped = candidate.plan["restart_audio"]
    assert snapped["start_s"] == pytest.approx(0.55)
    assert 0.4 <= candidate.start <= 0.55
    assert snapped["left_end_s"] == pytest.approx(0.4, abs=0.01)


def test_refined_restart_keeps_second_syllable_of_left_word() -> None:
    """A short gap inside company must not look like a restart pause."""
    words = (Word("t1:0", "company", 0.1, 0.75), Word("", "make", 0.75, 1.2),
             Word("", "a", 1.25, 1.35), Word("", "comp", 1.4, 1.5),
             Word("", "make", 1.8, 2.0), Word("", "a", 2.1, 2.2),
             Word("", "brand", 2.3, 2.5))
    sound = np.zeros(3 * 48_000, dtype=np.float32)
    for first, last in ((0.1, 0.4), (0.5, 0.8), (0.9, 1.2), (1.8, 2.0)):
        sound[round(first * 48_000):round(last * 48_000)] = 0.05
    segment = _PlannedAudio("t1", 0, 3, sound, words)
    refined = _RefinedSpan(0.7, 2.5, words[1:])

    candidate, = detect(words, refined=(refined,), restart_audio=segment,
                        max_pause_s=10).candidates
    assert "left_end_s" not in candidate.plan["restart_audio"]
    assert candidate.start >= words[0].end


@pytest.mark.parametrize(("clip_start", "expected"), [
    (1.7, 1.6), (1.4, 1.6), (1.39, 1.39),
])
def test_refined_restart_second_voice_search_is_bounded(
    clip_start: float, expected: float,
) -> None:
    """Backtrack inside kept voice and look ahead only two tenths through quiet."""
    words = (Word("", "build", 0.5, 0.8), Word("", "a", 0.9, 1),
             Word("", "basket", 1.05, 1.2), Word("", "build", clip_start, 1.8),
             Word("", "a", 1.9, 2), Word("", "garden", 2.1, 2.3))
    sound = np.zeros(3 * 48_000, dtype=np.float32)
    sound[24_000:38_400] = 0.05
    sound[76_800:86_400] = 0.05
    segment = _PlannedAudio("t1", 0, 3, sound, words)
    candidate, = detect(words, refined=(_RefinedSpan(0, 3, words),),
                        restart_audio=segment, max_pause_s=10).candidates
    assert candidate.plan["restart_audio"]["end_s"] == pytest.approx(expected)
    assert candidate.end <= expected + 1e-9
    assert expected - candidate.end < 1 / 30


@pytest.mark.parametrize(("silence_db", "expected"), [(-45, 0.35), (-35, 0.5)])
def test_refined_restart_uses_the_stage_silence_floor(silence_db: float, expected: float) -> None:
    """Background below the chosen floor must not extend a restart's voice run."""
    words = (Word("", "build", 0.75, 0.85), Word("", "a", 0.9, 1),
             Word("", "basket", 1.05, 1.2), Word("", "build", 1.6, 1.8),
             Word("", "a", 1.9, 2), Word("", "garden", 2.1, 2.3))
    sound = np.zeros(3 * 48_000, dtype=np.float32)
    sound[:24_000] = 0.01
    sound[24_000:38_400] = 0.05
    sound[76_800:86_400] = 0.05
    segment = _PlannedAudio("t1", 0, 3, sound, words)
    candidate, = detect(words, refined=(_RefinedSpan(0, 3, words),),
                        restart_audio=segment, silence_db=silence_db, max_pause_s=10).candidates
    assert candidate.plan["restart_audio"]["start_s"] == pytest.approx(expected)
    assert candidate.plan["silence_db"] == silence_db


def test_source_fragment_boundaries_do_not_use_refinement_audio() -> None:
    """Keep original word placement when no refined clip supplied the restart."""
    words = _words("before build a gard build a garden plan")
    segment = _PlannedAudio("t1", 0, 4, np.full(4 * 48_000, 0.05, dtype=np.float32), words)
    assert detect(words, restart_audio=segment, max_pause_s=10) == detect(words, max_pause_s=10)


def test_overlapping_and_adjacent_detections_merge_all_reasons() -> None:
    result = detect(_words("before um uh after"))
    assert len(_lexical(result)) == 1
    assert _lexical(result)[0].word_ids == ("t1:1", "t1:2")
    assert set(_lexical(result)[0].reasons) == {"filler um", "filler uh"}
    restart = _lexical(detect(_words("leave soon um leave soon")))[0]
    assert restart.kind == "restart"
    assert "filler um" in restart.reasons
    assert "repeated 2-gram" in restart.reasons


@pytest.mark.parametrize("window", [(0.5, 0.8), (0.45, 0.46), (0.95, 0.96)])
def test_protect_windows_apply_to_widened_spans(window: tuple[float, float]) -> None:
    words = _words("before um after")
    assert not _lexical(detect(words, protect=(window,)))
    assert _lexical(detect(words, protect=((2.0, 3.0),)))


def test_protected_restart_is_dropped() -> None:
    assert not detect(_words("leave soon leave soon"), protect=((0.3, 0.4),)).candidates


@pytest.mark.parametrize("side", ["pre", "post"])
def test_boundary_keeps_beat_breath_whole(side: str) -> None:
    if side == "pre":
        words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "um", 0.8, 1.1),
                 Word("t1:2", "after", 1.2, 1.5))
        expected = (0.8, 1.2)
    else:
        words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "um", 0.4, 0.7),
                 Word("t1:2", "after", 1.2, 1.5))
        expected = (0.3, 0.7)
    candidate = _lexical(detect(words))[0]
    assert (candidate.start, candidate.end) == expected
    assert candidate.plan["kept_side"] == side
    assert candidate.plan["beat_breath"]
    assert candidate.plan["joined_pause_s"] == pytest.approx(0.5)
    assert any(item.kind == "silence" for item in detect(words).candidates)


def test_boundary_aims_at_join_target_when_both_sides_are_short() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "um", 0.6, 0.9),
             Word("t1:2", "after", 1.1, 1.4))
    candidate = _lexical(detect(words))[0]
    assert candidate.start == 0.5
    assert candidate.end == 1.1
    assert candidate.plan["joined_pause_s"] == pytest.approx(0.2)
    assert not candidate.listen
    assert candidate.plan["kept_side"] == "pre"
    short = _lexical(detect(_words("before um after")))[0]
    assert short.plan["joined_pause_s"] == pytest.approx(0.1)


def test_zero_gap_join_is_flagged_for_listening() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "um", 0.3, 0.6),
             Word("t1:2", "after", 0.6, 0.9))
    candidate = _lexical(detect(words))[0]
    assert candidate.listen
    assert candidate.plan["joined_pause_s"] == 0
    assert _lexical(detect(words[1:]))[0].listen


@pytest.mark.parametrize("voiced", [False, True])
def test_touching_stutter_at_real_source_times_is_kept_for_join_review(voiced: bool) -> None:
    words = (Word("t1:322", "I", 144.04, 144.28),
             Word("t1:323", "I", 144.28, 144.52),
             Word("t1:324", "need", 144.52, 144.76))
    result = detect(words, level_at=(lambda _: -20.0) if voiced else None)
    candidate, = result.candidates
    assert candidate.kind == "stutter"
    assert candidate.word_ids == ("t1:322",)
    assert candidate.start == 4321 / 30
    assert candidate.end == 4328 / 30
    assert candidate.listen
    assert not result.notes


@pytest.mark.parametrize("voiced", [False, True])
def test_filler_touching_both_neighbours_uses_nearest_frames(voiced: bool) -> None:
    words = (Word("t1:0", "before", 0.0, 0.315),
             Word("t1:1", "um", 0.315, 0.555),
             Word("t1:2", "after", 0.555, 0.8))
    result = detect(words, level_at=(lambda _: -20.0) if voiced else None)
    candidate, = result.candidates
    assert candidate.word_ids == ("t1:1",)
    assert candidate.start == 9 / 30
    assert candidate.end == 17 / 30
    assert abs(candidate.start - words[1].start) <= 1 / 60
    assert abs(candidate.end - words[1].end) <= 1 / 60
    assert candidate.listen
    assert not result.notes


def test_touching_boundary_half_frame_ties_snap_up() -> None:
    words = (Word("t1:0", "before", 0.0, 0.35), Word("t1:1", "um", 0.35, 0.55),
             Word("t1:2", "after", 0.55, 0.8))
    candidate, = detect(words).candidates
    assert candidate.start == 11 / 30
    assert candidate.end == 17 / 30
    assert candidate.listen


def test_touching_boundary_flags_listening_even_with_pause_on_other_side() -> None:
    words = (Word("t1:0", "before", 0.0, 0.315), Word("t1:1", "um", 0.315, 0.555),
             Word("t1:2", "after", 0.8, 1.1))
    candidate, = detect(words).candidates
    assert candidate.listen
    assert candidate.plan["joined_pause_s"] > 0


def test_only_empty_snapped_word_removal_keeps_boundary_note() -> None:
    words = (Word("t1:0", "before", 0.0, 0.301), Word("t1:1", "um", 0.301, 0.309),
             Word("t1:2", "after", 0.309, 0.6))
    result = detect(words)
    assert not result.candidates
    assert result.notes == ("word removal has no safe silent frame boundary",)


@pytest.mark.parametrize("end", [0.3, 0.5])
def test_long_pause_removes_the_centred_excess(end: float) -> None:
    words = (Word("t1:0", "before", 0.0, end),
             Word("t1:1", "after", round(end + 0.9, 10), round(end + 1.2, 10)))
    candidate, = detect(words).candidates
    assert candidate.kind == "silence"
    assert candidate.word_ids == ()
    assert candidate.end - candidate.start == pytest.approx(0.6)
    assert candidate.start - end == pytest.approx(0.15, abs=1 / 60 + 1e-9)
    assert words[1].start - candidate.end == pytest.approx(0.15, abs=1 / 60 + 1e-9)
    assert not candidate.listen
    assert not detect(words, protect=((candidate.start, candidate.end),)).candidates
    assert not detect(words, max_pause_s=0.9).candidates
    assert detect(words, max_pause_s=0.6).candidates[0].end - detect(
        words, max_pause_s=0.6).candidates[0].start == pytest.approx(0.3)


def test_measured_pause_removals_stay_inside_silent_runs() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3),
             Word("t1:1", "after", 1.3, 1.6))

    old, = detect(words, max_pause_s=0.2).candidates
    assert old.start < 0.6 < 1.0 < old.end

    def silent_runs(first: float, last: float) -> list[tuple[float, float]]:
        assert (first, last) == (0.3, 1.3)
        return [(0.3, 0.6), (1.0, 1.3)]

    measured = detect(words, max_pause_s=0.2, silent_runs=silent_runs).candidates
    assert len(measured) == 2
    assert all(any(first <= item.start < item.end <= last
                   for first, last in silent_runs(0.3, 1.3)) for item in measured)
    assert all(item.kind == "silence" and item.word_ids == () for item in measured)


def test_measured_pause_below_limit_proposes_nothing() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3),
             Word("t1:1", "after", 1.3, 1.6))
    assert detect(words, max_pause_s=0.3).candidates
    assert not detect(words, max_pause_s=0.3,
                      silent_runs=lambda first, last: [(0.5, 0.7)]).candidates


def test_measured_pause_drops_run_without_two_internal_frame_boundaries() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3),
             Word("t1:1", "after", 1.3, 1.6))
    assert not detect(words, max_pause_s=0,
                      silent_runs=lambda first, last: [(0.305, 0.345)]).candidates


def test_measured_silent_runs_uses_quiet_floor_when_words_rule_is_too_low() -> None:
    sound = np.full(round(1.6 * 48_000), 10 ** (-60 / 20), dtype=np.float32)
    sound[:round(0.3 * 48_000)] = 10 ** (-25 / 20)
    sound[round(1.3 * 48_000):] = 10 ** (-25 / 20)
    segment = _PlannedAudio(
        "t1", 0, 1.6, sound, (), silence_db=-62.5, pause={"floor_db": -67.5},
    )

    run, = stage._measured_silent_runs(segment)(0.3, 1.3)
    assert run == pytest.approx((0.3, 1.3))
    assert stage._measured_silent_runs(replace(segment, pause=None))(0.3, 1.3) == []


def test_measured_silent_runs_uses_floor_margin_for_noisy_gaps() -> None:
    assert [edges.pause_quiet_threshold_db(floor) for floor in (-67.5, -41.1, -38)] == (
        pytest.approx([-52.5, -38.1, -35])
    )
    sound = np.full(round(1.6 * 48_000), 10 ** (-40 / 20), dtype=np.float32)
    sound[:round(0.3 * 48_000)] = 10 ** (-30 / 20)
    sound[round(1.3 * 48_000):] = 10 ** (-30 / 20)
    segment = _PlannedAudio(
        "t1", 0, 1.6, sound, (), silence_db=-35.5, pause={"floor_db": -41.0},
    )

    silent_runs = stage._measured_silent_runs(segment)
    run, = silent_runs(0, 1.6)
    assert run == pytest.approx((0.3, 1.3))
    assert silent_runs(0, 0.3) == silent_runs(1.3, 1.6) == []


def test_touching_pause_and_filler_spans_merge() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "um", 0.8, 1.1),
             Word("t1:2", "after", 1.1, 1.4))
    candidate, = detect(words, max_pause_s=0).candidates
    assert candidate.start == 0.3
    assert candidate.end == 1.1
    assert candidate.listen
    assert candidate.kind == "filler"
    assert len(candidate.reasons) == 2


def test_non_english_yields_pauses_only_and_explains_scope() -> None:
    words = (Word("t1:0", "um", 0.0, 0.3), Word("t1:1", "uh", 1.2, 1.5))
    result = detect(words, language="fr")
    assert not result.english_rules
    assert result.notes == ("filler rules are English only; pauses only",)
    assert len(result.candidates) == 1
    assert result.candidates[0].kind == "silence"
    assert not result.candidates[0].word_ids
    assert detect(_words("um"), language="en-US").english_rules
    assert not detect(_words("um"), language="fr").candidates


def test_optional_level_callback_finds_silent_frames_without_entering_words() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "um", 0.6, 0.9),
             Word("t1:2", "after", 1.1, 1.4))
    sampled = []

    def level_at(point: float) -> float:
        sampled.append(point)
        return -60.0 if abs(point - 16 / 30) < 1e-9 or abs(point - 32 / 30) < 1e-9 else -20.0

    candidate, = detect(words, level_at=level_at).candidates
    assert candidate.start == 16 / 30
    assert candidate.end == 32 / 30
    assert all(0.3 <= point <= 0.6 or 0.9 <= point <= 1.1 for point in sampled)
    result = detect(words, level_at=lambda _: -40.0)
    assert _lexical(result)[0].listen
    assert _lexical(result)[0].word_ids == ("t1:1",)
    assert not result.notes
    assert detect(words, silence_db=-30.0, level_at=lambda _: -40.0).candidates


def test_pause_callback_and_frame_safety() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "after", 1.2, 1.5))
    voiced, = detect(words, level_at=lambda _: -20.0).candidates
    assert voiced.listen
    candidate, = detect(words, level_at=lambda _: -60.0).candidates
    assert 0.3 <= candidate.start < candidate.end <= 1.2
    assert candidate.start * 30 == pytest.approx(round(candidate.start * 30))
    assert candidate.end * 30 == pytest.approx(round(candidate.end * 30))
    narrow = (Word("t1:0", "before", 0.0, 0.301), Word("t1:1", "um", 0.302, 0.6),
              Word("t1:2", "after", 0.7, 1.0))
    result = detect(narrow)
    candidate, = result.candidates
    assert candidate.start == 0.3
    assert candidate.end == 0.6
    assert candidate.listen
    assert not result.notes


def test_all_silent_callback_preserves_half_frame_snap_ties() -> None:
    words = (Word("t1:0", "before", 0.0, 0.3), Word("t1:1", "after", 1.2, 1.5))
    assert detect(words) == detect(words, level_at=lambda _: -60.0)
    candidate, = detect(words).candidates
    assert candidate.start == 14 / 30
    assert candidate.end == 32 / 30


def test_candidates_are_frozen_ids_stable_and_order_independent() -> None:
    words = _words("before um middle uh after", take="t2")
    first = detect(words)
    assert first == detect(words)
    ids = {candidate.word_ids: candidate_id(candidate) for candidate in first.candidates}
    assert ids == {candidate.word_ids: candidate_id(candidate)
                   for candidate in reversed(first.candidates)}
    assert len(set(ids.values())) == 2
    candidate = first.candidates[0]
    assert candidate.take == "t2"
    assert candidate_id(candidate) == candidate_id(replace(
        candidate, segment="other", reasons=("new reason",), listen=not candidate.listen, plan={}))
    assert candidate_id(candidate) != candidate_id(replace(candidate, kind="stutter"))
    with pytest.raises(FrozenInstanceError):
        candidate.start = 0.0


@pytest.mark.parametrize(("key", "value"), [("max_pause_s", -0.1), ("beat_breath_s", -1.0),
                                           ("join_target_s", float("nan")),
                                           ("silence_db", float("inf"))])
def test_invalid_planning_parameters(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=key):
        detect((), **{key: value})


def _project(tmp_path: Path, *, language: str = "en") -> tuple[Path, dict]:
    """Create only project documents, with no media or stage side effects."""
    words = Words("t1", "a" * 64, "source", 0.0, "none", None, "b" * 64,
                  _words("outside um before uh middle after um outside"))
    source = tmp_path / "ingest/t1/words.json"
    source.parent.mkdir(parents=True)
    payload = json.dumps(words.to_dict()).encode()
    source.write_bytes(payload)
    project = {"schema": "yap-studio/project@1", "name": "sample", "language": language,
               "takes": [{"id": "t1", "path": "source.mov"}],
               "rules": {"max_pause_s": 0.6, "beat_breath_s": 0.5}}
    (tmp_path / "project.json").write_text(json.dumps(project), encoding="utf-8")
    edit = {"schema": "yap-studio/edit@1",
            "words_sha256": {"t1": hashlib.sha256(payload).hexdigest()},
            "segments": [{"id": "s1", "take": "t1",
                          "from": {"word": "t1:2", "text": "before"},
                          "to": {"word": "t1:5", "text": "after"}},
                         {"id": "s2", "take": "t1",
                          "from": {"time": 0.9}, "to": {"time": 2.4}}]}
    (tmp_path / "edit.json").write_text(json.dumps(edit), encoding="utf-8")
    return source, edit


def test_project_wrapper_resolves_segments_and_leaves_files_unchanged(tmp_path: Path) -> None:
    _project(tmp_path)
    before = {file.relative_to(tmp_path): file.read_bytes()
              for file in tmp_path.rglob("*") if file.is_file()}
    candidates = candidates_for_project(tmp_path)
    assert len(candidates) == 2
    assert {candidate.segment for candidate in candidates} == {"s1", "s2"}
    assert all(candidate.word_ids == ("t1:3",) for candidate in candidates)
    assert all(candidate.take == "t1" for candidate in candidates)
    assert candidates_for_project(tmp_path, protect=((1.3, 1.6),)) == []
    assert before == {file.relative_to(tmp_path): file.read_bytes()
                      for file in tmp_path.rglob("*") if file.is_file()}


def test_project_wrapper_honours_language_and_pause_overrides(tmp_path: Path) -> None:
    source, edit = _project(tmp_path, language="fr")
    words = Words("t1", "a" * 64, "source", 0.0, "none", None, "b" * 64,
                  (Word("t1:0", "um", 0.0, 0.3), Word("t1:1", "uh", 1.2, 1.5)))
    payload = json.dumps(words.to_dict()).encode()
    source.write_bytes(payload)
    edit["words_sha256"]["t1"] = hashlib.sha256(payload).hexdigest()
    edit["segments"] = [{"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"time": 1.5}}]
    (tmp_path / "edit.json").write_text(json.dumps(edit), encoding="utf-8")
    candidate, = candidates_for_project(tmp_path)
    assert candidate.kind == "silence"
    assert candidate.end - candidate.start == pytest.approx(0.3)
    assert candidate.plan["want_pause_s"] == 0.6


@pytest.mark.parametrize("missing", [False, True])
def test_project_wrapper_reports_unreadable_words(tmp_path: Path, missing: bool) -> None:
    source, _ = _project(tmp_path)
    if missing:
        source.unlink()
    else:
        source.write_text("{", encoding="utf-8")
    with pytest.raises(ProjectFileError, match="ingest/t1/words.json"):
        candidates_for_project(tmp_path)


@pytest.mark.parametrize("case", ["missing_word", "guard", "changed_words", "unknown_take",
                                  "backwards"])
def test_project_wrapper_rejects_invalid_segment_references(tmp_path: Path, case: str) -> None:
    _, edit = _project(tmp_path)
    first = edit["segments"][0]
    if case == "missing_word":
        first["from"] = {"word": "t1:99"}
    elif case == "guard":
        first["from"]["text"] = "wrong"
    elif case == "changed_words":
        edit["words_sha256"]["t1"] = "c" * 64
    elif case == "unknown_take":
        first["take"] = "t2"
    else:
        first["from"] = {"time": 3}
        first["to"] = {"time": 1}
    (tmp_path / "edit.json").write_text(json.dumps(edit), encoding="utf-8")
    with pytest.raises(ProjectFileError):
        candidates_for_project(tmp_path)
