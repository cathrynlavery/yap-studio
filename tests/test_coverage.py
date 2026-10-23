"""Speech coverage and recovery without media tools."""

from __future__ import annotations

import numpy as np

from yap_studio.stages.coverage import merge, uncovered_speech, windows
from yap_studio.transcript import Word


def _levels(*spans: tuple[float, float, float, float]) -> np.ndarray:
    """Put alternating levels in spans above a quiet floor."""
    level = np.full(500, -65.0)
    for start, end, high, low in spans:
        indices = np.arange(round(start * 100), round(end * 100))
        level[indices] = np.where(indices % 2 == 0, high, low)
    return level


def test_modulated_uncovered_speech_is_found() -> None:
    clusters = uncovered_speech(_levels((1.0, 2.0, -20, -35)), [])
    assert clusters == [(1.0, 2.0)]


def test_words_cover_the_same_speech() -> None:
    words = [Word("longword", 1.0, 2.0)]
    assert uncovered_speech(_levels((1.1, 1.9, -20, -35)), words) == []


def test_steady_loud_noise_is_not_speech() -> None:
    assert uncovered_speech(_levels((1.0, 2.0, -20, -20)), []) == []


def test_continuous_speech_with_one_word_finds_uncovered_tail() -> None:
    indices = np.arange(1000)
    level = np.where(indices % 2 == 0, -20.0, -30.0)
    clusters = uncovered_speech(level, [Word("a", 0.0, 0.3)])
    assert clusters == [(0.4, 10.0)]


def test_digital_silence_is_not_speech() -> None:
    assert uncovered_speech(np.full(1000, -180.0), []) == []


def test_steady_loud_take_with_words_is_not_speech() -> None:
    words = [Word("word", float(i), i + 0.3) for i in range(6)]
    assert uncovered_speech(np.full(1000, -25.0), words) == []


def test_modulated_sound_below_word_reference_is_not_speech() -> None:
    level = _levels((0.0, 1.5, -20, -20), (2.0, 3.0, -43, -53))
    words = [Word("word", i * 0.1, i * 0.1 + 0.04) for i in range(10)]
    assert uncovered_speech(level, words) == []


def test_short_interword_gap_is_not_speech() -> None:
    words = [Word("before", 0.2, 0.6), Word("after", 2.0, 2.4)]
    assert uncovered_speech(_levels((1.0, 1.4, -20, -35)), words) == []


def test_stretched_word_tail_is_uncovered() -> None:
    clusters = uncovered_speech(_levels((2.0, 3.0, -20, -35)), [Word("it.", 1.0, 3.5)])
    assert clusters == [(2.0, 3.0)]


def test_covered_frame_splits_clusters() -> None:
    level = _levels((1.0, 1.8, -20, -35), (2.0, 2.8, -20, -35))
    clusters = uncovered_speech(level, [Word("a", 1.8, 2.0)])
    assert len(clusters) == 2
    assert clusters[0][1] < clusters[1][0]


def test_windows_include_complete_edge_words_and_merge() -> None:
    words = [Word("first", 0.4, 1.2), Word("last", 3.8, 4.7)]
    assert [tuple(round(time, 3) for time in span)
            for span in windows([(1.3, 2.0), (2.1, 3.2)], words, 5.0)] == [(0.0, 5.0)]


def test_windows_include_short_speech_before_a_neighboring_cluster() -> None:
    assert windows([(4.0, 4.8)], [Word("Alright.", 2.4, 2.87)], 7.0) == [(2.0, 6.8)]


def test_merge_trims_stretched_word_and_inserts_missing_words() -> None:
    old = [Word("it.", 1.0, 3.0), Word("comes", 4.0, 4.4)]
    clip = [
        Word("it", 0.0, 0.35),
        Word("missing", 0.9, 1.2),
        Word("words", 1.3, 1.6),
        Word("comes", 3.0, 3.4),
    ]
    merged, added = merge(old, (1.0, 5.0), clip)
    assert added == 2
    assert [word.text for word in merged] == ["it.", "missing", "words", "comes"]
    assert merged[0].end == 1.9
    assert merged[-1] == old[-1]


def test_merge_does_not_duplicate_repeated_edge_word() -> None:
    old = [Word("first", 1.0, 1.4), Word("last", 3.0, 3.4)]
    clip = [Word("first", 0.0, 0.4), Word("middle", 1.0, 1.4), Word("last", 2.0, 2.4)]
    merged, added = merge(old, (1.0, 4.0), clip)
    assert [word.text for word in merged] == ["first", "middle", "last"]
    assert added == 1


def test_merge_keeps_repeated_word_inside_stretched_old_word() -> None:
    old = [Word("it", 1.0, 9.0)]
    clip = [Word("alpha", 4.0, 4.4), Word("it", 5.0, 5.4), Word("beta", 6.0, 6.4)]
    merged, added = merge(old, (0.0, 10.0), clip)
    assert added == 3
    assert [word.text for word in merged] == ["it", "alpha", "it", "beta"]
    assert merged[0].end == 4.0


def test_merge_never_removes_old_words() -> None:
    old = [Word("original", 1.0, 1.5), Word("second", 2.0, 2.5)]
    merged, added = merge(old, (0.0, 3.0), [Word("different", 1.0, 1.5)])
    assert merged == old
    assert added == 0


def test_merge_keeps_later_repeated_it_when_first_it_was_stretched() -> None:
    old = [Word("it.", 1.0, 9.0), Word("So", 30.0, 30.4)]
    clip = [
        Word("the", 4.0, 4.3),
        Word("more", 4.4, 4.8),
        Word("expensive", 5.0, 5.8),
        Word("it", 9.0, 9.3),
        Word("is.", 9.5, 9.8),
        Word("So", 29.0, 29.4),
    ]
    merged, added = merge(old, (1.0, 31.0), clip)
    assert added == 5
    assert [word.text for word in merged] == [
        "it.", "the", "more", "expensive", "it", "is.", "So"
    ]
    assert merged[0].end == 5.0


def test_merge_keeps_every_word_in_large_replacement_block() -> None:
    old = [Word("again.", 1.0, 5.0), Word("You're", 16.0, 16.4)]
    text = ["It's", "kind", "of", "like", "a", "bunch", "of", "errands,", "and", "rather"]
    clip = [Word(word, 5.0 + i * 1.12, 5.4 + i * 1.12) for i, word in enumerate(text)]
    clip.append(Word("You're", 16.0, 16.4))
    merged, added = merge(old, (0.0, 17.0), clip)
    assert added == 10
    assert [word.text for word in merged] == ["again.", *text, "You're"]


def test_merge_matches_repeated_text_only_near_its_old_time() -> None:
    old = [Word("same", 1.0, 1.4)]
    distant, added = merge(old, (0.0, 4.0), [Word("same", 3.0, 3.4)])
    assert added == 1
    assert [word.text for word in distant] == ["same", "same"]
    nearby, added = merge(old, (0.0, 4.0), [Word("same", 1.1, 1.5)])
    assert nearby == old
    assert added == 0
