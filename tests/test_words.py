"""Word normalisation and transcript-window tests."""

from __future__ import annotations

import pytest

from yap_studio.transcript import Transcript, Word, compare_words, norm_words, words_between
from yap_studio.transcript.words import title_words


@pytest.mark.parametrize("text,expected", [
    ("25", ["25"]),
    ("2.5", ["two", "five"]),
    ("2026", ["2026"]),
    ("$5", ["five", "dollars"]),
    ("50%", ["50", "percent"]),
    ("A.P.I.2", ["a", "p", "i", "two"]),
])
def test_title_words_keeps_main_normalization(text: str, expected: list[str]) -> None:
    assert title_words(text) == expected


def test_norm_words_numbers_hyphens_and_apostrophes() -> None:
    assert norm_words("Seven 10-year-olds said 'hello' and don't leave.") == [
        "seven",
        "ten",
        "year",
        "olds",
        "said",
        "hello",
        "and",
        "don't",
        "leave",
    ]


def test_norm_words_equivalences_without_private_entries() -> None:
    assert norm_words("Kinda gonna wanna be OK, bc.") == [
        "kind",
        "of",
        "going",
        "to",
        "want",
        "to",
        "be",
        "okay",
        "bc",
    ]


def test_compare_words_exact() -> None:
    result = compare_words(["bright", "planet"], ["bright", "planet"])
    assert result == {"exact": True, "core_exact": True, "ratio": 1.0, "diff": []}


def test_compare_words_tolerates_only_edge_word_differences() -> None:
    first = compare_words(
        ["bright", "planets", "circle"],
        ["right", "planets", "circle"],
    )
    last = compare_words(
        ["bright", "planets", "circle"],
        ["bright", "planets", "cir", "call"],
    )
    middle = compare_words(
        ["bright", "planets", "circle"],
        ["bright", "moons", "circle"],
    )
    assert first["exact"] is False and first["core_exact"] is True
    assert last["exact"] is False and last["core_exact"] is True
    assert middle["core_exact"] is False


def test_compare_words_can_ignore_fillers() -> None:
    strict = compare_words(["well", "um", "okay"], ["well", "okay"])
    optional = compare_words(
        ["well", "um", "okay"],
        ["well", "uh", "okay"],
        fillers_optional=True,
    )
    assert strict["exact"] is False
    assert optional["exact"] is True


def test_words_between_selection_rules_include_straddling_word_only_for_overlap() -> None:
    straddling = Word("before", 0.5, 1.2)
    inside = Word("inside", 1.1, 1.4)
    boundary = Word("after", 2.0, 2.2)
    transcript = Transcript([straddling, inside, boundary])
    assert words_between(transcript, 1.0, 2.0, "midpoint") == [inside]
    assert words_between(transcript, 1.0, 2.0, "start") == [inside]
    assert words_between(transcript, 1.0, 2.0, "overlap") == [straddling, inside]
