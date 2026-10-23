"""Pure caption and title text rules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from yap_studio.text_rules import (
    apply_text_rules,
    apply_title_rules,
    balance_lines,
    near_vocabulary,
    number_merges,
    starts_sentence,
    vocabulary_fixes,
)


def normalized(text: str) -> str:
    words = text.split()
    for start, end, replacement in reversed(number_merges(words)):
        words[start:end] = [replacement]
    return " ".join(words)


@pytest.mark.parametrize(("source", "expected"), [
    ("twenty years", "20 years"), ("fifty percent", "50%"),
    ("five dollars", "$5"), ("two thousand twenty six", "2026"),
    ("twenty twenty six", "2026"), ("twenty twenty-six", "2026"),
    ("nineteen ninety", "1990"), ("fifty thousand", "50,000"),
    ("twenty twenty", "2020"), ("one million two thousand", "1,002,000"),
    ("one hundred and five", "105"), ("twenty-six", "26"),
    ("a hundred dollars", "$100"), ("20 percent", "20%"),
    ("fifty per cent", "50%"), ("fifty percent.", "50%."),
    ("twenty, five", "20, 5"), ("500 bucks", "500 bucks"),
    ("twenty, one two three", "twenty, one two three"),
    ("50 cents", "50 cents"), ("one minute", "1 minute"),
    ("one day", "one day"), ("one or two hours", "1 or 2 hours"),
    ("two three hours", "2-3 hours"),
    ("no one", "no one"), ("one of the", "one of the"),
    ("someone", "someone"), ("the one", "the one"),
    ("one take", "one take"), ("one time", "one time"),
    ("one two three four", "one two three four"),
    ("first half second", "first half second"),
])
def test_number_normalization(source: str, expected: str) -> None:
    assert normalized(source) == expected


@pytest.mark.parametrize(("source", "expected"), [
    ("five six people", "5-6 people"),
    ("ten fifteen minutes", "10-15 minutes"),
    ("twenty thirty", "2030"),
    ("seven eleven", "seven eleven"),
    ("two four hours", "two four hours"),
])
def test_only_plausible_adjacent_cardinals_form_ranges(source: str, expected: str) -> None:
    assert normalized(source) == expected


def test_commas_do_not_turn_counting_into_amounts() -> None:
    assert normalized("five, six, seven") == "five, six, seven"
    assert normalized("five six seven eight") == "five six seven eight"


@pytest.mark.parametrize(("source", "expected"), [
    ("nineteen nineties", "1990s"), ("twenty twenties", "2020s"),
    ("the nineties", "the nineties"), ("in my twenties", "in my twenties"),
])
def test_decades_need_a_century_pair(source: str, expected: str) -> None:
    assert normalized(source) == expected


@pytest.mark.parametrize(("source", "expected"), [
    ("two point five", "2.5"), ("zero point five", "0.5"),
    ("three point one four", "3.14"),
    ("two point five percent", "2.5%"),
])
def test_spoken_decimals(source: str, expected: str) -> None:
    assert normalized(source) == expected


@pytest.mark.parametrize(("source", "expected"), [
    ("a thousand people", "1,000 people"), ("two thousand", "2,000"),
    ("two thousand five hundred dollars", "$2,500"),
    ("two thousand twenty six", "2026"),
    ("two thousand and five", "2005"),
])
def test_thousands_use_commas_except_spoken_years(source: str, expected: str) -> None:
    assert normalized(source) == expected


def test_one_day_stays_an_idiom_while_other_units_convert() -> None:
    assert normalized("one day I will") == "one day I will"
    assert normalized("two days") == "2 days"
    assert normalized("one week") == "1 week"


def test_caption_builder_uses_one_text_rule_import() -> None:
    source = Path(__file__).parents[1] / "src/yap_studio/captions.py"
    imports = [node for node in ast.walk(ast.parse(source.read_text()))
               if isinstance(node, ast.ImportFrom) and node.module == "yap_studio.text_rules"]
    assert len(imports) == 1


def test_changelog_mentions_brand_command_and_spelling_report() -> None:
    source = (Path(__file__).parents[1] / "CHANGELOG.md").read_text()
    bullets = [line for line in source.splitlines() if line.startswith("- ")]
    bullet = next(line for line in bullets if "yap-studio brand words" in line)
    assert "receipt" in bullet and "spelling" in bullet


def test_balance_lines() -> None:
    assert balance_lines([10, 10, 10], 2, 40) == ()
    assert balance_lines([10, 10, 10], 2, 25) == (1,)
    assert balance_lines([30, 30], 2, 40) == ()
    assert balance_lines([10, 10, 40, 10], 2, 45) == (2,)
    assert balance_lines([10, 10, 10, 10], 0, 25) == (2,)
    assert balance_lines([10] * 5, 0, 40) == (3,)
    assert balance_lines([100, 100, 100], 0, 50) == (1,)


@pytest.mark.parametrize(("previous", "word", "expected"), [
    ("no", "No,", True), ("watch", "Why", True),
    ("me", "Alright,", True), ("TV", "I'm", False),
    ("call", "Jordan", False), ("me,", "Why", False),
    ("done.”", "Then", True), ("me", "TV", False),
])
def test_sentence_starts(previous: str, word: str, expected: bool) -> None:
    assert starts_sentence(previous, word) is expected


def test_vocabulary_helpers() -> None:
    assert vocabulary_fixes(["SparrowLight", "Instrument Serif"]) == {
        "sparrowlight": "SparrowLight", "instrument serif": "Instrument Serif",
    }
    assert apply_text_rules("sparrowlight's fifty percent.", ["SparrowLight"]) == \
        "SparrowLight's 50%."
    assert apply_text_rules("instrument serif", ["Instrument Serif"]) == "Instrument Serif"
    assert near_vocabulary(
        ["SparrowLight", "Sparrowlite", "sparrow", "light", "paper", "kettle", "yep"],
        ["SparrowLight", "PaperKettle", "Yap"],
    ) == [(1, 1, "SparrowLight"), (2, 2, "SparrowLight"), (4, 2, "PaperKettle")]
    assert near_vocabulary(["SparrowLight's", "SparrowLight."], ["SparrowLight"]) == []


@pytest.mark.parametrize(("line", "accent", "expected_line", "expected_accent"), [
    ("five dollars", "dollars", "$5", "$5"),
    ("twenty five", "twenty", "25", "25"),
    ("my sparrowlight story", "story", "my SparrowLight story", "story"),
    ("sparrowlight rocks", "sparrowlight", "SparrowLight rocks", "SparrowLight"),
])
def test_title_accent_follows_converted_source_tokens(
    line: str, accent: str, expected_line: str, expected_accent: str,
) -> None:
    assert apply_title_rules([line], accent, ["SparrowLight"]) == \
        ([expected_line], expected_accent)


def test_title_accent_outside_lines_uses_text_rules() -> None:
    assert apply_title_rules(["hello there"], "five dollars", []) == \
        (["hello there"], "$5")


@pytest.mark.parametrize(("line", "accent", "expected_line", "expected_accent"), [
    ("hello   world", "hello   world", "hello world", "hello world"),
    ("a  five dollars  b", "five dollars", "a $5 b", "$5"),
    ("hello world", "ell", "hello world", "ell"),
])
def test_title_accent_matches_converted_line_spacing(
    line: str, accent: str, expected_line: str, expected_accent: str,
) -> None:
    converted, key = apply_title_rules([line], accent, [])
    assert (converted, key) == ([expected_line], expected_accent)
    assert any(key in rendered for rendered in converted)


def test_title_vocabulary_prefers_longest_match() -> None:
    assert apply_text_rules("red blue", ["red", "Red Blue"]) == "Red Blue"
