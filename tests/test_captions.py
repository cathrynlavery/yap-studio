"""Caption behavior and shared style-preview cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from yap_studio.captions import CaptionWord, Phrase, build_phrases
from yap_studio.text_rules import apply_text_rules

CASES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "phrase-cases.json").read_text(encoding="utf-8")
)
BY_NAME = {case["name"]: case for case in CASES}


def _build(case: dict[str, Any]) -> list[Phrase]:
    return build_phrases(
        [CaptionWord(**word) for word in case["words"]], case["captions"], **case["options"]
    )


def _named(name: str) -> list[Phrase]:
    return _build(BY_NAME[name])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_every_shared_case_matches(case: dict[str, Any]) -> None:
    assert [phrase.to_dict() for phrase in _build(case)] == case["phrases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_same_input_twice_gives_equal_output(case: dict[str, Any]) -> None:
    assert _build(case) == _build(case)


def test_twelve_word_sentence_splits_into_three_without_ending_on_the() -> None:
    phrases = _named("twelve words under a four word limit")
    assert [len(phrase.words) for phrase in phrases] == [4, 4, 4]
    assert all(phrase.words[-1].text.lower() != "the" for phrase in phrases)
    assert [phrase.index for phrase in phrases] == [0, 1, 2]


def test_forward_word_moves_to_following_phrase() -> None:
    phrases = _named("forward word stays with following words")
    assert [phrase.text for phrase in phrases] == ["We put bold words", "on the bright screen."]


def test_comma_inside_candidate_moves_break_to_comma() -> None:
    phrases = _named("comma moves the break")
    assert [phrase.text for phrase in phrases] == [
        "Keep these words,",
        "then finish this sentence.",
    ]


def test_twenty_one_character_limit_splits_long_word_pair() -> None:
    phrases = _named("twenty one character limit")
    assert [phrase.text for phrase in phrases] == ["extraordinary", "possibilities."]
    assert all(len(phrase.text) <= 21 for phrase in phrases)


def test_gap_over_one_second_forces_break() -> None:
    assert [phrase.text for phrase in _named("gap over one second")] == [
        "Keep going",
        "after silence.",
    ]


def test_gap_threshold_is_strictly_over_one_second() -> None:
    assert [phrase.text for phrase in _named("exactly one second gap stays in a segment")] == [
        "Keep going."
    ]
    assert [phrase.text for phrase in _named("two words separated by over one second")] == [
        "Keep",
        "going.",
    ]


def test_stress_is_slowest_content_word_per_letter() -> None:
    phrase = _named("slowest content word gets stress")[0]
    assert phrase.stress_id == "t1:2"
    assert phrase.words[2].text == "signal"


def test_word_fix_keeps_first_id_and_both_words_time_span() -> None:
    phrase = _named("tail net fix spans both source words")[0]
    merged = phrase.words[1]
    assert phrase.text == "Try tailnet."
    assert merged.id == "t1:1"
    assert merged.source_ids == ("t1:1", "t1:2")
    assert (merged.start, merged.end) == (0.4, 1.1)
    assert phrase.word_ids == ("t1:0", "t1:1")


def test_rejected_filler_kept_in_cut_is_hidden_from_captions() -> None:
    phrase = _named("kept rejected filler is hidden")[0]
    assert phrase.text == "We keep going."
    assert phrase.word_ids == ("t1:0", "t1:2", "t1:3")
    assert phrase.words[1].start == 0.5


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_phrase_timing_never_overlaps_next_phrase(case: dict[str, Any]) -> None:
    phrases = _build(case)
    for phrase, following in zip(phrases, phrases[1:], strict=False):
        assert phrase.start <= phrase.end <= following.start


def test_quarter_second_spacing_overrides_minimum_and_gap() -> None:
    first, second = _named("phrases starting a quarter second apart")
    assert (first.start, first.end, second.start, second.end) == (0, 0.25, 0.25, 1.65)


def test_manual_stress_wins_and_first_word_pick_wins() -> None:
    assert _named("first manual pick wins")[0].stress_id == "t1:0"
    assert _named("manual stress through a merged source id")[0].stress_id == "t1:1"


def test_forced_break_can_refer_to_any_merged_source_id() -> None:
    assert [phrase.text for phrase in _named("forced break through a merged source id")] == [
        "Try",
        "tailnet today.",
    ]


def test_longest_fix_matches_case_and_preserves_trailing_punctuation() -> None:
    phrases = _named("longest case insensitive fix preserves punctuation")
    assert [phrase.text for phrase in phrases] == ["Try “tailnet.”", "network!"]
    assert phrases[1].words[0].source_ids == ("t1:3", "t1:4", "t1:5")


def test_hidden_words_drop_before_fixes() -> None:
    word = _named("fixes run after hidden words drop")[0].words[0]
    assert word.text == "tailnet."
    assert word.source_ids == ("t1:0", "t1:2")
    assert (word.start, word.end) == (0, 0.9)


def test_sentence_ends_ignore_trailing_quotes() -> None:
    assert [phrase.text for phrase in _named("quoted sentence ends")] == [
        "Say “hello.”",
        "Then “go!”",
        "Why?",
    ]


def test_internal_pause_moves_break() -> None:
    assert [phrase.text for phrase in _named("internal pause moves the break")] == [
        "Keep these words",
        "then finish strong.",
    ]


def test_equal_split_costs_favor_longer_phrases_lexicographically() -> None:
    first = _named("equal split costs prefer longer first phrase")
    second = _named("equal split costs prefer longer second phrase")
    assert [len(phrase.words) for phrase in first] == [3, 2]
    assert [len(phrase.words) for phrase in second] == [3, 3, 2]


def test_half_milliseconds_round_up() -> None:
    first, second = _named("half milliseconds round upward")
    assert (first.start, first.end, second.start, second.end) == (0.063, 0.753, 0.813, 2.213)


def test_long_word_is_feasible_alone() -> None:
    phrases = _named("single overlong word")
    assert len(phrases) == 1
    assert phrases[0].text == "extraordinarily."


def test_empty_and_fully_hidden_input_produce_no_phrases() -> None:
    assert _named("empty input") == []
    assert _named("hidden input") == []


@pytest.mark.parametrize("next_start", [0.1, 0.1001])
def test_zero_length_phrases_drop_after_rounding(next_start: float) -> None:
    phrases = build_phrases(
        [CaptionWord("a", "Empty.", 0.1, 0.2),
         CaptionWord("b", "Visible.", next_start, 0.3)],
        {"max_words_per_phrase": 1, "max_characters_per_phrase": 30},
    )
    assert [phrase.text for phrase in phrases] == ["Visible."]
    assert all(phrase.start < phrase.end for phrase in phrases)


def test_text_rules_keep_merged_number_sources_and_manual_marks() -> None:
    words = [CaptionWord("a", "twenty", 0, 0.2),
             CaptionWord("hidden", "um", 0.21, 0.3, hidden=True),
             CaptionWord("b", "six", 0.31, 0.5),
             CaptionWord("c", "hours", 0.51, 0.7)]
    limits = {"max_words_per_phrase": 8, "max_characters_per_phrase": 80}
    phrase = build_phrases(words, limits, text_rules=True, stress_ids={"b"})[0]
    assert phrase.text == "26 hours"
    assert phrase.words[0].source_ids == ("a", "b")
    assert (phrase.words[0].id, phrase.words[0].start, phrase.words[0].end) == \
        ("a", 0, 0.5)
    assert phrase.stress_id == "a"
    leading = [CaptionWord("before", "wait", -0.2, -0.1), *words]
    assert [part.text for part in build_phrases(
        leading, limits, text_rules=True, forced_breaks={"b"})] == ["wait 20", "6 hours"]


def test_number_merge_keeps_caption_pause() -> None:
    words = [CaptionWord("a", "two", 0, 0.2), CaptionWord("b", "hundred", 3.2, 3.4)]
    limits = {"max_words_per_phrase": 8, "max_characters_per_phrase": 80}
    phrases = build_phrases(words, limits, text_rules=True)
    assert [phrase.text for phrase in phrases] == ["2", "100"]
    assert all(phrase.text != "200" for phrase in phrases)


def test_number_merge_respects_forced_breaks_and_keeps_legacy_phrases() -> None:
    words = [CaptionWord("a", "twenty", 0, 0.2), CaptionWord("b", "one", 0.3, 0.5)]
    limits = {"max_words_per_phrase": 8, "max_characters_per_phrase": 80}
    legacy = build_phrases(words, limits, text_rules=True)
    assert [phrase.text for phrase in legacy] == ["21"]
    assert legacy == build_phrases(words, limits, text_rules=True, forced_breaks=())
    assert [phrase.text for phrase in build_phrases(
        words, limits, text_rules=True, forced_breaks={"b"})] == ["20", "one"]
    following = [*words, CaptionWord("c", "hours", 0.6, 0.8)]
    assert [phrase.text for phrase in build_phrases(
        following, limits, text_rules=True, forced_breaks={"c"})] == ["21", "hours"]


@pytest.mark.parametrize(("words", "word_fixes", "vocabulary", "expected"), [
    (["“sparrowlight”"], {}, ["SparrowLight"], "“SparrowLight”"),
    (["(sparrowlight's)"], {}, ["SparrowLight"], "(SparrowLight's)"),
    (["\"foo", "bar,"], {"foo bar": "Fixed"}, [], "\"Fixed,"),
])
def test_display_fix_keeps_opening_punctuation(
    words: list[str], word_fixes: dict[str, str], vocabulary: list[str], expected: str,
) -> None:
    source = [CaptionWord(str(i), word, i * 0.2, i * 0.2 + 0.1)
              for i, word in enumerate(words)]
    limits = {"max_words_per_phrase": 8, "max_characters_per_phrase": 80}
    phrase = build_phrases(source, limits, word_fixes=word_fixes, text_rules=True,
                           vocabulary=vocabulary)[0]
    assert phrase.text == expected
    assert phrase.words[0].source_ids == tuple(str(i) for i in range(len(words)))
    assert (phrase.words[0].start, phrase.words[0].end) == (0, (len(words) - 1) * 0.2 + 0.1)


def test_title_and_caption_choose_longest_vocabulary_fix() -> None:
    vocabulary = ["red", "Red Blue"]
    words = [CaptionWord("a", "red", 0, 0.1), CaptionWord("b", "blue", 0.2, 0.3)]
    limits = {"max_words_per_phrase": 8, "max_characters_per_phrase": 80}
    caption = build_phrases(words, limits, text_rules=True, vocabulary=vocabulary)[0]
    assert caption.text == "Red Blue"
    assert apply_text_rules("red blue", vocabulary) == caption.text


def test_text_rules_year_spans_all_four_sources() -> None:
    words = [CaptionWord(str(i), text, i * 0.2, i * 0.2 + 0.1)
             for i, text in enumerate("two thousand twenty six".split())]
    phrase = build_phrases(words, {"max_words_per_phrase": 8,
                                   "max_characters_per_phrase": 80}, text_rules=True)[0]
    assert phrase.words[0].text == "2026"
    assert phrase.words[0].source_ids == ("0", "1", "2", "3")
    assert phrase.words[0].start == 0
    assert phrase.words[0].end == pytest.approx(0.7)


def test_text_rules_do_not_end_a_group_on_a_converted_number() -> None:
    words = [CaptionWord(str(i), text, i * 0.3, i * 0.3 + 0.2)
             for i, text in enumerate(
                 "we are only here for two three hours I need that screen time".split())]
    limits = {"max_words_per_phrase": 5, "max_characters_per_phrase": 21}
    assert [phrase.text for phrase in build_phrases(words, limits)] == [
        "we are only here", "for two three", "hours I need", "that screen time"]
    forced_breaks = {"1"}
    unchanged = build_phrases(words, limits, forced_breaks=forced_breaks)
    assert [phrase.text for phrase in unchanged] == [
        "we", "are only here for two", "three hours I need", "that screen time"]

    phrases = build_phrases(words, limits, forced_breaks=forced_breaks, text_rules=True)
    assert all(phrase.words[-1].text != "2-3" for phrase in phrases)
    assert any("2-3 hours" in phrase.text for phrase in phrases)


def test_text_rules_sentence_start_and_vocabulary_precedence() -> None:
    words = [CaptionWord(str(i), text, i * 0.2, i * 0.2 + 0.1)
             for i, text in enumerate("to me no No, watch Why don't".split())]
    limits = {"max_words_per_phrase": 8, "max_characters_per_phrase": 80}
    assert [part.text for part in build_phrases(words, limits, text_rules=True)] == [
        "to me no", "No, watch", "Why don't"]
    named = build_phrases([CaptionWord("a", "sparrowlight's", 0, 0.1),
                           CaptionWord("b", "SPARROWLIGHT.", 0.2, 0.3)],
                          limits, text_rules=True, vocabulary=["SparrowLight"])
    assert [part.text for part in named] == ["SparrowLight's SparrowLight."]
    fixed = build_phrases([CaptionWord("a", "sparrowlight", 0, 0.1)], limits,
                          text_rules=True, vocabulary=["SparrowLight"],
                          word_fixes={"sparrowlight": "MyBrand"})
    assert fixed[0].text == "MyBrand"
    possessive = build_phrases([CaptionWord("a", "sparrowlight's", 0, 0.1)], limits,
                               text_rules=True, vocabulary=["SparrowLight"],
                               word_fixes={"sparrowlight's": "Owner's"})
    assert possessive[0].text == "Owner's"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "3"])
@pytest.mark.parametrize("field", ["max_words_per_phrase", "max_characters_per_phrase"])
def test_limits_must_be_positive_integers(field: str, limit: Any) -> None:
    captions = {"max_words_per_phrase": 3, "max_characters_per_phrase": 21, field: limit}
    with pytest.raises(ValueError, match="positive integers"):
        build_phrases([], captions)


@pytest.mark.parametrize("text_rules", [False, True])
@pytest.mark.parametrize(("first", "start", "breaks"), [
    ("Zindle", 8.0, ()),
    ("Zindle", 8.0, ("b",)),
    ("Zindle", 1.401, ()),
    ("Zindle.", 0.5, ()),
    ("Zindle!", 0.5, ()),
    ("Zindle?", 0.5, ()),
    ("“Zindle.”", 0.5, ()),
])
def test_multiword_fix_respects_speech_boundaries(
    first: str, start: float, breaks: tuple[str, ...], text_rules: bool,
) -> None:
    words = [CaptionWord("a", first, 0, 0.4), CaptionWord("b", "Vay", start, start + 0.4)]
    limits = {"max_words_per_phrase": 4, "max_characters_per_phrase": 80}
    phrases = build_phrases(words, limits, word_fixes={"zindle vay": "ZindleVay"},
                            forced_breaks=breaks, text_rules=text_rules)
    assert [phrase.text for phrase in phrases] == [first, "Vay"]
    assert [phrase.word_ids for phrase in phrases] == [("a",), ("b",)]
    assert [word.source_ids for phrase in phrases for word in phrase.words] == [("a",), ("b",)]
    assert [(word.start, word.end) for phrase in phrases for word in phrase.words] == [
        (0, 0.4), (start, start + 0.4),
    ]
    assert phrases[0].end <= phrases[1].start


@pytest.mark.parametrize("start", [0.5, 1.4])
@pytest.mark.parametrize("breaks", [(), ("b",)])
def test_multiword_fix_within_speech_keeps_ids_and_timing(
    start: float, breaks: tuple[str, ...],
) -> None:
    words = [CaptionWord("a", "Zindle", 0, 0.4), CaptionWord("b", "Vay.", start, start + 0.4)]
    limits = {"max_words_per_phrase": 4, "max_characters_per_phrase": 80}
    phrase, = build_phrases(words, limits, word_fixes={"zindle vay": "ZindleVay"},
                            forced_breaks=breaks)
    assert phrase.text == "ZindleVay."
    assert phrase.word_ids == ("a",)
    assert phrase.words[0].source_ids == ("a", "b")
    assert (phrase.words[0].start, phrase.words[0].end) == (0, start + 0.4)


@pytest.mark.parametrize("boundary", [1, 2])
def test_multiword_fix_checks_every_internal_boundary(boundary: int) -> None:
    words = [CaptionWord(word_id, text, i * 0.5 + (2 if i >= boundary else 0),
                         i * 0.5 + (2 if i >= boundary else 0) + 0.4)
             for i, (word_id, text) in enumerate(zip("abc", ["Zindle", "Vay", "Nuv"], strict=True))]
    limits = {"max_words_per_phrase": 4, "max_characters_per_phrase": 80}
    phrases = build_phrases(words, limits, word_fixes={"zindle vay nuv": "ZindleVayNuv"})
    assert [word.text for phrase in phrases for word in phrase.words] == ["Zindle", "Vay", "Nuv"]
    assert [word.source_ids for phrase in phrases for word in phrase.words] == [
        ("a",), ("b",), ("c",),
    ]


@pytest.mark.parametrize("occurrence", [False, True])
def test_single_word_fixes_keep_working_across_speech_boundaries(occurrence: bool) -> None:
    words = [CaptionWord("a", "zindle", 0, 0.4), CaptionWord("b", "vay", 8, 8.4)]
    limits = {"max_words_per_phrase": 4, "max_characters_per_phrase": 80}
    phrases = build_phrases(
        words, limits, word_fixes={"zindle vay": "ZindleVay", "zindle": "Zindle"},
        word_text_fixes={"b": "VayNuv"} if occurrence else {}, forced_breaks={"b"},
    )
    assert [phrase.text for phrase in phrases] == ["Zindle", "VayNuv" if occurrence else "vay"]
    assert [word.source_ids for phrase in phrases for word in phrase.words] == [("a",), ("b",)]


@pytest.mark.parametrize("segment", ["s1", "s2"])
@pytest.mark.parametrize("camera", [False, True])
def test_multiword_fix_respects_cut_segments(segment: str, camera: bool) -> None:
    from yap_studio.render.camera3d_look import display_words

    words = [
        {"id": "a", "text": "Zindle", "start": 0, "end": 0.4, "segment": "s1"},
        {"id": "b", "text": "Vay", "start": 0.5, "end": 0.9, "segment": segment},
    ]
    limits = {"max_words_per_phrase": 4, "max_characters_per_phrase": 80}
    if camera:
        shown = display_words(words, limits, word_fixes={"zindle vay": "ZindleVay"},
                              vocabulary=[], gaps=[])
    else:
        phrases = build_phrases([CaptionWord(**word) for word in words], limits,
                                word_fixes={"zindle vay": "ZindleVay"})
        shown = [{"text": word.text, "source_ids": list(word.source_ids)}
                 for phrase in phrases for word in phrase.words]
    assert [word["text"] for word in shown] == (
        ["ZindleVay"] if segment == "s1" else ["Zindle", "Vay"])
    assert [word["source_ids"] for word in shown] == (
        [["a", "b"]] if segment == "s1" else [["a"], ["b"]])


def test_vocabulary_fix_respects_speech_boundaries() -> None:
    words = [CaptionWord("a", "zindle", 0, 0.4), CaptionWord("b", "vay", 8, 8.4)]
    limits = {"max_words_per_phrase": 4, "max_characters_per_phrase": 80}
    phrases = build_phrases(words, limits, text_rules=True, vocabulary=["Zindle Vay"])
    assert [phrase.text for phrase in phrases] == ["zindle", "vay"]
    assert [word.source_ids for phrase in phrases for word in phrase.words] == [("a",), ("b",)]
