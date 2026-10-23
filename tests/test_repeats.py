"""Repeated ideas in synthetic retained scripts, on the cut clock."""

from __future__ import annotations

import pytest

from yap_studio.stages.repeats import RepeatWord, find_repeats, repeat_warnings


def script(*sentences: tuple[float, str]) -> list[RepeatWord]:
    words = []
    for start, sentence in sentences:
        for offset, text in enumerate(sentence.split()):
            words.append(RepeatWord(f"t1:{len(words)}", text, start + offset * 0.2,
                                    start + offset * 0.2 + 0.15))
    return words


def test_thesis_in_three_sentences_reports_three_pairs() -> None:
    words = script(
        (4, "The lantern needs steady amber light."),
        (31, "A lantern needs steady amber light!"),
        (68, "Every lantern needs steady amber light?"),
    )
    repeats = find_repeats(words)
    assert len(repeats) == 3
    assert [row["shared"] for row in repeats] == ["lantern needs steady amber light"] * 3
    assert [(row["first"]["time_s"], row["second"]["time_s"]) for row in repeats] == [
        (4.2, 31.2), (4.2, 68.2), (31.2, 68.2),
    ]
    assert repeats[0]["kind"] == "phrase"
    assert repeats[0]["first"]["sentence"] == "The lantern needs steady amber light."
    assert repeats[0]["first"]["words"] == [f"t1:{index}" for index in range(6)]
    assert repeats[0]["second"]["words"] == [f"t1:{index}" for index in range(6, 12)]
    assert repeat_warnings(repeats)[1] == (
        'Repeated: "lantern needs steady amber light" at 0:04 and again at 1:08 '
        '("Every lantern needs steady amber light?"). '
        "Keep it once unless the repeat is on purpose."
    )


def test_same_point_in_different_words() -> None:
    repeats = find_repeats(script(
        (12, "Lantern light makes the garden welcoming."),
        (40, "The garden glows with light from a lantern."),
    ))
    assert len(repeats) == 1
    assert repeats[0]["kind"] == "point"
    assert repeats[0]["shared"] == ["garden", "lantern", "light"]
    assert repeats[0]["first"]["time_s"] == 12
    assert repeats[0]["second"]["time_s"] == 40
    assert repeat_warnings(repeats) == [
        'Same point twice: 0:12 and 0:40 share "garden", "lantern", "light". '
        "Keep the clearer one unless the repeat is on purpose."
    ]


@pytest.mark.parametrize("sentences", [
    ((0, "I went to the I went to the store."),),
    ((0, "The lantern needs amber light."), (2.9, "The lantern needs amber light.")),
    ((0, "Garden light welcomes guests."), (10, "Garden light guides wandering moths."),
     (20, "Lantern light warms quiet evenings.")),
    ((0, "I think that."), (10, "I think that.")),
    ((0, "You know what I mean."), (10, "You know what I mean.")),
    ((0, "Like just really yeah."), (10, "Like just really yeah.")),
    ((0, "Lantern lantern glows."), (10, "Glows lantern lantern.")),
    ((0, "Lantern garden light warms quiet peaceful evenings."),
     (10, "Garden lantern light guides wandering moths across distant meadows.")),
])
def test_clean_scripts_and_restarts_do_not_warn(sentences: tuple[tuple[float, str], ...]) -> None:
    assert find_repeats(script(*sentences)) == []


def test_restart_boundary_is_three_seconds() -> None:
    assert len(find_repeats(script(
        (0, "Lantern needs amber light."), (3, "Lantern needs amber light."),
    ))) == 1


def test_restart_uses_phrase_start_and_does_not_fall_back_to_point() -> None:
    assert find_repeats(script(
        (0, "Bright clouds drift slowly but amber lantern glows."),
        (3.5, "Amber lantern glows."),
    )) == []


def test_same_point_at_exactly_half_the_shorter_content() -> None:
    repeats = find_repeats(script(
        (0, "Lantern garden light warms quiet evenings."),
        (10, "Garden lantern light guides wandering moths across distant meadows."),
    ))
    assert len(repeats) == 1
    assert repeats[0]["kind"] == "point"


def test_normalizes_punctuation_and_apostrophes_and_keeps_unfinished_sentence() -> None:
    repeats = find_repeats(script(
        (0, "A lantern's soft, amber light."),
        (10, "The LANTERN’S soft amber light"),
    ))
    assert len(repeats) == 1
    assert repeats[0]["shared"] == "lanterns soft amber light"
    assert repeats[0]["second"]["sentence"] == "The LANTERN’S soft amber light"


@pytest.mark.parametrize("ending", ["", "."])
def test_repeated_phrase_with_or_without_punctuation(ending: str) -> None:
    sentence = f"Amber lanterns glow brightly tonight{ending}"
    repeats = find_repeats(script((0, sentence), (10, sentence)))
    assert len(repeats) == 1
    assert repeats[0]["shared"] == "amber lanterns glow brightly tonight"
    assert repeats[0]["first"]["time_s"] == 0
    assert repeats[0]["second"]["time_s"] == 10
    assert len(repeat_warnings(repeats)) == 1


def test_unpunctuated_same_point_in_different_words() -> None:
    repeats = find_repeats(script(
        (0, "Lantern light makes the garden welcoming"),
        (10, "The garden glows with light from a lantern"),
    ))
    assert len(repeats) == 1
    assert repeats[0]["kind"] == "point"


@pytest.mark.parametrize("ending", ["", "."])
def test_short_phrase_does_not_warn_without_punctuation(ending: str) -> None:
    assert find_repeats(script((0, f"you know{ending}"), (10, f"you know{ending}"))) == []


def test_punctuated_rhetorical_repetition_keeps_one_sentence_despite_pause() -> None:
    assert find_repeats(script(
        (0, "Amber lanterns glow brightly tonight"),
        (10, "amber lanterns glow brightly tonight, I say again."),
    )) == []


@pytest.mark.parametrize(("opening", "closing"), [
    ("", ""), ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"),
    ("(", ")"), ("[", "]"), ("{", "}"), ("«", "»"), ("(“", "”)"),
])
def test_repeated_sentences_with_closing_delimiters(opening: str, closing: str) -> None:
    sentence = f"{opening}Amber lanterns glow brightly tonight.{closing}"
    repeats = find_repeats(script((0, sentence), (10, sentence)))
    assert len(repeats) == 1
    assert repeats[0]["shared"] == "amber lanterns glow brightly tonight"
    assert repeats[0]["first"]["sentence"] == sentence
    assert repeats[0]["second"]["sentence"] == sentence
    assert repeat_warnings(repeats) == [
        'Repeated: "amber lanterns glow brightly tonight" at 0:00 and again at 0:10 '
        f'("{sentence}"). Keep it once unless the repeat is on purpose.'
    ]


def test_phrase_needs_two_content_tokens() -> None:
    assert find_repeats(script((0, "It is the lantern."), (10, "It is the lantern."))) == []


def test_warning_cap_keeps_all_twelve_pairs_in_data() -> None:
    words = script(*[
        (index * 10, sentence)
        for index, sentence in enumerate([
            "Amber lantern glows.", "Amber lantern glows.",
            "Amber lantern glows.", "Amber lantern glows.",
            "Quiet river flows.", "Quiet river flows.",
            "Quiet river flows.", "Quiet river flows.",
        ])
    ])
    repeats = find_repeats(words)
    assert len(repeats) == 12
    warnings = repeat_warnings(repeats)
    assert len(warnings) == 9
    assert all(warning.startswith("Repeated:") for warning in warnings[:8])
    assert warnings[-1] == "4 more repeated sentence pairs."


def test_empty_script() -> None:
    assert find_repeats([]) == []
    assert repeat_warnings([]) == []
