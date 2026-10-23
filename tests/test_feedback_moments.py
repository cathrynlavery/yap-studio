"""Occurrence mapping preserves editorial identity across retimed cuts."""

from __future__ import annotations

import pytest

from yap_studio.feedback.moments import map_moment, occurrence_at, occurrences
from yap_studio.project.model import CutWord, CutWords


def _words(*items: tuple[str, str, float, float]) -> CutWords:
    return CutWords("output", {}, tuple(CutWord(i, s, f"word {i}", a, b) for i, s, a, b in items))


def test_inside_uses_offset_and_retimed_occurrence() -> None:
    source = _words(("a", "s1", 1.0, 2.0), ("b", "s1", 3.0, 4.0))
    target = _words(("a", "s1", 3.0, 4.0), ("b", "s1", 5.0, 6.0))
    result = map_moment(source, target, 1.4)
    assert result.status == "mapped"
    assert result.time_s == pytest.approx(3.4)
    assert result.word == occurrences(source)[0]
    assert result.target == occurrences(target)[0]
    assert result.candidates == ()
    assert result.to_dict() == {
        "status": "mapped", "time_s": 3.4,
        "word": {"id": "a", "segment": "s1", "text": "word a", "start": 1.0, "end": 2.0},
        "target": {"id": "a", "segment": "s1", "text": "word a", "start": 3.0, "end": 4.0},
        "candidates": [],
    }


def test_silence_preserves_offset_and_stays_before_next_word() -> None:
    source = _words(("a", "s1", 1.0, 2.0), ("b", "s1", 4.0, 5.0))
    target = _words(("a", "s1", 3.0, 4.0), ("b", "s1", 5.0, 6.0))
    anchor, offset, where = occurrence_at(source, 2.4)
    assert anchor == occurrences(source)[0]
    assert (offset, where) == (pytest.approx(0.4), "after")
    result = map_moment(source, target, 2.4)
    assert result.status == "mapped" and result.time_s == pytest.approx(4.4)
    capped = map_moment(source, target, 3.8)
    assert 4.99 < capped.time_s < 5.0


def test_half_open_boundary_anchors_after_word_or_inside_next() -> None:
    words = _words(("a", "s1", 1.0, 2.0), ("b", "s1", 2.0, 3.0))
    assert occurrence_at(words, 2.0) == (occurrences(words)[1], 0.0, "inside")
    assert occurrence_at(words, 3.0) == (occurrences(words)[1], 0.0, "after")


def test_before_first_word_preserves_negative_offset() -> None:
    source = _words(("a", "s1", 2.0, 3.0))
    target = _words(("a", "s1", 4.0, 5.0))
    assert occurrence_at(source, 0.5) == (occurrences(source)[0], -1.5, "before")
    assert map_moment(source, target, 0.5).time_s == 2.5
    assert map_moment(source, target, -8.0).time_s == 0.0


def test_shorter_target_word_clips_inside_offset_below_end() -> None:
    source = _words(("a", "s1", 1.0, 3.0))
    target = _words(("a", "s1", 4.0, 4.2))
    result = map_moment(source, target, 2.0)
    assert 4.19 < result.time_s < 4.2


def test_unique_id_in_new_segment_is_moved() -> None:
    source = _words(("a", "old", 1.0, 2.0))
    target = _words(("a", "new", 6.0, 7.0))
    result = map_moment(source, target, 1.25)
    assert result.status == "moved" and result.time_s == 6.25
    assert result.target == occurrences(target)[0]


def test_repeated_id_is_ambiguous_in_output_order() -> None:
    source = _words(("a", "old", 1.0, 2.0))
    target = _words(("a", "new1", 6.0, 7.0), ("a", "new2", 8.0, 9.0))
    result = map_moment(source, target, 1.25)
    assert result.status == "ambiguous" and result.time_s == 6.0
    assert result.target is None and result.word == occurrences(source)[0]
    assert result.candidates == occurrences(target)
    assert [item["segment"] for item in result.to_dict()["candidates"]] == ["new1", "new2"]


def test_exact_segment_wins_over_other_occurrences_of_same_id() -> None:
    source = _words(("a", "s1", 1.0, 2.0))
    target = _words(("a", "s2", 3.0, 4.0), ("a", "s1", 6.0, 7.0))
    result = map_moment(source, target, 1.25)
    assert result.status == "mapped" and result.time_s == 6.25
    assert result.candidates == ()


def test_deleted_stretch_maps_to_next_kept_occurrence() -> None:
    source = _words(("a", "s1", 1.0, 2.0), ("b", "s1", 2.0, 3.0), ("c", "s1", 4.0, 5.0))
    target = _words(("c", "s1", 1.0, 2.0))
    result = map_moment(source, target, 1.5)
    assert result.status == "removed" and result.time_s == 1.0
    assert result.word == occurrences(source)[0] and result.target == occurrences(target)[0]


def test_deleted_tail_walks_backward_to_end_of_kept_occurrence() -> None:
    source = _words(("a", "s1", 1.0, 2.0), ("b", "s1", 3.0, 4.0), ("c", "s1", 5.0, 6.0))
    target = _words(("a", "new", 0.0, 1.0))
    result = map_moment(source, target, 5.5)
    assert result.status == "removed" and result.time_s == 1.0
    assert result.target == occurrences(target)[0]


def test_deleted_search_skips_ambiguous_occurrences() -> None:
    source = _words(("a", "old", 0.0, 1.0), ("b", "old", 1.0, 2.0), ("c", "old", 2.0, 3.0))
    target = _words(("b", "new1", 0.0, 1.0), ("b", "new2", 1.0, 2.0), ("c", "new", 2.0, 3.0))
    result = map_moment(source, target, 0.5)
    assert result.status == "removed" and result.target == occurrences(target)[2]
    assert result.time_s == 2.0


def test_no_shared_id_is_removed_even_when_word_text_and_seconds_match() -> None:
    source = CutWords("output", {}, (CutWord("a", "s1", "same", 1.0, 2.0),))
    target = CutWords("output", {}, (CutWord("b", "s1", "same", 1.0, 2.0),))
    result = map_moment(source, target, 1.5)
    assert result.status == "removed" and result.time_s == 0.0
    assert result.target is None


@pytest.mark.parametrize("missing", ["source", "target", "empty_source", "empty_target"])
def test_no_word_map_available_falls_back_to_clamped_time(missing: str) -> None:
    source = _words(("a", "s1", 0.0, 1.0))
    target = source
    if missing == "source":
        source = None
    elif missing == "target":
        target = None
    elif missing == "empty_source":
        source = _words()
    else:
        target = _words()
    result = map_moment(source, target, 8.0, target_duration=4.0)
    assert result.status == "time" and result.time_s == 4.0
    assert result.word is None and result.target is None and result.candidates == ()
    assert map_moment(source, target, -1.0).time_s == 0.0
    assert occurrence_at(_words(), 1.0) == (None, 0.0, "none")


@pytest.mark.parametrize("case", ["mapped", "moved", "ambiguous", "removed"])
def test_all_destination_statuses_clamp_to_duration(case: str) -> None:
    source = _words(("a", "s1", 1.0, 2.0), ("b", "s1", 3.0, 4.0))
    targets = {
        "mapped": _words(("a", "s1", 6.0, 7.0)),
        "moved": _words(("a", "new", 6.0, 7.0)),
        "ambiguous": _words(("a", "new1", 6.0, 7.0), ("a", "new2", 8.0, 9.0)),
        "removed": _words(("b", "s1", 6.0, 7.0)),
    }
    result = map_moment(source, targets[case], 1.5, target_duration=3.0)
    assert result.status == case and result.time_s == 3.0
