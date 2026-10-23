"""Output-clock anchors retain kept-word and repeated-occurrence semantics."""

from __future__ import annotations

import pytest

from yap_studio.errors import ValidationError
from yap_studio.planner.anchors import frame_at, resolve_anchor
from yap_studio.project.model import Anchor, CutWord, CutWords


def synthetic_words() -> CutWords:
    """Two segments with a reused cold-open word and a removed source word."""
    words = [CutWord("t1:12", "cold-open", "percent", 0, 0.5, 1)]
    words.extend(
        CutWord(f"t2:{i}", "cold-open", f"other{i}", 0.5 + i * 0.3, 0.8 + i * 0.3, 1)
        for i in range(8)
    )
    words.extend(
        CutWord(
            f"t1:{i}",
            "story",
            "percent" if i == 12 else f"word{i}",
            6.4 + i * 0.5,
            6.9 + i * 0.5,
            1,
        )
        for i in range(30)
        if i != 13
    )
    return CutWords("output", {"t1": "a" * 64, "t2": "b" * 64}, tuple(words))


def resolve(anchor: Anchor, *, start: bool = True) -> float:
    return resolve_anchor(anchor, synthetic_words(), start=start, location="cards[2].start")


def test_word_edges_and_offsets() -> None:
    assert resolve(Anchor(word="t1:10")) == pytest.approx(11.4)
    assert resolve(Anchor(word="t1:10", edge="end"), start=False) == pytest.approx(11.9)
    assert resolve(Anchor(word="t1:10", edge="end", offset_s=0.25)) == pytest.approx(12.15)
    assert resolve(Anchor(word="t1:10", offset_s=-0.25)) == pytest.approx(11.15)
    # Explicit edges are preserved even when used for the other window end.
    assert resolve(Anchor(word="t1:10", edge="start"), start=False) == pytest.approx(11.4)


def test_time_anchor() -> None:
    assert resolve(Anchor(time=2.125)) == 2.125


def test_guard_uses_anchor_normalization() -> None:
    assert resolve(Anchor(word="t1:12", segment="story", text="PERCENT!")) == pytest.approx(12.4)
    with pytest.raises(ValidationError, match="guard 'points'.*'percent'") as error:
        resolve(Anchor(word="t1:12", segment="story", text="points"))
    assert error.value.parameter == "cards[2].start.text"


def test_removed_word_names_neighbours_in_same_take() -> None:
    with pytest.raises(ValidationError) as error:
        resolve(Anchor(word="t1:13"))
    assert error.value.parameter == "cards[2].start.word"
    message = error.value.detail
    assert "removed" in message and "t1:13" in message
    assert "before t1:12 ('percent')" in message
    assert "after t1:14 ('word14')" in message
    assert "t2:" not in message


@pytest.mark.parametrize(
    "word, before, after",
    [
        ("t1:40", "t1:29", "none"),
        ("t3:0", "none", "none"),
    ],
)
def test_missing_neighbours_are_explicit(word: str, before: str, after: str) -> None:
    with pytest.raises(ValidationError) as error:
        resolve(Anchor(word=word))
    assert f"before {before}" in error.value.detail
    assert f"after {after}" in error.value.detail


def test_ambiguous_occurrence_and_segment_guard() -> None:
    with pytest.raises(ValidationError, match="ambiguous.*cold-open.*story"):
        resolve(Anchor(word="t1:12"))
    assert resolve(Anchor(word="t1:12", segment="cold-open")) == 0
    assert resolve(Anchor(word="t1:12", segment="story")) == pytest.approx(12.4)
    with pytest.raises(ValidationError, match="no occurrence.*absent"):
        resolve(Anchor(word="t1:12", segment="absent"))


def test_snap_needs_the_timeline_stage() -> None:
    with pytest.raises(ValidationError, match="snap needs the timeline stage"):
        resolve(Anchor(word="t1:13", snap="nearest"))


@pytest.mark.parametrize(
    "seconds, fps, expected",
    [
        (0, 30, 0),
        (0.049, 30, 1),
        (0.05, 30, 2),
        (2.5, 1, 3),
        (12.4, 30, 372),
        (14.9, 30, 447),
        (-0.05, 30, -1),
    ],
)
def test_frame_at_verify_rounding(seconds: float, fps: int, expected: int) -> None:
    assert frame_at(seconds, fps) == expected
