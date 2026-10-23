"""Source word anchors respect the picture clock."""

from fractions import Fraction

import pytest

from yap_studio.project.anchors import nominal_edge
from yap_studio.project.model import Anchor, ProjectFileError, Segment, Word, Words


@pytest.mark.parametrize("offset", [None, 0])
@pytest.mark.parametrize("edge,start", [("end", False), ("end", True), ("start", False)])
def test_word_crossing_picture_end_clamps_only_the_overrunning_edge(
    offset: float | None, edge: str, start: bool,
) -> None:
    anchor = Anchor(word="t1:0", edge=edge, **({"offset_s": offset} if offset is not None else {}))
    segment = Segment("ending", "t1", Anchor(time=0), anchor)
    words = Words("t1", "1" * 64, "source", 0, "none", None, "2" * 64,
                  (Word("t1:0", "Strawberry.", 530.963, 533.283),))
    expected = Fraction("530.963") if edge == "start" else Fraction("533.280")
    assert nominal_edge(segment, anchor, words, Fraction("533.280"), start=start) == expected


@pytest.mark.parametrize("edge", ["start", "end"])
@pytest.mark.parametrize("start", [False, True])
@pytest.mark.parametrize("offset", [0, -0.1])
def test_collapsed_word_at_picture_end_is_not_an_anchor(edge, start, offset):
    anchor = Anchor(word="t1:0", edge=edge, offset_s=offset)
    segment = Segment("ending", "t1", Anchor(time=0), anchor)
    words = Words("t1", "1" * 64, "source", 0, "none", None, "2" * 64,
                  (Word("t1:0", "Later.", 3, 3),))
    with pytest.raises(ProjectFileError, match="anchor is outside the source picture"):
        nominal_edge(segment, anchor, words, Fraction(3), start=start)


@pytest.mark.parametrize("anchor,word_start", [
    (Anchor(word="t1:0", edge="end", offset_s=0.5), 530.963),
    (Anchor(time=533.283), 530.963),
    (Anchor(word="t1:0", edge="end"), 533.281),
    (Anchor(word="t1:0", edge="start"), 533.281),
    (Anchor(word="t1:0", edge="end", offset_s=-0.001), 530.963),
])
def test_anchors_outside_picture_still_raise(anchor: Anchor, word_start: float) -> None:
    segment = Segment("ending", "t1", Anchor(time=0), anchor)
    words = Words("t1", "1" * 64, "source", 0, "none", None, "2" * 64,
                  (Word("t1:0", "Strawberry.", word_start, 533.283),))
    with pytest.raises(ProjectFileError, match="anchor is outside the source picture"):
        nominal_edge(segment, anchor, words, Fraction("533.280"), start=False)
