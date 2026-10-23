"""Synthetic opening push timing and picture curve contracts."""

import pytest

from yap_studio import opening_push
from yap_studio.opening_push import OpeningEnd, opening_end
from yap_studio.project.model import CutWord
from yap_studio.zoom_curve import ZoomCurve, declared_moves, push_keys, scale_motion


@pytest.mark.parametrize("rows,cuts,expected", [
    ([("Hello.", 0, .9)], [], (27, "t1:0", "Hello.", "line")),
    ([("Long", 0, 1), ("line.", 1.1, 3)], [], (90, "t1:1", "line.", "line")),
    ([("Long", 0, 1), ("line", 1.2, 1.9), ("ends.", 2, 3)], [45],
     (45, "t1:1", "line", "cut")),
    ([("First,", 0, 1.2), ("clause", 1.3, 3), ("ends.", 4.1, 5)], [],
     (36, "t1:0", "First,", "clause")),
    ([("No", 0, 1), ("boundary", 1.1, 2.5), ("here", 2.6, 5)], [], None),
    ([("First.", .2, 4.2)], [], (126, "t1:0", "First.", "line")),
    ([("Late.", .2, 4.24)], [], None),
    ([("Early.", .2, .96)], [], None),
    ([("First.", .2, 1)], [], (30, "t1:0", "First.", "line")),
    ([("First,", 0, 1.2), ("ends.", 1.3, 3)], [], (90, "t1:1", "ends.", "line")),
    ([("First,", 0, 1.2), ("clause", 1.3, 5)], [90], (90, "t1:1", "clause", "cut")),
    ([("Early,", 0, .4), ('First;")', .5, 1), ("next:", 1.1, 2), ("ends.", 4.1, 5)], [],
     (30, "t1:1", 'First;")', "clause")),
    ([("Hi!", 0, .4), ('Again?")', .5, 1.5)], [], (45, "t1:1", 'Again?")', "line")),
    ([], [30], None),
    ([("Hi.", 0, .4), ("Next", .5, 5)], [12, 76], (76, "t1:1", "Next", "cut")),
])
def test_opening_end(rows, cuts, expected) -> None:
    words = tuple(CutWord(f"t1:{index}", "s1", text, start, end)
                  for index, (text, start, end) in enumerate(rows))
    assert opening_end(words, cuts) == (OpeningEnd(*expected) if expected else None)


@pytest.mark.parametrize("text,reason", [("Line.", "line"), ("Line,", "clause"), ("Line", None)])
@pytest.mark.parametrize("prior_word", [False, True])
def test_opening_late_pause_cut_keeps_boundary(text: str, reason: str | None,
                                              prior_word: bool) -> None:
    rows = ([("Earlier", .1, 1), (text, 1.1, 2.3)] if prior_word else [(text, .1, 2.3)])
    rows.append(("Next", 4.6, 5))
    words = tuple(CutWord(f"t1:{index}", "s1", value, start, stop)
                  for index, (value, start, stop) in enumerate(rows))
    expected = OpeningEnd(69, f"t1:{int(prior_word)}", text, reason) if reason else None
    assert opening_end(words, [132]) == expected


@pytest.mark.parametrize("text,cuts,end,reason", [
    ("Hello.", [36], 36, "line"),
    ("Hello", [36], 36, "cut"),
    ("Hello.", [36, 45], 36, "line"),
    ("Hello", [36, 45], 36, "cut"),
    ("Hello.", [81], 81, "line"),
    ("Hello", [81], 81, "cut"),
    ("Hello.", [132], 33, "line"),
    ("Hello", [132], None, None),
])
def test_opening_last_word_pause_cut(text: str, cuts: list[int], end: int | None,
                                    reason: str | None) -> None:
    word = CutWord("t1:0", "s1", text, .1, 1.1)
    expected = OpeningEnd(end, word.id, text, reason) if end is not None else None
    assert opening_end((word,), cuts) == expected


@pytest.mark.parametrize("end,target", [(27, 1.3), (31, 1.3), (90, 1.3), (27, 1.5), (90, 1.5)])
def test_opening_scale_curve(end: int, target: float) -> None:
    maximum = opening_push.end_scale(end, target)
    expected = min(target, target ** (end / 31))
    assert maximum == pytest.approx(expected)
    curve = ZoomCurve(0, end, push_keys(0, end, maximum, (.5, .42), "linear"))
    scales = curve.scales(0, end + 1)
    assert scales[0] == scales[end] == 1
    ratio = target ** (1 / 31) if end <= 31 else target ** (1 / end)
    ratios = [b / a for a, b in zip(scales[:end - 1], scales[1:end], strict=True)]
    assert ratios == pytest.approx([ratio] * (end - 1))
    if end < 31:
        assert maximum < target
    elif end > 31:
        assert maximum == target and ratio < target ** (1 / 31)
    assert scales[end - 2] < scales[end - 1] < maximum
    assert scale_motion(scales, 0, declared_moves(curve)) == ()
