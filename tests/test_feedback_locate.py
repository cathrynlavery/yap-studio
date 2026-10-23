"""Review spans use exact frames and occurrence identities on their own cut."""

from __future__ import annotations

from fractions import Fraction

import pytest

from yap_studio.feedback.locate import locate
from yap_studio.feedback.versions import LayerWindow
from yap_studio.project.model import CutWord, CutWords
from yap_studio.review.model import Comment

STAMP = "2026-10-05T12:00:00Z"
WORDS = CutWords("output", {}, (
    CutWord("t1:0", "s1", "Start", 0.2, 0.5),
    CutWord("t1:1", "s1", "really", 0.8, 1.1),
    CutWord("t1:0", "s2", "Start", 2.2, 2.5),
    CutWord("t1:2", "s2", "End", 2.8, 3.1),
))
PIECES = {"fps": "30/1", "pieces": [
    {"segment": "s1", "out_f0": 0, "n_frames": 60},
    {"segment": "s2", "out_f0": 60, "n_frames": 60},
]}
LAYERS = (
    LayerWindow("card", "graphic", 2.1, 2.7, None),
    LayerWindow("caption", "caption", 2.1, 3.2, "Start End"),
)


def _comment(
    comment_id="c1", *, kind="fix", target=None, start_frame=69, end_frame=None,
    start_s=2.3, end_s=None,
) -> Comment:
    return Comment(comment_id, kind, target, "Preserve this material", start_frame, end_frame,
                   start_s, end_s, STAMP, STAMP)


def _locate(comment: Comment, *, fps=Fraction(30), layers=LAYERS, words=WORDS):
    return locate(comment, fps=fps, duration_s=4.0, words=words, pieces=PIECES, layers=layers)


def test_card_point_and_kept_range_locate_occurrences_segments_and_layers() -> None:
    point = _locate(_comment(target="element", start_s=99.0))
    assert (point.status, point.start_frame, point.end_frame) == ("located", 69, 70)
    assert point.start_s == 2.3 and point.end_s == pytest.approx(70 / 30)
    assert [(word.id, word.segment) for word in point.words] == [("t1:0", "s2")]
    assert point.segments == ("s2",)
    assert [layer.id for layer in point.layers] == ["card", "caption"]
    kept = _locate(_comment(kind="keep", start_frame=66, end_frame=93))
    assert [(word.id, word.segment) for word in kept.words] == [("t1:0", "s2"), ("t1:2", "s2")]
    data = kept.to_dict()
    assert data["words"][0] == {
        "id": "t1:0", "segment": "s2", "text": "Start", "start": 2.2, "end": 2.5,
    }
    assert data["segments"] == ["s2"] and data["layers_known"] is True
    assert data["layers"][0] == {
        "id": "card", "kind": "graphic", "start": 2.1, "end": 2.7, "text": None,
    }


def test_silence_keeps_both_neighbours_and_piece() -> None:
    location = _locate(_comment(kind="keep", target="speech", start_frame=18))
    assert location.status == "silence" and location.words == ()
    assert location.before.id == "t1:0" and location.before.segment == "s1"
    assert location.after.id == "t1:1" and location.after.segment == "s1"
    assert location.after.start - location.before.end == pytest.approx(0.3)
    assert location.segments == ("s1",)
    assert location.to_dict()["before"]["end"] == 0.5


@pytest.mark.parametrize("time,expected", [(0.5, []), (0.8, ["t1:1"]), (2.2, ["t1:0"])])
def test_variable_rate_instant_half_open_boundaries(time: float, expected: list[str]) -> None:
    result = _locate(_comment(start_frame=None, start_s=time), fps=None)
    assert result.start_s == result.end_s == time
    assert result.start_frame is result.end_frame is None
    assert [word.id for word in result.words] == expected


def test_variable_rate_range_uses_piece_fps_and_layer_boundaries() -> None:
    result = _locate(_comment(start_frame=None, start_s=1.9, end_s=2.2), fps=None)
    assert result.words == () and result.segments == ("s1", "s2")
    assert [layer.id for layer in result.layers] == ["card", "caption"]
    assert result.after.segment == "s2"
    result = _locate(_comment(start_frame=None, start_s=2.7), fps=None)
    assert [layer.id for layer in result.layers] == ["caption"]


def test_whole_video_outside_unknown_and_empty_layers() -> None:
    whole = _locate(_comment(target="video"))
    assert whole.status == "whole_video" and whole.words and whole.layers
    assert _locate(_comment(start_frame=120)).status == "outside"
    assert _locate(_comment(target="video", start_frame=121)).status == "outside"
    assert _locate(_comment(), layers=None).to_dict()["layers_known"] is False
    assert _locate(_comment(), layers=()).layers_known is True
    assert _locate(_comment(start_frame=0)).before is None
    assert _locate(_comment(start_frame=110)).after is None


def test_fractional_fps_frames_are_authoritative() -> None:
    fps = Fraction(30000, 1001)
    result = _locate(_comment(start_frame=66, end_frame=75, start_s=0.0), fps=fps)
    assert result.start_s == float(Fraction(66) / fps)
    assert [(word.id, word.segment) for word in result.words] == [("t1:0", "s2")]
