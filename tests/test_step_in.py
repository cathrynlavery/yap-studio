"""Pure framing ladders on synthetic cut pieces."""

from dataclasses import asdict

import pytest

from yap_studio.faces.model import Box
from yap_studio.joins.steps import LADDER, jumps, plan_steps


def pieces(lengths=(30, 30, 30, 30, 30, 30)):
    result = []
    output = source = 0
    for length in lengths:
        result.append(dict(segment_id="s1", take="t1", src_f0=source,
                           src_f1=source + length, out_f0=output, n_frames=length))
        output += length
        source += length + 4
    return result


def plan(parts=None, *, words=(), covers=(), boxes="face", cards=(), card_margin=24,
         wide_from=()):
    parts = pieces() if parts is None else parts
    total = sum(p["n_frames"] for p in parts)
    if boxes == "face":
        boxes = {i: (Box(400, 400, 200, 250),) for i in range(total)}
    return plan_steps(parts, words, total, covers, boxes, cards=cards, card_margin=card_margin,
                      bottom=1500, bottom_margin=40, wide_from=wide_from)


def test_ladder_reset_summary_and_determinism():
    result = plan()
    assert LADDER == (1.15, 1.3, 1.45)
    assert [s.scale for s in result.shots] == [1, 1.15, 1.3, 1.45, 1, 1.15]
    assert result.summary() == dict(jumps=5, changed=5, step=4, reset=1, shift=0, hold=0)
    assert result.jumps[3].reason == "Back to wide after three steps."
    assert result == plan()
    assert asdict(result)["shots"][0]["center"] is None


def test_wide_from_resets_and_restarts():
    reason = "The opening push cuts back wide here."
    result = plan(wide_from=[(60, reason)])
    assert [s.level for s in result.shots[:4]] == [0, 1, 0, 1]
    assert result.shots[2].scale == 1
    assert result.shots[2].center is None
    assert result.shots[2].reason == reason
    assert result.jumps[1].handling == "reset"
    assert result.jumps[1].reason == reason
    assert result.shots[3].scale == 1.15


@pytest.mark.parametrize("kind,levels,reason", [
    ("story", [0, 1, 2, 3], "Steps in to 1.3x."),
    ("take", [0, 1, 0, 1], "The take changes here."),
])
def test_story_join_steps_and_take_change_resets(kind, levels, reason):
    parts = pieces()
    for part in parts[2:]:
        part["segment_id" if kind == "story" else "take"] = "new"
    result = plan(parts)
    assert [shot.level for shot in result.shots[:4]] == levels
    assert result.jumps[1].kind == kind
    assert result.jumps[1].reason == reason


@pytest.mark.parametrize("gap,scale", [(0.5, 1), (0.49, 1.15)])
def test_pause_boundary(gap, scale):
    words = [dict(start=0, end=0.75), dict(start=0.75 + gap, end=1.8)]
    assert plan(words=words).shots[1].scale == scale


def test_cut_word_pause_matches_dicts():
    from yap_studio.project.model import CutWord

    words = [CutWord("t1:0", "s1", "before", 0, .75),
             CutWord("t1:1", "s1", "after", 1.25, 1.8)]
    assert plan(words=words) == plan(words=[asdict(w) for w in words])


def test_short_shot_keeps_center():
    result = plan(pieces((30, 30, 14, 30)))
    assert [s.scale for s in result.shots] == [1, 1.15, 1.15, 1.3]
    assert result.shots[2].center == result.shots[1].center
    assert result.jumps[1].handling == "hold"
    assert result.shots[2].reason == "The next shot is shorter than 0.5 s."


def test_short_shot_checks_previous_center():
    boxes = {i: (Box(400 if i < 60 else 0, 400, 200, 250),) for i in range(104)}
    result = plan(pieces((30, 30, 14, 30)), boxes=boxes)
    assert result.shots[2].level == 0
    assert result.shots[2].reason == "The face would leave the safe area at 1.15x."


@pytest.mark.parametrize("kind", ["zoom", "insert", "overlay", "background", "transition",
                                  "moment", "grade_compare", "graphic"])
def test_cover_resets_and_restarts(kind):
    result = plan(covers=[(61, 62, "cover", kind)])
    assert [s.level for s in result.shots[:4]] == [0, 1, 0, 1]
    assert result.shots[2].reason == f"A {kind} covers this shot (cover)."


@pytest.mark.parametrize("shot", [2, 3])
def test_card_caps_step_at_lower_level(shot):
    result = plan(cards=[(shot * 30, (shot + 1) * 30, "card", Box(0, 0, 1080, 180))])
    assert result.shots[shot].scale == 1.15
    assert result.shots[shot + 1].scale == 1.3


def test_card_cap_keeps_previous_center_when_it_fits():
    boxes = {i: (Box(400 if i < 60 else 450, 400, 200, 250),) for i in range(90)}
    result = plan(pieces((30, 30, 30)), boxes=boxes,
                  cards=[(60, 90, "card", Box(0, 0, 1080, 180))])
    assert result.shots[2].level == result.shots[1].level == 1
    assert result.shots[2].center == result.shots[1].center
    assert result.jumps[1].handling == "hold"
    assert result.summary() == dict(jumps=2, changed=1, step=1, reset=0, shift=0, hold=1)


@pytest.mark.parametrize("face", [Box(0, 400, 200, 250), Box(400, 389, 200, 250)])
def test_card_cap_shifts_when_previous_center_breaks_face_or_card_clearance(face):
    boxes = {i: (Box(400, 400, 200, 250) if i < 60 else face,) for i in range(90)}
    result = plan(pieces((30, 30, 30)), boxes=boxes,
                  cards=[(60, 90, "card", Box(0, 0, 1080, 180))])
    assert result.shots[2].level == result.shots[1].level == 1
    assert result.shots[2].center != result.shots[1].center
    assert result.jumps[1].handling == "shift"
    assert result.summary() == dict(jumps=2, changed=2, step=1, reset=0, shift=1, hold=0)


def test_card_keeps_shot_wide():
    result = plan(cards=[(60, 90, "card", Box(0, 0, 1080, 200))])
    assert result.shots[2].scale == 1
    assert result.shots[2].center is None
    assert result.shots[2].reason == "A card needs the room around your head (card)."
    assert result.jumps[1].reason == result.shots[2].reason
    assert result.shots[3].scale == 1.15


@pytest.mark.parametrize("start,end", [(0, 30), (90, 120)])
def test_card_outside_shot_changes_nothing(start, end):
    parts = pieces((30, 30, 30))
    assert plan(parts, cards=[(start, end, "card", Box(0, 0, 1080, 200))]) == plan(parts)


def test_card_short_shot_goes_wide_instead_of_lowering_level():
    result = plan(pieces((30, 30, 30, 14, 30)),
                  cards=[(90, 104, "card", Box(0, 0, 1080, 180))])
    assert result.shots[2].scale == 1.3
    assert result.shots[3].scale == 1
    assert result.shots[3].reason == "A card needs the room around your head (card)."


@pytest.mark.parametrize("face_frame,scale", [(89, 1.15), (90, 1.3)])
def test_card_checks_every_face_on_overlapping_frames(face_frame, scale):
    boxes = {i: (Box(400, 400, 200, 250),) for i in range(180)}
    boxes[face_frame] += (Box(410, 300, 100, 100),)
    result = plan(boxes=boxes, cards=[(60, 90, "card", Box(0, 0, 1080, 150))])
    assert result.shots[2].scale == scale


@pytest.mark.parametrize("boxes", [None, {}])
def test_no_faces(boxes):
    result = plan(boxes=boxes)
    assert all(s.level == 0 for s in result.shots)
    assert all(s.reason == "No face was measured in this shot." for s in result.shots[1:])
    assert result.summary() == dict(jumps=5, changed=0, step=0, reset=0, shift=0, hold=5)


def test_face_cannot_fit():
    result = plan(boxes={i: (Box(0, 0, 1080, 1920),) for i in range(180)})
    assert all(s.level == 0 for s in result.shots)
    assert result.shots[1].reason == "The face would leave the safe area at 1.15x."


def test_jumps_ignore_contiguous_pieces():
    parts = pieces((30, 30, 30))
    parts[1]["src_f0"] = parts[0]["src_f1"]
    assert list(jumps(parts)) == [(60, "removal", parts[1], parts[2])]
