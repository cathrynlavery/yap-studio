"""Feedback operations produce pure story plans and deferred fillers actions."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from yap_studio.errors import ValidationError
from yap_studio.feedback.batch import FeedbackError
from yap_studio.feedback.edits import (
    OPERATIONS_SCHEMA,
    TIMELINE_KINDS,
    EditPlan,
    NeedsTimelineError,
    RemovalAction,
    _Planner,
    plan_edit,
    validate_operations,
)
from yap_studio.project.model import CutWord, CutWords, Edit, Word, Words, parse_document


@pytest.fixture
def base() -> dict:
    texts = ("Start", "um,", "so", "about.", "and", "that", "is", "end.")
    source = Words("t1", "a" * 64, "source", 0, "none", None, "b" * 64, tuple(
        Word(f"t1:{index}", text, index * 2 + 0.5, index * 2 + 1.0)
        for index, text in enumerate(texts)
    ))
    second = Words("t2", "c" * 64, "source", 0, "none", None, "d" * 64, (
        Word("t2:0", "Other", 30.0, 30.5), Word("t2:1", "take.", 31.0, 31.5),
    ))
    edit = {"schema": "yap-studio/edit@1", "words_sha256": {"t1": "e" * 64, "t2": "f" * 64},
            "segments": [
                {"id": "s1", "take": "t1", "from": {"word": "t1:0", "text": "Start"},
                 "to": {"word": "t1:3"}, "text": "Start um, so about.", "note": "opening"},
                {"id": "s2", "take": "t1", "from": {"word": "t1:7"},
                 "to": {"word": "t1:7", "edge": "end", "offset_s": 0.1}, "note": "closing"},
                {"id": "s3", "take": "t2", "from": {"time": 29.9},
                 "to": {"word": "t2:1"}, "text": "Other take."},
            ], "end_hold_s": 0.75}
    kept = [("s1", word) for word in source.words[:4]] + [("s2", source.words[7])]
    kept += [("s3", word) for word in second.words]
    cut_words = CutWords("output", dict(edit["words_sha256"]), tuple(
        CutWord(word.id, segment, word.text, index + 0.1, index + 0.4)
        for index, (segment, word) in enumerate(kept)
    ))
    restored = [{"id": "t1:4", "text": "and"}, {"id": "t1:5", "text": "that"},
                {"id": "t1:6", "text": "is"}]
    removed = {"id": "s1.0", "take": "t1", "start": 7.2, "end": 14.0,
               "kind": "story", "removal_ids": [], "words": restored,
               "text": "and that is", "words_known": True}
    pieces = {"v1": {"pieces": [
        {"segment": "s1", "take": "t1", "src_t0": "0", "src_t1": "36/5",
         "out_f0": 0, "n_frames": 120},
        {"segment": "s2", "take": "t1", "src_t0": "14", "src_t1": "15",
         "out_f0": 120, "n_frames": 30},
    ]}}
    return {"edit": edit, "cut_words": cut_words, "source_words": {"t1": source, "t2": second},
            "items": [
                {"comment": {"version": "v1", "id": "c1", "kind": "fix"}, "request": None},
                {"comment": {"version": "v1", "id": "c2", "kind": "keep"}, "request": None},
            ], "ranges": {"v1": [removed]}, "pieces": pieces}


def _document(*operations: dict, comment: str = "v1:c1") -> dict:
    return {"schema": OPERATIONS_SCHEMA, "batch": "b1",
            "items": [{"comment": comment, "operations": list(operations)}]}


def _remove(first: int, last: int, *, operation: str = "cut", segment: str = "s1") -> dict:
    return {"id": operation, "kind": "remove",
            "from": {"word": f"t1:{first}", "segment": segment},
            "to": {"word": f"t1:{last}", "segment": segment}}


def _trim(word: str, edge: str, *, segment: str = "s1", operation: str = "trim") -> dict:
    return {"id": operation, "kind": "trim", "segment": segment, "edge": edge, "word": word}


def _plan(base: dict, *operations: dict, comment: str = "v1:c1") -> EditPlan:
    document = validate_operations(_document(*operations, comment=comment))
    return plan_edit(**base, operations=document)


def _validation(error: ValidationError) -> str:
    message = f"{error.parameter}: {error.detail}"
    assert str(error) == f"Bad value for {message}"
    assert error.code == "invalid_parameter"
    return message


def _edit(base: dict, *, segment: str = "s1", drop: bool = False, **anchors: dict) -> dict:
    edit = copy.deepcopy(base["edit"])
    if drop:
        edit["segments"] = [item for item in edit["segments"] if item["id"] != segment]
    else:
        next(item for item in edit["segments"] if item["id"] == segment).update(anchors)
    return edit


def _change(operation: str, kind: str, summary: str, words: list[dict],
            *, segments: list[str] | None = None, comment: str = "v1:c1") -> dict:
    return {"operation": operation, "comment": comment, "kind": kind, "summary": summary,
            "segments": ["s1"] if segments is None else segments, "words": words}


def _words(base: dict, *indices: int) -> list[dict]:
    source = base["source_words"]["t1"].words
    return [{"id": source[index].id, "text": source[index].text} for index in indices]


def _cut_request(base: dict, *occurrences: tuple[str, str]) -> None:
    base["items"][0]["request"] = {
        "type": "cut", "start": 0.1, "end": 99,
        "words": [{"id": word.id, "segment": segment, "text": word.text}
                  for segment, identifier in occurrences for word in base["cut_words"].words
                  if word.segment == segment and word.id == identifier],
    }


def test_cut_validates_only_id_and_kind() -> None:
    document = _document({"id": "cut", "kind": "cut"})
    assert validate_operations(document) == document
    for extra in ({"segment": "s1"}, {"from": 1}, {"request": {}}):
        with pytest.raises(ValidationError, match="expected exactly these keys"):
            validate_operations(_document({"id": "cut", "kind": "cut", **extra}))


@pytest.mark.parametrize("indices,edge,anchors,bounds", [
    ((1, 2), None, {"text": "Start about."}, (1.75, 5.75)),
    ((0, 1), "start", {"from": {"word": "t1:2"}, "text": "so about."}, (0.38, 3.75)),
    ((2, 3), "end", {"to": {"word": "t1:1"}, "text": "Start um,"}, (3.75, 7.2)),
])
def test_cut_request_shapes_and_source_midpoints(base, indices, edge, anchors, bounds) -> None:
    _cut_request(base, *(("s1", f"t1:{index}") for index in indices))
    before = copy.deepcopy(base)
    plan = _plan(base, {"id": "cut", "kind": "cut"})
    assert base == before
    assert plan.edit == _edit(base, **anchors)
    span, = plan.cut_spans
    assert {key: span[key] for key in ("operation", "comment", "segment", "take", "edge",
                                      "words", "texts", "action")} == {
        "operation": "cut", "comment": "v1:c1", "segment": "s1", "take": "t1",
        "edge": edge, "words": [f"t1:{index}" for index in indices],
        "texts": [word["text"] for word in _words(base, *indices)],
        "action": 0 if edge is None else None,
    }
    assert (span["start"], span["end"]) == pytest.approx(bounds)
    assert len(plan.removal_actions) == (1 if edge is None else 0)
    if edge is None:
        action, = plan.removal_actions
        assert action.action == "add_manual" and action.operation == "cut"
        assert (action.start, action.end) == bounds
    assert plan.changes[0]["summary"].startswith(
        'Cut "' + " ".join(span["texts"]) + '" (2 words): ')


def test_cut_all_words_drops_segment_without_span(base) -> None:
    _cut_request(base, *(("s1", f"t1:{index}") for index in range(4)))
    plan = _plan(base, {"id": "cut", "kind": "cut"})
    assert plan.edit == _edit(base, drop=True)
    assert plan.cut_spans == plan.removal_actions == ()


def test_cut_spans_restore_pre_cut_anchors_and_options_skip_own_action(base) -> None:
    _cut_request(base, ("s1", "t1:1"))
    plan = _plan(base, _trim("t1:1", "start"), {"id": "cut", "kind": "cut"})
    span, = plan.cut_spans
    assert span["restore"] == {"edge": "start", "anchor": {"word": "t1:1"}}
    assert plan.edit["segments"][0]["from"] == {"word": "t1:2"}
    assert all(option["restore"] == span["restore"] for option in span["options"])
    _cut_request(base, ("s1", "t1:1"), ("s1", "t1:2"))
    span, = _plan(base, {"id": "cut", "kind": "cut"}).cut_spans
    assert span["restore"] is None and span["action"] == 0
    assert all(option["action"] == 0 for option in span["options"])


@pytest.mark.parametrize("edge", ["start", "end"])
@pytest.mark.parametrize("later_kind", ["trim", "remove"])
def test_edge_cut_combines_later_moves_at_the_final_boundary(base, edge, later_kind) -> None:
    index, boundary, extra = (0, 2, 1) if edge == "start" else (3, 1, 2)
    _cut_request(base, ("s1", f"t1:{index}"))
    later = (_trim(f"t1:{boundary}", edge, operation="more") if later_kind == "trim" else
             _remove(extra, extra, operation="more"))
    span, = _plan(base, {"id": "cut", "kind": "cut"}, later).cut_spans
    assert span["words"] == (["t1:0", "t1:1"] if edge == "start" else ["t1:2", "t1:3"])
    assert span["texts"] == (["Start", "um,"] if edge == "start" else ["so", "about."])
    assert span["end" if edge == "start" else "start"] == 3.75
    assert span["restore"]["anchor"] == base["edit"]["segments"][0][
        "from" if edge == "start" else "to"]
    assert span["anchor_after"] == {"word": f"t1:{extra}"}
    assert span["kept_before"] == [f"t1:{at}" for at in range(4)]
    assert span["combined_with"] == ["more"] and span["options"] == []


def test_adjacent_prefix_cut_notes_test_the_same_final_boundary(base) -> None:
    _cut_request(base, ("s1", "t1:0"))
    base["items"].append({"comment": {"version": "v1", "id": "c3", "kind": "fix"},
                          "request": {"type": "cut", "start": 1, "end": 2, "words": [
                              {"id": "t1:1", "segment": "s1", "text": "um,"}]}})
    document = _document({"id": "cut1", "kind": "cut"})
    document["items"].append({"comment": "v1:c3", "operations": [{"id": "cut2", "kind": "cut"}]})
    plan = plan_edit(**base, operations=document)
    first, second = plan.cut_spans
    assert first["words"] == ["t1:0", "t1:1"] and second["words"] == ["t1:1"]
    assert first["end"] == second["end"] == 3.75
    assert first["combined_with"] == ["cut2"] and first["options"] == []
    assert first["restore"]["anchor"] == {"word": "t1:0", "text": "Start"}
    assert second["restore"]["anchor"] == {"word": "t1:1"}


def test_later_segment_drop_removes_cut_spans(base) -> None:
    _cut_request(base, ("s1", "t1:0"))
    plan = _plan(base, {"id": "cut", "kind": "cut"},
                 {"id": "drop", "kind": "remove", "segment": "s1"})
    assert plan.cut_spans == () and plan.edit == _edit(base, drop=True)


@pytest.mark.parametrize("kind,index", [("filler", 0), ("story", 3)])
def test_later_restore_of_cut_words_is_refused(base, kind, index) -> None:
    _cut_request(base, ("s1", f"t1:{index}"))
    base["ranges"]["v1"] = [{"id": "s1.0", "take": "t1", "start": 0.38, "end": 7.2,
                              "kind": kind, "removal_ids": ["r1"] if kind == "filler" else [],
                              "words": _words(base, index), "words_known": True}]
    with pytest.raises(FeedbackError, match="a later operation brings back words cut by this note"):
        _plan(base, {"id": "cut", "kind": "cut"},
              {"id": "back", "kind": "restore", "range": "s1.0"})


@pytest.mark.parametrize("edge,index,inner", [("start", 0, 1.75), ("end", 3, 5.75)])
def test_edge_cut_without_later_move_preserves_span_and_options(base, edge, index, inner) -> None:
    _cut_request(base, ("s1", f"t1:{index}"))
    operation = {"id": "cut", "kind": "cut"}
    planner = _Planner(base["edit"], base["cut_words"], base["source_words"])
    planner.cut(operation, "v1:c1", base["items"][0])
    span, = planner.cut_spans
    assert span["anchor_after"] == {"word": f"t1:{1 if edge == 'start' else 2}"}
    assert span["kept_before"] == [f"t1:{index}" for index in range(4)]
    assert span["end" if edge == "start" else "start"] == inner and span["options"]
    assert _plan(base, operation).cut_spans == (span,)


@pytest.mark.parametrize("edge,index,gap,boundary", [("start", 0, 2, 1), ("end", 3, 1, 2)])
def test_edge_cut_restore_past_tested_boundary_preserves_options(
    base, edge, index, gap, boundary,
) -> None:
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        word for word in base["cut_words"].words if word.segment != "s1" or word.id != f"t1:{gap}"))
    source = base["source_words"]["t1"].words[gap]
    base["ranges"]["v1"] = [{"id": "s1.0", "take": "t1", "start": source.start,
                              "end": source.end, "kind": "filler", "removal_ids": ["r1"],
                              "words": _words(base, gap), "words_known": True}]
    _cut_request(base, ("s1", f"t1:{index}"))
    operation = {"id": "cut", "kind": "cut"}
    original, = _plan(base, operation).cut_spans
    plan = _plan(base, operation, {"id": "back", "kind": "restore", "range": "s1.0"})
    span, = plan.cut_spans
    anchor = "from" if edge == "start" else "to"
    assert plan.edit["segments"][0][anchor] == {"word": f"t1:{boundary}"}
    assert span == original and span["options"]
    action, = plan.removal_actions
    assert action.action == "reject" and action.removal_id == "r1"


def _edge_gap(base, edge, *, extra_gap=False):
    index, gap = (0, 1) if edge == "start" else (4, 3)
    gaps = {gap, 3 if edge == "start" else 1} if extra_gap else {gap}
    source = base["source_words"]["t1"].words[:5]
    base["edit"]["segments"][0]["to"] = {"word": "t1:4"}
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        CutWord(word.id, "s1", word.text, at + 0.1, at + 0.4)
        for at, word in enumerate(word for word in source
                                  if word.id not in {f"t1:{index}" for index in gaps})
    ) + tuple(word for word in base["cut_words"].words if word.segment != "s1"))
    _cut_request(base, ("s1", f"t1:{index}"))
    base["ranges"]["v1"] = [{"id": "s1.0", "take": "t1", "start": source[gap].start,
                              "end": source[gap].end, "kind": "filler", "removal_ids": ["r1"],
                              "words": _words(base, gap), "words_known": True}]


@pytest.mark.parametrize("edge", ["start", "end"])
@pytest.mark.parametrize("trim", [False, True])
def test_edge_cut_includes_restored_gap_at_final_edge_word(base, edge, trim) -> None:
    _edge_gap(base, edge)
    operations = [{"id": "cut", "kind": "cut"},
                  {"id": "back", "kind": "restore", "range": "s1.0"}]
    if trim:
        operations.append(_trim(f"t1:{3 if edge == 'start' else 1}", edge))
    plan = _plan(base, *operations)
    span, = plan.cut_spans
    expected = ([0, 1, 2] if trim else [0, 1]) if edge == "start" else (
        [2, 3, 4] if trim else [3, 4])
    assert span["words"] == [f"t1:{index}" for index in expected]
    assert span["texts"] == [word["text"] for word in _words(base, *expected)]
    inner = (5.75 if trim else 3.75) if edge == "start" else (3.75 if trim else 5.75)
    assert span["end" if edge == "start" else "start"] == inner
    anchor = "from" if edge == "start" else "to"
    assert span["restore"]["anchor"] == base["edit"]["segments"][0][anchor]
    assert plan.edit["segments"][0][anchor] == {
        "word": f"t1:{(3 if edge == 'start' else 1) if trim else 2}"}
    assert span["options"] == []


@pytest.mark.parametrize("edge,inner", [("start", 3.75), ("end", 5.75)])
@pytest.mark.parametrize("extra_gap", [False, True])
def test_edge_cut_across_saved_gap_snaps_to_final_word_without_restore(
    base, edge, inner, extra_gap,
) -> None:
    _edge_gap(base, edge, extra_gap=extra_gap)
    span, = _plan(base, {"id": "cut", "kind": "cut"}).cut_spans
    assert span["end" if edge == "start" else "start"] == inner
    assert span["words"] == [f"t1:{0 if edge == 'start' else 4}"]
    option, = span["options"]
    assert option["edge"] == edge and option["label"] == 'also cut "so"'
    assert option["words"] == (["t1:0", "t1:2"] if edge == "start" else ["t1:2", "t1:4"])
    expected = (7.75 if extra_gap else 5.75) if edge == "start" else (1.75 if extra_gap else 3.75)
    assert option["end" if edge == "start" else "start"] == expected


def test_middle_cut_edge_options_snap_across_saved_gaps(base) -> None:
    _edge_gap(base, "start", extra_gap=True)
    _cut_request(base, ("s1", "t1:2"))
    span, = _plan(base, {"id": "cut", "kind": "cut"}).cut_spans
    assert span["edge"] is None
    assert [(option["edge"], option["start"], option["end"]) for option in span["options"]] == [
        ("start", 0.38, 7.75), ("end", 1.75, 9.2),
    ]


def test_cut_groups_segments_and_refreshes_all_text(base) -> None:
    _cut_request(base, ("s3", "t2:0"), ("s1", "t1:3"))
    plan = _plan(base, {"id": "cut", "kind": "cut"})
    assert plan.changes[0]["segments"] == ["s3", "s1"]
    assert [span["segment"] for span in plan.cut_spans] == ["s3", "s1"]
    assert plan.edit["segments"][0]["text"] == "Start um, so"
    assert plan.edit["segments"][2]["text"] == "take."


@pytest.mark.parametrize("problem,message", [
    ("missing", 'the words to cut are no longer in the cut: "um,"'),
    ("changed", 'the words changed since the note: "um,"'),
    ("separated", "the words to cut are not next to each other in segment s1"),
    ("reversed", "the words to cut are not next to each other in segment s1"),
    ("segment", 'the words to cut are no longer in the cut: "um,"'),
])
def test_cut_refuses_stale_or_nonadjacent_occurrences(base, problem, message) -> None:
    _cut_request(base, ("s1", "t1:1"), ("s1", "t1:2"))
    words = base["items"][0]["request"]["words"]
    if problem == "missing":
        words[0]["id"] = "t1:99"
    elif problem == "changed":
        base["cut_words"] = replace(base["cut_words"], words=tuple(
            replace(word, text="different") if word.id == "t1:1" else word
            for word in base["cut_words"].words))
    elif problem == "separated":
        words[1].update(id="t1:3", text="about.")
    elif problem == "reversed":
        words.reverse()
    else:
        words[0]["segment"] = "gone"
    with pytest.raises(FeedbackError, match=message):
        _plan(base, {"id": "cut", "kind": "cut"})


def test_cut_preserves_last_segment_and_overlap_refusals(base) -> None:
    _cut_request(base, ("s1", "t1:1"), ("s1", "t1:2"))
    base["edit"]["segments"][1].update(**{"from": {"time": 2}, "to": {"time": 5}})
    with pytest.raises(FeedbackError, match="would also cut segment s2"):
        _plan(base, {"id": "cut", "kind": "cut"})
    base["edit"]["segments"] = base["edit"]["segments"][:1]
    _cut_request(base, *(("s1", f"t1:{index}") for index in range(4)))
    with pytest.raises(FeedbackError, match="would leave no segments"):
        _plan(base, {"id": "cut", "kind": "cut"})


def test_cut_options_use_prechange_words_and_skip_whole_segment(base) -> None:
    _cut_request(base, ("s1", "t1:1"), ("s1", "t1:2"))
    span, = _plan(base, {"id": "cut", "kind": "cut"}).cut_spans
    assert [(option["label"], option["words"], option["edge"], option["start"], option["end"])
            for option in span["options"]] == [
        ('also cut "Start"', ["t1:0", "t1:1", "t1:2"], "start", 0.38, 5.75),
        ('also cut "about."', ["t1:1", "t1:2", "t1:3"], "end", 1.75, 7.2),
        ('keep "um,"', ["t1:2"], None, 3.75, 5.75),
        ('keep "so"', ["t1:1"], None, 1.75, 3.75),
    ]
    _cut_request(base, ("s3", "t2:0"))
    span, = _plan(base, {"id": "cut", "kind": "cut"}).cut_spans
    assert span["options"] == []


def test_cut_requires_request_and_fix_comment(base) -> None:
    with pytest.raises(ValidationError, match="cut needs the item's cut request"):
        _plan(base, {"id": "cut", "kind": "cut"})
    _cut_request(base, ("s1", "t1:1"))
    base["items"][0]["comment"]["kind"] = "keep"
    with pytest.raises(ValidationError, match="cut requires a fix comment"):
        _plan(base, {"id": "cut", "kind": "cut"})


@pytest.mark.parametrize("operation", [
    {"id": "drop", "kind": "remove", "segment": "s1"}, _remove(0, 3, operation="drop"),
])
def test_drop_whole_segment(base: dict, operation: dict) -> None:
    assert _plan(base, operation) == EditPlan(_edit(base, drop=True), (), (
        _change("drop", "remove", 'Dropped segment s1 (4 words: "Start um, so about.").',
                _words(base, 0, 1, 2, 3)),
    ))


@pytest.mark.parametrize("first,last,anchors,summary", [
    (0, 1, {"from": {"word": "t1:2"}, "text": "so about."},
     'Removed "Start um," from the start of s1.'),
    (2, 3, {"to": {"word": "t1:1"}, "text": "Start um,"},
     'Removed "so about." from the end of s1.'),
])
def test_remove_prefix_and_suffix(base: dict, first: int, last: int,
                                  anchors: dict, summary: str) -> None:
    assert _plan(base, _remove(first, last)) == EditPlan(_edit(base, **anchors), (), (
        _change("cut", "remove", summary, _words(base, *range(first, last + 1))),
    ))


def test_middle_remove_exact_source_midpoints(base: dict) -> None:
    assert _plan(base, _remove(1, 2)) == EditPlan(_edit(base, text="Start about."), (
        RemovalAction("add_manual", "cut", "v1:c1", 'Remove "um, so" for v1:c1.',
                      take="t1", start=1.75, end=5.75),
    ), (_change("cut", "remove", 'Removed "um, so" inside s1 (fillers stage).',
                _words(base, 1, 2)),))


def test_middle_remove_uses_source_neighbours_not_kept_neighbours(base: dict) -> None:
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        word for word in base["cut_words"].words if word.id != "t1:1"
    ))
    assert _plan(base, _remove(2, 2)) == EditPlan(_edit(base, text="Start about."), (
        RemovalAction("add_manual", "cut", "v1:c1", 'Remove "so" for v1:c1.',
                      take="t1", start=3.75, end=5.75),
    ), (_change("cut", "remove", 'Removed "so" inside s1 (fillers stage).', _words(base, 2)),))


@pytest.mark.parametrize("anchors", [
    {"from": {"word": "t1:1"}, "to": {"word": "t1:2"}},
    {"from": {"time": 5.5}, "to": {"word": "t1:3"}},
    {"from": {"word": "t1:0"}, "to": {"time": 1.8}},
    {"from": {"time": 1.8}, "to": {"time": 2.0}},
])
def test_middle_remove_refuses_other_segment_source_overlap(base: dict, anchors: dict) -> None:
    base["edit"]["segments"][1].update(anchors)
    before = copy.deepcopy(base)
    error = None
    try:
        _plan(base, _remove(1, 2))
    except FeedbackError as caught:
        error = caught
    assert error is not None, "a take-wide removal must protect every other segment"
    assert "Operation cut (v1:c1)" in str(error)
    assert "s2" in str(error)
    assert base == before


@pytest.mark.parametrize("anchors,overlaps", [
    pytest.param({"from": {"word": "t1:3", "offset_s": -3},
                  "to": {"word": "t1:7"}}, True, id="negative-start-offset"),
    pytest.param({"from": {"word": "t1:2", "offset_s": 3},
                  "to": {"word": "t1:7"}}, False, id="positive-start-offset"),
    pytest.param({"from": {"time": 0.5},
                  "to": {"word": "t1:2", "offset_s": -3}}, False, id="negative-end-offset"),
    pytest.param({"from": {"time": 0.5},
                  "to": {"word": "t1:1", "offset_s": 2}}, True, id="positive-end-offset"),
    pytest.param({"from": {"word": "t1:3", "edge": "end", "offset_s": -1.25},
                  "to": {"word": "t1:7"}}, False, id="explicit-start-edge-touch"),
    pytest.param({"from": {"time": 0.5},
                  "to": {"word": "t1:1", "edge": "start", "offset_s": 1.25}},
                 False, id="explicit-end-edge-touch"),
])
def test_middle_remove_resolves_other_segment_word_edges_and_offsets(
    base: dict, anchors: dict, overlaps: bool,
) -> None:
    base["edit"]["segments"][1].update(anchors)
    before = copy.deepcopy(base)
    error = None
    plan = None
    try:
        plan = _plan(base, _remove(2, 2))
    except FeedbackError as caught:
        error = caught
    assert (error is not None) == overlaps, "overlap must use the resolved source edges"
    if overlaps:
        assert "the manual removal would also cut segment s2" in str(error)
    else:
        assert plan is not None
        assert len(plan.removal_actions) == 1
        assert (plan.removal_actions[0].start, plan.removal_actions[0].end) == (3.75, 5.75)
    assert base == before


@pytest.mark.parametrize("anchors,span", [
    ({"from": {"word": "t1:3"}, "to": {"word": "t1:7"}}, (6.38, 15.2)),
    ({"from": {"word": "t1:3", "edge": "end"},
      "to": {"word": "t1:7", "edge": "start"}}, (7.0, 14.5)),
])
def test_source_span_uses_word_edges_and_full_padding(
    base: dict, anchors: dict, span: tuple[float, float],
) -> None:
    base["edit"]["segments"][1].update(anchors)
    planner = _Planner(base["edit"], base["cut_words"], base["source_words"])
    before = copy.deepcopy(base)
    assert planner.source_span(base["edit"]["segments"][1], _remove(2, 2),
                               "v1:c1") == pytest.approx(span)
    assert base == before


@pytest.mark.parametrize("earlier", [
    {"id": "drop", "kind": "remove", "segment": "s2"},
    _trim("t1:7", "start", segment="s2"),
])
def test_middle_remove_checks_other_segments_after_earlier_operations(
    base: dict, earlier: dict,
) -> None:
    base["edit"]["segments"][1]["from"] = {"time": 1.8}
    plan = _plan(base, earlier, _remove(1, 2))
    assert len(plan.removal_actions) == 1


@pytest.mark.parametrize("anchors", [
    {"from": {"time": 5.75}, "to": {"time": 7.0}},
    {"from": {"time": 0.5}, "to": {"time": 1.75}},
])
def test_middle_remove_allows_source_spans_that_only_touch(base: dict, anchors: dict) -> None:
    base["edit"]["segments"][1].update(anchors)
    assert len(_plan(base, _remove(1, 2)).removal_actions) == 1


def test_later_trim_sees_prefix_removal(base: dict) -> None:
    assert _plan(base, _remove(0, 1), _trim("t1:2", "end")) == EditPlan(
        _edit(base, **{"from": {"word": "t1:2"}, "to": {"word": "t1:2"}, "text": "so"}), (), (
            _change("cut", "remove", 'Removed "Start um," from the start of s1.',
                    _words(base, 0, 1)),
            _change("trim", "trim", 'Trimmed s1 to end on "so".', _words(base, 3)),
        ),
    )


@pytest.mark.parametrize("later", [_trim("t1:0", "start"), _remove(0, 2, operation="trim")])
def test_referencing_an_already_removed_occurrence_is_refused(base: dict, later: dict) -> None:
    before = copy.deepcopy(base)
    with pytest.raises(ValidationError) as error:
        _plan(base, _remove(0, 1), later)
    assert _validation(error.value) == (
        "operation trim (v1:c1): word t1:0 is not a kept occurrence of s1; "
        "a trim must keep at least one word"
    )
    assert base == before


@pytest.mark.parametrize("edge,anchors,indices,summary", [
    ("start", {"from": {"word": "t1:2"}, "text": "so about."}, (0, 1),
     'Trimmed s1 to start on "so".'),
    ("end", {"to": {"word": "t1:2"}, "text": "Start um, so"}, (3,),
     'Trimmed s1 to end on "so".'),
])
def test_trim_start_and_end(base: dict, edge: str, anchors: dict,
                           indices: tuple[int, ...], summary: str) -> None:
    assert _plan(base, _trim("t1:2", edge)) == EditPlan(_edit(base, **anchors), (), (
        _change("trim", "trim", summary, _words(base, *indices)),
    ))


@pytest.mark.parametrize("word", ["t1:7", "t2:0"])
def test_trim_outside_segment_is_refused(base: dict, word: str) -> None:
    with pytest.raises(ValidationError) as error:
        _plan(base, _trim(word, "start"))
    assert _validation(error.value) == (
        f"operation trim (v1:c1): word {word} is not a kept occurrence of s1; "
        "a trim must keep at least one word"
    )


def test_trim_without_kept_words_is_refused(base: dict) -> None:
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        word for word in base["cut_words"].words if word.segment != "s1"
    ))
    with pytest.raises(ValidationError) as error:
        _plan(base, _trim("t1:0", "end"))
    assert _validation(error.value) == (
        "operation trim (v1:c1): word t1:0 is not a kept occurrence of s1; "
        "a trim must keep at least one word"
    )


@pytest.mark.parametrize("operation", [
    {"id": "drop", "kind": "remove", "segment": "s1"}, _remove(0, 3, operation="drop"),
])
def test_removing_last_segment_is_refused(base: dict, operation: dict) -> None:
    base["edit"]["segments"] = base["edit"]["segments"][:1]
    with pytest.raises(FeedbackError) as error:
        _plan(base, operation)
    assert str(error.value) == (
        "Operation drop (v1:c1): dropping the segment would leave no segments."
    )


def test_middle_remove_requires_matching_source_words(base: dict) -> None:
    base["source_words"]["t1"] = None
    with pytest.raises(FeedbackError) as error:
        _plan(base, _remove(1, 2))
    assert str(error.value) == ("Operation cut (v1:c1): take t1: its words file does not match "
                                "the saved version.")


@pytest.mark.parametrize("use_request", [False, True])
def test_restore_story_words_and_request_fallback(base: dict, use_request: bool) -> None:
    operation = {"id": "back", "kind": "restore"}
    if use_request:
        base["items"][0]["request"] = {
            "type": "restore", "range": "s1.0", "take": "t1", "start": 7.2, "end": 14.0,
            "text": "and that is",
        }
    else:
        operation["range"] = "s1.0"
    assert _plan(base, operation) == EditPlan(
        _edit(base, to={"word": "t1:6"}, text="Start um, so about. and that is"), (), (
        _change("back", "restore", 'Brought back "and that is" after s1.', _words(base, 4, 5, 6)),
    ))


@pytest.mark.parametrize("earlier", [_trim("t1:2", "end"), _remove(2, 3)])
def test_story_restore_refuses_an_earlier_trim(base: dict, earlier: dict) -> None:
    before = copy.deepcopy(base)
    error = None
    try:
        _plan(base, earlier, {"id": "back", "kind": "restore", "range": "s1.0"})
    except FeedbackError as caught:
        error = caught
    assert error is not None, "restore must not bring back words removed by an earlier trim"
    assert "Operation back (v1:c1)" in str(error)
    assert "segment s1 no longer ends where the removed part began" in str(error)
    assert "restore before trimming" in str(error)
    assert base == before


@pytest.mark.parametrize("anchor,known", [
    ({"time": 7.198}, True), ({"time": 7.198}, False),
    ({"word": "t1:2"}, False),
])
def test_story_restore_refuses_a_gap_before_the_reviewed_boundary(
    base: dict, anchor: dict, known: bool,
) -> None:
    base["edit"]["segments"][0]["to"] = anchor
    if not known:
        base["ranges"]["v1"][0].update(words_known=False, words=[], text=None)
    error = None
    try:
        _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"})
    except FeedbackError as caught:
        error = caught
    assert error is not None, "the previous segment must still meet the reviewed boundary"


@pytest.mark.parametrize("operation,expected", [
    (_trim("t1:2", "start"), "so about."),
    (_trim("t1:2", "end"), "Start um, so"),
    (_remove(0, 1), "so about."),
    (_remove(2, 3), "Start um,"),
    (_remove(1, 2), "Start about."),
    ({"id": "back", "kind": "restore", "range": "s1.0"},
     "Start um, so about. and that is"),
])
def test_changed_segment_text_matches_kept_occurrences(
    base: dict, operation: dict, expected: str,
) -> None:
    before = copy.deepcopy(base)
    plan = _plan(base, operation)
    assert plan.edit["segments"][0]["text"] == expected
    assert plan.edit["segments"][1:] == before["edit"]["segments"][1:]
    assert base == before


def test_changed_segment_text_uses_occurrence_text_and_preserves_absent_text(base: dict) -> None:
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        replace(word, text="spoken") if word.id == "t1:2" else word
        for word in base["cut_words"].words
    ))
    plan = _plan(base, _trim("t1:2", "start"), _trim("t1:7", "end", segment="s2",
                                                               operation="closing"))
    assert plan.edit["segments"][0]["text"] == "spoken about."
    assert "text" not in plan.edit["segments"][1]


def test_filler_restore_updates_segment_text_from_kept_occurrences(base: dict) -> None:
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        word for word in base["cut_words"].words if word.id not in {"t1:1", "t1:2"}
    ))
    base["edit"]["segments"][0]["text"] = "Start about."
    base["ranges"]["v1"][0].update(kind="filler", removal_ids=["r1"], start=1.75, end=5.75,
                                    words=_words(base, 1, 2), text="um, so")
    plan = _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"})
    assert plan.edit["segments"][0]["text"] == "Start um, so about."


@pytest.mark.parametrize("known", [False, True])
def test_restore_story_without_words_uses_time(base: dict, known: bool) -> None:
    base["ranges"]["v1"][0].update(words_known=known, words=[], text=None)
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"}) == EditPlan(
        _edit(base, to={"time": 14.0}), (), (
            _change("back", "restore", "Brought back removed audio after s1.", []),
        ),
    )


def test_restore_unknown_words_without_source_uses_exact_time_anchor(base: dict) -> None:
    base["edit"]["segments"][0]["to"] = {"time": 7.2}
    base["source_words"]["t1"] = None
    base["ranges"]["v1"][0].update(words_known=False, words=[], text=None)
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"}) == EditPlan(
        _edit(base, to={"time": 14.0}), (), (
            _change("back", "restore", "Brought back removed audio after s1.", []),
        ),
    )


@pytest.mark.parametrize("key,value", [("take", "t2"), ("start", 7.200001), ("end", 14.000001)])
def test_restore_mismatched_request_is_refused(base: dict, key: str, value: object) -> None:
    request = {"type": "restore", "range": "s1.0", "take": "t1", "start": 7.2, "end": 14.0}
    request[key] = value
    base["items"][0]["request"] = request
    with pytest.raises(FeedbackError) as error:
        _plan(base, {"id": "back", "kind": "restore"})
    assert str(error.value) == (
        "Operation back (v1:c1): the removed part changed from the restore request."
    )


@pytest.mark.parametrize("changed_take", [False, True])
def test_restore_missing_or_changed_previous_segment_is_refused(
    base: dict, changed_take: bool,
) -> None:
    operations = []
    if changed_take:
        base["edit"]["segments"][0].update(take="t2", **{
            "from": {"word": "t2:0"}, "to": {"word": "t2:1"},
        })
    else:
        operations.append({"id": "drop", "kind": "remove", "segment": "s1"})
    with pytest.raises(FeedbackError) as error:
        _plan(base, *operations, {"id": "back", "kind": "restore", "range": "s1.0"})
    assert str(error.value) == ("Operation back (v1:c1): the removed part no longer follows s1 "
                                "in this version.")
    assert error.value.suggested_action == (
        "Ask the creator with yap-studio feedback ask, or recompute the operations."
    )


@pytest.mark.parametrize("anchor", [{"word": "t1:4"}, {"time": 7.202}])
def test_restore_story_already_reaching_past_start_is_refused(base: dict, anchor: dict) -> None:
    base["edit"]["segments"][0]["to"] = anchor
    with pytest.raises(FeedbackError) as error:
        _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"})
    assert str(error.value) == (
        "Operation back (v1:c1): segment s1 no longer ends where the removed part began, "
        "for example after an earlier trim; restore before trimming or ask the creator."
    )


@pytest.mark.parametrize("kind", ["filler", "silence", "stutter", "restart", "manual"])
def test_restore_applied_removals_rejects_every_id_and_rejoins_words(base: dict, kind: str) -> None:
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        word for word in base["cut_words"].words if word.id not in {"t1:1", "t1:2"}
    ))
    base["ranges"]["v1"][0].update(kind=kind, removal_ids=["r1", "r2"], start=1.75, end=5.75,
                                    words=_words(base, 1, 2), text="um, so")
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"},
                 _trim("t1:2", "end")) == EditPlan(
        _edit(base, to={"word": "t1:2"}, text="Start um, so"), (
        RemovalAction("reject", "back", "v1:c1", f"Bring back the removed {kind} for v1:c1.",
                      removal_id="r1"),
        RemovalAction("reject", "back", "v1:c1", f"Bring back the removed {kind} for v1:c1.",
                      removal_id="r2"),
    ), (
        _change("back", "restore", f"Brought back a removed {kind} (fillers stage).",
                _words(base, 1, 2)),
        _change("trim", "trim", 'Trimmed s1 to end on "so".', _words(base, 3)),
    ))


def test_restore_trim_range_is_refused(base: dict) -> None:
    base["ranges"]["v1"][0]["kind"] = "trim"
    with pytest.raises(FeedbackError) as error:
        _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"})
    assert str(error.value) == ("Operation back (v1:c1): there is no removal or story cut to undo "
                                "at this trim range.")


def test_restore_resolves_comment_version_and_piece_index(base: dict) -> None:
    record = copy.deepcopy(base["ranges"]["v1"][0])
    record["id"] = "s1.1"
    base["ranges"]["v2"] = [record]
    base["pieces"]["v2"] = {"pieces": [base["pieces"]["v1"]["pieces"][1],
                                            base["pieces"]["v1"]["pieces"][0]]}
    base["items"][0]["comment"]["version"] = "v2"
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.1"},
                 comment="v2:c1") == EditPlan(
        _edit(base, to={"word": "t1:6"}, text="Start um, so about. and that is"), (), (
        _change("back", "restore", 'Brought back "and that is" after s1.', _words(base, 4, 5, 6),
                comment="v2:c1"),
    ))


@pytest.mark.parametrize("missing", ["range", "version", "pieces"])
def test_restore_missing_evidence_is_refused(base: dict, missing: str) -> None:
    if missing == "range":
        base["ranges"]["v1"] = []
    elif missing == "version":
        base["ranges"] = {}
    else:
        base["pieces"] = {}
    with pytest.raises(FeedbackError) as error:
        _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"})
    assert str(error.value) == (
        "Operation back (v1:c1): range s1.0 has no previous piece in version v1."
        if missing == "pieces" else "Operation back (v1:c1): range s1.0 is missing from version v1."
    )


def test_restore_without_range_or_request_is_refused(base: dict) -> None:
    with pytest.raises(ValidationError) as error:
        _plan(base, {"id": "back", "kind": "restore"})
    assert _validation(error.value) == (
        "operation back (v1:c1): restore needs a range or the item's restore request"
    )


def test_broll_work_is_deferred_without_mutating_story_plan(base: dict) -> None:
    operation = {"id": "move", "kind": "broll_retime", "element": "broll-one",
                 "start": {"word": "t1:1"}, "hold_s": 1.2}
    before = copy.deepcopy(base)
    plan = _plan(base, operation)
    assert plan.edit == base["edit"]
    assert plan.removal_actions == plan.changes == ()
    assert plan.broll_operations == ({**operation, "comment": "v1:c1"},)
    assert base == before
    plan.broll_operations[0]["start"]["word"] = "t1:2"
    assert operation["start"] == {"word": "t1:1"}


def test_keep_requires_keep_comment(base: dict) -> None:
    assert _plan(base, {"id": "keep", "kind": "keep"}, comment="v1:c2") == EditPlan(
        base["edit"], (), (
            _change("keep", "keep", "Kept, no edit", [], segments=[], comment="v1:c2"),
        ),
    )
    with pytest.raises(ValidationError) as error:
        _plan(base, {"id": "keep", "kind": "keep"})
    assert _validation(error.value) == "operation keep (v1:c1): keep requires a keep comment"


@pytest.mark.parametrize("kind", sorted(TIMELINE_KINDS))
def test_timeline_kinds_have_named_error(kind: str) -> None:
    with pytest.raises(NeedsTimelineError) as error:
        validate_operations(_document({"id": "later", "kind": kind, "position": 2}))
    assert isinstance(error.value, FeedbackError)
    assert error.value.code == "needs_timeline"
    assert str(error.value) == (
        "Operation later (v1:c1): caption and card positions need the timeline stage, "
        "which this build does not have yet."
    )
    assert error.value.suggested_action == "Reply needs_clarification or leave the note open."
    assert TIMELINE_KINDS == {
        "caption", "element", "title", "card", "graphic", "insert", "overlay", "zoom", "timeline",
        "move",
    }


def test_validation_duplicate_ids_and_comments() -> None:
    first = {"id": "same", "kind": "keep"}
    document = _document(first)
    document["items"].append({"comment": "v2:c1", "operations": [copy.deepcopy(first)]})
    with pytest.raises(ValidationError) as error:
        validate_operations(document)
    assert _validation(error.value) == "operation same (v2:c1): duplicate operation id"
    document["items"][1] = {"comment": "v1:c1", "operations": [{"id": "other", "kind": "keep"}]}
    with pytest.raises(ValidationError) as error:
        validate_operations(document)
    assert _validation(error.value) == "comment: duplicate comment v1:c1"


@pytest.mark.parametrize("operation,detail", [
    ({"id": "bad", "kind": "unknown"}, "unknown kind 'unknown'"),
    ({"id": "bad", "kind": "keep", "extra": True}, "expected exactly these keys: id, kind"),
    ({"id": "bad", "kind": "remove", "segment": "s1", "to": {}},
     "expected exactly these keys: id, kind, segment"),
    ({"id": "bad", "kind": "remove", "from": {"word": "t1:0", "segment": "s1"}},
     "expected exactly these keys: from, id, kind, to"),
    ({"id": "bad", "kind": "trim", "segment": "s1", "word": "t1:0", "edge": "middle"},
     "edge must be start or end"),
    ({"id": "bad", "kind": "trim", "segment": "s1", "word": "t1:00", "edge": "end"},
     "word must be a source word id like t1:12"),
    ({"id": "bad", "kind": "restore", "range": " "}, "range must be a nonempty string"),
    ({"id": "bad", "kind": ["remove"]}, "kind must be a string"),
])
def test_validation_operation_shapes(operation: dict, detail: str) -> None:
    with pytest.raises(ValidationError) as error:
        validate_operations(_document(operation))
    assert _validation(error.value) == f"operation bad (v1:c1): {detail}"


@pytest.mark.parametrize("comment", ["c1", "v1:", ":c1", "v0:c1", "v1:c1:extra", "v1: c1"])
def test_validation_bad_comment_references(comment: str) -> None:
    with pytest.raises(ValidationError) as error:
        validate_operations(_document({"id": "keep", "kind": "keep"}, comment=comment))
    assert _validation(error.value) == "comment: must be VERSION:ID, for example v1:c1"


@pytest.mark.parametrize("value", ["", " ", None, 1, []])
def test_validation_nonempty_operation_ids(value: object) -> None:
    with pytest.raises(ValidationError) as error:
        validate_operations(_document({"id": value, "kind": "keep"}))
    assert _validation(error.value) == f"operation {value} (v1:c1): id must be a nonempty string"


@pytest.mark.parametrize("case,expected", [
    ("empty", "operations (v1:c1): must contain at least one operation"),
    ("batch", "batch: must be b followed by a positive integer"),
    ("schema", f"schema: must be {OPERATIONS_SCHEMA}"),
    ("extra", "operations: expected exactly these keys: batch, items, schema"),
    ("item_extra", "item: expected exactly these keys: comment, operations"),
    ("items_type", "items: must be an array"),
])
def test_validation_document_shapes(case: str, expected: str) -> None:
    document = _document({"id": "keep", "kind": "keep"})
    if case == "empty":
        document["items"][0]["operations"] = []
    elif case == "batch":
        document["batch"] = "b0"
    elif case == "schema":
        document["schema"] = "other"
    elif case == "extra":
        document["extra"] = True
    elif case == "item_extra":
        document["items"][0]["extra"] = True
    else:
        document["items"] = {}
    with pytest.raises(ValidationError) as error:
        validate_operations(document)
    assert _validation(error.value) == expected


def test_validation_occurrence_anchors_are_strict() -> None:
    operation = _remove(0, 1)
    operation["from"]["extra"] = True
    with pytest.raises(ValidationError) as error:
        validate_operations(_document(operation))
    assert _validation(error.value) == (
        "operation cut (v1:c1).from: expected exactly these keys: segment, word"
    )
    operation = _remove(0, 1)
    operation["to"]["segment"] = "s2"
    with pytest.raises(ValidationError) as error:
        validate_operations(_document(operation))
    assert _validation(error.value) == (
        "operation cut (v1:c1): from and to must be in the same segment"
    )


def test_comment_not_in_batch_is_refused(base: dict) -> None:
    with pytest.raises(ValidationError) as error:
        _plan(base, {"id": "drop", "kind": "remove", "segment": "s1"}, comment="v2:c1")
    assert _validation(error.value) == (
        "operation drop (v2:c1): comment reference is not in this batch"
    )


def test_remove_reversed_range_and_missing_segment_are_refused(base: dict) -> None:
    with pytest.raises(ValidationError) as error:
        _plan(base, _remove(2, 1))
    assert _validation(error.value) == "operation cut (v1:c1): from must not be after to"
    with pytest.raises(ValidationError) as error:
        _plan(base, {"id": "drop", "kind": "remove", "segment": "s9"})
    assert _validation(error.value) == (
        "operation drop (v1:c1): segment s9 is not in the evolving edit"
    )


def test_inputs_untouched_segments_and_edit_contract_are_preserved(base: dict) -> None:
    document = _document(_remove(0, 1), _trim("t1:2", "end"))
    document["items"].append({"comment": "v1:c2", "operations": [{"id": "keep", "kind": "keep"}]})
    before = copy.deepcopy((base, document))
    normalized = validate_operations(document)
    assert normalized == document and normalized is not document
    normalized["items"][0]["operations"][0]["from"]["word"] = "t1:1"
    assert (base, document) == before
    plan = plan_edit(**base, operations=document)
    assert plan == EditPlan(
        _edit(base, **{"from": {"word": "t1:2"}, "to": {"word": "t1:2"}, "text": "so"}), (), (
        _change("cut", "remove", 'Removed "Start um," from the start of s1.', _words(base, 0, 1)),
        _change("trim", "trim", 'Trimmed s1 to end on "so".', _words(base, 3)),
        _change("keep", "keep", "Kept, no edit", [], segments=[], comment="v1:c2"),
    ))
    assert plan.edit["segments"][1:] == base["edit"]["segments"][1:]
    assert plan.edit["words_sha256"] == base["edit"]["words_sha256"]
    assert plan.edit["end_hold_s"] == 0.75
    assert isinstance(parse_document("edit", plan.edit), Edit)
    assert (base, document) == before
    plan.edit["segments"][1]["note"] = "new"
    plan.changes[0]["words"][0]["text"] = "new"
    assert (base, document) == before


def test_restore_inputs_are_not_mutated_and_restored_words_can_be_trimmed(base: dict) -> None:
    before = copy.deepcopy(base)
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"},
                 _trim("t1:5", "end")) == EditPlan(
        _edit(base, to={"word": "t1:5"}, text="Start um, so about. and that"), (), (
        _change("back", "restore", 'Brought back "and that is" after s1.', _words(base, 4, 5, 6)),
        _change("trim", "trim", 'Trimmed s1 to end on "that".', _words(base, 6)),
    ))
    assert base == before


def test_known_source_anchor_ids_must_exist(base: dict) -> None:
    base["source_words"]["t1"] = replace(base["source_words"]["t1"], words=tuple(
        word for word in base["source_words"]["t1"].words if word.id != "t1:2"
    ))
    with pytest.raises(ValidationError) as error:
        _plan(base, _trim("t1:2", "end"))
    assert _validation(error.value) == "operation trim (v1:c1): word t1:2 is missing from take t1"


def test_changes_limit_words_to_200_and_keep_short_summary(base: dict) -> None:
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        CutWord(f"t1:{index}", "s1", "word", index, index + 0.5) for index in range(205)
    ))
    plan = _plan(base, {"id": "drop", "kind": "remove", "segment": "s1"})
    assert plan == EditPlan(_edit(base, drop=True), (), (
        _change("drop", "remove", f'Dropped segment s1 (205 words: "{("word " * 24)[:117]}...").',
                [{"id": f"t1:{index}", "text": "word"} for index in range(200)]),
    ))


def test_restore_then_remove_sees_restored_occurrences(base: dict) -> None:
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"},
                 _remove(4, 6)) == EditPlan(base["edit"], (), (
        _change("back", "restore", 'Brought back "and that is" after s1.', _words(base, 4, 5, 6)),
        _change("cut", "remove", 'Removed "and that is" from the end of s1.',
                _words(base, 4, 5, 6)),
    ))


def test_repeated_source_word_is_qualified_by_segment(base: dict) -> None:
    base["edit"]["segments"][1].update(**{"from": {"word": "t1:0"}, "to": {"word": "t1:3"}})
    base["cut_words"] = replace(base["cut_words"], words=(
        *base["cut_words"].words[:4],
        *(CutWord(f"t1:{index}", "s2", text, index + 5.0, index + 5.5)
          for index, text in enumerate(("Start", "um,", "so", "about."))),
        *base["cut_words"].words[5:],
    ))
    expected = _edit(base, **{"from": {"word": "t1:2"}, "text": "so about."})
    expected["segments"][1]["to"] = {"word": "t1:0"}
    assert _plan(base, _remove(0, 1), _trim("t1:0", "end", segment="s2")) == EditPlan(
        expected, (), (
            _change("cut", "remove", 'Removed "Start um," from the start of s1.',
                    _words(base, 0, 1)),
            _change("trim", "trim", 'Trimmed s2 to end on "Start".', _words(base, 1, 2, 3),
                    segments=["s2"]),
        ),
    )


def test_operations_target_another_take(base: dict) -> None:
    operation = {"id": "cut", "kind": "remove",
                 "from": {"word": "t2:0", "segment": "s3"},
                 "to": {"word": "t2:0", "segment": "s3"}}
    assert _plan(base, operation) == EditPlan(
        _edit(base, segment="s3", **{"from": {"word": "t2:1"}, "text": "take."}), (), (
            _change("cut", "remove", 'Removed "Other" from the start of s3.',
                    [{"id": "t2:0", "text": "Other"}], segments=["s3"]),
        ),
    )


def test_empty_operations_returns_an_independent_valid_edit(base: dict) -> None:
    document = {"schema": OPERATIONS_SCHEMA, "batch": "b1", "items": []}
    plan = plan_edit(**base, operations=document)
    assert plan == EditPlan(base["edit"], (), ())
    assert plan.edit is not base["edit"]
    assert isinstance(parse_document("edit", plan.edit), Edit)
    base["edit"]["segments"][0]["to"] = {"word": "t1:999"}
    with pytest.raises(ValidationError) as error:
        plan_edit(**base, operations=document)
    assert _validation(error.value) == "edit segment s1.to: word t1:999 is missing from take t1"


@pytest.mark.parametrize("extra", [0.0, 0.001])
def test_restore_story_allows_one_millisecond_tolerance(base: dict, extra: float) -> None:
    base["edit"]["segments"][0]["to"] = {"time": 7.2 + extra}
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"}) == EditPlan(
        _edit(base, to={"word": "t1:6"}, text="Start um, so about. and that is"), (), (
            _change("back", "restore", 'Brought back "and that is" after s1.',
                    _words(base, 4, 5, 6)),
        ),
    )


@pytest.mark.parametrize("known", [False, True])
def test_restore_removal_without_piece_metadata_or_source_words(base: dict, known: bool) -> None:
    base["source_words"]["t1"] = None
    base["cut_words"] = replace(base["cut_words"], words=tuple(
        word for word in base["cut_words"].words if word.id not in {"t1:1", "t1:2"}
    ))
    restored = [{"id": "t1:1", "text": "um,"}, {"id": "t1:2", "text": "so"}] if known else []
    base["ranges"]["v1"][0].update(kind="filler", removal_ids=["r1", "r2"], words=restored,
                                    words_known=known, start=1.75, end=5.75)
    base["pieces"] = {}
    before = copy.deepcopy(base)
    assert _plan(base, {"id": "back", "kind": "restore", "range": "s1.0"}) == EditPlan(
        _edit(base, text="Start um, so about." if known else "Start about."), (
            RemovalAction("reject", "back", "v1:c1", "Bring back the removed filler for v1:c1.",
                          removal_id="r1"),
            RemovalAction("reject", "back", "v1:c1", "Bring back the removed filler for v1:c1.",
                          removal_id="r2"),
        ), (
            _change("back", "restore", "Brought back a removed filler (fillers stage).", restored),
        ),
    )
    assert base == before
