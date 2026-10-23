"""Exact review frame arithmetic, normalization, and ordering tests."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from yap_studio.errors import ValidationError
from yap_studio.review import (
    TARGETS,
    Comment,
    MediaFacts,
    frame_at,
    frame_mid,
    frame_start,
    frame_time_text,
    legacy_kind,
    normalize_input,
    sorted_comments,
)
from yap_studio.review.model import conform_comment

FPS = Fraction(30000, 1001)


def test_pin_rounding_patch_span_and_removal() -> None:
    fields = normalize_input({"text": "Move this", "start_frame": 440, "end_frame": 460,
                              "pin": {"x": 0.42134, "y": 0.31054, "frame": 447}}, _facts())
    assert fields["pin"] == {"x": 0.4213, "y": 0.3105, "frame": 447, "time_s": 14.9149}
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    assert normalize_input({"kind": "keep", "target": "element"}, _facts(), partial=True,
                           current=current)["pin"] == fields["pin"]
    with pytest.raises(ValidationError, match="must be on a frame inside the note"):
        normalize_input({"start_frame": 448}, _facts(), partial=True, current=current)
    assert normalize_input({"start_frame": 448, "pin": None}, _facts(), partial=True,
                           current=current)["pin"] is None


@pytest.mark.parametrize("pin", [
    {}, [], True, {"x": 0.5, "y": 0.5},
    {"x": 0.5, "y": 0.5, "frame": 447, "time_s": 14.9149},
    *[{"x": x, "y": 0.5, "frame": 447} for x in
      (True, float("nan"), float("inf"), -0.01, 1.01, "0.5", 10**400)],
    *[{"x": 0.5, "y": y, "frame": 447} for y in (False, float("nan"), -1, 2)],
    *[{"x": 0.5, "y": 0.5, "frame": f} for f in (True, -1, 447.0, 446, 448, 10**400)],
])
def test_invalid_pin_input(pin: object) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Note", "start_frame": 447, "pin": pin}, _facts())
    assert error.value.parameter == "pin"


@pytest.mark.parametrize("frame,valid", [(9, False), (10, True), (19, True), (20, False)])
def test_pin_range_is_half_open(frame: int, valid: bool) -> None:
    data = {"text": "Note", "start_frame": 10, "end_frame": 20,
            "pin": {"x": 0, "y": 1, "frame": frame}}
    if valid:
        assert normalize_input(data, _facts())["pin"]["frame"] == frame
    else:
        with pytest.raises(ValidationError, match="move or remove the pin"):
            normalize_input(data, _facts())


def test_variable_pin_precision_tolerance_and_conform() -> None:
    time = 1.234567891
    for offset in (0, 0.00049, -0.00049):
        fields = normalize_input({"text": "Note", "start_s": time,
                                  "pin": {"x": 1, "y": 0, "time_s": time + offset}},
                                 _facts(cfr=False))
        assert fields["pin"] == {"x": 1, "y": 0, "frame": None, "time_s": time + offset}
    for pin in ({"x": 0, "y": 0, "time_s": time + 0.00051},
                {"x": 0, "y": 0, "frame": 37},
                {"x": 0, "y": 0, "time_s": True},
                {"x": 0, "y": 0, "time_s": 10**400}):
        with pytest.raises(ValidationError):
            normalize_input({"text": "Note", "start_s": time, "pin": pin}, _facts(cfr=False))
    for t in (time, 2.0):
        assert normalize_input({"text": "Note", "start_s": time, "end_s": 2,
                                "pin": {"x": 0, "y": 0, "time_s": t}},
                               _facts(cfr=False))["pin"]["time_s"] == t
    pinned = replace(_comment("c1", 447),
                     pin={"x": 0.5, "y": 0.5, "frame": 447, "time_s": 14.9149})
    assert conform_comment(pinned, _facts(cfr=False)).pin == {**pinned.pin, "frame": None}


def test_time_only_pin_conforms_with_its_note_to_frame_timing() -> None:
    comment = replace(_comment("c1", 447), start_frame=None, start_s=14.915,
                      pin={"x": 0.5, "y": 0.5, "frame": None, "time_s": 14.915})
    result = conform_comment(comment, _facts())
    assert result.pin == {"x": 0.5, "y": 0.5, "frame": 447, "time_s": 14.9149}


def _facts(cfr: bool = True, duration_s: float | None = 60.0) -> MediaFacts:
    return MediaFacts("video.mp4", "a" * 64, duration_s, FPS if cfr else None, cfr)


def _comment(id: str, start: int, end: int | None = None) -> Comment:
    return Comment(
        id,
        "fix",
        None,
        "Note",
        start,
        end,
        float(frame_start(start, FPS)),
        float(frame_start(end, FPS)) if end is not None else None,
        "2026-09-29T10:00:00Z",
        "2026-09-29T10:00:00Z",
    )


def test_every_frame_in_sixty_seconds_maps_back_from_boundary_middle_and_early_float() -> None:
    for frame in range(1800):
        start = float(frame_start(frame, FPS))
        assert frame_at(start, FPS) == frame
        assert frame_at(float(frame_mid(frame, FPS)), FPS) == frame
        assert frame_at(start - 1e-6, FPS) == frame
    assert frame_at(-10.0, FPS) == 0
    assert frame_mid(448, FPS) == Fraction(897, 2) / FPS


def test_frame_display_uses_exact_time_and_floors() -> None:
    assert frame_time_text(0, FPS) == "0:00.0"
    assert frame_time_text(3000, FPS) == "1:40.1"
    assert frame_time_text(447, FPS) == "0:14.9"
    assert frame_time_text(1538, FPS) == "0:51.3"


def test_create_defaults_to_fix_and_normalizes_text_and_frame_seconds() -> None:
    result = normalize_input({"text": "  First\r\nSecond  ", "start_frame": 447}, _facts())
    assert result == {
        "kind": "fix",
        "target": None,
        "text": "First\nSecond",
        "start_frame": 447,
        "end_frame": None,
        "start_s": 14.9149,
        "end_s": None,
        "request": None,
        "pin": None,
    }
    assert (
        normalize_input({"text": "Range", "start_frame": 0, "end_frame": 96}, _facts())["end_s"]
        == 3.2032
    )


def test_one_frame_range_collapses_to_a_point() -> None:
    result = normalize_input({"text": "Note", "start_frame": 8, "end_frame": 9}, _facts())
    assert result["end_frame"] is None
    assert result["end_s"] is None


@pytest.mark.parametrize("cfr", [True, False])
@pytest.mark.parametrize("kind", ["fix", "keep"])
def test_kind_is_accepted_for_frames_and_seconds(cfr: bool, kind: str) -> None:
    timing = {"start_frame": 3} if cfr else {"start_s": 0.1}
    result = normalize_input({"kind": kind, "text": "Note", **timing}, _facts(cfr=cfr))
    assert result["kind"] == kind
    assert result["target"] is None
    assert "label" not in result


@pytest.mark.parametrize("kind", ["like", "change", "other", None, True, [], {}])
def test_invalid_kind_has_specific_validation_error(kind: object) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"kind": kind, "text": "Note", "start_frame": 0}, _facts())
    assert error.value.parameter == "kind"
    assert error.value.detail == "must be 'keep' or 'fix'"


def test_create_does_not_inherit_patch_kind_or_target() -> None:
    current = replace(_comment("c1", 10, 20), kind="keep", target="story")
    result = normalize_input({"text": "New", "start_frame": 3}, _facts(), current=current)
    assert result["kind"] == "fix"
    assert result["target"] is None


@pytest.mark.parametrize(("label", "kind"), [("like", "keep"), ("change", "fix")])
def test_legacy_alias_and_matching_kind(label: str, kind: str) -> None:
    assert legacy_kind(label) == kind
    payload = {"label": label, "text": "Note", "start_frame": 0}
    assert normalize_input(payload, _facts())["kind"] == kind
    assert normalize_input({**payload, "kind": kind}, _facts())["kind"] == kind
    current = replace(_comment("c1", 10, 20), kind="fix" if kind == "keep" else "keep")
    patched = normalize_input({"label": label}, _facts(), partial=True, current=current)
    assert patched["kind"] == kind
    with pytest.raises(ValidationError) as error:
        normalize_input({"label": label, "kind": current.kind}, _facts(),
                        partial=True, current=current)
    assert error.value.parameter == "label"
    assert error.value.detail == "disagrees with kind"


@pytest.mark.parametrize("label", ["other", "keep", None, True, [], {}])
def test_invalid_legacy_alias(label: object) -> None:
    with pytest.raises(ValueError):
        legacy_kind(label)
    with pytest.raises(ValidationError) as error:
        normalize_input({"label": label, "text": "Note", "start_frame": 0}, _facts())
    assert error.value.parameter == "label"
    assert error.value.detail == "must be 'like' or 'change'"


@pytest.mark.parametrize("target", TARGETS)
def test_target_set_kept_on_patch_and_cleared(target: str) -> None:
    assert TARGETS == ("story", "speech", "caption", "element", "video")
    fields = normalize_input({"target": target, "text": "Note", "start_frame": 10,
                              "end_frame": 20}, _facts())
    assert fields["target"] == target
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    patched = normalize_input({"kind": "keep"}, _facts(), partial=True, current=current)
    assert patched == {**fields, "kind": "keep"}
    cleared = normalize_input({"target": None}, _facts(), partial=True, current=current)
    assert cleared == {**fields, "target": None}


@pytest.mark.parametrize("target", ["graphic", "whole video", "other", "", True, [], {}])
def test_invalid_target_names_the_allowed_values(target: object) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"target": target, "text": "Note", "start_frame": 0}, _facts())
    assert error.value.parameter == "target"
    assert all(value in error.value.detail for value in TARGETS)


def test_conform_comment_clears_frames_for_variable_rate_without_changing_seconds() -> None:
    comment = _comment("c1", 30, 60)
    facts = replace(_facts(), cfr=False)
    assert conform_comment(comment, facts) == replace(comment, start_frame=None, end_frame=None)
    assert conform_comment(comment, replace(_facts(), fps=None)) == replace(
        comment, start_frame=None, end_frame=None
    )


def test_conform_comment_derives_frames_and_rounded_seconds_for_constant_rate() -> None:
    comment = replace(_comment("c1", 0), start_frame=None, start_s=1.2, end_s=2.2)
    conformed = conform_comment(comment, _facts())
    assert conformed == replace(
        comment, start_frame=35, end_frame=66, start_s=1.1678, end_s=2.2022
    )
    short = replace(comment, end_s=1.21)
    assert conform_comment(short, _facts()) == replace(
        short, start_frame=35, end_frame=None, start_s=1.1678, end_s=None
    )


def test_conform_comment_clamps_seconds_only_point_to_last_frame() -> None:
    facts = replace(_facts(), fps=Fraction(30), duration_s=1.0)
    comment = replace(_comment("c1", 0), start_frame=None, start_s=1.05)

    conformed = conform_comment(comment, facts)

    assert conformed == replace(comment, start_frame=29, start_s=0.9667)


def test_conform_comment_clamps_seconds_only_range_and_allows_text_patch() -> None:
    facts = replace(_facts(), fps=Fraction(30), duration_s=1.0)
    comment = replace(_comment("c1", 0), start_frame=None, start_s=0.9, end_s=1.08)

    conformed = conform_comment(comment, facts)

    assert conformed == replace(
        comment, start_frame=27, end_frame=30, start_s=0.9, end_s=1.0
    )
    assert normalize_input({"text": "Updated"}, facts, partial=True, current=conformed) == {
        "kind": "fix",
        "target": None,
        "text": "Updated",
        "start_frame": 27,
        "end_frame": 30,
        "start_s": 0.9,
        "end_s": 1.0,
        "request": None,
        "pin": None,
    }


def test_conform_comment_preserves_existing_frame_comment() -> None:
    comment = _comment("c1", 30, 60)
    assert conform_comment(comment, _facts()) is comment


def test_patch_inherits_fields_and_revalidates_kept_end() -> None:
    current = _comment("c1", 10, 20)
    updated = normalize_input({"start_frame": 15}, _facts(), partial=True, current=current)
    assert updated["start_frame"] == 15
    assert updated["end_frame"] == 20
    assert updated["kind"] == "fix"
    assert updated["text"] == "Note"
    with pytest.raises(ValidationError) as error:
        normalize_input({"start_frame": 20}, _facts(), partial=True, current=current)
    assert error.value.parameter == "end_frame"
    with pytest.raises(ValidationError) as error:
        normalize_input({}, _facts(), partial=True)
    assert error.value.parameter == "current"


@pytest.mark.parametrize(
    ("payload", "parameter"),
    [
        ({"text": "", "start_frame": 0}, "text"),
        ({"text": " " * 3, "start_frame": 0}, "text"),
        ({"text": "x" * 5001, "start_frame": 0}, "text"),
        ({"text": "Note", "start_frame": 0, "label": "other"}, "label"),
        ({"text": "Note", "start_frame": True}, "start_frame"),
        ({"text": "Note", "start_frame": 1.0}, "start_frame"),
        ({"text": "Note", "start_frame": -1}, "start_frame"),
        ({"text": "Note", "start_frame": 2, "end_frame": True}, "end_frame"),
        ({"text": "Note", "start_frame": 2, "end_frame": 2}, "end_frame"),
        ({"text": "Note", "start_frame": 2, "end_frame": 1}, "end_frame"),
        ({"text": "Note", "start_frame": 0, "start_s": 0.0}, "start_s"),
        ({"text": "Note", "start_frame": 0, "unused": 1}, "unused"),
        ({"text": "Note", "start_frame": 1830}, "start_frame"),
    ],
)
def test_invalid_constant_frame_rate_input(payload: dict, parameter: str) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input(payload, _facts(), partial=False)
    assert error.value.parameter == parameter


def test_text_limits_and_frame_count() -> None:
    assert len(normalize_input({"text": "x" * 5000, "start_frame": 0}, _facts())["text"]) == 5000
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Note", "start_frame": 0}, replace(_facts(), fps=None))
    assert error.value.parameter == "fps"


@pytest.mark.parametrize(
    ("fps", "last_frame", "frame_count"),
    [(Fraction(30), 1799, 1800), (FPS, 1797, 1798)],
)
def test_constant_frame_rate_bounds_follow_frame_count(
    fps: Fraction, last_frame: int, frame_count: int
) -> None:
    facts = replace(_facts(), fps=fps)
    assert normalize_input({"text": "Note", "start_frame": last_frame}, facts)[
        "start_frame"
    ] == last_frame
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Note", "start_frame": frame_count}, facts)
    assert error.value.parameter == "start_frame"
    assert normalize_input({"text": "Note", "start_frame": 0, "end_frame": frame_count}, facts)[
        "end_frame"
    ] == frame_count
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Note", "start_frame": 0, "end_frame": frame_count + 1}, facts)
    assert error.value.parameter == "end_frame"


def test_variable_frame_rate_keeps_full_precision_seconds_and_null_frames() -> None:
    start = 14.948266666666667
    result = normalize_input(
        {"kind": "keep", "text": "  Keep  ", "start_s": start, "end_s": 15.123456789},
        _facts(cfr=False),
    )
    assert result == {
        "kind": "keep",
        "target": None,
        "text": "Keep",
        "start_frame": None,
        "end_frame": None,
        "start_s": start,
        "end_s": 15.123456789,
        "request": None,
        "pin": None,
    }
    current = Comment("c2", "keep", None, "Keep", None, None, start, None, "now", "now")
    assert (
        normalize_input({"text": "New"}, _facts(cfr=False), partial=True, current=current)[
            "start_s"
        ]
        == start
    )


@pytest.mark.parametrize(
    ("payload", "parameter"),
    [
        ({"text": "Note", "start_s": True}, "start_s"),
        ({"text": "Note", "start_s": float("nan")}, "start_s"),
        ({"text": "Note", "start_s": float("inf")}, "start_s"),
        ({"text": "Note", "start_s": -0.1}, "start_s"),
        ({"text": "Note", "start_s": 1.0, "end_s": 1.0}, "end_s"),
        ({"text": "Note", "start_s": 1.0, "end_s": 0.5}, "end_s"),
        ({"text": "Note", "start_s": 0.0, "end_s": True}, "end_s"),
        ({"text": "Note", "start_s": 0.0, "start_frame": 0}, "start_frame"),
        ({"text": "Note", "start_s": 60.2}, "start_s"),
        ({"text": "Note", "start_s": 0.0, "end_s": 60.2}, "end_s"),
    ],
)
def test_invalid_variable_frame_rate_input(payload: dict, parameter: str) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input(payload, _facts(cfr=False))
    assert error.value.parameter == parameter


def test_variable_frame_rate_allows_time_just_past_probe_duration() -> None:
    facts = _facts(cfr=False)
    assert normalize_input({"text": "Point", "start_s": 60.05}, facts)["start_s"] == 60.05
    assert normalize_input({"text": "Range", "start_s": 59.0, "end_s": 60.05}, facts)[
        "end_s"
    ] == 60.05


def test_sort_uses_start_then_earlier_end_then_numeric_creation_order() -> None:
    comments = [
        _comment("c10", 5),
        _comment("c2", 5),
        _comment("c5", 5, 9),
        _comment("c4", 5, 8),
        _comment("c3", 3),
    ]
    assert [comment.id for comment in sorted_comments(comments)] == [
        "c3",
        "c2",
        "c10",
        "c4",
        "c5",
    ]
    assert comments[0].id == "c10"
    variable = [
        replace(_comment("c2", 0), start_frame=None, start_s=2.0),
        replace(_comment("c1", 0), start_frame=None, start_s=1.0),
    ]
    assert [comment.id for comment in sorted_comments(variable)] == ["c1", "c2"]


@pytest.mark.parametrize("cfr", [True, False])
def test_spelling_request_normalized_and_patch_retains_replaces_or_clears(cfr: bool) -> None:
    timing = {"start_frame": 3} if cfr else {"start_s": 0.1}
    request = {"type": "spelling", "from": " tail net ", "to": " tailnet "}
    fields = normalize_input({"kind": "fix", "target": "caption", "text": "Correct this",
                              "request": request, **timing}, _facts(cfr=cfr))
    expected = {"type": "spelling", "from": "tail net", "to": "tailnet"}
    assert fields["request"] == expected
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    assert normalize_input({"text": "More detail"}, _facts(cfr=cfr), partial=True,
                           current=current)["request"] == expected
    replacement = {**expected, "to": "tail-net"}
    assert normalize_input({"request": replacement}, _facts(cfr=cfr), partial=True,
                           current=current)["request"] == replacement
    assert normalize_input({"request": None, "kind": "keep", "target": "story"},
                           _facts(cfr=cfr), partial=True, current=current)["request"] is None
    assert normalize_input({"request": None}, _facts(cfr=cfr), partial=True,
                           current=current)["request"] is None
    assert request["from"] == " tail net "


@pytest.mark.parametrize(("kind", "target"), [
    ("keep", "caption"), ("fix", None), ("fix", "story"), ("fix", "speech"),
    ("fix", "element"), ("fix", "video"),
])
def test_spelling_needs_fix_caption_on_create_and_patch(kind, target) -> None:
    request = {"type": "spelling", "from": "tail net", "to": "tailnet"}
    payload = {"text": "Spelling", "start_frame": 0, "request": request}
    with pytest.raises(ValidationError) as error:
        normalize_input({**payload, "kind": kind, "target": target}, _facts())
    assert error.value.parameter == "request"
    assert error.value.detail == "a spelling fix needs a Fix note about a caption"
    fields = normalize_input({**payload, "kind": "fix", "target": "caption"}, _facts())
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    with pytest.raises(ValidationError) as error:
        normalize_input({"kind": kind, "target": target}, _facts(), partial=True, current=current)
    assert error.value.detail == "a spelling fix needs a Fix note about a caption"


@pytest.mark.parametrize("value", [
    [], "spelling", {}, {"type": "restore"},
    {"type": "spelling", "from": "wrong"},
    {"type": "spelling", "from": "wrong", "to": "right", "extra": 1},
    *[{"type": "spelling", "from": wrong, "to": right} for wrong, right in (
        (None, "right"), (1, "right"), ("wrong", []), ("", "right"), ("  ", "right"),
        ("wrong", ""), ("x" * 81, "right"), ("wrong", "x" * 81), ("same", " same "),
        ("two\nlines", "right"), ("wrong", "two\rlines"), ("wrong\n", "right"),
        ("wrong", "right\u2028word"),
    )],
])
def test_invalid_spelling_request(value) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"kind": "fix", "target": "caption", "text": "Spelling",
                         "start_frame": 0, "request": value}, _facts())
    assert error.value.parameter == "request"


def test_spelling_limits_and_legacy_kind_patch() -> None:
    request = {"type": "spelling", "from": "a", "to": "b" * 80}
    fields = normalize_input({"label": "change", "target": "caption", "text": "Spelling",
                              "start_frame": 0, "request": request}, _facts())
    assert fields["request"] == request
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    with pytest.raises(ValidationError, match="a spelling fix needs"):
        normalize_input({"label": "like"}, _facts(), partial=True, current=current)


RESTORE = {"type": "restore", "range": "s1.0", "take": "t1", "start": 1,
           "end": 2, "text": "um well"}


@pytest.mark.parametrize("cfr", [True, False])
@pytest.mark.parametrize("text", [None, "", "um well", "x" * 500])
def test_restore_request_round_trip_and_draft_patch(cfr: bool, text: str | None) -> None:
    request = {**RESTORE, "text": text}
    fields = normalize_input({"text": "Bring back this part", "kind": "fix", "target": "story",
                              "request": request,
                              **({"start_frame": 30} if cfr else {"start_s": 1.0})},
                             _facts(cfr=cfr))
    assert fields["request"] == request and fields["request"] is not request
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    assert normalize_input({"text": "More detail"}, _facts(cfr=cfr), partial=True,
                           current=current)["request"] == request
    replacement = {**request, "range": "s1.1", "start": 2, "end": 3}
    assert normalize_input({"request": replacement}, _facts(cfr=cfr), partial=True,
                           current=current)["request"] == replacement
    assert normalize_input({"request": None, "kind": "keep", "target": "caption"},
                           _facts(cfr=cfr), partial=True, current=current)["request"] is None


@pytest.mark.parametrize(("kind", "target"), [
    ("keep", "story"), ("fix", None), ("fix", "speech"), ("fix", "caption"),
    ("fix", "element"), ("fix", "video"),
])
def test_restore_needs_fix_story_on_create_and_patch(kind, target) -> None:
    payload = {"text": "Bring back", "start_frame": 30, "request": RESTORE}
    with pytest.raises(ValidationError, match="a restore request needs"):
        normalize_input({**payload, "kind": kind, "target": target}, _facts())
    fields = normalize_input({**payload, "kind": "fix", "target": "story"}, _facts())
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    with pytest.raises(ValidationError, match="a restore request needs"):
        normalize_input({"kind": kind, "target": target}, _facts(), partial=True, current=current)


@pytest.mark.parametrize("value", [
    {key: value for key, value in RESTORE.items() if key != "text"},
    {**RESTORE, "extra": True},
    *[{**RESTORE, key: value} for key, value in (
        ("range", None), ("range", ""), ("range", "  "), ("take", []), ("take", ""),
        ("start", -1), ("start", True), ("start", "1"), ("start", None),
        ("start", float("nan")), ("start", float("inf")),
        ("end", -1), ("end", False), ("end", "2"), ("end", None),
        ("end", float("nan")), ("end", float("inf")), ("end", 1), ("end", 0),
        ("end", 10 ** 1000), ("text", "x" * 501), ("text", []), ("text", 1),
    )],
])
def test_invalid_restore_request(value) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Bring back", "target": "story", "start_frame": 30,
                         "request": value}, _facts())
    assert error.value.parameter == "request"


CUT = {"type": "cut", "start": 1.001, "end": 2.002, "words": [
    {"id": "t1:40", "segment": "s3", "text": "so"},
    {"id": "t1:41", "segment": "s3", "text": "basically"},
]}


@pytest.mark.parametrize("cfr", [True, False])
def test_cut_request_round_trip_copies_words(cfr: bool) -> None:
    request = {**CUT, "words": [dict(word) for word in CUT["words"]]}
    request["words"][0].update(segment=" s3 ", text=" so ")
    timing = {"start_frame": 30, "end_frame": 60} if cfr else {
        "start_s": 1.001, "end_s": 2.002,
    }
    fields = normalize_input({"text": "Tighten this", "target": "story", "request": request,
                              **timing}, _facts(cfr=cfr))
    assert fields["request"] == CUT
    assert fields["request"] is not request
    assert fields["request"]["words"] is not request["words"]
    for returned, original in zip(fields["request"]["words"], request["words"], strict=True):
        assert returned is not original
    fields["request"]["words"][0]["text"] = "changed"
    assert request["words"][0] == {"id": "t1:40", "segment": " s3 ", "text": " so "}


@pytest.mark.parametrize(("kind", "target"), [
    ("keep", "story"), ("fix", None), ("fix", "speech"), ("fix", "caption"),
    ("fix", "element"), ("fix", "video"),
])
def test_cut_needs_fix_story_on_create_and_patch(kind, target) -> None:
    payload = {"text": "Tighten this", "start_frame": 30, "end_frame": 60, "request": CUT}
    fields = normalize_input({**payload, "target": "story"}, _facts())
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    for data, partial in ((payload, False), ({}, True)):
        with pytest.raises(ValidationError) as error:
            normalize_input({**data, "kind": kind, "target": target}, _facts(),
                            partial=partial, current=current)
        assert error.value.parameter == "request"
        assert error.value.detail == "a cut request needs a Fix note about the story"


@pytest.mark.parametrize("value", [
    *[{key: value for key, value in CUT.items() if key != missing}
      for missing in ("start", "end", "words")],
    {**CUT, "extra": True},
])
def test_cut_request_requires_exact_keys(value) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Tighten this", "target": "story", "start_frame": 30,
                         "end_frame": 60, "request": value}, _facts())
    assert error.value.parameter == "request"
    assert error.value.detail == "a cut request needs start, end and words"


@pytest.mark.parametrize(("key", "value", "detail"), [
    ("start", 2.002, "end must be greater than start"),
    ("start", 3, "end must be greater than start"),
    *[(key, value, "must be a finite non-negative number")
      for key in ("start", "end")
      for value in (True, False, -1, None, "1", float("nan"), float("inf"), float("-inf"))],
    *[(key, 10 ** 1000, "times must be finite non-negative numbers") for key in ("start", "end")],
])
def test_cut_request_validates_times(key: str, value: object, detail: str) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Tighten this", "target": "story", "start_frame": 30,
                         "end_frame": 60, "request": {**CUT, key: value}}, _facts())
    assert error.value.parameter == "request"
    assert error.value.detail == detail


@pytest.mark.parametrize("words", [None, {}, "so", (), [], [CUT["words"][0]] * 201])
def test_cut_request_limits_word_list(words) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Tighten this", "target": "story", "start_frame": 30,
                         "end_frame": 60, "request": {**CUT, "words": words}}, _facts())
    assert error.value.parameter == "request"
    assert error.value.detail == "words must list 1 to 200 words"


@pytest.mark.parametrize(("word", "detail"), [
    *[(word, "word must contain id, segment and text") for word in (
        None, "so", [], {}, {"id": "t1:40", "segment": "s3"},
        {**CUT["words"][0], "extra": True},
    )],
    *[({**CUT["words"][0], "id": value}, "word id must look like t1:12") for value in (
        None, 40, "", "t0:1", "t01:1", "t1:01", "t1:-1", "t1:1x", " t1:1", "t1:1\n", "t1:١",
    )],
    *[({**CUT["words"][0], "segment": value}, "segment must contain 1 to 200 characters")
      for value in (None, 3, "", "  ", "s" * 201, " " * 200 + "s")],
    *[({**CUT["words"][0], "text": value},
       "text must contain 1 to 200 characters, no newlines")
      for value in (None, 3, "", "  ", "x" * 201,
                    *[f"so{line_break}well" for line_break in "\r\n\v\f\x85\u2028\u2029"], "so\n")],
])
def test_cut_request_validates_each_word(word, detail: str) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Tighten this", "target": "story", "start_frame": 30,
                         "end_frame": 60, "request": {**CUT, "words": [word]}}, _facts())
    assert error.value.parameter == "request"
    assert error.value.detail == detail


def test_cut_request_rejects_repeated_occurrence_after_strip() -> None:
    words = [CUT["words"][0], {**CUT["words"][0], "segment": " s3 ", "text": "different"}]
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Tighten this", "target": "story", "start_frame": 30,
                         "end_frame": 60, "request": {**CUT, "words": words}}, _facts())
    assert error.value.parameter == "request"
    assert error.value.detail == "words must not repeat an occurrence"


@pytest.mark.parametrize("count", [1, 200])
def test_cut_request_accepts_word_limits_and_distinct_occurrences(count: int) -> None:
    words = [{"id": "t12:0", "segment": str(index).zfill(200), "text": "x" * 200}
             for index in range(count)]
    request = {"type": "cut", "start": 0, "end": 2, "words": words}
    fields = normalize_input({"text": "Tighten this", "target": "story", "start_s": 0,
                              "end_s": 2, "request": request}, _facts(cfr=False))
    assert fields["request"] == request
    assert type(fields["request"]["start"]) is float
    assert type(fields["request"]["end"]) is float


@pytest.mark.parametrize(("cfr", "timing"), [
    (True, {"start_frame": 30}), (True, {"start_frame": 30, "end_frame": 31}),
    (False, {"start_s": 1.001}),
])
def test_cut_request_needs_range(cfr: bool, timing: dict) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Tighten this", "target": "story", "request": CUT,
                         **timing}, _facts(cfr=cfr))
    assert error.value.parameter == "request"
    assert error.value.detail == "a cut request needs a range"


@pytest.mark.parametrize("cfr", [True, False])
@pytest.mark.parametrize("key", ["start", "end"])
def test_cut_request_must_match_note_range(cfr: bool, key: str) -> None:
    timing = {"start_frame": 30, "end_frame": 60} if cfr else {
        "start_s": 1.001, "end_s": 2.002,
    }
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Tighten this", "target": "story", **timing,
                         "request": {**CUT, key: CUT[key] + 0.0001}}, _facts(cfr=cfr))
    assert error.value.parameter == "request"
    assert error.value.detail == "a cut request must match the note's range; select the range again"


@pytest.mark.parametrize("cfr", [True, False])
def test_cut_request_range_matches_at_four_decimal_places(cfr: bool) -> None:
    timing = {"start_frame": 30, "end_frame": 60} if cfr else {
        "start_s": 1.00103, "end_s": 2.00203,
    }
    request = {**CUT, "start": 1.00101, "end": 2.00201}
    fields = normalize_input({"text": "Tighten this", "target": "story", "request": request,
                              **timing}, _facts(cfr=cfr))
    assert fields["request"] == request
    assert fields["start_s"] == (1.001 if cfr else 1.00103)
    assert fields["end_s"] == (2.002 if cfr else 2.00203)


@pytest.mark.parametrize("cfr", [True, False])
def test_cut_request_patch_retains_replaces_or_clears_with_range(cfr: bool) -> None:
    timing = {"start_frame": 30, "end_frame": 60} if cfr else {
        "start_s": 1.001, "end_s": 2.002,
    }
    fields = normalize_input({"text": "Tighten this", "target": "story", "request": CUT,
                              **timing}, _facts(cfr=cfr))
    current = Comment("c1", **fields, created_at="now", updated_at="now")
    for payload in ({"text": "More detail"},
                    {"text": "More detail", "kind": "fix", "target": "story",
                     "request": CUT, **timing}):
        assert normalize_input(payload, _facts(cfr=cfr), partial=True, current=current) == {
            **fields, "text": "More detail",
        }
    moved = {"end_frame": 90} if cfr else {"end_s": 3.003}
    with pytest.raises(ValidationError) as error:
        normalize_input(moved, _facts(cfr=cfr), partial=True, current=current)
    assert error.value.parameter == "request"
    assert error.value.detail == "a cut request must match the note's range; select the range again"
    replacement = {**CUT, "end": 3.003}
    updated = normalize_input({**moved, "request": replacement}, _facts(cfr=cfr),
                              partial=True, current=current)
    assert updated["request"] == replacement
    assert updated["end_s"] == 3.003
    cleared = normalize_input({**moved, "request": None}, _facts(cfr=cfr),
                              partial=True, current=current)
    assert cleared["request"] is None
    assert cleared["end_s"] == 3.003


@pytest.mark.parametrize("value", [[], "cut", {}, {"type": "unknown"}])
def test_unknown_request_type_message(value) -> None:
    with pytest.raises(ValidationError) as error:
        normalize_input({"text": "Note", "start_frame": 0, "request": value}, _facts())
    assert error.value.parameter == "request"
    assert error.value.detail == "must be a spelling, restore, cut or caption fix request"
