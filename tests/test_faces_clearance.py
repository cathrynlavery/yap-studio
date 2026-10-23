"""Deterministic geometry, timed runs, layout parsing, and clearance reports."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from test_faces_detect import VariableRateDetector, make_clip

from yap_studio.errors import InputError, ValidationError
from yap_studio.faces import (
    DEFAULT_CAPTION_GAP_PX,
    DEFAULT_FACE_CLEARANCE_PX,
    DEFAULT_HAIR,
    LAYOUT_SCHEMA,
    NO_HAIR,
    Box,
    CaptionPhrase,
    Element,
    FaceBox,
    FacesData,
    HairAllowance,
    Layout,
    VideoIdentity,
    check_clearance,
    check_layout,
    detect_faces,
    face_boxes_per_frame,
    find_gaps,
    gap_px,
    head_box,
    interpolate,
    load_layout,
)

FACE = Box(432, 676, 402, 402)
CARD = Element("card-1", "card", Box(60, 280, 960, 360), 0, 300)


def test_caption_box_windows_roundtrip_and_clearance() -> None:
    document = layout_document()
    document["elements"] = []
    document["captions"][0]["boxes"] = [{**FACE.to_dict(), "start_frame": 45, "end_frame": 60}]
    layout = Layout.from_dict(document)
    assert layout.to_dict()["captions"][0]["boxes"] == document["captions"][0]["boxes"]
    faces = [(FACE,)] * 45 + [()] * 15 + [(FACE,)] * 240
    assert not check_clearance(faces, (), layout.captions, margin_px=10, hair=NO_HAIR)
    element = Element("card", "card", FACE, 30, 45)
    assert not check_clearance([()] * 300, (element,), layout.captions,
                               margin_px=0, caption_gap_px=10)


@pytest.mark.parametrize("window", [
    {"start_frame": 29, "end_frame": 60}, {"start_frame": 45, "end_frame": 76},
    {"start_frame": 50, "end_frame": 50}, {"start_frame": True, "end_frame": 60},
    {"start_frame": 45}, {"end_frame": 60},
])
def test_caption_box_windows_must_fit_caption(window: dict) -> None:
    document = layout_document()
    document["captions"][0]["boxes"][0].update(window)
    with pytest.raises(ValidationError, match="boxes"):
        Layout.from_dict(document)


@pytest.mark.parametrize("windows", [[], [[30, 40]]])
def test_caption_face_windows_roundtrip_and_clearance(windows: list) -> None:
    document = layout_document()
    document["elements"] = []
    document["captions"][0].update(face_windows=windows, boxes=[FACE.to_dict()])
    layout = Layout.from_dict(document)
    assert layout.to_dict()["captions"][0]["face_windows"] == windows
    violations = check_clearance([(FACE,)] * 300, (), layout.captions, margin_px=10)
    assert [(v.start_frame, v.end_frame) for v in violations] == (
        [(30, 40)] if windows else [])


def layout_document() -> dict[str, Any]:
    return {
        "schema": LAYOUT_SCHEMA,
        "width": 1080,
        "height": 1920,
        "clearance_px": 100,
        "hair": None,
        "caption_gap_px": 100,
        "elements": [
            {
                "id": "card-1",
                "kind": "card",
                "x": 60,
                "y": 280,
                "w": 960,
                "h": 360,
                "start_frame": 0,
                "end_frame": 300,
            }
        ],
        "captions": [
            {
                "id": "phrase-1",
                "text": "the words on screen",
                "start_frame": 30,
                "end_frame": 75,
                "boxes": [{"x": 210, "y": 1510, "w": 660, "h": 92}],
            }
        ],
    }


def faces_data(*, measured: bool = True, boxes: tuple[Box, ...] = (FACE,)) -> FacesData:
    frames = tuple(tuple(FaceBox(box, 0.9, "detected") for box in boxes) for _ in range(300))
    return FacesData(
        "measured" if measured else "unmeasured",
        None if measured else "face detector unavailable",
        VideoIdentity("clip.mp4", "digest", 300, Fraction(30), 1080, 1920),
        {},
        {},
        frames if measured else (),
        tuple(find_gaps(frames)) if measured else (),
    )


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        (Box(30, 0, 10, 10), 20),
        (Box(0, 40, 10, 10), 30),
        (Box(30, 40, 10, 10), 30),
        (Box(8, 3, 10, 10), -2),
        (Box(10, 0, 10, 10), 0),
        (Box(2, 2, 4, 4), -6),
    ],
)
def test_signed_chebyshev_gap(other: Box, expected: float) -> None:
    origin = Box(0, 0, 10, 10)
    assert gap_px(origin, other) == expected
    assert gap_px(other, origin) == expected


def test_issue_39_card_needs_100_pixels_and_moving_it_up_passes() -> None:
    faces = [(FACE,)] * 300
    violations = check_clearance(faces, [CARD], [], margin_px=100, hair=NO_HAIR)
    assert len(violations) == 1
    run = violations[0]
    assert (run.start_frame, run.end_frame, run.frame) == (0, 300, 0)
    assert (run.kind, run.target_type, run.target_id, run.target_kind) == (
        "face",
        "element",
        "card-1",
        "card",
    )
    assert (run.gap_px, run.required_px, run.target_box, run.face) == (36, 100, CARD.box, FACE)
    assert run.text is None and run.other_id is None
    moved = replace(CARD, box=replace(CARD.box, y=120))
    assert check_clearance(faces, [moved], [], margin_px=100, hair=NO_HAIR) == []


def test_face_drops_in_for_one_run_with_first_worst_frame() -> None:
    faces = [(Box(432, 900, 402, 402),)] * 300
    faces[100:150] = [(FACE,)] * 50
    worst = Box(432, 600, 402, 402)
    faces[125] = (worst,)
    faces[130] = (worst,)
    (run,) = check_clearance(faces, [CARD], [], margin_px=100, hair=NO_HAIR)
    assert (run.start_frame, run.end_frame, run.frame) == (100, 150, 125)
    assert run.gap_px == -40
    assert run.face == worst


def test_face_moving_out_and_back_produces_two_runs() -> None:
    far = Box(432, 900, 402, 402)
    faces = [(FACE,)] * 50 + [(far,)] * 100 + [(FACE,)] * 150
    runs = check_clearance(faces, [CARD], [], margin_px=100, hair=NO_HAIR)
    assert [(run.start_frame, run.end_frame, run.frame) for run in runs] == [
        (0, 50, 0),
        (150, 300, 150),
    ]


def test_sparse_sampling_does_not_clear_a_card_over_the_first_face_position() -> None:
    first, last = (
        FaceBox(Box(0, 20, 10, 12), 0.9, "detected"),
        FaceBox(Box(10, 20, 10, 12), 0.9, "detected"),
    )
    sampled = interpolate({0: (first,), 3: (last,)}, 4)
    boxes = [tuple(face.box for face in frame) for frame in sampled]
    card = Element("card", "card", Box(0, 20, 2, 12), 1, 3)
    assert gap_px(card.box, last.box) == 8
    assert gap_px(card.box, first.box) == -2
    for frame in (1, 2):
        (run,) = check_clearance(
            boxes,
            [replace(card, start_frame=frame, end_frame=frame + 1)],
            [],
            margin_px=0,
            hair=NO_HAIR,
        )
        assert (run.start_frame, run.end_frame) == (frame, frame + 1)
        assert run.gap_px == -2


@pytest.mark.parametrize(
    ("face", "margin", "expected_gap"),
    [
        (Box(0, 110, 10, 10), 100, None),
        (Box(0, 109, 10, 10), 100, 99),
        (Box(0, 10, 10, 10), 0, None),
        (Box(0, 9, 10, 10), 0, -1),
    ],
)
def test_margin_edges(face: Box, margin: float, expected_gap: float | None) -> None:
    element = Element("title-1", "title", Box(0, 0, 10, 10), 0, 1)
    runs = check_clearance([(face,)], [element], [], margin_px=margin, hair=NO_HAIR)
    assert len(runs) == (0 if expected_gap is None else 1)
    if runs:
        assert runs[0].gap_px == expected_gap


@pytest.mark.parametrize("kind", ["card", "title", "overlay"])
def test_targets_only_check_their_window_and_empty_frames_split_runs(kind: str) -> None:
    target = Element("target", kind, Box(0, 0, 10, 10), 1, 5)
    face = Box(0, 0, 10, 10)
    runs = check_clearance(
        [(face,), (), (face,), (), (face,), (face,)], [target], [], margin_px=0, hair=NO_HAIR
    )
    assert [(run.start_frame, run.end_frame) for run in runs] == [(2, 3), (4, 5)]
    assert all(run.target_kind == kind for run in runs)
    assert check_clearance([()] * 6, [target], [], margin_px=100, hair=NO_HAIR) == []


def test_caption_glyphs_and_multiple_faces_pick_the_worst_pair() -> None:
    first, second = Box(0, 0, 10, 10), Box(100, 100, 30, 30)
    phrase = CaptionPhrase("phrase", "visible words", (first, second), 0, 2)
    nearby, worst = Box(0, 20, 10, 10), Box(105, 105, 30, 30)
    (run,) = check_clearance([(nearby, worst)] * 2, [], [phrase], margin_px=100, hair=NO_HAIR)
    assert (run.target_type, run.target_id, run.target_kind, run.text) == (
        "caption",
        "phrase",
        "caption",
        "visible words",
    )
    assert (run.target_box, run.face, run.gap_px) == (second, worst, -25)


def test_worst_caption_glyph_can_change_during_a_run() -> None:
    first, second = Box(0, 0, 10, 10), Box(100, 100, 30, 30)
    phrase = CaptionPhrase("phrase", "visible words", (first, second), 0, 2)
    faces = [(Box(0, 20, 10, 10),), (Box(105, 105, 30, 30),)]
    (run,) = check_clearance(faces, [], [phrase], margin_px=100, hair=NO_HAIR)
    assert (run.start_frame, run.end_frame, run.frame) == (0, 2, 1)
    assert run.target_box == second


def test_issue_15_caption_gap_names_phrase_and_card_and_uses_overlap_window() -> None:
    card = replace(CARD, box=Box(60, 1130, 960, 360), start_frame=40, end_frame=60)
    phrase = CaptionPhrase("phrase", "visible words", (Box(210, 1510, 660, 92),), 30, 75)
    (run,) = check_clearance(
        [()] * 300, [card], [phrase], margin_px=100, caption_gap_px=100, hair=NO_HAIR
    )
    assert (run.kind, run.target_type, run.target_id, run.other_id, run.text) == (
        "caption_gap",
        "caption",
        "phrase",
        "card-1",
        "visible words",
    )
    assert (run.start_frame, run.end_frame, run.frame, run.gap_px, run.required_px) == (
        40,
        60,
        40,
        20,
        100,
    )
    assert run.face is None and run.head is None and run.target_box == phrase.boxes[0]
    assert run.to_dict(lambda frame: frame / 30)["head"] is None
    moved = replace(card, box=replace(card.box, y=1030))
    assert (
        check_clearance(
            [()] * 300, [moved], [phrase], margin_px=100, caption_gap_px=100, hair=NO_HAIR
        )
        == []
    )
    untimed = replace(card, start_frame=75, end_frame=90)
    assert (
        check_clearance(
            [()] * 300, [untimed], [phrase], margin_px=100, caption_gap_px=100, hair=NO_HAIR
        )
        == []
    )
    assert (
        check_clearance(
            [()] * 300, [card], [phrase], margin_px=100, caption_gap_px=None, hair=NO_HAIR
        )
        == []
    )


def test_caption_gap_uses_strict_margin_and_worst_glyph() -> None:
    card = Element("card", "card", Box(0, 0, 100, 100), 0, 2)
    touching = Box(0, 100, 10, 10)
    phrase = CaptionPhrase("phrase", "words", (Box(0, 300, 10, 10), touching), 0, 2)
    assert (
        check_clearance([()] * 2, [card], [phrase], margin_px=0, caption_gap_px=0, hair=NO_HAIR)
        == []
    )
    overlap = replace(phrase, boxes=(phrase.boxes[0], Box(0, 90, 10, 20)))
    (run,) = check_clearance(
        [()] * 2, [card], [overlap], margin_px=0, caption_gap_px=0, hair=NO_HAIR
    )
    assert run.gap_px == -10 and run.target_box == overlap.boxes[1]
    exactly = replace(phrase, boxes=(Box(0, 200, 10, 10),))
    assert (
        check_clearance([()] * 2, [card], [exactly], margin_px=0, caption_gap_px=100, hair=NO_HAIR)
        == []
    )


def test_violation_order_is_time_then_face_then_id() -> None:
    elements = [
        Element("z", "overlay", Box(0, 0, 10, 10), 0, 3),
        Element("a", "title", Box(0, 0, 10, 10), 0, 3),
        Element("early", "card", Box(0, 0, 10, 10), 0, 1),
    ]
    phrase = CaptionPhrase("phrase", "words", (Box(0, 0, 10, 10),), 1, 3)
    runs = check_clearance(
        [(Box(0, 0, 10, 10),)] * 3, elements, [phrase], margin_px=0, caption_gap_px=0, hair=NO_HAIR
    )
    assert [(run.start_frame, run.kind, run.target_id, run.other_id) for run in runs] == [
        (0, "face", "a", None),
        (0, "face", "early", None),
        (0, "face", "z", None),
        (1, "face", "phrase", None),
        (1, "caption_gap", "phrase", "a"),
        (1, "caption_gap", "phrase", "z"),
    ]


@pytest.mark.parametrize("caption", [False, True])
def test_window_past_faces_is_rejected_even_without_any_faces(caption: bool) -> None:
    phrase = CaptionPhrase("phrase", "words", (CARD.box,), 0, 300)
    with pytest.raises(ValueError, match="end_frame"):
        check_clearance(
            [()] * 299,
            [] if caption else [CARD],
            [phrase] if caption else [],
            margin_px=0,
            hair=NO_HAIR,
        )


def test_layout_file_round_trip(tmp_path: Path) -> None:
    document = layout_document()
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    layout = load_layout(path)
    assert layout.to_dict() == document
    assert Layout.from_dict(layout.to_dict()) == layout
    assert layout.elements == (CARD,)


def test_optional_layout_fields_use_defaults() -> None:
    layout = Layout.from_dict({"schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920})
    assert layout.elements == layout.captions == ()
    assert layout.clearance_px is None and layout.hair is None
    assert layout.caption_gap_px == DEFAULT_CAPTION_GAP_PX == 100
    report = check_layout(faces_data(), layout)
    assert report.margin_px == DEFAULT_FACE_CLEARANCE_PX == 24
    assert report.margin_source == "default" and report.hair == DEFAULT_HAIR


@pytest.mark.parametrize(
    ("section", "key", "value", "parameter"),
    [
        (None, "schema", "wrong", "layout.schema"),
        (None, "width", True, "layout.width"),
        (None, "height", 0, "layout.height"),
        (None, "clearance_px", -1, "layout.clearance_px"),
        (None, "caption_gap_px", -1, "layout.caption_gap_px"),
        (None, "clearance_px", float("nan"), "layout.clearance_px"),
        (None, "caption_gap_px", float("inf"), "layout.caption_gap_px"),
        (None, "clearance_px", "100", "layout.clearance_px"),
        (None, "caption_gap_px", False, "layout.caption_gap_px"),
        (None, "elements", {}, "layout.elements"),
        (None, "captions", None, "layout.captions"),
        ("elements", "id", "", "layout.elements[0].id"),
        ("elements", "end_frame", 0, "layout.elements[0].end_frame"),
        ("elements", "start_frame", -1, "layout.elements[0].start_frame"),
        ("elements", "end_frame", 1.5, "layout.elements[0].end_frame"),
        ("elements", "kind", "unknown", "layout.elements[0].kind"),
        ("elements", "w", -1, "layout.elements[0].w"),
        ("elements", "x", "60", "layout.elements[0].x"),
        ("captions", "end_frame", 30, "layout.captions[0].end_frame"),
        ("captions", "boxes", [], "layout.captions[0].boxes"),
        ("captions", "text", 1, "layout.captions[0].text"),
    ],
)
def test_layout_invalid_fields_name_the_json_path(
    section: str | None, key: str, value: Any, parameter: str
) -> None:
    document = layout_document()
    record = document if section is None else document[section][0]
    record[key] = value
    with pytest.raises(ValidationError) as error:
        Layout.from_dict(document)
    assert error.value.parameter == parameter


def test_layout_missing_boxes_and_invalid_nested_box_keep_the_path() -> None:
    document = layout_document()
    del document["captions"][0]["boxes"]
    with pytest.raises(ValidationError) as error:
        Layout.from_dict(document)
    assert error.value.parameter == "layout.captions[0].boxes"
    document = layout_document()
    document["captions"][0]["boxes"][0]["h"] = -1
    with pytest.raises(ValidationError) as error:
        Layout.from_dict(document)
    assert error.value.parameter == "layout.captions[0].boxes[0].h"


@pytest.mark.parametrize("across_types", [False, True])
def test_layout_ids_are_unique(across_types: bool) -> None:
    document = layout_document()
    if across_types:
        document["captions"][0]["id"] = "card-1"
        parameter = "layout.captions[0].id"
    else:
        document["elements"].append(dict(document["elements"][0]))
        parameter = "layout.elements[1].id"
    with pytest.raises(ValidationError) as error:
        Layout.from_dict(document)
    assert error.value.parameter == parameter


def test_load_layout_reports_file_and_json_errors(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    with pytest.raises(InputError):
        load_layout(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid layout JSON"):
        load_layout(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValidationError, match="invalid layout JSON"):
        load_layout(path)
    path.write_text('{"schema": "wrong"}', encoding="utf-8")
    with pytest.raises(ValidationError) as error:
        load_layout(path)
    assert error.value.parameter == "layout.schema"


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("dimension", ["width", "height"])
def test_check_layout_rejects_dimension_mismatch(measured: bool, dimension: str) -> None:
    layout = replace(Layout.from_dict(layout_document()), **{dimension: 100})
    with pytest.raises(ValidationError) as error:
        check_layout(faces_data(measured=measured), layout)
    assert error.value.parameter == f"layout.{dimension}"


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("section", ["elements", "captions"])
def test_check_layout_rejects_windows_past_video(measured: bool, section: str) -> None:
    document = layout_document()
    document[section][0]["end_frame"] = 301
    with pytest.raises(ValidationError) as error:
        check_layout(faces_data(measured=measured), Layout.from_dict(document))
    assert error.value.parameter == f"layout.{section}[0].end_frame"


def test_check_layout_measured_status_margins_and_frame_conversion() -> None:
    faces = faces_data()
    layout = Layout.from_dict(layout_document())
    assert face_boxes_per_frame(faces) == [(FACE,)] * 300
    report = check_layout(faces, layout, hair=NO_HAIR)
    assert (report.status, report.reason, report.margin_px, report.caption_gap_px) == (
        "fail",
        None,
        100,
        100,
    )
    assert report.elements == report.captions == 1
    assert (report.frames_with_faces, report.frames_without_faces) == (300, 0)
    assert len(report.violations) == 1
    passed = check_layout(faces, layout, margin_px=36, caption_gap_px=20, hair=NO_HAIR)
    assert passed.status == "pass" and passed.violations == ()
    assert (passed.margin_px, passed.caption_gap_px) == (36, 20)
    assert check_layout(faces_data(boxes=()), layout).status == "pass"


@pytest.mark.parametrize("with_faces", [0, 17, 300])
def test_report_coverage_counts_the_whole_video(with_faces: int) -> None:
    data = faces_data()
    frames = data.frames[:with_faces] + ((),) * (300 - with_faces)
    data = replace(data, frames=frames, gaps=tuple(find_gaps(frames)))
    layout = Layout(1080, 1920, (replace(CARD, start_frame=100, end_frame=101),), ())
    report = check_layout(data, layout, margin_px=0, hair=NO_HAIR)
    assert report.status == "pass"
    assert (report.frames_with_faces, report.frames_without_faces) == (with_faces, 300 - with_faces)
    assert report.to_dict()["coverage"] == {
        "frames_with_faces": with_faces,
        "frames_without_faces": 300 - with_faces,
    }


def test_unmeasured_faces_still_check_caption_gap() -> None:
    faces = faces_data(measured=False)
    layout = Layout.from_dict(layout_document())
    report = check_layout(faces, layout)
    assert report.status == "unmeasured" and report.reason == faces.reason
    assert report.violations == ()
    assert report.frames_with_faces is report.frames_without_faces is None
    assert report.to_dict()["coverage"] is None
    with pytest.raises(ValueError, match="measured"):
        face_boxes_per_frame(faces)
    card = replace(CARD, box=Box(60, 1130, 960, 360))
    failed = check_layout(faces, replace(layout, elements=(card,)))
    assert failed.status == "fail"
    assert len(failed.violations) == 1 and failed.violations[0].kind == "caption_gap"
    assert failed.to_dict()["coverage"] is None
    assert (
        check_layout(faces, replace(layout, elements=(card,)), caption_gap_px=20).status
        == "unmeasured"
    )


@pytest.mark.parametrize(
    ("face_y", "expected_gap", "json_gap"),
    [(676.12345, 36.12345, 36.12), (638.76544, -1.23456, -1.23)],
)
def test_report_serialization_uses_exact_fps_and_rounded_seconds_and_pixels(
    face_y: float, expected_gap: float, json_gap: float
) -> None:
    face = replace(FACE, y=face_y)
    faces = faces_data(boxes=(face,))
    faces = replace(faces, video=replace(faces.video, fps=Fraction(30000, 1001)))
    layout = Layout(1080, 1920, (replace(CARD, start_frame=1, end_frame=3),), ())
    report = check_layout(faces, layout, margin_px=100.98765, hair=NO_HAIR)
    assert report.violations[0].gap_px == pytest.approx(expected_gap)
    assert report.violations[0].required_px == 100.98765
    document = report.to_dict()
    assert document["schema"] == "yap-studio/clearance-report@1"
    assert document["fps"] == "30000/1001"
    assert document["timing"] == "fps"
    assert document["hair"] == NO_HAIR.to_dict() and document["margin_source"] == "option"
    assert document["status"] == "fail" and document["frames"] == 300
    (run,) = document["violations"]
    assert (run["start_s"], run["end_s"], run["time_s"]) == (0.033, 0.1, 0.033)
    assert run["gap_px"] == json_gap and run["required_px"] == 100.99
    assert run["target_box"] == CARD.box.to_dict() and run["face"] == face.to_dict()
    assert run["head"] == face.to_dict()
    assert json.loads(json.dumps(document, allow_nan=False)) == document


def test_variable_rate_violation_uses_presentation_times(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "variable.mp4", frames=60, fps=Fraction(45, 2))
    data = detect_faces(clip, detector=VariableRateDetector())
    card = Element("card", "card", Box(10, 20, 10, 12), 30, 60)
    report = check_layout(data, Layout(108, 192, (card,), (), clearance_px=0), hair=NO_HAIR)
    assert report.times == data.times and report.end_time == data.end_time
    assert report.time_at(30) == 2.0
    assert report.time_at(60) == 3.0
    document = report.to_dict()
    assert document["timing"] == "pts"
    (violation,) = document["violations"]
    assert violation["start_frame"] == 30
    assert violation["start_s"] == violation["time_s"] == 2.0
    assert violation["end_s"] == 3.0
    assert document["coverage"] == {"frames_with_faces": 60, "frames_without_faces": 0}


@pytest.mark.parametrize("timed", [False, True])
def test_report_time_at_supports_end_and_rejects_invalid_indices(timed: bool) -> None:
    data = faces_data()
    if timed:
        data = replace(data, times=tuple(i / 15 for i in range(300)), end_time=20.0)
    report = check_layout(data, Layout(1080, 1920, (), ()))
    assert report.time_at(0) == 0.0
    assert report.time_at(300) == (20.0 if timed else 10.0)
    for frame in (-1, 301, True, 1.0):
        with pytest.raises(IndexError):
            report.time_at(frame)


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf"), True, "100"])
def test_invalid_runtime_margins_are_rejected(margin: Any) -> None:
    for key in ("margin_px", "caption_gap_px"):
        arguments = {"margin_px": 0, key: margin}
        with pytest.raises(ValueError, match=key):
            check_clearance([], [], [], **arguments, hair=NO_HAIR)
        with pytest.raises(ValidationError) as error:
            check_layout(faces_data(), Layout(1080, 1920, (), ()), **{key: margin})
        assert error.value.parameter == key


def test_head_box_default_arithmetic_and_no_hair_identity() -> None:
    face = Box(458, 629, 396, 396)
    head = head_box(face)
    assert (head.x, head.y, head.w, head.h) == pytest.approx((398.6, 391.4, 514.8, 633.6))
    assert head.y1 == pytest.approx(face.y1)
    assert head_box(face, NO_HAIR) == face
    assert head_box(Box(0, 10, 20, 100)).y == -50


@pytest.mark.parametrize("field", ["above", "sides"])
@pytest.mark.parametrize("value", [-0.01, 2.01, float("nan"), float("inf"), True, "0.6"])
def test_hair_allowance_rejects_invalid_ratios(field: str, value: Any) -> None:
    with pytest.raises(ValidationError) as caught:
        HairAllowance(**{field: value})
    assert caught.value.parameter == f"hair.{field}"
    assert caught.value.detail == "must be a finite number from 0 to 2"
    with pytest.raises(ValidationError) as caught:
        HairAllowance.from_dict({field: value}, "layout.hair")
    assert caught.value.parameter == f"layout.hair.{field}"
    assert caught.value.detail == "must be a finite number from 0 to 2"


@pytest.mark.parametrize("data", [None, [], "hair", 0, {1: 0.6}])
def test_hair_allowance_requires_an_object(data: Any) -> None:
    with pytest.raises(ValidationError) as caught:
        HairAllowance.from_dict(data, "layout.hair")
    assert caught.value.parameter == "layout.hair"


def test_hair_allowance_round_trip_defaults_and_unknown_key_path() -> None:
    assert DEFAULT_HAIR.to_dict() == {"above": 0.6, "sides": 0.15}
    assert HairAllowance.from_dict({}) == DEFAULT_HAIR
    assert HairAllowance.from_dict({"above": 0}) == HairAllowance(0, 0.15)
    assert HairAllowance.from_dict({"sides": 2}) == HairAllowance(0.6, 2)
    hair = HairAllowance(2, 0)
    assert HairAllowance.from_dict(hair.to_dict()) == hair
    with pytest.raises(ValidationError) as caught:
        HairAllowance.from_dict({"below": 0.1}, "layout.hair")
    assert caught.value.parameter == "layout.hair.below"
    assert "unknown" in caught.value.detail


def test_issue_39_hair_clearance_requires_card_at_top() -> None:
    face = Box(458, 629, 396, 396)
    faces = [(face,)] * 300
    head = head_box(face)
    for y, expected_gap in ((280, -248.6), (120, -88.6)):
        card = replace(CARD, box=replace(CARD.box, y=y))
        (violation,) = check_clearance(faces, [card], [], margin_px=DEFAULT_FACE_CLEARANCE_PX)
        assert (violation.start_frame, violation.end_frame, violation.frame) == (0, 300, 0)
        assert violation.gap_px == pytest.approx(expected_gap)
        assert violation.required_px == 24
        assert violation.face == face and violation.head == head
    old_card = replace(CARD, box=replace(CARD.box, y=120))
    assert check_clearance(faces, [old_card], [], margin_px=100, hair=NO_HAIR) == []
    top_card = replace(CARD, box=replace(CARD.box, y=0))
    assert gap_px(top_card.box, head) == pytest.approx(31.4)
    assert check_clearance(faces, [top_card], [], margin_px=DEFAULT_FACE_CLEARANCE_PX) == []
    layout = Layout(1080, 1920, (old_card,), ())
    report = check_layout(faces_data(boxes=(face,)), layout)
    assert report.status == "fail" and report.hair == DEFAULT_HAIR
    assert report.margin_px == 24 and report.margin_source == "default"
    assert report.violations[0].head == head


def test_hair_side_allowance_includes_clearance_margin() -> None:
    face = Box(458, 629, 396, 396)
    edge = face.x1 + 0.15 * face.w + DEFAULT_FACE_CLEARANCE_PX
    inside = Element("overlay", "overlay", Box(edge - 0.01, 629, 50, 100), 0, 1)
    outside = replace(inside, box=replace(inside.box, x=edge + 0.01))
    (violation,) = check_clearance([(face,)], [inside], [], margin_px=24)
    assert violation.gap_px == pytest.approx(23.99)
    assert violation.head == head_box(face)
    assert gap_px(outside.box, head_box(face)) == pytest.approx(24.01)
    assert check_clearance([(face,)], [outside], [], margin_px=24) == []


def test_caption_glyphs_measure_head_and_keep_worst_head_frame() -> None:
    first = Box(458, 650, 396, 396)
    worst = Box(458, 629, 396, 396)
    glyph = Box(60, 120, 960, 360)
    phrase = CaptionPhrase("phrase", "words", (Box(0, 1500, 10, 10), glyph), 0, 3)
    (run,) = check_clearance([(first,), (worst,), (worst,)], [], [phrase], margin_px=24)
    assert run.target_type == "caption" and run.target_box == glyph
    assert (run.start_frame, run.end_frame, run.frame) == (0, 3, 1)
    assert run.face == worst and run.head == head_box(worst)
    assert run.gap_px == pytest.approx(-88.6)


@pytest.mark.parametrize("explicit_null", [False, True])
def test_layout_unset_clearance_and_hair_serialize_as_null(explicit_null: bool) -> None:
    document = {"schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920}
    if explicit_null:
        document.update(clearance_px=None, hair=None)
    layout = Layout.from_dict(document)
    assert layout.clearance_px is None and layout.hair is None
    assert layout.to_dict()["clearance_px"] is None and layout.to_dict()["hair"] is None
    assert Layout.from_dict(layout.to_dict()) == layout


def test_layout_hair_round_trip() -> None:
    document = layout_document()
    document["hair"] = {"above": 0.4, "sides": 0.2}
    layout = Layout.from_dict(document)
    assert layout.hair == HairAllowance(0.4, 0.2)
    assert layout.to_dict() == document
    assert Layout.from_dict(layout.to_dict()) == layout


@pytest.mark.parametrize("field", ["above", "sides"])
@pytest.mark.parametrize("value", [-1, 3, float("nan"), float("inf"), True, "0.6"])
def test_layout_bad_hair_keeps_component_path(field: str, value: Any) -> None:
    document = layout_document()
    document["hair"] = {field: value}
    with pytest.raises(ValidationError) as caught:
        Layout.from_dict(document)
    assert caught.value.parameter == f"layout.hair.{field}"


@pytest.mark.parametrize(
    ("layout_margin", "option_margin", "source", "expected_margin", "expected_source"),
    [
        (None, None, None, 24, "default"),
        (0, None, None, 0, "layout"),
        (5, None, None, 5, "layout"),
        (5, 0, None, 0, "option"),
        (5, 10, None, 10, "option"),
        (None, 50, "style", 50, "style"),
    ],
)
def test_check_layout_margin_precedence_and_source(
    layout_margin: float | None,
    option_margin: float | None,
    source: str | None,
    expected_margin: float,
    expected_source: str,
) -> None:
    layout = Layout(1080, 1920, (), (), clearance_px=layout_margin)
    report = check_layout(faces_data(), layout, margin_px=option_margin, margin_source=source)
    assert report.margin_px == expected_margin and report.margin_source == expected_source
    assert report.to_dict()["margin_source"] == expected_source


@pytest.mark.parametrize(
    ("layout_hair", "option_hair", "expected"),
    [
        (None, None, DEFAULT_HAIR),
        (NO_HAIR, None, NO_HAIR),
        (HairAllowance(0.4, 0.2), None, HairAllowance(0.4, 0.2)),
        (HairAllowance(0.4, 0.2), NO_HAIR, NO_HAIR),
        (None, HairAllowance(1, 0.1), HairAllowance(1, 0.1)),
    ],
)
def test_check_layout_hair_precedence(
    layout_hair: HairAllowance | None,
    option_hair: HairAllowance | None,
    expected: HairAllowance,
) -> None:
    layout = Layout(1080, 1920, (), (), hair=layout_hair)
    report = check_layout(faces_data(), layout, hair=option_hair)
    assert report.hair == expected and report.to_dict()["hair"] == expected.to_dict()
