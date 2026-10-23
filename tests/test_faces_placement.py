"""Deterministic placement geometry and agreement with timed clearance checks."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from types import MappingProxyType

import pytest

from yap_studio.faces import (
    DEFAULT_FACE_CLEARANCE_PX,
    DEFAULT_HAIR,
    HEADROOM_BAND,
    NO_HAIR,
    TOP_BAND,
    Band,
    Box,
    Element,
    HairAllowance,
    PlacementHint,
    caption_band,
    check_clearance,
    gap_px,
    head_box,
    headroom_band,
    suggest_offset,
    top_band,
)
from yap_studio.style import resolve

FACE = Box(432, 676, 402, 402)


def test_bands_from_shipped_resolved_style() -> None:
    style = resolve("preset:clean_minimal").data
    assert headroom_band(style) == HEADROOM_BAND == Band(280, 640)
    assert caption_band(style) == Band(640, 1740)
    assert top_band(style) == TOP_BAND == Band(0, 640)


def test_bands_use_the_supplied_mapping() -> None:
    style = MappingProxyType(
        {
            "internal": {
                "key_text_top_px": 10,
                "key_text_bottom_px": 50,
                "caption_safe_bottom_px": 150,
            }
        }
    )
    assert headroom_band(style) == Band(10, 50)
    assert caption_band(style) == Band(50, 150)
    assert top_band(style) == Band(0, 50)


@pytest.mark.parametrize(
    ("top", "bottom"),
    [(-1, 10), (0, 0), (10, 5), (0.5, 10), (0, 10.5), (False, 10), (0, True)],
)
def test_invalid_band(top: int, bottom: int) -> None:
    with pytest.raises(ValueError, match="0 <= top < bottom"):
        Band(top, bottom)


@pytest.mark.parametrize("faces", [[], [()], [(), (), ()]])
def test_no_faces_keeps_centred_preference(faces: Sequence[Sequence[Box]]) -> None:
    hint = suggest_offset(faces, x=60, width=960, height=100, band=HEADROOM_BAND, margin_px=100)
    assert isinstance(hint, PlacementHint)
    assert (hint.y, hint.offset_px, hint.preferred_y) == (410, 0, 410)
    assert hint.reason == "no face in the window; keep the preferred position"
    assert hint.best_y is None and hint.best_gap_px is None
    assert hint.to_dict() == {
        "y": 410,
        "offset_px": 0,
        "preferred_y": 410,
        "band": {"top": 280, "bottom": 640},
        "margin_px": 100,
        "hair": DEFAULT_HAIR.to_dict(),
        "reason": hint.reason,
        "best_y": None,
        "best_gap_px": None,
    }


@pytest.mark.parametrize(("preferred", "expected"), [(-100, 280), (1000, 540), (420, 420)])
def test_explicit_preference_is_clamped(preferred: int, expected: int) -> None:
    hint = suggest_offset(
        [],
        x=60,
        width=960,
        height=100,
        band=HEADROOM_BAND,
        margin_px=100,
        preferred_y=preferred,
    )
    assert (hint.y, hint.preferred_y, hint.offset_px) == (expected, expected, 0)


def test_low_face_leaves_preferred_position_clear() -> None:
    hint = suggest_offset(
        [(Box(432, 1000, 402, 402),)],
        x=60,
        width=960,
        height=100,
        band=HEADROOM_BAND,
        margin_px=100,
        hair=NO_HAIR,
    )
    assert (hint.y, hint.offset_px) == (410, 0)
    assert hint.reason == "clear of the head at the preferred position"
    assert hint.best_y is None and hint.best_gap_px is None


def test_short_card_moves_up_to_exact_margin() -> None:
    face = Box(432, 600, 402, 402)
    hint = suggest_offset(
        [(face,)],
        x=60,
        width=960,
        height=100,
        band=HEADROOM_BAND,
        margin_px=100,
        hair=NO_HAIR,
    )
    assert (hint.y, hint.offset_px, hint.preferred_y) == (400, -10, 410)
    assert gap_px(Box(60, hint.y, 960, 100), face) == 100
    assert gap_px(Box(60, hint.y + 1, 960, 100), face) == 99
    assert hint.reason == "moved up 10 px to stay 100 px clear of the head"
    assert hint.best_y is None and hint.best_gap_px is None


def test_real_card_has_no_position_at_100_pixels_clearance() -> None:
    hint = suggest_offset(
        [(FACE,)] * 5,
        x=60,
        width=960,
        height=360,
        band=HEADROOM_BAND,
        margin_px=100,
        hair=NO_HAIR,
    )
    assert hint.y is None and hint.offset_px is None
    assert (hint.preferred_y, hint.best_y, hint.best_gap_px) == (280, 280, 36)
    assert hint.reason == (
        "no position in the band 280 to 640 keeps a 360 px element 100 px clear of the head; "
        "the best is y 280 with a gap of 36 px"
    )
    document = hint.to_dict()
    assert document["y"] is None and document["offset_px"] is None
    assert document["best_y"] == 280 and document["best_gap_px"] == 36


def test_real_card_fits_with_30_pixels_clearance() -> None:
    hint = suggest_offset(
        [(FACE,)],
        x=60,
        width=960,
        height=360,
        band=HEADROOM_BAND,
        margin_px=30,
        hair=NO_HAIR,
    )
    assert (hint.y, hint.offset_px) == (280, 0)
    assert hint.reason == "clear of the head at the preferred position"


def test_caption_moves_below_face_to_closest_position() -> None:
    band = caption_band(resolve("preset:clean_minimal").data)
    hint = suggest_offset(
        [(FACE,)] * 3, x=40, width=1000, height=200, band=band, margin_px=100, hair=NO_HAIR
    )
    assert (hint.y, hint.preferred_y, hint.offset_px) == (1178, 1090, 88)
    assert hint.y == FACE.y1 + 100
    assert gap_px(Box(40, hint.y, 1000, 200), FACE) == 100
    assert gap_px(Box(40, hint.y - 1, 1000, 200), FACE) == 99
    assert hint.reason == "moved down 88 px to stay 100 px clear of the head"


@pytest.mark.parametrize("face_x", [-210, 200, 250])
def test_horizontally_clear_faces_do_not_constrain_y(face_x: float) -> None:
    hint = suggest_offset(
        [(Box(face_x, 0, 10, 1000),)],
        x=0,
        width=100,
        height=100,
        band=HEADROOM_BAND,
        margin_px=100,
        hair=NO_HAIR,
    )
    assert (hint.y, hint.offset_px) == (410, 0)
    assert hint.reason == "clear of the head at the preferred position"


def test_union_of_moving_faces_constrains_the_whole_window() -> None:
    first = Box(0, 60, 20, 20)
    second = Box(0, 0, 20, 20)
    options = {"x": 0, "width": 20, "height": 20, "band": Band(0, 100), "margin_px": 10}
    assert suggest_offset([(first,)], **options, hair=NO_HAIR).y == 30
    assert suggest_offset([(second,)], **options, hair=NO_HAIR).y == 40
    hint = suggest_offset([(first,), (), (second,), (first, first)], **options, hair=NO_HAIR)
    assert (hint.y, hint.offset_px) == (30, -10)
    assert suggest_offset([(second, first)], **options, hair=NO_HAIR) == hint
    assert all(gap_px(Box(0, hint.y, 20, 20), face) >= 10 for face in (first, second))


def test_moving_faces_can_leave_no_common_position() -> None:
    options = {"x": 0, "width": 20, "height": 20, "band": Band(0, 100), "margin_px": 10}
    first = Box(0, 0, 20, 40)
    second = Box(0, 60, 20, 40)
    assert suggest_offset([(first,)], **options, hair=NO_HAIR).y == 50
    assert suggest_offset([(second,)], **options, hair=NO_HAIR).y == 30
    hint = suggest_offset([(first,), (), (second,)], **options, hair=NO_HAIR)
    assert hint.y is None
    assert (hint.best_y, hint.best_gap_px) == (40, 0)


def test_element_taller_than_band_returns_fit_reason() -> None:
    hint = suggest_offset(
        [(FACE,)],
        x=60,
        width=960,
        height=361,
        band=HEADROOM_BAND,
        margin_px=100,
    )
    assert hint.y is None and hint.offset_px is None
    assert hint.best_y is None and hint.best_gap_px is None
    assert hint.reason == "a 361 px element does not fit the band 280 to 640"


def test_equal_distance_clear_positions_choose_smaller_y() -> None:
    hint = suggest_offset(
        [(Box(0, 40, 20, 20),)],
        x=0,
        width=20,
        height=20,
        band=Band(0, 100),
        margin_px=10,
        hair=NO_HAIR,
    )
    assert (hint.y, hint.offset_px) == (10, -30)
    assert gap_px(Box(0, 70, 20, 20), Box(0, 40, 20, 20)) == 10


def test_touching_open_intervals_preserve_the_clear_integer() -> None:
    faces = [(Box(0, 0, 20, 20), Box(0, 40, 20, 20))]
    hint = suggest_offset(
        faces,
        x=0,
        width=20,
        height=20,
        band=Band(0, 60),
        margin_px=0,
        preferred_y=0,
        hair=NO_HAIR,
    )
    assert (hint.y, hint.offset_px) == (20, 20)
    assert all(gap_px(Box(0, hint.y, 20, 20), face) == 0 for face in faces[0])


@pytest.mark.parametrize(("face_y", "expected_y"), [(50, 20), (50.25, 20), (49.75, 19)])
def test_fractional_interval_ends_keep_only_clear_integers(face_y: float, expected_y: int) -> None:
    face = Box(0, face_y, 20, 100)
    hint = suggest_offset(
        [(face,)],
        x=0,
        width=20,
        height=20,
        band=Band(0, 100),
        margin_px=10,
        hair=NO_HAIR,
    )
    assert hint.y == expected_y
    assert gap_px(Box(0, hint.y, 20, 20), face) >= 10
    assert gap_px(Box(0, hint.y + 1, 20, 20), face) < 10


def test_fractional_element_height_has_integer_preference_and_fitting_range() -> None:
    hint = suggest_offset([], x=0, width=20, height=20.5, band=Band(0, 100), margin_px=10)
    assert (hint.y, hint.preferred_y) == (39, 39)
    assert type(hint.y) is int
    clamped = suggest_offset(
        [],
        x=0,
        width=20,
        height=20.5,
        band=Band(0, 100),
        margin_px=10,
        preferred_y=100,
    )
    assert (clamped.y, clamped.preferred_y) == (79, 79)
    assert clamped.y + 20.5 <= 100


@pytest.mark.parametrize(
    ("face", "expected_y"), [(Box(0, 1.3, 20, 10), 0), (Box(0, 0, 20, 0.8), 2)]
)
def test_decimal_boundaries_use_exact_clearance_arithmetic(face: Box, expected_y: int) -> None:
    hint = suggest_offset(
        [(face,)],
        x=0,
        width=20,
        height=0.1,
        band=Band(0, 3),
        margin_px=0.2,
        hair=NO_HAIR,
    )
    assert (hint.y, hint.preferred_y) == (expected_y, 1)
    assert gap_px(Box(0, 1, 20, 0.1), face) < 0.2
    element = Element("decimal", "card", Box(0, hint.y, 20, 0.1), 0, 1)
    assert check_clearance([(face,)], [element], [], margin_px=0.2, hair=NO_HAIR) == []


def test_blocked_band_best_gap_ties_choose_closest_to_preference() -> None:
    hint = suggest_offset(
        [(Box(5, 0, 20, 100),)],
        x=0,
        width=10,
        height=10,
        band=Band(0, 100),
        margin_px=10,
        preferred_y=37,
        hair=NO_HAIR,
    )
    assert hint.y is None
    assert (hint.best_y, hint.best_gap_px) == (37, -5)
    assert hint.reason == (
        "no position in the band 0 to 100 keeps a 10 px element 10 px clear of the head; "
        "the best is y 37 with an overlap of 5 px"
    )


def test_blocked_band_best_gap_considers_every_face_and_breaks_remaining_tie_upwards() -> None:
    faces = [(Box(0, 25, 20, 40),), (Box(0, 35, 20, 40),)]
    hint = suggest_offset(
        faces,
        x=0,
        width=20,
        height=20,
        band=Band(0, 100),
        margin_px=10,
        hair=NO_HAIR,
    )
    assert hint.y is None
    assert (hint.preferred_y, hint.best_y, hint.best_gap_px) == (40, 0, 5)
    assert min(gap_px(Box(0, 80, 20, 20), face) for frame in faces for face in frame) == 5


@pytest.mark.parametrize("hair", [NO_HAIR, DEFAULT_HAIR, HairAllowance(0.43, 0.12)])
def test_seeded_grid_agrees_with_clearance_and_brute_force_placement(hair: HairAllowance) -> None:
    rng = random.Random(7)
    outcomes = {"clear": 0, "blocked": 0, "too_tall": 0}
    for case in range(200):
        top = rng.randrange(30)
        band = Band(top, top + rng.randrange(5, 61))
        x = rng.randrange(-20, 41) / 2
        width = rng.randrange(0, 81) / 2
        height = rng.randrange(0, 141) / 2
        margin = rng.randrange(0, 41) / 2
        faces = [
            tuple(
                Box(
                    rng.randrange(-40, 101) / 2,
                    rng.randrange(-40, 181) / 2,
                    rng.randrange(0, 81) / 2,
                    rng.randrange(0, 121) / 2,
                )
                for _ in range(rng.randrange(4))
            )
            for _ in range(rng.randrange(1, 6))
        ]
        preferred = None if rng.randrange(2) else rng.randrange(top - 20, band.bottom + 21)
        hint = suggest_offset(
            faces,
            x=x,
            width=width,
            height=height,
            band=band,
            margin_px=margin,
            preferred_y=preferred,
            hair=hair,
        )
        if height > band.bottom - band.top:
            outcomes["too_tall"] += 1
            assert hint.y is None and hint.best_y is None and hint.best_gap_px is None, case
            continue

        allowed = range(band.top, math.floor(band.bottom - height) + 1)
        gaps = {
            y: min(
                (
                    gap_px(Box(x, y, width, height), head_box(face, hair))
                    for frame in faces
                    for face in frame
                ),
                default=math.inf,
            )
            for y in allowed
        }
        clear_positions = [y for y in allowed if gaps[y] >= margin]
        if clear_positions:
            outcomes["clear"] += 1
            expected_y = min(clear_positions, key=lambda y: (abs(y - hint.preferred_y), y))
            assert hint.y == expected_y, case
            assert hint.offset_px == hint.y - hint.preferred_y, case
            assert hint.best_y is None and hint.best_gap_px is None, case
            element = Element("target", "card", Box(x, hint.y, width, height), 0, len(faces))
            assert check_clearance(faces, [element], [], margin_px=margin, hair=hair) == [], case
        else:
            outcomes["blocked"] += 1
            assert hint.y is None and hint.offset_px is None, case
            expected_best = min(allowed, key=lambda y: (-gaps[y], abs(y - hint.preferred_y), y))
            assert (hint.best_y, hint.best_gap_px) == (expected_best, gaps[expected_best]), case
            for y in allowed:
                element = Element("target", "card", Box(x, y, width, height), 0, len(faces))
                assert check_clearance(faces, [element], [], margin_px=margin, hair=hair), (case, y)
    assert all(outcomes.values()), outcomes


def test_issue_39_top_band_keeps_card_clear_of_hair_at_preferred_position() -> None:
    face = Box(458, 629, 396, 396)
    hint = suggest_offset(
        [(face,)] * 300,
        x=60,
        width=960,
        height=360,
        band=TOP_BAND,
        margin_px=DEFAULT_FACE_CLEARANCE_PX,
        preferred_y=0,
    )
    assert (hint.y, hint.preferred_y, hint.offset_px) == (0, 0, 0)
    assert hint.reason == "clear of the head at the preferred position"
    assert hint.best_y is None and hint.best_gap_px is None
    assert hint.hair == DEFAULT_HAIR and hint.to_dict()["hair"] == DEFAULT_HAIR.to_dict()
    assert hint.margin_px == 24
    assert gap_px(Box(60, hint.y, 960, 360), head_box(face)) == pytest.approx(31.4)
    card = Element("card", "card", Box(60, hint.y, 960, 360), 0, 300)
    assert check_clearance([(face,)] * 300, [card], [], margin_px=24) == []


def test_issue_39_headroom_band_reports_best_gap_against_head() -> None:
    face = Box(458, 629, 396, 396)
    hint = suggest_offset(
        [(face,)],
        x=60,
        width=960,
        height=360,
        band=HEADROOM_BAND,
        margin_px=DEFAULT_FACE_CLEARANCE_PX,
    )
    assert hint.y is None and hint.offset_px is None
    assert hint.preferred_y == hint.best_y == 280
    assert hint.best_gap_px == pytest.approx(-248.6)
    assert hint.best_gap_px == gap_px(Box(60, 280, 960, 360), head_box(face))
    assert hint.reason == (
        "no position in the band 280 to 640 keeps a 360 px element 24 px clear of the head; "
        "the best is y 280 with an overlap of 249 px"
    )
