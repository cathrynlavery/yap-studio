"""Deterministic Y'CbCr effects on synthetic soft-edged people."""

from __future__ import annotations

import numpy as np
import pytest

from yap_studio.render.moments import moment_frame, person_colours


@pytest.fixture
def scene() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:192, :256]
    alpha = np.clip((42 - np.maximum(abs(x - 128), abs(y - 96))) / 3, 0, 1)
    picture = np.empty((192, 256, 3), dtype=np.uint8)
    picture[..., 0] = 70 + 80 * ((x // 4 + y // 4) % 2)
    picture[..., 1:] = (100, 145)
    picture[alpha > 0] = (160, 190, 70)
    return picture, alpha


def effect(style: str, picture: np.ndarray, alpha: np.ndarray, index: int = 15,
           frames: int = 36, **kwargs: object) -> np.ndarray:
    return moment_frame(style, picture, alpha, index=index, frames=frames, fps=30,
                        accent_color="#FF0000", **kwargs)


def test_person_colour_hook_is_identity(scene: tuple) -> None:
    picture, alpha = scene
    assert person_colours(picture, alpha) is picture


def test_color_pop_preserves_person_chroma(scene: tuple) -> None:
    picture, alpha = scene
    out = effect("color_pop", picture, alpha)
    np.testing.assert_array_equal(out[alpha == 1], picture[alpha == 1])
    assert np.all(out[alpha == 0, 1:] == 128)
    np.testing.assert_array_equal(out[..., 0], picture[..., 0])
    assert np.any(out[(alpha > 0) & (alpha < 1), 1] != 128)


def test_background_blur_fills_before_blurring(scene: tuple) -> None:
    picture, alpha = scene
    out = effect("background_blur", picture, alpha)
    np.testing.assert_array_equal(out[alpha == 1], picture[alpha == 1])
    np.testing.assert_array_equal(out[alpha == 0, 1:], picture[alpha == 0, 1:])
    assert out[:30, :30, 0].std() < picture[:30, :30, 0].std() / 4


def test_spotlight_falloff_preserves_person(scene: tuple) -> None:
    picture, alpha = scene
    picture[..., 0] = 160
    out = effect("spotlight", picture, alpha)
    np.testing.assert_array_equal(out[alpha == 1], picture[alpha == 1])
    assert out[96, 173, 0] > out[0, 0, 0]
    assert out[0, 0, 0] == pytest.approx(16 + (160 - 16) * 0.3, abs=1)


def test_spotlight_darkens_room_at_delivery_size() -> None:
    y, x = np.mgrid[:1920, :1080]
    distance = np.maximum(abs(x - 540) / 180, abs(y - 960) / 600)
    alpha = np.clip((1 - distance) * 30, 0, 1)
    picture = np.full((1920, 1080, 3), (200, 100, 145), dtype=np.uint8)
    picture[alpha > 0] = (160, 190, 70)
    out = effect("spotlight", picture, alpha)
    np.testing.assert_array_equal(out[alpha == 1], picture[alpha == 1])
    # Both the nearby room and the point 300 px beyond the right edge read as darkened.
    for offset in (60, 300):
        sample = (960, 720 + offset, 0)
        assert alpha[sample[:2]] == 0
        assert int(out[sample]) - 16 <= (int(picture[sample]) - 16) / 2


def test_outline_uses_accent_only_near_edge(scene: tuple) -> None:
    picture, alpha = scene
    out = effect("outline", picture, alpha)
    np.testing.assert_array_equal(out[:20], picture[:20])
    np.testing.assert_array_equal(out[alpha > 0], picture[alpha > 0])
    assert out[96, 171, 2] > picture[96, 171, 2]
    assert out[96, 171, 1] < 128


def test_outline_has_a_visible_ring_at_delivery_width() -> None:
    alpha = np.zeros((192, 1080))
    alpha[50:150, 440:640] = 1
    picture = np.full((192, 1080, 3), (100, 100, 145), dtype=np.uint8)
    picture[alpha > 0] = (160, 190, 70)
    out = effect("outline", picture, alpha)
    np.testing.assert_array_equal(out[alpha > 0], picture[alpha > 0])
    # The fourth background pixel carries the bright ring; the farther point has only glow.
    assert int(out[100, 643, 2]) - int(picture[100, 643, 2]) > 50
    assert out[100, 655, 2] > picture[100, 655, 2]
    np.testing.assert_array_equal(out[:, 700:], picture[:, 700:])


@pytest.mark.parametrize("style", ["color_pop", "background_blur", "spotlight", "outline"])
def test_easing_starts_and_ends_live(scene: tuple, style: str) -> None:
    picture, alpha = scene
    for index in (0, 35):
        np.testing.assert_array_equal(effect(style, picture, alpha, index), picture)
    middle = effect(style, picture, alpha)
    assert np.any(middle != picture)
    assert np.abs(effect(style, picture, alpha, 1).astype(float) - picture).mean() < (
        np.abs(middle.astype(float) - picture).mean())


def test_sticker_freeze_holds_transformed_first_frame_and_white_ring(scene: tuple) -> None:
    first, alpha = scene
    later = np.full_like(first, (60, 80, 150))
    kwargs = {"frozen": first, "frozen_alpha": alpha}
    flash = effect("sticker_freeze", first, alpha, 0, **kwargs)
    assert np.all(flash == (235, 128, 128))
    held = effect("sticker_freeze", later, alpha, 6, **kwargs)
    for index in range(7, 30):
        np.testing.assert_array_equal(effect("sticker_freeze", later, alpha, index, **kwargs), held)
    np.testing.assert_array_equal(effect("sticker_freeze", later, alpha, 35, **kwargs), later)
    # The penultimate frame still contains the freeze; the return uses all six frames.
    assert np.any(effect("sticker_freeze", later, alpha, 34, **kwargs) != later)
    # The frozen backdrop stays fixed, while a transformed silhouette and white ring pop forward.
    np.testing.assert_array_equal(held[0, 0], first[0, 0])
    assert np.count_nonzero(held[..., 0] == 235) > 0
    changed = np.any(held != first, axis=-1)
    assert changed[48:145, 78:180].any()
    assert np.count_nonzero(np.all(held == (160, 190, 70), axis=-1)) > np.count_nonzero(alpha == 1)
    # A tilted top edge reaches farther up on one side of the person.
    white = (held[..., 0] > 225) & (abs(held[..., 1].astype(int) - 128) < 3)
    ys, xs = np.nonzero(white)
    assert ys[xs < 128].min() != ys[xs > 128].min()
    assert np.ptp(xs) > np.ptp(np.nonzero(alpha)[1])
