"""Shared compare motion and deterministic style-coloured handles."""

import math

import numpy as np
import pytest

from yap_studio import grade
from yap_studio.style import resolve_style_data


def test_compare_position_and_expression():
    points = [(0, 0), (0.30, 0.78), (0.52, 0.30), (0.68, 0.50), (0.86, 0.50), (1, 0)]
    for share, position in points:
        assert grade.compare_position(round(share * 100), 101) == pytest.approx(position)
    expression = grade.compare_x_expr(37, 101, 1080)
    for frame in range(101):
        actual = eval(expression.replace("if(", "choose("), {"__builtins__": {}}, {
            "choose": lambda condition, yes, no: yes if condition else no,
            "lt": lambda a, b: a < b, "cos": math.cos, "PI": math.pi,
            "round": round, "n": frame + 37,
        })
        assert actual == pytest.approx(round(1080 * grade.compare_position(frame, 101)))
    assert grade.compare_position(75, 101) == 0.5
    assert grade.compare_position(0, 101) == grade.compare_position(100, 101) == 0


def test_compare_handle_uses_style_colours():
    style = resolve_style_data({"schema_version": 1, "captions": {"text_color": "#F8E4A0"},
                                "graphics": {"ink_color": "#123456"}})
    width, height, data = grade.handle_rgba(style)
    assert (width, height) == (96, 1920)
    assert data == grade.handle_rgba(style)[2]
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    knob_y = round(height * 0.66)
    np.testing.assert_array_equal(image[100, width // 2], [248, 228, 160, 255])
    for y, x in ((knob_y, 77), (knob_y-1, 34), (knob_y-1, 61)):
        np.testing.assert_array_equal(image[y, x], [18, 52, 86, 255])
    assert np.count_nonzero(np.all(image[knob_y-40:knob_y+40, :, :3]
                                  == [18, 52, 86], axis=-1)) > 100
    assert not np.any(np.all(image[920:1000, :, :3] == [18, 52, 86], axis=-1))
    assert image[100, 0, 3] == 0


def test_slider_style_default_and_grade_resolution():
    from yap_studio.grade import resolve_grade

    style = resolve_style_data({"schema_version": 1})
    assert style["grade"]["slider"] is False
    for look in ("film", "eyedropper"):
        style["grade"] = {"look": look, "color": "#1E6BD6", "slider": True}
        assert resolve_grade(style, None).look == look


@pytest.mark.parametrize("height,width", [(160, 90), (160, 160), (240, 90)])
def test_compare_preview_has_raw_left_grade_right_and_shared_handle(height, width):
    raw = np.full((height, width, 3), (32, 80, 110), dtype=np.uint8)
    spec = grade.GradeSpec("film", 1)
    graded = np.rint(grade.apply_rgb(raw/255, spec)*255).astype(np.uint8)
    result = grade.compare_still(raw, graded, {"captions": {"text_color": "#F8E4A0"},
                                               "graphics": {"ink_color": "#123456"}})
    centre = width // 2
    np.testing.assert_array_equal(result[:, :centre-10], raw[:, :centre-10])
    np.testing.assert_array_equal(result[:, centre+10:], graded[:, centre+10:])
    knob = (result[..., 0] > 180) & (result[..., 1] > 150)
    knob[:, centre-1:centre+1] = False
    ys, _ = np.nonzero(knob)
    assert len(ys) > 1
    assert abs(ys.mean() - height * 0.66) <= 1


def test_compare_identity_binds_colours_motion_and_fades(monkeypatch):
    style = {"captions": {"text_color": "#F8E4A0"}, "graphics": {"ink_color": "#123456"}}
    original = grade.describe_compare(style)
    style["captions"]["text_color"] = "#ABCDEF"
    assert grade.describe_compare(style)["handle_sha256"] != original["handle_sha256"]
    current = grade.describe_compare(style)
    monkeypatch.setattr(grade, "HANDLE_KNOB_Y", 0.5)
    assert grade.describe_compare(style)["handle_sha256"] != current["handle_sha256"]
    monkeypatch.setattr(grade, "HANDLE_KNOB_Y", 0.66)
    monkeypatch.setattr(grade, "COMPARE_POINTS", ((0, 0), (1, 0)))
    assert grade.describe_compare(style) != current
    monkeypatch.setattr(grade, "COMPARE_POINTS", current["positions"])
    monkeypatch.setattr(grade, "HANDLE_FADE_FRAMES", 7)
    assert grade.describe_compare(style) != current
