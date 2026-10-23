"""Eyedropper colour math and exact cut-frame sampling."""

from pathlib import Path

import numpy as np
import pytest
from test_grade import lab

from yap_studio import ffmpeg
from yap_studio.grade import GradeSpec, apply_rgb, cube_text


@pytest.mark.parametrize("color", ["#1E6BD6", "#2FAE5A", "#C03AA8", "#E0A020"])
def test_eyedropper_skin_and_neutral(color):
    skin = np.array([(255, 224, 196), (234, 192, 134), (224, 172, 140),
                     (198, 134, 66), (141, 85, 36), (92, 56, 38),
                     (255, 205, 180), (200, 150, 130), (120, 80, 70)]) / 255
    spec = GradeSpec("eyedropper", 1, color=color)
    before, after = lab(skin), lab(apply_rgb(skin, spec))
    def hues(values):
        return np.degrees(np.arctan2(values[..., 2], values[..., 1]))
    shifts = np.abs((hues(after) - hues(before) + 180) % 360 - 180)
    assert np.max(shifts) < 6, shifts
    grey = np.full(3, 0.6 ** (1 / 0.8))
    result = lab(apply_rgb(grey, spec))
    target = lab(np.array([int(color[i:i+2], 16) for i in (1, 3, 5)]) / 255)
    assert abs((hues(result) - hues(target) + 180) % 360 - 180) < 15
    assert np.linalg.norm(result[1:]) >= 0.01
    assert abs(result[0] - 0.6) < 1e-6
    np.testing.assert_array_equal(apply_rgb(skin, GradeSpec("eyedropper", 0, color=color)), skin)
    assert cube_text(spec) == cube_text(spec)


@pytest.mark.parametrize("color", [None, "#808080", "#000000", "#ffffff", "red", "#12345G"])
def test_eyedropper_bad_colour_has_fix(color):
    from yap_studio.grade import GradeError

    with pytest.raises(GradeError) as error:
        GradeSpec("eyedropper", color=color)
    assert error.value.suggested_action


@pytest.fixture
def swatch(tmp_path):
    target = tmp_path / "patch.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=blue:s=48x32:r=30,drawbox=x=0:y=0:w=25:h=17:color=red:t=fill:"
                "enable='eq(n,1)'", "-frames:v", "3", "-c:v", "libx264", "-crf", "0",
                "-pix_fmt", "yuv444p", "-colorspace", "bt709", "-color_trc", "bt709",
                "-color_primaries", "bt709", str(target)], timeout_s=30)
    return target


def test_sample_exact_frame_patch_and_clamped_edge(swatch: Path):
    from yap_studio.grade import sample_color

    red = sample_color(swatch, time_s=1/30, x=0.2, y=0.2)
    assert sample_color(swatch, time_s=1/30, x=0, y=0) == red
    assert sample_color(swatch, time_s=1/30, x=0.2, y=0.2) == red
    blue = sample_color(swatch, time_s=0, x=0.2, y=0.2)
    assert blue != red
    assert sample_color(swatch, time_s=2/30, x=0.2, y=0.2) == blue
    assert sample_color(swatch, time_s=1/30, x=1, y=1) == blue
    assert int(red[1:3], 16) > 240 and int(red[5:7], 16) < 10
    data = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(swatch), "-vf",
                       "trim=start_frame=1:end_frame=2", "-frames:v", "1", "-pix_fmt", "rgb24",
                       "-f", "rawvideo", "-"], timeout_s=30, binary=True).stdout
    rgb = np.frombuffer(data, np.uint8).reshape(32, 48, 3)
    mean = np.rint(rgb[10:23, 18:31].mean(axis=(0, 1))).astype(int)
    expected = "#" + "".join(f"{value:02X}" for value in mean)
    assert sample_color(swatch, time_s=1/30, x=0.5, y=0.5) == expected


@pytest.mark.parametrize("values", [
    {"time_s": -1}, {"time_s": 0.1}, {"time_s": 0.099}, {"time_s": float("nan")},
    {"x": -0.01}, {"x": 1.1}, {"y": float("inf")}, {"radius_px": -1},
])
def test_bad_sample_refused(swatch, values):
    from yap_studio.grade import GradeError, sample_color

    with pytest.raises(GradeError) as error:
        sample_color(swatch, **{"time_s": 0, "x": 0.5, "y": 0.5, **values})
    assert error.value.suggested_action


@pytest.mark.parametrize("selection", [None, {"sample": {}}, {"sample": "point"}])
def test_bad_project_grade_shape_has_fix(selection):
    from yap_studio.grade import GradeError, resolve_grade

    with pytest.raises(GradeError) as error:
        resolve_grade({"grade": selection}, None)
    assert error.value.suggested_action
