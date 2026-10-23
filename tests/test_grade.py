"""Film colour math and portable deterministic cubes."""

from pathlib import Path

import numpy as np
import pytest

from yap_studio.errors import YapError
from yap_studio.grade import GradeSpec, apply_rgb, cube_text, resolve_grade, write_cube


def lab(rgb: np.ndarray) -> np.ndarray:
    """Published Oklab transform, independently evaluated for colour assertions."""
    lms = rgb ** 2.4 @ np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]).T
    return np.cbrt(lms) @ np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]).T


def test_identity_cube_and_determinism(tmp_path: Path) -> None:
    spec = GradeSpec("film", 0, "style")
    grid = np.array([[r, g, b] for b in np.linspace(0, 1, 33)
                     for g in np.linspace(0, 1, 33) for r in np.linspace(0, 1, 33)])
    np.testing.assert_array_equal(apply_rgb(grid, spec), grid)
    text = cube_text(spec)
    rows = [line for line in text.splitlines() if line and line[0].isdigit()]
    assert len(rows) == 33 ** 3
    np.testing.assert_allclose(np.loadtxt(rows), grid, atol=1e-6)
    assert text == cube_text(spec) == write_cube(spec, tmp_path).read_text()
    assert "LUT_3D_SIZE 33" in text


def test_film_skin_hue_and_chroma() -> None:
    skin = np.array([(255, 224, 196), (234, 192, 134), (224, 172, 140),
                     (198, 134, 66), (141, 85, 36), (92, 56, 38),
                     (255, 205, 180), (200, 150, 130), (120, 80, 70)]) / 255
    before = lab(skin)
    after = lab(apply_rgb(skin, GradeSpec("film", 1, "style")))
    def hue(x: np.ndarray) -> np.ndarray:
        return np.degrees(np.arctan2(x[:, 2], x[:, 1]))
    clipped = skin.max(axis=1) * 255 >= 250
    # Gamut clipping in bright skin can change hue and chroma more at the top.
    shift = np.abs(hue(after) - hue(before))
    assert np.all(shift[~clipped] < 3) and np.all(shift[clipped] < 5), shift
    ratio = np.linalg.norm(after[:, 1:], axis=1) / np.linalg.norm(before[:, 1:], axis=1)
    assert np.all(ratio[~clipped] >= 0.92) and np.all(ratio[clipped] >= 0.88), ratio
    assert np.all(ratio <= 1.06), ratio


def test_neutrals_monotone_and_bounded() -> None:
    greys = np.repeat(np.linspace(0, 1, 1001)[:, None], 3, axis=1)
    result = apply_rgb(greys, GradeSpec("film", 1, "style"))
    measured = lab(result)
    assert np.all(np.diff(measured[:, 0]) >= 0)
    # The stronger teal shadows intentionally reach about 0.02954 Oklab chroma.
    assert np.max(np.linalg.norm(measured[:, 1:], axis=1)) < 0.030
    rng = np.random.default_rng(240)
    output = apply_rgb(rng.random((5, 20, 3)), GradeSpec("film", 1, "project"))
    assert np.isfinite(output).all() and output.min() >= 0 and output.max() <= 1


@pytest.mark.parametrize("bad", [
    {"look": "unknown"}, {"strength": -0.1}, {"strength": 1.1},
    {"strength": True}, {"strength": float("nan")}, {"strength": "1"},
    {"source": "unknown"}, {"version": 3}, {"version": True}, {"extra": 1},
])
def test_bad_specs_refused(bad: dict) -> None:
    with pytest.raises(YapError) as error:
        GradeSpec.from_dict(bad)
    assert error.value.suggested_action


def test_resolution_and_round_trip() -> None:
    assert resolve_grade({}, None) == GradeSpec("none", 1, "style")
    spec = resolve_grade({"grade": {"look": "film"}}, {"look": "none"})
    assert not spec.active and spec.source == "project"
    assert GradeSpec.from_dict(spec.to_dict()) == spec


def test_cube_escaped_option_path(tmp_path: Path) -> None:
    from yap_studio import ffmpeg
    from yap_studio.grade import graph_filter, rgba_filter

    folder = tmp_path / "a b'c:d,e;f[g]\\h"
    cube = write_cube(GradeSpec("film", 1), folder)
    for source, filters in (("color=gray:s=16x16", graph_filter(cube)),
                             ("color=gray:s=16x16,format=rgba", rgba_filter(cube))):
        ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source,
                    "-vf", filters, "-frames:v", "1", "-f", "null", "-"], timeout_s=30)
    result = ffmpeg.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=gray@0.3:s=16x16,format=rgba", "-vf", rgba_filter(cube) + ",alphaextract",
        "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ], timeout_s=30, binary=True)
    assert set(result.stdout) == {76}


@pytest.mark.parametrize("spec", [GradeSpec("film"),
                                  GradeSpec("eyedropper", color="#1E6BD6")])
def test_alpha_graph_matches_picture_grade_and_preserves_alpha(tmp_path: Path, spec) -> None:
    from yap_studio import ffmpeg, grade

    planes = np.empty((4, 16, 32), dtype=np.uint8)
    planes[:3] = np.array([71, 128, 128])[:, None, None]
    planes[3] = np.linspace(0, 255, 32).astype(np.uint8)[None, :]
    source = tmp_path / "layer.yuva"
    source.write_bytes(planes.tobytes())
    cube = write_cube(spec, tmp_path / "grade's cube")
    results = []
    for filters, pixel_format, channels in (
        (grade.graph_alpha_filter(cube), "yuva444p", 4),
        (grade.graph_filter(cube), "yuv444p", 3),
    ):
        raw = ffmpeg.run([
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuva444p",
            "-s", "32x16", "-framerate", "30", "-i", str(source), "-vf", filters,
            "-frames:v", "1", "-pix_fmt", pixel_format, "-f", "rawvideo", "-",
        ], timeout_s=30, binary=True).stdout
        results.append(np.frombuffer(raw, dtype=np.uint8).reshape(channels, 16, 32))
    alpha, picture = results
    assert np.array_equal(alpha[3], planes[3])
    np.testing.assert_allclose(alpha[:3].astype(float), picture, atol=1)
    assert np.abs(alpha[:3].astype(float) - planes[:3]).max() > 3
