"""Grade checks adapt to visible colour changes on each picture."""

import json
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from yap_studio import ffmpeg, receipt, video
from yap_studio.checks.grade import check_grade
from yap_studio.edl import load_edit_list
from yap_studio.edl.model import Hold, PictureRegion
from yap_studio.grade import GRADE_VERSION, GradeSpec, apply_rgb, graph_filter, write_cube
from yap_studio.plan import Plan


def grade_edits(root, reference, spec, duration):
    edl = root / "ranges.json"
    edl.write_text(json.dumps({
        "sources": {"main": str(reference)},
        "ranges": [{"source": "main", "start": 0, "end": duration}],
        "grade": {**{key: value for key, value in spec.to_dict().items() if key != "source"},
                  "version": GRADE_VERSION,
                  "reference": str(reference),
                  "reference_sha256": receipt.sha256_file(reference)},
    }))
    return load_edit_list(edl)


@pytest.mark.parametrize(("spec", "color", "bound"), [
    (GradeSpec("film", 0.01), (155, 155, 150), 2.5),
    (GradeSpec("film", 1), (155, 155, 150), None),
    (GradeSpec("film", 1), (0, 255, 0), 10),
    (GradeSpec("eyedropper", 0.5, color="#1E6BD6"), (0, 0, 0), 2.5),
    (GradeSpec("eyedropper", 0.5, color="#1E6BD6"), (155, 155, 150), None),
    (GradeSpec("eyedropper", 0.5, color="#1E6BD6"), (0, 255, 0), 10),
])
def test_grade_limit_follows_visible_magnitude(tmp_path, monkeypatch, spec, color, bound):
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"synthetic reference with mocked RGB decode")
    edits = grade_edits(tmp_path, reference, spec, 0.3)
    edits.picture_regions.append(PictureRegion(Fraction(0), Fraction(1), 45, 0, 45, 160))
    plan = Plan.build(edits, fps=Fraction(30), sample_rate=48000)
    info = SimpleNamespace(video_frames=9, width=90, height=160)
    final = tmp_path / "ungraded.mp4"

    def rgb(path, frames, **kwargs):
        pictures = np.full((len(frames), 160, 90, 3), color, dtype=np.uint8)
        # One outlier frame must not replace the median of the per-frame means.
        pictures[-1] = (0, 255, 0)
        pictures[:, :, 45:] = (255, 0, 0) if path == final else (0, 255, 0)
        return pictures

    monkeypatch.setattr(video, "rgb_frames", rgb)
    findings, coverage = check_grade(final, edits, plan, info)
    evidence = coverage["evidence"]
    magnitude = float(np.mean(np.abs(apply_rgb(np.array(color) / 255, spec) * 255 - color)))
    assert evidence["grade_magnitude"] == pytest.approx(magnitude)
    assert evidence["median_error"] == pytest.approx(magnitude)
    assert evidence["reference_error"] == 0
    assert evidence["limit"] == pytest.approx(min(10, max(2.5, 0.5 * magnitude)))
    if bound is not None:
        assert evidence["limit"] == bound
    assert bool(findings) == (magnitude > evidence["limit"])
    if findings:
        assert "looks ungraded" in findings[0].message
        assert findings[0].suggested_action == "Re-render with yap-studio assemble."


@pytest.fixture(scope="module")
def muted_film(tmp_path_factory):
    root = tmp_path_factory.mktemp("muted-film")
    raw = root / "portrait.rgb"
    yy, xx = np.mgrid[0:320, 0:180] / np.array([320, 180])[:, None, None]
    with raw.open("wb") as stream:
        for frame in range(60):
            x = xx - 0.012 * np.sin(frame / 9)
            rgb = np.full((320, 180, 3), (155, 155, 150), dtype=float)
            rgb += (6 * (xx - 0.5) + np.sin(frame / 15))[..., None]
            # Soft shirt, head and hair shapes against a muted wall.
            for cx, cy, rx, ry, color in (
                (0.5, 0.94, 0.39, 0.36, (135, 144, 149)),
                (0.5, 0.38, 0.19, 0.23, (175, 149, 135)),
                (0.5, 0.22, 0.18, 0.10, (115, 112, 109)),
            ):
                distance = ((x - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
                weight = np.clip((1 - distance) / 0.12, 0, 1)[..., None]
                rgb += weight * (color - rgb)
            stream.write(np.round(rgb).astype(np.uint8).tobytes())
    reference = root / "reference.mp4"
    encoding = ["-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709"]
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "180x320", "-r", "30", "-i", str(raw), "-vf",
                "scale=out_color_matrix=bt709:out_range=tv", *encoding, str(reference)],
               timeout_s=60)
    spec = GradeSpec("film", 1)
    final = root / "film.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(reference), "-vf",
                graph_filter(write_cube(spec, root)), *encoding, str(final)], timeout_s=60)
    edits = grade_edits(root, reference, spec, 2)
    info = ffmpeg.probe(final)
    plan = Plan.build(edits, fps=info.fps, sample_rate=48000)
    return reference, final, edits, info, plan


@pytest.mark.render
@pytest.mark.parametrize("graded", [True, False])
def test_muted_film_detects_missing_grade(muted_film, graded):
    reference, final, edits, info, plan = muted_film
    findings, coverage = check_grade(final if graded else reference, edits, plan, info)
    evidence = coverage["evidence"]
    print("muted Film", "correct" if graded else "ungraded", evidence)
    if graded:
        assert not findings, coverage
        assert evidence["median_error"] < evidence["limit"] / 2
    else:
        assert 5 < evidence["median_error"] < 9
        assert any(finding.check == "picture.grade" for finding in findings), coverage
        assert evidence["looks_ungraded"]
        assert "looks ungraded" in findings[0].message
    assert 5 < evidence["grade_magnitude"] < 9
    assert evidence["limit"] == pytest.approx(0.5 * evidence["grade_magnitude"])


def test_grade_samples_visible_frames_before_spreading(muted_film):
    reference, _, edits, info, plan = muted_film
    covered = set(np.linspace(0, info.video_frames-1, 24, dtype=int).tolist())
    regions = [PictureRegion(Fraction(frame, 30), Fraction(frame+1, 30), x, 0, 90, 320)
               for frame in sorted(covered) for x in (0, 90)]
    edits = replace(edits, picture_regions=regions)
    findings, coverage = check_grade(reference, edits, plan, info)
    assert coverage["ran"], coverage
    assert len(coverage["evidence"]["frames"]) == 24
    assert not covered.intersection(coverage["evidence"]["frames"])
    assert coverage["evidence"]["looks_ungraded"]
    assert any(finding.check == "picture.grade" for finding in findings)


@pytest.mark.parametrize("visible", [6, 8])
def test_grade_samples_later_visible_frames(muted_film, visible):
    reference, _, edits, info, plan = muted_film
    first_visible = info.video_frames - visible
    edits = replace(
        edits,
        picture_masks=[(Fraction(0), Fraction(15, 30))],
        holds=[Hold(1, 15, 30)],
        grade={**edits.grade, "skip_frames": [[30, 45]]},
        # Compare windows use full-frame regions, as do other covered pictures.
        picture_regions=[PictureRegion(Fraction(45, 30), Fraction(first_visible, 30),
                                       0, 0, 180, 320)],
    )
    findings, coverage = check_grade(reference, edits, plan, info)
    assert coverage["ran"], coverage
    assert coverage["evidence"]["frames"] == list(range(first_visible, info.video_frames))
    assert coverage["evidence"]["looks_ungraded"]
    assert len(findings) == 1 and findings[0].severity == "fail"


@pytest.mark.parametrize("visible", [0, 3, 5])
def test_grade_with_too_few_visible_frames_warns(muted_film, monkeypatch, visible):
    reference, _, edits, info, plan = muted_film
    edits = replace(edits, picture_masks=[
        (Fraction(0), Fraction(info.video_frames - visible, 30)),
    ])

    def unexpected_decode(*args, **kwargs):
        pytest.fail("A grade with too few visible frames must not decode samples")

    monkeypatch.setattr(video, "rgb_frames", unexpected_decode)
    findings, coverage = check_grade(reference, edits, plan, info)
    reason = f"{visible} usable frames (needs 6)"
    assert not coverage["ran"]
    assert coverage["reason"] == reason
    assert coverage["evidence"] == {"usable_frames": visible, "minimum_frames": 6}
    assert len(findings) == 1 and findings[0].severity == "warn"
    assert findings[0].check == "picture.grade"
    assert findings[0].message == (
        f"Picture colour grade not measured: {reason}. "
        "Check the final by eye or re-render with yap-studio assemble."
    )


def test_grade_incomplete_decode_warns(muted_film, monkeypatch):
    reference, final, edits, info, plan = muted_film
    monkeypatch.setattr(video, "rgb_frames", lambda *args, **kwargs: [])
    findings, coverage = check_grade(final, edits, plan, info)
    assert not coverage["ran"]
    assert "does not cover the sampled output frames" in coverage["reason"]
    assert coverage["evidence"]["usable_frames"] == info.video_frames
    assert len(findings) == 1 and findings[0].severity == "warn"
    assert "not measured" in findings[0].message


@pytest.mark.parametrize("declared", [False, True])
def test_grade_without_active_declaration_still_skips(muted_film, declared):
    reference, final, edits, info, plan = muted_film
    edits = replace(edits, grade={**edits.grade, "look": "none"} if declared else None)
    findings, coverage = check_grade(final, edits, plan, info)
    assert not findings and not coverage["ran"]
