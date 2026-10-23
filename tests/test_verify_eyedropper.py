"""Verify sampled-colour grades against synthetic picture references."""

import json

import pytest

from yap_studio import ffmpeg, receipt
from yap_studio.grade import GRADE_VERSION, GradeSpec, graph_filter, write_cube
from yap_studio.verify import verify

pytestmark = pytest.mark.render


@pytest.fixture(scope="module", params=[0.5, 1])
def eyedropper_final(tmp_path_factory, synth_media, request):
    root = tmp_path_factory.mktemp("eyedropper-verify")
    reference = synth_media.path("clean")
    outputs = {"ungraded": reference}
    strength = request.param
    for name, spec in (("correct", GradeSpec("eyedropper", strength, color="#1E6BD6")),
                       ("wrong", GradeSpec("eyedropper", strength, color="#E0A020")),
                       ("film", GradeSpec("film", 1))):
        cube = write_cube(spec, root / name)
        out = root / f"{name}.mp4"
        ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(reference), "-vf", graph_filter(cube),
                    "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
                    "-c:a", "copy", str(out)], timeout_s=120)
        outputs[name] = out
    edl = root / "ranges.json"
    edl.write_text(json.dumps({
        "sources": {"main": str(synth_media.path("calib"))},
        "ranges": [{"source": "main", "start": start, "end": end}
                   for start, end in synth_media.spans],
        "grade": {"look": "eyedropper", "strength": strength, "color": "#1E6BD6",
                  "version": GRADE_VERSION, "reference": str(reference),
                  "reference_sha256": receipt.sha256_file(reference)},
    }))
    return {"edl": edl, **outputs}


@pytest.mark.parametrize("kind", ["correct", "wrong", "ungraded", "film"])
def test_eyedropper_grade_verification(eyedropper_final, kind):
    report = verify(str(eyedropper_final[kind]), edl=str(eyedropper_final["edl"]),
                    asr="none", loudness="note", detail="full")
    check = report["checks"]["picture.grade"]
    assert check["ran"]
    print(kind, check["evidence"])
    if kind == "correct":
        assert report["status"] == "pass", report["findings"]
        assert check["evidence"]["median_error"] < check["evidence"]["limit"]
        assert all(row["confidence"] == "high" for row in report["segments"])
    else:
        assert check["status"] == "fail", check
