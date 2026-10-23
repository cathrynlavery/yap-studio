"""Eyedropper project commands sample without mutating and save portable points."""

import json
import shutil

import pytest
from test_grade_eyedropper import swatch  # noqa: F401

from yap_studio.cli import main


@pytest.fixture
def sampled_project(tmp_path, request):
    (tmp_path / "project.json").write_text(json.dumps({
        "schema": "yap-studio/project@1", "name": "sample", "style": "preset:clean_minimal",
        "takes": [{"id": "t1", "path": "take.mov"}]}))
    (tmp_path / "cut").mkdir()
    shutil.copyfile(request.getfixturevalue("swatch"), tmp_path / "cut/cut.mp4")
    return tmp_path


def test_sample_set_and_show(sampled_project, capsys):
    root = sampled_project
    target = root / "project.json"
    original = target.read_bytes()
    point = ["--time", str(1/30), "--x", "0.2", "--y", "0.2"]
    assert main(["grade", "sample", str(root), *point, "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["colorful"] and info["chroma"] >= 0.02
    assert target.read_bytes() == original
    assert main(["grade", "set", str(root), "--look", "eyedropper", *point, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["color"] == info["color"] and result["strength"] == 0.5
    saved = json.loads(target.read_bytes())["grade"]
    assert saved == {"look": "eyedropper", "strength": 0.5,
                     "sample": {"time_s": 1/30, "x": 0.2, "y": 0.2}}
    assert result["sample"] == saved["sample"]
    assert main(["grade", "show", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == result
    assert main(["grade", "show", str(root)]) == 0
    shown = capsys.readouterr().out
    assert info["color"] in shown and "Sample:" in shown and "x 0.2" in shown
    assert main(["grade", "set", str(root), "--look", "eyedropper", "--color", "#1e6bd6",
                 "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["color"] == "#1E6BD6"
    assert "sample" not in json.loads(target.read_bytes())["grade"]


@pytest.mark.parametrize("extra", [
    [], ["--color", "#808080"], ["--time", "0"],
    ["--color", "#1E6BD6", "--time", "0", "--x", "0.5", "--y", "0.5"],
    ["--time", "10", "--x", "0.5", "--y", "0.5"],
])
def test_bad_eye_selection_does_not_write(sampled_project, capsys, extra):
    original = (sampled_project / "project.json").read_bytes()
    assert main(["grade", "set", str(sampled_project), "--look", "eyedropper",
                 *extra, "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["suggested_action"]
    assert (sampled_project / "project.json").read_bytes() == original


def test_sample_requires_cut(sampled_project, capsys):
    (sampled_project / "cut/cut.mp4").unlink()
    assert main(["grade", "sample", str(sampled_project), "--time", "0", "--x", "0.5",
                 "--y", "0.5", "--json"]) == 2
    assert "run yap-studio cut DIR first" in capsys.readouterr().err.lower().replace("dir", "DIR")


def test_grey_sample_reports_chroma_without_writing(sampled_project, capsys):
    from yap_studio import ffmpeg

    original = (sampled_project / "project.json").read_bytes()
    ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=gray:s=48x32:r=30",
                "-frames:v", "3", "-c:v", "libx264", "-colorspace", "bt709",
                "-color_trc", "bt709", "-color_primaries", "bt709",
                str(sampled_project / "cut/cut.mp4")], timeout_s=30)
    assert main(["grade", "sample", str(sampled_project), "--time", "0", "--x", "0.5",
                 "--y", "0.5", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["colorful"] and result["chroma"] < 0.02
    assert (sampled_project / "project.json").read_bytes() == original
