"""Project grade commands retain the creator's style and validate writes."""

import json
from pathlib import Path

import pytest

from yap_studio.cli import main


def test_show_set_clear(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "project.json"
    project.write_text(json.dumps({"schema": "yap-studio/project@1", "name": "grade",
                                   "style": "preset:clean_minimal",
                                   "takes": [{"id": "t1", "path": "take.mov"}]}))
    original = project.read_bytes()
    assert main(["grade", "show", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "style"
    assert project.read_bytes() == original
    assert main(["grade", "set", str(tmp_path), "--look", "film", "--strength", "0.7",
                 "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "look": "film", "strength": 0.7, "source": "project"}
    assert json.loads(project.read_bytes())["grade"] == {"look": "film", "strength": 0.7}
    before = project.read_bytes()
    assert main(["grade", "set", str(tmp_path), "--look", "film", "--strength", "nan",
                 "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["suggested_action"]
    assert project.read_bytes() == before
    assert main(["grade", "set", str(tmp_path), "--look", "none"]) == 0
    assert "project" in capsys.readouterr().out
    assert main(["grade", "set", str(tmp_path), "--clear", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "style"
    assert "grade" not in json.loads(project.read_bytes())


def test_grade_style_override_and_missing_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["grade", "show", str(tmp_path), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["suggested_action"]
    (tmp_path / "style.json").write_text(json.dumps({
        "schema_version": 1, "grade": {"look": "film", "strength": 0.6}}))
    (tmp_path / "project.json").write_text(json.dumps({
        "schema": "yap-studio/project@1", "name": "grade", "style": "style.json",
        "takes": [{"id": "t1", "path": "take.mov"}]}))
    for arguments, expected in ((["show"], ("film", "style")),
                                 (["set", "--look", "none"], ("none", "project")),
                                 (["set", "--clear"], ("film", "style"))):
        assert main(["grade", arguments[0], str(tmp_path), *arguments[1:], "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert (result["look"], result["source"]) == expected
