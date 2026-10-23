"""The edit format is available without opening a project."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from yap_studio.cli import main
from yap_studio.project.model import SCHEMAS_DIR, SPECS, json_schema, parse_document


def test_edit_example_validates_without_mutation() -> None:
    from yap_studio.project.edit_format import edit_example

    example = edit_example()
    original = copy.deepcopy(example)
    assert parse_document("edit", copy.deepcopy(example))
    assert example == original


def test_printed_edit_example_validates(capsys: pytest.CaptureFixture) -> None:
    assert main(["edit", "schema"]) == 0
    output = capsys.readouterr().out
    assert "Replace the hash" in output.split("Example edit.json:\n", 1)[0]
    example = json.loads(output.split("Example edit.json:\n", 1)[1])
    assert parse_document("edit", example)


def test_edit_schema_json_matches_committed_schema(capsys: pytest.CaptureFixture) -> None:
    assert main(["edit", "schema", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json_schema("edit")
    assert printed == json.loads((SCHEMAS_DIR / "edit.schema.json").read_text())


def test_edit_schema_text_lists_every_field(capsys: pytest.CaptureFixture) -> None:
    assert main(["edit", "schema"]) == 0
    output = capsys.readouterr().out
    spec = SPECS["edit"]
    for name in spec["properties"]:
        assert f"  {name}:" in output
    segment_fields = spec["properties"]["segments"]["items"]["properties"]
    for name in segment_fields:
        assert f"    {name}:" in output
    for edge in ("from", "to"):
        section = output.split(f"{edge} anchor shapes:\n", 1)[1].split("\n\n", 1)[0]
        for choice in segment_fields[edge]["oneOf"]:
            for name in choice["properties"]:
                assert f"    {name}:" in section
    assert "edit check DIR" in output
    assert "edit rebase DIR" in output


def test_edit_schema_does_not_write_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["edit", "schema"]) == 0
    capsys.readouterr()
    assert list(tmp_path.iterdir()) == []
