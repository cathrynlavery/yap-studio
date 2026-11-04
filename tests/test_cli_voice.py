"""Voice commands expose structured results and errors."""

import json

import pytest

from yap_studio.cli import main


@pytest.mark.parametrize("as_json", [False, True])
def test_samples_reports_files_left_behind(tmp_path, monkeypatch, capsys, as_json):
    from yap_studio import voice

    notice = "Left voice/samples/s05.wav because it changed."
    result = {"samples": [], "notices": [notice]}
    monkeypatch.setattr(voice, "samples", lambda _: result)
    assert main(["voice", "samples", str(tmp_path), *(["--json"] if as_json else [])]) == 0
    captured = capsys.readouterr()
    if as_json:
        assert json.loads(captured.out) == result
        assert captured.err == ""
    else:
        assert captured.out == "Voice samples ready.\n"
        assert captured.err == notice + "\n"


def test_list_propose_json(voice_project, capsys):
    assert main(["voice", "list", str(voice_project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["fixes"] == []
    assert (
        main(
            [
                "voice",
                "propose",
                str(voice_project),
                "--from",
                "t1:2",
                "--to",
                "t1:3",
                "--say",
                "cold plunge",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["id"] == "f1"


@pytest.mark.parametrize(
    "command,args",
    [
        ("samples", []),
        ("consent", []),
        ("create", []),
        ("link", ["--name", "My voice"]),
        ("make", ["--fix", "f1"]),
        ("approve", ["--fix", "f1"]),
        ("reject", ["--fix", "f1"]),
        ("delete-voice", []),
        ("propose", ["--from", "bad", "--to", "bad", "--say", "water"]),
    ],
)
def test_error_shape(voice_project, capsys, command, args):
    assert main(["voice", command, str(voice_project), *args, "--json"]) == 2
    result = json.loads(capsys.readouterr().err)
    assert {"type", "code", "message"} <= result["error"].keys()


def test_json_workflow(voice_project, fake_voice, monkeypatch, capsys):
    from voice_helpers import linked

    linked(voice_project, monkeypatch)
    commands = [
        ["consent", "--yes"],
        ["create", "--dry-run"],
        ["link", "--name", "My voice", "--yes"],
        ["propose", "--from", "t1:2", "--to", "t1:3", "--say", "cold plunge"],
        ["make", "--fix", "f1", "--yes", "--max-credits", "20"],
        ["approve", "--fix", "f1"],
        ["reject", "--fix", "f1"],
        ["consent", "--withdraw"],
        ["delete-voice", "--yes"],
    ]
    for command, *options in commands:
        assert main(["voice", command, str(voice_project), *options, "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict) and not captured.err
        assert "fake-personal-voice-secret" not in captured.out
    for target in voice_project.rglob("*"):
        if target.is_file():
            assert b"fake-personal-voice-secret" not in target.read_bytes()
