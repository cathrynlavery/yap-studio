"""Ask once per video before any picture pricing or generation."""

import json

import pytest
from pictures_helpers import PROMPT, record_answer
from test_pictures_sheet import STYLE, requests
from test_stock_provider import calls
from test_stock_sheet import words
from test_timeline import write

from yap_studio import cli
from yap_studio.pictures import guard


@pytest.fixture
def video(tmp_path):
    write(tmp_path / "project.json", {
        "schema": "yap-studio/project@1", "name": "video", "style": "style.json",
        "takes": [{"id": "t1", "path": "own.mov"}], "asr": {"engine": "none"},
        "broll_sources": ["folder"],
    })
    write(tmp_path / "style.json", {"schema_version": 1, **STYLE})
    write(tmp_path / "cut/cut-words.json", words().to_dict())
    write(tmp_path / "plans/pictures.json", requests())
    return tmp_path


def ask_args(root):
    return ["pictures", "ask", str(root / "plans/pictures.json"),
            "--cut-words", str(root / "cut/cut-words.json"), "--json"]


def test_question_is_unselected_and_offline(video, fake_wavespeed, capsys):
    assert cli.main(["pictures", "question", str(video), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == "yap-studio/imagery-question@1"
    assert data["project"] == str(video.resolve()) and data["answer"] is None
    assert data["question"] == {
        "id": "imagery", "header": "Imagery", "prompt": PROMPT,
        "note": "Each picture is paid on your WaveSpeed account. "
                "You see each price before anything is made.",
        "kind": "choice", "choices": [
            {"value": "yes", "label": "Yes", "description":
             "I draft a picture for lines your media does not cover and show each price first.",
             "recommended": False},
            {"value": "no", "label": "No", "description":
             "No generated pictures in this video.", "recommended": False},
        ],
    }
    assert cli.main(["pictures", "question", str(video)]) == 0
    assert capsys.readouterr().out.strip() == "not asked yet: " + PROMPT
    assert calls(fake_wavespeed) == [] and not (video / "pictures").exists()


def test_answer_is_saved_and_can_change(video, fake_wavespeed, capsys):
    for answer in ("yes", "no", "yes"):
        assert cli.main(["pictures", "answer", str(video), answer]) == 0
        assert capsys.readouterr().out.strip() == f"Recorded: {answer} for this video."
        assert cli.main(["pictures", "question", str(video), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["answer"] == answer
        record = json.loads((video / "pictures/imagery.json").read_text())
        assert record["schema"] == "yap-studio/imagery-answer@1"
        assert record["question"] == PROMPT and record["answer"] == answer
        assert record["answered_at"].endswith("Z")
    assert cli.main(["pictures", "answer", str(video), "yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        (video / "pictures/imagery.json").read_text())
    assert cli.main(["pictures", "question", str(video)]) == 0
    assert capsys.readouterr().out.strip() == "this video: yes"
    assert calls(fake_wavespeed) == []


@pytest.mark.parametrize("answer,code,message,action", [
    ("off", "pictures_off", "Generated pictures are off in this style.",
     "Turn on generated pictures in the style first."),
    (None, "pictures_not_asked",
     "The creator has not said yes to generated pictures for this video.",
     'Ask "Do you want me to create imagery for this video?" and record it with '
     'yap-studio pictures answer DIR yes or no.'),
    ("no", "pictures_declined", "The creator said no to generated pictures for this video.",
     "Carry on with their own media. Ask again only if they bring it up."),
])
def test_refusal_precedes_all_cli_calls(video, fake_wavespeed, capsys, answer, code,
                                       message, action):
    if answer == "off":
        write(video / "style.json", {"schema_version": 1})
    elif answer:
        record_answer(video, answer)
    for argv in (ask_args(video), ["pictures", "make", str(video), "--pick", "1",
                                  "--yes", "--max-cost", "1", "--json"]):
        assert cli.main(argv) == 2
        error = json.loads(capsys.readouterr().err)["error"]
        assert error["code"] == code and error["message"] == message
        assert error["type"] == "validation_error"
        assert error["suggested_action"] == action
        assert calls(fake_wavespeed) == []
        assert not (video / "pictures/sheets").exists()
        assert not (video / "pictures/pictures.json").exists()


def test_yes_allows_prices(video, fake_wavespeed, capsys):
    record_answer(video)
    assert cli.main(ask_args(video)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert (video / "pictures/sheets/s1/ask-sheet.json").is_file()
    assert any(argv[0] == "price" for argv in calls(fake_wavespeed))
    assert not any(argv[0] == "run" for argv in calls(fake_wavespeed))


@pytest.mark.parametrize("payload", ["broken JSON", "[]", "{}",
    '{"schema":"yap-studio/imagery-answer@1","question":42,"answer":"yes",'
    '"answered_at":"2026-10-19T12:00:00Z"}',
    '{"schema":"yap-studio/imagery-answer@1","question":"","answer":"yes",'
    '"answered_at":"2026-10-19T12:00:00Z"}',
    '{"schema":"yap-studio/imagery-answer@1","question":"' + PROMPT + '",'
    '"answer":"yes","answered_at":"yesterday"}',
])
def test_broken_answer_names_file(video, fake_wavespeed, capsys, payload):
    target = record_answer(video)
    target.write_text(payload)
    for argv in (["pictures", "question", str(video), "--json"], ask_args(video)):
        assert cli.main(argv) == 2
        assert "pictures/imagery.json" in capsys.readouterr().err
    assert calls(fake_wavespeed) == []


@pytest.mark.parametrize("lane", ["agent", "broll"])
def test_no_refuses_generated_timeline_and_keeps_answer(video, fake_wavespeed, capsys, lane):
    target = record_answer(video)
    before = target.read_bytes()
    write(video / f"timeline/{lane}.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [{
            "id": f"{lane}-picture", "kind": "insert", "lane": lane,
            "start": {"time": 1}, "end": {"time": 3},
            "media": {"path": "pictures/files/synthetic.png", "sha256": "a" * 64},
        }],
    })
    assert cli.main(["pictures", "answer", str(video), "no", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "generated_in_timeline"
    assert f"{lane}-picture" in error["message"]
    assert "yap-studio timeline remove DIR --lane LANE --id ID" in error["suggested_action"]
    assert target.read_bytes() == before and calls(fake_wavespeed) == []


def test_answer_requires_project(tmp_path, fake_wavespeed, capsys):
    assert cli.main(["pictures", "answer", str(tmp_path), "yes", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "Bad value for project:" in error["message"]
    assert not (tmp_path / "pictures").exists() and calls(fake_wavespeed) == []


@pytest.mark.parametrize("json_output", [False, True])
def test_question_requires_project(tmp_path, fake_wavespeed, capsys, json_output):
    root = tmp_path / "mistyped"
    extra = ["--json"] if json_output else []
    assert cli.main(["pictures", "question", str(root), *extra]) == 2
    output = capsys.readouterr()
    assert "Bad value for project:" in output.err and not output.out
    assert not root.exists() and calls(fake_wavespeed) == []


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_older_question_wording_is_readable(video, fake_wavespeed, capsys, answer):
    target = record_answer(video, answer)
    data = json.loads(target.read_text())
    data["question"] = "Would you like imagery for this video?"
    write(target, data)
    before = target.read_bytes()
    assert guard.imagery_answer(video) == answer
    assert cli.main(["pictures", "question", str(video), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["answer"] == answer
    assert target.read_bytes() == before and calls(fake_wavespeed) == []
