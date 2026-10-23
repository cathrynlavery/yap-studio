"""Video choices, saved defaults, and project commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jsonschema
import pytest
from test_project import source as source

from yap_studio import cli, video_type
from yap_studio.errors import InputError, OutputPathError, ValidationError
from yap_studio.project import ProjectRefusal, init_project
from yap_studio.project.model import Project, ProjectFileError, Take, parse_document

EXPECTED_QUESTIONS = [
    {"id": "video_type", "header": "Video type", "prompt": "What are you making?",
     "note": "Each video can be a different kind. Your answer is remembered as the default "
             "for the next video.",
     "kind": "choice", "choices": [
        {"value": "talking_head", "label": "Talking head",
         "description": "You on camera. B-roll, cards and moments where they help.",
         "recommended": True},
        {"value": "voiceover", "label": "Voiceover",
         "description": "Your voice over your photos and clips, in the order you shot them.",
         "recommended": False},
        {"value": "green_screen", "label": "Green screen",
         "description": "You cut out over screenshots, photos or clips.", "recommended": False},
        {"value": "screen_recording", "label": "Screen recording",
         "description": "Your screen is the picture, with your voice over it.",
         "recommended": False},
     ]},
    {"id": "broll_sources", "header": "B-roll", "prompt": "Where can the agent get B-roll?",
     "note": "Pick everything it may use. It uses nothing else.",
     "kind": "multi", "only_for": ["talking_head", "voiceover", "green_screen"], "choices": [
        {"value": "folder", "label": "Inbox or a folder",
         "description": "Clips and photos you drop in or point to.", "recommended": True},
        {"value": "album", "label": "A Photos album",
         "description": "You name the album. Shots follow the day you took them.",
         "recommended": True},
        {"value": "stock", "label": "Stock clips",
         "description": "Free clips found online. You pick each one before it is used.",
         "recommended": False},
        {"value": "none", "label": "None", "description": "Only what is in your take.",
         "recommended": False, "only_for": ["talking_head", "green_screen"]},
     ]},
]


def project_file(root: Path, *, memo: bool = False, **fields) -> Path:
    target = root / "project.json"
    target.write_text(json.dumps({**Project("story", (Take("t1", "take.mov", voiceover=memo),))
                                 .to_dict(), **fields}))
    return target


def test_project_video_fields(tmp_path):
    data = Project("story", (Take("t1", "take.mov"),)).to_dict()
    assert "video_type" not in data and "broll_sources" not in data
    assert parse_document("project", data).broll_sources is None
    data.update(video_type="green_screen", broll_sources=["folder", "stock"])
    project = parse_document("project", data)
    assert project.video_type == "green_screen" and project.broll_sources == ("folder", "stock")
    assert project.to_dict() == data
    schema = Path(__file__).parents[1] / "src/yap_studio/project/schemas/project.schema.json"
    jsonschema.validate(data, json.loads(schema.read_text()))
    for field, value in [("video_type", "other"), ("broll_sources", ["other"]),
                         ("broll_sources", ["folder", "folder"]), ("video_type", None),
                         ("broll_sources", None)]:
        with pytest.raises(ProjectFileError):
            parse_document("project", {**data, field: value})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**data, field: value}, json.loads(schema.read_text()))


def test_defaults_round_trip():
    assert video_type.load_default() is None
    video_type.save_default("talking_head", ["stock", "folder", "folder"])
    assert video_type.load_default() == {"schema": video_type.DEFAULTS_SCHEMA,
                                         "video_type": "talking_head",
                                         "broll_sources": ["folder", "stock"]}


@pytest.mark.parametrize("content", ["{", "[]", "{}", '{"schema":"other"}',
    '{"schema":"yap-studio/video-defaults@1","video_type":"voiceover","broll_sources":[]}',
    '{"schema":"yap-studio/video-defaults@1","video_type":"other","broll_sources":[]}',
    '{"schema":"yap-studio/video-defaults@1","video_type":"talking_head",'
    '"broll_sources":["folder","folder"]}',
])
def test_invalid_defaults_name_file(content):
    target = video_type.defaults_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    with pytest.raises(InputError) as error:
        video_type.load_default()
    assert str(target) in str(error.value)
    assert "video-type set" in error.value.suggested_action


def test_default_write_error(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr("yap_studio.paths.atomic_write_text", fail)
    with pytest.raises(OutputPathError) as error:
        video_type.save_default("talking_head", [])
    assert "YAP_STUDIO_DATA_DIR" in error.value.suggested_action


def test_command_default_text_and_write_exit_code(monkeypatch, capsys):
    assert cli.main(["video-type", "show"]) == 0
    assert 'none saved; the agent asks "What are you making?"' in capsys.readouterr().out
    assert cli.main(["video-type", "set", "--type", "talking_head", "--broll", "none"]) == 0
    output = capsys.readouterr().out
    assert "default: talking head; no B-roll" in output and "Default saved." in output
    assert cli.main(["video-type", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["default"] == {
        "video_type": "talking_head", "broll_sources": []}

    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr("yap_studio.paths.atomic_write_text", fail)
    assert cli.main(["video-type", "set", "--type", "screen_recording", "--json"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert "YAP_STUDIO_DATA_DIR" in error["suggested_action"]


def test_sources():
    assert video_type.parse_sources(["none"]) == []
    assert video_type.parse_sources(["stock", "album", "folder", "stock"]) == [
        "folder", "album", "stock"]
    for bad in (["none", "folder"], ["other"]):
        with pytest.raises(ValidationError):
            video_type.parse_sources(bad)


@pytest.mark.parametrize(("kind", "sources", "expected"), [
    (None, None, "not chosen yet (yap-studio video-type show)"),
    ("talking_head", None, "talking head; B-roll not chosen yet"),
    ("talking_head", [], "talking head; no B-roll"),
    ("screen_recording", [], "screen recording; no B-roll"),
    ("talking_head", ["folder"], "talking head; B-roll from your inbox or a folder"),
    ("talking_head", ["folder", "album"],
     "talking head; B-roll from your inbox or a folder and a Photos album"),
    ("talking_head", ["folder", "album", "stock"],
     "talking head; B-roll from your inbox or a folder, a Photos album and stock clips"),
])
def test_describe(kind, sources, expected):
    assert video_type.describe(kind, sources) == expected


@pytest.mark.parametrize(("memo", "saved_type", "sources", "expected"), [
    (False, None, None, (None, None)),
    (False, "green_screen", ["folder"], ("green_screen", ["folder"])),
    (False, "voiceover", ["album"], ("talking_head", ["album"])),
    (True, "talking_head", ["folder"], ("voiceover", ["folder"])),
    (True, "talking_head", [], ("voiceover", None)),
    (True, "screen_recording", [], ("screen_recording", [])),
])
def test_init_defaults(source, tmp_path, memo, saved_type, sources, expected):
    if saved_type:
        video_type.save_default(saved_type, sources)
    media = source
    if memo:
        media = tmp_path / "memo.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vn", str(media)], check=True)
    project = init_project(tmp_path / "video", takes=[] if memo else [media],
                           voiceovers=[media] if memo else [])
    assert (project.video_type, list(project.broll_sources) if project.broll_sources is not None
            else None) == expected
    if not saved_type:
        assert video_type.load_default() is None


@pytest.mark.parametrize("sources", [[], ["folder"]])
def test_init_explicit_choices_ignore_corrupt_default(source, tmp_path, sources):
    target = video_type.defaults_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{")
    project = init_project(tmp_path / "video", takes=[source], video_type="talking_head",
                           broll_sources=sources)
    assert project.video_type == "talking_head" and project.broll_sources == tuple(sources)
    assert target.read_text() == "{"


def test_init_explicit_and_cli(source, tmp_path, capsys):
    root = tmp_path / "video"
    assert cli.main(["init", str(root), "--take", str(source), "--video-type", "green_screen",
                     "--broll", "stock", "--broll", "folder", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["video_type"] == "green_screen"
    assert result["broll_sources"] == ["folder", "stock"]
    assert video_type.load_default() is None


@pytest.mark.parametrize(("memo", "kind", "sources", "message"), [
    (False, "voiceover", ["folder"], "starts from an audio memo"),
    (True, "talking_head", None, "needs a video take"),
    (True, "green_screen", None, "needs a video take"),
    (True, "voiceover", ["none"], "needs pictures to show"),
])
def test_init_refuses_mismatch(source, tmp_path, memo, kind, sources, message):
    media = source
    if memo:
        media = tmp_path / "memo.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vn", str(media)], check=True)
    root = tmp_path / "video"
    with pytest.raises((InputError, ValidationError), match=message) as error:
        init_project(root, takes=[] if memo else [media], voiceovers=[media] if memo else [],
                     video_type=kind, broll_sources=sources)
    assert error.value.suggested_action and not (root / "project.json").exists()


def test_pickups_preserve_choices_and_refuse_flags(source, tmp_path):
    memo = tmp_path / "memo.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vn", str(memo)], check=True)
    root = tmp_path / "video"
    init_project(root, takes=[], voiceovers=[memo], video_type="voiceover", broll_sources=["album"])
    before = (root / "project.json").read_bytes()
    for flags in ({"video_type": "voiceover"}, {"broll_sources": []}):
        with pytest.raises(ProjectRefusal):
            init_project(root, takes=[], voiceovers=[memo], **flags)
        assert (root / "project.json").read_bytes() == before
    updated = init_project(root, takes=[], voiceovers=[memo])
    assert updated.video_type == "voiceover" and updated.broll_sources == ("album",)


def test_show_questions_and_project(tmp_path, capsys):
    assert cli.main(["video-type", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema": "yap-studio/video-type@1", "default": None, "project": None,
        "questions": EXPECTED_QUESTIONS}
    assert video_type.QUESTIONS == EXPECTED_QUESTIONS
    project_file(tmp_path, video_type="talking_head", broll_sources=[])
    assert cli.main(["video-type", "show", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["project"] == {
        "video_type": "talking_head", "broll_sources": []}


def test_set_first_answer_override_and_default(tmp_path, capsys):
    project_file(tmp_path)
    base = ["video-type", "set", str(tmp_path)]
    assert cli.main([*base, "--type", "talking_head", "--broll", "folder", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["default_saved"] and result["project"] == result["default"]
    assert cli.main([*base, "--type", "green_screen", "--broll", "stock", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["default_saved"] and result["default"]["video_type"] == "talking_head"
    assert result["project"] == {"video_type": "green_screen", "broll_sources": ["stock"]}
    assert cli.main([*base, "--default", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["default_saved"] and result["project"] == result["default"]
    before = (tmp_path / "project.json").read_bytes()
    assert cli.main([*base, "--type", "voiceover", "--json"]) == 2
    assert (tmp_path / "project.json").read_bytes() == before
    assert "audio memo" in json.loads(capsys.readouterr().err)["error"]["message"]


def test_set_default_requirements_and_screen(tmp_path, capsys):
    for flags in ([], ["--type", "voiceover"], ["--type", "voiceover", "--broll", "none"]):
        assert cli.main(["video-type", "set", *flags, "--json"]) == 2
        capsys.readouterr()
    assert cli.main(["video-type", "set", "--type", "screen_recording", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"project": None, "default": {"video_type": "screen_recording",
                                                  "broll_sources": []}, "default_saved": True}
    project_file(tmp_path)
    assert cli.main(["video-type", "set", str(tmp_path), "--type", "talking_head",
                     "--default", "--json"]) == 0
    capsys.readouterr()


def test_set_unrecorded_sources_and_inheritance(tmp_path, capsys):
    project_file(tmp_path)
    assert cli.main(["video-type", "set", str(tmp_path), "--type", "talking_head", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["project"]["broll_sources"] is None and not result["default_saved"]
    before = (tmp_path / "project.json").read_bytes()
    assert cli.main(["video-type", "set", str(tmp_path), "--default", "--json"]) == 2
    capsys.readouterr()
    assert (tmp_path / "project.json").read_bytes() == before
    video_type.save_default("green_screen", ["album"])
    assert cli.main(["video-type", "set", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["project"]["broll_sources"] == ["album"]


@pytest.mark.parametrize("existing_default", [False, True])
def test_set_default_write_failure_preserves_project(
    tmp_path, monkeypatch, capsys, existing_default,
):
    target = project_file(tmp_path, video_type="talking_head", broll_sources=["folder"])
    if existing_default:
        video_type.save_default("talking_head", ["folder"])
    before = target.read_bytes()
    default = video_type.defaults_path()
    old_default = default.read_bytes() if default.exists() else None

    def fail(*args, **kwargs):
        raise OSError("synthetic default write failure")

    monkeypatch.setattr("yap_studio.paths.atomic_write_text", fail)
    assert cli.main(["video-type", "set", str(tmp_path), "--type", "green_screen",
                     "--broll", "album", *(["--default"] if existing_default else []),
                     "--json"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert "synthetic default write failure" in error["message"]
    assert target.read_bytes() == before
    assert (default.read_bytes() if default.exists() else None) == old_default


@pytest.mark.parametrize("existing_default", [False, True])
def test_set_project_write_failure_restores_default(
    tmp_path, monkeypatch, capsys, existing_default,
):
    target = project_file(tmp_path, video_type="talking_head", broll_sources=["folder"])
    default = video_type.defaults_path()
    if existing_default:
        video_type.save_default("talking_head", ["folder"])
        default.write_text(json.dumps(video_type.load_default(), indent=4) + "\n\n")
    old_default = default.read_bytes() if default.exists() else None
    before = target.read_bytes()

    def fail(*args, **kwargs):
        assert video_type.load_default() == {
            "schema": video_type.DEFAULTS_SCHEMA, "video_type": "green_screen",
            "broll_sources": ["album"],
        }
        raise OutputPathError("synthetic project write failure")

    monkeypatch.setattr("yap_studio.video_type_command.write_json_cas", fail)
    assert cli.main(["video-type", "set", str(tmp_path), "--type", "green_screen",
                     "--broll", "album", *(["--default"] if existing_default else []),
                     "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["message"] == (
        "synthetic project write failure")
    assert target.read_bytes() == before
    assert (default.read_bytes() if default.exists() else None) == old_default


def test_save_default_only_if_missing():
    assert video_type.save_default("talking_head", ["folder"], only_if_missing=True) is True
    before = video_type.defaults_path().read_bytes()
    assert video_type.save_default("green_screen", ["album"], only_if_missing=True) is False
    assert video_type.defaults_path().read_bytes() == before
    assert video_type.save_default("green_screen", ["album"]) is True
    assert video_type.load_default()["video_type"] == "green_screen"


def test_save_default_holds_exclusive_lock(monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    original = video_type.paths.atomic_write_text

    def write(*args, **kwargs):
        with video_type.defaults_path().with_suffix(".lock").open("a+b") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return original(*args, **kwargs)

    monkeypatch.setattr(video_type.paths, "atomic_write_text", write)
    video_type.save_default("talking_head", ["folder"])
    with video_type.defaults_path().with_suffix(".lock").open("a+b") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)


@pytest.mark.parametrize("replace", [False, True])
def test_set_first_answer_keeps_concurrent_default(tmp_path, monkeypatch, capsys, replace):
    project_file(tmp_path)
    video_type.save_default("talking_head", ["folder"])
    before = video_type.defaults_path().read_bytes()
    load = video_type.load_default
    reads = 0

    def stale_once():
        nonlocal reads
        reads += 1
        return None if reads == 1 else load()

    monkeypatch.setattr(video_type, "load_default", stale_once)
    assert cli.main(["video-type", "set", str(tmp_path), "--type", "green_screen",
                     "--broll", "album", *(["--default"] if replace else []), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["default_saved"] is replace
    assert result["project"] == {"video_type": "green_screen", "broll_sources": ["album"]}
    if replace:
        assert result["default"] == result["project"]
    else:
        assert result["default"] == {"video_type": "talking_head", "broll_sources": ["folder"]}
        assert video_type.defaults_path().read_bytes() == before
