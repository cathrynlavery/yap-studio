"""First-run style answers and their real project and taste effects."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yap_studio import cli, ffmpeg
from yap_studio.errors import ValidationError
from yap_studio.project import load_project
from yap_studio.style import loader
from yap_studio.style.answers import (
    ANSWERS_SCHEMA,
    apply_answers,
    project_defaults,
    questions,
    style_changes,
    taste_rules,
    validate_answers,
)
from yap_studio.style.library import get_default, library_dir, save_pick, set_default
from yap_studio.taste.model import validate_body
from yap_studio.taste.store import approve_proposal, load_taste


def _answers(**values: object) -> dict:
    return {"schema": ANSWERS_SCHEMA, **values}


def test_questions_order_recommendations_and_current() -> None:
    rows = questions(_answers(cuts="relaxed", platforms=["reels"]))
    assert [row["id"] for row in rows] == [
        "finished_videos", "brand", "about", "platforms", "cuts", "music",
        "word_fixes", "never",
    ]
    assert rows[4]["current"] == "relaxed"
    assert rows[3]["current"] == ["reels"]
    assert rows[0]["choices"][0]["value"] == "yes"
    assert rows[0]["choices"][0]["recommended"] is True
    assert rows[4]["choices"][0]["value"] == "natural"
    assert rows[5]["choices"][0]["value"] == "none"
    assert all(row["current"] is None for row in rows if row["id"] not in {"cuts", "platforms"})


@pytest.mark.parametrize(("change", "location"), [
    ({"surprise": True}, "answers.surprise"),
    ({"cuts": "fast"}, "answers.cuts"),
    ({"brand": {"accent_color": "blue"}}, "answers.brand.accent_color"),
    ({"about": "x" * 201}, "answers.about"),
])
def test_validation_locations(change: dict, location: str) -> None:
    with pytest.raises(ValidationError, match=location.replace(".", r"\.")):
        validate_answers(_answers(**change))


@pytest.mark.parametrize(("other", "expected"), [
    ("", []), ("No jump scares.", ["No jump scares."]),
    ([], []), (["No jump scares."], ["No jump scares."]),
])
def test_never_other_normalizes_strings_without_mutating_answers(
    other: str | list[str], expected: list[str],
) -> None:
    answers = _answers(never={"other": other})
    validated = validate_answers(answers)
    assert validated["never"]["other"] == expected
    assert answers["never"]["other"] == other
    validated["never"]["other"].append("No zooms.")
    assert answers["never"]["other"] == other
    assert [rule["sentence"] for rule in taste_rules(answers)] == expected


@pytest.mark.parametrize("other", [
    None, True, 42, {}, " ", " leading space", "two\nlines", "x" * 201,
    [""], ["No zooms."] * 11,
])
def test_never_other_keeps_validation_for_invalid_rules(other: object) -> None:
    with pytest.raises(ValidationError, match=r"answers\.never\.other"):
        validate_answers(_answers(never={"other": other}))


@pytest.mark.parametrize(("other", "expected"), [
    ("", []), ("No jump scares.", ["No jump scares."]),
])
def test_from_answers_cli_accepts_never_other_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    other: str, expected: list[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(never={"cover_face": True, "other": other})))
    assert cli.main(["style", "from-answers", str(file), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["rules_approved"]) == 3 + len(expected)
    rules = load_taste().rules
    assert all(rule["status"] == "approved" for rule in rules)
    assert [rule["sentence"] for rule in rules if rule["machine"] is None] == expected


def test_music_and_brand_changes_follow_schema() -> None:
    base = loader.resolve("preset:clean_minimal").data
    assert style_changes(_answers(music="none"), base)["music"] == {"level": 0}
    assert style_changes(_answers(music="quiet"), base)["music"] == {"level": 0.04}
    assert style_changes(_answers(music="upbeat"), base)["music"] == {"level": 0.09}
    brand = {"website": "https://example.test", "accent_color": "#AABBCC"}
    assert style_changes(_answers(brand=brand), base)["brand"] == {
        "accent_color": "#AABBCC",
    }
    assert "pending" not in style_changes(_answers(brand=brand), base)


def test_project_defaults() -> None:
    assert project_defaults(_answers(cuts="tight"))["rules"] == {
        "max_pause_s": 0.2, "beat_breath_s": 0.3,
    }
    assert project_defaults(_answers(cuts="relaxed"))["rules"] == {
        "max_pause_s": 0.45, "beat_breath_s": 0.6,
    }
    all_vertical = project_defaults(_answers(platforms=["tiktok", "reels", "shorts"]))
    assert all_vertical["platform"] == "shorts"
    assert "platform" not in project_defaults(_answers(platforms=["youtube"]))
    assert project_defaults(_answers(music="none"))["music"] == {"mode": "none"}
    assert project_defaults(_answers(music="quiet"))["music"] == {
        "mode": "auto", "tags": ["quiet"],
    }
    assert project_defaults(_answers(word_fixes={"Yap": "YAP"}))["word_fixes"] == {"Yap": "YAP"}


def test_taste_answer_source_and_rules() -> None:
    rows = taste_rules(_answers(never={"cover_face": True, "emoji": True,
                                       "all_caps": True, "other": ["No jump scares."]}))
    assert len(rows) == 6
    assert {row["machine"]["layer"] for row in rows[:3]} == {"captions", "title", "cards"}
    for row in rows:
        validate_body(row)
        assert row["sources"][0]["type"] == "answer"
    with pytest.raises(ValidationError, match=r"sources\[0\]\.question"):
        validate_body({**rows[0], "sources": [{"type": "answer", "answer": "yes"}]})


def test_from_answers_cli_is_idempotent_and_revokes_only_answer_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My style")
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(music="quiet", never={"cover_face": True})))
    assert cli.main(["style", "from-answers", str(file), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["style_id"] == "my_style"
    assert get_default() == "my_style"
    assert (library_dir() / "my_style.json").is_file()
    assert loader.resolve().data["captions"]["placement"] == "avoid_face"
    assert loader.resolve().data["chapters"] == {"style": "none", "hold_s": 1.6}
    first_bytes = (library_dir() / "my_style.json").read_bytes()
    assert len([r for r in load_taste().rules if r["status"] == "approved"]) == 3
    revision = load_taste().revision
    assert cli.main(["style", "from-answers", str(file), "--json"]) == 0
    capsys.readouterr()
    assert load_taste().revision == revision
    assert (library_dir() / "my_style.json").read_bytes() == first_bytes
    file.write_text(json.dumps(_answers(music="quiet", never={"cover_face": False})))
    assert cli.main(["style", "from-answers", str(file), "--json"]) == 0
    capsys.readouterr()
    assert all(r["status"] == "revoked" for r in load_taste().rules)
    assert loader.resolve().data["captions"]["placement"] == "fixed"


def test_face_answer_edits_only_explicit_placement_choice() -> None:
    base = loader.resolve("preset:clean_minimal").data
    assert base["captions"]["placement"] == "avoid_face"
    assert style_changes(_answers(never={"cover_face": True}), base) == {
        "captions": {"placement": "avoid_face"},
    }
    assert style_changes(_answers(never={"cover_face": False}), base) == {
        "captions": {"placement": "fixed"},
    }
    assert style_changes(_answers(never={"emoji": True}), base) == {}


@pytest.mark.parametrize("never", [{}, {"emoji": True}, {"cover_face": True}])
def test_default_face_avoidance_survives_first_run_without_style_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, never: dict,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    source = tmp_path / "answers.json"
    source.write_text(json.dumps(_answers(never=never)))
    result = apply_answers(source)
    assert result["fields_set"] == [] and result["style_id"] is None
    assert loader.resolve().data["captions"]["placement"] == "avoid_face"


def test_no_style_change_leaves_library_and_default_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(cuts="tight")))
    workspace = tmp_path / "workspace"
    workspace_copy = workspace / "style/my-style.json"
    expected_keys = {"style_id", "style_saved", "fields_set", "rules_approved", "rules_revoked",
                     "project_defaults", "pending", "website_colors"}
    result = apply_answers(file, workspace=workspace)
    assert set(result) == expected_keys
    assert result["style_id"] is None
    assert result["fields_set"] == []
    assert get_default() is None
    assert not library_dir().exists()
    assert not workspace_copy.exists()
    assert cli.main(["style", "from-answers", str(file), "--workspace", str(workspace)]) == 0
    assert "Style unchanged." in capsys.readouterr().out
    assert get_default() is None
    assert not library_dir().exists()
    assert not workspace_copy.exists()

    file.write_text(json.dumps(_answers(music="quiet")))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My style")
    assert apply_answers(file)["style_id"] == "my_style"
    saved = (library_dir() / "my_style.json").read_bytes()
    config = (tmp_path / "config" / "config.json").read_bytes()
    file.write_text(json.dumps(_answers(cuts="tight")))
    result = apply_answers(file, workspace=workspace)
    assert result["style_id"] == "my_style"
    assert result["fields_set"] == []
    assert [path.name for path in library_dir().glob("*.json")] == ["my_style.json"]
    assert (library_dir() / "my_style.json").read_bytes() == saved
    assert (tmp_path / "config" / "config.json").read_bytes() == config
    assert not workspace_copy.exists()


def test_from_answers_workspace_writes_readable_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My style")
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(music="quiet")))
    workspace = tmp_path / "workspace"

    assert cli.main(["style", "from-answers", str(file), "--workspace", str(workspace),
                     "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["style_id"] == "my_style"
    workspace_copy = workspace / "style/my-style.json"
    assert workspace_copy.read_bytes() == (library_dir() / "my_style.json").read_bytes()
    saved = json.loads(workspace_copy.read_text())
    assert saved["name"] == "My style"
    assert "internal" not in saved and "schema" not in saved


def test_different_music_answers_update_saved_look_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    original = save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(music="quiet")))
    first = apply_answers(file)
    assert first["style_id"] == original.id and first["style_saved"] is True
    first_path = original.path
    first_bytes = first_path.read_bytes()
    first_data = json.loads(first_bytes)
    assert first_data["name"] == "My look"
    assert "internal" not in first_data
    assert loader.resolve(original.id).data["internal"]

    file.write_text(json.dumps(_answers(music="upbeat")))
    second = apply_answers(file)
    assert second["style_id"] == original.id
    assert second["fields_set"] == ["music.level"]
    assert get_default() == original.id
    assert first_path.read_bytes() != first_bytes
    assert list(library_dir().glob("*.json")) == [first_path]
    second_data = json.loads(first_path.read_text())
    assert second_data["name"] == "My look"
    assert "internal" not in second_data
    assert second_data["music"]["level"] == 0.09


@pytest.mark.parametrize("default", [None, "preset:electric"])
def test_answers_before_a_look_keep_fields_without_writing_a_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    default: str | None,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    if default:
        set_default(default)
    file = tmp_path / "style/answers.json"
    file.parent.mkdir()
    file.write_text(json.dumps(_answers(music="none", never={"cover_face": False},
                                       brand={"accent_color": "#114488"})))
    result = apply_answers(file, workspace=tmp_path)
    assert result["style_id"] is None and result["style_saved"] is False
    assert set(result["fields_set"]) == {
        "music.level", "captions.placement", "brand.accent_color",
    }
    assert get_default() == default
    assert not list(library_dir().glob("*.json"))
    assert not (tmp_path / "style/my-style.json").exists()
    assert cli.main(["style", "from-answers", str(file), "--workspace", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Style choices are kept with your answers until you save a look." in output
    assert "Default style:" not in output


def test_invalid_file_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(cuts="wrong")))
    assert cli.main(["style", "from-answers", str(file)]) == 2
    assert "answers.cuts" in capsys.readouterr().err


def test_questions_cli_marks_recommendations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(cuts="tight")))
    assert cli.main(["style", "questions", "--answers", str(file)]) == 0
    output = capsys.readouterr().out
    assert "1. Finished videos you like? Yes, pick a folder (recommended)" in output
    assert 'current: "tight"' in output


@pytest.mark.parametrize("json_output", [False, True])
def test_questions_cli_free_text_does_not_name_an_option(
    capsys: pytest.CaptureFixture[str], json_output: bool,
) -> None:
    assert cli.main(["style", "questions", *(["--json"] if json_output else [])]) == 0
    output = capsys.readouterr().out
    prompts = ({row["id"]: row["prompt"] for row in json.loads(output)}
               if json_output else {"text": output})
    text = " ".join(prompts.values())
    assert "Type your answer" in text
    assert "Type them as heard -> correct" in text
    assert "Type any additional rules" in text
    assert "Other" not in text and "Type something" not in text


def test_turning_off_answer_does_not_revoke_comment_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    comment = {
        "scope": "cards", "condition": None, "sentence": "No emoji on screen.",
        "machine": None, "context": None,
        "sources": [{"type": "comment", "media_sha256": "a" * 64,
                     "comment_id": "c1", "label": "fix", "start_frame": 1,
                     "end_frame": 2, "quote": "No emoji."}],
    }
    approved = approve_proposal(comment)
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(never={"emoji": False})))
    assert cli.main(["style", "from-answers", str(file)]) == 0
    capsys.readouterr()
    assert next(rule for rule in load_taste().rules if rule["id"] == approved.rule["id"])[
        "status"
    ] == "approved"


def test_setup_does_not_supersede_review_placement_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    comment = {
        "scope": "captions", "condition": None, "sentence": "Leave room for my face.",
        "machine": {"kind": "placement", "layer": "captions", "rule": "clear_of_face"},
        "context": None,
        "sources": [{"type": "comment", "media_sha256": "a" * 64,
                     "comment_id": "c1", "label": "fix", "start_frame": 1,
                     "end_frame": 2, "quote": "Leave room."}],
    }
    approved = approve_proposal(comment)
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(never={"cover_face": True})))
    assert cli.main(["style", "from-answers", str(file), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "taste" in result["pending"]
    assert next(rule for rule in load_taste().rules if rule["id"] == approved.rule["id"])[
        "status"
    ] == "approved"


def test_init_answers_write_project_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(
        ffmpeg, "probe", lambda _path: SimpleNamespace(has_video=True, has_audio=True)
    )
    take = tmp_path / "take.mov"
    take.write_bytes(b"synthetic take")
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(cuts="tight", platforms=["reels", "shorts"],
                                       music="quiet", word_fixes={"Yap": "YAP"},
                                       never={"cover_face": False},
                                       brand={"accent_color": "#114488"})))
    project_dir = tmp_path / "project"
    assert cli.main(["init", str(project_dir), "--take", str(take), "--answers", str(file)]) == 0
    capsys.readouterr()
    project = json.loads((project_dir / "project.json").read_text())
    assert project["rules"]["max_pause_s"] == 0.2
    assert project["rules"]["beat_breath_s"] == 0.3
    assert project["platform"] == "shorts"
    assert project["music"]["mode"] == "auto"
    assert project["music"]["tags"] == ["quiet"]
    assert project["word_fixes"] == {"Yap": "YAP"}
    loaded = load_project(project_dir / "project.json")
    assert loaded.to_dict() == project
    assert loaded.rules["max_pause_s"] == 0.2
    assert loaded.platform == "shorts"
    assert loaded.music["tags"] == ["quiet"]
    assert loaded.word_fixes == {"Yap": "YAP"}
    assert project["style"] == "styles/clean_minimal.json"
    snapshot = loader.resolve(str(project_dir / project["style"]))
    assert snapshot.data["music"]["level"] == 0.04
    assert snapshot.data["captions"]["placement"] == "fixed"
    assert snapshot.data["brand"]["accent_color"] == "#114488"
    assert get_default() is None and not library_dir().exists()


@pytest.mark.parametrize("default", [None, "preset:electric"])
@pytest.mark.parametrize("answers", [
    {}, {"cuts": "tight"}, {"never": {"cover_face": True}},
    {"brand": {"captions_font": "Unknown Face"}},
])
def test_init_answers_without_style_changes_keep_the_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default: str | None, answers: dict,
) -> None:
    from yap_studio.project import init_project

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    if default is not None:
        set_default(default)
    monkeypatch.setattr(ffmpeg, "probe", lambda _: SimpleNamespace(has_video=True, has_audio=True))
    take = tmp_path / "take.mov"
    take.write_bytes(b"synthetic take")
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(**answers)))
    directory = tmp_path / "project"
    project = init_project(directory, takes=[take], answers=file)
    assert project.style == (default or "preset:clean_minimal")
    assert not (directory / "styles").exists()
    if answers.get("cuts") == "tight":
        assert project.rules["max_pause_s"] == 0.2


def test_init_answer_style_copy_uses_preset_name_without_mutating_resolved_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio import project as module

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    selected = loader.resolve("preset:electric")
    before = copy.deepcopy(selected.unbranded_data)
    monkeypatch.setattr(module, "resolve", lambda _: selected)
    monkeypatch.setattr(ffmpeg, "probe", lambda _: SimpleNamespace(has_video=True, has_audio=True))
    take = tmp_path / "take.mov"
    take.write_bytes(b"synthetic take")
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(music="none", brand={"accent_color": "#114488"})))
    directory = tmp_path / "project"
    project = module.init_project(directory, takes=[take], answers=file)
    assert selected.unbranded_data == before
    assert project.style == "styles/electric.json"
    snapshot = json.loads((directory / project.style).read_text())
    assert snapshot["music"]["level"] == 0
    assert snapshot["brand"]["accent_color"] == "#114488"


@pytest.mark.parametrize("saved,explicit", [(True, False), (False, True)])
def test_init_answers_preserve_saved_or_explicit_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: bool, explicit: bool,
) -> None:
    from yap_studio.project import init_project

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    if saved:
        entry = save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    monkeypatch.setattr(
        ffmpeg, "probe", lambda _path: SimpleNamespace(has_video=True, has_audio=True)
    )
    take = tmp_path / "take.mov"
    take.write_bytes(b"synthetic take")
    file = tmp_path / "answers.json"
    file.write_text(json.dumps(_answers(music="none", never={"cover_face": False})))
    directory = tmp_path / "project"
    project = init_project(directory, takes=[take], answers=file,
                           style="preset:electric" if explicit else None)
    if saved:
        assert (directory / project.style).read_bytes() == entry.path.read_bytes()
    else:
        assert project.style == "preset:electric" and not (directory / "styles").exists()
