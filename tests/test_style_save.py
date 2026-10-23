"""Saving gallery picks and honoring card choices."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_planner_anchors import synthetic_words

from yap_studio import cards, ffmpeg
from yap_studio.cards import CARD_TEMPLATES
from yap_studio.cli import main
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.planner.cards import plan_cards
from yap_studio.project import init_project
from yap_studio.style import (
    PRESETS_DIR,
    SCHEMA_PATH,
    SHIPPED_PRESETS,
    ResolvedStyle,
    StyleError,
    cards_on,
    current_default_data,
    get_default,
    library_dir,
    read_style_file,
    resolve,
    resolve_style_data,
    save_pick,
)
from yap_studio.style import library as style_library

ROOT = Path(__file__).parents[1]
PROPOSED = ROOT / "docs/design/style-page/proposed-presets"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


@pytest.mark.parametrize("grade", [
    {"look": "film", "strength": 0.5, "color": None, "slider": False},
    {"look": "film", "strength": 0.5, "color": None, "slider": True},
    {"look": "eyedropper", "strength": 0.5, "color": "#1E6BD6", "slider": True},
])
def test_save_pick_default_and_workspace_copy(tmp_path: Path, grade: dict) -> None:
    data = resolve("preset:clean_minimal").to_json()
    data["name"] = "Old name"
    data["grade"] = grade
    entry = save_pick(data, name="My Look", workspace=tmp_path / "workspace")
    assert (entry.id, entry.name, entry.revision, entry.is_default) == (
        "my_look", "My Look", 1, True,
    )
    assert get_default() == entry.id
    assert entry.workspace_copy == tmp_path / "workspace/style/my-style.json"
    saved = json.loads(entry.path.read_text())
    assert saved["name"] == "My Look"
    assert saved["grade"] == resolve(entry.id).data["grade"] == data["grade"]
    assert "schema" not in saved and "internal" not in saved
    assert entry.workspace_copy.read_bytes() == entry.path.read_bytes()
    assert current_default_data() == resolve(entry.id).data


def test_revisions_reuse_identical_and_preserve_older_files(tmp_path: Path) -> None:
    first = save_pick({"schema_version": 1, "name": "My Look"}, workspace=tmp_path)
    original = first.path.read_bytes()
    assert (tmp_path / "style/my-style.json").read_bytes() == original
    same = save_pick({"schema_version": 1, "name": "My Look"})
    assert (same.id, same.revision) == (first.id, 1)
    second = save_pick(
        {"schema_version": 1, "name": "My Look", "graphics": {"style": "paper"}},
        workspace=tmp_path,
    )
    second_bytes = second.path.read_bytes()
    assert (tmp_path / "style/my-style.json").read_bytes() == second_bytes
    third = save_pick({"schema_version": 1, "name": "My Look", "graphics": {"style": "dark"}})
    assert (second.id, second.revision, third.id, third.revision) == (
        "my_look-2", 2, "my_look-3", 3,
    )
    assert first.path.read_bytes() == original
    assert second.path.read_bytes() == second_bytes
    assert get_default() == third.id


def test_find_saved_style_by_name_uses_newest_trimmed_case_insensitive_match() -> None:
    first = save_pick({"schema_version": 1}, name="My Look")
    second = save_pick({"schema_version": 1, "music": {"level": 0}}, name="MY LOOK")
    os.utime(first.path, ns=(1, 1))
    os.utime(second.path, ns=(2, 2))
    assert style_library.find_by_name("  my look  ").id == second.id
    assert style_library.find_by_name("Another look") is None


def test_save_pick_replace_keeps_id_and_updates_default_and_workspace(tmp_path: Path) -> None:
    first = save_pick({"schema_version": 1}, name="My Look", workspace=tmp_path)
    save_pick({"schema_version": 1}, name="Other")
    updated = save_pick({"schema_version": 1, "music": {"level": 0}}, name="My Look",
                        workspace=tmp_path, replace=first.id)
    assert updated.id == first.id and updated.path == first.path
    assert get_default() == first.id
    assert resolve(first.id).data["music"]["level"] == 0
    assert updated.workspace_copy.read_bytes() == first.path.read_bytes()
    assert len(list(library_dir().glob("*.json"))) == 2


@pytest.mark.parametrize("replace", [False, True])
def test_save_pick_keeps_workspace_ownership_after_replace(tmp_path: Path, replace: bool) -> None:
    first = save_pick({"schema_version": 1}, name="My Look", workspace=tmp_path)
    original = first.path.read_bytes()
    save_pick({"schema_version": 1, "music": {"level": 0}}, name="My Look", replace=first.id)
    assert first.workspace_copy.read_bytes() == original
    assert [entry.id for entry in style_library.list_entries()] == [first.id]
    assert list(library_dir().glob("*.json")) == [first.path]
    updated = save_pick({"schema_version": 1, "music": {"level": .12}},
                        name="My Look" if replace else "New Look", workspace=tmp_path,
                        replace=first.id if replace else None)
    assert updated.workspace_copy.read_bytes() == updated.path.read_bytes()
    foreign = b'{"schema_version": 1, "name": "Hand written"}\n'
    updated.workspace_copy.write_bytes(foreign)
    with pytest.raises(StyleError, match="not a copy"):
        save_pick({"schema_version": 1}, name="Another Look", workspace=tmp_path,
                  replace=first.id if replace else None)
    assert updated.workspace_copy.read_bytes() == foreign


def test_save_pick_replace_workspace_write_failure_restores_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = save_pick({"schema_version": 1}, name="My Look", workspace=tmp_path)
    original = first.path.read_bytes()
    original_write = style_library.paths.atomic_write_text

    def fail_workspace_write(path: Path, text: str) -> Path:
        if Path(path).parent.parent == first.workspace_copy.parent:
            raise OSError("synthetic workspace write failure")
        return original_write(path, text)

    monkeypatch.setattr(style_library.paths, "atomic_write_text", fail_workspace_write)
    with pytest.raises(StyleError) as error:
        save_pick({"schema_version": 1, "music": {"level": 0}}, name="My Look",
                  workspace=tmp_path, replace=first.id)
    assert first.path.read_bytes() == original
    assert first.workspace_copy.read_bytes() == original
    assert get_default() == first.id
    assert "style my_look was not updated because the copy could not be written" in str(error.value)


@pytest.mark.parametrize("second_action", ["save", "replace", "import"])
def test_save_pick_waits_for_failed_replace_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_action: str,
) -> None:
    first_workspace, second_workspace = tmp_path / "first", tmp_path / "second"
    first = save_pick({"schema_version": 1}, name="My Look", workspace=first_workspace)
    save_pick({"schema_version": 1}, name="My Look", workspace=second_workspace)
    source = tmp_path / "my_look.json"
    source.write_text(json.dumps({"schema_version": 1, "name": "My Look", "music": {"level": .12}}))
    first_blocked, release_first = threading.Event(), threading.Event()
    second_progress, second_done = threading.Event(), threading.Event()
    original_write = style_library.paths.atomic_write_text
    original_flock = fcntl.flock
    outcomes: dict[str, Exception | None] = {}
    second_entries: list[style_library.SavedPick | style_library.LibraryEntry] = []

    def fail_first_workspace_write(path: Path, text: str) -> Path:
        if Path(path).parent.parent == first.workspace_copy.parent:
            first_blocked.set()
            assert release_first.wait(5), "first writer was not released"
            raise OSError("synthetic workspace write failure")
        return original_write(path, text)

    def note_second_lock(descriptor: int, operation: int) -> None:
        if threading.current_thread() is second_thread and operation == fcntl.LOCK_EX:
            second_progress.set()
        original_flock(descriptor, operation)

    def replace(label: str, workspace: Path, level: float) -> None:
        try:
            if label == "second" and second_action == "import":
                entry = style_library.add(source, replace=True, make_default=True)
            else:
                entry = save_pick({"schema_version": 1, "music": {"level": level}}, name="My Look",
                                  workspace=workspace,
                                  replace=first.id if label == "first" or second_action == "replace"
                                  else None)
            if label == "second":
                second_entries.append(entry)
            outcomes[label] = None
        except Exception as exc:
            outcomes[label] = exc
        finally:
            if label == "second":
                second_done.set()
                second_progress.set()

    monkeypatch.setattr(style_library.paths, "atomic_write_text", fail_first_workspace_write)
    monkeypatch.setattr(fcntl, "flock", note_second_lock)
    first_thread = threading.Thread(target=replace, args=("first", first_workspace, 0))
    second_level = 0 if second_action == "save" else .12
    second_thread = threading.Thread(target=replace,
                                     args=("second", second_workspace, second_level))
    first_thread.start()
    try:
        assert first_blocked.wait(5), "first writer did not reach its workspace write"
        second_thread.start()
        assert second_progress.wait(5), "second writer did not reach the transaction"
        finished_early = second_done.is_set()
        blocked_bytes = first.path.read_bytes()
    finally:
        release_first.set()
        first_thread.join(5)
        if second_thread.ident is not None:
            second_thread.join(5)
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert isinstance(outcomes["first"], StyleError)
    assert "copy could not be written" in str(outcomes["first"])
    assert outcomes["second"] is None
    assert not finished_early, "second writer finished before the first rolled back"
    assert json.loads(blocked_bytes)["music"]["level"] == 0
    second = second_entries[0]
    final_bytes = second.path.read_bytes()
    assert json.loads(final_bytes)["music"]["level"] == second_level
    if second_action != "import":
        assert (second_workspace / "style/my-style.json").read_bytes() == final_bytes
        style_library._validate_workspace_copy(second.workspace_copy)
    assert get_default() == second.id
    assert {entry.id for entry in style_library.list_entries()} == {first.id, second.id}


def test_save_pick_unsafe_config_leaves_no_lock_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = tmp_path / ".ssh" / "yap-studio"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(unsafe))
    with pytest.raises(OutputPathError):
        save_pick({"schema_version": 1}, name="My Look")
    assert not unsafe.exists()
    assert not (unsafe / "styles/.save.lock").exists()
    assert not (tmp_path / ".ssh").exists()


def test_save_pick_refuses_symlinked_lock(tmp_path: Path) -> None:
    outside = tmp_path / "outside.lock"
    outside.write_text("synthetic lock target\n")
    original = outside.read_bytes()
    library_dir().mkdir(parents=True)
    lock = library_dir() / ".save.lock"
    lock.symlink_to(outside)
    with pytest.raises(StyleError, match="cannot lock style library") as error:
        save_pick({"schema_version": 1}, name="My Look")
    assert isinstance(error.value.__cause__, OSError)
    assert outside.read_bytes() == original
    assert lock.is_symlink()
    assert not list(library_dir().glob("*.json"))
    assert get_default() is None


@pytest.mark.parametrize("failure", ["open", "flock"])
@pytest.mark.parametrize("replace", [False, True])
def test_save_pick_reports_lock_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, replace: bool,
) -> None:
    first = save_pick({"schema_version": 1}, name="My Look", workspace=tmp_path)
    original = first.path.read_bytes()
    lock_error = OSError("synthetic lock failure")
    original_open = os.open

    def fail_open(
        path: Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> int:
        if Path(path) == library_dir() / ".save.lock":
            raise lock_error
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def fail_flock(descriptor: int, operation: int) -> None:
        raise lock_error

    if failure == "open":
        monkeypatch.setattr(os, "open", fail_open)
    else:
        monkeypatch.setattr(fcntl, "flock", fail_flock)
    with pytest.raises(StyleError, match="cannot lock style library") as error:
        save_pick({"schema_version": 1, "music": {"level": 0}}, name="My Look",
                  workspace=tmp_path, replace=first.id if replace else None)
    assert error.value.parameter == "style.library"
    assert error.value.__cause__ is lock_error
    assert first.path.read_bytes() == original
    assert first.workspace_copy.read_bytes() == original
    assert get_default() == first.id


def test_save_pick_replace_reports_restore_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = save_pick({"schema_version": 1}, name="My Look", workspace=tmp_path)
    original = first.path.read_text()
    original_write = style_library.paths.atomic_write_text
    original_entry = style_library._write_entry
    restore_error = OSError("synthetic restore failure")

    def fail_workspace_write(path: Path, text: str) -> Path:
        if Path(path).parent.parent == first.workspace_copy.parent:
            raise OSError("synthetic workspace write failure")
        return original_write(path, text)

    def fail_restore(path: Path, text: str) -> None:
        if path == first.path and text == original:
            raise restore_error
        original_entry(path, text)

    monkeypatch.setattr(style_library.paths, "atomic_write_text", fail_workspace_write)
    monkeypatch.setattr(style_library, "_write_entry", fail_restore)
    with pytest.raises(StyleError, match="style my_look could not be restored") as error:
        save_pick({"schema_version": 1, "music": {"level": 0}}, name="My Look",
                  workspace=tmp_path, replace=first.id)
    assert error.value.parameter == "style.library"
    assert error.value.__cause__ is restore_error
    assert "synthetic restore failure" in str(error.value)
    assert first.workspace_copy.read_text() == original


@pytest.mark.parametrize(("data", "name"), [
    ({"schema_version": 99}, "My Look"), ({"schema_version": 1}, "!!!"),
])
@pytest.mark.parametrize("replace", [None, "missing"])
def test_save_pick_invalid_input_leaves_no_library(
    data: dict, name: str, replace: str | None,
) -> None:
    with pytest.raises(StyleError):
        save_pick(data, name=name, replace=replace)
    assert not library_dir().exists()


def test_save_pick_replace_refuses_symlink_before_archiving(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"schema_version": 1, "name": "Outside"}\n')
    original = outside.read_bytes()
    library_dir().mkdir(parents=True)
    target = library_dir() / "linked.json"
    target.symlink_to(outside)
    with pytest.raises(StyleError, match="choose a saved library id") as error:
        save_pick({"schema_version": 1}, name="My Look", replace="linked")
    assert error.value.parameter == "style.replace"
    assert not (library_dir() / ".replaced").exists()
    assert outside.read_bytes() == original
    assert target.is_symlink()
    assert target.readlink() == outside
    assert get_default() is None


def test_save_pick_replace_refuses_unknown_id_and_foreign_workspace(tmp_path: Path) -> None:
    first = save_pick({"schema_version": 1}, name="My Look")
    before = first.path.read_bytes()
    with pytest.raises(StyleError):
        save_pick({"schema_version": 1}, name="My Look", replace="missing")
    workspace_copy = tmp_path / "style/my-style.json"
    workspace_copy.parent.mkdir()
    workspace_copy.write_text('{"schema_version": 1, "name": "Hand written"}')
    with pytest.raises(StyleError, match="not a copy"):
        save_pick({"schema_version": 1, "music": {"level": 0}}, name="My Look",
                  workspace=tmp_path, replace=first.id)
    assert first.path.read_bytes() == before


def test_save_pick_refuses_foreign_workspace_file_before_changing_library(tmp_path: Path) -> None:
    first = save_pick({"schema_version": 1, "name": "Earlier"})
    library_before = {path.name: path.read_bytes() for path in library_dir().glob("*.json")}
    workspace_copy = tmp_path / "workspace/style/my-style.json"
    workspace_copy.parent.mkdir(parents=True)
    foreign_bytes = b'{"schema_version": 1, "name": "Hand written"}\n'
    workspace_copy.write_bytes(foreign_bytes)

    with pytest.raises(
        StyleError,
        match=(
            "style/my-style.json is not a copy of a saved style; "
            "move it or pass another --workspace"
        ),
    ):
        save_pick({"schema_version": 1, "name": "My Look"}, workspace=workspace_copy.parent.parent)

    assert {path.name: path.read_bytes() for path in library_dir().glob("*.json")} == library_before
    assert get_default() == first.id
    assert workspace_copy.read_bytes() == foreign_bytes


def test_save_pick_workspace_write_failure_keeps_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = save_pick({"schema_version": 1, "name": "Earlier"})
    workspace_copy = tmp_path / "workspace/style/my-style.json"
    original_write = style_library.paths.atomic_write_text

    def fail_workspace_write(path: Path, text: str) -> Path:
        if Path(path) == workspace_copy:
            raise OSError("synthetic workspace write failure")
        return original_write(path, text)

    monkeypatch.setattr(style_library.paths, "atomic_write_text", fail_workspace_write)
    with pytest.raises(
        StyleError,
        match=(
            "saved to the library as my_look but not made the default "
            "because the copy could not be written"
        ),
    ):
        save_pick({"schema_version": 1, "name": "My Look"}, workspace=tmp_path / "workspace")

    assert (library_dir() / "my_look.json").is_file()
    assert get_default() == first.id
    assert not workspace_copy.exists()


def test_read_style_file_returns_raw_json(tmp_path: Path) -> None:
    source = tmp_path / "style.json"
    data = {"schema_version": 1, "name": "My Look"}
    source.write_text(json.dumps(data))
    assert read_style_file(source) == data


def test_projects_keep_style_when_default_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ffmpeg, "probe", lambda _: SimpleNamespace(has_video=True, has_audio=True))
    take = tmp_path / "take.mov"
    take.write_bytes(b"synthetic")
    first = save_pick({"schema_version": 1, "name": "My Look"})
    earlier = init_project(tmp_path / "earlier", takes=[take])
    assert earlier.style == "styles/my_look.json"
    second = save_pick({"schema_version": 1, "name": "My Look", "graphics": {"style": "dark"}})
    later = init_project(tmp_path / "later", takes=[take])
    assert later.style == "styles/my_look-2.json"
    assert resolve(tmp_path / "earlier" / earlier.style).data == resolve(first.id).data
    assert resolve(tmp_path / "later" / later.style).data == resolve(second.id).data


def test_style_save_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "style.json"
    source.write_text('{"schema_version": 1, "name": "My Look", '
                      '"grade": {"look": "film", "strength": 0.5}}')
    workspace = tmp_path / "workspace"
    assert main(["style", "save", str(source), "--workspace", str(workspace)]) == 0
    out = capsys.readouterr()
    assert 'Saved "My Look" as your style (my_look). New videos use it.' in out.out
    assert str(workspace / "style/my-style.json") in out.out
    assert resolve("my_look").data["grade"] == {
        "look": "film", "strength": 0.5, "color": None, "slider": False}
    assert main(["style", "save", str(source), "--name", "Other Look", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["id"], result["name"], result["revision"], result["is_default"]) == (
        "other_look", "Other Look", 1, True,
    )
    assert resolve(result["id"]).data["grade"] == {
        "look": "film", "strength": 0.5, "color": None, "slider": False}
    source.write_text("not json")
    assert main(["style", "save", str(source)]) == 2
    error = capsys.readouterr().err
    assert "error:" in error and "fix:" in error


def test_card_kinds_schema_and_presets() -> None:
    minimal = {"schema_version": 1, "graphics": {"card_kinds_off": ["stat", "flow"]}}
    assert resolve_style_data(minimal)["graphics"]["card_kinds_off"] == ["stat", "flow"]
    assert resolve_style_data({"schema_version": 1})["graphics"]["card_kinds_off"] == []
    for kinds in (["unknown"], ["stat", "stat"]):
        changed = copy.deepcopy(minimal)
        changed["graphics"]["card_kinds_off"] = kinds
        with pytest.raises(StyleError, match="card_kinds_off"):
            resolve_style_data(changed)
    for path in [*(PRESETS_DIR / f"{name}.json" for name in SHIPPED_PRESETS),
                 *PROPOSED.glob("*.json")]:
        if path.name != "order.json":
            resolve_style_data(json.loads(path.read_text()))


def test_card_kinds_schema_matches_card_templates() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    graphics = schema["properties"]["graphics"]["properties"]
    rule = graphics["card_kinds_off"]
    base_kinds = tuple(kind for kind in CARD_TEMPLATES if kind not in cards.OPT_IN_KINDS)
    assert tuple(rule["items"]["enum"]) == base_kinds
    assert rule["maxItems"] == len(base_kinds)
    on = graphics["card_kinds_on"]
    assert on["default"] == [] and on["uniqueItems"] is True
    assert "enum" not in on["items"]
    assert all(kind in CARD_TEMPLATES for kind in cards.OPT_IN_KINDS)


def test_card_kinds_on_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ())
    monkeypatch.setattr(registry_cards, "kinds", lambda: ())
    data = {"schema_version": 1, "graphics": {"card_kinds_on": ["unknown"]}}
    with pytest.raises(StyleError, match="no card kind needs turning on") as caught:
        resolve_style_data(data)
    assert caught.value.parameter == "style.graphics.card_kinds_on[0]"
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("fake_kind",))
    with pytest.raises(StyleError, match="choose one of fake_kind"):
        resolve_style_data(data)
    data["graphics"]["card_kinds_on"] = ["fake_kind"]
    assert resolve_style_data(data)["graphics"]["card_kinds_on"] == ["fake_kind"]
    data["graphics"]["card_kinds_on"] *= 2
    with pytest.raises(StyleError, match="choose each card kind only once"):
        resolve_style_data(data)


def test_card_kind_off_honors_opt_in_and_base_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("stat",))
    data = resolve("preset:clean_minimal").data
    data["graphics"].pop("card_kinds_on", None)
    data["graphics"]["card_kinds_off"] = []
    style = ResolvedStyle(data["name"], "test", data)
    assert cards.card_kind_off(style, "stat") is True
    assert "stat" not in cards_on(style)
    assert cards.card_kind_off({"graphics": {}}, "stat") is True
    assert cards.card_kind_off({"graphics": {}}, "text_tag") is False
    assert cards.card_kind_off({}, "text_tag") is False
    data["graphics"]["card_kinds_on"] = ["stat"]
    data["graphics"]["card_kinds_off"] = ["stat", "text_tag"]
    assert cards.card_kind_off(style, "stat") is False
    assert "stat" in cards_on(style)
    assert cards.card_kind_off(data, "text_tag") is True
    assert cards.card_kind_off(style, "flow") is False


def test_save_pick_round_trips_opt_in_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("stat",))
    entry = save_pick({"schema_version": 1, "name": "Opt in",
                       "graphics": {"card_kinds_on": ["stat"]}})
    style = resolve(entry.id)
    assert style.data["graphics"]["card_kinds_on"] == ["stat"]
    assert cards.card_kind_off(style, "stat") is False
    old = save_pick({"schema_version": 1, "name": "Old style",
                     "graphics": {"card_kinds_off": []}})
    old_style = resolve(old.id)
    assert old_style.data["graphics"]["card_kinds_on"] == []
    assert cards.card_kind_off(old_style, "stat") is True


def test_cards_on_includes_every_template_by_default() -> None:
    assert cards_on(resolve("preset:clean_minimal")) == tuple(
        kind for kind in CARD_TEMPLATES if kind not in cards.OPT_IN_KINDS
    )


@pytest.mark.parametrize("kind", ("prompt_bar", "comment_cta", "comment_drop", "icon_strike",
                                 "choices"))
def test_chat_kinds_belong_only_in_card_kinds_on(kind: str) -> None:
    data = resolve_style_data({"schema_version": 1, "graphics": {"card_kinds_on": [kind]}})
    assert kind in cards_on(ResolvedStyle(data["name"], "test", data))
    with pytest.raises(StyleError, match="card_kinds_off"):
        resolve_style_data({"schema_version": 1, "graphics": {"card_kinds_off": [kind]}})


def test_cards_on_excludes_opt_in_kinds_by_default() -> None:
    assert cards_on(resolve("preset:clean_minimal")) == tuple(
        kind for kind in CARD_TEMPLATES if kind not in cards.OPT_IN_KINDS)


def test_plan_cards_rejects_off_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_planner_cards import FRAMES, requests

    monkeypatch.setattr(cards, "OPT_IN_KINDS", tuple(
        kind for kind in cards.OPT_IN_KINDS if kind != "stat"))
    data = resolve("preset:clean_minimal").data
    data["graphics"]["card_kinds_off"] = ["stat"]
    style = ResolvedStyle(data["name"], "test", data)
    assert cards_on(style) == tuple(
        kind for kind in CARD_TEMPLATES if not cards.card_kind_off(style, kind))
    assert "stat" not in cards_on(style)
    assert cards_on(style) == tuple(
        kind for kind in CARD_TEMPLATES if kind != "stat" and kind not in cards.OPT_IN_KINDS
    )
    with pytest.raises(ValidationError, match="creator turned off stat cards") as caught:
        plan_cards(requests(), synthetic_words(), style, frames=FRAMES)
    assert caught.value.parameter == "cards[0].template"
    data["graphics"]["card_kinds_off"] = []
    planned = plan_cards(requests(), synthetic_words(), style, frames=FRAMES)
    assert planned.cards[0].template == "stat"


def test_plan_cards_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_planner_cards import FRAMES, requests

    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("stat",))
    data = resolve("preset:clean_minimal").data
    data["graphics"].pop("card_kinds_on", None)
    data["graphics"]["card_kinds_off"] = []
    style = ResolvedStyle(data["name"], "test", data)
    with pytest.raises(ValidationError, match="creator has not turned on stat cards") as caught:
        plan_cards(requests(), synthetic_words(), style, frames=FRAMES)
    assert caught.value.parameter == "cards[0].template"
    allowed = str(caught.value).split("use one of ")[1].split(" or no card")[0]
    assert "stat" not in allowed.split(", ")
    data["graphics"]["card_kinds_on"] = ["stat"]
    planned = plan_cards(requests(), synthetic_words(), style, frames=FRAMES)
    assert planned.cards[0].template == "stat"


def test_parser_does_not_import_save_implementation() -> None:
    script = (
        "import sys\n"
        "from yap_studio.cli import _parser\n"
        "before = set(sys.modules)\n"
        "parser = _parser()\n"
        "assert parser.parse_args(['style', 'save', 'look.json']).style_command == 'save'\n"
        "assert not {name for name in set(sys.modules) - before "
        "if name.startswith('yap_studio.style.')}\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
