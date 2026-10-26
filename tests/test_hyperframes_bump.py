"""The HyperFrames pin bump, tested on temporary copies with npm replaced by a stub."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from yap_studio.errors import ValidationError, YapError

ROOT = Path(__file__).parents[1]
LOCK = "src/yap_studio/render/renderer-lock/package-lock.json"
CATALOG = "src/yap_studio/hf_registry/catalog.json"
COMMIT = "b" * 40
pytestmark = pytest.mark.usefixtures("_no_real_node")


def _load(root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the copied script with its sibling release helpers importable."""
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "hyperframes_bump_fixture", root / "scripts" / "bump_hyperframes.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bump_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ModuleType]:
    """Copy every pinned file so a bump cannot change the real checkout."""
    module = _load(ROOT, monkeypatch)
    files = [*module.PINS, "CHANGELOG.md", LOCK, CATALOG, "NOTICE",
             "scripts/bump_hyperframes.py", "scripts/bump_version.py"]
    if (ROOT / module.CHANGE_NOTE).exists():
        files.append(module.CHANGE_NOTE)
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    source = tmp_path / CATALOG
    data = json.loads(source.read_text(encoding="utf-8"))
    for name, entry in data["items"].items():
        entry["files"] = {entry["main"]: _digest(_content(name))}
    data["notes"] = {"version": data["hyperframes_version"], "commit": data["registry"]["commit"]}
    source.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    module = _load(tmp_path, monkeypatch)

    def run_git(args: list[str], cwd: Path) -> str:
        version = _next(module.pinned_version(tmp_path))
        assert args == ["ls-remote", "https://github.com/heygen-com/hyperframes",
                        f"refs/tags/v{version}", f"refs/tags/v{version}^{{}}"]
        assert cwd == tmp_path
        return f"{'a' * 40}\trefs/tags/v{version}\n{COMMIT}\trefs/tags/v{version}^{{}}\n"

    monkeypatch.setattr(module, "run_git", run_git)
    return tmp_path, module


def _content(name: str) -> str:
    return f"<!-- hyperframes-registry-item: {name} -->\n<div>{name}</div>\n"


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _next(version: str) -> str:
    """Choose the next patch without coupling the test to a particular pin."""
    major, minor, patch = version.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def _snapshot(root: Path, module: ModuleType) -> dict[str, str | None]:
    """Read every file a bump may write."""
    snapshot = {relative: (root / relative).read_bytes().decode("utf-8")
                for relative in [*module.PINS, "CHANGELOG.md", LOCK, CATALOG, "NOTICE"]}
    note = root / module.CHANGE_NOTE
    snapshot[module.CHANGE_NOTE] = note.read_bytes().decode("utf-8") if note.exists() else None
    return snapshot


def _stub_npm(module: ModuleType, monkeypatch: pytest.MonkeyPatch, new: str, *,
              npm: str = "11.6.0", integrity: str = "sha512-fixture",
              payloads: dict[str, dict[str, str]] | None = None,
              failure: str | None = None) -> list[list[str]]:
    """Answer npm calls from memory and lock the requested version on install."""
    calls: list[list[str]] = []

    def run(args: list[str], cwd: Path) -> str:
        calls.append(args)
        if args == ["--version"]:
            return npm + "\n"
        if args[0] == "install":
            lock_path = cwd / "package-lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["packages"]["node_modules/hyperframes"].update(
                version=new, integrity="sha512-fixture")
            lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
            return ""
        if args[:2] == ["view", f"hyperframes@{new}"]:
            return json.dumps(integrity)
        if args[0] == "ci":
            assert args == ["ci", "--ignore-scripts", "--no-audit", "--no-fund"]
            assert not cwd.is_relative_to(module.ROOT)
            manifest = json.loads((cwd / "package.json").read_text(encoding="utf-8"))
            assert manifest["dependencies"]["hyperframes"] == new
            lock = json.loads((cwd / "package-lock.json").read_text(encoding="utf-8"))
            assert lock["packages"]["node_modules/hyperframes"]["version"] == new
            if failure == "ci":
                raise subprocess.CalledProcessError(1, ["npm", *args], stderr="fixture ci failed")
            data = json.loads((module.ROOT / CATALOG).read_text(encoding="utf-8"))
            contents = payloads if payloads is not None else {
                name: {entry["main"]: _content(name)} for name, entry in data["items"].items()
            }
            executable = cwd / "node_modules/.bin/hyperframes"
            executable.parent.mkdir(parents=True)
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\nfrom pathlib import Path\n"
                "name = sys.argv[2]\nstaging = Path(sys.argv[4])\n"
                "assert sys.argv[1:] == ['add', name, '--dir', str(staging), "
                "'--json', '--no-clipboard']\n"
                f"temporary = Path({str(cwd.parent)!r})\n"
                "for key in ['HOME', 'HYPERFRAMES_HISTORY_ROOT']:\n"
                "    folder = Path(os.environ[key])\n"
                "    assert folder.is_relative_to(temporary) and folder.is_dir()\n"
                "assert os.environ['HYPERFRAMES_NO_TELEMETRY'] == '1'\n"
                "assert os.environ.get('CI') == '1'\n"
                "config = json.loads((staging / 'hyperframes.json').read_text())\n"
                f"assert config == {{'registry': 'https://raw.githubusercontent.com/"
                f"heygen-com/hyperframes/{COMMIT}/registry', 'paths': {{'blocks': 'compositions', "
                "'components': 'compositions/components', 'assets': 'assets'}}\n"
                f"payloads = {contents!r}\nwritten = []\n"
                "for relative, content in payloads[name].items():\n"
                "    target = staging / relative\n"
                "    target.parent.mkdir(parents=True, exist_ok=True)\n"
                "    target.write_text(content)\n    written.append(str(target))\n"
                f"ok = not ({failure == 'add'!r} and name == {list(contents)[-1]!r})\n"
                "print(json.dumps({'ok': ok, 'written': written, 'error': 'fixture add failed'}))\n"
                "sys.exit(0 if ok else 7)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            return ""
        raise AssertionError(f"unexpected npm call {args}")

    monkeypatch.setattr(module, "run_npm", run)
    return calls


def test_bump_moves_every_pin_and_reports_lock_changes(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each pin changes once, nothing else in its file changes, and the lock is verified."""
    root, module = bump_copy
    current = module.pinned_version(root)
    new = _next(current)
    before = _snapshot(root, module)
    calls = _stub_npm(module, monkeypatch, new)
    result = module.bump(root, new)
    entries = json.loads(before[CATALOG])["items"]
    assert result == {"current": current, "new": new, "changed": True,
                      "lock_changes": [f"hyperframes {current} -> {new}"],
                      "registry": {"commit": COMMIT, "changed": [], "unchanged": list(entries)}}
    assert calls[1][:2] == ["install", "--package-lock-only"]
    assert "--ignore-scripts" in calls[1]
    for relative, pattern in module.PINS.items():
        old = re.search(pattern.format(v=re.escape(current)), before[relative])
        assert old is not None
        expected = before[relative].replace(old[0], old[0].replace(current, new), 1)
        assert (root / relative).read_text(encoding="utf-8") == expected
    changelog = (root / "CHANGELOG.md").read_bytes().decode("utf-8")
    pin = "- Pin the renderer to HyperFrames "
    unreleased, released = before["CHANGELOG.md"].split("## Unreleased\n", 1)[1].split(
        "\n## ", 1)
    updated, updated_released = changelog.split("## Unreleased\n", 1)[1].split("\n## ", 1)
    assert updated_released.encode("utf-8") == released.encode("utf-8")
    assert pin not in updated
    assert updated.splitlines(keepends=True) == [
        line for line in unreleased.splitlines(keepends=True) if not line.startswith(pin)
    ]
    note = (root / module.CHANGE_NOTE).read_text(encoding="utf-8")
    assert note.count(pin) == 1
    assert f"{pin}{new}" in note
    assert f"{pin}{current}" not in note
    from bump_version import change_notes

    assert [(path, kind) for path, kind, _ in change_notes(root)] == [
        (root / module.CHANGE_NOTE, "dependencies"),
    ]
    assert module.pinned_version(root) == new
    assert calls[-1] == ["ci", "--ignore-scripts", "--no-audit", "--no-fund"]
    data = json.loads(before[CATALOG])
    expected = before[CATALOG]
    for key, old, updated in [
        ("hyperframes_version", current, new),
        ("tag", f"v{current}", f"v{new}"),
        ("commit", data["registry"]["commit"], COMMIT),
        ("url", data["registry"]["url"],
         f"https://raw.githubusercontent.com/heygen-com/hyperframes/{COMMIT}/registry"),
    ]:
        expected = expected.replace(f'"{key}": "{old}"', f'"{key}": "{updated}"', 1)
    assert (root / CATALOG).read_text(encoding="utf-8") == expected


def test_bump_to_the_pinned_version_changes_nothing(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An up-to-date pin writes no file and never calls npm."""
    root, module = bump_copy
    current = module.pinned_version(root)
    before = _snapshot(root, module)
    calls = _stub_npm(module, monkeypatch, current)
    assert module.bump(root, current)["changed"] is False
    assert calls == []
    assert _snapshot(root, module) == before


@pytest.mark.parametrize("registry_only", [False, True])
def test_bump_updates_only_the_registry_notice_commit(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch, registry_only: bool,
) -> None:
    root, module = bump_copy
    current = module.pinned_version(root)
    new = current if registry_only else _next(current)
    notice = root / "NOTICE"
    original = notice.read_bytes().replace(b"\n", b"\r\n")
    old = re.search(rb"(?m)^at commit ([0-9a-f]{40})\b", original)[1]
    original = b"Unrelated at commit " + old + b".\r\n\r\n" + original
    notice.write_bytes(original)
    target = re.search(rb"(?m)^at commit ([0-9a-f]{40})\b", original)
    expected = original[:target.start(1)] + COMMIT.encode() + original[target.end(1):]
    before = _snapshot(root, module)
    _stub_npm(module, monkeypatch, new)
    if registry_only:
        monkeypatch.setattr(module, "run_git", lambda args, cwd: f"{COMMIT}\trefs/tags/v{new}\n")
    assert module.bump(root, new, registry_only=registry_only)["changed"] is True
    assert json.loads((root / CATALOG).read_text())["registry"]["commit"] == COMMIT
    assert notice.read_bytes() == expected
    if registry_only:
        after = _snapshot(root, module)
        for relative, content in before.items():
            if relative not in {CATALOG, "NOTICE"}:
                assert after[relative] == content


@pytest.mark.parametrize("registry_only", [False, True])
@pytest.mark.parametrize("occurrences", [0, 2])
def test_bump_refuses_missing_or_duplicate_notice_sentence_and_restores_every_file(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch,
    registry_only: bool, occurrences: int,
) -> None:
    root, module = bump_copy
    notice = root / "NOTICE"
    original = notice.read_bytes()
    prefix = b"Setup downloads the HyperFrames registry items"
    paragraph = next(part for part in original.split(b"\n\n") if part.startswith(prefix))
    if occurrences == 0:
        malformed = original.replace(prefix, b"Setup downloads other items")
    else:
        malformed = original.replace(paragraph, paragraph + b"\n\n" + paragraph)
    notice.write_bytes(malformed)
    before = _snapshot(root, module)
    current = module.pinned_version(root)
    new = current if registry_only else _next(current)
    _stub_npm(module, monkeypatch, new)
    if registry_only:
        monkeypatch.setattr(module, "run_git", lambda args, cwd: f"{COMMIT}\trefs/tags/v{new}\n")
    with pytest.raises(YapError, match="NOTICE.*files were restored") as caught:
        module.bump(root, new, registry_only=registry_only)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert _snapshot(root, module) == before


def test_bump_refuses_an_older_release(bump_copy: tuple[Path, ModuleType]) -> None:
    """A downgrade is refused before any file is read for writing."""
    root, module = bump_copy
    with pytest.raises(YapError, match="older than the pinned"):
        module.bump(root, "0.0.1")


def test_bump_refuses_npm_10_before_writing(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An npm that drops libc fields is refused with the install command."""
    root, module = bump_copy
    before = _snapshot(root, module)
    _stub_npm(module, monkeypatch, _next(module.pinned_version(root)), npm="10.9.8")
    with pytest.raises(YapError) as caught:
        module.bump(root, _next(module.pinned_version(root)))
    assert "npm install -g npm@11" in caught.value.suggested_action
    assert _snapshot(root, module) == before


def test_bump_restores_files_when_a_pin_moved(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pin that no longer matches stops the bump and leaves every file as it was."""
    root, module = bump_copy
    current = module.pinned_version(root)
    moved = root / "tests/test_studio_export.py"
    moved.write_text(moved.read_text(encoding="utf-8").replace(
        f'.version == "{current}"', f'.version  == "{current}"'), encoding="utf-8")
    before = _snapshot(root, module)
    _stub_npm(module, monkeypatch, _next(current))
    with pytest.raises(YapError, match="files were restored"):
        module.bump(root, _next(current))
    assert _snapshot(root, module) == before


@pytest.mark.parametrize("note_exists", [False, True])
def test_bump_restores_files_when_the_lock_does_not_match_npm(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch, note_exists: bool,
) -> None:
    """A lock whose integrity differs from the registry is rolled back with the pins."""
    root, module = bump_copy
    note = root / module.CHANGE_NOTE
    original_note = (
        f"- Pin the renderer to HyperFrames {module.pinned_version(root)} (creator words).\r\n\r\n"
    ).encode()
    if note_exists:
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_bytes(original_note)
    else:
        note.unlink(missing_ok=True)
    before = _snapshot(root, module)
    new = _next(module.pinned_version(root))
    _stub_npm(module, monkeypatch, new, integrity="sha512-other")
    with pytest.raises(YapError, match="does not match npm"):
        module.bump(root, new)
    assert _snapshot(root, module) == before
    if note_exists:
        assert note.read_bytes() == original_note
    else:
        assert not note.exists()


@pytest.mark.parametrize("change", ["hash", "add", "remove"])
def test_bump_refreshes_changed_item_files(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    """Changed bytes and added or removed files update only the affected item."""
    root, module = bump_copy
    source = root / CATALOG
    data = json.loads(source.read_text(encoding="utf-8"))
    name = next(iter(data["items"]))
    entry = data["items"][name]
    extra = "assets/extra.txt"
    if change == "remove":
        entry["files"][extra] = _digest("removed")
    # Keep unrelated metadata even when it repeats a hash being changed.
    entry["note"] = entry["files"][entry["main"]]
    before_text = (json.dumps(data, indent=2) + "\n").replace(
        '"files": {', '"files" :  {',
    ).replace('"status": "pilot"', '"status" :    "pilot"')
    source.write_text(before_text, encoding="utf-8")
    payloads = {
        key: {item["main"]: _content(key)} for key, item in data["items"].items()
    }
    if change == "hash":
        payloads[name][entry["main"]] += "changed\n"
    elif change == "add":
        payloads[name][extra] = "added\n"
    new = _next(module.pinned_version(root))
    _stub_npm(module, monkeypatch, new, payloads=payloads)
    result = module.bump(root, new)
    updated = json.loads(source.read_text(encoding="utf-8"))
    assert result["registry"] == {
        "commit": COMMIT, "changed": [name], "unchanged": list(data["items"])[1:],
    }
    expected = {relative: _digest(content) for relative, content in payloads[name].items()}
    assert updated["items"][name] == {**entry, "files": expected}
    assert list(updated["items"].values())[1:] == list(data["items"].values())[1:]
    if change == "hash":
        current = data["hyperframes_version"]
        expected_text = before_text.replace(
            f'"hyperframes_version": "{current}"', f'"hyperframes_version": "{new}"',
        ).replace(f'"tag": "v{current}"', f'"tag": "v{new}"').replace(
            f'"commit": "{data["registry"]["commit"]}"', f'"commit": "{COMMIT}"', 1,
        ).replace(data["registry"]["url"],
                  f"https://raw.githubusercontent.com/heygen-com/hyperframes/{COMMIT}/registry")
        expected_text = expected_text.replace(
            f'"{entry["main"]}": "{entry["files"][entry["main"]]}"',
            f'"{entry["main"]}": "{expected[entry["main"]]}"',
        )
        assert source.read_text(encoding="utf-8") == expected_text


@pytest.mark.parametrize("failure", ["tag", "ci", "add"])
def test_registry_failure_restores_every_file(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Registry failures restore the catalog as well as pins, lock and changelog."""
    root, module = bump_copy
    before = _snapshot(root, module)
    new = _next(module.pinned_version(root))
    _stub_npm(module, monkeypatch, new, failure=failure)
    if failure == "tag":
        monkeypatch.setattr(module, "run_git", lambda args, cwd: "")
    with pytest.raises(YapError, match={
        "tag": f"tag v{new}", "ci": "npm.*ci", "add": "fixture add failed",
    }[failure]) as caught:
        module.bump(root, new)
    assert caught.value.suggested_action
    if failure == "tag":
        assert "tag" in caught.value.suggested_action
    assert _snapshot(root, module) == before


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("note_exists", [False, True])
def test_registry_flag_refreshes_the_current_pin_without_writing(
    bump_copy: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], annotated: bool, note_exists: bool,
) -> None:
    """Refresh uses the pinned release, never looks up npm latest, and preserves all bytes."""
    root, module = bump_copy
    note = root / module.CHANGE_NOTE
    if note_exists:
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_bytes(b"- Creator words.\r\n\r\n")
    else:
        note.unlink(missing_ok=True)
    source = root / CATALOG
    data = json.loads(source.read_text(encoding="utf-8"))
    data["registry"]["commit"] = COMMIT
    data["registry"]["url"] = (
        f"https://raw.githubusercontent.com/heygen-com/hyperframes/{COMMIT}/registry"
    )
    source.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    notice = root / "NOTICE"
    notice.write_bytes(re.sub(rb"(?m)(^at commit )[0-9a-f]{40}\b",
                             lambda match: match[1] + COMMIT.encode(), notice.read_bytes()))
    before = _snapshot(root, module)
    before_bytes = source.read_bytes()
    current = module.pinned_version(root)

    def run_git(args: list[str], cwd: Path) -> str:
        assert args == ["ls-remote", "https://github.com/heygen-com/hyperframes",
                        f"refs/tags/v{current}", f"refs/tags/v{current}^{{}}"]
        assert cwd == root
        return (f"{'a' * 40}\trefs/tags/v{current}\n" if annotated else "") + (
            f"{COMMIT}\trefs/tags/v{current}" + ("^{}" if annotated else "") + "\n"
        )

    monkeypatch.setattr(module, "run_git", run_git)
    calls = _stub_npm(module, monkeypatch, current)
    monkeypatch.setattr(sys, "argv", ["bump_hyperframes.py", "--registry"])
    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["registry"] == {
        "commit": COMMIT, "changed": [], "unchanged": list(data["items"]),
    }
    assert result["changed"] is False
    assert calls == [["ci", "--ignore-scripts", "--no-audit", "--no-fund"]]
    assert _snapshot(root, module) == before
    assert source.read_bytes() == before_bytes


def test_registry_refresh_preserves_helper_metadata(bump_copy, monkeypatch):
    root, module = bump_copy
    source = root / CATALOG
    data = json.loads(source.read_text())
    main = "compositions/components/sample-helper.html"
    helper = {"type": "component", "status": "helper", "main": main,
              "upstream_path": "registry/components/sample-helper/sample-helper.html",
              "files": {main: _digest("old synthetic helper")}}
    data["items"]["sample-helper"] = helper
    source.write_text(json.dumps(data, indent=2) + "\n")
    new = _next(module.pinned_version(root))
    _stub_npm(module, monkeypatch, new)
    result = module.bump(root, new)
    assert "sample-helper" in result["registry"]["changed"]
    assert json.loads(source.read_text())["items"]["sample-helper"] == {
        **helper, "files": {main: _digest(_content("sample-helper"))}}


@pytest.mark.parametrize(("unreleased", "note", "expected_unreleased", "expected_note"), [
    ("## Unreleased\n\n- Pin the renderer to HyperFrames 1.0.0 (notes).\n- Other.\n\n",
     None, "## Unreleased\n\n- Other.\n\n",
     "- Pin the renderer to HyperFrames 1.0.1 (notes).\n"),
    ("## Unreleased\n\n- Other.\n\n", None, "## Unreleased\n\n- Other.\n\n",
     "- Pin the renderer to HyperFrames 1.0.1.\n"),
    ("## Unreleased\n\n", None, "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1.\n"),
    ("## Unreleased\n\n- Pin the renderer to HyperFrames 1.0.0.\n\n",
     None, "## Unreleased\n\n", "- Pin the renderer to HyperFrames 1.0.1.\n"),
    ("## Unreleased\n\n- Pin the renderer to HyperFrames 0.9.9.\n\n",
     "- Pin the renderer to HyperFrames 1.0.0 (creator words).\n", "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1 (creator words).\n"),
    ("## Unreleased\n\n- Pin the renderer to HyperFrames 0.9.9\n\n",
     "- Pin the renderer to HyperFrames 1.0.0 (creator words).\n", "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1 (creator words).\n"),
    ("## Unreleased\n\n- Pin the renderer to HyperFrames 1.0.0 (caption timing).\n- Other.\n\n",
     "- Pin the renderer to HyperFrames 1.0.0 (registry).\n", "## Unreleased\n\n- Other.\n\n",
     "- Pin the renderer to HyperFrames 1.0.1.\n"
     "- Pin the renderer to HyperFrames 1.0.0 (registry).\n"
     "- Pin the renderer to HyperFrames 1.0.0 (caption timing).\n"),
    ("## Unreleased\n\n- Pin the renderer to HyperFrames 0.9.9 (older words).\n\n",
     "- Pin the renderer to HyperFrames 1.0.0 (registry).\n", "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1 (registry).\n"
     "- Pin the renderer to HyperFrames 0.9.9 (older words).\n"),
    ("## Unreleased\n\n- Pin the renderer to HyperFrames 0.9.9 (older words).\n\n",
     "- Pin the renderer to HyperFrames 1.0.0 (registry).", "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1 (registry).\n"
     "- Pin the renderer to HyperFrames 0.9.9 (older words).\n"),
    ("## Unreleased\n\n", "- Something else.\n", "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1.\n- Something else.\n"),
    ("", None, "", "- Pin the renderer to HyperFrames 1.0.1.\n"),
    ("## Unreleased\n\n- First.  \n- Pin the renderer to HyperFrames 0.9.9 (older).\n"
     "- Middle.\n- Pin the renderer to HyperFrames 1.0.0 (notes).\n- Last.\n\n",
     None, "## Unreleased\n\n- First.  \n- Middle.\n- Last.\n\n",
     "- Pin the renderer to HyperFrames 0.9.9 (older).\n"
     "- Pin the renderer to HyperFrames 1.0.1 (notes).\n"),
    ("## Unreleased\n\n", "- Other.\n- Pin the renderer to HyperFrames 1.0.0 (words).\n\n\n",
     "## Unreleased\n\n", "- Other.\n- Pin the renderer to HyperFrames 1.0.1 (words).\n"),
    ("## Unreleased\n\n", "", "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1.\n"),
    ("## Unreleased\n\n", "- Pin the renderer to HyperFrames 1.0.0 (one).\n"
     "- Pin the renderer to HyperFrames 1.0.0 (two).", "## Unreleased\n\n",
     "- Pin the renderer to HyperFrames 1.0.1.\n"
     "- Pin the renderer to HyperFrames 1.0.0 (one).\n"
     "- Pin the renderer to HyperFrames 1.0.0 (two).\n"),
])
def test_pin_note_cleans_only_unreleased_and_preserves_creator_words(
    monkeypatch: pytest.MonkeyPatch, unreleased: str, note: str | None,
    expected_unreleased: str, expected_note: str,
) -> None:
    """Legacy pins leave Unreleased; existing notes and released sections keep their words."""
    module = _load(ROOT, monkeypatch)
    released = "## 0.1.0 (2026-01-01)\n\n- Pin the renderer to HyperFrames 1.0.0 (old).\n"
    text = f"# Changelog\n\n{unreleased}{released}"
    assert module._pin_note(text, note, "1.0.0", "1.0.1") == (
        f"# Changelog\n\n{expected_unreleased}{released}", expected_note,
    )
