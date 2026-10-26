"""Release metadata and gates tested with temporary copies and synthetic Git history."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from test_install_info import build_checkout

ROOT = Path(__file__).parents[1]
MANIFESTS = (
    ".claude-plugin/marketplace.json",
    "plugins/yap-studio/.claude-plugin/plugin.json",
    "plugins/yap-studio/.codex-plugin/plugin.json",
    "mcpb/manifest.json",
)
RELEASE_FILES = ("pyproject.toml", "uv.lock", "CHANGELOG.md", *MANIFESTS)


def _copy_scripts(root: Path) -> None:
    """Put each release script at its normal relative location in a fixture."""
    (root / "scripts").mkdir(exist_ok=True)
    for filename in ("bump_version.py", "check_version_bump.py"):
        shutil.copyfile(ROOT / "scripts" / filename, root / "scripts" / filename)


@pytest.fixture
def release_copy(tmp_path: Path) -> Path:
    """Copy release inputs so running the bump cannot change the real checkout."""
    for relative in RELEASE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    _copy_scripts(tmp_path)
    return tmp_path


@pytest.fixture
def bump_script(release_copy: Path) -> ModuleType:
    """Load the copied script so lock invocation can be tested without downloads."""
    spec = importlib.util.spec_from_file_location(
        "release_bump_fixture", release_copy / "scripts" / "bump_version.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(root: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a fixture script with closed stdin and no dependency downloads."""
    env = dict(os.environ)
    env.update(UV_OFFLINE="true", UV_PYTHON=sys.executable)
    return subprocess.run(
        [sys.executable, str(root / "scripts" / script), *args], cwd=root,
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=150, check=False,
    )


def _project(root: Path) -> dict[str, object]:
    """Read project metadata from a release fixture."""
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _next_version(root: Path) -> str:
    """Choose the next patch without coupling the test to a particular release."""
    major, minor, patch = str(_project(root)["version"]).split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def test_bump_updates_all_versions_and_preserves_other_fields(
    release_copy: Path, bump_script: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bump updates release metadata and requests a lock refresh in the same root."""
    root = release_copy
    new = _next_version(root)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    manifests = {
        relative: json.loads((root / relative).read_text(encoding="utf-8"))
        for relative in MANIFESTS
    }
    lock_command = Mock()
    monkeypatch.setattr(bump_script.subprocess, "run", lock_command)
    bump_script.bump(root, new, "2026-10-06")
    assert lock_command.call_count == 1
    assert lock_command.call_args.args == (["uv", "lock"],)
    assert lock_command.call_args.kwargs["cwd"] == root
    assert lock_command.call_args.kwargs["stdin"] == subprocess.DEVNULL
    assert lock_command.call_args.kwargs["timeout"] > 0
    assert lock_command.call_args.kwargs["check"] is True
    project["project"]["version"] = new
    assert tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")) == project
    for relative, expected in manifests.items():
        if relative == ".claude-plugin/marketplace.json":
            expected["metadata"]["version"] = new
            expected["plugins"][0]["version"] = new
        else:
            expected["version"] = new
        assert json.loads((root / relative).read_text(encoding="utf-8")) == expected
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.replace(f"## {new} (2026-10-06)\n\n", "", 1) == changelog
    assert re.findall(r"(?m)^## (\S+)", text)[:2] == ["Unreleased", new]


@pytest.mark.parametrize("value", ["0.0.0", "equal", "1.2", "v1.2.3", "1.2.3rc1", "01.2.3"])
def test_bump_refuses_invalid_or_non_increasing_version(
    release_copy: Path, value: str,
) -> None:
    """Bad versions fail before any release file is changed."""
    if value == "equal":
        value = str(_project(release_copy)["version"])
    before = {relative: (release_copy / relative).read_bytes() for relative in RELEASE_FILES}
    result = _run(release_copy, "bump_version.py", value)
    assert result.returncode == 1
    assert "version" in result.stderr
    assert "Traceback" not in result.stderr
    after = {relative: (release_copy / relative).read_bytes() for relative in RELEASE_FILES}
    assert after == before


@pytest.mark.parametrize("value", ["2026-02-30", "20261006"])
def test_bump_refuses_invalid_date(release_copy: Path, value: str) -> None:
    """Malformed dates cannot create a changelog entry or a partial bump."""
    before = {relative: (release_copy / relative).read_bytes() for relative in RELEASE_FILES}
    result = _run(release_copy, "bump_version.py", _next_version(release_copy), "--date", value)
    assert result.returncode == 1
    assert "date" in result.stderr
    after = {relative: (release_copy / relative).read_bytes() for relative in RELEASE_FILES}
    assert after == before


def test_bump_preserves_existing_changelog_section(
    release_copy: Path, bump_script: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared release notes survive the bump without a duplicate heading."""
    new = _next_version(release_copy)
    changelog = release_copy / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8").replace(
        "# Changelog\n", f"# Changelog\n\n## {new} (2026-10-06)\n\n- Synthetic change.\n", 1,
    )
    changelog.write_text(text, encoding="utf-8")
    monkeypatch.setattr(bump_script.subprocess, "run", Mock())
    bump_script.bump(release_copy, new, "2026-10-07")
    assert changelog.read_text(encoding="utf-8") == text


def test_bump_moves_unreleased_notes_into_dated_section(
    release_copy: Path, bump_script: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreleased notes become the release's dated notes, and the check accepts the bump."""
    current = str(_project(release_copy)["version"])
    project_text = (release_copy / "pyproject.toml").read_text(encoding="utf-8")
    build_checkout(release_copy, version=current)
    (release_copy / "pyproject.toml").write_text(project_text, encoding="utf-8")
    changelog = release_copy / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## Unreleased\n\n- First synthetic note.\n- Second synthetic note.\n\n"
        f"## {current} (2026-10-01)\n\n- Older change.\n",
        encoding="utf-8",
    )
    new = _next_version(release_copy)
    with monkeypatch.context() as resolver:
        resolver.setattr(bump_script.subprocess, "run", Mock())
        bump_script.bump(release_copy, new, "2026-10-06")
    text = changelog.read_text(encoding="utf-8")
    assert text == (
        f"# Changelog\n\n## Unreleased\n\n## {new} (2026-10-06)\n\n"
        "- First synthetic note.\n- Second synthetic note.\n\n"
        f"## {current} (2026-10-01)\n\n- Older change.\n"
    )
    section = bump_script.changelog_section(text, new)
    assert section is not None
    assert "- First synthetic note." in section and "- Second synthetic note." in section
    unreleased = text.split("## Unreleased\n", 1)[1].split("\n## ", 1)[0]
    assert unreleased.strip() == ""
    result = _run(release_copy, "check_version_bump.py", "--base", "main")
    assert result.returncode == 0, result.stderr


def test_bump_restores_release_files_when_lock_fails(
    release_copy: Path, bump_script: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed resolver restores both metadata and any partially written lock."""
    before = {relative: (release_copy / relative).read_bytes() for relative in RELEASE_FILES}

    def fail_lock(*args: object, **kwargs: object) -> None:
        (release_copy / "uv.lock").write_text("partial lock\n", encoding="utf-8")
        raise subprocess.CalledProcessError(1, ["uv", "lock"])

    monkeypatch.setattr(bump_script.subprocess, "run", fail_lock)
    with pytest.raises(bump_script.ReleaseError, match="release files were restored"):
        bump_script.bump(release_copy, _next_version(release_copy), "2026-10-06")
    after = {relative: (release_copy / relative).read_bytes() for relative in RELEASE_FILES}
    assert after == before


def test_bump_gathers_change_files_by_kind(
    release_copy: Path, bump_script: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Change files follow legacy notes in kind order and satisfy the release gate."""
    root = release_copy
    current = str(_project(root)["version"])
    project_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    build_checkout(root, version=current)
    (root / "pyproject.toml").write_text(project_text, encoding="utf-8")
    changelog = root / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## Unreleased\n\n- Legacy note.\n\n"
        f"## {current} (2026-10-01)\n\n- Older change.\n", encoding="utf-8",
    )
    folder = root / "changes"
    folder.mkdir()
    files = {
        "new.feature.md": "\n- New feature.  \n\n- Another feature.\t\n",
        "z-last.fix.md": "- Second fix.\n",
        "a-first.fix.md": "- First fix.\n",
        "update.dependencies.md": "- Updated dependency.\n",
        "coverage.tests.md": "- Added tests.\n",
        "README.md": "Change file instructions.\n",
        ".DS_Store": "ignored\n",
    }
    for filename, content in files.items():
        (folder / filename).write_text(content, encoding="utf-8")
    assert bump_script.change_notes(root) == [
        (folder / "a-first.fix.md", "fix", "- First fix."),
        (folder / "coverage.tests.md", "tests", "- Added tests."),
        (folder / "new.feature.md", "feature", "- New feature.\n- Another feature."),
        (folder / "update.dependencies.md", "dependencies", "- Updated dependency."),
        (folder / "z-last.fix.md", "fix", "- Second fix."),
    ]
    new = _next_version(root)
    with monkeypatch.context() as resolver:
        resolver.setattr(bump_script.subprocess, "run", Mock())
        bump_script.bump(root, new, "2026-10-06")
    assert changelog.read_text(encoding="utf-8") == (
        f"# Changelog\n\n## Unreleased\n\n## {new} (2026-10-06)\n\n- Legacy note.\n\n"
        "### Features\n\n- New feature.\n- Another feature.\n\n"
        "### Fixes\n\n- First fix.\n- Second fix.\n\n"
        "### Dependencies\n\n- Updated dependency.\n\n### Tests\n\n- Added tests.\n\n"
        f"## {current} (2026-10-01)\n\n- Older change.\n"
    )
    assert sorted(path.name for path in folder.iterdir()) == [".DS_Store", "README.md"]
    for filename in ("README.md", ".DS_Store"):
        assert (folder / filename).read_bytes() == files[filename].encode("utf-8")
    result = _run(root, "check_version_bump.py", "--base", "main")
    assert result.returncode == 0, result.stderr


def test_bump_with_only_change_files(
    release_copy: Path, bump_script: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without legacy notes, the section starts with the first kind heading."""
    changelog = release_copy / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## Unreleased\n", encoding="utf-8")
    folder = release_copy / "changes"
    folder.mkdir()
    (folder / "speed.performance.md").write_text("- Faster captions.\n", encoding="utf-8")
    (folder / "guide.docs.md").write_text("- Updated guide.\n", encoding="utf-8")
    monkeypatch.setattr(bump_script.subprocess, "run", Mock())
    new = _next_version(release_copy)
    bump_script.bump(release_copy, new, "2026-10-06")
    assert changelog.read_text(encoding="utf-8") == (
        f"# Changelog\n\n## Unreleased\n\n## {new} (2026-10-06)\n\n"
        "### Performance\n\n- Faster captions.\n\n### Docs\n\n- Updated guide.\n"
    )
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize(("filename", "content"), [
    ("x.feat.md", b"- Note.\n"), ("x.md", b"- Note.\n"),
    ("Bad_Name.fix.md", b"- Note.\n"), ("x.fix.txt", b"- Note.\n"),
    ("x.fix.md", None), ("x.fix.md", b""), ("x.fix.md", b"Prose.\n"),
    ("x.fix.md", b"- Note.\nProse.\n"), ("x.fix.md", b"* Note.\n"),
    ("x.fix.md", b"  - Indented.\n"), ("x.fix.md", b"# Heading\n"),
    ("x.fix.md", b"- \t\n"), ("x.fix.md", b"- \xff\n"),
    ("x--y.fix.md", b"- Note.\n"),
])
def test_bump_refuses_bad_change_files(
    release_copy: Path, filename: str, content: bytes | None,
) -> None:
    """Invalid entries fail plainly before any release file or change file is changed."""
    folder = release_copy / "changes"
    folder.mkdir()
    (folder / "a-valid.fix.md").write_bytes(b"\r\n- Keep this note.  \r\n")
    bad = folder / filename
    if content is None:
        bad.mkdir()
    else:
        bad.write_bytes(content)
    paths = [release_copy / relative for relative in RELEASE_FILES] + list(folder.iterdir())
    before = {path: path.read_bytes() if path.is_file() else None for path in paths}
    result = _run(release_copy, "bump_version.py", _next_version(release_copy))
    assert result.returncode == 1
    assert f"changes/{filename}" in result.stderr
    assert "Traceback" not in result.stderr
    assert {path: path.read_bytes() if path.is_file() else None for path in paths} == before
    assert set(folder.iterdir()) == {bad, folder / "a-valid.fix.md"}


@pytest.mark.parametrize("has_lock", [True, False])
def test_bump_restores_change_files_when_lock_fails(
    release_copy: Path, bump_script: ModuleType, monkeypatch: pytest.MonkeyPatch, has_lock: bool,
) -> None:
    """A failed resolver recreates deleted notes with their original bytes."""
    folder = release_copy / "changes"
    folder.mkdir()
    for filename in ("first.fix.md", "second.tests.md"):
        (folder / filename).write_bytes("\r\n- Keep café.  \r\n\r\n".encode())
    if not has_lock:
        (release_copy / "uv.lock").unlink()
    paths = [release_copy / relative for relative in RELEASE_FILES] + list(folder.iterdir())
    before = {path: path.read_bytes() if path.exists() else None for path in paths}

    def fail_lock(*args: object, **kwargs: object) -> None:
        assert list(folder.iterdir()) == []
        (release_copy / "uv.lock").write_text("partial lock\n", encoding="utf-8")
        raise subprocess.CalledProcessError(1, ["uv", "lock"])

    monkeypatch.setattr(bump_script.subprocess, "run", fail_lock)
    with pytest.raises(bump_script.ReleaseError, match="release files were restored"):
        bump_script.bump(release_copy, _next_version(release_copy), "2026-10-06")
    assert {path: path.read_bytes() if path.exists() else None for path in paths} == before


def test_bump_refuses_existing_section_with_change_files(release_copy: Path) -> None:
    """Prepared sections cannot silently leave new change files out of a release."""
    new = _next_version(release_copy)
    (release_copy / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## Unreleased\n\n## {new} (2026-10-06)\n\n- Prepared note.\n",
        encoding="utf-8",
    )
    folder = release_copy / "changes"
    folder.mkdir()
    note = folder / "pending.fix.md"
    note.write_bytes(b"- Pending note.\r\n")
    paths = [release_copy / relative for relative in RELEASE_FILES] + [note]
    before = {path: path.read_bytes() for path in paths}
    result = _run(release_copy, "bump_version.py", new)
    assert result.returncode == 1
    assert "CHANGELOG.md" in result.stderr and "remove the section" in result.stderr
    assert "by hand" in result.stderr and "retry" in result.stderr
    assert "Traceback" not in result.stderr
    assert {path: path.read_bytes() for path in paths} == before


def test_change_notes_without_pending_files(
    release_copy: Path, bump_script: ModuleType,
) -> None:
    """Missing folders and ignored entries produce no pending notes."""
    assert bump_script.change_notes(release_copy) == []
    folder = release_copy / "changes"
    folder.mkdir()
    (folder / "README.md").write_text("Instructions.\n", encoding="utf-8")
    (folder / ".DS_Store").write_bytes(b"ignored")
    (folder / ".hidden").mkdir()
    assert bump_script.change_notes(release_copy) == []


def test_repository_change_notes_are_valid() -> None:
    """Every pending note in the real checkout follows the documented format."""
    spec = importlib.util.spec_from_file_location(
        "repository_release_bump", ROOT / "scripts" / "bump_version.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT / "changes" / "README.md").is_file()
    module.change_notes(ROOT)


@pytest.mark.parametrize(
    ("version", "changelog", "expected"),
    [
        ("0.1.1", "## 0.1.1 (2026-10-05)\n\n- Synthetic change.\n", 0),
        ("0.1.0", "## 0.1.0 (2026-10-05)\n\n- Synthetic change.\n", 1),
        ("0.0.9", "## 0.0.9 (2026-10-05)\n\n- Synthetic change.\n", 1),
        ("0.1.1", "## 0.1.0 (2026-10-01)\n\n- Older change.\n", 1),
        ("0.1.1", "## 0.1.1 (2026-10-05)\n\n## 0.1.0 (2026-10-01)\n- Older change.\n", 1),
        ("0.1.1", "## 0.1.1 (2026-10-05)\n\nProse without a bullet.\n", 1),
        ("0.1.1", "## 0.1.1 (2026-10-05)\n\n- \n", 1),
    ],
)
def test_check_version_bump(
    tmp_path: Path, version: str, changelog: str, expected: int,
) -> None:
    """Read a synthetic base commit without issuing any Git write commands."""
    build_checkout(tmp_path)
    _copy_scripts(tmp_path)
    project = tmp_path / "pyproject.toml"
    project.write_text(
        project.read_text(encoding="utf-8").replace('"0.1.0"', f'"{version}"'),
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    result = _run(tmp_path, "check_version_bump.py", "--base", "main")
    assert result.returncode == expected, result.stderr
    if expected:
        assert "greater than base" in result.stderr or "at least one bullet" in result.stderr
        assert "Traceback" not in result.stderr


def test_check_compares_versions_numerically(tmp_path: Path) -> None:
    """A two-digit minor release sorts after a one-digit minor release."""
    build_checkout(tmp_path, version="0.2.0")
    _copy_scripts(tmp_path)
    project = tmp_path / "pyproject.toml"
    project.write_text(
        project.read_text(encoding="utf-8").replace('"0.2.0"', '"0.10.0"'), encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "## 0.10.0 (2026-10-05)\n\n- Synthetic change.\n", encoding="utf-8",
    )
    result = _run(tmp_path, "check_version_bump.py", "--base", "main")
    assert result.returncode == 0, result.stderr


def test_check_reports_unavailable_base(tmp_path: Path) -> None:
    """An unavailable base ref fails plainly instead of accepting an unchecked bump."""
    build_checkout(tmp_path)
    _copy_scripts(tmp_path)
    result = _run(tmp_path, "check_version_bump.py", "--base", "missing-ref")
    assert result.returncode == 1
    assert "Cannot read pyproject.toml at base missing-ref" in result.stderr
    assert "Traceback" not in result.stderr


def test_current_changelog_has_version_section() -> None:
    """The working release has dated notes containing at least one bullet."""
    version = str(_project(ROOT)["version"])
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(
        rf"(?m)^## {re.escape(version)} \([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\)$", text,
    )
    assert heading is not None
    section = text[heading.end():].split("\n## ", 1)[0]
    assert re.search(r"(?m)^- \S", section)


def test_current_lock_has_project_version() -> None:
    """Locked installs use the same release version as every manifest."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    project = next(package for package in lock["package"] if package["name"] == "yap-studio")
    assert project["version"] == _project(ROOT)["version"]


def test_release_uploads_the_built_workspace_zip() -> None:
    """The draft release carries the workspace zip next to the wheel."""
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    build = workflow.index("scripts/build_workspace.py")
    upload = workflow.index("gh release create")
    assert build < upload
    assert "dist/yap-studio-workspace-*.zip" in workflow[upload:]
    assert "dist/*.whl" in workflow[upload:]
