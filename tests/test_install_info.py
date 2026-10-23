"""Install identity tests using synthetic checkouts and read-only Git queries."""

from __future__ import annotations

import hashlib
import importlib
import struct
import zlib
from pathlib import Path
from types import ModuleType

import pytest

from yap_studio import __version__, tools
from yap_studio.errors import DependencyError, ProcessingError


def build_checkout(
    root: Path, version: str = "0.1.0", name: str = "yap-studio"
) -> tuple[Path, str]:
    """Write a synthetic one-commit fixture without running Git write commands."""
    package = root / "src" / "yap_studio"
    package.mkdir(parents=True)
    project = root / "pyproject.toml"
    project.write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8")
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (git_dir / "config").write_text(
        "[core]\nrepositoryformatversion = 0\nbare = false\n", encoding="ascii"
    )

    def object_id(kind: str, content: bytes) -> bytes:
        """Store a loose object using Git's public object format."""
        data = f"{kind} {len(content)}\0".encode("ascii") + content
        digest = hashlib.sha1(data).digest()
        target = git_dir / "objects" / digest.hex()[:2] / digest.hex()[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zlib.compress(data))
        return digest

    blob = object_id("blob", project.read_bytes())
    tree = object_id("tree", b"100644 pyproject.toml\0" + blob)
    commit = object_id(
        "commit",
        f"tree {tree.hex()}\n"
        "author Fixture <fixture@example.invalid> 0 +0000\n"
        "committer Fixture <fixture@example.invalid> 0 +0000\n\n"
        "Synthetic fixture\n".encode("ascii"),
    ).hex()
    (git_dir / "refs" / "heads" / "main").write_text(commit + "\n", encoding="ascii")
    facts = project.stat()
    filename = b"pyproject.toml"
    entry = struct.pack(
        "!10I20sH",
        int(facts.st_ctime), facts.st_ctime_ns % 1_000_000_000,
        int(facts.st_mtime), facts.st_mtime_ns % 1_000_000_000,
        facts.st_dev & 0xFFFFFFFF, facts.st_ino & 0xFFFFFFFF,
        0o100644, facts.st_uid, facts.st_gid, facts.st_size, blob, len(filename),
    ) + filename + b"\0"
    entry += b"\0" * (-len(entry) % 8)
    index = struct.pack("!4sII", b"DIRC", 2, 1) + entry
    (git_dir / "index").write_bytes(index + hashlib.sha1(index).digest())
    return package, commit[:12]


@pytest.fixture
def install_info() -> ModuleType:
    """Load the install helpers after test collection so absence is a test failure."""
    return importlib.import_module("yap_studio.install_info")


def test_source_checkout_and_commit(tmp_path: Path, install_info: ModuleType) -> None:
    package, commit = build_checkout(tmp_path)
    assert install_info.source_checkout(package) == tmp_path.resolve()
    assert install_info.source_commit(tmp_path) == (commit, False)
    (tmp_path / "untracked.txt").write_text("synthetic\n", encoding="utf-8")
    assert install_info.source_commit(tmp_path) == (commit, False)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "yap-studio"\nversion = "0.2.0"\n', encoding="utf-8"
    )
    assert install_info.source_commit(tmp_path) == (commit, True)


def test_source_checkout_rejects_wheel_inside_work_tree(
    tmp_path: Path, install_info: ModuleType
) -> None:
    """A wheel nested in a checkout does not run that checkout's source."""
    package, _ = build_checkout(tmp_path)
    wheel_package = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "yap_studio"
    wheel_package.mkdir(parents=True)
    for directory in (package, wheel_package):
        (directory / "__init__.py").write_text('"""Synthetic package."""\n', encoding="utf-8")
    assert install_info.source_checkout(package) == tmp_path.resolve()
    assert install_info.source_checkout(wheel_package) is None


def test_source_checkout_without_git(tmp_path: Path, install_info: ModuleType) -> None:
    package = tmp_path / "src" / "yap_studio"
    package.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "yap-studio"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    assert install_info.source_checkout(package) is None
    assert install_info.source_commit(tmp_path) is None


def test_source_checkout_rejects_other_project(tmp_path: Path, install_info: ModuleType) -> None:
    package, _ = build_checkout(tmp_path, name="synthetic-package")
    assert install_info.source_checkout(package) is None


def test_source_checkout_uses_package_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_info: ModuleType
) -> None:
    package, _ = build_checkout(tmp_path / "checkout")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(install_info, "__file__", str(package / "install_info.py"))
    assert install_info.source_checkout() == package.parents[1].resolve()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("0.1.0", "0.1.1", -1),
        ("0.1.1", "0.2.0", -1),
        ("0.2.0", "0.10.0", -1),
        ("0.10.2", "0.2.0", 1),
        ("0.1.0", "0.1.0", 0),
        ("0.1", "0.1.0", 0),
        ("unknown", "0.1.0", None),
        ("0.1.0", "0.2.0rc1", None),
        ("0+unknown", "0.1.0", None),
        ("", "0.1.0", None),
        ("0..1", "0.1.0", None),
        ("v0.1.0", "0.1.0", None),
        ("0.1.0\n", "0.1.0", None),
    ],
)
def test_compare_versions(
    install_info: ModuleType, left: str, right: str, expected: int | None
) -> None:
    assert install_info.compare_versions(left, right) == expected


def test_checkout_version_finds_nearest_matching_project(
    tmp_path: Path, install_info: ModuleType
) -> None:
    package, _ = build_checkout(tmp_path / "outer", version="0.2.0")
    inner = package / "nested"
    inner.mkdir()
    project = inner / "pyproject.toml"
    project.write_text(
        '[project]\nname = "synthetic-package"\nversion = "9.0.0"\n', encoding="utf-8"
    )
    start = inner / "child"
    start.mkdir()
    assert install_info.checkout_version(start) == (package.parents[1].resolve(), "0.2.0")
    project.write_text(
        '[project]\nname = "yap-studio"\nversion = "0.3.0"\n', encoding="utf-8"
    )
    assert install_info.checkout_version(start) == (inner.resolve(), "0.3.0")


@pytest.mark.parametrize(
    "content",
    ["invalid = [", '[project]\nname = "synthetic-package"', 'project = "yap-studio"',
     '[project]\nname = "yap-studio"\nversion = 1'],
)
def test_checkout_version_unknown_metadata(
    tmp_path: Path, install_info: ModuleType, content: str
) -> None:
    (tmp_path / "pyproject.toml").write_text(content, encoding="utf-8")
    assert install_info.checkout_version(tmp_path) is None


def test_checkout_version_outside_project(tmp_path: Path, install_info: ModuleType) -> None:
    assert install_info.checkout_version(tmp_path) is None


@pytest.mark.parametrize("modified", [False, True])
def test_version_text_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_info: ModuleType, modified: bool
) -> None:
    _, commit = build_checkout(tmp_path, version=__version__)
    if modified:
        (tmp_path / "pyproject.toml").write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(install_info, "source_checkout", lambda: tmp_path)
    suffix = ", modified" if modified else ""
    assert install_info.version_text() == f"yap-studio {__version__} (source {commit}{suffix})"


@pytest.mark.parametrize("known_commit", [False, True])
def test_version_text_uses_source_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_info: ModuleType, known_commit: bool
) -> None:
    """Source metadata stays paired with its commit even with an older distribution."""
    package, commit = build_checkout(tmp_path, version="0.1.2")
    monkeypatch.setattr(install_info, "__file__", str(package / "install_info.py"))
    monkeypatch.setattr(install_info, "__version__", "0.1.0")
    if not known_commit:
        monkeypatch.setattr(install_info, "source_commit", lambda root: None)
    identity = commit if known_commit else "unknown"
    assert install_info.version_text() == f"yap-studio 0.1.2 (source {identity})"


@pytest.mark.parametrize(
    "content",
    [None, "invalid = [", '[project]\nname = "yap-studio"',
     '[project]\nname = "yap-studio"\nversion = 1'],
)
def test_version_text_falls_back_when_source_metadata_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_info: ModuleType, content: str | None
) -> None:
    """An identified checkout may lose readable metadata before its version is read."""
    _, commit = build_checkout(tmp_path, version="0.1.2")
    project = tmp_path / "pyproject.toml"
    if content is None:
        project.unlink()
    else:
        project.write_text(content, encoding="utf-8")
    monkeypatch.setattr(install_info, "source_checkout", lambda: tmp_path)
    monkeypatch.setattr(install_info, "__version__", "0.1.0")
    assert install_info.version_text() == f"yap-studio 0.1.0 (source {commit}, modified)"


def test_version_text_wheel(monkeypatch: pytest.MonkeyPatch, install_info: ModuleType) -> None:
    monkeypatch.setattr(install_info, "source_checkout", lambda: None)
    assert install_info.version_text() == f"yap-studio {__version__}"


@pytest.mark.parametrize(
    "failure",
    [
        DependencyError("git missing", suggested_action="Install Git."),
        ProcessingError("git", -1, "timed out", code="git_timeout"),
        ProcessingError("git", 1, "not a repository", code="git_exit_1"),
        OSError("unavailable"),
    ],
)
def test_git_failures_are_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_info: ModuleType,
    failure: Exception, capsys: pytest.CaptureFixture[str],
) -> None:
    package, _ = build_checkout(tmp_path, version=__version__)

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(tools, "run_tool", fail)
    assert install_info.source_checkout(package) is None
    assert install_info.source_commit(tmp_path) is None
    monkeypatch.setattr(install_info, "source_checkout", lambda: tmp_path)
    assert install_info.version_text() == f"yap-studio {__version__} (source unknown)"
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_metadata_read_failure_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_info: ModuleType
) -> None:
    package, _ = build_checkout(tmp_path)

    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "open", denied)
    assert install_info.source_checkout(package) is None
    assert install_info.checkout_version(tmp_path) is None
