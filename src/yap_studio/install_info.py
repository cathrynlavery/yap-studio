"""Read the running package's install identity without changing its checkout."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import tomllib
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path

from yap_studio import __version__, tools
from yap_studio.errors import YapError

_GIT_TIMEOUT_S = 2.0


def build_id(distribution: importlib.metadata.Distribution | None = None) -> str | None:
    """Identify the installed package files from their RECORD hashes."""
    try:
        dist = distribution or importlib.metadata.distribution("yap-studio")
        files = dist.files
        if files is None:
            return None
        lines = []
        for file in files:
            path = str(file)
            if (path.startswith("yap_studio/") and "__pycache__" not in path
                    and not path.endswith(".pyc") and file.hash is not None):
                lines.append(f"{path},{file.hash.mode}={file.hash.value}")
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return None
    if not lines:
        return None
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def _git_text(root: Path, args: Sequence[str]) -> str | None:
    """Read Git output with a short deadline and no optional index writes."""
    try:
        result = tools.run_tool(
            "git", ["--no-optional-locks", "-C", str(root), *args], timeout_s=_GIT_TIMEOUT_S
        )
    except (YapError, OSError, ValueError):
        return None
    return result.stdout.strip()


def _project(root: Path) -> dict[str, object] | None:
    """Read matching project metadata, treating unreadable TOML as unknown."""
    try:
        with (root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream).get("project")
    except (OSError, ValueError):
        return None
    if not isinstance(project, dict) or project.get("name") != "yap-studio":
        return None
    return project


def source_checkout(package_dir: Path | None = None) -> Path | None:
    """Return the Git root only when the imported package is its source directory."""
    try:
        package = (package_dir if package_dir is not None else Path(__file__).parent).resolve()
        text = _git_text(package, ["rev-parse", "--show-toplevel"])
        if not text:
            return None
        root = Path(text).resolve()
        if package == (root / "src" / "yap_studio").resolve() and _project(root) is not None:
            return root
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def source_commit(root: Path) -> tuple[str, bool] | None:
    """Return a 12-character commit and tracked-file modification status."""
    commit = _git_text(root, ["rev-parse", "--short=12", "HEAD"])
    if commit is None or re.fullmatch(r"[0-9a-f]{12,64}", commit) is None:
        return None
    status = _git_text(root, ["status", "--porcelain", "--untracked-files=no"])
    if status is None:
        return None
    return commit[:12], bool(status)


def checkout_version(start: Path) -> tuple[Path, str] | None:
    """Find the nearest matching project and its version while walking upward."""
    try:
        start = start.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for root in (start, *start.parents):
        project = _project(root)
        if project is not None:
            version = project.get("version")
            return (root, version) if isinstance(version, str) else None
    return None


def workspace_agent(start: Path) -> str:
    """Read the setup app even when Home is stopped; old workspaces use Claude Code."""
    for root in (start, *start.parents):
        saved = root / ".yap-studio/setup.json"
        if saved.is_file():
            try:
                state = json.loads(saved.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return "claude-code"
            return ("codex" if isinstance(state, dict) and state.get("agent") == "codex"
                    else "claude-code")
        if (root / ".yap-studio-version").is_file():
            break
    return "claude-code"


def workspace_version(start: Path) -> tuple[Path, str] | None:
    """Find a packaged creator workspace while walking up from a project."""
    try:
        start = start.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for root in (start, *start.parents):
        marker = root / ".yap-studio-version"
        if not marker.is_file() or not (root / ".claude/skills/edit-video/SKILL.md").is_file():
            continue
        try:
            version = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        return (root, version) if version else None
    return None


def compare_versions(left: str, right: str) -> int | None:
    """Compare numeric release parts, returning -1, 0, 1, or unknown (None)."""
    if any(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value) is None for value in (left, right)):
        return None
    try:
        left_parts = tuple(int(part) for part in left.split("."))
        right_parts = tuple(int(part) for part in right.split("."))
    except ValueError:
        return None
    for first, second in zip_longest(left_parts, right_parts, fillvalue=0):
        if first != second:
            return -1 if first < second else 1
    return 0


def running_version(root: Path | None) -> str:
    """Read an identified source checkout's version, falling back to distribution metadata."""
    if root is not None:
        project = _project(root)
        if project is not None:
            version = project.get("version")
            if isinstance(version, str):
                return version
    return __version__


def version_text() -> str:
    """Describe the running version, including source identity when available."""
    root = source_checkout()
    text = f"yap-studio {running_version(root)}"
    if root is None:
        return text
    commit = source_commit(root)
    if commit is None:
        return f"{text} (source unknown)"
    short, modified = commit
    suffix = ", modified" if modified else ""
    return f"{text} (source {short}{suffix})"
