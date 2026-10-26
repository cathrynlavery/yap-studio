"""Bump release metadata and prepare a changelog section without writing to Git."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/bump_version.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/bump_version.py"
    )

import argparse
import json
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

from yap_studio.errors import ValidationError, YapError

ROOT = Path(__file__).resolve().parents[1]
CHANGE_KINDS = {
    "feature": "Features",
    "fix": "Fixes",
    "performance": "Performance",
    "dependencies": "Dependencies",
    "docs": "Docs",
    "tests": "Tests",
}
MANIFESTS = {
    ".claude-plugin/marketplace.json": 2,
    "plugins/yap-studio/.claude-plugin/plugin.json": 1,
    "plugins/yap-studio/.codex-plugin/plugin.json": 1,
    "mcpb/manifest.json": 1,
}


class ReleaseError(YapError):
    """Report a release preparation failure with a recovery action."""

    def __init__(self, message: str, suggested_action: str) -> None:
        super().__init__(
            message, error_type="release_error", code="release_failed",
            suggested_action=suggested_action,
        )


def version_parts(value: str) -> tuple[int, int, int]:
    """Validate a canonical X.Y.Z release and return its numeric components."""
    if re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value) is None:
        raise ValidationError("version", "use X.Y.Z with three non-negative integers")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def project_version(text: str) -> str:
    """Read and validate the project's release version from TOML."""
    value = tomllib.loads(text).get("project", {}).get("version")
    if not isinstance(value, str):
        raise ValidationError("pyproject.toml", "[project].version must be a string")
    version_parts(value)
    return value


def changelog_section(text: str, version: str) -> str | None:
    """Return only the dated section for a release, including its heading."""
    heading = re.search(
        rf"(?m)^## {re.escape(version)} \([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\)[ \t]*$",
        text,
    )
    if heading is None:
        return None
    following = re.search(r"(?m)^## ", text[heading.end():])
    end = len(text) if following is None else heading.end() + following.start()
    return text[heading.start():end].strip() + "\n"


def _project_update(text: str, current: str, new: str) -> str:
    """Replace the project version while preserving other TOML and comments."""
    section = re.search(r"(?ms)^\[project\][^\n]*\n(?P<body>.*?)(?=^\[|\Z)", text)
    if section is None:
        raise ValidationError("pyproject.toml", "missing [project] table")
    body, count = re.subn(
        rf"(?m)^(version[ \t]*=[ \t]*)([\"']){re.escape(current)}\2",
        rf"\g<1>\g<2>{new}\g<2>", section["body"],
    )
    if count != 1:
        raise ValidationError("pyproject.toml", "expected one project version assignment")
    return text[:section.start("body")] + body + text[section.end("body"):]


def _manifest_update(text: str, current: str, new: str, expected: int) -> str:
    """Replace only version values, preserving all other JSON text."""
    json.loads(text)
    pattern = r'("version"\s*:\s*)"([^"\\]*)"'
    versions = list(re.finditer(pattern, text))
    if len(versions) != expected or any(match[2] != current for match in versions):
        raise ValidationError("manifest", "version fields must match pyproject.toml")
    return re.sub(pattern, lambda match: f'{match[1]}"{new}"', text)


def change_notes(root: Path) -> list[tuple[Path, str, str]]:
    """Read and validate change files in filename order."""
    folder = root / "changes"
    if not folder.exists():
        return []
    notes = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or path.name == "README.md":
            continue
        parameter = f"changes/{path.name}"
        if not path.is_file():
            raise ValidationError(parameter, "replace this entry with a change file")
        name = re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*\.([a-z]+)\.md", path.name)
        if name is None or name[1] not in CHANGE_KINDS:
            raise ValidationError(
                parameter, "name it <short-slug>.<kind>.md with kind feature, fix, "
                "performance, dependencies, docs or tests",
            )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValidationError(parameter, "save the change file as UTF-8") from exc
        bullets = [line.rstrip() for line in lines if line.strip()]
        if not bullets or any(re.fullmatch(r"- \S.*", line) is None for line in bullets):
            raise ValidationError(parameter, "write one or more lines starting with '- '")
        notes.append((path, name[1], "\n".join(bullets)))
    return notes


def _changelog_release(
    text: str, new: str, release_date: str, gathered: list[tuple[Path, str, str]],
) -> str:
    """Move Unreleased notes into a dated section below an empty Unreleased heading."""
    unreleased = re.search(r"(?m)^## Unreleased[ \t]*$", text)
    if unreleased is None:
        first_section = re.search(r"(?m)^## ", text)
        start = end = len(text) if first_section is None else first_section.start()
        notes = ""
    else:
        start = unreleased.start()
        following = re.search(r"(?m)^## ", text[unreleased.end():])
        end = len(text) if following is None else unreleased.end() + following.start()
        notes = text[unreleased.end():end].strip()
    sections = [notes] if notes else []
    for kind, heading in CHANGE_KINDS.items():
        bullets = [bullets for _, note_kind, bullets in gathered if note_kind == kind]
        if bullets:
            sections.append(f"### {heading}\n\n" + "\n".join(bullets))
    notes = "\n\n".join(sections)
    prefix = text[:start].rstrip() + "\n\n"
    section = f"## Unreleased\n\n## {new} ({release_date})\n\n"
    if notes:
        section += notes + "\n\n"
    rest = text[end:]
    return prefix + (section if rest else section.rstrip() + "\n") + rest


def bump(root: Path, new: str, release_date: str) -> None:
    """Update all release versions and refresh the lockfile, rolling back on failure."""
    new_parts = version_parts(new)
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", release_date) is None:
        raise ValidationError("date", "use YYYY-MM-DD")
    try:
        date.fromisoformat(release_date)
    except ValueError as exc:
        raise ValidationError("date", "use a valid calendar date") from exc
    original = {"pyproject.toml": (root / "pyproject.toml").read_text(encoding="utf-8")}
    current = project_version(original["pyproject.toml"])
    if new_parts <= version_parts(current):
        raise ValidationError("version", f"{new} must be greater than {current}")
    updated = {"pyproject.toml": _project_update(original["pyproject.toml"], current, new)}
    for relative, expected in MANIFESTS.items():
        original[relative] = (root / relative).read_text(encoding="utf-8")
        updated[relative] = _manifest_update(original[relative], current, new, expected)
    changelog = root / "CHANGELOG.md"
    original["CHANGELOG.md"] = changelog.read_text(encoding="utf-8")
    gathered = change_notes(root)
    if changelog_section(original["CHANGELOG.md"], new) is None:
        updated["CHANGELOG.md"] = _changelog_release(
            original["CHANGELOG.md"], new, release_date, gathered,
        )
    elif gathered:
        raise ValidationError(
            "CHANGELOG.md", f"remove the section for {new}, or move the change notes into it "
            "by hand and delete their files, then retry",
        )
    original_notes = {path: path.read_bytes() for path, _, _ in gathered}
    lock = root / "uv.lock"
    original_lock = lock.read_bytes() if lock.exists() else None
    try:
        for relative, text in updated.items():
            (root / relative).write_text(text, encoding="utf-8")
        for path in original_notes:
            path.unlink()
        subprocess.run(
            ["uv", "lock"], cwd=root, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=120, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        for relative, text in original.items():
            (root / relative).write_text(text, encoding="utf-8")
        for path, content in original_notes.items():
            path.write_bytes(content)
        if original_lock is None:
            lock.unlink(missing_ok=True)
        else:
            lock.write_bytes(original_lock)
        raise ReleaseError(
            f"Version bump failed ({exc}); release files were restored.",
            "Check that uv is installed and uv lock succeeds, then retry the bump.",
        ) from exc


def main() -> int:
    """Apply a requested bump or print a plain actionable error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_version", metavar="NEW_VERSION")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    args = parser.parse_args()
    try:
        bump(ROOT, args.new_version, args.date)
    except (YapError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        action = exc.suggested_action if isinstance(exc, YapError) else "Check the release files."
        print(action, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
