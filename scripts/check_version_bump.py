"""Require a higher project version and a changelog bullet relative to a Git ref."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/check_version_bump.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/check_version_bump.py"
    )

import argparse
import re
import subprocess
from pathlib import Path

from bump_version import ReleaseError, changelog_section, project_version, version_parts

from yap_studio.errors import ValidationError, YapError

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path, base: str) -> None:
    """Compare working metadata with a base ref using only read-only Git queries."""
    current = project_version((root / "pyproject.toml").read_text(encoding="utf-8"))
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "show", "--end-of-options", f"{base}:pyproject.toml"],
            cwd=root, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseError(
            f"Cannot read pyproject.toml at base {base}.",
            "Use an available base ref and check that Git is installed.",
        ) from exc
    previous = project_version(result.stdout)
    if version_parts(current) <= version_parts(previous):
        raise ValidationError("version", f"{current} must be greater than base {previous}")
    section = changelog_section((root / "CHANGELOG.md").read_text(encoding="utf-8"), current)
    if section is None or re.search(r"(?m)^[ \t]*[-*+] [^\s]", section) is None:
        raise ValidationError(
            "CHANGELOG.md", f"add a dated {current} section with at least one bullet",
        )


def main() -> int:
    """Exit successfully only when the pull request records a release change."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, metavar="REF")
    args = parser.parse_args()
    try:
        check(ROOT, args.base)
    except (YapError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
