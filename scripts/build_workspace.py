"""Build a creator workspace from tracked files and a wheel built from tracked source."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/build_workspace.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/build_workspace.py"
    )

import argparse
import os
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_FILES = {
    "workspace/Start here.html": "Start here.html",
    "workspace/AGENTS.md": "AGENTS.md",
    "workspace/CLAUDE.md": "CLAUDE.md",
    "workspace/inbox/README.md": "inbox/README.md",
    "workspace/brand/fonts/README.md": "brand/fonts/README.md",
    "workspace/.claude/skills/edit-video/SKILL.md": ".claude/skills/edit-video/SKILL.md",
    "workspace/.claude/skills/voiceover-video/SKILL.md": ".claude/skills/voiceover-video/SKILL.md",
    "workspace/.claude/commands/edit.md": ".claude/commands/edit.md",
    "workspace/.claude/settings.json": ".claude/settings.json",
    "workspace/.codex/config.toml": ".codex/config.toml",
    "workspace/.codex/rules/yap-studio.rules": ".codex/rules/yap-studio.rules",
    "workspace/.cursor/permissions.json": ".cursor/permissions.json",
    "workspace/.cursor/cli.json": ".cursor/cli.json",
    "workspace/setup.py": "setup.py",
    "workspace/new_project.py": "new_project.py",
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
}
# Regular files in the zip keep discovery independent of unzip's symlink support.
WORKSPACE_COPIES = {
    ".agents/skills/edit-video/SKILL.md": "workspace/.claude/skills/edit-video/SKILL.md",
    ".agents/skills/voiceover-video/SKILL.md": "workspace/.claude/skills/voiceover-video/SKILL.md",
    ".agents/skills/edit/SKILL.md": "workspace/.claude/commands/edit.md",
}
BUILD_FILES = {"pyproject.toml", "README.md", "LICENSE", "NOTICE"}


def _tracked() -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--stage", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    )
    entries = {}
    for entry in result.stdout.split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            entries[os.fsdecode(path)] = metadata.split()[0].decode("ascii")
    return entries


def _version(source: Path) -> str:
    with (source / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def build(output_dir: Path) -> Path:
    """Write a zip whose workspace and wheel inputs come only from the Git index."""
    tracked = _tracked()
    required = set(WORKSPACE_FILES) | BUILD_FILES
    if missing := required - tracked.keys():
        raise ValueError(f"workspace inputs must be tracked: {', '.join(sorted(missing))}")
    workspace_files = WORKSPACE_FILES | {
        path: path for path in tracked if path.startswith("LICENSES/")
    }
    with tempfile.TemporaryDirectory(prefix="yap-workspace-") as temporary:
        staging = Path(temporary)
        source = staging / "source"
        wheel_inputs = {
            path for path in tracked
            if path in BUILD_FILES or path.startswith(("src/", "LICENSES/"))
        }
        inputs = sorted(set(workspace_files) | wheel_inputs)
        for relative in inputs:
            if tracked[relative] == "120000":
                raise ValueError(f"tracked input is a link: {relative}")
        source.mkdir()
        subprocess.run(
            ["git", "checkout-index", f"--prefix={source}{os.sep}", "-z", "--stdin"],
            input=b"".join(os.fsencode(path) + b"\0" for path in inputs),
            cwd=ROOT, check=True,
        )
        version = _version(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"yap-studio-workspace-{version}.zip"
        wheels = staging / "wheels"
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(wheels), str(source)],
            cwd=ROOT, check=True, stdin=subprocess.DEVNULL,
        )
        built = list(wheels.glob("*.whl"))
        if len(built) != 1:
            raise ValueError(f"expected one wheel; found {len(built)}")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, member in sorted(workspace_files.items()):
                archive.write(source / relative, member)
            for member, relative in sorted(WORKSPACE_COPIES.items()):
                archive.write(source / relative, member)
            archive.writestr(".yap-studio-version", version + "\n")
            archive.write(built[0], f"wheels/{built[0].name}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    print(build(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
