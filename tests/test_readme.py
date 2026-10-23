"""Check the documented walk-through and the documented commands."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import pytest

from yap_studio.cli import _argument_parser
from yap_studio.project.model import parse_document

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_images_exist_and_stay_under_10_mb() -> None:
    """Keep local README images available and small enough to load."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', readme, re.IGNORECASE)
    targets += re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    local_images = [target for target in targets if not target.startswith(("http://", "https://"))]
    assert local_images, "No local README images found"
    failures: list[str] = []
    for target in local_images:
        image = (ROOT / target).resolve()
        if not image.is_relative_to(ROOT) or not image.is_file():
            failures.append(f"{target}: missing")
            continue
        size = image.stat().st_size
        if size >= 10_000_000:
            failures.append(f"{target}: {size} bytes")
    assert not failures, "README image problems:\n" + "\n".join(failures)


def _walk_through() -> str:
    return (ROOT / "docs/editing.md").read_text(encoding="utf-8").split(
        "## Walk-through\n", 1,
    )[1].split("\n## ", 1)[0]


@pytest.mark.parametrize(("directory", "absolute"), [
    ("project", False),
    ("projects/2026-10-13-take", False),
    ("projects/2026-10-13-take", True),
    ("projects/my talking head take", True),
], ids=["source", "workspace-relative", "workspace-absolute", "workspace-spaces"])
def test_walk_through_starter_uses_the_supplied_project(
    tmp_path: Path, directory: str, absolute: bool,
) -> None:
    project = tmp_path / directory
    words_file = project / "ingest/t1/words.json"
    words_file.parent.mkdir(parents=True)
    raw = json.dumps({"words": [
        {"id": "t1:7", "text": "First", "start": 0.1, "end": 0.3},
        {"id": "t1:8", "text": "last", "start": 0.4, "end": 0.6},
    ]}).encode()
    words_file.write_bytes(raw)
    starter = re.findall(r"```sh\n(.*?)\n```", _walk_through(), re.DOTALL)[0]
    # Change only the shell argument, exactly as a workspace user would.
    argument = str(project) if absolute else directory
    starter = starter.replace("python3 - project <<", f"python3 - {shlex.quote(argument)} <<", 1)
    starter = starter.replace("python3 -", f"{shlex.quote(sys.executable)} -", 1)
    result = subprocess.run(
        ["sh"], input=starter, cwd=tmp_path, capture_output=True, text=True,
        timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    edit = json.loads((project / "edit.json").read_bytes())
    assert parse_document("edit", edit)
    assert edit["words_sha256"] == {"t1": hashlib.sha256(raw).hexdigest()}
    assert edit["segments"] == [{
        "id": "story", "take": "t1",
        "from": {"word": "t1:7"}, "to": {"word": "t1:8"},
    }]
    if directory != "project":
        assert not (tmp_path / "project").exists()


def test_walk_through_applies_removals_before_story_review() -> None:
    commands = [
        shlex.split(line)
        for block in re.findall(r"```console\n(.*?)\n```", _walk_through(), re.DOTALL)
        for line in block.splitlines()
        if line.startswith("yap-studio ")
    ]
    apply = commands.index(["yap-studio", "fillers", "project", "--apply"])
    cut = commands.index(["yap-studio", "cut", "project"])
    review = commands.index(["yap-studio", "review", "project/cut/cut.mp4"])
    run = commands.index(["yap-studio", "run", "project"])
    assert apply < cut < review < run


def test_documented_console_commands_parse() -> None:
    failures: list[str] = []
    checked = 0
    files = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "docs/recipes").glob("*.md")),
    ]
    for file in files:
        blocks = re.findall(
            r"^```console\n(.*?)^```$", file.read_text(encoding="utf-8"),
            re.MULTILINE | re.DOTALL,
        )
        for block in blocks:
            for line in block.splitlines():
                if not line.startswith("yap-studio "):
                    continue
                argv = shlex.split(line)[1:]
                if "--version" in argv or "--help" in argv:
                    continue
                checked += 1
                with redirect_stderr(StringIO()):
                    try:
                        parser, arguments = _argument_parser(argv)
                        parser.parse_args(arguments)
                    except SystemExit:
                        failures.append(f"{file.relative_to(ROOT)}: {line}")
    assert not failures, "Documented commands failed to parse:\n" + "\n".join(failures)
    assert checked >= 20, f"Only {checked} documented commands were checked"
