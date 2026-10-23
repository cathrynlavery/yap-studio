"""Script entry points report the project Python requirement before importing dependencies."""

from __future__ import annotations

import ast
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def _assert_version_hint(result: subprocess.CompletedProcess[str], script: Path) -> None:
    """An older Python exits with a command that works from any directory and no traceback."""
    assert result.returncode == 1
    assert f"cd {shlex.quote(str(ROOT))} && uv run python scripts/{script.name}" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.name)
def test_scripts_stop_before_imports_with_older_version_info(
    script: Path, tmp_path: Path,
) -> None:
    """Every script checks the Python version before importing project dependencies."""
    code = (
        "import runpy, sys; "
        "path = sys.argv[1]; "
        "sys.version_info = (3, 9, 6); "
        "sys.argv = [path]; "
        'runpy.run_path(path, run_name="__main__")'
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], cwd=tmp_path,
        capture_output=True, text=True, timeout=30, check=False,
    )
    _assert_version_hint(result, script)


def test_scripts_stop_on_real_older_python(tmp_path: Path) -> None:
    """Every script parses and exits cleanly on an installed older Python."""
    candidates = [Path("/usr/bin/python3")]
    candidates.extend(Path(found) for name in ("python3.9", "python3.10")
                      if (found := shutil.which(name)))
    for python in candidates:
        if not python.is_file():
            continue
        version = subprocess.run(
            [str(python), "-c", "import sys; print(sys.version_info[:2])"],
            cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
        )
        if version.returncode == 0 and ast.literal_eval(version.stdout.strip()) < (3, 11):
            break
    else:
        pytest.skip("Python older than 3.11 is unavailable")

    for script in SCRIPTS:
        result = subprocess.run(
            [str(python), str(script)], cwd=tmp_path,
            capture_output=True, text=True, timeout=30, check=False,
        )
        _assert_version_hint(result, script)
