"""Repository hygiene checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from hygiene_patterns import SYNC_SERVICE_FILES, VIDEO_TOOL_PATTERNS, _has_machine_details

ROOT = Path(__file__).parents[1]

def test_sync_service_exception_is_exact() -> None:
    assert SYNC_SERVICE_FILES == {"src/yap_studio/setup_checks.py", "workspace/setup.py"}
    name = "Drop" + "box"
    for relative in SYNC_SERVICE_FILES:
        assert not _has_machine_details(name, relative)
        assert _has_machine_details("/" + "Users" + "/someone", relative)
        assert _has_machine_details("op" + "://vault/item", relative)
    assert _has_machine_details(name, "tests/test_setup_checks.py")
    assert _has_machine_details(name, "src/yap_studio/other.py")


def test_tracked_files_do_not_leak_machine_details() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if inside.returncode or inside.stdout.strip() != "true":
        pytest.skip("directory is not a git work tree")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.split(b"\0")
    test_path = Path(__file__).relative_to(ROOT).as_posix()
    offenders: list[str] = []
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = raw_path.decode(errors="surrogateescape")
        if relative in {"LICENSE", "NOTICE", test_path} or relative.startswith("LICENSES/"):
            continue
        file = ROOT / relative
        if not file.is_file():
            continue  # Tracked paths may have a pending working-tree deletion.
        data = file.read_bytes()
        if b"\0" in data[:8192]:
            continue
        for line_number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if _has_machine_details(line, relative):
                offenders.append(f"{relative}:{line_number}")
    message = "machine-specific or secret-tool references found:\n" + "\n".join(offenders)
    assert not offenders, message


def test_tracked_files_do_not_name_other_video_tools() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if inside.returncode or inside.stdout.strip() != "true":
        pytest.skip("directory is not a git work tree")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.split(b"\0")
    test_path = Path(__file__).relative_to(ROOT).as_posix()
    patterns = VIDEO_TOOL_PATTERNS
    offenders: list[str] = []
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = raw_path.decode(errors="surrogateescape")
        if relative in {"LICENSE", "NOTICE", test_path} or relative.startswith("LICENSES/"):
            continue
        file = ROOT / relative
        if not file.is_file():
            continue  # Tracked paths may have a pending working-tree deletion.
        data = file.read_bytes()
        if b"\0" in data[:8192]:
            continue
        for line_number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns):
                offenders.append(f"{relative}:{line_number}")
    message = "other video tool names found:\n" + "\n".join(offenders)
    assert not offenders, message


def test_no_tracked_internal_plans() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", str(Path("docs") / "plans")],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=30,
    ).stdout.split(b"\0")
    assert not [os_path for os_path in tracked if os_path]
