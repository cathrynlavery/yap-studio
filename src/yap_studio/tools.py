"""Allowlisted runners for tools outside the ffmpeg family."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

from yap_studio import process
from yap_studio.errors import DependencyError, ProcessingError

ALLOWED_TOOLS = frozenset(
    {"node", "npm", "hyperframes", "xcrun", "swiftc", "yap-person-matte", "yap-face-boxes",
     "face-landmarks",
     "elevenlabs-pp-cli", "pexels-pp-cli",
     "wavespeed-pp-cli",
     "osxphotos",
     "git"}
)
_STDERR_LIMIT = 64 * 1024
MIN_NODE_MAJOR = 22
# Keep in sync with workspace/setup.py, which runs before the engine is installed.
HOMEBREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
# Keep in sync with workspace/setup.py, which runs before the engine is installed.
NODE_INSTALL_PLACES = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/homebrew/opt/node/bin",
    "/opt/homebrew/opt/node@*/bin",
    "/usr/local/opt/node/bin",
    "/usr/local/opt/node@*/bin",
    "~/.nvm/versions/node/*/bin",
    "~/.volta/bin",
)


def add_homebrew_path() -> None:
    """Append installed Homebrew bins without changing the creator's PATH order."""
    if sys.platform != "darwin":
        return
    folders = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for folder in HOMEBREW_BIN_DIRS:
        brew = Path(folder) / "brew"
        if folder not in folders and brew.is_file() and os.access(brew, os.X_OK):
            folders.append(folder)
            os.environ["PATH"] = os.pathsep.join(folders)


@cache
def _find_node(path: str, home: str) -> tuple[str, str, str] | None:
    """Cache a working Node/npm pair and version for this PATH and home directory."""
    folders = path.split(os.pathsep)
    for place in NODE_INSTALL_PLACES:
        pattern = home + place[1:] if place.startswith("~/") else place
        folders.extend(sorted(glob.glob(pattern), key=lambda folder: tuple(
            int(part) for part in re.findall(r"\d+", folder)), reverse=True))
    seen: set[Path] = set()
    for folder in folders:
        node, npm = Path(folder).absolute() / "node", Path(folder).absolute() / "npm"
        if not all(binary.is_file() and os.access(binary, os.X_OK) for binary in (node, npm)):
            continue
        resolved = node.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            result = subprocess.run(
                [str(node), "--version"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.fullmatch(r"v?(\d+)\.\d+\.\d+(?:[-+].*)?", result.stdout.strip())
        if result.returncode == 0 and match is not None and int(match[1]) >= MIN_NODE_MAJOR:
            return str(node), str(npm), result.stdout.strip()
    return None


def node_version() -> str | None:
    """Return the working Node's version from its cached probe."""
    pair = _find_node(os.environ.get("PATH", os.defpath), str(Path.home()))
    return pair[2] if pair is not None else None


def node_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Overlay PATH only when a child would otherwise run a different Node."""
    overlay = dict(env or {})
    pair = _find_node(os.environ.get("PATH", os.defpath), str(Path.home()))
    if pair is not None:
        path = overlay.get("PATH", os.environ.get("PATH", os.defpath))
        first = shutil.which("node", path=path)
        if first is None or Path(first).absolute() != Path(pair[0]):
            overlay["PATH"] = str(Path(pair[0]).parent) + os.pathsep + path
    return overlay


def _check_name(name: str) -> None:
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"unsupported tool: {name}")


def find_tool(name: str) -> str | None:
    """Find an allowlisted executable, requiring a working pair for Node/npm."""
    _check_name(name)
    if name in {"node", "npm"}:
        pair = _find_node(os.environ.get("PATH", os.defpath), str(Path.home()))
        if pair is not None:
            return pair[0] if name == "node" else pair[1]
        return None
    return shutil.which(name)


def _missing_tool(name: str) -> DependencyError:
    if name in {"node", "npm"}:
        return DependencyError.missing_node()
    if name == "hyperframes":
        return DependencyError.missing_renderer()
    if name in {"xcrun", "swiftc"}:
        return DependencyError.missing_xcode_tools()
    if name == "elevenlabs-pp-cli":
        return DependencyError.missing_elevenlabs_cli()
    if name == "pexels-pp-cli":
        return DependencyError.missing_pexels_cli()
    if name == "wavespeed-pp-cli":
        return DependencyError.missing_wavespeed_cli()
    if name == "osxphotos":
        return DependencyError(
            "osxphotos not found.", error_type="dependency_error", code="osxphotos_not_found",
            suggested_action="Install it with uv tool install osxphotos.",
        )
    if name == "git":
        return DependencyError(
            "git not found.",
            error_type="dependency_error",
            code="git_not_found",
            suggested_action="Install Git and make it available on PATH.",
        )
    if name == "yap-face-boxes":
        return DependencyError(
            "yap-face-boxes not found.",
            error_type="dependency_error",
            code="yap_face_boxes_not_found",
            suggested_action=(
                "Run yap-studio faces detect on macOS with the Xcode command line tools "
                "to build the helper."
            ),
        )
    if name == "face-landmarks":
        return DependencyError(
            "face-landmarks not found.",
            error_type="dependency_error",
            code="face_landmarks_not_found",
            suggested_action="Install the Xcode command line tools to build the helper.",
        )
    return DependencyError(
        f"{name} not found.",
        error_type="dependency_error",
        code="yap_person_matte_not_found",
        suggested_action="Run yap-studio doctor to prepare the cutout helper.",
    )


def run_tool(
    name: str,
    args: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
    executable: str | os.PathLike[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    stdout_path: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[str | None]:
    """Run an allowlisted tool with text output and a required deadline.

    An explicit executable supports cached tools. Only the ElevenLabs CLI may
    receive a stdin payload or redirect stdout; its caller must allow for the whole retry window.
    Environment values overlay the inherited environment.
    """
    _check_name(name)
    if input_bytes is not None and name != "elevenlabs-pp-cli":
        raise ValueError("stdin payloads are only supported for elevenlabs-pp-cli")
    if stdout_path is not None and name != "elevenlabs-pp-cli":
        raise ValueError("stdout_path is only supported for elevenlabs-pp-cli")
    if executable is None:
        found = find_tool(name)
    else:
        candidate = Path(executable).expanduser().absolute()
        found = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    if found is None:
        raise _missing_tool(name)
    command = [found, *args]
    if name in {"node", "npm", "hyperframes"}:
        env = node_env(env)
    result = process.spawn(
        command, timeout_s=timeout_s, tool=name, input_bytes=input_bytes, env=env, cwd=cwd,
        stdout_path=stdout_path,
    )
    if result.returncode:
        error = ProcessingError(
            " ".join(command),
            result.returncode,
            (result.stderr or "")[-_STDERR_LIMIT:],
            code=f"{name}_exit_{result.returncode}",
            stdout=(result.stdout or "")[-_STDERR_LIMIT:],
        )
        if name == "wavespeed-pp-cli":
            error.stdout = result.stdout
        raise error
    return result
