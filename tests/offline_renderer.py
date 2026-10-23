"""Wrap the installed renderer in a macOS sandbox that denies outbound networking."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from yap_studio.render.hyperframes import RendererInstall

_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_PROFILE = """(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
"""


def offline_supported() -> bool:
    """Return whether the host provides the macOS sandbox runner."""
    return sys.platform == "darwin" and _SANDBOX_EXEC.is_file()


def _profile(tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    profile = tmp_dir / "offline.sb"
    profile.write_text(_PROFILE, encoding="utf-8")
    return profile.absolute()


def offline_install(install: RendererInstall, tmp_dir: Path) -> RendererInstall:
    """Copy installation metadata with an executable that denies external networking."""
    profile = _profile(tmp_dir)
    wrapper = tmp_dir / "hyperframes-offline"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"exec {_SANDBOX_EXEC} -f {shlex.quote(str(profile))} "
        f'{shlex.quote(str(install.executable.absolute()))} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    return replace(install, executable=wrapper.absolute())


def assert_network_blocked(tmp_dir: Path) -> None:
    """Assert that an external socket connection fails under the render sandbox."""
    profile = _profile(tmp_dir)
    probe = (
        "import socket\n"
        "try:\n"
        '    connection = socket.create_connection(("1.1.1.1", 443), timeout=5)\n'
        "except OSError:\n"
        "    raise SystemExit(0)\n"
        "connection.close()\n"
        'raise SystemExit("sandbox allowed external networking")\n'
    )
    result = subprocess.run(
        [str(_SANDBOX_EXEC), "-f", str(profile), "python3", "-c", probe],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
