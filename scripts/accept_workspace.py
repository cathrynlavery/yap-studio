"""Accept a workspace zip by installing its wheel and fetching its home page."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/accept_workspace.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/accept_workspace.py"
    )

import json
import os
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def accept(archive: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="yap-accept-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(workspace)
        env = os.environ.copy()
        for name, folder in {
            "HOME": "home", "UV_TOOL_DIR": "tools", "UV_TOOL_BIN_DIR": "bin",
            "UV_CACHE_DIR": "uv-cache", "YAP_STUDIO_CONFIG": "config",
            "YAP_STUDIO_CACHE": "cache", "YAP_STUDIO_DATA_DIR": "data",
        }.items():
            target = root / folder
            target.mkdir(exist_ok=True)
            env[name] = str(target)
        env.pop("PYTHONPATH", None)
        env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", os.defpath)
        subprocess.run([sys.executable, "setup.py", "--check"], cwd=workspace, env=env,
                       stdin=subprocess.DEVNULL, check=True, timeout=660)
        engine = str(root / "bin" / "yap-studio")
        command = [engine, "home", "--workspace", str(workspace), "--json"]
        try:
            started = subprocess.run([*command, "--detach", "--no-open"], cwd=workspace,
                                     env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, check=True, timeout=60)
            url = json.loads(started.stdout)["url"]
            with urllib.request.urlopen(url, timeout=15) as page:
                if page.status != 200 or not page.read():
                    raise ValueError("Workspace home page must return HTTP 200 with a body.")
            print("Clean workspace install: home page returned HTTP 200.")
        finally:
            subprocess.run([*command, "--stop"], cwd=workspace, env=env,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           check=True, timeout=60)


if __name__ == "__main__":
    accept(Path(sys.argv[1]))
