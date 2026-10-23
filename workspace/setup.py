"""Install the bundled Yap Studio wheel and show commands for missing tools."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from new_project import _inbox_takes, _unused_takes, _used

ROOT = Path(__file__).resolve().parent
MIN_FREE_GB = 10
MIN_NODE_MAJOR = 22
HOME_STATE_TIMEOUT = 120
# Keep in sync with src/yap_studio/tools.py. Setup cannot import the engine yet.
HOMEBREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
# Keep in sync with src/yap_studio/tools.py. Setup cannot import the engine yet.
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
SYNC_FOLDERS = {
    "mobile": "Library/Mobile Documents",
    "cloud": "Library/CloudStorage",
    "cloud_docs": "com~apple~CloudDocs",
    "db": "Dropbox",
    "od": "OneDrive",
    "gd": "Google Drive",
    "gd_cloud": "GoogleDrive",
    "marker": ".dropbox",
}
SYNC_LABELS = {
    "ic": "iCloud Drive",
    "db": "Dropbox",
    "od": "OneDrive",
    "gd": "Google Drive",
}
MAC_HINTS = {
    "uv": "brew install uv",
    "ffmpeg": "brew install ffmpeg",
    "node": "brew install node@22",
}
APT_HINTS = {
    "ffmpeg": "sudo apt install ffmpeg",
    # Distribution nodejs packages are often older than 22.
    "node": "curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - "
            "&& sudo apt-get install -y nodejs",
}
UV_INSTALLER = "curl -LsSf https://astral.sh/uv/install.sh | sh"
FFMPEG_PAGE = "https://ffmpeg.org/download.html"
# Without a known package tool, setup names the install page instead.
OTHER_HINTS = {
    "uv": "https://docs.astral.sh/uv/getting-started/installation/",
    "ffmpeg": FFMPEG_PAGE,
    "node": "https://nodejs.org/en/download",
}


def _command(binary: str) -> str:
    return shlex.quote(binary)


def _brew_binary() -> str | None:
    if shutil.which("brew"):
        return "brew"
    for folder in HOMEBREW_BIN_DIRS:
        binary = Path(folder) / "brew"
        if binary.is_file() and os.access(binary, os.X_OK):
            return str(binary)
    return None


def _add_homebrew_path() -> None:
    """Append installed Homebrew bins without changing the creator's PATH order."""
    if sys.platform != "darwin":
        return
    folders = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for folder in HOMEBREW_BIN_DIRS:
        brew = Path(folder) / "brew"
        if folder not in folders and brew.is_file() and os.access(brew, os.X_OK):
            folders.append(folder)
            os.environ["PATH"] = os.pathsep.join(folders)


def _uv_binary() -> str | None:
    found = shutil.which("uv")
    if found or sys.platform != "darwin":
        return found
    home = Path.home()
    folders = [os.environ.get("XDG_BIN_HOME"),
               str(Path(os.environ["XDG_DATA_HOME"]) / "../bin")
               if os.environ.get("XDG_DATA_HOME") else None,
               str(home / ".local/bin"), str(home / ".cargo/bin")]
    for folder in folders:
        if folder:
            binary = Path(folder) / "uv"
            if binary.is_file() and os.access(binary, os.X_OK):
                return str(binary)
    return None


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _synced_service(path: Path, home: Path | None = None) -> str | None:
    """Keep this check local so setup works before the engine is installed."""
    path = path.resolve()
    home = (home or Path.home()).resolve()
    mobile = home / SYNC_FOLDERS["mobile"]
    cloud = home / SYNC_FOLDERS["cloud"]
    if _inside(path, mobile):
        return SYNC_LABELS["ic"]
    if _inside(path, cloud):
        relative = path.relative_to(cloud)
        if relative.parts:
            provider = relative.parts[0].split("-", 1)[0]
            return {SYNC_FOLDERS["gd_cloud"]: SYNC_LABELS["gd"]}.get(
                provider, provider)
    for key in ("db", "gd"):
        name, service = SYNC_FOLDERS[key], SYNC_LABELS[key]
        if _inside(path, home / name):
            return service
    if any(_inside(path, folder) for folder in home.glob(SYNC_FOLDERS["od"] + "*")):
        return SYNC_LABELS["od"]
    cloud_docs = mobile / SYNC_FOLDERS["cloud_docs"]
    for name in ("Desktop", "Documents"):
        if _inside(path, home / name) and (cloud_docs / name).exists():
            return SYNC_LABELS["ic"]
    if any((folder / SYNC_FOLDERS["marker"]).is_file()
           for folder in (path, *path.parents) if folder != home):
        return SYNC_LABELS["db"]
    return None


def _free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def _inbox_take_count() -> int:
    inbox = ROOT / "inbox"
    if not inbox.is_dir():
        return 0
    return len(_unused_takes(_inbox_takes(inbox), _used(inbox)))


def _result(
    ready: bool, refused: dict | None, steps: list[dict], json_output: bool, *,
    inbox_takes: int = 0, phone: dict | None = None, home: dict | None = None,
) -> int:
    if json_output:
        result = {"ready": ready, "refused": refused, "steps": steps}
        if ready:
            result["inbox_takes"] = inbox_takes
            result["phone"] = phone
            result["home"] = home
        print(json.dumps(result))
    return 0 if ready else 1


def _step(
    step_id: str, does: str, command: str | None, *, page: str | None = None,
    optional: bool = False,
) -> dict:
    return {"id": step_id, "does": does, "command": command, "page": page,
            "optional": optional}


def _hint_step(step_id: str, does: str, hint: str) -> dict:
    if hint.startswith("https://"):
        return _step(step_id, does, None, page=hint)
    return _step(step_id, does, hint)


def _step_instruction(step: dict) -> str:
    return step["command"] or f"Open {step['page']}"


def _shell_join(args: list[str]) -> str:
    # Step commands are POSIX shell text, like every other step here: the agent runs them
    # in Bash (Claude Code uses Git Bash on Windows), never in cmd.exe or PowerShell.
    return shlex.join(args)


def _tool_bin_dir(uv: str) -> Path:
    # UV_TOOL_BIN_DIR wins in uv too; otherwise ask uv, which also honors XDG_BIN_HOME.
    if os.environ.get("UV_TOOL_BIN_DIR"):
        return Path(os.environ["UV_TOOL_BIN_DIR"])
    try:
        found = subprocess.run([uv, "tool", "dir", "--bin"], capture_output=True, text=True,
                               timeout=30)
    except (OSError, subprocess.SubprocessError):
        found = None
    if found is not None and found.returncode == 0 and found.stdout.strip():
        return Path(found.stdout.strip())
    return Path(os.environ.get("UV_TOOL_BIN_DIR", Path.home() / ".local/bin"))


def _engine_commands(
    uv: str, wheel: Path, engine: Path,
) -> tuple[list[str], list[str], list[str]]:
    # A named file URL enables the wheel's extra and escapes spaces in its path.
    requirement = f"yap-studio[asr] @ {wheel.resolve().as_uri()}"
    return ([uv, "tool", "install", "--force", requirement],
            [str(engine), "speech", "download"],
            [str(engine), "renderer", "install"])


def _find_node(path: str, home: str) -> tuple[str, str] | None:
    """Match the engine's Node/npm lookup before it is installed."""
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
            return str(node), str(npm)
    return None


def _node_available() -> bool:
    """Require a supported Node with npm in the same folder."""
    return _find_node(os.environ.get("PATH", os.defpath), str(Path.home())) is not None


def _renderer_step(engine: Path) -> dict:
    return _step("renderer", "Install the renderer and browser for video graphics.",
                 f"{_command(str(engine))} renderer install")


def _engine_step(uv: str, wheel: Path, engine: Path, *, include_renderer: bool = True) -> dict:
    commands = _engine_commands(uv, wheel, engine)
    if not include_renderer:
        commands = commands[:2]
    command = " && ".join(_shell_join(args) for args in commands)
    # These estimates match the default speech backends' download notices.
    does = ("Install Yap Studio and its speech engine, then download the speech model "
            "(about 2.5 GB on a Mac with Apple silicon; about 670 MB on an Intel Mac)")
    does += (", then download and install the renderer and its browser for video graphics."
             if include_renderer else ".")
    return _step("engine", does, command)


def _engine_steps(uv: str, wheel: Path, engine: Path, hints: dict[str, str]) -> list[dict]:
    if _node_available():
        return [_engine_step(uv, wheel, engine)]
    return [
        _hint_step("node", "Install Node 22 or newer to run the renderer.", hints["node"]),
        _engine_step(uv, wheel, engine, include_renderer=False),
        _renderer_step(engine),
    ]


def _wheel_build(wheel: Path) -> str | None:
    """Identify bundled package files from their wheel RECORD hashes."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            records = [name for name in archive.namelist()
                       if name.endswith(".dist-info/RECORD")]
            if len(records) != 1:
                return None
            rows = csv.reader(io.StringIO(archive.read(records[0]).decode("utf-8")))
            lines = [f"{row[0]},{row[1]}" for row in rows if len(row) >= 2 and row[1]
                     and row[0].startswith("yap_studio/")
                     and "__pycache__" not in row[0] and not row[0].endswith(".pyc")]
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile, csv.Error):
        return None
    if not lines:
        return None
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def _install_wheel(
    uv: str, wheel: Path, engine: Path, *, json_output: bool = False,
) -> str | None:
    install, download, renderer = _engine_commands(uv, wheel, engine)
    try:
        subprocess.run(install, check=True,
                       stdout=sys.stderr if json_output else None, text=True)
        if not engine.is_file():
            # uv may have placed the executable somewhere else on PATH.
            found = shutil.which("yap-studio")
            if found is None:
                return f"installed, but yap-studio is not in {engine.parent} or on PATH"
            download = [found, *download[1:]]
            renderer = [found, *renderer[1:]]
        subprocess.run(download, check=True,
                       stdout=sys.stderr if json_output else None, text=True)
        if _node_available():
            subprocess.run(renderer, check=True,
                           stdout=sys.stderr if json_output else None, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(getattr(exc, "stderr", None) or exc)
    return None


def _missing_steps(
    report: dict, hints: dict[str, str], engine: Path, wheel: Path, uv: str,
) -> list[dict]:
    steps: list[dict] = []
    if not report["ffmpeg"]["available"] or not report["ffprobe"]["available"] or not all(
        report["filters"].values()
    ):
        old = any(report[tool].get("path") and not report[tool]["available"]
                  for tool in ("ffmpeg", "ffprobe")) or (
                      report["ffmpeg"]["available"] and report["ffprobe"]["available"])
        does = "Install ffmpeg, which reads and writes video files."
        if old:
            steps.append(_step("ffmpeg", does, None, page=FFMPEG_PAGE))
        else:
            steps.append(_hint_step("ffmpeg", does, hints["ffmpeg"]))
    if not report["node"]["available"]:
        steps.append(_hint_step("node", "Install Node 22 or newer to run the renderer.",
                                hints["node"]))
    if not report["renderer"]["available"] or not report["renderer"]["browser"]:
        steps.append(_renderer_step(engine))
    if not report["speech"]["available"]:
        command = " && ".join(_shell_join(args)
                              for args in _engine_commands(uv, wheel, engine)[:2])
        steps.append(_step("speech", "Install the speech engine and download its model "
                           "to transcribe takes.", command))
    elif not report["speech"].get("model", True):
        steps.append(_step("speech_model", "Download the speech model before the first take.",
                           f"{_command(str(engine))} speech download"))
    if sys.platform == "darwin" and not report.get("cutout", {}).get("available", True):
        steps.append(_step("cutout", "Install Xcode tools to build the optional cutout helper.",
                           "xcode-select --install", optional=True))
    tailscale = report.get("phone", {}).get("tailscale")
    if tailscale in ("missing", "installed"):
        action = "set up Tailscale" if tailscale == "missing" else "open Tailscale and sign in"
        steps.append(_step(
            "tailscale", f"Optional: {action} so review links open on your phone anywhere, "
            "not only on this Wi-Fi.", None, page="https://tailscale.com/download",
            optional=True,
        ))
    return steps


def _hints() -> dict[str, str]:
    """Pick install commands for this computer's package tools."""
    if sys.platform == "darwin":
        brew = _brew_binary()
        target = brew or ("/opt/homebrew/bin/brew" if platform.machine() == "arm64"
                          else "/usr/local/bin/brew")
        hints = {key: hint.replace("brew", _command(target)) for key, hint in MAC_HINTS.items()}
        if brew is None:
            hints["uv"] = UV_INSTALLER
        return hints
    if not sys.platform.startswith("linux"):
        return OTHER_HINTS
    apt = APT_HINTS if shutil.which("apt-get") else {}
    return {**OTHER_HINTS, "uv": UV_INSTALLER, **apt}


def _homebrew_steps(steps: list[dict], *, needs_brew: bool = False) -> list[dict]:
    if (sys.platform == "darwin" and _brew_binary() is None
            and (needs_brew or any(step["id"] in {"ffmpeg", "node"} for step in steps))):
        return [_step(
            "homebrew", "Homebrew installs ffmpeg and Node. Open the Terminal app, paste "
            "the command from brew.sh, type your Mac password when it asks, then say done.",
            None, page="https://brew.sh",
        ), *steps]
    return steps


def _doctor(engine: Path) -> tuple[subprocess.CompletedProcess[str], dict | None]:
    """Run the editing doctor; the report is None when the engine cannot give one."""
    # --edit makes doctor fail unless every editing dependency is ready.
    doctor = subprocess.run(
        [str(engine), "doctor", "--edit", "--json"], cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    try:
        report = json.loads(doctor.stdout)
    except ValueError:
        return doctor, None
    return doctor, report if isinstance(report, dict) and "ready_to_edit" in report else None


def _open_home(engine: Path, *, phone: bool) -> dict:
    command = [str(engine), "home", "--workspace", str(ROOT)]
    if phone:
        command.append("--phone")
    command.extend(("--no-open", "--detach", "--json"))
    try:
        # Allow the 120 second detached readiness window plus process startup.
        launched = subprocess.run(command, capture_output=True, text=True, timeout=150, check=False)
    except subprocess.TimeoutExpired:
        subprocess.run([str(engine), "home", "--stop", "--workspace", str(ROOT), "--json"],
                       capture_output=True, text=True, timeout=30, check=True)
        raise
    if launched.returncode:
        if not phone:
            raise ValueError("home did not start")
        reason = launched.stderr.strip() or launched.stdout.strip()
        try:
            reason = json.loads(reason)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        home = _open_home(engine, phone=False)
        home["phone_error"] = reason or f"Home exited with status {launched.returncode}."
        return home
    document = json.loads(launched.stdout)
    link = document["url"]
    parts = urlsplit(link)
    if (parts.scheme not in {"http", "https"} or not parts.netloc
            or not parse_qs(parts.query).get("t")):
        raise ValueError("home did not return a private link")
    state_url = f"{parts.scheme}://{parts.netloc}/api/state?{parts.query}"
    deadline = time.monotonic() + HOME_STATE_TIMEOUT
    print("Waiting for Yap Studio home to read your inbox...", file=sys.stderr, flush=True)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("home state did not answer")
        try:
            with urlopen(state_url, timeout=min(5, remaining)) as response:
                state = json.load(response)
                if (response.status != 200 or not isinstance(state, dict)
                        or state.get("schema") != "yap-studio/home@1"
                        or state.get("workspace") != ROOT.name):
                    raise ValueError("home state did not answer")
            break
        except HTTPError:
            raise
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(1, remaining))
    return {"url": link, "phone": document.get("phone")}


def main(argv: list[str] | None = None) -> int:
    # Install commands run in the creator's shell, which still has the original PATH.
    hints = _hints()
    _add_homebrew_path()
    parser = argparse.ArgumentParser(description="Set up the Yap Studio workspace")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--check", action="store_true",
        help="Install and check the bundled wheel without model or renderer downloads",
    )
    args = parser.parse_args([] if argv is None else argv)
    service = _synced_service(ROOT)
    if service is not None:
        target = Path.home() / "Yap Studio"
        move = None if target.exists() else f"mv {_command(str(ROOT))} {_command(str(target))}"
        if not args.json:
            print(
                f"This folder is in {service}. Synced folders move video files to the cloud "
                "to save space, so takes and renders go missing mid-edit. Move the whole folder "
                "somewhere that does not sync, such as your home folder, then run setup again:"
            )
            if move:
                print(f"  {move}")
            else:
                print(f"  {target} already exists; choose another local folder.")
        return _result(False, {"reason": "folder", "service": service, "move": move},
                       [], args.json)
    free_gb = _free_disk_gb(ROOT)
    if free_gb < MIN_FREE_GB:
        if not args.json:
            print(f"Only {free_gb:.1f} GB free; setup needs {MIN_FREE_GB} GB free.")
        return _result(False, {"reason": "disk", "service": None, "move": None},
                       [], args.json)
    wheels = list((ROOT / "wheels").glob("yap_studio-*.whl"))
    if len(wheels) != 1:
        if not args.json:
            print("Expected one bundled Yap Studio wheel.", file=sys.stderr)
        return _result(False, {"reason": "wheel", "service": None, "move": None},
                       [], args.json)
    uv = _uv_binary()
    if uv is None:
        step = _hint_step("uv", "Install uv to install the bundled engine.", hints["uv"])
        needs_brew = sys.platform == "darwin" and (
            not shutil.which("ffmpeg") or not shutil.which("ffprobe") or not _node_available())
        steps = _homebrew_steps([step], needs_brew=needs_brew)
        if not args.json:
            if steps[0]["id"] == "homebrew":
                print(f"{steps[0]['does']} Open {steps[0]['page']}", file=sys.stderr)
            print(f"Install uv, then run this setup again: {_step_instruction(step)}",
                  file=sys.stderr)
        return _result(False, None, steps, args.json)
    wheel = wheels[0]
    wheel_build = _wheel_build(wheel)
    if args.check:
        engine = _tool_bin_dir(uv) / "yap-studio"
        try:
            subprocess.run(_engine_commands(uv, wheel, engine)[0], check=True,
                           stdin=subprocess.DEVNULL, stdout=sys.stderr, timeout=600)
            _, report = _doctor(engine)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"Bundled wheel check failed: {exc}", file=sys.stderr)
            return 1
        if (wheel_build is None or report is None
                or report["install"].get("build") != wheel_build):
            print("Installed engine does not match the bundled wheel.", file=sys.stderr)
            return 1
        print(json.dumps({"installed": True, "build": wheel_build}))
        return 0
    bin_dir = _tool_bin_dir(uv)
    engine = bin_dir / "yap-studio"
    expected_version = (ROOT / ".yap-studio-version").read_text(encoding="utf-8").strip()
    current_version = None
    if engine.is_file():
        current = subprocess.run([str(engine), "--version"], capture_output=True, text=True)
        if current.returncode == 0:
            parts = current.stdout.split()
            if len(parts) >= 2 and parts[0] == "yap-studio":
                current_version = parts[1]
    installed = current_version != expected_version
    if installed:
        if args.json:
            return _result(False, None,
                           _homebrew_steps(_engine_steps(uv, wheel, engine, hints)), True)
        error = _install_wheel(uv, wheel, engine, json_output=args.json)
        if error is not None:
            if not args.json:
                print(f"Could not install the bundled engine: {error}", file=sys.stderr)
            return _result(False, {"reason": "engine install", "service": None, "move": None},
                           [], args.json)
    if not engine.is_file():
        found = shutil.which("yap-studio")
        if found is None:
            if not args.json:
                print(f"Installed the engine; add {bin_dir} to PATH and run setup again.")
            return _result(False, {"reason": "engine path", "service": None, "move": None},
                           [], args.json)
        engine = Path(found)
    doctor, report = _doctor(engine)
    stale = report is None or (wheel_build is not None
                               and report["install"].get("build") != wheel_build)
    if stale and not installed:
        # Same-version builds may lack doctor --edit or differ from the bundled wheel.
        if args.json:
            return _result(False, None,
                           _homebrew_steps(_engine_steps(uv, wheel, engine, hints)), True)
        error = _install_wheel(uv, wheel, engine, json_output=args.json)
        if error is not None:
            if not args.json:
                print(f"Could not install the bundled engine: {error}", file=sys.stderr)
            return _result(False, {"reason": "engine install", "service": None, "move": None},
                           [], args.json)
        installed = True
        doctor, report = _doctor(engine)
    if report is None:
        if not args.json:
            print(doctor.stdout or doctor.stderr, file=sys.stderr)
        return _result(False, {"reason": "doctor", "service": None, "move": None},
                       [], args.json)
    if installed and wheel_build is not None and report["install"].get("build") != wheel_build:
        if not args.json:
            print(f"Engine at {engine} is not the build bundled with this workspace; "
                  "another install may come first on PATH.", file=sys.stderr)
        return _result(False, {"reason": "engine build", "service": None, "move": None},
                       [], args.json)
    steps = _homebrew_steps(_missing_steps(report, hints, engine, wheel, uv))
    phone = {"via": report.get("phone", {}).get("via"),
             "tailscale": report.get("phone", {}).get("tailscale", "missing")}
    if not args.json:
        print(f"Engine {report['install']['version']} installed. Doctor ran.")
        for warning in report["warnings"]:
            print(f"Warning: {warning}")
        for number, step in enumerate(steps, 1):
            suffix = " (optional)" if step["optional"] else ""
            print(f"Step {number} of {len(steps)}{suffix}: {step['does']}\n"
                  f"  {_step_instruction(step)}")
    if doctor.returncode and report["ready_to_edit"]:
        if not args.json:
            print(doctor.stderr, file=sys.stderr)
        if args.json:
            _result(False, None, steps, True)
        return doctor.returncode
    ready = report["ready_to_edit"] and all(step["optional"] for step in steps)
    takes = 0
    home = None
    if ready:
        try:
            takes = _inbox_take_count()
        except (ValueError, OSError) as exc:
            if not args.json:
                print(f"Could not read the inbox: {exc}", file=sys.stderr)
            return _result(False, {"reason": "inbox", "service": None, "move": None,
                                   "error": str(exc)}, steps, args.json)
        try:
            home = _open_home(engine, phone=phone["via"] is not None)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            fix = "Run yap-studio home --workspace . --detach --json, then run setup again."
            if not args.json:
                print(f"Yap Studio home did not answer. {fix}", file=sys.stderr)
            return _result(False, {"reason": "home", "fix": fix}, steps, args.json)
    if not args.json:
        if ready:
            noun = "take" if takes == 1 else "takes"
            print("Setup complete")
            print(f"Yap Studio home: {home['url']}")
            if home["phone"]:
                print(f"Open on your phone: {home['phone']['url']}")
            print(f"Ready to edit. {takes} {noun} waiting in inbox/.")
            if home.get("phone_error"):
                print(f"Phone link unavailable: {home['phone_error']}")
            elif phone["via"] is None:
                print("Your phone needs the same Wi-Fi or Tailscale to open review links.")
            else:
                reach = ("anywhere through Tailscale" if phone["tailscale"] == "connected"
                         else "on the same Wi-Fi")
                print(f"Review links reach your phone {reach}.")
            if takes:
                print("Say 'edit this video' to start.")
            else:
                print("Put a take in inbox/, then say 'edit this video'.")
        else:
            print("Not ready to edit yet. Run the commands above, then run this setup again.")
    return _result(ready, None, steps, args.json, inbox_takes=takes, phone=phone, home=home)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
