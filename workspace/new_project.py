"""Copy unused inbox takes into a project, then name it from the transcript."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows has no fcntl.
    fcntl = None

ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm"})
MEMO_SUFFIXES = frozenset({".m4a", ".wav", ".mp3", ".aac", ".caf", ".aif", ".aiff", ".flac"})
SCHEMA = "yap-studio/inbox-used@1"
VIDEO_TYPES = ("talking_head", "voiceover", "green_screen", "screen_recording")
BROLL_SOURCES = ("folder", "album", "stock", "none")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:40].strip("-") or "take"


def _inbox_takes(inbox: Path) -> list[Path]:
    """List regular visible video and memo files without following links."""
    return [path for path in inbox.iterdir() if path.is_file() and not path.is_symlink()
            and not path.name.startswith(".")
            and path.suffix.lower() in VIDEO_SUFFIXES | MEMO_SUFFIXES]


def _probe(path: Path) -> dict | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type:stream_disposition=attached_pic", "-of", "json", str(path)],
            timeout=30, capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _kind(path: Path) -> str:
    data = _probe(path)
    streams = data.get("streams", []) if data is not None else []
    if isinstance(streams, list):
        streams = [stream for stream in streams if isinstance(stream, dict)]
        if any(stream.get("codec_type") == "video"
               and stream.get("disposition", {}).get("attached_pic") != 1 for stream in streams):
            return "take"
        if any(stream.get("codec_type") == "audio" for stream in streams):
            return "memo"
    return "memo" if path.suffix.lower() in MEMO_SUFFIXES else "take"


def _used(inbox: Path) -> dict:
    ledger = inbox / ".used.json"
    if not ledger.exists():
        return {"schema": SCHEMA, "takes": []}
    try:
        data = json.loads(ledger.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("inbox/.used.json is invalid; fix it before starting a project") from exc
    if (not isinstance(data, dict) or set(data) != {"schema", "takes"}
            or data.get("schema") != SCHEMA
            or not isinstance(data.get("takes"), list)
            or any(not isinstance(item, dict)
                   or set(item) != {"file", "bytes", "mtime_ns", "project", "used_at"}
                   or not isinstance(item["file"], str)
                   or not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool)
                   or not isinstance(item["mtime_ns"], int)
                   or isinstance(item["mtime_ns"], bool)
                   or not isinstance(item["project"], str)
                   or not isinstance(item["used_at"], str)
                   for item in data["takes"])):
        raise ValueError("inbox/.used.json has the wrong shape; fix it before starting a project")
    return data


def _unused_takes(available: list[Path], used: dict) -> list[Path]:
    identities = {(item["file"], item["bytes"], item["mtime_ns"]) for item in used["takes"]}
    unused = []
    for take in available:
        info = take.stat()
        if (take.name, info.st_size, info.st_mtime_ns) not in identities:
            unused.append(take)
    return unused


def _write_json(target: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=target.parent, prefix=".new-project-",
                                     suffix=".tmp", delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(data, output, indent=2, ensure_ascii=False)
            output.write("\n")
        except BaseException:
            temporary.unlink()
            raise
    try:
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink()
        raise


def _reserve(projects: Path, base: str) -> Path:
    projects.mkdir(exist_ok=True)
    for number in range(1, 1001):
        destination = projects / (base if number == 1 else f"{base}-{number}")
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            continue
    raise ValueError("Could not reserve a fresh project folder.")


@contextmanager
def _project_lock(projects: Path):
    if fcntl is None:
        yield
        return
    projects.mkdir(exist_ok=True)
    with (projects / ".new_project.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _tool_bin_dir() -> Path:
    # Match setup.py: UV_TOOL_BIN_DIR wins in uv too; otherwise ask uv (it honors XDG_BIN_HOME).
    if os.environ.get("UV_TOOL_BIN_DIR"):
        return Path(os.environ["UV_TOOL_BIN_DIR"])
    uv = shutil.which("uv")
    if uv is not None:
        try:
            found = subprocess.run([uv, "tool", "dir", "--bin"], capture_output=True,
                                   text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            found = None
        if found is not None and found.returncode == 0 and found.stdout.strip():
            return Path(found.stdout.strip())
    return Path.home() / ".local/bin"


def _engine() -> str:
    installed = _tool_bin_dir() / "yap-studio"
    return str(installed) if installed.is_file() else (
        shutil.which("yap-studio") or str(installed)
    )


def start(root: Path = ROOT, takes: list[str] | None = None, newest: int = 1, *,
          video_type: str | None = None, broll_sources: list[str] | None = None) -> Path:
    """Initialize a fresh project with copied takes and record their identities."""
    with _project_lock(root / "projects"):
        return _start_locked(root, takes, newest,
                             video_type=video_type, broll_sources=broll_sources)


def _select(root: Path, available: list[Path], used: dict,
            takes: list[str] | None, newest: int) -> list[Path]:
    if takes:
        by_name = {path.name: path for path in available}
        selected = []
        for name in takes:
            source = by_name.get(Path(name).name)
            named = Path(name)
            supplied = named if named.is_absolute() else root / named
            if (source is None or (named.parent != Path(".")
                                  and supplied.resolve() != source.resolve())):
                names = ", ".join(sorted(by_name)) or "none"
                raise ValueError(f"Take {name!r} is not in inbox/. Takes there: {names}.")
            if source in selected:
                raise ValueError(f"Take {source.name!r} was named twice.")
            selected.append(source)
    else:
        if newest < 1:
            raise ValueError("--newest must be at least 1.")
        unused = _unused_takes(available, used)
        if len(unused) < newest:
            raise ValueError("Every take in inbox/ is already in a project, or too few unused "
                             "takes remain. Name a take to reuse it: python3 new_project.py TAKE.")
        selected = sorted(unused, key=lambda path: (path.stat().st_mtime_ns, path.name),
                          reverse=True)[:newest]
        selected.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return selected


def _start_locked(root: Path, takes: list[str] | None, newest: int, *,
                  video_type: str | None = None, broll_sources: list[str] | None = None) -> Path:
    inbox = root / "inbox"
    used = _used(inbox)
    selected = _select(root, _inbox_takes(inbox), used, takes, newest)
    kinds = [_kind(source) for source in selected]
    if len(set(kinds)) > 1:
        if takes:
            raise ValueError("Audio memos and video takes cannot share a project. Name only "
                             "memos for a voiceover, or only video takes.")
        memos = ", ".join(p.name for p, kind in zip(selected, kinds, strict=True)
                          if kind == "memo")
        videos = ", ".join(p.name for p, kind in zip(selected, kinds, strict=True)
                           if kind == "take")
        raise ValueError(f"The {newest} newest unused files mix audio memos ({memos}) and "
                         f"video takes ({videos}). Name the files for this project: "
                         "python3 new_project.py FILE ...")
    engine = _engine()
    brand_fonts = root / "brand" / "fonts"
    if brand_fonts.exists():
        # Import the creator's own fonts before any project folder is made.
        subprocess.run(
            [engine, "brand", "fonts", "add", str(brand_fonts), "--replace"],
            cwd=root, check=True,
        )
    brand_words = root / "brand" / "words.txt"
    if brand_words.exists():
        subprocess.run([engine, "brand", "words", "add", "--file", str(brand_words)],
                       cwd=root, check=True)
    base = datetime.now().strftime("%Y-%m-%d") + "-" + _slug(selected[0].stem)
    destination = _reserve(root / "projects", base)
    try:
        copies = destination / "takes"
        copies.mkdir()
        identities = []
        for source in selected:
            before = source.stat()
            shutil.copy2(source, copies / source.name)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"{source.name} changed while it was being copied; wait "
                                 "until it finishes saving, then try again")
            identities.append((source, before.st_size, before.st_mtime_ns))
        command = [engine, "init", str(destination)]
        for source in selected:
            command.extend(["--voiceover" if kinds[0] == "memo" else "--take",
                            str(copies / source.name)])
        if video_type is not None:
            command.extend(["--video-type", video_type])
        for source in broll_sources or []:
            command.extend(["--broll", source])
        if (root / "style/answers.json").is_file():
            command.extend(["--answers", "style/answers.json"])
        subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
        now = datetime.now().astimezone().isoformat()
        for source, size, mtime_ns in identities:
            used["takes"].append({"file": source.name, "bytes": size,
                                  "mtime_ns": mtime_ns,
                                  "project": f"projects/{destination.name}", "used_at": now})
        _write_json(inbox / ".used.json", used)
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def add_pickups(root: Path, directory: Path, memos: list[str] | None, newest: int) -> Path:
    """Append copied inbox memos to a voiceover project under the project lock."""
    with _project_lock(root / "projects"):
        directory = Path(directory).absolute()
        if (directory.parent.resolve() != (root / "projects").resolve()
                or directory.is_symlink() or not (directory / "project.json").is_file()):
            raise ValueError("DIR must be a project directly inside projects/ with project.json.")
        project = json.loads((directory / "project.json").read_text())
        takes = project.get("takes", [])
        if not takes or not all(take.get("voiceover") is True for take in takes):
            raise ValueError(f"{directory} is not a voiceover project. Pickups need a project "
                             "started from an audio memo.")
        inbox = root / "inbox"
        used = _used(inbox)
        available = _inbox_takes(inbox)
        if not memos:
            available = [path for path in _unused_takes(available, used) if _kind(path) == "memo"]
        selected = _select(root, available, used, memos, newest)
        for source in selected:
            if _kind(source) != "memo":
                raise ValueError(f"{source.name} is a video take. Only audio memos can be "
                                 "added to a voiceover project.")
        copies = []
        identities = []
        manifest = directory / "project.json"
        standins = directory / ".yap/voiceover"
        snapshot = None
        try:
            folder = directory / "takes"
            folder.mkdir(exist_ok=True)
            for source in selected:
                target = folder / source.name
                number = 2
                while target.exists() or target.is_symlink():
                    target = folder / f"{source.stem}-{number}{source.suffix}"
                    number += 1
                before = source.stat()
                copies.append(target)
                shutil.copy2(source, target)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"{source.name} changed while it was being copied; wait "
                                     "until it finishes saving, then try again")
                identities.append((source, before.st_size, before.st_mtime_ns))
            command = [_engine(), "init", str(directory)]
            for target in copies:
                command.extend(["--voiceover", str(target)])
            snapshot = (manifest.read_bytes(), {path for path in standins.rglob("*")
                                               if path.is_file()})
            subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
            now = datetime.now().astimezone().isoformat()
            for source, size, mtime_ns in identities:
                used["takes"].append({"file": source.name, "bytes": size, "mtime_ns": mtime_ns,
                                      "project": f"projects/{directory.name}", "used_at": now})
            _write_json(inbox / ".used.json", used)
        except Exception:
            try:
                if snapshot is not None:
                    project_bytes, original_standins = snapshot
                    if not manifest.is_file() or manifest.read_bytes() != project_bytes:
                        manifest.write_bytes(project_bytes)
                    for path in standins.rglob("*"):
                        if path.is_file() and path not in original_standins:
                            path.unlink()
            finally:
                for target in copies:
                    target.unlink(missing_ok=True)
            raise
        return directory


def rename_project(root: Path, directory: Path, topic: str) -> Path:
    """Set the topic name before the story cut and keep the ledger in sync."""
    with _project_lock(root / "projects"):
        return _rename_project_locked(root, directory, topic)


def _rename_project_locked(root: Path, directory: Path, topic: str) -> Path:
    projects = root / "projects"
    directory = Path(directory).absolute()
    if (directory.parent.resolve() != projects.resolve() or directory.is_symlink()
            or not (directory / "project.json").is_file()):
        raise ValueError("DIR must be a project directly inside projects/ with project.json.")
    if (directory / "cut").exists() or (directory / "edit.json").exists():
        raise ValueError("Name the project before writing the story cut or edit.json.")
    match = re.match(r"^(\d{4}-\d{2}-\d{2})-", directory.name)
    if not match:
        raise ValueError("DIR must start with a YYYY-MM-DD date.")
    base = match.group(1) + "-" + _slug(topic)
    used = _used(root / "inbox")
    if directory.name == base or re.fullmatch(re.escape(base) + r"-\d+", directory.name):
        return directory
    project_bytes = (directory / "project.json").read_bytes()
    project = json.loads(project_bytes)
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise ValueError("project.json has no valid name.")
    destination = projects / base
    for number in range(2, 1001):
        if not destination.exists():
            break
        destination = projects / f"{base}-{number}"
    else:
        raise ValueError("Could not find a fresh project name.")
    directory.rename(destination)
    changed_items = []
    for item in used["takes"]:
        if item["project"] == f"projects/{directory.name}":
            item["project"] = f"projects/{destination.name}"
            changed_items.append(item)
    try:
        if changed_items:
            _write_json(root / "inbox/.used.json", used)
    except Exception:
        destination.rename(directory)
        raise
    project["name"] = destination.name
    try:
        _write_json(destination / "project.json", project)
    except Exception:
        try:
            if changed_items:
                for item in changed_items:
                    item["project"] = f"projects/{directory.name}"
                _write_json(root / "inbox/.used.json", used)
            (destination / "project.json").write_bytes(project_bytes)
        finally:
            destination.rename(directory)
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("takes", nargs="*")
    parser.add_argument("--newest", type=int, default=1)
    parser.add_argument("--name", nargs=2, metavar=("DIR", "TOPIC"))
    parser.add_argument("--add", type=Path, metavar="DIR")
    parser.add_argument("--type", choices=VIDEO_TYPES)
    parser.add_argument("--broll", action="append", choices=BROLL_SOURCES)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.add and args.name:
            raise ValueError("Do not combine --add and --name.")
        if (args.name or args.add) and (args.type is not None or args.broll is not None):
            raise ValueError("Do not combine --type or --broll with --name or --add.")
        if args.name:
            if args.takes:
                raise ValueError("Do not pass take names with --name.")
            destination = rename_project(ROOT, Path(args.name[0]), args.name[1])
        elif args.add:
            destination = add_pickups(ROOT, args.add, args.takes, args.newest)
        else:
            destination = start(ROOT, args.takes, args.newest,
                                video_type=args.type, broll_sources=args.broll)
        if args.json:
            names = []
            if not args.name:
                project = json.loads((destination / "project.json").read_text())
                names = [item["file"] for item in _used(ROOT / "inbox")["takes"]
                         if item["project"] == f"projects/{destination.name}"]
            result = {"project": str(destination), "takes": names,
                      "next": (f"yap-studio ingest {destination}" if not args.name else
                               f"yap-studio status {destination}")}
            if not args.name:
                result["voiceover"] = all(take.get("voiceover") is True
                                          for take in project["takes"])
            if not args.name and not args.add:
                result["video_type"] = project.get("video_type")
                result["broll_sources"] = project.get("broll_sources")
            print(json.dumps(result))
        else:
            print(destination)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        action = "name" if args.name else "start"
        print(f"Could not {action} the project: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
