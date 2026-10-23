"""Shared synthetic media fixtures."""

from __future__ import annotations

import fcntl
import json
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

_TESTS = Path(__file__).parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import synth  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_product_feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feedback tests may use a local server and a gh executable inside their temp folder."""
    from yap_studio import product_feedback

    monkeypatch.delenv("YAP_STUDIO_FEEDBACK_ENDPOINT", raising=False)
    monkeypatch.delenv("YAP_STUDIO_FEEDBACK_AUTO_SEND", raising=False)
    run = product_feedback._gh

    def local_gh(*arguments: str) -> str:
        executable = shutil.which("gh")
        if executable is None or not Path(executable).resolve().is_relative_to(tmp_path.resolve()):
            raise OSError("Real GitHub is disabled in tests")
        return run(*arguments)

    monkeypatch.setattr(product_feedback, "_gh", local_gh)


@pytest.fixture(autouse=True)
def _no_real_homebrew(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio import tools

    monkeypatch.setattr(tools, "HOMEBREW_BIN_DIRS", (str(tmp_path / "no-homebrew"),))


@pytest.fixture
def node_bins(tmp_path: Path) -> Callable[..., Path]:
    """Build a Node/npm pair without using a system Node installation."""
    def build(folder: str, version: str = "v26.1.0", *, npm: bool = True,
              returncode: int = 0) -> Path:
        directory = tmp_path / folder
        directory.mkdir(parents=True, exist_ok=True)
        node = directory / "node"
        node.write_text(
            f"#!{sys.executable}\nimport sys\n"
            f"print({version!r} if sys.argv[1:] == ['--version'] else {str(node)!r})\n"
            f"sys.exit({returncode})\n")
        node.chmod(0o755)
        if npm:
            binary = directory / "npm"
            binary.write_text("#!/usr/bin/env node\n")
            binary.chmod(0o755)
        return node

    return build


@pytest.fixture
def _no_real_node(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if request.node.get_closest_marker("render"):
        yield
        return

    from yap_studio import tools

    node = node_bins("default-node")
    monkeypatch.setenv("PATH", str(node.parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr(tools, "NODE_INSTALL_PLACES", (str(node.parent / "off-path"),))
    tools._find_node.cache_clear()
    yield
    tools._find_node.cache_clear()


_FolderSnapshot = dict[Path, dict[str, tuple[int, int, int]] | None]
_OWNER_ROOTS = pytest.StashKey[tuple[Path, ...]]()
_OWNER_SNAPSHOT = pytest.StashKey[_FolderSnapshot]()
_OWNER_LAST_SNAPSHOT = pytest.StashKey[_FolderSnapshot]()
_OWNER_AUDIT_INSTALLED = False
_OWNER_AUDIT_ROOTS: tuple[str, ...] = ()
_OWNER_CURRENT_NODEID: str | None = None
_OWNER_AUDIT_WRITES: dict[str, set[tuple[str, str]]] = {}
_OWNER_AUDIT_CANDIDATES: dict[str, dict[tuple[str, str], tuple[int, int, int] | None]] = {}
_OWNER_FOLDER_CHANGES: list[tuple[str, str, int, str]] = []
_OWNER_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_OWNER_WRITE_PATHS = {
    "open": ((0, None),),
    "os.rename": ((0, 2), (1, 3)),
    "os.replace": ((0, 2), (1, 3)),
    "os.mkdir": ((0, 2),),
    "os.remove": ((0, 1),),
    "os.rmdir": ((0, 1),),
    "os.symlink": ((1, 2),),
    "os.link": ((1, 3),),
    "os.truncate": ((0, None),),
    "os.utime": ((0, 3),),
    "os.chmod": ((0, 2),),
    "shutil.rmtree": ((0, 1),),
    "shutil.move": ((1, None),),
}


def _owner_path_state(path: str) -> tuple[int, int, int] | None:
    """Read one candidate's metadata without following links."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return stat.S_IFMT(info.st_mode), info.st_mtime_ns, info.st_size


def _audit_owner_write(event: str, args: tuple) -> None:
    """Record write attempts in this process without interrupting the writer."""
    targets = _OWNER_WRITE_PATHS.get(event)
    nodeid = _OWNER_CURRENT_NODEID
    if targets is None or nodeid is None:
        return
    try:
        if event == "open":
            mode, flags = args[1:3]
            if mode is not None:
                if not any(char in mode for char in "wax+"):
                    return
            elif not flags & _OWNER_WRITE_FLAGS:
                return
        for index, fd_index in targets:
            if isinstance(args[index], int):
                continue
            path = os.fsdecode(args[index])
            relative = not os.path.isabs(path)
            candidate = event == "open" and args[1] is None and relative
            dir_fd = args[fd_index] if fd_index is not None and len(args) > fd_index else None
            # open omits dir_fd, so even descriptor-relative opens use the working directory.
            if relative and dir_fd not in (None, -1):
                try:
                    if hasattr(fcntl, "F_GETPATH"):
                        directory = fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024))
                        directory = directory.split(b"\0", 1)[0]
                    else:
                        directory = os.readlink(f"/proc/self/fd/{dir_fd}")
                except (OSError, TypeError, ValueError):
                    continue
                path = os.path.join(os.fsdecode(directory), path)
            if event in (
                "os.rename", "os.replace", "os.mkdir", "os.remove", "os.rmdir",
                "os.symlink", "os.link", "shutil.rmtree",
            ):
                # Directory-entry changes affect the parent, not a symlink's target.
                path = os.path.abspath(path)
                path = os.path.join(
                    os.path.realpath(os.path.dirname(path)), os.path.basename(path),
                )
            else:
                path = os.path.realpath(path)
            if any(path == root or path.startswith(root + os.sep) for root in _OWNER_AUDIT_ROOTS):
                if candidate:
                    candidates = _OWNER_AUDIT_CANDIDATES.setdefault(nodeid, {})
                    write = (event, path)
                    if write not in candidates:
                        candidates[write] = _owner_path_state(path)
                else:
                    _OWNER_AUDIT_WRITES.setdefault(nodeid, set()).add((event, path))
    except (OSError, TypeError, ValueError, IndexError):
        return


def _account_home() -> Path:
    """Return the account's home independently of fixture overrides."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _owner_roots() -> tuple[Path, ...]:
    """Resolve the owner's stores without keeping any environment changes."""
    from yap_studio.cache import cache_root
    from yap_studio.review.store import data_dir
    from yap_studio.style.library import config_root

    with pytest.MonkeyPatch.context() as patch:
        for name in ("YAP_STUDIO_CACHE", "YAP_STUDIO_CONFIG", "YAP_STUDIO_DATA_DIR"):
            patch.delenv(name, raising=False)
        patch.setenv("HOME", str(_account_home()))
        return tuple(dict.fromkeys(root.absolute() for root in (
            cache_root(), config_root(), data_dir(),
        )))


def pytest_configure(config: pytest.Config) -> None:
    """Resolve the owner's stores before fixtures change the environment."""
    global _OWNER_AUDIT_INSTALLED, _OWNER_AUDIT_ROOTS, _OWNER_CURRENT_NODEID

    config.stash[_OWNER_ROOTS] = _owner_roots()
    _OWNER_AUDIT_ROOTS = tuple(os.path.realpath(root) for root in config.stash[_OWNER_ROOTS])
    _OWNER_CURRENT_NODEID = None
    _OWNER_AUDIT_WRITES.clear()
    _OWNER_AUDIT_CANDIDATES.clear()
    _OWNER_FOLDER_CHANGES.clear()
    if not _OWNER_AUDIT_INSTALLED:
        sys.addaudithook(_audit_owner_write)
        _OWNER_AUDIT_INSTALLED = True


def _snapshot_owner_folders(roots: Sequence[Path]) -> _FolderSnapshot:
    """Read folder metadata without following links or scanning node_modules."""
    snapshot: _FolderSnapshot = {}
    for root in roots:
        root_path = os.fspath(root)
        try:
            info = os.lstat(root_path)
        except OSError:
            snapshot[root] = None
            continue
        entries = {root_path: (stat.S_IFMT(info.st_mode), info.st_mtime_ns, info.st_size)}
        pending = [root_path] if (
            stat.S_ISDIR(info.st_mode) and os.path.basename(root_path) != "node_modules"
        ) else []
        while pending:
            try:
                with os.scandir(pending.pop()) as children:
                    for entry in children:
                        try:
                            info = entry.stat(follow_symlinks=False)
                            descend = (entry.name != "node_modules"
                                       and entry.is_dir(follow_symlinks=False))
                        except OSError:
                            continue
                        entries[entry.path] = (
                            stat.S_IFMT(info.st_mode), info.st_mtime_ns, info.st_size,
                        )
                        if descend:
                            pending.append(entry.path)
            except OSError:
                pass
        snapshot[root] = entries
    return snapshot


def _owner_folder_changes(
    before: _FolderSnapshot, after: _FolderSnapshot,
) -> list[tuple[str, Path, int]]:
    """List added, removed and modified entries using their metadata."""
    changes = []
    for root in sorted(before.keys() | after.keys()):
        old, new = before.get(root) or {}, after.get(root) or {}
        for path in sorted(old.keys() | new.keys()):
            if path not in old:
                changes.append(("added", Path(path), new[path][1]))
            elif path not in new:
                changes.append(("removed", Path(path), old[path][1]))
            elif old[path] != new[path]:
                changes.append(("changed", Path(path), new[path][1]))
    return changes


def _promote_owner_candidates(nodeid: str) -> set[tuple[str, str]]:
    """Confirm relative open candidates against their first recorded state."""
    candidates = _OWNER_AUDIT_CANDIDATES.pop(nodeid, {})
    return {write for write, before in candidates.items() if _owner_path_state(write[1]) != before}


def _owner_write_message(nodeid: str, writes: set[tuple[str, str]]) -> str:
    """Name this test's write events and paths."""
    lines = [f"{nodeid} wrote to the owner's real Yap Studio folders:"]
    lines.extend(f"{event} {path}" for event, path in sorted(writes)[:20])
    if len(writes) > 20:
        lines.append(f"... and {len(writes) - 20} more")
    lines.append("Tests must use the folders conftest sets.")
    return "\n".join(lines)


def _owner_folder_summary(entries: list[tuple[str, str, int, str]]) -> list[str]:
    """Group observed changes with the tests running at the time."""
    grouped: dict[tuple[str, str, int], set[str]] = {}
    for kind, path, mtime, nodeid in entries:
        grouped.setdefault((kind, path, mtime), set()).add(nodeid)
    lines = []
    for (kind, path, mtime), nodeids in sorted(grouped.items())[:30]:
        local = time.strftime("%H:%M:%S", time.localtime(mtime // 1_000_000_000))
        running = ", ".join(sorted(nodeids)[:5])
        if len(nodeids) > 5:
            running += f", +{len(nodeids) - 5} more"
        lines.append(f"{kind} {path} (mtime {local}) while running: {running}")
    if len(grouped) > 30:
        lines.append(f"... and {len(grouped) - 30} more")
    lines.append("Not a test failure; another process (a review server, another test run) may "
                 "have written them; tests that write there themselves fail on their own.")
    return lines


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Collect worker observations from teardown reports on the controller."""
    if report.when != "teardown":
        return
    for name, value in report.user_properties:
        if name == "yap_owner_folder_changes":
            for change in json.loads(value):
                _OWNER_FOLDER_CHANGES.append((
                    change["kind"], change["path"], change["mtime_ns"], report.nodeid,
                ))


def pytest_terminal_summary(terminalreporter) -> None:
    """Show observed changes once without turning them into test failures."""
    if _OWNER_FOLDER_CHANGES:
        terminalreporter.write_sep("=", "owner's real Yap Studio folders changed during this run")
        for line in _owner_folder_summary(_OWNER_FOLDER_CHANGES):
            terminalreporter.write_line(line)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    """Fail a test that outlives YAP_STUDIO_TEST_TIMEOUT seconds, naming where it waited."""
    limit = int(os.environ.get("YAP_STUDIO_TEST_TIMEOUT") or 0)
    if limit <= 0 or not hasattr(signal, "SIGALRM"):
        return (yield)

    def expired(_signum, _frame):
        raise TimeoutError(f"{item.nodeid} ran longer than {limit} s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(limit)
    try:
        return (yield)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> Iterator[None]:
    """Start the real-folder check before fixtures of any scope run."""
    global _OWNER_CURRENT_NODEID

    _OWNER_CURRENT_NODEID = item.nodeid
    before = item.config.stash.get(_OWNER_LAST_SNAPSHOT, None)
    if before is None:
        before = _snapshot_owner_folders(item.config.stash[_OWNER_ROOTS])
    item.stash[_OWNER_SNAPSHOT] = before
    yield


def _finish_detached_group(pid: int, identifier: str | Path, marker: str) -> None:
    """Finish a previously verified session without signaling a reused leader."""
    from yap_studio import background
    from yap_studio.project.lock import pid_alive
    from yap_studio.stages import run_detach

    deadline = time.monotonic() + 1
    while run_detach._group_running(pid) is True and time.monotonic() < deadline:
        time.sleep(0.05)
    if run_detach._group_running(pid) is True:
        if pid_alive(pid) and background.command_matches(pid, identifier, marker) is not True:
            raise AssertionError(f"cannot safely stop detached process group {pid}")
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while run_detach._group_running(pid) is True and time.monotonic() < deadline:
        time.sleep(0.05)
    assert run_detach._group_running(pid) is False, f"detached process group {pid} remains"


def _stop_detached_page(path: Path, saved: dict) -> None:
    """Use the page's ownership check, then finish its private process group."""
    from yap_studio import background

    pid = saved.get("pid")
    if type(pid) is not int or background.command_matches(pid, path) is not True:
        return
    try:
        private_group = os.getpgid(pid) == pid
    except ProcessLookupError:
        return
    background.stop_record(path, saved)
    if private_group:
        _finish_detached_group(pid, path, "--detached-record")
    elif background.command_matches(pid, path) is True:
        os.kill(pid, signal.SIGKILL)
    assert background.command_matches(pid, path) is False, f"detached page {pid} remains"
    background.remove_record(path)


def _detached_processes(
    roots: set[Path], data: Path | None,
) -> Iterator[tuple[int, Path, str, str]]:
    """Find private sessions whose marker paths belong to this test."""
    listing = subprocess.run(["ps", "-axo", "pid=,pgid=,command="], capture_output=True,
                             text=True, timeout=5, check=True)
    roots = {root.resolve() for root in roots}
    page_roots = roots | ({data.resolve()} if data is not None else set())
    for line in listing.stdout.splitlines():
        fields = line.split(None, 2)
        if (len(fields) != 3 or not fields[0].isdigit() or fields[0] != fields[1]
                or int(fields[0]) <= 0):
            continue
        pid, command = int(fields[0]), fields[2].rstrip()
        job = re.search(r"(?<!\S)--take-job(?:=|\s+)(.+?)(?=\s+--|$)", command)
        if job:
            path = Path(job[1])
            if path.is_absolute() and any(path.resolve().is_relative_to(root) for root in roots):
                yield pid, path, job[1], "--take-job"
            continue
        record = re.search(r"(?<!\S)--detached-record(?:=|\s+)(.+?)(?=\s+--|$)", command)
        if record:
            path = Path(record[1])
            if path.is_absolute() and any(path.resolve().is_relative_to(root)
                                          for root in page_roots):
                yield pid, path, record[1], "--detached-record"
            continue
        run = re.search(r"(?<!\S)--detached-run(?:=|\s+)(\S+)", command)
        project = re.search(r"(?<!\S)-m\s+yap_studio\s+run\s+(.+?)(?=\s+--|$)", command)
        if run and project:
            root = Path(project[1])
            if root.is_absolute() and any(root.resolve().is_relative_to(scope) for scope in roots):
                yield pid, root / ".yap/run/state.json", run[1], "--detached-run"


def _check_detached_processes(nodeid: str, roots: set[Path], data: Path | None) -> None:
    """Scan only this test's folders, without following shared cache symlinks."""
    from yap_studio import background
    from yap_studio.stages import run_detach

    records: set[Path] = set()
    detached_folder = False
    if data is not None:
        detached_folder = (data / "detached").is_dir()
        records.update((data / "detached").glob("*.json"))
    for root in roots:
        for directory, _folders, files in os.walk(root):
            folder = Path(directory)
            if folder.name == "detached":
                detached_folder = True
                records.update(folder / name for name in files if name.endswith(".json"))
            elif folder.parts[-2:] == ("transcripts", "jobs"):
                detached_folder = True
                records.update(folder / name for name in files if name.endswith(".json"))
            elif (folder.parts[-2:] == (".yap", "run")
                  and ("run.log" in files or "state.json" in files)):
                detached_folder = True
                if "state.json" in files:
                    records.add(folder / "state.json")
    leaks = []
    reported: set[int] = set()
    for path in sorted(records):
        saved = background.read_record(path)
        if saved is None or type(saved.get("pid")) is not int:
            continue
        pid = saved["pid"]
        try:
            if path.parent.parts[-2:] == ("transcripts", "jobs"):
                if background.command_matches(pid, path, "--take-job") is not True:
                    continue
                reported.add(pid)
                leaks.append(f"{path} (pid {pid})")
                if os.getpgid(pid) != pid:
                    raise AssertionError(f"take job {pid} has no private process group")
                os.killpg(pid, signal.SIGTERM)
                _finish_detached_group(pid, path, "--take-job")
                background.remove_record(path)
            elif path.name == "state.json" and path.parent.name == "run":
                if not run_detach.live(saved):
                    continue
                reported.add(pid)
                leaks.append(f"{path} (pid {pid})")
                # Never let a malformed state signal the test runner's own group.
                if os.getpgid(pid) != pid:
                    raise AssertionError(f"detached run {pid} has no private process group")
                run_detach.stop(path.parents[2], json_output=True)
                _finish_detached_group(pid, saved["id"], "--detached-run")
            elif background.command_matches(pid, path) is True:
                reported.add(pid)
                leaks.append(f"{path} (pid {pid})")
                _stop_detached_page(path, saved)
        except Exception as exc:
            leaks.append(f"{path} (pid {pid}): cleanup failed: {exc}")
    # Only tests with detached folders pay for a process inventory. A log can
    # outlive its record, so check argv even when the folder contains no JSON.
    if detached_folder:
        for pid, path, identifier, marker in _detached_processes(roots, data):
            if pid in reported or background.command_matches(pid, identifier, marker) is not True:
                continue
            try:
                if os.getpgid(pid) != pid:
                    continue
                missing = " (no record)" if not path.exists() else ""
                leaks.append(f"{path}{missing} (pid {pid})")
                if marker == "--detached-record":
                    _stop_detached_page(path, {"pid": pid})
                else:
                    os.killpg(pid, signal.SIGTERM)
                    _finish_detached_group(pid, identifier, marker)
            except ProcessLookupError:
                pass
            except Exception as exc:
                leaks.append(f"{path} (pid {pid}): cleanup failed: {exc}")
    if leaks:
        pytest.fail(f"{nodeid} left detached processes running:\n" + "\n".join(leaks))


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item) -> Iterator[None]:
    """Automatically check leftovers after every fixture and monkeypatch teardown."""
    global _OWNER_CURRENT_NODEID

    roots = {item.funcargs[name] for name in ("_owner_folders", "tmp_path")
             if name in item.funcargs}
    # Capture overrides before monkeypatch restores the owner's data directory.
    data = os.environ.get("YAP_STUDIO_DATA_DIR")
    try:
        yield
    finally:
        try:
            after = _snapshot_owner_folders(item.config.stash[_OWNER_ROOTS])
            item.config.stash[_OWNER_LAST_SNAPSHOT] = after
            failures = []
            try:
                _check_detached_processes(item.nodeid, roots, Path(data) if data else None)
            except pytest.fail.Exception as exc:
                failures.append(str(exc))
            folder_changes = _owner_folder_changes(item.stash[_OWNER_SNAPSHOT], after)
            writes = _OWNER_AUDIT_WRITES.pop(item.nodeid, set())
            writes.update(_promote_owner_candidates(item.nodeid))
            written_paths = {path for _event, path in writes}
            changes = [
                {"kind": kind, "path": str(path), "mtime_ns": mtime}
                for kind, path, mtime in folder_changes
                if not written_paths or os.path.realpath(path) not in written_paths
            ]
            if changes:
                item.user_properties.append(("yap_owner_folder_changes", json.dumps(changes)))
            if writes:
                failures.append(_owner_write_message(item.nodeid, writes))
            if failures:
                pytest.fail("\n\n".join(failures))
        finally:
            _OWNER_CURRENT_NODEID = None


class SynthMedia:
    """Build named synthetic media lazily and cache it for the test session."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._paths: dict[str, Path] = {}

    @property
    def spans(self) -> list[tuple[float, float]]:
        return synth.default_spans()

    def cached(
        self,
        key: str,
        build: Callable[[Path], object],
        suffix: str = ".mp4",
    ) -> Path:
        if key not in self._paths:
            out = self.root / f"{key}{suffix}"
            build(out)
            if not out.is_file():
                raise AssertionError(f"synthetic fixture builder did not create {out}")
            self._paths[key] = out
        return self._paths[key]

    def path(self, name: str) -> Path:
        """Return a named fixture, building it once on first use."""
        builders: dict[str, tuple[str, Callable[[Path], Any]]] = {
            "calib": (".mov", lambda out: synth.calib_source(out)),
            "calib_vertical": (
                ".mov",
                lambda out: synth.calib_source(out, size="180x320"),
            ),
            "clean": (
                ".mp4",
                lambda out: synth.render_clean(self.path("calib"), self.spans, out),
            ),
            "perclip": (
                ".mp4",
                lambda out: synth.render_perclip_aac(self.path("calib"), self.spans, out),
            ),
            "perclip_click3": (
                ".mp4",
                lambda out: synth.render_perclip_aac(
                    self.path("calib"), self.spans, out, no_fade_cut=3
                ),
            ),
            "clean_shift_samples": (
                ".mp4",
                lambda out: synth.shift_audio(self.path("clean"), out, 100, "samples"),
            ),
            "clean_shift_start": (
                ".mp4",
                lambda out: synth.shift_audio(self.path("clean"), out, 100, "start_time"),
            ),
            "crop_center": (
                ".mov",
                lambda out: synth.reframe(self.path("calib"), out, "crop", 0.5),
            ),
            "crop_offcenter": (
                ".mov",
                lambda out: synth.reframe(self.path("calib"), out, "crop", 0.2),
            ),
            "clean_crop": (
                ".mp4",
                lambda out: synth.reframe(self.path("clean"), out, "crop", 0.5),
            ),
            "perclip_click3_crop": (
                ".mp4",
                lambda out: synth.reframe(self.path("perclip_click3"), out, "crop", 0.5),
            ),
            "fit_vertical": (
                ".mov",
                lambda out: synth.reframe(self.path("calib_vertical"), out, "fit"),
            ),
            "rotated_vertical": (
                ".mov",
                lambda out: synth.rotate_metadata(self.path("calib_vertical"), out, 90),
            ),
            "slowpan": (".mov", lambda out: synth.slowpan_source(out)),
            # Each audio_* render writes its uncompressed reference beside it as .wav.
            "audio_gap": (
                ".mp4",
                lambda out: synth.audio_gap_render(out, out.with_suffix(".wav")),
            ),
            "audio_clean_priming": (
                ".mp4",
                lambda out: synth.audio_gap_render(out, out.with_suffix(".wav"), gap_samples=0),
            ),
            "audio_gap_long_picture": (
                ".mp4",
                lambda out: synth.audio_gap_render(
                    out, out.with_suffix(".wav"), video_seconds=4.5
                ),
            ),
        }
        if name == "spans":
            raise TypeError("spans is available as synth_media.spans, not as a path")
        if name not in builders:
            raise KeyError(f"unknown synthetic fixture: {name}")
        suffix, build = builders[name]
        return self.cached(name, build, suffix=suffix)


@pytest.fixture(scope="session")
def synth_media(tmp_path_factory: pytest.TempPathFactory) -> SynthMedia:
    """Provide session-cached synthetic media."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        pytest.fail(f"required media tool missing: {', '.join(missing)}")
    return SynthMedia(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def aac_tail(synth_media: SynthMedia) -> int:
    """Encoder padding this ffmpeg still decodes after the 4 s synthetic AAC encode.

    Builds that trim decoding to the MP4 edit list return exactly the encoded
    samples, so the tail is 0. Builds that skip only the priming keep the padding
    in the last AAC frame (Ubuntu's ffmpeg 6.1 keeps 512 samples here).
    """
    from yap_studio import audio

    tail = audio.timeline(synth_media.path("audio_clean_priming")).decoded_samples - 4 * 48_000
    assert 0 <= tail < 1024, tail
    return tail


_SHARD_CHUNK = 40


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Shard selected modules and keep speech and renderer tests opt-in."""
    shard = os.environ.get("YAP_STUDIO_TEST_SHARD")
    if shard is not None:
        if not re.fullmatch(r"[0-9]+/[0-9]+", shard):
            raise pytest.UsageError("YAP_STUDIO_TEST_SHARD must be i/n with 1 <= i <= n.")
        index, count = map(int, shard.split("/"))
        if not 1 <= index <= count:
            raise pytest.UsageError("YAP_STUDIO_TEST_SHARD must be i/n with 1 <= i <= n.")
        modules: dict[Path, list[pytest.Item]] = {}
        for item in items:
            modules.setdefault(item.path, []).append(item)
        # Big modules split into blocks so no single module outlasts a shard's time limit.
        chunks = {}
        for path, members in modules.items():
            members = sorted(members, key=lambda item: item.nodeid)
            for start in range(0, len(members), _SHARD_CHUNK):
                chunks[(path, start)] = members[start:start + _SHARD_CHUNK]
        sizes = [0] * count
        assigned = {}
        for key in sorted(chunks, key=lambda key: (-len(chunks[key]), key)):
            lightest = min(range(count), key=lambda number: sizes[number])
            for item in chunks[key]:
                assigned[item.nodeid] = lightest + 1
            sizes[lightest] += len(chunks[key])
        deselected = [item for item in items if assigned[item.nodeid] != index]
        items[:] = [item for item in items if assigned[item.nodeid] == index]
        config.hook.pytest_deselected(items=deselected)

    options = (("asr", "YAP_STUDIO_ASR_TESTS"), ("render", "YAP_STUDIO_RENDER_TESTS"))
    for marker, variable in options:
        if os.environ.get(variable) == "1":
            continue
        skip = pytest.mark.skip(reason=f"set {variable}=1 to run {marker} tests")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _owner_folders(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    _shared_helpers: Path, _shared_renderer_install: Path | None,
    _shared_cache_root: Path, _npm_cache: Path,
) -> Iterator[Path]:
    """Isolate creator stores and check HOME outside the render and speech suites."""
    from yap_studio import tools
    from yap_studio.render import hyperframes

    owner = tmp_path_factory.mktemp("owner")
    shared = _prepare_owner_cache(
        owner, _shared_helpers, _shared_renderer_install, _shared_cache_root,
    )

    # A test may undo its renderer patches. Keep the folder guard independent.
    with monkeypatch.context() as folders:
        folders.setenv("YAP_STUDIO_DATA_DIR", str(owner / "data"))
        folders.setenv("YAP_STUDIO_CONFIG", str(owner / "config"))
        folders.setenv("YAP_STUDIO_CACHE", str(owner / "cache"))

        userconfig = owner / "npmrc"
        userconfig.touch()
        for name, value in {
            "npm_config_cache": str(_npm_cache),
            "npm_config_userconfig": str(userconfig),
            "npm_config_update_notifier": "false",
            "npm_config_logs_dir": str(owner / "npm-logs"),
        }.items():
            folders.delenv(name.upper(), raising=False)
            folders.setenv(name, value)
        for name, directory in {
            "PUPPETEER_CACHE_DIR": "puppeteer",
            "NODE_COMPILE_CACHE": "node",
            "COREPACK_HOME": "corepack",
            "XDG_CACHE_HOME": "xdg",
            "XDG_CONFIG_HOME": "config",
            "XDG_STATE_HOME": "state",
            "HYPERFRAMES_FONT_CACHE_DIR": "fonts",
            "HYPERFRAMES_HISTORY_ROOT": "history",
            "HYPERFRAMES_EXTRACT_CACHE_DIR": "extracted-frames",
        }.items():
            folders.setenv(name, str(owner / "cache" / directory))
        folders.setenv("NODE_REPL_HISTORY", str(owner / "node-repl-history"))
        folders.delenv("HYPERFRAMES_BROWSER_PATH", raising=False)
        folders.delenv("PRODUCER_HEADLESS_SHELL_PATH", raising=False)
        metadata = hyperframes._install_metadata(shared)
        if metadata and metadata["browser_path"] and Path(metadata["browser_path"]).exists():
            folders.setenv("HYPERFRAMES_BROWSER_PATH", metadata["browser_path"])

        # HyperFrames 0.8.169 ignores PUPPETEER_CACHE_DIR for its own browser
        # downloads and locks. Isolate HOME for browser setup even in render tests.
        browser_home = owner / "browser-home"
        browser_home.mkdir()
        original = tools.run_tool

        def run_tool(name: str, args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
            if name == "hyperframes" and list(args[:2]) == ["browser", "ensure"]:
                kwargs["env"] = {**(kwargs.get("env") or {}), "HOME": str(browser_home)}
            return original(name, args, **kwargs)

        folders.setattr(tools, "run_tool", run_tool)

        # The matte model directory is hard-coded under HOME. Render and ASR tests
        # retain HOME and skip its leak check to reuse installed browser/model caches.
        # Their data, config and engine cache folders are still isolated.
        if request.node.get_closest_marker("render") or request.node.get_closest_marker("asr"):
            yield owner
            return

        home = owner / "home"
        home.mkdir()
        folders.setenv("HOME", str(home))
        # Subprocess tool caches have their own overrides. Speech model checks are
        # opt-in, so incidental CLI probes must not download models into each test.
        folders.setenv("UV_CACHE_DIR", str(owner / "uv-cache"))
        folders.setenv("HF_HOME", str(owner / "speech-cache"))
        folders.setenv("HF_HUB_CACHE", str(owner / "speech-cache" / "hub"))
        folders.setenv("HF_HUB_OFFLINE", "1")
        # Apple's older Python otherwise caches standard-library bytecode in HOME.
        # Reuse existing bytecode without rebuilding a cold cache per subprocess.
        folders.setenv("PYTHONDONTWRITEBYTECODE", "1")
        yield owner
        leaked = sorted(str(path.relative_to(home)) for path in home.rglob("*"))
        assert not leaked, f"{request.node.nodeid} wrote to its fake HOME:\n" + "\n".join(leaked)


@pytest.fixture(scope="session")
def _shared_cache_root() -> Path:
    """Resolve the real cache once before any function fixture changes HOME."""
    from yap_studio.cache import cache_root

    return cache_root().resolve()


@pytest.fixture(scope="session")
def _shared_helpers(
    tmp_path_factory: pytest.TempPathFactory, _shared_cache_root: Path,
) -> Path:
    """Reuse existing helper executables and compile missing ones once per worker."""
    helpers = tmp_path_factory.mktemp("helpers")
    real_helpers = _shared_cache_root / "helpers"
    if real_helpers.is_dir():
        for entry in real_helpers.iterdir():
            # Compiler temporaries are dotfiles, not published yap- executables.
            if (entry.name.startswith("yap-") and not entry.is_symlink()
                    and entry.is_file() and os.access(entry, os.X_OK)):
                (helpers / entry.name).symlink_to(entry)
    return helpers


@pytest.fixture(scope="session")
def _shared_renderer_install(_shared_cache_root: Path) -> Path | None:
    """Reuse pinned packages only when their metadata, executable and browser are valid."""
    from yap_studio.render import hyperframes

    prefix = _shared_cache_root / "renderer" / f"hyperframes-{hyperframes.HYPERFRAMES_VERSION}"
    return prefix if _renderer_install_failure(prefix) is None else None


def _prepare_owner_cache(
    owner: Path, helpers: Path, install: Path | None, shared_cache: Path,
) -> Path:
    """Reuse shared builds while keeping metadata repairs in temporary folders."""
    from yap_studio.render import hyperframes

    renderer = owner / "cache" / "renderer"
    renderer.mkdir(parents=True)
    shared = install or (
        shared_cache / "renderer" / f"hyperframes-{hyperframes.HYPERFRAMES_VERSION}"
    )
    # A browser repair can finish after the worker's session snapshot.
    # Recheck before each test, without installing or repairing anything in the real cache.
    if _renderer_install_failure(shared) is None:
        # Reuse build artifacts while keeping metadata repairs and new helper builds
        # in temporary folders. Removing a private prefix removes links, not targets.
        _link_renderer_install(shared, renderer)
    (owner / "cache" / "helpers").symlink_to(helpers, target_is_directory=True)
    return shared


@pytest.fixture(scope="session", autouse=True)
def _session_owner_folders(
    tmp_path_factory: pytest.TempPathFactory, _shared_cache_root: Path,
    _shared_helpers: Path, _shared_renderer_install: Path | None,
) -> Iterator[None]:
    """Isolate stores and ElevenLabs credentials before higher-scope fixtures run."""
    owner = tmp_path_factory.mktemp("session-owner")
    _prepare_owner_cache(owner, _shared_helpers, _shared_renderer_install, _shared_cache_root)
    with pytest.MonkeyPatch.context() as folders:
        # Sharding ends at collection; nested pytest probes need their whole suite.
        folders.delenv("YAP_STUDIO_TEST_SHARD", raising=False)
        folders.setenv("YAP_STUDIO_DATA_DIR", str(owner / "data"))
        folders.setenv("YAP_STUDIO_CONFIG", str(owner / "config"))
        folders.setenv("YAP_STUDIO_CACHE", str(owner / "cache"))
        folders.setenv("ELEVENLABS_CONFIG", str(owner / "elevenlabs/config.toml"))
        folders.delenv("ELEVENLABS_API_KEY", raising=False)
        folders.setenv("ELEVENLABS_BASE_URL", "http://127.0.0.1:9")
        yield


def _renderer_install_failure(prefix: Path) -> str | None:
    """Name the check which prevents this prefix from being shared."""
    from yap_studio.render import hyperframes

    metadata = hyperframes._install_metadata(prefix)
    if metadata is None:
        return f"metadata missing or invalid: {prefix / 'install.json'}"
    executable = prefix / "node_modules" / ".bin" / "hyperframes"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return f"executable missing or not executable: {executable}"
    browser = metadata["browser_path"]
    if not browser or not Path(browser).exists():
        return f"browser missing: {browser}"
    return None


def _link_renderer_install(shared: Path, renderer: Path) -> None:
    """Publish private metadata and shared package links in one step."""
    with tempfile.TemporaryDirectory(dir=renderer, prefix=".linked-") as directory:
        staging = Path(directory)
        for entry in shared.iterdir():
            target = staging / entry.name
            if entry.name == "install.json":
                shutil.copyfile(entry, target)
            else:
                target.symlink_to(entry, target_is_directory=entry.is_dir())
        os.replace(staging, renderer / shared.name)


@pytest.fixture(scope="session")
def _npm_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Reuse npm downloads only within this worker's temporary session."""
    return tmp_path_factory.mktemp("npm-cache")


@pytest.fixture(autouse=True)
def _no_edge_engine(
    monkeypatch: pytest.MonkeyPatch, _owner_folders: Path, request: pytest.FixtureRequest,
) -> None:
    """Keep synthetic cuts independent of installed speech tools unless opted in."""
    from yap_studio.project import status as project_status
    from yap_studio.stages import cut

    monkeypatch.setattr(cut, "_edge_backend", lambda context: None)
    monkeypatch.setattr(project_status, "_edge_engine_identity",
                        lambda context: {"engine": "none", "model": None})
    if request.node.get_closest_marker("asr") is None:
        verify = cut.verify
        monkeypatch.setattr(cut, "verify", lambda **kwargs: verify(**{**kwargs, "asr": "none"}))


@pytest.fixture(autouse=True)
def _no_real_osxphotos(
    monkeypatch: pytest.MonkeyPatch, _owner_folders: Path,
) -> Iterator[Callable[[Path], None]]:
    """Allow only the synthetic Photos exporter when a test opts in."""
    from yap_studio import tools

    with monkeypatch.context() as guard:
        original = tools.find_tool
        allowed: set[Path] = set()

        def find(name: str) -> str | None:
            if name != "osxphotos":
                return original(name)
            if not allowed:
                return None
            found = original(name)
            return found if found is not None and Path(found).resolve() in allowed else None

        guard.setattr(tools, "find_tool", find)
        try:
            yield lambda executable: allowed.add(executable.resolve())
        finally:
            # Undo test patches before closing guards on the same attributes.
            monkeypatch.undo()


@pytest.fixture(autouse=True)
def _no_real_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _owner_folders: Path,
) -> Iterator[Callable[[Path], None]]:
    """Require explicit test opt-in before resolving the ElevenLabs executable."""
    from yap_studio import sfx, tools

    # Renderer tests undo their mocks; SFX data and credentials must stay isolated.
    with monkeypatch.context() as guard:
        guard.setenv("ELEVENLABS_CONFIG", str(tmp_path / "elevenlabs/config.toml"))
        guard.delenv("ELEVENLABS_API_KEY", raising=False)
        guard.setenv("ELEVENLABS_BASE_URL", "http://127.0.0.1:9")
        guard.setattr(sfx, "sfx_cache_root", lambda: tmp_path / "sfx-cache")
        original = tools.find_tool
        allowed: set[Path] = set()

        def find(name: str) -> str | None:
            if name != "elevenlabs-pp-cli":
                return original(name)
            found = original(name)
            if found is None or Path(found).resolve() not in allowed:
                return None
            return found

        guard.setattr(tools, "find_tool", find)
        try:
            yield lambda executable: allowed.add(executable.resolve())
        finally:
            monkeypatch.undo()


@pytest.fixture(autouse=True)
def _no_real_pexels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _owner_folders: Path,
) -> Iterator[Callable[[Path], None]]:
    """Require explicit opt-in before resolving the Pexels executable."""
    from yap_studio import tools

    with monkeypatch.context() as guard:
        guard.delenv("PEXELS_API_KEY", raising=False)
        guard.setenv("PEXELS_HOME", str(tmp_path / "pexels"))
        original = tools.find_tool
        allowed: set[Path] = set()

        def find(name: str) -> str | None:
            if name != "pexels-pp-cli":
                return original(name)
            found = original(name)
            return found if found and Path(found).resolve() in allowed else None

        guard.setattr(tools, "find_tool", find)
        try:
            yield lambda executable: allowed.add(executable.resolve())
        finally:
            monkeypatch.undo()


@pytest.fixture
def fake_pexels(monkeypatch, tmp_path, _no_real_pexels):
    executable = Path(__file__).parent / "fakes/pexels-pp-cli"
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    _no_real_pexels(executable)
    log = tmp_path / "pexels.log"
    monkeypatch.setenv("FAKE_PEXELS_LOG", str(log))
    return log


@pytest.fixture(autouse=True)
def _no_real_wavespeed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _owner_folders: Path,
) -> Iterator[Callable[[Path], None]]:
    """Resolve only the synthetic WaveSpeed executable after explicit opt-in."""
    from yap_studio import tools

    with monkeypatch.context() as guard:
        guard.delenv("WAVESPEED_API_KEY", raising=False)
        guard.setenv("WAVESPEED_HOME", str(tmp_path / "wavespeed"))
        original = tools.find_tool
        allowed: set[Path] = set()

        def find(name: str) -> str | None:
            if name != "wavespeed-pp-cli":
                return original(name)
            found = original(name)
            return found if found and Path(found).resolve() in allowed else None

        guard.setattr(tools, "find_tool", find)
        try:
            yield lambda executable: allowed.add(executable.resolve())
        finally:
            monkeypatch.undo()


@pytest.fixture
def fake_wavespeed(monkeypatch, tmp_path, _no_real_wavespeed):
    executable = Path(__file__).parent / "fakes/wavespeed-pp-cli"
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    _no_real_wavespeed(executable)
    log = tmp_path / "wavespeed.log"
    monkeypatch.setenv("FAKE_WAVESPEED_LOG", str(log))
    return log
