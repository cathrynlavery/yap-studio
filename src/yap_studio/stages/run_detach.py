"""Project-local background runs, cancellation, and read-only status."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from yap_studio import background, paths
from yap_studio.errors import InputError, YapError
from yap_studio.project.lock import pid_alive


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def local_clock(started: str | None) -> str:
    """Format a run's start time for local readers."""
    try:
        return datetime.fromisoformat(started).astimezone().strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return "unknown"


def state_path(root: Path) -> Path:
    return root / ".yap/run/state.json"


def commands(root: Path) -> tuple[str, str]:
    quoted = shlex.quote(str(root))
    return f"yap-studio status {quoted} --wait 540", f"yap-studio run {quoted} --stop"


def live(state: dict | None, *, allow_unchecked: bool = False) -> bool:
    """Check ownership, allowing unknown process checks for read-only callers."""
    if not state or state.get("state") != "running" or not pid_alive(state.get("pid")):
        return False
    identifier = state.get("id")
    if not isinstance(identifier, str) or not identifier:
        return False
    matches = background.command_matches(state["pid"], identifier, "--detached-run")
    if matches is None:
        if allow_unchecked:
            return pid_alive(state["pid"])
        raise YapError(f"could not check background run process {state['pid']}; try again",
                       code="run_check_failed")
    return matches


def running_error(root: Path, state: dict | None = None) -> YapError:
    check, stop = commands(root)
    details = []
    if state:
        if state.get("pid") is not None:
            details.append(f"pid {state['pid']}")
        if state.get("started") is not None:
            details.append(f"started {local_clock(state['started'])}")
    detail = f" ({', '.join(details)})" if details else ""
    return YapError(f"a background run is already working on this project{detail}",
                    code="run_running",
                    suggested_action=f"Check it with {check}, or stop it with {stop}.")


@contextmanager
def run_lock(project_dir: Path) -> Iterator[None]:
    """Hold the shared run lock without waiting for another run."""
    root = project_dir.resolve()
    if not (root / "project.json").is_file():
        raise InputError(str(project_dir), "no project.json here; run yap-studio init first")
    if os.name != "posix":
        yield
        return
    import fcntl

    folder = paths.validate_output_dir(root / ".yap/run")
    fd = os.open(folder / "lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as lock:
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = background.read_record(state_path(root))
            if live(state, allow_unchecked=True):
                raise running_error(root, state) from None
            raise YapError("another yap-studio run is already working on this project",
                           code="run_running",
                           suggested_action="Wait for it to finish, then run again.") from None
        yield


def clear_stale(root: Path) -> None:
    """A newer foreground run supersedes an old background result."""
    path = state_path(root)
    if not path.exists():
        return
    try:
        running = live(background.read_record(path))
    except YapError as exc:
        if exc.code != "run_check_failed":
            raise
        return
    if not running:
        path.unlink(missing_ok=True)


def show_step(step: dict) -> None:
    print(f"{step['name']}: {'unchanged, ' if step['state'] == 'unchanged' else ''}"
          f"{step['detail'] if step['state'] != 'failed' else 'failed'}", flush=True)


def child(args: argparse.Namespace) -> int:
    """Hold the run lock and publish each finished step and the terminal result."""
    from yap_studio.stages.run import STEPS, run

    root = Path(args.directory).resolve()
    with run_lock(root):
        folder = root / ".yap/run"
        state = {
            "schema": "yap-studio/run-state@1", "id": args.detached_run,
            "pid": os.getpid(), "state": "running", "started": timestamp(), "finished": None,
            "options": {"note": args.note, "frames": args.frames,
                        "allow_violations": args.allow_violations},
            "step": "ingest", "steps": [], "result": None, "error": None,
            "log": str(folder / "run.log"),
        }

        def save() -> None:
            background.write_json(state_path(root), state)

        def on_step(step: dict) -> None:
            state["steps"].append(step)
            index = STEPS.index(step["name"]) + 1
            state["step"] = (step["name"] if step["state"] == "failed" else
                             STEPS[index] if index < len(STEPS) else None)
            save()
            show_step(step)

        def cancel(_signum: int, _frame: object) -> None:
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, cancel)
        try:
            save()
            result = run(root, note=args.note, allow_violations=args.allow_violations,
                         frames=args.frames, on_step=on_step)
            if result["status"] == "finished" and result.get("version") and not result.get("pilot"):
                from yap_studio.review.detach import ensure_versions

                result["review"] = ensure_versions(root)
                if result["review"]["status"] == "failed":
                    result["next"] = result["review"]["command"]
            state.update(state=result["status"], result=result)
            return result["stopped"]["exit"] if result["stopped"] else 0
        except KeyboardInterrupt:
            # Ignore a repeated stop while recording cancellation and unwinding children.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            state.update(state="cancelled", error={"message": f"stopped by {commands(root)[1]}"})
            return 130
        except Exception as exc:
            if not isinstance(exc, YapError):
                exc = InputError(str(root), f"cannot run the project: {exc}")
            state.update(state="failed", error=exc.to_dict())
            background.print_error(exc, json_output=args.json)
            return 2
        finally:
            state["finished"] = timestamp()
            try:
                save()
            finally:
                signal.signal(signal.SIGTERM, previous)


def start(args: argparse.Namespace) -> int:
    """Return as soon as the child has claimed the project and written its state."""
    import fcntl

    root = Path(args.directory).resolve()
    if not (root / "project.json").is_file():
        raise InputError(args.directory, "no project.json here; run yap-studio init first")
    folder = paths.validate_output_dir(root / ".yap/run")
    # Serialize parents so a rejected launch cannot truncate an active child's log.
    fd = os.open(folder / "launch.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = background.read_record(state_path(root))
        if live(previous):
            raise running_error(root, previous)
        identifier = uuid.uuid4().hex
        command = [sys.executable, "-m", "yap_studio", "run", str(root)]
        if args.note is not None:
            command.extend(("--note", args.note))
        if args.allow_violations:
            command.append("--allow-violations")
        if args.frames is not None:
            command.extend(("--frames", f"{args.frames[0]}-{args.frames[1]}"))
        command.extend(("--json", "--detached-run", identifier))
        log = folder / "run.log"
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as output:
            os.fchmod(output.fileno(), 0o600)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=output, close_fds=True,
                                       start_new_session=True)
        deadline = time.monotonic() + 60
        while True:
            saved = background.read_record(state_path(root))
            if saved is not None and saved.get("id") == identifier:
                break
            if process.poll() is not None:
                lines = background.log_lines(log)
                for line in lines:
                    background.relay_child_line(line, json_output=args.json)
                if not lines:
                    background.error("could not start background run", "run_start_failed",
                                     json_output=args.json)
                return 2
            if time.monotonic() >= deadline:
                background.terminate(process)
                raise YapError("background run did not become ready within 60 seconds; stopped it",
                               code="run_ready_timeout")
            time.sleep(0.1)
    check, stop = commands(root)
    if args.json:
        print(json.dumps({"schema": "yap-studio/run-detached@1", "status": saved["state"],
                          "pid": saved["pid"], "started": saved["started"], "log": str(log),
                          "check": check, "stop": stop}))
    else:
        print(f"Running in the background (pid {saved['pid']}). "
              "A long final can take more than 30 minutes.")
        print(f"Check progress: {check}")
        print(f"Stop it: {stop}")
    return 0


def _group_running(pgid: int) -> bool | None:
    """Check for non-zombie members, returning None when ps is unavailable."""
    try:
        result = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            int(fields[0])
            group = int(fields[1])
        except ValueError:
            continue
        if group == pgid and not fields[2].startswith("Z"):
            return True
    return False


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return _group_running(pid) is not False
    return True


def stop(root: Path, *, json_output: bool) -> int:
    """Signal only a verified run's process group, then wait for cleanup."""
    root = root.resolve()
    state = background.read_record(state_path(root))
    stopped = None
    if live(state):
        pid, step = state["pid"], state.get("step")
        try:
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + 30
            while _group_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if _group_alive(pid):
                # The leader is still alive but no longer runs our command; do not signal its group.
                if not live(state) and pid_alive(pid) and _group_alive(pid):
                    raise YapError(f"background run process {pid} could not be stopped",
                                   code="run_stop_failed")
                os.killpg(pid, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while _group_alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                if _group_alive(pid):
                    raise YapError(f"background run process group {pid} did not stop",
                                   code="run_stop_failed")
        except ProcessLookupError:
            pass
        except OSError as exc:
            if not isinstance(exc, PermissionError) or _group_running(pid) is not False:
                raise YapError(f"could not stop background run process {pid}: {exc}",
                               code="run_stop_failed") from exc
        latest = background.read_record(state_path(root))
        if latest and latest.get("id") == state["id"] and latest.get("state") == "running":
            latest.update(state="cancelled", finished=timestamp(),
                          error={"message": f"stopped by {commands(root)[1]}"})
            background.write_json(state_path(root), latest)
        stopped = {"pid": pid, "step": step}
    if json_output:
        print(json.dumps({"stopped": stopped}))
    elif stopped:
        print(f"Stopped the background run at {stopped['step']}.")
    else:
        print("No background run is working on this project.")
    return 0


def status(root: Path, stages: list[dict]) -> dict | None:
    """Project the run state without changing its file."""
    saved = background.read_record(state_path(root))
    if saved is None:
        return None
    state = saved.get("state")
    error = saved.get("error")
    if state == "running" and not live(saved, allow_unchecked=True):
        state = "lost"
        error = {"message": "the background run ended without a result; read "
                 f"{saved.get('log')}, then run yap-studio run {shlex.quote(str(root))} again"}
    elapsed = None
    try:
        end = (datetime.fromisoformat(saved["finished"]) if saved.get("finished")
               else datetime.now(UTC))
        elapsed = max(0, int((end - datetime.fromisoformat(saved["started"])).total_seconds()))
    except (KeyError, TypeError, ValueError):
        pass
    progress = next((item.get("progress") for item in stages if item["state"] == "running"), None)
    return {
        "state": state, "pid": saved.get("pid"), "started": saved.get("started"),
        "finished": saved.get("finished"), "elapsed_s": elapsed, "step": saved.get("step"),
        "progress": progress, "steps": saved.get("steps", []), "result": saved.get("result"),
        "error": error, "log": saved.get("log"),
        "stop": commands(root)[1] if state == "running" else None,
    }


def wait(root: Path, seconds: int) -> None:
    """Wait for a live run or stage, bounded by the caller's deadline."""
    from yap_studio.project.status import STAGES

    deadline = time.monotonic() + seconds
    while (live(background.read_record(state_path(root)), allow_unchecked=True)
           or any(pid_alive((background.read_record(root / spec.folder / "progress.json")
                             or {}).get("pid")) for spec in STAGES)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(2, remaining))


def status_lines(run: dict) -> list[str]:
    state = run["state"]
    step = run.get("step")
    result = run.get("result") or {}
    if state == "running":
        clock = local_clock(run.get("started"))
        minutes, seconds = divmod(run.get("elapsed_s") or 0, 60)
        line = f"run: running in the background since {clock} ({minutes}m {seconds}s), at {step}"
        progress = run.get("progress")
        if progress:
            line += f": {progress.get('step', 'working')}"
            if progress.get("fraction") is not None:
                line += f" ({progress['fraction']:.0%})"
        return [line]
    if state == "finished":
        if result.get("pilot"):
            return [f"run: finished, pilot {result['pilot']}"]
        lines = [f"run: finished, {result.get('final')} saved as {result.get('version')}"]
        if result.get("words"):
            from yap_studio.checks.kept_words import words_summary

            lines.append(f"words: {words_summary(result['words'])}")
        review = result.get("review") or {}
        if review.get("status") == "failed":
            lines.extend((review["message"], f"next: {review['command']}"))
        elif review.get("status") == "ready":
            for video in review["links"].get("videos", []):
                lines.append(f"review: {video['url']}")
        return lines
    if state == "stopped":
        stopped = result.get("stopped") or {}
        lines = [f"run: stopped at {stopped.get('step', step)}: {stopped.get('message', '')}"]
        if stopped.get("suggested_action"):
            lines.append(f"fix: {stopped['suggested_action']}")
        return lines
    if state == "cancelled":
        return [f"run: stopped by request at {step}"]
    message = (run.get("error") or {}).get("message", "")
    if state == "lost":
        return ["run: " + message.removeprefix("the background run ")]
    return [f"run: failed: {message}"]
