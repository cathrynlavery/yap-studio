"""Background runs publish progress and results and stop their process group."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def test_wait_returns_immediately_without_live_work(tmp_path, monkeypatch):
    from yap_studio.stages import run_detach

    monkeypatch.setattr(run_detach.time, "sleep", lambda _: pytest.fail("nothing to wait for"))
    run_detach.wait(tmp_path, 30)


def test_wait_waits_for_standalone_stage(tmp_path):
    from yap_studio.stages import run_detach

    stage = tmp_path / "fillers"
    stage.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"],
                             stdin=subprocess.DEVNULL)
    reaper = threading.Thread(target=child.wait)
    reaper.start()
    try:
        (stage / "progress.json").write_text(json.dumps({"pid": child.pid, "step": "testing"}))
        started = time.monotonic()
        run_detach.wait(tmp_path, 10)
        assert child.poll() is not None
        assert time.monotonic() - started < 9
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        reaper.join(timeout=5)


def test_finished_status_shows_words_summary():
    from yap_studio.stages.run_detach import status_lines

    lines = status_lines({"state": "finished", "result": {
        "final": "final/result.mp4", "version": "v2", "words": {
            "status": "pass", "baseline": "v1", "total": 331, "kept": 326,
            "dropped": [{"why": "approved"}] * 3 + [{"why": "story"}] * 2,
        },
    }})
    assert lines[-1] == ("words: 326 of 331 words of v1 kept; 5 removed: "
                         "3 at the creator's request, 2 by the story edit")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name, folder in (("YAP_STUDIO_DATA_DIR", "data"), ("YAP_STUDIO_CONFIG", "config"),
                         ("YAP_STUDIO_CACHE", "cache")):
        monkeypatch.setenv(name, str(tmp_path / folder))


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "my reel"
    root.mkdir()
    examples = json.loads((ROOT / "tests/fixtures/origin-examples.json").read_text())
    project = next(item["data"] for item in examples if item["contract"] == "project")
    project["style"] = "preset:clean_minimal"
    (root / "project.json").write_text(json.dumps(project))
    (tmp_path / "sitecustomize.py").write_text(r'''
import os
import subprocess
import sys
import time
from yap_studio.stages import run
from yap_studio import process
from yap_studio.errors import YapError
if os.environ.get("FAKE_START_ERROR") and "--detached-run" in sys.argv:
    from yap_studio.stages import run_detach
    def fail_start(args):
        raise YapError("Synthetic startup error.", code="fake_start_error",
                       suggested_action="Fix the project.")
    run_detach.child = fail_start
def fake(root, state):
    state["final"] = "final/demo.mp4"
    state["version"] = "v2"
    return True, "fake"
def assemble(root, state):
    mode = os.environ.get("FAKE_RUN", "slow")
    if mode in {"slow", "slow-cleanup"}:
        script = "from pathlib import Path; import os,time; Path(" + repr(
            str(root / "grandchild")) + ").write_text(str(os.getpid())); time.sleep(60)"
        if mode == "slow-cleanup":
            script = "\n".join([
                "import os, signal, time",
                "from pathlib import Path",
                "def stop(signum, frame):",
                "    time.sleep(1)",
                "    raise SystemExit(0)",
                "signal.signal(signal.SIGTERM, stop)",
                "Path(" + repr(str(root / "grandchild")) + ").write_text(str(os.getpid()))",
                "time.sleep(60)",
            ])
            subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.DEVNULL).wait()
        else:
            process.spawn([sys.executable, "-c", script], timeout_s=65, tool="fake")
    elif mode == "stop":
        raise YapError("Picture check failed.", code="assembly_checks_failed",
                       suggested_action="Fix the picture.")
    elif mode == "fail":
        raise RuntimeError("unexpected fake failure")
    else:
        time.sleep(1)
    return fake(root, state)
run._HANDLERS = {name: fake for name in run.STEPS}
run._HANDLERS["assemble"] = assemble
''')
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ROOT / "src")))}
    yield root, env
    stop_left_run(root)


def stop_left_run(root, timeout=60):
    """Kill a run a test left working and wait until its process group exits."""
    from yap_studio import background

    state = read_state(root)
    if (state and state["state"] == "running" and state["pid"] != os.getpid()
            and isinstance(state.get("id"), str) and state["id"]):
        pid = state["pid"]
        if background.command_matches(pid, state["id"], "--detached-run") is not True:
            return
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        # A killed process keeps its pid and argv until it finishes exiting and is reaped,
        # which takes longer on a busy machine.
        # The conftest leftover guard runs right after this teardown.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                pass
            time.sleep(0.05)
        pytest.fail(f"detached run process group {pid} did not exit within {timeout} s of SIGKILL")


def command(job, *args, timeout=12):
    root, env = job
    return subprocess.run([sys.executable, "-m", "yap_studio", args[0], str(root), *args[1:]],
                          cwd=ROOT, env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=timeout, check=False)


def read_state(root):
    try:
        return json.loads((root / ".yap/run/state.json").read_text())
    except FileNotFoundError:
        return None


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    assert predicate()


def start(job, *options):
    result = command(job, "run", "--detach", "--json", *options)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.e2e
def test_run_returns_status_and_stop_kills_group(job):
    root, env = job
    env["FAKE_RUN"] = "slow-cleanup"
    document = start(job, "--note", "Final", "--allow-violations", "--frames", "1-20")
    assert document["status"] == "running"
    assert document["schema"] == "yap-studio/run-detached@1"
    os.kill(document["pid"], 0)
    wait_for(lambda: (root / "grandchild").exists())
    grandchild = int((root / "grandchild").read_text())
    state = read_state(root)
    assert state["options"] == {"note": "Final", "frames": [1, 20], "allow_violations": True}
    assert (root / ".yap/run/state.json").stat().st_mode & 0o777 == 0o600
    assert Path(document["log"]).stat().st_mode & 0o777 == 0o600
    result = json.loads(command(job, "status", "--json").stdout)
    assert result["run"]["state"] == "running" and result["run"]["step"] == "assemble"
    assert "--wait 540" in result["next"] and "still working" in result["next"]
    stopped = command(job, "run", "--stop", "--json")
    assert stopped.returncode == 0, stopped.stderr
    for pid in (document["pid"], grandchild):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(document["pid"], 0)
    assert json.loads(stopped.stdout) == {"stopped": {"pid": document["pid"], "step": "assemble"}}
    assert read_state(root)["state"] == "cancelled"
    assert json.loads(command(job, "status", "--json").stdout)["run"]["state"] == "cancelled"
    assert json.loads(command(job, "run", "--stop", "--json").stdout) == {"stopped": None}


@pytest.mark.e2e
def test_finished_result_and_wait_returns_early(job):
    root, env = job
    env["FAKE_RUN"] = "quick"
    start(job)
    started = time.monotonic()
    result = command(job, "status", "--json", "--wait", "30")
    assert time.monotonic() - started < 8
    status = json.loads(result.stdout)["run"]
    assert status["state"] == "finished" and status["step"] is None
    assert status["result"]["status"] == "finished"
    assert status["result"]["final"] == "final/demo.mp4"
    assert "ingest: unchanged, fake" in (root / ".yap/run/run.log").read_text()
    assert "run: finished, final/demo.mp4 saved as v2" in command(job, "status").stdout


@pytest.mark.parametrize("mode,expected", [("stop", "stopped"), ("fail", "failed")])
def test_failed_run_is_visible(job, mode, expected):
    root, env = job
    env["FAKE_RUN"] = mode
    start(job)
    wait_for(lambda: read_state(root)["state"] != "running")
    assert read_state(root)["state"] == expected
    plain = command(job, "status").stdout
    if mode == "stop":
        assert "run: stopped at assemble: Picture check failed." in plain
        assert "fix: Fix the picture." in plain
    else:
        assert "run: failed:" in plain and "unexpected fake failure" in plain


def test_duplicate_and_racing_child_are_refused(job):
    root, _ = job
    first = start(job)
    duplicate = command(job, "run", "--detach", "--json")
    assert duplicate.returncode == 3
    error = json.loads(duplicate.stderr)["error"]
    assert error["code"] == "run_running" and str(first["pid"]) in error["message"]
    assert "--wait 540" in error["suggested_action"]
    racing = command(job, "run", "--detached-run", "racing", "--json")
    assert racing.returncode == 3
    assert json.loads(racing.stderr)["error"]["code"] == "run_running"
    assert read_state(root)["pid"] == first["pid"]


def test_lost_state_and_foreground_clears_old_result(job):
    root, env = job
    env["FAKE_RUN"] = "quick"
    folder = root / ".yap/run"
    folder.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=5)
    state = {"state": "running", "pid": dead.pid, "id": "lost", "started":
             "2026-10-17T00:00:00+00:00", "log": str(folder / "run.log"), "step": "assemble"}
    file = folder / "state.json"
    file.write_text(json.dumps(state))
    result = command(job, "status", "--json", "--wait", "30")
    assert json.loads(result.stdout)["run"]["state"] == "lost"
    assert json.loads(file.read_text())["state"] == "running"
    state["state"] = "finished"
    file.write_text(json.dumps(state))
    assert command(job, "run", "--json").returncode == 0
    assert not file.exists()


@pytest.mark.parametrize("detached", [False, True])
def test_run_requires_project_before_creating_run_directory(job, detached):
    root, _ = job
    (root / "project.json").unlink()
    result = command(job, "run", *(["--detach"] if detached else []), "--json")
    assert result.returncode == 2
    assert "no project.json here" in json.loads(result.stderr)["error"]["message"]
    assert not (root / ".yap/run").exists()


@pytest.mark.e2e
def test_foreground_refuses_live_detached_run_without_changing_state(job):
    from yap_studio.stages.run_detach import running_error

    root, env = job
    start(job)
    wait_for(lambda: (root / "grandchild").exists())
    file = root / ".yap/run/state.json"
    original = file.read_bytes()
    state = json.loads(original)
    result = command((root, {**env, "FAKE_RUN": "quick"}), "run", "--json")
    assert result.returncode == 3, result.stderr
    assert json.loads(result.stderr)["error"] == running_error(root, state).to_dict()
    assert file.read_bytes() == original
    os.kill(state["pid"], 0)


@pytest.mark.e2e
@pytest.mark.parametrize("stale_state", [None, "finished", "running"])
def test_foreground_holds_run_lock_against_detached_child(job, stale_state):
    root, env = job
    with open(root / "foreground.log", "w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "yap_studio", "run", str(root), "--json"],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
            close_fds=True, start_new_session=True,
        )
    try:
        wait_for(lambda: (root / "grandchild").exists())
        saved = None
        if stale_state is not None:
            saved = {"state": stale_state, "id": "old-background", "pid": process.pid,
                     "started": "2026-10-17T21:01:54+00:00"}
            (root / ".yap/run/state.json").write_text(json.dumps(saved))
        result = command((root, {**env, "FAKE_RUN": "quick"}), "run", "--detached-run", "racing",
                         "--json")
        assert result.returncode == 3, result.stderr
        error = json.loads(result.stderr)["error"]
        assert error["code"] == "run_running"
        assert error["message"] == "another yap-studio run is already working on this project"
        assert error["suggested_action"] == "Wait for it to finish, then run again."
        assert read_state(root) == saved
        assert process.poll() is None
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait(timeout=5)
        (root / ".yap/run/state.json").unlink(missing_ok=True)


def test_run_lock_reports_live_background_when_ps_is_unavailable(job, monkeypatch, capsys):
    from yap_studio import background, cli
    from yap_studio.stages import run_detach

    root, _ = job
    with run_detach.run_lock(root):
        state = {"state": "running", "id": "unchecked", "pid": os.getpid(),
                 "started": "2026-10-17T21:01:54+00:00"}
        run_detach.state_path(root).write_text(json.dumps(state))

        def unavailable(*_args, **_kwargs):
            raise FileNotFoundError("ps unavailable")

        monkeypatch.setattr(background.subprocess, "run", unavailable)
        assert cli.main(["run", str(root), "--json"]) == 3
        assert json.loads(capsys.readouterr().err)["error"] == (
            run_detach.running_error(root, state).to_dict())
        assert read_state(root) == state


@pytest.mark.parametrize("options", [["--detach"], ["--note", "Final"], ["--frames", "1-5"],
                                    ["--allow-violations"]])
def test_invalid_stop_options(job, options):
    result = command(job, "run", "--stop", *options)
    assert result.returncode == 2 and "cannot combine --stop" in result.stderr


@pytest.mark.parametrize("seconds", ["0", "3601", "no"])
def test_invalid_wait(job, seconds):
    assert command(job, "status", "--wait", seconds).returncode == 2


@pytest.mark.parametrize(("state", "expected"), [
    ({"state": "running", "started": "2026-10-17T21:01:54+00:00", "elapsed_s": 724,
      "step": "assemble", "progress": {"step": "assembling picture", "fraction": 0.1}},
     "run: running in the background since 21:01:54 (12m 4s), "
     "at assemble: assembling picture (10%)"),
    ({"state": "cancelled", "step": "assemble"}, "run: stopped by request at assemble"),
    ({"state": "failed", "error": {"message": "broken"}}, "run: failed: broken"),
    ({"state": "lost", "error": {"message": "the background run ended without a result"}},
     "run: ended without a result"),
])
def test_status_text(state, expected, monkeypatch):
    from yap_studio.stages.run_detach import status_lines

    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "UTC0")
            time.tzset()
            assert status_lines(state)[0] == expected
    finally:
        time.tzset()


@pytest.mark.parametrize(("zone", "clock"), [("EST5", "16:01:54"), ("JST-9", "06:01:54")])
def test_run_messages_use_local_time(tmp_path, monkeypatch, zone, clock):
    from yap_studio.stages.run_detach import running_error, status_lines

    state = {"state": "running", "started": "2026-10-17T21:01:54.856746+00:00", "elapsed_s": 724,
             "step": "assemble", "pid": 123}
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", zone)
            time.tzset()
            assert status_lines(state) == [
                f"run: running in the background since {clock} (12m 4s), at assemble"]
            assert running_error(tmp_path, state).message == (
                f"a background run is already working on this project (pid 123, started {clock})")
    finally:
        time.tzset()


def test_run_process_match_requires_the_whole_id(monkeypatch):
    from yap_studio import background

    monkeypatch.setattr(background.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout="python -m yap_studio run . --detached-run abc-other"))
    assert background.command_matches(123, "abc", "--detached-run") is False


@pytest.mark.parametrize("operation", ["run", "style"])
def test_background_needs_posix(tmp_path, monkeypatch, capsys, operation):
    from yap_studio import background, cli

    monkeypatch.setattr(background, "posix_supported", lambda: False)
    prefix = ["run", str(tmp_path)] if operation == "run" else ["style", "pick"]
    for option in ("--detach", "--stop"):
        assert cli.main([*prefix, option, "--json"]) == 2
        assert json.loads(capsys.readouterr().err)["error"]["code"] == "background_unsupported"


def left_run(tmp_path, monkeypatch, identifier):
    """Write a running state with a reaped pid and a fake clock."""
    root = tmp_path / "my reel"
    folder = root / ".yap/run"
    folder.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=5)
    (folder / "state.json").write_text(json.dumps({
        "state": "running", "id": identifier, "pid": dead.pid}))
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(
        0, clock[0] + seconds))
    return root, dead.pid, clock


@pytest.mark.parametrize("denied_kill", [False, True])
def test_left_run_teardown_waits_for_a_slow_group(tmp_path, monkeypatch, denied_kill):
    from yap_studio import background

    root, pid, clock = left_run(tmp_path, monkeypatch, "slow-exit")
    signals = []
    probes = []

    def killpg(group_pid, sig):
        assert group_pid == pid
        if sig:
            signals.append(sig)
            if denied_kill:
                raise PermissionError
            return
        probes.append(sig)
        if len(probes) > 5:
            raise ProcessLookupError
        if len(probes) > 3:
            raise PermissionError

    monkeypatch.setattr(background, "command_matches", lambda *_: True)
    monkeypatch.setattr(os, "killpg", killpg)
    stop_left_run(root)
    assert signals == [signal.SIGKILL]
    assert len(probes) == 6
    assert clock[0] < 60


def test_left_run_teardown_fails_when_the_group_never_exits(tmp_path, monkeypatch):
    from yap_studio import background

    root, pid, clock = left_run(tmp_path, monkeypatch, "never-exits")
    monkeypatch.setattr(background, "command_matches", lambda *_: True)
    monkeypatch.setattr(os, "killpg", lambda *_: None)
    with pytest.raises(pytest.fail.Exception, match=rf"{pid} did not exit"):
        stop_left_run(root)
    assert 60 <= clock[0] < 60.1


@pytest.mark.parametrize("matches", [False, None])
def test_left_run_teardown_never_signals_an_unchecked_pid(tmp_path, monkeypatch, matches):
    from yap_studio import background

    root, _, clock = left_run(tmp_path, monkeypatch, "unchecked")
    monkeypatch.setattr(background, "command_matches", lambda *_: matches)
    monkeypatch.setattr(os, "killpg", lambda *_: pytest.fail("signaled an unchecked process"))
    stop_left_run(root)
    assert clock[0] == 0


def test_run_stop_cannot_signal_an_unrelated_process(job, monkeypatch, capsys):
    from yap_studio import cli

    root, _ = job
    folder = root / ".yap/run"
    folder.mkdir(parents=True)
    state = {"state": "running", "id": "wrong-id", "pid": os.getpid()}
    (folder / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(os, "killpg", lambda *_: pytest.fail("signaled an unrelated process"))
    assert cli.main(["run", str(root), "--stop", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"stopped": None}


def test_run_check_failure_does_not_signal_or_change_state(job, monkeypatch, capsys):
    from yap_studio import background, cli

    root, _ = job
    folder = root / ".yap/run"
    folder.mkdir(parents=True)
    state = {"state": "running", "id": "unchecked", "pid": os.getpid()}
    path = folder / "state.json"
    path.write_text(json.dumps(state))
    monkeypatch.setattr(background, "command_matches", lambda *_: None)
    monkeypatch.setattr(os, "killpg", lambda *_: pytest.fail("signaled an unchecked process"))
    assert cli.main(["run", str(root), "--stop", "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "run_check_failed"
    assert json.loads(path.read_text()) == state


@pytest.mark.parametrize("json_output", [False, True])
def test_start_error_is_relayed(job, json_output):
    root, env = job
    env["FAKE_START_ERROR"] = "1"
    result = command(job, "run", "--detach", *(["--json"] if json_output else []))
    assert result.returncode == 2 and "Synthetic startup error." in result.stderr
    assert read_state(root) is None
    if json_output:
        assert json.loads(result.stderr)["error"]["code"] == "fake_start_error"
    else:
        assert "error: Synthetic startup error." in result.stderr
        assert "fix: Fix the project." in result.stderr


def test_simultaneous_run_parents_start_one_child(job):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: command(job, "run", "--detach", "--json"), range(2)))
    assert sorted(item.returncode for item in results) == [0, 3]
    root, _ = job
    assert read_state(root)["state"] == "running"
    # The refused parent must not truncate the active run's readable log.
    assert "ingest: unchanged, fake" in (root / ".yap/run/run.log").read_text()


def test_run_status_projects_stage_progress(job):
    root, _ = job
    start(job)
    wait_for(lambda: read_state(root)["step"] == "assemble")
    folder = root / "final"
    folder.mkdir()
    progress = {"pid": read_state(root)["pid"], "stage": "assemble", "step": "assembling picture",
                "fraction": 0.1, "started": "2026-10-17T21:01:54+00:00"}
    (folder / "progress.json").write_text(json.dumps(progress))
    result = command(job, "status", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["run"]["progress"] == progress
    assert "at assemble: assembling picture (10%)" in command(job, "status").stdout


@pytest.mark.parametrize("leader_alive", [False, True])
def test_stop_escalates_and_records_cancellation(job, monkeypatch, capsys, leader_alive):
    from yap_studio import background, cli
    from yap_studio.stages import run_detach

    root, _ = job
    folder = root / ".yap/run"
    folder.mkdir(parents=True)
    file = folder / "state.json"
    file.write_text(json.dumps({"state": "running", "id": "stubborn", "pid": os.getpid(),
                                "step": "assemble"}))
    signals = []
    clock = [0.0]
    killed_at = []
    checks = []

    def matches(*args):
        checks.append(args)
        return True

    def killpg(_pid, sig):
        if sig == 0:
            if killed_at and clock[0] >= killed_at[0] + 0.2:
                raise ProcessLookupError
        else:
            signals.append(sig)
            if sig == signal.SIGKILL:
                killed_at.append(clock[0])

    monkeypatch.setattr(run_detach, "pid_alive", lambda _: (
        signal.SIGKILL not in signals if leader_alive else signal.SIGTERM not in signals))
    monkeypatch.setattr(background, "command_matches", matches)
    monkeypatch.setattr(run_detach.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_detach.time, "sleep", lambda seconds: clock.__setitem__(
        0, clock[0] + seconds))
    monkeypatch.setattr(os, "killpg", killpg)
    assert cli.main(["run", str(root), "--stop", "--json"]) == 0
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert len(checks) == (2 if leader_alive else 1)
    assert 30 <= killed_at[0] < 30.2
    assert killed_at[0] + 0.2 <= clock[0] < killed_at[0] + 5
    with pytest.raises(ProcessLookupError):
        os.killpg(os.getpid(), 0)
    assert json.loads(capsys.readouterr().out)["stopped"]["step"] == "assemble"
    assert read_state(root)["state"] == "cancelled"


def test_stop_reports_failure_only_when_group_remains(job, monkeypatch, capsys):
    from yap_studio import background, cli
    from yap_studio.stages import run_detach

    root, _ = job
    file = run_detach.state_path(root)
    file.parent.mkdir(parents=True)
    state = {"state": "running", "id": "stubborn-group", "pid": os.getpid(), "step": "assemble"}
    file.write_text(json.dumps(state))
    signals = []
    clock = [0.0]
    monkeypatch.setattr(run_detach, "pid_alive", lambda _: signal.SIGTERM not in signals)
    monkeypatch.setattr(background, "command_matches", lambda *_: True)
    monkeypatch.setattr(run_detach.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_detach.time, "sleep", lambda seconds: clock.__setitem__(
        0, clock[0] + seconds))
    monkeypatch.setattr(os, "killpg", lambda _pid, sig: signals.append(sig) if sig else None)
    assert cli.main(["run", str(root), "--stop", "--json"]) == 2
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert 35 <= clock[0] < 35.3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "run_stop_failed"
    assert read_state(root) == state


@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("running", [False, True, None])
def test_stop_handles_permission_error(job, monkeypatch, capsys, denied_signal, running):
    from yap_studio import background, cli
    from yap_studio.stages import run_detach

    root, _ = job
    file = run_detach.state_path(root)
    file.parent.mkdir(parents=True)
    state = {"state": "running", "id": "exited-group", "pid": os.getpid(), "step": "assemble"}
    file.write_text(json.dumps(state))
    original = file.read_bytes()
    signals = []
    clock = [0.0]

    def killpg(pid, sig):
        assert pid == state["pid"]
        if sig:
            signals.append(sig)
        if sig == denied_signal or (sig == 0 and denied_signal == signal.SIGTERM):
            raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(run_detach, "pid_alive", lambda _: True)
    monkeypatch.setattr(background, "command_matches", lambda *_: True)
    monkeypatch.setattr(run_detach.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_detach.time, "sleep", lambda seconds: clock.__setitem__(
        0, clock[0] + seconds))
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(run_detach, "_group_running", lambda _: running, raising=False)
    result = cli.main(["run", str(root), "--stop", "--json"])
    assert signals == ([signal.SIGTERM] if denied_signal == signal.SIGTERM else
                       [signal.SIGTERM, signal.SIGKILL])
    if running is False:
        assert result == 0
        assert json.loads(capsys.readouterr().out) == {
            "stopped": {"pid": state["pid"], "step": "assemble"}}
        assert read_state(root)["state"] == "cancelled"
    else:
        assert result == 2
        assert json.loads(capsys.readouterr().err)["error"]["code"] == "run_stop_failed"
        assert file.read_bytes() == original


def test_stop_handles_permission_error_after_sigterm(job, monkeypatch, capsys):
    from yap_studio import background, cli
    from yap_studio.stages import run_detach

    root, _ = job
    file = run_detach.state_path(root)
    file.parent.mkdir(parents=True)
    state = {"state": "running", "id": "exited-group", "pid": os.getpid(), "step": "assemble"}
    file.write_text(json.dumps(state))
    signals = []

    def killpg(_pid, sig):
        if sig == 0:
            raise PermissionError(1, "Operation not permitted")
        signals.append(sig)

    monkeypatch.setattr(run_detach, "pid_alive", lambda _: True)
    monkeypatch.setattr(background, "command_matches", lambda *_: True)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(run_detach, "_group_running", lambda _: False, raising=False)
    assert cli.main(["run", str(root), "--stop", "--json"]) == 0
    assert signals == [signal.SIGTERM]
    assert json.loads(capsys.readouterr().out) == {
        "stopped": {"pid": state["pid"], "step": "assemble"}}
    assert read_state(root)["state"] == "cancelled"


@pytest.mark.parametrize("running", [False, True, None])
def test_group_alive_handles_permission_error(monkeypatch, running):
    from yap_studio.stages import run_detach

    def killpg(pid, sig):
        assert (pid, sig) == (123, 0)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(run_detach, "_group_running", lambda _: running, raising=False)
    assert run_detach._group_alive(123) is (running is not False)


@pytest.mark.parametrize("output,running", [
    ("1 123 Z\n2 123 Z+\n3 456 S\n", False),
    ("1 123 Z\n2 123 S\n", True),
    ("1 123 R+\n2 123 Z+\n", True),
    ("1 456 S\n2 789 R+\n", False),
    ("\nmalformed\n1 nope S\nnope 123 S\n1 123\n1 123 S extra\n", False),
    ("", False),
])
def test_group_running_parses_ps(monkeypatch, output, running):
    from yap_studio.stages import run_detach

    def ps(args, **kwargs):
        assert args == ["ps", "-A", "-o", "pid=,pgid=,stat="]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 5, "check": False}
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(run_detach.subprocess, "run", ps)
    assert run_detach._group_running(123) is running


@pytest.mark.parametrize("failure", [None, OSError("ps unavailable"),
                                     subprocess.TimeoutExpired("ps", 5)])
def test_group_running_handles_unavailable_ps(monkeypatch, failure):
    from yap_studio.stages import run_detach

    def ps(*_args, **_kwargs):
        if failure is not None:
            raise failure
        return SimpleNamespace(returncode=1, stdout="1 123 S\n")

    monkeypatch.setattr(run_detach.subprocess, "run", ps)
    assert run_detach._group_running(123) is None


def test_log_symlink_cannot_truncate_another_file(job):
    root, _ = job
    source = root / "source.txt"
    source.write_text("keep this synthetic input")
    folder = root / ".yap/run"
    folder.mkdir(parents=True)
    (folder / "run.log").symlink_to(source)
    result = command(job, "run", "--detach", "--json")
    assert result.returncode == 2
    assert source.read_text() == "keep this synthetic input"


@pytest.mark.parametrize("alive", [False, True])
def test_status_and_wait_tolerate_unavailable_ps(job, monkeypatch, alive):
    from yap_studio import background
    from yap_studio.stages import run_detach

    root, _ = job
    file = run_detach.state_path(root)
    file.parent.mkdir(parents=True)
    state = {"state": "running", "id": "unchecked", "pid": os.getpid(), "step": "assemble",
             "started": "2026-10-17T21:01:54+00:00", "log": str(file.with_name("run.log"))}
    file.write_text(json.dumps(state))
    original = file.read_bytes()
    clock = [0.0]
    sleeps = []

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("ps unavailable")

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(background.subprocess, "run", unavailable)
    monkeypatch.setattr(run_detach, "pid_alive", lambda _pid: alive)
    monkeypatch.setattr(run_detach.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_detach.time, "sleep", sleep)
    status = run_detach.status(root, [])
    assert status["state"] == ("running" if alive else "lost")
    assert bool(status["stop"]) is alive
    run_detach.wait(root, 5)
    assert sleeps == ([2, 2, 1] if alive else [])
    assert file.read_bytes() == original


@pytest.mark.parametrize("option", ["--detach", "--stop"])
def test_run_mutations_refuse_when_ps_is_unavailable(job, monkeypatch, capsys, option):
    from yap_studio import background, cli
    from yap_studio.stages import run_detach

    root, _ = job
    file = run_detach.state_path(root)
    file.parent.mkdir(parents=True)
    state = {"state": "running", "id": "unchecked", "pid": os.getpid()}
    file.write_text(json.dumps(state))

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("ps unavailable")

    monkeypatch.setattr(background.subprocess, "run", unavailable)
    monkeypatch.setattr(os, "killpg", lambda *_: pytest.fail("signaled an unchecked process"))
    assert cli.main(["run", str(root), option, "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "run_check_failed"
    assert read_state(root) == state


def test_foreground_keeps_state_when_ps_is_unavailable(job, monkeypatch, capsys):
    from yap_studio import background, cli
    from yap_studio.stages import run as chain
    from yap_studio.stages import run_detach

    root, _ = job
    file = run_detach.state_path(root)
    file.parent.mkdir(parents=True)
    state = {"state": "running", "id": "unchecked", "pid": os.getpid()}
    file.write_text(json.dumps(state))

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("ps unavailable")

    monkeypatch.setattr(background.subprocess, "run", unavailable)
    monkeypatch.setattr(chain, "_HANDLERS", {
        name: lambda *_: (True, "fake") for name in chain.STEPS})
    assert cli.main(["run", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "finished"
    assert file.read_text() == json.dumps(state)


def test_detached_child_never_checks_its_own_process(job, monkeypatch):
    from yap_studio import background, cli
    from yap_studio.stages import run as chain

    root, _ = job

    def unexpected_check(*_args, **_kwargs):
        pytest.fail("the detached child checked its own process")

    monkeypatch.setattr(background.subprocess, "run", unexpected_check)
    monkeypatch.setattr(chain, "_HANDLERS", {
        name: lambda *_: (True, "fake") for name in chain.STEPS})
    assert cli.main(["run", str(root), "--detached-run", "own-state", "--json"]) == 0
    assert read_state(root)["state"] == "finished"
