"""A page owns its speech job and every descendant process."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_transcript_shared import CountingBackend

from yap_studio import cache
from yap_studio.project.lock import pid_alive
from yap_studio.stages import run_detach
from yap_studio.transcript import Transcript, shared, take_job

FAKE_JOB = '''
import json, os, signal, subprocess, sys, time
from pathlib import Path
from yap_studio import background
from yap_studio.transcript.take_job import watch_parent
watch_parent()
grandchild = subprocess.Popen(["sleep", "60"])
record = Path(sys.argv[sys.argv.index("--take-job") + 1])
background.write_json(record.with_suffix(".pids"), [os.getpid(), grandchild.pid])
time.sleep(60)
'''


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    pytest.fail("timed out waiting for the fake take job")


@pytest.fixture
def job_command(tmp_path):
    script = tmp_path / "fake_job.py"
    script.write_text(FAKE_JOB)
    return [sys.executable, str(script)]


def job_pids():
    records = list(cache.cache_dir("transcripts", "jobs").glob("*.pids"))
    return (records[0], json.loads(records[0].read_text())) if records else None


def group_gone(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return run_detach._group_running(pid) is False
    return False


@pytest.mark.parametrize("running", [False, True, None])
def test_group_permission_error_checks_live_members(monkeypatch, running):
    def denied(*_args):
        raise PermissionError("synthetic exited process group")

    monkeypatch.setattr(take_job.os, "killpg", denied)
    monkeypatch.setattr(run_detach, "_group_running", lambda _: running)
    assert take_job._group_exists(123) is (running is not False)
    if running is False:
        take_job._signal_group(123, signal.SIGTERM)
    else:
        with pytest.raises(PermissionError):
            take_job._signal_group(123, signal.SIGTERM)


def test_close_kills_whole_group(tmp_path, job_command):
    take = tmp_path / "take.mp4"
    take.touch()
    jobs = take_job.TakeJobs(child_command=job_command)
    try:
        assert jobs.start(take, CountingBackend())
        record, pids = wait_for(job_pids)
        assert not jobs.start(take, CountingBackend())
        assert all(pid_alive(pid) for pid in pids)
    finally:
        jobs.close(timeout_s=0.2)
    wait_for(lambda: group_gone(pids[0]))
    assert not any(pid_alive(pid) for pid in pids)
    assert not record.with_suffix(".json").exists()
    jobs.close()
    assert not jobs.start(take, CountingBackend())


@pytest.mark.parametrize("quantization,suffix", [(None, ":full"), ("int8", "")])
def test_child_model_spec_preserves_onnx_quantization(
    tmp_path, job_command, monkeypatch, quantization, suffix,
):
    backend = CountingBackend()
    backend.name = "parakeet-onnx"
    backend.quantization = quantization
    take = tmp_path / "take.mp4"
    take.touch()
    command = []
    spawn = subprocess.Popen

    def capture(arguments, **kwargs):
        command.extend(arguments)
        return spawn(arguments, **kwargs)

    monkeypatch.setattr(take_job.subprocess, "Popen", capture)
    jobs = take_job.TakeJobs(child_command=job_command)
    try:
        assert jobs.start(take, backend)
        wait_for(job_pids)
        assert command[command.index("--model") + 1] == backend.model + suffix
    finally:
        jobs.close(timeout_s=0.2)


def test_start_racing_close_cannot_leave_a_child(tmp_path, job_command, monkeypatch):
    take = tmp_path / "take.mp4"
    take.touch()
    jobs = take_job.TakeJobs(child_command=job_command)
    entered, release = threading.Event(), threading.Event()
    spawn = subprocess.Popen

    def blocked_spawn(*args, **kwargs):
        child = spawn(*args, **kwargs)
        entered.set()
        assert release.wait(8)
        return child

    monkeypatch.setattr(subprocess, "Popen", blocked_spawn)
    with ThreadPoolExecutor(2) as pool:
        start = pool.submit(jobs.start, take, CountingBackend())
        try:
            assert entered.wait(8)
            record, pids = wait_for(job_pids)
            closing = pool.submit(jobs.close, 0.2)
            release.set()
            assert start.result(timeout=8)
            closing.result(timeout=8)
        finally:
            release.set()
            jobs.close(timeout_s=0.2)
    assert group_gone(pids[0])
    assert not record.with_suffix(".json").exists()
    assert not jobs.start(take, CountingBackend())


def test_close_escalates_for_a_stubborn_descendant(tmp_path, job_command):
    from pathlib import Path

    script = Path(job_command[-1])
    script.write_text(FAKE_JOB.replace(
        "grandchild = ", "signal.signal(signal.SIGTERM, signal.SIG_IGN)\ngrandchild = ",
    ).replace("record = ", "signal.signal(signal.SIGTERM, signal.SIG_DFL)\nrecord = "))
    take = tmp_path / "take.mp4"
    take.touch()
    jobs = take_job.TakeJobs(child_command=job_command)
    try:
        assert jobs.start(take, CountingBackend())
        record, pids = wait_for(job_pids)
    finally:
        jobs.close(timeout_s=0.1)
    assert group_gone(pids[0])
    assert not any(pid_alive(pid) for pid in pids)
    assert not record.with_suffix(".json").exists()


def test_parent_sigkill_closes_pipe(tmp_path, job_command):
    take = tmp_path / "take.mp4"
    take.touch()
    code = '''
import sys, time
from pathlib import Path
from yap_studio.transcript.take_job import TakeJobs
class Backend:
    name = "parakeet-mlx"
    model = "test-model"
    def version(self): return "test-version"
    def download_notice(self): return None
jobs = TakeJobs(child_command=sys.argv[2:])
assert jobs.start(Path(sys.argv[1]), Backend())
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, "-c", code, str(take), *job_command],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pids = []
    try:
        _, pids = wait_for(job_pids)
        parent.kill()
        parent.wait(timeout=5)
        wait_for(lambda: group_gone(pids[0]))
        assert not any(pid_alive(pid) for pid in pids)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if pids and not group_gone(pids[0]):
            os.killpg(pids[0], signal.SIGKILL)


def test_start_skips_cached_missing_and_undownloaded(tmp_path, job_command, monkeypatch):
    take = tmp_path / "take.mp4"
    backend = CountingBackend()
    jobs = take_job.TakeJobs(child_command=job_command)
    try:
        assert not jobs.start(take, backend)
        take.touch()
        backend.notice = "download required"
        assert not jobs.start(take, backend)
        backend.notice = None
        digest = shared.take_digest(take)
        shared.store(backend, shared.Request(digest, 0, take), Transcript([]))
        assert not jobs.start(take, backend)
        monkeypatch.setattr(take_job.background, "posix_supported", lambda: False)
        assert not jobs.start(take, backend)
    finally:
        jobs.close()
    jobs.close()


def test_run_reuses_raw_cache(tmp_path):
    from test_gallery_still import clip

    take = clip(tmp_path / "take.mp4")
    backend = CountingBackend()
    take_job.run(take, backend)
    take_job.run(take, backend)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("available", [False, True])
def test_main_does_not_download_or_print(tmp_path, monkeypatch, capsys, available):
    record = tmp_path / "job.json"
    record.write_text("{}")
    backend = CountingBackend()
    backend.notice = "download needed"
    monkeypatch.setattr(take_job, "resolve_backend", lambda *_: backend if available else None)
    monkeypatch.setattr(take_job, "watch_parent", lambda: None)
    monkeypatch.setattr(take_job.os, "nice", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["take_job", str(tmp_path / "missing.mp4"), "--engine",
                                     backend.name, "--model", backend.model, "--take-job",
                                     str(record)])
    previous = signal.getsignal(signal.SIGTERM)
    try:
        assert take_job.main() == 0
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert backend.calls == []
    assert not record.exists()
    assert capsys.readouterr().out == ""
