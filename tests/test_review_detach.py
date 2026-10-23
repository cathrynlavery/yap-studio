"""Detached review subprocesses and cleanup."""

from __future__ import annotations

import http.client
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from yap_studio import background
from yap_studio.cli import main
from yap_studio.review import detach

ROOT = Path(__file__).parents[1]


@pytest.fixture
def review_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["YAP_STUDIO_DATA_DIR"] = str(tmp_path / "data")
    return env


@pytest.fixture
def video(tmp_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=64x36:rate=30000/1001", "-t", "2", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-y", str(path)],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True,
    )
    return path


def command(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "yap_studio", "review", *args], cwd=ROOT, env=env,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False,
    )


def records(env: dict[str, str]) -> list[Path]:
    return list((Path(env["YAP_STUDIO_DATA_DIR"]) / "detached").glob("review-*.json"))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def cleanup(env: dict[str, str]) -> None:
    for path in records(env):
        pid = json.loads(path.read_text())["pid"]
        if alive(pid):
            os.kill(pid, signal.SIGKILL)


def review_payload(url: str) -> dict:
    parsed = urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    try:
        connection.request("GET", parsed.path.rstrip("/") + "/api/review?" + parsed.query)
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.e2e
def test_detach_returns_without_holding_output_pipes_and_stop_cleans_up(
    video: Path, review_env: dict[str, str],
) -> None:
    try:
        result = command(review_env, str(video), "--detach", "--no-open", "--json", "--port", "0")
        assert result.returncode == 0, result.stderr
        document = json.loads(result.stdout)
        url = document["videos"][0]["url"]
        assert review_payload(url)["media"]["name"] == "sample.mp4"
        [record] = records(review_env)
        saved = json.loads(record.read_text())
        assert saved["pid"] == document["detached"]["pid"]
        assert saved["command"] == [sys.executable, "-m", "yap_studio", "review", str(video),
                                    "--port", "0", "--json", "--no-open"]
        assert alive(saved["pid"])
        log = Path(saved["log"])
        stopped = command(review_env, "--stop", "--json")
        assert stopped.returncode == 0, stopped.stderr
        assert json.loads(stopped.stdout)["stopped"] == [{"pid": saved["pid"], "url": url}]
        parsed = urlsplit(url)
        with pytest.raises(OSError):
            connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
            connection.request("GET", "/api/review?" + parsed.query)
            connection.getresponse()
        assert not record.exists() and not log.exists()
    finally:
        cleanup(review_env)


@pytest.mark.e2e
def test_second_detach_replaces_first(video: Path, review_env: dict[str, str]) -> None:
    try:
        first = command(review_env, str(video), "--detach", "--no-open", "--json")
        assert first.returncode == 0, first.stderr
        first_pid = json.loads(first.stdout)["detached"]["pid"]
        second = command(review_env, str(video), "--detach", "--no-open", "--json")
        assert second.returncode == 0, second.stderr
        second_doc = json.loads(second.stdout)
        assert second_doc["detached"]["pid"] != first_pid
        assert not alive(first_pid)
        assert review_payload(second_doc["videos"][0]["url"])["media"]["name"] == "sample.mp4"
        assert command(review_env, "--stop").returncode == 0
    finally:
        cleanup(review_env)


@pytest.mark.e2e
def test_invalid_replacement_keeps_earlier_review(video: Path, review_env: dict[str, str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    second = video.with_name("second.mp4")
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=64x36:rate=30000/1001", "-t", "2", "-vf", "hue=s=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(second)],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True,
    )
    try:
        first = command(review_env, str(video), str(second), "--detach", "--no-open", "--json")
        assert first.returncode == 0, first.stderr
        document = json.loads(first.stdout)
        pid = document["detached"]["pid"]
        url = document["videos"][0]["url"]
        second.unlink()
        replacement = command(review_env, str(video), str(second),
                              "--detach", "--no-open", "--json")
        assert replacement.returncode == 2
        error = json.loads(replacement.stderr)["error"]
        assert error["code"] == "invalid_input"
        assert "Cannot read input" in error["message"]
        assert alive(pid)
        assert review_payload(url)["media"]["name"] == "sample.mp4"
    finally:
        command(review_env, "--stop")
        cleanup(review_env)


def test_failed_replacement_relaunches_earlier_command(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    directory = tmp_path / "data" / "detached"
    directory.mkdir(parents=True)
    video = tmp_path / "sample.mp4"
    old_command = [sys.executable, "-m", "yap_studio", "review", str(video),
                   "--json", "--no-open"]
    old = directory / "review-old.json"
    link = "http://127.0.0.1:12345/?t=token"
    old.write_text(json.dumps({"pid": 12345, "media": [str(video)], "versions": False,
                               "command": old_command, "log": str(old.with_suffix(".log")),
                               "document": {"list": None, "videos": [{"url": link}]}}))
    old.with_suffix(".log").write_text("old output\n")

    def stopped(path: Path, _document: dict) -> tuple[bool, bool]:
        background.remove_record(path)
        return True, True

    monkeypatch.setattr(background, "stop_record", stopped)
    calls: list[list[str]] = []

    def popen(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        if len(calls) == 2:
            record = Path(argv[-1])
            record.write_text(json.dumps({"pid": 54321, "media": [str(video)],
                                          "versions": False, "command": old_command,
                                          "document": {"list": None,
                                                       "videos": [{"url": link}]}}))
        return SimpleNamespace(poll=lambda: 2 if len(calls) == 1 else None)

    monkeypatch.setattr(background.subprocess, "Popen", popen)
    args = SimpleNamespace(media=[str(video)], versions=False, port=None, reviewer=None,
                           allow_host=[], public_base=None, phone=False, json=True, no_open=True)
    assert detach.start(args) == 2
    assert len(calls) == 2
    assert calls[1][:-2] == old_command
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert lines[0]["error"]["code"] == "review_start_failed"
    assert lines[1] == {"kept": {"pid": 54321, "url": link}}


def test_stop_json_reports_successes_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "data" / "detached"
    directory.mkdir(parents=True)
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    for pid in (101, 202):
        (directory / f"review-{pid}.json").write_text(json.dumps({
            "pid": pid, "media": [str(tmp_path / "sample.mp4")],
            "document": {"videos": [{"url": f"http://127.0.0.1:{pid}/"}]},
        }))
    monkeypatch.setattr(background, "stop_record", lambda _path, document: (
        (True, True) if document["pid"] == 101 else (True, False)
    ))

    assert detach.stop(None, json_output=True) == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "stopped": [{"pid": 101, "url": "http://127.0.0.1:101/"}],
    }
    error = json.loads(output.err)["error"]
    assert error["code"] == "review_stop_failed"


@pytest.mark.parametrize("json_output", [False, True], ids=["plain", "json"])
def test_detach_child_json_error_uses_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    child_error = {"error": {"code": "invalid_input", "message": "bad clip",
                             "suggested_action": "Choose another clip."}}
    line = json.dumps(child_error)

    def popen(_argv: list[str], **kwargs: object) -> SimpleNamespace:
        kwargs["stdout"].write(line + "\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(poll=lambda: 2)

    monkeypatch.setattr(background.subprocess, "Popen", popen)
    args = SimpleNamespace(media=[str(tmp_path / "clip.mp4")], versions=False, port=None,
                           reviewer=None, allow_host=[], public_base=None, phone=False,
                           json=json_output, no_open=True)
    assert detach.start(args) == 2
    output = capsys.readouterr()
    assert output.out == ""
    if json_output:
        assert output.err == line + "\n"
    else:
        assert output.err == "error: bad clip\nfix: Choose another clip.\n"
        assert "{" not in output.err


@pytest.mark.e2e
def test_versions_detach_reuses_saved_link(video: Path, review_env: dict[str, str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    newer = video.with_name("newer.mp4")
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=64x36:rate=30000/1001", "-t", "2", "-vf", "hue=s=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(newer)],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True,
    )
    args = (str(video), str(newer), "--versions", "--detach", "--no-open", "--json")
    try:
        first = command(review_env, *args)
        assert first.returncode == 0, first.stderr
        first_doc = json.loads(first.stdout)
        second = command(review_env, *args)
        assert second.returncode == 0, second.stderr
        second_doc = json.loads(second.stdout)
        assert second_doc["videos"][0]["url"] == first_doc["videos"][0]["url"]
        assert second_doc["detached"]["pid"] == first_doc["detached"]["pid"]
        assert alive(first_doc["detached"]["pid"])
        assert review_payload(second_doc["videos"][0]["url"])["media"]["name"] == "newer.mp4"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: command(review_env, *args), range(2)))
        for result in results:
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["detached"]["pid"] == first_doc["detached"]["pid"]
        assert len(records(review_env)) == 1
    finally:
        command(review_env, "--stop")
        cleanup(review_env)


def test_stop_without_records(review_env: dict[str, str], monkeypatch, capsys) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", review_env["YAP_STUDIO_DATA_DIR"])
    assert main(["review", "--stop"]) == 0
    assert capsys.readouterr().out == "No background review is running.\n"


def test_stop_removes_stale_record(review_env: dict[str, str]) -> None:
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait(timeout=5)
    directory = Path(review_env["YAP_STUDIO_DATA_DIR"]) / "detached"
    directory.mkdir(parents=True)
    log = directory / "review-stale.log"
    log.write_text("old output\n")
    record = directory / "review-stale.json"
    record.write_text(json.dumps({"pid": finished.pid, "media": [], "versions": False,
                                  "document": {"list": None, "videos": []}, "log": str(log)}))
    result = command(review_env, "--stop")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "No background review is running.\n"
    assert not record.exists() and not log.exists()


def test_stop_timeout_keeps_record_for_retry(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    directory = tmp_path / "data" / "detached"
    directory.mkdir(parents=True)
    record = directory / "review-waiting.json"
    log = directory / "review-waiting.log"
    log.write_text("review output\n")
    record.write_text(json.dumps({"pid": 12345, "media": [], "versions": False,
                                  "document": {"list": None, "videos": []}, "log": str(log)}))
    monkeypatch.setattr(background, "command_matches", lambda _pid, _path: True)
    monkeypatch.setattr(background.os, "kill", lambda _pid, _signal: None)
    ticks = iter((0.0, 11.0))
    monkeypatch.setattr(background.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(background.time, "sleep", lambda _seconds: None)

    assert main(["review", "--stop"]) == 1
    assert "review process 12345 did not stop" in capsys.readouterr().err
    assert record.exists() and log.exists()


def test_background_review_requires_posix(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(background, "os", SimpleNamespace(name="nt"))
    assert main(["review", "--stop"]) == 2
    assert main(["review", str(tmp_path / "missing.mp4"), "--detach"]) == 2
    captured = capsys.readouterr()
    assert captured.err.count("error: background reviews need macOS or Linux") == 2
    assert "fix: run review without --detach and keep it open" in captured.err
    assert not (tmp_path / "data").exists()


def test_stop_keeps_record_if_process_check_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    directory = tmp_path / "data" / "detached"
    directory.mkdir(parents=True)
    record = directory / "review-unchecked.json"
    log = directory / "review-unchecked.log"
    record.write_text(json.dumps({"pid": 12345, "media": [], "versions": False,
                                  "document": {"list": None, "videos": []}, "log": str(log)}))
    log.write_text("old output\n")
    monkeypatch.setattr(background.subprocess, "run", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(FileNotFoundError("ps")))
    assert main(["review", "--stop"]) == 1
    assert "12345" in capsys.readouterr().err
    assert record.exists() and log.exists()


@pytest.mark.e2e
def test_stop_leaves_foreground_review_running(video: Path, review_env: dict[str, str]) -> None:
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "yap_studio", "review", str(video), "--no-open", "--json"],
        cwd=ROOT, env=review_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        lines: queue.Queue[str] = queue.Queue()
        thread = threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True)
        thread.start()
        line = lines.get(timeout=60)
        assert line, process.stderr.read() if process.poll() is not None else ""
        url = json.loads(line)["videos"][0]["url"]
        assert command(review_env, "--stop").returncode == 0
        assert review_payload(url)["media"]["name"] == "sample.mp4"
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.e2e
def test_detach_reports_child_startup_error(tmp_path: Path, review_env: dict[str, str]) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    result = command(review_env, str(bad), "--detach", "--no-open", "--json")
    assert result.returncode == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] and error["message"]
    assert records(review_env) == []


@pytest.mark.e2e
def test_child_exit_removes_record_and_log(video: Path, review_env: dict[str, str]) -> None:
    pid = None
    try:
        result = command(review_env, str(video), "--detach", "--no-open", "--json")
        assert result.returncode == 0, result.stderr
        [record] = records(review_env)
        pid = json.loads(record.read_text())["pid"]
        log = record.with_suffix(".log")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while record.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not record.exists() and not log.exists()
    finally:
        if pid is not None and alive(pid):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.e2e
def test_detached_record_and_log_are_private(video: Path, review_env: dict[str, str]) -> None:
    directory = Path(review_env["YAP_STUDIO_DATA_DIR"]) / "detached"
    directory.mkdir(parents=True, mode=0o755)
    directory.chmod(0o755)
    previous_umask = os.umask(0o022)
    try:
        result = command(review_env, str(video), "--detach", "--no-open", "--json")
        assert result.returncode == 0, result.stderr
        [record] = records(review_env)
        log = record.with_suffix(".log")
        assert directory.stat().st_mode & 0o777 == 0o700
        assert record.stat().st_mode & 0o777 == 0o600
        assert log.stat().st_mode & 0o777 == 0o600
    finally:
        os.umask(previous_umask)
        command(review_env, "--stop")
        cleanup(review_env)


def test_missing_media_and_invalid_stop_options(review_env: dict[str, str]) -> None:
    missing = command(review_env)
    assert missing.returncode == 2
    assert "error: give at least one video or project folder" in missing.stderr
    invalid = command(review_env, "--stop", "--phone")
    assert invalid.returncode == 2
    assert "error:" in invalid.stderr
