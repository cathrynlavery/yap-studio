"""Detached style pages keep their links until saved and closed or stopped."""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name, folder in (("YAP_STUDIO_DATA_DIR", "data"), ("YAP_STUDIO_CONFIG", "config"),
                         ("YAP_STUDIO_CACHE", "cache")):
        monkeypatch.setenv(name, str(tmp_path / folder))


@pytest.fixture
def page(tmp_path):
    from conftest import _stop_detached_page

    from yap_studio import background

    runtime = tmp_path / "renderer/node_modules/hyperframes/dist/hyperframe.runtime.iife.js"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("window.__runtime=1")
    (tmp_path / "sitecustomize.py").write_text('''
import os
from pathlib import Path
from yap_studio.render import hyperframes
from yap_studio.render.hyperframes import RendererInstall
from yap_studio.gallery import command
from yap_studio.review import phone
root = Path(__file__).parent / "renderer"
hyperframes.installed = lambda: (None if os.environ.get("FAKE_GALLERY_NO_RENDERER") else
    RendererInstall("0", root, root / "bin", None, "v22"))
hyperframes.gsap_file = lambda install: None
phone.tailscale_name = lambda: None
phone.lan_address = lambda: None if os.environ.get("FAKE_GALLERY_NO_NETWORK") else "127.0.0.1"
command.IDLE_AFTER_SAVE_S = 0.6
''')
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ROOT / "src")))}
    yield env
    # A page can remove or rewrite its record while unwinding. Recover its identity
    # from argv as well, restricted to this fixture's private data directory.
    listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                             text=True, timeout=5, check=True)
    marker = (r"(?<!\S)--detached-record\s+(" + re.escape(
        str(Path(env["YAP_STUDIO_DATA_DIR"]) / "detached"))
        + r"/style-pick-[0-9a-f]+\.json)(?=\s|$)")
    pages = {}
    for line in listing.stdout.splitlines():
        if match := re.search(marker, line):
            pages[Path(match[1])] = {"pid": int(line.split()[0])}
    for record in records(env):
        saved = background.read_record(record)
        if saved is not None:
            pages.setdefault(record, saved)
    for record, saved in pages.items():
        _stop_detached_page(record, saved)


def detached_logs(data_dir) -> str:
    """Show detached page logs so a start that times out explains itself."""
    folder = Path(data_dir) / "detached"
    return "\n".join(f"--- {log.name}\n{log.read_text(errors='replace')[-4000:]}"
                     for log in sorted(folder.glob("*.log"))) if folder.is_dir() else ""


def command(env, *args, cwd=ROOT):
    # Replacing a page that ignores SIGTERM waits 10 s before the new page starts.
    try:
        return subprocess.run([sys.executable, "-m", "yap_studio", "style", "pick", *args],
                              cwd=cwd, env=env, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"{exc}\n{detached_logs(env['YAP_STUDIO_DATA_DIR'])}") from exc


def records(env):
    return list((Path(env["YAP_STUDIO_DATA_DIR"]) / "detached").glob("style-pick-*.json"))


def request(url, route="/api/state", body=None):
    parsed = urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    try:
        headers = {"Content-Type": "application/json",
                   "X-Gallery-Token": parse_qs(parsed.query)["t"][0]}
        connection.request("POST" if body else "GET", route + "?" + parsed.query,
                           body=json.dumps(body) if body else None, headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read())
    finally:
        connection.close()


def start(env, *args):
    result = command(env, "--detach", "--json", "--no-open", *args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.e2e
def test_page_survives_parent_reuses_replaces_and_stops(page):
    first = start(page)
    assert request(first["url"])["presets"]
    [record] = records(page)
    saved = json.loads(record.read_text())
    assert saved["cwd"] == str(ROOT)
    assert saved["pid"] == first["detached"]["pid"]
    assert record.stat().st_mode & 0o777 == 0o600
    assert record.with_suffix(".log").stat().st_mode & 0o777 == 0o600
    same = start(page)
    assert same["url"] == first["url"]
    assert same["detached"] == {**first["detached"], "reused": True}
    second = start(page, "--style", "preset:clean_minimal", "--phone")
    assert second["detached"]["pid"] != first["detached"]["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(first["detached"]["pid"], 0)
    assert second["phone"]["url"] == second["url"]
    [record] = records(page)
    stopped = command(page, "--stop", "--json")
    assert stopped.returncode == 0, stopped.stderr
    assert json.loads(stopped.stdout) == {"stopped": [
        {"pid": second["detached"]["pid"], "url": second["url"]}]}
    assert not record.exists() and not record.with_suffix(".log").exists()
    with pytest.raises(OSError):
        request(second["url"])
    assert command(page, "--stop").stdout == "No background style page is running.\n"


@pytest.mark.e2e
def test_only_saved_idle_page_exits(page, tmp_path):
    document = start(page, "--workspace", str(tmp_path / "workspace"))
    time.sleep(1)
    style = request(document["url"])["style"]
    request(document["url"], "/api/save", {"style": style, "name": "My style"})
    assert (tmp_path / "workspace/style/my-style.json").is_file()
    # An open tab's preview polls keep the saved page alive.
    for _ in range(4):
        time.sleep(0.25)
        request(document["url"], "/api/preview")
    deadline = time.monotonic() + 5
    while records(page) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert records(page) == []
    with pytest.raises(OSError):
        request(document["url"])


@pytest.mark.parametrize("json_output", [False, True])
def test_startup_error_is_relayed(page, tmp_path, json_output):
    result = command(page, "--detach", "--no-open", "--still", str(tmp_path / "missing.png"),
                     *(["--json"] if json_output else []))
    assert result.returncode == 2
    assert "existing PNG" in result.stderr
    if json_output:
        assert json.loads(result.stderr)["error"]["message"]
    else:
        assert "error:" in result.stderr
    assert records(page) == []


@pytest.mark.parametrize("options", [["--detach"], ["--phone"], ["--host", "127.0.0.1"],
                                    ["--port", "0"], ["--inbox", "."], ["--still", "x.png"],
                                    ["--style", "preset:clean_minimal"], ["--workspace", "."]])
def test_invalid_stop_options(page, options):
    result = command(page, "--stop", *options)
    assert result.returncode == 2
    assert "cannot combine --stop" in result.stderr


@pytest.mark.e2e
def test_simultaneous_starts_reuse_one_page(page):
    with ThreadPoolExecutor(max_workers=2) as pool:
        documents = list(pool.map(lambda _: start(page), range(2)))
    assert documents[0]["url"] == documents[1]["url"]
    assert len(records(page)) == 1


@pytest.mark.parametrize("source", ["inbox", "still"])
def test_relative_paths_are_absolute_in_child(page, tmp_path, source):
    (tmp_path / "inbox").mkdir()
    (tmp_path / "still.png").write_bytes(b"synthetic image")
    value = "inbox" if source == "inbox" else "still.png"
    result = command(page, "--detach", "--no-open", "--json", "--workspace", ".",
                     f"--{source}", value, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    saved = json.loads(records(page)[0].read_text())
    assert saved["cwd"] == str(tmp_path)
    assert saved["command"][saved["command"].index(f"--{source}") + 1] == str(tmp_path / value)
    assert saved["command"][saved["command"].index("--workspace") + 1] == str(tmp_path)
    assert request(json.loads(result.stdout)["url"])["style"]


@pytest.mark.parametrize("failure", ["check", "stop"])
def test_stop_failures_keep_records(tmp_path, monkeypatch, capsys, failure):
    from yap_studio import background, cli

    folder = tmp_path / "data/detached"
    folder.mkdir(parents=True)
    record = folder / "style-pick-test.json"
    record.write_text(json.dumps({"pid": 123, "document": {"url": "http://example.invalid"}}))
    monkeypatch.setattr(background, "stop_record", lambda *_: (True, None if failure == "check"
                                                              else False))
    assert cli.main(["style", "pick", "--stop", "--json"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == f"style_pick_{failure}_failed"
    assert record.exists()


def test_record_process_match_requires_the_record_argument(tmp_path, monkeypatch):
    from yap_studio import background

    record = tmp_path / "style-pick.json"
    monkeypatch.setattr(background.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout=f"python --detached-record other {record}"))
    assert background.command_matches(123, record) is False


def test_style_signal_error_reports_stop_failure(tmp_path, monkeypatch, capsys):
    from yap_studio import background, cli

    folder = tmp_path / "data/detached"
    folder.mkdir(parents=True)
    record = folder / "style-pick-test.json"
    record.write_text(json.dumps({"pid": 123, "document": {"url": "http://example.invalid"}}))

    def cannot_signal(*_args):
        raise PermissionError("cannot signal")

    monkeypatch.setattr(background, "stop_record", cannot_signal)
    assert cli.main(["style", "pick", "--stop", "--json"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "style_pick_stop_failed"
    assert record.exists()


@pytest.mark.e2e
def test_different_cwd_replaces_page_with_the_same_options(page, tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    for folder in (first_dir, second_dir):
        (folder / "inbox").mkdir(parents=True)
    options = ("--detach", "--no-open", "--json")
    first = command(page, *options, cwd=first_dir)
    assert first.returncode == 0, first.stderr
    first_doc = json.loads(first.stdout)
    second = command(page, *options, cwd=second_dir)
    assert second.returncode == 0, second.stderr
    second_doc = json.loads(second.stdout)
    assert second_doc["detached"]["pid"] != first_doc["detached"]["pid"]
    assert "reused" not in second_doc["detached"]
    with pytest.raises(ProcessLookupError):
        os.kill(first_doc["detached"]["pid"], 0)
    assert request(second_doc["url"])["style"]
    assert json.loads(records(page)[0].read_text())["cwd"] == str(second_dir)


@pytest.mark.parametrize("operation", ["--stop", "--detach"])
@pytest.mark.parametrize("saved", [{}, {"pid": None}, {"pid": "123"}, {"pid": True},
                                   {"pid": 2147483647, "document": None}])
def test_malformed_stale_style_record_is_removed(page, saved, operation):
    folder = Path(page["YAP_STUDIO_DATA_DIR"]) / "detached"
    folder.mkdir(parents=True)
    record = folder / "style-pick-malformed.json"
    record.write_text(json.dumps(saved))
    log = record.with_suffix(".log")
    log.write_text("synthetic output")
    result = command(page, operation, "--no-open", "--json")
    assert result.returncode == 0, result.stderr
    assert not record.exists() and not log.exists()


@pytest.mark.e2e
@pytest.mark.parametrize("operation", ["--stop", "--detach"])
@pytest.mark.parametrize("document", [None, {}, []])
def test_malformed_live_style_record_is_stopped(page, document, operation):
    first = start(page)
    [record] = records(page)
    saved = json.loads(record.read_text())
    if document is None:
        saved.pop("document")
    else:
        saved["document"] = document
    record.write_text(json.dumps(saved))
    result = command(page, operation, "--no-open", "--json")
    assert result.returncode == 0, result.stderr
    if operation == "--stop":
        assert json.loads(result.stdout) == {"stopped": [{"pid": saved["pid"], "url": None}]}
    else:
        second = json.loads(result.stdout)
        assert second["detached"]["pid"] != first["detached"]["pid"]
        assert request(second["url"])["style"]
    with pytest.raises(ProcessLookupError):
        os.kill(first["detached"]["pid"], 0)
    assert not record.exists() and not record.with_suffix(".log").exists()


@pytest.mark.parametrize("failure", ["work", "mapping", "both"])
def test_detached_record_is_removed_when_cleanup_raises(tmp_path, monkeypatch, failure):
    from test_gallery_pick import _install

    from yap_studio import cli
    from yap_studio.gallery import command as gallery_command
    from yap_studio.gallery.server import GalleryServer
    from yap_studio.render import hyperframes
    from yap_studio.review import phone

    def cleanup_work():
        if failure in {"work", "both"}:
            raise OSError("synthetic work cleanup failure")

    def cleanup_mapping():
        if failure in {"mapping", "both"}:
            raise OSError("synthetic mapping cleanup failure")

    def serve(_server):
        raise KeyboardInterrupt

    install = _install(tmp_path / "renderer")
    monkeypatch.setattr(hyperframes, "installed", lambda: install)
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _: None)
    monkeypatch.setattr(gallery_command.tempfile, "TemporaryDirectory", lambda **_: SimpleNamespace(
        name=str(tmp_path / "work"), cleanup=cleanup_work))
    monkeypatch.setattr(phone, "tailscale_name", lambda: "synthetic.example")
    monkeypatch.setattr(phone, "tailscale_origin", lambda _: (
        "https://synthetic.example", cleanup_mapping))
    monkeypatch.setattr(GalleryServer, "serve_forever", serve)
    folder = tmp_path / "data/detached"
    folder.mkdir(parents=True)
    record = folder / "style-pick-cleanup.json"
    log = record.with_suffix(".log")
    log.write_text("synthetic output")
    args = ["style", "pick", "--no-open", "--json", "--phone", "--detached-record", str(record)]
    monkeypatch.setattr(sys, "argv", ["yap-studio", *args])
    with pytest.raises(OSError, match="synthetic .* cleanup failure"):
        cli.main(args)
    assert not record.exists() and not log.exists()


@pytest.mark.e2e
@pytest.mark.parametrize("failure", ["still", "style", "renderer", "network"])
def test_failed_replacement_keeps_earlier_style_page(page, tmp_path, failure):
    first = start(page)
    [record] = records(page)
    original = record.read_bytes()
    env = dict(page)
    options = []
    if failure == "still":
        options = ["--still", str(tmp_path / "missing.png")]
    elif failure == "style":
        options = ["--style", "preset:missing"]
    elif failure == "renderer":
        env["FAKE_GALLERY_NO_RENDERER"] = "1"
        options = ["--style", "preset:clean_minimal"]
    else:
        env["FAKE_GALLERY_NO_NETWORK"] = "1"
        options = ["--phone"]
    result = command(env, "--detach", "--json", "--no-open", *options)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stderr)["error"]["message"]
    os.kill(first["detached"]["pid"], 0)
    assert request(first["url"])["style"]
    assert record.read_bytes() == original
    assert records(page) == [record]


@pytest.mark.e2e
def test_replacement_is_ready_before_earlier_style_page_stops(page, monkeypatch, capsys):
    from yap_studio import background, cli

    first = start(page)
    ready = background.ready
    checks = []

    def check_ready(*args, **kwargs):
        saved = ready(*args, **kwargs)
        assert saved is not None
        os.kill(first["detached"]["pid"], 0)
        assert request(first["url"])["style"]
        assert request(saved["document"]["url"])["style"]
        checks.append(saved["pid"])
        return saved

    monkeypatch.setenv("PYTHONPATH", page["PYTHONPATH"])
    monkeypatch.setattr(background, "ready", check_ready)
    assert cli.main(["style", "pick", "--detach", "--json", "--no-open",
                     "--style", "preset:clean_minimal"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert checks == [second["detached"]["pid"]]
    # The earlier page has stopped serving when the command returns; the system may still be
    # reaping its exited process for a moment on a busy machine, so wait briefly for the pid.
    deadline = time.monotonic() + 5
    while True:
        try:
            os.kill(first["detached"]["pid"], 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "the earlier style page process did not exit"
        time.sleep(0.05)
    assert request(second["url"])["style"]


@pytest.mark.e2e
@pytest.mark.parametrize("json_output", [False, True])
def test_failed_fixed_port_replacement_restores_earlier_command(page, tmp_path, json_output):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = str(listener.getsockname()[1])
    first = start(page, "--port", port)
    old = json.loads(records(page)[0].read_text())
    result = command(page, "--detach", "--no-open", "--port", port,
                     "--still", str(tmp_path / "missing.png"),
                     *(["--json"] if json_output else []))
    assert result.returncode == 2, result.stderr
    [record] = records(page)
    saved = json.loads(record.read_text())
    assert saved["command"] == old["command"] and saved["cwd"] == old["cwd"]
    assert saved["pid"] != first["detached"]["pid"]
    assert str(urlsplit(saved["document"]["url"]).port) == port
    assert request(saved["document"]["url"])["style"]
    with pytest.raises(ProcessLookupError):
        os.kill(first["detached"]["pid"], 0)
    if json_output:
        lines = [json.loads(line) for line in result.stderr.splitlines()]
        assert lines[0]["error"]["message"]
        assert lines[1] == {"kept": {"pid": saved["pid"], "url": saved["document"]["url"]}}
    else:
        assert "error:" in result.stderr
        assert re.search(r"kept the earlier style page at (\S+)", result.stderr)[1] == (
            saved["document"]["url"])
