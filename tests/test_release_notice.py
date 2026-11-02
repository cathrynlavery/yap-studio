"""Quiet release notices use synthetic GitHub replies and isolated stores."""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from yap_studio.install_info import running_version

MESSAGE = 'Yap Studio 2.3.4 is out. Say "update Yap Studio".'
LINK = "https://github.com/cathrynlavery/yap-studio/releases/tag/v2.3.4"
NOTICE = {"text": MESSAGE, "url": LINK}
DAY = 86400


@pytest.fixture
def release(tmp_path, monkeypatch, _owner_folders):
    from yap_studio import release_notice

    monkeypatch.delenv("YAP_STUDIO_NO_UPDATE_NOTICE", raising=False)
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(release_notice.install_info, "running_version", lambda root: "1.0.0")
    now = [1800000000.0]
    monkeypatch.setattr(release_notice.time, "time", lambda: now[0])
    calls = []

    def urlopen(request, *, timeout):
        calls.append((request, timeout, threading.get_ident()))
        return io.BytesIO(json.dumps({"tag_name": "v2.3.4", "html_url": "https://evil.invalid"})
                          .encode())

    monkeypatch.setattr(release_notice, "urlopen", urlopen)
    return SimpleNamespace(module=release_notice, now=now, calls=calls,
                           cache=tmp_path / "cache/release-notice.json")


def seed(release, **values):
    release.cache.parent.mkdir(parents=True, exist_ok=True)
    release.cache.write_text(json.dumps({"checked_at": release.now[0],
                                         "latest_tag": "v2.3.4", **values}))


@pytest.mark.parametrize("installed,expected", [
    ("1.0.0", NOTICE), ("2.3.4", None), ("3.0.0", None), ("0+unknown", None),
])
def test_versions_and_safe_link(release, monkeypatch, installed, expected):
    monkeypatch.setattr(release.module.install_info, "running_version", lambda root: installed)
    assert release.module.notice(refresh=True) == expected
    request, timeout, _ = release.calls[0]
    assert request.full_url == (
        "https://api.github.com/repos/cathrynlavery/yap-studio/releases/latest")
    assert request.get_method() == "GET"
    assert request.data is None
    assert dict(request.header_items()) == {"User-agent": "Yap-Studio"}
    assert 0 < timeout <= 3
    assert json.loads(release.cache.read_text())["latest_tag"] == "v2.3.4"


def test_source_checkout_version_overrides_older_metadata(release, monkeypatch, tmp_path):
    checkout = tmp_path / "source"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "yap-studio"\nversion = "2.3.4"\n')
    monkeypatch.setattr(release.module.install_info, "source_checkout", lambda: checkout)
    monkeypatch.setattr(release.module.install_info, "running_version", running_version)
    monkeypatch.setattr(release.module.install_info, "__version__", "1.0.0")
    assert release.module.notice(refresh=True) is None
    assert len(release.calls) == 1


@pytest.mark.parametrize("tag", [
    "2.3.4", "v2.3", "v2.3.4-beta", "v2.3.4\n", "v2.3.4/../../bad",
    '<img src=x onerror="alert(1)">', "https://evil.invalid", None, 234,
])
def test_reject_untrusted_tags(release, monkeypatch, tag):
    monkeypatch.setattr(release.module, "urlopen", lambda *a, **kw:
                        io.BytesIO(json.dumps({"tag_name": tag}).encode()))
    assert release.module.notice(refresh=True) is None
    assert json.loads(release.cache.read_text())["latest_tag"] is None


def test_cache_and_daily_refresh(release):
    assert release.module.notice(refresh=True) == NOTICE
    release.now[0] += DAY - 1
    assert release.module.notice(refresh=True) == NOTICE
    assert len(release.calls) == 1
    release.now[0] += 1
    assert release.module.notice(refresh=True) == NOTICE
    assert len(release.calls) == 2


@pytest.mark.parametrize("failure", [URLError("offline"), TimeoutError("slow"),
                                     ValueError("bad response"), OSError("read failed")])
def test_failure_is_quiet_and_records_attempt(release, monkeypatch, capsys, failure):
    def fail(*args, **kwargs):
        release.calls.append(True)
        raise failure

    seed(release, checked_at=release.now[0] - DAY, latest_tag=None)
    monkeypatch.setattr(release.module, "urlopen", fail)
    assert release.module.notice(refresh=True) is None
    assert json.loads(release.cache.read_text()) == {
        "checked_at": release.now[0], "latest_tag": None,
    }
    release.now[0] += DAY - 1
    assert release.module.notice(refresh=True) is None
    assert len(release.calls) == 1
    release.now[0] += 1
    assert release.module.notice(refresh=True) is None
    assert len(release.calls) == 2
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("response", ["failure", "malformed"])
def test_failed_refresh_preserves_known_release(release, monkeypatch, response):
    seed(release, checked_at=release.now[0] - 25 * 3600, latest_tag="v9.9.9")

    def urlopen(*args, **kwargs):
        release.calls.append(True)
        if response == "failure":
            raise URLError("offline")
        return io.BytesIO(b'{"tag_name": "v9.9.9/evil"}')

    monkeypatch.setattr(release.module, "urlopen", urlopen)
    expected = {
        "text": 'Yap Studio 9.9.9 is out. Say "update Yap Studio".',
        "url": "https://github.com/cathrynlavery/yap-studio/releases/tag/v9.9.9",
    }
    assert release.module.notice(refresh=True) == expected
    assert release.module.notice() == expected
    assert json.loads(release.cache.read_text()) == {
        "checked_at": release.now[0], "latest_tag": "v9.9.9",
    }
    release.module.notice(refresh=True)
    assert len(release.calls) == 1


@pytest.mark.parametrize("payload", [b"{", b"[]", b'{}', b'{"tag_name": "v2.3.4"', b'\xff'])
def test_bad_response_is_ignored(release, monkeypatch, payload):
    monkeypatch.setattr(release.module, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    assert release.module.notice(refresh=True) is None
    assert json.loads(release.cache.read_text())["latest_tag"] is None


@pytest.mark.parametrize("contents", [
    "{", "[]", "{}", '{"latest_tag":"v2.3.4"}',
    '{"checked_at":1800000000}',
    '{"checked_at":1800000001,"latest_tag":"v2.3.4"}',
    '{"checked_at":NaN,"latest_tag":"v2.3.4"}',
    '{"checked_at":true,"latest_tag":"v2.3.4"}',
    '{"checked_at":1800000000,"latest_tag":"v2.3.4/evil"}',
])
def test_invalid_cache_is_stale_and_replaced(release, contents):
    release.cache.parent.mkdir()
    release.cache.write_text(contents)
    assert release.module.notice() is None
    assert release.calls == []
    assert release.module.notice(refresh=True) == NOTICE
    assert len(release.calls) == 1
    assert json.loads(release.cache.read_text())["latest_tag"] == "v2.3.4"


@pytest.mark.parametrize("switch", ["env", "setting"])
def test_off_hides_cached_notice_and_never_calls(release, monkeypatch, switch):
    seed(release, checked_at=release.now[0] - DAY)
    if switch == "env":
        monkeypatch.setenv("YAP_STUDIO_NO_UPDATE_NOTICE", "1")
    else:
        release.module.setting(False)
    assert release.module.notice() is None
    assert release.module.notice(refresh=True) is None
    release.module.refresh_async()
    assert release.calls == []


def test_setting_command_preserves_config(release, capsys):
    from yap_studio import cli, product_feedback
    from yap_studio.style.library import _config

    product_feedback.setting("never")
    assert cli.main(["update-notice", "off"]) == 0
    assert _config()["update_notice"] is False
    assert _config()["feedback_mode"] == "never"
    assert cli.main(["update-notice", "on"]) == 0
    assert _config()["update_notice"] is True
    capsys.readouterr()


def test_write_failure_is_quiet(release, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(release.module.paths, "atomic_write_text", fail)
    assert release.module.notice(refresh=True) is None
    assert capsys.readouterr() == ("", "")


def test_concurrent_checks_make_one_call(release, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    original = release.module.urlopen

    def slow(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(release.module, "urlopen", slow)
    worker = threading.Thread(target=lambda: release.module.notice(refresh=True))
    worker.start()
    try:
        assert entered.wait(5)
        assert release.module.notice(refresh=True) is None
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(release.calls) == 1
    assert release.module.notice() == NOTICE


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.usefixtures("_no_real_node")
def test_doctor_notice_keeps_exit_code(release, monkeypatch, capsys, edit):
    from yap_studio import cli, setup_checks
    from yap_studio.review import phone

    monkeypatch.setattr(cli, "resolve_tool", lambda name: name)
    monkeypatch.setattr(cli, "check_version", lambda name: None)
    monkeypatch.setattr(cli, "tool_versions", lambda: {"ffmpeg": "6.1", "ffprobe": "6.1"})
    monkeypatch.setattr(cli, "has_filters", lambda names: dict.fromkeys(names, True))
    monkeypatch.setattr(cli, "available_engines", lambda: {})
    monkeypatch.setattr(cli, "resolve_backend", lambda: None)
    monkeypatch.setattr(cli.tools, "find_tool", lambda name: None)
    monkeypatch.setattr(setup_checks, "cutout_reason", lambda: None)
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: None)
    args = ["doctor", *(["--edit"] if edit else [])]
    monkeypatch.setenv("YAP_STUDIO_NO_UPDATE_NOTICE", "1")
    baseline = cli.main(args)
    capsys.readouterr()
    monkeypatch.delenv("YAP_STUDIO_NO_UPDATE_NOTICE")
    assert cli.main(args) == baseline == (2 if edit else 0)
    assert f"update   {MESSAGE} {LINK}" in capsys.readouterr().out.splitlines()
    assert cli.main([*args, "--json"]) == baseline
    assert json.loads(capsys.readouterr().out)["release_notice"] == NOTICE
    assert len(release.calls) == 1


def test_home_snapshot_reads_only_cache(release, tmp_path):
    from yap_studio.homepage.state import HomeState

    (tmp_path / "inbox").mkdir()
    seed(release, checked_at=release.now[0] - DAY)
    assert HomeState(tmp_path).snapshot()["release_notice"] == NOTICE
    assert release.calls == []


def test_home_request_does_not_wait_for_daemon_check(release, tmp_path, monkeypatch):
    from test_home_server import request

    from yap_studio.homepage.server import HomeServer

    (tmp_path / "inbox").mkdir()
    entered, finish = threading.Event(), threading.Event()
    check_threads = []
    original = release.module.urlopen

    def slow(*args, **kwargs):
        check_threads.append(threading.current_thread())
        entered.set()
        assert finish.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(release.module, "urlopen", slow)
    home = HomeServer(tmp_path)
    server_thread = threading.Thread(target=home.serve_forever, daemon=True)
    server_thread.start()
    try:
        code, _, body = request(home)
        assert code == 200
        assert json.loads(body)["release_notice"] is None
        assert entered.wait(5)
        assert all(thread.daemon for thread in check_threads)
        finish.set()
        for thread in check_threads:
            thread.join(5)
        assert json.loads(request(home)[2])["release_notice"] == NOTICE
        assert len(release.calls) == 1
    finally:
        finish.set()
        home.shutdown()
        home.server_close()
        server_thread.join(5)
        for thread in check_threads:
            thread.join(5)
