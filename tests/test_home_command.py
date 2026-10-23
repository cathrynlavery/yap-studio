"""CLI parsing, phone links and real detached page lifecycle in isolated folders."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
from urllib.parse import urlsplit

import pytest

from yap_studio import background, cli
from yap_studio.homepage import detach


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "inbox").mkdir()
    return tmp_path


def test_missing_workspace_fix(tmp_path, capsys):
    assert cli.main(["home", "--workspace", str(tmp_path), "--no-open", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "inbox" in error["message"] and error["suggested_action"]


def _cli(*arguments):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yap_studio", *arguments], text=True, capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        from test_style_pick_detach import detached_logs

        logs = detached_logs(os.environ.get("YAP_STUDIO_DATA_DIR", ""))
        raise AssertionError(f"{exc}\n{logs}") from exc
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_detach_live_reuse_stop_scoped(workspace, tmp_path):
    first = None
    other = tmp_path / "other"
    (other / "inbox").mkdir(parents=True)
    try:
        first = _cli("home", "--workspace", str(workspace), "--detach", "--json", "--no-open")
        parsed = urlsplit(first["url"])
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("GET", "/api/state?" + parsed.query)
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["workspace"] == workspace.name
        conn.close()
        again = _cli("home", "--workspace", str(workspace), "--detach", "--json", "--no-open")
        assert again["detached"]["pid"] == first["detached"]["pid"]
        assert again["detached"]["reused"] is True
        other_page = _cli("home", "--workspace", str(other), "--detach", "--json", "--no-open")
        stopped = _cli("home", "--stop", "--workspace", str(workspace), "--json")
        assert [row["pid"] for row in stopped["stopped"]] == [first["detached"]["pid"]]
        assert len(background.records("home")) == 1
        remaining = _cli("home", "--stop", "--json")
        assert remaining["stopped"][0]["pid"] == other_page["detached"]["pid"]
    finally:
        _cli("home", "--stop", "--json")


@pytest.mark.parametrize("fallback", [False, True])
def test_phone_local_url_stays_local(workspace, monkeypatch, capsys, fallback):
    from yap_studio.homepage import server
    from yap_studio.review import phone

    cleaned = []
    opened = []
    monkeypatch.setattr(phone, "tailscale_name", lambda: "home.test")
    monkeypatch.setattr(phone, "lan_address", lambda: "192.168.1.2")

    def tailscale(port):
        if fallback:
            raise OSError("not available")
        return "https://home.test", lambda: cleaned.append(True)

    monkeypatch.setattr(phone, "tailscale_origin", tailscale)
    monkeypatch.setattr(server.HomeServer, "serve_forever", lambda self: None)
    monkeypatch.setattr(detach.webbrowser, "open", opened.append)
    assert cli.main(["home", "--workspace", str(workspace), "--phone", "--json"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["url"].startswith("http://127.0.0.1:")
    assert row["phone"]["via"] == ("wifi" if fallback else "tailscale")
    assert opened == [row["url"]]
    assert bool(cleaned) is not fallback


def test_wait_ack_cli(workspace, capsys):
    from yap_studio.homepage import bridge

    request = bridge.create_request(workspace, "learn", {})
    assert (
        cli.main(["home", "wait", "--workspace", str(workspace), "--timeout", "1", "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["request"]["id"] == request["id"]
    assert (
        cli.main(
            [
                "home",
                "ack",
                request["id"],
                "--workspace",
                str(workspace),
                "--state",
                "done",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["state"] == "done"


@pytest.mark.parametrize("length,expected", [(1999, 0), (2000, 0), (2001, 2)])
def test_ack_reply_length_boundary(workspace, capsys, length, expected):
    from yap_studio.homepage import bridge

    row = bridge.create_request(workspace, "learn", {})
    reply = "é" * length
    assert (
        cli.main(
            [
                "home",
                "ack",
                row["id"],
                "--workspace",
                str(workspace),
                "--state",
                "done",
                "--message",
                reply,
                "--json",
            ]
        )
        == expected
    )
    captured = capsys.readouterr()
    if expected == 0:
        assert json.loads(captured.out)["ack"]["message"] == reply
    else:
        assert json.loads(captured.err)["error"]["message"]
        assert bridge.requests(workspace)[0]["state"] == "pending"
